use super::*;
use crate::{
    bgg::public_key::BggPublicKey,
    lookup::ggh15::{pubkey::column_chunk_count, public_lookup_gpu_device_ids},
    poly::PolyParams,
};
use rayon::prelude::*;
use std::{
    ops::Mul,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
use tracing::debug;

use crate::lookup::ggh15::encoding::{
    GGH15PublicLookupSharedState, build_public_lookup_output_chunk,
};

pub(super) struct GpuPublicLookupSharedByDevice<M: PolyMatrix> {
    pub device_id: i32,
    pub params: <<M as PolyMatrix>::P as Poly>::Params,
    pub shared: GGH15PublicLookupSharedState<M>,
}

struct DecodedPublicLookupSlot<'a, M: PolyMatrix> {
    slot_idx: usize,
    slot_started: Instant,
    prepare_ms: f64,
    _marker: std::marker::PhantomData<&'a GpuPublicLookupSharedByDevice<M>>,
    c_b0_bytes: Arc<[u8]>,
    input_vector_bytes: Arc<[u8]>,
    x_bytes: Arc<[u8]>,
}

struct DecodedPublicLookupChunk<'a, M: PolyMatrix> {
    slot_start: usize,
    chunk_len: usize,
    chunk_started: Instant,
    slots: Vec<DecodedPublicLookupSlot<'a, M>>,
}

struct LoadedPublicLookupTask<M: PolyMatrix> {
    device_slot: usize,
    slot_local_idx: usize,
    chunk_idx: usize,
    c_b0: M,
    input_vector: M,
    x: M::P,
}

pub(super) fn effective_gpu_slot_parallelism(requested: usize) -> usize {
    requested.min(public_lookup_gpu_device_ids().len().max(1)).max(1)
}

fn slot_device_ids_for_parallelism(slot_parallelism: usize) -> Vec<i32> {
    let effective_parallelism = effective_gpu_slot_parallelism(slot_parallelism);
    public_lookup_gpu_device_ids().into_iter().take(effective_parallelism).collect()
}

pub(super) fn prepare_public_lookup_shared_by_device<M>(
    params: &<M::P as Poly>::Params,
    shared: &GGH15PublicLookupSharedState<M>,
    slot_parallelism: usize,
) -> Vec<GpuPublicLookupSharedByDevice<M>>
where
    M: PolyMatrix,
{
    let gadget_matrix_bytes = Arc::<[u8]>::from(shared.gadget_matrix.as_ref().to_compact_bytes());
    let out_pubkey_matrix_bytes = Arc::<[u8]>::from(shared.out_pubkey.matrix.to_compact_bytes());

    slot_device_ids_for_parallelism(slot_parallelism)
        .into_par_iter()
        .map(|device_id| {
            let local_params = params.params_for_device(device_id);
            GpuPublicLookupSharedByDevice {
                device_id,
                params: local_params.clone(),
                shared: GGH15PublicLookupSharedState {
                    d: shared.d,
                    m_g: shared.m_g,
                    k_small: shared.k_small,
                    gadget_matrix: Arc::new(M::from_compact_bytes(
                        &local_params,
                        gadget_matrix_bytes.as_ref(),
                    )),
                    preimage_gate1_id_prefix: shared.preimage_gate1_id_prefix.clone(),
                    preimage_gate2_identity_id_prefix: shared
                        .preimage_gate2_identity_id_prefix
                        .clone(),
                    preimage_gate2_gy_id_prefix: shared.preimage_gate2_gy_id_prefix.clone(),
                    preimage_gate2_v_id_prefix: shared.preimage_gate2_v_id_prefix.clone(),
                    preimage_gate2_vx_small_id_prefixes: shared
                        .preimage_gate2_vx_small_id_prefixes
                        .clone(),
                    out_pubkey: BggPublicKey {
                        matrix: M::from_compact_bytes(
                            &local_params,
                            out_pubkey_matrix_bytes.as_ref(),
                        ),
                        reveal_plaintext: shared.out_pubkey.reveal_plaintext,
                    },
                },
            }
        })
        .collect()
}

fn decode_public_lookup_chunk<'a, M>(
    input_slot_count: usize,
    gate_id: GateId,
    lut_id: usize,
    completed_slots: &AtomicUsize,
    slot_start: usize,
    chunk_len: usize,
    shared_by_device: &'a [GpuPublicLookupSharedByDevice<M>],
    c_b0_chunk: &'a [Arc<[u8]>],
    input_vector_chunk: &'a [Arc<[u8]>],
    x_chunk: &'a [Arc<[u8]>],
) -> DecodedPublicLookupChunk<'a, M>
where
    M: PolyMatrix + Send + Sync,
    M::P: Send + Sync,
{
    let chunk_started = Instant::now();
    let device_ids = shared_by_device.iter().map(|entry| entry.device_id).collect::<Vec<_>>();
    debug!(
        "GGH15 BGG poly-encoding gpu slot chunk started: gate_id={}, lut_id={}, slot_range=[{}, {}), chunk_size={}, worker_device_ids={:?}, completed_before={}/{}",
        gate_id,
        lut_id,
        slot_start,
        slot_start + chunk_len,
        chunk_len,
        device_ids,
        completed_slots.load(Ordering::Relaxed),
        input_slot_count
    );
    let slots = (0..chunk_len)
        .into_par_iter()
        .map(|offset| {
            let slot_started = Instant::now();
            let slot_idx = slot_start + offset;
            let prepare_started = Instant::now();
            let c_b0_bytes = Arc::clone(&c_b0_chunk[offset]);
            let input_vector_bytes = Arc::clone(&input_vector_chunk[offset]);
            let x_bytes = Arc::clone(&x_chunk[offset]);
            let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;
            DecodedPublicLookupSlot {
                slot_idx,
                slot_started,
                prepare_ms,
                _marker: std::marker::PhantomData,
                c_b0_bytes,
                input_vector_bytes,
                x_bytes,
            }
        })
        .collect::<Vec<_>>();
    DecodedPublicLookupChunk { slot_start, chunk_len, chunk_started, slots }
}

fn apply_and_serialize_public_lookup_chunk<M, HS>(
    input_slot_count: usize,
    gate_id: GateId,
    lut_id: usize,
    plt: &PublicLut<M::P>,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    completed_slots: &AtomicUsize,
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    decoded_chunk: DecodedPublicLookupChunk<'_, M>,
) -> Vec<(Arc<[u8]>, Arc<[u8]>)>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let DecodedPublicLookupChunk { slot_start, chunk_len, chunk_started, slots } = decoded_chunk;
    let chunk_count = column_chunk_count(shared_by_device[0].shared.m_g);
    let base_params = &shared_by_device[0].params;
    let slot_plaintext_bytes = slots
        .iter()
        .map(|slot| {
            let x = M::P::from_compact_bytes(base_params, slot.x_bytes.as_ref());
            let x_u64 = x.const_coeff_u64();
            let (_, y) = plt
                .get(base_params, x_u64)
                .unwrap_or_else(|| panic!("{:?} not found in LUT for gate {}", x_u64, gate_id));
            Arc::<[u8]>::from(M::P::from_elem_to_constant(base_params, &y).to_compact_bytes())
        })
        .collect::<Vec<_>>();
    let mut chunk_bytes_by_slot = (0..chunk_len)
        .map(|_| (0..chunk_count).map(|_| None).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let tasks = (0..chunk_len)
        .flat_map(|slot_local_idx| {
            (0..chunk_count).map(move |chunk_idx| (slot_local_idx, chunk_idx))
        })
        .collect::<Vec<_>>();
    let wave_width = shared_by_device.len().max(1);
    let mut wave_iter = tasks.chunks(wave_width);
    if let Some(initial_wave) = wave_iter.next() {
        let mut current_loaded_wave =
            load_public_lookup_wave_tasks::<M>(shared_by_device, &slots, initial_wave);
        for wave in wave_iter {
            let (wave_results, next_loaded_wave) = rayon::join(
                || {
                    compute_public_lookup_wave::<M, HS>(
                        plt,
                        dir,
                        checkpoint_prefix,
                        hash_key,
                        gate_id,
                        lut_id,
                        shared_by_device,
                        current_loaded_wave,
                    )
                },
                || load_public_lookup_wave_tasks::<M>(shared_by_device, &slots, wave),
            );
            for (slot_local_idx, chunk_idx, bytes) in wave_results {
                chunk_bytes_by_slot[slot_local_idx][chunk_idx] = Some(bytes);
            }
            current_loaded_wave = next_loaded_wave;
        }
        for (slot_local_idx, chunk_idx, bytes) in compute_public_lookup_wave::<M, HS>(
            plt,
            dir,
            checkpoint_prefix,
            hash_key,
            gate_id,
            lut_id,
            shared_by_device,
            current_loaded_wave,
        ) {
            chunk_bytes_by_slot[slot_local_idx][chunk_idx] = Some(bytes);
        }
    }
    let chunk_outputs = slots
        .into_iter()
        .enumerate()
        .map(|(slot_local_idx, decoded_slot)| {
            let DecodedPublicLookupSlot { slot_idx, slot_started, prepare_ms, .. } = decoded_slot;
            let apply_started = Instant::now();
            let output_chunks = chunk_bytes_by_slot[slot_local_idx]
                .drain(..)
                .enumerate()
                .map(|(chunk_idx, maybe_bytes)| {
                    let chunk_bytes = maybe_bytes.unwrap_or_else(|| {
                        panic!(
                            "missing public-lookup output chunk bytes for slot_local_idx={}, chunk_idx={}",
                            slot_local_idx, chunk_idx
                        )
                    });
                    M::from_compact_bytes(base_params, &chunk_bytes)
                })
                .collect::<Vec<_>>();
            let out_vector = if output_chunks.len() == 1 {
                output_chunks
                    .into_iter()
                    .next()
                    .expect("public-lookup output chunk list must be non-empty")
            } else {
                let mut iter = output_chunks.into_iter();
                let first = iter.next().expect("public-lookup output chunk list must be non-empty");
                first.concat_columns_owned(iter.collect())
            };
            let apply_ms = apply_started.elapsed().as_secs_f64() * 1000.0;
            let serialize_started = Instant::now();
            let vector_bytes = Arc::<[u8]>::from(out_vector.into_compact_bytes());
            let plaintext_bytes = Arc::clone(&slot_plaintext_bytes[slot_local_idx]);
            let serialize_ms = serialize_started.elapsed().as_secs_f64() * 1000.0;
            let completed_slot_count = completed_slots.fetch_add(1, Ordering::Relaxed) + 1;
            debug_slot_stage_timings(
                "GGH15 BGG poly-encoding gpu slot",
                gate_id,
                lut_id,
                slot_idx,
                completed_slot_count,
                input_slot_count,
                prepare_ms,
                apply_ms,
                serialize_ms,
                slot_started.elapsed().as_secs_f64() * 1000.0,
                None,
            );
            (vector_bytes, plaintext_bytes)
        })
        .collect::<Vec<_>>();
    debug!(
        "GGH15 BGG poly-encoding gpu slot chunk finished: gate_id={}, lut_id={}, slot_range=[{}, {}), completed_after={}/{}, elapsed_ms={:.3}",
        gate_id,
        lut_id,
        slot_start,
        slot_start + chunk_len,
        completed_slots.load(Ordering::Relaxed),
        input_slot_count,
        chunk_started.elapsed().as_secs_f64() * 1000.0
    );
    chunk_outputs
}

fn load_public_lookup_wave_tasks<M>(
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    slots: &[DecodedPublicLookupSlot<'_, M>],
    wave: &[(usize, usize)],
) -> Vec<LoadedPublicLookupTask<M>>
where
    M: PolyMatrix + Send + Sync,
    M::P: Send + Sync,
{
    wave.par_iter()
        .enumerate()
        .map(|(device_slot, (slot_local_idx, chunk_idx))| {
            let shared_dev = &shared_by_device[device_slot];
            let slot = &slots[*slot_local_idx];
            LoadedPublicLookupTask {
                device_slot,
                slot_local_idx: *slot_local_idx,
                chunk_idx: *chunk_idx,
                c_b0: M::from_compact_bytes(&shared_dev.params, slot.c_b0_bytes.as_ref()),
                input_vector: M::from_compact_bytes(
                    &shared_dev.params,
                    slot.input_vector_bytes.as_ref(),
                ),
                x: M::P::from_compact_bytes(&shared_dev.params, slot.x_bytes.as_ref()),
            }
        })
        .collect::<Vec<_>>()
}

fn compute_public_lookup_wave<M, HS>(
    plt: &PublicLut<M::P>,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    loaded_wave: Vec<LoadedPublicLookupTask<M>>,
) -> Vec<(usize, usize, Vec<u8>)>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    loaded_wave
        .into_par_iter()
        .map(|loaded| {
            let shared_dev = &shared_by_device[loaded.device_slot];
            let output_chunk = build_public_lookup_output_chunk::<M, HS>(
                &shared_dev.params,
                plt,
                dir,
                checkpoint_prefix,
                hash_key,
                &loaded.c_b0,
                &shared_dev.shared,
                &loaded.input_vector,
                &loaded.x,
                gate_id,
                lut_id,
                loaded.chunk_idx,
            );
            (loaded.slot_local_idx, loaded.chunk_idx, output_chunk.into_compact_bytes())
        })
        .collect::<Vec<_>>()
}

pub(super) fn evaluate_public_lookup_slots_gpu<M, HS>(
    params: &<M::P as Poly>::Params,
    plt: &PublicLut<M::P>,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    input: &BggPolyEncoding<M>,
    plaintext_compact_bytes_by_slot: &[Arc<[u8]>],
    c_b0_compact_bytes_by_slot: &[Arc<[u8]>],
    shared: &GGH15PublicLookupSharedState<M>,
    configured_parallelism: usize,
) -> (Vec<Arc<[u8]>>, Vec<Arc<[u8]>>)
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let evaluate_started = Instant::now();
    debug!(
        "GGH15 BGG poly-encoding gpu slot evaluation started: gate_id={}, lut_id={}, slot_count={}, slot_parallelism={}",
        gate_id,
        lut_id,
        input.num_slots(),
        configured_parallelism
    );
    let prepare_started = Instant::now();
    let shared_by_device =
        prepare_public_lookup_shared_by_device::<M>(params, shared, configured_parallelism);
    let prepared_device_ids =
        shared_by_device.iter().map(|entry| entry.device_id).collect::<Vec<_>>();
    debug!(
        "Prepared GGH15 BGG poly-encoding gpu shared state: gate_id={}, lut_id={}, device_count={}, device_ids={:?}, elapsed_ms={:.3}",
        gate_id,
        lut_id,
        shared_by_device.len(),
        prepared_device_ids,
        prepare_started.elapsed().as_secs_f64() * 1000.0
    );
    let slot_count = input.num_slots();
    let mut output_vector_bytes = Vec::with_capacity(slot_count);
    let mut output_plaintext_bytes = Vec::with_capacity(slot_count);
    let completed_slots = AtomicUsize::new(0);
    let chunk_width = configured_parallelism;
    let initial_chunk_len = chunk_width.min(slot_count);
    let mut current_decoded_chunk = (initial_chunk_len > 0).then(|| {
        decode_public_lookup_chunk(
            slot_count,
            gate_id,
            lut_id,
            &completed_slots,
            0,
            initial_chunk_len,
            &shared_by_device,
            &c_b0_compact_bytes_by_slot[..initial_chunk_len],
            &input.vector_bytes[..initial_chunk_len],
            &plaintext_compact_bytes_by_slot[..initial_chunk_len],
        )
    });
    let mut next_slot_start = initial_chunk_len;

    while let Some(decoded_chunk) = current_decoded_chunk.take() {
        if next_slot_start < slot_count {
            let next_chunk_len = (next_slot_start + chunk_width).min(slot_count) - next_slot_start;
            let (chunk_outputs, decoded_next_chunk) = rayon::join(
                || {
                    apply_and_serialize_public_lookup_chunk::<M, HS>(
                        slot_count,
                        gate_id,
                        lut_id,
                        plt,
                        dir,
                        checkpoint_prefix,
                        hash_key,
                        &completed_slots,
                        &shared_by_device,
                        decoded_chunk,
                    )
                },
                || {
                    decode_public_lookup_chunk(
                        slot_count,
                        gate_id,
                        lut_id,
                        &completed_slots,
                        next_slot_start,
                        next_chunk_len,
                        &shared_by_device,
                        &c_b0_compact_bytes_by_slot
                            [next_slot_start..next_slot_start + next_chunk_len],
                        &input.vector_bytes[next_slot_start..next_slot_start + next_chunk_len],
                        &plaintext_compact_bytes_by_slot
                            [next_slot_start..next_slot_start + next_chunk_len],
                    )
                },
            );
            let (chunk_vector_bytes, chunk_plaintext_bytes): (Vec<_>, Vec<_>) =
                chunk_outputs.into_iter().unzip();
            output_vector_bytes.extend(chunk_vector_bytes);
            output_plaintext_bytes.extend(chunk_plaintext_bytes);
            current_decoded_chunk = Some(decoded_next_chunk);
            next_slot_start += next_chunk_len;
        } else {
            let chunk_outputs = apply_and_serialize_public_lookup_chunk::<M, HS>(
                slot_count,
                gate_id,
                lut_id,
                plt,
                dir,
                checkpoint_prefix,
                hash_key,
                &completed_slots,
                &shared_by_device,
                decoded_chunk,
            );
            let (chunk_vector_bytes, chunk_plaintext_bytes): (Vec<_>, Vec<_>) =
                chunk_outputs.into_iter().unzip();
            output_vector_bytes.extend(chunk_vector_bytes);
            output_plaintext_bytes.extend(chunk_plaintext_bytes);
        }
    }

    debug!(
        "GGH15 BGG poly-encoding gpu slot evaluation finished: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
        gate_id,
        lut_id,
        slot_count,
        evaluate_started.elapsed().as_secs_f64() * 1000.0
    );
    (output_vector_bytes, output_plaintext_bytes)
}

#[cfg(test)]
mod tests {
    use super::{effective_gpu_slot_parallelism, slot_device_ids_for_parallelism};
    use crate::{
        __PAIR, __TestState, lookup::ggh15::public_lookup_round_robin_device_slot,
        poly::dcrt::gpu::detected_gpu_device_ids,
    };

    #[sequential_test::sequential]
    #[test]
    fn test_ggh15_slot_device_ids_uses_detected_gpu_ids() {
        let detected_gpu_ids = detected_gpu_device_ids();

        if detected_gpu_ids.is_empty() {
            let panic = std::panic::catch_unwind(|| slot_device_ids_for_parallelism(1))
                .expect_err("without detected GPUs the helper should reject slot processing");
            let panic_msg = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .expect("panic payload should be a string");
            assert!(panic_msg.contains("at least one GPU device is required"));
            return;
        }

        let slot_parallelism = detected_gpu_ids.len().min(2);
        assert_eq!(
            slot_device_ids_for_parallelism(slot_parallelism),
            detected_gpu_ids[..slot_parallelism].to_vec()
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ggh15_effective_gpu_slot_parallelism_clamps_to_detected_gpus() {
        let detected_count = detected_gpu_device_ids().len().max(1);
        assert_eq!(effective_gpu_slot_parallelism(detected_count + 5), detected_count);
    }

    #[test]
    fn test_ggh15_round_robin_slot_device_slot_balances_logical_slots() {
        let device_count = 3;
        let logical_slots = 11;
        let mut counts = vec![0usize; device_count];
        for slot_idx in 0..logical_slots {
            counts[public_lookup_round_robin_device_slot(slot_idx, device_count)] += 1;
        }

        let min_count = *counts.iter().min().expect("counts must be non-empty");
        let max_count = *counts.iter().max().expect("counts must be non-empty");
        assert!(max_count - min_count <= 1, "counts={counts:?}");
    }
}
