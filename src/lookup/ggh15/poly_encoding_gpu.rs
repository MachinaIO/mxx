use super::*;
use crate::{
    bgg::public_key::BggPublicKey,
    lookup::ggh15::{public_lookup_gpu_device_ids, public_lookup_round_robin_device_slot},
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

use crate::lookup::ggh15::encoding::GGH15PublicLookupSharedState;

pub(super) struct GpuPublicLookupSharedByDevice<M: PolyMatrix> {
    pub device_id: i32,
    pub params: <<M as PolyMatrix>::P as Poly>::Params,
    pub shared: GGH15PublicLookupSharedState<M>,
}

struct DecodedPublicLookupSlot<'a, M: PolyMatrix> {
    slot_idx: usize,
    slot_started: Instant,
    decode_ms: f64,
    device_shared: &'a GpuPublicLookupSharedByDevice<M>,
    c_b0: M,
    input_vector: M,
    x: M::P,
}

struct DecodedPublicLookupChunk<'a, M: PolyMatrix> {
    slot_start: usize,
    chunk_len: usize,
    chunk_started: Instant,
    device_ids: Vec<i32>,
    slots: Vec<DecodedPublicLookupSlot<'a, M>>,
}

fn slot_device_ids() -> Vec<i32> {
    public_lookup_gpu_device_ids()
}

fn shared_device_for_slot_idx<M>(
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    slot_idx: usize,
) -> &GpuPublicLookupSharedByDevice<M>
where
    M: PolyMatrix,
{
    let device_slot = public_lookup_round_robin_device_slot(slot_idx, shared_by_device.len());
    &shared_by_device[device_slot]
}

pub(super) fn prepare_public_lookup_shared_by_device<M>(
    params: &<M::P as Poly>::Params,
    shared: &GGH15PublicLookupSharedState<M>,
) -> Vec<GpuPublicLookupSharedByDevice<M>>
where
    M: PolyMatrix,
{
    let gadget_matrix_bytes = Arc::<[u8]>::from(shared.gadget_matrix.as_ref().to_compact_bytes());
    let preimage_gate1_bytes = Arc::<[u8]>::from(shared.preimage_gate1.as_ref().to_compact_bytes());
    let preimage_gate2_identity_bytes =
        Arc::<[u8]>::from(shared.preimage_gate2_identity.as_ref().to_compact_bytes());
    let preimage_gate2_gy_bytes =
        Arc::<[u8]>::from(shared.preimage_gate2_gy.as_ref().to_compact_bytes());
    let preimage_gate2_v_bytes =
        Arc::<[u8]>::from(shared.preimage_gate2_v.as_ref().to_compact_bytes());
    let preimage_gate2_vx_chunk_bytes = shared
        .preimage_gate2_vx_chunks
        .iter()
        .map(|chunk: &Arc<M>| Arc::<[u8]>::from(chunk.as_ref().to_compact_bytes()))
        .collect::<Vec<_>>();
    let u_g_decomposed_bytes = Arc::<[u8]>::from(shared.u_g_decomposed.as_ref().to_compact_bytes());
    let out_pubkey_matrix_bytes = Arc::<[u8]>::from(shared.out_pubkey.matrix.to_compact_bytes());

    slot_device_ids()
        .into_iter()
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
                    preimage_gate1: Arc::new(M::from_compact_bytes(
                        &local_params,
                        preimage_gate1_bytes.as_ref(),
                    )),
                    preimage_gate2_identity: Arc::new(M::from_compact_bytes(
                        &local_params,
                        preimage_gate2_identity_bytes.as_ref(),
                    )),
                    preimage_gate2_gy: Arc::new(M::from_compact_bytes(
                        &local_params,
                        preimage_gate2_gy_bytes.as_ref(),
                    )),
                    preimage_gate2_v: Arc::new(M::from_compact_bytes(
                        &local_params,
                        preimage_gate2_v_bytes.as_ref(),
                    )),
                    preimage_gate2_vx_chunks: preimage_gate2_vx_chunk_bytes
                        .iter()
                        .map(|chunk_bytes: &Arc<[u8]>| {
                            Arc::new(M::from_compact_bytes(&local_params, chunk_bytes.as_ref()))
                        })
                        .collect(),
                    u_g_decomposed: Arc::new(M::from_compact_bytes(
                        &local_params,
                        u_g_decomposed_bytes.as_ref(),
                    )),
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
    let device_ids = (0..chunk_len)
        .map(|offset| shared_device_for_slot_idx(shared_by_device, slot_start + offset).device_id)
        .collect::<Vec<_>>();
    debug!(
        "GGH15 BGG poly-encoding gpu slot chunk started: gate_id={}, lut_id={}, slot_range=[{}, {}), chunk_size={}, device_ids={:?}, completed_before={}/{}",
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
            let device_shared = shared_device_for_slot_idx(shared_by_device, slot_idx);
            let local_params = &device_shared.params;
            let decode_started = Instant::now();
            let c_b0 = M::from_compact_bytes(local_params, c_b0_chunk[offset].as_ref());
            let input_vector =
                M::from_compact_bytes(local_params, input_vector_chunk[offset].as_ref());
            let x = M::P::from_compact_bytes(local_params, x_chunk[offset].as_ref());
            let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
            DecodedPublicLookupSlot {
                slot_idx,
                slot_started,
                decode_ms,
                device_shared,
                c_b0,
                input_vector,
                x,
            }
        })
        .collect::<Vec<_>>();
    DecodedPublicLookupChunk { slot_start, chunk_len, chunk_started, device_ids, slots }
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
    decoded_chunk: DecodedPublicLookupChunk<'_, M>,
) -> Vec<(Arc<[u8]>, Arc<[u8]>)>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let DecodedPublicLookupChunk { slot_start, chunk_len, chunk_started, device_ids, slots } =
        decoded_chunk;
    let chunk_outputs = slots
        .into_par_iter()
        .map(|decoded_slot| {
            let DecodedPublicLookupSlot {
                slot_idx,
                slot_started,
                decode_ms,
                device_shared,
                c_b0,
                input_vector,
                x,
            } = decoded_slot;
            let device_id = device_shared.device_id;
            let local_params = &device_shared.params;
            let apply_started = Instant::now();
            let slot_output = apply_public_lookup_to_slot::<M, HS>(
                local_params,
                plt,
                dir,
                checkpoint_prefix,
                hash_key,
                &c_b0,
                &device_shared.shared,
                &input_vector,
                &x,
                gate_id,
                lut_id,
            );
            let apply_ms = apply_started.elapsed().as_secs_f64() * 1000.0;
            let serialize_started = Instant::now();
            let vector_bytes = Arc::<[u8]>::from(slot_output.vector.into_compact_bytes());
            let plaintext_bytes = Arc::<[u8]>::from(slot_output.plaintext.to_compact_bytes());
            let serialize_ms = serialize_started.elapsed().as_secs_f64() * 1000.0;
            drop(c_b0);
            drop(input_vector);
            drop(x);
            let completed_slot_count = completed_slots.fetch_add(1, Ordering::Relaxed) + 1;
            debug_slot_stage_timings(
                "GGH15 BGG poly-encoding gpu slot",
                gate_id,
                lut_id,
                slot_idx,
                completed_slot_count,
                input_slot_count,
                decode_ms,
                apply_ms,
                serialize_ms,
                slot_started.elapsed().as_secs_f64() * 1000.0,
                Some(device_id),
            );
            (vector_bytes, plaintext_bytes)
        })
        .collect::<Vec<_>>();
    debug!(
        "GGH15 BGG poly-encoding gpu slot chunk finished: gate_id={}, lut_id={}, slot_range=[{}, {}), completed_after={}/{}, elapsed_ms={:.3}, device_ids={:?}",
        gate_id,
        lut_id,
        slot_start,
        slot_start + chunk_len,
        completed_slots.load(Ordering::Relaxed),
        input_slot_count,
        chunk_started.elapsed().as_secs_f64() * 1000.0,
        device_ids
    );
    chunk_outputs
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
    let shared_by_device = prepare_public_lookup_shared_by_device::<M>(params, shared);
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
    use super::slot_device_ids;
    use crate::{
        __PAIR, __TestState, lookup::ggh15::public_lookup_round_robin_device_slot,
        poly::dcrt::gpu::detected_gpu_device_ids,
    };

    #[sequential_test::sequential]
    #[test]
    fn test_ggh15_slot_device_ids_uses_detected_gpu_ids() {
        let detected_gpu_ids = detected_gpu_device_ids();

        if detected_gpu_ids.is_empty() {
            let panic = std::panic::catch_unwind(slot_device_ids)
                .expect_err("without detected GPUs the helper should reject slot processing");
            let panic_msg = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .expect("panic payload should be a string");
            assert!(panic_msg.contains("at least one GPU device is required"));
            return;
        }

        assert_eq!(slot_device_ids(), detected_gpu_ids);
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
