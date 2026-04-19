use super::*;
use crate::{
    lookup::ggh15::{pubkey::column_chunk_count, public_lookup_gpu_device_ids},
    poly::PolyParams,
};
use rayon::prelude::*;
use std::{ops::Mul, path::Path, sync::Arc, time::Instant};
use tracing::debug;

use crate::lookup::ggh15::encoding::{
    GGH15PublicLookupSharedState, sample_public_lookup_u_g_chunk,
};

pub(super) struct GpuPublicLookupSharedByDevice<M: PolyMatrix> {
    pub device_id: i32,
    pub params: <<M as PolyMatrix>::P as Poly>::Params,
}

struct LoadedPublicLookupSlot<M: PolyMatrix> {
    slot_idx: usize,
    slot_started: Instant,
    decode_ms: f64,
    c_b0_by_device: Vec<M>,
    input_vector_by_device: Vec<M>,
    x: M::P,
}

struct LoadedPublicLookupStageChunk<M: PolyMatrix> {
    device_slot: usize,
    chunk_idx: usize,
    rhs_chunk: M,
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
    _shared: &GGH15PublicLookupSharedState<M>,
    slot_parallelism: usize,
) -> Vec<GpuPublicLookupSharedByDevice<M>>
where
    M: PolyMatrix,
{
    slot_device_ids_for_parallelism(slot_parallelism)
        .into_par_iter()
        .map(|device_id| {
            let local_params = params.params_for_device(device_id);
            GpuPublicLookupSharedByDevice { device_id, params: local_params.clone() }
        })
        .collect()
}

fn concat_public_lookup_chunk_bytes<M>(
    params: &<M::P as Poly>::Params,
    chunk_bytes: Vec<Vec<u8>>,
) -> M
where
    M: PolyMatrix,
{
    if chunk_bytes.len() == 1 {
        return M::from_compact_bytes(params, &chunk_bytes[0]);
    }
    let mut chunk_iter = chunk_bytes.iter().map(|bytes| M::from_compact_bytes(params, bytes));
    let first = chunk_iter.next().expect("public-lookup chunk byte list must be non-empty");
    first.concat_columns_owned(chunk_iter.collect())
}

fn redistribute_public_lookup_stage_matrix<M>(
    matrix: &M,
    params_by_target: &[&<M::P as Poly>::Params],
) -> Vec<M>
where
    M: PolyMatrix,
{
    let matrix_bytes = matrix.to_compact_bytes();
    params_by_target.iter().map(|params| M::from_compact_bytes(params, &matrix_bytes)).collect()
}

fn collect_pipelined_chunk_bytes<Loaded, FL, FC>(
    chunk_count: usize,
    wave_width: usize,
    load_wave: FL,
    compute_wave: FC,
) -> Vec<Vec<u8>>
where
    Loaded: Send,
    FL: Fn(usize, usize) -> Vec<Loaded> + Sync,
    FC: Fn(Vec<Loaded>) -> Vec<(usize, Vec<u8>)> + Sync,
{
    if chunk_count == 0 {
        return Vec::new();
    }
    let wave_width = wave_width.max(1).min(chunk_count);
    let mut chunk_bytes = (0..chunk_count).map(|_| None).collect::<Vec<_>>();
    let mut next_chunk_idx = 0usize;
    let mut current_wave = load_wave(next_chunk_idx, wave_width);
    next_chunk_idx += current_wave.len();

    while !current_wave.is_empty() {
        if next_chunk_idx < chunk_count {
            let next_wave_width = (chunk_count - next_chunk_idx).min(wave_width);
            let (wave_results, next_wave) = rayon::join(
                || compute_wave(current_wave),
                || load_wave(next_chunk_idx, next_wave_width),
            );
            for (chunk_idx, bytes) in wave_results {
                chunk_bytes[chunk_idx] = Some(bytes);
            }
            current_wave = next_wave;
            next_chunk_idx += next_wave_width;
        } else {
            for (chunk_idx, bytes) in compute_wave(current_wave) {
                chunk_bytes[chunk_idx] = Some(bytes);
            }
            break;
        }
    }

    chunk_bytes
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, maybe_bytes)| {
            maybe_bytes.unwrap_or_else(|| {
                panic!("missing public-lookup chunk bytes for chunk {chunk_idx}")
            })
        })
        .collect()
}

fn load_public_lookup_slot<M>(
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    slot_idx: usize,
    c_b0_bytes: &Arc<[u8]>,
    input_vector_bytes: &Arc<[u8]>,
    x_bytes: &Arc<[u8]>,
) -> LoadedPublicLookupSlot<M>
where
    M: PolyMatrix + Send + Sync,
    M::P: Send + Sync,
{
    let slot_started = Instant::now();
    let decode_started = Instant::now();
    let x = M::P::from_compact_bytes(&shared_by_device[0].params, x_bytes.as_ref());
    let decoded_by_device = shared_by_device
        .par_iter()
        .map(|shared| {
            (
                M::from_compact_bytes(&shared.params, c_b0_bytes.as_ref()),
                M::from_compact_bytes(&shared.params, input_vector_bytes.as_ref()),
            )
        })
        .collect::<Vec<_>>();
    let (c_b0_by_device, input_vector_by_device): (Vec<_>, Vec<_>) =
        decoded_by_device.into_iter().unzip();
    LoadedPublicLookupSlot {
        slot_idx,
        slot_started,
        decode_ms: decode_started.elapsed().as_secs_f64() * 1000.0,
        c_b0_by_device,
        input_vector_by_device,
        x,
    }
}

fn load_checkpoint_column_wave<M>(
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    dir: &Path,
    id_prefix: &str,
    total_cols: usize,
    label: &str,
    wave_start: usize,
    wave_width: usize,
) -> Vec<LoadedPublicLookupStageChunk<M>>
where
    M: PolyMatrix + Send + Sync,
{
    (wave_start..wave_start + wave_width)
        .into_par_iter()
        .enumerate()
        .map(|(device_slot, chunk_idx)| {
            let shared = &shared_by_device[device_slot];
            let rhs_chunk = read_matrix_column_chunk(
                &shared.params,
                dir,
                id_prefix,
                total_cols,
                chunk_idx,
                label,
            );
            LoadedPublicLookupStageChunk { device_slot, chunk_idx, rhs_chunk }
        })
        .collect()
}

fn load_serialized_rhs_wave<M>(
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    rhs_chunk_bytes: &[Arc<[u8]>],
    wave_start: usize,
    wave_width: usize,
) -> Vec<LoadedPublicLookupStageChunk<M>>
where
    M: PolyMatrix + Send + Sync,
{
    (wave_start..wave_start + wave_width)
        .into_par_iter()
        .enumerate()
        .map(|(device_slot, chunk_idx)| {
            let shared = &shared_by_device[device_slot];
            let rhs_chunk =
                M::from_compact_bytes(&shared.params, rhs_chunk_bytes[chunk_idx].as_ref());
            LoadedPublicLookupStageChunk { device_slot, chunk_idx, rhs_chunk }
        })
        .collect()
}

fn compute_left_mul_wave<M>(
    lhs_by_device: &[M],
    loaded_wave: Vec<LoadedPublicLookupStageChunk<M>>,
) -> Vec<(usize, Vec<u8>)>
where
    M: PolyMatrix + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    loaded_wave
        .into_par_iter()
        .map(|loaded| {
            (
                loaded.chunk_idx,
                (&lhs_by_device[loaded.device_slot] * &loaded.rhs_chunk).into_compact_bytes(),
            )
        })
        .collect()
}

fn build_left_mul_checkpoint_stage_matrix<M>(
    params: &<M::P as Poly>::Params,
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    lhs_by_device: &[M],
    dir: &Path,
    id_prefix: &str,
    total_cols: usize,
    label: &str,
) -> M
where
    M: PolyMatrix + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let chunk_count = column_chunk_count(total_cols);
    let chunk_bytes = collect_pipelined_chunk_bytes(
        chunk_count,
        shared_by_device.len().max(1),
        |wave_start, wave_width| {
            load_checkpoint_column_wave(
                shared_by_device,
                dir,
                id_prefix,
                total_cols,
                label,
                wave_start,
                wave_width,
            )
        },
        |loaded_wave| compute_left_mul_wave(lhs_by_device, loaded_wave),
    );
    concat_public_lookup_chunk_bytes(params, chunk_bytes)
}

fn build_left_mul_checkpoint_stage_matrix_with_reloads<M>(
    params: &<M::P as Poly>::Params,
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    lhs_by_device: &[M],
    dir: &Path,
    id_prefix: &str,
    total_cols: usize,
    label: &str,
) -> (M, Vec<M>)
where
    M: PolyMatrix + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let base_matrix = build_left_mul_checkpoint_stage_matrix(
        params,
        shared_by_device,
        lhs_by_device,
        dir,
        id_prefix,
        total_cols,
        label,
    );
    let reload_params = shared_by_device.iter().map(|shared| &shared.params).collect::<Vec<_>>();
    let reloaded = redistribute_public_lookup_stage_matrix(&base_matrix, &reload_params);
    (base_matrix, reloaded)
}

fn build_serialized_rhs_stage_matrix<M>(
    params: &<M::P as Poly>::Params,
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    lhs_by_device: &[M],
    rhs_chunk_bytes: &[Arc<[u8]>],
) -> M
where
    M: PolyMatrix + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let chunk_count = rhs_chunk_bytes.len();
    let chunk_bytes = collect_pipelined_chunk_bytes(
        chunk_count,
        shared_by_device.len().max(1),
        |wave_start, wave_width| {
            load_serialized_rhs_wave(shared_by_device, rhs_chunk_bytes, wave_start, wave_width)
        },
        |loaded_wave| compute_left_mul_wave(lhs_by_device, loaded_wave),
    );
    concat_public_lookup_chunk_bytes(params, chunk_bytes)
}

fn build_serialized_rhs_stage_matrix_with_reloads<M>(
    params: &<M::P as Poly>::Params,
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    lhs_by_device: &[M],
    rhs_chunk_bytes: &[Arc<[u8]>],
) -> (M, Vec<M>)
where
    M: PolyMatrix + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let base_matrix =
        build_serialized_rhs_stage_matrix(params, shared_by_device, lhs_by_device, rhs_chunk_bytes);
    let reload_params = shared_by_device.iter().map(|shared| &shared.params).collect::<Vec<_>>();
    let reloaded = redistribute_public_lookup_stage_matrix(&base_matrix, &reload_params);
    (base_matrix, reloaded)
}

fn output_slot_for_params_gpu<M, HS>(
    params: &<M::P as Poly>::Params,
    plt: &PublicLut<M::P>,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    shared: &GGH15PublicLookupSharedState<M>,
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    c_b0_bytes: &Arc<[u8]>,
    input_vector_bytes: &Arc<[u8]>,
    x_bytes: &Arc<[u8]>,
    slot_idx: usize,
    completed_slots: usize,
    total_slots: usize,
) -> (Arc<[u8]>, Arc<[u8]>)
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let loaded = load_public_lookup_slot(
        shared_by_device,
        slot_idx,
        c_b0_bytes,
        input_vector_bytes,
        x_bytes,
    );
    let x_u64 = loaded.x.const_coeff_u64();
    let (k, y) = plt
        .get(params, x_u64)
        .unwrap_or_else(|| panic!("{:?} not found in LUT for gate {}", x_u64, gate_id));
    let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
    let y_poly = M::P::from_elem_to_constant(params, &y);
    let lut_aux_row_id = format!("{checkpoint_prefix}_lut_aux_{lut_id}_idx{k}");
    let output_chunk_count = column_chunk_count(shared.m_g);
    let gate1_total_cols = trapdoor_public_column_count::<M>(params, shared.d);

    let apply_started = Instant::now();

    // Stage inputs that are not checkpoints are materialized once on the base params, then each
    // later stage loads only the chunk it needs while the previous wave computes.
    let gy = shared.gadget_matrix.as_ref().clone() * y_poly.clone();
    let gy_chunk_bytes = (0..output_chunk_count)
        .map(|chunk_idx| {
            let (col_start, col_len) = column_chunk_bounds(shared.m_g, chunk_idx);
            Arc::<[u8]>::from(
                gy.slice_columns(col_start, col_start + col_len).decompose().into_compact_bytes(),
            )
        })
        .collect::<Vec<_>>();
    let scalar_by_digit = small_decomposed_scalar_digits::<M>(params, &loaded.x, shared.k_small);
    let v_idx_id = format!("ggh15_lut_v_idx_{}_{}", lut_id, k_usize);
    let v_idx_chunk_bytes = (0..output_chunk_count)
        .map(|chunk_idx| {
            let (col_start, col_len) = column_chunk_bounds(shared.m_g, chunk_idx);
            Arc::<[u8]>::from(
                HS::new()
                    .sample_hash_decomposed_columns(
                        params,
                        hash_key,
                        v_idx_id.clone(),
                        shared.d,
                        shared.m_g,
                        col_start,
                        col_len,
                        DistType::FinRingDist,
                    )
                    .into_compact_bytes(),
            )
        })
        .collect::<Vec<_>>();
    let v_idx_chunks = v_idx_chunk_bytes
        .iter()
        .map(|bytes| M::from_compact_bytes(params, bytes.as_ref()))
        .collect::<Vec<_>>();
    let vx_chunk_bytes_by_small = (0..shared.k_small)
        .map(|small_chunk_idx| {
            v_idx_chunks
                .iter()
                .map(|v_idx_chunk| {
                    Arc::<[u8]>::from(
                        build_small_decomposed_scalar_mul_chunk::<M>(
                            params,
                            v_idx_chunk,
                            &scalar_by_digit,
                            small_chunk_idx,
                        )
                        .into_compact_bytes(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let u_g_id = format!("ggh15_lut_u_g_matrix_{}", gate_id);
    let u_g_chunk_bytes = (0..output_chunk_count)
        .map(|chunk_idx| {
            Arc::<[u8]>::from(
                sample_public_lookup_u_g_chunk::<M, HS>(
                    params, hash_key, &u_g_id, shared.d, shared.m_g, chunk_idx,
                )
                .into_compact_bytes(),
            )
        })
        .collect::<Vec<_>>();

    // `public_lut` now follows lhs-first stage algebra throughout:
    //
    // 1. Build each `lhs * checkpoint` stage matrix across devices and gather it on the base
    //    device/host boundary.
    // 2. Redistribute that full stage matrix to every device.
    // 3. Multiply the redistributed lhs-stage matrix by the next rhs chunk family.
    //
    // The only overlap kept here is "load/store next chunk wave" versus "compute current chunk
    // wave". There is no independent compute-vs-compute overlap.
    let mut c_const = build_left_mul_checkpoint_stage_matrix(
        params,
        shared_by_device,
        &loaded.c_b0_by_device,
        dir,
        &shared.preimage_gate2_identity_id_prefix,
        shared.m_g,
        "preimage_gate2_identity",
    );

    let (_, c_b0_preimage_gy_by_device) = build_left_mul_checkpoint_stage_matrix_with_reloads(
        params,
        shared_by_device,
        &loaded.c_b0_by_device,
        dir,
        &shared.preimage_gate2_gy_id_prefix,
        shared.m_g,
        "preimage_gate2_gy",
    );
    c_const.add_in_place(&build_serialized_rhs_stage_matrix(
        params,
        shared_by_device,
        &c_b0_preimage_gy_by_device,
        &gy_chunk_bytes,
    ));

    let (_, c_b0_preimage_v_by_device) = build_left_mul_checkpoint_stage_matrix_with_reloads(
        params,
        shared_by_device,
        &loaded.c_b0_by_device,
        dir,
        &shared.preimage_gate2_v_id_prefix,
        shared.m_g,
        "preimage_gate2_v",
    );
    c_const.add_in_place(&build_serialized_rhs_stage_matrix(
        params,
        shared_by_device,
        &c_b0_preimage_v_by_device,
        &v_idx_chunk_bytes,
    ));

    for (small_chunk_idx, vx_chunk_bytes) in vx_chunk_bytes_by_small.iter().enumerate() {
        let (_, c_b0_preimage_vx_by_device) = build_left_mul_checkpoint_stage_matrix_with_reloads(
            params,
            shared_by_device,
            &loaded.c_b0_by_device,
            dir,
            &shared.preimage_gate2_vx_small_id_prefixes[small_chunk_idx],
            shared.m_g,
            "preimage_gate2_vx_small",
        );
        c_const.add_in_place(&build_serialized_rhs_stage_matrix(
            params,
            shared_by_device,
            &c_b0_preimage_vx_by_device,
            vx_chunk_bytes,
        ));
    }

    let (_, c_b0_preimage_gate1_by_device) = build_left_mul_checkpoint_stage_matrix_with_reloads(
        params,
        shared_by_device,
        &loaded.c_b0_by_device,
        dir,
        &shared.preimage_gate1_id_prefix,
        gate1_total_cols,
        "preimage_gate1",
    );
    c_const = c_const -
        build_left_mul_checkpoint_stage_matrix(
            params,
            shared_by_device,
            &c_b0_preimage_gate1_by_device,
            dir,
            &lut_aux_row_id,
            shared.m_g,
            "preimage_lut",
        );

    let (input_times_u_g, input_times_u_g_by_device) =
        build_serialized_rhs_stage_matrix_with_reloads(
            params,
            shared_by_device,
            &loaded.input_vector_by_device,
            &u_g_chunk_bytes,
        );
    drop(input_times_u_g);
    c_const.add_in_place(&build_serialized_rhs_stage_matrix(
        params,
        shared_by_device,
        &input_times_u_g_by_device,
        &v_idx_chunk_bytes,
    ));

    let apply_ms = apply_started.elapsed().as_secs_f64() * 1000.0;
    let serialize_started = Instant::now();
    let vector_bytes = Arc::<[u8]>::from(c_const.into_compact_bytes());
    let plaintext_bytes = Arc::<[u8]>::from(y_poly.to_compact_bytes());
    let serialize_ms = serialize_started.elapsed().as_secs_f64() * 1000.0;
    debug_slot_stage_timings(
        "GGH15 BGG poly-encoding gpu slot",
        gate_id,
        lut_id,
        loaded.slot_idx,
        completed_slots,
        total_slots,
        loaded.decode_ms,
        apply_ms,
        serialize_ms,
        loaded.slot_started.elapsed().as_secs_f64() * 1000.0,
        None,
    );
    (vector_bytes, plaintext_bytes)
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

    for slot_idx in 0..slot_count {
        let (vector_bytes, plaintext_bytes) = output_slot_for_params_gpu::<M, HS>(
            params,
            plt,
            dir,
            checkpoint_prefix,
            hash_key,
            gate_id,
            lut_id,
            shared,
            &shared_by_device,
            &c_b0_compact_bytes_by_slot[slot_idx],
            &input.vector_bytes[slot_idx],
            &plaintext_compact_bytes_by_slot[slot_idx],
            slot_idx,
            slot_idx + 1,
            slot_count,
        );
        output_vector_bytes.push(vector_bytes);
        output_plaintext_bytes.push(plaintext_bytes);
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
