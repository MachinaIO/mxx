use super::*;
use crate::{
    bgg::public_key::BggPublicKey,
    poly::{PolyParams, dcrt::gpu::detected_gpu_device_ids},
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

pub(super) fn slot_device_ids(slot_parallelism: usize) -> Vec<i32> {
    let device_ids = detected_gpu_device_ids();
    assert!(
        !device_ids.is_empty(),
        "at least one GPU device is required for GGH15 BGG poly-encoding lookup"
    );
    assert!(
        slot_parallelism <= device_ids.len(),
        "GGH15 BGG poly-encoding slot parallelism must be <= detected GPU count: requested={}, devices={}",
        slot_parallelism,
        device_ids.len()
    );
    device_ids.into_iter().take(slot_parallelism).collect()
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

    slot_device_ids(slot_parallelism)
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
    let mut output_vector_bytes = Vec::with_capacity(input.num_slots());
    let mut output_plaintext_bytes = Vec::with_capacity(input.num_slots());
    let completed_slots = AtomicUsize::new(0);

    for slot_start in (0..input.num_slots()).step_by(shared_by_device.len()) {
        let chunk_len = (slot_start + shared_by_device.len()).min(input.num_slots()) - slot_start;
        let chunk_started = Instant::now();
        let chunk_shared = &shared_by_device[..chunk_len];
        let chunk_device_ids = chunk_shared.iter().map(|entry| entry.device_id).collect::<Vec<_>>();
        debug!(
            "GGH15 BGG poly-encoding gpu slot chunk started: gate_id={}, lut_id={}, slot_range=[{}, {}), chunk_size={}, device_ids={:?}, completed_before={}/{}",
            gate_id,
            lut_id,
            slot_start,
            slot_start + chunk_len,
            chunk_len,
            chunk_device_ids,
            completed_slots.load(Ordering::Relaxed),
            input.num_slots()
        );
        let c_b0_chunk = &c_b0_compact_bytes_by_slot[slot_start..slot_start + chunk_len];
        let input_vector_chunk = &input.vector_bytes[slot_start..slot_start + chunk_len];
        let x_chunk = &plaintext_compact_bytes_by_slot[slot_start..slot_start + chunk_len];
        let mut chunk_outputs = (0..chunk_len)
            .into_par_iter()
            .map(|offset| {
                let slot_idx = slot_start + offset;
                let slot_started = Instant::now();
                let device_shared = &chunk_shared[offset];
                let device_id = device_shared.device_id;
                let local_params = &device_shared.params;
                let decode_started = Instant::now();
                let c_b0 = M::from_compact_bytes(local_params, c_b0_chunk[offset].as_ref());
                let input_vector =
                    M::from_compact_bytes(local_params, input_vector_chunk[offset].as_ref());
                let x = M::P::from_compact_bytes(local_params, x_chunk[offset].as_ref());
                let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
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
                    input.num_slots(),
                    decode_ms,
                    apply_ms,
                    serialize_ms,
                    slot_started.elapsed().as_secs_f64() * 1000.0,
                    Some(device_id),
                );
                (vector_bytes, plaintext_bytes)
            })
            .collect::<Vec<_>>();
        for (vector_bytes, plaintext_bytes) in chunk_outputs.drain(..) {
            output_vector_bytes.push(vector_bytes);
            output_plaintext_bytes.push(plaintext_bytes);
        }
        debug!(
            "GGH15 BGG poly-encoding gpu slot chunk finished: gate_id={}, lut_id={}, slot_range=[{}, {}), completed_after={}/{}, elapsed_ms={:.3}",
            gate_id,
            lut_id,
            slot_start,
            slot_start + chunk_len,
            completed_slots.load(Ordering::Relaxed),
            input.num_slots(),
            chunk_started.elapsed().as_secs_f64() * 1000.0
        );
    }

    debug!(
        "GGH15 BGG poly-encoding gpu slot evaluation finished: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
        gate_id,
        lut_id,
        input.num_slots(),
        evaluate_started.elapsed().as_secs_f64() * 1000.0
    );
    (output_vector_bytes, output_plaintext_bytes)
}

#[cfg(test)]
mod tests {
    use super::slot_device_ids;
    use crate::{__PAIR, __TestState, poly::dcrt::gpu::detected_gpu_device_ids};

    #[sequential_test::sequential]
    #[test]
    fn test_ggh15_slot_device_ids_uses_detected_gpu_ids() {
        let detected_gpu_ids = detected_gpu_device_ids();

        if detected_gpu_ids.is_empty() {
            let panic = std::panic::catch_unwind(|| slot_device_ids(1))
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
            slot_device_ids(slot_parallelism),
            detected_gpu_ids[..slot_parallelism].to_vec()
        );
    }
}
