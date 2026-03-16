use super::*;
use crate::{bgg::public_key::BggPublicKey, poly::PolyParams};
use rayon::prelude::*;
use std::{ops::Mul, path::Path, sync::Arc};

use crate::lookup::ggh15::encoding::GGH15PublicLookupSharedState;

pub(super) struct GpuPublicLookupSharedByDevice<M: PolyMatrix> {
    pub device_id: i32,
    pub params: <<M as PolyMatrix>::P as Poly>::Params,
    pub shared: GGH15PublicLookupSharedState<M>,
}

pub(super) fn slot_device_ids<M>(
    params: &<M::P as Poly>::Params,
    slot_parallelism: usize,
) -> Vec<i32>
where
    M: PolyMatrix,
{
    let device_ids = params.device_ids();
    assert!(
        !device_ids.is_empty(),
        "at least one GPU device is required for GGH15 BGG poly-encoding lookup"
    );
    device_ids.into_iter().take(slot_parallelism.max(1)).collect()
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

    slot_device_ids::<M>(params, slot_parallelism)
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
    let shared_by_device =
        prepare_public_lookup_shared_by_device::<M>(params, shared, configured_parallelism);
    let mut output_vector_bytes = Vec::with_capacity(input.num_slots());
    let mut output_plaintext_bytes = Vec::with_capacity(input.num_slots());

    for slot_start in (0..input.num_slots()).step_by(shared_by_device.len()) {
        let chunk_len = (slot_start + shared_by_device.len()).min(input.num_slots()) - slot_start;
        let chunk_shared = &shared_by_device[..chunk_len];
        let c_b0_chunk = &c_b0_compact_bytes_by_slot[slot_start..slot_start + chunk_len];
        let input_vector_chunk = &input.vector_bytes[slot_start..slot_start + chunk_len];
        let x_chunk = &plaintext_compact_bytes_by_slot[slot_start..slot_start + chunk_len];
        let mut chunk_outputs = (0..chunk_len)
            .into_par_iter()
            .map(|offset| {
                let device_shared = &chunk_shared[offset];
                let _ = device_shared.device_id;
                let local_params = &device_shared.params;
                let c_b0 = M::from_compact_bytes(local_params, c_b0_chunk[offset].as_ref());
                let input_vector =
                    M::from_compact_bytes(local_params, input_vector_chunk[offset].as_ref());
                let x = M::P::from_compact_bytes(local_params, x_chunk[offset].as_ref());
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
                drop(c_b0);
                drop(input_vector);
                drop(x);
                (
                    Arc::<[u8]>::from(slot_output.vector.into_compact_bytes()),
                    Arc::<[u8]>::from(slot_output.plaintext.to_compact_bytes()),
                )
            })
            .collect::<Vec<_>>();
        for (vector_bytes, plaintext_bytes) in chunk_outputs.drain(..) {
            output_vector_bytes.push(vector_bytes);
            output_plaintext_bytes.push(plaintext_bytes);
        }
    }

    (output_vector_bytes, output_plaintext_bytes)
}
