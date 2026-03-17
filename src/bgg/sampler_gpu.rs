use crate::{
    matrix::PolyMatrix,
    poly::{Poly, PolyParams, dcrt::gpu::detected_gpu_device_ids},
};
use std::sync::Arc;

fn slot_device_ids(slot_parallelism: usize) -> Vec<i32> {
    let device_ids = detected_gpu_device_ids();
    assert!(
        !device_ids.is_empty(),
        "at least one GPU device is required for BGG poly-encoding slot processing"
    );
    assert!(
        slot_parallelism <= device_ids.len(),
        "BGG poly-encoding slot parallelism must be <= params GPU count: requested={}, params_gpus={}",
        slot_parallelism,
        device_ids.len()
    );
    device_ids.into_iter().take(slot_parallelism).collect()
}

fn effective_slot_parallelism_gpu_with_requested(num_slots: usize, requested: usize) -> usize {
    requested.min(num_slots).min(detected_gpu_device_ids().len().max(1))
}

pub(super) fn effective_slot_parallelism_gpu(num_slots: usize) -> usize {
    let requested = crate::env::bgg_poly_encoding_slot_parallelism().max(1);
    effective_slot_parallelism_gpu_with_requested(num_slots, requested)
}

pub(super) struct GpuSamplerSharedByDevice<M: PolyMatrix> {
    pub params: <M::P as Poly>::Params,
    pub base_secret_vec: Arc<M>,
    pub all_public_key_matrix: Arc<M>,
    pub gadget: Arc<M>,
}

pub(super) fn prepare_sampler_shared_by_device<M>(
    params: &<M::P as Poly>::Params,
    base_secret_vec: &M,
    all_public_key_matrix: &M,
    gadget: &M,
    slot_parallelism: usize,
) -> Vec<GpuSamplerSharedByDevice<M>>
where
    M: PolyMatrix,
{
    let base_secret_vec_bytes = Arc::<[u8]>::from(base_secret_vec.to_compact_bytes());
    let all_public_key_matrix_bytes = Arc::<[u8]>::from(all_public_key_matrix.to_compact_bytes());
    let gadget_bytes = Arc::<[u8]>::from(gadget.to_compact_bytes());

    slot_device_ids(slot_parallelism)
        .into_iter()
        .map(|device_id| {
            let local_params = params.params_for_device(device_id);
            GpuSamplerSharedByDevice {
                params: local_params.clone(),
                base_secret_vec: Arc::new(M::from_compact_bytes(
                    &local_params,
                    base_secret_vec_bytes.as_ref(),
                )),
                all_public_key_matrix: Arc::new(M::from_compact_bytes(
                    &local_params,
                    all_public_key_matrix_bytes.as_ref(),
                )),
                gadget: Arc::new(M::from_compact_bytes(&local_params, gadget_bytes.as_ref())),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::effective_slot_parallelism_gpu_with_requested;
    use crate::{__PAIR, __TestState, poly::dcrt::gpu::detected_gpu_device_ids};

    #[sequential_test::sequential]
    #[test]
    fn test_effective_slot_parallelism_gpu_uses_detected_gpu_ids() {
        let detected_count = detected_gpu_device_ids().len().max(1);
        let num_slots = detected_count + 3;
        let actual = effective_slot_parallelism_gpu_with_requested(num_slots, detected_count);
        let expected = detected_count.min(num_slots);
        assert_eq!(actual, expected);
    }
}
