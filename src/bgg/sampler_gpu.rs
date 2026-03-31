use crate::{
    matrix::PolyMatrix,
    poly::{Poly, PolyParams, dcrt::gpu::detected_gpu_device_ids},
    sampler::{DistType, PolyUniformSampler},
};
use rayon::prelude::*;
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

fn effective_gauss_sigma(gauss_sigma: Option<f64>) -> Option<f64> {
    match gauss_sigma {
        Some(sigma) if sigma == 0.0 => None,
        other => other,
    }
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

pub(super) fn sample_poly_encoding_vectors_gpu<S, T>(
    params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
    base_secret_vec: &S::M,
    all_public_key_matrix: &S::M,
    gadget: &S::M,
    plaintexts: &[T],
    slot_secret_mats: &[Vec<u8>],
    num_slots: usize,
    packed_input_size: usize,
    ncol: usize,
    gauss_sigma: Option<f64>,
) -> Vec<Vec<Arc<[u8]>>>
where
    S: PolyUniformSampler,
    T: AsRef<[Arc<[u8]>]> + Sync,
{
    let slot_parallelism = effective_slot_parallelism_gpu(num_slots).max(1);
    let gauss_sigma = effective_gauss_sigma(gauss_sigma);
    let sampler_shared_by_device = prepare_sampler_shared_by_device(
        params,
        base_secret_vec,
        all_public_key_matrix,
        gadget,
        slot_parallelism,
    );
    let mut encoding_vector_bytes = (0..packed_input_size)
        .map(|_| Vec::with_capacity(num_slots))
        .collect::<Vec<Vec<Arc<[u8]>>>>();

    for slot_start in (0..num_slots).step_by(slot_parallelism) {
        let chunk_len = (slot_start + slot_parallelism).min(num_slots) - slot_start;
        let chunk_sampler_shared_by_device = &sampler_shared_by_device[..chunk_len];
        let chunk_slot_vectors = (0..chunk_len)
            .into_par_iter()
            .map(|offset| {
                let slot = slot_start + offset;
                let shared = &chunk_sampler_shared_by_device[offset];
                let local_params = &shared.params;
                let base_secret_vec = shared.base_secret_vec.as_ref();
                let all_public_key_matrix = shared.all_public_key_matrix.as_ref();
                let gadget = shared.gadget.as_ref();
                let slot_secret_mat =
                    S::M::from_compact_bytes(local_params, slot_secret_mats[slot].as_ref());
                let transformed_secret_vec = base_secret_vec.clone() * &slot_secret_mat;
                let error: S::M = match gauss_sigma {
                    None => S::M::zero(local_params, 1, ncol * packed_input_size),
                    Some(sigma) => {
                        let error_sampler = S::new();
                        error_sampler.sample_uniform(
                            local_params,
                            1,
                            ncol * packed_input_size,
                            DistType::GaussDist { sigma },
                        )
                    }
                };
                let slot_plaintexts = std::iter::once(
                    <<S as PolyUniformSampler>::M as PolyMatrix>::P::const_one(local_params),
                )
                .chain(plaintexts.iter().map(|plaintext_row| {
                    <<S as PolyUniformSampler>::M as PolyMatrix>::P::from_compact_bytes(
                        local_params,
                        plaintext_row.as_ref()[slot].as_ref(),
                    )
                }))
                .collect::<Vec<_>>();
                let encoded_polys_vec = S::M::from_poly_vec_row(local_params, slot_plaintexts);
                let first_term = transformed_secret_vec.clone() * all_public_key_matrix;
                let second_term = encoded_polys_vec.tensor(&(transformed_secret_vec * gadget));
                let all_vector = first_term - second_term + error;

                (0..packed_input_size)
                    .map(|idx| {
                        Arc::<[u8]>::from(
                            all_vector
                                .slice_columns(ncol * idx, ncol * (idx + 1))
                                .into_compact_bytes(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for slot_vector_bytes in chunk_slot_vectors {
            for (idx, vector_bytes) in slot_vector_bytes.into_iter().enumerate() {
                encoding_vector_bytes[idx].push(vector_bytes);
            }
        }
    }

    encoding_vector_bytes
}

#[cfg(test)]
mod tests {
    use super::{effective_gauss_sigma, effective_slot_parallelism_gpu_with_requested};
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

    #[test]
    fn test_effective_gauss_sigma_treats_zero_as_none() {
        assert_eq!(effective_gauss_sigma(None), None);
        assert_eq!(effective_gauss_sigma(Some(0.0)), None);
        assert_eq!(effective_gauss_sigma(Some(1.25)), Some(1.25));
    }
}
