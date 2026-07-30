use super::{
    Backend, PreimageRequest,
    poly::{PolyBackend, PolyBackendError},
};
use mxx_primitives::{
    matrix::PolyMatrix,
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
};

pub(super) fn sample_preimage_batch<M, U, H, T>(
    backend: &mut PolyBackend<M, U, H, T>,
    requests: Vec<PreimageRequest<M, T::Trapdoor>>,
) -> Result<Vec<M>, PolyBackendError>
where
    M: PolyMatrix + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    use mxx_primitives::sampler::trapdoor::GpuPreimageRequest;

    let Some(first) = requests.first() else {
        return Ok(Vec::new());
    };
    if requests
        .iter()
        .any(|request| request.matrix_type != first.matrix_type || request.sigma != first.sigma)
    {
        return requests
            .into_iter()
            .map(|request| {
                backend.sample_preimage(
                    &request.matrix_type,
                    request.sigma,
                    &request.trapdoor,
                    &request.public,
                    &request.target,
                )
            })
            .collect();
    }
    let parameters = backend.parameters(&first.matrix_type)?;
    let sampler = T::new(parameters, first.sigma);
    let batched = requests
        .iter()
        .enumerate()
        .map(|(entry_idx, request)| GpuPreimageRequest {
            entry_idx,
            params: parameters,
            trapdoor: &request.trapdoor,
            public_matrix: &request.public,
            target: request.target.clone(),
        })
        .collect();
    let mut results = sampler.preimage_batched_sharded(batched);
    results.sort_unstable_by_key(|(entry_idx, _)| *entry_idx);
    Ok(results.into_iter().map(|(_, matrix)| matrix).collect())
}
