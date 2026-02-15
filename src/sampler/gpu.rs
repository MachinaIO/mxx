use crate::{
    matrix::{
        PolyMatrix,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuMatrixSampleDist},
    },
    poly::{Poly, dcrt::gpu::GpuDCRTPolyParams},
    sampler::{DistType, PolyHashSampler, PolyUniformSampler},
};
use digest::OutputSizeUser;
use rand::{Rng, rng};
use std::marker::PhantomData;

#[derive(Debug, Clone, Default)]
pub struct GpuDCRTPolyUniformSampler {}

impl PolyUniformSampler for GpuDCRTPolyUniformSampler {
    type M = GpuDCRTPolyMatrix;

    fn new() -> Self {
        Self {}
    }

    fn sample_poly(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        dist: &DistType,
    ) -> <Self::M as PolyMatrix>::P {
        let sampled = self.sample_uniform(params, 1, 1, *dist);
        sampled.entry(0, 0)
    }

    fn sample_uniform(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        sample_gpu_matrix_native(params, nrow, ncol, dist)
    }
}

#[derive(Debug, Clone)]
pub struct GpuDCRTPolyHashSampler<H: OutputSizeUser + digest::Digest> {
    _h: PhantomData<H>,
}

impl<H> PolyHashSampler<[u8; 32]> for GpuDCRTPolyHashSampler<H>
where
    H: OutputSizeUser + digest::Digest + Send + Sync,
{
    type M = GpuDCRTPolyMatrix;

    fn new() -> Self {
        Self { _h: PhantomData }
    }

    fn sample_hash<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        let seed = hash_seed_for_matrix::<H>(key, tag.as_ref());
        sample_gpu_matrix_with_seed(params, nrow, ncol, dist, seed)
    }
}

fn hash_seed_for_matrix<H: digest::Digest>(key: [u8; 32], tag: &[u8]) -> u64 {
    let mut hasher = H::new();
    hasher.update(b"GpuDCRTPolyHashSampler/v1");
    hasher.update(key);
    hasher.update(tag);
    let digest = hasher.finalize();
    let mut seed_bytes = [0u8; 8];
    let take = digest.len().min(seed_bytes.len());
    seed_bytes[..take].copy_from_slice(&digest[..take]);
    if take == 0 {
        return 0;
    }
    for i in take..seed_bytes.len() {
        seed_bytes[i] = digest[i % take];
    }
    u64::from_le_bytes(seed_bytes)
}

fn sample_gpu_matrix_native(
    params: &GpuDCRTPolyParams,
    nrow: usize,
    ncol: usize,
    dist: DistType,
) -> GpuDCRTPolyMatrix {
    let mut prng = rng();
    let seed: u64 = prng.random();
    sample_gpu_matrix_with_seed(params, nrow, ncol, dist, seed)
}

fn sample_gpu_matrix_with_seed(
    params: &GpuDCRTPolyParams,
    nrow: usize,
    ncol: usize,
    dist: DistType,
    seed: u64,
) -> GpuDCRTPolyMatrix {
    if nrow == 0 || ncol == 0 {
        return GpuDCRTPolyMatrix::zero(params, nrow, ncol);
    }
    match dist {
        DistType::FinRingDist => GpuDCRTPolyMatrix::sample_distribution(
            params,
            nrow,
            ncol,
            GpuMatrixSampleDist::Uniform,
            0.0,
            seed,
        ),
        DistType::GaussDist { sigma } => GpuDCRTPolyMatrix::sample_distribution(
            params,
            nrow,
            ncol,
            GpuMatrixSampleDist::Gauss,
            sigma,
            seed,
        ),
        DistType::BitDist => GpuDCRTPolyMatrix::sample_distribution(
            params,
            nrow,
            ncol,
            GpuMatrixSampleDist::Bit,
            0.0,
            seed,
        ),
        DistType::TernaryDist => GpuDCRTPolyMatrix::sample_distribution(
            params,
            nrow,
            ncol,
            GpuMatrixSampleDist::Ternary,
            0.0,
            seed,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        element::PolyElem,
        matrix::PolyMatrix,
        poly::{
            Poly, PolyParams,
            dcrt::{gpu::gpu_device_sync, params::DCRTPolyParams},
        },
    };
    use keccak_asm::Keccak256;
    use num_bigint::BigUint;
    use sequential_test::sequential;

    fn gpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 16, 8)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _, _) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    #[test]
    #[sequential]
    fn test_gpu_uniform_sampler_size() {
        gpu_device_sync();
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sampler = GpuDCRTPolyUniformSampler::new();
        let sampled = sampler.sample_uniform(&params, 3, 4, DistType::FinRingDist);
        assert_eq!(sampled.row_size(), 3);
        assert_eq!(sampled.col_size(), 4);
    }

    #[test]
    #[sequential]
    fn test_gpu_hash_sampler_is_deterministic() {
        gpu_device_sync();
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sampler = GpuDCRTPolyHashSampler::<Keccak256>::new();
        let key = [7u8; 32];
        let tag = b"gpu-hash";
        let sampled1 = sampler.sample_hash(&params, key, tag, 4, 5, DistType::FinRingDist);
        let sampled2 = sampler.sample_hash(&params, key, tag, 4, 5, DistType::FinRingDist);
        assert_eq!(sampled1, sampled2);
    }

    #[test]
    #[sequential]
    fn test_sample_gpu_matrix_with_seed_gauss_coeff_lt_6sigma() {
        gpu_device_sync();
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sigma = 4.578;
        let sampled = sample_gpu_matrix_with_seed(
            &params,
            4,
            5,
            DistType::GaussDist { sigma },
            0x1234_5678_9abc_def0,
        );

        let bound = sigma * 6.0;
        let strict_upper = BigUint::from(bound.ceil() as u64);
        let q = params.modulus();

        for i in 0..sampled.row_size() {
            for j in 0..sampled.col_size() {
                let poly = sampled.entry(i, j);
                for (k, coeff) in poly.coeffs().into_iter().enumerate() {
                    let value = coeff.value().clone();
                    let neg = q.as_ref() - &value;
                    let centered_abs = if value < neg { value } else { neg };
                    assert!(
                        centered_abs < strict_upper,
                        "gauss coeff bound violated at ({i},{j}) coeff={k}: centered_abs={} >= {} (sigma={}, 6sigma={})",
                        centered_abs,
                        strict_upper,
                        sigma,
                        bound
                    );
                }
            }
        }
    }
}
