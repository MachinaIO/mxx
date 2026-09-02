use crate::{
    matrix::{
        PolyMatrix,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuMatrixSampleDist},
    },
    poly::{
        Poly,
        dcrt::gpu::{GpuDCRTPolyParams, GpuRngSeed},
    },
    sampler::{DistType, PolyHashSampler, PolyUniformSampler},
};
use digest::OutputSizeUser;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
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
        let sampled = self.sample_uniform(params, 1, 1, dist.clone());
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

    fn sample_hash_gadget_source<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        let seed = hash_seed_for_matrix::<H>(key, tag.as_ref());
        sample_gpu_matrix_with_seed_coeff(params, nrow, ncol, dist, seed)
    }

    fn sample_hash_columns<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        total_ncol: usize,
        col_start: usize,
        col_len: usize,
        dist: DistType,
    ) -> Self::M {
        let seed = hash_seed_for_matrix::<H>(key, tag.as_ref());
        sample_gpu_matrix_with_seed_columns(
            params, nrow, total_ncol, col_start, col_len, dist, seed,
        )
    }

    fn sample_hash_decomposed<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        let seed = hash_seed_for_matrix::<H>(key, tag.as_ref());
        sample_gpu_matrix_with_seed_coeff(params, nrow, ncol, dist, seed).decompose()
    }

    fn sample_hash_small_decomposed<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        key: [u8; 32],
        tag: B,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> Self::M {
        let seed = hash_seed_for_matrix::<H>(key, tag.as_ref());
        sample_gpu_matrix_with_seed_coeff(params, nrow, ncol, dist, seed).small_decompose()
    }
}

fn hash_seed_for_matrix<H: digest::Digest>(key: [u8; 32], tag: &[u8]) -> GpuRngSeed {
    let mut seed_bytes = [0u8; 32];
    let mut written = 0usize;
    let mut counter = 0u32;
    while written < seed_bytes.len() {
        let mut hasher = H::new();
        hasher.update(b"GpuDCRTPolyHashSampler/v2");
        hasher.update(key);
        hasher.update(tag);
        hasher.update(counter.to_le_bytes());
        let digest = hasher.finalize();
        assert!(!digest.is_empty(), "digest output must not be empty");
        let take = (seed_bytes.len() - written).min(digest.len());
        seed_bytes[written..written + take].copy_from_slice(&digest[..take]);
        written += take;
        counter = counter.wrapping_add(1);
    }
    GpuRngSeed::from_bytes(seed_bytes)
}

pub(crate) fn random_gpu_rng_seed() -> GpuRngSeed {
    let mut seed_bytes = [0u8; 32];
    rng().fill(&mut seed_bytes);
    GpuRngSeed::from_bytes(seed_bytes)
}

fn sample_gpu_matrix_native(
    params: &GpuDCRTPolyParams,
    nrow: usize,
    ncol: usize,
    dist: DistType,
) -> GpuDCRTPolyMatrix {
    sample_gpu_matrix_with_seed(params, nrow, ncol, dist, random_gpu_rng_seed())
}

fn sample_gpu_matrix_with_seed(
    params: &GpuDCRTPolyParams,
    nrow: usize,
    ncol: usize,
    dist: DistType,
    seed: GpuRngSeed,
) -> GpuDCRTPolyMatrix {
    sample_gpu_matrix_with_seed_format(params, nrow, ncol, dist, seed, true)
}

fn sample_gpu_matrix_with_seed_coeff(
    params: &GpuDCRTPolyParams,
    nrow: usize,
    ncol: usize,
    dist: DistType,
    seed: GpuRngSeed,
) -> GpuDCRTPolyMatrix {
    sample_gpu_matrix_with_seed_format(params, nrow, ncol, dist, seed, false)
}

fn sample_gpu_matrix_with_seed_format(
    params: &GpuDCRTPolyParams,
    nrow: usize,
    ncol: usize,
    dist: DistType,
    seed: GpuRngSeed,
    is_ntt: bool,
) -> GpuDCRTPolyMatrix {
    let sample = |dist, sigma, max_coefficient_bound| {
        if is_ntt {
            GpuDCRTPolyMatrix::sample_distribution(
                params,
                nrow,
                ncol,
                dist,
                sigma,
                max_coefficient_bound,
                seed,
            )
        } else {
            GpuDCRTPolyMatrix::sample_distribution_coeff(
                params,
                nrow,
                ncol,
                dist,
                sigma,
                max_coefficient_bound,
                seed,
            )
        }
    };
    match dist {
        DistType::FinRingDist => sample(GpuMatrixSampleDist::Uniform, 0.0, u64::MAX),
        DistType::GaussDist { sigma, max_coefficient_bound } => sample(
            GpuMatrixSampleDist::Gauss,
            sigma,
            gpu_coefficient_cutoff(max_coefficient_bound.as_ref()),
        ),
        DistType::BitDist => sample(GpuMatrixSampleDist::Bit, 0.0, u64::MAX),
        DistType::TernaryDist => sample(GpuMatrixSampleDist::Ternary, 0.0, u64::MAX),
    }
}

fn sample_gpu_matrix_with_seed_columns(
    params: &GpuDCRTPolyParams,
    nrow: usize,
    total_ncol: usize,
    col_start: usize,
    col_len: usize,
    dist: DistType,
    seed: GpuRngSeed,
) -> GpuDCRTPolyMatrix {
    if nrow == 0 || col_len == 0 {
        return GpuDCRTPolyMatrix::zero(params, nrow, col_len);
    }
    match dist {
        DistType::FinRingDist => GpuDCRTPolyMatrix::sample_distribution_columns(
            params,
            nrow,
            total_ncol,
            col_start,
            col_len,
            GpuMatrixSampleDist::Uniform,
            0.0,
            u64::MAX,
            seed,
        ),
        DistType::GaussDist { sigma, max_coefficient_bound } => {
            GpuDCRTPolyMatrix::sample_distribution_columns(
                params,
                nrow,
                total_ncol,
                col_start,
                col_len,
                GpuMatrixSampleDist::Gauss,
                sigma,
                gpu_coefficient_cutoff(max_coefficient_bound.as_ref()),
                seed,
            )
        }
        DistType::BitDist => GpuDCRTPolyMatrix::sample_distribution_columns(
            params,
            nrow,
            total_ncol,
            col_start,
            col_len,
            GpuMatrixSampleDist::Bit,
            0.0,
            u64::MAX,
            seed,
        ),
        DistType::TernaryDist => GpuDCRTPolyMatrix::sample_distribution_columns(
            params,
            nrow,
            total_ncol,
            col_start,
            col_len,
            GpuMatrixSampleDist::Ternary,
            0.0,
            u64::MAX,
            seed,
        ),
    }
}

fn gpu_coefficient_cutoff(max_coefficient_bound: Option<&BigUint>) -> u64 {
    max_coefficient_bound.and_then(ToPrimitive::to_u64).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        matrix::PolyMatrix,
        poly::{
            Poly, PolyParams,
            dcrt::{gpu::gpu_device_sync, params::DCRTPolyParams},
        },
    };
    use keccak_asm::Keccak256;
    use num_bigint::BigUint;
    use serial_test::serial as sequential;

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
    fn test_gpu_hash_sampler_decomposed_matches_legacy_path() {
        gpu_device_sync();
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sampler = GpuDCRTPolyHashSampler::<Keccak256>::new();
        let key = [11u8; 32];
        let tag = b"gpu-hash-decomposed";

        let sampled_decomposed =
            sampler.sample_hash_decomposed(&params, key, tag, 3, 4, DistType::FinRingDist);
        let sampled_legacy = sampler.sample_hash(&params, key, tag, 3, 4, DistType::FinRingDist);
        let sampled_gadget_source =
            sampler.sample_hash_gadget_source(&params, key, tag, 3, 4, DistType::FinRingDist);
        assert!(!sampled_gadget_source.is_ntt());
        assert_eq!(sampled_gadget_source.ensure_eval_domain(), sampled_legacy);
        let sampled_legacy_decomposed = sampled_legacy.decompose();
        assert_eq!(sampled_decomposed, sampled_legacy_decomposed);
    }

    #[test]
    #[sequential]
    fn test_gpu_hash_sampler_column_subrange_matches_full_sample() {
        gpu_device_sync();
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sampler = GpuDCRTPolyHashSampler::<Keccak256>::new();
        let key = [13u8; 32];
        let tag = b"gpu-hash-columns";

        let full = sampler.sample_hash(&params, key, tag, 4, 9, DistType::FinRingDist);
        let chunk =
            sampler.sample_hash_columns(&params, key, tag, 4, 9, 2, 3, DistType::FinRingDist);
        assert_eq!(chunk, full.slice_columns(2, 5));

        let decomposed = sampler.sample_hash_decomposed_columns(
            &params,
            key,
            tag,
            4,
            9,
            2,
            3,
            DistType::FinRingDist,
        );
        assert_eq!(decomposed, full.slice_columns(2, 5).decompose());

        let small_decomposed = sampler.sample_hash_small_decomposed_columns(
            &params,
            key,
            tag,
            4,
            9,
            2,
            3,
            DistType::FinRingDist,
        );
        assert_eq!(small_decomposed, full.slice_columns(2, 5).small_decompose());
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
            DistType::GaussDist { sigma, max_coefficient_bound: None },
            GpuRngSeed::from_bytes([0x5au8; 32]),
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

    #[test]
    #[sequential]
    fn test_gpu_truncated_gaussian_never_exceeds_integer_cutoff() {
        gpu_device_sync();
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let cutoff = BigUint::from(2u8);
        let sampled = sample_gpu_matrix_with_seed(
            &params,
            4,
            5,
            DistType::GaussDist { sigma: 4.578, max_coefficient_bound: Some(cutoff.clone()) },
            GpuRngSeed::from_bytes([0x6bu8; 32]),
        );

        assert!(crate::sampler::bounds::matrix_within_coefficient_bound(&sampled, &cutoff));
    }
}
