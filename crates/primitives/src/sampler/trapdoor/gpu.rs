#[cfg(test)]
use crate::sampler::bounds::matrix_within_coefficient_bound;
use crate::{
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{Poly, PolyParams, dcrt::gpu::GpuDCRTPolyParams},
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler,
        gpu::{GpuDCRTPolyUniformSampler, random_gpu_rng_seed},
    },
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

const SPECTRAL_CONSTANT: f64 = 1.8;

fn preimage_c(base: u32, sigma: f64) -> f64 {
    (base as f64 + 1.0) * sigma
}

fn preimage_smoothing_parameter(base: u32, sigma: f64, d: usize, n: usize, k: usize) -> f64 {
    SPECTRAL_CONSTANT *
        (base as f64 + 1.0) *
        sigma *
        sigma *
        (((d * n * k) as f64).sqrt() + ((2 * n) as f64).sqrt() + 4.7)
}

fn coeff_cached_matrix(src: &GpuDCRTPolyMatrix) -> GpuDCRTPolyMatrix {
    src.clone().into_coeff_domain()
}

struct GpuPerturbationSamples {
    p1: GpuDCRTPolyMatrix,
    p2: GpuDCRTPolyMatrix,
}

#[derive(Debug, Clone)]
struct GpuP1CovarianceCacheEntry {
    c: f64,
    s: f64,
    dgg_stddev: f64,
    cache: Arc<crate::matrix::gpu_dcrt_poly::GpuP1CovarianceCache>,
}

#[derive(Debug, Clone)]
pub struct GpuDCRTTrapdoor {
    pub r: GpuDCRTPolyMatrix,
    pub e: GpuDCRTPolyMatrix,
    a_mat_coeff: GpuDCRTPolyMatrix,
    b_mat_coeff: GpuDCRTPolyMatrix,
    d_mat_coeff: GpuDCRTPolyMatrix,
    p1_covariance_cache: Arc<Mutex<Option<GpuP1CovarianceCacheEntry>>>,
}

impl PartialEq for GpuDCRTTrapdoor {
    fn eq(&self, other: &Self) -> bool {
        self.r == other.r &&
            self.e == other.e &&
            self.a_mat_coeff == other.a_mat_coeff &&
            self.b_mat_coeff == other.b_mat_coeff &&
            self.d_mat_coeff == other.d_mat_coeff
    }
}

impl Eq for GpuDCRTTrapdoor {}

impl GpuDCRTTrapdoor {
    /// Waits for every matrix required to consume this trapdoor.
    pub fn wait_until_ready(&self) {
        self.r.wait_until_ready();
        self.e.wait_until_ready();
        self.a_mat_coeff.wait_until_ready();
        self.b_mat_coeff.wait_until_ready();
        self.d_mat_coeff.wait_until_ready();
    }

    pub fn new(params: &GpuDCRTPolyParams, size: usize, sigma: f64) -> Self {
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let log_base_q = params.modulus_digits();
        let dist = DistType::GaussDist { sigma, max_coefficient_bound: None };
        let r = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist.clone());
        let e = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        let a_mat_coeff = coeff_cached_matrix(&(&r * &r.transpose()));
        let b_mat_coeff = coeff_cached_matrix(&(&r * &e.transpose()));
        let d_mat_coeff = coeff_cached_matrix(&(&e * &e.transpose()));
        let p1_covariance_cache = Arc::new(Mutex::new(None));
        Self { r, e, a_mat_coeff, b_mat_coeff, d_mat_coeff, p1_covariance_cache }
    }

    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mats = [&self.r, &self.e];
        let mut parts = Vec::with_capacity(mats.len());
        let mut total_len = 0usize;
        for mat in mats {
            let bytes = mat.to_compact_bytes();
            total_len += 8 + bytes.len();
            parts.push(bytes);
        }
        let mut out = Vec::with_capacity(total_len);
        for bytes in parts {
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    pub fn from_compact_bytes(params: &GpuDCRTPolyParams, bytes: &[u8]) -> Option<Self> {
        let mut offset = 0usize;
        let next = |buf: &[u8], offset: &mut usize| -> Option<Vec<u8>> {
            if *offset + 8 > buf.len() {
                return None;
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&buf[*offset..*offset + 8]);
            let len = u64::from_le_bytes(len_bytes) as usize;
            *offset += 8;
            if *offset + len > buf.len() {
                return None;
            }
            let out = buf[*offset..*offset + len].to_vec();
            *offset += len;
            Some(out)
        };
        let r_bytes = next(bytes, &mut offset)?;
        let e_bytes = next(bytes, &mut offset)?;
        if offset != bytes.len() {
            return None;
        }

        let r = GpuDCRTPolyMatrix::from_compact_bytes(params, &r_bytes);
        let e = GpuDCRTPolyMatrix::from_compact_bytes(params, &e_bytes);
        let a_mat_coeff = coeff_cached_matrix(&(&r * &r.transpose()));
        let b_mat_coeff = coeff_cached_matrix(&(&r * &e.transpose()));
        let d_mat_coeff = coeff_cached_matrix(&(&e * &e.transpose()));
        let p1_covariance_cache = Arc::new(Mutex::new(None));
        Some(Self { r, e, a_mat_coeff, b_mat_coeff, d_mat_coeff, p1_covariance_cache })
    }
}

fn p1_covariance_parameters(
    params: &GpuDCRTPolyParams,
    d: usize,
    dgg_stddev: f64,
) -> (f64, f64, f64) {
    let base = 1 << params.base_bits();
    let n = params.ring_dimension() as usize;
    let k = params.modulus_digits();
    let c = preimage_c(base, dgg_stddev);
    let s = preimage_smoothing_parameter(base, dgg_stddev, d, n, k);
    (c, s, dgg_stddev)
}

fn get_or_create_p1_covariance_cache(
    trapdoor: &GpuDCRTTrapdoor,
    c: f64,
    s: f64,
    dgg_stddev: f64,
) -> Arc<crate::matrix::gpu_dcrt_poly::GpuP1CovarianceCache> {
    let mut guard = trapdoor.p1_covariance_cache.lock().expect("p1 cache mutex poisoned");
    if let Some(entry) = guard.as_ref() &&
        entry.c == c &&
        entry.s == s &&
        entry.dgg_stddev == dgg_stddev
    {
        return entry.cache.clone();
    }

    let cache = Arc::new(GpuDCRTPolyMatrix::create_p1_covariance_cache(
        &trapdoor.a_mat_coeff,
        &trapdoor.b_mat_coeff,
        &trapdoor.d_mat_coeff,
        c,
        s,
        dgg_stddev,
    ));
    *guard = Some(GpuP1CovarianceCacheEntry { c, s, dgg_stddev, cache: cache.clone() });
    cache
}

#[derive(Debug, Clone)]
pub struct GpuDCRTPolyTrapdoorSampler {
    sigma: f64,
    base: u32,
    c: f64,
}

#[derive(Debug)]
pub struct GpuPreimageRequest<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    pub entry_idx: usize,
    pub params: &'a <<M as PolyMatrix>::P as Poly>::Params,
    pub trapdoor: &'a T,
    pub public_matrix: &'a M,
    pub target: &'a M,
    pub max_coefficient_bound: BigUint,
}

impl PolyTrapdoorSampler for GpuDCRTPolyTrapdoorSampler {
    type M = GpuDCRTPolyMatrix;
    type Trapdoor = GpuDCRTTrapdoor;

    fn new(params: &<<Self::M as PolyMatrix>::P as Poly>::Params, sigma: f64) -> Self {
        let base = 1 << params.base_bits();
        let c = preimage_c(base, sigma);
        Self { sigma, base, c }
    }

    fn trapdoor(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        size: usize,
    ) -> (Self::Trapdoor, Self::M) {
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let trapdoor = GpuDCRTTrapdoor::new(params, size, self.sigma);
        let a_bar = uniform_sampler.sample_uniform(params, size, size, DistType::FinRingDist);
        let g = GpuDCRTPolyMatrix::gadget_matrix(params, size);
        let a0 = a_bar.concat_columns(&[&GpuDCRTPolyMatrix::identity(params, size, None)]);
        let a1 = &g - &(&a_bar * &trapdoor.r + &trapdoor.e);
        let a = a0.concat_columns(&[&a1]);
        (trapdoor, a)
    }

    fn trapdoor_to_bytes(trapdoor: &Self::Trapdoor) -> Vec<u8> {
        trapdoor.to_compact_bytes()
    }

    fn trapdoor_from_bytes(
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        bytes: &[u8],
    ) -> Option<Self::Trapdoor> {
        GpuDCRTTrapdoor::from_compact_bytes(params, bytes)
    }

    fn preimage(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M {
        let preimage_start = Instant::now();
        let d = public_matrix.row_size();
        let target_cols = target.col_size();
        debug_assert_eq!(
            target.row_size(),
            d,
            "Target matrix should have the same number of rows as the public matrix",
        );
        tracing::debug!(d = d, target_cols = target_cols, "gpu preimage: start");

        let param_start = Instant::now();
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let s = preimage_smoothing_parameter(self.base, self.sigma, d, n, k);
        let dgg_large_std = (s * s - self.c * self.c).sqrt();
        tracing::debug!(
            elapsed_ms = param_start.elapsed().as_secs_f64() * 1_000.0,
            d = d,
            n = n,
            k = k,
            s = s,
            dgg_large_std = dgg_large_std,
            "gpu preimage: parameters derived"
        );

        let p_hat_start = Instant::now();
        let GpuPerturbationSamples { p1, p2 } = sample_pert_square_mat_gpu_native_parts(
            params,
            trapdoor,
            s,
            self.c,
            self.sigma,
            dgg_large_std,
            target_cols,
        );
        tracing::debug!(
            elapsed_ms = p_hat_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: sampled perturbation blocks"
        );

        let perturb_start = Instant::now();
        let p1_rows = p1.row_size();
        let p2_rows = p2.row_size();
        debug_assert_eq!(
            public_matrix.col_size(),
            p1_rows + p2_rows,
            "public matrix columns must match perturbation rows",
        );
        let perturbed_syndrome =
            GpuDCRTPolyMatrix::preimage_residual(target, public_matrix, &p1, &p2);
        tracing::debug!(
            elapsed_ms = perturb_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: computed perturbed_syndrome"
        );

        // Materialize the final layout before sampling z so p1/p2 can be released before the
        // largest correction buffers are live. The correction itself remains one fused kernel.
        let mut out = GpuDCRTPolyMatrix::preimage_output_from_perturbation(p1, p2, target_cols);
        let assemble_start = Instant::now();
        let gauss_start = Instant::now();
        let z_hat_mat =
            perturbed_syndrome.gauss_samp_gq_arb_base(self.c, self.sigma, random_gpu_rng_seed());
        tracing::debug!(
            elapsed_ms = gauss_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: sampled z_hat_mat with gauss_samp_gq_arb_base"
        );

        out.preimage_add_correction(&trapdoor.r, &trapdoor.e, &z_hat_mat);
        tracing::debug!(
            elapsed_ms = assemble_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: assembled output matrix with fused correction"
        );
        tracing::debug!(
            elapsed_ms = preimage_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: finished"
        );
        out
    }

    fn preimage_batched_sharded<'a>(
        &self,
        requests: Vec<GpuPreimageRequest<'a, Self::M, Self::Trapdoor>>,
    ) -> Vec<(usize, Self::M)>
    where
        Self::Trapdoor: Send + Sync + 'a,
        Self::M: 'a,
    {
        tracing::debug!(
            request_count = requests.len(),
            "gpu preimage: start multi-target sharded dispatch"
        );
        let batch_start = Instant::now();
        let common_bound = requests
            .first()
            .map(|request| request.max_coefficient_bound.clone())
            .expect("preimage batch must not be empty");
        assert!(
            requests.iter().all(|request| request.max_coefficient_bound == common_bound),
            "GPU preimage batch requires one common coefficient cutoff"
        );
        let mut pending = requests;
        let mut results = Vec::with_capacity(pending.len());
        let mut round = 0usize;
        while !pending.is_empty() {
            let round_start = Instant::now();
            let pending_before = pending.len();
            let sampled = pending
                .into_par_iter()
                .map(|request| {
                    let candidate = self.preimage(
                        request.params,
                        request.trapdoor,
                        request.public_matrix,
                        request.target,
                    );
                    (request, candidate)
                })
                .collect::<Vec<_>>();
            let (requests, mut candidates): (Vec<_>, Vec<_>) = sampled.into_iter().unzip();
            let accepted =
                GpuDCRTPolyMatrix::batch_within_coefficient_bound(&mut candidates, &common_bound);
            pending = Vec::new();
            let mut accepted_candidates = Vec::new();
            for ((request, candidate), accepted) in
                requests.into_iter().zip(candidates).zip(accepted)
            {
                if accepted {
                    accepted_candidates.push((request.entry_idx, candidate));
                } else {
                    pending.push(request);
                }
            }
            let (entry_indices, mut matrices): (Vec<_>, Vec<_>) =
                accepted_candidates.into_iter().unzip();
            let accepted_count = matrices.len();
            let pending_after = pending.len();
            GpuDCRTPolyMatrix::ntt_batch_in_place(&mut matrices);
            results.extend(entry_indices.into_iter().zip(matrices));
            tracing::debug!(
                round,
                pending_before,
                accepted = accepted_count,
                rejected = pending_after,
                pending_after,
                elapsed_ms = round_start.elapsed().as_secs_f64() * 1_000.0,
                "gpu preimage: rejection round"
            );
            round += 1;
        }
        tracing::debug!(
            rounds = round,
            elapsed_ms = batch_start.elapsed().as_secs_f64() * 1_000.0,
            "gpu preimage: finished multi-target sharded dispatch"
        );
        results
    }

    fn preimage_extend(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        trapdoor: &Self::Trapdoor,
        public_matrix: &Self::M,
        ext_matrix: &Self::M,
        target: &Self::M,
    ) -> Self::M {
        let d = public_matrix.row_size();
        let ext_ncol = ext_matrix.col_size();
        let target_ncol = target.col_size();
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let s = preimage_smoothing_parameter(self.base, self.sigma, d, n, k);

        let dist = DistType::GaussDist { sigma: s, max_coefficient_bound: None };
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let preimage_right = uniform_sampler.sample_uniform(params, ext_ncol, target_ncol, dist);
        let t = target - &(ext_matrix * &preimage_right);
        let preimage_left = self.preimage(params, trapdoor, public_matrix, &t);
        preimage_left.concat_rows(&[&preimage_right])
    }
}

fn sample_pert_square_mat_gpu_native_parts(
    params: &GpuDCRTPolyParams,
    trapdoor: &GpuDCRTTrapdoor,
    s: f64,
    c: f64,
    dgg_stddev: f64,
    sigma_large: f64,
    total_ncol: usize,
) -> GpuPerturbationSamples {
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let d = trapdoor.r.row_size();
    let dk = trapdoor.r.col_size();
    let num_blocks = total_ncol.div_ceil(d);
    let padded_ncol = num_blocks * d;
    let padding_ncol = padded_ncol - total_ncol;
    tracing::debug!(
        d = d,
        dk = dk,
        total_ncol = total_ncol,
        padded_ncol = padded_ncol,
        padding_ncol = padding_ncol,
        "gpu preimage sample_pert: start"
    );

    // p2 is sampled directly on GPU as in the Karney branch of OpenFHE.
    let p2 = uniform_sampler.sample_uniform(
        params,
        dk,
        padded_ncol,
        DistType::GaussDist { sigma: sigma_large, max_coefficient_bound: None },
    );
    tracing::debug!("gpu preimage sample_pert: sampled p2");
    let tp2 = GpuDCRTPolyMatrix::mul_vertical_pair(&trapdoor.r, &trapdoor.e, &p2);
    tracing::debug!("gpu preimage sample_pert: computed tp2");

    // Keep perturbation generation on device: this sampler uses the full
    // 2d x 2d covariance induced by (A, B, D) and Tp2.
    debug_assert_eq!(
        (c, s, dgg_stddev),
        p1_covariance_parameters(params, d, dgg_stddev),
        "cached p1 covariance parameters must match the current preimage parameters",
    );
    let p1_covariance_cache = get_or_create_p1_covariance_cache(trapdoor, c, s, dgg_stddev);
    let p1 = GpuDCRTPolyMatrix::sample_p1_full_cached(
        p1_covariance_cache.as_ref(),
        tp2,
        random_gpu_rng_seed(),
    );
    tracing::debug!("gpu preimage sample_pert: sampled p1");

    GpuPerturbationSamples { p1, p2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        matrix::PolyMatrix,
        poly::{
            PolyParams,
            dcrt::{
                gpu::{detected_gpu_device_ids, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::bounds::{compute_preimage_sigma, hard_cutoff_from_sigma_bound},
    };
    use bigdecimal::{BigDecimal, FromPrimitive};
    use num_bigint::BigUint;
    use serial_test::serial as sequential;

    const SIGMA: f64 = 4.578;

    fn gpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 16, 8)
    }

    fn sample_pert_square_mat_gpu_native(
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        s: f64,
        c: f64,
        dgg_stddev: f64,
        sigma_large: f64,
        total_ncol: usize,
    ) -> GpuDCRTPolyMatrix {
        let GpuPerturbationSamples { p1, p2 } = sample_pert_square_mat_gpu_native_parts(
            params,
            trapdoor,
            s,
            c,
            dgg_stddev,
            sigma_large,
            total_ncol,
        );
        let d = trapdoor.r.row_size();
        let num_blocks = total_ncol.div_ceil(d);
        let padded_ncol = num_blocks * d;
        let padding_ncol = padded_ncol - total_ncol;
        let mut p_hat = GpuDCRTPolyMatrix::new_empty_with_state(
            params,
            p1.row_size() + p2.row_size(),
            total_ncol,
            p1.level(),
            p1.is_ntt(),
        );
        debug_assert!(p1.col_size() >= total_ncol, "p1 must include target columns");
        debug_assert!(p2.col_size() >= total_ncol, "p2 must include target columns");
        p_hat.copy_block_from(&p1, 0, 0, 0, 0, p1.row_size(), total_ncol);
        p_hat.copy_block_from(&p2, p1.row_size(), 0, 0, 0, p2.row_size(), total_ncol);
        tracing::debug!("gpu preimage sample_pert: assembled p_hat without concat+slice");
        if padding_ncol > 0 {
            tracing::debug!("gpu preimage sample_pert: skipped padded columns during assembly");
        }
        p_hat
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _, _) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    #[test]
    #[sequential]
    fn test_gpu_trapdoor_generation() {
        gpu_device_sync();
        let size: usize = 3;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);

        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let expected_rows = size;
        let expected_cols = (params.modulus_digits() + 2) * size;
        assert_eq!(public_matrix.row_size(), expected_rows);
        assert_eq!(public_matrix.col_size(), expected_cols);

        let k = params.modulus_digits();
        let identity = GpuDCRTPolyMatrix::identity(&params, size * k, None);
        let trapdoor_matrix = trapdoor.r.concat_rows(&[&trapdoor.e, &identity]);
        let muled = public_matrix * trapdoor_matrix;
        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&params, size);
        assert_eq!(muled, gadget_matrix);
    }

    #[test]
    #[sequential]
    fn test_gpu_trapdoor_round_trip_bytes() {
        gpu_device_sync();
        let size: usize = 3;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, _public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let bytes =
            <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::trapdoor_to_bytes(&trapdoor);
        let decoded = <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::trapdoor_from_bytes(
            &params, &bytes,
        )
        .expect("trapdoor bytes should decode");
        let reencoded =
            <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::trapdoor_to_bytes(&decoded);
        assert_eq!(
            bytes, reencoded,
            "trapdoor compact bytes should be stable across decode/encode"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_generation_square() {
        gpu_device_sync();
        let size = 3usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);

        let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
        let product = &public_matrix * &preimage;
        assert_eq!(product, target);
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_generation_variable_chunk_widths() {
        gpu_device_sync();
        let size = 3usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();

        for chunk_width in [1usize, 2, 3, 5, 8] {
            let target =
                uniform_sampler.sample_uniform(&params, size, chunk_width, DistType::FinRingDist);
            let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
            assert_eq!(preimage.col_size(), chunk_width);
            assert_eq!(
                &public_matrix * &preimage,
                target,
                "fused preimage relation failed for runtime chunk width {chunk_width}"
            );
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_reuses_trapdoor_cache_for_distinct_targets() {
        gpu_device_sync();
        let size = 3usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();

        let first_target =
            uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);
        let second_target =
            uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);
        assert_ne!(first_target, second_target, "targets should differ");

        let first_preimage =
            trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &first_target);
        let second_preimage =
            trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &second_target);

        assert_eq!(&public_matrix * &first_preimage, first_target);
        assert_eq!(&public_matrix * &second_preimage, second_target);
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_generation_square_not_plain_gadget_solution() {
        gpu_device_sync();
        let size = 3usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);

        // Deterministic gadget preimage baseline:
        // z_plain = [R*z; E*z; z], where z = decompose(target).
        let z_plain = target.decompose();
        let z_plain_former = (&trapdoor.r * &z_plain).concat_rows(&[&(&trapdoor.e * &z_plain)]);
        let z_plain_full = z_plain_former.concat_rows(&[&z_plain]);
        assert_eq!(&public_matrix * &z_plain_full, target);

        let sampled = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
        assert_eq!(&public_matrix * &sampled, target);
        assert_ne!(
            sampled, z_plain_full,
            "preimage sampler should not collapse to the plain deterministic gadget preimage"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_sampler_parameters_follow_instance_sigma() {
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17);
        let params = gpu_params_from_cpu(&cpu_params);
        let base = 1u32 << params.base_bits();
        let default_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let larger_sigma = SIGMA * 1.5;
        let larger_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, larger_sigma);
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let size = 2usize;
        let default_s = preimage_smoothing_parameter(base, default_sampler.sigma, size, n, k);
        let larger_s = preimage_smoothing_parameter(base, larger_sampler.sigma, size, n, k);

        assert_eq!(default_sampler.c, preimage_c(base, SIGMA));
        assert_eq!(larger_sampler.c, preimage_c(base, larger_sigma));
        assert_eq!(
            p1_covariance_parameters(&params, size, larger_sigma),
            (larger_sampler.c, larger_s, larger_sigma)
        );
        assert!(larger_sampler.c > default_sampler.c);
        assert!(larger_s > default_s);
    }

    #[test]
    #[sequential]
    fn test_gpu_batched_preimages_respect_exact_request_cutoff() {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let targets = (0..2)
            .map(|_| uniform_sampler.sample_uniform(&params, size, 1, DistType::FinRingDist))
            .collect::<Vec<_>>();

        let ring_dim_sqrt = BigDecimal::from_u32(params.ring_dimension())
            .expect("ring dimension should convert to BigDecimal")
            .sqrt()
            .expect("ring dimension sqrt should exist");
        let base = BigDecimal::from_biguint(BigUint::from(1u32) << params.base_bits(), 0);
        let preimage_sigma = compute_preimage_sigma(
            &ring_dim_sqrt,
            (size * params.modulus_digits()) as u64,
            &base,
            None,
            Some(SIGMA),
        );
        let cutoff = hard_cutoff_from_sigma_bound(&preimage_sigma);
        let requests = targets
            .iter()
            .enumerate()
            .map(|(entry_idx, target)| GpuPreimageRequest {
                entry_idx,
                params: &params,
                trapdoor: &trapdoor,
                public_matrix: &public_matrix,
                target,
                max_coefficient_bound: cutoff.clone(),
            })
            .collect();

        let mut outputs = sampler.preimage_batched_sharded(requests);
        outputs.sort_unstable_by_key(|(entry_idx, _)| *entry_idx);
        for (expected_idx, ((entry_idx, preimage), target)) in
            outputs.iter().zip(&targets).enumerate()
        {
            assert_eq!(*entry_idx, expected_idx);
            assert_eq!(&public_matrix * preimage, *target);
            assert!(matrix_within_coefficient_bound(&preimage.to_cpu_matrix(), &cutoff));
        }
    }

    fn assert_gpu_preimage_reconstructs_target_and_respects_norm_bound(
        sigma: f64,
        bound_sigma: Option<f64>,
    ) {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17);
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, sigma);
        let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params, size);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();

        let ring_dim_sqrt = BigDecimal::from_u32(params.ring_dimension())
            .expect("ring dimension should convert to BigDecimal")
            .sqrt()
            .expect("ring dimension sqrt should exist");
        let base = BigDecimal::from_biguint(BigUint::from(1u32) << params.base_bits(), 0);
        let m_g = (size * params.modulus_digits()) as u64;
        let preimage_sigma = compute_preimage_sigma(&ring_dim_sqrt, m_g, &base, None, bound_sigma);
        let preimage_bound = hard_cutoff_from_sigma_bound(&preimage_sigma);
        let modulus = params.modulus();

        for sample_idx in 0..4usize {
            let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);
            let preimage = trapdoor_sampler.preimage(&params, &trapdoor, &public_matrix, &target);
            assert_eq!(&public_matrix * &preimage, target);

            for i in 0..preimage.row_size() {
                for j in 0..preimage.col_size() {
                    let poly = preimage.entry(i, j);
                    for (k, coeff) in poly.coeffs().into_iter().enumerate() {
                        let value = coeff.value().clone();
                        let neg = modulus.as_ref() - &value;
                        let centered_abs = if value < neg { value } else { neg };
                        assert!(
                            centered_abs <= preimage_bound,
                            "preimage coeff exceeds preimage maximum coefficient bound at sample={}, row={}, col={}, coeff_idx={}, centered_abs={}, bound={}",
                            sample_idx,
                            i,
                            j,
                            k,
                            centered_abs,
                            preimage_bound
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_coefficients_below_compute_preimage_sigma() {
        assert_gpu_preimage_reconstructs_target_and_respects_norm_bound(SIGMA, None);
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_coefficients_below_compute_preimage_sigma_non_default_sigma() {
        let sigma = SIGMA * 1.25;
        assert_gpu_preimage_reconstructs_target_and_respects_norm_bound(sigma, Some(sigma));
    }

    #[test]
    #[sequential]
    fn test_gpu_p_hat_coefficients_below_compute_preimage_sigma() {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17);
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, _public_matrix) = trapdoor_sampler.trapdoor(&params, size);

        let ring_dim_sqrt = BigDecimal::from_u32(params.ring_dimension())
            .expect("ring dimension should convert to BigDecimal")
            .sqrt()
            .expect("ring dimension sqrt should exist");
        let base = BigDecimal::from_biguint(BigUint::from(1u32) << params.base_bits(), 0);
        let m_g = (size * params.modulus_digits()) as u64;
        let preimage_sigma = compute_preimage_sigma(&ring_dim_sqrt, m_g, &base, None, None);
        let preimage_bound = hard_cutoff_from_sigma_bound(&preimage_sigma);
        let modulus = params.modulus();
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let base_u32 = 1u32 << params.base_bits();
        let c = preimage_c(base_u32, SIGMA);
        let s = preimage_smoothing_parameter(base_u32, SIGMA, size, n, k);
        let dgg_large_std = (s * s - c.powi(2)).sqrt();

        for sample_idx in 0..4usize {
            let p_hat = sample_pert_square_mat_gpu_native(
                &params,
                &trapdoor,
                s,
                c,
                SIGMA,
                dgg_large_std,
                size,
            );
            for i in 0..p_hat.row_size() {
                for j in 0..p_hat.col_size() {
                    let poly = p_hat.entry(i, j);
                    for (coeff_idx, coeff) in poly.coeffs().into_iter().enumerate() {
                        let value = coeff.value().clone();
                        let neg = modulus.as_ref() - &value;
                        let centered_abs = if value < neg { value } else { neg };
                        assert!(
                            centered_abs <= preimage_bound,
                            "p_hat coeff exceeds preimage maximum coefficient bound at sample={}, row={}, col={}, coeff_idx={}, centered_abs={}, bound={}",
                            sample_idx,
                            i,
                            j,
                            coeff_idx,
                            centered_abs,
                            preimage_bound
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_compact_cross_device_restore_relation_and_norm() {
        gpu_device_sync();
        let device_ids = detected_gpu_device_ids();
        if device_ids.len() < 2 {
            return;
        }

        let size = 2usize;
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17);
        let base_params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&base_params, SIGMA);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();

        let ring_dim_sqrt = BigDecimal::from_u32(base_params.ring_dimension())
            .expect("ring dimension should convert to BigDecimal")
            .sqrt()
            .expect("ring dimension sqrt should exist");
        let base = BigDecimal::from_biguint(BigUint::from(1u32) << base_params.base_bits(), 0);
        let m_g = (size * base_params.modulus_digits()) as u64;
        let preimage_sigma = compute_preimage_sigma(&ring_dim_sqrt, m_g, &base, None, None);
        let preimage_bound = hard_cutoff_from_sigma_bound(&preimage_sigma);
        let modulus = base_params.modulus();

        struct DeviceCase {
            src_device: i32,
            dst_device: i32,
            public_matrix_bytes: Vec<u8>,
            target_bytes: Vec<u8>,
            preimage_bytes: Vec<u8>,
        }

        let mut cases = Vec::with_capacity(device_ids.len());
        for (idx, src_device) in device_ids.iter().copied().enumerate() {
            let dst_device = device_ids[(idx + 1) % device_ids.len()];
            assert_ne!(src_device, dst_device, "src and dst devices must differ");

            let src_params = base_params.params_for_device(src_device);
            let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&src_params, size);
            let target =
                uniform_sampler.sample_uniform(&src_params, size, size, DistType::FinRingDist);
            let preimage =
                trapdoor_sampler.preimage(&src_params, &trapdoor, &public_matrix, &target);
            assert_eq!(
                &public_matrix * &preimage,
                target,
                "source-device preimage relation failed on device {}",
                src_device
            );

            cases.push(DeviceCase {
                src_device,
                dst_device,
                public_matrix_bytes: public_matrix.to_compact_bytes(),
                target_bytes: target.to_compact_bytes(),
                preimage_bytes: preimage.to_compact_bytes(),
            });
        }

        for case in cases {
            let dst_params = base_params.params_for_device(case.dst_device);
            let public_matrix =
                GpuDCRTPolyMatrix::from_compact_bytes(&dst_params, &case.public_matrix_bytes);
            let target = GpuDCRTPolyMatrix::from_compact_bytes(&dst_params, &case.target_bytes);
            let preimage = GpuDCRTPolyMatrix::from_compact_bytes(&dst_params, &case.preimage_bytes);

            assert_eq!(
                &public_matrix * &preimage,
                target,
                "cross-device restored preimage relation failed (src_device={}, dst_device={})",
                case.src_device,
                case.dst_device
            );

            for i in 0..preimage.row_size() {
                for j in 0..preimage.col_size() {
                    let poly = preimage.entry(i, j);
                    for (coeff_idx, coeff) in poly.coeffs().into_iter().enumerate() {
                        let value = coeff.value().clone();
                        let neg = modulus.as_ref() - &value;
                        let centered_abs = if value < neg { value } else { neg };
                        assert!(
                            centered_abs <= preimage_bound,
                            "restored preimage coeff exceeds preimage maximum coefficient bound (src_device={}, dst_device={}, row={}, col={}, coeff_idx={}, centered_abs={}, bound={})",
                            case.src_device,
                            case.dst_device,
                            i,
                            j,
                            coeff_idx,
                            centered_abs,
                            preimage_bound
                        );
                    }
                }
            }
        }
    }
}
