use crate::{
    matrix::{
        PolyMatrix, SmallMatrixError,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuSmallMatrix},
    },
    poly::{Poly, PolyParams, dcrt::gpu::GpuDCRTPolyParams},
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler,
        bounds::default_preimage_cutoff,
        gpu::{GpuDCRTPolyUniformSampler, random_gpu_rng_seed},
    },
};
use num_bigint::BigUint;
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

fn dcrt_matrix_bytes(
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
) -> Result<usize, SmallMatrixError> {
    params
        .matrix_allocation_bytes(params.crt_depth().saturating_sub(1), rows, columns, true)
        .map(|allocation| allocation.total_bytes)
        .map_err(|_| SmallMatrixError::DimensionOverflow)
}

enum RetryFailure {
    Error(SmallMatrixError),
    Exhausted(usize),
}

#[inline]
fn residency_fits(required_bytes: usize, budget_bytes: usize) -> bool {
    required_bytes <= budget_bytes
}

fn bounded_retry<T, F>(attempts: usize, mut attempt: F) -> Result<T, RetryFailure>
where
    F: FnMut() -> Result<Option<T>, SmallMatrixError>,
{
    for _ in 0..attempts {
        if let Some(value) = attempt().map_err(RetryFailure::Error)? {
            return Ok(value);
        }
    }
    Err(RetryFailure::Exhausted(attempts))
}

impl GpuDCRTPolyTrapdoorSampler {
    /// Produce the bounded preimage directly in compact GPU storage. Each
    /// retry expands only one complete K-by-C_s candidate tile.
    fn bounded_preimage(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        max_coefficient_bound: BigUint,
    ) -> Result<GpuSmallMatrix, SmallMatrixError> {
        let d = public_matrix.row_size();
        let k = public_matrix.col_size();
        let columns = target.col_size();
        if target.row_size() != d || k == 0 || columns == 0 {
            return Err(SmallMatrixError::ShapeMismatch);
        }
        if public_matrix.params != *params || target.params != *params {
            return Err(SmallMatrixError::ParameterMismatch);
        }
        if trapdoor.r.params != *params || trapdoor.e.params != *params {
            return Err(SmallMatrixError::ParameterMismatch);
        }
        if public_matrix.params.gpu_ids() != params.gpu_ids() ||
            target.params.gpu_ids() != params.gpu_ids() ||
            trapdoor.r.params.gpu_ids() != params.gpu_ids() ||
            trapdoor.e.params.gpu_ids() != params.gpu_ids()
        {
            return Err(SmallMatrixError::DeviceMismatch);
        }
        let budget = crate::env::gpu_small_matrix_residency_bytes()
            .map_err(|_| SmallMatrixError::InvalidConfig)?;
        let allocator_headroom_bytes = crate::env::gpu_small_matrix_allocator_headroom_bytes()
            .map_err(|_| SmallMatrixError::InvalidConfig)?;
        let admission_budget = budget.checked_sub(allocator_headroom_bytes).ok_or(
            SmallMatrixError::ResourceExhausted {
                requested_bytes: allocator_headroom_bytes,
                budget_bytes: budget,
            },
        )?;
        let magnitude_bytes = usize::try_from(max_coefficient_bound.bits().div_ceil(8))
            .map_err(|_| SmallMatrixError::WidthOverflow)?
            .max(1);
        let attempts = crate::env::gpu_preimage_max_tile_attempts()
            .map_err(|_| SmallMatrixError::InvalidConfig)?;
        let mut tile_columns = crate::env::mul_small_rhs_tile_columns()
            .map_err(|_| SmallMatrixError::InvalidConfig)?
            .unwrap_or(1)
            .min(columns);
        let persistent_bytes = [
            public_matrix,
            target,
            &trapdoor.r,
            &trapdoor.e,
            &trapdoor.a_mat_coeff,
            &trapdoor.b_mat_coeff,
            &trapdoor.d_mat_coeff,
        ]
        .into_iter()
        .map(|matrix| dcrt_matrix_bytes(params, matrix.row_size(), matrix.col_size()))
        .try_fold(0usize, |sum, bytes| {
            let bytes = bytes?;
            sum.checked_add(bytes).ok_or(SmallMatrixError::DimensionOverflow)
        })?;
        let compact_bytes = k
            .checked_mul(columns)
            .and_then(|value| value.checked_mul(params.ring_dimension() as usize))
            .and_then(|value| value.checked_mul(1 + magnitude_bytes))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let limbs = params.crt_depth();
        let coefficient_words = usize::try_from(params.modulus().bits().div_ceil(64))
            .map_err(|_| SmallMatrixError::DimensionOverflow)?
            .max(1);
        let pack_check_scratch = limbs
            .checked_mul(
                std::mem::size_of::<*const u8>() +
                    std::mem::size_of::<usize>() +
                    std::mem::size_of::<u8>() +
                    std::mem::size_of::<u64>(),
            )
            .and_then(|bytes| {
                limbs
                    .checked_mul(limbs)
                    .and_then(|entries| entries.checked_mul(std::mem::size_of::<u64>()))
                    .and_then(|garner| bytes.checked_add(garner))
            })
            .and_then(|bytes| {
                coefficient_words
                    .checked_mul(3 * std::mem::size_of::<u64>())
                    .and_then(|words| bytes.checked_add(words))
            })
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        // The device decision word and its pinned host mirror are explicit;
        // opaque page/event/allocator costs are covered by the separate fixed
        // headroom rather than pretending sizeof(handle) measures them.
        let device_acceptance_control_bytes = std::mem::size_of::<i32>();
        let pinned_host_acceptance_control_bytes = std::mem::size_of::<i32>();
        let sampler_event_bytes = 2 * std::mem::size_of::<usize>();
        while tile_columns > 0 {
            let candidate = dcrt_matrix_bytes(params, k, tile_columns)?;
            let perturbation = dcrt_matrix_bytes(params, 2 * d, tile_columns)?
                .checked_add(dcrt_matrix_bytes(params, trapdoor.r.col_size(), tile_columns)?)
                .ok_or(SmallMatrixError::DimensionOverflow)?;
            let residual = dcrt_matrix_bytes(params, d, tile_columns)?;
            let z_hat = dcrt_matrix_bytes(params, trapdoor.r.col_size(), tile_columns)?;
            let target_tile = dcrt_matrix_bytes(params, d, tile_columns)?;
            let live = persistent_bytes
                .checked_add(compact_bytes)
                .and_then(|v| v.checked_add(candidate))
                .and_then(|v| v.checked_add(perturbation))
                .and_then(|v| v.checked_add(residual))
                .and_then(|v| v.checked_add(z_hat))
                .and_then(|v| v.checked_add(target_tile))
                .and_then(|v| v.checked_add(pack_check_scratch))
                .and_then(|v| v.checked_add(device_acceptance_control_bytes))
                .and_then(|v| v.checked_add(pinned_host_acceptance_control_bytes))
                .and_then(|v| v.checked_add(sampler_event_bytes))
                .ok_or(SmallMatrixError::DimensionOverflow)?;
            if residency_fits(live, admission_budget) {
                break;
            }
            tile_columns -= 1;
        }
        if tile_columns == 0 {
            return Err(SmallMatrixError::ResourceExhausted {
                requested_bytes: budget.saturating_add(1),
                budget_bytes: budget,
            });
        }
        let mut destination = GpuSmallMatrix::new_empty_checked(
            params,
            k,
            columns,
            max_coefficient_bound,
            magnitude_bytes,
            admission_budget,
        )?;
        let candidate = dcrt_matrix_bytes(params, k, tile_columns)?;
        let perturbation = dcrt_matrix_bytes(params, 2 * d, tile_columns)?
            .checked_add(dcrt_matrix_bytes(params, trapdoor.r.col_size(), tile_columns)?)
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let residual = dcrt_matrix_bytes(params, d, tile_columns)?;
        let z_hat = dcrt_matrix_bytes(params, trapdoor.r.col_size(), tile_columns)?;
        let target_tile = dcrt_matrix_bytes(params, d, tile_columns)?;
        let check_scratch = residual
            .checked_add(z_hat)
            .and_then(|bytes| bytes.checked_add(pack_check_scratch))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let report = destination.sampler_allocation_report(
            persistent_bytes.checked_add(target_tile).ok_or(SmallMatrixError::DimensionOverflow)?,
            candidate,
            perturbation,
            check_scratch,
            0,
            // Full-matrix queries include their deterministic handles. These
            // are the compact owner and decision-event handles only.
            sampler_event_bytes,
            device_acceptance_control_bytes,
            pinned_host_acceptance_control_bytes,
            allocator_headroom_bytes,
        )?;
        if !residency_fits(report.sampler_peak_bytes, budget) {
            return Err(SmallMatrixError::ResourceExhausted {
                requested_bytes: report.sampler_peak_bytes,
                budget_bytes: budget,
            });
        }
        tracing::debug!(
            persistent_bytes = report.persistent_bytes,
            compact_destination_bytes = report.compact_destination_bytes,
            candidate_bytes = report.candidate_bytes,
            perturbation_bytes = report.perturbation_bytes,
            check_scratch_bytes = report.check_scratch_bytes,
            packed_staging_bytes = report.packed_staging_bytes,
            sampler_event_bytes = report.sampler_event_bytes,
            device_acceptance_control_bytes = report.device_acceptance_control_bytes,
            pinned_host_acceptance_control_bytes = report.pinned_host_acceptance_control_bytes,
            allocator_headroom_bytes = report.allocator_headroom_bytes,
            sampler_peak_bytes = report.sampler_peak_bytes,
            budget_bytes = budget,
            "gpu preimage compact sampler residency"
        );
        for column_start in (0..columns).step_by(tile_columns) {
            let column_count = tile_columns.min(columns - column_start);
            let tile_target = target.slice_columns(column_start, column_start + column_count);
            let outcome = bounded_retry(attempts, || {
                let candidate = expanded_preimage_candidate(
                    self,
                    params,
                    trapdoor,
                    public_matrix,
                    &tile_target,
                )
                .into_coeff_domain();
                let accepted = destination.try_pack_checked_tile(
                    &candidate,
                    0,
                    column_start,
                    k,
                    column_count,
                )?;
                drop(candidate);
                Ok(accepted.then_some(()))
                // Drop only after the flag decision; the GPU owner releases
                // rejected storage in stream order before the next attempt.
            });
            match outcome {
                Ok(()) => {}
                Err(RetryFailure::Error(error)) => return Err(error),
                Err(RetryFailure::Exhausted(attempt_count)) => {
                    return Err(SmallMatrixError::AttemptExhausted {
                        column_start,
                        column_count,
                        attempts: attempt_count,
                    });
                }
            }
        }
        Ok(destination)
    }
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
        max_coefficient_bound: BigUint,
    ) -> Result<GpuSmallMatrix, SmallMatrixError> {
        let minimum = default_preimage_cutoff(
            params.ring_dimension(),
            public_matrix.row_size(),
            params.modulus_digits(),
            self.base,
            self.sigma,
        )
        .ok_or(SmallMatrixError::InvalidConfig)?;
        if max_coefficient_bound < minimum {
            return Err(SmallMatrixError::PreimageBoundTooSmall {
                requested: max_coefficient_bound,
                minimum,
            });
        }
        self.bounded_preimage(params, trapdoor, public_matrix, target, max_coefficient_bound)
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
        let preimage_left = expanded_preimage_candidate(self, params, trapdoor, public_matrix, &t);
        preimage_left.concat_rows(&[&preimage_right])
    }
}

fn expanded_preimage_candidate(
    sampler: &GpuDCRTPolyTrapdoorSampler,
    params: &GpuDCRTPolyParams,
    trapdoor: &GpuDCRTTrapdoor,
    public_matrix: &GpuDCRTPolyMatrix,
    target: &GpuDCRTPolyMatrix,
) -> GpuDCRTPolyMatrix {
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
    let s = preimage_smoothing_parameter(sampler.base, sampler.sigma, d, n, k);
    let dgg_large_std = (s * s - sampler.c * sampler.c).sqrt();
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
        sampler.c,
        sampler.sigma,
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
    let perturbed_syndrome = GpuDCRTPolyMatrix::preimage_residual(target, public_matrix, &p1, &p2);
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
        perturbed_syndrome.gauss_samp_gq_arb_base(sampler.c, sampler.sigma, random_gpu_rng_seed());
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
    tracing::debug!(d = d, dk = dk, total_ncol = total_ncol, "gpu preimage sample_pert: start");

    // p2 is sampled directly on GPU as in the Karney branch of OpenFHE.  The
    // covariance sampler accepts arbitrary column counts; retaining the
    // requested tile width avoids allocating an artificial d-column tail.
    let p2 = uniform_sampler.sample_uniform(
        params,
        dk,
        total_ncol,
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
        matrix::{PolyMatrix, PolyMatrixSmallRhs, SmallPolyMatrix},
        poly::{
            PolyParams,
            dcrt::{
                gpu::{detected_gpu_device_ids, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::bounds::{
            compute_preimage_sigma, default_preimage_cutoff, hard_cutoff_from_sigma_bound,
        },
    };
    use bigdecimal::{BigDecimal, FromPrimitive};
    use num_bigint::BigUint;
    use num_traits::Zero;
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
        p_hat
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_perturbation_keeps_single_column_tail() {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17);
        let params = gpu_params_from_cpu(&cpu_params);
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, _) = trapdoor_sampler.trapdoor(&params, size);
        let n = params.ring_dimension() as usize;
        let k = params.modulus_digits();
        let base = 1u32 << params.base_bits();
        let c = preimage_c(base, SIGMA);
        let s = preimage_smoothing_parameter(base, SIGMA, size, n, k);
        let dgg_large_std = (s * s - c.powi(2)).sqrt();

        let perturbation =
            sample_pert_square_mat_gpu_native(&params, &trapdoor, s, c, SIGMA, dgg_large_std, 1);
        assert_eq!(perturbation.col_size(), 1);
    }

    #[test]
    #[sequential]
    fn test_gpu_compact_preimage_preserves_relation_and_bound() {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = sampler.trapdoor(&params, size);
        let target = GpuDCRTPolyUniformSampler::new().sample_uniform(
            &params,
            size,
            3,
            DistType::FinRingDist,
        );
        let bound = default_preimage_cutoff(
            params.ring_dimension(),
            public_matrix.row_size(),
            params.modulus_digits(),
            1u32 << params.base_bits(),
            SIGMA,
        )
        .expect("default preimage cutoff should be computable");
        let compact = sampler
            .preimage(&params, &trapdoor, &public_matrix, &target, bound.clone())
            .expect("compact preimage should be sampled");
        assert_eq!(compact.rows_count(), public_matrix.col_size());
        assert_eq!(compact.columns_count(), target.col_size());
        assert_eq!(public_matrix.multiply_small_rhs(compact).unwrap(), target);
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_rejects_bound_below_default_cutoff_before_sampling() {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = sampler.trapdoor(&params, size);
        let target = GpuDCRTPolyUniformSampler::new().sample_uniform(
            &params,
            size,
            1,
            DistType::FinRingDist,
        );
        let minimum = default_preimage_cutoff(
            params.ring_dimension(),
            public_matrix.row_size(),
            params.modulus_digits(),
            1u32 << params.base_bits(),
            SIGMA,
        )
        .expect("default preimage cutoff should be computable");
        let requested = &minimum - BigUint::from(1u8);
        assert_eq!(
            sampler.preimage(&params, &trapdoor, &public_matrix, &target, requested.clone()),
            Err(SmallMatrixError::PreimageBoundTooSmall { requested, minimum })
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_compact_preimage_rejects_k_by_one_budget_before_allocation() {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = sampler.trapdoor(&params, size);
        let target = GpuDCRTPolyMatrix::identity(&params, size, None).slice_columns(0, 1);
        let previous = std::env::var_os("MXX_GPU_SMALL_MATRIX_RESIDENCY_BYTES");
        unsafe { std::env::set_var("MXX_GPU_SMALL_MATRIX_RESIDENCY_BYTES", "1") };
        let result = sampler.bounded_preimage(
            &params,
            &trapdoor,
            &public_matrix,
            &target,
            BigUint::from(1u32) << 20,
        );
        match previous {
            Some(value) => unsafe {
                std::env::set_var("MXX_GPU_SMALL_MATRIX_RESIDENCY_BYTES", value)
            },
            None => unsafe { std::env::remove_var("MXX_GPU_SMALL_MATRIX_RESIDENCY_BYTES") },
        }
        assert!(matches!(result, Err(SmallMatrixError::ResourceExhausted { .. })));
    }

    fn canonical_maximum(payload: &[u8], magnitude_bytes: usize) -> BigUint {
        let width = 1 + magnitude_bytes;
        payload
            .chunks_exact(width)
            .map(|coefficient| BigUint::from_bytes_le(&coefficient[1..]))
            .max()
            .unwrap_or_default()
    }

    fn sample_candidate_with_canonical_maximum(
        sampler: &GpuDCRTPolyTrapdoorSampler,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
    ) -> (GpuDCRTPolyMatrix, BigUint) {
        let candidate =
            expanded_preimage_candidate(sampler, params, trapdoor, public_matrix, target)
                .into_coeff_domain();
        let inspection_bound = params.modulus().as_ref() >> 1u8;
        let mut canonical = GpuSmallMatrix::new_empty(
            params,
            candidate.row_size(),
            candidate.col_size(),
            inspection_bound,
        )
        .expect("canonical inspection owner");
        assert!(
            canonical
                .try_pack_checked_tile(
                    &candidate,
                    0,
                    0,
                    candidate.row_size(),
                    candidate.col_size(),
                )
                .expect("production candidate check/pack")
        );
        let payload = canonical.to_canonical_coefficients().expect("canonical candidate bytes");
        let maximum = canonical_maximum(&payload, canonical.magnitude_width());
        (candidate, maximum)
    }

    #[test]
    #[sequential]
    fn test_gpu_bounded_retry_rejects_then_packs_real_preimage_candidate() {
        gpu_device_sync();
        const SEARCH_LIMIT: usize = 12;
        let size = 2usize;
        let params = gpu_params_from_cpu(&gpu_test_params());
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = sampler.trapdoor(&params, size);
        let target = GpuDCRTPolyUniformSampler::new().sample_uniform(
            &params,
            size,
            1,
            DistType::FinRingDist,
        );

        let mut sampled = Vec::new();
        for _ in 0..SEARCH_LIMIT {
            sampled.push(sample_candidate_with_canonical_maximum(
                &sampler,
                &params,
                &trapdoor,
                &public_matrix,
                &target,
            ));
            sampled.sort_by(|left, right| left.1.cmp(&right.1));
            if sampled.first().is_some_and(|low| {
                sampled.last().is_some_and(|high| high.1 > low.1 && !low.1.is_zero())
            }) {
                break;
            }
        }
        let (low_candidate, low_maximum) = sampled.remove(0);
        let (high_candidate, high_maximum) =
            sampled.pop().expect("bounded search must produce at least two real candidates");
        assert!(
            high_maximum > low_maximum && !low_maximum.is_zero(),
            "failed to find distinct nonzero candidate maxima in {SEARCH_LIMIT} draws"
        );

        let mut destination =
            GpuSmallMatrix::new_empty(&params, public_matrix.col_size(), 1, low_maximum.clone())
                .expect("bounded destination");
        let mut candidates = [Some(high_candidate), Some(low_candidate)].into_iter();
        let mut attempt_count = 0usize;
        let outcome = bounded_retry(2, || {
            attempt_count += 1;
            let candidate =
                candidates.next().flatten().expect("retry must consume exactly two candidates");
            let accepted = destination.try_pack_checked_tile(
                &candidate,
                0,
                0,
                candidate.row_size(),
                candidate.col_size(),
            )?;
            drop(candidate);
            Ok(accepted.then_some(()))
        });
        assert!(outcome.is_ok());
        assert_eq!(attempt_count, 2);
        assert_eq!(public_matrix.multiply_small_rhs(destination).unwrap(), target);
        params.fence_released_memory();
    }

    #[test]
    #[sequential]
    fn test_gpu_bounded_retry_exactly_exhausts_real_preimage_candidates() {
        gpu_device_sync();
        const MAX_ATTEMPTS: usize = 3;
        let size = 2usize;
        let params = gpu_params_from_cpu(&gpu_test_params());
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (trapdoor, public_matrix) = sampler.trapdoor(&params, size);
        let target = GpuDCRTPolyMatrix::identity(&params, size, None).slice_columns(0, 1);
        let previous = std::env::var_os("MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS");
        unsafe {
            std::env::set_var("MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS", MAX_ATTEMPTS.to_string())
        };
        // A nonzero target cannot have an all-zero relation-valid preimage, so
        // the exact zero bound forces every production candidate check/pack to
        // reject without relying on an injected acceptance sequence.
        let outcome =
            sampler.bounded_preimage(&params, &trapdoor, &public_matrix, &target, BigUint::ZERO);
        match previous {
            Some(value) => unsafe {
                std::env::set_var("MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS", value)
            },
            None => unsafe { std::env::remove_var("MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS") },
        }
        assert!(matches!(
            outcome,
            Err(SmallMatrixError::AttemptExhausted {
                column_start: 0,
                column_count: 1,
                attempts: MAX_ATTEMPTS,
            })
        ));
        params.fence_released_memory();
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _, _) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    fn permissive_preimage_bound(params: &GpuDCRTPolyParams) -> BigUint {
        params.modulus().as_ref() >> 1u8
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

        let preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &target,
                permissive_preimage_bound(&params),
            )
            .expect("permissive bound should accept a preimage");
        let product = public_matrix.multiply_small_rhs(preimage).unwrap();
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
            let preimage = trapdoor_sampler
                .preimage(
                    &params,
                    &trapdoor,
                    &public_matrix,
                    &target,
                    permissive_preimage_bound(&params),
                )
                .expect("permissive bound should accept a preimage");
            assert_eq!(preimage.columns_count(), chunk_width);
            assert_eq!(
                public_matrix.multiply_small_rhs(preimage).unwrap(),
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

        let first_preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &first_target,
                permissive_preimage_bound(&params),
            )
            .expect("permissive bound should accept the first preimage");
        let second_preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &second_target,
                permissive_preimage_bound(&params),
            )
            .expect("permissive bound should accept the second preimage");

        assert_eq!(public_matrix.multiply_small_rhs(first_preimage).unwrap(), first_target);
        assert_eq!(public_matrix.multiply_small_rhs(second_preimage).unwrap(), second_target);
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

        let bound = permissive_preimage_bound(&params);
        let mut plain = GpuSmallMatrix::new_empty(
            &params,
            z_plain_full.row_size(),
            z_plain_full.col_size(),
            bound.clone(),
        )
        .expect("plain compact owner");
        assert!(
            plain
                .try_pack_checked_tile(
                    &z_plain_full.clone().into_coeff_domain(),
                    0,
                    0,
                    z_plain_full.row_size(),
                    z_plain_full.col_size(),
                )
                .expect("plain gadget preimage should fit the permissive bound")
        );
        let sampled = trapdoor_sampler
            .preimage(&params, &trapdoor, &public_matrix, &target, bound)
            .expect("permissive bound should accept a sampled preimage");
        assert_eq!(public_matrix.multiply_small_rhs(sampled.clone()).unwrap(), target);
        assert_ne!(
            sampled.to_canonical_coefficients().unwrap(),
            plain.to_canonical_coefficients().unwrap(),
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
    fn test_gpu_multiple_preimages_respect_exact_request_cutoff() {
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
        let outputs = targets
            .iter()
            .map(|target| {
                sampler
                    .preimage(&params, &trapdoor, &public_matrix, target, cutoff.clone())
                    .expect("bounded preimage")
            })
            .collect::<Vec<_>>();

        for (preimage, target) in outputs.into_iter().zip(&targets) {
            assert_eq!(preimage.max_coefficient_bound(), &cutoff);
            assert_eq!(public_matrix.multiply_small_rhs(preimage).unwrap(), *target);
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
        for sample_idx in 0..4usize {
            let target = uniform_sampler.sample_uniform(&params, size, size, DistType::FinRingDist);
            let preimage = trapdoor_sampler
                .preimage(&params, &trapdoor, &public_matrix, &target, preimage_bound.clone())
                .expect("bounded sampler should return a valid preimage");
            assert_eq!(preimage.max_coefficient_bound(), &preimage_bound);
            let maximum = canonical_maximum(
                &preimage.to_canonical_coefficients().expect("canonical preimage"),
                preimage.magnitude_width(),
            );
            assert!(
                maximum <= preimage_bound,
                "preimage coeff exceeds maximum coefficient bound at sample={sample_idx}, maximum={maximum}, bound={preimage_bound}"
            );
            assert_eq!(public_matrix.multiply_small_rhs(preimage).unwrap(), target);
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
        struct DeviceCase {
            src_device: i32,
            dst_device: i32,
            public_matrix_bytes: Vec<u8>,
            target_bytes: Vec<u8>,
            preimage_payload: Vec<u8>,
        }

        let mut cases = Vec::with_capacity(device_ids.len());
        for (idx, src_device) in device_ids.iter().copied().enumerate() {
            let dst_device = device_ids[(idx + 1) % device_ids.len()];
            assert_ne!(src_device, dst_device, "src and dst devices must differ");

            let src_params = base_params.params_for_device(src_device);
            let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&src_params, size);
            let target =
                uniform_sampler.sample_uniform(&src_params, size, size, DistType::FinRingDist);
            let preimage = trapdoor_sampler
                .preimage(&src_params, &trapdoor, &public_matrix, &target, preimage_bound.clone())
                .expect("bounded source-device preimage");
            assert_eq!(
                public_matrix.multiply_small_rhs(preimage.clone()).unwrap(),
                target,
                "source-device preimage relation failed on device {}",
                src_device
            );

            cases.push(DeviceCase {
                src_device,
                dst_device,
                public_matrix_bytes: public_matrix.to_compact_bytes(),
                target_bytes: target.to_compact_bytes(),
                preimage_payload: preimage
                    .to_canonical_coefficients()
                    .expect("canonical compact preimage"),
            });
        }

        for case in cases {
            let dst_params = base_params.params_for_device(case.dst_device);
            let public_matrix =
                GpuDCRTPolyMatrix::from_compact_bytes(&dst_params, &case.public_matrix_bytes);
            let target = GpuDCRTPolyMatrix::from_compact_bytes(&dst_params, &case.target_bytes);
            let preimage = GpuSmallMatrix::from_canonical_coefficients(
                &dst_params,
                public_matrix.col_size(),
                target.col_size(),
                preimage_bound.clone(),
                &case.preimage_payload,
            )
            .expect("restore compact preimage on destination device");

            assert_eq!(
                public_matrix.multiply_small_rhs(preimage.clone()).unwrap(),
                target,
                "cross-device restored preimage relation failed (src_device={}, dst_device={})",
                case.src_device,
                case.dst_device
            );

            let maximum = canonical_maximum(
                &preimage.to_canonical_coefficients().expect("canonical restored preimage"),
                preimage.magnitude_width(),
            );
            assert!(
                maximum <= preimage_bound,
                "restored preimage exceeds maximum coefficient bound (src_device={}, dst_device={}, maximum={}, bound={})",
                case.src_device,
                case.dst_device,
                maximum,
                preimage_bound
            );
        }
    }
}
