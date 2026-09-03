use crate::{
    matrix::{
        PolyMatrix, PolyMatrixColumnSource, SmallMatrixError,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuSmallMatrix},
    },
    poly::{
        Poly, PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, GpuRngSeed},
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler,
        bounds::default_preimage_cutoff,
        gpu::{GpuDCRTPolyUniformSampler, random_gpu_rng_seed},
    },
};
use digest::Digest;
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

fn preimage_seed(base: [u8; 32], stage: &[u8], column_start: usize, attempt: usize) -> GpuRngSeed {
    let mut hasher = keccak_asm::Keccak256::new();
    hasher.update(b"mxx-preimage-sampler/v1");
    hasher.update(base);
    hasher.update((stage.len() as u64).to_le_bytes());
    hasher.update(stage);
    hasher.update((column_start as u64).to_le_bytes());
    hasher.update((attempt as u64).to_le_bytes());
    GpuRngSeed::from_bytes(hasher.finalize().into())
}

/// Sample a bounded preimage directly into one resident all-column compact owner.
/// Expanded candidates are limited to one target-column tile and are discarded
/// immediately after the device-side bound decision.
impl GpuDCRTPolyTrapdoorSampler {
    pub fn preimage_small(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &dyn PolyMatrixColumnSource<GpuDCRTPolyMatrix>,
        max_coefficient_bound: BigUint,
        randomness_seed: [u8; 32],
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
        let d = public_matrix.row_size();
        let rows = public_matrix.col_size();
        let columns = target.col_size();
        let target_global_column_start = target.global_column_start();
        if rows == 0 || columns == 0 || target.row_size() != d {
            return Err(SmallMatrixError::ShapeMismatch);
        }
        if public_matrix.params != *params ||
            trapdoor.r.params != *params ||
            trapdoor.e.params != *params
        {
            return Err(SmallMatrixError::ParameterMismatch);
        }
        if public_matrix.params.gpu_ids() != params.gpu_ids() ||
            trapdoor.r.params.gpu_ids() != params.gpu_ids() ||
            trapdoor.e.params.gpu_ids() != params.gpu_ids()
        {
            return Err(SmallMatrixError::DeviceMismatch);
        }

        let budget = params.vram_budget_bytes();
        let magnitude_bytes = usize::try_from(max_coefficient_bound.bits().div_ceil(8))
            .map_err(|_| SmallMatrixError::WidthOverflow)?
            .max(1);
        let attempts = crate::env::gpu_preimage_max_tile_attempts()
            .map_err(|_| SmallMatrixError::InvalidConfig)?;
        let mut tile_columns = columns;
        let matrix_bytes = |matrix: &GpuDCRTPolyMatrix| {
            params
                .matrix_allocation_bytes(
                    params.crt_depth().saturating_sub(1),
                    matrix.row_size(),
                    matrix.col_size(),
                    true,
                )
                .map(|allocation| allocation.total_bytes)
                .map_err(|_| SmallMatrixError::DimensionOverflow)
        };
        let persistent_bytes = [
            public_matrix,
            &trapdoor.r,
            &trapdoor.e,
            &trapdoor.a_mat_coeff,
            &trapdoor.b_mat_coeff,
            &trapdoor.d_mat_coeff,
        ]
        .into_iter()
        .map(matrix_bytes)
        .try_fold(0usize, |sum, bytes| {
            sum.checked_add(bytes?).ok_or(SmallMatrixError::DimensionOverflow)
        })?;
        let limbs = params.crt_depth();
        let coefficient_words = usize::try_from(params.modulus().bits().div_ceil(64))
            .map_err(|_| SmallMatrixError::DimensionOverflow)?
            .max(1);
        let hard_cutoff_plan_bytes = limbs
            .checked_mul(limbs)
            .and_then(|entries| entries.checked_mul(std::mem::size_of::<u64>()))
            .and_then(|bytes| {
                coefficient_words
                    .checked_mul(3 * std::mem::size_of::<u64>())
                    .and_then(|words| bytes.checked_add(words))
            })
            .and_then(|bytes| {
                limbs
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|subset| bytes.checked_add(subset))
            })
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let sampler_event_bytes = 4usize
            .checked_add(
                4usize
                    .checked_mul(params.crt_depth())
                    .ok_or(SmallMatrixError::DimensionOverflow)?,
            )
            .and_then(|count| count.checked_mul(std::mem::size_of::<usize>()))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let mut destination = GpuSmallMatrix::new_empty_checked(
            params,
            rows,
            columns,
            max_coefficient_bound,
            magnitude_bytes,
            budget,
        )?;
        let mut selected_report = None;
        while tile_columns > 0 {
            let padded_columns = padded_tile_columns(tile_columns, d)?;
            let candidate = matrix_bytes_for(params, rows, tile_columns)?;
            let p1 = matrix_bytes_for(params, 2 * d, padded_columns)?;
            let p2 = matrix_bytes_for(params, trapdoor.r.col_size(), padded_columns)?;
            let tp2 = matrix_bytes_for(params, 2 * d, padded_columns)?;
            let perturbation = p1
                .checked_add(p2)
                .and_then(|value| value.checked_add(tp2))
                .ok_or(SmallMatrixError::DimensionOverflow)?;
            let residual = matrix_bytes_for(params, d, tile_columns)?;
            let z_hat = matrix_bytes_for(params, trapdoor.r.col_size(), tile_columns)?;
            let target_tile = matrix_bytes_for(params, d, tile_columns)?;
            let sampled_integer_bytes = (2usize)
                .checked_mul(d)
                .and_then(|value| value.checked_mul(padded_columns))
                .and_then(|value| value.checked_mul(params.ring_dimension() as usize))
                .and_then(|value| value.checked_mul(std::mem::size_of::<i64>()))
                .ok_or(SmallMatrixError::DimensionOverflow)?;
            // MatrixTrapdoor.cu uses stack-local buffers through m <= 8 and
            // allocates this workspace only for the large-kernel path.
            let sampled_workspace_bytes = if 2 * d > 8 {
                sampled_integer_bytes.checked_mul(2).ok_or(SmallMatrixError::DimensionOverflow)?
            } else {
                0
            };
            let candidate_workspace = candidate
                .checked_add(sampled_integer_bytes)
                .and_then(|value| value.checked_add(sampled_workspace_bytes))
                .ok_or(SmallMatrixError::DimensionOverflow)?;
            let check_scratch =
                residual.checked_add(z_hat).ok_or(SmallMatrixError::DimensionOverflow)?;
            let packed_staging = rows
                .checked_mul(tile_columns)
                .and_then(|value| value.checked_mul(params.ring_dimension() as usize))
                .and_then(|value| value.checked_mul(1 + magnitude_bytes))
                .ok_or(SmallMatrixError::DimensionOverflow)?;
            let report = destination.sampler_allocation_report(
                persistent_bytes
                    .checked_add(target_tile)
                    .ok_or(SmallMatrixError::DimensionOverflow)?,
                candidate_workspace,
                perturbation,
                check_scratch,
                hard_cutoff_plan_bytes,
                packed_staging,
                sampler_event_bytes,
                std::mem::size_of::<i32>(),
                std::mem::size_of::<i32>(),
            )?;
            if report.fits_budget(budget) {
                selected_report = Some(report);
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
        let report = selected_report.ok_or(SmallMatrixError::InvalidConfig)?;
        destination.prepare_preimage_hard_cutoff();
        tracing::debug!(
            persistent_bytes = report.persistent_bytes,
            compact_destination_bytes = report.compact_destination_bytes,
            candidate_bytes = report.candidate_bytes,
            perturbation_bytes = report.perturbation_bytes,
            check_scratch_bytes = report.check_scratch_bytes,
            hard_cutoff_plan_bytes = report.hard_cutoff_plan_bytes,
            packed_staging_bytes = report.packed_staging_bytes,
            sampler_event_bytes = report.sampler_event_bytes,
            device_acceptance_control_bytes = report.device_acceptance_control_bytes,
            pinned_host_acceptance_control_bytes = report.pinned_host_acceptance_control_bytes,
            sampler_peak_bytes = report.sampler_peak_bytes,
            budget_bytes = budget,
            tile_columns,
            "gpu preimage compact sampler residency"
        );
        for column_start in (0..columns).step_by(tile_columns) {
            let column_end = (column_start + tile_columns).min(columns);
            let column_count = column_end - column_start;
            let tile_target = target.load_columns(column_start, column_end);
            if tile_target.params != *params ||
                tile_target.params.gpu_ids() != params.gpu_ids() ||
                tile_target.row_size() != d ||
                tile_target.col_size() != column_count
            {
                return Err(SmallMatrixError::ParameterMismatch);
            }
            // Acceptance is independent for each logical global column. This
            // is essential: a rejected column must not resample neighboring
            // columns merely because they share a storage tile.
            for local_column in 0..column_count {
                let local_column_index = column_start
                    .checked_add(local_column)
                    .ok_or(SmallMatrixError::DimensionOverflow)?;
                let global_column = target_global_column_start
                    .checked_add(local_column_index)
                    .ok_or(SmallMatrixError::DimensionOverflow)?;
                let single_target = tile_target.slice_columns(local_column, local_column + 1);
                let mut accepted = false;
                for attempt in 0..attempts {
                    let column_seed =
                        preimage_seed(randomness_seed, b"candidate", global_column, attempt);
                    let candidate = self.preimage_matrix_seeded(
                        params,
                        trapdoor,
                        public_matrix,
                        &single_target,
                        column_seed.to_bytes(),
                    );
                    let candidate = candidate.into_coeff_domain();
                    accepted = destination.try_pack_preimage_hard_cutoff_tile(
                        &candidate,
                        0,
                        local_column_index,
                        rows,
                        1,
                    )?;
                    drop(candidate);
                    if accepted {
                        break;
                    }
                }
                if !accepted {
                    let column_end =
                        global_column.checked_add(1).ok_or(SmallMatrixError::DimensionOverflow)?;
                    return Err(SmallMatrixError::AttemptExhausted {
                        column_start: global_column,
                        column_end,
                        attempts,
                    });
                }
            }
        }
        Ok(destination)
    }
}

fn matrix_bytes_for(
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
) -> Result<usize, SmallMatrixError> {
    params
        .matrix_allocation_bytes(params.crt_depth().saturating_sub(1), rows, columns, true)
        .map(|allocation| allocation.total_bytes)
        .map_err(|_| SmallMatrixError::DimensionOverflow)
}

fn padded_tile_columns(columns: usize, d: usize) -> Result<usize, SmallMatrixError> {
    if d == 0 {
        return Err(SmallMatrixError::InvalidShape);
    }
    columns
        .checked_add(d - 1)
        .and_then(|value| value.checked_div(d))
        .and_then(|value| value.checked_mul(d))
        .ok_or(SmallMatrixError::DimensionOverflow)
}

impl GpuDCRTPolyTrapdoorSampler {
    fn preimage_matrix(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
    ) -> GpuDCRTPolyMatrix {
        self.preimage_matrix_seeded(
            params,
            trapdoor,
            public_matrix,
            target,
            random_gpu_rng_seed().to_bytes(),
        )
    }

    fn preimage_matrix_seeded(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        randomness_seed: [u8; 32],
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
            GpuRngSeed::from_bytes(randomness_seed),
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

        let mut out = GpuDCRTPolyMatrix::preimage_output_from_perturbation(p1, p2, target_cols);
        let assemble_start = Instant::now();
        let gauss_start = Instant::now();
        let z_hat_mat = perturbed_syndrome.gauss_samp_gq_arb_base(
            self.c,
            self.sigma,
            preimage_seed(randomness_seed, b"z", 0, 0),
        );
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
        target: &dyn PolyMatrixColumnSource<Self::M>,
        max_coefficient_bound: BigUint,
        randomness_seed: [u8; 32],
    ) -> Result<GpuSmallMatrix, SmallMatrixError> {
        self.preimage_small(
            params,
            trapdoor,
            public_matrix,
            target,
            max_coefficient_bound,
            randomness_seed,
        )
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
        let preimage_left = self.preimage_matrix(params, trapdoor, public_matrix, &t);
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
    randomness_seed: GpuRngSeed,
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
    let p2 = uniform_sampler.sample_uniform_with_seed(
        params,
        dk,
        padded_ncol,
        DistType::GaussDist { sigma: sigma_large, max_coefficient_bound: None },
        preimage_seed(randomness_seed.to_bytes(), b"p2", 0, 0),
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
        preimage_seed(randomness_seed.to_bytes(), b"p1", 0, 0),
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

    #[test]
    fn sampler_padding_rounds_partial_column_tiles_to_block_width() {
        assert_eq!(padded_tile_columns(1, 8).unwrap(), 8);
        assert_eq!(padded_tile_columns(7, 8).unwrap(), 8);
        assert_eq!(padded_tile_columns(8, 8).unwrap(), 8);
        assert_eq!(padded_tile_columns(9, 8).unwrap(), 16);
        assert!(matches!(padded_tile_columns(1, 0), Err(SmallMatrixError::InvalidShape)));
    }

    #[test]
    fn preimage_seed_is_partition_invariant_per_global_column_and_attempt() {
        let request = [0x5au8; 32];
        let from_unit_tiles = (0..4)
            .map(|column| preimage_seed(request, b"candidate", column, 0))
            .collect::<Vec<_>>();
        let from_wide_tiles = (0..2)
            .chain(2..4)
            .map(|column| preimage_seed(request, b"candidate", column, 0))
            .collect::<Vec<_>>();
        assert_eq!(from_unit_tiles, from_wide_tiles);
        assert_ne!(
            preimage_seed(request, b"candidate", 2, 0),
            preimage_seed(request, b"candidate", 2, 1)
        );
    }

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
            preimage_seed([0u8; 32], b"test", 0, 0),
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

        let preimage =
            trapdoor_sampler.preimage_matrix(&params, &trapdoor, &public_matrix, &target);
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
            let preimage =
                trapdoor_sampler.preimage_matrix(&params, &trapdoor, &public_matrix, &target);
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
            trapdoor_sampler.preimage_matrix(&params, &trapdoor, &public_matrix, &first_target);
        let second_preimage =
            trapdoor_sampler.preimage_matrix(&params, &trapdoor, &public_matrix, &second_target);

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

        let sampled = trapdoor_sampler.preimage_matrix(&params, &trapdoor, &public_matrix, &target);
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
            let preimage =
                trapdoor_sampler.preimage_matrix(&params, &trapdoor, &public_matrix, &target);
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
                trapdoor_sampler.preimage_matrix(&src_params, &trapdoor, &public_matrix, &target);
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
