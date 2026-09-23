use crate::{
    matrix::{
        PolyMatrix, PolyMatrixColumnSource, SmallMatrixError,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuMatrixOutputDescriptor, GpuSmallMatrix},
    },
    poly::{
        Poly, PolyParams,
        dcrt::gpu::{
            GpuDCRTPolyParams, GpuNativeEvent, GpuNativeGraphError, GpuNativeLaunchStream,
            GpuRngSeed,
        },
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler,
        bounds::default_preimage_cutoff,
        gpu::{GpuDCRTPolyUniformSampler, random_gpu_rng_seed, sample_gpu_matrix_with_seed},
    },
};
use digest::Digest;
use num_bigint::BigUint;
use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
    time::Instant,
};

const SPECTRAL_CONSTANT: f64 = 1.8;

/// Certified allocation evidence for the complete `trapdoor()` production
/// call. Every matrix component is sized through the native allocator query;
/// no generic matrix-byte or measured-peak fallback is used.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrapdoorAllocationEvidence {
    pub trapdoor_r_bytes: usize,
    pub trapdoor_e_bytes: usize,
    pub trapdoor_a_coeff_bytes: usize,
    pub trapdoor_b_coeff_bytes: usize,
    pub trapdoor_d_coeff_bytes: usize,
    pub public_a_bar_bytes: usize,
    pub gadget_bytes: usize,
    pub identity_bytes: usize,
    pub a0_assembly_bytes: usize,
    pub a1_assembly_bytes: usize,
    pub public_output_bytes: usize,
    /// Temporary matrix owners live during nested products and conversion.
    pub scratch_bytes: usize,
    pub control_bytes: usize,
    pub cache_bytes: usize,
    pub host_bytes: usize,
    pub pinned_host_bytes: usize,
    pub evidence_kind: PreimageAllocationEvidenceKind,
}

impl TrapdoorAllocationEvidence {
    pub fn trapdoor_bytes(&self) -> usize {
        self.trapdoor_r_bytes
            .saturating_add(self.trapdoor_e_bytes)
            .saturating_add(self.trapdoor_a_coeff_bytes)
            .saturating_add(self.trapdoor_b_coeff_bytes)
            .saturating_add(self.trapdoor_d_coeff_bytes)
    }

    pub fn assembly_bytes(&self) -> usize {
        self.public_a_bar_bytes
            .saturating_add(self.gadget_bytes)
            .saturating_add(self.identity_bytes)
            .saturating_add(self.a0_assembly_bytes)
            .saturating_add(self.a1_assembly_bytes)
    }

    pub fn total_device_bytes(&self) -> usize {
        self.trapdoor_bytes()
            .saturating_add(self.public_output_bytes)
            .saturating_add(self.scratch_bytes)
            .saturating_add(self.assembly_bytes())
            .saturating_add(self.cache_bytes)
            .saturating_add(self.control_bytes)
    }
}

fn trapdoor_matrix_bytes(
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    is_ntt: bool,
) -> Result<usize, SmallMatrixError> {
    params
        .matrix_allocation_bytes(params.crt_depth().saturating_sub(1), rows, columns, is_ntt)
        .map(|allocation| allocation.total_bytes)
        .map_err(|_| SmallMatrixError::DimensionOverflow)
}

#[cfg(test)]
fn gpu_params_from_cpu(params: &crate::poly::dcrt::params::DCRTPolyParams) -> GpuDCRTPolyParams {
    let (moduli, _, _) = params.to_crt();
    GpuDCRTPolyParams::new(
        params.ring_dimension(),
        moduli,
        params.base_bits(),
        Some(params.dropped_moduli()),
    )
}

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

fn coeff_cached_matrix(src: GpuDCRTPolyMatrix) -> GpuDCRTPolyMatrix {
    src.into_coeff_domain()
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
    re: GpuDCRTPolyMatrix,
    p1_covariance_cache: Arc<Mutex<Option<GpuP1CovarianceCacheEntry>>>,
}

/// Immutable trapdoor destination layout.  Public and secret/cached
/// components are described independently so capture preparation never needs
/// to serialize a secret to host bytes and restore it later.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuTrapdoorOutputDescriptor {
    pub public: GpuMatrixOutputDescriptor,
    pub secret_r: GpuMatrixOutputDescriptor,
    pub secret_e: GpuMatrixOutputDescriptor,
    pub secret_a: GpuMatrixOutputDescriptor,
    pub secret_b: GpuMatrixOutputDescriptor,
    pub secret_d: GpuMatrixOutputDescriptor,
    pub secret_re: GpuMatrixOutputDescriptor,
}

impl PartialEq for GpuDCRTTrapdoor {
    fn eq(&self, other: &Self) -> bool {
        self.r == other.r &&
            self.e == other.e &&
            self.a_mat_coeff == other.a_mat_coeff &&
            self.b_mat_coeff == other.b_mat_coeff &&
            self.d_mat_coeff == other.d_mat_coeff &&
            self.re == other.re
    }
}

impl Eq for GpuDCRTTrapdoor {}

impl GpuDCRTTrapdoor {
    /// All resident secret components, including the coefficient covariance
    /// products consumed by preimage sampling, in their stable capture order.
    pub fn capture_components(&self) -> [&GpuDCRTPolyMatrix; 6] {
        [&self.r, &self.e, &self.a_mat_coeff, &self.b_mat_coeff, &self.d_mat_coeff, &self.re]
    }

    pub fn capture_components_mut(&mut self) -> [&mut GpuDCRTPolyMatrix; 6] {
        [
            &mut self.r,
            &mut self.e,
            &mut self.a_mat_coeff,
            &mut self.b_mat_coeff,
            &mut self.d_mat_coeff,
            &mut self.re,
        ]
    }

    /// Return a handle retaining every native matrix allocation in this
    /// trapdoor. This is intentionally separate from [`Clone`], whose
    /// established semantics remain deep device copies. The covariance cache
    /// is shared as well, so a retained handle keeps cached native resources
    /// and the matrix CUDA contexts alive for the same lifetime.
    pub fn clone_shallow(&self) -> Self {
        Self {
            r: self.r.clone_shallow(),
            e: self.e.clone_shallow(),
            a_mat_coeff: self.a_mat_coeff.clone_shallow(),
            b_mat_coeff: self.b_mat_coeff.clone_shallow(),
            d_mat_coeff: self.d_mat_coeff.clone_shallow(),
            re: self.re.clone_shallow(),
            p1_covariance_cache: Arc::clone(&self.p1_covariance_cache),
        }
    }

    pub fn output_descriptor_with_public(
        &self,
        public: &GpuDCRTPolyMatrix,
    ) -> Result<GpuTrapdoorOutputDescriptor, String> {
        Ok(GpuTrapdoorOutputDescriptor {
            public: public.output_descriptor()?,
            secret_r: self.r.output_descriptor()?,
            secret_e: self.e.output_descriptor()?,
            secret_a: self.a_mat_coeff.output_descriptor()?,
            secret_b: self.b_mat_coeff.output_descriptor()?,
            secret_d: self.d_mat_coeff.output_descriptor()?,
            secret_re: self.re.output_descriptor()?,
        })
    }

    pub fn output_descriptor(&self) -> Result<GpuTrapdoorOutputDescriptor, String> {
        self.output_descriptor_with_public(&self.r)
    }

    /// Build the complete fresh-owner contract for a regular sampled
    /// trapdoor without sampling an exemplar.  The public matrix and `r/e`
    /// are native evaluation-domain owners; the cached coefficient matrices
    /// are coefficient-domain owners derived from them.
    pub fn output_descriptor_for_shape(
        params: &GpuDCRTPolyParams,
        size: usize,
        public_columns: usize,
    ) -> Result<GpuTrapdoorOutputDescriptor, String> {
        let level = params
            .crt_depth()
            .checked_sub(1)
            .ok_or_else(|| "trapdoor output has empty CRT depth".to_owned())?;
        let public =
            GpuMatrixOutputDescriptor::for_shape(params, size, public_columns, level, true)?;
        let secret_r = GpuMatrixOutputDescriptor::for_shape(
            params,
            size,
            size.checked_mul(params.modulus_digits())
                .ok_or_else(|| "trapdoor shape overflow".to_owned())?,
            level,
            true,
        )?;
        let secret_e = secret_r.clone();
        let secret_a = GpuMatrixOutputDescriptor::for_shape(params, size, size, level, false)?;
        let secret_b = secret_a.clone();
        let secret_d = secret_a.clone();
        let secret_re = GpuMatrixOutputDescriptor::for_shape(
            params,
            size.checked_mul(2).ok_or_else(|| "trapdoor shape overflow".to_owned())?,
            size.checked_mul(params.modulus_digits())
                .ok_or_else(|| "trapdoor shape overflow".to_owned())?,
            level,
            true,
        )?;
        Ok(GpuTrapdoorOutputDescriptor {
            public,
            secret_r,
            secret_e,
            secret_a,
            secret_b,
            secret_d,
            secret_re,
        })
    }

    /// Allocate a fresh public/secret trapdoor pair exactly from its frozen
    /// output contract.  No sampled exemplar or host payload is retained.
    #[doc(hidden)]
    pub fn allocate_from_descriptor(
        descriptor: &GpuTrapdoorOutputDescriptor,
    ) -> (GpuDCRTPolyMatrix, Self) {
        let secret = Self {
            r: descriptor.secret_r.allocate(),
            e: descriptor.secret_e.allocate(),
            a_mat_coeff: descriptor.secret_a.allocate(),
            b_mat_coeff: descriptor.secret_b.allocate(),
            d_mat_coeff: descriptor.secret_d.allocate(),
            re: descriptor.secret_re.allocate(),
            p1_covariance_cache: Arc::new(Mutex::new(None)),
        };
        (descriptor.public.allocate(), secret)
    }

    /// Copy every public, secret, and cached component directly into a
    /// preallocated destination owner. This is the capture-safe replacement
    /// for host compact serialization/restore.
    #[doc(hidden)]
    pub fn copy_into(&self, destination: &mut Self) -> Result<(), GpuNativeGraphError> {
        self.r.copy_into(&mut destination.r)?;
        self.e.copy_into(&mut destination.e)?;
        self.a_mat_coeff.copy_into(&mut destination.a_mat_coeff)?;
        self.b_mat_coeff.copy_into(&mut destination.b_mat_coeff)?;
        self.d_mat_coeff.copy_into(&mut destination.d_mat_coeff)?;
        self.re.copy_into(&mut destination.re)?;
        Ok(())
    }

    /// Copy the separately-owned public matrix and all secret components in
    /// one destination-taking operation.
    #[doc(hidden)]
    pub fn copy_public_and_secret_into(
        &self,
        public: &GpuDCRTPolyMatrix,
        destination_public: &mut GpuDCRTPolyMatrix,
        destination_secret: &mut Self,
    ) -> Result<(), GpuNativeGraphError> {
        public.copy_into(destination_public)?;
        self.copy_into(destination_secret)
    }

    /// Allocate every trapdoor component in `params` and enqueue direct
    /// device copies for the complete secret owner.  The covariance cache is
    /// deliberately not copied: it is a destination-local derived cache and
    /// is rebuilt only when a preimage job actually needs it.
    #[doc(hidden)]
    pub fn copy_to_params_direct(
        &self,
        params: &GpuDCRTPolyParams,
    ) -> Result<Self, GpuNativeGraphError> {
        let r = self.r.copy_to_params_direct(params)?;
        let e = self.e.copy_to_params_direct(params)?;
        let a_mat_coeff = self.a_mat_coeff.copy_to_params_direct(params)?;
        let b_mat_coeff = self.b_mat_coeff.copy_to_params_direct(params)?;
        let d_mat_coeff = self.d_mat_coeff.copy_to_params_direct(params)?;
        let re = self.re.copy_to_params_direct(params)?;
        Ok(Self {
            r,
            e,
            a_mat_coeff,
            b_mat_coeff,
            d_mat_coeff,
            re,
            p1_covariance_cache: Arc::new(Mutex::new(None)),
        })
    }

    /// Waits for every matrix required to consume this trapdoor.
    pub fn wait_until_ready(&self) {
        self.r.wait_until_ready();
        self.e.wait_until_ready();
        self.a_mat_coeff.wait_until_ready();
        self.b_mat_coeff.wait_until_ready();
        self.d_mat_coeff.wait_until_ready();
        self.re.wait_until_ready();
    }

    /// Queue readiness dependencies for every secret trapdoor component on a
    /// compiled graph stream. This keeps the complete secret owner contract
    /// together; callers cannot accidentally protect only `r`/`e` while the
    /// cached coefficient components are still in flight.
    #[doc(hidden)]
    pub fn wait_compiled_inputs(
        &self,
        consumer_device: i32,
        launch_stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        for component in self.capture_components() {
            component.wait_compiled_inputs(consumer_device, launch_stream, true)?;
        }
        Ok(())
    }

    /// Attach the compiled submission completion to every secret component's
    /// release stream as a read-only consumer. The writer events remain
    /// independent so concurrent graph readers are not serialized.
    #[doc(hidden)]
    pub fn protect_compiled_submission(
        &self,
        consumer_device: i32,
        launch_stream: &GpuNativeLaunchStream,
        completion: &GpuNativeEvent,
    ) -> Result<(), GpuNativeGraphError> {
        for component in self.capture_components() {
            component.protect_compiled_submission(
                consumer_device,
                launch_stream,
                completion,
                true,
            )?;
        }
        Ok(())
    }

    /// Exact retained owner bytes after trapdoor construction.  This is
    /// intentionally separate from `trapdoor_allocation_evidence`, whose
    /// certified envelope includes temporary construction products and must
    /// not be charged as resident setup state.
    pub fn retained_allocation_bytes(&self) -> Result<usize, String> {
        self.capture_components().into_iter().try_fold(0usize, |total, matrix| {
            total
                .checked_add(matrix.allocation_bytes()?.total_bytes)
                .ok_or_else(|| "trapdoor retained allocation overflow".into())
        })
    }

    pub fn new(params: &GpuDCRTPolyParams, size: usize, sigma: f64) -> Self {
        assert_eq!(
            params.dropped_moduli(),
            0,
            "exact trapdoor sampling requires dropped_moduli = 0"
        );
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let log_base_q = params.modulus_digits();
        let dist = DistType::GaussDist { sigma, max_coefficient_bound: None };
        let r = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist.clone());
        let e = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        let a_mat_coeff = coeff_cached_matrix(&r * &r.transpose());
        let b_mat_coeff = coeff_cached_matrix(&r * &e.transpose());
        let d_mat_coeff = coeff_cached_matrix(&e * &e.transpose());
        let re = r.concat_rows(&[&e]);
        let p1_covariance_cache = Arc::new(Mutex::new(None));
        Self { r, e, a_mat_coeff, b_mat_coeff, d_mat_coeff, re, p1_covariance_cache }
    }

    /// Compatibility artifact export for host persistence. Capture and
    /// compiled execution must use `copy_into`/`output_descriptor` so secret
    /// components never take a host serialize/restore round trip.
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mats = self.capture_components();
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

    /// Compatibility artifact import. This is intentionally not part of the
    /// capture path; use preallocated component destinations there.
    pub fn from_compact_bytes(params: &GpuDCRTPolyParams, bytes: &[u8]) -> Option<Self> {
        let mut offset = 0usize;
        let next = |buf: &[u8], offset: &mut usize| -> Option<Vec<u8>> {
            let header_end = offset.checked_add(8)?;
            if header_end > buf.len() {
                return None;
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&buf[*offset..header_end]);
            let len = usize::try_from(u64::from_le_bytes(len_bytes)).ok()?;
            *offset = header_end;
            let payload_end = offset.checked_add(len)?;
            if payload_end > buf.len() {
                return None;
            }
            let out = buf[*offset..payload_end].to_vec();
            *offset = payload_end;
            Some(out)
        };
        let r_bytes = next(bytes, &mut offset)?;
        let e_bytes = next(bytes, &mut offset)?;
        let a_bytes = next(bytes, &mut offset)?;
        let b_bytes = next(bytes, &mut offset)?;
        let d_bytes = next(bytes, &mut offset)?;
        let re_bytes = next(bytes, &mut offset)?;
        if offset != bytes.len() {
            return None;
        }

        let r = GpuDCRTPolyMatrix::from_compact_bytes(params, &r_bytes);
        let e = GpuDCRTPolyMatrix::from_compact_bytes(params, &e_bytes);
        let a_mat_coeff = GpuDCRTPolyMatrix::from_compact_bytes(params, &a_bytes);
        let b_mat_coeff = GpuDCRTPolyMatrix::from_compact_bytes(params, &b_bytes);
        let d_mat_coeff = GpuDCRTPolyMatrix::from_compact_bytes(params, &d_bytes);
        let re = GpuDCRTPolyMatrix::from_compact_bytes(params, &re_bytes);
        let p1_covariance_cache = Arc::new(Mutex::new(None));
        Some(Self { r, e, a_mat_coeff, b_mat_coeff, d_mat_coeff, re, p1_covariance_cache })
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

/// Build the P1 covariance cache for a trapdoor that is about to enter a
/// fixed preimage job.  The cache is owned by the trapdoor, so keeping that
/// owner alive is what makes a subsequent warm call reuse the exact native
/// allocation.  This method intentionally does not sample a preimage or
/// allocate any tile workspace.
fn build_p1_covariance_cache(params: &GpuDCRTPolyParams, trapdoor: &GpuDCRTTrapdoor, sigma: f64) {
    let d = trapdoor.r.row_size();
    let (c, s, dgg_stddev) = p1_covariance_parameters(params, d, sigma);
    let _ = get_or_create_p1_covariance_cache(trapdoor, c, s, dgg_stddev);
}

fn cached_p1_covariance_cache(
    params: &GpuDCRTPolyParams,
    trapdoor: &GpuDCRTTrapdoor,
    sigma: f64,
) -> Option<Arc<crate::matrix::gpu_dcrt_poly::GpuP1CovarianceCache>> {
    let d = trapdoor.r.row_size();
    let (c, s, dgg_stddev) = p1_covariance_parameters(params, d, sigma);
    let guard = trapdoor.p1_covariance_cache.lock().expect("p1 cache mutex poisoned");
    guard.as_ref().and_then(|entry| {
        (entry.c == c && entry.s == s && entry.dgg_stddev == dgg_stddev)
            .then(|| entry.cache.clone())
    })
}

#[derive(Debug, Clone)]
pub struct GpuDCRTPolyTrapdoorSampler {
    sigma: f64,
    base: u32,
    c: f64,
}

fn preimage_seed(base: [u8; 32], stage: &[u8], column_start: usize, attempt: usize) -> GpuRngSeed {
    GpuRngSeed::from_bytes(preimage_seed_bytes(base, stage, column_start, attempt))
}

/// Return the canonical seed bytes for one logical preimage attempt.
///
/// This is deliberately kept as the single Rust-side definition of the seed
/// ABI.  The CUDA retry adapter must consume these bytes without adding a
/// device/frame/physical-GPU coordinate: those coordinates are scheduling
/// details and would change transcript identity.  In particular, callers may
/// use this helper to build the fixed device control block before launching a
/// retry graph.
pub fn preimage_seed_bytes(
    base: [u8; 32],
    stage: &[u8],
    column_start: usize,
    attempt: usize,
) -> [u8; 32] {
    let mut hasher = keccak_asm::Keccak256::new();
    hasher.update(b"mxx-preimage-sampler/v1");
    hasher.update(base);
    hasher.update((stage.len() as u64).to_le_bytes());
    hasher.update(stage);
    hasher.update((column_start as u64).to_le_bytes());
    hasher.update((attempt as u64).to_le_bytes());
    hasher.finalize().into()
}

/// Canonical retry seed ABI for a captured conditional body.  The fixed
/// control block is part of the logical transcript: physical device/shard
/// coordinates are deliberately excluded, while nonce, logical instance,
/// global column and attempt are all consumed so a body replay can never
/// regenerate the same candidate for a different retry.
pub fn preimage_seed_bytes_with_control(
    control: &PreimageLaunchControl,
    stage: &[u8],
    attempt: u32,
) -> [u8; 32] {
    let mut hasher = keccak_asm::Keccak256::new();
    hasher.update(b"mxx-preimage-sampler/v2");
    hasher.update(control.execution_nonce);
    hasher.update(control.logical_instance.to_le_bytes());
    hasher.update(control.global_column_start.to_le_bytes());
    hasher.update((stage.len() as u64).to_le_bytes());
    hasher.update(stage);
    hasher.update(attempt.to_le_bytes());
    hasher.finalize().into()
}

impl PreimageLaunchControl {
    /// Derive a unique logical attempt seed without introducing physical
    /// placement into the transcript.
    pub fn attempt_seed(&self, stage: &[u8], attempt: u32) -> GpuRngSeed {
        GpuRngSeed::from_bytes(preimage_seed_bytes_with_control(self, stage, attempt))
    }
}

/// Device-visible result of one fixed retry region.
///
/// The record is intentionally small and has a stable C layout.  It is reset
/// for every tile invocation and copied to the host exactly once after the
/// retry region, where `error_code != 0` prevents dependent work from being
/// submitted.  `accepted` is a count (rather than a bool) so a malformed
/// multi-accept body can fail closed in the host gate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreimageStatus {
    pub attempts: u32,
    pub accepted: u32,
    pub error_code: u32,
    pub reserved: u32,
}

impl PreimageStatus {
    pub const SUCCESS: u32 = 0;
    pub const EXHAUSTED: u32 = 1;
    pub const INVALID_ATTEMPT: u32 = 2;

    pub const fn reset() -> Self {
        Self { attempts: 0, accepted: 0, error_code: Self::SUCCESS, reserved: 0 }
    }

    pub const fn succeeded(self) -> bool {
        self.error_code == Self::SUCCESS && self.accepted == 1
    }

    pub const fn exhausted(self) -> bool {
        self.error_code == Self::EXHAUSTED
    }
}

/// Stable device control block for a retry invocation.
///
/// Pointer roots used by the attempt adapter live in the fixed-layout device
/// binding table referenced by `binding_table`; they are not captured as
/// direct per-run kernel arguments.  The launch stream updates this block and
/// the status record before entering the conditional body.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreimageLaunchControl {
    pub execution_nonce: [u8; 32],
    pub logical_instance: u64,
    pub global_column_start: u64,
    pub max_attempts: u32,
    pub attempt: u32,
    pub binding_table: u64,
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

#[cfg(test)]
enum RetryFailure {
    Error,
    Exhausted,
}

/// Whether the first use of a trapdoor covariance cache is part of the
/// measured operation.  Cold and warm resource envelopes are deliberately
/// different execution classes: cold setup owns a transient workspace while
/// warm sampling only sees the retained cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreimageCacheState {
    Cold,
    Warm,
}

/// The representation and ordered CRT basis used by a fixed preimage call.
/// This is part of the evidence identity; changing either the compact format
/// or the active modulus order invalidates the evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreimageFormatContext {
    pub active_moduli: Vec<u64>,
    pub crt_level: usize,
    pub coefficient_magnitude_bytes: usize,
    pub compact_output: bool,
}

/// Identity of the retained P1 covariance cache.  The owner token is tied to
/// one trapdoor object, while the numeric parameters invalidate evidence when
/// the sampler configuration changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreimageCacheIdentity {
    pub owner_token: u64,
    pub c: f64,
    pub smoothing: f64,
    pub dgg_stddev: f64,
}

impl PreimageCacheIdentity {
    /// Convert native cache ownership and sampler parameters into an opaque
    /// public identity.  The digest contains no trapdoor/material bytes; it
    /// only binds the cache owner token, floating-point sampler parameters,
    /// and the ordered native basis supplied by the caller.
    pub fn opaque_digest(&self, sampler_parameters: &[u64], ordered_basis: &[u64]) -> [u8; 32] {
        let mut hasher = keccak_asm::Keccak256::new();
        hasher.update(b"mxx-preimage-cache-identity-v1");
        hasher.update(self.owner_token.to_le_bytes());
        hasher.update(self.c.to_bits().to_le_bytes());
        hasher.update(self.smoothing.to_bits().to_le_bytes());
        hasher.update(self.dgg_stddev.to_bits().to_le_bytes());
        hasher.update((sampler_parameters.len() as u64).to_le_bytes());
        for value in sampler_parameters {
            hasher.update(value.to_le_bytes());
        }
        hasher.update((ordered_basis.len() as u64).to_le_bytes());
        for value in ordered_basis {
            hasher.update(value.to_le_bytes());
        }
        hasher.finalize().into()
    }
}

/// Context carried with every allocation envelope.  In particular, a result
/// for width 8 or retry cap 64 is never silently reused for another fixed
/// production call.
#[derive(Clone, Debug, PartialEq)]
pub struct PreimageEvidenceContext {
    pub state: PreimageCacheState,
    pub tile_columns: usize,
    pub max_attempts: usize,
    pub max_coefficient_bound: BigUint,
    pub hard_cutoff_plan_bytes: usize,
    pub format: PreimageFormatContext,
    pub source_device_ids: Vec<i32>,
    pub destination_device_ids: Vec<i32>,
    pub cache_identity: PreimageCacheIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreimageAllocationEvidenceKind {
    /// Every component is returned by an exact native size query or checked
    /// fixed-call allocation rule.
    Exact,
    /// The fixed-call topology is certified, but one or more components are
    /// conservative bounds rather than byte-for-byte allocator observations.
    Certified,
}

/// Exact/certified allocation envelope for one *fixed* production call.
/// Every component is a live allocation class, not an empirical width slope.
/// The aggregate peaks are checked sums of these components for the declared
/// lifetime and state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreimageAllocationEnvelope {
    pub public_matrix_bytes: usize,
    pub trapdoor_bytes: usize,
    pub resident_target_bytes: usize,
    pub retained_covariance_cache_bytes: usize,
    pub compact_output_bytes: usize,
    pub candidate_workspace_bytes: usize,
    pub perturbation_workspace_bytes: usize,
    pub scratch_bytes: usize,
    pub source_device_staging_bytes: usize,
    pub destination_device_staging_bytes: usize,
    pub host_staging_bytes: usize,
    pub device_control_bytes: usize,
    pub pinned_host_control_bytes: usize,
    pub sampler_event_bytes: usize,
    pub cold_transient_workspace_bytes: usize,
    pub sampler_peak_bytes: usize,
    pub cold_sampler_peak_bytes: usize,
    pub evidence_kind: PreimageAllocationEvidenceKind,
}

impl PreimageAllocationEnvelope {
    fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize, SmallMatrixError> {
        values.into_iter().try_fold(0usize, |sum, value| {
            sum.checked_add(value).ok_or(SmallMatrixError::DimensionOverflow)
        })
    }

    pub fn fits_budget(&self, budget_bytes: usize) -> bool {
        self.cold_sampler_peak_bytes <= budget_bytes
    }
}

/// Complete allocation evidence for one fixed width and cache state.
#[derive(Clone, Debug, PartialEq)]
pub struct PreimageAllocationEvidence {
    pub context: PreimageEvidenceContext,
    pub envelope: PreimageAllocationEnvelope,
}

/// The resource choices made by the fleet planner for one preimage job.
///
/// A fixed sampler never changes either value in response to an allocation
/// failure.  In particular, `tile_columns` is not a hint for another search;
/// it is the upper bound of every tile submitted by the job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedPreimageConfig {
    pub tile_columns: NonZeroUsize,
    pub max_attempts: NonZeroUsize,
}

impl FixedPreimageConfig {
    pub fn new(tile_columns: usize, max_attempts: usize) -> Option<Self> {
        Some(Self {
            tile_columns: NonZeroUsize::new(tile_columns)?,
            max_attempts: NonZeroUsize::new(max_attempts)?,
        })
    }
}

/// Pure accounting for one fixed preimage job.  This query performs no GPU
/// allocation or submission and deliberately includes the full trapdoor,
/// public matrix, resident target, compact destination, cutoff, packing and
/// control/event owners.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreimageFootprint {
    pub persistent_bytes: usize,
    pub compact_output_bytes: usize,
    pub scratch_bytes: usize,
    pub device_control_bytes: usize,
    pub pinned_control_bytes: usize,
    pub hard_cutoff_plan_bytes: usize,
    pub packed_staging_bytes: usize,
    pub sampler_event_bytes: usize,
    /// Workspace allocated only while the cold covariance cache is built.
    /// It is released after the precompute event and must not inflate the
    /// warm steady-state sampler report.
    pub cold_transient_workspace_bytes: usize,
    pub sampler_peak_bytes: usize,
    /// Peak required while creating the cache, including the retained cache
    /// and the cold transient workspace.
    pub cold_sampler_peak_bytes: usize,
    pub tile_columns: usize,
    pub max_attempts: usize,
}

struct TileWorkspaceBytes {
    candidate: usize,
    perturbation: usize,
    residual: usize,
    z_hat: usize,
    target_tile: usize,
    packed_staging: usize,
}

impl PreimageFootprint {
    pub fn fits_budget(&self, budget_bytes: usize) -> bool {
        self.cold_sampler_peak_bytes <= budget_bytes
    }
}

#[cfg(test)]
fn bounded_retry<T, F>(attempts: usize, mut attempt: F) -> Result<T, RetryFailure>
where
    F: FnMut() -> Result<Option<T>, SmallMatrixError>,
{
    for _ in 0..attempts {
        if let Some(value) = attempt().map_err(|_| RetryFailure::Error)? {
            return Ok(value);
        }
    }
    Err(RetryFailure::Exhausted)
}

impl GpuDCRTPolyTrapdoorSampler {
    /// Enqueue the cached p1 candidate kernel on a conditional-body stream.
    /// All destination and scratch owners must have been allocated before
    /// capture; the device seed is re-read on each body iteration.
    pub fn sample_p1_cached_into_body(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        tp2: &GpuDCRTPolyMatrix,
        sampled_out: *mut std::ffi::c_void,
        workspace: *mut std::ffi::c_void,
        workspace_bytes: usize,
        device_seed: *const std::ffi::c_void,
        out: &mut GpuDCRTPolyMatrix,
        stream: &GpuNativeLaunchStream,
        tp2_binding_index: u32,
        seed_binding_index: u32,
        sampled_binding_index: u32,
        workspace_binding_index: u32,
    ) {
        let d = trapdoor.r.row_size();
        let s = preimage_smoothing_parameter(
            self.base,
            self.sigma,
            d,
            params.ring_dimension() as usize,
            params.modulus_digits(),
        );
        let dgg_stddev = (s * s - self.c * self.c).sqrt();
        let cache = get_or_create_p1_covariance_cache(trapdoor, self.c, s, dgg_stddev);
        GpuDCRTPolyMatrix::sample_p1_full_cached_into(
            cache.as_ref(),
            tp2,
            sampled_out,
            workspace,
            workspace_bytes,
            device_seed,
            out,
            stream,
            tp2_binding_index,
            seed_binding_index,
            sampled_binding_index,
            workspace_binding_index,
        );
    }

    /// Enqueue the complete Gaussian digit generation, scatter, and NTT
    /// sequence on the retry-body stream using fixed caller-owned scratch.
    pub fn sample_gauss_into_body(
        &self,
        source: &GpuDCRTPolyMatrix,
        output: &mut GpuDCRTPolyMatrix,
        sampled_digits: *mut std::ffi::c_void,
        sampled_capacity_bytes: usize,
        device_seed: *const std::ffi::c_void,
        dst_bases_device: *mut std::ffi::c_void,
        dst_strides_device: *const std::ffi::c_void,
        dst_coeff_bytes_device: *const std::ffi::c_void,
        dst_moduli_device: *const std::ffi::c_void,
        stream: &GpuNativeLaunchStream,
        source_binding_index: u32,
        out_bases_binding_index: u32,
        out_strides_binding_index: u32,
        out_coeff_bytes_binding_index: u32,
        out_moduli_binding_index: u32,
        seed_binding_index: u32,
        sampled_binding_index: u32,
    ) {
        GpuDCRTPolyMatrix::gauss_samp_gq_arb_base_into(
            source,
            output,
            sampled_digits,
            sampled_capacity_bytes,
            device_seed,
            dst_bases_device,
            dst_strides_device,
            dst_coeff_bytes_device,
            dst_moduli_device,
            self.c,
            stream,
            source_binding_index,
            out_bases_binding_index,
            out_strides_binding_index,
            out_coeff_bytes_binding_index,
            out_moduli_binding_index,
            seed_binding_index,
            sampled_binding_index,
        );
    }

    /// Construct the production covariance cache without running the sampler.
    /// Warmup uses this as the timed cold/setup operation, then retains the
    /// trapdoor owner for the corresponding local-job measurement.
    pub fn build_preimage_covariance_cache(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
    ) {
        build_p1_covariance_cache(params, trapdoor, self.sigma);
        // Cache construction is asynchronous in CUDA.  The setup timing
        // boundary must include completion of the native covariance build.
        crate::poly::dcrt::gpu::gpu_device_sync();
    }

    /// Return the already-retained production covariance cache.  Unlike the
    /// normal sampler entry point this never creates a cache on a miss; this
    /// makes a warm measurement fail closed instead of silently including cold
    /// setup work in its local-job timing.
    pub fn has_retained_preimage_covariance_cache(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
    ) -> bool {
        cached_p1_covariance_cache(params, trapdoor, self.sigma).is_some()
    }

    fn validate_preimage_inputs(
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &dyn PolyMatrixColumnSource<GpuDCRTPolyMatrix>,
        max_coefficient_bound: &BigUint,
    ) -> Result<(usize, usize, usize, usize), SmallMatrixError> {
        let d = public_matrix.row_size();
        let k = public_matrix.col_size();
        let columns = target.col_size();
        if target.row_size() != d || k == 0 || columns == 0 {
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
        let magnitude_bytes = usize::try_from(max_coefficient_bound.bits().div_ceil(8))
            .map_err(|_| SmallMatrixError::WidthOverflow)?
            .max(1);
        Ok((d, k, columns, magnitude_bytes))
    }

    fn covariance_cache_footprint(
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
    ) -> Result<(usize, usize), SmallMatrixError> {
        let n = params.ring_dimension() as usize;
        let cache_rows = trapdoor
            .a_mat_coeff
            .row_size()
            .checked_mul(2)
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let factor = n.checked_mul(cache_rows).ok_or(SmallMatrixError::DimensionOverflow)?;
        let update = factor.checked_mul(cache_rows).ok_or(SmallMatrixError::DimensionOverflow)?;
        let factor_bytes = factor
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let update_bytes = update
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let retained = factor_bytes
            .checked_add(update_bytes)
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<usize>()))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        // MatrixTrapdoor.cu allocates a covariance workspace with the same
        // size as update_coeff while sqrt_var and update_coeff are already
        // live. It is freed after the precompute event, so this is cold-only.
        Ok((retained, update_bytes))
    }

    fn cache_identity(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        d: usize,
    ) -> PreimageCacheIdentity {
        let (c, s, dgg_stddev) = p1_covariance_parameters(params, d, self.sigma);
        PreimageCacheIdentity {
            // The Arc is owned by the trapdoor and therefore changes when a
            // compactly decoded/new trapdoor is used.  This is intentionally
            // an opaque identity rather than a performance-cache key based on
            // a debug representation of secret material.
            owner_token: Arc::as_ptr(&trapdoor.p1_covariance_cache) as usize as u64,
            c,
            smoothing: s,
            dgg_stddev,
        }
    }

    fn tile_workspace_bytes(
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        d: usize,
        k: usize,
        tile_columns: usize,
        magnitude_bytes: usize,
    ) -> Result<TileWorkspaceBytes, SmallMatrixError> {
        let candidate = dcrt_matrix_bytes(params, k, tile_columns)?;
        let perturbation = dcrt_matrix_bytes(params, 2 * d, tile_columns)?
            .checked_add(dcrt_matrix_bytes(params, trapdoor.r.col_size(), tile_columns)?)
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let residual = dcrt_matrix_bytes(params, d, tile_columns)?;
        let z_hat = dcrt_matrix_bytes(params, trapdoor.r.col_size(), tile_columns)?;
        let target_tile = dcrt_matrix_bytes(params, d, tile_columns)?;
        let packed_staging = k
            .checked_mul(tile_columns)
            .and_then(|value| value.checked_mul(params.ring_dimension() as usize))
            .and_then(|value| value.checked_mul(1 + magnitude_bytes))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        Ok(TileWorkspaceBytes {
            candidate,
            perturbation,
            residual,
            z_hat,
            target_tile,
            packed_staging,
        })
    }

    fn allocation_evidence(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &dyn PolyMatrixColumnSource<GpuDCRTPolyMatrix>,
        max_coefficient_bound: &BigUint,
        config: FixedPreimageConfig,
        state: PreimageCacheState,
    ) -> Result<PreimageAllocationEvidence, SmallMatrixError> {
        let (d, k, columns, magnitude_bytes) = Self::validate_preimage_inputs(
            params,
            trapdoor,
            public_matrix,
            target,
            max_coefficient_bound,
        )?;
        let public_matrix_bytes =
            dcrt_matrix_bytes(params, public_matrix.row_size(), public_matrix.col_size())?;
        let trapdoor_bytes = [
            &trapdoor.r,
            &trapdoor.e,
            &trapdoor.a_mat_coeff,
            &trapdoor.b_mat_coeff,
            &trapdoor.d_mat_coeff,
        ]
        .into_iter()
        .map(|matrix| dcrt_matrix_bytes(params, matrix.row_size(), matrix.col_size()))
        .try_fold(0usize, |sum, bytes| {
            sum.checked_add(bytes?).ok_or(SmallMatrixError::DimensionOverflow)
        })?;
        let resident_target_bytes = target.resident_matrix().map_or(Ok(0), |matrix| {
            dcrt_matrix_bytes(params, matrix.row_size(), matrix.col_size())
        })?;
        let (retained_covariance_cache_bytes, cold_transient_workspace_bytes) =
            Self::covariance_cache_footprint(params, trapdoor)?;
        let compact_output_bytes = k
            .checked_mul(columns)
            .and_then(|value| value.checked_mul(params.ring_dimension() as usize))
            .and_then(|value| value.checked_mul(1 + magnitude_bytes))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
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
        let device_control_bytes = std::mem::size_of::<PreimageStatus>();
        let pinned_host_control_bytes = std::mem::size_of::<PreimageStatus>();
        let sampler_event_bytes = 2 * std::mem::size_of::<usize>();
        let workspace = Self::tile_workspace_bytes(
            params,
            trapdoor,
            d,
            k,
            config.tile_columns.get(),
            magnitude_bytes,
        )?;
        let scratch_bytes = workspace
            .residual
            .checked_add(workspace.z_hat)
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        // The fixed native call stages the target tile on the source side and
        // packs the accepted compact result through host staging.  The output
        // itself is already represented by compact_output_bytes; don't charge
        // it a second time as destination staging.
        let source_device_staging_bytes = workspace.target_tile;
        let destination_device_staging_bytes = 0;
        let warm_peak = PreimageAllocationEnvelope::checked_sum([
            public_matrix_bytes,
            trapdoor_bytes,
            resident_target_bytes,
            retained_covariance_cache_bytes,
            compact_output_bytes,
            workspace.candidate,
            workspace.perturbation,
            scratch_bytes,
            source_device_staging_bytes,
            destination_device_staging_bytes,
            workspace.packed_staging,
            device_control_bytes,
            pinned_host_control_bytes,
            sampler_event_bytes,
            hard_cutoff_plan_bytes,
        ])?;
        let cold_peak = warm_peak
            .checked_add(if matches!(state, PreimageCacheState::Cold) {
                cold_transient_workspace_bytes
            } else {
                0
            })
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let envelope = PreimageAllocationEnvelope {
            public_matrix_bytes,
            trapdoor_bytes,
            resident_target_bytes,
            retained_covariance_cache_bytes,
            compact_output_bytes,
            candidate_workspace_bytes: workspace.candidate,
            perturbation_workspace_bytes: workspace.perturbation,
            scratch_bytes,
            source_device_staging_bytes,
            destination_device_staging_bytes,
            host_staging_bytes: workspace.packed_staging,
            device_control_bytes,
            pinned_host_control_bytes,
            sampler_event_bytes,
            cold_transient_workspace_bytes: if matches!(state, PreimageCacheState::Cold) {
                cold_transient_workspace_bytes
            } else {
                0
            },
            sampler_peak_bytes: warm_peak,
            cold_sampler_peak_bytes: cold_peak,
            evidence_kind: PreimageAllocationEvidenceKind::Certified,
        };
        let context = PreimageEvidenceContext {
            state,
            tile_columns: config.tile_columns.get(),
            max_attempts: config.max_attempts.get(),
            max_coefficient_bound: max_coefficient_bound.clone(),
            hard_cutoff_plan_bytes,
            format: PreimageFormatContext {
                active_moduli: params.moduli().to_vec(),
                crt_level: params.crt_depth().saturating_sub(1),
                coefficient_magnitude_bytes: magnitude_bytes,
                compact_output: true,
            },
            source_device_ids: params.gpu_ids().to_vec(),
            destination_device_ids: params.gpu_ids().to_vec(),
            cache_identity: self.cache_identity(params, trapdoor, d),
        };
        Ok(PreimageAllocationEvidence { context, envelope })
    }

    /// Return the exact/certified envelope for the fixed native call at this
    /// width.  This query never scales another width's peak and never chooses
    /// a different tile or retry policy.  Callers must retain the returned
    /// context with the evidence when freezing a production plan.
    pub fn preimage_allocation_evidence(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &dyn PolyMatrixColumnSource<GpuDCRTPolyMatrix>,
        max_coefficient_bound: &BigUint,
        config: FixedPreimageConfig,
        state: PreimageCacheState,
    ) -> Result<PreimageAllocationEvidence, SmallMatrixError> {
        self.allocation_evidence(
            params,
            trapdoor,
            public_matrix,
            target,
            max_coefficient_bound,
            config,
            state,
        )
    }

    /// Query the fixed job's complete sampler footprint without allocating or
    /// submitting any GPU work.  The caller owns the choice of tile width.
    pub fn preimage_footprint(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &dyn PolyMatrixColumnSource<GpuDCRTPolyMatrix>,
        max_coefficient_bound: &BigUint,
        config: FixedPreimageConfig,
    ) -> Result<PreimageFootprint, SmallMatrixError> {
        let evidence = self.preimage_allocation_evidence(
            params,
            trapdoor,
            public_matrix,
            target,
            max_coefficient_bound,
            config,
            PreimageCacheState::Cold,
        )?;
        let envelope = evidence.envelope;
        let persistent_bytes = PreimageAllocationEnvelope::checked_sum([
            envelope.public_matrix_bytes,
            envelope.trapdoor_bytes,
            envelope.resident_target_bytes,
            envelope.retained_covariance_cache_bytes,
        ])?;
        Ok(PreimageFootprint {
            persistent_bytes,
            compact_output_bytes: envelope.compact_output_bytes,
            scratch_bytes: envelope.scratch_bytes,
            device_control_bytes: envelope.device_control_bytes,
            pinned_control_bytes: envelope.pinned_host_control_bytes,
            hard_cutoff_plan_bytes: evidence.context.hard_cutoff_plan_bytes,
            packed_staging_bytes: envelope.host_staging_bytes,
            sampler_event_bytes: envelope.sampler_event_bytes,
            cold_transient_workspace_bytes: envelope.cold_transient_workspace_bytes,
            sampler_peak_bytes: envelope.sampler_peak_bytes,
            cold_sampler_peak_bytes: envelope.cold_sampler_peak_bytes,
            tile_columns: evidence.context.tile_columns,
            max_attempts: evidence.context.max_attempts,
        })
    }

    /// Execute one previously planned preimage job.  Allocation pressure is a
    /// hard error; this path never searches for another width or device.
    pub fn bounded_preimage_with_config(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &dyn PolyMatrixColumnSource<GpuDCRTPolyMatrix>,
        max_coefficient_bound: BigUint,
        config: FixedPreimageConfig,
        randomness_seed: [u8; 32],
    ) -> Result<GpuSmallMatrix, SmallMatrixError> {
        let footprint = self.preimage_footprint(
            params,
            trapdoor,
            public_matrix,
            target,
            &max_coefficient_bound,
            config,
        )?;
        self.execute_preimage_with_config(
            params,
            trapdoor,
            public_matrix,
            target,
            max_coefficient_bound,
            config,
            footprint,
            randomness_seed,
        )
    }

    fn execute_preimage_with_config(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &dyn PolyMatrixColumnSource<GpuDCRTPolyMatrix>,
        max_coefficient_bound: BigUint,
        config: FixedPreimageConfig,
        footprint: PreimageFootprint,
        randomness_seed: [u8; 32],
    ) -> Result<GpuSmallMatrix, SmallMatrixError> {
        let (d, k, columns, magnitude_bytes) = Self::validate_preimage_inputs(
            params,
            trapdoor,
            public_matrix,
            target,
            &max_coefficient_bound,
        )?;
        let budget = params.vram_budget_bytes();
        if !footprint.fits_budget(budget) {
            return Err(SmallMatrixError::ResourceExhausted {
                requested_bytes: footprint.cold_sampler_peak_bytes,
                budget_bytes: budget,
            });
        }
        let tile_columns = config.tile_columns.get();
        let workspace =
            Self::tile_workspace_bytes(params, trapdoor, d, k, tile_columns, magnitude_bytes)?;
        let mut destination = GpuSmallMatrix::new_empty_checked(
            params,
            k,
            columns,
            max_coefficient_bound,
            magnitude_bytes,
            budget,
        )?;
        // The retry body reuses one fixed device staging tile. Acceptance is
        // latched on the device; the host does not observe it between
        // attempts. A single status D2H is queued after the bounded body.
        destination.prepare_preimage_hard_cutoff_for_tile(k, tile_columns);
        let report = destination.sampler_allocation_report(
            footprint
                .persistent_bytes
                .checked_add(workspace.target_tile)
                .ok_or(SmallMatrixError::DimensionOverflow)?,
            workspace.candidate,
            workspace.perturbation,
            footprint.scratch_bytes,
            footprint.hard_cutoff_plan_bytes,
            footprint.packed_staging_bytes,
            footprint.sampler_event_bytes,
            footprint.device_control_bytes,
            footprint.pinned_control_bytes,
        )?;
        if report.sampler_peak_bytes != footprint.sampler_peak_bytes {
            return Err(SmallMatrixError::InvalidConfig);
        }
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
            "gpu preimage compact sampler residency"
        );
        for column_start in (0..columns).step_by(tile_columns) {
            let column_count = tile_columns.min(columns - column_start);
            let tile_target = target.load_columns(column_start, column_start + column_count);
            if tile_target.params != *params || tile_target.params.gpu_ids() != params.gpu_ids() {
                return Err(SmallMatrixError::ParameterMismatch);
            }
            let global_column = target
                .global_column_start()
                .checked_add(column_start)
                .ok_or(SmallMatrixError::DimensionOverflow)?;
            for attempt in 0..config.max_attempts.get() {
                let candidate = expanded_preimage_candidate(
                    self,
                    params,
                    trapdoor,
                    public_matrix,
                    &tile_target,
                    preimage_seed(randomness_seed, b"candidate", global_column, attempt),
                )
                .into_coeff_domain();
                destination.submit_preimage_hard_cutoff_tile(
                    &candidate,
                    0,
                    column_start,
                    k,
                    column_count,
                    attempt as u32,
                );
                drop(candidate);
            }
            destination.mark_preimage_exhausted();
            destination.copy_preimage_status_async();
            let status = destination.wait_preimage_status();
            if status.error_code != PreimageStatus::SUCCESS {
                return Err(SmallMatrixError::AttemptExhausted {
                    column_start,
                    column_count,
                    attempts: status.attempts as usize,
                });
            }
        }
        Ok(destination)
    }

    /// Legacy/benchmark entry point.  Its resource search is intentionally
    /// kept here; production callers use `bounded_preimage_with_config`.
    fn bounded_preimage(
        &self,
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public_matrix: &GpuDCRTPolyMatrix,
        target: &dyn PolyMatrixColumnSource<GpuDCRTPolyMatrix>,
        max_coefficient_bound: BigUint,
        randomness_seed: [u8; 32],
    ) -> Result<GpuSmallMatrix, SmallMatrixError> {
        let (_, _, columns, _) = Self::validate_preimage_inputs(
            params,
            trapdoor,
            public_matrix,
            target,
            &max_coefficient_bound,
        )?;
        let attempts = crate::env::gpu_preimage_max_tile_attempts()
            .map_err(|_| SmallMatrixError::InvalidConfig)?;
        let mut tile_columns = columns;
        let budget = params.vram_budget_bytes();
        while tile_columns > 0 {
            let config = FixedPreimageConfig::new(tile_columns, attempts)
                .ok_or(SmallMatrixError::InvalidConfig)?;
            let footprint = self.preimage_footprint(
                params,
                trapdoor,
                public_matrix,
                target,
                &max_coefficient_bound,
                config,
            )?;
            if footprint.fits_budget(budget) {
                return self.execute_preimage_with_config(
                    params,
                    trapdoor,
                    public_matrix,
                    target,
                    max_coefficient_bound,
                    config,
                    footprint,
                    randomness_seed,
                );
            }
            tile_columns -= 1;
        }
        Err(SmallMatrixError::ResourceExhausted {
            requested_bytes: budget.saturating_add(1),
            budget_bytes: budget,
        })
    }
}

impl GpuDCRTPolyTrapdoorSampler {
    /// Query the complete native allocation topology of `trapdoor()` without
    /// constructing a trapdoor or submitting CUDA work.
    pub fn trapdoor_allocation_evidence(
        params: &GpuDCRTPolyParams,
        size: usize,
    ) -> Result<TrapdoorAllocationEvidence, SmallMatrixError> {
        if size == 0 || params.dropped_moduli() != 0 {
            return Err(SmallMatrixError::InvalidConfig);
        }
        let digits = params.modulus_digits().max(1);
        let trapdoor_columns =
            size.checked_mul(digits).ok_or(SmallMatrixError::DimensionOverflow)?;
        let r = trapdoor_matrix_bytes(params, size, trapdoor_columns, true)?;
        let e = r;
        let coeff_product = trapdoor_matrix_bytes(params, size, size, false)?;
        let eval_product = trapdoor_matrix_bytes(params, size, size, true)?;
        let transpose = trapdoor_matrix_bytes(params, trapdoor_columns, size, true)?;
        let a_bar = coeff_product;
        let gadget = trapdoor_matrix_bytes(params, size, trapdoor_columns, true)?;
        let identity = coeff_product;
        let a0 = trapdoor_matrix_bytes(params, size, size.saturating_mul(2), true)?;
        let a1 = gadget;
        let public_output = trapdoor_matrix_bytes(
            params,
            size,
            size.checked_mul(2)
                .and_then(|value| value.checked_add(trapdoor_columns))
                .ok_or(SmallMatrixError::DimensionOverflow)?,
            true,
        )?;
        // A/B/D generation has one transpose and one product live at a time.
        // The a1 expression has product, sum, and final subtraction outputs
        // live together; summing these owners is a certified upper bound for
        // the native lifetime, not a measured peak.
        let scratch = transpose
            .checked_add(eval_product)
            .and_then(|value| value.checked_add(eval_product.saturating_mul(2)))
            .and_then(|value| value.checked_add(gadget))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        Ok(TrapdoorAllocationEvidence {
            trapdoor_r_bytes: r,
            trapdoor_e_bytes: e,
            trapdoor_a_coeff_bytes: coeff_product,
            trapdoor_b_coeff_bytes: coeff_product,
            trapdoor_d_coeff_bytes: coeff_product,
            public_a_bar_bytes: a_bar,
            gadget_bytes: gadget,
            identity_bytes: identity,
            a0_assembly_bytes: a0,
            a1_assembly_bytes: a1,
            public_output_bytes: public_output,
            scratch_bytes: scratch,
            control_bytes: 0,
            cache_bytes: 0,
            host_bytes: 0,
            pinned_host_bytes: 0,
            evidence_kind: PreimageAllocationEvidenceKind::Certified,
        })
    }
}

impl PolyTrapdoorSampler for GpuDCRTPolyTrapdoorSampler {
    type M = GpuDCRTPolyMatrix;
    type Trapdoor = GpuDCRTTrapdoor;

    fn new(params: &<<Self::M as PolyMatrix>::P as Poly>::Params, sigma: f64) -> Self {
        assert_eq!(
            params.dropped_moduli(),
            0,
            "exact trapdoor sampling requires dropped_moduli = 0"
        );
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
        let g = GpuDCRTPolyMatrix::gadget_matrix(params, size, None);
        let identity = GpuDCRTPolyMatrix::identity_columns(params, size, 0, size);
        let a0 = a_bar.concat_columns(&[&identity]);
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
        if params.dropped_moduli() != 0 {
            return Err(SmallMatrixError::InvalidConfig);
        }
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
        self.bounded_preimage(
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
        let preimage_left = expanded_preimage_candidate(
            self,
            params,
            trapdoor,
            public_matrix,
            &t,
            random_gpu_rng_seed(),
        );
        preimage_left.concat_rows(&[&preimage_right])
    }
}

fn expanded_preimage_candidate(
    sampler: &GpuDCRTPolyTrapdoorSampler,
    params: &GpuDCRTPolyParams,
    trapdoor: &GpuDCRTTrapdoor,
    public_matrix: &GpuDCRTPolyMatrix,
    target: &GpuDCRTPolyMatrix,
    randomness_seed: GpuRngSeed,
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
        preimage_seed(randomness_seed.to_bytes(), b"perturb", 0, 0),
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
    let z_hat_mat = perturbed_syndrome.gauss_samp_gq_arb_base(
        sampler.c,
        sampler.sigma,
        preimage_seed(randomness_seed.to_bytes(), b"z", 0, 0),
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
    let d = trapdoor.r.row_size();
    let dk = trapdoor.r.col_size();
    tracing::debug!(d = d, dk = dk, total_ncol = total_ncol, "gpu preimage sample_pert: start");

    // p2 is sampled directly on GPU as in the Karney branch of OpenFHE.  The
    // covariance sampler accepts arbitrary column counts; retaining the
    // requested tile width avoids allocating an artificial d-column tail.
    let p2 = sample_gpu_matrix_with_seed(
        params,
        dk,
        total_ncol,
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
        matrix::{PolyMatrix, PolyMatrixSmallRhs, SmallPolyMatrix},
        poly::{
            PolyParams,
            dcrt::{
                gpu::{GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
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

    #[test]
    #[sequential]
    fn test_gpu_trapdoor_shallow_clone_preserves_component_identity() {
        if detected_gpu_device_ids().is_empty() {
            return;
        }
        let params = GpuDCRTPolyParams::new(16, vec![65_537, 67_073], 8, None);
        let descriptor = GpuDCRTTrapdoor::output_descriptor_for_shape(&params, 1, 2)
            .expect("trapdoor output descriptor");
        let (_, trapdoor) = GpuDCRTTrapdoor::allocate_from_descriptor(&descriptor);
        trapdoor.wait_until_ready();
        let expected = [
            trapdoor.r.binding_components().expect("r bindings"),
            trapdoor.e.binding_components().expect("e bindings"),
            trapdoor.a_mat_coeff.binding_components().expect("a bindings"),
            trapdoor.b_mat_coeff.binding_components().expect("b bindings"),
            trapdoor.d_mat_coeff.binding_components().expect("d bindings"),
        ];
        let retained = trapdoor.clone_shallow();
        drop(trapdoor);
        let actual = [
            retained.r.binding_components().expect("retained r bindings"),
            retained.e.binding_components().expect("retained e bindings"),
            retained.a_mat_coeff.binding_components().expect("retained a bindings"),
            retained.b_mat_coeff.binding_components().expect("retained b bindings"),
            retained.d_mat_coeff.binding_components().expect("retained d bindings"),
        ];
        assert_eq!(expected, actual, "shallow trapdoor handles must retain native identities");
    }

    #[test]
    fn fixed_preimage_config_requires_nonzero_choices() {
        assert_eq!(FixedPreimageConfig::new(0, 1), None);
        assert_eq!(FixedPreimageConfig::new(1, 0), None);
        let config = FixedPreimageConfig::new(3, 7).expect("positive fixed choices");
        assert_eq!(config.tile_columns.get(), 3);
        assert_eq!(config.max_attempts.get(), 7);
    }

    #[test]
    fn preimage_seed_uses_logical_attempt_coordinates_and_keccak_abi() {
        let base = [0xabu8; 32];
        let t0 = preimage_seed_bytes(base, b"candidate", 11, 0);
        let t1 = preimage_seed_bytes(base, b"candidate", 11, 1);
        let other_column = preimage_seed_bytes(base, b"candidate", 12, 0);
        assert_ne!(t0, t1, "t=0 and t=1 must be distinct logical draws");
        assert_ne!(t0, other_column, "column identity must be part of the seed");
        assert_eq!(t0, preimage_seed(base, b"candidate", 11, 0).to_bytes());

        // Keep this byte-order check close to the ABI definition: all integer
        // fields are LE64 and the stage is length-prefixed, rather than being
        // concatenated with a delimiter or hashed with SHA3.
        let mut reference = keccak_asm::Keccak256::new();
        reference.update(b"mxx-preimage-sampler/v1");
        reference.update(base);
        reference.update((9u64).to_le_bytes());
        reference.update(b"candidate");
        reference.update((11u64).to_le_bytes());
        reference.update((0u64).to_le_bytes());
        assert_eq!(t0, <[u8; 32]>::from(reference.finalize()));
    }

    #[test]
    fn preimage_status_is_a_single_run_resettable_gate() {
        assert_eq!(std::mem::size_of::<PreimageStatus>(), 16);
        let status = PreimageStatus::reset();
        assert_eq!(status, PreimageStatus::default());
        assert!(!status.succeeded());

        let accepted = PreimageStatus { attempts: 1, accepted: 1, ..status };
        assert!(accepted.succeeded());
        assert!(!accepted.exhausted());

        let exhausted = PreimageStatus {
            attempts: 4,
            accepted: 0,
            error_code: PreimageStatus::EXHAUSTED,
            reserved: 0,
        };
        assert!(exhausted.exhausted());
        assert!(!exhausted.succeeded());
        assert_eq!(PreimageStatus::reset(), status, "the next tile/run starts clean");
    }

    #[test]
    fn preimage_launch_control_has_stable_fixed_layout() {
        assert_eq!(std::mem::size_of::<PreimageLaunchControl>(), 64);
        let control = PreimageLaunchControl {
            execution_nonce: [0x11; 32],
            logical_instance: 3,
            global_column_start: 9,
            max_attempts: 7,
            attempt: 0,
            binding_table: 0x1000,
        };
        assert_eq!(control.max_attempts, 7);
        assert_eq!(control.binding_table, 0x1000);
    }

    #[test]
    fn captured_retry_seed_consumes_nonce_instance_column_and_attempt() {
        let control = PreimageLaunchControl {
            execution_nonce: [0x11; 32],
            logical_instance: 3,
            global_column_start: 9,
            max_attempts: 7,
            attempt: 0,
            binding_table: 0x1000,
        };
        let baseline = control.attempt_seed(b"candidate", 0).to_bytes();
        assert_ne!(baseline, control.attempt_seed(b"candidate", 1).to_bytes());
        let mut changed = control;
        changed.execution_nonce[0] ^= 1;
        assert_ne!(baseline, changed.attempt_seed(b"candidate", 0).to_bytes());
        changed = control;
        changed.logical_instance += 1;
        assert_ne!(baseline, changed.attempt_seed(b"candidate", 0).to_bytes());
        changed = control;
        changed.global_column_start += 1;
        assert_ne!(baseline, changed.attempt_seed(b"candidate", 0).to_bytes());

        let mut reference = keccak_asm::Keccak256::new();
        reference.update(b"mxx-preimage-sampler/v2");
        reference.update(control.execution_nonce);
        reference.update(control.logical_instance.to_le_bytes());
        reference.update(control.global_column_start.to_le_bytes());
        reference.update((9u64).to_le_bytes());
        reference.update(b"candidate");
        reference.update(0u32.to_le_bytes());
        assert_eq!(baseline, <[u8; 32]>::from(reference.finalize()));
    }

    #[test]
    fn retry_stage_seeds_are_byte_exact_and_distinct_per_attempt() {
        let control = PreimageLaunchControl {
            execution_nonce: [0x42; 32],
            logical_instance: 19,
            global_column_start: 73,
            max_attempts: 4,
            attempt: 0,
            binding_table: 0,
        };
        for stage in [b"p2".as_slice(), b"p1".as_slice(), b"z".as_slice()] {
            let mut previous = None;
            for attempt in 0..control.max_attempts {
                let actual = preimage_seed_bytes(
                    control.execution_nonce,
                    stage,
                    control.global_column_start as usize,
                    attempt as usize,
                );
                let mut reference = keccak_asm::Keccak256::new();
                reference.update(b"mxx-preimage-sampler/v1");
                reference.update(control.execution_nonce);
                reference.update((stage.len() as u64).to_le_bytes());
                reference.update(stage);
                reference.update((control.global_column_start as u64).to_le_bytes());
                reference.update((attempt as u64).to_le_bytes());
                assert_eq!(actual, <[u8; 32]>::from(reference.finalize()));
                if let Some(previous) = previous {
                    assert_ne!(previous, actual, "stage seed repeated across attempts");
                }
                previous = Some(actual);
            }
        }
        assert_ne!(
            preimage_seed_bytes(
                control.execution_nonce,
                b"p2",
                control.global_column_start as usize,
                0
            ),
            preimage_seed_bytes(
                control.execution_nonce,
                b"p1",
                control.global_column_start as usize,
                0
            ),
        );
        assert_ne!(
            preimage_seed_bytes(
                control.execution_nonce,
                b"p1",
                control.global_column_start as usize,
                0
            ),
            preimage_seed_bytes(
                control.execution_nonce,
                b"z",
                control.global_column_start as usize,
                0
            ),
        );
    }

    #[test]
    fn preimage_cache_identity_digest_binds_owner_parameters_and_basis_order() {
        let identity =
            PreimageCacheIdentity { owner_token: 7, c: 2.0, smoothing: 3.0, dgg_stddev: 4.0 };
        let baseline = identity.opaque_digest(&[8, 16], &[17, 19, 23]);
        assert_ne!(baseline, identity.opaque_digest(&[8, 16], &[19, 17, 23]));
        assert_ne!(baseline, identity.opaque_digest(&[8, 32], &[17, 19, 23]));
        let owner_changed = PreimageCacheIdentity { owner_token: 8, ..identity };
        assert_ne!(baseline, owner_changed.opaque_digest(&[8, 16], &[17, 19, 23]));
    }

    #[test]
    fn preimage_footprint_separates_cold_covariance_workspace_from_warm_peak() {
        let footprint = PreimageFootprint {
            persistent_bytes: 40,
            compact_output_bytes: 10,
            scratch_bytes: 20,
            device_control_bytes: 1,
            pinned_control_bytes: 1,
            hard_cutoff_plan_bytes: 2,
            packed_staging_bytes: 3,
            sampler_event_bytes: 1,
            cold_transient_workspace_bytes: 17,
            sampler_peak_bytes: 78,
            cold_sampler_peak_bytes: 95,
            tile_columns: 1,
            max_attempts: 1,
        };
        assert_eq!(footprint.cold_sampler_peak_bytes, footprint.sampler_peak_bytes + 17);
        assert!(!footprint.fits_budget(94));
        assert!(footprint.fits_budget(95));
        assert_eq!(footprint.sampler_peak_bytes, 78);
    }

    fn evidence_envelope(transient: usize, width: usize) -> PreimageAllocationEvidence {
        let envelope = PreimageAllocationEnvelope {
            public_matrix_bytes: 11,
            trapdoor_bytes: 13,
            resident_target_bytes: 17,
            retained_covariance_cache_bytes: 19,
            compact_output_bytes: width * 23,
            candidate_workspace_bytes: width * 29,
            perturbation_workspace_bytes: width * 31,
            scratch_bytes: width * 37,
            source_device_staging_bytes: width * 41,
            destination_device_staging_bytes: 0,
            host_staging_bytes: width * 43,
            device_control_bytes: 3,
            pinned_host_control_bytes: 5,
            sampler_event_bytes: 7,
            cold_transient_workspace_bytes: transient,
            sampler_peak_bytes: 0,
            cold_sampler_peak_bytes: 0,
            evidence_kind: PreimageAllocationEvidenceKind::Certified,
        };
        let warm = PreimageAllocationEnvelope::checked_sum([
            envelope.public_matrix_bytes,
            envelope.trapdoor_bytes,
            envelope.resident_target_bytes,
            envelope.retained_covariance_cache_bytes,
            envelope.compact_output_bytes,
            envelope.candidate_workspace_bytes,
            envelope.perturbation_workspace_bytes,
            envelope.scratch_bytes,
            envelope.source_device_staging_bytes,
            envelope.destination_device_staging_bytes,
            envelope.host_staging_bytes,
            envelope.device_control_bytes,
            envelope.pinned_host_control_bytes,
            envelope.sampler_event_bytes,
        ])
        .unwrap();
        let mut envelope = envelope;
        envelope.sampler_peak_bytes = warm;
        envelope.cold_sampler_peak_bytes = warm + transient;
        PreimageAllocationEvidence {
            context: PreimageEvidenceContext {
                state: if transient == 0 {
                    PreimageCacheState::Warm
                } else {
                    PreimageCacheState::Cold
                },
                tile_columns: width,
                max_attempts: 4,
                max_coefficient_bound: BigUint::from(255u32),
                hard_cutoff_plan_bytes: 2,
                format: PreimageFormatContext {
                    active_moduli: vec![17, 19, 23],
                    crt_level: 2,
                    coefficient_magnitude_bytes: 1,
                    compact_output: true,
                },
                source_device_ids: vec![0],
                destination_device_ids: vec![0],
                cache_identity: PreimageCacheIdentity {
                    owner_token: 99,
                    c: 2.0,
                    smoothing: 3.0,
                    dgg_stddev: 4.0,
                },
            },
            envelope,
        }
    }

    #[test]
    fn preimage_allocation_evidence_distinguishes_cold_and_warm() {
        let cold = evidence_envelope(47, 1);
        let warm = evidence_envelope(0, 1);
        assert_eq!(cold.context.tile_columns, warm.context.tile_columns);
        assert_ne!(cold.context.state, warm.context.state);
        assert_eq!(cold.envelope.sampler_peak_bytes, warm.envelope.sampler_peak_bytes);
        assert_eq!(cold.envelope.cold_sampler_peak_bytes, warm.envelope.sampler_peak_bytes + 47);
        assert_eq!(cold.envelope.retained_covariance_cache_bytes, 19);
    }

    #[test]
    fn preimage_allocation_evidence_keeps_direct_width_one_and_eight_points() {
        let width_one = evidence_envelope(0, 1);
        let width_eight = evidence_envelope(0, 8);
        assert_eq!(width_one.context.tile_columns, 1);
        assert_eq!(width_eight.context.tile_columns, 8);
        assert_ne!(width_one.envelope.sampler_peak_bytes, width_eight.envelope.sampler_peak_bytes);
        // A width-8 point carries its own exact envelope; there is no API
        // operation that multiplies the width-1 point into this value.
        assert_eq!(width_eight.envelope.compact_output_bytes, 8 * 23);
    }

    #[test]
    fn preimage_evidence_context_invalidates_cache_retry_and_cutoff_changes() {
        let baseline = evidence_envelope(0, 1);
        let mut retry_changed = baseline.clone();
        retry_changed.context.max_attempts = 5;
        let mut cutoff_changed = baseline.clone();
        cutoff_changed.context.max_coefficient_bound = BigUint::from(511u32);
        let mut cache_changed = baseline.clone();
        cache_changed.context.cache_identity.owner_token += 1;
        assert_ne!(baseline.context, retry_changed.context);
        assert_ne!(baseline.context, cutoff_changed.context);
        assert_ne!(baseline.context, cache_changed.context);
    }

    #[test]
    #[sequential]
    fn covariance_cache_builder_is_session_owned_and_idempotent() {
        let devices = detected_gpu_device_ids();
        if devices.is_empty() {
            return;
        }
        let params = GpuDCRTPolyParams::new(128, vec![65_537, 67_073], 8, None);
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (first, _) = sampler.trapdoor(&params, 1);
        assert!(!sampler.has_retained_preimage_covariance_cache(&params, &first));
        sampler.build_preimage_covariance_cache(&params, &first);
        assert!(sampler.has_retained_preimage_covariance_cache(&params, &first));
        // A distinct trapdoor owner must enter its own cold/setup path even
        // when all numerical sampler parameters are identical.
        let (second, _) = sampler.trapdoor(&params, 1);
        assert!(!sampler.has_retained_preimage_covariance_cache(&params, &second));
        gpu_device_sync();
    }

    fn gpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 16, 8, None, None)
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
            random_gpu_rng_seed(),
        );
        let mut p_hat = GpuDCRTPolyMatrix::new_empty_with_state(
            params,
            p1.row_size() + p2.row_size(),
            total_ncol,
            p1.level(),
            p1.is_ntt(),
            None,
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
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17, None, None);
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
        let seed = rand::random();
        let source = crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone());
        let compact = sampler
            .preimage(&params, &trapdoor, &public_matrix, &source, bound.clone(), seed)
            .expect("compact preimage should be sampled");
        let repeated = sampler
            .preimage(&params, &trapdoor, &public_matrix, &source, bound, seed)
            .expect("same-seed compact preimage should be sampled");
        assert_eq!(compact, repeated, "the same seed and tile schedule must reproduce the output");
        assert_eq!(compact.rows_count(), public_matrix.col_size());
        assert_eq!(compact.columns_count(), target.col_size());
        assert_eq!(public_matrix.multiply_small_rhs(&compact).unwrap(), target);
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
            sampler.preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                requested.clone(),
                rand::random()
            ),
            Err(SmallMatrixError::PreimageBoundTooSmall { requested, minimum })
        );
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
        let candidate = expanded_preimage_candidate(
            sampler,
            params,
            trapdoor,
            public_matrix,
            target,
            random_gpu_rng_seed(),
        )
        .into_coeff_domain();
        let inspection_bound = params.modulus().as_ref() >> 1u8;
        let mut canonical = GpuSmallMatrix::new_empty(
            params,
            candidate.row_size(),
            candidate.col_size(),
            inspection_bound,
        )
        .expect("canonical inspection owner");
        canonical.prepare_preimage_hard_cutoff();
        assert!(
            canonical
                .try_pack_preimage_hard_cutoff_tile(
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
        destination.prepare_preimage_hard_cutoff();
        let mut candidates = [Some(high_candidate), Some(low_candidate)].into_iter();
        let mut attempt_count = 0usize;
        let outcome = bounded_retry(2, || {
            attempt_count += 1;
            let candidate =
                candidates.next().flatten().expect("retry must consume exactly two candidates");
            let accepted = destination.try_pack_preimage_hard_cutoff_tile(
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
        assert_eq!(public_matrix.multiply_small_rhs(&destination).unwrap(), target);
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
        let outcome = sampler.bounded_preimage(
            &params,
            &trapdoor,
            &public_matrix,
            &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
            BigUint::ZERO,
            rand::random(),
        );
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
        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&params, size, None);
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
        let cpu_decoded =
            crate::sampler::trapdoor::DCRTTrapdoor::try_from_compact_bytes(&cpu_params, &bytes)
                .expect("GPU trapdoor bytes must use the six-leaf CPU codec");
        assert_eq!(cpu_decoded.to_compact_bytes(), bytes);
        assert_eq!(cpu_decoded.re.row_size(), 2 * size);
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
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                permissive_preimage_bound(&params),
                rand::random(),
            )
            .expect("permissive bound should accept a preimage");
        let product = public_matrix.multiply_small_rhs(&preimage).unwrap();
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
                    &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                    permissive_preimage_bound(&params),
                    rand::random(),
                )
                .expect("permissive bound should accept a preimage");
            assert_eq!(preimage.columns_count(), chunk_width);
            assert_eq!(
                public_matrix.multiply_small_rhs(&preimage).unwrap(),
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
                &crate::matrix::ResidentPolyMatrixColumnSource::new(first_target.clone()),
                permissive_preimage_bound(&params),
                rand::random(),
            )
            .expect("permissive bound should accept the first preimage");
        let second_preimage = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(second_target.clone()),
                permissive_preimage_bound(&params),
                rand::random(),
            )
            .expect("permissive bound should accept the second preimage");

        assert_eq!(public_matrix.multiply_small_rhs(&first_preimage).unwrap(), first_target);
        assert_eq!(public_matrix.multiply_small_rhs(&second_preimage).unwrap(), second_target);
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
        plain.prepare_preimage_hard_cutoff();
        assert!(
            plain
                .try_pack_preimage_hard_cutoff_tile(
                    &z_plain_full.clone().into_coeff_domain(),
                    0,
                    0,
                    z_plain_full.row_size(),
                    z_plain_full.col_size(),
                )
                .expect("plain gadget preimage should fit the permissive bound")
        );
        let sampled = trapdoor_sampler
            .preimage(
                &params,
                &trapdoor,
                &public_matrix,
                &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                bound,
                rand::random(),
            )
            .expect("permissive bound should accept a sampled preimage");
        assert_eq!(public_matrix.multiply_small_rhs(&sampled).unwrap(), target);
        assert_ne!(
            sampled.to_canonical_coefficients().unwrap(),
            plain.to_canonical_coefficients().unwrap(),
            "preimage sampler should not collapse to the plain deterministic gadget preimage"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_sampler_parameters_follow_instance_sigma() {
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17, None, None);
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
                    .preimage(
                        &params,
                        &trapdoor,
                        &public_matrix,
                        &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                        cutoff.clone(),
                        rand::random(),
                    )
                    .expect("bounded preimage")
            })
            .collect::<Vec<_>>();

        for (preimage, target) in outputs.into_iter().zip(&targets) {
            assert_eq!(preimage.max_coefficient_bound(), &cutoff);
            assert_eq!(public_matrix.multiply_small_rhs(&preimage).unwrap(), *target);
        }
    }

    fn assert_gpu_preimage_reconstructs_target_and_respects_norm_bound(
        sigma: f64,
        bound_sigma: Option<f64>,
    ) {
        gpu_device_sync();
        let size = 2usize;
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17, None, None);
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
                .preimage(
                    &params,
                    &trapdoor,
                    &public_matrix,
                    &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                    preimage_bound.clone(),
                    rand::random(),
                )
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
            assert_eq!(public_matrix.multiply_small_rhs(&preimage).unwrap(), target);
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
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17, None, None);
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
        let cpu_params = DCRTPolyParams::new(1 << 10, 5, 51, 17, None, None);
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

            let src_params = base_params.params_for_device(src_device, None);
            let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&src_params, size);
            let target =
                uniform_sampler.sample_uniform(&src_params, size, size, DistType::FinRingDist);
            let preimage = trapdoor_sampler
                .preimage(
                    &src_params,
                    &trapdoor,
                    &public_matrix,
                    &crate::matrix::ResidentPolyMatrixColumnSource::new(target.clone()),
                    preimage_bound.clone(),
                    rand::random(),
                )
                .expect("bounded source-device preimage");
            assert_eq!(
                public_matrix.multiply_small_rhs(&preimage).unwrap(),
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
            let dst_params = base_params.params_for_device(case.dst_device, None);
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
                public_matrix.multiply_small_rhs(&preimage).unwrap(),
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
