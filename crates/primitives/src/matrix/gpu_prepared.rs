//! Persistent, destination-aware GPU primitive plans.
//!
//! Plans retain immutable launch metadata and fixed destination owners. Replay
//! replaces input payloads without changing storage or native launch geometry.

use super::{
    GpuDCRTPolyMatrix, GpuMatrixModulusConversion, GpuMatrixRnsConversion, GpuMatrixSampleDist,
    GpuSmallMatrix,
};
use crate::{
    matrix::{PolyMatrix, SmallPolyMatrix},
    poly::{PolyParams, dcrt::gpu::last_error_string},
};
use std::{ops::Range, ptr::NonNull, sync::Arc};

#[path = "gpu_schedule.rs"]
mod gpu_schedule;
pub use gpu_schedule::{GpuPreparedSchedule, GpuPreparedSchedulePlan};
#[path = "gpu_compact_decompose.rs"]
mod gpu_compact_decompose;
pub use gpu_compact_decompose::GpuPreparedCompactDecompose;
#[path = "gpu_scalar.rs"]
mod gpu_scalar;
pub use gpu_scalar::{
    GpuPreparedScalarBuffer, GpuPreparedScalarMatrixSelect, GpuPreparedScalarOp,
    GpuPreparedScalarOpcode, GpuPreparedScalarPack, GpuPreparedThreshold,
};

use crate::poly::dcrt::gpu::{
    GpuMatrixBatchView, GpuMatrixRange, GpuMatrixTransformPlanOpaque,
    GpuPreparedArithmeticLayout as NativeArithmeticLayout, GpuPreparedArithmeticOpaque,
    GpuPreparedCenteredRebaseOpaque, GpuPreparedCompactUploadOpaque,
    GpuPreparedConstCoeffReadbackOpaque, GpuPreparedCrtRecomposeOpaque,
    GpuPreparedGadgetDecomposeOpaque, GpuPreparedInputCopyOpaque,
    GpuPreparedModulusConversionOpaque, GpuPreparedRectLayout as NativeRectLayout,
    GpuPreparedRnsUploadOpaque, GpuPreparedSamplingOpaque, GpuPreparedSmallRhsOpaque,
    GpuPreparedSmallUploadOpaque, GpuPreparedTransposeOpaque, PinnedHostBuffer,
    gpu_matrix_defer_compact_upload_pinned_free, gpu_matrix_defer_const_coeff_readback_pinned_free,
    gpu_matrix_defer_rns_upload_pinned_free, gpu_matrix_defer_small_upload_pinned_free,
    gpu_matrix_destroy_arithmetic_plan, gpu_matrix_destroy_centered_rebase,
    gpu_matrix_destroy_compact_upload, gpu_matrix_destroy_const_coeff_readback,
    gpu_matrix_destroy_gadget_decompose, gpu_matrix_destroy_input_copy,
    gpu_matrix_destroy_ntt_plan, gpu_matrix_destroy_prepared_crt_recompose,
    gpu_matrix_destroy_prepared_modulus_conversion, gpu_matrix_destroy_prepared_small_rhs,
    gpu_matrix_destroy_rns_upload, gpu_matrix_destroy_sampling, gpu_matrix_destroy_small_upload,
    gpu_matrix_destroy_transpose, gpu_matrix_prepare_arithmetic,
    gpu_matrix_prepare_arithmetic_with_layout, gpu_matrix_prepare_centered_rebase,
    gpu_matrix_prepare_centered_rebase_with_layout, gpu_matrix_prepare_compact_upload,
    gpu_matrix_prepare_const_coeff_readback, gpu_matrix_prepare_crt_recompose,
    gpu_matrix_prepare_crt_recompose_with_layout, gpu_matrix_prepare_gadget_decompose,
    gpu_matrix_prepare_gadget_decompose_with_layout, gpu_matrix_prepare_input_copy,
    gpu_matrix_prepare_input_copy_with_layout, gpu_matrix_prepare_modulus_conversion,
    gpu_matrix_prepare_modulus_conversion_with_layout, gpu_matrix_prepare_ntt_plan,
    gpu_matrix_prepare_ntt_plan_with_layout, gpu_matrix_prepare_rns_conversion,
    gpu_matrix_prepare_rns_conversion_with_layout, gpu_matrix_prepare_rns_upload,
    gpu_matrix_prepare_sampling, gpu_matrix_prepare_small_rhs, gpu_matrix_prepare_small_upload,
    gpu_matrix_prepare_transpose, gpu_matrix_prepare_transpose_with_layout,
    gpu_matrix_query_compact_upload, gpu_matrix_query_const_coeff_readback,
    gpu_matrix_query_rns_upload, gpu_matrix_query_small_upload, gpu_matrix_submit_arithmetic,
    gpu_matrix_submit_centered_rebase, gpu_matrix_submit_compact_upload,
    gpu_matrix_submit_const_coeff_readback, gpu_matrix_submit_crt_recompose,
    gpu_matrix_submit_gadget_decompose, gpu_matrix_submit_input_copy,
    gpu_matrix_submit_modulus_conversion, gpu_matrix_submit_ntt_plan, gpu_matrix_submit_rns_upload,
    gpu_matrix_submit_sampling, gpu_matrix_submit_small_rhs, gpu_matrix_submit_small_upload,
    gpu_matrix_submit_transpose, gpu_matrix_wait_compact_upload,
    gpu_matrix_wait_const_coeff_readback, gpu_matrix_wait_rns_upload, gpu_matrix_wait_small_upload,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedRectLayout {
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub limb_count: usize,
    pub workspace_bytes: usize,
    pub alignment: usize,
    pub event_count: usize,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub stage_role: u32,
    pub device: i32,
}

impl PreparedRectLayout {
    fn from_native(native: NativeRectLayout) -> Self {
        Self {
            rows: native.rows,
            columns: native.columns,
            ring_dimension: native.ring_dimension,
            limb_count: native.limb_count,
            workspace_bytes: native.workspace_bytes,
            alignment: native.alignment,
            event_count: native.event_count,
            grid: [native.grid_x, native.grid_y, native.grid_z],
            block: [native.block_x, native.block_y, native.block_z],
            stage_role: native.stage_role as u32,
            device: native.device,
        }
    }

    pub fn query_input_copy(
        ring_dimension: usize,
        limb_count: usize,
        rows: usize,
        columns: usize,
        device: i32,
    ) -> Result<Self, String> {
        let mut native = NativeRectLayout::default();
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_matrix_query_input_copy_layout(
                ring_dimension,
                limb_count,
                rows,
                columns,
                device,
                &mut native,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_native(native))
    }

    pub fn query_transpose(
        ring_dimension: usize,
        limb_count: usize,
        rows: usize,
        columns: usize,
        device: i32,
    ) -> Result<Self, String> {
        let mut native = NativeRectLayout::default();
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_matrix_query_transpose_layout(
                ring_dimension,
                limb_count,
                rows,
                columns,
                device,
                &mut native,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_native(native))
    }
}

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{ToPrimitive, Zero};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

/// Fixed-output sampler replay. The native plan owns the descriptor table,
/// stream, launch geometry, distribution parameters, and optional coefficient
/// to evaluation transform. Submission only supplies the logical RNG seed.
pub struct GpuPreparedSampling {
    raw: NonNull<GpuPreparedSamplingOpaque>,
    output: Arc<GpuDCRTPolyMatrix>,
}

pub struct GpuPreparedSamplingInFlight {
    plan: Arc<GpuPreparedSampling>,
}

/// Fixed geometry hash sampler. Keccak key/tag expansion is performed once at
/// bind time; replay only submits the resulting seed to the native sampling
/// kernel. The output owner and native descriptor/stream/event plan are kept
/// alive by this object.
pub struct GpuPreparedHashSample {
    sampler: Arc<GpuPreparedSampling>,
    tag: Box<[u8]>,
    seed: crate::poly::dcrt::gpu::GpuRngSeed,
}

/// Native descriptors for the two-stage hash-compact command. Hash sampling
/// fills an ordinary scratch owner first; compact decomposition then performs
/// its optional inverse transform/correction and writes the compact payload.
#[derive(Clone, Debug)]
pub struct GpuPreparedHashCompactLayout {
    pub sampling: super::PreparedPlanLayout,
    pub decomposition: super::PreparedPlanLayout,
}

pub struct GpuPreparedHashCompactCommand {
    hash: Arc<GpuPreparedHashSample>,
    decomposition: Arc<GpuPreparedCompactDecompose>,
}

/// Fixed coefficient-domain readback. The native plan captures every source
/// limb's stream, pitch and copy width; this owner keeps both the source
/// matrix and the prepare-owned pinned destination alive until the host wait.
pub struct GpuPreparedConstCoeffReadback {
    raw: NonNull<GpuPreparedConstCoeffReadbackOpaque>,
    source: Arc<GpuDCRTPolyMatrix>,
    words: PinnedHostBuffer<u64>,
    layout: super::PreparedPlanLayout,
    busy: AtomicBool,
}

pub struct GpuPreparedConstCoeffReadbackInFlight {
    plan: Arc<GpuPreparedConstCoeffReadback>,
}

/// Fixed RNS upload. The host staging allocation and every device workspace,
/// destination descriptor, stream and completion event are prepared once.
/// The only per-submit host work is replacing the bytes in the same pinned
/// staging allocation.
pub struct GpuPreparedRnsUpload {
    raw: NonNull<GpuPreparedRnsUploadOpaque>,
    layout: super::PreparedPlanLayout,
    target: Arc<GpuDCRTPolyMatrix>,
    bytes: Mutex<PinnedHostBuffer<u8>>,
    bytes_per_poly: usize,
    polynomial_count: usize,
    ring_dimension: usize,
    source_evaluation: bool,
    moduli: Box<[u64]>,
    busy: AtomicBool,
}

pub struct GpuPreparedRnsUploadInFlight {
    plan: Arc<GpuPreparedRnsUpload>,
}

/// Fixed compact-artifact replay upload. The artifact header is decoded at
/// the Rust boundary; native submission receives only the borrowed payload in
/// this pinned, warmup-owned slot.
pub struct GpuPreparedCompactUpload {
    raw: NonNull<GpuPreparedCompactUploadOpaque>,
    target: Arc<GpuDCRTPolyMatrix>,
    payload: Mutex<PinnedHostBuffer<u8>>,
    layout: super::PreparedPlanLayout,
    busy: AtomicBool,
}

pub struct GpuPreparedCompactUploadInFlight {
    plan: Arc<GpuPreparedCompactUpload>,
}

/// Fixed canonical compact-matrix replay upload for bounded/preimage values.
pub struct GpuPreparedSmallUpload {
    raw: NonNull<GpuPreparedSmallUploadOpaque>,
    target: Arc<GpuSmallMatrix>,
    payload: Mutex<PinnedHostBuffer<u8>>,
    layout: super::PreparedPlanLayout,
    busy: AtomicBool,
}

pub struct GpuPreparedSmallUploadInFlight {
    plan: Arc<GpuPreparedSmallUpload>,
}

/// Host-side CRT reconstruction paired with a fixed RNS readback. The CRT
/// basis, modulus and result shape are all fixed during bind; callers access
/// the reusable result container through `with_values` after completion.
pub struct GpuPreparedRnsReconstruction {
    readback: Arc<GpuPreparedConstCoeffReadback>,
    layout: super::PreparedPlanLayout,
    reconstruction: Box<[num_bigint::BigUint]>,
    modulus: num_bigint::BigUint,
    coefficient_count: usize,
    values: Mutex<Box<[num_bigint::BigUint]>>,
}

pub struct GpuPreparedRnsReconstructionInFlight {
    plan: Arc<GpuPreparedRnsReconstruction>,
    readback: GpuPreparedConstCoeffReadbackInFlight,
}

unsafe impl Send for GpuPreparedConstCoeffReadback {}
unsafe impl Sync for GpuPreparedConstCoeffReadback {}
unsafe impl Send for GpuPreparedConstCoeffReadbackInFlight {}
unsafe impl Sync for GpuPreparedConstCoeffReadbackInFlight {}

impl GpuPreparedConstCoeffReadback {
    pub fn bind(
        source: Arc<GpuDCRTPolyMatrix>,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
        layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let (rows, columns) = source.size();
        let word_count = rows
            .checked_mul(columns)
            .and_then(|count| count.checked_mul(words_per_poly))
            .ok_or_else(|| "prepared coefficient readback output size overflow".to_string())?;
        let mut words = PinnedHostBuffer::zeroed(source.params(), word_count);
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_const_coeff_readback(
                source.raw,
                words.as_mut_slice().as_mut_ptr(),
                words_per_poly,
                coefficient_index,
                coefficient_count,
                layout.native_ptr(),
                &mut raw as *mut *mut GpuPreparedConstCoeffReadbackOpaque,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or_else(|| {
                "native prepared coefficient readback returned no plan".to_string()
            })?,
            source,
            words,
            layout,
            busy: AtomicBool::new(false),
        }))
    }

    pub fn bind_all(
        source: Arc<GpuDCRTPolyMatrix>,
        words_per_poly: usize,
        layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        Self::bind(
            source.clone(),
            words_per_poly,
            0,
            source.params().ring_dimension() as usize,
            layout,
        )
    }

    pub fn submit(self: &Arc<Self>) -> Result<GpuPreparedConstCoeffReadbackInFlight, String> {
        debug_assert!(!self.source.raw.is_null());
        self.release_completed_submission()?;
        if self.busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err("prepared coefficient readback has unretired work".to_string());
        }
        let status = unsafe { gpu_matrix_submit_const_coeff_readback(self.raw.as_ptr()) };
        if status != 0 {
            // The native submit may have queued copies whose completion it could
            // not record. The plan stays busy: only a proven terminal completion
            // releases the pinned destination before the plan itself drops.
            return Err(last_error_string());
        }
        Ok(GpuPreparedConstCoeffReadbackInFlight { plan: Arc::clone(self) })
    }

    pub fn words(&self) -> &[u64] {
        self.words.as_slice()
    }

    pub fn allocation_layout(&self) -> &super::PreparedPlanLayout {
        &self.layout
    }
}

impl GpuPreparedConstCoeffReadbackInFlight {
    /// Explicit host boundary for the readback results. Dropping the handle
    /// without this call is allowed and never waits for the device.
    pub fn wait(&self) -> Result<&[u64], String> {
        self.plan.wait()?;
        self.plan.busy.store(false, Ordering::Release);
        Ok(self.plan.words())
    }
}

impl Drop for GpuPreparedConstCoeffReadback {
    fn drop(&mut self) {
        if self.busy.load(Ordering::Acquire) {
            // The pinned destination may still be read by a queued copy, so its
            // ownership moves to the context-owned reclaimer behind freshly
            // recorded terminal events for every stream this plan submits on.
            // The reclaimer frees it once those events complete; a refused
            // handover keeps the allocation unreachable instead of freeing
            // memory in use. Neither path waits here.
            let pending =
                std::mem::replace(&mut self.words, PinnedHostBuffer::new(self.source.params()));
            unsafe {
                gpu_matrix_defer_const_coeff_readback_pinned_free(
                    self.raw.as_ptr(),
                    pending.into_raw().cast(),
                )
            };
        }
        unsafe { gpu_matrix_destroy_const_coeff_readback(self.raw.as_ptr()) };
    }
}

impl GpuPreparedConstCoeffReadback {
    /// Nonblocking release of a plan whose newest submission can be proven
    /// complete. The native query only answers for the terminal generation it
    /// recorded, so a partially submitted or unrecorded event keeps the plan
    /// busy instead of allowing an early reclaim.
    fn release_completed_submission(&self) -> Result<(), String> {
        if !self.busy.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut ready = 0;
        let status =
            unsafe { gpu_matrix_query_const_coeff_readback(self.raw.as_ptr(), &mut ready) };
        if status != 0 {
            return Err(last_error_string());
        }
        if ready != 0 {
            self.busy.store(false, Ordering::Release);
        }
        Ok(())
    }

    fn wait(&self) -> Result<(), String> {
        let status = unsafe { gpu_matrix_wait_const_coeff_readback(self.raw.as_ptr()) };
        if status != 0 { Err(last_error_string()) } else { Ok(()) }
    }
}

unsafe impl Send for GpuPreparedRnsUpload {}
unsafe impl Sync for GpuPreparedRnsUpload {}
unsafe impl Send for GpuPreparedRnsUploadInFlight {}
unsafe impl Sync for GpuPreparedRnsUploadInFlight {}

fn scalar_residue(value: &BigInt, modulus: u64) -> u64 {
    let residue = value.magnitude().iter_u64_digits().rev().fold(0u64, |acc, word| {
        (((u128::from(acc) << 64) | u128::from(word)) % u128::from(modulus)) as u64
    });
    if value.sign() == Sign::Minus && residue != 0 { modulus - residue } else { residue }
}

impl GpuPreparedRnsUpload {
    pub fn bind(
        target: Arc<GpuDCRTPolyMatrix>,
        bytes_per_poly: usize,
        format: i32,
        transform_to_eval: bool,
        layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let (rows, columns) = target.size();
        let source_evaluation = format == crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL;
        if target.is_ntt() != (source_evaluation || transform_to_eval) ||
            (source_evaluation && transform_to_eval)
        {
            return Err("prepared RNS upload destination format does not match transform".into());
        }
        let byte_count = rows
            .checked_mul(columns)
            .and_then(|count| count.checked_mul(bytes_per_poly))
            .ok_or_else(|| "prepared RNS upload input size overflow".to_string())?;
        let bytes = PinnedHostBuffer::zeroed(target.params(), byte_count);
        let ring_dimension = target.params().ring_dimension() as usize;
        let moduli =
            target.params().moduli().iter().take(target.level() + 1).copied().collect::<Box<_>>();
        let host_pointer = bytes.as_slice().as_ptr();
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_rns_upload(
                target.raw,
                host_pointer,
                bytes_per_poly,
                format,
                transform_to_eval,
                layout.native_ptr(),
                &mut raw as *mut *mut GpuPreparedRnsUploadOpaque,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw)
                .ok_or_else(|| "native prepared RNS upload returned no plan".to_string())?,
            layout,
            target,
            bytes: Mutex::new(bytes),
            bytes_per_poly,
            polynomial_count: rows * columns,
            ring_dimension,
            source_evaluation,
            moduli,
            busy: AtomicBool::new(false),
        }))
    }

    pub fn submit(self: &Arc<Self>, input: &[u8]) -> Result<GpuPreparedRnsUploadInFlight, String> {
        self.begin_submit()?;
        let copy_status = self.bytes.lock().map_err(|_| {
            self.busy.store(false, Ordering::Release);
            "prepared RNS upload staging lock is poisoned".to_string()
        });
        let mut bytes = match copy_status {
            Ok(bytes) => bytes,
            Err(error) => return Err(error),
        };
        if bytes.as_slice().len() != input.len() {
            self.busy.store(false, Ordering::Release);
            return Err("prepared RNS upload input size mismatch".to_string());
        }
        bytes.as_mut_slice().copy_from_slice(input);
        drop(bytes);
        self.submit_staged()
    }

    /// Submit a column shard from one canonical CPU staging payload.  The
    /// payload header and row-major layout remain owned by the caller; this
    /// method copies the strided shard directly into this plan's already
    /// allocated pinned buffer, so replay performs no temporary allocation.
    pub fn submit_columns(
        self: &Arc<Self>,
        input: &[u8],
        full_columns: usize,
        start: usize,
        end: usize,
    ) -> Result<GpuPreparedRnsUploadInFlight, String> {
        let rows = self.target.row_size();
        let local_columns = end
            .checked_sub(start)
            .ok_or_else(|| "prepared RNS upload column range is invalid".to_string())?;
        if end > full_columns || local_columns != self.target.col_size() {
            return Err("prepared RNS upload column range does not match target".into());
        }
        let payload_bytes = rows
            .checked_mul(full_columns)
            .and_then(|count| count.checked_mul(self.bytes_per_poly))
            .ok_or_else(|| "prepared RNS upload source size overflow".to_string())?;
        let header_bytes = input
            .len()
            .checked_sub(payload_bytes)
            .ok_or_else(|| "prepared RNS upload source payload is truncated".to_string())?;
        let shard_row_bytes = local_columns
            .checked_mul(self.bytes_per_poly)
            .ok_or_else(|| "prepared RNS upload target row size overflow".to_string())?;
        let expected_bytes = rows
            .checked_mul(shard_row_bytes)
            .ok_or_else(|| "prepared RNS upload target size overflow".to_string())?;
        let source_row_stride = full_columns
            .checked_mul(self.bytes_per_poly)
            .ok_or_else(|| "prepared RNS upload source row size overflow".to_string())?;
        let source_start = start
            .checked_mul(self.bytes_per_poly)
            .ok_or_else(|| "prepared RNS upload source offset overflow".to_string())?;
        self.begin_submit()?;
        let copy_status = self.bytes.lock().map_err(|_| {
            self.busy.store(false, Ordering::Release);
            "prepared RNS upload staging lock is poisoned".to_string()
        });
        let mut bytes = match copy_status {
            Ok(bytes) => bytes,
            Err(error) => return Err(error),
        };
        if bytes.as_slice().len() != expected_bytes {
            self.busy.store(false, Ordering::Release);
            return Err("prepared RNS upload target size mismatch".into());
        }
        for row in 0..rows {
            let source_offset = header_bytes
                .checked_add(
                    row.checked_mul(source_row_stride)
                        .and_then(|offset| offset.checked_add(source_start))
                        .ok_or_else(|| "prepared RNS upload source offset overflow".to_string())?,
                )
                .ok_or_else(|| "prepared RNS upload source offset overflow".to_string())?;
            let source_end = source_offset
                .checked_add(shard_row_bytes)
                .ok_or_else(|| "prepared RNS upload source range overflow".to_string())?;
            let target_offset = row
                .checked_mul(shard_row_bytes)
                .ok_or_else(|| "prepared RNS upload target offset overflow".to_string())?;
            bytes.as_mut_slice()[target_offset..target_offset + shard_row_bytes]
                .copy_from_slice(&input[source_offset..source_end]);
        }
        drop(bytes);
        self.submit_staged()
    }

    /// Stage canonical values into the fixed RNS upload layout and replay the
    /// prepared device command. The matrix shape and modulus list are fixed at
    /// bind time, so this performs no allocation or layout discovery.
    pub fn submit_values(
        self: &Arc<Self>,
        values: &[BigInt],
    ) -> Result<GpuPreparedRnsUploadInFlight, String> {
        self.begin_submit()?;
        let expected = match self.polynomial_count.checked_mul(self.ring_dimension) {
            Some(expected) => expected,
            None => {
                self.busy.store(false, Ordering::Release);
                return Err("prepared RNS upload value count overflow".to_string());
            }
        };
        if values.len() != expected {
            self.busy.store(false, Ordering::Release);
            return Err("prepared RNS upload value count mismatch".to_string());
        }
        let mut bytes = self.bytes.lock().map_err(|_| {
            self.busy.store(false, Ordering::Release);
            "prepared RNS upload staging lock is poisoned".to_string()
        })?;
        bytes.as_mut_slice().fill(0);
        for poly in 0..self.polynomial_count {
            for coefficient in 0..self.ring_dimension {
                let value = &values[poly * self.ring_dimension + coefficient];
                for (limb, modulus) in self.moduli.iter().copied().enumerate() {
                    let residue = scalar_residue(value, modulus);
                    let offset = poly * self.bytes_per_poly +
                        limb * self.ring_dimension * std::mem::size_of::<u64>() +
                        coefficient * std::mem::size_of::<u64>();
                    bytes.as_mut_slice()[offset..offset + std::mem::size_of::<u64>()]
                        .copy_from_slice(&residue.to_le_bytes());
                }
            }
        }
        drop(bytes);
        self.submit_staged()
    }

    /// Stage a constant in the declared source domain. This is the
    /// fixed host boundary for LiftIntegerToConstantPolynomial.
    pub fn submit_constant(
        self: &Arc<Self>,
        value: &BigInt,
    ) -> Result<GpuPreparedRnsUploadInFlight, String> {
        self.begin_submit()?;
        let mut bytes = self.bytes.lock().map_err(|_| {
            self.busy.store(false, Ordering::Release);
            "prepared RNS upload staging lock is poisoned".to_string()
        })?;
        bytes.as_mut_slice().fill(0);
        for poly in 0..self.polynomial_count {
            for coefficient in 0..if self.source_evaluation { self.ring_dimension } else { 1 } {
                for (limb, modulus) in self.moduli.iter().copied().enumerate() {
                    let residue = scalar_residue(value, modulus);
                    let offset = poly * self.bytes_per_poly +
                        limb * self.ring_dimension * std::mem::size_of::<u64>() +
                        coefficient * std::mem::size_of::<u64>();
                    bytes.as_mut_slice()[offset..offset + std::mem::size_of::<u64>()]
                        .copy_from_slice(&residue.to_le_bytes());
                }
            }
        }
        drop(bytes);
        self.submit_staged()
    }

    /// Stage packed little-endian coefficient bits for
    /// PackPolynomialCoefficients.
    pub fn submit_bits(
        self: &Arc<Self>,
        bits: &[bool],
        coefficient_bits: usize,
    ) -> Result<GpuPreparedRnsUploadInFlight, String> {
        let expected = self
            .polynomial_count
            .checked_mul(self.ring_dimension)
            .and_then(|count| count.checked_mul(coefficient_bits))
            .ok_or_else(|| "prepared RNS upload bit count overflow".to_string())?;
        if coefficient_bits == 0 || bits.len() != expected {
            return Err("prepared RNS upload bit count mismatch".to_string());
        }
        self.begin_submit()?;
        let mut bytes = self.bytes.lock().map_err(|_| {
            self.busy.store(false, Ordering::Release);
            "prepared RNS upload staging lock is poisoned".to_string()
        })?;
        bytes.as_mut_slice().fill(0);
        for poly in 0..self.polynomial_count {
            for coefficient in 0..self.ring_dimension {
                let start = (poly * self.ring_dimension + coefficient) * coefficient_bits;
                for (limb, modulus) in self.moduli.iter().copied().enumerate() {
                    let residue = bits[start..start + coefficient_bits].iter().rev().fold(
                        0u64,
                        |value, bit| {
                            ((u128::from(value) * 2 + u128::from(*bit)) % u128::from(modulus))
                                as u64
                        },
                    );
                    let offset = poly * self.bytes_per_poly +
                        limb * self.ring_dimension * std::mem::size_of::<u64>() +
                        coefficient * std::mem::size_of::<u64>();
                    bytes.as_mut_slice()[offset..offset + std::mem::size_of::<u64>()]
                        .copy_from_slice(&residue.to_le_bytes());
                }
            }
        }
        drop(bytes);
        self.submit_staged()
    }

    fn begin_submit(&self) -> Result<(), String> {
        self.release_completed_submission()?;
        if self.busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err("prepared RNS upload has unretired work".to_string());
        }
        Ok(())
    }

    fn submit_staged(self: &Arc<Self>) -> Result<GpuPreparedRnsUploadInFlight, String> {
        let status = unsafe { gpu_matrix_submit_rns_upload(self.raw.as_ptr()) };
        if status != 0 {
            // A native submit may have queued an earlier limb before failing.
            // The plan stays busy: only a proven terminal completion releases
            // the pinned staging allocation before the plan itself drops.
            return Err(last_error_string());
        }
        Ok(GpuPreparedRnsUploadInFlight { plan: Arc::clone(self) })
    }

    pub fn target(&self) -> &GpuDCRTPolyMatrix {
        &self.target
    }

    pub fn allocation_layout(&self) -> &super::PreparedPlanLayout {
        &self.layout
    }
}

unsafe impl Send for GpuPreparedCompactUpload {}
unsafe impl Sync for GpuPreparedCompactUpload {}
unsafe impl Send for GpuPreparedCompactUploadInFlight {}
unsafe impl Sync for GpuPreparedCompactUploadInFlight {}
unsafe impl Send for GpuPreparedSmallUpload {}
unsafe impl Sync for GpuPreparedSmallUpload {}
unsafe impl Send for GpuPreparedSmallUploadInFlight {}
unsafe impl Sync for GpuPreparedSmallUploadInFlight {}

impl GpuPreparedCompactUpload {
    pub fn allocation_layout(&self) -> &super::PreparedPlanLayout {
        &self.layout
    }

    pub fn bind(
        target: Arc<GpuDCRTPolyMatrix>,
        payload_capacity: usize,
        layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let mut payload = PinnedHostBuffer::zeroed(target.params(), payload_capacity);
        // Preparation reserves the workspace for the widest valid compact
        // codec of this target basis. Runtime artifacts may use a narrower
        // observed width, but binding a placeholder width would under-size
        // the native workspace and make replay correctness depend on input.
        let max_coeff_bits = target
            .params()
            .moduli()
            .iter()
            .map(|modulus| (u64::BITS - modulus.leading_zeros()) as usize)
            .try_fold(0usize, |total, bits| total.checked_add(bits))
            .ok_or("prepared compact upload codec width overflow")?;
        let max_coeff_bits = u16::try_from(max_coeff_bits)
            .map_err(|_| "prepared compact upload codec width exceeds u16")?;
        if max_coeff_bits == 0 {
            return Err("prepared compact upload codec width is zero".into());
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_compact_upload(
                target.raw,
                payload.as_mut_slice().as_mut_ptr(),
                payload_capacity,
                max_coeff_bits,
                layout.native_ptr(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("prepared compact upload returned no plan")?,
            target,
            payload: Mutex::new(payload),
            layout,
            busy: AtomicBool::new(false),
        }))
    }

    pub fn submit_artifact(
        self: &Arc<Self>,
        bytes: &[u8],
    ) -> Result<GpuPreparedCompactUploadInFlight, String> {
        self.release_completed()?;
        let (max_bits, payload) = GpuDCRTPolyMatrix::validate_compact_bytes(
            bytes,
            self.target.row_size(),
            self.target.col_size(),
            self.target.level(),
            self.target.params().ring_dimension() as usize,
            self.target.is_ntt(),
        )?;
        if self.busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err("prepared compact upload has unretired work".into());
        }
        let mut staging = self.payload.lock().map_err(|_| {
            self.busy.store(false, Ordering::Release);
            "prepared compact upload staging lock is poisoned".to_string()
        })?;
        if payload.len() > staging.as_slice().len() {
            self.busy.store(false, Ordering::Release);
            return Err("prepared compact upload payload exceeds fixed staging".into());
        }
        staging.as_mut_slice()[..payload.len()].copy_from_slice(payload);
        let status =
            unsafe { gpu_matrix_submit_compact_upload(self.raw.as_ptr(), max_bits, payload.len()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(GpuPreparedCompactUploadInFlight { plan: Arc::clone(self) })
    }

    fn release_completed(&self) -> Result<(), String> {
        if !self.busy.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut ready = 0;
        let status = unsafe { gpu_matrix_query_compact_upload(self.raw.as_ptr(), &mut ready) };
        if status != 0 {
            return Err(last_error_string());
        }
        if ready != 0 {
            self.busy.store(false, Ordering::Release);
        }
        Ok(())
    }

    pub fn is_complete(&self) -> Result<bool, String> {
        self.release_completed()?;
        Ok(!self.busy.load(Ordering::Acquire))
    }
}

impl GpuPreparedCompactUploadInFlight {
    pub fn wait(&self) -> Result<(), String> {
        let status = unsafe { gpu_matrix_wait_compact_upload(self.plan.raw.as_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        self.plan.busy.store(false, Ordering::Release);
        Ok(())
    }
}

impl Drop for GpuPreparedCompactUpload {
    fn drop(&mut self) {
        if self.busy.load(Ordering::Acquire) {
            let payload = match self.payload.get_mut() {
                Ok(payload) => payload,
                Err(poisoned) => poisoned.into_inner(),
            };
            let pending = std::mem::replace(payload, PinnedHostBuffer::new(self.target.params()));
            let _ = unsafe {
                gpu_matrix_defer_compact_upload_pinned_free(
                    self.raw.as_ptr(),
                    pending.into_raw().cast(),
                )
            };
        }
        unsafe { gpu_matrix_destroy_compact_upload(self.raw.as_ptr()) };
    }
}

impl GpuPreparedSmallUpload {
    pub fn allocation_layout(&self) -> &super::PreparedPlanLayout {
        &self.layout
    }

    pub fn bind(
        target: Arc<GpuSmallMatrix>,
        layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let mut payload = PinnedHostBuffer::zeroed(&target.params, target.resident_payload_bytes());
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_small_upload(
                target.raw,
                payload.as_mut_slice().as_mut_ptr(),
                payload.as_slice().len(),
                layout.native_ptr(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("prepared small upload returned no plan")?,
            target,
            payload: Mutex::new(payload),
            layout,
            busy: AtomicBool::new(false),
        }))
    }

    pub fn submit_payload(
        self: &Arc<Self>,
        payload: &[u8],
    ) -> Result<GpuPreparedSmallUploadInFlight, String> {
        self.release_completed()?;
        if payload.len() != self.target.resident_payload_bytes() {
            return Err("prepared small upload payload length mismatch".into());
        }
        if self.busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err("prepared small upload has unretired work".into());
        }
        let mut staging = self.payload.lock().map_err(|_| {
            self.busy.store(false, Ordering::Release);
            "prepared small upload staging lock is poisoned".to_string()
        })?;
        staging.as_mut_slice().copy_from_slice(payload);
        let status = unsafe { gpu_matrix_submit_small_upload(self.raw.as_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(GpuPreparedSmallUploadInFlight { plan: Arc::clone(self) })
    }

    fn release_completed(&self) -> Result<(), String> {
        if !self.busy.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut ready = 0;
        let status = unsafe { gpu_matrix_query_small_upload(self.raw.as_ptr(), &mut ready) };
        if status != 0 {
            return Err(last_error_string());
        }
        if ready != 0 {
            self.busy.store(false, Ordering::Release);
        }
        Ok(())
    }

    pub fn is_complete(&self) -> Result<bool, String> {
        self.release_completed()?;
        Ok(!self.busy.load(Ordering::Acquire))
    }
}

impl GpuPreparedSmallUploadInFlight {
    pub fn wait(&self) -> Result<(), String> {
        let status = unsafe { gpu_matrix_wait_small_upload(self.plan.raw.as_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        self.plan.busy.store(false, Ordering::Release);
        Ok(())
    }
}

impl Drop for GpuPreparedSmallUpload {
    fn drop(&mut self) {
        if self.busy.load(Ordering::Acquire) {
            let payload = match self.payload.get_mut() {
                Ok(payload) => payload,
                Err(poisoned) => poisoned.into_inner(),
            };
            let pending = std::mem::replace(payload, PinnedHostBuffer::new(self.target.params()));
            let _ = unsafe {
                gpu_matrix_defer_small_upload_pinned_free(
                    self.raw.as_ptr(),
                    pending.into_raw().cast(),
                )
            };
        }
        unsafe { gpu_matrix_destroy_small_upload(self.raw.as_ptr()) };
    }
}

impl GpuPreparedRnsUploadInFlight {
    /// Explicit host boundary for the upload. Dropping the handle without this
    /// call is allowed and never waits for the device.
    pub fn wait(&self) -> Result<(), String> {
        self.plan.wait()?;
        self.plan.busy.store(false, Ordering::Release);
        Ok(())
    }

    pub fn target(&self) -> &GpuDCRTPolyMatrix {
        self.plan.target()
    }
}

impl GpuPreparedRnsReconstruction {
    pub fn bind_all(
        source: Arc<GpuDCRTPolyMatrix>,
        layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        Self::bind(source.clone(), 0, source.params().ring_dimension() as usize, layout)
    }

    pub fn bind(
        source: Arc<GpuDCRTPolyMatrix>,
        coefficient_index: usize,
        coefficient_count: usize,
        layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let level = source.level();
        let ring_dimension = source.params().ring_dimension() as usize;
        let coefficient_end =
            coefficient_index.checked_add(coefficient_count).ok_or_else(|| {
                "prepared CRT reconstruction coefficient range is invalid".to_string()
            })?;
        if coefficient_end > ring_dimension {
            return Err("prepared CRT reconstruction coefficient range is invalid".to_string());
        }
        let limb_count = level + 1;
        let result_count = source
            .size()
            .0
            .checked_mul(source.size().1)
            .and_then(|count| count.checked_mul(coefficient_count))
            .ok_or_else(|| "prepared CRT reconstruction output size overflow".to_string())?;
        let words_per_poly = limb_count
            .checked_mul(coefficient_count)
            .ok_or_else(|| "prepared CRT reconstruction word size overflow".to_string())?;
        let readback = GpuPreparedConstCoeffReadback::bind(
            Arc::clone(&source),
            words_per_poly,
            coefficient_index,
            coefficient_count,
            layout.clone(),
        )?;
        let modulus = source.params.modulus_for_level(level);
        let reconstruction = source.params.reconstruct_coeffs_for_level(level).into_boxed_slice();
        let values = (0..result_count)
            .map(|_| num_bigint::BigUint::zero())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Arc::new(Self {
            readback,
            layout,
            reconstruction,
            modulus,
            coefficient_count,
            values: Mutex::new(values),
        }))
    }

    pub fn submit(self: &Arc<Self>) -> Result<GpuPreparedRnsReconstructionInFlight, String> {
        let readback = self.readback.submit()?;
        Ok(GpuPreparedRnsReconstructionInFlight { plan: Arc::clone(self), readback })
    }

    pub fn with_values<R>(&self, function: impl FnOnce(&[num_bigint::BigUint]) -> R) -> R {
        let values = self.values.lock().expect("prepared CRT result lock poisoned");
        function(&values)
    }

    pub fn allocation_layout(&self) -> &super::PreparedPlanLayout {
        &self.layout
    }
}

impl GpuPreparedRnsReconstructionInFlight {
    pub fn wait(&self) -> Result<(), String> {
        let words = self.readback.wait()?;
        let mut values =
            self.plan.values.lock().map_err(|_| "prepared CRT result lock poisoned".to_string())?;
        let coefficient_count = self.plan.coefficient_count;
        let words_per_poly = self.plan.reconstruction.len() * coefficient_count;
        for (poly_index, output) in values.chunks_exact_mut(coefficient_count).enumerate() {
            let poly_words = &words[poly_index * words_per_poly..(poly_index + 1) * words_per_poly];
            for (coefficient, value) in output.iter_mut().enumerate() {
                value.set_zero();
                for (limb, basis) in self.plan.reconstruction.iter().enumerate() {
                    *value += basis *
                        num_bigint::BigUint::from(
                            poly_words[limb * coefficient_count + coefficient],
                        );
                }
                *value %= &self.plan.modulus;
            }
        }
        Ok(())
    }

    pub fn with_values<R>(&self, function: impl FnOnce(&[num_bigint::BigUint]) -> R) -> R {
        self.plan.with_values(function)
    }
}

impl Drop for GpuPreparedRnsUpload {
    fn drop(&mut self) {
        if self.busy.load(Ordering::Acquire) {
            // The pinned staging allocation may still be read by a queued
            // upload copy, so its ownership moves to the context-owned
            // reclaimer behind freshly recorded terminal events for every
            // stream this plan submits on. The reclaimer frees it once those
            // events complete; a refused handover keeps the allocation
            // unreachable instead of freeing memory in use. Neither path waits
            // here, and a poisoned staging lock cannot panic during shutdown.
            let host = self.bytes.get_mut().unwrap_or_else(|poisoned| poisoned.into_inner());
            let pending = std::mem::replace(host, PinnedHostBuffer::new(self.target.params()));
            unsafe {
                gpu_matrix_defer_rns_upload_pinned_free(self.raw.as_ptr(), pending.into_raw())
            };
        }
        unsafe { gpu_matrix_destroy_rns_upload(self.raw.as_ptr()) };
    }
}

impl GpuPreparedRnsUpload {
    /// Nonblocking release of a plan whose newest submission can be proven
    /// complete. The native query only answers for the terminal generation it
    /// recorded, so a partially submitted or unrecorded event keeps the plan
    /// busy instead of allowing an early reclaim.
    fn release_completed_submission(&self) -> Result<(), String> {
        if !self.busy.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut ready = 0;
        let status = unsafe { gpu_matrix_query_rns_upload(self.raw.as_ptr(), &mut ready) };
        if status != 0 {
            return Err(last_error_string());
        }
        if ready != 0 {
            self.busy.store(false, Ordering::Release);
        }
        Ok(())
    }

    fn wait(&self) -> Result<(), String> {
        let status = unsafe { gpu_matrix_wait_rns_upload(self.raw.as_ptr()) };
        if status != 0 { Err(last_error_string()) } else { Ok(()) }
    }
}

unsafe impl Send for GpuPreparedSampling {}
unsafe impl Sync for GpuPreparedSampling {}
unsafe impl Send for GpuPreparedSamplingInFlight {}
unsafe impl Sync for GpuPreparedSamplingInFlight {}

impl GpuPreparedSampling {
    /// # Safety
    /// Preparation only, with exclusive access to this sampler. The cutoff
    /// must outlive the sampler and belong to the same exclusive invocation.
    pub(crate) unsafe fn bind_preimage_acceptance(
        &self,
        cutoff: &super::GpuPreparedPreimageCutoff,
        job: usize,
    ) -> Result<(), String> {
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_preimage_mask_sampling(
                self.raw.as_ptr(),
                cutoff.native_raw(),
                job,
            )
        };
        if status == 0 { Ok(()) } else { Err(last_error_string()) }
    }

    pub fn bind(
        output: Arc<GpuDCRTPolyMatrix>,
        dist: GpuMatrixSampleDist,
        sigma: f64,
        max_coefficient_bound: u64,
        full_ncol: usize,
        col_offset: usize,
        range: Option<GpuPreparedRange>,
    ) -> Result<Arc<Self>, String> {
        let range = range.map(|range| GpuMatrixRange {
            row_start: range.rows.start,
            row_end: range.rows.end,
            column_start: range.columns.start,
            column_end: range.columns.end,
        });
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_sampling(
                output.raw,
                dist.as_ffi(),
                sigma,
                max_coefficient_bound,
                output.params().modulus().to_u64().unwrap_or(0),
                full_ncol,
                col_offset,
                range.as_ref().map_or(std::ptr::null(), |range| range),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared sampling returned no plan")?,
            output,
        }))
    }

    /// Bind sampling using the exact metadata-only descriptor produced during
    /// warmup. A native mismatch is an error; it cannot silently re-plan.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_with_layout(
        output: Arc<GpuDCRTPolyMatrix>,
        dist: GpuMatrixSampleDist,
        sigma: f64,
        max_coefficient_bound: u64,
        full_ncol: usize,
        col_offset: usize,
        range: Option<GpuPreparedRange>,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let range = range.map(|range| GpuMatrixRange {
            row_start: range.rows.start,
            row_end: range.rows.end,
            column_start: range.columns.start,
            column_end: range.columns.end,
        });
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_matrix_prepare_sampling_with_layout(
                output.raw,
                dist.as_ffi(),
                sigma,
                max_coefficient_bound,
                output.params().modulus().to_u64().unwrap_or(0),
                full_ncol,
                col_offset,
                range.as_ref().map_or(std::ptr::null(), |range| range),
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared sampling returned no plan")?,
            output,
        }))
    }

    pub fn submit(
        self: &Arc<Self>,
        seed: crate::poly::dcrt::gpu::GpuRngSeed,
    ) -> Result<GpuPreparedSamplingInFlight, String> {
        let status = unsafe { gpu_matrix_submit_sampling(self.raw.as_ptr(), seed) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(GpuPreparedSamplingInFlight { plan: Arc::clone(self) })
    }

    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        &self.output
    }
}

impl GpuPreparedSamplingInFlight {
    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        self.plan.output()
    }
}

impl GpuPreparedHashSample {
    pub fn bind(
        output: Arc<GpuDCRTPolyMatrix>,
        key: [u8; 32],
        tag: &[u8],
        dist: GpuMatrixSampleDist,
        sigma: f64,
        max_coefficient_bound: u64,
        full_ncol: usize,
        col_offset: usize,
        range: Option<GpuPreparedRange>,
    ) -> Result<Arc<Self>, String> {
        let seed = crate::sampler::gpu::hash_seed_for_matrix::<keccak_asm::Keccak256>(key, tag);
        Ok(Arc::new(Self {
            sampler: GpuPreparedSampling::bind(
                output,
                dist,
                sigma,
                max_coefficient_bound,
                full_ncol,
                col_offset,
                range,
            )?,
            tag: tag.to_vec().into_boxed_slice(),
            seed,
        }))
    }

    pub fn submit(&self) -> Result<GpuPreparedSamplingInFlight, String> {
        self.sampler.submit(self.seed)
    }

    pub fn submit_seed(
        &self,
        seed: crate::poly::dcrt::gpu::GpuRngSeed,
    ) -> Result<GpuPreparedSamplingInFlight, String> {
        self.sampler.submit(seed)
    }

    /// Replay the fixed geometry with a fresh runtime key.  The tag and all
    /// matrix launch metadata remain immutable; only the derived seed changes.
    pub fn submit_key(&self, key: [u8; 32]) -> Result<GpuPreparedSamplingInFlight, String> {
        let seed =
            crate::sampler::gpu::hash_seed_for_matrix::<keccak_asm::Keccak256>(key, &self.tag);
        self.sampler.submit(seed)
    }

    pub fn submit_key_with_tag(
        &self,
        key: [u8; 32],
        tag: &[u8],
    ) -> Result<GpuPreparedSamplingInFlight, String> {
        let seed = crate::sampler::gpu::hash_seed_for_matrix::<keccak_asm::Keccak256>(key, tag);
        self.sampler.submit(seed)
    }

    pub fn bind_with_layout(
        output: Arc<GpuDCRTPolyMatrix>,
        key: [u8; 32],
        tag: &[u8],
        dist: GpuMatrixSampleDist,
        sigma: f64,
        max_coefficient_bound: u64,
        full_ncol: usize,
        col_offset: usize,
        range: Option<GpuPreparedRange>,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let seed = crate::sampler::gpu::hash_seed_for_matrix::<keccak_asm::Keccak256>(key, tag);
        Ok(Arc::new(Self {
            sampler: GpuPreparedSampling::bind_with_layout(
                output,
                dist,
                sigma,
                max_coefficient_bound,
                full_ncol,
                col_offset,
                range,
                plan_layout,
            )?,
            tag: tag.to_vec().into_boxed_slice(),
            seed,
        }))
    }

    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        self.sampler.output()
    }
}

impl GpuPreparedHashCompactCommand {
    #[allow(clippy::too_many_arguments)]
    pub fn bind_with_layout(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuSmallMatrix>,
        key: [u8; 32],
        tag: &[u8],
        dist: GpuMatrixSampleDist,
        sigma: f64,
        max_coefficient_bound: u64,
        full_ncol: usize,
        col_offset: usize,
        small: bool,
        digit_count: Option<usize>,
        layout: &GpuPreparedHashCompactLayout,
    ) -> Result<Arc<Self>, String> {
        let hash = GpuPreparedHashSample::bind_with_layout(
            Arc::clone(&source),
            key,
            tag,
            dist,
            sigma,
            max_coefficient_bound,
            full_ncol,
            col_offset,
            None,
            &layout.sampling,
        )?;
        let decomposition = GpuPreparedCompactDecompose::bind_with_layout(
            source,
            output,
            small,
            digit_count,
            &layout.decomposition,
        )?;
        Ok(Arc::new(Self { hash, decomposition }))
    }

    pub fn submit_key_with_tag(&self, key: [u8; 32], tag: &[u8]) -> Result<(), String> {
        self.hash.submit_key_with_tag(key, tag)?;
        self.decomposition.submit()
    }
}

impl Drop for GpuPreparedSampling {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_sampling(self.raw.as_ptr()) };
    }
}

pub(super) fn conversion_arguments(
    source: &GpuDCRTPolyMatrix,
    target_params: &crate::poly::dcrt::gpu::GpuDCRTPolyParams,
    conversion: GpuMatrixModulusConversion,
) -> Result<(i32, u64, Vec<u64>, Vec<u64>), String> {
    conversion.validate(source.params(), source.level(), target_params)?;
    let source_primes = &source.params().moduli()[..=source.level()];
    let target_primes = target_params.moduli();
    let (mode, plaintext_modulus) = match conversion {
        GpuMatrixModulusConversion::Reduce => (0, 0),
        GpuMatrixModulusConversion::Round => (1, 0),
        GpuMatrixModulusConversion::CenteredExtend => (2, 0),
        GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus } => (3, plaintext_modulus),
    };
    let division_inverses = target_primes
        .iter()
        .map(|prime| {
            if mode == 0 || mode == 2 {
                return Ok(1);
            }
            let product = source_primes
                .iter()
                .filter(|p| !target_primes.contains(p))
                .fold(1u64, |product, p| {
                    ((product as u128 * (*p % prime) as u128) % *prime as u128) as u64
                });
            crate::utils::mod_inverse(product, *prime)
                .ok_or("discarded CRT product is not invertible")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_scales = source_primes
        .iter()
        .map(|prime| {
            if mode != 3 {
                Ok(1)
            } else {
                crate::utils::mod_inverse(plaintext_modulus % prime, *prime)
                    .ok_or("plaintext modulus is not invertible")
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((mode, plaintext_modulus, division_inverses, input_scales))
}

struct GpuPreparedContextAnchors {
    source: crate::poly::dcrt::gpu::GpuDCRTPolyParams,
    target: crate::poly::dcrt::gpu::GpuDCRTPolyParams,
}

pub struct GpuPreparedModulusConversion {
    raw: NonNull<GpuPreparedModulusConversionOpaque>,
    // Keep both native execution contexts alive for the lifetime of the plan.
    contexts: GpuPreparedContextAnchors,
    source_rows: usize,
    source_columns: usize,
    source_level: usize,
    target_rows: usize,
    target_columns: usize,
    target_level: usize,
    source_evaluation: bool,
    target_evaluation: bool,
}

pub struct GpuPreparedModulusCommand {
    plan: Arc<GpuPreparedModulusConversion>,
    source_owner: Arc<GpuDCRTPolyMatrix>,
    // The enclosing prepared execution retains the output owner and passes it
    // by borrow at submission, so this pointer cannot outlive that owner.
    target: *mut crate::poly::dcrt::gpu::GpuMatrixOpaque,
    rows: usize,
    columns: usize,
    target_rows: usize,
}

unsafe impl Send for GpuPreparedModulusCommand {}
unsafe impl Sync for GpuPreparedModulusCommand {}

/// An output-bound forward NTT. The native plan retains the exact matrix
/// address and rectangle, so submission cannot silently target another owner.
pub struct GpuPreparedTransform {
    raw: NonNull<GpuMatrixTransformPlanOpaque>,
    matrix: *mut crate::poly::dcrt::gpu::GpuMatrixOpaque,
    forward: bool,
}

unsafe impl Send for GpuPreparedTransform {}
unsafe impl Sync for GpuPreparedTransform {}

impl GpuPreparedTransform {
    pub fn new_forward(matrix_owner: &GpuDCRTPolyMatrix) -> Result<Self, String> {
        Self::new(matrix_owner, true)
    }

    pub fn new_inverse(matrix_owner: &GpuDCRTPolyMatrix) -> Result<Self, String> {
        Self::new(matrix_owner, false)
    }

    fn new(matrix_owner: &GpuDCRTPolyMatrix, forward: bool) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_ntt_plan(matrix_owner.raw, std::ptr::null(), forward, &mut raw)
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("native prepared NTT returned no plan")?,
            matrix: matrix_owner.raw,
            forward,
        })
    }

    /// Bind a transform to the descriptor produced during pure preparation.
    /// Native binding rejects a stream or geometry mismatch and never chooses
    /// a replacement plan.
    pub fn new_with_layout(
        matrix_owner: &GpuDCRTPolyMatrix,
        forward: bool,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_ntt_plan_with_layout(
                matrix_owner.raw,
                std::ptr::null(),
                forward,
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("native prepared NTT returned no plan")?,
            matrix: matrix_owner.raw,
            forward,
        })
    }

    pub(crate) fn output_is_ntt(&self) -> bool {
        self.forward
    }

    pub fn submit(&self, matrix: &mut GpuDCRTPolyMatrix) -> Result<(), String> {
        self.submit_shared(matrix)?;
        matrix.is_ntt = self.forward;
        Ok(())
    }

    /// Submit the fixed transform against an owner held behind an `Arc`.
    /// Native format metadata is updated by the CUDA plan; callers use this
    /// form when the Rust owner already has the planned destination format.
    pub fn submit_shared(&self, matrix: &GpuDCRTPolyMatrix) -> Result<(), String> {
        if self.matrix != matrix.raw {
            return Err("prepared NTT matrix owner mismatch".into());
        }
        let status = unsafe { gpu_matrix_submit_ntt_plan(self.raw.as_ptr(), matrix.raw) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }
}

impl Drop for GpuPreparedTransform {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_ntt_plan(self.raw.as_ptr()) };
    }
}

/// Owner-bound transpose replay. The source, destination, and rectangle are
/// fixed during binding, so execution does not rediscover a view or owner.
pub struct GpuPreparedTranspose {
    raw: NonNull<GpuPreparedTransposeOpaque>,
    source: Arc<GpuDCRTPolyMatrix>,
    output: Arc<GpuDCRTPolyMatrix>,
    layout: PreparedRectLayout,
}

unsafe impl Send for GpuPreparedTranspose {}
unsafe impl Sync for GpuPreparedTranspose {}

impl GpuPreparedTranspose {
    pub fn bind(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        view: Option<GpuPreparedView>,
    ) -> Result<Arc<Self>, String> {
        if source.params() != output.params() ||
            source.level() != output.level() ||
            source.is_ntt() != output.is_ntt() ||
            Arc::ptr_eq(&source, &output)
        {
            return Err("prepared transpose owner contract mismatch".to_string());
        }
        let view = view.map_or(
            GpuMatrixBatchView {
                left: GpuMatrixRange {
                    row_start: 0,
                    row_end: source.row_size(),
                    column_start: 0,
                    column_end: source.col_size(),
                },
                right: GpuMatrixRange {
                    row_start: 0,
                    row_end: source.row_size(),
                    column_start: 0,
                    column_end: source.col_size(),
                },
                output: GpuMatrixRange {
                    row_start: 0,
                    row_end: output.row_size(),
                    column_start: 0,
                    column_end: output.col_size(),
                },
            },
            |view| GpuMatrixBatchView {
                left: GpuMatrixRange {
                    row_start: view.left.rows.start,
                    row_end: view.left.rows.end,
                    column_start: view.left.columns.start,
                    column_end: view.left.columns.end,
                },
                right: GpuMatrixRange {
                    row_start: view.right.rows.start,
                    row_end: view.right.rows.end,
                    column_start: view.right.columns.start,
                    column_end: view.right.columns.end,
                },
                output: GpuMatrixRange {
                    row_start: view.output.rows.start,
                    row_end: view.output.rows.end,
                    column_start: view.output.columns.start,
                    column_end: view.output.columns.end,
                },
            },
        );
        let rows = view.left.row_end - view.left.row_start;
        let columns = view.left.column_end - view.left.column_start;
        let device = source
            .params()
            .device_ids()
            .first()
            .copied()
            .ok_or_else(|| "prepared transpose has no device".to_string())?;
        let layout = PreparedRectLayout::query_transpose(
            source.params().ring_dimension() as usize,
            source.level() + 1,
            rows,
            columns,
            device,
        )?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_transpose(
                output.raw,
                source.raw,
                &view,
                &mut raw as *mut *mut GpuPreparedTransposeOpaque,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared transpose returned no plan")?,
            source,
            output,
            layout,
        }))
    }

    pub fn bind_with_layout(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        view: Option<GpuPreparedView>,
        plan_layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        if source.params() != output.params() ||
            source.level() != output.level() ||
            source.is_ntt() != output.is_ntt() ||
            Arc::ptr_eq(&source, &output)
        {
            return Err("prepared transpose owner contract mismatch".to_string());
        }
        let native_view = view.as_ref().map(|view| GpuMatrixBatchView {
            left: GpuMatrixRange {
                row_start: view.left.rows.start,
                row_end: view.left.rows.end,
                column_start: view.left.columns.start,
                column_end: view.left.columns.end,
            },
            right: GpuMatrixRange {
                row_start: view.right.rows.start,
                row_end: view.right.rows.end,
                column_start: view.right.columns.start,
                column_end: view.right.columns.end,
            },
            output: GpuMatrixRange {
                row_start: view.output.rows.start,
                row_end: view.output.rows.end,
                column_start: view.output.columns.start,
                column_end: view.output.columns.end,
            },
        });
        let rows = native_view
            .as_ref()
            .map_or(source.row_size(), |view| view.left.row_end - view.left.row_start);
        let columns = native_view
            .as_ref()
            .map_or(source.col_size(), |view| view.left.column_end - view.left.column_start);
        let layout = plan_layout.rectangular_layout(
            rows,
            columns,
            source.params().ring_dimension() as usize,
            source.level() + 1,
            2,
        )?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_transpose_with_layout(
                output.raw,
                source.raw,
                native_view.as_ref().map_or(std::ptr::null(), |view| view),
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared transpose returned no plan")?,
            source,
            output,
            layout,
        }))
    }

    pub fn layout(&self) -> &PreparedRectLayout {
        &self.layout
    }

    pub fn submit(&self) -> Result<(), String> {
        debug_assert_ne!(self.source.raw, self.output.raw);
        let status = unsafe { gpu_matrix_submit_transpose(self.raw.as_ptr()) };
        if status != 0 { Err(last_error_string()) } else { Ok(()) }
    }

    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        &self.output
    }

    pub fn source(&self) -> &GpuDCRTPolyMatrix {
        &self.source
    }
}

impl Drop for GpuPreparedTranspose {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_transpose(self.raw.as_ptr()) };
    }
}

/// Owner-bound centered rebase replay. The destination basis and rectangle are
/// fixed by the two retained matrix owners and the bind-time view.
pub struct GpuPreparedCenteredRebase {
    raw: NonNull<GpuPreparedCenteredRebaseOpaque>,
    source: Arc<GpuDCRTPolyMatrix>,
    output: Arc<GpuDCRTPolyMatrix>,
}

unsafe impl Send for GpuPreparedCenteredRebase {}
unsafe impl Sync for GpuPreparedCenteredRebase {}

impl GpuPreparedCenteredRebase {
    pub fn bind(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        view: Option<GpuPreparedView>,
    ) -> Result<Arc<Self>, String> {
        if source.level() != 0 ||
            source.is_ntt() ||
            source.params().ring_dimension() != output.params().ring_dimension() ||
            source.params().execution_owner_id() != output.params().execution_owner_id() ||
            Arc::ptr_eq(&source, &output)
        {
            return Err("prepared centered rebase owner contract mismatch".to_string());
        }
        let view = view.map_or(
            GpuMatrixBatchView {
                left: GpuMatrixRange {
                    row_start: 0,
                    row_end: source.row_size(),
                    column_start: 0,
                    column_end: source.col_size(),
                },
                right: GpuMatrixRange {
                    row_start: 0,
                    row_end: source.row_size(),
                    column_start: 0,
                    column_end: source.col_size(),
                },
                output: GpuMatrixRange {
                    row_start: 0,
                    row_end: output.row_size(),
                    column_start: 0,
                    column_end: output.col_size(),
                },
            },
            |view| GpuMatrixBatchView {
                left: GpuMatrixRange {
                    row_start: view.left.rows.start,
                    row_end: view.left.rows.end,
                    column_start: view.left.columns.start,
                    column_end: view.left.columns.end,
                },
                right: GpuMatrixRange {
                    row_start: view.right.rows.start,
                    row_end: view.right.rows.end,
                    column_start: view.right.columns.start,
                    column_end: view.right.columns.end,
                },
                output: GpuMatrixRange {
                    row_start: view.output.rows.start,
                    row_end: view.output.rows.end,
                    column_start: view.output.columns.start,
                    column_end: view.output.columns.end,
                },
            },
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_centered_rebase(
                output.raw,
                source.raw,
                &view,
                &mut raw as *mut *mut GpuPreparedCenteredRebaseOpaque,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared centered rebase returned no plan")?,
            source,
            output,
        }))
    }

    pub fn bind_with_layout(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        view: Option<GpuPreparedView>,
        plan_layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        if source.level() != 0 ||
            source.is_ntt() ||
            source.params().ring_dimension() != output.params().ring_dimension() ||
            source.params().execution_owner_id() != output.params().execution_owner_id() ||
            Arc::ptr_eq(&source, &output)
        {
            return Err("prepared centered rebase owner contract mismatch".to_string());
        }
        let native_view = view.as_ref().map(|view| GpuMatrixBatchView {
            left: GpuMatrixRange {
                row_start: view.left.rows.start,
                row_end: view.left.rows.end,
                column_start: view.left.columns.start,
                column_end: view.left.columns.end,
            },
            right: GpuMatrixRange {
                row_start: view.right.rows.start,
                row_end: view.right.rows.end,
                column_start: view.right.columns.start,
                column_end: view.right.columns.end,
            },
            output: GpuMatrixRange {
                row_start: view.output.rows.start,
                row_end: view.output.rows.end,
                column_start: view.output.columns.start,
                column_end: view.output.columns.end,
            },
        });
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_centered_rebase_with_layout(
                output.raw,
                source.raw,
                native_view.as_ref().map_or(std::ptr::null(), |view| view),
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared centered rebase returned no plan")?,
            source,
            output,
        }))
    }

    pub fn submit(&self) -> Result<(), String> {
        debug_assert_ne!(self.source.raw, self.output.raw);
        let status = unsafe { gpu_matrix_submit_centered_rebase(self.raw.as_ptr()) };
        if status != 0 { Err(last_error_string()) } else { Ok(()) }
    }

    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        &self.output
    }

    pub fn source(&self) -> &GpuDCRTPolyMatrix {
        &self.source
    }
}

impl Drop for GpuPreparedCenteredRebase {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_centered_rebase(self.raw.as_ptr()) };
    }
}

/// Owner-bound gadget decomposition. The destination shape and whether the
/// compact or full decomposition is used are fixed at bind time.
pub struct GpuPreparedGadgetDecompose {
    raw: NonNull<GpuPreparedGadgetDecomposeOpaque>,
    source: Arc<GpuDCRTPolyMatrix>,
    output: Arc<GpuDCRTPolyMatrix>,
}

unsafe impl Send for GpuPreparedGadgetDecompose {}
unsafe impl Sync for GpuPreparedGadgetDecompose {}

impl GpuPreparedGadgetDecompose {
    pub fn bind(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        small: bool,
        dropped_moduli: usize,
    ) -> Result<Arc<Self>, String> {
        if Arc::ptr_eq(&source, &output) || source.params() != output.params() {
            return Err("prepared gadget decomposition owner contract mismatch".to_string());
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_gadget_decompose(
                source.raw,
                source.params().base_bits(),
                output.raw,
                i32::from(small),
                dropped_moduli,
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or_else(|| {
                "native prepared gadget decomposition returned no plan".to_string()
            })?,
            source,
            output,
        }))
    }

    pub fn bind_with_layout(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        small: bool,
        dropped_moduli: usize,
        plan_layout: super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        if Arc::ptr_eq(&source, &output) || source.params() != output.params() {
            return Err("prepared gadget decomposition owner contract mismatch".to_string());
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_gadget_decompose_with_layout(
                source.raw,
                source.params().base_bits(),
                output.raw,
                i32::from(small),
                dropped_moduli,
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw)
                .ok_or("native prepared gadget decomposition returned no plan")?,
            source,
            output,
        }))
    }

    pub fn submit(&self) -> Result<(), String> {
        debug_assert_ne!(self.source.raw, self.output.raw);
        let status = unsafe { gpu_matrix_submit_gadget_decompose(self.raw.as_ptr()) };
        if status != 0 { Err(last_error_string()) } else { Ok(()) }
    }

    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        &self.output
    }
}

impl Drop for GpuPreparedGadgetDecompose {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_gadget_decompose(self.raw.as_ptr()) };
    }
}

unsafe impl Send for GpuPreparedModulusConversion {}
unsafe impl Sync for GpuPreparedModulusConversion {}

impl GpuPreparedModulusConversion {
    pub fn bind(
        plan: Arc<Self>,
        source: Arc<GpuDCRTPolyMatrix>,
        target: &GpuDCRTPolyMatrix,
    ) -> Result<GpuPreparedModulusCommand, String> {
        if plan.contexts.source.context_identity() != source.params().context_identity() ||
            plan.contexts.target.context_identity() != target.params().context_identity() ||
            source.params().moduli() != plan.contexts.source.moduli() ||
            target.params().moduli() != plan.contexts.target.moduli() ||
            source.params().device_ids() != plan.contexts.source.device_ids() ||
            target.params().device_ids() != plan.contexts.target.device_ids() ||
            source.row_size() != plan.source_rows ||
            source.col_size() != plan.source_columns ||
            source.level() != plan.source_level ||
            target.row_size() != plan.target_rows ||
            target.col_size() != plan.target_columns ||
            target.level() != plan.target_level ||
            source.is_ntt() != plan.source_evaluation ||
            target.is_ntt() != plan.target_evaluation
        {
            return Err("prepared conversion binding does not match its plan".into());
        }
        let rows = source.row_size();
        let columns = source.col_size();
        Ok(GpuPreparedModulusCommand {
            plan,
            source_owner: source,
            target: target.raw,
            rows,
            columns,
            target_rows: target.row_size(),
        })
    }

    /// Prepare a full-matrix conversion.  The source must be coefficient-domain
    /// when the conversion needs rounding or extension; callers can retain an
    /// eval-domain source separately and transform their fixed staging owner.
    pub fn new(
        source: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        conversion: GpuMatrixModulusConversion,
    ) -> Result<Self, String> {
        let (mode, plaintext_modulus, division_inverses, input_scales) =
            conversion_arguments(source, target.params(), conversion)?;
        if mode != 0 && source.is_ntt() {
            return Err("prepared rounding conversion requires coefficient-domain source".into());
        }
        if source.row_size() != target.row_size() || source.col_size() != target.col_size() {
            return Err("prepared conversion source and destination shapes differ".into());
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_modulus_conversion(
                source.raw,
                target.raw,
                mode,
                division_inverses.as_ptr(),
                division_inverses.len(),
                plaintext_modulus,
                input_scales.as_ptr(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        let raw = NonNull::new(raw).ok_or("native prepared conversion returned no plan")?;
        Ok(Self {
            raw,
            contexts: GpuPreparedContextAnchors {
                source: source.params().clone(),
                target: target.params().clone(),
            },
            source_rows: source.row_size(),
            source_columns: source.col_size(),
            source_level: source.level(),
            target_rows: target.row_size(),
            target_columns: target.col_size(),
            target_level: target.level(),
            source_evaluation: source.is_ntt(),
            target_evaluation: target.is_ntt(),
        })
    }

    /// Prepare a coefficient/modulus conversion while consuming the exact
    /// descriptor saved by warmup.  Native binding rejects any stream or
    /// structural drift before returning the executable plan.
    pub fn new_with_layout(
        source: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        conversion: GpuMatrixModulusConversion,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Self, String> {
        let (mode, plaintext_modulus, division_inverses, input_scales) =
            conversion_arguments(source, target.params(), conversion)?;
        if mode != 0 && source.is_ntt() {
            return Err("prepared rounding conversion requires coefficient-domain source".into());
        }
        if source.row_size() != target.row_size() || source.col_size() != target.col_size() {
            return Err("prepared conversion source and destination shapes differ".into());
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_modulus_conversion_with_layout(
                source.raw,
                target.raw,
                mode,
                division_inverses.as_ptr(),
                division_inverses.len(),
                plaintext_modulus,
                input_scales.as_ptr(),
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("native prepared conversion returned no plan")?,
            contexts: GpuPreparedContextAnchors {
                source: source.params().clone(),
                target: target.params().clone(),
            },
            source_rows: source.row_size(),
            source_columns: source.col_size(),
            source_level: source.level(),
            target_rows: target.row_size(),
            target_columns: target.col_size(),
            target_level: target.level(),
            source_evaluation: source.is_ntt(),
            target_evaluation: target.is_ntt(),
        })
    }

    /// Prepare the coefficient-domain RNS ModDown used by BGV modulus
    /// switching.  It uses the same public CRT plan as the ordinary primitive,
    /// but stores its normalized weights in the native plan for reuse.
    pub fn new_rns_down(
        source: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        plaintext_modulus: u64,
    ) -> Result<Self, String> {
        Self::new_rns(source, target, GpuMatrixRnsConversion::Down { plaintext_modulus })
    }

    pub fn new_rns_up(
        source: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        digit_size: usize,
        normalize: bool,
    ) -> Result<Self, String> {
        Self::new_rns(source, target, GpuMatrixRnsConversion::Up { digit_size, normalize })
    }

    pub fn new_rns_down_with_layout(
        source: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        plaintext_modulus: u64,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Self, String> {
        Self::new_rns_with_layout(
            source,
            target,
            GpuMatrixRnsConversion::Down { plaintext_modulus },
            plan_layout,
        )
    }

    pub fn new_rns_up_with_layout(
        source: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        digit_size: usize,
        normalize: bool,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Self, String> {
        Self::new_rns_with_layout(
            source,
            target,
            GpuMatrixRnsConversion::Up { digit_size, normalize },
            plan_layout,
        )
    }

    fn new_rns_with_layout(
        source: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        conversion: GpuMatrixRnsConversion,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Self, String> {
        if source.is_ntt() {
            return Err("prepared RNS conversion requires coefficient-domain source".into());
        }
        let groups = conversion.validate(source.params(), source.level(), target.params())?;
        if target.row_size() != source.row_size() * groups || target.col_size() != source.col_size()
        {
            return Err("prepared RNS conversion destination shape does not match its groups".into());
        }
        let rns_plan = conversion.plan(source.params(), target.params())?;
        let (digit_size, plaintext_modulus) = match conversion {
            GpuMatrixRnsConversion::Up { digit_size, .. } => (digit_size, 0),
            GpuMatrixRnsConversion::Down { plaintext_modulus } => {
                (source.params().crt_depth(), plaintext_modulus)
            }
        };
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_rns_conversion_with_layout(
                source.raw,
                target.raw,
                digit_size,
                plaintext_modulus,
                rns_plan.scales.as_ptr(),
                rns_plan.inverses.as_ptr(),
                rns_plan.inverses.len(),
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("native prepared RNS conversion returned no plan")?,
            contexts: GpuPreparedContextAnchors {
                source: source.params().clone(),
                target: target.params().clone(),
            },
            source_rows: source.row_size(),
            source_columns: source.col_size(),
            source_level: source.level(),
            target_rows: target.row_size(),
            target_columns: target.col_size(),
            target_level: target.level(),
            source_evaluation: source.is_ntt(),
            target_evaluation: target.is_ntt(),
        })
    }

    fn new_rns(
        source: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        conversion: GpuMatrixRnsConversion,
    ) -> Result<Self, String> {
        if source.is_ntt() {
            return Err("prepared RNS conversion requires coefficient-domain source".into());
        }
        let groups = conversion.validate(source.params(), source.level(), target.params())?;
        if target.row_size() != source.row_size() * groups || target.col_size() != source.col_size()
        {
            return Err("prepared RNS conversion destination shape does not match its groups".into());
        }
        let rns_plan = conversion.plan(source.params(), target.params())?;
        let (digit_size, plaintext_modulus) = match conversion {
            GpuMatrixRnsConversion::Up { digit_size, .. } => (digit_size, 0),
            GpuMatrixRnsConversion::Down { plaintext_modulus } => {
                (source.params().crt_depth(), plaintext_modulus)
            }
        };
        let mut raw_plan = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_rns_conversion(
                source.raw,
                target.raw,
                digit_size,
                plaintext_modulus,
                rns_plan.scales.as_ptr(),
                rns_plan.inverses.as_ptr(),
                rns_plan.inverses.len(),
                &mut raw_plan,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        let raw =
            NonNull::new(raw_plan).ok_or("native prepared RNS conversion returned no plan")?;
        Ok(Self {
            raw,
            contexts: GpuPreparedContextAnchors {
                source: source.params().clone(),
                target: target.params().clone(),
            },
            source_rows: source.row_size(),
            source_columns: source.col_size(),
            source_level: source.level(),
            target_rows: target.row_size(),
            target_columns: target.col_size(),
            target_level: target.level(),
            source_evaluation: source.is_ntt(),
            target_evaluation: target.is_ntt(),
        })
    }
}

impl GpuPreparedModulusCommand {
    pub fn submit(&self, target: &GpuDCRTPolyMatrix) -> Result<(), String> {
        let source = &self.source_owner;
        let plan = &self.plan;
        if self.target != target.raw ||
            plan.contexts.source.context_identity() != source.params().context_identity() ||
            plan.contexts.target.context_identity() != target.params().context_identity() ||
            source.params().moduli() != plan.contexts.source.moduli() ||
            target.params().moduli() != plan.contexts.target.moduli() ||
            source.params().device_ids() != plan.contexts.source.device_ids() ||
            target.params().device_ids() != plan.contexts.target.device_ids() ||
            source.row_size() != plan.source_rows ||
            source.col_size() != plan.source_columns ||
            source.level() != plan.source_level ||
            target.row_size() != plan.target_rows ||
            target.col_size() != plan.target_columns ||
            target.level() != plan.target_level ||
            source.is_ntt() != plan.source_evaluation ||
            target.is_ntt() != plan.target_evaluation
        {
            return Err("prepared conversion owner or view no longer matches its plan".into());
        }
        if self.rows != plan.source_rows ||
            self.columns != plan.source_columns ||
            self.target_rows != plan.target_rows
        {
            return Err("prepared conversion view no longer matches its plan".into());
        }
        let input_range = GpuMatrixRange {
            row_start: 0,
            row_end: self.rows,
            column_start: 0,
            column_end: self.columns,
        };
        let output_range = GpuMatrixRange {
            row_start: 0,
            row_end: self.target_rows,
            column_start: 0,
            column_end: self.columns,
        };
        let view =
            GpuMatrixBatchView { left: input_range, right: input_range, output: output_range };
        let status = unsafe {
            gpu_matrix_submit_modulus_conversion(
                self.plan.raw.as_ptr(),
                self.target,
                self.source_owner.raw,
                &view,
                false,
            )
        };
        if status != 0 { Err(last_error_string()) } else { Ok(()) }
    }
}

impl Drop for GpuPreparedModulusConversion {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_prepared_modulus_conversion(self.raw.as_ptr()) };
    }
}

/// The closed set of arithmetic schedules that can be replayed without
/// repeating native shape, descriptor, or kernel selection work.
pub enum GpuPreparedArithmeticKind {
    Copy,
    Add,
    Subtract,
    Negate,
    Scale { residues: Vec<u64> },
    Automorphism { index: usize },
    Tensor,
    TensorSumRows { rows: Vec<usize>, offsets: Vec<usize> },
    Multiply,
}

/// Deterministic arithmetic launch metadata produced without owner lookup or
/// CUDA resource creation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparedArithmeticLayout {
    pub kind: u32,
    pub device: i32,
    pub ring_dimension: usize,
    pub limb_count: usize,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub workspace_bytes: usize,
    pub alignment: usize,
    pub event_count: usize,
    pub thin: bool,
    pub lazy_reduction: bool,
}

impl PreparedArithmeticLayout {
    pub fn query(
        kind: &GpuPreparedArithmeticKind,
        ring_dimension: usize,
        limb_count: usize,
        left_rows: usize,
        left_columns: usize,
        right_rows: usize,
        right_columns: usize,
        output_rows: usize,
        output_columns: usize,
        column_start: usize,
        group_count: usize,
        term_count: usize,
        device: i32,
        evaluation_format: bool,
        thin: bool,
        lazy_reduction: bool,
    ) -> Result<Self, String> {
        let (kind_id, _, _, _, _) = arithmetic_kind_arguments(kind);
        let mut native = NativeArithmeticLayout::default();
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_matrix_query_arithmetic_layout(
                ring_dimension,
                limb_count,
                left_rows,
                left_columns,
                right_rows,
                right_columns,
                output_rows,
                output_columns,
                column_start,
                group_count,
                term_count,
                kind_id,
                device,
                evaluation_format as i32,
                thin as i32,
                lazy_reduction as i32,
                &mut native,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            kind: native.kind as u32,
            device: native.device,
            ring_dimension: native.ring_dimension,
            limb_count: native.limb_count,
            grid: [native.grid_x, native.grid_y, native.grid_z],
            block: [native.block_x, native.block_y, native.block_z],
            workspace_bytes: native.workspace_bytes,
            alignment: native.alignment,
            event_count: native.event_count,
            thin: native.thin != 0,
            lazy_reduction: native.lazy_reduction != 0,
        })
    }
}

fn arithmetic_kind_arguments(
    kind: &GpuPreparedArithmeticKind,
) -> (i32, Vec<usize>, Vec<usize>, Vec<u64>, usize) {
    match kind {
        GpuPreparedArithmeticKind::Copy => (0, Vec::new(), Vec::new(), Vec::new(), 0),
        GpuPreparedArithmeticKind::Add => (1, Vec::new(), Vec::new(), Vec::new(), 0),
        GpuPreparedArithmeticKind::Tensor => (2, Vec::new(), Vec::new(), Vec::new(), 0),
        GpuPreparedArithmeticKind::TensorSumRows { rows, offsets } => {
            (3, rows.clone(), offsets.clone(), Vec::new(), 0)
        }
        GpuPreparedArithmeticKind::Multiply => (4, Vec::new(), Vec::new(), Vec::new(), 0),
        GpuPreparedArithmeticKind::Subtract => (5, Vec::new(), Vec::new(), Vec::new(), 0),
        GpuPreparedArithmeticKind::Negate => (6, Vec::new(), Vec::new(), Vec::new(), 0),
        GpuPreparedArithmeticKind::Scale { residues } => {
            (7, Vec::new(), Vec::new(), residues.clone(), 0)
        }
        GpuPreparedArithmeticKind::Automorphism { index } => {
            (8, Vec::new(), Vec::new(), Vec::new(), *index)
        }
    }
}

pub struct GpuPreparedArithmetic {
    raw: NonNull<GpuPreparedArithmeticOpaque>,
    layout: PreparedArithmeticLayout,
}

unsafe impl Send for GpuPreparedArithmetic {}
unsafe impl Sync for GpuPreparedArithmetic {}

impl GpuPreparedArithmetic {
    pub fn layout(&self) -> &PreparedArithmeticLayout {
        &self.layout
    }
}

pub struct GpuPreparedArithmeticCommand {
    plan: GpuPreparedArithmetic,
    lhs: Arc<GpuDCRTPolyMatrix>,
    rhs: Option<Arc<GpuDCRTPolyMatrix>>,
    output: Arc<GpuDCRTPolyMatrix>,
}

/// Fixed command tape for `bias + sum(coefficient * lhs * rhs)`.
///
/// The intermediate product/scaled owners are allocated and all arithmetic
/// plans are built at preparation time. Submission only replays this fixed
/// sequence; it does not select kernels or allocate scratch.
pub struct GpuPreparedAccumulateCommand {
    commands: Box<[GpuPreparedArithmeticCommand]>,
}

/// Ordered native descriptors for the complete accumulate tape.  Accumulate
/// is not one arithmetic launch: it may copy a bias, multiply each term,
/// scale non-unit coefficients, and add/copy every subsequent value.  Keeping
/// the descriptors in claim order makes that composite shape explicit to the
/// binder instead of allowing it to rebuild a plan for each intermediate.
#[derive(Clone, Debug)]
pub struct GpuPreparedAccumulateLayout {
    stages: Box<[super::PreparedPlanLayout]>,
    intermediate_owners: Box<[super::PreparedOwnerLayout]>,
}

impl PartialEq for GpuPreparedAccumulateLayout {
    fn eq(&self, other: &Self) -> bool {
        self.stages == other.stages &&
            self.intermediate_owners.iter().zip(&other.intermediate_owners).all(
                |(left, right)| {
                    left.execution_owner_identity() == right.execution_owner_identity() &&
                        left.execution_class() == right.execution_class() &&
                        left.partition_count() == right.partition_count()
                },
            ) &&
            self.intermediate_owners.len() == other.intermediate_owners.len()
    }
}

impl Eq for GpuPreparedAccumulateLayout {}

impl GpuPreparedAccumulateLayout {
    pub fn new(stages: Vec<super::PreparedPlanLayout>) -> Result<Self, String> {
        Self::with_owners(stages, Vec::new())
    }

    pub fn with_owners(
        stages: Vec<super::PreparedPlanLayout>,
        intermediate_owners: Vec<super::PreparedOwnerLayout>,
    ) -> Result<Self, String> {
        if stages.is_empty() {
            return Err("prepared accumulate layout requires at least one stage".into());
        }
        Ok(Self {
            stages: stages.into_boxed_slice(),
            intermediate_owners: intermediate_owners.into_boxed_slice(),
        })
    }

    pub fn stages(&self) -> &[super::PreparedPlanLayout] {
        &self.stages
    }

    pub fn intermediate_owners(&self) -> &[super::PreparedOwnerLayout] {
        &self.intermediate_owners
    }
}

unsafe impl Send for GpuPreparedArithmeticCommand {}
unsafe impl Sync for GpuPreparedArithmeticCommand {}

impl GpuPreparedArithmetic {
    pub fn new(
        kind: &GpuPreparedArithmeticKind,
        lhs: &GpuDCRTPolyMatrix,
        rhs: Option<&GpuDCRTPolyMatrix>,
        output: &GpuDCRTPolyMatrix,
    ) -> Result<Self, String> {
        Self::new_with_view(kind, lhs, rhs, output, None, 0)
    }

    pub fn new_with_view(
        kind: &GpuPreparedArithmeticKind,
        lhs: &GpuDCRTPolyMatrix,
        rhs: Option<&GpuDCRTPolyMatrix>,
        output: &GpuDCRTPolyMatrix,
        view: Option<GpuPreparedView>,
        column_start: usize,
    ) -> Result<Self, String> {
        let (kind_id, rows, offsets, scalar_residues, automorphism_index) =
            arithmetic_kind_arguments(kind);
        let rhs = rhs.unwrap_or(lhs);
        let view = view.map(|view| GpuMatrixBatchView {
            left: GpuMatrixRange {
                row_start: view.left.rows.start,
                row_end: view.left.rows.end,
                column_start: view.left.columns.start,
                column_end: view.left.columns.end,
            },
            right: GpuMatrixRange {
                row_start: view.right.rows.start,
                row_end: view.right.rows.end,
                column_start: view.right.columns.start,
                column_end: view.right.columns.end,
            },
            output: GpuMatrixRange {
                row_start: view.output.rows.start,
                row_end: view.output.rows.end,
                column_start: view.output.columns.start,
                column_end: view.output.columns.end,
            },
        });
        let (left_rows, left_columns, right_rows, right_columns, output_rows, output_columns) =
            view.as_ref().map_or(
                (
                    lhs.row_size(),
                    lhs.col_size(),
                    rhs.row_size(),
                    rhs.col_size(),
                    output.row_size(),
                    output.col_size(),
                ),
                |view| {
                    (
                        view.left.row_end - view.left.row_start,
                        view.left.column_end - view.left.column_start,
                        view.right.row_end - view.right.row_start,
                        view.right.column_end - view.right.column_start,
                        view.output.row_end - view.output.row_start,
                        view.output.column_end - view.output.column_start,
                    )
                },
            );
        let device = lhs
            .params()
            .device_ids()
            .first()
            .copied()
            .ok_or_else(|| "prepared arithmetic has no device".to_string())?;
        // The native matmul plan chooses these fast paths independently for
        // every active CRT limb.  Keep the metadata query on the same
        // conservative aggregate: a shared layout may advertise the narrow
        // path only when every limb can consume it.
        let is_multiply = matches!(kind, GpuPreparedArithmeticKind::Multiply);
        let active_moduli = lhs.params().moduli().iter().take(lhs.level() + 1);
        let thin = is_multiply &&
            view.is_none() &&
            left_rows == 1 &&
            active_moduli.clone().all(|&modulus| modulus > 1 && modulus <= u32::MAX as u64);
        let lazy_reduction = thin &&
            lhs.params().moduli().iter().take(lhs.level() + 1).all(|&modulus| {
                let factor = u128::from(modulus - 1);
                factor * factor * (left_columns as u128) <= u128::from(u64::MAX)
            });
        let layout = PreparedArithmeticLayout::query(
            kind,
            lhs.params().ring_dimension() as usize,
            lhs.level() + 1,
            left_rows,
            left_columns,
            right_rows,
            right_columns,
            output_rows,
            output_columns,
            column_start,
            offsets.len().saturating_sub(1),
            rows.len(),
            device,
            lhs.is_ntt(),
            thin,
            lazy_reduction,
        )?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_arithmetic(
                output.raw,
                lhs.raw,
                rhs.raw,
                kind_id,
                if rows.is_empty() { std::ptr::null() } else { rows.as_ptr() },
                if offsets.is_empty() { std::ptr::null() } else { offsets.as_ptr() },
                offsets.len().saturating_sub(1),
                rows.len(),
                view.as_ref().map_or(std::ptr::null(), |view| view),
                column_start,
                if scalar_residues.is_empty() {
                    std::ptr::null()
                } else {
                    scalar_residues.as_ptr()
                },
                scalar_residues.len(),
                automorphism_index,
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("native prepared arithmetic returned no plan")?,
            layout,
        })
    }

    pub fn bind(
        kind: GpuPreparedArithmeticKind,
        lhs: Arc<GpuDCRTPolyMatrix>,
        rhs: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
    ) -> Result<GpuPreparedArithmeticCommand, String> {
        Self::bind_with_view(kind, lhs, rhs, output, None, 0)
    }

    pub fn bind_with_view(
        kind: GpuPreparedArithmeticKind,
        lhs: Arc<GpuDCRTPolyMatrix>,
        rhs: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
        view: Option<GpuPreparedView>,
        column_start: usize,
    ) -> Result<GpuPreparedArithmeticCommand, String> {
        let plan = Self::new_with_view(&kind, &lhs, rhs.as_deref(), &output, view, column_start)?;
        Ok(GpuPreparedArithmeticCommand { plan, lhs, rhs, output })
    }

    /// Bind an arithmetic stage using the descriptor saved by metadata-only
    /// preparation. All shape, fast-path, workspace, and stream decisions are
    /// consumed from `plan_layout`; the native binder rejects drift.
    pub fn bind_with_view_and_layout(
        kind: GpuPreparedArithmeticKind,
        lhs: Arc<GpuDCRTPolyMatrix>,
        rhs: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
        view: Option<GpuPreparedView>,
        column_start: usize,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<GpuPreparedArithmeticCommand, String> {
        let (kind_id, rows, offsets, scalar_residues, automorphism_index) =
            arithmetic_kind_arguments(&kind);
        let rhs_ref = rhs.as_deref().unwrap_or(&lhs);
        let native_view = view.as_ref().map(|view| GpuMatrixBatchView {
            left: GpuMatrixRange {
                row_start: view.left.rows.start,
                row_end: view.left.rows.end,
                column_start: view.left.columns.start,
                column_end: view.left.columns.end,
            },
            right: GpuMatrixRange {
                row_start: view.right.rows.start,
                row_end: view.right.rows.end,
                column_start: view.right.columns.start,
                column_end: view.right.columns.end,
            },
            output: GpuMatrixRange {
                row_start: view.output.rows.start,
                row_end: view.output.rows.end,
                column_start: view.output.columns.start,
                column_end: view.output.columns.end,
            },
        });
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_arithmetic_with_layout(
                output.raw,
                lhs.raw,
                rhs_ref.raw,
                kind_id,
                rows.as_ptr(),
                offsets.as_ptr(),
                offsets.len().saturating_sub(1),
                rows.len(),
                native_view.as_ref().map_or(std::ptr::null(), |view| view),
                column_start,
                scalar_residues.as_ptr(),
                scalar_residues.len(),
                automorphism_index,
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(GpuPreparedArithmeticCommand {
            plan: GpuPreparedArithmetic {
                raw: NonNull::new(raw).ok_or("native prepared arithmetic returned no plan")?,
                // The saved descriptor is the source of native geometry; no
                // post-bind metadata query is permitted on this path.
                layout: PreparedArithmeticLayout::default(),
            },
            lhs,
            rhs,
            output,
        })
    }
}

impl GpuPreparedArithmeticCommand {
    pub fn output_owner(&self) -> Arc<GpuDCRTPolyMatrix> {
        Arc::clone(&self.output)
    }

    pub fn submit(&self) -> Result<(), String> {
        if self.output.raw == self.lhs.raw ||
            self.rhs.as_ref().is_some_and(|rhs| rhs.raw == self.output.raw)
        {
            return Err("prepared arithmetic output aliases an input owner".into());
        }
        let status = unsafe { gpu_matrix_submit_arithmetic(self.plan.raw.as_ptr()) };
        if status != 0 { Err(last_error_string()) } else { Ok(()) }
    }
}

impl GpuPreparedAccumulateCommand {
    pub fn bind(
        terms: Vec<(Arc<GpuDCRTPolyMatrix>, Arc<GpuDCRTPolyMatrix>, Vec<u64>)>,
        bias: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
    ) -> Result<Self, String> {
        if terms.is_empty() {
            return Err("prepared accumulate requires at least one product".into());
        }
        let mut commands = Vec::with_capacity(terms.len() * 2 + 1);
        let mut first = true;
        if let Some(bias) = bias {
            commands.push(GpuPreparedArithmetic::bind(
                GpuPreparedArithmeticKind::Copy,
                Arc::clone(&bias),
                None,
                Arc::clone(&output),
            )?);
            first = false;
        }
        for (lhs, rhs, residues) in terms {
            let product = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                lhs.params(),
                lhs.row_size(),
                rhs.col_size(),
                lhs.level(),
                true,
                None,
            ));
            let product_command = GpuPreparedArithmetic::bind(
                GpuPreparedArithmeticKind::Multiply,
                lhs,
                Some(rhs),
                Arc::clone(&product),
            )?;
            commands.push(product_command);
            let value = if residues.iter().all(|&residue| residue == 1) {
                product
            } else {
                let scaled = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                    product.params(),
                    product.row_size(),
                    product.col_size(),
                    product.level(),
                    true,
                    None,
                ));
                commands.push(GpuPreparedArithmetic::bind(
                    GpuPreparedArithmeticKind::Scale { residues },
                    Arc::clone(&product),
                    None,
                    Arc::clone(&scaled),
                )?);
                scaled
            };
            if first {
                commands.push(GpuPreparedArithmetic::bind(
                    GpuPreparedArithmeticKind::Copy,
                    value,
                    None,
                    Arc::clone(&output),
                )?);
                first = false;
            } else {
                let sum = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                    output.params(),
                    output.row_size(),
                    output.col_size(),
                    output.level(),
                    true,
                    None,
                ));
                commands.push(GpuPreparedArithmetic::bind(
                    GpuPreparedArithmeticKind::Add,
                    Arc::clone(&output),
                    Some(value),
                    Arc::clone(&sum),
                )?);
                commands.push(GpuPreparedArithmetic::bind(
                    GpuPreparedArithmeticKind::Copy,
                    sum,
                    None,
                    Arc::clone(&output),
                )?);
            }
        }
        Ok(Self { commands: commands.into_boxed_slice() })
    }

    /// Bind every accumulate sub-stage from one saved, ordered descriptor
    /// bundle. The owner allocation order and arithmetic payload are fixed by
    /// the caller's warmup bundle; no stage-level planner is consulted here.
    pub fn bind_with_layout(
        terms: Vec<(Arc<GpuDCRTPolyMatrix>, Arc<GpuDCRTPolyMatrix>, Vec<u64>)>,
        bias: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
        layout: &GpuPreparedAccumulateLayout,
    ) -> Result<Self, String> {
        if terms.is_empty() {
            return Err("prepared accumulate requires at least one product".into());
        }
        let scaled =
            terms.iter().filter(|(_, _, residues)| !residues.iter().all(|&r| r == 1)).count();
        let expected = terms.len() +
            scaled +
            if bias.is_some() {
                1 + terms.len() * 2
            } else {
                1 + terms.len().saturating_sub(1) * 2
            };
        if layout.stages.len() != expected {
            return Err("prepared accumulate stage bundle length mismatch".into());
        }
        let mut stage = 0;
        let expected_owners = terms.len() + scaled + terms.len().saturating_sub(1);
        if layout.intermediate_owners.len() != expected_owners {
            return Err("prepared accumulate owner bundle length mismatch".into());
        }
        let mut owner = 0;
        let mut commands = Vec::with_capacity(expected);
        let mut first = true;
        if let Some(bias) = bias {
            commands.push(GpuPreparedArithmetic::bind_with_view_and_layout(
                GpuPreparedArithmeticKind::Copy,
                Arc::clone(&bias),
                None,
                Arc::clone(&output),
                None,
                0,
                &layout.stages[stage],
            )?);
            stage += 1;
            first = false;
        }
        for (lhs, rhs, residues) in terms {
            let product = Arc::new(GpuDCRTPolyMatrix::new_empty_with_owner_layout(
                lhs.params(),
                lhs.row_size(),
                rhs.col_size(),
                lhs.level(),
                true,
                None,
                &layout.intermediate_owners[owner],
            )?);
            owner += 1;
            commands.push(GpuPreparedArithmetic::bind_with_view_and_layout(
                GpuPreparedArithmeticKind::Multiply,
                Arc::clone(&lhs),
                Some(Arc::clone(&rhs)),
                Arc::clone(&product),
                None,
                0,
                &layout.stages[stage],
            )?);
            stage += 1;
            let value = if residues.iter().all(|&residue| residue == 1) {
                product
            } else {
                let scaled = Arc::new(GpuDCRTPolyMatrix::new_empty_with_owner_layout(
                    product.params(),
                    product.row_size(),
                    product.col_size(),
                    product.level(),
                    true,
                    None,
                    &layout.intermediate_owners[owner],
                )?);
                owner += 1;
                commands.push(GpuPreparedArithmetic::bind_with_view_and_layout(
                    GpuPreparedArithmeticKind::Scale { residues },
                    Arc::clone(&product),
                    None,
                    Arc::clone(&scaled),
                    None,
                    0,
                    &layout.stages[stage],
                )?);
                stage += 1;
                scaled
            };
            if first {
                commands.push(GpuPreparedArithmetic::bind_with_view_and_layout(
                    GpuPreparedArithmeticKind::Copy,
                    value,
                    None,
                    Arc::clone(&output),
                    None,
                    0,
                    &layout.stages[stage],
                )?);
                stage += 1;
                first = false;
            } else {
                let sum = Arc::new(GpuDCRTPolyMatrix::new_empty_with_owner_layout(
                    output.params(),
                    output.row_size(),
                    output.col_size(),
                    output.level(),
                    true,
                    None,
                    &layout.intermediate_owners[owner],
                )?);
                owner += 1;
                commands.push(GpuPreparedArithmetic::bind_with_view_and_layout(
                    GpuPreparedArithmeticKind::Add,
                    Arc::clone(&output),
                    Some(value),
                    Arc::clone(&sum),
                    None,
                    0,
                    &layout.stages[stage],
                )?);
                stage += 1;
                commands.push(GpuPreparedArithmetic::bind_with_view_and_layout(
                    GpuPreparedArithmeticKind::Copy,
                    sum,
                    None,
                    Arc::clone(&output),
                    None,
                    0,
                    &layout.stages[stage],
                )?);
                stage += 1;
            }
        }
        debug_assert_eq!(stage, expected);
        Ok(Self { commands: commands.into_boxed_slice() })
    }

    pub fn submit(&self) -> Result<(), String> {
        for command in &self.commands {
            command.submit()?;
        }
        Ok(())
    }
}

impl Drop for GpuPreparedArithmetic {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_arithmetic_plan(self.raw.as_ptr()) };
    }
}

/// A compact-RHS multiplication whose RHS NTT and typed workspace are
/// prepared once. The output and compact RHS owners are retained by the plan;
/// each submission may provide a fresh, same-shape evaluation-domain input.
pub struct GpuPreparedSmallRhs {
    raw: NonNull<GpuPreparedSmallRhsOpaque>,
    layout: super::PreparedPlanLayout,
    output: Arc<GpuDCRTPolyMatrix>,
    source_template: Arc<GpuDCRTPolyMatrix>,
    rhs: Arc<GpuSmallMatrix>,
}

pub struct GpuPreparedSmallRhsInFlight {
    _plan: Arc<GpuPreparedSmallRhs>,
    source: Arc<GpuDCRTPolyMatrix>,
    output: Arc<GpuDCRTPolyMatrix>,
}

unsafe impl Send for GpuPreparedSmallRhs {}
unsafe impl Sync for GpuPreparedSmallRhs {}
unsafe impl Send for GpuPreparedSmallRhsInFlight {}
unsafe impl Sync for GpuPreparedSmallRhsInFlight {}

impl GpuPreparedSmallRhs {
    pub fn bind(
        output: Arc<GpuDCRTPolyMatrix>,
        source_template: Arc<GpuDCRTPolyMatrix>,
        rhs: Arc<GpuSmallMatrix>,
        residency_budget_bytes: usize,
    ) -> Result<Arc<Self>, String> {
        let owner = output.prepared_owner_layout()?;
        let layout = super::PreparedPlanLayout::small_rhs_with_owner(
            output.params(),
            output.level(),
            source_template.col_size(),
            output.col_size(),
            residency_budget_bytes,
            &owner,
        )?;
        Self::bind_with_layout(output, source_template, rhs, residency_budget_bytes, &layout)
    }

    pub fn bind_with_layout(
        output: Arc<GpuDCRTPolyMatrix>,
        source_template: Arc<GpuDCRTPolyMatrix>,
        rhs: Arc<GpuSmallMatrix>,
        residency_budget_bytes: usize,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_small_rhs(
                source_template.raw,
                output.raw,
                rhs.raw,
                residency_budget_bytes,
                plan_layout.native_ptr(),
                &mut raw,
            )
        };
        if status == 2 {
            return Err("prepared compact RHS workspace exceeds residency budget".into());
        }
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared compact RHS returned no plan")?,
            layout: plan_layout.clone(),
            output,
            source_template,
            rhs,
        }))
    }

    pub fn submit(
        self: &Arc<Self>,
        source: Arc<GpuDCRTPolyMatrix>,
    ) -> Result<GpuPreparedSmallRhsInFlight, String> {
        if source.params() != self.source_template.params() ||
            source.level() != self.source_template.level() ||
            source.row_size() != self.source_template.row_size() ||
            source.col_size() != self.source_template.col_size() ||
            !source.is_ntt()
        {
            return Err("prepared compact RHS input contract mismatch".into());
        }
        let status = unsafe { gpu_matrix_submit_small_rhs(self.raw.as_ptr(), source.raw) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(GpuPreparedSmallRhsInFlight {
            _plan: Arc::clone(self),
            source,
            output: Arc::clone(&self.output),
        })
    }

    /// Submit with a caller-retained source owner. The enclosing prepared
    /// execution retains that owner until its output lease retires.
    pub fn submit_borrowed(&self, source: &GpuDCRTPolyMatrix) -> Result<(), String> {
        if source.params() != self.source_template.params() ||
            source.level() != self.source_template.level() ||
            source.row_size() != self.source_template.row_size() ||
            source.col_size() != self.source_template.col_size() ||
            !source.is_ntt()
        {
            return Err("prepared compact RHS input contract mismatch".into());
        }
        let status = unsafe { gpu_matrix_submit_small_rhs(self.raw.as_ptr(), source.raw) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }

    pub fn rhs(&self) -> &GpuSmallMatrix {
        &self.rhs
    }
    pub fn allocation_layout(&self) -> &[super::PreparedAllocationLayout] {
        self.layout.allocations()
    }
}

impl GpuPreparedSmallRhsInFlight {
    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        &self.output
    }

    pub fn source(&self) -> &GpuDCRTPolyMatrix {
        &self.source
    }
}

impl Drop for GpuPreparedSmallRhs {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_prepared_small_rhs(self.raw.as_ptr()) };
    }
}

/// A fixed full-row CRT recomposition. CRT arithmetic metadata and, for more
/// than two levels, the device metadata table are prepared once; submissions
/// only replace the source descriptor pointers.
pub struct GpuPreparedCrtRecompose {
    raw: NonNull<GpuPreparedCrtRecomposeOpaque>,
    output: Arc<GpuDCRTPolyMatrix>,
    levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
}

pub struct GpuPreparedCrtRecomposeInFlight {
    _plan: Arc<GpuPreparedCrtRecompose>,
    levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
    output: Arc<GpuDCRTPolyMatrix>,
}

unsafe impl Send for GpuPreparedCrtRecompose {}
unsafe impl Sync for GpuPreparedCrtRecompose {}
unsafe impl Send for GpuPreparedCrtRecomposeInFlight {}
unsafe impl Sync for GpuPreparedCrtRecomposeInFlight {}

impl GpuPreparedCrtRecompose {
    pub fn bind(
        output: Arc<GpuDCRTPolyMatrix>,
        levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
        plaintext_moduli: Vec<u64>,
        reconstruction_residues: Vec<u64>,
    ) -> Result<Arc<Self>, String> {
        let raw_levels = levels.iter().map(|level| level.raw.cast_const()).collect::<Vec<_>>();
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_crt_recompose(
                raw_levels.as_ptr(),
                raw_levels.len(),
                plaintext_moduli.as_ptr(),
                reconstruction_residues.as_ptr(),
                output.params().crt_depth(),
                output.raw,
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared CRT recomposition returned no plan")?,
            output,
            levels,
        }))
    }

    pub fn bind_with_layout(
        output: Arc<GpuDCRTPolyMatrix>,
        levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
        plaintext_moduli: Vec<u64>,
        reconstruction_residues: Vec<u64>,
        plan_layout: &super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let raw_levels = levels.iter().map(|level| level.raw.cast_const()).collect::<Vec<_>>();
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_crt_recompose_with_layout(
                raw_levels.as_ptr(),
                raw_levels.len(),
                plaintext_moduli.as_ptr(),
                reconstruction_residues.as_ptr(),
                output.params().crt_depth(),
                output.raw,
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native prepared CRT recomposition returned no plan")?,
            output,
            levels,
        }))
    }

    pub fn submit(
        self: &Arc<Self>,
        levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
    ) -> Result<GpuPreparedCrtRecomposeInFlight, String> {
        if levels.len() != self.levels.len() ||
            levels.iter().any(|level| level.is_ntt()) ||
            levels
                .iter()
                .zip(self.levels.iter())
                .any(|(level, prepared)| !Arc::ptr_eq(level, prepared))
        {
            return Err("prepared CRT recomposition level contract mismatch".into());
        }
        let mut raw_levels = [std::ptr::null(); 64];
        for (index, level) in levels.iter().enumerate() {
            raw_levels[index] = level.raw.cast_const();
        }
        let status = unsafe {
            gpu_matrix_submit_crt_recompose(self.raw.as_ptr(), raw_levels.as_ptr(), levels.len())
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(GpuPreparedCrtRecomposeInFlight {
            _plan: Arc::clone(self),
            levels,
            output: Arc::clone(&self.output),
        })
    }
}

impl GpuPreparedCrtRecomposeInFlight {
    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        &self.output
    }

    pub fn levels(&self) -> &[Arc<GpuDCRTPolyMatrix>] {
        &self.levels
    }
}

impl Drop for GpuPreparedCrtRecompose {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_prepared_crt_recompose(self.raw.as_ptr()) };
    }
}

pub struct GpuPreparedInputCopy {
    raw: NonNull<GpuPreparedInputCopyOpaque>,
    output: Arc<GpuDCRTPolyMatrix>,
    source_template: Arc<GpuDCRTPolyMatrix>,
    layout: PreparedRectLayout,
}

#[derive(Clone, Debug)]
pub struct GpuPreparedRange {
    pub rows: Range<usize>,
    pub columns: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct GpuPreparedView {
    pub left: GpuPreparedRange,
    pub right: GpuPreparedRange,
    pub output: GpuPreparedRange,
}

pub struct GpuPreparedInputCopyInFlight {
    source: Arc<GpuDCRTPolyMatrix>,
    output: Arc<GpuDCRTPolyMatrix>,
}

unsafe impl Send for GpuPreparedInputCopy {}
unsafe impl Sync for GpuPreparedInputCopy {}
unsafe impl Send for GpuPreparedInputCopyInFlight {}
unsafe impl Sync for GpuPreparedInputCopyInFlight {}

impl GpuPreparedInputCopy {
    pub fn output_owner(&self) -> Arc<GpuDCRTPolyMatrix> {
        Arc::clone(&self.output)
    }

    pub fn bind(
        output: Arc<GpuDCRTPolyMatrix>,
        source_template: Arc<GpuDCRTPolyMatrix>,
        view: Option<GpuPreparedView>,
    ) -> Result<Self, String> {
        let view = view.map(|view| GpuMatrixBatchView {
            left: GpuMatrixRange {
                row_start: view.left.rows.start,
                row_end: view.left.rows.end,
                column_start: view.left.columns.start,
                column_end: view.left.columns.end,
            },
            right: GpuMatrixRange {
                row_start: view.right.rows.start,
                row_end: view.right.rows.end,
                column_start: view.right.columns.start,
                column_end: view.right.columns.end,
            },
            output: GpuMatrixRange {
                row_start: view.output.rows.start,
                row_end: view.output.rows.end,
                column_start: view.output.columns.start,
                column_end: view.output.columns.end,
            },
        });
        let rows = view
            .as_ref()
            .map_or(source_template.row_size(), |view| view.left.row_end - view.left.row_start);
        let columns = view.as_ref().map_or(source_template.col_size(), |view| {
            view.left.column_end - view.left.column_start
        });
        let device = source_template
            .params()
            .device_ids()
            .first()
            .copied()
            .ok_or_else(|| "prepared input copy has no device".to_string())?;
        let layout = PreparedRectLayout::query_input_copy(
            source_template.params().ring_dimension() as usize,
            source_template.level() + 1,
            rows,
            columns,
            device,
        )?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_input_copy(
                output.raw,
                source_template.raw,
                view.as_ref().map_or(std::ptr::null(), |view| view),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("native prepared input copy returned no plan")?,
            output,
            source_template,
            layout,
        })
    }

    /// Bind using the exact metadata-only descriptor produced during warmup.
    /// Native preparation consumes and validates this descriptor before the
    /// executable is published.
    pub fn bind_with_layout(
        output: Arc<GpuDCRTPolyMatrix>,
        source_template: Arc<GpuDCRTPolyMatrix>,
        view: Option<GpuPreparedView>,
        plan_layout: super::PreparedPlanLayout,
    ) -> Result<Self, String> {
        let native_view = view.as_ref().map(|view| GpuMatrixBatchView {
            left: GpuMatrixRange {
                row_start: view.left.rows.start,
                row_end: view.left.rows.end,
                column_start: view.left.columns.start,
                column_end: view.left.columns.end,
            },
            right: GpuMatrixRange {
                row_start: view.right.rows.start,
                row_end: view.right.rows.end,
                column_start: view.right.columns.start,
                column_end: view.right.columns.end,
            },
            output: GpuMatrixRange {
                row_start: view.output.rows.start,
                row_end: view.output.rows.end,
                column_start: view.output.columns.start,
                column_end: view.output.columns.end,
            },
        });
        let rows = native_view
            .as_ref()
            .map_or(source_template.row_size(), |view| view.left.row_end - view.left.row_start);
        let columns = native_view.as_ref().map_or(source_template.col_size(), |view| {
            view.left.column_end - view.left.column_start
        });
        let layout = plan_layout.rectangular_layout(
            rows,
            columns,
            source_template.params().ring_dimension() as usize,
            source_template.level() + 1,
            1,
        )?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_prepare_input_copy_with_layout(
                output.raw,
                source_template.raw,
                native_view.as_ref().map_or(std::ptr::null(), |view| view),
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("native prepared input copy returned no plan")?,
            output,
            source_template,
            layout,
        })
    }

    pub fn layout(&self) -> &PreparedRectLayout {
        &self.layout
    }

    pub fn submit(
        &self,
        source: Arc<GpuDCRTPolyMatrix>,
    ) -> Result<GpuPreparedInputCopyInFlight, String> {
        if source.params() != self.source_template.params() ||
            source.level() != self.source_template.level()
        {
            return Err("prepared input copy source parameters changed".into());
        }
        let status = unsafe { gpu_matrix_submit_input_copy(self.raw.as_ptr(), source.raw) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(GpuPreparedInputCopyInFlight { source, output: Arc::clone(&self.output) })
    }

    /// Submit a fixed copy from a caller-owned source. The caller must retain
    /// `source` until all readers of the prepared output have retired.
    pub fn submit_borrowed(&self, source: &GpuDCRTPolyMatrix) -> Result<(), String> {
        if source.params() != self.source_template.params() ||
            source.level() != self.source_template.level()
        {
            return Err("prepared input copy source parameters changed".into());
        }
        let status = unsafe { gpu_matrix_submit_input_copy(self.raw.as_ptr(), source.raw) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }
}

impl GpuPreparedInputCopyInFlight {
    pub fn source(&self) -> &GpuDCRTPolyMatrix {
        &self.source
    }

    pub fn output(&self) -> &GpuDCRTPolyMatrix {
        &self.output
    }
}

impl Drop for GpuPreparedInputCopy {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_input_copy(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    /// Extracts one exported native definition so a source-level contract can
    /// be asserted without a device.
    fn cuda_function_body<'a>(source: &'a str, name: &str) -> &'a str {
        let start = source.find(name).unwrap_or_else(|| panic!("missing {name}"));
        let end = source[start + name.len()..]
            .find("\nextern \"C\"")
            .map(|offset| start + name.len() + offset)
            .unwrap_or(source.len());
        &source[start..end]
    }

    /// The schedule readiness query is the runtime's nonblocking polling path:
    /// a stream that is still running must report `ready = false` with a
    /// success status instead of poisoning a healthy schedule.
    #[test]
    fn test_gpu_prepared_schedule_readiness_reports_pending_streams() {
        let source = include_str!("../../cuda/src/matrix/gpu_schedule.cu");
        let body = cuda_function_body(source, "gpu_prepared_schedule_is_ready");
        let pending = body.split("cudaErrorNotReady").nth(1).expect("pending-status branch");
        let branch = &pending[..pending.find('}').expect("pending-status branch body")];
        assert!(branch.contains("*ready = false"), "a pending stream must report ready = false");
        assert!(
            branch.contains("cudaSuccess"),
            "a pending stream must not be reported as an error"
        );
    }

    /// A partially submitted plan keeps its pinned allocation and can only
    /// report completion for a terminal event it recorded for that submission.
    #[test]
    fn test_gpu_prepared_submission_failures_record_the_terminal_generation() {
        let source = include_str!("../../cuda/src/matrix/MatrixSerde.cu");
        assert!(
            source.contains("coefficient_index, coefficient_count, {}, 1, {}"),
            "prepared readback must start with a nonzero generation so pre-submit polling is not ready"
        );
        assert!(
            source.contains("limb.completion_resource->quarantine()"),
            "uncertain completion-event recording must quarantine its prepared slot"
        );
        assert!(
            source.contains("consumer_tracked_generation"),
            "readback failure cleanup must track each affected limb's readonly consumer"
        );
        let readback_submit = cuda_function_body(source, "gpu_matrix_submit_const_coeff_readback");
        assert!(
            readback_submit
                .contains("plan->submission_generation == std::numeric_limits<uint64_t>::max()"),
            "prepared readback must reject generation exhaustion before increment"
        );
        let exhaustion_guard = readback_submit
            .find("plan->submission_generation == std::numeric_limits<uint64_t>::max()")
            .expect("generation exhaustion guard");
        let output_clear = readback_submit
            .find("std::fill_n(plan->words, total_words, static_cast<uint64_t>(0))")
            .expect("prepared readback output clear");
        assert!(
            exhaustion_guard < output_clear,
            "generation exhaustion must be rejected before mutating readback output"
        );
        for query in ["gpu_matrix_query_const_coeff_readback", "gpu_matrix_query_rns_upload"] {
            let body = cuda_function_body(source, query);
            assert!(
                body.contains("armed_generation != plan->submission_generation"),
                "{query} must ignore limbs that queued nothing for the newest submission"
            );
            assert!(
                body.contains("recorded_generation != plan->submission_generation"),
                "{query} must refuse to answer for an armed but unrecorded terminal event"
            );
        }
        for (submit, failure) in [
            ("gpu_matrix_submit_const_coeff_readback", "fail_const_coeff_readback_submission"),
            ("gpu_matrix_submit_rns_upload", "fail_rns_upload_submission"),
        ] {
            let body = cuda_function_body(source, submit);
            assert!(
                body.contains("armed_generation = generation"),
                "{submit} must arm the terminal generation of every limb that queues a copy"
            );
            // Once a submission starts queueing, every failure return goes
            // through the recorder, so no path can leave an unrecorded
            // generation that a reclaim would mistake for completion.
            let queued = body.find("++plan->submission_generation").expect("submission generation");
            let tail = &body[queued..];
            assert!(
                !tail.contains("return set_error(") && !tail.contains("return status;"),
                "{submit} must record the terminal generation on every failure path"
            );
            assert!(tail.contains(failure), "{submit} must report failures through {failure}");
        }
    }

    /// An unfinished plan hands its pinned allocation to the context-owned
    /// reclaimer instead of waiting, growing a global deferred list, or freeing
    /// memory a queued copy may still read.
    #[test]
    fn test_gpu_prepared_pinned_retirement_is_owned_by_the_native_reclaimer() {
        let plan_source = include_str!("gpu_prepared.rs");
        let global_registry = concat!("deferred_", "prepared_plans");
        assert!(
            !plan_source.contains(global_registry),
            "plan retirement must not grow an unbounded global deferred list"
        );
        let native_source = include_str!("../../cuda/src/matrix/MatrixSerde.cu");
        for transfer in [
            "gpu_matrix_defer_const_coeff_readback_pinned_free",
            "gpu_matrix_defer_rns_upload_pinned_free",
        ] {
            assert!(plan_source.contains(transfer), "{transfer} must retire a pinned allocation");
            let body = cuda_function_body(native_source, transfer);
            assert!(
                body.contains("defer_pinned_free_behind_plan_streams"),
                "{transfer} must retire the allocation through its submission streams"
            );
        }
        // The shared retirement records one fresh terminal event per submission
        // stream and hands the pointer to the context-owned reclaimer.
        let retirement = cuda_function_body(native_source, "defer_pinned_free_behind_plan_streams");
        assert!(
            retirement.contains("serde_build_event_set_from_streams"),
            "pinned retirement must record a terminal event for every submission stream"
        );
        assert!(
            retirement.contains("gpu_event_set_defer_pinned_free"),
            "pinned retirement must hand the pointer to the context-owned reclaimer"
        );
    }

    /// Nothing waits on a schedule boundary that no caller connects to.
    #[test]
    fn test_gpu_prepared_schedule_exposes_no_unused_wait_boundary() {
        assert!(!include_str!("gpu_schedule.rs").contains("gpu_prepared_schedule_wait"));
        assert!(
            !include_str!("../../cuda/src/matrix/gpu_schedule.cu")
                .contains("gpu_prepared_schedule_wait")
        );
    }

    #[test]
    fn test_gpu_prepared_arithmetic_layout_covers_grid_and_stream_metadata() {
        let add = super::PreparedArithmeticLayout::query(
            &super::GpuPreparedArithmeticKind::Add,
            1024,
            4,
            3,
            5,
            3,
            5,
            3,
            5,
            0,
            0,
            0,
            2,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(add.grid, [((3 * 5 * 1024) as u32).div_ceil(256), 1, 4]);
        assert_eq!(add.block, [256, 1, 1]);
        assert_eq!(add.event_count, 4);
        let multiply = super::PreparedArithmeticLayout::query(
            &super::GpuPreparedArithmeticKind::Multiply,
            1024,
            2,
            1,
            32,
            32,
            8,
            1,
            8,
            0,
            0,
            0,
            0,
            true,
            true,
            true,
        )
        .unwrap();
        assert!(multiply.thin);
        assert!(multiply.lazy_reduction);
    }

    #[test]
    fn test_gpu_prepared_layout_queries_are_native_pure_planning() {
        let ntt = include_str!("../../cuda/src/matrix/MatrixNTT.cu");
        let arithmetic = include_str!("../../cuda/src/matrix/MatrixArith.cu");
        for (source, name, marker) in [
            // The NTT file has a legacy prepare helper between the pure query
            // and the exported prepare entry point; stop at that helper so
            // this check covers only the query definition.
            (ntt, "gpu_matrix_query_ntt_layout", "\nstatic int prepare_ntt_plan_legacy_impl"),
            (arithmetic, "gpu_matrix_query_arithmetic_layout", "\n    struct PreparedMatmulLaunch"),
        ] {
            // Match the native definition, not an earlier call site.  The
            // NTT prepare path queries this pure layout helper before the
            // definition; slicing from that call would incorrectly include
            // the prepare function's stream handle in this source contract.
            let start = source.find(&format!("extern \"C\" int {name}")).unwrap();
            let end = source[start..].find(marker).map(|offset| start + offset).unwrap();
            let body = &source[start..end];
            for forbidden in
                ["cudaMalloc", "cudaFree", "cudaEvent", "cudaStream", "cudaMemcpy", "<<<"]
            {
                assert!(!body.contains(forbidden), "{name} must not call {forbidden}");
            }
        }
    }

    #[test]
    fn test_gpu_prepared_rect_layouts_cover_views_and_limb_capacity() {
        let input = super::PreparedRectLayout::query_input_copy(4096, 6, 3, 7, 12).unwrap();
        assert_eq!(input, super::PreparedRectLayout::query_input_copy(4096, 6, 3, 7, 12).unwrap());
        assert_eq!(input.grid, [((3 * 7 * 4096) as u32).div_ceil(256), 1, 6]);
        assert_eq!(input.stage_role, 1);
        let transpose = super::PreparedRectLayout::query_transpose(4096, 2, 11, 5, 12).unwrap();
        assert_eq!(transpose.grid[2], 2);
        assert_eq!(transpose.stage_role, 2);
    }

    #[test]
    fn test_composite_bundles_preserve_bind_order_and_native_consumption() {
        let source = include_str!("../../cuda/src/matrix/MatrixArith.cu");
        assert!(source.contains("gpu_matrix_prepare_arithmetic_with_layout"));
        assert!(source.contains("gpu_prepared_require_allocation"));
        let compact = include_str!("../../cuda/src/matrix/gpu_compact_decompose.cu");
        assert!(compact.contains("gpu_small_matrix_prepare_decompose_with_layout"));
        assert!(compact.contains("layout->allocation_count"));
        let bundle = super::GpuPreparedAccumulateLayout::new(Vec::new());
        assert!(bundle.is_err(), "an empty accumulate tape cannot be bound");
    }
}
