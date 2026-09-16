//! Slot-owned Gaussian preimage phases, including content-dependent covariance.

use super::GpuDCRTPolyMatrix;
use crate::poly::dcrt::gpu::{
    GpuPreparedPreimagePhasesOpaque, GpuRngSeed, gpu_preimage_destroy_phases,
    gpu_preimage_prepare_phases, gpu_preimage_refresh_covariance, gpu_preimage_submit_gadget,
    gpu_preimage_submit_p1, last_error_string,
};
use std::{ptr::NonNull, sync::Arc};

pub struct GpuPreparedPreimagePhases {
    raw: NonNull<GpuPreparedPreimagePhasesOpaque>,
    owners: [Arc<GpuDCRTPolyMatrix>; 7],
}

unsafe impl Send for GpuPreparedPreimagePhases {}

impl GpuPreparedPreimagePhases {
    pub fn allocation_layout(
        params: &crate::poly::dcrt::gpu::GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
    ) -> Result<[super::GpuPreparedWorkspaceLayout; 6], String> {
        let mut layouts = std::mem::MaybeUninit::<[super::GpuPreparedWorkspaceLayout; 6]>::uninit();
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_preimage_phase_layout(
                params.ctx_raw(),
                rows,
                columns,
                layouts.as_mut_ptr().cast(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(unsafe { layouts.assume_init() })
    }
    /// # Safety
    /// Preparation only. The cutoff must outlive this plan and share its
    /// exclusive invocation; it cannot be reset until all readers complete.
    pub(crate) unsafe fn bind_acceptance(
        &mut self,
        cutoff: &super::GpuPreparedPreimageCutoff,
        job: usize,
    ) -> Result<(), String> {
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_preimage_mask_phases(
                self.raw.as_ptr(),
                cutoff.native_raw(),
                job,
            )
        };
        if status == 0 { Ok(()) } else { Err(last_error_string()) }
    }

    /// Owners are Gram A/B/D, perturbation product, P1 output, residual, gadget
    /// output. Contents may change each invocation, but their storage is fixed.
    pub fn bind(
        owners: [Arc<GpuDCRTPolyMatrix>; 7],
        base_bits: u32,
        c: f64,
        smoothing: f64,
        sigma: f64,
    ) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_preimage_prepare_phases(
                owners[0].raw,
                owners[1].raw,
                owners[2].raw,
                owners[3].raw,
                owners[4].raw,
                owners[5].raw,
                owners[6].raw,
                base_bits,
                c,
                smoothing,
                sigma,
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("prepared preimage phases returned no plan")?,
            owners,
        })
    }

    pub fn refresh_covariance(&mut self) -> Result<(), String> {
        let status = unsafe { gpu_preimage_refresh_covariance(self.raw.as_ptr()) };
        if status == 0 { Ok(()) } else { Err(last_error_string()) }
    }

    pub fn sample_p1(&mut self, seed: GpuRngSeed) -> Result<(), String> {
        let status = unsafe { gpu_preimage_submit_p1(self.raw.as_ptr(), seed) };
        if status == 0 { Ok(()) } else { Err(last_error_string()) }
    }

    pub fn sample_gadget(&mut self, seed: GpuRngSeed) -> Result<(), String> {
        let status = unsafe { gpu_preimage_submit_gadget(self.raw.as_ptr(), seed) };
        if status == 0 { Ok(()) } else { Err(last_error_string()) }
    }

    pub fn owners(&self) -> &[Arc<GpuDCRTPolyMatrix>; 7] {
        &self.owners
    }

    pub(super) fn schedule_record(&self) -> (u32, *const std::ffi::c_void) {
        (19, self.raw.as_ptr().cast())
    }
}

impl Drop for GpuPreparedPreimagePhases {
    fn drop(&mut self) {
        unsafe { gpu_preimage_destroy_phases(self.raw.as_ptr()) };
    }
}
