//! Fixed preimage acceptance and first-success publication.

use super::{GpuDCRTPolyMatrix, GpuSmallMatrix};
use crate::poly::{
    PolyParams,
    dcrt::gpu::{
        GpuPreparedPreimageCutoffOpaque, PinnedHostBuffer, gpu_small_matrix_begin_preimage_cutoff,
        gpu_small_matrix_destroy_preimage_cutoff, gpu_small_matrix_finish_preimage_cutoff,
        gpu_small_matrix_prepare_preimage_cutoff, gpu_small_matrix_submit_preimage_cutoff,
        gpu_small_matrix_wait_preimage_cutoff, last_error_string,
    },
};
use std::{ptr::NonNull, sync::Arc};

/// All geometry and owners are fixed at preparation. Each submission checks
/// the current candidates, and writes only destinations that have not already
/// accepted a candidate in this invocation. The sampler retains its existing
/// retry limit; this primitive neither chooses nor changes that limit.
pub struct GpuPreparedPreimageCutoff {
    raw: NonNull<GpuPreparedPreimageCutoffOpaque>,
    sources: Box<[Arc<GpuDCRTPolyMatrix>]>,
    destinations: Box<[Arc<GpuSmallMatrix>]>,
    status: PinnedHostBuffer<i32>,
    finished: bool,
}

unsafe impl Send for GpuPreparedPreimageCutoff {}

impl GpuPreparedPreimageCutoff {
    /// Metadata-only cutoff planner used by the prepared resource resolver.
    /// It mirrors the native geometry calculation without requiring a live
    /// compact destination owner.
    pub fn allocation_layout_for_shape(
        params: &crate::poly::dcrt::gpu::GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        magnitude_bytes: usize,
        job_count: usize,
    ) -> Result<Vec<super::GpuPreparedWorkspaceLayout>, String> {
        if job_count == 0 {
            return Err("prepared preimage cutoff job count is zero".into());
        }
        let mut layouts =
            vec![
                super::GpuPreparedWorkspaceLayout {
                    bytes: 0,
                    alignment: 1,
                    kind: super::GpuPreparedSlotKind::CompactWorkspace,
                };
                job_count.checked_add(3).ok_or("prepared preimage cutoff layout count overflow")?
            ];
        let mut count = 0;
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_preimage_cutoff_batch_layout_shape(
                params.ring_dimension() as usize,
                rows,
                columns,
                magnitude_bytes,
                job_count,
                layouts.as_mut_ptr(),
                layouts.len(),
                &mut count,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(layouts[..count].to_vec())
    }

    pub fn allocation_layout(
        output: &GpuSmallMatrix,
        job_count: usize,
    ) -> Result<Vec<super::GpuPreparedWorkspaceLayout>, String> {
        if job_count == 0 {
            return Err("prepared preimage cutoff job count is zero".into());
        }
        let mut layouts =
            vec![
                super::GpuPreparedWorkspaceLayout {
                    bytes: 0,
                    alignment: 1,
                    kind: super::GpuPreparedSlotKind::CompactWorkspace,
                };
                job_count.checked_add(3).ok_or("prepared preimage cutoff layout count overflow")?
            ];
        let mut count = 0;
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_preimage_cutoff_batch_layout(
                output.raw,
                job_count,
                layouts.as_mut_ptr(),
                layouts.len(),
                &mut count,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(layouts[..count].to_vec())
    }
    pub(crate) fn native_raw(&self) -> *mut GpuPreparedPreimageCutoffOpaque {
        self.raw.as_ptr()
    }

    pub(super) fn schedule_record(&self) -> (u32, *const std::ffi::c_void) {
        (18, self.raw.as_ptr().cast())
    }

    pub fn bind_with_layout(
        jobs: Vec<(Arc<GpuSmallMatrix>, Arc<GpuDCRTPolyMatrix>, usize, usize)>,
        layouts: &[super::GpuPreparedWorkspaceLayout],
    ) -> Result<Self, String> {
        let first = jobs.first().ok_or("prepared preimage cutoff batch is empty")?;
        let expected =
            jobs.len().checked_add(3).ok_or("prepared preimage cutoff layout count overflow")?;
        if layouts.len() != expected {
            return Err("prepared preimage cutoff saved layout count mismatch".into());
        }
        let mut status = PinnedHostBuffer::zeroed(&first.0.params, jobs.len());
        let destinations = jobs.iter().map(|job| job.0.raw).collect::<Vec<_>>();
        let sources = jobs.iter().map(|job| job.1.raw.cast_const()).collect::<Vec<_>>();
        let rows = jobs.iter().map(|job| job.2).collect::<Vec<_>>();
        let columns = jobs.iter().map(|job| job.3).collect::<Vec<_>>();
        let mut raw = std::ptr::null_mut();
        let result = unsafe {
            gpu_small_matrix_prepare_preimage_cutoff(
                destinations.as_ptr(),
                sources.as_ptr(),
                rows.as_ptr(),
                columns.as_ptr(),
                jobs.len(),
                status.as_mut_slice().as_mut_ptr(),
                layouts.as_ptr(),
                layouts.len(),
                &mut raw,
            )
        };
        if result != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("prepared preimage cutoff returned no plan")?,
            sources: jobs.iter().map(|job| Arc::clone(&job.1)).collect(),
            destinations: jobs.into_iter().map(|job| job.0).collect(),
            status,
            finished: false,
        })
    }

    pub fn begin(&mut self) -> Result<(), String> {
        let result = unsafe { gpu_small_matrix_begin_preimage_cutoff(self.raw.as_ptr()) };
        if result != 0 {
            return Err(last_error_string());
        }
        self.finished = false;
        Ok(())
    }

    pub fn submit(&mut self) -> Result<(), String> {
        let result = unsafe { gpu_small_matrix_submit_preimage_cutoff(self.raw.as_ptr()) };
        if result != 0 { Err(last_error_string()) } else { Ok(()) }
    }

    /// The explicit host boundary. Zero status is a sampling rejection, while
    /// an error is a GPU execution failure. No host acceptance is read by submit.
    pub fn wait(&mut self) -> Result<&[i32], String> {
        if !self.finished {
            let result = unsafe { gpu_small_matrix_finish_preimage_cutoff(self.raw.as_ptr()) };
            if result != 0 {
                return Err(last_error_string());
            }
            self.finished = true;
        }
        let result = unsafe { gpu_small_matrix_wait_preimage_cutoff(self.raw.as_ptr()) };
        if result != 0 { Err(last_error_string()) } else { Ok(self.status.as_slice()) }
    }

    pub fn sources(&self) -> &[Arc<GpuDCRTPolyMatrix>] {
        &self.sources
    }

    /// Starts the preallocated status readback once, then queries its event.
    /// This never waits for device completion.
    pub fn poll(&mut self) -> Result<Option<&[i32]>, String> {
        if !self.finished {
            let status = unsafe { gpu_small_matrix_finish_preimage_cutoff(self.raw.as_ptr()) };
            if status != 0 {
                return Err(last_error_string());
            }
            self.finished = true;
        }
        let mut ready = false;
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_preimage_cutoff_is_ready(self.raw.as_ptr(), &mut ready)
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(ready.then(|| self.status.as_slice()))
    }

    pub fn destinations(&self) -> &[Arc<GpuSmallMatrix>] {
        &self.destinations
    }
}

impl Drop for GpuPreparedPreimageCutoff {
    fn drop(&mut self) {
        // Pinned readback is only submitted by wait. Retain its storage through
        // completion even if a preceding wait reported a GPU error.
        if self.finished {
            unsafe { gpu_small_matrix_wait_preimage_cutoff(self.raw.as_ptr()) };
        }
        unsafe { gpu_small_matrix_destroy_preimage_cutoff(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::{PolyMatrix, SmallPolyMatrix, gpu_dcrt_poly::GpuPreparedInputCopy},
        poly::{
            Poly,
            dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
        },
    };
    use num_bigint::BigUint;
    use rand::Rng;

    #[test]
    #[serial_test::serial]
    fn test_gpu_prepared_preimage_cutoff_preserves_first_success_and_reuses_status() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let params = GpuDCRTPolyParams::new(n, vec![131_009], 2, None);
        let accepted_value = rand::rng().random_range(2usize..50);
        let bound = BigUint::from(64u32);
        let source = |value| {
            Arc::new(
                GpuDCRTPolyMatrix::from_poly_vec_row(
                    &params,
                    vec![GpuDCRTPoly::from_usize_to_constant(&params, value)],
                )
                .into_coeff_domain(),
            )
        };
        let initial = [source(accepted_value), source(100), source(101)];
        let replacements = [source(accepted_value + 1), source(accepted_value), source(102)];
        let make_output = || {
            GpuSmallMatrix::from_canonical_coefficients(
                &params,
                1,
                1,
                bound.clone(),
                &vec![0; n as usize * 2],
            )
            .unwrap()
        };
        let actual = (0..3).map(|_| Arc::new(make_output())).collect::<Vec<_>>();
        let mut expected = (0..3).map(|_| make_output()).collect::<Vec<_>>();
        for value in &expected {
            value.prepare_preimage_hard_cutoff();
        }
        assert!(expected[0].try_pack_preimage_hard_cutoff_tile(&initial[0], 0, 0, 1, 1).unwrap());
        assert!(
            expected[1].try_pack_preimage_hard_cutoff_tile(&replacements[1], 0, 0, 1, 1).unwrap()
        );
        assert!(
            !expected[2].try_pack_preimage_hard_cutoff_tile(&replacements[2], 0, 0, 1, 1).unwrap()
        );
        let copies = initial
            .iter()
            .zip(&replacements)
            .map(|(output, input)| {
                GpuPreparedInputCopy::bind(Arc::clone(output), Arc::clone(input), None).unwrap()
            })
            .collect::<Vec<_>>();
        let jobs = actual
            .iter()
            .zip(&initial)
            .map(|(dst, src)| (Arc::clone(dst), Arc::clone(src), 0, 0))
            .collect::<Vec<_>>();
        let layouts =
            GpuPreparedPreimageCutoff::allocation_layout(actual[0].as_ref(), jobs.len()).unwrap();
        let mut command = GpuPreparedPreimageCutoff::bind_with_layout(jobs, &layouts).unwrap();
        command.begin().unwrap();
        command.submit().unwrap();
        for (copy, replacement) in copies.iter().zip(&replacements) {
            copy.submit_borrowed(replacement).unwrap();
        }
        command.submit().unwrap();
        assert_eq!(command.wait().unwrap(), &[1, 1, 0]);
        for (output, expected) in actual.iter().zip(&expected) {
            assert_eq!(
                output.to_canonical_coefficients().unwrap(),
                expected.to_canonical_coefficients().unwrap(),
            );
        }
        command.begin().unwrap();
        command.submit().unwrap();
        assert_eq!(command.wait().unwrap(), &[1, 1, 0]);
        assert!(
            expected[0].try_pack_preimage_hard_cutoff_tile(&replacements[0], 0, 0, 1, 1).unwrap()
        );
        assert_eq!(
            actual[0].to_canonical_coefficients().unwrap(),
            expected[0].to_canonical_coefficients().unwrap(),
        );
        // Sequential replay cannot erase an earlier exhausted invocation.
        command.begin().unwrap();
        command.submit().unwrap();
        copies[2].submit_borrowed(&replacements[1]).unwrap();
        command.begin().unwrap();
        command.submit().unwrap();
        assert_eq!(command.wait().unwrap(), &[1, 1, 0]);
        command.begin().unwrap();
        command.submit().unwrap();
        assert_eq!(command.wait().unwrap(), &[1, 1, 1]);
    }
}
