//! Destination-aware compact decomposition of an exclusive scratch matrix.

use super::*;

enum CompactDecomposeOpaque {}
unsafe extern "C" {
    fn gpu_small_matrix_prepare_decompose(
        source: *mut crate::poly::dcrt::gpu::GpuMatrixOpaque,
        output: *mut crate::poly::dcrt::gpu::GpuSmallMatrixOpaque,
        base_bits: u32,
        small: bool,
        dropped: usize,
        out: *mut *mut CompactDecomposeOpaque,
    ) -> i32;
    fn gpu_small_matrix_prepare_decompose_with_layout(
        source: *mut crate::poly::dcrt::gpu::GpuMatrixOpaque,
        output: *mut crate::poly::dcrt::gpu::GpuSmallMatrixOpaque,
        base_bits: u32,
        small: bool,
        dropped: usize,
        layout: *const std::ffi::c_void,
        out: *mut *mut CompactDecomposeOpaque,
    ) -> i32;
    fn gpu_small_matrix_submit_decompose(plan: *const CompactDecomposeOpaque) -> i32;
    fn gpu_small_matrix_destroy_decompose(plan: *mut CompactDecomposeOpaque);
}

/// The source is exclusively owned scratch within a prepared execution slot:
/// its producer must overwrite every coefficient before each submission.
/// This plan may inverse-transform/correct the source in place, then emits the
/// same compact representation as `gadget_decompose` into its fixed owner.
pub struct GpuPreparedCompactDecompose {
    raw: NonNull<CompactDecomposeOpaque>,
    source: Arc<GpuDCRTPolyMatrix>,
    output: Arc<GpuSmallMatrix>,
}

unsafe impl Send for GpuPreparedCompactDecompose {}
unsafe impl Sync for GpuPreparedCompactDecompose {}

impl GpuPreparedCompactDecompose {
    /// Allocate under the caller's prepared CompactPayload dispatch permit.
    /// No zero/fill kernels or temporary ordinary matrix are constructed.
    pub fn allocate_output(
        source: &GpuDCRTPolyMatrix,
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Arc<GpuSmallMatrix>, String> {
        let layout = source
            .params()
            .compact_decomposition_layout(small, digit_count)
            .map_err(|error| error.to_string())?;
        let rows = source
            .row_size()
            .checked_mul(layout.rows_per_input_row)
            .ok_or("compact decomposition row count overflow")?;
        GpuSmallMatrix::new_empty(
            source.params(),
            rows,
            source.col_size(),
            layout.max_coefficient_bound,
        )
        .map(Arc::new)
        .map_err(|error| error.to_string())
    }

    pub fn bind(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuSmallMatrix>,
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Arc<Self>, String> {
        let layout = source
            .params()
            .compact_decomposition_layout(small, digit_count)
            .map_err(|error| error.to_string())?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_small_matrix_prepare_decompose(
                source.raw,
                output.raw,
                source.params().base_bits(),
                small,
                layout.dropped_moduli,
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native compact plan missing")?,
            source,
            output,
        }))
    }

    /// Bind compact decomposition against the immutable descriptor assembled
    /// during warmup. The native entry point validates the saved compact
    /// payload, correction workspace, inverse-transform staging, and stream
    /// claims before publishing the executable plan.
    pub fn bind_with_layout(
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuSmallMatrix>,
        small: bool,
        digit_count: Option<usize>,
        plan_layout: &crate::matrix::gpu_dcrt_poly::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let layout = source
            .params()
            .compact_decomposition_layout(small, digit_count)
            .map_err(|error| error.to_string())?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_small_matrix_prepare_decompose_with_layout(
                source.raw,
                output.raw,
                source.params().base_bits(),
                small,
                layout.dropped_moduli,
                plan_layout.native_ptr().cast(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("native compact plan missing")?,
            source,
            output,
        }))
    }

    pub fn submit(&self) -> Result<(), String> {
        let status = unsafe { gpu_small_matrix_submit_decompose(self.raw.as_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }

    pub fn output(&self) -> &Arc<GpuSmallMatrix> {
        &self.output
    }

    pub(super) fn schedule_record(&self) -> (u32, *const std::ffi::c_void) {
        (12, self.raw.as_ptr().cast())
    }

    pub(super) fn source_owner(&self) -> &Arc<GpuDCRTPolyMatrix> {
        &self.source
    }
}

impl Drop for GpuPreparedCompactDecompose {
    fn drop(&mut self) {
        unsafe { gpu_small_matrix_destroy_decompose(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::{PolyMatrixSmallRhs, SmallPolyMatrix},
        poly::dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
        sampler::gpu::{hash_seed_for_matrix, sample_seeded_gadget_source_columns},
    };
    use rand::Rng;

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_compact_hash_replay_matches_trusted_decomposition() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse().expect("ring dimension"))
            .unwrap_or(32);
        let crt_bits = std::env::var("MXX_PRIMITIVE_TEST_CRT_BITS")
            .map(|v| v.parse().expect("CRT bits"))
            .unwrap_or(17);
        for dropped in [0, 1] {
            let cpu = DCRTPolyParams::new(n, 2, crt_bits, 4, None, Some(dropped));
            let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, Some(dropped));
            for small in [false, true] {
                for evaluation in [false, true] {
                    let source = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                        &params,
                        2,
                        2,
                        params.crt_depth() - 1,
                        evaluation,
                        None,
                    ));
                    let output =
                        GpuPreparedCompactDecompose::allocate_output(&source, small, None).unwrap();
                    let plan = GpuPreparedCompactDecompose::bind(
                        Arc::clone(&source),
                        Arc::clone(&output),
                        small,
                        None,
                    )
                    .unwrap();
                    let lhs_seed =
                        crate::poly::dcrt::gpu::GpuRngSeed::from_bytes(rand::rng().random());
                    let lhs = Arc::new(
                        sample_seeded_gadget_source_columns(
                            &params,
                            1,
                            output.rows,
                            0,
                            output.rows,
                            lhs_seed,
                        )
                        .ensure_eval_domain(),
                    );
                    let product = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                        &params,
                        1,
                        output.columns,
                        params.crt_depth() - 1,
                        true,
                        None,
                    ));
                    let multiply = GpuPreparedSmallRhs::bind(
                        Arc::clone(&product),
                        Arc::clone(&lhs),
                        Arc::clone(&output),
                        usize::MAX,
                    )
                    .unwrap();
                    let hash = GpuPreparedHashSample::bind(
                        Arc::clone(&source),
                        rand::rng().random(),
                        b"prepared-compact",
                        GpuMatrixSampleDist::Uniform,
                        0.0,
                        u64::MAX,
                        5,
                        2,
                        None,
                    )
                    .unwrap();
                    // Different runtime keys/tags prove that replay replaces payload,
                    // rather than exposing the last invocation's compact owner.
                    for invocation in 0u64..3 {
                        let key = rand::rng().random();
                        let tag = invocation.to_be_bytes();
                        let seed = hash_seed_for_matrix::<keccak_asm::Keccak256>(key, &tag);
                        let expected_compact =
                            sample_seeded_gadget_source_columns(&params, 2, 5, 2, 2, seed)
                                .gadget_decompose(small, None)
                                .unwrap();
                        let expected_product =
                            lhs.multiply_small_rhs(&expected_compact).unwrap().to_cpu_matrix();
                        let expected = expected_compact.to_canonical_coefficients().unwrap();
                        #[cfg(feature = "gpu-instrumentation")]
                        {
                            crate::poly::dcrt::gpu::gpu_test_reset_work_counters();
                            crate::poly::dcrt::gpu::gpu_test_set_work_gate(true);
                        }
                        hash.submit_key_with_tag(key, &tag).unwrap();
                        plan.submit().unwrap();
                        multiply.submit_borrowed(&lhs).unwrap();
                        #[cfg(feature = "gpu-instrumentation")]
                        {
                            crate::poly::dcrt::gpu::gpu_test_set_work_gate(false);
                            let (events, streams, validations, allocations, kernels, measurements) =
                                crate::poly::dcrt::gpu::gpu_test_work_counters();
                            assert_eq!(
                                (events, streams, validations, allocations, measurements),
                                (0, 0, 0, 0, 0)
                            );
                            assert!(kernels >= 3);
                        }
                        assert_eq!(output.to_canonical_coefficients().unwrap(), expected);
                        assert_eq!(product.to_cpu_matrix(), expected_product);
                    }
                }
            }
        }
    }
}
