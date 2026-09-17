//! Native allocation requirements for matrix decomposition and CRT transforms.
//!
//! These are allocator requests from the production layout calculation. Device
//! allocator rounding and opaque driver resources must be admitted separately;
//! neither a request nor an observed pool peak is a complete memory bound.

use super::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout};
use crate::{
    matrix::SmallMatrixError,
    poly::{
        PolyParams,
        dcrt::gpu::{
            GPU_POLY_FORMAT_COEFF, GPU_POLY_FORMAT_EVAL, GpuContextOpaque, GpuDCRTPolyParams,
            GpuMatrixAllocationBytes, last_error_string,
        },
    },
};
use num_bigint::BigUint;
use std::ffi::c_int;

/// Compact digit layout shared by owned, borrowed and retained decomposition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuCompactDecompositionLayout {
    pub rows_per_input_row: usize,
    pub dropped_moduli: usize,
    pub max_coefficient_bound: BigUint,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuMatrixTransformWorkspaceBytes {
    /// Entire device metadata layout, including storage borrowed from the output.
    pub workspace_bytes: usize,
    /// Additional device allocator request; borrowed output auxiliary bytes add zero.
    pub additional_bytes: usize,
    /// Host staging request, retired after its asynchronous transfer completes.
    pub pinned_bytes: usize,
    pub alignment: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuMatrixDecomposeWorkspaceBytes {
    /// Full coefficient-domain copy when the original cannot be read directly.
    pub coefficient_copy: GpuMatrixAllocationBytes,
    /// Correction metadata; excludes bytes already present in coefficient_copy.
    pub correction: GpuMatrixTransformWorkspaceBytes,
    pub output_rows: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMatrixCrtOperation {
    ConvertModulus = 0,
    CenteredRebase = 1,
    Recompose = 2,
    RnsConversion = 3,
}

unsafe extern "C" {
    fn gpu_matrix_query_small_rhs_workspace_bytes(
        ctx: *const GpuContextOpaque,
        level: c_int,
        inner: usize,
        columns: usize,
        narrow_bytes: *mut usize,
        wide_bytes: *mut usize,
    ) -> c_int;

    fn gpu_matrix_query_decompose_workspace_bytes(
        ctx: *const GpuContextOpaque,
        level: c_int,
        rows: usize,
        columns: usize,
        format: c_int,
        base_bits: u32,
        small: c_int,
        dropped_moduli: usize,
        out: *mut GpuMatrixDecomposeWorkspaceBytes,
    ) -> c_int;
    fn gpu_matrix_query_gadget_correction_workspace_bytes(
        ctx: *const GpuContextOpaque,
        level: c_int,
        rows: usize,
        columns: usize,
        dropped_moduli: usize,
        out: *mut GpuMatrixTransformWorkspaceBytes,
    ) -> c_int;
    fn gpu_matrix_query_crt_workspace_bytes(
        ctx: *const GpuContextOpaque,
        level: c_int,
        rows: usize,
        columns: usize,
        operation: GpuMatrixCrtOperation,
        source_limb_count: usize,
        level_count: usize,
        with_views: bool,
        out: *mut GpuMatrixTransformWorkspaceBytes,
    ) -> c_int;
}

impl GpuDCRTPolyParams {
    pub fn compact_decomposition_layout(
        &self,
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<GpuCompactDecompositionLayout, SmallMatrixError> {
        let base = BigUint::from(1u8) << self.base_bits();
        let max_coefficient_bound =
            if small { &base - BigUint::from(1u8) } else { (&base + BigUint::from(1u8)) >> 1 };
        let default_digits = if small {
            self.crt_bits().div_ceil(self.base_bits() as usize)
        } else {
            self.modulus_digits()
        };
        let digits = digit_count.unwrap_or(default_digits);
        let dropped_moduli = if small {
            if digits != default_digits {
                return Err(SmallMatrixError::InvalidConfig);
            }
            self.dropped_moduli()
        } else {
            self.gadget_dropped_moduli(Some(digits)).ok_or(SmallMatrixError::InvalidConfig)?
        };
        if !small && dropped_moduli > 0 && !self.supports_shared_crt_correction() {
            return Err(SmallMatrixError::InvalidConfig);
        }
        Ok(GpuCompactDecompositionLayout {
            rows_per_input_row: digits,
            dropped_moduli,
            max_coefficient_bound,
        })
    }

    /// Exact expanded compact RHS spans in native allocation order. Every span
    /// covers all rows, active limbs of its word width, and the current columns.
    pub fn small_rhs_workspaces(
        &self,
        level: usize,
        inner: usize,
        columns: usize,
    ) -> Result<Vec<GpuPreparedWorkspaceLayout>, String> {
        let level = c_int::try_from(level).map_err(|_| "compact RHS level overflow")?;
        let (mut narrow, mut wide) = (0, 0);
        let status = unsafe {
            gpu_matrix_query_small_rhs_workspace_bytes(
                self.ctx_raw(),
                level,
                inner,
                columns,
                &mut narrow,
                &mut wide,
            )
        };
        if status != 0 {
            return Err(crate::poly::dcrt::gpu::last_error_string());
        }
        Ok([(narrow, std::mem::align_of::<u32>()), (wide, std::mem::align_of::<u64>())]
            .into_iter()
            .filter(|(bytes, _)| *bytes != 0)
            .map(|(bytes, alignment)| GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompactWorkspace,
                bytes,
                alignment,
            })
            .collect())
    }

    /// Query before constructing either the source copy or retained output.
    /// `small` refers to the ordinary matrix small-decomposition operation.
    pub fn matrix_decompose_workspace_bytes(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
        is_ntt: bool,
        base_bits: u32,
        small: bool,
        dropped_moduli: usize,
    ) -> Result<GpuMatrixDecomposeWorkspaceBytes, String> {
        let level = c_int::try_from(level).map_err(|_| "decomposition level overflow")?;
        let mut plan = GpuMatrixDecomposeWorkspaceBytes::default();
        let status = unsafe {
            gpu_matrix_query_decompose_workspace_bytes(
                self.ctx_raw(),
                level,
                rows,
                columns,
                if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF },
                base_bits,
                c_int::from(small),
                dropped_moduli,
                &mut plan,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(plan)
    }

    pub fn matrix_gadget_correction_workspace_bytes(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
        dropped_moduli: usize,
    ) -> Result<GpuMatrixTransformWorkspaceBytes, String> {
        let level = c_int::try_from(level).map_err(|_| "gadget correction level overflow")?;
        let mut plan = GpuMatrixTransformWorkspaceBytes::default();
        let status = unsafe {
            gpu_matrix_query_gadget_correction_workspace_bytes(
                self.ctx_raw(),
                level,
                rows,
                columns,
                dropped_moduli,
                &mut plan,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(plan)
    }

    /// The receiver describes the output. Source owners, retained output and
    /// parameter tables are separate from this temporary metadata request.
    /// `with_views` selects retained conversion ranges, which use setup-owned
    /// CRT tables and bounded launch arguments instead of temporary metadata.
    pub fn matrix_crt_workspace_bytes(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
        operation: GpuMatrixCrtOperation,
        source_limb_count: usize,
        level_count: usize,
        with_views: bool,
    ) -> Result<GpuMatrixTransformWorkspaceBytes, String> {
        let level = c_int::try_from(level).map_err(|_| "CRT transform level overflow")?;
        let mut plan = GpuMatrixTransformWorkspaceBytes::default();
        let status = unsafe {
            gpu_matrix_query_crt_workspace_bytes(
                self.ctx_raw(),
                level,
                rows,
                columns,
                operation,
                source_limb_count,
                level_count,
                with_views,
                &mut plan,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_transform_decompose_query_matches_shapes_and_pending_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        for dropped in [0, 1] {
            let cpu = DCRTPolyParams::new(n, 3, 17, 4, None, Some(dropped));
            let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, Some(dropped));
            for (rows, columns) in [(2, 3), (5, 5)] {
                let input = DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    rows,
                    columns,
                    DistType::TernaryDist,
                );
                for is_ntt in [false, true] {
                    for small in [false, true] {
                        if small && dropped != 0 {
                            continue;
                        }
                        let plan = params
                            .matrix_decompose_workspace_bytes(
                                params.crt_depth() - 1,
                                rows,
                                columns,
                                is_ntt,
                                params.base_bits(),
                                small,
                                dropped,
                            )
                            .unwrap();
                        assert_eq!(plan.coefficient_copy.total_bytes != 0, is_ntt || dropped != 0);
                        assert_eq!(plan.correction.additional_bytes, 0);
                        let mut source = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
                        if !is_ntt {
                            source.intt_all_in_place();
                        }
                        let (output, expected) = if small {
                            (source.small_decompose(), input.small_decompose())
                        } else {
                            (source.decompose(), input.decompose())
                        };
                        assert_eq!(output.row_size(), plan.output_rows);
                        let reader = output.transpose();
                        drop(source);
                        drop(output);
                        assert_eq!(
                            reader.to_cpu_matrix(),
                            expected.transpose(),
                            "small={small}, dropped={dropped}, is_ntt={is_ntt}, shape=({rows},{columns})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_transform_small_decompose_preserves_each_mixed_width_residue() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let negative = -crate::matrix::dcrt_poly::DCRTPolyMatrix::identity(&cpu, 2, None);
            let random =
                DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
            for input in [negative, random] {
                for is_ntt in [false, true] {
                    let mut source = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
                    if !is_ntt {
                        source.intt_all_in_place();
                    }
                    let output = source.small_decompose();
                    let reader = output.transpose();
                    drop(source);
                    drop(output);
                    assert_eq!(reader.to_cpu_matrix(), input.small_decompose().transpose());
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_transform_crt_arena_classes_preserve_pending_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let target_cpu = DCRTPolyParams::new(n, 2, 17, 4, Some(cpu.to_crt().0[..2].to_vec()), None);
        let target = GpuDCRTPolyParams::new(n, target_cpu.to_crt().0, 4, None);
        let source_params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            target.device_ids(),
            Some(1),
            Some(&target),
            None,
        );
        let plan = |columns| {
            target
                .matrix_crt_workspace_bytes(
                    target.crt_depth() - 1,
                    1,
                    columns,
                    GpuMatrixCrtOperation::ConvertModulus,
                    source_params.crt_depth(),
                    1,
                    false,
                )
                .unwrap()
        };
        assert!(plan(1).additional_bytes > 0);
        let mut borrowed_columns = 2usize;
        while plan(borrowed_columns).additional_bytes != 0 {
            borrowed_columns = borrowed_columns.checked_mul(2).unwrap();
            assert!(
                borrowed_columns <= 4096,
                "bounded metadata must fit an ordinary auxiliary owner"
            );
        }
        let mut readers = Vec::new();
        for columns in [1, borrowed_columns] {
            let requirements = plan(columns);
            assert_eq!(requirements.pinned_bytes, requirements.workspace_bytes);
            let input = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                1,
                columns,
                DistType::FinRingDist,
            );
            let expected = input.reduce_modulus(&target_cpu).transpose();
            let source = GpuDCRTPolyMatrix::from_cpu_matrix(&source_params, &input);
            let output = source.reduce_modulus(&target);
            let reader = output.transpose();
            drop(source);
            drop(output);
            readers.push((reader, expected));
        }
        // Both arena classes and their source owners are released while the
        // downstream readers remain pending. Trusted CPU primitives are the oracle.
        for (reader, expected) in readers {
            assert_eq!(reader.to_cpu_matrix(), expected);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_transform_queries_reject_unsupported_shapes_before_allocation() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let level = params.crt_depth() - 1;
        for (rows, columns, base_bits, small, dropped) in [
            (usize::MAX, 2, 4, false, 0),
            (2, 3, 0, false, 0),
            (2, 3, 4, true, 1),
            (2, 3, 4, false, level + 1),
        ] {
            assert!(
                params
                    .matrix_decompose_workspace_bytes(
                        level, rows, columns, true, base_bits, small, dropped,
                    )
                    .is_err()
            );
        }
        assert!(
            params
                .matrix_crt_workspace_bytes(
                    usize::MAX,
                    2,
                    3,
                    GpuMatrixCrtOperation::Recompose,
                    3,
                    2,
                    false,
                )
                .is_err()
        );
    }
}
