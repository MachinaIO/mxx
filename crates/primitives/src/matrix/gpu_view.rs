//! Borrowed ordinary matrix ranges. Views keep their allocation owner and its
//! native writer/reader events; creating a view allocates no device payload.

use super::*;
use crate::poly::dcrt::gpu::{GpuMatrixBatchView, GpuMatrixRange, last_error_string};
use num_bigint::BigInt;

/// Exact ordinary CRT conversion. Rounding and centered representatives match
/// the existing CPU primitives; this descriptor contains only public policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMatrixModulusConversion {
    Reduce,
    Round,
    CenteredExtend,
    BlockSwitch { plaintext_modulus: u64 },
}

impl GpuMatrixModulusConversion {
    pub fn validate(
        self,
        source: &GpuDCRTPolyParams,
        level: usize,
        target: &GpuDCRTPolyParams,
    ) -> Result<(), String> {
        if level >= source.crt_depth() ||
            source.ring_dimension() != target.ring_dimension() ||
            source.execution_owner_id() != target.execution_owner_id() ||
            source.device_ids() != target.device_ids() ||
            level >= 64 ||
            target.crt_depth() > 64
        {
            return Err("CRT conversion requires compatible dimensions, execution, placement and at most 64 active limbs".into());
        }
        let input = &source.moduli()[..=level];
        let output = target.moduli();
        let valid = match self {
            Self::Reduce | Self::Round => output.iter().all(|p| input.contains(p)),
            Self::CenteredExtend => {
                level + 1 == source.crt_depth() && input.iter().all(|p| output.contains(p))
            }
            Self::BlockSwitch { plaintext_modulus: t } => {
                level + 1 == source.crt_depth() &&
                    t >= 1 &&
                    output.len() < input.len() &&
                    output.iter().all(|p| input.contains(p)) &&
                    input.iter().all(|p| t % p != 0)
            }
        };
        if !valid {
            return Err("invalid exact CRT source/destination bases or plaintext modulus".into());
        }
        Ok(())
    }
}

/// Range-generatable constants. Global column positions refer to the complete
/// logical matrix; destination row/column offsets refer only to its storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMatrixRangeConstant {
    Zero { total_columns: usize },
    Identity,
    UnitRow { total_columns: usize, index: usize },
    UnitColumn { index: usize },
    Gadget { small: bool, digit_count: Option<usize> },
}

impl GpuMatrixRangeConstant {
    pub(super) fn validate(
        self,
        params: &GpuDCRTPolyParams,
        height: usize,
        global_column_start: usize,
        columns: usize,
    ) -> Result<(i32, usize, usize, bool, usize), String> {
        let (mode, total_columns, unit_index, small, dropped) = match self {
            GpuMatrixRangeConstant::Zero { total_columns } => (0, total_columns, 0, false, 0),
            GpuMatrixRangeConstant::Identity => (1, height, 0, false, 0),
            GpuMatrixRangeConstant::UnitRow { total_columns, index } => {
                if height != 1 || index >= total_columns {
                    return Err("invalid unit row shape or index".into());
                }
                (2, total_columns, index, false, 0)
            }
            GpuMatrixRangeConstant::UnitColumn { index } => {
                if index >= height {
                    return Err("invalid unit column index".into());
                }
                (4, 1, index, false, 0)
            }
            GpuMatrixRangeConstant::Gadget { small, digit_count } => {
                let digits_per_tower = params.crt_bits().div_ceil(params.base_bits() as usize);
                let default = if small {
                    digits_per_tower
                } else {
                    digits_per_tower
                        .checked_mul(params.crt_depth() - params.dropped_moduli())
                        .ok_or("gadget digit count overflow")?
                };
                let digits = digit_count.unwrap_or(default);
                let dropped = if small {
                    if digits != default {
                        return Err("invalid small gadget digit count".into());
                    }
                    params.dropped_moduli()
                } else {
                    params
                        .gadget_dropped_moduli(Some(digits))
                        .ok_or("invalid gadget digit count")?
                };
                (
                    3,
                    height.checked_mul(digits).ok_or("gadget column count overflow")?,
                    0,
                    small,
                    dropped,
                )
            }
        };
        if global_column_start > total_columns || columns > total_columns - global_column_start {
            return Err("constant global range is outside the full matrix".into());
        }
        Ok((mode, total_columns, unit_index, small, dropped))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GpuDCRTPolyMatrixColumnView<'a> {
    owner: &'a GpuDCRTPolyMatrix,
    range: GpuMatrixRange,
}

impl GpuDCRTPolyMatrix {
    /// Fill a retained rectangle without allocating another GPU matrix or
    /// metadata workspace. Other entries and the output format are preserved;
    /// nonzero constants require evaluation format.
    pub fn fill_constant_columns(
        &mut self,
        rows: Range<usize>,
        columns: Range<usize>,
        global_column_start: usize,
        constant: GpuMatrixRangeConstant,
    ) -> Result<(), String> {
        if rows.start > rows.end ||
            rows.end > self.nrow ||
            columns.start > columns.end ||
            columns.end > self.ncol ||
            (!self.is_ntt && !matches!(constant, GpuMatrixRangeConstant::Zero { .. }))
        {
            return Err("constant destination range or format is invalid".into());
        }
        let height = rows.end - rows.start;
        let (mode, total_columns, unit_index, small, dropped) = constant.validate(
            &self.params,
            height,
            global_column_start,
            columns.end - columns.start,
        )?;
        let range = GpuMatrixRange {
            row_start: rows.start,
            row_end: rows.end,
            column_start: columns.start,
            column_end: columns.end,
        };
        let status = unsafe {
            gpu_matrix_fill_constant_columns(
                self.raw,
                &range,
                global_column_start,
                mode,
                total_columns,
                unit_index,
                self.params.base_bits(),
                i32::from(small),
                dropped,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }

    /// Sample a logical column range directly into a retained rectangle. Rows
    /// start at zero in the logical matrix; physical offsets only select storage.
    /// Reuse one seed for all pieces of one logical sample. Coefficient/evaluation
    /// format and entries outside the destination are preserved.
    pub fn fill_distribution_columns(
        &mut self,
        rows: Range<usize>,
        columns: Range<usize>,
        total_columns: usize,
        global_column_start: usize,
        dist: GpuMatrixSampleDist,
        sigma: f64,
        max_coefficient_bound: u64,
        seed: GpuRngSeed,
    ) -> Result<(), String> {
        if rows.start > rows.end ||
            rows.end > self.nrow ||
            columns.start > columns.end ||
            columns.end > self.ncol ||
            global_column_start > total_columns ||
            columns.end - columns.start > total_columns - global_column_start ||
            (dist == GpuMatrixSampleDist::Gauss && (!sigma.is_finite() || sigma <= 0.0))
        {
            return Err("invalid sampling range or Gaussian sigma".into());
        }
        (rows.end - rows.start)
            .checked_mul(total_columns)
            .ok_or("sampling logical shape overflow")?;
        let range = GpuMatrixRange {
            row_start: rows.start,
            row_end: rows.end,
            column_start: columns.start,
            column_end: columns.end,
        };
        let status = unsafe {
            gpu_matrix_sample_distribution_columns(
                self.raw,
                dist.as_ffi(),
                sigma,
                max_coefficient_bound,
                self.params.modulus().to_u64().unwrap_or(0),
                seed,
                total_columns,
                global_column_start,
                &range,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }

    /// Fill a retained singleton from signed coefficients or native evaluation slots.
    /// CPU residue conversion is parallel; transfer and NTT stay on the GPU.
    /// The existing RNS loader owns pinned staging until DMA completion.
    pub fn fill_polynomial(
        &mut self,
        coefficients: &[BigInt],
        input_evaluation: bool,
    ) -> Result<(), String> {
        let n = self.params.ring_dimension() as usize;
        if self.size() != (1, 1) || coefficients.len() > n {
            return Err("polynomial constant requires a singleton and at most N coefficients".into());
        }
        let bytes = (self.level + 1)
            .checked_mul(n)
            .and_then(|count| count.checked_mul(8))
            .ok_or("polynomial staging size overflow")?;
        let moduli = self.params.moduli()[..=self.level]
            .par_iter()
            .map(|q| BigInt::from(*q))
            .collect::<Vec<_>>();
        let mut residues = vec![0u8; bytes];
        residues.par_chunks_exact_mut(8).enumerate().for_each(|(index, target)| {
            let value = coefficients
                .get(index % n)
                .map(|value| {
                    ((value % &moduli[index / n] + &moduli[index / n]) % &moduli[index / n])
                        .to_u64()
                        .expect("CRT residue fits u64")
                })
                .unwrap_or(0);
            target.copy_from_slice(&value.to_le_bytes());
        });
        let evaluation = self.is_ntt;
        self.load_rns_bytes(
            &residues,
            bytes,
            if input_evaluation { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF },
        );
        if evaluation != input_evaluation {
            if evaluation {
                self.ntt_all_in_place();
            } else {
                self.intt_all_in_place();
            }
        }
        Ok(())
    }

    /// Number of one-row temporary matrices in the bounded binary reduction
    /// trees. This exact allocation sequence is shared with runtime admission.
    pub fn row_reduction_intermediates(rows: &[Vec<usize>]) -> Result<usize, String> {
        rows.iter().filter(|group| group.len() > 32).try_fold(0usize, |count, group| {
            group
                .len()
                .div_ceil(32)
                .checked_mul(2)
                .and_then(|nodes| nodes.checked_sub(1))
                .and_then(|nodes| count.checked_add(nodes))
                .ok_or_else(|| "row reduction intermediate count overflow".into())
        })
    }

    pub fn column_view(
        &self,
        columns: Range<usize>,
    ) -> Result<GpuDCRTPolyMatrixColumnView<'_>, String> {
        if columns.start > columns.end || columns.end > self.ncol {
            return Err("GPU column view lies outside its storage owner".into());
        }
        Ok(GpuDCRTPolyMatrixColumnView {
            owner: self,
            range: GpuMatrixRange {
                row_start: 0,
                row_end: self.nrow,
                column_start: columns.start,
                column_end: columns.end,
            },
        })
    }
}

impl GpuDCRTPolyMatrixColumnView<'_> {
    pub fn params(&self) -> &GpuDCRTPolyParams {
        &self.owner.params
    }

    pub fn size(&self) -> (usize, usize) {
        (self.range.row_end - self.range.row_start, self.range.column_end - self.range.column_start)
    }

    /// Restrict rows relative to this view while retaining the original owner.
    pub fn row_view(mut self, rows: Range<usize>) -> Result<Self, String> {
        if rows.start > rows.end || rows.end > self.size().0 {
            return Err("GPU row view lies outside its source range".into());
        }
        self.range.row_end = self.range.row_start + rows.end;
        self.range.row_start += rows.start;
        Ok(self)
    }

    fn destination(
        &self,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
        is_ntt: bool,
        shape: (usize, usize),
    ) -> Result<(GpuDCRTPolyMatrix, GpuMatrixRange), String> {
        let (rows, columns) = shape;
        let (output, output_rows, output_columns) = match destination {
            Some(destination) => destination,
            None => (
                GpuDCRTPolyMatrix::new_empty_with_state(
                    self.params(),
                    rows,
                    columns,
                    self.owner.level,
                    is_ntt,
                    None,
                ),
                0..rows,
                0..columns,
            ),
        };
        if output_rows.start > output_rows.end ||
            output_rows.end > output.nrow ||
            output_columns.start > output_columns.end ||
            output_columns.end > output.ncol ||
            output_rows.end - output_rows.start != rows ||
            output_columns.end - output_columns.start != columns ||
            output.params != self.owner.params ||
            output.level != self.owner.level ||
            output.is_ntt != is_ntt ||
            output.raw == self.owner.raw
        {
            return Err("GPU destination range, parameters, format, or owner do not match".into());
        }
        Ok((
            output,
            GpuMatrixRange {
                row_start: output_rows.start,
                row_end: output_rows.end,
                column_start: output_columns.start,
                column_end: output_columns.end,
            },
        ))
    }

    /// Fill an optional preallocated rectangle and return its full owner.
    /// Other entries and the owner's format are preserved. None allocates an
    /// output with this view's shape; no host completion wait is introduced.
    pub fn negate(
        &self,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let (output, range) = self.destination(destination, self.owner.is_ntt, self.size())?;
        let (rows, columns) = self.size();
        if rows != 0 && columns != 0 {
            let views = GpuMatrixBatchView { left: self.range, right: self.range, output: range };
            let status = unsafe {
                gpu_matrix_negate_batch(
                    [output.raw].as_ptr(),
                    [self.owner.raw.cast_const()].as_ptr(),
                    &views,
                    1,
                )
            };
            if status != 0 {
                return Err(last_error_string());
            }
        }
        Ok(output)
    }

    /// Copy a borrowed rectangle directly into its admitted destination.
    pub fn copy(
        &self,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let Some((output, rows, columns)) = destination else {
            let (output, range) = self.destination(None, self.owner.is_ntt, self.size())?;
            return self.copy(Some((
                output,
                range.row_start..range.row_end,
                range.column_start..range.column_end,
            )));
        };
        if rows.start > rows.end ||
            rows.end > output.nrow ||
            columns.start > columns.end ||
            columns.end > output.ncol ||
            (rows.end - rows.start, columns.end - columns.start) != self.size() ||
            output.level != self.owner.level ||
            output.is_ntt != self.owner.is_ntt ||
            output.raw == self.owner.raw
        {
            return Err("copy destination shape, level, format or ownership differs".into());
        }
        if output.params.ctx_raw() == self.owner.params.ctx_raw() {
            let status = unsafe {
                gpu_matrix_copy_block(
                    output.raw,
                    self.owner.raw,
                    rows.start,
                    columns.start,
                    self.range.row_start,
                    self.range.column_start,
                    self.size().0,
                    self.size().1,
                )
            };
            if status != 0 {
                return Err(last_error_string());
            }
        } else {
            let view = GpuMatrixBatchView {
                left: self.range,
                right: self.range,
                output: GpuMatrixRange {
                    row_start: rows.start,
                    row_end: rows.end,
                    column_start: columns.start,
                    column_end: columns.end,
                },
            };
            let mut copied = 0;
            let status =
                unsafe { gpu_matrix_copy_peer(output.raw, self.owner.raw, &mut copied, &view) };
            if status != 0 {
                return Err(last_error_string());
            }
            if copied == 0 {
                return Err("peer transport is unavailable for the prepared matrix layout".into());
            }
        }
        Ok(output)
    }

    /// Add stacked row views to this RHS in bounded native batches. Every batch
    /// fills its disjoint destination rows while retaining one output owner.
    pub fn add_row_blocks(
        &self,
        blocks: &[Self],
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let rows = blocks
            .iter()
            .try_fold(0usize, |rows, block| rows.checked_add(block.size().0))
            .ok_or("row block height overflow")?;
        if blocks.is_empty() ||
            !self.owner.is_ntt ||
            rows != self.size().0 ||
            blocks.iter().any(|block| {
                !block.owner.is_ntt ||
                    block.size().1 != self.size().1 ||
                    block.owner.params != self.owner.params ||
                    block.owner.level != self.owner.level ||
                    destination.as_ref().is_some_and(|(out, _, _)| out.raw == block.owner.raw)
            })
        {
            return Err("row block shape, format, parameters or output ownership differs".into());
        }
        let (output, output_range) = self.destination(destination, true, self.size())?;
        let mut offset = 0;
        for batch in blocks.chunks(16) {
            let rows = batch.iter().map(|block| block.size().0).sum::<usize>();
            if rows != 0 && self.size().1 != 0 {
                let raw =
                    batch.iter().map(|block| block.owner.raw.cast_const()).collect::<Vec<_>>();
                let ranges = batch.iter().map(|block| block.range).collect::<Vec<_>>();
                let view = GpuMatrixBatchView {
                    left: self.range,
                    right: GpuMatrixRange {
                        row_start: self.range.row_start + offset,
                        row_end: self.range.row_start + offset + rows,
                        ..self.range
                    },
                    output: GpuMatrixRange {
                        row_start: output_range.row_start + offset,
                        row_end: output_range.row_start + offset + rows,
                        ..output_range
                    },
                };
                let status = unsafe {
                    gpu_matrix_add_row_blocks(
                        output.raw,
                        raw.as_ptr(),
                        raw.len(),
                        self.owner.raw,
                        ranges.as_ptr(),
                        &view,
                    )
                };
                if status != 0 {
                    return Err(last_error_string());
                }
            }
            offset += rows;
        }
        Ok(output)
    }

    /// Compute an exact column interval of the tensor of both input views.
    /// Optional row groups fuse the bounded tensor reduction in the same kernel.
    /// Global product columns may cross factor boundaries; neither factor is
    /// replaced by a smaller matrix to approximate the requested range.
    pub fn tensor(
        &self,
        right: &GpuDCRTPolyMatrixColumnView<'_>,
        groups: Option<&[Vec<usize>]>,
        columns: Range<usize>,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        if destination.as_ref().is_some_and(|(output, _, _)| output.raw == right.owner.raw) {
            return Err("tensor output aliases its right input".into());
        }
        let product_rows =
            self.size().0.checked_mul(right.size().0).ok_or("tensor rows overflow")?;
        let product_columns =
            self.size().1.checked_mul(right.size().1).ok_or("tensor columns overflow")?;
        if !self.owner.is_ntt ||
            !right.owner.is_ntt ||
            self.owner.params != right.owner.params ||
            self.owner.level != right.owner.level ||
            columns.start > columns.end ||
            columns.end > product_columns
        {
            return Err(
                "tensor ranges require compatible evaluation inputs and valid product columns"
                    .into(),
            );
        }
        let mut flat = [0usize; 32];
        let mut offsets = [0usize; 17];
        let rows = if let Some(groups) = groups {
            let terms = groups
                .iter()
                .try_fold(0usize, |count, group| count.checked_add(group.len()))
                .ok_or("tensor row terms overflow")?;
            if groups
                .iter()
                .any(|group| group.is_empty() || group.iter().any(|&row| row >= product_rows))
            {
                return Err("tensor row groups contain an empty group or invalid product row".into());
            }
            if groups.len() > 16 || terms > 32 {
                return self.reduce_row_groups(
                    groups,
                    columns.end - columns.start,
                    true,
                    destination,
                    |groups, output| {
                        self.tensor(right, Some(groups), columns.clone(), Some(output))
                    },
                );
            }
            for (index, group) in groups.iter().enumerate() {
                offsets[index + 1] = offsets[index] + group.len();
                flat[offsets[index]..offsets[index + 1]].copy_from_slice(group);
            }
            groups.len()
        } else {
            product_rows
        };
        let (output, range) =
            self.destination(destination, true, (rows, columns.end - columns.start))?;
        if output.raw == right.owner.raw {
            return Err("tensor output aliases its right input".into());
        }
        if rows == 0 || columns.is_empty() {
            return Ok(output);
        }
        let view = GpuMatrixBatchView { left: self.range, right: right.range, output: range };
        let status = unsafe {
            if let Some(groups) = groups {
                gpu_matrix_tensor_sum_rows(
                    output.raw,
                    self.owner.raw,
                    right.owner.raw,
                    flat.as_ptr(),
                    offsets.as_ptr(),
                    groups.len(),
                    offsets[groups.len()],
                    &view,
                    columns.start,
                )
            } else {
                gpu_matrix_tensor(output.raw, self.owner.raw, right.owner.raw, &view, columns.start)
            }
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Convert a rectangle directly into the destination basis. Retained
    /// rectangles use setup-owned CRT tables and bounded launch arguments.
    pub fn convert_modulus(
        &self,
        parameters: &GpuDCRTPolyParams,
        conversion: GpuMatrixModulusConversion,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        conversion.validate(self.params(), self.owner.level, parameters)?;
        let (rows, columns) = self.size();
        let with_views = destination.is_some() ||
            self.range.row_start != 0 ||
            self.range.column_start != 0 ||
            self.range.row_end != self.owner.nrow ||
            self.range.column_end != self.owner.ncol;
        let is_reduce = conversion == GpuMatrixModulusConversion::Reduce;
        let (output, output_rows, output_columns) = destination.unwrap_or_else(|| {
            (
                GpuDCRTPolyMatrix::new_empty_with_state(
                    parameters,
                    rows,
                    columns,
                    parameters.crt_depth() - 1,
                    !is_reduce || self.owner.is_ntt,
                    None,
                ),
                0..rows,
                0..columns,
            )
        });
        if output_rows.start > output_rows.end ||
            output_rows.end > output.nrow ||
            output_columns.start > output_columns.end ||
            output_columns.end > output.ncol ||
            output_rows.end - output_rows.start != rows ||
            output_columns.end - output_columns.start != columns ||
            output.params.context_identity() != parameters.context_identity() ||
            output.level + 1 != parameters.crt_depth() ||
            (is_reduce && output.is_ntt != self.owner.is_ntt) ||
            output.raw == self.owner.raw
        {
            return Err("CRT destination range, parameters, format or owner do not match".into());
        }
        if rows == 0 || columns == 0 {
            return Ok(output);
        }
        let source_primes = &self.params().moduli()[..=self.owner.level];
        let target_primes = parameters.moduli();
        let (mode, plaintext_modulus) = match conversion {
            GpuMatrixModulusConversion::Reduce => (0, 0),
            GpuMatrixModulusConversion::Round => (1, 0),
            GpuMatrixModulusConversion::CenteredExtend => (2, 0),
            GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus } => (3, plaintext_modulus),
        };
        let division_inverses = target_primes
            .par_iter()
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
            .par_iter()
            .map(|prime| {
                if mode != 3 {
                    Ok(1)
                } else {
                    crate::utils::mod_inverse(plaintext_modulus % prime, *prime)
                        .ok_or("plaintext modulus is not invertible")
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let coefficients = (!is_reduce && self.owner.is_ntt)
            .then(|| self.copy(None).map(|input| input.into_coeff_domain()))
            .transpose()?;
        let input = if let Some(coefficients) = &coefficients {
            coefficients.column_view(0..columns)?
        } else {
            *self
        };
        let view = GpuMatrixBatchView {
            left: input.range,
            right: input.range,
            output: GpuMatrixRange {
                row_start: output_rows.start,
                row_end: output_rows.end,
                column_start: output_columns.start,
                column_end: output_columns.end,
            },
        };
        let status = unsafe {
            gpu_matrix_convert_modulus(
                output.raw,
                input.owner.raw,
                mode,
                division_inverses.as_ptr(),
                division_inverses.len(),
                plaintext_modulus,
                input_scales.as_ptr(),
                if with_views { &view } else { std::ptr::null() },
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Decompose borrowed row blocks into a retained compact rectangle.
    /// Coefficient inputs without approximate correction are read directly.
    /// Other inputs use only range-sized GPU copies. In admitted execution,
    /// reserve the copies in source order, then any correction workspaces in
    /// source order, followed by
    /// the compact payload only when no retained destination is supplied.
    pub fn gadget_decompose_row_blocks(
        blocks: &[Self],
        small: bool,
        digit_count: Option<usize>,
        destination: Option<(GpuSmallMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuSmallMatrix, String> {
        let first = blocks.first().ok_or("compact decomposition needs row blocks")?;
        let columns = first.size().1;
        if blocks.len() > 32 ||
            columns == 0 ||
            blocks.iter().any(|block| {
                block.size().0 == 0 ||
                    block.size().1 != columns ||
                    block.params() != first.params() ||
                    block.params().context_identity() != first.params().context_identity() ||
                    block.owner.level + 1 != block.params().crt_depth()
            })
        {
            return Err(
                "compact decomposition requires nonempty compatible full-level blocks".into()
            );
        }
        let layout = first
            .params()
            .compact_decomposition_layout(small, digit_count)
            .map_err(|error| error.to_string())?;
        let rows = blocks
            .iter()
            .try_fold(0usize, |rows, block| rows.checked_add(block.size().0))
            .and_then(|rows| rows.checked_mul(layout.rows_per_input_row))
            .ok_or("compact decomposition row count overflow")?;
        if let Some((output, destination_rows, destination_columns)) = &destination {
            if &output.params != first.params() ||
                output.params.context_identity() != first.params().context_identity() ||
                output.max_coefficient_bound != layout.max_coefficient_bound ||
                destination_rows.start >= destination_rows.end ||
                destination_rows.end > output.rows ||
                destination_rows.end - destination_rows.start != rows ||
                destination_columns.start >= destination_columns.end ||
                destination_columns.end > output.columns ||
                destination_columns.end - destination_columns.start != columns
            {
                return Err(
                    "compact decomposition destination shape, context or bound differs".into()
                );
            }
        }
        // Native claims are ordered and thread-local. Allocate copies on the
        // dispatch owner, then batch independent transforms without allocations.
        let mut copies = blocks
            .iter()
            .map(|block| {
                (block.owner.is_ntt || (!small && layout.dropped_moduli != 0))
                    .then(|| block.copy(None))
                    .transpose()
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for (index, source) in copies.iter().enumerate() {
            if let Some(source) = source.as_ref().filter(|source| source.is_ntt) {
                groups.entry(source.nrow).or_default().push(index);
            }
        }
        groups.par_iter().try_for_each(|(_, indices)| {
            let raw = indices
                .iter()
                .map(|&index| copies[index].as_ref().unwrap().raw)
                .collect::<Vec<_>>();
            let status = unsafe { gpu_matrix_intt_batch(raw.as_ptr(), ptr::null(), raw.len()) };
            if status == 0 { Ok(()) } else { Err(last_error_string()) }
        })?;
        for source in copies.iter_mut().flatten() {
            source.is_ntt = false;
            if !small && layout.dropped_moduli != 0 {
                let status = unsafe {
                    crate::poly::dcrt::gpu::gpu_matrix_correct_gadget_residues(
                        source.raw,
                        layout.dropped_moduli,
                    )
                };
                if status != 0 {
                    return Err(last_error_string());
                }
            }
        }
        let (output, destination_rows, destination_columns) = match destination {
            Some(destination) => destination,
            None => (
                GpuSmallMatrix::new_empty(
                    first.params(),
                    rows,
                    columns,
                    layout.max_coefficient_bound.clone(),
                )
                .map_err(|error| error.to_string())?,
                0..rows,
                0..columns,
            ),
        };
        let mut sources = Vec::with_capacity(blocks.len());
        let mut ranges = Vec::with_capacity(blocks.len());
        for (block, copy) in blocks.iter().zip(&copies) {
            if let Some(copy) = copy {
                sources.push(copy.raw.cast_const());
                ranges.push(GpuMatrixRange {
                    row_start: 0,
                    row_end: copy.nrow,
                    column_start: 0,
                    column_end: copy.ncol,
                });
            } else {
                sources.push(block.owner.raw.cast_const());
                ranges.push(block.range);
            }
        }
        let output_range = GpuMatrixRange {
            row_start: destination_rows.start,
            row_end: destination_rows.end,
            column_start: destination_columns.start,
            column_end: destination_columns.end,
        };
        let words = GpuSmallMatrix::bound_words(&layout.max_coefficient_bound);
        let status = unsafe {
            gpu_small_matrix_decompose_base(
                sources.as_ptr(),
                sources.len(),
                first.params().base_bits(),
                i32::from(small),
                words.as_ptr(),
                words.len(),
                output.raw,
                layout.dropped_moduli,
                ranges.as_ptr(),
                &output_range,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Multiply compact columns into retained ordinary rectangles. The RHS is
    /// expanded once for all row blocks, using only the current compact view.
    pub fn multiply_small_rhs_row_blocks(
        blocks: &[Self],
        rhs: &GpuSmallMatrix,
        destinations: Option<Vec<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>>,
    ) -> Result<Vec<GpuDCRTPolyMatrix>, SmallMatrixError> {
        let first = blocks.first().ok_or(SmallMatrixError::ShapeMismatch)?;
        if blocks.len() > 32 ||
            blocks.iter().any(|block| {
                block.params().context_identity() != first.params().context_identity() ||
                    block.owner.level != first.owner.level ||
                    block.size().1 != rhs.rows
            })
        {
            return Err(SmallMatrixError::ShapeMismatch);
        }
        if blocks
            .iter()
            .any(|block| !block.owner.is_ntt || block.owner.level + 1 != block.params().crt_depth())
        {
            return Err(SmallMatrixError::InvalidConfig);
        }
        if first.params().gpu_ids() != rhs.params.gpu_ids() {
            return Err(SmallMatrixError::DeviceMismatch);
        }
        if first.params() != &rhs.params {
            return Err(SmallMatrixError::ParameterMismatch);
        }
        if first.params().context_identity() != rhs.params.context_identity() {
            return Err(SmallMatrixError::ContextMismatch);
        }
        let with_views = destinations.is_some() ||
            blocks.iter().any(|block| {
                block.range.row_start != 0 ||
                    block.range.column_start != 0 ||
                    block.range.row_end != block.owner.nrow ||
                    block.range.column_end != block.owner.ncol
            });
        if destinations.as_ref().is_some_and(|values| values.len() != blocks.len()) {
            return Err(SmallMatrixError::ShapeMismatch);
        }
        // Query the full retained owners before any output allocation. Native
        // dispatch consumes output claims in order on its owning host thread.
        let (lhs_bytes, output_bytes) = blocks
            .par_iter()
            .enumerate()
            .map(|(index, block)| {
                let source = block.owner;
                let lhs = source
                    .params
                    .matrix_allocation_bytes(source.level, source.nrow, source.ncol, source.is_ntt)
                    .map_err(|_| SmallMatrixError::DimensionOverflow)?
                    .total_bytes;
                let output = if let Some(values) = &destinations {
                    let out = &values[index].0;
                    out.params.matrix_allocation_bytes(out.level, out.nrow, out.ncol, out.is_ntt)
                } else {
                    source.params.matrix_allocation_bytes(
                        source.level,
                        block.size().0,
                        rhs.columns,
                        true,
                    )
                }
                .map_err(|_| SmallMatrixError::DimensionOverflow)?
                .total_bytes;
                Ok::<_, SmallMatrixError>((lhs, output))
            })
            .try_reduce(
                || (0usize, 0usize),
                |(lhs, output), (left, right)| {
                    Ok((
                        lhs.checked_add(left).ok_or(SmallMatrixError::DimensionOverflow)?,
                        output.checked_add(right).ok_or(SmallMatrixError::DimensionOverflow)?,
                    ))
                },
            )?;
        let report = rhs.allocation_report_from_planner_totals(lhs_bytes, output_bytes)?;
        let budget = first.params().vram_budget_bytes();
        validate_small_rhs_budget(&report, budget)?;
        let output_layouts = if let Some(destinations) = destinations {
            if destinations.len() != blocks.len() {
                return Err(SmallMatrixError::ShapeMismatch);
            }
            destinations
        } else {
            blocks
                .iter()
                .map(|block| {
                    (
                        GpuDCRTPolyMatrix::new_empty_with_state(
                            block.params(),
                            block.size().0,
                            rhs.columns,
                            block.owner.level,
                            true,
                            None,
                        ),
                        0..block.size().0,
                        0..rhs.columns,
                    )
                })
                .collect()
        };
        let mut views = Vec::with_capacity(blocks.len());
        for (block, (output, rows, columns)) in blocks.iter().zip(&output_layouts) {
            if rows.start > rows.end ||
                rows.end > output.nrow ||
                rows.end - rows.start != block.size().0 ||
                columns.start > columns.end ||
                columns.end > output.ncol ||
                columns.end - columns.start != rhs.columns ||
                output.params.context_identity() != first.params().context_identity() ||
                output.level != first.owner.level ||
                !output.is_ntt ||
                blocks.iter().any(|source| source.owner.raw == output.raw)
            {
                return Err(SmallMatrixError::ShapeMismatch);
            }
            views.push(GpuMatrixBatchView {
                left: block.range,
                right: block.range,
                output: GpuMatrixRange {
                    row_start: rows.start,
                    row_end: rows.end,
                    column_start: columns.start,
                    column_end: columns.end,
                },
            });
        }
        let raw_outputs =
            output_layouts.iter().map(|(output, _, _)| output.raw).collect::<Vec<_>>();
        let raw_inputs =
            blocks.iter().map(|block| block.owner.raw.cast_const()).collect::<Vec<_>>();
        let mut raw_report = GpuSmallMatrixAllocationReportRaw::default();
        let status = unsafe {
            gpu_matrix_mul_small_rhs(
                raw_outputs.as_ptr(),
                raw_inputs.as_ptr(),
                blocks.len(),
                rhs.raw,
                budget,
                &mut raw_report,
                if with_views { views.as_ptr() } else { std::ptr::null() },
            )
        };
        if status == 2 {
            return Err(SmallMatrixError::ResourceExhausted {
                requested_bytes: raw_report.high_water_bytes,
                budget_bytes: budget,
            });
        }
        check_status(status, "gpu_matrix_mul_small_rhs");
        validate_small_rhs_runtime_report(&report, &raw_report, budget)?;
        Ok(output_layouts.into_iter().map(|(output, _, _)| output).collect())
    }

    /// Reconstruct one plaintext CRT row from independently typed source bases.
    /// Inputs retain their coefficient/evaluation format and their ownership.
    pub fn crt_recompose(
        levels: &[Self],
        plaintext_moduli: &[u64],
        reconstruction_residues: &[u64],
        parameters: &GpuDCRTPolyParams,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let first = levels.first().ok_or("CRT recomposition needs at least one level")?;
        let columns = first.size().1;
        if levels.len() != plaintext_moduli.len() ||
            plaintext_moduli.contains(&0) ||
            levels.len().checked_mul(parameters.crt_depth()) !=
                Some(reconstruction_residues.len()) ||
            parameters.crt_depth() > 64 ||
            levels.iter().any(|level| {
                level.size() != (1, columns) ||
                    level.params().ring_dimension() != parameters.ring_dimension() ||
                    level.owner.level >= level.params().crt_depth() ||
                    level.owner.level >= 64 ||
                    level.params().device_ids() != parameters.device_ids() ||
                    level.params().execution_owner_id() != parameters.execution_owner_id()
            })
        {
            return Err("invalid CRT level count, basis, shape, modulus or execution owner".into());
        }
        let with_views = destination.is_some() ||
            levels.iter().any(|level| {
                level.range.row_start != 0 ||
                    level.range.column_start != 0 ||
                    level.range.row_end != level.owner.nrow ||
                    level.range.column_end != level.owner.ncol
            });
        let (output, rows, columns_range) = destination.unwrap_or_else(|| {
            (
                GpuDCRTPolyMatrix::new_empty_with_state(
                    parameters,
                    1,
                    columns,
                    parameters.crt_depth() - 1,
                    true,
                    None,
                ),
                0..1,
                0..columns,
            )
        });
        if rows.start >= rows.end ||
            rows.end > output.nrow ||
            rows.end - rows.start != 1 ||
            columns_range.start > columns_range.end ||
            columns_range.end > output.ncol ||
            columns_range.end - columns_range.start != columns ||
            output.params.context_identity() != parameters.context_identity() ||
            output.level + 1 != parameters.crt_depth() ||
            levels.iter().any(|level| level.owner.raw == output.raw)
        {
            return Err("CRT destination range, parameters or owner do not match".into());
        }
        if columns == 0 {
            return Ok(output);
        }
        let coefficients = levels
            .par_iter()
            .map(|level| {
                level
                    .owner
                    .is_ntt
                    .then(|| level.copy(None).map(GpuDCRTPolyMatrix::into_coeff_domain))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let views = levels
            .iter()
            .zip(&coefficients)
            .map(
                |(level, copy)| {
                    if let Some(copy) = copy { copy.column_view(0..columns) } else { Ok(*level) }
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        let raw = views.iter().map(|level| level.owner.raw.cast_const()).collect::<Vec<_>>();
        let ranges = views.iter().map(|level| level.range).collect::<Vec<_>>();
        let output_range = GpuMatrixRange {
            row_start: rows.start,
            row_end: rows.end,
            column_start: columns_range.start,
            column_end: columns_range.end,
        };
        let status = unsafe {
            gpu_matrix_crt_recompose(
                output.raw,
                raw.as_ptr(),
                raw.len(),
                plaintext_moduli.as_ptr(),
                reconstruction_residues.as_ptr(),
                parameters.crt_depth(),
                if with_views { ranges.as_ptr() } else { std::ptr::null() },
                if with_views { &output_range } else { std::ptr::null() },
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Convert an RNS rectangle into retained rows. ModUp group rows are
    /// contiguous logical blocks, independent of either owner's physical pitch.
    pub fn rns_conversion(
        &self,
        parameters: &GpuDCRTPolyParams,
        conversion: GpuMatrixRnsConversion,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let groups = conversion.validate(self.params(), self.owner.level, parameters)?;
        let (input_rows, columns) = self.size();
        let rows = input_rows.checked_mul(groups).ok_or("RNS output row count overflow")?;
        let with_views = destination.is_some() ||
            self.range.row_start != 0 ||
            self.range.column_start != 0 ||
            self.range.row_end != self.owner.nrow ||
            self.range.column_end != self.owner.ncol;
        let (output, output_rows, output_columns) = destination.unwrap_or_else(|| {
            (
                GpuDCRTPolyMatrix::new_empty_with_state(
                    parameters,
                    rows,
                    columns,
                    parameters.crt_depth() - 1,
                    true,
                    None,
                ),
                0..rows,
                0..columns,
            )
        });
        if output_rows.start > output_rows.end ||
            output_rows.end > output.nrow ||
            output_columns.start > output_columns.end ||
            output_columns.end > output.ncol ||
            output_rows.end - output_rows.start != rows ||
            output_columns.end - output_columns.start != columns ||
            output.params.context_identity() != parameters.context_identity() ||
            output.level + 1 != parameters.crt_depth() ||
            output.raw == self.owner.raw
        {
            return Err("RNS destination range, parameters or owner do not match".into());
        }
        if rows == 0 || columns == 0 {
            return Ok(output);
        }
        let plan = conversion.plan(self.params(), parameters)?;
        let coefficients = self
            .owner
            .is_ntt
            .then(|| self.copy(None).map(|x| x.into_coeff_domain()))
            .transpose()?;
        let input = if let Some(coefficients) = &coefficients {
            coefficients.column_view(0..columns)?
        } else {
            *self
        };
        let (digit_size, plaintext_modulus) = match conversion {
            GpuMatrixRnsConversion::Up { digit_size, .. } => (digit_size, 0),
            GpuMatrixRnsConversion::Down { plaintext_modulus } => {
                (self.params().crt_depth(), plaintext_modulus)
            }
        };
        let view = GpuMatrixBatchView {
            left: input.range,
            right: input.range,
            output: GpuMatrixRange {
                row_start: output_rows.start,
                row_end: output_rows.end,
                column_start: output_columns.start,
                column_end: output_columns.end,
            },
        };
        let status = unsafe {
            gpu_matrix_rns_conversion(
                output.raw,
                input.owner.raw,
                digit_size,
                plaintext_modulus,
                plan.scales.as_ptr(),
                plan.inverses.as_ptr(),
                plan.weights.as_ref().map_or(std::ptr::null(), |weights| weights.as_ptr()),
                if with_views { &view } else { std::ptr::null() },
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Lift centered single-limb coefficients into another registered CRT basis.
    /// Only the selected output rectangle is transformed back to evaluation form.
    pub fn centered_rebase(
        &self,
        parameters: &GpuDCRTPolyParams,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        if self.params().ring_dimension() != parameters.ring_dimension() ||
            self.params().crt_depth() != 1 ||
            self.owner.level != 0 ||
            self.params().execution_owner_id() != parameters.execution_owner_id() ||
            self.params().device_ids() != parameters.device_ids()
        {
            return Err("centered rebase requires matching dimensions, placement, shared execution, and one source CRT limb".into());
        }
        let (rows, columns) = self.size();
        let (output, output_rows, output_columns) = destination.unwrap_or_else(|| {
            (
                GpuDCRTPolyMatrix::new_empty_with_state(
                    parameters,
                    rows,
                    columns,
                    parameters.crt_depth() - 1,
                    true,
                    None,
                ),
                0..rows,
                0..columns,
            )
        });
        if output_rows.start > output_rows.end ||
            output_rows.end > output.nrow ||
            output_columns.start > output_columns.end ||
            output_columns.end > output.ncol ||
            output_rows.end - output_rows.start != rows ||
            output_columns.end - output_columns.start != columns ||
            output.params.context_identity() != parameters.context_identity() ||
            output.level + 1 != parameters.crt_depth() ||
            output.raw == self.owner.raw
        {
            return Err(
                "centered rebase destination range, parameters, or owner do not match".into()
            );
        }
        if rows == 0 || columns == 0 {
            return Ok(output);
        }
        let coefficients = self
            .owner
            .is_ntt
            .then(|| self.copy(None).map(|input| input.into_coeff_domain()))
            .transpose()?;
        let input = if let Some(coefficients) = &coefficients {
            coefficients.column_view(0..columns)?
        } else {
            *self
        };
        let view = GpuMatrixBatchView {
            left: input.range,
            right: input.range,
            output: GpuMatrixRange {
                row_start: output_rows.start,
                row_end: output_rows.end,
                column_start: output_columns.start,
                column_end: output_columns.end,
            },
        };
        let status = unsafe { gpu_matrix_centered_rebase(output.raw, input.owner.raw, &view) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Transpose this rectangle directly into the retained destination. Source
    /// and destination strides and format are preserved without a staging copy.
    pub fn transpose(
        &self,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let (rows, columns) = self.size();
        let (output, range) = self.destination(destination, self.owner.is_ntt, (columns, rows))?;
        let views = GpuMatrixBatchView { left: self.range, right: self.range, output: range };
        let status = unsafe { gpu_matrix_transpose(output.raw, self.owner.raw, &views) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Reduce rows into an existing rectangle. Large groups use bounded native
    /// leaves and a parallel binary tree with explicitly accounted intermediates.
    pub fn sum_rows(
        &self,
        rows: &[Vec<usize>],
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let terms = rows
            .iter()
            .try_fold(0usize, |total, group| total.checked_add(group.len()))
            .ok_or("row sum term count overflow")?;
        if rows
            .iter()
            .any(|group| group.is_empty() || group.iter().any(|&row| row >= self.size().0))
        {
            return Err("row groups contain an empty group or invalid input row".into());
        }
        if rows.len() > 16 || terms > 32 {
            return self.reduce_row_groups(
                rows,
                self.size().1,
                self.owner.is_ntt,
                destination,
                |groups, output| self.sum_rows(groups, Some(output)),
            );
        }
        let (output, range) =
            self.destination(destination, self.owner.is_ntt, (rows.len(), self.size().1))?;
        if rows.is_empty() || self.size().1 == 0 {
            return Ok(output);
        }
        let mut flat = [0usize; 32];
        let mut offsets = [0usize; 17];
        for (index, group) in rows.iter().enumerate() {
            offsets[index + 1] = offsets[index] + group.len();
            flat[offsets[index]..offsets[index + 1]].copy_from_slice(group);
        }
        let views = GpuMatrixBatchView { left: self.range, right: self.range, output: range };
        let status = unsafe {
            gpu_matrix_sum_rows(
                output.raw,
                self.owner.raw,
                flat.as_ptr(),
                offsets.as_ptr(),
                rows.len(),
                terms,
                &views,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    fn reduce_row_groups(
        &self,
        groups: &[Vec<usize>],
        columns: usize,
        evaluation: bool,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
        leaf: impl Fn(
            &[Vec<usize>],
            (GpuDCRTPolyMatrix, Range<usize>, Range<usize>),
        ) -> Result<GpuDCRTPolyMatrix, String>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        GpuDCRTPolyMatrix::row_reduction_intermediates(groups)?;
        let (mut output, range) =
            self.destination(destination, evaluation, (groups.len(), columns))?;
        if columns == 0 {
            return Ok(output);
        }
        // One destination owns all row batches. Submit its writes in order;
        // internal tree levels own disjoint matrices and run through Rayon.
        let mut start = 0;
        while start < groups.len() {
            if groups[start].len() <= 32 {
                let mut end = start;
                let mut terms = 0;
                while end < groups.len() && end - start < 16 && terms + groups[end].len() <= 32 {
                    terms += groups[end].len();
                    end += 1;
                }
                output = leaf(
                    &groups[start..end],
                    (
                        output,
                        range.row_start + start..range.row_start + end,
                        range.column_start..range.column_end,
                    ),
                )?;
                start = end;
                continue;
            }
            let leaves = groups[start].len().div_ceil(32);
            // Allocate on the dispatch thread before parallel arithmetic. Native
            // claims follow this exact sequence and need no thread-local permit
            // on the allocation-free parent kernels.
            let mut nodes = (0..2 * leaves - 1)
                .map(|_| {
                    Some(GpuDCRTPolyMatrix::new_empty_with_state(
                        &self.owner.params,
                        1,
                        columns,
                        self.owner.level,
                        evaluation,
                        None,
                    ))
                })
                .collect::<Vec<_>>();
            for (index, rows) in groups[start].chunks(32).enumerate() {
                nodes[index] =
                    Some(leaf(&[rows.to_vec()], (nodes[index].take().unwrap(), 0..1, 0..columns))?);
            }
            let mut level = (0..leaves).collect::<Vec<_>>();
            let mut next = leaves;
            while level.len() > 1 {
                let pairs = level.len() / 2;
                let (sources, destinations) = nodes.split_at_mut(next);
                destinations[..pairs].par_iter_mut().zip(level.par_chunks_exact(2)).try_for_each(
                    |(output, inputs)| -> Result<(), String> {
                        let left = sources[inputs[0]].as_ref().unwrap().column_view(0..columns)?;
                        let right = sources[inputs[1]].as_ref().unwrap().column_view(0..columns)?;
                        *output = Some(
                            left.add(&right, Some((output.take().unwrap(), 0..1, 0..columns)))?,
                        );
                        Ok(())
                    },
                )?;
                let mut parents = (next..next + pairs).collect::<Vec<_>>();
                if level.len() % 2 != 0 {
                    parents.push(*level.last().unwrap());
                }
                next += pairs;
                level = parents;
            }
            debug_assert_eq!(next, nodes.len());
            output = nodes[level[0]].as_ref().unwrap().column_view(0..columns)?.copy(Some((
                output,
                range.row_start + start..range.row_start + start + 1,
                range.column_start..range.column_end,
            )))?;
            // Native source-release joins retain every asynchronous tree reader.
            drop(nodes);
            start += 1;
        }
        Ok(output)
    }

    fn binary(
        &self,
        right: &Self,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
        operation: i32,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        if self.size() != right.size() ||
            self.params() != right.params() ||
            self.owner.level != right.owner.level ||
            self.owner.is_ntt != right.owner.is_ntt
        {
            return Err("GPU binary views require matching shape, parameters, and format".into());
        }
        let (output, range) = self.destination(destination, self.owner.is_ntt, self.size())?;
        if output.raw == right.owner.raw {
            return Err("GPU destination aliases a binary input owner".into());
        }
        let (rows, columns) = self.size();
        if rows != 0 && columns != 0 {
            let views = GpuMatrixBatchView { left: self.range, right: right.range, output: range };
            let status = unsafe {
                gpu_matrix_binary_batch(
                    [output.raw].as_ptr(),
                    [self.owner.raw.cast_const()].as_ptr(),
                    [right.owner.raw.cast_const()].as_ptr(),
                    &views,
                    1,
                    operation,
                )
            };
            if status != 0 {
                return Err(last_error_string());
            }
        }
        Ok(output)
    }

    pub fn add(
        &self,
        right: &Self,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        self.binary(right, destination, 0)
    }

    pub fn sub(
        &self,
        right: &Self,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        self.binary(right, destination, 1)
    }

    /// Multiply resident evaluation ranges directly into an optional full
    /// output owner. Scalar matrix operands broadcast without extracting a GPU
    /// polynomial. Format conversion belongs to explicit input preparation.
    pub fn multiply(
        &self,
        right: &Self,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let (left, right) = if self.size() == (1, 1) && self.owner.size() == (1, 1) {
            (right, self)
        } else {
            (self, right)
        };
        let scalar = right.size() == (1, 1) && right.owner.size() == (1, 1);
        if left.params() != right.params() ||
            left.params().ctx_raw() != right.params().ctx_raw() ||
            left.owner.level != right.owner.level ||
            !left.owner.is_ntt ||
            !right.owner.is_ntt ||
            (!scalar && left.size().1 != right.size().0)
        {
            return Err("GPU multiplication views need compatible evaluation inputs; scalar owners must be complete".into());
        }
        if !scalar && left.size().1 == 0 {
            return Err("GPU multiplication views require a nonempty contraction".into());
        }
        let shape = if scalar { left.size() } else { (left.size().0, right.size().1) };
        let (output, range) = left.destination(destination, true, shape)?;
        if output.raw == right.owner.raw || output.params.ctx_raw() != left.params().ctx_raw() {
            return Err(
                "GPU multiplication destination aliases an input or has a different context".into(),
            );
        }
        if shape.0 == 0 || shape.1 == 0 {
            return Ok(output);
        }
        let views = GpuMatrixBatchView { left: left.range, right: right.range, output: range };
        let status = unsafe {
            if scalar {
                gpu_matrix_mul_scalar_batch(
                    [output.raw].as_ptr(),
                    [left.owner.raw.cast_const()].as_ptr(),
                    [right.owner.raw.cast_const()].as_ptr(),
                    &views,
                    1,
                    std::ptr::null(),
                )
            } else {
                gpu_matrix_mul_batch(
                    [output.raw].as_ptr(),
                    [left.owner.raw.cast_const()].as_ptr(),
                    [right.owner.raw.cast_const()].as_ptr(),
                    &views,
                    1,
                )
            }
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// The destination is evaluation format. Coefficient inputs use an explicit
    /// range-sized conversion; unrequested source columns/rows are never copied.
    pub fn multiply_poly(
        &self,
        scalar: &GpuDCRTPoly,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        if self.params() != scalar.params_ref() {
            return Err("GPU column-view scalar parameters do not match".into());
        }
        let (output, range) = self.destination(destination, true, self.size())?;
        let (rows, columns) = self.size();
        if rows == 0 || columns == 0 {
            return Ok(output);
        }
        let converted = (!self.owner.is_ntt).then(|| {
            let mut matrix = self.owner.slice(
                self.range.row_start,
                self.range.row_end,
                self.range.column_start,
                self.range.column_end,
            );
            matrix.ntt_all_in_place();
            matrix
        });
        let matrix = converted.as_ref().unwrap_or(self.owner);
        let source_range = if converted.is_some() {
            GpuMatrixRange { row_start: 0, row_end: rows, column_start: 0, column_end: columns }
        } else {
            self.range
        };
        let scalar_eval = (!scalar.is_ntt()).then(|| {
            let mut scalar = scalar.clone();
            scalar.ntt_in_place();
            scalar
        });
        let scalar = scalar_eval.as_ref().unwrap_or(scalar);
        let views = GpuMatrixBatchView { left: source_range, right: source_range, output: range };
        let status = unsafe {
            gpu_matrix_mul_scalar_batch(
                [output.raw].as_ptr(),
                [matrix.raw.cast_const()].as_ptr(),
                [scalar.inner().raw.cast_const()].as_ptr(),
                &views,
                1,
                std::ptr::null(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Sum integer-weighted products into one retained rectangle. Products use
    /// evaluation values; native launch metadata is bounded independently of
    /// the number of terms and introduces no intermediate GPU matrices.
    pub fn multiply_accumulate(
        products: &[(BigInt, Self, Self)],
        bias: Option<Self>,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let (_, first, _) = products.first().ok_or("accumulate needs at least one product")?;
        let product_shape = |left: &Self, right: &Self| -> Result<(usize, usize), String> {
            if left.size() == (1, 1) {
                Ok(right.size())
            } else if right.size() == (1, 1) {
                Ok(left.size())
            } else if left.size().1 == right.size().0 {
                Ok((left.size().0, right.size().1))
            } else {
                Err("accumulate inner dimensions differ".into())
            }
        };
        let shape = product_shape(&products[0].1, &products[0].2)?;
        let valid = |source: &Self| {
            source.owner.params == first.owner.params &&
                source.owner.level == first.owner.level &&
                source.owner.is_ntt &&
                !destination.as_ref().is_some_and(|(out, _, _)| out.raw == source.owner.raw)
        };
        for (_, left, right) in products {
            if !valid(left) || !valid(right) || product_shape(left, right)? != shape {
                return Err("accumulate shape, parameters, format or ownership differs".into());
            }
        }
        if bias.is_some_and(|bias| !valid(&bias) || bias.size() != shape) {
            return Err("accumulate bias differs from its products".into());
        }
        let (output, range) = first.destination(destination, true, shape)?;
        let residues = products
            .par_iter()
            .flat_map_iter(|(coefficient, _, _)| {
                first.params().moduli()[..=first.owner.level].iter().map(move |&modulus| {
                    let modulus = BigInt::from(modulus);
                    ((coefficient % &modulus + &modulus) % &modulus).to_u64().unwrap()
                })
            })
            .collect::<Vec<_>>();
        let left =
            products.iter().map(|(_, left, _)| left.owner.raw.cast_const()).collect::<Vec<_>>();
        let right =
            products.iter().map(|(_, _, right)| right.owner.raw.cast_const()).collect::<Vec<_>>();
        let views = products
            .iter()
            .map(|(_, left, right)| GpuMatrixBatchView {
                left: left.range,
                right: right.range,
                output: range,
            })
            .collect::<Vec<_>>();
        let bias_range = bias.map(|bias| bias.range);
        let status = unsafe {
            gpu_matrix_mul_accumulate_batch(
                [output.raw].as_ptr(),
                left.as_ptr(),
                right.as_ptr(),
                ptr::null(),
                [bias.map_or(ptr::null(), |bias| bias.owner.raw.cast_const())].as_ptr(),
                ptr::null(),
                1,
                products.len(),
                views.as_ptr(),
                bias_range.as_ref().map_or(ptr::null(), |range| range),
                residues.as_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(output)
    }

    /// Scale by a host integer without creating a scalar polynomial on the GPU.
    /// Explicit destinations preserve the input format; a standalone result is
    /// returned in evaluation format, matching ordinary polynomial scaling.
    pub fn scale_integer(
        &self,
        scalar: &num_bigint::BigInt,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let transform_output = destination.is_none();
        let (mut output, range) = self.destination(destination, self.owner.is_ntt, self.size())?;
        let (rows, columns) = self.size();
        if rows != 0 && columns != 0 {
            let residues = self.params().moduli()[..=self.owner.level]
                .par_iter()
                .map(|&modulus| {
                    let modulus = num_bigint::BigInt::from(modulus);
                    ((scalar % &modulus + &modulus) % &modulus).to_u64().unwrap()
                })
                .collect::<Vec<_>>();
            let views = GpuMatrixBatchView { left: self.range, right: self.range, output: range };
            let status = unsafe {
                gpu_matrix_mul_scalar_batch(
                    [output.raw].as_ptr(),
                    [self.owner.raw.cast_const()].as_ptr(),
                    std::ptr::null(),
                    &views,
                    1,
                    residues.as_ptr(),
                )
            };
            if status != 0 {
                return Err(last_error_string());
            }
        }
        if transform_output {
            output.ntt_all_in_place();
        }
        Ok(output)
    }

    /// Evaluation destinations use a direct permutation of the evaluation
    /// values. Coefficient destinations preserve the signed coefficient map;
    /// evaluation inputs then require a range-sized inverse-NTT scratch owner.
    /// Without a destination, retain the ordinary evaluation result.
    pub fn ring_automorphism(
        &self,
        index: usize,
        destination: Option<(GpuDCRTPolyMatrix, Range<usize>, Range<usize>)>,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        let n = self.params().ring_dimension() as usize;
        if !n.is_power_of_two() ||
            n > usize::MAX / 2 ||
            index == 0 ||
            index >= 2 * n ||
            index % 2 == 0
        {
            return Err("invalid GPU column-view automorphism index".into());
        }
        let transform_output = destination.is_none();
        let (rows, columns) = self.size();
        // The ordinary empty automorphism returns coefficient format because
        // there are no polynomials to transform. Preserve that metadata contract.
        let evaluation = destination
            .as_ref()
            .map_or(self.owner.is_ntt && rows != 0 && columns != 0, |(value, _, _)| value.is_ntt);
        if evaluation && !self.owner.is_ntt {
            return Err("evaluation automorphism destination requires evaluation input".into());
        }
        let (mut output, range) = self.destination(destination, evaluation, self.size())?;
        if rows == 0 || columns == 0 {
            return Ok(output);
        }
        let coefficients = (self.owner.is_ntt && !evaluation).then(|| {
            let mut matrix = self.owner.slice(
                self.range.row_start,
                self.range.row_end,
                self.range.column_start,
                self.range.column_end,
            );
            matrix.intt_all_in_place();
            matrix
        });
        let source = coefficients.as_ref().unwrap_or(self.owner);
        let source_range = if coefficients.is_some() {
            GpuMatrixRange { row_start: 0, row_end: rows, column_start: 0, column_end: columns }
        } else {
            self.range
        };
        let views = GpuMatrixBatchView { left: source_range, right: source_range, output: range };
        let status = unsafe {
            gpu_matrix_ring_automorphism_batch(
                [output.raw].as_ptr(),
                [source.raw.cast_const()].as_ptr(),
                [index].as_ptr(),
                &views,
                1,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        if transform_output {
            output.ntt_all_in_place();
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::gpu::{GpuMatrixBatchOperation, gpu_default_mempool_usage},
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_compact_decomposition_rectangles_preserve_readers() {
        use crate::matrix::{CpuSmallMatrix, PolyMatrixSmallRhs, SmallPolyMatrix};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let width = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for dropped in [0, 1] {
            for small in [false, true] {
                for mixed_formats in [false, true] {
                    let primes = vec![narrow[0], wide, narrow[1]];
                    let cpu = DCRTPolyParams::new(n, 3, 54, 4, Some(primes.clone()), Some(dropped));
                    let parameters = GpuDCRTPolyParams::new(n, primes, 4, Some(dropped));
                    let layout = parameters.compact_decomposition_layout(small, None).unwrap();
                    let originals = (0..2)
                        .map(|i| {
                            let value = DCRTPolyUniformSampler::new().sample_uniform(
                                &cpu,
                                3 + i,
                                width + 3 + i,
                                if small { DistType::BitDist } else { DistType::FinRingDist },
                            );
                            if small {
                                // Shared unsigned coefficients exercise the bounded
                                // small decomposition contract in every CRT tower.
                                value *
                                    crate::poly::dcrt::poly::DCRTPoly::from_biguint_to_constant(
                                        &cpu,
                                        BigUint::from(
                                            parameters.moduli().iter().min().unwrap() - 1,
                                        ),
                                    )
                            } else {
                                value
                            }
                        })
                        .collect::<Vec<_>>();
                    let expected = DCRTPolyMatrix::gadget_decompose_row_blocks(
                        originals
                            .iter()
                            .enumerate()
                            .map(|(i, value)| value.slice(1, 2 + i, 1, width + 1))
                            .collect(),
                        small,
                        None,
                    )
                    .unwrap();
                    assert_eq!(expected.rows(), 3 * layout.rows_per_input_row);
                    assert_eq!(expected.max_coefficient_bound(), &layout.max_coefficient_bound);
                    let sources = originals
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            let source = GpuDCRTPolyMatrix::from_cpu_matrix(&parameters, value);
                            if mixed_formats && i == 0 {
                                source
                            } else {
                                source.into_coeff_domain()
                            }
                        })
                        .collect::<Vec<_>>();
                    let all_views = sources
                        .iter()
                        .enumerate()
                        .map(|(i, source)| {
                            source.column_view(1..width + 1).unwrap().row_view(1..2 + i).unwrap()
                        })
                        .collect::<Vec<_>>();
                    let fresh = GpuDCRTPolyMatrixColumnView::gadget_decompose_row_blocks(
                        &all_views, small, None, None,
                    )
                    .unwrap();
                    assert_eq!(
                        fresh.to_canonical_coefficients().unwrap(),
                        expected.to_canonical_coefficients().unwrap()
                    );
                    drop(fresh);
                    let output_rows = expected.rows() + 2;
                    let output_columns = width + 4;
                    let initial = DCRTPolyUniformSampler::new().sample_uniform(
                        &cpu,
                        output_rows,
                        output_columns,
                        DistType::TernaryDist,
                    );
                    let initial =
                        CpuSmallMatrix::new(initial, layout.max_coefficient_bound.clone()).unwrap();
                    let mut final_value = initial.value().clone();
                    final_value.copy_block_from(
                        expected.value(),
                        1,
                        2,
                        0,
                        0,
                        expected.rows(),
                        width,
                    );
                    let final_value =
                        CpuSmallMatrix::new(final_value, layout.max_coefficient_bound.clone())
                            .unwrap();
                    let mut output = GpuSmallMatrix::from_canonical_coefficients(
                        &parameters,
                        output_rows,
                        output_columns,
                        layout.max_coefficient_bound.clone(),
                        &initial.to_canonical_coefficients().unwrap(),
                    )
                    .unwrap();
                    let old_reader = output.slice_columns(0, output_columns);
                    let copy_indices = sources
                        .iter()
                        .enumerate()
                        .filter_map(|(i, source)| {
                            (source.is_ntt || (!small && dropped != 0)).then_some(i)
                        })
                        .collect::<Vec<_>>();
                    // These mixed-width bases need only the copy's existing
                    // auxiliary bytes for the GPU correction metadata.
                    for &i in &copy_indices {
                        for columns in [1, 2] {
                            assert_eq!(
                                parameters
                                    .matrix_gadget_correction_workspace_bytes(
                                        2,
                                        1 + i,
                                        columns,
                                        if small { 0 } else { dropped },
                                    )
                                    .unwrap()
                                    .additional_bytes,
                                0
                            );
                        }
                    }
                    let scratch = GpuPreparedStorage::new(
                        (0..copy_indices.len().max(1))
                            .map(|_| GpuDCRTPolyMatrix::zero(&parameters, 2, 2))
                            .collect(),
                        None,
                    )
                    .unwrap();
                    let transfer = parameters.rns_transfer_workspace(2, width + 4, 4).unwrap();
                    let mut readback_layouts = vec![
                        GpuPreparedWorkspaceLayout {
                            kind: GpuPreparedSlotKind::CompactPayload,
                            bytes: output.resident_payload_bytes,
                            alignment: 256,
                        },
                        transfer,
                    ];
                    readback_layouts.extend(std::iter::repeat_n(
                        GpuPreparedWorkspaceLayout {
                            kind: GpuPreparedSlotKind::CompletionEvent,
                            bytes: 0,
                            alignment: 1,
                        },
                        parameters.crt_depth() + 1,
                    ));
                    let readback = GpuPreparedStorage::new(
                        (0..2)
                            .map(|_| GpuDCRTPolyMatrix::zero(&parameters, width + 4, 4))
                            .collect(),
                        Some(&readback_layouts),
                    )
                    .unwrap();
                    GpuPreparedStorage::finish_setup(&[&scratch, &readback]).unwrap();
                    let claims = |columns| {
                        copy_indices
                            .iter()
                            .enumerate()
                            .map(|(slot, &i)| {
                                scratch.slot_identity(slot).unwrap().matrix_request(
                                    1 + i,
                                    columns,
                                    sources[i].is_ntt,
                                )
                            })
                            .collect::<Vec<_>>()
                    };
                    let mut reservation = scratch.reserve(&claims(2)).unwrap();
                    for start in (0..width).step_by(2) {
                        let end = (start + 2).min(width);
                        if start != 0 && !copy_indices.is_empty() {
                            reservation.rearm(&claims(end - start)).unwrap();
                        }
                        let dispatch = reservation.enter(Vec::new()).unwrap();
                        let views = sources
                            .iter()
                            .enumerate()
                            .map(|(i, source)| {
                                source
                                    .column_view(start + 1..end + 1)
                                    .unwrap()
                                    .row_view(1..2 + i)
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        output = GpuDCRTPolyMatrixColumnView::gadget_decompose_row_blocks(
                            &views,
                            small,
                            None,
                            Some((output, 1..1 + expected.rows(), start + 2..end + 2)),
                        )
                        .unwrap();
                        reservation = dispatch.finish().unwrap().pop().unwrap();
                    }
                    drop(reservation);
                    let source_readers = sources
                        .iter()
                        .enumerate()
                        .map(|(i, source)| {
                            assert_eq!(source.is_ntt, mixed_formats && i == 0);
                            let claim = readback.slot_identity(i).unwrap().matrix_request(
                                source.ncol,
                                source.nrow,
                                source.is_ntt,
                            );
                            let dispatch =
                                readback.reserve(&[claim]).unwrap().enter(Vec::new()).unwrap();
                            let mut reader = source
                                .column_view(0..source.ncol)
                                .unwrap()
                                .transpose(None)
                                .unwrap();
                            // The CPU export consumes evaluation format. Transform
                            // this owned snapshot in place, preserving its source.
                            reader.ntt_all_in_place();
                            drop(dispatch.finish().unwrap());
                            reader
                        })
                        .collect::<Vec<_>>();
                    let claim = readback
                        .slot_identity(2)
                        .unwrap()
                        .workspace_request(output.resident_payload_bytes, 256);
                    let dispatch = readback.reserve(&[claim]).unwrap().enter(Vec::new()).unwrap();
                    let reader = output.slice_columns(0, output_columns);
                    drop(dispatch.finish().unwrap());
                    drop((output, sources, scratch));
                    for (value, expected) in [(&old_reader, &initial), (&reader, &final_value)] {
                        let claim = readback.slot_identity(4).unwrap().workspace_request(0, 1);
                        let dispatch =
                            readback.reserve(&[claim]).unwrap().enter(Vec::new()).unwrap();
                        assert_eq!(
                            value.to_canonical_coefficients().unwrap(),
                            expected.to_canonical_coefficients().unwrap()
                        );
                        drop(dispatch.finish().unwrap());
                    }
                    for (value, expected) in source_readers.iter().zip(&originals) {
                        let transfer = parameters
                            .rns_transfer_workspace(value.level(), value.nrow, value.ncol)
                            .unwrap();
                        let events = value.rns_store_completion_events().unwrap();
                        let claims = (3..4 + events)
                            .map(|slot| {
                                let id = readback.slot_identity(slot).unwrap();
                                id.workspace_request(
                                    if slot == 3 { transfer.bytes } else { 0 },
                                    id.alignment(),
                                )
                            })
                            .collect::<Vec<_>>();
                        let dispatch =
                            readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                        assert_eq!(value.to_cpu_matrix().transpose(), *expected);
                        drop(dispatch.finish().unwrap());
                    }
                    assert!(parameters.compact_decomposition_layout(small, Some(0)).is_err());
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_small_rhs_rectangles_reuse_expansion_and_preserve_readers() {
        use crate::matrix::{PolyMatrixSmallRhs, SmallPolyMatrix};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let width = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for primes in [vec![narrow[0], wide, narrow[1]], narrow, vec![wide]] {
            let bits = if primes.contains(&wide) { 54 } else { 17 };
            let cpu = DCRTPolyParams::new(n, primes.len(), bits, 4, Some(primes.clone()), None);
            let parameters = GpuDCRTPolyParams::new(n, primes, 4, None);
            let original = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                1,
                width + 2,
                DistType::FinRingDist,
            );
            let compact = original.gadget_decompose(false, None).unwrap();
            let rhs = GpuSmallMatrix::from_canonical_coefficients(
                &parameters,
                compact.rows(),
                compact.columns(),
                compact.max_coefficient_bound().clone(),
                &compact.to_canonical_coefficients().unwrap(),
            )
            .unwrap();
            let inner = compact.rows();
            let originals = (0..2)
                .map(|i| {
                    DCRTPolyUniformSampler::new().sample_uniform(
                        &cpu,
                        4 + i,
                        inner + 2 + i,
                        DistType::FinRingDist,
                    )
                })
                .collect::<Vec<_>>();
            let expected = originals
                .iter()
                .enumerate()
                .map(|(i, left)| {
                    left.slice(1, 3 + i, 1, inner + 1)
                        .multiply_small_rhs(&compact)
                        .unwrap()
                        .slice_columns(1, width + 1)
                })
                .collect::<Vec<_>>();
            let sources = originals
                .iter()
                .map(|cpu| GpuDCRTPolyMatrix::from_cpu_matrix(&parameters, cpu))
                .collect::<Vec<_>>();
            let initial = (0..2)
                .map(|i| {
                    DCRTPolyUniformSampler::new().sample_uniform(
                        &cpu,
                        4 + i,
                        width + 4,
                        DistType::FinRingDist,
                    )
                })
                .collect::<Vec<_>>();
            let mut outputs = initial
                .iter()
                .map(|cpu| GpuDCRTPolyMatrix::from_cpu_matrix(&parameters, cpu))
                .collect::<Vec<_>>();
            let old_readers = outputs.iter().map(PolyMatrix::transpose).collect::<Vec<_>>();
            let layouts =
                parameters.small_rhs_workspaces(parameters.crt_depth() - 1, inner, 2).unwrap();
            assert_eq!(layouts.len(), if parameters.crt_depth() == 3 { 2 } else { 1 });
            let scratch = GpuPreparedStorage::new(
                vec![GpuDCRTPolyMatrix::zero(&parameters, 1, 1)],
                Some(&layouts),
            )
            .unwrap();
            let mut readback_layouts = vec![
                parameters
                    .rns_transfer_workspace(parameters.crt_depth() - 1, width + 4, 5)
                    .unwrap(),
            ];
            readback_layouts.extend(std::iter::repeat_n(
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                },
                parameters.crt_depth(),
            ));
            let readback = GpuPreparedStorage::new(
                (0..2).map(|_| GpuDCRTPolyMatrix::zero(&parameters, width + 4, 5)).collect(),
                Some(&readback_layouts),
            )
            .unwrap();
            GpuPreparedStorage::finish_setup(&[&scratch, &readback]).unwrap();
            let request = |columns| {
                parameters
                    .small_rhs_workspaces(parameters.crt_depth() - 1, inner, columns)
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, layout)| {
                        scratch
                            .slot_identity(i + 1)
                            .unwrap()
                            .workspace_request(layout.bytes, layout.alignment)
                    })
                    .collect::<Vec<_>>()
            };
            let mut reservation = scratch.reserve(&request(2)).unwrap();
            for start in (0..width).step_by(2) {
                let end = (start + 2).min(width);
                if start != 0 {
                    reservation.rearm(&request(end - start)).unwrap();
                }
                let dispatch = reservation.enter(Vec::new()).unwrap();
                let right = rhs.column_view(start + 1, end + 1);
                let views = sources
                    .iter()
                    .enumerate()
                    .map(|(i, source)| {
                        source.column_view(1..inner + 1).unwrap().row_view(1..3 + i).unwrap()
                    })
                    .collect::<Vec<_>>();
                let destinations = outputs
                    .into_iter()
                    .enumerate()
                    .map(|(i, output)| (output, 1..3 + i, start + 2..end + 2))
                    .collect();
                outputs = GpuDCRTPolyMatrixColumnView::multiply_small_rhs_row_blocks(
                    &views,
                    right.as_ref(),
                    Some(destinations),
                )
                .unwrap();
                reservation = dispatch.finish().unwrap().pop().unwrap();
            }
            drop(reservation);
            let readers = outputs
                .iter()
                .enumerate()
                .map(|(i, output)| {
                    let claim =
                        readback.slot_identity(i).unwrap().matrix_request(width + 4, 4 + i, true);
                    let dispatch = readback.reserve(&[claim]).unwrap().enter(Vec::new()).unwrap();
                    let reader =
                        output.column_view(0..output.col_size()).unwrap().transpose(None).unwrap();
                    drop(dispatch.finish().unwrap());
                    reader
                })
                .collect::<Vec<_>>();
            drop((sources, outputs, rhs, scratch));
            for (i, (old, reader)) in old_readers.iter().zip(&readers).enumerate() {
                let mut results = Vec::new();
                for value in [old, reader] {
                    let transfer = parameters
                        .rns_transfer_workspace(value.level(), value.row_size(), value.col_size())
                        .unwrap();
                    let events = value.rns_store_completion_events().unwrap();
                    let requests = (2..3 + events)
                        .map(|slot| {
                            let id = readback.slot_identity(slot).unwrap();
                            id.workspace_request(
                                if slot == 2 { transfer.bytes } else { 0 },
                                id.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&requests).unwrap().enter(Vec::new()).unwrap();
                    results.push(value.to_cpu_matrix().transpose());
                    drop(dispatch.finish().unwrap());
                }
                assert_eq!(results[0], initial[i]);
                let actual = &results[1];
                assert_eq!(actual.slice(1, 3 + i, 2, width + 2), expected[i]);
                assert_eq!(actual.slice(0, 1, 0, width + 4), initial[i].slice(0, 1, 0, width + 4));
                assert_eq!(
                    actual.slice(3 + i, 4 + i, 0, width + 4),
                    initial[i].slice(3 + i, 4 + i, 0, width + 4)
                );
                assert_eq!(actual.slice(0, 4 + i, 0, 2), initial[i].slice(0, 4 + i, 0, 2));
                assert_eq!(
                    actual.slice(0, 4 + i, width + 2, width + 4),
                    initial[i].slice(0, 4 + i, width + 2, width + 4)
                );
            }
            assert!(parameters.small_rhs_workspaces(usize::MAX, 1, 1).is_err());
            assert!(
                parameters.small_rhs_workspaces(parameters.crt_depth() - 1, usize::MAX, 2).is_err()
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_crt_recompose_rectangles_preserve_sources_and_lifetimes() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let width = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let cpus = [
            DCRTPolyParams::new(n, 3, 54, 4, Some(vec![narrow[0], wide, narrow[1]]), None),
            DCRTPolyParams::new(n, 2, 54, 4, Some(vec![wide, narrow[0]]), None),
            DCRTPolyParams::new(n, 1, 17, 4, Some(vec![narrow[1]]), None),
        ];
        let target = GpuDCRTPolyParams::new(n, cpus[0].to_crt().0, 4, None);
        let parameters = cpus
            .iter()
            .map(|p| {
                GpuDCRTPolyParams::new_with_gpu(
                    n,
                    p.to_crt().0,
                    4,
                    target.gpu_ids().to_vec(),
                    Some(1),
                    Some(&target),
                    None,
                )
            })
            .collect::<Vec<_>>();
        for count in [1, 2, 5] {
            let plaintext = (0..count).map(|i| [17, 19, 257, 263, 65537][i]).collect::<Vec<u64>>();
            let residues = (0..count)
                .flat_map(|i| target.moduli().iter().map(move |q| q - (i as u64 + 1)))
                .collect::<Vec<_>>();
            let originals = (0..count)
                .map(|i| {
                    DCRTPolyUniformSampler::new().sample_uniform(
                        &cpus[i % 3],
                        4,
                        width + i + 3,
                        DistType::FinRingDist,
                    )
                })
                .collect::<Vec<_>>();
            let selected =
                originals.iter().map(|m| m.slice(1, 2, 1, width + 1)).collect::<Vec<_>>();
            // Existing whole-matrix primitive checks the same arithmetic with
            // independent, zero-offset owners; runtime tests also use the CPU oracle.
            let reference_levels = selected
                .iter()
                .enumerate()
                .map(|(i, cpu)| GpuDCRTPolyMatrix::from_cpu_matrix(&parameters[i % 3], cpu))
                .collect::<Vec<_>>();
            let expected = GpuDCRTPolyMatrix::crt_recompose_levels(
                &reference_levels,
                &plaintext,
                &residues,
                &target,
            )
            .to_cpu_matrix();
            drop(reference_levels);
            let layout = target
                .matrix_crt_workspace_bytes(
                    target.crt_depth() - 1,
                    3,
                    width + 4,
                    GpuMatrixCrtOperation::Recompose,
                    1,
                    count,
                    true,
                )
                .unwrap();
            assert_eq!(layout.pinned_bytes, 0);
            assert_eq!(layout.additional_bytes, layout.workspace_bytes);
            assert_eq!(layout.additional_bytes != 0, count > 2);
            for output_evaluation in [false, true] {
                let inputs = originals
                    .iter()
                    .enumerate()
                    .map(|(i, cpu)| {
                        let mut input = GpuDCRTPolyMatrix::from_cpu_matrix(&parameters[i % 3], cpu);
                        if i % 2 == 0 {
                            input.intt_all_in_place();
                        }
                        input
                    })
                    .collect::<Vec<_>>();
                let initial = DCRTPolyUniformSampler::new().sample_uniform(
                    &cpus[0],
                    3,
                    width + 4,
                    DistType::FinRingDist,
                );
                let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(&target, &initial);
                if !output_evaluation {
                    output.intt_all_in_place();
                }
                let old_reader = output.transpose();
                let raw = output.raw;
                for start in (0..width).step_by(2) {
                    let end = (start + 2).min(width);
                    let views = inputs
                        .iter()
                        .map(|input| {
                            input.column_view(start + 1..end + 1).unwrap().row_view(1..2).unwrap()
                        })
                        .collect::<Vec<_>>();
                    output = GpuDCRTPolyMatrixColumnView::crt_recompose(
                        &views,
                        &plaintext,
                        &residues,
                        &target,
                        Some((output, 1..2, start + 2..end + 2)),
                    )
                    .unwrap();
                }
                let empty = inputs
                    .iter()
                    .map(|input| input.column_view(1..1).unwrap().row_view(1..2).unwrap())
                    .collect::<Vec<_>>();
                output = GpuDCRTPolyMatrixColumnView::crt_recompose(
                    &empty,
                    &plaintext,
                    &residues,
                    &target,
                    Some((output, 1..2, 2..2)),
                )
                .unwrap();
                assert_eq!(output.raw, raw);
                assert_eq!(output.is_ntt(), output_evaluation);
                for (i, input) in inputs.iter().enumerate() {
                    assert_eq!(input.is_ntt(), i % 2 != 0);
                }
                let reader = output.transpose();
                drop((inputs, output));
                assert_eq!(old_reader.to_cpu_matrix(), initial.transpose());
                let actual = reader.to_cpu_matrix().transpose();
                assert_eq!(actual.slice(1, 2, 2, width + 2), expected);
                assert_eq!(actual.slice(0, 1, 0, width + 4), initial.slice(0, 1, 0, width + 4));
                assert_eq!(actual.slice(2, 3, 0, width + 4), initial.slice(2, 3, 0, width + 4));
                assert_eq!(actual.slice(0, 3, 0, 2), initial.slice(0, 3, 0, 2));
                assert_eq!(
                    actual.slice(0, 3, width + 2, width + 4),
                    initial.slice(0, 3, width + 2, width + 4)
                );
            }
        }
        let input = GpuDCRTPolyMatrix::zero(&target, 1, 1);
        let view = input.column_view(0..1).unwrap();
        assert!(
            GpuDCRTPolyMatrixColumnView::crt_recompose(&[view], &[0], &[1, 1, 1], &target, None)
                .is_err()
        );
        assert!(
            GpuDCRTPolyMatrixColumnView::crt_recompose(&[view], &[17], &[1], &target, None)
                .is_err()
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_rns_rectangles_preserve_groups_and_lifetimes() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let width = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        for depth in [3, 9] {
            let narrow = DCRTPolyParams::new(n, depth - 1, 17, 4, None, None).to_crt().0;
            let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
            let mut primes = narrow.clone();
            primes.insert(1, wide);
            let full_cpu = DCRTPolyParams::new(n, depth, 54, 4, Some(primes.clone()), None);
            let small_cpu = DCRTPolyParams::new(
                n,
                depth - 1,
                17,
                4,
                Some(narrow.into_iter().rev().collect()),
                None,
            );
            let full = GpuDCRTPolyParams::new(n, primes, 4, None);
            let small = GpuDCRTPolyParams::new_with_gpu(
                n,
                small_cpu.to_crt().0,
                4,
                full.gpu_ids().to_vec(),
                Some(1),
                Some(&full),
                None,
            );
            for conversion in [
                GpuMatrixRnsConversion::Up { digit_size: 3, normalize: false },
                GpuMatrixRnsConversion::Up { digit_size: 3, normalize: true },
                GpuMatrixRnsConversion::Down { plaintext_modulus: 3 },
            ] {
                let (source, source_cpu, target, target_cpu) = match conversion {
                    GpuMatrixRnsConversion::Up { .. } => (&small, &small_cpu, &full, &full_cpu),
                    GpuMatrixRnsConversion::Down { .. } => (&full, &full_cpu, &small, &small_cpu),
                };
                let original = DCRTPolyUniformSampler::new().sample_uniform(
                    source_cpu,
                    4,
                    width + 2,
                    DistType::FinRingDist,
                );
                let selected = original.slice(1, 3, 1, width + 1);
                let expected = match conversion {
                    GpuMatrixRnsConversion::Up { digit_size, normalize } => {
                        selected.rns_mod_up(target_cpu, digit_size, normalize).unwrap()
                    }
                    GpuMatrixRnsConversion::Down { plaintext_modulus } => {
                        selected.rns_mod_down(target_cpu, plaintext_modulus).unwrap()
                    }
                };
                let rows = expected.row_size();
                let layout = target
                    .matrix_crt_workspace_bytes(
                        target.crt_depth() - 1,
                        rows + 2,
                        width + 4,
                        GpuMatrixCrtOperation::RnsConversion,
                        source.crt_depth(),
                        1,
                        true,
                    )
                    .unwrap();
                assert_eq!(layout.pinned_bytes, 0);
                assert_eq!(layout.additional_bytes, layout.workspace_bytes);
                assert_eq!(layout.additional_bytes != 0, depth == 9);
                for input_evaluation in [false, true] {
                    for output_evaluation in [false, true] {
                        let mut input = GpuDCRTPolyMatrix::from_cpu_matrix(source, &original);
                        if !input_evaluation {
                            input.intt_all_in_place();
                        }
                        let initial = DCRTPolyUniformSampler::new().sample_uniform(
                            target_cpu,
                            rows + 2,
                            width + 4,
                            DistType::FinRingDist,
                        );
                        let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(target, &initial);
                        if !output_evaluation {
                            output.intt_all_in_place();
                        }
                        let old_reader = output.transpose();
                        let raw = output.raw;
                        for start in (0..width).step_by(2) {
                            let end = (start + 2).min(width);
                            output = input
                                .column_view(start + 1..end + 1)
                                .unwrap()
                                .row_view(1..3)
                                .unwrap()
                                .rns_conversion(
                                    target,
                                    conversion,
                                    Some((output, 1..rows + 1, start + 2..end + 2)),
                                )
                                .unwrap();
                        }
                        output = input
                            .column_view(1..1)
                            .unwrap()
                            .row_view(1..3)
                            .unwrap()
                            .rns_conversion(target, conversion, Some((output, 1..rows + 1, 2..2)))
                            .unwrap();
                        assert_eq!(output.raw, raw);
                        assert_eq!(output.is_ntt(), output_evaluation);
                        assert_eq!(input.is_ntt(), input_evaluation);
                        let reader = output.transpose();
                        drop((input, output));
                        assert_eq!(old_reader.to_cpu_matrix(), initial.transpose());
                        let actual = reader.to_cpu_matrix().transpose();
                        assert_eq!(actual.slice(1, rows + 1, 2, width + 2), expected);
                        assert_eq!(
                            actual.slice(0, 1, 0, width + 4),
                            initial.slice(0, 1, 0, width + 4)
                        );
                        assert_eq!(
                            actual.slice(rows + 1, rows + 2, 0, width + 4),
                            initial.slice(rows + 1, rows + 2, 0, width + 4)
                        );
                        assert_eq!(
                            actual.slice(0, rows + 2, 0, 2),
                            initial.slice(0, rows + 2, 0, 2)
                        );
                        assert_eq!(
                            actual.slice(0, rows + 2, width + 2, width + 4),
                            initial.slice(0, rows + 2, width + 2, width + 4)
                        );
                    }
                }
            }
            assert!(
                GpuMatrixRnsConversion::Up { digit_size: 0, normalize: false }
                    .validate(&small, depth - 2, &full)
                    .is_err()
            );
            assert!(
                GpuMatrixRnsConversion::Down { plaintext_modulus: wide }
                    .validate(&full, depth - 1, &small)
                    .is_err()
            );
            assert!(
                GpuMatrixRnsConversion::Down { plaintext_modulus: 1 }
                    .validate(&full, depth - 1, &small)
                    .is_err()
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_modulus_conversion_rectangles_preserve_rounding_and_lifetimes() {
        use super::super::GpuMatrixCrtOperation;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let width = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for primes in [vec![narrow[0], wide, narrow[1]], vec![wide, narrow[1], narrow[0]]] {
            let full_cpu = DCRTPolyParams::new(n, 3, 54, 4, Some(primes.clone()), None);
            let small_cpu =
                DCRTPolyParams::new(n, 2, 17, 4, Some(vec![narrow[1], narrow[0]]), None);
            let full = GpuDCRTPolyParams::new(n, primes, 4, None);
            let small = GpuDCRTPolyParams::new_with_gpu(
                n,
                small_cpu.to_crt().0,
                4,
                full.gpu_ids().to_vec(),
                Some(1),
                Some(&full),
                None,
            );
            for conversion in [
                GpuMatrixModulusConversion::Reduce,
                GpuMatrixModulusConversion::Round,
                GpuMatrixModulusConversion::CenteredExtend,
                GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus: 3 },
            ] {
                let (source, source_cpu, target, target_cpu) =
                    if conversion == GpuMatrixModulusConversion::CenteredExtend {
                        (&small, &small_cpu, &full, &full_cpu)
                    } else {
                        (&full, &full_cpu, &small, &small_cpu)
                    };
                let original = DCRTPolyUniformSampler::new().sample_uniform(
                    source_cpu,
                    5,
                    width + 3,
                    DistType::FinRingDist,
                );
                let selected = original.slice(1, 4, 1, width + 1);
                let expected = match conversion {
                    GpuMatrixModulusConversion::Reduce => selected.reduce_modulus(target_cpu),
                    GpuMatrixModulusConversion::Round => selected.modulus_switch(target_cpu),
                    GpuMatrixModulusConversion::CenteredExtend => {
                        selected.centered_extend(target_cpu).unwrap()
                    }
                    GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus } => {
                        selected.block_mod_switch(target_cpu, plaintext_modulus).unwrap()
                    }
                };
                // Ensure retained ranges also work when a whole owner could fit
                // metadata in its auxiliary arena. Retained ranges need no scratch.
                let mut owner_columns = width + 4;
                while target
                    .matrix_crt_workspace_bytes(
                        target.crt_depth() - 1,
                        5,
                        owner_columns,
                        GpuMatrixCrtOperation::ConvertModulus,
                        source.crt_depth(),
                        1,
                        false,
                    )
                    .unwrap()
                    .additional_bytes !=
                    0
                {
                    owner_columns *= 2;
                }
                let layout = target
                    .matrix_crt_workspace_bytes(
                        target.crt_depth() - 1,
                        5,
                        owner_columns,
                        GpuMatrixCrtOperation::ConvertModulus,
                        source.crt_depth(),
                        1,
                        true,
                    )
                    .unwrap();
                assert_eq!(layout.additional_bytes, layout.workspace_bytes);
                assert_eq!(layout.additional_bytes, 0);
                assert_eq!(layout.pinned_bytes, 0);
                for input_evaluation in [false, true] {
                    for output_evaluation in [false, true] {
                        if conversion == GpuMatrixModulusConversion::Reduce &&
                            input_evaluation != output_evaluation
                        {
                            continue;
                        }
                        let mut input = GpuDCRTPolyMatrix::from_cpu_matrix(source, &original);
                        if !input_evaluation {
                            input.intt_all_in_place();
                        }
                        let initial = DCRTPolyUniformSampler::new().sample_uniform(
                            target_cpu,
                            5,
                            owner_columns,
                            DistType::FinRingDist,
                        );
                        let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(target, &initial);
                        if !output_evaluation {
                            output.intt_all_in_place();
                        }
                        let old_reader = output.transpose();
                        let raw = output.raw;
                        for start in (0..width).step_by(2) {
                            let end = (start + 2).min(width);
                            output = input
                                .column_view(start + 1..end + 1)
                                .unwrap()
                                .row_view(1..4)
                                .unwrap()
                                .convert_modulus(
                                    target,
                                    conversion,
                                    Some((output, 1..4, start + 2..end + 2)),
                                )
                                .unwrap();
                        }
                        output = input
                            .column_view(1..1)
                            .unwrap()
                            .row_view(1..4)
                            .unwrap()
                            .convert_modulus(target, conversion, Some((output, 1..4, 2..2)))
                            .unwrap();
                        assert_eq!(output.raw, raw);
                        assert_eq!(output.is_ntt(), output_evaluation);
                        assert_eq!(input.is_ntt(), input_evaluation);
                        let reader = output.transpose();
                        drop((input, output));
                        assert_eq!(old_reader.to_cpu_matrix(), initial.transpose());
                        let actual = reader.to_cpu_matrix().transpose();
                        assert_eq!(actual.slice(1, 4, 2, width + 2), expected);
                        assert_eq!(
                            actual.slice(0, 1, 0, owner_columns),
                            initial.slice(0, 1, 0, owner_columns)
                        );
                        assert_eq!(
                            actual.slice(4, 5, 0, owner_columns),
                            initial.slice(4, 5, 0, owner_columns)
                        );
                        assert_eq!(actual.slice(0, 5, 0, 2), initial.slice(0, 5, 0, 2));
                        assert_eq!(
                            actual.slice(0, 5, width + 2, owner_columns),
                            initial.slice(0, 5, width + 2, owner_columns)
                        );
                    }
                }
            }
            assert!(GpuMatrixModulusConversion::Round.validate(&small, 1, &full).is_err());
            assert!(
                GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus: 0 }
                    .validate(&full, 2, &small)
                    .is_err()
            );
            assert!(
                GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus: narrow[0] }
                    .validate(&full, 2, &small)
                    .is_err()
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_centered_rebase_rectangles_preserve_formats_and_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let width = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let target_cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let target = GpuDCRTPolyParams::new(n, moduli, 4, None);
            for modulus in [narrow, wide] {
                let source_cpu = DCRTPolyParams::new(
                    n,
                    1,
                    if modulus == narrow { 17 } else { 54 },
                    4,
                    Some(vec![modulus]),
                    None,
                );
                let source = GpuDCRTPolyParams::new_with_gpu(
                    n,
                    vec![modulus],
                    4,
                    target.gpu_ids().to_vec(),
                    Some(1),
                    Some(&target),
                    None,
                );
                let original = DCRTPolyUniformSampler::new().sample_uniform(
                    &source_cpu,
                    5,
                    width + 3,
                    DistType::FinRingDist,
                );
                let expected =
                    original.slice(1, 4, 1, width + 1).centered_rebase(&target_cpu).unwrap();
                for input_evaluation in [false, true] {
                    for output_evaluation in [false, true] {
                        let mut input = GpuDCRTPolyMatrix::from_cpu_matrix(&source, &original);
                        if !input_evaluation {
                            input.intt_all_in_place();
                        }
                        let initial = DCRTPolyUniformSampler::new().sample_uniform(
                            &target_cpu,
                            5,
                            width + 4,
                            DistType::FinRingDist,
                        );
                        let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(&target, &initial);
                        if !output_evaluation {
                            output.intt_all_in_place();
                        }
                        let old_reader = output.transpose();
                        let raw = output.raw;
                        for start in (0..width).step_by(2) {
                            let end = (start + 2).min(width);
                            output = input
                                .column_view(start + 1..end + 1)
                                .unwrap()
                                .row_view(1..4)
                                .unwrap()
                                .centered_rebase(&target, Some((output, 1..4, start + 2..end + 2)))
                                .unwrap();
                        }
                        output = input
                            .column_view(1..1)
                            .unwrap()
                            .row_view(1..4)
                            .unwrap()
                            .centered_rebase(&target, Some((output, 1..4, 2..2)))
                            .unwrap();
                        assert_eq!(output.raw, raw);
                        assert_eq!(output.is_ntt(), output_evaluation);
                        assert_eq!(input.is_ntt(), input_evaluation);
                        let reader = output.transpose();
                        drop((input, output));
                        assert_eq!(old_reader.to_cpu_matrix(), initial.transpose());
                        let actual = reader.to_cpu_matrix().transpose();
                        assert_eq!(actual.slice(1, 4, 2, width + 2), expected);
                        assert_eq!(
                            actual.slice(0, 1, 0, width + 4),
                            initial.slice(0, 1, 0, width + 4)
                        );
                        assert_eq!(
                            actual.slice(4, 5, 0, width + 4),
                            initial.slice(4, 5, 0, width + 4)
                        );
                        assert_eq!(actual.slice(0, 5, 0, 2), initial.slice(0, 5, 0, 2));
                        assert_eq!(
                            actual.slice(0, 5, width + 2, width + 4),
                            initial.slice(0, 5, width + 2, width + 4)
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_retained_polynomial_preserves_format_and_dma_lifetime() {
        use crate::poly::dcrt::poly::DCRTPoly;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let modulus = BigInt::from(params.modulus().as_ref().clone());
            let cases = [
                Vec::new(),
                vec![BigInt::from(-1), (&modulus << 5) + 3],
                (0..n)
                    .into_par_iter()
                    .map(|_| BigInt::from(rand::random::<i64>()) << 20)
                    .collect::<Vec<_>>(),
            ];
            for coefficients in cases {
                let expected_coefficients = coefficients
                    .par_iter()
                    .map(|value| ((value % &modulus + &modulus) % &modulus).to_biguint().unwrap())
                    .collect::<Vec<_>>();
                let expected = DCRTPolyMatrix::from_poly_vec(
                    &cpu,
                    vec![vec![DCRTPoly::from_biguints(&cpu, &expected_coefficients)]],
                );
                for evaluation in [false, true] {
                    let initial = DCRTPolyUniformSampler::new().sample_uniform(
                        &cpu,
                        1,
                        1,
                        DistType::FinRingDist,
                    );
                    let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &initial);
                    if !evaluation {
                        output.intt_all_in_place();
                    }
                    let old_reader = output.transpose();
                    let raw = output.raw;
                    output.fill_polynomial(&coefficients, false).unwrap();
                    assert_eq!(output.raw, raw);
                    assert_eq!(output.is_ntt(), evaluation);
                    let reader = output.transpose();
                    drop(output);
                    assert_eq!(old_reader.to_cpu_matrix(), initial);
                    assert_eq!(reader.to_cpu_matrix(), expected);
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_sample_rectangles_preserve_streams_formats_and_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let width = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            for distribution in [
                GpuMatrixSampleDist::Uniform,
                GpuMatrixSampleDist::Gauss,
                GpuMatrixSampleDist::Bit,
                GpuMatrixSampleDist::Ternary,
            ] {
                let seed = crate::sampler::gpu::random_gpu_rng_seed();
                let full = GpuDCRTPolyMatrix::sample_distribution_coeff(
                    &params,
                    3,
                    width + 4,
                    distribution,
                    4.578,
                    3,
                    seed,
                )
                .to_cpu_matrix();
                for evaluation in [false, true] {
                    let initial = DCRTPolyUniformSampler::new().sample_uniform(
                        &cpu,
                        5,
                        width + 4,
                        DistType::FinRingDist,
                    );
                    let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &initial);
                    if !evaluation {
                        output.intt_all_in_place();
                    }
                    let old_reader = output.transpose();
                    let raw = output.raw;
                    for local in (0..width).step_by(2) {
                        output
                            .fill_distribution_columns(
                                1..4,
                                local + 2..(local + 2).min(width) + 2,
                                width + 4,
                                local + 1,
                                distribution,
                                4.578,
                                3,
                                seed,
                            )
                            .unwrap();
                    }
                    output
                        .fill_distribution_columns(
                            1..4,
                            width + 2..width + 2,
                            width + 4,
                            width + 1,
                            distribution,
                            4.578,
                            3,
                            seed,
                        )
                        .unwrap();
                    assert_eq!(output.raw, raw);
                    assert_eq!(output.is_ntt(), evaluation);
                    assert!(
                        output
                            .fill_distribution_columns(
                                1..4,
                                2..width + 2,
                                width + 4,
                                usize::MAX,
                                distribution,
                                4.578,
                                3,
                                seed
                            )
                            .is_err()
                    );
                    assert!(
                        output
                            .fill_distribution_columns(
                                1..4,
                                2..width + 2,
                                width + 4,
                                1,
                                GpuMatrixSampleDist::Gauss,
                                f64::INFINITY,
                                3,
                                seed
                            )
                            .is_err()
                    );
                    let reader = output.transpose();
                    drop(output);
                    assert_eq!(old_reader.to_cpu_matrix(), initial.transpose());
                    let actual = reader.to_cpu_matrix().transpose();
                    assert_eq!(actual.slice(1, 4, 2, width + 2), full.slice_columns(1, width + 1));
                    assert_eq!(actual.slice(0, 1, 0, width + 4), initial.slice(0, 1, 0, width + 4));
                    assert_eq!(actual.slice(4, 5, 0, width + 4), initial.slice(4, 5, 0, width + 4));
                    assert_eq!(actual.slice(0, 5, 0, 2), initial.slice(0, 5, 0, 2));
                    assert_eq!(
                        actual.slice(0, 5, width + 2, width + 4),
                        initial.slice(0, 5, width + 2, width + 4)
                    );
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_sampling_ranges_use_only_retained_output_slots() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let seed = crate::sampler::gpu::random_gpu_rng_seed();
        let expected = GpuDCRTPolyMatrix::sample_distribution(
            &params,
            3,
            columns,
            GpuMatrixSampleDist::Gauss,
            4.578,
            3,
            seed,
        )
        .to_cpu_matrix();
        let mut layouts =
            vec![params.rns_transfer_workspace(params.crt_depth() - 1, 3, columns).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            params.crt_depth(),
        ));
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 3, columns)],
            Some(&layouts),
        )
        .unwrap();
        GpuPreparedStorage::finish_setup(&[&storage]).unwrap();
        for _ in 0..2 {
            let request = storage.slot_identity(0).unwrap().matrix_request(3, columns, true);
            let dispatch = storage.reserve(&[request]).unwrap().enter(Vec::new()).unwrap();
            let mut output = GpuDCRTPolyMatrix::new_empty_with_state(
                &params,
                3,
                columns,
                params.crt_depth() - 1,
                true,
                None,
            );
            for start in (0..columns).step_by(2) {
                output
                    .fill_distribution_columns(
                        0..3,
                        start..(start + 2).min(columns),
                        columns,
                        start,
                        GpuMatrixSampleDist::Gauss,
                        4.578,
                        3,
                        seed,
                    )
                    .unwrap();
            }
            drop(dispatch.finish().unwrap());
            let events = output.rns_store_completion_events().unwrap();
            let claims = (1..2 + events)
                .map(|i| {
                    let slot = storage.slot_identity(i).unwrap();
                    slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                })
                .collect::<Vec<_>>();
            let dispatch = storage.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
            assert_eq!(output.to_cpu_matrix(), expected);
            drop(dispatch.finish().unwrap());
            drop(output);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_constant_rectangles_preserve_global_indices_and_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let digits = params.crt_bits().div_ceil(params.base_bits() as usize);
            let mut cases = vec![
                (
                    GpuMatrixRangeConstant::Zero { total_columns: columns + 4 },
                    DCRTPolyMatrix::zero(&cpu, 3, columns + 4),
                    2,
                    columns,
                ),
                (
                    GpuMatrixRangeConstant::Identity,
                    DCRTPolyMatrix::identity(&cpu, columns + 2, None),
                    1,
                    columns,
                ),
                (
                    GpuMatrixRangeConstant::UnitRow { total_columns: columns + 4, index: 3 },
                    DCRTPolyMatrix::unit_row_vector(&cpu, columns + 4, 3),
                    2,
                    columns,
                ),
                (
                    GpuMatrixRangeConstant::UnitColumn { index: 1 },
                    DCRTPolyMatrix::unit_column_vector(&cpu, 3, 1),
                    0,
                    1,
                ),
                (
                    GpuMatrixRangeConstant::Gadget { small: true, digit_count: None },
                    DCRTPolyMatrix::small_gadget_matrix(&cpu, 3),
                    digits - 1,
                    columns.min(digits + 1),
                ),
            ];
            for digit_count in [Some(digits), None] {
                let expected = DCRTPolyMatrix::gadget_matrix(&cpu, 3, digit_count);
                let start = expected.col_size() / 3 - 1;
                cases.push((
                    GpuMatrixRangeConstant::Gadget { small: false, digit_count },
                    expected,
                    start,
                    columns.min(digits + 1),
                ));
            }
            for (constant, full, start, width) in cases {
                let rows = full.row_size();
                let initial = DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    rows + 2,
                    width + 4,
                    DistType::FinRingDist,
                );
                let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &initial);
                let old_reader = output.transpose();
                let raw = output.raw;
                for local in (0..width).step_by(2) {
                    output
                        .fill_constant_columns(
                            1..rows + 1,
                            local + 2..(local + 2).min(width) + 2,
                            start + local,
                            constant,
                        )
                        .unwrap();
                }
                output
                    .fill_constant_columns(
                        1..rows + 1,
                        width + 2..width + 2,
                        start + width,
                        constant,
                    )
                    .unwrap();
                assert_eq!(output.raw, raw);
                assert!(
                    output
                        .fill_constant_columns(1..rows + 1, 2..width + 2, usize::MAX, constant)
                        .is_err()
                );
                let reader = output.transpose();
                drop(output);
                assert_eq!(old_reader.to_cpu_matrix(), initial.transpose());
                let actual = reader.to_cpu_matrix().transpose();
                assert_eq!(
                    actual.slice(1, rows + 1, 2, width + 2),
                    full.slice_columns(start, start + width)
                );
                assert_eq!(actual.slice(0, 1, 0, width + 4), initial.slice(0, 1, 0, width + 4));
                assert_eq!(
                    actual.slice(rows + 1, rows + 2, 0, width + 4),
                    initial.slice(rows + 1, rows + 2, 0, width + 4)
                );
                assert_eq!(actual.slice(0, rows + 2, 0, 2), initial.slice(0, rows + 2, 0, 2));
                assert_eq!(
                    actual.slice(0, rows + 2, width + 2, width + 4),
                    initial.slice(0, rows + 2, width + 2, width + 4)
                );
            }
            let initial = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                3,
                columns + 2,
                DistType::FinRingDist,
            );
            let mut coefficient = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &initial);
            coefficient.intt_all_in_place();
            assert!(
                coefficient
                    .fill_constant_columns(0..3, 0..1, 0, GpuMatrixRangeConstant::Identity)
                    .is_err()
            );
            coefficient
                .fill_constant_columns(
                    1..2,
                    1..columns + 1,
                    0,
                    GpuMatrixRangeConstant::Zero { total_columns: columns },
                )
                .unwrap();
            assert!(!coefficient.is_ntt());
            let actual = coefficient.to_cpu_matrix();
            assert_eq!(actual.slice(1, 2, 1, columns + 1), DCRTPolyMatrix::zero(&cpu, 1, columns));
            assert_eq!(actual.slice(0, 1, 0, columns + 2), initial.slice(0, 1, 0, columns + 2));
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_constant_ranges_use_only_retained_output_slots() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let mut layouts =
            vec![params.rns_transfer_workspace(params.crt_depth() - 1, columns, columns).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            params.crt_depth(),
        ));
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, columns, columns)],
            Some(&layouts),
        )
        .unwrap();
        GpuPreparedStorage::finish_setup(&[&storage]).unwrap();
        for _ in 0..2 {
            let request = storage.slot_identity(0).unwrap().matrix_request(columns, columns, true);
            let dispatch = storage.reserve(&[request]).unwrap().enter(Vec::new()).unwrap();
            let mut output = GpuDCRTPolyMatrix::new_empty_with_state(
                &params,
                columns,
                columns,
                params.crt_depth() - 1,
                true,
                None,
            );
            for start in (0..columns).step_by(2) {
                output
                    .fill_constant_columns(
                        0..columns,
                        start..(start + 2).min(columns),
                        start,
                        GpuMatrixRangeConstant::Identity,
                    )
                    .unwrap();
            }
            drop(dispatch.finish().unwrap());
            let events = output.rns_store_completion_events().unwrap();
            let claims = (1..2 + events)
                .map(|i| {
                    let slot = storage.slot_identity(i).unwrap();
                    slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                })
                .collect::<Vec<_>>();
            let dispatch = storage.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
            assert_eq!(output.to_cpu_matrix(), DCRTPolyMatrix::identity(&cpu, columns, None));
            drop(dispatch.finish().unwrap());
            drop(output);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_accumulate_views_cover_scalars_wide_limbs_and_early_release() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let shapes = [
                (3, 2, 2, columns),
                (3, 3, 3, columns),
                (1, 1, 3, columns),
                (3, columns, 1, 1),
                (3, 0, 0, columns),
                (3, 2, 2, columns),
                (3, 1, 1, columns),
            ];
            let coefficients = [
                BigInt::from(7),
                BigInt::from(-3),
                BigInt::from(2),
                -(BigInt::from(1) << 137usize) - BigInt::from(13),
                BigInt::from(1),
                BigInt::from(0),
                BigInt::from(1),
            ];
            let owners = shapes
                .par_iter()
                .map(|&(lr, lc, rr, rc)| {
                    let sample = |r, c| {
                        DCRTPolyUniformSampler::new().sample_uniform(
                            &cpu,
                            r,
                            c,
                            DistType::FinRingDist,
                        )
                    };
                    (sample(lr + 2, lc + 3), sample(rr + 3, rc + 2))
                })
                .collect::<Vec<_>>();
            let bias_cpu = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                5,
                columns + 3,
                DistType::FinRingDist,
            );
            let initial = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                5,
                columns + 4,
                DistType::FinRingDist,
            );
            let expected_terms = owners
                .par_iter()
                .zip(shapes.par_iter())
                .zip(coefficients.par_iter())
                .map(|(((left, right), &(lr, lc, rr, rc)), coefficient)| {
                    let left = left.slice(1, lr + 1, 2, lc + 2);
                    let right = right.slice(2, rr + 2, 1, rc + 1);
                    let product = if (lr, lc) == (1, 1) {
                        right.multiply_poly_out_of_place(&left.entry(0, 0))
                    } else if (rr, rc) == (1, 1) {
                        left.multiply_poly_out_of_place(&right.entry(0, 0))
                    } else if lc == 0 {
                        DCRTPolyMatrix::zero(&cpu, lr, rc)
                    } else {
                        left.multiply_out_of_place(&right)
                    };
                    let modulus = BigInt::from(cpu.modulus().as_ref().clone());
                    let residue =
                        ((coefficient % &modulus + &modulus) % &modulus).to_biguint().unwrap();
                    product.multiply_poly_out_of_place(&DCRTPoly::from_biguint_to_constant(
                        &cpu, residue,
                    ))
                })
                .collect::<Vec<_>>();
            let owners = owners
                .par_iter()
                .map(|(left, right)| {
                    (
                        GpuDCRTPolyMatrix::from_cpu_matrix(&params, left),
                        GpuDCRTPolyMatrix::from_cpu_matrix(&params, right),
                    )
                })
                .collect::<Vec<_>>();
            let bias = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &bias_cpu);
            for with_bias in [false, true] {
                let output = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &initial);
                let old_reader = output.transpose();
                let products = owners
                    .iter()
                    .zip(shapes)
                    .zip(coefficients.iter())
                    .map(|(((left, right), (lr, lc, rr, rc)), coefficient)| {
                        (
                            coefficient.clone(),
                            left.column_view(2..lc + 2).unwrap().row_view(1..lr + 1).unwrap(),
                            right.column_view(1..rc + 1).unwrap().row_view(2..rr + 2).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                let output = GpuDCRTPolyMatrixColumnView::multiply_accumulate(
                    &products,
                    with_bias
                        .then(|| bias.column_view(1..columns + 1).unwrap().row_view(1..4).unwrap()),
                    Some((output, 1..4, 2..columns + 2)),
                )
                .unwrap();
                let reader = output.transpose();
                drop(output);
                let expected = expected_terms.iter().fold(
                    if with_bias {
                        bias_cpu.slice(1, 4, 1, columns + 1)
                    } else {
                        DCRTPolyMatrix::zero(&cpu, 3, columns)
                    },
                    |sum, term| sum.add_out_of_place(term),
                );
                let actual = reader.to_cpu_matrix().transpose();
                assert_eq!(old_reader.to_cpu_matrix(), initial.transpose());
                assert_eq!(actual.slice(1, 4, 2, columns + 2), expected);
                assert_eq!(actual.slice(0, 1, 0, columns + 4), initial.slice(0, 1, 0, columns + 4));
                assert_eq!(actual.slice(4, 5, 0, columns + 4), initial.slice(4, 5, 0, columns + 4));
                assert_eq!(actual.slice(0, 5, 0, 2), initial.slice(0, 5, 0, 2));
                assert_eq!(
                    actual.slice(0, 5, columns + 2, columns + 4),
                    initial.slice(0, 5, columns + 2, columns + 4)
                );
            }
            let products = owners
                .iter()
                .zip(shapes)
                .zip(coefficients.iter())
                .map(|(((left, right), (lr, lc, rr, rc)), coefficient)| {
                    (
                        coefficient.clone(),
                        left.column_view(2..lc + 2).unwrap().row_view(1..lr + 1).unwrap(),
                        right.column_view(1..rc + 1).unwrap().row_view(2..rr + 2).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            let output =
                GpuDCRTPolyMatrixColumnView::multiply_accumulate(&products, None, None).unwrap();
            drop((owners, bias));
            assert_eq!(
                output.to_cpu_matrix(),
                expected_terms.iter().fold(DCRTPolyMatrix::zero(&cpu, 3, columns), |sum, term| sum
                    .add_out_of_place(term))
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_row_block_views_preserve_offsets_and_retain_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let blocks = (0..18)
                .into_par_iter()
                .map(|i| {
                    DCRTPolyUniformSampler::new().sample_uniform(
                        &cpu,
                        4,
                        columns + 1 + i % 2,
                        DistType::FinRingDist,
                    )
                })
                .collect::<Vec<_>>();
            let pieces = blocks
                .par_iter()
                .enumerate()
                .map(|(i, block)| block.slice(1, 1 + i % 3, i % 2, columns + i % 2))
                .collect::<Vec<_>>();
            let stacked = pieces[0].concat_rows(&pieces[1..].iter().collect::<Vec<_>>());
            let rows = stacked.row_size();
            let rhs_cpu = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                rows + 3,
                columns + 4,
                DistType::FinRingDist,
            );
            let initial = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                rows + 2,
                columns + 3,
                DistType::FinRingDist,
            );
            let expected = stacked.add_out_of_place(&rhs_cpu.slice(2, rows + 2, 1, columns + 1));
            let blocks = blocks
                .par_iter()
                .map(|block| GpuDCRTPolyMatrix::from_cpu_matrix(&params, block))
                .collect::<Vec<_>>();
            let rhs = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &rhs_cpu);
            let output = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &initial);
            let initial_reader = output.transpose();
            let views = blocks
                .par_iter()
                .enumerate()
                .map(|(i, block)| {
                    block
                        .column_view(i % 2..columns + i % 2)
                        .unwrap()
                        .row_view(1..1 + i % 3)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let output = rhs
                .column_view(1..columns + 1)
                .unwrap()
                .row_view(2..rows + 2)
                .unwrap()
                .add_row_blocks(&views, Some((output, 1..rows + 1, 2..columns + 2)))
                .unwrap();
            let reader = output.transpose();
            drop((output, blocks, rhs));
            let actual = reader.to_cpu_matrix().transpose();
            assert_eq!(initial_reader.to_cpu_matrix(), initial.transpose());
            assert_eq!(actual.slice(1, rows + 1, 2, columns + 2), expected);
            assert_eq!(actual.slice(0, 1, 0, columns + 3), initial.slice(0, 1, 0, columns + 3));
            assert_eq!(
                actual.slice(rows + 1, rows + 2, 0, columns + 3),
                initial.slice(rows + 1, rows + 2, 0, columns + 3)
            );
            assert_eq!(actual.slice(0, rows + 2, 0, 2), initial.slice(0, rows + 2, 0, 2));
            assert_eq!(
                actual.slice(0, rows + 2, columns + 2, columns + 3),
                initial.slice(0, rows + 2, columns + 2, columns + 3)
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_peer_rectangles_preserve_offsets_and_early_release() {
        use crate::poly::dcrt::gpu::detected_gpu_device_ids;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(vec![narrow, wide]), None);
        let devices = detected_gpu_device_ids();
        let base = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![devices[0]],
            Some(1),
            None,
            None,
        );
        let target = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![devices[0]],
            Some(1),
            Some(&base),
            None,
        );
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            5,
            columns + 2,
            DistType::FinRingDist,
        );
        for evaluation in [false, true] {
            let mut source = GpuDCRTPolyMatrix::from_cpu_matrix(&base, &original);
            let mut output = GpuDCRTPolyMatrix::zero(&target, 4, columns + 2);
            if !evaluation {
                source.intt_all_in_place();
                output.intt_all_in_place();
            }
            let pointer = output.raw;
            for range in [0..1, 1..columns] {
                output = source
                    .column_view(1 + range.start..1 + range.end)
                    .unwrap()
                    .row_view(2..4)
                    .unwrap()
                    .copy(Some((output, 1..3, 1 + range.start..1 + range.end)))
                    .unwrap();
            }
            assert_eq!(pointer, output.raw);
            drop(source);
            let actual = output.to_cpu_matrix();
            assert_eq!(actual.slice(1, 3, 1, columns + 1), original.slice(2, 4, 1, columns + 1));
            assert_eq!(
                actual.slice(0, 1, 0, columns + 2),
                DCRTPolyMatrix::zero(&cpu, 1, columns + 2)
            );
            assert_eq!(
                actual.slice(3, 4, 0, columns + 2),
                DCRTPolyMatrix::zero(&cpu, 1, columns + 2)
            );
            assert_eq!(actual.slice(0, 4, 0, 1), DCRTPolyMatrix::zero(&cpu, 4, 1));
            assert_eq!(
                actual.slice(0, 4, columns + 1, columns + 2),
                DCRTPolyMatrix::zero(&cpu, 4, 1)
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_tensor_views_cross_factor_boundaries_and_retain_inputs() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let left_cpu = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                5,
                columns + 2,
                DistType::FinRingDist,
            );
            let right_cpu =
                DCRTPolyUniformSampler::new().sample_uniform(&cpu, 4, 4, DistType::FinRingDist);
            let left = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left_cpu);
            let right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &right_cpu);
            let expected =
                left_cpu.slice(1, 4, 1, columns + 1).tensor(&right_cpu.slice(1, 3, 1, 3));
            let groups = vec![vec![5, 0, 5], vec![2]];
            let mut outputs = Vec::new();
            for reduction in [None, Some(groups.as_slice())] {
                let rows = reduction.map_or(6, <[Vec<usize>]>::len);
                let mut output = GpuDCRTPolyMatrix::zero(&params, rows + 2, 2 * columns + 2);
                let pointer = output.raw;
                for range in [0..1, 1..3, 3..2 * columns] {
                    output = left
                        .column_view(1..columns + 1)
                        .unwrap()
                        .row_view(1..4)
                        .unwrap()
                        .tensor(
                            &right.column_view(1..3).unwrap().row_view(1..3).unwrap(),
                            reduction,
                            range.clone(),
                            Some((output, 1..rows + 1, range.start + 1..range.end + 1)),
                        )
                        .unwrap();
                }
                assert_eq!(output.raw, pointer);
                outputs.push((output, rows));
            }
            drop(left);
            drop(right);
            for ((output, rows), expected) in
                outputs.into_iter().zip([expected.clone(), expected.sum_rows(&groups)])
            {
                let actual = output.to_cpu_matrix();
                assert_eq!(actual.slice(1, rows + 1, 1, 2 * columns + 1), expected);
                assert_eq!(
                    actual.slice(0, 1, 0, 2 * columns + 2),
                    DCRTPolyMatrix::zero(&cpu, 1, 2 * columns + 2)
                );
                assert_eq!(
                    actual.slice(rows + 1, rows + 2, 0, 2 * columns + 2),
                    DCRTPolyMatrix::zero(&cpu, 1, 2 * columns + 2)
                );
                assert_eq!(
                    actual.slice(0, rows + 2, 0, 1),
                    DCRTPolyMatrix::zero(&cpu, rows + 2, 1)
                );
                assert_eq!(
                    actual.slice(0, rows + 2, 2 * columns + 1, 2 * columns + 2),
                    DCRTPolyMatrix::zero(&cpu, rows + 2, 1)
                );
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_row_transform_views_preserve_rectangles_and_source_lifetime() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let original = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                5,
                columns + 2,
                DistType::FinRingDist,
            );
            let groups = vec![vec![2, 0, 2], vec![1]];
            let source = original.slice(1, 4, 1, columns + 1);
            for evaluation in [false, true] {
                let mut input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
                let mut transposed = GpuDCRTPolyMatrix::zero(&params, columns + 2, 5);
                let mut summed = GpuDCRTPolyMatrix::zero(&params, 4, columns + 2);
                if !evaluation {
                    input.intt_all_in_place();
                    transposed.intt_all_in_place();
                    summed.intt_all_in_place();
                }
                let transpose_owner = transposed.raw;
                let sum_owner = summed.raw;
                for range in [0..1, 1..3] {
                    transposed = input
                        .column_view(1..columns + 1)
                        .unwrap()
                        .row_view(1 + range.start..1 + range.end)
                        .unwrap()
                        .transpose(Some((
                            transposed,
                            1..columns + 1,
                            1 + range.start..1 + range.end,
                        )))
                        .unwrap();
                }
                for range in [0..1, 1..columns] {
                    summed = input
                        .column_view(1 + range.start..1 + range.end)
                        .unwrap()
                        .row_view(1..4)
                        .unwrap()
                        .sum_rows(&groups, Some((summed, 1..3, 1 + range.start..1 + range.end)))
                        .unwrap();
                }
                assert_eq!(transposed.raw, transpose_owner);
                assert_eq!(summed.raw, sum_owner);
                assert_eq!(transposed.is_ntt(), evaluation);
                assert_eq!(summed.is_ntt(), evaluation);
                drop(input);
                // The producer owner is gone before either consumer is observed.
                for (actual, expected, rows, cols) in [
                    (transposed.to_cpu_matrix(), source.transpose(), columns, 3),
                    (summed.to_cpu_matrix(), source.sum_rows(&groups), 2, columns),
                ] {
                    assert_eq!(actual.slice(1, rows + 1, 1, cols + 1), expected);
                    assert_eq!(
                        actual.slice(0, 1, 0, cols + 2),
                        DCRTPolyMatrix::zero(&cpu, 1, cols + 2)
                    );
                    assert_eq!(
                        actual.slice(rows + 1, rows + 2, 0, cols + 2),
                        DCRTPolyMatrix::zero(&cpu, 1, cols + 2)
                    );
                    assert_eq!(
                        actual.slice(0, rows + 2, 0, 1),
                        DCRTPolyMatrix::zero(&cpu, rows + 2, 1)
                    );
                    assert_eq!(
                        actual.slice(0, rows + 2, cols + 1, cols + 2),
                        DCRTPolyMatrix::zero(&cpu, rows + 2, 1)
                    );
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_multiply_views_preserve_offsets_and_retire_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [narrow.clone(), vec![narrow[0], wide]] {
            let bits = moduli.iter().map(|q| (64 - q.leading_zeros()) as usize).max().unwrap();
            let cpu = DCRTPolyParams::new(n, 2, bits, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            // One row exercises the thin Barrett kernel for narrow limbs;
            // multiple rows and mixed-width limbs exercise the tiled kernel.
            for rows in [1, 3] {
                let inner = 3;
                let lhs = DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    rows + 2,
                    inner + 2,
                    DistType::FinRingDist,
                );
                let rhs = DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    inner + 2,
                    columns + 4,
                    DistType::FinRingDist,
                );
                let expected = lhs
                    .slice(1, rows + 1, 1, inner + 1)
                    .multiply_out_of_place(&rhs.slice(1, inner + 1, 2, columns + 2));
                let left = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &lhs);
                let right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &rhs);
                let mut output = GpuDCRTPolyMatrix::zero(&params, rows + 2, columns + 2);
                let pointer = output.raw;
                for range in [0..columns - 1, columns - 1..columns] {
                    output = left
                        .column_view(1..inner + 1)
                        .unwrap()
                        .row_view(1..rows + 1)
                        .unwrap()
                        .multiply(
                            &right
                                .column_view(range.start + 2..range.end + 2)
                                .unwrap()
                                .row_view(1..inner + 1)
                                .unwrap(),
                            Some((output, 1..rows + 1, range.start + 1..range.end + 1)),
                        )
                        .unwrap();
                }
                assert_eq!(output.raw, pointer);
                drop(left);
                drop(right);
                // Releasing both readers before readback must preserve their
                // allocations until both disjoint output ranges are complete.
                let actual = output.to_cpu_matrix();
                assert_eq!(actual.slice(1, rows + 1, 1, columns + 1), expected);
                assert_eq!(
                    actual.slice(0, 1, 0, columns + 2),
                    DCRTPolyMatrix::zero(&cpu, 1, columns + 2)
                );
                assert_eq!(
                    actual.slice(rows + 1, rows + 2, 0, columns + 2),
                    DCRTPolyMatrix::zero(&cpu, 1, columns + 2)
                );
                assert_eq!(
                    actual.slice(0, rows + 2, 0, 1),
                    DCRTPolyMatrix::zero(&cpu, rows + 2, 1)
                );
                assert_eq!(
                    actual.slice(0, rows + 2, columns + 1, columns + 2),
                    DCRTPolyMatrix::zero(&cpu, rows + 2, 1)
                );
            }
            let original = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                2,
                columns,
                DistType::FinRingDist,
            );
            let scalar =
                DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
            let expected = original.multiply_poly_out_of_place(&scalar.entry(0, 0));
            let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
            let scalar = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &scalar);
            for reverse in [false, true] {
                let matrix = input.column_view(0..columns).unwrap();
                let factor = scalar.column_view(0..1).unwrap();
                let output = if reverse {
                    factor.multiply(&matrix, None)
                } else {
                    matrix.multiply(&factor, None)
                }
                .unwrap();
                assert_eq!(output.to_cpu_matrix(), expected);
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_unary_views_match_signed_scalar_and_automorphism_references() {
        use num_bigint::BigInt;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(1);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let original = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                4,
                columns + 3,
                DistType::FinRingDist,
            );
            let expected_input = original.slice(1, 3, 1, columns + 1);
            for evaluation in [false, true] {
                let mut input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
                if !evaluation {
                    input.intt_all_in_place();
                }
                let view = input.column_view(1..columns + 1).unwrap().row_view(1..3).unwrap();
                for scalar in [
                    BigInt::from(0),
                    BigInt::from(7),
                    -(BigInt::from(1) << 137usize) - BigInt::from(13),
                ] {
                    let modulus = BigInt::from(cpu.modulus().as_ref().clone());
                    let reduced =
                        ((&scalar % &modulus + &modulus) % &modulus).to_biguint().unwrap();
                    let expected = expected_input.multiply_poly_out_of_place(
                        &DCRTPoly::from_biguint_to_constant(&cpu, reduced),
                    );
                    let mut destination = GpuDCRTPolyMatrix::zero(&params, 4, columns + 2);
                    if !evaluation {
                        destination.intt_all_in_place();
                    }
                    let pointer = destination.raw;
                    let output = view
                        .scale_integer(&scalar, Some((destination, 1..3, 1..columns + 1)))
                        .unwrap();
                    assert_eq!(output.raw, pointer);
                    assert_eq!(output.is_ntt(), evaluation);
                    let actual = output.to_cpu_matrix();
                    assert_eq!(actual.slice(1, 3, 1, columns + 1), expected);
                    assert_eq!(
                        actual.slice(0, 1, 0, columns + 2),
                        DCRTPolyMatrix::zero(&cpu, 1, columns + 2)
                    );
                    assert_eq!(
                        actual.slice(3, 4, 0, columns + 2),
                        DCRTPolyMatrix::zero(&cpu, 1, columns + 2)
                    );
                    assert_eq!(actual.slice(0, 4, 0, 1), DCRTPolyMatrix::zero(&cpu, 4, 1));
                    assert_eq!(
                        actual.slice(0, 4, columns + 1, columns + 2),
                        DCRTPolyMatrix::zero(&cpu, 4, 1)
                    );
                    let standalone = view.scale_integer(&scalar, None).unwrap();
                    assert!(standalone.is_ntt());
                    assert_eq!(standalone.to_cpu_matrix(), expected);
                }
                for index in [1, 3, 2 * n as usize - 1] {
                    let output = view.ring_automorphism(index, None).unwrap();
                    assert!(output.is_ntt());
                    assert_eq!(
                        output.to_cpu_matrix(),
                        expected_input.ring_automorphism_out_of_place(index)
                    );
                    if evaluation {
                        let destination = GpuDCRTPolyMatrix::zero(&params, 2, columns);
                        let output = view
                            .ring_automorphism(index, Some((destination, 0..2, 0..columns)))
                            .unwrap();
                        assert!(output.is_ntt());
                        assert_eq!(
                            output.to_cpu_matrix(),
                            expected_input.ring_automorphism_out_of_place(index)
                        );
                    }
                }
                assert_eq!(input.to_cpu_matrix(), original);
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_column_views_preserve_mixed_width_rows_and_pending_readers() {
        // Pool counters are process-wide. Isolate this allocation-free view
        // assertion from unrelated tests that create or release CUDA owners.
        const CHILD: &str = "MXX_GPU_COLUMN_VIEW_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let module = module_path!().split_once("::").unwrap().1;
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg(format!(
                    "{module}::test_gpu_column_views_preserve_mixed_width_rows_and_pending_readers"
                ))
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
            return;
        }
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let rows = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("matrix rows"))
            .unwrap_or(5);
        let columns = rows.checked_add(6).unwrap();
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let input = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                rows,
                columns,
                DistType::FinRingDist,
            );
            let expected = input.slice(0, rows, 2, columns - 1);
            let scalar = DCRTPoly::from_biguint_to_constant(&cpu, BigUint::from(7u32));
            let gpu_scalar = GpuDCRTPoly::from_biguint_to_constant(&params, BigUint::from(7u32));
            for evaluation in [false, true] {
                let mut owner = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
                if !evaluation {
                    owner.intt_all_in_place();
                }
                owner.wait_until_ready();
                params.fence_released_memory();
                let before = gpu_default_mempool_usage(params.device_ids()[0]).unwrap();
                let view = owner.column_view(2..columns - 1).unwrap();
                let alias = view;
                assert_eq!(alias.size(), (rows, columns - 3));
                assert_eq!(
                    gpu_default_mempool_usage(params.device_ids()[0]).unwrap().used_current,
                    before.used_current
                );
                let results = [
                    view.negate(None).unwrap(),
                    alias.multiply_poly(&gpu_scalar, None).unwrap(),
                    view.ring_automorphism(3, None).unwrap(),
                ];
                let readers = results.par_iter().map(PolyMatrix::transpose).collect::<Vec<_>>();
                drop(results);
                drop(owner);
                let expected = [
                    expected.negate_out_of_place(),
                    expected.multiply_poly_out_of_place(&scalar),
                    expected.ring_automorphism_out_of_place(3),
                ];
                for (reader, expected) in readers.into_iter().zip(expected) {
                    assert_eq!(reader.to_cpu_matrix(), expected.transpose());
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_column_views_reject_invalid_ranges_and_query_metadata() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let owner = GpuDCRTPolyMatrix::zero(&params, 2, 5);
        assert!(owner.column_view(4..3).is_err());
        assert!(owner.column_view(0..6).is_err());
        assert!(owner.column_view(usize::MAX..usize::MAX).is_err());
        assert!(owner.column_view(1..4).unwrap().row_view(2..1).is_err());
        assert!(owner.column_view(1..4).unwrap().row_view(0..3).is_err());
        assert!(owner.column_view(1..4).unwrap().row_view(usize::MAX..usize::MAX).is_err());
        assert!(owner.column_view(1..4).unwrap().ring_automorphism(2, None).is_err());
        let empty = owner.column_view(2..2).unwrap();
        assert_eq!(empty.negate(None).unwrap().size(), (2, 0));
        assert_eq!(empty.ring_automorphism(3, None).unwrap().size(), (2, 0));
        let destination = GpuDCRTPolyMatrix::zero(&params, 4, 7);
        let pointer = destination.raw;
        let destination = empty.negate(Some((destination, 1..3, 4..4))).unwrap();
        assert_eq!(destination.raw, pointer);
        assert_eq!(destination.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 4, 7));
        let destination = GpuDCRTPolyMatrix::zero(&params, 4, 7);
        assert!(empty.negate(Some((destination, 0..3, 4..4))).is_err());
        let destination = GpuDCRTPolyMatrix::zero(&params, 4, 7);
        // Evaluation destinations now support a direct permutation. Verify the
        // complete owner instead of the superseded unsupported-format error.
        let destination = owner
            .column_view(1..4)
            .unwrap()
            .ring_automorphism(3, Some((destination, 1..3, 2..5)))
            .unwrap();
        assert_eq!(destination.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 4, 7));
        for (rows, columns) in [(2, 0), (0, 3)] {
            let owner = GpuDCRTPolyMatrix::zero(&params, rows, columns);
            let viewed = owner.column_view(0..columns).unwrap().ring_automorphism(3, None).unwrap();
            let ordinary = owner.ring_automorphism_out_of_place(3);
            assert_eq!(viewed.is_ntt, ordinary.is_ntt);
            assert_eq!(viewed.size(), ordinary.size());
        }
        for operation in [
            GpuMatrixBatchOperation::Binary,
            GpuMatrixBatchOperation::Negate,
            GpuMatrixBatchOperation::Scalar,
            GpuMatrixBatchOperation::Automorphism,
            GpuMatrixBatchOperation::Multiply,
        ] {
            for count in [1, 1024] {
                let full = params
                    .matrix_batch_workspace_bytes(1, (2, 3), count, 1, operation, false)
                    .unwrap();
                let view = params
                    .matrix_batch_workspace_bytes(1, (2, 3), count, 1, operation, true)
                    .unwrap();
                assert!(
                    view.workspace_bytes >=
                        full.workspace_bytes + count * 6 * std::mem::size_of::<usize>()
                );
            }
        }
        assert!(
            params
                .matrix_batch_workspace_bytes(
                    1,
                    (2, 3),
                    1,
                    1,
                    GpuMatrixBatchOperation::Accumulate,
                    true
                )
                .is_err()
        );
        let empty_cpu = DCRTPolyMatrix::zero(&cpu, 2, 0);
        assert_eq!(empty.negate(None).unwrap().to_cpu_matrix(), empty_cpu);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_column_view_batches_keep_independent_owner_pitches() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let inputs = [7, 9].map(|columns| {
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist)
        });
        let ranges = [
            GpuMatrixRange { row_start: 0, row_end: 2, column_start: 1, column_end: 4 },
            GpuMatrixRange { row_start: 0, row_end: 2, column_start: 4, column_end: 7 },
        ];
        let views = ranges.map(|range| GpuMatrixBatchView {
            left: range,
            right: range,
            output: GpuMatrixRange { row_start: 0, row_end: 2, column_start: 0, column_end: 3 },
        });
        for evaluation in [false, true] {
            let owners = inputs.each_ref().map(|input| {
                let mut matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, input);
                if !evaluation {
                    matrix.intt_all_in_place();
                }
                matrix
            });
            let pointers = owners.each_ref().map(|matrix| matrix.raw.cast_const());
            let outputs = [0, 1].map(|_| {
                GpuDCRTPolyMatrix::new_empty_with_state(&params, 2, 3, 1, evaluation, None)
            });
            let destinations = outputs.each_ref().map(|matrix| matrix.raw);
            let status = unsafe {
                gpu_matrix_negate_batch(destinations.as_ptr(), pointers.as_ptr(), views.as_ptr(), 2)
            };
            assert_eq!(status, 0, "{}", last_error_string());
            let scalars = [3u32, 7].map(|value| {
                let mut scalar =
                    GpuDCRTPoly::from_biguint_to_constant(&params, BigUint::from(value));
                scalar.ntt_in_place();
                scalar
            });
            let mut transformed = [0, 1].map(|_| {
                GpuDCRTPolyMatrix::new_empty_with_state(&params, 2, 3, 1, evaluation, None)
            });
            let destinations = transformed.each_ref().map(|matrix| matrix.raw);
            let status = if evaluation {
                let scalars = scalars.each_ref().map(|scalar| scalar.inner().raw.cast_const());
                unsafe {
                    gpu_matrix_mul_scalar_batch(
                        destinations.as_ptr(),
                        pointers.as_ptr(),
                        scalars.as_ptr(),
                        views.as_ptr(),
                        2,
                        std::ptr::null(),
                    )
                }
            } else {
                unsafe {
                    gpu_matrix_ring_automorphism_batch(
                        destinations.as_ptr(),
                        pointers.as_ptr(),
                        [3usize, 5].as_ptr(),
                        views.as_ptr(),
                        2,
                    )
                }
            };
            assert_eq!(status, 0, "{}", last_error_string());
            if !evaluation {
                transformed.iter_mut().for_each(GpuDCRTPolyMatrix::ntt_all_in_place);
            }
            let readers = transformed.each_ref().map(PolyMatrix::transpose);
            drop(transformed);
            drop(owners);
            drop(scalars);
            for (index, ((output, reader), range)) in
                outputs.into_iter().zip(readers).zip(ranges).enumerate()
            {
                let source = inputs[index].slice(0, 2, range.column_start, range.column_end);
                assert_eq!(output.to_cpu_matrix(), source.negate_out_of_place());
                let expected = if evaluation {
                    source.multiply_poly_out_of_place(&DCRTPoly::from_biguint_to_constant(
                        &cpu,
                        BigUint::from([3u32, 7][index]),
                    ))
                } else {
                    source.ring_automorphism_out_of_place([3usize, 5][index])
                };
                assert_eq!(reader.to_cpu_matrix(), expected.transpose());
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_column_view_batches_reject_cross_input_and_output_aliases() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let ranges =
            [GpuMatrixRange { row_start: 0, row_end: 2, column_start: 0, column_end: 3 }; 2];
        let views =
            ranges.map(|range| GpuMatrixBatchView { left: range, right: range, output: range });
        let mut scalar = GpuDCRTPoly::from_biguint_to_constant(&params, BigUint::from(3u32));
        scalar.ntt_in_place();
        for operation in 0..5 {
            let mut owners = [0, 1, 2].map(|_| GpuDCRTPolyMatrix::zero(&params, 2, 3));
            if operation == 2 {
                owners.iter_mut().for_each(GpuDCRTPolyMatrix::intt_all_in_place);
            }
            let inputs = [owners[0].raw.cast_const(), owners[1].raw.cast_const()];
            for outputs in [[owners[1].raw, owners[2].raw], [owners[2].raw; 2]] {
                let status = unsafe {
                    match operation {
                        0 => gpu_matrix_negate_batch(
                            outputs.as_ptr(),
                            inputs.as_ptr(),
                            views.as_ptr(),
                            2,
                        ),
                        1 => gpu_matrix_mul_scalar_batch(
                            outputs.as_ptr(),
                            inputs.as_ptr(),
                            [scalar.inner().raw.cast_const(); 2].as_ptr(),
                            views.as_ptr(),
                            2,
                            std::ptr::null(),
                        ),
                        2 => gpu_matrix_ring_automorphism_batch(
                            outputs.as_ptr(),
                            inputs.as_ptr(),
                            [3usize, 5].as_ptr(),
                            views.as_ptr(),
                            2,
                        ),
                        _ => gpu_matrix_binary_batch(
                            outputs.as_ptr(),
                            inputs.as_ptr(),
                            [inputs[1], inputs[0]].as_ptr(),
                            views.as_ptr(),
                            2,
                            operation - 3,
                        ),
                    }
                };
                assert_ne!(status, 0, "aliased output accepted for operation {operation}");
            }
            let expected = DCRTPolyMatrix::zero(&cpu, 2, 3);
            for owner in owners {
                assert_eq!(owner.to_cpu_matrix(), expected);
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_destination_views_preserve_full_owners_and_pending_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let rows = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("matrix rows"))
            .unwrap_or(2)
            .max(1);
        let columns = 3;
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli, 4, None);
            let left_cpu = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                rows + 3,
                columns + 7,
                DistType::FinRingDist,
            );
            let right_cpu = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                rows + 5,
                columns + 11,
                DistType::FinRingDist,
            );
            let initial = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                2 * rows + 2,
                2 * columns + 3,
                DistType::FinRingDist,
            );
            let left_expected = left_cpu.slice(1, rows + 1, 2, columns + 2);
            let right_expected = right_cpu.slice(2, rows + 2, 5, columns + 5);
            let magnitude = (BigUint::from(1u32) << 100usize) + BigUint::from(7u32);
            let scalar_cpu = -DCRTPoly::from_biguint_to_constant(&cpu, magnitude.clone());
            let mut scalar = -GpuDCRTPoly::from_biguint_to_constant(&params, magnitude);
            scalar.ntt_in_place();
            for evaluation in [false, true] {
                let mut left = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left_cpu);
                let mut right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &right_cpu);
                if !evaluation {
                    left.intt_all_in_place();
                    right.intt_all_in_place();
                }
                let left_view =
                    left.column_view(2..columns + 2).unwrap().row_view(1..rows + 1).unwrap();
                let right_view =
                    right.column_view(5..columns + 5).unwrap().row_view(2..rows + 2).unwrap();
                let mut results = Vec::new();
                for operation in 0..5 {
                    let output_eval = match operation {
                        3 => true,
                        4 => false,
                        _ => evaluation,
                    };
                    let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &initial);
                    if !output_eval {
                        output.intt_all_in_place();
                    }
                    let pointer = output.raw;
                    let original_reader = output.transpose();
                    let apply = |output, destination_rows, destination_columns| {
                        let destination = Some((output, destination_rows, destination_columns));
                        match operation {
                            0 => left_view.negate(destination),
                            1 => left_view.add(&right_view, destination),
                            2 => left_view.sub(&right_view, destination),
                            3 => left_view.multiply_poly(&scalar, destination),
                            _ => left_view.ring_automorphism(3, destination),
                        }
                        .unwrap()
                    };
                    let piece = match operation {
                        0 => left_expected.negate_out_of_place(),
                        1 => left_expected.add_out_of_place(&right_expected),
                        2 => left_expected.sub_out_of_place(&right_expected),
                        3 => left_expected.multiply_poly_out_of_place(&scalar_cpu),
                        _ => left_expected.ring_automorphism_out_of_place(3),
                    };
                    output = apply(output, 1..rows + 1, 1..columns + 1);
                    assert_eq!(output.raw, pointer);
                    assert_eq!(output.is_ntt, output_eval);
                    let intermediate_reader = output.transpose();
                    let mut intermediate = initial.clone();
                    intermediate.copy_block_from(&piece, 1, 1, 0, 0, rows, columns);
                    output = apply(output, rows + 1..2 * rows + 1, columns + 2..2 * columns + 2);
                    assert_eq!(output.raw, pointer);
                    if operation == 4 {
                        output.ntt_all_in_place();
                    }
                    let final_reader = output.transpose();
                    drop(output);
                    let mut expected = intermediate.clone();
                    expected.copy_block_from(&piece, rows + 1, columns + 2, 0, 0, rows, columns);
                    results.push((original_reader, initial.transpose()));
                    results.push((intermediate_reader, intermediate.transpose()));
                    results.push((final_reader, expected.transpose()));
                }
                drop(left);
                drop(right);
                for (actual, expected) in results {
                    assert_eq!(actual.to_cpu_matrix(), expected);
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_destination_batches_share_disjoint_rectangles_and_reject_overlap() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(vec![wide, narrow]), None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let left_cpu = [(5, 8), (7, 11)].map(|(rows, columns)| {
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, rows, columns, DistType::FinRingDist)
        });
        let right_cpu = [(9, 13), (6, 10)].map(|(rows, columns)| {
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, rows, columns, DistType::FinRingDist)
        });
        let initial =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 6, 10, DistType::FinRingDist);
        let rectangle = |row_start, column_start| GpuMatrixRange {
            row_start,
            row_end: row_start + 2,
            column_start,
            column_end: column_start + 3,
        };
        let views = [
            GpuMatrixBatchView {
                left: rectangle(1, 2),
                right: rectangle(3, 7),
                output: rectangle(1, 1),
            },
            GpuMatrixBatchView {
                left: rectangle(4, 6),
                right: rectangle(2, 5),
                output: rectangle(3, 5),
            },
        ];
        for operation in 0..5 {
            let coefficient = operation == 4;
            let left = left_cpu.each_ref().map(|input| {
                let mut matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, input);
                if coefficient {
                    matrix.intt_all_in_place();
                }
                matrix
            });
            let right = right_cpu.each_ref().map(|input| {
                let mut matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, input);
                if coefficient {
                    matrix.intt_all_in_place();
                }
                matrix
            });
            let mut output = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &initial);
            if coefficient {
                output.intt_all_in_place();
            }
            let initial_reader = output.transpose();
            let pointers = [output.raw; 2];
            let left_pointers = left.each_ref().map(|matrix| matrix.raw.cast_const());
            let right_pointers = right.each_ref().map(|matrix| matrix.raw.cast_const());
            let mut scalar = GpuDCRTPoly::from_biguint_to_constant(&params, BigUint::from(7u32));
            scalar.ntt_in_place();
            let dispatch = |ranges: &[GpuMatrixBatchView; 2]| unsafe {
                match operation {
                    0 => gpu_matrix_negate_batch(
                        pointers.as_ptr(),
                        left_pointers.as_ptr(),
                        ranges.as_ptr(),
                        2,
                    ),
                    1 | 2 => gpu_matrix_binary_batch(
                        pointers.as_ptr(),
                        left_pointers.as_ptr(),
                        right_pointers.as_ptr(),
                        ranges.as_ptr(),
                        2,
                        operation - 1,
                    ),
                    3 => gpu_matrix_mul_scalar_batch(
                        pointers.as_ptr(),
                        left_pointers.as_ptr(),
                        [scalar.inner().raw.cast_const(); 2].as_ptr(),
                        ranges.as_ptr(),
                        2,
                        std::ptr::null(),
                    ),
                    _ => gpu_matrix_ring_automorphism_batch(
                        pointers.as_ptr(),
                        left_pointers.as_ptr(),
                        [3usize, 5].as_ptr(),
                        ranges.as_ptr(),
                        2,
                    ),
                }
            };
            let mut overlapping = views;
            overlapping[1].output = rectangle(2, 2);
            assert_ne!(dispatch(&overlapping), 0);
            let mut invalid = views;
            invalid[1].output.row_end = usize::MAX;
            assert_ne!(dispatch(&invalid), 0);
            invalid = views;
            invalid[1].left.row_start = invalid[1].left.row_end;
            assert_ne!(dispatch(&invalid), 0);
            let status = dispatch(&views);
            assert_eq!(status, 0, "{}", last_error_string());
            let mut expected = initial.clone();
            for (index, view) in views.iter().enumerate() {
                let left = left_cpu[index].slice(
                    view.left.row_start,
                    view.left.row_end,
                    view.left.column_start,
                    view.left.column_end,
                );
                let right = right_cpu[index].slice(
                    view.right.row_start,
                    view.right.row_end,
                    view.right.column_start,
                    view.right.column_end,
                );
                let piece = match operation {
                    0 => left.negate_out_of_place(),
                    1 => left.add_out_of_place(&right),
                    2 => left.sub_out_of_place(&right),
                    3 => left.multiply_poly_out_of_place(&DCRTPoly::from_biguint_to_constant(
                        &cpu,
                        BigUint::from(7u32),
                    )),
                    _ => left.ring_automorphism_out_of_place([3usize, 5][index]),
                };
                expected.copy_block_from(
                    &piece,
                    view.output.row_start,
                    view.output.column_start,
                    0,
                    0,
                    2,
                    3,
                );
            }
            if coefficient {
                output.ntt_all_in_place();
            }
            let reader = output.transpose();
            drop(output);
            drop(left);
            drop(right);
            drop(scalar);
            assert_eq!(initial_reader.to_cpu_matrix(), initial.transpose());
            assert_eq!(reader.to_cpu_matrix(), expected.transpose());
        }
    }
}
