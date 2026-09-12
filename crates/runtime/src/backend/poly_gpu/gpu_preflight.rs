//! Isolated column calibration using the same range kernels as production.
//!
//! This moves covered pilot classes to the typed preflight boundary. Complete
//! prepared-storage admission and the remaining invocation classes are separate
//! requirements; default-pool profiles here do not certify a physical bound.

use super::*;
use crate::gpu_invocation::{GpuColumnSourceLayout, GpuInvocation};
use mxx_ir_core::node::{HashVariant, MatrixBinaryOp};
use mxx_primitives::matrix::gpu_dcrt_poly::GpuCpuStagingLayout;

type ColumnRunner<T> =
    dyn Fn(usize, &mut DeviceBackend, usize, usize) -> Result<T, PolyBackendError> + Send + Sync;

impl GpuDcrtBackend {
    pub(super) fn calibrate_column_operation<T: PilotReady + Send + 'static>(
        &mut self,
        columns: usize,
        operation: Arc<ColumnRunner<T>>,
    ) -> Result<(), PolyBackendError> {
        self.calibrate_column_waves(columns, move |fleet, wave| {
            let runner = operation.clone();
            fleet.launch_column_wave(wave, move |device, backend, start, end| {
                runner(device, backend, start, end)
            })
        })
    }

    pub(super) fn calibrate_column_waves<T: PilotReady>(
        &mut self,
        columns: usize,
        execute: impl Fn(
            &mut Self,
            &[(usize, usize, usize)],
        ) -> Result<Vec<GpuColumnShard<T>>, PolyBackendError>,
    ) -> Result<(), PolyBackendError> {
        if columns == 0 {
            self.pending_pilot = None;
            self.pending_profile = None;
            return Ok(());
        }
        while self.runtime_pilot_is_pending() {
            let wave = self.next_column_wave(0, columns);
            let launched = execute(self, &wave)?;
            let completed = self.finish_runtime_pilot(&launched)?;
            drop(launched);
            if !completed {
                self.restart_runtime_pilot_after_fixed_inputs()
                    .map_err(PolyBackendError::GpuCalibration)?;
            }
        }
        Ok(())
    }

    pub(super) fn preimage_column_runner<'a>(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &GpuFleetTrapdoor,
        public: &GpuFleetMatrix,
        target: &'a dyn PolyMatrixColumnSource<GpuFleetMatrix>,
        randomness_seed: [u8; 32],
    ) -> Result<
        impl Fn(
            &mut Self,
            &[(usize, usize, usize)],
        ) -> Result<Vec<GpuColumnShard<GpuSmallMatrix>>, PolyBackendError>
        + use<'a>,
        PolyBackendError,
    > {
        self.validate_preimage_bound(ty, sigma, gadget_base, digit_count, max_coefficient_bound)?;
        if trapdoor.values.len() != self.devices.len() {
            return Err(PolyBackendError::InvalidInteger);
        }
        let public_replicas = self.full_matrix_replicas(public)?;
        if self.runtime_pilot_is_pending() {
            trapdoor.wait_until_ready();
            public_replicas.iter().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let public_replicas = Arc::new(public_replicas);
        let gadget_base = Arc::new(gadget_base.clone());
        let max_coefficient_bound = Arc::new(max_coefficient_bound.clone());
        let source_global_column_start = target.global_column_start();
        let target_rows = target.row_size();
        let ty = ty.clone();
        let trapdoors = trapdoor.values.clone();
        Ok(move |fleet: &mut Self, wave: &[(usize, usize, usize)]| {
            let targets = wave
                .par_iter()
                .map(|&(device, start, end)| (device, target.column_range(start, end)))
                .collect::<Vec<_>>();
            let ty = ty.clone();
            let trapdoors = trapdoors.clone();
            let public_replicas = public_replicas.clone();
            let gadget_base = gadget_base.clone();
            let max_coefficient_bound = max_coefficient_bound.clone();
            let launched = fleet.launch_column_wave(wave, move |device, backend, start, end| {
                let input = &targets
                    .iter()
                    .find(|(owner, _)| *owner == device)
                    .expect("prepared target for active device")
                    .1;
                let local_ty = ConcreteMatrixType { columns: end - start, ..ty.clone() };
                let global_column_start =
                    preimage_seed_column_start(source_global_column_start, start)?;
                // Staged targets remain on the host until the primitive sampler
                // materializes its actual inner tile on this worker's owner.
                let target = OffsetGpuColumnSource::on_device(
                    backend,
                    &local_ty,
                    target_rows,
                    input,
                    global_column_start,
                )?;
                backend.sample_preimage(
                    &local_ty,
                    sigma,
                    &gadget_base,
                    digit_count,
                    &max_coefficient_bound,
                    &trapdoors[device],
                    &public_replicas[device],
                    &target,
                    randomness_seed,
                )
            })?;
            Ok::<_, PolyBackendError>(launched)
        })
    }

    pub(super) fn uniform_column_runner(
        ty: &ConcreteMatrixType,
        range: &SampleRange,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let ty = ty.clone();
        let range = range.clone();
        move |_, backend, start, end| {
            backend
                .sample_uniform(&ConcreteMatrixType { columns: end - start, ..ty.clone() }, &range)
        }
    }

    pub(super) fn gaussian_column_runner(
        ty: &ConcreteMatrixType,
        sigma: f64,
        bound: &BigInt,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let ty = ty.clone();
        let bound = bound.clone();
        move |_, backend, start, end| {
            backend.sample_gaussian(
                &ConcreteMatrixType { columns: end - start, ..ty.clone() },
                sigma,
                &bound,
            )
        }
    }

    pub(super) fn hash_column_runner(
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let ty = ty.clone();
        let tag = tag.to_vec();
        move |_, backend, start, end| {
            Ok(GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash_columns(
                backend.parameters(&ty)?,
                key,
                &tag,
                ty.rows,
                ty.columns,
                start,
                end - start,
                DistType::FinRingDist,
            ))
        }
    }

    pub(super) fn decomposed_hash_column_runner(
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        digit_count: usize,
        small: bool,
    ) -> Result<
        impl Fn(usize, &mut DeviceBackend, usize, usize) -> Result<GpuSmallMatrix, PolyBackendError>
        + Send
        + Sync
        + 'static,
        PolyBackendError,
    > {
        if digit_count == 0 || !ty.rows.is_multiple_of(digit_count) {
            return Err(PolyBackendError::InvalidInteger);
        }
        let ty = ty.clone();
        let tag = tag.to_vec();
        Ok(move |_, backend: &mut DeviceBackend, start, end| {
            let source = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                .sample_hash_gadget_source_columns(
                    backend.parameters(&ty)?,
                    key,
                    &tag,
                    ty.rows / digit_count,
                    ty.columns,
                    start,
                    end - start,
                    DistType::FinRingDist,
                );
            source.gadget_decompose(small, Some(digit_count)).map_err(PolyBackendError::from)
        })
    }

    pub(super) fn unary_column_runner(
        value: &GpuFleetMatrix,
        operation: GpuUnaryColumnOperation,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let input = value.clone();
        move |_, backend, start, end| {
            if let GpuUnaryColumnOperation::Materialized(operation) = &operation {
                let piece = Self::matrix_operand_on_device(backend, &input, start, end)?;
                return operation(backend, &piece);
            }
            // Production jobs intersect the input owner's stored intervals.
            // Retain the complete allocation and pass its real row pitch to
            // the native view kernel instead of copying an interior range.
            let resident = input.shards.iter().position(|shard| {
                shard.global_column_start <= start &&
                    end <= shard.global_column_start + shard.value.col_size() &&
                    backend.matrix_is_on_active_placement(&shard.value)
            });
            let (piece, columns) = if let Some(index) = resident {
                let offset = input.shards[index].global_column_start;
                (
                    GpuMatrixOperand::Resident { shards: input.shards.clone(), index },
                    start - offset..end - offset,
                )
            } else {
                // Existing representative pilots may execute on a device
                // that does not own this range. Their transport remains
                // explicit until isolated preflight replaces runtime pilots.
                (
                    GpuMatrixOperand::Materialized(Self::matrix_piece_on_device(
                        backend, &input, start, end,
                    )?),
                    0..end - start,
                )
            };
            let view = piece.column_view(columns).map_err(PolyBackendError::GpuSubmission)?;
            match &operation {
                GpuUnaryColumnOperation::Negate => view.negate(None),
                GpuUnaryColumnOperation::Scale(scalar) => view.scale_integer(scalar, None),
                GpuUnaryColumnOperation::Automorphism(index) => {
                    view.ring_automorphism(*index, None)
                }
                GpuUnaryColumnOperation::Materialized(_) => unreachable!(),
            }
            .map_err(PolyBackendError::GpuSubmission)
        }
    }

    pub(super) fn transpose_column_runner(
        value: &GpuFleetMatrix,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let input = value.clone();
        move |_, backend, start, end| {
            if start == 0 && end == input.rows {
                let input = Self::matrix_operand_on_device(backend, &input, 0, input.columns)?;
                backend.transpose(&input)
            } else {
                // A destination column interval is exactly a source row range;
                // every source column shard contributes to those output rows.
                let input = Self::matrix_rows_on_device(backend, &input, start, end)?;
                backend.transpose(&input)
            }
        }
    }
    pub(super) fn tensor_column_runner(
        left: &GpuFleetMatrix,
        right: &GpuFleetMatrix,
        columns: usize,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let (left, right) = (left.clone(), right.clone());
        move |_, backend, start, end| {
            if start == 0 && end == columns {
                let left = Self::matrix_operand_on_device(backend, &left, 0, left.columns)?;
                let right = Self::matrix_operand_on_device(backend, &right, 0, right.columns)?;
                return backend.tensor(&left, &right);
            }
            let pieces = tensor_column_segments(start, end, right.columns)
                .into_iter()
                .map(|(left_column, right_start, right_end)| {
                    let left = Self::matrix_operand_on_device(
                        backend,
                        &left,
                        left_column,
                        left_column + 1,
                    )?;
                    let right =
                        Self::matrix_operand_on_device(backend, &right, right_start, right_end)?;
                    backend.tensor(&left, &right)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut pieces = pieces.into_iter();
            let first = pieces.next().expect("nonempty tensor range");
            Ok(if pieces.len() == 0 { first } else { first.concat_columns_owned(pieces.collect()) })
        }
    }
    pub(super) fn tensor_sum_rows_column_runner(
        left: &GpuFleetMatrix,
        right: &GpuFleetMatrix,
        groups: &[Vec<usize>],
        columns: usize,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let groups = Arc::new(groups.to_vec());
        let (left, right) = (left.clone(), right.clone());
        move |_, backend, start, end| {
            if start == 0 && end == columns {
                let left = Self::matrix_operand_on_device(backend, &left, 0, left.columns)?;
                let right = Self::matrix_operand_on_device(backend, &right, 0, right.columns)?;
                return backend.tensor_sum_rows(&left, &right, &groups);
            }
            let pieces = tensor_column_segments(start, end, right.columns)
                .into_iter()
                .map(|(left_column, right_start, right_end)| {
                    let left = Self::matrix_operand_on_device(
                        backend,
                        &left,
                        left_column,
                        left_column + 1,
                    )?;
                    let right =
                        Self::matrix_operand_on_device(backend, &right, right_start, right_end)?;
                    backend.tensor_sum_rows(&left, &right, &groups)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut pieces = pieces.into_iter();
            let first = pieces.next().expect("nonempty tensor range");
            Ok(if pieces.len() == 0 { first } else { first.concat_columns_owned(pieces.collect()) })
        }
    }
    pub(super) fn gadget_decompose_row_blocks_column_runner(
        blocks: &[&GpuFleetMatrix],
        small: bool,
        digit_count: Option<usize>,
    ) -> impl Fn(usize, &mut DeviceBackend, usize, usize) -> Result<GpuSmallMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let owned_blocks = blocks.iter().map(|block| (*block).clone()).collect::<Vec<_>>();
        move |_, backend, start, end| {
            let pieces = owned_blocks
                .iter()
                .map(|block| Self::matrix_operand_on_device(backend, block, start, end))
                .collect::<Result<Vec<_>, _>>()?;
            backend.gadget_decompose_row_blocks(
                &pieces.iter().map(|piece| piece.as_ref()).collect::<Vec<_>>(),
                small,
                digit_count,
            )
        }
    }
    pub(super) fn add_row_blocks_column_runner(
        blocks: &[&GpuFleetMatrix],
        right: &GpuFleetMatrix,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let owned_blocks = blocks.iter().map(|block| (*block).clone()).collect::<Vec<_>>();
        let input_right = right.clone();
        move |_, backend, start, end| {
            let local = owned_blocks
                .iter()
                .map(|block| Self::matrix_operand_on_device(backend, block, start, end))
                .collect::<Result<Vec<_>, _>>()?;
            let right = Self::matrix_operand_on_device(backend, &input_right, start, end)?;
            let refs = local.iter().map(|block| block.as_ref()).collect::<Vec<_>>();
            backend.add_row_blocks(&refs, &right)
        }
    }
    pub(super) fn crt_recompose_column_runner(
        levels: &[GpuFleetMatrix],
        plaintext_moduli: &[BigInt],
        reconstruction_coefficients: &[BigInt],
        destination: &ConcreteMatrixType,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let owned_levels = levels.to_vec();
        let plaintext_moduli = plaintext_moduli.to_vec();
        let reconstruction_coefficients = reconstruction_coefficients.to_vec();
        let destination = destination.clone();
        move |_, backend, start, end| {
            let local = owned_levels
                .iter()
                .map(|level| Self::matrix_piece_on_device(backend, level, start, end))
                .collect::<Result<Vec<_>, _>>()?;
            backend.crt_recompose(
                &local,
                &plaintext_moduli,
                &reconstruction_coefficients,
                &destination,
            )
        }
    }
    pub(super) fn binary_column_runner(
        left: &GpuFleetMatrix,
        right: &GpuFleetMatrix,
        operation: impl Fn(
            &mut DeviceBackend,
            &GpuDCRTPolyMatrix,
            &GpuDCRTPolyMatrix,
        ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
        + Send
        + Sync
        + 'static,
    ) -> impl Fn(
        usize,
        &mut DeviceBackend,
        usize,
        usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
    + Send
    + Sync
    + 'static {
        let (input_left, input_right) = (left.clone(), right.clone());
        move |_, backend, start, end| {
            let left = Self::matrix_operand_on_device(backend, &input_left, start, end)?;
            let right = Self::matrix_operand_on_device(backend, &input_right, start, end)?;
            operation(backend, &left, &right)
        }
    }
    pub(super) fn multiply_column_runner(
        &mut self,
        scalable: &GpuFleetMatrix,
        fixed: &GpuFleetMatrix,
        scales_left: bool,
    ) -> Result<
        impl Fn(usize, &mut DeviceBackend, usize, usize) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
        + Send
        + Sync
        + 'static
        + use<>,
        PolyBackendError,
    > {
        let replicas = self
            .devices
            .par_iter_mut()
            .map(|(_, backend)| Self::matrix_operand_on_device(backend, fixed, 0, fixed.columns))
            .collect::<Result<Vec<_>, _>>()?;
        if self.runtime_pilot_is_pending() {
            scalable.wait_until_ready();
            replicas.iter().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let input = scalable.clone();
        Ok(move |device: usize, backend: &mut DeviceBackend, start, end| {
            let piece = Self::matrix_operand_on_device(backend, &input, start, end)?;
            if scales_left {
                backend.multiply(piece.as_ref(), replicas[device].as_ref())
            } else {
                backend.multiply(replicas[device].as_ref(), piece.as_ref())
            }
        })
    }

    pub(super) fn constant_column_runner(
        &self,
        ty: &ConcreteMatrixType,
        value: &ConstantMatrix,
        env: &ParamEnv,
    ) -> Result<Option<Arc<ColumnRunner<GpuDCRTPolyMatrix>>>, PolyBackendError> {
        #[derive(Clone, Copy)]
        enum RangeConstant {
            Zero,
            Identity,
            UnitRow(usize),
            Gadget(bool),
            SingleColumn,
        }
        let range_constant = match value {
            ConstantMatrix::Zero => Some(RangeConstant::Zero),
            ConstantMatrix::Identity if ty.rows == ty.columns => Some(RangeConstant::Identity),
            ConstantMatrix::UnitRow { index } if ty.rows == 1 => Some(RangeConstant::UnitRow(
                index
                    .evaluate(env)
                    .ok()
                    .and_then(|value| value.to_usize())
                    .filter(|index| *index < ty.columns)
                    .ok_or(PolyBackendError::InvalidInteger)?,
            )),
            ConstantMatrix::Gadget { base, small } => {
                if ty.rows == 0 || !ty.columns.is_multiple_of(ty.rows) {
                    return Err(PolyBackendError::InvalidInteger);
                }
                let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                self.validate_gadget_layout(ty, &base, ty.columns / ty.rows, *small)?;
                Some(RangeConstant::Gadget(*small))
            }
            _ if ty.columns == 1 &&
                (self.pending_pilot.is_some() ||
                    self.active_operation.is_some_and(|operation| {
                        self.operation_widths.contains_key(&operation)
                    })) =>
            {
                Some(RangeConstant::SingleColumn)
            }
            _ => None,
        };
        let Some(range_constant) = range_constant else {
            return Ok(None);
        };
        let value = value.clone();
        let env = env.clone();
        let ty = ty.clone();
        Ok(Some(Arc::new(move |_, backend, start, end| {
            let local_ty = ConcreteMatrixType { columns: end - start, ..ty.clone() };
            let params = backend.parameters(&local_ty)?;
            let value = match range_constant {
                RangeConstant::Zero => GpuDCRTPolyMatrix::zero(params, ty.rows, end - start),
                RangeConstant::Identity => {
                    GpuDCRTPolyMatrix::identity_columns(params, ty.rows, start, end - start)
                }
                RangeConstant::UnitRow(index) => GpuDCRTPolyMatrix::unit_row_columns(
                    params,
                    ty.columns,
                    index,
                    start,
                    end - start,
                ),
                RangeConstant::Gadget(small) => GpuDCRTPolyMatrix::gadget_columns(
                    params,
                    ty.rows,
                    small,
                    start,
                    end - start,
                    Some(ty.columns / ty.rows),
                ),
                RangeConstant::SingleColumn => backend.constant_matrix(&local_ty, &value, &env)?,
            };
            Ok(value)
        })))
    }

    pub(super) fn accumulate_column_runner(
        &mut self,
        request: &MatrixMulAccumulateRequest<GpuFleetMatrix>,
        output_rows: usize,
        output_columns: usize,
    ) -> Result<Arc<ColumnRunner<GpuDCRTPolyMatrix>>, PolyBackendError> {
        let mut fixed_replicas = Vec::with_capacity(request.products.len());
        for (_, left, right) in &request.products {
            let scales_left =
                gpu_matrix_multiply_scales_left(left.rows, left.columns, right.rows, right.columns);
            let product_rows = if left.size() == (1, 1) { right.rows } else { left.rows };
            let product_columns = if scales_left { left.columns } else { right.columns };
            if (product_rows, product_columns) != (output_rows, output_columns) {
                return Err(PolyBackendError::InvalidInteger);
            }
            let fixed = if scales_left { right } else { left };
            fixed_replicas.push(self.full_matrix_replicas(fixed)?);
        }
        if self.runtime_pilot_is_pending() {
            request.products.iter().for_each(|(_, left, right)| {
                let scalable = if gpu_matrix_multiply_scales_left(
                    left.rows,
                    left.columns,
                    right.rows,
                    right.columns,
                ) {
                    left
                } else {
                    right
                };
                scalable.wait_until_ready();
            });
            if let Some(bias) = &request.bias {
                bias.wait_until_ready();
            }
            fixed_replicas.iter().flatten().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let owned_request = request.clone();
        Ok(Arc::new(move |device, backend, start, end| {
            let request = &owned_request;
            let products = request
                .products
                .iter()
                .zip(fixed_replicas.iter())
                .map(|(product, replicas)| {
                    let scales_left = gpu_matrix_multiply_scales_left(
                        product.1.rows,
                        product.1.columns,
                        product.2.rows,
                        product.2.columns,
                    );
                    let scalable = if scales_left { &product.1 } else { &product.2 };
                    let piece =
                        Arc::new(Self::matrix_piece_on_device(backend, scalable, start, end)?);
                    let (left, right) = if scales_left {
                        (piece, replicas[device].clone())
                    } else {
                        (replicas[device].clone(), piece)
                    };
                    Ok((product.0.clone(), left, right))
                })
                .collect::<Result<Vec<_>, PolyBackendError>>()?;
            let bias = request
                .bias
                .as_ref()
                .map(|bias| Self::matrix_piece_on_device(backend, bias, start, end).map(Arc::new))
                .transpose()?;
            backend.matrix_mul_accumulate(MatrixMulAccumulateRequest { products, bias })
        }))
    }
    pub(super) fn slice_column_runner(
        value: &GpuFleetMatrix,
        row_range: IndexRange,
    ) -> Arc<ColumnRunner<GpuDCRTPolyMatrix>> {
        let input = value.clone();
        Arc::new(move |_, backend, start, end| {
            let piece = Self::matrix_operand_on_device(backend, &input, start, end)?;
            Ok(piece.slice_rows(row_range.start, row_range.end))
        })
    }
    pub(super) fn small_rhs_column_replicas(
        &mut self,
        lhs: &GpuFleetMatrix,
        rhs: &GpuFleetSmallMatrix,
    ) -> Result<Arc<Vec<GpuMatrixOperand>>, PolyBackendError> {
        let lhs_replicas = self
            .devices
            .par_iter_mut()
            .map(|(_, backend)| Self::matrix_operand_on_device(backend, lhs, 0, lhs.columns))
            .collect::<Result<Vec<_>, _>>()?;
        let lhs_replicas = Arc::new(lhs_replicas);
        let pilot_rhs = if self.pending_pilot.is_some() {
            Some(self.compact_pilot_columns(rhs)?)
        } else {
            None
        };
        if self.runtime_pilot_is_pending() {
            rhs.wait_until_ready();
            lhs_replicas.iter().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        if let Some(pilot_rhs) = pilot_rhs {
            let pilot_rhs = Arc::new(pilot_rhs);
            loop {
                let rhs = pilot_rhs.clone();
                let lhs = lhs_replicas.clone();
                let launched = self
                    .enqueue
                    .map(&mut self.devices, move |device, (device_id, backend)| {
                        let Some(rhs) = rhs.get(device) else {
                            return Ok(None);
                        };
                        backend.multiply_small_rhs(&lhs[device], rhs).map(|value| {
                            Some(GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: 0,
                                value,
                            })
                        })
                    })
                    .map_err(PolyBackendError::from)?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                let completed = self.finish_runtime_pilot(&launched)?;
                drop(launched);
                if completed {
                    break;
                }
                self.restart_runtime_pilot_after_fixed_inputs()
                    .map_err(PolyBackendError::GpuCalibration)?;
            }
        }
        Ok(lhs_replicas)
    }
    pub(super) fn concat_column_runner(
        inputs: &[&GpuFleetMatrix],
        axis: ConcatAxis,
    ) -> Result<(usize, usize, Arc<ColumnRunner<GpuDCRTPolyMatrix>>), PolyBackendError> {
        let first = *inputs.first().ok_or(PolyBackendError::InvalidConstantShape)?;
        if inputs.iter().any(|input| match axis {
            ConcatAxis::Rows => input.columns != first.columns,
            ConcatAxis::Columns => input.rows != first.rows,
            ConcatAxis::Diagonal => false,
        }) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let rows = if axis == ConcatAxis::Columns {
            first.rows
        } else {
            inputs.iter().try_fold(0usize, |rows, input| {
                rows.checked_add(input.rows).ok_or(PolyBackendError::InvalidInteger)
            })?
        };
        let mut columns = 0usize;
        let mut offsets = Vec::with_capacity(inputs.len());
        if axis == ConcatAxis::Rows {
            columns = first.columns;
        } else {
            for input in inputs {
                offsets.push(columns);
                columns =
                    columns.checked_add(input.columns).ok_or(PolyBackendError::InvalidInteger)?;
            }
        }
        let prototype = inputs.iter().find_map(|input| input.shards.first()).map(|shard| {
            (
                BigInt::from(shard.value.params().modulus().as_ref().clone()),
                shard.value.params().ring_dimension() as usize,
            )
        });
        let input_columns = inputs.iter().map(|input| input.columns).collect::<Vec<_>>();
        let owned_inputs = inputs.par_iter().map(|input| (*input).clone()).collect::<Vec<_>>();
        let runner =
            Arc::new(move |_, backend: &mut DeviceBackend, start: usize, end: usize| match axis {
                ConcatAxis::Rows => {
                    let local = owned_inputs
                        .iter()
                        .map(|input| Self::matrix_operand_on_device(backend, input, start, end))
                        .collect::<Result<Vec<_>, _>>()?;
                    backend.concat(
                        &local.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                        ConcatAxis::Rows,
                    )
                }
                ConcatAxis::Columns => {
                    let mut pieces = Vec::new();
                    let first_input = offsets.partition_point(|offset| *offset <= start) - 1;
                    for (input, offset) in
                        owned_inputs[first_input..].iter().zip(&offsets[first_input..])
                    {
                        let overlap_start = start.max(*offset);
                        let overlap_end = end.min(*offset + input.columns);
                        if overlap_start < overlap_end {
                            pieces.push(Self::matrix_piece_on_device(
                                backend,
                                input,
                                overlap_start - offset,
                                overlap_end - offset,
                            )?);
                        }
                        if *offset + input.columns >= end {
                            break;
                        }
                    }
                    let mut pieces = pieces.into_iter();
                    let first = pieces.next().ok_or(PolyBackendError::InvalidConstantShape)?;
                    Ok(first.concat_columns_owned(pieces.collect()))
                }
                ConcatAxis::Diagonal => {
                    let (modulus, ring_dimension) =
                        prototype.as_ref().ok_or(PolyBackendError::InvalidConstantShape)?;
                    Self::diagonal_range_on_device(
                        backend,
                        &owned_inputs.iter().collect::<Vec<_>>(),
                        &input_columns,
                        rows,
                        modulus,
                        *ring_dimension,
                        start,
                        end,
                    )
                }
            });
        Ok((rows, columns, runner))
    }

    pub(super) fn compact_pilot_columns(
        &self,
        rhs: &GpuFleetSmallMatrix,
    ) -> Result<Vec<GpuSmallMatrix>, PolyBackendError> {
        let source = rhs.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
        let local = source.value.slice_columns(0, 1);
        let payload = local.to_canonical_coefficients()?;
        let params = local.params();
        let local_type = ConcreteMatrixType {
            modulus: BigInt::from(params.modulus().as_ref().clone()),
            ring_dimension: params.ring_dimension() as usize,
            rows: rhs.rows,
            columns: 1,
        };
        Ok(self
            .devices
            .iter()
            .take(2)
            .map(|(_, backend)| {
                let target_params = backend.parameters(&local_type)?;
                GpuSmallMatrix::from_canonical_coefficients(
                    target_params,
                    rhs.rows,
                    1,
                    local.max_coefficient_bound().clone(),
                    &payload,
                )
                .map_err(PolyBackendError::from)
            })
            .collect::<Result<Vec<_>, _>>()?)
    }

    pub(super) fn small_rhs_block_replicas(
        &mut self,
        blocks: &[&GpuFleetMatrix],
        rhs: &GpuFleetSmallMatrix,
    ) -> Result<Arc<Vec<Vec<GpuMatrixOperand>>>, PolyBackendError> {
        if blocks.is_empty() ||
            blocks.len() > 32 ||
            blocks.iter().any(|block| block.columns != rhs.rows)
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let replicas = Arc::new(
            self.devices
                .par_iter_mut()
                .map(|(_, backend)| {
                    blocks
                        .iter()
                        .map(|block| {
                            Self::matrix_operand_on_device(backend, block, 0, block.columns)
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let pilots = if self.runtime_pilot_is_pending() && rhs.columns > 0 {
            let pilots = Arc::new(self.compact_pilot_columns(rhs)?);
            rhs.wait_until_ready();
            replicas.iter().flatten().for_each(|value| value.wait_until_ready());
            Some(pilots)
        } else {
            None
        };
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        if let Some(pilots) = pilots {
            let inputs = replicas.clone();
            self.calibrate_column_waves(rhs.columns, move |fleet, wave| {
                let inputs = inputs.clone();
                let pilots = pilots.clone();
                fleet.launch_column_wave(wave, move |device, backend, _, _| {
                    let references = inputs[device].iter().map(AsRef::as_ref).collect::<Vec<_>>();
                    backend.multiply_small_rhs_row_blocks(&references, &pilots[device])
                })
            })?;
        } else if rhs.columns == 0 {
            self.pending_pilot = None;
            self.pending_profile = None;
        }
        Ok(replicas)
    }

    pub(super) fn preflight_column_operations(
        &mut self,
        requests: &[(
            usize,
            GpuInvocation<'_, GpuFleetMatrix, GpuFleetSmallMatrix, GpuFleetTrapdoor>,
        )],
    ) -> Result<(), PolyBackendError> {
        // Validate the complete batch before running even an isolated pilot.
        if requests.iter().any(|(placement, _)| *placement != 0) {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        for (_, request) in requests {
            if !self.runtime_pilot_is_pending() {
                continue;
            }
            let unary = match request {
                GpuInvocation::ModulusSwitch { value, destination } => {
                    let destination = (*destination).clone();
                    Some((
                        *value,
                        GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                            backend.modulus_switch(input, &destination)
                        })),
                    ))
                }
                GpuInvocation::CenteredRebase { value, destination } => {
                    let destination = (*destination).clone();
                    Some((
                        *value,
                        GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                            backend.centered_rebase(input, &destination)
                        })),
                    ))
                }
                GpuInvocation::ReduceModulus { value, destination } => {
                    let destination = (*destination).clone();
                    Some((
                        *value,
                        GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                            backend.reduce_modulus(input, &destination)
                        })),
                    ))
                }
                GpuInvocation::CenteredExtend { value, destination } => {
                    let destination = (*destination).clone();
                    Some((
                        *value,
                        GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                            backend.centered_extend(input, &destination)
                        })),
                    ))
                }
                GpuInvocation::RnsModUp {
                    value,
                    destination,
                    source_moduli,
                    digit_size,
                    normalize,
                } => {
                    let destination = (*destination).clone();
                    let moduli = source_moduli.to_vec();
                    let (digit_size, normalize) = (*digit_size, *normalize);
                    Some((
                        *value,
                        GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                            backend.rns_mod_up(input, &destination, &moduli, digit_size, normalize)
                        })),
                    ))
                }
                GpuInvocation::RnsModDown {
                    value,
                    destination,
                    source_moduli,
                    plaintext_modulus,
                } => {
                    let destination = (*destination).clone();
                    let moduli = source_moduli.to_vec();
                    let plaintext_modulus = *plaintext_modulus;
                    Some((
                        *value,
                        GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                            backend.rns_mod_down(input, &destination, &moduli, plaintext_modulus)
                        })),
                    ))
                }
                GpuInvocation::BlockModSwitch { value, destination, plaintext_modulus } => {
                    let destination = (*destination).clone();
                    let plaintext_modulus = *plaintext_modulus;
                    Some((
                        *value,
                        GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                            backend.block_mod_switch(input, &destination, plaintext_modulus)
                        })),
                    ))
                }
                GpuInvocation::SumRows { value, rows } => {
                    let rows = rows.to_vec();
                    Some((
                        *value,
                        GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                            backend.sum_rows(input, &rows)
                        })),
                    ))
                }
                _ => None,
            };
            if let Some((value, operation)) = unary {
                self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
                self.calibrate_column_operation(
                    value.columns,
                    Arc::new(Self::unary_column_runner(value, operation)),
                )?;
                continue;
            }
            match request {
                GpuInvocation::Concat { inputs, axis } => {
                    let (_, columns, runner) = Self::concat_column_runner(inputs, *axis)?;
                    self.restart_runtime_pilot_after_matrix_inputs(inputs)?;
                    self.calibrate_column_operation(columns, runner)?;
                }
                GpuInvocation::Accumulate { request } => {
                    let first =
                        request.products.first().ok_or(PolyBackendError::InvalidConstantShape)?;
                    let scales_left = gpu_matrix_multiply_scales_left(
                        first.1.rows,
                        first.1.columns,
                        first.2.rows,
                        first.2.columns,
                    );
                    let columns = if scales_left { first.1.columns } else { first.2.columns };
                    let rows = if first.1.size() == (1, 1) { first.2.rows } else { first.1.rows };
                    let runner = self.accumulate_column_runner(request, rows, columns)?;
                    self.calibrate_column_operation(columns, runner)?;
                }
                GpuInvocation::Slice { value, rows, columns } => {
                    let row_range =
                        rows.cloned().unwrap_or(IndexRange { start: 0, end: value.rows });
                    let column_range =
                        columns.cloned().unwrap_or(IndexRange { start: 0, end: value.columns });
                    if row_range.start > row_range.end ||
                        row_range.end > value.rows ||
                        column_range.start > column_range.end ||
                        column_range.end > value.columns
                    {
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    let runner = Self::slice_column_runner(value, row_range);
                    self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
                    self.calibrate_column_operation(
                        column_range.end - column_range.start,
                        Arc::new(move |device, backend, start, end| {
                            runner(
                                device,
                                backend,
                                column_range.start + start,
                                column_range.start + end,
                            )
                        }),
                    )?;
                }
                GpuInvocation::MultiplySmallRhsRowBlocks { blocks, right } => {
                    self.small_rhs_block_replicas(blocks, right)?;
                }
                GpuInvocation::MultiplySmallRhs { left, right } => {
                    self.small_rhs_column_replicas(left, right)?;
                }
                GpuInvocation::Constant { ty, value, env } => {
                    if let Some(runner) = self.constant_column_runner(ty, value, env)? {
                        self.restart_runtime_pilot_after_fixed_inputs()
                            .map_err(PolyBackendError::GpuCalibration)?;
                        self.calibrate_column_operation(ty.columns, runner)?;
                    }
                }
                GpuInvocation::Binary { operation, left, right } => match operation {
                    MatrixBinaryOp::Add | MatrixBinaryOp::Subtract => {
                        if left.size() != right.size() {
                            return Err(PolyBackendError::InvalidConstantShape);
                        }
                        let operation = *operation;
                        let runner =
                            Self::binary_column_runner(left, right, move |backend, left, right| {
                                match operation {
                                    MatrixBinaryOp::Add => backend.add(left, right),
                                    MatrixBinaryOp::Subtract => backend.sub(left, right),
                                    MatrixBinaryOp::Multiply => unreachable!(),
                                }
                            });
                        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
                        self.calibrate_column_operation(left.columns, Arc::new(runner))?;
                    }
                    MatrixBinaryOp::Multiply => {
                        if left.size() != (1, 1) &&
                            right.size() != (1, 1) &&
                            left.columns != right.rows
                        {
                            return Err(PolyBackendError::InvalidConstantShape);
                        }
                        let scales_left = gpu_matrix_multiply_scales_left(
                            left.rows,
                            left.columns,
                            right.rows,
                            right.columns,
                        );
                        let (scalable, fixed) =
                            if scales_left { (left, right) } else { (right, left) };
                        let runner = self.multiply_column_runner(scalable, fixed, scales_left)?;
                        self.calibrate_column_operation(scalable.columns, Arc::new(runner))?;
                    }
                },
                GpuInvocation::Transpose { value } => {
                    if value.rows > 0 && value.columns == 0 && value.shards.is_empty() {
                        return Err(PolyBackendError::GpuSubmission(
                            "cannot transpose a shardless empty matrix without parameter metadata"
                                .into(),
                        ));
                    }
                    self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
                    self.calibrate_column_operation(
                        value.rows,
                        Arc::new(Self::transpose_column_runner(value)),
                    )?;
                }
                GpuInvocation::Tensor { left, right } => {
                    left.rows.checked_mul(right.rows).ok_or(PolyBackendError::InvalidInteger)?;
                    let columns = left
                        .columns
                        .checked_mul(right.columns)
                        .ok_or(PolyBackendError::InvalidInteger)?;
                    self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
                    self.calibrate_column_operation(
                        columns,
                        Arc::new(Self::tensor_column_runner(left, right, columns)),
                    )?;
                }
                GpuInvocation::TensorSumRows { left, right, rows } => {
                    let columns = left
                        .columns
                        .checked_mul(right.columns)
                        .ok_or(PolyBackendError::InvalidInteger)?;
                    self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
                    self.calibrate_column_operation(
                        columns,
                        Arc::new(Self::tensor_sum_rows_column_runner(left, right, rows, columns)),
                    )?;
                }
                GpuInvocation::GadgetDecompose { value, small, digit_count } => {
                    self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
                    self.calibrate_column_operation(
                        value.columns,
                        Arc::new(Self::gadget_decompose_row_blocks_column_runner(
                            &[value],
                            *small,
                            *digit_count,
                        )),
                    )?;
                }
                GpuInvocation::GadgetDecomposeRowBlocks { blocks, small, digit_count } => {
                    let first = blocks.first().ok_or(PolyBackendError::InvalidConstantShape)?;
                    if blocks.iter().any(|block| block.columns != first.columns) {
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    self.restart_runtime_pilot_after_matrix_inputs(blocks)?;
                    self.calibrate_column_operation(
                        first.columns,
                        Arc::new(Self::gadget_decompose_row_blocks_column_runner(
                            blocks,
                            *small,
                            *digit_count,
                        )),
                    )?;
                }
                GpuInvocation::AddRowBlocks { blocks, right } => {
                    let rows = blocks.iter().try_fold(0usize, |rows, block| {
                        rows.checked_add(block.rows).ok_or(PolyBackendError::InvalidInteger)
                    })?;
                    if blocks.is_empty() ||
                        rows != right.rows ||
                        blocks.iter().any(|block| block.columns != right.columns)
                    {
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    let inputs =
                        blocks.iter().copied().chain(std::iter::once(*right)).collect::<Vec<_>>();
                    self.restart_runtime_pilot_after_matrix_inputs(&inputs)?;
                    self.calibrate_column_operation(
                        right.columns,
                        Arc::new(Self::add_row_blocks_column_runner(blocks, right)),
                    )?;
                }
                GpuInvocation::CrtRecompose {
                    levels,
                    plaintext_moduli,
                    reconstruction_coefficients,
                    destination,
                } => {
                    let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
                    if levels.iter().any(|level| level.size() != first.size()) {
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    self.restart_runtime_pilot_after_matrix_inputs(
                        &levels.iter().collect::<Vec<_>>(),
                    )?;
                    self.calibrate_column_operation(
                        first.columns,
                        Arc::new(Self::crt_recompose_column_runner(
                            levels,
                            plaintext_moduli,
                            reconstruction_coefficients,
                            destination,
                        )),
                    )?;
                }
                GpuInvocation::SampleUniform { ty, range } => {
                    self.restart_runtime_pilot_after_fixed_inputs()
                        .map_err(PolyBackendError::GpuCalibration)?;
                    self.calibrate_column_operation(
                        ty.columns,
                        Arc::new(Self::uniform_column_runner(ty, range)),
                    )?;
                }
                GpuInvocation::SampleGaussian { ty, sigma, max_coefficient_bound } => {
                    self.restart_runtime_pilot_after_fixed_inputs()
                        .map_err(PolyBackendError::GpuCalibration)?;
                    self.calibrate_column_operation(
                        ty.columns,
                        Arc::new(Self::gaussian_column_runner(ty, *sigma, max_coefficient_bound)),
                    )?;
                }
                GpuInvocation::SampleHash { ty, variant, tag_bytes, gadget_base, digit_count } => {
                    // The request contains neither the production key nor its tag.
                    // Retain the exact launch layout and length with independent data.
                    let key = rand::random();
                    let tag = vec![0; *tag_bytes];
                    match (variant, gadget_base, digit_count) {
                        (HashVariant::Plain, None, None) => {
                            self.restart_runtime_pilot_after_fixed_inputs()
                                .map_err(PolyBackendError::GpuCalibration)?;
                            self.calibrate_column_operation(
                                ty.columns,
                                Arc::new(Self::hash_column_runner(ty, key, &tag)),
                            )?;
                        }
                        (
                            HashVariant::Decomposed | HashVariant::SmallDecomposed,
                            Some(base),
                            Some(count),
                        ) => {
                            let small = *variant == HashVariant::SmallDecomposed;
                            self.validate_gadget_layout(ty, base, *count, small)?;
                            let runner =
                                Self::decomposed_hash_column_runner(ty, key, &tag, *count, small)?;
                            self.restart_runtime_pilot_after_fixed_inputs()
                                .map_err(PolyBackendError::GpuCalibration)?;
                            self.calibrate_column_operation(ty.columns, Arc::new(runner))?;
                        }
                        _ => return Err(PolyBackendError::InvalidInteger),
                    }
                }
                GpuInvocation::SamplePreimage {
                    schema,
                    sigma,
                    gadget_base,
                    digit_count,
                    trapdoor,
                    public,
                    target,
                } => {
                    let GpuColumnSourceLayout::RnsStaging {
                        matrix_type,
                        global_column_start,
                        level,
                        is_ntt,
                        bytes_per_poly,
                    } = target
                    else {
                        return Err(PolyBackendError::GpuCalibration(
                            "GPU preimage calibration requires a native staging layout".into(),
                        ));
                    };
                    if matrix_type.columns != schema.matrix.columns ||
                        matrix_type.modulus != schema.matrix.modulus ||
                        matrix_type.ring_dimension != schema.matrix.ring_dimension ||
                        global_column_start.checked_add(matrix_type.columns).is_none()
                    {
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    let parameters = self.devices[0].1.parameters(matrix_type)?;
                    let layout = GpuCpuStagingLayout {
                        rows: matrix_type.rows,
                        columns: usize::from(matrix_type.columns != 0),
                        level: *level,
                        is_ntt: *is_ntt,
                        bytes_per_poly: *bytes_per_poly,
                    };
                    let bytes =
                        layout.zero_bytes(parameters).map_err(PolyBackendError::GpuCalibration)?;
                    let source = IsolatedPreimageSource {
                        rows: matrix_type.rows,
                        columns: matrix_type.columns,
                        global_column_start: *global_column_start,
                        data: PolyMatrixColumnData::staged(
                            parameters,
                            Arc::new(bytes),
                            0,
                            layout.columns,
                        ),
                    };
                    let runner = self.preimage_column_runner(
                        &schema.matrix,
                        *sigma,
                        gadget_base,
                        *digit_count,
                        &schema.max_coefficient_bound,
                        trapdoor,
                        public,
                        &source,
                        rand::random(),
                    )?;
                    self.calibrate_column_waves(source.columns, runner)?;
                }
                GpuInvocation::Negate { value } |
                GpuInvocation::ScaleInteger { value, .. } |
                GpuInvocation::RingAutomorphism { value, .. } => {
                    let operation = match request {
                        GpuInvocation::Negate { .. } => GpuUnaryColumnOperation::Negate,
                        GpuInvocation::ScaleInteger { scalar, .. } => {
                            GpuUnaryColumnOperation::Scale((*scalar).clone())
                        }
                        GpuInvocation::RingAutomorphism { index, .. } => {
                            GpuUnaryColumnOperation::Automorphism(*index)
                        }
                        _ => unreachable!(),
                    };
                    self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
                    self.calibrate_column_operation(
                        value.columns,
                        Arc::new(Self::unary_column_runner(value, operation)),
                    )?;
                }
                // Other classes retain their existing implementation until their
                // range preparation and complete allocation inventories are connected.
                _ => {}
            }
        }
        Ok(())
    }
}

/// A one-column representative with the original target's global coordinates.
/// The production loader is absent; each active role gets an independent load
/// of these immutable host bytes on its own execution owner.
#[derive(Debug)]
struct IsolatedPreimageSource {
    rows: usize,
    columns: usize,
    global_column_start: usize,
    data: PolyMatrixColumnData<GpuFleetMatrix>,
}

impl PolyMatrixColumnSource<GpuFleetMatrix> for IsolatedPreimageSource {
    fn row_size(&self) -> usize {
        self.rows
    }
    fn col_size(&self) -> usize {
        self.columns
    }
    fn global_column_start(&self) -> usize {
        self.global_column_start
    }
    fn column_range(&self, start: usize, end: usize) -> PolyMatrixColumnData<GpuFleetMatrix> {
        assert!(
            start <= end && end <= self.columns && end - start <= 1,
            "isolated preimage pilot exceeds its prepared representative"
        );
        self.data.subrange(0, end - start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::{
        poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
        sampler::PolyTrapdoorSampler,
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_arithmetic_and_compact_blocks_match_primitives() {
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse().expect("ring dimension"))
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse().expect("columns"))
            .unwrap_or(3);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let ty = ConcreteMatrixType {
            modulus: params.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 2,
            columns,
        };
        let raw =
            backend.devices[0].1.sample_hash(&ty, rand::random(), b"arithmetic-preflight").unwrap();
        let left = GpuFleetMatrix::from_matrix(raw.clone());
        let right_raw = raw.transpose();
        let right = GpuFleetMatrix::from_matrix(right_raw.clone());
        let check =
            |backend: &mut GpuDcrtBackend,
             invocation: GpuInvocation<
                '_,
                GpuFleetMatrix,
                GpuFleetSmallMatrix,
                GpuFleetTrapdoor,
            >,
             execute: &dyn Fn(&mut GpuDcrtBackend) -> Result<GpuFleetMatrix, PolyBackendError>,
             expected: GpuDCRTPolyMatrix| {
                let operation = rand::random();
                backend.select_operation(operation, true).unwrap();
                backend.preflight_gpu_operations(&[(0, invocation)]).unwrap();
                assert!(backend.pending_pilot.is_none(), "preflight must finish before production");
                assert!(backend.operation_profiles.contains_key(&operation));
                let actual = execute(backend).unwrap();
                assert_eq!(backend.gather_matrix_for_host(&actual).unwrap(), expected);
            };
        check(
            &mut backend,
            GpuInvocation::Constant {
                ty: &ty,
                value: &ConstantMatrix::Zero,
                env: &ParamEnv::default(),
            },
            &|backend| backend.constant_matrix(&ty, &ConstantMatrix::Zero, &ParamEnv::default()),
            GpuDCRTPolyMatrix::zero(&params, ty.rows, ty.columns),
        );
        check(
            &mut backend,
            GpuInvocation::Binary { operation: MatrixBinaryOp::Add, left: &left, right: &left },
            &|backend| backend.add(&left, &left),
            raw.add_out_of_place(&raw),
        );
        check(
            &mut backend,
            GpuInvocation::Binary {
                operation: MatrixBinaryOp::Multiply,
                left: &left,
                right: &right,
            },
            &|backend| backend.multiply(&left, &right),
            &raw * &right_raw,
        );
        check(
            &mut backend,
            GpuInvocation::Transpose { value: &left },
            &|backend| backend.transpose(&left),
            right_raw.clone(),
        );
        check(
            &mut backend,
            GpuInvocation::Tensor { left: &left, right: &left },
            &|backend| backend.tensor(&left, &left),
            raw.tensor(&raw),
        );
        for axis in [ConcatAxis::Rows, ConcatAxis::Columns, ConcatAxis::Diagonal] {
            let expected = match axis {
                ConcatAxis::Rows => raw.concat_rows(&[&raw]),
                ConcatAxis::Columns => raw.concat_columns(&[&raw]),
                ConcatAxis::Diagonal => raw.concat_diag(&[&raw]),
            };
            check(
                &mut backend,
                GpuInvocation::Concat { inputs: &[&left, &left], axis },
                &|backend| backend.concat(&[&left, &left], axis),
                expected,
            );
        }
        let row_range = IndexRange { start: 0, end: 1 };
        let col_range = IndexRange { start: columns / 2, end: columns };
        check(
            &mut backend,
            GpuInvocation::Slice {
                value: &left,
                rows: Some(&row_range),
                columns: Some(&col_range),
            },
            &|backend| backend.slice(&left, Some(&row_range), Some(&col_range)),
            raw.slice_rows(0, 1).slice_columns(col_range.start, col_range.end),
        );
        let request = MatrixMulAccumulateRequest {
            products: vec![(1.into(), Arc::new(left.clone()), Arc::new(right.clone()))],
            bias: None,
        };
        check(
            &mut backend,
            GpuInvocation::Accumulate { request: &request },
            &|backend| backend.matrix_mul_accumulate(request.clone()),
            &raw * &right_raw,
        );
        backend.select_operation(rand::random(), true).unwrap();
        backend
            .preflight_gpu_operations(&[(
                0,
                GpuInvocation::GadgetDecompose { value: &left, small: false, digit_count: None },
            )])
            .unwrap();
        assert!(backend.pending_pilot.is_none());
        let digits = backend.gadget_decompose(&left, false, None).unwrap();
        let expected_digits = raw.gadget_decompose(false, None).unwrap();
        assert_eq!(digits.shards[0].value, expected_digits);
        let lhs_type = ConcreteMatrixType { columns: digits.rows, ..ty };
        let lhs = backend.devices[0]
            .1
            .sample_hash(&lhs_type, rand::random(), b"compact-preflight-left")
            .unwrap();
        let upper = GpuFleetMatrix::from_matrix(lhs.slice_rows(0, 1));
        let lower = GpuFleetMatrix::from_matrix(lhs.slice_rows(1, 2));
        backend.select_operation(rand::random(), true).unwrap();
        backend
            .preflight_gpu_operations(&[(
                0,
                GpuInvocation::MultiplySmallRhsRowBlocks {
                    blocks: &[&upper, &lower],
                    right: &digits,
                },
            )])
            .unwrap();
        assert!(backend.pending_pilot.is_none());
        let outputs = backend.multiply_small_rhs_row_blocks(&[&upper, &lower], &digits).unwrap();
        let expected = lhs.multiply_small_rhs(&expected_digits).unwrap();
        for (row, output) in outputs.iter().enumerate() {
            assert_eq!(
                backend.gather_matrix_for_host(output).unwrap(),
                expected.slice_rows(row, row + 1)
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_calibrates_before_hash_and_unary_production() {
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse().expect("ring dimension"))
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse().expect("columns"))
            .unwrap_or(5);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let ty = ConcreteMatrixType {
            modulus: params.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 2,
            columns,
        };
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let operation = rand::random();
        let key = rand::random();
        let tag = b"production-key-and-tag-are-not-pilot-inputs";
        backend.select_operation(operation, true).unwrap();
        backend
            .preflight_gpu_operations(&[(
                0,
                GpuInvocation::SampleHash {
                    ty: &ty,
                    variant: HashVariant::Plain,
                    tag_bytes: tag.len(),
                    gadget_base: None,
                    digit_count: None,
                },
            )])
            .unwrap();
        assert!(backend.pending_pilot.is_none());
        let profile = backend.operation_profiles[&operation].clone();
        let output = backend.sample_hash(&ty, key, tag).unwrap();
        assert_eq!(backend.operation_profiles[&operation], profile);
        let expected = backend.devices[0].1.sample_hash(&ty, key, tag).unwrap();
        assert_eq!(backend.gather_matrix_for_host(&output).unwrap(), expected);
        for request in [
            GpuInvocation::Negate { value: &output },
            GpuInvocation::RingAutomorphism { value: &output, index: 3 },
        ] {
            let operation = rand::random();
            backend.select_operation(operation, true).unwrap();
            backend.preflight_gpu_operations(&[(0, request)]).unwrap();
            assert!(backend.pending_pilot.is_none());
            assert!(backend.operation_profiles.contains_key(&operation));
        }
        let actual = backend.ring_automorphism(&output, 3).unwrap();
        assert_eq!(
            backend.gather_matrix_for_host(&actual).unwrap(),
            expected.ring_automorphism_out_of_place(3)
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_preimage_uses_isolated_target_and_preserves_seed() {
        // Pool observations must be isolated from other tests that create native
        // contexts under a different serial-test lock.
        const CHILD: &str = "MXX_GPU_PREFLIGHT_PREIMAGE_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let module = module_path!().split_once("::").unwrap().1;
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg(format!(
                    "{module}::test_gpu_preflight_preimage_uses_isolated_target_and_preserves_seed"
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
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse().expect("ring dimension"))
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse().expect("columns"))
            .unwrap_or(2);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, 4.578);
        let (trapdoor, public) = sampler.trapdoor(&params, 1);
        let public = GpuFleetMatrix::from_matrix(public);
        let trapdoor = GpuFleetTrapdoor { values: Arc::new(vec![trapdoor]) };
        let target_type = ConcreteMatrixType {
            modulus: params.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 1,
            columns,
        };
        let target = backend.devices[0]
            .1
            .sample_hash(&target_type, rand::random(), b"actual-preimage-target")
            .unwrap();
        let expected = target.clone();
        let bytes = target.to_cpu_staging_bytes();
        let layout = backend.gpu_preimage_source_layout(&target_type, &bytes, 0).unwrap();
        let source = PreimageTarget::<GpuFleetMatrix>::staged(&params, 1, columns, Arc::new(bytes));
        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType { rows: public.columns, ..target_type },
            max_coefficient_bound: BigInt::from(params.modulus().as_ref() >> 1u8),
        };
        let base = BigInt::from(1u64 << params.base_bits());
        let digits = params.modulus_digits();
        let operation = rand::random();
        backend.select_operation(operation, true).unwrap();
        backend
            .preflight_gpu_operations(&[(
                0,
                GpuInvocation::SamplePreimage {
                    schema: &schema,
                    sigma: 4.578,
                    gadget_base: &base,
                    digit_count: digits,
                    trapdoor: &trapdoor,
                    public: &public,
                    target: &layout,
                },
            )])
            .unwrap();
        assert!(backend.pending_pilot.is_none());
        assert!(backend.operation_profiles.contains_key(&operation));
        let seed = rand::random();
        let actual = backend
            .sample_preimage(
                &schema.matrix,
                4.578,
                &base,
                digits,
                &schema.max_coefficient_bound,
                &trapdoor,
                &public,
                &source,
                seed,
            )
            .unwrap();
        let reference = sampler
            .preimage(
                &params,
                &trapdoor.values[0],
                &public.shards[0].value,
                &mxx_primitives::matrix::ResidentPolyMatrixColumnSource::new(expected.clone()),
                schema.max_coefficient_bound.to_biguint().unwrap(),
                seed,
            )
            .unwrap();
        assert_eq!(actual.shards.len(), 1);
        assert_eq!(actual.shards[0].value, reference);
        assert_eq!(
            public.shards[0].value.multiply_small_rhs(&actual.shards[0].value).unwrap(),
            expected
        );
    }
}
