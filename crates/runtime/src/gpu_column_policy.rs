//! Pure column-capability and range-lowering helpers.
//!
//! This module deliberately contains no allocator, device query, profiling, or
//! backend state.  It is the single source of truth for translating a planned
//! output column range into the input ranges needed by an operation.

use mxx_ir_core::{
    expr::IntExpr,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

/// The coarse resource capability of an effective operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ColumnCapability {
    SameColumns,
    FixedOperandColumns,
    GeneratedColumns,
    MappedColumns,
    SingleDevice,
    HostOrControl,
    /// A future or otherwise unlisted operation. It must be rejected during
    /// planning instead of silently taking a host/GPU-0 fallback.
    Unsupported,
}

/// Explicit allowlist of operations understood by the column planner. The
/// classifier is intentionally separate from [`ColumnCapability`]: two
/// operations can share a capability while still having different range
/// lowering semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EffectiveGpuOperation {
    GeneratedConstant,
    SingleDeviceConstant,
    UniformResidueSample,
    UniformIntervalSample,
    GaussianSample,
    HashSample,
    TrapdoorSample,
    GadgetTrapdoor,
    TrapdoorPublic,
    PreimageSample,
    GadgetDecompose,
    MatrixScale,
    MatrixNegate,
    RingAutomorphism,
    ModulusSwitch,
    ModulusReduce,
    CenteredRebase,
    RnsModUp,
    RnsModDown,
    CrtRecompose,
    MatrixAdd,
    MatrixSubtract,
    MatrixMultiply,
    MatrixMulAccumulate,
    MatrixMulSmallRhs,
    Transpose,
    Slice,
    Tensor,
    ConcatRows,
    ConcatColumns,
    ConcatDiagonal,
    HostOrControl,
    Unsupported,
}

/// A half-open global column range.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ColumnRange {
    pub start: usize,
    pub end: usize,
}

impl ColumnRange {
    pub const fn empty(at: usize) -> Self {
        Self { start: at, end: at }
    }

    pub const fn new(start: usize, end: usize) -> Option<Self> {
        if start <= end { Some(Self { start, end }) } else { None }
    }

    pub const fn len(self) -> usize {
        self.end - self.start
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    fn checked_translate(self, offset: usize) -> Option<Self> {
        Some(Self { start: self.start.checked_add(offset)?, end: self.end.checked_add(offset)? })
    }
}

/// A lowered read range. `operand` is the operation argument index, not a
/// device or an instance index.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputColumnRange {
    pub operand: usize,
    pub range: ColumnRange,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ColumnPolicyError {
    #[error("output column range [{start}, {end}) is outside the output width {columns}")]
    InvalidOutputRange { start: usize, end: usize, columns: usize },
    #[error("operation argument {operand} is not a matrix-like value")]
    NonMatrixOperand { operand: usize },
    #[error("operation requires input argument {operand}")]
    MissingOperand { operand: usize },
    #[error("column mapping for {operation} is not supported")]
    UnsupportedOperation { operation: &'static str },
    #[error("column range arithmetic overflow")]
    ArithmeticOverflow,
}

fn matrix_type(ty: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
    match ty {
        ConcreteWireType::IndexedFamily { element, .. } => matrix_type(element),
        _ => ty.matrix_type(),
    }
}

fn matrix_at<'a>(
    arguments: &'a [ConcreteWireType],
    operand: usize,
) -> Result<&'a ConcreteMatrixType, ColumnPolicyError> {
    arguments
        .get(operand)
        .ok_or(ColumnPolicyError::MissingOperand { operand })
        .and_then(|ty| matrix_type(ty).ok_or(ColumnPolicyError::NonMatrixOperand { operand }))
}

/// A preimage job retains the complete public/trapdoor pair and slices only
/// the target. The same mapper is used before staging and in warmup.
pub fn preimage_input_ranges(
    arguments: &[ConcreteWireType],
    output: ColumnRange,
) -> Result<Vec<InputColumnRange>, ColumnPolicyError> {
    let public = matrix_at(arguments, 0)?;
    let trapdoor = matrix_at(arguments, 1)?;
    let target = matrix_at(arguments, 2)?;
    if output.start > output.end || output.end > target.columns {
        return Err(ColumnPolicyError::InvalidOutputRange {
            start: output.start,
            end: output.end,
            columns: target.columns,
        });
    }
    if output.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![
        InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: public.columns } },
        InputColumnRange { operand: 1, range: ColumnRange { start: 0, end: trapdoor.columns } },
        InputColumnRange { operand: 2, range: output },
    ])
}

fn constant_usize(expr: &IntExpr) -> Option<usize> {
    match expr {
        IntExpr::Const(value) => value.to_usize(),
        _ => None,
    }
}

fn output_columns(arguments: &[ConcreteWireType], capability: ColumnCapability) -> usize {
    match capability {
        ColumnCapability::GeneratedColumns => {
            arguments.first().and_then(matrix_type).map_or(0, |m| m.columns)
        }
        ColumnCapability::FixedOperandColumns => {
            arguments.iter().filter_map(matrix_type).find(|matrix| matrix.columns > 1).map_or_else(
                || arguments.first().and_then(matrix_type).map_or(0, |m| m.columns),
                |m| m.columns,
            )
        }
        _ => arguments.iter().filter_map(matrix_type).map(|m| m.columns).max().unwrap_or(0),
    }
}

fn inferred_output_columns(kind: &NodeKind, arguments: &[ConcreteWireType]) -> usize {
    match kind {
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => {
            match (arguments.first().and_then(matrix_type), arguments.get(1).and_then(matrix_type))
            {
                (Some(left), Some(right)) if left.is_scalar() && right.is_scalar() => 1,
                (Some(left), Some(right)) if left.is_scalar() => right.columns,
                (Some(left), Some(right)) if right.is_scalar() => left.columns,
                (_, Some(right)) => right.columns,
                _ => 0,
            }
        }
        NodeKind::MatrixMulSmallRhs => {
            arguments.get(1).and_then(matrix_type).map_or(0, |matrix| matrix.columns)
        }
        NodeKind::MatrixMulAccumulate { .. } => arguments
            .get(0)
            .and_then(matrix_type)
            .zip(arguments.get(1).and_then(matrix_type))
            .map_or(0, |(left, right)| {
                if left.is_scalar() && right.is_scalar() {
                    1
                } else if left.is_scalar() {
                    right.columns
                } else if right.is_scalar() {
                    left.columns
                } else {
                    right.columns
                }
            }),
        NodeKind::PreimageSample { .. } => {
            arguments.get(2).and_then(matrix_type).map_or(0, |matrix| matrix.columns)
        }
        NodeKind::Slice { columns: Some(range), .. } => {
            match (constant_usize(&range.start), constant_usize(&range.end)) {
                (Some(start), Some(end)) => end.saturating_sub(start),
                _ => arguments.first().and_then(matrix_type).map_or(0, |matrix| matrix.columns),
            }
        }
        NodeKind::Concat { axis: ConcatAxis::Columns | ConcatAxis::Diagonal } => arguments
            .iter()
            .filter_map(matrix_type)
            .fold(0usize, |sum, matrix| sum.checked_add(matrix.columns).unwrap_or(0)),
        NodeKind::Tensor => arguments
            .iter()
            .filter_map(matrix_type)
            .map(|matrix| matrix.columns)
            .reduce(|left, right| left.checked_mul(right).unwrap_or(0))
            .unwrap_or(0),
        NodeKind::Transpose => {
            arguments.first().and_then(matrix_type).map_or(0, |matrix| matrix.rows)
        }
        _ => output_columns(arguments, column_capability(kind, arguments)),
    }
}

/// Classify a node against the explicit GPU-operation allowlist. Unknown node
/// kinds intentionally remain `Unsupported` even if they are harmless on CPU.
pub fn effective_gpu_operation(kind: &NodeKind) -> EffectiveGpuOperation {
    match kind {
        NodeKind::ConstantMatrix { value, .. } => match value {
            ConstantMatrix::Zero |
            ConstantMatrix::Identity |
            ConstantMatrix::UnitRow { .. } |
            ConstantMatrix::UnitColumn { .. } |
            ConstantMatrix::Gadget { .. } => EffectiveGpuOperation::GeneratedConstant,
            ConstantMatrix::PowerOfBase { .. } |
            ConstantMatrix::Rotation { .. } |
            ConstantMatrix::Polynomial { .. } => EffectiveGpuOperation::SingleDeviceConstant,
        },
        NodeKind::UniformResidueSample { .. } => EffectiveGpuOperation::UniformResidueSample,
        NodeKind::UniformIntervalSample { .. } => EffectiveGpuOperation::UniformIntervalSample,
        NodeKind::GaussianSample { .. } => EffectiveGpuOperation::GaussianSample,
        NodeKind::HashSample { .. } => EffectiveGpuOperation::HashSample,
        NodeKind::TrapdoorSample { .. } => EffectiveGpuOperation::TrapdoorSample,
        NodeKind::GadgetTrapdoor { .. } => EffectiveGpuOperation::GadgetTrapdoor,
        NodeKind::TrapdoorPublic => EffectiveGpuOperation::TrapdoorPublic,
        NodeKind::PreimageSample { .. } => EffectiveGpuOperation::PreimageSample,
        NodeKind::GadgetDecompose { .. } => EffectiveGpuOperation::GadgetDecompose,
        NodeKind::MatrixScale { .. } => EffectiveGpuOperation::MatrixScale,
        NodeKind::MatrixNegate => EffectiveGpuOperation::MatrixNegate,
        NodeKind::RingAutomorphism { .. } => EffectiveGpuOperation::RingAutomorphism,
        NodeKind::ModulusSwitch { .. } => EffectiveGpuOperation::ModulusSwitch,
        NodeKind::ModulusReduce { .. } => EffectiveGpuOperation::ModulusReduce,
        NodeKind::CenteredRebase { .. } => EffectiveGpuOperation::CenteredRebase,
        NodeKind::RnsModUp { .. } => EffectiveGpuOperation::RnsModUp,
        NodeKind::RnsModDown { .. } => EffectiveGpuOperation::RnsModDown,
        NodeKind::CrtRecompose { .. } => EffectiveGpuOperation::CrtRecompose,
        NodeKind::MatrixBinary(MatrixBinaryOp::Add) => EffectiveGpuOperation::MatrixAdd,
        NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) => EffectiveGpuOperation::MatrixSubtract,
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => EffectiveGpuOperation::MatrixMultiply,
        NodeKind::MatrixMulAccumulate { .. } => EffectiveGpuOperation::MatrixMulAccumulate,
        NodeKind::MatrixMulSmallRhs => EffectiveGpuOperation::MatrixMulSmallRhs,
        NodeKind::Transpose => EffectiveGpuOperation::Transpose,
        NodeKind::Slice { .. } => EffectiveGpuOperation::Slice,
        NodeKind::Tensor => EffectiveGpuOperation::Tensor,
        NodeKind::Concat { axis: ConcatAxis::Rows } => EffectiveGpuOperation::ConcatRows,
        NodeKind::Concat { axis: ConcatAxis::Columns } => EffectiveGpuOperation::ConcatColumns,
        NodeKind::Concat { axis: ConcatAxis::Diagonal } => EffectiveGpuOperation::ConcatDiagonal,
        NodeKind::Input { .. } |
        NodeKind::ConstantInt(_) |
        NodeKind::EvaluateInt(_) |
        NodeKind::ConstantReal(_) |
        NodeKind::ConstantBool(_) |
        NodeKind::IntBinary(_) |
        NodeKind::IntCompare(_) |
        NodeKind::BitExtract { .. } |
        NodeKind::IntToReal |
        NodeKind::BoolToInt |
        NodeKind::RealBinary(_) |
        NodeKind::RealSqrt |
        NodeKind::ExtractCoefficient { .. } |
        NodeKind::LiftIntegerToConstantPolynomial { .. } |
        NodeKind::ThresholdDecode { .. } |
        NodeKind::PackPolynomialCoefficients { .. } |
        NodeKind::PolynomialFromValues { .. } |
        NodeKind::PolynomialValues { .. } |
        NodeKind::SubgraphCall(_) |
        NodeKind::ParallelLoop(_) |
        NodeKind::SequentialLoop(_) |
        NodeKind::FamilyPack { .. } |
        NodeKind::FamilyGetStatic { .. } |
        NodeKind::FamilyGetDynamic |
        NodeKind::Select { .. } => EffectiveGpuOperation::HostOrControl,
    }
}

/// Whether an operation is explicitly known to remain on the host/control
/// path. This is deliberately narrower than "not column-separable".
pub fn known_host_or_control(kind: &NodeKind) -> bool {
    effective_gpu_operation(kind) == EffectiveGpuOperation::HostOrControl
}

/// Classify an effective operation using only validated operation and concrete
/// argument types. Unknown operations never become host/control fallbacks.
pub fn column_capability(kind: &NodeKind, arguments: &[ConcreteWireType]) -> ColumnCapability {
    capability_for_effective_operation(effective_gpu_operation(kind), arguments)
}

/// Return the canonical capability for a classified operation. The argument
/// slice is reserved for future validated shape refinements and is deliberately
/// not used to invent a new capability at runtime.
pub fn capability_for_effective_operation(
    operation: EffectiveGpuOperation,
    arguments: &[ConcreteWireType],
) -> ColumnCapability {
    match operation {
        EffectiveGpuOperation::GeneratedConstant |
        EffectiveGpuOperation::UniformResidueSample |
        EffectiveGpuOperation::UniformIntervalSample |
        EffectiveGpuOperation::GaussianSample |
        EffectiveGpuOperation::HashSample => ColumnCapability::GeneratedColumns,
        EffectiveGpuOperation::SingleDeviceConstant |
        EffectiveGpuOperation::TrapdoorSample |
        EffectiveGpuOperation::GadgetTrapdoor |
        EffectiveGpuOperation::TrapdoorPublic => ColumnCapability::SingleDevice,
        EffectiveGpuOperation::PreimageSample | EffectiveGpuOperation::GadgetDecompose => {
            ColumnCapability::FixedOperandColumns
        }
        EffectiveGpuOperation::MatrixScale |
        EffectiveGpuOperation::MatrixNegate |
        EffectiveGpuOperation::RingAutomorphism |
        EffectiveGpuOperation::ModulusSwitch |
        EffectiveGpuOperation::ModulusReduce |
        EffectiveGpuOperation::CenteredRebase |
        EffectiveGpuOperation::RnsModUp |
        EffectiveGpuOperation::RnsModDown |
        EffectiveGpuOperation::CrtRecompose |
        EffectiveGpuOperation::MatrixAdd |
        EffectiveGpuOperation::MatrixSubtract => ColumnCapability::SameColumns,
        EffectiveGpuOperation::MatrixMultiply |
        EffectiveGpuOperation::MatrixMulAccumulate |
        EffectiveGpuOperation::MatrixMulSmallRhs => ColumnCapability::FixedOperandColumns,
        EffectiveGpuOperation::Transpose |
        EffectiveGpuOperation::Slice |
        EffectiveGpuOperation::Tensor |
        EffectiveGpuOperation::ConcatRows |
        EffectiveGpuOperation::ConcatColumns |
        EffectiveGpuOperation::ConcatDiagonal => ColumnCapability::MappedColumns,
        EffectiveGpuOperation::HostOrControl => ColumnCapability::HostOrControl,
        EffectiveGpuOperation::Unsupported => {
            let _ = arguments;
            ColumnCapability::Unsupported
        }
    }
}

/// Fallible capability lookup for planning boundaries. Callers that are about
/// to freeze a plan should use this form so an unsupported operation cannot be
/// mistaken for a host/control operation.
pub fn checked_column_capability(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
) -> Result<ColumnCapability, ColumnPolicyError> {
    let capability = column_capability(kind, arguments);
    if capability == ColumnCapability::Unsupported {
        Err(ColumnPolicyError::UnsupportedOperation { operation: "unknown GPU operation" })
    } else {
        Ok(capability)
    }
}

/// Return the input ranges needed for an output range.  The output width is
/// inferred from the concrete arguments; callers with a separately validated
/// output type can use [`map_output_range_to_inputs_with_output`].
pub fn map_output_range_to_inputs(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    output: ColumnRange,
) -> Result<Vec<InputColumnRange>, ColumnPolicyError> {
    // Generated values have no input type from which to infer their output
    // width. Their concrete output contract is validated by the caller, so
    // this convenience entry point only needs to lower the requested range.
    let capability = checked_column_capability(kind, arguments)?;
    let columns = if capability == ColumnCapability::GeneratedColumns {
        output.end
    } else {
        inferred_output_columns(kind, arguments)
    };
    map_output_range_to_inputs_with_output(kind, arguments, columns, output)
}

/// As [`map_output_range_to_inputs`], with an explicit validated output width.
/// Keeping this operation pure lets warmup and production share exactly the
/// same mapping even when output rows/columns are not inferable from operands.
pub fn map_output_range_to_inputs_with_output(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    output_columns: usize,
    output: ColumnRange,
) -> Result<Vec<InputColumnRange>, ColumnPolicyError> {
    match effective_gpu_operation(kind) {
        EffectiveGpuOperation::Unsupported => {
            return Err(ColumnPolicyError::UnsupportedOperation {
                operation: "unknown GPU operation",
            });
        }
        EffectiveGpuOperation::HostOrControl => {
            return Err(ColumnPolicyError::UnsupportedOperation {
                operation: "host/control operation",
            });
        }
        _ => {}
    }
    if output.end > output_columns {
        return Err(ColumnPolicyError::InvalidOutputRange {
            start: output.start,
            end: output.end,
            columns: output_columns,
        });
    }
    if output.start > output.end {
        return Err(ColumnPolicyError::InvalidOutputRange {
            start: output.start,
            end: output.end,
            columns: output_columns,
        });
    }
    let capability = column_capability(kind, arguments);
    let mut ranges = Vec::new();
    let mut push = |operand: usize, range: ColumnRange| {
        if !range.is_empty() {
            ranges.push(InputColumnRange { operand, range });
        }
    };
    match kind {
        NodeKind::ConstantMatrix { value, .. } => match value {
            ConstantMatrix::Zero |
            ConstantMatrix::Identity |
            ConstantMatrix::UnitRow { .. } |
            ConstantMatrix::UnitColumn { .. } |
            ConstantMatrix::Gadget { .. } => {}
            _ => {
                return Err(ColumnPolicyError::UnsupportedOperation {
                    operation: "single-device constant",
                });
            }
        },
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } => {}
        NodeKind::TrapdoorSample { .. } => {
            return Err(ColumnPolicyError::UnsupportedOperation { operation: "trapdoor sampling" });
        }
        NodeKind::GadgetTrapdoor { .. } | NodeKind::TrapdoorPublic => {
            return Err(ColumnPolicyError::UnsupportedOperation {
                operation: "single-device operation",
            });
        }
        NodeKind::MatrixScale { .. } |
        NodeKind::MatrixNegate |
        NodeKind::RingAutomorphism { .. } |
        NodeKind::ModulusSwitch { .. } |
        NodeKind::ModulusReduce { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::RnsModUp { .. } |
        NodeKind::RnsModDown { .. } |
        NodeKind::CrtRecompose { .. } |
        NodeKind::MatrixBinary(MatrixBinaryOp::Add | MatrixBinaryOp::Subtract) => {
            for (index, argument) in arguments.iter().enumerate() {
                if let Some(matrix) = matrix_type(argument) {
                    {
                        if output.end > matrix.columns {
                            return Err(ColumnPolicyError::InvalidOutputRange {
                                start: output.start,
                                end: output.end,
                                columns: matrix.columns,
                            });
                        }
                        push(index, output);
                    }
                }
            }
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => {
            let left = matrix_at(arguments, 0)?;
            let right = matrix_at(arguments, 1)?;
            if left.is_scalar() {
                push(0, ColumnRange { start: 0, end: 1 });
                if output.end > right.columns {
                    return Err(ColumnPolicyError::InvalidOutputRange {
                        start: output.start,
                        end: output.end,
                        columns: right.columns,
                    });
                }
                push(1, output);
            } else if right.is_scalar() {
                if output.end > left.columns {
                    return Err(ColumnPolicyError::InvalidOutputRange {
                        start: output.start,
                        end: output.end,
                        columns: left.columns,
                    });
                }
                push(0, output);
                push(1, ColumnRange { start: 0, end: 1 });
            } else {
                if output.end > right.columns {
                    return Err(ColumnPolicyError::InvalidOutputRange {
                        start: output.start,
                        end: output.end,
                        columns: right.columns,
                    });
                }
                push(0, ColumnRange { start: 0, end: left.columns });
                push(1, output);
            }
        }
        NodeKind::MatrixMulSmallRhs => {
            let left = matrix_at(arguments, 0)?;
            let right = matrix_at(arguments, 1)?;
            push(
                0,
                ColumnRange::new(0, left.columns).ok_or(ColumnPolicyError::ArithmeticOverflow)?,
            );
            if output.end > right.columns {
                return Err(ColumnPolicyError::InvalidOutputRange {
                    start: output.start,
                    end: output.end,
                    columns: right.columns,
                });
            }
            push(1, output);
        }
        NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
            for product in 0..coefficients.len() {
                let left = matrix_at(arguments, product * 2)?;
                let right = matrix_at(arguments, product * 2 + 1)?;
                let left_operand = product * 2;
                let right_operand = left_operand + 1;
                if left.is_scalar() {
                    push(left_operand, ColumnRange { start: 0, end: 1 });
                    if output.end > right.columns {
                        return Err(ColumnPolicyError::InvalidOutputRange {
                            start: output.start,
                            end: output.end,
                            columns: right.columns,
                        });
                    }
                    push(right_operand, output);
                } else if right.is_scalar() {
                    if output.end > left.columns {
                        return Err(ColumnPolicyError::InvalidOutputRange {
                            start: output.start,
                            end: output.end,
                            columns: left.columns,
                        });
                    }
                    push(left_operand, output);
                    push(right_operand, ColumnRange { start: 0, end: 1 });
                } else {
                    if output.end > right.columns {
                        return Err(ColumnPolicyError::InvalidOutputRange {
                            start: output.start,
                            end: output.end,
                            columns: right.columns,
                        });
                    }
                    push(
                        left_operand,
                        ColumnRange::new(0, left.columns)
                            .ok_or(ColumnPolicyError::ArithmeticOverflow)?,
                    );
                    push(right_operand, output);
                }
            }
            if *has_bias {
                let bias = matrix_at(arguments, coefficients.len() * 2)?;
                if bias.is_scalar() {
                    push(coefficients.len() * 2, ColumnRange { start: 0, end: 1 });
                } else {
                    push(coefficients.len() * 2, output);
                }
            }
        }
        NodeKind::PreimageSample { .. } => {
            return preimage_input_ranges(arguments, output);
        }
        NodeKind::GadgetDecompose { .. } => {
            for (index, argument) in arguments.iter().enumerate() {
                if let Some(matrix) = matrix_type(argument) {
                    if output.end > matrix.columns {
                        return Err(ColumnPolicyError::InvalidOutputRange {
                            start: output.start,
                            end: output.end,
                            columns: matrix.columns,
                        });
                    }
                    push(index, output);
                }
            }
        }
        NodeKind::Slice { columns, .. } => {
            let input = matrix_at(arguments, 0)?;
            let offset = columns
                .as_ref()
                .map(|range| range.start.clone())
                .and_then(|expr| constant_usize(&expr))
                .unwrap_or(0);
            let mapped =
                output.checked_translate(offset).ok_or(ColumnPolicyError::ArithmeticOverflow)?;
            let slice_end = columns
                .as_ref()
                .and_then(|range| constant_usize(&range.end))
                .unwrap_or(input.columns);
            if mapped.end > input.columns || mapped.end > slice_end {
                return Err(ColumnPolicyError::InvalidOutputRange {
                    start: mapped.start,
                    end: mapped.end,
                    columns: slice_end.min(input.columns),
                });
            }
            push(0, mapped);
        }
        NodeKind::Transpose => {
            let input = matrix_at(arguments, 0)?;
            if output.end > input.rows {
                return Err(ColumnPolicyError::InvalidOutputRange {
                    start: output.start,
                    end: output.end,
                    columns: input.rows,
                });
            }
            // A transpose's output columns correspond to input rows.  The
            // range is intentionally retained as a column-range descriptor;
            // backend lowering interprets it against the mapped axis.
            push(0, output);
        }
        NodeKind::Concat { axis: ConcatAxis::Rows } => {
            for index in 0..arguments.len() {
                let matrix = matrix_at(arguments, index)?;
                if output.end > matrix.columns {
                    return Err(ColumnPolicyError::InvalidOutputRange {
                        start: output.start,
                        end: output.end,
                        columns: matrix.columns,
                    });
                }
                push(index, output);
            }
        }
        NodeKind::Concat { axis: ConcatAxis::Columns | ConcatAxis::Diagonal } => {
            let mut offset = 0usize;
            for index in 0..arguments.len() {
                let matrix = matrix_at(arguments, index)?;
                let end = offset
                    .checked_add(matrix.columns)
                    .ok_or(ColumnPolicyError::ArithmeticOverflow)?;
                let start = output.start.max(offset);
                let stop = output.end.min(end);
                if start < stop {
                    push(index, ColumnRange { start: start - offset, end: stop - offset });
                }
                offset = end;
            }
            if output.end > offset {
                return Err(ColumnPolicyError::InvalidOutputRange {
                    start: output.start,
                    end: output.end,
                    columns: offset,
                });
            }
        }
        NodeKind::Tensor => {
            let left = matrix_at(arguments, 0)?;
            let right = matrix_at(arguments, 1)?;
            let width = left
                .columns
                .checked_mul(right.columns)
                .ok_or(ColumnPolicyError::ArithmeticOverflow)?;
            if output.end > width {
                return Err(ColumnPolicyError::InvalidOutputRange {
                    start: output.start,
                    end: output.end,
                    columns: width,
                });
            }
            // Each output segment may cross a right-hand tensor segment. Return
            // the minimal participating ranges on both operands.
            if !output.is_empty() {
                let mut column = output.start;
                while column < output.end {
                    let left_column = column / right.columns;
                    let right_start = column % right.columns;
                    let count = (right.columns - right_start).min(output.end - column);
                    push(0, ColumnRange { start: left_column, end: left_column + 1 });
                    push(1, ColumnRange { start: right_start, end: right_start + count });
                    column += count;
                }
            }
        }
        _ => {
            return Err(ColumnPolicyError::UnsupportedOperation {
                operation: "host/control operation",
            });
        }
    }
    let _ = capability;
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{expr::IntExpr, node::IndexRange};
    use num_bigint::BigInt;

    fn matrix(rows: usize, columns: usize) -> ConcreteWireType {
        ConcreteWireType::Matrix(ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 8,
            rows,
            columns,
        })
    }

    #[test]
    fn preimage_fixed_pair_is_never_column_sliced() {
        let arguments = [matrix(2, 8), matrix(2, 8), matrix(2, 20)];
        let reads = preimage_input_ranges(&arguments, ColumnRange { start: 12, end: 17 }).unwrap();
        assert_eq!(
            reads,
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: 8 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 0, end: 8 } },
                InputColumnRange { operand: 2, range: ColumnRange { start: 12, end: 17 } },
            ]
        );
        assert!(preimage_input_ranges(&arguments[..2], ColumnRange { start: 0, end: 1 }).is_err());
        assert!(preimage_input_ranges(&arguments, ColumnRange { start: 19, end: 21 }).is_err());
    }

    #[test]
    fn tensor_mapping_preserves_cross_boundary_segments() {
        let reads = map_output_range_to_inputs(
            &NodeKind::Tensor,
            &[matrix(2, 3), matrix(4, 5)],
            ColumnRange { start: 4, end: 8 },
        )
        .unwrap();
        assert_eq!(
            reads,
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: 1 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 4, end: 5 } },
                InputColumnRange { operand: 0, range: ColumnRange { start: 1, end: 2 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 0, end: 3 } },
            ]
        );
        assert_eq!(
            map_output_range_to_inputs(
                &NodeKind::MatrixNegate,
                &[matrix(1, 1)],
                ColumnRange { start: 0, end: 1 }
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn mapped_column_reads_are_correct() {
        let args = vec![matrix(2, 3), matrix(4, 5)];
        let result = map_output_range_to_inputs(
            &NodeKind::Concat { axis: ConcatAxis::Columns },
            &args,
            ColumnRange { start: 2, end: 6 },
        )
        .unwrap();
        assert_eq!(
            result,
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 2, end: 3 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 0, end: 3 } },
            ]
        );
        let slice = NodeKind::Slice {
            rows: None,
            columns: Some(IndexRange { start: IntExpr::constant(1), end: IntExpr::constant(4) }),
        };
        assert_eq!(
            map_output_range_to_inputs(&slice, &[matrix(2, 6)], ColumnRange { start: 0, end: 2 })
                .unwrap(),
            vec![InputColumnRange { operand: 0, range: ColumnRange { start: 1, end: 3 } }]
        );
    }

    #[test]
    fn scalar_multiply_positions_are_distinct() {
        let left_scalar = vec![matrix(1, 1), matrix(2, 4)];
        let right_scalar = vec![matrix(2, 4), matrix(1, 1)];
        let both_scalar = vec![matrix(1, 1), matrix(1, 1)];
        let kind = NodeKind::MatrixBinary(MatrixBinaryOp::Multiply);
        assert_eq!(column_capability(&kind, &left_scalar), ColumnCapability::FixedOperandColumns);
        assert_eq!(
            map_output_range_to_inputs(&kind, &left_scalar, ColumnRange { start: 1, end: 3 })
                .unwrap(),
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: 1 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 1, end: 3 } },
            ]
        );
        assert_eq!(
            map_output_range_to_inputs(&kind, &right_scalar, ColumnRange { start: 1, end: 3 })
                .unwrap(),
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 1, end: 3 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 0, end: 1 } },
            ]
        );
        assert!(
            map_output_range_to_inputs(&kind, &both_scalar, ColumnRange { start: 0, end: 1 })
                .unwrap() ==
                vec![
                    InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: 1 } },
                    InputColumnRange { operand: 1, range: ColumnRange { start: 0, end: 1 } },
                ]
        );
        let normal = vec![matrix(3, 6), matrix(6, 4)];
        assert_eq!(
            map_output_range_to_inputs(&kind, &normal, ColumnRange { start: 1, end: 3 }).unwrap(),
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: 6 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 1, end: 3 } },
            ]
        );
        let right_scalar = vec![matrix(3, 6), matrix(1, 1)];
        assert_eq!(
            map_output_range_to_inputs(&kind, &right_scalar, ColumnRange { start: 4, end: 6 })
                .unwrap(),
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 4, end: 6 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 0, end: 1 } },
            ]
        );
    }

    #[test]
    fn trapdoor_generation_is_not_column_sampling() {
        let kind = NodeKind::GadgetTrapdoor {
            matrix_type: mxx_ir_core::types::MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(8),
                rows: IntExpr::constant(2),
                columns: IntExpr::constant(4),
            },
            base: IntExpr::constant(2),
        };
        assert_eq!(column_capability(&kind, &[matrix(2, 4)]), ColumnCapability::SingleDevice);
        assert!(matches!(
            map_output_range_to_inputs(&kind, &[matrix(2, 4)], ColumnRange { start: 0, end: 1 }),
            Err(ColumnPolicyError::UnsupportedOperation { .. })
        ));
        let sample = NodeKind::UniformResidueSample {
            matrix_type: mxx_ir_core::types::MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(8),
                rows: IntExpr::constant(2),
                columns: IntExpr::constant(4),
            },
        };
        assert_eq!(column_capability(&sample, &[matrix(2, 4)]), ColumnCapability::GeneratedColumns);
        let trapdoor_sample = NodeKind::TrapdoorSample {
            matrix_type: mxx_ir_core::types::MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(8),
                rows: IntExpr::constant(2),
                columns: IntExpr::constant(4),
            },
            sigma: mxx_ir_core::expr::RealExpr::from_integer(1),
            gadget_base: IntExpr::constant(2),
            digit_count: IntExpr::constant(2),
            preimage_max_coefficient_bound: IntExpr::constant(3),
        };
        assert_eq!(
            column_capability(&trapdoor_sample, &[matrix(2, 4)]),
            ColumnCapability::SingleDevice
        );
    }

    #[test]
    fn operation_allowlist_keeps_known_host_nodes_explicit() {
        let kind = NodeKind::ConstantInt(num_bigint::BigInt::from(3));
        assert_eq!(effective_gpu_operation(&kind), EffectiveGpuOperation::HostOrControl);
        assert_eq!(column_capability(&kind, &[]), ColumnCapability::HostOrControl);
        assert!(matches!(
            map_output_range_to_inputs(&kind, &[], ColumnRange { start: 0, end: 1 }),
            Err(ColumnPolicyError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn warmup_and_runtime_share_range_lowering() {
        let args = vec![matrix(2, 4), matrix(3, 8)];
        let kind = NodeKind::MatrixMulSmallRhs;
        let warmup =
            map_output_range_to_inputs(&kind, &args, ColumnRange { start: 2, end: 6 }).unwrap();
        let production =
            map_output_range_to_inputs(&kind, &args, ColumnRange { start: 2, end: 6 }).unwrap();
        assert_eq!(warmup, production);
        assert_eq!(warmup[0].range, ColumnRange { start: 0, end: 4 });
        assert_eq!(warmup[1].range, ColumnRange { start: 2, end: 6 });
        let accumulate = NodeKind::MatrixMulAccumulate {
            coefficients: vec![IntExpr::constant(1)],
            has_bias: true,
        };
        let accumulate_inputs = vec![matrix(2, 4), matrix(4, 8), matrix(2, 8)];
        assert_eq!(
            map_output_range_to_inputs(
                &accumulate,
                &accumulate_inputs,
                ColumnRange { start: 3, end: 7 },
            )
            .unwrap(),
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: 4 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 3, end: 7 } },
                InputColumnRange { operand: 2, range: ColumnRange { start: 3, end: 7 } },
            ]
        );
    }

    #[test]
    fn derived_cache_is_not_semantic_state() {
        let args = vec![matrix(2, 3), matrix(2, 5)];
        let kind = NodeKind::Tensor;
        let uncached =
            map_output_range_to_inputs(&kind, &args, ColumnRange { start: 2, end: 7 }).unwrap();
        let cached_equivalent = map_output_range_to_inputs_with_output(
            &kind,
            &args,
            15,
            ColumnRange { start: 2, end: 7 },
        )
        .unwrap();
        assert_eq!(uncached, cached_equivalent);
    }
}
