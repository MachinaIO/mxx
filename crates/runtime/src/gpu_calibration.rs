//! GPU operation identities, separability helpers, memory snapshots, and column waterfilling.

use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv, concretize_wire_type, encoding,
    node::{ConcatAxis, ConstantMatrix, IndexRange, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, NodeId, WireType},
};
use rayon::prelude::*;
use serde::Serialize;
use std::{cell::RefCell, collections::VecDeque};

struct OperationIdentityCacheEntry {
    kind: NodeKind,
    arguments: Vec<ConcreteWireType>,
    outputs: Vec<ConcreteWireType>,
    bindings: ParamEnv,
    identity: [u8; 32],
}

/// Row-block addition has a different allocation footprint from ordinary Add.
/// Preserve the ordered layout, including empty blocks and the native kernel's
/// block-count fallback boundary, instead of sharing Add's identity.
pub fn gpu_row_block_add_operation_identity(
    add_identity: [u8; 32],
    blocks: &[ConcreteMatrixType],
) -> Result<[u8; 32], String> {
    encoding::hash_canonical(&("mxx-runtime/gpu-row-block-add/v1", add_identity, blocks))
        .map_err(|error| error.to_string())
}

/// Sparse row sums have their own kernel and temporary allocation footprint.
pub fn gpu_sum_rows_operation_identity(
    source: &ConcreteMatrixType,
    output: &ConcreteMatrixType,
    rows: &[Vec<usize>],
) -> Result<[u8; 32], String> {
    encoding::hash_canonical(&("mxx-runtime/gpu-sum-rows/v1", source, output, rows))
        .map_err(|error| error.to_string())
}

/// Tensor row sums consume both operand layouts and use an independent identity.
pub fn gpu_tensor_sum_rows_operation_identity(
    left: &ConcreteMatrixType,
    right: &ConcreteMatrixType,
    output: &ConcreteMatrixType,
    rows: &[Vec<usize>],
) -> Result<[u8; 32], String> {
    encoding::hash_canonical(&("mxx-runtime/gpu-tensor-sum-rows/v1", left, right, output, rows))
        .map_err(|error| error.to_string())
}

thread_local! {
    // Cache only public operation metadata, never backend state or measured widths.
    // Full input equality makes reuse independent of graph mutation and node IDs.
    // A bounded thread-local cache avoids serializing concurrent graph execution.
    static OPERATION_IDENTITIES: RefCell<VecDeque<OperationIdentityCacheEntry>> =
        const { RefCell::new(VecDeque::new()) };
}

pub fn gpu_operation_is_column_separable(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::ConstantMatrix { .. } |
            NodeKind::UniformResidueSample { .. } |
            NodeKind::UniformIntervalSample { .. } |
            NodeKind::GaussianSample { .. } |
            NodeKind::HashSample { .. } |
            NodeKind::GadgetTrapdoor { .. } |
            NodeKind::PreimageSample { .. } |
            NodeKind::GadgetDecompose { .. } |
            NodeKind::MatrixScale { .. } |
            NodeKind::MatrixNegate |
            NodeKind::RingAutomorphism { .. } |
            NodeKind::ModulusSwitch { .. } |
            NodeKind::ModulusReduce { .. } |
            NodeKind::CenteredExtend { .. } |
            NodeKind::BlockModSwitch { .. } |
            NodeKind::CenteredRebase { .. } |
            NodeKind::RnsModUp { .. } |
            NodeKind::RnsModDown { .. } |
            NodeKind::MatrixBinary(_) |
            NodeKind::MatrixMulAccumulate { .. } |
            NodeKind::MatrixMulSmallRhs |
            NodeKind::CrtRecompose { .. } |
            NodeKind::Concat { .. } |
            NodeKind::Transpose |
            NodeKind::Tensor |
            NodeKind::Slice { .. }
    )
}

/// Refines the kind-level range capability using validated concrete operand types.
pub fn gpu_operation_is_column_separable_for_types(
    kind: &NodeKind,
    concrete_argument_types: &[ConcreteWireType],
) -> bool {
    fn matrix_type(ty: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
        match ty {
            ConcreteWireType::IndexedFamily { element, .. } => matrix_type(element),
            _ => ty.matrix_type(),
        }
    }
    if !gpu_operation_is_column_separable(kind) {
        return false;
    }
    let NodeKind::MatrixMulAccumulate { coefficients, .. } = kind else {
        return true;
    };
    coefficients.iter().enumerate().all(|(product, _)| {
        concrete_argument_types.get(2 * product).and_then(matrix_type).is_some() &&
            concrete_argument_types.get(2 * product + 1).and_then(matrix_type).is_some()
    })
}

pub(crate) fn gpu_calibration_groups(
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    kind: &NodeKind,
    declared_argument_types: &[WireType],
    declared_output_types: &[WireType],
    envs: &[ParamEnv],
) -> Result<Vec<(Option<[u8; 32]>, Vec<usize>)>, String> {
    let operations = envs
        .par_iter()
        .map(|env| {
            let argument_types = declared_argument_types
                .iter()
                .map(|ty| concretize_wire_type(ty, env, scope_id, node_id))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?;
            let output_types = declared_output_types
                .iter()
                .map(|ty| concretize_wire_type(ty, env, scope_id, node_id))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?;
            gpu_operation_is_column_separable_for_types(kind, &argument_types)
                .then(|| {
                    gpu_calibration_operation_identity(kind, &argument_types, &output_types, env)
                })
                .transpose()
        })
        .collect::<Result<Vec<_>, String>>()?;
    // Rayon preserves input order. Publish groups in first-occurrence order so
    // batch/transcript ordering does not depend on worker scheduling.
    let mut groups = Vec::<(Option<[u8; 32]>, Vec<usize>)>::new();
    for (instance, operation) in operations.into_iter().enumerate() {
        if let Some((_, members)) = groups.iter_mut().find(|(candidate, _)| *candidate == operation)
        {
            members.push(instance);
        } else {
            groups.push((operation, vec![instance]));
        }
    }
    Ok(groups)
}

pub fn gpu_matrix_multiply_scales_left(
    left_rows: usize,
    left_columns: usize,
    right_rows: usize,
    right_columns: usize,
) -> bool {
    (right_rows, right_columns) == (1, 1) && (left_rows, left_columns) != (1, 1)
}

/// Canonical per-primitive identity shared by estimator collection and runtime
/// preflight. Selector-only loop values and hash-domain tags do not affect the
/// allocation/kernel path and are deliberately erased;
/// concrete types retain every loop-dependent shape and bound that does affect it.
pub fn gpu_calibration_operation_identity(
    kind: &NodeKind,
    concrete_argument_types: &[ConcreteWireType],
    concrete_output_types: &[ConcreteWireType],
    bindings: &ParamEnv,
) -> Result<[u8; 32], String> {
    if let Some(identity) = OPERATION_IDENTITIES.with(|cache| {
        let mut cache = cache.borrow_mut();
        let position = cache.iter().position(|entry| {
            entry.kind == *kind &&
                entry.arguments == concrete_argument_types &&
                entry.outputs == concrete_output_types &&
                entry.bindings == *bindings
        })?;
        let entry = cache.remove(position).expect("cached operation position");
        let identity = entry.identity;
        cache.push_back(entry);
        Some(identity)
    }) {
        return Ok(identity);
    }
    #[derive(Serialize)]
    struct OperationIdentity<'a> {
        kind: &'a NodeKind,
        concrete_argument_types: &'a [ConcreteWireType],
        concrete_output_types: &'a [ConcreteWireType],
        bindings: &'a ParamEnv,
    }

    let shape_bindings = ParamEnv {
        integers: bindings.integers.clone(),
        reals: bindings.reals.clone(),
        loop_indices: Default::default(),
    };
    fn one_column(ty: &mut ConcreteWireType) {
        match ty {
            ConcreteWireType::Matrix(matrix) |
            ConcreteWireType::SmallMatrix { matrix, .. } |
            ConcreteWireType::Preimage { matrix, .. } |
            ConcreteWireType::Trapdoor { matrix, .. } => matrix.columns = 1,
            ConcreteWireType::IndexedFamily { element, .. } => one_column(element),
            _ => {}
        }
    }
    fn matrix_type(ty: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
        match ty {
            ConcreteWireType::IndexedFamily { element, .. } => matrix_type(element),
            _ => ty.matrix_type(),
        }
    }
    fn normalize_output(types: &mut [ConcreteWireType]) {
        types.iter_mut().for_each(one_column);
    }
    fn output_columns(types: &mut [ConcreteWireType], columns: usize) {
        for ty in types {
            if let Some(matrix) = match ty {
                ConcreteWireType::Matrix(matrix) |
                ConcreteWireType::SmallMatrix { matrix, .. } |
                ConcreteWireType::Preimage { matrix, .. } |
                ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix),
                _ => None,
            } {
                matrix.columns = columns;
            }
        }
    }

    let mut argument_types = concrete_argument_types.to_vec();
    let mut output_types = concrete_output_types.to_vec();
    let mut shape_kind = kind.clone();
    let column_separable_for_types =
        gpu_operation_is_column_separable_for_types(kind, concrete_argument_types);
    match &mut shape_kind {
        NodeKind::ConstantMatrix { matrix_type, value } => {
            matrix_type.columns = IntExpr::constant(1);
            normalize_output(&mut output_types);
            match value {
                ConstantMatrix::UnitRow { index } | ConstantMatrix::UnitColumn { index } => {
                    *index = IntExpr::constant(0);
                }
                ConstantMatrix::Rotation { exponent } => *exponent = IntExpr::constant(0),
                // Coefficient values change the uploaded contents, not the
                // full-ring GPU allocation or NTT path. Keep the variant and
                // coefficient count, and erase only its numerical data.
                ConstantMatrix::Polynomial { coefficients } => {
                    coefficients.fill(IntExpr::constant(0));
                }
                ConstantMatrix::PowerOfBase { base, exponent } => {
                    *base = IntExpr::constant(2);
                    *exponent = IntExpr::constant(0);
                }
                _ => {}
            }
        }
        NodeKind::GadgetTrapdoor { matrix_type, .. } => {
            matrix_type.columns = IntExpr::constant(1);
            normalize_output(&mut output_types);
        }
        NodeKind::UniformResidueSample { matrix_type } |
        NodeKind::UniformIntervalSample { matrix_type, .. } |
        NodeKind::GaussianSample { matrix_type, .. } |
        NodeKind::HashSample { matrix_type, .. } => {
            matrix_type.columns = IntExpr::constant(1);
            normalize_output(&mut output_types);
        }
        NodeKind::PreimageSample { matrix_type, .. } => {
            matrix_type.columns = IntExpr::constant(1);
            if let Some(target) = argument_types.get_mut(2) {
                one_column(target);
            }
            normalize_output(&mut output_types);
        }
        NodeKind::GadgetDecompose { .. } |
        NodeKind::MatrixScale { .. } |
        NodeKind::ModulusSwitch { .. } |
        NodeKind::ModulusReduce { .. } |
        NodeKind::CenteredExtend { .. } |
        NodeKind::BlockModSwitch { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::RnsModUp { .. } |
        NodeKind::RnsModDown { .. } |
        NodeKind::MatrixNegate => {
            if let Some(input) = argument_types.first_mut() {
                one_column(input);
            }
            normalize_output(&mut output_types);
        }
        NodeKind::RingAutomorphism { index } => {
            // The automorphism selector changes values but not the GPU kernel shape or cost.
            // Normalize only this calibration identity; the graph's semantic kind is untouched.
            *index = IntExpr::constant(1);
            if let Some(input) = argument_types.first_mut() {
                one_column(input);
            }
            normalize_output(&mut output_types);
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Add | MatrixBinaryOp::Subtract) => {
            argument_types.iter_mut().for_each(one_column);
            normalize_output(&mut output_types);
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => {
            let scale_left = matches!(
                (argument_types.first().and_then(matrix_type),
                 argument_types.get(1).and_then(matrix_type)),
                (Some(left), Some(right))
                    if gpu_matrix_multiply_scales_left(
                        left.rows, left.columns, right.rows, right.columns)
            );
            let scalable_argument = if scale_left { 0 } else { 1 };
            if let Some(argument) = argument_types.get_mut(scalable_argument) {
                one_column(argument);
            }
            normalize_output(&mut output_types);
        }
        NodeKind::MatrixMulSmallRhs => {
            if let Some(rhs) = argument_types.get_mut(1) {
                one_column(rhs);
            }
            normalize_output(&mut output_types);
        }
        NodeKind::MatrixMulAccumulate { has_bias, .. } if column_separable_for_types => {
            let product_argument_count = argument_types.len() - usize::from(*has_bias);
            for left_index in (0..product_argument_count).step_by(2) {
                let left =
                    matrix_type(&argument_types[left_index]).expect("validated fused multiply LHS");
                let right = matrix_type(&argument_types[left_index + 1])
                    .expect("validated fused multiply RHS");
                let scalable = if gpu_matrix_multiply_scales_left(
                    left.rows,
                    left.columns,
                    right.rows,
                    right.columns,
                ) {
                    left_index
                } else {
                    left_index + 1
                };
                one_column(&mut argument_types[scalable]);
            }
            if *has_bias {
                one_column(argument_types.last_mut().expect("bias argument exists"));
            }
            normalize_output(&mut output_types);
        }
        NodeKind::MatrixMulAccumulate { .. } => {}
        NodeKind::CrtRecompose { .. } | NodeKind::Concat { axis: ConcatAxis::Rows } => {
            argument_types.iter_mut().for_each(one_column);
            normalize_output(&mut output_types);
        }
        NodeKind::Transpose => {
            if let Some(input) = argument_types.first_mut().and_then(|ty| match ty {
                ConcreteWireType::Matrix(matrix) |
                ConcreteWireType::SmallMatrix { matrix, .. } |
                ConcreteWireType::Preimage { matrix, .. } |
                ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix),
                _ => None,
            }) {
                input.rows = 1;
            }
            normalize_output(&mut output_types);
        }
        NodeKind::Tensor => {
            // Output columns are laid out as left-column groups of C_r columns. C_r controls
            // segment boundaries and kernel launch count, so retain the complete right shape and
            // normalize only the repeatable left-column dimension.
            if let Some(left) = argument_types.first_mut() {
                one_column(left);
            }
            let right_columns = argument_types
                .get(1)
                .and_then(matrix_type)
                .map(|matrix| matrix.columns)
                .ok_or_else(|| "tensor calibration RHS has no matrix type".to_owned())?;
            output_columns(&mut output_types, right_columns);
        }
        NodeKind::Concat { axis: ConcatAxis::Columns } => {
            argument_types.iter_mut().for_each(one_column);
            // Retain one column per input so the canonical schemas continue to satisfy the
            // concat output relation while excluding every scalable source width.
            output_columns(&mut output_types, argument_types.len());
        }
        // Diagonal ranges create one padded row block per input. Every original block width
        // determines its global offset and overlap pattern, so no column field is safely erased.
        NodeKind::Concat { axis: ConcatAxis::Diagonal } => {}
        NodeKind::Slice { rows, columns } => {
            let output_rows = output_types
                .first()
                .and_then(ConcreteWireType::matrix_type)
                .map(|matrix| matrix.rows);
            if let Some(input) = argument_types.first_mut() {
                if let Some(matrix) = match input {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. } |
                    ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix),
                    _ => None,
                } {
                    if let Some(output_rows) = output_rows {
                        matrix.rows = output_rows;
                    }
                    matrix.columns = 1;
                }
            }
            normalize_output(&mut output_types);
            *rows = output_rows.map(|rows| IndexRange {
                start: IntExpr::constant(0),
                end: IntExpr::constant(rows),
            });
            *columns = Some(IndexRange { start: IntExpr::constant(0), end: IntExpr::constant(1) });
        }
        NodeKind::ExtractCoefficient { position, .. } => {
            *position = IntExpr::constant(0);
        }
        _ => {}
    }
    if let NodeKind::HashSample { tag_prefix, tag_components, .. } = &mut shape_kind {
        tag_prefix.fill(0);
        for component in tag_components {
            use mxx_ir_core::node::HashTagComponent;
            match component {
                HashTagComponent::Bytes(bytes) => bytes.fill(0),
                HashTagComponent::Integer(expression) |
                HashTagComponent::Decimal(expression) |
                HashTagComponent::U64Le(expression) => *expression = IntExpr::constant(0),
                HashTagComponent::Operand(_) => {}
            }
        }
    }

    let identity = encoding::hash_canonical(&OperationIdentity {
        kind: &shape_kind,
        concrete_argument_types: &argument_types,
        concrete_output_types: &output_types,
        bindings: &shape_bindings,
    })
    .map_err(|error| error.to_string())?;
    OPERATION_IDENTITIES.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() == 64 {
            cache.pop_front();
        }
        cache.push_back(OperationIdentityCacheEntry {
            kind: kind.clone(),
            arguments: concrete_argument_types.to_vec(),
            outputs: concrete_output_types.to_vec(),
            bindings: bindings.clone(),
            identity,
        });
    });
    Ok(identity)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuDeviceMemory {
    pub total_bytes: u64,
    pub resident_bytes: u64,
}

impl GpuDeviceMemory {
    pub fn budget_bytes(self, vram_percent: u32) -> Result<u64, GpuCalibrationError> {
        self.validate()?;
        if !(1..=100).contains(&vram_percent) {
            return Err(GpuCalibrationError::InvalidVramPercent(vram_percent));
        }
        let percent = u64::from(vram_percent);
        (self.total_bytes / 100)
            .checked_mul(percent)
            .and_then(|whole| {
                (self.total_bytes % 100)
                    .checked_mul(percent)
                    .and_then(|remainder| whole.checked_add(remainder / 100))
            })
            .ok_or(GpuCalibrationError::ArithmeticOverflow)
    }

    fn validate(self) -> Result<(), GpuCalibrationError> {
        if self.resident_bytes > self.total_bytes {
            Err(GpuCalibrationError::ResidentMemoryExceedsTotal {
                total_bytes: self.total_bytes,
                resident_bytes: self.resident_bytes,
            })
        } else {
            Ok(())
        }
    }
}

/// Assign columns using the smallest common water level over per-device caps.
/// The result is deterministic in device order and never exceeds total capacity.
/// Fresh-output admission must reject insufficient aggregate capacity before
/// calling this; wave iteration may deliberately assign only the next capacity.
pub fn gpu_capped_waterfill_columns(
    capacities: &[usize],
    remaining_columns: usize,
) -> Result<Vec<usize>, GpuCalibrationError> {
    if capacities.is_empty() {
        return Err(GpuCalibrationError::ZeroGpuCount);
    }
    let fleet_capacity = capacities.iter().try_fold(0usize, |total, capacity| {
        total.checked_add(*capacity).ok_or(GpuCalibrationError::ArithmeticOverflow)
    })?;
    let target = remaining_columns.min(fleet_capacity);
    if target == 0 {
        return Ok(vec![0; capacities.len()]);
    }
    let mut low = 1usize;
    let mut high = capacities.iter().copied().max().expect("nonempty GPU capacities");
    while low < high {
        let level = low + (high - low) / 2;
        let covered = capacities
            .iter()
            .try_fold(0usize, |sum, capacity| sum.checked_add((*capacity).min(level)))
            .ok_or(GpuCalibrationError::ArithmeticOverflow)?;
        if covered >= target {
            high = level;
        } else {
            low = level + 1;
        }
    }
    let level = low;
    let mut assigned =
        capacities.iter().map(|capacity| (*capacity).min(level - 1)).collect::<Vec<_>>();
    let baseline = assigned.iter().try_fold(0usize, |sum, columns| {
        sum.checked_add(*columns).ok_or(GpuCalibrationError::ArithmeticOverflow)
    })?;
    let mut remainder = target - baseline;
    for (columns, capacity) in assigned.iter_mut().zip(capacities) {
        if remainder == 0 {
            break;
        }
        if *capacity >= level {
            *columns += 1;
            remainder -= 1;
        }
    }
    debug_assert_eq!(remainder, 0);
    Ok(assigned)
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuCalibrationError {
    #[error("GPU VRAM percentage must be between 1 and 100, got {0}")]
    InvalidVramPercent(u32),
    #[error("resident GPU memory {resident_bytes} exceeds total memory {total_bytes}")]
    ResidentMemoryExceedsTotal { total_bytes: u64, resident_bytes: u64 },
    #[error("GPU fleet must contain at least one device")]
    ZeroGpuCount,
    #[error("GPU calibration arithmetic overflow")]
    ArithmeticOverflow,
}
#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::types::ConcreteMatrixType;
    use num_bigint::BigInt;

    #[test]
    fn sum_rows_identity_preserves_order_and_duplicate_terms() {
        let source = ConcreteMatrixType {
            modulus: BigInt::from(97),
            ring_dimension: 8,
            rows: 4,
            columns: 1,
        };
        let output = ConcreteMatrixType { rows: 2, ..source.clone() };
        let identity =
            gpu_sum_rows_operation_identity(&source, &output, &[vec![0], vec![1, 2]]).unwrap();
        assert_ne!(
            identity,
            gpu_sum_rows_operation_identity(&source, &output, &[vec![1, 2], vec![0]]).unwrap()
        );
        assert_ne!(
            identity,
            gpu_sum_rows_operation_identity(&source, &output, &[vec![0], vec![1, 1, 2]]).unwrap()
        );
    }

    #[test]
    fn tensor_sum_rows_identity_preserves_both_operand_layouts() {
        let left = ConcreteMatrixType {
            modulus: BigInt::from(97),
            ring_dimension: 8,
            rows: 2,
            columns: 1,
        };
        let right = ConcreteMatrixType { rows: 3, ..left.clone() };
        let source = ConcreteMatrixType { rows: 6, ..left.clone() };
        let output = ConcreteMatrixType { rows: 1, ..left.clone() };
        let rows = [vec![0, 2]];
        let identity =
            gpu_tensor_sum_rows_operation_identity(&left, &right, &output, &rows).unwrap();
        assert_ne!(
            identity,
            gpu_tensor_sum_rows_operation_identity(&right, &left, &output, &rows).unwrap()
        );
        assert_ne!(
            identity,
            gpu_tensor_sum_rows_operation_identity(&left, &right, &output, &[vec![2, 0]]).unwrap()
        );
        assert_ne!(identity, gpu_sum_rows_operation_identity(&source, &output, &rows).unwrap());
    }

    #[test]
    fn row_block_add_identity_separates_layouts_and_native_fallback() {
        let block = ConcreteMatrixType {
            modulus: BigInt::from(97),
            ring_dimension: 8,
            rows: 1,
            columns: 3,
        };
        let base = [7; 32];
        let native = gpu_row_block_add_operation_identity(base, &vec![block.clone(); 16]).unwrap();
        let fallback =
            gpu_row_block_add_operation_identity(base, &vec![block.clone(); 17]).unwrap();
        assert_ne!(native, fallback);
        assert_ne!(native, base);
        let tall = ConcreteMatrixType { rows: 2, ..block.clone() };
        let forward =
            gpu_row_block_add_operation_identity(base, &[block.clone(), tall.clone()]).unwrap();
        let reversed = gpu_row_block_add_operation_identity(base, &[tall, block.clone()]).unwrap();
        assert_ne!(forward, reversed);
        assert_ne!(
            forward,
            gpu_row_block_add_operation_identity(
                [8; 32],
                &[block.clone(), ConcreteMatrixType { rows: 2, ..block }]
            )
            .unwrap()
        );
    }

    #[test]
    fn operation_identity_normalizes_column_separable_shapes() {
        let matrix = |columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(257u16),
                ring_dimension: 16,
                rows: 3,
                columns,
            })
        };
        let full = gpu_calibration_operation_identity(
            &NodeKind::MatrixNegate,
            &[matrix(29)],
            &[matrix(29)],
            &ParamEnv::default(),
        )
        .unwrap();
        let representative = gpu_calibration_operation_identity(
            &NodeKind::MatrixNegate,
            &[matrix(1)],
            &[matrix(1)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_eq!(full, representative);
    }

    #[test]
    fn operation_identity_cache_preserves_parameter_changes_and_eviction() {
        let kind = NodeKind::MatrixNegate;
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            modulus: BigInt::from(257),
            ring_dimension: 16,
            rows: 2,
            columns: 4,
        });
        let types = [matrix];
        OPERATION_IDENTITIES.with(|cache| cache.borrow_mut().clear());
        let identities = (0..70)
            .map(|parameter| {
                let bindings = ParamEnv {
                    integers: [("parameter".to_owned(), BigInt::from(parameter))].into(),
                    ..ParamEnv::default()
                };
                let identity =
                    gpu_calibration_operation_identity(&kind, &types, &types, &bindings).unwrap();
                assert_eq!(
                    identity,
                    gpu_calibration_operation_identity(&kind, &types, &types, &bindings).unwrap()
                );
                (bindings, identity)
            })
            .collect::<Vec<_>>();
        assert!(identities.windows(2).all(|pair| pair[0].1 != pair[1].1));
        OPERATION_IDENTITIES.with(|cache| assert_eq!(cache.borrow().len(), 64));
        // Old entries were evicted; recomputation must retain canonical identity.
        for (bindings, identity) in identities {
            assert_eq!(
                identity,
                gpu_calibration_operation_identity(&kind, &types, &types, &bindings).unwrap()
            );
        }
    }

    #[test]
    fn automorphism_identity_ignores_selector_and_column_count() {
        let matrix = |columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(257u16),
                ring_dimension: 16,
                rows: 3,
                columns,
            })
        };
        let kind = NodeKind::RingAutomorphism { index: IntExpr::constant(3) };
        let full = gpu_calibration_operation_identity(
            &kind,
            &[matrix(29)],
            &[matrix(29)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_eq!(kind, NodeKind::RingAutomorphism { index: IntExpr::constant(3) });
        let representative = gpu_calibration_operation_identity(
            &NodeKind::RingAutomorphism { index: IntExpr::constant(3) },
            &[matrix(1)],
            &[matrix(1)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_eq!(full, representative);

        let different_index = gpu_calibration_operation_identity(
            &NodeKind::RingAutomorphism { index: IntExpr::constant(5) },
            &[matrix(29)],
            &[matrix(29)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_eq!(full, different_index);

        let different_shape = gpu_calibration_operation_identity(
            &kind,
            &[ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(257u16),
                ring_dimension: 16,
                rows: 4,
                columns: 29,
            })],
            &[ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(257u16),
                ring_dimension: 16,
                rows: 4,
                columns: 29,
            })],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_ne!(full, different_shape);
    }

    #[test]
    fn modulus_conversion_identity_preserves_both_rings_and_normalizes_columns() {
        let matrix = |modulus, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(modulus),
                ring_dimension: 32,
                rows: 2,
                columns,
            })
        };
        let kind = NodeKind::ModulusSwitch { modulus: IntExpr::constant(15) };
        let identity = gpu_calibration_operation_identity(
            &kind,
            &[matrix(105, 17)],
            &[matrix(15, 17)],
            &ParamEnv::default(),
        )
        .unwrap();
        let one_column = gpu_calibration_operation_identity(
            &kind,
            &[matrix(105, 1)],
            &[matrix(15, 1)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_eq!(identity, one_column);
        let different_source = gpu_calibration_operation_identity(
            &kind,
            &[matrix(1155, 17)],
            &[matrix(15, 17)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_ne!(identity, different_source);
        let reduced = gpu_calibration_operation_identity(
            &NodeKind::ModulusReduce { modulus: IntExpr::constant(15) },
            &[matrix(105, 17)],
            &[matrix(15, 17)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_ne!(identity, reduced);
    }

    #[test]
    fn calibration_groups_preserve_loop_dependent_preimage_bounds() {
        let matrix = mxx_ir_core::types::MatrixType {
            modulus: IntExpr::constant(257),
            ring_dimension: IntExpr::constant(16),
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(4),
        };
        let bound = IntExpr::Add(Box::new(IntExpr::constant(7)), Box::new(IntExpr::LoopIndex(0)));
        let kind = NodeKind::PreimageSample {
            matrix_type: matrix.clone(),
            max_coefficient_bound: bound.clone(),
        };
        let arguments = vec![
            WireType::Matrix(matrix.clone()),
            WireType::Matrix(matrix.clone()),
            WireType::Matrix(matrix.clone()),
        ];
        let outputs = vec![WireType::Preimage { matrix, max_coefficient_bound: bound }];
        let env = |index| ParamEnv {
            loop_indices: [(0, BigInt::from(index))].into_iter().collect(),
            ..ParamEnv::default()
        };
        let groups = gpu_calibration_groups(
            &FrozenGraphScopeId::Root,
            NodeId(9),
            &kind,
            &arguments,
            &outputs,
            &[env(0), env(1), env(0)],
        )
        .unwrap();

        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|(operation, _)| operation.is_some()));
        assert_eq!(groups[0].1, vec![0, 2]);
        assert_eq!(groups[1].1, vec![1]);
        assert_ne!(groups[0].0, groups[1].0);
    }

    #[test]
    fn operation_identity_preserves_tensor_segments_and_diagonal_block_layout() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(257u16),
                ring_dimension: 16,
                rows,
                columns,
            })
        };
        let env = ParamEnv::default();
        let tensor_full = gpu_calibration_operation_identity(
            &NodeKind::Tensor,
            &[matrix(2, 7), matrix(3, 11)],
            &[matrix(6, 77)],
            &env,
        )
        .unwrap();
        let tensor_one_left_group = gpu_calibration_operation_identity(
            &NodeKind::Tensor,
            &[matrix(2, 1), matrix(3, 11)],
            &[matrix(6, 11)],
            &env,
        )
        .unwrap();
        assert_eq!(tensor_full, tensor_one_left_group);
        let different_right_segment = gpu_calibration_operation_identity(
            &NodeKind::Tensor,
            &[matrix(2, 1), matrix(3, 5)],
            &[matrix(6, 5)],
            &env,
        )
        .unwrap();
        assert_ne!(tensor_full, different_right_segment);

        let diagonal_full = gpu_calibration_operation_identity(
            &NodeKind::Concat { axis: ConcatAxis::Diagonal },
            &[matrix(2, 7), matrix(3, 11)],
            &[matrix(5, 18)],
            &env,
        )
        .unwrap();
        let different_diagonal_layout = gpu_calibration_operation_identity(
            &NodeKind::Concat { axis: ConcatAxis::Diagonal },
            &[matrix(2, 1), matrix(3, 1)],
            &[matrix(5, 2)],
            &env,
        )
        .unwrap();
        assert_ne!(diagonal_full, different_diagonal_layout);
    }

    #[test]
    fn multiply_identity_normalizes_the_runtime_scaled_operand() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(257u16),
                ring_dimension: 16,
                rows,
                columns,
            })
        };
        let identity = |arguments: &[ConcreteWireType], output: ConcreteWireType| {
            gpu_calibration_operation_identity(
                &NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                arguments,
                &[output],
                &ParamEnv::default(),
            )
            .unwrap()
        };

        assert_eq!(
            identity(&[matrix(1, 1), matrix(1, 10)], matrix(1, 10)),
            identity(&[matrix(1, 1), matrix(1, 1)], matrix(1, 1))
        );
        assert_eq!(
            identity(&[matrix(2, 10), matrix(1, 1)], matrix(2, 10)),
            identity(&[matrix(2, 1), matrix(1, 1)], matrix(2, 1))
        );
        assert_eq!(
            identity(&[matrix(2, 3), matrix(3, 10)], matrix(2, 10)),
            identity(&[matrix(2, 3), matrix(3, 1)], matrix(2, 1))
        );
    }

    #[test]
    fn fused_multiply_separability_and_identity_support_both_orientations() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(257u16),
                ring_dimension: 16,
                rows,
                columns,
            })
        };
        let kind = NodeKind::MatrixMulAccumulate {
            coefficients: vec![IntExpr::constant(1)],
            has_bias: false,
        };

        assert!(gpu_operation_is_column_separable_for_types(&kind, &[matrix(1, 1), matrix(1, 10)]));
        assert!(gpu_operation_is_column_separable_for_types(&kind, &[matrix(2, 3), matrix(3, 10)]));
        assert!(gpu_operation_is_column_separable_for_types(&kind, &[matrix(2, 10), matrix(1, 1)]));

        let unsupported_full = gpu_calibration_operation_identity(
            &kind,
            &[matrix(2, 10), matrix(1, 1)],
            &[matrix(2, 10)],
            &ParamEnv::default(),
        )
        .unwrap();
        let unsupported_reduced = gpu_calibration_operation_identity(
            &kind,
            &[matrix(2, 1), matrix(1, 1)],
            &[matrix(2, 1)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_eq!(unsupported_full, unsupported_reduced);

        let mixed_kind = NodeKind::MatrixMulAccumulate {
            coefficients: vec![IntExpr::constant(1), IntExpr::constant(1)],
            has_bias: false,
        };
        assert!(gpu_operation_is_column_separable_for_types(
            &mixed_kind,
            &[matrix(1, 1), matrix(2, 10), matrix(2, 10), matrix(1, 1)]
        ));
        let mixed_full = gpu_calibration_operation_identity(
            &mixed_kind,
            &[matrix(1, 1), matrix(2, 10), matrix(2, 10), matrix(1, 1)],
            &[matrix(2, 10)],
            &ParamEnv::default(),
        )
        .unwrap();
        let mixed_reduced = gpu_calibration_operation_identity(
            &mixed_kind,
            &[matrix(1, 1), matrix(2, 1), matrix(2, 1), matrix(1, 1)],
            &[matrix(2, 1)],
            &ParamEnv::default(),
        )
        .unwrap();
        assert_eq!(mixed_full, mixed_reduced);
    }

    #[test]
    fn gadget_trapdoor_is_column_separable_and_normalized() {
        let kind = |columns| NodeKind::GadgetTrapdoor {
            matrix_type: mxx_ir_core::types::MatrixType {
                modulus: IntExpr::constant(257),
                ring_dimension: IntExpr::constant(16),
                rows: IntExpr::constant(3),
                columns: IntExpr::constant(columns),
            },
            base: IntExpr::constant(4),
        };
        let output = |columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(257u16),
                ring_dimension: 16,
                rows: 3,
                columns,
            })
        };

        assert!(gpu_operation_is_column_separable(&kind(30)));
        assert_eq!(
            gpu_calibration_operation_identity(&kind(30), &[], &[output(30)], &ParamEnv::default())
                .unwrap(),
            gpu_calibration_operation_identity(&kind(1), &[], &[output(1)], &ParamEnv::default())
                .unwrap()
        );
    }

    #[test]
    fn operation_identity_normalizes_only_proven_selectors() {
        let matrix_type = mxx_ir_core::types::MatrixType {
            modulus: IntExpr::constant(257),
            ring_dimension: IntExpr::constant(16),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(8),
        };
        let concrete = ConcreteWireType::Matrix(ConcreteMatrixType {
            modulus: BigInt::from(257u16),
            ring_dimension: 16,
            rows: 1,
            columns: 8,
        });
        let constant_identity = |value| {
            gpu_calibration_operation_identity(
                &NodeKind::ConstantMatrix { matrix_type: matrix_type.clone(), value },
                &[],
                std::slice::from_ref(&concrete),
                &ParamEnv::default(),
            )
            .unwrap()
        };
        assert_eq!(
            constant_identity(ConstantMatrix::Rotation { exponent: IntExpr::constant(1) }),
            constant_identity(ConstantMatrix::Rotation { exponent: IntExpr::constant(7) })
        );
        assert_eq!(
            constant_identity(ConstantMatrix::UnitRow { index: IntExpr::constant(1) }),
            constant_identity(ConstantMatrix::UnitRow { index: IntExpr::constant(7) })
        );
        assert_eq!(
            constant_identity(ConstantMatrix::UnitColumn { index: IntExpr::constant(1) }),
            constant_identity(ConstantMatrix::UnitColumn { index: IntExpr::constant(7) })
        );
        assert_eq!(
            constant_identity(ConstantMatrix::PowerOfBase {
                base: IntExpr::constant(4),
                exponent: IntExpr::constant(1),
            }),
            constant_identity(ConstantMatrix::PowerOfBase {
                base: IntExpr::constant(4),
                exponent: IntExpr::constant(2),
            })
        );
        assert_eq!(
            constant_identity(ConstantMatrix::Polynomial {
                coefficients: vec![IntExpr::constant(1)],
            }),
            constant_identity(ConstantMatrix::Polynomial {
                coefficients: vec![IntExpr::constant(2)],
            })
        );

        let extract = |position| {
            gpu_calibration_operation_identity(
                &NodeKind::ExtractCoefficient {
                    position: IntExpr::constant(position),
                    canonical_input_exclusive_upper: None,
                },
                std::slice::from_ref(&concrete),
                &[ConcreteWireType::Int],
                &ParamEnv::default(),
            )
            .unwrap()
        };
        assert_eq!(extract(1), extract(7));
    }

    #[test]
    fn capped_waterfill_balances_equal_and_unequal_capacities() {
        assert_eq!(gpu_capped_waterfill_columns(&[100, 100], 176).unwrap(), vec![88, 88]);
        assert_eq!(gpu_capped_waterfill_columns(&[20, 100, 100], 176).unwrap(), vec![20, 78, 78]);
        assert_eq!(gpu_capped_waterfill_columns(&[10; 4], 2).unwrap(), vec![1, 1, 0, 0]);
        assert_eq!(gpu_capped_waterfill_columns(&[2, 3, 3], usize::MAX).unwrap(), vec![2, 3, 3]);
    }

    #[test]
    fn capped_waterfill_rejects_empty_gpu_fleet() {
        assert_eq!(gpu_capped_waterfill_columns(&[], 0), Err(GpuCalibrationError::ZeroGpuCount));
    }

    #[test]
    fn capped_waterfill_exhaustively_preserves_capacity_and_minimal_level() {
        for gpu_count in 1..=5 {
            for gpu0 in 1..=7 {
                for nonzero in 1..=7 {
                    let capacities = (0..gpu_count)
                        .map(|device| if device == 0 { gpu0 } else { nonzero })
                        .collect::<Vec<_>>();
                    let capacity = capacities.iter().sum::<usize>();
                    for requested in 0..=capacity + 2 {
                        let assigned =
                            gpu_capped_waterfill_columns(&capacities, requested).unwrap();
                        let target = requested.min(capacity);
                        assert_eq!(assigned.len(), gpu_count);
                        assert_eq!(assigned.iter().sum::<usize>(), target);
                        assert!(assigned.iter().zip(&capacities).all(|(used, cap)| used <= cap));
                        if target >= gpu_count {
                            assert!(assigned.iter().all(|used| *used > 0));
                        }
                        if target > 0 {
                            let level = assigned.iter().copied().max().unwrap();
                            assert!(assigned.iter().zip(&capacities).all(|(used, cap)| {
                                *used == (*cap).min(level) || *used == (*cap).min(level - 1)
                            }));
                            assert!(
                                capacities.iter().map(|cap| (*cap).min(level)).sum::<usize>() >=
                                    target
                            );
                            if level > 1 {
                                assert!(
                                    capacities
                                        .iter()
                                        .map(|cap| (*cap).min(level - 1))
                                        .sum::<usize>() <
                                        target
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn output_caps_allow_idle_devices_and_distinct_nonzero_headroom() {
        assert_eq!(gpu_capped_waterfill_columns(&[0, 2, 9, 4], 11).unwrap(), vec![0, 2, 5, 4]);
        assert_eq!(gpu_capped_waterfill_columns(&[0, 0, 0], 0).unwrap(), vec![0, 0, 0]);
        assert_eq!(
            gpu_capped_waterfill_columns(&[usize::MAX], usize::MAX).unwrap(),
            vec![usize::MAX]
        );
        assert_eq!(
            gpu_capped_waterfill_columns(&[usize::MAX, 1], 1),
            Err(GpuCalibrationError::ArithmeticOverflow)
        );
    }

    #[test]
    fn device_memory_budget_validates_residency_and_percentage() {
        let memory = GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 300 };
        assert_eq!(memory.budget_bytes(80).unwrap(), 800);
        assert_eq!(memory.budget_bytes(0), Err(GpuCalibrationError::InvalidVramPercent(0)));
        assert_eq!(
            (GpuDeviceMemory { total_bytes: 100, resident_bytes: 101 }).budget_bytes(80),
            Err(GpuCalibrationError::ResidentMemoryExceedsTotal {
                total_bytes: 100,
                resident_bytes: 101,
            })
        );
    }
}
