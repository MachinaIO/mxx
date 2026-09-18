//! Setup-time calibration shared by the GPU estimator and runtime scheduler.
//!
//! The fixed planner consumes measured points scoped to one concrete device
//! and execution context.  Older role-based pilot helpers remain below only
//! for the legacy dynamic backend; they are deliberately not part of the
//! fixed-plan calibration contract.

use crate::gpu_execution_plan::{FrozenGpuPlan, GpuExecutionSiteKey};
use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv, concretize_wire_type, encoding,
    node::{ConcatAxis, ConstantMatrix, IndexRange, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, NodeId, WireType},
};
use mxx_primitives::poly::dcrt::gpu::GpuDeviceIdentity;
use serde::Serialize;
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    fmt,
    sync::{Arc, RwLock},
};

struct OperationIdentityCacheEntry {
    kind: NodeKind,
    arguments: Vec<ConcreteWireType>,
    outputs: Vec<ConcreteWireType>,
    bindings: ParamEnv,
    identity: [u8; 32],
}

/// Row-block addition has a different allocation footprint from ordinary Add.
/// Preserve the ordered layout, including empty blocks and the native kernel's
/// block-count fallback boundary, instead of sharing Add's measured profile.
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

/// Tensor row sums consume both operand layouts and use an independent pilot.
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

/// Canonical identity for the homogeneous-fleet memory policy used by both
/// estimator pilots and runtime preflight. Device ordinals are intentionally
/// excluded: GPU 0 has its own measured role and every nonzero GPU reuses the
/// representative nonzero-role slope.
pub fn gpu_calibration_environment(
    representative_device: &GpuDeviceIdentity,
    device_count: usize,
    vram_percent: u32,
) -> Arc<[u8]> {
    const POLICY_REVISION: &[u8] = b"mxx-runtime-gpu-fleet-column-sharding-v2";
    let mut encoded = b"mxx-gpu-fleet-calibration-environment-v2".to_vec();
    encoded.extend_from_slice(&(POLICY_REVISION.len() as u64).to_le_bytes());
    encoded.extend_from_slice(POLICY_REVISION);
    encoded.extend_from_slice(&(representative_device.name.len() as u64).to_le_bytes());
    encoded.extend_from_slice(representative_device.name.as_bytes());
    encoded.extend_from_slice(&representative_device.compute_major.to_le_bytes());
    encoded.extend_from_slice(&representative_device.compute_minor.to_le_bytes());
    encoded.extend_from_slice(&(representative_device.total_global_memory as u64).to_le_bytes());
    encoded.extend_from_slice(&(device_count as u64).to_le_bytes());
    // Role-layout revision: GPU 0 is calibrated separately and GPU 1 is the
    // representative for every homogeneous nonzero device.
    encoded.extend_from_slice(&1u32.to_le_bytes());
    encoded.extend_from_slice(&vram_percent.to_le_bytes());
    encoded.into()
}

/// Build an environment identity for a concrete device/context.
///
/// The old [`gpu_calibration_environment`] helper intentionally shared a
/// representative nonzero-GPU role.  That is unsuitable for fixed planning:
/// device identity, native implementation revision, and CUDA context
/// generation are all part of the evidence that makes a measurement usable.
/// Keep those facts in the key instead of deriving a bytes-per-column slope
/// from another device.
pub fn gpu_device_calibration_environment(
    device: &GpuDeviceIdentity,
    device_id: i32,
    backend_revision: &str,
    context_generation: u64,
    vram_percent: u32,
) -> Arc<[u8]> {
    let mut encoded = b"mxx-gpu-device-calibration-environment-v3".to_vec();
    encoded.extend_from_slice(&device_id.to_le_bytes());
    encoded.extend_from_slice(&(backend_revision.len() as u64).to_le_bytes());
    encoded.extend_from_slice(backend_revision.as_bytes());
    encoded.extend_from_slice(&context_generation.to_le_bytes());
    encoded.extend_from_slice(&(device.name.len() as u64).to_le_bytes());
    encoded.extend_from_slice(device.name.as_bytes());
    encoded.extend_from_slice(&device.compute_major.to_le_bytes());
    encoded.extend_from_slice(&device.compute_minor.to_le_bytes());
    encoded.extend_from_slice(&(device.total_global_memory as u64).to_le_bytes());
    encoded.extend_from_slice(&vram_percent.to_le_bytes());
    encoded.into()
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
    let mut groups = Vec::<(Option<[u8; 32]>, Vec<usize>)>::new();
    for (instance, env) in envs.iter().enumerate() {
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
        let operation = gpu_operation_is_column_separable_for_types(kind, &argument_types)
            .then(|| gpu_calibration_operation_identity(kind, &argument_types, &output_types, env))
            .transpose()?;
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
                .and_then(ConcreteWireType::matrix_type)
                .map(|matrix| matrix.columns)
                .unwrap_or(1);
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

/// Exact identity of a calibrated operation in a particular GPU environment.
///
/// Callers provide their canonical semantic encoding rather than a graph node
/// number. The environment encoding must cover every non-operation input that
/// can change allocation behavior (for example the implementation revision and
/// GPU model). `Arc` keeps registry lookup keys cheap to clone while byte-wise
/// equality makes cache hits exact rather than dependent on a digest collision.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct GpuCalibrationKey {
    operation: Arc<[u8]>,
    environment: Arc<[u8]>,
}

impl GpuCalibrationKey {
    pub fn new(operation: impl Into<Arc<[u8]>>, environment: impl Into<Arc<[u8]>>) -> Self {
        Self { operation: operation.into(), environment: environment.into() }
    }

    pub fn operation(&self) -> &[u8] {
        &self.operation
    }

    pub fn environment(&self) -> &[u8] {
        &self.environment
    }

    /// Construct a lookup key only from complete measured-context provenance.
    /// The legacy two-byte-slice constructor remains for dynamic callers, but
    /// fixed planning should prefer this form so operation and environment
    /// cannot be accidentally detached from their device context.
    pub fn from_context(context: &GpuCalibrationContext) -> Self {
        Self::new(context.operation.clone(), context.environment.clone())
    }
}

impl fmt::Debug for GpuCalibrationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuCalibrationKey")
            .field("operation_bytes", &self.operation.len())
            .field("environment_bytes", &self.environment.len())
            .finish()
    }
}

/// Provenance attached to a measured memory point used by fixed planning.
///
/// This is intentionally device-specific.  In particular, there is no
/// `Gpu0`/`Nonzero` role in this type: two devices may only share evidence if
/// their complete contexts are equal and the caller explicitly reuses the
/// same key.  A native revision or context-generation change invalidates the
/// evidence through ordinary typed equality.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct GpuCalibrationContext {
    operation: Arc<[u8]>,
    environment: Arc<[u8]>,
    device_id: i32,
    backend_revision: Arc<[u8]>,
    context_generation: u64,
}

impl GpuCalibrationContext {
    pub fn new(
        operation: impl AsRef<[u8]>,
        environment: impl AsRef<[u8]>,
        device_id: i32,
        backend_revision: impl AsRef<[u8]>,
        context_generation: u64,
    ) -> Self {
        Self {
            operation: Arc::from(operation.as_ref()),
            environment: Arc::from(environment.as_ref()),
            device_id,
            backend_revision: Arc::from(backend_revision.as_ref()),
            context_generation,
        }
    }

    pub fn operation(&self) -> &[u8] {
        &self.operation
    }

    pub fn environment(&self) -> &[u8] {
        &self.environment
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn backend_revision(&self) -> &[u8] {
        &self.backend_revision
    }

    pub fn context_generation(&self) -> u64 {
        self.context_generation
    }

    pub fn key(&self) -> GpuCalibrationKey {
        GpuCalibrationKey::from_context(self)
    }
}

/// One measured incremental memory observation at a positive local width.
///
/// `peak_bytes` is tied to the exact execution class in the surrounding
/// [`GpuCalibrationContext`].  It is not a per-column slope and must never be
/// multiplied from one pilot to manufacture another width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMeasuredMemoryPoint {
    pub columns: usize,
    pub peak_bytes: u64,
}

/// Point-first memory evidence for one device and execution context.
///
/// The table retains every measured anchor.  A query at an interior width may
/// use bounded affine interpolation for ranking/reporting, but a hard memory
/// admission decision must use a directly measured anchor (or an independent
/// exact allocation bound).  There is deliberately no origin scaling,
/// quadratic term, or cross-device role fallback here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuMeasuredMemoryProfile {
    context: GpuCalibrationContext,
    points: Arc<[GpuMeasuredMemoryPoint]>,
}

impl GpuMeasuredMemoryProfile {
    pub fn from_points(
        context: GpuCalibrationContext,
        points: impl IntoIterator<Item = GpuMeasuredMemoryPoint>,
    ) -> Result<Self, GpuCalibrationError> {
        let points = points.into_iter().collect::<Vec<_>>();
        if points.is_empty() {
            return Err(GpuCalibrationError::NoMeasuredMemoryPoints);
        }
        let mut previous = None;
        for point in &points {
            if point.columns == 0 {
                return Err(GpuCalibrationError::ZeroMeasuredMemoryColumns);
            }
            if point.peak_bytes == 0 {
                return Err(GpuCalibrationError::ZeroMeasuredMemoryPeak);
            }
            if previous.is_some_and(|columns| point.columns <= columns) {
                return Err(GpuCalibrationError::UnsortedMeasuredMemoryPoints);
            }
            previous = Some(point.columns);
        }
        Ok(Self { context, points: points.into() })
    }

    pub fn context(&self) -> &GpuCalibrationContext {
        &self.context
    }

    pub fn points(&self) -> &[GpuMeasuredMemoryPoint] {
        &self.points
    }

    pub fn measured_anchor(&self, columns: usize) -> Option<GpuMeasuredMemoryPoint> {
        self.points.iter().copied().find(|point| point.columns == columns)
    }

    /// Resolve a bounded memory estimate at `columns` using direct lookup or
    /// affine interpolation between adjacent measured anchors.  Extrapolation
    /// is rejected so an invalid context cannot silently look usable.
    pub fn estimate_bytes(&self, columns: usize) -> Result<u64, GpuCalibrationError> {
        if columns == 0 {
            return Err(GpuCalibrationError::ZeroMeasuredMemoryColumns);
        }
        if let Some(point) = self.measured_anchor(columns) {
            return Ok(point.peak_bytes);
        }
        let upper = self.points.iter().find(|point| point.columns > columns);
        let lower = self.points.iter().rev().find(|point| point.columns < columns);
        let (Some(lower), Some(upper)) = (lower, upper) else {
            return Err(GpuCalibrationError::MeasuredMemoryExtrapolation { columns });
        };
        let x = u128::try_from(columns).map_err(|_| GpuCalibrationError::ArithmeticOverflow)?;
        let x0 =
            u128::try_from(lower.columns).map_err(|_| GpuCalibrationError::ArithmeticOverflow)?;
        let x1 =
            u128::try_from(upper.columns).map_err(|_| GpuCalibrationError::ArithmeticOverflow)?;
        let y0 = u128::from(lower.peak_bytes);
        let y1 = u128::from(upper.peak_bytes);
        let denominator = x1 - x0;
        let base = y0.checked_mul(denominator).ok_or(GpuCalibrationError::ArithmeticOverflow)?;
        let delta = if y1 >= y0 {
            (y1 - y0).checked_mul(x - x0).ok_or(GpuCalibrationError::ArithmeticOverflow)?
        } else {
            (y0 - y1).checked_mul(x - x0).ok_or(GpuCalibrationError::ArithmeticOverflow)?
        };
        let numerator = if y1 >= y0 { base.checked_add(delta) } else { base.checked_sub(delta) }
            .ok_or(GpuCalibrationError::ArithmeticOverflow)?;
        let rounded = numerator
            .checked_add(denominator - 1)
            .ok_or(GpuCalibrationError::ArithmeticOverflow)? /
            denominator;
        u64::try_from(rounded).map_err(|_| GpuCalibrationError::ArithmeticOverflow)
    }

    /// Return the largest directly measured anchor that fits the current
    /// budget.  Interpolated resource values are intentionally not accepted
    /// as a hard admission certificate.
    pub fn largest_fitting_anchor(
        &self,
        memory: GpuDeviceMemory,
        vram_percent: u32,
    ) -> Result<usize, GpuCalibrationError> {
        memory.validate()?;
        if !(1..=100).contains(&vram_percent) {
            return Err(GpuCalibrationError::InvalidVramPercent(vram_percent));
        }
        let budget = (u128::from(memory.total_bytes) * u128::from(vram_percent)) / 100;
        let available = budget.saturating_sub(u128::from(memory.resident_bytes));
        self.points
            .iter()
            .filter(|point| u128::from(point.peak_bytes) <= available)
            .map(|point| point.columns)
            .max()
            .ok_or(GpuCalibrationError::NoFittingMeasuredMemoryAnchor)
    }
}

/// Legacy single-pilot memory cost used only by the dynamic backend.
///
/// Fixed planning must use [`GpuMeasuredMemoryProfile`] instead.  This type
/// remains source-compatible for the dynamic compatibility path, but its
/// `bytes_per_column` value is not valid evidence for a fixed plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuDeviceCalibration {
    pilot_columns: usize,
    pilot_peak_bytes: u64,
    bytes_per_column: u64,
}

impl GpuDeviceCalibration {
    pub fn from_pilot(
        pilot_columns: usize,
        pilot_peak_bytes: u64,
    ) -> Result<Self, GpuCalibrationError> {
        if pilot_columns == 0 {
            return Err(GpuCalibrationError::ZeroPilotColumns);
        }
        if pilot_peak_bytes == 0 {
            return Err(GpuCalibrationError::ZeroPilotPeak);
        }
        let columns =
            u64::try_from(pilot_columns).map_err(|_| GpuCalibrationError::ArithmeticOverflow)?;
        let bytes_per_column = pilot_peak_bytes
            .checked_add(columns - 1)
            .ok_or(GpuCalibrationError::ArithmeticOverflow)? /
            columns;
        Ok(Self { pilot_columns, pilot_peak_bytes, bytes_per_column })
    }

    pub fn pilot_columns(self) -> usize {
        self.pilot_columns
    }

    pub fn pilot_peak_bytes(self) -> u64 {
        self.pilot_peak_bytes
    }

    pub fn bytes_per_column(self) -> u64 {
        self.bytes_per_column
    }

    fn derive_width(
        self,
        memory: GpuDeviceMemory,
        vram_percent: u32,
        role: GpuDeviceRole,
    ) -> Result<usize, GpuCalibrationError> {
        memory.validate()?;
        if !(1..=100).contains(&vram_percent) {
            return Err(GpuCalibrationError::InvalidVramPercent(vram_percent));
        }
        let percent = u64::from(vram_percent);
        let budget_bytes = (memory.total_bytes / 100)
            .checked_mul(percent)
            .and_then(|whole| {
                (memory.total_bytes % 100)
                    .checked_mul(percent)
                    .map(|remainder| whole + remainder / 100)
            })
            .ok_or(GpuCalibrationError::ArithmeticOverflow)?;
        let available_bytes = budget_bytes.saturating_sub(memory.resident_bytes);
        let width = available_bytes / self.bytes_per_column;
        if width == 0 {
            return Err(GpuCalibrationError::InsufficientMemory {
                role,
                budget_bytes,
                resident_bytes: memory.resident_bytes,
                bytes_per_column: self.bytes_per_column,
            });
        }
        usize::try_from(width).map_err(|_| GpuCalibrationError::ArithmeticOverflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuDeviceMemory {
    pub total_bytes: u64,
    pub resident_bytes: u64,
}

impl GpuDeviceMemory {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuDeviceRole {
    Gpu0,
    Nonzero,
}

/// Legacy role-based pilot results for the dynamic backend.
///
/// Fixed planning must not reuse the `nonzero` role as a representative for
/// another physical device; use one [`GpuMeasuredMemoryProfile`] per concrete
/// device/context instead.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuCalibrationProfile {
    pub gpu0: GpuDeviceCalibration,
    pub nonzero: Option<GpuDeviceCalibration>,
}

impl GpuCalibrationProfile {
    /// Return every per-device width frozen for a site. This is the preferred
    /// adapter for heterogeneous fleets; unlike calibration it never queries
    /// current resident memory.
    pub fn fixed_columns_for_plan(
        plan: &FrozenGpuPlan,
        site: GpuExecutionSiteKey,
    ) -> Result<Vec<usize>, GpuCalibrationError> {
        plan.validate()
            .map_err(|error| GpuCalibrationError::InvalidFrozenPlan(error.to_string()))?;
        let choice = plan.node_choice(site).ok_or(GpuCalibrationError::MissingPlanSite(site))?;
        if choice.columns_per_job.is_empty() ||
            choice.columns_per_job.iter().all(|width| *width == 0)
        {
            return Err(GpuCalibrationError::ZeroPlanWidth(site));
        }
        Ok(choice.columns_per_job.clone())
    }

    /// Return the widths already frozen in a plan. Production callers must
    /// use this adapter instead of applying the legacy calibration slope to
    /// current allocator residency; residency-dependent [`Self::derive_widths`]
    /// is retained only for the dynamic compatibility backend.
    pub fn fixed_widths_for_plan(
        plan: &FrozenGpuPlan,
        site: GpuExecutionSiteKey,
    ) -> Result<GpuColumnWidths, GpuCalibrationError> {
        let columns = Self::fixed_columns_for_plan(plan, site)?;
        let gpu0 = columns[0];
        let nonzero = columns.get(1).copied();
        if let Some(expected) = nonzero {
            if columns[1..].iter().any(|width| *width != expected) {
                return Err(GpuCalibrationError::PlanNonzeroWidthMismatch(site));
            }
        }
        if gpu0 == 0 && nonzero.is_none_or(|width| width == 0) {
            return Err(GpuCalibrationError::ZeroPlanWidth(site));
        }
        Ok(GpuColumnWidths { gpu0, nonzero })
    }

    pub fn derive_widths(
        &self,
        gpu0_memory: GpuDeviceMemory,
        nonzero_memory: Option<GpuDeviceMemory>,
        vram_percent: u32,
    ) -> Result<GpuColumnWidths, GpuCalibrationError> {
        let gpu0 = self.gpu0.derive_width(gpu0_memory, vram_percent, GpuDeviceRole::Gpu0)?;
        let nonzero =
            match nonzero_memory {
                Some(memory) => Some(
                    self.nonzero
                        .ok_or(GpuCalibrationError::MissingNonzeroCalibration)?
                        .derive_width(memory, vram_percent, GpuDeviceRole::Nonzero)?,
                ),
                None => None,
            };
        Ok(GpuColumnWidths { gpu0, nonzero })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuColumnWidths {
    pub gpu0: usize,
    pub nonzero: Option<usize>,
}

impl GpuColumnWidths {
    /// The CUDA default pool's high-water mark is device-global. When more
    /// than one mxx context is live on a role device, the pilot still proves
    /// that one column executes, but its shared high-water delta cannot safely
    /// justify scaling beyond that column.
    pub fn constrain_for_live_contexts(
        mut self,
        gpu0_live_contexts: usize,
        nonzero_live_contexts: Option<usize>,
    ) -> Self {
        if gpu0_live_contexts > 1 {
            self.gpu0 = 1;
        }
        if nonzero_live_contexts.is_some_and(|count| count > 1) {
            self.nonzero = self.nonzero.map(|_| 1);
        }
        self
    }

    pub fn columns_per_wave(self, gpu_count: usize) -> Result<usize, GpuCalibrationError> {
        if gpu_count == 0 {
            return Err(GpuCalibrationError::ZeroGpuCount);
        }
        if self.gpu0 == 0 {
            return Err(GpuCalibrationError::ZeroRoleWidth(GpuDeviceRole::Gpu0));
        }
        let nonzero_gpu_count = gpu_count - 1;
        if nonzero_gpu_count == 0 {
            return Ok(self.gpu0);
        }
        let nonzero = self.nonzero.ok_or(GpuCalibrationError::MissingNonzeroCalibration)?;
        if nonzero == 0 {
            return Err(GpuCalibrationError::ZeroRoleWidth(GpuDeviceRole::Nonzero));
        }
        nonzero
            .checked_mul(nonzero_gpu_count)
            .and_then(|columns| columns.checked_add(self.gpu0))
            .ok_or(GpuCalibrationError::ArithmeticOverflow)
    }

    pub fn chunk_count(
        self,
        total_columns: usize,
        gpu_count: usize,
    ) -> Result<usize, GpuCalibrationError> {
        if total_columns == 0 {
            return Ok(0);
        }
        let columns_per_wave = self.columns_per_wave(gpu_count)?;
        Ok(total_columns.div_ceil(columns_per_wave))
    }
}

/// Assigns one wave using the smallest common cap that covers the requested columns.
/// The result is deterministic in device order and sums to at most one fleet wave.
pub fn gpu_capped_waterfill_columns(
    widths: GpuColumnWidths,
    gpu_count: usize,
    remaining_columns: usize,
) -> Result<Vec<usize>, GpuCalibrationError> {
    let fleet_capacity = widths.columns_per_wave(gpu_count)?;
    let target = remaining_columns.min(fleet_capacity);
    if target == 0 {
        return Ok(vec![0; gpu_count]);
    }
    let nonzero = widths.nonzero.unwrap_or(widths.gpu0);
    let capacities = (0..gpu_count)
        .map(|device| if device == 0 { widths.gpu0 } else { nonzero })
        .collect::<Vec<_>>();
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
        if capacity >= level {
            *columns += 1;
            remainder -= 1;
        }
    }
    debug_assert_eq!(remainder, 0);
    Ok(assigned)
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuCalibrationError {
    #[error("fixed calibration has no measured memory points")]
    NoMeasuredMemoryPoints,
    #[error("fixed calibration memory points must use positive columns")]
    ZeroMeasuredMemoryColumns,
    #[error("fixed calibration memory points must use positive peaks")]
    ZeroMeasuredMemoryPeak,
    #[error("fixed calibration memory points must be strictly sorted")]
    UnsortedMeasuredMemoryPoints,
    #[error("no measured memory anchor can fit the requested budget")]
    NoFittingMeasuredMemoryAnchor,
    #[error("fixed calibration cannot extrapolate memory at {columns} columns")]
    MeasuredMemoryExtrapolation { columns: usize },
    #[error("GPU calibration pilot must contain at least one column")]
    ZeroPilotColumns,
    #[error("GPU calibration pilot observed no additional device memory")]
    ZeroPilotPeak,
    #[error("GPU VRAM percentage must be between 1 and 100, got {0}")]
    InvalidVramPercent(u32),
    #[error("resident GPU memory {resident_bytes} exceeds total memory {total_bytes}")]
    ResidentMemoryExceedsTotal { total_bytes: u64, resident_bytes: u64 },
    #[error(
        "{role:?} cannot fit one calibrated column: budget={budget_bytes}, resident={resident_bytes}, bytes_per_column={bytes_per_column}"
    )]
    InsufficientMemory {
        role: GpuDeviceRole,
        budget_bytes: u64,
        resident_bytes: u64,
        bytes_per_column: u64,
    },
    #[error("nonzero GPUs require a representative nonzero-GPU calibration")]
    MissingNonzeroCalibration,
    #[error("GPU fleet must contain at least one device")]
    ZeroGpuCount,
    #[error("{0:?} calibrated column width must be positive")]
    ZeroRoleWidth(GpuDeviceRole),
    #[error("GPU calibration arithmetic overflow")]
    ArithmeticOverflow,
    #[error("could not query current GPU allocator usage")]
    MemoryQueryFailed,
    #[error("GPU allocator high-water {peak_bytes} is below pilot baseline {baseline_bytes}")]
    InvalidPeakBaseline { peak_bytes: u64, baseline_bytes: u64 },
    #[error("GPU calibration operation identity must contain 32 bytes, got {0}")]
    InvalidOperationIdentityLength(usize),
    #[error("GPU calibration registry is frozen")]
    RegistryFrozen,
    #[error("frozen GPU plan has no node site {0:?}")]
    MissingPlanSite(GpuExecutionSiteKey),
    #[error("frozen GPU plan has no active column width at node site {0:?}")]
    ZeroPlanWidth(GpuExecutionSiteKey),
    #[error("frozen GPU plan has different nonzero widths at node site {0:?}")]
    PlanNonzeroWidthMismatch(GpuExecutionSiteKey),
    #[error("frozen GPU plan is invalid: {0}")]
    InvalidFrozenPlan(String),
}

#[derive(Default)]
struct RegistryState {
    profiles: HashMap<GpuCalibrationKey, Arc<GpuCalibrationProfile>>,
    frozen: bool,
}

/// Setup-time calibration cache shared by estimator and runtime.
///
/// Populate it during estimator or runtime preflight, then call [`Self::freeze`]
/// and pass the returned lock-free snapshot to hot execution. Clones share the
/// same setup registry and frozen state.
#[derive(Clone, Default)]
pub struct GpuCalibrationRegistry {
    state: Arc<RwLock<RegistryState>>,
}

impl GpuCalibrationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &GpuCalibrationKey) -> Option<Arc<GpuCalibrationProfile>> {
        self.state
            .read()
            .expect("GPU calibration registry lock poisoned")
            .profiles
            .get(key)
            .cloned()
    }

    pub fn insert(
        &self,
        key: GpuCalibrationKey,
        profile: GpuCalibrationProfile,
    ) -> Result<Option<Arc<GpuCalibrationProfile>>, GpuCalibrationError> {
        let mut state = self.state.write().expect("GPU calibration registry lock poisoned");
        if state.frozen {
            return Err(GpuCalibrationError::RegistryFrozen);
        }
        Ok(state.profiles.insert(key, Arc::new(profile)))
    }

    pub fn freeze(&self) -> FrozenGpuCalibrationRegistry {
        let mut state = self.state.write().expect("GPU calibration registry lock poisoned");
        state.frozen = true;
        FrozenGpuCalibrationRegistry { profiles: Arc::new(state.profiles.clone()) }
    }

    pub fn is_frozen(&self) -> bool {
        self.state.read().expect("GPU calibration registry lock poisoned").frozen
    }

    pub fn len(&self) -> usize {
        self.state.read().expect("GPU calibration registry lock poisoned").profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Immutable, lock-free calibration lookup used by the execution hot path.
#[derive(Clone, Default)]
pub struct FrozenGpuCalibrationRegistry {
    profiles: Arc<HashMap<GpuCalibrationKey, Arc<GpuCalibrationProfile>>>,
}

impl FrozenGpuCalibrationRegistry {
    pub fn get(&self, key: &GpuCalibrationKey) -> Option<Arc<GpuCalibrationProfile>> {
        self.profiles.get(key).cloned()
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gpu_execution_plan::{
            FrozenGpuPlan, GpuDeviceBudget, GpuExecutionSiteKey, GpuLayout, GpuNodeChoice,
            GpuPlanContract,
        },
        gpu_schedule::GpuColumnInterval,
    };
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
    fn fixed_plan_widths_never_read_current_residency() {
        let contract = GpuPlanContract {
            graph_specification_hash: [1; 32],
            backend_identity: "test".into(),
            logical_to_physical_devices: vec![0, 1],
            device_budgets: vec![
                GpuDeviceBudget {
                    device: 0,
                    device_bytes: 100,
                    pinned_host_bytes: 100,
                    host_bytes: 100,
                },
                GpuDeviceBudget {
                    device: 1,
                    device_bytes: 100,
                    pinned_host_bytes: 100,
                    host_bytes: 100,
                },
            ],
            shape_contract_hash: [2; 32],
            backend_revision: "test".into(),
        };
        let site = GpuExecutionSiteKey { site: 4, shape_class: 0, instance_class: 0 };
        let plan = FrozenGpuPlan::new(
            contract,
            vec![GpuLayout {
                id: 1,
                columns: 8,
                rows: 1,
                ring_dimension: 1,
                representation: "matrix".into(),
                instance_device_stride: 0,
                owner_intervals: vec![
                    GpuColumnInterval { device: 0, start: 0, end: 4 },
                    GpuColumnInterval { device: 1, start: 4, end: 8 },
                ],
            }],
            vec![],
            vec![GpuNodeChoice {
                loop_site: None,
                key: site,
                operation_identity: [3; 32],
                effective_operation: crate::gpu_column_policy::EffectiveGpuOperation::MatrixAdd,
                column_capability: crate::gpu_column_policy::ColumnCapability::SameColumns,
                output_layouts: vec![1],
                columns_per_job: vec![3, 5],
                implementation_variant: "test".into(),
                preimage_max_attempts: None,
            }],
        )
        .unwrap();
        let widths = GpuCalibrationProfile::fixed_widths_for_plan(&plan, site).unwrap();
        assert_eq!(widths, GpuColumnWidths { gpu0: 3, nonzero: Some(5) });
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

    fn calibration(pilot_columns: usize, pilot_peak_bytes: u64) -> GpuDeviceCalibration {
        GpuDeviceCalibration::from_pilot(pilot_columns, pilot_peak_bytes).unwrap()
    }

    fn device(name: &str, major: i32, minor: i32, bytes: usize) -> GpuDeviceIdentity {
        GpuDeviceIdentity {
            name: name.into(),
            compute_major: major,
            compute_minor: minor,
            total_global_memory: bytes,
        }
    }

    #[test]
    fn environment_identity_rejects_incompatible_fleets() {
        let ada = device("Example GPU", 8, 9, 24 << 30);
        let hopper = device("Example GPU", 9, 0, 24 << 30);
        let baseline = gpu_calibration_environment(&ada, 2, 80);
        assert_ne!(baseline, gpu_calibration_environment(&hopper, 2, 80));
        assert_ne!(baseline, gpu_calibration_environment(&ada, 1, 80));
        assert_ne!(baseline, gpu_calibration_environment(&ada, 2, 75));
        assert_ne!(
            baseline,
            gpu_calibration_environment(&device("Other GPU", 8, 9, 24 << 30), 2, 80)
        );
    }

    #[test]
    fn fixed_memory_profile_keeps_all_anchors_and_affine_intercept() {
        let device = device("Example GPU", 8, 9, 24 << 30);
        let context = GpuCalibrationContext::new(
            [7; 32],
            gpu_device_calibration_environment(&device, 3, "native-r42", 11, 80),
            3,
            "native-r42",
            11,
        );
        let profile = GpuMeasuredMemoryProfile::from_points(
            context,
            [
                GpuMeasuredMemoryPoint { columns: 2, peak_bytes: 13 },
                GpuMeasuredMemoryPoint { columns: 5, peak_bytes: 31 },
                GpuMeasuredMemoryPoint { columns: 9, peak_bytes: 56 },
            ],
        )
        .unwrap();
        assert_eq!(profile.points().len(), 3);
        assert_eq!(profile.estimate_bytes(5).unwrap(), 31);
        // 2 -> 13 and 5 -> 31 gives 19 at width 3.  Origin scaling would
        // incorrectly produce 19.5 (rounded to 20), proving the intercept is
        // retained by the bounded affine resolver.
        assert_eq!(profile.estimate_bytes(3).unwrap(), 19);
        assert_eq!(profile.estimate_bytes(7).unwrap(), 44);
        assert_eq!(
            profile.estimate_bytes(1),
            Err(GpuCalibrationError::MeasuredMemoryExtrapolation { columns: 1 })
        );
        assert_eq!(
            profile.estimate_bytes(10),
            Err(GpuCalibrationError::MeasuredMemoryExtrapolation { columns: 10 })
        );
    }

    #[test]
    fn fixed_memory_profile_rejects_bad_anchor_tables() {
        let context = GpuCalibrationContext::new([1; 32], [2; 32], 0, "rev", 1);
        assert_eq!(
            GpuMeasuredMemoryProfile::from_points(context.clone(), []),
            Err(GpuCalibrationError::NoMeasuredMemoryPoints)
        );
        assert_eq!(
            GpuMeasuredMemoryProfile::from_points(
                context.clone(),
                [
                    GpuMeasuredMemoryPoint { columns: 2, peak_bytes: 1 },
                    GpuMeasuredMemoryPoint { columns: 2, peak_bytes: 2 }
                ],
            ),
            Err(GpuCalibrationError::UnsortedMeasuredMemoryPoints)
        );
        assert_eq!(
            GpuMeasuredMemoryProfile::from_points(
                context,
                [GpuMeasuredMemoryPoint { columns: 0, peak_bytes: 1 }],
            ),
            Err(GpuCalibrationError::ZeroMeasuredMemoryColumns)
        );
    }

    #[test]
    fn fixed_memory_admission_uses_measured_anchors_only() {
        let context = GpuCalibrationContext::new([1; 32], [2; 32], 0, "rev", 1);
        let profile = GpuMeasuredMemoryProfile::from_points(
            context,
            [
                GpuMeasuredMemoryPoint { columns: 1, peak_bytes: 40 },
                GpuMeasuredMemoryPoint { columns: 4, peak_bytes: 100 },
            ],
        )
        .unwrap();
        assert_eq!(
            profile
                .largest_fitting_anchor(
                    GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 },
                    80,
                )
                .unwrap(),
            1
        );
        assert_eq!(
            profile.largest_fitting_anchor(
                GpuDeviceMemory { total_bytes: 100, resident_bytes: 100 },
                80,
            ),
            Err(GpuCalibrationError::NoFittingMeasuredMemoryAnchor)
        );
    }

    #[test]
    fn fixed_context_invalidates_device_revision_and_context_generation() {
        let gpu = device("Example GPU", 8, 9, 24 << 30);
        let environment = gpu_device_calibration_environment(&gpu, 0, "native-r42", 1, 80);
        assert_ne!(environment, gpu_device_calibration_environment(&gpu, 1, "native-r42", 1, 80));
        let same = GpuCalibrationContext::new([7; 32], environment.clone(), 0, "native-r42", 1);
        assert_ne!(
            same,
            GpuCalibrationContext::new([7; 32], environment.clone(), 1, "native-r42", 1)
        );
        assert_ne!(
            same,
            GpuCalibrationContext::new([7; 32], environment.clone(), 0, "native-r43", 1)
        );
        assert_ne!(same, GpuCalibrationContext::new([7; 32], environment, 0, "native-r42", 2));
        let changed_device = device("Other GPU", 8, 9, 24 << 30);
        assert_ne!(
            same.environment(),
            gpu_device_calibration_environment(&changed_device, 0, "native-r42", 1, 80).as_ref()
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
    fn calibration_key_preserves_bounds_after_column_normalization() {
        let matrix = ConcreteMatrixType {
            modulus: BigInt::from(257u16),
            ring_dimension: 16,
            rows: 3,
            columns: 29,
        };
        let kind = NodeKind::GadgetDecompose {
            base: IntExpr::constant(4),
            small: false,
            digit_count: IntExpr::constant(3),
        };
        let operation = |bound| {
            gpu_calibration_operation_identity(
                &kind,
                &[ConcreteWireType::Matrix(matrix.clone())],
                &[ConcreteWireType::Preimage {
                    matrix: matrix.clone(),
                    max_coefficient_bound: BigInt::from(bound),
                }],
                &ParamEnv::default(),
            )
            .unwrap()
        };

        let first = GpuCalibrationKey::new(operation(7), &b"same-environment"[..]);
        let second = GpuCalibrationKey::new(operation(8), &b"same-environment"[..]);
        assert_ne!(first, second);
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
        assert_ne!(
            constant_identity(ConstantMatrix::PowerOfBase {
                base: IntExpr::constant(4),
                exponent: IntExpr::constant(1),
            }),
            constant_identity(ConstantMatrix::PowerOfBase {
                base: IntExpr::constant(4),
                exponent: IntExpr::constant(2),
            })
        );
        assert_ne!(
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
    fn gpu0_and_nonzero_widths_use_separate_baselines() {
        let profile =
            GpuCalibrationProfile { gpu0: calibration(2, 200), nonzero: Some(calibration(4, 320)) };
        let widths = profile
            .derive_widths(
                GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 300 },
                Some(GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 80 }),
                80,
            )
            .unwrap();

        assert_eq!(widths, GpuColumnWidths { gpu0: 5, nonzero: Some(9) });
        assert_eq!(widths.columns_per_wave(3).unwrap(), 23);
        assert_eq!(widths.chunk_count(47, 3).unwrap(), 3);
    }

    #[test]
    fn shared_default_pool_cannot_justify_more_than_one_column() {
        let widths = GpuColumnWidths { gpu0: 17, nonzero: Some(29) };
        assert_eq!(
            widths.constrain_for_live_contexts(2, Some(1)),
            GpuColumnWidths { gpu0: 1, nonzero: Some(29) }
        );
        assert_eq!(
            widths.constrain_for_live_contexts(1, Some(3)),
            GpuColumnWidths { gpu0: 17, nonzero: Some(1) }
        );
        assert_eq!(widths.constrain_for_live_contexts(1, Some(1)), widths);
    }

    #[test]
    fn pilot_cost_rounds_up_and_width_derivation_fails_closed() {
        let calibration = calibration(3, 10);
        assert_eq!(calibration.bytes_per_column(), 4);
        let profile = GpuCalibrationProfile { gpu0: calibration, nonzero: None };
        assert_eq!(
            profile.derive_widths(
                GpuDeviceMemory { total_bytes: 100, resident_bytes: 78 },
                None,
                80,
            ),
            Err(GpuCalibrationError::InsufficientMemory {
                role: GpuDeviceRole::Gpu0,
                budget_bytes: 80,
                resident_bytes: 78,
                bytes_per_column: 4,
            })
        );
    }

    #[test]
    fn capped_waterfill_balances_equal_and_unequal_capacities() {
        assert_eq!(
            gpu_capped_waterfill_columns(GpuColumnWidths { gpu0: 100, nonzero: Some(100) }, 2, 176)
                .unwrap(),
            vec![88, 88]
        );
        assert_eq!(
            gpu_capped_waterfill_columns(GpuColumnWidths { gpu0: 20, nonzero: Some(100) }, 3, 176)
                .unwrap(),
            vec![20, 78, 78]
        );
        assert_eq!(
            gpu_capped_waterfill_columns(GpuColumnWidths { gpu0: 10, nonzero: Some(10) }, 4, 2)
                .unwrap(),
            vec![1, 1, 0, 0]
        );
        assert_eq!(
            gpu_capped_waterfill_columns(
                GpuColumnWidths { gpu0: 2, nonzero: Some(3) },
                3,
                usize::MAX
            )
            .unwrap(),
            vec![2, 3, 3]
        );
    }

    #[test]
    fn capped_waterfill_exhaustively_preserves_capacity_and_minimal_level() {
        for gpu_count in 1..=5 {
            for gpu0 in 1..=7 {
                for nonzero in 1..=7 {
                    let widths =
                        GpuColumnWidths { gpu0, nonzero: (gpu_count > 1).then_some(nonzero) };
                    let capacities = (0..gpu_count)
                        .map(|device| if device == 0 { gpu0 } else { nonzero })
                        .collect::<Vec<_>>();
                    let capacity = capacities.iter().sum::<usize>();
                    for requested in 0..=capacity + 2 {
                        let assigned =
                            gpu_capped_waterfill_columns(widths, gpu_count, requested).unwrap();
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
    fn manually_constructed_zero_role_widths_fail_closed() {
        assert_eq!(
            GpuColumnWidths { gpu0: 0, nonzero: None }.chunk_count(1, 1),
            Err(GpuCalibrationError::ZeroRoleWidth(GpuDeviceRole::Gpu0))
        );
        assert_eq!(
            GpuColumnWidths { gpu0: 1, nonzero: Some(0) }.chunk_count(2, 2),
            Err(GpuCalibrationError::ZeroRoleWidth(GpuDeviceRole::Nonzero))
        );
    }

    #[test]
    fn registry_has_exact_environment_hits_and_shared_misses() {
        let registry = GpuCalibrationRegistry::new();
        let same_operation_a = GpuCalibrationKey::new(&b"multiply"[..], &b"cuda-a"[..]);
        let same_operation_b = GpuCalibrationKey::new(&b"multiply"[..], &b"cuda-b"[..]);
        let profile = GpuCalibrationProfile { gpu0: calibration(1, 64), nonzero: None };

        assert!(registry.get(&same_operation_a).is_none());
        registry.insert(same_operation_a.clone(), profile.clone()).unwrap();
        assert_eq!(registry.get(&same_operation_a).as_deref(), Some(&profile));
        assert!(registry.get(&same_operation_b).is_none());

        let shared = registry.clone();
        let frozen = registry.freeze();
        assert!(shared.is_frozen());
        assert_eq!(shared.len(), 1);
        assert_eq!(frozen.get(&same_operation_a).as_deref(), Some(&profile));
        assert_eq!(frozen.len(), 1);
        assert_eq!(
            shared.insert(same_operation_b, profile),
            Err(GpuCalibrationError::RegistryFrozen)
        );
    }
}
