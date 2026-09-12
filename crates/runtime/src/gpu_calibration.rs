//! Setup-time calibration shared by the GPU estimator and runtime scheduler.
//!
//! Profiles associate pilot observations with supported native allocation classes.
//! Slopes provide candidate hints; admission uses complete requirements after
//! reserving retained outputs. Unvalidated observations remain diagnostic only.
//! Each active device is checked independently, with separate GPU 0 and nonzero
//! role observations and no cached live headroom or executable owners.

use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv, concretize_wire_type, encoding,
    node::{ConcatAxis, ConstantMatrix, IndexRange, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, NodeId, WireType},
};
use mxx_primitives::{
    matrix::gpu_dcrt_poly::{GpuPreparedRequest, GpuPreparedStorage},
    poly::dcrt::gpu::GpuDeviceIdentity,
};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
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

/// A contiguous width range with a monotone native temporary bound when known.
/// The identity includes actual parameter, format, preparation and range classes.
/// Unknown bounds describe diagnostic logical ranges only, never supported
/// native allocation classes. The bound contract identifies its memory metric:
/// a logical prepared-occupancy bound does not also bound physical provisioning,
/// or managed allocator overhead. Opaque CUDA growth is outside managed admission.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct GpuAllocationClass {
    pub identity: [u8; 32],
    pub bound_identity: Option<[u8; 32]>,
    pub minimum_columns: usize,
    pub maximum_columns: usize,
}

impl GpuAllocationClass {
    fn validate(self) -> Result<(), GpuCalibrationError> {
        if self.minimum_columns == 0 || self.minimum_columns > self.maximum_columns {
            return Err(GpuCalibrationError::InvalidAllocationClass);
        }
        Ok(())
    }

    fn contains(self, columns: usize) -> bool {
        (self.minimum_columns..=self.maximum_columns).contains(&columns)
    }
}

/// The provenance and units of a calibration observation. Prepared occupancy
/// measures claimed spans, not CUDA-pool growth or physical residency. Its
/// configuration identity describes the prepared layout and resource inventory,
/// never a particular executable token or a live allocation address.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub enum GpuCalibrationMetric {
    DefaultPoolIncrementalBytes,
    PreparedOccupiedSpanBytes { storage_configuration: [u8; 32] },
    AllocationFree,
}

/// Capacity in the same units as the pilot slope. Neither variant establishes
/// admission: prepared spans still require exact layout/resource fitting, and
/// independently bounded managed demand must fit its separate planning budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCandidateCapacity {
    DefaultPoolHeadroomBytes(u64),
    PreparedAvailableBytes(u64),
}

/// Exact semantic, environment, metric and allocation-class identity. Operation
/// remains the canonical dispatch identity; class metadata prevents its normalized shape
/// from reusing a pilot across native allocation or range boundaries.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct GpuCalibrationKey {
    operation: Arc<[u8]>,
    environment: Arc<[u8]>,
    class: GpuAllocationClass,
    metric: GpuCalibrationMetric,
}

impl GpuCalibrationKey {
    pub fn new(
        operation: impl Into<Arc<[u8]>>,
        environment: impl Into<Arc<[u8]>>,
        class: GpuAllocationClass,
        metric: GpuCalibrationMetric,
    ) -> Self {
        Self { operation: operation.into(), environment: environment.into(), class, metric }
    }

    pub fn operation(&self) -> &[u8] {
        &self.operation
    }

    pub fn environment(&self) -> &[u8] {
        &self.environment
    }

    pub fn class(&self) -> GpuAllocationClass {
        self.class
    }

    pub fn metric(&self) -> GpuCalibrationMetric {
        self.metric
    }
}

impl fmt::Debug for GpuCalibrationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuCalibrationKey")
            .field("operation_bytes", &self.operation.len())
            .field("environment_bytes", &self.environment.len())
            .field("class", &self.class)
            .field("metric", &self.metric)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCalibrationObservation {
    /// A declared zero-allocation view/no-op, not an unexplained zero pilot peak.
    AllocationFree,
    Allocating {
        metric: GpuCalibrationMetric,
        pilot_columns: usize,
        incremental_peak_bytes: u64,
        bytes_per_column_hint: u64,
        /// None preserves an unvalidated default-pool diagnostic sample without
        /// promoting its measured peak to proof. Prepared observations require Some.
        validated_bound_bytes: Option<u64>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuDeviceCalibration {
    class: GpuAllocationClass,
    observation: GpuCalibrationObservation,
}

impl GpuDeviceCalibration {
    /// Execute an isolated prepared pilot using the supplied production range
    /// runner. All claims are reserved before entry and all managed allocation
    /// domains require exact permits, including previously unused domains.
    /// Keep fixed baseline owners alive throughout this call. Native layout
    /// demand bounds the observation independently of its measured peak.
    ///
    /// Include every prepared store on this physical execution owner, with an
    /// empty request list for retained-only stores. Related rings share one
    /// actual simultaneous occupancy peak. Request groups are in production
    /// allocation order. Managed resource coverage and physical setup acceptance
    /// remain separate from this logical metric.
    pub fn measure_prepared<T>(
        class: GpuAllocationClass,
        pilot_columns: usize,
        storage_configuration: [u8; 32],
        groups: &[(&GpuPreparedStorage, &[GpuPreparedRequest])],
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<(Self, T), String> {
        Self::measure_prepared_with_steps(
            class,
            pilot_columns,
            storage_configuration,
            groups,
            &[],
            run,
        )
    }

    /// Like `measure_prepared`, with `steps`: exact claims the pilot's runner
    /// holds itself through ordered dispatch extensions (host-driven steps such
    /// as bounded sampling attempts). They count toward the declared demand
    /// bound but are not reserved up front, since the runner claims them.
    pub fn measure_prepared_with_steps<T>(
        class: GpuAllocationClass,
        pilot_columns: usize,
        storage_configuration: [u8; 32],
        groups: &[(&GpuPreparedStorage, &[GpuPreparedRequest])],
        steps: &[(&GpuPreparedStorage, &[GpuPreparedRequest])],
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<(Self, T), String> {
        class.validate().map_err(|error| error.to_string())?;
        if class.bound_identity.is_none() || !class.contains(pilot_columns) {
            return Err(
                "prepared pilot requires a native bound identity and a width in its class".into()
            );
        }
        let demands = groups
            .par_iter()
            .chain(steps.par_iter())
            .map(|(storage, requests)| storage.demand(requests).map(|demand| demand.device_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let demand = demands.into_iter().try_fold(0usize, |total, bytes| {
            total.checked_add(bytes).ok_or("prepared pilot demand overflow")
        })?;
        if demand == 0 {
            return Err(
                "allocating prepared pilot requires nonzero native device-span demand".into()
            );
        }
        // Allocation order can revisit a related store. Measure its backing
        // once while retaining separate ordered, disjoint reservation groups.
        let mut seen = HashSet::new();
        let stores = groups
            .iter()
            .filter_map(|(storage, _)| seen.insert(storage.identity()).then_some(*storage))
            .collect::<Vec<_>>();
        let (baseline, _) = GpuPreparedStorage::joint_occupancy(&stores, true)?;
        let mut reservations = groups
            .iter()
            .map(|(storage, requests)| {
                let mut reservation = storage.reserve(requests)?;
                reservation.require_all_resources()?;
                Ok(reservation)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let first = reservations.remove(0); // Joint reset rejects an empty inventory.
        let dispatch = first.enter(reservations)?;
        let result = run()?;
        drop(dispatch.finish()?);
        let (_, observed_peak) = GpuPreparedStorage::joint_occupancy(&stores, false)?;
        let peak = observed_peak
            .checked_sub(baseline)
            .ok_or("prepared pilot peak is below its retained baseline")?;
        let calibration = Self::from_pilot(
            class,
            pilot_columns,
            u64::try_from(peak).map_err(|_| "prepared pilot peak overflow")?,
            Some(u64::try_from(demand).map_err(|_| "prepared pilot demand overflow")?),
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration },
        )
        .map_err(|error| error.to_string())?;
        Ok((calibration, result))
    }

    /// The bound is supplied independently by checked native pilot preparation
    /// in the selected metric's units, never copied from the observed peak.
    /// Prepared observations require a bound and native layout identity; their
    /// peak counts actual occupied output/scratch spans above fixed preparation,
    /// with aliases and reused spans counted once. Reserved but unclaimed slots
    /// and physical backing provisioned at setup are not incremental occupancy.
    /// Default-pool observations may remain unvalidated diagnostics.
    pub fn from_pilot(
        class: GpuAllocationClass,
        pilot_columns: usize,
        incremental_peak_bytes: u64,
        pilot_bound_bytes: Option<u64>,
        metric: GpuCalibrationMetric,
    ) -> Result<Self, GpuCalibrationError> {
        class.validate()?;
        if pilot_columns == 0 {
            return Err(GpuCalibrationError::ZeroPilotColumns);
        }
        if !class.contains(pilot_columns) {
            return Err(GpuCalibrationError::WidthOutsideClass { columns: pilot_columns });
        }
        if metric == GpuCalibrationMetric::AllocationFree {
            return Err(GpuCalibrationError::AllocatingMetricRequired);
        }
        if incremental_peak_bytes == 0 {
            return Err(GpuCalibrationError::ZeroPilotPeak);
        }
        if matches!(metric, GpuCalibrationMetric::PreparedOccupiedSpanBytes { .. }) &&
            pilot_bound_bytes.is_none()
        {
            return Err(GpuCalibrationError::UnvalidatedAllocationClass);
        }
        if let Some(bound_bytes) = pilot_bound_bytes {
            if class.bound_identity.is_none() {
                return Err(GpuCalibrationError::UnvalidatedAllocationClass);
            }
            if incremental_peak_bytes > bound_bytes {
                return Err(GpuCalibrationError::PilotExceedsBound {
                    peak_bytes: incremental_peak_bytes,
                    bound_bytes,
                });
            }
        }
        let columns =
            u64::try_from(pilot_columns).map_err(|_| GpuCalibrationError::ArithmeticOverflow)?;
        let bytes_per_column_hint = incremental_peak_bytes.div_ceil(columns);
        Ok(Self {
            class,
            observation: GpuCalibrationObservation::Allocating {
                metric,
                pilot_columns,
                incremental_peak_bytes,
                bytes_per_column_hint,
                validated_bound_bytes: pilot_bound_bytes,
            },
        })
    }

    /// The production provider must establish zero additional physical
    /// allocations and logical claims for this view/no-op class. Prepared
    /// output/scratch claims do not qualify. This does not launch a pilot.
    pub fn allocation_free(class: GpuAllocationClass) -> Result<Self, GpuCalibrationError> {
        class.validate()?;
        if class.bound_identity.is_none() {
            return Err(GpuCalibrationError::UnvalidatedAllocationClass);
        }
        Ok(Self { class, observation: GpuCalibrationObservation::AllocationFree })
    }

    pub fn class(self) -> GpuAllocationClass {
        self.class
    }

    pub fn observation(self) -> GpuCalibrationObservation {
        self.observation
    }

    pub fn metric(self) -> GpuCalibrationMetric {
        match self.observation {
            GpuCalibrationObservation::AllocationFree => GpuCalibrationMetric::AllocationFree,
            GpuCalibrationObservation::Allocating { metric, .. } => metric,
        }
    }

    /// Validates only the observed peak against its independent declared bound
    /// in the same metric. It does not certify physical provisioning, native
    /// coverage, or a production invocation's complete resource fit.
    pub fn has_validated_bound(self) -> bool {
        self.class.bound_identity.is_some() &&
            match self.observation {
                GpuCalibrationObservation::AllocationFree => true,
                GpuCalibrationObservation::Allocating { validated_bound_bytes, .. } => {
                    validated_bound_bytes.is_some()
                }
            }
    }

    /// Derive only a slope candidate, capped by the remaining range and class.
    /// Prepared capacity is the internal free occupancy after retained owners
    /// are reserved; it must not be replaced by physical budget headroom.
    /// A candidate below the class minimum is returned for the active-owner
    /// planner to reject. This method never checks physical or native slot fit.
    pub fn candidate_width(
        self,
        capacity: GpuCandidateCapacity,
        remaining_columns: usize,
    ) -> Result<usize, GpuCalibrationError> {
        let candidate = match self.observation {
            GpuCalibrationObservation::AllocationFree => remaining_columns,
            GpuCalibrationObservation::Allocating { metric, bytes_per_column_hint, .. } => {
                let bytes = match (metric, capacity) {
                    (
                        GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                        GpuCandidateCapacity::DefaultPoolHeadroomBytes(bytes),
                    ) |
                    (
                        GpuCalibrationMetric::PreparedOccupiedSpanBytes { .. },
                        GpuCandidateCapacity::PreparedAvailableBytes(bytes),
                    ) => bytes,
                    _ => return Err(GpuCalibrationError::CandidateMetricMismatch { metric }),
                };
                let maximum = u64::try_from(remaining_columns.min(self.class.maximum_columns))
                    .map_err(|_| GpuCalibrationError::ArithmeticOverflow)?;
                usize::try_from((bytes / bytes_per_column_hint).min(maximum))
                    .map_err(|_| GpuCalibrationError::ArithmeticOverflow)?
            }
        };
        Ok(candidate.min(remaining_columns).min(self.class.maximum_columns))
    }
}

/// One active owner's invocation-local budget. Fixed preparation is already
/// charged when candidate headroom is observed. The exact budget snapshot also
/// includes every retained output owner, once, before temporary width search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuWidthAdmission {
    pub device: usize,
    pub remaining_columns: usize,
    /// Capacity in the pilot's units, used only as a slope hint. Prepared
    /// capacity excludes retained reservations; physical backing stays charged.
    pub candidate_capacity: GpuCandidateCapacity,
    pub budget_bytes: u64,
    pub charged_bytes_after_outputs: u64,
}

/// Requirements for one width in a provider-declared monotone allocation class.
/// Prepared requests use actual native storage, never a cached fit boolean or a
/// scalar payload estimate. All fixed/output reservations must already be held.
/// This covers the listed resources only; the invocation's finite resource seal
/// and managed capacity checks remain necessary before dispatch is permitted.
pub enum GpuTemporaryRequirement<'a> {
    /// Independently justified bound on remaining managed allocation demand.
    BoundedBytes(u64),
    Prepared {
        storage_configuration: [u8; 32],
        /// Independently bounded remaining managed demand, excluding
        /// all prepared backing already charged at setup.
        remaining_bound_bytes: u64,
        /// One complete request group per participating native storage owner.
        /// Keep the same ordered owners across all widths in this class.
        resources: Vec<(&'a GpuPreparedStorage, Vec<GpuPreparedRequest>)>,
    },
}

struct GpuRequirementFit {
    fits: bool,
    bounded_bytes: Option<u64>,
    storage_ids: Vec<u64>,
}

impl GpuTemporaryRequirement<'_> {
    fn fit(
        &self,
        calibration: GpuDeviceCalibration,
        available_managed_bytes: u64,
    ) -> Result<GpuRequirementFit, GpuCalibrationError> {
        match self {
            Self::BoundedBytes(bytes) => {
                if matches!(
                    calibration.metric(),
                    GpuCalibrationMetric::PreparedOccupiedSpanBytes { .. }
                ) {
                    return Err(GpuCalibrationError::PreparedNativeFitRequired);
                }
                if calibration.observation == GpuCalibrationObservation::AllocationFree &&
                    *bytes != 0
                {
                    return Err(GpuCalibrationError::AllocationFreeRequirement { bytes: *bytes });
                }
                Ok(GpuRequirementFit {
                    fits: *bytes <= available_managed_bytes,
                    bounded_bytes: Some(*bytes),
                    storage_ids: Vec::new(),
                })
            }
            Self::Prepared { storage_configuration, remaining_bound_bytes, resources } => {
                if calibration.metric() !=
                    (GpuCalibrationMetric::PreparedOccupiedSpanBytes {
                        storage_configuration: *storage_configuration,
                    })
                {
                    return Err(GpuCalibrationError::CalibrationMetricMismatch);
                }
                let mut unique = HashSet::with_capacity(resources.len());
                if resources.iter().any(|(storage, _)| !unique.insert(storage.identity())) {
                    return Err(GpuCalibrationError::NativeFit(
                        "duplicate prepared storage owner".into(),
                    ));
                }
                // These are host-only native layout/state queries. Evaluate every
                // participating context, including separate related CRT owners.
                let fits = resources
                    .par_iter()
                    .map(|(storage, requests)| {
                        storage.fits(requests).map_err(GpuCalibrationError::NativeFit)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(GpuRequirementFit {
                    fits: fits.into_iter().all(|fits| fits) &&
                        *remaining_bound_bytes <= available_managed_bytes,
                    bounded_bytes: Some(*remaining_bound_bytes),
                    storage_ids: resources.iter().map(|(storage, _)| storage.identity()).collect(),
                })
            }
        }
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuDeviceRole {
    Gpu0,
    Nonzero,
}

/// Reusable class-specific observations for active fleet roles. A missing role
/// is not a zero-width calibration and does not require a dummy pilot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuCalibrationProfile {
    pub gpu0: Option<GpuDeviceCalibration>,
    pub nonzero: Option<GpuDeviceCalibration>,
}

impl GpuCalibrationProfile {
    fn role(&self, device: usize) -> Result<GpuDeviceCalibration, GpuCalibrationError> {
        if device == 0 {
            self.gpu0.ok_or(GpuCalibrationError::MissingGpu0Calibration)
        } else {
            self.nonzero.ok_or(GpuCalibrationError::MissingNonzeroCalibration)
        }
    }

    /// Largest class-valid widths after complete output reservation. Requirements
    /// must cover a monotone interval of the exact production resource class.
    /// Native prepared fitting checks each storage's current layouts/availability;
    /// a byte slope only limits the candidate. The callback must not allocate on
    /// the GPU, submit, wait, or release retained preparation/output reservations.
    ///
    /// This is an advisory search, not an executable memory permit. After choosing
    /// widths, atomically reserve every native request and any separately bounded
    /// remaining demand. Dispatch additionally requires an accepted resource seal.
    /// The post-output charge is the accepted initial adjusted residency plus
    /// live managed allocation reservations. Prepared backing is included once;
    /// slot reuse adds no managed allocation charge. Later physical observations
    /// do not change this charge. Do not derive it from logical request bytes.
    pub fn derive_widths<'a>(
        &self,
        owners: &[GpuWidthAdmission],
        temporary_requirements: impl Fn(
            usize,
            &GpuAllocationClass,
            usize,
        )
            -> Result<GpuTemporaryRequirement<'a>, GpuCalibrationError>
        + Sync,
    ) -> Result<GpuColumnWidths, GpuCalibrationError> {
        if owners.is_empty() {
            return Err(GpuCalibrationError::ZeroGpuCount);
        }
        let mut devices = HashSet::with_capacity(owners.len());
        if owners.iter().any(|owner| !devices.insert(owner.device)) {
            return Err(GpuCalibrationError::DuplicateDevice);
        }
        let widths = owners
            .par_iter()
            .map(|owner| {
                if owner.remaining_columns == 0 {
                    return Ok(None);
                }
                let calibration = self.role(owner.device)?;
                if !calibration.has_validated_bound() {
                    return Err(GpuCalibrationError::UnvalidatedAllocationClass);
                }
                // A prepared observation cannot use physical headroom as its hint,
                // even if a scalar requirement callback would report zero bytes.
                if matches!(
                    calibration.metric(),
                    GpuCalibrationMetric::PreparedOccupiedSpanBytes { .. }
                ) && !matches!(
                    owner.candidate_capacity,
                    GpuCandidateCapacity::PreparedAvailableBytes(_)
                ) {
                    return Err(GpuCalibrationError::PreparedNativeFitRequired);
                }
                let available = owner
                    .budget_bytes
                    .checked_sub(owner.charged_bytes_after_outputs)
                    .ok_or(GpuCalibrationError::InvalidAdmission {
                    device: owner.device,
                    budget_bytes: owner.budget_bytes,
                    charged_bytes: owner.charged_bytes_after_outputs,
                })?;
                if let GpuCandidateCapacity::DefaultPoolHeadroomBytes(bytes) =
                    owner.candidate_capacity
                {
                    if bytes > owner.budget_bytes {
                        return Err(GpuCalibrationError::InvalidCandidateHeadroom {
                            device: owner.device,
                        });
                    }
                }
                let class = calibration.class;
                let candidate = calibration
                    .candidate_width(owner.candidate_capacity, owner.remaining_columns)?;
                let prepared = matches!(
                    calibration.metric(),
                    GpuCalibrationMetric::PreparedOccupiedSpanBytes { .. }
                );
                if owner.remaining_columns < class.minimum_columns ||
                    (!prepared && candidate < class.minimum_columns)
                {
                    return Err(GpuCalibrationError::InsufficientClassWidth {
                        device: owner.device,
                        minimum_columns: class.minimum_columns,
                        candidate_columns: candidate,
                    });
                }
                let requirement = |columns| {
                    temporary_requirements(owner.device, &class, columns)?
                        .fit(calibration, available)
                };
                let minimum = requirement(class.minimum_columns)?;
                if !minimum.fits {
                    return Err(match minimum.bounded_bytes {
                        Some(bytes) if bytes > available => {
                            GpuCalibrationError::InsufficientTemporaryMemory {
                                device: owner.device,
                                minimum_columns: class.minimum_columns,
                                requested_bytes: bytes,
                                available_bytes: available,
                            }
                        }
                        _ => GpuCalibrationError::InsufficientPreparedCapacity {
                            device: owner.device,
                            minimum_columns: class.minimum_columns,
                        },
                    });
                }
                // Retained outputs may consume the entire prepared capacity
                // while a range only writes those outputs. A zero slope hint
                // cannot reject a native-fitting minimum job. With no scratch
                // inventory, check the class's full range rather than forcing
                // an artificial one-column wave.
                let candidate = if prepared &&
                    minimum.storage_ids.is_empty() &&
                    minimum.bounded_bytes == Some(0)
                {
                    owner.remaining_columns.min(class.maximum_columns)
                } else {
                    candidate.max(class.minimum_columns)
                };
                let checked = |columns| {
                    let fit = requirement(columns)?;
                    if fit.storage_ids != minimum.storage_ids ||
                        fit.bounded_bytes.is_some() != minimum.bounded_bytes.is_some()
                    {
                        return Err(GpuCalibrationError::UnstableTemporaryRequirement {
                            device: owner.device,
                            columns,
                        });
                    }
                    if let (Some(bytes), Some(minimum_bytes)) =
                        (fit.bounded_bytes, minimum.bounded_bytes)
                    {
                        if bytes < minimum_bytes {
                            return Err(GpuCalibrationError::NonmonotoneTemporaryRequirement);
                        }
                    }
                    Ok(fit.fits)
                };
                let mut low = class.minimum_columns;
                let mut high = candidate;
                if checked(high)? {
                    low = high;
                } else {
                    high -= 1;
                }
                while low < high {
                    let middle = low + (high - low).div_ceil(2);
                    if checked(middle)? {
                        low = middle;
                    } else {
                        high = middle - 1;
                    }
                }
                if !checked(low)? {
                    return Err(GpuCalibrationError::UnstableTemporaryRequirement {
                        device: owner.device,
                        columns: low,
                    });
                }
                Ok(Some((owner.device, low, minimum.storage_ids)))
            })
            .collect::<Result<Vec<_>, GpuCalibrationError>>()?;
        let mut result = GpuColumnWidths { gpu0: None, nonzero: None };
        for &(device, width, _) in widths.iter().flatten() {
            if device == 0 {
                result.gpu0 = Some(width);
            } else {
                result.nonzero = Some(result.nonzero.map_or(width, |previous| previous.min(width)));
            }
        }
        // The shared nonzero role may choose a smaller width than an owner's
        // individual result. Check the actual shared width on every active owner
        // and reject changed native ownership before returning the plan hint.
        owners.par_iter().zip(&widths).try_for_each(|(owner, selected)| {
            let Some((device, _, storage_ids)) = selected else {
                return Ok(());
            };
            let width = if *device == 0 { result.gpu0 } else { result.nonzero }
                .expect("an active owner contributes a role width");
            let calibration = self.role(*device)?;
            let fit = temporary_requirements(*device, &calibration.class, width)?
                .fit(calibration, owner.budget_bytes - owner.charged_bytes_after_outputs)?;
            if !fit.fits || fit.storage_ids != *storage_ids {
                return Err(GpuCalibrationError::UnstableTemporaryRequirement {
                    device: *device,
                    columns: width,
                });
            }
            Ok(())
        })?;
        Ok(result)
    }

    /// Default-pool diagnostic slope candidates only: no output reservation or native bound
    /// is checked. These widths must never be treated as admitted production
    /// capacities. The budget is a planning target here: if its headroom is
    /// exhausted, still propose the smallest supported job and let actual CUDA
    /// allocation determine whether it can run. Managed admission in `derive_widths`
    /// remains strict. Absent role observations remain absent in the result.
    pub fn candidate_widths(
        &self,
        devices: &[GpuDeviceMemory],
        vram_percent: u32,
    ) -> Result<GpuColumnWidths, GpuCalibrationError> {
        if devices.is_empty() {
            return Err(GpuCalibrationError::ZeroGpuCount);
        }
        let widths = devices
            .par_iter()
            .enumerate()
            .map(|(device, memory)| {
                let calibration = if device == 0 { self.gpu0 } else { self.nonzero };
                let Some(calibration) = calibration else { return Ok(None) };
                let budget = memory.budget_bytes(vram_percent)?;
                let candidate = calibration.candidate_width(
                    GpuCandidateCapacity::DefaultPoolHeadroomBytes(
                        budget.saturating_sub(memory.resident_bytes),
                    ),
                    calibration.class.maximum_columns,
                )?;
                let candidate = candidate.max(calibration.class.minimum_columns);
                Ok(Some((device, candidate)))
            })
            .collect::<Result<Vec<_>, GpuCalibrationError>>()?;
        let mut result = GpuColumnWidths { gpu0: None, nonzero: None };
        for (device, width) in widths.into_iter().flatten() {
            if device == 0 {
                result.gpu0 = Some(width);
            } else {
                result.nonzero = Some(result.nonzero.map_or(width, |previous| previous.min(width)));
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuColumnWidths {
    pub gpu0: Option<usize>,
    pub nonzero: Option<usize>,
}

impl GpuColumnWidths {
    /// Expand role limits. The frozen ownership schedule must still restrict
    /// work to its admitted active owners; this is not an active-device mask.
    pub fn device_capacities(self, gpu_count: usize) -> Result<Vec<usize>, GpuCalibrationError> {
        self.columns_per_wave(gpu_count)?;
        Ok((0..gpu_count)
            .map(
                |device| {
                    if device == 0 { self.gpu0.unwrap_or(0) } else { self.nonzero.unwrap_or(0) }
                },
            )
            .collect())
    }

    pub fn columns_per_wave(self, gpu_count: usize) -> Result<usize, GpuCalibrationError> {
        if gpu_count == 0 {
            return Err(GpuCalibrationError::ZeroGpuCount);
        }
        if self.gpu0 == Some(0) {
            return Err(GpuCalibrationError::ZeroRoleWidth(GpuDeviceRole::Gpu0));
        }
        if self.nonzero == Some(0) {
            return Err(GpuCalibrationError::ZeroRoleWidth(GpuDeviceRole::Nonzero));
        }
        self.nonzero
            .unwrap_or(0)
            .checked_mul(gpu_count - 1)
            .and_then(|columns| columns.checked_add(self.gpu0.unwrap_or(0)))
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
        if columns_per_wave == 0 {
            return Err(GpuCalibrationError::NoActiveRole);
        }
        Ok(total_columns.div_ceil(columns_per_wave))
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
    #[error("GPU calibration pilot must contain at least one column")]
    ZeroPilotColumns,
    #[error("GPU allocating pilot observed zero increment in its declared memory metric")]
    ZeroPilotPeak,
    #[error("an allocating GPU pilot requires a pool or prepared-occupancy metric")]
    AllocatingMetricRequired,
    #[error("GPU calibration candidate capacity does not use {metric:?} units")]
    CandidateMetricMismatch { metric: GpuCalibrationMetric },
    #[error("prepared GPU widths require the complete native span/resource fit predicate")]
    PreparedNativeFitRequired,
    #[error("GPU {device} cannot fit prepared resources at class minimum {minimum_columns}")]
    InsufficientPreparedCapacity { device: usize, minimum_columns: usize },
    #[error("native prepared GPU fit failed: {0}")]
    NativeFit(String),
    #[error("GPU VRAM percentage must be between 1 and 100, got {0}")]
    InvalidVramPercent(u32),
    #[error("resident GPU memory {resident_bytes} exceeds total memory {total_bytes}")]
    ResidentMemoryExceedsTotal { total_bytes: u64, resident_bytes: u64 },
    #[error("GPU allocation class must have a nonempty positive width interval")]
    InvalidAllocationClass,
    #[error("GPU width {columns} is outside the declared allocation class")]
    WidthOutsideClass { columns: usize },
    #[error("GPU allocation class lacks a validated native bound")]
    UnvalidatedAllocationClass,
    #[error("GPU pilot peak {peak_bytes} exceeds its bound {bound_bytes} in the declared metric")]
    PilotExceedsBound { peak_bytes: u64, bound_bytes: u64 },
    #[error("GPU {device} candidate {candidate_columns} is below class minimum {minimum_columns}")]
    InsufficientClassWidth { device: usize, minimum_columns: usize, candidate_columns: usize },
    #[error(
        "GPU {device} cannot fit class minimum {minimum_columns}: requested={requested_bytes}, available={available_bytes}"
    )]
    InsufficientTemporaryMemory {
        device: usize,
        minimum_columns: usize,
        requested_bytes: u64,
        available_bytes: u64,
    },
    #[error("GPU {device} post-output charge {charged_bytes} exceeds budget {budget_bytes}")]
    InvalidAdmission { device: usize, budget_bytes: u64, charged_bytes: u64 },
    #[error("GPU {device} candidate headroom exceeds its budget")]
    InvalidCandidateHeadroom { device: usize },
    #[error("allocation-free GPU class requires {bytes} additional bytes")]
    AllocationFreeRequirement { bytes: u64 },
    #[error("GPU temporary requirements are not monotone within their allocation class")]
    NonmonotoneTemporaryRequirement,
    #[error("GPU {device} temporary requirements changed while validating width {columns}")]
    UnstableTemporaryRequirement { device: usize, columns: usize },
    #[error("GPU width admission contains a duplicate device")]
    DuplicateDevice,
    #[error("GPU calibration profile and key have different allocation classes")]
    AllocationClassMismatch,
    #[error(
        "GPU calibration profile and key have different memory metrics or prepared configurations"
    )]
    CalibrationMetricMismatch,
    #[error("GPU calibration profile contains no observed role")]
    EmptyProfile,
    #[error("GPU 0 requires its own calibration")]
    MissingGpu0Calibration,
    #[error("a nonempty GPU operation has no active role capacity")]
    NoActiveRole,
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
    #[error("GPU {device} has {live_contexts} execution owners; calibration requires one")]
    NonexclusiveContext { device: usize, live_contexts: usize },
    #[error("GPU allocator high-water {peak_bytes} is below pilot baseline {baseline_bytes}")]
    InvalidPeakBaseline { peak_bytes: u64, baseline_bytes: u64 },
    #[error("GPU calibration operation identity must contain 32 bytes, got {0}")]
    InvalidOperationIdentityLength(usize),
    #[error("GPU calibration registry is frozen")]
    RegistryFrozen,
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

impl From<FrozenGpuCalibrationRegistry> for GpuCalibrationRegistry {
    fn from(snapshot: FrozenGpuCalibrationRegistry) -> Self {
        // A new setup registry owns only profiles. Existing frozen readers keep
        // their snapshot; native slots, values and residency are never copied.
        Self {
            state: Arc::new(RwLock::new(RegistryState {
                profiles: (*snapshot.profiles).clone(),
                frozen: false,
            })),
        }
    }
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
        mut profile: GpuCalibrationProfile,
    ) -> Result<Option<Arc<GpuCalibrationProfile>>, GpuCalibrationError> {
        let mut state = self.state.write().expect("GPU calibration registry lock poisoned");
        if state.frozen {
            return Err(GpuCalibrationError::RegistryFrozen);
        }
        key.class.validate()?;
        if profile.gpu0.is_none() && profile.nonzero.is_none() {
            return Err(GpuCalibrationError::EmptyProfile);
        }
        if [profile.gpu0, profile.nonzero].into_iter().flatten().any(|role| role.class != key.class)
        {
            return Err(GpuCalibrationError::AllocationClassMismatch);
        }
        if [profile.gpu0, profile.nonzero]
            .into_iter()
            .flatten()
            .any(|role| role.metric() != key.metric)
        {
            return Err(GpuCalibrationError::CalibrationMetricMismatch);
        }
        if let Some(previous) = state.profiles.get(&key) {
            profile.gpu0 = profile.gpu0.or(previous.gpu0);
            profile.nonzero = profile.nonzero.or(previous.nonzero);
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

    // Synthetic arithmetic model for pure tests, not a native GPU bound.
    fn test_class() -> GpuAllocationClass {
        GpuAllocationClass {
            identity: [17; 32],
            bound_identity: Some([18; 32]),
            minimum_columns: 1,
            maximum_columns: 64,
        }
    }

    fn calibration(pilot_columns: usize, pilot_peak_bytes: u64) -> GpuDeviceCalibration {
        GpuDeviceCalibration::from_pilot(
            test_class(),
            pilot_columns,
            pilot_peak_bytes,
            Some(256 * pilot_columns as u64),
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        )
        .unwrap()
    }

    fn admissions(memory: &[GpuDeviceMemory]) -> Vec<GpuWidthAdmission> {
        memory
            .iter()
            .enumerate()
            .map(|(device, memory)| {
                let budget_bytes = memory.budget_bytes(80).unwrap();
                GpuWidthAdmission {
                    device,
                    remaining_columns: 64,
                    candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(
                        budget_bytes.saturating_sub(memory.resident_bytes),
                    ),
                    budget_bytes,
                    charged_bytes_after_outputs: memory.resident_bytes,
                }
            })
            .collect()
    }

    fn linear_requirement(
        device: usize,
        class: &GpuAllocationClass,
        columns: usize,
    ) -> Result<GpuTemporaryRequirement<'static>, GpuCalibrationError> {
        assert_eq!(*class, test_class());
        Ok(GpuTemporaryRequirement::BoundedBytes(
            columns as u64 * if device == 0 { 100 } else { 80 },
        ))
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

        let first = GpuCalibrationKey::new(
            operation(7),
            &b"same-environment"[..],
            test_class(),
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        );
        let second = GpuCalibrationKey::new(
            operation(8),
            &b"same-environment"[..],
            test_class(),
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        );
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
    fn gpu0_and_nonzero_widths_use_separate_baselines() {
        let profile = GpuCalibrationProfile {
            gpu0: Some(calibration(2, 200)),
            nonzero: Some(calibration(4, 320)),
        };
        let memory = [
            GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 300 },
            GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 80 },
        ];
        let widths = profile.derive_widths(&admissions(&memory), linear_requirement).unwrap();
        assert_eq!(widths, GpuColumnWidths { gpu0: Some(5), nonzero: Some(9) });
        assert_eq!(widths.columns_per_wave(3).unwrap(), 23);
        assert_eq!(widths.chunk_count(47, 3).unwrap(), 3);
        assert_eq!(profile.candidate_widths(&memory, 80).unwrap(), widths);
    }

    #[test]
    fn test_budget_exhaustion_keeps_diagnostic_jobs_but_never_grants_managed_admission() {
        let profile = GpuCalibrationProfile {
            gpu0: Some(calibration(2, 200)),
            nonzero: Some(calibration(4, 320)),
        };
        for resident in [800, 900, 1000] {
            let memory = [GpuDeviceMemory { total_bytes: 1000, resident_bytes: resident }; 3];
            assert_eq!(
                profile.candidate_widths(&memory, 80).unwrap(),
                GpuColumnWidths { gpu0: Some(1), nonzero: Some(1) }
            );
            assert!(profile.derive_widths(&admissions(&memory), linear_requirement).is_err());
        }
        let class = GpuAllocationClass { minimum_columns: 4, maximum_columns: 8, ..test_class() };
        let calibration = GpuDeviceCalibration::from_pilot(
            class,
            4,
            320,
            None,
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        )
        .unwrap();
        let profile = GpuCalibrationProfile { gpu0: Some(calibration), nonzero: None };
        assert_eq!(
            profile
                .candidate_widths(&[GpuDeviceMemory { total_bytes: 1000, resident_bytes: 900 }], 80)
                .unwrap(),
            GpuColumnWidths { gpu0: Some(4), nonzero: None }
        );
    }

    #[test]
    fn nonzero_profile_uses_every_devices_current_headroom() {
        let profile = GpuCalibrationProfile {
            gpu0: Some(calibration(2, 200)),
            nonzero: Some(calibration(4, 320)),
        };
        let mut memory = vec![
            GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 300 },
            GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 80 },
            GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 640 },
            GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 400 },
        ];
        assert_eq!(
            profile.derive_widths(&admissions(&memory), linear_requirement).unwrap(),
            GpuColumnWidths { gpu0: Some(5), nonzero: Some(2) }
        );
        memory[2].resident_bytes = 721;
        assert!(matches!(
            profile.derive_widths(&admissions(&memory), linear_requirement),
            Err(GpuCalibrationError::InsufficientClassWidth { device: 2, .. })
        ));
        memory[2].resident_bytes = 0;
        assert_eq!(
            profile.derive_widths(&admissions(&memory), linear_requirement).unwrap(),
            GpuColumnWidths { gpu0: Some(5), nonzero: Some(5) }
        );
        assert_eq!(
            profile.derive_widths(&[], linear_requirement),
            Err(GpuCalibrationError::ZeroGpuCount)
        );
    }

    #[test]
    fn pilot_cost_rounds_up_and_width_derivation_fails_closed() {
        let calibration = calibration(3, 10);
        assert!(matches!(
            calibration.observation(),
            GpuCalibrationObservation::Allocating { bytes_per_column_hint: 4, .. }
        ));
        let profile = GpuCalibrationProfile { gpu0: Some(calibration), nonzero: None };
        assert_eq!(
            profile.derive_widths(
                &admissions(&[GpuDeviceMemory { total_bytes: 100, resident_bytes: 78 }]),
                |_, _, columns| Ok(GpuTemporaryRequirement::BoundedBytes(4 * columns as u64))
            ),
            Err(GpuCalibrationError::InsufficientClassWidth {
                device: 0,
                minimum_columns: 1,
                candidate_columns: 0,
            })
        );
    }

    #[test]
    fn exact_width_search_accounts_for_reserved_outputs_and_nonlinear_temporaries() {
        let class = GpuAllocationClass { maximum_columns: 16, ..test_class() };
        let profile = GpuCalibrationProfile {
            gpu0: Some(
                GpuDeviceCalibration::from_pilot(
                    class,
                    1,
                    8,
                    Some(32),
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                )
                .unwrap(),
            ),
            nonzero: None,
        };
        let owner = GpuWidthAdmission {
            device: 0,
            remaining_columns: 40,
            candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(1_000),
            budget_bytes: 1_000,
            charged_bytes_after_outputs: 900,
        };
        let widths = profile
            .derive_widths(&[owner], |_, queried_class, columns| {
                assert_eq!(*queried_class, class);
                assert!(class.contains(columns));
                Ok(GpuTemporaryRequirement::BoundedBytes(5 + 3 * columns as u64 * columns as u64))
            })
            .unwrap();
        assert_eq!(widths.gpu0, Some(5));
        assert_eq!(widths.nonzero, None);
        // Output-only work fits even when retained outputs consume the budget:
        // their pilot slope limits the candidate, but is not charged twice.
        let full_output = GpuWidthAdmission { charged_bytes_after_outputs: 1_000, ..owner };
        assert_eq!(
            profile
                .derive_widths(&[full_output], |_, _, _| Ok(GpuTemporaryRequirement::BoundedBytes(
                    0
                )))
                .unwrap()
                .gpu0,
            Some(16)
        );
        let wide_class = GpuAllocationClass { maximum_columns: usize::MAX, ..test_class() };
        let wide = GpuCalibrationProfile {
            gpu0: Some(
                GpuDeviceCalibration::from_pilot(
                    wide_class,
                    1,
                    1,
                    Some(8),
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                )
                .unwrap(),
            ),
            nonzero: None,
        };
        let wide_owner = GpuWidthAdmission {
            device: 0,
            remaining_columns: usize::MAX,
            candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(usize::MAX as u64),
            budget_bytes: usize::MAX as u64,
            charged_bytes_after_outputs: usize::MAX as u64 - 2,
        };
        assert_eq!(
            wide.derive_widths(&[wide_owner], |_, _, columns| Ok(
                GpuTemporaryRequirement::BoundedBytes(columns as u64)
            ))
            .unwrap()
            .gpu0,
            Some(2)
        );
    }

    #[test]
    fn allocation_class_boundaries_and_minimum_jobs_are_respected() {
        for (minimum_columns, maximum_columns) in [(1, 4), (5, 20)] {
            let class = GpuAllocationClass { minimum_columns, maximum_columns, ..test_class() };
            let profile = GpuCalibrationProfile {
                gpu0: Some(
                    GpuDeviceCalibration::from_pilot(
                        class,
                        minimum_columns,
                        1,
                        Some(64),
                        GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                    )
                    .unwrap(),
                ),
                nonzero: None,
            };
            let owner = GpuWidthAdmission {
                device: 0,
                remaining_columns: 25,
                candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(100),
                budget_bytes: 100,
                charged_bytes_after_outputs: 10,
            };
            assert_eq!(
                profile
                    .derive_widths(&[owner], |_, _, columns| {
                        assert!((minimum_columns..=maximum_columns).contains(&columns));
                        Ok(GpuTemporaryRequirement::BoundedBytes(0))
                    })
                    .unwrap()
                    .gpu0,
                Some(maximum_columns)
            );
            let too_short = GpuWidthAdmission { remaining_columns: minimum_columns - 1, ..owner };
            if minimum_columns > 1 {
                assert!(matches!(
                    profile.derive_widths(&[too_short], |_, _, _| panic!(
                        "invalid width reached native query"
                    )),
                    Err(GpuCalibrationError::InsufficientClassWidth { .. })
                ));
            }
            assert!(matches!(
                profile.derive_widths(&[owner], |_, _, _| Ok(
                    GpuTemporaryRequirement::BoundedBytes(91)
                )),
                Err(GpuCalibrationError::InsufficientTemporaryMemory { .. })
            ));
        }
    }

    #[test]
    fn allocation_free_profiles_use_active_logical_ranges_without_a_pilot() {
        let free = GpuDeviceCalibration::allocation_free(test_class()).unwrap();
        assert_eq!(free.observation(), GpuCalibrationObservation::AllocationFree);
        let profile = GpuCalibrationProfile { gpu0: None, nonzero: Some(free) };
        let owners = [
            GpuWidthAdmission {
                device: 0,
                remaining_columns: 0,
                candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(0),
                budget_bytes: 0,
                charged_bytes_after_outputs: 0,
            },
            GpuWidthAdmission {
                device: 2,
                remaining_columns: 9,
                candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(0),
                budget_bytes: 100,
                charged_bytes_after_outputs: 100,
            },
            GpuWidthAdmission {
                device: 3,
                remaining_columns: 6,
                candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(0),
                budget_bytes: 100,
                charged_bytes_after_outputs: 100,
            },
        ];
        let widths = profile
            .derive_widths(&owners, |device, _, _| {
                assert_ne!(device, 0);
                Ok(GpuTemporaryRequirement::BoundedBytes(0))
            })
            .unwrap();
        assert_eq!(widths, GpuColumnWidths { gpu0: None, nonzero: Some(6) });
        assert_eq!(widths.device_capacities(4).unwrap(), vec![0, 6, 6, 6]);
        assert_eq!(
            profile
                .derive_widths(&owners[0..1], |_, _, _| panic!("inactive owner queried"))
                .unwrap(),
            GpuColumnWidths { gpu0: None, nonzero: None }
        );
        assert!(matches!(
            profile.derive_widths(&owners, |_, _, _| Ok(GpuTemporaryRequirement::BoundedBytes(1))),
            Err(GpuCalibrationError::AllocationFreeRequirement { bytes: 1 })
        ));
        let active_zero = GpuWidthAdmission { remaining_columns: 1, ..owners[0] };
        assert_eq!(
            profile.derive_widths(&[active_zero], |_, _, _| Ok(
                GpuTemporaryRequirement::BoundedBytes(0)
            )),
            Err(GpuCalibrationError::MissingGpu0Calibration)
        );
    }

    #[test]
    fn unvalidated_observations_remain_diagnostics_and_invalid_pilots_fail() {
        let unknown = GpuAllocationClass { bound_identity: None, ..test_class() };
        let observation = GpuDeviceCalibration::from_pilot(
            unknown,
            2,
            10,
            None,
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        )
        .unwrap();
        assert!(!observation.has_validated_bound());
        let profile = GpuCalibrationProfile { gpu0: Some(observation), nonzero: None };
        let memory = [GpuDeviceMemory { total_bytes: 100, resident_bytes: 10 }];
        assert_eq!(profile.candidate_widths(&memory, 80).unwrap().gpu0, Some(14));
        assert_eq!(
            profile.derive_widths(&admissions(&memory), |_, _, _| panic!(
                "unverified observation queried"
            )),
            Err(GpuCalibrationError::UnvalidatedAllocationClass)
        );
        assert_eq!(
            GpuDeviceCalibration::allocation_free(unknown),
            Err(GpuCalibrationError::UnvalidatedAllocationClass)
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(
                unknown,
                1,
                10,
                Some(20),
                GpuCalibrationMetric::DefaultPoolIncrementalBytes
            ),
            Err(GpuCalibrationError::UnvalidatedAllocationClass)
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(
                test_class(),
                0,
                10,
                Some(20),
                GpuCalibrationMetric::DefaultPoolIncrementalBytes
            ),
            Err(GpuCalibrationError::ZeroPilotColumns)
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(
                test_class(),
                1,
                0,
                Some(20),
                GpuCalibrationMetric::DefaultPoolIncrementalBytes
            ),
            Err(GpuCalibrationError::ZeroPilotPeak)
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(
                test_class(),
                1,
                21,
                Some(20),
                GpuCalibrationMetric::DefaultPoolIncrementalBytes
            ),
            Err(GpuCalibrationError::PilotExceedsBound { peak_bytes: 21, bound_bytes: 20 })
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(
                test_class(),
                65,
                10,
                Some(20),
                GpuCalibrationMetric::DefaultPoolIncrementalBytes
            ),
            Err(GpuCalibrationError::WidthOutsideClass { columns: 65 })
        );
        assert_eq!(
            GpuDeviceCalibration::allocation_free(GpuAllocationClass {
                minimum_columns: 0,
                ..test_class()
            }),
            Err(GpuCalibrationError::InvalidAllocationClass)
        );
        let unvalidated = GpuDeviceCalibration::from_pilot(
            test_class(),
            1,
            10,
            None,
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        )
        .unwrap();
        assert!(!unvalidated.has_validated_bound());
        assert_eq!(
            GpuCalibrationProfile { gpu0: Some(unvalidated), nonzero: None }
                .derive_widths(&admissions(&memory), |_, _, _| Ok(
                    GpuTemporaryRequirement::BoundedBytes(0)
                )),
            Err(GpuCalibrationError::UnvalidatedAllocationClass)
        );
        let large_peak = GpuDeviceCalibration::from_pilot(
            unknown,
            2,
            u64::MAX,
            None,
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        )
        .unwrap();
        assert!(matches!(large_peak.observation(), GpuCalibrationObservation::Allocating {
            bytes_per_column_hint, ..
        } if bytes_per_column_hint == u64::MAX / 2 + 1));
    }

    #[test]
    fn admission_rejects_duplicates_invalid_charges_and_missing_active_roles() {
        let profile = GpuCalibrationProfile { gpu0: Some(calibration(1, 100)), nonzero: None };
        let owner = GpuWidthAdmission {
            device: 0,
            remaining_columns: 4,
            candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(500),
            budget_bytes: 500,
            charged_bytes_after_outputs: 0,
        };
        assert_eq!(
            profile.derive_widths(&[owner, owner], linear_requirement),
            Err(GpuCalibrationError::DuplicateDevice)
        );
        assert!(matches!(
            profile.derive_widths(
                &[GpuWidthAdmission { charged_bytes_after_outputs: 501, ..owner }],
                linear_requirement
            ),
            Err(GpuCalibrationError::InvalidAdmission { .. })
        ));
        assert!(matches!(
            profile.derive_widths(
                &[GpuWidthAdmission {
                    candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(501),
                    ..owner
                }],
                linear_requirement
            ),
            Err(GpuCalibrationError::InvalidCandidateHeadroom { .. })
        ));
        assert_eq!(
            profile.derive_widths(&[GpuWidthAdmission { device: 2, ..owner }], linear_requirement),
            Err(GpuCalibrationError::MissingNonzeroCalibration)
        );
        assert_eq!(
            profile.derive_widths(&[owner], |_, _, columns| Ok(
                GpuTemporaryRequirement::BoundedBytes(10 - columns as u64)
            )),
            Err(GpuCalibrationError::NonmonotoneTemporaryRequirement)
        );
    }

    #[test]
    fn prepared_pilot_requires_nonzero_claims_and_an_independent_native_bound() {
        let metric =
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration: [31; 32] };
        let class = test_class();
        let observed = GpuDeviceCalibration::from_pilot(class, 3, 10, Some(12), metric).unwrap();
        assert!(observed.has_validated_bound());
        assert_eq!(observed.metric(), metric);
        assert_eq!(
            observed.observation(),
            GpuCalibrationObservation::Allocating {
                metric,
                pilot_columns: 3,
                incremental_peak_bytes: 10,
                bytes_per_column_hint: 4,
                validated_bound_bytes: Some(12),
            }
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(class, 3, 0, Some(12), metric),
            Err(GpuCalibrationError::ZeroPilotPeak),
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(class, 3, 10, None, metric),
            Err(GpuCalibrationError::UnvalidatedAllocationClass),
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(
                GpuAllocationClass { bound_identity: None, ..class },
                3,
                10,
                Some(12),
                metric,
            ),
            Err(GpuCalibrationError::UnvalidatedAllocationClass),
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(class, 3, 13, Some(12), metric),
            Err(GpuCalibrationError::PilotExceedsBound { peak_bytes: 13, bound_bytes: 12 }),
        );
        assert_eq!(
            GpuDeviceCalibration::from_pilot(
                class,
                3,
                10,
                Some(12),
                GpuCalibrationMetric::AllocationFree,
            ),
            Err(GpuCalibrationError::AllocatingMetricRequired),
        );
    }

    #[test]
    fn prepared_width_hint_uses_internal_capacity_and_rejects_pool_units() {
        let metric =
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration: [31; 32] };
        let prepared =
            GpuDeviceCalibration::from_pilot(test_class(), 3, 10, Some(12), metric).unwrap();
        assert_eq!(
            prepared.candidate_width(GpuCandidateCapacity::PreparedAvailableBytes(39), 50),
            Ok(9),
        );
        assert_eq!(
            prepared.candidate_width(GpuCandidateCapacity::PreparedAvailableBytes(39), 4),
            Ok(4),
        );
        assert_eq!(
            prepared.candidate_width(GpuCandidateCapacity::PreparedAvailableBytes(u64::MAX), 100),
            Ok(test_class().maximum_columns),
        );
        assert_eq!(
            prepared.candidate_width(GpuCandidateCapacity::DefaultPoolHeadroomBytes(39), 50),
            Err(GpuCalibrationError::CandidateMetricMismatch { metric }),
        );
        let pool = calibration(3, 10);
        assert_eq!(
            pool.candidate_width(GpuCandidateCapacity::PreparedAvailableBytes(39), 50),
            Err(GpuCalibrationError::CandidateMetricMismatch {
                metric: GpuCalibrationMetric::DefaultPoolIncrementalBytes,
            }),
        );
        let profile = GpuCalibrationProfile { gpu0: Some(prepared), nonzero: None };
        assert_eq!(
            profile
                .candidate_widths(&[GpuDeviceMemory { total_bytes: 100, resident_bytes: 80 }], 80),
            Err(GpuCalibrationError::CandidateMetricMismatch { metric }),
        );
    }

    #[test]
    fn scalar_admission_cannot_certify_prepared_slots_and_skips_inactive_roles() {
        let prepared = GpuDeviceCalibration::from_pilot(
            test_class(),
            1,
            8,
            Some(16),
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration: [31; 32] },
        )
        .unwrap();
        let profile = GpuCalibrationProfile { gpu0: None, nonzero: Some(prepared) };
        let inactive = GpuWidthAdmission {
            device: 0,
            remaining_columns: 0,
            candidate_capacity: GpuCandidateCapacity::DefaultPoolHeadroomBytes(0),
            budget_bytes: 100,
            charged_bytes_after_outputs: 100,
        };
        let active = GpuWidthAdmission { device: 2, remaining_columns: 4, ..inactive };
        assert_eq!(
            profile.derive_widths(&[inactive, active], |_, _, _| {
                panic!("prepared resources reached a scalar proof")
            }),
            Err(GpuCalibrationError::PreparedNativeFitRequired),
        );
        assert_eq!(
            profile.derive_widths(&[inactive], |_, _, _| panic!("inactive owner queried")),
            Ok(GpuColumnWidths { gpu0: None, nonzero: None }),
        );
        assert_eq!(
            prepared.candidate_width(GpuCandidateCapacity::PreparedAvailableBytes(32), 4),
            Ok(4),
        );
        assert_eq!(
            profile.derive_widths(
                &[GpuWidthAdmission { remaining_columns: 4, ..inactive }],
                |_, _, _| panic!("missing role queried"),
            ),
            Err(GpuCalibrationError::MissingGpu0Calibration),
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_pilot_measures_claims_above_fixed_baseline() {
        use mxx_primitives::{
            matrix::{
                PolyMatrix,
                gpu_dcrt_poly::{
                    GpuDCRTPolyMatrix, GpuPreparedSlotKind, GpuPreparedWorkspaceLayout,
                },
            },
            poly::{
                PolyParams,
                dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
            },
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(4);
        assert!(columns > 0);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let expected = original.slice(0, 2, 0, 1).negate_out_of_place();
        let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let transfer = params.rns_transfer_workspace(params.crt_depth() - 1, 2, 1).unwrap();
        let storage = GpuPreparedStorage::new(
            (0..3).map(|_| GpuDCRTPolyMatrix::zero(&params, 2, columns)).collect(),
            Some(&[
                transfer,
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                },
            ]),
        )
        .unwrap();
        let fixed_claim = storage.slot_identity(0).unwrap().matrix_request(2, columns, true);
        let dispatch = storage.reserve(&[fixed_claim]).unwrap().enter(Vec::new()).unwrap();
        let fixed = GpuDCRTPolyMatrix::zero(&params, 2, columns);
        drop(dispatch.finish().unwrap());
        params.fence_released_memory();
        let baseline = storage.occupancy().unwrap().occupied_bytes();
        assert!(baseline > 0);
        let pilot_claim = storage.slot_identity(1).unwrap().matrix_request(2, 1, true);
        let readback = [
            storage.slot_identity(3).unwrap().workspace_request(transfer.bytes, transfer.alignment),
            storage.slot_identity(4).unwrap().workspace_request(0, 1),
        ];
        let layout = params.matrix_allocation_bytes(params.crt_depth() - 1, 2, 1, true).unwrap();
        let demanded = storage.demand(&[pilot_claim, readback[0], readback[1]]).unwrap();
        assert_eq!(demanded.device_bytes, layout.data_bytes + layout.aux_bytes + transfer.bytes);
        assert_eq!(demanded.resource_slots, 1);
        assert_eq!(demanded.pinned_bytes, 0);
        assert!(storage.demand(&[pilot_claim, pilot_claim]).is_err());
        let class = GpuAllocationClass { maximum_columns: columns, ..test_class() };
        let run = || input.column_view(0..1)?.negate(None);
        let (calibration, pilot) = GpuDeviceCalibration::measure_prepared(
            class,
            1,
            [61; 32],
            &[(&storage, &[pilot_claim])],
            run,
        )
        .unwrap();
        let GpuCalibrationObservation::Allocating {
            incremental_peak_bytes,
            validated_bound_bytes,
            ..
        } = calibration.observation()
        else {
            panic!("prepared negate is allocating")
        };
        assert_eq!(incremental_peak_bytes, (layout.data_bytes + layout.aux_bytes) as u64);
        assert_eq!(validated_bound_bytes, Some(incremental_peak_bytes));
        assert!(
            storage.occupancy().unwrap().occupied_high_water_bytes() >=
                baseline + incremental_peak_bytes as usize
        );
        // Use the same production range runner with the next pre-reserved output.
        let production_claim = storage.slot_identity(2).unwrap().matrix_request(2, 1, true);
        let dispatch = storage.reserve(&[production_claim]).unwrap().enter(Vec::new()).unwrap();
        let output = run().unwrap();
        drop(dispatch.finish().unwrap());
        for value in [&pilot, &output] {
            let dispatch = storage.reserve(&readback).unwrap().enter(Vec::new()).unwrap();
            assert_eq!(value.to_cpu_matrix(), expected);
            drop(dispatch.finish().unwrap());
        }
        drop(fixed);
        assert!(pilot.release().is_err(), "a previously unused event domain must require permits");
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_pilot_measures_joint_related_ring_claims() {
        use mxx_primitives::{
            matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
            poly::{
                PolyParams,
                dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
            },
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(4);
        assert!(columns > 0);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let related = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0[..1].to_vec(),
            4,
            params.device_ids(),
            Some(1),
            Some(&params),
            None,
        );
        let stores = [&params, &related]
            .into_iter()
            .map(|params| {
                GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(params, 1, columns)], None)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let claims = stores
            .iter()
            .map(|store| [store.slot_identity(0).unwrap().matrix_request(1, 1, true)])
            .collect::<Vec<_>>();
        let groups = stores
            .iter()
            .zip(&claims)
            .map(|(store, claims)| (store, claims.as_slice()))
            .collect::<Vec<_>>();
        let expected = groups
            .iter()
            .map(|(store, claims)| store.demand(claims).unwrap().device_bytes)
            .sum::<usize>() as u64;
        let class = GpuAllocationClass { maximum_columns: columns, ..test_class() };
        params.fence_released_memory();
        let failed = GpuDeviceCalibration::measure_prepared(class, 1, [62; 32], &groups, || {
            let first = GpuDCRTPolyMatrix::zero(&params, 1, 1);
            let second = GpuDCRTPolyMatrix::zero(&related, 1, 1);
            drop((first, second));
            Err::<(), _>("injected pilot failure".to_owned())
        });
        assert!(failed.is_err());
        params.fence_released_memory();
        let (calibration, outputs) =
            GpuDeviceCalibration::measure_prepared(class, 1, [62; 32], &groups, || {
                Ok((
                    GpuDCRTPolyMatrix::zero(&params, 1, 1),
                    GpuDCRTPolyMatrix::zero(&related, 1, 1),
                ))
            })
            .unwrap();
        let GpuCalibrationObservation::Allocating {
            incremental_peak_bytes,
            validated_bound_bytes,
            ..
        } = calibration.observation()
        else {
            panic!("joint prepared pilot is allocating")
        };
        assert_eq!(incremental_peak_bytes, expected);
        assert_eq!(validated_bound_bytes, Some(expected));
        assert!(groups.iter().all(|(store, claims)| !store.fits(claims).unwrap()));
        drop(outputs);
        params.fence_released_memory();
        let inventory = stores.iter().collect::<Vec<_>>();
        assert_eq!(GpuPreparedStorage::joint_occupancy(&inventory, true).unwrap(), (0, 0));
        assert!(groups.iter().all(|(store, claims)| store.fits(claims).unwrap()));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_width_fit_uses_logical_slots_with_fully_charged_backing() {
        use mxx_primitives::{
            matrix::{
                PolyMatrix,
                gpu_dcrt_poly::{
                    GpuDCRTPolyMatrix, GpuPreparedSlotKind, GpuPreparedWorkspaceLayout,
                },
            },
            poly::{
                PolyParams,
                dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
            },
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let maximum = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(8);
        assert!(maximum >= 4);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let scratch_limit = maximum / 2;
        let mut storage = GpuPreparedStorage::new(
            (0..2).into_par_iter().map(|_| GpuDCRTPolyMatrix::zero(&params, 2, maximum)).collect(),
            Some(&[GpuPreparedWorkspaceLayout {
                bytes: 8 * scratch_limit,
                alignment: 8,
                kind: GpuPreparedSlotKind::BatchWorkspace,
            }]),
        )
        .unwrap();
        let output = storage.slot_identity(0).unwrap();
        let scratch = storage.slot_identity(1).unwrap();
        let workspace = storage.slot_identity(2).unwrap();
        // Observe a real private one-column pilot. The native layout supplies
        // the independent logical bound; no pool peak is used as a bound.
        let dispatch = storage
            .reserve(&[scratch.matrix_request(2, 1, true)])
            .unwrap()
            .enter(Vec::new())
            .unwrap();
        let pilot = GpuDCRTPolyMatrix::zero(&params, 2, 1);
        drop(dispatch.finish().unwrap());
        let peak = storage.occupancy().unwrap().occupied_high_water_bytes();
        let layout = params.matrix_allocation_bytes(1, 2, 1, true).unwrap();
        let configuration = [41; 32];
        let calibration = GpuDeviceCalibration::from_pilot(
            GpuAllocationClass { maximum_columns: maximum, ..test_class() },
            1,
            peak as u64,
            Some((layout.data_bytes + layout.aux_bytes) as u64),
            GpuCalibrationMetric::PreparedOccupiedSpanBytes {
                storage_configuration: configuration,
            },
        )
        .unwrap();
        drop(pilot);
        params.fence_released_memory();
        storage.reset_occupied_high_water().unwrap();
        let retained = storage.reserve(&[output.matrix_request(2, maximum, true)]).unwrap();
        let capacity = storage.occupancy().unwrap();
        let owner = GpuWidthAdmission {
            device: 0,
            remaining_columns: maximum,
            candidate_capacity: GpuCandidateCapacity::PreparedAvailableBytes(
                (capacity.requested_capacity_bytes() - capacity.reserved_bytes()) as u64,
            ),
            // Synthetic ledger values exercise independent dimensions, not a
            // claim that this test has physically sealed its CUDA execution.
            budget_bytes: 100,
            charged_bytes_after_outputs: 100,
        };
        assert!(
            calibration.candidate_width(owner.candidate_capacity, maximum).unwrap() > scratch_limit
        );
        let profile = GpuCalibrationProfile { gpu0: Some(calibration), nonzero: None };
        // This synthetic width provider uses real typed native requests. The
        // workspace limit, rather than the matrix byte slope, cuts the width.
        let requests = |columns| {
            vec![
                scratch.matrix_request(2, columns, true),
                workspace.workspace_request(8 * columns, 8),
            ]
        };
        let requirements = |_, _: &GpuAllocationClass, columns| {
            Ok(GpuTemporaryRequirement::Prepared {
                remaining_bound_bytes: 0,
                storage_configuration: configuration,
                resources: vec![(&storage, requests(columns))],
            })
        };
        assert_eq!(
            profile.derive_widths(&[owner], requirements).unwrap().gpu0,
            Some(scratch_limit)
        );
        assert!(!storage.fits(&[output.matrix_request(2, 1, true)]).unwrap());
        // A returned width is advisory. Competing native ownership invalidates
        // it; both the native reservation and a new width search must reject.
        let competing = storage.reserve(&requests(scratch_limit)).unwrap();
        assert!(storage.reserve(&requests(scratch_limit)).is_err());
        assert!(matches!(
            profile.derive_widths(
                &[GpuWidthAdmission { charged_bytes_after_outputs: 0, ..owner }],
                requirements
            ),
            Err(GpuCalibrationError::InsufficientPreparedCapacity { device: 0, .. })
        ));
        drop(competing);
        assert_eq!(
            profile.derive_widths(&[owner], requirements).unwrap().gpu0,
            Some(scratch_limit)
        );
        assert_eq!(
            profile.derive_widths(
                &[GpuWidthAdmission { charged_bytes_after_outputs: 101, ..owner }],
                |_, _, _| panic!("invalid physical charge reached native fit")
            ),
            Err(GpuCalibrationError::InvalidAdmission {
                device: 0,
                budget_bytes: 100,
                charged_bytes: 101
            })
        );
        assert_eq!(
            profile.derive_widths(&[owner], |_, _, _| Ok(GpuTemporaryRequirement::BoundedBytes(0))),
            Err(GpuCalibrationError::PreparedNativeFitRequired)
        );
        assert_eq!(
            profile.derive_widths(&[owner], |_, _, columns| Ok(
                GpuTemporaryRequirement::Prepared {
                    remaining_bound_bytes: 0,
                    storage_configuration: [42; 32],
                    resources: vec![(&storage, requests(columns))],
                }
            )),
            Err(GpuCalibrationError::CalibrationMetricMismatch)
        );
        assert!(matches!(
            profile.derive_widths(&[owner], |_, _, columns| Ok(
                GpuTemporaryRequirement::Prepared {
                    remaining_bound_bytes: 0,
                    storage_configuration: configuration,
                    resources: vec![(&storage, requests(columns)), (&storage, Vec::new())],
                }
            )),
            Err(GpuCalibrationError::NativeFit(_))
        ));
        drop(retained);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_width_fit_checks_each_execution_owner_and_all_contexts() {
        use mxx_primitives::{
            matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
            poly::{
                PolyParams,
                dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
            },
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let maximum = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(6);
        assert!(maximum >= 4);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        // Three independent execution owners on the locally selected GPU.
        // This exercises native-owner fitting, not physical three-GPU behavior.
        let storages = (0..3)
            .into_par_iter()
            .map(|_| {
                let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
                GpuPreparedStorage::new(
                    (0..maximum)
                        .into_par_iter()
                        .map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 1))
                        .collect(),
                    None,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let configuration = [43; 32];
        let profile_entry = GpuDeviceCalibration::from_pilot(
            GpuAllocationClass { maximum_columns: maximum, ..test_class() },
            1,
            1,
            Some(1),
            GpuCalibrationMetric::PreparedOccupiedSpanBytes {
                storage_configuration: configuration,
            },
        )
        .unwrap();
        // The unit's synthetic slope deliberately leaves native capacity as the
        // limiting condition. Production pilot evidence is tested separately.
        let profile =
            GpuCalibrationProfile { gpu0: Some(profile_entry), nonzero: Some(profile_entry) };
        let owners = (0..3)
            .map(|device| GpuWidthAdmission {
                device,
                remaining_columns: maximum,
                candidate_capacity: GpuCandidateCapacity::PreparedAvailableBytes(maximum as u64),
                budget_bytes: 100,
                charged_bytes_after_outputs: 100,
            })
            .collect::<Vec<_>>();
        let requests = |device: usize, start: usize, end: usize| {
            (start..end)
                .map(|index| {
                    storages[device].slot_identity(index).unwrap().matrix_request(1, 1, true)
                })
                .collect::<Vec<_>>()
        };
        let retained = (1..3)
            .map(|device| {
                storages[device].reserve(&requests(device, maximum - device, maximum)).unwrap()
            })
            .collect::<Vec<_>>();
        let requirements = |device, _: &GpuAllocationClass, columns| {
            Ok(GpuTemporaryRequirement::Prepared {
                remaining_bound_bytes: 0,
                storage_configuration: configuration,
                resources: vec![(&storages[device], requests(device, 0, columns))],
            })
        };
        let widths = profile.derive_widths(&owners, requirements).unwrap();
        assert_eq!(widths, GpuColumnWidths { gpu0: Some(maximum), nonzero: Some(maximum - 2) });
        // Every context in one owner's group participates. A second context's
        // retained slot cannot be ignored just because the first context fits.
        assert!(matches!(
            profile.derive_widths(&owners[0..1], |_, _, columns| Ok(
                GpuTemporaryRequirement::Prepared {
                    remaining_bound_bytes: 0,
                    storage_configuration: configuration,
                    resources: vec![
                        (&storages[0], requests(0, 0, columns)),
                        (&storages[2], requests(2, maximum - 1, maximum))
                    ],
                }
            )),
            Err(GpuCalibrationError::InsufficientPreparedCapacity { device: 0, .. })
        ));
        // Width-dependent substitution of the native owner violates the frozen
        // resource class even when each queried request fits independently.
        assert!(matches!(
            profile.derive_widths(&owners[0..1], |_, _, columns| {
                let device = usize::from(columns != 1);
                Ok(GpuTemporaryRequirement::Prepared {
                    remaining_bound_bytes: 0,
                    storage_configuration: configuration,
                    resources: vec![(&storages[device], requests(device, 0, 1))],
                })
            }),
            Err(GpuCalibrationError::UnstableTemporaryRequirement { .. })
        ));
        drop(retained);
        assert_eq!(
            profile.derive_widths(&owners, requirements).unwrap(),
            GpuColumnWidths { gpu0: Some(maximum), nonzero: Some(maximum) }
        );
    }

    #[test]
    fn registry_separates_pool_prepared_and_storage_configurations() {
        let registry = GpuCalibrationRegistry::new();
        let class = test_class();
        let pool_metric = GpuCalibrationMetric::DefaultPoolIncrementalBytes;
        let prepared_metric =
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration: [31; 32] };
        let other_configuration =
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration: [32; 32] };
        let key = |metric| GpuCalibrationKey::new(&b"op"[..], &b"environment"[..], class, metric);
        let profile = |metric| GpuCalibrationProfile {
            gpu0: Some(GpuDeviceCalibration::from_pilot(class, 2, 10, Some(20), metric).unwrap()),
            nonzero: None,
        };
        let pool = profile(pool_metric);
        let prepared = profile(prepared_metric);
        registry.insert(key(pool_metric), pool.clone()).unwrap();
        assert!(registry.get(&key(prepared_metric)).is_none());
        registry.insert(key(prepared_metric), prepared.clone()).unwrap();
        assert!(registry.get(&key(other_configuration)).is_none());
        assert_eq!(key(prepared_metric).metric(), prepared_metric);
        assert_eq!(registry.len(), 2);
        for (metric, conflicting) in [
            (pool_metric, prepared.clone()),
            (prepared_metric, pool.clone()),
            (other_configuration, prepared.clone()),
            (GpuCalibrationMetric::AllocationFree, prepared.clone()),
        ] {
            assert_eq!(
                registry.insert(key(metric), conflicting),
                Err(GpuCalibrationError::CalibrationMetricMismatch),
            );
        }
        assert_eq!(
            registry.insert(
                key(prepared_metric),
                GpuCalibrationProfile { gpu0: prepared.gpu0, nonzero: pool.gpu0 }
            ),
            Err(GpuCalibrationError::CalibrationMetricMismatch),
        );
        assert_eq!(registry.len(), 2);
        let frozen = registry.freeze();
        assert_eq!(frozen.get(&key(pool_metric)).as_deref(), Some(&pool));
        assert_eq!(frozen.get(&key(prepared_metric)).as_deref(), Some(&prepared));
        assert!(frozen.get(&key(other_configuration)).is_none());
    }

    #[test]
    fn capped_waterfill_balances_equal_and_unequal_capacities() {
        assert_eq!(gpu_capped_waterfill_columns(&[100, 100], 176).unwrap(), vec![88, 88]);
        assert_eq!(gpu_capped_waterfill_columns(&[20, 100, 100], 176).unwrap(), vec![20, 78, 78]);
        assert_eq!(gpu_capped_waterfill_columns(&[10; 4], 2).unwrap(), vec![1, 1, 0, 0]);
        assert_eq!(gpu_capped_waterfill_columns(&[2, 3, 3], usize::MAX).unwrap(), vec![2, 3, 3]);
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
    fn manually_constructed_zero_role_widths_fail_closed() {
        assert_eq!(
            GpuColumnWidths { gpu0: Some(0), nonzero: None }.chunk_count(1, 1),
            Err(GpuCalibrationError::ZeroRoleWidth(GpuDeviceRole::Gpu0))
        );
        assert_eq!(
            GpuColumnWidths { gpu0: Some(1), nonzero: Some(0) }.chunk_count(2, 2),
            Err(GpuCalibrationError::ZeroRoleWidth(GpuDeviceRole::Nonzero))
        );
    }

    #[test]
    fn registry_has_exact_environment_hits_and_shared_misses() {
        let registry = GpuCalibrationRegistry::new();
        let same_operation_a = GpuCalibrationKey::new(
            &b"multiply"[..],
            &b"cuda-a"[..],
            test_class(),
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        );
        let same_operation_b = GpuCalibrationKey::new(
            &b"multiply"[..],
            &b"cuda-b"[..],
            test_class(),
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        );
        let profile = GpuCalibrationProfile { gpu0: Some(calibration(1, 64)), nonzero: None };

        assert!(registry.get(&same_operation_a).is_none());
        registry.insert(same_operation_a.clone(), profile.clone()).unwrap();
        assert_eq!(registry.get(&same_operation_a).as_deref(), Some(&profile));
        assert!(registry.get(&same_operation_b).is_none());

        let unknown = GpuAllocationClass { bound_identity: None, ..test_class() };
        assert!(
            registry
                .get(&GpuCalibrationKey::new(
                    &b"multiply"[..],
                    &b"cuda-a"[..],
                    unknown,
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes
                ))
                .is_none()
        );
        assert!(
            registry
                .get(&GpuCalibrationKey::new(
                    &b"multiply"[..],
                    &b"cuda-a"[..],
                    GpuAllocationClass { maximum_columns: 4, ..test_class() },
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes
                ))
                .is_none()
        );
        assert!(
            registry
                .get(&GpuCalibrationKey::new(
                    &b"multiply"[..],
                    &b"cuda-a"[..],
                    GpuAllocationClass { identity: [19; 32], ..test_class() },
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes
                ))
                .is_none()
        );
        assert_eq!(
            registry.insert(
                GpuCalibrationKey::new(
                    &b"multiply"[..],
                    &b"cuda-a"[..],
                    unknown,
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes
                ),
                profile.clone()
            ),
            Err(GpuCalibrationError::AllocationClassMismatch)
        );
        assert_eq!(
            registry.insert(
                same_operation_a.clone(),
                GpuCalibrationProfile { gpu0: None, nonzero: None }
            ),
            Err(GpuCalibrationError::EmptyProfile)
        );

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
    #[test]
    fn registry_merges_independently_observed_roles() {
        let registry = GpuCalibrationRegistry::new();
        let key = GpuCalibrationKey::new(
            &b"operation"[..],
            &b"environment"[..],
            test_class(),
            GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        );
        let gpu0 = calibration(1, 100);
        let nonzero = calibration(1, 80);
        registry
            .insert(key.clone(), GpuCalibrationProfile { gpu0: Some(gpu0), nonzero: None })
            .unwrap();
        registry
            .insert(key.clone(), GpuCalibrationProfile { gpu0: None, nonzero: Some(nonzero) })
            .unwrap();
        assert_eq!(
            registry.get(&key).as_deref(),
            Some(&GpuCalibrationProfile { gpu0: Some(gpu0), nonzero: Some(nonzero) })
        );
        assert_eq!(registry.len(), 1);
        let inactive = GpuColumnWidths { gpu0: None, nonzero: None };
        assert_eq!(inactive.device_capacities(3).unwrap(), vec![0, 0, 0]);
        assert_eq!(inactive.chunk_count(0, 3).unwrap(), 0);
        assert_eq!(inactive.chunk_count(1, 3), Err(GpuCalibrationError::NoActiveRole));
    }
}
