//! Setup-time calibration shared by the GPU estimator and runtime scheduler.
//!
//! A profile records only the column-scaled allocation cost observed by a
//! pilot operation. The scheduler combines that reusable cost with the
//! current resident allocation on each device role. GPU 0 is deliberately
//! separate because it may own setup data that is absent from the other,
//! otherwise-identical GPUs.

use mxx_ir_core::{
    IntExpr, ParamEnv, encoding,
    node::{ConcatAxis, ConstantMatrix, IndexRange, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::poly::dcrt::gpu::GpuDeviceIdentity;
use serde::Serialize;
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, RwLock},
};

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
        NodeKind::ConstantMatrix {
            value: ConstantMatrix::Zero |
                ConstantMatrix::Identity |
                ConstantMatrix::UnitRow { .. } |
                ConstantMatrix::Gadget { .. },
            ..
        } | NodeKind::UniformResidueSample { .. } |
            NodeKind::UniformIntervalSample { .. } |
            NodeKind::GaussianSample { .. } |
            NodeKind::HashSample { .. } |
            NodeKind::GadgetTrapdoor { .. } |
            NodeKind::PreimageSample { .. } |
            NodeKind::FamilyPreimageSample { .. } |
            NodeKind::GadgetDecompose { .. } |
            NodeKind::MatrixScale { .. } |
            NodeKind::RingAutomorphism { .. } |
            NodeKind::MatrixNegate |
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
            ConcreteWireType::Family { element, .. } => matrix_type(element),
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

pub fn gpu_matrix_multiply_scales_left(
    left_rows: usize,
    left_columns: usize,
    right_rows: usize,
    right_columns: usize,
) -> bool {
    (right_rows, right_columns) == (1, 1) && (left_rows, left_columns) != (1, 1)
}

/// Canonical per-primitive identity shared by estimator collection and runtime
/// preflight. Automorphism selectors, selector-only loop values, and hash-domain
/// tags do not affect the allocation/kernel path and are deliberately erased;
/// concrete types retain every loop-dependent shape and bound that does affect it.
pub fn gpu_calibration_operation_identity(
    kind: &NodeKind,
    concrete_argument_types: &[ConcreteWireType],
    concrete_output_types: &[ConcreteWireType],
    bindings: &ParamEnv,
) -> Result<[u8; 32], String> {
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
            ConcreteWireType::Family { element, .. } => one_column(element),
            _ => {}
        }
    }
    fn matrix_type(ty: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
        match ty {
            ConcreteWireType::Family { element, .. } => matrix_type(element),
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
        NodeKind::PreimageSample { matrix_type, .. } |
        NodeKind::FamilyPreimageSample { matrix_type, .. } => {
            matrix_type.columns = IntExpr::constant(1);
            if let Some(target) = argument_types.get_mut(2) {
                one_column(target);
            }
            normalize_output(&mut output_types);
        }
        NodeKind::GadgetDecompose { .. } |
        NodeKind::MatrixScale { .. } |
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
    if let NodeKind::HashSample {
        tag_prefix,
        tag_expressions,
        tag_decimal_expressions,
        tag_u64_le_expressions,
        ..
    } = &mut shape_kind
    {
        tag_prefix.fill(0);
        for expression in tag_expressions
            .iter_mut()
            .chain(tag_decimal_expressions.iter_mut())
            .chain(tag_u64_le_expressions.iter_mut())
        {
            *expression = IntExpr::constant(0);
        }
    }

    encoding::hash_canonical(&OperationIdentity {
        kind: &shape_kind,
        concrete_argument_types: &argument_types,
        concrete_output_types: &output_types,
        bindings: &shape_bindings,
    })
    .map_err(|error| error.to_string())
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

/// Column-scaled memory cost measured by one pilot execution on one device.
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

/// Reusable pilot results for GPU 0 and the representative nonzero GPU.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuCalibrationProfile {
    pub gpu0: GpuDeviceCalibration,
    pub nonzero: Option<GpuDeviceCalibration>,
}

impl GpuCalibrationProfile {
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
    use mxx_ir_core::types::ConcreteMatrixType;
    use num_bigint::BigInt;

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
