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
    types::ConcreteWireType,
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
            NodeKind::PreimageSample { .. } |
            NodeKind::GadgetDecompose { .. } |
            NodeKind::MatrixScale { .. } |
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

/// Canonical per-primitive identity shared by estimator collection and runtime
/// preflight. Selector-only loop values and hash-domain tags do not affect the
/// allocation/kernel path and are deliberately erased; concrete types retain
/// every loop-dependent shape and bound that does affect it.
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
            _ => {}
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
    match &mut shape_kind {
        NodeKind::ConstantMatrix { matrix_type, .. } => {
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
        NodeKind::MatrixNegate => {
            if let Some(input) = argument_types.first_mut() {
                one_column(input);
            }
            normalize_output(&mut output_types);
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Add | MatrixBinaryOp::Subtract) => {
            argument_types.iter_mut().for_each(one_column);
            normalize_output(&mut output_types);
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) | NodeKind::MatrixMulSmallRhs => {
            if let Some(rhs) = argument_types.get_mut(1) {
                one_column(rhs);
            }
            normalize_output(&mut output_types);
        }
        NodeKind::MatrixMulAccumulate { has_bias, .. } => {
            let product_argument_count = argument_types.len() - usize::from(*has_bias);
            for rhs in (1..product_argument_count).step_by(2) {
                one_column(&mut argument_types[rhs]);
            }
            if *has_bias {
                one_column(argument_types.last_mut().expect("bias argument exists"));
            }
            normalize_output(&mut output_types);
        }
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
