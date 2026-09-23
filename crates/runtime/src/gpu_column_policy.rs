//! Pure column-capability and range-lowering helpers.
//!
//! This module deliberately contains no allocator, device query, profiling, or
//! backend state.  It is the single source of truth for translating a planned
//! output column range into the input ranges needed by an operation.

use mxx_ir_core::{
    expr::IntExpr,
    node::{ConcatAxis, ConstantMatrix, LoopInputMode, MatrixBinaryOp, NodeKind},
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
    /// A zero-copy projection of an existing native owner.  Alias operations
    /// participate in the same frozen placement as their source but do not
    /// launch a kernel or allocate a destination.
    NativeAlias,
    HostOrControl,
    /// A future or otherwise unlisted operation. It must be rejected during
    /// planning instead of silently taking a host/GPU-0 fallback.
    Unsupported,
}

/// Explicit allowlist of operations understood by the column planner. The
/// classifier is intentionally separate from [`ColumnCapability`]: two
/// operations can share a capability while still having different range
/// lowering semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum EffectiveGpuOperation {
    GeneratedConstant,
    SingleDeviceConstant,
    UniformResidueSample,
    UniformIntervalSample,
    GaussianSample,
    HashSample,
    HashIntFamily,
    TrapdoorSample,
    GadgetTrapdoor,
    LiftIntegerToConstantPolynomial,
    TrapdoorPublic,
    ExtractCoefficient,
    ThresholdDecode,
    PackPolynomialCoefficients,
    PolynomialFromValues,
    PolynomialValues,
    PreimageSample,
    GadgetDecompose,
    MatrixScale,
    MatrixNegate,
    RingAutomorphism,
    ModulusSwitch,
    ModulusReduce,
    CenteredRebase,
    CenteredRoundDivide,
    BlockModSwitch,
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

impl EffectiveGpuOperation {
    /// Every effective operation that can be produced by the validated IR
    /// classifier.  `Unsupported` is deliberately absent: it is a sentinel
    /// for malformed/future input and must never acquire a warmup profile.
    /// Keep this inventory next to the classifier so adding an operation is a
    /// compile-time-visible change to the warmup coverage contract.
    pub const ALL: &'static [Self] = &[
        Self::GeneratedConstant,
        Self::SingleDeviceConstant,
        Self::UniformResidueSample,
        Self::UniformIntervalSample,
        Self::GaussianSample,
        Self::HashSample,
        Self::HashIntFamily,
        Self::TrapdoorSample,
        Self::GadgetTrapdoor,
        Self::LiftIntegerToConstantPolynomial,
        Self::TrapdoorPublic,
        Self::ExtractCoefficient,
        Self::ThresholdDecode,
        Self::PackPolynomialCoefficients,
        Self::PolynomialFromValues,
        Self::PolynomialValues,
        Self::PreimageSample,
        Self::GadgetDecompose,
        Self::MatrixScale,
        Self::MatrixNegate,
        Self::RingAutomorphism,
        Self::ModulusSwitch,
        Self::ModulusReduce,
        Self::CenteredRebase,
        Self::CenteredRoundDivide,
        Self::BlockModSwitch,
        Self::RnsModUp,
        Self::RnsModDown,
        Self::CrtRecompose,
        Self::MatrixAdd,
        Self::MatrixSubtract,
        Self::MatrixMultiply,
        Self::MatrixMulAccumulate,
        Self::MatrixMulSmallRhs,
        Self::Transpose,
        Self::Slice,
        Self::Tensor,
        Self::ConcatRows,
        Self::ConcatColumns,
        Self::ConcatDiagonal,
        Self::HostOrControl,
    ];

    pub const fn all() -> &'static [Self] {
        Self::ALL
    }

    /// Return the host/device boundary associated with this effective
    /// operation.  This mirrors the canonical profile-domain classification
    /// and is intentionally exhaustive so adding a boundary primitive
    /// requires updating both dispatch inventories.
    pub const fn transfer_kind(self) -> WarmupTransferKind {
        match self {
            Self::GeneratedConstant |
            Self::SingleDeviceConstant |
            Self::UniformResidueSample |
            Self::UniformIntervalSample |
            Self::GaussianSample |
            Self::HashSample |
            Self::HashIntFamily |
            Self::TrapdoorSample |
            Self::GadgetTrapdoor |
            Self::LiftIntegerToConstantPolynomial |
            Self::TrapdoorPublic |
            Self::ExtractCoefficient |
            Self::ThresholdDecode |
            Self::PackPolynomialCoefficients |
            Self::PolynomialFromValues |
            Self::PolynomialValues |
            Self::PreimageSample |
            Self::GadgetDecompose |
            Self::MatrixScale |
            Self::MatrixNegate |
            Self::RingAutomorphism |
            Self::ModulusSwitch |
            Self::ModulusReduce |
            Self::CenteredRebase |
            Self::CenteredRoundDivide |
            Self::BlockModSwitch |
            Self::RnsModUp |
            Self::RnsModDown |
            Self::CrtRecompose |
            Self::MatrixAdd |
            Self::MatrixSubtract |
            Self::MatrixMultiply |
            Self::MatrixMulAccumulate |
            Self::MatrixMulSmallRhs |
            Self::Transpose |
            Self::Slice |
            Self::Tensor |
            Self::ConcatRows |
            Self::ConcatColumns |
            Self::ConcatDiagonal |
            Self::HostOrControl |
            Self::Unsupported => WarmupTransferKind::None,
        }
    }
}

/// The canonical profile domain for every IR node and every fixed fused
/// primitive.  This is deliberately a closed enum: adding an IR operation
/// requires adding its profile domain here and updating the exhaustive
/// classifier below.  Host operations remain in the domain because their
/// setup cost is still measured; they are never silently omitted from the
/// provider contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum CanonicalWarmupProfileDomain {
    Input,
    ConstantInt,
    EvaluateInt,
    ConstantReal,
    ConstantBool,
    ConstantMatrixZero,
    ConstantMatrixIdentity,
    ConstantMatrixUnitRow,
    ConstantMatrixUnitColumn,
    ConstantMatrixGadget,
    ConstantMatrixPowerOfBase,
    ConstantMatrixRotation,
    ConstantMatrixPolynomial,
    GadgetTrapdoor,
    TrapdoorPublic,
    IntBinary,
    IntCompare,
    BitExtract,
    IntToReal,
    BoolToInt,
    RealBinary,
    RealSqrt,
    MatrixAdd,
    MatrixSubtract,
    MatrixMultiply,
    MatrixMulAccumulate,
    MatrixMulSmallRhs,
    MatrixNegate,
    MatrixScale,
    RingAutomorphism,
    ModulusSwitch,
    ModulusReduce,
    CenteredRebase,
    CenteredRoundDivide,
    BlockModSwitch,
    RnsModUp,
    RnsModDown,
    Transpose,
    Slice,
    Tensor,
    ConcatRows,
    ConcatColumns,
    ConcatDiagonal,
    UniformResidueSample,
    UniformIntervalSample,
    GaussianSample,
    HashSample,
    HashIntFamily,
    TrapdoorSample,
    PreimageSample,
    GadgetDecompose,
    ExtractCoefficient,
    LiftIntegerToConstantPolynomial,
    ThresholdDecode,
    CrtRecompose,
    PackPolynomialCoefficients,
    PolynomialFromValues,
    PolynomialValues,
    SubgraphCall,
    ParallelLoop,
    SequentialLoop,
    FamilyPack,
    FamilyGetStatic,
    FamilyGetDynamic,
    Select,
    /// Fused sparse row sum over one matrix.
    FusedRowSum,
    /// Fused sparse row sum over a tensor product.
    FusedTensorRowSum,
    /// Fused addition of physical row blocks.
    FusedRowBlockAdd,
    /// Fused gadget decomposition batch.
    FusedDecompose,
    /// Fused compact-product batch (decomposition followed by small-RHS mul).
    FusedCompactProduct,
    /// Fused preimage sampling batch.
    FusedPreimageBatch,
}

/// Whether a canonical profile is measured through a GPU kernel or a host
/// implementation.  Both variants require a measured elapsed time.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum WarmupMeasurementKind {
    GpuMeasured,
    HostMeasured,
}

/// Physical transfer work associated with a host-visible boundary primitive.
///
/// The primitive's elapsed host work and the transfer/staging envelope are
/// intentionally separate warmup concerns.  In particular, a coefficient
/// extraction must not be presented as a GPU kernel merely because obtaining
/// its coefficients synchronizes/copies a device value.  Providers use this
/// classification to register a second `Transfer` implementation variant for
/// the production boundary when the route actually crosses the host.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum WarmupTransferKind {
    None,
    HostToDevice,
    DeviceToHost,
    Peer,
    HostStaging,
}

impl WarmupTransferKind {
    /// The complete set of physical boundary classes.  A boundary primitive
    /// may additionally have a host-timed profile; transfer accounting is a
    /// separate, explicit route selected from this closed set.
    pub const ALL: &'static [Self] =
        &[Self::None, Self::HostToDevice, Self::DeviceToHost, Self::Peer, Self::HostStaging];

    pub const fn all() -> &'static [Self] {
        Self::ALL
    }

    /// Whether this boundary moves bytes from the host into a device or in
    /// the opposite direction.  Keeping the direction on the canonical
    /// profile domain makes it impossible for the estimator to accidentally
    /// account a host-only primitive as a device kernel.
    pub const fn is_host_to_device(self) -> bool {
        matches!(self, Self::HostToDevice)
    }

    pub const fn is_device_to_host(self) -> bool {
        matches!(self, Self::DeviceToHost)
    }
}

/// Names used in logs and provider diagnostics.  The string is stable across
/// Rust refactors and intentionally does not include shape or instance data;
/// those belong to [`GpuWarmupOperationSignature`](crate::backend::GpuWarmupOperationSignature).
impl CanonicalWarmupProfileDomain {
    /// Exhaustive profile-domain inventory for all ordinary IR primitives and
    /// all fixed fused dispatches.  This is intentionally independent of
    /// `NodeKind` values so the provider can validate a descriptor before it
    /// constructs a representative.  No entry represents an excluded,
    /// unsupported, or size-only operation.
    pub const ALL: &'static [Self] = &[
        Self::Input,
        Self::ConstantInt,
        Self::EvaluateInt,
        Self::ConstantReal,
        Self::ConstantBool,
        Self::ConstantMatrixZero,
        Self::ConstantMatrixIdentity,
        Self::ConstantMatrixUnitRow,
        Self::ConstantMatrixUnitColumn,
        Self::ConstantMatrixGadget,
        Self::ConstantMatrixPowerOfBase,
        Self::ConstantMatrixRotation,
        Self::ConstantMatrixPolynomial,
        Self::GadgetTrapdoor,
        Self::TrapdoorPublic,
        Self::IntBinary,
        Self::IntCompare,
        Self::BitExtract,
        Self::IntToReal,
        Self::BoolToInt,
        Self::RealBinary,
        Self::RealSqrt,
        Self::MatrixAdd,
        Self::MatrixSubtract,
        Self::MatrixMultiply,
        Self::MatrixMulAccumulate,
        Self::MatrixMulSmallRhs,
        Self::MatrixNegate,
        Self::MatrixScale,
        Self::RingAutomorphism,
        Self::ModulusSwitch,
        Self::ModulusReduce,
        Self::CenteredRebase,
        Self::CenteredRoundDivide,
        Self::BlockModSwitch,
        Self::RnsModUp,
        Self::RnsModDown,
        Self::Transpose,
        Self::Slice,
        Self::Tensor,
        Self::ConcatRows,
        Self::ConcatColumns,
        Self::ConcatDiagonal,
        Self::UniformResidueSample,
        Self::UniformIntervalSample,
        Self::GaussianSample,
        Self::HashSample,
        Self::HashIntFamily,
        Self::TrapdoorSample,
        Self::PreimageSample,
        Self::GadgetDecompose,
        Self::ExtractCoefficient,
        Self::LiftIntegerToConstantPolynomial,
        Self::ThresholdDecode,
        Self::CrtRecompose,
        Self::PackPolynomialCoefficients,
        Self::PolynomialFromValues,
        Self::PolynomialValues,
        Self::SubgraphCall,
        Self::ParallelLoop,
        Self::SequentialLoop,
        Self::FamilyPack,
        Self::FamilyGetStatic,
        Self::FamilyGetDynamic,
        Self::Select,
        Self::FusedRowSum,
        Self::FusedTensorRowSum,
        Self::FusedRowBlockAdd,
        Self::FusedDecompose,
        Self::FusedCompactProduct,
        Self::FusedPreimageBatch,
    ];

    pub const fn all() -> &'static [Self] {
        Self::ALL
    }

    /// Whether this domain belongs to the measured warmup inventory.  This
    /// guard is intentionally separate from `measurement_kind`: an enum
    /// value can be syntactically valid while still being omitted from the
    /// provider's closed coverage table.
    pub fn is_profileable(self) -> bool {
        Self::ALL.contains(&self)
    }

    pub const fn identity(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::ConstantInt => "constant_int",
            Self::EvaluateInt => "evaluate_int",
            Self::ConstantReal => "constant_real",
            Self::ConstantBool => "constant_bool",
            Self::ConstantMatrixZero => "constant_matrix_zero",
            Self::ConstantMatrixIdentity => "constant_matrix_identity",
            Self::ConstantMatrixUnitRow => "constant_matrix_unit_row",
            Self::ConstantMatrixUnitColumn => "constant_matrix_unit_column",
            Self::ConstantMatrixGadget => "constant_matrix_gadget",
            Self::ConstantMatrixPowerOfBase => "constant_matrix_power_of_base",
            Self::ConstantMatrixRotation => "constant_matrix_rotation",
            Self::ConstantMatrixPolynomial => "constant_matrix_polynomial",
            Self::GadgetTrapdoor => "gadget_trapdoor",
            Self::TrapdoorPublic => "trapdoor_public",
            Self::ExtractCoefficient => "extract_coefficient",
            Self::ThresholdDecode => "threshold_decode",
            Self::PackPolynomialCoefficients => "pack_polynomial_coefficients",
            Self::PolynomialFromValues => "polynomial_from_values",
            Self::PolynomialValues => "polynomial_values",
            Self::IntBinary => "int_binary",
            Self::IntCompare => "int_compare",
            Self::BitExtract => "bit_extract",
            Self::IntToReal => "int_to_real",
            Self::BoolToInt => "bool_to_int",
            Self::RealBinary => "real_binary",
            Self::RealSqrt => "real_sqrt",
            Self::MatrixAdd => "matrix_add",
            Self::MatrixSubtract => "matrix_subtract",
            Self::MatrixMultiply => "matrix_multiply",
            Self::MatrixMulAccumulate => "matrix_mul_accumulate",
            Self::MatrixMulSmallRhs => "matrix_mul_small_rhs",
            Self::MatrixNegate => "matrix_negate",
            Self::MatrixScale => "matrix_scale",
            Self::RingAutomorphism => "ring_automorphism",
            Self::ModulusSwitch => "modulus_switch",
            Self::ModulusReduce => "modulus_reduce",
            Self::CenteredRebase => "centered_rebase",
            Self::CenteredRoundDivide => "centered_round_divide",
            Self::BlockModSwitch => "block_mod_switch",
            Self::RnsModUp => "rns_mod_up",
            Self::RnsModDown => "rns_mod_down",
            Self::Transpose => "transpose",
            Self::Slice => "slice",
            Self::Tensor => "tensor",
            Self::ConcatRows => "concat_rows",
            Self::ConcatColumns => "concat_columns",
            Self::ConcatDiagonal => "concat_diagonal",
            Self::UniformResidueSample => "uniform_residue_sample",
            Self::UniformIntervalSample => "uniform_interval_sample",
            Self::GaussianSample => "gaussian_sample",
            Self::HashSample => "hash_sample",
            Self::HashIntFamily => "hash_int_family",
            Self::TrapdoorSample => "trapdoor_sample",
            Self::PreimageSample => "preimage_sample",
            Self::GadgetDecompose => "gadget_decompose",
            Self::LiftIntegerToConstantPolynomial => "lift_integer_to_constant_polynomial",
            Self::CrtRecompose => "crt_recompose",
            Self::SubgraphCall => "subgraph_call",
            Self::ParallelLoop => "parallel_loop",
            Self::SequentialLoop => "sequential_loop",
            Self::FamilyPack => "family_pack",
            Self::FamilyGetStatic => "family_get_static",
            Self::FamilyGetDynamic => "family_get_dynamic",
            Self::Select => "select",
            Self::FusedRowSum => "fused_row_sum",
            Self::FusedTensorRowSum => "fused_tensor_row_sum",
            Self::FusedRowBlockAdd => "fused_row_block_add",
            Self::FusedDecompose => "fused_decompose",
            Self::FusedCompactProduct => "fused_compact_product",
            Self::FusedPreimageBatch => "fused_preimage_batch",
        }
    }

    pub const fn measurement_kind(self) -> WarmupMeasurementKind {
        match self {
            Self::Input |
            Self::TrapdoorPublic |
            Self::ConstantReal |
            Self::IntToReal |
            Self::RealBinary |
            Self::RealSqrt => WarmupMeasurementKind::HostMeasured,
            Self::ConstantInt |
            Self::EvaluateInt |
            Self::ConstantBool |
            Self::IntBinary |
            Self::IntCompare |
            Self::BitExtract |
            Self::BoolToInt |
            Self::SubgraphCall |
            Self::ParallelLoop |
            Self::SequentialLoop |
            Self::FamilyPack |
            Self::FamilyGetStatic |
            Self::FamilyGetDynamic |
            Self::Select |
            Self::MatrixNegate |
            Self::MatrixScale |
            Self::RingAutomorphism |
            Self::ModulusSwitch |
            Self::ModulusReduce |
            Self::CenteredRebase |
            Self::CenteredRoundDivide |
            Self::BlockModSwitch |
            Self::RnsModUp |
            Self::RnsModDown |
            Self::Transpose |
            Self::Slice |
            Self::Tensor |
            Self::ConcatRows |
            Self::ConcatColumns |
            Self::ConcatDiagonal |
            Self::ConstantMatrixZero |
            Self::ConstantMatrixIdentity |
            Self::ConstantMatrixUnitRow |
            Self::ConstantMatrixUnitColumn |
            Self::ConstantMatrixGadget |
            Self::ConstantMatrixPowerOfBase |
            Self::ConstantMatrixRotation |
            Self::ConstantMatrixPolynomial |
            Self::GadgetTrapdoor |
            Self::MatrixAdd |
            Self::MatrixSubtract |
            Self::MatrixMultiply |
            Self::MatrixMulAccumulate |
            Self::MatrixMulSmallRhs |
            Self::UniformResidueSample |
            Self::UniformIntervalSample |
            Self::GaussianSample |
            Self::HashSample |
            Self::HashIntFamily |
            Self::TrapdoorSample |
            Self::PreimageSample |
            Self::GadgetDecompose |
            Self::LiftIntegerToConstantPolynomial |
            Self::ExtractCoefficient |
            Self::ThresholdDecode |
            Self::PackPolynomialCoefficients |
            Self::PolynomialFromValues |
            Self::PolynomialValues |
            Self::CrtRecompose |
            Self::FusedRowSum |
            Self::FusedTensorRowSum |
            Self::FusedRowBlockAdd |
            Self::FusedDecompose |
            Self::FusedCompactProduct |
            Self::FusedPreimageBatch => WarmupMeasurementKind::GpuMeasured,
        }
    }

    /// Return the transfer stage, if this primitive's production boundary
    /// crosses between host and device representations.  This does not change
    /// [`measurement_kind`](Self::measurement_kind): the primitive itself is
    /// still timed through its real host implementation, while the provider
    /// may account for the physical copy as a distinct transport stage.
    pub const fn transfer_kind(self) -> WarmupTransferKind {
        match self {
            Self::Input |
            Self::ConstantInt |
            Self::EvaluateInt |
            Self::ConstantReal |
            Self::ConstantBool |
            Self::ConstantMatrixZero |
            Self::ConstantMatrixIdentity |
            Self::ConstantMatrixUnitRow |
            Self::ConstantMatrixUnitColumn |
            Self::ConstantMatrixGadget |
            Self::ConstantMatrixPowerOfBase |
            Self::ConstantMatrixRotation |
            Self::ConstantMatrixPolynomial |
            Self::GadgetTrapdoor |
            Self::TrapdoorPublic |
            Self::IntBinary |
            Self::IntCompare |
            Self::BitExtract |
            Self::IntToReal |
            Self::BoolToInt |
            Self::RealBinary |
            Self::RealSqrt |
            Self::MatrixAdd |
            Self::MatrixSubtract |
            Self::MatrixMultiply |
            Self::MatrixMulAccumulate |
            Self::MatrixMulSmallRhs |
            Self::MatrixNegate |
            Self::MatrixScale |
            Self::RingAutomorphism |
            Self::ModulusSwitch |
            Self::ModulusReduce |
            Self::CenteredRebase |
            Self::CenteredRoundDivide |
            Self::BlockModSwitch |
            Self::RnsModUp |
            Self::RnsModDown |
            Self::Transpose |
            Self::Slice |
            Self::Tensor |
            Self::ConcatRows |
            Self::ConcatColumns |
            Self::ConcatDiagonal |
            Self::UniformResidueSample |
            Self::UniformIntervalSample |
            Self::GaussianSample |
            Self::HashSample |
            Self::HashIntFamily |
            Self::TrapdoorSample |
            Self::PreimageSample |
            Self::GadgetDecompose |
            Self::LiftIntegerToConstantPolynomial |
            Self::ExtractCoefficient |
            Self::ThresholdDecode |
            Self::PackPolynomialCoefficients |
            Self::PolynomialFromValues |
            Self::PolynomialValues |
            Self::CrtRecompose |
            Self::SubgraphCall |
            Self::ParallelLoop |
            Self::SequentialLoop |
            Self::FamilyPack |
            Self::FamilyGetStatic |
            Self::FamilyGetDynamic |
            Self::Select |
            Self::FusedRowSum |
            Self::FusedTensorRowSum |
            Self::FusedRowBlockAdd |
            Self::FusedDecompose |
            Self::FusedCompactProduct |
            Self::FusedPreimageBatch => WarmupTransferKind::None,
        }
    }

    pub const fn has_transport_stage(self) -> bool {
        !matches!(self.transfer_kind(), WarmupTransferKind::None)
    }

    pub const fn is_gpu_measured(self) -> bool {
        matches!(self.measurement_kind(), WarmupMeasurementKind::GpuMeasured)
    }

    /// The transfer profile is keyed independently from the host primitive
    /// profile.  It uses the same canonical operation identity, but a
    /// `Transfer` implementation variant and byte coordinate (see
    /// `GpuExecutionRouteDescriptor::transfer_bytes`).
    pub const fn transfer_profile_variant(self) -> bool {
        self.has_transport_stage()
    }
}

/// Exhaustive canonical domain for an IR node.  In particular, each matrix
/// constant and concat axis has a distinct profile domain, so a provider can
/// never accidentally reuse a zero/identity or row/column/diagonal profile.
pub fn canonical_warmup_profile_domain(kind: &NodeKind) -> CanonicalWarmupProfileDomain {
    match kind {
        NodeKind::Input { .. } => CanonicalWarmupProfileDomain::Input,
        NodeKind::ConstantInt(_) => CanonicalWarmupProfileDomain::ConstantInt,
        NodeKind::EvaluateInt(_) => CanonicalWarmupProfileDomain::EvaluateInt,
        NodeKind::ConstantReal(_) => CanonicalWarmupProfileDomain::ConstantReal,
        NodeKind::ConstantBool(_) => CanonicalWarmupProfileDomain::ConstantBool,
        NodeKind::ConstantMatrix { value, .. } => match value {
            ConstantMatrix::Zero => CanonicalWarmupProfileDomain::ConstantMatrixZero,
            ConstantMatrix::Identity => CanonicalWarmupProfileDomain::ConstantMatrixIdentity,
            ConstantMatrix::UnitRow { .. } => CanonicalWarmupProfileDomain::ConstantMatrixUnitRow,
            ConstantMatrix::UnitColumn { .. } => {
                CanonicalWarmupProfileDomain::ConstantMatrixUnitColumn
            }
            ConstantMatrix::Gadget { .. } => CanonicalWarmupProfileDomain::ConstantMatrixGadget,
            ConstantMatrix::PowerOfBase { .. } => {
                CanonicalWarmupProfileDomain::ConstantMatrixPowerOfBase
            }
            ConstantMatrix::Rotation { .. } => CanonicalWarmupProfileDomain::ConstantMatrixRotation,
            ConstantMatrix::Polynomial { .. } => {
                CanonicalWarmupProfileDomain::ConstantMatrixPolynomial
            }
        },
        NodeKind::GadgetTrapdoor { .. } => CanonicalWarmupProfileDomain::GadgetTrapdoor,
        NodeKind::TrapdoorPublic => CanonicalWarmupProfileDomain::TrapdoorPublic,
        NodeKind::IntBinary(_) => CanonicalWarmupProfileDomain::IntBinary,
        NodeKind::IntCompare(_) => CanonicalWarmupProfileDomain::IntCompare,
        NodeKind::BitExtract { .. } => CanonicalWarmupProfileDomain::BitExtract,
        NodeKind::IntToReal => CanonicalWarmupProfileDomain::IntToReal,
        NodeKind::BoolToInt => CanonicalWarmupProfileDomain::BoolToInt,
        NodeKind::RealBinary(_) => CanonicalWarmupProfileDomain::RealBinary,
        NodeKind::RealSqrt => CanonicalWarmupProfileDomain::RealSqrt,
        NodeKind::MatrixBinary(MatrixBinaryOp::Add) => CanonicalWarmupProfileDomain::MatrixAdd,
        NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) => {
            CanonicalWarmupProfileDomain::MatrixSubtract
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => {
            CanonicalWarmupProfileDomain::MatrixMultiply
        }
        NodeKind::MatrixMulAccumulate { .. } => CanonicalWarmupProfileDomain::MatrixMulAccumulate,
        NodeKind::MatrixMulSmallRhs => CanonicalWarmupProfileDomain::MatrixMulSmallRhs,
        NodeKind::MatrixNegate => CanonicalWarmupProfileDomain::MatrixNegate,
        NodeKind::MatrixScale { .. } => CanonicalWarmupProfileDomain::MatrixScale,
        NodeKind::RingAutomorphism { .. } => CanonicalWarmupProfileDomain::RingAutomorphism,
        NodeKind::ModulusSwitch { .. } => CanonicalWarmupProfileDomain::ModulusSwitch,
        NodeKind::ModulusReduce { .. } => CanonicalWarmupProfileDomain::ModulusReduce,
        NodeKind::CenteredRebase { .. } => CanonicalWarmupProfileDomain::CenteredRebase,
        NodeKind::CenteredRoundDivide { .. } => CanonicalWarmupProfileDomain::CenteredRoundDivide,
        NodeKind::BlockModSwitch { .. } => CanonicalWarmupProfileDomain::BlockModSwitch,
        NodeKind::RnsModUp { .. } => CanonicalWarmupProfileDomain::RnsModUp,
        NodeKind::RnsModDown { .. } => CanonicalWarmupProfileDomain::RnsModDown,
        NodeKind::Transpose => CanonicalWarmupProfileDomain::Transpose,
        NodeKind::Slice { .. } => CanonicalWarmupProfileDomain::Slice,
        NodeKind::Tensor => CanonicalWarmupProfileDomain::Tensor,
        NodeKind::Concat { axis: ConcatAxis::Rows } => CanonicalWarmupProfileDomain::ConcatRows,
        NodeKind::Concat { axis: ConcatAxis::Columns } => {
            CanonicalWarmupProfileDomain::ConcatColumns
        }
        NodeKind::Concat { axis: ConcatAxis::Diagonal } => {
            CanonicalWarmupProfileDomain::ConcatDiagonal
        }
        NodeKind::UniformResidueSample { .. } => CanonicalWarmupProfileDomain::UniformResidueSample,
        NodeKind::UniformIntervalSample { .. } => {
            CanonicalWarmupProfileDomain::UniformIntervalSample
        }
        NodeKind::GaussianSample { .. } => CanonicalWarmupProfileDomain::GaussianSample,
        NodeKind::HashSample { .. } => CanonicalWarmupProfileDomain::HashSample,
        NodeKind::HashIntFamily { .. } => CanonicalWarmupProfileDomain::HashIntFamily,
        NodeKind::TrapdoorSample { .. } => CanonicalWarmupProfileDomain::TrapdoorSample,
        NodeKind::PreimageSample { .. } => CanonicalWarmupProfileDomain::PreimageSample,
        NodeKind::GadgetDecompose { .. } => CanonicalWarmupProfileDomain::GadgetDecompose,
        NodeKind::ExtractCoefficient { .. } => CanonicalWarmupProfileDomain::ExtractCoefficient,
        NodeKind::LiftIntegerToConstantPolynomial { .. } => {
            CanonicalWarmupProfileDomain::LiftIntegerToConstantPolynomial
        }
        NodeKind::ThresholdDecode { .. } => CanonicalWarmupProfileDomain::ThresholdDecode,
        NodeKind::CrtRecompose { .. } => CanonicalWarmupProfileDomain::CrtRecompose,
        NodeKind::PackPolynomialCoefficients { .. } => {
            CanonicalWarmupProfileDomain::PackPolynomialCoefficients
        }
        NodeKind::PolynomialFromValues { .. } => CanonicalWarmupProfileDomain::PolynomialFromValues,
        NodeKind::PolynomialValues { .. } => CanonicalWarmupProfileDomain::PolynomialValues,
        NodeKind::SubgraphCall(_) => CanonicalWarmupProfileDomain::SubgraphCall,
        NodeKind::ParallelLoop(_) => CanonicalWarmupProfileDomain::ParallelLoop,
        NodeKind::SequentialLoop(_) => CanonicalWarmupProfileDomain::SequentialLoop,
        NodeKind::FamilyPack { .. } => CanonicalWarmupProfileDomain::FamilyPack,
        NodeKind::FamilyGetStatic { .. } => CanonicalWarmupProfileDomain::FamilyGetStatic,
        NodeKind::FamilyGetDynamic => CanonicalWarmupProfileDomain::FamilyGetDynamic,
        NodeKind::Select { .. } => CanonicalWarmupProfileDomain::Select,
    }
}

/// Distinct fixed fused identities.  `RowSum` with a tensor RHS is not
/// interchangeable with ordinary row sum, and every listed primitive has its
/// own kernel/workspace profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum FusedWarmupOperation {
    RowSum,
    TensorRowSum,
    RowBlockAdd,
    Decompose,
    CompactProduct,
    PreimageBatch,
}

impl FusedWarmupOperation {
    /// Closed inventory of fused production dispatches.  Every fused entry
    /// has a distinct canonical profile domain and must be measured through
    /// its own production lowering.
    pub const ALL: &'static [Self] = &[
        Self::RowSum,
        Self::TensorRowSum,
        Self::RowBlockAdd,
        Self::Decompose,
        Self::CompactProduct,
        Self::PreimageBatch,
    ];

    pub const fn all() -> &'static [Self] {
        Self::ALL
    }
}

pub const fn fused_warmup_profile_domain(
    operation: FusedWarmupOperation,
) -> CanonicalWarmupProfileDomain {
    match operation {
        FusedWarmupOperation::RowSum => CanonicalWarmupProfileDomain::FusedRowSum,
        FusedWarmupOperation::TensorRowSum => CanonicalWarmupProfileDomain::FusedTensorRowSum,
        FusedWarmupOperation::RowBlockAdd => CanonicalWarmupProfileDomain::FusedRowBlockAdd,
        FusedWarmupOperation::Decompose => CanonicalWarmupProfileDomain::FusedDecompose,
        FusedWarmupOperation::CompactProduct => CanonicalWarmupProfileDomain::FusedCompactProduct,
        FusedWarmupOperation::PreimageBatch => CanonicalWarmupProfileDomain::FusedPreimageBatch,
    }
}

pub type WarmupProfileDomain = CanonicalWarmupProfileDomain;
pub type CanonicalGpuWarmupProfileDomain = CanonicalWarmupProfileDomain;
pub type GpuWarmupMeasurementKind = WarmupMeasurementKind;

pub const fn canonical_fused_warmup_profile_domain(
    operation: FusedWarmupOperation,
) -> CanonicalWarmupProfileDomain {
    fused_warmup_profile_domain(operation)
}

/// The concrete implementation family used by both warmup representatives and
/// fixed dispatch.  This is intentionally typed instead of being inferred
/// from an operation hash: mapped and compact variants have different
/// materialization and workspace contracts even when their IR node is the
/// same.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuExecutionVariant {
    Primitive(EffectiveGpuOperation),
    Fused(FusedWarmupOperation),
    MappedSlice,
    MappedTranspose,
    MappedTensor,
    MappedConcatRows,
    MappedConcatColumns,
    MappedConcatDiagonal,
    CompactFragmentMultiply,
    CompactFragmentAssembly,
    PreimageSource,
    PreimageSampling,
    PreimagePacking,
}

/// A local output fragment is not necessarily a complete logical output.  In
/// particular, a planned job can produce a tail, a tensor segment, or a piece
/// of a compact RHS shard.  Keeping this classification alongside the mapped
/// ranges prevents a warmup request from accidentally timing the full value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuFragmentClass {
    Full,
    PlannedJob,
    Tail,
    Mapped,
    TensorSegment,
    ConcatFragment,
    CompactFragment,
    CompactAssembly,
    SourceStaging,
}

/// The route of a physical source transfer.  `HostStaging` is explicit so a
/// provider can charge it separately; it is never silently treated as a
/// device-local operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuTransferRoute {
    Resident,
    Peer,
    HostStaging,
}

/// The route of one physical source owner for a mapped production range.
///
/// The route descriptor remains `Copy` so it can be embedded in profile keys
/// and passed through the hot dispatch path without reference-counted state.
/// A bounded fixed array also makes the complete source ownership part of the
/// key instead of collapsing a multi-shard input to its first owner.  The
/// bound is deliberately generous for the current fleet lowering; callers
/// that exceed it are rejected by [`GpuExecutionRouteDescriptor::with_source_routes`].
pub const GPU_ROUTE_MAX_SOURCES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GpuExecutionSourceRoute {
    pub source_owner: usize,
    pub source_range: ColumnRange,
    pub route: GpuTransferRoute,
    pub source_staging_bytes: usize,
    pub host_staging_bytes: usize,
    pub pinned_host_staging_bytes: usize,
}

impl GpuExecutionSourceRoute {
    pub const fn new(
        source_owner: usize,
        source_range: ColumnRange,
        route: GpuTransferRoute,
        source_staging_bytes: usize,
        host_staging_bytes: usize,
        pinned_host_staging_bytes: usize,
    ) -> Self {
        Self {
            source_owner,
            source_range,
            route,
            source_staging_bytes,
            host_staging_bytes,
            pinned_host_staging_bytes,
        }
    }
}

const EMPTY_SOURCE_ROUTE: GpuExecutionSourceRoute = GpuExecutionSourceRoute {
    source_owner: 0,
    source_range: ColumnRange { start: 0, end: 0 },
    route: GpuTransferRoute::Resident,
    source_staging_bytes: 0,
    host_staging_bytes: 0,
    pinned_host_staging_bytes: 0,
};

/// The complete physical transfer contract for one lowered range.  The
/// column mapper knows the logical source/destination ranges, while the fleet
/// adapter fills in the physical devices and representation conversion.  A
/// route is intentionally richer than [`GpuTransferRoute`]: equal widths and
/// shapes are not interchangeable when one invocation reads a compact shard,
/// assembles a mapped concat fragment, or stages through host memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GpuExecutionRouteDescriptor {
    pub route: GpuTransferRoute,
    pub source_device: Option<usize>,
    pub destination_device: Option<usize>,
    pub source_range: ColumnRange,
    pub destination_range: ColumnRange,
    pub source_compact: bool,
    pub destination_compact: bool,
    pub fragment: GpuFragmentClass,
    /// Bytes copied through a source-device staging owner before the kernel.
    /// These are supplied by the native fleet adapter; the pure mapper keeps
    /// the field at zero until a physical route is selected.
    pub source_staging_bytes: usize,
    pub host_staging_bytes: usize,
    pub pinned_host_staging_bytes: usize,
    /// Ordered, normalized source fragments and per-source physical transfer
    /// facts. A physical owner may occur more than once when distinct mapped
    /// operands leave a real gap in the shared column coordinate. The legacy scalar fields above
    /// remain aggregate convenience values; profile identity and cache validation use this
    /// complete inventory.
    pub source_route_count: u8,
    pub source_routes: [GpuExecutionSourceRoute; GPU_ROUTE_MAX_SOURCES],
}

/// Physical facts collected by the production fleet before a bounded job is
/// submitted.  Warmup carries these facts alongside the logical range rather
/// than reconstructing a route from a width (which would incorrectly turn a
/// peer or host-staged job into a device-local profile).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GpuRouteResolutionInput {
    pub source_device: Option<usize>,
    pub destination_device: Option<usize>,
    pub source_range: ColumnRange,
    pub destination_range: ColumnRange,
    pub source_is_resident: bool,
    pub peer_available: bool,
    pub source_compact: bool,
    pub destination_compact: bool,
    pub fragment: GpuFragmentClass,
    pub source_staging_bytes: usize,
    pub host_staging_bytes: usize,
    pub pinned_host_staging_bytes: usize,
}

/// Resolve the route identity shared by fixed execution and setup warmup.
/// Keeping this conversion pure lets the fleet perform the device capability
/// query while ensuring both paths serialize exactly the same route key.
pub fn resolve_gpu_route(input: GpuRouteResolutionInput) -> GpuExecutionRouteDescriptor {
    let route = if input.source_is_resident {
        GpuTransferRoute::Resident
    } else if input.peer_available {
        GpuTransferRoute::Peer
    } else {
        GpuTransferRoute::HostStaging
    };
    GpuExecutionRouteDescriptor {
        route,
        source_device: input.source_device,
        destination_device: input.destination_device,
        source_range: input.source_range,
        destination_range: input.destination_range,
        source_compact: input.source_compact,
        destination_compact: input.destination_compact,
        fragment: input.fragment,
        source_staging_bytes: input.source_staging_bytes,
        host_staging_bytes: input.host_staging_bytes,
        pinned_host_staging_bytes: input.pinned_host_staging_bytes,
        source_route_count: u8::from(input.source_device.is_some()),
        source_routes: if let Some(source_owner) = input.source_device {
            [GpuExecutionSourceRoute::new(
                source_owner,
                input.source_range,
                route,
                input.source_staging_bytes,
                input.host_staging_bytes,
                input.pinned_host_staging_bytes,
            ); GPU_ROUTE_MAX_SOURCES]
        } else {
            [EMPTY_SOURCE_ROUTE; GPU_ROUTE_MAX_SOURCES]
        },
    }
}

impl GpuExecutionRouteDescriptor {
    pub fn device_local(device: usize, range: ColumnRange, fragment: GpuFragmentClass) -> Self {
        Self {
            route: GpuTransferRoute::Resident,
            source_device: Some(device),
            destination_device: Some(device),
            source_range: range,
            destination_range: range,
            source_compact: false,
            destination_compact: false,
            fragment,
            source_staging_bytes: 0,
            host_staging_bytes: 0,
            pinned_host_staging_bytes: 0,
            source_route_count: 1,
            source_routes: [GpuExecutionSourceRoute::new(
                device,
                range,
                GpuTransferRoute::Resident,
                0,
                0,
                0,
            ); GPU_ROUTE_MAX_SOURCES],
        }
    }

    /// Replace the single-source convenience route with the complete ordered
    /// source inventory. Repeated owners are merged in first-seen order only
    /// when their intervals overlap, touch, or another retained owner covers
    /// the gap. A genuinely sparse same-owner mapping remains as distinct
    /// fragments so the descriptor never invents physical coverage.
    pub fn with_source_routes(
        mut self,
        routes: impl IntoIterator<Item = GpuExecutionSourceRoute>,
    ) -> Option<Self> {
        let mut merged = [EMPTY_SOURCE_ROUTE; GPU_ROUTE_MAX_SOURCES];
        let mut count = 0usize;
        for route in routes {
            if route.source_range.is_empty() {
                return None;
            }
            if let Some(index) = merged[..count].iter().position(|existing| {
                if existing.source_owner != route.source_owner {
                    return false;
                }
                // A source route stores one contiguous global interval. It is
                // safe to bridge a gap only when another retained owner
                // already covers that gap; otherwise min/max would invent
                // physical coverage that the route does not provide.
                if existing.source_range.end >= route.source_range.start &&
                    route.source_range.end >= existing.source_range.start
                {
                    return true;
                }
                let (gap_start, gap_end) = if existing.source_range.end < route.source_range.start {
                    (existing.source_range.end, route.source_range.start)
                } else {
                    (route.source_range.end, existing.source_range.start)
                };
                let mut cursor = gap_start;
                while cursor < gap_end {
                    let Some(next) = merged[..count]
                        .iter()
                        .filter(|other| other.source_owner != route.source_owner)
                        .filter(|other| {
                            other.source_range.start <= cursor && other.source_range.end > cursor
                        })
                        .map(|other| other.source_range.end)
                        .max()
                    else {
                        return false;
                    };
                    cursor = next.min(gap_end);
                }
                true
            }) {
                let existing = &mut merged[index];
                existing.source_range = ColumnRange {
                    start: existing.source_range.start.min(route.source_range.start),
                    end: existing.source_range.end.max(route.source_range.end),
                };
                existing.source_staging_bytes =
                    existing.source_staging_bytes.checked_add(route.source_staging_bytes)?;
                existing.host_staging_bytes =
                    existing.host_staging_bytes.checked_add(route.host_staging_bytes)?;
                existing.pinned_host_staging_bytes = existing
                    .pinned_host_staging_bytes
                    .checked_add(route.pinned_host_staging_bytes)?;
                if existing.route != route.route {
                    existing.route = if matches!(existing.route, GpuTransferRoute::HostStaging) ||
                        matches!(route.route, GpuTransferRoute::HostStaging)
                    {
                        GpuTransferRoute::HostStaging
                    } else {
                        GpuTransferRoute::Peer
                    };
                }
            } else {
                if count == GPU_ROUTE_MAX_SOURCES {
                    return None;
                }
                merged[count] = route;
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        self.source_route_count = count as u8;
        self.source_routes = merged;
        self.source_staging_bytes = merged[..count]
            .iter()
            .try_fold(0usize, |sum, route| sum.checked_add(route.source_staging_bytes))?;
        self.host_staging_bytes = merged[..count]
            .iter()
            .try_fold(0usize, |sum, route| sum.checked_add(route.host_staging_bytes))?;
        self.pinned_host_staging_bytes = merged[..count]
            .iter()
            .try_fold(0usize, |sum, route| sum.checked_add(route.pinned_host_staging_bytes))?;
        Some(self)
    }

    pub fn source_routes(&self) -> &[GpuExecutionSourceRoute] {
        &self.source_routes[..usize::from(self.source_route_count)]
    }

    pub fn validate(&self) -> bool {
        let source_routes = self.source_routes();
        let source_bounds = source_routes.iter().fold(None, |bounds, route| {
            Some(bounds.map_or(route.source_range, |bounds: ColumnRange| ColumnRange {
                start: bounds.start.min(route.source_range.start),
                end: bounds.end.max(route.source_range.end),
            }))
        });
        self.source_range.start <= self.source_range.end &&
            self.destination_range.start <= self.destination_range.end &&
            self.source_route_count > 0 &&
            usize::from(self.source_route_count) <= GPU_ROUTE_MAX_SOURCES &&
            source_bounds == Some(self.source_range) &&
            source_routes.iter().enumerate().all(|(index, route)| {
                !route.source_range.is_empty() &&
                    route.source_range.start >= self.source_range.start &&
                    route.source_range.end <= self.source_range.end &&
                    source_routes[..index]
                        .iter()
                        .filter(|prior| prior.source_owner == route.source_owner)
                        .all(|prior| {
                            prior.source_range.end < route.source_range.start ||
                                route.source_range.end < prior.source_range.start
                        })
            }) &&
            match self.route {
                GpuTransferRoute::Resident => {
                    self.source_device.is_some() &&
                        self.destination_device == self.source_device &&
                        self.host_staging_bytes == 0 &&
                        self.pinned_host_staging_bytes == 0 &&
                        source_routes.iter().all(|route| {
                            route.source_owner == self.destination_device.unwrap() &&
                                route.route == GpuTransferRoute::Resident &&
                                route.source_staging_bytes == 0 &&
                                route.host_staging_bytes == 0 &&
                                route.pinned_host_staging_bytes == 0
                        })
                }
                GpuTransferRoute::Peer => {
                    self.source_device.is_some() &&
                        self.destination_device.is_some() &&
                        self.source_device != self.destination_device &&
                        self.host_staging_bytes == 0 &&
                        self.pinned_host_staging_bytes == 0
                }
                GpuTransferRoute::HostStaging => {
                    self.source_device.is_some() &&
                        self.destination_device.is_some() &&
                        // A host codec boundary uses one device owner for
                        // both ends of its host/device envelope. It is still
                        // a real transfer, distinguished by nonzero payload;
                        // zero-byte same-owner staging is never a route.
                        (self.source_device != self.destination_device || self.host_staging_bytes > 0)
                }
            }
    }

    pub fn with_staging_bytes(mut self, source: usize, host: usize, pinned_host: usize) -> Self {
        self.source_staging_bytes = source;
        self.host_staging_bytes = host;
        self.pinned_host_staging_bytes = pinned_host;
        self
    }

    /// Number of bytes moved by the physical route. Resident routes have no
    /// transfer profile. Peer and host staging record the payload once even
    /// though the route descriptor carries both source and host staging
    /// ownership for memory accounting.
    pub fn transfer_bytes(&self) -> usize {
        match self.route {
            GpuTransferRoute::HostStaging => self.host_staging_bytes,
            GpuTransferRoute::Peer => self.source_staging_bytes,
            GpuTransferRoute::Resident => 0,
        }
    }

    pub fn has_transfer_bytes(&self) -> bool {
        self.transfer_bytes() > 0
    }
}

/// Shared typed request consumed by range-aware warmup adapters and the fixed
/// fleet lowering.  `inputs` is produced by the canonical mapper below and
/// therefore has the same logical/global-column semantics in both phases.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuExecutionRange {
    pub variant: GpuExecutionVariant,
    pub output: ColumnRange,
    pub inputs: Vec<InputColumnRange>,
    pub fragment: GpuFragmentClass,
}

impl GpuExecutionRange {
    /// Validate the global mapper result against the actual source storage.
    /// This guard is used at the fleet boundary before materialization and is
    /// deliberately strict: a caller cannot replace a non-zero mapped range
    /// with a rebased local `[0, width)` representative.
    pub fn validate_global_ranges(
        &self,
        source_columns: &[usize],
    ) -> Result<(), ColumnPolicyError> {
        if self.output.is_empty() ||
            self.inputs.iter().any(|input| {
                input.operand >= source_columns.len() ||
                    input.range.is_empty() ||
                    input.range.end > source_columns[input.operand]
            })
        {
            return Err(ColumnPolicyError::InvalidOutputRange {
                start: self.output.start,
                end: self.output.end,
                columns: self.output.end,
            });
        }
        Ok(())
    }

    pub fn transfer_route(
        &self,
        source_is_resident: bool,
        peer_available: bool,
    ) -> GpuTransferRoute {
        if source_is_resident {
            GpuTransferRoute::Resident
        } else if peer_available {
            GpuTransferRoute::Peer
        } else {
            GpuTransferRoute::HostStaging
        }
    }

    /// Build the route identity after the fleet has selected source and
    /// destination placements.  This is the single conversion point from
    /// the logical mapper to warmup's physical execution class.
    pub fn route_descriptor(
        &self,
        source_device: Option<usize>,
        destination_device: Option<usize>,
        source_is_resident: bool,
        peer_available: bool,
        source_compact: bool,
        destination_compact: bool,
    ) -> GpuExecutionRouteDescriptor {
        resolve_gpu_route(GpuRouteResolutionInput {
            source_device,
            destination_device,
            source_range: self.inputs.first().map_or(self.output, |input| input.range),
            destination_range: self.output,
            source_is_resident,
            peer_available,
            source_compact,
            destination_compact,
            fragment: self.fragment,
            source_staging_bytes: 0,
            host_staging_bytes: 0,
            pinned_host_staging_bytes: 0,
        })
    }
}

/// Classify an output width produced by a fixed job.  This helper is kept
/// pure so the same tail/fragment identity can be attached to a measured
/// representative and to a production job without inspecting GPU state.
pub const fn classify_gpu_fragment(planned_width: usize, output: ColumnRange) -> GpuFragmentClass {
    if output.is_empty() {
        GpuFragmentClass::Mapped
    } else if output.len() == planned_width {
        GpuFragmentClass::PlannedJob
    } else if output.len() < planned_width {
        GpuFragmentClass::Tail
    } else {
        GpuFragmentClass::Mapped
    }
}

/// Build the canonical typed range request used by production and warmup.
/// No range is accepted until it has been validated and lowered through the
/// operation allowlist.
pub fn gpu_execution_range(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    output_columns: usize,
    output: ColumnRange,
    planned_width: usize,
) -> Result<GpuExecutionRange, ColumnPolicyError> {
    let inputs = map_output_range_to_inputs_with_output(kind, arguments, output_columns, output)?;
    let variant = match kind {
        NodeKind::Slice { .. } => GpuExecutionVariant::MappedSlice,
        NodeKind::Transpose => GpuExecutionVariant::MappedTranspose,
        NodeKind::Tensor => GpuExecutionVariant::MappedTensor,
        NodeKind::Concat { axis: ConcatAxis::Rows } => GpuExecutionVariant::MappedConcatRows,
        NodeKind::Concat { axis: ConcatAxis::Columns } => GpuExecutionVariant::MappedConcatColumns,
        NodeKind::Concat { axis: ConcatAxis::Diagonal } => {
            GpuExecutionVariant::MappedConcatDiagonal
        }
        NodeKind::MatrixMulSmallRhs => GpuExecutionVariant::CompactFragmentMultiply,
        NodeKind::PreimageSample { .. } => GpuExecutionVariant::PreimageSampling,
        NodeKind::Input { .. } |
        NodeKind::ConstantInt(_) |
        NodeKind::EvaluateInt(_) |
        NodeKind::ConstantReal(_) |
        NodeKind::ConstantBool(_) |
        NodeKind::ConstantMatrix { .. } |
        NodeKind::GadgetTrapdoor { .. } |
        NodeKind::TrapdoorPublic |
        NodeKind::IntBinary(_) |
        NodeKind::IntCompare(_) |
        NodeKind::BitExtract { .. } |
        NodeKind::IntToReal |
        NodeKind::BoolToInt |
        NodeKind::RealBinary(_) |
        NodeKind::RealSqrt |
        NodeKind::MatrixBinary(_) |
        NodeKind::MatrixMulAccumulate { .. } |
        NodeKind::MatrixNegate |
        NodeKind::MatrixScale { .. } |
        NodeKind::RingAutomorphism { .. } |
        NodeKind::ModulusSwitch { .. } |
        NodeKind::ModulusReduce { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::CenteredRoundDivide { .. } |
        NodeKind::BlockModSwitch { .. } |
        NodeKind::RnsModUp { .. } |
        NodeKind::RnsModDown { .. } |
        NodeKind::CrtRecompose { .. } |
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } |
        NodeKind::HashIntFamily { .. } |
        NodeKind::TrapdoorSample { .. } |
        NodeKind::GadgetDecompose { .. } |
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
        NodeKind::Select { .. } => GpuExecutionVariant::Primitive(effective_gpu_operation(kind)),
    };
    let fragment = classify_gpu_fragment(planned_width, output);
    Ok(GpuExecutionRange { variant, output, inputs, fragment })
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

/// Closed execution disposition shared by policy and capture lowering.
/// Every DSL node must choose one concrete execution family here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuNodeDisposition {
    Input,
    Resident,
    TypedReal,
    NativeAlias,
    Native(EffectiveGpuOperation),
}

pub fn gpu_node_disposition(kind: &NodeKind) -> GpuNodeDisposition {
    match kind {
        NodeKind::ConstantMatrix { value, .. } => match value {
            ConstantMatrix::Zero |
            ConstantMatrix::Identity |
            ConstantMatrix::UnitRow { .. } |
            ConstantMatrix::UnitColumn { .. } |
            ConstantMatrix::Gadget { .. } => {
                GpuNodeDisposition::Native(EffectiveGpuOperation::GeneratedConstant)
            }
            ConstantMatrix::PowerOfBase { .. } |
            ConstantMatrix::Rotation { .. } |
            ConstantMatrix::Polynomial { .. } => {
                GpuNodeDisposition::Native(EffectiveGpuOperation::SingleDeviceConstant)
            }
        },
        NodeKind::UniformResidueSample { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::UniformResidueSample)
        }
        NodeKind::UniformIntervalSample { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::UniformIntervalSample)
        }
        NodeKind::GaussianSample { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::GaussianSample)
        }
        NodeKind::HashSample { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::HashSample)
        }
        NodeKind::HashIntFamily { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::HashIntFamily)
        }
        NodeKind::TrapdoorSample { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::TrapdoorSample)
        }
        NodeKind::GadgetTrapdoor { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::GadgetTrapdoor)
        }
        NodeKind::LiftIntegerToConstantPolynomial { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::LiftIntegerToConstantPolynomial)
        }
        NodeKind::TrapdoorPublic => GpuNodeDisposition::NativeAlias,
        NodeKind::PreimageSample { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::PreimageSample)
        }
        NodeKind::GadgetDecompose { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::GadgetDecompose)
        }
        NodeKind::MatrixScale { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::MatrixScale)
        }
        NodeKind::MatrixNegate => GpuNodeDisposition::Native(EffectiveGpuOperation::MatrixNegate),
        NodeKind::RingAutomorphism { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::RingAutomorphism)
        }
        NodeKind::ModulusSwitch { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::ModulusSwitch)
        }
        NodeKind::ModulusReduce { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::ModulusReduce)
        }
        NodeKind::CenteredRebase { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::CenteredRebase)
        }
        NodeKind::CenteredRoundDivide { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::CenteredRoundDivide)
        }
        NodeKind::BlockModSwitch { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::BlockModSwitch)
        }
        NodeKind::RnsModUp { .. } => GpuNodeDisposition::Native(EffectiveGpuOperation::RnsModUp),
        NodeKind::RnsModDown { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::RnsModDown)
        }
        NodeKind::CrtRecompose { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::CrtRecompose)
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Add) => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::MatrixAdd)
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::MatrixSubtract)
        }
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::MatrixMultiply)
        }
        NodeKind::MatrixMulAccumulate { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::MatrixMulAccumulate)
        }
        NodeKind::MatrixMulSmallRhs => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::MatrixMulSmallRhs)
        }
        NodeKind::Transpose => GpuNodeDisposition::Native(EffectiveGpuOperation::Transpose),
        NodeKind::Slice { .. } => GpuNodeDisposition::Native(EffectiveGpuOperation::Slice),
        NodeKind::Tensor => GpuNodeDisposition::Native(EffectiveGpuOperation::Tensor),
        NodeKind::Concat { axis: ConcatAxis::Rows } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::ConcatRows)
        }
        NodeKind::Concat { axis: ConcatAxis::Columns } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::ConcatColumns)
        }
        NodeKind::Concat { axis: ConcatAxis::Diagonal } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::ConcatDiagonal)
        }
        NodeKind::Input { .. } => GpuNodeDisposition::Input,
        NodeKind::ConstantReal(_) |
        NodeKind::IntToReal |
        NodeKind::RealBinary(_) |
        NodeKind::RealSqrt => GpuNodeDisposition::TypedReal,
        NodeKind::ConstantInt(_) |
        NodeKind::EvaluateInt(_) |
        NodeKind::ConstantBool(_) |
        NodeKind::IntBinary(_) |
        NodeKind::IntCompare(_) |
        NodeKind::BitExtract { .. } |
        NodeKind::BoolToInt |
        NodeKind::SubgraphCall(_) |
        NodeKind::ParallelLoop(_) |
        NodeKind::SequentialLoop(_) |
        NodeKind::FamilyPack { .. } |
        NodeKind::FamilyGetStatic { .. } |
        NodeKind::FamilyGetDynamic |
        NodeKind::Select { .. } => GpuNodeDisposition::Resident,
        NodeKind::ExtractCoefficient { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::ExtractCoefficient)
        }
        NodeKind::ThresholdDecode { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::ThresholdDecode)
        }
        NodeKind::PackPolynomialCoefficients { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::PackPolynomialCoefficients)
        }
        NodeKind::PolynomialFromValues { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::PolynomialFromValues)
        }
        NodeKind::PolynomialValues { .. } => {
            GpuNodeDisposition::Native(EffectiveGpuOperation::PolynomialValues)
        }
    }
}

/// Refine the kind-only disposition with the validated wire shapes.  Matrix
/// families stay in the resident control path: their owners are retained by
/// the frame and dynamic selection is performed by the device gather.  Static
/// matrix selection remains a native alias because its index is frozen.
pub fn gpu_node_disposition_for_types(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    outputs: &[ConcreteWireType],
) -> GpuNodeDisposition {
    let matrix_family_static_alias = matches!(kind, NodeKind::FamilyGetStatic { .. }) &&
        arguments.first().is_some_and(matrix_family_type) &&
        outputs.first().is_some_and(|ty| ty.matrix_type().is_some());
    if matrix_family_static_alias {
        GpuNodeDisposition::NativeAlias
    } else {
        gpu_node_disposition(kind)
    }
}

fn matrix_family_type(ty: &ConcreteWireType) -> bool {
    matches!(
        ty,
        ConcreteWireType::IndexedFamily { element, .. }
            if element.matrix_type().is_some()
    )
}

pub fn effective_gpu_operation(kind: &NodeKind) -> EffectiveGpuOperation {
    match gpu_node_disposition(kind) {
        GpuNodeDisposition::Input |
        GpuNodeDisposition::Resident |
        GpuNodeDisposition::TypedReal => EffectiveGpuOperation::HostOrControl,
        GpuNodeDisposition::NativeAlias => EffectiveGpuOperation::TrapdoorPublic,
        GpuNodeDisposition::Native(operation) => operation,
    }
}

pub fn effective_gpu_operation_for_types(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    outputs: &[ConcreteWireType],
) -> EffectiveGpuOperation {
    match gpu_node_disposition_for_types(kind, arguments, outputs) {
        GpuNodeDisposition::Input |
        GpuNodeDisposition::Resident |
        GpuNodeDisposition::TypedReal => EffectiveGpuOperation::HostOrControl,
        GpuNodeDisposition::NativeAlias => EffectiveGpuOperation::TrapdoorPublic,
        GpuNodeDisposition::Native(operation) => operation,
    }
}

pub fn is_resident_control_operation(kind: &NodeKind) -> bool {
    matches!(gpu_node_disposition(kind), GpuNodeDisposition::Resident)
}

pub fn is_resident_control_operation_for_types(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    outputs: &[ConcreteWireType],
) -> bool {
    matches!(gpu_node_disposition_for_types(kind, arguments, outputs), GpuNodeDisposition::Resident)
}

/// Whether an operation is explicitly known to remain on the host/control
/// path. This is deliberately narrower than "not column-separable".
pub fn known_host_or_control(kind: &NodeKind) -> bool {
    matches!(effective_gpu_operation(kind), EffectiveGpuOperation::HostOrControl)
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
        EffectiveGpuOperation::LiftIntegerToConstantPolynomial |
        EffectiveGpuOperation::UniformResidueSample |
        EffectiveGpuOperation::UniformIntervalSample |
        EffectiveGpuOperation::GaussianSample |
        EffectiveGpuOperation::HashSample => ColumnCapability::GeneratedColumns,
        EffectiveGpuOperation::HashIntFamily => ColumnCapability::SingleDevice,
        EffectiveGpuOperation::SingleDeviceConstant |
        EffectiveGpuOperation::TrapdoorSample |
        EffectiveGpuOperation::GadgetTrapdoor |
        EffectiveGpuOperation::ExtractCoefficient |
        EffectiveGpuOperation::ThresholdDecode |
        EffectiveGpuOperation::PackPolynomialCoefficients |
        EffectiveGpuOperation::PolynomialFromValues |
        EffectiveGpuOperation::PolynomialValues => ColumnCapability::SingleDevice,
        EffectiveGpuOperation::PreimageSample | EffectiveGpuOperation::GadgetDecompose => {
            ColumnCapability::FixedOperandColumns
        }
        EffectiveGpuOperation::MatrixScale |
        EffectiveGpuOperation::MatrixNegate |
        EffectiveGpuOperation::RingAutomorphism |
        EffectiveGpuOperation::ModulusSwitch |
        EffectiveGpuOperation::ModulusReduce |
        EffectiveGpuOperation::CenteredRebase |
        EffectiveGpuOperation::CenteredRoundDivide |
        EffectiveGpuOperation::BlockModSwitch |
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
        EffectiveGpuOperation::TrapdoorPublic => ColumnCapability::NativeAlias,
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
    let columns = if is_resident_control_operation(kind) {
        resident_output_columns(kind, arguments)?
    } else if capability == ColumnCapability::GeneratedColumns {
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
    if is_resident_control_operation(kind) {
        return map_resident_control_range(kind, arguments, output_columns, output);
    }
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
        EffectiveGpuOperation::TrapdoorPublic => {
            if arguments.len() != 1 {
                return Err(ColumnPolicyError::MissingOperand { operand: 0 });
            }
            if let Some(matrix) = matrix_type(&arguments[0]) {
                if output.end > matrix.columns {
                    return Err(ColumnPolicyError::InvalidOutputRange {
                        start: output.start,
                        end: output.end,
                        columns: matrix.columns,
                    });
                }
            }
            return Ok(if output.is_empty() {
                Vec::new()
            } else {
                vec![InputColumnRange { operand: 0, range: output }]
            });
        }
        EffectiveGpuOperation::GeneratedConstant |
        EffectiveGpuOperation::SingleDeviceConstant |
        EffectiveGpuOperation::UniformResidueSample |
        EffectiveGpuOperation::UniformIntervalSample |
        EffectiveGpuOperation::GaussianSample |
        EffectiveGpuOperation::HashSample |
        EffectiveGpuOperation::HashIntFamily |
        EffectiveGpuOperation::TrapdoorSample |
        EffectiveGpuOperation::GadgetTrapdoor |
        EffectiveGpuOperation::LiftIntegerToConstantPolynomial |
        EffectiveGpuOperation::ExtractCoefficient |
        EffectiveGpuOperation::ThresholdDecode |
        EffectiveGpuOperation::PackPolynomialCoefficients |
        EffectiveGpuOperation::PolynomialFromValues |
        EffectiveGpuOperation::PolynomialValues |
        EffectiveGpuOperation::PreimageSample |
        EffectiveGpuOperation::GadgetDecompose |
        EffectiveGpuOperation::MatrixScale |
        EffectiveGpuOperation::MatrixNegate |
        EffectiveGpuOperation::RingAutomorphism |
        EffectiveGpuOperation::ModulusSwitch |
        EffectiveGpuOperation::ModulusReduce |
        EffectiveGpuOperation::CenteredRebase |
        EffectiveGpuOperation::CenteredRoundDivide |
        EffectiveGpuOperation::BlockModSwitch |
        EffectiveGpuOperation::RnsModUp |
        EffectiveGpuOperation::RnsModDown |
        EffectiveGpuOperation::CrtRecompose |
        EffectiveGpuOperation::MatrixAdd |
        EffectiveGpuOperation::MatrixSubtract |
        EffectiveGpuOperation::MatrixMultiply |
        EffectiveGpuOperation::MatrixMulAccumulate |
        EffectiveGpuOperation::MatrixMulSmallRhs |
        EffectiveGpuOperation::Transpose |
        EffectiveGpuOperation::Slice |
        EffectiveGpuOperation::Tensor |
        EffectiveGpuOperation::ConcatRows |
        EffectiveGpuOperation::ConcatColumns |
        EffectiveGpuOperation::ConcatDiagonal => {}
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
            // These constants are validated as 1x1 values and are executed
            // as one complete operation on a single device.  They have no
            // input columns to lower, but must still participate in the
            // range contract so the production and warmup paths can share
            // the same mapper.
            ConstantMatrix::PowerOfBase { .. } |
            ConstantMatrix::Rotation { .. } |
            ConstantMatrix::Polynomial { .. } => {}
        },
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } |
        NodeKind::HashIntFamily { .. } |
        NodeKind::LiftIntegerToConstantPolynomial { .. } |
        NodeKind::PackPolynomialCoefficients { .. } |
        NodeKind::PolynomialFromValues { .. } => {}
        NodeKind::TrapdoorSample { .. } => {
            return Err(ColumnPolicyError::UnsupportedOperation { operation: "trapdoor sampling" });
        }
        NodeKind::GadgetTrapdoor { .. } => {
            return Err(ColumnPolicyError::UnsupportedOperation {
                operation: "single-device operation",
            });
        }
        NodeKind::TrapdoorPublic => {
            if arguments.len() != 1 {
                return Err(ColumnPolicyError::MissingOperand { operand: 0 });
            }
            push(0, output);
        }
        NodeKind::MatrixScale { .. } |
        NodeKind::MatrixNegate |
        NodeKind::RingAutomorphism { .. } |
        NodeKind::ModulusSwitch { .. } |
        NodeKind::ModulusReduce { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::CenteredRoundDivide { .. } |
        NodeKind::BlockModSwitch { .. } |
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
            // Lowered row-block products retain each physical LHS, followed
            // by their shared compact RHS, instead of a logical concat.
            let rhs = arguments
                .len()
                .checked_sub(1)
                .filter(|rhs| *rhs > 0)
                .ok_or(ColumnPolicyError::ArithmeticOverflow)?;
            let right = matrix_at(arguments, rhs)?;
            for operand in 0..rhs {
                let left = matrix_at(arguments, operand)?;
                push(
                    operand,
                    ColumnRange::new(0, left.columns)
                        .ok_or(ColumnPolicyError::ArithmeticOverflow)?,
                );
            }
            if output.end > right.columns {
                return Err(ColumnPolicyError::InvalidOutputRange {
                    start: output.start,
                    end: output.end,
                    columns: right.columns,
                });
            }
            push(rhs, output);
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
            // A public GadgetTrapdoor-backed preimage is lowered to the
            // fixed decomposition kernel.  Its effective descriptor contains
            // only the target operand, whereas a sampled preimage retains the
            // public/trapdoor/target triple and uses the ordinary preimage
            // mapping.
            if arguments.len() == 1 {
                let input = matrix_at(arguments, 0)?;
                if output.end > input.columns {
                    return Err(ColumnPolicyError::InvalidOutputRange {
                        start: output.start,
                        end: output.end,
                        columns: input.columns,
                    });
                }
                push(0, output);
                return Ok(ranges);
            }
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
        NodeKind::SubgraphCall(_) |
        NodeKind::ParallelLoop(_) |
        NodeKind::SequentialLoop(_) |
        NodeKind::FamilyPack { .. } |
        NodeKind::FamilyGetStatic { .. } |
        NodeKind::FamilyGetDynamic |
        NodeKind::Select { .. } |
        NodeKind::ExtractCoefficient { .. } |
        NodeKind::ThresholdDecode { .. } |
        NodeKind::PolynomialValues { .. } => {
            return Err(ColumnPolicyError::UnsupportedOperation {
                operation: "host/control operation",
            });
        }
    }
    let _ = capability;
    Ok(ranges)
}

fn resident_integer(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::Int | ConcreteWireType::ConstantInt)
}

fn resident_bool(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::Bool | ConcreteWireType::ConstantBool)
}

fn resident_family(ty: &ConcreteWireType) -> Option<usize> {
    match ty {
        ConcreteWireType::IndexedFamily { element, count }
            if resident_integer(element) || resident_bool(element) =>
        {
            Some(*count)
        }
        _ => None,
    }
}

fn resident_output_columns(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
) -> Result<usize, ColumnPolicyError> {
    match kind {
        NodeKind::ConstantInt(_) | NodeKind::EvaluateInt(_) | NodeKind::ConstantBool(_) => Ok(1),
        NodeKind::IntBinary(_) | NodeKind::IntCompare(_) => {
            if arguments.iter().all(resident_integer) {
                Ok(1)
            } else {
                Err(ColumnPolicyError::NonMatrixOperand { operand: 0 })
            }
        }
        NodeKind::BitExtract { .. } => {
            if arguments.first().is_some_and(resident_integer) {
                Ok(1)
            } else {
                Err(ColumnPolicyError::NonMatrixOperand { operand: 0 })
            }
        }
        NodeKind::BoolToInt => {
            if arguments.first().is_some_and(resident_bool) {
                Ok(1)
            } else {
                Err(ColumnPolicyError::NonMatrixOperand { operand: 0 })
            }
        }
        NodeKind::FamilyPack { .. } => {
            if arguments.iter().all(|ty| resident_integer(ty) || resident_bool(ty)) {
                Ok(arguments.len())
            } else {
                Err(ColumnPolicyError::NonMatrixOperand {
                    operand: arguments
                        .iter()
                        .position(|ty| !resident_integer(ty) && !resident_bool(ty))
                        .unwrap_or(0),
                })
            }
        }
        NodeKind::FamilyGetStatic { .. } | NodeKind::FamilyGetDynamic => Ok(1),
        NodeKind::Select { .. } => Ok(1),
        NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_) | NodeKind::SubgraphCall(_) => {
            Ok(arguments
                .iter()
                .filter_map(matrix_type)
                .map(|matrix| matrix.columns)
                .max()
                .unwrap_or(1))
        }
        _ => {
            Err(ColumnPolicyError::UnsupportedOperation { operation: "resident control operation" })
        }
    }
}

fn map_resident_control_range(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    output_columns: usize,
    output: ColumnRange,
) -> Result<Vec<InputColumnRange>, ColumnPolicyError> {
    if output.start > output.end || output.end > output_columns {
        return Err(ColumnPolicyError::InvalidOutputRange {
            start: output.start,
            end: output.end,
            columns: output_columns,
        });
    }
    if output.is_empty() {
        return Ok(Vec::new());
    }
    let mut ranges = Vec::new();
    let mut push = |operand: usize, range: ColumnRange| {
        if !range.is_empty() {
            ranges.push(InputColumnRange { operand, range });
        }
    };
    let scalar = ColumnRange { start: 0, end: 1 };
    match kind {
        NodeKind::ConstantInt(_) | NodeKind::EvaluateInt(_) | NodeKind::ConstantBool(_) => {}
        NodeKind::IntBinary(_) | NodeKind::IntCompare(_) => {
            for (operand, ty) in arguments.iter().enumerate() {
                if !resident_integer(ty) {
                    return Err(ColumnPolicyError::NonMatrixOperand { operand });
                }
                push(operand, scalar);
            }
        }
        NodeKind::BitExtract { .. } => {
            if !arguments.first().is_some_and(resident_integer) {
                return Err(ColumnPolicyError::NonMatrixOperand { operand: 0 });
            }
            push(0, scalar);
        }
        NodeKind::BoolToInt => {
            if !arguments.first().is_some_and(resident_bool) {
                return Err(ColumnPolicyError::NonMatrixOperand { operand: 0 });
            }
            push(0, scalar);
        }
        NodeKind::FamilyPack { .. } => {
            for (operand, ty) in arguments.iter().enumerate() {
                if !resident_integer(ty) && !resident_bool(ty) {
                    return Err(ColumnPolicyError::NonMatrixOperand { operand });
                }
                push(operand, scalar);
            }
        }
        NodeKind::FamilyGetStatic { index } => {
            let count = resident_family(
                arguments.first().ok_or(ColumnPolicyError::MissingOperand { operand: 0 })?,
            )
            .ok_or(ColumnPolicyError::NonMatrixOperand { operand: 0 })?;
            let range = constant_usize(index)
                .and_then(|index| index.checked_add(1).map(|end| ColumnRange { start: index, end }))
                .filter(|range| range.end <= count)
                .unwrap_or(ColumnRange { start: 0, end: count });
            push(0, range);
        }
        NodeKind::FamilyGetDynamic => {
            let count = resident_family(
                arguments.first().ok_or(ColumnPolicyError::MissingOperand { operand: 0 })?,
            )
            .ok_or(ColumnPolicyError::NonMatrixOperand { operand: 0 })?;
            push(0, ColumnRange { start: 0, end: count });
            if !arguments.get(1).is_some_and(resident_integer) {
                return Err(ColumnPolicyError::NonMatrixOperand { operand: 1 });
            }
            push(1, scalar);
        }
        NodeKind::Select { .. } => {
            if !arguments.first().is_some_and(resident_integer) {
                return Err(ColumnPolicyError::NonMatrixOperand { operand: 0 });
            }
            push(0, scalar);
            for (operand, ty) in arguments.iter().enumerate().skip(1) {
                if !resident_integer(ty) && !resident_bool(ty) {
                    return Err(ColumnPolicyError::NonMatrixOperand { operand });
                }
                push(operand, scalar);
            }
        }
        NodeKind::ParallelLoop(spec) => {
            for (operand, ty) in arguments.iter().enumerate() {
                let range = match spec
                    .input_modes
                    .get(operand)
                    .copied()
                    .unwrap_or(LoopInputMode::Broadcast)
                {
                    LoopInputMode::Broadcast => scalar,
                    LoopInputMode::Zip => {
                        let width = matrix_type(ty).map_or(1, |matrix| matrix.columns);
                        if output.end > width {
                            return Err(ColumnPolicyError::InvalidOutputRange {
                                start: output.start,
                                end: output.end,
                                columns: width,
                            });
                        }
                        output
                    }
                    LoopInputMode::ZipOffset { offset } => {
                        let mapped = output
                            .checked_translate(offset)
                            .ok_or(ColumnPolicyError::ArithmeticOverflow)?;
                        let width = matrix_type(ty).map_or(1, |matrix| matrix.columns);
                        if mapped.end > width {
                            return Err(ColumnPolicyError::InvalidOutputRange {
                                start: mapped.start,
                                end: mapped.end,
                                columns: width,
                            });
                        }
                        mapped
                    }
                };
                push(operand, range);
            }
        }
        NodeKind::SequentialLoop(_) | NodeKind::SubgraphCall(_) => {
            for (operand, ty) in arguments.iter().enumerate() {
                let width = matrix_type(ty).map_or(1, |matrix| matrix.columns);
                if output.end > width {
                    return Err(ColumnPolicyError::InvalidOutputRange {
                        start: output.start,
                        end: output.end,
                        columns: width,
                    });
                }
                push(operand, if matrix_type(ty).is_some() { output } else { scalar });
            }
        }
        _ => {
            return Err(ColumnPolicyError::UnsupportedOperation {
                operation: "resident control operation",
            })
        }
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        expr::{IntExpr, RealExpr},
        node::{
            ConcatAxis, ConstantMatrix, HashVariant, IndexRange, IntBinaryOp, IntCompareOp,
            MatrixBinaryOp, ParallelLoop, RealBinaryOp, SampleRange, SequentialLoop, SubgraphCall,
        },
        types::{MatrixType, WireType},
    };
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
    fn every_dsl_primitive_has_an_explicit_gpu_or_resident_lowering() {
        let scalar_matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let real = RealExpr::FromInt(IntExpr::constant(1));
        let integer = IntExpr::constant(1);
        let primitives = vec![
            (
                "input",
                NodeKind::Input { name: "x".into(), wire_type: WireType::Int, artifact: None },
            ),
            ("constant_int", NodeKind::ConstantInt(BigInt::from(1))),
            ("evaluate_int", NodeKind::EvaluateInt(integer.clone())),
            ("constant_real", NodeKind::ConstantReal(real.clone())),
            ("constant_bool", NodeKind::ConstantBool(true)),
            (
                "constant_matrix",
                NodeKind::ConstantMatrix {
                    matrix_type: scalar_matrix.clone(),
                    value: ConstantMatrix::Zero,
                },
            ),
            (
                "gadget_trapdoor",
                NodeKind::GadgetTrapdoor {
                    matrix_type: scalar_matrix.clone(),
                    base: integer.clone(),
                },
            ),
            ("trapdoor_public", NodeKind::TrapdoorPublic),
            ("int_binary", NodeKind::IntBinary(IntBinaryOp::Add)),
            ("int_compare", NodeKind::IntCompare(IntCompareOp::Equal)),
            ("bit_extract", NodeKind::BitExtract { bit: integer.clone() }),
            ("int_to_real", NodeKind::IntToReal),
            ("bool_to_int", NodeKind::BoolToInt),
            ("real_binary", NodeKind::RealBinary(RealBinaryOp::Add)),
            ("real_sqrt", NodeKind::RealSqrt),
            ("matrix_binary", NodeKind::MatrixBinary(MatrixBinaryOp::Add)),
            (
                "matrix_mul_accumulate",
                NodeKind::MatrixMulAccumulate {
                    coefficients: vec![integer.clone()],
                    has_bias: true,
                },
            ),
            ("matrix_mul_small_rhs", NodeKind::MatrixMulSmallRhs),
            ("matrix_negate", NodeKind::MatrixNegate),
            ("matrix_scale", NodeKind::MatrixScale { scalar: integer.clone() }),
            ("ring_automorphism", NodeKind::RingAutomorphism { index: integer.clone() }),
            ("modulus_switch", NodeKind::ModulusSwitch { modulus: integer.clone() }),
            ("modulus_reduce", NodeKind::ModulusReduce { modulus: integer.clone() }),
            ("centered_rebase", NodeKind::CenteredRebase { modulus: integer.clone() }),
            ("centered_round_divide", NodeKind::CenteredRoundDivide { divisor: integer.clone() }),
            (
                "rns_mod_up",
                NodeKind::RnsModUp {
                    modulus: integer.clone(),
                    source_moduli: vec![17],
                    digit_size: 1,
                    normalize: false,
                },
            ),
            (
                "rns_mod_down",
                NodeKind::RnsModDown {
                    modulus: integer.clone(),
                    source_moduli: vec![17],
                    plaintext_modulus: integer.clone(),
                },
            ),
            (
                "block_mod_switch",
                NodeKind::BlockModSwitch {
                    modulus: integer.clone(),
                    source_moduli: vec![17],
                    plaintext_modulus: integer.clone(),
                },
            ),
            ("transpose", NodeKind::Transpose),
            (
                "slice",
                NodeKind::Slice {
                    rows: None,
                    columns: Some(IndexRange { start: integer.clone(), end: IntExpr::constant(2) }),
                },
            ),
            ("tensor", NodeKind::Tensor),
            ("concat", NodeKind::Concat { axis: ConcatAxis::Rows }),
            (
                "uniform_residue_sample",
                NodeKind::UniformResidueSample { matrix_type: scalar_matrix.clone() },
            ),
            (
                "uniform_interval_sample",
                NodeKind::UniformIntervalSample {
                    matrix_type: scalar_matrix.clone(),
                    range: SampleRange {
                        minimum: IntExpr::constant(-1),
                        maximum: IntExpr::constant(1),
                    },
                },
            ),
            (
                "gaussian_sample",
                NodeKind::GaussianSample {
                    matrix_type: scalar_matrix.clone(),
                    sigma: real.clone(),
                    max_coefficient_bound: integer.clone(),
                },
            ),
            (
                "hash_sample",
                NodeKind::HashSample {
                    matrix_type: scalar_matrix.clone(),
                    variant: HashVariant::Plain,
                    tag_prefix: Vec::new(),
                    tag_components: Vec::new(),
                    base: None,
                    digit_count: None,
                },
            ),
            (
                "trapdoor_sample",
                NodeKind::TrapdoorSample {
                    matrix_type: scalar_matrix.clone(),
                    sigma: real.clone(),
                    gadget_base: integer.clone(),
                    digit_count: integer.clone(),
                    preimage_max_coefficient_bound: integer.clone(),
                },
            ),
            (
                "preimage_sample",
                NodeKind::PreimageSample {
                    matrix_type: scalar_matrix.clone(),
                    max_coefficient_bound: integer.clone(),
                },
            ),
            (
                "gadget_decompose",
                NodeKind::GadgetDecompose {
                    base: integer.clone(),
                    small: false,
                    digit_count: integer.clone(),
                },
            ),
            (
                "extract_coefficient",
                NodeKind::ExtractCoefficient {
                    position: integer.clone(),
                    canonical_input_exclusive_upper: None,
                },
            ),
            (
                "lift_integer",
                NodeKind::LiftIntegerToConstantPolynomial { matrix_type: scalar_matrix.clone() },
            ),
            (
                "threshold_decode",
                NodeKind::ThresholdDecode {
                    plaintext_modulus: integer.clone(),
                    length: integer.clone(),
                    output_bool: true,
                },
            ),
            (
                "crt_recompose",
                NodeKind::CrtRecompose {
                    modulus: integer.clone(),
                    plaintext_moduli: vec![integer.clone()],
                    reconstruction_coefficients: vec![integer.clone()],
                },
            ),
            (
                "pack_polynomial_coefficients",
                NodeKind::PackPolynomialCoefficients {
                    matrix_type: scalar_matrix.clone(),
                    coefficient_bits: integer.clone(),
                },
            ),
            (
                "polynomial_from_values",
                NodeKind::PolynomialFromValues {
                    matrix_type: scalar_matrix.clone(),
                    evaluation: true,
                },
            ),
            ("polynomial_values", NodeKind::PolynomialValues { evaluation: true }),
            (
                "subgraph_call",
                NodeKind::SubgraphCall(SubgraphCall {
                    definition: "coverage".into(),
                    bindings: Vec::new(),
                    canonical_input_exclusive_uppers: Vec::new(),
                }),
            ),
            (
                "parallel_loop",
                NodeKind::ParallelLoop(ParallelLoop {
                    count: integer.clone(),
                    minimum_count: 0,
                    index_slot: 0,
                    bindings: Vec::new(),
                    input_modes: Vec::new(),
                }),
            ),
            (
                "sequential_loop",
                NodeKind::SequentialLoop(SequentialLoop {
                    count: integer.clone(),
                    index_slot: 0,
                    bindings: Vec::new(),
                    carried_count: 0,
                }),
            ),
            ("family_pack", NodeKind::FamilyPack { count: integer.clone() }),
            ("family_get_static", NodeKind::FamilyGetStatic { index: integer.clone() }),
            ("family_get_dynamic", NodeKind::FamilyGetDynamic),
            ("select", NodeKind::Select { count: integer }),
        ];
        assert!(
            EffectiveGpuOperation::all()
                .iter()
                .all(|operation| { *operation != EffectiveGpuOperation::Unsupported })
        );
        for (name, kind) in primitives {
            let operation = effective_gpu_operation(&kind);
            assert_ne!(operation, EffectiveGpuOperation::Unsupported, "{name} is unclassified");
            assert!(
                EffectiveGpuOperation::all().contains(&operation),
                "{name} lacks operation inventory"
            );
            assert!(
                canonical_warmup_profile_domain(&kind).is_profileable(),
                "{name} lacks profile domain"
            );
            assert_ne!(
                column_capability(&kind, &[]),
                ColumnCapability::Unsupported,
                "{name} lacks capability"
            );
        }
    }

    #[test]
    fn static_family_get_lowers_to_a_bounded_resident_range() {
        let family = ConcreteWireType::IndexedFamily {
            element: Box::new(ConcreteWireType::Int),
            count: 8192,
        };
        let kind = NodeKind::FamilyGetStatic { index: IntExpr::constant(80) };
        assert_eq!(effective_gpu_operation(&kind), EffectiveGpuOperation::HostOrControl);
        assert!(is_resident_control_operation(&kind));
        assert_eq!(
            map_output_range_to_inputs_with_output(
                &kind,
                &[family],
                1,
                ColumnRange { start: 0, end: 1 },
            )
            .unwrap(),
            vec![InputColumnRange { operand: 0, range: ColumnRange { start: 80, end: 81 } }]
        );
    }

    #[test]
    fn dynamic_matrix_family_get_stays_in_resident_control() {
        let family = ConcreteWireType::IndexedFamily { element: Box::new(matrix(2, 4)), count: 4 };
        let kind = NodeKind::FamilyGetDynamic;
        assert_eq!(
            gpu_node_disposition_for_types(
                &kind,
                &[family, ConcreteWireType::Int],
                &[matrix(2, 4)],
            ),
            GpuNodeDisposition::Resident
        );
        assert_eq!(
            effective_gpu_operation_for_types(
                &kind,
                &[
                    ConcreteWireType::IndexedFamily { element: Box::new(matrix(2, 4)), count: 4 },
                    ConcreteWireType::Int,
                ],
                &[matrix(2, 4)],
            ),
            EffectiveGpuOperation::HostOrControl
        );
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
    fn all_constant_variants_have_a_range_mapping() {
        let matrix_type = mxx_ir_core::types::MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let constants = [
            ConstantMatrix::Zero,
            ConstantMatrix::Identity,
            ConstantMatrix::UnitRow { index: IntExpr::constant(0) },
            ConstantMatrix::UnitColumn { index: IntExpr::constant(0) },
            ConstantMatrix::Gadget { base: IntExpr::constant(2), small: false },
            ConstantMatrix::PowerOfBase {
                base: IntExpr::constant(2),
                exponent: IntExpr::constant(3),
            },
            ConstantMatrix::Rotation { exponent: IntExpr::constant(1) },
            ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(1)] },
        ];
        for value in constants {
            let kind = NodeKind::ConstantMatrix { matrix_type: matrix_type.clone(), value };
            assert_eq!(
                map_output_range_to_inputs_with_output(
                    &kind,
                    &[],
                    1,
                    ColumnRange { start: 0, end: 1 },
                )
                .unwrap(),
                Vec::<InputColumnRange>::new()
            );
        }
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
    fn operation_allowlist_keeps_real_host_nodes_explicit() {
        let kind = NodeKind::ConstantInt(num_bigint::BigInt::from(3));
        assert_eq!(effective_gpu_operation(&kind), EffectiveGpuOperation::HostOrControl);
        assert_eq!(column_capability(&kind, &[]), ColumnCapability::HostOrControl);
        assert_eq!(
            map_output_range_to_inputs(&kind, &[], ColumnRange { start: 0, end: 1 }),
            Ok(vec![])
        );
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

    #[test]
    fn canonical_domains_separate_constants_concat_axes_and_fused_kernels() {
        let ty = mxx_ir_core::types::MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(3),
        };
        let constant = |value| NodeKind::ConstantMatrix { matrix_type: ty.clone(), value };
        let domains = [
            canonical_warmup_profile_domain(&constant(ConstantMatrix::Zero)),
            canonical_warmup_profile_domain(&constant(ConstantMatrix::Identity)),
            canonical_warmup_profile_domain(&constant(ConstantMatrix::UnitRow {
                index: IntExpr::constant(0),
            })),
            canonical_warmup_profile_domain(&constant(ConstantMatrix::UnitColumn {
                index: IntExpr::constant(0),
            })),
            canonical_warmup_profile_domain(&constant(ConstantMatrix::Gadget {
                base: IntExpr::constant(2),
                small: false,
            })),
            canonical_warmup_profile_domain(&constant(ConstantMatrix::PowerOfBase {
                base: IntExpr::constant(2),
                exponent: IntExpr::constant(3),
            })),
            canonical_warmup_profile_domain(&constant(ConstantMatrix::Rotation {
                exponent: IntExpr::constant(1),
            })),
            canonical_warmup_profile_domain(&constant(ConstantMatrix::Polynomial {
                coefficients: vec![IntExpr::constant(1)],
            })),
        ];
        let unique = domains.iter().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), domains.len());

        let concat_domains = [
            canonical_warmup_profile_domain(&NodeKind::Concat { axis: ConcatAxis::Rows }),
            canonical_warmup_profile_domain(&NodeKind::Concat { axis: ConcatAxis::Columns }),
            canonical_warmup_profile_domain(&NodeKind::Concat { axis: ConcatAxis::Diagonal }),
        ];
        assert_eq!(
            concat_domains.iter().collect::<std::collections::BTreeSet<_>>().len(),
            concat_domains.len()
        );

        let fused = [
            fused_warmup_profile_domain(FusedWarmupOperation::RowSum),
            fused_warmup_profile_domain(FusedWarmupOperation::TensorRowSum),
            fused_warmup_profile_domain(FusedWarmupOperation::RowBlockAdd),
            fused_warmup_profile_domain(FusedWarmupOperation::Decompose),
            fused_warmup_profile_domain(FusedWarmupOperation::CompactProduct),
            fused_warmup_profile_domain(FusedWarmupOperation::PreimageBatch),
        ];
        assert_eq!(fused.iter().collect::<std::collections::BTreeSet<_>>().len(), fused.len());
        assert!(fused.iter().all(|domain| domain.is_gpu_measured()));
    }

    #[test]
    fn every_canonical_domain_has_explicit_measurement_owner() {
        let gpu = canonical_warmup_profile_domain(&NodeKind::MatrixNegate);
        let host = canonical_warmup_profile_domain(&NodeKind::ConstantReal(
            mxx_ir_core::expr::RealExpr::from_integer(1),
        ));
        assert_eq!(gpu.measurement_kind(), WarmupMeasurementKind::GpuMeasured);
        assert_eq!(host.measurement_kind(), WarmupMeasurementKind::HostMeasured);
        assert!(!host.is_gpu_measured());
        assert_ne!(gpu.identity(), host.identity());
    }

    #[test]
    fn host_projection_and_gpu_boundaries_have_distinct_measurement_routes() {
        assert_eq!(
            effective_gpu_operation(&NodeKind::TrapdoorPublic),
            EffectiveGpuOperation::TrapdoorPublic
        );
        assert!(EffectiveGpuOperation::all().contains(&EffectiveGpuOperation::TrapdoorPublic));
        assert!(!known_host_or_control(&NodeKind::TrapdoorPublic));
        assert_eq!(
            canonical_warmup_profile_domain(&NodeKind::TrapdoorPublic).measurement_kind(),
            WarmupMeasurementKind::HostMeasured
        );
        assert_eq!(
            canonical_warmup_profile_domain(&NodeKind::TrapdoorPublic),
            CanonicalWarmupProfileDomain::TrapdoorPublic
        );
        assert!(canonical_warmup_profile_domain(&NodeKind::TrapdoorPublic).is_profileable());
        assert_eq!(
            column_capability(&NodeKind::TrapdoorPublic, &[]),
            ColumnCapability::NativeAlias
        );
        assert!(matches!(
            map_output_range_to_inputs(
                &NodeKind::TrapdoorPublic,
                &[],
                ColumnRange { start: 0, end: 1 }
            ),
            Err(ColumnPolicyError::MissingOperand { operand: 0 })
        ));

        let matrix = matrix(1, 1);
        assert_eq!(
            map_output_range_to_inputs(
                &NodeKind::TrapdoorPublic,
                std::slice::from_ref(&matrix),
                ColumnRange { start: 0, end: 1 }
            )
            .expect("trapdoor public is a native alias"),
            vec![InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: 1 } }]
        );
        let extract = NodeKind::ExtractCoefficient {
            position: IntExpr::constant(0),
            canonical_input_exclusive_upper: None,
        };
        let threshold = NodeKind::ThresholdDecode {
            plaintext_modulus: IntExpr::constant(2),
            length: IntExpr::constant(1),
            output_bool: true,
        };
        for kind in [&extract, &threshold] {
            assert_eq!(
                canonical_warmup_profile_domain(kind).measurement_kind(),
                WarmupMeasurementKind::GpuMeasured
            );
            assert_eq!(
                column_capability(kind, std::slice::from_ref(&matrix)),
                ColumnCapability::SingleDevice
            );
        }
        assert_eq!(
            canonical_warmup_profile_domain(&extract).transfer_kind(),
            WarmupTransferKind::None
        );
        assert_eq!(
            canonical_warmup_profile_domain(&threshold).transfer_kind(),
            WarmupTransferKind::None
        );
    }

    #[test]
    fn transfer_coordinate_is_bytes_and_does_not_double_count_staging_owners() {
        let descriptor = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 2, end: 6 },
            GpuFragmentClass::PlannedJob,
        )
        .with_staging_bytes(128, 128, 64);
        // `with_staging_bytes` alone does not change a resident route: no
        // host transfer profile is created for a device-local invocation.
        assert_eq!(descriptor.transfer_bytes(), 0);

        let staged = GpuExecutionRouteDescriptor {
            route: GpuTransferRoute::HostStaging,
            source_device: Some(0),
            destination_device: Some(1),
            source_range: ColumnRange { start: 2, end: 6 },
            destination_range: ColumnRange { start: 0, end: 4 },
            source_compact: false,
            destination_compact: true,
            fragment: GpuFragmentClass::CompactFragment,
            source_staging_bytes: 128,
            host_staging_bytes: 128,
            pinned_host_staging_bytes: 64,
            source_route_count: 1,
            source_routes: [GpuExecutionSourceRoute::new(
                0,
                ColumnRange { start: 2, end: 6 },
                GpuTransferRoute::HostStaging,
                128,
                128,
                64,
            ); GPU_ROUTE_MAX_SOURCES],
        };
        assert!(staged.validate());
        let base = GpuRouteResolutionInput {
            source_device: Some(0),
            destination_device: Some(1),
            source_range: staged.source_range,
            destination_range: staged.destination_range,
            source_is_resident: false,
            peer_available: false,
            source_compact: false,
            destination_compact: false,
            fragment: GpuFragmentClass::Full,
            source_staging_bytes: 0,
            host_staging_bytes: 0,
            pinned_host_staging_bytes: 0,
        };
        let boundary = resolve_gpu_route(GpuRouteResolutionInput {
            source_device: Some(0),
            destination_device: Some(0),
            peer_available: false,
            source_staging_bytes: 64,
            host_staging_bytes: 64,
            pinned_host_staging_bytes: 0,
            ..base
        });
        assert!(boundary.validate());
        assert_eq!(boundary.transfer_bytes(), 64);
        let empty_boundary = resolve_gpu_route(GpuRouteResolutionInput {
            source_device: Some(0),
            destination_device: Some(0),
            peer_available: false,
            ..base
        });
        assert!(!empty_boundary.validate());
        assert_eq!(staged.transfer_bytes(), 128);
        assert!(staged.has_transfer_bytes());
    }

    #[test]
    fn route_identity_keeps_ordered_deduplicated_source_owners() {
        let descriptor = resolve_gpu_route(GpuRouteResolutionInput {
            source_device: Some(4),
            destination_device: Some(2),
            source_range: ColumnRange { start: 0, end: 8 },
            destination_range: ColumnRange { start: 0, end: 8 },
            source_is_resident: false,
            peer_available: false,
            source_compact: false,
            destination_compact: false,
            fragment: GpuFragmentClass::ConcatFragment,
            source_staging_bytes: 35,
            host_staging_bytes: 20,
            pinned_host_staging_bytes: 4,
        })
        .with_source_routes([
            GpuExecutionSourceRoute::new(
                4,
                ColumnRange { start: 0, end: 3 },
                GpuTransferRoute::Peer,
                10,
                0,
                0,
            ),
            GpuExecutionSourceRoute::new(
                7,
                ColumnRange { start: 3, end: 6 },
                GpuTransferRoute::HostStaging,
                20,
                20,
                4,
            ),
            GpuExecutionSourceRoute::new(
                4,
                ColumnRange { start: 6, end: 8 },
                GpuTransferRoute::Peer,
                5,
                0,
                0,
            ),
        ])
        .expect("bounded route inventory");
        assert!(descriptor.validate());
        assert_eq!(descriptor.source_routes().len(), 2);
        assert_eq!(descriptor.source_routes()[0].source_owner, 4);
        assert_eq!(descriptor.source_routes()[0].source_range, ColumnRange { start: 0, end: 8 });
        assert_eq!(descriptor.source_staging_bytes, 35);
        assert_eq!(descriptor.host_staging_bytes, 20);
    }

    #[test]
    fn route_identity_preserves_sparse_same_owner_fragments() {
        let descriptor = resolve_gpu_route(GpuRouteResolutionInput {
            source_device: Some(0),
            destination_device: Some(0),
            source_range: ColumnRange { start: 0, end: 3 },
            destination_range: ColumnRange { start: 2, end: 3 },
            source_is_resident: true,
            peer_available: false,
            source_compact: false,
            destination_compact: false,
            fragment: GpuFragmentClass::Mapped,
            source_staging_bytes: 0,
            host_staging_bytes: 0,
            pinned_host_staging_bytes: 0,
        })
        .with_source_routes([
            GpuExecutionSourceRoute::new(
                0,
                ColumnRange { start: 0, end: 1 },
                GpuTransferRoute::Resident,
                0,
                0,
                0,
            ),
            GpuExecutionSourceRoute::new(
                0,
                ColumnRange { start: 2, end: 3 },
                GpuTransferRoute::Resident,
                0,
                0,
                0,
            ),
        ])
        .expect("sparse route inventory");
        assert!(descriptor.validate());
        assert_eq!(descriptor.source_routes().len(), 2);
        assert_eq!(descriptor.source_routes()[0].source_range, ColumnRange { start: 0, end: 1 });
        assert_eq!(descriptor.source_routes()[1].source_range, ColumnRange { start: 2, end: 3 });
    }

    #[test]
    fn integer_lift_is_gpu_measured_and_range_generated() {
        let kind = NodeKind::LiftIntegerToConstantPolynomial {
            matrix_type: mxx_ir_core::types::MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(8),
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
            },
        };
        assert_eq!(
            effective_gpu_operation(&kind),
            EffectiveGpuOperation::LiftIntegerToConstantPolynomial
        );
        assert_eq!(
            canonical_warmup_profile_domain(&kind).measurement_kind(),
            WarmupMeasurementKind::GpuMeasured
        );
        assert_eq!(column_capability(&kind, &[]), ColumnCapability::GeneratedColumns);
        let request =
            gpu_execution_range(&kind, &[], 4, ColumnRange { start: 1, end: 3 }, 4).unwrap();
        assert!(request.inputs.is_empty());
    }

    #[test]
    fn resident_codec_boundaries_are_gpu_measured_without_transfer_routes() {
        let matrix_type = mxx_ir_core::types::MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let cases = [
            (
                NodeKind::PackPolynomialCoefficients {
                    matrix_type: matrix_type.clone(),
                    coefficient_bits: IntExpr::constant(1),
                },
                WarmupTransferKind::None,
            ),
            (
                NodeKind::PolynomialFromValues {
                    matrix_type: matrix_type.clone(),
                    evaluation: false,
                },
                WarmupTransferKind::None,
            ),
            (NodeKind::PolynomialValues { evaluation: false }, WarmupTransferKind::None),
        ];
        for (kind, transfer) in cases {
            let domain = canonical_warmup_profile_domain(&kind);
            assert_eq!(domain.measurement_kind(), WarmupMeasurementKind::GpuMeasured);
            assert_eq!(domain.transfer_kind(), transfer);
            assert!(!domain.has_transport_stage());
        }
        assert_eq!(
            canonical_warmup_profile_domain(&NodeKind::TrapdoorPublic).transfer_kind(),
            WarmupTransferKind::None
        );
    }

    #[test]
    fn canonical_domain_is_closed_and_every_entry_has_identity_and_measurement() {
        let all = CanonicalWarmupProfileDomain::all();
        let names =
            all.iter().map(|domain| domain.identity()).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), all.len());
        assert!(all.iter().all(|domain| !domain.identity().is_empty()));
        assert!(all.iter().all(|domain| matches!(
            domain.measurement_kind(),
            WarmupMeasurementKind::GpuMeasured | WarmupMeasurementKind::HostMeasured
        )));
        assert!(all.iter().all(|domain| match domain.transfer_kind() {
            WarmupTransferKind::None |
            WarmupTransferKind::HostToDevice |
            WarmupTransferKind::DeviceToHost |
            WarmupTransferKind::Peer |
            WarmupTransferKind::HostStaging => true,
        }));
    }

    #[test]
    fn primitive_and_transfer_inventories_are_closed_and_profileable() {
        let effective = EffectiveGpuOperation::all();
        assert!(!effective.is_empty());
        assert!(!effective.contains(&EffectiveGpuOperation::Unsupported));
        assert_eq!(
            effective.iter().collect::<std::collections::BTreeSet<_>>().len(),
            effective.len(),
            "duplicate effective primitive inventory entry"
        );
        for operation in effective {
            assert_ne!(
                capability_for_effective_operation(*operation, &[]),
                ColumnCapability::Unsupported,
                "effective primitive {operation:?} has no planner capability"
            );
        }

        let fused = FusedWarmupOperation::all();
        assert_eq!(fused.len(), 6);
        assert_eq!(
            fused
                .iter()
                .map(|operation| fused_warmup_profile_domain(*operation))
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            fused.len(),
            "fused dispatches must not share a profile domain"
        );
        assert!(fused.iter().all(|operation| {
            fused_warmup_profile_domain(*operation).measurement_kind() ==
                WarmupMeasurementKind::GpuMeasured
        }));

        assert_eq!(WarmupTransferKind::all().len(), 5);
        assert_eq!(WarmupTransferKind::all()[0], WarmupTransferKind::None);
        assert_eq!(WarmupTransferKind::all()[1], WarmupTransferKind::HostToDevice);
        assert_eq!(WarmupTransferKind::all()[2], WarmupTransferKind::DeviceToHost);
        assert!(CanonicalWarmupProfileDomain::all().iter().all(|domain| {
            domain.is_profileable() &&
                !domain.identity().is_empty() &&
                (!domain.has_transport_stage() ||
                    domain.measurement_kind() == WarmupMeasurementKind::HostMeasured)
        }));
    }

    #[test]
    fn typed_execution_range_preserves_mapped_global_and_local_ranges() {
        let args = vec![matrix(2, 8)];
        let kind = NodeKind::Slice {
            rows: None,
            columns: Some(IndexRange { start: IntExpr::constant(3), end: IntExpr::constant(7) }),
        };
        let request =
            gpu_execution_range(&kind, &args, 4, ColumnRange { start: 1, end: 3 }, 4).unwrap();
        assert_eq!(request.variant, GpuExecutionVariant::MappedSlice);
        assert_eq!(request.fragment, GpuFragmentClass::Tail);
        assert_eq!(request.inputs[0].range, ColumnRange { start: 4, end: 6 });
        assert_eq!(request.transfer_route(false, true), GpuTransferRoute::Peer);
        assert_eq!(request.transfer_route(false, false), GpuTransferRoute::HostStaging);
    }

    #[test]
    fn typed_job_range_preserves_global_mapper_boundary() {
        let args = vec![matrix(2, 8)];
        let kind = NodeKind::Slice {
            rows: None,
            columns: Some(IndexRange { start: IntExpr::constant(3), end: IntExpr::constant(7) }),
        };
        let request =
            gpu_execution_range(&kind, &args, 4, ColumnRange { start: 1, end: 3 }, 4).unwrap();
        assert!(request.validate_global_ranges(&[8]).is_ok());
        assert_eq!(request.inputs[0].range, ColumnRange { start: 4, end: 6 });
        assert_ne!(request.inputs[0].range.start, 0);
    }

    #[test]
    fn typed_execution_range_distinguishes_compact_and_tail_fragments() {
        let args = vec![
            matrix(2, 4),
            ConcreteWireType::SmallMatrix {
                matrix: ConcreteMatrixType {
                    modulus: BigInt::from(17),
                    ring_dimension: 8,
                    rows: 4,
                    columns: 9,
                },
                max_coefficient_bound: BigInt::from(3),
            },
        ];
        let request = gpu_execution_range(
            &NodeKind::MatrixMulSmallRhs,
            &args,
            9,
            ColumnRange { start: 0, end: 3 },
            4,
        )
        .unwrap();
        assert_eq!(request.variant, GpuExecutionVariant::CompactFragmentMultiply);
        assert_eq!(request.fragment, GpuFragmentClass::Tail);
        assert_eq!(request.inputs[0].range, ColumnRange { start: 0, end: 4 });
        assert_eq!(request.inputs[1].range, ColumnRange { start: 0, end: 3 });
    }

    #[test]
    fn lowered_compact_product_maps_every_physical_block_and_the_final_rhs() {
        let args = vec![matrix(2, 4), matrix(3, 4), matrix(4, 9)];
        let mapped = map_output_range_to_inputs_with_output(
            &NodeKind::MatrixMulSmallRhs,
            &args,
            9,
            ColumnRange { start: 5, end: 8 },
        )
        .unwrap();
        assert_eq!(
            mapped,
            vec![
                InputColumnRange { operand: 0, range: ColumnRange { start: 0, end: 4 } },
                InputColumnRange { operand: 1, range: ColumnRange { start: 0, end: 4 } },
                InputColumnRange { operand: 2, range: ColumnRange { start: 5, end: 8 } },
            ]
        );
    }

    #[test]
    fn route_descriptor_keeps_physical_route_and_conversion_identity() {
        let args = vec![matrix(2, 8)];
        let execution = gpu_execution_range(
            &NodeKind::Slice {
                rows: None,
                columns: Some(IndexRange {
                    start: IntExpr::constant(0),
                    end: IntExpr::constant(8),
                }),
            },
            &args,
            8,
            ColumnRange { start: 2, end: 6 },
            4,
        )
        .unwrap();
        let local = execution.route_descriptor(Some(0), Some(0), true, false, false, false);
        assert_eq!(local.route, GpuTransferRoute::Resident);
        assert!(local.validate());
        let peer = execution.route_descriptor(Some(0), Some(1), false, true, true, false);
        assert_eq!(peer.route, GpuTransferRoute::Peer);
        assert_ne!(local, peer);
        let staged = execution
            .route_descriptor(Some(0), Some(1), false, false, true, false)
            .with_staging_bytes(128, 128, 64);
        assert_eq!(staged.route, GpuTransferRoute::HostStaging);
        assert!(staged.validate());
        assert_eq!(staged.source_staging_bytes, 128);
        assert_eq!(staged.host_staging_bytes, 128);
        assert_eq!(staged.pinned_host_staging_bytes, 64);
    }

    #[test]
    fn shared_route_resolver_matches_fixed_route_classes() {
        let base = GpuRouteResolutionInput {
            source_device: Some(0),
            destination_device: Some(1),
            source_range: ColumnRange { start: 2, end: 6 },
            destination_range: ColumnRange { start: 0, end: 4 },
            source_is_resident: false,
            peer_available: true,
            source_compact: true,
            destination_compact: false,
            fragment: GpuFragmentClass::CompactFragment,
            source_staging_bytes: 0,
            host_staging_bytes: 0,
            pinned_host_staging_bytes: 0,
        };
        let peer = resolve_gpu_route(base);
        assert_eq!(peer.route, GpuTransferRoute::Peer);
        assert_eq!(peer.source_range, base.source_range);
        assert!(peer.source_compact);

        let staged = resolve_gpu_route(GpuRouteResolutionInput {
            peer_available: false,
            source_staging_bytes: 64,
            host_staging_bytes: 64,
            pinned_host_staging_bytes: 32,
            ..base
        });
        assert_eq!(staged.route, GpuTransferRoute::HostStaging);
        assert!(staged.validate());
    }
}
