//! Immutable, bounded representation of the compiled GPU dependency program.
//!
//! This module deliberately contains no executor or CUDA calls.  Preparation
//! lowers work to these templates; the runtime only instantiates a frame,
//! patches its registered arguments, and advances the two signals on each
//! operation.  Keeping the graph representation here makes it possible to
//! validate ownership and resource ordering without accidentally adding a
//! second allocator or a per-iteration command list.

use crate::{
    backend::{GpuEffectiveInputs, PlannedNodeBatchRequest},
    gpu_column_policy::EffectiveGpuOperation,
    gpu_execution_plan::LayoutId,
    gpu_schedule::GpuColumnJob,
};
use mxx_ir_core::{
    ParamEnv,
    artifact::ManifestArtifact,
    expr::IntExpr,
    graph::FrozenGraphScopeId,
    node::{
        ConcatAxis, ConstantMatrix, HashTagComponent, IntBinaryOp, IntCompareOp, LoopInputMode,
        NodeKind,
    },
    types::{ConcreteMatrixType, ConcreteWireType, NodeId, WireRef},
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    num::NonZeroUsize,
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct OpId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct RegionId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ValueSlot(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct FrameId(pub u32);

/// A frame token is never valid after its slot has been reused.  In
/// particular, a late CUDA/worker notification cannot mutate a new instance
/// merely because both instances use the same frame slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct FrameGeneration {
    pub frame: FrameId,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum Signal {
    /// Launch/copy enqueue, completion event registration, and native lifetime
    /// protection all succeeded.
    Submitted,
    /// Device/worker completion succeeded.  Status-bearing operations only
    /// emit this signal after their success gate passes.
    Done,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct Edge {
    pub source: OpId,
    pub signal: Signal,
    pub destination: OpId,
}

/// Which production owns a member-level artifact key.  This is kept in the
/// compiled program so dynamic keys do not accidentally become keys in the
/// current run's production.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProductionBinding {
    Existing(mxx_ir_core::artifact::ProductionId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum MemberBinding {
    Scalar,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedArtifactKey {
    pub production: ProductionBinding,
    pub name: Arc<str>,
    pub member: MemberBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedArtifactLoad {
    pub key: PreparedArtifactKey,
    pub descriptor: ManifestArtifact,
    pub destination: ValueSlot,
    pub staged: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeIntegerEncoding {
    SignedWords(usize),
    UnsignedWord,
    SignedWord,
}

/// The complete type vocabulary which may occupy a resident control slot.
///
/// A resident slot is not an untyped integer scratch location.  Keeping the
/// concrete wire type here is important for two reasons: a replay can reject
/// a shape-compatible value with the wrong semantic type, and family
/// operations can be checked without inventing an integer encoding for an
/// arbitrary value.  The `IndexedFamily` case is recursive, so nested
/// families remain typed as well; there is deliberately no catch-all case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentSlotType {
    Integer { wire_type: ConcreteWireType, encoding: NativeIntegerEncoding },
    Boolean { wire_type: ConcreteWireType },
    Real { wire_type: ConcreteWireType },
    Bytes { wire_type: ConcreteWireType },
    TypedBlob { wire_type: ConcreteWireType },
    Matrix { wire_type: ConcreteWireType },
    Trapdoor { wire_type: ConcreteWireType },
    SmallMatrix { wire_type: ConcreteWireType },
    Preimage { wire_type: ConcreteWireType },
    IndexedFamily { wire_type: ConcreteWireType, element: Box<ResidentSlotType>, count: usize },
}

impl ResidentSlotType {
    pub(crate) fn wire_type(&self) -> &ConcreteWireType {
        match self {
            Self::Integer { wire_type, .. } |
            Self::Boolean { wire_type } |
            Self::Real { wire_type } |
            Self::Bytes { wire_type } |
            Self::TypedBlob { wire_type } |
            Self::Matrix { wire_type } |
            Self::Trapdoor { wire_type } |
            Self::SmallMatrix { wire_type } |
            Self::Preimage { wire_type } => wire_type,
            Self::IndexedFamily { wire_type, .. } => wire_type,
        }
    }

    pub(crate) fn integer_encoding(&self) -> Option<NativeIntegerEncoding> {
        match self {
            Self::Integer { encoding, .. } => Some(*encoding),
            Self::IndexedFamily { element, .. } => element.integer_encoding(),
            _ => None,
        }
    }

    pub(crate) fn slot_count(&self) -> Option<usize> {
        match self {
            Self::IndexedFamily { count, .. } => Some(*count),
            _ => None,
        }
    }
}

/// A value slot together with the exact DSL type frozen for that slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentTypedSlot {
    pub slot: ValueSlot,
    pub ty: ResidentSlotType,
}

/// Stable identity for a resident slot.  `ValueSlot` is globally allocated by
/// the arena, but retaining the lexical scope here is intentional: nested
/// loop bodies may reuse a wire/node number and a physical owner lookup must
/// never resolve one body's lane to another body's owner.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ResidentSlotIdentity {
    pub scope: FrozenGraphScopeId,
    pub slot: ValueSlot,
}

/// A physical component selection for one resident lane.  This is deliberately
/// separate from `ResidentSlotType`: a widened lane buffer is storage, not a
/// wider DSL matrix or family element.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum ResidentLaneSelection {
    /// One owner is visible to every active lane (ordinary scalar/broadcast).
    Broadcast,
    /// Lane `i` starts at `base + i * lane_stride` in this component.
    Strided { lane_stride: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ResidentPhysicalComponentLayout {
    pub component: NativeValueComponent,
    /// Distance between adjacent lanes in component elements/records.  It is
    /// not a matrix-column count and must not be used to rewrite a DSL type.
    pub lane_stride: usize,
    pub selection: ResidentLaneSelection,
}

/// Physical storage contract for one semantic slot in one lexical scope.
/// `batch_axes` describes enclosing resident batches from outermost to
/// innermost; `wave_capacity` is the allocation/capture capacity and
/// `active_lanes` is the phase's currently valid prefix (full or tail).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentPhysicalSlotLayout {
    pub identity: ResidentSlotIdentity,
    pub batch_axes: Box<[usize]>,
    pub wave_capacity: NonZeroUsize,
    pub active_lanes: usize,
    pub components: Box<[ResidentPhysicalComponentLayout]>,
}

/// Stable resident-only identity for one native pointer binding.  `job` is
/// retained because one source interval may be emitted by multiple waves;
/// omitting it would collapse otherwise distinct graph arguments.  This key
/// never changes the ordinary `RegionBinding`/`BindingSource` contract.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct PhysicalBindingKey {
    pub scope: FrozenGraphScopeId,
    pub slot: ValueSlot,
    pub lane: usize,
    pub device: i32,
    pub shard: u32,
    pub component: NativeValueComponent,
    pub job: u32,
    pub address_addend: u64,
    pub selection: ResidentPhysicalBindingSelection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum ResidentPhysicalBindingSelection {
    /// A broadcast edge has one physical owner address shared by every lane.
    /// The key still carries the lane so kernel argument mapping can retain
    /// lane identity; replay resolves all of those keys to the same owner.
    SharedBroadcast,
    /// An indexed-family descriptor element. This is an absolute semantic
    /// member and must not be offset by a loop wave base.
    AbsoluteFamilyElement(usize),
    /// A lane in the current full/tail replay wave. The runtime combines this
    /// with the scoped wave-base binding when selecting a dynamic instance.
    WaveRelativeLane(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentPhysicalBinding {
    pub key: PhysicalBindingKey,
    pub index: u32,
    pub access: BindingAccess,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentPhaseBindingSchema {
    pub phase: ResidentPhaseId,
    pub bindings: Box<[ResidentPhysicalBinding]>,
}

impl ResidentPhysicalSlotLayout {
    pub(crate) fn for_slot(
        scope: FrozenGraphScopeId,
        value: &ResidentTypedSlot,
        wave_capacity: NonZeroUsize,
        active_lanes: usize,
        batch_axes: impl IntoIterator<Item = usize>,
        components: impl IntoIterator<Item = NativeValueComponent>,
    ) -> Self {
        let active_lanes = active_lanes.min(wave_capacity.get());
        let components = components
            .into_iter()
            .map(|component| ResidentPhysicalComponentLayout {
                component,
                lane_stride: 1,
                selection: ResidentLaneSelection::Strided { lane_stride: 1 },
            })
            .collect::<Vec<_>>();
        Self {
            identity: ResidentSlotIdentity { scope, slot: value.slot },
            batch_axes: batch_axes.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            wave_capacity,
            active_lanes,
            components: components.into_boxed_slice(),
        }
    }

    pub(crate) fn broadcast(
        scope: FrozenGraphScopeId,
        value: &ResidentTypedSlot,
        wave_capacity: NonZeroUsize,
        active_lanes: usize,
        batch_axes: impl IntoIterator<Item = usize>,
        components: impl IntoIterator<Item = NativeValueComponent>,
    ) -> Self {
        let mut layout =
            Self::for_slot(scope, value, wave_capacity, active_lanes, batch_axes, components);
        for component in &mut layout.components {
            component.selection = ResidentLaneSelection::Broadcast;
            component.lane_stride = 0;
        }
        layout
    }
}

impl ResidentTypedSlot {
    pub(crate) fn new(slot: ValueSlot, ty: ResidentSlotType) -> Self {
        Self { slot, ty }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ResidentProgramId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ResidentRegionId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ResidentPhaseId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ResidentInstructionId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ResidentBindingId(pub u32);

/// The resident scalar/control vocabulary used by a compiled GPU frame.
///
/// Expressions are retained in the schema.  In particular, a loop index is
/// never replaced with the first exemplar's value while compiling a frame;
/// the device evaluator retains exact division and floor-division semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentControlOperation {
    ConstantInt {
        value: num_bigint::BigInt,
    },
    EvaluateInt {
        expression: IntExpr,
    },
    ConstantBool {
        value: bool,
    },
    IntBinary {
        operation: IntBinaryOp,
    },
    IntCompare {
        operation: IntCompareOp,
    },
    BitExtract {
        bit: IntExpr,
    },
    BoolToInt,
    FamilyPack {
        count: IntExpr,
        element: Box<ResidentSlotType>,
        output: Box<ResidentSlotType>,
    },
    FamilyGetStatic {
        index: IntExpr,
        family: Box<ResidentSlotType>,
        output: Box<ResidentSlotType>,
    },
    FamilyGetDynamic {
        family: Box<ResidentSlotType>,
        index: Box<ResidentSlotType>,
        output: Box<ResidentSlotType>,
    },
    Select {
        count: IntExpr,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentControlOutput {
    Direct { value: ResidentTypedSlot },
    IndexedFamily { value: ResidentTypedSlot },
}

impl ResidentControlOutput {
    pub(crate) fn value(&self) -> &ResidentTypedSlot {
        match self {
            Self::Direct { value } | Self::IndexedFamily { value } => value,
        }
    }

    pub(crate) fn slot(&self) -> ValueSlot {
        self.value().slot
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentControlWireSlot {
    pub scope: FrozenGraphScopeId,
    pub wire: WireRef,
    pub value: ResidentTypedSlot,
}

/// A root-region input that must be resolved when a resident program is
/// instantiated.  The scope is part of the identity: resolving this by the
/// bare `WireRef` is unsound because child scopes reuse node/port numbers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentControlExternalInput {
    pub scope: FrozenGraphScopeId,
    pub wire: WireRef,
    pub value: ResidentTypedSlot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentControlOwnerLayout {
    pub value: ResidentTypedSlot,
    pub layout: Option<LayoutId>,
    pub physical: ResidentPhysicalSlotLayout,
    pub components: Box<[NativeValueComponent]>,
    pub owner: GpuCaptureOutputOwnerSpec,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentControlBinding {
    pub id: ResidentBindingId,
    pub source: ResidentBindingSource,
    pub access: BindingAccess,
}

/// A resident binding refers to a slot, never to a host owner or a guessed
/// shard. Native owner resolution happens at the explicit owner-layout
/// boundary when the frame is instantiated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentBindingSource {
    Slot(ValueSlot),
    Expression(IntExpr),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentLoopImport {
    pub parent: ResidentTypedSlot,
    pub child: ResidentTypedSlot,
    pub mode: LoopInputMode,
}

/// The immutable selection performed while importing a value into a loop
/// body.  Keeping the loop index and offset in the schema is important for
/// indexed families: a replay must select the current lane's element rather
/// than silently binding element zero.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentImportSelection {
    Broadcast,
    Zip { loop_index: ResidentTypedSlot, offset: usize },
    FamilyElement { loop_index: ResidentTypedSlot, offset: usize },
}

/// A typed parent-to-child import.  This is deliberately independent of a
/// native leaf, so external inputs remain bindable even when no native node
/// happens to consume the imported value directly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentTypedImportBinding {
    pub region: ResidentRegionId,
    pub parent: ResidentTypedSlot,
    pub child: ResidentTypedSlot,
    pub selection: ResidentImportSelection,
    pub parent_physical: ResidentPhysicalSlotLayout,
    pub child_physical: ResidentPhysicalSlotLayout,
    pub parent_components: Box<[NativeValueComponent]>,
    pub child_components: Box<[NativeValueComponent]>,
}

/// Per-loop indexing state used by the resident replay adapter.  All three
/// values are scoped to the child region, which keeps nested loops independent
/// even though their body NodeIds may be reused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentLoopWave {
    pub index: ResidentTypedSlot,
    pub wave_base: ResidentTypedSlot,
    pub active_lane: ResidentTypedSlot,
    pub width: NonZeroUsize,
    pub index_expression: IntExpr,
    /// Enclosing batch capacities, retained independently from `width` so a
    /// nested body can be dispatched without deriving axes from matrix shape.
    pub batch_axes: Box<[usize]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentLoopExport {
    pub child: ResidentControlOutput,
    pub parent: ResidentControlOutput,
    pub child_physical: Option<ResidentPhysicalSlotLayout>,
    pub parent_physical: Option<ResidentPhysicalSlotLayout>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentCarriedBinding {
    pub index: usize,
    pub initial: ResidentTypedSlot,
    pub body: ResidentTypedSlot,
    pub output: ResidentTypedSlot,
}

/// An instruction in the flat resident-control arena. Child scopes refer to
/// region/phase/instruction IDs rather than owning nested replay trees.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentControlInstructionKind {
    Scalar(ResidentControlOperation),
    Native {
        payload: Box<ResidentNativeInstruction>,
    },
    ParallelLoop {
        count: IntExpr,
        minimum_count: usize,
        index_slot: ResidentTypedSlot,
        child: ResidentRegionId,
        imports: Box<[ResidentLoopImport]>,
        exports: Box<[ResidentLoopExport]>,
    },
    SequentialLoop {
        count: IntExpr,
        index_slot: ResidentTypedSlot,
        child: Option<ResidentRegionId>,
        carried: Box<[ResidentCarriedBinding]>,
        imports: Box<[ResidentLoopImport]>,
        exports: Box<[ResidentLoopExport]>,
    },
    SubgraphCall {
        definition: Arc<str>,
        child: Option<ResidentRegionId>,
        imports: Box<[ResidentLoopImport]>,
        exports: Box<[ResidentLoopExport]>,
    },
}

impl ResidentControlInstructionKind {
    pub(crate) fn native_physical_layouts_mut(&mut self) -> Vec<&mut ResidentPhysicalSlotLayout> {
        match self {
            Self::Native { payload } => payload
                .physical_inputs
                .iter_mut()
                .chain(payload.physical_outputs.iter_mut())
                .collect(),
            Self::Scalar(_) |
            Self::ParallelLoop { .. } |
            Self::SequentialLoop { .. } |
            Self::SubgraphCall { .. } => Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentNativeMatrixProduct {
    pub coefficient: num_bigint::BigInt,
    pub left: WireRef,
    pub right: WireRef,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentNativeUnary {
    RingAutomorphism {
        index: usize,
    },
    ModulusSwitch {
        destination: ConcreteMatrixType,
    },
    ModulusReduce {
        destination: ConcreteMatrixType,
    },
    CenteredRebase {
        destination: ConcreteMatrixType,
    },
    CenteredRoundDivide {
        divisor: num_bigint::BigInt,
    },
    BlockModSwitch {
        destination: ConcreteMatrixType,
        source_moduli: Box<[u64]>,
        plaintext_modulus: num_bigint::BigInt,
    },
    RnsModUp {
        destination: ConcreteMatrixType,
        source_moduli: Box<[u64]>,
        digit_size: usize,
        normalize: bool,
    },
    RnsModDown {
        destination: ConcreteMatrixType,
        source_moduli: Box<[u64]>,
        plaintext_modulus: u64,
    },
    Transpose,
    Slice {
        rows: Option<(usize, usize)>,
        columns: Option<(usize, usize)>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentNativeOrdinary {
    Constant {
        ty: ConcreteMatrixType,
        value: ConstantMatrix,
        env: ParamEnv,
        single_device: bool,
    },
    LiftIntegerToConstantPolynomial {
        ty: ConcreteMatrixType,
        coefficient: WireRef,
    },
    PolynomialFromValues {
        ty: ConcreteMatrixType,
        values: WireRef,
        evaluation: bool,
    },
    PolynomialValues {
        value: WireRef,
        evaluation: bool,
    },
    ExtractCoefficient {
        value: WireRef,
        position: usize,
    },
    ThresholdDecode {
        value: WireRef,
        plaintext_modulus: num_bigint::BigInt,
        length: usize,
        output_bool: bool,
    },
    PackPolynomialCoefficients {
        ty: ConcreteMatrixType,
        bits: WireRef,
        coefficient_bits: usize,
    },
    MatrixBinary {
        operation: mxx_ir_core::node::MatrixBinaryOp,
        left: WireRef,
        right: WireRef,
    },
    MatrixMulSmallRhs {
        left: WireRef,
        right: WireRef,
    },
    MatrixMulAccumulate {
        products: Box<[ResidentNativeMatrixProduct]>,
        bias: Option<WireRef>,
    },
    Negate {
        value: WireRef,
    },
    Scale {
        value: WireRef,
        scalar: num_bigint::BigInt,
    },
    Unary {
        operation: ResidentNativeUnary,
        value: WireRef,
    },
    Tensor {
        left: WireRef,
        right: WireRef,
    },
    Concat {
        inputs: Box<[WireRef]>,
        axis: ConcatAxis,
    },
    CrtRecompose {
        levels: Box<[WireRef]>,
        plaintext_moduli: Box<[num_bigint::BigInt]>,
        reconstruction_coefficients: Box<[num_bigint::BigInt]>,
        destination: ConcreteMatrixType,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentNativeFused {
    RowSum { source: WireRef, right: Option<WireRef>, rows: Vec<Vec<usize>> },
    TensorRowSums { source: WireRef, right: WireRef, rows: Vec<Vec<Vec<usize>>> },
    Decompose { blocks: Box<[Vec<WireRef>]>, small: bool, digits: usize },
    SmallProduct { blocks: Box<[Vec<WireRef>]>, rhs: WireRef },
    Add { blocks: Box<[Vec<WireRef>]>, right: WireRef },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentNativeGeneration {
    Uniform {
        ty: ConcreteMatrixType,
        minimum: num_bigint::BigInt,
        maximum: num_bigint::BigInt,
    },
    Gaussian {
        ty: ConcreteMatrixType,
        sigma_bits: u64,
        max_coefficient_bound: num_bigint::BigInt,
    },
    Hash {
        ty: ConcreteMatrixType,
        key: WireRef,
        tag_prefix: Box<[u8]>,
        tag_components: Box<[HashTagComponent]>,
    },
    HashDecomposed {
        ty: ConcreteMatrixType,
        key: WireRef,
        tag_prefix: Box<[u8]>,
        tag_components: Box<[HashTagComponent]>,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
        small: bool,
    },
}

/// Owner-free, fully evaluated request parameters for a resident native leaf.
/// Wire references are resolved to the replay frame's typed owners by the
/// fleet; no `RuntimeValue`, device pointer, or host materialization is kept.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResidentNativePrepared {
    Alias {
        input: WireRef,
    },
    Ordinary(ResidentNativeOrdinary),
    CompactCenteredRebase {
        input: WireRef,
        destination: ConcreteMatrixType,
    },
    Fused(ResidentNativeFused),
    Generation(ResidentNativeGeneration),
    Trapdoor {
        ty: ConcreteMatrixType,
        sigma_bits: u64,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
    },
    Decomposition {
        input: WireRef,
        small: bool,
        digits: usize,
    },
    Preimage {
        public: WireRef,
        trapdoor: WireRef,
        target: WireRef,
        matrix_type: ConcreteMatrixType,
        sigma_bits: u64,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
        max_coefficient_bound: num_bigint::BigInt,
        randomness_seed: [u8; 32],
    },
}

/// Immutable source-owner schema for one native leaf input.  Parent/child
/// ownership and loop selection live in `ResidentTypedImportBinding`; this
/// leaf table only records which typed wire the operation consumes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentNativeSourceBinding {
    pub wire: WireRef,
    pub child: ResidentTypedSlot,
    pub physical: ResidentPhysicalSlotLayout,
}

/// Immutable native work embedded in a resident loop/subgraph.  It mirrors
/// the value-only parts of `CaptureStep` so the fleet can submit the same
/// fixed request path as an ordinary capture step.  It contains no runtime
/// owner or device pointer; owners are resolved from typed slots at replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentNativeInstruction {
    pub scope: FrozenGraphScopeId,
    pub site: crate::gpu_execution_plan::GpuExecutionSiteKey,
    pub node: NodeId,
    pub kind: NodeKind,
    pub original_arguments: Box<[WireRef]>,
    pub effective_inputs: GpuEffectiveInputs,
    pub row_sum: Option<Vec<Vec<usize>>>,
    pub row_blocks: Vec<Vec<WireRef>>,
    pub metadata: PlannedNodeBatchRequest,
    pub prepared: ResidentNativePrepared,
    pub source_bindings: Box<[ResidentNativeSourceBinding]>,
    pub operation: EffectiveGpuOperation,
    pub physical_device: i32,
    pub jobs: Box<[GpuColumnJob]>,
    pub outputs: Box<[GpuCaptureOutputLayout]>,
    pub physical_inputs: Box<[ResidentPhysicalSlotLayout]>,
    pub physical_outputs: Box<[ResidentPhysicalSlotLayout]>,
    pub physical_bindings: Box<[ResidentPhysicalBinding]>,
    pub tail_physical_bindings: Box<[ResidentPhysicalBinding]>,
    pub dispatch_geometry: Option<ResidentNativeDispatchGeometry>,
    pub bindings: Box<[RegionBinding]>,
}

/// Frozen phase routing for a native resident leaf. The fleet must select the
/// tail executable by this descriptor; it must not infer a lane window from a
/// widened matrix column count or from the first job in `jobs`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentNativeDispatchGeometry {
    pub full_phase: ResidentPhaseId,
    pub tail_phase: Option<ResidentPhaseId>,
    pub wave_capacity: NonZeroUsize,
    pub full_active_lanes: usize,
    pub tail_active_lanes: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledResidentControlInstruction {
    pub id: ResidentInstructionId,
    pub node: NodeId,
    pub kind: ResidentControlInstructionKind,
    pub inputs: Box<[ResidentTypedSlot]>,
    pub outputs: Box<[ResidentControlOutput]>,
    pub bindings: Box<[ResidentBindingId]>,
    pub owner_layouts: Box<[ResidentControlOwnerLayout]>,
    pub status: Option<ResidentTypedSlot>,
    /// Phase identity is explicit because the same operation template may be
    /// replayed once for a full wave and once for a tail wave.
    pub phase: Option<ResidentPhaseId>,
    pub tail_phase: Option<ResidentPhaseId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledResidentControlPhase {
    pub id: ResidentPhaseId,
    pub region: ResidentRegionId,
    pub width: NonZeroUsize,
    pub instructions: Box<[ResidentInstructionId]>,
    pub ranges: Box<[ResidentControlRange]>,
    pub geometry: ResidentPhaseGeometry,
    pub wave_base: Option<ResidentTypedSlot>,
    pub active_lane: Option<ResidentTypedSlot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidentPhaseKind {
    Full,
    Tail,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentPhaseGeometry {
    pub kind: ResidentPhaseKind,
    pub wave_capacity: NonZeroUsize,
    pub active_lanes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentControlRange {
    /// A compact leaf run in a phase. Structural instructions split ranges;
    /// this is a template descriptor and never expands by instance count.
    pub instructions: Box<[ResidentInstructionId]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledResidentControlRegion {
    pub id: ResidentRegionId,
    pub scope: FrozenGraphScopeId,
    pub physical_device: i32,
    pub instance_count: u64,
    pub wave_width: NonZeroUsize,
    pub phases: Box<[ResidentPhaseId]>,
    pub tail: Option<ResidentPhaseId>,
    pub inputs: Box<[ResidentTypedSlot]>,
    pub outputs: Box<[ResidentControlOutput]>,
    pub imports: Box<[ResidentLoopImport]>,
    pub exports: Box<[ResidentLoopExport]>,
    pub wave: Option<ResidentLoopWave>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledResidentControlProgram {
    pub id: ResidentProgramId,
    pub root: ResidentRegionId,
    pub regions: Box<[CompiledResidentControlRegion]>,
    pub phases: Box<[CompiledResidentControlPhase]>,
    pub instructions: Box<[CompiledResidentControlInstruction]>,
    pub wire_slots: Box<[ResidentControlWireSlot]>,
    pub external_inputs: Box<[ResidentControlExternalInput]>,
    pub typed_imports: Box<[ResidentTypedImportBinding]>,
    pub root_outputs: Box<[ResidentTypedSlot]>,
    pub bindings: Box<[ResidentControlBinding]>,
    pub slot_layouts: Box<[ResidentPhysicalSlotLayout]>,
    pub phase_bindings: Box<[ResidentPhaseBindingSchema]>,
    pub slot_count: u32,
}

impl CompiledResidentControlProgram {
    pub(crate) fn validate_shape(&self) -> Result<(), CompiledScheduleError> {
        let regions = self.regions.iter().map(|region| region.id).collect::<BTreeSet<_>>();
        let phases = self.phases.iter().map(|phase| phase.id).collect::<BTreeSet<_>>();
        let instructions =
            self.instructions.iter().map(|instruction| instruction.id).collect::<BTreeSet<_>>();
        if regions.len() != self.regions.len() ||
            phases.len() != self.phases.len() ||
            instructions.len() != self.instructions.len()
        {
            return Err(CompiledScheduleError::DuplicateResidentArenaId);
        }
        if !regions.contains(&self.root) {
            return Err(CompiledScheduleError::UnknownResidentRegion(self.root));
        }
        let binding_ids = self.bindings.iter().map(|binding| binding.id).collect::<BTreeSet<_>>();
        if binding_ids.len() != self.bindings.len() {
            return Err(CompiledScheduleError::DuplicateResidentBinding);
        }
        if self.wire_slots.iter().any(|wire| wire.value.slot.0 >= self.slot_count) {
            return Err(CompiledScheduleError::ValueSlotOutOfRange);
        }
        if self.external_inputs.iter().any(|input| input.value.slot.0 >= self.slot_count) {
            return Err(CompiledScheduleError::ValueSlotOutOfRange);
        }
        if self.typed_imports.iter().any(|import| {
            import.region.0 as usize >= self.regions.len() ||
                import.parent.slot.0 >= self.slot_count ||
                import.child.slot.0 >= self.slot_count ||
                match &import.selection {
                    ResidentImportSelection::Broadcast => false,
                    ResidentImportSelection::Zip { loop_index, .. } |
                    ResidentImportSelection::FamilyElement { loop_index, .. } => {
                        loop_index.slot.0 >= self.slot_count
                    }
                }
        }) {
            return Err(CompiledScheduleError::ValueSlotOutOfRange);
        }
        if self.root_outputs.iter().any(|output| output.slot.0 >= self.slot_count) {
            return Err(CompiledScheduleError::ValueSlotOutOfRange);
        }
        if self.slot_layouts.iter().any(|layout| {
            layout.identity.slot.0 >= self.slot_count ||
                layout.active_lanes > layout.wave_capacity.get() ||
                layout.components.iter().any(|component| {
                    matches!(component.selection, ResidentLaneSelection::Broadcast) &&
                        component.lane_stride != 0
                })
        }) {
            return Err(CompiledScheduleError::ValueSlotOutOfRange);
        }
        let phase_ids = phases.clone();
        for schema in &self.phase_bindings {
            if !phase_ids.contains(&schema.phase) {
                return Err(CompiledScheduleError::UnknownResidentPhase(schema.phase));
            }
            let mut keys = BTreeSet::new();
            for binding in &schema.bindings {
                if binding.key.slot.0 >= self.slot_count || !keys.insert(binding.key.clone()) {
                    return Err(CompiledScheduleError::ValueSlotOutOfRange);
                }
            }
        }
        for input in &self.external_inputs {
            if !self.wire_slots.iter().any(|wire| {
                wire.scope == input.scope &&
                    wire.wire == input.wire &&
                    wire.value.slot == input.value.slot &&
                    wire.value.ty == input.value.ty
            }) {
                return Err(CompiledScheduleError::UnknownResidentWireSlot);
            }
        }
        for region in &self.regions {
            if region.inputs.iter().any(|slot| slot.slot.0 >= self.slot_count) ||
                region.outputs.iter().any(|output| match output {
                    ResidentControlOutput::Direct { value } |
                    ResidentControlOutput::IndexedFamily { value } => {
                        value.slot.0 >= self.slot_count
                    }
                })
            {
                return Err(CompiledScheduleError::ValueSlotOutOfRange);
            }
            let resident_slot_out_of_range =
                |slot: &ResidentTypedSlot| slot.slot.0 >= self.slot_count;
            if region.imports.iter().any(|import| {
                resident_slot_out_of_range(&import.parent) ||
                    resident_slot_out_of_range(&import.child)
            }) || region.exports.iter().any(|export| {
                resident_slot_out_of_range(export.child.value()) ||
                    resident_slot_out_of_range(export.parent.value())
            }) {
                return Err(CompiledScheduleError::ValueSlotOutOfRange);
            }
            if let Some(wave) = &region.wave {
                if wave.index.slot.0 >= self.slot_count ||
                    wave.wave_base.slot.0 >= self.slot_count ||
                    wave.active_lane.slot.0 >= self.slot_count
                {
                    return Err(CompiledScheduleError::ValueSlotOutOfRange);
                }
            }
            if region.instance_count == 0 && (!region.phases.is_empty() || region.tail.is_some()) {
                return Err(CompiledScheduleError::EmptyControlRepeat);
            }
            if region.instance_count != 0 && region.phases.is_empty() && region.tail.is_none() {
                return Err(CompiledScheduleError::MissingControlTemplate);
            }
            if region.phases.len() > region.wave_width.get() {
                return Err(CompiledScheduleError::ControlWaveWidthExceeded);
            }
            for phase in region.phases.iter().chain(region.tail.iter()) {
                if !phases.contains(phase) {
                    return Err(CompiledScheduleError::UnknownResidentPhase(*phase));
                }
            }
            for phase_id in &region.phases {
                let phase = self
                    .phases
                    .iter()
                    .find(|phase| phase.id == *phase_id)
                    .ok_or(CompiledScheduleError::UnknownResidentPhase(*phase_id))?;
                if phase.region != region.id ||
                    phase.geometry.kind != ResidentPhaseKind::Full ||
                    phase.width != region.wave_width ||
                    phase.geometry.wave_capacity != region.wave_width ||
                    phase.geometry.active_lanes != phase.width.get() ||
                    phase.geometry.active_lanes != region.wave_width.get()
                {
                    return Err(CompiledScheduleError::ControlWaveWidthExceeded);
                }
            }
            if let Some(tail_id) = region.tail {
                let tail = self
                    .phases
                    .iter()
                    .find(|phase| phase.id == tail_id)
                    .ok_or(CompiledScheduleError::UnknownResidentPhase(tail_id))?;
                if tail.region != region.id ||
                    tail.geometry.kind != ResidentPhaseKind::Tail ||
                    tail.width >= region.wave_width ||
                    tail.geometry.wave_capacity != region.wave_width ||
                    tail.geometry.active_lanes != tail.width.get() ||
                    tail.geometry.active_lanes == 0 ||
                    tail.geometry.active_lanes >= region.wave_width.get()
                {
                    return Err(CompiledScheduleError::ControlWaveWidthExceeded);
                }
            }
        }
        for phase in &self.phases {
            let region = self
                .regions
                .iter()
                .find(|region| region.id == phase.region)
                .ok_or(CompiledScheduleError::UnknownResidentRegion(phase.region))?;
            if phase.width.get() == 0 {
                return Err(CompiledScheduleError::ControlWaveWidthExceeded);
            }
            if phase.geometry.wave_capacity != region.wave_width ||
                phase.geometry.active_lanes != phase.width.get() ||
                phase.geometry.active_lanes == 0 ||
                match phase.geometry.kind {
                    ResidentPhaseKind::Full => {
                        phase.geometry.active_lanes != phase.geometry.wave_capacity.get()
                    }
                    ResidentPhaseKind::Tail => {
                        phase.geometry.active_lanes >= phase.geometry.wave_capacity.get()
                    }
                }
            {
                return Err(CompiledScheduleError::ControlWaveWidthExceeded);
            }
            if phase.wave_base.as_ref().is_some_and(|slot| slot.slot.0 >= self.slot_count) ||
                phase.active_lane.as_ref().is_some_and(|slot| slot.slot.0 >= self.slot_count)
            {
                return Err(CompiledScheduleError::ValueSlotOutOfRange);
            }
            for instruction in &phase.instructions {
                if !instructions.contains(instruction) {
                    return Err(CompiledScheduleError::UnknownResidentInstruction(*instruction));
                }
                let compiled = self
                    .instructions
                    .iter()
                    .find(|candidate| candidate.id == *instruction)
                    .ok_or(CompiledScheduleError::UnknownResidentInstruction(*instruction))?;
                if compiled.phase != Some(phase.id) && compiled.tail_phase != Some(phase.id) {
                    return Err(CompiledScheduleError::UnknownResidentInstruction(*instruction));
                }
            }
            for range in &phase.ranges {
                if range.instructions.is_empty() ||
                    range.instructions.iter().any(|instruction| {
                        !phase.instructions.contains(instruction) ||
                            !instructions.contains(instruction)
                    })
                {
                    return Err(CompiledScheduleError::UnknownResidentInstruction(
                        range.instructions.first().copied().unwrap_or(ResidentInstructionId(0)),
                    ));
                }
            }
        }
        for instruction in &self.instructions {
            if instruction.inputs.iter().any(|slot| slot.slot.0 >= self.slot_count) ||
                instruction.outputs.iter().any(|output| match output {
                    ResidentControlOutput::Direct { value } |
                    ResidentControlOutput::IndexedFamily { value } => {
                        value.slot.0 >= self.slot_count
                    }
                }) ||
                instruction.status.as_ref().is_some_and(|slot| slot.slot.0 >= self.slot_count)
            {
                return Err(CompiledScheduleError::ValueSlotOutOfRange);
            }
            let resident_slot_out_of_range =
                |slot: &ResidentTypedSlot| slot.slot.0 >= self.slot_count;
            let child_edges_out_of_range = match &instruction.kind {
                ResidentControlInstructionKind::ParallelLoop {
                    index_slot,
                    imports,
                    exports,
                    ..
                } |
                ResidentControlInstructionKind::SequentialLoop {
                    index_slot,
                    imports,
                    exports,
                    ..
                } => {
                    resident_slot_out_of_range(index_slot) ||
                        imports.iter().any(|import| {
                            resident_slot_out_of_range(&import.parent) ||
                                resident_slot_out_of_range(&import.child)
                        }) ||
                        exports.iter().any(|export| {
                            resident_slot_out_of_range(export.child.value()) ||
                                resident_slot_out_of_range(export.parent.value())
                        })
                }
                ResidentControlInstructionKind::SubgraphCall { imports, exports, .. } => {
                    imports.iter().any(|import| {
                        resident_slot_out_of_range(&import.parent) ||
                            resident_slot_out_of_range(&import.child)
                    }) || exports.iter().any(|export| {
                        resident_slot_out_of_range(export.child.value()) ||
                            resident_slot_out_of_range(export.parent.value())
                    })
                }
                ResidentControlInstructionKind::Scalar(_) |
                ResidentControlInstructionKind::Native { .. } => false,
            };
            if child_edges_out_of_range {
                return Err(CompiledScheduleError::ValueSlotOutOfRange);
            }
            if let ResidentControlInstructionKind::SequentialLoop { carried, .. } =
                &instruction.kind
            {
                if carried.iter().any(|binding| {
                    resident_slot_out_of_range(&binding.initial) ||
                        resident_slot_out_of_range(&binding.body) ||
                        resident_slot_out_of_range(&binding.output)
                }) {
                    return Err(CompiledScheduleError::ValueSlotOutOfRange);
                }
            }
            if instruction.bindings.iter().any(|binding| !binding_ids.contains(binding)) {
                return Err(CompiledScheduleError::UnknownResidentBinding);
            }
            if instruction.phase.is_some_and(|phase| !phases.contains(&phase)) ||
                instruction.tail_phase.is_some_and(|phase| !phases.contains(&phase))
            {
                return Err(CompiledScheduleError::UnknownResidentPhase(
                    instruction.phase.or(instruction.tail_phase).unwrap(),
                ));
            }
            let child = match &instruction.kind {
                ResidentControlInstructionKind::ParallelLoop { child, .. } => Some(*child),
                ResidentControlInstructionKind::SequentialLoop { child: Some(child), .. } |
                ResidentControlInstructionKind::SubgraphCall { child: Some(child), .. } => {
                    Some(*child)
                }
                _ => None,
            };
            if let Some(child) = child {
                if !regions.contains(&child) {
                    return Err(CompiledScheduleError::UnknownResidentRegion(child));
                }
            }
        }
        Ok(())
    }
}

/// Native storage contract for an output which is not a matrix owner. Integer
/// families are kept in device memory between GPU nodes; this metadata is
/// frozen with the graph so replay never guesses a host representation or a
/// device from the current value table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IntegerValuesOutputSpec {
    pub count: usize,
    pub encoding: NativeIntegerEncoding,
    pub device: i32,
    pub mode: IntegerValuesOutputMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IntegerValuesOutputMode {
    Produced,
    StaticAlias { source: ValueSlot, offset: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuCaptureOutputOwnerSpec {
    Native,
    PublicTrapdoor,
    IntegerValues { spec: IntegerValuesOutputSpec, mode: IntegerValuesOutputMode },
}

/// Access performed by one native graph binding.  The access is part of the
/// frozen schema rather than inferred from the runtime value: a replay must
/// reject a value which is only shape-compatible but cannot satisfy the
/// producer/consumer contract captured during preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum BindingAccess {
    Input,
    Output,
    InOut,
}

/// A native pointer-bearing field of a run-local GPU value.  The enum is a
/// schema, not an address: addresses are resolved from the owner selected for
/// the current frame.  Keeping this list here makes the compiler, native
/// launch-site registration, and runtime resolver use one vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum NativeValueComponent {
    MatrixData,
    MatrixDescriptors,
    MatrixAuxiliary,
    CompactPayload,
    CompactDeviceStatus,
    CompactHostStatus,
    CompactHardCutoffStaging,
    TrapdoorR,
    TrapdoorE,
    TrapdoorCovarianceA,
    TrapdoorCovarianceB,
    TrapdoorCovarianceD,
    TrapdoorPublic,
    IntegerValues,
    /// A fixed-size Bytes32 value passed by value to a captured native kernel.
    /// It is a graph scalar binding, not a device pointer.
    Bytes32,
}

/// Source of a fixed native graph binding.  Values are frame slots and
/// constants are copied into the native binding vector at submission time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BindingSource {
    /// Resolve one component of one run-local owner. `shard` is in the
    /// frozen fleet/global column order and is never inferred from a vector
    /// position during replay.
    ValueComponent {
        slot: ValueSlot,
        shard: u32,
        component: NativeValueComponent,
        address_addend: u64,
    },
}

/// One immutable binding in a compiled region.  `index` is the native graph
/// binding index; it must remain stable across all replays of the region.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegionBinding {
    pub index: u32,
    pub source: BindingSource,
    pub access: BindingAccess,
}

/// Capture-time matrix layout facts used to decide whether a replay input can
/// retain its resident owner.  Device addresses are deliberately omitted:
/// replay patches those from the current owner.  The remaining fields are the
/// native binding layout that must stay identical for a zero-copy replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapturedMatrixInputLayout {
    pub slot: ValueSlot,
    pub physical_device: i32,
    pub rows: usize,
    pub columns: usize,
    pub level: usize,
    pub is_ntt: bool,
    pub components: Box<[CapturedMatrixBindingComponent]>,
    pub single_owner: bool,
    pub fragments: Box<[CapturedMatrixFragmentLayout]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapturedMatrixFragmentLayout {
    pub column_start: usize,
    pub columns: usize,
    pub components: Box<[CapturedMatrixBindingComponent]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CapturedMatrixBindingComponent {
    pub physical_device: i32,
    pub limb_count: usize,
    pub bytes_per_poly: usize,
    pub data_bytes: usize,
    pub ring_dimension: usize,
    pub device_descriptor_stride: usize,
    pub auxiliary_slots_per_poly: usize,
    pub auxiliary_slots_total: usize,
}

/// Native work retained by a compiled region.  The payload is deliberately
/// value-free; native executable handles are owned by the GPU backend adapter
/// and are associated with this schema by `RegionId`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeComponent {
    Fixed {
        operation_identity: [u8; 32],
        implementation_variant: Arc<str>,
        jobs: Box<[GpuColumnJob]>,
    },
    PreimageRetry {
        operation_identity: [u8; 32],
        max_attempts: u32,
        jobs: Box<[GpuColumnJob]>,
    },
}

/// Binding indices needed by the conditional preimage retry adapter.
///
/// The indices are copied from the frozen region schema; they are never
/// reconstructed from the order of values or native descriptors during a
/// replay.  One entry is retained per physical shard because a fleet output
/// may have more than one compact owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreimageRetryBindingSchema {
    pub max_attempts: u32,
    pub output: Box<[u32]>,
    pub fixed_scratch: Box<[u32]>,
    /// Pointer patched into the conditional gate's attempt argument.  This is
    /// retained separately from the scratch binding because graph replay must
    /// not infer the role from compact-owner component order.
    pub attempt: Box<[u32]>,
    /// Pointer patched into the conditional body control argument.
    pub control: Box<[u32]>,
    /// Pointer patched into the conditional gate's status argument.
    pub status: Box<[u32]>,
    pub host_status: Box<[u32]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledRegion {
    pub id: RegionId,
    pub physical_device: i32,
    pub operation_identity: [u8; 32],
    pub component: NativeComponent,
    pub bindings: Box<[RegionBinding]>,
    pub inputs: Box<[ValueSlot]>,
    pub outputs: Box<[ValueSlot]>,
}

impl CompiledRegion {
    pub(crate) fn validate_schema(&self) -> Result<(), CompiledScheduleError> {
        let mut indices = BTreeSet::new();
        for binding in &self.bindings {
            if !indices.insert(binding.index) {
                return Err(CompiledScheduleError::DuplicateBinding(binding.index));
            }
            let BindingSource::ValueComponent { slot, shard, .. } = binding.source;
            if slot.0 == u32::MAX || shard == u32::MAX {
                return Err(CompiledScheduleError::ValueSlotOutOfRange);
            }
        }
        if self.inputs.iter().any(|slot| self.outputs.contains(slot)) {
            return Err(CompiledScheduleError::RegionInputOutputOverlap);
        }
        Ok(())
    }

    /// Return the preimage-specific binding indices for a retry region.
    /// Non-preimage regions intentionally return `None`, so callers cannot
    /// accidentally apply a status gate to an ordinary compact operation.
    pub(crate) fn preimage_retry_bindings(&self) -> Option<PreimageRetryBindingSchema> {
        let max_attempts = match &self.component {
            NativeComponent::PreimageRetry { max_attempts, .. } => *max_attempts,
            _ => return None,
        };
        let mut output = Vec::new();
        let mut fixed_scratch = Vec::new();
        let mut attempt = Vec::new();
        let mut control = Vec::new();
        let mut status = Vec::new();
        let mut host_status = Vec::new();
        for binding in &self.bindings {
            let BindingSource::ValueComponent { component, .. } = &binding.source;
            match component {
                NativeValueComponent::CompactPayload => output.push(binding.index),
                NativeValueComponent::CompactHardCutoffStaging => {
                    fixed_scratch.push(binding.index);
                    attempt.push(binding.index);
                }
                NativeValueComponent::CompactDeviceStatus => {
                    control.push(binding.index);
                    status.push(binding.index);
                }
                NativeValueComponent::CompactHostStatus => host_status.push(binding.index),
                _ => {}
            }
        }
        Some(PreimageRetryBindingSchema {
            max_attempts,
            output: output.into_boxed_slice(),
            fixed_scratch: fixed_scratch.into_boxed_slice(),
            attempt: attempt.into_boxed_slice(),
            control: control.into_boxed_slice(),
            status: status.into_boxed_slice(),
            host_status: host_status.into_boxed_slice(),
        })
    }
}

/// Whether a captured output remains visible after the current scope.  Local
/// outputs may be released as soon as their last consumer submits; boundary
/// outputs are transferred to the caller/session and therefore survive the
/// frame's release join.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CaptureOutputClass {
    Boundary,
    Local,
}

/// Frozen output layout generated from the validated request and scope
/// liveness. It contains no owner, pointer, or payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuCaptureOutputLayout {
    pub wire: WireRef,
    pub slot: ValueSlot,
    /// Concrete owner type frozen with the request.  The scheduler uses this
    /// only to allocate a fresh frame owner; no exemplar or native address is
    /// retained in the compiled plan.
    pub wire_type: ConcreteWireType,
    pub class: CaptureOutputClass,
    pub layout: Option<LayoutId>,
    pub components: Box<[NativeValueComponent]>,
    pub owner: GpuCaptureOutputOwnerSpec,
    /// Matrix CRT representation required by the producer operation.
    pub matrix_is_ntt: bool,
}

/// Output schema for one captured request.  This is the only output metadata
/// retained by a compiled plan; exemplar owners are capture-scoped and must
/// not be stored here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuCaptureOutputSpec {
    pub outputs: Box<[GpuCaptureOutputLayout]>,
}

/// Concrete boundary metadata.  It describes an existing wire/owner
/// contract; it is not another runtime buffer representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BoundaryValueSpec {
    pub scope: FrozenGraphScopeId,
    pub wire: WireRef,
    pub wire_type: ConcreteWireType,
    pub layout: Option<LayoutId>,
    pub devices: Box<[i32]>,
    pub class: CaptureOutputClass,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TypedRealOperation {
    Constant(u64),
    IntToReal,
    Binary(mxx_ir_core::node::RealBinaryOp),
    Sqrt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CompiledOp {
    MatrixFamilyPack,
    MatrixFamilyGetStatic {
        index: usize,
    },
    TrapdoorPublic {
        source: ValueSlot,
        destination: ValueSlot,
    },
    Real {
        operation: TypedRealOperation,
        inputs: Box<[ValueSlot]>,
        destination: ValueSlot,
    },
    Gpu(RegionId),
    /// A resident scalar/family control operation. It is part of the frozen
    /// dependency frame and is never represented as a host producer/barrier.
    ResidentControl {
        program: ResidentProgramId,
        physical_device: i32,
    },
    Import(PreparedArtifactLoad),
    ReleaseOwners {
        values: Box<[ValueSlot]>,
        devices: Box<[i32]>,
    },
    /// A dependency join.  It is not a device-wide synchronization.
    Barrier,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OpTemplate {
    pub kind: CompiledOp,
    pub inputs: Box<[ValueSlot]>,
    pub outputs: Box<[ValueSlot]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FrameTemplate {
    pub ops: Box<[OpTemplate]>,
    pub edges: Box<[Edge]>,
    /// One counter per operation.  Every edge kind consumes this same counter.
    pub initial_dependencies: Box<[u32]>,
    pub successors: Box<[Box<[(Signal, OpId)]>]>,
    pub value_count: usize,
}

impl FrameTemplate {
    pub(crate) fn validate(&self) -> Result<(), CompiledScheduleError> {
        if self.initial_dependencies.len() != self.ops.len() ||
            self.successors.len() != self.ops.len()
        {
            return Err(CompiledScheduleError::MetadataLength);
        }
        for op in &self.ops {
            if op
                .inputs
                .iter()
                .chain(op.outputs.iter())
                .any(|slot| slot.0 as usize >= self.value_count)
            {
                return Err(CompiledScheduleError::ValueSlotOutOfRange);
            }
        }
        let mut seen = BTreeSet::new();
        let mut incoming = vec![0u32; self.ops.len()];
        for edge in &self.edges {
            let source = edge.source.0 as usize;
            let destination = edge.destination.0 as usize;
            if source >= self.ops.len() || destination >= self.ops.len() || source == destination {
                return Err(CompiledScheduleError::InvalidEdge(*edge));
            }
            if !seen.insert(*edge) {
                return Err(CompiledScheduleError::DuplicateEdge(*edge));
            }
            incoming[destination] = incoming[destination]
                .checked_add(1)
                .ok_or(CompiledScheduleError::DependencyOverflow)?;
        }
        if incoming.as_slice() != self.initial_dependencies.as_ref() {
            return Err(CompiledScheduleError::DependencyMismatch);
        }
        let mut expected_successors = vec![Vec::new(); self.ops.len()];
        for edge in &self.edges {
            expected_successors[edge.source.0 as usize].push((edge.signal, edge.destination));
        }
        for successors in &mut expected_successors {
            successors.sort_unstable();
        }
        if expected_successors
            .iter()
            .zip(self.successors.iter())
            .any(|(expected, actual)| expected.as_slice() != actual.as_ref())
        {
            return Err(CompiledScheduleError::SuccessorMismatch);
        }
        validate_acyclic(&self.edges, self.ops.len())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CompiledBlock {
    Once(Arc<FrameTemplate>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledProtocol {
    pub blocks: Box<[CompiledBlock]>,
    pub regions: Vec<CompiledRegion>,
    pub resident_control_programs: Box<[CompiledResidentControlProgram]>,
    pub boundary_values: Box<[BoundaryValueSpec]>,
    pub output_specs: Box<[GpuCaptureOutputSpec]>,
    pub wire_slots: BTreeMap<WireRef, ValueSlot>,
    pub matrix_input_layouts: BTreeMap<RegionId, Box<[CapturedMatrixInputLayout]>>,
}

impl CompiledProtocol {
    pub(crate) fn validate(&self) -> Result<(), CompiledScheduleError> {
        for region in &self.regions {
            region.validate_schema()?;
        }
        let region_ids = self.regions.iter().map(|region| region.id).collect::<BTreeSet<_>>();
        let resident_program_ids = self
            .resident_control_programs
            .iter()
            .map(|program| program.id)
            .collect::<BTreeSet<_>>();
        if resident_program_ids.len() != self.resident_control_programs.len() {
            return Err(CompiledScheduleError::DuplicateResidentProgram);
        }
        for program in &self.resident_control_programs {
            program.validate_shape()?;
        }
        for block in &self.blocks {
            let CompiledBlock::Once(frame) = block;
            frame.validate()?;
            validate_region_references(frame, &region_ids, &resident_program_ids)?;
        }
        let mut ids = BTreeSet::new();
        if self.regions.iter().any(|region| !ids.insert(region.id)) {
            return Err(CompiledScheduleError::DuplicateRegion);
        }
        for spec in &self.output_specs {
            let mut slots = BTreeSet::new();
            for output in &spec.outputs {
                if !slots.insert(output.slot) {
                    return Err(CompiledScheduleError::DuplicateOutputSlot);
                }
            }
        }
        Ok(())
    }
}

fn validate_region_references(
    frame: &FrameTemplate,
    region_ids: &BTreeSet<RegionId>,
    resident_program_ids: &BTreeSet<ResidentProgramId>,
) -> Result<(), CompiledScheduleError> {
    for op in &frame.ops {
        match &op.kind {
            CompiledOp::Gpu(region) => {
                if !region_ids.contains(region) {
                    return Err(CompiledScheduleError::UnknownRegion(*region));
                }
            }
            CompiledOp::ResidentControl { program, .. }
                if !resident_program_ids.contains(program) =>
            {
                return Err(CompiledScheduleError::UnknownResidentProgram(*program));
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum CompiledScheduleError {
    #[error("compiled frame metadata length does not match operation count")]
    MetadataLength,
    #[error("compiled operation refers to an out-of-range value slot")]
    ValueSlotOutOfRange,
    #[error("invalid compiled edge {0:?}")]
    InvalidEdge(Edge),
    #[error("duplicate compiled edge {0:?}")]
    DuplicateEdge(Edge),
    #[error("compiled dependency counter overflow")]
    DependencyOverflow,
    #[error("compiled dependency counters do not match edges")]
    DependencyMismatch,
    #[error("compiled successor table does not match edges")]
    SuccessorMismatch,
    #[error("compiled dependency graph contains a cycle")]
    Cycle,
    #[error("compiled region ids are not unique")]
    DuplicateRegion,
    #[error("compiled output specification contains a duplicate value slot")]
    DuplicateOutputSlot,
    #[error("compiled operation refers to an unknown region {0:?}")]
    UnknownRegion(RegionId),
    #[error("compiled region contains duplicate native binding index {0}")]
    DuplicateBinding(u32),
    #[error("compiled region input and output access overlap")]
    RegionInputOutputOverlap,
    #[error("a zero-instance resident control repeat contains templates")]
    EmptyControlRepeat,
    #[error("a non-empty resident control repeat has no template")]
    MissingControlTemplate,
    #[error("resident control phase count exceeds its active wave width")]
    ControlWaveWidthExceeded,
    #[error("compiled resident program ids are not unique")]
    DuplicateResidentProgram,
    #[error("compiled resident arena ids are not unique")]
    DuplicateResidentArenaId,
    #[error("compiled resident binding ids are not unique")]
    DuplicateResidentBinding,
    #[error("compiled operation refers to an unknown resident program {0:?}")]
    UnknownResidentProgram(ResidentProgramId),
    #[error("resident program refers to an unknown region {0:?}")]
    UnknownResidentRegion(ResidentRegionId),
    #[error("resident program refers to an unknown phase {0:?}")]
    UnknownResidentPhase(ResidentPhaseId),
    #[error("resident phase refers to an unknown instruction {0:?}")]
    UnknownResidentInstruction(ResidentInstructionId),
    #[error("resident instruction refers to an unknown binding")]
    UnknownResidentBinding,
    #[error("resident external input refers to an unknown wire slot")]
    UnknownResidentWireSlot,
}

/// Mutable builder used only during compilation.  `finish` canonicalizes and
/// freezes all edge tables, so runtime dependency counters never need to scan
/// the operation list or reconstruct the graph.
pub(crate) struct FrameTemplateBuilder {
    ops: Vec<OpTemplate>,
    edges: BTreeSet<Edge>,
    value_count: usize,
}

impl FrameTemplateBuilder {
    pub(crate) fn new(value_count: usize) -> Self {
        Self { ops: Vec::new(), edges: BTreeSet::new(), value_count }
    }

    pub(crate) fn add_op(
        &mut self,
        kind: CompiledOp,
        inputs: impl IntoIterator<Item = ValueSlot>,
        outputs: impl IntoIterator<Item = ValueSlot>,
    ) -> Result<OpId, CompiledScheduleError> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        let outputs = outputs.into_iter().collect::<Vec<_>>();
        if inputs.iter().chain(outputs.iter()).any(|slot| slot.0 as usize >= self.value_count) {
            return Err(CompiledScheduleError::ValueSlotOutOfRange);
        }
        let id = OpId(
            u32::try_from(self.ops.len()).map_err(|_| CompiledScheduleError::DependencyOverflow)?,
        );
        self.ops.push(OpTemplate {
            kind,
            inputs: inputs.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
        });
        Ok(id)
    }

    pub(crate) fn add_edge(
        &mut self,
        source: OpId,
        signal: Signal,
        destination: OpId,
    ) -> Result<(), CompiledScheduleError> {
        if source == destination ||
            source.0 as usize >= self.ops.len() ||
            destination.0 as usize >= self.ops.len()
        {
            return Err(CompiledScheduleError::InvalidEdge(Edge { source, signal, destination }));
        }
        self.edges.insert(Edge { source, signal, destination });
        Ok(())
    }

    /// Add a selected device-local serialization edge.  This method is
    /// intentionally explicit: independent work remains overlapped unless the
    /// frozen resource plan requested this boundary.
    pub(crate) fn add_resource_serial_edge(
        &mut self,
        predecessor: OpId,
        successor: OpId,
    ) -> Result<(), CompiledScheduleError> {
        self.add_edge(predecessor, Signal::Done, successor)
    }

    /// Add all consumers to one release join.  The owner is removed only
    /// after each consumer has completed: host-side resident/control work may
    /// still need the value table after the native submission itself.
    pub(crate) fn add_release_op(
        &mut self,
        values: impl IntoIterator<Item = ValueSlot>,
        devices: impl IntoIterator<Item = i32>,
        consumers: impl IntoIterator<Item = OpId>,
    ) -> Result<OpId, CompiledScheduleError> {
        let release = self.add_op(
            CompiledOp::ReleaseOwners {
                values: values.into_iter().collect::<Vec<_>>().into_boxed_slice(),
                devices: devices.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            },
            [],
            [],
        )?;
        for consumer in consumers {
            self.add_edge(consumer, Signal::Done, release)?;
        }
        Ok(release)
    }

    pub(crate) fn finish(self) -> Result<FrameTemplate, CompiledScheduleError> {
        let ops_len = self.ops.len();
        let edges = self.edges.into_iter().collect::<Vec<_>>();
        validate_acyclic(&edges, ops_len)?;
        let mut initial_dependencies = vec![0u32; ops_len];
        let mut successors = vec![Vec::<(Signal, OpId)>::new(); ops_len];
        for edge in &edges {
            initial_dependencies[edge.destination.0 as usize] += 1;
            successors[edge.source.0 as usize].push((edge.signal, edge.destination));
        }
        for next in &mut successors {
            next.sort_unstable();
        }
        let frame = FrameTemplate {
            ops: self.ops.into_boxed_slice(),
            edges: edges.into_boxed_slice(),
            initial_dependencies: initial_dependencies.into_boxed_slice(),
            successors: successors
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            value_count: self.value_count,
        };
        frame.validate()?;
        Ok(frame)
    }
}

fn validate_acyclic(edges: &[Edge], op_count: usize) -> Result<(), CompiledScheduleError> {
    let mut incoming = vec![0usize; op_count];
    let mut successors = vec![Vec::<usize>::new(); op_count];
    for edge in edges {
        incoming[edge.destination.0 as usize] += 1;
        successors[edge.source.0 as usize].push(edge.destination.0 as usize);
    }
    let mut ready = (0..op_count).filter(|index| incoming[*index] == 0).collect::<VecDeque<_>>();
    let mut visited = 0usize;
    while let Some(node) = ready.pop_front() {
        visited += 1;
        for successor in &successors[node] {
            incoming[*successor] -= 1;
            if incoming[*successor] == 0 {
                ready.push_back(*successor);
            }
        }
    }
    (visited == op_count).then_some(()).ok_or(CompiledScheduleError::Cycle)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OperationPhase {
    Pending,
    Submitted,
    Done,
}

/// Run-local dependency counters.  A single counter is shared by Submitted
/// and Done edges; the signal determines when each edge decrements it.
#[derive(Clone, Debug)]
pub(crate) struct FrameRunState {
    generation: FrameGeneration,
    dependencies: Vec<u32>,
    phases: Vec<OperationPhase>,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum FrameRunError {
    #[error("notification belongs to stale frame generation {0:?}")]
    StaleGeneration(FrameGeneration),
    #[error("operation {0:?} is not ready")]
    NotReady(OpId),
    #[error("operation {0:?} has already been submitted")]
    AlreadySubmitted(OpId),
    #[error("operation {0:?} completed before submission")]
    NotSubmitted(OpId),
}

impl FrameRunState {
    pub(crate) fn new(frame: &FrameTemplate, generation: FrameGeneration) -> Self {
        Self {
            generation,
            dependencies: frame.initial_dependencies.to_vec(),
            phases: vec![OperationPhase::Pending; frame.ops.len()],
        }
    }

    pub(crate) fn is_ready(&self, token: FrameGeneration, op: OpId) -> Result<bool, FrameRunError> {
        self.check_generation(token)?;
        let index = self.phase(op)?;
        Ok(self.phases[index] == OperationPhase::Pending && self.dependencies[index] == 0)
    }

    /// Publish Submitted only after launch/copy enqueue and native lifetime
    /// protection succeed.  Done-only edges remain pending until completion.
    pub(crate) fn mark_submitted(
        &mut self,
        frame: &FrameTemplate,
        token: FrameGeneration,
        op: OpId,
    ) -> Result<Box<[OpId]>, FrameRunError> {
        self.check_generation(token)?;
        let index = self.phase(op)?;
        if self.phases[index] != OperationPhase::Pending {
            return Err(FrameRunError::AlreadySubmitted(op));
        }
        if self.dependencies[index] != 0 {
            return Err(FrameRunError::NotReady(op));
        }
        self.phases[index] = OperationPhase::Submitted;
        Ok(self.release_signal(frame, op, Signal::Submitted))
    }

    /// Publish Done only after the completion event and any status gate pass.
    pub(crate) fn mark_done(
        &mut self,
        frame: &FrameTemplate,
        token: FrameGeneration,
        op: OpId,
    ) -> Result<Box<[OpId]>, FrameRunError> {
        self.check_generation(token)?;
        let index = self.phase(op)?;
        if self.phases[index] != OperationPhase::Submitted {
            return Err(FrameRunError::NotSubmitted(op));
        }
        self.phases[index] = OperationPhase::Done;
        Ok(self.release_signal(frame, op, Signal::Done))
    }

    fn check_generation(&self, token: FrameGeneration) -> Result<(), FrameRunError> {
        (token == self.generation).then_some(()).ok_or(FrameRunError::StaleGeneration(token))
    }

    fn phase(&self, op: OpId) -> Result<usize, FrameRunError> {
        let index = op.0 as usize;
        (index < self.phases.len()).then_some(index).ok_or(FrameRunError::NotReady(op))
    }

    fn release_signal(
        &mut self,
        frame: &FrameTemplate,
        source: OpId,
        signal: Signal,
    ) -> Box<[OpId]> {
        let mut ready = Vec::new();
        for (edge_signal, destination) in &frame.successors[source.0 as usize] {
            if *edge_signal != signal {
                continue;
            }
            let index = destination.0 as usize;
            debug_assert!(self.dependencies[index] > 0);
            if self.dependencies[index] == 0 {
                continue;
            }
            self.dependencies[index] -= 1;
            if self.dependencies[index] == 0 && self.phases[index] == OperationPhase::Pending {
                ready.push(*destination);
            }
        }
        ready.into_boxed_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(region: u32) -> CompiledOp {
        CompiledOp::Gpu(RegionId(region))
    }

    #[test]
    fn serial_resource_edge_is_done_only_when_requested() {
        let mut builder = FrameTemplateBuilder::new(0);
        let first = builder.add_op(gpu(0), [], []).unwrap();
        let second = builder.add_op(gpu(1), [], []).unwrap();
        builder.add_resource_serial_edge(first, second).unwrap();
        let frame = builder.finish().unwrap();
        assert_eq!(frame.initial_dependencies.as_ref(), &[0, 1]);
        assert_eq!(
            frame.edges.as_ref(),
            &[Edge { source: first, signal: Signal::Done, destination: second }]
        );
    }

    #[test]
    fn independent_regions_remain_overlapped() {
        let mut builder = FrameTemplateBuilder::new(0);
        let first = builder.add_op(gpu(0), [], []).unwrap();
        let second = builder.add_op(gpu(1), [], []).unwrap();
        let frame = builder.finish().unwrap();
        assert_eq!(frame.initial_dependencies.as_ref(), &[0, 0]);
        assert!(frame.edges.is_empty());
        assert_eq!(frame.successors[first.0 as usize].len(), 0);
        assert_eq!(frame.successors[second.0 as usize].len(), 0);
    }

    #[test]
    fn release_join_waits_for_all_consumers_completion() {
        let mut builder = FrameTemplateBuilder::new(1);
        let consumer_a = builder.add_op(gpu(0), [ValueSlot(0)], []).unwrap();
        let consumer_b = builder.add_op(gpu(1), [ValueSlot(0)], []).unwrap();
        let release =
            builder.add_release_op([ValueSlot(0)], [3], [consumer_a, consumer_b]).unwrap();
        let frame = builder.finish().unwrap();
        assert_eq!(frame.initial_dependencies[release.0 as usize], 2);
        assert!(
            frame
                .edges
                .iter()
                .all(|edge| edge.destination == release && edge.signal == Signal::Done)
        );
    }

    #[test]
    fn submitted_and_done_release_their_own_edges() {
        let mut builder = FrameTemplateBuilder::new(0);
        let source = builder.add_op(gpu(0), [], []).unwrap();
        let on_submit = builder.add_op(gpu(1), [], []).unwrap();
        let on_done = builder.add_op(gpu(2), [], []).unwrap();
        builder.add_edge(source, Signal::Submitted, on_submit).unwrap();
        builder.add_edge(source, Signal::Done, on_done).unwrap();
        let frame = builder.finish().unwrap();
        let token = FrameGeneration { frame: FrameId(0), generation: 0 };
        let mut state = FrameRunState::new(&frame, token);
        assert_eq!(state.mark_submitted(&frame, token, source).unwrap().as_ref(), &[on_submit]);
        assert!(!state.is_ready(token, on_done).unwrap());
        assert_eq!(state.mark_done(&frame, token, source).unwrap().as_ref(), &[on_done]);
        assert!(state.is_ready(token, on_done).unwrap());
    }

    #[test]
    fn late_notification_cannot_touch_a_reused_frame_state() {
        let mut builder = FrameTemplateBuilder::new(0);
        let source = builder.add_op(gpu(0), [], []).unwrap();
        let frame = builder.finish().unwrap();
        let current = FrameGeneration { frame: FrameId(2), generation: 9 };
        let stale = FrameGeneration { frame: FrameId(2), generation: 8 };
        let mut state = FrameRunState::new(&frame, current);
        assert_eq!(
            state.mark_submitted(&frame, stale, source),
            Err(FrameRunError::StaleGeneration(stale))
        );
        assert!(state.is_ready(current, source).unwrap());
    }

    #[test]
    fn duplicate_edges_are_collapsed_before_freezing() {
        let mut builder = FrameTemplateBuilder::new(0);
        let first = builder.add_op(gpu(0), [], []).unwrap();
        let second = builder.add_op(gpu(1), [], []).unwrap();
        builder.add_edge(first, Signal::Submitted, second).unwrap();
        builder.add_edge(first, Signal::Submitted, second).unwrap();
        let frame = builder.finish().unwrap();
        assert_eq!(frame.edges.len(), 1);
    }

    #[test]
    fn cycles_are_rejected() {
        let mut builder = FrameTemplateBuilder::new(0);
        let first = builder.add_op(gpu(0), [], []).unwrap();
        let second = builder.add_op(gpu(1), [], []).unwrap();
        builder.add_edge(first, Signal::Done, second).unwrap();
        builder.add_edge(second, Signal::Done, first).unwrap();
        assert_eq!(builder.finish(), Err(CompiledScheduleError::Cycle));
    }

    #[test]
    fn region_schema_preserves_fixed_binding_access() {
        let region = CompiledRegion {
            id: RegionId(7),
            physical_device: 0,
            operation_identity: [3; 32],
            component: NativeComponent::Fixed {
                operation_identity: [3; 32],
                implementation_variant: Arc::from("test"),
                jobs: Box::new([]),
            },
            bindings: vec![
                RegionBinding {
                    index: 0,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(1),
                        shard: 0,
                        component: NativeValueComponent::MatrixData,
                        address_addend: 0,
                    },
                    access: BindingAccess::Input,
                },
                RegionBinding {
                    index: 1,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(2),
                        shard: 0,
                        component: NativeValueComponent::MatrixData,
                        address_addend: 0,
                    },
                    access: BindingAccess::Output,
                },
            ]
            .into_boxed_slice(),
            inputs: vec![ValueSlot(1)].into_boxed_slice(),
            outputs: vec![ValueSlot(2)].into_boxed_slice(),
        };
        assert!(region.validate_schema().is_ok());
        assert_eq!(region.bindings[0].access, BindingAccess::Input);
        assert_eq!(
            region.bindings[1].source,
            BindingSource::ValueComponent {
                slot: ValueSlot(2),
                shard: 0,
                component: NativeValueComponent::MatrixData,
                address_addend: 0,
            }
        );
    }

    #[test]
    fn region_schema_rejects_duplicate_native_binding_indices() {
        let region = CompiledRegion {
            id: RegionId(0),
            physical_device: 0,
            operation_identity: [0; 32],
            component: NativeComponent::Fixed {
                operation_identity: [0; 32],
                implementation_variant: Arc::from("test"),
                jobs: Box::new([]),
            },
            bindings: vec![
                RegionBinding {
                    index: 0,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(0),
                        shard: 0,
                        component: NativeValueComponent::MatrixData,
                        address_addend: 0,
                    },
                    access: BindingAccess::Input,
                },
                RegionBinding {
                    index: 0,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(1),
                        shard: 0,
                        component: NativeValueComponent::MatrixData,
                        address_addend: 0,
                    },
                    access: BindingAccess::Input,
                },
            ]
            .into_boxed_slice(),
            inputs: vec![ValueSlot(0)].into_boxed_slice(),
            outputs: Box::new([]),
        };
        assert_eq!(region.validate_schema(), Err(CompiledScheduleError::DuplicateBinding(0)));
    }

    #[test]
    fn preimage_region_exposes_status_and_scratch_binding_indices() {
        let region = CompiledRegion {
            id: RegionId(2),
            physical_device: 0,
            operation_identity: [4; 32],
            component: NativeComponent::PreimageRetry {
                operation_identity: [4; 32],
                max_attempts: 8,
                jobs: Box::new([]),
            },
            bindings: vec![
                RegionBinding {
                    index: 3,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(5),
                        shard: 0,
                        component: NativeValueComponent::CompactPayload,
                        address_addend: 0,
                    },
                    access: BindingAccess::Output,
                },
                RegionBinding {
                    index: 4,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(5),
                        shard: 0,
                        component: NativeValueComponent::CompactHardCutoffStaging,
                        address_addend: 0,
                    },
                    access: BindingAccess::InOut,
                },
                RegionBinding {
                    index: 5,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(5),
                        shard: 0,
                        component: NativeValueComponent::CompactDeviceStatus,
                        address_addend: 0,
                    },
                    access: BindingAccess::InOut,
                },
                RegionBinding {
                    index: 6,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(5),
                        shard: 0,
                        component: NativeValueComponent::CompactHostStatus,
                        address_addend: 0,
                    },
                    access: BindingAccess::Output,
                },
            ]
            .into_boxed_slice(),
            inputs: Box::new([]),
            outputs: vec![ValueSlot(5)].into_boxed_slice(),
        };
        assert_eq!(
            region.preimage_retry_bindings(),
            Some(PreimageRetryBindingSchema {
                max_attempts: 8,
                output: vec![3].into_boxed_slice(),
                fixed_scratch: vec![4].into_boxed_slice(),
                attempt: vec![4].into_boxed_slice(),
                control: vec![5].into_boxed_slice(),
                status: vec![5].into_boxed_slice(),
                host_status: vec![6].into_boxed_slice(),
            })
        );
    }

    #[test]
    fn resident_lane_layout_keeps_one_by_one_matrix_semantics() {
        let matrix_type = ConcreteMatrixType::scalar(num_bigint::BigInt::from(97u8), 8);
        let value = ResidentTypedSlot::new(
            ValueSlot(4),
            ResidentSlotType::Matrix { wire_type: ConcreteWireType::Matrix(matrix_type.clone()) },
        );
        let full = ResidentPhysicalSlotLayout::for_slot(
            FrozenGraphScopeId::Root,
            &value,
            NonZeroUsize::new(2).unwrap(),
            2,
            [2],
            [NativeValueComponent::MatrixData, NativeValueComponent::MatrixDescriptors],
        );
        let tail = ResidentPhysicalSlotLayout::for_slot(
            FrozenGraphScopeId::Root,
            &value,
            NonZeroUsize::new(2).unwrap(),
            1,
            [2],
            [NativeValueComponent::MatrixData],
        );
        assert_eq!(value.ty.wire_type(), &ConcreteWireType::Matrix(matrix_type));
        assert_eq!(full.wave_capacity.get(), 2);
        assert_eq!(full.active_lanes, 2);
        assert_eq!(tail.active_lanes, 1);
        assert_eq!(full.batch_axes.as_ref(), &[2]);
        assert_eq!(full.components[0].selection, ResidentLaneSelection::Strided { lane_stride: 1 });

        let nested_scope = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: NodeId(9),
        };
        let nested = ResidentPhysicalSlotLayout::for_slot(
            nested_scope.clone(),
            &value,
            NonZeroUsize::new(2).unwrap(),
            2,
            [2, 2],
            [NativeValueComponent::MatrixData],
        );
        assert_ne!(full.identity, nested.identity);
        assert_eq!(nested.identity.scope, nested_scope);
        assert_eq!(nested.batch_axes.as_ref(), &[2, 2]);
    }

    #[test]
    fn resident_phase_geometry_distinguishes_full_and_tail() {
        let capacity = NonZeroUsize::new(2).unwrap();
        let full = ResidentPhaseGeometry {
            kind: ResidentPhaseKind::Full,
            wave_capacity: capacity,
            active_lanes: 2,
        };
        let tail = ResidentPhaseGeometry {
            kind: ResidentPhaseKind::Tail,
            wave_capacity: capacity,
            active_lanes: 1,
        };
        assert_eq!(full.wave_capacity, tail.wave_capacity);
        assert_ne!(full.kind, tail.kind);
        assert!(tail.active_lanes < full.active_lanes);
    }

    fn resident_shape_program(
        tail_geometry: ResidentPhaseGeometry,
    ) -> CompiledResidentControlProgram {
        let region_id = ResidentRegionId(0);
        let full_id = ResidentPhaseId(0);
        let tail_id = ResidentPhaseId(1);
        let wave_width = NonZeroUsize::new(64).unwrap();
        CompiledResidentControlProgram {
            id: ResidentProgramId(0),
            root: region_id,
            regions: vec![CompiledResidentControlRegion {
                id: region_id,
                scope: FrozenGraphScopeId::Root,
                physical_device: 0,
                instance_count: 67,
                wave_width,
                phases: vec![full_id].into_boxed_slice(),
                tail: Some(tail_id),
                inputs: Box::new([]),
                outputs: Box::new([]),
                imports: Box::new([]),
                exports: Box::new([]),
                wave: None,
            }]
            .into_boxed_slice(),
            phases: vec![
                CompiledResidentControlPhase {
                    id: full_id,
                    region: region_id,
                    width: wave_width,
                    instructions: Box::new([]),
                    ranges: Box::new([]),
                    geometry: ResidentPhaseGeometry {
                        kind: ResidentPhaseKind::Full,
                        wave_capacity: wave_width,
                        active_lanes: 64,
                    },
                    wave_base: None,
                    active_lane: None,
                },
                CompiledResidentControlPhase {
                    id: tail_id,
                    region: region_id,
                    width: NonZeroUsize::new(3).unwrap(),
                    instructions: Box::new([]),
                    ranges: Box::new([]),
                    geometry: tail_geometry,
                    wave_base: None,
                    active_lane: None,
                },
            ]
            .into_boxed_slice(),
            instructions: Box::new([]),
            wire_slots: Box::new([]),
            external_inputs: Box::new([]),
            typed_imports: Box::new([]),
            root_outputs: Box::new([]),
            bindings: Box::new([]),
            slot_layouts: Box::new([]),
            phase_bindings: Box::new([]),
            slot_count: 0,
        }
    }

    #[test]
    fn resident_shape_accepts_capacity_64_tail_3() {
        let program = resident_shape_program(ResidentPhaseGeometry {
            kind: ResidentPhaseKind::Tail,
            wave_capacity: NonZeroUsize::new(64).unwrap(),
            active_lanes: 3,
        });
        assert_eq!(program.validate_shape(), Ok(()));
    }

    #[test]
    fn resident_shape_rejects_malformed_tail_geometry_and_identity() {
        let wrong_capacity = resident_shape_program(ResidentPhaseGeometry {
            kind: ResidentPhaseKind::Tail,
            wave_capacity: NonZeroUsize::new(3).unwrap(),
            active_lanes: 3,
        });
        assert_eq!(
            wrong_capacity.validate_shape(),
            Err(CompiledScheduleError::ControlWaveWidthExceeded)
        );

        let full_tail = resident_shape_program(ResidentPhaseGeometry {
            kind: ResidentPhaseKind::Tail,
            wave_capacity: NonZeroUsize::new(64).unwrap(),
            active_lanes: 64,
        });
        assert_eq!(
            full_tail.validate_shape(),
            Err(CompiledScheduleError::ControlWaveWidthExceeded)
        );

        let mut wrong_region = resident_shape_program(ResidentPhaseGeometry {
            kind: ResidentPhaseKind::Tail,
            wave_capacity: NonZeroUsize::new(64).unwrap(),
            active_lanes: 3,
        });
        wrong_region.phases[1].region = ResidentRegionId(9);
        assert_eq!(
            wrong_region.validate_shape(),
            Err(CompiledScheduleError::ControlWaveWidthExceeded)
        );
    }

    #[test]
    fn resident_physical_binding_keys_keep_phase_lane_and_nested_alias_identity() {
        let nested = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: NodeId(3),
        };
        let component = NativeValueComponent::MatrixData;
        let key = |scope, lane, job| PhysicalBindingKey {
            scope,
            slot: ValueSlot(7),
            lane,
            device: 2,
            shard: 0,
            component,
            job,
            address_addend: 0,
            selection: ResidentPhysicalBindingSelection::WaveRelativeLane(lane),
        };
        let full = ResidentPhaseBindingSchema {
            phase: ResidentPhaseId(1),
            bindings: vec![
                ResidentPhysicalBinding {
                    key: key(FrozenGraphScopeId::Root, 0, 0),
                    index: 0,
                    access: BindingAccess::Input,
                },
                ResidentPhysicalBinding {
                    key: key(FrozenGraphScopeId::Root, 1, 0),
                    index: 1,
                    access: BindingAccess::Input,
                },
            ]
            .into_boxed_slice(),
        };
        let tail = ResidentPhaseBindingSchema {
            phase: ResidentPhaseId(2),
            bindings: vec![ResidentPhysicalBinding {
                key: key(FrozenGraphScopeId::Root, 0, 1),
                index: 0,
                access: BindingAccess::InOut,
            }]
            .into_boxed_slice(),
        };
        assert_ne!(full.bindings[0].key, full.bindings[1].key);
        assert_ne!(full.bindings[0].key, tail.bindings[0].key);
        assert_ne!(
            full.bindings[0].key,
            key(nested.clone(), 0, 0),
            "nested scope aliases must not resolve to the root slot"
        );
        assert_eq!(tail.bindings[0].access, BindingAccess::InOut);
        let absolute_element = PhysicalBindingKey {
            selection: ResidentPhysicalBindingSelection::AbsoluteFamilyElement(0),
            ..key(FrozenGraphScopeId::Root, 0, 0)
        };
        let second_absolute_element = PhysicalBindingKey {
            selection: ResidentPhysicalBindingSelection::AbsoluteFamilyElement(1),
            ..key(FrozenGraphScopeId::Root, 0, 0)
        };
        assert_ne!(absolute_element, second_absolute_element);
        assert_ne!(absolute_element, full.bindings[0].key);
    }
}
