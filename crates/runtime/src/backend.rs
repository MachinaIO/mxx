use crate::{
    gpu_column_policy::{
        CanonicalWarmupProfileDomain, ColumnRange, FusedWarmupOperation,
        GpuExecutionRouteDescriptor, GpuRouteResolutionInput, WarmupMeasurementKind,
        fused_warmup_profile_domain, resolve_gpu_route,
    },
    gpu_execution_plan::FrozenGpuPlan,
    host_control::HostControlChild,
    transcript::DrawSite,
};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ArtifactType, ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    graph::FrozenGraphScopeId,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp},
    types::{ConcreteMatrixType, ConcreteWireType, InstantiationFrame, NodeId, WireRef},
};
use mxx_primitives::matrix::{PolyMatrix, PolyMatrixColumnSource};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{self, Debug},
    sync::Arc,
};

pub mod poly;
#[cfg(feature = "gpu")]
pub mod poly_gpu;

#[derive(Clone, Debug)]
pub struct PreimageRequest<M, T> {
    pub instance_slot: usize,
    /// Fixed-plan site/layout contract for the sampled instance.  Legacy
    /// callers leave this absent; fixed executor paths always populate it.
    pub fixed_metadata: Option<PlannedNodeBatchRequest>,
    pub matrix_type: ConcreteMatrixType,
    pub sigma: f64,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub max_coefficient_bound: BigInt,
    pub trapdoor: Arc<T>,
    pub public: Arc<M>,
    pub target: Arc<dyn PolyMatrixColumnSource<M>>,
    /// Seed for deterministic GPU sampling with a fixed column schedule.
    pub randomness_seed: [u8; 32],
}

/// Opaque identity of the native preimage covariance cache.  Runtime profile
/// keys carry only this digest: trapdoors and other secret material never
/// cross the warmup/cache boundary.  The digest binds native ownership,
/// sampler parameters, and the ordered basis used by the production call.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct PreimageCacheIdentity([u8; 32]);

impl PreimageCacheIdentity {
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    pub const fn digest(self) -> [u8; 32] {
        self.0
    }

    /// Native identity conversion is available only with the GPU primitive
    /// implementation.  A caller without a native identity cannot construct
    /// this value and therefore fails closed when a preimage profile is
    /// requested.
    #[cfg(feature = "gpu")]
    pub fn from_native(
        native: &mxx_primitives::sampler::trapdoor::gpu::PreimageCacheIdentity,
        sampler_parameters: &[u64],
        ordered_basis: &[u64],
    ) -> Self {
        Self(native.opaque_digest(sampler_parameters, ordered_basis))
    }
}

/// Metadata passed to a backend immediately before a fixed node batch is
/// submitted.  It deliberately contains no matrix owner or native pointer;
/// those are bound by the backend to the current runtime values.  The
/// executor keeps the original instance slots so a backend cannot accidentally
/// use a compressed batch index as a sampling or output coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedNodeBatchRequest {
    pub site: u64,
    pub shape_class: u64,
    pub instance_class: u64,
    pub operation_identity: [u8; 32],
    pub implementation_variant: String,
    pub output_layouts: Vec<crate::gpu_execution_plan::LayoutId>,
    /// Explicit port-to-layout bindings. `output_layouts` is retained as the
    /// compact plan representation, while this field prevents a backend from
    /// accidentally treating a reordered fused output as positional-only.
    pub output_port_layouts: Vec<(usize, crate::gpu_execution_plan::LayoutId)>,
    /// Concrete source layouts checked at submit time.  `layout_id` is absent
    /// when a source is a runtime value rather than a planned output.
    pub source_layouts: Vec<PlannedLayoutMetadata>,
    /// Effective operands after lowering fused nodes.  A compact product's
    /// logical concat is represented by its origin wires and lowered row
    /// blocks, while its compact RHS remains a distinct typed operand.
    /// `source_layouts` therefore describes only the physical block sources
    /// for that operation, never an ambiguous logical `(concat, rhs)` pair.
    pub effective_operands: Vec<PlannedOperandMetadata>,
    pub output_layout_metadata: Vec<PlannedLayoutMetadata>,
    pub columns_per_job: Vec<usize>,
    pub instance_slots: Vec<usize>,
    pub instance_paths: Vec<Vec<InstantiationFrame>>,
    pub draw_sites: Vec<Option<DrawSite>>,
    pub randomness_seeds: Vec<Option<[u8; 32]>>,
    pub output_ports: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlannedOperandMetadata {
    RowBlock {
        logical_operand: usize,
        block_index: usize,
        origin: WireRef,
        layout: PlannedLayoutMetadata,
    },
    CompactRhs {
        logical_operand: usize,
        origin: WireRef,
        layout: PlannedLayoutMetadata,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedLayoutMetadata {
    pub layout_id: Option<crate::gpu_execution_plan::LayoutId>,
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub representation: String,
}

impl PlannedLayoutMetadata {
    pub fn for_type(
        ty: &ConcreteWireType,
        layout_id: Option<crate::gpu_execution_plan::LayoutId>,
    ) -> Self {
        let matrix = ty.matrix_type();
        Self {
            layout_id,
            rows: matrix.map_or(0, |matrix| matrix.rows),
            columns: matrix.map_or(0, |matrix| matrix.columns),
            ring_dimension: matrix.map_or(0, |matrix| matrix.ring_dimension),
            representation: format!("{ty:?}"),
        }
    }
}

impl PlannedNodeBatchRequest {
    /// Canonical fixed metadata for a lowered operation. Warmup supplies its
    /// candidate output layouts; execution supplies the frozen plan layouts.
    /// Instance paths and random draws are bound by the respective caller.
    pub fn for_lowered_operation(
        key: crate::gpu_execution_plan::GpuExecutionSiteKey,
        operation_identity: [u8; 32],
        implementation_variant: String,
        inputs: GpuEffectiveInputs,
        outputs: Vec<PlannedLayoutMetadata>,
        columns_per_job: Vec<usize>,
    ) -> Self {
        let output_layouts =
            outputs.iter().filter_map(|output| output.layout_id).collect::<Vec<_>>();
        Self {
            site: key.site,
            shape_class: key.shape_class,
            instance_class: key.instance_class,
            operation_identity,
            implementation_variant,
            output_port_layouts: output_layouts
                .iter()
                .enumerate()
                .map(|(port, layout)| (port, *layout))
                .collect(),
            output_layouts,
            source_layouts: inputs.source_layouts,
            effective_operands: inputs.operands,
            output_ports: outputs.len(),
            output_layout_metadata: outputs,
            columns_per_job,
            instance_slots: vec![0],
            instance_paths: vec![Vec::new()],
            draw_sites: vec![None],
            randomness_seeds: vec![None],
        }
    }
}

/// Concrete storage facts supplied by a backend to the warmup planner.  This
/// is intentionally keyed by the concrete matrix type rather than by a
/// caller-provided tower count: the backend is the authority for the ordered
/// CRT basis, active level, representation, and native limb width.
/// Storage encoding is a value-level fact, not a free-form label.  The
/// `Unknown` variant is retained only at the boundary so malformed contracts
/// can be rejected without silently changing their meaning; native backends
/// should construct one of the two known variants directly.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum BackendStorageRepresentation {
    FullDcrt,
    CompactBounded,
    Unknown(String),
}

impl BackendStorageRepresentation {
    pub const fn full_dcrt() -> Self {
        Self::FullDcrt
    }

    pub const fn compact_bounded() -> Self {
        Self::CompactBounded
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::FullDcrt => "full_dcrt",
            Self::CompactBounded => "compact_bounded",
            Self::Unknown(value) => value.as_str(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.as_str().is_empty()
    }

    pub fn is_known(&self) -> bool {
        !matches!(self, Self::Unknown(_))
    }
}

impl From<&str> for BackendStorageRepresentation {
    fn from(value: &str) -> Self {
        match value {
            "full_dcrt" => Self::FullDcrt,
            "compact_bounded" => Self::CompactBounded,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

impl From<String> for BackendStorageRepresentation {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl PartialEq<&str> for BackendStorageRepresentation {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct BackendStorageDescriptor {
    pub representation: BackendStorageRepresentation,
    pub ordered_crt_basis: Vec<u64>,
    pub level: usize,
    pub limb_bytes: usize,
}

impl BackendStorageDescriptor {
    pub fn active_towers(&self) -> usize {
        self.level.saturating_add(1)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendStorageContract {
    /// Keyed by the complete concrete wire type, not just its matrix shape.
    /// This keeps full-DCRT matrices distinct from compact bounded RHS and
    /// preimage values that happen to share rows/columns/modulus.
    pub descriptors: BTreeMap<ConcreteWireType, BackendStorageDescriptor>,
    pub active_crt_towers: usize,
    pub crt_limb_bytes: usize,
}

/// Validate the structural part of a backend storage contract before a
/// backend-specific equality check.  In particular, a nonempty set of
/// concrete matrix types may not be represented by an empty descriptor map or
/// by a fabricated one-tower summary of a deeper CRT basis.
pub fn validate_backend_storage_contract(
    types: &[ConcreteWireType],
    contract: &BackendStorageContract,
) -> Result<(), String> {
    if contract.active_crt_towers == 0 || contract.crt_limb_bytes == 0 {
        return Err("GPU storage contract has no active CRT towers or limb width".into());
    }
    let mut expected = types.to_vec();
    expected.sort_unstable();
    expected.dedup();
    let actual = contract.descriptors.keys().cloned().collect::<Vec<_>>();
    if actual != expected {
        return Err("GPU storage contract descriptor types do not match concrete inputs".into());
    }
    let deepest = contract
        .descriptors
        .values()
        .map(|descriptor| descriptor.level.saturating_add(1))
        .max()
        .unwrap_or(0);
    if deepest != contract.active_crt_towers {
        return Err("GPU storage contract active CRT depth disagrees with its descriptors".into());
    }
    if contract.descriptors.values().any(|descriptor| {
        descriptor.representation.is_empty() ||
            !descriptor.representation.is_known() ||
            descriptor.ordered_crt_basis.is_empty() ||
            descriptor.level >= descriptor.ordered_crt_basis.len() ||
            descriptor.limb_bytes != contract.crt_limb_bytes
    }) {
        return Err("GPU storage contract contains an invalid physical descriptor".into());
    }
    Ok(())
}

/// Fixed-plan fused primitives grouped across sibling instances.
///
/// Metadata is mandatory so a fixed request cannot accidentally enter the
/// backend without its frozen site, typed operands, and instance binding.
pub enum FusedBatchRequest<M, S> {
    RowSum {
        metadata: PlannedNodeBatchRequest,
        source: Arc<M>,
        right: Option<Arc<M>>,
        rows: Vec<Vec<usize>>,
    },
    Decompose {
        metadata: PlannedNodeBatchRequest,
        blocks: Vec<Arc<M>>,
        small: bool,
        digits: usize,
    },
    SmallProduct {
        metadata: PlannedNodeBatchRequest,
        blocks: Vec<Arc<M>>,
        rhs: Arc<S>,
    },
    Add {
        metadata: PlannedNodeBatchRequest,
        blocks: Vec<Arc<M>>,
        right: Arc<M>,
    },
}

impl<M, S> FusedBatchRequest<M, S> {
    /// Return the distinct canonical domain for the actual fused kernel. A
    /// row sum with a RHS is tensor-row-sum and must not reuse ordinary row
    /// sum's workspace profile.
    pub fn canonical_profile_domain(&self) -> CanonicalWarmupProfileDomain {
        match self {
            Self::RowSum { right: Some(_), .. } => {
                fused_warmup_profile_domain(FusedWarmupOperation::TensorRowSum)
            }
            Self::RowSum { right: None, .. } => {
                fused_warmup_profile_domain(FusedWarmupOperation::RowSum)
            }
            Self::Decompose { .. } => fused_warmup_profile_domain(FusedWarmupOperation::Decompose),
            Self::SmallProduct { .. } => {
                fused_warmup_profile_domain(FusedWarmupOperation::CompactProduct)
            }
            Self::Add { .. } => fused_warmup_profile_domain(FusedWarmupOperation::RowBlockAdd),
        }
    }
}

impl<M, S> DynamicFusedBatchRequest<M, S> {
    /// Dynamic requests use the same operation identity as their fixed
    /// counterparts, so a provider can use one closed domain for pilots and
    /// production without an implicit generic-fused fallback.
    pub fn canonical_profile_domain(&self) -> CanonicalWarmupProfileDomain {
        match self {
            Self::RowSum { right: Some(_), .. } => {
                fused_warmup_profile_domain(FusedWarmupOperation::TensorRowSum)
            }
            Self::RowSum { right: None, .. } => {
                fused_warmup_profile_domain(FusedWarmupOperation::RowSum)
            }
            Self::Decompose { .. } => fused_warmup_profile_domain(FusedWarmupOperation::Decompose),
            Self::SmallProduct { .. } => {
                fused_warmup_profile_domain(FusedWarmupOperation::CompactProduct)
            }
            Self::Add { .. } => fused_warmup_profile_domain(FusedWarmupOperation::RowBlockAdd),
        }
    }
}

/// Canonical domain for the batched preimage entry point. It is kept as a
/// function because preimage requests carry per-instance sampler metadata,
/// not a `FusedBatchRequest` payload.
pub const fn preimage_batch_profile_domain() -> CanonicalWarmupProfileDomain {
    fused_warmup_profile_domain(FusedWarmupOperation::PreimageBatch)
}

/// Dynamic fused primitives retain the backend's ordinary CPU-compatible
/// fallback semantics. They are a distinct type so dynamic execution cannot
/// manufacture a fixed request with absent plan metadata.
pub enum DynamicFusedBatchRequest<M, S> {
    RowSum { source: Arc<M>, right: Option<Arc<M>>, rows: Vec<Vec<usize>> },
    Decompose { blocks: Vec<Arc<M>>, small: bool, digits: usize },
    SmallProduct { blocks: Vec<Arc<M>>, rhs: Arc<S> },
    Add { blocks: Vec<Arc<M>>, right: Arc<M> },
}

pub enum FusedBatchOutput<M, S> {
    Matrices(Vec<M>),
    Small(S),
}

/// A sibling batch for ordinary fixed-plan operations.  The metadata is
/// carried per original instance rather than per compressed batch position;
/// a fleet backend can therefore rotate owners using the frozen slot/path
/// without rediscovering a pilot or selecting a different implementation.
pub enum FixedOperationBatchRequest<M, S> {
    /// A constant whose native constructor is indivisible and pinned to the
    /// owner selected by the frozen plan.  Keeping this distinct from
    /// generated-column constants prevents a SingleDeviceConstant from
    /// silently taking the sharded generated path.
    SingleDeviceConstant {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        value: ConstantMatrix,
        env: ParamEnv,
    },
    GeneratedConstant {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        value: ConstantMatrix,
        env: ParamEnv,
    },
    LiftIntegerToConstantPolynomial {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        coefficient: BigInt,
    },
    MatrixBinary {
        metadata: PlannedNodeBatchRequest,
        operation: MatrixBinaryOp,
        left: Arc<M>,
        right: Arc<M>,
    },
    MatrixMulSmallRhs {
        metadata: PlannedNodeBatchRequest,
        left: Arc<M>,
        right: Arc<S>,
    },
    MatrixMulAccumulate {
        metadata: PlannedNodeBatchRequest,
        request: MatrixMulAccumulateRequest<M>,
    },
    Negate {
        metadata: PlannedNodeBatchRequest,
        value: Arc<M>,
    },
    Scale {
        metadata: PlannedNodeBatchRequest,
        value: Arc<M>,
        scalar: BigInt,
    },
    UnaryTransform {
        metadata: PlannedNodeBatchRequest,
        operation: FixedUnaryOperation,
        value: Arc<M>,
    },
    Tensor {
        metadata: PlannedNodeBatchRequest,
        left: Arc<M>,
        right: Arc<M>,
    },
    Concat {
        metadata: PlannedNodeBatchRequest,
        inputs: Vec<Arc<M>>,
        axis: ConcatAxis,
    },
    CrtRecompose {
        metadata: PlannedNodeBatchRequest,
        levels: Vec<M>,
        plaintext_moduli: Vec<BigInt>,
        reconstruction_coefficients: Vec<BigInt>,
        destination: ConcreteMatrixType,
    },
}

/// A fixed-plan trapdoor construction request.  Trapdoor sampling produces a
/// public matrix and secret native value together, so it cannot use the
/// matrix-only fixed-operation batch response.  The metadata is mandatory and
/// binds the request to the already-frozen owner/layout/implementation.
pub struct FixedTrapdoorRequest {
    pub metadata: PlannedNodeBatchRequest,
    pub ty: ConcreteMatrixType,
    pub sigma: f64,
    pub gadget_base: BigInt,
    pub digit_count: usize,
}

#[derive(Clone, Debug)]
pub enum FixedUnaryOperation {
    RingAutomorphism {
        index: usize,
    },
    ModulusSwitch {
        destination: ConcreteMatrixType,
    },
    ReduceModulus {
        destination: ConcreteMatrixType,
    },
    CenteredRebase {
        destination: ConcreteMatrixType,
    },
    RnsModUp {
        destination: ConcreteMatrixType,
        source_moduli: Vec<u64>,
        digit_size: usize,
        normalize: bool,
    },
    RnsModDown {
        destination: ConcreteMatrixType,
        source_moduli: Vec<u64>,
        plaintext_modulus: u64,
    },
    Transpose,
    Slice {
        rows: Option<IndexRange>,
        columns: Option<IndexRange>,
    },
}

/// Fixed-plan generated-column requests.  Sampling backends receive one
/// request per original instance, so owner rotation never depends on a
/// compressed batch ordinal.
pub enum FixedGenerationRequest {
    Uniform {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        range: SampleRange,
    },
    Gaussian {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: BigInt,
    },
    Hash {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        key: [u8; 32],
        tag: Vec<u8>,
    },
    HashDecomposed {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        key: [u8; 32],
        tag: Vec<u8>,
        gadget_base: BigInt,
        digit_count: usize,
        small: bool,
    },
}

pub enum FixedGenerationOutput<M, S> {
    Matrix(M),
    Small(S),
}

pub struct FixedGadgetDecomposeRequest<M> {
    pub metadata: PlannedNodeBatchRequest,
    pub input: Arc<M>,
    pub small: bool,
    pub digits: usize,
}

/// Full logical preimage target whose expanded columns are loaded on demand.
/// The staged constructor owns host bytes, allowing a GPU owner to be dropped
/// before sampling while preserving the original matrix dimensions.
pub struct PreimageTarget<M> {
    rows: usize,
    columns: usize,
    loader: Arc<dyn Fn(usize, usize) -> M + Send + Sync>,
}

impl<M> Clone for PreimageTarget<M> {
    fn clone(&self) -> Self {
        Self { rows: self.rows, columns: self.columns, loader: self.loader.clone() }
    }
}

impl<M> Debug for PreimageTarget<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreimageTarget")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl<M> PreimageTarget<M>
where
    M: PolyMatrix + 'static,
{
    pub fn staged(
        params: <M::P as mxx_primitives::poly::Poly>::Params,
        rows: usize,
        columns: usize,
        bytes: Arc<Vec<u8>>,
    ) -> Self {
        let loader = Arc::new(move |start: usize, end: usize| {
            M::from_cpu_staging_columns(&params, bytes.as_slice(), start, end)
        });
        Self { rows, columns, loader }
    }

    pub fn resident(value: Arc<M>) -> Self {
        let rows = value.row_size();
        let columns = value.col_size();
        let loader = Arc::new(move |start: usize, end: usize| value.slice_columns(start, end));
        Self { rows, columns, loader }
    }
}

impl<M> PolyMatrixColumnSource<M> for PreimageTarget<M>
where
    M: PolyMatrix + 'static,
{
    fn row_size(&self) -> usize {
        self.rows
    }

    fn col_size(&self) -> usize {
        self.columns
    }

    fn load_columns(&self, start: usize, end: usize) -> M {
        assert!(start <= end && end <= self.columns, "invalid preimage target column interval");
        (self.loader)(start, end)
    }
}

#[derive(Clone, Debug)]
pub struct MatrixMulAccumulateRequest<M> {
    pub products: Vec<(BigInt, Arc<M>, Arc<M>)>,
    pub bias: Option<Arc<M>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IndexRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleRange {
    pub minimum: BigInt,
    pub maximum: BigInt,
}

/// The identity of an operation whose setup-time GPU profile is being
/// collected.  Shape and instance classes are part of the key because the
/// same operation can use different kernels (and therefore different
/// workspaces) for different concrete shapes or loop bindings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct GpuWarmupOperationSignature {
    pub operation: [u8; 32],
    pub shape_class: u64,
    pub instance_class: u64,
}

/// One exact production range candidate requested by warmup.  `tile_width`
/// is the candidate `b`; `range` identifies the output columns exercised by
/// the device-local production path.  A provider must measure this candidate
/// or return [`GpuWarmupProfileError::MissingProfile`].
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GpuWarmupProfileRequest {
    pub signature: GpuWarmupOperationSignature,
    /// Logical device whose device-local production range is measured.
    pub device: usize,
    /// Physical execution identity.  A logical ordinal is not sufficient to
    /// reuse a native measurement after device replacement or context reset.
    pub device_identity: GpuWarmupDeviceIdentity,
    pub tile_width: usize,
    pub range: IndexRange,
    /// Authoritative identity of the range actually submitted by production.
    /// These are kept separate from the requested tile width because a tail,
    /// mapped segment, or compact fragment may execute a different range.
    pub executed_range_start: usize,
    pub executed_range_class: GpuWarmupFragmentClass,
    /// Physical transfer/fragment context used by the representative call.
    pub route: GpuWarmupRoute,
    /// Physical route selected by the production range mapper.  `route` is
    /// retained as a compact diagnostic discriminator; providers and cache
    /// keys must use this complete descriptor so source/destination ranges,
    /// compact conversion, and staging bytes cannot collide.
    pub route_descriptor: GpuExecutionRouteDescriptor,
    /// Exact per-job resolver facts used to produce `route_descriptor`.
    /// Keeping this on the request prevents a provider from silently
    /// substituting a device-local representative for a peer/staged job.
    pub route_resolver: Option<GpuWarmupRouteResolverData>,
    /// Fused output port whose physical layout binds this local union job.
    /// `None` is the canonical value for ordinary (single-output) jobs;
    /// fused jobs must carry the exact port selected by fixed dispatch.
    pub binding_port: Option<usize>,
    pub fragment: GpuWarmupFragmentClass,
    /// Sampler/cache settings that affect both allocation and elapsed time.
    /// `cache_identity` is an opaque, canonical digest supplied by the native
    /// provider; it must not be reconstructed from a display name.
    pub retry_cap: Option<usize>,
    pub cache_identity: Option<[u8; 32]>,
    /// Cache lifecycle state represented by this point.  Ordinary operations
    /// use `Warm`; preimage setup and steady-state sampling deliberately use
    /// separate requests so their evidence cannot be merged accidentally.
    pub cache_state: GpuWarmupCacheState,
    /// Timing scope of the requested production-equivalent call.  In
    /// particular, preimage cold setup is a setup point while its sampler
    /// point is a local job.
    pub timing_scope: GpuWarmupTimingScope,
}

impl GpuWarmupProfileRequest {
    /// The interpolation coordinate for this request.  Ordinary production
    /// jobs use local column width; a transfer-only profile uses the exact
    /// payload byte count.  Callers constructing a transfer request must set
    /// `tile_width` to the value returned by
    /// `route_descriptor.transfer_bytes()` and use a non-empty physical
    /// range, so cache identity cannot collapse two payload sizes.
    pub fn coordinate(&self) -> usize {
        if self.timing_scope == GpuWarmupTimingScope::Transfer {
            self.route_descriptor.transfer_bytes()
        } else {
            self.tile_width
        }
    }

    pub fn validate_coordinate(&self) -> Result<usize, GpuWarmupProfileError> {
        let coordinate = self.coordinate();
        if coordinate == 0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "warmup profile coordinate must be positive".into(),
            ));
        }
        Ok(coordinate)
    }

    pub fn with_preimage_cache_identity(mut self, identity: PreimageCacheIdentity) -> Self {
        self.cache_identity = Some(identity.digest());
        self
    }

    pub fn preimage_cache_identity(&self) -> Option<PreimageCacheIdentity> {
        self.cache_identity.map(PreimageCacheIdentity::from_digest)
    }

    /// Require the native cache identity at the preimage boundary.  This is
    /// intentionally explicit because ordinary operations have no retained
    /// sampler cache and legitimately leave `cache_identity` absent.
    pub fn require_preimage_cache_identity(
        &self,
    ) -> Result<PreimageCacheIdentity, GpuWarmupProfileError> {
        self.preimage_cache_identity().ok_or_else(|| {
            GpuWarmupProfileError::InvalidMeasurement(
                "preimage warmup requires a native cache identity".into(),
            )
        })
    }
}

/// Setup-only descriptor registered by the validated warmup lifecycle before
/// it asks for any candidate profile.  It carries the effective operation
/// identity plus the concrete production inputs needed by a backend adapter;
/// fixed execution never receives this descriptor.
#[derive(Clone, Debug)]
pub struct GpuWarmupOperationDescriptor {
    pub signature: GpuWarmupOperationSignature,
    pub scope: FrozenGraphScopeId,
    pub node: NodeId,
    pub kind: mxx_ir_core::node::NodeKind,
    /// Exact physical inputs and fused row groups selected by executor lowering.
    pub inputs: GpuEffectiveInputs,
    pub concrete_argument_types: Vec<ConcreteWireType>,
    pub concrete_output_types: Vec<ConcreteWireType>,
    pub bindings: ParamEnv,
    pub effective_operation: String,
    /// Closed profile domain for the operation actually dispatched by the
    /// production path.  This must not be reconstructed from `kind`: fused
    /// row sums, row-block additions, decomposition, compact products, and
    /// preimage batches all retain an ordinary IR node kind while using a
    /// different kernel and workspace contract.
    pub profile_domain: CanonicalWarmupProfileDomain,
    /// Fused operation selected by the production lowering, if any.  The
    /// provider uses this to bind its representative to the same fused
    /// dispatch entry point used by fixed execution.
    pub fused_operation: Option<FusedWarmupOperation>,
    pub implementation_variant: String,
    /// Physical storage inventory captured from the selected production
    /// layout.  Warmup uses this to build the same bounded range request as
    /// fixed execution; it must not infer ownership from the logical width.
    pub source_layouts: Vec<GpuWarmupStorageLayout>,
    pub output_layout: Option<GpuWarmupStorageLayout>,
    /// Route resolver data is supplied by the fleet adapter.  `None` is
    /// accepted only for host-only operations; GPU requests must carry this
    /// contract and cannot silently fall back to `DeviceLocal`.
    pub route_resolver: Option<GpuWarmupRouteResolverData>,
    /// Validated child scope, bindings, and body for host/control operations.
    /// GPU operation descriptors leave this unset; structural providers use it
    /// to execute the same control dispatch as production without inventing a
    /// synthetic empty result.
    pub host_control: Option<HostControlChild>,
}

/// Shared, value-only lowering contract for measurement and fixed dispatch.
/// Origins are ordered exactly as the concrete backend request; logical
/// concatenations that execution elides never appear as replacement inputs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GpuEffectiveInputs {
    pub origins: Vec<WireRef>,
    pub source_layouts: Vec<PlannedLayoutMetadata>,
    pub operands: Vec<PlannedOperandMetadata>,
    pub row_groups: Vec<Vec<usize>>,
}

/// Value-only physical layout facts used by setup-time warmup.  The owner
/// intervals are `(device, start, end)` in global column coordinates.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GpuWarmupStorageLayout {
    pub layout_id: Option<crate::gpu_execution_plan::LayoutId>,
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub representation: BackendStorageRepresentation,
    /// Deterministic owner rotation used by fixed sibling dispatch.  The
    /// intervals below describe slot zero; callers resolving a concrete
    /// instance must apply this stride before deriving source owners.
    pub instance_device_stride: usize,
    pub owner_intervals: Vec<(usize, usize, usize)>,
}

/// Physical route facts for one production range.  The fleet owns capability
/// discovery; warmup only invokes the shared pure resolver below.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GpuWarmupRouteResolverData {
    pub source_layouts: Vec<GpuWarmupStorageLayout>,
    pub output_layout: Option<GpuWarmupStorageLayout>,
    pub source_owners: Vec<usize>,
    pub destination_owner: usize,
    pub source_range: ColumnRange,
    pub destination_range: ColumnRange,
    pub source_compact: bool,
    pub destination_compact: bool,
    pub peer_available: bool,
    pub source_staging_bytes: usize,
    pub host_staging_bytes: usize,
    pub pinned_host_staging_bytes: usize,
}

impl GpuWarmupRouteResolverData {
    /// Whether this resolver contains an actual fleet capability result.
    ///
    /// A resolver assembled from logical layouts alone has no information
    /// about peer access or host staging and is represented by
    /// `peer_available == false` with zero staging bytes.  That combination
    /// is only conclusive for a resident range; for every cross-device range
    /// it is unresolved and must not be admitted as a device-local profile.
    pub fn has_physical_capability(&self) -> bool {
        if self.source_owners.is_empty() {
            return false;
        }
        let resident = self.source_owners.iter().all(|owner| *owner == self.destination_owner);
        resident ||
            self.peer_available ||
            self.source_staging_bytes != 0 ||
            self.host_staging_bytes != 0 ||
            self.pinned_host_staging_bytes != 0
    }

    /// A logical cross-device mapping has an owner/destination pair, but no
    /// peer or staging capability result yet. Such a mapping is a provider
    /// query, not a device-local route and must never be used as a cache key.
    pub fn is_unresolved_cross_device(&self) -> bool {
        !self.has_physical_capability() &&
            self.source_owners.iter().any(|owner| *owner != self.destination_owner)
    }

    pub fn resolve(
        &self,
        fragment: crate::gpu_column_policy::GpuFragmentClass,
    ) -> GpuExecutionRouteDescriptor {
        self.resolve_for(
            self.source_range,
            self.destination_range,
            fragment,
            self.source_compact,
            self.destination_compact,
        )
    }

    pub fn resolve_for(
        &self,
        source_range: ColumnRange,
        destination_range: ColumnRange,
        fragment: crate::gpu_column_policy::GpuFragmentClass,
        source_compact: bool,
        destination_compact: bool,
    ) -> GpuExecutionRouteDescriptor {
        // Derive ownership from every interval intersecting the requested
        // range.  A range that is not contained in one owner interval must
        // never be silently assigned to the destination job's device: that
        // used to turn a cross-device concat/tensor operand into a bogus
        // Resident profile.  The aggregate route remains a convenience for
        // the fixed materializer, while source_routes retain each owner's
        // physical fragment and are part of the profile key.
        let fragments = self.source_fragments_for_range(source_range);
        let source_is_fully_covered = range_is_fully_covered(source_range, &fragments);
        let source_is_resident = source_is_fully_covered &&
            !fragments.is_empty() &&
            fragments.iter().all(|(owner, _)| *owner == self.destination_owner);
        let source_owner = fragments
            .iter()
            .find(|(owner, _)| *owner != self.destination_owner)
            .or_else(|| fragments.first())
            .map(|(owner, _)| *owner)
            .filter(|_| source_is_fully_covered)
            .or_else(|| (!fragments.is_empty()).then(|| fragments[0].0));
        let descriptor = resolve_gpu_route(GpuRouteResolutionInput {
            source_device: source_owner,
            destination_device: Some(self.destination_owner),
            source_range,
            destination_range,
            source_is_resident,
            peer_available: self.peer_available,
            source_compact,
            destination_compact,
            fragment,
            source_staging_bytes: self.source_staging_bytes,
            host_staging_bytes: self.host_staging_bytes,
            pinned_host_staging_bytes: self.pinned_host_staging_bytes,
        });

        let mut source_routes = Vec::with_capacity(fragments.len());
        let mut remaining_source = self.source_staging_bytes;
        let mut remaining_host = self.host_staging_bytes;
        let mut remaining_pinned = self.pinned_host_staging_bytes;
        let total_columns = fragments
            .iter()
            .filter(|(owner, _)| *owner != self.destination_owner)
            .map(|(_, range)| range.len())
            .sum::<usize>()
            .max(1);
        for (index, (owner, source_range)) in fragments.iter().copied().enumerate() {
            // A destination-owned fragment can remain resident even when a
            // different fragment in the same logical range requires peer or
            // host staging.  The aggregate descriptor is deliberately not
            // Resident in that mixed case, but the per-source route records
            // the true local/remote path used by the provider.
            let route = if owner == self.destination_owner {
                crate::gpu_column_policy::GpuTransferRoute::Resident
            } else if self.peer_available {
                crate::gpu_column_policy::GpuTransferRoute::Peer
            } else {
                crate::gpu_column_policy::GpuTransferRoute::HostStaging
            };
            // Preserve aggregate staging totals while assigning bytes to each
            // owner.  The final owner receives the remainder, avoiding any
            // rounding loss in the route identity.
            let transfer_required = route != crate::gpu_column_policy::GpuTransferRoute::Resident;
            let last_transfer = !fragments[index + 1..]
                .iter()
                .any(|(next_owner, _)| *next_owner != self.destination_owner);
            let (source_bytes, host_bytes, pinned_bytes) = if !transfer_required {
                (0, 0, 0)
            } else if last_transfer {
                (remaining_source, remaining_host, remaining_pinned)
            } else {
                let share = source_range.len();
                let total = total_columns;
                let source = self.source_staging_bytes.saturating_mul(share) / total;
                let host = self.host_staging_bytes.saturating_mul(share) / total;
                let pinned = self.pinned_host_staging_bytes.saturating_mul(share) / total;
                remaining_source = remaining_source.saturating_sub(source);
                remaining_host = remaining_host.saturating_sub(host);
                remaining_pinned = remaining_pinned.saturating_sub(pinned);
                (source, host, pinned)
            };
            source_routes.push(crate::gpu_column_policy::GpuExecutionSourceRoute::new(
                owner,
                source_range,
                route,
                source_bytes,
                host_bytes,
                pinned_bytes,
            ));
        }
        descriptor.with_source_routes(source_routes).unwrap_or_else(|| {
            // An invalid/uncovered range is deliberately represented by an
            // invalid descriptor.  The caller will fail closed instead of
            // admitting a synthetic device-local route.
            let mut invalid = descriptor;
            invalid.source_route_count = 0;
            invalid
        })
    }

    /// Return all physical owner fragments intersecting `range`, ordered by
    /// global column position.  Duplicate intervals from multiple operands
    /// are retained only once; adjacent fragments owned by the same device
    /// are coalesced by `with_source_routes` when the descriptor is built.
    fn source_fragments_for_range(&self, range: ColumnRange) -> Vec<(usize, ColumnRange)> {
        let mut fragments = Vec::new();
        // Some value-only tests and host adapters provide an explicit owner
        // inventory without retaining layout intervals.  That inventory is
        // still authoritative; it is not a fallback to the destination job
        // device.  Production GPU layouts always take the interval path
        // below, which splits crossing ranges precisely.
        if self.source_layouts.is_empty() {
            let mut owners = self.source_owners.clone();
            owners.sort_unstable();
            owners.dedup();
            return owners.into_iter().map(|owner| (owner, range)).collect();
        }
        for layout in &self.source_layouts {
            for (device, interval_start, interval_end) in &layout.owner_intervals {
                let overlap_start = (*interval_start).max(self.source_range.start);
                let overlap_start = overlap_start.max(range.start);
                let overlap_end = (*interval_end).min(self.source_range.end).min(range.end);
                if overlap_start < overlap_end {
                    let fragment =
                        (*device, ColumnRange { start: overlap_start, end: overlap_end });
                    if !fragments.contains(&fragment) {
                        fragments.push(fragment);
                    }
                }
            }
        }
        fragments.sort_unstable_by_key(|(owner, range)| (range.start, range.end, *owner));
        fragments
    }
}

fn range_is_fully_covered(range: ColumnRange, fragments: &[(usize, ColumnRange)]) -> bool {
    if range.is_empty() || fragments.is_empty() {
        return false;
    }
    let mut cursor = range.start;
    for (_, fragment) in fragments {
        if fragment.end <= cursor {
            continue;
        }
        if fragment.start > cursor {
            return false;
        }
        cursor = cursor.max(fragment.end);
        if cursor >= range.end {
            return true;
        }
    }
    cursor >= range.end
}

impl GpuWarmupOperationDescriptor {
    /// Resolve the closed profile domain from the validated IR operation.
    /// Providers should use this instead of matching debug strings or
    /// silently assigning a generic profile.
    pub fn canonical_profile_domain(&self) -> CanonicalWarmupProfileDomain {
        self.profile_domain
    }

    pub fn measurement_kind(&self) -> WarmupMeasurementKind {
        self.canonical_profile_domain().measurement_kind()
    }

    pub fn transfer_kind(&self) -> crate::gpu_column_policy::WarmupTransferKind {
        self.canonical_profile_domain().transfer_kind()
    }
}

/// Profile lifecycle state.  Every admitted profile is timed; the enum is
/// retained as an explicit discriminator for serialized planner diagnostics,
/// but there is deliberately no size-only escape hatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GpuWarmupProfileKind {
    Measured,
}

/// Time and workspace for one operation signature and tile-width candidate.
/// Every admitted profile carries a finite measured elapsed time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuWarmupProfile {
    pub kind: GpuWarmupProfileKind,
    /// Distinguishes a timed GPU-kernel profile from a timed host profile.
    /// This is separate from `kind` to preserve the warmup planner's existing
    /// compatibility match while making the provider contract explicit.
    pub measurement: WarmupMeasurementKind,
    /// Required elapsed time for this measured candidate.
    pub time_seconds: f64,
    pub workspace_bytes: u64,
    /// Optional sampler metadata carried by a preimage provider. These are
    /// absent for ordinary operations and must never be inferred by the
    /// planner from a generic workspace measurement.
    pub preimage_max_attempts: Option<usize>,
    pub preimage_certified_tile_width: Option<usize>,
    /// Width-specific native preimage allocation evidence.  This is copied
    /// through the setup provider so the planner can perform hard admission
    /// on the primitive envelope instead of treating it as generic scratch.
    pub preimage_footprint: Option<crate::gpu_warmup::GpuPreimageFootprint>,
    /// Native sampler ownership resolved by the provider.  This is opaque to
    /// runtime (only the digest crosses the warmup boundary) and is absent
    /// only for non-preimage operations.
    pub resolved_cache_identity: Option<[u8; 32]>,
    /// Complete memory evidence observed for this point.  This is kept on
    /// the provider result as well as on the canonical point table so a
    /// provider cannot accidentally discard the evidence kind while
    /// translating a measurement into planner metadata.
    pub memory: GpuWarmupMemoryObservations,
    /// Incremental residency changes caused by the measured production
    /// invocation.  These are signed because a stage may release a retained
    /// buffer while producing its result.
    pub resident_delta: GpuWarmupResidencyDelta,
    /// Number of timed production repetitions represented by
    /// `time_seconds`.  One is a valid measurement; the warmup policy may
    /// request more, but the data model must not invent a minimum.
    pub repetitions: usize,
    /// Timing spread (for example max-min or standard deviation) in seconds.
    /// A single repetition has no spread and is represented by zero.
    pub spread_seconds: f64,
    pub provenance: GpuWarmupProvenance,
    /// Preserve cache and timing scope on the point itself.  They are also
    /// part of `GpuWarmupProfileKey`; duplicating them here makes it possible
    /// to validate provider output before it enters the canonical table.
    pub cache_state: GpuWarmupCacheState,
    pub timing_scope: GpuWarmupTimingScope,
    /// Physical route returned by the inclusive production job.  The request
    /// route is only a candidate; after a provider has executed the real
    /// materializer this value is authoritative for the canonical session
    /// key.  Host-only profiles leave it unset.
    pub resolved_route_descriptor: Option<GpuExecutionRouteDescriptor>,
}

/// A typed implementation identity for a measured effective operation.  The
/// operation domain and this variant are deliberately separate: a fused row
/// sum and an ordinary matrix operation must not share a profile merely
/// because they happen to have the same IR node kind.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuWarmupEffectiveVariant {
    Ordinary,
    Fused(FusedWarmupOperation),
    Host,
    Transfer,
    Custom(String),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuWarmupCacheState {
    Cold,
    Warm,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GpuWarmupDeviceIdentity {
    pub logical_device: usize,
    pub device_identity: String,
    pub native_revision: String,
    /// Native CUDA context generation.  Recreating a context invalidates
    /// allocator and event evidence even when the physical device is the same.
    pub context_generation: u64,
}

impl GpuWarmupDeviceIdentity {
    pub fn new(
        logical_device: usize,
        device_identity: impl Into<String>,
        native_revision: impl Into<String>,
    ) -> Self {
        Self {
            logical_device,
            device_identity: device_identity.into(),
            native_revision: native_revision.into(),
            context_generation: 0,
        }
    }

    pub fn with_context_generation(mut self, context_generation: u64) -> Self {
        self.context_generation = context_generation;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuWarmupRoute {
    DeviceLocal,
    HostStaging,
    PeerToPeer,
    HostOnly,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuWarmupFragmentClass {
    Whole,
    Tail,
    Mapped,
    Fragmented,
    SingleDevice,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuWarmupTimingScope {
    LocalJob,
    Setup,
    ContainingStage,
    Transfer,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum GpuWarmupProvenance {
    ProductionEquivalent,
    ImportedValidated,
    ExactAccounting,
    CoveredByContainingStage,
}

/// Memory evidence that is safe to use for a hard resource-admission check.
/// Measured peaks and interpolated estimates remain useful for diagnostics and
/// ranking, but cannot certify that a frozen plan fits on a device.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum MemoryEvidenceKind {
    /// No memory observation was supplied.  This is deliberately not
    /// admissible and is the default for callers that have not measured a
    /// candidate yet.
    Unspecified,
    ExactQuery,
    CertifiedEnvelope,
    MeasuredPeak,
    LinearEstimate,
}

impl MemoryEvidenceKind {
    pub const fn is_hard_admission(self) -> bool {
        matches!(self, Self::ExactQuery | Self::CertifiedEnvelope)
    }
}

/// The non-interpolated part of a performance execution class.  Width is the
/// only coordinate a table may interpolate; every other shape/native value is
/// kept in this class and therefore has to match exactly.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GpuWarmupProfileKey {
    pub effective_domain: CanonicalWarmupProfileDomain,
    pub implementation_variant: GpuWarmupEffectiveVariant,
    pub operation_identity: [u8; 32],
    pub noninterpolated_shape: Vec<usize>,
    pub native_parameters: Vec<u64>,
    pub device: GpuWarmupDeviceIdentity,
    /// Actual range start/class are part of the non-interpolated execution
    /// identity.  Width is the only value that may be interpolated.
    pub executed_range_start: usize,
    pub executed_range_class: GpuWarmupFragmentClass,
    pub retry_cap: Option<usize>,
    pub cache_identity: Option<[u8; 32]>,
    pub cache_state: GpuWarmupCacheState,
    pub route: GpuWarmupRoute,
    pub route_descriptor: GpuExecutionRouteDescriptor,
    /// Fused union binding port. Distinct output ports can share all range
    /// and route coordinates while requiring different output geometry.
    #[serde(default)]
    pub binding_port: Option<usize>,
    pub fragment: GpuWarmupFragmentClass,
    pub timing_scope: GpuWarmupTimingScope,
}

impl GpuWarmupProfileKey {
    pub fn execution_class_matches(&self, other: &Self) -> bool {
        self.effective_domain == other.effective_domain &&
            self.implementation_variant == other.implementation_variant &&
            self.operation_identity == other.operation_identity &&
            self.noninterpolated_shape == other.noninterpolated_shape &&
            self.native_parameters == other.native_parameters &&
            self.device == other.device &&
            self.executed_range_start == other.executed_range_start &&
            self.executed_range_class == other.executed_range_class &&
            self.retry_cap == other.retry_cap &&
            self.cache_identity == other.cache_identity &&
            self.cache_state == other.cache_state &&
            self.route == other.route &&
            self.route_descriptor == other.route_descriptor &&
            self.binding_port == other.binding_port &&
            self.fragment == other.fragment &&
            self.timing_scope == other.timing_scope
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuWarmupMemoryObservations {
    pub affected_devices: BTreeMap<GpuWarmupDeviceIdentity, u64>,
    pub host_bytes: u64,
    pub pinned_host_bytes: u64,
    pub evidence: MemoryEvidenceKind,
}

impl Default for GpuWarmupMemoryObservations {
    fn default() -> Self {
        Self {
            affected_devices: BTreeMap::new(),
            host_bytes: 0,
            pinned_host_bytes: 0,
            evidence: MemoryEvidenceKind::Unspecified,
        }
    }
}

impl GpuWarmupMemoryObservations {
    /// Explicit zero-byte evidence for a host-only/no-work operation.  An
    /// empty device map with exact evidence is otherwise ambiguous and is
    /// rejected for GPU measurements by [`GpuWarmupProfile::validate`].
    pub fn explicit_exact_zero() -> Self {
        Self {
            affected_devices: BTreeMap::new(),
            host_bytes: 0,
            pinned_host_bytes: 0,
            evidence: MemoryEvidenceKind::ExactQuery,
        }
    }

    fn interpolate(
        lower: &Self,
        upper: &Self,
        coordinate: usize,
        lower_coordinate: usize,
        upper_coordinate: usize,
    ) -> Result<Self, GpuWarmupProfileTableError> {
        if lower.affected_devices.keys().ne(upper.affected_devices.keys()) {
            return Err(GpuWarmupProfileTableError::IncompatibleInterval);
        }
        let interpolate_bytes = |a: u64, b: u64| {
            interpolate_u64_ceil(a, b, coordinate, lower_coordinate, upper_coordinate)
        };
        let affected_devices = lower
            .affected_devices
            .iter()
            .map(|(device, bytes)| {
                Ok((device.clone(), interpolate_bytes(*bytes, upper.affected_devices[device])?))
            })
            .collect::<Result<BTreeMap<_, _>, GpuWarmupProfileTableError>>()?;
        Ok(Self {
            affected_devices,
            host_bytes: interpolate_bytes(lower.host_bytes, upper.host_bytes)?,
            pinned_host_bytes: interpolate_bytes(lower.pinned_host_bytes, upper.pinned_host_bytes)?,
            evidence: MemoryEvidenceKind::LinearEstimate,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuWarmupResidencyDelta {
    pub affected_devices: BTreeMap<GpuWarmupDeviceIdentity, i64>,
    pub host_bytes: i64,
    pub pinned_host_bytes: i64,
}

impl Default for GpuWarmupResidencyDelta {
    fn default() -> Self {
        Self { affected_devices: BTreeMap::new(), host_bytes: 0, pinned_host_bytes: 0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuWarmupProfilePoint {
    pub key: GpuWarmupProfileKey,
    /// The actual positive local job width (or transfer byte count).
    pub coordinate: usize,
    pub executed_range: IndexRange,
    pub executed_range_class: GpuWarmupFragmentClass,
    pub repetitions: usize,
    pub mean_seconds: f64,
    pub spread_seconds: f64,
    pub memory: GpuWarmupMemoryObservations,
    pub resident_delta: GpuWarmupResidencyDelta,
    pub provenance: GpuWarmupProvenance,
    /// Details needed to reconstruct the provider result from the canonical
    /// session table.  Keeping these on the point avoids a second
    /// request-to-profile cache while preserving width-specific workspace and
    /// sampler admission metadata on a cache hit.
    pub workspace_bytes: u64,
    pub preimage_max_attempts: Option<usize>,
    pub preimage_certified_tile_width: Option<usize>,
    pub preimage_footprint: Option<crate::gpu_warmup::GpuPreimageFootprint>,
    pub resolved_cache_identity: Option<[u8; 32]>,
    /// Full physical route observed at this coordinate. The table key strips
    /// range ends and staging byte counts to define an interpolation class.
    pub resolved_route_descriptor: Option<GpuExecutionRouteDescriptor>,
}

impl GpuWarmupProfilePoint {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key: GpuWarmupProfileKey,
        coordinate: usize,
        executed_range: IndexRange,
        executed_range_class: GpuWarmupFragmentClass,
        repetitions: usize,
        mean_seconds: f64,
        spread_seconds: f64,
        memory: GpuWarmupMemoryObservations,
        resident_delta: GpuWarmupResidencyDelta,
        provenance: GpuWarmupProvenance,
    ) -> Result<Self, GpuWarmupProfileTableError> {
        if coordinate == 0 ||
            executed_range.start >= executed_range.end ||
            executed_range.start != key.executed_range_start ||
            executed_range_class != key.executed_range_class ||
            repetitions == 0
        {
            return Err(GpuWarmupProfileTableError::InvalidPoint(
                "coordinate, range, and repetitions must be positive".into(),
            ));
        }
        if !mean_seconds.is_finite() ||
            mean_seconds <= 0.0 ||
            !spread_seconds.is_finite() ||
            spread_seconds < 0.0
        {
            return Err(GpuWarmupProfileTableError::InvalidPoint(
                "point timing must be finite, positive, and have non-negative spread".into(),
            ));
        }
        Ok(Self {
            key,
            coordinate,
            executed_range,
            executed_range_class,
            repetitions,
            mean_seconds,
            spread_seconds,
            memory,
            resident_delta,
            provenance,
            workspace_bytes: 0,
            preimage_max_attempts: None,
            preimage_certified_tile_width: None,
            preimage_footprint: None,
            resolved_cache_identity: None,
            resolved_route_descriptor: None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuWarmupValidationTolerance {
    pub absolute_seconds: f64,
    pub relative: f64,
}

pub type GpuWarmupIntervalTolerance = GpuWarmupValidationTolerance;

impl Default for GpuWarmupValidationTolerance {
    fn default() -> Self {
        Self { absolute_seconds: 0.0, relative: 0.05 }
    }
}

impl GpuWarmupValidationTolerance {
    pub fn new(absolute_seconds: f64, relative: f64) -> Result<Self, GpuWarmupProfileTableError> {
        let tolerance = Self { absolute_seconds, relative };
        if !absolute_seconds.is_finite() ||
            absolute_seconds < 0.0 ||
            !relative.is_finite() ||
            relative < 0.0
        {
            return Err(GpuWarmupProfileTableError::InvalidTolerance);
        }
        Ok(tolerance)
    }

    fn allows(
        self,
        predicted: f64,
        observed: f64,
    ) -> Result<(f64, f64), GpuWarmupProfileTableError> {
        if !predicted.is_finite() || !observed.is_finite() || predicted < 0.0 || observed < 0.0 {
            return Err(GpuWarmupProfileTableError::InvalidResolution);
        }
        let error = (observed - predicted).abs();
        let allowed = self.absolute_seconds + self.relative * predicted.abs().max(observed.abs());
        if !allowed.is_finite() {
            return Err(GpuWarmupProfileTableError::InvalidTolerance);
        }
        Ok((error, allowed))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuWarmupValidatedInterval {
    pub lower_coordinate: usize,
    pub upper_coordinate: usize,
    pub holdout_coordinates: BTreeSet<usize>,
    pub tolerance: GpuWarmupValidationTolerance,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuWarmupIntervalValidation {
    Validated(GpuWarmupValidatedInterval),
    /// The measured holdout remains in `points`, but the interval was not
    /// activated.  The caller may promote it to an active knot and measure a
    /// replacement anchor/segment without losing the original observation.
    HoldoutOutOfTolerance {
        coordinate: usize,
        predicted_seconds: f64,
        observed_seconds: f64,
        absolute_error_seconds: f64,
        allowed_error_seconds: f64,
    },
}

impl GpuWarmupIntervalValidation {
    pub fn is_validated(&self) -> bool {
        matches!(self, Self::Validated(_))
    }

    pub fn holdout_coordinate_to_promote(&self) -> Option<usize> {
        match self {
            Self::HoldoutOutOfTolerance { coordinate, .. } => Some(*coordinate),
            Self::Validated(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuWarmupProfileTableError {
    InvalidPoint(String),
    DuplicateCoordinate,
    IncompatibleKey,
    MissingAnchor,
    InvalidInterval,
    IncompatibleInterval,
    NoValidatedInterval,
    NonAdmissibleMemory(MemoryEvidenceKind),
    InvalidResolution,
    InvalidTolerance,
    ArithmeticOverflow,
}

impl fmt::Display for GpuWarmupProfileTableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPoint(message) => formatter.write_str(message),
            Self::DuplicateCoordinate => formatter.write_str("duplicate profile coordinate"),
            Self::IncompatibleKey => {
                formatter.write_str("profile point belongs to another execution class")
            }
            Self::MissingAnchor => {
                formatter.write_str("profile interval references a missing anchor")
            }
            Self::InvalidInterval => formatter.write_str("invalid profile interpolation interval"),
            Self::IncompatibleInterval => {
                formatter.write_str("profile interval has incompatible observations")
            }
            Self::NoValidatedInterval => {
                formatter.write_str("no validated interpolation interval contains coordinate")
            }
            Self::NonAdmissibleMemory(kind) => {
                write!(formatter, "memory evidence {kind:?} cannot certify admission")
            }
            Self::InvalidResolution => {
                formatter.write_str("resolved profile is not finite and positive")
            }
            Self::InvalidTolerance => {
                formatter.write_str("interval validation tolerance must be finite and non-negative")
            }
            Self::ArithmeticOverflow => formatter.write_str("profile arithmetic overflow"),
        }
    }
}

impl std::error::Error for GpuWarmupProfileTableError {}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuWarmupResolvedPoint {
    pub coordinate: usize,
    pub time_seconds: f64,
    pub memory: GpuWarmupMemoryObservations,
    pub interpolated: bool,
}

/// Ordered, setup-only points for one effective execution class.  Intervals
/// are explicitly validated by the collector; the resolver never extrapolates
/// or scales a single anchor.
#[derive(Clone, Debug)]
pub struct GpuWarmupProfileTable {
    key: GpuWarmupProfileKey,
    points: BTreeMap<usize, GpuWarmupProfilePoint>,
    active_knots: BTreeSet<usize>,
    intervals: BTreeMap<(usize, usize), GpuWarmupValidatedInterval>,
}

impl GpuWarmupProfileTable {
    pub fn new(key: GpuWarmupProfileKey) -> Self {
        Self {
            key,
            points: BTreeMap::new(),
            active_knots: BTreeSet::new(),
            intervals: BTreeMap::new(),
        }
    }

    pub fn key(&self) -> &GpuWarmupProfileKey {
        &self.key
    }

    pub fn points(&self) -> &BTreeMap<usize, GpuWarmupProfilePoint> {
        &self.points
    }

    pub fn active_knots(&self) -> &BTreeSet<usize> {
        &self.active_knots
    }

    pub fn intervals(&self) -> &BTreeMap<(usize, usize), GpuWarmupValidatedInterval> {
        &self.intervals
    }

    pub fn insert_point(
        &mut self,
        point: GpuWarmupProfilePoint,
    ) -> Result<(), GpuWarmupProfileTableError> {
        if !point.key.execution_class_matches(&self.key) {
            return Err(GpuWarmupProfileTableError::IncompatibleKey);
        }
        if point.executed_range.start != point.key.executed_range_start ||
            point.executed_range_class != point.key.executed_range_class
        {
            return Err(GpuWarmupProfileTableError::IncompatibleKey);
        }
        if self.points.contains_key(&point.coordinate) {
            return Err(GpuWarmupProfileTableError::DuplicateCoordinate);
        }
        self.points.insert(point.coordinate, point);
        Ok(())
    }

    pub fn validate_interval(
        &mut self,
        lower_coordinate: usize,
        upper_coordinate: usize,
        holdout_coordinates: impl IntoIterator<Item = usize>,
    ) -> Result<(), GpuWarmupProfileTableError> {
        match self.validate_interval_with_policy(
            lower_coordinate,
            upper_coordinate,
            holdout_coordinates,
            GpuWarmupValidationTolerance::default(),
        )? {
            GpuWarmupIntervalValidation::Validated(_) => Ok(()),
            GpuWarmupIntervalValidation::HoldoutOutOfTolerance { .. } => {
                Err(GpuWarmupProfileTableError::InvalidInterval)
            }
        }
    }

    /// Validate an affine segment against the actual measured holdout points.
    /// The points are already owned by this table; only their coordinates are
    /// passed here so an interval cannot accidentally validate a second copy
    /// of a measurement.  A failed holdout is returned as a typed result and
    /// does not activate the interval.
    pub fn validate_interval_with_policy(
        &mut self,
        lower_coordinate: usize,
        upper_coordinate: usize,
        holdout_coordinates: impl IntoIterator<Item = usize>,
        tolerance: GpuWarmupValidationTolerance,
    ) -> Result<GpuWarmupIntervalValidation, GpuWarmupProfileTableError> {
        // Validate the policy even when the interval itself is malformed so
        // callers cannot accidentally proceed with a NaN/negative policy.
        GpuWarmupValidationTolerance::new(tolerance.absolute_seconds, tolerance.relative)?;
        if lower_coordinate == 0 ||
            lower_coordinate >= upper_coordinate ||
            !self.points.contains_key(&lower_coordinate) ||
            !self.points.contains_key(&upper_coordinate)
        {
            return Err(GpuWarmupProfileTableError::InvalidInterval);
        }
        let holdout_coordinates = holdout_coordinates.into_iter().collect::<BTreeSet<_>>();
        if holdout_coordinates.iter().any(|coordinate| {
            *coordinate <= lower_coordinate ||
                *coordinate >= upper_coordinate ||
                !self.points.contains_key(coordinate)
        }) {
            return Err(GpuWarmupProfileTableError::MissingAnchor);
        }

        let lower =
            self.points.get(&lower_coordinate).ok_or(GpuWarmupProfileTableError::MissingAnchor)?;
        let upper =
            self.points.get(&upper_coordinate).ok_or(GpuWarmupProfileTableError::MissingAnchor)?;
        if lower.key != upper.key {
            return Err(GpuWarmupProfileTableError::IncompatibleInterval);
        }
        for coordinate in &holdout_coordinates {
            let holdout =
                self.points.get(coordinate).ok_or(GpuWarmupProfileTableError::MissingAnchor)?;
            let predicted = interpolate_f64(
                lower.mean_seconds,
                upper.mean_seconds,
                *coordinate,
                lower_coordinate,
                upper_coordinate,
            )?;
            let (absolute_error_seconds, allowed_error_seconds) =
                tolerance.allows(predicted, holdout.mean_seconds)?;
            if absolute_error_seconds > allowed_error_seconds {
                return Ok(GpuWarmupIntervalValidation::HoldoutOutOfTolerance {
                    coordinate: *coordinate,
                    predicted_seconds: predicted,
                    observed_seconds: holdout.mean_seconds,
                    absolute_error_seconds,
                    allowed_error_seconds,
                });
            }
        }
        self.active_knots.insert(lower_coordinate);
        self.active_knots.insert(upper_coordinate);
        let interval = GpuWarmupValidatedInterval {
            lower_coordinate,
            upper_coordinate,
            holdout_coordinates,
            tolerance,
        };
        self.intervals.insert((lower_coordinate, upper_coordinate), interval.clone());
        Ok(GpuWarmupIntervalValidation::Validated(interval))
    }

    /// Promote a retained measured holdout to an active knot after a failed
    /// segment check.  This keeps subdivision explicit and never installs an
    /// unvalidated segment implicitly.
    pub fn promote_holdout(&mut self, coordinate: usize) -> Result<(), GpuWarmupProfileTableError> {
        if coordinate == 0 || !self.points.contains_key(&coordinate) {
            return Err(GpuWarmupProfileTableError::MissingAnchor);
        }
        self.active_knots.insert(coordinate);
        Ok(())
    }

    pub fn resolve(
        &self,
        coordinate: usize,
    ) -> Result<GpuWarmupResolvedPoint, GpuWarmupProfileTableError> {
        if let Some(point) = self.points.get(&coordinate) {
            return Self::resolved(coordinate, point.mean_seconds, point.memory.clone(), false);
        }
        let (_, interval) = self
            .intervals
            .iter()
            .find(|((lower, upper), _)| *lower < coordinate && coordinate < *upper)
            .ok_or(GpuWarmupProfileTableError::NoValidatedInterval)?;
        let lower = self
            .points
            .get(&interval.lower_coordinate)
            .ok_or(GpuWarmupProfileTableError::MissingAnchor)?;
        let upper = self
            .points
            .get(&interval.upper_coordinate)
            .ok_or(GpuWarmupProfileTableError::MissingAnchor)?;
        if lower.key != upper.key {
            return Err(GpuWarmupProfileTableError::IncompatibleInterval);
        }
        let time_seconds = interpolate_f64(
            lower.mean_seconds,
            upper.mean_seconds,
            coordinate,
            lower.coordinate,
            upper.coordinate,
        )?;
        let memory = GpuWarmupMemoryObservations::interpolate(
            &lower.memory,
            &upper.memory,
            coordinate,
            lower.coordinate,
            upper.coordinate,
        )?;
        Self::resolved(coordinate, time_seconds, memory, true)
    }

    pub fn resolve_for_admission(
        &self,
        coordinate: usize,
    ) -> Result<GpuWarmupResolvedPoint, GpuWarmupProfileTableError> {
        let resolved = self.resolve(coordinate)?;
        if !resolved.memory.evidence.is_hard_admission() {
            return Err(GpuWarmupProfileTableError::NonAdmissibleMemory(resolved.memory.evidence));
        }
        Ok(resolved)
    }

    fn resolved(
        coordinate: usize,
        time_seconds: f64,
        memory: GpuWarmupMemoryObservations,
        interpolated: bool,
    ) -> Result<GpuWarmupResolvedPoint, GpuWarmupProfileTableError> {
        if coordinate == 0 || !time_seconds.is_finite() || time_seconds <= 0.0 {
            return Err(GpuWarmupProfileTableError::InvalidResolution);
        }
        Ok(GpuWarmupResolvedPoint { coordinate, time_seconds, memory, interpolated })
    }
}

fn interpolate_f64(
    lower: f64,
    upper: f64,
    coordinate: usize,
    lower_coordinate: usize,
    upper_coordinate: usize,
) -> Result<f64, GpuWarmupProfileTableError> {
    if !lower.is_finite() || !upper.is_finite() || lower_coordinate >= upper_coordinate {
        return Err(GpuWarmupProfileTableError::InvalidResolution);
    }
    let fraction =
        (coordinate - lower_coordinate) as f64 / (upper_coordinate - lower_coordinate) as f64;
    let value = lower + (upper - lower) * fraction;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(GpuWarmupProfileTableError::InvalidResolution)
    }
}

fn interpolate_u64_ceil(
    lower: u64,
    upper: u64,
    coordinate: usize,
    lower_coordinate: usize,
    upper_coordinate: usize,
) -> Result<u64, GpuWarmupProfileTableError> {
    if lower_coordinate >= upper_coordinate ||
        coordinate < lower_coordinate ||
        coordinate > upper_coordinate
    {
        return Err(GpuWarmupProfileTableError::InvalidResolution);
    }
    let numerator = (lower as u128)
        .checked_mul((upper_coordinate - coordinate) as u128)
        .and_then(|value| {
            value.checked_add((upper as u128).checked_mul((coordinate - lower_coordinate) as u128)?)
        })
        .ok_or(GpuWarmupProfileTableError::ArithmeticOverflow)?;
    let denominator = (upper_coordinate - lower_coordinate) as u128;
    let rounded = numerator
        .checked_add(denominator - 1)
        .ok_or(GpuWarmupProfileTableError::ArithmeticOverflow)? /
        denominator;
    u64::try_from(rounded).map_err(|_| GpuWarmupProfileTableError::ArithmeticOverflow)
}

/// Session-wide canonical cache.  It owns profile tables and has no provider
/// or lookup-only secondary map, so one setup session has one authority for
/// every measured point and validated interval.
#[derive(Clone, Debug, Default)]
pub struct GpuWarmupSessionProfileCache {
    tables: BTreeMap<GpuWarmupProfileKey, GpuWarmupProfileTable>,
}

/// Short names used by planner/collector integrations.  They are aliases of
/// the canonical runtime types above, not a second profile hierarchy.
pub type ProfileKey = GpuWarmupProfileKey;
pub type ProfilePoint = GpuWarmupProfilePoint;
pub type ProfileTable = GpuWarmupProfileTable;
pub type SessionProfileCache = GpuWarmupSessionProfileCache;

impl GpuWarmupSessionProfileCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_point(
        &mut self,
        point: GpuWarmupProfilePoint,
    ) -> Result<(), GpuWarmupProfileTableError> {
        let key = point.key.clone();
        if let Some(table) = self.tables.get_mut(&key) {
            return table.insert_point(point);
        }
        // Stage the first point before publishing a new table.  This makes a
        // failed point/context contract atomic: no empty or partially
        // initialized table is left behind for a later lookup to mistake for
        // authoritative data.
        let mut table = GpuWarmupProfileTable::new(key.clone());
        table.insert_point(point)?;
        self.tables.insert(key, table);
        Ok(())
    }

    /// Transactionally import a batch of provider points.  A duplicate
    /// coordinate or incompatible execution context rejects the complete
    /// batch, leaving this cache unchanged.
    pub fn insert_points(
        &mut self,
        points: impl IntoIterator<Item = GpuWarmupProfilePoint>,
    ) -> Result<(), GpuWarmupProfileTableError> {
        let mut staged = self.clone();
        for point in points {
            staged.insert_point(point)?;
        }
        *self = staged;
        Ok(())
    }

    /// Insert a provider result directly into the canonical table.  This is
    /// the preferred boundary for providers that return one point at a time;
    /// no provider-owned profile map is consulted or merged implicitly.
    pub fn insert_profile(
        &mut self,
        key: GpuWarmupProfileKey,
        coordinate: usize,
        executed_range: IndexRange,
        profile: GpuWarmupProfile,
    ) -> Result<(), GpuWarmupProfileError> {
        let point = profile.into_point(key, coordinate, executed_range)?;
        self.insert_point(point)
            .map_err(|error| GpuWarmupProfileError::InvalidMeasurement(error.to_string()))
    }

    pub fn register_table(
        &mut self,
        table: GpuWarmupProfileTable,
    ) -> Result<(), GpuWarmupProfileTableError> {
        let key = table.key.clone();
        if self.tables.contains_key(&key) {
            return Err(GpuWarmupProfileTableError::IncompatibleKey);
        }
        self.tables.insert(key, table);
        Ok(())
    }

    /// Register a complete table set atomically.  The session cache remains
    /// the only authoritative point store; providers return points/tables but
    /// do not retain a second conflicting registry here.
    pub fn register_tables(
        &mut self,
        tables: impl IntoIterator<Item = GpuWarmupProfileTable>,
    ) -> Result<(), GpuWarmupProfileTableError> {
        let mut staged = self.clone();
        for table in tables {
            staged.register_table(table)?;
        }
        *self = staged;
        Ok(())
    }

    pub fn tables(&self) -> &BTreeMap<GpuWarmupProfileKey, GpuWarmupProfileTable> {
        &self.tables
    }

    pub fn table(&self, key: &GpuWarmupProfileKey) -> Option<&GpuWarmupProfileTable> {
        self.tables.get(key)
    }

    pub fn table_mut(&mut self, key: &GpuWarmupProfileKey) -> Option<&mut GpuWarmupProfileTable> {
        self.tables.get_mut(key)
    }

    /// Return the native sampler identity resolved by an earlier cold point
    /// in this setup session.  The lookup is deliberately keyed by the full
    /// operation signature and physical device; shape-only or semantic-hash
    /// matches are not allowed to manufacture a warm cache owner.
    pub fn preimage_cache_identity(
        &self,
        signature: GpuWarmupOperationSignature,
        device: &GpuWarmupDeviceIdentity,
    ) -> Result<Option<[u8; 32]>, GpuWarmupProfileError> {
        let mut identity = None;
        for key in self.tables.keys().filter(|key| {
            matches!(
                key.effective_domain,
                CanonicalWarmupProfileDomain::PreimageSample |
                    CanonicalWarmupProfileDomain::FusedPreimageBatch
            ) && key.operation_identity == signature.operation &&
                key.native_parameters
                    .starts_with(&[signature.shape_class, signature.instance_class]) &&
                key.device == *device
        }) {
            let Some(candidate) = key.cache_identity else { continue };
            if let Some(previous) = identity {
                if previous != candidate {
                    return Err(GpuWarmupProfileError::InvalidMeasurement(
                        "preimage setup resolved multiple native cache owners for one operation/device"
                            .into(),
                    ));
                }
            } else {
                identity = Some(candidate);
            }
        }
        Ok(identity)
    }

    /// Return an exact provider result already owned by this setup session.
    /// This is intentionally an exact-point lookup; interpolation remains a
    /// planner/table concern and never causes a provider call to be skipped
    /// for a new measurement coordinate.
    pub fn exact_profile(
        &self,
        key: &GpuWarmupProfileKey,
        coordinate: usize,
    ) -> Result<Option<GpuWarmupProfile>, GpuWarmupProfileError> {
        let Some(point) = self.tables.get(key).and_then(|table| table.points().get(&coordinate))
        else {
            return Ok(None);
        };
        // Do not trust the table coordinate alone.  A stale point imported
        // under a colliding key must never be returned for another physical
        // range or fragment class.
        if !point.key.execution_class_matches(key) ||
            point.executed_range.start != key.executed_range_start ||
            point.executed_range_class != key.executed_range_class
        {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "cached profile executed range does not match its execution key".into(),
            ));
        }
        let mut profile = GpuWarmupProfile::measured_with_observation(
            point.mean_seconds,
            point.workspace_bytes,
            key.effective_domain.measurement_kind(),
            point.memory.clone(),
            point.resident_delta.clone(),
            point.repetitions,
            point.spread_seconds,
            point.provenance,
            key.cache_state,
            key.timing_scope,
        )?;
        profile.preimage_max_attempts = point.preimage_max_attempts;
        profile.preimage_certified_tile_width = point.preimage_certified_tile_width;
        profile.preimage_footprint = point.preimage_footprint.clone();
        profile.resolved_cache_identity = point.resolved_cache_identity;
        // A cache hit must retain the same authoritative physical route as
        // the measured response. Returning an unresolved route here creates
        // a second, incompatible job key on repeated collection.
        profile.resolved_route_descriptor = point.resolved_route_descriptor;
        Ok(Some(profile))
    }

    /// Exact lookup with the complete request context.  Keeping this check at
    /// the cache boundary prevents callers from accidentally treating a
    /// matching operation hash/coordinate as equivalent when route, fragment,
    /// physical device, retry cap, or retained-cache identity changed.
    pub fn exact_profile_for_request(
        &self,
        key: &GpuWarmupProfileKey,
        request: &GpuWarmupProfileRequest,
    ) -> Result<Option<GpuWarmupProfile>, GpuWarmupProfileError> {
        let preimage_profile = matches!(
            key.effective_domain,
            CanonicalWarmupProfileDomain::PreimageSample |
                CanonicalWarmupProfileDomain::FusedPreimageBatch
        );
        if preimage_profile && request.cache_identity.is_none() {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "preimage profile lookup requires native cache identity".into(),
            ));
        }
        if request.coordinate() == 0 ||
            request.range.start != request.executed_range_start ||
            request.device_identity.logical_device != request.device ||
            key.operation_identity != request.signature.operation ||
            key.device != request.device_identity ||
            key.executed_range_start != request.executed_range_start ||
            key.executed_range_class != request.executed_range_class ||
            key.route != request.route ||
            key.route_descriptor != request.route_descriptor ||
            key.binding_port != request.binding_port ||
            key.fragment != request.fragment ||
            key.retry_cap != request.retry_cap ||
            key.cache_identity != request.cache_identity ||
            key.cache_state != request.cache_state ||
            key.timing_scope != request.timing_scope
        {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "warmup profile request context does not match its cache key".into(),
            ));
        }
        self.exact_profile(key, request.coordinate())
    }

    pub fn validate_interval(
        &mut self,
        key: &GpuWarmupProfileKey,
        lower_coordinate: usize,
        upper_coordinate: usize,
        holdout_coordinates: impl IntoIterator<Item = usize>,
    ) -> Result<(), GpuWarmupProfileTableError> {
        self.tables
            .get_mut(key)
            .ok_or(GpuWarmupProfileTableError::MissingAnchor)?
            .validate_interval(lower_coordinate, upper_coordinate, holdout_coordinates)
    }

    pub fn validate_interval_with_policy(
        &mut self,
        key: &GpuWarmupProfileKey,
        lower_coordinate: usize,
        upper_coordinate: usize,
        holdout_coordinates: impl IntoIterator<Item = usize>,
        tolerance: GpuWarmupValidationTolerance,
    ) -> Result<GpuWarmupIntervalValidation, GpuWarmupProfileTableError> {
        self.tables
            .get_mut(key)
            .ok_or(GpuWarmupProfileTableError::MissingAnchor)?
            .validate_interval_with_policy(
                lower_coordinate,
                upper_coordinate,
                holdout_coordinates,
                tolerance,
            )
    }

    pub fn promote_holdout(
        &mut self,
        key: &GpuWarmupProfileKey,
        coordinate: usize,
    ) -> Result<(), GpuWarmupProfileTableError> {
        self.tables
            .get_mut(key)
            .ok_or(GpuWarmupProfileTableError::MissingAnchor)?
            .promote_holdout(coordinate)
    }

    pub fn resolve(
        &self,
        key: &GpuWarmupProfileKey,
        coordinate: usize,
    ) -> Result<GpuWarmupResolvedPoint, GpuWarmupProfileTableError> {
        self.tables.get(key).ok_or(GpuWarmupProfileTableError::MissingAnchor)?.resolve(coordinate)
    }

    pub fn resolve_for_admission(
        &self,
        key: &GpuWarmupProfileKey,
        coordinate: usize,
    ) -> Result<GpuWarmupResolvedPoint, GpuWarmupProfileTableError> {
        self.tables
            .get(key)
            .ok_or(GpuWarmupProfileTableError::MissingAnchor)?
            .resolve_for_admission(coordinate)
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

#[derive(Clone, Debug)]
struct GpuWarmupProfileObservation {
    memory: GpuWarmupMemoryObservations,
    resident_delta: GpuWarmupResidencyDelta,
    repetitions: usize,
    spread_seconds: f64,
    provenance: GpuWarmupProvenance,
    cache_state: GpuWarmupCacheState,
    timing_scope: GpuWarmupTimingScope,
}

impl GpuWarmupProfile {
    fn default_observation(measurement: WarmupMeasurementKind) -> GpuWarmupProfileObservation {
        GpuWarmupProfileObservation {
            memory: GpuWarmupMemoryObservations {
                // A legacy `measured` call represents an observed point, not
                // a proof of an allocation envelope.  Callers that possess an
                // exact query or certified envelope use the explicit
                // observation constructor below and keep that evidence.
                evidence: if measurement == WarmupMeasurementKind::HostMeasured {
                    MemoryEvidenceKind::ExactQuery
                } else {
                    MemoryEvidenceKind::MeasuredPeak
                },
                ..GpuWarmupMemoryObservations::default()
            },
            resident_delta: GpuWarmupResidencyDelta::default(),
            repetitions: 1,
            spread_seconds: 0.0,
            provenance: GpuWarmupProvenance::ProductionEquivalent,
            cache_state: GpuWarmupCacheState::Warm,
            timing_scope: if measurement == WarmupMeasurementKind::HostMeasured {
                GpuWarmupTimingScope::ContainingStage
            } else {
                GpuWarmupTimingScope::LocalJob
            },
        }
    }

    /// Construct a profile while preserving the complete measurement
    /// observation.  In particular, `ExactQuery` and `CertifiedEnvelope`
    /// remain distinct from an observed `MeasuredPeak`.
    #[allow(clippy::too_many_arguments)]
    pub fn measured_with_observation(
        time_seconds: f64,
        workspace_bytes: u64,
        measurement: WarmupMeasurementKind,
        memory: GpuWarmupMemoryObservations,
        resident_delta: GpuWarmupResidencyDelta,
        repetitions: usize,
        spread_seconds: f64,
        provenance: GpuWarmupProvenance,
        cache_state: GpuWarmupCacheState,
        timing_scope: GpuWarmupTimingScope,
    ) -> Result<Self, GpuWarmupProfileError> {
        Self::validate_observation(repetitions, spread_seconds)?;
        let profile = Self {
            kind: GpuWarmupProfileKind::Measured,
            measurement,
            time_seconds,
            workspace_bytes,
            preimage_max_attempts: None,
            preimage_certified_tile_width: None,
            preimage_footprint: None,
            resolved_cache_identity: None,
            memory,
            resident_delta,
            repetitions,
            spread_seconds,
            provenance,
            cache_state,
            timing_scope,
            resolved_route_descriptor: None,
        };
        profile.validate()
    }

    /// Convenience constructor for a provider that already has a canonical
    /// memory observation.  The profile's workspace is deliberately kept
    /// separate from the complete per-device observation map.
    pub fn measured_with_memory(
        time_seconds: f64,
        workspace_bytes: u64,
        measurement: WarmupMeasurementKind,
        memory: GpuWarmupMemoryObservations,
        repetitions: usize,
        spread_seconds: f64,
        cache_state: GpuWarmupCacheState,
        timing_scope: GpuWarmupTimingScope,
    ) -> Result<Self, GpuWarmupProfileError> {
        Self::measured_with_observation(
            time_seconds,
            workspace_bytes,
            measurement,
            memory,
            GpuWarmupResidencyDelta::default(),
            repetitions,
            spread_seconds,
            GpuWarmupProvenance::ProductionEquivalent,
            cache_state,
            timing_scope,
        )
    }

    /// Convert this provider result into the canonical session point without
    /// dropping context or evidence.  The key remains authoritative for the
    /// execution class; duplicated cache/timing fields are checked by the
    /// caller when importing the point.
    pub fn into_point(
        self,
        key: GpuWarmupProfileKey,
        coordinate: usize,
        executed_range: IndexRange,
    ) -> Result<GpuWarmupProfilePoint, GpuWarmupProfileError> {
        if key.cache_state != self.cache_state || key.timing_scope != self.timing_scope {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "profile cache/timing context does not match its execution key".into(),
            ));
        }
        let executed_range_class = key.executed_range_class;
        let mut point = GpuWarmupProfilePoint::new(
            key,
            coordinate,
            executed_range,
            executed_range_class,
            self.repetitions,
            self.time_seconds,
            self.spread_seconds,
            self.memory,
            self.resident_delta,
            self.provenance,
        )
        .map_err(|error| GpuWarmupProfileError::InvalidMeasurement(error.to_string()))?;
        point.workspace_bytes = self.workspace_bytes;
        point.preimage_max_attempts = self.preimage_max_attempts;
        point.preimage_certified_tile_width = self.preimage_certified_tile_width;
        point.preimage_footprint = self.preimage_footprint;
        point.resolved_cache_identity = self.resolved_cache_identity;
        point.resolved_route_descriptor = self.resolved_route_descriptor;
        Ok(point)
    }

    fn validate_observation(
        repetitions: usize,
        spread_seconds: f64,
    ) -> Result<(), GpuWarmupProfileError> {
        if repetitions == 0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "measured profile repetitions must be positive".into(),
            ));
        }
        if !spread_seconds.is_finite() || spread_seconds < 0.0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "measured profile spread must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }

    pub fn measured(
        time_seconds: f64,
        workspace_bytes: u64,
    ) -> Result<Self, GpuWarmupProfileError> {
        if !time_seconds.is_finite() || time_seconds <= 0.0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "measured GPU time must be finite and positive".into(),
            ));
        }
        let observation = Self::default_observation(WarmupMeasurementKind::GpuMeasured);
        Ok(Self {
            kind: GpuWarmupProfileKind::Measured,
            measurement: WarmupMeasurementKind::GpuMeasured,
            time_seconds,
            workspace_bytes,
            preimage_max_attempts: None,
            preimage_certified_tile_width: None,
            preimage_footprint: None,
            resolved_cache_identity: None,
            memory: GpuWarmupMemoryObservations {
                affected_devices: observation.memory.affected_devices,
                host_bytes: observation.memory.host_bytes,
                pinned_host_bytes: observation.memory.pinned_host_bytes,
                evidence: observation.memory.evidence,
            },
            resident_delta: observation.resident_delta,
            repetitions: observation.repetitions,
            spread_seconds: observation.spread_seconds,
            provenance: observation.provenance,
            cache_state: observation.cache_state,
            timing_scope: observation.timing_scope,
            resolved_route_descriptor: None,
        })
    }

    pub fn measured_preimage(
        time_seconds: f64,
        workspace_bytes: u64,
        max_attempts: usize,
        certified_tile_width: usize,
    ) -> Result<Self, GpuWarmupProfileError> {
        if !time_seconds.is_finite() || time_seconds <= 0.0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "measured GPU time must be finite and positive".into(),
            ));
        }
        let observation = Self::default_observation(WarmupMeasurementKind::GpuMeasured);
        Ok(Self {
            kind: GpuWarmupProfileKind::Measured,
            measurement: WarmupMeasurementKind::GpuMeasured,
            time_seconds,
            workspace_bytes,
            preimage_max_attempts: Some(max_attempts),
            preimage_certified_tile_width: Some(certified_tile_width),
            preimage_footprint: None,
            resolved_cache_identity: None,
            memory: observation.memory,
            resident_delta: observation.resident_delta,
            repetitions: observation.repetitions,
            spread_seconds: observation.spread_seconds,
            provenance: observation.provenance,
            cache_state: observation.cache_state,
            timing_scope: observation.timing_scope,
            resolved_route_descriptor: None,
        })
    }

    /// Construct a timed host profile. Host profiles intentionally carry no
    /// GPU workspace, but still require an actual elapsed-time measurement.
    pub fn host_measured(time_seconds: f64) -> Result<Self, GpuWarmupProfileError> {
        Self::measured_with_kind(time_seconds, 0, WarmupMeasurementKind::HostMeasured)
    }

    /// Construct a profile using the canonical operation domain selected by
    /// the validated graph. This is the provider-facing constructor that
    /// prevents host operations from being mislabeled as GPU measurements.
    pub fn measured_for_domain(
        domain: CanonicalWarmupProfileDomain,
        time_seconds: f64,
        workspace_bytes: u64,
    ) -> Result<Self, GpuWarmupProfileError> {
        Self::measured_with_kind(time_seconds, workspace_bytes, domain.measurement_kind())
    }

    fn measured_with_kind(
        time_seconds: f64,
        workspace_bytes: u64,
        measurement: WarmupMeasurementKind,
    ) -> Result<Self, GpuWarmupProfileError> {
        if !time_seconds.is_finite() || time_seconds <= 0.0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "measured profile time must be finite and positive".into(),
            ));
        }
        if measurement == WarmupMeasurementKind::HostMeasured && workspace_bytes != 0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "host profile must not claim GPU workspace".into(),
            ));
        }
        let observation = Self::default_observation(measurement);
        Ok(Self {
            kind: GpuWarmupProfileKind::Measured,
            measurement,
            time_seconds,
            workspace_bytes,
            preimage_max_attempts: None,
            preimage_certified_tile_width: None,
            preimage_footprint: None,
            resolved_cache_identity: None,
            memory: observation.memory,
            resident_delta: observation.resident_delta,
            repetitions: observation.repetitions,
            spread_seconds: observation.spread_seconds,
            provenance: observation.provenance,
            cache_state: observation.cache_state,
            timing_scope: observation.timing_scope,
            resolved_route_descriptor: None,
        })
    }

    pub fn validate(self) -> Result<Self, GpuWarmupProfileError> {
        Self::validate_observation(self.repetitions, self.spread_seconds)?;
        if self.resolved_route_descriptor.as_ref().is_some_and(|descriptor| !descriptor.validate())
        {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "profile contains an invalid authoritative production route".into(),
            ));
        }
        if self.measurement == WarmupMeasurementKind::HostMeasured && self.workspace_bytes != 0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "host profile must not claim GPU workspace".into(),
            ));
        }
        if let Some(ref footprint) = self.preimage_footprint {
            let Some(width) = self.preimage_certified_tile_width else {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "preimage footprint requires a certified tile width".into(),
                ));
            };
            if width == 0 || footprint.certified_tile_width != Some(width) {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "preimage footprint width does not match profile metadata".into(),
                ));
            }
        }
        if self.measurement == WarmupMeasurementKind::HostMeasured &&
            self.timing_scope != GpuWarmupTimingScope::Transfer
        {
            // Host work may use ordinary pageable host memory, but it must
            // never claim a GPU device observation.  A zero pinned value is a
            // valid observation; do not reject it merely because the host
            // routine did not stage through pinned memory.
            if !self.memory.affected_devices.is_empty() ||
                self.memory.host_bytes != 0 ||
                self.memory.pinned_host_bytes != 0 ||
                self.memory.evidence != MemoryEvidenceKind::ExactQuery
            {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "host profile must carry explicit exact-zero memory evidence".into(),
                ));
            }
        } else if self.memory.evidence.is_hard_admission() &&
            self.memory.affected_devices.is_empty()
        {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "GPU exact/certified memory evidence must identify at least one affected device"
                    .into(),
            ));
        }
        if self.kind == GpuWarmupProfileKind::Measured &&
            self.time_seconds.is_finite() &&
            self.time_seconds > 0.0
        {
            Ok(self)
        } else {
            Err(GpuWarmupProfileError::InvalidMeasurement(
                "measured profile must contain a finite positive time".into(),
            ))
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuWarmupProfileError {
    MissingProfile(GpuWarmupProfileRequest),
    InvalidMeasurement(String),
    /// The production measurement could not fit this candidate in device
    /// memory.  This is deliberately separate from [`Measurement`]: warmup
    /// may discard this candidate and continue with a smaller one, while all
    /// other measurement failures remain fatal.
    OutOfMemory(String),
    Measurement(String),
}

impl fmt::Display for GpuWarmupProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingProfile(request) => write!(
                formatter,
                "missing GPU warmup profile for operation {:?}, device {}, tile width {}, range [{}, {})",
                request.signature,
                request.device,
                request.tile_width,
                request.range.start,
                request.range.end
            ),
            Self::InvalidMeasurement(message) => formatter.write_str(message),
            Self::OutOfMemory(message) => {
                write!(formatter, "GPU warmup measurement ran out of memory: {message}")
            }
            Self::Measurement(message) => {
                write!(formatter, "GPU warmup measurement failed: {message}")
            }
        }
    }
}

impl std::error::Error for GpuWarmupProfileError {}

/// Setup-time source of warmup profiles. Implementations must use the same
/// device-local production range operation that fixed execution will submit;
/// this trait is intentionally unavailable as a production-operation fallback.
pub trait GpuWarmupProfileProvider {
    /// Binds the validated operation descriptor before any candidate request
    /// is measured. A provider must not rely on caller-side profile injection
    /// to discover the production operation.
    fn register_operation(
        &mut self,
        descriptor: GpuWarmupOperationDescriptor,
    ) -> Result<(), GpuWarmupProfileError>;

    fn measure(
        &mut self,
        request: &GpuWarmupProfileRequest,
    ) -> Result<GpuWarmupProfile, GpuWarmupProfileError>;
}

impl<P> GpuWarmupProfileProvider for &mut P
where
    P: GpuWarmupProfileProvider + ?Sized,
{
    fn register_operation(
        &mut self,
        descriptor: GpuWarmupOperationDescriptor,
    ) -> Result<(), GpuWarmupProfileError> {
        (**self).register_operation(descriptor)
    }

    fn measure(
        &mut self,
        request: &GpuWarmupProfileRequest,
    ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
        (**self).measure(request)
    }
}

/// Read-only profile set handed from setup into fixed execution.  It exposes
/// lookup only and deliberately does not implement [`GpuWarmupProfileProvider`]
/// or any measurement API, so fixed code cannot trigger a late production
/// range measurement through the type system.
#[derive(Clone, Debug, Default)]
pub struct FrozenGpuWarmupProfiles {
    profiles: HashMap<GpuWarmupProfileRequest, GpuWarmupProfile>,
}

impl FrozenGpuWarmupProfiles {
    pub fn cached_profile(
        &self,
        request: &GpuWarmupProfileRequest,
    ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
        self.profiles
            .get(request)
            .cloned()
            .ok_or_else(|| GpuWarmupProfileError::MissingProfile(request.clone()))
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

pub trait Backend {
    type Matrix: Clone + Debug + PartialEq + Send + Sync + 'static;
    type SmallMatrix: Clone + Debug + PartialEq + Send + Sync;
    type Trapdoor: Clone + Debug + Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    fn fused_batch(
        &mut self,
        requests: Vec<DynamicFusedBatchRequest<Self::Matrix, Self::SmallMatrix>>,
    ) -> Result<Vec<FusedBatchOutput<Self::Matrix, Self::SmallMatrix>>, Self::Error> {
        requests
            .into_iter()
            .map(|request| match request {
                DynamicFusedBatchRequest::RowSum { source, right, rows } => {
                    let output = match right {
                        Some(right) => self.tensor_sum_rows(&source, &right, &rows),
                        None => self.sum_rows(&source, &rows),
                    }?;
                    Ok(FusedBatchOutput::Matrices(vec![output]))
                }
                DynamicFusedBatchRequest::Decompose { blocks, small, digits } => self
                    .gadget_decompose_row_blocks(
                        &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                        small,
                        Some(digits),
                    )
                    .map(FusedBatchOutput::Small),
                DynamicFusedBatchRequest::SmallProduct { blocks, rhs } => self
                    .multiply_small_rhs_row_blocks(
                        &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                        &rhs,
                    )
                    .map(FusedBatchOutput::Matrices),
                DynamicFusedBatchRequest::Add { blocks, right } => self
                    .add_row_blocks(&blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(), &right)
                    .map(|value| FusedBatchOutput::Matrices(vec![value])),
            })
            .collect()
    }

    /// Dispatches a fixed fused batch. Metadata is required by the type, so
    /// fixed execution has no dynamic/pilot fallback state to interpret.
    fn fixed_fused_batch(
        &mut self,
        requests: Vec<FusedBatchRequest<Self::Matrix, Self::SmallMatrix>>,
    ) -> Result<Vec<FusedBatchOutput<Self::Matrix, Self::SmallMatrix>>, Self::Error> {
        requests
            .into_iter()
            .map(|request| match request {
                FusedBatchRequest::RowSum { source, right, rows, metadata: _ } => {
                    let output = match right {
                        Some(right) => self.tensor_sum_rows(&source, &right, &rows),
                        None => self.sum_rows(&source, &rows),
                    }?;
                    Ok(FusedBatchOutput::Matrices(vec![output]))
                }
                FusedBatchRequest::Decompose { blocks, small, digits, metadata: _ } => self
                    .gadget_decompose_row_blocks(
                        &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                        small,
                        Some(digits),
                    )
                    .map(FusedBatchOutput::Small),
                FusedBatchRequest::SmallProduct { blocks, rhs, metadata: _ } => self
                    .multiply_small_rhs_row_blocks(
                        &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                        &rhs,
                    )
                    .map(FusedBatchOutput::Matrices),
                FusedBatchRequest::Add { blocks, right, metadata: _ } => self
                    .add_row_blocks(&blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(), &right)
                    .map(|value| FusedBatchOutput::Matrices(vec![value])),
            })
            .collect()
    }

    /// Dispatches one fixed-plan sibling batch.  The default implementation
    /// preserves CPU semantics, while GPU fleets override this hook to submit
    /// the complete wave as one owner-rotated batch.  In particular, callers
    /// must not replace this with `select_gpu_operation` or a pilot-derived
    /// single-instance dispatch when a frozen plan is active.
    fn fixed_operation_batch(
        &mut self,
        requests: Vec<FixedOperationBatchRequest<Self::Matrix, Self::SmallMatrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        requests
            .into_iter()
            .map(|request| match request {
                FixedOperationBatchRequest::SingleDeviceConstant {
                    ty,
                    value,
                    env,
                    metadata: _,
                } => self.constant_matrix(&ty, &value, &env),
                FixedOperationBatchRequest::GeneratedConstant { ty, value, env, metadata: _ } => {
                    self.constant_matrix(&ty, &value, &env)
                }
                FixedOperationBatchRequest::LiftIntegerToConstantPolynomial {
                    ty,
                    coefficient,
                    metadata: _,
                } => {
                    let identity =
                        self.constant_matrix(&ty, &ConstantMatrix::Identity, &ParamEnv::default())?;
                    self.scale_integer(&identity, &coefficient)
                }
                FixedOperationBatchRequest::MatrixBinary {
                    operation,
                    left,
                    right,
                    metadata: _,
                } => match operation {
                    MatrixBinaryOp::Add => self.add(&left, &right),
                    MatrixBinaryOp::Subtract => self.sub(&left, &right),
                    MatrixBinaryOp::Multiply => self.multiply(&left, &right),
                },
                FixedOperationBatchRequest::MatrixMulSmallRhs { left, right, metadata: _ } => {
                    self.multiply_small_rhs(&left, &right)
                }
                FixedOperationBatchRequest::MatrixMulAccumulate { request, metadata: _ } => {
                    self.matrix_mul_accumulate(request)
                }
                FixedOperationBatchRequest::Negate { value, metadata: _ } => self.negate(&value),
                FixedOperationBatchRequest::Scale { value, scalar, metadata: _ } => {
                    self.scale_integer(&value, &scalar)
                }
                FixedOperationBatchRequest::UnaryTransform { operation, value, metadata: _ } => {
                    match operation {
                        FixedUnaryOperation::RingAutomorphism { index } => {
                            self.ring_automorphism(&value, index)
                        }
                        FixedUnaryOperation::ModulusSwitch { destination } => {
                            self.modulus_switch(&value, &destination)
                        }
                        FixedUnaryOperation::ReduceModulus { destination } => {
                            self.reduce_modulus(&value, &destination)
                        }
                        FixedUnaryOperation::CenteredRebase { destination } => {
                            self.centered_rebase(&value, &destination)
                        }
                        FixedUnaryOperation::RnsModUp {
                            destination,
                            source_moduli,
                            digit_size,
                            normalize,
                        } => self.rns_mod_up(
                            &value,
                            &destination,
                            &source_moduli,
                            digit_size,
                            normalize,
                        ),
                        FixedUnaryOperation::RnsModDown {
                            destination,
                            source_moduli,
                            plaintext_modulus,
                        } => self.rns_mod_down(
                            &value,
                            &destination,
                            &source_moduli,
                            plaintext_modulus,
                        ),
                        FixedUnaryOperation::Transpose => self.transpose(&value),
                        FixedUnaryOperation::Slice { rows, columns } => {
                            self.slice(&value, rows.as_ref(), columns.as_ref())
                        }
                    }
                }
                FixedOperationBatchRequest::Tensor { left, right, metadata: _ } => {
                    self.tensor(&left, &right)
                }
                FixedOperationBatchRequest::Concat { inputs, axis, metadata: _ } => {
                    let refs = inputs.iter().map(Arc::as_ref).collect::<Vec<_>>();
                    self.concat(&refs, axis)
                }
                FixedOperationBatchRequest::CrtRecompose {
                    levels,
                    plaintext_moduli,
                    reconstruction_coefficients,
                    destination,
                    metadata: _,
                } => self.crt_recompose(
                    &levels,
                    &plaintext_moduli,
                    &reconstruction_coefficients,
                    &destination,
                ),
            })
            .collect()
    }

    fn fixed_generation_batch(
        &mut self,
        requests: Vec<FixedGenerationRequest>,
    ) -> Result<Vec<FixedGenerationOutput<Self::Matrix, Self::SmallMatrix>>, Self::Error> {
        requests
            .into_iter()
            .map(|request| match request {
                FixedGenerationRequest::Uniform { ty, range, metadata: _ } => {
                    self.sample_uniform(&ty, &range).map(FixedGenerationOutput::Matrix)
                }
                FixedGenerationRequest::Gaussian {
                    ty,
                    sigma,
                    max_coefficient_bound,
                    metadata: _,
                } => self
                    .sample_gaussian(&ty, sigma, &max_coefficient_bound)
                    .map(FixedGenerationOutput::Matrix),
                FixedGenerationRequest::Hash { ty, key, tag, metadata: _ } => {
                    self.sample_hash(&ty, key, &tag).map(FixedGenerationOutput::Matrix)
                }
                FixedGenerationRequest::HashDecomposed {
                    ty,
                    key,
                    tag,
                    gadget_base,
                    digit_count,
                    small,
                    metadata: _,
                } => {
                    let output = if small {
                        self.sample_hash_small_decomposed(
                            &ty,
                            key,
                            &tag,
                            &gadget_base,
                            digit_count,
                        )?
                    } else {
                        self.sample_hash_decomposed(&ty, key, &tag, &gadget_base, digit_count)?
                    };
                    Ok(FixedGenerationOutput::Small(output))
                }
            })
            .collect()
    }

    fn fixed_gadget_decompose_batch(
        &mut self,
        requests: Vec<FixedGadgetDecomposeRequest<Self::Matrix>>,
    ) -> Result<Vec<Self::SmallMatrix>, Self::Error> {
        requests
            .into_iter()
            .map(|request| {
                self.gadget_decompose(&request.input, request.small, Some(request.digits))
            })
            .collect()
    }

    /// Imports one polynomial from coefficient or native evaluation values.
    fn polynomial_from_values(
        &mut self,
        ty: &ConcreteMatrixType,
        values: &[BigInt],
        evaluation: bool,
    ) -> Result<Self::Matrix, Self::Error>;

    /// Exports one scalar polynomial in coefficient or native evaluation order.
    fn polynomial_values(
        &mut self,
        value: &Self::Matrix,
        evaluation: bool,
    ) -> Result<Vec<BigInt>, Self::Error>;

    /// Selects the setup-time GPU calibration for the next primitive. CPU and
    /// non-fleet backends ignore this hook.
    fn select_gpu_operation(&mut self, _operation: [u8; 32]) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Validates backend-specific parts of a frozen plan before the first
    /// dispatch.  CPU backends intentionally accept the metadata and retain
    /// their existing execution semantics.  GPU fleet backends may compare
    /// device identity, representation, and revision here.
    fn validate_frozen_gpu_plan(&self, _plan: &FrozenGpuPlan) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Returns the backend's current runtime contract when it can describe
    /// one. The executor compares it with the frozen contract before any
    /// input materialization or submit, covering backend identity/revision,
    /// logical-to-physical mapping, budgets, and shape/representation/source
    /// layout metadata. Backends without a fleet contract return `None`.
    fn gpu_runtime_contract(
        &self,
        _validated: &mxx_ir_core::ValidatedGraph,
        _inputs: &BTreeMap<String, RuntimeValue<Self>>,
    ) -> Result<Option<crate::gpu_execution_plan::GpuPlanContract>, Self::Error>
    where
        Self: Sized,
    {
        Ok(None)
    }

    /// Returns backend-authoritative physical storage facts for the concrete
    /// matrix types used by warmup.  GPU implementations must derive these
    /// from their registered native parameters; callers must not synthesize a
    /// tower count or leave the descriptor map empty.
    fn gpu_physical_storage_contract(
        &self,
        _types: &[ConcreteWireType],
    ) -> Result<Option<BackendStorageContract>, Self::Error>
    where
        Self: Sized,
    {
        Ok(None)
    }

    /// Compares caller-provided warmup storage metadata with the backend's
    /// native descriptor.  The default is intentionally a no-op for CPU
    /// backends; fleet backends override it to reject stale or fabricated
    /// `active_crt_towers` and descriptor maps before planning.
    fn validate_gpu_physical_storage_contract(
        &self,
        types: &[ConcreteWireType],
        supplied: &BackendStorageContract,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let _ = (types, supplied);
        Ok(())
    }

    /// Freeze public resource limits before warmup. CPU backends have no GPU
    /// contract; fleet backends validate device limits against their contexts.
    fn configure_gpu_plan_budgets(
        &mut self,
        _budgets: &[crate::gpu_execution_plan::GpuDeviceBudget],
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Installs the value-only plan into a backend's fixed-dispatch state.
    /// The default CPU implementation has no state to install.  GPU fleet
    /// implementations should bind the plan without allocating or profiling.
    fn install_frozen_gpu_plan(&mut self, _plan: &FrozenGpuPlan) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Receives the fixed node choice before any primitive is submitted.
    /// Implementations must not use this hook to discover a different width,
    /// owner, device, or pilot result.  The default is the CPU boundary and
    /// leaves the ordinary operation methods untouched.
    fn prepare_fixed_node_batch(
        &mut self,
        _request: &PlannedNodeBatchRequest,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Whether the backend has installed a fixed production plan. This is
    /// separate from `placement_count`: a fleet may expose one executor
    /// placement while dispatching fixed jobs across several logical GPUs.
    fn fixed_plan_active(&self) -> bool {
        false
    }

    fn placement_count(&self) -> usize {
        1
    }
    fn active_placement(&self) -> usize {
        0
    }
    fn set_active_placement(&mut self, placement: usize) -> bool {
        placement == 0
    }
    fn matrix_to_active_placement(
        &mut self,
        value: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        Ok(value.clone())
    }
    fn matrix_is_on_active_placement(&self, _value: &Self::Matrix) -> bool {
        true
    }
    fn small_matrix_to_active_placement(
        &mut self,
        value: &Self::SmallMatrix,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        Ok(value.clone())
    }
    fn small_matrix_is_on_active_placement(&self, _value: &Self::SmallMatrix) -> bool {
        true
    }
    /// Waits only for releases queued on backend-owned release streams.
    fn fence_released_memory(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn matrix_to_placements(
        &mut self,
        value: &Self::Matrix,
    ) -> Result<Vec<Option<Self::Matrix>>, Self::Error> {
        let original = self.active_placement();
        let result = (|| {
            let mut placed = Vec::with_capacity(self.placement_count());
            for placement in 0..self.placement_count() {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                placed.push(if self.matrix_is_on_active_placement(value) {
                    None
                } else {
                    Some(self.matrix_to_active_placement(value)?)
                });
            }
            Ok(placed)
        })();
        assert!(self.set_active_placement(original), "backend rejected its active placement");
        result
    }
    fn small_matrix_to_placements(
        &mut self,
        value: &Self::SmallMatrix,
    ) -> Result<Vec<Option<Self::SmallMatrix>>, Self::Error> {
        let original = self.active_placement();
        let result = (|| {
            let mut placed = Vec::with_capacity(self.placement_count());
            for placement in 0..self.placement_count() {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                placed.push(if self.small_matrix_is_on_active_placement(value) {
                    None
                } else {
                    Some(self.small_matrix_to_active_placement(value)?)
                });
            }
            Ok(placed)
        })();
        assert!(self.set_active_placement(original), "backend rejected its active placement");
        result
    }
    fn trapdoor_to_active_placement(
        &mut self,
        _ty: &ConcreteMatrixType,
        value: &Self::Trapdoor,
    ) -> Result<Self::Trapdoor, Self::Error> {
        Ok(value.clone())
    }
    fn trapdoor_to_placements(
        &mut self,
        ty: &ConcreteMatrixType,
        value: &Self::Trapdoor,
        source_placement: usize,
    ) -> Result<Vec<Self::Trapdoor>, Self::Error> {
        let original = self.active_placement();
        let result = (|| {
            let mut placed = Vec::with_capacity(self.placement_count());
            for placement in 0..self.placement_count() {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                placed.push(if placement == source_placement {
                    value.clone()
                } else {
                    self.trapdoor_to_active_placement(ty, value)?
                });
            }
            Ok(placed)
        })();
        assert!(self.set_active_placement(original), "backend rejected its active placement");
        result
    }

    fn constant_matrix(
        &mut self,
        ty: &ConcreteMatrixType,
        value: &ConstantMatrix,
        env: &ParamEnv,
    ) -> Result<Self::Matrix, Self::Error>;
    fn add(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn add_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        let left = self.concat(blocks, ConcatAxis::Rows)?;
        self.add(&left, right)
    }
    fn add_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(left, right)| self.add(&left, &right)).collect()
    }
    fn sub(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sub_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(left, right)| self.sub(&left, &right)).collect()
    }
    fn multiply(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn multiply_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(left, right)| self.multiply(&left, &right)).collect()
    }
    fn matrix_mul_accumulate(
        &mut self,
        request: MatrixMulAccumulateRequest<Self::Matrix>,
    ) -> Result<Self::Matrix, Self::Error> {
        let mut products = request.products.into_iter();
        let (coefficient, left, right) =
            products.next().expect("validated multi-row GEMM has a product");
        let product = self.multiply(&left, &right)?;
        let mut output = self.scale_integer(&product, &coefficient)?;
        for (coefficient, left, right) in products {
            let product = self.multiply(&left, &right)?;
            let product = self.scale_integer(&product, &coefficient)?;
            output = self.add(&output, &product)?;
        }
        if let Some(bias) = request.bias {
            output = self.add(&output, &bias)?;
        }
        Ok(output)
    }
    fn matrix_mul_accumulate_batch(
        &mut self,
        requests: Vec<MatrixMulAccumulateRequest<Self::Matrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        requests.into_iter().map(|request| self.matrix_mul_accumulate(request)).collect()
    }
    fn negate(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error>;
    fn negate_batch(
        &mut self,
        inputs: Vec<Arc<Self::Matrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|value| self.negate(&value)).collect()
    }
    fn scale_integer(
        &mut self,
        value: &Self::Matrix,
        scalar: &BigInt,
    ) -> Result<Self::Matrix, Self::Error>;
    fn scale_integer_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, BigInt)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(value, scalar)| self.scale_integer(&value, &scalar)).collect()
    }
    fn ring_automorphism(
        &mut self,
        value: &Self::Matrix,
        index: usize,
    ) -> Result<Self::Matrix, Self::Error>;

    fn modulus_switch(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error>;
    fn reduce_modulus(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error>;

    fn centered_rebase(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error>;

    fn rns_mod_up(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        source_moduli: &[u64],
        digit_size: usize,
        normalize: bool,
    ) -> Result<Self::Matrix, Self::Error>;

    fn rns_mod_down(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        source_moduli: &[u64],
        plaintext_modulus: u64,
    ) -> Result<Self::Matrix, Self::Error>;

    fn ring_automorphism_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, usize)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(value, index)| self.ring_automorphism(&value, index)).collect()
    }

    fn preimage_target(
        &mut self,
        value: Arc<Self::Matrix>,
    ) -> Result<(Arc<dyn PolyMatrixColumnSource<Self::Matrix>>, Arc<Vec<u8>>), Self::Error>;

    fn matrix_from_cpu_staging_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error>;

    fn preimage_target_from_staging(
        &self,
        ty: &ConcreteMatrixType,
        rows: usize,
        columns: usize,
        bytes: Arc<Vec<u8>>,
    ) -> Result<Arc<dyn PolyMatrixColumnSource<Self::Matrix>>, Self::Error>;

    fn validate_preimage_bound(
        &self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
    ) -> Result<(), Self::Error>;

    fn transpose(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error>;
    fn slice(
        &mut self,
        value: &Self::Matrix,
        rows: Option<&IndexRange>,
        columns: Option<&IndexRange>,
    ) -> Result<Self::Matrix, Self::Error>;
    /// Sums each nonempty group of source rows into one output row.
    /// Indices must be in bounds; repeated indices retain their multiplicity.
    fn sum_rows(
        &mut self,
        value: &Self::Matrix,
        rows: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        if rows.is_empty() {
            return self.slice(value, Some(&IndexRange { start: 0, end: 0 }), None);
        }
        let output = rows
            .iter()
            .map(|group| {
                let (first, rest) = group.split_first().expect("row sum group must be nonempty");
                let mut sum =
                    self.slice(value, Some(&IndexRange { start: *first, end: first + 1 }), None)?;
                for row in rest {
                    let term =
                        self.slice(value, Some(&IndexRange { start: *row, end: row + 1 }), None)?;
                    sum = self.add(&sum, &term)?;
                }
                Ok(sum)
            })
            .collect::<Result<Vec<_>, Self::Error>>()?;
        self.concat(&output.iter().collect::<Vec<_>>(), ConcatAxis::Rows)
    }
    /// Sum selected tensor-product rows without requiring a materialized tensor.
    fn tensor_sum_rows(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
        rows: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        let tensor = self.tensor(left, right)?;
        self.sum_rows(&tensor, rows)
    }
    fn tensor(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn concat(
        &mut self,
        inputs: &[&Self::Matrix],
        axis: ConcatAxis,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_uniform(
        &mut self,
        ty: &ConcreteMatrixType,
        range: &SampleRange,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_gaussian(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: &BigInt,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_hash(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_hash_decomposed(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn sample_hash_small_decomposed(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn sample_trapdoor(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<(Self::Matrix, Self::Trapdoor), Self::Error>;

    /// Fixed-plan trapdoor construction.  GPU fleets override this entry
    /// point to consume the frozen node metadata and owner; the default keeps
    /// CPU backends equivalent to their ordinary sampler.
    fn fixed_sample_trapdoor(
        &mut self,
        request: FixedTrapdoorRequest,
    ) -> Result<(Self::Matrix, Self::Trapdoor), Self::Error> {
        self.sample_trapdoor(&request.ty, request.sigma, &request.gadget_base, request.digit_count)
    }
    fn sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &Self::Trapdoor,
        public: &Self::Matrix,
        target: &dyn PolyMatrixColumnSource<Self::Matrix>,
        randomness_seed: [u8; 32],
    ) -> Result<Self::SmallMatrix, Self::Error>;

    /// Fixed-plan preimage entry point. GPU fleet implementations consume the
    /// already selected owner intervals and tile widths here. CPU and legacy
    /// backends retain their ordinary sampler semantics through the default.
    fn fixed_sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &Self::Trapdoor,
        public: &Self::Matrix,
        target: &dyn PolyMatrixColumnSource<Self::Matrix>,
        randomness_seed: [u8; 32],
    ) -> Result<Self::SmallMatrix, Self::Error> {
        self.sample_preimage(
            ty,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
            trapdoor,
            public,
            target,
            randomness_seed,
        )
    }

    fn sample_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<Self::Matrix, Self::Trapdoor>>,
    ) -> Result<Vec<Self::SmallMatrix>, Self::Error> {
        requests
            .into_iter()
            .map(|request| {
                let sample = if self.fixed_plan_active() {
                    Self::fixed_sample_preimage
                } else {
                    Self::sample_preimage
                };
                sample(
                    self,
                    &request.matrix_type,
                    request.sigma,
                    &request.gadget_base,
                    request.digit_count,
                    &request.max_coefficient_bound,
                    request.trapdoor.as_ref(),
                    request.public.as_ref(),
                    request.target.as_ref(),
                    request.randomness_seed,
                )
            })
            .collect()
    }
    fn sample_preimage_batches_by_placement(
        &mut self,
        batches: Vec<(usize, Vec<PreimageRequest<Self::Matrix, Self::Trapdoor>>)>,
    ) -> Result<Vec<(usize, Vec<Self::SmallMatrix>)>, Self::Error> {
        let original = self.active_placement();
        let result = batches
            .into_iter()
            .map(|(placement, requests)| {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                self.sample_preimage_batch(requests).map(|outputs| (placement, outputs))
            })
            .collect();
        assert!(self.set_active_placement(original), "backend rejected its active placement");
        result
    }
    fn validate_gadget_layout(
        &self,
        _ty: &ConcreteMatrixType,
        _gadget_base: &BigInt,
        _digit_count: usize,
        _small: bool,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Inclusive full-modulus reconstruction error for this backend's regular gadget.
    /// Exact non-CRT backends keep the default zero bound.
    fn gadget_error_bound(
        &self,
        _ty: &ConcreteMatrixType,
        _digit_count: Option<usize>,
    ) -> Result<BigInt, Self::Error> {
        Ok(BigInt::from(0u8))
    }
    fn gadget_decompose(
        &mut self,
        value: &Self::Matrix,
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn multiply_small_rhs(
        &mut self,
        lhs: &Self::Matrix,
        rhs: &Self::SmallMatrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn gadget_decompose_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        let value = self.concat(blocks, ConcatAxis::Rows)?;
        self.gadget_decompose(&value, small, digit_count)
    }
    fn multiply_small_rhs_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        rhs: &Self::SmallMatrix,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        blocks.iter().map(|block| self.multiply_small_rhs(block, rhs)).collect()
    }
    fn extract_coefficient(
        &mut self,
        value: &Self::Matrix,
        position: usize,
    ) -> Result<BigInt, Self::Error>;
    fn threshold_decode(
        &mut self,
        value: &Self::Matrix,
        plaintext_modulus: &BigInt,
        length: usize,
    ) -> Result<Vec<BigInt>, Self::Error>;
    fn pack_polynomial_coefficients(
        &mut self,
        ty: &ConcreteMatrixType,
        bits: &[bool],
        coefficient_bits: usize,
    ) -> Result<Self::Matrix, Self::Error>;
    fn crt_recompose(
        &mut self,
        levels: &[Self::Matrix],
        plaintext_moduli: &[BigInt],
        reconstruction_coefficients: &[BigInt],
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error>;

    fn matrix_to_bytes(&self, value: &Self::Matrix) -> Vec<u8>;
    fn matrices_to_bytes(&self, values: &[&Self::Matrix]) -> Vec<Vec<u8>> {
        values.iter().map(|value| self.matrix_to_bytes(value)).collect()
    }
    /// Decodes an intact payload from this backend's matching matrix codec and type.
    /// Malformed bytes or mismatched metadata violate the caller contract; implementations
    /// may panic. The result reports supported backend failures, not arbitrary corruption.
    fn matrix_from_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error>;
    fn small_matrix_to_bytes(
        &self,
        value: &Self::SmallMatrix,
        expected_schema: &ConcreteBoundedMatrixSchema,
        semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Vec<u8>, Self::Error>;
    fn small_matrix_from_bytes(
        &self,
        expected_schema: &ConcreteBoundedMatrixSchema,
        bytes: &[u8],
        expected_semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn trapdoor_to_bytes(&self, value: &Self::Trapdoor) -> Vec<u8>;
    fn trapdoor_from_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Trapdoor, Self::Error>;
}

/// A caller-supplied value must match its validated concrete wire type in full, including
/// shape, ring/CRT parameters, representation, and sampler metadata. Trapdoor public and
/// secret material must belong together. This obligation applies recursively to families;
/// execution's value-kind checks do not certify it or inspect resident matrix contents.
pub enum RuntimeValue<B: Backend> {
    Int(BigInt),
    Real(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    TypedBlob(Vec<u8>),
    Matrix(Arc<B::Matrix>),
    /// Host-staged matrix; expanded columns are loaded only when consumed.
    HostMatrix {
        matrix_type: ConcreteMatrixType,
        bytes: Arc<Vec<u8>>,
    },
    SmallMatrix(Arc<B::SmallMatrix>),
    Trapdoor {
        secret: Option<Arc<B::Trapdoor>>,
        public: Arc<B::Matrix>,
        matrix_type: ConcreteMatrixType,
        sigma: f64,
        gadget_base: BigInt,
        digit_count: usize,
        gadget_small: Option<bool>,
    },
    LazyArtifact {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        index: Option<usize>,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    LazyArtifactFamily {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    StagedArtifact {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        index: usize,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    StagedArtifactFamily {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    IndexedFamily(Vec<RuntimeValue<B>>),
}

impl<B: Backend> Clone for RuntimeValue<B> {
    fn clone(&self) -> Self {
        match self {
            Self::Int(value) => Self::Int(value.clone()),
            Self::Real(value) => Self::Real(*value),
            Self::Bool(value) => Self::Bool(*value),
            Self::Bytes(value) => Self::Bytes(value.clone()),
            Self::TypedBlob(value) => Self::TypedBlob(value.clone()),
            Self::Matrix(value) => Self::Matrix(value.clone()),
            Self::HostMatrix { matrix_type, bytes } => {
                Self::HostMatrix { matrix_type: matrix_type.clone(), bytes: bytes.clone() }
            }
            Self::SmallMatrix(value) => Self::SmallMatrix(value.clone()),
            Self::Trapdoor {
                secret,
                public,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                gadget_small,
            } => Self::Trapdoor {
                secret: secret.clone(),
                public: public.clone(),
                matrix_type: matrix_type.clone(),
                sigma: *sigma,
                gadget_base: gadget_base.clone(),
                digit_count: *digit_count,
                gadget_small: *gadget_small,
            },
            Self::LazyArtifact { production, name, index, descriptor } => Self::LazyArtifact {
                production: production.clone(),
                name: name.clone(),
                index: *index,
                descriptor: descriptor.clone(),
            },
            Self::LazyArtifactFamily { production, name, descriptor } => Self::LazyArtifactFamily {
                production: production.clone(),
                name: name.clone(),
                descriptor: descriptor.clone(),
            },
            Self::StagedArtifact { production, name, index, descriptor } => Self::StagedArtifact {
                production: production.clone(),
                name: name.clone(),
                index: *index,
                descriptor: descriptor.clone(),
            },
            Self::StagedArtifactFamily { production, name, descriptor } => {
                Self::StagedArtifactFamily {
                    production: production.clone(),
                    name: name.clone(),
                    descriptor: descriptor.clone(),
                }
            }
            Self::IndexedFamily(values) => Self::IndexedFamily(values.clone()),
        }
    }
}

impl<B: Backend> RuntimeValue<B> {
    pub(crate) fn releases_backend_resources_on_drop(&self) -> bool {
        match self {
            Self::Matrix(matrix) => Arc::strong_count(matrix) == 1,
            Self::SmallMatrix(matrix) => Arc::strong_count(matrix) == 1,
            Self::Trapdoor { secret, public, .. } => {
                Arc::strong_count(public) == 1 ||
                    secret.as_ref().is_some_and(|secret| Arc::strong_count(secret) == 1)
            }
            Self::IndexedFamily(values) => {
                values.iter().any(Self::releases_backend_resources_on_drop)
            }
            Self::HostMatrix { .. } |
            Self::Int(_) |
            Self::Real(_) |
            Self::Bool(_) |
            Self::Bytes(_) |
            Self::TypedBlob(_) |
            Self::LazyArtifact { .. } |
            Self::LazyArtifactFamily { .. } |
            Self::StagedArtifact { .. } |
            Self::StagedArtifactFamily { .. } => false,
        }
    }
}

impl<B: Backend> RuntimeValue<B> {
    pub fn matrix(value: B::Matrix) -> Self {
        Self::Matrix(Arc::new(value))
    }

    pub fn small_matrix(value: B::SmallMatrix) -> Self {
        Self::SmallMatrix(Arc::new(value))
    }

    /// Check the complete runtime shape against validated wire metadata.
    ///
    /// Input binding and warmup representatives must use this same check.
    /// Keeping it on the value prevents callers from accidentally accepting a
    /// scalar or a compact/preimage value merely because it has a compatible
    /// top-level Rust representation.
    pub fn matches_wire_type(&self, concrete: &ConcreteWireType) -> bool {
        match (self, concrete) {
            (Self::Int(_), ConcreteWireType::ConstantInt | ConcreteWireType::Int) |
            (Self::Real(_), ConcreteWireType::ConstantReal | ConcreteWireType::Real) |
            (Self::Bool(_), ConcreteWireType::ConstantBool | ConcreteWireType::Bool) |
            (Self::Bytes(_), ConcreteWireType::Bytes { .. }) |
            (Self::TypedBlob(_), ConcreteWireType::TypedBlob { .. }) |
            (Self::Matrix(_) | Self::HostMatrix { .. }, ConcreteWireType::Matrix(_)) |
            (
                Self::SmallMatrix(_),
                ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. },
            ) |
            (Self::Trapdoor { .. }, ConcreteWireType::Trapdoor { .. }) => true,
            (Self::IndexedFamily(values), ConcreteWireType::IndexedFamily { element, count }) => {
                values.len() == *count &&
                    values.iter().all(|value| value.matches_wire_type(element))
            }
            (
                Self::LazyArtifactFamily { descriptor, .. } |
                Self::StagedArtifactFamily { descriptor, .. },
                ConcreteWireType::IndexedFamily { element, count },
            ) => {
                descriptor.family_count == Some(*count) &&
                    ArtifactType::from_wire_type(element).as_ref() ==
                        Some(&descriptor.artifact_type)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod storage_contract_tests {
    use super::*;

    fn matrix_type() -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: BigInt::from(17u8), ring_dimension: 8, rows: 1, columns: 2 }
    }

    fn descriptor(level: usize) -> BackendStorageDescriptor {
        BackendStorageDescriptor {
            representation: "full_dcrt".into(),
            ordered_crt_basis: vec![17, 19, 23, 29],
            level,
            limb_bytes: 8,
        }
    }

    #[test]
    fn storage_contract_rejects_empty_map_and_fabricated_one_tower_depth() {
        let ty = matrix_type();
        let wire = ConcreteWireType::Matrix(ty.clone());
        let empty = BackendStorageContract {
            descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
        };
        assert!(validate_backend_storage_contract(std::slice::from_ref(&wire), &empty).is_err());

        let four_tower = BackendStorageContract {
            descriptors: BTreeMap::from([(wire.clone(), descriptor(3))]),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
        };
        assert!(validate_backend_storage_contract(&[wire], &four_tower).is_err());
    }

    #[test]
    fn storage_contract_accepts_ordered_native_descriptor() {
        let ty = matrix_type();
        let wire = ConcreteWireType::Matrix(ty.clone());
        let contract = BackendStorageContract {
            descriptors: BTreeMap::from([(wire.clone(), descriptor(3))]),
            active_crt_towers: 4,
            crt_limb_bytes: 8,
        };
        validate_backend_storage_contract(&[wire], &contract).unwrap();
    }

    #[test]
    fn storage_representation_is_typed_and_unknown_values_fail_validation() {
        assert_eq!(BackendStorageRepresentation::full_dcrt().as_str(), "full_dcrt");
        assert_eq!(BackendStorageRepresentation::compact_bounded().as_str(), "compact_bounded");
        let ty = matrix_type();
        let wire = ConcreteWireType::Matrix(ty);
        let contract = BackendStorageContract {
            descriptors: BTreeMap::from([(
                wire.clone(),
                BackendStorageDescriptor {
                    representation: "future_encoding".into(),
                    ordered_crt_basis: vec![17, 19],
                    level: 1,
                    limb_bytes: 8,
                },
            )]),
            active_crt_towers: 2,
            crt_limb_bytes: 8,
        };
        assert!(validate_backend_storage_contract(&[wire], &contract).is_err());
    }
}

#[cfg(test)]
mod warmup_profile_tests {
    use super::*;
    use std::{
        cell::{Cell, RefCell},
        collections::HashMap,
    };

    #[derive(Default)]
    struct FakeProvider {
        calls: Cell<usize>,
        registered: RefCell<HashMap<GpuWarmupOperationSignature, CanonicalWarmupProfileDomain>>,
    }

    impl GpuWarmupProfileProvider for FakeProvider {
        fn register_operation(
            &mut self,
            descriptor: GpuWarmupOperationDescriptor,
        ) -> Result<(), GpuWarmupProfileError> {
            self.registered.borrow_mut().insert(descriptor.signature, descriptor.profile_domain);
            Ok(())
        }

        fn measure(
            &mut self,
            request: &GpuWarmupProfileRequest,
        ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
            self.calls.set(self.calls.get() + 1);
            // Deliberately nonlinear: the planner must be able to select a
            // smaller candidate after it has collected both points.
            let time = if request.tile_width == 2 { 1.0 } else { 4.0 };
            let domain = self
                .registered
                .borrow()
                .get(&request.signature)
                .copied()
                .unwrap_or(CanonicalWarmupProfileDomain::MatrixAdd);
            let workspace = if domain.measurement_kind() == WarmupMeasurementKind::HostMeasured {
                0
            } else {
                request.tile_width as u64 * 10
            };
            GpuWarmupProfile::measured_for_domain(domain, time, workspace)
        }
    }

    fn request(tile_width: usize) -> GpuWarmupProfileRequest {
        request_with_signature(
            tile_width,
            GpuWarmupOperationSignature { operation: [7; 32], shape_class: 3, instance_class: 1 },
        )
    }

    fn request_with_signature(
        tile_width: usize,
        signature: GpuWarmupOperationSignature,
    ) -> GpuWarmupProfileRequest {
        let device_identity = GpuWarmupDeviceIdentity::new(0, "gpu-0", "native-r1");
        GpuWarmupProfileRequest {
            signature,
            device: 0,
            device_identity,
            tile_width,
            range: IndexRange { start: 4, end: 4 + tile_width },
            executed_range_start: 4,
            executed_range_class: GpuWarmupFragmentClass::Whole,
            route: GpuWarmupRoute::DeviceLocal,
            route_descriptor: GpuExecutionRouteDescriptor::device_local(
                0,
                crate::gpu_column_policy::ColumnRange { start: 4, end: 4 + tile_width },
                crate::gpu_column_policy::GpuFragmentClass::Full,
            ),
            route_resolver: None,
            binding_port: None,
            fragment: GpuWarmupFragmentClass::Whole,
            retry_cap: None,
            cache_identity: None,
            cache_state: GpuWarmupCacheState::Warm,
            timing_scope: GpuWarmupTimingScope::LocalJob,
        }
    }

    #[test]
    fn canonical_inventory_is_backed_by_real_node_variants() {
        use crate::gpu_column_policy::canonical_warmup_profile_domain;
        use mxx_ir_core::{
            expr::RealExpr,
            node::{
                ConcatAxis, ConstantMatrix, HashTagComponent, HashVariant, IntBinaryOp,
                IntCompareOp, LoopInputMode, MatrixBinaryOp, NodeKind, ParallelLoop, RealBinaryOp,
                SampleRange, SequentialLoop, SubgraphCall,
            },
            types::{MatrixType, WireType},
        };

        // This table is deliberately made from the actual IR variants.  In
        // particular, it must not use one Input node with a different profile
        // label: that used to let the inventory test pass while production
        // representatives for constants, samplers, boundaries, and control
        // nodes were still absent.
        let one = || mxx_ir_core::IntExpr::constant(1);
        let matrix =
            || MatrixType { modulus: one(), ring_dimension: one(), rows: one(), columns: one() };
        let nodes = vec![
            NodeKind::Input { name: "input".into(), wire_type: WireType::Int, artifact: None },
            NodeKind::ConstantInt(1.into()),
            NodeKind::EvaluateInt(one()),
            NodeKind::ConstantReal(RealExpr::from_integer(1)),
            NodeKind::ConstantBool(true),
            NodeKind::ConstantMatrix { matrix_type: matrix(), value: ConstantMatrix::Zero },
            NodeKind::ConstantMatrix { matrix_type: matrix(), value: ConstantMatrix::Identity },
            NodeKind::ConstantMatrix {
                matrix_type: matrix(),
                value: ConstantMatrix::UnitRow { index: one() },
            },
            NodeKind::ConstantMatrix {
                matrix_type: matrix(),
                value: ConstantMatrix::UnitColumn { index: one() },
            },
            NodeKind::ConstantMatrix {
                matrix_type: matrix(),
                value: ConstantMatrix::Gadget { base: one(), small: false },
            },
            NodeKind::ConstantMatrix {
                matrix_type: matrix(),
                value: ConstantMatrix::PowerOfBase { base: one(), exponent: one() },
            },
            NodeKind::ConstantMatrix {
                matrix_type: matrix(),
                value: ConstantMatrix::Rotation { exponent: one() },
            },
            NodeKind::ConstantMatrix {
                matrix_type: matrix(),
                value: ConstantMatrix::Polynomial { coefficients: vec![one()] },
            },
            NodeKind::GadgetTrapdoor { matrix_type: matrix(), base: one() },
            NodeKind::TrapdoorPublic,
            NodeKind::IntBinary(IntBinaryOp::Add),
            NodeKind::IntCompare(IntCompareOp::Equal),
            NodeKind::BitExtract { bit: one() },
            NodeKind::IntToReal,
            NodeKind::BoolToInt,
            NodeKind::RealBinary(RealBinaryOp::Add),
            NodeKind::RealSqrt,
            NodeKind::MatrixBinary(MatrixBinaryOp::Add),
            NodeKind::MatrixBinary(MatrixBinaryOp::Subtract),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            NodeKind::MatrixMulAccumulate { coefficients: vec![one()], has_bias: true },
            NodeKind::MatrixMulSmallRhs,
            NodeKind::MatrixNegate,
            NodeKind::MatrixScale { scalar: one() },
            NodeKind::RingAutomorphism { index: one() },
            NodeKind::ModulusSwitch { modulus: one() },
            NodeKind::ModulusReduce { modulus: one() },
            NodeKind::CenteredRebase { modulus: one() },
            NodeKind::RnsModUp {
                modulus: one(),
                source_moduli: vec![17],
                digit_size: 1,
                normalize: false,
            },
            NodeKind::RnsModDown {
                modulus: one(),
                source_moduli: vec![17],
                plaintext_modulus: one(),
            },
            NodeKind::Transpose,
            NodeKind::Slice {
                rows: Some(mxx_ir_core::node::IndexRange { start: one(), end: one() + one() }),
                columns: None,
            },
            NodeKind::Tensor,
            NodeKind::Concat { axis: ConcatAxis::Rows },
            NodeKind::Concat { axis: ConcatAxis::Columns },
            NodeKind::Concat { axis: ConcatAxis::Diagonal },
            NodeKind::UniformResidueSample { matrix_type: matrix() },
            NodeKind::UniformIntervalSample {
                matrix_type: matrix(),
                range: SampleRange { minimum: one(), maximum: one() + one() },
            },
            NodeKind::GaussianSample {
                matrix_type: matrix(),
                sigma: RealExpr::from_integer(1),
                max_coefficient_bound: one(),
            },
            NodeKind::HashSample {
                matrix_type: matrix(),
                variant: HashVariant::Plain,
                tag_prefix: vec![1],
                tag_components: vec![HashTagComponent::Bytes(vec![1])],
                base: None,
                digit_count: None,
            },
            NodeKind::TrapdoorSample {
                matrix_type: matrix(),
                sigma: RealExpr::from_integer(1),
                gadget_base: one(),
                digit_count: one(),
                preimage_max_coefficient_bound: one(),
            },
            NodeKind::PreimageSample { matrix_type: matrix(), max_coefficient_bound: one() },
            NodeKind::GadgetDecompose { base: one(), small: false, digit_count: one() },
            NodeKind::ExtractCoefficient { position: one(), canonical_input_exclusive_upper: None },
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type: matrix() },
            NodeKind::ThresholdDecode {
                plaintext_modulus: one(),
                length: one(),
                output_bool: true,
            },
            NodeKind::CrtRecompose {
                modulus: one(),
                plaintext_moduli: vec![one()],
                reconstruction_coefficients: vec![one()],
            },
            NodeKind::PackPolynomialCoefficients { matrix_type: matrix(), coefficient_bits: one() },
            NodeKind::PolynomialFromValues { matrix_type: matrix(), evaluation: false },
            NodeKind::PolynomialValues { evaluation: false },
            NodeKind::SubgraphCall(SubgraphCall {
                definition: "case".into(),
                bindings: vec![],
                canonical_input_exclusive_uppers: vec![],
            }),
            NodeKind::ParallelLoop(ParallelLoop {
                count: one(),
                minimum_count: 1,
                index_slot: 0,
                bindings: vec![],
                input_modes: vec![LoopInputMode::Broadcast],
            }),
            NodeKind::SequentialLoop(SequentialLoop {
                count: one(),
                index_slot: 0,
                bindings: vec![],
                carried_count: 1,
            }),
            NodeKind::FamilyPack { count: one() },
            NodeKind::FamilyGetStatic { index: one() },
            NodeKind::FamilyGetDynamic,
            NodeKind::Select { count: one() },
        ];
        let expected = CanonicalWarmupProfileDomain::all()
            .iter()
            .copied()
            .filter(|domain| {
                !matches!(
                    domain,
                    CanonicalWarmupProfileDomain::FusedRowSum |
                        CanonicalWarmupProfileDomain::FusedTensorRowSum |
                        CanonicalWarmupProfileDomain::FusedRowBlockAdd |
                        CanonicalWarmupProfileDomain::FusedDecompose |
                        CanonicalWarmupProfileDomain::FusedCompactProduct |
                        CanonicalWarmupProfileDomain::FusedPreimageBatch
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(nodes.len(), expected.len(), "ordinary profile inventory drifted");
        for (node, domain) in nodes.iter().zip(expected) {
            assert_eq!(canonical_warmup_profile_domain(node), domain);
            assert!(domain.is_profileable());
        }
        assert_eq!(
            FusedWarmupOperation::all()
                .iter()
                .map(|operation| fused_warmup_profile_domain(*operation))
                .collect::<BTreeSet<_>>()
                .len(),
            FusedWarmupOperation::all().len(),
            "fused production variants must have distinct profile domains"
        );
    }

    #[test]
    fn profiles_without_measured_time_are_rejected() {
        let profile = GpuWarmupProfile {
            kind: GpuWarmupProfileKind::Measured,
            measurement: WarmupMeasurementKind::GpuMeasured,
            time_seconds: f64::NAN,
            workspace_bytes: 64,
            preimage_max_attempts: None,
            preimage_certified_tile_width: None,
            preimage_footprint: None,
            resolved_cache_identity: None,
            memory: GpuWarmupMemoryObservations::default(),
            resident_delta: GpuWarmupResidencyDelta::default(),
            repetitions: 1,
            spread_seconds: 0.0,
            provenance: GpuWarmupProvenance::ProductionEquivalent,
            cache_state: GpuWarmupCacheState::Warm,
            timing_scope: GpuWarmupTimingScope::LocalJob,
            resolved_route_descriptor: None,
        };
        assert!(profile.validate().is_err());
    }

    #[test]
    fn host_profiles_require_time_and_cannot_claim_gpu_workspace() {
        let profile = GpuWarmupProfile::host_measured(0.25).unwrap();
        assert_eq!(profile.measurement, WarmupMeasurementKind::HostMeasured);
        assert_eq!(profile.workspace_bytes, 0);
        assert!(
            GpuWarmupProfile::measured_for_domain(
                CanonicalWarmupProfileDomain::ConstantInt,
                0.25,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn measured_profiles_reject_zero_time_but_exact_accounting_is_explicit() {
        assert!(GpuWarmupProfile::host_measured(0.0).is_err());
        assert!(GpuWarmupProfile::measured(0.0, 0).is_err());
        assert!(
            GpuWarmupProfile::measured_for_domain(
                CanonicalWarmupProfileDomain::ConstantInt,
                0.0,
                0,
            )
            .is_err()
        );
        assert_eq!(
            GpuWarmupMemoryObservations::explicit_exact_zero().evidence,
            MemoryEvidenceKind::ExactQuery
        );
    }

    #[test]
    fn session_cache_reconstructs_exact_provider_result() {
        let mut provider = FakeProvider::default();
        let request = request(2);
        let profile = provider.measure(&request).unwrap();
        let key = GpuWarmupProfileKey {
            effective_domain: CanonicalWarmupProfileDomain::MatrixAdd,
            implementation_variant: GpuWarmupEffectiveVariant::Ordinary,
            operation_identity: request.signature.operation,
            noninterpolated_shape: Vec::new(),
            native_parameters: vec![
                request.signature.shape_class,
                request.signature.instance_class,
            ],
            device: GpuWarmupDeviceIdentity::new(0, "gpu-0", "native-r1"),
            executed_range_start: request.range.start,
            executed_range_class: request.executed_range_class,
            retry_cap: request.retry_cap,
            cache_identity: request.cache_identity,
            cache_state: request.cache_state,
            route: GpuWarmupRoute::DeviceLocal,
            route_descriptor: request.route_descriptor,
            binding_port: None,
            fragment: GpuWarmupFragmentClass::Whole,
            timing_scope: request.timing_scope,
        };
        let mut session = GpuWarmupSessionProfileCache::new();
        session.insert_profile(key.clone(), request.tile_width, request.range, profile).unwrap();
        let cached = session.exact_profile(&key, 2).unwrap().unwrap();
        assert_eq!(cached.time_seconds, 1.0);
        assert_eq!(cached.workspace_bytes, 20);
        assert_eq!(provider.calls.get(), 1);
    }

    #[test]
    fn session_cache_exact_request_requires_matching_binding_port_in_both_directions() {
        let make_key = |request: &GpuWarmupProfileRequest| GpuWarmupProfileKey {
            effective_domain: CanonicalWarmupProfileDomain::MatrixAdd,
            implementation_variant: GpuWarmupEffectiveVariant::Ordinary,
            operation_identity: request.signature.operation,
            noninterpolated_shape: Vec::new(),
            native_parameters: vec![
                request.signature.shape_class,
                request.signature.instance_class,
            ],
            device: request.device_identity.clone(),
            executed_range_start: request.executed_range_start,
            executed_range_class: request.executed_range_class,
            retry_cap: request.retry_cap,
            cache_identity: request.cache_identity,
            cache_state: request.cache_state,
            route: request.route,
            route_descriptor: request.route_descriptor,
            binding_port: request.binding_port,
            fragment: request.fragment,
            timing_scope: request.timing_scope,
        };

        for (stored_port, mismatching_port) in [(Some(0), Some(1)), (Some(1), Some(0))] {
            let mut stored_request = request(2);
            stored_request.binding_port = stored_port;
            let key = make_key(&stored_request);
            let profile =
                GpuWarmupProfile::measured_for_domain(key.effective_domain, 1.0, 20).unwrap();
            let mut cache = GpuWarmupSessionProfileCache::new();
            cache
                .insert_profile(
                    key.clone(),
                    stored_request.tile_width,
                    stored_request.range.clone(),
                    profile,
                )
                .unwrap();

            assert!(cache.exact_profile_for_request(&key, &stored_request).unwrap().is_some());

            let mut mismatching_request = stored_request.clone();
            mismatching_request.binding_port = mismatching_port;
            assert!(matches!(
                cache.exact_profile_for_request(&key, &mismatching_request),
                Err(GpuWarmupProfileError::InvalidMeasurement(_))
            ));
        }
    }
}

#[cfg(test)]
mod canonical_profile_table_tests {
    use super::*;

    fn key() -> GpuWarmupProfileKey {
        GpuWarmupProfileKey {
            effective_domain: CanonicalWarmupProfileDomain::MatrixAdd,
            implementation_variant: GpuWarmupEffectiveVariant::Ordinary,
            operation_identity: [11; 32],
            noninterpolated_shape: vec![4, 32],
            native_parameters: vec![64, 8],
            device: GpuWarmupDeviceIdentity::new(0, "gpu-0", "native-r1"),
            executed_range_start: 2,
            executed_range_class: GpuWarmupFragmentClass::Whole,
            retry_cap: None,
            cache_identity: None,
            cache_state: GpuWarmupCacheState::Warm,
            route: GpuWarmupRoute::DeviceLocal,
            route_descriptor: GpuExecutionRouteDescriptor::device_local(
                0,
                crate::gpu_column_policy::ColumnRange { start: 2, end: 6 },
                crate::gpu_column_policy::GpuFragmentClass::Full,
            ),
            binding_port: None,
            fragment: GpuWarmupFragmentClass::Whole,
            timing_scope: GpuWarmupTimingScope::LocalJob,
        }
    }

    fn point(
        key: GpuWarmupProfileKey,
        coordinate: usize,
        mean_seconds: f64,
        evidence: MemoryEvidenceKind,
    ) -> GpuWarmupProfilePoint {
        GpuWarmupProfilePoint::new(
            key,
            coordinate,
            IndexRange { start: 2, end: 2 + coordinate },
            GpuWarmupFragmentClass::Whole,
            3,
            mean_seconds,
            0.01,
            GpuWarmupMemoryObservations {
                affected_devices: BTreeMap::from([(
                    GpuWarmupDeviceIdentity::new(0, "gpu-0", "native-r1"),
                    coordinate as u64 * 10,
                )]),
                host_bytes: coordinate as u64,
                pinned_host_bytes: coordinate as u64 * 2,
                evidence,
            },
            GpuWarmupResidencyDelta::default(),
            GpuWarmupProvenance::ProductionEquivalent,
        )
        .unwrap()
    }

    #[test]
    fn l01_exact_point_is_returned_without_interpolation() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class, 4, 0.25, MemoryEvidenceKind::ExactQuery)).unwrap();
        let resolved = table.resolve(4).unwrap();
        assert_eq!(resolved.time_seconds, 0.25);
        assert!(!resolved.interpolated);
    }

    #[test]
    fn l02_only_validated_adjacent_interval_is_interpolated() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class.clone(), 2, 0.2, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.insert_point(point(class.clone(), 6, 0.6, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.validate_interval(2, 6, []).unwrap();
        let resolved = table.resolve(4).unwrap();
        assert_eq!(resolved.time_seconds, 0.4);
        assert!(resolved.interpolated);
    }

    #[test]
    fn l03_single_anchor_and_extrapolation_are_rejected() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class, 4, 0.4, MemoryEvidenceKind::ExactQuery)).unwrap();
        assert!(matches!(table.resolve(8), Err(GpuWarmupProfileTableError::NoValidatedInterval)));
    }

    #[test]
    fn l04_different_execution_class_cannot_enter_table() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        let mut other = class;
        other.cache_state = GpuWarmupCacheState::Cold;
        assert!(matches!(
            table.insert_point(point(other, 4, 0.4, MemoryEvidenceKind::ExactQuery)),
            Err(GpuWarmupProfileTableError::IncompatibleKey)
        ));
    }

    #[test]
    fn l05_point_requires_positive_coordinate_range_repetitions_and_time() {
        let class = key();
        assert!(
            GpuWarmupProfilePoint::new(
                class,
                0,
                IndexRange { start: 1, end: 2 },
                GpuWarmupFragmentClass::Whole,
                1,
                0.1,
                0.0,
                GpuWarmupMemoryObservations::default(),
                GpuWarmupResidencyDelta::default(),
                GpuWarmupProvenance::ProductionEquivalent,
            )
            .is_err()
        );
    }

    #[test]
    fn l06_memory_interpolation_uses_checked_ceil() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class.clone(), 2, 0.2, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.insert_point(point(class.clone(), 5, 0.5, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.validate_interval(2, 5, []).unwrap();
        let resolved = table.resolve(3).unwrap();
        // The anchors are 20 bytes at width 2 and 50 bytes at width 5;
        // width 3 is the exact affine value 30 (the checked-ceil rule also
        // applies to the host/pinned components below).
        assert_eq!(resolved.memory.affected_devices.values().next().copied(), Some(30));
        assert_eq!(resolved.memory.evidence, MemoryEvidenceKind::LinearEstimate);
    }

    #[test]
    fn l07_hard_admission_rejects_interpolated_memory() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class.clone(), 2, 0.2, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.insert_point(point(class.clone(), 5, 0.5, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.validate_interval(2, 5, []).unwrap();
        assert!(matches!(
            table.resolve_for_admission(3),
            Err(GpuWarmupProfileTableError::NonAdmissibleMemory(
                MemoryEvidenceKind::LinearEstimate
            ))
        ));
    }

    #[test]
    fn l08_hard_admission_accepts_exact_and_certified_memory_only() {
        let class = key();
        for evidence in [MemoryEvidenceKind::ExactQuery, MemoryEvidenceKind::CertifiedEnvelope] {
            let mut table = GpuWarmupProfileTable::new(class.clone());
            table.insert_point(point(class.clone(), 4, 0.4, evidence)).unwrap();
            assert!(table.resolve_for_admission(4).is_ok());
        }
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class, 4, 0.4, MemoryEvidenceKind::MeasuredPeak)).unwrap();
        assert!(table.resolve_for_admission(4).is_err());
    }

    #[test]
    fn l14_session_cache_has_one_table_authority_per_execution_class() {
        let class = key();
        let mut cache = GpuWarmupSessionProfileCache::new();
        let mut measured = point(class.clone(), 4, 0.4, MemoryEvidenceKind::ExactQuery);
        measured.resolved_route_descriptor = Some(GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 4 },
            crate::gpu_column_policy::GpuFragmentClass::Full,
        ));
        let physical_route = measured.resolved_route_descriptor;
        cache.insert_point(measured).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.resolve(&class, 4).unwrap().time_seconds, 0.4);
        assert_eq!(
            cache.exact_profile(&class, 4).unwrap().unwrap().resolved_route_descriptor,
            physical_route
        );
        assert!(cache.resolve(&class, 8).is_err());
    }

    #[test]
    fn l04_duplicate_coordinates_and_batch_conflicts_fail_closed() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class.clone(), 2, 0.2, MemoryEvidenceKind::ExactQuery)).unwrap();
        assert!(matches!(
            table.insert_point(point(class.clone(), 2, 0.3, MemoryEvidenceKind::ExactQuery)),
            Err(GpuWarmupProfileTableError::DuplicateCoordinate)
        ));

        let mut cache = GpuWarmupSessionProfileCache::new();
        let result = cache.insert_points([
            point(class.clone(), 2, 0.2, MemoryEvidenceKind::ExactQuery),
            point(class, 2, 0.3, MemoryEvidenceKind::ExactQuery),
        ]);
        assert!(matches!(result, Err(GpuWarmupProfileTableError::DuplicateCoordinate)));
        assert!(cache.is_empty());
    }

    #[test]
    fn l05_failed_holdout_is_typed_and_can_be_promoted() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class.clone(), 2, 0.2, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.insert_point(point(class.clone(), 4, 0.9, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.insert_point(point(class, 6, 0.6, MemoryEvidenceKind::ExactQuery)).unwrap();
        let result = table
            .validate_interval_with_policy(
                2,
                6,
                [4],
                GpuWarmupValidationTolerance::new(0.0, 0.01).unwrap(),
            )
            .unwrap();
        assert_eq!(result.holdout_coordinate_to_promote(), Some(4));
        assert!(table.intervals().is_empty());
        table.promote_holdout(4).unwrap();
        assert!(table.active_knots().contains(&4));
    }

    #[test]
    fn l06_holdout_validation_requires_finite_nonnegative_policy() {
        assert!(GpuWarmupValidationTolerance::new(f64::NAN, 0.1).is_err());
        assert!(GpuWarmupValidationTolerance::new(0.1, f64::INFINITY).is_err());
        assert!(GpuWarmupValidationTolerance::new(-0.1, 0.1).is_err());
    }

    #[test]
    fn l07_profile_keeps_exact_memory_evidence_and_single_sample_spread() {
        let device = GpuWarmupDeviceIdentity::new(0, "gpu-0", "native-r1");
        let memory = GpuWarmupMemoryObservations {
            affected_devices: BTreeMap::from([(device, 4096)]),
            host_bytes: 11,
            pinned_host_bytes: 13,
            evidence: MemoryEvidenceKind::ExactQuery,
        };
        let profile = GpuWarmupProfile::measured_with_observation(
            0.25,
            4096,
            WarmupMeasurementKind::GpuMeasured,
            memory.clone(),
            GpuWarmupResidencyDelta::default(),
            1,
            0.0,
            GpuWarmupProvenance::ProductionEquivalent,
            GpuWarmupCacheState::Cold,
            GpuWarmupTimingScope::Setup,
        )
        .unwrap();
        assert_eq!(profile.memory, memory);
        assert_eq!(profile.memory.evidence, MemoryEvidenceKind::ExactQuery);
        assert_eq!(profile.repetitions, 1);
        assert_eq!(profile.spread_seconds, 0.0);
        assert_eq!(profile.cache_state, GpuWarmupCacheState::Cold);
        assert_eq!(profile.timing_scope, GpuWarmupTimingScope::Setup);
    }

    #[test]
    fn l08_interval_uses_measured_holdout_time_not_coordinate_only() {
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class.clone(), 2, 0.2, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.insert_point(point(class.clone(), 4, 0.4, MemoryEvidenceKind::ExactQuery)).unwrap();
        table.insert_point(point(class, 6, 0.6, MemoryEvidenceKind::ExactQuery)).unwrap();
        let validation = table
            .validate_interval_with_policy(2, 6, [4], GpuWarmupValidationTolerance::default())
            .unwrap();
        assert!(validation.is_validated());
        assert!(table.resolve(5).is_ok());
    }

    #[test]
    fn p10_route_and_fragment_context_are_distinct_cache_classes() {
        let base = key();
        let mut peer = base.clone();
        peer.route = GpuWarmupRoute::PeerToPeer;
        peer.route_descriptor.route = crate::gpu_column_policy::GpuTransferRoute::Peer;
        peer.route_descriptor.source_device = Some(1);
        peer.route_descriptor.destination_device = Some(0);
        let mut tail = base.clone();
        tail.fragment = GpuWarmupFragmentClass::Tail;
        assert!(!base.execution_class_matches(&peer));
        assert!(!base.execution_class_matches(&tail));

        let mut cache = GpuWarmupSessionProfileCache::new();
        cache.insert_point(point(base.clone(), 4, 0.4, MemoryEvidenceKind::ExactQuery)).unwrap();
        assert!(cache.resolve(&peer, 4).is_err());
        assert!(cache.resolve(&tail, 4).is_err());
    }

    #[test]
    fn p13_context_generation_invalidates_native_profile_class() {
        let base = key();
        let mut restarted = base.clone();
        restarted.device.context_generation = 1;
        assert!(!base.execution_class_matches(&restarted));

        let mut cache = GpuWarmupSessionProfileCache::new();
        cache.insert_point(point(base.clone(), 4, 0.4, MemoryEvidenceKind::ExactQuery)).unwrap();
        assert!(cache.resolve(&restarted, 4).is_err());
    }

    #[test]
    fn preimage_session_identity_requires_same_owner_and_rejects_native_mismatch() {
        let mut cold = key();
        cold.effective_domain = CanonicalWarmupProfileDomain::PreimageSample;
        cold.cache_identity = Some([7; 32]);
        let signature = GpuWarmupOperationSignature {
            operation: cold.operation_identity,
            shape_class: cold.native_parameters[0],
            instance_class: cold.native_parameters[1],
        };
        let mut cache = GpuWarmupSessionProfileCache::new();
        cache.insert_point(point(cold.clone(), 4, 0.4, MemoryEvidenceKind::ExactQuery)).unwrap();
        let device = cold.device.clone();
        assert_eq!(cache.preimage_cache_identity(signature, &device).unwrap(), Some([7; 32]));

        let mut different_owner = cold;
        different_owner.cache_identity = Some([8; 32]);
        cache.insert_point(point(different_owner, 4, 0.4, MemoryEvidenceKind::ExactQuery)).unwrap();
        assert!(cache.preimage_cache_identity(signature, &device).is_err());
    }

    #[test]
    fn empty_memory_evidence_is_not_exact_by_default() {
        assert_eq!(
            GpuWarmupMemoryObservations::default().evidence,
            MemoryEvidenceKind::Unspecified
        );
        let class = key();
        let mut table = GpuWarmupProfileTable::new(class.clone());
        table.insert_point(point(class, 4, 0.4, MemoryEvidenceKind::Unspecified)).unwrap();
        assert!(matches!(
            table.resolve_for_admission(4),
            Err(GpuWarmupProfileTableError::NonAdmissibleMemory(MemoryEvidenceKind::Unspecified))
        ));
    }

    #[test]
    fn exact_zero_memory_is_host_only_and_gpu_exact_requires_device() {
        let host = GpuWarmupProfile::measured_with_observation(
            0.1,
            0,
            WarmupMeasurementKind::HostMeasured,
            GpuWarmupMemoryObservations::explicit_exact_zero(),
            GpuWarmupResidencyDelta::default(),
            1,
            0.0,
            GpuWarmupProvenance::ProductionEquivalent,
            GpuWarmupCacheState::Warm,
            GpuWarmupTimingScope::ContainingStage,
        )
        .unwrap();
        assert_eq!(host.memory.evidence, MemoryEvidenceKind::ExactQuery);
        let mut transfer = host.clone();
        transfer.timing_scope = GpuWarmupTimingScope::Transfer;
        transfer.memory.affected_devices.insert(key().device, 128);
        transfer.memory.host_bytes = 128;
        transfer.memory.evidence = MemoryEvidenceKind::CertifiedEnvelope;
        assert!(transfer.clone().validate().is_ok());
        transfer.timing_scope = GpuWarmupTimingScope::ContainingStage;
        assert!(transfer.validate().is_err());

        assert!(
            GpuWarmupProfile::measured_with_observation(
                0.1,
                0,
                WarmupMeasurementKind::GpuMeasured,
                GpuWarmupMemoryObservations::explicit_exact_zero(),
                GpuWarmupResidencyDelta::default(),
                1,
                0.0,
                GpuWarmupProvenance::ProductionEquivalent,
                GpuWarmupCacheState::Warm,
                GpuWarmupTimingScope::LocalJob,
            )
            .is_err()
        );
    }

    #[test]
    fn range_class_mismatch_cannot_enter_session_cache() {
        let class = key();
        let mut point = point(class, 4, 0.4, MemoryEvidenceKind::ExactQuery);
        point.executed_range_class = GpuWarmupFragmentClass::Tail;
        let mut cache = GpuWarmupSessionProfileCache::new();
        assert!(matches!(
            cache.insert_point(point),
            Err(GpuWarmupProfileTableError::IncompatibleKey)
        ));
    }
}
