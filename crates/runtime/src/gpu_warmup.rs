//! Pure warmup planning for the fixed GPU execution path.
//!
//! The planner in this module is deliberately value-only.  It consumes fake
//! profiles (or profiles produced by the GPU measurement adapter), chooses a
//! joint loop wave/tile-width candidate, and emits the small metadata plan
//! consumed by the executor.  It does not query CUDA, own a buffer, or retain
//! an input value.  Keeping this boundary pure makes the resource accounting
//! testable on machines without a GPU and prevents production from silently
//! re-planning on an OOM or a changed allocator state.

use crate::{
    backend::{
        BackendStorageContract, BackendStorageDescriptor, validate_backend_storage_contract,
    },
    gpu_column_policy::{
        ColumnCapability, ColumnRange, EffectiveGpuOperation, capability_for_effective_operation,
        effective_gpu_operation, map_output_range_to_inputs_with_output,
    },
    gpu_execution_plan::{
        FrozenGpuPlan, GpuDeviceBudget, GpuExecutionSiteKey, GpuLayout, GpuLoopChoice,
        GpuLoopSiteKey, GpuNodeChoice, GpuPlanContract, LayoutId, scope_shape_class,
    },
    gpu_schedule::{GpuColumnInterval, GpuColumnJob, GpuColumnSchedule, GpuScheduleError},
};
#[cfg(not(feature = "gpu"))]
use mxx_ir_core::encoding;
use mxx_ir_core::{
    ValidatedGraph,
    encoding::spec_hash,
    graph::FrozenGraphScopeId,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt,
    num::NonZeroUsize,
};

/// The resource classes charged to a stage.  Keeping the classes explicit is
/// useful in reports: a candidate that fits only by reducing `b` must not hide
/// a retained output, replica, or cache behind a generic workspace number.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuResourceCost {
    pub live: u64,
    pub outputs: u64,
    pub replicas: u64,
    pub caches: u64,
    pub transfers: u64,
    pub scratch: u64,
    pub pinned_host: u64,
    pub host: u64,
}

impl GpuResourceCost {
    pub const fn zero() -> Self {
        Self {
            live: 0,
            outputs: 0,
            replicas: 0,
            caches: 0,
            transfers: 0,
            scratch: 0,
            pinned_host: 0,
            host: 0,
        }
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(Self {
            live: self.live.checked_add(rhs.live)?,
            outputs: self.outputs.checked_add(rhs.outputs)?,
            replicas: self.replicas.checked_add(rhs.replicas)?,
            caches: self.caches.checked_add(rhs.caches)?,
            transfers: self.transfers.checked_add(rhs.transfers)?,
            scratch: self.scratch.checked_add(rhs.scratch)?,
            pinned_host: self.pinned_host.checked_add(rhs.pinned_host)?,
            host: self.host.checked_add(rhs.host)?,
        })
    }

    pub fn checked_mul(self, factor: usize) -> Option<Self> {
        let factor = u64::try_from(factor).ok()?;
        Some(Self {
            live: self.live.checked_mul(factor)?,
            outputs: self.outputs.checked_mul(factor)?,
            replicas: self.replicas.checked_mul(factor)?,
            caches: self.caches.checked_mul(factor)?,
            transfers: self.transfers.checked_mul(factor)?,
            scratch: self.scratch.checked_mul(factor)?,
            pinned_host: self.pinned_host.checked_mul(factor)?,
            host: self.host.checked_mul(factor)?,
        })
    }

    pub fn device_bytes(self) -> u64 {
        self.live
            .saturating_add(self.outputs)
            .saturating_add(self.replicas)
            .saturating_add(self.caches)
            .saturating_add(self.transfers)
            .saturating_add(self.scratch)
    }

    pub fn total_bytes(self) -> u64 {
        self.device_bytes().saturating_add(self.pinned_host).saturating_add(self.host)
    }
}

/// Per-instance and per-job cost model.  The model is intentionally
/// conservative and additive.  `fixed` includes resident/setup state; the
/// other terms describe resources whose lifetime overlaps a wave or a local
/// column job.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuStageCostModel {
    pub fixed: GpuResourceCost,
    pub per_instance: GpuResourceCost,
    pub per_output_column: GpuResourceCost,
    pub per_job: GpuResourceCost,
    /// Width-dependent temporary storage; unlike retained output storage,
    /// this is live for only the current device-local job.
    pub per_job_column: GpuResourceCost,
    pub time: GpuTimeModel,
}

impl GpuStageCostModel {
    fn peak_for(
        &self,
        wave_instances: usize,
        output_columns: usize,
        local_jobs: usize,
        job_columns: usize,
    ) -> Result<GpuResourceCost, GpuWarmupError> {
        let mut peak = self.fixed;
        peak = peak
            .checked_add(
                self.per_instance
                    .checked_mul(wave_instances)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?,
            )
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        peak = peak
            .checked_add(
                self.per_output_column
                    .checked_mul(output_columns)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?,
            )
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        peak = peak
            .checked_add(
                self.per_job_column
                    .checked_mul(job_columns)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?,
            )
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        peak.checked_add(
            self.per_job.checked_mul(local_jobs).ok_or(GpuWarmupError::ArithmeticOverflow)?,
        )
        .ok_or(GpuWarmupError::ArithmeticOverflow)
    }
}

/// Time of one local job.  Device jobs in one logical wave run concurrently;
/// jobs sharing a device are summed in deterministic schedule order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuTimeModel {
    pub fixed_seconds: f64,
    pub per_column_seconds: f64,
    /// Optional nonlinear penalty for very wide tiles.  It models the common
    /// case where throughput improvement flattens after a kernel's preferred
    /// tile size, and is what lets a smaller feasible `b` win T29.
    pub per_column_squared_seconds: f64,
    pub per_job_seconds: f64,
    pub transfer_seconds: f64,
    pub wave_overhead_seconds: f64,
}

impl GpuTimeModel {
    pub fn job_seconds(self, columns: usize) -> f64 {
        self.fixed_seconds +
            self.per_column_seconds * columns as f64 +
            self.per_column_squared_seconds * (columns as f64) * (columns as f64) +
            self.per_job_seconds +
            self.transfer_seconds
    }
}

/// Where a profile came from.  This is diagnostic metadata only; it never
/// carries a secret or a random seed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GpuProfileProvenance {
    SizeQuery,
    MeasuredPoint,
    ConservativeEstimate,
}

impl Default for GpuProfileProvenance {
    fn default() -> Self {
        Self::ConservativeEstimate
    }
}

/// A fake or measured operation profile consumed by [`plan_gpu_warmup`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuWarmupNode {
    pub key: GpuExecutionSiteKey,
    pub operation_identity: [u8; 32],
    pub effective_operation: EffectiveGpuOperation,
    pub column_capability: ColumnCapability,
    pub output_layouts: Vec<LayoutId>,
    pub implementation_variant: String,
    pub loop_site: Option<GpuLoopSiteKey>,
    pub output_columns: usize,
    /// Candidate local tile widths.  The planner evaluates every supplied
    /// width; it never assumes that the largest feasible width is optimal.
    pub tile_widths: Vec<usize>,
    pub cost: Vec<GpuStageCostModel>,
    pub provenance: GpuProfileProvenance,
    pub preimage_max_attempts: Option<usize>,
    /// Cold preimage allocation footprint. Present only for preimage sites;
    /// production planning refuses to emit a preimage plan without it.
    pub preimage_footprint: Option<Vec<GpuPreimageFootprint>>,
}

/// A loop shape and its resource context.  Nested loops intentionally force
/// `wave_instances == 1`; their parent wave's live cost remains in the
/// profile's fixed/per-instance terms.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuWarmupLoop {
    pub key: GpuLoopSiteKey,
    pub loop_count: usize,
    pub wave_candidates: Vec<usize>,
    pub nested: bool,
}

/// Input to the pure planner.  Layout owner intervals are preserved when
/// supplied.  Empty intervals are filled with a deterministic contiguous
/// balanced layout, which is useful for fake profiles and first-use values.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuWarmupInput {
    pub contract: GpuPlanContract,
    pub layouts: Vec<GpuLayout>,
    pub loops: Vec<GpuWarmupLoop>,
    pub nodes: Vec<GpuWarmupNode>,
}

/// Profile material supplied by the measurement adapter for one effective
/// operation.  The graph-derived entry point below combines this with alias,
/// fusion, and liveness information from [`ValidatedGraph`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuValidatedNodeProfile {
    pub tile_widths: Vec<usize>,
    pub cost: Vec<GpuStageCostModel>,
    pub implementation_variant: String,
    pub output_layout: LayoutId,
    pub provenance: GpuProfileProvenance,
    pub preimage_max_attempts: Option<usize>,
    pub preimage_footprint: Option<Vec<GpuPreimageFootprint>>,
}

/// Allocation classes measured by the production preimage dispatch. These are
/// deliberately separate so a cold trapdoor/cache path cannot be hidden in a
/// generic scratch estimate. Values are per logical device for one stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuPreimageFootprint {
    pub persistent: GpuResourceCost,
    pub compact: GpuResourceCost,
    pub scratch: GpuResourceCost,
    pub control: GpuResourceCost,
    pub cold_cache: GpuResourceCost,
    /// Temporary covariance workspace held only while a cold cache is built.
    /// It is distinct from the retained cache so warm steady-state accounting
    /// does not charge the transient allocation as a permanent resource.
    pub cold_transient_workspace: GpuResourceCost,
    /// The footprint is certified for this local tile width. A width sizing
    /// term may additionally charge each tile when a candidate uses a larger
    /// `b`; this prevents one measured preimage allocation from being reused
    /// optimistically for every candidate.
    pub certified_tile_width: Option<usize>,
    pub per_tile: GpuResourceCost,
    /// Primitive-derived conservative sizing model. Widths above the
    /// certified tile are rejected unless this model was supplied.
    pub width_sizing_model: Option<GpuResourceCost>,
    pub provenance: GpuProfileProvenance,
}

impl GpuPreimageFootprint {
    fn as_resource_cost_for_width(
        self,
        width: usize,
        site: GpuExecutionSiteKey,
    ) -> Result<GpuResourceCost, GpuWarmupError> {
        let mut cost = self
            .persistent
            .checked_add(self.compact)
            .and_then(|cost| cost.checked_add(self.scratch))
            .and_then(|cost| cost.checked_add(self.control))
            .and_then(|cost| cost.checked_add(self.cold_cache))
            .and_then(|cost| cost.checked_add(self.cold_transient_workspace))
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        let certified = self
            .certified_tile_width
            .filter(|certified| *certified > 0)
            .ok_or(GpuWarmupError::MissingPreimageProfile { site })?;
        let tile_count = width.max(1).div_ceil(certified);
        if tile_count > 1 && self.width_sizing_model.is_none() {
            return Err(GpuWarmupError::MissingPreimageProfile { site });
        }
        let per_tile = self.width_sizing_model.unwrap_or(self.per_tile);
        cost = cost
            .checked_add(
                per_tile.checked_mul(tile_count).ok_or(GpuWarmupError::ArithmeticOverflow)?,
            )
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        Ok(cost)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuValidatedWarmupConfig {
    pub contract: GpuPlanContract,
    pub layouts: Vec<GpuLayout>,
    pub default_tile_widths: Vec<usize>,
    pub default_cost: Vec<GpuStageCostModel>,
    pub default_implementation_variant: String,
    pub profiles: BTreeMap<[u8; 32], GpuValidatedNodeProfile>,
    /// Effective identities supplied by the production alias/fusion pass.
    /// The key is `(scope_shape_class, local_node_id)`; omitting a key uses
    /// the ordinary validated operation identity.
    pub effective_operation_identities: BTreeMap<(u64, u64), [u8; 32]>,
    pub effective_operations: BTreeMap<(u64, u64), EffectiveGpuOperation>,
    /// Concrete storage descriptors keyed by the exact validated wire type.
    /// This preserves ordered CRT basis/level differences across RNS inputs,
    /// outputs, retained values, and transfer accounting.
    pub storage_descriptors: BTreeMap<ConcreteWireType, GpuPhysicalStorageDescriptor>,
    /// Ordered active CRT towers and native limb width supplied by the
    /// concrete backend descriptor.
    pub active_crt_towers: usize,
    pub crt_limb_bytes: usize,
    /// Effective sibling-wave upper bound inherited from
    /// [`crate::executor::ExecutionConfig`].
    /// It is part of warmup admission, not a post-hoc executor hint.
    pub max_parallel_instances: NonZeroUsize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GpuStorageRepresentation {
    FullDcrt,
    CompactBounded,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuPhysicalStorageDescriptor {
    pub representation: GpuStorageRepresentation,
    pub ordered_basis: Vec<u64>,
    pub level: usize,
    pub limb_bytes: usize,
}

impl GpuPhysicalStorageDescriptor {
    fn active_towers(&self) -> usize {
        self.level.saturating_add(1)
    }
}

fn valid_storage_descriptor(descriptor: &GpuPhysicalStorageDescriptor) -> bool {
    !descriptor.ordered_basis.is_empty() &&
        descriptor.level < descriptor.ordered_basis.len() &&
        descriptor.limb_bytes > 0
}

fn validated_storage_wire_types(validated: &ValidatedGraph) -> BTreeSet<ConcreteWireType> {
    fn visit(ty: &ConcreteWireType, types: &mut BTreeSet<ConcreteWireType>) {
        types.insert(ty.clone());
        if let ConcreteWireType::IndexedFamily { element, .. } = ty {
            visit(element, types);
        }
    }

    let mut types = BTreeSet::new();
    for scope in validated.scopes.values() {
        for ty in scope.wire_types.values() {
            visit(ty, &mut types);
        }
    }
    types
}

fn backend_descriptor_to_gpu(
    descriptor: &BackendStorageDescriptor,
) -> Result<GpuPhysicalStorageDescriptor, String> {
    let representation = match descriptor.representation.as_str() {
        "full_dcrt" => GpuStorageRepresentation::FullDcrt,
        "compact_bounded" => GpuStorageRepresentation::CompactBounded,
        other => return Err(format!("unsupported backend storage representation {other:?}")),
    };
    let converted = GpuPhysicalStorageDescriptor {
        representation,
        ordered_basis: descriptor.ordered_crt_basis.clone(),
        level: descriptor.level,
        limb_bytes: descriptor.limb_bytes,
    };
    valid_storage_descriptor(&converted)
        .then_some(converted)
        .ok_or_else(|| "backend storage descriptor is structurally invalid".into())
}

fn caller_storage_metadata_matches(
    config: &GpuValidatedWarmupConfig,
    wire_types: &BTreeSet<ConcreteWireType>,
    backend: &BackendStorageContract,
) -> Result<(), String> {
    // An empty map is the deliberately supported "backend fills this in"
    // form. Any non-empty caller map is treated as an assertion and must be
    // complete, so a stale one-tower descriptor cannot be silently accepted.
    if config.storage_descriptors.is_empty() {
        return Ok(());
    }
    if config.active_crt_towers != backend.active_crt_towers ||
        config.crt_limb_bytes != backend.crt_limb_bytes
    {
        return Err("caller GPU storage summary disagrees with backend contract".into());
    }
    let matrix_wires =
        wire_types.iter().filter(|ty| ty.matrix_type().is_some()).collect::<Vec<_>>();
    if config.storage_descriptors.len() != matrix_wires.len() {
        return Err("caller GPU storage descriptor map is incomplete".into());
    }
    for wire in matrix_wires {
        let supplied = config
            .storage_descriptors
            .get(wire)
            .ok_or_else(|| "caller GPU storage descriptor map is incomplete".to_owned())?;
        let matrix = wire.matrix_type().expect("filtered matrix wire");
        let expected = backend
            .descriptors
            .get(matrix)
            .ok_or_else(|| "backend storage contract omits a concrete matrix type".to_owned())?;
        let expected = backend_descriptor_to_gpu(expected)?;
        if *supplied != expected {
            return Err("caller GPU storage descriptor does not match backend contract".into());
        }
    }
    Ok(())
}

fn apply_backend_storage_contract<B: crate::Backend>(
    validated: &ValidatedGraph,
    backend: &B,
    config: &mut GpuValidatedWarmupConfig,
) -> Result<(), GpuWarmupError> {
    let wire_types = validated_storage_wire_types(validated);
    let matrix_types = wire_types
        .iter()
        .filter_map(ConcreteWireType::matrix_type)
        .cloned()
        .collect::<BTreeSet<_>>();
    if matrix_types.is_empty() {
        config.storage_descriptors.clear();
        return Ok(());
    }
    let matrix_types = matrix_types.into_iter().collect::<Vec<_>>();
    let contract = backend
        .gpu_physical_storage_contract(&matrix_types)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
        .ok_or_else(|| {
            GpuWarmupError::ValidatedGraph(
                "backend has no authoritative GPU physical storage contract".into(),
            )
        })?;
    validate_backend_storage_contract(&matrix_types, &contract)
        .map_err(GpuWarmupError::ValidatedGraph)?;
    backend
        .validate_gpu_physical_storage_contract(&matrix_types, &contract)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    caller_storage_metadata_matches(config, &wire_types, &contract)
        .map_err(GpuWarmupError::ValidatedGraph)?;

    let mut descriptors = BTreeMap::new();
    for wire in wire_types.iter().filter(|ty| ty.matrix_type().is_some()) {
        let matrix = wire.matrix_type().expect("filtered matrix wire");
        let descriptor = contract.descriptors.get(matrix).ok_or_else(|| {
            GpuWarmupError::ValidatedGraph(
                "backend storage contract omits a concrete matrix type".into(),
            )
        })?;
        descriptors.insert(
            wire.clone(),
            backend_descriptor_to_gpu(descriptor).map_err(GpuWarmupError::ValidatedGraph)?,
        );
    }
    config.storage_descriptors = descriptors;
    config.active_crt_towers = contract.active_crt_towers;
    config.crt_limb_bytes = contract.crt_limb_bytes;
    Ok(())
}

fn validated_wire_bytes(
    ty: &ConcreteWireType,
    descriptors: &BTreeMap<ConcreteWireType, GpuPhysicalStorageDescriptor>,
    active_crt_towers: usize,
    crt_limb_bytes: usize,
) -> u64 {
    let fallback_representation =
        if matches!(ty, ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. }) {
            GpuStorageRepresentation::CompactBounded
        } else {
            GpuStorageRepresentation::FullDcrt
        };
    let (representation, towers, limb_bytes) = descriptors.get(ty).map_or(
        (fallback_representation, active_crt_towers, crt_limb_bytes),
        |descriptor| {
            (descriptor.representation.clone(), descriptor.active_towers(), descriptor.limb_bytes)
        },
    );
    match ty {
        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Trapdoor { matrix, .. } => {
            (matrix.rows as u64)
                .saturating_mul(matrix.columns as u64)
                .saturating_mul(matrix.ring_dimension as u64)
                .saturating_mul(towers as u64)
                .saturating_mul(limb_bytes as u64)
        }
        ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
        ConcreteWireType::Preimage { matrix, max_coefficient_bound }
            if representation == GpuStorageRepresentation::CompactBounded =>
        {
            let bound_bytes = max_coefficient_bound
                .to_biguint()
                .map(|bound| bound.bits().div_ceil(8).max(1))
                .unwrap_or(u64::MAX);
            (matrix.rows as u64)
                .saturating_mul(matrix.columns as u64)
                .saturating_mul(matrix.ring_dimension as u64)
                .saturating_mul(1u64.saturating_add(bound_bytes))
        }
        ConcreteWireType::SmallMatrix { matrix, .. } |
        ConcreteWireType::Preimage { matrix, .. } => (matrix.rows as u64)
            .saturating_mul(matrix.columns as u64)
            .saturating_mul(matrix.ring_dimension as u64)
            .saturating_mul(towers as u64)
            .saturating_mul(limb_bytes as u64),
        ConcreteWireType::IndexedFamily { element, count } => {
            validated_wire_bytes(element, descriptors, active_crt_towers, crt_limb_bytes)
                .saturating_mul(*count as u64)
        }
        ConcreteWireType::Bytes { length } => *length as u64,
        _ => 0,
    }
}

#[cfg(feature = "gpu")]
fn validated_operation_identity(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    outputs: &[ConcreteWireType],
    bindings: &mxx_ir_core::ParamEnv,
) -> Result<[u8; 32], GpuWarmupError> {
    crate::gpu_calibration::gpu_calibration_operation_identity(kind, arguments, outputs, bindings)
        .map_err(GpuWarmupError::ValidatedGraph)
}

#[cfg(not(feature = "gpu"))]
fn validated_operation_identity(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    outputs: &[ConcreteWireType],
    bindings: &mxx_ir_core::ParamEnv,
) -> Result<[u8; 32], GpuWarmupError> {
    encoding::hash_canonical(&(kind, arguments, outputs, bindings))
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuStageReport {
    pub key: GpuExecutionSiteKey,
    pub wave_instances: usize,
    pub columns_per_job: Vec<usize>,
    pub peak: Vec<GpuResourceCost>,
    pub predicted_seconds: f64,
    pub provenance: GpuProfileProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuWarmupReport {
    pub predicted_seconds: f64,
    pub limiting_stage: Option<GpuExecutionSiteKey>,
    pub stages: Vec<GpuStageReport>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuWarmupResult {
    pub plan: FrozenGpuPlan,
    pub report: GpuWarmupReport,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuWarmupError {
    #[error("GPU warmup requires at least one device")]
    EmptyFleet,
    #[error("GPU warmup profile has {actual} device costs; expected {expected}")]
    DeviceCount { expected: usize, actual: usize },
    #[error("GPU warmup has no candidate for node {0:?}")]
    NoNodeCandidate(GpuExecutionSiteKey),
    #[error("GPU warmup does not support effective GPU operation {operation} at {site:?}")]
    UnsupportedOperation { site: GpuExecutionSiteKey, operation: String },
    #[error(
        "GPU warmup requires a measured preimage footprint and fixed attempt bound at {site:?}"
    )]
    MissingPreimageProfile { site: GpuExecutionSiteKey },
    #[error("GPU warmup has no candidate for loop {0:?}")]
    NoLoopCandidate(GpuLoopSiteKey),
    #[error("GPU warmup candidate width must be positive")]
    ZeroWidth,
    #[error(
        "GPU warmup candidate does not fit its device budget at {site:?} on device {device}: {peak} > {budget}"
    )]
    ResourceExhausted { site: GpuExecutionSiteKey, device: usize, peak: u64, budget: u64 },
    #[error(
        "GPU warmup candidate does not fit host budget at {site:?} on device {device}: pinned={pinned}, host={host}"
    )]
    HostResourceExhausted { site: GpuExecutionSiteKey, device: usize, pinned: u64, host: u64 },
    #[error("GPU warmup arithmetic overflow")]
    ArithmeticOverflow,
    #[error("validated graph cannot be converted into a GPU warmup profile: {0}")]
    ValidatedGraph(String),
    #[error("GPU warmup schedule is invalid: {0}")]
    InvalidSchedule(#[from] GpuScheduleError),
    #[error("GPU warmup could not construct the frozen plan: {0}")]
    InvalidPlan(String),
}

#[derive(Clone, Debug)]
struct NodeSelection {
    seconds: f64,
    report: GpuStageReport,
}

fn candidate_values(values: &[usize], max: usize) -> Vec<usize> {
    let mut candidates =
        values.iter().copied().filter(|value| *value > 0 && *value <= max).collect::<Vec<_>>();
    if candidates.is_empty() {
        let mut power = 1usize;
        while power < max {
            candidates.push(power);
            power = power.saturating_mul(2);
            if power == 0 {
                break;
            }
        }
        if max > 0 {
            candidates.push(max);
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn balanced_intervals(columns: usize, devices: usize) -> Vec<GpuColumnInterval> {
    if columns == 0 {
        return Vec::new();
    }
    let base = columns / devices;
    let remainder = columns % devices;
    let mut start = 0;
    (0..devices)
        .filter_map(|device| {
            let length = base + usize::from(device < remainder);
            let interval =
                (length > 0).then_some(GpuColumnInterval { device, start, end: start + length });
            start += length;
            interval
        })
        .collect()
}

fn layout_for(layout: &GpuLayout, devices: usize) -> GpuLayout {
    if layout.owner_intervals.is_empty() && layout.columns > 0 {
        let mut copy = layout.clone();
        copy.owner_intervals = balanced_intervals(layout.columns, devices);
        copy
    } else {
        layout.clone()
    }
}

fn width_vectors(
    node: &GpuWarmupNode,
    layout: &GpuLayout,
    devices: usize,
    overrides: Option<&[Vec<usize>]>,
) -> Result<Vec<Vec<usize>>, GpuWarmupError> {
    let max_width = node.output_columns.max(1);
    if let Some(candidates) = overrides {
        if candidates.len() != devices {
            return Err(GpuWarmupError::DeviceCount { expected: devices, actual: candidates.len() });
        }
        let per_device = (0..devices)
            .map(|device| {
                let active = layout.instance_device_stride > 0 ||
                    layout.owner_intervals.iter().any(|interval| {
                        interval.device == device && interval.start < interval.end
                    });
                if active {
                    let values = candidate_values(&candidates[device], max_width);
                    if values.is_empty() {
                        Err(GpuWarmupError::NoNodeCandidate(node.key))
                    } else {
                        Ok(values)
                    }
                } else {
                    Ok(vec![0])
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut result = Vec::new();
        fn visit(
            index: usize,
            values: &[Vec<usize>],
            current: &mut Vec<usize>,
            result: &mut Vec<Vec<usize>>,
        ) {
            if result.len() >= 4096 {
                return;
            }
            if index == values.len() {
                result.push(current.clone());
                return;
            }
            for value in &values[index] {
                current.push(*value);
                visit(index + 1, values, current, result);
                current.pop();
            }
        }
        visit(0, &per_device, &mut Vec::new(), &mut result);
        return Ok(result);
    }
    Ok(candidate_values(&node.tile_widths, max_width)
        .into_iter()
        .map(|width| {
            (0..devices)
                .map(|device| {
                    (layout.instance_device_stride > 0 ||
                        layout.owner_intervals.iter().any(|interval| {
                            interval.device == device && interval.start < interval.end
                        }))
                    .then_some(width)
                    .unwrap_or(0)
                })
                .collect()
        })
        .collect())
}

pub fn gpu_batch_wave_time(
    schedules: &[&GpuColumnSchedule],
    times: &[GpuTimeModel],
) -> Result<f64, GpuWarmupError> {
    let mut total = 0.0;
    for class in GpuColumnSchedule::batch_wave_classes(schedules) {
        let mut by_device = BTreeMap::<usize, f64>::new();
        for (_, job) in class.jobs {
            let width = job.end.checked_sub(job.start).ok_or(GpuWarmupError::ArithmeticOverflow)?;
            let time = times.get(job.device).ok_or(GpuWarmupError::DeviceCount {
                expected: schedules.first().map_or(0, |schedule| schedule.widths().len()),
                actual: times.len(),
            })?;
            *by_device.entry(job.device).or_default() += time.job_seconds(width);
        }
        let overhead = times.iter().map(|time| time.wave_overhead_seconds).fold(0.0, f64::max);
        let latency = by_device.values().copied().fold(0.0, f64::max) + overhead;
        total += latency * class.multiplicity as f64;
    }
    Ok(total)
}

/// Host/control stages have no column ranges, while single-device stages are
/// bound as one full operation to one owner; neither is forced through the
/// fused column-union builder.
/// Shared timing for host/control and single-owner operations.  These stages
/// do not produce column schedules: host/control uses zero output columns,
/// while a single-device operation passes its full width and executes once per
/// instance on the pinned owner.
pub fn gpu_non_column_batch_wave_time(
    instances: usize,
    columns: usize,
    owner: usize,
    times: &[GpuTimeModel],
) -> Result<f64, GpuWarmupError> {
    if instances == 0 {
        return Ok(0.0);
    }
    let time = times.get(owner).ok_or(GpuWarmupError::DeviceCount {
        expected: owner.saturating_add(1),
        actual: times.len(),
    })?;
    Ok(instances as f64 * (time.job_seconds(columns) + time.wave_overhead_seconds))
}

/// Reconstruct the production union's interval jobs without materializing all
/// future jobs. Unlike a same-wave port merge, this deliberately includes
/// jobs from every source interval and retains their source wave; a wide port
/// can overlap a narrower port's later logical wave.
fn compressed_jobs_over_ranges(
    schedule: &GpuColumnSchedule,
    ranges: &[(usize, usize)],
) -> Vec<(usize, GpuColumnJob)> {
    let mut wave_starts = vec![0usize; schedule.widths().len()];
    let mut jobs = Vec::new();
    for (source_interval, interval) in schedule.intervals().iter().enumerate() {
        let width = schedule.widths()[interval.device];
        let wave_start = wave_starts[interval.device];
        let count = (interval.end - interval.start).div_ceil(width);
        wave_starts[interval.device] += count;
        for &(range_start, range_end) in ranges {
            if range_start >= range_end ||
                interval.end <= range_start ||
                interval.start >= range_end
            {
                continue;
            }
            let offset = range_start.saturating_sub(interval.start) / width * width;
            let mut start = interval.start + offset;
            while start < range_end && start < interval.end {
                let end = start + width.min(interval.end - start);
                if end > range_start {
                    jobs.push((
                        wave_start + (start - interval.start) / width,
                        GpuColumnJob { device: interval.device, source_interval, start, end },
                    ));
                }
                if end == interval.end {
                    break;
                }
                start = end;
            }
        }
    }
    jobs.sort_unstable_by_key(|(wave, job)| (job.start, job.end, job.device, *wave));
    jobs.dedup();
    jobs
}

fn checked_lcm(lhs: usize, rhs: usize) -> Result<usize, GpuWarmupError> {
    if lhs == 0 || rhs == 0 {
        return Ok(0);
    }
    let mut a = lhs;
    let mut b = rhs;
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    (lhs / a).checked_mul(rhs).ok_or(GpuWarmupError::ArithmeticOverflow)
}

fn gpu_multi_output_wave_time_at(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
    times: &[GpuTimeModel],
    logical_wave: usize,
) -> Result<f64, GpuWarmupError> {
    let mut by_device = BTreeMap::<usize, f64>::new();
    for instance in 0..instances {
        let mut candidate_ranges = schedules_by_port
            .iter()
            .flat_map(|port| port[instance].wave_jobs(logical_wave))
            .filter_map(|job| (job.start < job.end).then_some((job.start, job.end)))
            .collect::<Vec<_>>();
        if candidate_ranges.is_empty() {
            continue;
        }
        candidate_ranges.sort_unstable();
        candidate_ranges.dedup();
        let port_jobs = schedules_by_port
            .iter()
            .map(|port| compressed_jobs_over_ranges(&port[instance], &candidate_ranges))
            .collect::<Vec<_>>();
        let mut boundaries = port_jobs
            .iter()
            .flat_map(|jobs| jobs.iter().flat_map(|(_, job)| [job.start, job.end]))
            .collect::<Vec<_>>();
        boundaries.sort_unstable();
        boundaries.dedup();
        for pair in boundaries.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            if start >= end {
                continue;
            }
            let containing = port_jobs
                .iter()
                .flat_map(|jobs| jobs.iter())
                .filter(|(_, job)| job.start <= start && end <= job.end)
                .collect::<Vec<_>>();
            let Some(source_wave) = containing.iter().map(|(wave, _)| *wave).max() else {
                // The production union builder skips a local span without a
                // representative source job (not an invalid schedule).
                continue;
            };
            if source_wave != logical_wave {
                continue;
            }
            let device = containing.iter().map(|(_, job)| job.device).min().unwrap();
            let time = times.get(device).ok_or(GpuWarmupError::DeviceCount {
                expected: schedules_by_port
                    .first()
                    .and_then(|port| port.first())
                    .map_or(0, |schedule| schedule.widths().len()),
                actual: times.len(),
            })?;
            let width = end.checked_sub(start).ok_or(GpuWarmupError::ArithmeticOverflow)?;
            *by_device.entry(device).or_default() += time.job_seconds(width);
        }
    }
    let overhead = times.iter().map(|time| time.wave_overhead_seconds).fold(0.0, f64::max);
    Ok(by_device.values().copied().fold(0.0, f64::max) + overhead)
}

fn gpu_multi_output_wave_period(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
    wave: usize,
) -> Result<usize, GpuWarmupError> {
    let mut period = 1usize;
    for port in schedules_by_port {
        for schedule in port.iter().take(instances) {
            for job in schedule.wave_jobs(wave) {
                let width =
                    job.end.checked_sub(job.start).ok_or(GpuWarmupError::ArithmeticOverflow)?;
                if width != 0 {
                    period = checked_lcm(period, width)?;
                }
            }
        }
    }
    Ok(period)
}

fn schedule_wave_breakpoints_at(
    schedule: &GpuColumnSchedule,
    coordinate: usize,
) -> Result<Vec<usize>, GpuWarmupError> {
    let mut wave_starts = vec![0usize; schedule.widths().len()];
    let mut breakpoints = Vec::new();
    for interval in schedule.intervals() {
        let width = schedule.widths()[interval.device];
        let count = (interval.end - interval.start).div_ceil(width);
        let wave_start = wave_starts[interval.device];
        if interval.start <= coordinate && coordinate <= interval.end {
            let offset = (coordinate - interval.start) / width;
            let wave = wave_start
                .checked_add(offset.min(count))
                .ok_or(GpuWarmupError::ArithmeticOverflow)?;
            breakpoints.push(wave);
            if offset < count {
                breakpoints.push(wave.checked_add(1).ok_or(GpuWarmupError::ArithmeticOverflow)?);
            }
        }
        wave_starts[interval.device] =
            wave_start.checked_add(count).ok_or(GpuWarmupError::ArithmeticOverflow)?;
    }
    breakpoints.sort_unstable();
    breakpoints.dedup();
    Ok(breakpoints)
}

fn schedule_interval_wave_data(
    schedule: &GpuColumnSchedule,
) -> Result<Vec<(GpuColumnInterval, usize, usize)>, GpuWarmupError> {
    let mut wave_starts = vec![0usize; schedule.widths().len()];
    let mut data = Vec::with_capacity(schedule.intervals().len());
    for &interval in schedule.intervals() {
        let width = schedule.widths()[interval.device];
        let count = (interval.end - interval.start).div_ceil(width);
        let wave_start = wave_starts[interval.device];
        data.push((interval, wave_start, width));
        wave_starts[interval.device] =
            wave_start.checked_add(count).ok_or(GpuWarmupError::ArithmeticOverflow)?;
    }
    Ok(data)
}

fn cross_port_source_wave_coordinates(
    left: &GpuColumnSchedule,
    right: &GpuColumnSchedule,
) -> Result<Vec<usize>, GpuWarmupError> {
    let mut coordinates = Vec::new();
    for (left_interval, left_wave, left_width) in schedule_interval_wave_data(left)? {
        for (right_interval, right_wave, right_width) in schedule_interval_wave_data(right)? {
            let start = left_interval.start.max(right_interval.start);
            let end = left_interval.end.min(right_interval.end);
            if start >= end {
                continue;
            }
            let period = checked_lcm(left_width, right_width)?;
            if period <= 4096 {
                let sample_end =
                    end.min(start.checked_add(period).ok_or(GpuWarmupError::ArithmeticOverflow)?);
                coordinates.extend(start..sample_end);
            }
            // Solve the continuous crossing of the two floor-wave functions,
            // then inspect a checked neighbourhood. The bounded sample above
            // handles all phase crossings for ordinary tile widths; this
            // candidate covers large coprime widths without column iteration.
            if left_width != right_width {
                let l_start = i128::try_from(left_interval.start)
                    .map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let r_start = i128::try_from(right_interval.start)
                    .map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let l_wave =
                    i128::try_from(left_wave).map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let r_wave =
                    i128::try_from(right_wave).map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let l_width =
                    i128::try_from(left_width).map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let r_width =
                    i128::try_from(right_width).map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let numerator = (r_wave
                    .checked_sub(l_wave)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?
                    .checked_mul(l_width)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?
                    .checked_mul(r_width)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?)
                .checked_add(
                    l_start
                        .checked_mul(r_width)
                        .ok_or(GpuWarmupError::ArithmeticOverflow)?
                        .checked_sub(
                            r_start
                                .checked_mul(l_width)
                                .ok_or(GpuWarmupError::ArithmeticOverflow)?,
                        )
                        .ok_or(GpuWarmupError::ArithmeticOverflow)?,
                )
                .ok_or(GpuWarmupError::ArithmeticOverflow)?;
                let denominator =
                    r_width.checked_sub(l_width).ok_or(GpuWarmupError::ArithmeticOverflow)?;
                if denominator != 0 {
                    let root = numerator.div_euclid(denominator);
                    for delta in -2i128..=2 {
                        let candidate =
                            root.checked_add(delta).ok_or(GpuWarmupError::ArithmeticOverflow)?;
                        if candidate >=
                            i128::try_from(start)
                                .map_err(|_| GpuWarmupError::ArithmeticOverflow)? &&
                            candidate <
                                i128::try_from(end)
                                    .map_err(|_| GpuWarmupError::ArithmeticOverflow)?
                        {
                            coordinates.push(
                                usize::try_from(candidate)
                                    .map_err(|_| GpuWarmupError::ArithmeticOverflow)?,
                            );
                        }
                    }
                }
            }
        }
    }
    coordinates.sort_unstable();
    coordinates.dedup();
    Ok(coordinates)
}

/// Time model for one primitive with multiple output ports. The lazy union
/// wave iterator preserves cross-instance concurrency without materializing
/// all union jobs or a vector indexed by the number of columns.
pub fn gpu_multi_output_batch_wave_time(
    schedules: &[Vec<GpuColumnSchedule>],
    times: &[GpuTimeModel],
) -> Result<f64, GpuWarmupError> {
    let instances = schedules.len();
    let port_count = schedules.first().map_or(0, Vec::len);
    if instances == 0 || port_count == 0 {
        return Ok(0.0);
    }
    if schedules.iter().any(|ports| ports.len() != port_count) {
        return Err(GpuWarmupError::InvalidPlan("multi-output schedule port count mismatch".into()));
    }
    let schedules_by_port = (0..port_count)
        .map(|port| schedules.iter().map(|instance| instance[port].clone()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    // A single (or identical multi-output) port has no cross-port boundary;
    // use the schedule's compressed wave classes directly. This is the
    // constant-memory path for huge width-one ranges.
    if port_count == 1 || schedules_by_port.iter().skip(1).all(|port| port == &schedules_by_port[0])
    {
        let references = schedules_by_port[0].iter().collect::<Vec<_>>();
        return gpu_batch_wave_time(&references, times);
    }
    let wave_count = schedules_by_port
        .iter()
        .flat_map(|port| port.iter().map(GpuColumnSchedule::wave_count))
        .max()
        .unwrap_or(0);
    let mut boundaries = vec![0, wave_count];
    for port in &schedules_by_port {
        for schedule in port.iter().take(instances) {
            for class in schedule.wave_classes() {
                let end = class
                    .first_wave
                    .checked_add(class.multiplicity)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?;
                boundaries.extend([class.first_wave, end]);
            }
        }
    }
    // A source interval transition can change which port contributes the
    // maximum source wave before any individual schedule changes class. Add
    // every stored global owner boundary and project it onto every other
    // schedule's wave coordinate. The resulting set is still bounded by the
    // stored interval count, while preserving cross-port ownership changes.
    let mut owner_boundaries = BTreeSet::new();
    for port in &schedules_by_port {
        for schedule in port.iter().take(instances) {
            for interval in schedule.intervals() {
                owner_boundaries.insert(interval.start);
                owner_boundaries.insert(interval.end);
            }
        }
    }
    // Within overlapping owner intervals, the max source-wave contributor can
    // change even when neither schedule reaches a class endpoint. Include the
    // checked floor-function crossing coordinates before projecting them to
    // every schedule's wave axis.
    for left_port in 0..schedules_by_port.len() {
        for right_port in left_port + 1..schedules_by_port.len() {
            for instance in 0..instances {
                owner_boundaries.extend(cross_port_source_wave_coordinates(
                    &schedules_by_port[left_port][instance],
                    &schedules_by_port[right_port][instance],
                )?);
            }
        }
    }
    for port in &schedules_by_port {
        for schedule in port.iter().take(instances) {
            for &coordinate in &owner_boundaries {
                boundaries.extend(schedule_wave_breakpoints_at(schedule, coordinate)?);
            }
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut total = 0.0;
    for pair in boundaries.windows(2) {
        let (first_wave, end_wave) = (pair[0], pair[1]);
        if first_wave >= end_wave {
            continue;
        }
        let period = gpu_multi_output_wave_period(&schedules_by_port, instances, first_wave)?;
        let span = end_wave - first_wave;
        let sample = period.min(span);
        let mut sample_seconds = 0.0;
        for offset in 0..sample {
            sample_seconds += gpu_multi_output_wave_time_at(
                &schedules_by_port,
                instances,
                times,
                first_wave + offset,
            )?;
        }
        let cycles = span / period;
        let remainder = span % period;
        total += sample_seconds * cycles as f64;
        for offset in 0..remainder {
            total += gpu_multi_output_wave_time_at(
                &schedules_by_port,
                instances,
                times,
                first_wave + cycles * period + offset,
            )?;
        }
    }
    Ok(total)
}

fn output_layouts<'a>(
    node: &GpuWarmupNode,
    layouts: &'a BTreeMap<LayoutId, &'a GpuLayout>,
) -> Result<Vec<&'a GpuLayout>, GpuWarmupError> {
    if node.output_layouts.is_empty() {
        return Err(GpuWarmupError::NoNodeCandidate(node.key));
    }
    node.output_layouts
        .iter()
        .map(|layout_id| {
            layouts
                .get(layout_id)
                .copied()
                .ok_or_else(|| GpuWarmupError::InvalidPlan(format!("unknown layout {layout_id}")))
        })
        .collect()
}

fn choose_node_with_candidates(
    node: &GpuWarmupNode,
    layouts: &[&GpuLayout],
    devices: usize,
    budgets: &[GpuDeviceBudget],
    wave_instances: usize,
    loop_count: usize,
    overrides: Option<&[Vec<usize>]>,
) -> Result<NodeSelection, GpuWarmupError> {
    if node.cost.len() != devices {
        return Err(GpuWarmupError::DeviceCount { expected: devices, actual: node.cost.len() });
    }
    if node.preimage_max_attempts.is_some() != node.preimage_footprint.is_some() {
        return Err(GpuWarmupError::MissingPreimageProfile { site: node.key });
    }
    let layout = layouts.first().ok_or(GpuWarmupError::NoNodeCandidate(node.key))?;
    let candidates = width_vectors(node, layout, devices, overrides)?;
    if candidates.is_empty() {
        return Err(GpuWarmupError::NoNodeCandidate(node.key));
    }
    let mut best: Option<(f64, Vec<usize>, Vec<GpuResourceCost>)> = None;
    let mut host_failure = None;
    let mut resource_failure = None;
    let mut profile_failure = false;
    for widths in candidates {
        let mut peaks = Vec::with_capacity(devices);
        let single_device = node.column_capability == ColumnCapability::SingleDevice;
        let column_work = !matches!(
            node.column_capability,
            ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
        ) && layouts.iter().any(|layout| layout.columns > 0);
        // Gadget/trapdoor single-device dispatch is contractually pinned to
        // logical GPU0. Owner candidates must never move this operation.
        let single_owner = 0;
        let schedules_owned = if column_work {
            let schedules_by_port = layouts
                .iter()
                .map(|layout| {
                    (0..wave_instances)
                        .map(|instance| layout.schedule(&widths, instance))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            (0..wave_instances)
                .map(|instance| {
                    schedules_by_port.iter().map(|port| port[instance].clone()).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut valid = true;
        let mut device_failure = false;
        let mut candidate_host_failure = None;
        for device in 0..devices {
            let scheduled_owned = schedules_owned
                .iter()
                .flat_map(|ports| ports.iter())
                .flat_map(|schedule| schedule.intervals().iter())
                .filter(|interval| interval.device == device)
                .map(|interval| interval.end - interval.start)
                .sum::<usize>();
            let owned = if single_device && device == single_owner {
                node.output_columns
            } else if single_device {
                0
            } else {
                scheduled_owned
            };
            // All sibling jobs on a device execute sequentially.
            let concurrent_jobs = usize::from(owned > 0 && wave_instances > 0);
            let output_columns = owned;
            let mut stage_cost = node.cost[device].clone();
            if let Some(footprints) = &node.preimage_footprint {
                let footprint = match footprints
                    .get(device)
                    .ok_or(GpuWarmupError::DeviceCount {
                        expected: devices,
                        actual: footprints.len(),
                    })?
                    .as_resource_cost_for_width(widths[device], node.key)
                {
                    Ok(footprint) => footprint,
                    Err(GpuWarmupError::MissingPreimageProfile { .. }) => {
                        profile_failure = true;
                        valid = false;
                        break;
                    }
                    Err(error) => return Err(error),
                };
                stage_cost.fixed = stage_cost
                    .fixed
                    .checked_add(footprint)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?;
            }
            let peak = stage_cost.peak_for(
                wave_instances,
                output_columns,
                concurrent_jobs,
                widths[device].min(owned),
            )?;
            if peak.device_bytes() > budgets[device].device_bytes {
                valid = false;
                device_failure = true;
                if resource_failure.is_none() {
                    resource_failure =
                        Some((device, peak.device_bytes(), budgets[device].device_bytes));
                }
            }
            if peak.pinned_host > budgets[device].pinned_host_bytes ||
                peak.host > budgets[device].host_bytes
            {
                valid = false;
                candidate_host_failure = Some((device, peak.pinned_host, peak.host));
            }
            peaks.push(peak);
        }
        if !valid {
            if !device_failure && host_failure.is_none() {
                host_failure = candidate_host_failure;
            }
            continue;
        }
        let times = node.cost.iter().map(|cost| cost.time).collect::<Vec<_>>();
        let wave_seconds = if column_work {
            gpu_multi_output_batch_wave_time(&schedules_owned, &times)?
        } else {
            gpu_non_column_batch_wave_time(
                wave_instances,
                if single_device { node.output_columns } else { 0 },
                if single_device { single_owner } else { 0 },
                &times,
            )?
        };
        let full_waves = loop_count / wave_instances;
        let tail = loop_count % wave_instances;
        let seconds = wave_seconds * full_waves as f64 +
            if tail > 0 {
                if column_work {
                    gpu_multi_output_batch_wave_time(&schedules_owned[..tail], &times)?
                } else {
                    gpu_non_column_batch_wave_time(
                        tail,
                        if single_device { node.output_columns } else { 0 },
                        if single_device { single_owner } else { 0 },
                        &times,
                    )?
                }
            } else {
                0.0
            };
        if best.as_ref().is_none_or(|(best_seconds, best_widths, _)| {
            seconds.total_cmp(best_seconds) == Ordering::Less ||
                (seconds.total_cmp(best_seconds) == Ordering::Equal && widths < *best_widths)
        }) {
            best = Some((seconds, widths.clone(), peaks));
        }
    }
    let Some((seconds, widths, peak)) = best else {
        if let Some((device, pinned, host)) = host_failure {
            return Err(GpuWarmupError::HostResourceExhausted {
                site: node.key,
                device,
                pinned,
                host,
            });
        }
        if profile_failure && resource_failure.is_none() {
            return Err(GpuWarmupError::MissingPreimageProfile { site: node.key });
        }
        return Err(GpuWarmupError::ResourceExhausted {
            site: node.key,
            device: resource_failure.map_or(0, |failure| failure.0),
            peak: resource_failure.map_or(u64::MAX, |failure| failure.1),
            budget: resource_failure.map_or(budgets[0].device_bytes, |failure| failure.2),
        });
    };
    Ok(NodeSelection {
        seconds,
        report: GpuStageReport {
            key: node.key,
            wave_instances,
            columns_per_job: widths,
            peak,
            predicted_seconds: seconds,
            provenance: node.provenance,
        },
    })
}

fn loop_context(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
) -> Result<Option<(GpuLoopSiteKey, usize, bool)>, GpuWarmupError> {
    let (parent, owner, nested) = match scope_id {
        FrozenGraphScopeId::ParallelBody { parent, owner } => (&**parent, *owner, false),
        FrozenGraphScopeId::SequentialBody { parent, owner } => (&**parent, *owner, true),
        _ => return Ok(None),
    };
    let parent_scope = validated.source.scope(parent).ok_or_else(|| {
        GpuWarmupError::ValidatedGraph(format!("missing parent scope for {scope_id:?}"))
    })?;
    let loop_node = parent_scope
        .node(owner)
        .ok_or_else(|| GpuWarmupError::ValidatedGraph(format!("missing loop owner {owner:?}")))?;
    let count = match loop_node.kind() {
        NodeKind::ParallelLoop(spec) => spec.count.evaluate(&validated.bindings),
        NodeKind::SequentialLoop(spec) => spec.count.evaluate(&validated.bindings),
        _ => return Err(GpuWarmupError::ValidatedGraph("loop body has a non-loop owner".into())),
    }
    .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
    .to_usize()
    .ok_or_else(|| GpuWarmupError::ValidatedGraph("loop count does not fit usize".into()))?;
    let shape_class = scope_shape_class(validated, parent)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    let nested = nested || !matches!(parent, FrozenGraphScopeId::Root);
    Ok(Some((GpuLoopSiteKey { site: owner.0, shape_class }, count, nested)))
}

fn graph_loop_wave_candidates(
    count: usize,
    devices: usize,
    max_parallel_instances: usize,
) -> Vec<usize> {
    let mut candidates = vec![1];
    let mut power = 2usize;
    while power < count.max(1) {
        candidates.push(power);
        power = power.saturating_mul(2);
        if power == 0 {
            break;
        }
    }
    if count > 1 {
        candidates.push(devices.min(count));
    }
    candidates.push(max_parallel_instances.min(count.max(1)));
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

/// Bound graph-derived wave candidates before planning. A wave larger than
/// the fleet's concurrent owner count cannot improve dispatch concurrency;
/// per-instance resource terms can tighten that bound further. This keeps a
/// billion-iteration loop symbolic instead of allocating billion schedules.
fn bound_loop_wave_candidates(
    loop_input: &GpuWarmupLoop,
    nodes: &[GpuWarmupNode],
    budgets: &[GpuDeviceBudget],
    max_parallel_instances: usize,
) -> usize {
    // Keep the symbolic candidate set bounded even when a profile has no
    // measured per-instance bytes. Resource terms tighten this conservative
    // bound; fleet size is deliberately not a concurrency cap because one
    // owner can still amortize launch overhead across sibling instances.
    let mut bound = loop_input.loop_count.max(1).min(max_parallel_instances.max(1));
    for node in nodes.iter().filter(|node| node.loop_site == Some(loop_input.key)) {
        for (device, cost) in node.cost.iter().enumerate() {
            let Some(budget) = budgets.get(device) else { continue };
            let mut device_bound = usize::MAX;
            if cost.per_instance.device_bytes() > 0 {
                device_bound = device_bound.min(
                    usize::try_from(budget.device_bytes / cost.per_instance.device_bytes())
                        .unwrap_or(usize::MAX),
                );
            }
            if cost.per_instance.pinned_host > 0 {
                device_bound = device_bound.min(
                    usize::try_from(budget.pinned_host_bytes / cost.per_instance.pinned_host)
                        .unwrap_or(usize::MAX),
                );
            }
            if cost.per_instance.host > 0 {
                device_bound = device_bound.min(
                    usize::try_from(budget.host_bytes / cost.per_instance.host)
                        .unwrap_or(usize::MAX),
                );
            }
            if device_bound != usize::MAX {
                bound = bound.min(device_bound.max(1));
            }
        }
    }
    bound.max(1)
}

/// Build the pure planner input from a validated graph.  This is the runtime
/// lifecycle entry point: operation identities use the same calibration
/// identity as production (when GPU support is enabled), and graph liveness,
/// retained outputs, artifact transfers, and effective variant metadata are
/// folded into the profile before candidate selection.
fn warmup_input_from_validated_with_limit(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupInput, GpuWarmupError> {
    let devices = config.contract.logical_to_physical_devices.len();
    if devices == 0 {
        return Err(GpuWarmupError::EmptyFleet);
    }
    if config.active_crt_towers == 0 || config.crt_limb_bytes == 0 {
        return Err(GpuWarmupError::ValidatedGraph(
            "validated warmup requires an ordered active CRT descriptor".into(),
        ));
    }
    if config.storage_descriptors.values().any(|descriptor| !valid_storage_descriptor(descriptor)) {
        return Err(GpuWarmupError::ValidatedGraph(
            "validated warmup has an invalid ordered CRT storage descriptor".into(),
        ));
    }
    if config.default_cost.len() != devices {
        return Err(GpuWarmupError::DeviceCount {
            expected: devices,
            actual: config.default_cost.len(),
        });
    }
    let graph_hash = spec_hash(&validated.source, &validated.bindings)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
        .0;
    let mut contract = config.contract.clone();
    if contract.graph_specification_hash == [0; 32] {
        contract.graph_specification_hash = graph_hash;
    }
    let mut layouts = config.layouts.clone();
    let fallback_layout = layouts.first().map(|layout| layout.id).ok_or_else(|| {
        GpuWarmupError::ValidatedGraph("validated warmup requires one output layout".into())
    })?;
    let mut loops = BTreeMap::<GpuLoopSiteKey, GpuWarmupLoop>::new();
    let mut nodes = Vec::new();
    let mut value_strides = BTreeMap::<(u64, WireRef), usize>::new();
    for (scope_id, checked) in &validated.scopes {
        let shape_class = scope_shape_class(validated, scope_id)
            .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
        let loop_info = loop_context(validated, scope_id)?;
        if let Some((key, count, nested)) = loop_info {
            let candidates =
                graph_loop_wave_candidates(count, devices, config.max_parallel_instances.get());
            loops.entry(key).or_insert(GpuWarmupLoop {
                key,
                loop_count: count,
                wave_candidates: candidates,
                nested,
            });
        }
        let scope = validated.source.scope(scope_id).ok_or_else(|| {
            GpuWarmupError::ValidatedGraph(format!("missing graph scope {scope_id:?}"))
        })?;
        for (position, handle) in checked.execution_order.iter().enumerate() {
            let node_id = NodeId(position as u64);
            let arguments = scope.arguments(handle).ok_or_else(|| {
                GpuWarmupError::ValidatedGraph(format!(
                    "missing arguments for {scope_id:?}/{node_id:?}"
                ))
            })?;
            let argument_types =
                arguments.iter().map(|wire| checked.wire_types[wire].clone()).collect::<Vec<_>>();
            let (output_types, effective_identity) =
                crate::executor::gpu_effective_site_metadata(validated, scope_id, node_id)
                    .map_err(GpuWarmupError::ValidatedGraph)?;
            let ordinary_identity = validated_operation_identity(
                handle.kind(),
                &argument_types,
                &output_types,
                &validated.bindings,
            )?;
            let identity = config
                .effective_operation_identities
                .get(&(shape_class, node_id.0))
                .copied()
                .or(effective_identity)
                .unwrap_or(ordinary_identity);
            let effective_operation = config
                .effective_operations
                .get(&(shape_class, node_id.0))
                .copied()
                .unwrap_or_else(|| effective_gpu_operation(handle.kind()));
            if effective_operation == EffectiveGpuOperation::Unsupported {
                return Err(GpuWarmupError::UnsupportedOperation {
                    site: GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class: 0 },
                    operation: format!("{:?}", handle.kind()),
                });
            }
            let profile = config.profiles.get(&identity);
            let is_preimage = effective_operation == EffectiveGpuOperation::PreimageSample;
            let preimage_footprint = if is_preimage {
                let profile = profile.ok_or_else(|| GpuWarmupError::MissingPreimageProfile {
                    site: GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class: 0 },
                })?;
                if profile.preimage_max_attempts.is_none() ||
                    profile.preimage_footprint.as_ref().is_none_or(|footprint| {
                        footprint.len() != devices ||
                            footprint.iter().any(|entry| {
                                entry.certified_tile_width.is_none_or(|width| width == 0)
                            })
                    })
                {
                    return Err(GpuWarmupError::MissingPreimageProfile {
                        site: GpuExecutionSiteKey {
                            site: node_id.0,
                            shape_class,
                            instance_class: 0,
                        },
                    });
                }
                profile.preimage_footprint.clone()
            } else {
                None
            };
            let mut cost = profile
                .map(|profile| profile.cost.clone())
                .unwrap_or_else(|| config.default_cost.clone());
            let live_bytes = checked
                .liveness
                .last_use
                .iter()
                // The current node's outputs are charged once in `outputs`;
                // excluding them here avoids treating an output that is also
                // retained as two allocations at the dispatch boundary.
                .filter(|(wire, _)| (wire.node.0 as usize) < position)
                .filter(|(_, last_use)| **last_use >= position)
                .filter_map(|(wire, _)| checked.wire_types.get(wire))
                .map(|ty| {
                    validated_wire_bytes(
                        ty,
                        &config.storage_descriptors,
                        config.active_crt_towers,
                        config.crt_limb_bytes,
                    )
                })
                .sum::<u64>();
            let output_bytes = output_types
                .iter()
                .map(|ty| {
                    validated_wire_bytes(
                        ty,
                        &config.storage_descriptors,
                        config.active_crt_towers,
                        config.crt_limb_bytes,
                    )
                })
                .sum::<u64>();
            let input_transfer_bytes = arguments
                .iter()
                .filter_map(|wire| {
                    checked.artifact_inputs.get(wire).map(|_| checked.wire_types[wire].clone())
                })
                .map(|ty| {
                    validated_wire_bytes(
                        &ty,
                        &config.storage_descriptors,
                        config.active_crt_towers,
                        config.crt_limb_bytes,
                    )
                })
                .sum::<u64>();
            let layout_id = profile.map(|profile| profile.output_layout).unwrap_or(fallback_layout);
            let columns = output_types
                .iter()
                .filter_map(|ty| ty.matrix_type().map(|matrix| matrix.columns))
                .max()
                .or_else(|| {
                    argument_types
                        .iter()
                        .filter_map(|ty| ty.matrix_type().map(|matrix| matrix.columns))
                        .max()
                })
                .unwrap_or_else(|| {
                    layouts
                        .iter()
                        .find(|layout| layout.id == layout_id)
                        .map_or(0, |layout| layout.columns)
                });
            let capability =
                capability_for_effective_operation(effective_operation, &argument_types);
            let lowered_range_bytes = if columns > 0 &&
                !matches!(
                    capability,
                    ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
                ) {
                map_output_range_to_inputs_with_output(
                    handle.kind(),
                    &argument_types,
                    columns,
                    ColumnRange { start: 0, end: columns },
                )
                .map_err(|error| {
                    GpuWarmupError::ValidatedGraph(format!(
                        "unsupported range mapping at {scope_id:?}/{node_id:?}: {error}"
                    ))
                })?
                .into_iter()
                .filter_map(|input| {
                    argument_types.get(input.operand).map(|ty| (ty, input.range.len()))
                })
                .map(|(ty, count)| {
                    let columns = ty.matrix_type().map_or(1, |matrix| {
                        if matches!(handle.kind(), NodeKind::Transpose) {
                            matrix.rows.max(1)
                        } else {
                            matrix.columns.max(1)
                        }
                    });
                    validated_wire_bytes(
                        ty,
                        &config.storage_descriptors,
                        config.active_crt_towers,
                        config.crt_limb_bytes,
                    )
                    .saturating_mul(count as u64) /
                        columns as u64
                })
                .sum::<u64>()
            } else {
                0
            };
            // Artifact-backed inputs and lowered input slices can describe
            // the same storage. Charge the union's conservative byte bound,
            // rather than double-counting a full input for a full output
            // range. The measured profile still carries route-specific
            // staging/replica/scratch costs.
            let transfer_bytes = input_transfer_bytes.max(lowered_range_bytes);
            let retained = output_types.iter().enumerate().any(|(port, _)| {
                let wire = WireRef { node: node_id, port: Port(port as u32) };
                checked.liveness.retained.contains(&wire) || scope.outputs().contains(&wire)
            });
            for device_cost in &mut cost {
                device_cost.per_instance.live =
                    device_cost.per_instance.live.saturating_add(live_bytes);
                device_cost.per_instance.outputs =
                    device_cost.per_instance.outputs.saturating_add(output_bytes);
                device_cost.per_instance.transfers =
                    device_cost.per_instance.transfers.saturating_add(transfer_bytes);
            }
            let variant = profile
                .map(|profile| profile.implementation_variant.clone())
                .unwrap_or_else(|| config.default_implementation_variant.clone());
            let variant = format!(
                "{variant};effective={:?};scope={shape_class};live={live_bytes};retained={retained}",
                handle.kind()
            );
            let key = GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class: 0 };
            // Each output has its own concrete layout. A profile layout supplies
            // placement only when its width matches; it cannot supply the shape.
            let template = layouts
                .iter()
                .find(|layout| layout.id == layout_id)
                .cloned()
                .ok_or_else(|| GpuWarmupError::InvalidPlan("missing profile layout".into()))?;
            let mut output_layouts = Vec::new();
            for (port, ty) in output_types.iter().enumerate() {
                let id = layouts
                    .iter()
                    .map(|layout| layout.id)
                    .max()
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?;
                let matrix = ty.matrix_type();
                let port_columns = matrix.map_or(0, |matrix| matrix.columns);
                let inherited_operand = match handle.kind() {
                    NodeKind::PreimageSample { .. } => 2,
                    NodeKind::MatrixMulSmallRhs => 1,
                    NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply) => {
                        if argument_types
                            .get(1)
                            .and_then(ConcreteWireType::matrix_type)
                            .is_some_and(|ty| ty.is_scalar())
                        {
                            0
                        } else {
                            1
                        }
                    }
                    _ => 0,
                };
                let stride = if capability == ColumnCapability::GeneratedColumns &&
                    loop_info.is_some() &&
                    port_columns == 1
                {
                    1
                } else {
                    arguments
                        .get(inherited_operand)
                        .and_then(|wire| value_strides.get(&(shape_class, *wire)))
                        .copied()
                        .unwrap_or(template.instance_device_stride)
                };
                value_strides.insert(
                    (shape_class, WireRef { node: node_id, port: Port(port as u32) }),
                    stride,
                );
                layouts.push(GpuLayout {
                    id,
                    columns: port_columns,
                    rows: matrix.map_or(0, |matrix| matrix.rows),
                    ring_dimension: matrix.map_or(0, |matrix| matrix.ring_dimension),
                    representation: format!("{ty:?}"),
                    instance_device_stride: stride,
                    owner_intervals: if template.columns == port_columns {
                        template.owner_intervals.clone()
                    } else {
                        Vec::new()
                    },
                });
                output_layouts.push(id);
            }
            if output_layouts.is_empty() {
                continue;
            }
            nodes.push(GpuWarmupNode {
                key,
                operation_identity: identity,
                effective_operation,
                column_capability: capability,
                output_layouts,
                implementation_variant: variant,
                loop_site: loop_info.as_ref().map(|(key, _, _)| *key),
                output_columns: columns,
                tile_widths: profile
                    .map(|profile| profile.tile_widths.clone())
                    .unwrap_or_else(|| config.default_tile_widths.clone()),
                cost,
                provenance: profile
                    .map(|profile| profile.provenance)
                    .unwrap_or(GpuProfileProvenance::ConservativeEstimate),
                preimage_max_attempts: profile.and_then(|profile| profile.preimage_max_attempts),
                preimage_footprint,
            });
        }
    }
    layouts.sort_by_key(|layout| layout.id);
    let mut loops = loops.into_values().collect::<Vec<_>>();
    for loop_input in &mut loops {
        let bound = bound_loop_wave_candidates(
            loop_input,
            &nodes,
            &contract.device_budgets,
            config.max_parallel_instances.get(),
        );
        loop_input.wave_candidates.retain(|candidate| *candidate <= bound);
        if loop_input.wave_candidates.is_empty() {
            loop_input.wave_candidates.push(bound);
        }
    }
    Ok(GpuWarmupInput { contract, layouts, loops, nodes })
}

/// Build the pure planner input using the default execution wave limit.
pub fn warmup_input_from_validated(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupInput, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_input_from_validated_with_execution_config(validated, config, &execution_config)
}

/// Build the pure planner input using the exact execution wave limit.
pub fn warmup_input_from_validated_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupInput, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    warmup_input_from_validated_with_limit(validated, &config)
}

fn warmup_gpu_from_validated_with_limit(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let input = warmup_input_from_validated_with_limit(validated, config)?;
    plan_gpu_warmup(&input)
}

/// Build and freeze a production plan directly from validated graph metadata.
pub fn warmup_gpu_from_validated(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_execution_config(validated, config, &execution_config)
}

/// Production warmup entry point that inherits the executor's effective
/// sibling-wave limit. The limit is frozen into candidate generation before
/// planning, so a later executor cannot silently narrow an already admitted
/// wave.
pub fn warmup_gpu_from_validated_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    warmup_gpu_from_validated_with_limit(validated, &config)
}

/// Build the warmup contract from the same backend and input metadata used
/// by fixed execution, without reading payloads or advancing sampling state.
pub fn warmup_gpu_for_inputs<B: crate::Backend>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: &BTreeMap<String, crate::RuntimeValue<B>>,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_for_inputs_with_execution_config(
        validated,
        backend,
        inputs,
        config,
        &execution_config,
    )
}

/// Backend-backed warmup with an explicit execution wave limit.
pub fn warmup_gpu_for_inputs_with_execution_config<B: crate::Backend>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: &BTreeMap<String, crate::RuntimeValue<B>>,
    config: &GpuValidatedWarmupConfig,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    backend
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    config.contract = backend
        .gpu_runtime_contract(validated, inputs)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
        .ok_or_else(|| {
            GpuWarmupError::ValidatedGraph("backend has no fixed GPU contract".into())
        })?;
    apply_backend_storage_contract(validated, backend, &mut config)?;
    warmup_gpu_from_validated_with_limit(validated, &config)
}

/// Heterogeneous-fleet lifecycle entry point. The backend supplies the exact
/// runtime contract (including representation and source-shape hash) before
/// candidate exploration; the resulting plan is therefore directly usable by
/// fixed production dispatch without a second residency-based selection.
pub fn warmup_gpu_for_inputs_with_candidates<B: crate::Backend>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: &BTreeMap<String, crate::RuntimeValue<B>>,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_for_inputs_with_candidates_with_execution_config(
        validated,
        backend,
        inputs,
        config,
        device_candidates,
        owner_candidates,
        &execution_config,
    )
}

/// Heterogeneous-fleet warmup with the exact execution wave limit.
pub fn warmup_gpu_for_inputs_with_candidates_with_execution_config<B: crate::Backend>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: &BTreeMap<String, crate::RuntimeValue<B>>,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    backend
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    config.contract = backend
        .gpu_runtime_contract(validated, inputs)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
        .ok_or_else(|| {
            GpuWarmupError::ValidatedGraph("backend has no fixed GPU contract".into())
        })?;
    apply_backend_storage_contract(validated, backend, &mut config)?;
    let input = warmup_input_from_validated_with_limit(validated, &config)?;
    plan_gpu_warmup_with_device_and_layout_candidates(&input, device_candidates, owner_candidates)
}

/// Variant used when the backend has measured resident allocations separately
/// from operation profiles. Resident bytes are charged to every stage peak;
/// they are never converted into a new runtime width.
pub fn warmup_gpu_from_validated_with_resident(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    resident_bytes: &[u64],
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_resident_with_execution_config(
        validated,
        config,
        resident_bytes,
        &execution_config,
    )
}

/// Resident-aware warmup with the exact execution wave limit.
pub fn warmup_gpu_from_validated_with_resident_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    resident_bytes: &[u64],
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    let mut input = warmup_input_from_validated_with_limit(validated, &config)?;
    if resident_bytes.len() != input.contract.logical_to_physical_devices.len() {
        return Err(GpuWarmupError::DeviceCount {
            expected: input.contract.logical_to_physical_devices.len(),
            actual: resident_bytes.len(),
        });
    }
    for node in &mut input.nodes {
        for (device, cost) in node.cost.iter_mut().enumerate() {
            cost.fixed.live = cost.fixed.live.saturating_add(resident_bytes[device]);
        }
    }
    plan_gpu_warmup(&input)
}

/// Production entry point for a heterogeneous fleet. Width candidates are
/// supplied per execution site and logical device, while graph-derived range
/// and liveness metadata still comes from [`warmup_input_from_validated`].
pub fn warmup_gpu_from_validated_with_device_candidates(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_device_candidates_with_execution_config(
        validated,
        config,
        candidates,
        &execution_config,
    )
}

/// Device-candidate warmup with the exact execution wave limit.
pub fn warmup_gpu_from_validated_with_device_candidates_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    let input = warmup_input_from_validated_with_limit(validated, &config)?;
    plan_gpu_warmup_with_device_candidates(&input, candidates)
}

/// Production entry point for a heterogeneous fleet whose owner placement is
/// also a warmup choice. The candidate map is keyed by layout id and contains
/// complete, ordered owner intervals for that layout. Width and owner choices
/// are evaluated together against the same dispatch schedule and budgets.
pub fn warmup_gpu_from_validated_with_device_and_layout_candidates(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_device_and_layout_candidates_with_execution_config(
        validated,
        config,
        device_candidates,
        owner_candidates,
        &execution_config,
    )
}

/// Device/owner-candidate warmup with the exact execution wave limit.
pub fn warmup_gpu_from_validated_with_device_and_layout_candidates_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    let input = warmup_input_from_validated_with_limit(validated, &config)?;
    plan_gpu_warmup_with_device_and_layout_candidates(&input, device_candidates, owner_candidates)
}

/// Construct a fixed plan from pure profiles.  The objective is predicted
/// protocol time, subject to every per-device, pinned-host, and host budget.
/// Candidate comparison is deterministic (`time`, then widths, then input
/// order), so this function is suitable for reproducible reports.
pub fn plan_gpu_warmup(input: &GpuWarmupInput) -> Result<GpuWarmupResult, GpuWarmupError> {
    plan_gpu_warmup_with_overrides(input, None)
}

/// Plan with explicit per-device tile candidates. The outer map is keyed by
/// execution site and each inner vector is indexed by logical device. This is
/// the heterogeneous-fleet entry point; it preserves the scalar API above for
/// homogeneous fake profiles.
pub fn plan_gpu_warmup_with_device_candidates(
    input: &GpuWarmupInput,
    candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    plan_gpu_warmup_with_overrides(input, Some(candidates))
}

/// Plan with both per-device widths and owner-layout candidates. Every
/// complete layout combination is evaluated with the width candidates, so an
/// owner is not selected first and then justified by a later width choice.
pub fn plan_gpu_warmup_with_device_and_layout_candidates(
    input: &GpuWarmupInput,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let variants = owner_layout_variants(input, owner_candidates)?;
    let mut best: Option<(GpuWarmupResult, Vec<(usize, usize, usize)>)> = None;
    let mut admission_failure = None;
    for layouts in variants {
        let mut variant = input.clone();
        variant.layouts = layouts;
        let result = match plan_gpu_warmup_with_overrides(&variant, Some(device_candidates)) {
            Ok(result) => result,
            Err(error @ GpuWarmupError::ResourceExhausted { .. }) |
            Err(error @ GpuWarmupError::HostResourceExhausted { .. }) => {
                admission_failure = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };
        let key = variant
            .layouts
            .iter()
            .flat_map(|layout| {
                layout
                    .owner_intervals
                    .iter()
                    .map(|interval| (interval.device, interval.start, interval.end))
            })
            .collect::<Vec<_>>();
        let replace = best.as_ref().is_none_or(|(current, current_key)| {
            result.report.predicted_seconds.total_cmp(&current.report.predicted_seconds) ==
                Ordering::Less ||
                (result.report.predicted_seconds.total_cmp(&current.report.predicted_seconds) ==
                    Ordering::Equal &&
                    key < *current_key)
        });
        if replace {
            best = Some((result, key));
        }
    }
    best.map(|(result, _)| result)
        .ok_or_else(|| admission_failure.unwrap_or(GpuWarmupError::EmptyFleet))
}

/// Homogeneous-width convenience form of owner-layout exploration.
pub fn plan_gpu_warmup_with_layout_candidates(
    input: &GpuWarmupInput,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    plan_gpu_warmup_with_device_and_layout_candidates(input, &BTreeMap::new(), owner_candidates)
}

fn owner_layout_variants(
    input: &GpuWarmupInput,
    candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<Vec<Vec<GpuLayout>>, GpuWarmupError> {
    let single_device_layouts = input
        .nodes
        .iter()
        .filter(|node| node.column_capability == ColumnCapability::SingleDevice)
        .flat_map(|node| node.output_layouts.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut variants = vec![Vec::new()];
    for layout in &input.layouts {
        if single_device_layouts.contains(&layout.id) {
            let mut fixed = layout.clone();
            fixed.owner_intervals = gpu0_owner_intervals(fixed.columns);
            fixed.instance_device_stride = 0;
            let mut next = Vec::new();
            for prefix in variants.into_iter() {
                let mut combined = prefix;
                combined.push(fixed.clone());
                next.push(combined);
            }
            variants = next;
            continue;
        }
        let mut choices = candidates
            .get(&layout.id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|owner_intervals| {
                let mut candidate = layout.clone();
                candidate.owner_intervals = owner_intervals;
                candidate
            })
            .collect::<Vec<_>>();
        if choices.is_empty() {
            choices.push(layout.clone());
        }
        choices.sort_by_key(|candidate| {
            candidate
                .owner_intervals
                .iter()
                .map(|interval| (interval.device, interval.start, interval.end))
                .collect::<Vec<_>>()
        });
        choices.dedup_by(|left, right| left.owner_intervals == right.owner_intervals);
        let mut next = Vec::new();
        for prefix in variants.into_iter() {
            for choice in &choices {
                if next.len() >= 4096 {
                    break;
                }
                let mut combined = prefix.clone();
                combined.push(choice.clone());
                next.push(combined);
            }
            if next.len() >= 4096 {
                break;
            }
        }
        variants = next;
    }
    Ok(variants)
}

fn gpu0_owner_intervals(columns: usize) -> Vec<GpuColumnInterval> {
    (columns > 0)
        .then_some(GpuColumnInterval { device: 0, start: 0, end: columns })
        .into_iter()
        .collect()
}

fn normalize_single_device_layout(layout: &GpuLayout, single_device: bool) -> GpuLayout {
    if !single_device {
        return layout.clone();
    }
    let mut fixed = layout.clone();
    fixed.owner_intervals = gpu0_owner_intervals(fixed.columns);
    fixed.instance_device_stride = 0;
    fixed
}

fn plan_gpu_warmup_with_overrides(
    input: &GpuWarmupInput,
    overrides: Option<&BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    input.contract.validate().map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?;
    let devices = input.contract.logical_to_physical_devices.len();
    if devices == 0 {
        return Err(GpuWarmupError::EmptyFleet);
    }
    let budgets = &input.contract.device_budgets;
    let single_device_layouts = input
        .nodes
        .iter()
        .filter(|node| node.column_capability == ColumnCapability::SingleDevice)
        .flat_map(|node| node.output_layouts.iter().copied())
        .collect::<BTreeSet<_>>();
    let layouts = input
        .layouts
        .iter()
        .map(|layout| {
            normalize_single_device_layout(layout, single_device_layouts.contains(&layout.id))
        })
        .map(|layout| layout_for(&layout, devices))
        .collect::<Vec<_>>();
    let layout_map = layouts.iter().map(|layout| (layout.id, layout)).collect::<BTreeMap<_, _>>();
    let mut loop_choices = Vec::new();
    let mut loop_widths = BTreeMap::new();
    for loop_input in &input.loops {
        let configured_bound = loop_input
            .wave_candidates
            .iter()
            .copied()
            .filter(|candidate| *candidate > 0)
            .max()
            .unwrap_or(1)
            .min(loop_input.loop_count.max(1));
        let mut candidates = candidate_values(&loop_input.wave_candidates, configured_bound);
        if loop_input.nested {
            candidates = vec![1];
        }
        if candidates.is_empty() {
            return Err(GpuWarmupError::NoLoopCandidate(loop_input.key));
        }
        // The loop itself has no cost; choose the largest admissible wave so
        // that independent instances can occupy the fleet.  Node profiles may
        // still reject it, in which case the planner tries smaller candidates.
        candidates.sort_unstable();
        loop_widths.insert(loop_input.key, candidates);
    }
    let mut reports = Vec::new();
    let mut selections: Vec<Option<(usize, NodeSelection)>> = vec![None; input.nodes.len()];
    let mut chosen_loop_w = BTreeMap::new();
    for loop_input in &input.loops {
        let candidates = loop_widths
            .get(&loop_input.key)
            .ok_or(GpuWarmupError::NoLoopCandidate(loop_input.key))?;
        let grouped = input
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.loop_site == Some(loop_input.key))
            .collect::<Vec<_>>();
        let mut best: Option<(f64, usize, Vec<(usize, NodeSelection)>)> = None;
        let mut admission_failure = None;
        for wave in candidates {
            let mut current = Vec::new();
            let mut total = 0.0;
            let mut rejected = false;
            for (index, node) in &grouped {
                let node_layouts = output_layouts(node, &layout_map)?;
                let override_candidates = overrides.and_then(|map| map.get(&node.key));
                match choose_node_with_candidates(
                    node,
                    &node_layouts,
                    devices,
                    budgets,
                    *wave,
                    loop_input.loop_count,
                    override_candidates.map(|values| values.as_slice()),
                ) {
                    Ok(selection) => {
                        total += selection.seconds;
                        current.push((*index, selection));
                    }
                    Err(error @ GpuWarmupError::ResourceExhausted { .. }) |
                    Err(error @ GpuWarmupError::HostResourceExhausted { .. }) => {
                        admission_failure = Some(error);
                        rejected = true;
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
            if rejected {
                continue;
            }
            let replace = best.as_ref().is_none_or(|(seconds, best_wave, _)| {
                total.total_cmp(seconds) == Ordering::Less ||
                    (total.total_cmp(seconds) == Ordering::Equal && *wave < *best_wave)
            });
            if replace {
                best = Some((total, *wave, current));
            }
        }
        if let Some((_, wave, selected)) = best {
            chosen_loop_w.insert(loop_input.key, wave);
            for (index, selection) in selected {
                selections[index] = Some((wave, selection));
            }
        } else if !grouped.is_empty() {
            return Err(
                admission_failure.unwrap_or(GpuWarmupError::NoNodeCandidate(grouped[0].1.key))
            );
        }
    }
    for (index, node) in input.nodes.iter().enumerate() {
        if selections[index].is_none() {
            let node_layouts = output_layouts(node, &layout_map)?;
            let override_candidates = overrides.and_then(|map| map.get(&node.key));
            let selection = choose_node_with_candidates(
                node,
                &node_layouts,
                devices,
                budgets,
                1,
                1,
                override_candidates.map(|values| values.as_slice()),
            )?;
            selections[index] = Some((1, selection));
        }
        let (_, selection) = selections[index].take().expect("selection installed");
        reports.push(selection.report.clone());
    }
    for loop_input in &input.loops {
        let wave = chosen_loop_w
            .get(&loop_input.key)
            .copied()
            .or_else(|| loop_widths.get(&loop_input.key).and_then(|values| values.last().copied()))
            .unwrap_or(1);
        let wave = if loop_input.nested { 1 } else { wave.min(loop_input.loop_count.max(1)) };
        loop_choices.push(GpuLoopChoice {
            key: loop_input.key,
            loop_count: loop_input.loop_count,
            wave_instances: wave,
            tail_instances: if loop_input.loop_count == 0 {
                0
            } else {
                loop_input.loop_count % wave
            },
        });
    }
    let mut nodes = Vec::with_capacity(input.nodes.len());
    for node in &input.nodes {
        let report = reports
            .iter()
            .find(|report| report.key == node.key)
            .ok_or(GpuWarmupError::NoNodeCandidate(node.key))?;
        nodes.push(GpuNodeChoice {
            key: node.key,
            loop_site: node.loop_site,
            operation_identity: node.operation_identity,
            effective_operation: node.effective_operation,
            column_capability: node.column_capability,
            output_layouts: node.output_layouts.clone(),
            columns_per_job: report.columns_per_job.clone(),
            implementation_variant: node.implementation_variant.clone(),
            preimage_max_attempts: node.preimage_max_attempts,
        });
    }
    let total_seconds = reports.iter().map(|report| report.predicted_seconds).sum();
    let limiting_stage = reports
        .iter()
        .max_by(|left, right| left.predicted_seconds.total_cmp(&right.predicted_seconds))
        .map(|report| report.key);
    let plan = FrozenGpuPlan::new(input.contract.clone(), layouts, loop_choices, nodes)
        .map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?;
    Ok(GpuWarmupResult {
        plan,
        report: GpuWarmupReport {
            predicted_seconds: total_seconds,
            limiting_stage,
            stages: reports,
            reason: "minimum predicted protocol time under fixed resource budgets".into(),
        },
    })
}

/// Short alias used by runtime callers.  It remains pure and does not imply
/// that a production execution may invoke it after the plan is frozen.
pub fn warmup_gpu(input: &GpuWarmupInput) -> Result<GpuWarmupResult, GpuWarmupError> {
    plan_gpu_warmup(input)
}

impl fmt::Display for GpuProfileProvenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SizeQuery => "size-query",
            Self::MeasuredPoint => "measured-point",
            Self::ConservativeEstimate => "conservative-estimate",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_column_policy::{ColumnRange, map_output_range_to_inputs};
    use mxx_ir_core::{
        node::{ConcatAxis, NodeKind},
        types::ConcreteWireType,
    };
    use num_bigint::BigInt;

    fn contract(devices: usize, budget: u64) -> GpuPlanContract {
        GpuPlanContract {
            graph_specification_hash: [1; 32],
            backend_identity: "fake".into(),
            logical_to_physical_devices: (0..devices).collect(),
            device_budgets: (0..devices)
                .map(|device| GpuDeviceBudget {
                    device,
                    device_bytes: budget,
                    pinned_host_bytes: budget,
                    host_bytes: budget,
                })
                .collect(),
            shape_contract_hash: [2; 32],
            backend_revision: "test".into(),
        }
    }

    fn layout(columns: usize, devices: usize) -> GpuLayout {
        GpuLayout {
            id: 1,
            columns,
            rows: 1,
            ring_dimension: 1,
            representation: "matrix".into(),
            instance_device_stride: 0,
            owner_intervals: balanced_intervals(columns, devices),
        }
    }

    fn node(columns: usize, costs: Vec<GpuStageCostModel>, widths: Vec<usize>) -> GpuWarmupNode {
        GpuWarmupNode {
            key: GpuExecutionSiteKey { site: 1, shape_class: 0, instance_class: 0 },
            operation_identity: [3; 32],
            effective_operation: EffectiveGpuOperation::MatrixAdd,
            column_capability: ColumnCapability::SameColumns,
            output_layouts: vec![1],
            implementation_variant: "fake".into(),
            loop_site: Some(GpuLoopSiteKey { site: 7, shape_class: 0 }),
            output_columns: columns,
            tile_widths: widths,
            cost: costs,
            provenance: GpuProfileProvenance::MeasuredPoint,
            preimage_max_attempts: None,
            preimage_footprint: None,
        }
    }

    #[test]
    fn joint_wave_and_tile_budget() {
        let costs = vec![
            GpuStageCostModel {
                per_instance: GpuResourceCost { live: 30, ..Default::default() },
                per_job: GpuResourceCost { scratch: 10, ..Default::default() },
                per_job_column: GpuResourceCost { scratch: 2, ..Default::default() },
                time: GpuTimeModel {
                    per_column_seconds: 1.0,
                    per_job_seconds: 0.1,
                    wave_overhead_seconds: 0.5,
                    ..Default::default()
                },
                ..Default::default()
            };
            2
        ];
        let input = GpuWarmupInput {
            contract: contract(2, 100),
            layouts: vec![layout(8, 2)],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 8,
                wave_candidates: vec![1, 2, 4, 8],
                nested: false,
            }],
            nodes: vec![node(8, costs, vec![1, 2, 4, 8])],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert!(result.plan.loops[0].wave_instances >= 2);
        assert!(result.plan.nodes[0].columns_per_job.iter().all(|width| *width > 0));
        assert!(result.report.stages[0].peak.iter().all(|peak| peak.device_bytes() <= 100));
    }

    #[test]
    fn tile_width_is_not_largest_feasible_width() {
        let costs = vec![
            GpuStageCostModel {
                time: GpuTimeModel {
                    per_column_seconds: 1.0,
                    per_job_seconds: 4.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            1
        ];
        let input = GpuWarmupInput {
            contract: contract(1, 10_000),
            layouts: vec![layout(8, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(8, costs, vec![1, 2, 4, 8]) }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![8]);
        // A nonlinear wide-tile penalty makes smaller tiles win despite all
        // candidates being feasible; selecting the largest feasible width is
        // therefore not a valid planner strategy.
        let mut slower = input.clone();
        slower.nodes[0].cost[0].time = GpuTimeModel {
            per_column_seconds: 1.0,
            per_column_squared_seconds: 0.5,
            per_job_seconds: 4.0,
            ..Default::default()
        };
        assert_eq!(plan_gpu_warmup(&slower).unwrap().plan.nodes[0].columns_per_job, vec![2]);
    }

    #[test]
    fn large_loops_use_compact_plans() {
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(1, 1)],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 2, shape_class: 0 },
                loop_count: usize::MAX,
                wave_candidates: vec![1, 8, 64],
                nested: false,
            }],
            nodes: vec![],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.loops.len(), 1);
        assert_eq!(result.plan.loops[0].loop_count, usize::MAX);
    }

    #[test]
    fn graph_wave_candidates_include_non_power_of_two_fleet_width() {
        assert_eq!(graph_loop_wave_candidates(8, 3, 64), vec![1, 2, 3, 4, 8]);
        assert!(graph_loop_wave_candidates(usize::MAX, 3, 8192).contains(&8192));
        let mut output = layout(1, 3);
        output.instance_device_stride = 1;
        let input = GpuWarmupInput {
            contract: contract(3, 3),
            layouts: vec![output],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 8,
                wave_candidates: graph_loop_wave_candidates(8, 3, 64),
                nested: false,
            }],
            nodes: vec![GpuWarmupNode {
                loop_site: Some(GpuLoopSiteKey { site: 7, shape_class: 0 }),
                cost: vec![
                    GpuStageCostModel {
                        per_instance: GpuResourceCost { scratch: 1, ..Default::default() },
                        time: GpuTimeModel {
                            per_job_seconds: 1.0,
                            wave_overhead_seconds: 1.0,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    3
                ],
                ..node(1, vec![], vec![1])
            }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.loops[0].wave_instances, 3);
    }

    #[test]
    fn nested_loop_budget_is_not_multiplied() {
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(1, 1)],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 2, shape_class: 0 },
                loop_count: 8,
                wave_candidates: vec![1, 4],
                nested: true,
            }],
            nodes: vec![],
        };
        assert_eq!(plan_gpu_warmup(&input).unwrap().plan.loops[0].wave_instances, 1);
    }

    #[test]
    fn full_outputs_and_caches_are_accounted() {
        let costs = vec![GpuStageCostModel {
            fixed: GpuResourceCost { outputs: 90, caches: 20, ..Default::default() },
            ..Default::default()
        }];
        let input = GpuWarmupInput {
            contract: contract(1, 100),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(4, costs, vec![1, 4]) }],
        };
        assert!(matches!(plan_gpu_warmup(&input), Err(GpuWarmupError::ResourceExhausted { .. })));
    }

    #[test]
    fn deterministic_tie_break_is_stable() {
        let costs = vec![GpuStageCostModel::default()];
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(4, costs, vec![4, 2, 1]) }],
        };
        let first = plan_gpu_warmup(&input).unwrap();
        let second = plan_gpu_warmup(&input).unwrap();
        assert_eq!(first.plan, second.plan);
    }

    #[test]
    fn per_device_tile_candidates_and_budgets_are_jointly_selected() {
        let costs = vec![
            GpuStageCostModel {
                fixed: GpuResourceCost { scratch: 4, ..Default::default() },
                time: GpuTimeModel { per_job_seconds: 1.0, ..Default::default() },
                ..Default::default()
            },
            GpuStageCostModel {
                fixed: GpuResourceCost { scratch: 4, ..Default::default() },
                time: GpuTimeModel { per_job_seconds: 1.0, ..Default::default() },
                ..Default::default()
            },
        ];
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![layout(8, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(8, costs, vec![1, 2, 4, 8]) }],
        };
        let key = input.nodes[0].key;
        let candidates = BTreeMap::from([(key, vec![vec![2], vec![1, 4]])]);
        let result = plan_gpu_warmup_with_device_candidates(&input, &candidates).unwrap();
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![2, 4]);
    }

    #[test]
    fn owner_layout_and_device_width_are_jointly_selected() {
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![layout(8, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                loop_site: None,
                cost: vec![
                    GpuStageCostModel {
                        time: GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
                        ..Default::default()
                    },
                    GpuStageCostModel {
                        time: GpuTimeModel { per_column_seconds: 0.1, ..Default::default() },
                        ..Default::default()
                    },
                ],
                ..node(8, vec![], vec![])
            }],
        };
        let owners = BTreeMap::from([(
            1,
            vec![
                vec![
                    GpuColumnInterval { device: 0, start: 0, end: 4 },
                    GpuColumnInterval { device: 1, start: 4, end: 8 },
                ],
                vec![
                    GpuColumnInterval { device: 0, start: 0, end: 2 },
                    GpuColumnInterval { device: 1, start: 2, end: 8 },
                ],
            ],
        )]);
        let widths = BTreeMap::from([(input.nodes[0].key, vec![vec![2], vec![2]])]);
        let result =
            plan_gpu_warmup_with_device_and_layout_candidates(&input, &widths, &owners).unwrap();
        assert_eq!(result.plan.layout(1).unwrap().owner_intervals[0].end, 2);
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![2, 2]);
    }

    #[test]
    fn preimage_requires_cold_footprint_and_fixed_attempts() {
        let costs = vec![GpuStageCostModel::default()];
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::PreimageSample,
                column_capability: ColumnCapability::FixedOperandColumns,
                preimage_max_attempts: Some(32),
                ..node(4, costs, vec![1, 2])
            }],
        };
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::MissingPreimageProfile { .. })
        ));
        let mut input = input;
        input.nodes[0].preimage_footprint = Some(vec![GpuPreimageFootprint {
            persistent: GpuResourceCost { live: 11, ..Default::default() },
            compact: GpuResourceCost { outputs: 13, ..Default::default() },
            scratch: GpuResourceCost { scratch: 17, ..Default::default() },
            control: GpuResourceCost { pinned_host: 19, host: 23, ..Default::default() },
            cold_cache: GpuResourceCost { caches: 29, ..Default::default() },
            cold_transient_workspace: GpuResourceCost { scratch: 7, ..Default::default() },
            certified_tile_width: Some(1),
            per_tile: GpuResourceCost::zero(),
            width_sizing_model: None,
            provenance: GpuProfileProvenance::MeasuredPoint,
        }]);
        let result = plan_gpu_warmup(&input).unwrap();
        let peak = result.report.stages[0].peak[0];
        assert_eq!(peak.live, 11);
        assert_eq!(peak.outputs, 13);
        assert_eq!(peak.scratch, 24);
        assert_eq!(peak.caches, 29);
        assert_eq!(peak.pinned_host, 19);
        assert_eq!(peak.host, 23);
        assert_eq!(result.plan.nodes[0].preimage_max_attempts, Some(32));
        input.contract.device_budgets[0].device_bytes = 23;
        assert!(matches!(plan_gpu_warmup(&input), Err(GpuWarmupError::ResourceExhausted { .. })));
    }

    #[test]
    fn preimage_footprint_scales_with_each_tile_candidate() {
        let mut input = GpuWarmupInput {
            contract: contract(1, 15),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::PreimageSample,
                column_capability: ColumnCapability::FixedOperandColumns,
                tile_widths: vec![1, 2],
                preimage_max_attempts: Some(8),
                preimage_footprint: Some(vec![GpuPreimageFootprint {
                    certified_tile_width: Some(1),
                    per_tile: GpuResourceCost { scratch: 5, ..Default::default() },
                    provenance: GpuProfileProvenance::MeasuredPoint,
                    ..Default::default()
                }]),
                ..node(4, vec![GpuStageCostModel::default()], vec![1, 2])
            }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![1]);
        assert_eq!(result.report.stages[0].peak[0].scratch, 5);
        input.contract.device_budgets[0].device_bytes = 4;
        assert!(matches!(plan_gpu_warmup(&input), Err(GpuWarmupError::ResourceExhausted { .. })));
    }

    #[test]
    fn preimage_certified_width_does_not_authorize_wider_tiles_without_model() {
        let input = GpuWarmupInput {
            contract: contract(1, 1 << 20),
            layouts: vec![layout(64, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::PreimageSample,
                column_capability: ColumnCapability::FixedOperandColumns,
                tile_widths: vec![64],
                preimage_max_attempts: Some(1),
                preimage_footprint: Some(vec![GpuPreimageFootprint {
                    certified_tile_width: Some(1),
                    per_tile: GpuResourceCost::zero(),
                    width_sizing_model: None,
                    provenance: GpuProfileProvenance::MeasuredPoint,
                    ..Default::default()
                }]),
                ..node(64, vec![GpuStageCostModel::default()], vec![64])
            }],
        };
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::MissingPreimageProfile { .. })
        ));
    }

    #[test]
    fn dcrt_bytes_count_ordered_active_towers_and_keep_compact_separate() {
        let matrix = ConcreteWireType::Matrix(mxx_ir_core::types::ConcreteMatrixType {
            modulus: BigInt::from(97u64),
            ring_dimension: 8,
            rows: 2,
            columns: 3,
        });
        assert_eq!(validated_wire_bytes(&matrix, &BTreeMap::new(), 4, 8), 2 * 3 * 8 * 4 * 8);
        let compact = ConcreteWireType::SmallMatrix {
            matrix: mxx_ir_core::types::ConcreteMatrixType {
                modulus: BigInt::from(97u64),
                ring_dimension: 8,
                rows: 2,
                columns: 3,
            },
            max_coefficient_bound: BigInt::from(257u64),
        };
        assert_eq!(validated_wire_bytes(&compact, &BTreeMap::new(), 4, 8), 2 * 3 * 8 * 3);
        let mut descriptors = BTreeMap::new();
        descriptors.insert(
            matrix.clone(),
            GpuPhysicalStorageDescriptor {
                representation: GpuStorageRepresentation::FullDcrt,
                ordered_basis: vec![17, 19, 23],
                level: 1,
                limb_bytes: 4,
            },
        );
        assert_eq!(validated_wire_bytes(&matrix, &descriptors, 1, 8), 2 * 3 * 8 * 2 * 4);
        let top_level = GpuPhysicalStorageDescriptor {
            representation: GpuStorageRepresentation::FullDcrt,
            ordered_basis: vec![17, 19, 23, 29],
            level: 3,
            limb_bytes: 4,
        };
        assert!(valid_storage_descriptor(&top_level));
        assert_eq!(top_level.active_towers(), 4);
        assert_eq!(
            validated_wire_bytes(
                &matrix,
                &BTreeMap::from([(matrix.clone(), top_level.clone())]),
                1,
                8,
            ),
            2 * 3 * 8 * 4 * 4
        );
        let three_tower = GpuPhysicalStorageDescriptor {
            ordered_basis: vec![17, 19, 23],
            level: 2,
            ..top_level.clone()
        };
        let three_bytes =
            validated_wire_bytes(&matrix, &BTreeMap::from([(matrix.clone(), three_tower)]), 1, 8);
        let four_bytes = validated_wire_bytes(
            &matrix,
            &BTreeMap::from([(matrix.clone(), top_level.clone())]),
            1,
            8,
        );
        assert!(three_bytes < four_bytes);
        assert!(!valid_storage_descriptor(&GpuPhysicalStorageDescriptor {
            ordered_basis: Vec::new(),
            ..top_level.clone()
        }));
        assert!(!valid_storage_descriptor(&GpuPhysicalStorageDescriptor { level: 4, ..top_level }));
    }

    #[test]
    fn backend_storage_contract_replaces_empty_and_rejects_one_tower_spoof() {
        let matrix = mxx_ir_core::types::ConcreteMatrixType {
            modulus: BigInt::from(97u64),
            ring_dimension: 8,
            rows: 1,
            columns: 2,
        };
        let wire = ConcreteWireType::Matrix(matrix.clone());
        let backend = BackendStorageContract {
            descriptors: BTreeMap::from([(
                matrix,
                BackendStorageDescriptor {
                    representation: "full_dcrt".into(),
                    ordered_crt_basis: vec![17, 19, 23, 29],
                    level: 3,
                    limb_bytes: 8,
                },
            )]),
            active_crt_towers: 4,
            crt_limb_bytes: 8,
        };
        let wire_types = BTreeSet::from([wire.clone()]);
        let mut config = GpuValidatedWarmupConfig {
            contract: contract(1, 1 << 20),
            layouts: vec![],
            default_tile_widths: vec![1],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "backend-storage-test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        assert!(caller_storage_metadata_matches(&config, &wire_types, &backend).is_ok());
        config.storage_descriptors.insert(
            wire,
            GpuPhysicalStorageDescriptor {
                representation: GpuStorageRepresentation::FullDcrt,
                ordered_basis: vec![17],
                level: 0,
                limb_bytes: 8,
            },
        );
        assert!(caller_storage_metadata_matches(&config, &wire_types, &backend).is_err());
    }

    #[test]
    fn tiny_column_instances_choose_a_real_rotating_wave() {
        let mut output = layout(1, 4);
        output.instance_device_stride = 1;
        let input = GpuWarmupInput {
            contract: contract(4, 1000),
            layouts: vec![output],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 16,
                wave_candidates: vec![1, 2, 4, 8],
                nested: false,
            }],
            nodes: vec![node(
                1,
                vec![
                    GpuStageCostModel {
                        time: GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
                        ..Default::default()
                    };
                    4
                ],
                vec![1],
            )],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.loops[0].wave_instances, 4);
        assert_eq!(result.report.predicted_seconds, 4.0);
        let node = &result.plan.nodes[0];
        let layout = result.plan.layout(node.output_layouts[0]).unwrap();
        for instance in 0..4 {
            assert_eq!(
                layout.schedule(&node.columns_per_job, instance).unwrap().intervals()[0].device,
                instance
            );
        }
        // A replay-filtered request retains slot three rather than becoming
        // the first placement after compaction.
        assert_eq!(layout.schedule(&node.columns_per_job, 3).unwrap().intervals()[0].device, 3);
    }

    #[test]
    fn validated_multi_output_shapes_and_scope_keys_are_preserved() {
        use mxx_dsl::{DslContext, MatType, Ring, Subgraph};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let pair = Subgraph::define(
            "pair",
            (MatType(ring.matrix_type((2, 3))), MatType(ring.matrix_type((4, 5)))),
            |values| Ok(values),
        )
        .unwrap();
        let (left, right) =
            pair.call((ring.input("left", (2, 3)), ring.input("right", (4, 5)))).unwrap();
        let graph = DslContext::new("port-shapes")
            .output("left", left)
            .unwrap()
            .output("right", right)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(2, 1 << 24),
            layouts: vec![layout(3, 2)],
            default_tile_widths: vec![1, 2],
            default_cost: vec![GpuStageCostModel::default(); 2],
            default_implementation_variant: "test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        let multi = input.nodes.iter().find(|node| node.output_layouts.len() == 2).unwrap();
        let shapes = multi
            .output_layouts
            .iter()
            .map(|id| {
                let layout = input.layouts.iter().find(|layout| layout.id == *id).unwrap();
                (layout.rows, layout.columns)
            })
            .collect::<Vec<_>>();
        assert_eq!(shapes, vec![(2, 3), (4, 5)]);
        for scope in graph.scopes.keys() {
            let class = scope_shape_class(&graph, scope).unwrap();
            assert!(input.nodes.iter().any(|node| node.key.shape_class == class));
        }
        assert!(input.nodes.iter().any(|node| node.key.shape_class != multi.key.shape_class));
    }

    #[test]
    fn per_device_scratch_is_reused_across_sibling_jobs() {
        let costs = vec![GpuStageCostModel {
            per_job: GpuResourceCost { scratch: 60, ..Default::default() },
            time: GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
            ..Default::default()
        }];
        let input = GpuWarmupInput {
            contract: contract(1, 100),
            layouts: vec![layout(4, 1)],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 5,
                wave_candidates: vec![4],
                nested: false,
            }],
            nodes: vec![node(4, costs, vec![1])],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.report.stages[0].peak[0].scratch, 60);
        assert_eq!(result.report.predicted_seconds, 20.0);
    }

    #[test]
    fn validated_graph_entry_uses_site_profiles_and_liveness() {
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let graph = DslContext::new("warmup-validated-entry")
            .output("out", ring.input("x", (2, 4)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut config = GpuValidatedWarmupConfig {
            contract: contract(1, 1 << 20),
            layouts: vec![layout(4, 1)],
            default_tile_widths: vec![1, 2, 4],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "validated-default".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        config.contract.graph_specification_hash = [0; 32];
        config.effective_operation_identities.insert((0, 0), [9; 32]);
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        assert_ne!(input.contract.graph_specification_hash, [0; 32]);
        assert!(!input.nodes.is_empty());
        assert_eq!(input.nodes[0].operation_identity, [9; 32]);
        assert!(input.nodes.iter().all(|node| node.implementation_variant.contains("live=")));
        let result = warmup_gpu_from_validated(&graph, &config).unwrap();
        assert_eq!(
            result.plan.contract.graph_specification_hash,
            input.contract.graph_specification_hash
        );
    }

    #[test]
    fn validated_billion_loop_is_bounded_before_planning() {
        use mxx_dsl::{DslContext, Ring, parallel};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let value = ring.input("x", (2, 4));
        let family = parallel(usize::MAX, |_| Ok(value.clone())).unwrap();
        let graph = DslContext::new("bounded-validated-loop")
            .output("out", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(2, u64::MAX),
            layouts: vec![layout(4, 2)],
            default_tile_widths: vec![1, 2, 4],
            default_cost: vec![GpuStageCostModel::default(); 2],
            default_implementation_variant: "bounded-loop-test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        let loop_input =
            input.loops.iter().find(|loop_input| loop_input.loop_count == usize::MAX).unwrap();
        assert!(loop_input.wave_candidates.iter().all(|candidate| *candidate <= 64));
        assert!(loop_input.wave_candidates.contains(&4));
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.loops.len(), 1);
        assert!(result.plan.loops[0].wave_instances <= 64);
    }

    #[test]
    fn validated_loop_uses_overhead_to_choose_wave_four_on_one_gpu() {
        use mxx_dsl::{DslContext, Ring, parallel};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let value = ring.input("x", (1, 1));
        let family = parallel(8usize, |_| Ok(value.clone() + value.clone())).unwrap();
        let graph = DslContext::new("timed-validated-loop")
            .output("out", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(1, 4096),
            layouts: vec![layout(1, 1)],
            default_tile_widths: vec![1],
            default_cost: vec![GpuStageCostModel {
                per_instance: GpuResourceCost { scratch: 512, ..Default::default() },
                time: GpuTimeModel {
                    per_job_seconds: 1.0,
                    wave_overhead_seconds: 1.0,
                    ..Default::default()
                },
                ..Default::default()
            }],
            default_implementation_variant: "timed-loop-test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        let result = warmup_gpu_from_validated(&graph, &config).unwrap();
        let loop_input = input.loops.iter().find(|loop_input| loop_input.loop_count == 8).unwrap();
        assert!(loop_input.wave_candidates.contains(&4));
        assert_eq!(
            result.plan.loops.iter().find(|choice| choice.loop_count == 8).unwrap().wave_instances,
            4
        );
    }

    #[test]
    fn every_public_pure_entry_applies_execution_wave_limit() {
        use mxx_dsl::{DslContext, Ring, parallel};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let value = ring.input("x", (1, 1));
        let family = parallel(16usize, |_| Ok(value.clone() + value.clone())).unwrap();
        let graph = DslContext::new("public-entry-wave-limit")
            .output("out", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(1, u64::MAX),
            layouts: vec![layout(1, 1)],
            default_tile_widths: vec![1],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "public-entry-limit".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let execution_config = crate::executor::ExecutionConfig {
            max_parallel_instances: NonZeroUsize::new(8).unwrap(),
            ..Default::default()
        };
        let input =
            warmup_input_from_validated_with_execution_config(&graph, &config, &execution_config)
                .unwrap();
        assert!(input.loops.iter().all(|loop_input| {
            loop_input.wave_candidates.iter().all(|candidate| *candidate <= 8)
        }));
        let result =
            warmup_gpu_from_validated_with_execution_config(&graph, &config, &execution_config)
                .unwrap();
        assert!(result.plan.loops.iter().all(|choice| choice.wave_instances <= 8));
        let resident = warmup_gpu_from_validated_with_resident_with_execution_config(
            &graph,
            &config,
            &[0],
            &execution_config,
        )
        .unwrap();
        assert!(resident.plan.loops.iter().all(|choice| choice.wave_instances <= 8));
        let device = warmup_gpu_from_validated_with_device_candidates_with_execution_config(
            &graph,
            &config,
            &BTreeMap::new(),
            &execution_config,
        )
        .unwrap();
        assert!(device.plan.loops.iter().all(|choice| choice.wave_instances <= 8));
        let owner =
            warmup_gpu_from_validated_with_device_and_layout_candidates_with_execution_config(
                &graph,
                &config,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &execution_config,
            )
            .unwrap();
        assert!(owner.plan.loops.iter().all(|choice| choice.wave_instances <= 8));
    }

    #[test]
    fn unsupported_effective_operation_fails_warmup() {
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let graph = DslContext::new("unsupported-effective-operation")
            .output("out", ring.input("x", (2, 2)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut config = GpuValidatedWarmupConfig {
            contract: contract(1, 1 << 20),
            layouts: vec![layout(2, 1)],
            default_tile_widths: vec![1, 2],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "unsupported-test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        config.effective_operations.insert((0, 0), EffectiveGpuOperation::Unsupported);
        assert!(matches!(
            warmup_input_from_validated(&graph, &config),
            Err(GpuWarmupError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn warmup_does_not_mutate_production_state() {
        let costs = vec![GpuStageCostModel::default()];
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(4, costs, vec![1, 2]) }],
        };
        let before = input.clone();
        let _ = plan_gpu_warmup(&input).unwrap();
        assert_eq!(input, before);
    }

    #[test]
    fn single_device_and_host_boundaries_have_costs() {
        let costs = vec![GpuStageCostModel {
            fixed: GpuResourceCost {
                transfers: 5,
                pinned_host: 101,
                host: 7,
                ..Default::default()
            },
            ..Default::default()
        }];
        let input = GpuWarmupInput {
            contract: contract(1, 100),
            layouts: vec![layout(1, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(1, costs, vec![1]) }],
        };
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::HostResourceExhausted { .. })
        ));
    }

    #[test]
    fn single_device_uses_one_owner_for_full_operation_cost() {
        let costs = vec![
            GpuStageCostModel {
                per_output_column: GpuResourceCost { outputs: 1, ..Default::default() },
                time: GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
                ..Default::default()
            },
            GpuStageCostModel {
                per_output_column: GpuResourceCost { outputs: 1, ..Default::default() },
                time: GpuTimeModel { per_column_seconds: 100.0, ..Default::default() },
                ..Default::default()
            },
        ];
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![layout(4, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::TrapdoorSample,
                column_capability: ColumnCapability::SingleDevice,
                ..node(4, costs, vec![1, 2, 4])
            }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.report.predicted_seconds, 4.0);
        assert_eq!(result.report.stages[0].peak[0].outputs, 4);
        assert_eq!(result.report.stages[0].peak[1].outputs, 0);
    }

    #[test]
    fn single_device_owner_candidates_are_forced_to_gpu0() {
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![GpuLayout {
                owner_intervals: vec![GpuColumnInterval { device: 1, start: 0, end: 4 }],
                ..layout(4, 2)
            }],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::TrapdoorSample,
                column_capability: ColumnCapability::SingleDevice,
                ..node(4, vec![GpuStageCostModel::default(); 2], vec![1, 2, 4])
            }],
        };
        let owners = BTreeMap::from([(
            1,
            vec![
                vec![GpuColumnInterval { device: 1, start: 0, end: 4 }],
                vec![GpuColumnInterval { device: 0, start: 0, end: 4 }],
            ],
        )]);
        let result =
            plan_gpu_warmup_with_device_and_layout_candidates(&input, &BTreeMap::new(), &owners)
                .unwrap();
        let layout = result.plan.layout(1).unwrap();
        assert_eq!(layout.instance_device_stride, 0);
        assert_eq!(layout.owner_intervals, gpu0_owner_intervals(4));
    }

    #[test]
    fn host_control_time_does_not_create_column_union_jobs() {
        let time = GpuTimeModel {
            fixed_seconds: 2.0,
            per_job_seconds: 1.0,
            transfer_seconds: 0.5,
            wave_overhead_seconds: 0.25,
            ..Default::default()
        };
        assert_eq!(gpu_non_column_batch_wave_time(3, 0, 0, &[time]).unwrap(), 11.25);
    }

    #[test]
    fn huge_width_one_union_uses_compressed_schedule_classes() {
        let port0 = GpuColumnSchedule::new(
            usize::MAX,
            vec![1, 1],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            usize::MAX,
            vec![1, 1],
            vec![GpuColumnInterval { device: 1, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let predicted = gpu_multi_output_batch_wave_time(
            &[vec![port0, port1]],
            &[
                GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
                GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
            ],
        )
        .unwrap();
        assert!(predicted.is_finite());
    }

    #[test]
    fn compressed_union_matches_enumerated_asymmetric_table() {
        fn enumerated(schedules: &[Vec<GpuColumnSchedule>], times: &[GpuTimeModel]) -> f64 {
            let instances = schedules.len();
            let ports = schedules[0].len();
            let by_port = (0..ports)
                .map(|port| {
                    schedules.iter().map(|instance| instance[port].clone()).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut total = 0.0;
            for wave in
                crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, instances).unwrap()
            {
                let mut by_device = BTreeMap::<usize, f64>::new();
                for job in wave.unwrap() {
                    let width = job.range.end - job.range.start;
                    *by_device.entry(job.device).or_default() +=
                        times[job.device].job_seconds(width);
                }
                total += by_device.values().copied().fold(0.0, f64::max) +
                    times.iter().map(|time| time.wave_overhead_seconds).fold(0.0, f64::max);
            }
            total
        }

        let time = [GpuTimeModel { per_column_seconds: 1.0, ..Default::default() }; 2];
        let cases = [
            (
                4,
                vec![2, 1],
                vec![GpuColumnInterval { device: 0, start: 0, end: 4 }],
                vec![GpuColumnInterval { device: 1, start: 0, end: 4 }],
                4.0,
            ),
            (
                5,
                vec![2, 3],
                vec![GpuColumnInterval { device: 0, start: 0, end: 5 }],
                vec![GpuColumnInterval { device: 1, start: 0, end: 5 }],
                5.0,
            ),
        ];
        for (columns, widths, owners0, owners1, expected) in cases {
            let port0 =
                GpuColumnSchedule::new(columns, vec![widths[0], widths[0]], owners0).unwrap();
            let port1 =
                GpuColumnSchedule::new(columns, vec![widths[1], widths[1]], owners1).unwrap();
            let schedules = vec![vec![port0, port1]];
            assert_eq!(enumerated(&schedules, &time), expected);
            assert_eq!(gpu_multi_output_batch_wave_time(&schedules, &time).unwrap(), expected);
        }
    }

    #[test]
    fn compressed_union_phase_drift_matches_production() {
        let port0 = GpuColumnSchedule::new(
            6,
            vec![2],
            vec![GpuColumnInterval { device: 0, start: 0, end: 6 }],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            6,
            vec![3],
            vec![GpuColumnInterval { device: 0, start: 0, end: 6 }],
        )
        .unwrap();
        let times = [GpuTimeModel { per_column_seconds: 1.0, ..Default::default() }];
        let by_port = vec![vec![port0.clone()], vec![port1.clone()]];
        let expected = crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, 1)
            .unwrap()
            .map(|wave| {
                wave.unwrap()
                    .into_iter()
                    .map(|job| (job.range.end - job.range.start) as f64)
                    .sum::<f64>()
            })
            .sum::<f64>();
        assert_eq!(expected, 6.0);
        assert_eq!(
            gpu_multi_output_batch_wave_time(&[vec![port0, port1]], &times).unwrap(),
            expected
        );
    }

    #[test]
    fn compressed_union_matches_production_for_small_width_table() {
        let times = [GpuTimeModel { per_column_seconds: 1.0, ..Default::default() }];
        for columns in 1..=12 {
            for left_width in 1..=4 {
                for right_width in 1..=4 {
                    let left = GpuColumnSchedule::new(
                        columns,
                        vec![left_width],
                        vec![GpuColumnInterval { device: 0, start: 0, end: columns }],
                    )
                    .unwrap();
                    let right = GpuColumnSchedule::new(
                        columns,
                        vec![right_width],
                        vec![GpuColumnInterval { device: 0, start: 0, end: columns }],
                    )
                    .unwrap();
                    let by_port = vec![vec![left.clone()], vec![right.clone()]];
                    let expected = crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, 1)
                        .unwrap()
                        .map(|wave| {
                            wave.unwrap()
                                .into_iter()
                                .map(|job| {
                                    times[job.device].job_seconds(job.range.end - job.range.start)
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>();
                    let actual =
                        gpu_multi_output_batch_wave_time(&[vec![left, right]], &times).unwrap();
                    assert_eq!(
                        actual, expected,
                        "columns={columns}, widths={left_width},{right_width}"
                    );
                }
            }
        }
    }

    #[test]
    fn compressed_union_preserves_cross_port_owner_transition_costs() {
        let port0 = GpuColumnSchedule::new(
            8,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 4 },
                GpuColumnInterval { device: 1, start: 4, end: 8 },
            ],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            8,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 3 },
                GpuColumnInterval { device: 1, start: 3, end: 8 },
            ],
        )
        .unwrap();
        let times = [
            GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
            GpuTimeModel { per_column_seconds: 10.0, ..Default::default() },
        ];
        let by_port = vec![vec![port0.clone()], vec![port1.clone()]];
        let expected = crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, 1)
            .unwrap()
            .map(|wave| {
                let mut by_device = BTreeMap::<usize, f64>::new();
                for job in wave.unwrap() {
                    *by_device.entry(job.device).or_default() +=
                        times[job.device].job_seconds(job.range.end - job.range.start);
                }
                by_device.values().copied().fold(0.0, f64::max)
            })
            .sum::<f64>();
        assert_eq!(expected, 41.0);
        assert_eq!(
            gpu_multi_output_batch_wave_time(&[vec![port0, port1]], &times).unwrap(),
            expected
        );
    }

    #[test]
    fn compressed_union_handles_wide_port_crossing_narrow_owner_shift() {
        let wide = GpuColumnSchedule::new(
            100,
            vec![2, 2],
            vec![GpuColumnInterval { device: 0, start: 0, end: 100 }],
        )
        .unwrap();
        let shifted = GpuColumnSchedule::new(
            100,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 30 },
                GpuColumnInterval { device: 1, start: 30, end: 100 },
            ],
        )
        .unwrap();
        let times = [
            GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
            GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
        ];
        let by_port = vec![vec![wide.clone()], vec![shifted.clone()]];
        let expected = crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, 1)
            .unwrap()
            .map(|wave| {
                wave.unwrap()
                    .into_iter()
                    .map(|job| times[job.device].job_seconds(job.range.end - job.range.start))
                    .sum::<f64>()
            })
            .sum::<f64>();
        assert_eq!(expected, 100.0);
        assert_eq!(
            gpu_multi_output_batch_wave_time(&[vec![wide, shifted]], &times).unwrap(),
            expected
        );
    }

    #[test]
    fn differing_huge_port_owners_use_bounded_union_classes() {
        let columns = 1_000_000_000usize;
        let half = columns / 2;
        let port0 = GpuColumnSchedule::new(
            columns,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: half },
                GpuColumnInterval { device: 1, start: half, end: columns },
            ],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            columns,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 1, start: 0, end: half },
                GpuColumnInterval { device: 0, start: half, end: columns },
            ],
        )
        .unwrap();
        let time = GpuTimeModel {
            per_column_seconds: 1.0,
            per_job_seconds: 1.0,
            wave_overhead_seconds: 1.0,
            ..Default::default()
        };
        let predicted =
            gpu_multi_output_batch_wave_time(&[vec![port0, port1]], &[time, time]).unwrap();
        assert!(predicted.is_finite());
        assert!(predicted > columns as f64);
    }

    #[test]
    fn shared_staging_peak_is_charged_to_source_device() {
        let costs = vec![
            GpuStageCostModel {
                fixed: GpuResourceCost { transfers: 80, ..Default::default() },
                ..Default::default()
            },
            GpuStageCostModel::default(),
        ];
        let input = GpuWarmupInput {
            contract: contract(2, 100),
            layouts: vec![layout(2, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(2, costs, vec![1]) }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert!(result.report.stages[0].peak[0].transfers >= 80);
        assert_eq!(result.report.stages[0].peak[1].transfers, 0);
    }

    #[test]
    fn schedule_time_model_matches_dispatch() {
        let schedule = GpuColumnSchedule::new(
            9,
            vec![2, 2],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 5 },
                GpuColumnInterval { device: 1, start: 5, end: 9 },
            ],
        )
        .unwrap();
        let time =
            GpuTimeModel { fixed_seconds: 1.0, per_column_seconds: 1.0, ..Default::default() };
        let predicted = gpu_batch_wave_time(&[&schedule], &[time, time]).unwrap();
        // Dispatch waves are (2,2), (2,2), (1): max latencies 3, 3, 2.
        assert_eq!(predicted, 8.0);
    }

    fn matrix(columns: usize) -> ConcreteWireType {
        ConcreteWireType::Matrix(mxx_ir_core::types::ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 8,
            rows: 2,
            columns,
        })
    }

    #[test]
    fn warmup_and_runtime_share_range_lowering() {
        let kind = NodeKind::Concat { axis: ConcatAxis::Columns };
        let args = vec![matrix(3), matrix(5)];
        let output = ColumnRange { start: 2, end: 6 };
        let warmup = map_output_range_to_inputs(&kind, &args, output).unwrap();
        let production = map_output_range_to_inputs(&kind, &args, output).unwrap();
        assert_eq!(warmup, production);
    }

    #[test]
    fn derived_cache_is_not_semantic_state() {
        let costs = vec![GpuStageCostModel::default(); 2];
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![layout(8, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(8, costs, vec![1, 2, 4]) }],
        };
        let without_cache = plan_gpu_warmup(&input).unwrap().plan;
        let with_cache = plan_gpu_warmup(&input).unwrap().plan;
        assert_eq!(without_cache, with_cache);
    }
}
