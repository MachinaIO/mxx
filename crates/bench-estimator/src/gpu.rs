//! GPU measurements for individual validated IR nodes.
//!
//! This adapter constructs representative zero-valued inputs and invokes the same
//! backend operations as the runtime. It measures operation cost; it is not a
//! second graph executor and does not define node semantics.

#[cfg(test)]
#[path = "gpu_lifecycle_tests.rs"]
mod gpu_lifecycle_tests;
#[cfg(test)]
#[path = "gpu_multigpu_lifecycle_tests.rs"]
mod gpu_multigpu_lifecycle_tests;
#[cfg(test)]
#[path = "gpu_primitive_lifecycle_tests.rs"]
mod gpu_primitive_lifecycle_tests;

use crate::{
    MeasurementBackend, MeasurementNode, NodeMeasurement,
    harness::{MeasurementHarnessConfig, MemoryProbe, measure_batch_operation},
};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ArtifactAvailability, ArtifactType},
    encoding,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::{
    gpu_memory::{GpuMemoryRange, GpuMemoryShape},
    matrix::{PolyMatrix, PolyMatrixColumnSource, SmallPolyMatrix, gpu_dcrt_poly::GpuSmallMatrix},
    poly::dcrt::gpu::{
        GpuOutOfMemory, gpu_default_mempool_reset_high_water, gpu_default_mempool_usage,
        gpu_device_memory_usage, gpu_device_runtime_identity, gpu_memory_info,
    },
    sampler::trapdoor::gpu::{
        PreimageAllocationEvidence, PreimageCacheIdentity as NativePreimageCacheIdentity,
        PreimageCacheState,
    },
};
use mxx_runtime::{
    Backend,
    backend::{
        GpuWarmupCacheState, GpuWarmupDeviceIdentity, GpuWarmupEffectiveVariant,
        GpuWarmupMemoryObservations, GpuWarmupOperationDescriptor, GpuWarmupOperationSignature,
        GpuWarmupProfile, GpuWarmupProfileError, GpuWarmupProfileKey, GpuWarmupProfileProvider,
        GpuWarmupProfileRequest, GpuWarmupProvenance, GpuWarmupResidencyDelta, GpuWarmupRoute,
        GpuWarmupTimingScope, IndexRange, MatrixMulAccumulateRequest, MemoryEvidenceKind,
        PreimageCacheIdentity, PreimageRequest, RuntimeValue, SampleRange,
        poly::PolyBackendError,
        poly_gpu::{
            GpuAffectedResourceEnvelope, GpuAllocationComponents, GpuAllocationEvidenceKind,
            GpuDcrtBackend, GpuFleetMatrix, GpuFleetSmallMatrix, GpuFleetTrapdoor,
            GpuLocalProductionInput, GpuLocalProductionJobRequest, GpuLocalProductionSource,
            GpuProductionCompletion,
        },
    },
    gpu_calibration::{
        GpuCalibrationKey, GpuCalibrationProfile, GpuCalibrationRegistry, GpuColumnWidths,
        GpuDeviceCalibration, GpuDeviceMemory, gpu_calibration_environment,
        gpu_calibration_operation_identity, gpu_capped_waterfill_columns,
        gpu_matrix_multiply_scales_left,
    },
    gpu_column_policy::{
        CanonicalWarmupProfileDomain, ColumnCapability, ColumnRange, FusedWarmupOperation,
        GpuExecutionRange, GpuExecutionRouteDescriptor, GpuExecutionVariant,
        GpuFragmentClass as TypedFragmentClass, GpuTransferRoute, InputColumnRange,
        WarmupMeasurementKind, WarmupTransferKind, canonical_warmup_profile_domain,
        column_capability, effective_gpu_operation, fused_warmup_profile_domain,
        gpu_execution_range, map_output_range_to_inputs_with_output,
    },
    gpu_execution_plan::{FrozenGpuPlan, GpuExecutionSiteKey},
    gpu_warmup::{
        GpuPreimageFootprint, GpuProfileProvenance, GpuResourceCost, GpuTimeModel,
        gpu_non_column_batch_wave_time,
    },
    host_control::{
        HostControlBodyAction, HostControlChild, HostPrimitiveValue, dispatch_extract_coefficient,
        dispatch_pack_polynomial_coefficients, dispatch_polynomial_from_values,
        dispatch_polynomial_values, dispatch_threshold_decode, measure_host_container_primitive,
        measure_host_control_with, measure_host_primitive, measure_runtime_container_primitive,
        measure_trapdoor_public, measure_typed_runtime_input,
    },
};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    panic::{self, AssertUnwindSafe},
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
};
use tracing::{debug, info};

#[derive(Debug)]
pub struct GpuMeasurementError(String);

const GPU_OOM_ERROR_PREFIX: &str = "mxx-gpu-out-of-memory: ";

impl fmt::Display for GpuMeasurementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GpuMeasurementError {}

impl GpuMeasurementError {
    fn out_of_memory(message: impl Into<String>) -> Self {
        Self(format!("{GPU_OOM_ERROR_PREFIX}{}", message.into()))
    }

    fn is_out_of_memory(&self) -> bool {
        self.0.starts_with(GPU_OOM_ERROR_PREFIX)
    }
}

struct GpuMemoryProbe {
    device_id: i32,
}

#[derive(Clone, Copy)]
struct GpuMemoryMeasurementBaseline {
    pool_used_current: usize,
    context_generation: u64,
    owned_contexts: usize,
}

fn require_exclusive_measurement_context(
    device_id: i32,
    live_contexts: usize,
    owned_contexts: usize,
) -> Result<(), GpuMeasurementError> {
    if owned_contexts > 0 && live_contexts == owned_contexts {
        Ok(())
    } else {
        Err(GpuMeasurementError(format!(
            "GPU {device_id} has {live_contexts} live mxx contexts but the measurement backend owns {owned_contexts}; exclusive CUDA mempool measurement is required"
        )))
    }
}

/// Select the worker for a physical device.  In replicated multi-GPU setup
/// every worker backend may contain the entire fleet, so backend containment
/// alone is ambiguous.  The worker's explicit device owner is authoritative;
/// containment is retained only for legacy/shared-backend fallback.
fn select_worker_index_for_physical(
    worker_count: usize,
    physical: i32,
    worker_device_id: impl Fn(usize) -> i32,
    backend_contains: impl Fn(usize) -> bool,
) -> Option<usize> {
    (0..worker_count)
        .find(|&index| worker_device_id(index) == physical)
        .or_else(|| (0..worker_count).find(|&index| backend_contains(index)))
}

fn begin_gpu_memory_measurement(
    worker: &mut GpuMeasurementWorker,
) -> Result<GpuMemoryMeasurementBaseline, GpuMeasurementError> {
    let device_id = worker.device_id;
    let memory = gpu_device_memory_usage(device_id).map_err(GpuMeasurementError)?;
    let owned_contexts = worker.backend.owned_context_count(device_id);
    require_exclusive_measurement_context(device_id, memory.live_contexts, owned_contexts)?;
    // Matrix readiness precedes owner destruction, which queues frees on separate release
    // streams. Match the runtime calibration boundary before sampling the allocator baseline.
    // This fences release events only, outside the measured operation.
    worker
        .backend
        .fence_released_memory()
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
    gpu_default_mempool_reset_high_water(device_id).map_err(GpuMeasurementError)?;
    let pool_used_current =
        gpu_default_mempool_usage(device_id).map_err(GpuMeasurementError)?.used_current;
    Ok(GpuMemoryMeasurementBaseline {
        pool_used_current,
        context_generation: memory.context_generation,
        owned_contexts,
    })
}

fn finish_gpu_memory_measurement(
    device_id: i32,
    baseline: GpuMemoryMeasurementBaseline,
) -> Result<u64, GpuMeasurementError> {
    let memory = gpu_device_memory_usage(device_id).map_err(GpuMeasurementError)?;
    require_exclusive_measurement_context(
        device_id,
        memory.live_contexts,
        baseline.owned_contexts,
    )?;
    if memory.context_generation != baseline.context_generation {
        return Err(GpuMeasurementError(format!(
            "GPU {device_id} context generation changed during CUDA mempool measurement"
        )));
    }
    let high_water = gpu_default_mempool_usage(device_id).map_err(GpuMeasurementError)?.used_high;
    u64::try_from(high_water.checked_sub(baseline.pool_used_current).ok_or_else(|| {
        GpuMeasurementError("GPU mempool high-water is below its measurement baseline".into())
    })?)
    .map_err(|_| GpuMeasurementError("GPU workspace exceeds u64".to_owned()))
}

impl MemoryProbe for GpuMemoryProbe {
    type Error = GpuMeasurementError;

    fn current_bytes(&self) -> Result<u64, Self::Error> {
        let memory = gpu_default_mempool_usage(self.device_id).map_err(GpuMeasurementError)?;
        u64::try_from(memory.used_current)
            .map_err(|_| GpuMeasurementError("GPU memory usage exceeds u64".to_owned()))
    }
}

#[derive(Clone)]
struct PreparedMeasurement {
    arguments: Vec<Option<Arc<GpuFleetMatrix>>>,
    small_arguments: Vec<Option<Arc<GpuFleetSmallMatrix>>>,
    preimage_trapdoor: Option<(GpuFleetMatrix, GpuFleetTrapdoor, f64, BigInt, usize, BigInt)>,
    preimage_target: Option<Arc<dyn PolyMatrixColumnSource<GpuFleetMatrix>>>,
}

/// Host-only representatives are prepared once, outside the timed loop.
/// Runtime values retain their real ownership shape so family/select timing
/// exercises the same clone/materialization boundary as production.
struct PreparedHostPrimitive {
    typed_inputs:
        Option<(BTreeMap<String, RuntimeValue<GpuDcrtBackend>>, String, ConcreteWireType)>,
    trapdoor: Option<RuntimeValue<GpuDcrtBackend>>,
    scalar_inputs: Vec<HostPrimitiveValue>,
    family_inputs: Vec<RuntimeValue<GpuDcrtBackend>>,
    dynamic_index: Option<RuntimeValue<GpuDcrtBackend>>,
    choices: Vec<RuntimeValue<GpuDcrtBackend>>,
}

impl PreparedMeasurement {
    fn merge(mut self, other: Self) -> Self {
        for (target, value) in self.arguments.iter_mut().zip(other.arguments) {
            if target.is_none() {
                *target = value;
            }
        }
        for (target, value) in self.small_arguments.iter_mut().zip(other.small_arguments) {
            if target.is_none() {
                *target = value;
            }
        }
        if self.preimage_trapdoor.is_none() {
            self.preimage_trapdoor = other.preimage_trapdoor;
        }
        if self.preimage_target.is_none() {
            self.preimage_target = other.preimage_target;
        }
        self
    }

    fn merge_for_representative(
        self,
        scaled: Self,
        fixed_arguments: &[bool],
    ) -> Result<Self, GpuMeasurementError> {
        if self.arguments.len() == scaled.arguments.len() &&
            self.small_arguments.len() == scaled.small_arguments.len()
        {
            return Ok(self.merge(scaled));
        }
        let fixed_is_empty = self.arguments.iter().all(Option::is_none) &&
            self.small_arguments.iter().all(Option::is_none) &&
            self.preimage_trapdoor.is_none() &&
            self.preimage_target.is_none();
        if fixed_arguments.len() == scaled.arguments.len() &&
            fixed_arguments.iter().all(|fixed| !fixed) &&
            fixed_is_empty
        {
            return Ok(scaled);
        }
        Err(GpuMeasurementError(format!(
            "fixed/scaled representative arity mismatch (fixed matrices={}, fixed compact={}, scaled matrices={}, scaled compact={}, ownership={fixed_arguments:?})",
            self.arguments.len(),
            self.small_arguments.len(),
            scaled.arguments.len(),
            scaled.small_arguments.len()
        )))
    }

    fn finish(&self) {
        self.arguments.iter().flatten().for_each(|value| value.wait_until_ready());
        self.small_arguments.iter().flatten().for_each(|value| value.wait_until_ready());
        if let Some((public, trapdoor, ..)) = &self.preimage_trapdoor {
            public.wait_until_ready();
            trapdoor.wait_until_ready();
        }
    }
}

enum GpuMeasurementOutput {
    Matrix(GpuFleetMatrix),
    SmallMatrix(GpuFleetSmallMatrix),
    Trapdoor(GpuFleetTrapdoor),
}

impl GpuMeasurementOutput {
    fn matrix(value: GpuFleetMatrix) -> Self {
        Self::Matrix(value)
    }

    fn finish(&self) {
        match self {
            Self::Matrix(value) => {
                value.shards().iter().for_each(|shard| shard.value.wait_until_ready());
            }
            // Compact outputs must be fenced before the timed callback returns; retain the
            // compact owner and do not materialize it as a full DCRT matrix.
            Self::SmallMatrix(value) => {
                value.shards().iter().for_each(|shard| shard.value.wait_until_ready());
            }
            Self::Trapdoor(value) => value.wait_until_ready(),
        }
    }
}

struct GpuMeasurementOutputs(Vec<GpuMeasurementOutput>);

impl GpuProductionCompletion for GpuMeasurementOutputs {
    fn complete(&self) {
        self.0.iter().for_each(GpuMeasurementOutput::finish);
    }
}

struct GpuMeasurementWorker {
    backend: GpuDcrtBackend,
    device_id: i32,
    /// The last inclusive production job observation.  Keep the route and
    /// owner resource envelope at the provider boundary instead of reducing
    /// the job to a bare duration; warmup admission needs every physical
    /// owner (including host/pinned staging), not just the callback device.
    last_production_job: Option<ProductionJobObservation>,
}

#[derive(Clone, Debug)]
struct ProductionJobObservation {
    elapsed: std::time::Duration,
    routes: Vec<mxx_runtime::gpu_column_policy::GpuExecutionRouteDescriptor>,
    resources: Vec<GpuAffectedResourceEnvelope>,
}

/// Collapse the per-input route response from the inclusive production job
/// into the route identity used by one warmup point.  The requested resolver
/// is only a candidate; this function is intentionally based on the routes
/// returned by the fleet after it has inspected actual shard residency.
fn authoritative_production_route(
    requested: GpuExecutionRouteDescriptor,
    observed: &[GpuExecutionRouteDescriptor],
) -> Option<GpuExecutionRouteDescriptor> {
    let first = *observed.first()?;
    if observed.len() == 1 {
        return first.validate().then_some(first);
    }
    let destination = observed
        .first()
        .and_then(|route| route.destination_device)
        .filter(|destination| {
            observed.iter().all(|route| route.destination_device == Some(*destination))
        })
        .or(requested.destination_device);
    let route = if observed.iter().any(|route| route.route == GpuTransferRoute::HostStaging) {
        GpuTransferRoute::HostStaging
    } else if observed.iter().any(|route| route.route == GpuTransferRoute::Peer) {
        GpuTransferRoute::Peer
    } else {
        GpuTransferRoute::Resident
    };
    let source_device = observed
        .iter()
        .find_map(|route| route.source_device.filter(|source| Some(*source) != destination))
        .or_else(|| observed.iter().find_map(|route| route.source_device))
        .or(requested.source_device);
    let source_range = observed.iter().fold(first.source_range, |range, route| ColumnRange {
        start: range.start.min(route.source_range.start),
        end: range.end.max(route.source_range.end),
    });
    let destination_range =
        observed.iter().fold(first.destination_range, |range, route| ColumnRange {
            start: range.start.min(route.destination_range.start),
            end: range.end.max(route.destination_range.end),
        });
    let mut aggregate = first;
    aggregate.route = route;
    aggregate.source_device = source_device;
    aggregate.destination_device = destination;
    aggregate.source_range = source_range;
    aggregate.destination_range = destination_range;
    aggregate.source_compact = observed.iter().any(|route| route.source_compact);
    aggregate.destination_compact = observed.iter().any(|route| route.destination_compact);
    aggregate.source_staging_bytes = observed.iter().map(|route| route.source_staging_bytes).sum();
    aggregate.host_staging_bytes = observed.iter().map(|route| route.host_staging_bytes).sum();
    aggregate.pinned_host_staging_bytes =
        observed.iter().map(|route| route.pinned_host_staging_bytes).sum();
    let source_routes = observed.iter().flat_map(|route| route.source_routes().iter().copied());
    aggregate = aggregate.with_source_routes(source_routes)?;
    aggregate.validate().then_some(aggregate)
}

#[derive(Clone)]
struct PendingMeasurement {
    warmup: Option<GpuWarmupOperationDescriptor>,
    key: [u8; 32],
    scope: mxx_ir_core::FrozenGraphScopeId,
    id: mxx_ir_core::types::NodeId,
    kind: NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
    bindings: ParamEnv,
    preimage_sample: bool,
}

struct RepresentativeMeasurement {
    inputs: mxx_runtime::backend::GpuEffectiveInputs,
    source_layouts: Vec<mxx_runtime::backend::GpuWarmupStorageLayout>,
    fixed_metadata: Option<mxx_runtime::backend::PlannedNodeBatchRequest>,
    retry_cap: Option<usize>,
    kind: NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
    fixed_arguments: Vec<bool>,
    output_range: Option<IndexRange>,
}

fn family_leaf_type(wire_type: &ConcreteWireType) -> &ConcreteWireType {
    match wire_type {
        ConcreteWireType::IndexedFamily { element, .. } => family_leaf_type(element),
        _ => wire_type,
    }
}

fn family_leaf_type_mut(wire_type: &mut ConcreteWireType) -> &mut ConcreteWireType {
    match wire_type {
        ConcreteWireType::IndexedFamily { element, .. } => family_leaf_type_mut(element),
        _ => wire_type,
    }
}

fn matrix_leaf_type(wire_type: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
    family_leaf_type(wire_type).matrix_type()
}

fn matrix_leaf_type_mut(wire_type: &mut ConcreteWireType) -> Option<&mut ConcreteMatrixType> {
    match family_leaf_type_mut(wire_type) {
        ConcreteWireType::Matrix(matrix) |
        ConcreteWireType::SmallMatrix { matrix, .. } |
        ConcreteWireType::Preimage { matrix, .. } |
        ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix),
        _ => None,
    }
}

impl RepresentativeMeasurement {
    fn measured_columns(&self) -> Option<usize> {
        self.output_range
            .as_ref()
            .map(|range| range.end - range.start)
            .or_else(|| {
                self.concrete_output_types
                    .iter()
                    .find_map(GpuNodeMeasurementBackend::matrix_columns)
            })
            .or_else(|| {
                self.concrete_argument_types
                    .iter()
                    .find_map(GpuNodeMeasurementBackend::matrix_columns)
            })
            .filter(|columns| *columns > 0)
    }
}

fn extrapolate_fleet_waves(full_wave: &NodeMeasurement, wave_count: usize) -> NodeMeasurement {
    NodeMeasurement {
        work_seconds: full_wave.work_seconds * wave_count as f64,
        latency_seconds: full_wave.latency_seconds,
        cumulative_wave_seconds: full_wave.latency_seconds * wave_count as f64,
        independent_wave_count: wave_count,
        measured_wave_workspace_bytes: full_wave.measured_wave_workspace_bytes,
        // Workspace is the peak of one bounded wave.  Independent waves are
        // sequential in the production schedule; multiplying a one-wave
        // peak by their count is an empirical resource extrapolation and
        // over-admits no real concurrent allocation.
        workspace_bytes: full_wave.workspace_bytes,
    }
}

fn aggregate_fleet_wave(
    measurements: impl IntoIterator<Item = NodeMeasurement>,
    fleet_latency_seconds: f64,
) -> NodeMeasurement {
    measurements.into_iter().fold(NodeMeasurement::default(), |mut fleet, device| {
        // Work and workspace are aggregate fleet resources. Fleet latency is measured around the
        // coordinated enqueue/completion join and is installed after aggregating device work.
        fleet.work_seconds += device.work_seconds;
        fleet.latency_seconds = fleet_latency_seconds;
        fleet.cumulative_wave_seconds = fleet_latency_seconds;
        fleet.measured_wave_workspace_bytes = fleet
            .measured_wave_workspace_bytes
            .saturating_add(device.measured_wave_workspace_bytes);
        fleet.workspace_bytes = fleet.workspace_bytes.saturating_add(device.workspace_bytes);
        fleet
    })
}

impl PendingMeasurement {
    fn representative_bytes(&self) -> u128 {
        fn wire_bytes(wire_type: &ConcreteWireType) -> u128 {
            match wire_type {
                ConcreteWireType::Matrix(matrix) => (matrix.rows as u128)
                    .saturating_mul(matrix.columns as u128)
                    .saturating_mul(matrix.ring_dimension as u128)
                    .saturating_mul(8),
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
                ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                    compact_matrix_bytes_u128(matrix, max_coefficient_bound)
                }
                ConcreteWireType::Trapdoor { matrix, .. } => (matrix.rows as u128)
                    .saturating_mul(matrix.columns as u128)
                    .saturating_mul(matrix.ring_dimension as u128)
                    .saturating_mul(8),
                ConcreteWireType::IndexedFamily { element, count } => {
                    wire_bytes(element).saturating_mul(*count as u128)
                }
                ConcreteWireType::Bytes { length } => *length as u128,
                ConcreteWireType::TypedBlob { .. } |
                ConcreteWireType::ConstantInt |
                ConcreteWireType::ConstantReal |
                ConcreteWireType::ConstantBool |
                ConcreteWireType::Int |
                ConcreteWireType::Real |
                ConcreteWireType::Bool => 0,
            }
        }

        self.concrete_argument_types
            .iter()
            .chain(&self.concrete_output_types)
            .map(wire_bytes)
            .max()
            .unwrap_or(0)
    }
}

pub struct GpuNodeMeasurementBackend {
    workers: Vec<GpuMeasurementWorker>,
    harness: MeasurementHarnessConfig,
    /// Setup-time snapshot. Measurement and calibration must not reread process environment.
    vram_percent: u32,
    calibration_registry: GpuCalibrationRegistry,
    measurements: HashMap<[u8; 32], NodeMeasurement>,
    /// Preimage retry metadata is not part of the generic profile point.  It
    /// is retained only as validated sampler metadata, keyed by the same
    /// canonical execution class as the point table.
    preimage_profile_metadata: HashMap<(GpuWarmupProfileKey, usize), (usize, usize)>,
    /// Width-specific native allocation evidence retained only during setup.
    /// The value is copied into the canonical profile point; fixed execution
    /// never receives this primitive-owned object.
    preimage_footprints: HashMap<(GpuWarmupProfileKey, usize), GpuPreimageFootprint>,
    preimage_memory: HashMap<(GpuWarmupProfileKey, usize), GpuWarmupMemoryObservations>,
    /// Trapdoors used by preimage warmup are session-owned.  A cold setup
    /// request populates this map and the following warm local-job request
    /// reuses the same native cache owner instead of creating a new trapdoor
    /// during preparation.
    preimage_session_inputs: HashMap<(GpuWarmupOperationSignature, usize), PreparedMeasurement>,
    /// Native cache owner observed by each warmup session/device.  Evidence
    /// from a warm request is accepted only when it came from the same owner
    /// as the cold setup session; a newly constructed sampler must not pass
    /// merely because its shape and retry settings match.
    preimage_evidence_identities:
        HashMap<(GpuWarmupOperationSignature, usize), NativePreimageCacheIdentity>,
    /// Setup-only operation descriptors retained so a profile miss can run
    /// the actual device-local production range instead of manufacturing a
    /// zero/default cost. Descriptors are discarded by fixed execution code.
    warmup_operations: HashMap<GpuWarmupOperationSignature, PendingMeasurement>,
    /// The descriptor's closed profile contract is kept alongside the
    /// representative.  It must not be reconstructed from `NodeKind`: fused
    /// lowering deliberately keeps the ordinary IR kind while selecting a
    /// different production kernel and workspace contract.
    warmup_profile_domains: HashMap<GpuWarmupOperationSignature, CanonicalWarmupProfileDomain>,
    warmup_fused_operations: HashMap<GpuWarmupOperationSignature, Option<FusedWarmupOperation>>,
    warmup_implementation_variants: HashMap<GpuWarmupOperationSignature, String>,
    /// Number of setup-provider measurement calls that reached the production
    /// measurement path.  This is intentionally separate from the legacy
    /// fleet calibration registry: the canonical warmup session owns profile
    /// points and may legitimately leave that registry empty.
    warmup_measurement_calls: Arc<AtomicUsize>,
    warmup_measurement_provenances: Vec<GpuWarmupProvenance>,
    /// Production dispatch evidence retained for setup diagnostics and tests.
    /// This is populated only after the inclusive production job has returned;
    /// it is never used as a substitute for the measured elapsed time.
    warmup_dispatch_records: Vec<GpuWarmupDispatchRecord>,
    /// Validated child contracts for host/control nodes, keyed by their
    /// concrete scope site so collected estimator measurements can reuse the
    /// same descriptor as warmup profile requests.
    host_control_operations:
        HashMap<(mxx_ir_core::FrozenGraphScopeId, mxx_ir_core::types::NodeId), HostControlChild>,
    pending: HashMap<[u8; 32], PendingMeasurement>,
    collecting: bool,
}

/// One successful warmup dispatch through the public production range API.
/// Keeping this small, value-only record makes it possible to assert that a
/// profile was produced by the selected ordinary/fused/host/transfer route,
/// rather than by a provider that merely fabricated a profile value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuWarmupDispatchRecord {
    pub profile_domain: CanonicalWarmupProfileDomain,
    pub fused_operation: Option<FusedWarmupOperation>,
    pub range: IndexRange,
    pub timing_scope: GpuWarmupTimingScope,
    pub measurement: WarmupMeasurementKind,
    pub inputs: mxx_runtime::backend::GpuEffectiveInputs,
    pub argument_types: Vec<ConcreteWireType>,
    pub output_types: Vec<ConcreteWireType>,
}

impl GpuNodeMeasurementBackend {
    fn physical_device_ids(&self) -> Vec<i32> {
        self.workers.iter().flat_map(|worker| worker.backend.physical_device_ids()).collect()
    }

    fn physical_device_index(&self, physical: i32) -> Option<usize> {
        self.physical_device_ids().into_iter().position(|device| device == physical)
    }

    fn worker_index_for_physical_device(&self, physical: i32) -> Option<usize> {
        select_worker_index_for_physical(
            self.workers.len(),
            physical,
            |index| self.workers[index].device_id,
            |index| self.workers[index].backend.physical_device_ids().contains(&physical),
        )
    }

    fn worker_index_for_logical_device(&self, logical: usize) -> Option<usize> {
        let physical = self.physical_device_ids().get(logical).copied()?;
        self.worker_index_for_physical_device(physical)
    }

    /// Convert the native sampler owner returned by the cold setup query into
    /// the opaque runtime identity.  Only the digest leaves this provider;
    /// trapdoor ownership and numeric sampler parameters never enter a
    /// profile or planner key in raw form.
    fn opaque_preimage_identity(evidence: &PreimageAllocationEvidence) -> [u8; 32] {
        let context = &evidence.context;
        let sampler_parameters = [
            context.format.crt_level as u64,
            context.format.coefficient_magnitude_bytes as u64,
            u64::from(context.format.compact_output),
        ];
        PreimageCacheIdentity::from_native(
            &context.cache_identity,
            &sampler_parameters,
            &context.format.active_moduli,
        )
        .digest()
    }

    fn is_host_backend_boundary(kind: &NodeKind) -> bool {
        matches!(
            kind,
            NodeKind::ExtractCoefficient { .. } |
                NodeKind::ThresholdDecode { .. } |
                NodeKind::PackPolynomialCoefficients { .. } |
                NodeKind::PolynomialFromValues { .. } |
                NodeKind::PolynomialValues { .. }
        )
    }

    fn boundary_route(
        worker: &GpuMeasurementWorker,
        representative: &RepresentativeMeasurement,
    ) -> Result<GpuExecutionRouteDescriptor, GpuMeasurementError> {
        let ty = representative
            .concrete_output_types
            .iter()
            .chain(&representative.concrete_argument_types)
            .find_map(|ty| ty.matrix_type())
            .ok_or_else(|| GpuMeasurementError("boundary has no matrix type".into()))?;
        // Native polynomial upload/download moves one u64 residue for each
        // coefficient of each active CRT limb. This is a transfer payload,
        // not an estimate of the complete native allocation.
        let limbs = worker
            .backend
            .ring_crt_depth(ty)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let bytes = ty
            .ring_dimension
            .checked_mul(limbs)
            .and_then(|count| count.checked_mul(size_of::<u64>()))
            .ok_or_else(|| GpuMeasurementError("boundary transfer size overflows".into()))?;
        Ok(mxx_runtime::gpu_column_policy::resolve_gpu_route(
            mxx_runtime::gpu_column_policy::GpuRouteResolutionInput {
                source_device: Some(0),
                destination_device: Some(0),
                source_range: ColumnRange { start: 0, end: 1 },
                destination_range: ColumnRange { start: 0, end: 1 },
                source_is_resident: false,
                peer_available: false,
                source_compact: false,
                destination_compact: false,
                fragment: TypedFragmentClass::Full,
                source_staging_bytes: bytes,
                host_staging_bytes: bytes,
                pinned_host_staging_bytes: 0,
            },
        ))
    }

    /// Creates a representative GPU measurement backend for validated IR nodes.
    pub fn new(backends: Vec<(GpuDcrtBackend, i32)>, harness: MeasurementHarnessConfig) -> Self {
        assert!(!backends.is_empty(), "GPU measurement requires at least one backend");
        let vram_percent = backends[0].0.vram_percent();
        assert!(
            backends.iter().all(|(backend, _)| backend.vram_percent() == vram_percent),
            "all GPU measurement contexts must use the same VRAM percentage"
        );
        let mut owners = backends
            .iter()
            .flat_map(|(backend, _)| backend.physical_device_ids())
            .collect::<Vec<_>>();
        owners.dedup();
        let workers = backends
            .into_iter()
            .map(|(mut backend, device_id)| {
                backend
                    .set_measurement_owners(owners.clone())
                    .expect("validated measurement owners");
                GpuMeasurementWorker { backend, device_id, last_production_job: None }
            })
            .collect();
        Self::from_workers(workers, harness, vram_percent)
    }

    fn from_workers(
        workers: Vec<GpuMeasurementWorker>,
        harness: MeasurementHarnessConfig,
        vram_percent: u32,
    ) -> Self {
        Self {
            workers,
            harness,
            vram_percent,
            calibration_registry: GpuCalibrationRegistry::new(),
            measurements: HashMap::new(),
            preimage_profile_metadata: HashMap::new(),
            preimage_footprints: HashMap::new(),
            preimage_memory: HashMap::new(),
            preimage_session_inputs: HashMap::new(),
            preimage_evidence_identities: HashMap::new(),
            warmup_operations: HashMap::new(),
            warmup_profile_domains: HashMap::new(),
            warmup_fused_operations: HashMap::new(),
            warmup_implementation_variants: HashMap::new(),
            warmup_measurement_calls: Arc::new(AtomicUsize::new(0)),
            warmup_measurement_provenances: Vec::new(),
            warmup_dispatch_records: Vec::new(),
            host_control_operations: HashMap::new(),
            pending: HashMap::new(),
            collecting: true,
        }
    }

    /// Returns the setup-time profiles populated by measurement. Runtime may reuse the
    /// same registry and recompute widths from its then-current GPU residency.
    pub fn calibration_registry(&self) -> GpuCalibrationRegistry {
        self.calibration_registry.clone()
    }

    /// Number of actual setup-time profile measurements performed by this
    /// provider.  Runtime session-cache hits do not call the provider and
    /// therefore do not increase this count.
    pub fn warmup_measurement_call_count(&self) -> usize {
        self.warmup_measurement_calls.load(Ordering::SeqCst)
    }

    /// Share the immutable call-count observation with a test or setup
    /// harness that must drop this provider before fixed execution starts.
    pub fn warmup_measurement_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.warmup_measurement_calls)
    }

    /// Provenance of successfully returned setup-time profiles, in call
    /// order.  The runtime session table remains authoritative; this view is
    /// only diagnostic/test evidence that a provider call really produced a
    /// production-equivalent point.
    pub fn warmup_measurement_provenances(&self) -> &[GpuWarmupProvenance] {
        &self.warmup_measurement_provenances
    }

    /// Successful public production dispatches observed during setup.
    pub fn warmup_dispatch_records(&self) -> &[GpuWarmupDispatchRecord] {
        &self.warmup_dispatch_records
    }

    fn record_warmup_dispatch(
        &mut self,
        request: &GpuWarmupProfileRequest,
        domain: CanonicalWarmupProfileDomain,
        fused_operation: Option<FusedWarmupOperation>,
    ) {
        self.warmup_dispatch_records.push(GpuWarmupDispatchRecord {
            profile_domain: domain,
            fused_operation,
            range: request.range.clone(),
            timing_scope: request.timing_scope,
            measurement: domain.measurement_kind(),
            inputs: self
                .warmup_operations
                .get(&request.signature)
                .and_then(|pending| pending.warmup.as_ref())
                .map(|descriptor| descriptor.inputs.clone())
                .unwrap_or_default(),
            argument_types: self
                .warmup_operations
                .get(&request.signature)
                .map(|pending| pending.concrete_argument_types.clone())
                .unwrap_or_default(),
            output_types: self
                .warmup_operations
                .get(&request.signature)
                .map(|pending| pending.concrete_output_types.clone())
                .unwrap_or_default(),
        });
    }

    fn profile_shape(types: &[ConcreteWireType]) -> Vec<usize> {
        let mut shape = Vec::new();
        for wire_type in types {
            let Some(matrix) = matrix_leaf_type(wire_type) else { continue };
            // Columns are the interpolation coordinate.  Keep every other
            // physical shape property in the execution class so points from
            // another matrix regime cannot be reused accidentally.
            shape.extend([matrix.rows, matrix.ring_dimension]);
        }
        shape
    }

    fn profile_context_words(
        descriptor: &GpuWarmupOperationDescriptor,
    ) -> Result<Vec<u64>, GpuWarmupProfileError> {
        // Columns are the interpolation coordinate.  Do not serialize them
        // into the execution-class digest: doing so would create one profile
        // table per width and make otherwise valid measured points unable to
        // share an interpolation table.  The remaining matrix shape is kept
        // by `profile_shape` below.
        #[derive(Serialize)]
        struct ProfileMatrix {
            modulus: BigInt,
            ring_dimension: usize,
            rows: usize,
        }
        #[derive(Serialize)]
        enum ProfileType {
            ConstantInt,
            ConstantReal,
            ConstantBool,
            Int,
            Real,
            Bool,
            Bytes {
                length: usize,
            },
            TypedBlob {
                type_name: String,
                schema_hash: [u8; 32],
            },
            Matrix(ProfileMatrix),
            Trapdoor {
                matrix: ProfileMatrix,
                sigma: mxx_ir_core::expr::RealExpr,
                gadget_base: BigInt,
                digit_count: usize,
                preimage_max_coefficient_bound: BigInt,
            },
            SmallMatrix {
                matrix: ProfileMatrix,
                max_coefficient_bound: BigInt,
            },
            Preimage {
                matrix: ProfileMatrix,
                max_coefficient_bound: BigInt,
            },
            IndexedFamily {
                element: Box<ProfileType>,
                count: usize,
            },
        }
        fn profile_matrix(matrix: &ConcreteMatrixType) -> ProfileMatrix {
            ProfileMatrix {
                modulus: matrix.modulus.clone(),
                ring_dimension: matrix.ring_dimension,
                rows: matrix.rows,
            }
        }
        fn profile_type(wire: &ConcreteWireType) -> ProfileType {
            match wire {
                ConcreteWireType::ConstantInt => ProfileType::ConstantInt,
                ConcreteWireType::ConstantReal => ProfileType::ConstantReal,
                ConcreteWireType::ConstantBool => ProfileType::ConstantBool,
                ConcreteWireType::Int => ProfileType::Int,
                ConcreteWireType::Real => ProfileType::Real,
                ConcreteWireType::Bool => ProfileType::Bool,
                ConcreteWireType::Bytes { length } => ProfileType::Bytes { length: *length },
                ConcreteWireType::TypedBlob { type_name, schema_hash } => ProfileType::TypedBlob {
                    type_name: type_name.clone(),
                    schema_hash: *schema_hash,
                },
                ConcreteWireType::Matrix(matrix) => ProfileType::Matrix(profile_matrix(matrix)),
                ConcreteWireType::Trapdoor {
                    matrix,
                    sigma,
                    gadget_base,
                    digit_count,
                    preimage_max_coefficient_bound,
                } => ProfileType::Trapdoor {
                    matrix: profile_matrix(matrix),
                    sigma: sigma.clone(),
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                    preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
                },
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } => {
                    ProfileType::SmallMatrix {
                        matrix: profile_matrix(matrix),
                        max_coefficient_bound: max_coefficient_bound.clone(),
                    }
                }
                ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                    ProfileType::Preimage {
                        matrix: profile_matrix(matrix),
                        max_coefficient_bound: max_coefficient_bound.clone(),
                    }
                }
                ConcreteWireType::IndexedFamily { element, count } => ProfileType::IndexedFamily {
                    element: Box::new(profile_type(element)),
                    count: *count,
                },
            }
        }
        #[derive(Serialize)]
        struct ProfileContext<'a> {
            effective_operation: &'a str,
            row_groups: &'a [Vec<usize>],
            arguments: Vec<ProfileType>,
            outputs: Vec<ProfileType>,
            bindings: &'a ParamEnv,
        }
        let hash = encoding::hash_canonical(&ProfileContext {
            effective_operation: &descriptor.effective_operation,
            row_groups: &descriptor.inputs.row_groups,
            arguments: descriptor.concrete_argument_types.iter().map(profile_type).collect(),
            outputs: descriptor.concrete_output_types.iter().map(profile_type).collect(),
            bindings: &descriptor.bindings,
        })
        .map_err(|error| GpuWarmupProfileError::InvalidMeasurement(error.to_string()))?;
        Ok(hash
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("hash chunks are 8 bytes")))
            .collect())
    }

    fn profile_key(
        &self,
        request: &GpuWarmupProfileRequest,
        descriptor: &GpuWarmupOperationDescriptor,
        domain: CanonicalWarmupProfileDomain,
        device: usize,
    ) -> Result<GpuWarmupProfileKey, GpuWarmupProfileError> {
        let expected_route = if domain.measurement_kind() == WarmupMeasurementKind::HostMeasured &&
            request.timing_scope != GpuWarmupTimingScope::Transfer
        {
            GpuWarmupRoute::HostOnly
        } else {
            match request.route_descriptor.route {
                mxx_runtime::gpu_column_policy::GpuTransferRoute::Resident => {
                    GpuWarmupRoute::DeviceLocal
                }
                mxx_runtime::gpu_column_policy::GpuTransferRoute::Peer => {
                    GpuWarmupRoute::PeerToPeer
                }
                mxx_runtime::gpu_column_policy::GpuTransferRoute::HostStaging => {
                    GpuWarmupRoute::HostStaging
                }
            }
        };
        if request.route != expected_route &&
            !(domain.measurement_kind() == WarmupMeasurementKind::HostMeasured &&
                request.timing_scope != GpuWarmupTimingScope::Transfer)
        {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "warmup request route disagrees with physical route descriptor".into(),
            ));
        }
        if (domain.measurement_kind() == WarmupMeasurementKind::GpuMeasured ||
            request.timing_scope == GpuWarmupTimingScope::Transfer) &&
            !request.route_descriptor.validate() &&
            request.route_resolver.is_none()
        {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "warmup request has an invalid physical route descriptor".into(),
            ));
        }
        let physical = self.physical_device_ids().get(device).copied().ok_or_else(|| {
            GpuWarmupProfileError::Measurement("warmup device is unavailable".into())
        })?;
        let identity =
            gpu_device_runtime_identity(physical).map_err(GpuWarmupProfileError::Measurement)?;
        let implementation_variant = if request.timing_scope == GpuWarmupTimingScope::Transfer {
            GpuWarmupEffectiveVariant::Transfer
        } else {
            descriptor.fused_operation.map_or_else(
                || GpuWarmupEffectiveVariant::Custom(descriptor.implementation_variant.clone()),
                GpuWarmupEffectiveVariant::Fused,
            )
        };
        // The validated request carries the route selected by production
        // lowering.  Re-deriving it from the IR kind loses peer/host staging
        // information for mapped and fragmented jobs and would let two
        // physically different dispatches share one profile point.
        let route = if domain.measurement_kind() == WarmupMeasurementKind::HostMeasured &&
            request.timing_scope != GpuWarmupTimingScope::Transfer
        {
            GpuWarmupRoute::HostOnly
        } else {
            request.route
        };
        let width = request.range.end.checked_sub(request.range.start).ok_or_else(|| {
            GpuWarmupProfileError::InvalidMeasurement("warmup range is reversed".into())
        })?;
        if width == 0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "warmup profile coordinate must be non-empty".into(),
            ));
        }
        let fragment = request.fragment;
        let mut native_parameters =
            vec![request.signature.shape_class, request.signature.instance_class];
        native_parameters.extend(Self::profile_context_words(descriptor)?);
        Ok(GpuWarmupProfileKey {
            effective_domain: domain,
            implementation_variant,
            operation_identity: request.signature.operation,
            noninterpolated_shape: Self::profile_shape(
                &descriptor
                    .concrete_argument_types
                    .iter()
                    .chain(&descriptor.concrete_output_types)
                    .cloned()
                    .collect::<Vec<_>>(),
            ),
            native_parameters,
            device: GpuWarmupDeviceIdentity::new(
                device,
                format!(
                    "uuid={}-{}-sm{}.{}-mem{}",
                    identity.uuid,
                    identity.name,
                    identity.compute_major,
                    identity.compute_minor,
                    identity.total_global_memory
                ),
                format!(
                    "{}:driver{}:runtime{}",
                    identity.native_kernel_revision,
                    identity.driver_version,
                    identity.runtime_version,
                ),
            )
            .with_context_generation(identity.context_generation),
            executed_range_start: request.executed_range_start,
            executed_range_class: request.executed_range_class,
            retry_cap: request.retry_cap,
            cache_identity: request.cache_identity,
            // The request carries the lifecycle state. Cold setup and warm
            // sampling must never share a point table, even when their shape
            // and width are identical.
            cache_state: request.cache_state,
            route,
            route_descriptor: request.route_descriptor,
            binding_port: request.binding_port,
            fragment,
            timing_scope: request.timing_scope,
        })
    }

    fn preimage_footprint_from_evidence(
        evidence: &PreimageAllocationEvidence,
    ) -> Result<GpuPreimageFootprint, GpuMeasurementError> {
        let envelope = &evidence.envelope;
        let add = |values: &[usize]| {
            values.iter().try_fold(0usize, |total, value| total.checked_add(*value))
        };
        let persistent = add(&[
            envelope.public_matrix_bytes,
            envelope.trapdoor_bytes,
            envelope.resident_target_bytes,
        ])
        .ok_or_else(|| GpuMeasurementError("preimage persistent bytes overflow".into()))?;
        let scratch = add(&[
            envelope.candidate_workspace_bytes,
            envelope.perturbation_workspace_bytes,
            envelope.scratch_bytes,
        ])
        .ok_or_else(|| GpuMeasurementError("preimage scratch bytes overflow".into()))?;
        let transfer =
            add(&[envelope.source_device_staging_bytes, envelope.destination_device_staging_bytes])
                .ok_or_else(|| GpuMeasurementError("preimage staging bytes overflow".into()))?;
        let control = add(&[envelope.device_control_bytes, envelope.sampler_event_bytes])
            .ok_or_else(|| GpuMeasurementError("preimage control bytes overflow".into()))?;
        let mut persistent_cost = GpuResourceCost::zero();
        persistent_cost.live = u64::try_from(persistent)
            .map_err(|_| GpuMeasurementError("preimage persistent bytes exceed u64".into()))?;
        let mut compact_cost = GpuResourceCost::zero();
        compact_cost.outputs = u64::try_from(envelope.compact_output_bytes)
            .map_err(|_| GpuMeasurementError("preimage compact bytes exceed u64".into()))?;
        let mut scratch_cost = GpuResourceCost::zero();
        scratch_cost.scratch = u64::try_from(scratch)
            .map_err(|_| GpuMeasurementError("preimage scratch bytes exceed u64".into()))?;
        let transfer_bytes = u64::try_from(transfer)
            .map_err(|_| GpuMeasurementError("preimage transfer bytes exceed u64".into()))?;
        scratch_cost.transfers = transfer_bytes;
        let mut control_cost = GpuResourceCost::zero();
        control_cost.live = u64::try_from(control)
            .map_err(|_| GpuMeasurementError("preimage control bytes exceed u64".into()))?;
        control_cost.pinned_host = u64::try_from(envelope.pinned_host_control_bytes)
            .map_err(|_| GpuMeasurementError("preimage pinned control exceeds u64".into()))?;
        control_cost.host = u64::try_from(envelope.host_staging_bytes)
            .map_err(|_| GpuMeasurementError("preimage host staging exceeds u64".into()))?;
        let mut cache_cost = GpuResourceCost::zero();
        cache_cost.caches = u64::try_from(envelope.retained_covariance_cache_bytes)
            .map_err(|_| GpuMeasurementError("preimage cache bytes exceed u64".into()))?;
        let mut cold_cost = GpuResourceCost::zero();
        cold_cost.scratch = u64::try_from(envelope.cold_transient_workspace_bytes)
            .map_err(|_| GpuMeasurementError("preimage cold workspace exceeds u64".into()))?;
        let width = evidence.context.tile_columns;
        if width == 0 {
            return Err(GpuMeasurementError("preimage evidence has zero tile width".into()));
        }
        Ok(GpuPreimageFootprint {
            persistent: persistent_cost,
            compact: compact_cost,
            scratch: scratch_cost,
            control: control_cost,
            cold_cache: cache_cost,
            cold_transient_workspace: cold_cost,
            certified_tile_width: Some(width),
            per_tile: GpuResourceCost::zero(),
            width_sizing_model: None,
            width_costs: BTreeMap::new(),
            provenance: GpuProfileProvenance::MeasuredPoint,
        })
    }

    /// Record native preimage allocation evidence for a fixed profile point.
    /// Callers must provide the same retry bound used by fixed dispatch; the
    /// evidence context itself rejects reuse across another bound/cache state.
    pub fn register_warmup_preimage_evidence(
        &mut self,
        request: GpuWarmupProfileRequest,
        evidence: PreimageAllocationEvidence,
        max_attempts: usize,
    ) -> Result<(), GpuMeasurementError> {
        if max_attempts == 0 || evidence.context.tile_columns == 0 {
            return Err(GpuMeasurementError(
                "preimage evidence has invalid fixed configuration".into(),
            ));
        }
        let expected_state = match request.cache_state {
            GpuWarmupCacheState::Cold => PreimageCacheState::Cold,
            GpuWarmupCacheState::Warm => PreimageCacheState::Warm,
        };
        if evidence.context.state != expected_state {
            return Err(GpuMeasurementError(
                "preimage evidence cache state does not match the requested lifecycle".into(),
            ));
        }
        let requested_width = request
            .range
            .end
            .checked_sub(request.range.start)
            .ok_or_else(|| GpuMeasurementError("preimage evidence range is reversed".into()))?;
        if evidence.context.tile_columns != requested_width ||
            evidence.context.max_attempts != max_attempts
        {
            return Err(GpuMeasurementError(
                "preimage evidence context does not match the requested fixed call".into(),
            ));
        }
        let descriptor = self.warmup_operations.get(&request.signature).ok_or_else(|| {
            GpuMeasurementError(
                "cannot register preimage evidence before operation registration".into(),
            )
        })?;
        let domain =
            self.warmup_profile_domains.get(&request.signature).copied().ok_or_else(|| {
                GpuMeasurementError("preimage operation has no canonical profile domain".into())
            })?;
        let key = self
            .profile_key_for_pending(&request, descriptor, domain)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let footprint = Self::preimage_footprint_from_evidence(&evidence)?;
        let device_bytes = evidence
            .envelope
            .public_matrix_bytes
            .checked_add(evidence.envelope.trapdoor_bytes)
            .and_then(|bytes| bytes.checked_add(evidence.envelope.resident_target_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.retained_covariance_cache_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.compact_output_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.candidate_workspace_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.perturbation_workspace_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.scratch_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.source_device_staging_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.destination_device_staging_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.device_control_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.sampler_event_bytes))
            .and_then(|bytes| bytes.checked_add(evidence.envelope.cold_transient_workspace_bytes))
            .ok_or_else(|| GpuMeasurementError("preimage device bytes overflow".into()))?;
        let device_bytes = u64::try_from(device_bytes)
            .map_err(|_| GpuMeasurementError("preimage peak bytes exceed u64".into()))?;
        let host_bytes = u64::try_from(evidence.envelope.host_staging_bytes)
            .map_err(|_| GpuMeasurementError("preimage host staging exceeds u64".into()))?;
        let pinned_host_bytes = u64::try_from(evidence.envelope.pinned_host_control_bytes)
            .map_err(|_| GpuMeasurementError("preimage pinned control exceeds u64".into()))?;
        let evidence_kind = match evidence.envelope.evidence_kind {
            mxx_primitives::sampler::trapdoor::gpu::PreimageAllocationEvidenceKind::Exact => {
                MemoryEvidenceKind::ExactQuery
            }
            mxx_primitives::sampler::trapdoor::gpu::PreimageAllocationEvidenceKind::Certified => {
                MemoryEvidenceKind::CertifiedEnvelope
            }
        };
        let core_bytes = device_bytes
            .saturating_sub(evidence.envelope.source_device_staging_bytes as u64)
            .saturating_sub(evidence.envelope.destination_device_staging_bytes as u64);
        let mut affected_devices = BTreeMap::new();
        let mut add_physical = |physical: i32, bytes: usize| -> Result<(), GpuMeasurementError> {
            if bytes == 0 {
                return Ok(());
            }
            let identity = gpu_device_runtime_identity(physical).map_err(GpuMeasurementError)?;
            let logical = self.physical_device_index(physical).unwrap_or(key.device.logical_device);
            let owner = GpuWarmupDeviceIdentity::new(
                logical,
                format!(
                    "uuid={}-{}-sm{}.{}-mem{}",
                    identity.uuid,
                    identity.name,
                    identity.compute_major,
                    identity.compute_minor,
                    identity.total_global_memory
                ),
                format!(
                    "{}:driver{}:runtime{}",
                    identity.native_kernel_revision,
                    identity.driver_version,
                    identity.runtime_version,
                ),
            )
            .with_context_generation(identity.context_generation);
            let bytes = u64::try_from(bytes)
                .map_err(|_| GpuMeasurementError("preimage device bytes exceed u64".into()))?;
            *affected_devices.entry(owner).or_insert(0u64) =
                affected_devices.get(&owner).copied().unwrap_or(0).saturating_add(bytes);
            Ok(())
        };
        if let Some(&destination) = evidence.context.destination_device_ids.first() {
            add_physical(destination, usize::try_from(core_bytes).unwrap_or(usize::MAX))?;
            add_physical(destination, evidence.envelope.destination_device_staging_bytes)?;
        }
        if let Some(&source) = evidence.context.source_device_ids.first() {
            add_physical(source, evidence.envelope.source_device_staging_bytes)?;
        }
        if affected_devices.is_empty() {
            affected_devices.insert(key.device.clone(), device_bytes);
        }
        let width_key = (key.clone(), evidence.context.tile_columns);
        self.preimage_profile_metadata
            .insert(width_key.clone(), (max_attempts, evidence.context.tile_columns));
        self.preimage_footprints.insert(width_key.clone(), footprint);
        self.preimage_memory.insert(
            width_key,
            GpuWarmupMemoryObservations {
                affected_devices,
                host_bytes,
                pinned_host_bytes,
                evidence: evidence_kind,
            },
        );
        Ok(())
    }

    fn descriptor_for_pending(
        &self,
        signature: GpuWarmupOperationSignature,
        pending: &PendingMeasurement,
        domain: CanonicalWarmupProfileDomain,
    ) -> GpuWarmupOperationDescriptor {
        if let Some(descriptor) = &pending.warmup {
            return descriptor.clone();
        }
        GpuWarmupOperationDescriptor {
            inputs: Default::default(),
            signature,
            scope: pending.scope.clone(),
            node: pending.id,
            kind: pending.kind.clone(),
            concrete_argument_types: pending.concrete_argument_types.clone(),
            concrete_output_types: pending.concrete_output_types.clone(),
            bindings: pending.bindings.clone(),
            effective_operation: domain.identity().to_owned(),
            profile_domain: domain,
            fused_operation: self.warmup_fused_operations.get(&signature).copied().flatten(),
            implementation_variant: self
                .warmup_implementation_variants
                .get(&signature)
                .cloned()
                .unwrap_or_else(|| domain.identity().to_owned()),
            source_layouts: Vec::new(),
            output_layout: None,
            route_resolver: None,
            host_control: self
                .host_control_operations
                .get(&(pending.scope.clone(), pending.id))
                .cloned(),
        }
    }

    fn profile_key_for_pending(
        &self,
        request: &GpuWarmupProfileRequest,
        pending: &PendingMeasurement,
        domain: CanonicalWarmupProfileDomain,
    ) -> Result<GpuWarmupProfileKey, GpuWarmupProfileError> {
        let descriptor = self.descriptor_for_pending(request.signature, pending, domain);
        let device = if matches!(
            column_capability(&descriptor.kind, &descriptor.concrete_argument_types),
            ColumnCapability::SingleDevice | ColumnCapability::HostOrControl
        ) {
            0
        } else {
            request.device
        };
        self.profile_key(request, &descriptor, domain, device)
    }

    /// Query the primitive sampler's fixed-width cold allocation envelope for
    /// a registered preimage operation. The representative is prepared using
    /// the same production inputs as the timed fused dispatch; no width slope
    /// or generic workspace estimate is substituted for the native query.
    pub fn preimage_allocation_evidence_for_request(
        &mut self,
        request: &GpuWarmupProfileRequest,
        max_attempts: usize,
    ) -> Result<PreimageAllocationEvidence, GpuMeasurementError> {
        if max_attempts == 0 {
            return Err(GpuMeasurementError("preimage max_attempts must be positive".into()));
        }
        let descriptor =
            self.warmup_operations.get(&request.signature).cloned().ok_or_else(|| {
                GpuMeasurementError("preimage operation is not registered".into())
            })?;
        if !matches!(descriptor.kind, NodeKind::PreimageSample { .. }) {
            return Err(GpuMeasurementError(
                "allocation evidence requested for non-preimage operation".into(),
            ));
        }
        let width = request
            .range
            .end
            .checked_sub(request.range.start)
            .ok_or_else(|| GpuMeasurementError("preimage evidence range is reversed".into()))?;
        if width == 0 || width > request.tile_width {
            return Err(GpuMeasurementError("preimage evidence range has invalid width".into()));
        }
        let device = self
            .worker_index_for_logical_device(request.device)
            .ok_or_else(|| GpuMeasurementError("preimage evidence device is unavailable".into()))?;
        let worker = self
            .workers
            .get_mut(device)
            .ok_or_else(|| GpuMeasurementError("preimage evidence device is unavailable".into()))?;
        let representative = Self::representative_at(&descriptor, request.range.start, width);
        let node = MeasurementNode {
            scope: &descriptor.scope,
            id: descriptor.id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let prepared = if let Some(prepared) =
            self.preimage_session_inputs.get(&(request.signature.clone(), device)).cloned()
        {
            prepared
        } else {
            Self::prepare(&mut worker.backend, &node, &descriptor.bindings, None)?
        };
        let (public, trapdoor, sigma, _gadget_base, _digit_count, bound) =
            prepared.preimage_trapdoor.as_ref().ok_or_else(|| {
                GpuMeasurementError("preimage trapdoor preparation is missing".into())
            })?;
        let target = prepared
            .preimage_target
            .as_ref()
            .ok_or_else(|| GpuMeasurementError("preimage target preparation is missing".into()))?;
        let matrix_type = representative
            .concrete_output_types
            .iter()
            .find_map(|wire_type| wire_type.matrix_type())
            .ok_or_else(|| GpuMeasurementError("preimage output matrix type is missing".into()))?;
        let config =
            mxx_primitives::sampler::trapdoor::gpu::FixedPreimageConfig::new(width, max_attempts)
                .ok_or_else(|| {
                GpuMeasurementError("preimage evidence configuration is invalid".into())
            })?;
        // The runtime request carries an opaque identity generated from the
        // native sampler owner.  Requiring it here prevents an evidence query
        // from being admitted as a shape-only profile.  The native identity
        // itself is retained below so cold and warm observations are checked
        // against the same sampler owner session.
        if request.cache_state == GpuWarmupCacheState::Warm {
            request
                .require_preimage_cache_identity()
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
        }
        let evidence = worker
            .backend
            .preimage_allocation_evidence_for_range(
                matrix_type,
                *sigma,
                bound,
                trapdoor,
                public,
                target.as_ref(),
                request.range.start,
                request.range.end,
                config,
                match request.cache_state {
                    GpuWarmupCacheState::Cold => PreimageCacheState::Cold,
                    GpuWarmupCacheState::Warm => PreimageCacheState::Warm,
                },
            )
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let session_key = (request.signature.clone(), device);
        if let Some(previous) = self.preimage_evidence_identities.get(&session_key) {
            if previous != &evidence.context.cache_identity {
                return Err(GpuMeasurementError(
                    "preimage allocation evidence changed native cache owner within one warmup session"
                        .into(),
                ));
            }
        } else {
            self.preimage_evidence_identities.insert(session_key, evidence.context.cache_identity);
        }
        Ok(evidence)
    }

    /// Query the native allocation envelope for every GPU-measured warmup
    /// point.  The query is deliberately independent of the allocator peak
    /// measured by the timing harness: the latter is useful as a diagnostic
    /// workspace value, while this exact/certified envelope is the only
    /// observation used for hard admission.  `output_inclusive` is checked
    /// here so the output owner is added exactly once and is never folded into
    /// the measured workspace a second time.
    fn warmup_allocation_observations(
        worker_device_ids: &[i32],
        worker: &mut GpuMeasurementWorker,
        domain: CanonicalWarmupProfileDomain,
        representative: &RepresentativeMeasurement,
        prepared: &PreparedMeasurement,
        request: &GpuWarmupProfileRequest,
        bindings: &ParamEnv,
    ) -> Result<GpuWarmupMemoryObservations, GpuMeasurementError> {
        let output_columns = representative
            .concrete_output_types
            .iter()
            .find_map(Self::matrix_columns)
            .unwrap_or(request.tile_width);
        // Representatives retain validated global shapes and ownership.
        // Lower the actual measured global output range, then query input
        // replicas and the output width independently (transpose, tensor and
        // decomposition need not have equal input/output column counts).
        let local_width = output_columns.max(1);
        let range = representative
            .output_range
            .as_ref()
            .map_or(0..local_width, |range| range.start..range.end);
        if range.start >= range.end {
            return Err(GpuMeasurementError("warmup allocation query has an empty range".into()));
        }
        let gadget_preimage = domain == CanonicalWarmupProfileDomain::FusedDecompose &&
            matches!(representative.kind, NodeKind::PreimageSample { .. });
        let mapped_inputs = if gadget_preimage {
            let target_operand =
                if representative.concrete_argument_types.len() >= 3 { 2 } else { 0 };
            let target = representative
                .concrete_argument_types
                .get(target_operand)
                .and_then(Self::matrix_columns)
                .ok_or_else(|| {
                    GpuMeasurementError(
                        "gadget preimage allocation query lacks a target matrix".into(),
                    )
                })?;
            vec![InputColumnRange {
                operand: target_operand,
                range: ColumnRange { start: range.start, end: range.end.min(target) },
            }]
        } else if column_capability(&representative.kind, &representative.concrete_argument_types) ==
            ColumnCapability::SingleDevice
        {
            representative
                .concrete_argument_types
                .iter()
                .enumerate()
                .filter_map(|(operand, ty)| {
                    Self::matrix_columns(ty).map(|end| InputColumnRange {
                        operand,
                        range: ColumnRange { start: 0, end },
                    })
                })
                .collect()
        } else {
            map_output_range_to_inputs_with_output(
                &representative.kind,
                &representative.concrete_argument_types,
                local_width,
                ColumnRange { start: range.start, end: range.end },
            )
            .map_err(|error| GpuMeasurementError(error.to_string()))?
        };
        let first_input = mapped_inputs
            .iter()
            .find_map(|mapped| prepared.arguments.get(mapped.operand).and_then(Option::as_ref));
        let envelope = if let Some(input) = first_input {
            let mapped =
                mapped_inputs
                    .iter()
                    .find(|mapped| {
                        prepared.arguments.get(mapped.operand).and_then(Option::as_ref).is_some_and(
                            |candidate| std::ptr::eq(candidate.as_ref(), input.as_ref()),
                        )
                    })
                    .ok_or_else(|| {
                        GpuMeasurementError(
                            "mapped warmup input disappeared before allocation query".into(),
                        )
                    })?;
            let input_end = mapped.range.end.min(input.size().1);
            let input_start = mapped.range.start.min(input_end);
            if input_start >= input_end {
                return Err(GpuMeasurementError(
                    "warmup allocation query range is outside the representative input".into(),
                ));
            }
            let mut components = worker
                .backend
                .range_replica_allocation_components(
                    input,
                    input_start..input_end,
                    worker.device_id,
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
            components.evidence = Some(GpuAllocationEvidenceKind::ExactQuery);
            components.product_count = match &representative.kind {
                NodeKind::MatrixMulAccumulate { coefficients, .. } => coefficients.len(),
                _ => 1,
            };
            components.level_count = representative.concrete_argument_types.len().max(1);
            if domain == CanonicalWarmupProfileDomain::FusedTensorRowSum {
                let groups = representative.inputs.row_groups.clone();
                let rhs = prepared.arguments.get(1).and_then(Option::as_ref).ok_or_else(|| {
                    GpuMeasurementError(
                        "tensor row-sum allocation query lacks its rhs owner".into(),
                    )
                })?;
                let lhs_shard = input.shards().first().ok_or_else(|| {
                    GpuMeasurementError("tensor row-sum allocation query lacks lhs shard".into())
                })?;
                let rhs_shard = rhs.shards().first().ok_or_else(|| {
                    GpuMeasurementError("tensor row-sum allocation query lacks rhs shard".into())
                })?;
                let lhs_size = input.size();
                let rhs_size = rhs.size();
                components.input_shape = Some(
                    GpuMemoryShape::new(
                        lhs_shard.value.level(),
                        lhs_size.0,
                        lhs_size.1,
                        lhs_shard.value.is_ntt(),
                        GpuMemoryRange::new(input_start, input_end).ok_or_else(|| {
                            GpuMeasurementError(
                                "tensor row-sum lhs allocation range is empty".into(),
                            )
                        })?,
                    )
                    .map_err(|error| GpuMeasurementError(error.to_string()))?,
                );
                components.tensor_row_sum_rhs_shape = Some(
                    GpuMemoryShape::new(
                        rhs_shard.value.level(),
                        rhs_size.0,
                        rhs_size.1,
                        rhs_shard.value.is_ntt(),
                        GpuMemoryRange::new(0, rhs_size.1).ok_or_else(|| {
                            GpuMeasurementError(
                                "tensor row-sum rhs allocation range is empty".into(),
                            )
                        })?,
                    )
                    .map_err(|error| GpuMeasurementError(error.to_string()))?,
                );
                components.tensor_row_sum_groups = Some(groups);
            }
            if domain == CanonicalWarmupProfileDomain::FusedDecompose {
                let (small, digits) = match &representative.kind {
                    NodeKind::GadgetDecompose { small, digit_count, .. } => (
                        *small,
                        digit_count
                            .evaluate(bindings)
                            .map_err(|error| GpuMeasurementError(error.to_string()))?
                            .to_usize()
                            .ok_or_else(|| {
                                GpuMeasurementError("invalid decomposition digit count".into())
                            })?,
                    ),
                    NodeKind::PreimageSample { .. } => {
                        if let Some(ConcreteWireType::Trapdoor { digit_count, .. }) =
                            representative.concrete_argument_types.get(1)
                        {
                            (false, *digit_count)
                        } else {
                            let target = representative
                                .concrete_argument_types
                                .first()
                                .and_then(matrix_leaf_type)
                                .map(|matrix| matrix.rows)
                                .ok_or_else(|| {
                                    GpuMeasurementError(
                                        "gadget preimage allocation query lacks target matrix"
                                            .into(),
                                    )
                                })?;
                            let output = representative
                                .concrete_output_types
                                .first()
                                .and_then(matrix_leaf_type)
                                .map(|matrix| matrix.rows)
                                .ok_or_else(|| {
                                    GpuMeasurementError(
                                        "gadget preimage allocation query lacks output matrix"
                                            .into(),
                                    )
                                })?;
                            let digits = output.checked_div(target).ok_or_else(|| {
                                GpuMeasurementError(
                                    "gadget preimage allocation query has invalid row metadata"
                                        .into(),
                                )
                            })?;
                            (false, digits)
                        }
                    }
                    _ => {
                        return Err(GpuMeasurementError(
                            "fused decomposition lacks its native contract".into(),
                        ))
                    }
                };
                components.fused_decompose_evidence = Some(
                    worker
                        .backend
                        .fused_decompose_allocation_evidence(
                            &prepared
                                .arguments
                                .iter()
                                .flatten()
                                .map(Arc::as_ref)
                                .collect::<Vec<_>>(),
                            small,
                            digits,
                            range.clone(),
                        )
                        .map_err(|error| GpuMeasurementError(error.to_string()))?,
                );
            }
            if matches!(
                domain,
                CanonicalWarmupProfileDomain::MatrixMulSmallRhs |
                    CanonicalWarmupProfileDomain::FusedCompactProduct
            ) {
                let rhs = prepared.small_arguments.iter().flatten().next().ok_or_else(|| {
                    GpuMeasurementError("compact product has no native RHS".into())
                })?;
                components.small_rhs_report = Some(
                    worker
                        .backend
                        .compact_product_allocation_report(input, rhs, range.clone())
                        .map_err(|error| GpuMeasurementError(error.to_string()))?,
                );
                if domain == CanonicalWarmupProfileDomain::FusedCompactProduct {
                    let report =
                        components.small_rhs_report.as_mut().expect("installed compact report");
                    for block in prepared
                        .arguments
                        .iter()
                        .flatten()
                        .filter(|block| !Arc::ptr_eq(block, input))
                    {
                        let extra = worker
                            .backend
                            .compact_product_allocation_report(block, rhs, range.clone())
                            .map_err(|error| GpuMeasurementError(error.to_string()))?;
                        report.full_output_bytes = report
                            .full_output_bytes
                            .checked_add(extra.full_output_bytes)
                            .ok_or_else(|| {
                                GpuMeasurementError("fused output allocation overflows".into())
                            })?;
                        report.event_overhead_bytes = report
                            .event_overhead_bytes
                            .checked_add(extra.event_overhead_bytes)
                            .ok_or_else(|| {
                                GpuMeasurementError("fused output events overflow".into())
                            })?;
                        report.expanded_rhs_workspace_bytes = report
                            .expanded_rhs_workspace_bytes
                            .max(extra.expanded_rhs_workspace_bytes);
                    }
                }
            }
            worker
                .backend
                .matrix_range_allocation_envelope(
                    domain,
                    input,
                    representative
                        .concrete_output_types
                        .iter()
                        .find_map(|ty| ty.matrix_type())
                        .ok_or_else(|| {
                            GpuMeasurementError(
                                "GPU operation lacks its concrete output type".into(),
                            )
                        })?,
                    range.end - range.start,
                    input_start..input_end,
                    components,
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?
        } else if domain == CanonicalWarmupProfileDomain::CenteredRebase {
            // Compact CenteredRebase has no regular DCRT argument.  Query
            // the native compact owner topology directly so warmup charges
            // the same compact-to-compact D2D/alias envelope as production;
            // falling through to generated/full-modulus accounting would
            // widen the payload and admit a different resource contract.
            let mapped = mapped_inputs.first().ok_or_else(|| {
                GpuMeasurementError("compact centered rebase has no mapped input".into())
            })?;
            let input =
                prepared.small_arguments.get(mapped.operand).and_then(Option::as_ref).ok_or_else(
                    || GpuMeasurementError("compact centered rebase input owner is missing".into()),
                )?;
            let output_type = representative
                .concrete_output_types
                .iter()
                .find_map(|ty| ty.matrix_type())
                .ok_or_else(|| {
                    GpuMeasurementError("compact centered rebase output type is missing".into())
                })?;
            let input_start = mapped.range.start.min(input.size().1);
            let input_end = mapped.range.end.min(input.size().1);
            if input_start >= input_end {
                return Err(GpuMeasurementError(
                    "compact centered rebase allocation range is empty".into(),
                ));
            }
            worker
                .backend
                .compact_centered_rebase_allocation_envelope(
                    input,
                    output_type,
                    input_start..input_end,
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?
        } else {
            let matrix_type = representative
                .concrete_output_types
                .iter()
                .find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. } |
                    ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix.clone()),
                    _ => None,
                })
                .ok_or_else(|| {
                    GpuMeasurementError(
                        "GPU warmup allocation query has no matrix output or input".into(),
                    )
                })?;
            worker
                .backend
                .generated_range_allocation_envelope(
                    domain,
                    &matrix_type,
                    range,
                    GpuAllocationComponents {
                        evidence: Some(GpuAllocationEvidenceKind::ExactQuery),
                        ..GpuAllocationComponents::default()
                    },
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?
        };
        let mut envelope = envelope;
        // Account for every actual retained input participating in the
        // lowered operation.  The native output query above is anchored at
        // the first mapped operand, while the remaining operands may still
        // require residency (and, on a sharded fleet, a destination replica).
        for (operand, input) in prepared.arguments.iter().enumerate() {
            let Some(input) = input else { continue };
            if mapped_inputs.iter().any(|mapped| mapped.operand == operand) {
                if !std::ptr::eq(
                    input.as_ref(),
                    first_input.map(Arc::as_ref).unwrap_or(input.as_ref()),
                ) {
                    let resident = worker
                        .backend
                        .resident_allocation_bytes(input)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    envelope.input_resident_bytes =
                        envelope.input_resident_bytes.checked_add(resident).ok_or_else(|| {
                            GpuMeasurementError("GPU input residency overflows usize".into())
                        })?;
                }
                if let Some(mapped) = mapped_inputs.iter().find(|mapped| mapped.operand == operand)
                {
                    let components = worker
                        .backend
                        .range_replica_allocation_components(
                            input,
                            mapped.range.start..mapped.range.end.min(input.size().1),
                            worker.device_id,
                        )
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    envelope.replica_bytes = envelope
                        .replica_bytes
                        .checked_add(components.replica_bytes)
                        .ok_or_else(|| GpuMeasurementError("GPU replica bytes overflow".into()))?;
                }
            }
        }
        // Preserve the native ownership split.  A multi-device range may
        // retain source shards, allocate a destination output/replica, and
        // briefly stage bytes on a third owner.  Collapsing the inclusive
        // envelope onto `request.device` makes an otherwise fitting plan look
        // valid on the wrong GPU, so carry a physical-device map all the way
        // into the warmup observation.
        let mut per_device_bytes = BTreeMap::<i32, usize>::new();
        let mut add_device_bytes = |device: i32, bytes: usize| -> Result<(), GpuMeasurementError> {
            let entry = per_device_bytes.entry(device).or_insert(0);
            *entry = entry.checked_add(bytes).ok_or_else(|| {
                GpuMeasurementError("GPU per-device envelope overflows usize".into())
            })?;
            Ok(())
        };
        for input in prepared.arguments.iter().flatten() {
            for (owner, bytes) in worker
                .backend
                .resident_allocation_bytes_by_device(input)
                .map_err(|error| GpuMeasurementError(error.to_string()))?
            {
                add_device_bytes(owner, bytes)?;
            }
        }
        for input in prepared.small_arguments.iter().flatten() {
            for (owner, bytes) in worker
                .backend
                .resident_small_allocation_bytes_by_device(input)
                .map_err(|error| GpuMeasurementError(error.to_string()))?
            {
                add_device_bytes(owner, bytes)?;
            }
        }
        let destination_physical = request
            .route_descriptor
            .destination_device
            .and_then(|logical| worker_device_ids.get(logical).copied())
            .unwrap_or(worker.device_id);
        let destination_bytes = envelope
            .output_bytes
            .saturating_add(envelope.auxiliary_bytes)
            .saturating_add(envelope.scratch_bytes)
            .saturating_add(envelope.assembly_bytes)
            .saturating_add(envelope.replica_bytes)
            .saturating_add(envelope.destination_device_bytes)
            .saturating_add(envelope.transfer_bytes);
        add_device_bytes(destination_physical, destination_bytes)?;
        for source_route in request.route_descriptor.source_routes() {
            if let Some(source_physical) = worker_device_ids.get(source_route.source_owner).copied()
            {
                add_device_bytes(source_physical, source_route.source_staging_bytes)?;
            }
        }
        // Host-staged transfers retain host/pinned bytes separately; the
        // source and destination device staging allocations above remain on
        // their respective physical owners.  If no resident input was
        // available (e.g. generated constants), the destination entry still
        // records the complete native output envelope.
        envelope.per_device_bytes = per_device_bytes;
        if !envelope.output_inclusive {
            return Err(GpuMeasurementError(
                "GPU warmup allocation envelope must include the output owner".into(),
            ));
        }
        let device_bytes = envelope
            .total_device_bytes()
            .ok_or_else(|| GpuMeasurementError("GPU allocation envelope overflows usize".into()))?;
        let host_bytes = u64::try_from(envelope.host_bytes)
            .map_err(|_| GpuMeasurementError("GPU host staging exceeds u64".into()))?;
        let pinned_host_bytes = u64::try_from(envelope.pinned_host_bytes)
            .map_err(|_| GpuMeasurementError("GPU pinned staging exceeds u64".into()))?;
        let identity =
            gpu_device_runtime_identity(worker.device_id).map_err(GpuMeasurementError)?;
        let device = GpuWarmupDeviceIdentity::new(
            request.device,
            format!(
                "uuid={}-{}-sm{}.{}-mem{}",
                identity.uuid,
                identity.name,
                identity.compute_major,
                identity.compute_minor,
                identity.total_global_memory
            ),
            format!(
                "{}:driver{}:runtime{}",
                identity.native_kernel_revision, identity.driver_version, identity.runtime_version,
            ),
        )
        .with_context_generation(identity.context_generation);
        let evidence = match envelope.evidence {
            GpuAllocationEvidenceKind::ExactQuery => MemoryEvidenceKind::ExactQuery,
            GpuAllocationEvidenceKind::CertifiedAllocationEnvelope => {
                MemoryEvidenceKind::CertifiedEnvelope
            }
        };
        let affected_devices = if envelope.per_device_bytes.is_empty() {
            BTreeMap::from([(
                device,
                u64::try_from(device_bytes).map_err(|_| {
                    GpuMeasurementError("GPU allocation envelope exceeds u64".into())
                })?,
            )])
        } else {
            envelope
                .per_device_bytes
                .iter()
                .map(|(physical, bytes)| {
                    let identity =
                        gpu_device_runtime_identity(*physical).map_err(GpuMeasurementError)?;
                    let owner = GpuWarmupDeviceIdentity::new(
                        worker_device_ids
                            .iter()
                            .position(|device_id| *device_id == *physical)
                            .unwrap_or(request.device),
                        format!(
                            "uuid={}-{}-sm{}.{}-mem{}",
                            identity.uuid,
                            identity.name,
                            identity.compute_major,
                            identity.compute_minor,
                            identity.total_global_memory
                        ),
                        format!(
                            "{}:driver{}:runtime{}",
                            identity.native_kernel_revision,
                            identity.driver_version,
                            identity.runtime_version,
                        ),
                    )
                    .with_context_generation(identity.context_generation);
                    Ok((
                        owner,
                        u64::try_from(*bytes).map_err(|_| {
                            GpuMeasurementError("GPU per-device envelope exceeds u64".into())
                        })?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, GpuMeasurementError>>()?
        };
        Ok(GpuWarmupMemoryObservations {
            affected_devices,
            host_bytes,
            pinned_host_bytes,
            evidence,
        })
    }

    /// Measures and stores one exact setup-time candidate through the
    /// device-local production range path. This is deliberately separate from
    /// [`Self::measure_collected`]: warmup may request several nonlinear `b`
    /// candidates, while fixed execution has no access to this method.
    /// Convert every typed or returned CUDA OOM from the setup measurement
    /// into an infeasible candidate only after the same recovery boundary used
    /// for panic-originating allocator failures.  In particular, callers must
    /// never observe an OOM while release work is still queued on the backend
    /// streams: the next (usually smaller) candidate is allowed to reuse the
    /// context only after this fence and allocator/context validation pass.
    fn recover_measurement_oom(&mut self, physical: i32, message: String) -> GpuWarmupProfileError {
        let worker_index = self.worker_index_for_physical_device(physical);
        let Some(worker) = worker_index.and_then(|index| self.workers.get_mut(index)) else {
            return GpuWarmupProfileError::Measurement(format!(
                "OOM cleanup cannot find selected physical device {physical}: {message}"
            ));
        };

        if let Err(error) = worker.backend.fence_released_memory() {
            return GpuWarmupProfileError::Measurement(format!(
                "OOM release fence failed on GPU {physical}: {error}"
            ));
        }

        // The failed allocation may have left the async pool's high-water
        // marker and temporary blocks live. Reset it only after the release
        // stream has been fenced, and treat a reset failure as fatal: silently
        // admitting the next candidate would make its memory evidence stale.
        if let Err(error) = gpu_default_mempool_reset_high_water(physical) {
            return GpuWarmupProfileError::Measurement(format!(
                "OOM allocator cleanup failed on GPU {physical}: {error}"
            ));
        }

        let memory = match gpu_device_memory_usage(physical) {
            Ok(memory) => memory,
            Err(error) => {
                return GpuWarmupProfileError::Measurement(format!(
                    "OOM context usability validation failed on GPU {physical}: {error}"
                ));
            }
        };
        let owned_contexts = self
            .workers
            .iter()
            .map(|worker| worker.backend.owned_context_count(physical))
            .sum::<usize>();
        if memory.live_contexts != owned_contexts || owned_contexts == 0 {
            return GpuWarmupProfileError::Measurement(format!(
                "OOM context usability validation found {} live contexts on GPU {physical}",
                memory.live_contexts
            ));
        }
        // Re-read after allocator cleanup.  Besides checking that the context
        // remains usable, this catches a reset/recreate race before the OOM is
        // returned to the candidate loop.
        let validated = match gpu_device_memory_usage(physical) {
            Ok(memory) => memory,
            Err(error) => {
                return GpuWarmupProfileError::Measurement(format!(
                    "OOM post-cleanup context validation failed on GPU {physical}: {error}"
                ));
            }
        };
        if validated.live_contexts != owned_contexts ||
            validated.context_generation != memory.context_generation
        {
            return GpuWarmupProfileError::Measurement(format!(
                "GPU {physical} context changed during OOM cleanup (generation {} -> {}, live contexts {} -> {})",
                memory.context_generation,
                validated.context_generation,
                memory.live_contexts,
                validated.live_contexts
            ));
        }
        GpuWarmupProfileError::OutOfMemory(message)
    }

    /// An inclusive range measurement can have live allocations on both the
    /// source and destination GPUs (and on either owner when a route changes
    /// to host staging).  Recover every affected context before admitting the
    /// next candidate; cleaning only the request device leaves queued frees
    /// on the other owner and turns the following small candidate into a
    /// false OOM.
    fn recover_measurement_oom_for_request(
        &mut self,
        request: &GpuWarmupProfileRequest,
        message: String,
    ) -> GpuWarmupProfileError {
        let mut devices = BTreeSet::new();
        devices.insert(self.measurement_device_for_request(request));
        if let Some(device) = request.route_descriptor.source_device {
            devices.insert(device);
        }
        if let Some(device) = request.route_descriptor.destination_device {
            devices.insert(device);
        }
        // A mapped/concat/tensor range may touch several physical source
        // owners.  Cleanup must fence every owner recorded by the route
        // contract, otherwise a queued release on an unlisted shard can make
        // the next (smaller) candidate appear to OOM spuriously.
        devices.extend(
            request.route_descriptor.source_routes().iter().map(|route| route.source_owner),
        );
        let physical_devices = self
            .workers
            .iter()
            .flat_map(|worker| worker.backend.physical_device_ids())
            .collect::<Vec<_>>();
        let mut first_failure = None;
        for device in devices {
            let Some(physical) = physical_devices.get(device).copied() else {
                if first_failure.is_none() {
                    first_failure = Some(GpuWarmupProfileError::Measurement(format!(
                        "OOM cleanup route owner {device} is not present in the measurement fleet"
                    )));
                }
                continue;
            };
            let result = self.recover_measurement_oom(physical, message.clone());
            if !matches!(result, GpuWarmupProfileError::OutOfMemory(_)) && first_failure.is_none() {
                first_failure = Some(result);
            }
        }
        first_failure.unwrap_or(GpuWarmupProfileError::OutOfMemory(message))
    }

    fn measurement_device_for_request(&self, request: &GpuWarmupProfileRequest) -> usize {
        self.warmup_operations
            .get(&request.signature)
            .filter(|descriptor| {
                column_capability(&descriptor.kind, &descriptor.concrete_argument_types) ==
                    ColumnCapability::SingleDevice ||
                    column_capability(&descriptor.kind, &descriptor.concrete_argument_types) ==
                        ColumnCapability::HostOrControl
            })
            .map(|_| 0)
            .unwrap_or(request.device)
    }

    fn classify_measurement_error(error: GpuMeasurementError) -> GpuWarmupProfileError {
        if error.is_out_of_memory() {
            GpuWarmupProfileError::OutOfMemory(error.to_string())
        } else {
            GpuWarmupProfileError::Measurement(error.to_string())
        }
    }

    /// Public setup boundary.  The inner implementation deliberately returns
    /// typed OOM errors from fallible backend calls; this wrapper gives those
    /// errors exactly the same release-fence, allocator cleanup, and context
    /// validation as typed OOM panic payloads.
    pub fn measure_warmup_profile(
        &mut self,
        request: &GpuWarmupProfileRequest,
    ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
        let result =
            panic::catch_unwind(AssertUnwindSafe(|| self.measure_warmup_profile_inner(request)));
        match result {
            Ok(Ok(profile)) => Ok(profile),
            Ok(Err(GpuWarmupProfileError::OutOfMemory(message))) => {
                Err(self.recover_measurement_oom_for_request(request, message))
            }
            Ok(Err(error)) => Err(error),
            Err(payload) => match payload.downcast::<GpuOutOfMemory>() {
                Ok(error) => {
                    Err(self.recover_measurement_oom_for_request(request, error.to_string()))
                }
                Err(payload) => panic::resume_unwind(payload),
            },
        }
    }

    fn measure_warmup_profile_inner(
        &mut self,
        request: &GpuWarmupProfileRequest,
    ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
        let descriptor = self
            .warmup_operations
            .get(&request.signature)
            .cloned()
            .ok_or_else(|| GpuWarmupProfileError::MissingProfile(request.clone()))?;
        let profile_domain =
            self.warmup_profile_domains.get(&request.signature).copied().ok_or_else(|| {
                GpuWarmupProfileError::Measurement(
                    "warmup operation has no registered canonical profile domain".into(),
                )
            })?;
        if request.timing_scope == GpuWarmupTimingScope::Transfer {
            if request.route_descriptor.transfer_bytes() == 0 {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "Transfer profile requires a positive physical transfer payload".into(),
                ));
            }
        }
        let selected_device =
            if column_capability(&descriptor.kind, &descriptor.concrete_argument_types) ==
                ColumnCapability::SingleDevice
            {
                0
            } else {
                self.worker_index_for_logical_device(request.device).ok_or_else(|| {
                    GpuWarmupProfileError::Measurement("warmup device is unavailable".into())
                })?
            };
        if profile_domain.measurement_kind() == WarmupMeasurementKind::GpuMeasured &&
            request.timing_scope != GpuWarmupTimingScope::Transfer
        {
            let resolver = request.route_resolver.as_ref().ok_or_else(|| {
                GpuWarmupProfileError::InvalidMeasurement(format!(
                    "GPU warmup request for {profile_domain:?} is missing its production route resolver"
                ))
            })?;
            let fragment = match request.fragment {
                mxx_runtime::backend::GpuWarmupFragmentClass::Whole => TypedFragmentClass::Full,
                mxx_runtime::backend::GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
                mxx_runtime::backend::GpuWarmupFragmentClass::Mapped => TypedFragmentClass::Mapped,
                mxx_runtime::backend::GpuWarmupFragmentClass::Fragmented => {
                    TypedFragmentClass::CompactFragment
                }
                mxx_runtime::backend::GpuWarmupFragmentClass::SingleDevice => {
                    TypedFragmentClass::Full
                }
            };
            let expected_route = resolver.resolve_for(
                request.route_descriptor.source_range,
                request.route_descriptor.destination_range,
                fragment,
                request.route_descriptor.source_compact,
                request.route_descriptor.destination_compact,
            );
            if expected_route != request.route_descriptor &&
                !mxx_runtime::gpu_warmup::route_response_matches_resolver(
                    request.route_descriptor,
                    resolver,
                )
            {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "warmup route descriptor does not match the production route resolver".into(),
                ));
            }
        }
        let profile_key = self.profile_key_for_pending(request, &descriptor, profile_domain)?;
        let coordinate = request.range.end.checked_sub(request.range.start).ok_or_else(|| {
            GpuWarmupProfileError::InvalidMeasurement("warmup range is reversed".into())
        })?;
        if coordinate == 0 {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "warmup range must be non-empty".into(),
            ));
        }
        if self.harness.measured_iterations == 0 {
            return Err(GpuWarmupProfileError::Measurement(
                "warmup measurement requires at least one measured repetition".into(),
            ));
        }
        if !self.collecting {
            return Err(GpuWarmupProfileError::Measurement(
                "GPU warmup production measurement is setup-only and its collection window is closed"
                    .into(),
            ));
        }
        // This function is the provider's actual measurement boundary.  The
        // runtime session cache may satisfy later requests without entering
        // here, so count calls here rather than counting registered points.
        self.warmup_measurement_calls.fetch_add(1, Ordering::SeqCst);
        let fused_operation =
            self.warmup_fused_operations.get(&request.signature).copied().flatten();
        if let Some(fused) = fused_operation {
            let inputs = &descriptor
                .warmup
                .as_ref()
                .ok_or_else(|| {
                    GpuWarmupProfileError::InvalidMeasurement(
                        "fused measurement requires a validated descriptor".into(),
                    )
                })?
                .inputs;
            if inputs.origins.len() != descriptor.concrete_argument_types.len() ||
                (matches!(
                    fused,
                    FusedWarmupOperation::RowSum | FusedWarmupOperation::TensorRowSum
                ) && inputs.row_groups.is_empty())
            {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "fused descriptor must retain validated physical origins and row groups".into(),
                ));
            }
            let expected = fused_warmup_profile_domain(fused);
            if profile_domain != expected {
                return Err(GpuWarmupProfileError::InvalidMeasurement(format!(
                    "fused operation {fused:?} has profile domain {profile_domain:?}, expected {expected:?}"
                )));
            }
        }
        let capability = column_capability(&descriptor.kind, &descriptor.concrete_argument_types);
        if capability == ColumnCapability::HostOrControl &&
            matches!(
                descriptor.kind,
                NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_)
            )
        {
            let child =
                self.host_control_operations.get(&(descriptor.scope.clone(), descriptor.id));
            let (elapsed, spread, repetitions) = Self::measure_host_control_repeated(
                &self.harness,
                &descriptor.scope,
                descriptor.id,
                &descriptor.kind,
                &descriptor.bindings,
                child,
            )
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
            if !elapsed.is_finite() || elapsed <= 0.0 {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "host warmup timing must be finite and positive".into(),
                ));
            }
            let profile = GpuWarmupProfile::measured_with_observation(
                elapsed,
                0,
                WarmupMeasurementKind::HostMeasured,
                GpuWarmupMemoryObservations::explicit_exact_zero(),
                GpuWarmupResidencyDelta::default(),
                repetitions,
                spread,
                GpuWarmupProvenance::ProductionEquivalent,
                profile_key.cache_state,
                profile_key.timing_scope,
            )?;
            self.record_warmup_dispatch(request, profile_domain, fused_operation);
            self.warmup_measurement_provenances.push(profile.provenance);
            return Ok(profile);
        }
        // Host-visible codec primitives have two deliberately disjoint
        // measurements.  The main point times only the host dispatch after
        // its representative has been prepared; it does not enter the
        // inclusive local-production materializer (which owns D2H/H2D and
        // staging).  The latter is measured separately by the Transfer
        // request below.  This keeps a containing-stage composition from
        // charging one physical copy twice while still recording a positive
        // host time and exact zero GPU workspace.
        if profile_domain.measurement_kind() == WarmupMeasurementKind::HostMeasured &&
            Self::is_host_backend_boundary(&descriptor.kind) &&
            request.timing_scope != GpuWarmupTimingScope::Transfer
        {
            if selected_device >= self.workers.len() {
                return Err(GpuWarmupProfileError::Measurement(
                    "warmup production selected device is unavailable".into(),
                ));
            }
            let representative = RepresentativeMeasurement {
                source_layouts: Vec::new(),
                fixed_metadata: None,
                retry_cap: None,
                inputs: Default::default(),
                kind: descriptor.kind.clone(),
                concrete_argument_types: descriptor.concrete_argument_types.clone(),
                concrete_output_types: descriptor.concrete_output_types.clone(),
                fixed_arguments: Self::fixed_arguments(
                    &descriptor.kind,
                    &descriptor.concrete_argument_types,
                ),
                output_range: None,
            };
            let (elapsed, spread, repetitions) = Self::measure_host_boundary_repeated(
                &mut self.workers[selected_device],
                &self.harness,
                &descriptor.scope,
                descriptor.id,
                &descriptor.bindings,
                &representative,
            )
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
            let mut profile = GpuWarmupProfile::measured_with_observation(
                elapsed,
                0,
                WarmupMeasurementKind::HostMeasured,
                GpuWarmupMemoryObservations::explicit_exact_zero(),
                GpuWarmupResidencyDelta::default(),
                repetitions,
                spread,
                GpuWarmupProvenance::ProductionEquivalent,
                profile_key.cache_state,
                profile_key.timing_scope,
            )?;
            profile.resolved_route_descriptor = Some(
                Self::boundary_route(&self.workers[selected_device], &representative)
                    .map_err(Self::classify_measurement_error)?,
            );
            self.record_warmup_dispatch(request, profile_domain, fused_operation);
            self.warmup_measurement_provenances.push(profile.provenance);
            return Ok(profile);
        }
        // Scalar/container host primitives have no device-local production
        // range.  Dispatch them through the shared host implementation first
        // so their real host elapsed time is not hidden behind matrix setup
        // (and so a matrix-shaped semantic type cannot force a fake GPU
        // measurement).  Device/host boundary primitives below still use
        // their production backend path because they have an explicit
        // transfer stage.
        if profile_domain.measurement_kind() == WarmupMeasurementKind::HostMeasured &&
            Self::scalar_host_kind(&descriptor.kind)
        {
            // Matrix-shaped host inputs still need a typed representative for
            // the executor's input-clone boundary.  Constructing that value
            // uses the backend's constant constructor, which requires an
            // active one-column operation even though the timed host profile
            // charges no device workspace.  This setup width is never used
            // as a production GPU plan and remains outside the timed span.
            let has_nonzero_device = self.workers.len() > 1;
            self.workers[selected_device].backend.set_column_widths_for_operation(
                request.signature.operation,
                GpuColumnWidths { gpu0: 1, nonzero: has_nonzero_device.then_some(1) },
            );
            self.workers[selected_device]
                .backend
                .select_operation(request.signature.operation)
                .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
            let node = MeasurementNode {
                scope: &descriptor.scope,
                id: descriptor.id,
                kind: &descriptor.kind,
                arguments: &[],
                argument_kinds: &[],
                argument_types: &[],
                output_types: &[],
                concrete_argument_types: descriptor.concrete_argument_types.clone(),
                concrete_output_types: descriptor.concrete_output_types.clone(),
            };
            let (elapsed, spread, repetitions) = Self::measure_scalar_host_repeated(
                &mut self.workers[selected_device],
                &node,
                &descriptor.bindings,
                &self.harness,
            )
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
            let profile = GpuWarmupProfile::measured_with_observation(
                elapsed,
                0,
                WarmupMeasurementKind::HostMeasured,
                GpuWarmupMemoryObservations::explicit_exact_zero(),
                GpuWarmupResidencyDelta::default(),
                repetitions,
                spread,
                GpuWarmupProvenance::ProductionEquivalent,
                profile_key.cache_state,
                profile_key.timing_scope,
            )?;
            self.record_warmup_dispatch(request, profile_domain, fused_operation);
            self.warmup_measurement_provenances.push(profile.provenance);
            return Ok(profile);
        }
        let columns = descriptor
            .concrete_output_types
            .iter()
            .find_map(Self::matrix_columns)
            .or_else(|| descriptor.concrete_argument_types.iter().find_map(Self::matrix_columns))
            .unwrap_or(0);
        let range_capable = profile_domain.measurement_kind() == WarmupMeasurementKind::GpuMeasured &&
            !matches!(
                capability,
                ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
            );
        if range_capable &&
            (request.range.start >= request.range.end ||
                request.range.end - request.range.start > request.tile_width)
        {
            return Err(GpuWarmupProfileError::Measurement(
                "warmup production range must be non-empty and no wider than its tile width".into(),
            ));
        }
        if range_capable &&
            (request.range.end > columns || request.device >= self.physical_device_ids().len())
        {
            return Err(GpuWarmupProfileError::Measurement(
                "warmup production range is outside the selected device".into(),
            ));
        }
        // Host/control primitives have no device-local range.  Single-device
        // constants and samplers likewise execute their complete production
        // operation on GPU0; a range in the planner request is only a width
        // candidate for the column-separable path.  All other primitives use
        // the same local type/range lowering as fleet production.
        let mut representative = if range_capable {
            Self::representative_at(
                &descriptor,
                request.range.start,
                request.range.end - request.range.start,
            )
        } else {
            RepresentativeMeasurement {
                source_layouts: Vec::new(),
                fixed_metadata: None,
                retry_cap: None,
                inputs: Default::default(),
                kind: descriptor.kind.clone(),
                concrete_argument_types: descriptor.concrete_argument_types.clone(),
                concrete_output_types: descriptor.concrete_output_types.clone(),
                fixed_arguments: Self::fixed_arguments(
                    &descriptor.kind,
                    &descriptor.concrete_argument_types,
                ),
                output_range: None,
            }
        };
        representative.inputs = descriptor
            .warmup
            .as_ref()
            .map(|descriptor| descriptor.inputs.clone())
            .unwrap_or_default();
        representative.source_layouts = descriptor
            .warmup
            .as_ref()
            .map(|descriptor| descriptor.source_layouts.clone())
            .unwrap_or_default();
        representative.retry_cap = request.retry_cap;
        representative.fixed_metadata =
            Some(mxx_runtime::backend::PlannedNodeBatchRequest::for_lowered_operation(
                mxx_runtime::gpu_execution_plan::GpuExecutionSiteKey {
                    site: descriptor.id.0,
                    shape_class: request.signature.shape_class,
                    instance_class: request.signature.instance_class,
                },
                request.signature.operation,
                self.warmup_implementation_variants[&request.signature].clone(),
                representative.inputs.clone(),
                descriptor
                    .concrete_output_types
                    .iter()
                    .enumerate()
                    .map(|(port, ty)| {
                        mxx_runtime::backend::PlannedLayoutMetadata::for_type(ty, Some(port as u32))
                    })
                    .collect(),
                vec![request.tile_width],
            ));
        let device = selected_device;
        if device >= self.workers.len() {
            return Err(GpuWarmupProfileError::Measurement(
                "warmup production selected device is unavailable".into(),
            ));
        }
        let preimage_session_prepared =
            if fused_operation == Some(FusedWarmupOperation::PreimageBatch) {
                Some(
                    self.preimage_session_prepared(
                        &request.signature,
                        device,
                        &descriptor,
                        &representative,
                        request.cache_state == GpuWarmupCacheState::Warm,
                    )
                    .map_err(Self::classify_measurement_error)?,
                )
            } else {
                None
            };
        // Preimage warmup must use the same fixed retry policy as production,
        // and the native allocator envelope is part of the profile admission
        // contract. Query it automatically for every candidate using the
        // session-owned trapdoor/target prepared above.
        let mut resolved_cache_identity = None;
        if fused_operation == Some(FusedWarmupOperation::PreimageBatch) {
            let max_attempts = match request.retry_cap {
                Some(cap) => cap,
                None => mxx_primitives::env::gpu_preimage_max_tile_attempts()
                    .map_err(GpuWarmupProfileError::Measurement)?,
            };
            representative.retry_cap = Some(max_attempts);
            let evidence = match panic::catch_unwind(AssertUnwindSafe(|| {
                self.preimage_allocation_evidence_for_request(request, max_attempts)
            })) {
                Ok(result) => result.map_err(Self::classify_measurement_error)?,
                Err(payload) => match payload.downcast::<GpuOutOfMemory>() {
                    Ok(error) => return Err(GpuWarmupProfileError::OutOfMemory(error.to_string())),
                    Err(payload) => panic::resume_unwind(payload),
                },
            };
            let native_identity = Self::opaque_preimage_identity(&evidence);
            if request.cache_identity.is_some_and(|requested| requested != native_identity) {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "native preimage cache identity does not match the request".into(),
                ));
            }
            resolved_cache_identity = Some(native_identity);
            self.register_warmup_preimage_evidence(request.clone(), evidence, max_attempts)
                .map_err(Self::classify_measurement_error)?;
        }
        let resident_before =
            if profile_domain.measurement_kind() == WarmupMeasurementKind::GpuMeasured {
                Some(
                    Self::device_memory(self.workers[device].device_id)
                        .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?
                        .0
                        .resident_bytes,
                )
            } else {
                None
            };
        // Keep width selection and preparation inside the same typed panic
        // boundary as execution.  CUDA allocators may reserve scratch while
        // selecting/preparing an operation, so an OOM there is just as much
        // an infeasible candidate as an OOM from the kernel itself.
        let measurement = match panic::catch_unwind(AssertUnwindSafe(|| {
            if range_capable || capability == ColumnCapability::SingleDevice {
                self.workers[device].backend.set_column_widths_for_operation(
                    request.signature.operation,
                    GpuColumnWidths { gpu0: request.tile_width.max(1), nonzero: None },
                );
                self.workers[device]
                    .backend
                    .select_operation(request.signature.operation)
                    .map_err(|error| {
                        let message = error.to_string();
                        if message.to_ascii_lowercase().contains("out of memory") ||
                            message.to_ascii_lowercase().contains("cuda_error_out_of_memory")
                        {
                            GpuMeasurementError::out_of_memory(message)
                        } else {
                            GpuMeasurementError(message)
                        }
                    })?;
            }
            if fused_operation == Some(FusedWarmupOperation::PreimageBatch) &&
                request.cache_state == GpuWarmupCacheState::Cold &&
                request.timing_scope == GpuWarmupTimingScope::Setup
            {
                let prepared = preimage_session_prepared.as_ref().ok_or_else(|| {
                    GpuMeasurementError("cold preimage setup has no prepared session".into())
                })?;
                let (_, _, sigma, ..) = prepared.preimage_trapdoor.as_ref().ok_or_else(|| {
                    GpuMeasurementError("cold preimage setup has no trapdoor metadata".into())
                })?;
                let matrix_type = representative
                    .concrete_output_types
                    .iter()
                    .find_map(|wire_type| wire_type.matrix_type())
                    .ok_or_else(|| {
                        GpuMeasurementError("cold preimage setup has no output matrix type".into())
                    })?;
                Self::measure_preimage_cache_setup(
                    &mut self.workers[device],
                    prepared,
                    matrix_type,
                    *sigma,
                )
            } else {
                if request.timing_scope == GpuWarmupTimingScope::Transfer {
                    Self::measure_transfer_representative(
                        &mut self.workers[device],
                        &self.harness,
                        &descriptor.scope,
                        descriptor.id,
                        &descriptor.bindings,
                        &representative,
                    )
                } else {
                    match fused_operation {
                        Some(fused) => Self::measure_fused_representative(
                            &mut self.workers[device],
                            &self.harness,
                            &descriptor.scope,
                            descriptor.id,
                            &descriptor.bindings,
                            &representative,
                            fused,
                            request.binding_port.unwrap_or(0),
                            preimage_session_prepared.as_ref(),
                        ),
                        None => Self::measure_representative(
                            &mut self.workers[device],
                            &self.harness,
                            &descriptor.scope,
                            descriptor.id,
                            &descriptor.bindings,
                            &representative,
                            None,
                        ),
                    }
                }
            }
        })) {
            Ok(Ok(measurement)) => measurement,
            Ok(Err(error)) => {
                if error.is_out_of_memory() {
                    return Err(GpuWarmupProfileError::OutOfMemory(error.to_string()));
                }
                return Err(GpuWarmupProfileError::Measurement(error.to_string()));
            }
            Err(payload) => match payload.downcast::<GpuOutOfMemory>() {
                Ok(error) => return Err(GpuWarmupProfileError::OutOfMemory(error.to_string())),
                Err(payload) => panic::resume_unwind(payload),
            },
        };
        let workspace_bytes =
            if profile_domain.measurement_kind() == WarmupMeasurementKind::HostMeasured {
                0
            } else {
                measurement.measured_wave_workspace_bytes
            };
        let resident_delta = if let Some(before) = resident_before {
            let after = Self::device_memory(self.workers[device].device_id)
                .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?
                .0
                .resident_bytes;
            let delta = i64::try_from(after).unwrap_or(i64::MAX) -
                i64::try_from(before).unwrap_or(i64::MAX);
            let mut devices = BTreeMap::new();
            devices.insert(profile_key.device.clone(), delta);
            GpuWarmupResidencyDelta { affected_devices: devices, ..Default::default() }
        } else {
            GpuWarmupResidencyDelta::default()
        };
        if !measurement.cumulative_wave_seconds.is_finite() ||
            measurement.cumulative_wave_seconds <= 0.0
        {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "warmup production timing must be finite and positive".into(),
            ));
        }
        let memory = if let Some(memory) =
            self.preimage_memory.get(&(profile_key.clone(), request.tile_width)).cloned()
        {
            memory
        } else if profile_domain.measurement_kind() == WarmupMeasurementKind::GpuMeasured {
            // Recreate the un-timed representative solely to issue the native
            // owner-size query. The query covers ordinary, fused, constant,
            // lift, transfer and range paths alike; preimage has already
            // installed its sampler-specific exact envelope above.
            let query_node = MeasurementNode {
                scope: &descriptor.scope,
                id: descriptor.id,
                kind: &representative.kind,
                arguments: &[],
                argument_kinds: &[],
                argument_types: &[],
                output_types: &[],
                concrete_argument_types: representative.concrete_argument_types.clone(),
                concrete_output_types: representative.concrete_output_types.clone(),
            };
            let query_prepared = Self::prepare(
                &mut self.workers[device].backend,
                &query_node,
                &descriptor.bindings,
                None,
            )
            .map_err(Self::classify_measurement_error)?;
            let query_prepared = Self::place_prepared_sources(
                &mut self.workers[device],
                &representative,
                query_prepared,
            )
            .map_err(Self::classify_measurement_error)?;
            let worker_device_ids = self
                .workers
                .iter()
                .flat_map(|worker| worker.backend.physical_device_ids())
                .collect::<Vec<_>>();
            Self::warmup_allocation_observations(
                &worker_device_ids,
                &mut self.workers[device],
                profile_domain,
                &representative,
                &query_prepared,
                request,
                &descriptor.bindings,
            )
            .map_err(Self::classify_measurement_error)?
        } else if Self::is_host_backend_boundary(&descriptor.kind) {
            // The independently measured transfer owns a real native matrix
            // and host payload. Never relabel the sampled allocator peak as
            // an exact allocation query.
            let matrix = representative
                .concrete_argument_types
                .iter()
                .chain(&representative.concrete_output_types)
                .find_map(|ty| ty.matrix_type())
                .ok_or_else(|| {
                    GpuWarmupProfileError::Measurement("boundary has no native matrix type".into())
                })?;
            let envelope = self.workers[device]
                .backend
                .generated_range_allocation_envelope(
                    profile_domain,
                    matrix,
                    0..matrix.columns,
                    GpuAllocationComponents {
                        evidence: Some(GpuAllocationEvidenceKind::ExactQuery),
                        ..Default::default()
                    },
                )
                .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
            let bytes = envelope.total_device_bytes().ok_or_else(|| {
                GpuWarmupProfileError::Measurement("boundary allocation overflows".into())
            })?;
            let mut affected_devices = BTreeMap::new();
            affected_devices.insert(profile_key.device.clone(), bytes as u64);
            GpuWarmupMemoryObservations {
                affected_devices,
                evidence: MemoryEvidenceKind::ExactQuery,
                ..GpuWarmupMemoryObservations::default()
            }
        } else {
            GpuWarmupMemoryObservations {
                evidence: MemoryEvidenceKind::ExactQuery,
                ..GpuWarmupMemoryObservations::default()
            }
        };
        let memory = self
            .merge_production_memory(memory, self.workers[device].last_production_job.as_ref())
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
        // The fleet response is the authoritative physical execution result.
        // Do not let the logical request's provisional resolver become the
        // cache identity when production selected a peer/host-staged route or
        // a different source shard set.
        let resolved_route_descriptor = if request.timing_scope == GpuWarmupTimingScope::Setup &&
            fused_operation == Some(FusedWarmupOperation::PreimageBatch)
        {
            // Cache setup allocates on the selected native owner but performs
            // no operand transport. Its route is not the subsequent sampling
            // job's route, and no sampling dispatch is run just to obtain it.
            Some(request.route_descriptor)
        } else if request.timing_scope == GpuWarmupTimingScope::Transfer ||
            profile_domain.measurement_kind() == WarmupMeasurementKind::GpuMeasured
        {
            self.workers[device]
                .last_production_job
                .as_ref()
                .and_then(|observation| {
                    authoritative_production_route(request.route_descriptor, &observation.routes)
                })
                // Generated GPU primitives have no source owner to report in
                // the inclusive job response.  Their destination-local route
                // is still the real production route (and is validated by the
                // same resolver), so retain that value instead of rejecting a
                // valid constant/sampler/lift dispatch as unprofileable.
                .or_else(|| {
                    let has_matrix_source = descriptor
                        .concrete_argument_types
                        .iter()
                        .any(|wire_type| matrix_leaf_type(wire_type).is_some());
                    (request.timing_scope != GpuWarmupTimingScope::Transfer &&
                        !has_matrix_source &&
                        request.route_descriptor.validate())
                    .then_some(request.route_descriptor)
                })
        } else {
            None
        };
        if (request.timing_scope == GpuWarmupTimingScope::Transfer ||
            profile_domain.measurement_kind() == WarmupMeasurementKind::GpuMeasured) &&
            resolved_route_descriptor.is_none()
        {
            return Err(GpuWarmupProfileError::Measurement(
                "inclusive production measurement returned no valid physical route".into(),
            ));
        }
        let mut profile = GpuWarmupProfile::measured_with_observation(
            measurement.cumulative_wave_seconds,
            workspace_bytes,
            profile_domain.measurement_kind(),
            memory.clone(),
            resident_delta.clone(),
            self.harness.measured_iterations,
            0.0,
            GpuWarmupProvenance::ProductionEquivalent,
            profile_key.cache_state,
            profile_key.timing_scope,
        )?;
        profile.resolved_route_descriptor = resolved_route_descriptor;
        if let Some((attempts, width)) =
            self.preimage_profile_metadata.get(&(profile_key.clone(), request.tile_width)).copied()
        {
            profile.preimage_max_attempts = Some(attempts);
            profile.preimage_certified_tile_width = Some(width);
        }
        if let Some(footprint) =
            self.preimage_footprints.get(&(profile_key, request.tile_width)).cloned()
        {
            profile.preimage_footprint = Some(footprint);
        }
        profile.resolved_cache_identity = resolved_cache_identity;
        self.record_warmup_dispatch(request, profile_domain, fused_operation);
        self.warmup_measurement_provenances.push(profile.provenance);
        Ok(profile)
    }

    /// Read the widths frozen by a runtime plan. Plan-aware estimation never
    /// derives a new width from current allocator residency.
    pub fn fixed_plan_widths(
        plan: &FrozenGpuPlan,
        site: GpuExecutionSiteKey,
    ) -> Result<GpuColumnWidths, GpuMeasurementError> {
        GpuCalibrationProfile::fixed_widths_for_plan(plan, site)
            .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    /// Heterogeneous-fleet form of [`Self::fixed_plan_widths`].
    pub fn fixed_plan_columns(
        plan: &FrozenGpuPlan,
        site: GpuExecutionSiteKey,
    ) -> Result<Vec<usize>, GpuMeasurementError> {
        GpuCalibrationProfile::fixed_columns_for_plan(plan, site)
            .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    /// Compute one fixed-plan wave using the same schedule/batch-wave time
    /// model consumed by runtime warmup. No width or owner is re-selected.
    pub fn fixed_plan_wave_seconds(
        plan: &FrozenGpuPlan,
        site: GpuExecutionSiteKey,
        model: &[GpuTimeModel],
    ) -> Result<f64, GpuMeasurementError> {
        let node = plan
            .node_choice(site)
            .ok_or_else(|| GpuMeasurementError(format!("missing fixed plan site {site:?}")))?;
        let layout_id = *node
            .output_layouts
            .first()
            .ok_or_else(|| GpuMeasurementError("fixed plan site has no output layout".into()))?;
        let layout = plan
            .layout(layout_id)
            .ok_or_else(|| GpuMeasurementError(format!("missing fixed plan layout {layout_id}")))?;
        let instances = node
            .loop_site
            .map(|key| {
                plan.loop_choice(key)
                    .map(|choice| choice.wave_instances.min(choice.loop_count))
                    .ok_or_else(|| GpuMeasurementError("missing loop choice".into()))
            })
            .transpose()?
            .unwrap_or(1);
        if matches!(
            node.column_capability,
            ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
        ) {
            let columns = if node.column_capability == ColumnCapability::SingleDevice {
                layout.columns
            } else {
                0
            };
            return gpu_non_column_batch_wave_time(instances, columns, 0, model)
                .map_err(|error| GpuMeasurementError(error.to_string()));
        }
        let schedules = (0..instances)
            .map(|instance| {
                node.output_layouts
                    .iter()
                    .map(|id| {
                        let layout = plan.layout(*id).ok_or_else(|| {
                            GpuMeasurementError("missing output port layout".into())
                        })?;
                        layout
                            .schedule(&node.columns_per_job, instance)
                            .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        mxx_runtime::gpu_warmup::gpu_multi_output_batch_wave_time(&schedules, model)
            .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    /// Measures every collected shape as one fleet operation. Column-separable nodes use all
    /// workers for the same primitive instead of assigning unrelated primitives to different GPUs.
    pub fn measure_collected(&mut self) -> Result<(), GpuMeasurementError> {
        self.collecting = false;
        let mut requests = std::mem::take(&mut self.pending).into_values().collect::<Vec<_>>();
        requests.sort_by(|left, right| {
            right
                .preimage_sample
                .cmp(&left.preimage_sample)
                .then_with(|| right.representative_bytes().cmp(&left.representative_bytes()))
        });
        let measurement_count = requests.len();
        info!(
            measurement_workers = self.workers.len(),
            measurement_count, "measuring collected GPU node shapes in parallel"
        );
        for request in requests {
            debug!(
                scope = ?request.scope,
                node = request.id.0,
                kind = ?request.kind,
                concrete_argument_types = ?request.concrete_argument_types,
                concrete_output_types = ?request.concrete_output_types,
                measurement_key = ?request.key,
                "starting GPU node measurement"
            );
            let measurement = self.measure_fleet_request(&request)?;
            self.measurements.insert(request.key, measurement);
        }
        Ok(())
    }

    #[cfg(test)]
    fn column_separable(kind: &NodeKind) -> bool {
        mxx_runtime::gpu_calibration::gpu_operation_is_column_separable(kind)
    }

    fn matrix_columns(wire_type: &ConcreteWireType) -> Option<usize> {
        match wire_type {
            ConcreteWireType::Matrix(matrix) |
            ConcreteWireType::SmallMatrix { matrix, .. } |
            ConcreteWireType::Preimage { matrix, .. } |
            ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix.columns),
            ConcreteWireType::IndexedFamily { element, .. } => Self::matrix_columns(element),
            _ => None,
        }
    }

    fn matrix_multiply_scaled_argument(concrete_argument_types: &[ConcreteWireType]) -> usize {
        let left = concrete_argument_types
            .first()
            .and_then(matrix_leaf_type)
            .expect("validated matrix multiplication must have a matrix LHS");
        let right = concrete_argument_types
            .get(1)
            .and_then(matrix_leaf_type)
            .expect("validated matrix multiplication must have a matrix RHS");
        if gpu_matrix_multiply_scales_left(left.rows, left.columns, right.rows, right.columns) {
            0
        } else {
            1
        }
    }

    fn argument_is_fixed_for(
        kind: &NodeKind,
        concrete_argument_types: &[ConcreteWireType],
        index: usize,
    ) -> bool {
        match kind {
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => {
                index != Self::matrix_multiply_scaled_argument(concrete_argument_types)
            }
            NodeKind::MatrixMulSmallRhs => index == 0,
            NodeKind::MatrixMulAccumulate { coefficients, .. } => {
                if index >= 2 * coefficients.len() {
                    return false;
                }
                let product = index / 2;
                let left = matrix_leaf_type(&concrete_argument_types[2 * product]).unwrap();
                let right = matrix_leaf_type(&concrete_argument_types[2 * product + 1]).unwrap();
                let scalable = if gpu_matrix_multiply_scales_left(
                    left.rows,
                    left.columns,
                    right.rows,
                    right.columns,
                ) {
                    2 * product
                } else {
                    2 * product + 1
                };
                index != scalable
            }
            NodeKind::PreimageSample { .. } => {
                // A GadgetTrapdoor-backed preimage exposes only its target to
                // the effective fixed decomposition descriptor.  The sampled
                // preimage descriptor retains public/trapdoor metadata in the
                // first two slots and keeps only the target dynamic.
                concrete_argument_types.len() >= 3 && index < 2
            }
            NodeKind::Concat { axis: ConcatAxis::Diagonal } => true,
            _ => false,
        }
    }

    #[cfg(test)]
    fn argument_is_fixed(node: &MeasurementNode<'_>, index: usize) -> bool {
        Self::argument_is_fixed_for(node.kind, &node.concrete_argument_types, index)
    }

    fn fixed_arguments(kind: &NodeKind, concrete_argument_types: &[ConcreteWireType]) -> Vec<bool> {
        (0..concrete_argument_types.len())
            .map(|index| Self::argument_is_fixed_for(kind, concrete_argument_types, index))
            .collect()
    }

    fn verify_concat_representative(
        node: &MeasurementNode<'_>,
        axis: ConcatAxis,
        output_range: Option<&IndexRange>,
    ) -> Result<(), GpuMeasurementError> {
        let output =
            node.concrete_output_types.iter().find_map(matrix_leaf_type).ok_or_else(|| {
                GpuMeasurementError(format!("node {:?} concat output is not a matrix", node.id))
            })?;
        let inputs = node
            .concrete_argument_types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                matrix_leaf_type(ty).ok_or_else(|| {
                    GpuMeasurementError(format!(
                        "node {:?} concat argument {index} is not a matrix",
                        node.id
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if inputs.is_empty() {
            return Err(GpuMeasurementError(format!(
                "node {:?} concat representative has no retained inputs",
                node.id
            )));
        }
        let checked_sum = |dimension: fn(&ConcreteMatrixType) -> usize| {
            inputs
                .iter()
                .try_fold(0usize, |sum, input| sum.checked_add(dimension(input)))
                .ok_or_else(|| {
                    GpuMeasurementError(format!("node {:?} concat dimensions overflow", node.id))
                })
        };
        match axis {
            ConcatAxis::Columns => {
                if inputs.iter().any(|input| input.rows != output.rows || input.columns == 0) {
                    return Err(GpuMeasurementError(format!(
                        "node {:?} column concat retained piece geometry is invalid",
                        node.id
                    )));
                }
                let columns = checked_sum(|input| input.columns)?;
                if columns != output.columns {
                    return Err(GpuMeasurementError(format!(
                        "node {:?} column concat retained widths sum to {columns}, expected {}",
                        node.id, output.columns
                    )));
                }
                if let Some(range) = output_range {
                    let width = range.end.checked_sub(range.start).ok_or_else(|| {
                        GpuMeasurementError(format!(
                            "node {:?} column concat representative range is reversed",
                            node.id
                        ))
                    })?;
                    // The concrete output keeps the validated full logical
                    // width while the representative may cover only one
                    // production range.  A partial range is therefore
                    // valid as long as it lies within that output; requiring
                    // equality here rejects precisely the ranged concat
                    // representatives used by fleet warmup.
                    if width == 0 || range.end > output.columns {
                        return Err(GpuMeasurementError(format!(
                            "node {:?} column concat range {range:?} is outside output width {}",
                            node.id, output.columns
                        )));
                    }
                }
            }
            ConcatAxis::Rows => {
                if inputs.iter().any(|input| input.columns != output.columns || input.rows == 0) {
                    return Err(GpuMeasurementError(format!(
                        "node {:?} row concat representative geometry is invalid",
                        node.id
                    )));
                }
                if let Some(range) = output_range {
                    let width = range.end.checked_sub(range.start).ok_or_else(|| {
                        GpuMeasurementError(format!(
                            "node {:?} row concat representative range is reversed",
                            node.id
                        ))
                    })?;
                    // The concrete representative is local to the selected
                    // device.  Its range coordinates remain global, so only
                    // the width is meaningful here.
                    if width == 0 || range.end > output.columns {
                        return Err(GpuMeasurementError(format!(
                            "node {:?} row concat range {range:?} is outside output width {}",
                            node.id, output.columns
                        )));
                    }
                }
                let rows = checked_sum(|input| input.rows)?;
                if rows != output.rows {
                    return Err(GpuMeasurementError(format!(
                        "node {:?} row concat retained heights sum to {rows}, expected {}",
                        node.id, output.rows
                    )));
                }
            }
            ConcatAxis::Diagonal => {
                let rows = checked_sum(|input| input.rows)?;
                let columns = checked_sum(|input| input.columns)?;
                if rows != output.rows || columns != output.columns {
                    return Err(GpuMeasurementError(format!(
                        "node {:?} diagonal concat block geometry is invalid",
                        node.id
                    )));
                }
                if let Some(range) = output_range &&
                    (range.start >= range.end || range.end > output.columns)
                {
                    return Err(GpuMeasurementError(format!(
                        "node {:?} diagonal concat range is outside its output",
                        node.id
                    )));
                }
            }
        }
        Ok(())
    }

    fn request_columns(request: &PendingMeasurement) -> Option<usize> {
        let capability = column_capability(&request.kind, &request.concrete_argument_types);
        if matches!(capability, ColumnCapability::HostOrControl | ColumnCapability::SingleDevice) {
            return None;
        }
        let columns = request
            .concrete_output_types
            .iter()
            .find_map(Self::matrix_columns)
            .or_else(|| request.concrete_argument_types.iter().find_map(Self::matrix_columns))
            .filter(|columns| *columns > 0)?;
        let output = request.concrete_output_types.iter().find_map(ConcreteWireType::matrix_type);
        match (&request.kind, output) {
            (NodeKind::ConstantMatrix { value: ConstantMatrix::Identity, .. }, Some(matrix))
                if matrix.rows != matrix.columns =>
            {
                None
            }
            (
                NodeKind::ConstantMatrix { value: ConstantMatrix::UnitRow { .. }, .. },
                Some(matrix),
            ) if matrix.rows != 1 => None,
            (
                NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { .. }, .. },
                Some(matrix),
            ) if matrix.rows == 0 || !matrix.columns.is_multiple_of(matrix.rows) => None,
            // These constants are single-device production calls.  They must
            // be measured as the complete operation, even when the planner
            // is probing a width candidate for neighbouring column-separable
            // nodes.
            (
                NodeKind::ConstantMatrix {
                    value:
                        ConstantMatrix::PowerOfBase { .. } |
                        ConstantMatrix::Rotation { .. } |
                        ConstantMatrix::Polynomial { .. },
                    ..
                },
                _,
            ) => None,
            (NodeKind::GadgetTrapdoor { .. }, Some(matrix))
                if matrix.rows == 0 || !matrix.columns.is_multiple_of(matrix.rows) =>
            {
                None
            }
            _ => Some(columns),
        }
    }

    /// Return whether a representative is expected to execute one production
    /// output-column range.  Keep this in the estimator beside
    /// `representative_at`: the runtime policy is the authority for the
    /// classification, while this helper makes it impossible for a newly
    /// ranged primitive to silently fall back to a full-operation pilot.
    fn is_range_capable(request: &PendingMeasurement) -> bool {
        !matches!(
            column_capability(&request.kind, &request.concrete_argument_types),
            ColumnCapability::HostOrControl |
                ColumnCapability::SingleDevice |
                ColumnCapability::Unsupported
        )
    }

    fn calibration_representative(request: &PendingMeasurement) -> RepresentativeMeasurement {
        if let NodeKind::ConstantMatrix { value: ConstantMatrix::UnitRow { index }, .. } =
            &request.kind
        {
            if let Some(index) =
                index.evaluate(&request.bindings).ok().and_then(|value| value.to_usize())
            {
                return Self::representative_at(request, index, 1);
            }
        }
        if matches!(request.kind, NodeKind::Concat { axis: ConcatAxis::Columns }) {
            let input_columns = request
                .concrete_argument_types
                .iter()
                .filter_map(Self::matrix_columns)
                .collect::<Vec<_>>();
            let total_columns = input_columns.iter().sum::<usize>();
            if total_columns > 1 {
                let mut boundary = 0usize;
                let first_boundary = input_columns.into_iter().find_map(|columns| {
                    boundary = boundary.checked_add(columns)?;
                    (boundary > 0 && boundary < total_columns).then_some(boundary)
                });
                if let Some(boundary) = first_boundary {
                    // A one-column prefix sees only one concat piece.  Straddle a real input
                    // boundary so calibration includes the location-dependent piece/launch cost
                    // without materializing the complete concatenation.
                    return Self::representative_at(request, boundary - 1, 2);
                }
            }
        }
        // Calibration must execute a genuinely small production shape.  The
        // normal ranged representative intentionally retains full validated
        // wire types and relies on the range mapper, but a pilot has no
        // mapper-owned source shards yet.  Shape the pilot to one column and
        // retain fixed-operand ownership from the original request.
        let node = MeasurementNode {
            scope: &request.scope,
            id: request.id,
            kind: &request.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: request.concrete_argument_types.clone(),
            concrete_output_types: request.concrete_output_types.clone(),
        };
        let (kind, concrete_argument_types, concrete_output_types, _, _) =
            Self::representative_node(&node, 1);
        RepresentativeMeasurement {
            source_layouts: Vec::new(),
            fixed_metadata: None,
            retry_cap: None,
            inputs: Default::default(),
            kind,
            concrete_argument_types,
            concrete_output_types,
            fixed_arguments: Self::fixed_arguments(&request.kind, &request.concrete_argument_types),
            output_range: Self::is_range_capable(request)
                .then_some(IndexRange { start: 0, end: 1 }),
        }
    }

    fn representative_at(
        request: &PendingMeasurement,
        global_column_start: usize,
        columns: usize,
    ) -> RepresentativeMeasurement {
        // A representative is a production request at a particular global
        // range, not a synthetic local graph.  Keep the validated global
        // wire types and node parameters intact: the fleet mapper is the
        // single owner of converting that range into physical local pieces
        // (including tensor boundaries, concat fragments, and peer/staged
        // copies).  Mixing local types with global coordinates makes non-zero
        // ranges silently address the wrong input on non-zero devices.
        let global_column_end =
            global_column_start.checked_add(columns).expect("representative column range overflow");
        RepresentativeMeasurement {
            source_layouts: Vec::new(),
            fixed_metadata: None,
            retry_cap: None,
            inputs: Default::default(),
            kind: request.kind.clone(),
            concrete_argument_types: request.concrete_argument_types.clone(),
            concrete_output_types: request.concrete_output_types.clone(),
            fixed_arguments: Self::fixed_arguments(&request.kind, &request.concrete_argument_types),
            output_range: (Self::is_range_capable(request) ||
                matches!(request.kind, NodeKind::GadgetTrapdoor { .. }))
            .then_some(IndexRange { start: global_column_start, end: global_column_end }),
        }
    }

    fn calibration_operation_key(
        request: &PendingMeasurement,
    ) -> Result<[u8; 32], GpuMeasurementError> {
        gpu_calibration_operation_identity(
            &request.kind,
            &request.concrete_argument_types,
            &request.concrete_output_types,
            &request.bindings,
        )
        .map_err(GpuMeasurementError)
    }

    fn device_memory(device_id: i32) -> Result<(GpuDeviceMemory, usize), GpuMeasurementError> {
        let usage = gpu_device_memory_usage(device_id).map_err(GpuMeasurementError)?;
        Ok((
            GpuDeviceMemory {
                total_bytes: u64::try_from(usage.total)
                    .map_err(|_| GpuMeasurementError("GPU total memory exceeds u64".to_owned()))?,
                resident_bytes: u64::try_from(usage.resident).map_err(|_| {
                    GpuMeasurementError("GPU resident memory exceeds u64".to_owned())
                })?,
            },
            usage.live_contexts,
        ))
    }

    /// Translate the inclusive fleet job envelope into the provider's public
    /// memory evidence.  The production job already resolved physical owners
    /// and staging paths; preserving that map here avoids collapsing a peer
    /// source allocation onto the destination worker.
    fn production_memory_observation(
        &self,
        observation: &ProductionJobObservation,
    ) -> Result<GpuWarmupMemoryObservations, GpuMeasurementError> {
        let mut affected_devices = BTreeMap::new();
        let mut host_bytes = 0u64;
        let mut pinned_host_bytes = 0u64;
        for resource in &observation.resources {
            let identity = gpu_device_runtime_identity(resource.physical_device)
                .map_err(GpuMeasurementError)?;
            let owner = GpuWarmupDeviceIdentity::new(
                self.physical_device_index(resource.physical_device)
                    .unwrap_or(resource.device_index),
                format!(
                    "uuid={}-{}-sm{}.{}-mem{}",
                    identity.uuid,
                    identity.name,
                    identity.compute_major,
                    identity.compute_minor,
                    identity.total_global_memory
                ),
                format!(
                    "{}:driver{}:runtime{}",
                    identity.native_kernel_revision,
                    identity.driver_version,
                    identity.runtime_version,
                ),
            )
            .with_context_generation(identity.context_generation);
            let bytes = u64::try_from(resource.device_bytes)
                .map_err(|_| GpuMeasurementError("production device bytes exceed u64".into()))?;
            affected_devices
                .entry(owner)
                .and_modify(|total: &mut u64| *total = total.saturating_add(bytes))
                .or_insert(bytes);
            host_bytes =
                host_bytes.saturating_add(u64::try_from(resource.host_bytes).map_err(|_| {
                    GpuMeasurementError("production host staging exceeds u64".into())
                })?);
            pinned_host_bytes = pinned_host_bytes.saturating_add(
                u64::try_from(resource.pinned_host_bytes).map_err(|_| {
                    GpuMeasurementError("production pinned staging exceeds u64".into())
                })?,
            );
        }
        Ok(GpuWarmupMemoryObservations {
            affected_devices,
            host_bytes,
            pinned_host_bytes,
            // These are native allocation-query sizes and explicit transport
            // buffers, not allocator peaks. Summing them conservatively
            // bounds simultaneously retained transport owners.
            evidence: MemoryEvidenceKind::CertifiedEnvelope,
        })
    }

    fn merge_production_memory(
        &self,
        measured: GpuWarmupMemoryObservations,
        observation: Option<&ProductionJobObservation>,
    ) -> Result<GpuWarmupMemoryObservations, GpuMeasurementError> {
        let Some(observation) = observation else { return Ok(measured) };
        let production = self.production_memory_observation(observation)?;
        let mut affected_devices = measured.affected_devices;
        for (device, bytes) in production.affected_devices {
            affected_devices
                .entry(device)
                .and_modify(|current| *current = current.saturating_add(bytes))
                .or_insert(bytes);
        }
        Ok(GpuWarmupMemoryObservations {
            affected_devices,
            // Kernel and transport owners may coexist. Without a proven
            // lifetime-disjoint decomposition, their sum is the safe bound;
            // a maximum could omit a source replica or staging allocation.
            host_bytes: measured.host_bytes.saturating_add(production.host_bytes),
            pinned_host_bytes: measured
                .pinned_host_bytes
                .saturating_add(production.pinned_host_bytes),
            evidence: if measured.evidence.is_hard_admission() &&
                production.evidence.is_hard_admission()
            {
                MemoryEvidenceKind::CertifiedEnvelope
            } else {
                MemoryEvidenceKind::Unspecified
            },
        })
    }

    fn measure_fleet_request(
        &mut self,
        request: &PendingMeasurement,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        let Some(total_columns) = Self::request_columns(request) else {
            if matches!(
                request.kind,
                NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_)
            ) {
                let child = self.host_control_operations.get(&(request.scope.clone(), request.id));
                let (seconds, _, _) = Self::measure_host_control_repeated(
                    &self.harness,
                    &request.scope,
                    request.id,
                    &request.kind,
                    &request.bindings,
                    child,
                )?;
                return Ok(NodeMeasurement {
                    work_seconds: seconds,
                    latency_seconds: seconds,
                    cumulative_wave_seconds: seconds,
                    independent_wave_count: 1,
                    measured_wave_workspace_bytes: 0,
                    workspace_bytes: 0,
                });
            }
            let representative = RepresentativeMeasurement {
                source_layouts: Vec::new(),
                fixed_metadata: None,
                retry_cap: None,
                inputs: Default::default(),
                kind: request.kind.clone(),
                concrete_argument_types: request.concrete_argument_types.clone(),
                concrete_output_types: request.concrete_output_types.clone(),
                fixed_arguments: Self::fixed_arguments(
                    &request.kind,
                    &request.concrete_argument_types,
                ),
                output_range: None,
            };
            return Self::measure_representative(
                &mut self.workers[0],
                &self.harness,
                &request.scope,
                request.id,
                &request.bindings,
                &representative,
                None,
            );
        };

        let initial_device_states = self
            .workers
            .iter()
            .map(|worker| Self::device_memory(worker.device_id))
            .collect::<Result<Vec<_>, _>>()?;
        let vram_percent = self.vram_percent;
        let representative_device =
            mxx_primitives::poly::dcrt::gpu::gpu_device_identity(self.workers[0].device_id)
                .map_err(GpuMeasurementError)?;
        let operation = Self::calibration_operation_key(request)?;
        let calibration_key = GpuCalibrationKey::new(
            operation.as_slice(),
            gpu_calibration_environment(&representative_device, self.workers.len(), vram_percent),
        );
        if let Some((worker, (_, live_contexts))) =
            self.workers.iter().zip(&initial_device_states).find(|(worker, (_, contexts))| {
                *contexts != worker.backend.owned_context_count(worker.device_id)
            })
        {
            return Err(GpuMeasurementError(format!(
                "GPU {} has {live_contexts} live mxx contexts; fleet calibration requires exclusive CUDA mempool measurement",
                worker.device_id
            )));
        }
        let profile = if let Some(profile) = self.calibration_registry.get(&calibration_key) {
            profile
        } else {
            let pilot = Self::calibration_representative(request);
            let pilot_columns = pilot.measured_columns().ok_or_else(|| {
                GpuMeasurementError("GPU calibration pilot has no matrix columns".to_owned())
            })?;
            let calibrated_roles = self.workers.len().min(2);
            let barrier = Arc::new(Barrier::new(calibrated_roles));
            let pilot_measurements = self.workers[..calibrated_roles]
                .par_iter_mut()
                .map(|worker| {
                    Self::calibrate_representative(
                        worker,
                        &request.scope,
                        request.id,
                        &request.bindings,
                        &pilot,
                        operation,
                        pilot_columns,
                        Some(barrier.as_ref()),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let calibration = |measurement: &NodeMeasurement| {
                GpuDeviceCalibration::from_pilot(pilot_columns, measurement.workspace_bytes)
                    .map_err(|error| GpuMeasurementError(error.to_string()))
            };
            let profile = GpuCalibrationProfile {
                gpu0: calibration(&pilot_measurements[0])?,
                nonzero: pilot_measurements.get(1).map(calibration).transpose()?,
            };
            self.calibration_registry
                .insert(calibration_key.clone(), profile)
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
            let profile = self
                .calibration_registry
                .get(&calibration_key)
                .expect("new GPU calibration profile must be present");
            profile
        };

        // Derive role widths from the production baseline: every column-independent operand is
        // staged and complete, while column-scaled inputs, outputs, and workspaces are absent.
        // Keep these owners live until after the snapshots so async release cannot understate the
        // planned resident set.  Fleet measurement prepares its own operands after width selection.
        let baseline_representative = Self::calibration_representative(request);
        let fixed_inputs = self
            .workers
            .par_iter_mut()
            .map(|worker| {
                worker.backend.set_column_widths_for_operation(
                    operation,
                    GpuColumnWidths { gpu0: 1, nonzero: None },
                );
                worker
                    .backend
                    .select_operation(operation)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let node = MeasurementNode {
                    scope: &request.scope,
                    id: request.id,
                    kind: &baseline_representative.kind,
                    arguments: &[],
                    argument_kinds: &[],
                    argument_types: &[],
                    output_types: &[],
                    concrete_argument_types: baseline_representative
                        .concrete_argument_types
                        .clone(),
                    concrete_output_types: baseline_representative.concrete_output_types.clone(),
                };
                let prepared = Self::prepare(
                    &mut worker.backend,
                    &node,
                    &request.bindings,
                    Some((&baseline_representative.fixed_arguments, true)),
                )?;
                prepared.finish();
                Ok(prepared)
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
        let device_memories = self
            .workers
            .par_iter_mut()
            .map(|worker| {
                // The pilot has been dropped; make its queued frees reusable before admitting
                // a full fleet wave against the pool's effective available memory.
                worker
                    .backend
                    .fence_released_memory()
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                Self::device_memory(worker.device_id)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let gpu0_memory = device_memories[0].0;
        let nonzero_memory = device_memories.get(1).map(|(memory, _)| *memory);
        let widths = profile
            .derive_widths(gpu0_memory, nonzero_memory, vram_percent)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let fleet_columns = widths
            .columns_per_wave(self.workers.len())
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let wave_count = total_columns.div_ceil(fleet_columns);
        let assigned_columns =
            gpu_capped_waterfill_columns(widths, self.workers.len(), total_columns)
                .map_err(|error| GpuMeasurementError(error.to_string()))?;

        for (index, worker) in self.workers.iter().enumerate() {
            let physical = gpu_memory_info(worker.device_id).map_err(GpuMeasurementError)?;
            let pool = gpu_default_mempool_usage(worker.device_id).map_err(GpuMeasurementError)?;
            let memory = device_memories[index].0;
            let calibration =
                if index == 0 { profile.gpu0 } else { profile.nonzero.unwrap_or(profile.gpu0) };
            info!(
                scope = ?request.scope, node = request.id.0, device_id = worker.device_id,
                physical_total_bytes = physical.total, physical_free_bytes = physical.free,
                pool_used_bytes = pool.used_current, pool_reserved_bytes = pool.reserved_current,
                effective_resident_bytes = memory.resident_bytes,
                budget_bytes = u128::from(memory.total_bytes) * u128::from(vram_percent) / 100,
                pilot_columns = calibration.pilot_columns(),
                pilot_peak_bytes = calibration.pilot_peak_bytes(),
                bytes_per_column = calibration.bytes_per_column(),
                assigned_columns = assigned_columns[index],
                "GPU fleet calibration memory snapshot"
            );
        }

        info!(
            scope = ?request.scope,
            node = request.id.0,
            gpu_count = self.workers.len(),
            gpu0_columns = widths.gpu0,
            nonzero_gpu_columns = widths.nonzero,
            fleet_wave_columns = fleet_columns,
            total_columns,
            wave_count,
            vram_percent,
            "derived GPU fleet column widths"
        );

        let mut global_column_start = 0;
        let representatives = assigned_columns
            .into_iter()
            .map(|columns| {
                let start = global_column_start;
                global_column_start += columns;
                (columns > 0).then(|| Self::representative_at(request, start, columns))
            })
            .collect();
        let (measurements, fleet_latency_seconds) = Self::measure_fleet_wave(
            &mut self.workers,
            &self.harness,
            &request.scope,
            request.id,
            &request.bindings,
            operation,
            representatives,
            fixed_inputs,
        )
        .map_err(|error| {
            GpuMeasurementError(format!(
                "calibrated GPU fleet width verification failed (gpu0={}, nonzero={:?}): {error}",
                widths.gpu0, widths.nonzero
            ))
        })?;
        for (worker, measurement) in self.workers.iter().zip(&measurements) {
            if let Some(measurement) = measurement {
                debug!(
                    device_id = worker.device_id,
                    kind = ?request.kind,
                    arguments = ?request.concrete_argument_types,
                    outputs = ?request.concrete_output_types,
                    measurement = ?measurement,
                    "GPU representative measurement complete"
                );
                info!(
                    device_id = worker.device_id,
                    workspace_bytes = measurement.workspace_bytes,
                    latency_seconds = measurement.latency_seconds,
                    "measured GPU fleet device wave"
                );
            }
        }
        let full_wave =
            aggregate_fleet_wave(measurements.into_iter().flatten(), fleet_latency_seconds);
        let measurement = extrapolate_fleet_waves(&full_wave, wave_count);
        info!(
            scope = ?request.scope, node = request.id.0, kind = ?request.kind,
            wave_count, fleet_wave_latency_seconds = full_wave.latency_seconds,
            work_seconds = measurement.work_seconds,
            cumulative_wave_seconds = measurement.cumulative_wave_seconds,
            fleet_wave_workspace_bytes = full_wave.workspace_bytes,
            "measured independent GPU fleet waves"
        );
        if request.preimage_sample {
            info!(
                scope = ?request.scope,
                node = request.id.0,
                work_seconds = measurement.work_seconds,
                dependency_latency_seconds = measurement.latency_seconds,
                cumulative_wave_seconds = measurement.cumulative_wave_seconds,
                measured_wave_workspace_bytes = measurement.measured_wave_workspace_bytes,
                ideal_concurrent_workspace_bytes = measurement.workspace_bytes,
                "estimated fleet-wide GPU preimage sampler from measured waves"
            );
        }
        Ok(measurement)
    }

    fn measurement_key(
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<[u8; 32], GpuMeasurementError> {
        #[derive(Serialize)]
        struct MeasurementCacheKey<'a> {
            operation: [u8; 32],
            concrete_argument_types: &'a [ConcreteWireType],
            concrete_output_types: &'a [ConcreteWireType],
        }

        let operation = gpu_calibration_operation_identity(
            node.kind,
            &node.concrete_argument_types,
            &node.concrete_output_types,
            bindings,
        )
        .map_err(GpuMeasurementError)?;
        encoding::hash_canonical(&MeasurementCacheKey {
            operation,
            concrete_argument_types: &node.concrete_argument_types,
            concrete_output_types: &node.concrete_output_types,
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn representative_node<'a>(
        node: &'a MeasurementNode<'a>,
        column_limit: usize,
    ) -> (NodeKind, Vec<ConcreteWireType>, Vec<ConcreteWireType>, f64, Option<usize>) {
        assert!(column_limit > 0, "GPU representative column count must be nonzero");
        let capped_columns = |matrix: &ConcreteMatrixType| {
            (matrix.columns > column_limit).then(|| {
                let representative_columns = column_limit;
                let full_waves = matrix.columns / representative_columns;
                let remainder_columns = matrix.columns % representative_columns;
                (
                    representative_columns,
                    full_waves as f64,
                    (remainder_columns > 0).then_some(remainder_columns),
                )
            })
        };
        let mut kind = node.kind.clone();
        let mut argument_types = node.concrete_argument_types.clone();
        let mut output_types = node.concrete_output_types.clone();
        let mut scale = 1.0;
        let mut remainder_columns = None;

        match &mut kind {
            NodeKind::ConstantMatrix { matrix_type, .. } => {
                let Some(output) = output_types.iter_mut().find_map(matrix_leaf_type_mut) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::GadgetTrapdoor { matrix_type, .. } |
            NodeKind::TrapdoorSample { matrix_type, .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } |
                    ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::UniformResidueSample { matrix_type } |
            NodeKind::UniformIntervalSample { matrix_type, .. } |
            NodeKind::GaussianSample { matrix_type, .. } |
            NodeKind::HashSample { matrix_type, .. } |
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::Slice { rows, columns } => {
                let Some(output) = output_types.iter_mut().find_map(matrix_leaf_type_mut) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                let Some(input) =
                    argument_types.first_mut().and_then(|wire_type| match wire_type {
                        ConcreteWireType::Matrix(matrix) |
                        ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                        _ => None,
                    })
                else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    input.rows = output.rows;
                    input.columns = representative_columns;
                    output.columns = representative_columns;
                    if let Some(range) = rows {
                        range.start = mxx_ir_core::IntExpr::constant(0);
                        range.end = mxx_ir_core::IntExpr::constant(output.rows);
                    }
                    if let Some(range) = columns {
                        range.start = mxx_ir_core::IntExpr::constant(0);
                        range.end = mxx_ir_core::IntExpr::constant(representative_columns);
                    }
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::Transpose => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    let Some(input) =
                        argument_types.first_mut().and_then(|wire_type| match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        })
                    else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    input.rows = representative_columns;
                    output.columns = representative_columns;
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::Tensor => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if capped_columns(output).is_some() {
                    let original_output_columns = output.columns;
                    let [left_wire, right_wire, ..] = argument_types.as_mut_slice() else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    let Some(left) = (match left_wire {
                        ConcreteWireType::Matrix(matrix) |
                        ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                        _ => None,
                    }) else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    let Some(right) = (match right_wire {
                        ConcreteWireType::Matrix(matrix) |
                        ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                        _ => None,
                    }) else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    // Production splits output ranges at original C_r boundaries. Preserve C_r
                    // and measure enough complete left-column groups to cover the requested W;
                    // the final group may conservatively include the range's partial segment.
                    if column_limit < right.columns {
                        left.columns = 1;
                        right.columns = column_limit;
                    } else {
                        left.columns =
                            left.columns.min(column_limit.div_ceil(right.columns).max(1));
                    }
                    output.columns = left.columns * right.columns;
                    scale = original_output_columns.div_ceil(output.columns) as f64;
                }
            }
            NodeKind::Concat { axis: ConcatAxis::Rows } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    for wire_type in &mut argument_types {
                        let Some(input) = (match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        }) else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        input.columns = representative_columns;
                    }
                    output.columns = representative_columns;
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::Concat { axis: ConcatAxis::Columns } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if capped_columns(output).is_some() {
                    let original_columns = output.columns;
                    let target_columns = original_columns.min(column_limit);
                    let mut representative_columns = 0usize;
                    let mut remaining_columns = target_columns;
                    let mut representative_arguments = Vec::new();
                    for mut wire_type in std::mem::take(&mut argument_types) {
                        if remaining_columns == 0 {
                            break;
                        }
                        let Some(input) = (match &mut wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        }) else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        input.columns = remaining_columns.min(input.columns);
                        remaining_columns -= input.columns;
                        representative_columns =
                            representative_columns.saturating_add(input.columns);
                        representative_arguments.push(wire_type);
                    }
                    argument_types = representative_arguments;
                    output.columns = representative_columns;
                    scale = original_columns.div_ceil(representative_columns.max(1)) as f64;
                }
            }
            NodeKind::GadgetDecompose { .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    let Some(input) =
                        argument_types.first_mut().and_then(|wire_type| match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        })
                    else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    input.columns = representative_columns;
                    output.columns = representative_columns;
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::MatrixScale { .. } |
            NodeKind::RingAutomorphism { .. } |
            NodeKind::ModulusSwitch { .. } |
            NodeKind::ModulusReduce { .. } |
            NodeKind::CenteredRebase { .. } |
            NodeKind::BlockModSwitch { .. } |
            NodeKind::RnsModUp { .. } |
            NodeKind::RnsModDown { .. } |
            NodeKind::MatrixNegate => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    let Some(input) =
                        argument_types.first_mut().and_then(|wire_type| match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        })
                    else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    input.columns = representative_columns;
                    output.columns = representative_columns;
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::MatrixBinary(operation) => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                match operation {
                    MatrixBinaryOp::Add | MatrixBinaryOp::Subtract => {
                        if let Some((representative_columns, column_scale, column_remainder)) =
                            capped_columns(output)
                        {
                            for index in [0, 1] {
                                let Some(input) =
                                    argument_types.get_mut(index).and_then(|wire_type| {
                                        match wire_type {
                                            ConcreteWireType::Matrix(matrix) |
                                            ConcreteWireType::Preimage { matrix, .. } => {
                                                Some(matrix)
                                            }
                                            _ => None,
                                        }
                                    })
                                else {
                                    return (
                                        kind,
                                        argument_types,
                                        output_types,
                                        scale,
                                        remainder_columns,
                                    );
                                };
                                input.columns = representative_columns;
                            }
                            output.columns = representative_columns;
                            scale = column_scale;
                            remainder_columns = column_remainder;
                        }
                    }
                    MatrixBinaryOp::Multiply => {
                        let scalable_argument =
                            Self::matrix_multiply_scaled_argument(&argument_types);
                        let Some(scalable) = argument_types
                            .get_mut(scalable_argument)
                            .and_then(matrix_leaf_type_mut)
                        else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        if let Some((representative_columns, column_scale, column_remainder)) =
                            capped_columns(output)
                        {
                            scalable.columns = representative_columns;
                            output.columns = representative_columns;
                            scale = column_scale;
                            remainder_columns = column_remainder;
                        }
                    }
                }
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let Some(output) = output_types.iter_mut().find_map(matrix_leaf_type_mut) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    for product in 0..coefficients.len() {
                        let left = matrix_leaf_type(&argument_types[2 * product]).unwrap();
                        let right = matrix_leaf_type(&argument_types[2 * product + 1]).unwrap();
                        let scalable_index = if gpu_matrix_multiply_scales_left(
                            left.rows,
                            left.columns,
                            right.rows,
                            right.columns,
                        ) {
                            2 * product
                        } else {
                            2 * product + 1
                        };
                        let Some(scalable) =
                            argument_types.get_mut(scalable_index).and_then(matrix_leaf_type_mut)
                        else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        scalable.columns = representative_columns;
                    }
                    if *has_bias {
                        let Some(bias) = argument_types
                            .get_mut(2 * coefficients.len())
                            .and_then(matrix_leaf_type_mut)
                        else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        bias.columns = representative_columns;
                    }
                    output.columns = representative_columns;
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::MatrixMulSmallRhs => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                let Some(rhs) = argument_types.get_mut(1).and_then(|wire_type| match wire_type {
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    rhs.columns = representative_columns;
                    output.columns = representative_columns;
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::PreimageSample { matrix_type, .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    let Some(target) =
                        argument_types.get_mut(2).and_then(|wire_type| match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        })
                    else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    target.columns = representative_columns;
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::CrtRecompose { .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    for wire_type in &mut argument_types {
                        let Some(input) = (match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        }) else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        input.columns = representative_columns;
                    }
                    output.columns = representative_columns;
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::ExtractCoefficient { .. } | NodeKind::ThresholdDecode { .. } => {
                let Some(input) =
                    argument_types.first_mut().and_then(|wire_type| match wire_type {
                        ConcreteWireType::Matrix(matrix) |
                        ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                        _ => None,
                    })
                else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(input)
                {
                    input.columns = representative_columns;
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            NodeKind::PackPolynomialCoefficients { matrix_type, .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    let original_columns = output.columns;
                    let Some(ConcreteWireType::IndexedFamily { count, .. }) =
                        argument_types.first_mut()
                    else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    *count = count.div_ceil(original_columns);
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            _ => {}
        }
        (kind, argument_types, output_types, scale, remainder_columns)
    }

    fn prepare(
        backend: &mut GpuDcrtBackend,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        fixed_phase: Option<(&[bool], bool)>,
    ) -> Result<PreparedMeasurement, GpuMeasurementError> {
        let phase = fixed_phase.map(|(_, fixed)| fixed);
        if let Some((fixed_arguments, _)) = fixed_phase &&
            fixed_arguments.len() != node.concrete_argument_types.len()
        {
            return Err(GpuMeasurementError(format!(
                "node {:?} fixed-argument ownership has {} entries for {} arguments",
                node.id,
                fixed_arguments.len(),
                node.concrete_argument_types.len()
            )));
        }
        let mut arguments = Vec::with_capacity(node.concrete_argument_types.len());
        let mut small_arguments = Vec::with_capacity(node.concrete_argument_types.len());
        for (index, wire_type) in node.concrete_argument_types.iter().enumerate() {
            let selected =
                fixed_phase.is_none_or(|(fixed_arguments, fixed)| fixed_arguments[index] == fixed);
            // Preimage preparation below creates the real public/trapdoor pair; the first two
            // logical inputs are metadata owners rather than measurement operands.
            let selected = selected &&
                !(matches!(node.kind, NodeKind::PreimageSample { .. }) &&
                    node.concrete_argument_types.len() >= 3 &&
                    index < 2);
            if !selected {
                arguments.push(None);
                small_arguments.push(None);
                continue;
            }
            match family_leaf_type(wire_type) {
                ConcreteWireType::Matrix(matrix) => {
                    let value = backend
                        .constant_matrix(matrix, &ConstantMatrix::Zero, bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    arguments.push(Some(Arc::new(value)));
                    small_arguments.push(None);
                }
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
                ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                    let parameters = backend
                        .constant_matrix(matrix, &ConstantMatrix::Zero, bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    let max_coefficient_bound =
                        max_coefficient_bound.to_biguint().ok_or_else(|| {
                            GpuMeasurementError(
                                "compact matrix coefficient bound must be nonnegative".to_owned(),
                            )
                        })?;
                    let magnitude_bytes = usize::try_from(max_coefficient_bound.bits().div_ceil(8))
                        .map_err(|_| {
                            GpuMeasurementError(
                                "compact matrix bound width overflows usize".to_owned(),
                            )
                        })?
                        .max(1);
                    let payload_len = matrix
                        .rows
                        .checked_mul(matrix.columns)
                        .and_then(|value| value.checked_mul(matrix.ring_dimension as usize))
                        .and_then(|value| value.checked_mul(1 + magnitude_bytes))
                        .ok_or_else(|| {
                            GpuMeasurementError(
                                "compact matrix payload length overflows".to_owned(),
                            )
                        })?;
                    let value = GpuSmallMatrix::from_canonical_coefficients(
                        parameters
                            .shards()
                            .first()
                            .expect("single-device estimator matrix needs one shard")
                            .value
                            .params(),
                        matrix.rows,
                        matrix.columns,
                        max_coefficient_bound,
                        &vec![0u8; payload_len],
                    )
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    arguments.push(None);
                    small_arguments.push(Some(Arc::new(GpuFleetSmallMatrix::from(value))));
                }
                _ => {
                    arguments.push(None);
                    small_arguments.push(None);
                }
            }
        }
        let preimage_trapdoor = if matches!(node.kind, NodeKind::PreimageSample { .. }) &&
            node.concrete_argument_types.len() >= 3 &&
            phase != Some(false)
        {
            let Some(ConcreteWireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound: _,
            }) = node.concrete_argument_types.get(1)
            else {
                return Err(GpuMeasurementError(
                    "preimage measurement is missing trapdoor metadata".to_owned(),
                ));
            };
            let sigma = sigma
                .evaluate_f64(bindings)
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
            let (public, trapdoor) = backend
                .sample_trapdoor(matrix, sigma, gadget_base, *digit_count)
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
            public.shards().iter().for_each(|shard| shard.value.wait_until_ready());
            trapdoor.wait_until_ready();
            Some((
                public,
                trapdoor,
                sigma,
                gadget_base.clone(),
                *digit_count,
                node.concrete_output_types
                    .iter()
                    .find_map(|wire_type| match wire_type {
                        ConcreteWireType::Preimage { max_coefficient_bound, .. } => {
                            Some(max_coefficient_bound.clone())
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        GpuMeasurementError(
                            "preimage measurement is missing its declared output bound".to_owned(),
                        )
                    })?,
            ))
        } else {
            None
        };
        if let Some((public, ..)) = &preimage_trapdoor {
            arguments[0] = Some(Arc::new(public.clone()));
        }
        let preimage_target = if matches!(node.kind, NodeKind::PreimageSample { .. }) &&
            phase != Some(true)
        {
            let target_index = if node.concrete_argument_types.len() >= 3 { 2 } else { 0 };
            let target = arguments.get(target_index).and_then(Option::as_ref).ok_or_else(|| {
                GpuMeasurementError("missing prepared preimage target".to_owned())
            })?;
            Some(
                backend
                    .preimage_target(target.clone())
                    .map_err(|error| GpuMeasurementError(error.to_string()))?
                    .0,
            )
        } else {
            None
        };
        Ok(PreparedMeasurement { arguments, small_arguments, preimage_trapdoor, preimage_target })
    }

    fn preimage_session_prepared(
        &mut self,
        signature: &GpuWarmupOperationSignature,
        device: usize,
        descriptor: &PendingMeasurement,
        representative: &RepresentativeMeasurement,
        reuse_existing: bool,
    ) -> Result<PreparedMeasurement, GpuMeasurementError> {
        let key = (signature.clone(), device);
        if reuse_existing && let Some(prepared) = self.preimage_session_inputs.get(&key) {
            return Ok(prepared.clone());
        }
        if reuse_existing {
            return Err(GpuMeasurementError(
                "warm preimage measurement has no retained cold-session trapdoor owner".into(),
            ));
        }
        let node = MeasurementNode {
            scope: &descriptor.scope,
            id: descriptor.id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let prepared =
            Self::prepare(&mut self.workers[device].backend, &node, &descriptor.bindings, None)?;
        if prepared.preimage_trapdoor.is_none() || prepared.preimage_target.is_none() {
            return Err(GpuMeasurementError(
                "preimage session preparation did not produce a trapdoor owner and target".into(),
            ));
        }
        self.preimage_session_inputs.insert(key, prepared.clone());
        Ok(prepared)
    }

    fn run_fleet_iteration(
        workers: &mut [GpuMeasurementWorker],
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        prepared: &[Option<(RepresentativeMeasurement, PreparedMeasurement)>],
    ) -> Result<(Vec<Option<f64>>, f64), GpuMeasurementError> {
        let active_workers = prepared.iter().flatten().count();
        if active_workers == 0 {
            return Err(GpuMeasurementError("GPU fleet wave has no active device".to_owned()));
        }
        let barrier = Arc::new(Barrier::new(active_workers));
        // Start before Rayon enqueues the fleet work and stop only after every active worker has
        // observed its output completion event and joined. This is the fleet wall latency, not a
        // maximum assembled from independently timed device runs.
        let fleet_started = std::time::Instant::now();
        let device_seconds = workers
            .par_iter_mut()
            .zip(prepared.par_iter())
            .map(|(worker, prepared)| {
                let Some((representative, prepared)) = prepared else {
                    return Ok(None);
                };
                let node = MeasurementNode {
                    scope,
                    id,
                    kind: &representative.kind,
                    arguments: &[],
                    argument_kinds: &[],
                    argument_types: &[],
                    output_types: &[],
                    concrete_argument_types: representative.concrete_argument_types.clone(),
                    concrete_output_types: representative.concrete_output_types.clone(),
                };
                barrier.wait();
                let device_started = std::time::Instant::now();
                Self::execute_local_production_measurement(
                    worker,
                    &node,
                    bindings,
                    representative,
                    prepared,
                    1,
                    None,
                    None,
                    false,
                )?;
                Ok(Some(device_started.elapsed().as_secs_f64()))
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
        Ok((device_seconds, fleet_started.elapsed().as_secs_f64()))
    }

    fn measure_fleet_wave(
        workers: &mut [GpuMeasurementWorker],
        harness: &MeasurementHarnessConfig,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        operation: [u8; 32],
        representatives: Vec<Option<RepresentativeMeasurement>>,
        fixed_inputs: Vec<PreparedMeasurement>,
    ) -> Result<(Vec<Option<NodeMeasurement>>, f64), GpuMeasurementError> {
        if harness.measured_iterations == 0 {
            return Err(GpuMeasurementError("measured iteration count must be positive".to_owned()));
        }
        let prepared = workers
            .par_iter_mut()
            .zip(representatives.into_par_iter())
            .zip(fixed_inputs.into_par_iter())
            .map(|((worker, representative), fixed)| {
                representative
                    .map(|representative| {
                        let columns = representative.measured_columns().ok_or_else(|| {
                            GpuMeasurementError(
                                "fleet representative has no matrix columns".to_owned(),
                            )
                        })?;
                        worker.backend.set_column_widths_for_operation(
                            operation,
                            GpuColumnWidths { gpu0: columns, nonzero: None },
                        );
                        worker
                            .backend
                            .select_operation(operation)
                            .map_err(|error| GpuMeasurementError(error.to_string()))?;
                        let node = MeasurementNode {
                            scope,
                            id,
                            kind: &representative.kind,
                            arguments: &[],
                            argument_kinds: &[],
                            argument_types: &[],
                            output_types: &[],
                            concrete_argument_types: representative.concrete_argument_types.clone(),
                            concrete_output_types: representative.concrete_output_types.clone(),
                        };
                        let scaled = Self::prepare(
                            &mut worker.backend,
                            &node,
                            bindings,
                            Some((&representative.fixed_arguments, false)),
                        )?;
                        let prepared = fixed
                            .merge_for_representative(scaled, &representative.fixed_arguments)?;
                        Ok((representative, prepared))
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;

        for _ in 0..harness.warm_up_iterations {
            let _ = Self::run_fleet_iteration(workers, scope, id, bindings, &prepared)?;
        }
        let baselines = workers
            .par_iter_mut()
            .zip(prepared.par_iter())
            .map(|(worker, prepared)| {
                if prepared.is_none() {
                    return Ok(None);
                }
                begin_gpu_memory_measurement(worker).map(Some)
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;

        let mut device_seconds = vec![0.0; workers.len()];
        let mut fleet_seconds = 0.0;
        for _ in 0..harness.measured_iterations {
            let (iteration_devices, iteration_fleet) =
                Self::run_fleet_iteration(workers, scope, id, bindings, &prepared)?;
            for (total, elapsed) in device_seconds.iter_mut().zip(iteration_devices) {
                *total += elapsed.unwrap_or(0.0);
            }
            fleet_seconds += iteration_fleet;
        }
        let iterations = harness.measured_iterations as f64;
        let measurements = workers
            .par_iter()
            .zip(baselines.into_par_iter())
            .zip(device_seconds.into_par_iter())
            .map(|((worker, baseline), seconds)| {
                let Some(baseline) = baseline else {
                    return Ok(None);
                };
                let workspace_bytes = finish_gpu_memory_measurement(worker.device_id, baseline)?;
                let seconds = seconds / iterations;
                Ok(Some(NodeMeasurement {
                    work_seconds: seconds,
                    latency_seconds: seconds,
                    cumulative_wave_seconds: seconds,
                    independent_wave_count: 1,
                    measured_wave_workspace_bytes: workspace_bytes,
                    workspace_bytes,
                }))
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
        Ok((measurements, fleet_seconds / iterations))
    }

    fn place_prepared_sources(
        worker: &mut GpuMeasurementWorker,
        representative: &RepresentativeMeasurement,
        mut prepared: PreparedMeasurement,
    ) -> Result<PreparedMeasurement, GpuMeasurementError> {
        // Effective lowering may intentionally retain only a non-prefix
        // operand. GadgetTrapdoor-backed preimage descriptors compact the
        // target to argument 0, while sampled preimages retain validation
        // metadata in arguments 0/1 and consume argument 2.
        let source_indices = if matches!(representative.kind, NodeKind::PreimageSample { .. }) &&
            representative.source_layouts.len() == 1
        {
            vec![if representative.concrete_argument_types.len() >= 3 { 2 } else { 0 }]
        } else {
            (0..representative.concrete_argument_types.len()).collect()
        };
        for (layout, index) in representative.source_layouts.iter().zip(source_indices) {
            let Some(ty) = representative.concrete_argument_types.get(index) else { continue };
            if ty.matrix_type().is_none() {
                continue;
            }
            if let Some(value) = prepared.arguments.get(index).and_then(Option::as_ref) {
                prepared.arguments[index] = Some(Arc::new(
                    worker
                        .backend
                        .place_measurement_matrix(value, layout)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?,
                ));
            }
            if let Some(value) = prepared.small_arguments.get(index).and_then(Option::as_ref) {
                prepared.small_arguments[index] = Some(Arc::new(
                    worker
                        .backend
                        .place_measurement_compact(value, layout)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?,
                ));
            }
        }
        if matches!(representative.kind, NodeKind::PreimageSample { .. }) {
            let target_index =
                if representative.concrete_argument_types.len() >= 3 { 2 } else { 0 };
            if let Some(target) = prepared.arguments.get(target_index).and_then(Option::as_ref) {
                prepared.preimage_target = Some(
                    worker
                        .backend
                        .preimage_target(target.clone())
                        .map_err(|error| GpuMeasurementError(error.to_string()))?
                        .0,
                );
            }
        }
        prepared.finish();
        Ok(prepared)
    }

    fn measure_representative(
        worker: &mut GpuMeasurementWorker,
        harness: &MeasurementHarnessConfig,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
        barrier: Option<&Barrier>,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        if let Some(barrier) = barrier {
            barrier.wait();
        }
        let node = MeasurementNode {
            scope,
            id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let prepared = Self::prepare(&mut worker.backend, &node, bindings, None)?;
        let prepared = Self::place_prepared_sources(worker, representative, prepared)?;
        let baseline = begin_gpu_memory_measurement(worker)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(harness, &probe, 1, |representative_batch| {
            if operation_error.is_some() {
                return;
            }
            if let Err(error) = Self::execute_local_production_measurement(
                worker,
                &node,
                bindings,
                representative,
                &prepared,
                representative_batch,
                None,
                None,
                false,
            ) {
                operation_error = Some(error);
            }
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        let mut measurement = measured.measurement;
        measurement.workspace_bytes = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        measurement.measured_wave_workspace_bytes = measurement.workspace_bytes;
        Ok(measurement)
    }

    fn measure_preimage_cache_setup(
        worker: &mut GpuMeasurementWorker,
        prepared: &PreparedMeasurement,
        matrix_type: &ConcreteMatrixType,
        sigma: f64,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        let (_, trapdoor, ..) = prepared.preimage_trapdoor.as_ref().ok_or_else(|| {
            GpuMeasurementError("cold preimage setup has no retained trapdoor owner".into())
        })?;
        let baseline = begin_gpu_memory_measurement(worker)?;
        let started = std::time::Instant::now();
        let built = worker
            .backend
            .prepare_preimage_covariance_cache_for_measurement(matrix_type, sigma, trapdoor, 0)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if !built {
            return Err(GpuMeasurementError(
                "cold preimage setup unexpectedly reused a retained covariance cache".into(),
            ));
        }
        let seconds = started.elapsed().as_secs_f64();
        let workspace = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        Ok(NodeMeasurement {
            work_seconds: seconds,
            latency_seconds: seconds,
            cumulative_wave_seconds: seconds,
            independent_wave_count: 1,
            measured_wave_workspace_bytes: workspace,
            workspace_bytes: workspace,
        })
    }

    /// Invoke the production fused entry point for a warmup representative.
    /// The ordinary IR kind is intentionally not dispatched here: fused
    /// lowering has a distinct kernel/workspace contract.
    fn run_fused_batch(
        backend: &mut GpuDcrtBackend,
        fused: FusedWarmupOperation,
        representative: &RepresentativeMeasurement,
        bindings: &ParamEnv,
        prepared: &PreparedMeasurement,
        batch_size: usize,
        binding_port: usize,
    ) -> Result<Vec<GpuMeasurementOutput>, GpuMeasurementError> {
        let matrix = |index: usize| {
            prepared.arguments.get(index).and_then(Option::as_ref).cloned().ok_or_else(|| {
                GpuMeasurementError(format!("fused matrix argument {index} is missing"))
            })
        };
        let output_range = representative.output_range.as_ref().ok_or_else(|| {
            GpuMeasurementError("fused production measurement requires an explicit range".into())
        })?;
        let (range_start, range_end) = (output_range.start, output_range.end);
        let metadata = representative.fixed_metadata.as_ref().ok_or_else(|| {
            GpuMeasurementError("fused measurement lacks canonical fixed metadata".into())
        })?;
        match fused {
            FusedWarmupOperation::RowSum |
            FusedWarmupOperation::TensorRowSum |
            FusedWarmupOperation::RowBlockAdd |
            FusedWarmupOperation::Decompose |
            FusedWarmupOperation::CompactProduct => {
                let mut outputs = Vec::new();
                for instance in 0..batch_size {
                    let mut metadata = metadata.clone();
                    metadata.instance_slots = vec![instance];
                    let blocks =
                        || prepared.arguments.iter().flatten().cloned().collect::<Vec<_>>();
                    let request = match fused {
                        FusedWarmupOperation::RowSum | FusedWarmupOperation::TensorRowSum => {
                            if representative.inputs.row_groups.is_empty() {
                                return Err(GpuMeasurementError(
                                    "missing validated row groups".into(),
                                ));
                            }
                            mxx_runtime::backend::FusedBatchRequest::RowSum {
                                metadata,
                                source: matrix(0)?,
                                right: if fused == FusedWarmupOperation::TensorRowSum {
                                    Some(matrix(1)?)
                                } else {
                                    None
                                },
                                rows: representative.inputs.row_groups.clone(),
                            }
                        }
                        FusedWarmupOperation::RowBlockAdd => {
                            let mut blocks = blocks();
                            let right = blocks
                                .pop()
                                .ok_or_else(|| GpuMeasurementError("missing fused RHS".into()))?;
                            mxx_runtime::backend::FusedBatchRequest::Add { metadata, blocks, right }
                        }
                        FusedWarmupOperation::Decompose => {
                            let (small, digits) = match &representative.kind {
                                NodeKind::GadgetDecompose { small, digit_count, .. } => (
                                    *small,
                                    digit_count
                                        .evaluate(bindings)
                                        .map_err(|error| GpuMeasurementError(error.to_string()))?
                                        .to_usize()
                                        .ok_or_else(|| {
                                            GpuMeasurementError("invalid digit count".into())
                                        })?,
                                ),
                                NodeKind::PreimageSample { .. } => {
                                    if let Some(ConcreteWireType::Trapdoor {
                                        digit_count, ..
                                    }) = representative.concrete_argument_types.get(1)
                                    {
                                        (false, *digit_count)
                                    } else {
                                        let target = representative
                                            .concrete_argument_types
                                            .first()
                                            .and_then(matrix_leaf_type)
                                            .ok_or_else(|| {
                                                GpuMeasurementError(
                                                    "gadget preimage lacks target metadata".into(),
                                                )
                                            })?;
                                        let output = representative
                                            .concrete_output_types
                                            .first()
                                            .and_then(matrix_leaf_type)
                                            .ok_or_else(|| {
                                                GpuMeasurementError(
                                                    "gadget preimage lacks output metadata".into(),
                                                )
                                            })?;
                                        let digits = output
                                            .rows
                                            .checked_div(target.rows)
                                            .ok_or_else(|| {
                                                GpuMeasurementError(
                                                    "gadget preimage has invalid row metadata"
                                                        .into(),
                                                )
                                            })?;
                                        (false, digits)
                                    }
                                }
                                _ => {
                                    return Err(GpuMeasurementError(
                                        "invalid fused decomposition kind".into(),
                                    ))
                                }
                            };
                            mxx_runtime::backend::FusedBatchRequest::Decompose {
                                metadata,
                                blocks: blocks(),
                                small,
                                digits,
                            }
                        }
                        FusedWarmupOperation::CompactProduct => {
                            let rhs = prepared
                                .small_arguments
                                .last()
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| GpuMeasurementError("missing compact RHS".into()))?;
                            mxx_runtime::backend::FusedBatchRequest::SmallProduct {
                                metadata,
                                blocks: blocks(),
                                rhs,
                            }
                        }
                        FusedWarmupOperation::PreimageBatch => unreachable!(),
                    };
                    match backend
                        .fixed_fused_range_for_measurement(
                            request,
                            ColumnRange { start: range_start, end: range_end },
                            binding_port,
                        )
                        .map_err(|error| GpuMeasurementError(error.to_string()))?
                    {
                        mxx_runtime::backend::FusedBatchOutput::Matrices(values) => {
                            outputs.extend(values.into_iter().map(GpuMeasurementOutput::matrix))
                        }
                        mxx_runtime::backend::FusedBatchOutput::Small(value) => {
                            outputs.push(GpuMeasurementOutput::SmallMatrix(value))
                        }
                    }
                }
                Ok(outputs)
            }
            FusedWarmupOperation::PreimageBatch => {
                let ty = representative
                    .concrete_output_types
                    .iter()
                    .find_map(|wire_type| wire_type.matrix_type())
                    .ok_or_else(|| {
                        GpuMeasurementError("fused preimage has no matrix output".into())
                    })?;
                let (public, trapdoor, sigma, gadget_base, digit_count, bound) =
                    prepared.preimage_trapdoor.as_ref().ok_or_else(|| {
                        GpuMeasurementError("fused preimage trapdoor is missing".into())
                    })?;
                let target = prepared.preimage_target.as_ref().ok_or_else(|| {
                    GpuMeasurementError("fused preimage target is missing".into())
                })?;
                let requests: Vec<PreimageRequest<GpuFleetMatrix, GpuFleetTrapdoor>> = (0..
                    batch_size)
                    .map(|index| {
                        let mut randomness_seed = [0x50; 32];
                        randomness_seed[..size_of::<usize>()].copy_from_slice(&index.to_le_bytes());
                        PreimageRequest {
                            fixed_metadata: Some({
                                let mut metadata = metadata.clone();
                                metadata.instance_slots = vec![index];
                                metadata.randomness_seeds = vec![Some(randomness_seed)];
                                metadata
                            }),
                            instance_slot: index,
                            matrix_type: ty.clone(),
                            sigma: *sigma,
                            gadget_base: gadget_base.clone(),
                            digit_count: *digit_count,
                            max_coefficient_bound: bound.clone(),
                            trapdoor: Arc::new(trapdoor.clone()),
                            public: Arc::new(public.clone()),
                            target: target.clone(),
                            randomness_seed,
                        }
                    })
                    .collect();
                let max_attempts = representative.retry_cap.ok_or_else(|| {
                    GpuMeasurementError("preimage measurement requires the frozen retry cap".into())
                })?;
                requests
                    .into_iter()
                    .map(|request| {
                        backend
                            .fixed_preimage_range_for_measurement(
                                request,
                                ColumnRange { start: range_start, end: range_end },
                                max_attempts,
                            )
                            .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| error)?
                    .into_iter()
                    .map(|value| Ok(GpuMeasurementOutput::SmallMatrix(value)))
                    .collect()
            }
        }
    }

    /// Run a representative through the same range-aware production job
    /// boundary used by fixed dispatch.  The source values remain typed fleet
    /// owners and the mapper keeps global coordinates; only the lowered
    /// materialized values are handed to the operation callback.  In
    /// particular, this prevents the collector from measuring a detached
    /// `prepare`/`finish` phase followed by an unrelated kernel timer.
    fn execute_local_production_measurement(
        worker: &mut GpuMeasurementWorker,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
        prepared: &PreparedMeasurement,
        batch_size: usize,
        fused: Option<FusedWarmupOperation>,
        binding_port: Option<usize>,
        transfer_only: bool,
    ) -> Result<(), GpuMeasurementError> {
        let output_columns = representative
            .concrete_output_types
            .iter()
            .find_map(Self::matrix_columns)
            .or_else(|| {
                representative.concrete_argument_types.iter().find_map(Self::matrix_columns)
            })
            .ok_or_else(|| {
                GpuMeasurementError("production job has no matrix output or input".into())
            })?;
        let output = representative.output_range.as_ref().map_or(
            ColumnRange { start: 0, end: output_columns },
            |range| ColumnRange { start: range.start, end: range.end },
        );
        if output.is_empty() {
            return Err(GpuMeasurementError("production job output range is empty".into()));
        }
        let mut execution = if let Some(fused) = fused {
            let inputs = match fused {
                FusedWarmupOperation::CompactProduct => representative
                    .concrete_argument_types
                    .iter()
                    .enumerate()
                    .filter_map(|(operand, ty)| {
                        let columns = Self::matrix_columns(ty)?;
                        Some(InputColumnRange {
                            operand,
                            range: if operand + 1 == representative.concrete_argument_types.len() {
                                output
                            } else {
                                ColumnRange { start: 0, end: columns }
                            },
                        })
                    })
                    .collect(),
                _ => {
                    let kind = match fused {
                        FusedWarmupOperation::RowSum => NodeKind::MatrixNegate,
                        FusedWarmupOperation::TensorRowSum => NodeKind::Tensor,
                        _ => node.kind.clone(),
                    };
                    map_output_range_to_inputs_with_output(
                        &kind,
                        &representative.concrete_argument_types,
                        output_columns,
                        output,
                    )
                    .map_err(|error| GpuMeasurementError(error.to_string()))?
                }
            };
            GpuExecutionRange {
                variant: GpuExecutionVariant::Fused(fused),
                output,
                inputs,
                fragment: TypedFragmentClass::PlannedJob,
            }
        } else if Self::is_host_backend_boundary(node.kind) ||
            column_capability(node.kind, &representative.concrete_argument_types) ==
                ColumnCapability::SingleDevice
        {
            // Boundary primitives still consume their real production matrix
            // inputs, but they have no column-lowering rule: extraction and
            // codec dispatch operate on the complete host-visible value.
            // Build the typed full-range request explicitly so their host
            // elapsed time is measured through the same materializer and any
            // route/staging cost is retained in the transfer profile.
            let inputs = representative
                .concrete_argument_types
                .iter()
                .enumerate()
                .filter_map(|(operand, ty)| {
                    let columns = Self::matrix_columns(ty)?;
                    Some(InputColumnRange {
                        operand,
                        range: ColumnRange { start: 0, end: columns },
                    })
                })
                .collect();
            GpuExecutionRange {
                variant: GpuExecutionVariant::Primitive(effective_gpu_operation(node.kind)),
                output,
                inputs,
                fragment: TypedFragmentClass::Full,
            }
        } else {
            gpu_execution_range(
                node.kind,
                &representative.concrete_argument_types,
                output_columns,
                output,
                output.len(),
            )
            .map_err(|error| GpuMeasurementError(error.to_string()))?
        };

        // The range mapper addresses logical node operands.  The production
        // job source list contains only typed matrix/compact owners, so keep
        // an explicit operand remap instead of silently shifting ranges or
        // rebuilding a synthetic local matrix.
        let mut source_indices = vec![None; representative.concrete_argument_types.len()];
        let mut source_operands = Vec::new();
        let mut sources = Vec::new();
        for operand in 0..representative.concrete_argument_types.len() {
            if let Some(value) = prepared.arguments.get(operand).and_then(Option::as_ref) {
                source_indices[operand] = Some(sources.len());
                source_operands.push(operand);
                sources.push(GpuLocalProductionSource::Matrix(value.as_ref()));
            } else if let Some(value) =
                prepared.small_arguments.get(operand).and_then(Option::as_ref)
            {
                source_indices[operand] = Some(sources.len());
                source_operands.push(operand);
                sources.push(GpuLocalProductionSource::Compact(value.as_ref()));
            }
        }
        if matches!(node.kind, NodeKind::PreimageSample { .. }) &&
            fused != Some(FusedWarmupOperation::Decompose)
        {
            if prepared.preimage_trapdoor.is_none() {
                return Err(GpuMeasurementError(
                    "preimage job lacks its native trapdoor owner".into(),
                ));
            }
            // The opaque trapdoor is consumed by fixed_sample_preimage's
            // native context, not the matrix/compact transport adapter. Its
            // residency and workspace have a sampler-specific certificate.
            execution.inputs.retain(|input| input.operand != 1);
        }
        for input in &mut execution.inputs {
            input.operand =
                source_indices.get(input.operand).and_then(|index| *index).ok_or_else(|| {
                    GpuMeasurementError(format!(
                        "production range input {} has no typed prepared source",
                        input.operand
                    ))
                })?;
        }
        let request = GpuLocalProductionJobRequest {
            execution: &execution,
            sources: &sources,
            destination_device: worker.device_id,
            materialize_inputs: transfer_only ||
                (fused.is_none() &&
                    !matches!(
                        node.kind,
                        NodeKind::Slice { .. } |
                            NodeKind::Transpose |
                            NodeKind::Tensor |
                            NodeKind::Concat { .. }
                    )),
            destination_compact: matches!(
                representative.concrete_output_types.iter().find(|ty| {
                    matches!(
                        ty,
                        ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. }
                    )
                }),
                Some(_)
            ),
        };
        let result = worker
            .backend
            .execute_local_production_job(request, |backend, context| {
                if transfer_only {
                    // The production materializer has already performed the
                    // exact resident/peer/host-staging copy path.  For a
                    // device-to-host boundary, force the same host-visible
                    // readback fence that the production codec uses, but do
                    // not run its decoding/arithmetic core here.  H2D is
                    // already closed by materialization; D2H is closed by
                    // this explicit backend codec read.
                    if effective_gpu_operation(node.kind).transfer_kind() ==
                        WarmupTransferKind::DeviceToHost
                    {
                        for input in &context.inputs {
                            if let GpuLocalProductionInput::Matrix { value, .. } = input {
                                std::hint::black_box(value.to_compact_bytes());
                            }
                        }
                    }
                    return Ok::<GpuMeasurementOutputs, PolyBackendError>(GpuMeasurementOutputs(
                        Vec::new(),
                    ));
                }
                // Consume the values actually materialized by the job. This
                // keeps mapped/peer/staged measurements honest: the callback
                // must not accidentally fall back to the pre-materialized
                // setup owner when a route performed a conversion.
                // Mapped operations and fused lowerings keep the semantic
                // fleet owners here. Their production range adapters consume
                // the global output range (Slice/Transpose/Tensor/Concat and
                // all fused variants); replacing them with already-local
                // materialized pieces would apply the global offset twice.
                // The inclusive job has still materialized every mapped input
                // and charged the actual route/resource envelope above.
                let semantic_range_operation = matches!(
                    node.kind,
                    NodeKind::Slice { .. } |
                        NodeKind::Transpose |
                        NodeKind::Tensor |
                        NodeKind::Concat { .. }
                );
                let mut execution_prepared = prepared.clone();
                if fused.is_none() && !semantic_range_operation {
                    for input in &context.inputs {
                        if let GpuLocalProductionInput::Matrix { operand, value, .. } = input {
                            let operand = source_operands[*operand];
                            if let Some(slot) = execution_prepared.arguments.get_mut(operand) {
                                *slot = Some(Arc::new(GpuFleetMatrix::from(value.clone())));
                            }
                            if operand == 2 && matches!(node.kind, NodeKind::PreimageSample { .. })
                            {
                                execution_prepared.preimage_target = Some(
                                    backend
                                        .preimage_target(Arc::new(GpuFleetMatrix::from(
                                            value.clone(),
                                        )))
                                        .map_err(|error| {
                                            PolyBackendError::GpuCalibration(error.to_string())
                                        })?
                                        .0,
                                );
                            }
                        } else if let GpuLocalProductionInput::Compact { operand, value, .. } =
                            input
                        {
                            let operand = source_operands[*operand];
                            execution_prepared.small_arguments[operand] = Some(Arc::new(
                                GpuFleetSmallMatrix::from_column_pieces(value.clone())?,
                            ));
                        }
                    }
                }
                let outputs = if let Some(fused) = fused {
                    let binding_port = binding_port
                        .ok_or_else(|| {
                            GpuMeasurementError(
                                "fused production measurement lacks binding port".into(),
                            )
                        })
                        .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
                    Self::run_fused_batch(
                        backend,
                        fused,
                        representative,
                        bindings,
                        &execution_prepared,
                        batch_size,
                        binding_port,
                    )
                    .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?
                } else {
                    Self::run_node(
                        backend,
                        node,
                        bindings,
                        batch_size,
                        &execution_prepared,
                        representative.output_range.as_ref(),
                    )
                    .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?
                };
                // Completion belongs to the same production job interval.
                // Return the actual production outputs through the inclusive
                // job result.  The fleet boundary fences them via
                // `GpuProductionCompletion`; retaining them in the result
                // prevents this provider from accidentally timing a detached
                // kernel and then dropping the output before completion.
                Ok::<GpuMeasurementOutputs, PolyBackendError>(GpuMeasurementOutputs(outputs))
            })
            .map_err(|error| {
                let message = error.to_string();
                if message.to_ascii_lowercase().contains("out of memory") ||
                    message.to_ascii_lowercase().contains("cuda_error_out_of_memory")
                {
                    GpuMeasurementError::out_of_memory(message)
                } else {
                    GpuMeasurementError(message)
                }
            })?;
        worker.last_production_job = Some(ProductionJobObservation {
            elapsed: result.elapsed,
            routes: result.routes,
            resources: result.resources,
        });
        Ok(())
    }

    fn measure_fused_representative(
        worker: &mut GpuMeasurementWorker,
        harness: &MeasurementHarnessConfig,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
        fused: FusedWarmupOperation,
        binding_port: usize,
        session_prepared: Option<&PreparedMeasurement>,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        let node = MeasurementNode {
            scope,
            id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let prepared = match session_prepared {
            Some(prepared) => prepared.clone(),
            None if fused == FusedWarmupOperation::Decompose &&
                matches!(representative.kind, NodeKind::PreimageSample { .. }) =>
            {
                // GadgetTrapdoor-backed preimage is not a sampled preimage.
                // Prepare only the target operand (the fixed decomposition
                // path) and never manufacture a secret trapdoor for warmup.
                Self::prepare(
                    &mut worker.backend,
                    &node,
                    bindings,
                    Some((&representative.fixed_arguments, false)),
                )?
            }
            None => Self::prepare(&mut worker.backend, &node, bindings, None)?,
        };
        let prepared = Self::place_prepared_sources(worker, representative, prepared)?;
        let baseline = begin_gpu_memory_measurement(worker)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(harness, &probe, 1, |batch_size| {
            if operation_error.is_some() {
                return;
            }
            if let Err(error) = Self::execute_local_production_measurement(
                worker,
                &node,
                bindings,
                representative,
                &prepared,
                batch_size,
                Some(fused),
                Some(binding_port),
                false,
            ) {
                operation_error = Some(error);
            }
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        let mut measurement = measured.measurement;
        measurement.workspace_bytes = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        measurement.measured_wave_workspace_bytes = measurement.workspace_bytes;
        Ok(measurement)
    }

    /// Measure a transfer-only production point.  This deliberately enters
    /// the same fleet materializer as a normal local job and returns before
    /// the kernel callback, so D2H/H2D, peer copies, and host staging are
    /// included in the point while kernel/assembly work is not.
    fn measure_transfer_representative(
        worker: &mut GpuMeasurementWorker,
        harness: &MeasurementHarnessConfig,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        let node = MeasurementNode {
            scope,
            id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let prepared = Self::prepare(&mut worker.backend, &node, bindings, None)?;
        let boundary = Self::is_host_backend_boundary(node.kind);
        let boundary_type = representative
            .concrete_output_types
            .iter()
            .chain(&representative.concrete_argument_types)
            .find_map(|ty| ty.matrix_type());
        let canonical =
            vec![num_bigint::BigUint::from(0u8); boundary_type.map_or(0, |ty| ty.ring_dimension)];
        let boundary_route =
            if boundary { Some(Self::boundary_route(worker, representative)?) } else { None };
        let prepared = Self::place_prepared_sources(worker, representative, prepared)?;
        let baseline = begin_gpu_memory_measurement(worker)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(harness, &probe, 1, |batch_size| {
            if operation_error.is_some() {
                return;
            }
            if let Some(route) = boundary_route {
                let started = std::time::Instant::now();
                let result = (|| -> Result<(), GpuMeasurementError> {
                    for _ in 0..batch_size {
                        match node.kind {
                            NodeKind::PolynomialFromValues { .. } |
                            NodeKind::PackPolynomialCoefficients { .. } => {
                                let evaluation = matches!(
                                    node.kind,
                                    NodeKind::PolynomialFromValues { evaluation: true, .. }
                                );
                                let value = worker
                                    .backend
                                    .upload_polynomial_values(
                                        boundary_type.ok_or_else(|| {
                                            GpuMeasurementError("missing boundary type".into())
                                        })?,
                                        &canonical,
                                        evaluation,
                                    )
                                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                                value.complete();
                            }
                            _ => {
                                let source = prepared
                                    .arguments
                                    .first()
                                    .and_then(Option::as_ref)
                                    .ok_or_else(|| {
                                        GpuMeasurementError("missing download source".into())
                                    })?;
                                std::hint::black_box(
                                    worker
                                        .backend
                                        .download_polynomial_values(
                                            source,
                                            matches!(
                                                node.kind,
                                                NodeKind::PolynomialValues { evaluation: true }
                                            ),
                                        )
                                        .map_err(|error| GpuMeasurementError(error.to_string()))?,
                                );
                            }
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    operation_error = Some(error);
                    return;
                }
                worker.last_production_job = Some(ProductionJobObservation {
                    elapsed: started.elapsed(),
                    routes: vec![route],
                    resources: vec![GpuAffectedResourceEnvelope {
                        device_index: 0,
                        physical_device: worker.device_id,
                        device_bytes: route.source_staging_bytes,
                        host_bytes: route.host_staging_bytes,
                        pinned_host_bytes: route.pinned_host_staging_bytes,
                    }],
                });
                return;
            }
            if let Err(error) = Self::execute_local_production_measurement(
                worker,
                &node,
                bindings,
                representative,
                &prepared,
                batch_size,
                None,
                None,
                true,
            ) {
                operation_error = Some(error);
            }
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        let mut measurement = measured.measurement;
        measurement.workspace_bytes = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        measurement.measured_wave_workspace_bytes = measurement.workspace_bytes;
        Ok(measurement)
    }

    fn calibrate_representative(
        worker: &mut GpuMeasurementWorker,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
        operation: [u8; 32],
        pilot_columns: usize,
        barrier: Option<&Barrier>,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        if let Some(barrier) = barrier {
            barrier.wait();
        }
        let node = MeasurementNode {
            scope,
            id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        worker.backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: pilot_columns, nonzero: None },
        );
        worker
            .backend
            .select_operation(operation)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let fixed = Self::prepare(
            &mut worker.backend,
            &node,
            bindings,
            Some((&representative.fixed_arguments, true)),
        )?;
        let baseline = begin_gpu_memory_measurement(worker)?;
        let scaled = Self::prepare(
            &mut worker.backend,
            &node,
            bindings,
            Some((&representative.fixed_arguments, false)),
        )?;
        let prepared = fixed.merge(scaled);
        Self::execute_local_production_measurement(
            worker,
            &node,
            bindings,
            representative,
            &prepared,
            1,
            None,
            None,
            false,
        )?;
        let latency_seconds = worker
            .last_production_job
            .as_ref()
            .ok_or_else(|| GpuMeasurementError("production job returned no observation".into()))?
            .elapsed
            .as_secs_f64();
        let workspace_bytes = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        Ok(NodeMeasurement {
            work_seconds: latency_seconds,
            latency_seconds,
            cumulative_wave_seconds: latency_seconds,
            independent_wave_count: 1,
            measured_wave_workspace_bytes: workspace_bytes,
            workspace_bytes,
        })
    }

    /// Measure a scalar host primitive through the same dispatch used by the
    /// production executor.  Host-only primitives consume no VRAM, but their
    /// elapsed host time remains part of the warmup profile.  Inputs here are
    /// representative values because the estimator has no runtime value map;
    /// the operation itself is never reimplemented in this crate.
    fn representative_runtime_value(
        backend: &mut GpuDcrtBackend,
        wire_type: &ConcreteWireType,
        bindings: &ParamEnv,
    ) -> Result<RuntimeValue<GpuDcrtBackend>, GpuMeasurementError> {
        if let ConcreteWireType::IndexedFamily { element, count } = wire_type {
            // Preserve the exact family cardinality for typed input lookup,
            // but clone one payload exemplar.  Matrix/compact values are
            // Arc-backed, so this keeps family storage O(count) while VRAM
            // remains bounded instead of allocating count independent GPUs.
            let exemplar = Self::representative_runtime_value(backend, element, bindings)?;
            return Ok(RuntimeValue::IndexedFamily((0..*count).map(|_| exemplar.clone()).collect()));
        }
        match wire_type {
            ConcreteWireType::ConstantInt | ConcreteWireType::Int => {
                Ok(RuntimeValue::Int(BigInt::from(7)))
            }
            ConcreteWireType::ConstantReal | ConcreteWireType::Real => Ok(RuntimeValue::Real(1.25)),
            ConcreteWireType::ConstantBool | ConcreteWireType::Bool => Ok(RuntimeValue::Bool(true)),
            ConcreteWireType::Bytes { length } => Ok(RuntimeValue::Bytes(vec![0x5a; *length])),
            ConcreteWireType::TypedBlob { .. } => Ok(RuntimeValue::TypedBlob(vec![0x5a])),
            ConcreteWireType::Matrix(matrix) => backend
                .constant_matrix(matrix, &ConstantMatrix::Zero, bindings)
                .map(RuntimeValue::matrix)
                .map_err(|error| GpuMeasurementError(error.to_string())),
            ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } => {
                let owner = backend
                    .constant_matrix(matrix, &ConstantMatrix::Zero, bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let params = owner
                    .shards()
                    .first()
                    .ok_or_else(|| {
                        GpuMeasurementError("compact representative has no shard".into())
                    })?
                    .value
                    .params()
                    .clone();
                let bound = max_coefficient_bound.to_biguint().ok_or_else(|| {
                    GpuMeasurementError("compact representative bound must be nonnegative".into())
                })?;
                let magnitude_bytes =
                    usize::try_from(bound.bits().div_ceil(8)).unwrap_or(usize::MAX).max(1);
                let payload_len = matrix
                    .rows
                    .checked_mul(matrix.columns)
                    .and_then(|value| value.checked_mul(matrix.ring_dimension as usize))
                    .and_then(|value| value.checked_mul(1 + magnitude_bytes))
                    .ok_or_else(|| {
                        GpuMeasurementError("compact representative is too large".into())
                    })?;
                let value = GpuSmallMatrix::from_canonical_coefficients(
                    &params,
                    matrix.rows,
                    matrix.columns,
                    bound,
                    &vec![0u8; payload_len],
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
                Ok(RuntimeValue::small_matrix(GpuFleetSmallMatrix::from(value)))
            }
            ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                let owner = backend
                    .constant_matrix(matrix, &ConstantMatrix::Zero, bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let params = owner
                    .shards()
                    .first()
                    .ok_or_else(|| {
                        GpuMeasurementError("preimage representative has no shard".into())
                    })?
                    .value
                    .params()
                    .clone();
                let bound = max_coefficient_bound.to_biguint().ok_or_else(|| {
                    GpuMeasurementError("preimage representative bound must be nonnegative".into())
                })?;
                let magnitude_bytes =
                    usize::try_from(bound.bits().div_ceil(8)).unwrap_or(usize::MAX).max(1);
                let payload_len = matrix
                    .rows
                    .checked_mul(matrix.columns)
                    .and_then(|value| value.checked_mul(matrix.ring_dimension as usize))
                    .and_then(|value| value.checked_mul(1 + magnitude_bytes))
                    .ok_or_else(|| {
                        GpuMeasurementError("preimage representative is too large".into())
                    })?;
                let value = GpuSmallMatrix::from_canonical_coefficients(
                    &params,
                    matrix.rows,
                    matrix.columns,
                    bound,
                    &vec![0u8; payload_len],
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
                Ok(RuntimeValue::preimage(GpuFleetSmallMatrix::from(value)))
            }
            ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count, .. } => {
                let sigma = sigma
                    .evaluate_f64(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let (public, secret) = backend
                    .sample_trapdoor(matrix, sigma, gadget_base, *digit_count)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                Ok(RuntimeValue::Trapdoor {
                    secret: Some(Arc::new(secret)),
                    public: Arc::new(public),
                    matrix_type: matrix.clone(),
                    sigma,
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                    gadget_small: None,
                })
            }
            // Families are handled by the recursive fast path above.  Keep
            // the match exhaustive so adding a new family-shaped wire kind
            // cannot silently fall through to a scalar representative.
            ConcreteWireType::IndexedFamily { .. } => unreachable!("family handled above"),
        }
    }

    /// Build one payload that can stand for a family member during host-side
    /// container timing.  Host family access does not consume every member's
    /// backend value, so materializing a fresh matrix for each member would
    /// make setup VRAM scale with the family cardinality.  Nested families
    /// retain their *exact* container shape and cardinality; cloning the
    /// representative only clones the O(count) host vectors while the
    /// heavyweight matrix/compact owners remain Arc-backed and shared.
    fn representative_family_member(
        backend: &mut GpuDcrtBackend,
        wire_type: &ConcreteWireType,
        bindings: &ParamEnv,
    ) -> Result<RuntimeValue<GpuDcrtBackend>, GpuMeasurementError> {
        match wire_type {
            ConcreteWireType::IndexedFamily { element, count } => {
                if *count == 0 {
                    return Err(GpuMeasurementError(
                        "host family representative cannot use an empty nested family".into(),
                    ));
                }
                let exemplar = Self::representative_family_member(backend, element, bindings)?;
                Ok(RuntimeValue::IndexedFamily((0..*count).map(|_| exemplar.clone()).collect()))
            }
            _ => Self::representative_runtime_value(backend, wire_type, bindings),
        }
    }

    fn evaluated_family_count(wire_type: &ConcreteWireType) -> Result<usize, GpuMeasurementError> {
        let ConcreteWireType::IndexedFamily { count, .. } = wire_type else {
            return Err(GpuMeasurementError("host family input is not indexed".into()));
        };
        if *count == 0 {
            return Err(GpuMeasurementError(
                "host family representative cannot use an empty family".into(),
            ));
        }
        Ok(*count)
    }

    fn run_host_primitive(
        backend: &mut GpuDcrtBackend,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        batch_size: usize,
    ) -> Result<(), GpuMeasurementError> {
        if matches!(node.kind, NodeKind::Input { .. }) {
            let NodeKind::Input { name, .. } = node.kind else { unreachable!() };
            let output = node.concrete_output_types.first().ok_or_else(|| {
                GpuMeasurementError(format!("input node {:?} has no output type", node.id))
            })?;
            let value = Self::representative_runtime_value(backend, output, bindings)?;
            let inputs = BTreeMap::from([(name.clone(), value)]);
            return measure_typed_runtime_input(&inputs, name, output, batch_size)
                .map(|_| ())
                .map_err(|error| GpuMeasurementError(error.to_string()));
        }
        if matches!(node.kind, NodeKind::TrapdoorPublic) {
            let trapdoor = node.concrete_argument_types.first().ok_or_else(|| {
                GpuMeasurementError("TrapdoorPublic has no trapdoor input".into())
            })?;
            let value = Self::representative_runtime_value(backend, trapdoor, bindings)?;
            return measure_trapdoor_public(&value, batch_size)
                .map(|_| ())
                .map_err(|error| GpuMeasurementError(error.to_string()));
        }
        let inputs = match node.kind {
            NodeKind::IntBinary(_) | NodeKind::IntCompare(_) => {
                vec![
                    HostPrimitiveValue::Int(BigInt::from(7)),
                    HostPrimitiveValue::Int(BigInt::from(3)),
                ]
            }
            NodeKind::BitExtract { .. } | NodeKind::IntToReal => {
                vec![HostPrimitiveValue::Int(BigInt::from(7))]
            }
            NodeKind::BoolToInt => vec![HostPrimitiveValue::Bool(true)],
            NodeKind::RealBinary(_) => {
                vec![HostPrimitiveValue::Real(1.25), HostPrimitiveValue::Real(0.75)]
            }
            NodeKind::RealSqrt => vec![HostPrimitiveValue::Real(1.25)],
            _ => Vec::new(),
        };
        let elapsed = match node.kind {
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic |
            NodeKind::Select { .. } => {
                measure_host_container_primitive(node.id, node.kind, bindings, batch_size)
            }
            _ => measure_host_primitive(node.id, node.kind, bindings, &inputs, batch_size),
        };
        elapsed.map(|_| ()).map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn prepare_host_primitive(
        backend: &mut GpuDcrtBackend,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<PreparedHostPrimitive, GpuMeasurementError> {
        let mut prepared = PreparedHostPrimitive {
            typed_inputs: None,
            trapdoor: None,
            scalar_inputs: Vec::new(),
            family_inputs: Vec::new(),
            dynamic_index: None,
            choices: Vec::new(),
        };
        if let NodeKind::Input { name, .. } = node.kind {
            let output = node.concrete_output_types.first().ok_or_else(|| {
                GpuMeasurementError(format!("input node {:?} has no output type", node.id))
            })?;
            let value = Self::representative_runtime_value(backend, output, bindings)?;
            prepared.typed_inputs =
                Some((BTreeMap::from([(name.clone(), value)]), name.clone(), output.clone()));
            return Ok(prepared);
        }
        if matches!(node.kind, NodeKind::TrapdoorPublic) {
            let trapdoor = node.concrete_argument_types.first().ok_or_else(|| {
                GpuMeasurementError("TrapdoorPublic has no trapdoor input".into())
            })?;
            prepared.trapdoor =
                Some(Self::representative_runtime_value(backend, trapdoor, bindings)?);
            return Ok(prepared);
        }
        match node.kind {
            NodeKind::FamilyPack { .. } => {
                let member_types = if node.concrete_argument_types.is_empty() {
                    match node.concrete_output_types.first() {
                        Some(ConcreteWireType::IndexedFamily { element, count }) => {
                            vec![element.as_ref().clone(); *count]
                        }
                        _ => Vec::new(),
                    }
                } else {
                    node.concrete_argument_types.clone()
                };
                // FamilyPack's host work is intentionally still O(count): the
                // timed runtime helper clones this vector into the resulting
                // family.  Each member shares one bounded representative
                // payload rather than allocating an independent GPU object.
                if let Some(first) = member_types.first() {
                    let exemplar = Self::representative_family_member(backend, first, bindings)?;
                    prepared.family_inputs =
                        member_types.iter().map(|_| exemplar.clone()).collect();
                }
            }
            NodeKind::FamilyGetStatic { .. } | NodeKind::FamilyGetDynamic => {
                let family = node
                    .concrete_argument_types
                    .first()
                    .cloned()
                    .or_else(|| {
                        let output = node.concrete_output_types.first()?.clone();
                        let count = match node.kind {
                            NodeKind::FamilyGetStatic { index } => index
                                .evaluate(bindings)
                                .ok()
                                .and_then(|value| value.to_usize())
                                .and_then(|value| value.checked_add(1))
                                .unwrap_or(1),
                            NodeKind::FamilyGetDynamic => 1,
                            _ => 1,
                        };
                        Some(ConcreteWireType::IndexedFamily { element: Box::new(output), count })
                    })
                    .ok_or_else(|| {
                        GpuMeasurementError("family selection has no family input".into())
                    })?;
                let count = Self::evaluated_family_count(&family)?;
                let ConcreteWireType::IndexedFamily { element, .. } = &family else {
                    unreachable!("evaluated_family_count validated indexed family");
                };
                let exemplar = Self::representative_family_member(backend, element, bindings)?;
                let selected_index = match node.kind {
                    NodeKind::FamilyGetStatic { index } => index
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?
                        .to_usize()
                        .ok_or_else(|| {
                            GpuMeasurementError("family index does not fit usize".into())
                        })?,
                    NodeKind::FamilyGetDynamic => 0,
                    _ => unreachable!(),
                };
                if selected_index >= count {
                    return Err(GpuMeasurementError(format!(
                        "family representative index {selected_index} is out of range for {count} members"
                    )));
                }
                // Static access may select a nonzero member.  Preserve the
                // complete family shape: production clones/accesses a real
                // indexed container, and replacing unselected nested values
                // with scalar sentinels would measure a different operation.
                let family_values = (0..count).map(|_| exemplar.clone()).collect::<Vec<_>>();
                prepared.family_inputs.push(RuntimeValue::IndexedFamily(family_values));
                if matches!(node.kind, NodeKind::FamilyGetDynamic) {
                    // Index zero is valid for every nonempty family and is
                    // deliberately independent of the generic Int exemplar.
                    prepared.dynamic_index = Some(RuntimeValue::Int(BigInt::from(0u8)));
                }
            }
            NodeKind::Select { .. } => {
                let count = match node.kind {
                    NodeKind::Select { count } => count
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?
                        .to_usize()
                        .ok_or_else(|| {
                            GpuMeasurementError("select count does not fit usize".into())
                        })?,
                    _ => unreachable!(),
                };
                if count == 0 {
                    return Err(GpuMeasurementError(
                        "select representative cannot use zero choices".into(),
                    ));
                }
                // Zero is in range for every nonempty Select.  Do not use the
                // generic Int representative (7), which is invalid for the
                // common two-choice case.
                prepared.dynamic_index = Some(RuntimeValue::Int(BigInt::from(0u8)));
                let choice_types = if node.concrete_argument_types.len() > 1 {
                    node.concrete_argument_types.iter().skip(1).cloned().collect()
                } else {
                    let output = node
                        .concrete_output_types
                        .first()
                        .cloned()
                        .unwrap_or(ConcreteWireType::Int);
                    vec![output; count]
                };
                if choice_types.is_empty() {
                    return Err(GpuMeasurementError("select representative has no choices".into()));
                }
                // Preserve each choice's complete nested shape.  Reusing the
                // first representative for homogeneous choices keeps one GPU
                // owner; heterogeneous choices are materialized once per
                // distinct type and then cloned as O(count) host entries.
                let mut representatives =
                    HashMap::<ConcreteWireType, RuntimeValue<GpuDcrtBackend>>::new();
                prepared.choices = choice_types
                    .iter()
                    .take(count)
                    .map(|choice_type| {
                        if let Some(value) = representatives.get(choice_type) {
                            return Ok(value.clone());
                        }
                        let value =
                            Self::representative_family_member(backend, choice_type, bindings)?;
                        representatives.insert(choice_type.clone(), value.clone());
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                if prepared.choices.len() != count {
                    return Err(GpuMeasurementError(format!(
                        "select representative has {} choices for count {count}",
                        prepared.choices.len()
                    )));
                }
            }
            NodeKind::IntBinary(_) | NodeKind::IntCompare(_) => {
                prepared.scalar_inputs = vec![
                    HostPrimitiveValue::Int(BigInt::from(7)),
                    HostPrimitiveValue::Int(BigInt::from(3)),
                ];
            }
            NodeKind::BitExtract { .. } | NodeKind::IntToReal => {
                prepared.scalar_inputs = vec![HostPrimitiveValue::Int(BigInt::from(7))];
            }
            NodeKind::BoolToInt => {
                prepared.scalar_inputs = vec![HostPrimitiveValue::Bool(true)];
            }
            NodeKind::RealBinary(_) => {
                prepared.scalar_inputs =
                    vec![HostPrimitiveValue::Real(1.25), HostPrimitiveValue::Real(0.75)];
            }
            NodeKind::RealSqrt => {
                prepared.scalar_inputs = vec![HostPrimitiveValue::Real(1.25)];
            }
            _ => {}
        }
        Ok(prepared)
    }

    fn run_prepared_host_primitive(
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        prepared: &PreparedHostPrimitive,
    ) -> Result<(), GpuMeasurementError> {
        if let Some((inputs, name, expected)) = &prepared.typed_inputs {
            return measure_typed_runtime_input(inputs, name, expected, 1)
                .map(|_| ())
                .map_err(|error| GpuMeasurementError(error.to_string()));
        }
        if let Some(value) = &prepared.trapdoor {
            return measure_trapdoor_public(value, 1)
                .map(|_| ())
                .map_err(|error| GpuMeasurementError(error.to_string()));
        }
        if matches!(
            node.kind,
            NodeKind::FamilyPack { .. } |
                NodeKind::FamilyGetStatic { .. } |
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. }
        ) {
            return measure_runtime_container_primitive(
                node.id,
                node.kind,
                bindings,
                &prepared.family_inputs,
                prepared.dynamic_index.as_ref(),
                &prepared.choices,
                1,
            )
            .map(|_| ())
            .map_err(|error| GpuMeasurementError(error.to_string()));
        }
        measure_host_primitive(node.id, node.kind, bindings, &prepared.scalar_inputs, 1)
            .map(|_| ())
            .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn scalar_host_kind(kind: &NodeKind) -> bool {
        matches!(
            kind,
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
                NodeKind::FamilyPack { .. } |
                NodeKind::FamilyGetStatic { .. } |
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. } |
                NodeKind::TrapdoorPublic
        )
    }

    fn measure_scalar_host_repeated(
        worker: &mut GpuMeasurementWorker,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        harness: &MeasurementHarnessConfig,
    ) -> Result<(f64, f64, usize), GpuMeasurementError> {
        if harness.measured_iterations == 0 {
            return Err(GpuMeasurementError(
                "host warmup measurement requires at least one measured repetition".into(),
            ));
        }
        let prepared = Self::prepare_host_primitive(&mut worker.backend, node, bindings)?;
        for _ in 0..harness.warm_up_iterations {
            Self::run_prepared_host_primitive(node, bindings, &prepared)?;
        }
        let mut samples = Vec::with_capacity(harness.measured_iterations);
        for _ in 0..harness.measured_iterations {
            let started = std::time::Instant::now();
            Self::run_prepared_host_primitive(node, bindings, &prepared)?;
            samples.push(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE));
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let spread = if samples.len() > 1 {
            let variance =
                samples.iter().map(|sample| (sample - mean) * (sample - mean)).sum::<f64>() /
                    samples.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        Ok((mean, spread, samples.len()))
    }

    fn measure_host_boundary_repeated(
        worker: &mut GpuMeasurementWorker,
        config: &MeasurementHarnessConfig,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
    ) -> Result<(f64, f64, usize), GpuMeasurementError> {
        if config.measured_iterations == 0 {
            return Err(GpuMeasurementError(
                "host warmup measurement requires at least one measured repetition".into(),
            ));
        }
        let node = MeasurementNode {
            scope,
            id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        // Preparation is setup work.  In particular, a matrix upload made to
        // build the representative must never be charged to the host core.
        let prepared = Self::prepare(&mut worker.backend, &node, bindings, None)?;
        let matrix_type = representative
            .concrete_output_types
            .iter()
            .chain(&representative.concrete_argument_types)
            .find_map(|ty| ty.matrix_type())
            .ok_or_else(|| GpuMeasurementError("boundary has no matrix contract".into()))?;
        let downloaded = match node.kind {
            NodeKind::ExtractCoefficient { .. } |
            NodeKind::ThresholdDecode { .. } |
            NodeKind::PolynomialValues { .. } => {
                let value = prepared
                    .arguments
                    .first()
                    .and_then(Option::as_ref)
                    .ok_or_else(|| GpuMeasurementError("boundary input is missing".into()))?;
                worker
                    .backend
                    .download_polynomial_values(
                        value,
                        matches!(node.kind, NodeKind::PolynomialValues { evaluation: true }),
                    )
                    .map_err(|error| GpuMeasurementError(error.to_string()))?
            }
            _ => Vec::new(),
        };
        let evaluate = |value: &mxx_ir_core::IntExpr| {
            value.evaluate(bindings).map_err(|error| GpuMeasurementError(error.to_string()))
        };
        let as_usize = |value: &mxx_ir_core::IntExpr| {
            evaluate(value)?
                .to_usize()
                .ok_or_else(|| GpuMeasurementError("boundary dimension overflows".into()))
        };
        let run_host = || -> Result<(), GpuMeasurementError> {
            use mxx_runtime::backend::poly::{
                canonical_polynomial_values, extract_host_coefficient, pack_polynomial_bits,
                polynomial_host_values, threshold_decode_coefficients,
            };
            let error = |error: PolyBackendError| GpuMeasurementError(error.to_string());
            match node.kind {
                NodeKind::ExtractCoefficient { position, .. } => {
                    std::hint::black_box(
                        extract_host_coefficient(&downloaded, as_usize(position)?)
                            .map_err(error)?,
                    );
                }
                NodeKind::ThresholdDecode { plaintext_modulus, length, .. } => {
                    std::hint::black_box(threshold_decode_coefficients(
                        downloaded.clone(),
                        &matrix_type.modulus,
                        &evaluate(plaintext_modulus)?,
                        as_usize(length)?,
                    ));
                }
                NodeKind::PolynomialValues { .. } => {
                    std::hint::black_box(polynomial_host_values(downloaded.clone()));
                }
                NodeKind::PolynomialFromValues { .. } => {
                    std::hint::black_box(
                        canonical_polynomial_values(
                            matrix_type,
                            &vec![BigInt::from(0); matrix_type.ring_dimension],
                        )
                        .map_err(error)?,
                    );
                }
                NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
                    let bits = as_usize(coefficient_bits)?;
                    std::hint::black_box(
                        pack_polynomial_bits(
                            matrix_type,
                            &vec![false; matrix_type.ring_dimension * bits],
                            bits,
                        )
                        .map_err(error)?,
                    );
                }
                _ => return Err(GpuMeasurementError("not a host codec".into())),
            }
            Ok(())
        };
        for _ in 0..config.warm_up_iterations {
            run_host()?;
        }
        let mut samples = Vec::with_capacity(config.measured_iterations);
        for _ in 0..config.measured_iterations {
            let started = std::time::Instant::now();
            run_host()?;
            samples.push(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE));
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let spread = if samples.len() > 1 {
            let variance =
                samples.iter().map(|sample| (sample - mean) * (sample - mean)).sum::<f64>() /
                    samples.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        if !mean.is_finite() || mean <= 0.0 || !spread.is_finite() {
            return Err(GpuMeasurementError(
                "host warmup timing must be finite and positive".into(),
            ));
        }
        Ok((mean, spread, samples.len()))
    }

    /// Measure one validated structural dispatch through the shared
    /// production/warmup walker.  The walker owns loop binding and child
    /// traversal; this callback only executes the leaf host primitive.
    fn measure_host_control_once(
        _scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        kind: &NodeKind,
        bindings: &ParamEnv,
        child: Option<&HostControlChild>,
    ) -> Result<f64, GpuMeasurementError> {
        measure_host_control_with(id, kind, bindings, child, |invocation| {
            // Structural control walkers do not own the GPU backend; their
            // concrete leaf timing is charged by the enclosing production
            // node path. Keep traversal here side-effect free.
            let _ = invocation;
            Ok::<_, GpuMeasurementError>(HostControlBodyAction::VisitChildren)
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn measure_host_control_repeated(
        config: &MeasurementHarnessConfig,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        kind: &NodeKind,
        bindings: &ParamEnv,
        child: Option<&HostControlChild>,
    ) -> Result<(f64, f64, usize), GpuMeasurementError> {
        if config.measured_iterations == 0 {
            return Err(GpuMeasurementError(
                "host warmup measurement requires at least one measured repetition".into(),
            ));
        }
        for _ in 0..config.warm_up_iterations {
            std::hint::black_box(Self::measure_host_control_once(
                scope, id, kind, bindings, child,
            )?);
        }
        let mut samples = Vec::with_capacity(config.measured_iterations);
        for _ in 0..config.measured_iterations {
            samples.push(Self::measure_host_control_once(scope, id, kind, bindings, child)?);
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let spread = if samples.len() > 1 {
            let variance =
                samples.iter().map(|sample| (sample - mean) * (sample - mean)).sum::<f64>() /
                    samples.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        if !mean.is_finite() || mean <= 0.0 || !spread.is_finite() {
            return Err(GpuMeasurementError(
                "host warmup timing must be finite and positive".into(),
            ));
        }
        Ok((mean, spread, samples.len()))
    }

    fn run_node(
        backend: &mut GpuDcrtBackend,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        batch_size: usize,
        prepared: &PreparedMeasurement,
        output_range: Option<&IndexRange>,
    ) -> Result<Vec<GpuMeasurementOutput>, GpuMeasurementError> {
        let matrix_arc = |index: usize| {
            prepared.arguments.get(index).and_then(Option::as_ref).cloned().ok_or_else(|| {
                GpuMeasurementError(format!("node {:?} argument {index} is not a matrix", node.id))
            })
        };
        let matrix = |index: usize| {
            prepared.arguments.get(index).and_then(Option::as_ref).map(Arc::as_ref).ok_or_else(
                || {
                    GpuMeasurementError(format!(
                        "node {:?} argument {index} is not a matrix",
                        node.id
                    ))
                },
            )
        };
        let small_matrix_arc = |index: usize| {
            prepared.small_arguments.get(index).and_then(Option::as_ref).cloned().ok_or_else(|| {
                GpuMeasurementError(format!(
                    "node {:?} argument {index} is not a compact matrix",
                    node.id
                ))
            })
        };
        let output_matrix_type = || {
            node.concrete_output_types
                .iter()
                .find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix.clone()),
                    ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix.clone()),
                    _ => None,
                })
                .ok_or_else(|| {
                    GpuMeasurementError(format!("node {:?} has no matrix output", node.id))
                })
        };
        // Concrete types intentionally remain the full semantic graph types.
        // The inclusive fleet job below materializes only the requested global
        // range, so a local fragment is expected to have fewer columns than
        // `output_matrix_type()`.  Rejecting that shape here would measure a
        // synthetic full-width operation instead of the production tile.
        let evaluate_usize = |expression: &mxx_ir_core::IntExpr| {
            expression
                .evaluate(bindings)
                .map_err(|error| GpuMeasurementError(error.to_string()))?
                .to_usize()
                .ok_or_else(|| {
                    GpuMeasurementError("integer expression does not fit usize".to_owned())
                })
        };
        let backend_error = |error: <GpuDcrtBackend as Backend>::Error| {
            let message = error.to_string();
            if message.to_ascii_lowercase().contains("out of memory") ||
                message.to_ascii_lowercase().contains("cuda_error_out_of_memory")
            {
                GpuMeasurementError::out_of_memory(message)
            } else {
                GpuMeasurementError(message)
            }
        };
        let matrix_outputs = |outputs: Result<Vec<GpuFleetMatrix>, GpuMeasurementError>| {
            outputs.map(|outputs| outputs.into_iter().map(GpuMeasurementOutput::matrix).collect())
        };
        if matches!(
            node.kind,
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
                NodeKind::FamilyPack { .. } |
                NodeKind::FamilyGetStatic { .. } |
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. }
        ) {
            Self::run_host_primitive(backend, node, bindings, batch_size)?;
            return Ok(Vec::new());
        }
        match node.kind {
            NodeKind::ConstantMatrix { value, .. } => {
                let full_ty = output_matrix_type()?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            if let Some(range) = output_range {
                                backend
                                    .generated_constant_range_for_measurement(
                                        &full_ty,
                                        value,
                                        bindings,
                                        range.start,
                                        range.end,
                                    )
                                    .map_err(backend_error)
                            } else {
                                backend
                                    .constant_matrix(&full_ty, value, bindings)
                                    .map_err(backend_error)
                            }
                        })
                        .collect(),
                )
            }
            NodeKind::GadgetTrapdoor { base, .. } => {
                let (full_ty, digit_count) = node
                    .concrete_output_types
                    .iter()
                    .find_map(|wire_type| match wire_type {
                        ConcreteWireType::Trapdoor { matrix, digit_count, .. } => {
                            Some((matrix.clone(), *digit_count))
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        GpuMeasurementError(format!("node {:?} has no trapdoor output", node.id))
                    })?;
                let value = ConstantMatrix::Gadget { base: base.clone(), small: false };
                matrix_outputs(if let Some(range) = output_range {
                    let local_columns = range.end.checked_sub(range.start).ok_or_else(|| {
                        GpuMeasurementError(
                            "gadget trapdoor representative range is reversed".to_owned(),
                        )
                    })?;
                    if range.end > full_ty.columns {
                        return Err(GpuMeasurementError(
                            "gadget trapdoor representative range is outside its output".to_owned(),
                        ));
                    }
                    let gadget_base = base
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .measurement_gadget_columns(
                                    &full_ty,
                                    &gadget_base,
                                    digit_count,
                                    range.start,
                                    local_columns,
                                )
                                .map_err(backend_error)
                        })
                        .collect()
                } else {
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .constant_matrix(&full_ty, &value, bindings)
                                .map_err(backend_error)
                        })
                        .collect()
                })
            }
            NodeKind::MatrixBinary(operation) => {
                let inputs = (0..batch_size)
                    .map(|_| Ok((matrix_arc(0)?, matrix_arc(1)?)))
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                match operation {
                    MatrixBinaryOp::Add => backend.add_batch(inputs),
                    MatrixBinaryOp::Subtract => backend.sub_batch(inputs),
                    MatrixBinaryOp::Multiply => backend.multiply_batch(inputs),
                }
                .map_err(backend_error)
                .map(|outputs| outputs.into_iter().map(GpuMeasurementOutput::matrix).collect())
            }
            NodeKind::MatrixMulSmallRhs => {
                let lhs = matrix_arc(0)?;
                let rhs = small_matrix_arc(1)?;
                (0..batch_size)
                    .map(|_| {
                        backend
                            .multiply_small_rhs(lhs.as_ref(), rhs.as_ref())
                            .map(GpuMeasurementOutput::matrix)
                            .map_err(backend_error)
                    })
                    .collect()
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let mut requests = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    let mut products = Vec::with_capacity(coefficients.len());
                    for (product, coefficient) in coefficients.iter().enumerate() {
                        products.push((
                            coefficient
                                .evaluate(bindings)
                                .map_err(|error| GpuMeasurementError(error.to_string()))?,
                            matrix_arc(2 * product)?,
                            matrix_arc(2 * product + 1)?,
                        ));
                    }
                    let bias =
                        if *has_bias { Some(matrix_arc(2 * coefficients.len())?) } else { None };
                    requests.push(MatrixMulAccumulateRequest { products, bias });
                }
                backend
                    .matrix_mul_accumulate_batch(requests)
                    .map_err(backend_error)
                    .map(|outputs| outputs.into_iter().map(GpuMeasurementOutput::matrix).collect())
            }
            NodeKind::MatrixNegate => backend
                .negate_batch(
                    (0..batch_size).map(|_| matrix_arc(0)).collect::<Result<Vec<_>, _>>()?,
                )
                .map_err(backend_error)
                .map(|outputs| outputs.into_iter().map(GpuMeasurementOutput::matrix).collect()),
            NodeKind::MatrixScale { scalar } => {
                let scalar = scalar
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                backend
                    .scale_integer_batch(
                        (0..batch_size)
                            .map(|_| Ok((matrix_arc(0)?, scalar.clone())))
                            .collect::<Result<Vec<_>, GpuMeasurementError>>()?,
                    )
                    .map_err(backend_error)
                    .map(|outputs| outputs.into_iter().map(GpuMeasurementOutput::matrix).collect())
            }
            NodeKind::RnsModUp { source_moduli, digit_size, normalize, .. } => {
                let destination = output_matrix_type()?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .rns_mod_up(
                                    matrix(0)?,
                                    &destination,
                                    source_moduli,
                                    *digit_size,
                                    *normalize,
                                )
                                .map_err(backend_error)
                        })
                        .collect(),
                )
            }
            NodeKind::RnsModDown { source_moduli, plaintext_modulus, .. } => {
                let destination = output_matrix_type()?;
                let t = plaintext_modulus
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?
                    .to_u64()
                    .ok_or_else(|| {
                        GpuMeasurementError("plaintext modulus does not fit u64".into())
                    })?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .rns_mod_down(matrix(0)?, &destination, source_moduli, t)
                                .map_err(backend_error)
                        })
                        .collect(),
                )
            }
            NodeKind::BlockModSwitch { source_moduli, plaintext_modulus, .. } => {
                let destination = output_matrix_type()?;
                let t = plaintext_modulus
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .block_mod_switch(matrix(0)?, &destination, source_moduli, &t)
                                .map_err(backend_error)
                        })
                        .collect(),
                )
            }
            NodeKind::CenteredRebase { .. }
                if matches!(
                    node.concrete_argument_types.first(),
                    Some(ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. })
                ) =>
            {
                let destination = output_matrix_type()?;
                (0..batch_size)
                    .map(|_| {
                        backend
                            .centered_rebase_small(small_matrix_arc(0)?.as_ref(), &destination)
                            .map(GpuMeasurementOutput::SmallMatrix)
                            .map_err(backend_error)
                    })
                    .collect()
            }
            NodeKind::ModulusSwitch { .. } |
            NodeKind::ModulusReduce { .. } |
            NodeKind::CenteredRebase { .. } => {
                let destination = output_matrix_type()?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            let source = matrix(0)?;
                            if matches!(node.kind, NodeKind::ModulusSwitch { .. }) {
                                backend.modulus_switch(source, &destination)
                            } else if matches!(node.kind, NodeKind::CenteredRebase { .. }) {
                                backend.centered_rebase(source, &destination)
                            } else {
                                backend.reduce_modulus(source, &destination)
                            }
                            .map_err(backend_error)
                        })
                        .collect(),
                )
            }
            NodeKind::RingAutomorphism { index } => {
                let index = evaluate_usize(index)?;
                backend
                    .ring_automorphism_batch(
                        (0..batch_size).map(|_| Ok((matrix_arc(0)?, index))).collect::<Result<
                            Vec<_>,
                            GpuMeasurementError,
                        >>(
                        )?,
                    )
                    .map_err(backend_error)
                    .map(|outputs| outputs.into_iter().map(GpuMeasurementOutput::matrix).collect())
            }
            NodeKind::Transpose => matrix_outputs(
                (0..batch_size)
                    .map(|_| {
                        if let Some(range) = output_range {
                            backend
                                .transpose_range_for_measurement(matrix(0)?, range.start, range.end)
                                .map_err(backend_error)
                        } else {
                            backend.transpose(matrix(0)?).map_err(backend_error)
                        }
                    })
                    .collect(),
            ),
            NodeKind::Slice { rows, columns } => {
                let rows = rows
                    .as_ref()
                    .map(|range| {
                        Ok(IndexRange {
                            start: evaluate_usize(&range.start)?,
                            end: evaluate_usize(&range.end)?,
                        })
                    })
                    .transpose()?;
                let columns = columns
                    .as_ref()
                    .map(|range| {
                        Ok(IndexRange {
                            start: evaluate_usize(&range.start)?,
                            end: evaluate_usize(&range.end)?,
                        })
                    })
                    .transpose()?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            if let Some(range) = output_range {
                                backend
                                    .slice_range_for_measurement(
                                        matrix(0)?,
                                        rows.clone(),
                                        columns.clone(),
                                        range.start,
                                        range.end,
                                    )
                                    .map_err(backend_error)
                            } else {
                                backend
                                    .slice(matrix(0)?, rows.as_ref(), columns.as_ref())
                                    .map_err(backend_error)
                            }
                        })
                        .collect(),
                )
            }
            NodeKind::Tensor => matrix_outputs(if let Some(range) = output_range {
                (0..batch_size)
                    .map(|_| {
                        backend
                            .tensor_range_for_measurement(
                                matrix(0)?,
                                matrix(1)?,
                                range.start,
                                range.end,
                            )
                            .map_err(backend_error)
                    })
                    .collect()
            } else {
                (0..batch_size)
                    .map(|_| backend.tensor(matrix(0)?, matrix(1)?).map_err(backend_error))
                    .collect()
            }),
            NodeKind::Concat { axis } => {
                Self::verify_concat_representative(node, *axis, output_range)?;
                let inputs = prepared
                    .arguments
                    .iter()
                    .map(|value| {
                        value.as_ref().map(Arc::as_ref).ok_or_else(|| {
                            GpuMeasurementError("concat argument is not a matrix".to_owned())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                matrix_outputs(if let Some(range) = output_range {
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .concat_range_for_measurement(
                                    &inputs,
                                    *axis,
                                    range.start,
                                    range.end,
                                )
                                .map_err(backend_error)
                        })
                        .collect()
                } else {
                    (0..batch_size)
                        .map(|_| backend.concat(&inputs, *axis).map_err(backend_error))
                        .collect()
                })
            }
            NodeKind::UniformResidueSample { .. } => {
                let ty = output_matrix_type()?;
                let range = SampleRange {
                    minimum: BigInt::from(0),
                    maximum: &ty.modulus - BigInt::from(1),
                };
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            if let Some(output) = output_range {
                                backend
                                    .uniform_range_for_measurement(
                                        &ty,
                                        &range,
                                        output.start,
                                        output.end,
                                    )
                                    .map_err(backend_error)
                            } else {
                                backend.sample_uniform(&ty, &range).map_err(backend_error)
                            }
                        })
                        .collect(),
                )
            }
            NodeKind::UniformIntervalSample { range, .. } => {
                let ty = output_matrix_type()?;
                let range = SampleRange {
                    minimum: range
                        .minimum
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?,
                    maximum: range
                        .maximum
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?,
                };
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            if let Some(output) = output_range {
                                backend
                                    .uniform_range_for_measurement(
                                        &ty,
                                        &range,
                                        output.start,
                                        output.end,
                                    )
                                    .map_err(backend_error)
                            } else {
                                backend.sample_uniform(&ty, &range).map_err(backend_error)
                            }
                        })
                        .collect(),
                )
            }
            NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                let ty = output_matrix_type()?;
                let sigma = sigma
                    .evaluate_f64(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let max_coefficient_bound = max_coefficient_bound
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            if let Some(output) = output_range {
                                backend
                                    .gaussian_range_for_measurement(
                                        &ty,
                                        sigma,
                                        &max_coefficient_bound,
                                        output.start,
                                        output.end,
                                    )
                                    .map_err(backend_error)
                            } else {
                                backend
                                    .sample_gaussian(&ty, sigma, &max_coefficient_bound)
                                    .map_err(backend_error)
                            }
                        })
                        .collect(),
                )
            }
            NodeKind::HashSample { variant, tag_prefix, base, digit_count, .. } => {
                let ty = output_matrix_type()?;
                let gadget_base = base
                    .as_ref()
                    .map(|base| {
                        base.evaluate(bindings)
                            .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .transpose()?;
                let digit_count = digit_count.as_ref().map(evaluate_usize).transpose()?;
                match (variant, gadget_base.as_ref(), digit_count) {
                    (mxx_ir_core::node::HashVariant::Plain, None, None) => matrix_outputs(
                        (0..batch_size)
                            .map(|_| {
                                if let Some(output) = output_range {
                                    backend
                                        .hash_range_for_measurement(
                                            &ty,
                                            [0x53; 32],
                                            tag_prefix,
                                            output.start,
                                            output.end,
                                        )
                                        .map_err(backend_error)
                                } else {
                                    backend
                                        .sample_hash(&ty, [0x53; 32], tag_prefix)
                                        .map_err(backend_error)
                                }
                            })
                            .collect(),
                    ),
                    (mxx_ir_core::node::HashVariant::Decomposed, Some(base), Some(count)) |
                    (mxx_ir_core::node::HashVariant::SmallDecomposed, Some(base), Some(count)) => {
                        let small = *variant == mxx_ir_core::node::HashVariant::SmallDecomposed;
                        (0..batch_size)
                            .map(|_| {
                                if let Some(output) = output_range {
                                    backend
                                        .hash_decomposed_range_for_measurement(
                                            &ty,
                                            [0x53; 32],
                                            tag_prefix,
                                            base,
                                            count,
                                            small,
                                            output.start,
                                            output.end,
                                        )
                                        .map(GpuMeasurementOutput::SmallMatrix)
                                        .map_err(backend_error)
                                } else if small {
                                    backend
                                        .sample_hash_small_decomposed(
                                            &ty, [0x53; 32], tag_prefix, base, count,
                                        )
                                        .map(GpuMeasurementOutput::SmallMatrix)
                                        .map_err(backend_error)
                                } else {
                                    backend
                                        .sample_hash_decomposed(
                                            &ty, [0x53; 32], tag_prefix, base, count,
                                        )
                                        .map(GpuMeasurementOutput::SmallMatrix)
                                        .map_err(backend_error)
                                }
                            })
                            .collect()
                    }
                    _ => Err(GpuMeasurementError(
                        "hash variant and gadget layout do not match".to_owned(),
                    )),
                }
            }
            NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } => {
                let ty = output_matrix_type()?;
                let sigma = sigma
                    .evaluate_f64(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let gadget_base = gadget_base
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let digit_count = evaluate_usize(digit_count)?;
                let mut outputs = Vec::with_capacity(batch_size * 2);
                for _ in 0..batch_size {
                    let (public, trapdoor) = backend
                        .sample_trapdoor(&ty, sigma, &gadget_base, digit_count)
                        .map_err(backend_error)?;
                    outputs.push(GpuMeasurementOutput::Matrix(public));
                    outputs.push(GpuMeasurementOutput::Trapdoor(trapdoor));
                }
                Ok(outputs)
            }
            NodeKind::PreimageSample { .. } => {
                let ty = output_matrix_type()?;
                let (public, trapdoor, sigma, gadget_base, digit_count, max_coefficient_bound) =
                    prepared.preimage_trapdoor.as_ref().ok_or_else(|| {
                        GpuMeasurementError("missing prepared trapdoor".to_owned())
                    })?;
                let target = prepared.preimage_target.as_ref().ok_or_else(|| {
                    GpuMeasurementError("missing prepared preimage target".to_owned())
                })?;
                if batch_size == 1 {
                    backend
                        .sample_preimage(
                            &ty,
                            *sigma,
                            gadget_base,
                            *digit_count,
                            max_coefficient_bound,
                            trapdoor,
                            public,
                            target.as_ref(),
                            [0x50; 32],
                        )
                        .map(|output| vec![GpuMeasurementOutput::SmallMatrix(output)])
                        .map_err(backend_error)
                } else {
                    backend
                        .sample_preimage_batch(
                            (0..batch_size)
                                .enumerate()
                                .map(|(index, _)| {
                                    let mut randomness_seed = [0x50; 32];
                                    randomness_seed[..size_of::<usize>()]
                                        .copy_from_slice(&index.to_le_bytes());
                                    PreimageRequest {
                                        fixed_metadata: None,
                                        instance_slot: index,
                                        matrix_type: ty.clone(),
                                        sigma: *sigma,
                                        gadget_base: gadget_base.clone(),
                                        digit_count: *digit_count,
                                        max_coefficient_bound: max_coefficient_bound.clone(),
                                        trapdoor: Arc::new(trapdoor.clone()),
                                        public: Arc::new(public.clone()),
                                        target: target.clone(),
                                        randomness_seed,
                                    }
                                })
                                .collect(),
                        )
                        .map(|outputs| {
                            outputs.into_iter().map(GpuMeasurementOutput::SmallMatrix).collect()
                        })
                        .map_err(backend_error)
                }
            }
            NodeKind::GadgetDecompose { small, digit_count, .. } => (0..batch_size)
                .map(|_| {
                    backend
                        .gadget_decompose(matrix(0)?, *small, Some(evaluate_usize(digit_count)?))
                        .map(GpuMeasurementOutput::SmallMatrix)
                        .map_err(backend_error)
                })
                .collect(),
            NodeKind::ExtractCoefficient { position, .. } => {
                let position = evaluate_usize(position)?;
                for _ in 0..batch_size {
                    dispatch_extract_coefficient(backend, matrix(0)?, position)
                        .map_err(backend_error)?;
                }
                Ok(Vec::new())
            }
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type } => {
                let modulus = matrix_type
                    .modulus
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                if modulus <= BigInt::one() {
                    return Err(GpuMeasurementError(
                        "constant-polynomial lift matrix modulus must exceed one".to_owned(),
                    ));
                }
                let positive_dimension = |expression: &mxx_ir_core::IntExpr, label: &str| {
                    expression
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?
                        .to_usize()
                        .filter(|value| *value > 0)
                        .ok_or_else(|| {
                            GpuMeasurementError(format!(
                                "constant-polynomial lift matrix {label} must be a positive usize"
                            ))
                        })
                };
                let ty = ConcreteMatrixType {
                    modulus,
                    ring_dimension: positive_dimension(
                        &matrix_type.ring_dimension,
                        "ring dimension",
                    )?,
                    rows: positive_dimension(&matrix_type.rows, "rows")?,
                    columns: positive_dimension(&matrix_type.columns, "columns")?,
                };
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            if let Some(output) = output_range {
                                backend
                                    .lift_integer_to_constant_polynomial_range_for_measurement(
                                        &ty,
                                        &BigInt::from(0),
                                        output.start,
                                        output.end,
                                    )
                                    .map_err(backend_error)
                            } else {
                                let identity = backend
                                    .constant_matrix(
                                        &ty,
                                        &mxx_ir_core::node::ConstantMatrix::Identity,
                                        bindings,
                                    )
                                    .map_err(backend_error)?;
                                backend
                                    .scale_integer(&identity, &BigInt::from(0))
                                    .map_err(backend_error)
                            }
                        })
                        .collect(),
                )
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, .. } => {
                let modulus = plaintext_modulus
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let length = evaluate_usize(length)?;
                for _ in 0..batch_size {
                    dispatch_threshold_decode(backend, matrix(0)?, &modulus, length)
                        .map_err(backend_error)?;
                }
                Ok(Vec::new())
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients, .. } => {
                let destination = output_matrix_type()?;
                let levels = prepared
                    .arguments
                    .iter()
                    .map(|value| {
                        value.as_ref().map(|value| value.as_ref().clone()).ok_or_else(|| {
                            GpuMeasurementError("CRT argument is not a matrix".to_owned())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let plaintext_moduli = plaintext_moduli
                    .iter()
                    .map(|value| {
                        value
                            .evaluate(bindings)
                            .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let reconstruction_coefficients = reconstruction_coefficients
                    .iter()
                    .map(|value| {
                        value
                            .evaluate(bindings)
                            .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .crt_recompose(
                                    &levels,
                                    &plaintext_moduli,
                                    &reconstruction_coefficients,
                                    &destination,
                                )
                                .map_err(backend_error)
                        })
                        .collect(),
                )
            }
            NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
                let ty = output_matrix_type()?;
                let coefficient_bits = evaluate_usize(coefficient_bits)?;
                let count = match node.concrete_argument_types.first() {
                    Some(ConcreteWireType::IndexedFamily { count, .. }) => *count,
                    _ => {
                        return Err(GpuMeasurementError(
                            "packed coefficient input is not a family".to_owned(),
                        ));
                    }
                };
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            // Keep the representative away from the
                            // all-zero fast path. The family shape is still
                            // taken from validated metadata; only its bit
                            // payload is chosen to exercise the production
                            // packing loop.
                            let bits = (0..count).map(|index| index % 2 == 0).collect::<Vec<_>>();
                            dispatch_pack_polynomial_coefficients(
                                backend,
                                &ty,
                                &bits,
                                coefficient_bits,
                            )
                            .map_err(backend_error)
                        })
                        .collect(),
                )
            }
            NodeKind::PolynomialFromValues { evaluation, .. } => {
                let ty = output_matrix_type()?;
                let count = match node.concrete_argument_types.first() {
                    Some(ConcreteWireType::IndexedFamily { count, .. }) => *count,
                    _ => {
                        return Err(GpuMeasurementError(
                            "polynomial value input is not an indexed family".to_owned(),
                        ));
                    }
                };
                let values = (0..count).map(|index| BigInt::from(index as u64)).collect::<Vec<_>>();
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            dispatch_polynomial_from_values(backend, &ty, &values, *evaluation)
                                .map_err(backend_error)
                        })
                        .collect(),
                )
            }
            NodeKind::PolynomialValues { evaluation } => {
                let value = matrix(0)?;
                for _ in 0..batch_size {
                    let values = dispatch_polynomial_values(backend, value, *evaluation)
                        .map_err(backend_error)?;
                    std::hint::black_box(values);
                }
                Ok(Vec::new())
            }
            NodeKind::TrapdoorPublic => {
                Self::run_host_primitive(backend, node, bindings, batch_size)?;
                Ok(Vec::new())
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
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic |
            NodeKind::Select { .. } => Ok(Vec::new()),
            NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_) => {
                measure_host_control_with(node.id, node.kind, bindings, None, |_invocation| {
                    Ok::<_, GpuMeasurementError>(HostControlBodyAction::VisitChildren)
                })
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
                Ok(Vec::new())
            }
        }
    }
}

impl GpuWarmupProfileProvider for GpuNodeMeasurementBackend {
    fn register_operation(
        &mut self,
        descriptor: GpuWarmupOperationDescriptor,
    ) -> Result<(), GpuWarmupProfileError> {
        let signature = descriptor.signature;
        let profile_domain = descriptor.profile_domain;
        let fused_operation = descriptor.fused_operation;
        let implementation_variant = descriptor.implementation_variant.clone();
        if !profile_domain.is_profileable() {
            return Err(GpuWarmupProfileError::InvalidMeasurement(format!(
                "operation descriptor uses a non-profileable warmup domain {profile_domain:?}"
            )));
        }
        if fused_operation.is_none() {
            let expected = canonical_warmup_profile_domain(&descriptor.kind);
            if profile_domain != expected {
                return Err(GpuWarmupProfileError::InvalidMeasurement(format!(
                    "ordinary operation {:?} has profile domain {profile_domain:?}, expected {expected:?}",
                    descriptor.kind
                )));
            }
        }
        if let Some(fused) = fused_operation {
            let expected = fused_warmup_profile_domain(fused);
            if profile_domain != expected {
                return Err(GpuWarmupProfileError::InvalidMeasurement(format!(
                    "fused operation {fused:?} has profile domain {profile_domain:?}, expected {expected:?}"
                )));
            }
            let prefix = format!("{}:", expected.identity());
            if !implementation_variant.starts_with(&prefix) {
                return Err(GpuWarmupProfileError::InvalidMeasurement(format!(
                    "fused implementation variant {implementation_variant:?} does not match profile domain {}",
                    expected.identity()
                )));
            }
        }
        // Reject a reused signature whose effective lowering contract differs.
        // Reusing an ordinary-node profile for a fused site would make setup
        // time/workspace predictions unsound even when the IR node kind is
        // identical.  The runtime cache performs the same check, but the
        // provider must fail closed on its own public registration boundary.
        if let Some(previous) = self.warmup_profile_domains.get(&signature) {
            if *previous != profile_domain {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "warmup signature was registered with multiple profile domains".into(),
                ));
            }
            if self.warmup_fused_operations.get(&signature).copied().flatten() != fused_operation {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "warmup signature was registered with multiple fused operations".into(),
                ));
            }
            if self
                .warmup_implementation_variants
                .get(&signature)
                .is_some_and(|variant| variant != &implementation_variant)
            {
                return Err(GpuWarmupProfileError::InvalidMeasurement(
                    "warmup signature was registered with multiple implementation variants".into(),
                ));
            }
        }
        // A PreimageSample IR node backed by GadgetTrapdoor is lowered to
        // fixed gadget decomposition and must not inherit sampler-specific
        // retry/cache admission or wave accounting. Only the sampled
        // trapdoor lowering is a preimage sampler.
        let preimage_sample = fused_operation == Some(FusedWarmupOperation::PreimageBatch);
        if let Some(host_control) = descriptor.host_control.clone() {
            self.host_control_operations
                .insert((descriptor.scope.clone(), descriptor.node), host_control);
        }
        let pending = PendingMeasurement {
            warmup: Some(descriptor.clone()),
            key: signature.operation,
            scope: descriptor.scope,
            id: descriptor.node,
            kind: descriptor.kind,
            concrete_argument_types: descriptor.concrete_argument_types,
            concrete_output_types: descriptor.concrete_output_types,
            bindings: descriptor.bindings,
            preimage_sample,
        };
        self.warmup_operations.insert(signature, pending);
        self.warmup_profile_domains.insert(signature, profile_domain);
        self.warmup_fused_operations.insert(signature, fused_operation);
        self.warmup_implementation_variants.insert(signature, implementation_variant);
        Ok(())
    }

    fn measure(
        &mut self,
        request: &GpuWarmupProfileRequest,
    ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
        self.measure_warmup_profile(request)
    }
}

impl MeasurementBackend for GpuNodeMeasurementBackend {
    type Error = GpuMeasurementError;

    fn measure(
        &mut self,
        _graph: &str,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<NodeMeasurement, Self::Error> {
        let measurement_key = Self::measurement_key(node, bindings)?;
        if let Some(measurement) = self.measurements.get(&measurement_key) {
            return Ok(measurement.clone());
        }
        if !self.collecting {
            return Err(GpuMeasurementError(format!(
                "GPU node shape at {:?} node {:?} was not collected before measurement",
                node.scope, node.id
            )));
        }
        self.pending.entry(measurement_key).or_insert_with(|| PendingMeasurement {
            warmup: None,
            key: measurement_key,
            scope: node.scope.clone(),
            id: node.id,
            kind: node.kind.clone(),
            concrete_argument_types: node.concrete_argument_types.clone(),
            concrete_output_types: node.concrete_output_types.clone(),
            bindings: bindings.clone(),
            preimage_sample: matches!(node.kind, NodeKind::PreimageSample { .. }),
        });
        Ok(NodeMeasurement::default())
    }

    fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
        match wire_type {
            ConcreteWireType::Matrix(matrix) | ConcreteWireType::Trapdoor { matrix, .. } => {
                // Missing ring metadata cannot be represented as a cheap zero-byte
                // buffer. Actual execution will report the registry error.
                self.workers
                    .first()
                    .and_then(|worker| worker.backend.ring_crt_depth(matrix).ok())
                    .map_or(u64::MAX, |depth| matrix_bytes(matrix, depth))
            }
            ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
            ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                compact_matrix_bytes(matrix, max_coefficient_bound)
            }
            ConcreteWireType::IndexedFamily { element, count } => self
                .persistent_bytes(element)
                .saturating_mul(u64::try_from(*count).unwrap_or(u64::MAX)),
            ConcreteWireType::Bytes { length } => u64::try_from(*length).unwrap_or(u64::MAX),
            ConcreteWireType::TypedBlob { .. } => 0,
            ConcreteWireType::ConstantInt |
            ConcreteWireType::ConstantReal |
            ConcreteWireType::ConstantBool |
            ConcreteWireType::Int |
            ConcreteWireType::Real |
            ConcreteWireType::Bool => 0,
        }
    }
    fn persistent_bytes_for_node(&self, kind: &NodeKind, wire_type: &ConcreteWireType) -> u64 {
        if matches!(kind, NodeKind::ParallelLoop(_)) &&
            matches!(
                wire_type,
                ConcreteWireType::IndexedFamily { element, .. }
                    if ArtifactType::from_wire_type(element).is_some()
            )
        {
            // Runtime stages each artifact-compatible lane after its bounded loop wave. The
            // live lane values are accounted in the child peak, not as one resident family.
            return 0;
        }
        if matches!(kind, NodeKind::Input { artifact: Some(_), .. }) &&
            matches!(wire_type, ConcreteWireType::IndexedFamily { .. })
        {
            // Artifact families remain store-backed descriptors; consumers materialize only
            // their selected members.
            0
        } else {
            self.persistent_bytes(wire_type)
        }
    }

    fn persistent_storage_bytes_for_node(
        &self,
        _kind: &NodeKind,
        wire_type: &ConcreteWireType,
    ) -> u64 {
        match wire_type {
            ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
            ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                compact_artifact_bytes(matrix, max_coefficient_bound)
            }
            _ => self.persistent_bytes(wire_type),
        }
    }

    fn transmitted_bytes_for_node(&self, kind: &NodeKind, wire_type: &ConcreteWireType) -> u64 {
        match kind {
            NodeKind::Input { artifact: Some(artifact), .. }
                if artifact.availability == ArtifactAvailability::Cached =>
            {
                0
            }
            NodeKind::Input { .. } => self.persistent_storage_bytes_for_node(kind, wire_type),
            _ => 0,
        }
    }

    fn cache_bytes_for_node(&self, kind: &NodeKind, wire_type: &ConcreteWireType) -> u64 {
        match kind {
            NodeKind::Input { artifact: Some(artifact), .. }
                if artifact.availability == ArtifactAvailability::Cached =>
            {
                self.persistent_storage_bytes_for_node(kind, wire_type)
            }
            _ => 0,
        }
    }

    fn artifact_payload_bytes_for_node(
        &self,
        _kind: &NodeKind,
        _wire_type: &ConcreteWireType,
        manifest_payload_bytes: Option<u64>,
    ) -> u64 {
        // Production artifact inputs are rejected by the estimator before
        // reaching this hook when no exact store-captured size is available.
        // Keeping this hook total preserves the measurement-backend trait for
        // non-artifact synthetic inputs without reintroducing a resident-byte
        // or modulus-bit approximation as production authority.
        manifest_payload_bytes.unwrap_or(0)
    }

    fn persistent_alias_argument(&self, kind: &NodeKind, output_port: usize) -> Option<usize> {
        (output_port == 0 && matches!(kind, NodeKind::GadgetDecompose { .. })).then_some(0)
    }
}

fn matrix_bytes(matrix: &ConcreteMatrixType, crt_depth: usize) -> u64 {
    u64::try_from(matrix.rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(matrix.columns).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(matrix.ring_dimension).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(crt_depth).unwrap_or(u64::MAX))
        .saturating_mul(8)
}

fn compact_matrix_bytes(matrix: &ConcreteMatrixType, max_coefficient_bound: &BigInt) -> u64 {
    u64::try_from(compact_matrix_bytes_u128(matrix, max_coefficient_bound)).unwrap_or(u64::MAX)
}

fn compact_matrix_bytes_u128(matrix: &ConcreteMatrixType, max_coefficient_bound: &BigInt) -> u128 {
    let magnitude_bytes = max_coefficient_bound
        .to_biguint()
        .map(|bound| u128::from(bound.bits().div_ceil(8).max(1)))
        .unwrap_or(u128::MAX);
    (matrix.rows as u128)
        .saturating_mul(matrix.columns as u128)
        .saturating_mul(matrix.ring_dimension as u128)
        .saturating_mul(1u128.saturating_add(magnitude_bytes))
}

fn compact_artifact_bytes(matrix: &ConcreteMatrixType, max_coefficient_bound: &BigInt) -> u64 {
    let bound_bytes = max_coefficient_bound
        .to_biguint()
        .map(|bound| bound.to_bytes_le().len().max(1))
        .unwrap_or(0);
    let framing = 4usize
        .saturating_add(1)
        .saturating_add(8)
        .saturating_add(8)
        .saturating_add(8)
        .saturating_add(4)
        .saturating_add(bound_bytes)
        .saturating_add(4)
        .saturating_add(8);
    compact_matrix_bytes(matrix, max_coefficient_bound)
        .saturating_add(u64::try_from(framing).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::{
        GpuMeasurementError, GpuMeasurementOutput, GpuNodeMeasurementBackend, PendingMeasurement,
        PreparedMeasurement, aggregate_fleet_wave, authoritative_production_route,
        compact_matrix_bytes, extrapolate_fleet_waves, gpu_capped_waterfill_columns, matrix_bytes,
        require_exclusive_measurement_context, select_worker_index_for_physical,
    };
    use crate::{MeasurementNode, NodeMeasurement, harness::MeasurementHarnessConfig};
    use mxx_ir_core::{
        FrozenGraphScopeId, IntExpr, ParamEnv, RealExpr,
        node::{ConcatAxis, ConstantMatrix, HashVariant, IndexRange, MatrixBinaryOp, NodeKind},
        types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId},
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams},
        sampler::trapdoor::gpu::{
            PreimageAllocationEnvelope, PreimageAllocationEvidence, PreimageAllocationEvidenceKind,
            PreimageCacheIdentity, PreimageCacheState, PreimageEvidenceContext,
            PreimageFormatContext,
        },
    };
    use mxx_runtime::{
        backend::{
            GpuWarmupCacheState, GpuWarmupDeviceIdentity, GpuWarmupFragmentClass,
            GpuWarmupOperationDescriptor, GpuWarmupOperationSignature, GpuWarmupProfileProvider,
            GpuWarmupProfileRequest, GpuWarmupProvenance, GpuWarmupTimingScope,
            IndexRange as RuntimeIndexRange, MemoryEvidenceKind, poly::PolyBackendError,
            poly_gpu::gpu_backend,
        },
        gpu_calibration::GpuColumnWidths,
        gpu_column_policy::{
            CanonicalWarmupProfileDomain, ColumnRange, FusedWarmupOperation,
            GpuExecutionRouteDescriptor, GpuFragmentClass, GpuRouteResolutionInput,
            GpuTransferRoute, WarmupMeasurementKind, resolve_gpu_route,
        },
        gpu_execution_plan::{
            FrozenGpuPlan, GpuDeviceBudget, GpuExecutionSiteKey, GpuLayout, GpuLoopChoice,
            GpuLoopSiteKey, GpuNodeChoice, GpuPlanContract, LayoutId,
        },
        gpu_schedule::GpuColumnInterval,
        gpu_warmup::{GpuResourceCost, GpuTimeModel},
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeSet;

    #[test]
    fn worker_selection_prefers_explicit_owner_in_replicated_full_fleet() {
        let worker_devices = [0, 1];
        let replicated_fleet = [vec![0, 1], vec![0, 1]];
        let select = |physical| {
            select_worker_index_for_physical(
                worker_devices.len(),
                physical,
                |index| worker_devices[index],
                |index| replicated_fleet[index].contains(&physical),
            )
        };

        assert_eq!(select(0), Some(0));
        assert_eq!(select(1), Some(1));

        // Shared-backend configurations have no exact worker owner and still
        // use containment as the compatibility fallback.
        let shared_devices = [7, 8];
        let shared_fleet = [vec![0, 1], vec![0, 1]];
        assert_eq!(
            select_worker_index_for_physical(
                shared_devices.len(),
                1,
                |index| shared_devices[index],
                |index| shared_fleet[index].contains(&1),
            ),
            Some(0)
        );
    }

    #[test]
    fn canonical_profile_inventory_is_closed_and_provider_rejects_domain_drift() {
        use mxx_runtime::gpu_column_policy::{
            CanonicalWarmupProfileDomain, WarmupMeasurementKind, WarmupTransferKind,
        };

        let domains = CanonicalWarmupProfileDomain::all();
        let identities = domains.iter().map(|domain| domain.identity()).collect::<BTreeSet<_>>();
        assert_eq!(identities.len(), domains.len(), "canonical profile identity collision");
        assert!(domains.iter().all(|domain| !domain.identity().is_empty()));
        assert!(
            domains
                .iter()
                .filter(|domain| domain.transfer_kind() != WarmupTransferKind::None)
                .all(|domain| domain.measurement_kind() == WarmupMeasurementKind::HostMeasured)
        );

        let signature = GpuWarmupOperationSignature {
            operation: [0x6d; 32],
            shape_class: 1,
            instance_class: 0,
        };
        let kind = NodeKind::Input {
            name: "input".into(),
            wire_type: mxx_ir_core::types::WireType::Int,
            artifact: None,
        };
        let descriptor =
            |profile_domain: CanonicalWarmupProfileDomain| GpuWarmupOperationDescriptor {
                inputs: Default::default(),
                signature,
                scope: FrozenGraphScopeId::Root,
                node: NodeId(0),
                kind: kind.clone(),
                concrete_argument_types: Vec::new(),
                concrete_output_types: vec![ConcreteWireType::Int],
                bindings: ParamEnv::default(),
                effective_operation: profile_domain.identity().into(),
                profile_domain,
                fused_operation: None,
                implementation_variant: profile_domain.identity().into(),
                source_layouts: Vec::new(),
                output_layout: None,
                route_resolver: None,
                host_control: None,
            };
        let mut provider = GpuNodeMeasurementBackend::from_workers(
            Vec::new(),
            MeasurementHarnessConfig::default(),
            0,
        );
        provider
            .register_operation(descriptor(CanonicalWarmupProfileDomain::Input))
            .expect("matching ordinary domain must register");
        let error = provider
            .register_operation(descriptor(CanonicalWarmupProfileDomain::MatrixAdd))
            .expect_err("ordinary domain drift must be rejected at registration");
        assert!(error.to_string().contains("profile domain"));
    }

    #[test]
    fn production_route_response_is_authoritative_for_profile_identity() {
        let requested = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 4 },
            GpuFragmentClass::Mapped,
        );
        let observed = vec![
            resolve_gpu_route(GpuRouteResolutionInput {
                source_device: Some(1),
                destination_device: Some(0),
                source_range: ColumnRange { start: 0, end: 2 },
                destination_range: ColumnRange { start: 0, end: 2 },
                source_is_resident: false,
                peer_available: false,
                source_compact: false,
                destination_compact: false,
                fragment: GpuFragmentClass::Mapped,
                source_staging_bytes: 32,
                host_staging_bytes: 64,
                pinned_host_staging_bytes: 64,
            }),
            resolve_gpu_route(GpuRouteResolutionInput {
                source_device: Some(1),
                destination_device: Some(0),
                source_range: ColumnRange { start: 2, end: 4 },
                destination_range: ColumnRange { start: 2, end: 4 },
                source_is_resident: false,
                peer_available: false,
                source_compact: false,
                destination_compact: false,
                fragment: GpuFragmentClass::Mapped,
                source_staging_bytes: 48,
                host_staging_bytes: 96,
                pinned_host_staging_bytes: 96,
            }),
        ];
        let resolved = authoritative_production_route(requested, &observed)
            .expect("production response must yield a physical route");
        assert_eq!(resolved.route, GpuTransferRoute::HostStaging);
        assert_eq!(resolved.source_device, Some(1));
        assert_eq!(resolved.destination_device, Some(0));
        assert_eq!(resolved.source_range, ColumnRange { start: 0, end: 4 });
        assert_eq!(resolved.host_staging_bytes, 160);
        assert_eq!(resolved.pinned_host_staging_bytes, 160);
        assert_ne!(resolved, requested);
        assert!(resolved.validate());
    }

    /// Exercise every host/control profile through the real setup provider.
    /// The table deliberately carries the concrete node variant and wire
    /// contract for each case; a profile-only fake would not execute these
    /// production host dispatches and would therefore miss invalid
    /// representatives.
    #[cfg(feature = "gpu")]
    #[test]
    #[serial_test::serial(gpu_context)]
    fn real_provider_dispatches_all_host_control_inventory_variants() {
        use mxx_ir_core::{
            expr::RealExpr,
            node::{
                IntBinaryOp, IntCompareOp, LoopInputMode, NodeKind, ParallelLoop, RealBinaryOp,
                SequentialLoop, SubgraphCall,
            },
            types::ConcreteWireType,
        };
        use mxx_primitives::poly::dcrt::gpu::{
            GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync,
        };
        use mxx_runtime::{
            backend::{GpuWarmupProfileProvider, GpuWarmupRoute, poly_gpu::gpu_backend_on},
            gpu_column_policy::GpuFragmentClass,
        };
        use num_bigint::BigInt;
        use std::time::Duration;

        let device = detected_gpu_device_ids()
            .into_iter()
            .next()
            .expect("GPU feature tests require one detected device");
        let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
        let harness = MeasurementHarnessConfig {
            warm_up_iterations: 0,
            measured_iterations: 1,
            memory_poll_interval: Duration::ZERO,
        };
        let measurement_backend = gpu_backend_on([parameters], [device]);
        let mut provider =
            GpuNodeMeasurementBackend::new(vec![(measurement_backend, device)], harness);
        let int = ConcreteWireType::Int;
        let real = ConcreteWireType::Real;
        let bool_ = ConcreteWireType::Bool;
        let cases = vec![
            (
                CanonicalWarmupProfileDomain::Input,
                NodeKind::Input {
                    name: "host-input".into(),
                    wire_type: mxx_ir_core::types::WireType::Int,
                    artifact: None,
                },
                vec![],
                vec![int.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantInt,
                NodeKind::ConstantInt(BigInt::from(7)),
                vec![],
                vec![int.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::EvaluateInt,
                NodeKind::EvaluateInt(mxx_ir_core::IntExpr::constant(7)),
                vec![],
                vec![int.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantReal,
                NodeKind::ConstantReal(RealExpr::from_integer(1)),
                vec![],
                vec![real.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantBool,
                NodeKind::ConstantBool(true),
                vec![],
                vec![bool_.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::IntBinary,
                NodeKind::IntBinary(IntBinaryOp::Add),
                vec![int.clone(), int.clone()],
                vec![int.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::IntCompare,
                NodeKind::IntCompare(IntCompareOp::Equal),
                vec![int.clone(), int.clone()],
                vec![bool_.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::BitExtract,
                NodeKind::BitExtract { bit: mxx_ir_core::IntExpr::constant(0) },
                vec![int.clone()],
                vec![bool_.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::IntToReal,
                NodeKind::IntToReal,
                vec![int.clone()],
                vec![real.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::BoolToInt,
                NodeKind::BoolToInt,
                vec![bool_.clone()],
                vec![int.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::RealBinary,
                NodeKind::RealBinary(RealBinaryOp::Add),
                vec![real.clone(), real.clone()],
                vec![real.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::RealSqrt,
                NodeKind::RealSqrt,
                vec![real.clone()],
                vec![real.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::FamilyPack,
                NodeKind::FamilyPack { count: mxx_ir_core::IntExpr::constant(2) },
                vec![],
                vec![ConcreteWireType::IndexedFamily { element: Box::new(int.clone()), count: 2 }],
            ),
            (
                CanonicalWarmupProfileDomain::FamilyGetStatic,
                NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(0) },
                vec![],
                vec![int.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::FamilyGetDynamic,
                NodeKind::FamilyGetDynamic,
                vec![],
                vec![int.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::Select,
                NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
                vec![],
                vec![int.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::SubgraphCall,
                NodeKind::SubgraphCall(SubgraphCall {
                    definition: "host".into(),
                    bindings: vec![],
                    canonical_input_exclusive_uppers: vec![],
                }),
                vec![],
                vec![],
            ),
            (
                CanonicalWarmupProfileDomain::ParallelLoop,
                NodeKind::ParallelLoop(ParallelLoop {
                    count: mxx_ir_core::IntExpr::constant(1),
                    minimum_count: 1,
                    index_slot: 0,
                    bindings: vec![],
                    input_modes: vec![LoopInputMode::Broadcast],
                }),
                vec![],
                vec![],
            ),
            (
                CanonicalWarmupProfileDomain::SequentialLoop,
                NodeKind::SequentialLoop(SequentialLoop {
                    count: mxx_ir_core::IntExpr::constant(1),
                    index_slot: 0,
                    bindings: vec![],
                    carried_count: 0,
                }),
                vec![],
                vec![],
            ),
        ];
        for (index, (domain, kind, arguments, outputs)) in cases.into_iter().enumerate() {
            assert_eq!(domain.measurement_kind(), WarmupMeasurementKind::HostMeasured);
            let signature = GpuWarmupOperationSignature {
                operation: [index as u8 + 1; 32],
                shape_class: index as u64 + 1,
                instance_class: 0,
            };
            provider
                .register_operation(GpuWarmupOperationDescriptor {
                    inputs: Default::default(),
                    signature,
                    scope: mxx_ir_core::FrozenGraphScopeId::Root,
                    node: mxx_ir_core::types::NodeId(index as u64),
                    kind,
                    concrete_argument_types: arguments,
                    concrete_output_types: outputs,
                    bindings: ParamEnv::default(),
                    effective_operation: domain.identity().into(),
                    profile_domain: domain,
                    fused_operation: None,
                    implementation_variant: domain.identity().into(),
                    source_layouts: Vec::new(),
                    output_layout: None,
                    route_resolver: None,
                    host_control: None,
                })
                .expect("host inventory descriptor must register");
            let request = GpuWarmupProfileRequest {
                signature,
                device: 0,
                device_identity: GpuWarmupDeviceIdentity::new(0, "test", "test"),
                tile_width: 1,
                range: RuntimeIndexRange { start: 0, end: 1 },
                executed_range_start: 0,
                executed_range_class: GpuWarmupFragmentClass::Whole,
                route: GpuWarmupRoute::HostOnly,
                route_descriptor: GpuExecutionRouteDescriptor::device_local(
                    0,
                    ColumnRange { start: 0, end: 1 },
                    GpuFragmentClass::Full,
                ),
                route_resolver: None,
                binding_port: None,
                fragment: GpuWarmupFragmentClass::Whole,
                retry_cap: None,
                cache_identity: None,
                cache_state: GpuWarmupCacheState::Warm,
                timing_scope: GpuWarmupTimingScope::LocalJob,
            };
            let profile =
                provider.measure(&request).expect("host production dispatch must measure");
            assert_eq!(profile.measurement, WarmupMeasurementKind::HostMeasured);
            assert!(profile.time_seconds.is_finite() && profile.time_seconds > 0.0);
            assert_eq!(profile.workspace_bytes, 0);
            assert_eq!(profile.provenance, GpuWarmupProvenance::ProductionEquivalent);
            assert_eq!(profile.timing_scope, GpuWarmupTimingScope::LocalJob);
            assert_eq!(profile.memory.evidence, MemoryEvidenceKind::ExactQuery);
            assert!(provider.warmup_dispatch_records().iter().any(|record| {
                record.profile_domain == domain && record.fused_operation.is_none()
            }));
        }
        gpu_device_sync();
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[serial_test::serial(gpu_context)]
    fn host_family_representatives_use_valid_indices_and_bounded_payloads() {
        use mxx_primitives::poly::dcrt::gpu::{
            GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync,
        };
        use mxx_runtime::backend::{RuntimeValue, poly_gpu::gpu_backend_on};
        use num_bigint::Sign;

        let device = detected_gpu_device_ids()
            .into_iter()
            .next()
            .expect("GPU feature tests require one detected device");
        let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let matrix = |columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: modulus.clone(),
                ring_dimension: parameters.ring_dimension() as usize,
                rows: 1,
                columns,
            })
        };
        let scope = FrozenGraphScopeId::Root;
        let mut backend = gpu_backend_on([parameters.clone()], [device]);
        backend.select_operation([0xD1; 32]).expect("select setup operation");
        for (id, count) in [1usize, 2, 7, 8].into_iter().enumerate() {
            let family_type =
                ConcreteWireType::IndexedFamily { element: Box::new(matrix(1)), count };
            let arguments = [family_type, ConcreteWireType::Int];
            let outputs = [matrix(1)];
            let kind = NodeKind::FamilyGetDynamic;
            let node = MeasurementNode {
                scope: &scope,
                id: NodeId(10 + id as u64),
                kind: &kind,
                arguments: &[],
                argument_kinds: &[],
                argument_types: &[],
                output_types: &[],
                concrete_argument_types: arguments.to_vec(),
                concrete_output_types: outputs.to_vec(),
            };
            let prepared = GpuNodeMeasurementBackend::prepare_host_primitive(
                &mut backend,
                &node,
                &ParamEnv::default(),
            )
            .expect("nonempty dynamic family must prepare");
            assert!(matches!(
                prepared.dynamic_index,
                Some(RuntimeValue::Int(ref value)) if value == &BigInt::from(0u8)
            ));
            let Some(RuntimeValue::IndexedFamily(values)) = prepared.family_inputs.first() else {
                panic!("dynamic family representative must remain indexed");
            };
            assert_eq!(values.len(), count);
        }

        let family_type =
            ConcreteWireType::IndexedFamily { element: Box::new(matrix(64)), count: 64 };
        let dynamic_arguments = [family_type, ConcreteWireType::Int];
        let dynamic_outputs = [matrix(64)];
        let dynamic_kind = NodeKind::FamilyGetDynamic;
        let dynamic_node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &dynamic_kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: dynamic_arguments.to_vec(),
            concrete_output_types: dynamic_outputs.to_vec(),
        };
        let prepared = GpuNodeMeasurementBackend::prepare_host_primitive(
            &mut backend,
            &dynamic_node,
            &ParamEnv::default(),
        )
        .expect("nonempty dynamic family must prepare");
        assert!(matches!(
            prepared.dynamic_index,
            Some(RuntimeValue::Int(ref value)) if value == &BigInt::from(0u8)
        ));
        let Some(RuntimeValue::IndexedFamily(values)) = prepared.family_inputs.first() else {
            panic!("dynamic family representative must remain indexed");
        };
        assert_eq!(values.len(), 64, "family access must preserve its container cardinality");
        assert!(matches!(values.first(), Some(RuntimeValue::Matrix(_))));

        // Nested family representatives must retain every container layer's
        // cardinality.  The leaf matrix owner is shared, so this exercises
        // real O(count) vector construction/cloning without count×GPU payload.
        for nested_count in [1usize, 2, 1024] {
            let nested = ConcreteWireType::IndexedFamily {
                element: Box::new(matrix(1)),
                count: nested_count,
            };
            let outer =
                ConcreteWireType::IndexedFamily { element: Box::new(nested), count: nested_count };
            let nested_arguments = [outer, ConcreteWireType::Int];
            let nested_outputs = [ConcreteWireType::IndexedFamily {
                element: Box::new(matrix(1)),
                count: nested_count,
            }];
            let nested_kind = NodeKind::FamilyGetDynamic;
            let nested_node = MeasurementNode {
                scope: &scope,
                id: NodeId(100 + nested_count as u64),
                kind: &nested_kind,
                arguments: &[],
                argument_kinds: &[],
                argument_types: &[],
                output_types: &[],
                concrete_argument_types: nested_arguments.to_vec(),
                concrete_output_types: nested_outputs.to_vec(),
            };
            let prepared = GpuNodeMeasurementBackend::prepare_host_primitive(
                &mut backend,
                &nested_node,
                &ParamEnv::default(),
            )
            .expect("nested family representative must prepare");
            let Some(RuntimeValue::IndexedFamily(outer_values)) = prepared.family_inputs.first()
            else {
                panic!("nested family representative must remain indexed");
            };
            assert_eq!(outer_values.len(), nested_count);
            let Some(RuntimeValue::IndexedFamily(first_inner)) = outer_values.first() else {
                panic!("nested family member must remain indexed");
            };
            assert_eq!(first_inner.len(), nested_count);
            let Some(RuntimeValue::Matrix(first_leaf)) = first_inner.first() else {
                panic!("nested family leaf must remain a matrix");
            };
            for outer_value in outer_values {
                let RuntimeValue::IndexedFamily(inner_values) = outer_value else {
                    panic!("every nested family member must retain its shape");
                };
                assert_eq!(inner_values.len(), nested_count);
                for inner_value in inner_values {
                    let RuntimeValue::Matrix(leaf) = inner_value else {
                        panic!("every nested family leaf must remain a matrix");
                    };
                    assert!(std::sync::Arc::ptr_eq(first_leaf, &leaf));
                }
            }
        }

        let select_kind = NodeKind::Select { count: IntExpr::constant(2) };
        let select_outputs = [matrix(64)];
        let select_node = MeasurementNode {
            scope: &scope,
            id: NodeId(2),
            kind: &select_kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: Vec::new(),
            concrete_output_types: select_outputs.to_vec(),
        };
        let prepared = GpuNodeMeasurementBackend::prepare_host_primitive(
            &mut backend,
            &select_node,
            &ParamEnv::default(),
        )
        .expect("nonempty select must prepare");
        assert!(matches!(
            prepared.dynamic_index,
            Some(RuntimeValue::Int(ref value)) if value == &BigInt::from(0u8)
        ));
        assert_eq!(prepared.choices.len(), 2);
        let (Some(RuntimeValue::Matrix(first)), Some(RuntimeValue::Matrix(second))) =
            (prepared.choices.first(), prepared.choices.get(1))
        else {
            panic!("select choices must contain matrix exemplars");
        };
        assert!(std::sync::Arc::ptr_eq(first, second));

        let pack_kind = NodeKind::FamilyPack { count: IntExpr::constant(64) };
        let pack_outputs =
            [ConcreteWireType::IndexedFamily { element: Box::new(matrix(64)), count: 64 }];
        let pack_node = MeasurementNode {
            scope: &scope,
            id: NodeId(3),
            kind: &pack_kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: Vec::new(),
            concrete_output_types: pack_outputs.to_vec(),
        };
        let prepared = GpuNodeMeasurementBackend::prepare_host_primitive(
            &mut backend,
            &pack_node,
            &ParamEnv::default(),
        )
        .expect("family pack must prepare");
        assert_eq!(prepared.family_inputs.len(), 64);
        let matrices = prepared
            .family_inputs
            .iter()
            .map(|value| match value {
                RuntimeValue::Matrix(matrix) => matrix,
                _ => panic!("family pack representative must preserve payload kind"),
            })
            .collect::<Vec<_>>();
        assert!(matrices.windows(2).all(|pair| std::sync::Arc::ptr_eq(pair[0], pair[1])));

        let empty_kind = NodeKind::Select { count: IntExpr::constant(0) };
        let empty_outputs = [ConcreteWireType::Int];
        let empty_node = MeasurementNode {
            scope: &scope,
            id: NodeId(4),
            kind: &empty_kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: Vec::new(),
            concrete_output_types: empty_outputs.to_vec(),
        };
        assert!(
            GpuNodeMeasurementBackend::prepare_host_primitive(
                &mut backend,
                &empty_node,
                &ParamEnv::default(),
            )
            .is_err()
        );
        gpu_device_sync();
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[serial_test::serial(gpu_context)]
    fn real_provider_dispatches_gpu_inventory_cases_with_ranges() {
        use mxx_ir_core::{
            expr::RealExpr,
            node::{
                ConcatAxis, ConstantMatrix, HashVariant, MatrixBinaryOp, NodeKind, SampleRange,
            },
            types::{ConcreteMatrixType, ConcreteWireType, MatrixType},
        };
        use mxx_primitives::poly::dcrt::gpu::{
            GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync,
        };
        use mxx_runtime::{
            backend::{
                GpuWarmupProfileProvider, GpuWarmupRoute, GpuWarmupRouteResolverData,
                poly_gpu::gpu_backend_on,
            },
            gpu_column_policy::GpuFragmentClass,
        };
        use num_bigint::{BigInt, Sign};
        use std::time::Duration;

        let device = detected_gpu_device_ids()
            .into_iter()
            .next()
            .expect("GPU feature tests require one detected device");
        let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let concrete =
            ConcreteMatrixType { modulus: modulus.clone(), ring_dimension: 8, rows: 1, columns: 1 };
        let symbolic = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(modulus.clone()),
            ring_dimension: mxx_ir_core::IntExpr::constant(8),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let gadget_concrete =
            ConcreteWireType::Matrix(ConcreteMatrixType { columns: 6, ..concrete.clone() });
        let gadget_symbolic =
            MatrixType { columns: mxx_ir_core::IntExpr::constant(6), ..symbolic.clone() };
        let matrix = ConcreteWireType::Matrix(concrete.clone());
        let row_concat =
            ConcreteWireType::Matrix(ConcreteMatrixType { rows: 2, ..concrete.clone() });
        let column_concat =
            ConcreteWireType::Matrix(ConcreteMatrixType { columns: 2, ..concrete.clone() });
        let diagonal_concat = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 2,
            columns: 2,
            ..concrete.clone()
        });
        let small = ConcreteWireType::SmallMatrix {
            matrix: concrete.clone(),
            max_coefficient_bound: BigInt::from(3),
        };
        let decompose_output = ConcreteWireType::SmallMatrix {
            matrix: ConcreteMatrixType { rows: 3, ..concrete.clone() },
            max_coefficient_bound: BigInt::from(255),
        };
        let harness = MeasurementHarnessConfig {
            warm_up_iterations: 0,
            measured_iterations: 1,
            memory_poll_interval: Duration::ZERO,
        };
        let measurement_backend = gpu_backend_on([parameters], [device]);
        let mut provider =
            GpuNodeMeasurementBackend::new(vec![(measurement_backend, device)], harness);
        let resolver = GpuWarmupRouteResolverData {
            source_layouts: Vec::new(),
            output_layout: None,
            source_owners: vec![0],
            destination_owner: 0,
            source_range: ColumnRange { start: 0, end: 1 },
            destination_range: ColumnRange { start: 0, end: 1 },
            source_compact: false,
            destination_compact: false,
            peer_available: false,
            source_staging_bytes: 0,
            host_staging_bytes: 0,
            pinned_host_staging_bytes: 0,
        };
        let cases: Vec<(
            CanonicalWarmupProfileDomain,
            NodeKind,
            Vec<ConcreteWireType>,
            Vec<ConcreteWireType>,
        )> = vec![
            (
                CanonicalWarmupProfileDomain::ConstantMatrixZero,
                NodeKind::ConstantMatrix {
                    matrix_type: symbolic.clone(),
                    value: ConstantMatrix::Zero,
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantMatrixIdentity,
                NodeKind::ConstantMatrix {
                    matrix_type: symbolic.clone(),
                    value: ConstantMatrix::Identity,
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantMatrixUnitRow,
                NodeKind::ConstantMatrix {
                    matrix_type: symbolic.clone(),
                    value: ConstantMatrix::UnitRow { index: mxx_ir_core::IntExpr::constant(0) },
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantMatrixUnitColumn,
                NodeKind::ConstantMatrix {
                    matrix_type: symbolic.clone(),
                    value: ConstantMatrix::UnitColumn { index: mxx_ir_core::IntExpr::constant(0) },
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantMatrixGadget,
                NodeKind::ConstantMatrix {
                    matrix_type: gadget_symbolic,
                    value: ConstantMatrix::Gadget {
                        base: mxx_ir_core::IntExpr::constant(2),
                        small: false,
                    },
                },
                vec![],
                vec![gadget_concrete],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantMatrixPowerOfBase,
                NodeKind::ConstantMatrix {
                    matrix_type: symbolic.clone(),
                    value: ConstantMatrix::PowerOfBase {
                        base: mxx_ir_core::IntExpr::constant(2),
                        exponent: mxx_ir_core::IntExpr::constant(1),
                    },
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantMatrixRotation,
                NodeKind::ConstantMatrix {
                    matrix_type: symbolic.clone(),
                    value: ConstantMatrix::Rotation { exponent: mxx_ir_core::IntExpr::constant(1) },
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConstantMatrixPolynomial,
                NodeKind::ConstantMatrix {
                    matrix_type: symbolic.clone(),
                    value: ConstantMatrix::Polynomial {
                        coefficients: vec![mxx_ir_core::IntExpr::constant(1)],
                    },
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::UniformResidueSample,
                NodeKind::UniformResidueSample { matrix_type: symbolic.clone() },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::UniformIntervalSample,
                NodeKind::UniformIntervalSample {
                    matrix_type: symbolic.clone(),
                    range: SampleRange {
                        minimum: mxx_ir_core::IntExpr::constant(0),
                        maximum: mxx_ir_core::IntExpr::constant(1),
                    },
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::GaussianSample,
                NodeKind::GaussianSample {
                    matrix_type: symbolic.clone(),
                    sigma: RealExpr::from_integer(1),
                    max_coefficient_bound: mxx_ir_core::IntExpr::constant(2),
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::HashSample,
                NodeKind::HashSample {
                    matrix_type: symbolic.clone(),
                    variant: HashVariant::Plain,
                    tag_prefix: vec![7],
                    tag_components: vec![],
                    base: None,
                    digit_count: None,
                },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::MatrixAdd,
                NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                vec![matrix.clone(), matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::MatrixSubtract,
                NodeKind::MatrixBinary(MatrixBinaryOp::Subtract),
                vec![matrix.clone(), matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::MatrixMultiply,
                NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                vec![matrix.clone(), matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::MatrixMulSmallRhs,
                NodeKind::MatrixMulSmallRhs,
                vec![matrix.clone(), small.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::MatrixMulAccumulate,
                NodeKind::MatrixMulAccumulate {
                    coefficients: vec![mxx_ir_core::IntExpr::constant(1)],
                    has_bias: true,
                },
                vec![matrix.clone(), matrix.clone(), matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::MatrixNegate,
                NodeKind::MatrixNegate,
                vec![matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::MatrixScale,
                NodeKind::MatrixScale { scalar: mxx_ir_core::IntExpr::constant(2) },
                vec![matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::RingAutomorphism,
                NodeKind::RingAutomorphism { index: mxx_ir_core::IntExpr::constant(1) },
                vec![matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::Transpose,
                NodeKind::Transpose,
                vec![matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::Slice,
                NodeKind::Slice { rows: None, columns: None },
                vec![matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::Tensor,
                NodeKind::Tensor,
                vec![matrix.clone(), matrix.clone()],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ConcatRows,
                NodeKind::Concat { axis: ConcatAxis::Rows },
                vec![matrix.clone(), matrix.clone()],
                vec![row_concat],
            ),
            (
                CanonicalWarmupProfileDomain::ConcatColumns,
                NodeKind::Concat { axis: ConcatAxis::Columns },
                vec![matrix.clone(), matrix.clone()],
                vec![column_concat],
            ),
            (
                CanonicalWarmupProfileDomain::ConcatDiagonal,
                NodeKind::Concat { axis: ConcatAxis::Diagonal },
                vec![matrix.clone(), matrix.clone()],
                vec![diagonal_concat],
            ),
            (
                CanonicalWarmupProfileDomain::GadgetDecompose,
                NodeKind::GadgetDecompose {
                    base: mxx_ir_core::IntExpr::constant(2),
                    small: true,
                    digit_count: mxx_ir_core::IntExpr::constant(3),
                },
                vec![matrix.clone()],
                vec![decompose_output],
            ),
            (
                CanonicalWarmupProfileDomain::LiftIntegerToConstantPolynomial,
                NodeKind::LiftIntegerToConstantPolynomial { matrix_type: symbolic.clone() },
                vec![],
                vec![matrix.clone()],
            ),
            (
                CanonicalWarmupProfileDomain::ModulusReduce,
                NodeKind::ModulusReduce { modulus: mxx_ir_core::IntExpr::constant(131_009) },
                vec![matrix.clone()],
                vec![matrix.clone()],
            ),
        ];
        for (index, (domain, kind, arguments, outputs)) in cases.into_iter().enumerate() {
            let width = outputs
                .iter()
                .find_map(|wire_type| wire_type.matrix_type().map(|matrix| matrix.columns))
                .unwrap_or(1);
            let case_resolver = GpuWarmupRouteResolverData {
                source_range: ColumnRange { start: 0, end: width },
                destination_range: ColumnRange { start: 0, end: width },
                ..resolver.clone()
            };
            let signature = GpuWarmupOperationSignature {
                operation: [0x80 + index as u8; 32],
                shape_class: index as u64 + 1,
                instance_class: 0,
            };
            provider
                .register_operation(GpuWarmupOperationDescriptor {
                    inputs: Default::default(),
                    signature,
                    scope: mxx_ir_core::FrozenGraphScopeId::Root,
                    node: mxx_ir_core::types::NodeId(10_000 + index as u64),
                    kind,
                    concrete_argument_types: arguments,
                    concrete_output_types: outputs,
                    bindings: ParamEnv::default(),
                    effective_operation: domain.identity().into(),
                    profile_domain: domain,
                    fused_operation: None,
                    implementation_variant: domain.identity().into(),
                    source_layouts: Vec::new(),
                    output_layout: None,
                    route_resolver: Some(case_resolver.clone()),
                    host_control: None,
                })
                .expect("GPU inventory descriptor must register");
            let request = GpuWarmupProfileRequest {
                signature,
                device: 0,
                device_identity: GpuWarmupDeviceIdentity::new(0, "test", "test"),
                tile_width: width,
                range: RuntimeIndexRange { start: 0, end: width },
                executed_range_start: 0,
                executed_range_class: GpuWarmupFragmentClass::Whole,
                route: GpuWarmupRoute::DeviceLocal,
                route_descriptor: case_resolver.resolve(GpuFragmentClass::Full),
                route_resolver: Some(case_resolver),
                binding_port: None,
                fragment: GpuWarmupFragmentClass::Whole,
                retry_cap: None,
                cache_identity: None,
                cache_state: GpuWarmupCacheState::Warm,
                timing_scope: GpuWarmupTimingScope::LocalJob,
            };
            let profile = provider.measure(&request).unwrap_or_else(|error| {
                panic!("GPU production case {domain:?} must measure: {error}")
            });
            assert_eq!(profile.measurement, WarmupMeasurementKind::GpuMeasured);
            assert!(profile.time_seconds.is_finite() && profile.time_seconds > 0.0);
            assert_eq!(profile.provenance, GpuWarmupProvenance::ProductionEquivalent);
            assert_eq!(profile.timing_scope, GpuWarmupTimingScope::LocalJob);
            assert!(!profile.memory.affected_devices.is_empty());
            assert!(provider.warmup_dispatch_records().iter().any(|record| {
                record.profile_domain == domain &&
                    record.fused_operation.is_none() &&
                    record.range.start < record.range.end
            }));
        }
        // Fused variants are covered by gpu_lifecycle_tests through validated
        // DSL lowering; hand-built descriptors cannot represent that contract.
        gpu_device_sync();
    }

    #[test]
    fn host_control_repetition_policy_accepts_one_and_rejects_zero() {
        let kind = NodeKind::SubgraphCall(mxx_ir_core::node::SubgraphCall {
            definition: "empty".into(),
            bindings: Vec::new(),
            canonical_input_exclusive_uppers: Vec::new(),
        });
        let scope = FrozenGraphScopeId::Root;
        let bindings = ParamEnv::default();
        let one = GpuNodeMeasurementBackend::measure_host_control_repeated(
            &MeasurementHarnessConfig {
                warm_up_iterations: 0,
                measured_iterations: 1,
                memory_poll_interval: std::time::Duration::ZERO,
            },
            &scope,
            NodeId(0),
            &kind,
            &bindings,
            None,
        )
        .expect("one host repetition is valid");
        assert_eq!(one.2, 1);
        assert!(one.0.is_finite() && one.0 > 0.0);
        assert_eq!(one.1, 0.0);
        let zero = GpuNodeMeasurementBackend::measure_host_control_repeated(
            &MeasurementHarnessConfig {
                warm_up_iterations: 0,
                measured_iterations: 0,
                memory_poll_interval: std::time::Duration::ZERO,
            },
            &scope,
            NodeId(0),
            &kind,
            &bindings,
            None,
        );
        assert!(matches!(zero, Err(GpuMeasurementError(_))));
    }

    #[test]
    #[serial_test::serial]
    fn gpu_backend_keeps_the_setup_time_vram_percentage() {
        let name = "MXX_GPU_VRAM_PERCENT";
        let original = std::env::var_os(name);
        unsafe { std::env::set_var(name, "37") };
        let backend = GpuNodeMeasurementBackend::from_workers(
            Vec::new(),
            crate::harness::MeasurementHarnessConfig::default(),
            37,
        );

        unsafe { std::env::set_var(name, "91") };
        let captured = backend.vram_percent;
        match original {
            Some(value) => unsafe { std::env::set_var(name, value) },
            None => unsafe { std::env::remove_var(name) },
        }

        assert_eq!(captured, 37);
    }

    #[test]
    fn fixed_plan_widths_are_consumed_without_recalibration() {
        let site = GpuExecutionSiteKey { site: 8, shape_class: 0, instance_class: 0 };
        let plan = FrozenGpuPlan::new(
            GpuPlanContract {
                graph_specification_hash: [1; 32],
                backend_identity: "bench".into(),
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
            },
            vec![GpuLayout {
                id: 1,
                columns: 2,
                rows: 1,
                ring_dimension: 1,
                representation: "matrix".into(),
                instance_device_stride: 0,
                owner_intervals: vec![
                    GpuColumnInterval { device: 0, start: 0, end: 1 },
                    GpuColumnInterval { device: 1, start: 1, end: 2 },
                ],
            }],
            vec![],
            vec![GpuNodeChoice {
                key: site,
                loop_site: None,
                operation_identity: [3; 32],
                effective_operation:
                    mxx_runtime::gpu_column_policy::EffectiveGpuOperation::MatrixAdd,
                column_capability: mxx_runtime::gpu_column_policy::ColumnCapability::SameColumns,
                output_layouts: vec![1],
                columns_per_job: vec![2, 4],
                implementation_variant: "test".into(),
                preimage_max_attempts: None,
            }],
        )
        .unwrap();
        assert_eq!(
            GpuNodeMeasurementBackend::fixed_plan_widths(&plan, site).unwrap(),
            GpuColumnWidths { gpu0: 2, nonzero: Some(4) }
        );
        assert_eq!(
            GpuNodeMeasurementBackend::fixed_plan_wave_seconds(
                &plan,
                site,
                &vec![mxx_runtime::gpu_warmup::GpuTimeModel::default(); 2],
            )
            .unwrap(),
            0.0
        );
    }

    #[test]
    fn fixed_plan_multi_output_ports_are_one_primitive_with_sibling_batch() {
        let site = GpuExecutionSiteKey { site: 9, shape_class: 0, instance_class: 0 };
        let contract = GpuPlanContract {
            graph_specification_hash: [1; 32],
            backend_identity: "bench".into(),
            logical_to_physical_devices: vec![0, 1],
            device_budgets: vec![
                GpuDeviceBudget {
                    device: 0,
                    device_bytes: 1000,
                    pinned_host_bytes: 1000,
                    host_bytes: 1000,
                },
                GpuDeviceBudget {
                    device: 1,
                    device_bytes: 1000,
                    pinned_host_bytes: 1000,
                    host_bytes: 1000,
                },
            ],
            shape_contract_hash: [2; 32],
            backend_revision: "test".into(),
        };
        let interval = |device, start, end| GpuColumnInterval { device, start, end };
        let plan = FrozenGpuPlan::new(
            contract,
            vec![
                GpuLayout {
                    id: 1,
                    columns: 4,
                    rows: 1,
                    ring_dimension: 1,
                    representation: "left".into(),
                    instance_device_stride: 0,
                    owner_intervals: vec![interval(0, 0, 2), interval(1, 2, 4)],
                },
                GpuLayout {
                    id: 2,
                    columns: 3,
                    rows: 1,
                    ring_dimension: 1,
                    representation: "right".into(),
                    instance_device_stride: 0,
                    owner_intervals: vec![interval(0, 0, 1), interval(1, 1, 3)],
                },
            ],
            vec![GpuLoopChoice {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 2,
                wave_instances: 2,
                tail_instances: 0,
            }],
            vec![GpuNodeChoice {
                key: site,
                loop_site: Some(GpuLoopSiteKey { site: 7, shape_class: 0 }),
                operation_identity: [3; 32],
                effective_operation:
                    mxx_runtime::gpu_column_policy::EffectiveGpuOperation::MatrixAdd,
                column_capability: mxx_runtime::gpu_column_policy::ColumnCapability::SameColumns,
                output_layouts: vec![1, 2],
                columns_per_job: vec![2, 2],
                implementation_variant: "test".into(),
                preimage_max_attempts: None,
            }],
        )
        .unwrap();
        let model =
            vec![
                GpuTimeModel { fixed_seconds: 1.0, per_column_seconds: 1.0, ..Default::default() };
                2
            ];
        // The shared union preserves the source logical-wave boundaries of
        // both ports.  The differing owner partitions therefore produce two
        // four-second waves for the sibling batch, rather than independently
        // summing one schedule per output port.
        assert_eq!(
            GpuNodeMeasurementBackend::fixed_plan_wave_seconds(&plan, site, &model).unwrap(),
            8.0
        );
    }

    #[test]
    fn shared_cuda_pool_is_an_explicit_measurement_error() {
        assert!(require_exclusive_measurement_context(0, 1, 1).is_ok());
        assert!(require_exclusive_measurement_context(0, 2, 2).is_ok());
        let error = require_exclusive_measurement_context(3, 2, 1).unwrap_err();
        assert_eq!(
            error.to_string(),
            "GPU 3 has 2 live mxx contexts but the measurement backend owns 1; exclusive CUDA mempool measurement is required"
        );
    }

    #[test]
    fn fleet_wave_extrapolation_uses_conservative_full_waves() {
        let full_wave = NodeMeasurement {
            work_seconds: 38.0,
            latency_seconds: 39.0,
            cumulative_wave_seconds: 39.0,
            independent_wave_count: 1,
            measured_wave_workspace_bytes: 56,
            workspace_bytes: 56,
        };
        let measurement = extrapolate_fleet_waves(&full_wave, 7);

        assert_eq!(measurement.work_seconds, 266.0);
        assert_eq!(measurement.latency_seconds, 39.0);
        assert_eq!(measurement.cumulative_wave_seconds, 273.0);
        assert_eq!(measurement.independent_wave_count, 7);
        assert_eq!(measurement.workspace_bytes, 56);
        assert_eq!(measurement.measured_wave_workspace_bytes, 56);

        // A synthetic large wave count must not scale one-wave latency or scratch.
        let packed = extrapolate_fleet_waves(&full_wave, 1usize << 40);
        assert_eq!(packed.latency_seconds, 39.0);
        assert_eq!(packed.measured_wave_workspace_bytes, 56);
        assert_eq!(packed.cumulative_wave_seconds, 39.0 * ((1u64 << 40) as f64));
        assert_eq!(packed.work_seconds, 38.0 * ((1u64 << 40) as f64));
    }

    #[test]
    fn fleet_wave_sums_work_and_workspace_and_uses_measured_wall_latency() {
        let measurement = aggregate_fleet_wave(
            [
                NodeMeasurement {
                    work_seconds: 2.0,
                    latency_seconds: 2.0,
                    cumulative_wave_seconds: 2.0,
                    independent_wave_count: 1,
                    measured_wave_workspace_bytes: 30,
                    workspace_bytes: 30,
                },
                NodeMeasurement {
                    work_seconds: 3.0,
                    latency_seconds: 3.0,
                    cumulative_wave_seconds: 3.0,
                    independent_wave_count: 1,
                    measured_wave_workspace_bytes: 20,
                    workspace_bytes: 20,
                },
            ],
            3.25,
        );

        assert_eq!(measurement.work_seconds, 5.0);
        assert_eq!(measurement.latency_seconds, 3.25);
        assert_eq!(measurement.cumulative_wave_seconds, 3.25);
        assert_eq!(measurement.workspace_bytes, 50);
    }

    #[test]
    fn only_production_range_operations_are_fleet_separable() {
        let matrix_type = MatrixType {
            rows: IntExpr::constant(8),
            columns: IntExpr::constant(8),
            ring_dimension: IntExpr::constant(32),
            modulus: IntExpr::constant(257),
        };

        assert!(GpuNodeMeasurementBackend::column_separable(&NodeKind::ConstantMatrix {
            matrix_type: matrix_type.clone(),
            value: ConstantMatrix::Identity
        }));
        assert!(GpuNodeMeasurementBackend::column_separable(&NodeKind::Transpose));
        assert!(GpuNodeMeasurementBackend::column_separable(&NodeKind::Concat {
            axis: mxx_ir_core::node::ConcatAxis::Columns,
        }));
        assert!(GpuNodeMeasurementBackend::column_separable(&NodeKind::Tensor));
        assert!(GpuNodeMeasurementBackend::column_separable(&NodeKind::Concat {
            axis: mxx_ir_core::node::ConcatAxis::Diagonal,
        }));
        assert!(GpuNodeMeasurementBackend::column_separable(&NodeKind::ConstantMatrix {
            matrix_type: MatrixType { rows: 1.into(), columns: 1.into(), ..matrix_type },
            value: ConstantMatrix::Rotation { exponent: IntExpr::constant(1) },
        }));
    }

    #[test]
    fn indexed_constant_preserves_production_variant() {
        let matrix_type = MatrixType {
            rows: IntExpr::constant(8),
            columns: IntExpr::constant(8),
            ring_dimension: IntExpr::constant(32),
            modulus: IntExpr::constant(257),
        };
        let kind = NodeKind::ConstantMatrix { matrix_type, value: ConstantMatrix::Identity };
        let concrete = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 8,
            columns: 8,
            ring_dimension: 32,
            modulus: BigInt::from(257u16),
        });
        let scope = FrozenGraphScopeId::Root;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![],
            concrete_output_types: vec![concrete],
        };

        let (representative, _, outputs, _, _) =
            GpuNodeMeasurementBackend::representative_node(&node, 3);
        assert!(matches!(
            representative,
            NodeKind::ConstantMatrix { value: ConstantMatrix::Identity, .. }
        ));
        assert_eq!(outputs[0].matrix_type().expect("matrix output").columns, 3);

        let request = PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(1),
            kind: kind.clone(),
            concrete_argument_types: vec![],
            concrete_output_types: vec![ConcreteWireType::Matrix(ConcreteMatrixType {
                rows: 8,
                columns: 8,
                ring_dimension: 32,
                modulus: BigInt::from(257u16),
            })],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };
        let ranged = GpuNodeMeasurementBackend::representative_at(&request, 3, 2);
        assert!(matches!(
            ranged.kind,
            NodeKind::ConstantMatrix { value: ConstantMatrix::Identity, .. }
        ));
        assert_eq!(ranged.concrete_output_types[0].matrix_type().unwrap().columns, 8);
        assert_eq!(ranged.output_range.map(|range| (range.start, range.end)), Some((3, 5)));

        let unit_row_request = PendingMeasurement {
            warmup: None,
            kind: NodeKind::ConstantMatrix {
                matrix_type: MatrixType {
                    rows: IntExpr::constant(1),
                    columns: IntExpr::constant(8),
                    ring_dimension: IntExpr::constant(32),
                    modulus: IntExpr::constant(257),
                },
                value: ConstantMatrix::UnitRow { index: IntExpr::constant(6) },
            },
            concrete_output_types: vec![ConcreteWireType::Matrix(ConcreteMatrixType {
                rows: 1,
                columns: 8,
                ring_dimension: 32,
                modulus: BigInt::from(257u16),
            })],
            ..request
        };
        let unit_row_pilot =
            GpuNodeMeasurementBackend::calibration_representative(&unit_row_request);
        assert_eq!(unit_row_pilot.output_range.map(|range| (range.start, range.end)), Some((6, 7)));
    }

    #[test]
    fn transpose_and_column_concat_preserve_column_work_units() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                rows,
                columns,
                ring_dimension: 32,
                modulus: BigInt::from(257u16),
            })
        };
        let scope = FrozenGraphScopeId::Root;
        let transpose = NodeKind::Transpose;
        let transpose_node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &transpose,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![matrix(11, 7)],
            concrete_output_types: vec![matrix(7, 11)],
        };
        let (_, inputs, outputs, _, _) =
            GpuNodeMeasurementBackend::representative_node(&transpose_node, 3);
        assert_eq!(inputs[0].matrix_type().expect("transpose input").rows, 3);
        assert_eq!(outputs[0].matrix_type().expect("transpose output").columns, 3);

        let concat = NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Columns };
        let concat_node = MeasurementNode {
            scope: &scope,
            id: NodeId(2),
            kind: &concat,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![matrix(2, 2), matrix(2, 5)],
            concrete_output_types: vec![matrix(2, 7)],
        };
        let (_, inputs, outputs, _, _) =
            GpuNodeMeasurementBackend::representative_node(&concat_node, 3);
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].matrix_type().expect("first concat input").columns, 2);
        assert_eq!(inputs[1].matrix_type().expect("second concat input").columns, 1);
        assert_eq!(outputs[0].matrix_type().expect("concat output").columns, 3);
    }

    #[test]
    fn diagonal_concat_representative_keeps_every_row_block() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                rows,
                columns,
                ring_dimension: 32,
                modulus: BigInt::from(257u16),
            })
        };
        let scope = FrozenGraphScopeId::Root;
        let kind = NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Diagonal };
        let request = PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope,
            id: NodeId(1),
            kind,
            concrete_argument_types: vec![matrix(2, 8), matrix(3, 13), matrix(5, 21)],
            concrete_output_types: vec![matrix(10, 42)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };

        let representative = GpuNodeMeasurementBackend::representative_at(&request, 9, 1);
        assert_eq!(representative.concrete_argument_types.len(), 3);
        assert_eq!(
            representative
                .concrete_argument_types
                .iter()
                .map(|input| input.matrix_type().unwrap().columns)
                .collect::<Vec<_>>(),
            vec![8, 13, 21]
        );
        assert_eq!(representative.concrete_output_types[0].matrix_type().unwrap().columns, 42);
        assert_eq!(
            representative.output_range.as_ref().map(|range| (range.start, range.end)),
            Some((9, 10))
        );
        assert_eq!(representative.measured_columns(), Some(1));
    }

    #[test]
    fn matrix_storage_counts_entries_coefficients_and_crt_limbs() {
        let matrix = ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: 8,
            modulus: BigInt::from(257u16),
        };

        assert_eq!(matrix_bytes(&matrix, 4), 2 * 3 * 8 * 4 * 8);
    }

    #[test]
    fn compact_storage_uses_the_declared_bound_width() {
        let matrix = ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: 8,
            modulus: BigInt::from(257u16),
        };

        // 257 needs two magnitude bytes; every coefficient also carries one sign byte.
        assert_eq!(compact_matrix_bytes(&matrix, &BigInt::from(257u16)), 2 * 3 * 8 * 3);
    }

    #[test]
    fn measurement_cache_key_uses_semantics_not_node_identity() {
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: 8,
            modulus: BigInt::from(257u16),
        });
        let kind = NodeKind::MatrixNegate;
        let root = FrozenGraphScopeId::Root;
        let subgraph = FrozenGraphScopeId::Subgraph { canonical_name: "other".to_owned() };
        let first = MeasurementNode {
            scope: &root,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![matrix.clone()],
            concrete_output_types: vec![matrix.clone()],
        };
        let second = MeasurementNode {
            scope: &subgraph,
            id: NodeId(99),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![matrix.clone()],
            concrete_output_types: vec![matrix],
        };

        let first_key = GpuNodeMeasurementBackend::measurement_key(&first, &ParamEnv::default())
            .expect("cache key");
        let second_key = GpuNodeMeasurementBackend::measurement_key(&second, &ParamEnv::default())
            .expect("cache key");

        assert_eq!(first_key, second_key);
    }

    #[test]
    fn measurement_cache_key_ignores_loop_index_values() {
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 80,
            columns: 80,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        });
        let kind = NodeKind::HashSample {
            matrix_type: MatrixType {
                rows: IntExpr::constant(80),
                columns: IntExpr::constant(80),
                ring_dimension: IntExpr::constant(65_536),
                modulus: IntExpr::constant(257),
            },
            variant: HashVariant::Decomposed,
            tag_prefix: Vec::new(),
            tag_components: vec![mxx_ir_core::node::HashTagComponent::Integer(IntExpr::LoopIndex(
                0,
            ))],
            base: Some(IntExpr::constant(16_384)),
            digit_count: Some(IntExpr::constant(80)),
        };
        let scope = FrozenGraphScopeId::Root;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: Vec::new(),
            concrete_output_types: vec![matrix],
        };
        let mut first_bindings = ParamEnv::default();
        first_bindings.loop_indices.insert(0, BigInt::from(1));
        let mut second_bindings = ParamEnv::default();
        second_bindings.loop_indices.insert(0, BigInt::from(3_720));

        let first_key = GpuNodeMeasurementBackend::measurement_key(&node, &first_bindings)
            .expect("first cache key");
        let second_key = GpuNodeMeasurementBackend::measurement_key(&node, &second_bindings)
            .expect("second cache key");

        assert_eq!(first_key, second_key);
    }

    #[test]
    fn measurement_cache_key_ignores_ring_automorphism_index() {
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: 8,
            modulus: BigInt::from(257u16),
        });
        let first_kind = NodeKind::RingAutomorphism { index: IntExpr::constant(3) };
        let second_kind = NodeKind::RingAutomorphism { index: IntExpr::constant(5) };
        let scope = FrozenGraphScopeId::Root;
        let node = |kind| MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![matrix.clone()],
            concrete_output_types: vec![matrix.clone()],
        };

        let first_key =
            GpuNodeMeasurementBackend::measurement_key(&node(&first_kind), &ParamEnv::default())
                .expect("first cache key");
        let second_key =
            GpuNodeMeasurementBackend::measurement_key(&node(&second_kind), &ParamEnv::default())
                .expect("second cache key");

        assert_eq!(first_key, second_key);
    }

    #[test]
    fn ring_automorphism_measurement_uses_bounded_column_representative() {
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 2,
            columns: 10,
            ring_dimension: 8,
            modulus: BigInt::from(257u16),
        });
        let kind = NodeKind::RingAutomorphism { index: IntExpr::constant(3) };
        let scope = FrozenGraphScopeId::Root;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![matrix.clone()],
            concrete_output_types: vec![matrix],
        };

        let (representative_kind, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 4);

        assert_eq!(representative_kind, kind);
        assert_eq!(node.kind, &NodeKind::RingAutomorphism { index: IntExpr::constant(3) });
        assert_eq!(arguments[0].matrix_type().unwrap().columns, 4);
        assert_eq!(outputs[0].matrix_type().unwrap().columns, 4);
        assert_eq!(scale, 2.0);
        assert_eq!(remainder_columns, Some(2));
    }

    #[test]
    fn matrix_multiply_representatives_follow_scalar_runtime_semantics() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                rows,
                columns,
                ring_dimension: 8,
                modulus: BigInt::from(257u16),
            })
        };
        let kind = NodeKind::MatrixBinary(MatrixBinaryOp::Multiply);
        let scope = FrozenGraphScopeId::Root;
        let cases = [
            ([matrix(1, 1), matrix(1, 10)], matrix(1, 10), [1, 4], 0),
            ([matrix(2, 10), matrix(1, 1)], matrix(2, 10), [4, 1], 1),
            ([matrix(2, 3), matrix(3, 10)], matrix(2, 10), [3, 4], 0),
        ];

        for (arguments, output, expected_columns, fixed_argument) in cases {
            let node = MeasurementNode {
                scope: &scope,
                id: NodeId(1),
                kind: &kind,
                arguments: &[],
                argument_kinds: &[],
                argument_types: &[],
                output_types: &[],
                concrete_argument_types: arguments.into(),
                concrete_output_types: vec![output],
            };
            let (_, arguments, outputs, scale, remainder_columns) =
                GpuNodeMeasurementBackend::representative_node(&node, 4);

            assert_eq!(arguments[0].matrix_type().unwrap().columns, expected_columns[0]);
            assert_eq!(arguments[1].matrix_type().unwrap().columns, expected_columns[1]);
            assert_eq!(outputs[0].matrix_type().unwrap().columns, 4);
            assert_eq!(scale, 2.0);
            assert_eq!(remainder_columns, Some(2));
            assert!(GpuNodeMeasurementBackend::argument_is_fixed(&node, fixed_argument));
            assert!(!GpuNodeMeasurementBackend::argument_is_fixed(&node, 1 - fixed_argument));
        }
    }

    #[test]
    fn matrix_multiply_fixed_ownership_survives_one_column_calibration() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                rows,
                columns,
                ring_dimension: 8,
                modulus: BigInt::from(257u16),
            })
        };
        let request = PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(7),
            kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            concrete_argument_types: vec![matrix(2, 1), matrix(1, 176)],
            concrete_output_types: vec![matrix(2, 176)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };

        let pilot = GpuNodeMeasurementBackend::calibration_representative(&request);
        assert_eq!(pilot.concrete_argument_types[0].matrix_type().unwrap().columns, 1);
        assert_eq!(pilot.concrete_argument_types[1].matrix_type().unwrap().columns, 1);
        assert_eq!(pilot.concrete_output_types[0].matrix_type().unwrap().columns, 1);
        assert_eq!(pilot.fixed_arguments, vec![true, false]);

        // The shaped pilot alone is ambiguous: it looks like a scalar-right product.  Fixed
        // ownership must therefore remain the decision made from the unshaped request.
        let pilot_node = MeasurementNode {
            scope: &request.scope,
            id: request.id,
            kind: &pilot.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: pilot.concrete_argument_types.clone(),
            concrete_output_types: pilot.concrete_output_types.clone(),
        };
        assert_eq!(
            (0..2)
                .map(|index| GpuNodeMeasurementBackend::argument_is_fixed(&pilot_node, index))
                .collect::<Vec<_>>(),
            vec![false, true]
        );

        let gpu_representative = GpuNodeMeasurementBackend::representative_at(&request, 0, 88);
        assert_eq!(gpu_representative.concrete_argument_types[0].matrix_type().unwrap().columns, 1);
        assert_eq!(
            gpu_representative.concrete_argument_types[1].matrix_type().unwrap().columns,
            176
        );
        assert_eq!(gpu_representative.concrete_output_types[0].matrix_type().unwrap().columns, 176);
        assert_eq!(gpu_representative.fixed_arguments, vec![true, false]);
    }

    #[test]
    fn fused_multiply_with_scalar_rhs_uses_left_column_representative() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                rows,
                columns,
                ring_dimension: 8,
                modulus: BigInt::from(257u16),
            })
        };
        let request = PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(1),
            kind: NodeKind::MatrixMulAccumulate {
                coefficients: vec![IntExpr::constant(1)],
                has_bias: false,
            },
            concrete_argument_types: vec![matrix(2, 10), matrix(1, 1)],
            concrete_output_types: vec![matrix(2, 10)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };

        assert_eq!(GpuNodeMeasurementBackend::request_columns(&request), Some(10));
        let node = MeasurementNode {
            scope: &request.scope,
            id: request.id,
            kind: &request.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: request.concrete_argument_types.clone(),
            concrete_output_types: request.concrete_output_types.clone(),
        };
        let (_, arguments, outputs, scale, remainder) =
            GpuNodeMeasurementBackend::representative_node(&node, 4);
        assert_eq!(arguments[0].matrix_type().unwrap().columns, 4);
        assert_eq!(arguments[1].matrix_type().unwrap().columns, 1);
        assert_eq!(outputs[0].matrix_type().unwrap().columns, 4);
        assert_eq!(scale, 2.0);
        assert_eq!(remainder, Some(2));
        assert!(!GpuNodeMeasurementBackend::argument_is_fixed(&node, 0));
        assert!(GpuNodeMeasurementBackend::argument_is_fixed(&node, 1));
    }

    #[test]
    fn gadget_trapdoor_ranged_representative_preserves_full_identity() {
        let concrete = ConcreteMatrixType {
            rows: 2,
            columns: 10,
            ring_dimension: 8,
            modulus: BigInt::from(257u16),
        };
        let kind = NodeKind::GadgetTrapdoor {
            matrix_type: MatrixType {
                rows: IntExpr::constant(2),
                columns: IntExpr::constant(10),
                ring_dimension: IntExpr::constant(8),
                modulus: IntExpr::constant(257),
            },
            base: IntExpr::constant(4),
        };
        let request = PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(1),
            kind: kind.clone(),
            concrete_argument_types: Vec::new(),
            concrete_output_types: vec![ConcreteWireType::Trapdoor {
                matrix: concrete,
                sigma: RealExpr::from_integer(4),
                gadget_base: BigInt::from(4),
                digit_count: 5,
                preimage_max_coefficient_bound: BigInt::from(0),
            }],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };

        assert!(GpuNodeMeasurementBackend::column_separable(&kind));
        let representative = GpuNodeMeasurementBackend::representative_at(&request, 3, 4);
        let NodeKind::GadgetTrapdoor { matrix_type, .. } = &representative.kind else {
            panic!("gadget trapdoor representative kind");
        };
        assert_eq!(matrix_type.columns, IntExpr::constant(10));
        assert_eq!(representative.concrete_output_types[0].matrix_type().unwrap().columns, 10);
        assert_eq!(representative.measured_columns(), Some(4));
        assert_eq!(representative.output_range.map(|range| (range.start, range.end)), Some((3, 7)));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_gadget_trapdoor_ranged_measurement_matches_full_gadget_slice() {
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let rows = 2;
        let full_columns = rows * params.modulus_digits();
        let full_ty = ConcreteMatrixType {
            rows,
            columns: full_columns,
            ring_dimension: 32,
            modulus: BigInt::from(131_009u32),
        };
        let kind = NodeKind::GadgetTrapdoor {
            matrix_type: MatrixType {
                rows: IntExpr::constant(rows),
                columns: IntExpr::constant(full_columns),
                ring_dimension: IntExpr::constant(32),
                modulus: IntExpr::constant(131_009),
            },
            base: IntExpr::constant(4),
        };
        let scope = FrozenGraphScopeId::Root;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: Vec::new(),
            concrete_output_types: vec![ConcreteWireType::Trapdoor {
                matrix: full_ty.clone(),
                sigma: RealExpr::from_integer(4),
                gadget_base: BigInt::from(4),
                digit_count: params.modulus_digits(),
                preimage_max_coefficient_bound: BigInt::from(0),
            }],
        };
        let prepared = PreparedMeasurement {
            arguments: Vec::new(),
            small_arguments: Vec::new(),
            preimage_trapdoor: None,
            preimage_target: None,
        };
        let range = mxx_runtime::backend::IndexRange { start: 7, end: 12 };
        let operation = [73u8; 32];
        let mut backend = gpu_backend([params.clone()]);
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: range.end - range.start, nonzero: None },
        );
        backend.select_operation(operation).unwrap();

        for (declared_base, declared_digits) in [
            (BigInt::from(8), params.modulus_digits()),
            (BigInt::from(4), params.modulus_digits() + 1),
        ] {
            let error = backend
                .measurement_gadget_columns(
                    &full_ty,
                    &declared_base,
                    declared_digits,
                    range.start,
                    range.end - range.start,
                )
                .unwrap_err();
            assert!(matches!(
                error,
                PolyBackendError::GadgetLayoutMismatch {
                    declared_base: actual_base,
                    declared_digits: actual_digits,
                    backend_base,
                    backend_digits,
                } if actual_base == declared_base &&
                    actual_digits == declared_digits &&
                    backend_base == BigInt::from(4) &&
                    backend_digits == params.modulus_digits()
            ));
        }

        let mut outputs = GpuNodeMeasurementBackend::run_node(
            &mut backend,
            &node,
            &ParamEnv::default(),
            1,
            &prepared,
            Some(&range),
        )
        .unwrap();
        let GpuMeasurementOutput::Matrix(actual) = outputs.pop().unwrap() else {
            panic!("gadget trapdoor measurement must produce a matrix");
        };
        assert!(outputs.is_empty());
        let actual = &actual.shards()[0].value;
        let expected = GpuDCRTPolyMatrix::gadget_matrix(actual.params(), rows, None)
            .slice_columns(range.start, range.end);
        assert_eq!(actual.to_cpu_matrix(), expected.to_cpu_matrix());
    }

    #[test]
    fn measurement_cache_key_ignores_hash_tag_values() {
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 80,
            columns: 80,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        });
        let hash_kind = |tag_prefix, tag_expression| NodeKind::HashSample {
            matrix_type: MatrixType {
                rows: IntExpr::constant(80),
                columns: IntExpr::constant(80),
                ring_dimension: IntExpr::constant(65_536),
                modulus: IntExpr::constant(257),
            },
            variant: HashVariant::Decomposed,
            tag_prefix,
            tag_components: vec![mxx_ir_core::node::HashTagComponent::Integer(tag_expression)],
            base: Some(IntExpr::constant(16_384)),
            digit_count: Some(IntExpr::constant(80)),
        };
        let first_kind = hash_kind(vec![1, 2, 3], IntExpr::constant(7));
        let second_kind = hash_kind(vec![9, 8, 7], IntExpr::constant(3_720));
        let scope = FrozenGraphScopeId::Root;
        let node = |kind| MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: Vec::new(),
            concrete_output_types: vec![matrix.clone()],
        };

        let first = node(&first_kind);
        let second = node(&second_kind);
        let first_key = GpuNodeMeasurementBackend::measurement_key(&first, &ParamEnv::default())
            .expect("first cache key");
        let second_key = GpuNodeMeasurementBackend::measurement_key(&second, &ParamEnv::default())
            .expect("second cache key");

        assert_eq!(first_key, second_key);
    }

    #[test]
    fn oversized_hash_measurement_is_representative_and_scaled() {
        let concrete = ConcreteMatrixType {
            rows: 1,
            columns: 8_722,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let symbolic = MatrixType {
            rows: IntExpr::constant(concrete.rows),
            columns: IntExpr::constant(concrete.columns),
            ring_dimension: IntExpr::constant(concrete.ring_dimension),
            modulus: IntExpr::constant(concrete.modulus.clone()),
        };
        let kind = NodeKind::HashSample {
            matrix_type: symbolic,
            variant: HashVariant::Plain,
            tag_prefix: Vec::new(),
            tag_components: Vec::new(),
            base: None,
            digit_count: None,
        };
        let scope = FrozenGraphScopeId::Root;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: Vec::new(),
            concrete_output_types: vec![ConcreteWireType::Matrix(concrete)],
        };

        let (kind, _, output_types, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 4);
        let NodeKind::HashSample { matrix_type, .. } = kind else {
            panic!("hash representative kind");
        };
        let ConcreteWireType::Matrix(output) = &output_types[0] else {
            panic!("hash representative output");
        };
        assert_eq!((output.rows, output.columns), (1, 4));
        assert_eq!(matrix_type.columns, IntExpr::constant(4));
        assert_eq!(scale, 2_180.0);
        assert_eq!(remainder_columns, Some(2));
    }

    #[test]
    fn single_column_sampler_preserves_all_rows() {
        let concrete = ConcreteMatrixType {
            rows: 2_621_440,
            columns: 1,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let symbolic = MatrixType {
            rows: IntExpr::constant(concrete.rows),
            columns: IntExpr::constant(concrete.columns),
            ring_dimension: IntExpr::constant(concrete.ring_dimension),
            modulus: IntExpr::constant(concrete.modulus.clone()),
        };
        let kind = NodeKind::UniformIntervalSample {
            matrix_type: symbolic,
            range: mxx_ir_core::node::SampleRange {
                minimum: IntExpr::constant(-1),
                maximum: IntExpr::constant(1),
            },
        };
        let scope = FrozenGraphScopeId::Root;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: Vec::new(),
            concrete_output_types: vec![ConcreteWireType::Matrix(concrete)],
        };

        let (kind, _, output_types, scale, _) =
            GpuNodeMeasurementBackend::representative_node(&node, 4);
        let NodeKind::UniformIntervalSample { matrix_type, .. } = kind else {
            panic!("uniform representative kind");
        };
        let ConcreteWireType::Matrix(output) = &output_types[0] else {
            panic!("uniform representative output");
        };
        assert_eq!((output.rows, output.columns), (2_621_440, 1));
        assert_eq!(matrix_type.rows, IntExpr::constant(2_621_440));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn slice_measurement_uses_representative_column_limit() {
        let matrix = |columns| ConcreteMatrixType {
            rows: 1,
            columns,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let kind = NodeKind::Slice {
            rows: None,
            columns: Some(IndexRange { start: IntExpr::constant(80), end: IntExpr::constant(160) }),
        };
        let scope = FrozenGraphScopeId::Root;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![ConcreteWireType::Matrix(matrix(8_720))],
            concrete_output_types: vec![ConcreteWireType::Matrix(matrix(80))],
        };

        let (kind, argument_types, _, scale, _) =
            GpuNodeMeasurementBackend::representative_node(&node, 4);
        let NodeKind::Slice { columns: Some(columns), .. } = kind else {
            panic!("slice representative kind");
        };
        let ConcreteWireType::Matrix(input) = &argument_types[0] else {
            panic!("slice representative input");
        };
        assert_eq!((input.rows, input.columns), (1, 4));
        assert_eq!(columns.start, IntExpr::constant(0));
        assert_eq!(columns.end, IntExpr::constant(4));
        assert_eq!(scale, 20.0);
    }

    #[test]
    fn ranged_representatives_cover_generated_and_mapped_primitives() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                rows,
                columns,
                ring_dimension: 8,
                modulus: BigInt::from(17),
            })
        };
        let matrix_type = |rows, columns| MatrixType {
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
            ring_dimension: IntExpr::constant(8),
            modulus: IntExpr::constant(17),
        };
        let pending = |kind, arguments, outputs| PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(1),
            kind,
            concrete_argument_types: arguments,
            concrete_output_types: outputs,
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };
        let start = 13;
        let width = 4;

        let sampler = pending(
            NodeKind::UniformResidueSample { matrix_type: matrix_type(2, 64) },
            Vec::new(),
            vec![matrix(2, 64)],
        );
        let representative = GpuNodeMeasurementBackend::representative_at(&sampler, start, width);
        assert_eq!(
            representative.output_range,
            Some(RuntimeIndexRange { start, end: start + width })
        );
        assert_eq!(representative.concrete_output_types[0].matrix_type().unwrap().columns, 64);

        let slice = pending(
            NodeKind::Slice {
                rows: None,
                columns: Some(IndexRange {
                    start: IntExpr::constant(32),
                    end: IntExpr::constant(96),
                }),
            },
            vec![matrix(3, 128)],
            vec![matrix(3, 64)],
        );
        let representative = GpuNodeMeasurementBackend::representative_at(&slice, start, width);
        assert_eq!(
            representative.output_range,
            Some(RuntimeIndexRange { start, end: start + width })
        );
        assert_eq!(representative.concrete_argument_types[0].matrix_type().unwrap().columns, 128);
        let NodeKind::Slice { columns: Some(columns), .. } = &representative.kind else {
            panic!("slice representative kind");
        };
        assert_eq!(
            (columns.start.clone(), columns.end.clone()),
            (IntExpr::constant(32), IntExpr::constant(96))
        );

        let transpose = pending(NodeKind::Transpose, vec![matrix(64, 3)], vec![matrix(3, 64)]);
        let representative = GpuNodeMeasurementBackend::representative_at(&transpose, start, width);
        assert_eq!(
            representative.output_range,
            Some(RuntimeIndexRange { start, end: start + width })
        );
        assert_eq!(representative.concrete_argument_types[0].matrix_type().unwrap().rows, 64);
        assert_eq!(representative.concrete_output_types[0].matrix_type().unwrap().columns, 64);

        let row_concat = pending(
            NodeKind::Concat { axis: ConcatAxis::Rows },
            vec![matrix(2, 64), matrix(3, 64)],
            vec![matrix(5, 64)],
        );
        let representative =
            GpuNodeMeasurementBackend::representative_at(&row_concat, start, width);
        assert_eq!(
            representative.output_range,
            Some(RuntimeIndexRange { start, end: start + width })
        );
        assert!(
            representative.concrete_argument_types.iter().all(|wire| wire
                .matrix_type()
                .unwrap()
                .columns ==
                64)
        );
        assert_eq!(representative.concrete_output_types[0].matrix_type().unwrap().columns, 64);
    }

    #[test]
    fn tensor_fleet_representative_measures_exact_assigned_range() {
        let matrix = |rows, columns| ConcreteMatrixType {
            rows,
            columns,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let kind = NodeKind::Tensor;
        let scope = FrozenGraphScopeId::Root;
        let request = PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope,
            id: NodeId(1),
            kind,
            concrete_argument_types: vec![
                ConcreteWireType::Matrix(matrix(2, 2)),
                ConcreteWireType::Matrix(matrix(3, 40)),
            ],
            concrete_output_types: vec![ConcreteWireType::Matrix(matrix(6, 80))],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };

        let representative = GpuNodeMeasurementBackend::representative_at(&request, 35, 45);
        let ConcreteWireType::Matrix(left) = &representative.concrete_argument_types[0] else {
            panic!("tensor representative left input");
        };
        let ConcreteWireType::Matrix(right) = &representative.concrete_argument_types[1] else {
            panic!("tensor representative right input");
        };
        let ConcreteWireType::Matrix(output) = &representative.concrete_output_types[0] else {
            panic!("tensor representative output");
        };
        assert_eq!((left.columns, right.columns), (2, 40));
        assert_eq!(output.columns, 80);
        assert_eq!(representative.measured_columns(), Some(45));
        assert_eq!(
            representative.output_range.as_ref().map(|range| (range.start, range.end)),
            Some((35, 80))
        );
    }

    #[test]
    fn column_concat_fleet_representative_uses_actual_global_intersections() {
        let matrix = |columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                rows: 2,
                columns,
                ring_dimension: 32,
                modulus: BigInt::from(257u16),
            })
        };
        let request = PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(1),
            kind: NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Columns },
            concrete_argument_types: vec![matrix(2), matrix(100), matrix(3)],
            concrete_output_types: vec![matrix(105)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };

        let representative = GpuNodeMeasurementBackend::representative_at(&request, 50, 4);
        assert_eq!(representative.concrete_argument_types.len(), 3);
        assert_eq!(
            representative
                .concrete_argument_types
                .iter()
                .map(|input| input.matrix_type().unwrap().columns)
                .collect::<Vec<_>>(),
            vec![2, 100, 3]
        );
        assert_eq!(representative.measured_columns(), Some(4));

        let crossing = GpuNodeMeasurementBackend::representative_at(&request, 1, 4);
        assert_eq!(crossing.concrete_argument_types.len(), 3);
        assert_eq!(
            crossing
                .concrete_argument_types
                .iter()
                .map(|input| input.matrix_type().unwrap().columns)
                .collect::<Vec<_>>(),
            vec![2, 100, 3]
        );
        assert_eq!(crossing.measured_columns(), Some(4));

        let pilot = GpuNodeMeasurementBackend::calibration_representative(&request);
        assert_eq!(pilot.measured_columns(), Some(2));
        assert_eq!(pilot.concrete_argument_types.len(), 3);
        assert_eq!(
            pilot
                .concrete_argument_types
                .iter()
                .map(|input| input.matrix_type().unwrap().columns)
                .collect::<Vec<_>>(),
            vec![2, 100, 3]
        );

        let two_column_request = PendingMeasurement {
            warmup: None,
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(4),
            kind: NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Columns },
            concrete_argument_types: vec![matrix(1), matrix(1)],
            concrete_output_types: vec![matrix(2)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };
        let assignments =
            gpu_capped_waterfill_columns(GpuColumnWidths { gpu0: 1154, nonzero: Some(1154) }, 2, 2)
                .unwrap();
        assert_eq!(assignments, vec![1, 1]);
        let mut start = 0;
        for columns in assignments {
            let representative =
                GpuNodeMeasurementBackend::representative_at(&two_column_request, start, columns);
            assert_eq!(representative.concrete_argument_types.len(), 2);
            assert_eq!(representative.fixed_arguments, vec![false, false]);
            assert_eq!(
                representative.output_range.as_ref().map(|range| (range.start, range.end)),
                Some((start, start + 1))
            );
            let node = MeasurementNode {
                scope: &two_column_request.scope,
                id: two_column_request.id,
                kind: &representative.kind,
                arguments: &[],
                argument_kinds: &[],
                argument_types: &[],
                output_types: &[],
                concrete_argument_types: representative.concrete_argument_types.clone(),
                concrete_output_types: representative.concrete_output_types.clone(),
            };
            GpuNodeMeasurementBackend::verify_concat_representative(
                &node,
                mxx_ir_core::node::ConcatAxis::Columns,
                representative.output_range.as_ref(),
            )
            .unwrap();

            let empty_prepared = |arguments| PreparedMeasurement {
                arguments: (0..arguments).map(|_| None).collect(),
                small_arguments: (0..arguments).map(|_| None).collect(),
                preimage_trapdoor: None,
                preimage_target: None,
            };
            let merged = empty_prepared(2)
                .merge_for_representative(empty_prepared(2), &representative.fixed_arguments)
                .unwrap();
            assert_eq!(merged.arguments.len(), 2);
            start += columns;
        }
        assert_eq!(start, 2);
    }

    #[test]
    fn column_separable_measurements_do_not_materialize_full_digit_matrices() {
        let matrix = |rows, columns| ConcreteMatrixType {
            rows,
            columns,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let scope = FrozenGraphScopeId::Root;
        let gadget_kind = NodeKind::GadgetDecompose {
            base: IntExpr::constant(16_384),
            small: false,
            digit_count: IntExpr::constant(80),
        };
        let gadget_node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &gadget_kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![ConcreteWireType::Matrix(matrix(1, 80))],
            concrete_output_types: vec![ConcreteWireType::Preimage {
                matrix: matrix(80, 80),
                max_coefficient_bound: BigInt::from(8_192),
            }],
        };
        let (_, gadget_arguments, gadget_outputs, gadget_scale, _) =
            GpuNodeMeasurementBackend::representative_node(&gadget_node, 4);
        let ConcreteWireType::Matrix(gadget_input) = &gadget_arguments[0] else {
            panic!("gadget representative input");
        };
        let ConcreteWireType::Preimage { matrix: gadget_output, max_coefficient_bound } =
            &gadget_outputs[0]
        else {
            panic!("gadget representative output");
        };
        assert_eq!((gadget_input.rows, gadget_input.columns), (1, 4));
        assert_eq!((gadget_output.rows, gadget_output.columns), (80, 4));
        assert_eq!(max_coefficient_bound, &BigInt::from(8_192));
        assert_eq!(gadget_scale, 20.0);

        let multiply_kind = NodeKind::MatrixBinary(MatrixBinaryOp::Multiply);
        let multiply_node = MeasurementNode {
            scope: &scope,
            id: NodeId(2),
            kind: &multiply_kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![
                ConcreteWireType::Matrix(matrix(1, 80)),
                ConcreteWireType::Matrix(matrix(80, 80)),
            ],
            concrete_output_types: vec![ConcreteWireType::Matrix(matrix(1, 80))],
        };
        let (_, multiply_arguments, multiply_outputs, multiply_scale, _) =
            GpuNodeMeasurementBackend::representative_node(&multiply_node, 4);
        let ConcreteWireType::Matrix(rhs) = &multiply_arguments[1] else {
            panic!("multiply representative rhs");
        };
        let ConcreteWireType::Matrix(product) = &multiply_outputs[0] else {
            panic!("multiply representative output");
        };
        assert_eq!((rhs.rows, rhs.columns), (80, 4));
        assert_eq!((product.rows, product.columns), (1, 4));
        assert_eq!(multiply_scale, 20.0);

        let accumulate_kind = NodeKind::MatrixMulAccumulate {
            coefficients: vec![IntExpr::constant(1), IntExpr::constant(3)],
            has_bias: true,
        };
        let accumulate_node = MeasurementNode {
            scope: &scope,
            id: NodeId(3),
            kind: &accumulate_kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![
                ConcreteWireType::Matrix(matrix(1, 82)),
                ConcreteWireType::Matrix(matrix(82, 80)),
                ConcreteWireType::Matrix(matrix(1, 80)),
                ConcreteWireType::Matrix(matrix(80, 80)),
                ConcreteWireType::Matrix(matrix(1, 80)),
            ],
            concrete_output_types: vec![ConcreteWireType::Matrix(matrix(1, 80))],
        };
        let (_, accumulate_arguments, accumulate_outputs, accumulate_scale, _) =
            GpuNodeMeasurementBackend::representative_node(&accumulate_node, 4);
        for index in [1, 3, 4] {
            let ConcreteWireType::Matrix(matrix) = &accumulate_arguments[index] else {
                panic!("multiply-accumulate representative matrix input");
            };
            assert_eq!(matrix.columns, 4);
        }
        let ConcreteWireType::Matrix(accumulate_output) = &accumulate_outputs[0] else {
            panic!("multiply-accumulate representative output");
        };
        assert_eq!((accumulate_output.rows, accumulate_output.columns), (1, 4));
        assert_eq!(accumulate_scale, 20.0);
    }

    #[test]
    fn preimage_measurement_uses_representative_column_limit() {
        let matrix = |rows, columns| ConcreteMatrixType {
            rows,
            columns,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let symbolic = MatrixType {
            rows: IntExpr::constant(82),
            columns: IntExpr::constant(80),
            ring_dimension: IntExpr::constant(65_536),
            modulus: IntExpr::constant(257),
        };
        let kind = NodeKind::PreimageSample {
            matrix_type: symbolic,
            max_coefficient_bound: IntExpr::constant(100),
        };
        let scope = FrozenGraphScopeId::Root;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![
                ConcreteWireType::Matrix(matrix(1, 82)),
                ConcreteWireType::Bytes { length: 0 },
                ConcreteWireType::Matrix(matrix(1, 80)),
            ],
            concrete_output_types: vec![ConcreteWireType::Preimage {
                matrix: matrix(82, 80),
                max_coefficient_bound: BigInt::from(100),
            }],
        };

        let (kind, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 4);
        let NodeKind::PreimageSample { matrix_type, .. } = kind else {
            panic!("preimage representative kind");
        };
        let ConcreteWireType::Matrix(target) = &arguments[2] else {
            panic!("preimage representative target");
        };
        let ConcreteWireType::Preimage { matrix: output, max_coefficient_bound } = &outputs[0]
        else {
            panic!("preimage representative output");
        };
        assert_eq!(target.columns, 4);
        assert_eq!((output.rows, output.columns), (82, 4));
        assert_eq!(matrix_type.columns, IntExpr::constant(4));
        assert_eq!(max_coefficient_bound, &BigInt::from(100));
        assert_eq!(scale, 20.0);
        assert_eq!(remainder_columns, None);

        let (_, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 12);
        let ConcreteWireType::Matrix(target) = &arguments[2] else {
            panic!("preimage representative target");
        };
        let ConcreteWireType::Preimage { matrix: output, .. } = &outputs[0] else {
            panic!("preimage representative output");
        };
        assert_eq!(target.columns, 12);
        assert_eq!((output.rows, output.columns), (82, 12));
        assert_eq!(scale, 6.0);
        assert_eq!(remainder_columns, Some(8));
    }

    #[test]
    fn preimage_evidence_maps_fixed_width_classes_without_scaling() {
        let evidence = PreimageAllocationEvidence {
            context: PreimageEvidenceContext {
                state: PreimageCacheState::Cold,
                tile_columns: 4,
                max_attempts: 7,
                max_coefficient_bound: BigUint::from(31u8),
                hard_cutoff_plan_bytes: 13,
                format: PreimageFormatContext {
                    active_moduli: vec![17, 257],
                    crt_level: 1,
                    coefficient_magnitude_bytes: 1,
                    compact_output: true,
                },
                source_device_ids: vec![0],
                destination_device_ids: vec![0],
                cache_identity: PreimageCacheIdentity {
                    owner_token: 9,
                    c: 1.0,
                    smoothing: 2.0,
                    dgg_stddev: 3.0,
                },
            },
            envelope: PreimageAllocationEnvelope {
                public_matrix_bytes: 10,
                trapdoor_bytes: 20,
                resident_target_bytes: 30,
                retained_covariance_cache_bytes: 40,
                compact_output_bytes: 50,
                candidate_workspace_bytes: 60,
                perturbation_workspace_bytes: 70,
                scratch_bytes: 80,
                source_device_staging_bytes: 90,
                destination_device_staging_bytes: 100,
                host_staging_bytes: 110,
                device_control_bytes: 120,
                pinned_host_control_bytes: 130,
                sampler_event_bytes: 140,
                cold_transient_workspace_bytes: 150,
                sampler_peak_bytes: 0,
                cold_sampler_peak_bytes: 0,
                evidence_kind: PreimageAllocationEvidenceKind::Certified,
            },
        };
        let footprint = GpuNodeMeasurementBackend::preimage_footprint_from_evidence(&evidence)
            .expect("native fixed-width evidence should map");
        assert_eq!(footprint.certified_tile_width, Some(4));
        assert_eq!(footprint.persistent.live, 60);
        assert_eq!(footprint.compact.outputs, 50);
        assert_eq!(footprint.scratch.scratch, 210);
        assert_eq!(footprint.scratch.transfers, 190);
        assert_eq!(footprint.control.live, 260);
        assert_eq!(footprint.control.pinned_host, 130);
        assert_eq!(footprint.control.host, 110);
        assert_eq!(footprint.cold_cache.caches, 40);
        assert_eq!(footprint.cold_transient_workspace.scratch, 150);
        assert_eq!(footprint.per_tile, GpuResourceCost::zero());
    }

    /// Exercise the complete setup boundary with the production measurement
    /// adapter. This deliberately does not seed `profiles` or inject a
    /// synthetic provider: every profile point used by the frozen plan is
    /// collected from the device-local production range.
    #[cfg(feature = "gpu")]
    #[test]
    #[serial_test::serial(gpu_context)]
    fn real_provider_freezes_fused_add_plan_and_fixed_gpu_execution_is_read_only() {
        use mxx_dsl::{DslContext, Mat, Ring};
        use mxx_ir_core::node::ConcatAxis;
        use mxx_primitives::poly::dcrt::gpu::{
            GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync,
        };
        use mxx_runtime::{
            Backend, RuntimeValue,
            artifact::MemoryArtifactStore,
            backend::{GpuWarmupProvenance, poly_gpu::gpu_backend_on},
            executor::execute_with_gpu_plan,
            gpu_warmup::{GpuProfileProvenance, GpuStageCostModel, GpuValidatedWarmupConfig},
            transcript::SamplingMode,
        };
        use num_bigint::{BigInt, Sign};
        use std::{collections::BTreeMap, num::NonZeroUsize, time::Duration};

        let device = detected_gpu_device_ids()
            .into_iter()
            .next()
            .expect("GPU feature tests require one detected device");
        let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, 8usize);
        let left = ring.input("left", (1, 3));
        let right = ring.input("right", (2, 3));
        let blocks = Mat::concat(ConcatAxis::Rows, vec![left.clone(), right.clone()]);
        let reversed = Mat::concat(ConcatAxis::Rows, vec![right.clone(), left.clone()]);
        let graph = DslContext::new("bench-real-provider-row-block-add")
            .output("out", blocks + reversed)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();

        let mut layouts = BTreeMap::<ConcreteMatrixType, LayoutId>::new();
        for checked in graph.scopes.values() {
            for ty in checked.wire_types.values() {
                if let Some(matrix) = ty.matrix_type() {
                    let id = layouts.len() as LayoutId + 1;
                    layouts.entry(matrix.clone()).or_insert(id);
                }
            }
        }
        let frozen_layouts = layouts
            .iter()
            .map(|(matrix, id)| GpuLayout {
                id: *id,
                columns: matrix.columns,
                rows: matrix.rows,
                ring_dimension: matrix.ring_dimension,
                representation: format!("{:?}", ConcreteWireType::Matrix(matrix.clone())),
                instance_device_stride: 0,
                owner_intervals: Vec::new(),
            })
            .collect::<Vec<_>>();

        // Keep setup measurement and fixed execution sequential. The provider
        // owns the only live measurement backend while sampling; it is dropped
        // before creating the production execution backend.
        let harness = MeasurementHarnessConfig {
            warm_up_iterations: 0,
            measured_iterations: 1,
            memory_poll_interval: Duration::ZERO,
        };
        let measurement_backend = gpu_backend_on([parameters.clone()], [device]);
        let mut provider =
            GpuNodeMeasurementBackend::new(vec![(measurement_backend, device)], harness);
        let config = GpuValidatedWarmupConfig {
            contract: GpuPlanContract {
                graph_specification_hash: [0; 32],
                backend_identity: "bench-real-provider-placeholder".into(),
                logical_to_physical_devices: vec![device as usize],
                device_budgets: vec![GpuDeviceBudget {
                    device: 0,
                    device_bytes: u64::MAX,
                    pinned_host_bytes: u64::MAX,
                    host_bytes: u64::MAX,
                }],
                shape_contract_hash: [0; 32],
                backend_revision: "bench-real-provider-placeholder".into(),
            },
            layouts: frozen_layouts,
            default_tile_widths: vec![1, 2, 3],
            // Deliberately unreachable fallback values. The measured
            // provenance assertion below prevents a provider miss from
            // silently making this test pass.
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "bench-real-provider-placeholder".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 2,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(1).unwrap(),
        };
        let mut warmup = mxx_runtime::gpu_warmup::warmup_gpu_from_validated_with_provider(
            &graph,
            &config,
            &mut provider,
        )
        .expect("real provider must collect all row-block-add candidates");
        assert!(!warmup.plan.nodes.is_empty());
        // Canonical warmup points live in the runtime session cache; the
        // provider's legacy fleet-calibration registry is intentionally not
        // populated by this setup path.  Prove that setup really entered the
        // production measurement boundary and returned measured provenance.
        assert!(provider.warmup_measurement_call_count() > 0);
        assert!(!provider.warmup_measurement_provenances().is_empty());
        assert!(
            provider.warmup_dispatch_records().iter().any(|record| record.profile_domain ==
                CanonicalWarmupProfileDomain::FusedRowBlockAdd &&
                record.fused_operation == Some(FusedWarmupOperation::RowBlockAdd) &&
                record.measurement == WarmupMeasurementKind::GpuMeasured &&
                record.range.start < record.range.end),
            "row-block add warmup must enter the fused production range API"
        );
        assert!(
            provider
                .warmup_measurement_provenances()
                .iter()
                .all(|provenance| *provenance == GpuWarmupProvenance::ProductionEquivalent)
        );
        assert!(warmup.report.predicted_seconds.is_finite());
        assert!(warmup.report.predicted_seconds > 0.0);
        assert!(
            warmup
                .report
                .stages
                .iter()
                .all(|stage| stage.provenance != GpuProfileProvenance::ConservativeEstimate)
        );
        let provider_measurement_count = provider.warmup_measurement_call_count();
        let provider_measurement_counter = provider.warmup_measurement_counter();
        drop(provider);
        gpu_device_sync();

        // Create the real execution backend only after setup measurement. Its
        // runtime contract replaces the setup-only placeholder in the frozen
        // value plan; no profile point is recomputed here.
        let mut backend = gpu_backend_on([parameters.clone()], [device]);
        // Direct backend sampling is setup scaffolding for the E2E assertion,
        // not a planned graph dispatch.  Select a temporary operation so the
        // normal runtime pilot derives widths before the first generated
        // input; the frozen graph plan is installed below for production.
        backend.select_operation([0xA5; 32]).unwrap();
        let matrix_type = |rows, columns| ConcreteMatrixType {
            modulus: BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
            rows,
            columns,
        };
        let left_value = backend.sample_hash(&matrix_type(1, 3), [1; 32], b"bench-left").unwrap();
        let right_value = backend.sample_hash(&matrix_type(2, 3), [2; 32], b"bench-right").unwrap();
        let expected_rhs = backend.concat(&[&right_value, &left_value], ConcatAxis::Rows).unwrap();
        let expected = backend.add_row_blocks(&[&left_value, &right_value], &expected_rhs).unwrap();
        let inputs = BTreeMap::from([
            ("left".to_owned(), RuntimeValue::matrix(left_value)),
            ("right".to_owned(), RuntimeValue::matrix(right_value)),
        ]);
        warmup.plan.contract = backend
            .gpu_runtime_contract(&graph, &inputs)
            .unwrap()
            .expect("execution backend must expose a GPU contract");
        warmup.plan.validate().unwrap();
        let output = execute_with_gpu_plan(
            &graph,
            &warmup.plan,
            &mut backend,
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("frozen plan must execute through the real GPU backend");
        let RuntimeValue::Matrix(actual) = &output.outputs["out"] else {
            panic!("row-block add output is not a matrix");
        };
        assert_eq!(actual.size(), expected.size());
        assert_eq!(backend.matrix_to_bytes(actual), backend.matrix_to_bytes(&expected));
        assert_eq!(
            provider_measurement_counter.load(std::sync::atomic::Ordering::SeqCst),
            provider_measurement_count
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[serial_test::serial(gpu_context)]
    fn real_provider_freezes_preimage_rhs_compact_product_and_fixed_outputs() {
        use mxx_dsl::{DslContext, Mat, Ring};
        use mxx_ir_core::node::{ConcatAxis, IndexRange};
        use mxx_primitives::poly::dcrt::gpu::{
            GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync,
        };
        use mxx_runtime::{
            Backend, RuntimeValue,
            artifact::MemoryArtifactStore,
            backend::poly_gpu::gpu_backend_on,
            executor::execute_with_gpu_plan,
            gpu_warmup::{GpuProfileProvenance, GpuStageCostModel, GpuValidatedWarmupConfig},
            transcript::SamplingMode,
        };
        use num_bigint::{BigInt, Sign};
        use std::{collections::BTreeMap, num::NonZeroUsize, time::Duration};

        let device = detected_gpu_device_ids()
            .into_iter()
            .next()
            .expect("GPU feature tests require one detected device");
        let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, 8usize);
        let digits = parameters.modulus_digits();
        let compact_input = Mat::concat(
            ConcatAxis::Rows,
            vec![ring.input("compact-a", (1, 2)), ring.input("compact-b", (2, 2))],
        );
        let compact_left = Mat::concat(
            ConcatAxis::Rows,
            vec![
                ring.input("compact-x", (1, 3 * digits)),
                ring.input("compact-y", (1, 3 * digits)),
            ],
        );
        let compact_product = compact_input
            .decompose(1u64 << parameters.base_bits(), digits)
            .mul_small_rhs(compact_left);
        let graph = DslContext::new("bench-real-provider-preimage-rhs-product")
            .output(
                "first",
                compact_product
                    .clone()
                    .slice(Some(IndexRange { start: 0.into(), end: 1.into() }), None),
            )
            .unwrap()
            .output(
                "tail",
                compact_product.slice(Some(IndexRange { start: 1.into(), end: 2.into() }), None),
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();

        let mut layouts = BTreeMap::<ConcreteMatrixType, LayoutId>::new();
        for checked in graph.scopes.values() {
            for ty in checked.wire_types.values() {
                if let Some(matrix) = ty.matrix_type() {
                    let id = layouts.len() as LayoutId + 1;
                    layouts.entry(matrix.clone()).or_insert(id);
                }
            }
        }
        let frozen_layouts = layouts
            .iter()
            .map(|(matrix, id)| GpuLayout {
                id: *id,
                columns: matrix.columns,
                rows: matrix.rows,
                ring_dimension: matrix.ring_dimension,
                representation: format!("{:?}", ConcreteWireType::Matrix(matrix.clone())),
                instance_device_stride: 0,
                owner_intervals: Vec::new(),
            })
            .collect::<Vec<_>>();
        let harness = MeasurementHarnessConfig {
            warm_up_iterations: 0,
            measured_iterations: 1,
            memory_poll_interval: Duration::ZERO,
        };
        let measurement_backend = gpu_backend_on([parameters.clone()], [device]);
        let mut provider =
            GpuNodeMeasurementBackend::new(vec![(measurement_backend, device)], harness);
        let config = GpuValidatedWarmupConfig {
            contract: GpuPlanContract {
                graph_specification_hash: [0; 32],
                backend_identity: "bench-real-provider-placeholder".into(),
                logical_to_physical_devices: vec![device as usize],
                device_budgets: vec![GpuDeviceBudget {
                    device: 0,
                    device_bytes: u64::MAX,
                    pinned_host_bytes: u64::MAX,
                    host_bytes: u64::MAX,
                }],
                shape_contract_hash: [0; 32],
                backend_revision: "bench-real-provider-placeholder".into(),
            },
            layouts: frozen_layouts,
            default_tile_widths: vec![1, 2],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "bench-real-provider-placeholder".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 2,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(1).unwrap(),
        };
        let mut warmup = mxx_runtime::gpu_warmup::warmup_gpu_from_validated_with_provider(
            &graph,
            &config,
            &mut provider,
        )
        .expect("real provider must measure preimage-RHS compact product");
        assert!(
            warmup
                .report
                .stages
                .iter()
                .all(|stage| stage.provenance != GpuProfileProvenance::ConservativeEstimate)
        );
        assert!(
            provider.warmup_dispatch_records().iter().any(|record| record.profile_domain ==
                CanonicalWarmupProfileDomain::FusedCompactProduct &&
                record.fused_operation == Some(FusedWarmupOperation::CompactProduct) &&
                record.measurement == WarmupMeasurementKind::GpuMeasured &&
                record.range.start < record.range.end),
            "compact-product warmup must enter the fused production range API"
        );
        let provider_measurement_count = provider.calibration_registry().len();
        let provider_registry = provider.calibration_registry();
        drop(provider);
        gpu_device_sync();

        let mut backend = gpu_backend_on([parameters.clone()], [device]);
        // Seed direct setup inputs through the normal pilot path; fixed graph
        // execution is installed only after all expected values are built.
        backend.select_operation([0xA6; 32]).unwrap();
        let mut matrix = |rows: usize, columns: usize, seed: u8| {
            backend
                .sample_hash(
                    &ConcreteMatrixType {
                        modulus: BigInt::from_biguint(
                            Sign::Plus,
                            parameters.modulus().as_ref().clone(),
                        ),
                        ring_dimension: parameters.ring_dimension() as usize,
                        rows,
                        columns,
                    },
                    [seed; 32],
                    b"bench-compact-input",
                )
                .unwrap()
        };
        let compact_a = matrix(1, 2, 3);
        let compact_b = matrix(2, 2, 17);
        let compact_x = matrix(1, 3 * digits, 31);
        let compact_y = matrix(1, 3 * digits, 47);
        let compact_input = backend.concat(&[&compact_a, &compact_b], ConcatAxis::Rows).unwrap();
        let compact_left = backend.concat(&[&compact_x, &compact_y], ConcatAxis::Rows).unwrap();
        let rhs = backend.gadget_decompose(&compact_input, false, Some(digits)).unwrap();
        let expected = backend.multiply_small_rhs(&compact_left, &rhs).unwrap();
        let expected_first =
            backend.slice(&expected, Some(&RuntimeIndexRange { start: 0, end: 1 }), None).unwrap();
        let expected_tail =
            backend.slice(&expected, Some(&RuntimeIndexRange { start: 1, end: 2 }), None).unwrap();
        let inputs = BTreeMap::from([
            ("compact-a".to_owned(), RuntimeValue::matrix(compact_a)),
            ("compact-b".to_owned(), RuntimeValue::matrix(compact_b)),
            ("compact-x".to_owned(), RuntimeValue::matrix(compact_x)),
            ("compact-y".to_owned(), RuntimeValue::matrix(compact_y)),
        ]);
        warmup.plan.contract = backend
            .gpu_runtime_contract(&graph, &inputs)
            .unwrap()
            .expect("execution backend must expose a GPU contract");
        warmup.plan.validate().unwrap();
        let output = execute_with_gpu_plan(
            &graph,
            &warmup.plan,
            &mut backend,
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("frozen compact product plan must execute on GPU");
        let RuntimeValue::Matrix(first) = &output.outputs["first"] else {
            panic!("compact first output is not a matrix");
        };
        let RuntimeValue::Matrix(tail) = &output.outputs["tail"] else {
            panic!("compact tail output is not a matrix");
        };
        assert_eq!(backend.matrix_to_bytes(first), backend.matrix_to_bytes(&expected_first));
        assert_eq!(backend.matrix_to_bytes(tail), backend.matrix_to_bytes(&expected_tail));
        assert_eq!(provider_registry.len(), provider_measurement_count);
    }
}
