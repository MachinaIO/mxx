//! GPU measurements for individual validated IR nodes.
//!
//! This adapter constructs representative zero-valued inputs and invokes the same
//! backend operations as the runtime. It measures operation cost; it is not a
//! second graph executor and does not define node semantics.

pub use self::harness::GpuWarmupMeasurementConfig;
use self::harness::{MemoryProbe, measure_batch_operation};
use crate::{
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
            GpuAffectedResourceEnvelope, GpuAllocationComponents, GpuAllocationEnvelope,
            GpuAllocationEvidenceKind, GpuDcrtBackend, GpuFleetMatrix, GpuFleetSignedValues,
            GpuFleetSmallMatrix, GpuFleetTrapdoor, GpuLocalProductionInput,
            GpuLocalProductionJobRequest, GpuLocalProductionSource, GpuProductionCompletion,
        },
    },
    executor::ExecutionConfig,
    gpu_calibration::{
        GpuCalibrationProfile, GpuColumnWidths, GpuDeviceMemory, gpu_matrix_multiply_scales_left,
    },
    gpu_column_policy::{
        CanonicalWarmupProfileDomain, ColumnCapability, ColumnRange, FusedWarmupOperation,
        GpuExecutionRange, GpuExecutionRouteDescriptor, GpuExecutionVariant,
        GpuFragmentClass as TypedFragmentClass, GpuTransferRoute, InputColumnRange,
        WarmupMeasurementKind, WarmupTransferKind, canonical_warmup_profile_domain,
        column_capability, effective_gpu_operation, fused_warmup_profile_domain,
        gpu_execution_range, is_resident_control_operation_for_types,
        map_output_range_to_inputs_with_output,
    },
    gpu_execution_plan::{
        FrozenGpuPlan, FrozenGpuPlanIndex, GpuDeviceBudget, GpuExecutionSiteKey, GpuLayout,
        LayoutId,
    },
    gpu_warmup::{
        GpuPreimageFootprint, GpuProfileProvenance, GpuResourceCost, GpuStageCostModel,
        GpuTimeModel, GpuValidatedWarmupConfig, GpuWarmupError, GpuWarmupReport,
        gpu_non_column_batch_wave_time,
    },
    host_control::{
        HostControlChild, HostPrimitiveValue, measure_host_primitive, measure_trapdoor_public,
        measure_typed_runtime_input,
    },
};
use mxx_ir_core::{
    ParamEnv, ValidatedGraph, encoding,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::{
    gpu_memory::{GpuMemoryRange, GpuMemoryShape},
    matrix::{PolyMatrix, PolyMatrixColumnSource, SmallPolyMatrix, gpu_dcrt_poly::GpuSmallMatrix},
    poly::{
        PolyParams,
        dcrt::gpu::{
            GpuDCRTPolyParams, GpuOutOfMemory, gpu_default_mempool_reset_high_water,
            gpu_default_mempool_usage, gpu_device_memory_usage, gpu_device_runtime_identity,
            gpu_graph_memory_snapshot, gpu_memory_info,
        },
    },
    sampler::trapdoor::gpu::{
        PreimageAllocationEvidence, PreimageCacheIdentity as NativePreimageCacheIdentity,
        PreimageCacheState,
    },
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    panic::{self, AssertUnwindSafe},
    sync::{
        Arc, Barrier, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

#[path = "gpu_measurement/harness.rs"]
mod harness;
#[path = "gpu_measurement/lifecycle_tests.rs"]
#[cfg(test)]
mod lifecycle_tests;
#[path = "gpu_measurement/multigpu_lifecycle_tests.rs"]
#[cfg(test)]
mod multigpu_lifecycle_tests;
#[path = "gpu_measurement/primitive_lifecycle_tests.rs"]
#[cfg(test)]
mod primitive_lifecycle_tests;

/// One measured operation point shared by all homogeneous fleet members.
/// `incremental_peak_bytes` is the inclusive `P - U0` observation; it is not
/// a scratch-only estimate. `retained_delta_bytes` is retained as a separate
/// diagnostic because its lifetime need not coincide with the peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMeasuredCostPoint {
    pub mean_seconds: f64,
    pub spread_seconds: f64,
    pub repetitions: usize,
    pub incremental_peak_bytes: u64,
    pub retained_delta_bytes: u64,
    /// Peak reserved bytes observed through the native graph-pool snapshot.
    /// This is kept separate from ordinary async-pool high-water bytes.
    pub graph_pool_peak_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum GpuMeasurementTransportClass {
    Resident,
    Peer { source_device: i32, destination_device: i32 },
    HostStaged { source_device: i32, destination_device: i32 },
}

/// Identity of a measured resident operation.  Physical GPU identity is
/// deliberately absent for resident work: a homogeneous fleet shares this
/// point. Transfer endpoints remain part of the transport class.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct GpuMeasuredCostKey {
    pub implementation_variant: GpuWarmupEffectiveVariant,
    pub operation_identity: [u8; 32],
    pub transport_class: GpuMeasurementTransportClass,
}

/// Canonical setup-time measured-cost table. Width is the sole interpolation
/// coordinate and is intentionally kept outside the operation identity.
#[derive(Clone, Debug, Default)]
pub struct GpuMeasuredCostCache {
    points: HashMap<(GpuMeasuredCostKey, usize), GpuMeasuredCostPoint>,
}

impl GpuMeasuredCostCache {
    pub fn exact(&self, key: &GpuMeasuredCostKey, width: usize) -> Option<GpuMeasuredCostPoint> {
        self.points.get(&(key.clone(), width)).copied()
    }

    pub fn insert(&mut self, key: GpuMeasuredCostKey, width: usize, point: GpuMeasuredCostPoint) {
        self.points.insert((key, width), point);
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Record only the timing/measurement scalars that are safe to share
    /// across equivalent resident jobs.  Resource evidence is intentionally
    /// rebuilt by the planner from the current request; a cache hit must never
    /// authorize another device or route by copying its memory envelope.
    pub fn insert_profile(
        &mut self,
        key: GpuMeasuredCostKey,
        width: usize,
        profile: &GpuWarmupProfile,
    ) -> Result<(), GpuMeasurementError> {
        if width == 0 || !profile.time_seconds.is_finite() || profile.time_seconds <= 0.0 {
            return Err(GpuMeasurementError(
                "measured-cost cache requires a positive width and finite positive time".into(),
            ));
        }
        if profile.repetitions == 0 ||
            !profile.spread_seconds.is_finite() ||
            profile.spread_seconds < 0.0
        {
            return Err(GpuMeasurementError(
                "measured-cost cache requires positive repetitions and finite spread".into(),
            ));
        }
        self.insert(
            key,
            width,
            GpuMeasuredCostPoint {
                mean_seconds: profile.time_seconds,
                spread_seconds: profile.spread_seconds,
                repetitions: profile.repetitions,
                // `workspace_bytes` is the inclusive P-U0 high-water point;
                // it is deliberately not reduced to a scratch-only residual.
                incremental_peak_bytes: profile.workspace_bytes,
                // Provider-side profiles created from a cache hit do not
                // carry this diagnostic; measured providers populate it via
                // `insert_measurement_point` below.
                retained_delta_bytes: 0,
                graph_pool_peak_bytes: 0,
            },
        );
        Ok(())
    }

    fn insert_measurement_point(
        &mut self,
        key: GpuMeasuredCostKey,
        width: usize,
        profile: &GpuWarmupProfile,
        measurement: &NodeMeasurement,
    ) -> Result<(), GpuMeasurementError> {
        if width == 0 || !profile.time_seconds.is_finite() || profile.time_seconds <= 0.0 {
            return Err(GpuMeasurementError(
                "measured-cost cache requires a positive width and finite positive time".into(),
            ));
        }
        self.insert(
            key,
            width,
            GpuMeasuredCostPoint {
                mean_seconds: profile.time_seconds,
                spread_seconds: profile.spread_seconds,
                repetitions: profile.repetitions,
                incremental_peak_bytes: measurement.workspace_bytes,
                retained_delta_bytes: measurement.retained_delta_bytes,
                graph_pool_peak_bytes: measurement.graph_pool_workspace_bytes,
            },
        );
        Ok(())
    }
}

fn measured_transport_class(
    descriptor: GpuExecutionRouteDescriptor,
) -> GpuMeasurementTransportClass {
    let endpoint =
        |device: Option<usize>| device.and_then(|device| i32::try_from(device).ok()).unwrap_or(-1);
    match descriptor.route {
        GpuTransferRoute::Resident => GpuMeasurementTransportClass::Resident,
        GpuTransferRoute::Peer => GpuMeasurementTransportClass::Peer {
            source_device: endpoint(descriptor.source_device),
            destination_device: endpoint(descriptor.destination_device),
        },
        GpuTransferRoute::HostStaging => GpuMeasurementTransportClass::HostStaged {
            source_device: endpoint(descriptor.source_device),
            destination_device: endpoint(descriptor.destination_device),
        },
    }
}

fn measured_cost_key(key: &GpuWarmupProfileKey, profile: &GpuWarmupProfile) -> GpuMeasuredCostKey {
    GpuMeasuredCostKey {
        implementation_variant: key.implementation_variant.clone(),
        operation_identity: key.operation_identity,
        transport_class: measured_transport_class(
            profile.resolved_route_descriptor.unwrap_or(key.route_descriptor),
        ),
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct NodeMeasurement {
    work_seconds: f64,
    latency_seconds: f64,
    cumulative_wave_seconds: f64,
    independent_wave_count: usize,
    measured_wave_workspace_bytes: u64,
    workspace_bytes: u64,
    retained_delta_bytes: u64,
    spread_seconds: f64,
    graph_pool_workspace_bytes: u64,
}

#[derive(Clone, Debug)]
struct MeasurementNode<'a> {
    id: mxx_ir_core::NodeId,
    kind: &'a NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
}

#[derive(Debug)]
pub struct GpuMeasurementError(String);

const GPU_OOM_ERROR_PREFIX: &str = "mxx-gpu-out-of-memory: ";

impl fmt::Display for GpuMeasurementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GpuMeasurementError {}

fn hash_int_family_spec(
    kind: &NodeKind,
    bindings: &ParamEnv,
) -> Result<Option<(usize, BigUint, Vec<u8>)>, GpuMeasurementError> {
    let NodeKind::HashIntFamily { count, modulus, tag_prefix, tag_components } = kind else {
        return Ok(None);
    };
    let count = count
        .evaluate(bindings)
        .map_err(|error| GpuMeasurementError(error.to_string()))?
        .to_usize()
        .ok_or_else(|| GpuMeasurementError("hash integer-family count is invalid".into()))?;
    let modulus = modulus
        .evaluate(bindings)
        .map_err(|error| GpuMeasurementError(error.to_string()))?
        .to_biguint()
        .ok_or_else(|| {
            GpuMeasurementError("hash integer-family modulus must be positive".into())
        })?;
    let mut tag = tag_prefix.clone();
    for component in tag_components {
        use mxx_ir_core::node::HashTagComponent;
        match component {
            HashTagComponent::Bytes(bytes) => {
                tag.push(0);
                tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                tag.extend_from_slice(bytes);
            }
            HashTagComponent::Integer(expression) => {
                let value = expression
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let (sign, bytes) = value.to_bytes_be();
                tag.push(1);
                tag.push(u8::from(sign == num_bigint::Sign::Minus));
                tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                tag.extend_from_slice(&bytes);
            }
            HashTagComponent::Decimal(expression) => {
                let decimal = expression
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?
                    .to_string();
                tag.push(2);
                tag.extend_from_slice(&(decimal.len() as u64).to_be_bytes());
                tag.extend_from_slice(decimal.as_bytes());
            }
            HashTagComponent::U64Le(expression) => {
                let value = expression
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?
                    .to_u64()
                    .ok_or_else(|| {
                        GpuMeasurementError("hash integer-family tag does not fit u64".into())
                    })?;
                tag.push(3);
                tag.extend_from_slice(&value.to_le_bytes());
            }
            HashTagComponent::Operand(_) => {
                return Err(GpuMeasurementError(
                    "hash integer-family warmup requires a static typed tag".into(),
                ));
            }
        }
    }
    Ok(Some((count, modulus, tag)))
}

fn hash_int_family_owner_bytes(
    count: usize,
    modulus: &BigUint,
) -> Result<usize, GpuMeasurementError> {
    let bits = if modulus == &BigUint::from(1u8) {
        0
    } else {
        (modulus - BigUint::from(1u8)).bits() as usize
    };
    let words = bits.div_ceil(64).max(1);
    count
        .checked_mul(words.checked_add(1).and_then(|words| words.checked_mul(8)).ok_or_else(
            || GpuMeasurementError("hash integer-family word size overflows usize".into()),
        )?)
        .ok_or_else(|| GpuMeasurementError("hash integer-family owner size overflows usize".into()))
}

impl GpuMeasurementError {
    fn out_of_memory(message: impl Into<String>) -> Self {
        Self(format!("{GPU_OOM_ERROR_PREFIX}{}", message.into()))
    }

    fn is_out_of_memory(&self) -> bool {
        self.0.starts_with(GPU_OOM_ERROR_PREFIX)
    }
}

fn warmup_node_phase_error(
    phase: &str,
    node: &MeasurementNode<'_>,
    error: impl fmt::Display,
) -> GpuMeasurementError {
    GpuMeasurementError(format!(
        "GPU warmup {phase} phase failed: node={:?} kind={:?}: {error}",
        node.id, node.kind
    ))
}

struct GpuMemoryProbe {
    device_id: i32,
}

struct GpuMemoryMeasurementBaseline {
    pool_used_high: usize,
    graph_reserved_current: u64,
    /// The CUDA default mempool counters are process-wide per physical
    /// device.  Keep this permit until the matching peak is collected so two
    /// setup providers cannot reset/read the same high-water mark at once.
    /// This is deliberately narrower than serializing GPU tests or runtime
    /// execution: only the interval that observes the shared counters is
    /// single-flight.
    permit: MutexGuard<'static, ()>,
}

const GPU_MEASUREMENT_DEVICE_LOCK_COUNT: usize = 256;

static GPU_MEASUREMENT_DEVICE_LOCKS: OnceLock<[Mutex<()>; GPU_MEASUREMENT_DEVICE_LOCK_COUNT]> =
    OnceLock::new();

fn gpu_measurement_device_permit(
    device_id: i32,
) -> Result<MutexGuard<'static, ()>, GpuMeasurementError> {
    let index = usize::try_from(device_id).map_err(|_| {
        GpuMeasurementError(format!("GPU device id {device_id} cannot be measured"))
    })?;
    let locks =
        GPU_MEASUREMENT_DEVICE_LOCKS.get_or_init(|| std::array::from_fn(|_| Mutex::new(())));
    locks
        .get(index)
        .ok_or_else(|| GpuMeasurementError(format!("GPU device id {device_id} is out of range")))?
        .lock()
        .map_err(|_| GpuMeasurementError("GPU measurement device lock poisoned".into()))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GpuMemoryMeasurementPeaks {
    ordinary_pool_bytes: u64,
    graph_pool_bytes: u64,
}

/// Select the worker that owns a physical device.
fn select_worker_index_for_physical(
    worker_count: usize,
    physical: i32,
    worker_device_id: impl Fn(usize) -> i32,
) -> Option<usize> {
    (0..worker_count).find(|&index| worker_device_id(index) == physical)
}

fn begin_gpu_memory_measurement(
    worker: &mut GpuMeasurementWorker,
) -> Result<GpuMemoryMeasurementBaseline, GpuMeasurementError> {
    let device_id = worker.device_id;
    // The ordinary CUDA pool and its high-water counter are shared by all
    // backend-owned contexts on a physical device.  Measure one owner at a
    // time so another setup provider cannot change the baseline or peak while
    // this operation is being sampled.
    let permit = gpu_measurement_device_permit(device_id)?;
    let memory = gpu_device_memory_usage(device_id).map_err(GpuMeasurementError)?;
    if memory.live_contexts == 0 {
        return Err(GpuMeasurementError(format!(
            "GPU {device_id} has no live mxx context for measurement"
        )));
    }
    // Matrix readiness precedes owner destruction, which queues frees on separate release
    // streams. Match the runtime calibration boundary before sampling the allocator baseline.
    // This fences release events only, outside the measured operation.
    worker
        .backend
        .fence_released_memory()
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
    let pool = gpu_default_mempool_usage(device_id).map_err(GpuMeasurementError)?;
    // CUDA exposes one default mempool per physical device and only permits
    // resetting its high-water mark while exactly one mxx context is live.
    // Independent runtimes may legitimately remain live while this provider
    // owns the measurement permit, so retain the current high-water baseline
    // whenever the native API cannot reset it.
    let pool_used_high = if memory.live_contexts == 1 {
        gpu_default_mempool_reset_high_water(device_id).map_err(GpuMeasurementError)?;
        let reset = gpu_default_mempool_usage(device_id).map_err(GpuMeasurementError)?;
        reset.used_high
    } else {
        pool.used_high
    };
    let graph_reserved_current = gpu_graph_memory_snapshot(device_id)
        .map_err(|error| GpuMeasurementError(error.to_string()))?
        .reserved_current;
    Ok(GpuMemoryMeasurementBaseline { pool_used_high, graph_reserved_current, permit })
}

fn finish_gpu_memory_measurement(
    device_id: i32,
    baseline: GpuMemoryMeasurementBaseline,
) -> Result<GpuMemoryMeasurementPeaks, GpuMeasurementError> {
    let GpuMemoryMeasurementBaseline { pool_used_high, graph_reserved_current, permit } = baseline;
    let result = (|| {
        let memory = gpu_device_memory_usage(device_id).map_err(GpuMeasurementError)?;
        if memory.live_contexts == 0 {
            return Err(GpuMeasurementError(format!(
                "GPU {device_id} has no live mxx context after measurement"
            )));
        }
        let high_water =
            gpu_default_mempool_usage(device_id).map_err(GpuMeasurementError)?.used_high;
        let ordinary_pool_bytes =
            u64::try_from(high_water.checked_sub(pool_used_high).ok_or_else(|| {
                GpuMeasurementError(
                    "GPU mempool high-water is below its measurement baseline".into(),
                )
            })?)
            .map_err(|_| GpuMeasurementError("GPU workspace exceeds u64".to_owned()))?;
        let graph_pool = gpu_graph_memory_snapshot(device_id)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let graph_pool_bytes = graph_pool.reserved_high.saturating_sub(graph_reserved_current);
        Ok(GpuMemoryMeasurementPeaks { ordinary_pool_bytes, graph_pool_bytes })
    })();
    drop(permit);
    result
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
/// Only input ownership, the native trapdoor alias, and the four typed real
/// operations use this path; integer and family operations remain resident.
struct PreparedHostPrimitive {
    typed_inputs:
        Option<(BTreeMap<String, RuntimeValue<GpuDcrtBackend>>, String, ConcreteWireType)>,
    trapdoor: Option<RuntimeValue<GpuDcrtBackend>>,
    scalar_inputs: Vec<HostPrimitiveValue>,
}

impl PreparedMeasurement {
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
    IntegerValues(GpuFleetSignedValues),
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
            Self::IntegerValues(value) => {
                value.wait_until_ready().expect("resident integer measurement completion")
            }
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
    routes: Vec<crate::gpu_column_policy::GpuExecutionRouteDescriptor>,
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
    // The per-source physical fragments are the production authority. An
    // outer descriptor may carry a provisional cross-device class while all
    // materialized fragments for this bounded job are resident. Derive the
    // aggregate class from the retained fragments so a zero-byte provisional
    // HostStaging envelope cannot override actual resident execution.
    aggregate.route = if aggregate
        .source_routes()
        .iter()
        .any(|route| route.route == GpuTransferRoute::HostStaging)
    {
        GpuTransferRoute::HostStaging
    } else if aggregate.source_routes().iter().any(|route| route.route == GpuTransferRoute::Peer) {
        GpuTransferRoute::Peer
    } else {
        GpuTransferRoute::Resident
    };
    aggregate.source_device = if aggregate.route == GpuTransferRoute::Resident {
        aggregate.destination_device
    } else {
        aggregate
            .source_routes()
            .iter()
            .find_map(|route| {
                (Some(route.source_owner) != aggregate.destination_device)
                    .then_some(route.source_owner)
            })
            .or_else(|| aggregate.source_routes().first().map(|route| route.source_owner))
    };
    aggregate.validate().then_some(aggregate)
}

#[derive(Clone)]
struct PendingMeasurement {
    warmup: Option<GpuWarmupOperationDescriptor>,
    scope: mxx_ir_core::FrozenGraphScopeId,
    id: mxx_ir_core::types::NodeId,
    kind: NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
    bindings: ParamEnv,
}

struct RepresentativeMeasurement {
    inputs: crate::backend::GpuEffectiveInputs,
    source_layouts: Vec<crate::backend::GpuWarmupStorageLayout>,
    fixed_metadata: Option<crate::backend::PlannedNodeBatchRequest>,
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

fn matrix_leaf_type(wire_type: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
    family_leaf_type(wire_type).matrix_type()
}

/// Return the row topology consumed by the native tensor row-sum allocation
/// query.  A shared dispatch concatenates every logical output's row groups
/// into one physical result, while an ordinary tensor row sum has only the
/// leader's `row_groups` inventory.
fn tensor_row_sum_allocation_groups(
    inputs: &crate::backend::GpuEffectiveInputs,
) -> Option<Vec<Vec<usize>>> {
    if !inputs.row_sum_groups.is_empty() {
        Some(inputs.row_sum_groups.iter().flatten().cloned().collect())
    } else if !inputs.row_groups.is_empty() {
        Some(inputs.row_groups.clone())
    } else {
        None
    }
}

pub struct ProductionGpuWarmupProvider {
    workers: Vec<GpuMeasurementWorker>,
    harness: GpuWarmupMeasurementConfig,
    /// Setup-time timing cache. It stores only timing and measurement
    /// scalars; the planner must rebuild route-specific resource evidence for
    /// each current job before admission.
    measured_costs: GpuMeasuredCostCache,
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
    /// Bounded diagnostic inventory of the most recent host fixture. These
    /// owners end with measurement and are not production residency deltas.
    last_host_setup_memory: Option<GpuWarmupMemoryObservations>,
    /// Validated child contracts for host/control nodes, keyed by their
    /// concrete scope site so collected estimator measurements can reuse the
    /// same descriptor as warmup profile requests.
    host_control_operations:
        HashMap<(mxx_ir_core::FrozenGraphScopeId, mxx_ir_core::types::NodeId), HostControlChild>,
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
    pub inputs: crate::backend::GpuEffectiveInputs,
    pub argument_types: Vec<ConcreteWireType>,
    pub output_types: Vec<ConcreteWireType>,
}

impl ProductionGpuWarmupProvider {
    fn physical_device_ids(&self) -> Vec<i32> {
        let mut seen = HashSet::new();
        self.workers
            .iter()
            .flat_map(|worker| worker.backend.physical_device_ids())
            .filter(|device| seen.insert(*device))
            .collect()
    }

    pub fn last_host_setup_memory(&self) -> Option<&GpuWarmupMemoryObservations> {
        self.last_host_setup_memory.as_ref()
    }

    fn physical_device_index(&self, physical: i32) -> Option<usize> {
        crate::gpu_execution_plan::logical_device_for_physical(
            &self.physical_device_ids(),
            physical,
        )
        .ok()
    }

    fn worker_index_for_physical_device(&self, physical: i32) -> Option<usize> {
        select_worker_index_for_physical(self.workers.len(), physical, |index| {
            self.workers[index].device_id
        })
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

    fn is_native_polynomial_primitive(kind: &NodeKind) -> bool {
        matches!(
            kind,
            NodeKind::ExtractCoefficient { .. } |
                NodeKind::ThresholdDecode { .. } |
                NodeKind::PackPolynomialCoefficients { .. } |
                NodeKind::PolynomialFromValues { .. } |
                NodeKind::PolynomialValues { .. }
        )
    }

    /// Creates a representative GPU measurement backend for validated IR nodes.
    pub fn new(backends: Vec<(GpuDcrtBackend, i32)>, harness: GpuWarmupMeasurementConfig) -> Self {
        assert!(!backends.is_empty(), "GPU measurement requires at least one backend");
        let vram_percent = backends[0].0.vram_percent();
        assert!(
            backends.iter().all(|(backend, _)| backend.vram_percent() == vram_percent),
            "all GPU measurement contexts must use the same VRAM percentage"
        );
        let mut seen_owners = HashSet::new();
        let owners = backends
            .iter()
            .flat_map(|(backend, _)| backend.physical_device_ids())
            .filter(|device| seen_owners.insert(*device))
            .collect::<Vec<_>>();
        let workers = backends
            .into_iter()
            .map(|(mut backend, device_id)| {
                backend
                    .set_measurement_owners(owners.clone())
                    .expect("validated measurement owners");
                GpuMeasurementWorker { backend, device_id, last_production_job: None }
            })
            .collect();
        Self::from_workers(workers, harness)
    }

    fn from_workers(
        workers: Vec<GpuMeasurementWorker>,
        harness: GpuWarmupMeasurementConfig,
    ) -> Self {
        Self {
            workers,
            harness,
            measured_costs: GpuMeasuredCostCache::default(),
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
            last_host_setup_memory: None,
            host_control_operations: HashMap::new(),
        }
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

    /// Return the setup cache accumulated by this provider. The cache is
    /// copied into prepared execution state before the provider is dropped.
    pub fn measured_costs(&self) -> &GpuMeasuredCostCache {
        &self.measured_costs
    }

    /// Successful public production dispatches observed during setup.
    pub fn warmup_dispatch_records(&self) -> &[GpuWarmupDispatchRecord] {
        &self.warmup_dispatch_records
    }

    fn record_warmup_dispatch(
        &mut self,
        request: &GpuWarmupProfileRequest,
        domain: CanonicalWarmupProfileDomain,
        _fused_operation: Option<FusedWarmupOperation>,
    ) {
        // The registered descriptor is the production dispatch authority. Do
        // not reconstruct the fusion kind from a parallel signature index:
        // an execution-class cache lookup may share a signature while the
        // descriptor carries the exact ordinary/fused lowering selected for
        // this site.
        let fused_operation = self
            .warmup_operations
            .get(&request.signature)
            .and_then(|pending| pending.warmup.as_ref())
            .and_then(|descriptor| descriptor.fused_operation);
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
            #[serde(skip_serializing_if = "Option::is_none")]
            row_sum_groups: Option<&'a [Vec<Vec<usize>>]>,
            arguments: Vec<ProfileType>,
            outputs: Vec<ProfileType>,
            bindings: &'a ParamEnv,
        }
        let hash = encoding::hash_canonical(&ProfileContext {
            effective_operation: &descriptor.effective_operation,
            row_groups: &descriptor.inputs.row_groups,
            row_sum_groups: (!descriptor.inputs.row_sum_groups.is_empty())
                .then_some(descriptor.inputs.row_sum_groups.as_slice()),
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
                crate::gpu_column_policy::GpuTransferRoute::Resident => GpuWarmupRoute::DeviceLocal,
                crate::gpu_column_policy::GpuTransferRoute::Peer => GpuWarmupRoute::PeerToPeer,
                crate::gpu_column_policy::GpuTransferRoute::HostStaging => {
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
            id: descriptor.id,
            kind: &representative.kind,
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
        let (public, trapdoor, sigma, _, _, bound) =
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
        let envelope = if let Some((count, modulus, _tag)) =
            hash_int_family_spec(&representative.kind, bindings)?
        {
            // HashIntFamily has no resident matrix input and its output is a
            // packed SignedWords owner. The kernel allocates the full family
            // on one device, so use the exact owner size instead of asking the
            // matrix allocation path to infer a nonexistent modulus/shape.
            GpuAllocationEnvelope {
                input_resident_bytes: 0,
                output_bytes: hash_int_family_owner_bytes(count, &modulus)?,
                auxiliary_bytes: 0,
                scratch_bytes: 0,
                transfer_bytes: 0,
                assembly_bytes: 0,
                replica_bytes: 0,
                source_device_bytes: 0,
                destination_device_bytes: 0,
                host_bytes: 0,
                pinned_host_bytes: 0,
                per_device_bytes: BTreeMap::new(),
                output_inclusive: true,
                evidence: GpuAllocationEvidenceKind::CertifiedAllocationEnvelope,
            }
        } else if let Some(input) = first_input {
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
            if domain == CanonicalWarmupProfileDomain::CrtRecompose {
                let mut allocations = Vec::new();
                for mapped in &mapped_inputs {
                    let Some(level) =
                        prepared.arguments.get(mapped.operand).and_then(Option::as_ref)
                    else {
                        return Err(GpuMeasurementError(
                            "CRT recomposition allocation query lacks a level owner".into(),
                        ));
                    };
                    let evidence = worker
                        .backend
                        .source_allocation_evidence_for_range(
                            level,
                            mapped.range.start..mapped.range.end.min(level.size().1),
                        )
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    allocations.extend(evidence.allocations);
                }
                if allocations.len() != representative.concrete_argument_types.len() {
                    return Err(GpuMeasurementError(
                        "CRT recomposition source evidence does not match level count".into(),
                    ));
                }
                components.source_allocations =
                    Some(mxx_primitives::gpu_memory::GpuSourceAllocationEvidence { allocations });
            }
            if domain == CanonicalWarmupProfileDomain::FusedTensorRowSum {
                let groups =
                    tensor_row_sum_allocation_groups(&representative.inputs).ok_or_else(|| {
                        GpuMeasurementError(
                            "tensor row-sum allocation query lacks row-group topology".into(),
                        )
                    })?;
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
                // The production observation normally supplies these
                // materialization owners through merge_production_memory.
                // Keep the same piece classifier available to query-only
                // callers, but do not add the owners twice when the measured
                // production job already reported them.
                if domain == CanonicalWarmupProfileDomain::FusedCompactProduct ||
                    worker.last_production_job.is_none()
                {
                    let temporary = worker
                        .backend
                        .compact_piece_materialization_components(
                            rhs,
                            range.clone(),
                            worker.device_id,
                        )
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    components.replica_bytes =
                        components.replica_bytes.checked_add(temporary.replica_bytes).ok_or_else(
                            || GpuMeasurementError("compact temporary owners overflow".into()),
                        )?;
                    for (device, bytes) in temporary.per_device_bytes {
                        let entry = components.per_device_bytes.entry(device).or_insert(0);
                        *entry = entry.checked_add(bytes).ok_or_else(|| {
                            GpuMeasurementError("compact per-device owners overflow".into())
                        })?;
                    }
                }
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
                        .chain(representative.concrete_argument_types.iter())
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
        // Operation-specific native queries may expose temporary owners that
        // are not retained inputs (for example compact piece imports).
        // Preserve those physical owners in the same map used by production
        // observations; transport resources are merged separately below.
        for (owner, bytes) in &envelope.per_device_bytes {
            add_device_bytes(*owner, *bytes)?;
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
        // Host-staged transfers retain host/pinned bytes separately and are
        // supplied by the production observation merge.  Do not add route
        // staging here as well: production materialization already reports
        // each source/destination owner exactly once. If no resident input was
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
    /// warmup may request several nonlinear `b`
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

        let memory = match gpu_device_memory_usage(physical) {
            Ok(memory) => memory,
            Err(error) => {
                return GpuWarmupProfileError::Measurement(format!(
                    "OOM context usability validation failed on GPU {physical}: {error}"
                ));
            }
        };
        if memory.live_contexts == 0 {
            return GpuWarmupProfileError::Measurement(format!(
                "OOM context usability validation found no live context on GPU {physical}"
            ));
        }

        // The failed allocation may have left the async pool's high-water
        // marker and temporary blocks live. CUDA only allows resetting that
        // marker with one live context; with independent runtimes, preserve
        // the shared marker and let the next measurement use a fresh
        // high-water baseline instead of serializing or rejecting the other
        // owner.
        if memory.live_contexts == 1 {
            if let Err(error) = gpu_default_mempool_reset_high_water(physical) {
                return GpuWarmupProfileError::Measurement(format!(
                    "OOM allocator cleanup failed on GPU {physical}: {error}"
                ));
            }
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
        if validated.live_contexts == 0 || validated.context_generation != memory.context_generation
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
                crate::backend::GpuWarmupFragmentClass::Whole => TypedFragmentClass::Full,
                crate::backend::GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
                crate::backend::GpuWarmupFragmentClass::Mapped => TypedFragmentClass::Mapped,
                crate::backend::GpuWarmupFragmentClass::Fragmented => {
                    TypedFragmentClass::CompactFragment
                }
                crate::backend::GpuWarmupFragmentClass::SingleDevice => TypedFragmentClass::Full,
            };
            let expected_route = resolver.resolve_for(
                request.route_descriptor.source_range,
                request.route_descriptor.destination_range,
                fragment,
                request.route_descriptor.source_compact,
                request.route_descriptor.destination_compact,
            );
            if expected_route != request.route_descriptor &&
                !crate::gpu_warmup::route_response_matches_resolver(
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
        // Integer/bool control operations that are lowered into the resident
        // GPU region do not have a matrix-native request. Measure their real
        // integer owner allocation directly. Real-valued host operations and
        // explicit host/device boundaries use the paths below.
        if Self::resident_control_kind_for_types(
            &descriptor.kind,
            &descriptor.concrete_argument_types,
            &descriptor.concrete_output_types,
        ) && request.timing_scope != GpuWarmupTimingScope::Transfer
        {
            let node = MeasurementNode {
                id: descriptor.id,
                kind: &descriptor.kind,
                concrete_argument_types: descriptor.concrete_argument_types.clone(),
                concrete_output_types: descriptor.concrete_output_types.clone(),
            };
            let (measurement, memory) = Self::measure_resident_control_repeated(
                &mut self.workers[selected_device],
                selected_device,
                &node,
                &self.harness,
            )
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
            let mut profile = GpuWarmupProfile::measured_with_observation(
                measurement.cumulative_wave_seconds,
                measurement.workspace_bytes,
                profile_domain.measurement_kind(),
                memory,
                GpuWarmupResidencyDelta::default(),
                self.harness.measured_iterations,
                measurement.spread_seconds,
                GpuWarmupProvenance::ProductionEquivalent,
                profile_key.cache_state,
                profile_key.timing_scope,
            )?;
            profile.resolved_route_descriptor = Some(request.route_descriptor);
            self.measured_costs
                .insert_measurement_point(
                    measured_cost_key(&profile_key, &profile),
                    request.coordinate(),
                    &profile,
                    &measurement,
                )
                .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
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
                .map_err(|error| {
                    GpuWarmupProfileError::Measurement(format!(
                        "GPU warmup scalar operation selection failed: signature={:?} device={} tile_width=1: {error}",
                        request.signature, selected_device
                    ))
                })?;
            let node = MeasurementNode {
                id: descriptor.id,
                kind: &descriptor.kind,
                concrete_argument_types: descriptor.concrete_argument_types.clone(),
                concrete_output_types: descriptor.concrete_output_types.clone(),
            };
            let (elapsed, spread, repetitions, setup_memory) = Self::measure_scalar_host_repeated(
                &mut self.workers[selected_device],
                selected_device,
                &node,
                &descriptor.bindings,
                &self.harness,
            )
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
            self.last_host_setup_memory = Some(setup_memory);
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
            self.measured_costs
                .insert_profile(
                    measured_cost_key(&profile_key, &profile),
                    request.coordinate(),
                    &profile,
                )
                .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
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
            Some(crate::backend::PlannedNodeBatchRequest::for_lowered_operation(
                crate::gpu_execution_plan::GpuExecutionSiteKey {
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
                        crate::backend::PlannedLayoutMetadata::for_type(ty, Some(port as u32))
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
                            GpuMeasurementError(format!(
                                "GPU warmup operation selection failed: signature={:?} device={} tile_width={} range={:?}: {message}",
                                request.signature, device, request.tile_width, request.range
                            ))
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
                        descriptor.id,
                        &descriptor.bindings,
                        &representative,
                    )
                } else {
                    match fused_operation {
                        Some(fused) => Self::measure_fused_representative(
                            &mut self.workers[device],
                            &self.harness,
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
                id: descriptor.id,
                kind: &representative.kind,
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
        } else if Self::is_native_polynomial_primitive(&descriptor.kind) {
            // Native polynomial primitives retain the matrix representation
            // even when their exposed output is an integer family.
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
            measurement.spread_seconds,
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
            self.preimage_footprints.get(&(profile_key.clone(), request.tile_width)).cloned()
        {
            profile.preimage_footprint = Some(footprint);
        }
        profile.resolved_cache_identity = resolved_cache_identity;
        self.measured_costs
            .insert_measurement_point(
                measured_cost_key(&profile_key, &profile),
                request.coordinate(),
                &profile,
                &measurement,
            )
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
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
        let index = FrozenGpuPlanIndex::build(plan)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let node = index
            .node_choice(plan, site)
            .ok_or_else(|| GpuMeasurementError(format!("missing fixed plan site {site:?}")))?;
        let layout_id = *node
            .output_layouts
            .first()
            .ok_or_else(|| GpuMeasurementError("fixed plan site has no output layout".into()))?;
        let layout = index
            .layout(plan, layout_id)
            .ok_or_else(|| GpuMeasurementError(format!("missing fixed plan layout {layout_id}")))?;
        let instances = node
            .loop_site
            .map(|key| {
                index
                    .loop_choice(plan, key)
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
                        let layout = index.layout(plan, *id).ok_or_else(|| {
                            GpuMeasurementError("missing output port layout".into())
                        })?;
                        layout
                            .schedule(&node.columns_per_job, instance)
                            .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        crate::gpu_warmup::gpu_multi_output_batch_wave_time(&schedules, model)
            .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    #[cfg(test)]
    fn column_separable(kind: &NodeKind) -> bool {
        crate::gpu_calibration::gpu_operation_is_column_separable(kind)
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

    fn representative_at(
        request: &PendingMeasurement,
        global_column_start: usize,
        columns: usize,
    ) -> RepresentativeMeasurement {
        let global_column_end =
            global_column_start.checked_add(columns).expect("representative column range overflow");
        let range_capable = !matches!(
            column_capability(&request.kind, &request.concrete_argument_types),
            ColumnCapability::HostOrControl |
                ColumnCapability::SingleDevice |
                ColumnCapability::Unsupported
        );
        RepresentativeMeasurement {
            source_layouts: Vec::new(),
            fixed_metadata: None,
            retry_cap: None,
            inputs: Default::default(),
            kind: request.kind.clone(),
            concrete_argument_types: request.concrete_argument_types.clone(),
            concrete_output_types: request.concrete_output_types.clone(),
            fixed_arguments: Self::fixed_arguments(&request.kind, &request.concrete_argument_types),
            output_range: (range_capable ||
                matches!(request.kind, NodeKind::GadgetTrapdoor { .. }))
            .then_some(IndexRange { start: global_column_start, end: global_column_end }),
        }
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
        // Query every compact owner as one allocation-free aggregate before
        // constructing any of them, preserving simultaneous setup peaks.
        let compact_types =
            node.concrete_argument_types
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    fixed_phase
                        .is_none_or(|(fixed_arguments, fixed)| fixed_arguments[*index] == fixed) &&
                        !(matches!(node.kind, NodeKind::PreimageSample { .. }) &&
                            node.concrete_argument_types.len() >= 3 &&
                            *index < 2)
                })
                .map(|(_, wire_type)| wire_type)
                .collect::<Vec<_>>();
        Self::preflight_compact_setup(backend, &compact_types)?;
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
                compact @ (ConcreteWireType::SmallMatrix { .. } |
                ConcreteWireType::Preimage { .. }) => {
                    let value =
                        match Self::representative_runtime_value(backend, compact, bindings)? {
                            RuntimeValue::SmallMatrix(value) | RuntimeValue::Preimage(value) => {
                                value
                            }
                            _ => unreachable!("compact representative preserves its semantic kind"),
                        };
                    arguments.push(None);
                    small_arguments.push(Some(value));
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

    fn preflight_compact_setup(
        backend: &GpuDcrtBackend,
        wire_types: &[&ConcreteWireType],
    ) -> Result<(), GpuMeasurementError> {
        // Host and pinned-host staging are executable transient allocations,
        // not planner dimensions. Keep this preflight limited to the device
        // owner budget; the production constructor performs the host
        // allocation and returns an ordinary allocation error if it fails.
        let mut totals = BTreeMap::<i32, u128>::new();
        for wire_type in wire_types {
            let compact = match family_leaf_type(wire_type) {
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
                ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                    (matrix, max_coefficient_bound)
                }
                _ => continue,
            };
            // `parameters` is the same first registered placement consumed by
            // the compact constructor. Never admit against another device.
            let params = backend
                .parameters(compact.0)
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
            let owner = *params.device_ids().first().ok_or_else(|| {
                GpuMeasurementError("compact setup has no selected device owner".into())
            })?;
            let bound = compact.1.to_biguint().ok_or_else(|| {
                GpuMeasurementError("compact representative bound must be nonnegative".into())
            })?;
            let evidence = GpuSmallMatrix::canonical_import_allocation_evidence(
                params,
                compact.0.rows,
                compact.0.columns,
                &bound,
            )
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
            let entry = totals.entry(owner).or_default();
            *entry = entry
                .checked_add(evidence.device.total_bytes as u128)
                .ok_or_else(|| GpuMeasurementError("compact setup device bytes overflow".into()))?;
        }
        if totals.is_empty() {
            return Ok(());
        }
        let physical = backend.physical_device_ids();
        let configured = backend.configured_plan_budgets();
        for (owner, device_bytes) in totals {
            let index = physical.iter().position(|device| *device == owner).ok_or_else(|| {
                GpuMeasurementError(format!("compact setup owner {owner} is not in this fleet"))
            })?;
            let usage = gpu_device_memory_usage(owner).map_err(GpuMeasurementError)?;
            let device_budget = configured
                .and_then(|budgets| budgets.get(index).map(|budget| budget.device_bytes as u128))
                .unwrap_or_else(|| {
                    (usage.total.max(0) as u128).saturating_mul(backend.vram_percent() as u128) /
                        100
                });
            if (usage.resident.max(0) as u128).saturating_add(device_bytes) > device_budget {
                return Err(GpuMeasurementError(format!(
                    "compact setup device allocation exceeds selected owner budget: device={owner}, required={device_bytes}, budget={device_budget}"
                )));
            }
        }
        Ok(())
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
            id: descriptor.id,
            kind: &representative.kind,
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
            if matrix_leaf_type(ty).is_none() {
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
        harness: &GpuWarmupMeasurementConfig,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
        barrier: Option<&Barrier>,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        if let Some(barrier) = barrier {
            barrier.wait();
        }
        let node = MeasurementNode {
            id,
            kind: &representative.kind,
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let prepared = Self::prepare(&mut worker.backend, &node, bindings, None)
            .map_err(|error| warmup_node_phase_error("representative preparation", &node, error))?;
        let prepared = Self::place_prepared_sources(worker, representative, prepared)
            .map_err(|error| warmup_node_phase_error("source placement", &node, error))?;
        let baseline = begin_gpu_memory_measurement(worker)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(harness, &probe, 1, |representative_batch| {
            if operation_error.is_some() {
                return GpuMeasurementOutputs(Vec::new());
            }
            match Self::execute_local_production_measurement(
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
                Ok(output) => output,
                Err(error) => {
                    operation_error = Some(error);
                    GpuMeasurementOutputs(Vec::new())
                }
            }
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        let mut measurement = measured.measurement;
        let peaks = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        measurement.workspace_bytes = peaks.ordinary_pool_bytes;
        measurement.measured_wave_workspace_bytes = measurement.workspace_bytes;
        measurement.graph_pool_workspace_bytes = peaks.graph_pool_bytes;
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
        let peaks = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        Ok(NodeMeasurement {
            work_seconds: seconds,
            latency_seconds: seconds,
            cumulative_wave_seconds: seconds,
            independent_wave_count: 1,
            measured_wave_workspace_bytes: peaks.ordinary_pool_bytes,
            workspace_bytes: peaks.ordinary_pool_bytes,
            retained_delta_bytes: 0,
            spread_seconds: 0.0,
            graph_pool_workspace_bytes: peaks.graph_pool_bytes,
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
                            if fused == FusedWarmupOperation::TensorRowSum &&
                                !representative.inputs.row_sum_groups.is_empty()
                            {
                                crate::backend::FusedBatchRequest::TensorRowSums {
                                    metadata,
                                    source: matrix(0)?,
                                    right: matrix(1)?,
                                    rows: representative.inputs.row_sum_groups.clone(),
                                }
                            } else {
                                crate::backend::FusedBatchRequest::RowSum {
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
                        }
                        FusedWarmupOperation::RowBlockAdd => {
                            let mut blocks = blocks();
                            let right = blocks
                                .pop()
                                .ok_or_else(|| GpuMeasurementError("missing fused RHS".into()))?;
                            crate::backend::FusedBatchRequest::Add { metadata, blocks, right }
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
                            crate::backend::FusedBatchRequest::Decompose {
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
                            crate::backend::FusedBatchRequest::SmallProduct {
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
                        crate::backend::FusedBatchOutput::Matrices(values) => {
                            outputs.extend(values.into_iter().map(GpuMeasurementOutput::matrix))
                        }
                        crate::backend::FusedBatchOutput::Small(value) => {
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
    ) -> Result<GpuMeasurementOutputs, GpuMeasurementError> {
        if let NodeKind::HashIntFamily { count, modulus, tag_prefix, tag_components } = node.kind {
            if fused.is_some() || binding_port.is_some() || transfer_only {
                return Err(GpuMeasurementError(
                    "hash integer-family warmup requires an ordinary native launch".into(),
                ));
            }
            let count = count
                .evaluate(bindings)
                .map_err(|error| GpuMeasurementError(error.to_string()))?
                .to_usize()
                .ok_or_else(|| {
                    GpuMeasurementError("hash integer-family count is invalid".into())
                })?;
            let modulus = modulus
                .evaluate(bindings)
                .map_err(|error| GpuMeasurementError(error.to_string()))?
                .to_biguint()
                .ok_or_else(|| {
                    GpuMeasurementError("hash integer-family modulus must be positive".into())
                })?;
            if representative
                .output_range
                .as_ref()
                .is_some_and(|range| range.start != 0 || (range.end != 1 && range.end != count))
            {
                return Err(GpuMeasurementError(
                    "hash integer-family warmup cannot measure a partial family".into(),
                ));
            }
            let mut tag = tag_prefix.clone();
            for component in tag_components {
                use mxx_ir_core::node::HashTagComponent;
                match component {
                    HashTagComponent::Bytes(bytes) => {
                        tag.push(0);
                        tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                        tag.extend_from_slice(bytes);
                    }
                    HashTagComponent::Integer(expression) => {
                        let value = expression
                            .evaluate(bindings)
                            .map_err(|error| GpuMeasurementError(error.to_string()))?;
                        let (sign, bytes) = value.to_bytes_be();
                        tag.push(1);
                        tag.push(u8::from(sign == num_bigint::Sign::Minus));
                        tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                        tag.extend_from_slice(&bytes);
                    }
                    HashTagComponent::Decimal(expression) => {
                        let decimal = expression
                            .evaluate(bindings)
                            .map_err(|error| GpuMeasurementError(error.to_string()))?
                            .to_string();
                        tag.push(2);
                        tag.extend_from_slice(&(decimal.len() as u64).to_be_bytes());
                        tag.extend_from_slice(decimal.as_bytes());
                    }
                    HashTagComponent::U64Le(expression) => {
                        let value = expression
                            .evaluate(bindings)
                            .map_err(|error| GpuMeasurementError(error.to_string()))?
                            .to_u64()
                            .ok_or_else(|| {
                                GpuMeasurementError(
                                    "hash integer-family tag does not fit u64".into(),
                                )
                            })?;
                        tag.push(3);
                        tag.extend_from_slice(&value.to_le_bytes());
                    }
                    HashTagComponent::Operand(_) => {
                        return Err(GpuMeasurementError(
                            "hash integer-family warmup requires a static typed tag".into(),
                        ));
                    }
                }
            }
            worker.last_production_job = None;
            let outputs = (0..batch_size)
                .map(|_| {
                    worker
                        .backend
                        .sample_hash_int_values_on_device(
                            worker.device_id,
                            count,
                            &modulus,
                            [0x53; 32],
                            &tag,
                        )
                        .map(GpuMeasurementOutput::IntegerValues)
                        .map_err(|error| GpuMeasurementError(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(GpuMeasurementOutputs(outputs));
        }
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
        } else if Self::is_native_polynomial_primitive(node.kind) ||
            column_capability(node.kind, &representative.concrete_argument_types) ==
                ColumnCapability::SingleDevice
        {
            // Scalar polynomial primitives consume a complete native value;
            // their matrix input is indivisible even when the result is an
            // integer family rather than a matrix column range.
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
        let source_placement = sources
            .iter()
            .map(|source| match source {
                GpuLocalProductionSource::Matrix(value) => format!(
                    "matrix {}x{} {:?}",
                    value.size().0,
                    value.size().1,
                    value
                        .shards()
                        .iter()
                        .map(|shard| (
                            shard.device_id,
                            shard.global_column_start,
                            shard.value.col_size()
                        ))
                        .collect::<Vec<_>>()
                ),
                GpuLocalProductionSource::Compact(value) => format!(
                    "compact {}x{} {:?}",
                    value.size().0,
                    value.size().1,
                    value
                        .shards()
                        .iter()
                        .map(|shard| (
                            shard.device_id,
                            shard.global_column_start,
                            shard.value.columns()
                        ))
                        .collect::<Vec<_>>()
                ),
            })
            .collect::<Vec<_>>();
        let operation_context = representative.fixed_metadata.as_ref().map(|metadata| {
            (
                metadata.site,
                metadata.shape_class,
                metadata.instance_class,
                metadata.operation_identity,
                metadata.implementation_variant.as_str(),
            )
        });
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
                    GpuMeasurementError(format!(
                        "GPU warmup production failed: node={:?} kind={:?} operation={operation_context:?} destination={} output={:?} inputs={:?} sources={source_placement:?}: {message}",
                        node.id, node.kind, worker.device_id, execution.output, execution.inputs,
                    ))
                }
            })?;
        worker.last_production_job =
            Some(ProductionJobObservation { routes: result.routes, resources: result.resources });
        Ok(result.value)
    }

    fn measure_fused_representative(
        worker: &mut GpuMeasurementWorker,
        harness: &GpuWarmupMeasurementConfig,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
        fused: FusedWarmupOperation,
        binding_port: usize,
        session_prepared: Option<&PreparedMeasurement>,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        let node = MeasurementNode {
            id,
            kind: &representative.kind,
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let prepared = match session_prepared {
            Some(prepared) => Ok(prepared.clone()),
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
                )
            }
            None => Self::prepare(&mut worker.backend, &node, bindings, None),
        }
        .map_err(|error| warmup_node_phase_error("fused preparation", &node, error))?;
        let prepared = Self::place_prepared_sources(worker, representative, prepared)
            .map_err(|error| warmup_node_phase_error("source placement", &node, error))?;
        let baseline = begin_gpu_memory_measurement(worker)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(harness, &probe, 1, |batch_size| {
            if operation_error.is_some() {
                return GpuMeasurementOutputs(Vec::new());
            }
            match Self::execute_local_production_measurement(
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
                Ok(output) => output,
                Err(error) => {
                    operation_error = Some(error);
                    GpuMeasurementOutputs(Vec::new())
                }
            }
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        let mut measurement = measured.measurement;
        let peaks = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        measurement.workspace_bytes = peaks.ordinary_pool_bytes;
        measurement.measured_wave_workspace_bytes = measurement.workspace_bytes;
        measurement.graph_pool_workspace_bytes = peaks.graph_pool_bytes;
        Ok(measurement)
    }

    /// Measure a transfer-only production point.  This deliberately enters
    /// the same fleet materializer as a normal local job and returns before
    /// the kernel callback, so D2H/H2D, peer copies, and host staging are
    /// included in the point while kernel/assembly work is not.
    fn measure_transfer_representative(
        worker: &mut GpuMeasurementWorker,
        harness: &GpuWarmupMeasurementConfig,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        representative: &RepresentativeMeasurement,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        let node = MeasurementNode {
            id,
            kind: &representative.kind,
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let prepared = Self::prepare(&mut worker.backend, &node, bindings, None)
            .map_err(|error| warmup_node_phase_error("transfer preparation", &node, error))?;
        let prepared = Self::place_prepared_sources(worker, representative, prepared)
            .map_err(|error| warmup_node_phase_error("transfer source placement", &node, error))?;
        let baseline = begin_gpu_memory_measurement(worker)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(harness, &probe, 1, |batch_size| {
            if operation_error.is_some() {
                return GpuMeasurementOutputs(Vec::new());
            }
            match Self::execute_local_production_measurement(
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
                Ok(output) => output,
                Err(error) => {
                    operation_error = Some(error);
                    GpuMeasurementOutputs(Vec::new())
                }
            }
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        let mut measurement = measured.measurement;
        let peaks = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        measurement.workspace_bytes = peaks.ordinary_pool_bytes;
        measurement.measured_wave_workspace_bytes = measurement.workspace_bytes;
        measurement.graph_pool_workspace_bytes = peaks.graph_pool_bytes;
        Ok(measurement)
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
            ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
            ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                let params = backend
                    .parameters(matrix)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let bound = max_coefficient_bound.to_biguint().ok_or_else(|| {
                    GpuMeasurementError("compact representative bound must be nonnegative".into())
                })?;
                let payload_len = GpuSmallMatrix::canonical_import_allocation_evidence(
                    params,
                    matrix.rows,
                    matrix.columns,
                    &bound,
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?
                .pageable_host_bytes;
                let value = GpuSmallMatrix::from_canonical_coefficients(
                    params,
                    matrix.rows,
                    matrix.columns,
                    bound,
                    &vec![0u8; payload_len],
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let value = GpuFleetSmallMatrix::from(value);
                Ok(if matches!(wire_type, ConcreteWireType::Preimage { .. }) {
                    RuntimeValue::preimage(value)
                } else {
                    RuntimeValue::small_matrix(value)
                })
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
            NodeKind::IntToReal => {
                vec![HostPrimitiveValue::Int(BigInt::from(7))]
            }
            NodeKind::RealBinary(_) => {
                vec![HostPrimitiveValue::Real(1.25), HostPrimitiveValue::Real(0.75)]
            }
            NodeKind::RealSqrt => vec![HostPrimitiveValue::Real(1.25)],
            NodeKind::ConstantReal(_) => Vec::new(),
            _ => {
                return Err(GpuMeasurementError(
                    "non-real operation reached typed host measurement".into(),
                ))
            }
        };
        let elapsed = measure_host_primitive(node.id, node.kind, bindings, &inputs, batch_size);
        elapsed.map(|_| ()).map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn prepare_host_primitive_inner(
        backend: &mut GpuDcrtBackend,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<PreparedHostPrimitive, GpuMeasurementError> {
        let mut compact_types = Vec::new();
        match node.kind {
            NodeKind::Input { .. } => {
                if let Some(output) = node.concrete_output_types.first() {
                    compact_types.push(output);
                }
            }
            NodeKind::TrapdoorPublic => {
                if let Some(input) = node.concrete_argument_types.first() {
                    compact_types.push(input);
                }
            }
            _ => {}
        }
        Self::preflight_compact_setup(backend, &compact_types)?;
        let mut prepared =
            PreparedHostPrimitive { typed_inputs: None, trapdoor: None, scalar_inputs: Vec::new() };
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
            NodeKind::IntToReal => {
                prepared.scalar_inputs = vec![HostPrimitiveValue::Int(BigInt::from(7))];
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
        measure_host_primitive(node.id, node.kind, bindings, &prepared.scalar_inputs, 1)
            .map(|_| ())
            .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn scalar_host_kind(kind: &NodeKind) -> bool {
        matches!(
            kind,
            NodeKind::Input { .. } |
                NodeKind::TrapdoorPublic |
                NodeKind::ConstantReal(_) |
                NodeKind::IntToReal |
                NodeKind::RealBinary(_) |
                NodeKind::RealSqrt
        )
    }

    fn resident_control_kind_for_types(
        kind: &NodeKind,
        arguments: &[ConcreteWireType],
        outputs: &[ConcreteWireType],
    ) -> bool {
        is_resident_control_operation_for_types(kind, arguments, outputs) &&
            arguments.iter().chain(outputs).all(|ty| {
                matrix_leaf_type(ty).is_none() && Self::resident_control_owner_count(ty).is_some()
            })
    }

    fn resident_control_owner_count(wire_type: &ConcreteWireType) -> Option<usize> {
        match wire_type {
            ConcreteWireType::ConstantInt |
            ConcreteWireType::Int |
            ConcreteWireType::ConstantBool |
            ConcreteWireType::Bool => Some(1),
            ConcreteWireType::IndexedFamily { element, count } => {
                Self::resident_control_owner_count(element)?.checked_mul(*count)
            }
            _ => None,
        }
    }

    /// Measure the actual resident integer owner allocation used by migrated
    /// control operations. This keeps the allocator/mempool interval under
    /// the same per-device single-flight guard as matrix GPU measurements.
    fn measure_resident_control_repeated(
        worker: &mut GpuMeasurementWorker,
        logical_device: usize,
        node: &MeasurementNode<'_>,
        harness: &GpuWarmupMeasurementConfig,
    ) -> Result<(NodeMeasurement, GpuWarmupMemoryObservations), GpuMeasurementError> {
        let output_owner_count = node
            .concrete_output_types
            .iter()
            .filter_map(Self::resident_control_owner_count)
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or_else(|| GpuMeasurementError("resident control owner bytes overflow".into()))?;
        let owner_count = if output_owner_count > 0 {
            output_owner_count
        } else {
            node.concrete_argument_types
                .iter()
                .filter_map(Self::resident_control_owner_count)
                .try_fold(0usize, |total, count| total.checked_add(count))
                .ok_or_else(|| {
                    GpuMeasurementError("resident control owner bytes overflow".into())
                })?
        };
        if owner_count == 0 {
            return Err(GpuMeasurementError(
                "resident control measurement has no integer/bool owner".into(),
            ));
        }
        let host_values = vec![BigInt::from(7); owner_count];
        let baseline = begin_gpu_memory_measurement(worker)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(harness, &probe, 1, |batch_size| {
            if operation_error.is_some() {
                return Vec::new();
            }
            let mut owners = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let owner = match worker
                    .backend
                    .integer_values_from_host_on_device(worker.device_id, &host_values)
                {
                    Ok(owner) => owner,
                    Err(error) => {
                        operation_error = Some(GpuMeasurementError(error.to_string()));
                        return Vec::new();
                    }
                };
                if let Err(error) = owner.wait_until_ready() {
                    operation_error = Some(GpuMeasurementError(error.to_string()));
                    return Vec::new();
                }
                owners.push(owner);
            }
            owners
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            let _ = finish_gpu_memory_measurement(worker.device_id, baseline);
            return Err(error);
        }
        let peaks = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        let mut measurement = measured.measurement;
        measurement.workspace_bytes = peaks.ordinary_pool_bytes;
        measurement.measured_wave_workspace_bytes = measurement.workspace_bytes;
        measurement.graph_pool_workspace_bytes = peaks.graph_pool_bytes;
        let identity =
            gpu_device_runtime_identity(worker.device_id).map_err(GpuMeasurementError)?;
        let owner = GpuWarmupDeviceIdentity::new(
            logical_device,
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
                identity.native_kernel_revision, identity.driver_version, identity.runtime_version
            ),
        )
        .with_context_generation(identity.context_generation);
        let bytes = u64::try_from(owner_count)
            .ok()
            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
            .ok_or_else(|| GpuMeasurementError("resident integer owner bytes overflow".into()))?;
        let memory = GpuWarmupMemoryObservations {
            affected_devices: BTreeMap::from([(owner, bytes)]),
            host_bytes: 0,
            pinned_host_bytes: 0,
            evidence: MemoryEvidenceKind::ExactQuery,
        };
        Ok((measurement, memory))
    }

    fn measure_scalar_host_repeated(
        worker: &mut GpuMeasurementWorker,
        logical_device: usize,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        harness: &GpuWarmupMeasurementConfig,
    ) -> Result<(f64, f64, usize, GpuWarmupMemoryObservations), GpuMeasurementError> {
        if harness.measured_iterations == 0 {
            return Err(GpuMeasurementError(
                "host warmup measurement requires at least one measured repetition".into(),
            ));
        }
        let prepared = Self::prepare_host_primitive_inner(&mut worker.backend, node, bindings)?;
        let setup_memory = Self::host_setup_memory(&worker.backend, &prepared, logical_device)?;
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
        Ok((mean, spread, samples.len(), setup_memory))
    }

    fn host_setup_memory(
        backend: &GpuDcrtBackend,
        prepared: &PreparedHostPrimitive,
        logical_device: usize,
    ) -> Result<GpuWarmupMemoryObservations, GpuMeasurementError> {
        fn visit(
            backend: &GpuDcrtBackend,
            value: &RuntimeValue<GpuDcrtBackend>,
            logical_device: usize,
            matrices: &mut HashSet<usize>,
            compacts: &mut HashSet<usize>,
            secrets: &mut HashSet<usize>,
            observations: &mut GpuWarmupMemoryObservations,
        ) -> Result<(), GpuMeasurementError> {
            match value {
                RuntimeValue::Matrix(matrix) => {
                    if !matrices.insert(Arc::as_ptr(matrix) as usize) {
                        return Ok(());
                    }
                    for (device, bytes) in backend
                        .resident_allocation_bytes_by_device(matrix)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?
                    {
                        let identity =
                            gpu_device_runtime_identity(device).map_err(GpuMeasurementError)?;
                        let owner = GpuWarmupDeviceIdentity::new(
                            logical_device,
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
                                identity.runtime_version
                            ),
                        )
                        .with_context_generation(identity.context_generation);
                        let entry = observations.affected_devices.entry(owner).or_default();
                        *entry = entry.checked_add(bytes as u64).ok_or_else(|| {
                            GpuMeasurementError("host setup device bytes overflow".into())
                        })?;
                    }
                }
                RuntimeValue::Trapdoor { secret, public, .. } => {
                    // Public A and the retained secret owners are real setup
                    // residency.  Deduplicate Arc-backed values because a
                    // trapdoor can be referenced by several host containers.
                    visit(
                        backend,
                        &RuntimeValue::Matrix(public.clone()),
                        logical_device,
                        matrices,
                        compacts,
                        secrets,
                        observations,
                    )?;
                    if let Some(secret) = secret {
                        if secrets.insert(Arc::as_ptr(secret) as usize) {
                            for (device, bytes) in secret
                                .retained_allocation_bytes_by_device()
                                .map_err(|error| GpuMeasurementError(error.to_string()))?
                            {
                                let identity = gpu_device_runtime_identity(device)
                                    .map_err(GpuMeasurementError)?;
                                let owner = GpuWarmupDeviceIdentity::new(
                                    logical_device,
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
                                        identity.runtime_version
                                    ),
                                )
                                .with_context_generation(identity.context_generation);
                                let entry = observations.affected_devices.entry(owner).or_default();
                                *entry = entry.checked_add(bytes as u64).ok_or_else(|| {
                                    GpuMeasurementError("host setup trapdoor bytes overflow".into())
                                })?;
                            }
                        }
                    }
                }
                RuntimeValue::SmallMatrix(compact) | RuntimeValue::Preimage(compact) => {
                    if !compacts.insert(Arc::as_ptr(compact) as usize) {
                        return Ok(());
                    }
                    for shard in compact.shards() {
                        let allocation =
                            shard.value.allocation_bytes().map_err(GpuMeasurementError)?;
                        let identity = gpu_device_runtime_identity(shard.device_id)
                            .map_err(GpuMeasurementError)?;
                        let owner = GpuWarmupDeviceIdentity::new(
                            logical_device,
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
                                identity.runtime_version
                            ),
                        )
                        .with_context_generation(identity.context_generation);
                        let entry = observations.affected_devices.entry(owner).or_default();
                        *entry =
                            entry.checked_add(allocation.total_bytes as u64).ok_or_else(|| {
                                GpuMeasurementError("host setup compact bytes overflow".into())
                            })?;
                        // Canonical import buffers are pageable/pinned
                        // temporaries. They are dropped after setup and must
                        // not become persistent residency deltas.
                    }
                }
                RuntimeValue::IndexedFamily(values) => {
                    for value in values {
                        visit(
                            backend,
                            value,
                            logical_device,
                            matrices,
                            compacts,
                            secrets,
                            observations,
                        )?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        let mut observations = GpuWarmupMemoryObservations {
            evidence: MemoryEvidenceKind::ExactQuery,
            ..Default::default()
        };
        let mut matrices = HashSet::new();
        let mut compacts = HashSet::new();
        let mut secrets = HashSet::new();
        for value in prepared
            .typed_inputs
            .as_ref()
            .into_iter()
            .flat_map(|(inputs, _, _)| inputs.values())
            .chain(prepared.trapdoor.iter())
        {
            visit(
                backend,
                value,
                logical_device,
                &mut matrices,
                &mut compacts,
                &mut secrets,
                &mut observations,
            )?;
        }
        Ok(observations)
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
                NodeKind::ConstantReal(_) |
                NodeKind::IntToReal |
                NodeKind::RealBinary(_) |
                NodeKind::RealSqrt |
                NodeKind::TrapdoorPublic
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
            NodeKind::CenteredRoundDivide { divisor } => {
                let divisor = divisor
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .centered_round_divide(matrix(0)?, &divisor)
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
            NodeKind::HashIntFamily { count, modulus, tag_prefix, tag_components } => {
                let count = evaluate_usize(count)?;
                let modulus = modulus
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?
                    .to_biguint()
                    .ok_or_else(|| {
                        GpuMeasurementError(
                            "hash integer-family modulus must be positive".to_owned(),
                        )
                    })?;
                if output_range.is_some_and(|range| range.start != 0 || range.end != count) {
                    return Err(GpuMeasurementError(
                        "hash integer-family warmup requires the complete single-device family"
                            .to_owned(),
                    ));
                }
                let mut tag = tag_prefix.clone();
                for component in tag_components {
                    use mxx_ir_core::node::HashTagComponent;
                    match component {
                        HashTagComponent::Bytes(bytes) => {
                            tag.push(0);
                            tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                            tag.extend_from_slice(bytes);
                        }
                        HashTagComponent::Integer(expression) => {
                            let value = expression
                                .evaluate(bindings)
                                .map_err(|error| GpuMeasurementError(error.to_string()))?;
                            let (sign, bytes) = value.to_bytes_be();
                            tag.push(1);
                            tag.push(u8::from(sign == num_bigint::Sign::Minus));
                            tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                            tag.extend_from_slice(&bytes);
                        }
                        HashTagComponent::Decimal(expression) => {
                            let decimal = expression
                                .evaluate(bindings)
                                .map_err(|error| GpuMeasurementError(error.to_string()))?
                                .to_string();
                            tag.push(2);
                            tag.extend_from_slice(&(decimal.len() as u64).to_be_bytes());
                            tag.extend_from_slice(decimal.as_bytes());
                        }
                        HashTagComponent::U64Le(expression) => {
                            let value = expression
                                .evaluate(bindings)
                                .map_err(|error| GpuMeasurementError(error.to_string()))?
                                .to_u64()
                                .ok_or_else(|| {
                                    GpuMeasurementError(
                                        "hash integer-family tag does not fit u64".to_owned(),
                                    )
                                })?;
                            tag.push(3);
                            tag.extend_from_slice(&value.to_le_bytes());
                        }
                        HashTagComponent::Operand(_) => {
                            return Err(GpuMeasurementError(
                                "hash integer-family warmup requires a static typed tag".to_owned(),
                            ));
                        }
                    }
                }
                (0..batch_size)
                    .map(|_| {
                        backend
                            .sample_hash_int_values(count, &modulus, [0x53; 32], &tag)
                            .map(GpuMeasurementOutput::IntegerValues)
                            .map_err(backend_error)
                    })
                    .collect()
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
                backend
                    .fixed_batch_extract_coefficient(
                        vec![(matrix_arc(0)?, position); batch_size],
                        &(0..batch_size).collect::<Vec<_>>(),
                    )
                    .map(|values| {
                        values.into_iter().map(GpuMeasurementOutput::IntegerValues).collect()
                    })
                    .map_err(backend_error)
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
                let output_bool = node.concrete_output_types.first().is_some_and(|ty| {
                    matches!(ty, ConcreteWireType::Bool | ConcreteWireType::ConstantBool)
                });
                backend
                    .fixed_batch_threshold_decode(
                        vec![(matrix_arc(0)?, modulus, length, output_bool); batch_size],
                        &(0..batch_size).collect::<Vec<_>>(),
                    )
                    .map(|values| {
                        values
                            .into_iter()
                            .flatten()
                            .map(GpuMeasurementOutput::IntegerValues)
                            .collect()
                    })
                    .map_err(backend_error)
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
                            let bits = (0..count)
                                .map(|index| BigInt::from(u8::from(index % coefficient_bits == 0)))
                                .collect::<Vec<_>>();
                            let bits =
                                backend.integer_values_from_host(&bits).map_err(backend_error)?;
                            backend
                                .fixed_batch_pack_polynomial_coefficients(
                                    vec![(ty.clone(), Arc::new(bits), coefficient_bits)],
                                    &[0],
                                )
                                .map_err(backend_error)?
                                .pop()
                                .ok_or_else(|| {
                                    GpuMeasurementError("native pack produced no matrix".into())
                                })
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
                            let values =
                                backend.integer_values_from_host(&values).map_err(backend_error)?;
                            backend
                                .polynomial_from_integer_values(&ty, &values, *evaluation)
                                .map_err(backend_error)
                        })
                        .collect(),
                )
            }
            NodeKind::PolynomialValues { evaluation } => {
                let value = matrix(0)?;
                (0..batch_size)
                    .map(|_| {
                        backend
                            .polynomial_values_resident(value, *evaluation)
                            .map(GpuMeasurementOutput::IntegerValues)
                            .map_err(backend_error)
                    })
                    .collect()
            }
            NodeKind::TrapdoorPublic => {
                Self::run_host_primitive(backend, node, bindings, batch_size)?;
                Ok(Vec::new())
            }
            NodeKind::Input { .. } |
            NodeKind::ConstantReal(_) |
            NodeKind::IntToReal |
            NodeKind::RealBinary(_) |
            NodeKind::RealSqrt => Ok(Vec::new()),
            NodeKind::ConstantInt(_) |
            NodeKind::EvaluateInt(_) |
            NodeKind::ConstantBool(_) |
            NodeKind::IntBinary(_) |
            NodeKind::IntCompare(_) |
            NodeKind::BitExtract { .. } |
            NodeKind::BoolToInt |
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic |
            NodeKind::Select { .. } |
            NodeKind::SubgraphCall(_) |
            NodeKind::ParallelLoop(_) |
            NodeKind::SequentialLoop(_) => Err(GpuMeasurementError(
                "resident control node reached matrix measurement without an integer owner".into(),
            )),
        }
    }
}

impl GpuWarmupProfileProvider for ProductionGpuWarmupProvider {
    fn configure_gpu_plan_budgets(
        &mut self,
        budgets: &[crate::gpu_execution_plan::GpuDeviceBudget],
    ) -> Result<(), GpuWarmupProfileError> {
        if self.workers.iter().any(|worker| worker.backend.frozen_plan().is_some()) {
            return Err(GpuWarmupProfileError::Measurement(
                "cannot change budgets after installing a frozen plan".into(),
            ));
        }
        let fleet = self.physical_device_ids();
        // Validate every projection before mutating any worker's setup budget.
        let projected = self
            .workers
            .iter()
            .map(|worker| {
                crate::gpu_execution_plan::project_gpu_device_budgets(
                    &fleet,
                    &worker.backend.physical_device_ids(),
                    budgets,
                )
                .map_err(GpuWarmupProfileError::Measurement)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (worker, budget) in self.workers.iter_mut().zip(projected) {
            worker
                .backend
                .configure_gpu_plan_budgets(&budget)
                .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
        }
        Ok(())
    }

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
        if let Some(host_control) = descriptor.host_control.clone() {
            self.host_control_operations
                .insert((descriptor.scope.clone(), descriptor.node), host_control);
        }
        let pending = PendingMeasurement {
            warmup: Some(descriptor.clone()),
            scope: descriptor.scope,
            id: descriptor.node,
            kind: descriptor.kind,
            concrete_argument_types: descriptor.concrete_argument_types,
            concrete_output_types: descriptor.concrete_output_types,
            bindings: descriptor.bindings,
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

/// Exact inputs for one canonical GPU preparation. Preparation derives the
/// backend contract, layouts, storage descriptors, and finite resource
/// budgets from the graph/backend pair. Callers supply only graph-independent
/// policy and measurement inputs.
pub(crate) struct GpuPreparationRequest<'a> {
    pub validated: ValidatedGraph,
    pub backend: &'a mut GpuDcrtBackend,
    pub inputs: &'a BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
    pub parameters: &'a [GpuDCRTPolyParams],
    pub default_tile_widths: Vec<usize>,
    pub implementation_variant: String,
    pub measurement_config: GpuWarmupMeasurementConfig,
    pub execution_config: ExecutionConfig,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GpuPreparationError {
    #[error("GPU warmup preparation failed: {0}")]
    Warmup(#[from] GpuWarmupError),
    #[error("GPU warmup preparation input is invalid: {0}")]
    InvalidInput(String),
}

/// Immutable setup output consumed by the compiled runtime. This is an
/// internal lowering result, not an execution facade: execution owns the
/// compiled plan and is entered only through `GpuRuntime::execute`.
#[derive(Clone, Debug)]
pub(crate) struct GpuPreparedSetup {
    validated: Arc<ValidatedGraph>,
    plan: Arc<FrozenGpuPlan>,
    plan_index: FrozenGpuPlanIndex,
    measured_costs: GpuMeasuredCostCache,
    report: GpuWarmupReport,
}

impl GpuPreparedSetup {
    pub(crate) fn validated(&self) -> &ValidatedGraph {
        &self.validated
    }

    pub(crate) fn plan(&self) -> &FrozenGpuPlan {
        &self.plan
    }

    /// Setup-time measured points used to rank the frozen candidate. Only
    /// timing and measurement scalars are reusable; route/resource evidence
    /// remains on the per-job warmup profile and is validated independently.
    pub(crate) fn measured_costs(&self) -> &GpuMeasuredCostCache {
        &self.measured_costs
    }

    /// Return the immutable lookup index paired with the frozen plan.  Runtime
    /// graph-capture preparation uses the same validated index as warmup so it
    /// cannot reconstruct layouts or node choices from operation identities.
    pub(crate) fn plan_index(&self) -> &FrozenGpuPlanIndex {
        &self.plan_index
    }

    pub(crate) fn report(&self) -> &GpuWarmupReport {
        &self.report
    }
}

/// Prepare one exact production GPU execution.  Measurement is performed at
/// most once here; all returned state is detached from the provider and is
/// safe to reuse for fixed execution and pure reporting.
pub(crate) fn prepare_gpu_setup(
    request: GpuPreparationRequest<'_>,
) -> Result<GpuPreparedSetup, GpuPreparationError> {
    // Setup baselines must not observe release work queued by a previous
    // preparation or execution. This is a backend-owned release-stream
    // boundary; callers should not have to reach through the runtime to fence
    // it before asking for a plan.
    request
        .backend
        .fence_released_memory()
        .map_err(|error| GpuPreparationError::InvalidInput(error.to_string()))?;
    let device_ids = request.backend.physical_device_ids();
    if device_ids.is_empty() {
        return Err(GpuPreparationError::InvalidInput("GPU fleet is empty".into()));
    }
    if request.parameters.is_empty() {
        return Err(GpuPreparationError::InvalidInput(
            "GPU preparation requires at least one native parameter set".into(),
        ));
    }
    let device_set = device_ids.iter().copied().collect::<BTreeSet<_>>();
    let parameter_devices = request
        .parameters
        .iter()
        .flat_map(|parameters| parameters.device_ids())
        .collect::<BTreeSet<_>>();
    if request.parameters.iter().any(|parameters| parameters.device_ids().is_empty()) ||
        parameter_devices != device_set
    {
        return Err(GpuPreparationError::InvalidInput(
            "native parameter device ownership disagrees with the production fleet".into(),
        ));
    }
    // A runtime may prepare a replacement graph after executing an earlier
    // plan.  The backend's frozen plan is the mutable execution binding, not
    // part of the caller-owned `GpuExecutionPlan`; release it before deriving
    // the replacement contract so budget admission does not mistake a valid
    // re-plan for an unsupported placement.
    request.backend.clear_frozen_plan();
    let mut contract = request
        .backend
        .gpu_runtime_contract(&request.validated, request.inputs)
        .map_err(|error| GpuPreparationError::InvalidInput(error.to_string()))?
        .ok_or_else(|| {
            GpuPreparationError::InvalidInput("backend has no fixed GPU contract".into())
        })?;
    contract.device_budgets = contract
        .logical_to_physical_devices
        .iter()
        .enumerate()
        .map(|(device, physical)| {
            Ok(GpuDeviceBudget {
                device,
                device_bytes: gpu_memory_info(*physical as i32)
                    .map_err(GpuPreparationError::InvalidInput)?
                    .free as u64,
                // Host/pageable and pinned RAM are executable-resource
                // requirements, not planner dimensions. Allocation failure
                // is reported by execution/store and never triggers a GPU
                // candidate fallback.
                pinned_host_bytes: 0,
                host_bytes: 0,
            })
        })
        .collect::<Result<Vec<_>, GpuPreparationError>>()?;
    let mut matrices = BTreeMap::<mxx_ir_core::types::ConcreteMatrixType, LayoutId>::new();
    for scope in request.validated.scopes.values() {
        for wire in scope.wire_types.values() {
            if let Some(matrix) = wire.matrix_type() {
                let id = matrices.len() as LayoutId + 1;
                matrices.entry(matrix.clone()).or_insert(id);
            }
        }
    }
    let layouts = matrices
        .into_iter()
        .map(|(matrix, id)| GpuLayout {
            id,
            columns: matrix.columns,
            rows: matrix.rows,
            ring_dimension: matrix.ring_dimension,
            representation: format!("{matrix:?}"),
            instance_device_stride: 0,
            owner_intervals: Vec::new(),
        })
        .collect();
    let device_count = contract.logical_to_physical_devices.len();
    let warmup_config = GpuValidatedWarmupConfig {
        contract,
        layouts,
        default_tile_widths: request.default_tile_widths,
        default_cost: vec![GpuStageCostModel::default(); device_count],
        default_implementation_variant: request.implementation_variant,
        profiles: BTreeMap::new(),
        effective_operation_identities: BTreeMap::new(),
        effective_operations: BTreeMap::new(),
        storage_descriptors: BTreeMap::new(),
        active_crt_towers: 0,
        crt_limb_bytes: 0,
        max_parallel_instances: request.execution_config.max_parallel_instances,
    };
    let measurement_backends = device_ids
        .iter()
        .map(|device| {
            request
                .backend
                .measurement_backend()
                .map(|backend| (backend, *device))
                .map_err(|error| GpuPreparationError::InvalidInput(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut provider =
        ProductionGpuWarmupProvider::new(measurement_backends, request.measurement_config);
    let warmup = crate::gpu_warmup::warmup_gpu_for_inputs_with_execution_config(
        &request.validated,
        request.backend,
        request.inputs,
        &warmup_config,
        &mut provider,
        &request.execution_config,
    )?;
    let measured_costs = provider.measured_costs().clone();
    drop(provider);
    let plan = Arc::new(warmup.plan);
    let plan_index = FrozenGpuPlanIndex::build(&plan)
        .map_err(|error| GpuPreparationError::InvalidInput(error.to_string()))?;
    Ok(GpuPreparedSetup {
        validated: Arc::new(request.validated),
        plan,
        plan_index,
        measured_costs,
        report: warmup.report,
    })
}

#[cfg(test)]
fn matrix_bytes(matrix: &ConcreteMatrixType, crt_depth: usize) -> u64 {
    u64::try_from(matrix.rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(matrix.columns).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(matrix.ring_dimension).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(crt_depth).unwrap_or(u64::MAX))
        .saturating_mul(8)
}

#[cfg(test)]
fn compact_matrix_bytes(matrix: &ConcreteMatrixType, max_coefficient_bound: &BigInt) -> u64 {
    u64::try_from(compact_matrix_bytes_u128(matrix, max_coefficient_bound)).unwrap_or(u64::MAX)
}

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::{
        GpuMeasuredCostCache, GpuMeasuredCostKey, GpuMeasuredCostPoint, GpuMeasurementOutput,
        GpuMeasurementTransportClass, MeasurementNode, PendingMeasurement, PreparedMeasurement,
        ProductionGpuWarmupProvider, authoritative_production_route, compact_matrix_bytes,
        harness::GpuWarmupMeasurementConfig, hash_int_family_owner_bytes, matrix_bytes,
        select_worker_index_for_physical, tensor_row_sum_allocation_groups,
    };
    use crate::{
        backend::{
            GpuEffectiveInputs, GpuWarmupCacheState, GpuWarmupDeviceIdentity,
            GpuWarmupFragmentClass, GpuWarmupOperationDescriptor, GpuWarmupOperationSignature,
            GpuWarmupProfileProvider, GpuWarmupProfileRequest, GpuWarmupProvenance,
            GpuWarmupResidencyDelta, GpuWarmupTimingScope, IndexRange as RuntimeIndexRange,
            MemoryEvidenceKind, poly::PolyBackendError, poly_gpu::gpu_backend,
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
    use mxx_ir_core::{
        FrozenGraphScopeId, IntExpr, ParamEnv, RealExpr,
        node::{ConcatAxis, ConstantMatrix, IndexRange, NodeKind},
        types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId},
    };
    use mxx_primitives::{
        gpu_memory::TensorRowSumImplementation,
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams},
        sampler::trapdoor::gpu::{
            PreimageAllocationEnvelope, PreimageAllocationEvidence, PreimageAllocationEvidenceKind,
            PreimageCacheIdentity, PreimageCacheState, PreimageEvidenceContext,
            PreimageFormatContext,
        },
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeSet;

    #[test]
    fn measured_cost_cache_keeps_resident_points_shared_and_routes_distinct() {
        let operation = GpuMeasuredCostKey {
            implementation_variant: crate::backend::GpuWarmupEffectiveVariant::Ordinary,
            operation_identity: [0x11; 32],
            transport_class: GpuMeasurementTransportClass::Resident,
        };
        let peer = GpuMeasuredCostKey {
            transport_class: GpuMeasurementTransportClass::Peer {
                source_device: 0,
                destination_device: 1,
            },
            ..operation.clone()
        };
        let mut cache = GpuMeasuredCostCache::default();
        cache.insert(
            operation.clone(),
            8,
            GpuMeasuredCostPoint {
                mean_seconds: 1.25,
                spread_seconds: 0.05,
                repetitions: 2,
                incremental_peak_bytes: 100,
                retained_delta_bytes: 40,
                graph_pool_peak_bytes: 0,
            },
        );
        cache.insert(
            peer.clone(),
            8,
            GpuMeasuredCostPoint {
                mean_seconds: 2.5,
                spread_seconds: 0.1,
                repetitions: 2,
                incremental_peak_bytes: 180,
                retained_delta_bytes: 80,
                graph_pool_peak_bytes: 12,
            },
        );
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.exact(&operation, 8).unwrap().retained_delta_bytes, 40);
        assert_eq!(cache.exact(&peer, 8).unwrap().incremental_peak_bytes, 180);
        assert!(cache.exact(&operation, 4).is_none());
    }

    #[test]
    fn grouped_tensor_row_sum_allocation_uses_every_output_group() {
        let small_grouped = vec![vec![0], vec![1, 2], vec![3]];
        assert_eq!(
            TensorRowSumImplementation::for_groups(&small_grouped),
            TensorRowSumImplementation::FusedKernel
        );

        let first_output = (0..16).map(|row| vec![row % 4]).collect::<Vec<_>>();
        let second_output = (0..17).map(|row| vec![row % 4]).collect::<Vec<_>>();
        let inputs = GpuEffectiveInputs {
            row_groups: first_output.clone(),
            row_sum_groups: vec![first_output.clone(), second_output],
            ..Default::default()
        };
        let groups = tensor_row_sum_allocation_groups(&inputs).expect("grouped topology");
        assert_eq!(groups.len(), 33);
        assert_eq!(groups.iter().map(Vec::len).sum::<usize>(), 33);
        // The native allocation query uses this same topology to choose the
        // reduction implementation and account its workspace. All grouped
        // outputs therefore cross the materialized-reduction threshold,
        // whereas the leader-only topology does not.
        assert_eq!(
            TensorRowSumImplementation::for_groups(&groups),
            TensorRowSumImplementation::MaterializedTensor
        );
        assert_eq!(
            TensorRowSumImplementation::for_groups(&inputs.row_groups),
            TensorRowSumImplementation::FusedKernel
        );
    }

    #[test]
    fn grouped_tensor_row_sum_topology_is_part_of_profile_context() {
        let leader = vec![vec![0], vec![1]];
        let make_descriptor = |row_sum_groups| GpuWarmupOperationDescriptor {
            inputs: GpuEffectiveInputs {
                row_groups: leader.clone(),
                row_sum_groups,
                ..Default::default()
            },
            signature: GpuWarmupOperationSignature {
                operation: [0x42; 32],
                shape_class: 0,
                instance_class: 0,
            },
            scope: FrozenGraphScopeId::Root,
            node: NodeId(0),
            kind: NodeKind::Tensor,
            concrete_argument_types: Vec::new(),
            concrete_output_types: Vec::new(),
            bindings: ParamEnv::default(),
            effective_operation: "fused_tensor_row_sum".into(),
            profile_domain: CanonicalWarmupProfileDomain::FusedTensorRowSum,
            fused_operation: Some(FusedWarmupOperation::TensorRowSum),
            implementation_variant: "test".into(),
            source_layouts: Vec::new(),
            output_layout: None,
            route_resolver: None,
            host_control: None,
        };
        let first = make_descriptor(vec![leader.clone(), vec![vec![2]]]);
        let changed_follower = make_descriptor(vec![leader.clone(), vec![vec![3]]]);
        let identical = make_descriptor(first.inputs.row_sum_groups.clone());

        let first_context = ProductionGpuWarmupProvider::profile_context_words(&first).unwrap();
        let changed_context =
            ProductionGpuWarmupProvider::profile_context_words(&changed_follower).unwrap();
        let identical_context =
            ProductionGpuWarmupProvider::profile_context_words(&identical).unwrap();
        assert_ne!(first_context, changed_context);
        assert_eq!(first_context, identical_context);
    }

    #[test]
    fn worker_selection_prefers_explicit_owner_in_replicated_full_fleet() {
        let worker_devices = [0, 1];
        let select = |physical| {
            select_worker_index_for_physical(worker_devices.len(), physical, |index| {
                worker_devices[index]
            })
        };

        assert_eq!(select(0), Some(0));
        assert_eq!(select(1), Some(1));
    }

    #[test]
    fn canonical_profile_inventory_is_closed_and_provider_rejects_domain_drift() {
        use crate::gpu_column_policy::{
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
        let mut provider = ProductionGpuWarmupProvider::from_workers(
            Vec::new(),
            GpuWarmupMeasurementConfig::default(),
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

    #[test]
    fn production_route_response_uses_physical_fragments_over_provisional_envelope() {
        let requested = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 3 },
            GpuFragmentClass::Tail,
        );
        let first = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 1 },
            GpuFragmentClass::Tail,
        );
        let mut last = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 2, end: 3 },
            GpuFragmentClass::Tail,
        );
        last.route = GpuTransferRoute::HostStaging;
        let resolved = authoritative_production_route(requested, &[first, last])
            .expect("physical resident fragments form a valid sparse route");
        assert_eq!(resolved.route, GpuTransferRoute::Resident);
        assert_eq!(resolved.source_range, ColumnRange { start: 0, end: 3 });
        assert_eq!(resolved.source_routes().len(), 2);
        assert!(resolved.validate());
    }

    /// Exercise every control profile through the real setup provider.
    /// The table deliberately carries the concrete node variant and wire
    /// contract for each case; a profile-only fake would not execute these
    /// production dispatches and would therefore miss invalid representatives.
    #[cfg(feature = "gpu")]
    #[test]
    #[serial_test::serial(gpu_context)]
    fn real_provider_dispatches_all_control_inventory_variants() {
        use crate::{
            backend::{
                GpuWarmupProfileProvider, GpuWarmupRoute, GpuWarmupRouteResolverData,
                poly_gpu::gpu_backend_on,
            },
            gpu_column_policy::GpuFragmentClass,
        };
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
        use num_bigint::BigInt;
        use std::time::Duration;

        let device = detected_gpu_device_ids()
            .into_iter()
            .next()
            .expect("GPU feature tests require one detected device");
        let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
        let harness = GpuWarmupMeasurementConfig {
            warm_up_iterations: 0,
            measured_iterations: 1,
            memory_poll_interval: Duration::ZERO,
        };
        let measurement_backend = gpu_backend_on([parameters], [device]);
        let mut provider =
            ProductionGpuWarmupProvider::new(vec![(measurement_backend, device)], harness);
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
                vec![int.clone()],
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
                vec![int.clone()],
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
                vec![int.clone()],
            ),
        ];
        for (index, (domain, kind, arguments, outputs)) in cases.into_iter().enumerate() {
            let measurement_kind = domain.measurement_kind();
            let gpu_measured = measurement_kind == WarmupMeasurementKind::GpuMeasured;
            let resolver = gpu_measured.then_some(GpuWarmupRouteResolverData {
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
            });
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
                    route_resolver: resolver.clone(),
                    host_control: None,
                })
                .expect("control inventory descriptor must register");
            let request = GpuWarmupProfileRequest {
                signature,
                device: 0,
                device_identity: GpuWarmupDeviceIdentity::new(0, "test", "test"),
                tile_width: 1,
                range: RuntimeIndexRange { start: 0, end: 1 },
                executed_range_start: 0,
                executed_range_class: GpuWarmupFragmentClass::Whole,
                route: if gpu_measured {
                    GpuWarmupRoute::DeviceLocal
                } else {
                    GpuWarmupRoute::HostOnly
                },
                route_descriptor: GpuExecutionRouteDescriptor::device_local(
                    0,
                    ColumnRange { start: 0, end: 1 },
                    GpuFragmentClass::Full,
                ),
                route_resolver: resolver,
                binding_port: None,
                fragment: GpuWarmupFragmentClass::Whole,
                retry_cap: None,
                cache_identity: None,
                cache_state: GpuWarmupCacheState::Warm,
                timing_scope: GpuWarmupTimingScope::LocalJob,
            };
            let profile = provider.measure(&request).unwrap_or_else(|error| {
                panic!("control production case {domain:?} must measure: {error}")
            });
            assert_eq!(profile.measurement, measurement_kind);
            assert_eq!(
                profile.resident_delta,
                GpuWarmupResidencyDelta::default(),
                "temporary measurement owners and aliased inputs are not production cache growth"
            );
            assert!(profile.time_seconds.is_finite() && profile.time_seconds > 0.0);
            if gpu_measured {
                assert!(profile.memory.affected_devices.values().any(|bytes| *bytes > 0));
            } else {
                assert_eq!(profile.workspace_bytes, 0);
            }
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
    fn real_provider_dispatches_gpu_inventory_cases_with_ranges() {
        use crate::{
            backend::{
                GpuWarmupProfileProvider, GpuWarmupRoute, GpuWarmupRouteResolverData,
                poly_gpu::gpu_backend_on,
            },
            gpu_column_policy::GpuFragmentClass,
        };
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
        let harness = GpuWarmupMeasurementConfig {
            warm_up_iterations: 0,
            measured_iterations: 1,
            memory_poll_interval: Duration::ZERO,
        };
        let measurement_backend = gpu_backend_on([parameters], [device]);
        let mut provider =
            ProductionGpuWarmupProvider::new(vec![(measurement_backend, device)], harness);
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
                CanonicalWarmupProfileDomain::HashIntFamily,
                NodeKind::HashIntFamily {
                    count: mxx_ir_core::IntExpr::constant(1),
                    modulus: mxx_ir_core::IntExpr::constant(256),
                    tag_prefix: vec![7],
                    tag_components: vec![],
                },
                vec![],
                vec![ConcreteWireType::IndexedFamily {
                    element: Box::new(ConcreteWireType::Int),
                    count: 1,
                }],
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
    #[serial_test::serial(gpu_context)]
    fn test_gpu_provider_projects_budgets_for_disjoint_and_replicated_workers() {
        use crate::backend::poly_gpu::gpu_backend_on;
        let dimension = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let primes = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(
            dimension, 1, 30, 8, None, None,
        )
        .to_crt()
        .0;
        let parameters = GpuDCRTPolyParams::new(dimension, primes, 8, None);
        let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();
        for replicated in [false, true] {
            let backends = devices
                .iter()
                .map(|physical| {
                    let owners = if replicated { devices.clone() } else { vec![*physical] };
                    (gpu_backend_on([parameters.clone()], owners), *physical)
                })
                .collect();
            let mut provider =
                ProductionGpuWarmupProvider::new(backends, GpuWarmupMeasurementConfig::default());
            let budgets = devices
                .iter()
                .enumerate()
                .map(|(logical, _)| GpuDeviceBudget {
                    device: logical,
                    device_bytes: 1_000_000 + logical as u64,
                    host_bytes: 2_000_000 + logical as u64,
                    pinned_host_bytes: 3_000_000 + logical as u64,
                })
                .collect::<Vec<_>>();
            provider.configure_gpu_plan_budgets(&budgets).unwrap();
            assert_eq!(provider.physical_device_ids(), devices);
            for worker in &provider.workers {
                let expected = crate::gpu_execution_plan::project_gpu_device_budgets(
                    &devices,
                    &worker.backend.physical_device_ids(),
                    &budgets,
                )
                .unwrap();
                assert_eq!(worker.backend.configured_plan_budgets(), Some(expected.as_slice()));
            }
            assert!(provider.configure_gpu_plan_budgets(&[]).is_err());
        }
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
                effective_operation: crate::gpu_column_policy::EffectiveGpuOperation::MatrixAdd,
                column_capability: crate::gpu_column_policy::ColumnCapability::SameColumns,
                output_layouts: vec![1],
                columns_per_job: vec![2, 4],
                implementation_variant: "test".into(),
                preimage_max_attempts: None,
            }],
        )
        .unwrap();
        assert_eq!(
            ProductionGpuWarmupProvider::fixed_plan_widths(&plan, site).unwrap(),
            GpuColumnWidths { gpu0: 2, nonzero: Some(4) }
        );
        assert_eq!(
            ProductionGpuWarmupProvider::fixed_plan_wave_seconds(
                &plan,
                site,
                &vec![crate::gpu_warmup::GpuTimeModel::default(); 2],
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
                effective_operation: crate::gpu_column_policy::EffectiveGpuOperation::MatrixAdd,
                column_capability: crate::gpu_column_policy::ColumnCapability::SameColumns,
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
            ProductionGpuWarmupProvider::fixed_plan_wave_seconds(&plan, site, &model).unwrap(),
            8.0
        );
    }

    #[test]
    fn only_production_range_operations_are_fleet_separable() {
        let matrix_type = MatrixType {
            rows: IntExpr::constant(8),
            columns: IntExpr::constant(8),
            ring_dimension: IntExpr::constant(32),
            modulus: IntExpr::constant(257),
        };

        assert!(ProductionGpuWarmupProvider::column_separable(&NodeKind::ConstantMatrix {
            matrix_type: matrix_type.clone(),
            value: ConstantMatrix::Identity
        }));
        assert!(ProductionGpuWarmupProvider::column_separable(&NodeKind::Transpose));
        assert!(ProductionGpuWarmupProvider::column_separable(&NodeKind::Concat {
            axis: mxx_ir_core::node::ConcatAxis::Columns,
        }));
        assert!(ProductionGpuWarmupProvider::column_separable(&NodeKind::Tensor));
        assert!(ProductionGpuWarmupProvider::column_separable(&NodeKind::Concat {
            axis: mxx_ir_core::node::ConcatAxis::Diagonal,
        }));
        assert!(ProductionGpuWarmupProvider::column_separable(&NodeKind::ConstantMatrix {
            matrix_type: MatrixType { rows: 1.into(), columns: 1.into(), ..matrix_type },
            value: ConstantMatrix::Rotation { exponent: IntExpr::constant(1) },
        }));
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
    fn hash_integer_family_allocation_matches_signed_words_owner() {
        let count = 12;
        assert_eq!(hash_int_family_owner_bytes(count, &BigUint::from(1u8)).unwrap(), 192);
        assert_eq!(hash_int_family_owner_bytes(count, &(BigUint::from(1u8) << 32)).unwrap(), 192);
        assert_eq!(hash_int_family_owner_bytes(count, &(BigUint::from(1u8) << 129)).unwrap(), 384);
        assert!(hash_int_family_owner_bytes(usize::MAX, &(BigUint::from(1u8) << 65)).is_err());
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
        let node = MeasurementNode {
            id: NodeId(1),
            kind: &kind,
            concrete_argument_types: Vec::new(),
            concrete_output_types: vec![ConcreteWireType::Trapdoor {
                matrix: full_ty.clone(),
                sigma: RealExpr::from_integer(4),
                gadget_base: BigInt::from(4),
                digit_count: params.modulus_digits(),
                preimage_max_coefficient_bound: BigInt::from(2),
            }],
        };
        let prepared = PreparedMeasurement {
            arguments: Vec::new(),
            small_arguments: Vec::new(),
            preimage_trapdoor: None,
            preimage_target: None,
        };
        let range = crate::backend::IndexRange { start: 7, end: 12 };
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

        let mut outputs = ProductionGpuWarmupProvider::run_node(
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
            scope: FrozenGraphScopeId::Root,
            id: NodeId(1),
            kind,
            concrete_argument_types: arguments,
            concrete_output_types: outputs,
            bindings: ParamEnv::default(),
        };
        let start = 13;
        let width = 4;

        let sampler = pending(
            NodeKind::UniformResidueSample { matrix_type: matrix_type(2, 64) },
            Vec::new(),
            vec![matrix(2, 64)],
        );
        let representative = ProductionGpuWarmupProvider::representative_at(&sampler, start, width);
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
        let representative = ProductionGpuWarmupProvider::representative_at(&slice, start, width);
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
        let representative =
            ProductionGpuWarmupProvider::representative_at(&transpose, start, width);
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
            ProductionGpuWarmupProvider::representative_at(&row_concat, start, width);
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
        let footprint = ProductionGpuWarmupProvider::preimage_footprint_from_evidence(&evidence)
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
        use crate::{
            Backend, RuntimeValue,
            artifact::MemoryArtifactStore,
            backend::{GpuWarmupProvenance, poly_gpu::gpu_backend_on},
            executor::{ExecutionConfig, ExecutionPlan, execute},
            gpu_warmup::{GpuProfileProvenance, GpuStageCostModel, GpuValidatedWarmupConfig},
            transcript::SamplingMode,
        };
        use mxx_dsl::{DslContext, Mat, Ring};
        use mxx_ir_core::node::ConcatAxis;
        use mxx_primitives::poly::dcrt::gpu::{
            GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync, gpu_memory_info,
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
        let harness = GpuWarmupMeasurementConfig {
            warm_up_iterations: 0,
            measured_iterations: 1,
            memory_poll_interval: Duration::ZERO,
        };
        let measurement_backend = gpu_backend_on([parameters.clone()], [device]);
        let mut provider =
            ProductionGpuWarmupProvider::new(vec![(measurement_backend, device)], harness);
        let config = GpuValidatedWarmupConfig {
            contract: GpuPlanContract {
                graph_specification_hash: [0; 32],
                backend_identity: "bench-real-provider-placeholder".into(),
                logical_to_physical_devices: vec![device as usize],
                device_budgets: vec![GpuDeviceBudget {
                    device: 0,
                    device_bytes: gpu_memory_info(device).expect("query GPU memory").total as u64,
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
        let mut warmup = crate::gpu_warmup::warmup_gpu_from_validated_with_provider(
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
        let output = execute(
            &graph,
            &mut backend,
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
            ExecutionConfig {
                plan: ExecutionPlan::FrozenGpu(std::sync::Arc::new(warmup.plan.clone())),
                ..ExecutionConfig::default()
            },
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
}
