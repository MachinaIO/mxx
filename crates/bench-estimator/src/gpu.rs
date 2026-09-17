//! GPU measurements for individual validated IR nodes.
//!
//! This adapter constructs representative zero-valued inputs and invokes the same
//! backend operations as the runtime. It measures operation cost; it is not a
//! second graph executor and does not define node semantics.
#[path = "dataflow_gpu.rs"]
mod dataflow;
use crate::{
    MeasurementBackend, MeasurementNode, NodeMeasurement, harness::MeasurementHarnessConfig,
};
use mxx_ir_core::{
    Graph, GraphOutput, NodeHandle, ParamEnv, WireType,
    artifact::{ArtifactConfidentiality, ArtifactType},
    encoding,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, MatrixType},
    validate,
};
use mxx_primitives::{
    matrix::{
        PolyMatrixColumnSource, SmallPolyMatrix,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuMatrixSampleDist, GpuSmallMatrix},
    },
    poly::dcrt::gpu::{
        gpu_default_mempool_reset_high_water, gpu_default_mempool_usage, gpu_device_memory_usage,
        gpu_memory_info,
    },
};
use mxx_runtime::{
    Backend, ExecutionResult, RuntimeValue,
    backend::{
        IndexRange,
        poly_gpu::{GpuDcrtBackend, GpuFleetMatrix, GpuFleetSmallMatrix, GpuFleetTrapdoor},
    },
    gpu_calibration::{
        gpu_calibration_operation_identity, gpu_capped_waterfill_columns,
        gpu_matrix_multiply_scales_left, gpu_operation_is_column_separable_for_types,
    },
    gpu_enqueue::GpuEnqueuePool,
    gpu_schedule::{GpuColumnInterval, GpuColumnJob, GpuColumnSchedule},
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::Arc,
};
use tracing::{debug, info};

#[derive(Debug)]
pub struct GpuMeasurementError(pub(crate) String);

impl fmt::Display for GpuMeasurementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GpuMeasurementError {}

#[derive(Clone, Copy)]
struct GpuMemoryMeasurementBaseline {
    pool_used_current: usize,
    context_generation: u64,
}

fn require_exclusive_measurement_context(
    device_id: i32,
    live_contexts: usize,
) -> Result<(), GpuMeasurementError> {
    if live_contexts == 1 {
        Ok(())
    } else {
        Err(GpuMeasurementError(format!(
            "GPU {device_id} has {live_contexts} live mxx contexts; exclusive CUDA mempool measurement is required"
        )))
    }
}

fn begin_gpu_memory_measurement(
    worker: &mut GpuMeasurementWorker,
) -> Result<GpuMemoryMeasurementBaseline, GpuMeasurementError> {
    let device_id = worker.device_id;
    let memory = gpu_device_memory_usage(device_id).map_err(GpuMeasurementError)?;
    require_exclusive_measurement_context(device_id, memory.live_contexts)?;
    // Matrix readiness precedes owner destruction, which queues frees on separate release
    // streams. Match the runtime prepared boundary before sampling the allocator baseline.
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
    })
}

fn finish_gpu_memory_measurement(
    device_id: i32,
    baseline: GpuMemoryMeasurementBaseline,
) -> Result<u64, GpuMeasurementError> {
    let memory = gpu_device_memory_usage(device_id).map_err(GpuMeasurementError)?;
    require_exclusive_measurement_context(device_id, memory.live_contexts)?;
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

#[derive(Clone)]
struct PreparedMeasurement {
    arguments: Vec<Option<Arc<GpuFleetMatrix>>>,
    small_arguments: Vec<Option<Arc<GpuFleetSmallMatrix>>>,
    preimage_trapdoor: Option<(GpuFleetMatrix, GpuFleetTrapdoor, f64, BigInt, usize, BigInt)>,
    preimage_target: Option<Arc<dyn PolyMatrixColumnSource<GpuFleetMatrix>>>,
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

    fn finish(&self) -> Result<(), GpuMeasurementError> {
        self.arguments
            .iter()
            .flatten()
            .try_for_each(|value| value.wait_until_ready().map_err(GpuMeasurementError))?;
        self.small_arguments
            .iter()
            .flatten()
            .try_for_each(|value| value.wait_until_ready().map_err(GpuMeasurementError))?;
        if let Some((public, trapdoor, ..)) = &self.preimage_trapdoor {
            public.wait_until_ready().map_err(GpuMeasurementError)?;
            trapdoor.wait_until_ready().map_err(GpuMeasurementError)?;
        }
        Ok(())
    }
}

struct GpuMeasurementWorker {
    backend: GpuDcrtBackend,
    device_id: i32,
}

#[derive(Clone)]
struct PendingMeasurement {
    key: [u8; 32],
    scope: mxx_ir_core::FrozenGraphScopeId,
    id: mxx_ir_core::types::NodeId,
    kind: NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
    bindings: ParamEnv,
    preimage_sample: bool,
}

#[derive(Clone)]
struct RepresentativeMeasurement {
    kind: NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
    fixed_arguments: Vec<bool>,
    output_range: Option<IndexRange>,
}

struct PreparedProgram {
    representative: RepresentativeMeasurement,
    members: Vec<PreparedMeasurement>,
    graph: mxx_ir_core::ValidatedGraph,
}

fn fleet_replay_timer(timed: bool) -> Option<std::time::Instant> {
    timed.then(std::time::Instant::now)
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

fn extrapolate_fleet_waves(full_wave: &NodeMeasurement, wave_count: usize) -> NodeMeasurement {
    NodeMeasurement {
        work_seconds: full_wave.work_seconds * wave_count as f64,
        latency_seconds: full_wave.latency_seconds,
        cumulative_wave_seconds: full_wave.latency_seconds * wave_count as f64,
        independent_wave_count: wave_count,
        measured_wave_workspace_bytes: full_wave.measured_wave_workspace_bytes,
        workspace_bytes: full_wave
            .workspace_bytes
            .saturating_mul(u64::try_from(wave_count).unwrap_or(u64::MAX)),
    }
}

/// The current shape-only API has no retained input ownership. These classes
/// describe an explicit synthetic fresh-placement scenario, not runtime admission.
#[derive(Clone, Debug, Serialize)]
struct NominalGpuWaveClass {
    schedule: GpuColumnSchedule,
    global_column_start: usize,
    multiplicity: usize,
}

fn nominal_gpu_wave_classes(
    total_columns: usize,
    capacities: &[usize],
) -> Result<Vec<NominalGpuWaveClass>, GpuMeasurementError> {
    let fleet_columns = capacities
        .iter()
        .try_fold(0usize, |total, width| total.checked_add(*width))
        .ok_or_else(|| GpuMeasurementError("GPU fleet capacity overflow".into()))?;
    if fleet_columns == 0 {
        return Err(GpuMeasurementError("GPU fleet capacity must be positive".into()));
    }
    let full_waves = total_columns / fleet_columns;
    let remainder = total_columns % fleet_columns;
    let mut classes = Vec::with_capacity(2);
    for (columns, global_column_start, multiplicity) in [
        (fleet_columns, 0, full_waves),
        (remainder, total_columns - remainder, usize::from(remainder > 0)),
    ] {
        if multiplicity == 0 {
            continue;
        }
        let assigned = gpu_capped_waterfill_columns(capacities, columns)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let mut next = 0;
        let intervals = assigned
            .into_iter()
            .enumerate()
            .filter_map(|(device, width)| {
                let start = next;
                next += width;
                (width > 0).then_some(GpuColumnInterval { device, start, end: next })
            })
            .collect();
        let schedule = GpuColumnSchedule::new(columns, capacities.to_vec(), intervals)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        classes.push(NominalGpuWaveClass { schedule, global_column_start, multiplicity });
    }
    Ok(classes)
}

/// One measured fleet wave class: the representative jobs in global column
/// coordinates, how many production waves share that shape, and the placement
/// it was derived from. Classes come from the representative shape placement.
#[derive(Clone, Debug, Serialize)]
struct MeasuredWaveClass {
    schedule: GpuColumnSchedule,
    representative: Vec<GpuColumnJob>,
    multiplicity: usize,
    first_wave: usize,
    source: &'static str,
}

const NOMINAL_SCENARIO: &str = "synthetic fresh placement; no invocation admission";

fn nominal_measured_classes(classes: Vec<NominalGpuWaveClass>) -> Vec<MeasuredWaveClass> {
    classes
        .into_iter()
        .map(|class| {
            let representative = class
                .schedule
                .wave_jobs(0)
                .into_iter()
                .map(|job| GpuColumnJob {
                    start: class.global_column_start + job.start,
                    end: class.global_column_start + job.end,
                    ..job
                })
                .collect();
            MeasuredWaveClass {
                schedule: class.schedule,
                representative,
                multiplicity: class.multiplicity,
                first_wave: 0,
                source: NOMINAL_SCENARIO,
            }
        })
        .collect()
}

fn accumulate_wave_class(total: &mut NodeMeasurement, wave: &NodeMeasurement, multiplicity: usize) {
    let class = extrapolate_fleet_waves(wave, multiplicity);
    total.work_seconds += class.work_seconds;
    total.cumulative_wave_seconds += class.cumulative_wave_seconds;
    total.latency_seconds = total.latency_seconds.max(class.latency_seconds);
    total.independent_wave_count = total.independent_wave_count.saturating_add(multiplicity);
    total.measured_wave_workspace_bytes =
        total.measured_wave_workspace_bytes.max(class.measured_wave_workspace_bytes);
    total.workspace_bytes = total.workspace_bytes.saturating_add(class.workspace_bytes);
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
    enqueue: GpuEnqueuePool,
    harness: MeasurementHarnessConfig,
    /// Setup-time snapshot for the fixed estimator execution policy.
    vram_percent: u32,
    measurements: HashMap<[u8; 32], NodeMeasurement>,
    measured_fleet_waves: HashMap<[u8; 32], NodeMeasurement>,
    /// Shape requests resolve only within this backend's frozen synthetic observation set.
    measurement_keys: HashMap<[u8; 32], [u8; 32]>,
    pending: HashMap<[u8; 32], PendingMeasurement>,
    collecting: bool,
    transfers: dataflow::TransferMeasurements,
}

impl GpuNodeMeasurementBackend {
    /// Creates a representative GPU measurement backend for validated IR nodes.
    pub fn new(backends: Vec<(GpuDcrtBackend, i32)>, harness: MeasurementHarnessConfig) -> Self {
        assert!(!backends.is_empty(), "GPU measurement requires at least one backend");
        let vram_percent = backends[0].0.vram_percent();
        assert!(
            backends.iter().all(|(backend, _)| backend.vram_percent() == vram_percent),
            "all GPU measurement contexts must use the same VRAM percentage"
        );
        let workers = backends
            .into_iter()
            .map(|(backend, device_id)| GpuMeasurementWorker { backend, device_id })
            .collect();
        Self::from_workers(workers, harness, vram_percent)
    }

    fn from_workers(
        workers: Vec<GpuMeasurementWorker>,
        harness: MeasurementHarnessConfig,
        vram_percent: u32,
    ) -> Self {
        let enqueue = GpuEnqueuePool::new(workers.len()).expect("create GPU enqueue workers");
        Self {
            workers,
            enqueue,
            harness,
            vram_percent,
            measurements: HashMap::new(),
            measured_fleet_waves: HashMap::new(),
            measurement_keys: HashMap::new(),
            pending: HashMap::new(),
            collecting: true,
            transfers: dataflow::TransferMeasurements::default(),
        }
    }

    fn zero_cost(kind: &NodeKind) -> bool {
        matches!(
            kind,
            NodeKind::Input { .. } |
                NodeKind::ConstantInt(_) |
                NodeKind::EvaluateInt(_) |
                NodeKind::ConstantReal(_) |
                NodeKind::ConstantBool(_) |
                NodeKind::TrapdoorPublic |
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
        )
    }

    /// Measures every collected shape as one fleet operation. Column-separable nodes use all
    /// workers for the same primitive instead of assigning unrelated primitives to different GPUs.
    pub fn measure_collected(&mut self) -> Result<(), GpuMeasurementError> {
        if !self.enqueue.is_healthy() {
            return Err(GpuMeasurementError("GPU enqueue workers are unavailable".into()));
        }
        self.collecting = false;
        let mut requests = std::mem::take(&mut self.pending).into_values().collect::<Vec<_>>();
        requests.par_sort_by(|left, right| {
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
            self.measurements.insert(self.measurement_keys[&request.key], measurement);
        }
        self.transfers.measure(&mut self.workers, &self.harness)?;
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
            NodeKind::PreimageSample { .. } => index < 2,
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

    fn request_columns(request: &PendingMeasurement) -> Option<usize> {
        if !gpu_operation_is_column_separable_for_types(
            &request.kind,
            &request.concrete_argument_types,
        ) {
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
            (NodeKind::GadgetTrapdoor { .. }, Some(matrix))
                if matrix.rows == 0 || !matrix.columns.is_multiple_of(matrix.rows) =>
            {
                None
            }
            _ => Some(columns),
        }
    }

    fn representative(request: &PendingMeasurement, columns: usize) -> RepresentativeMeasurement {
        Self::representative_at(request, 0, columns)
    }

    fn fixed_input_representative(request: &PendingMeasurement) -> RepresentativeMeasurement {
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
                    // boundary so setup includes the location-dependent piece/launch cost
                    // without materializing the complete concatenation.
                    return Self::representative_at(request, boundary - 1, 2);
                }
            }
        }
        Self::representative(request, 1)
    }

    fn representative_at(
        request: &PendingMeasurement,
        global_column_start: usize,
        columns: usize,
    ) -> RepresentativeMeasurement {
        let global_column_end =
            global_column_start.checked_add(columns).expect("representative column range overflow");
        if matches!(
            request.kind,
            NodeKind::ConstantMatrix {
                value: ConstantMatrix::Zero |
                    ConstantMatrix::Identity |
                    ConstantMatrix::UnitRow { .. } |
                    ConstantMatrix::Gadget { .. },
                ..
            }
        ) {
            return RepresentativeMeasurement {
                kind: request.kind.clone(),
                concrete_argument_types: request.concrete_argument_types.clone(),
                concrete_output_types: request.concrete_output_types.clone(),
                fixed_arguments: Self::fixed_arguments(
                    &request.kind,
                    &request.concrete_argument_types,
                ),
                output_range: Some(IndexRange {
                    start: global_column_start,
                    end: global_column_end,
                }),
            };
        }
        if matches!(request.kind, NodeKind::GadgetTrapdoor { .. }) {
            return RepresentativeMeasurement {
                kind: request.kind.clone(),
                concrete_argument_types: request.concrete_argument_types.clone(),
                concrete_output_types: request.concrete_output_types.clone(),
                fixed_arguments: Self::fixed_arguments(
                    &request.kind,
                    &request.concrete_argument_types,
                ),
                output_range: Some(IndexRange {
                    start: global_column_start,
                    end: global_column_end,
                }),
            };
        }
        if matches!(request.kind, NodeKind::Concat { axis: ConcatAxis::Diagonal }) {
            return RepresentativeMeasurement {
                kind: request.kind.clone(),
                concrete_argument_types: request.concrete_argument_types.clone(),
                concrete_output_types: request.concrete_output_types.clone(),
                fixed_arguments: Self::fixed_arguments(
                    &request.kind,
                    &request.concrete_argument_types,
                ),
                output_range: Some(IndexRange {
                    start: global_column_start,
                    end: global_column_end,
                }),
            };
        }
        if matches!(request.kind, NodeKind::Tensor) {
            let mut concrete_argument_types = request.concrete_argument_types.clone();
            let [left_wire, right_wire, ..] = concrete_argument_types.as_mut_slice() else {
                panic!("validated tensor representative must have two matrix arguments");
            };
            let right_columns =
                right_wire.matrix_type().expect("validated tensor RHS must be a matrix").columns;
            assert!(right_columns > 0, "validated tensor RHS must have columns");
            let local_start = global_column_start % right_columns;
            let local_end = local_start
                .checked_add(columns)
                .expect("tensor representative column range overflow");
            match left_wire {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage { matrix, .. } => {
                    // A prepared representative is a standalone exact operation, not a full
                    // tensor with a bookkeeping slice. Keep both operands at the requested
                    // output width so validation derives that same width from Tensor.
                    matrix.columns = 1
                }
                _ => panic!("validated tensor LHS must be a matrix"),
            }
            match right_wire {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage { matrix, .. } => {
                    matrix.columns = columns
                }
                _ => panic!("validated tensor RHS must be a matrix"),
            }
            let mut concrete_output_types = request.concrete_output_types.clone();
            let output = concrete_output_types
                .iter_mut()
                .find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                })
                .expect("validated tensor must have a matrix output");
            output.columns = columns;
            return RepresentativeMeasurement {
                kind: request.kind.clone(),
                concrete_argument_types,
                concrete_output_types,
                fixed_arguments: Self::fixed_arguments(
                    &request.kind,
                    &request.concrete_argument_types,
                ),
                output_range: Some(IndexRange { start: local_start, end: local_end }),
            };
        }
        if matches!(request.kind, NodeKind::Concat { axis: ConcatAxis::Columns }) {
            let mut input_start = 0usize;
            let mut concrete_argument_types = Vec::new();
            for mut wire_type in request.concrete_argument_types.clone() {
                let input = match &mut wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => matrix,
                    _ => panic!("validated column concat argument must be a matrix"),
                };
                let input_end = input_start
                    .checked_add(input.columns)
                    .expect("concat representative column range overflow");
                let overlap_start = global_column_start.max(input_start);
                let overlap_end = global_column_end.min(input_end);
                if overlap_start < overlap_end {
                    input.columns = overlap_end - overlap_start;
                    concrete_argument_types.push(wire_type);
                }
                input_start = input_end;
            }
            let mut concrete_output_types = request.concrete_output_types.clone();
            let output = concrete_output_types
                .iter_mut()
                .find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                })
                .expect("validated column concat must have a matrix output");
            output.columns = columns;
            let fixed_arguments = vec![false; concrete_argument_types.len()];
            return RepresentativeMeasurement {
                kind: request.kind.clone(),
                concrete_argument_types,
                concrete_output_types,
                fixed_arguments,
                output_range: Some(IndexRange {
                    start: global_column_start,
                    end: global_column_end,
                }),
            };
        }
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
            Self::representative_node(&node, columns);
        RepresentativeMeasurement {
            kind,
            concrete_argument_types,
            concrete_output_types,
            fixed_arguments: Self::fixed_arguments(&request.kind, &request.concrete_argument_types),
            output_range: None,
        }
    }

    fn measure_fleet_request(
        &mut self,
        request: &PendingMeasurement,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        if !self.enqueue.is_healthy() {
            return Err(GpuMeasurementError("GPU enqueue workers are unavailable".into()));
        }
        let Some(total_columns) = Self::request_columns(request) else {
            let representative = RepresentativeMeasurement {
                kind: request.kind.clone(),
                concrete_argument_types: request.concrete_argument_types.clone(),
                concrete_output_types: request.concrete_output_types.clone(),
                fixed_arguments: Self::fixed_arguments(
                    &request.kind,
                    &request.concrete_argument_types,
                ),
                output_range: None,
            };
            self.record_observation_key(request, &[], &[], NOMINAL_SCENARIO)?;
            return Self::measure_representative(
                &mut self.workers[0],
                &self.harness,
                &request.scope,
                request.id,
                &request.bindings,
                &representative,
            );
        };

        // Prepared runtime warmup is the only resource admission and lowering path.  The
        // scheduler uses a conservative equal column partition; if a representative cannot be
        // lowered or admitted, warmup returns that error explicitly.
        let baseline_representative = Self::fixed_input_representative(request);
        let fixed_inputs = self
            .workers
            .par_iter_mut()
            .map(|worker| {
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
                    None,
                )?;
                prepared.finish()?;
                Ok(vec![prepared])
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
        let capacities = vec![total_columns; self.workers.len()];
        let fleet_columns = total_columns;
        let classes =
            nominal_measured_classes(nominal_gpu_wave_classes(total_columns, &capacities)?);
        let scenario = NOMINAL_SCENARIO;
        self.record_observation_key(request, &capacities, &classes, scenario)?;
        let wave_count = classes.iter().map(|class| class.multiplicity).sum::<usize>();
        let assigned_columns = gpu_capped_waterfill_columns(&capacities, total_columns)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        info!(
            scope = ?request.scope,
            node = request.id.0,
            gpu_count = self.workers.len(),
            fleet_wave_columns = fleet_columns,
            total_columns,
            wave_count,
            assigned_columns = ?assigned_columns,
            "prepared GPU fleet representative partition"
        );

        let offset_sensitive = matches!(
            request.kind,
            NodeKind::ConstantMatrix {
                value: ConstantMatrix::Identity |
                    ConstantMatrix::UnitRow { .. } |
                    ConstantMatrix::Gadget { .. },
                ..
            } | NodeKind::GadgetTrapdoor { .. } |
                NodeKind::Tensor |
                NodeKind::Concat { axis: ConcatAxis::Columns | ConcatAxis::Diagonal }
        );
        let mut measurement = NodeMeasurement { independent_wave_count: 0, ..Default::default() };
        // Position-dependent constants and range mappings are measured at each actual
        // offset. Their descriptors are generated lazily, so RAM never scales with waves.
        let expanded: Box<dyn Iterator<Item = MeasuredWaveClass> + '_> = if !offset_sensitive {
            Box::new(classes.into_iter())
        } else {
            Box::new(classes.into_iter().flat_map(|class| {
                (0..class.multiplicity).map(move |index| MeasuredWaveClass {
                    schedule: class.schedule.clone(),
                    representative: class
                        .representative
                        .iter()
                        .map(|job| GpuColumnJob {
                            start: job.start + index * fleet_columns,
                            end: job.end + index * fleet_columns,
                            ..*job
                        })
                        .collect(),
                    multiplicity: 1,
                    first_wave: index,
                    source: class.source,
                })
            }))
        };
        for class in expanded {
            let mut representatives = vec![None; self.workers.len()];
            for job in &class.representative {
                representatives[job.device] =
                    Some(Self::representative_at(request, job.start, job.end - job.start));
            }
            let wave_inputs = fixed_inputs.clone();
            let batch_sizes = representatives
                .iter()
                .zip(&wave_inputs)
                .map(
                    |(representative, inputs)| {
                        if representative.is_some() { inputs.len() } else { 0 }
                    },
                )
                .collect::<Vec<_>>();
            let wave_key = self.wave_observation_key(request, &class, &batch_sizes)?;
            if let Some(observed) = self.measured_fleet_waves.get(&wave_key) {
                accumulate_wave_class(&mut measurement, observed, class.multiplicity);
                continue;
            }
            let (measurements, fleet_wave_wall_seconds) = Self::measure_fleet_wave(
                &mut self.workers,
                &mut self.enqueue,
                &self.harness,
                &request.scope,
                request.id,
                &request.bindings,
                representatives
                    .into_iter()
                    .zip(wave_inputs)
                    .map(|(representative, inputs)| {
                        representative
                            .map(|representative| vec![(representative, inputs)])
                            .unwrap_or_default()
                    })
                    .collect(),
            )
            .map_err(|error| {
                GpuMeasurementError(format!(
                    "GPU wave class measurement failed ({}; {:?}): {error}",
                    class.source, class.representative,
                ))
            })?;
            let device_elapsed_seconds = measurements
                .iter()
                .map(|measurement| {
                    measurement.as_ref().map_or(0.0, |measurement| measurement.work_seconds)
                })
                .collect::<Vec<_>>();
            let full_wave =
                aggregate_fleet_wave(measurements.into_iter().flatten(), fleet_wave_wall_seconds);
            info!(
                scope = ?request.scope, node = request.id.0, kind = ?request.kind,
                scenario = class.source,
                timing_contract = "execution-owner CUDA events and coordinated host wall",
                schedule = ?class.schedule, representative_jobs = ?class.representative,
                first_wave = class.first_wave,
                multiplicity = class.multiplicity, ?device_elapsed_seconds, fleet_wave_wall_seconds,
                measured_wave_workspace_bytes = full_wave.measured_wave_workspace_bytes,
                measured_outputs_included = true,
                "measured GPU fleet wave class"
            );
            self.measured_fleet_waves.insert(wave_key, full_wave.clone());
            accumulate_wave_class(&mut measurement, &full_wave, class.multiplicity);
        }
        info!(
            scope = ?request.scope, node = request.id.0, kind = ?request.kind,
            scenario, wave_count, ideal_dependency_latency_seconds = measurement.latency_seconds,
            work_seconds = measurement.work_seconds,
            cumulative_wave_seconds = measurement.cumulative_wave_seconds,
            measured_wave_workspace_bytes = measurement.measured_wave_workspace_bytes,
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

    /// One measured fleet wave, independent of graph identity and repetition
    /// count. Bounded-batch request keys include canonical operand sharing.
    fn wave_observation_key(
        &self,
        request: &PendingMeasurement,
        class: &MeasuredWaveClass,
        batch_sizes: &[usize],
    ) -> Result<[u8; 32], GpuMeasurementError> {
        encoding::hash_canonical(&(
            request.key,
            &class.representative,
            batch_sizes,
            self.workers.iter().map(|worker| worker.device_id).collect::<Vec<_>>(),
            self.vram_percent,
            class.source,
            "prepared operand identities encoded in the request class",
            "execution-owner CUDA events; coordinated host wall; retirement excluded",
        ))
        .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn record_observation_key(
        &mut self,
        request: &PendingMeasurement,
        capacities: &[usize],
        classes: &[MeasuredWaveClass],
        scenario: &str,
    ) -> Result<(), GpuMeasurementError> {
        let devices = self.workers.iter().map(|worker| worker.device_id).collect::<Vec<_>>();
        let key = encoding::hash_canonical(&(
            request.key,
            devices,
            capacities,
            classes,
            self.vram_percent,
            (scenario, "prepared resident inputs; primitive local transfers"),
            "execution-owner CUDA event span; coordinated host wall; output retirement excluded",
        ))
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        self.measurement_keys.insert(request.key, key);
        Ok(())
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
            timing_contract: &'static str,
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
            timing_contract: "execution-owner CUDA event span and coordinated host wall",
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
            NodeKind::RnsModUp { .. } |
            NodeKind::RnsModDown { .. } |
            NodeKind::CenteredExtend { .. } |
            NodeKind::BlockModSwitch { .. } |
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
        input_layouts: Option<&[Option<mxx_runtime::backend::poly_gpu::GpuMatrixDescriptor>]>,
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
            let selected =
                selected && !(matches!(node.kind, NodeKind::PreimageSample { .. }) && index < 2);
            if !selected {
                arguments.push(None);
                small_arguments.push(None);
                continue;
            }
            let leaf = family_leaf_type(wire_type);
            let Some(matrix) = leaf.matrix_type() else {
                arguments.push(None);
                small_arguments.push(None);
                continue;
            };
            use mxx_primitives::poly::{PolyParams, dcrt::gpu::GpuRngSeed};
            use mxx_runtime::backend::poly_gpu::{GpuColumnShard, GpuMatrixFragmentDescriptor};
            let fragments =
                if let Some(Some(layout)) = input_layouts.and_then(|inputs| inputs.get(index)) {
                    layout.shards.clone()
                } else {
                    let parameters = backend
                        .resource_parameters(matrix)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    let counts = gpu_capped_waterfill_columns(
                        &vec![matrix.columns; parameters.len()],
                        matrix.columns,
                    )
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    let mut start = 0;
                    parameters
                        .into_iter()
                        .zip(counts)
                        .filter_map(|(parameters, columns)| {
                            let fragment = (columns > 0).then(|| GpuColumnShard {
                                device_id: parameters.device_ids()[0],
                                global_column_start: start,
                                value: GpuMatrixFragmentDescriptor {
                                    level: parameters.crt_depth() - 1,
                                    parameters,
                                    columns,
                                    evaluation: matches!(leaf, ConcreteWireType::Matrix(_)),
                                },
                            });
                            start += columns;
                            fragment
                        })
                        .collect()
                };
            match leaf {
                ConcreteWireType::Matrix(matrix) => {
                    // One random logical sample, created directly in the modeled
                    // CRT levels, formats and contexts. No conversion trial is needed.
                    let seed = GpuRngSeed::from_bytes(rand::random());
                    let shards = fragments
                        .into_par_iter()
                        .map(|fragment| {
                            let mut value = GpuDCRTPolyMatrix::new_empty_with_state(
                                &fragment.value.parameters,
                                matrix.rows,
                                fragment.value.columns,
                                fragment.value.level,
                                fragment.value.evaluation,
                                None,
                            );
                            value
                                .fill_distribution_columns(
                                    0..matrix.rows,
                                    0..fragment.value.columns,
                                    matrix.columns,
                                    fragment.global_column_start,
                                    GpuMatrixSampleDist::Uniform,
                                    0.0,
                                    u64::MAX,
                                    seed,
                                )
                                .map_err(GpuMeasurementError)?;
                            Ok(GpuColumnShard {
                                device_id: fragment.device_id,
                                global_column_start: fragment.global_column_start,
                                value,
                            })
                        })
                        .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                    arguments.push(Some(Arc::new(GpuFleetMatrix::new(
                        matrix.rows,
                        matrix.columns,
                        shards,
                    ))));
                    small_arguments.push(None);
                }
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
                ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                    let bound =
                        max_coefficient_bound.to_biguint().expect("validated compact bound");
                    let magnitude_bytes = usize::try_from(bound.bits().div_ceil(8))
                        .map_err(|_| {
                            GpuMeasurementError("compact bound width overflows usize".into())
                        })?
                        .max(1);
                    let shards = fragments
                        .into_par_iter()
                        .map(|fragment| {
                            let columns = fragment.value.columns;
                            let payload_len = matrix
                                .rows
                                .checked_mul(columns)
                                .and_then(|count| count.checked_mul(matrix.ring_dimension))
                                .and_then(|count| count.checked_mul(1 + magnitude_bytes))
                                .ok_or_else(|| {
                                    GpuMeasurementError(
                                        "compact matrix payload length overflows".into(),
                                    )
                                })?;
                            let mut payload = vec![0; payload_len];
                            if bound.bits() != 0 {
                                payload.par_chunks_mut(1 + magnitude_bytes).enumerate().for_each(
                                    |(index, coefficient)| {
                                        let local_column = index / matrix.ring_dimension;
                                        let global = ((local_column / columns) * matrix.columns +
                                            fragment.global_column_start +
                                            local_column % columns) *
                                            matrix.ring_dimension +
                                            index % matrix.ring_dimension;
                                        coefficient[0] = (global % 3) as u8;
                                        coefficient[1] = u8::from(coefficient[0] != 0);
                                    },
                                );
                            }
                            let value = GpuSmallMatrix::from_canonical_coefficients(
                                &fragment.value.parameters,
                                matrix.rows,
                                columns,
                                bound.clone(),
                                &payload,
                            )
                            .map_err(|error| GpuMeasurementError(error.to_string()))?;
                            Ok(GpuColumnShard {
                                device_id: fragment.device_id,
                                global_column_start: fragment.global_column_start,
                                value,
                            })
                        })
                        .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                    arguments.push(None);
                    small_arguments.push(Some(Arc::new(GpuFleetSmallMatrix::new(
                        matrix.rows,
                        matrix.columns,
                        shards,
                    ))));
                }
                _ => {
                    arguments.push(None);
                    small_arguments.push(None);
                }
            }
        }

        let preimage_trapdoor = if matches!(node.kind, NodeKind::PreimageSample { .. }) &&
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
            trapdoor.wait_until_ready().map_err(GpuMeasurementError)?;
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
        let preimage_target = if matches!(node.kind, NodeKind::PreimageSample { .. }) &&
            phase != Some(true) &&
            fixed_phase.is_none_or(|(selected, phase)| selected[2] == phase)
        {
            let target = arguments.get(2).and_then(Option::as_ref).ok_or_else(|| {
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

    fn wire_type(ty: &ConcreteWireType) -> WireType {
        match ty {
            ConcreteWireType::ConstantInt => WireType::ConstantInt,
            ConcreteWireType::ConstantReal => WireType::ConstantReal,
            ConcreteWireType::ConstantBool => WireType::ConstantBool,
            ConcreteWireType::Int => WireType::Int,
            ConcreteWireType::Real => WireType::Real,
            ConcreteWireType::Bool => WireType::Bool,
            ConcreteWireType::Bytes { length } => {
                WireType::Bytes { length: mxx_ir_core::IntExpr::constant(*length) }
            }
            ConcreteWireType::TypedBlob { type_name, schema_hash } => {
                WireType::TypedBlob { type_name: type_name.clone(), schema_hash: *schema_hash }
            }
            ConcreteWireType::Matrix(matrix) => WireType::Matrix(MatrixType {
                modulus: mxx_ir_core::IntExpr::constant(matrix.modulus.clone()),
                ring_dimension: mxx_ir_core::IntExpr::constant(matrix.ring_dimension),
                rows: mxx_ir_core::IntExpr::constant(matrix.rows),
                columns: mxx_ir_core::IntExpr::constant(matrix.columns),
            }),
            ConcreteWireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            } => WireType::Trapdoor {
                matrix: match Self::wire_type(&ConcreteWireType::Matrix(matrix.clone())) {
                    WireType::Matrix(matrix) => matrix,
                    _ => unreachable!(),
                },
                sigma: sigma.clone(),
                gadget_base: mxx_ir_core::IntExpr::constant(gadget_base.clone()),
                digit_count: mxx_ir_core::IntExpr::constant(*digit_count),
                preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(
                    preimage_max_coefficient_bound.clone(),
                ),
            },
            ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } => {
                WireType::SmallMatrix {
                    matrix: match Self::wire_type(&ConcreteWireType::Matrix(matrix.clone())) {
                        WireType::Matrix(matrix) => matrix,
                        _ => unreachable!(),
                    },
                    max_coefficient_bound: mxx_ir_core::IntExpr::constant(
                        max_coefficient_bound.clone(),
                    ),
                }
            }
            ConcreteWireType::Preimage { matrix, max_coefficient_bound } => WireType::Preimage {
                matrix: match Self::wire_type(&ConcreteWireType::Matrix(matrix.clone())) {
                    WireType::Matrix(matrix) => matrix,
                    _ => unreachable!(),
                },
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(
                    max_coefficient_bound.clone(),
                ),
            },
            ConcreteWireType::IndexedFamily { element, count } => WireType::IndexedFamily {
                element: Box::new(Self::wire_type(element)),
                count: mxx_ir_core::IntExpr::constant(*count),
            },
        }
    }

    fn representative_graph(
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        output_range: Option<&IndexRange>,
    ) -> Result<mxx_ir_core::ValidatedGraph, GpuMeasurementError> {
        let mut arguments = Vec::with_capacity(node.concrete_argument_types.len());
        for (index, concrete) in node.concrete_argument_types.iter().enumerate() {
            let wire_type = Self::wire_type(concrete);
            let value = NodeHandle::new(
                NodeKind::Input {
                    name: format!("argument_{index}"),
                    wire_type: wire_type.clone(),
                    artifact: None,
                },
                vec![],
                vec![wire_type],
            )
            .output(0)
            .ok_or_else(|| GpuMeasurementError("representative input has no output".into()))?;
            arguments.push(value);
        }
        let mut output_types =
            node.concrete_output_types.iter().map(Self::wire_type).collect::<Vec<_>>();
        let mut kind = node.kind.clone();
        if let Some(range) = output_range {
            let width = range.end.checked_sub(range.start).ok_or_else(|| {
                GpuMeasurementError("representative output range is reversed".into())
            })?;
            if width == 0 {
                return Err(GpuMeasurementError("representative output range is empty".into()));
            }
            if matches!(kind, NodeKind::Concat { axis: ConcatAxis::Diagonal }) {
                return Err(GpuMeasurementError(
                    "diagonal concat range cannot be lowered as an exact prepared representative"
                        .into(),
                ));
            }
            for output in &mut output_types {
                let matrix = match output {
                    WireType::Matrix(matrix) |
                    WireType::SmallMatrix { matrix, .. } |
                    WireType::Preimage { matrix, .. } |
                    WireType::Trapdoor { matrix, .. } => Some(matrix),
                    _ => None,
                };
                if let Some(matrix) = matrix {
                    matrix.columns = mxx_ir_core::IntExpr::constant(width);
                }
            }
            match &mut kind {
                NodeKind::ConstantMatrix { matrix_type, value } => {
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(width);
                    if let ConstantMatrix::UnitRow { index } = value {
                        let index = index.evaluate(bindings).map_err(|error| {
                            GpuMeasurementError(format!("unit-row representative index: {error}"))
                        })?;
                        let local = index
                            .to_usize()
                            .and_then(|index| index.checked_sub(range.start))
                            .filter(|index| *index < width)
                            .ok_or_else(|| {
                                GpuMeasurementError(
                                    "unit-row representative index is outside its output range"
                                        .into(),
                                )
                            })?;
                        *value = ConstantMatrix::UnitRow {
                            index: mxx_ir_core::IntExpr::constant(local),
                        };
                    }
                }
                NodeKind::GadgetTrapdoor { matrix_type, .. } => {
                    let rows = matrix_type
                        .rows
                        .evaluate(bindings)
                        .ok()
                        .and_then(|rows| rows.to_usize())
                        .filter(|rows| *rows > 0)
                        .ok_or_else(|| {
                            GpuMeasurementError(
                                "gadget trapdoor representative has invalid row count".into(),
                            )
                        })?;
                    if !width.is_multiple_of(rows) {
                        return Err(GpuMeasurementError(
                            "gadget trapdoor output range is not a complete gadget block".into(),
                        ));
                    }
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(width);
                }
                _ => {}
            }
        }
        let output = NodeHandle::new(kind, arguments, output_types.clone());
        let outputs = output_types
            .iter()
            .enumerate()
            .map(|(index, _)| {
                (
                    format!("output_{index}"),
                    GraphOutput {
                        value: output.output(index as u32).expect("output type exists"),
                        confidentiality: None,
                    },
                )
            })
            .collect();
        let (graph, _) = Graph::freeze(
            "gpu-estimator-representative",
            vec![],
            outputs,
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        validate(&graph, bindings).map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn runtime_input(
        ty: &ConcreteWireType,
        index: usize,
        prepared: &PreparedMeasurement,
    ) -> Result<RuntimeValue<GpuDcrtBackend>, GpuMeasurementError> {
        if let ConcreteWireType::IndexedFamily { element, count } = ty {
            let values = (0..*count)
                .map(|_| Self::runtime_input(element, index, prepared))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(RuntimeValue::IndexedFamily(values));
        }
        if matches!(ty, ConcreteWireType::Trapdoor { .. }) && index == 1 {
            let Some((public, secret, sigma, gadget_base, digit_count, _)) =
                &prepared.preimage_trapdoor
            else {
                return Err(GpuMeasurementError(
                    "prepared representative is missing its trapdoor".into(),
                ));
            };
            let matrix_type = ty
                .matrix_type()
                .ok_or_else(|| GpuMeasurementError("trapdoor has no matrix type".into()))?;
            return Ok(RuntimeValue::Trapdoor {
                secret: Some(Arc::new(secret.clone())),
                public: Arc::new(public.clone()),
                matrix_type: matrix_type.clone(),
                sigma: *sigma,
                gadget_base: gadget_base.clone(),
                digit_count: *digit_count,
                gadget_small: None,
            });
        }
        match ty {
            ConcreteWireType::Matrix(_) => prepared.arguments[index]
                .as_ref()
                .map(|value| RuntimeValue::matrix(value.as_ref().clone()))
                .ok_or_else(|| {
                    GpuMeasurementError(format!("missing prepared matrix argument {index}"))
                }),
            ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => prepared
                .small_arguments[index]
                .as_ref()
                .map(|value| RuntimeValue::small_matrix(value.as_ref().clone()))
                .ok_or_else(|| {
                    GpuMeasurementError(format!("missing prepared compact argument {index}"))
                }),
            ConcreteWireType::Trapdoor { .. } => prepared.arguments[index]
                .as_ref()
                .map(|value| RuntimeValue::matrix(value.as_ref().clone()))
                .ok_or_else(|| {
                    GpuMeasurementError(format!(
                        "missing prepared trapdoor public argument {index}"
                    ))
                }),
            ConcreteWireType::ConstantInt | ConcreteWireType::Int => {
                Ok(RuntimeValue::Int(BigInt::from(0)))
            }
            ConcreteWireType::ConstantReal | ConcreteWireType::Real => Ok(RuntimeValue::Real(0.0)),
            ConcreteWireType::ConstantBool | ConcreteWireType::Bool => {
                Ok(RuntimeValue::Bool(false))
            }
            ConcreteWireType::Bytes { length } => Ok(RuntimeValue::Bytes(vec![0; *length])),
            ConcreteWireType::TypedBlob { .. } => Ok(RuntimeValue::TypedBlob(Vec::new())),
            ConcreteWireType::IndexedFamily { .. } => unreachable!("family handled above"),
        }
    }

    fn run_prepared_program(
        backend: &mut GpuDcrtBackend,
        program: &PreparedProgram,
    ) -> Result<Vec<ExecutionResult<GpuDcrtBackend>>, GpuMeasurementError> {
        program
            .members
            .iter()
            .map(|member| {
                let inputs = program
                    .representative
                    .concrete_argument_types
                    .iter()
                    .enumerate()
                    .map(|(index, ty)| {
                        Self::runtime_input(ty, index, member)
                            .map(|value| (format!("argument_{index}"), value))
                    })
                    .collect::<Result<BTreeMap<_, _>, _>>()?;
                let mut store = mxx_runtime::MemoryArtifactStore::default();
                let mut result = mxx_runtime::execute(
                    &program.graph,
                    backend,
                    inputs,
                    &mut store,
                    mxx_runtime::transcript::SamplingMode::Fresh,
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
                // Prepared execution returns a lease-backed result.  Waiting on every
                // published output before returning is part of the measurement lifecycle:
                // dropping the result may queue release work, but it cannot make the next
                // prepared slot reusable while its terminal stream is still in flight.
                let output_names = result.output_names().map(str::to_owned).collect::<Vec<_>>();
                for name in output_names {
                    result
                        .materialize_output(&name, backend, &mut store)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                }
                Ok(result)
            })
            .collect()
    }

    fn warm_up_prepared_program(
        backend: &mut GpuDcrtBackend,
        program: &PreparedProgram,
    ) -> Result<(), GpuMeasurementError> {
        let first_inputs = program
            .representative
            .concrete_argument_types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                Self::runtime_input(ty, index, &program.members[0])
                    .map(|value| (format!("argument_{index}"), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        backend
            .warm_up_prepared_graph(
                &program.graph,
                &first_inputs,
                &mxx_runtime::ExecutionConfig::default(),
            )
            .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn run_device_iteration(
        worker: &mut GpuMeasurementWorker,
        program: &PreparedProgram,
        timed: bool,
    ) -> Result<(f64, Vec<ExecutionResult<GpuDcrtBackend>>), GpuMeasurementError> {
        let parameters = worker.backend.device_parameters();
        if parameters.len() != 1 {
            return Err(GpuMeasurementError(
                "each measurement worker must own exactly one configured GPU".into(),
            ));
        }
        let timing = if timed {
            Some(parameters[0].begin_device_timing().map_err(GpuMeasurementError)?)
        } else {
            None
        };
        let outputs = Self::run_prepared_program(&mut worker.backend, program)?;
        // Join all execution-owner streams before waiting. Retain outputs until the
        // caller has stopped its fleet wall clock; retirement belongs outside this timer.
        let seconds = if let Some(timing) = timing {
            let spans = timing.finish().map_err(GpuMeasurementError)?;
            if spans.len() != 1 || spans[0].0 != worker.device_id {
                return Err(GpuMeasurementError(
                    "device timing returned a different GPU fleet".into(),
                ));
            }
            spans[0].1
        } else {
            0.0
        };
        Ok((seconds, outputs))
    }

    fn run_fleet_iteration(
        workers: &mut Vec<GpuMeasurementWorker>,
        enqueue: &mut GpuEnqueuePool,
        prepared: &Arc<Vec<Vec<PreparedProgram>>>,
        prepare: bool,
        timed: bool,
    ) -> Result<(Vec<Option<f64>>, f64), GpuMeasurementError> {
        let active_workers = prepared.iter().filter(|groups| !groups.is_empty()).count();
        if active_workers == 0 {
            return Err(GpuMeasurementError("GPU fleet wave has no active device".to_owned()));
        }
        let prepared = prepared.clone();
        // Setup and warmup are deliberately not assigned a fleet wall timestamp. The timer is
        // created only for the replay path, after every prepared program has been published.
        let fleet_started = fleet_replay_timer(timed);
        let completed = enqueue
            .map(workers, move |device, worker| {
                let groups = &prepared[device];
                if groups.is_empty() {
                    return Ok(None);
                }
                let parameters = worker.backend.device_parameters();
                if prepare {
                    // Publish every representative before starting the device timer. Warmup is
                    // setup, not part of the prepared wave latency.
                    for program in groups {
                        Self::warm_up_prepared_program(&mut worker.backend, program)?;
                        drop(Self::run_prepared_program(&mut worker.backend, program)?);
                    }
                }
                let timing = if timed {
                    Some(parameters[0].begin_device_timing().map_err(GpuMeasurementError)?)
                } else {
                    None
                };
                let mut outputs = Vec::new();
                // Different range classes still share one execution-owner timer.
                // Enqueue every group before joining, retaining all wave outputs.
                for program in groups {
                    outputs.extend(Self::run_prepared_program(&mut worker.backend, program)?);
                }
                let seconds = if let Some(timing) = timing {
                    let spans = timing.finish().map_err(GpuMeasurementError)?;
                    if spans.len() != 1 || spans[0].0 != worker.device_id {
                        return Err(GpuMeasurementError(
                            "device timing returned a different GPU fleet".into(),
                        ));
                    }
                    spans[0].1
                } else {
                    0.0
                };
                Ok(Some((seconds, outputs)))
            })
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let fleet_wave_wall_seconds =
            fleet_started.map_or(0.0, |started| started.elapsed().as_secs_f64());
        let device_seconds = completed
            .into_iter()
            .map(|result| {
                result.map(|(seconds, outputs)| {
                    drop(outputs);
                    seconds
                })
            })
            .collect();
        Ok((device_seconds, fleet_wave_wall_seconds))
    }

    fn measure_fleet_wave(
        workers: &mut Vec<GpuMeasurementWorker>,
        enqueue: &mut GpuEnqueuePool,
        harness: &MeasurementHarnessConfig,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        id: mxx_ir_core::types::NodeId,
        bindings: &ParamEnv,
        waves: Vec<Vec<(RepresentativeMeasurement, Vec<PreparedMeasurement>)>>,
    ) -> Result<(Vec<Option<NodeMeasurement>>, f64), GpuMeasurementError> {
        if harness.measured_iterations == 0 {
            return Err(GpuMeasurementError("measured iteration count must be positive".to_owned()));
        }
        let prepared = workers
            .par_iter_mut()
            .zip(waves.into_par_iter())
            .map(|(worker, groups)| {
                let groups = groups
                    .into_iter()
                    .map(|(representative, fixed)| {
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
                        // Each entry supplies the actual fixed/borrowed owners
                        // of one member. Existing scaled operands may also be
                        // supplied to preserve shared or partially shared inputs.
                        // Only missing operands are generated, outside timing.
                        let batch = fixed
                            .into_iter()
                            .map(|fixed| {
                                let present = representative
                                    .fixed_arguments
                                    .iter()
                                    .enumerate()
                                    .map(|(index, required_fixed)| {
                                        *required_fixed ||
                                            fixed
                                                .arguments
                                                .get(index)
                                                .is_some_and(Option::is_some) ||
                                            fixed
                                                .small_arguments
                                                .get(index)
                                                .is_some_and(Option::is_some)
                                    })
                                    .collect::<Vec<_>>();
                                let scaled = Self::prepare(
                                    &mut worker.backend,
                                    &node,
                                    bindings,
                                    Some((&present, false)),
                                    None,
                                )?;
                                let prepared = fixed.merge_for_representative(
                                    scaled,
                                    &representative.fixed_arguments,
                                )?;
                                prepared.finish()?;
                                Ok(prepared)
                            })
                            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;

                        let graph = Self::representative_graph(
                            &node,
                            bindings,
                            representative.output_range.as_ref(),
                        )?;
                        Ok(PreparedProgram { representative, members: batch, graph })
                    })
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                Ok(groups)
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;

        let prepared = Arc::new(prepared);
        // Warm up exactly once before memory baselines and all host/device timers. Subsequent
        // warmup iterations exercise only the already-published prepared command tape.
        let _ = Self::run_fleet_iteration(workers, enqueue, &prepared, true, false)?;
        for _ in 0..harness.warm_up_iterations {
            let _ = Self::run_fleet_iteration(workers, enqueue, &prepared, false, false)?;
        }
        let baselines = workers
            .par_iter_mut()
            .zip(prepared.par_iter())
            .map(|(worker, prepared)| {
                if prepared.is_empty() {
                    return Ok(None);
                }
                begin_gpu_memory_measurement(worker).map(Some)
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;

        let mut device_seconds = vec![0.0; workers.len()];
        let mut fleet_seconds = 0.0;
        for _ in 0..harness.measured_iterations {
            let (iteration_devices, iteration_fleet) =
                Self::run_fleet_iteration(workers, enqueue, &prepared, false, true)?;
            for (total, elapsed) in device_seconds.iter_mut().zip(iteration_devices) {
                *total += elapsed.unwrap_or(0.0);
            }
            fleet_seconds += iteration_fleet;
        }
        let iterations = harness.measured_iterations as f64;
        let measurements = workers
            .par_iter_mut()
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
                    latency_seconds: fleet_seconds / iterations,
                    cumulative_wave_seconds: fleet_seconds / iterations,
                    independent_wave_count: 1,
                    measured_wave_workspace_bytes: workspace_bytes,
                    workspace_bytes,
                }))
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
        Ok((measurements, fleet_seconds / iterations))
    }

    fn measure_representative(
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
        let prepared = Self::prepare(&mut worker.backend, &node, bindings, None, None)?;
        prepared.finish()?;
        let graph =
            Self::representative_graph(&node, bindings, representative.output_range.as_ref())?;
        let program = PreparedProgram {
            representative: representative.clone(),
            members: vec![prepared],
            graph,
        };
        if harness.measured_iterations == 0 {
            return Err(GpuMeasurementError("measured iteration count must be positive".into()));
        }
        // Publish the prepared representative before allocator and timing baselines. All later
        // iterations, including explicit warmup iterations, replay this immutable program.
        Self::warm_up_prepared_program(&mut worker.backend, &program)?;
        drop(Self::run_prepared_program(&mut worker.backend, &program)?);
        for _ in 0..harness.warm_up_iterations {
            Self::run_device_iteration(worker, &program, false)?;
        }
        // Warmup output releases must complete before resetting the measured peak.
        let baseline = begin_gpu_memory_measurement(worker)?;
        let mut device_elapsed_seconds = 0.0;
        let mut fleet_wave_wall_seconds = 0.0;
        for _ in 0..harness.measured_iterations {
            let started = std::time::Instant::now();
            let (device_seconds, outputs) = Self::run_device_iteration(worker, &program, true)?;
            fleet_wave_wall_seconds += started.elapsed().as_secs_f64();
            device_elapsed_seconds += device_seconds;
            drop(outputs);
        }
        let iterations = harness.measured_iterations as f64;
        let workspace_bytes = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        info!(
            device_id = worker.device_id,
            device_elapsed_seconds = device_elapsed_seconds / iterations,
            fleet_wave_wall_seconds = fleet_wave_wall_seconds / iterations,
            measured_wave_workspace_bytes = workspace_bytes,
            measured_outputs_included = true,
            scenario = "synthetic fresh placement; no invocation admission",
            "measured atomic GPU operation"
        );
        Ok(NodeMeasurement {
            work_seconds: device_elapsed_seconds / iterations,
            latency_seconds: fleet_wave_wall_seconds / iterations,
            cumulative_wave_seconds: fleet_wave_wall_seconds / iterations,
            independent_wave_count: 1,
            measured_wave_workspace_bytes: workspace_bytes,
            workspace_bytes,
        })
    }
}

impl MeasurementBackend for GpuNodeMeasurementBackend {
    type Error = GpuMeasurementError;

    fn measure(
        &mut self,
        _: &str,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<NodeMeasurement, Self::Error> {
        if matches!(
            node.kind,
            NodeKind::PolynomialFromValues { .. } | NodeKind::PolynomialValues { .. }
        ) {
            return Err(GpuMeasurementError(
                "polynomial value import/export measurement is not supported".to_owned(),
            ));
        }
        if Self::zero_cost(node.kind) {
            return Ok(NodeMeasurement::default());
        }
        let measurement_key = Self::measurement_key(node, bindings)?;
        if let Some(measurement) =
            self.measurement_keys.get(&measurement_key).and_then(|key| self.measurements.get(key))
        {
            return Ok(measurement.clone());
        }
        if !self.collecting {
            return Err(GpuMeasurementError(format!(
                "GPU node shape at {:?} node {:?} was not collected before measurement",
                node.scope, node.id
            )));
        }
        self.pending.entry(measurement_key).or_insert_with(|| PendingMeasurement {
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

    fn measurement_scenario(&self) -> crate::MeasurementScenario {
        crate::MeasurementScenario::SyntheticFreshPlacement
    }

    fn executor_dispatch_seconds(&self) -> f64 {
        self.transfers.dispatch_seconds
    }
    fn measure_transfer(
        &mut self,
        kind: crate::dataflow::TransferKind,
        ty: &ConcreteWireType,
    ) -> Result<f64, Self::Error> {
        self.transfers.get(kind, ty, self.collecting)
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
                if artifact.confidentiality == ArtifactConfidentiality::Private =>
            {
                0
            }
            NodeKind::Input { .. } => self.persistent_storage_bytes_for_node(kind, wire_type),
            _ => 0,
        }
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
        GpuNodeMeasurementBackend, PendingMeasurement, PreparedMeasurement, accumulate_wave_class,
        aggregate_fleet_wave, compact_matrix_bytes, extrapolate_fleet_waves, fleet_replay_timer,
        gpu_capped_waterfill_columns, matrix_bytes, nominal_gpu_wave_classes,
        require_exclusive_measurement_context,
    };
    use crate::{MeasurementNode, NodeMeasurement};
    use mxx_ir_core::{
        FrozenGraphScopeId, IntExpr, ParamEnv, RealExpr,
        node::{ConstantMatrix, HashVariant, IndexRange, MatrixBinaryOp, NodeKind},
        types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId},
    };
    use mxx_primitives::poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams};
    use mxx_runtime::backend::IndexRange as RuntimeIndexRange;
    use num_bigint::BigInt;

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
    fn shared_cuda_pool_is_an_explicit_measurement_error() {
        assert!(require_exclusive_measurement_context(0, 1).is_ok());
        let error = require_exclusive_measurement_context(3, 2).unwrap_err();
        assert_eq!(
            error.to_string(),
            "GPU 3 has 2 live mxx contexts; exclusive CUDA mempool measurement is required"
        );
    }

    #[test]
    fn nominal_gpu_classes_keep_partial_wave_devices_and_exact_multiplicities() {
        let classes = nominal_gpu_wave_classes(23, &[4, 6]).unwrap();
        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0].multiplicity, 2);
        assert_eq!(classes[1].multiplicity, 1);
        assert_eq!(classes[1].global_column_start, 20);
        let tail = classes[1].schedule.waves().next().unwrap();
        assert_eq!(
            tail.iter().map(|job| (job.device, job.end - job.start)).collect::<Vec<_>>(),
            vec![(0, 2), (1, 1)]
        );
        let idle = nominal_gpu_wave_classes(1, &[4, 6, 3]).unwrap();
        assert_eq!(idle[0].schedule.waves().next().unwrap().len(), 1);
        let large = nominal_gpu_wave_classes(usize::MAX, &[1]).unwrap();
        assert_eq!(large.len(), 1);
        assert_eq!(large[0].multiplicity, usize::MAX);
        assert!(nominal_gpu_wave_classes(1, &[0]).is_err());
    }

    #[test]
    fn fleet_timer_exists_only_for_prepared_replay() {
        assert!(fleet_replay_timer(false).is_none());
        assert!(fleet_replay_timer(true).is_some());
    }

    #[test]
    fn gpu_class_aggregation_keeps_event_work_wall_cost_and_ideal_latency_distinct() {
        let mut total = NodeMeasurement { independent_wave_count: 0, ..Default::default() };
        let full = NodeMeasurement {
            work_seconds: 7.0,
            latency_seconds: 5.0,
            cumulative_wave_seconds: 5.0,
            independent_wave_count: 1,
            measured_wave_workspace_bytes: 13,
            workspace_bytes: 13,
        };
        // A smaller class may take longer; there is no timing monotonicity assumption.
        let tail = NodeMeasurement {
            work_seconds: 8.0,
            latency_seconds: 9.0,
            cumulative_wave_seconds: 9.0,
            independent_wave_count: 1,
            measured_wave_workspace_bytes: 11,
            workspace_bytes: 11,
        };
        accumulate_wave_class(&mut total, &full, 3);
        accumulate_wave_class(&mut total, &tail, 1);
        assert_eq!(total.work_seconds, 29.0);
        assert_eq!(total.cumulative_wave_seconds, 24.0);
        assert_eq!(total.latency_seconds, 9.0);
        assert_eq!(total.independent_wave_count, 4);
        assert_eq!(total.measured_wave_workspace_bytes, 13);
        assert_eq!(total.workspace_bytes, 50);
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
        assert_eq!(measurement.workspace_bytes, 392);
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
        let unit_row_representative =
            GpuNodeMeasurementBackend::fixed_input_representative(&unit_row_request);
        assert_eq!(
            unit_row_representative.output_range.map(|range| (range.start, range.end)),
            Some((6, 7))
        );
    }

    #[test]
    fn prepared_representative_preserves_requested_range_shape() {
        let kind = NodeKind::ConstantMatrix {
            matrix_type: MatrixType {
                rows: IntExpr::constant(8),
                columns: IntExpr::constant(8),
                ring_dimension: IntExpr::constant(32),
                modulus: IntExpr::constant(257),
            },
            value: ConstantMatrix::Identity,
        };
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
        let range = RuntimeIndexRange { start: 3, end: 5 };
        let graph = GpuNodeMeasurementBackend::representative_graph(
            &node,
            &ParamEnv::default(),
            Some(&range),
        )
        .expect("range representative validates");
        let representative = graph.source.root_scope().nodes().last().expect("representative node");
        let mxx_ir_core::WireType::Matrix(output) = &representative.output_types()[0] else {
            panic!("representative output is not a matrix");
        };
        assert_eq!(output.columns, IntExpr::constant(2));
        let NodeKind::ConstantMatrix { matrix_type, .. } = representative.kind() else {
            panic!("representative kind changed");
        };
        assert_eq!(matrix_type.columns, IntExpr::constant(2));
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
    #[serial_test::serial(gpu_context)]
    fn test_gpu_repeated_protocol_nodes_share_one_measurement() {
        use crate::MeasurementBackend;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(8);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 1, 17, 1, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 1, None);
        let ty = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 1,
            columns: 2,
            ring_dimension: n as usize,
            modulus: BigInt::from(params.modulus().as_ref().clone()),
        });
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        let backend = mxx_runtime::backend::poly::gpu::gpu_backend_on([params], [device]);
        let mut estimator = GpuNodeMeasurementBackend::new(
            vec![(backend, device)],
            crate::harness::MeasurementHarnessConfig {
                warm_up_iterations: 1,
                measured_iterations: 1,
                ..Default::default()
            },
        );
        let bindings = ParamEnv::default();
        let kind = NodeKind::MatrixNegate;
        let scope = FrozenGraphScopeId::Root;
        for id in 0..1024 {
            let node = MeasurementNode {
                scope: &scope,
                id: NodeId(id),
                kind: &kind,
                arguments: &[],
                argument_kinds: &[],
                argument_types: &[],
                output_types: &[],
                concrete_argument_types: vec![ty.clone()],
                concrete_output_types: vec![ty.clone()],
            };
            estimator.measure("repeated", &node, &bindings).unwrap();
        }
        assert_eq!(estimator.pending.len(), 1);
        estimator.measure_collected().unwrap();
        assert_eq!(estimator.measurements.len(), 1);
        assert_eq!(estimator.measured_fleet_waves.len(), 1);
        for id in 0..1024 {
            let node = MeasurementNode {
                scope: &scope,
                id: NodeId(id),
                kind: &kind,
                arguments: &[],
                argument_kinds: &[],
                argument_types: &[],
                output_types: &[],
                concrete_argument_types: vec![ty.clone()],
                concrete_output_types: vec![ty.clone()],
            };
            assert!(estimator.measure("repeated", &node, &bindings).unwrap().work_seconds > 0.0);
        }
        assert!(estimator.pending.is_empty());
        assert_eq!(estimator.measurements.len(), 1);
        assert_eq!(estimator.measured_fleet_waves.len(), 1);
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
    fn matrix_multiply_fixed_ownership_survives_one_column_setup() {
        let matrix = |rows, columns| {
            ConcreteWireType::Matrix(ConcreteMatrixType {
                rows,
                columns,
                ring_dimension: 8,
                modulus: BigInt::from(257u16),
            })
        };
        let request = PendingMeasurement {
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(7),
            kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            concrete_argument_types: vec![matrix(2, 1), matrix(1, 176)],
            concrete_output_types: vec![matrix(2, 176)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };

        let representative = GpuNodeMeasurementBackend::fixed_input_representative(&request);
        assert_eq!(representative.concrete_argument_types[0].matrix_type().unwrap().columns, 1);
        assert_eq!(representative.concrete_argument_types[1].matrix_type().unwrap().columns, 1);
        assert_eq!(representative.concrete_output_types[0].matrix_type().unwrap().columns, 1);
        assert_eq!(representative.fixed_arguments, vec![true, false]);

        // The shaped representative alone is ambiguous: it looks like a scalar-right product.
        // Fixed ownership must therefore remain the decision made from the unshaped
        // request.
        let representative_node = MeasurementNode {
            scope: &request.scope,
            id: request.id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        assert_eq!(
            (0..2)
                .map(|index| GpuNodeMeasurementBackend::argument_is_fixed(
                    &representative_node,
                    index
                ))
                .collect::<Vec<_>>(),
            vec![false, true]
        );

        let gpu_representative = GpuNodeMeasurementBackend::representative_at(&request, 0, 88);
        assert_eq!(gpu_representative.concrete_argument_types[0].matrix_type().unwrap().columns, 1);
        assert_eq!(
            gpu_representative.concrete_argument_types[1].matrix_type().unwrap().columns,
            88
        );
        assert_eq!(gpu_representative.concrete_output_types[0].matrix_type().unwrap().columns, 88);
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
        assert_eq!(representative.output_range.map(|range| (range.start, range.end)), Some((3, 7)));
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
        assert_eq!((left.columns, right.columns), (1, 45));
        assert_eq!(output.columns, 45);
        assert_eq!(
            representative.output_range.as_ref().map(|range| (range.start, range.end)),
            Some((35, 80))
        );
        let representative_node = MeasurementNode {
            scope: &request.scope,
            id: request.id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        let graph = GpuNodeMeasurementBackend::representative_graph(
            &representative_node,
            &request.bindings,
            representative.output_range.as_ref(),
        )
        .expect("tensor representative validates");
        assert_eq!(
            graph
                .root_scope()
                .wire_types
                .get(&graph.source.root_scope().outputs()[0])
                .and_then(ConcreteWireType::matrix_type)
                .expect("tensor output shape")
                .columns,
            45
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
        assert_eq!(representative.concrete_argument_types.len(), 1);
        assert_eq!(representative.concrete_argument_types[0].matrix_type().unwrap().columns, 4);

        let crossing = GpuNodeMeasurementBackend::representative_at(&request, 1, 4);
        assert_eq!(crossing.concrete_argument_types.len(), 2);
        assert_eq!(
            crossing
                .concrete_argument_types
                .iter()
                .map(|input| input.matrix_type().unwrap().columns)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );

        let representative = GpuNodeMeasurementBackend::fixed_input_representative(&request);
        assert_eq!(representative.concrete_argument_types.len(), 2);
        assert_eq!(
            representative
                .concrete_argument_types
                .iter()
                .map(|input| input.matrix_type().unwrap().columns)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );

        let two_column_request = PendingMeasurement {
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(4),
            kind: NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Columns },
            concrete_argument_types: vec![matrix(1), matrix(1)],
            concrete_output_types: vec![matrix(2)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
        };
        let assignments = gpu_capped_waterfill_columns(&[1154, 1154], 2).unwrap();
        assert_eq!(assignments, vec![1, 1]);
        let mut start = 0;
        for columns in assignments {
            let representative =
                GpuNodeMeasurementBackend::representative_at(&two_column_request, start, columns);
            assert_eq!(representative.concrete_argument_types.len(), 1);
            assert_eq!(representative.fixed_arguments, vec![false]);
            assert_eq!(
                representative.output_range.as_ref().map(|range| (range.start, range.end)),
                Some((start, start + 1))
            );
            let empty_prepared = |arguments| PreparedMeasurement {
                arguments: (0..arguments).map(|_| None).collect(),
                small_arguments: (0..arguments).map(|_| None).collect(),
                preimage_trapdoor: None,
                preimage_target: None,
            };
            let merged = empty_prepared(2)
                .merge_for_representative(empty_prepared(1), &representative.fixed_arguments)
                .unwrap();
            assert_eq!(merged.arguments.len(), 1);
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
}
