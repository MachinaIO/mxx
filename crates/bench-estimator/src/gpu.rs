//! GPU measurements for individual validated IR nodes.
//!
//! This adapter constructs representative zero-valued inputs and invokes the same
//! backend operations as the runtime. It measures operation cost; it is not a
//! second graph executor and does not define node semantics.

use crate::{
    MeasurementBackend, MeasurementNode, NodeMeasurement,
    harness::{MeasurementHarnessConfig, MemoryProbe, measure_batch_operation},
};
use mxx_ir_core::{
    ParamEnv,
    artifact::ArtifactType,
    encoding,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixColumnSource, SmallPolyMatrix,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuSmallMatrix},
    },
    poly::dcrt::gpu::{
        gpu_default_mempool_reset_high_water, gpu_default_mempool_usage, gpu_device_memory_usage,
    },
};
use mxx_runtime::{
    Backend,
    backend::{
        IndexRange, MatrixMulAccumulateRequest, PreimageRequest, SampleRange,
        poly_gpu::{GpuDcrtBackend, GpuFleetMatrix, GpuFleetSmallMatrix, GpuFleetTrapdoor},
    },
    gpu_calibration::{
        GpuCalibrationKey, GpuCalibrationProfile, GpuCalibrationRegistry, GpuColumnWidths,
        GpuDeviceCalibration, GpuDeviceMemory, gpu_calibration_environment,
        gpu_calibration_operation_identity, gpu_capped_waterfill_columns,
        gpu_matrix_multiply_scales_left, gpu_operation_is_column_separable,
        gpu_operation_is_column_separable_for_types,
    },
};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Barrier},
};
use tracing::{debug, info};

#[derive(Debug)]
pub struct GpuMeasurementError(String);

impl fmt::Display for GpuMeasurementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GpuMeasurementError {}

struct GpuMemoryProbe {
    device_id: i32,
}

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
    device_id: i32,
) -> Result<GpuMemoryMeasurementBaseline, GpuMeasurementError> {
    let memory = gpu_device_memory_usage(device_id).map_err(GpuMeasurementError)?;
    require_exclusive_measurement_context(device_id, memory.live_contexts)?;
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

impl MemoryProbe for GpuMemoryProbe {
    type Error = GpuMeasurementError;

    fn current_bytes(&self) -> Result<u64, Self::Error> {
        let memory = gpu_default_mempool_usage(self.device_id).map_err(GpuMeasurementError)?;
        u64::try_from(memory.used_current)
            .map_err(|_| GpuMeasurementError("GPU memory usage exceeds u64".to_owned()))
    }
}

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
        }
    }
}

struct GpuMeasurementWorker {
    backend: GpuDcrtBackend,
    device_id: i32,
}

struct PendingMeasurement {
    key: [u8; 32],
    scope: mxx_ir_core::FrozenGraphScopeId,
    id: mxx_ir_core::types::NodeId,
    kind: NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
    bindings: ParamEnv,
    preimage_sample: bool,
    operation_batch_size: usize,
}

struct RepresentativeMeasurement {
    kind: NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
    fixed_arguments: Vec<bool>,
    output_range: Option<IndexRange>,
    operation_batch_size: usize,
}

fn family_leaf_type(wire_type: &ConcreteWireType) -> &ConcreteWireType {
    match wire_type {
        ConcreteWireType::Family { element, .. } => family_leaf_type(element),
        _ => wire_type,
    }
}

fn family_leaf_type_mut(wire_type: &mut ConcreteWireType) -> &mut ConcreteWireType {
    match wire_type {
        ConcreteWireType::Family { element, .. } => family_leaf_type_mut(element),
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

fn operation_batch_size(node: &MeasurementNode<'_>) -> Result<usize, GpuMeasurementError> {
    if !matches!(node.kind, NodeKind::FamilyPreimageSample { .. }) {
        return Ok(1);
    }
    let output = node
        .concrete_output_types
        .first()
        .ok_or_else(|| GpuMeasurementError("family preimage output type is missing".to_owned()))?;
    let ConcreteWireType::Family { element, shape } = output else {
        return Err(GpuMeasurementError("family preimage output is not a family".to_owned()));
    };
    if !matches!(element.as_ref(), ConcreteWireType::Preimage { .. }) {
        return Err(GpuMeasurementError(
            "family preimage output does not contain preimages".to_owned(),
        ));
    }
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |count, extent| count.checked_mul(*extent)).ok_or_else(|| {
        GpuMeasurementError("family preimage output cardinality overflows usize".to_owned())
    })
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
    let wave_count = wave_count as f64;
    NodeMeasurement {
        work_seconds: full_wave.work_seconds * wave_count,
        latency_seconds: full_wave.latency_seconds * wave_count,
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
                ConcreteWireType::Family { element, shape } => shape
                    .iter()
                    .try_fold(wire_bytes(element), |bytes, size| bytes.checked_mul(*size as u128))
                    .unwrap_or(u128::MAX),
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
    crt_depth: usize,
    /// Setup-time snapshot. Measurement and calibration must not reread process environment.
    vram_percent: u32,
    calibration_registry: GpuCalibrationRegistry,
    measurements: HashMap<[u8; 32], NodeMeasurement>,
    pending: HashMap<[u8; 32], PendingMeasurement>,
    collecting: bool,
}

impl GpuNodeMeasurementBackend {
    /// Creates a representative GPU measurement backend for validated IR nodes.
    pub fn new(
        backends: Vec<(GpuDcrtBackend, i32)>,
        harness: MeasurementHarnessConfig,
        crt_depth: usize,
    ) -> Self {
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
        Self::from_workers(workers, harness, crt_depth, vram_percent)
    }

    fn from_workers(
        workers: Vec<GpuMeasurementWorker>,
        harness: MeasurementHarnessConfig,
        crt_depth: usize,
        vram_percent: u32,
    ) -> Self {
        Self {
            workers,
            harness,
            crt_depth,
            vram_percent,
            calibration_registry: GpuCalibrationRegistry::new(),
            measurements: HashMap::new(),
            pending: HashMap::new(),
            collecting: true,
        }
    }

    /// Returns the setup-time profiles populated by measurement. Runtime may reuse the
    /// same registry and recompute widths from its then-current GPU residency.
    pub fn calibration_registry(&self) -> GpuCalibrationRegistry {
        self.calibration_registry.clone()
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
                operation_batch_size = request.operation_batch_size,
                measurement_key = ?request.key,
                "starting GPU node measurement"
            );
            let measurement = self.measure_fleet_request(&request)?;
            self.measurements.insert(request.key, measurement);
        }
        Ok(())
    }

    fn column_separable(kind: &NodeKind) -> bool {
        gpu_operation_is_column_separable(kind)
    }

    fn matrix_columns(wire_type: &ConcreteWireType) -> Option<usize> {
        match wire_type {
            ConcreteWireType::Matrix(matrix) |
            ConcreteWireType::SmallMatrix { matrix, .. } |
            ConcreteWireType::Preimage { matrix, .. } |
            ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix.columns),
            ConcreteWireType::Family { element, .. } => Self::matrix_columns(element),
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
            NodeKind::PreimageSample { .. } | NodeKind::FamilyPreimageSample { .. } => index < 2,
            NodeKind::Concat { axis: ConcatAxis::Diagonal } => true,
            _ => false,
        }
    }

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
                    if width != output.columns {
                        return Err(GpuMeasurementError(format!(
                            "node {:?} column concat range width {width} does not match output width {}",
                            node.id, output.columns
                        )));
                    }
                }
            }
            ConcatAxis::Rows => {
                if output_range.is_some() ||
                    inputs.iter().any(|input| input.columns != output.columns || input.rows == 0)
                {
                    return Err(GpuMeasurementError(format!(
                        "node {:?} row concat representative geometry is invalid",
                        node.id
                    )));
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
                operation_batch_size: request.operation_batch_size,
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
                operation_batch_size: request.operation_batch_size,
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
            let left_columns = local_end.div_ceil(right_columns);
            match left_wire {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage { matrix, .. } => {
                    matrix.columns = left_columns
                }
                _ => panic!("validated tensor LHS must be a matrix"),
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
                operation_batch_size: request.operation_batch_size,
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
                operation_batch_size: request.operation_batch_size,
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
            operation_batch_size: request.operation_batch_size,
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

    fn measure_fleet_request(
        &mut self,
        request: &PendingMeasurement,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
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
                operation_batch_size: request.operation_batch_size,
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
            self.workers.iter().zip(&initial_device_states).find(|(_, (_, contexts))| *contexts > 1)
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
            .iter()
            .map(|worker| Self::device_memory(worker.device_id))
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
        if request.preimage_sample {
            info!(
                scope = ?request.scope,
                node = request.id.0,
                work_seconds = measurement.work_seconds,
                latency_seconds = measurement.latency_seconds,
                workspace_bytes = measurement.workspace_bytes,
                "measured fleet-wide GPU preimage sampler"
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

        // A family preimage node performs the scalar preimage operation once for every
        // family member. Measurement keeps the structured family cardinality separately and
        // presents one leaf operation to the production backend.
        if let NodeKind::FamilyPreimageSample { matrix_type, max_coefficient_bound } = &kind {
            kind = NodeKind::PreimageSample {
                matrix_type: matrix_type.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
            };
            argument_types = argument_types.iter().map(family_leaf_type).cloned().collect();
            output_types = output_types
                .iter()
                .map(|wire_type| match family_leaf_type(wire_type) {
                    ConcreteWireType::Matrix(matrix) => ConcreteWireType::Preimage {
                        matrix: matrix.clone(),
                        max_coefficient_bound: BigInt::from(0),
                    },
                    ConcreteWireType::Preimage { matrix, max_coefficient_bound } |
                    ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } => {
                        ConcreteWireType::Preimage {
                            matrix: matrix.clone(),
                            max_coefficient_bound: max_coefficient_bound.clone(),
                        }
                    }
                    other => other.clone(),
                })
                .collect();
        }

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
                    let Some(ConcreteWireType::Family { shape, .. }) = argument_types.first_mut()
                    else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    // The runtime flattens the family in row-major order. Preserve every
                    // logical axis and reduce only the final, output-column-aligned axis.
                    if let Some(last) = shape.last_mut() {
                        *last = (*last).div_ceil(original_columns);
                    }
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
            let selected =
                selected && !(matches!(node.kind, NodeKind::PreimageSample { .. }) && index < 2);
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
        let preimage_target =
            if matches!(node.kind, NodeKind::PreimageSample { .. }) && phase != Some(true) {
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
                let outputs = Self::run_node(
                    &mut worker.backend,
                    &node,
                    bindings,
                    representative.operation_batch_size,
                    prepared,
                    representative.output_range.as_ref(),
                )?;
                outputs.iter().for_each(GpuMeasurementOutput::finish);
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
                        prepared.finish();
                        Ok((representative, prepared))
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;

        for _ in 0..harness.warm_up_iterations {
            let _ = Self::run_fleet_iteration(workers, scope, id, bindings, &prepared)?;
        }
        let baselines = workers
            .par_iter()
            .zip(prepared.par_iter())
            .map(|(worker, prepared)| {
                if prepared.is_none() {
                    return Ok(None);
                }
                begin_gpu_memory_measurement(worker.device_id).map(Some)
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
        prepared.finish();
        let baseline = begin_gpu_memory_measurement(worker.device_id)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(
            harness,
            &probe,
            representative.operation_batch_size,
            |representative_batch| {
                if operation_error.is_some() {
                    return;
                }
                match Self::run_node(
                    &mut worker.backend,
                    &node,
                    bindings,
                    representative_batch,
                    &prepared,
                    None,
                ) {
                    Ok(outputs) => {
                        outputs.iter().for_each(GpuMeasurementOutput::finish);
                    }
                    Err(error) => operation_error = Some(error),
                }
            },
        )
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        let mut measurement = measured.measurement;
        measurement.workspace_bytes = finish_gpu_memory_measurement(worker.device_id, baseline)?;
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
        fixed.finish();
        let baseline = begin_gpu_memory_measurement(worker.device_id)?;
        let scaled = Self::prepare(
            &mut worker.backend,
            &node,
            bindings,
            Some((&representative.fixed_arguments, false)),
        )?;
        let prepared = fixed.merge(scaled);
        let started = std::time::Instant::now();
        let outputs = Self::run_node(
            &mut worker.backend,
            &node,
            bindings,
            representative.operation_batch_size,
            &prepared,
            representative.output_range.as_ref(),
        )?;
        outputs.iter().for_each(GpuMeasurementOutput::finish);
        let latency_seconds = started.elapsed().as_secs_f64();
        let workspace_bytes = finish_gpu_memory_measurement(worker.device_id, baseline)?;
        Ok(NodeMeasurement { work_seconds: latency_seconds, latency_seconds, workspace_bytes })
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
        let evaluate_usize = |expression: &mxx_ir_core::IntExpr| {
            expression
                .evaluate(bindings)
                .map_err(|error| GpuMeasurementError(error.to_string()))?
                .to_usize()
                .ok_or_else(|| {
                    GpuMeasurementError("integer expression does not fit usize".to_owned())
                })
        };
        let backend_error =
            |error: <GpuDcrtBackend as Backend>::Error| GpuMeasurementError(error.to_string());
        let matrix_outputs = |outputs: Result<Vec<GpuFleetMatrix>, GpuMeasurementError>| {
            outputs.map(|outputs| outputs.into_iter().map(GpuMeasurementOutput::matrix).collect())
        };
        match node.kind {
            NodeKind::ConstantMatrix { value, .. } => {
                let full_ty = output_matrix_type()?;
                matrix_outputs(if let Some(range) = output_range {
                    let local_columns = range.end.checked_sub(range.start).ok_or_else(|| {
                        GpuMeasurementError("constant representative range is reversed".to_owned())
                    })?;
                    let local_ty = ConcreteMatrixType { columns: local_columns, ..full_ty.clone() };
                    (0..batch_size)
                        .map(|_| {
                            let parameter_owner = backend
                                .constant_matrix(&local_ty, &ConstantMatrix::Zero, bindings)
                                .map_err(backend_error)?;
                            let params = parameter_owner
                                .shards()
                                .first()
                                .expect("single-device estimator matrix needs one shard")
                                .value
                                .params()
                                .clone();
                            if matches!(value, ConstantMatrix::Zero) {
                                return Ok(parameter_owner);
                            }
                            drop(parameter_owner);
                            let local = match value {
                                ConstantMatrix::Zero => unreachable!(),
                                ConstantMatrix::Identity => GpuDCRTPolyMatrix::identity_columns(
                                    &params,
                                    full_ty.rows,
                                    range.start,
                                    local_columns,
                                ),
                                ConstantMatrix::UnitRow { index } => {
                                    let index = evaluate_usize(index)?;
                                    GpuDCRTPolyMatrix::unit_row_columns(
                                        &params,
                                        full_ty.columns,
                                        index,
                                        range.start,
                                        local_columns,
                                    )
                                }
                                ConstantMatrix::Gadget { small, .. } => {
                                    GpuDCRTPolyMatrix::gadget_columns(
                                        &params,
                                        full_ty.rows,
                                        *small,
                                        range.start,
                                        local_columns,
                                    )
                                }
                                _ => unreachable!("only range-capable constants receive a range"),
                            };
                            Ok(GpuFleetMatrix::from(local))
                        })
                        .collect()
                } else {
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .constant_matrix(&full_ty, value, bindings)
                                .map_err(backend_error)
                        })
                        .collect()
                })
            }
            NodeKind::GadgetTrapdoor { base, .. } => {
                let ty = output_matrix_type()?;
                let value = ConstantMatrix::Gadget { base: base.clone(), small: false };
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend.constant_matrix(&ty, &value, bindings).map_err(backend_error)
                        })
                        .collect(),
                )
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
                    .map(|_| backend.transpose(matrix(0)?).map_err(backend_error))
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
                            backend
                                .slice(matrix(0)?, rows.as_ref(), columns.as_ref())
                                .map_err(backend_error)
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
                matrix_outputs(if *axis == ConcatAxis::Diagonal {
                    if let Some(range) = output_range {
                        (0..batch_size)
                            .map(|_| {
                                backend
                                    .diagonal_concat_range_for_measurement(
                                        &inputs,
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
                    }
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
                        .map(|_| backend.sample_uniform(&ty, &range).map_err(backend_error))
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
                        .map(|_| backend.sample_uniform(&ty, &range).map_err(backend_error))
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
                            backend
                                .sample_gaussian(&ty, sigma, &max_coefficient_bound)
                                .map_err(backend_error)
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
                                backend.sample_hash(&ty, [0x53; 32], tag_prefix, *variant, None)
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(backend_error),
                    ),
                    (mxx_ir_core::node::HashVariant::Decomposed, Some(base), Some(count)) => (0..
                        batch_size)
                        .map(|_| {
                            backend
                                .sample_hash_small(
                                    &ty,
                                    [0x53; 32],
                                    tag_prefix,
                                    *variant,
                                    (base, count),
                                )
                                .map(GpuMeasurementOutput::SmallMatrix)
                                .map_err(backend_error)
                        })
                        .collect(),
                    (mxx_ir_core::node::HashVariant::SmallDecomposed, Some(base), Some(count)) => {
                        (0..batch_size)
                            .map(|_| {
                                backend
                                    .sample_hash_small(
                                        &ty,
                                        [0x53; 32],
                                        tag_prefix,
                                        *variant,
                                        (base, count),
                                    )
                                    .map(GpuMeasurementOutput::SmallMatrix)
                                    .map_err(backend_error)
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
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .sample_trapdoor(&ty, sigma, &gadget_base, digit_count)
                                .map(|(public, _)| public)
                                .map_err(backend_error)
                        })
                        .collect(),
                )
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
            NodeKind::GadgetDecompose { small, .. } => (0..batch_size)
                .map(|_| {
                    backend
                        .gadget_decompose(matrix(0)?, *small)
                        .map(GpuMeasurementOutput::SmallMatrix)
                        .map_err(backend_error)
                })
                .collect(),
            NodeKind::ExtractCoefficient { position, .. } => {
                let position = evaluate_usize(position)?;
                for _ in 0..batch_size {
                    backend.extract_coefficient(matrix(0)?, position).map_err(backend_error)?;
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
                    backend
                        .threshold_decode(matrix(0)?, &modulus, length)
                        .map_err(backend_error)?;
                }
                Ok(Vec::new())
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
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
                    Some(ConcreteWireType::Family { shape, .. }) => shape
                        .iter()
                        .try_fold(1usize, |count, size| count.checked_mul(*size))
                        .ok_or_else(|| {
                            GpuMeasurementError(
                                "packed coefficient family size overflows usize".to_owned(),
                            )
                        })?,
                    _ => {
                        return Err(GpuMeasurementError(
                            "packed coefficient input is not a family".to_owned(),
                        ));
                    }
                };
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend
                                .pack_polynomial_coefficients(
                                    &ty,
                                    &vec![false; count],
                                    coefficient_bits,
                                )
                                .map_err(backend_error)
                        })
                        .collect(),
                )
            }
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
            NodeKind::SubgraphCall(_) |
            NodeKind::SequentialLoop(_) |
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic { .. } |
            NodeKind::FamilySelectAxis { .. } |
            NodeKind::FamilyReindex { .. } |
            NodeKind::FamilyGather { .. } |
            NodeKind::ParallelGrid(_) |
            NodeKind::Select { .. } => Ok(Vec::new()),
            NodeKind::FamilyPreimageSample { .. } => Err(GpuMeasurementError(
                "family preimage sampling requires a representative lane".to_owned(),
            )),
        }
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
        if matches!(
            node.kind,
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
                NodeKind::FamilyGetDynamic { .. } |
                NodeKind::FamilySelectAxis { .. } |
                NodeKind::FamilyReindex { .. } |
                NodeKind::FamilyGather { .. } |
                NodeKind::ParallelGrid(_) |
                NodeKind::Select { .. }
        ) {
            return Ok(NodeMeasurement::default());
        }
        let operation_batch_size = operation_batch_size(node)?;
        if operation_batch_size == 0 {
            return Ok(NodeMeasurement::default());
        }
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
            key: measurement_key,
            scope: node.scope.clone(),
            id: node.id,
            kind: node.kind.clone(),
            concrete_argument_types: node.concrete_argument_types.clone(),
            concrete_output_types: node.concrete_output_types.clone(),
            bindings: bindings.clone(),
            preimage_sample: matches!(
                node.kind,
                NodeKind::PreimageSample { .. } | NodeKind::FamilyPreimageSample { .. }
            ),
            operation_batch_size,
        });
        Ok(NodeMeasurement::default())
    }

    fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
        match wire_type {
            ConcreteWireType::Matrix(matrix) => matrix_bytes(matrix, self.crt_depth),
            ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
            ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                compact_matrix_bytes(matrix, max_coefficient_bound)
            }
            ConcreteWireType::Trapdoor { matrix, .. } => matrix_bytes(matrix, self.crt_depth),
            ConcreteWireType::Family { element, shape } => {
                shape.iter().fold(self.persistent_bytes(element), |bytes, size| {
                    bytes.saturating_mul(u64::try_from(*size).unwrap_or(u64::MAX))
                })
            }
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
        if matches!(kind, NodeKind::ParallelGrid(_)) &&
            matches!(
                wire_type,
                ConcreteWireType::Family { element, .. }
                    if ArtifactType::from_wire_type(element).is_some()
            )
        {
            // Runtime stages each artifact-compatible lane after its bounded grid wave. The
            // live lane values are accounted in the child peak, not as one resident family.
            return 0;
        }
        if matches!(kind, NodeKind::Input { artifact: Some(_), .. }) &&
            matches!(wire_type, ConcreteWireType::Family { .. })
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
        matches!(kind, NodeKind::Input { .. })
            .then(|| self.persistent_storage_bytes_for_node(kind, wire_type))
            .unwrap_or(0)
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
        GpuNodeMeasurementBackend, PendingMeasurement, PreparedMeasurement, aggregate_fleet_wave,
        compact_matrix_bytes, extrapolate_fleet_waves, matrix_bytes,
        require_exclusive_measurement_context,
    };
    use crate::{MeasurementNode, NodeMeasurement};
    use mxx_ir_core::{
        FrozenGraphScopeId, IntExpr, ParamEnv, RealExpr,
        node::{ConstantMatrix, HashVariant, IndexRange, MatrixBinaryOp, NodeKind},
        types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId},
    };
    use mxx_runtime::gpu_calibration::{GpuColumnWidths, gpu_capped_waterfill_columns};
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
            1,
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
    fn fleet_wave_extrapolation_uses_conservative_full_waves() {
        let full_wave =
            NodeMeasurement { work_seconds: 38.0, latency_seconds: 39.0, workspace_bytes: 56 };
        let measurement = extrapolate_fleet_waves(&full_wave, 7);

        assert_eq!(measurement.work_seconds, 266.0);
        assert_eq!(measurement.latency_seconds, 273.0);
        assert_eq!(measurement.workspace_bytes, 56);
    }

    #[test]
    fn fleet_wave_sums_work_and_workspace_and_uses_measured_wall_latency() {
        let measurement = aggregate_fleet_wave(
            [
                NodeMeasurement { work_seconds: 2.0, latency_seconds: 2.0, workspace_bytes: 30 },
                NodeMeasurement { work_seconds: 3.0, latency_seconds: 3.0, workspace_bytes: 20 },
            ],
            3.25,
        );

        assert_eq!(measurement.work_seconds, 5.0);
        assert_eq!(measurement.latency_seconds, 3.25);
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
        assert!(GpuNodeMeasurementBackend::column_separable(&NodeKind::RingAutomorphism {
            index: IntExpr::constant(3),
        }));
        assert!(GpuNodeMeasurementBackend::column_separable(&NodeKind::Concat {
            axis: mxx_ir_core::node::ConcatAxis::Diagonal,
        }));
        assert!(!GpuNodeMeasurementBackend::column_separable(&NodeKind::ConstantMatrix {
            matrix_type,
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
            operation_batch_size: 1,
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
            key: [0; 32],
            scope,
            id: NodeId(1),
            kind,
            concrete_argument_types: vec![matrix(2, 8), matrix(3, 13), matrix(5, 21)],
            concrete_output_types: vec![matrix(10, 42)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
            operation_batch_size: 1,
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
            tag_expressions: vec![IntExpr::LoopIndex(0)],
            tag_decimal_expressions: Vec::new(),
            tag_u64_le_expressions: Vec::new(),
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
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(7),
            kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            concrete_argument_types: vec![matrix(2, 1), matrix(1, 176)],
            concrete_output_types: vec![matrix(2, 176)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
            operation_batch_size: 1,
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
            operation_batch_size: 1,
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
    fn gadget_trapdoor_measurement_uses_column_representative() {
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
                matrix: concrete,
                sigma: RealExpr::from_integer(4),
                gadget_base: BigInt::from(4),
                digit_count: 5,
                preimage_max_coefficient_bound: BigInt::from(0),
            }],
        };

        assert!(GpuNodeMeasurementBackend::column_separable(&kind));
        let (representative, _, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 4);
        let NodeKind::GadgetTrapdoor { matrix_type, .. } = representative else {
            panic!("gadget trapdoor representative kind");
        };
        assert_eq!(matrix_type.columns, IntExpr::constant(4));
        assert_eq!(outputs[0].matrix_type().unwrap().columns, 4);
        assert_eq!(scale, 2.0);
        assert_eq!(remainder_columns, Some(2));
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
            tag_expressions: vec![tag_expression],
            tag_decimal_expressions: Vec::new(),
            tag_u64_le_expressions: Vec::new(),
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
            tag_expressions: Vec::new(),
            tag_decimal_expressions: Vec::new(),
            tag_u64_le_expressions: Vec::new(),
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
            operation_batch_size: 1,
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
        assert_eq!(output.columns, 45);
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
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(1),
            kind: NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Columns },
            concrete_argument_types: vec![matrix(2), matrix(100), matrix(3)],
            concrete_output_types: vec![matrix(105)],
            bindings: ParamEnv::default(),
            preimage_sample: false,
            operation_batch_size: 1,
        };

        let representative = GpuNodeMeasurementBackend::representative_at(&request, 50, 4);
        assert_eq!(representative.concrete_argument_types.len(), 1);
        assert_eq!(representative.concrete_argument_types[0].matrix_type().unwrap().columns, 4);
        assert_eq!(representative.measured_columns(), Some(4));

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
        assert_eq!(crossing.measured_columns(), Some(4));

        let pilot = GpuNodeMeasurementBackend::calibration_representative(&request);
        assert_eq!(pilot.measured_columns(), Some(2));
        assert_eq!(pilot.concrete_argument_types.len(), 2);
        assert_eq!(
            pilot
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
            operation_batch_size: 1,
        };
        let assignments =
            gpu_capped_waterfill_columns(GpuColumnWidths { gpu0: 1154, nonzero: Some(1154) }, 2, 2)
                .unwrap();
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

    #[test]
    fn family_preimage_conversion_preserves_fixed_operand_ownership() {
        let matrix = |rows, columns| ConcreteMatrixType {
            rows,
            columns,
            ring_dimension: 8,
            modulus: BigInt::from(257u16),
        };
        let family =
            |element, shape| ConcreteWireType::Family { element: Box::new(element), shape };
        let request = PendingMeasurement {
            key: [0; 32],
            scope: FrozenGraphScopeId::Root,
            id: NodeId(8),
            kind: NodeKind::FamilyPreimageSample {
                matrix_type: MatrixType {
                    rows: IntExpr::constant(3),
                    columns: IntExpr::constant(80),
                    ring_dimension: IntExpr::constant(8),
                    modulus: IntExpr::constant(257),
                },
                max_coefficient_bound: IntExpr::constant(100),
            },
            concrete_argument_types: vec![
                family(ConcreteWireType::Matrix(matrix(2, 3)), vec![2]),
                family(
                    ConcreteWireType::Trapdoor {
                        matrix: matrix(2, 3),
                        sigma: RealExpr::from_integer(4),
                        gadget_base: BigInt::from(2),
                        digit_count: 2,
                        preimage_max_coefficient_bound: BigInt::from(100),
                    },
                    vec![2],
                ),
                family(ConcreteWireType::Matrix(matrix(2, 80)), vec![2, 4]),
            ],
            concrete_output_types: vec![family(
                ConcreteWireType::Preimage {
                    matrix: matrix(3, 80),
                    max_coefficient_bound: BigInt::from(100),
                },
                vec![2, 4],
            )],
            bindings: ParamEnv::default(),
            preimage_sample: true,
            operation_batch_size: 8,
        };

        let representative = GpuNodeMeasurementBackend::representative_at(&request, 0, 4);
        assert!(matches!(representative.kind, NodeKind::PreimageSample { .. }));
        assert_eq!(representative.fixed_arguments, vec![true, true, false]);
        assert!(
            representative
                .concrete_argument_types
                .iter()
                .all(|ty| { !matches!(ty, ConcreteWireType::Family { .. }) })
        );
        assert_eq!(representative.concrete_argument_types[2].matrix_type().unwrap().columns, 4);
        assert_eq!(representative.concrete_output_types[0].matrix_type().unwrap().columns, 4);
        assert_eq!(representative.operation_batch_size, 8);
    }
}
