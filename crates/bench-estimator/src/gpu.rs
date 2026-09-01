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
    ParamEnv, encoding,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::{
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix, poly::dcrt::gpu::gpu_memory_info,
    sampler::trapdoor::gpu::GpuDCRTTrapdoor,
};
use mxx_runtime::{
    Backend,
    backend::{
        IndexRange, MatrixMulAccumulateRequest, PreimageRequest, SampleRange,
        poly::gpu::GpuDcrtBackend,
    },
};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{Arc, Mutex},
};
use tracing::info;

// Oversized column-separable operations are measured as bounded multi-column waves. The logical
// output size remains unchanged in the estimator's persistent-memory model.
const REPRESENTATIVE_MATRIX_BYTES: u64 = 2 * 1024 * 1024 * 1024;

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

impl MemoryProbe for GpuMemoryProbe {
    type Error = GpuMeasurementError;

    fn current_bytes(&self) -> Result<u64, Self::Error> {
        let memory = gpu_memory_info(self.device_id).map_err(GpuMeasurementError)?;
        u64::try_from(memory.total.saturating_sub(memory.free))
            .map_err(|_| GpuMeasurementError("GPU memory usage exceeds u64".to_owned()))
    }
}

struct PreparedMeasurement {
    arguments: Vec<Option<Arc<GpuDCRTPolyMatrix>>>,
    preimage_trapdoor: Option<(GpuDCRTPolyMatrix, GpuDCRTTrapdoor, f64, BigInt, usize, BigInt)>,
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
    scale: f64,
    remainder: Option<RepresentativeMeasurement>,
    preimage_sample: bool,
    operation_batch_size: usize,
}

struct RepresentativeMeasurement {
    kind: NodeKind,
    concrete_argument_types: Vec<ConcreteWireType>,
    concrete_output_types: Vec<ConcreteWireType>,
    operation_batch_size: usize,
}

fn family_leaf_type(wire_type: &ConcreteWireType) -> &ConcreteWireType {
    match wire_type {
        ConcreteWireType::Family { element, .. } => family_leaf_type(element),
        _ => wire_type,
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
    if !matches!(element.as_ref(), ConcreteWireType::Preimage(_)) {
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

fn extrapolate_column_waves(
    full_wave: &NodeMeasurement,
    full_wave_count: f64,
    remainder_wave: Option<&NodeMeasurement>,
) -> NodeMeasurement {
    NodeMeasurement {
        work_seconds: full_wave.work_seconds * full_wave_count +
            remainder_wave.map_or(0.0, |value| value.work_seconds),
        latency_seconds: full_wave.latency_seconds * full_wave_count +
            remainder_wave.map_or(0.0, |value| value.latency_seconds),
        workspace_bytes: remainder_wave.map_or(full_wave.workspace_bytes, |value| {
            full_wave.workspace_bytes.max(value.workspace_bytes)
        }),
    }
}

impl PendingMeasurement {
    fn representative_bytes(&self) -> u128 {
        fn wire_bytes(wire_type: &ConcreteWireType) -> u128 {
            match wire_type {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                    (matrix.rows as u128)
                        .saturating_mul(matrix.columns as u128)
                        .saturating_mul(matrix.ring_dimension as u128)
                        .saturating_mul(8)
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
    column_wave_size: usize,
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
        column_wave_size: usize,
    ) -> Self {
        assert!(!backends.is_empty(), "GPU measurement requires at least one backend");
        assert!(column_wave_size > 0, "GPU measurement column wave must be nonzero");
        let workers = backends
            .into_iter()
            .map(|(backend, device_id)| GpuMeasurementWorker { backend, device_id })
            .collect();
        Self {
            workers,
            harness,
            crt_depth,
            column_wave_size,
            measurements: HashMap::new(),
            pending: HashMap::new(),
            collecting: true,
        }
    }

    /// Sets the columns executed together by one physical GPU for subsequent estimator passes.
    pub fn set_column_wave_size(&mut self, column_wave_size: usize) {
        assert!(column_wave_size > 0, "GPU measurement column wave must be nonzero");
        self.column_wave_size = column_wave_size;
    }

    /// Measures all shapes collected by prior estimator passes, distributing each unique shape
    /// to exactly one GPU. Subsequent estimator passes read the completed measurement cache.
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
        let requests = Mutex::new(VecDeque::from(requests));
        info!(
            measurement_workers = self.workers.len(),
            measurement_count, "measuring collected GPU node shapes in parallel"
        );
        let harness = &self.harness;
        let measured = self
            .workers
            .par_iter_mut()
            .map(|worker| {
                let mut completed = Vec::new();
                let mut representative_work_seconds = 0.0;
                loop {
                    let Some(request) = requests
                        .lock()
                        .expect("GPU measurement request queue poisoned")
                        .pop_front()
                    else {
                        break;
                    };
                    let measurement = Self::measure_request(worker, harness, &request)?;
                    representative_work_seconds += measurement.work_seconds / request.scale;
                    completed.push((request.key, measurement));
                }
                info!(
                    device_id = worker.device_id,
                    measurement_count = completed.len(),
                    representative_work_seconds,
                    "completed GPU measurement worker"
                );
                Ok(completed)
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
        self.measurements.extend(measured.into_iter().flatten());
        Ok(())
    }

    fn measurement_key(
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        column_wave_size: usize,
    ) -> Result<[u8; 32], GpuMeasurementError> {
        #[derive(Serialize)]
        struct MeasurementKey<'a> {
            kind: &'a NodeKind,
            concrete_argument_types: &'a [ConcreteWireType],
            concrete_output_types: &'a [ConcreteWireType],
            bindings: &'a ParamEnv,
            column_wave_size: usize,
        }

        // Loop-index values select different logical family elements, but they do not change the
        // GPU operation shape. Keeping them in the key forces the same lookup-body operation to be
        // measured once per table entry. Compile-time integer and real parameters remain keyed.
        let shape_bindings = ParamEnv {
            integers: bindings.integers.clone(),
            reals: bindings.reals.clone(),
            loop_indices: Default::default(),
        };
        let mut shape_kind = node.kind.clone();
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
                *expression = mxx_ir_core::IntExpr::constant(0);
            }
        }

        encoding::hash_canonical(&MeasurementKey {
            kind: &shape_kind,
            concrete_argument_types: &node.concrete_argument_types,
            concrete_output_types: &node.concrete_output_types,
            bindings: &shape_bindings,
            column_wave_size,
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn representative_node<'a>(
        node: &'a MeasurementNode<'a>,
        crt_depth: usize,
        column_wave_size: usize,
    ) -> (NodeKind, Vec<ConcreteWireType>, Vec<ConcreteWireType>, f64, Option<usize>) {
        assert!(column_wave_size > 0, "GPU measurement column wave must be nonzero");
        let capped_columns = |matrix: &ConcreteMatrixType| {
            (matrix.columns > 1 && matrix_bytes(matrix, crt_depth) > REPRESENTATIVE_MATRIX_BYTES)
                .then(|| {
                    let representative_columns = matrix.columns.min(column_wave_size);
                    let full_waves = matrix.columns / representative_columns;
                    let remainder_columns = matrix.columns % representative_columns;
                    (
                        representative_columns,
                        full_waves as f64,
                        (remainder_columns > 0).then_some(remainder_columns),
                    )
                })
        };
        let capped_rows = |matrix: &ConcreteMatrixType| {
            (matrix.rows > 1 && matrix_bytes(matrix, crt_depth) > REPRESENTATIVE_MATRIX_BYTES)
                .then_some((1, matrix.rows as f64))
        };
        let mut kind = node.kind.clone();
        let mut argument_types = node.concrete_argument_types.clone();
        let mut output_types = node.concrete_output_types.clone();
        let mut scale = 1.0;
        let mut remainder_columns = None;

        // Family preimage sampling performs the same matrix operation as the scalar
        // preimage sampler once one representative family lane is selected. GPU cost
        // depends on that lane's matrix shapes, not on the family container itself.
        if let NodeKind::FamilyPreimageSample { matrix_type, max_coefficient_bound } = &kind {
            kind = NodeKind::PreimageSample {
                matrix_type: matrix_type.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
            };
            argument_types = argument_types
                .iter()
                .map(|wire_type| family_leaf_type(wire_type).clone())
                .collect();
            output_types = output_types
                .iter()
                .map(|wire_type| match family_leaf_type(wire_type) {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        ConcreteWireType::Preimage(matrix.clone())
                    }
                    other => other.clone(),
                })
                .collect();
        }

        match &mut kind {
            NodeKind::ConstantMatrix { matrix_type, .. } |
            NodeKind::GadgetTrapdoor { matrix_type, .. } |
            NodeKind::TrapdoorSample { matrix_type, .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                } else if let Some((representative_rows, row_scale)) = capped_rows(output) {
                    output.rows = representative_rows;
                    matrix_type.rows = mxx_ir_core::IntExpr::constant(representative_rows);
                    scale = row_scale;
                }
            }
            NodeKind::Slice { rows, columns } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                let Some(input) =
                    argument_types.first_mut().and_then(|wire_type| match wire_type {
                        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                            Some(matrix)
                        }
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
                } else if matrix_bytes(input, crt_depth) > REPRESENTATIVE_MATRIX_BYTES &&
                    matrix_bytes(output, crt_depth) <= REPRESENTATIVE_MATRIX_BYTES
                {
                    input.rows = output.rows;
                    input.columns = output.columns;
                    if let Some(range) = rows {
                        range.start = mxx_ir_core::IntExpr::constant(0);
                        range.end = mxx_ir_core::IntExpr::constant(output.rows);
                    }
                    if let Some(range) = columns {
                        range.start = mxx_ir_core::IntExpr::constant(0);
                        range.end = mxx_ir_core::IntExpr::constant(output.columns);
                    }
                }
            }
            NodeKind::Transpose => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
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
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                            Some(matrix)
                        }
                        _ => None,
                    }) else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    let Some(right) = (match right_wire {
                        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                            Some(matrix)
                        }
                        _ => None,
                    }) else {
                        return (kind, argument_types, output_types, scale, remainder_columns);
                    };
                    let mut representative_columns = (1usize, 1usize);
                    for left_columns in 1..=left.columns.min(column_wave_size) {
                        let right_columns = right.columns.min(column_wave_size / left_columns);
                        if left_columns * right_columns >
                            representative_columns.0 * representative_columns.1
                        {
                            representative_columns = (left_columns, right_columns);
                        }
                    }
                    left.columns = representative_columns.0;
                    right.columns = representative_columns.1;
                    output.columns = representative_columns.0 * representative_columns.1;
                    scale = original_output_columns.div_ceil(output.columns) as f64;
                }
            }
            NodeKind::Concat { axis: ConcatAxis::Rows } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
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
            NodeKind::Concat { axis: ConcatAxis::Columns | ConcatAxis::Diagonal } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if capped_columns(output).is_some() {
                    let original_columns = output.columns;
                    let minimum_columns = argument_types.len();
                    let target_columns =
                        original_columns.min(column_wave_size.max(minimum_columns));
                    let mut representative_columns = 0usize;
                    let mut remaining_columns = target_columns.saturating_sub(minimum_columns);
                    for wire_type in &mut argument_types {
                        let Some(input) = (match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
                            _ => None,
                        }) else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        let extra_columns = remaining_columns.min(input.columns.saturating_sub(1));
                        input.columns = 1 + extra_columns;
                        remaining_columns -= extra_columns;
                        representative_columns =
                            representative_columns.saturating_add(input.columns);
                    }
                    output.columns = representative_columns;
                    scale = original_columns.div_ceil(representative_columns.max(1)) as f64;
                }
            }
            NodeKind::GadgetDecompose { .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
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
            NodeKind::MatrixScale { .. } | NodeKind::MatrixNegate => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
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
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                                            ConcreteWireType::Preimage(matrix) => Some(matrix),
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
                        let Some(rhs) =
                            argument_types.get_mut(1).and_then(|wire_type| match wire_type {
                                ConcreteWireType::Matrix(matrix) |
                                ConcreteWireType::Preimage(matrix) => Some(matrix),
                                _ => None,
                            })
                        else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        if let Some((representative_columns, column_scale, column_remainder)) =
                            capped_columns(rhs)
                        {
                            rhs.columns = representative_columns;
                            output.columns = representative_columns;
                            scale = column_scale;
                            remainder_columns = column_remainder;
                        }
                    }
                }
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let representative = coefficients.iter().enumerate().find_map(|(product, _)| {
                    argument_types.get(2 * product + 1).and_then(|wire_type| match wire_type {
                        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                            capped_columns(matrix)
                        }
                        _ => None,
                    })
                });
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output).or(representative)
                {
                    for product in 0..coefficients.len() {
                        let Some(rhs) =
                            argument_types.get_mut(2 * product + 1).and_then(|wire_type| {
                                match wire_type {
                                    ConcreteWireType::Matrix(matrix) |
                                    ConcreteWireType::Preimage(matrix) => Some(matrix),
                                    _ => None,
                                }
                            })
                        else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        rhs.columns = representative_columns;
                    }
                    if *has_bias {
                        let Some(bias) =
                            argument_types.get_mut(2 * coefficients.len()).and_then(|wire_type| {
                                match wire_type {
                                    ConcreteWireType::Matrix(matrix) |
                                    ConcreteWireType::Preimage(matrix) => Some(matrix),
                                    _ => None,
                                }
                            })
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
            NodeKind::PreimageSample { matrix_type, .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
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
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
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
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
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
                        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                            Some(matrix)
                        }
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
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output)
                {
                    let original_columns = output.columns;
                    match argument_types.first_mut() {
                        Some(ConcreteWireType::Family { shape, .. }) => {
                            // The pack input is flattened by the runtime.  For a rank-N family,
                            // preserve its axis structure and scale only the axis corresponding
                            // to the output columns when it is present.
                            if let Some(last) = shape.last_mut() {
                                *last = (*last).div_ceil(original_columns);
                            }
                        }
                        _ => {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        }
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
    ) -> Result<PreparedMeasurement, GpuMeasurementError> {
        let arguments = node
            .concrete_argument_types
            .iter()
            .map(|wire_type| match wire_type {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => backend
                    .constant_matrix(&matrix, &ConstantMatrix::Zero, bindings)
                    .map(|matrix| Some(Arc::new(matrix)))
                    .map_err(|error| GpuMeasurementError(error.to_string())),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let preimage_trapdoor = if matches!(node.kind, NodeKind::PreimageSample { .. }) {
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
            public.wait_until_ready();
            Some((
                public,
                trapdoor,
                sigma,
                gadget_base.clone(),
                *digit_count,
                // Benchmark one production sampler draw instead of including a random number of
                // rejection retries. Centered coefficients are at most q / 2, so using the ring
                // modulus as the cutoff preserves the ordinary bound-check path while accepting
                // the first draw deterministically.
                matrix.modulus.clone(),
            ))
        } else {
            None
        };
        Ok(PreparedMeasurement { arguments, preimage_trapdoor })
    }

    fn measure_request(
        worker: &mut GpuMeasurementWorker,
        harness: &MeasurementHarnessConfig,
        request: &PendingMeasurement,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        let full_wave = RepresentativeMeasurement {
            kind: request.kind.clone(),
            concrete_argument_types: request.concrete_argument_types.clone(),
            concrete_output_types: request.concrete_output_types.clone(),
            operation_batch_size: request.operation_batch_size,
        };
        let measured = Self::measure_representative(
            worker,
            harness,
            &request.scope,
            request.id,
            &request.bindings,
            &full_wave,
        )?;
        let remainder = request
            .remainder
            .as_ref()
            .map(|remainder| {
                Self::measure_representative(
                    worker,
                    harness,
                    &request.scope,
                    request.id,
                    &request.bindings,
                    remainder,
                )
            })
            .transpose()?;
        let measurement = extrapolate_column_waves(&measured, request.scale, remainder.as_ref());
        if request.scale > 1.0 || request.remainder.is_some() {
            info!(
                device_id = worker.device_id,
                scope = ?request.scope,
                node = request.id.0,
                full_wave_count = request.scale,
                full_wave = ?measured,
                remainder_wave = ?remainder,
                extrapolated = ?measurement,
                "extrapolated GPU column-wave measurement"
            );
        }
        if request.preimage_sample {
            info!(
                device_id = worker.device_id,
                scope = ?request.scope,
                node = request.id.0,
                work_seconds = measurement.work_seconds,
                latency_seconds = measurement.latency_seconds,
                workspace_bytes = measurement.workspace_bytes,
                "measured representative GPU preimage sampler"
            );
        }
        Ok(measurement)
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
        let prepared = Self::prepare(&mut worker.backend, &node, bindings)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(
            harness,
            &probe,
            representative.operation_batch_size,
            |operation_batch| {
                if operation_error.is_some() {
                    return;
                }
                match Self::run_node(
                    &mut worker.backend,
                    &node,
                    bindings,
                    operation_batch,
                    &prepared,
                ) {
                    Ok(outputs) => outputs.iter().for_each(GpuDCRTPolyMatrix::wait_until_ready),
                    Err(error) => operation_error = Some(error),
                }
            },
        )
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        Ok(measured.measurement)
    }

    fn run_node(
        backend: &mut GpuDcrtBackend,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        batch_size: usize,
        prepared: &PreparedMeasurement,
    ) -> Result<Vec<GpuDCRTPolyMatrix>, GpuMeasurementError> {
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
        let output_matrix_type = || {
            node.concrete_output_types
                .iter()
                .find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix.clone())
                    }
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
        match node.kind {
            NodeKind::ConstantMatrix { value, .. } => {
                let ty = output_matrix_type()?;
                (0..batch_size)
                    .map(|_| backend.constant_matrix(&ty, value, bindings).map_err(backend_error))
                    .collect()
            }
            NodeKind::GadgetTrapdoor { base, .. } => {
                let ty = output_matrix_type()?;
                let value = ConstantMatrix::Gadget { base: base.clone(), small: false };
                (0..batch_size)
                    .map(|_| backend.constant_matrix(&ty, &value, bindings).map_err(backend_error))
                    .collect()
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
            }
            NodeKind::ApplyPreimage => {
                let inputs = (0..batch_size)
                    .map(|_| Ok((matrix_arc(0)?, matrix_arc(1)?)))
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                backend.multiply_batch(inputs).map_err(backend_error)
            }
            NodeKind::PreimageBinary(operation) => {
                let inputs = (0..batch_size)
                    .map(|_| Ok((matrix_arc(0)?, matrix_arc(1)?)))
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                match operation {
                    mxx_ir_core::node::PreimageBinaryOp::Add => backend.add_batch(inputs),
                    mxx_ir_core::node::PreimageBinaryOp::RightMultiplyExact |
                    mxx_ir_core::node::PreimageBinaryOp::ComposeExactDecomposition => {
                        backend.multiply_batch(inputs)
                    }
                }
                .map_err(backend_error)
            }
            NodeKind::PreimageConcatColumns => (0..batch_size)
                .map(|_| {
                    let inputs = prepared
                        .arguments
                        .iter()
                        .map(|value| {
                            value.as_deref().ok_or_else(|| {
                                GpuMeasurementError("preimage concat input is not a matrix".into())
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    backend.concat(&inputs, ConcatAxis::Columns).map_err(backend_error)
                })
                .collect(),
            NodeKind::DecompositionEntry { row, column } => {
                let row = evaluate_usize(row)?;
                let column = evaluate_usize(column)?;
                let rows = IndexRange { start: row, end: row + 1 };
                let columns = IndexRange { start: column, end: column + 1 };
                (0..batch_size)
                    .map(|_| {
                        backend
                            .slice(matrix(0)?, Some(&rows), Some(&columns))
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
                backend.matrix_mul_accumulate_batch(requests).map_err(backend_error)
            }
            NodeKind::MatrixNegate => backend
                .negate_batch(
                    (0..batch_size).map(|_| matrix_arc(0)).collect::<Result<Vec<_>, _>>()?,
                )
                .map_err(backend_error),
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
            }
            NodeKind::Transpose => (0..batch_size)
                .map(|_| backend.transpose(matrix(0)?).map_err(backend_error))
                .collect(),
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
                (0..batch_size)
                    .map(|_| {
                        backend
                            .slice(matrix(0)?, rows.as_ref(), columns.as_ref())
                            .map_err(backend_error)
                    })
                    .collect()
            }
            NodeKind::Tensor => (0..batch_size)
                .map(|_| backend.tensor(matrix(0)?, matrix(1)?).map_err(backend_error))
                .collect(),
            NodeKind::Concat { axis } => {
                let inputs = prepared
                    .arguments
                    .iter()
                    .map(|value| {
                        value.as_ref().map(Arc::as_ref).ok_or_else(|| {
                            GpuMeasurementError("concat argument is not a matrix".to_owned())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (0..batch_size)
                    .map(|_| backend.concat(&inputs, *axis).map_err(backend_error))
                    .collect()
            }
            NodeKind::UniformResidueSample { .. } => {
                let ty = output_matrix_type()?;
                let range = SampleRange {
                    minimum: BigInt::from(0),
                    maximum: &ty.modulus - BigInt::from(1),
                };
                (0..batch_size)
                    .map(|_| backend.sample_uniform(&ty, &range).map_err(backend_error))
                    .collect()
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
                (0..batch_size)
                    .map(|_| backend.sample_uniform(&ty, &range).map_err(backend_error))
                    .collect()
            }
            NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                let ty = output_matrix_type()?;
                let sigma = sigma
                    .evaluate_f64(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let max_coefficient_bound = max_coefficient_bound
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                (0..batch_size)
                    .map(|_| {
                        backend
                            .sample_gaussian(&ty, sigma, &max_coefficient_bound)
                            .map_err(backend_error)
                    })
                    .collect()
            }
            NodeKind::HashSample { tag_prefix, .. } => {
                let ty = output_matrix_type()?;
                (0..batch_size)
                    .map(|_| {
                        backend
                            .sample_hash(
                                &ty,
                                [0x53; 32],
                                tag_prefix,
                                mxx_ir_core::node::HashVariant::Plain,
                                None,
                            )
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
                (0..batch_size)
                    .map(|_| {
                        backend
                            .sample_trapdoor(&ty, sigma, &gadget_base, digit_count)
                            .map(|(public, _)| public)
                            .map_err(backend_error)
                    })
                    .collect()
            }
            NodeKind::PreimageSample { .. } => {
                let ty = output_matrix_type()?;
                let (public, trapdoor, sigma, gadget_base, digit_count, max_coefficient_bound) =
                    prepared.preimage_trapdoor.as_ref().ok_or_else(|| {
                        GpuMeasurementError("missing prepared trapdoor".to_owned())
                    })?;
                let target = matrix_arc(2)?;
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
                        )
                        .map(|output| vec![output])
                        .map_err(backend_error)
                } else {
                    backend
                        .sample_preimage_batch(
                            (0..batch_size)
                                .map(|_| PreimageRequest {
                                    matrix_type: ty.clone(),
                                    sigma: *sigma,
                                    gadget_base: gadget_base.clone(),
                                    digit_count: *digit_count,
                                    max_coefficient_bound: max_coefficient_bound.clone(),
                                    trapdoor: Arc::new(trapdoor.clone()),
                                    public: Arc::new(public.clone()),
                                    target: target.clone(),
                                })
                                .collect(),
                        )
                        .map_err(backend_error)
                }
            }
            NodeKind::GadgetDecompose { small, .. } => (0..batch_size)
                .map(|_| backend.gadget_decompose(matrix(0)?, *small).map_err(backend_error))
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
                (0..batch_size)
                    .map(|_| {
                        let identity = backend
                            .constant_matrix(
                                &ty,
                                &mxx_ir_core::node::ConstantMatrix::Identity,
                                bindings,
                            )
                            .map_err(backend_error)?;
                        backend.scale_integer(&identity, &BigInt::from(0)).map_err(backend_error)
                    })
                    .collect()
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
                (0..batch_size)
                    .map(|_| {
                        backend
                            .crt_recompose(&levels, &plaintext_moduli, &reconstruction_coefficients)
                            .map_err(backend_error)
                    })
                    .collect()
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
                    .collect()
            }
            NodeKind::Input { .. } |
            NodeKind::ConstantInt(_) |
            NodeKind::EvaluateInt(_) |
            NodeKind::ConstantReal(_) |
            NodeKind::ConstantBool(_) |
            NodeKind::MaterializePreimageExact |
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
        // A family with a zero extent is a valid runtime value, but it launches
        // no preimage-sampling work and therefore needs no representative GPU
        // measurement or cache entry.
        if operation_batch_size == 0 {
            return Ok(NodeMeasurement::default());
        }
        let (
            representative_kind,
            representative_argument_types,
            representative_output_types,
            scale,
            remainder_columns,
        ) = Self::representative_node(node, self.crt_depth, self.column_wave_size);
        let representative_node = MeasurementNode {
            scope: node.scope,
            id: node.id,
            kind: &representative_kind,
            arguments: node.arguments,
            argument_kinds: node.argument_kinds,
            argument_types: node.argument_types,
            output_types: node.output_types,
            concrete_argument_types: representative_argument_types,
            concrete_output_types: representative_output_types,
        };
        let measurement_key = Self::measurement_key(node, bindings, self.column_wave_size)?;
        if let Some(measurement) = self.measurements.get(&measurement_key) {
            return Ok(measurement.clone());
        }
        if !self.collecting {
            return Err(GpuMeasurementError(format!(
                "GPU node shape at {:?} node {:?} was not collected before measurement",
                node.scope, node.id
            )));
        }
        let remainder = remainder_columns.map(|columns| {
            let (kind, concrete_argument_types, concrete_output_types, _, _) =
                Self::representative_node(node, self.crt_depth, columns);
            RepresentativeMeasurement {
                kind,
                concrete_argument_types,
                concrete_output_types,
                operation_batch_size,
            }
        });
        self.pending.entry(measurement_key).or_insert_with(|| PendingMeasurement {
            key: measurement_key,
            scope: node.scope.clone(),
            id: node.id,
            kind: representative_node.kind.clone(),
            concrete_argument_types: representative_node.concrete_argument_types,
            concrete_output_types: representative_node.concrete_output_types,
            bindings: bindings.clone(),
            scale,
            remainder,
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
            ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                matrix_bytes(matrix, self.crt_depth)
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
}

fn matrix_bytes(matrix: &ConcreteMatrixType, crt_depth: usize) -> u64 {
    u64::try_from(matrix.rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(matrix.columns).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(matrix.ring_dimension).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(crt_depth).unwrap_or(u64::MAX))
        .saturating_mul(8)
}

#[cfg(test)]
mod tests {
    use super::{
        GpuNodeMeasurementBackend, MeasurementHarnessConfig, extrapolate_column_waves,
        matrix_bytes, operation_batch_size,
    };
    use crate::{MeasurementBackend, MeasurementNode, NodeMeasurement};
    use mxx_ir_core::{
        FrozenGraphScopeId, IntExpr, ParamEnv,
        node::{IndexRange, MatrixBinaryOp, NodeKind},
        types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId},
    };
    use num_bigint::BigInt;
    use std::collections::HashMap;

    #[test]
    fn column_wave_extrapolation_measures_the_remainder_separately() {
        let full_wave =
            NodeMeasurement { work_seconds: 38.0, latency_seconds: 39.0, workspace_bytes: 56 };
        let remainder_wave =
            NodeMeasurement { work_seconds: 25.0, latency_seconds: 26.0, workspace_bytes: 35 };

        let measurement = extrapolate_column_waves(&full_wave, 6.0, Some(&remainder_wave));

        assert_eq!(measurement.work_seconds, 253.0);
        assert_eq!(measurement.latency_seconds, 260.0);
        assert_eq!(measurement.workspace_bytes, 56);
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

        let first_key = GpuNodeMeasurementBackend::measurement_key(&first, &ParamEnv::default(), 1)
            .expect("cache key");
        let second_key =
            GpuNodeMeasurementBackend::measurement_key(&second, &ParamEnv::default(), 1)
                .expect("cache key");
        let wider_wave_key =
            GpuNodeMeasurementBackend::measurement_key(&first, &ParamEnv::default(), 4)
                .expect("cache key");

        assert_eq!(first_key, second_key);
        assert_ne!(first_key, wider_wave_key);
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
            tag_prefix: Vec::new(),
            tag_expressions: vec![IntExpr::LoopIndex(0)],
            tag_decimal_expressions: Vec::new(),
            tag_u64_le_expressions: Vec::new(),
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

        let first_key = GpuNodeMeasurementBackend::measurement_key(&node, &first_bindings, 1)
            .expect("first cache key");
        let second_key = GpuNodeMeasurementBackend::measurement_key(&node, &second_bindings, 1)
            .expect("second cache key");

        assert_eq!(first_key, second_key);
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
            tag_prefix,
            tag_expressions: vec![tag_expression],
            tag_decimal_expressions: Vec::new(),
            tag_u64_le_expressions: Vec::new(),
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
        let first_key = GpuNodeMeasurementBackend::measurement_key(&first, &ParamEnv::default(), 1)
            .expect("first cache key");
        let second_key =
            GpuNodeMeasurementBackend::measurement_key(&second, &ParamEnv::default(), 1)
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
            tag_prefix: Vec::new(),
            tag_expressions: Vec::new(),
            tag_decimal_expressions: Vec::new(),
            tag_u64_le_expressions: Vec::new(),
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
            GpuNodeMeasurementBackend::representative_node(&node, 40, 4);
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
    fn oversized_single_column_sampler_uses_one_row() {
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
            GpuNodeMeasurementBackend::representative_node(&node, 40, 4);
        let NodeKind::UniformIntervalSample { matrix_type, .. } = kind else {
            panic!("uniform representative kind");
        };
        let ConcreteWireType::Matrix(output) = &output_types[0] else {
            panic!("uniform representative output");
        };
        assert_eq!((output.rows, output.columns), (1, 1));
        assert_eq!(matrix_type.rows, IntExpr::constant(1));
        assert_eq!(scale, 2_621_440.0);
    }

    #[test]
    fn slice_measurement_uses_only_the_copied_output_shape() {
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
            GpuNodeMeasurementBackend::representative_node(&node, 40, 4);
        let NodeKind::Slice { columns: Some(columns), .. } = kind else {
            panic!("slice representative kind");
        };
        let ConcreteWireType::Matrix(input) = &argument_types[0] else {
            panic!("slice representative input");
        };
        assert_eq!((input.rows, input.columns), (1, 80));
        assert_eq!(columns.start, IntExpr::constant(0));
        assert_eq!(columns.end, IntExpr::constant(80));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn tensor_measurement_preserves_representative_column_product() {
        let matrix = |rows, columns| ConcreteMatrixType {
            rows,
            columns,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let kind = NodeKind::Tensor;
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
                ConcreteWireType::Matrix(matrix(2, 2)),
                ConcreteWireType::Matrix(matrix(3, 40)),
            ],
            concrete_output_types: vec![ConcreteWireType::Matrix(matrix(6, 80))],
        };

        let (_, arguments, outputs, scale, _) =
            GpuNodeMeasurementBackend::representative_node(&node, 40, 4);
        let ConcreteWireType::Matrix(left) = &arguments[0] else {
            panic!("tensor representative left input");
        };
        let ConcreteWireType::Matrix(right) = &arguments[1] else {
            panic!("tensor representative right input");
        };
        let ConcreteWireType::Matrix(output) = &outputs[0] else {
            panic!("tensor representative output");
        };
        assert_eq!((left.columns, right.columns), (1, 4));
        assert_eq!(output.columns, left.columns * right.columns);
        assert_eq!(scale, 20.0);
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
            concrete_output_types: vec![ConcreteWireType::Preimage(matrix(80, 80))],
        };
        let (_, gadget_arguments, gadget_outputs, gadget_scale, _) =
            GpuNodeMeasurementBackend::representative_node(&gadget_node, 40, 4);
        let ConcreteWireType::Matrix(gadget_input) = &gadget_arguments[0] else {
            panic!("gadget representative input");
        };
        let ConcreteWireType::Preimage(gadget_output) = &gadget_outputs[0] else {
            panic!("gadget representative output");
        };
        assert_eq!((gadget_input.rows, gadget_input.columns), (1, 4));
        assert_eq!((gadget_output.rows, gadget_output.columns), (80, 4));
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
            GpuNodeMeasurementBackend::representative_node(&multiply_node, 40, 4);
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
            GpuNodeMeasurementBackend::representative_node(&accumulate_node, 40, 4);
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
    fn preimage_measurement_uses_configured_column_wave() {
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
            concrete_output_types: vec![ConcreteWireType::Preimage(matrix(82, 80))],
        };

        let (kind, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 40, 4);
        let NodeKind::PreimageSample { matrix_type, .. } = kind else {
            panic!("preimage representative kind");
        };
        let ConcreteWireType::Matrix(target) = &arguments[2] else {
            panic!("preimage representative target");
        };
        let ConcreteWireType::Preimage(output) = &outputs[0] else {
            panic!("preimage representative output");
        };
        assert_eq!(target.columns, 4);
        assert_eq!((output.rows, output.columns), (82, 4));
        assert_eq!(matrix_type.columns, IntExpr::constant(4));
        assert_eq!(scale, 20.0);
        assert_eq!(remainder_columns, None);

        let (_, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 40, 12);
        let ConcreteWireType::Matrix(target) = &arguments[2] else {
            panic!("preimage representative target");
        };
        let ConcreteWireType::Preimage(output) = &outputs[0] else {
            panic!("preimage representative output");
        };
        assert_eq!(target.columns, 12);
        assert_eq!((output.rows, output.columns), (82, 12));
        assert_eq!(scale, 6.0);
        assert_eq!(remainder_columns, Some(8));
    }

    #[test]
    fn family_preimage_measurement_preserves_the_runtime_batch_cardinality() {
        let kind = NodeKind::FamilyPreimageSample {
            matrix_type: MatrixType {
                rows: IntExpr::constant(3),
                columns: IntExpr::constant(1),
                ring_dimension: IntExpr::constant(32),
                modulus: IntExpr::constant(257),
            },
            max_coefficient_bound: IntExpr::constant(100),
        };
        let scope = FrozenGraphScopeId::Root;
        let make_node = |concrete_output_types| MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![],
            concrete_output_types,
        };
        let node = make_node(vec![ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Preimage(ConcreteMatrixType {
                rows: 3,
                columns: 1,
                ring_dimension: 32,
                modulus: BigInt::from(257u16),
            })),
            shape: vec![2, 3],
        }]);
        assert_eq!(operation_batch_size(&node).unwrap(), 6);

        let mut empty_outputs = node.concrete_output_types.clone();
        let ConcreteWireType::Family { shape, .. } = &mut empty_outputs[0] else {
            unreachable!();
        };
        *shape = vec![usize::MAX, 2, 0];
        let empty = make_node(empty_outputs);
        assert_eq!(operation_batch_size(&empty).unwrap(), 0);

        let non_family = make_node(vec![ConcreteWireType::Preimage(ConcreteMatrixType {
            rows: 3,
            columns: 1,
            ring_dimension: 32,
            modulus: BigInt::from(257u16),
        })]);
        assert!(operation_batch_size(&non_family).is_err());
        assert!(operation_batch_size(&make_node(vec![])).is_err());

        let malformed = make_node(vec![ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Int),
            shape: vec![1],
        }]);
        assert!(operation_batch_size(&malformed).is_err());

        let overflow = make_node(vec![ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Preimage(ConcreteMatrixType {
                rows: 3,
                columns: 1,
                ring_dimension: 32,
                modulus: BigInt::from(257u16),
            })),
            shape: vec![usize::MAX, 2],
        }]);
        assert!(operation_batch_size(&overflow).is_err());
    }

    #[test]
    fn empty_family_preimage_measurement_is_zero_without_gpu_collection() {
        let kind = NodeKind::FamilyPreimageSample {
            matrix_type: MatrixType {
                rows: IntExpr::constant(3),
                columns: IntExpr::constant(1),
                ring_dimension: IntExpr::constant(32),
                modulus: IntExpr::constant(257),
            },
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
            concrete_argument_types: vec![],
            concrete_output_types: vec![ConcreteWireType::Family {
                element: Box::new(ConcreteWireType::Preimage(ConcreteMatrixType {
                    rows: 3,
                    columns: 1,
                    ring_dimension: 32,
                    modulus: BigInt::from(257u16),
                })),
                shape: vec![4, 0],
            }],
        };
        let mut backend = GpuNodeMeasurementBackend {
            workers: Vec::new(),
            harness: MeasurementHarnessConfig::default(),
            crt_depth: 2,
            column_wave_size: 4,
            measurements: HashMap::new(),
            pending: HashMap::new(),
            collecting: false,
        };

        let measured = backend.measure("empty-family", &node, &ParamEnv::default()).unwrap();

        assert_eq!(measured, NodeMeasurement::default());
        assert!(backend.pending.is_empty());
    }
}
