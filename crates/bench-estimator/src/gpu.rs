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
    types::{ConcreteMatrixType, ConcreteWireType, WireRef},
};
use mxx_primitives::{
    env::mul_small_rhs_tile_columns,
    matrix::gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuSmallMatrix},
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, gpu_memory_info},
    },
    sampler::trapdoor::gpu::GpuDCRTTrapdoor,
};
use mxx_runtime::{
    Backend,
    backend::{
        IndexRange, MatrixMulAccumulateRequest, PreimageRequest, SampleRange,
        poly::gpu::{GpuDcrtBackend, gpu_backend_on},
    },
};
use num_bigint::{BigInt, Sign};
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fmt,
    hint::black_box,
    sync::{Arc, Mutex},
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

fn append_tag_integer(tag: &mut Vec<u8>, value: &BigInt) {
    let (sign, bytes) = value.to_bytes_be();
    tag.push(if matches!(sign, Sign::Minus) { 1 } else { 0 });
    tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    tag.extend_from_slice(&bytes);
}

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
    small_arguments: Vec<Option<Arc<GpuSmallMatrix>>>,
    preimage_trapdoor: Option<(GpuDCRTPolyMatrix, GpuDCRTTrapdoor, f64, BigInt, usize, BigInt)>,
    hash_key: [u8; 32],
}

enum GpuMeasurementOutput {
    Matrix(GpuDCRTPolyMatrix),
    SmallMatrix(GpuSmallMatrix),
}

impl GpuMeasurementOutput {
    fn wait_until_ready(&self) {
        match self {
            Self::Matrix(value) => value.wait_until_ready(),
            Self::SmallMatrix(value) => value.wait_until_ready(),
        }
    }
}

struct GpuMeasurementWorker {
    backend: Option<GpuDcrtBackend>,
    device_id: i32,
    ring_dimension: u32,
    moduli: Vec<u64>,
    base_bits: u32,
}

impl GpuMeasurementWorker {
    fn fresh_backend(&self) -> GpuDcrtBackend {
        let parameters = GpuDCRTPolyParams::new_with_gpu(
            self.ring_dimension,
            self.moduli.clone(),
            self.base_bits,
            vec![self.device_id],
            Some(1),
        );
        gpu_backend_on([parameters], [self.device_id])
    }

    fn reset_backend(&mut self) {
        drop(self.backend.take());
        self.backend = Some(self.fresh_backend());
    }

    fn backend(&mut self) -> &mut GpuDcrtBackend {
        self.backend.as_mut().expect("GPU measurement worker backend is initialized")
    }
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
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage { matrix, .. } => {
                    (matrix.rows as u128)
                        .saturating_mul(matrix.columns as u128)
                        .saturating_mul(matrix.ring_dimension as u128)
                        .saturating_mul(8)
                }
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } => {
                    compact_matrix_bytes(matrix, max_coefficient_bound) as u128
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
    fn canonicalize_constant_matrix_payload(
        kind: &mut NodeKind,
        output_types: &[ConcreteWireType],
    ) {
        let NodeKind::ConstantMatrix { value, .. } = kind else {
            return;
        };
        let Some(matrix) = output_types.iter().find_map(|wire_type| match wire_type {
            ConcreteWireType::Matrix(matrix) |
            ConcreteWireType::SmallMatrix { matrix, .. } |
            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
            _ => None,
        }) else {
            return;
        };

        match value {
            ConstantMatrix::Rotation { exponent } => {
                // Constructing a monomial matrix follows the same backend path for every valid
                // exponent. Use the last nonzero monomial as the representative so large lookup
                // families do not request one GPU measurement per rotation, while avoiding the
                // potentially special constant-polynomial case X^0.
                *exponent = mxx_ir_core::IntExpr::constant(matrix.ring_dimension.saturating_sub(1));
            }
            ConstantMatrix::UnitRow { index } | ConstantMatrix::UnitColumn { index } => {
                // Position does not change allocation or GPU work. Zero remains valid if the
                // representative matrix is later reduced to one row or column.
                *index = mxx_ir_core::IntExpr::constant(0);
            }
            _ => {}
        }
    }

    /// Creates a representative GPU measurement backend for validated IR nodes.
    pub fn new(
        gpu_parameters: &GpuDCRTPolyParams,
        device_ids: Vec<i32>,
        harness: MeasurementHarnessConfig,
        crt_depth: usize,
        column_wave_size: usize,
    ) -> Self {
        assert!(!device_ids.is_empty(), "GPU measurement requires at least one backend");
        assert!(column_wave_size > 0, "GPU measurement column wave must be nonzero");
        let workers = device_ids
            .into_iter()
            .map(|device_id| {
                let mut worker = GpuMeasurementWorker {
                    backend: None,
                    device_id,
                    ring_dimension: gpu_parameters.ring_dimension(),
                    moduli: gpu_parameters.moduli().to_vec(),
                    base_bits: gpu_parameters.base_bits(),
                };
                worker.reset_backend();
                worker
            })
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
        let mut measurement_kinds = BTreeMap::<String, usize>::new();
        for request in &requests {
            let rendered = format!("{:?}", request.kind);
            let end = rendered.find([' ', '{', '(']).unwrap_or(rendered.len());
            *measurement_kinds.entry(rendered[..end].to_owned()).or_default() += 1;
        }
        for (kind, count) in &measurement_kinds {
            info!(kind, count, "collected GPU measurement kind");
        }
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
                    debug!(
                        device_id = worker.device_id,
                        kind = ?request.kind,
                        arguments = ?request.concrete_argument_types,
                        outputs = ?request.concrete_output_types,
                        scale = request.scale,
                        "GPU representative measurement begin"
                    );
                    let measurement = Self::measure_request(worker, harness, &request)?;
                    // Representative values are dead once their timing and workspace have been
                    // recorded.  Fence the backend's deferred-release stream before the worker
                    // allocates the next full-shape O(log q) input; otherwise a completed
                    // preimage measurement can leave its near-VRAM-limit allocation pending and
                    // make an unrelated subsequent measurement fail spuriously.
                    worker
                        .backend()
                        .fence_released_memory()
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    let memory = gpu_memory_info(worker.device_id).map_err(GpuMeasurementError)?;
                    let used = memory.total.saturating_sub(memory.free);
                    if used.saturating_mul(4) >= memory.total.saturating_mul(3) {
                        info!(
                            device_id = worker.device_id,
                            used_bytes = used,
                            total_bytes = memory.total,
                            "resetting high-water GPU measurement context"
                        );
                        worker.reset_backend();
                    }
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
            decomposition: Option<&'a NodeKind>,
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
        // The GPU automorphism kernel and its allocation depend on the matrix
        // geometry, not on which odd ring automorphism index is selected.
        // Canonicalize to one non-identity index so a large rotation family is
        // measured once and never inherits a potentially cheaper identity case.
        if let NodeKind::RingAutomorphism { index } = &mut shape_kind {
            *index = mxx_ir_core::IntExpr::constant(3);
        }
        Self::canonicalize_constant_matrix_payload(&mut shape_kind, &node.concrete_output_types);

        encoding::hash_canonical(&MeasurementKey {
            kind: &shape_kind,
            decomposition: node
                .argument_kinds
                .get(1)
                .copied()
                .filter(|kind| matches!(kind, NodeKind::GadgetDecompose { .. })),
            concrete_argument_types: &node.concrete_argument_types,
            concrete_output_types: &node.concrete_output_types,
            bindings: &shape_bindings,
            column_wave_size,
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    #[cfg(test)]
    fn representative_node<'a>(
        node: &'a MeasurementNode<'a>,
        crt_depth: usize,
        column_wave_size: usize,
    ) -> (NodeKind, Vec<ConcreteWireType>, Vec<ConcreteWireType>, f64, Option<usize>) {
        Self::representative_node_with_axis_bound(
            node,
            &ParamEnv::default(),
            crt_depth,
            column_wave_size,
        )
    }

    fn representative_node_with_axis_bound<'a>(
        node: &'a MeasurementNode<'a>,
        _bindings: &ParamEnv,
        crt_depth: usize,
        column_wave_size: usize,
    ) -> (NodeKind, Vec<ConcreteWireType>, Vec<ConcreteWireType>, f64, Option<usize>) {
        assert!(column_wave_size > 0, "GPU measurement column wave must be nonzero");
        assert!(crt_depth > 0, "CRT depth must be nonzero");
        let capped_column_axis = |columns: usize| {
            let representative_columns = columns.min(column_wave_size);
            let full_waves = columns / representative_columns;
            let remainder_columns = columns % representative_columns;
            (
                representative_columns,
                full_waves as f64,
                (remainder_columns > 0).then_some(remainder_columns),
            )
        };
        // A representative is a real production column chunk.  Its row and reduction
        // dimensions must remain exact; the old axis-bound heuristic silently replaced those
        // dimensions with a smaller mathematical problem.  Only a column-separable operation
        // may reduce its target-column axis to the configured production chunk width.
        let capped_columns = |matrix: &ConcreteMatrixType| {
            (matrix.columns > column_wave_size).then(|| capped_column_axis(matrix.columns))
        };
        let mut kind = node.kind.clone();
        if let NodeKind::RingAutomorphism { index } = &mut kind {
            *index = mxx_ir_core::IntExpr::constant(3);
        }
        let mut argument_types = node.concrete_argument_types.clone();
        let mut output_types = node.concrete_output_types.clone();
        Self::canonicalize_constant_matrix_payload(&mut kind, &output_types);
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
                    ConcreteWireType::Matrix(matrix) => ConcreteWireType::Preimage {
                        matrix: matrix.clone(),
                        max_coefficient_bound: BigInt::from(0),
                    },
                    ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                        ConcreteWireType::Preimage {
                            matrix: matrix.clone(),
                            max_coefficient_bound: max_coefficient_bound.clone(),
                        }
                    }
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
            // Sampling is a complete production operation unless a real range API is used.
            // Keep the exact matrix dimensions; representative column waves would misprice it.
            NodeKind::UniformResidueSample { .. } |
            NodeKind::UniformIntervalSample { .. } |
            NodeKind::GaussianSample { .. } |
            NodeKind::LiftIntegerToConstantPolynomial { .. } => {}
            // Hash sampling is keyed by one logical request.  It is not a column-wave
            // operation, so measuring a reduced representative would misprice the hash path.
            NodeKind::HashSample { .. } => {}
            NodeKind::Slice { rows: _, columns } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                let Some(_input) =
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
                    output.columns = representative_columns;
                    if let Some(range) = columns {
                        range.end = mxx_ir_core::IntExpr::Add(
                            Box::new(range.start.clone()),
                            Box::new(mxx_ir_core::IntExpr::constant(representative_columns)),
                        );
                    }
                    scale = column_scale;
                    remainder_columns = column_remainder;
                }
            }
            // A transpose changes which axis is the target-column axis.  There is no
            // production transpose-column chunk operation to benchmark here, so retain the
            // exact full shape rather than fabricating a smaller input row dimension.
            NodeKind::Transpose => {}
            // Tensor couples both input column axes into the output layout.  A column tile is
            // therefore not an independent instance of the operation; keep the exact production
            // shape and let the backend measure its own workspace schedule.
            NodeKind::Tensor => {}
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
            NodeKind::Concat { axis: ConcatAxis::Columns | ConcatAxis::Diagonal } => {
                // Column and diagonal concatenation changes the source-to-output column map.
                // A truncated call is not a production operation unless the backend exposes an
                // explicit range API (it currently does not), so retain the complete shapes.
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
            NodeKind::MatrixScale { .. } | NodeKind::MatrixNegate => {
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
                                            ConcreteWireType::SmallMatrix { matrix, .. } |
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
                        let [lhs_wire, rhs_wire, ..] = argument_types.as_mut_slice() else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        let Some(_) = (match lhs_wire {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::SmallMatrix { matrix, .. } |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        }) else {
                            return (kind, argument_types, output_types, scale, remainder_columns);
                        };
                        let Some(rhs) = (match rhs_wire {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                            _ => None,
                        }) else {
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
            NodeKind::MatrixMulSmallRhs => {
                // The production API receives one complete all-column compact RHS owner and
                // performs any DCRT/NTT column tiling internally.  Measure that exact call;
                // replacing either owner or output dimensions with a tile would benchmark a
                // different mathematical operation and double-count the internal waves.
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let representative = coefficients.iter().enumerate().find_map(|(product, _)| {
                    argument_types.get(2 * product + 1).and_then(|wire_type| match wire_type {
                        ConcreteWireType::Matrix(matrix) |
                        ConcreteWireType::Preimage { matrix, .. } => capped_columns(matrix),
                        _ => None,
                    })
                });
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
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
                                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
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
                                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
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
            NodeKind::PreimageSample { .. } => {
                // The sampler receives the complete target and returns one complete compact
                // destination.  Its candidate/perturbation workspaces are column-tiled inside
                // the production primitive, so retain the exact public shape and measure this
                // one production call.
            }
            NodeKind::CrtRecompose { .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix),
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale, remainder_columns);
                };
                // CrtRecompose receives one matrix for every CRT limb.  Even when each limb and
                // the output are 1 x ell (linear size), the complete operation owns
                // crt_depth x ell entries and is therefore quadratic in log(q).  Recomposition
                // is column-separable, so retain every CRT limb while measuring one output-column
                // wave and extrapolating across the unchanged logical column count.
                let aggregate_is_quadratic =
                    argument_types.len() > 1 && output.columns > column_wave_size;
                if let Some((representative_columns, column_scale, column_remainder)) =
                    capped_columns(output).or_else(|| {
                        aggregate_is_quadratic.then(|| capped_column_axis(output.columns))
                    })
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
            // These nodes consume only the explicitly requested coefficient/threshold input;
            // unused matrix columns do not represent repeated production work.
            NodeKind::ExtractCoefficient { .. } | NodeKind::ThresholdDecode { .. } => {}
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
                ConcreteWireType::Matrix(matrix) => backend
                    .constant_matrix(&matrix, &ConstantMatrix::Zero, bindings)
                    .map(|matrix| Some(Arc::new(matrix)))
                    .map_err(|error| GpuMeasurementError(error.to_string())),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let small_arguments = node
            .concrete_argument_types
            .iter()
            .map(|wire_type| match wire_type {
                ConcreteWireType::SmallMatrix { matrix: _, max_coefficient_bound } |
                ConcreteWireType::Preimage { matrix: _, max_coefficient_bound } => {
                    let matrix_type = wire_type
                        .matrix_type()
                        .expect("bounded matrix carries a matrix type")
                        .clone();
                    // Route the benchmark through the same canonical compact codec used by
                    // artifact/range loading.  The zero matrix is only a deterministic source
                    // payload for the measurement; the owner itself is reconstructed from
                    // compact bytes, rather than treating a full matrix as the production RHS.
                    let magnitude_bytes = usize::try_from(max_coefficient_bound.bits().div_ceil(8))
                        .map_err(|_| {
                            GpuMeasurementError(
                                "compact matrix bound width overflows usize".to_owned(),
                            )
                        })?
                        .max(1);
                    let payload_len = matrix_type
                        .rows
                        .checked_mul(matrix_type.columns)
                        .and_then(|count| count.checked_mul(matrix_type.ring_dimension))
                        .and_then(|count| count.checked_mul(1 + magnitude_bytes))
                        .ok_or_else(|| {
                            GpuMeasurementError(
                                "compact matrix payload length overflows".to_owned(),
                            )
                        })?;
                    let schema = mxx_ir_core::artifact::ConcreteBoundedMatrixSchema {
                        matrix: matrix_type,
                        max_coefficient_bound: max_coefficient_bound.clone(),
                    };
                    // The production codec consumes the compact coefficient payload.  Build
                    // one deterministic zero payload with the complete logical dimensions;
                    // the backend allocates one all-column owner and performs the same
                    // canonical load used by artifact-backed execution.
                    let bytes = vec![0u8; payload_len];
                    let semantic_kind = if matches!(wire_type, ConcreteWireType::Preimage { .. }) {
                        mxx_ir_core::artifact::SmallMatrixSemanticKind::Preimage
                    } else {
                        mxx_ir_core::artifact::SmallMatrixSemanticKind::Generic
                    };
                    backend
                        .small_matrix_from_bytes(&schema, &bytes, semantic_kind)
                        .map_err(|error| GpuMeasurementError(error.to_string()))
                        .map(|value| Some(Arc::new(value)))
                }
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
            let max_coefficient_bound = match node.kind {
                NodeKind::PreimageSample { max_coefficient_bound, .. } => max_coefficient_bound
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?,
                _ => {
                    return Err(GpuMeasurementError(
                        "preimage measurement is missing its requested cutoff".to_owned(),
                    ));
                }
            };
            Some((
                public,
                trapdoor,
                sigma,
                gadget_base.clone(),
                *digit_count,
                // Use the exact production cutoff. In particular, a tight cutoff below the
                // sampler's conservative default must fail closed before a sampling attempt.
                max_coefficient_bound,
            ))
        } else {
            None
        };
        // Hash sampling needs a real, deterministic request identity.  Hashing a fixed fixture
        // key would make the benchmark exercise a different PRF input than the graph under
        // measurement.  Include the canonical node descriptor, argument identities, and fully
        // concrete shape so the generated value is tied to the actual accepted request.
        #[derive(serde::Serialize)]
        struct HashRequestIdentity<'a> {
            kind: &'a NodeKind,
            arguments: &'a [WireRef],
            concrete_argument_types: &'a [ConcreteWireType],
            concrete_output_types: &'a [ConcreteWireType],
        }
        let hash_key = encoding::hash_canonical(&HashRequestIdentity {
            kind: node.kind,
            arguments: node.arguments,
            concrete_argument_types: &node.concrete_argument_types,
            concrete_output_types: &node.concrete_output_types,
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        Ok(PreparedMeasurement { arguments, small_arguments, preimage_trapdoor, hash_key })
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
        let measure_wave = |worker: &mut GpuMeasurementWorker,
                            representative: &RepresentativeMeasurement|
         -> Result<NodeMeasurement, GpuMeasurementError> {
            let ring_dimension =
                representative.concrete_argument_types.first().map(family_leaf_type).and_then(
                    |wire_type| match wire_type {
                        ConcreteWireType::Matrix(matrix) |
                        ConcreteWireType::Preimage { matrix, .. } => Some(matrix.ring_dimension),
                        _ => None,
                    },
                );
            if matches!(representative.kind, NodeKind::RingAutomorphism { .. }) {
                let n = ring_dimension.ok_or_else(|| {
                    GpuMeasurementError("ring automorphism input type is missing".to_owned())
                })?;
                let two_n = n.checked_mul(2).ok_or_else(|| {
                    GpuMeasurementError("ring automorphism dimension overflows usize".to_owned())
                })?;
                let mut indices = vec![3usize, n.saturating_sub(1), n.saturating_add(1), two_n - 1];
                indices.retain(|index| *index > 1 && *index < two_n && index % 2 == 1);
                indices.sort_unstable();
                indices.dedup();
                let mut maximum = NodeMeasurement::default();
                for index in &indices {
                    let mut representative = RepresentativeMeasurement {
                        kind: representative.kind.clone(),
                        concrete_argument_types: representative.concrete_argument_types.clone(),
                        concrete_output_types: representative.concrete_output_types.clone(),
                        operation_batch_size: representative.operation_batch_size,
                    };
                    representative.kind = NodeKind::RingAutomorphism {
                        index: mxx_ir_core::IntExpr::constant(*index),
                    };
                    let measured = Self::measure_representative(
                        worker,
                        harness,
                        &request.scope,
                        request.id,
                        &request.bindings,
                        &representative,
                    )?;
                    maximum.work_seconds = maximum.work_seconds.max(measured.work_seconds);
                    maximum.latency_seconds = maximum.latency_seconds.max(measured.latency_seconds);
                    maximum.workspace_bytes = maximum.workspace_bytes.max(measured.workspace_bytes);
                }
                info!(
                    device_id = worker.device_id,
                    ?indices,
                    maximum = ?maximum,
                    "measured conservative ring-automorphism access patterns"
                );
                Ok(maximum)
            } else {
                Self::measure_representative(
                    worker,
                    harness,
                    &request.scope,
                    request.id,
                    &request.bindings,
                    representative,
                )
            }
        };
        let measured = measure_wave(worker, &full_wave)?;
        let remainder = request
            .remainder
            .as_ref()
            .map(|remainder| measure_wave(worker, remainder))
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
        let prepared = Self::prepare(worker.backend(), &node, bindings)?;
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
                match Self::run_node(worker.backend(), &node, bindings, operation_batch, &prepared)
                {
                    Ok(outputs) => {
                        outputs.iter().for_each(GpuMeasurementOutput::wait_until_ready);
                        // Observe the same owner boundary used by runtime persistence: a
                        // producer is not considered complete until its result is readable and
                        // its canonical output bytes can be handed to the artifact store.
                        for output in &outputs {
                            let result = match output {
                                GpuMeasurementOutput::Matrix(value) => {
                                    black_box(worker.backend().matrix_to_bytes(value));
                                    Ok(())
                                }
                                GpuMeasurementOutput::SmallMatrix(value) => {
                                    let wire_type = node
                                        .concrete_output_types
                                        .iter()
                                        .find(|wire_type| {
                                            matches!(
                                                wire_type,
                                                ConcreteWireType::SmallMatrix { .. } |
                                                    ConcreteWireType::Preimage { .. }
                                            )
                                        })
                                        .ok_or_else(|| {
                                            GpuMeasurementError(
                                                "compact output schema is missing".to_owned(),
                                            )
                                        });
                                    wire_type.and_then(|wire_type| {
                                        let (matrix, max_coefficient_bound, semantic_kind) =
                                            match wire_type {
                                                ConcreteWireType::SmallMatrix {
                                                    matrix,
                                                    max_coefficient_bound,
                                                } => (
                                                    matrix,
                                                    max_coefficient_bound,
                                                    mxx_ir_core::artifact::SmallMatrixSemanticKind::Generic,
                                                ),
                                                ConcreteWireType::Preimage {
                                                    matrix,
                                                    max_coefficient_bound,
                                                } => (
                                                    matrix,
                                                    max_coefficient_bound,
                                                    mxx_ir_core::artifact::SmallMatrixSemanticKind::Preimage,
                                                ),
                                                _ => unreachable!(),
                                            };
                                        let schema =
                                            mxx_ir_core::artifact::ConcreteBoundedMatrixSchema {
                                                matrix: matrix.clone(),
                                                max_coefficient_bound: max_coefficient_bound.clone(),
                                            };
                                        let bytes = worker
                                            .backend()
                                            .small_matrix_to_bytes(value, &schema, semantic_kind)
                                            .map_err(|error| {
                                                GpuMeasurementError(error.to_string())
                                            })?;
                                        black_box(bytes);
                                        Ok(())
                                    })
                                }
                            };
                            if let Err(error) = result {
                                operation_error = Some(error);
                                break;
                            }
                        }
                    }
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
        let small_matrix = |index: usize| {
            prepared.small_arguments.get(index).and_then(Option::as_ref).ok_or_else(|| {
                GpuMeasurementError(format!(
                    "node {:?} argument {index} is not a bounded matrix",
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
        let matrix_outputs = |outputs: Result<Vec<GpuDCRTPolyMatrix>, GpuMeasurementError>| {
            outputs.map(|values| values.into_iter().map(GpuMeasurementOutput::Matrix).collect())
        };
        match node.kind {
            NodeKind::ConstantMatrix { value, .. } => {
                let ty = output_matrix_type()?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend.constant_matrix(&ty, value, bindings).map_err(backend_error)
                        })
                        .collect(),
                )
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
                matrix_outputs(
                    match operation {
                        MatrixBinaryOp::Add => backend.add_batch(inputs),
                        MatrixBinaryOp::Subtract => backend.sub_batch(inputs),
                        MatrixBinaryOp::Multiply => backend.multiply_batch(inputs),
                    }
                    .map_err(backend_error),
                )
            }
            NodeKind::MatrixMulSmallRhs => {
                let left = matrix_arc(0)?;
                let right = small_matrix(1)?;
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| {
                            backend.multiply_small_rhs(left.as_ref(), right).map_err(backend_error)
                        })
                        .collect(),
                )
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
                matrix_outputs(backend.matrix_mul_accumulate_batch(requests).map_err(backend_error))
            }
            NodeKind::MatrixNegate => matrix_outputs(
                backend
                    .negate_batch(
                        (0..batch_size).map(|_| matrix_arc(0)).collect::<Result<Vec<_>, _>>()?,
                    )
                    .map_err(backend_error),
            ),
            NodeKind::MatrixScale { scalar } => {
                let scalar = scalar
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                matrix_outputs(
                    backend
                        .scale_integer_batch(
                            (0..batch_size)
                                .map(|_| Ok((matrix_arc(0)?, scalar.clone())))
                                .collect::<Result<Vec<_>, GpuMeasurementError>>()?,
                        )
                        .map_err(backend_error),
                )
            }
            NodeKind::RingAutomorphism { index } => {
                let index = evaluate_usize(index)?;
                matrix_outputs(
                    backend
                        .ring_automorphism_batch(
                            (0..batch_size).map(|_| Ok((matrix_arc(0)?, index))).collect::<Result<
                                Vec<_>,
                                GpuMeasurementError,
                            >>(
                            )?,
                        )
                        .map_err(backend_error),
                )
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
            NodeKind::Tensor => matrix_outputs(
                (0..batch_size)
                    .map(|_| backend.tensor(matrix(0)?, matrix(1)?).map_err(backend_error))
                    .collect(),
            ),
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
                matrix_outputs(
                    (0..batch_size)
                        .map(|_| backend.concat(&inputs, *axis).map_err(backend_error))
                        .collect(),
                )
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
            NodeKind::HashSample {
                variant,
                tag_prefix,
                tag_expressions,
                tag_decimal_expressions,
                tag_u64_le_expressions,
                base,
                digit_count,
                ..
            } => {
                let ty = output_matrix_type()?;
                let mut tag = tag_prefix.clone();
                for expression in tag_expressions {
                    let value = expression
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    append_tag_integer(&mut tag, &value);
                }
                for expression in tag_decimal_expressions {
                    let value = expression
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    tag.extend_from_slice(value.to_string().as_bytes());
                }
                for expression in tag_u64_le_expressions {
                    let value = expression
                        .evaluate(bindings)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?
                        .to_u64()
                        .ok_or_else(|| {
                            GpuMeasurementError(
                                "little-endian hash tag component must fit in u64".to_owned(),
                            )
                        })?;
                    tag.extend_from_slice(&value.to_le_bytes());
                }
                let gadget_layout = base
                    .as_ref()
                    .zip(digit_count.as_ref())
                    .map(|(base, digit_count)| {
                        Ok((
                            base.evaluate(bindings)
                                .map_err(|error| GpuMeasurementError(error.to_string()))?,
                            evaluate_usize(digit_count)?,
                        ))
                    })
                    .transpose()?;
                matrix_outputs(
                    backend
                        .sample_hash_batch(
                            (0..batch_size)
                                .map(|_| mxx_runtime::backend::HashSampleRequest {
                                    matrix_type: ty.clone(),
                                    key: prepared.hash_key,
                                    tag: tag.clone(),
                                    variant: *variant,
                                    gadget_layout: gadget_layout.clone(),
                                })
                                .collect(),
                        )
                        .map_err(backend_error),
                )
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
                let target = matrix_arc(2)?;
                // The production sampler consumes a column source, not a monolithic DCRT
                // matrix.  Let the backend perform the same staging/observation conversion used
                // by runtime; its returned source loads complete rows for one column tile.
                let (target_source, target_staging) =
                    backend.preimage_target(target).map_err(backend_error)?;
                black_box(target_staging);
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
                            target_source.as_ref(),
                            [0u8; 32],
                        )
                        .map(|output| vec![GpuMeasurementOutput::SmallMatrix(output)])
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
                                    target: target_source.clone(),
                                    randomness_seed: [0u8; 32],
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
        // The estimator charges compact decomposition together with the consuming
        // MatrixMulSmallRhs production measurement, so it must not charge the producer twice.
        if matches!(node.kind, NodeKind::GadgetDecompose { .. }) {
            return Ok(NodeMeasurement::default());
        }
        if node.argument_kinds.iter().any(|kind| matches!(kind, NodeKind::GadgetDecompose { .. })) &&
            !matches!(node.kind, NodeKind::MatrixMulSmallRhs)
        {
            return Err(GpuMeasurementError(format!(
                "lazy gadget decomposition at {:?} node {:?} has unsupported consumer {:?}; arguments={:?}, argument_kinds={:?}",
                node.scope, node.id, node.kind, node.arguments, node.argument_kinds
            )));
        }
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
        ) = Self::representative_node_with_axis_bound(
            node,
            bindings,
            self.crt_depth,
            self.column_wave_size,
        );
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
                Self::representative_node_with_axis_bound(node, bindings, self.crt_depth, columns);
            RepresentativeMeasurement {
                kind,
                concrete_argument_types,
                concrete_output_types,
                operation_batch_size,
            }
        });
        if !self.pending.contains_key(&measurement_key) {
            info!(
                scope = ?node.scope,
                node = ?node.id,
                kind = ?representative_node.kind,
                original_arguments = ?node.concrete_argument_types,
                original_outputs = ?node.concrete_output_types,
                measured_arguments = ?representative_node.concrete_argument_types,
                measured_outputs = ?representative_node.concrete_output_types,
                column_wave_scale = scale,
                remainder_columns,
                "planned GPU representative measurement"
            );
        }
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
            // Runtime serializes every artifact-compatible lane output at the end of its
            // bounded ParallelGrid wave and retains only a StagedArtifactFamily descriptor.
            // The simultaneously live lane values are already included in the child peak and
            // multiplied by the active wave size in the generic estimator. Charging the entire
            // logical family here would incorrectly assume all lanes remain GPU-resident.
            return 0;
        }
        let artifact_family_input = matches!(kind, NodeKind::Input { artifact: Some(_), .. }) &&
            matches!(wire_type, ConcreteWireType::Family { .. });
        if artifact_family_input {
            // Artifact families remain store-backed descriptors. Runtime loads only the member
            // requested by the current bounded grid wave, whose live matrix is charged in that
            // consumer body. A scalar compact artifact, in contrast, is loaded once as the
            // complete all-column owner and must remain visible in resident VRAM accounting.
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
        if !matches!(kind, NodeKind::Input { .. }) {
            return 0;
        }
        self.persistent_storage_bytes_for_node(kind, wire_type)
    }

    fn chunk_count_for_node(&self, node: &MeasurementNode<'_>) -> usize {
        if matches!(
            node.kind,
            NodeKind::MatrixMulSmallRhs |
                NodeKind::PreimageSample { .. } |
                NodeKind::FamilyPreimageSample { .. }
        ) {
            // These production APIs receive the complete all-column owner/target and perform
            // their internal column tiling themselves.  The measured latency already includes
            // every tile; expose the exact column-wave count for reporting without multiplying
            // that latency a second time in the estimator.
            let columns = node
                .concrete_output_types
                .iter()
                .find_map(|wire_type| family_leaf_type(wire_type).matrix_type().map(|m| m.columns))
                .or_else(|| {
                    node.concrete_argument_types.iter().find_map(|wire_type| {
                        family_leaf_type(wire_type).matrix_type().map(|m| m.columns)
                    })
                })
                .unwrap_or(1);
            let tile_columns = mul_small_rhs_tile_columns().ok().flatten().unwrap_or(1).max(1);
            return columns.div_ceil(tile_columns.min(columns.max(1)));
        }
        // Only operations for which `representative_node_with_axis_bound` creates a
        // production column wave are chunked.  Wide logical values on logical/no-op nodes
        // (notably HashSample, coefficient/threshold extraction, and Transpose) are measured
        // once at their exact shape and must not acquire fictional chunk waves.
        let chunked = matches!(
            node.kind,
            NodeKind::UniformResidueSample { .. } |
                NodeKind::UniformIntervalSample { .. } |
                NodeKind::GaussianSample { .. } |
                NodeKind::LiftIntegerToConstantPolynomial { .. } |
                NodeKind::Slice { .. } |
                NodeKind::Tensor |
                NodeKind::Concat { .. } |
                NodeKind::GadgetDecompose { .. } |
                NodeKind::MatrixScale { .. } |
                NodeKind::MatrixNegate |
                NodeKind::MatrixBinary(_) |
                NodeKind::MatrixMulAccumulate { .. } |
                NodeKind::CrtRecompose { .. } |
                NodeKind::PackPolynomialCoefficients { .. }
        );
        if !chunked {
            return 1;
        }
        let columns = node
            .concrete_output_types
            .iter()
            .find_map(|wire_type| wire_type.matrix_type().map(|matrix| matrix.columns));
        columns.map_or(1, |columns| columns.div_ceil(self.column_wave_size))
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
    let magnitude_bytes = max_coefficient_bound
        .to_biguint()
        .map(|bound| bound.to_bytes_le().len().max(1))
        .unwrap_or(usize::MAX);
    let coefficient_count = matrix
        .rows
        .checked_mul(matrix.columns)
        .and_then(|count| count.checked_mul(matrix.ring_dimension))
        .unwrap_or(usize::MAX);
    let payload =
        coefficient_count.checked_mul(1usize.saturating_add(magnitude_bytes)).unwrap_or(usize::MAX);
    // This is the resident owner allocation: one sign-plus-magnitude encoding for every
    // logical coefficient.  It deliberately has no CRT-limb expansion and no artifact
    // framing.  In particular, the all-column owner is K*C*N*s_B bytes regardless of the
    // internal column/reduction wave configuration.
    u64::try_from(payload).unwrap_or(u64::MAX)
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
        GpuNodeMeasurementBackend, MeasurementHarnessConfig, compact_matrix_bytes,
        extrapolate_column_waves, matrix_bytes, operation_batch_size,
    };
    use crate::{MeasurementBackend, MeasurementNode, NodeMeasurement};
    use mxx_ir_core::{
        FrozenGraphScopeId, IntExpr, ParamEnv,
        node::{ConstantMatrix, IndexRange, MatrixBinaryOp, NodeKind},
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
    fn gadget_decomposition_retains_the_full_compact_output_shape() {
        let expanded = ConcreteWireType::Preimage {
            matrix: ConcreteMatrixType {
                rows: 80,
                columns: 80,
                ring_dimension: 65_536,
                modulus: BigInt::from(257u16),
            },
            max_coefficient_bound: BigInt::from(128),
        };
        let kind = NodeKind::GadgetDecompose {
            base: IntExpr::constant(16_384),
            small: false,
            digit_count: IntExpr::constant(80),
        };
        let backend = GpuNodeMeasurementBackend {
            workers: Vec::new(),
            harness: MeasurementHarnessConfig::default(),
            crt_depth: 44,
            column_wave_size: 1,
            measurements: HashMap::new(),
            pending: HashMap::new(),
            collecting: false,
        };

        let ConcreteWireType::Preimage { matrix, max_coefficient_bound } = &expanded else {
            unreachable!()
        };
        assert_eq!(
            backend.persistent_bytes_for_node(&kind, &expanded),
            compact_matrix_bytes(matrix, max_coefficient_bound)
        );
    }

    #[test]
    fn parallel_grid_artifact_family_is_staged_instead_of_gpu_resident() {
        let element = ConcreteMatrixType {
            rows: 176,
            columns: 176,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let family = ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Matrix(element)),
            shape: vec![usize::MAX],
        };
        let kind = NodeKind::ParallelGrid(mxx_ir_core::node::ParallelGrid {
            shape: vec![IntExpr::constant(usize::MAX)],
            index_slots: vec![0],
            bindings: Vec::new(),
            input_modes: Vec::new(),
        });
        let backend = GpuNodeMeasurementBackend {
            workers: Vec::new(),
            harness: MeasurementHarnessConfig::default(),
            crt_depth: 44,
            column_wave_size: 1,
            measurements: HashMap::new(),
            pending: HashMap::new(),
            collecting: false,
        };

        assert_eq!(backend.persistent_bytes_for_node(&kind, &family), 0);
    }

    #[test]
    fn artifact_family_input_is_store_backed_but_scalar_artifact_is_resident() {
        let matrix = ConcreteMatrixType {
            rows: 2,
            columns: 176,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let scalar = ConcreteWireType::Matrix(matrix.clone());
        let family = ConcreteWireType::Family {
            element: Box::new(scalar.clone()),
            shape: vec![690_733_711_360],
        };
        let kind = NodeKind::Input {
            name: "staged-family".to_owned(),
            wire_type: mxx_ir_core::types::WireType::Family {
                element: Box::new(mxx_ir_core::types::WireType::Matrix(
                    mxx_ir_core::types::MatrixType {
                        modulus: IntExpr::constant(257),
                        ring_dimension: IntExpr::constant(65_536),
                        rows: IntExpr::constant(2),
                        columns: IntExpr::constant(176),
                    },
                )),
                shape: vec![IntExpr::constant(690_733_711_360usize)],
            },
            artifact: Some(mxx_ir_core::node::ArtifactInput {
                production_id: mxx_ir_core::artifact::ProductionId {
                    spec_hash: mxx_ir_core::artifact::SpecHash([7; 32]),
                    execution_nonce: [9; 32],
                },
                artifact_name: "staged-family".to_owned(),
                confidentiality: mxx_ir_core::artifact::ArtifactConfidentiality::Public,
            }),
        };
        let backend = GpuNodeMeasurementBackend {
            workers: Vec::new(),
            harness: MeasurementHarnessConfig::default(),
            crt_depth: 44,
            column_wave_size: 1,
            measurements: HashMap::new(),
            pending: HashMap::new(),
            collecting: false,
        };

        assert_eq!(backend.persistent_bytes_for_node(&kind, &family), 0);
        assert_eq!(backend.persistent_bytes_for_node(&kind, &scalar), matrix_bytes(&matrix, 44));
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
            variant: mxx_ir_core::node::HashVariant::Plain,
            tag_prefix: Vec::new(),
            tag_expressions: vec![IntExpr::LoopIndex(0)],
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
            variant: mxx_ir_core::node::HashVariant::Plain,
            tag_prefix,
            tag_expressions: vec![tag_expression],
            tag_decimal_expressions: Vec::new(),
            tag_u64_le_expressions: Vec::new(),
            base: None,
            digit_count: None,
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
    fn measurement_cache_key_ignores_ring_automorphism_index() {
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 2,
            columns: 176,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        });
        let first_kind = NodeKind::RingAutomorphism { index: IntExpr::constant(3) };
        let second_kind = NodeKind::RingAutomorphism { index: IntExpr::constant(127_489) };
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

        let first = node(&first_kind);
        let second = node(&second_kind);
        let first_key = GpuNodeMeasurementBackend::measurement_key(&first, &ParamEnv::default(), 1)
            .expect("first cache key");
        let second_key =
            GpuNodeMeasurementBackend::measurement_key(&second, &ParamEnv::default(), 1)
                .expect("second cache key");

        assert_eq!(first_key, second_key);
        let (representative, _, _, _, _) =
            GpuNodeMeasurementBackend::representative_node(&second, 44, 1);
        assert_eq!(representative, NodeKind::RingAutomorphism { index: IntExpr::constant(3) });
    }

    #[test]
    fn measurement_cache_key_ignores_constant_rotation_exponent() {
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            rows: 1,
            columns: 1,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        });
        let first_kind = NodeKind::ConstantMatrix {
            matrix_type: MatrixType {
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
                ring_dimension: IntExpr::constant(65_536),
                modulus: IntExpr::constant(257),
            },
            value: ConstantMatrix::Rotation { exponent: IntExpr::constant(1) },
        };
        let second_kind = NodeKind::ConstantMatrix {
            matrix_type: MatrixType {
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
                ring_dimension: IntExpr::constant(65_536),
                modulus: IntExpr::constant(257),
            },
            value: ConstantMatrix::Rotation { exponent: IntExpr::constant(32_769) },
        };
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
        let (representative, _, _, _, _) =
            GpuNodeMeasurementBackend::representative_node(&second, 44, 1);
        let NodeKind::ConstantMatrix { value: ConstantMatrix::Rotation { exponent }, .. } =
            representative
        else {
            panic!("expected a constant rotation representative");
        };
        assert_eq!(exponent, IntExpr::constant(65_535));
    }

    #[test]
    fn constant_gadget_measurement_keeps_the_full_valid_shape() {
        let concrete = ConcreteMatrixType {
            rows: 2,
            columns: 176,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let kind = NodeKind::ConstantMatrix {
            matrix_type: MatrixType {
                rows: IntExpr::constant(concrete.rows),
                columns: IntExpr::constant(concrete.columns),
                ring_dimension: IntExpr::constant(concrete.ring_dimension),
                modulus: IntExpr::constant(concrete.modulus.clone()),
            },
            value: ConstantMatrix::Gadget { base: IntExpr::constant(65_536), small: false },
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
            concrete_output_types: vec![ConcreteWireType::Matrix(concrete.clone())],
        };

        let (representative, _, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 44, 1);

        assert_eq!(representative, kind);
        assert_eq!(outputs, vec![ConcreteWireType::Matrix(concrete)]);
        assert_eq!(scale, 1.0);
        assert_eq!(remainder_columns, None);
    }

    #[test]
    fn linear_entry_hash_measurement_keeps_the_full_shape() {
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
            variant: mxx_ir_core::node::HashVariant::Plain,
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
            GpuNodeMeasurementBackend::representative_node(&node, 40, 4);
        let NodeKind::HashSample { matrix_type, .. } = kind else {
            panic!("hash representative kind");
        };
        let ConcreteWireType::Matrix(output) = &output_types[0] else {
            panic!("hash representative output");
        };
        assert_eq!((output.rows, output.columns), (1, 8_722));
        assert_eq!(matrix_type.columns, IntExpr::constant(8_722));
        assert_eq!(scale, 1.0);
        assert_eq!(remainder_columns, None);
    }

    #[test]
    fn linear_entry_single_column_sampler_keeps_the_full_shape() {
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
        assert_eq!((output.rows, output.columns), (2_621_440, 1));
        assert_eq!(matrix_type.rows, IntExpr::constant(2_621_440));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn linear_entry_slice_measurement_keeps_the_full_input_shape() {
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
        assert_eq!((input.rows, input.columns), (1, 8_720));
        assert_eq!(columns.start, IntExpr::constant(80));
        assert_eq!(columns.end.evaluate(&ParamEnv::default()).unwrap(), BigInt::from(84));
        assert_eq!(scale, 20.0);
    }

    #[test]
    fn crt_recompose_splits_quadratic_aggregate_by_output_columns() {
        let matrix = |columns| ConcreteMatrixType {
            rows: 1,
            columns,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let kind = NodeKind::CrtRecompose {
            plaintext_moduli: vec![IntExpr::constant(257); 44],
            reconstruction_coefficients: vec![IntExpr::constant(1); 44],
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
            concrete_argument_types: vec![ConcreteWireType::Matrix(matrix(176)); 44],
            concrete_output_types: vec![ConcreteWireType::Matrix(matrix(176))],
        };

        let (_, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 44, 1);
        assert_eq!(arguments.len(), 44);
        for argument in arguments {
            let ConcreteWireType::Matrix(argument) = argument else {
                panic!("CRT recomposition representative input");
            };
            assert_eq!((argument.rows, argument.columns), (1, 1));
        }
        let ConcreteWireType::Matrix(output) = &outputs[0] else {
            panic!("CRT recomposition representative output");
        };
        assert_eq!((output.rows, output.columns), (1, 1));
        assert_eq!(scale, 176.0);
        assert_eq!(remainder_columns, None);
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
        assert_eq!((left.columns, right.columns), (2, 40));
        assert_eq!(output.columns, 80);
        assert_eq!(scale, 1.0);
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
                max_coefficient_bound: BigInt::from(0),
            }],
        };
        let (_, gadget_arguments, gadget_outputs, gadget_scale, _) =
            GpuNodeMeasurementBackend::representative_node(&gadget_node, 40, 4);
        let ConcreteWireType::Matrix(gadget_input) = &gadget_arguments[0] else {
            panic!("gadget representative input");
        };
        let ConcreteWireType::Preimage { matrix: gadget_output, .. } = &gadget_outputs[0] else {
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
    fn ordinary_matrix_products_do_not_split_the_inner_dimension() {
        let matrix = |rows, columns| ConcreteMatrixType {
            rows,
            columns,
            ring_dimension: 65_536,
            modulus: BigInt::from(257u16),
        };
        let scope = FrozenGraphScopeId::Root;
        let multiply_kind = NodeKind::MatrixBinary(MatrixBinaryOp::Multiply);
        let multiply_node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &multiply_kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![
                ConcreteWireType::Matrix(matrix(88, 88)),
                ConcreteWireType::Matrix(matrix(88, 88)),
            ],
            concrete_output_types: vec![ConcreteWireType::Matrix(matrix(88, 88))],
        };

        let (_, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&multiply_node, 44, 4);
        let ConcreteWireType::Matrix(lhs) = &arguments[0] else {
            panic!("multiply representative lhs");
        };
        let ConcreteWireType::Matrix(rhs) = &arguments[1] else {
            panic!("multiply representative rhs");
        };
        let ConcreteWireType::Matrix(output) = &outputs[0] else {
            panic!("multiply representative output");
        };
        assert_eq!((lhs.rows, lhs.columns), (88, 88));
        assert_eq!((rhs.rows, rhs.columns), (88, 4));
        assert_eq!((output.rows, output.columns), (88, 4));
        assert_eq!(scale, 22.0);
        assert_eq!(remainder_columns, None);
    }

    #[test]
    fn preimage_measurement_preserves_complete_shape_for_internal_tiling() {
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
            GpuNodeMeasurementBackend::representative_node(&node, 40, 4);
        let NodeKind::PreimageSample { matrix_type, .. } = kind else {
            panic!("preimage representative kind");
        };
        let ConcreteWireType::Matrix(target) = &arguments[2] else {
            panic!("preimage representative target");
        };
        let ConcreteWireType::Preimage { matrix: output, .. } = &outputs[0] else {
            panic!("preimage representative output");
        };
        assert_eq!(target.columns, 80);
        assert_eq!((output.rows, output.columns), (82, 80));
        assert_eq!(matrix_type.columns, IntExpr::constant(80));
        assert_eq!(scale, 1.0);
        assert_eq!(remainder_columns, None);

        let (_, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 40, 12);
        let ConcreteWireType::Matrix(target) = &arguments[2] else {
            panic!("preimage representative target");
        };
        let ConcreteWireType::Preimage { matrix: output, .. } = &outputs[0] else {
            panic!("preimage representative output");
        };
        assert_eq!(target.columns, 80);
        assert_eq!((output.rows, output.columns), (82, 80));
        assert_eq!(scale, 1.0);
        assert_eq!(remainder_columns, None);
    }

    #[test]
    fn compact_rhs_measurement_preserves_all_columns_for_internal_tiling() {
        let matrix = |rows, columns| ConcreteMatrixType {
            rows,
            columns,
            ring_dimension: 64,
            modulus: BigInt::from(257u16),
        };
        let scope = FrozenGraphScopeId::Root;
        let kind = NodeKind::MatrixMulSmallRhs;
        let node = MeasurementNode {
            scope: &scope,
            id: NodeId(1),
            kind: &kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: vec![
                ConcreteWireType::Matrix(matrix(3, 80)),
                ConcreteWireType::SmallMatrix {
                    matrix: matrix(80, 176),
                    max_coefficient_bound: BigInt::from(255),
                },
            ],
            concrete_output_types: vec![ConcreteWireType::Matrix(matrix(3, 176))],
        };

        let (_, arguments, outputs, scale, remainder_columns) =
            GpuNodeMeasurementBackend::representative_node(&node, 40, 4);
        let ConcreteWireType::SmallMatrix { matrix: rhs, .. } = &arguments[1] else {
            panic!("compact RHS representative");
        };
        let ConcreteWireType::Matrix(output) = &outputs[0] else {
            panic!("compact RHS output representative");
        };
        assert_eq!((rhs.rows, rhs.columns), (80, 176));
        assert_eq!((output.rows, output.columns), (3, 176));
        assert_eq!(scale, 1.0);
        assert_eq!(remainder_columns, None);
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
            element: Box::new(ConcreteWireType::Preimage {
                matrix: ConcreteMatrixType {
                    rows: 3,
                    columns: 1,
                    ring_dimension: 32,
                    modulus: BigInt::from(257u16),
                },
                max_coefficient_bound: BigInt::from(100),
            }),
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

        let non_family = make_node(vec![ConcreteWireType::Preimage {
            matrix: ConcreteMatrixType {
                rows: 3,
                columns: 1,
                ring_dimension: 32,
                modulus: BigInt::from(257u16),
            },
            max_coefficient_bound: BigInt::from(100),
        }]);
        assert!(operation_batch_size(&non_family).is_err());
        assert!(operation_batch_size(&make_node(vec![])).is_err());

        let malformed = make_node(vec![ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Int),
            shape: vec![1],
        }]);
        assert!(operation_batch_size(&malformed).is_err());

        let overflow = make_node(vec![ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Preimage {
                matrix: ConcreteMatrixType {
                    rows: 3,
                    columns: 1,
                    ring_dimension: 32,
                    modulus: BigInt::from(257u16),
                },
                max_coefficient_bound: BigInt::from(100),
            }),
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
                element: Box::new(ConcreteWireType::Preimage {
                    matrix: ConcreteMatrixType {
                        rows: 3,
                        columns: 1,
                        ring_dimension: 32,
                        modulus: BigInt::from(257u16),
                    },
                    max_coefficient_bound: BigInt::from(100),
                }),
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
