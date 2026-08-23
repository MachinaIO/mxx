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
    backend::{IndexRange, PreimageRequest, SampleRange, poly::gpu::GpuDcrtBackend},
};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use serde::Serialize;
use std::{collections::HashMap, fmt, sync::Arc};
use tracing::{debug, info};

// Oversized column-separable operations are measured as one-column waves. The logical output size
// remains unchanged in the estimator's persistent-memory model.
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
    preimage_sample: bool,
}

pub struct GpuNodeMeasurementBackend {
    workers: Vec<GpuMeasurementWorker>,
    harness: MeasurementHarnessConfig,
    crt_depth: usize,
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
        let workers = backends
            .into_iter()
            .map(|(backend, device_id)| GpuMeasurementWorker { backend, device_id })
            .collect();
        Self {
            workers,
            harness,
            crt_depth,
            measurements: HashMap::new(),
            pending: HashMap::new(),
            collecting: true,
        }
    }

    /// Measures all shapes collected by prior estimator passes, distributing each unique shape
    /// to exactly one GPU. Subsequent estimator passes read the completed measurement cache.
    pub fn measure_collected(&mut self) -> Result<(), GpuMeasurementError> {
        self.collecting = false;
        let mut requests = std::mem::take(&mut self.pending).into_values().collect::<Vec<_>>();
        requests.sort_by(|left, right| right.scale.total_cmp(&left.scale));
        let mut buckets = (0..self.workers.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (index, request) in requests.into_iter().enumerate() {
            buckets[index % self.workers.len()].push(request);
        }
        info!(
            gpu_count = self.workers.len(),
            measurement_count = buckets.iter().map(Vec::len).sum::<usize>(),
            "measuring collected GPU node shapes in parallel"
        );
        let harness = &self.harness;
        let measured = self
            .workers
            .par_iter_mut()
            .zip(buckets.into_par_iter())
            .map(|(worker, requests)| {
                let mut completed = Vec::with_capacity(requests.len());
                let mut representative_work_seconds = 0.0;
                for request in requests {
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
    ) -> Result<[u8; 32], GpuMeasurementError> {
        #[derive(Serialize)]
        struct MeasurementKey<'a> {
            kind: &'a NodeKind,
            concrete_argument_types: &'a [ConcreteWireType],
            concrete_output_types: &'a [ConcreteWireType],
            bindings: &'a ParamEnv,
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
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    fn representative_node<'a>(
        node: &'a MeasurementNode<'a>,
        crt_depth: usize,
    ) -> (NodeKind, Vec<ConcreteWireType>, Vec<ConcreteWireType>, f64) {
        let capped_columns = |matrix: &ConcreteMatrixType| {
            (matrix.columns > 1 && matrix_bytes(matrix, crt_depth) > REPRESENTATIVE_MATRIX_BYTES)
                .then_some((1, matrix.columns as f64))
        };
        let capped_rows = |matrix: &ConcreteMatrixType| {
            (matrix.rows > 1 && matrix_bytes(matrix, crt_depth) > REPRESENTATIVE_MATRIX_BYTES)
                .then_some((1, matrix.rows as f64))
        };
        let mut kind = node.kind.clone();
        let mut argument_types = node.concrete_argument_types.clone();
        let mut output_types = node.concrete_output_types.clone();
        let mut scale = 1.0;

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
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
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
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
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
                    return (kind, argument_types, output_types, scale);
                };
                let Some(input) =
                    argument_types.first_mut().and_then(|wire_type| match wire_type {
                        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                            Some(matrix)
                        }
                        _ => None,
                    })
                else {
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
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
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    let Some(input) =
                        argument_types.first_mut().and_then(|wire_type| match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
                            _ => None,
                        })
                    else {
                        return (kind, argument_types, output_types, scale);
                    };
                    input.rows = representative_columns;
                    output.columns = representative_columns;
                    scale = column_scale;
                }
            }
            NodeKind::Tensor => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    for index in [0, 1] {
                        let Some(input) =
                            argument_types.get_mut(index).and_then(|wire_type| match wire_type {
                                ConcreteWireType::Matrix(matrix) |
                                ConcreteWireType::Preimage(matrix) => Some(matrix),
                                _ => None,
                            })
                        else {
                            return (kind, argument_types, output_types, scale);
                        };
                        input.columns = representative_columns;
                    }
                    output.columns = representative_columns;
                    scale = column_scale;
                }
            }
            NodeKind::Concat { axis: ConcatAxis::Rows } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    for wire_type in &mut argument_types {
                        let Some(input) = (match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
                            _ => None,
                        }) else {
                            return (kind, argument_types, output_types, scale);
                        };
                        input.columns = representative_columns;
                    }
                    output.columns = representative_columns;
                    scale = column_scale;
                }
            }
            NodeKind::Concat { axis: ConcatAxis::Columns | ConcatAxis::Diagonal } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                if capped_columns(output).is_some() {
                    let original_columns = output.columns;
                    let mut representative_columns = 0usize;
                    for wire_type in &mut argument_types {
                        let Some(input) = (match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
                            _ => None,
                        }) else {
                            return (kind, argument_types, output_types, scale);
                        };
                        input.columns = 1;
                        representative_columns = representative_columns.saturating_add(1);
                    }
                    output.columns = representative_columns;
                    scale = original_columns as f64 / representative_columns.max(1) as f64;
                }
            }
            NodeKind::GadgetDecompose { .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    let Some(input) =
                        argument_types.first_mut().and_then(|wire_type| match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
                            _ => None,
                        })
                    else {
                        return (kind, argument_types, output_types, scale);
                    };
                    input.columns = representative_columns;
                    output.columns = representative_columns;
                    scale = column_scale;
                }
            }
            NodeKind::MatrixScale { .. } | NodeKind::MatrixNegate => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    let Some(input) =
                        argument_types.first_mut().and_then(|wire_type| match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
                            _ => None,
                        })
                    else {
                        return (kind, argument_types, output_types, scale);
                    };
                    input.columns = representative_columns;
                    output.columns = representative_columns;
                    scale = column_scale;
                }
            }
            NodeKind::MatrixBinary(operation) => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                match operation {
                    MatrixBinaryOp::Add | MatrixBinaryOp::Subtract => {
                        if let Some((representative_columns, column_scale)) = capped_columns(output)
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
                                    return (kind, argument_types, output_types, scale);
                                };
                                input.columns = representative_columns;
                            }
                            output.columns = representative_columns;
                            scale = column_scale;
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
                            return (kind, argument_types, output_types, scale);
                        };
                        if let Some((representative_columns, column_scale)) = capped_columns(rhs) {
                            rhs.columns = representative_columns;
                            output.columns = representative_columns;
                            scale = column_scale;
                        }
                    }
                }
            }
            NodeKind::PreimageSample { matrix_type, .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    let Some(target) =
                        argument_types.get_mut(2).and_then(|wire_type| match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
                            _ => None,
                        })
                    else {
                        return (kind, argument_types, output_types, scale);
                    };
                    target.columns = representative_columns;
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
                }
            }
            NodeKind::CrtRecompose { .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    for wire_type in &mut argument_types {
                        let Some(input) = (match wire_type {
                            ConcreteWireType::Matrix(matrix) |
                            ConcreteWireType::Preimage(matrix) => Some(matrix),
                            _ => None,
                        }) else {
                            return (kind, argument_types, output_types, scale);
                        };
                        input.columns = representative_columns;
                    }
                    output.columns = representative_columns;
                    scale = column_scale;
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
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(input) {
                    input.columns = representative_columns;
                    scale = column_scale;
                }
            }
            NodeKind::PackPolynomialCoefficients { matrix_type, .. } => {
                let Some(output) = output_types.iter_mut().find_map(|wire_type| match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        Some(matrix)
                    }
                    _ => None,
                }) else {
                    return (kind, argument_types, output_types, scale);
                };
                if let Some((representative_columns, column_scale)) = capped_columns(output) {
                    let original_columns = output.columns;
                    let Some(ConcreteWireType::IndexedFamily { count, .. }) =
                        argument_types.first_mut()
                    else {
                        return (kind, argument_types, output_types, scale);
                    };
                    *count = count.div_ceil(original_columns);
                    output.columns = representative_columns;
                    matrix_type.columns = mxx_ir_core::IntExpr::constant(representative_columns);
                    scale = column_scale;
                }
            }
            _ => {}
        }
        (kind, argument_types, output_types, scale)
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
                preimage_max_coefficient_bound,
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
                preimage_max_coefficient_bound.clone(),
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
        if request.preimage_sample {
            info!(
                device_id = worker.device_id,
                scope = ?request.scope,
                node = request.id.0,
                "measuring representative GPU preimage sampler"
            );
        }
        let prepared = Self::prepare(&mut worker.backend, &node, &request.bindings)?;
        let probe = GpuMemoryProbe { device_id: worker.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(harness, &probe, 1, |representative_batch| {
            if operation_error.is_some() {
                return;
            }
            match Self::run_node(
                &mut worker.backend,
                &node,
                &request.bindings,
                representative_batch,
                &prepared,
            ) {
                Ok(outputs) => outputs.iter().for_each(GpuDCRTPolyMatrix::wait_until_ready),
                Err(error) => operation_error = Some(error),
            }
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        let measurement = NodeMeasurement {
            work_seconds: measured.measurement.work_seconds * request.scale,
            latency_seconds: measured.measurement.latency_seconds * request.scale,
            workspace_bytes: measured.measurement.workspace_bytes,
        };
        if request.scale > 1.0 {
            info!(
                device_id = worker.device_id,
                scope = ?request.scope,
                node = request.id.0,
                scale = request.scale,
                representative = ?measured.measurement,
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
                let gadget_layout = gadget_base.as_ref().zip(digit_count);
                (0..batch_size)
                    .map(|_| {
                        backend
                            .sample_hash(&ty, [0x53; 32], tag_prefix, *variant, gadget_layout)
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
                    Some(ConcreteWireType::IndexedFamily { count, .. }) => *count,
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
            NodeKind::TrapdoorPublic |
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
            NodeKind::Select { .. } => Ok(Vec::new()),
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
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. }
        ) {
            return Ok(NodeMeasurement::default());
        }
        let (
            representative_kind,
            representative_argument_types,
            representative_output_types,
            scale,
        ) = Self::representative_node(node, self.crt_depth);
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
        let measurement_key = Self::measurement_key(node, bindings)?;
        if let Some(measurement) = self.measurements.get(&measurement_key) {
            debug!(
                scope = ?node.scope,
                node = node.id.0,
                measurement = ?measurement,
                "reused cached GPU node measurement"
            );
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
            kind: representative_node.kind.clone(),
            concrete_argument_types: representative_node.concrete_argument_types,
            concrete_output_types: representative_node.concrete_output_types,
            bindings: bindings.clone(),
            scale,
            preimage_sample: matches!(node.kind, NodeKind::PreimageSample { .. }),
        });
        Ok(NodeMeasurement::default())
    }

    fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
        match wire_type {
            ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                matrix_bytes(matrix, self.crt_depth)
            }
            ConcreteWireType::Trapdoor { matrix, .. } => matrix_bytes(matrix, self.crt_depth),
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
    use super::{GpuNodeMeasurementBackend, matrix_bytes};
    use crate::MeasurementNode;
    use mxx_ir_core::{
        FrozenGraphScopeId, IntExpr, ParamEnv,
        node::{HashVariant, IndexRange, MatrixBinaryOp, NodeKind},
        types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId},
    };
    use num_bigint::BigInt;

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
            columns: 8_720,
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

        let (kind, _, output_types, scale) =
            GpuNodeMeasurementBackend::representative_node(&node, 40);
        let NodeKind::HashSample { matrix_type, .. } = kind else {
            panic!("hash representative kind");
        };
        let ConcreteWireType::Matrix(output) = &output_types[0] else {
            panic!("hash representative output");
        };
        assert_eq!((output.rows, output.columns), (1, 1));
        assert_eq!(matrix_type.columns, IntExpr::constant(1));
        assert_eq!(scale, 8_720.0);
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

        let (kind, _, output_types, scale) =
            GpuNodeMeasurementBackend::representative_node(&node, 40);
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

        let (kind, argument_types, _, scale) =
            GpuNodeMeasurementBackend::representative_node(&node, 40);
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
        let (_, gadget_arguments, gadget_outputs, gadget_scale) =
            GpuNodeMeasurementBackend::representative_node(&gadget_node, 40);
        let ConcreteWireType::Matrix(gadget_input) = &gadget_arguments[0] else {
            panic!("gadget representative input");
        };
        let ConcreteWireType::Preimage(gadget_output) = &gadget_outputs[0] else {
            panic!("gadget representative output");
        };
        assert_eq!((gadget_input.rows, gadget_input.columns), (1, 1));
        assert_eq!((gadget_output.rows, gadget_output.columns), (80, 1));
        assert_eq!(gadget_scale, 80.0);

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
        let (_, multiply_arguments, multiply_outputs, multiply_scale) =
            GpuNodeMeasurementBackend::representative_node(&multiply_node, 40);
        let ConcreteWireType::Matrix(rhs) = &multiply_arguments[1] else {
            panic!("multiply representative rhs");
        };
        let ConcreteWireType::Matrix(product) = &multiply_outputs[0] else {
            panic!("multiply representative output");
        };
        assert_eq!((rhs.rows, rhs.columns), (80, 1));
        assert_eq!((product.rows, product.columns), (1, 1));
        assert_eq!(multiply_scale, 80.0);
    }

    #[test]
    fn preimage_measurement_uses_one_target_column() {
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

        let (kind, arguments, outputs, scale) =
            GpuNodeMeasurementBackend::representative_node(&node, 40);
        let NodeKind::PreimageSample { matrix_type, .. } = kind else {
            panic!("preimage representative kind");
        };
        let ConcreteWireType::Matrix(target) = &arguments[2] else {
            panic!("preimage representative target");
        };
        let ConcreteWireType::Preimage(output) = &outputs[0] else {
            panic!("preimage representative output");
        };
        assert_eq!(target.columns, 1);
        assert_eq!((output.rows, output.columns), (82, 1));
        assert_eq!(matrix_type.columns, IntExpr::constant(1));
        assert_eq!(scale, 80.0);
    }
}
