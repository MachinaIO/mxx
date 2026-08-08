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
    node::{ConstantMatrix, MatrixBinaryOp, NodeKind},
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
use num_traits::ToPrimitive;
use std::{fmt, sync::Arc};
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

impl MemoryProbe for GpuMemoryProbe {
    type Error = GpuMeasurementError;

    fn current_bytes(&self) -> Result<u64, Self::Error> {
        let memory = gpu_memory_info(self.device_id).map_err(GpuMeasurementError)?;
        u64::try_from(memory.total.saturating_sub(memory.free))
            .map_err(|_| GpuMeasurementError("GPU memory usage exceeds u64".to_owned()))
    }
}

struct PreparedMeasurement {
    arguments: Vec<Option<GpuDCRTPolyMatrix>>,
    preimage_trapdoor: Option<(GpuDCRTPolyMatrix, GpuDCRTTrapdoor, f64, BigInt, usize, BigInt)>,
}

pub struct GpuNodeMeasurementBackend {
    backend: GpuDcrtBackend,
    device_id: i32,
    harness: MeasurementHarnessConfig,
    crt_depth: usize,
}

impl GpuNodeMeasurementBackend {
    /// Creates a representative GPU measurement backend for validated IR nodes.
    pub fn new(
        backend: GpuDcrtBackend,
        device_id: i32,
        harness: MeasurementHarnessConfig,
        crt_depth: usize,
    ) -> Self {
        Self { backend, device_id, harness, crt_depth }
    }

    fn prepare(
        &mut self,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<PreparedMeasurement, GpuMeasurementError> {
        let arguments = node
            .concrete_argument_types
            .iter()
            .map(|wire_type| match wire_type {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => self
                    .backend
                    .constant_matrix(&matrix, &ConstantMatrix::Zero, bindings)
                    .map(Some)
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
            let (public, trapdoor) = self
                .backend
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

    fn run_node(
        backend: &mut GpuDcrtBackend,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        batch_size: usize,
        prepared: &PreparedMeasurement,
    ) -> Result<Vec<GpuDCRTPolyMatrix>, GpuMeasurementError> {
        let matrix = |index: usize| {
            prepared.arguments.get(index).and_then(Option::as_ref).cloned().ok_or_else(|| {
                GpuMeasurementError(format!("node {:?} argument {index} is not a matrix", node.id))
            })
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
                    .map(|_| Ok((matrix(0)?, matrix(1)?)))
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                match operation {
                    MatrixBinaryOp::Add => backend.add_batch(inputs),
                    MatrixBinaryOp::Subtract => backend.sub_batch(inputs),
                    MatrixBinaryOp::Multiply => backend.multiply_batch(inputs),
                }
                .map_err(backend_error)
            }
            NodeKind::MatrixNegate => backend
                .negate_batch((0..batch_size).map(|_| matrix(0)).collect::<Result<Vec<_>, _>>()?)
                .map_err(backend_error),
            NodeKind::MatrixScale { scalar } => {
                let scalar = scalar
                    .evaluate(bindings)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                backend
                    .scale_integer_batch(
                        (0..batch_size)
                            .map(|_| Ok((matrix(0)?, scalar.clone())))
                            .collect::<Result<Vec<_>, GpuMeasurementError>>()?,
                    )
                    .map_err(backend_error)
            }
            NodeKind::Transpose => (0..batch_size)
                .map(|_| backend.transpose(&matrix(0)?).map_err(backend_error))
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
                            .slice(&matrix(0)?, rows.as_ref(), columns.as_ref())
                            .map_err(backend_error)
                    })
                    .collect()
            }
            NodeKind::Tensor => (0..batch_size)
                .map(|_| backend.tensor(&matrix(0)?, &matrix(1)?).map_err(backend_error))
                .collect(),
            NodeKind::Concat { axis } => {
                let inputs = prepared
                    .arguments
                    .iter()
                    .map(|value| {
                        value.as_ref().ok_or_else(|| {
                            GpuMeasurementError("concat argument is not a matrix".to_owned())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (0..batch_size)
                    .map(|_| backend.concat(&inputs, *axis).map_err(backend_error))
                    .collect()
            }
            NodeKind::Reshape { rows, columns } => {
                let rows = evaluate_usize(rows)?;
                let columns = evaluate_usize(columns)?;
                (0..batch_size)
                    .map(|_| backend.reshape(&matrix(0)?, rows, columns).map_err(backend_error))
                    .collect()
            }
            NodeKind::UniformSample { range, .. } => {
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
            NodeKind::HashSample { variant, tag_prefix, .. } => {
                let ty = output_matrix_type()?;
                (0..batch_size)
                    .map(|_| {
                        backend
                            .sample_hash(&ty, [0x53; 32], tag_prefix, *variant)
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
                let target = matrix(2)?;
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
                            &target,
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
                                    target: Arc::new(target.clone()),
                                })
                                .collect(),
                        )
                        .map_err(backend_error)
                }
            }
            NodeKind::GadgetDecompose { small, .. } => (0..batch_size)
                .map(|_| backend.gadget_decompose(&matrix(0)?, *small).map_err(backend_error))
                .collect(),
            NodeKind::ExtractCoefficient { position } => {
                let position = evaluate_usize(position)?;
                for _ in 0..batch_size {
                    backend.extract_coefficient(&matrix(0)?, position).map_err(backend_error)?;
                }
                Ok(Vec::new())
            }
            NodeKind::ConstantCoefficient { position } => {
                let ty = output_matrix_type()?;
                let position = evaluate_usize(position)?;
                (0..batch_size)
                    .map(|_| {
                        let coefficient = backend
                            .extract_coefficient(&matrix(0)?, position)
                            .map_err(backend_error)?;
                        let identity = backend
                            .constant_matrix(&ty, &ConstantMatrix::Identity, bindings)
                            .map_err(backend_error)?;
                        backend.scale_integer(&identity, &coefficient).map_err(backend_error)
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
                        .threshold_decode(&matrix(0)?, &modulus, length)
                        .map_err(backend_error)?;
                }
                Ok(Vec::new())
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
                let levels = prepared
                    .arguments
                    .iter()
                    .map(|value| {
                        value.clone().ok_or_else(|| {
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
        let preimage_sample = matches!(node.kind, NodeKind::PreimageSample { .. });
        if preimage_sample {
            info!(
                scope = ?node.scope,
                node = node.id.0,
                "measuring representative GPU preimage sampler"
            );
        }
        let prepared = self.prepare(node, bindings)?;
        let probe = GpuMemoryProbe { device_id: self.device_id };
        let mut operation_error = None;
        let measured = measure_batch_operation(&self.harness, &probe, 1, |representative_batch| {
            if operation_error.is_some() {
                return;
            }
            match Self::run_node(&mut self.backend, node, bindings, representative_batch, &prepared)
            {
                Ok(outputs) => outputs.iter().for_each(GpuDCRTPolyMatrix::wait_until_ready),
                Err(error) => operation_error = Some(error),
            }
        })
        .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(error) = operation_error {
            return Err(error);
        }
        if preimage_sample {
            info!(
                scope = ?node.scope,
                node = node.id.0,
                work_seconds = measured.measurement.work_seconds,
                latency_seconds = measured.measurement.latency_seconds,
                workspace_bytes = measured.measurement.workspace_bytes,
                "measured representative GPU preimage sampler"
            );
        }
        debug!(
            scope = ?node.scope,
            node = node.id.0,
            batch_size = 1,
            measurement = ?measured.measurement,
            "cached GPU node measurement"
        );
        Ok(measured.measurement)
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
    use super::matrix_bytes;
    use mxx_ir_core::types::ConcreteMatrixType;
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
}
