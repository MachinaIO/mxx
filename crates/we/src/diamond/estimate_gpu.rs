//! GPU measurements for the concrete operations used by Diamond WE graphs.
//!
//! This module deliberately measures only Diamond-reachable operations through the production
//! runtime backend. It is not a general GPU cost-model framework.

use mxx_bench_estimator::{
    MeasurementBackend, MeasurementNode, NodeMeasurement, harness::MeasurementHarnessConfig,
};
use mxx_ir_core::{
    ParamEnv,
    expr::IntExpr,
    node::{ConstantMatrix, HashVariant, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, MatrixType},
};
use mxx_primitives::{
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::dcrt::gpu::GpuDCRTPolyParams,
    sampler::trapdoor::GpuDCRTTrapdoor,
};
use mxx_runtime::{
    Backend,
    backend::{
        IndexRange, SampleRange,
        poly_gpu::{GpuDcrtBackend, gpu_backend_on},
    },
};
use num_bigint::BigInt;
use num_traits::One;
use std::{collections::BTreeMap, sync::Arc, time::Instant};
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum DiamondGpuMeasurementError {
    #[error("Diamond GPU measurement backend failed: {0}")]
    Backend(String),
    #[error("unsupported Diamond GPU measurement at {scope:?} node {node:?}: {kind:?}")]
    Unsupported {
        scope: mxx_ir_core::FrozenGraphScopeId,
        node: mxx_ir_core::NodeId,
        kind: NodeKind,
    },
    #[error("Diamond GPU measurement requires a matrix argument")]
    MatrixArgument,
    #[error("Diamond GPU measurement requires a trapdoor argument")]
    TrapdoorArgument,
    #[error("Diamond GPU measurement expression failed: {0}")]
    Expression(String),
}

enum ReadyOutput {
    Matrix(GpuDCRTPolyMatrix),
    Trapdoor { public: GpuDCRTPolyMatrix, secret: GpuDCRTTrapdoor },
    Host,
}

impl ReadyOutput {
    fn wait_until_ready(self) {
        match self {
            Self::Matrix(matrix) => matrix.wait_until_ready(),
            Self::Trapdoor { public, secret } => {
                public.wait_until_ready();
                secret.wait_until_ready();
            }
            Self::Host => {}
        }
    }
}

/// Measures one primitive invocation on every production placement and uses the slowest placement.
/// A primitive's event-complete wall time is both work and latency; loop multiplicity is modeled by
/// `mxx-bench-estimator`.
pub struct DiamondGpuMeasurementBackend {
    backend: GpuDcrtBackend,
    harness: MeasurementHarnessConfig,
    matrix_cache: BTreeMap<(usize, ConcreteMatrixType), Arc<GpuDCRTPolyMatrix>>,
    trapdoor_cache:
        BTreeMap<(usize, ConcreteWireType), (Arc<GpuDCRTPolyMatrix>, Arc<GpuDCRTTrapdoor>)>,
    measurements: BTreeMap<String, NodeMeasurement>,
    crt_depth: usize,
}

impl DiamondGpuMeasurementBackend {
    pub fn new(
        parameters: GpuDCRTPolyParams,
        device_ids: &[i32],
        harness: MeasurementHarnessConfig,
    ) -> Self {
        Self {
            crt_depth: parameters.crt_depth(),
            backend: gpu_backend_on([parameters], device_ids.iter().copied()),
            harness,
            matrix_cache: BTreeMap::new(),
            trapdoor_cache: BTreeMap::new(),
            measurements: BTreeMap::new(),
        }
    }

    fn backend_error(error: impl std::fmt::Display) -> DiamondGpuMeasurementError {
        DiamondGpuMeasurementError::Backend(error.to_string())
    }

    fn matrix(
        &mut self,
        ty: &ConcreteMatrixType,
    ) -> Result<Arc<GpuDCRTPolyMatrix>, DiamondGpuMeasurementError> {
        let key = (self.backend.active_placement(), ty.clone());
        if !self.matrix_cache.contains_key(&key) {
            let value = self
                .backend
                .sample_uniform(
                    ty,
                    &SampleRange { minimum: BigInt::from(0), maximum: &ty.modulus - BigInt::one() },
                )
                .map_err(Self::backend_error)?;
            value.wait_until_ready();
            self.matrix_cache.insert(key.clone(), Arc::new(value));
        }
        Ok(Arc::clone(self.matrix_cache.get(&key).expect("inserted measurement matrix")))
    }

    fn trapdoor(
        &mut self,
        ty: &ConcreteWireType,
        bindings: &ParamEnv,
    ) -> Result<(Arc<GpuDCRTPolyMatrix>, Arc<GpuDCRTTrapdoor>), DiamondGpuMeasurementError> {
        let ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count, .. } = ty else {
            return Err(DiamondGpuMeasurementError::TrapdoorArgument);
        };
        let key = (self.backend.active_placement(), ty.clone());
        if !self.trapdoor_cache.contains_key(&key) {
            let sigma = sigma
                .evaluate_f64(bindings)
                .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
            let (public, secret) = self
                .backend
                .sample_trapdoor(matrix, sigma, gadget_base, *digit_count)
                .map_err(Self::backend_error)?;
            public.wait_until_ready();
            secret.wait_until_ready();
            self.trapdoor_cache.insert(key.clone(), (Arc::new(public), Arc::new(secret)));
        }
        let (public, secret) =
            self.trapdoor_cache.get(&key).expect("inserted measurement trapdoor");
        Ok((Arc::clone(public), Arc::clone(secret)))
    }

    fn matrix_argument<'a>(
        node: &'a MeasurementNode<'_>,
        index: usize,
    ) -> Result<&'a ConcreteMatrixType, DiamondGpuMeasurementError> {
        node.argument_types
            .get(index)
            .and_then(ConcreteWireType::matrix_type)
            .ok_or(DiamondGpuMeasurementError::MatrixArgument)
    }

    fn matrix_output<'a>(
        node: &'a MeasurementNode<'_>,
    ) -> Result<&'a ConcreteMatrixType, DiamondGpuMeasurementError> {
        node.concrete_output_types
            .first()
            .and_then(ConcreteWireType::matrix_type)
            .ok_or(DiamondGpuMeasurementError::MatrixArgument)
    }

    fn evaluate_matrix_type(
        matrix_type: &MatrixType,
        bindings: &ParamEnv,
    ) -> Result<ConcreteMatrixType, DiamondGpuMeasurementError> {
        let evaluate_positive =
            |expression: &IntExpr, label: &str| -> Result<usize, DiamondGpuMeasurementError> {
                let value = expression
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                let value = usize::try_from(value).map_err(|_| {
                    DiamondGpuMeasurementError::Expression(format!(
                        "constant-polynomial lift {label} is not usize"
                    ))
                })?;
                if value == 0 {
                    return Err(DiamondGpuMeasurementError::Expression(format!(
                        "constant-polynomial lift {label} must be positive"
                    )));
                }
                Ok(value)
            };
        let modulus = matrix_type
            .modulus
            .evaluate(bindings)
            .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
        if modulus <= BigInt::one() {
            return Err(DiamondGpuMeasurementError::Expression(
                "constant-polynomial lift modulus must exceed one".to_owned(),
            ));
        }
        Ok(ConcreteMatrixType {
            modulus,
            ring_dimension: evaluate_positive(&matrix_type.ring_dimension, "ring dimension")?,
            rows: evaluate_positive(&matrix_type.rows, "row count")?,
            columns: evaluate_positive(&matrix_type.columns, "column count")?,
        })
    }

    fn cache_key(node: &MeasurementNode<'_>, bindings: &ParamEnv) -> String {
        format!(
            "{:?}:{:?}:{:?}:{:?}:{:?}",
            node.scope, node.kind, node.argument_kinds, node.argument_types, bindings
        )
    }

    fn measure_placements<F>(
        &mut self,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
        mut operation: F,
    ) -> Result<NodeMeasurement, DiamondGpuMeasurementError>
    where
        F: FnMut(&mut Self) -> Result<ReadyOutput, DiamondGpuMeasurementError>,
    {
        let original = self.backend.active_placement();
        let mut slowest = 0.0f64;
        for placement in 0..self.backend.placement_count() {
            assert!(self.backend.set_active_placement(placement));
            // Build and fence all representative inputs before warm-up and timing.
            self.prepare_inputs(node, bindings)?;
            for _ in 0..self.harness.warm_up_iterations {
                operation(self)?.wait_until_ready();
            }
            if self.harness.measured_iterations == 0 {
                return Err(DiamondGpuMeasurementError::Expression(
                    "measurement iteration count must be positive".to_owned(),
                ));
            }
            let started = Instant::now();
            for _ in 0..self.harness.measured_iterations {
                operation(self)?.wait_until_ready();
            }
            let seconds = started.elapsed().as_secs_f64() / self.harness.measured_iterations as f64;
            slowest = slowest.max(seconds);
        }
        assert!(self.backend.set_active_placement(original));
        Ok(NodeMeasurement { work_seconds: slowest, latency_seconds: slowest, workspace_bytes: 0 })
    }

    fn prepare_inputs(
        &mut self,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<(), DiamondGpuMeasurementError> {
        for ty in node.argument_types {
            match ty {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                    self.matrix(matrix)?.wait_until_ready();
                }
                ConcreteWireType::Trapdoor { .. } => {
                    let (public, secret) = self.trapdoor(ty, bindings)?;
                    public.wait_until_ready();
                    secret.wait_until_ready();
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn measure_node(
        &mut self,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<NodeMeasurement, DiamondGpuMeasurementError> {
        match node.kind {
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
            NodeKind::TrapdoorPublic |
            NodeKind::SubgraphCall(_) |
            NodeKind::SequentialLoop(_) |
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic { .. } |
            NodeKind::Select { .. } => Ok(NodeMeasurement::default()),
            NodeKind::ConstantMatrix { value, .. } => {
                let output = Self::matrix_output(node)?.clone();
                self.measure_placements(node, bindings, |this| {
                    this.backend
                        .constant_matrix(&output, value, bindings)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::GadgetTrapdoor { base, .. } => {
                let output = Self::matrix_output(node)?.clone();
                self.measure_placements(node, bindings, |this| {
                    this.backend
                        .constant_matrix(
                            &output,
                            &ConstantMatrix::Gadget { base: base.clone(), small: false },
                            bindings,
                        )
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::MatrixBinary(operation) => {
                let left = Self::matrix_argument(node, 0)?.clone();
                let right = Self::matrix_argument(node, 1)?.clone();
                self.measure_placements(node, bindings, |this| {
                    let left = this.matrix(&left)?;
                    let right = this.matrix(&right)?;
                    let value = match operation {
                        MatrixBinaryOp::Add => this.backend.add(&left, &right),
                        MatrixBinaryOp::Subtract => this.backend.sub(&left, &right),
                        MatrixBinaryOp::Multiply => this.backend.multiply(&left, &right),
                    }
                    .map_err(Self::backend_error)?;
                    Ok(ReadyOutput::Matrix(value))
                })
            }
            NodeKind::MatrixNegate => {
                let input = Self::matrix_argument(node, 0)?.clone();
                self.measure_placements(node, bindings, |this| {
                    let input = this.matrix(&input)?;
                    this.backend
                        .negate(&input)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::MatrixScale { scalar } => {
                let input = Self::matrix_argument(node, 0)?.clone();
                let scalar = scalar
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                self.measure_placements(node, bindings, |this| {
                    let input = this.matrix(&input)?;
                    this.backend
                        .scale_integer(&input, &scalar)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::Transpose => {
                let input = Self::matrix_argument(node, 0)?.clone();
                self.measure_placements(node, bindings, |this| {
                    let input = this.matrix(&input)?;
                    this.backend
                        .transpose(&input)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::Slice { rows, columns } => {
                let input = Self::matrix_argument(node, 0)?.clone();
                let rows = rows
                    .as_ref()
                    .map(|range| {
                        Ok(IndexRange {
                            start: range
                                .start
                                .evaluate(bindings)
                                .map_err(|error| {
                                    DiamondGpuMeasurementError::Expression(error.to_string())
                                })?
                                .try_into()
                                .map_err(|_| {
                                    DiamondGpuMeasurementError::Expression(
                                        "slice row start is not usize".to_owned(),
                                    )
                                })?,
                            end: range
                                .end
                                .evaluate(bindings)
                                .map_err(|error| {
                                    DiamondGpuMeasurementError::Expression(error.to_string())
                                })?
                                .try_into()
                                .map_err(|_| {
                                    DiamondGpuMeasurementError::Expression(
                                        "slice row end is not usize".to_owned(),
                                    )
                                })?,
                        })
                    })
                    .transpose()?;
                let columns = columns
                    .as_ref()
                    .map(|range| {
                        Ok(IndexRange {
                            start: range
                                .start
                                .evaluate(bindings)
                                .map_err(|error| {
                                    DiamondGpuMeasurementError::Expression(error.to_string())
                                })?
                                .try_into()
                                .map_err(|_| {
                                    DiamondGpuMeasurementError::Expression(
                                        "slice column start is not usize".to_owned(),
                                    )
                                })?,
                            end: range
                                .end
                                .evaluate(bindings)
                                .map_err(|error| {
                                    DiamondGpuMeasurementError::Expression(error.to_string())
                                })?
                                .try_into()
                                .map_err(|_| {
                                    DiamondGpuMeasurementError::Expression(
                                        "slice column end is not usize".to_owned(),
                                    )
                                })?,
                        })
                    })
                    .transpose()?;
                self.measure_placements(node, bindings, |this| {
                    let input = this.matrix(&input)?;
                    this.backend
                        .slice(&input, rows.as_ref(), columns.as_ref())
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::Tensor => {
                let left = Self::matrix_argument(node, 0)?.clone();
                let right = Self::matrix_argument(node, 1)?.clone();
                self.measure_placements(node, bindings, |this| {
                    let left = this.matrix(&left)?;
                    let right = this.matrix(&right)?;
                    this.backend
                        .tensor(&left, &right)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::Concat { axis } => {
                let inputs = node
                    .argument_types
                    .iter()
                    .filter_map(ConcreteWireType::matrix_type)
                    .cloned()
                    .collect::<Vec<_>>();
                self.measure_placements(node, bindings, |this| {
                    let values =
                        inputs.iter().map(|ty| this.matrix(ty)).collect::<Result<Vec<_>, _>>()?;
                    let refs = values.iter().map(Arc::as_ref).collect::<Vec<_>>();
                    this.backend
                        .concat(&refs, *axis)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::UniformResidueSample { .. } => {
                let output = Self::matrix_output(node)?.clone();
                let range = SampleRange {
                    minimum: BigInt::from(0),
                    maximum: &output.modulus - BigInt::one(),
                };
                self.measure_placements(node, bindings, |this| {
                    this.backend
                        .sample_uniform(&output, &range)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::UniformIntervalSample { range, .. } => {
                let output = Self::matrix_output(node)?.clone();
                let range = SampleRange {
                    minimum: range.minimum.evaluate(bindings).map_err(|error| {
                        DiamondGpuMeasurementError::Expression(error.to_string())
                    })?,
                    maximum: range.maximum.evaluate(bindings).map_err(|error| {
                        DiamondGpuMeasurementError::Expression(error.to_string())
                    })?,
                };
                self.measure_placements(node, bindings, |this| {
                    this.backend
                        .sample_uniform(&output, &range)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                let output = Self::matrix_output(node)?.clone();
                let sigma = sigma
                    .evaluate_f64(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                let bound = max_coefficient_bound
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                self.measure_placements(node, bindings, |this| {
                    this.backend
                        .sample_gaussian(&output, sigma, &bound)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::HashSample { tag_prefix, .. } => {
                let output = Self::matrix_output(node)?.clone();
                let tag = tag_prefix.clone();
                self.measure_placements(node, bindings, |this| {
                    this.backend
                        .sample_hash(&output, [0x5a; 32], &tag, HashVariant::Plain, None)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } => {
                let output = Self::matrix_output(node)?.clone();
                let sigma = sigma
                    .evaluate_f64(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                let base = gadget_base
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                let digits = digit_count
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?
                    .try_into()
                    .map_err(|_| {
                        DiamondGpuMeasurementError::Expression(
                            "digit count is not usize".to_owned(),
                        )
                    })?;
                self.measure_placements(node, bindings, |this| {
                    let (public, secret) = this
                        .backend
                        .sample_trapdoor(&output, sigma, &base, digits)
                        .map_err(Self::backend_error)?;
                    Ok(ReadyOutput::Trapdoor { public, secret })
                })
            }
            NodeKind::PreimageSample { max_coefficient_bound, .. } => {
                let trapdoor_type = node
                    .argument_types
                    .get(1)
                    .ok_or(DiamondGpuMeasurementError::TrapdoorArgument)?
                    .clone();
                let output = Self::matrix_output(node)?.clone();
                let bound = max_coefficient_bound
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                let gadget_trapdoor =
                    matches!(node.argument_kinds.get(1), Some(NodeKind::GadgetTrapdoor { .. }));
                self.measure_placements(node, bindings, |this| {
                    if gadget_trapdoor {
                        let target_type = Self::matrix_argument(node, 2)?.clone();
                        let target = this.matrix(&target_type)?;
                        let ConcreteWireType::Trapdoor { gadget_base, digit_count, .. } =
                            &trapdoor_type
                        else {
                            return Err(DiamondGpuMeasurementError::TrapdoorArgument);
                        };
                        this.backend
                            .validate_gadget_layout(&target_type, gadget_base, *digit_count, false)
                            .map_err(Self::backend_error)?;
                        return this
                            .backend
                            .gadget_decompose(&target, false)
                            .map(ReadyOutput::Matrix)
                            .map_err(Self::backend_error);
                    }
                    let (public, secret) = this.trapdoor(&trapdoor_type, bindings)?;
                    let target_ty = ConcreteMatrixType {
                        modulus: output.modulus.clone(),
                        ring_dimension: output.ring_dimension,
                        rows: public.row_size(),
                        columns: output.columns,
                    };
                    let target = this.matrix(&target_ty)?;
                    let ConcreteWireType::Trapdoor { sigma, gadget_base, digit_count, .. } =
                        &trapdoor_type
                    else {
                        return Err(DiamondGpuMeasurementError::TrapdoorArgument)
                    };
                    let sigma = sigma.evaluate_f64(bindings).map_err(|error| {
                        DiamondGpuMeasurementError::Expression(error.to_string())
                    })?;
                    this.backend
                        .sample_preimage(
                            &output,
                            sigma,
                            gadget_base,
                            *digit_count,
                            &bound,
                            &secret,
                            &public,
                            &target,
                        )
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let input_type = Self::matrix_argument(node, 0)?.clone();
                let base = base
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                let digit_count = digit_count
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?
                    .try_into()
                    .map_err(|_| {
                        DiamondGpuMeasurementError::Expression(
                            "gadget decomposition digit count is not usize".to_owned(),
                        )
                    })?;
                self.measure_placements(node, bindings, |this| {
                    let input = this.matrix(&input_type)?;
                    this.backend
                        .validate_gadget_layout(&input_type, &base, digit_count, *small)
                        .map_err(Self::backend_error)?;
                    this.backend
                        .gadget_decompose(&input, *small)
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::ExtractCoefficient { position, .. } => {
                let input = Self::matrix_argument(node, 0)?.clone();
                let position = position
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?
                    .try_into()
                    .map_err(|_| {
                        DiamondGpuMeasurementError::Expression(
                            "coefficient position is not usize".to_owned(),
                        )
                    })?;
                self.measure_placements(node, bindings, |this| {
                    let input = this.matrix(&input)?;
                    this.backend
                        .extract_coefficient(&input, position)
                        .map_err(Self::backend_error)?;
                    Ok(ReadyOutput::Host)
                })
            }
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type } => {
                let output = Self::evaluate_matrix_type(matrix_type, bindings)?;
                self.measure_placements(node, bindings, |this| {
                    let identity = this
                        .backend
                        .constant_matrix(
                            &output,
                            &mxx_ir_core::node::ConstantMatrix::Identity,
                            bindings,
                        )
                        .map_err(Self::backend_error)?;
                    this.backend
                        .scale_integer(&identity, &BigInt::from(0))
                        .map(ReadyOutput::Matrix)
                        .map_err(Self::backend_error)
                })
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, .. } => {
                let input = Self::matrix_argument(node, 0)?.clone();
                let plaintext = plaintext_modulus
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?;
                let length = length
                    .evaluate(bindings)
                    .map_err(|error| DiamondGpuMeasurementError::Expression(error.to_string()))?
                    .try_into()
                    .map_err(|_| {
                        DiamondGpuMeasurementError::Expression(
                            "decode length is not usize".to_owned(),
                        )
                    })?;
                self.measure_placements(node, bindings, |this| {
                    let input = this.matrix(&input)?;
                    this.backend
                        .threshold_decode(&input, &plaintext, length)
                        .map_err(Self::backend_error)?;
                    Ok(ReadyOutput::Host)
                })
            }
            _ => Err(DiamondGpuMeasurementError::Unsupported {
                scope: node.scope.clone(),
                node: node.id,
                kind: node.kind.clone(),
            }),
        }
    }
}

impl MeasurementBackend for DiamondGpuMeasurementBackend {
    type Error = DiamondGpuMeasurementError;

    fn measure(
        &mut self,
        _graph: &str,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<NodeMeasurement, Self::Error> {
        let key = Self::cache_key(node, bindings);
        if let Some(measurement) = self.measurements.get(&key) {
            debug!(cache_key = %key, "reusing Diamond GPU primitive measurement");
            return Ok(measurement.clone());
        }
        info!(scope = ?node.scope, node = node.id.0, kind = ?node.kind, "measuring Diamond GPU primitive");
        let started = Instant::now();
        let measurement = self.measure_node(node, bindings)?;
        info!(
            scope = ?node.scope,
            node = node.id.0,
            elapsed_seconds = started.elapsed().as_secs_f64(),
            work_seconds = measurement.work_seconds,
            latency_seconds = measurement.latency_seconds,
            workspace_unmeasured = true,
            "finished Diamond GPU primitive measurement"
        );
        self.measurements.insert(key, measurement.clone());
        Ok(measurement)
    }

    fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
        let matrix_bytes = |matrix: &ConcreteMatrixType| {
            (matrix.rows as u64)
                .saturating_mul(matrix.columns as u64)
                .saturating_mul(matrix.ring_dimension as u64)
                .saturating_mul(self.crt_depth as u64)
                .saturating_mul(8)
        };
        match wire_type {
            ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                matrix_bytes(matrix)
            }
            ConcreteWireType::Trapdoor { matrix, .. } => matrix_bytes(matrix).saturating_mul(3),
            _ => 0,
        }
    }
}
