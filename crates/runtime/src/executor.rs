use crate::{
    artifact::{ArtifactKey, ArtifactStore},
    backend::{Backend, PreimageRequest, RuntimeValue},
    liveness,
    transcript::{DrawSite, RecordedValue, SamplingMode, TranscriptError},
};
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::ProductionId,
    expr::euclidean_div_rem,
    graph::Graph,
    node::{IntBinaryOp, IntCompareOp, MatrixBinaryOp, Node, NodeKind, RealBinaryOp},
    types::{
        ConcreteMatrixType, ConcreteWireType, InstantiationFrame, NodeId, Port, WireId, WireRef,
    },
};
use num_bigint::{BigInt, Sign};
use num_traits::{One, ToPrimitive, Zero};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

pub struct ExecutionResult<B: Backend> {
    pub outputs: BTreeMap<String, RuntimeValue<B>>,
    pub output_families: BTreeMap<String, Vec<RuntimeValue<B>>>,
    pub production_id: Option<ProductionId>,
}

pub type ExecutionTrace<B> = BTreeMap<WireId, RuntimeValue<B>>;

type TrapdoorParts<B> = (
    Option<<B as Backend>::Trapdoor>,
    <B as Backend>::Matrix,
    ConcreteMatrixType,
    f64,
    Option<bool>,
);

struct InstanceResult<B: Backend> {
    outputs: BTreeMap<String, RuntimeValue<B>>,
    output_families: BTreeMap<String, Vec<RuntimeValue<B>>>,
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("backend operation failed: {0}")]
    Backend(String),
    #[error("artifact operation failed: {0}")]
    Artifact(String),
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
    #[error("input {0} was not provided")]
    MissingInput(String),
    #[error("wire {0:?} is unavailable")]
    MissingWire(WireRef),
    #[error("wire {0:?} has the wrong runtime value kind")]
    ValueKind(WireRef),
    #[error("integer division by zero at node {0:?}")]
    DivisionByZero(NodeId),
    #[error("invalid real operation at node {0:?}")]
    InvalidRealOperation(NodeId),
    #[error("select index {index} is outside [0, {count}) at node {node:?}")]
    SelectIndexOutOfRange { node: NodeId, index: BigInt, count: usize },
    #[error("subgraph {name} does not exist at node {node:?}")]
    MissingSubgraph { node: NodeId, name: String },
    #[error("validated wire metadata is missing for {0:?}")]
    MissingMetadata(WireId),
    #[error("runtime expression failed at node {node:?}: {message}")]
    Expression { node: NodeId, message: String },
    #[error("manifest operation failed: {0}")]
    Manifest(String),
}

pub fn execute<B, S>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: BTreeMap<String, RuntimeValue<B>>,
    artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>,
) -> Result<ExecutionResult<B>, ExecutionError>
where
    B: Backend,
    S: ArtifactStore,
{
    execute_internal(validated, backend, inputs, artifact_store, sampling_mode, false)
        .map(|(result, _)| result)
}

/// Executes a graph while retaining every intermediate wire value. This is
/// intended for diagnostics and optional analysis; ordinary execution keeps
/// using the liveness drop schedule without retaining a trace.
pub fn execute_with_trace<B, S>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: BTreeMap<String, RuntimeValue<B>>,
    artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>,
) -> Result<(ExecutionResult<B>, ExecutionTrace<B>), ExecutionError>
where
    B: Backend,
    S: ArtifactStore,
{
    execute_internal(validated, backend, inputs, artifact_store, sampling_mode, true)
}

fn execute_internal<B, S>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: BTreeMap<String, RuntimeValue<B>>,
    artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>,
    capture_trace: bool,
) -> Result<(ExecutionResult<B>, ExecutionTrace<B>), ExecutionError>
where
    B: Backend,
    S: ArtifactStore,
{
    let mut executor = Executor {
        validated,
        backend,
        artifact_store,
        sampling_mode,
        trace: capture_trace.then(BTreeMap::new),
    };
    let instance =
        executor.execute_instance(&validated.source, &validated.bindings, Vec::new(), inputs)?;
    let production_id = executor.persist_outputs(&instance.outputs, &instance.output_families)?;
    let result = ExecutionResult {
        outputs: instance.outputs,
        output_families: instance.output_families,
        production_id,
    };
    Ok((result, executor.trace.take().unwrap_or_default()))
}

struct Executor<'a, B: Backend, S: ArtifactStore> {
    validated: &'a ValidatedGraph,
    backend: &'a mut B,
    artifact_store: &'a mut S,
    sampling_mode: SamplingMode<'a>,
    trace: Option<ExecutionTrace<B>>,
}

impl<B, S> Executor<'_, B, S>
where
    B: Backend,
    S: ArtifactStore,
{
    fn execute_instance(
        &mut self,
        graph: &Graph,
        env: &ParamEnv,
        path: Vec<InstantiationFrame>,
        inputs: BTreeMap<String, RuntimeValue<B>>,
    ) -> Result<InstanceResult<B>, ExecutionError> {
        self.execute_instances_batch(graph, vec![env.clone()], vec![path], vec![inputs])
            .map(|mut instances| instances.pop().expect("single execution returns one instance"))
    }

    fn execute_instances_batch(
        &mut self,
        graph: &Graph,
        envs: Vec<ParamEnv>,
        paths: Vec<Vec<InstantiationFrame>>,
        inputs: Vec<BTreeMap<String, RuntimeValue<B>>>,
    ) -> Result<Vec<InstanceResult<B>>, ExecutionError> {
        debug_assert_eq!(envs.len(), paths.len());
        debug_assert_eq!(envs.len(), inputs.len());
        if envs.is_empty() {
            return Ok(Vec::new());
        }
        let schedule = liveness::analyze(graph);
        let mut values = (0..envs.len())
            .map(|_| BTreeMap::<WireRef, RuntimeValue<B>>::new())
            .collect::<Vec<_>>();
        for (position, node) in graph.nodes.iter().enumerate() {
            if matches!(node.kind, NodeKind::PreimageSample { .. }) && envs.len() > 1 {
                self.execute_preimage_batch(&paths, node, &mut values)?;
            } else if matches!(node.kind, NodeKind::Select { .. }) {
                for index in 0..envs.len() {
                    self.execute_select(
                        &envs[index],
                        node,
                        &schedule,
                        position,
                        &mut values[index],
                    )?;
                }
            } else {
                for index in 0..envs.len() {
                    self.execute_node(
                        graph,
                        &envs[index],
                        &paths[index],
                        node,
                        &inputs[index],
                        &mut values[index],
                    )?;
                }
            }
            for index in 0..envs.len() {
                if let Some(trace) = &mut self.trace {
                    for (wire, value) in
                        values[index].iter().filter(|(wire, _)| wire.node == node.id)
                    {
                        trace.insert(
                            WireId { instantiation_path: paths[index].clone(), wire: *wire },
                            value.clone(),
                        );
                    }
                }
                for argument in &node.args {
                    if schedule.last_use.get(argument) == Some(&position) &&
                        !schedule.outputs.contains(argument)
                    {
                        values[index].remove(argument);
                    }
                }
            }
        }
        let mut instances = Vec::with_capacity(values.len());
        for mut instance_values in values {
            let mut outputs = BTreeMap::new();
            for (name, wire) in &graph.outputs {
                outputs.insert(name.clone(), self.materialize(&mut instance_values, *wire)?);
            }
            let mut output_families = BTreeMap::new();
            for node in &graph.nodes {
                let NodeKind::Output { name } = &node.kind else {
                    continue;
                };
                if node.args.len() <= 1 {
                    continue;
                }
                let members = (0..node.args.len())
                    .map(|port| {
                        self.materialize(
                            &mut instance_values,
                            WireRef { node: node.id, port: Port(port as u32) },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                output_families.insert(name.clone(), members);
            }
            instances.push(InstanceResult { outputs, output_families });
        }
        Ok(instances)
    }

    fn persist_outputs(
        &mut self,
        outputs: &BTreeMap<String, RuntimeValue<B>>,
        output_families: &BTreeMap<String, Vec<RuntimeValue<B>>>,
    ) -> Result<Option<ProductionId>, ExecutionError> {
        let spec_hash =
            mxx_ir_core::encoding::spec_hash(&self.validated.source, &self.validated.bindings)
                .map_err(|error| ExecutionError::Manifest(error.to_string()))?;
        let production = mxx_ir_core::artifact::production_id(spec_hash, rand::random());
        let mut artifacts = BTreeMap::new();
        'outputs: for (name, output_wire) in &self.validated.outputs {
            if let Some(members) = output_families.get(name) {
                let family_wires = (0..members.len())
                    .map(|port| WireId {
                        instantiation_path: Vec::new(),
                        wire: WireRef { node: output_wire.node, port: Port(port as u32) },
                    })
                    .collect::<Vec<_>>();
                let matrix_type = self.matrix_type(&[], *output_wire)?;
                let mut family_hasher = Sha256::new();
                for (index, member) in members.iter().enumerate() {
                    let RuntimeValue::Matrix(matrix) = member else {
                        continue 'outputs;
                    };
                    let bytes = self.backend.matrix_to_bytes(matrix);
                    family_hasher.update((index as u64).to_le_bytes());
                    family_hasher.update((bytes.len() as u64).to_le_bytes());
                    family_hasher.update(&bytes);
                    self.artifact_store
                        .store_matrix(
                            ArtifactKey {
                                production: production.clone(),
                                name: name.clone(),
                                index: Some(index),
                            },
                            &matrix_type,
                            bytes,
                        )
                        .map_err(Self::artifact_error)?;
                }
                artifacts.insert(
                    name.clone(),
                    mxx_ir_core::artifact::ExportArtifact {
                        wire: family_wires[0].clone(),
                        wire_type: matrix_type,
                        family: Some(family_wires),
                        content_hash: Some(family_hasher.finalize().into()),
                        layout: None,
                    },
                );
                continue;
            }
            let Some(RuntimeValue::Matrix(matrix)) = outputs.get(name) else {
                continue;
            };
            let wire = WireId { instantiation_path: Vec::new(), wire: *output_wire };
            let matrix_type = self.matrix_type(&[], *output_wire)?;
            let bytes = self.backend.matrix_to_bytes(matrix);
            let content_hash = Sha256::digest(&bytes).into();
            self.artifact_store
                .store_matrix(
                    ArtifactKey { production: production.clone(), name: name.clone(), index: None },
                    &matrix_type,
                    bytes,
                )
                .map_err(Self::artifact_error)?;
            artifacts.insert(
                name.clone(),
                mxx_ir_core::artifact::ExportArtifact {
                    wire,
                    wire_type: matrix_type,
                    family: None,
                    content_hash: Some(content_hash),
                    layout: None,
                },
            );
        }
        if artifacts.is_empty() {
            return Ok(None);
        }
        let manifest = mxx_ir_core::artifact::export_manifest(production.clone(), &artifacts);
        self.artifact_store.store_manifest(manifest).map_err(Self::artifact_error)?;
        Ok(Some(production))
    }

    fn execute_node(
        &mut self,
        graph: &Graph,
        env: &ParamEnv,
        path: &[InstantiationFrame],
        node: &Node,
        inputs: &BTreeMap<String, RuntimeValue<B>>,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
    ) -> Result<(), ExecutionError> {
        match &node.kind {
            NodeKind::Input { name, wire_type: _, artifact } => {
                if let Some(artifact) = artifact {
                    let count = artifact
                        .family_count
                        .as_ref()
                        .map(|count| self.eval_usize(node.id, count, env))
                        .transpose()?
                        .unwrap_or(1);
                    for index in 0..count {
                        let wire = WireRef { node: node.id, port: Port(index as u32) };
                        let matrix_type = self.matrix_type(path, wire)?;
                        values.insert(
                            wire,
                            RuntimeValue::LazyArtifact {
                                production: artifact.production_id.clone(),
                                name: artifact.artifact_name.clone(),
                                index: artifact.family_count.as_ref().map(|_| index),
                                matrix_type,
                            },
                        );
                    }
                } else {
                    let value = inputs
                        .get(name)
                        .cloned()
                        .ok_or_else(|| ExecutionError::MissingInput(name.clone()))?;
                    values.insert(WireRef { node: node.id, port: Port(0) }, value);
                }
            }
            NodeKind::Output { .. } => {
                for (port, argument) in node.args.iter().enumerate() {
                    let value = self.value(values, *argument)?;
                    values.insert(WireRef { node: node.id, port: Port(port as u32) }, value);
                }
            }
            NodeKind::ConstantInt(value) => {
                self.put(values, node.id, 0, RuntimeValue::Int(value.clone()));
            }
            NodeKind::ConstantReal(value) => {
                let value = value
                    .evaluate_f64(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                self.put(values, node.id, 0, RuntimeValue::Real(value));
            }
            NodeKind::ConstantBool(value) => {
                self.put(values, node.id, 0, RuntimeValue::Bool(*value));
            }
            NodeKind::ConstantMatrix { value, .. } => {
                let ty = self.matrix_type(path, WireRef { node: node.id, port: Port(0) })?;
                let matrix =
                    self.backend.constant_matrix(&ty, value, env).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(matrix));
            }
            NodeKind::GadgetTrapdoor { .. } => {
                let ty = self.trapdoor_type(path, WireRef { node: node.id, port: Port(0) })?;
                let sigma =
                    self.trapdoor_sigma(path, WireRef { node: node.id, port: Port(0) }, env)?;
                let public = self
                    .backend
                    .constant_matrix(
                        &ty,
                        &mxx_ir_core::node::ConstantMatrix::Gadget {
                            base: match &node.kind {
                                NodeKind::GadgetTrapdoor { base, .. } => base.clone(),
                                _ => unreachable!(),
                            },
                            small: false,
                        },
                        env,
                    )
                    .map_err(Self::backend_error)?;
                self.put(
                    values,
                    node.id,
                    0,
                    RuntimeValue::Trapdoor {
                        secret: None,
                        public,
                        matrix_type: ty,
                        sigma,
                        gadget_small: Some(false),
                    },
                );
            }
            NodeKind::IntBinary(operation) => {
                let left = self.int(values, node.args[0])?;
                let right = self.int(values, node.args[1])?;
                let output = match operation {
                    IntBinaryOp::Add => left + right,
                    IntBinaryOp::Subtract => left - right,
                    IntBinaryOp::Multiply => left * right,
                    IntBinaryOp::Divide => {
                        euclidean_div_rem(&left, &right)
                            .map_err(|_| ExecutionError::DivisionByZero(node.id))?
                            .0
                    }
                    IntBinaryOp::Remainder => {
                        euclidean_div_rem(&left, &right)
                            .map_err(|_| ExecutionError::DivisionByZero(node.id))?
                            .1
                    }
                };
                self.put(values, node.id, 0, RuntimeValue::Int(output));
            }
            NodeKind::IntCompare(operation) => {
                let left = self.int(values, node.args[0])?;
                let right = self.int(values, node.args[1])?;
                let output = match operation {
                    IntCompareOp::Equal => left == right,
                    IntCompareOp::Less => left < right,
                    IntCompareOp::LessEqual => left <= right,
                };
                self.put(values, node.id, 0, RuntimeValue::Bool(output));
            }
            NodeKind::BitExtract { bit } => {
                let value = self.int(values, node.args[0])?;
                let bit = self.eval_usize(node.id, bit, env)?;
                let output = ((value >> bit) & BigInt::one()) == BigInt::one();
                self.put(values, node.id, 0, RuntimeValue::Bool(output));
            }
            NodeKind::IntToReal => {
                let value = self
                    .int(values, node.args[0])?
                    .to_f64()
                    .ok_or(ExecutionError::InvalidRealOperation(node.id))?;
                self.put(values, node.id, 0, RuntimeValue::Real(value));
            }
            NodeKind::BoolToInt => {
                let value = self.boolean(values, node.args[0])?;
                self.put(values, node.id, 0, RuntimeValue::Int(BigInt::from(value as u8)));
            }
            NodeKind::RealBinary(operation) => {
                let left = self.real(values, node.args[0])?;
                let right = self.real(values, node.args[1])?;
                let output = match operation {
                    RealBinaryOp::Add => left + right,
                    RealBinaryOp::Subtract => left - right,
                    RealBinaryOp::Multiply => left * right,
                    RealBinaryOp::Divide if right == 0.0 => {
                        return Err(ExecutionError::InvalidRealOperation(node.id));
                    }
                    RealBinaryOp::Divide => left / right,
                };
                if !output.is_finite() {
                    return Err(ExecutionError::InvalidRealOperation(node.id));
                }
                self.put(values, node.id, 0, RuntimeValue::Real(output));
            }
            NodeKind::RealSqrt => {
                let value = self.real(values, node.args[0])?;
                if value < 0.0 {
                    return Err(ExecutionError::InvalidRealOperation(node.id));
                }
                self.put(values, node.id, 0, RuntimeValue::Real(value.sqrt()));
            }
            NodeKind::MatrixBinary(operation) => {
                let left = self.matrix(values, node.args[0])?;
                let right = self.matrix(values, node.args[1])?;
                let output = match operation {
                    MatrixBinaryOp::Add => self.backend.add(&left, &right),
                    MatrixBinaryOp::Subtract => self.backend.sub(&left, &right),
                    MatrixBinaryOp::Multiply => self.backend.multiply(&left, &right),
                }
                .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::MatrixNegate => {
                let input = self.matrix(values, node.args[0])?;
                let output = self.backend.negate(&input).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::MatrixScale { scalar } => {
                let input = self.matrix(values, node.args[0])?;
                let scalar =
                    scalar.evaluate(env).map_err(|error| self.expression_error(node.id, error))?;
                let output =
                    self.backend.scale_integer(&input, &scalar).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::Transpose => {
                let input = self.matrix(values, node.args[0])?;
                let output = self.backend.transpose(&input).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::Slice { rows, columns } => {
                let input = self.matrix(values, node.args[0])?;
                let output = self
                    .backend
                    .slice(&input, rows.as_ref(), columns.as_ref())
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::Tensor => {
                let left = self.matrix(values, node.args[0])?;
                let right = self.matrix(values, node.args[1])?;
                let output = self.backend.tensor(&left, &right).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::Concat { axis } => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.matrix(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = self.backend.concat(&inputs, *axis).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::Reshape { rows, columns } => {
                let input = self.matrix(values, node.args[0])?;
                let rows = self.eval_usize(node.id, rows, env)?;
                let columns = self.eval_usize(node.id, columns, env)?;
                let output =
                    self.backend.reshape(&input, rows, columns).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::UniformSample { range, .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(path, wire)?;
                let value = self
                    .sample_matrix(path, wire, &ty, |backend| backend.sample_uniform(&ty, range))?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(value));
            }
            NodeKind::GaussianSample { sigma, .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(path, wire)?;
                let sigma = sigma
                    .evaluate_f64(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_gaussian(&ty, sigma)
                })?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(value));
            }
            NodeKind::HashSample { variant, tag_prefix, tag_expressions, .. } => {
                let key = self.bytes(values, node.args[0])?;
                let key: [u8; 32] =
                    key.try_into().map_err(|_| ExecutionError::ValueKind(node.args[0]))?;
                let mut tag = tag_prefix.clone();
                for expression in tag_expressions {
                    let value = expression
                        .evaluate(env)
                        .map_err(|error| self.expression_error(node.id, error))?;
                    append_tag_integer(&mut tag, &value);
                }
                for wire in node.args.iter().skip(1) {
                    append_tag_integer(&mut tag, &self.int(values, *wire)?);
                }
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(path, wire)?;
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_hash(&ty, key, &tag, *variant)
                })?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(value));
            }
            NodeKind::TrapdoorSample { sigma, .. } => {
                let matrix_wire = WireRef { node: node.id, port: Port(0) };
                let trapdoor_wire = WireRef { node: node.id, port: Port(1) };
                let ty = self.matrix_type(path, matrix_wire)?;
                let sigma = sigma
                    .evaluate_f64(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let (public, secret) =
                    self.sample_trapdoor(path, matrix_wire, trapdoor_wire, &ty, sigma)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(public.clone()));
                self.put(
                    values,
                    node.id,
                    1,
                    RuntimeValue::Trapdoor {
                        secret: Some(secret),
                        public,
                        matrix_type: ty,
                        sigma,
                        gadget_small: None,
                    },
                );
            }
            NodeKind::PreimageSample { .. } => {
                let (secret, public, _, sigma, gadget_small) =
                    self.trapdoor(values, node.args[0])?;
                let target = self.matrix(values, node.args[1])?;
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(path, wire)?;
                let value = if let Some(small) = gadget_small {
                    self.backend.gadget_decompose(&target, small).map_err(Self::backend_error)?
                } else {
                    let secret =
                        secret.as_ref().expect("sampled trapdoor must carry secret material");
                    self.sample_matrix(path, wire, &ty, |backend| {
                        backend.sample_preimage(&ty, sigma, secret, &public, &target)
                    })?
                };
                self.put(values, node.id, 0, RuntimeValue::Matrix(value));
            }
            NodeKind::GadgetDecompose { small, .. } => {
                let input = self.matrix(values, node.args[0])?;
                let output =
                    self.backend.gadget_decompose(&input, *small).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::ModDown { target_modulus } => {
                let input = self.matrix(values, node.args[0])?;
                let target = target_modulus
                    .evaluate(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let output =
                    self.backend.modulus_down(&input, &target).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::ModUp { .. } => {
                let input = self.matrix(values, node.args[0])?;
                let ty = self.matrix_type(path, WireRef { node: node.id, port: Port(0) })?;
                let output = self.backend.modulus_up(&input, &ty).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(output));
            }
            NodeKind::ExtractCoefficient { position } => {
                let input = self.matrix(values, node.args[0])?;
                let position = self.eval_usize(node.id, position, env)?;
                let output = self
                    .backend
                    .extract_coefficient(&input, position)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Int(output));
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                let input = self.matrix(values, node.args[0])?;
                let plaintext = plaintext_modulus
                    .evaluate(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let length = self.eval_usize(node.id, length, env)?;
                let decoded = self
                    .backend
                    .threshold_decode(&input, &plaintext, length)
                    .map_err(Self::backend_error)?;
                for (port, value) in decoded.into_iter().enumerate() {
                    let value = if *output_bool {
                        RuntimeValue::Bool(!value.is_zero())
                    } else {
                        RuntimeValue::Int(value)
                    };
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::SubgraphCall(call) => {
                let child = graph.subgraphs.get(&call.graph).ok_or_else(|| {
                    ExecutionError::MissingSubgraph { node: node.id, name: call.graph.clone() }
                })?;
                let child_env = self.child_env(env, &call.bindings, None, node.id)?;
                let child_inputs = self.child_inputs(child, node, values)?;
                let mut child_path = path.to_vec();
                child_path.push(InstantiationFrame { call: node.id, loop_index: None });
                let outputs =
                    self.execute_instance(child, &child_env, child_path, child_inputs)?.outputs;
                for (port, value) in outputs.into_values().enumerate() {
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::ParallelLoop(loop_node) => {
                let child = graph.subgraphs.get(&loop_node.graph).ok_or_else(|| {
                    ExecutionError::MissingSubgraph { node: node.id, name: loop_node.graph.clone() }
                })?;
                let count = self.eval_usize(node.id, &loop_node.count, env)?;
                let mut child_envs = Vec::with_capacity(count);
                let mut child_paths = Vec::with_capacity(count);
                let mut child_inputs = Vec::with_capacity(count);
                for index in 0..count {
                    child_envs.push(self.child_env(
                        env,
                        &loop_node.bindings,
                        Some((&loop_node.index_variable, index)),
                        node.id,
                    )?);
                    let mut child_path = path.to_vec();
                    child_path
                        .push(InstantiationFrame { call: node.id, loop_index: Some(index as u64) });
                    child_paths.push(child_path);
                    child_inputs.push(self.child_inputs(child, node, values)?);
                }
                let instances =
                    self.execute_instances_batch(child, child_envs, child_paths, child_inputs)?;
                let mut next_port = 0u32;
                for instance in instances {
                    for value in instance.outputs.into_values() {
                        self.put(values, node.id, next_port, value);
                        next_port += 1;
                    }
                }
            }
            NodeKind::Select { count } => {
                let count = self.eval_usize(node.id, count, env)?;
                let index = self.int(values, node.args[0])?;
                let Some(index_usize) = index.to_usize().filter(|index| *index < count) else {
                    return Err(ExecutionError::SelectIndexOutOfRange {
                        node: node.id,
                        index,
                        count,
                    });
                };
                let selected_wire = node.args[index_usize + 1];
                let selected = self.materialize(values, selected_wire)?;
                self.put(values, node.id, 0, selected);
            }
        }
        Ok(())
    }

    fn execute_preimage_batch(
        &mut self,
        paths: &[Vec<InstantiationFrame>],
        node: &Node,
        values: &mut [BTreeMap<WireRef, RuntimeValue<B>>],
    ) -> Result<(), ExecutionError> {
        struct Pending<M, T> {
            instance: usize,
            wire: WireRef,
            path: Vec<InstantiationFrame>,
            request: PreimageRequest<M, T>,
        }

        let mut pending = Vec::new();
        for instance in 0..values.len() {
            let (secret, public, _, sigma, gadget_small) =
                self.trapdoor(&values[instance], node.args[0])?;
            let target = self.matrix(&mut values[instance], node.args[1])?;
            let wire = WireRef { node: node.id, port: Port(0) };
            if let Some(small) = gadget_small {
                let value =
                    self.backend.gadget_decompose(&target, small).map_err(Self::backend_error)?;
                self.put(&mut values[instance], node.id, 0, RuntimeValue::Matrix(value));
                continue;
            }
            let matrix_type = self.matrix_type(&paths[instance], wire)?;
            pending.push(Pending {
                instance,
                wire,
                path: paths[instance].clone(),
                request: PreimageRequest {
                    matrix_type,
                    sigma,
                    trapdoor: secret.expect("sampled trapdoor must carry secret material"),
                    public,
                    target,
                },
            });
        }
        if pending.is_empty() {
            return Ok(());
        }

        let replayed = match &self.sampling_mode {
            SamplingMode::Replay(replayer) => {
                let mut outputs = Vec::with_capacity(pending.len());
                for request in &pending {
                    let site = DrawSite {
                        instantiation_path: request.path.clone(),
                        node: request.wire.node,
                        port: request.wire.port,
                    };
                    match replayer.get(&site)? {
                        RecordedValue::Matrix { bytes, .. } => outputs.push(
                            self.backend
                                .matrix_from_bytes(&request.request.matrix_type, bytes)
                                .map_err(Self::backend_error)?,
                        ),
                        RecordedValue::Trapdoor { .. } => {
                            return Err(TranscriptError::KindMismatch(site).into());
                        }
                    }
                }
                Some(outputs)
            }
            SamplingMode::Fresh | SamplingMode::Record(_) => None,
        };
        let outputs = if let Some(outputs) = replayed {
            outputs
        } else {
            self.backend
                .sample_preimage_batch(
                    pending.iter().map(|request| request.request.clone()).collect(),
                )
                .map_err(Self::backend_error)?
        };
        debug_assert_eq!(outputs.len(), pending.len());

        if let SamplingMode::Record(recorder) = &mut self.sampling_mode {
            for (request, output) in pending.iter().zip(&outputs) {
                recorder.record(
                    DrawSite {
                        instantiation_path: request.path.clone(),
                        node: request.wire.node,
                        port: request.wire.port,
                    },
                    RecordedValue::Matrix {
                        matrix_type: request.request.matrix_type.clone(),
                        bytes: self.backend.matrix_to_bytes(output),
                    },
                )?;
            }
        }
        for (request, output) in pending.into_iter().zip(outputs) {
            self.put(&mut values[request.instance], node.id, 0, RuntimeValue::Matrix(output));
        }
        Ok(())
    }

    fn execute_select(
        &mut self,
        env: &ParamEnv,
        node: &Node,
        schedule: &liveness::LivenessSchedule,
        position: usize,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
    ) -> Result<(), ExecutionError> {
        let NodeKind::Select { count } = &node.kind else {
            unreachable!("select dispatcher only receives select nodes")
        };
        let count = self.eval_usize(node.id, count, env)?;
        let index = self.int(values, node.args[0])?;
        let Some(index_usize) = index.to_usize().filter(|index| *index < count) else {
            return Err(ExecutionError::SelectIndexOutOfRange { node: node.id, index, count });
        };
        let selected_wire = node.args[index_usize + 1];
        let can_move = schedule.last_use.get(&selected_wire) == Some(&position) &&
            !schedule.outputs.contains(&selected_wire);
        let selected = if can_move {
            let value =
                values.remove(&selected_wire).ok_or(ExecutionError::MissingWire(selected_wire))?;
            self.materialize_value(value)?
        } else {
            self.materialize(values, selected_wire)?
        };
        self.put(values, node.id, 0, selected);
        Ok(())
    }

    fn sample_matrix<F>(
        &mut self,
        path: &[InstantiationFrame],
        wire: WireRef,
        ty: &ConcreteMatrixType,
        fresh: F,
    ) -> Result<B::Matrix, ExecutionError>
    where
        F: FnOnce(&mut B) -> Result<B::Matrix, B::Error>,
    {
        let site = DrawSite { instantiation_path: path.to_vec(), node: wire.node, port: wire.port };
        match &mut self.sampling_mode {
            SamplingMode::Fresh => fresh(self.backend).map_err(Self::backend_error),
            SamplingMode::Record(recorder) => {
                let value = fresh(self.backend).map_err(Self::backend_error)?;
                recorder.record(
                    site,
                    RecordedValue::Matrix {
                        matrix_type: ty.clone(),
                        bytes: self.backend.matrix_to_bytes(&value),
                    },
                )?;
                Ok(value)
            }
            SamplingMode::Replay(replayer) => match replayer.get(&site)? {
                RecordedValue::Matrix { bytes, .. } => {
                    self.backend.matrix_from_bytes(ty, bytes).map_err(Self::backend_error)
                }
                RecordedValue::Trapdoor { .. } => Err(TranscriptError::KindMismatch(site).into()),
            },
        }
    }

    fn sample_trapdoor(
        &mut self,
        path: &[InstantiationFrame],
        matrix_wire: WireRef,
        trapdoor_wire: WireRef,
        ty: &ConcreteMatrixType,
        sigma: f64,
    ) -> Result<(B::Matrix, B::Trapdoor), ExecutionError> {
        let matrix_site = DrawSite {
            instantiation_path: path.to_vec(),
            node: matrix_wire.node,
            port: matrix_wire.port,
        };
        let trapdoor_site = DrawSite {
            instantiation_path: path.to_vec(),
            node: trapdoor_wire.node,
            port: trapdoor_wire.port,
        };
        match &mut self.sampling_mode {
            SamplingMode::Fresh => {
                self.backend.sample_trapdoor(ty, sigma).map_err(Self::backend_error)
            }
            SamplingMode::Record(recorder) => {
                let (public, secret) =
                    self.backend.sample_trapdoor(ty, sigma).map_err(Self::backend_error)?;
                let public_bytes = self.backend.matrix_to_bytes(&public);
                recorder.record(
                    matrix_site,
                    RecordedValue::Matrix { matrix_type: ty.clone(), bytes: public_bytes.clone() },
                )?;
                recorder.record(
                    trapdoor_site,
                    RecordedValue::Trapdoor {
                        matrix_type: ty.clone(),
                        public_bytes,
                        trapdoor_bytes: self.backend.trapdoor_to_bytes(&secret),
                    },
                )?;
                Ok((public, secret))
            }
            SamplingMode::Replay(replayer) => {
                let public = match replayer.get(&matrix_site)? {
                    RecordedValue::Matrix { bytes, .. } => {
                        self.backend.matrix_from_bytes(ty, bytes).map_err(Self::backend_error)?
                    }
                    RecordedValue::Trapdoor { .. } => {
                        return Err(TranscriptError::KindMismatch(matrix_site).into());
                    }
                };
                let secret = match replayer.get(&trapdoor_site)? {
                    RecordedValue::Trapdoor { trapdoor_bytes, .. } => self
                        .backend
                        .trapdoor_from_bytes(ty, trapdoor_bytes)
                        .map_err(Self::backend_error)?,
                    RecordedValue::Matrix { .. } => {
                        return Err(TranscriptError::KindMismatch(trapdoor_site).into());
                    }
                };
                Ok((public, secret))
            }
        }
    }

    fn materialize(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<RuntimeValue<B>, ExecutionError> {
        let value = values.get(&wire).cloned().ok_or(ExecutionError::MissingWire(wire))?;
        let was_lazy = matches!(value, RuntimeValue::LazyArtifact { .. });
        let value = self.materialize_value(value)?;
        if was_lazy {
            values.insert(wire, value.clone());
        }
        Ok(value)
    }

    fn materialize_value(
        &mut self,
        value: RuntimeValue<B>,
    ) -> Result<RuntimeValue<B>, ExecutionError> {
        if let RuntimeValue::LazyArtifact { production, name, index, matrix_type } = value {
            let key = ArtifactKey { production, name, index };
            let bytes = self.artifact_store.load_matrix(&key).map_err(Self::artifact_error)?;
            let matrix = self
                .backend
                .matrix_from_bytes(&matrix_type, &bytes)
                .map_err(Self::backend_error)?;
            Ok(RuntimeValue::Matrix(matrix))
        } else {
            Ok(value)
        }
    }

    fn matrix(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<B::Matrix, ExecutionError> {
        match self.materialize(values, wire)? {
            RuntimeValue::Matrix(value) => Ok(value),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn value(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<RuntimeValue<B>, ExecutionError> {
        values.get(&wire).cloned().ok_or(ExecutionError::MissingWire(wire))
    }

    fn int(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<BigInt, ExecutionError> {
        match self.value(values, wire)? {
            RuntimeValue::Int(value) => Ok(value),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn real(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<f64, ExecutionError> {
        match self.value(values, wire)? {
            RuntimeValue::Real(value) => Ok(value),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn boolean(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<bool, ExecutionError> {
        match self.value(values, wire)? {
            RuntimeValue::Bool(value) => Ok(value),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn bytes(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<Vec<u8>, ExecutionError> {
        match self.value(values, wire)? {
            RuntimeValue::Bytes(value) => Ok(value),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn trapdoor(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<TrapdoorParts<B>, ExecutionError> {
        match self.value(values, wire)? {
            RuntimeValue::Trapdoor { secret, public, matrix_type, sigma, gadget_small } => {
                Ok((secret, public, matrix_type, sigma, gadget_small))
            }
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn put(
        &self,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
        node: NodeId,
        port: u32,
        value: RuntimeValue<B>,
    ) {
        values.insert(WireRef { node, port: Port(port) }, value);
    }

    fn matrix_type(
        &self,
        path: &[InstantiationFrame],
        wire: WireRef,
    ) -> Result<ConcreteMatrixType, ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        self.validated
            .wires
            .get(&id)
            .and_then(|wire| wire.matrix_type().cloned())
            .ok_or(ExecutionError::MissingMetadata(id))
    }

    fn trapdoor_type(
        &self,
        path: &[InstantiationFrame],
        wire: WireRef,
    ) -> Result<ConcreteMatrixType, ExecutionError> {
        self.matrix_type(path, wire)
    }

    fn trapdoor_sigma(
        &self,
        path: &[InstantiationFrame],
        wire: WireRef,
        env: &ParamEnv,
    ) -> Result<f64, ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        match self.validated.wires.get(&id) {
            Some(ConcreteWireType::Trapdoor { sigma, .. }) => {
                sigma.evaluate_f64(env).map_err(|error| self.expression_error(wire.node, error))
            }
            _ => Err(ExecutionError::MissingMetadata(id)),
        }
    }

    fn child_inputs(
        &self,
        child: &Graph,
        node: &Node,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
    ) -> Result<BTreeMap<String, RuntimeValue<B>>, ExecutionError> {
        let names = child.nodes.iter().filter_map(|node| match &node.kind {
            NodeKind::Input { name, .. } => Some(name),
            _ => None,
        });
        names
            .zip(&node.args)
            .map(|(name, wire)| Ok((name.clone(), self.value(values, *wire)?)))
            .collect()
    }

    fn child_env(
        &self,
        parent: &ParamEnv,
        bindings: &[(String, mxx_ir_core::IntExpr)],
        loop_index: Option<(&str, usize)>,
        node: NodeId,
    ) -> Result<ParamEnv, ExecutionError> {
        let mut env = parent.clone();
        if let Some((name, index)) = loop_index {
            env.integers.insert(name.to_owned(), BigInt::from(index));
        }
        for (name, expression) in bindings {
            let value =
                expression.evaluate(&env).map_err(|error| self.expression_error(node, error))?;
            env.integers.insert(name.clone(), value);
        }
        Ok(env)
    }

    fn eval_usize(
        &self,
        node: NodeId,
        expression: &mxx_ir_core::IntExpr,
        env: &ParamEnv,
    ) -> Result<usize, ExecutionError> {
        expression
            .evaluate(env)
            .map_err(|error| self.expression_error(node, error))?
            .to_usize()
            .ok_or_else(|| ExecutionError::Expression {
                node,
                message: "expression does not fit usize".to_owned(),
            })
    }

    fn expression_error(&self, node: NodeId, error: impl std::fmt::Display) -> ExecutionError {
        ExecutionError::Expression { node, message: error.to_string() }
    }

    fn backend_error(error: B::Error) -> ExecutionError {
        ExecutionError::Backend(error.to_string())
    }

    fn artifact_error(error: S::Error) -> ExecutionError {
        ExecutionError::Artifact(error.to_string())
    }
}

fn append_tag_integer(tag: &mut Vec<u8>, value: &BigInt) {
    let (sign, bytes) = value.to_bytes_be();
    tag.push(match sign {
        Sign::Minus => 1,
        Sign::NoSign | Sign::Plus => 0,
    });
    tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    tag.extend_from_slice(&bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{ArtifactKey, MemoryArtifactStore},
        backend::poly::cpu_backend,
        transcript::{SamplingMode, TranscriptRecorder},
    };
    use mxx_ir_core::{
        Graph,
        artifact::{ProductionId, SpecHash},
        graph::{CompileParameter, CompileParameterKind},
        node::{ArtifactInput, ConcatAxis, IndexRange, Node, SampleRange},
        types::{MatrixType, WireType},
        validate,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        modulus::modulus_raise,
        poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
    };
    use std::{collections::BTreeSet, sync::Arc};

    fn wire(node: u64, port: u32) -> WireRef {
        WireRef { node: NodeId(node), port: Port(port) }
    }

    fn matrix_type(parameters: &DCRTPolyParams) -> MatrixType {
        let modulus: Arc<num_bigint::BigUint> = parameters.modulus().into();
        MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(BigInt::from_biguint(
                Sign::Plus,
                modulus.as_ref().clone(),
            )),
            ring_dimension: mxx_ir_core::IntExpr::constant(parameters.ring_dimension()),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        }
    }

    fn sample_graph(parameters: &DCRTPolyParams) -> Graph {
        Graph {
            name: "sample".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::UniformSample {
                    matrix_type: matrix_type(parameters),
                    range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("out".to_owned(), wire(1, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        }
    }

    #[test]
    fn transcript_replay_is_bit_identical() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let graph = sample_graph(&parameters);
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut recorder = TranscriptRecorder::default();
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let recorded = execute(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Record(&mut recorder),
        )
        .expect("record execution");
        let replayer = recorder.into_replayer();
        let mut replay_backend = cpu_backend([parameters]);
        let replayed = execute(
            &validated,
            &mut replay_backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Replay(&replayer),
        )
        .expect("replay execution");
        let recorded_production = recorded.production_id.clone().expect("recorded manifest");
        let replayed_production = replayed.production_id.clone().expect("replayed manifest");
        let RuntimeValue::Matrix(recorded_matrix) = &recorded.outputs["out"] else {
            panic!("matrix output");
        };
        let RuntimeValue::Matrix(replayed_matrix) = &replayed.outputs["out"] else {
            panic!("matrix output");
        };
        assert_eq!(recorded_matrix, replayed_matrix);
        assert_ne!(recorded_production, replayed_production);
        let recorded_manifest = store.manifest(&recorded_production).expect("recorded manifest");
        let replayed_manifest = store.manifest(&replayed_production).expect("replayed manifest");
        let recorded_hash =
            recorded_manifest.artifacts["out"].content_hash.expect("recorded content hash");
        let replayed_hash =
            replayed_manifest.artifacts["out"].content_hash.expect("replayed content hash");
        assert_eq!(recorded_hash, replayed_hash);
    }

    #[test]
    fn fresh_uniform_sampling_obeys_its_support_and_varies_across_runs() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let graph = sample_graph(&parameters);
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters]);
        let mut samples = BTreeSet::new();
        for run in 0..16 {
            let result =
                execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                    .expect("fresh execution");
            let RuntimeValue::Matrix(matrix) = &result.outputs["out"] else {
                panic!("matrix output")
            };
            for position in 0..8 {
                let coefficient =
                    backend.extract_coefficient(matrix, position).expect("centered coefficient");
                assert!(
                    (BigInt::from(-1)..=BigInt::from(1)).contains(&coefficient),
                    "sample at run {run}, coefficient {position} escaped its declared support"
                );
            }
            samples.insert(backend.matrix_to_bytes(matrix));
        }
        assert!(samples.len() > 1, "fresh sampling returned one repeated matrix");
    }

    #[test]
    fn traced_execution_retains_intermediate_wires_without_changing_outputs() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let graph = sample_graph(&parameters);
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters]);
        let (result, trace) = execute_with_trace(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("traced execution");
        let traced = trace
            .get(&WireId { instantiation_path: Vec::new(), wire: wire(1, 0) })
            .expect("sample wire");
        let RuntimeValue::Matrix(traced) = traced else { panic!("matrix trace") };
        let RuntimeValue::Matrix(output) = &result.outputs["out"] else { panic!("matrix output") };
        assert_eq!(traced, output);
    }

    #[test]
    fn trapdoor_transcript_replays_serialized_secret_and_public_values() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let graph = Graph {
            name: "trapdoor-transcript".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::TrapdoorSample {
                    matrix_type: matrix_type(&parameters),
                    sigma: mxx_ir_core::RealExpr::Rational(
                        mxx_ir_core::expr::Rational::new(BigInt::from(4), BigInt::from(1))
                            .expect("rational"),
                    ),
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("public".to_owned(), wire(1, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut recorder = TranscriptRecorder::default();
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let recorded = execute(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Record(&mut recorder),
        )
        .expect("record trapdoor");
        let replayer = recorder.into_replayer();
        let mut replay_backend = cpu_backend([parameters]);
        let replayed = execute(
            &validated,
            &mut replay_backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Replay(&replayer),
        )
        .expect("replay trapdoor");
        let RuntimeValue::Matrix(recorded_public) = &recorded.outputs["public"] else {
            panic!("matrix output");
        };
        let RuntimeValue::Matrix(replayed_public) = &replayed.outputs["public"] else {
            panic!("matrix output");
        };
        assert_eq!(recorded_public, replayed_public);
    }

    #[test]
    fn selected_preimage_rewrite_is_value_preserving() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digits = parameters.modulus_digits();
        let mut public_type = matrix_type(&parameters);
        public_type.columns = mxx_ir_core::IntExpr::constant(2 + digits);
        let target_type = matrix_type(&parameters);
        let preimage_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(2 + digits),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..target_type.clone()
        };
        let sigma = mxx_ir_core::RealExpr::Rational(
            mxx_ir_core::expr::Rational::new(BigInt::from(4), BigInt::one())
                .expect("positive sigma"),
        );
        let common_nodes = || {
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::from(1)),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::TrapdoorSample {
                        matrix_type: public_type.clone(),
                        sigma: sigma.clone(),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::TrapdoorSample {
                        matrix_type: public_type.clone(),
                        sigma: sigma.clone(),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::UniformSample {
                        matrix_type: target_type.clone(),
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(5),
                    kind: NodeKind::UniformSample {
                        matrix_type: target_type.clone(),
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(6),
                    kind: NodeKind::PreimageSample { matrix_type: preimage_type.clone() },
                    args: vec![wire(2, 1), wire(4, 0)],
                },
                Node {
                    id: NodeId(7),
                    kind: NodeKind::PreimageSample { matrix_type: preimage_type.clone() },
                    args: vec![wire(3, 1), wire(5, 0)],
                },
            ]
        };
        let graph = |name: &str, mut nodes: Vec<Node>, expanded: bool| {
            let output = if expanded {
                nodes.push(Node {
                    id: NodeId(8),
                    kind: NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(4, 0), wire(5, 0)],
                });
                wire(8, 0)
            } else {
                nodes.extend([
                    Node {
                        id: NodeId(8),
                        kind: NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
                        args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
                    },
                    Node {
                        id: NodeId(9),
                        kind: NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
                        args: vec![wire(1, 0), wire(6, 0), wire(7, 0)],
                    },
                    Node {
                        id: NodeId(10),
                        kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                        args: vec![wire(8, 0), wire(9, 0)],
                    },
                ]);
                wire(10, 0)
            };
            Graph {
                name: name.to_owned(),
                parameters: Vec::new(),
                input_types: BTreeMap::new(),
                nodes,
                outputs: BTreeMap::from([("out".to_owned(), output)]),
                subgraphs: BTreeMap::new(),
                real_constants: BTreeMap::new(),
            }
        };
        let original = graph("selected-preimage-original", common_nodes(), false);
        let expanded = graph("selected-preimage-expanded", common_nodes(), true);
        let original = validate(&original, &ParamEnv::default()).expect("original validation");
        let expanded = validate(&expanded, &ParamEnv::default()).expect("expanded validation");

        let mut recorder = TranscriptRecorder::default();
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let original = execute(
            &original,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Record(&mut recorder),
        )
        .expect("original execution");
        let replayer = recorder.into_replayer();
        let mut backend = cpu_backend([parameters]);
        let expanded = execute(
            &expanded,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Replay(&replayer),
        )
        .expect("expanded execution");
        let RuntimeValue::Matrix(original) = &original.outputs["out"] else {
            panic!("original matrix output")
        };
        let RuntimeValue::Matrix(expanded) = &expanded.outputs["out"] else {
            panic!("expanded matrix output")
        };
        assert_eq!(original, expanded);
    }

    #[test]
    fn output_family_is_stored_and_manifested_by_index() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let ty = matrix_type(&parameters);
        let graph = Graph {
            name: "output-family".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: ty.clone(),
                        value: mxx_ir_core::node::ConstantMatrix::Identity,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: ty,
                        value: mxx_ir_core::node::ConstantMatrix::Rotation {
                            exponent: mxx_ir_core::IntExpr::constant(1),
                        },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::Output { name: "family".to_owned() },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            outputs: BTreeMap::from([("family".to_owned(), wire(3, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters]);
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        assert_eq!(result.output_families["family"].len(), 2);
        let production = result.production_id.expect("family manifest");
        let manifest = store.manifest(&production).expect("stored manifest");
        assert_eq!(manifest.artifacts["family"].family_count.expect("family count"), 2);
        assert!(manifest.artifacts["family"].content_hash.is_some());
        for index in 0..2 {
            store
                .load_matrix(&ArtifactKey {
                    production: production.clone(),
                    name: "family".to_owned(),
                    index: Some(index),
                })
                .expect("stored family member");
        }
    }

    #[test]
    fn select_loads_only_the_selected_artifact_member() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let symbolic_type = matrix_type(&parameters);
        let modulus: Arc<num_bigint::BigUint> = parameters.modulus().into();
        let concrete_type = ConcreteMatrixType {
            modulus: BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()),
            ring_dimension: 8,
            rows: 1,
            columns: 1,
        };
        let production = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let source = Graph {
            name: "lazy-select".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::from([(
                "family".to_owned(),
                WireType::Matrix(symbolic_type.clone()),
            )]),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "family".to_owned(),
                        wire_type: WireType::Matrix(symbolic_type),
                        artifact: Some(ArtifactInput {
                            production_id: production.clone(),
                            artifact_name: "family".to_owned(),
                            family_count: Some(mxx_ir_core::IntExpr::constant(2)),
                        }),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantInt(BigInt::from(1)),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
                    args: vec![wire(2, 0), wire(1, 0), wire(1, 1)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(3, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let wires = [wire(1, 0), wire(1, 1), wire(3, 0)]
            .into_iter()
            .map(|wire| {
                (
                    WireId { instantiation_path: Vec::new(), wire },
                    ConcreteWireType::Matrix(concrete_type.clone()),
                )
            })
            .chain([(
                WireId { instantiation_path: Vec::new(), wire: wire(2, 0) },
                ConcreteWireType::ConstantInt,
            )])
            .collect();
        let validated = ValidatedGraph {
            source,
            bindings: ParamEnv::default(),
            wires,
            outputs: BTreeMap::from([("out".to_owned(), wire(3, 0))]),
            warnings: Vec::new(),
        };
        let first = DCRTPolyMatrix::zero(&parameters, 1, 1);
        let second = DCRTPolyMatrix::identity(&parameters, 1, None);
        let first_key = ArtifactKey {
            production: production.clone(),
            name: "family".to_owned(),
            index: Some(0),
        };
        let second_key = ArtifactKey { production, name: "family".to_owned(), index: Some(1) };
        let mut store = MemoryArtifactStore::default();
        store.insert(first_key.clone(), concrete_type.clone(), first.to_compact_bytes());
        store.insert(second_key.clone(), concrete_type, second.to_compact_bytes());
        let mut backend = cpu_backend([parameters]);
        let output =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        let RuntimeValue::Matrix(actual) = &output.outputs["out"] else {
            panic!("matrix output");
        };
        assert_eq!(actual, &second);
        assert_eq!(store.load_count(&first_key), 0);
        assert_eq!(store.load_count(&second_key), 1);
    }

    #[test]
    fn gadget_trapdoor_executes_as_exact_gadget_decomposition() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut scalar_type = matrix_type(&parameters);
        scalar_type.rows = mxx_ir_core::IntExpr::constant(1);
        scalar_type.columns = mxx_ir_core::IntExpr::constant(1);
        let mut gadget_type = scalar_type.clone();
        gadget_type.columns = mxx_ir_core::IntExpr::constant(parameters.modulus_digits());
        let preimage_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(parameters.modulus_digits()),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..scalar_type.clone()
        };
        let graph = Graph {
            name: "gadget-trapdoor".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: scalar_type,
                        value: mxx_ir_core::node::ConstantMatrix::Identity,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::GadgetTrapdoor {
                        matrix_type: gadget_type,
                        base: mxx_ir_core::IntExpr::constant(
                            BigInt::one() << parameters.base_bits(),
                        ),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::PreimageSample { matrix_type: preimage_type },
                    args: vec![wire(2, 0), wire(1, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(3, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let expected = DCRTPolyMatrix::identity(&parameters, 1, None).decompose();
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        let RuntimeValue::Matrix(actual) = &result.outputs["out"] else {
            panic!("matrix output");
        };
        assert_eq!(actual, &expected);
    }

    #[test]
    fn decomposed_hash_is_an_exact_preimage_of_the_plain_hash() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let plain_type = matrix_type(&parameters);
        let decomposed_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(parameters.modulus_digits()),
            ..plain_type.clone()
        };
        let base = mxx_ir_core::IntExpr::constant(BigInt::one() << parameters.base_bits());
        let hash = |id, matrix_type, variant, base| Node {
            id: NodeId(id),
            kind: NodeKind::HashSample {
                matrix_type,
                variant,
                tag_prefix: b"hash-conformance".to_vec(),
                tag_expressions: vec![mxx_ir_core::IntExpr::constant(7)],
                base,
                digit_count: None,
            },
            args: vec![wire(1, 0)],
        };
        let graph = Graph {
            name: "hash-conformance".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::from([(
                "key".to_owned(),
                WireType::Bytes { length: mxx_ir_core::IntExpr::constant(32) },
            )]),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "key".to_owned(),
                        wire_type: WireType::Bytes { length: mxx_ir_core::IntExpr::constant(32) },
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                hash(2, plain_type, mxx_ir_core::node::HashVariant::Plain, None),
                hash(3, decomposed_type, mxx_ir_core::node::HashVariant::Decomposed, Some(base)),
            ],
            outputs: BTreeMap::from([
                ("plain".to_owned(), wire(2, 0)),
                ("decomposed".to_owned(), wire(3, 0)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &validated,
            &mut backend,
            BTreeMap::from([("key".to_owned(), RuntimeValue::Bytes(vec![19; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("execution");
        let RuntimeValue::Matrix(plain) = &result.outputs["plain"] else {
            panic!("plain matrix");
        };
        let RuntimeValue::Matrix(decomposed) = &result.outputs["decomposed"] else {
            panic!("decomposed matrix");
        };
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, 1);
        assert_eq!(&(gadget * decomposed), plain);
    }

    #[test]
    fn small_decomposition_uses_explicit_dcrt_digit_count() {
        let parameters = DCRTPolyParams::new(8, 2, 17, 1);
        let (_, crt_bits, _) = parameters.to_crt();
        let digit_count = crt_bits.div_ceil(parameters.base_bits() as usize);
        let graph = Graph {
            name: "small-decomposition".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: matrix_type(&parameters),
                        value: mxx_ir_core::node::ConstantMatrix::Identity,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::GadgetDecompose {
                        base: mxx_ir_core::IntExpr::constant(
                            BigInt::one() << parameters.base_bits(),
                        ),
                        small: true,
                        digit_count: Some(mxx_ir_core::IntExpr::constant(digit_count)),
                    },
                    args: vec![wire(1, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(2, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        let RuntimeValue::Matrix(actual) = &result.outputs["out"] else {
            panic!("matrix output");
        };
        let expected = DCRTPolyMatrix::identity(&parameters, 1, None).small_decompose();
        assert_eq!(actual, &expected);
        assert_eq!(actual.row_size(), digit_count);
    }

    #[test]
    fn deterministic_matrix_nodes_match_direct_cpu_operations() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let scalar = matrix_type(&parameters);
        let row = MatrixType { columns: mxx_ir_core::IntExpr::constant(2), ..scalar.clone() };
        let column = MatrixType { rows: mxx_ir_core::IntExpr::constant(2), ..scalar.clone() };
        let graph = Graph {
            name: "deterministic-node-conformance".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: scalar.clone(),
                        value: mxx_ir_core::node::ConstantMatrix::Identity,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: scalar.clone(),
                        value: mxx_ir_core::node::ConstantMatrix::Rotation {
                            exponent: mxx_ir_core::IntExpr::constant(1),
                        },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Subtract),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(5),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node { id: NodeId(6), kind: NodeKind::MatrixNegate, args: vec![wire(2, 0)] },
                Node {
                    id: NodeId(7),
                    kind: NodeKind::MatrixScale { scalar: mxx_ir_core::IntExpr::constant(3) },
                    args: vec![wire(2, 0)],
                },
                Node { id: NodeId(8), kind: NodeKind::Transpose, args: vec![wire(2, 0)] },
                Node {
                    id: NodeId(9),
                    kind: NodeKind::Slice {
                        rows: Some(IndexRange { start: 0, end: 1 }),
                        columns: Some(IndexRange { start: 0, end: 1 }),
                    },
                    args: vec![wire(2, 0)],
                },
                Node { id: NodeId(10), kind: NodeKind::Tensor, args: vec![wire(1, 0), wire(2, 0)] },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::Concat { axis: ConcatAxis::Columns },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(12),
                    kind: NodeKind::Reshape {
                        rows: mxx_ir_core::IntExpr::constant(2),
                        columns: mxx_ir_core::IntExpr::constant(1),
                    },
                    args: vec![wire(11, 0)],
                },
                Node {
                    id: NodeId(13),
                    kind: NodeKind::ExtractCoefficient {
                        position: mxx_ir_core::IntExpr::constant(0),
                    },
                    args: vec![wire(3, 0)],
                },
                Node {
                    id: NodeId(14),
                    kind: NodeKind::ThresholdDecode {
                        plaintext_modulus: mxx_ir_core::IntExpr::constant(2),
                        length: mxx_ir_core::IntExpr::constant(2),
                        output_bool: false,
                    },
                    args: vec![wire(3, 0)],
                },
            ],
            outputs: BTreeMap::from([
                ("add".to_owned(), wire(3, 0)),
                ("sub".to_owned(), wire(4, 0)),
                ("mul".to_owned(), wire(5, 0)),
                ("neg".to_owned(), wire(6, 0)),
                ("scale".to_owned(), wire(7, 0)),
                ("transpose".to_owned(), wire(8, 0)),
                ("slice".to_owned(), wire(9, 0)),
                ("tensor".to_owned(), wire(10, 0)),
                ("concat".to_owned(), wire(11, 0)),
                ("reshape".to_owned(), wire(12, 0)),
                ("coefficient".to_owned(), wire(13, 0)),
                ("decoded_0".to_owned(), wire(14, 0)),
                ("decoded_1".to_owned(), wire(14, 1)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        // Keep the declared shapes visible in the test: the elaborator derives
        // exactly these types for concat and reshape.
        assert_eq!(row.columns, mxx_ir_core::IntExpr::constant(2));
        assert_eq!(column.rows, mxx_ir_core::IntExpr::constant(2));
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");

        let one = DCRTPolyMatrix::identity(&parameters, 1, None);
        let x = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![vec![<DCRTPolyMatrix as PolyMatrix>::P::const_rotate_poly(&parameters, 1)]],
        );
        let expected = BTreeMap::from([
            ("add", one.clone() + &x),
            ("sub", one.clone() - &x),
            ("mul", one.clone() * &x),
            ("neg", -x.clone()),
            (
                "scale",
                x.clone() *
                    <DCRTPolyMatrix as PolyMatrix>::P::from_usize_to_constant(&parameters, 3),
            ),
            ("transpose", x.transpose()),
            ("slice", x.slice(0, 1, 0, 1)),
            ("tensor", one.tensor(&x)),
            ("concat", one.concat_columns(&[&x])),
            (
                "reshape",
                DCRTPolyMatrix::from_poly_vec(
                    &parameters,
                    vec![vec![one.entry(0, 0)], vec![x.entry(0, 0)]],
                ),
            ),
        ]);
        for (name, expected) in expected {
            let RuntimeValue::Matrix(actual) = &result.outputs[name] else {
                panic!("{name} must be a matrix");
            };
            assert_eq!(actual, &expected, "{name}");
        }
        assert!(matches!(
            &result.outputs["coefficient"],
            RuntimeValue::Int(value) if value == &BigInt::one()
        ));
        assert!(matches!(
            (&result.outputs["decoded_0"], &result.outputs["decoded_1"]),
            (RuntimeValue::Int(_), RuntimeValue::Int(_))
        ));
    }

    #[test]
    fn modulus_nodes_match_direct_primitive_calls() {
        let source_parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let target_parameters = DCRTPolyParams::new(8, 2, 20, 4);
        let source_type = matrix_type(&source_parameters);
        let target_modulus: Arc<num_bigint::BigUint> = target_parameters.modulus().into();
        let source_modulus: Arc<num_bigint::BigUint> = source_parameters.modulus().into();
        let graph = Graph {
            name: "modulus-node-conformance".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: source_type,
                        value: mxx_ir_core::node::ConstantMatrix::Rotation {
                            exponent: mxx_ir_core::IntExpr::constant(1),
                        },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ModUp {
                        target_modulus: mxx_ir_core::IntExpr::constant(BigInt::from_biguint(
                            Sign::Plus,
                            target_modulus.as_ref().clone(),
                        )),
                    },
                    args: vec![wire(1, 0)],
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::ModDown {
                        target_modulus: mxx_ir_core::IntExpr::constant(BigInt::from_biguint(
                            Sign::Plus,
                            source_modulus.as_ref().clone(),
                        )),
                    },
                    args: vec![wire(2, 0)],
                },
            ],
            outputs: BTreeMap::from([
                ("up".to_owned(), wire(2, 0)),
                ("down".to_owned(), wire(3, 0)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([source_parameters.clone(), target_parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");

        let source = DCRTPolyMatrix::from_poly_vec(
            &source_parameters,
            vec![vec![<DCRTPolyMatrix as PolyMatrix>::P::const_rotate_poly(&source_parameters, 1)]],
        );
        let expected_up =
            modulus_raise(&source, &source_parameters, &target_parameters).expect("direct mod-up");
        let expected_down = expected_up.modulus_switch(&source_parameters.modulus());
        let RuntimeValue::Matrix(actual_up) = &result.outputs["up"] else {
            panic!("mod-up output must be a matrix");
        };
        let RuntimeValue::Matrix(actual_down) = &result.outputs["down"] else {
            panic!("mod-down output must be a matrix");
        };
        assert_eq!(actual_up, &expected_up);
        assert_eq!(actual_down, &expected_down);
    }

    #[test]
    fn scalar_nodes_follow_normative_runtime_arithmetic() {
        let rational = |numerator, denominator| {
            mxx_ir_core::RealExpr::Rational(
                mxx_ir_core::expr::Rational::new(
                    BigInt::from(numerator),
                    BigInt::from(denominator),
                )
                .expect("rational"),
            )
        };
        let graph = Graph {
            name: "scalar-node-conformance".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::from(-7)),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantInt(BigInt::from(3)),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::IntBinary(IntBinaryOp::Divide),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::IntBinary(IntBinaryOp::Remainder),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(5),
                    kind: NodeKind::IntCompare(IntCompareOp::Less),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node { id: NodeId(6), kind: NodeKind::BoolToInt, args: vec![wire(5, 0)] },
                Node { id: NodeId(7), kind: NodeKind::IntToReal, args: vec![wire(2, 0)] },
                Node {
                    id: NodeId(8),
                    kind: NodeKind::ConstantReal(rational(9, 4)),
                    args: Vec::new(),
                },
                Node { id: NodeId(9), kind: NodeKind::RealSqrt, args: vec![wire(8, 0)] },
                Node {
                    id: NodeId(10),
                    kind: NodeKind::RealBinary(RealBinaryOp::Add),
                    args: vec![wire(7, 0), wire(9, 0)],
                },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::BitExtract { bit: mxx_ir_core::IntExpr::constant(1) },
                    args: vec![wire(4, 0)],
                },
            ],
            outputs: BTreeMap::from([
                ("quotient".to_owned(), wire(3, 0)),
                ("remainder".to_owned(), wire(4, 0)),
                ("comparison".to_owned(), wire(5, 0)),
                ("bool_int".to_owned(), wire(6, 0)),
                ("sqrt".to_owned(), wire(9, 0)),
                ("real_sum".to_owned(), wire(10, 0)),
                ("bit".to_owned(), wire(11, 0)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        assert!(matches!(
            &result.outputs["quotient"],
            RuntimeValue::Int(value) if value == &BigInt::from(-3)
        ));
        assert!(matches!(
            &result.outputs["remainder"],
            RuntimeValue::Int(value) if value == &BigInt::from(2)
        ));
        assert!(matches!(
            (&result.outputs["comparison"], &result.outputs["bool_int"]),
            (RuntimeValue::Bool(true), RuntimeValue::Int(value)) if value == &BigInt::one()
        ));
        assert!(matches!(
            &result.outputs["sqrt"],
            RuntimeValue::Real(value) if *value >= 1.5 && *value - 1.5 < 1e-12
        ));
        assert!(matches!(
            &result.outputs["real_sum"],
            RuntimeValue::Real(value) if *value >= 4.5 && *value - 4.5 < 1e-12
        ));
        assert!(matches!(&result.outputs["bit"], RuntimeValue::Bool(true)));
    }

    #[test]
    fn integer_division_by_zero_is_a_runtime_error() {
        let graph = Graph {
            name: "division-by-zero".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::one()),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantInt(BigInt::zero()),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::IntBinary(IntBinaryOp::Divide),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(3, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        assert!(matches!(
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh,),
            Err(ExecutionError::DivisionByZero(NodeId(3)))
        ));
    }

    #[test]
    fn parallel_loop_stamps_bindings_and_exposes_each_iteration_port() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let body = Graph {
            name: "rotation-body".to_owned(),
            parameters: vec![CompileParameter {
                name: "i".to_owned(),
                kind: CompileParameterKind::Integer,
            }],
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::ConstantMatrix {
                    matrix_type: matrix_type(&parameters),
                    value: mxx_ir_core::node::ConstantMatrix::Rotation {
                        exponent: mxx_ir_core::IntExpr::Var("i".to_owned()),
                    },
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("out".to_owned(), wire(1, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let mut graph = Graph {
            name: "parallel-rotations".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(10),
                kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                    graph: body.name.clone(),
                    count: mxx_ir_core::IntExpr::constant(3),
                    index_variable: "i".to_owned(),
                    bindings: Vec::new(),
                }),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([
                ("rotation_0".to_owned(), wire(10, 0)),
                ("rotation_1".to_owned(), wire(10, 1)),
                ("rotation_2".to_owned(), wire(10, 2)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        for index in 0..3 {
            let expected = DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![vec![<DCRTPolyMatrix as PolyMatrix>::P::const_rotate_poly(
                    &parameters,
                    index,
                )]],
            );
            let RuntimeValue::Matrix(actual) = &result.outputs[&format!("rotation_{index}")] else {
                panic!("loop output must be a matrix");
            };
            assert_eq!(actual, &expected);
        }
    }

    #[test]
    fn parallel_loop_batches_preimage_nodes_across_iterations() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut public_type = matrix_type(&parameters);
        public_type.columns = mxx_ir_core::IntExpr::constant(2 + parameters.modulus_digits());
        let target_type = matrix_type(&parameters);
        let preimage_type = MatrixType {
            rows: public_type.columns.clone(),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..target_type.clone()
        };
        let sigma = mxx_ir_core::RealExpr::Rational(
            mxx_ir_core::expr::Rational::new(BigInt::from(4), BigInt::one())
                .expect("positive sigma"),
        );
        let trapdoor_wire_type =
            WireType::Trapdoor { matrix: public_type.clone(), sigma: sigma.clone() };
        let body = Graph {
            name: "preimage-body".to_owned(),
            parameters: vec![CompileParameter {
                name: "i".to_owned(),
                kind: CompileParameterKind::Integer,
            }],
            input_types: BTreeMap::from([
                ("trapdoor".to_owned(), trapdoor_wire_type.clone()),
                ("target".to_owned(), WireType::Matrix(target_type.clone())),
            ]),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "trapdoor".to_owned(),
                        wire_type: trapdoor_wire_type,
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::Input {
                        name: "target".to_owned(),
                        wire_type: WireType::Matrix(target_type.clone()),
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::PreimageSample { matrix_type: preimage_type },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(3, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let mut graph = Graph {
            name: "batched-preimages".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::TrapdoorSample { matrix_type: public_type, sigma },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::UniformSample {
                        matrix_type: target_type,
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(10),
                    kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                        graph: body.name.clone(),
                        count: mxx_ir_core::IntExpr::constant(3),
                        index_variable: "i".to_owned(),
                        bindings: Vec::new(),
                    }),
                    args: vec![wire(1, 1), wire(2, 0)],
                },
            ],
            outputs: BTreeMap::from([
                ("preimage_0".to_owned(), wire(10, 0)),
                ("preimage_1".to_owned(), wire(10, 1)),
                ("preimage_2".to_owned(), wire(10, 2)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let mut recorder = TranscriptRecorder::default();
        let recorded = execute(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Record(&mut recorder),
        )
        .expect("batched execution");
        assert_eq!(backend.preimage_batch_calls(), 1);
        let replayer = recorder.into_replayer();
        let mut replay_backend = cpu_backend([parameters]);
        let replayed = execute(
            &validated,
            &mut replay_backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Replay(&replayer),
        )
        .expect("batched replay");
        assert_eq!(replay_backend.preimage_batch_calls(), 0);
        for index in 0..3 {
            let name = format!("preimage_{index}");
            let RuntimeValue::Matrix(recorded) = &recorded.outputs[&name] else {
                panic!("recorded preimage")
            };
            let RuntimeValue::Matrix(replayed) = &replayed.outputs[&name] else {
                panic!("replayed preimage")
            };
            assert_eq!(recorded, replayed);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[serial_test::serial]
    fn hash_sampling_is_bit_exact_between_cpu_and_gpu_backends() {
        use crate::backend::poly::gpu::gpu_backend;
        use mxx_primitives::poly::dcrt::{
            gpu::{GpuDCRTPolyParams, gpu_device_sync},
            params::DCRTPolyParams,
        };

        gpu_device_sync();
        let cpu_parameters = DCRTPolyParams::new(128, 2, 16, 8);
        let (moduli, _, _) = cpu_parameters.to_crt();
        let gpu_parameters = GpuDCRTPolyParams::new(
            cpu_parameters.ring_dimension(),
            moduli,
            cpu_parameters.base_bits(),
        );
        let hash_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(2),
            columns: mxx_ir_core::IntExpr::constant(3),
            ..matrix_type(&cpu_parameters)
        };
        let graph = Graph {
            name: "cpu-gpu-hash-conformance".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::from([(
                "key".to_owned(),
                WireType::Bytes { length: mxx_ir_core::IntExpr::constant(32) },
            )]),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "key".to_owned(),
                        wire_type: WireType::Bytes { length: mxx_ir_core::IntExpr::constant(32) },
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::HashSample {
                        matrix_type: hash_type,
                        variant: mxx_ir_core::node::HashVariant::Plain,
                        tag_prefix: b"runtime-cpu-gpu".to_vec(),
                        tag_expressions: vec![mxx_ir_core::IntExpr::constant(11)],
                        base: None,
                        digit_count: None,
                    },
                    args: vec![wire(1, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(2, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let inputs = BTreeMap::from([("key".to_owned(), RuntimeValue::Bytes(vec![23; 32]))]);
        let mut cpu = cpu_backend([cpu_parameters]);
        let mut cpu_store = MemoryArtifactStore::default();
        let cpu_result = execute(&validated, &mut cpu, inputs, &mut cpu_store, SamplingMode::Fresh)
            .expect("CPU execution");
        let mut gpu = gpu_backend([gpu_parameters]);
        let mut gpu_store = MemoryArtifactStore::default();
        let gpu_result = execute(
            &validated,
            &mut gpu,
            BTreeMap::from([("key".to_owned(), RuntimeValue::Bytes(vec![23; 32]))]),
            &mut gpu_store,
            SamplingMode::Fresh,
        )
        .expect("GPU execution");
        let RuntimeValue::Matrix(cpu_matrix) = &cpu_result.outputs["out"] else {
            panic!("CPU output must be a matrix");
        };
        let RuntimeValue::Matrix(gpu_matrix) = &gpu_result.outputs["out"] else {
            panic!("GPU output must be a matrix");
        };
        assert_eq!(cpu_matrix.to_cpu_staging_bytes(), gpu_matrix.to_cpu_staging_bytes());
        gpu_device_sync();
    }
}
