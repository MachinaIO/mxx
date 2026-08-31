use crate::{
    artifact::{ArtifactKey, ArtifactPayload, ArtifactStore},
    backend::{
        Backend, IndexRange as RuntimeIndexRange, MatrixMulAccumulateRequest, PreimageRequest,
        RuntimeValue, SampleRange as RuntimeSampleRange,
    },
    session::{ArtifactHandle, SessionDescriptor, SessionStore},
    transcript::{DrawSite, RecordedValue, SamplingMode, TranscriptError},
};
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::{ArtifactType, ManifestArtifact, ProductionId},
    expr::{IndexExpr, euclidean_div_rem},
    graph::{FrozenGraphScopeId, GraphScope},
    node::{
        GridInputMode, HashVariant, IntBinaryOp, IntCompareOp, MatrixBinaryOp, NodeKind,
        RealBinaryOp,
    },
    types::{
        ConcreteMatrixType, ConcreteWireType, InstantiationFrame, NodeId, Port, WireId, WireRef,
    },
};
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;
use tracing::info;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionConfig {
    /// Maximum number of sibling loop-body instances executed in one wave.
    ///
    /// This bounds each wave's intermediate working set and backend batch
    /// size. Artifact-compatible family outputs are streamed through the
    /// artifact store; scalar-only families are accumulated in memory.
    pub max_parallel_instances: NonZeroUsize,
    /// Optional progress reporting for actual preimage sampler invocations.
    pub preimage_progress: Option<PreimageProgressConfig>,
    /// Optionally fence backend release streams after this many executed nodes.
    /// This bounds queued releases without waiting unrelated live matrices.
    pub release_fence_interval: Option<NonZeroUsize>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_parallel_instances: NonZeroUsize::new(64).expect("64 is nonzero"),
            preimage_progress: None,
            release_fence_interval: None,
        }
    }
}

/// Reporting contract for a known number of preimage sampler invocations.
///
/// The executor increments this count only after its backend has returned the
/// sampled preimages. Replayed transcript values are deliberately excluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreimageProgressConfig {
    pub total: usize,
    pub report_interval: NonZeroUsize,
}

pub struct ExecutionResult<B: Backend> {
    pub outputs: BTreeMap<String, RuntimeValue<B>>,
    pub production_id: Option<ProductionId>,
    pub artifact_handles: BTreeMap<String, Vec<ArtifactHandle>>,
    pub staged_family_leases: Vec<StagedFamilyLease>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedFamilyLease {
    pub production: ProductionId,
    pub name: String,
    pub descriptor: ManifestArtifact,
}

impl<B: Backend> ExecutionResult<B> {
    /// Materializes a named output that the executor returned as a lazy or streamed artifact.
    ///
    /// Parallel-loop families may be streamed through the artifact store so their live backend
    /// memory stays bounded by the configured wave size. Callers that need the complete output
    /// value can explicitly load it after execution. The staged payloads remain owned by this
    /// result and are deleted by [`Self::cleanup_staged`].
    pub fn materialize_output<S: ArtifactStore>(
        &mut self,
        name: &str,
        backend: &B,
        store: &mut S,
    ) -> Result<&RuntimeValue<B>, ExecutionError> {
        let value = self
            .outputs
            .get(name)
            .cloned()
            .ok_or_else(|| ExecutionError::MissingOutput(name.to_owned()))?;
        let value = materialize_runtime_value(value, backend, store)?;
        self.outputs.insert(name.to_owned(), value);
        Ok(&self.outputs[name])
    }

    /// Deletes ephemeral streamed families returned by this execution.
    ///
    /// A returned staged family remains readable until this method is called.
    /// Persisted families are replaced with final lazy artifact handles and do
    /// not require this cleanup.
    pub fn cleanup_staged<S: ArtifactStore>(
        &mut self,
        store: &mut S,
    ) -> Result<(), ExecutionError> {
        for lease in &self.staged_family_leases {
            let count = lease
                .descriptor
                .family_shape
                .as_deref()
                .and_then(shape_product)
                .ok_or_else(|| {
                    ExecutionError::Manifest("staged family lease has no cardinality".to_owned())
                })?;
            for index in 0..count {
                store
                    .remove_staged(&ArtifactKey {
                        production: lease.production.clone(),
                        name: lease.name.clone(),
                        index: Some(index),
                    })
                    .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
            }
        }
        self.staged_family_leases.clear();
        Ok(())
    }
}

pub type ExecutionTrace<B> = BTreeMap<WireId, RuntimeValue<B>>;

type TrapdoorParts<B> = (
    Option<Arc<<B as Backend>::Trapdoor>>,
    Arc<<B as Backend>::Matrix>,
    ConcreteMatrixType,
    f64,
    BigInt,
    usize,
    Option<bool>,
);

struct InstanceResult<B: Backend> {
    outputs: Vec<RuntimeValue<B>>,
}

struct ExecutableNode<'a> {
    id: NodeId,
    kind: &'a NodeKind,
    args: Vec<WireRef>,
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("backend operation failed: {0}")]
    Backend(String),
    #[error("preimage progress expected {expected} generated preimages but observed {actual}")]
    PreimageProgressMismatch { expected: usize, actual: usize },
    #[error("artifact operation failed: {0}")]
    Artifact(String),
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
    #[error("input {0} was not provided")]
    MissingInput(String),
    #[error("output {0} does not exist")]
    MissingOutput(String),
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
    #[error("backend placement {placement} is outside [0, {count})")]
    BackendPlacement { placement: usize, count: usize },
    #[error("backend returned an invalid parallel batch length at node {0:?}")]
    InvalidBatch(NodeId),
    #[error("preimage public matrix does not match the trapdoor public matrix at node {0:?}")]
    PreimagePublicMismatch(NodeId),
    #[error("manifest operation failed: {0}")]
    Manifest(String),
    #[error("scratch cleanup failed: {message}")]
    StagedCleanup { message: String, leases: Vec<StagedFamilyLease> },
}

impl ExecutionError {
    /// Retries cleanup owned by a failed execution.
    pub fn cleanup_staged<S: ArtifactStore>(
        &mut self,
        store: &mut S,
    ) -> Result<(), ExecutionError> {
        let Self::StagedCleanup { leases, .. } = self else {
            return Ok(());
        };
        for lease in leases.iter() {
            let count = lease
                .descriptor
                .family_shape
                .as_deref()
                .and_then(shape_product)
                .ok_or_else(|| {
                    ExecutionError::Manifest("staged family lease has no cardinality".to_owned())
                })?;
            for index in 0..count {
                store
                    .remove_staged(&ArtifactKey {
                        production: lease.production.clone(),
                        name: lease.name.clone(),
                        index: Some(index),
                    })
                    .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
            }
        }
        leases.clear();
        Ok(())
    }
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
    S: SessionStore,
{
    execute_with_config(
        validated,
        backend,
        inputs,
        artifact_store,
        sampling_mode,
        ExecutionConfig::default(),
    )
}

pub fn execute_with_config<B, S>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: BTreeMap<String, RuntimeValue<B>>,
    artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>,
    config: ExecutionConfig,
) -> Result<ExecutionResult<B>, ExecutionError>
where
    B: Backend,
    S: SessionStore,
{
    execute_internal(validated, backend, inputs, artifact_store, sampling_mode, false, None, config)
        .map(|(result, _)| result)
}

pub fn execute_in_session<B, S>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: BTreeMap<String, RuntimeValue<B>>,
    artifact_store: &mut S,
    execution_nonce: [u8; 32],
) -> Result<ExecutionResult<B>, ExecutionError>
where
    B: Backend,
    S: SessionStore,
{
    execute_in_session_with_config(
        validated,
        backend,
        inputs,
        artifact_store,
        execution_nonce,
        ExecutionConfig::default(),
    )
}

pub fn execute_in_session_with_config<B, S>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: BTreeMap<String, RuntimeValue<B>>,
    artifact_store: &mut S,
    execution_nonce: [u8; 32],
    config: ExecutionConfig,
) -> Result<ExecutionResult<B>, ExecutionError>
where
    B: Backend,
    S: SessionStore,
{
    let spec_hash = mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
        .map_err(|error| ExecutionError::Manifest(error.to_string()))?;
    let production = mxx_ir_core::artifact::production_id(spec_hash, execution_nonce);
    let input_digest = runtime_inputs_digest(backend, &inputs)?;
    let descriptor = SessionDescriptor::new(
        production.clone(),
        validated.source.name().to_owned(),
        input_digest,
    );
    artifact_store
        .open_session(&descriptor)
        .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
    match execute_internal(
        validated,
        backend,
        inputs,
        artifact_store,
        SamplingMode::Fresh,
        false,
        Some(production.clone()),
        config,
    ) {
        Ok((result, _)) => Ok(result),
        Err(error) => {
            let _ = artifact_store.release_session(&production);
            Err(error)
        }
    }
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
    S: SessionStore,
{
    execute_internal(
        validated,
        backend,
        inputs,
        artifact_store,
        sampling_mode,
        true,
        None,
        ExecutionConfig::default(),
    )
}

fn execute_internal<B, S>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: BTreeMap<String, RuntimeValue<B>>,
    artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>,
    capture_trace: bool,
    session: Option<ProductionId>,
    config: ExecutionConfig,
) -> Result<(ExecutionResult<B>, ExecutionTrace<B>), ExecutionError>
where
    B: Backend,
    S: SessionStore,
{
    let spec_hash = mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
        .map_err(|error| ExecutionError::Manifest(error.to_string()))?;
    let production = session
        .clone()
        .unwrap_or_else(|| mxx_ir_core::artifact::production_id(spec_hash, rand::random()));
    let mut executor = Executor {
        validated,
        backend,
        artifact_store,
        sampling_mode,
        trace: capture_trace.then(BTreeMap::new),
        session,
        config,
        production,
        staged_families: BTreeMap::new(),
        preimage_progress: config.preimage_progress.map(PreimageProgress::new),
        executed_node_count: 0,
        last_release_fence_node_count: 0,
        has_pending_releases: false,
    };
    let inputs = inputs
        .into_iter()
        .map(|(name, value)| Ok((name, executor.value_for_placement(value, 0)?)))
        .collect::<Result<_, ExecutionError>>()?;
    let mut instance = match executor.execute_instance(
        &FrozenGraphScopeId::Root,
        &validated.bindings,
        Vec::new(),
        inputs,
        0,
    ) {
        Ok(instance) => instance,
        Err(error) => {
            return match executor.cleanup_all_staged_families() {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(cleanup_error),
            };
        }
    };
    if let Err(error) = executor.finish_preimage_progress() {
        return match executor.cleanup_all_staged_families() {
            Ok(()) => Err(error),
            Err(cleanup_error) => Err(cleanup_error),
        };
    }
    let mut named_outputs = validated
        .source
        .outputs()
        .keys()
        .cloned()
        .zip(instance.outputs.drain(..))
        .collect::<BTreeMap<_, _>>();
    let (production_id, artifact_handles) = match executor.persist_outputs(&mut named_outputs) {
        Ok(persisted) => persisted,
        Err(error) => {
            return match executor.cleanup_all_staged_families() {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(cleanup_error),
            };
        }
    };
    let staged_family_leases = executor.cleanup_unreturned_staged_families(&named_outputs)?;
    if let Some(production) = &executor.session {
        if let Err(error) = executor.artifact_store.release_session(production) {
            return Err(ExecutionError::StagedCleanup {
                message: format!("session release failed: {error}"),
                leases: staged_family_leases,
            });
        }
    }
    executor.fence_pending_releases()?;
    let result = ExecutionResult {
        outputs: named_outputs,
        production_id,
        artifact_handles,
        staged_family_leases,
    };
    Ok((result, executor.trace.take().unwrap_or_default()))
}

struct Executor<'a, B: Backend, S: SessionStore> {
    validated: &'a ValidatedGraph,
    backend: &'a mut B,
    artifact_store: &'a mut S,
    sampling_mode: SamplingMode<'a>,
    trace: Option<ExecutionTrace<B>>,
    session: Option<ProductionId>,
    config: ExecutionConfig,
    production: ProductionId,
    staged_families: BTreeMap<(ProductionId, String), ManifestArtifact>,
    preimage_progress: Option<PreimageProgress>,
    executed_node_count: usize,
    last_release_fence_node_count: usize,
    has_pending_releases: bool,
}

struct PreparedPreimage<M, T> {
    placement: usize,
    site: DrawSite,
    request: PreimageRequest<M, T>,
}

struct PreimageProgress {
    config: PreimageProgressConfig,
    completed: usize,
    last_reported: usize,
    started: Instant,
}

impl PreimageProgress {
    fn new(config: PreimageProgressConfig) -> Self {
        info!(
            total = config.total,
            report_interval = config.report_interval.get(),
            "preimage generation progress started"
        );
        Self { config, completed: 0, last_reported: 0, started: Instant::now() }
    }

    fn record(&mut self, count: usize) {
        self.completed = self.completed.saturating_add(count);
        let final_report = self.completed >= self.config.total;
        if !final_report &&
            self.completed.saturating_sub(self.last_reported) < self.config.report_interval.get()
        {
            return;
        }
        let elapsed = self.started.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let rate_per_second = (self.completed as f64) / elapsed_seconds.max(f64::MIN_POSITIVE);
        let remaining = self.config.total.saturating_sub(self.completed);
        let eta = Duration::from_secs_f64((remaining as f64) / rate_per_second);
        info!(
            completed = self.completed,
            total = self.config.total,
            percent = (self.completed as f64) * 100.0 / (self.config.total.max(1) as f64),
            rate_per_second,
            elapsed = ?elapsed,
            eta = ?eta,
            "preimage generation progress"
        );
        self.last_reported = self.completed;
    }

    fn finish(&self) -> Result<(), ExecutionError> {
        if self.completed != self.config.total {
            return Err(ExecutionError::PreimageProgressMismatch {
                expected: self.config.total,
                actual: self.completed,
            });
        }
        info!(
            completed = self.completed,
            total = self.config.total,
            elapsed = ?self.started.elapsed(),
            "preimage generation completed"
        );
        Ok(())
    }
}

impl<B, S> Executor<'_, B, S>
where
    B: Backend,
    S: SessionStore,
{
    fn execute_instance(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        path: Vec<InstantiationFrame>,
        inputs: BTreeMap<String, RuntimeValue<B>>,
        placement: usize,
    ) -> Result<InstanceResult<B>, ExecutionError> {
        self.execute_instances_batch(
            scope_id,
            vec![env.clone()],
            vec![path],
            vec![inputs],
            vec![placement],
        )
        .map(|mut instances| instances.pop().expect("single execution returns one instance"))
    }

    fn execute_instances_batch(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        envs: Vec<ParamEnv>,
        paths: Vec<Vec<InstantiationFrame>>,
        inputs: Vec<BTreeMap<String, RuntimeValue<B>>>,
        placements: Vec<usize>,
    ) -> Result<Vec<InstanceResult<B>>, ExecutionError> {
        debug_assert_eq!(envs.len(), paths.len());
        debug_assert_eq!(envs.len(), inputs.len());
        debug_assert_eq!(envs.len(), placements.len());
        if envs.is_empty() {
            return Ok(Vec::new());
        }
        let scope = self.validated.source.scope(scope_id).ok_or_else(|| {
            ExecutionError::MissingSubgraph { node: NodeId(0), name: format!("{scope_id:?}") }
        })?;
        let validated_scope = self.validated.scope(scope_id).ok_or_else(|| {
            ExecutionError::MissingSubgraph { node: NodeId(0), name: format!("{scope_id:?}") }
        })?;
        let schedule = &validated_scope.liveness;
        let mut values = (0..envs.len())
            .map(|_| BTreeMap::<WireRef, RuntimeValue<B>>::new())
            .collect::<Vec<_>>();
        for (position, handle) in validated_scope.execution_order.iter().enumerate() {
            let node = ExecutableNode {
                id: NodeId(position as u64),
                kind: handle.kind(),
                args: scope.arguments(handle).expect("validated node belongs to its scope"),
            };
            if matches!(node.kind, NodeKind::PreimageSample { .. }) && envs.len() > 1 {
                self.execute_preimage_batch(
                    scope_id,
                    &envs,
                    &paths,
                    &placements,
                    &node,
                    &mut values,
                )?;
            } else if envs.len() > 1 &&
                self.execute_parallel_matrix_node_by_placement(
                    &placements,
                    &envs,
                    &node,
                    &mut values,
                )?
            {
            } else if matches!(node.kind, NodeKind::Select { .. }) {
                for index in 0..envs.len() {
                    self.set_placement(placements[index])?;
                    self.execute_select(
                        &envs[index],
                        &node,
                        schedule,
                        position,
                        &mut values[index],
                    )?;
                }
            } else {
                for index in 0..envs.len() {
                    self.set_placement(placements[index])?;
                    self.execute_node(
                        scope_id,
                        &envs[index],
                        &paths[index],
                        &node,
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
                        !schedule.retained.contains(argument)
                    {
                        if let Some(value) = values[index].remove(argument) {
                            self.has_pending_releases |= value.releases_backend_resources_on_drop();
                        }
                    }
                }
            }
            self.executed_node_count = self.executed_node_count.saturating_add(envs.len());
            if self.has_pending_releases &&
                self.config.release_fence_interval.is_some_and(|interval| {
                    self.executed_node_count.saturating_sub(self.last_release_fence_node_count) >=
                        interval.get()
                })
            {
                self.fence_pending_releases()?;
                info!(
                    scope = ?scope_id,
                    scope_completed_nodes = position + 1,
                    scope_total_nodes = validated_scope.execution_order.len(),
                    total_executed_nodes = self.executed_node_count,
                    instances = values.len(),
                    "execution progress checkpoint"
                );
            }
        }
        let mut instances = Vec::with_capacity(values.len());
        for (index, mut instance_values) in values.into_iter().enumerate() {
            self.set_placement(placements[index])?;
            let outputs = scope
                .outputs()
                .iter()
                .map(|wire| self.materialize(&mut instance_values, *wire))
                .collect::<Result<Vec<_>, _>>()?;
            instances.push(InstanceResult { outputs });
        }
        Ok(instances)
    }

    fn execute_parallel_matrix_node_by_placement(
        &mut self,
        placements: &[usize],
        envs: &[ParamEnv],
        node: &ExecutableNode<'_>,
        values: &mut [BTreeMap<WireRef, RuntimeValue<B>>],
    ) -> Result<bool, ExecutionError> {
        if !matches!(
            node.kind,
            NodeKind::MatrixBinary(_) |
                NodeKind::MatrixMulAccumulate { .. } |
                NodeKind::MatrixNegate |
                NodeKind::MatrixScale { .. }
        ) {
            return Ok(false);
        }
        for placement in 0..self.backend.placement_count() {
            let indices = placements
                .iter()
                .enumerate()
                .filter_map(|(index, assigned)| (*assigned == placement).then_some(index))
                .collect::<Vec<_>>();
            if !indices.is_empty() {
                self.execute_parallel_matrix_node(placement, envs, node, values, &indices)?;
            }
        }
        Ok(true)
    }

    fn execute_parallel_matrix_node(
        &mut self,
        placement: usize,
        envs: &[ParamEnv],
        node: &ExecutableNode<'_>,
        values: &mut [BTreeMap<WireRef, RuntimeValue<B>>],
        indices: &[usize],
    ) -> Result<(), ExecutionError> {
        self.set_placement(placement)?;
        let outputs = match node.kind {
            NodeKind::MatrixBinary(operation) => {
                let mut inputs = Vec::with_capacity(indices.len());
                for index in indices {
                    let instance = &mut values[*index];
                    let left = self.matrix(instance, node.args[0])?;
                    let right = self.matrix(instance, node.args[1])?;
                    inputs.push((left, right));
                }
                match operation {
                    MatrixBinaryOp::Add => self.backend.add_batch(inputs),
                    MatrixBinaryOp::Subtract => self.backend.sub_batch(inputs),
                    MatrixBinaryOp::Multiply => self.backend.multiply_batch(inputs),
                }
                .map_err(Self::backend_error)?
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let mut requests = Vec::with_capacity(indices.len());
                for index in indices {
                    let env = &envs[*index];
                    let instance = &mut values[*index];
                    let mut products = Vec::with_capacity(coefficients.len());
                    for (product, coefficient) in coefficients.iter().enumerate() {
                        products.push((
                            coefficient
                                .evaluate(env)
                                .map_err(|error| self.expression_error(node.id, error))?,
                            self.matrix(instance, node.args[2 * product])?,
                            self.matrix(instance, node.args[2 * product + 1])?,
                        ));
                    }
                    let bias = if *has_bias {
                        Some(self.matrix(instance, node.args[2 * coefficients.len()])?)
                    } else {
                        None
                    };
                    requests.push(MatrixMulAccumulateRequest { products, bias });
                }
                self.backend.matrix_mul_accumulate_batch(requests).map_err(Self::backend_error)?
            }
            NodeKind::MatrixNegate => {
                let mut inputs = Vec::with_capacity(indices.len());
                for index in indices {
                    let instance = &mut values[*index];
                    inputs.push(self.matrix(instance, node.args[0])?);
                }
                self.backend.negate_batch(inputs).map_err(Self::backend_error)?
            }
            NodeKind::MatrixScale { scalar } => {
                let mut inputs = Vec::with_capacity(indices.len());
                for index in indices {
                    let env = &envs[*index];
                    let instance = &mut values[*index];
                    let value = self.matrix(instance, node.args[0])?;
                    let scalar = scalar
                        .evaluate(env)
                        .map_err(|error| self.expression_error(node.id, error))?;
                    inputs.push((value, scalar));
                }
                self.backend.scale_integer_batch(inputs).map_err(Self::backend_error)?
            }
            _ => unreachable!("matrix batch kind checked by caller"),
        };
        if outputs.len() != indices.len() {
            return Err(ExecutionError::InvalidBatch(node.id));
        }
        for (index, output) in indices.iter().zip(outputs) {
            self.put(&mut values[*index], node.id, 0, RuntimeValue::matrix(output));
        }
        Ok(())
    }

    fn persist_outputs(
        &mut self,
        outputs: &mut BTreeMap<String, RuntimeValue<B>>,
    ) -> Result<(Option<ProductionId>, BTreeMap<String, Vec<ArtifactHandle>>), ExecutionError> {
        let production = self.production.clone();
        let mut artifacts = BTreeMap::new();
        let mut handles = BTreeMap::<String, Vec<ArtifactHandle>>::new();
        let mut staged_replacements = Vec::new();
        for (name, output_root) in self.validated.source.outputs() {
            let Some(confidentiality) = output_root.confidentiality else {
                continue;
            };
            let Some(output) = outputs.get(name) else {
                continue;
            };
            let wire = WireId { instantiation_path: Vec::new(), wire: output_root.value };
            let concrete_type = self
                .validated
                .root_scope()
                .wire_types
                .get(&output_root.value)
                .ok_or_else(|| ExecutionError::MissingMetadata(wire.clone()))?;
            let (element_type, family_shape) = match concrete_type {
                ConcreteWireType::Family { element, shape } => {
                    (element.as_ref(), Some(shape.clone()))
                }
                scalar => (scalar, None),
            };
            let artifact_type = ArtifactType::from_wire_type(element_type).ok_or_else(|| {
                ExecutionError::Manifest(format!("output {name} is not artifact-compatible"))
            })?;
            if let RuntimeValue::StagedArtifactFamily {
                production: staged_production,
                name: staged_name,
                descriptor,
            } = output
            {
                let Some(shape) = family_shape.as_deref() else {
                    return Err(ExecutionError::Manifest(format!(
                        "output {name} is staged as a family but validated as a scalar"
                    )));
                };
                if descriptor.artifact_type != artifact_type ||
                    descriptor.family_shape.as_deref() != Some(shape)
                {
                    return Err(ExecutionError::Manifest(format!(
                        "output {name} staged descriptor does not match validated metadata"
                    )));
                }
                let mut family_hasher = Sha256::new();
                for index in 0..shape_product(shape)
                    .ok_or_else(|| ExecutionError::Manifest("family shape overflow".to_owned()))?
                {
                    let staged_key = ArtifactKey {
                        production: staged_production.clone(),
                        name: staged_name.clone(),
                        index: Some(index),
                    };
                    let payload = self
                        .artifact_store
                        .load_staged(&staged_key, descriptor)
                        .map_err(Self::artifact_error)?;
                    let bytes = crate::artifact::payload_bytes(&payload);
                    family_hasher.update((index as u64).to_le_bytes());
                    family_hasher.update((bytes.len() as u64).to_le_bytes());
                    family_hasher.update(&bytes);
                    let handle = ArtifactHandle {
                        key: ArtifactKey {
                            production: production.clone(),
                            name: name.clone(),
                            index: Some(index),
                        },
                        artifact_type: artifact_type.clone(),
                        confidentiality,
                        layout: None,
                    };
                    self.artifact_store
                        .store(
                            handle.key.clone(),
                            &artifact_type,
                            confidentiality,
                            handle.layout.as_deref(),
                            payload,
                        )
                        .map_err(Self::artifact_error)?;
                    if self.session.is_some() {
                        self.artifact_store
                            .commit_artifact(&handle)
                            .map_err(Self::artifact_error)?;
                    }
                    handles.entry(name.clone()).or_default().push(handle);
                }
                artifacts.insert(
                    name.clone(),
                    mxx_ir_core::artifact::ExportArtifact {
                        wire,
                        artifact_type,
                        family_shape: family_shape.clone(),
                        confidentiality,
                        content_hash: Some(family_hasher.finalize().into()),
                        layout: None,
                    },
                );
                staged_replacements.push((
                    name.clone(),
                    staged_production.clone(),
                    staged_name.clone(),
                    shape_product(shape).ok_or_else(|| {
                        ExecutionError::Manifest("family shape overflow".to_owned())
                    })?,
                ));
                continue;
            }
            if let RuntimeValue::Family(members) = output {
                if family_shape.as_deref().and_then(shape_product) != Some(members.len()) {
                    return Err(ExecutionError::Manifest(format!(
                        "output {name} family count does not match validated metadata"
                    )));
                }
                let mut family_hasher = Sha256::new();
                for (index, member) in members.iter().enumerate() {
                    let (payload, bytes) = self.encode_artifact(member, &artifact_type)?;
                    family_hasher.update((index as u64).to_le_bytes());
                    family_hasher.update((bytes.len() as u64).to_le_bytes());
                    family_hasher.update(&bytes);
                    let handle = ArtifactHandle {
                        key: ArtifactKey {
                            production: production.clone(),
                            name: name.clone(),
                            index: Some(index),
                        },
                        artifact_type: artifact_type.clone(),
                        confidentiality,
                        layout: None,
                    };
                    self.artifact_store
                        .store(
                            handle.key.clone(),
                            &artifact_type,
                            confidentiality,
                            handle.layout.as_deref(),
                            payload,
                        )
                        .map_err(Self::artifact_error)?;
                    if self.session.is_some() {
                        self.artifact_store
                            .commit_artifact(&handle)
                            .map_err(Self::artifact_error)?;
                    }
                    handles.entry(name.clone()).or_default().push(handle);
                }
                artifacts.insert(
                    name.clone(),
                    mxx_ir_core::artifact::ExportArtifact {
                        wire,
                        artifact_type,
                        family_shape: family_shape.clone(),
                        confidentiality,
                        content_hash: Some(family_hasher.finalize().into()),
                        layout: None,
                    },
                );
                continue;
            }
            let (payload, bytes) = self.encode_artifact(output, &artifact_type)?;
            let content_hash = Sha256::digest(&bytes).into();
            let handle = ArtifactHandle {
                key: ArtifactKey {
                    production: production.clone(),
                    name: name.clone(),
                    index: None,
                },
                artifact_type: artifact_type.clone(),
                confidentiality,
                layout: None,
            };
            self.artifact_store
                .store(
                    handle.key.clone(),
                    &artifact_type,
                    confidentiality,
                    handle.layout.as_deref(),
                    payload,
                )
                .map_err(Self::artifact_error)?;
            if self.session.is_some() {
                self.artifact_store.commit_artifact(&handle).map_err(Self::artifact_error)?;
            }
            handles.entry(name.clone()).or_default().push(handle);
            artifacts.insert(
                name.clone(),
                mxx_ir_core::artifact::ExportArtifact {
                    wire,
                    artifact_type,
                    family_shape: None,
                    confidentiality,
                    content_hash: Some(content_hash),
                    layout: None,
                },
            );
        }
        if artifacts.is_empty() {
            if self.session.is_none() {
                return Ok((None, handles));
            }
        }
        let manifest = mxx_ir_core::artifact::export_manifest(production.clone(), &artifacts);
        let replacement_descriptors = staged_replacements
            .iter()
            .map(|(name, _, _, _)| {
                let descriptor = manifest
                    .artifacts
                    .get(name)
                    .cloned()
                    .expect("staged output was inserted into the manifest");
                (name.clone(), descriptor)
            })
            .collect::<BTreeMap<_, _>>();
        if self.session.is_some() {
            self.artifact_store.finalize_session(manifest).map_err(Self::artifact_error)?;
        } else {
            self.artifact_store.store_manifest(manifest).map_err(Self::artifact_error)?;
        }
        for (name, staged_production, staged_name, count) in staged_replacements {
            for index in 0..count {
                self.artifact_store
                    .remove_staged(&ArtifactKey {
                        production: staged_production.clone(),
                        name: staged_name.clone(),
                        index: Some(index),
                    })
                    .map_err(Self::artifact_error)?;
            }
            outputs.insert(
                name.clone(),
                RuntimeValue::LazyArtifactFamily {
                    production: production.clone(),
                    name: name.clone(),
                    descriptor: replacement_descriptors[&name].clone(),
                },
            );
        }
        Ok((Some(production), handles))
    }

    fn cleanup_unreturned_staged_families(
        &mut self,
        outputs: &BTreeMap<String, RuntimeValue<B>>,
    ) -> Result<Vec<StagedFamilyLease>, ExecutionError> {
        let mut retained = BTreeMap::new();
        for value in outputs.values() {
            collect_staged_families(value, &mut retained);
        }
        let staged = self.staged_families.clone();
        let mut leases = Vec::new();
        for ((production, name), descriptor) in staged {
            if retained.contains_key(&(production.clone(), name.clone())) {
                leases.push(StagedFamilyLease { production, name, descriptor });
                continue;
            }
            let Some(shape) = descriptor.family_shape.as_deref() else {
                return Err(self.staged_cleanup_error("staged family descriptor has no cardinality"));
            };
            for index in 0..shape_product(shape)
                .ok_or_else(|| self.staged_cleanup_error("staged family shape overflow"))?
            {
                let key = ArtifactKey {
                    production: production.clone(),
                    name: name.clone(),
                    index: Some(index),
                };
                if let Err(error) = self.artifact_store.remove_staged(&key) {
                    return Err(self.staged_cleanup_error(error));
                }
            }
        }
        self.staged_families.clear();
        Ok(leases)
    }

    fn cleanup_all_staged_families(&mut self) -> Result<(), ExecutionError> {
        let staged = self.staged_families.clone();
        for ((production, name), descriptor) in staged {
            let Some(shape) = descriptor.family_shape.as_deref() else {
                return Err(self.staged_cleanup_error("staged family descriptor has no cardinality"));
            };
            for index in 0..shape_product(shape)
                .ok_or_else(|| self.staged_cleanup_error("staged family shape overflow"))?
            {
                let key = ArtifactKey {
                    production: production.clone(),
                    name: name.clone(),
                    index: Some(index),
                };
                if let Err(error) = self.artifact_store.remove_staged(&key) {
                    return Err(self.staged_cleanup_error(error));
                }
            }
        }
        self.staged_families.clear();
        Ok(())
    }

    fn staged_cleanup_error(&self, error: impl std::fmt::Display) -> ExecutionError {
        let leases = self
            .staged_families
            .iter()
            .map(|((production, name), descriptor)| StagedFamilyLease {
                production: production.clone(),
                name: name.clone(),
                descriptor: descriptor.clone(),
            })
            .collect();
        ExecutionError::StagedCleanup { message: error.to_string(), leases }
    }

    fn encode_artifact(
        &self,
        value: &RuntimeValue<B>,
        artifact_type: &ArtifactType,
    ) -> Result<(ArtifactPayload, Vec<u8>), ExecutionError> {
        match (value, artifact_type) {
            (RuntimeValue::Matrix(matrix), ArtifactType::Matrix(_)) => {
                let bytes = self.backend.matrix_to_bytes(matrix);
                Ok((ArtifactPayload::Matrix(bytes.clone()), bytes))
            }
            (RuntimeValue::Bytes(bytes), ArtifactType::Bytes { length })
                if bytes.len() == *length =>
            {
                Ok((ArtifactPayload::Bytes(bytes.clone()), bytes.clone()))
            }
            (RuntimeValue::TypedBlob(bytes), ArtifactType::TypedBlob { .. }) => {
                Ok((ArtifactPayload::TypedBlob(bytes.clone()), bytes.clone()))
            }
            (
                RuntimeValue::Trapdoor { secret: Some(secret), public, .. },
                ArtifactType::Trapdoor { .. },
            ) => {
                let public_bytes = self.backend.matrix_to_bytes(public);
                let secret_bytes = self.backend.trapdoor_to_bytes(secret);
                let mut canonical = Vec::with_capacity(
                    16usize.saturating_add(public_bytes.len()).saturating_add(secret_bytes.len()),
                );
                canonical.extend_from_slice(&(public_bytes.len() as u64).to_le_bytes());
                canonical.extend_from_slice(&public_bytes);
                canonical.extend_from_slice(&(secret_bytes.len() as u64).to_le_bytes());
                canonical.extend_from_slice(&secret_bytes);
                Ok((ArtifactPayload::Trapdoor { public_bytes, secret_bytes }, canonical))
            }
            _ => Err(ExecutionError::Manifest(
                "runtime value does not match declared artifact type".to_owned(),
            )),
        }
    }

    fn execute_node(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        path: &[InstantiationFrame],
        node: &ExecutableNode<'_>,
        inputs: &BTreeMap<String, RuntimeValue<B>>,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
    ) -> Result<(), ExecutionError> {
        match &node.kind {
            NodeKind::Input { name, wire_type: _, artifact } => {
                if let Some(artifact) = artifact {
                    let wire = WireRef { node: node.id, port: Port(0) };
                    let wire_id = WireId { instantiation_path: path.to_vec(), wire };
                    let concrete = self
                        .validated
                        .scope(scope_id)
                        .and_then(|scope| scope.wire_types.get(&wire))
                        .ok_or_else(|| ExecutionError::MissingMetadata(wire_id.clone()))?;
                    let descriptor = self
                        .validated
                        .scope(scope_id)
                        .and_then(|scope| scope.artifact_inputs.get(&wire))
                        .cloned()
                        .ok_or_else(|| ExecutionError::MissingMetadata(wire_id.clone()))?;
                    if let ConcreteWireType::Family { element, shape } = concrete {
                        let declared_shape = shape.as_slice();
                        let artifact_type =
                            ArtifactType::from_wire_type(element).ok_or_else(|| {
                                ExecutionError::Manifest(
                                    "indexed artifact has unsupported element type".to_owned(),
                                )
                            })?;
                        if descriptor.artifact_type != artifact_type ||
                            descriptor.family_shape.as_deref() != Some(declared_shape)
                        {
                            return Err(ExecutionError::Manifest(
                                "validated artifact descriptor does not match its wire metadata"
                                    .to_owned(),
                            ));
                        }
                        values.insert(
                            wire,
                            RuntimeValue::LazyArtifactFamily {
                                production: artifact.production_id.clone(),
                                name: artifact.artifact_name.clone(),
                                descriptor,
                            },
                        );
                        return Ok(());
                    }
                    let artifact_type =
                        ArtifactType::from_wire_type(concrete).ok_or_else(|| {
                            ExecutionError::Manifest(
                                "artifact has unsupported wire type".to_owned(),
                            )
                        })?;
                    if descriptor.artifact_type != artifact_type ||
                        descriptor.family_shape.is_some()
                    {
                        return Err(ExecutionError::Manifest(
                            "validated artifact descriptor does not match its wire metadata"
                                .to_owned(),
                        ));
                    }
                    values.insert(
                        wire,
                        RuntimeValue::LazyArtifact {
                            production: artifact.production_id.clone(),
                            name: artifact.artifact_name.clone(),
                            index: None,
                            descriptor,
                        },
                    );
                } else {
                    let value = inputs
                        .get(name)
                        .cloned()
                        .ok_or_else(|| ExecutionError::MissingInput(name.clone()))?;
                    values.insert(WireRef { node: node.id, port: Port(0) }, value);
                }
            }
            NodeKind::ConstantInt(value) => {
                self.put(values, node.id, 0, RuntimeValue::Int(value.clone()));
            }
            NodeKind::EvaluateInt(value) => {
                let value =
                    value.evaluate(env).map_err(|error| self.expression_error(node.id, error))?;
                self.put(values, node.id, 0, RuntimeValue::Int(value));
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
                let ty =
                    self.matrix_type(scope_id, path, WireRef { node: node.id, port: Port(0) })?;
                let matrix =
                    self.backend.constant_matrix(&ty, value, env).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(matrix));
            }
            NodeKind::GadgetTrapdoor { .. } => {
                let trapdoor_wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.trapdoor_type(scope_id, path, trapdoor_wire)?;
                let sigma = self.trapdoor_sigma(scope_id, path, trapdoor_wire)?;
                let (gadget_base, digit_count) =
                    self.trapdoor_layout(scope_id, path, trapdoor_wire)?;
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
                        public: Arc::new(public),
                        matrix_type: ty,
                        sigma,
                        gadget_base,
                        digit_count,
                        gadget_small: Some(false),
                    },
                );
            }
            NodeKind::TrapdoorPublic => {
                let value = self.materialize(values, node.args[0])?;
                let RuntimeValue::Trapdoor { public, .. } = value else {
                    return Err(ExecutionError::ValueKind(node.args[0]));
                };
                self.put(values, node.id, 0, RuntimeValue::Matrix(public));
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
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::ApplyPreimage => {
                let left = self.matrix(values, node.args[0])?;
                let right = self.matrix(values, node.args[1])?;
                let output = self.backend.multiply(&left, &right).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let mut products = Vec::with_capacity(coefficients.len());
                for (product, coefficient) in coefficients.iter().enumerate() {
                    products.push((
                        coefficient
                            .evaluate(env)
                            .map_err(|error| self.expression_error(node.id, error))?,
                        self.matrix(values, node.args[2 * product])?,
                        self.matrix(values, node.args[2 * product + 1])?,
                    ));
                }
                let bias = if *has_bias {
                    Some(self.matrix(values, node.args[2 * coefficients.len()])?)
                } else {
                    None
                };
                let output = self
                    .backend
                    .matrix_mul_accumulate(MatrixMulAccumulateRequest { products, bias })
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::MatrixNegate => {
                let input = self.matrix(values, node.args[0])?;
                let output = self.backend.negate(&input).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::MatrixScale { scalar } => {
                let input = self.matrix(values, node.args[0])?;
                let scalar =
                    scalar.evaluate(env).map_err(|error| self.expression_error(node.id, error))?;
                let output =
                    self.backend.scale_integer(&input, &scalar).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::Transpose => {
                let input = self.matrix(values, node.args[0])?;
                let output = self.backend.transpose(&input).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::Slice { rows, columns } => {
                let input = self.matrix(values, node.args[0])?;
                let rows = rows
                    .as_ref()
                    .map(|range| {
                        Ok::<_, ExecutionError>(RuntimeIndexRange {
                            start: self.eval_usize(node.id, &range.start, env)?,
                            end: self.eval_usize(node.id, &range.end, env)?,
                        })
                    })
                    .transpose()?;
                let columns = columns
                    .as_ref()
                    .map(|range| {
                        Ok::<_, ExecutionError>(RuntimeIndexRange {
                            start: self.eval_usize(node.id, &range.start, env)?,
                            end: self.eval_usize(node.id, &range.end, env)?,
                        })
                    })
                    .transpose()?;
                let output = self
                    .backend
                    .slice(&input, rows.as_ref(), columns.as_ref())
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::Tensor => {
                let left = self.matrix(values, node.args[0])?;
                let right = self.matrix(values, node.args[1])?;
                let output = self.backend.tensor(&left, &right).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::Concat { axis } => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.matrix(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let inputs = inputs.iter().map(Arc::as_ref).collect::<Vec<_>>();
                let output = self.backend.concat(&inputs, *axis).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::UniformResidueSample { .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, wire)?;
                let range = RuntimeSampleRange {
                    minimum: BigInt::from(0),
                    maximum: &ty.modulus - BigInt::from(1),
                };
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_uniform(&ty, &range)
                })?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::UniformIntervalSample { range, .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, wire)?;
                let range = RuntimeSampleRange {
                    minimum: range
                        .minimum
                        .evaluate(env)
                        .map_err(|error| self.expression_error(node.id, error))?,
                    maximum: range
                        .maximum
                        .evaluate(env)
                        .map_err(|error| self.expression_error(node.id, error))?,
                };
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_uniform(&ty, &range)
                })?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, wire)?;
                let sigma = sigma
                    .evaluate_f64(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let max_coefficient_bound = max_coefficient_bound
                    .evaluate(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_gaussian(&ty, sigma, &max_coefficient_bound)
                })?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::HashSample {
                tag_prefix,
                tag_expressions,
                tag_decimal_expressions,
                tag_u64_le_expressions,
                ..
            } => {
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
                for expression in tag_decimal_expressions {
                    let value = expression
                        .evaluate(env)
                        .map_err(|error| self.expression_error(node.id, error))?;
                    tag.extend_from_slice(value.to_string().as_bytes());
                }
                for expression in tag_u64_le_expressions {
                    let value = expression
                        .evaluate(env)
                        .map_err(|error| self.expression_error(node.id, error))?
                        .to_u64()
                        .ok_or_else(|| ExecutionError::Expression {
                            node: node.id,
                            message: "little-endian hash tag component must fit in u64".to_owned(),
                        })?;
                    tag.extend_from_slice(&value.to_le_bytes());
                }
                for wire in node.args.iter().skip(1) {
                    append_tag_integer(&mut tag, &self.int(values, *wire)?);
                }
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, wire)?;
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_hash(&ty, key, &tag, HashVariant::Plain, None)
                })?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } => {
                let matrix_wire = WireRef { node: node.id, port: Port(0) };
                let trapdoor_wire = WireRef { node: node.id, port: Port(1) };
                let ty = self.matrix_type(scope_id, path, matrix_wire)?;
                let sigma = sigma
                    .evaluate_f64(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let gadget_base = gadget_base
                    .evaluate(env)
                    .map_err(|error| self.expression_error(node.id, error))?
                    .abs();
                let digit_count = self.eval_usize(node.id, digit_count, env)?;
                let (public, secret) = self.sample_trapdoor(
                    path,
                    matrix_wire,
                    trapdoor_wire,
                    &ty,
                    sigma,
                    &gadget_base,
                    digit_count,
                )?;
                let public = Arc::new(public);
                self.put(values, node.id, 0, RuntimeValue::Matrix(public.clone()));
                self.put(
                    values,
                    node.id,
                    1,
                    RuntimeValue::Trapdoor {
                        secret: Some(Arc::new(secret)),
                        public,
                        matrix_type: ty,
                        sigma,
                        gadget_base,
                        digit_count,
                        gadget_small: None,
                    },
                );
            }
            NodeKind::PreimageSample { max_coefficient_bound, .. } => {
                let public = self.matrix(values, node.args[0])?;
                let (secret, trapdoor_public, _, sigma, gadget_base, digit_count, gadget_small) =
                    self.trapdoor(values, node.args[1])?;
                if !Arc::ptr_eq(&public, &trapdoor_public) &&
                    public.as_ref() != trapdoor_public.as_ref()
                {
                    return Err(ExecutionError::PreimagePublicMismatch(node.id));
                }
                let target = self.matrix(values, node.args[2])?;
                let target_type = self.matrix_type(scope_id, path, node.args[2])?;
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, wire)?;
                let max_coefficient_bound = max_coefficient_bound
                    .evaluate(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let (value, sampled) = if let Some(small) = gadget_small {
                    self.backend
                        .validate_gadget_layout(&target_type, &gadget_base, digit_count, small)
                        .map_err(Self::backend_error)?;
                    (
                        self.backend
                            .gadget_decompose(&target, small)
                            .map_err(Self::backend_error)?,
                        false,
                    )
                } else {
                    let secret =
                        secret.as_ref().expect("sampled trapdoor must carry secret material");
                    self.sample_matrix_with_status(path, wire, &ty, |backend| {
                        backend.sample_preimage(
                            &ty,
                            sigma,
                            &gadget_base,
                            digit_count,
                            &max_coefficient_bound,
                            secret,
                            &public,
                            &target,
                        )
                    })?
                };
                if sampled {
                    self.record_preimages(1);
                }
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let input = self.matrix(values, node.args[0])?;
                let input_type = self.matrix_type(scope_id, path, node.args[0])?;
                let output_type =
                    self.matrix_type(scope_id, path, WireRef { node: node.id, port: Port(0) })?;
                let base =
                    base.evaluate(env).map_err(|error| self.expression_error(node.id, error))?;
                let digit_count = self.eval_usize(node.id, digit_count, env)?;
                self.backend
                    .validate_gadget_layout(&input_type, &base, digit_count, *small)
                    .map_err(Self::backend_error)?;
                let expected_rows = input_type.rows.checked_mul(digit_count).ok_or_else(|| {
                    ExecutionError::Expression {
                        node: node.id,
                        message: "gadget decomposition output row count overflow".to_owned(),
                    }
                })?;
                if output_type.rows != expected_rows {
                    return Err(ExecutionError::Expression {
                        node: node.id,
                        message: "gadget decomposition output type disagrees with digit count"
                            .to_owned(),
                    });
                }
                let output =
                    self.backend.gadget_decompose(&input, *small).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::MaterializePreimageExact => {
                let input = self.matrix(values, node.args[0])?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(input));
            }
            NodeKind::PreimageBinary(operation) => {
                let left = self.matrix(values, node.args[0])?;
                let right = self.matrix(values, node.args[1])?;
                let output = match operation {
                    mxx_ir_core::node::PreimageBinaryOp::Add => self.backend.add(&left, &right),
                    mxx_ir_core::node::PreimageBinaryOp::RightMultiplyExact |
                    mxx_ir_core::node::PreimageBinaryOp::ComposeExactDecomposition => {
                        self.backend.multiply(&left, &right)
                    }
                }
                .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::PreimageConcatColumns => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.matrix(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let inputs = inputs.iter().map(Arc::as_ref).collect::<Vec<_>>();
                let output = self
                    .backend
                    .concat(&inputs, mxx_ir_core::node::ConcatAxis::Columns)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::DecompositionEntry { row, column } => {
                let input = self.matrix(values, node.args[0])?;
                let row = self.eval_usize(node.id, row, env)?;
                let column = self.eval_usize(node.id, column, env)?;
                let rows = RuntimeIndexRange { start: row, end: row + 1 };
                let columns = RuntimeIndexRange { start: column, end: column + 1 };
                let output = self
                    .backend
                    .slice(&input, Some(&rows), Some(&columns))
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::ExtractCoefficient { position, .. } => {
                let input = self.matrix(values, node.args[0])?;
                let position = self.eval_usize(node.id, position, env)?;
                let output = self
                    .backend
                    .extract_coefficient(&input, position)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Int(output));
            }
            NodeKind::LiftIntegerToConstantPolynomial { .. } => {
                let coefficient = self.int(values, node.args[0])?;
                let ty =
                    self.matrix_type(scope_id, path, WireRef { node: node.id, port: Port(0) })?;
                let identity = self
                    .backend
                    .constant_matrix(&ty, &mxx_ir_core::node::ConstantMatrix::Identity, env)
                    .map_err(Self::backend_error)?;
                let output = self
                    .backend
                    .scale_integer(&identity, &coefficient)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
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
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
                let levels = node
                    .args
                    .iter()
                    .map(|wire| self.matrix(values, *wire).map(|value| value.as_ref().clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                let plaintext_moduli = plaintext_moduli
                    .iter()
                    .map(|value| {
                        value.evaluate(env).map_err(|error| self.expression_error(node.id, error))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let reconstruction_coefficients = reconstruction_coefficients
                    .iter()
                    .map(|value| {
                        value.evaluate(env).map_err(|error| self.expression_error(node.id, error))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output = self
                    .backend
                    .crt_recompose(&levels, &plaintext_moduli, &reconstruction_coefficients)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
                let coefficient_bits = self.eval_usize(node.id, coefficient_bits, env)?;
                let count = self.family_count(values, node.args[0])?;
                let mut bits = Vec::with_capacity(count);
                for index in 0..count {
                    let member = self.family_member(values, node.args[0], index, node.id)?;
                    let RuntimeValue::Bool(bit) = member else {
                        return Err(ExecutionError::ValueKind(node.args[0]));
                    };
                    bits.push(bit);
                }
                let ty = self
                    .validated_wire_type(scope_id, WireRef { node: node.id, port: Port(0) })
                    .and_then(ConcreteWireType::matrix_type)
                    .cloned()
                    .ok_or_else(|| {
                        ExecutionError::MissingMetadata(WireId {
                            instantiation_path: path.to_vec(),
                            wire: WireRef { node: node.id, port: Port(0) },
                        })
                    })?;
                let output = self
                    .backend
                    .pack_polynomial_coefficients(&ty, &bits, coefficient_bits)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::SubgraphCall(call) => {
                let child_id =
                    self.validated.source.child_scope_id(scope_id, node.id).ok_or_else(|| {
                        ExecutionError::MissingSubgraph {
                            node: node.id,
                            name: call.definition.clone(),
                        }
                    })?;
                let child = self.validated.source.scope(&child_id).ok_or_else(|| {
                    ExecutionError::MissingSubgraph { node: node.id, name: call.definition.clone() }
                })?;
                let child_env = self.child_env(env, &call.bindings, None, node.id)?;
                let child_inputs = self.child_inputs(child, node, values)?;
                let mut child_path = path.to_vec();
                child_path.push(InstantiationFrame { call: node.id, loop_index: None });
                let placement = self.backend.active_placement();
                let outputs = self
                    .execute_instance(&child_id, &child_env, child_path, child_inputs, placement)?
                    .outputs;
                for (port, value) in outputs.into_iter().enumerate() {
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::SequentialLoop(loop_node) => {
                let child_id =
                    self.validated.source.child_scope_id(scope_id, node.id).ok_or_else(|| {
                        ExecutionError::MissingSubgraph {
                            node: node.id,
                            name: format!("sequential body at {:?}", node.id),
                        }
                    })?;
                let child = self.validated.source.scope(&child_id).ok_or_else(|| {
                    ExecutionError::MissingSubgraph {
                        node: node.id,
                        name: format!("sequential body at {:?}", node.id),
                    }
                })?;
                let count = self.eval_usize(node.id, &loop_node.count, env)?;
                let parent_placement = self.backend.active_placement();
                let mut carried = node.args[..loop_node.carried_count]
                    .iter()
                    .map(|wire| self.value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let invariants = node.args[loop_node.carried_count..]
                    .iter()
                    .map(|wire| self.value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let input_names = child
                    .inputs()
                    .iter()
                    .map(|wire| {
                        let input = child.node(wire.node).expect("validated sequential input node");
                        let NodeKind::Input { name, .. } = input.kind() else {
                            unreachable!("validated sequential input must reference an input node")
                        };
                        name.clone()
                    })
                    .collect::<Vec<_>>();
                for index in 0..count {
                    self.set_placement(parent_placement)?;
                    let child_env = self.child_env(
                        env,
                        &loop_node.bindings,
                        Some((loop_node.index_slot, index)),
                        node.id,
                    )?;
                    let child_inputs = input_names
                        .iter()
                        .cloned()
                        .zip(carried.iter().chain(&invariants).cloned())
                        .collect::<BTreeMap<_, _>>();
                    let mut child_path = path.to_vec();
                    child_path
                        .push(InstantiationFrame { call: node.id, loop_index: Some(index as u64) });
                    carried = self
                        .execute_instance(
                            &child_id,
                            &child_env,
                            child_path,
                            child_inputs,
                            parent_placement,
                        )?
                        .outputs;
                }
                self.set_placement(parent_placement)?;
                for (port, value) in carried.into_iter().enumerate() {
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::FamilyPack { shape } => {
                let shape = shape
                    .iter()
                    .map(|extent| self.eval_usize(node.id, extent, env))
                    .collect::<Result<Vec<_>, _>>()?;
                let count = shape_product(&shape).ok_or_else(|| ExecutionError::Expression {
                    node: node.id,
                    message: "family shape overflow".into(),
                })?;
                if count == 0 || node.args.len() != count {
                    return Err(ExecutionError::Expression {
                        node: node.id,
                        message: "family pack argument count mismatch".to_owned(),
                    });
                }
                let members = node
                    .args
                    .iter()
                    .map(|wire| self.value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                self.put(values, node.id, 0, RuntimeValue::Family(members));
            }
            NodeKind::FamilyGetStatic { indices } => {
                let shape = self.family_shape(scope_id, node.args[0])?;
                if indices.len() != shape.len() {
                    return Err(ExecutionError::ValueKind(node.args[0]));
                }
                let mut index = 0;
                for (axis, expr) in indices.iter().enumerate() {
                    let value = expr
                        .evaluate(env)
                        .map_err(|error| self.expression_error(node.id, error))?
                        .to_usize()
                        .ok_or_else(|| ExecutionError::Expression {
                            node: node.id,
                            message: "family index is not a nonnegative usize".into(),
                        })?;
                    if value >= shape[axis] {
                        return Err(ExecutionError::SelectIndexOutOfRange {
                            node: node.id,
                            index: value.into(),
                            count: shape[axis],
                        });
                    }
                    index = index * shape[axis] + value;
                }
                let selected = self.family_member(values, node.args[0], index, node.id)?;
                self.put(values, node.id, 0, selected);
            }
            NodeKind::FamilyGetDynamic { rank } => {
                let shape = self.family_shape(scope_id, node.args[0])?;
                if *rank != shape.len() || node.args.len() != rank.saturating_add(1) {
                    return Err(ExecutionError::ValueKind(node.args[0]));
                }
                let mut index = 0usize;
                for (axis, wire) in node.args.iter().skip(1).enumerate() {
                    let value = self
                        .int(values, *wire)?
                        .to_usize()
                        .ok_or(ExecutionError::ValueKind(*wire))?;
                    if value >= shape[axis] {
                        return Err(ExecutionError::SelectIndexOutOfRange {
                            node: node.id,
                            index: value.into(),
                            count: shape[axis],
                        });
                    }
                    index = index * shape[axis] + value;
                }
                let selected = self.family_member(values, node.args[0], index, node.id)?;
                self.put(values, node.id, 0, selected);
            }
            NodeKind::FamilyGather { output_shape, input_rank } => {
                let output_shape = output_shape
                    .iter()
                    .map(|extent| self.eval_usize(node.id, extent, env))
                    .collect::<Result<Vec<_>, _>>()?;
                let output_count = output_shape
                    .iter()
                    .try_fold(1usize, |product, extent| product.checked_mul(*extent))
                    .ok_or_else(|| ExecutionError::Expression {
                        node: node.id,
                        message: "family gather shape overflow".into(),
                    })?;
                if node.args.len() != input_rank.saturating_add(1) {
                    return Err(ExecutionError::ValueKind(node.args[0]));
                }
                let mut output = Vec::with_capacity(output_count);
                for lane in 0..output_count {
                    let mut coordinates = vec![0usize; output_shape.len()];
                    let mut remainder = lane;
                    for axis in (0..output_shape.len()).rev() {
                        coordinates[axis] = remainder % output_shape[axis];
                        remainder /= output_shape[axis];
                    }
                    let mut source_index = 0usize;
                    for (axis, selector_wire) in node.args.iter().skip(1).enumerate() {
                        let selector = self.family_member(values, *selector_wire, lane, node.id)?;
                        let index = match selector {
                            RuntimeValue::Int(value) => value.to_usize(),
                            _ => None,
                        }
                        .ok_or(ExecutionError::ValueKind(*selector_wire))?;
                        let source_shape = self.family_shape(scope_id, node.args[0])?;
                        if index >= source_shape[axis] {
                            return Err(ExecutionError::SelectIndexOutOfRange {
                                node: node.id,
                                index: index.into(),
                                count: source_shape[axis],
                            });
                        }
                        source_index = source_index
                            .checked_mul(source_shape[axis])
                            .and_then(|value| value.checked_add(index))
                            .ok_or_else(|| ExecutionError::Expression {
                                node: node.id,
                                message: "family gather index overflow".into(),
                            })?;
                    }
                    output.push(self.family_member(values, node.args[0], source_index, node.id)?);
                }
                self.put(values, node.id, 0, RuntimeValue::Family(output));
            }
            NodeKind::FamilySelectAxis { axis } => {
                let input_shape = self.family_shape(scope_id, node.args[0])?;
                let output_wire = WireRef { node: node.id, port: Port(0) };
                let output_shape = match self
                    .validated
                    .scope(scope_id)
                    .and_then(|scope| scope.wire_types.get(&output_wire))
                {
                    Some(ConcreteWireType::Family { shape, .. }) => shape.clone(),
                    Some(_) => Vec::new(),
                    None => {
                        return Err(ExecutionError::MissingMetadata(WireId {
                            instantiation_path: Vec::new(),
                            wire: output_wire,
                        }))
                    }
                };
                if *axis >= input_shape.len() {
                    return Err(ExecutionError::ValueKind(node.args[0]));
                }
                let count =
                    shape_product(&output_shape).ok_or_else(|| ExecutionError::Expression {
                        node: node.id,
                        message: "family shape overflow".into(),
                    })?;
                let selector = values
                    .get(&node.args[1])
                    .cloned()
                    .ok_or(ExecutionError::MissingWire(node.args[1]))?;
                if output_shape.is_empty() {
                    let selected = match selector {
                        RuntimeValue::Int(value) => value.to_usize(),
                        _ => None,
                    }
                    .ok_or(ExecutionError::ValueKind(node.args[1]))?;
                    if input_shape.len() != 1 || selected >= input_shape[0] {
                        return Err(ExecutionError::ValueKind(node.args[1]));
                    }
                    let member = self.family_member(values, node.args[0], selected, node.id)?;
                    self.put(values, node.id, 0, member);
                    return Ok(());
                }
                let mut result = Vec::with_capacity(count);
                for lane in 0..count {
                    let selected = match &selector {
                        RuntimeValue::Int(value) => value.to_usize(),
                        _ => match self.family_member(values, node.args[1], lane, node.id)? {
                            RuntimeValue::Int(value) => value.to_usize(),
                            _ => None,
                        },
                    }
                    .ok_or(ExecutionError::ValueKind(node.args[1]))?;
                    if selected >= input_shape[*axis] {
                        return Err(ExecutionError::SelectIndexOutOfRange {
                            node: node.id,
                            index: selected.into(),
                            count: input_shape[*axis],
                        });
                    }
                    let output_coordinates = row_major_coordinates(lane, &output_shape);
                    let mut input_coordinates = Vec::with_capacity(input_shape.len());
                    let mut output_axis = 0;
                    for input_axis in 0..input_shape.len() {
                        if input_axis == *axis {
                            input_coordinates.push(selected);
                        } else {
                            input_coordinates.push(output_coordinates[output_axis]);
                            output_axis += 1;
                        }
                    }
                    result.push(self.family_member(
                        values,
                        node.args[0],
                        row_major_offset(&input_coordinates, &input_shape),
                        node.id,
                    )?);
                }
                self.put(values, node.id, 0, RuntimeValue::Family(result));
            }
            NodeKind::FamilyReindex { map, .. } => {
                let input_shape = self.family_shape(scope_id, node.args[0])?;
                let output_shape =
                    self.family_shape(scope_id, WireRef { node: node.id, port: Port(0) })?;
                let count =
                    shape_product(&output_shape).ok_or_else(|| ExecutionError::Expression {
                        node: node.id,
                        message: "family shape overflow".into(),
                    })?;
                let mut result = Vec::with_capacity(count);
                for lane in 0..count {
                    let coordinates = row_major_coordinates(lane, &output_shape);
                    let mapped = map
                        .input_indices
                        .iter()
                        .map(|expr| eval_index_expr(expr, &coordinates, env))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|message| ExecutionError::Expression { node: node.id, message })?;
                    if mapped.len() != input_shape.len() ||
                        mapped.iter().enumerate().any(|(axis, value)| {
                            value.to_usize().is_none_or(|index| index >= input_shape[axis])
                        })
                    {
                        return Err(ExecutionError::Expression {
                            node: node.id,
                            message: "family reindex coordinate is out of range".into(),
                        });
                    }
                    let mapped = mapped
                        .into_iter()
                        .map(|value| value.to_usize().expect("checked index"))
                        .collect::<Vec<_>>();
                    result.push(self.family_member(
                        values,
                        node.args[0],
                        row_major_offset(&mapped, &input_shape),
                        node.id,
                    )?);
                }
                self.put(values, node.id, 0, RuntimeValue::Family(result));
            }
            NodeKind::FamilyPreimageSample { .. } => {
                self.execute_family_preimage(scope_id, env, path, node, values)?;
            }
            NodeKind::ParallelGrid(grid) => {
                let child_id =
                    self.validated.source.child_scope_id(scope_id, node.id).ok_or_else(|| {
                        ExecutionError::MissingSubgraph {
                            node: node.id,
                            name: "parallel grid body".into(),
                        }
                    })?;
                let child = self.validated.source.scope(&child_id).ok_or_else(|| {
                    ExecutionError::MissingSubgraph {
                        node: node.id,
                        name: "parallel grid body".into(),
                    }
                })?;
                let shape = grid
                    .shape
                    .iter()
                    .map(|extent| self.eval_usize(node.id, extent, env))
                    .collect::<Result<Vec<_>, _>>()?;
                let count = shape_product(&shape).ok_or_else(|| ExecutionError::Expression {
                    node: node.id,
                    message: "parallel grid shape overflow".into(),
                })?;
                if grid.index_slots.len() != shape.len() ||
                    grid.input_modes.len() != node.args.len()
                {
                    return Err(ExecutionError::ValueKind(WireRef { node: node.id, port: Port(0) }));
                }
                let mut outputs = (0..child.outputs().len())
                    .map(|_| Vec::with_capacity(count))
                    .collect::<Vec<_>>();
                for lane in 0..count {
                    let coordinates = row_major_coordinates(lane, &shape);
                    let mut child_env = self.child_env(env, &grid.bindings, None, node.id)?;
                    for (slot, coordinate) in
                        grid.index_slots.iter().zip(coordinates.iter().copied())
                    {
                        child_env.loop_indices.insert(*slot, BigInt::from(coordinate));
                    }
                    let mut child_path = path.to_vec();
                    child_path
                        .push(InstantiationFrame { call: node.id, loop_index: Some(lane as u64) });
                    let child_inputs = self.grid_child_inputs(
                        scope_id,
                        child,
                        node,
                        values,
                        &grid.input_modes,
                        &coordinates,
                        &child_env,
                    )?;
                    let instance = self.execute_instance(
                        &child_id,
                        &child_env,
                        child_path,
                        child_inputs,
                        self.backend.active_placement(),
                    )?;
                    for (port, value) in instance.outputs.into_iter().enumerate() {
                        outputs[port].push(value);
                    }
                }
                for (port, output) in outputs.into_iter().enumerate() {
                    self.put(values, node.id, port as u32, RuntimeValue::Family(output));
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

    fn family_count(
        &mut self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<usize, ExecutionError> {
        match values.get(&wire).ok_or(ExecutionError::MissingWire(wire))? {
            RuntimeValue::LazyArtifactFamily { descriptor, .. } |
            RuntimeValue::StagedArtifactFamily { descriptor, .. } => descriptor
                .family_shape
                .as_deref()
                .and_then(shape_product)
                .ok_or(ExecutionError::ValueKind(wire)),
            RuntimeValue::Family(values) => Ok(values.len()),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn family_shape(
        &self,
        scope_id: &FrozenGraphScopeId,
        wire: WireRef,
    ) -> Result<Vec<usize>, ExecutionError> {
        let ty = self
            .validated
            .scope(scope_id)
            .and_then(|scope| scope.wire_types.get(&wire))
            .ok_or_else(|| {
                ExecutionError::MissingMetadata(WireId { instantiation_path: Vec::new(), wire })
            })?;
        match ty {
            ConcreteWireType::Family { shape, .. } => Ok(shape.clone()),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn execute_family_preimage(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        path: &[InstantiationFrame],
        node: &ExecutableNode<'_>,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
    ) -> Result<(), ExecutionError> {
        // A cardinality-one family is still a family: its sole artifact must
        // be loaded through family_member rather than treated as a scalar.
        let source_family_count = self.family_count(values, node.args[0]).ok();
        let trapdoor_family_count = self.family_count(values, node.args[1]).ok();
        if source_family_count != trapdoor_family_count {
            return Err(ExecutionError::ValueKind(node.args[1]));
        }
        let source_count = source_family_count.unwrap_or(1);
        let target_count = self.family_count(values, node.args[2])?;
        if source_count == 0 || target_count % source_count != 0 {
            return Err(ExecutionError::ValueKind(node.args[2]));
        }
        let branch_count = target_count / source_count;
        let max_coefficient_bound = match node.kind {
            NodeKind::FamilyPreimageSample { max_coefficient_bound, .. } => max_coefficient_bound
                .evaluate(env)
                .map_err(|error| self.expression_error(node.id, error))?,
            _ => unreachable!("family preimage helper called for another node"),
        };
        let output_wire = WireRef { node: node.id, port: Port(0) };
        let matrix_type = self
            .validated_wire_type(scope_id, output_wire)
            .and_then(|ty| match ty {
                ConcreteWireType::Family { element, .. } => element.matrix_type().cloned(),
                _ => ty.matrix_type().cloned(),
            })
            .ok_or_else(|| {
                ExecutionError::MissingMetadata(WireId {
                    instantiation_path: path.to_vec(),
                    wire: output_wire,
                })
            })?;
        let placement = self.backend.active_placement();
        let mut pending = Vec::with_capacity(target_count);
        for lane in 0..target_count {
            let group = lane / branch_count;
            let public = if source_family_count.is_some() {
                match self.family_member(values, node.args[0], group, node.id)? {
                    RuntimeValue::Matrix(value) => value,
                    _ => return Err(ExecutionError::ValueKind(node.args[0])),
                }
            } else {
                self.matrix(values, node.args[0])?
            };
            let trapdoor = if trapdoor_family_count.is_some() {
                self.family_member(values, node.args[1], group, node.id)?
            } else {
                self.materialize(values, node.args[1])?
            };
            let RuntimeValue::Trapdoor {
                secret: Some(secret),
                public: trapdoor_public,
                sigma,
                gadget_base,
                digit_count,
                ..
            } = trapdoor
            else {
                return Err(ExecutionError::ValueKind(node.args[1]));
            };
            if public.as_ref() != trapdoor_public.as_ref() {
                return Err(ExecutionError::PreimagePublicMismatch(node.id));
            }
            let target = match self.family_member(values, node.args[2], lane, node.id)? {
                RuntimeValue::Matrix(value) => value,
                _ => return Err(ExecutionError::ValueKind(node.args[2])),
            };
            let mut lane_path = path.to_vec();
            // K[i,j] is one stochastic draw per flattened (i,j) lane. Adding
            // the lane to the draw path gives record/replay and sessions a
            // stable, collision-free identity for every sampled preimage.
            lane_path.push(InstantiationFrame { call: node.id, loop_index: Some(lane as u64) });
            pending.push(PreparedPreimage {
                placement,
                site: DrawSite {
                    instantiation_path: lane_path,
                    node: output_wire.node,
                    port: output_wire.port,
                },
                request: PreimageRequest {
                    matrix_type: matrix_type.clone(),
                    sigma,
                    gadget_base,
                    digit_count,
                    max_coefficient_bound: max_coefficient_bound.clone(),
                    trapdoor: secret,
                    public,
                    target,
                },
            });
        }
        let outputs = self
            .sample_preimage_requests(&pending)?
            .into_iter()
            .map(RuntimeValue::matrix)
            .collect();
        self.put(values, node.id, 0, RuntimeValue::Family(outputs));
        Ok(())
    }

    fn family_member(
        &mut self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
        index: usize,
        node: NodeId,
    ) -> Result<RuntimeValue<B>, ExecutionError> {
        let count = self.family_count(values, wire)?;
        if index >= count {
            return Err(ExecutionError::SelectIndexOutOfRange {
                node,
                index: BigInt::from(index),
                count,
            });
        }
        let member = self.family_member_value(values, wire, index)?;
        self.materialize_value(member)
    }

    fn family_member_value(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
        index: usize,
    ) -> Result<RuntimeValue<B>, ExecutionError> {
        match values.get(&wire).ok_or(ExecutionError::MissingWire(wire))? {
            RuntimeValue::LazyArtifactFamily { production, name, descriptor }
                if descriptor
                    .family_shape
                    .as_deref()
                    .and_then(shape_product)
                    .is_some_and(|count| index < count) =>
            {
                Ok(RuntimeValue::LazyArtifact {
                    production: production.clone(),
                    name: name.clone(),
                    index: Some(index),
                    descriptor: descriptor.clone(),
                })
            }
            RuntimeValue::StagedArtifactFamily { production, name, descriptor }
                if descriptor
                    .family_shape
                    .as_deref()
                    .and_then(shape_product)
                    .is_some_and(|count| index < count) =>
            {
                Ok(RuntimeValue::StagedArtifact {
                    production: production.clone(),
                    name: name.clone(),
                    index,
                    descriptor: descriptor.clone(),
                })
            }
            RuntimeValue::Family(values) => {
                values.get(index).cloned().ok_or(ExecutionError::ValueKind(wire))
            }
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn execute_preimage_batch(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        envs: &[ParamEnv],
        paths: &[Vec<InstantiationFrame>],
        placements: &[usize],
        node: &ExecutableNode<'_>,
        values: &mut [BTreeMap<WireRef, RuntimeValue<B>>],
    ) -> Result<(), ExecutionError> {
        let mut destinations = Vec::new();
        let mut pending = Vec::new();
        for instance in 0..values.len() {
            self.set_placement(placements[instance])?;
            let public = self.matrix(&mut values[instance], node.args[0])?;
            let (secret, trapdoor_public, _, sigma, gadget_base, digit_count, gadget_small) =
                self.trapdoor(&mut values[instance], node.args[1])?;
            if !Arc::ptr_eq(&public, &trapdoor_public) &&
                public.as_ref() != trapdoor_public.as_ref()
            {
                return Err(ExecutionError::PreimagePublicMismatch(node.id));
            }
            let target = self.matrix(&mut values[instance], node.args[2])?;
            let wire = WireRef { node: node.id, port: Port(0) };
            if let Some(small) = gadget_small {
                let value =
                    self.backend.gadget_decompose(&target, small).map_err(Self::backend_error)?;
                self.put(&mut values[instance], node.id, 0, RuntimeValue::matrix(value));
                continue;
            }
            let matrix_type = self.matrix_type(scope_id, &paths[instance], wire)?;
            let NodeKind::PreimageSample { max_coefficient_bound, .. } = node.kind else {
                unreachable!("preimage batch only handles preimage nodes")
            };
            let max_coefficient_bound = max_coefficient_bound
                .evaluate(&envs[instance])
                .map_err(|error| self.expression_error(node.id, error))?;
            destinations.push(instance);
            pending.push(PreparedPreimage {
                placement: placements[instance],
                site: DrawSite {
                    instantiation_path: paths[instance].clone(),
                    node: wire.node,
                    port: wire.port,
                },
                request: PreimageRequest {
                    matrix_type,
                    sigma,
                    gadget_base,
                    digit_count,
                    max_coefficient_bound,
                    trapdoor: secret.expect("sampled trapdoor must carry secret material"),
                    public,
                    target,
                },
            });
        }
        if pending.is_empty() {
            return Ok(());
        }
        let outputs = self.sample_preimage_requests(&pending)?;
        for (instance, output) in destinations.into_iter().zip(outputs) {
            self.put(&mut values[instance], node.id, 0, RuntimeValue::matrix(output));
        }
        Ok(())
    }

    fn sample_preimage_requests(
        &mut self,
        pending: &[PreparedPreimage<B::Matrix, B::Trapdoor>],
    ) -> Result<Vec<B::Matrix>, ExecutionError> {
        // Both ordinary loop batches and FamilyPreimageSample use this path,
        // so session recovery, transcript replay, progress accounting, and
        // backend batching cannot diverge between the two representations.
        if let Some(production) = self.session.clone() {
            let mut outputs = (0..pending.len()).map(|_| None).collect::<Vec<_>>();
            let mut missing = Vec::new();
            for (index, prepared) in pending.iter().enumerate() {
                match self
                    .artifact_store
                    .transcript_entry(&production, &prepared.site)
                    .map_err(Self::artifact_error)?
                {
                    Some(RecordedValue::Matrix { matrix_type, bytes })
                        if matrix_type == prepared.request.matrix_type =>
                    {
                        self.set_placement(prepared.placement)?;
                        outputs[index] = Some(
                            self.backend
                                .matrix_from_bytes(&prepared.request.matrix_type, &bytes)
                                .map_err(Self::backend_error)?,
                        );
                    }
                    Some(RecordedValue::Matrix { .. } | RecordedValue::Trapdoor { .. }) => {
                        return Err(TranscriptError::KindMismatch(prepared.site.clone()).into());
                    }
                    None => missing.push(index),
                }
            }
            if !missing.is_empty() {
                let mut sampled = self.sample_preimage_indices(pending, &missing)?;
                let serialized = self.backend.matrices_to_bytes(
                    &missing
                        .iter()
                        .map(|index| {
                            sampled[*index].as_ref().expect("every missing preimage was sampled")
                        })
                        .collect::<Vec<_>>(),
                );
                let entries = missing
                    .iter()
                    .zip(serialized)
                    .map(|(index, bytes)| {
                        (
                            pending[*index].site.clone(),
                            RecordedValue::Matrix {
                                matrix_type: pending[*index].request.matrix_type.clone(),
                                bytes,
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                self.artifact_store
                    .record_transcript_batch(&production, &entries)
                    .map_err(Self::artifact_error)?;
                for index in missing {
                    outputs[index] = sampled[index].take();
                }
            }
            return Ok(outputs
                .into_iter()
                .map(|output| output.expect("every session preimage draw is resolved"))
                .collect());
        }

        let replayed = match &self.sampling_mode {
            SamplingMode::Replay(replayer) => {
                let recorded = pending
                    .iter()
                    .map(|prepared| replayer.get(&prepared.site).cloned())
                    .collect::<Result<Vec<_>, _>>()?;
                let mut outputs = Vec::with_capacity(pending.len());
                for (prepared, recorded) in pending.iter().zip(recorded) {
                    match recorded {
                        RecordedValue::Matrix { matrix_type, bytes }
                            if matrix_type == prepared.request.matrix_type =>
                        {
                            self.set_placement(prepared.placement)?;
                            outputs.push(
                                self.backend
                                    .matrix_from_bytes(&prepared.request.matrix_type, &bytes)
                                    .map_err(Self::backend_error)?,
                            );
                        }
                        RecordedValue::Matrix { .. } | RecordedValue::Trapdoor { .. } => {
                            return Err(TranscriptError::KindMismatch(prepared.site.clone()).into());
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
            let indices = (0..pending.len()).collect::<Vec<_>>();
            self.sample_preimage_indices(pending, &indices)?
                .into_iter()
                .map(|output| output.expect("every preimage request was sampled"))
                .collect()
        };
        if let SamplingMode::Record(recorder) = &mut self.sampling_mode {
            let serialized = self.backend.matrices_to_bytes(&outputs.iter().collect::<Vec<_>>());
            for (prepared, bytes) in pending.iter().zip(serialized) {
                recorder.record(
                    prepared.site.clone(),
                    RecordedValue::Matrix {
                        matrix_type: prepared.request.matrix_type.clone(),
                        bytes,
                    },
                )?;
            }
        }
        Ok(outputs)
    }

    fn sample_preimage_indices(
        &mut self,
        pending: &[PreparedPreimage<B::Matrix, B::Trapdoor>],
        indices: &[usize],
    ) -> Result<Vec<Option<B::Matrix>>, ExecutionError> {
        // Requests are grouped once per placement. Each group becomes one
        // backend batch, keeping all limbs of a matrix on the same device and
        // avoiding one sampler launch per family lane.
        let mut outputs = (0..pending.len()).map(|_| None).collect::<Vec<_>>();
        let groups = (0..self.backend.placement_count())
            .filter_map(|placement| {
                let group_indices = indices
                    .iter()
                    .copied()
                    .filter(|index| pending[*index].placement == placement)
                    .collect::<Vec<_>>();
                (!group_indices.is_empty()).then(|| {
                    let requests = group_indices
                        .iter()
                        .map(|index| pending[*index].request.clone())
                        .collect::<Vec<_>>();
                    (placement, group_indices, requests)
                })
            })
            .collect::<Vec<_>>();
        let batches =
            groups.iter().map(|(placement, _, requests)| (*placement, requests.clone())).collect();
        let sampled_groups = self
            .backend
            .sample_preimage_batches_by_placement(batches)
            .map_err(Self::backend_error)?;
        for ((expected_placement, group_indices, _), (placement, sampled)) in
            groups.into_iter().zip(sampled_groups)
        {
            debug_assert_eq!(placement, expected_placement);
            self.record_preimages(sampled.len());
            for (index, output) in group_indices.into_iter().zip(sampled) {
                outputs[index] = Some(output);
            }
        }
        Ok(outputs)
    }

    fn execute_select(
        &mut self,
        env: &ParamEnv,
        node: &ExecutableNode<'_>,
        schedule: &mxx_ir_core::LivenessSchedule,
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
            !schedule.retained.contains(&selected_wire);
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
        self.sample_matrix_with_status(path, wire, ty, fresh).map(|(value, _)| value)
    }

    fn sample_matrix_with_status<F>(
        &mut self,
        path: &[InstantiationFrame],
        wire: WireRef,
        ty: &ConcreteMatrixType,
        fresh: F,
    ) -> Result<(B::Matrix, bool), ExecutionError>
    where
        F: FnOnce(&mut B) -> Result<B::Matrix, B::Error>,
    {
        let site = DrawSite { instantiation_path: path.to_vec(), node: wire.node, port: wire.port };
        if let Some(production) = self.session.clone() {
            if let Some(recorded) = self
                .artifact_store
                .transcript_entry(&production, &site)
                .map_err(Self::artifact_error)?
            {
                return match recorded {
                    RecordedValue::Matrix { matrix_type, bytes } if matrix_type == *ty => self
                        .backend
                        .matrix_from_bytes(ty, &bytes)
                        .map(|value| (value, false))
                        .map_err(Self::backend_error),
                    RecordedValue::Matrix { .. } | RecordedValue::Trapdoor { .. } => {
                        Err(TranscriptError::KindMismatch(site).into())
                    }
                };
            }
            let value = fresh(self.backend).map_err(Self::backend_error)?;
            self.artifact_store
                .record_transcript_batch(
                    &production,
                    &[(
                        site,
                        RecordedValue::Matrix {
                            matrix_type: ty.clone(),
                            bytes: self.backend.matrix_to_bytes(&value),
                        },
                    )],
                )
                .map_err(Self::artifact_error)?;
            return Ok((value, true));
        }
        match &mut self.sampling_mode {
            SamplingMode::Fresh => {
                fresh(self.backend).map(|value| (value, true)).map_err(Self::backend_error)
            }
            SamplingMode::Record(recorder) => {
                let value = fresh(self.backend).map_err(Self::backend_error)?;
                recorder.record(
                    site,
                    RecordedValue::Matrix {
                        matrix_type: ty.clone(),
                        bytes: self.backend.matrix_to_bytes(&value),
                    },
                )?;
                Ok((value, true))
            }
            SamplingMode::Replay(replayer) => match replayer.get(&site)? {
                RecordedValue::Matrix { bytes, .. } => self
                    .backend
                    .matrix_from_bytes(ty, bytes)
                    .map(|value| (value, false))
                    .map_err(Self::backend_error),
                RecordedValue::Trapdoor { .. } => Err(TranscriptError::KindMismatch(site).into()),
            },
        }
    }

    fn record_preimages(&mut self, count: usize) {
        if let Some(progress) = &mut self.preimage_progress {
            progress.record(count);
        }
    }

    fn finish_preimage_progress(&self) -> Result<(), ExecutionError> {
        self.preimage_progress.as_ref().map_or(Ok(()), PreimageProgress::finish)
    }

    fn sample_trapdoor(
        &mut self,
        path: &[InstantiationFrame],
        matrix_wire: WireRef,
        trapdoor_wire: WireRef,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
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
        if let Some(production) = self.session.clone() {
            let recorded_public = self
                .artifact_store
                .transcript_entry(&production, &matrix_site)
                .map_err(Self::artifact_error)?;
            let recorded_trapdoor = self
                .artifact_store
                .transcript_entry(&production, &trapdoor_site)
                .map_err(Self::artifact_error)?;
            return match (recorded_public, recorded_trapdoor) {
                (
                    Some(RecordedValue::Matrix { matrix_type, bytes }),
                    Some(RecordedValue::Trapdoor {
                        matrix_type: secret_type,
                        public_bytes,
                        trapdoor_bytes,
                    }),
                ) if matrix_type == *ty && secret_type == *ty && bytes == public_bytes => {
                    let public =
                        self.backend.matrix_from_bytes(ty, &bytes).map_err(Self::backend_error)?;
                    let secret = self
                        .backend
                        .trapdoor_from_bytes(ty, &trapdoor_bytes)
                        .map_err(Self::backend_error)?;
                    Ok((public, secret))
                }
                (None, None) => {
                    let (public, secret) = self
                        .backend
                        .sample_trapdoor(ty, sigma, gadget_base, digit_count)
                        .map_err(Self::backend_error)?;
                    let public_bytes = self.backend.matrix_to_bytes(&public);
                    self.artifact_store
                        .record_transcript_batch(
                            &production,
                            &[
                                (
                                    matrix_site,
                                    RecordedValue::Matrix {
                                        matrix_type: ty.clone(),
                                        bytes: public_bytes.clone(),
                                    },
                                ),
                                (
                                    trapdoor_site,
                                    RecordedValue::Trapdoor {
                                        matrix_type: ty.clone(),
                                        public_bytes,
                                        trapdoor_bytes: self.backend.trapdoor_to_bytes(&secret),
                                    },
                                ),
                            ],
                        )
                        .map_err(Self::artifact_error)?;
                    Ok((public, secret))
                }
                _ => Err(ExecutionError::Manifest(
                    "session contains an incomplete or inconsistent trapdoor draw".to_owned(),
                )),
            };
        }
        match &mut self.sampling_mode {
            SamplingMode::Fresh => self
                .backend
                .sample_trapdoor(ty, sigma, gadget_base, digit_count)
                .map_err(Self::backend_error),
            SamplingMode::Record(recorder) => {
                let (public, secret) = self
                    .backend
                    .sample_trapdoor(ty, sigma, gadget_base, digit_count)
                    .map_err(Self::backend_error)?;
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
        let was_lazy = matches!(
            value,
            RuntimeValue::LazyArtifact { .. } | RuntimeValue::StagedArtifact { .. }
        );
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
        if let RuntimeValue::LazyArtifact { production, name, index, descriptor } = value {
            let key = ArtifactKey { production, name, index };
            let artifact_type = descriptor.artifact_type.clone();
            let payload =
                self.artifact_store.load(&key, &descriptor).map_err(Self::artifact_error)?;
            self.decode_artifact(artifact_type, payload)
        } else if let RuntimeValue::StagedArtifact { production, name, index, descriptor } = value {
            let key = ArtifactKey { production, name, index: Some(index) };
            let artifact_type = descriptor.artifact_type.clone();
            let payload =
                self.artifact_store.load_staged(&key, &descriptor).map_err(Self::artifact_error)?;
            self.decode_artifact(artifact_type, payload)
        } else {
            Ok(value)
        }
    }

    fn decode_artifact(
        &self,
        artifact_type: ArtifactType,
        payload: ArtifactPayload,
    ) -> Result<RuntimeValue<B>, ExecutionError> {
        decode_artifact(self.backend, artifact_type, payload)
    }

    fn matrix(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<Arc<B::Matrix>, ExecutionError> {
        match self.materialize(values, wire)? {
            RuntimeValue::Matrix(value) => Ok(value),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn fence_pending_releases(&mut self) -> Result<(), ExecutionError> {
        if self.has_pending_releases {
            self.backend.fence_released_memory().map_err(Self::backend_error)?;
            self.has_pending_releases = false;
            self.last_release_fence_node_count = self.executed_node_count;
        }
        Ok(())
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
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<Vec<u8>, ExecutionError> {
        match self.materialize(values, wire)? {
            RuntimeValue::Bytes(value) => Ok(value),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn trapdoor(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<TrapdoorParts<B>, ExecutionError> {
        match self.materialize(values, wire)? {
            RuntimeValue::Trapdoor {
                secret,
                public,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                gadget_small,
            } => Ok((secret, public, matrix_type, sigma, gadget_base, digit_count, gadget_small)),
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

    fn set_placement(&mut self, placement: usize) -> Result<(), ExecutionError> {
        if self.backend.set_active_placement(placement) {
            Ok(())
        } else {
            Err(ExecutionError::BackendPlacement {
                placement,
                count: self.backend.placement_count(),
            })
        }
    }

    fn value_for_placement(
        &mut self,
        value: RuntimeValue<B>,
        placement: usize,
    ) -> Result<RuntimeValue<B>, ExecutionError> {
        self.set_placement(placement)?;
        Ok(match value {
            RuntimeValue::Matrix(matrix) => {
                if self.backend.matrix_is_on_active_placement(matrix.as_ref()) {
                    RuntimeValue::Matrix(matrix)
                } else {
                    RuntimeValue::matrix(
                        self.backend
                            .matrix_to_active_placement(matrix.as_ref())
                            .map_err(Self::backend_error)?,
                    )
                }
            }
            RuntimeValue::Trapdoor {
                secret,
                public,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                gadget_small,
            } => {
                if self.backend.matrix_is_on_active_placement(public.as_ref()) {
                    RuntimeValue::Trapdoor {
                        secret,
                        public,
                        matrix_type,
                        sigma,
                        gadget_base,
                        digit_count,
                        gadget_small,
                    }
                } else {
                    let public = Arc::new(
                        self.backend
                            .matrix_to_active_placement(public.as_ref())
                            .map_err(Self::backend_error)?,
                    );
                    let secret = secret
                        .map(|secret| {
                            self.backend
                                .trapdoor_to_active_placement(&matrix_type, secret.as_ref())
                                .map(Arc::new)
                                .map_err(Self::backend_error)
                        })
                        .transpose()?;
                    RuntimeValue::Trapdoor {
                        secret,
                        public,
                        matrix_type,
                        sigma,
                        gadget_base,
                        digit_count,
                        gadget_small,
                    }
                }
            }
            RuntimeValue::Family(values) => RuntimeValue::Family(
                values
                    .into_iter()
                    .map(|value| self.value_for_placement(value, placement))
                    .collect::<Result<_, _>>()?,
            ),
            value => value,
        })
    }

    fn matrix_type(
        &self,
        scope_id: &FrozenGraphScopeId,
        path: &[InstantiationFrame],
        wire: WireRef,
    ) -> Result<ConcreteMatrixType, ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        self.validated_wire_type(scope_id, wire)
            .and_then(|wire| wire.matrix_type().cloned())
            .ok_or(ExecutionError::MissingMetadata(id))
    }

    fn validated_wire_type(
        &self,
        scope_id: &FrozenGraphScopeId,
        wire: WireRef,
    ) -> Option<&ConcreteWireType> {
        self.validated.scope(scope_id)?.wire_types.get(&wire)
    }

    fn trapdoor_type(
        &self,
        scope_id: &FrozenGraphScopeId,
        path: &[InstantiationFrame],
        wire: WireRef,
    ) -> Result<ConcreteMatrixType, ExecutionError> {
        self.matrix_type(scope_id, path, wire)
    }

    fn trapdoor_sigma(
        &self,
        scope_id: &FrozenGraphScopeId,
        path: &[InstantiationFrame],
        wire: WireRef,
    ) -> Result<f64, ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        match self.validated_wire_type(scope_id, wire) {
            Some(ConcreteWireType::Trapdoor { sigma, .. }) => sigma
                .evaluate_f64(&ParamEnv::default())
                .map_err(|error| self.expression_error(wire.node, error)),
            _ => Err(ExecutionError::MissingMetadata(id)),
        }
    }

    fn trapdoor_layout(
        &self,
        scope_id: &FrozenGraphScopeId,
        path: &[InstantiationFrame],
        wire: WireRef,
    ) -> Result<(BigInt, usize), ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        match self.validated_wire_type(scope_id, wire) {
            Some(ConcreteWireType::Trapdoor { gadget_base, digit_count, .. }) => {
                Ok((gadget_base.clone(), *digit_count))
            }
            _ => Err(ExecutionError::MissingMetadata(id)),
        }
    }

    fn child_inputs(
        &self,
        child: &GraphScope,
        node: &ExecutableNode<'_>,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
    ) -> Result<BTreeMap<String, RuntimeValue<B>>, ExecutionError> {
        let names = child.inputs().iter().map(|wire| {
            let input = child.node(wire.node).expect("validated child input node");
            let NodeKind::Input { name, .. } = input.kind() else {
                unreachable!("validated child input must reference an input node")
            };
            name
        });
        names
            .zip(&node.args)
            .map(|(name, wire)| Ok((name.clone(), self.value(values, *wire)?)))
            .collect()
    }

    fn grid_child_inputs(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        child: &GraphScope,
        node: &ExecutableNode<'_>,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        modes: &[GridInputMode],
        coordinates: &[usize],
        env: &ParamEnv,
    ) -> Result<BTreeMap<String, RuntimeValue<B>>, ExecutionError> {
        if modes.len() != node.args.len() || child.inputs().len() != node.args.len() {
            return Err(ExecutionError::ValueKind(WireRef { node: node.id, port: Port(0) }));
        }
        child
            .inputs()
            .iter()
            .zip(node.args.iter())
            .zip(modes)
            .map(|((input, wire), mode)| {
                let input_node = child.node(input.node).expect("validated child input node");
                let NodeKind::Input { name, .. } = input_node.kind() else {
                    unreachable!("validated child input must reference an input node")
                };
                let value = match mode {
                    GridInputMode::Broadcast => self.value(values, *wire)?,
                    GridInputMode::Reindex { map } => {
                        let input_shape = self.family_shape(scope_id, *wire)?;
                        let mapped = map
                            .input_indices
                            .iter()
                            .map(|expr| eval_index_expr(expr, coordinates, env))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|message| ExecutionError::Expression {
                                node: node.id,
                                message,
                            })?;
                        if mapped.len() != input_shape.len() {
                            return Err(ExecutionError::ValueKind(*wire));
                        }
                        let mapped = mapped
                            .into_iter()
                            .map(|value| {
                                value.to_usize().ok_or_else(|| ExecutionError::Expression {
                                    node: node.id,
                                    message: "grid reindex result is not a nonnegative index"
                                        .into(),
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        if mapped
                            .iter()
                            .enumerate()
                            .any(|(axis, value)| *value >= input_shape[axis])
                        {
                            return Err(ExecutionError::SelectIndexOutOfRange {
                                node: node.id,
                                index: BigInt::from(0),
                                count: 0,
                            });
                        }
                        self.family_member(
                            values,
                            *wire,
                            row_major_offset(&mapped, &input_shape),
                            node.id,
                        )?
                    }
                };
                Ok((name.clone(), value))
            })
            .collect()
    }

    fn child_env(
        &self,
        parent: &ParamEnv,
        bindings: &[(String, mxx_ir_core::IntExpr)],
        loop_index: Option<(u32, usize)>,
        node: NodeId,
    ) -> Result<ParamEnv, ExecutionError> {
        let mut env = parent.clone();
        if let Some((slot, index)) = loop_index {
            env.loop_indices.insert(slot, BigInt::from(index));
        }
        let expression_env = env.clone();
        for (name, expression) in bindings {
            let value = expression
                .evaluate(&expression_env)
                .map_err(|error| self.expression_error(node, error))?;
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

fn runtime_inputs_digest<B: Backend>(
    backend: &B,
    inputs: &BTreeMap<String, RuntimeValue<B>>,
) -> Result<[u8; 32], ExecutionError> {
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-runtime-session-inputs-v1");
    for (name, value) in inputs {
        hash_sized(&mut hasher, name.as_bytes());
        hash_runtime_value(backend, value, &mut hasher)?;
    }
    Ok(hasher.finalize().into())
}

fn hash_runtime_value<B: Backend>(
    backend: &B,
    value: &RuntimeValue<B>,
    hasher: &mut Sha256,
) -> Result<(), ExecutionError> {
    match value {
        RuntimeValue::Int(value) => {
            hasher.update([0]);
            hash_sized(hasher, value.to_string().as_bytes());
        }
        RuntimeValue::Real(value) => {
            hasher.update([1]);
            hasher.update(value.to_bits().to_le_bytes());
        }
        RuntimeValue::Bool(value) => {
            hasher.update([2, u8::from(*value)]);
        }
        RuntimeValue::Bytes(value) => {
            hasher.update([3]);
            hash_sized(hasher, value);
        }
        RuntimeValue::TypedBlob(value) => {
            hasher.update([4]);
            hash_sized(hasher, value);
        }
        RuntimeValue::Matrix(value) => {
            hasher.update([5]);
            hash_sized(hasher, &backend.matrix_to_bytes(value));
        }
        RuntimeValue::Trapdoor {
            secret,
            public,
            matrix_type,
            sigma,
            gadget_base,
            digit_count,
            gadget_small,
        } => {
            hasher.update([6]);
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(matrix_type)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
            hasher.update(sigma.to_bits().to_le_bytes());
            hash_sized(hasher, gadget_base.to_string().as_bytes());
            hasher.update(digit_count.to_le_bytes());
            hasher.update([gadget_small.map_or(0, |small| if small { 2 } else { 1 })]);
            hash_sized(hasher, &backend.matrix_to_bytes(public));
            match secret {
                Some(secret) => {
                    hasher.update([1]);
                    hash_sized(hasher, &backend.trapdoor_to_bytes(secret));
                }
                None => hasher.update([0]),
            }
        }
        RuntimeValue::LazyArtifact { production, name, index, descriptor } => {
            hasher.update([7]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hasher.update(index.unwrap_or(usize::MAX).to_le_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
        }
        RuntimeValue::LazyArtifactFamily { production, name, descriptor } => {
            hasher.update([8]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
        }
        RuntimeValue::StagedArtifact { production, name, index, descriptor } => {
            hasher.update([9]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hasher.update(index.to_le_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
        }
        RuntimeValue::StagedArtifactFamily { production, name, descriptor } => {
            hasher.update([10]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
        }
        RuntimeValue::Family(values) => {
            hasher.update([11]);
            hasher.update(values.len().to_le_bytes());
            for value in values {
                hash_runtime_value(backend, value, hasher)?;
            }
        }
    }
    Ok(())
}

fn hash_sized(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(bytes.len().to_le_bytes());
    hasher.update(bytes);
}

fn collect_staged_families<B: Backend>(
    value: &RuntimeValue<B>,
    families: &mut BTreeMap<(ProductionId, String), usize>,
) {
    match value {
        RuntimeValue::StagedArtifactFamily { production, name, descriptor } => {
            if let Some(count) = descriptor.family_shape.as_deref().and_then(shape_product) {
                families.insert((production.clone(), name.clone()), count);
            }
        }
        RuntimeValue::Family(values) => {
            for value in values {
                collect_staged_families(value, families);
            }
        }
        _ => {}
    }
}

fn shape_product(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |product, extent| product.checked_mul(*extent))
}

fn row_major_coordinates(mut offset: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = offset % shape[axis];
        offset /= shape[axis];
    }
    coordinates
}

fn row_major_offset(coordinates: &[usize], shape: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(shape)
        .fold(0, |offset, (coordinate, extent)| offset * extent + coordinate)
}

fn eval_index_expr(
    expr: &IndexExpr,
    coordinates: &[usize],
    env: &ParamEnv,
) -> Result<BigInt, String> {
    match expr {
        IndexExpr::Axis(axis) => coordinates
            .get(*axis)
            .copied()
            .map(BigInt::from)
            .ok_or_else(|| "index axis is out of range".into()),
        IndexExpr::Parameter(name) => {
            env.integers.get(name).cloned().ok_or_else(|| format!("unbound index parameter {name}"))
        }
        IndexExpr::LoopIndex(slot) => env
            .loop_indices
            .get(slot)
            .cloned()
            .ok_or_else(|| format!("unbound index loop slot {slot}")),
        IndexExpr::Constant(value) => Ok(value.clone()),
        IndexExpr::Add(left, right) => {
            Ok(eval_index_expr(left, coordinates, env)? + eval_index_expr(right, coordinates, env)?)
        }
        IndexExpr::Subtract(left, right) => {
            Ok(eval_index_expr(left, coordinates, env)? - eval_index_expr(right, coordinates, env)?)
        }
        IndexExpr::Multiply(left, right) => {
            Ok(eval_index_expr(left, coordinates, env)? * eval_index_expr(right, coordinates, env)?)
        }
        IndexExpr::Divide(left, right) => {
            let divisor = eval_index_expr(right, coordinates, env)?;
            if divisor.is_zero() {
                Err("index division by zero".into())
            } else {
                Ok(eval_index_expr(left, coordinates, env)? / divisor)
            }
        }
        IndexExpr::Remainder(left, right) => {
            let divisor = eval_index_expr(right, coordinates, env)?;
            if divisor.is_zero() {
                Err("index remainder by zero".into())
            } else {
                Ok(eval_index_expr(left, coordinates, env)? % divisor)
            }
        }
        IndexExpr::Equal(left, right) => Ok(BigInt::from(
            eval_index_expr(left, coordinates, env)? == eval_index_expr(right, coordinates, env)?,
        )),
        IndexExpr::Less(left, right) => Ok(BigInt::from(
            eval_index_expr(left, coordinates, env)? < eval_index_expr(right, coordinates, env)?,
        )),
        IndexExpr::LessEqual(left, right) => Ok(BigInt::from(
            eval_index_expr(left, coordinates, env)? <= eval_index_expr(right, coordinates, env)?,
        )),
        IndexExpr::Log2Ceil(value) => {
            let value = eval_index_expr(value, coordinates, env)?
                .to_biguint()
                .ok_or_else(|| "log2ceil argument must be positive".to_owned())?;
            if value.is_zero() {
                return Err("log2ceil argument must be positive".into());
            }
            let floor = value.bits() - 1;
            Ok(BigInt::from(if value == (num_bigint::BigUint::one() << floor as usize) {
                floor
            } else {
                floor + 1
            }))
        }
        IndexExpr::Select { selector, branches } => {
            let index = eval_index_expr(selector, coordinates, env)?
                .to_usize()
                .ok_or_else(|| "index selector is not a nonnegative integer".to_owned())?;
            branches
                .get(index)
                .ok_or_else(|| "index selector is out of range".to_owned())
                .and_then(|branch| eval_index_expr(branch, coordinates, env))
        }
    }
}

fn materialize_runtime_value<B: Backend, S: ArtifactStore>(
    value: RuntimeValue<B>,
    backend: &B,
    store: &mut S,
) -> Result<RuntimeValue<B>, ExecutionError> {
    match value {
        RuntimeValue::LazyArtifact { production, name, index, descriptor } => {
            let artifact_type = descriptor.artifact_type.clone();
            let payload = store
                .load(&ArtifactKey { production, name, index }, &descriptor)
                .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
            decode_artifact(backend, artifact_type, payload)
        }
        RuntimeValue::StagedArtifact { production, name, index, descriptor } => {
            let artifact_type = descriptor.artifact_type.clone();
            let payload = store
                .load_staged(&ArtifactKey { production, name, index: Some(index) }, &descriptor)
                .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
            decode_artifact(backend, artifact_type, payload)
        }
        RuntimeValue::LazyArtifactFamily { production, name, descriptor } => {
            materialize_artifact_family(production, name, descriptor, false, backend, store)
        }
        RuntimeValue::StagedArtifactFamily { production, name, descriptor } => {
            materialize_artifact_family(production, name, descriptor, true, backend, store)
        }
        RuntimeValue::Family(values) => values
            .into_iter()
            .map(|value| materialize_runtime_value(value, backend, store))
            .collect::<Result<Vec<_>, _>>()
            .map(RuntimeValue::Family),
        value => Ok(value),
    }
}

fn materialize_artifact_family<B: Backend, S: ArtifactStore>(
    production: ProductionId,
    name: String,
    descriptor: ManifestArtifact,
    staged: bool,
    backend: &B,
    store: &mut S,
) -> Result<RuntimeValue<B>, ExecutionError> {
    let count =
        descriptor.family_shape.as_deref().and_then(shape_product).ok_or_else(|| {
            ExecutionError::Manifest("artifact family has no cardinality".to_owned())
        })?;
    let artifact_type = descriptor.artifact_type.clone();
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let key =
            ArtifactKey { production: production.clone(), name: name.clone(), index: Some(index) };
        let payload = if staged {
            store.load_staged(&key, &descriptor)
        } else {
            store.load(&key, &descriptor)
        }
        .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
        values.push(decode_artifact(backend, artifact_type.clone(), payload)?);
    }
    Ok(RuntimeValue::Family(values))
}

fn decode_artifact<B: Backend>(
    backend: &B,
    artifact_type: ArtifactType,
    payload: ArtifactPayload,
) -> Result<RuntimeValue<B>, ExecutionError> {
    match (artifact_type, payload) {
        (ArtifactType::Matrix(matrix_type), ArtifactPayload::Matrix(bytes)) => {
            let matrix = backend
                .matrix_from_bytes(&matrix_type, &bytes)
                .map_err(|error| ExecutionError::Backend(error.to_string()))?;
            Ok(RuntimeValue::matrix(matrix))
        }
        (ArtifactType::Bytes { length }, ArtifactPayload::Bytes(bytes))
            if bytes.len() == length =>
        {
            Ok(RuntimeValue::Bytes(bytes))
        }
        (ArtifactType::TypedBlob { .. }, ArtifactPayload::TypedBlob(bytes)) => {
            Ok(RuntimeValue::TypedBlob(bytes))
        }
        (
            ArtifactType::Trapdoor { matrix, sigma, gadget_base, digit_count, .. },
            ArtifactPayload::Trapdoor { public_bytes, secret_bytes },
        ) => {
            let public = backend
                .matrix_from_bytes(&matrix, &public_bytes)
                .map_err(|error| ExecutionError::Backend(error.to_string()))?;
            let secret = backend
                .trapdoor_from_bytes(&matrix, &secret_bytes)
                .map_err(|error| ExecutionError::Backend(error.to_string()))?;
            let sigma = sigma
                .evaluate_f64(&ParamEnv::default())
                .map_err(|error| ExecutionError::Manifest(error.to_string()))?;
            Ok(RuntimeValue::Trapdoor {
                secret: Some(Arc::new(secret)),
                public: Arc::new(public),
                matrix_type: matrix,
                sigma,
                gadget_base,
                digit_count,
                gadget_small: None,
            })
        }
        _ => Err(ExecutionError::Artifact(
            "stored payload kind does not match artifact descriptor".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::MemoryArtifactStore,
        backend::poly::{CpuDcrtBackend, cpu_backend},
        transcript::TranscriptRecorder,
    };
    use mxx_dsl::{
        DslContext, Family, HashTag, Int, MatType, Parallel, Ring, Sequential, Subgraph,
        parallel_zip,
    };
    use mxx_ir_core::{
        Graph, GraphOutput, IntExpr, NodeHandle, RealExpr, ValueHandle, WireType,
        artifact::ArtifactConfidentiality,
        node::{IntBinaryOp, IntCompareOp, NodeKind, RealBinaryOp},
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use num_bigint::{BigInt, Sign};
    use num_traits::ToPrimitive;
    use rand::Rng;

    #[test]
    fn preimage_progress_requires_the_configured_exact_total() {
        let mut progress = PreimageProgress {
            config: PreimageProgressConfig {
                total: 3,
                report_interval: NonZeroUsize::new(2).expect("nonzero interval"),
            },
            completed: 0,
            last_reported: 0,
            started: Instant::now(),
        };
        progress.record(2);
        assert!(matches!(
            progress.finish(),
            Err(ExecutionError::PreimageProgressMismatch { expected: 3, actual: 2 })
        ));
        progress.record(1);
        assert!(progress.finish().is_ok());
    }

    #[test]
    fn canonical_polynomial_coefficient_bits_roundtrip_on_cpu() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus = parameters.modulus();
        let coefficient_bits = parameters.modulus_bits();
        let ring_dimension = parameters.ring_dimension() as usize;
        let upper = modulus.to_u64().expect("test modulus fits u64");
        let mut rng = rand::rng();
        let mut coefficients = (0..ring_dimension)
            .map(|_| num_bigint::BigUint::from(rng.random_range(0..upper)))
            .collect::<Vec<_>>();
        coefficients[0] = num_bigint::BigUint::from(0u8);
        coefficients[1] = modulus.as_ref() - num_bigint::BigUint::from(1u8);
        let polynomial = DCRTPoly::from_biguints(&parameters, &coefficients);
        let input_matrix = DCRTPolyMatrix::from_poly_vec_row(&parameters, vec![polynomial]);

        let modulus = BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone());
        let ring = Ring::new(modulus, ring_dimension);
        let input = ring.input("input", (1, 1));
        let bits = input
            .canonical_coefficient_bits(ring_dimension, coefficient_bits)
            .expect("coefficient bits");
        let reconstructed = ring.pack_polynomial_coefficients(bits, coefficient_bits);
        let graph = DslContext::new("canonical-polynomial-coefficient-bits")
            .output("reconstructed", reconstructed)
            .expect("output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");

        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::from([("input".to_owned(), RuntimeValue::matrix(input_matrix.clone()))]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execution");
        assert_eq!(matrix_output(&result, "reconstructed"), &input_matrix);
    }

    #[test]
    fn integer_lift_writes_only_the_constant_polynomial_coefficient() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let context = DslContext::new("integer-lift-constant-polynomial");
        let coefficient = context.int_family_input("coefficient", 1).get_static(0);
        let lifted = coefficient.lift_to_constant_polynomial(ring.matrix_type((1, 1)));
        let expected = ring.polynomial([IntExpr::constant(-3)]);
        let graph = context
            .output("lifted", lifted)
            .expect("lifted output")
            .output("expected", expected)
            .expect("expected output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::from([(
                "coefficient".to_owned(),
                RuntimeValue::Family(vec![RuntimeValue::Int(BigInt::from(-3))]),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execution");
        assert_eq!(matrix_output(&result, "lifted"), matrix_output(&result, "expected"));
    }

    #[test]
    fn range_loop_executes_in_bounded_waves_with_concrete_loop_indices() {
        let parameters = DCRTPolyParams::default();
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let family = Parallel::range(3)
            .map(|index| ring.polynomial([index.expression()]))
            .expect("range loop");
        let built = DslContext::new("runtime-range")
            .family_output("values", family)
            .expect("output")
            .build()
            .expect("build");
        let validated = built.validate(&ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(2).expect("nonzero"),
                ..ExecutionConfig::default()
            },
        )
        .expect("execution");
        if let RuntimeValue::StagedArtifactFamily { descriptor, .. } = &result.outputs["values"] {
            assert_eq!(descriptor.family_shape, Some(vec![3]));
        }
        let RuntimeValue::Family(values) = result
            .materialize_output("values", &backend, &mut store)
            .expect("materialize range output")
        else {
            panic!("materialized range output is not a family");
        };
        assert_eq!(values.len(), 3);
        result.cleanup_staged(&mut store).expect("staged cleanup");
    }

    #[test]
    fn dynamic_integer_hash_tags_are_deterministic_and_row_distinct() {
        let parameters = DCRTPolyParams::default();
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let key = ring.bytes_input("key", 32);
        let rows = Family::pack(
            [2usize, 0, 1]
                .into_iter()
                .map(|row| Int::constant(row).add(Int::constant(0)))
                .collect(),
        )
        .expect("row family");
        let dummy = Family::pack(vec![Int::constant(0); 3]).expect("dummy family");
        let samples = parallel_zip((rows, dummy), move |_, (row, _)| {
            let mut tag = HashTag::from(b"mxx-bgg/lwe-lookup-low/v2:test:row:".as_slice());
            tag.push(row);
            ring.hash_matrix(key.clone(), tag, (1, 1))
        })
        .expect("dynamic hash family");
        let validated = DslContext::new("runtime-dynamic-hash-tags")
            .family_output("samples", samples)
            .expect("sample output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");

        let execute_samples = |backend: &mut CpuDcrtBackend, store: &mut MemoryArtifactStore| {
            let mut result = execute(
                &validated,
                backend,
                BTreeMap::from([("key".to_owned(), RuntimeValue::Bytes(vec![0x5a; 32]))]),
                store,
                SamplingMode::Fresh,
            )
            .expect("dynamic hash execution");
            let RuntimeValue::Family(values) =
                result.materialize_output("samples", backend, store).expect("materialized hashes")
            else {
                panic!("dynamic hashes must materialize as a family")
            };
            values
                .iter()
                .map(|value| {
                    let RuntimeValue::Matrix(matrix) = value else {
                        panic!("dynamic hash member must be a matrix")
                    };
                    matrix.as_ref().clone()
                })
                .collect::<Vec<_>>()
        };

        let mut first_backend = cpu_backend([parameters.clone()]);
        let mut first_store = MemoryArtifactStore::default();
        let first = execute_samples(&mut first_backend, &mut first_store);
        let mut second_backend = cpu_backend([parameters]);
        let mut second_store = MemoryArtifactStore::default();
        let second = execute_samples(&mut second_backend, &mut second_store);
        assert_eq!(first, second);
        assert_ne!(first[0], first[1]);
        assert_ne!(first[0], first[2]);
        assert_ne!(first[1], first[2]);
    }

    #[test]
    fn trapdoor_families_sample_preimages_and_persist_each_member() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let digit_count = parameters.modulus_digits();
        let gadget_base = BigInt::from(1u64 << parameters.base_bits());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let trapdoors = Parallel::range(2)
            .map_values(|_| ring.sample_trapdoor(1, 5, gadget_base.clone(), digit_count, 1_000_000))
            .expect("trapdoor family");
        let public = trapdoors.public_matrices();
        let swapped_public = Family::pack(vec![public.get_static(1), public.get_static(0)])
            .expect("swapped public family");
        let single_trapdoor = Parallel::range(1)
            .map_values(|_| ring.sample_trapdoor(1, 5, gadget_base.clone(), digit_count, 1_000_000))
            .expect("single trapdoor family");
        let targets = Parallel::range(2)
            .map(|index| {
                ring.polynomial([IntExpr::Add(
                    Box::new(index.expression()),
                    Box::new(IntExpr::constant(1)),
                )
                .canonicalize()])
            })
            .expect("targets");
        let expected_targets = targets.clone();
        let preimages = trapdoors
            .clone()
            .parallel_zip_mat_values(targets, |_, trapdoor, target| {
                trapdoor.sample_preimage(target, (digit_count + 2, 1))
            })
            .expect("preimages");
        let products =
            parallel_zip((trapdoors.public_matrices(), preimages), |_, (public, preimage)| {
                public.apply_preimage(preimage)
            })
            .expect("products");
        let static_target = ring.polynomial([3.into()]);
        let static_trapdoor = trapdoors.get_static(0);
        let static_public = static_trapdoor.public_matrix();
        let static_preimage =
            static_trapdoor.sample_preimage(static_target.clone(), (digit_count + 2, 1));
        let static_product = static_public.clone().apply_preimage(static_preimage);
        let indices = DslContext::new("trapdoor-family-indices").int_family_input("indices", 1);
        let dynamic_target = ring.polynomial([4.into()]);
        let dynamic_trapdoor = trapdoors.get(indices.get_static(0));
        let dynamic_public = dynamic_trapdoor.public_matrix();
        let dynamic_preimage =
            dynamic_trapdoor.sample_preimage(dynamic_target.clone(), (digit_count + 2, 1));
        let dynamic_product = dynamic_public.clone().apply_preimage(dynamic_preimage);
        let validated = DslContext::new("runtime-trapdoor-family")
            .public_family_output("public", trapdoors.public_matrices())
            .expect("public family output")
            .public_family_output("public-swapped", swapped_public)
            .expect("swapped public family output")
            .private_trapdoor_family_output("trapdoors", trapdoors)
            .expect("private trapdoor family output")
            .public_family_output("public-one", single_trapdoor.public_matrices())
            .expect("single public family output")
            .private_trapdoor_family_output("trapdoors-one", single_trapdoor)
            .expect("single private trapdoor family output")
            .output("product-0", products.get_static(0))
            .expect("first product")
            .output("product-1", products.get_static(1))
            .expect("second product")
            .output("target-0", expected_targets.get_static(0))
            .expect("first target")
            .output("target-1", expected_targets.get_static(1))
            .expect("second target")
            .output("static-product", static_product)
            .expect("static product")
            .output("static-target", static_target)
            .expect("static target")
            .output("dynamic-product", dynamic_product)
            .expect("dynamic product")
            .output("dynamic-target", dynamic_target)
            .expect("dynamic target")
            .output("static-public", static_public)
            .expect("static public")
            .output("dynamic-public", dynamic_public)
            .expect("dynamic public")
            .output("expected-public-0", public.get_static(0))
            .expect("expected first public")
            .output("expected-public-1", public.get_static(1))
            .expect("expected second public")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute_with_config(
            &validated,
            &mut backend,
            BTreeMap::from([(
                "indices".to_owned(),
                RuntimeValue::Family(vec![RuntimeValue::Int(1.into())]),
            )]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(2).expect("nonzero"),
                ..ExecutionConfig::default()
            },
        )
        .expect("execution");

        assert_eq!(result.artifact_handles["public"].len(), 2);
        assert_eq!(result.artifact_handles["public-swapped"].len(), 2);
        assert_eq!(result.artifact_handles["trapdoors"].len(), 2);
        assert_eq!(result.artifact_handles["public-one"].len(), 1);
        assert_eq!(result.artifact_handles["trapdoors-one"].len(), 1);
        assert_eq!(matrix_output(&result, "product-0"), matrix_output(&result, "target-0"));
        assert_eq!(matrix_output(&result, "product-1"), matrix_output(&result, "target-1"));
        assert_eq!(
            matrix_output(&result, "static-product"),
            matrix_output(&result, "static-target")
        );
        assert_eq!(
            matrix_output(&result, "dynamic-product"),
            matrix_output(&result, "dynamic-target")
        );
        assert_eq!(
            matrix_output(&result, "static-public"),
            matrix_output(&result, "expected-public-0")
        );
        assert_eq!(
            matrix_output(&result, "dynamic-public"),
            matrix_output(&result, "expected-public-1")
        );
        assert_ne!(
            matrix_output(&result, "expected-public-0"),
            matrix_output(&result, "expected-public-1")
        );
        assert_ne!(matrix_output(&result, "target-0"), matrix_output(&result, "target-1"));

        let production = result.production_id.expect("artifact production");
        let manifest = store.manifest(&production).expect("artifact manifest").clone();
        assert_eq!(manifest.artifacts["public"].confidentiality, ArtifactConfidentiality::Public);
        assert!(manifest.artifacts["public"].content_hash.is_some());
        assert_eq!(
            manifest.artifacts["trapdoors"].confidentiality,
            ArtifactConfidentiality::Private
        );
        assert!(manifest.artifacts["trapdoors"].content_hash.is_none());
        let imported_one = ring.trapdoor_family_artifact_input(
            production.clone(),
            "public-one",
            "trapdoors-one",
            1,
            1,
            5,
            gadget_base.clone(),
            digit_count,
            1_000_000,
        );
        let target_one = Parallel::grid(vec![IntExpr::constant(1), IntExpr::constant(1)])
            .map(|_| ring.polynomial([1.into()]))
            .expect("single import target");
        let preimage_one = imported_one
            .clone()
            .sample_preimage_branches(target_one.clone(), (digit_count + 2, 1))
            .expect("single import preimage");
        let product_one = imported_one.public_matrices().get_static(0).apply_preimage(
            preimage_one.get_static(vec![IndexExpr::constant(0), IndexExpr::constant(0)]),
        );
        let imported_one_graph = DslContext::new("runtime-single-imported-trapdoor-family")
            .output("product", product_one)
            .expect("single imported product")
            .output(
                "target",
                target_one.get_static(vec![IndexExpr::constant(0), IndexExpr::constant(0)]),
            )
            .expect("single imported target")
            .build()
            .expect("single import build")
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .expect("single import validation");
        let imported_one_result = execute(
            &imported_one_graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("single import execution");
        assert_eq!(
            matrix_output(&imported_one_result, "product"),
            matrix_output(&imported_one_result, "target")
        );
        let imported = ring.trapdoor_family_artifact_input(
            production.clone(),
            "public",
            "trapdoors",
            2,
            1,
            5,
            gadget_base.clone(),
            digit_count,
            1_000_000,
        );
        let imported_targets = Parallel::grid(vec![IntExpr::constant(2), IntExpr::constant(1)])
            .map(|indices| {
                ring.polynomial([IntExpr::Add(
                    Box::new(indices[0].expression()),
                    Box::new(IntExpr::constant(1)),
                )
                .canonicalize()])
            })
            .expect("import targets");
        let expected_imported_targets = imported_targets.clone();
        let imported_preimages = imported
            .clone()
            .sample_preimage_branches(imported_targets, (digit_count + 2, 1))
            .expect("imported preimages");
        let imported_public = imported.public_matrices();
        let imported_product_0 = imported_public.get_static(0).apply_preimage(
            imported_preimages.get_static(vec![IndexExpr::constant(0), IndexExpr::constant(0)]),
        );
        let imported_product_1 = imported_public.get_static(1).apply_preimage(
            imported_preimages.get_static(vec![IndexExpr::constant(1), IndexExpr::constant(0)]),
        );
        let imported_graph = DslContext::new("runtime-imported-trapdoor-family")
            .output("product-0", imported_product_0)
            .expect("first imported product")
            .output("product-1", imported_product_1)
            .expect("second imported product")
            .output(
                "target-0",
                expected_imported_targets
                    .get_static(vec![IndexExpr::constant(0), IndexExpr::constant(0)]),
            )
            .expect("first imported target")
            .output(
                "target-1",
                expected_imported_targets
                    .get_static(vec![IndexExpr::constant(1), IndexExpr::constant(0)]),
            )
            .expect("second imported target")
            .build()
            .expect("import build")
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .expect("import validation");
        let batch_calls_before = backend.preimage_batch_calls();
        let mut recorder = TranscriptRecorder::default();
        let imported_result = execute_with_config(
            &imported_graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Record(&mut recorder),
            ExecutionConfig {
                preimage_progress: Some(PreimageProgressConfig {
                    total: 2,
                    report_interval: NonZeroUsize::new(2).expect("nonzero"),
                }),
                ..ExecutionConfig::default()
            },
        )
        .expect("import execution");
        assert_eq!(backend.preimage_batch_calls(), batch_calls_before + 1);
        assert_eq!(recorder.iter().count(), 2);
        assert_eq!(
            matrix_output(&imported_result, "product-0"),
            matrix_output(&imported_result, "target-0")
        );
        assert_eq!(
            matrix_output(&imported_result, "product-1"),
            matrix_output(&imported_result, "target-1")
        );
        let replayer = recorder.into_replayer();
        let batch_calls_before_replay = backend.preimage_batch_calls();
        let replayed_result = execute(
            &imported_graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Replay(&replayer),
        )
        .expect("import replay");
        assert_eq!(backend.preimage_batch_calls(), batch_calls_before_replay);
        assert_eq!(
            matrix_output(&replayed_result, "product-0"),
            matrix_output(&imported_result, "product-0")
        );
        assert_eq!(
            matrix_output(&replayed_result, "product-1"),
            matrix_output(&imported_result, "product-1")
        );

        let mismatched = ring.trapdoor_family_artifact_input(
            production.clone(),
            "public-swapped",
            "trapdoors",
            2,
            1,
            5,
            gadget_base.clone(),
            digit_count,
            1_000_000,
        );
        let scalar_trapdoor = mismatched.get_static(0);
        let scalar_public = scalar_trapdoor.public_matrix();
        let scalar_preimage =
            scalar_trapdoor.sample_preimage(ring.polynomial([1.into()]), (digit_count + 2, 1));
        let scalar_mismatch_graph = DslContext::new("runtime-mismatched-scalar-trapdoor")
            .output("product", scalar_public.apply_preimage(scalar_preimage))
            .expect("mismatched scalar output")
            .build()
            .expect("mismatched scalar build")
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .expect("mismatched scalar validation");
        assert!(matches!(
            execute(
                &scalar_mismatch_graph,
                &mut backend,
                BTreeMap::new(),
                &mut store,
                SamplingMode::Fresh,
            ),
            Err(ExecutionError::PreimagePublicMismatch(_))
        ));

        let batch_targets = Parallel::range(2)
            .map(|index| ring.polynomial([index.expression()]))
            .expect("mismatched batch targets");
        let batch_preimages = mismatched
            .clone()
            .parallel_zip_mat_values(batch_targets, |_, trapdoor, target| {
                trapdoor.sample_preimage(target, (digit_count + 2, 1))
            })
            .expect("mismatched batch preimages");
        let batch_products = parallel_zip(
            (mismatched.public_matrices(), batch_preimages),
            |_, (public, preimage)| public.apply_preimage(preimage),
        )
        .expect("mismatched batch products");
        let batch_mismatch_graph = DslContext::new("runtime-mismatched-batch-trapdoor")
            .output("product", batch_products.get_static(0))
            .expect("mismatched batch output")
            .build()
            .expect("mismatched batch build")
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .expect("mismatched batch validation");
        assert!(matches!(
            execute(
                &batch_mismatch_graph,
                &mut backend,
                BTreeMap::new(),
                &mut store,
                SamplingMode::Fresh,
            ),
            Err(ExecutionError::PreimagePublicMismatch(_))
        ));
    }

    #[test]
    fn sequential_scan_carries_each_iteration_output_into_the_next_iteration() {
        let context = DslContext::new("runtime-sequential-scan");
        let increments = context.int_family_input("increments", 3);
        let total = Sequential::range(3)
            .scan(Int::constant(0), increments, |index, total, increments| {
                Ok(total.add(increments.get(index.as_int())))
            })
            .expect("sequential scan");
        let validated = context
            .int_output("total", total)
            .expect("output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let result = execute(
            &validated,
            &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4)]),
            BTreeMap::from([(
                "increments".to_owned(),
                RuntimeValue::Family(vec![
                    RuntimeValue::Int(1.into()),
                    RuntimeValue::Int(2.into()),
                    RuntimeValue::Int(3.into()),
                ]),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execution");
        assert!(
            matches!(&result.outputs["total"], RuntimeValue::Int(value) if value == &BigInt::from(6))
        );

        let untouched = Sequential::range(0)
            .scan(Int::constant(7), Int::constant(99), |_, state, _| Ok(state))
            .expect("empty sequential scan");
        let validated = DslContext::new("runtime-empty-sequential-scan")
            .int_output("value", untouched)
            .expect("output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let result = execute(
            &validated,
            &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4)]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execution");
        assert!(
            matches!(&result.outputs["value"], RuntimeValue::Int(value) if value == &BigInt::from(7))
        );

        let initial =
            Family::<Int>::pack(vec![Int::constant(0), Int::constant(0)]).expect("initial family");
        let state = Sequential::range(3)
            .scan(initial, Int::constant(0), |layer, state, _| {
                state.parallel_map(|_, value| value.add(layer.as_int()))
            })
            .expect("nested sequential and parallel loop");
        let validated = DslContext::new("runtime-nested-sequential-parallel")
            .int_family_output("state", state)
            .expect("output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let result = execute(
            &validated,
            &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4)]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execution");
        let RuntimeValue::Family(state) = &result.outputs["state"] else {
            panic!("state output is not a family")
        };
        assert!(
            state.iter().all(
                |value| matches!(value, RuntimeValue::Int(value) if value == &BigInt::from(3))
            )
        );
    }

    #[test]
    fn nested_parallel_segment_pack_executes_little_endian_bits() {
        let context = DslContext::new("runtime-segmented-bit-pack");
        let bits = context.int_family_input("bits", 6);
        let packed = bits.parallel_pack_little_endian_bits(2, 3).expect("segmented bit packing");
        let validated = context
            .int_family_output("packed", packed)
            .expect("output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let result = execute(
            &validated,
            &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4)]),
            BTreeMap::from([(
                "bits".to_owned(),
                RuntimeValue::Family(
                    [1, 0, 1, 0, 1, 1]
                        .into_iter()
                        .map(|bit| RuntimeValue::Int(BigInt::from(bit)))
                        .collect(),
                ),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execution");
        let RuntimeValue::Family(packed) = &result.outputs["packed"] else {
            panic!("packed output is not a family")
        };
        assert!(matches!(&packed[0], RuntimeValue::Int(value) if value == &BigInt::from(5)));
        assert!(matches!(&packed[1], RuntimeValue::Int(value) if value == &BigInt::from(6)));
    }

    #[test]
    fn parallel_zip_many_executes_matrix_batches_in_bounded_waves() {
        let parameters = DCRTPolyParams::default();
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let families = (0..4)
            .map(|_| {
                Family::pack(vec![ring.identity(1), ring.zero((1, 1))]).expect("matrix family")
            })
            .collect::<Vec<_>>();
        let sums = Family::parallel_zip_many_values(families, |_, inputs| {
            inputs.into_iter().reduce(|left, right| left + right).expect("non-empty batch")
        })
        .expect("parallel zip many");
        let first_sum = sums.get_static(0);
        let second_sum = sums.get_static(1);
        let built = DslContext::new("runtime-parallel-zip-many")
            .output("first-sum", first_sum)
            .expect("first output")
            .output("second-sum", second_sum)
            .expect("second output")
            .build()
            .expect("build");
        let validated = built.validate(&ParamEnv::default()).expect("validation");
        let result = execute_with_config(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
            ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(2).expect("nonzero"),
                ..ExecutionConfig::default()
            },
        )
        .expect("execution");
        let four = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![DCRTPoly::from_usize_to_constant(&parameters, 4)],
        );
        let RuntimeValue::Matrix(first) = &result.outputs["first-sum"] else {
            panic!("first parallel output must be a matrix")
        };
        let RuntimeValue::Matrix(second) = &result.outputs["second-sum"] else {
            panic!("second parallel output must be a matrix")
        };
        assert_eq!(first.as_ref(), &four);
        assert_eq!(second.as_ref(), &DCRTPolyMatrix::zero(&parameters, 1, 1));
    }

    #[test]
    fn empty_range_loop_elaborates_and_executes_without_phantom_members() {
        let parameters = DCRTPolyParams::default();
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let family = Parallel::range(0)
            .map(|index| ring.polynomial([index.expression()]))
            .expect("empty range loop");
        let built = DslContext::new("runtime-empty-range")
            .family_output("values", family)
            .expect("output")
            .build()
            .expect("build");
        let validated = built.validate(&ParamEnv::default()).expect("validation");
        let result = execute(
            &validated,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execution");
        match &result.outputs["values"] {
            RuntimeValue::Family(values) => assert!(values.is_empty()),
            RuntimeValue::StagedArtifactFamily { descriptor, .. } |
            RuntimeValue::LazyArtifactFamily { descriptor, .. } => {
                assert_eq!(descriptor.family_shape, Some(vec![0]));
            }
            RuntimeValue::Int(_) => panic!("empty range output became an integer"),
            RuntimeValue::Real(_) => panic!("empty range output became a real"),
            RuntimeValue::Bool(_) => panic!("empty range output became a bool"),
            RuntimeValue::Bytes(_) => panic!("empty range output became bytes"),
            RuntimeValue::TypedBlob(_) => panic!("empty range output became a blob"),
            RuntimeValue::Matrix(_) => panic!("empty range output became a matrix"),
            RuntimeValue::Trapdoor { .. } => panic!("empty range output became a trapdoor"),
            RuntimeValue::LazyArtifact { .. } | RuntimeValue::StagedArtifact { .. } => {
                panic!("empty range output became a scalar artifact")
            }
        }
    }

    #[test]
    fn child_arguments_follow_declared_input_order() {
        let parameters = DCRTPolyParams::default();
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let ty = MatType(ring.matrix_type((1, 1)));
        let reverse = Subgraph::define("reverse", (ty.clone(), ty), |(left, right)| (right, left))
            .expect("subgraph");
        let one = ring.polynomial([1.into()]);
        let two = ring.polynomial([2.into()]);
        let (actual_two, actual_one) =
            reverse.call((one.clone(), two.clone())).expect("subgraph call");
        let left = Family::pack(vec![one.clone(), two.clone()]).expect("left family");
        let right = Family::pack(vec![two.clone(), one.clone()]).expect("right family");
        let zipped = left.parallel_zip(right.clone(), |_, _left, right| right).expect("zip");
        let actual_family_zero = zipped.get_static(0);
        let actual_family_one = zipped.get_static(1);
        let expected_family_zero = right.get_static(0);
        let expected_family_one = right.get_static(1);
        let built = DslContext::new("runtime-child-input-order")
            .output("actual-one", actual_one)
            .expect("output")
            .output("actual-two", actual_two)
            .expect("output")
            .output("expected-one", one)
            .expect("output")
            .output("expected-two", two)
            .expect("output")
            .output("actual-family-zero", actual_family_zero)
            .expect("output")
            .output("actual-family-one", actual_family_one)
            .expect("output")
            .output("expected-family-zero", expected_family_zero)
            .expect("output")
            .output("expected-family-one", expected_family_one)
            .expect("output")
            .build()
            .expect("build");
        let validated = built.validate(&ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        let bytes = |value: &RuntimeValue<_>| match value {
            RuntimeValue::Matrix(matrix) => backend.matrix_to_bytes(matrix),
            _ => panic!("expected matrix"),
        };
        assert_eq!(bytes(&result.outputs["actual-one"]), bytes(&result.outputs["expected-one"]));
        assert_eq!(bytes(&result.outputs["actual-two"]), bytes(&result.outputs["expected-two"]));
        assert_eq!(
            bytes(&result.outputs["actual-family-zero"]),
            bytes(&result.outputs["expected-family-zero"]),
        );
        assert_eq!(
            bytes(&result.outputs["actual-family-one"]),
            bytes(&result.outputs["expected-family-one"]),
        );
    }

    fn matrix_output<'a>(
        result: &'a ExecutionResult<crate::backend::poly::CpuDcrtBackend>,
        name: &str,
    ) -> &'a DCRTPolyMatrix {
        let RuntimeValue::Matrix(value) = &result.outputs[name] else {
            panic!("{name} is not a matrix")
        };
        value
    }

    #[test]
    fn transcript_replay_and_trace_preserve_sampled_execution_exactly() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let sample = ring.gaussian((1, 1), 3, 19);
        let built = DslContext::new("runtime-transcript-and-trace")
            .output("sample", sample.clone())
            .expect("sample output")
            .output("double", sample.clone() + sample)
            .expect("double output")
            .build()
            .expect("build");
        let validated = built.validate(&ParamEnv::default()).expect("validation");

        let mut recorder = crate::transcript::TranscriptRecorder::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let (recorded, trace) = execute_with_trace(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Record(&mut recorder),
        )
        .expect("recorded execution");
        assert_eq!(recorder.iter().count(), 1);
        assert!(trace.len() >= 2, "trace must retain the sample and dependent sum");

        let replayer = recorder.into_replayer();
        let replayed = execute(
            &validated,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Replay(&replayer),
        )
        .expect("replayed execution");
        assert_eq!(matrix_output(&recorded, "sample"), matrix_output(&replayed, "sample"));
        assert_eq!(matrix_output(&recorded, "double"), matrix_output(&replayed, "double"));
    }

    #[test]
    fn resumable_session_reuses_draws_and_rejects_changed_inputs() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let sampled = DslContext::new("runtime-resumable-sample")
            .private_output("sample", ring.gaussian((1, 1), 3, 19))
            .expect("private sample")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let nonce = [47u8; 32];
        let mut store = MemoryArtifactStore::default();
        let first = execute_in_session(
            &sampled,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut store,
            nonce,
        )
        .expect("first session execution");
        let second = execute_in_session(
            &sampled,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut store,
            nonce,
        )
        .expect("resumed session execution");
        assert_eq!(first.production_id, second.production_id);
        assert_eq!(matrix_output(&first, "sample"), matrix_output(&second, "sample"));

        let input = ring.input("input", (1, 1));
        let input_graph = DslContext::new("runtime-session-input-identity")
            .private_output("output", input)
            .expect("private output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let mut input_store = MemoryArtifactStore::default();
        let zero = DCRTPolyMatrix::zero(&parameters, 1, 1);
        execute_in_session(
            &input_graph,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([("input".to_owned(), RuntimeValue::matrix(zero))]),
            &mut input_store,
            [53u8; 32],
        )
        .expect("initial input session");
        let one = DCRTPolyMatrix::identity(&parameters, 1, None);
        assert!(matches!(
            execute_in_session(
                &input_graph,
                &mut cpu_backend([parameters.clone()]),
                BTreeMap::from([("input".to_owned(), RuntimeValue::matrix(one))]),
                &mut input_store,
                [53u8; 32],
            ),
            Err(ExecutionError::Artifact(_))
        ));
    }

    fn scalar_value(
        kind: NodeKind,
        arguments: Vec<ValueHandle>,
        output_type: WireType,
    ) -> ValueHandle {
        NodeHandle::new(kind, arguments, vec![output_type]).output(0).expect("scalar output")
    }

    #[test]
    fn scalar_nodes_follow_euclidean_and_real_arithmetic_contracts() {
        let minus_seven = scalar_value(
            NodeKind::ConstantInt(BigInt::from(-7)),
            Vec::new(),
            WireType::ConstantInt,
        );
        let three =
            scalar_value(NodeKind::ConstantInt(BigInt::from(3)), Vec::new(), WireType::ConstantInt);
        let quotient = scalar_value(
            NodeKind::IntBinary(IntBinaryOp::Divide),
            vec![minus_seven.clone(), three.clone()],
            WireType::Int,
        );
        let remainder = scalar_value(
            NodeKind::IntBinary(IntBinaryOp::Remainder),
            vec![minus_seven.clone(), three.clone()],
            WireType::Int,
        );
        let less = scalar_value(
            NodeKind::IntCompare(IntCompareOp::Less),
            vec![minus_seven, three],
            WireType::Bool,
        );
        let nine = scalar_value(
            NodeKind::ConstantReal(RealExpr::from_integer(9)),
            Vec::new(),
            WireType::ConstantReal,
        );
        let square_root = scalar_value(NodeKind::RealSqrt, vec![nine], WireType::Real);
        let two = scalar_value(
            NodeKind::ConstantReal(RealExpr::from_integer(2)),
            Vec::new(),
            WireType::ConstantReal,
        );
        let real_product = scalar_value(
            NodeKind::RealBinary(RealBinaryOp::Multiply),
            vec![square_root.clone(), two],
            WireType::Real,
        );
        let graph = Graph::freeze(
            "runtime-scalar-contracts",
            Vec::new(),
            BTreeMap::from([
                ("quotient".to_owned(), GraphOutput { value: quotient, confidentiality: None }),
                ("remainder".to_owned(), GraphOutput { value: remainder, confidentiality: None }),
                ("less".to_owned(), GraphOutput { value: less, confidentiality: None }),
                ("sqrt".to_owned(), GraphOutput { value: square_root, confidentiality: None }),
                ("product".to_owned(), GraphOutput { value: real_product, confidentiality: None }),
            ]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze")
        .0;
        let validated = mxx_ir_core::validate(&graph, &ParamEnv::default()).expect("validation");
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let result = execute(
            &validated,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execution");
        assert!(
            matches!(&result.outputs["quotient"], RuntimeValue::Int(value) if value == &BigInt::from(-3))
        );
        assert!(
            matches!(&result.outputs["remainder"], RuntimeValue::Int(value) if value == &BigInt::from(2))
        );
        assert!(matches!(&result.outputs["less"], RuntimeValue::Bool(true)));
        assert!(
            matches!(&result.outputs["sqrt"], RuntimeValue::Real(value) if (*value - 3.0).abs() < 1e-12)
        );
        assert!(
            matches!(&result.outputs["product"], RuntimeValue::Real(value) if (*value - 6.0).abs() < 1e-12)
        );
    }

    #[test]
    fn integer_division_by_zero_is_a_runtime_error() {
        let one =
            scalar_value(NodeKind::ConstantInt(BigInt::from(1)), Vec::new(), WireType::ConstantInt);
        let zero =
            scalar_value(NodeKind::ConstantInt(BigInt::from(0)), Vec::new(), WireType::ConstantInt);
        let quotient =
            scalar_value(NodeKind::IntBinary(IntBinaryOp::Divide), vec![one, zero], WireType::Int);
        let graph = Graph::freeze(
            "runtime-division-by-zero",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: quotient, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze")
        .0;
        let validated = mxx_ir_core::validate(&graph, &ParamEnv::default()).expect("validation");
        assert!(matches!(
            execute(
                &validated,
                &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4)]),
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            ),
            Err(ExecutionError::DivisionByZero(_))
        ));
    }

    #[test]
    fn dynamic_family_access_selects_the_runtime_index_and_rejects_out_of_range() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let family = Family::pack(vec![ring.polynomial([10.into()]), ring.polynomial([20.into()])])
            .expect("family");
        let selected = family.get(ring.input("index", (1, 1)).extract_coefficient(0));
        let validated = DslContext::new("runtime-dynamic-family")
            .output("selected", selected)
            .expect("output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation");
        let index_value = |index| {
            DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![vec![DCRTPoly::from_usize_to_constant(&parameters, index)]],
            )
        };
        let selected = execute(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([("index".to_owned(), RuntimeValue::matrix(index_value(1)))]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("selected execution");
        let expected = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![vec![DCRTPoly::from_usize_to_constant(&parameters, 20)]],
        );
        assert_eq!(matrix_output(&selected, "selected"), &expected);

        assert!(matches!(
            execute(
                &validated,
                &mut cpu_backend([parameters.clone()]),
                BTreeMap::from([("index".to_owned(), RuntimeValue::matrix(index_value(2)))]),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            ),
            Err(ExecutionError::SelectIndexOutOfRange { index, count: 2, .. }) if index == BigInt::from(2)
        ));
    }
}
