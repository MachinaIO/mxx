use crate::{
    artifact::{ArtifactKey, ArtifactPayload, ArtifactStore},
    backend::{Backend, PreimageRequest, RuntimeValue},
    liveness,
    session::{ArtifactHandle, SessionDescriptor, SessionStore},
    transcript::{DrawSite, RecordedValue, SamplingMode, TranscriptError},
};
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::{ArtifactConfidentiality, ArtifactType, ManifestArtifact, ProductionId},
    expr::euclidean_div_rem,
    graph::Graph,
    node::{
        IntBinaryOp, IntCompareOp, LoopInputMode, MatrixBinaryOp, Node, NodeKind, RealBinaryOp,
    },
    types::{
        ConcreteMatrixType, ConcreteWireType, InstantiationFrame, NodeId, Port, WireId, WireRef,
    },
};
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionConfig {
    /// Maximum number of sibling loop-body instances executed in one wave.
    ///
    /// This bounds each wave's intermediate working set and backend batch
    /// size. Artifact-compatible family outputs are streamed through the
    /// artifact store; scalar-only families are accumulated in memory.
    pub max_parallel_instances: NonZeroUsize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self { max_parallel_instances: NonZeroUsize::new(64).expect("64 is nonzero") }
    }
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
            let count = lease.descriptor.family_count.ok_or_else(|| {
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
    outputs: BTreeMap<String, RuntimeValue<B>>,
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
    #[error("backend placement {placement} is outside [0, {count})")]
    BackendPlacement { placement: usize, count: usize },
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
            let count = lease.descriptor.family_count.ok_or_else(|| {
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
    let descriptor =
        SessionDescriptor::new(production.clone(), validated.source.name.clone(), input_digest);
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
    let scratch_production = scratch_production_id(&production, rand::random());
    let mut executor = Executor {
        validated,
        backend,
        artifact_store,
        sampling_mode,
        trace: capture_trace.then(BTreeMap::new),
        session,
        config,
        production,
        scratch_production,
        staged_families: BTreeMap::new(),
    };
    let inputs = inputs
        .into_iter()
        .map(|(name, value)| Ok((name, executor.value_for_placement(value, 0)?)))
        .collect::<Result<_, ExecutionError>>()?;
    let mut instance = match executor.execute_instance(
        &validated.source,
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
    let (production_id, artifact_handles) = match executor.persist_outputs(&mut instance.outputs) {
        Ok(persisted) => persisted,
        Err(error) => {
            return match executor.cleanup_all_staged_families() {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(cleanup_error),
            };
        }
    };
    let staged_family_leases = executor.cleanup_unreturned_staged_families(&instance.outputs)?;
    if let Some(production) = &executor.session {
        if let Err(error) = executor.artifact_store.release_session(production) {
            return Err(ExecutionError::StagedCleanup {
                message: format!("session release failed: {error}"),
                leases: staged_family_leases,
            });
        }
    }
    let result = ExecutionResult {
        outputs: instance.outputs,
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
    scratch_production: ProductionId,
    staged_families: BTreeMap<(ProductionId, String), ManifestArtifact>,
}

impl<B, S> Executor<'_, B, S>
where
    B: Backend,
    S: SessionStore,
{
    fn execute_instance(
        &mut self,
        graph: &Graph,
        env: &ParamEnv,
        path: Vec<InstantiationFrame>,
        inputs: BTreeMap<String, RuntimeValue<B>>,
        placement: usize,
    ) -> Result<InstanceResult<B>, ExecutionError> {
        self.execute_instances_batch(
            graph,
            vec![env.clone()],
            vec![path],
            vec![inputs],
            vec![placement],
        )
        .map(|mut instances| instances.pop().expect("single execution returns one instance"))
    }

    fn execute_instances_batch(
        &mut self,
        graph: &Graph,
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
        let schedule = liveness::analyze(graph);
        let mut values = (0..envs.len())
            .map(|_| BTreeMap::<WireRef, RuntimeValue<B>>::new())
            .collect::<Vec<_>>();
        for (position, node) in graph.nodes.iter().enumerate() {
            if matches!(node.kind, NodeKind::PreimageSample { .. }) && envs.len() > 1 {
                self.execute_preimage_batch(&paths, &placements, node, &mut values)?;
            } else if matches!(node.kind, NodeKind::Select { .. }) {
                for index in 0..envs.len() {
                    self.set_placement(placements[index])?;
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
                    self.set_placement(placements[index])?;
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
        for (index, mut instance_values) in values.into_iter().enumerate() {
            self.set_placement(placements[index])?;
            let mut outputs = BTreeMap::new();
            for (name, wire) in &graph.outputs {
                outputs.insert(name.clone(), self.materialize(&mut instance_values, *wire)?);
            }
            instances.push(InstanceResult { outputs });
        }
        Ok(instances)
    }

    fn persist_outputs(
        &mut self,
        outputs: &mut BTreeMap<String, RuntimeValue<B>>,
    ) -> Result<(Option<ProductionId>, BTreeMap<String, Vec<ArtifactHandle>>), ExecutionError> {
        let production = self.production.clone();
        let mut artifacts = BTreeMap::new();
        let mut handles = BTreeMap::<String, Vec<ArtifactHandle>>::new();
        let mut staged_replacements = Vec::new();
        for (name, output_wire) in &self.validated.outputs {
            let Some(confidentiality) = self.output_confidentiality(name, *output_wire) else {
                continue;
            };
            let Some(output) = outputs.get(name) else {
                continue;
            };
            let wire = WireId { instantiation_path: Vec::new(), wire: *output_wire };
            let concrete_type = self
                .validated
                .wires
                .get(&wire)
                .ok_or_else(|| ExecutionError::MissingMetadata(wire.clone()))?;
            let (element_type, family_count) = match concrete_type {
                ConcreteWireType::IndexedFamily { element, count } => {
                    (element.as_ref(), Some(*count))
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
                let Some(count) = family_count else {
                    return Err(ExecutionError::Manifest(format!(
                        "output {name} is staged as a family but validated as a scalar"
                    )));
                };
                if descriptor.artifact_type != artifact_type ||
                    descriptor.family_count != Some(count)
                {
                    return Err(ExecutionError::Manifest(format!(
                        "output {name} staged descriptor does not match validated metadata"
                    )));
                }
                let mut family_hasher = Sha256::new();
                for index in 0..count {
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
                        family_count,
                        confidentiality,
                        content_hash: Some(family_hasher.finalize().into()),
                        layout: None,
                    },
                );
                staged_replacements.push((
                    name.clone(),
                    staged_production.clone(),
                    staged_name.clone(),
                    count,
                ));
                continue;
            }
            if let RuntimeValue::IndexedFamily(members) = output {
                if family_count != Some(members.len()) {
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
                        family_count,
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
                    family_count: None,
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

    fn output_confidentiality(&self, name: &str, wire: WireRef) -> Option<ArtifactConfidentiality> {
        self.validated.source.node(wire.node).and_then(|node| match &node.kind {
            NodeKind::Output {
                name: output_name,
                artifact_confidentiality: Some(confidentiality),
            } if output_name == name => Some(*confidentiality),
            _ => None,
        })
    }

    fn staged_family_descriptor(
        &mut self,
        path: &[InstantiationFrame],
        node: NodeId,
        port: u32,
        count: usize,
    ) -> Result<Option<(String, ManifestArtifact)>, ExecutionError> {
        let wire_id =
            WireId { instantiation_path: path.to_vec(), wire: WireRef { node, port: Port(port) } };
        let Some(ConcreteWireType::IndexedFamily { element, count: validated_count }) =
            self.validated_wire_type(&wire_id)
        else {
            return Ok(None);
        };
        if *validated_count != count {
            return Err(ExecutionError::MissingMetadata(wire_id));
        }
        let Some(artifact_type) = ArtifactType::from_wire_type(element) else {
            return Ok(None);
        };
        let encoded = mxx_ir_core::encoding::canonical_json(&wire_id)
            .map_err(|error| ExecutionError::Manifest(error.to_string()))?;
        let digest = Sha256::digest(encoded);
        let name = format!("runtime-staged-{}", hex_bytes(&digest));
        let descriptor = ManifestArtifact {
            artifact_type,
            family_count: Some(count),
            confidentiality: ArtifactConfidentiality::Private,
            content_hash: None,
            layout: Some("runtime/staged-family-v1".to_owned()),
        };
        self.staged_families
            .insert((self.scratch_production.clone(), name.clone()), descriptor.clone());
        Ok(Some((name, descriptor)))
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
            let Some(count) = descriptor.family_count else {
                return Err(self.staged_cleanup_error("staged family descriptor has no cardinality"));
            };
            for index in 0..count {
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
            let Some(count) = descriptor.family_count else {
                return Err(self.staged_cleanup_error("staged family descriptor has no cardinality"));
            };
            for index in 0..count {
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
                    let wire = WireRef { node: node.id, port: Port(0) };
                    let wire_id = WireId { instantiation_path: path.to_vec(), wire };
                    let concrete = self
                        .validated
                        .wires
                        .get(&wire_id)
                        .ok_or_else(|| ExecutionError::MissingMetadata(wire_id.clone()))?;
                    let descriptor = self
                        .validated
                        .artifact_inputs
                        .get(&wire_id)
                        .cloned()
                        .ok_or_else(|| ExecutionError::MissingMetadata(wire_id.clone()))?;
                    if let ConcreteWireType::IndexedFamily { element, count: declared_count } =
                        concrete
                    {
                        let artifact_type =
                            ArtifactType::from_wire_type(element).ok_or_else(|| {
                                ExecutionError::Manifest(
                                    "indexed artifact has unsupported element type".to_owned(),
                                )
                            })?;
                        if descriptor.artifact_type != artifact_type ||
                            descriptor.family_count != Some(*declared_count)
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
                        descriptor.family_count.is_some()
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
            NodeKind::Output { .. } => {
                for (port, argument) in node.args.iter().enumerate() {
                    let value = self.value(values, *argument)?;
                    values.insert(WireRef { node: node.id, port: Port(port as u32) }, value);
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
                let ty = self.matrix_type(path, WireRef { node: node.id, port: Port(0) })?;
                let matrix =
                    self.backend.constant_matrix(&ty, value, env).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(matrix));
            }
            NodeKind::GadgetTrapdoor { .. } => {
                let trapdoor_wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.trapdoor_type(path, trapdoor_wire)?;
                let sigma = self.trapdoor_sigma(path, trapdoor_wire)?;
                let (gadget_base, digit_count) = self.trapdoor_layout(path, trapdoor_wire)?;
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
            NodeKind::Reshape { rows, columns } => {
                let input = self.matrix(values, node.args[0])?;
                let rows = self.eval_usize(node.id, rows, env)?;
                let columns = self.eval_usize(node.id, columns, env)?;
                let output =
                    self.backend.reshape(&input, rows, columns).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::UniformSample { range, .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(path, wire)?;
                let value = self
                    .sample_matrix(path, wire, &ty, |backend| backend.sample_uniform(&ty, range))?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
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
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::HashSample {
                variant,
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
                let ty = self.matrix_type(path, wire)?;
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_hash(&ty, key, &tag, *variant)
                })?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } => {
                let matrix_wire = WireRef { node: node.id, port: Port(0) };
                let trapdoor_wire = WireRef { node: node.id, port: Port(1) };
                let ty = self.matrix_type(path, matrix_wire)?;
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
            NodeKind::PreimageSample { .. } => {
                let (secret, public, _, sigma, gadget_base, digit_count, gadget_small) =
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
                        backend.sample_preimage(
                            &ty,
                            sigma,
                            &gadget_base,
                            digit_count,
                            secret,
                            &public,
                            &target,
                        )
                    })?
                };
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let input = self.matrix(values, node.args[0])?;
                let input_type = self.matrix_type(path, node.args[0])?;
                let output_type =
                    self.matrix_type(path, WireRef { node: node.id, port: Port(0) })?;
                let base =
                    base.evaluate(env).map_err(|error| self.expression_error(node.id, error))?;
                let digit_count = match digit_count {
                    Some(count) => self.eval_usize(node.id, count, env)?,
                    None if output_type.rows.is_multiple_of(input_type.rows) => {
                        output_type.rows / input_type.rows
                    }
                    None => {
                        return Err(ExecutionError::Expression {
                            node: node.id,
                            message:
                                "gadget decomposition row count is not divisible by input rows"
                                    .to_owned(),
                        });
                    }
                };
                self.backend
                    .validate_gadget_layout(&input_type, &base, digit_count, *small)
                    .map_err(Self::backend_error)?;
                let output =
                    self.backend.gadget_decompose(&input, *small).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::ModDown { target_modulus } => {
                let input = self.matrix(values, node.args[0])?;
                let target = target_modulus
                    .evaluate(env)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let output =
                    self.backend.modulus_down(&input, &target).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::ModUp { .. } => {
                let input = self.matrix(values, node.args[0])?;
                let ty = self.matrix_type(path, WireRef { node: node.id, port: Port(0) })?;
                let output = self.backend.modulus_up(&input, &ty).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
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
            NodeKind::ConstantCoefficient { position } => {
                let input = self.matrix(values, node.args[0])?;
                let position = self.eval_usize(node.id, position, env)?;
                let coefficient = self
                    .backend
                    .extract_coefficient(&input, position)
                    .map_err(Self::backend_error)?;
                let ty = self.matrix_type(path, WireRef { node: node.id, port: Port(0) })?;
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
            NodeKind::SubgraphCall(call) => {
                let child = graph.subgraphs.get(&call.graph).ok_or_else(|| {
                    ExecutionError::MissingSubgraph { node: node.id, name: call.graph.clone() }
                })?;
                let child_env = self.child_env(env, &call.bindings, None, node.id)?;
                let child_inputs = self.child_inputs(child, node, values)?;
                let mut child_path = path.to_vec();
                child_path.push(InstantiationFrame { call: node.id, loop_index: None });
                let placement = self.backend.active_placement();
                let outputs = self
                    .execute_instance(child, &child_env, child_path, child_inputs, placement)?
                    .outputs;
                for (port, value) in outputs.into_values().enumerate() {
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::ParallelLoop(loop_node) => {
                let child = graph.subgraphs.get(&loop_node.graph).ok_or_else(|| {
                    ExecutionError::MissingSubgraph { node: node.id, name: loop_node.graph.clone() }
                })?;
                let count = self.eval_usize(node.id, &loop_node.count, env)?;
                let staged = (0..child.outputs.len())
                    .map(|port| self.staged_family_descriptor(path, node.id, port as u32, count))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut families =
                    staged
                        .iter()
                        .map(|descriptor| {
                            if descriptor.is_some() {
                                Vec::new()
                            } else {
                                Vec::with_capacity(count)
                            }
                        })
                        .collect::<Vec<_>>();
                let placement_count = self.backend.placement_count();
                if placement_count == 0 {
                    return Err(ExecutionError::BackendPlacement { placement: 0, count: 0 });
                }
                let parent_placement = self.backend.active_placement();
                let mut broadcast_inputs = (0..placement_count)
                    .map(|_| (0..node.args.len()).map(|_| None).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                for (argument, (wire, mode)) in
                    node.args.iter().zip(&loop_node.input_modes).enumerate()
                {
                    if matches!(mode, LoopInputMode::Broadcast) {
                        let placed = self.values_for_placements(self.value(values, *wire)?)?;
                        for (placement, value) in placed.into_iter().enumerate() {
                            broadcast_inputs[placement][argument] = Some(value);
                        }
                    }
                }
                self.set_placement(parent_placement)?;
                let wave_size = self.config.max_parallel_instances.get();
                for wave_start in (0..count).step_by(wave_size) {
                    let wave_end = count.min(wave_start.saturating_add(wave_size));
                    let wave_len = wave_end - wave_start;
                    let mut child_envs = Vec::with_capacity(wave_len);
                    let mut child_paths = Vec::with_capacity(wave_len);
                    let mut child_inputs = Vec::with_capacity(wave_len);
                    let mut child_placements = Vec::with_capacity(wave_len);
                    for index in wave_start..wave_end {
                        let placement = index % placement_count;
                        child_envs.push(self.child_env(
                            env,
                            &loop_node.bindings,
                            Some((&loop_node.index_variable, index)),
                            node.id,
                        )?);
                        let mut child_path = path.to_vec();
                        child_path.push(InstantiationFrame {
                            call: node.id,
                            loop_index: Some(index as u64),
                        });
                        child_paths.push(child_path);
                        child_placements.push(placement);
                        child_inputs.push(self.loop_child_inputs(
                            child,
                            node,
                            &loop_node.input_modes,
                            index,
                            placement,
                            &broadcast_inputs[placement],
                            values,
                        )?);
                    }
                    let instances = self.execute_instances_batch(
                        child,
                        child_envs,
                        child_paths,
                        child_inputs,
                        child_placements,
                    )?;
                    for (offset, instance) in instances.into_iter().enumerate() {
                        for (port, value) in instance.outputs.into_values().enumerate() {
                            if let Some((name, descriptor)) = &staged[port] {
                                let (payload, _) =
                                    self.encode_artifact(&value, &descriptor.artifact_type)?;
                                self.artifact_store
                                    .store(
                                        ArtifactKey {
                                            production: self.scratch_production.clone(),
                                            name: name.clone(),
                                            index: Some(wave_start + offset),
                                        },
                                        &descriptor.artifact_type,
                                        descriptor.confidentiality,
                                        descriptor.layout.as_deref(),
                                        payload,
                                    )
                                    .map_err(Self::artifact_error)?;
                            } else {
                                families[port].push(value);
                            }
                        }
                    }
                }
                self.set_placement(parent_placement)?;
                for (port, family) in families.into_iter().enumerate() {
                    let value = match &staged[port] {
                        Some((name, descriptor)) => RuntimeValue::StagedArtifactFamily {
                            production: self.scratch_production.clone(),
                            name: name.clone(),
                            descriptor: descriptor.clone(),
                        },
                        None => RuntimeValue::IndexedFamily(family),
                    };
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::FamilyPack { count } => {
                let count = self.eval_usize(node.id, count, env)?;
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
                self.put(values, node.id, 0, RuntimeValue::IndexedFamily(members));
            }
            NodeKind::FamilyGetStatic { index } => {
                let index = self.eval_usize(node.id, index, env)?;
                let selected = self.family_member(values, node.args[0], index, node.id)?;
                self.put(values, node.id, 0, selected);
            }
            NodeKind::FamilyGetDynamic => {
                let index = self.int(values, node.args[1])?;
                let Some(index) = index.to_usize() else {
                    let count = self.family_count(values, node.args[0])?;
                    return Err(ExecutionError::SelectIndexOutOfRange {
                        node: node.id,
                        index,
                        count,
                    });
                };
                let selected = self.family_member(values, node.args[0], index, node.id)?;
                self.put(values, node.id, 0, selected);
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
        &self,
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
        wire: WireRef,
    ) -> Result<usize, ExecutionError> {
        match values.get(&wire).ok_or(ExecutionError::MissingWire(wire))? {
            RuntimeValue::LazyArtifactFamily { descriptor, .. } |
            RuntimeValue::StagedArtifactFamily { descriptor, .. } => {
                descriptor.family_count.ok_or(ExecutionError::ValueKind(wire))
            }
            RuntimeValue::IndexedFamily(values) => Ok(values.len()),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
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
                if descriptor.family_count.is_some_and(|count| index < count) =>
            {
                Ok(RuntimeValue::LazyArtifact {
                    production: production.clone(),
                    name: name.clone(),
                    index: Some(index),
                    descriptor: descriptor.clone(),
                })
            }
            RuntimeValue::StagedArtifactFamily { production, name, descriptor }
                if descriptor.family_count.is_some_and(|count| index < count) =>
            {
                Ok(RuntimeValue::StagedArtifact {
                    production: production.clone(),
                    name: name.clone(),
                    index,
                    descriptor: descriptor.clone(),
                })
            }
            RuntimeValue::IndexedFamily(values) => {
                values.get(index).cloned().ok_or(ExecutionError::ValueKind(wire))
            }
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn execute_preimage_batch(
        &mut self,
        paths: &[Vec<InstantiationFrame>],
        placements: &[usize],
        node: &Node,
        values: &mut [BTreeMap<WireRef, RuntimeValue<B>>],
    ) -> Result<(), ExecutionError> {
        struct Pending<M, T> {
            instance: usize,
            placement: usize,
            wire: WireRef,
            path: Vec<InstantiationFrame>,
            request: PreimageRequest<M, T>,
        }

        let mut pending = Vec::new();
        for instance in 0..values.len() {
            self.set_placement(placements[instance])?;
            let (secret, public, _, sigma, gadget_base, digit_count, gadget_small) =
                self.trapdoor(&mut values[instance], node.args[0])?;
            let target = self.matrix(&mut values[instance], node.args[1])?;
            let wire = WireRef { node: node.id, port: Port(0) };
            if let Some(small) = gadget_small {
                let value =
                    self.backend.gadget_decompose(&target, small).map_err(Self::backend_error)?;
                self.put(&mut values[instance], node.id, 0, RuntimeValue::matrix(value));
                continue;
            }
            let matrix_type = self.matrix_type(&paths[instance], wire)?;
            pending.push(Pending {
                instance,
                placement: placements[instance],
                wire,
                path: paths[instance].clone(),
                request: PreimageRequest {
                    matrix_type,
                    sigma,
                    gadget_base,
                    digit_count,
                    trapdoor: secret.expect("sampled trapdoor must carry secret material"),
                    public,
                    target,
                },
            });
        }
        if pending.is_empty() {
            return Ok(());
        }

        if let Some(production) = self.session.clone() {
            let mut outputs = (0..pending.len()).map(|_| None).collect::<Vec<Option<B::Matrix>>>();
            let mut missing = Vec::new();
            for (index, request) in pending.iter().enumerate() {
                let site = DrawSite {
                    instantiation_path: request.path.clone(),
                    node: request.wire.node,
                    port: request.wire.port,
                };
                match self
                    .artifact_store
                    .transcript_entry(&production, &site)
                    .map_err(Self::artifact_error)?
                {
                    Some(RecordedValue::Matrix { matrix_type, bytes })
                        if matrix_type == request.request.matrix_type =>
                    {
                        self.set_placement(request.placement)?;
                        outputs[index] = Some(
                            self.backend
                                .matrix_from_bytes(&request.request.matrix_type, &bytes)
                                .map_err(Self::backend_error)?,
                        );
                    }
                    Some(RecordedValue::Matrix { .. } | RecordedValue::Trapdoor { .. }) => {
                        return Err(TranscriptError::KindMismatch(site).into());
                    }
                    None => missing.push(index),
                }
            }
            if !missing.is_empty() {
                let mut sampled_by_index =
                    (0..pending.len()).map(|_| None).collect::<Vec<Option<B::Matrix>>>();
                for placement in 0..self.backend.placement_count() {
                    let indices = missing
                        .iter()
                        .copied()
                        .filter(|index| pending[*index].placement == placement)
                        .collect::<Vec<_>>();
                    if indices.is_empty() {
                        continue;
                    }
                    self.set_placement(placement)?;
                    let sampled = self
                        .backend
                        .sample_preimage_batch(
                            indices.iter().map(|index| pending[*index].request.clone()).collect(),
                        )
                        .map_err(Self::backend_error)?;
                    for (index, output) in indices.into_iter().zip(sampled) {
                        sampled_by_index[index] = Some(output);
                    }
                }
                let entries = missing
                    .iter()
                    .map(|index| {
                        let output = sampled_by_index[*index]
                            .as_ref()
                            .expect("every missing preimage was sampled");
                        let request = &pending[*index];
                        (
                            DrawSite {
                                instantiation_path: request.path.clone(),
                                node: request.wire.node,
                                port: request.wire.port,
                            },
                            RecordedValue::Matrix {
                                matrix_type: request.request.matrix_type.clone(),
                                bytes: self.backend.matrix_to_bytes(output),
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                self.artifact_store
                    .record_transcript_batch(&production, &entries)
                    .map_err(Self::artifact_error)?;
                for index in missing {
                    outputs[index] = sampled_by_index[index].take();
                }
            }
            for (request, output) in pending.into_iter().zip(outputs) {
                self.put(
                    &mut values[request.instance],
                    node.id,
                    0,
                    RuntimeValue::matrix(output.expect("every session preimage draw is resolved")),
                );
            }
            return Ok(());
        }

        let replayed = match &self.sampling_mode {
            SamplingMode::Replay(replayer) => {
                let recorded = pending
                    .iter()
                    .map(|request| {
                        let site = DrawSite {
                            instantiation_path: request.path.clone(),
                            node: request.wire.node,
                            port: request.wire.port,
                        };
                        replayer.get(&site).cloned()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut outputs = Vec::with_capacity(pending.len());
                for (request, recorded) in pending.iter().zip(recorded) {
                    match recorded {
                        RecordedValue::Matrix { bytes, .. } => {
                            self.set_placement(request.placement)?;
                            outputs.push(
                                self.backend
                                    .matrix_from_bytes(&request.request.matrix_type, &bytes)
                                    .map_err(Self::backend_error)?,
                            );
                        }
                        RecordedValue::Trapdoor { .. } => {
                            return Err(TranscriptError::KindMismatch(DrawSite {
                                instantiation_path: request.path.clone(),
                                node: request.wire.node,
                                port: request.wire.port,
                            })
                            .into());
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
            let mut outputs = (0..pending.len()).map(|_| None).collect::<Vec<Option<B::Matrix>>>();
            for placement in 0..self.backend.placement_count() {
                let indices = pending
                    .iter()
                    .enumerate()
                    .filter_map(|(index, request)| {
                        (request.placement == placement).then_some(index)
                    })
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    continue;
                }
                self.set_placement(placement)?;
                let sampled = self
                    .backend
                    .sample_preimage_batch(
                        indices.iter().map(|index| pending[*index].request.clone()).collect(),
                    )
                    .map_err(Self::backend_error)?;
                for (index, output) in indices.into_iter().zip(sampled) {
                    outputs[index] = Some(output);
                }
            }
            outputs
                .into_iter()
                .map(|output| output.expect("every preimage request was assigned to a placement"))
                .collect()
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
            self.put(&mut values[request.instance], node.id, 0, RuntimeValue::matrix(output));
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
        if let Some(production) = self.session.clone() {
            if let Some(recorded) = self
                .artifact_store
                .transcript_entry(&production, &site)
                .map_err(Self::artifact_error)?
            {
                return match recorded {
                    RecordedValue::Matrix { matrix_type, bytes } if matrix_type == *ty => {
                        self.backend.matrix_from_bytes(ty, &bytes).map_err(Self::backend_error)
                    }
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
            return Ok(value);
        }
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
        match (artifact_type, payload) {
            (ArtifactType::Matrix(matrix_type), ArtifactPayload::Matrix(bytes)) => {
                let matrix = self
                    .backend
                    .matrix_from_bytes(&matrix_type, &bytes)
                    .map_err(Self::backend_error)?;
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
                ArtifactType::Trapdoor { matrix, sigma, gadget_base, digit_count },
                ArtifactPayload::Trapdoor { public_bytes, secret_bytes },
            ) => {
                let public = self
                    .backend
                    .matrix_from_bytes(&matrix, &public_bytes)
                    .map_err(Self::backend_error)?;
                let secret = self
                    .backend
                    .trapdoor_from_bytes(&matrix, &secret_bytes)
                    .map_err(Self::backend_error)?;
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
            RuntimeValue::IndexedFamily(values) => RuntimeValue::IndexedFamily(
                values
                    .into_iter()
                    .map(|value| self.value_for_placement(value, placement))
                    .collect::<Result<_, _>>()?,
            ),
            value => value,
        })
    }

    fn values_for_placements(
        &mut self,
        value: RuntimeValue<B>,
    ) -> Result<Vec<RuntimeValue<B>>, ExecutionError> {
        let placement_count = self.backend.placement_count();
        Ok(match value {
            RuntimeValue::Matrix(matrix) => self
                .backend
                .matrix_to_placements(matrix.as_ref())
                .map_err(Self::backend_error)?
                .into_iter()
                .map(|placed| {
                    placed
                        .map(RuntimeValue::matrix)
                        .unwrap_or_else(|| RuntimeValue::Matrix(matrix.clone()))
                })
                .collect(),
            RuntimeValue::Trapdoor {
                secret,
                public,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                gadget_small,
            } => {
                let original = self.backend.active_placement();
                let mut source_placement = None;
                for placement in 0..placement_count {
                    self.set_placement(placement)?;
                    if self.backend.matrix_is_on_active_placement(public.as_ref()) {
                        source_placement = Some(placement);
                        break;
                    }
                }
                self.set_placement(original)?;
                let source_placement = source_placement.ok_or_else(|| {
                    ExecutionError::Backend(
                        "trapdoor public matrix does not belong to a registered placement"
                            .to_owned(),
                    )
                })?;
                let source_public = public.clone();
                let publics = self
                    .backend
                    .matrix_to_placements(public.as_ref())
                    .map_err(Self::backend_error)?;
                let secrets = secret
                    .map(|secret| {
                        self.backend
                            .trapdoor_to_placements(&matrix_type, secret.as_ref(), source_placement)
                            .map_err(Self::backend_error)
                    })
                    .transpose()?;
                let secrets = secrets
                    .map(|secrets| secrets.into_iter().map(Some).collect())
                    .unwrap_or_else(|| (0..placement_count).map(|_| None).collect::<Vec<_>>());
                publics
                    .into_iter()
                    .zip(secrets)
                    .map(|(placed_public, secret)| RuntimeValue::Trapdoor {
                        secret: secret.map(Arc::new),
                        public: placed_public
                            .map(Arc::new)
                            .unwrap_or_else(|| source_public.clone()),
                        matrix_type: matrix_type.clone(),
                        sigma,
                        gadget_base: gadget_base.clone(),
                        digit_count,
                        gadget_small,
                    })
                    .collect()
            }
            RuntimeValue::IndexedFamily(values) => {
                let mut families = (0..placement_count)
                    .map(|_| Vec::with_capacity(values.len()))
                    .collect::<Vec<_>>();
                for value in values {
                    for (placement, value) in
                        self.values_for_placements(value)?.into_iter().enumerate()
                    {
                        families[placement].push(value);
                    }
                }
                families.into_iter().map(RuntimeValue::IndexedFamily).collect()
            }
            value @ (RuntimeValue::LazyArtifact { .. } | RuntimeValue::StagedArtifact { .. }) => {
                self.set_placement(0)?;
                let materialized = self.materialize_value(value)?;
                self.values_for_placements(materialized)?
            }
            value => vec![value; placement_count],
        })
    }

    fn matrix_type(
        &self,
        path: &[InstantiationFrame],
        wire: WireRef,
    ) -> Result<ConcreteMatrixType, ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        self.validated_wire_type(&id)
            .and_then(|wire| wire.matrix_type().cloned())
            .ok_or(ExecutionError::MissingMetadata(id))
    }

    fn validated_wire_type(&self, id: &WireId) -> Option<&ConcreteWireType> {
        self.validated.wires.get(id).or_else(|| {
            let mut representative = id.clone();
            let mut changed = false;
            for frame in &mut representative.instantiation_path {
                if frame.loop_index.is_some_and(|index| index != 0) {
                    frame.loop_index = Some(0);
                    changed = true;
                }
            }
            changed.then(|| self.validated.wires.get(&representative)).flatten()
        })
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
    ) -> Result<f64, ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        match self.validated_wire_type(&id) {
            Some(ConcreteWireType::Trapdoor { sigma, .. }) => sigma
                .evaluate_f64(&ParamEnv::default())
                .map_err(|error| self.expression_error(wire.node, error)),
            _ => Err(ExecutionError::MissingMetadata(id)),
        }
    }

    fn trapdoor_layout(
        &self,
        path: &[InstantiationFrame],
        wire: WireRef,
    ) -> Result<(BigInt, usize), ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        match self.validated_wire_type(&id) {
            Some(ConcreteWireType::Trapdoor { gadget_base, digit_count, .. }) => {
                Ok((gadget_base.clone(), *digit_count))
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

    fn loop_child_inputs(
        &mut self,
        child: &Graph,
        node: &Node,
        modes: &[LoopInputMode],
        index: usize,
        placement: usize,
        broadcast_inputs: &[Option<RuntimeValue<B>>],
        values: &BTreeMap<WireRef, RuntimeValue<B>>,
    ) -> Result<BTreeMap<String, RuntimeValue<B>>, ExecutionError> {
        let names = child.nodes.iter().filter_map(|node| match &node.kind {
            NodeKind::Input { name, .. } => Some(name),
            _ => None,
        });
        if modes.len() != node.args.len() || broadcast_inputs.len() != node.args.len() {
            return Err(ExecutionError::ValueKind(WireRef { node: node.id, port: Port(0) }));
        }
        names
            .zip(&node.args)
            .zip(modes)
            .zip(broadcast_inputs)
            .map(|(((name, wire), mode), broadcast)| {
                let value = match mode {
                    LoopInputMode::Broadcast => {
                        broadcast.clone().ok_or(ExecutionError::ValueKind(WireRef {
                            node: node.id,
                            port: Port(0),
                        }))?
                    }
                    LoopInputMode::Zip | LoopInputMode::ZipOffset { .. } => {
                        let offset = match mode {
                            LoopInputMode::Zip => 0,
                            LoopInputMode::ZipOffset { offset } => *offset,
                            LoopInputMode::Broadcast => unreachable!(),
                        };
                        let index = index.checked_add(offset).ok_or(
                            ExecutionError::SelectIndexOutOfRange {
                                node: node.id,
                                index: BigInt::from(index),
                                count: self.family_count(values, *wire)?,
                            },
                        )?;
                        self.value_for_placement(
                            self.family_member_value(values, *wire, index)?,
                            placement,
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
        loop_index: Option<(&str, usize)>,
        node: NodeId,
    ) -> Result<ParamEnv, ExecutionError> {
        let mut env = parent.clone();
        if let Some((name, index)) = loop_index {
            env.integers.insert(name.to_owned(), BigInt::from(index));
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
        RuntimeValue::IndexedFamily(values) => {
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

fn scratch_production_id(
    production: &ProductionId,
    scratch_execution_nonce: [u8; 32],
) -> ProductionId {
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-runtime/staged-family/v1");
    hasher.update(&production.spec_hash.0);
    hasher.update(production.execution_nonce);
    ProductionId {
        spec_hash: mxx_ir_core::artifact::SpecHash(hasher.finalize().into()),
        execution_nonce: scratch_execution_nonce,
    }
}

fn collect_staged_families<B: Backend>(
    value: &RuntimeValue<B>,
    families: &mut BTreeMap<(ProductionId, String), usize>,
) {
    match value {
        RuntimeValue::StagedArtifactFamily { production, name, descriptor } => {
            if let Some(count) = descriptor.family_count {
                families.insert((production.clone(), name.clone()), count);
            }
        }
        RuntimeValue::IndexedFamily(values) => {
            for value in values {
                collect_staged_families(value, families);
            }
        }
        _ => {}
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{
            ArtifactKey, ArtifactPayload, ArtifactStore, MemoryArtifactError, MemoryArtifactStore,
        },
        backend::poly::{CpuDcrtBackend, cpu_backend},
        filesystem_store::FilesystemArtifactStore,
        transcript::{SamplingMode, TranscriptRecorder},
    };
    use mxx_ir_core::{
        Graph, GraphBuilder,
        artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        },
        graph::{CompileParameter, CompileParameterKind},
        node::{ConcatAxis, ConstantMatrix, IndexRange, Node, SampleRange},
        types::{MatrixType, WireType},
        validate, validate_with_manifests,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        modulus::modulus_raise,
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use std::{collections::BTreeSet, sync::Arc};

    fn wire(node: u64, port: u32) -> WireRef {
        WireRef { node: NodeId(node), port: Port(port) }
    }

    struct FailOnceCleanupStore<'a> {
        inner: &'a mut MemoryArtifactStore,
        fail_next_remove: bool,
        fail_next_release: bool,
    }

    impl ArtifactStore for FailOnceCleanupStore<'_> {
        type Error = MemoryArtifactError;

        fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
            self.inner.load_manifest(production)
        }

        fn load(
            &mut self,
            key: &ArtifactKey,
            descriptor: &ManifestArtifact,
        ) -> Result<ArtifactPayload, Self::Error> {
            self.inner.load(key, descriptor)
        }

        fn store(
            &mut self,
            key: ArtifactKey,
            artifact_type: &ArtifactType,
            confidentiality: ArtifactConfidentiality,
            layout: Option<&str>,
            payload: ArtifactPayload,
        ) -> Result<(), Self::Error> {
            self.inner.store(key, artifact_type, confidentiality, layout, payload)
        }

        fn load_staged(
            &mut self,
            key: &ArtifactKey,
            descriptor: &ManifestArtifact,
        ) -> Result<ArtifactPayload, Self::Error> {
            self.inner.load_staged(key, descriptor)
        }

        fn remove_staged(&mut self, key: &ArtifactKey) -> Result<(), Self::Error> {
            if self.fail_next_remove {
                self.fail_next_remove = false;
                Err(MemoryArtifactError::Missing(key.clone()))
            } else {
                self.inner.remove_staged(key)
            }
        }

        fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
            self.inner.store_manifest(manifest)
        }
    }

    impl SessionStore for FailOnceCleanupStore<'_> {
        fn resolve_session_nonce(
            &mut self,
            descriptor: &crate::session::SessionAliasDescriptor,
        ) -> Result<[u8; 32], Self::Error> {
            self.inner.resolve_session_nonce(descriptor)
        }

        fn open_session(
            &mut self,
            descriptor: &SessionDescriptor,
        ) -> Result<crate::session::SessionStatus, Self::Error> {
            self.inner.open_session(descriptor)
        }

        fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
            if self.fail_next_release {
                self.fail_next_release = false;
                Err(MemoryArtifactError::SessionNotOpen(production.clone()))
            } else {
                self.inner.release_session(production)
            }
        }

        fn transcript_entry(
            &mut self,
            production: &ProductionId,
            site: &DrawSite,
        ) -> Result<Option<RecordedValue>, Self::Error> {
            self.inner.transcript_entry(production, site)
        }

        fn record_transcript_batch(
            &mut self,
            production: &ProductionId,
            entries: &[(DrawSite, RecordedValue)],
        ) -> Result<(), Self::Error> {
            self.inner.record_transcript_batch(production, entries)
        }

        fn commit_artifact(&mut self, handle: &ArtifactHandle) -> Result<(), Self::Error> {
            self.inner.commit_artifact(handle)
        }

        fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
            self.inner.finalize_session(manifest)
        }
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
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::UniformSample {
                        matrix_type: matrix_type(parameters),
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::Output {
                        name: "out".to_owned(),
                        artifact_confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                    args: vec![wire(1, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(2, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        }
    }

    fn ephemeral_rotation_family_graph(
        parameters: &DCRTPolyParams,
        name: &str,
        count: usize,
    ) -> Graph {
        let body = Graph {
            name: format!("{name}-body"),
            parameters: vec![CompileParameter {
                name: "i".to_owned(),
                kind: CompileParameterKind::Integer,
            }],
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::ConstantMatrix {
                    matrix_type: matrix_type(parameters),
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
            name: name.to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(10),
                kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                    graph: body.name.clone(),
                    count: mxx_ir_core::IntExpr::constant(count),
                    minimum_count: 0,
                    index_variable: "i".to_owned(),
                    bindings: Vec::new(),
                    input_modes: Vec::new(),
                }),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("family".to_owned(), wire(10, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        graph
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
        let mut public_type = matrix_type(&parameters);
        public_type.columns = mxx_ir_core::IntExpr::constant(2 + parameters.modulus_digits());
        let graph = Graph {
            name: "trapdoor-transcript".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::TrapdoorSample {
                    matrix_type: public_type,
                    sigma: mxx_ir_core::RealExpr::Rational(
                        mxx_ir_core::expr::Rational::new(BigInt::from(4), BigInt::from(1))
                            .expect("rational"),
                    ),
                    gadget_base: mxx_ir_core::IntExpr::constant(
                        BigInt::one() << parameters.base_bits(),
                    ),
                    digit_count: mxx_ir_core::IntExpr::constant(parameters.modulus_digits()),
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
                        gadget_base: mxx_ir_core::IntExpr::constant(
                            BigInt::one() << parameters.base_bits(),
                        ),
                        digit_count: mxx_ir_core::IntExpr::constant(digits),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::TrapdoorSample {
                        matrix_type: public_type.clone(),
                        sigma: sigma.clone(),
                        gadget_base: mxx_ir_core::IntExpr::constant(
                            BigInt::one() << parameters.base_bits(),
                        ),
                        digit_count: mxx_ir_core::IntExpr::constant(digits),
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
                    kind: NodeKind::FamilyPack { count: mxx_ir_core::IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::Output {
                        name: "family".to_owned(),
                        artifact_confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                    args: vec![wire(3, 0)],
                },
            ],
            outputs: BTreeMap::from([("family".to_owned(), wire(4, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters]);
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        let RuntimeValue::IndexedFamily(members) = &result.outputs["family"] else {
            panic!("indexed family output");
        };
        assert_eq!(members.len(), 2);
        let production = result.production_id.expect("family manifest");
        let manifest = store.manifest(&production).expect("stored manifest");
        assert_eq!(manifest.artifacts["family"].family_count.expect("family count"), 2);
        assert!(manifest.artifacts["family"].content_hash.is_some());
        let descriptor = manifest.artifacts["family"].clone();
        for index in 0..2 {
            store
                .load(
                    &ArtifactKey {
                        production: production.clone(),
                        name: "family".to_owned(),
                        index: Some(index),
                    },
                    &descriptor,
                )
                .expect("stored family member");
        }
        assert_eq!(
            store.family_hash_verification_count(),
            1,
            "aggregate family content must be verified once rather than once per member"
        );
    }

    #[test]
    fn dynamic_family_get_loads_only_the_selected_artifact_member() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let symbolic_type = matrix_type(&parameters);
        let modulus: Arc<num_bigint::BigUint> = parameters.modulus().into();
        let concrete_type = ConcreteMatrixType {
            modulus: BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()),
            ring_dimension: 8,
            rows: 1,
            columns: 1,
        };
        let production = ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(
                "family".to_owned(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Matrix(concrete_type.clone()),
                    family_count: Some(2),
                    confidentiality: ArtifactConfidentiality::Public,
                    content_hash: None,
                    layout: None,
                },
            )]),
        };
        let mut builder = GraphBuilder::new("lazy-family-get", Vec::new());
        let family = builder.artifact_family_input(
            "family",
            symbolic_type,
            production.clone(),
            "family",
            mxx_ir_core::IntExpr::constant(2),
            ArtifactConfidentiality::Public,
        );
        let index = builder.integer_input("index");
        let selected = builder.family_get_dynamic(&family, index);
        builder.output("out", &selected, ArtifactConfidentiality::Public);
        let validated = validate_with_manifests(
            &builder.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([(production.clone(), manifest.clone())]),
        )
        .expect("family graph validation");

        let first = DCRTPolyMatrix::zero(&parameters, 1, 1);
        let second = DCRTPolyMatrix::identity(&parameters, 1, None);
        let first_key = ArtifactKey {
            production: production.clone(),
            name: "family".to_owned(),
            index: Some(0),
        };
        let second_key = ArtifactKey { production, name: "family".to_owned(), index: Some(1) };
        let mut store = MemoryArtifactStore::default();
        store.store_manifest(manifest).expect("family manifest");
        store
            .insert(
                first_key.clone(),
                ArtifactType::Matrix(concrete_type.clone()),
                ArtifactConfidentiality::Public,
                ArtifactPayload::Matrix(first.to_compact_bytes()),
            )
            .expect("first artifact");
        store
            .insert(
                second_key.clone(),
                ArtifactType::Matrix(concrete_type),
                ArtifactConfidentiality::Public,
                ArtifactPayload::Matrix(second.to_compact_bytes()),
            )
            .expect("second artifact");
        let mut backend = cpu_backend([parameters]);
        let output = execute(
            &validated,
            &mut backend,
            BTreeMap::from([("index".to_owned(), RuntimeValue::Int(BigInt::from(1)))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("execution");
        let RuntimeValue::Matrix(actual) = &output.outputs["out"] else {
            panic!("matrix output");
        };
        assert_eq!(actual.as_ref(), &second);
        assert_eq!(store.load_count(&first_key), 0);
        assert_eq!(store.load_count(&second_key), 1);
    }

    #[test]
    fn typed_bytes_and_private_blob_artifacts_round_trip() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let schema_hash = [19; 32];
        let mut producer = GraphBuilder::new("typed-artifact-producer", Vec::new());
        let bytes = producer.bytes_input("bytes", 4);
        let blob = producer.typed_blob_input("blob", "test/private-state", schema_hash);
        producer.output_wire("bytes", bytes, ArtifactConfidentiality::Public);
        producer.output_wire("blob", blob, ArtifactConfidentiality::Private);
        let producer =
            validate(&producer.finish(), &ParamEnv::default()).expect("producer validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let produced = execute(
            &producer,
            &mut backend,
            BTreeMap::from([
                ("bytes".to_owned(), RuntimeValue::Bytes(vec![1, 2, 3, 4])),
                ("blob".to_owned(), RuntimeValue::TypedBlob(vec![8, 9, 10])),
            ]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("producer execution");
        let production = produced.production_id.expect("typed artifact manifest");
        let manifest = store.manifest(&production).expect("manifest").clone();
        assert_eq!(manifest.artifacts["bytes"].artifact_type, ArtifactType::Bytes { length: 4 });
        assert_eq!(
            manifest.artifacts["blob"].artifact_type,
            ArtifactType::TypedBlob { type_name: "test/private-state".to_owned(), schema_hash }
        );
        assert!(manifest.artifacts["bytes"].content_hash.is_some());
        assert_eq!(manifest.artifacts["blob"].confidentiality, ArtifactConfidentiality::Private);
        assert!(manifest.artifacts["blob"].content_hash.is_none());

        let mut consumer = GraphBuilder::new("typed-artifact-consumer", Vec::new());
        let bytes = consumer.artifact_bytes_input(
            "bytes",
            mxx_ir_core::IntExpr::constant(4),
            production.clone(),
            "bytes",
            ArtifactConfidentiality::Public,
        );
        let blob = consumer.artifact_typed_blob_input(
            "blob",
            "test/private-state",
            schema_hash,
            production.clone(),
            "blob",
            ArtifactConfidentiality::Private,
        );
        consumer.value_output_wire("bytes", bytes);
        consumer.value_output_wire("blob", blob);
        let consumer = validate_with_manifests(
            &consumer.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([(production, manifest)]),
        )
        .expect("consumer validation");
        let mut consumer_backend = cpu_backend([parameters]);
        let consumed = execute(
            &consumer,
            &mut consumer_backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("consumer execution");
        assert!(matches!(
            &consumed.outputs["bytes"],
            RuntimeValue::Bytes(bytes) if bytes == &[1, 2, 3, 4]
        ));
        assert!(matches!(
            &consumed.outputs["blob"],
            RuntimeValue::TypedBlob(bytes) if bytes == &[8, 9, 10]
        ));
        assert!(consumed.production_id.is_none());
    }

    #[test]
    fn imported_bytes_materialize_when_consumed_by_hash_sampling() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut producer = GraphBuilder::new("hash-key-artifact-producer", Vec::new());
        let key = producer.bytes_input("key", 32);
        producer.output_wire("key", key, ArtifactConfidentiality::Private);
        let producer =
            validate(&producer.finish(), &ParamEnv::default()).expect("producer validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let produced = execute(
            &producer,
            &mut backend,
            BTreeMap::from([("key".to_owned(), RuntimeValue::Bytes(vec![23; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("producer execution");
        let production = produced.production_id.expect("key manifest");
        let manifest = store.manifest(&production).expect("manifest").clone();

        let mut consumer = GraphBuilder::new("hash-key-artifact-consumer", Vec::new());
        let key = consumer.artifact_bytes_input(
            "key",
            mxx_ir_core::IntExpr::constant(32),
            production.clone(),
            "key",
            ArtifactConfidentiality::Private,
        );
        let hash = consumer.hash_sample(
            key,
            matrix_type(&parameters),
            mxx_ir_core::node::HashVariant::Plain,
            b"imported-key".to_vec(),
            Vec::new(),
            None,
            None,
        );
        consumer.value_output_wire("hash", hash.wire);
        let consumer = validate_with_manifests(
            &consumer.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([(production.clone(), manifest.clone())]),
        )
        .expect("consumer validation");
        let consumed =
            execute(&consumer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("imported key is materialized for hashing");
        assert!(matches!(consumed.outputs["hash"], RuntimeValue::Matrix(_)));

        let mut substituted_store = MemoryArtifactStore::default();
        let mut substituted_manifest = manifest;
        substituted_manifest.artifacts.get_mut("key").expect("key descriptor").layout =
            Some("substituted-layout".to_owned());
        substituted_store
            .store_manifest(substituted_manifest)
            .expect("internally consistent substituted manifest");
        substituted_store
            .insert(
                ArtifactKey { production, name: "key".to_owned(), index: None },
                ArtifactType::Bytes { length: 32 },
                ArtifactConfidentiality::Private,
                ArtifactPayload::Bytes(vec![23; 32]),
            )
            .expect("substituted payload");
        let error = execute(
            &consumer,
            &mut backend,
            BTreeMap::new(),
            &mut substituted_store,
            SamplingMode::Fresh,
        )
        .err()
        .expect("runtime must reject a manifest other than the validated manifest");
        assert!(matches!(error, ExecutionError::Artifact(_)));
    }

    #[test]
    fn resumable_session_reuses_draws_and_returns_typed_private_handles() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut builder = GraphBuilder::new("resumable-session", Vec::new());
        let sample = builder.gaussian_sample(
            matrix_type(&parameters),
            mxx_ir_core::RealExpr::FromInt(mxx_ir_core::IntExpr::constant(3)),
        );
        builder.output("sample", &sample, ArtifactConfidentiality::Private);
        let validated =
            validate(&builder.finish(), &ParamEnv::default()).expect("session graph validation");
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let nonce = [47; 32];
        let first =
            execute_in_session(&validated, &mut backend, BTreeMap::new(), &mut store, nonce)
                .expect("first session execution");
        let production = first.production_id.clone().expect("session production");
        let RuntimeValue::Matrix(first_matrix) = &first.outputs["sample"] else {
            panic!("matrix output")
        };
        assert_eq!(store.session_status(&production), Some(crate::SessionStatus::Finalized));
        assert_eq!(store.transcript_len(&production), Some(1));
        let [handle] = first.artifact_handles["sample"].as_slice() else {
            panic!("one typed handle")
        };
        assert_eq!(handle.key.production, production);
        assert_eq!(handle.confidentiality, ArtifactConfidentiality::Private);
        assert!(matches!(handle.artifact_type, ArtifactType::Matrix(_)));

        let second =
            execute_in_session(&validated, &mut backend, BTreeMap::new(), &mut store, nonce)
                .expect("resumed finalized session");
        let RuntimeValue::Matrix(second_matrix) = &second.outputs["sample"] else {
            panic!("matrix output")
        };
        assert_eq!(backend.matrix_to_bytes(first_matrix), backend.matrix_to_bytes(second_matrix));
        assert_eq!(second.production_id, Some(production.clone()));
        assert_eq!(store.transcript_len(&production), Some(1));
    }

    #[test]
    fn session_release_failure_returns_retryable_staged_family_ownership() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let body = Graph {
            name: "release-failure-body".to_owned(),
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
            name: "release-failure-root".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(10),
                kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                    graph: body.name.clone(),
                    count: mxx_ir_core::IntExpr::constant(2),
                    minimum_count: 0,
                    index_variable: "i".to_owned(),
                    bindings: Vec::new(),
                    input_modes: Vec::new(),
                }),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("family".to_owned(), wire(10, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let mut error = {
            let mut failing = FailOnceCleanupStore {
                inner: &mut store,
                fail_next_remove: false,
                fail_next_release: true,
            };
            execute_in_session(&validated, &mut backend, BTreeMap::new(), &mut failing, [59; 32])
                .err()
                .expect("the first session release must fail")
        };
        let ExecutionError::StagedCleanup { leases, .. } = &error else {
            panic!("session release failure must return staged-family ownership");
        };
        assert_eq!(leases.len(), 1);
        error.cleanup_staged(&mut store).expect("caller can retry cleanup after the release retry");
        let ExecutionError::StagedCleanup { leases, .. } = &error else {
            unreachable!();
        };
        assert!(leases.is_empty());
    }

    #[test]
    fn repeated_session_executions_return_exclusive_staged_family_leases() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let graph = ephemeral_rotation_family_graph(&parameters, "exclusive-session-leases", 2);
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let nonce = [61; 32];

        let mut first =
            execute_in_session(&validated, &mut backend, BTreeMap::new(), &mut store, nonce)
                .expect("first session execution");
        let mut second =
            execute_in_session(&validated, &mut backend, BTreeMap::new(), &mut store, nonce)
                .expect("second session execution");
        let RuntimeValue::StagedArtifactFamily {
            production: second_production,
            name: second_name,
            descriptor: second_descriptor,
        } = &second.outputs["family"]
        else {
            panic!("ephemeral output must remain staged");
        };
        let second_key = ArtifactKey {
            production: second_production.clone(),
            name: second_name.clone(),
            index: Some(0),
        };
        let second_descriptor = second_descriptor.clone();
        assert_ne!(
            first.staged_family_leases[0].production, second.staged_family_leases[0].production,
            "each execution must own exclusive scratch coordinates"
        );

        first.cleanup_staged(&mut store).expect("cleanup first execution");
        store
            .load_staged(&second_key, &second_descriptor)
            .expect("first cleanup must not invalidate the second lease");
        second.cleanup_staged(&mut store).expect("cleanup second execution");
        assert!(store.load_staged(&second_key, &second_descriptor).is_err());
    }

    #[test]
    fn session_identity_rejects_changed_runtime_inputs() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut builder = GraphBuilder::new("session-input-identity", Vec::new());
        let bytes = builder.bytes_input("bytes", 4);
        builder.output_wire("bytes", bytes, ArtifactConfidentiality::Private);
        let validated =
            validate(&builder.finish(), &ParamEnv::default()).expect("session graph validation");
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let nonce = [53; 32];
        execute_in_session(
            &validated,
            &mut backend,
            BTreeMap::from([("bytes".to_owned(), RuntimeValue::Bytes(vec![1, 2, 3, 4]))]),
            &mut store,
            nonce,
        )
        .expect("first session execution");
        let error = match execute_in_session(
            &validated,
            &mut backend,
            BTreeMap::from([("bytes".to_owned(), RuntimeValue::Bytes(vec![4, 3, 2, 1]))]),
            &mut store,
            nonce,
        ) {
            Ok(_) => panic!("changed runtime input must conflict with the immutable session"),
            Err(error) => error,
        };
        assert!(
            matches!(error, ExecutionError::Artifact(message) if message.contains("descriptor conflicts"))
        );
    }

    #[test]
    fn session_repairs_only_missing_draws_from_a_partial_transcript() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut builder = GraphBuilder::new("partial-session-transcript", Vec::new());
        let left = builder.gaussian_sample(
            matrix_type(&parameters),
            mxx_ir_core::RealExpr::FromInt(mxx_ir_core::IntExpr::constant(3)),
        );
        let right = builder.gaussian_sample(
            matrix_type(&parameters),
            mxx_ir_core::RealExpr::FromInt(mxx_ir_core::IntExpr::constant(3)),
        );
        let sum =
            builder.matrix_binary(MatrixBinaryOp::Add, &left, &right, matrix_type(&parameters));
        builder.output("sum", &sum, ArtifactConfidentiality::Public);
        let validated =
            validate(&builder.finish(), &ParamEnv::default()).expect("session graph validation");
        let mut backend = cpu_backend([parameters]);
        let nonce = [59; 32];
        let spec_hash =
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings).expect("hash");
        let production = mxx_ir_core::artifact::production_id(spec_hash, nonce);
        let descriptor = SessionDescriptor::new(
            production.clone(),
            validated.source.name.clone(),
            runtime_inputs_digest(&backend, &BTreeMap::new()).expect("input digest"),
        );
        let first_wire = WireRef { node: NodeId(0), port: Port(0) };
        let first_type = validated
            .wires
            .get(&WireId { instantiation_path: Vec::new(), wire: first_wire })
            .and_then(ConcreteWireType::matrix_type)
            .cloned()
            .expect("first sample type");
        let first_value =
            backend.sample_gaussian(&first_type, 3.0).expect("partial transcript sample");
        let mut store = MemoryArtifactStore::default();
        store.open_session(&descriptor).expect("open partial session");
        store
            .record_transcript_batch(
                &production,
                &[(
                    DrawSite {
                        instantiation_path: Vec::new(),
                        node: first_wire.node,
                        port: first_wire.port,
                    },
                    RecordedValue::Matrix {
                        matrix_type: first_type,
                        bytes: backend.matrix_to_bytes(&first_value),
                    },
                )],
            )
            .expect("record first draw");
        store.release_session(&production).expect("release partial session");

        execute_in_session(&validated, &mut backend, BTreeMap::new(), &mut store, nonce)
            .expect("resume partial session");
        assert_eq!(store.transcript_len(&production), Some(2));
        assert_eq!(store.session_status(&production), Some(crate::SessionStatus::Finalized));
    }

    #[test]
    fn finalized_filesystem_session_repairs_a_missing_committed_payload() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut builder = GraphBuilder::new("repair-finalized-payload", Vec::new());
        let value = builder.constant_matrix(matrix_type(&parameters), ConstantMatrix::Identity);
        builder.output("value", &value, ArtifactConfidentiality::Public);
        let validated =
            validate(&builder.finish(), &ParamEnv::default()).expect("session graph validation");
        let temporary = tempfile::tempdir().expect("temporary artifact store");
        let mut store =
            FilesystemArtifactStore::open(temporary.path()).expect("filesystem artifact store");
        let mut backend = cpu_backend([parameters]);
        let nonce = [61; 32];
        let first =
            execute_in_session(&validated, &mut backend, BTreeMap::new(), &mut store, nonce)
                .expect("first finalized execution");
        let production = first.production_id.expect("session production");
        let manifest = store.load_manifest(&production).expect("finalized manifest");
        let descriptor = manifest.artifacts["value"].clone();
        let key =
            ArtifactKey { production: production.clone(), name: "value".to_owned(), index: None };
        store.remove_staged(&key).expect("fault injection removes one finalized payload");
        assert!(
            store.load(&key, &descriptor).is_err(),
            "the finalized manifest must not mask a missing payload"
        );

        let second =
            execute_in_session(&validated, &mut backend, BTreeMap::new(), &mut store, nonce)
                .expect("retry repairs finalized payload");
        assert_eq!(second.production_id, Some(production.clone()));
        assert_eq!(
            store.load_manifest(&production).expect("repaired manifest"),
            manifest,
            "repair must preserve the immutable finalized manifest"
        );
        store.load(&key, &descriptor).expect("retry recreates the missing committed payload");
    }

    #[test]
    fn private_trapdoor_artifact_round_trips_without_a_public_digest() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let matrix_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(parameters.modulus_digits().saturating_add(2)),
            ..matrix_type(&parameters)
        };
        let sigma = mxx_ir_core::expr::Rational::new(BigInt::from(13), BigInt::from(2))
            .expect("positive sigma");
        let gadget_base = mxx_ir_core::IntExpr::constant(BigInt::one() << parameters.base_bits());
        let digit_count = mxx_ir_core::IntExpr::constant(parameters.modulus_digits());
        let mut producer = GraphBuilder::new(
            "trapdoor-artifact-producer",
            vec![CompileParameter {
                name: "producer_sigma".to_owned(),
                kind: CompileParameterKind::Real,
            }],
        );
        let trapdoor = producer.trapdoor_sample(
            matrix_type.clone(),
            mxx_ir_core::RealExpr::Var("producer_sigma".to_owned()),
            gadget_base.clone(),
            digit_count.clone(),
        );
        producer.output_wire("trapdoor", trapdoor.wire, ArtifactConfidentiality::Private);
        let producer_bindings = ParamEnv {
            reals: BTreeMap::from([("producer_sigma".to_owned(), sigma.clone())]),
            ..ParamEnv::default()
        };
        let producer =
            validate(&producer.finish(), &producer_bindings).expect("producer validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let produced =
            execute(&producer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("producer execution");
        let RuntimeValue::Trapdoor { secret: Some(secret), public, .. } =
            &produced.outputs["trapdoor"]
        else {
            panic!("sampled trapdoor output");
        };
        let expected_secret = backend.trapdoor_to_bytes(secret);
        let expected_public = backend.matrix_to_bytes(public);
        let production = produced.production_id.expect("trapdoor manifest");
        let manifest = store.manifest(&production).expect("manifest").clone();
        assert!(matches!(
            manifest.artifacts["trapdoor"].artifact_type,
            ArtifactType::Trapdoor { .. }
        ));
        assert_eq!(
            manifest.artifacts["trapdoor"].confidentiality,
            ArtifactConfidentiality::Private
        );
        assert!(manifest.artifacts["trapdoor"].content_hash.is_none());

        let mut consumer = GraphBuilder::new(
            "trapdoor-artifact-consumer",
            vec![CompileParameter {
                name: "consumer_sigma".to_owned(),
                kind: CompileParameterKind::Real,
            }],
        );
        let trapdoor = consumer.artifact_trapdoor_input(
            "trapdoor",
            matrix_type.clone(),
            mxx_ir_core::RealExpr::Var("consumer_sigma".to_owned()),
            gadget_base.clone(),
            digit_count.clone(),
            production.clone(),
            "trapdoor",
            ArtifactConfidentiality::Private,
        );
        let target_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..matrix_type.clone()
        };
        let target =
            consumer.constant_matrix(target_type.clone(), mxx_ir_core::node::ConstantMatrix::Zero);
        let preimage_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(parameters.modulus_digits().saturating_add(2)),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..target_type.clone()
        };
        let preimage = consumer.preimage_sample(&trapdoor, &target, preimage_type.clone());
        consumer.value_output_wire("public", trapdoor.public.wire);
        consumer.value_output_wire("trapdoor", trapdoor.wire);
        consumer.value_output_wire("preimage", preimage.wire);
        let consumer = validate_with_manifests(
            &consumer.finish(),
            &ParamEnv {
                reals: BTreeMap::from([("consumer_sigma".to_owned(), sigma.clone())]),
                ..ParamEnv::default()
            },
            &BTreeMap::from([(production.clone(), manifest.clone())]),
        )
        .expect("consumer validation");
        let mut consumer_backend = cpu_backend([parameters]);
        let consumed = execute(
            &consumer,
            &mut consumer_backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("consumer execution");
        let RuntimeValue::Matrix(public) = &consumed.outputs["public"] else {
            panic!("trapdoor public matrix");
        };
        let RuntimeValue::Trapdoor { secret: Some(secret), .. } = &consumed.outputs["trapdoor"]
        else {
            panic!("loaded trapdoor");
        };
        assert!(matches!(consumed.outputs["preimage"], RuntimeValue::Matrix(_)));
        assert_eq!(consumer_backend.matrix_to_bytes(public), expected_public);
        assert_eq!(consumer_backend.trapdoor_to_bytes(secret), expected_secret);
        assert!(consumed.production_id.is_none());

        let direct_graph = Graph {
            name: "direct-imported-trapdoor-preimage".to_owned(),
            parameters: vec![CompileParameter {
                name: "direct_sigma".to_owned(),
                kind: CompileParameterKind::Real,
            }],
            input_types: BTreeMap::from([(
                "trapdoor".to_owned(),
                WireType::Trapdoor {
                    matrix: matrix_type.clone(),
                    sigma: mxx_ir_core::RealExpr::Var("direct_sigma".to_owned()),
                    gadget_base: gadget_base.clone(),
                    digit_count: digit_count.clone(),
                },
            )]),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "trapdoor".to_owned(),
                        wire_type: WireType::Trapdoor {
                            matrix: matrix_type,
                            sigma: mxx_ir_core::RealExpr::Var("direct_sigma".to_owned()),
                            gadget_base,
                            digit_count,
                        },
                        artifact: Some(mxx_ir_core::node::ArtifactInput {
                            production_id: production.clone(),
                            artifact_name: "trapdoor".to_owned(),
                            confidentiality: ArtifactConfidentiality::Private,
                        }),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: target_type,
                        value: mxx_ir_core::node::ConstantMatrix::Zero,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::PreimageSample { matrix_type: preimage_type },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            outputs: BTreeMap::from([("preimage".to_owned(), wire(3, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let direct = validate_with_manifests(
            &direct_graph,
            &ParamEnv {
                reals: BTreeMap::from([("direct_sigma".to_owned(), sigma)]),
                ..ParamEnv::default()
            },
            &BTreeMap::from([(production, manifest)]),
        )
        .expect("direct preimage consumer validation");
        let direct = execute(
            &direct,
            &mut consumer_backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("PreimageSample materializes its lazy trapdoor directly");
        assert!(matches!(direct.outputs["preimage"], RuntimeValue::Matrix(_)));
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
        assert_eq!(actual.as_ref(), &expected);
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
                tag_decimal_expressions: Vec::new(),
                tag_u64_le_expressions: Vec::new(),
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
        assert_eq!(&(gadget * decomposed.as_ref()), plain.as_ref());
    }

    #[test]
    fn decimal_hash_tag_expression_matches_the_equivalent_literal_tag() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let matrix_type = matrix_type(&parameters);
        let hash = |id, tag_prefix, tag_decimal_expressions| Node {
            id: NodeId(id),
            kind: NodeKind::HashSample {
                matrix_type: matrix_type.clone(),
                variant: mxx_ir_core::node::HashVariant::Plain,
                tag_prefix,
                tag_expressions: Vec::new(),
                tag_decimal_expressions,
                tag_u64_le_expressions: Vec::new(),
                base: None,
                digit_count: None,
            },
            args: vec![wire(1, 0)],
        };
        let graph = Graph {
            name: "decimal-hash-tag".to_owned(),
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
                hash(2, b"slot_transfer_slot_a_".to_vec(), vec![mxx_ir_core::IntExpr::constant(7)]),
                hash(3, b"slot_transfer_slot_a_7".to_vec(), Vec::new()),
            ],
            outputs: BTreeMap::from([
                ("expression".to_owned(), wire(2, 0)),
                ("literal".to_owned(), wire(3, 0)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &validated,
            &mut backend,
            BTreeMap::from([("key".to_owned(), RuntimeValue::Bytes(vec![19; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("execution");
        let RuntimeValue::Matrix(expression) = &result.outputs["expression"] else {
            panic!("expression matrix");
        };
        let RuntimeValue::Matrix(literal) = &result.outputs["literal"] else {
            panic!("literal matrix");
        };
        assert_eq!(expression.as_ref(), literal.as_ref());
    }

    #[test]
    fn loop_integer_expression_can_select_a_family_member() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let matrix_type = matrix_type(&parameters);
        let mut body = GraphBuilder::new("loop-integer-expression-body", Vec::new());
        let family =
            body.family_input("0_family", matrix_type.clone(), mxx_ir_core::IntExpr::constant(2));
        let iteration = body.evaluate_int(mxx_ir_core::IntExpr::Var("iteration".to_owned()));
        let one = body.constant_int(1);
        let zero = body.constant_int(0);
        let source = body.select_wire(iteration, &[one, zero]);
        let selected = body.family_get_dynamic(&family, source);
        body.value_output_wire("0_selected", selected.wire);

        let mut builder = GraphBuilder::new("loop-integer-expression", Vec::new());
        let zero_matrix = builder.constant_matrix(matrix_type.clone(), ConstantMatrix::Zero);
        let identity = builder.constant_matrix(matrix_type.clone(), ConstantMatrix::Identity);
        let family = builder.family_pack(&[zero_matrix, identity]).expect("family");
        let output = builder
            .parallel_loop(
                body.finish(),
                mxx_ir_core::IntExpr::constant(2),
                "iteration",
                Vec::new(),
                vec![family.wire],
                vec![mxx_ir_core::node::LoopInputMode::Broadcast],
                std::slice::from_ref(&matrix_type),
            )
            .expect("parallel loop")
            .remove(0);
        let first = builder.family_get_static(&output, mxx_ir_core::IntExpr::constant(0));
        let second = builder.family_get_static(&output, mxx_ir_core::IntExpr::constant(1));
        builder.value_output_wire("first", first.wire);
        builder.value_output_wire("second", second.wire);

        let validated = validate(&builder.finish(), &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        let RuntimeValue::Matrix(first) = &result.outputs["first"] else {
            panic!("first matrix");
        };
        let RuntimeValue::Matrix(second) = &result.outputs["second"] else {
            panic!("second matrix");
        };
        assert_eq!(first.as_ref(), &DCRTPolyMatrix::identity(&parameters, 1, None));
        assert_eq!(second.as_ref(), &DCRTPolyMatrix::zero(&parameters, 1, 1));
    }

    #[test]
    fn constant_coefficient_keeps_only_the_selected_polynomial_coefficient() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let matrix_type = matrix_type(&parameters);
        let mut builder = GraphBuilder::new("constant-coefficient", Vec::new());
        let input = builder.input("input", matrix_type);
        let output = builder.constant_coefficient(&input, mxx_ir_core::IntExpr::constant(2));
        builder.value_output_wire("output", output.wire);
        let validated = validate(&builder.finish(), &ParamEnv::default()).expect("validation");
        let input = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![vec![DCRTPoly::from_u32s(&parameters, &[3, 5, 7, 11])]],
        );
        let expected = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![vec![DCRTPoly::from_usize_to_constant(&parameters, 7)]],
        );
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &validated,
            &mut backend,
            BTreeMap::from([("input".to_owned(), RuntimeValue::matrix(input))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("execution");
        let RuntimeValue::Matrix(actual) = &result.outputs["output"] else {
            panic!("output matrix");
        };
        assert_eq!(actual.as_ref(), &expected);
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
        assert_eq!(actual.as_ref(), &expected);
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
                Node {
                    id: NodeId(15),
                    kind: NodeKind::Concat { axis: ConcatAxis::Rows },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(16),
                    kind: NodeKind::Concat { axis: ConcatAxis::Diagonal },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(17),
                    kind: NodeKind::ThresholdDecode {
                        plaintext_modulus: mxx_ir_core::IntExpr::constant(2),
                        length: mxx_ir_core::IntExpr::constant(2),
                        output_bool: true,
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
                ("concat_rows".to_owned(), wire(15, 0)),
                ("concat_diagonal".to_owned(), wire(16, 0)),
                ("decoded_bool_0".to_owned(), wire(17, 0)),
                ("decoded_bool_1".to_owned(), wire(17, 1)),
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
            ("concat_rows", one.concat_rows(&[&x])),
            ("concat_diagonal", one.concat_diag(&[&x])),
        ]);
        for (name, expected) in expected {
            let RuntimeValue::Matrix(actual) = &result.outputs[name] else {
                panic!("{name} must be a matrix");
            };
            assert_eq!(actual.as_ref(), &expected, "{name}");
        }
        assert!(matches!(
            &result.outputs["coefficient"],
            RuntimeValue::Int(value) if value == &BigInt::one()
        ));
        assert!(matches!(
            (&result.outputs["decoded_0"], &result.outputs["decoded_1"]),
            (RuntimeValue::Int(_), RuntimeValue::Int(_))
        ));
        assert!(matches!(
            (&result.outputs["decoded_bool_0"], &result.outputs["decoded_bool_1"]),
            (RuntimeValue::Bool(_), RuntimeValue::Bool(_))
        ));
    }

    #[test]
    fn constant_matrix_variants_match_direct_cpu_constructors() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let scalar_type = matrix_type(&parameters);
        let row_type =
            MatrixType { columns: mxx_ir_core::IntExpr::constant(3), ..scalar_type.clone() };
        let column_type =
            MatrixType { rows: mxx_ir_core::IntExpr::constant(3), ..scalar_type.clone() };
        let regular_gadget = DCRTPolyMatrix::gadget_matrix(&parameters, 1);
        let small_gadget = DCRTPolyMatrix::small_gadget_matrix(&parameters, 1);
        let regular_gadget_type = MatrixType {
            columns: mxx_ir_core::IntExpr::constant(regular_gadget.col_size()),
            ..scalar_type.clone()
        };
        let small_gadget_type = MatrixType {
            columns: mxx_ir_core::IntExpr::constant(small_gadget.col_size()),
            ..scalar_type.clone()
        };
        let gadget_base = mxx_ir_core::IntExpr::constant(BigInt::one() << parameters.base_bits());
        let graph = Graph {
            name: "constant-matrix-variant-conformance".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: scalar_type.clone(),
                        value: ConstantMatrix::Zero,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: scalar_type.clone(),
                        value: ConstantMatrix::Identity,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: row_type,
                        value: ConstantMatrix::UnitRow { index: mxx_ir_core::IntExpr::constant(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: column_type,
                        value: ConstantMatrix::UnitColumn {
                            index: mxx_ir_core::IntExpr::constant(2),
                        },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(5),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: regular_gadget_type,
                        value: ConstantMatrix::Gadget { base: gadget_base.clone(), small: false },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(6),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: small_gadget_type,
                        value: ConstantMatrix::Gadget { base: gadget_base, small: true },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(7),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: scalar_type,
                        value: ConstantMatrix::PowerOfBase {
                            base: mxx_ir_core::IntExpr::constant(3),
                            exponent: mxx_ir_core::IntExpr::constant(4),
                        },
                    },
                    args: Vec::new(),
                },
            ],
            outputs: BTreeMap::from([
                ("zero".to_owned(), wire(1, 0)),
                ("identity".to_owned(), wire(2, 0)),
                ("unit_row".to_owned(), wire(3, 0)),
                ("unit_column".to_owned(), wire(4, 0)),
                ("gadget".to_owned(), wire(5, 0)),
                ("small_gadget".to_owned(), wire(6, 0)),
                ("power".to_owned(), wire(7, 0)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        let expected = BTreeMap::from([
            ("zero", DCRTPolyMatrix::zero(&parameters, 1, 1)),
            ("identity", DCRTPolyMatrix::identity(&parameters, 1, None)),
            ("unit_row", DCRTPolyMatrix::unit_row_vector(&parameters, 3, 1)),
            ("unit_column", DCRTPolyMatrix::unit_column_vector(&parameters, 3, 2)),
            ("gadget", regular_gadget),
            ("small_gadget", small_gadget),
            (
                "power",
                DCRTPolyMatrix::from_poly_vec(
                    &parameters,
                    vec![vec![<DCRTPolyMatrix as PolyMatrix>::P::from_usize_to_constant(
                        &parameters,
                        81,
                    )]],
                ),
            ),
        ]);
        for (name, expected) in expected {
            let RuntimeValue::Matrix(actual) = &result.outputs[name] else {
                panic!("{name} must be a matrix");
            };
            assert_eq!(actual.as_ref(), &expected, "{name}");
        }
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
        assert_eq!(actual_up.as_ref(), &expected_up);
        assert_eq!(actual_down.as_ref(), &expected_down);
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
                Node { id: NodeId(12), kind: NodeKind::ConstantBool(false), args: Vec::new() },
                Node { id: NodeId(13), kind: NodeKind::BoolToInt, args: vec![wire(12, 0)] },
                Node {
                    id: NodeId(14),
                    kind: NodeKind::IntBinary(IntBinaryOp::Add),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(15),
                    kind: NodeKind::IntBinary(IntBinaryOp::Subtract),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(16),
                    kind: NodeKind::IntBinary(IntBinaryOp::Multiply),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(17),
                    kind: NodeKind::IntCompare(IntCompareOp::Equal),
                    args: vec![wire(2, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(18),
                    kind: NodeKind::IntCompare(IntCompareOp::LessEqual),
                    args: vec![wire(2, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(19),
                    kind: NodeKind::ConstantReal(rational(2, 1)),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(20),
                    kind: NodeKind::RealBinary(RealBinaryOp::Subtract),
                    args: vec![wire(8, 0), wire(19, 0)],
                },
                Node {
                    id: NodeId(21),
                    kind: NodeKind::RealBinary(RealBinaryOp::Multiply),
                    args: vec![wire(8, 0), wire(19, 0)],
                },
                Node {
                    id: NodeId(22),
                    kind: NodeKind::RealBinary(RealBinaryOp::Divide),
                    args: vec![wire(8, 0), wire(19, 0)],
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
                ("constant_bool".to_owned(), wire(12, 0)),
                ("constant_bool_int".to_owned(), wire(13, 0)),
                ("int_add".to_owned(), wire(14, 0)),
                ("int_subtract".to_owned(), wire(15, 0)),
                ("int_multiply".to_owned(), wire(16, 0)),
                ("int_equal".to_owned(), wire(17, 0)),
                ("int_less_equal".to_owned(), wire(18, 0)),
                ("real_subtract".to_owned(), wire(20, 0)),
                ("real_multiply".to_owned(), wire(21, 0)),
                ("real_divide".to_owned(), wire(22, 0)),
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
        assert!(matches!(&result.outputs["constant_bool"], RuntimeValue::Bool(false)));
        assert!(matches!(
            &result.outputs["constant_bool_int"],
            RuntimeValue::Int(value) if value.is_zero()
        ));
        assert!(matches!(
            &result.outputs["int_add"],
            RuntimeValue::Int(value) if value == &BigInt::from(-4)
        ));
        assert!(matches!(
            &result.outputs["int_subtract"],
            RuntimeValue::Int(value) if value == &BigInt::from(-10)
        ));
        assert!(matches!(
            &result.outputs["int_multiply"],
            RuntimeValue::Int(value) if value == &BigInt::from(-21)
        ));
        assert!(matches!(&result.outputs["int_equal"], RuntimeValue::Bool(true)));
        assert!(matches!(&result.outputs["int_less_equal"], RuntimeValue::Bool(true)));
        assert!(matches!(
            &result.outputs["real_subtract"],
            RuntimeValue::Real(value) if (*value - 0.25).abs() < 1e-12
        ));
        assert!(matches!(
            &result.outputs["real_multiply"],
            RuntimeValue::Real(value) if (*value - 4.5).abs() < 1e-12
        ));
        assert!(matches!(
            &result.outputs["real_divide"],
            RuntimeValue::Real(value) if (*value - 1.125).abs() < 1e-12
        ));
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
    fn subgraph_call_forwards_inputs_outputs_and_parameter_bindings() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let scalar_type = matrix_type(&parameters);
        let child = Graph {
            name: "scaled-child".to_owned(),
            parameters: vec![CompileParameter {
                name: "factor".to_owned(),
                kind: CompileParameterKind::Integer,
            }],
            input_types: BTreeMap::from([(
                "input".to_owned(),
                WireType::Matrix(scalar_type.clone()),
            )]),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "input".to_owned(),
                        wire_type: WireType::Matrix(scalar_type.clone()),
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::MatrixScale {
                        scalar: mxx_ir_core::IntExpr::Var("factor".to_owned()),
                    },
                    args: vec![wire(1, 0)],
                },
            ],
            outputs: BTreeMap::from([("scaled".to_owned(), wire(2, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let graph = Graph {
            name: "subgraph-call-conformance".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: scalar_type,
                        value: ConstantMatrix::Rotation {
                            exponent: mxx_ir_core::IntExpr::constant(1),
                        },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::SubgraphCall(mxx_ir_core::node::SubgraphCall {
                        graph: child.name.clone(),
                        bindings: vec![("factor".to_owned(), mxx_ir_core::IntExpr::constant(3))],
                    }),
                    args: vec![wire(1, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(2, 0))]),
            subgraphs: BTreeMap::from([(child.name.clone(), Box::new(child))]),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("execution");
        let input = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![vec![<DCRTPolyMatrix as PolyMatrix>::P::const_rotate_poly(&parameters, 1)]],
        );
        let expected =
            input * <DCRTPolyMatrix as PolyMatrix>::P::from_usize_to_constant(&parameters, 3);
        let RuntimeValue::Matrix(actual) = &result.outputs["out"] else {
            panic!("subgraph output must be a matrix");
        };
        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    fn parallel_loop_stamps_bindings_and_exposes_a_first_class_family() {
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
            nodes: vec![
                Node {
                    id: NodeId(10),
                    kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                        graph: body.name.clone(),
                        count: mxx_ir_core::IntExpr::constant(3),
                        minimum_count: 0,
                        index_variable: "i".to_owned(),
                        bindings: Vec::new(),
                        input_modes: Vec::new(),
                    }),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(0) },
                    args: vec![wire(10, 0)],
                },
                Node {
                    id: NodeId(12),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(1) },
                    args: vec![wire(10, 0)],
                },
                Node {
                    id: NodeId(13),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(2) },
                    args: vec![wire(10, 0)],
                },
            ],
            outputs: BTreeMap::from([
                ("family".to_owned(), wire(10, 0)),
                ("rotation_0".to_owned(), wire(11, 0)),
                ("rotation_1".to_owned(), wire(12, 0)),
                ("rotation_2".to_owned(), wire(13, 0)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let mut result =
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
            assert_eq!(actual.as_ref(), &expected);
        }
        let RuntimeValue::StagedArtifactFamily { production, name, descriptor } =
            &result.outputs["family"]
        else {
            panic!("ephemeral matrix family must remain a staged family");
        };
        let production = production.clone();
        let name = name.clone();
        let descriptor = descriptor.clone();
        let staged_key = ArtifactKey { production, name, index: Some(0) };
        store.load_staged(&staged_key, &descriptor).expect("lease keeps staged family readable");
        assert_eq!(result.staged_family_leases.len(), 1);
        {
            let mut failing = FailOnceCleanupStore {
                inner: &mut store,
                fail_next_remove: true,
                fail_next_release: false,
            };
            assert!(result.cleanup_staged(&mut failing).is_err());
        }
        assert_eq!(
            result.staged_family_leases.len(),
            1,
            "a failed cleanup must preserve the lease for retry"
        );
        result.cleanup_staged(&mut store).expect("explicit staged-family cleanup");
        assert!(store.load_staged(&staged_key, &descriptor).is_err());
        assert!(result.staged_family_leases.is_empty());
    }

    #[test]
    fn parallel_loop_places_instances_round_robin_and_preloads_broadcasts_once() {
        let placement_zero = DCRTPolyParams::new(8, 1, 20, 4);
        let placement_one = DCRTPolyParams::new(8, 1, 20, 5);
        let ty = WireType::Matrix(matrix_type(&placement_zero));
        let body = Graph {
            name: "placed-broadcast-body".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::from([("shared".to_owned(), ty.clone())]),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::Input {
                    name: "shared".to_owned(),
                    wire_type: ty.clone(),
                    artifact: None,
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("out".to_owned(), wire(1, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let mut graph = Graph {
            name: "placed-broadcast-loop".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::from([("shared".to_owned(), ty.clone())]),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "shared".to_owned(),
                        wire_type: ty,
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                        graph: body.name.clone(),
                        count: mxx_ir_core::IntExpr::constant(66),
                        minimum_count: 0,
                        index_variable: "i".to_owned(),
                        bindings: Vec::new(),
                        input_modes: vec![LoopInputMode::Broadcast],
                    }),
                    args: vec![wire(1, 0)],
                },
            ],
            outputs: BTreeMap::from([("family".to_owned(), wire(2, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));

        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let shared = DCRTPolyMatrix::identity(&placement_zero, 1, None);
        let modulus: Arc<num_bigint::BigUint> = placement_zero.modulus().into();
        let descriptor = ManifestArtifact {
            artifact_type: ArtifactType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()),
                ring_dimension: placement_zero.ring_dimension() as usize,
                rows: 1,
                columns: 1,
            }),
            family_count: None,
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: None,
            layout: None,
        };
        let production = ProductionId { spec_hash: SpecHash([41; 32]), execution_nonce: [42; 32] };
        let key =
            ArtifactKey { production: production.clone(), name: "shared".to_owned(), index: None };
        let mut backend =
            CpuDcrtBackend::new_with_placements(vec![vec![placement_zero], vec![placement_one]]);
        let mut store = MemoryArtifactStore::default();
        store
            .store_manifest(Manifest {
                ir_version: mxx_ir_core::encoding::IR_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([("shared".to_owned(), descriptor.clone())]),
            })
            .expect("broadcast manifest");
        store
            .insert(
                key.clone(),
                descriptor.artifact_type.clone(),
                descriptor.confidentiality,
                ArtifactPayload::Matrix(shared.to_compact_bytes()),
            )
            .expect("broadcast artifact");
        let (_, trace) = execute_with_trace(
            &validated,
            &mut backend,
            BTreeMap::from([(
                "shared".to_owned(),
                RuntimeValue::LazyArtifact {
                    production,
                    name: "shared".to_owned(),
                    index: None,
                    descriptor,
                },
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("placed execution");

        let matrices = [0u64, 1, 64, 65]
            .into_iter()
            .map(|index| {
                let id = WireId {
                    instantiation_path: vec![InstantiationFrame {
                        call: NodeId(2),
                        loop_index: Some(index),
                    }],
                    wire: wire(1, 0),
                };
                let RuntimeValue::Matrix(matrix) = trace.get(&id).expect("loop input trace") else {
                    panic!("loop input must be a matrix");
                };
                matrix.clone()
            })
            .collect::<Vec<_>>();
        assert_eq!(matrices[0].params().base_bits(), 4);
        assert_eq!(matrices[1].params().base_bits(), 5);
        assert_eq!(matrices[2].params().base_bits(), 4);
        assert_eq!(matrices[3].params().base_bits(), 5);
        assert!(Arc::ptr_eq(&matrices[0], &matrices[2]));
        assert!(Arc::ptr_eq(&matrices[1], &matrices[3]));
        assert_eq!(store.load_count(&key), 1);
    }

    #[test]
    fn persisted_parallel_family_streams_through_the_filesystem_store() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let body = Graph {
            name: "streamed-rotation-body".to_owned(),
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
            name: "streamed-parallel-family".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(10),
                    kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                        graph: body.name.clone(),
                        count: mxx_ir_core::IntExpr::constant(5),
                        minimum_count: 0,
                        index_variable: "i".to_owned(),
                        bindings: Vec::new(),
                        input_modes: Vec::new(),
                    }),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::Output {
                        name: "family".to_owned(),
                        artifact_confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                    args: vec![wire(10, 0)],
                },
            ],
            outputs: BTreeMap::from([("family".to_owned(), wire(11, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let temporary = tempfile::tempdir().expect("temporary artifact store");
        let mut store =
            FilesystemArtifactStore::open(temporary.path()).expect("filesystem artifact store");
        let mut backend = cpu_backend([parameters]);
        let result = execute_with_config(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig { max_parallel_instances: NonZeroUsize::new(2).expect("nonzero wave") },
        )
        .expect("streamed execution");
        let production = result.production_id.expect("persisted production");
        let RuntimeValue::LazyArtifactFamily { production: output_production, name, descriptor } =
            &result.outputs["family"]
        else {
            panic!("persisted streamed output must become a lazy final artifact family");
        };
        assert_eq!(output_production, &production);
        assert_eq!(descriptor.family_count, Some(5));
        for index in 0..5 {
            store
                .load(
                    &ArtifactKey {
                        production: production.clone(),
                        name: name.clone(),
                        index: Some(index),
                    },
                    descriptor,
                )
                .expect("final streamed member");
        }
        assert_eq!(
            store.family_hash_verification_count(),
            1,
            "filesystem family hash must be verified once rather than once per member"
        );
        assert!(
            result.staged_family_leases.is_empty(),
            "persisted streamed families must not return scratch cleanup ownership"
        );
    }

    #[test]
    fn nested_parallel_loops_use_representative_metadata_for_every_outer_iteration() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let inner = Graph {
            name: "nested-inner".to_owned(),
            parameters: vec![CompileParameter {
                name: "j".to_owned(),
                kind: CompileParameterKind::Integer,
            }],
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::ConstantMatrix {
                    matrix_type: matrix_type(&parameters),
                    value: mxx_ir_core::node::ConstantMatrix::Rotation {
                        exponent: mxx_ir_core::IntExpr::Var("j".to_owned()),
                    },
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("out".to_owned(), wire(1, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let mut outer = Graph {
            name: "nested-outer".to_owned(),
            parameters: vec![CompileParameter {
                name: "i".to_owned(),
                kind: CompileParameterKind::Integer,
            }],
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(10),
                    kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                        graph: inner.name.clone(),
                        count: mxx_ir_core::IntExpr::constant(2),
                        minimum_count: 0,
                        index_variable: "j".to_owned(),
                        bindings: Vec::new(),
                        input_modes: Vec::new(),
                    }),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(0) },
                    args: vec![wire(10, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(11, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        outer.subgraphs.insert(inner.name.clone(), Box::new(inner));
        let mut graph = Graph {
            name: "nested-root".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(20),
                    kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                        graph: outer.name.clone(),
                        count: mxx_ir_core::IntExpr::constant(3),
                        minimum_count: 0,
                        index_variable: "i".to_owned(),
                        bindings: Vec::new(),
                        input_modes: Vec::new(),
                    }),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(21),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(2) },
                    args: vec![wire(20, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(21, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(outer.name.clone(), Box::new(outer));
        let validated = validate(&graph, &ParamEnv::default()).expect("nested validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute_with_config(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig { max_parallel_instances: NonZeroUsize::new(2).expect("nonzero wave") },
        )
        .expect("all outer iterations use representative nested metadata");
        let RuntimeValue::Matrix(actual) = &result.outputs["out"] else {
            panic!("nested output must be a matrix");
        };
        let expected = DCRTPolyMatrix::identity(&parameters, 1, None);
        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    fn failed_execution_returns_retryable_scratch_cleanup_ownership() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let body = Graph {
            name: "cleanup-failure-body".to_owned(),
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
            name: "cleanup-failure-root".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(10),
                    kind: NodeKind::ParallelLoop(mxx_ir_core::node::ParallelLoop {
                        graph: body.name.clone(),
                        count: mxx_ir_core::IntExpr::constant(2),
                        minimum_count: 0,
                        index_variable: "i".to_owned(),
                        bindings: Vec::new(),
                        input_modes: Vec::new(),
                    }),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::ConstantInt(BigInt::one()),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(12),
                    kind: NodeKind::ConstantInt(BigInt::zero()),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(13),
                    kind: NodeKind::IntBinary(IntBinaryOp::Divide),
                    args: vec![wire(11, 0), wire(12, 0)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(13, 0))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let mut error = {
            let mut failing = FailOnceCleanupStore {
                inner: &mut store,
                fail_next_remove: true,
                fail_next_release: false,
            };
            execute(&validated, &mut backend, BTreeMap::new(), &mut failing, SamplingMode::Fresh)
                .err()
                .expect("division and first cleanup attempt fail")
        };
        let ExecutionError::StagedCleanup { leases, .. } = &error else {
            panic!("failed execution must return cleanup ownership");
        };
        assert_eq!(leases.len(), 1);
        error.cleanup_staged(&mut store).expect("retry cleanup from execution error");
        let ExecutionError::StagedCleanup { leases, .. } = &error else {
            unreachable!();
        };
        assert!(leases.is_empty());
    }

    #[test]
    fn parallel_loop_groups_preimage_batches_by_wave_and_placement() {
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
        let trapdoor_wire_type = WireType::Trapdoor {
            matrix: public_type.clone(),
            sigma: sigma.clone(),
            gadget_base: mxx_ir_core::IntExpr::constant(BigInt::one() << parameters.base_bits()),
            digit_count: mxx_ir_core::IntExpr::constant(parameters.modulus_digits()),
        };
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
                    kind: NodeKind::TrapdoorSample {
                        matrix_type: public_type,
                        sigma,
                        gadget_base: mxx_ir_core::IntExpr::constant(
                            BigInt::one() << parameters.base_bits(),
                        ),
                        digit_count: mxx_ir_core::IntExpr::constant(parameters.modulus_digits()),
                    },
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
                        count: mxx_ir_core::IntExpr::constant(5),
                        minimum_count: 0,
                        index_variable: "i".to_owned(),
                        bindings: Vec::new(),
                        input_modes: vec![
                            mxx_ir_core::node::LoopInputMode::Broadcast,
                            mxx_ir_core::node::LoopInputMode::Broadcast,
                        ],
                    }),
                    args: vec![wire(1, 1), wire(2, 0)],
                },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(0) },
                    args: vec![wire(10, 0)],
                },
                Node {
                    id: NodeId(12),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(1) },
                    args: vec![wire(10, 0)],
                },
                Node {
                    id: NodeId(13),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(2) },
                    args: vec![wire(10, 0)],
                },
                Node {
                    id: NodeId(14),
                    kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(4) },
                    args: vec![wire(10, 0)],
                },
            ],
            outputs: BTreeMap::from([
                ("preimage_0".to_owned(), wire(11, 0)),
                ("preimage_1".to_owned(), wire(12, 0)),
                ("preimage_2".to_owned(), wire(13, 0)),
                ("preimage_4".to_owned(), wire(14, 0)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = CpuDcrtBackend::new_with_placements(vec![
            vec![parameters.clone()],
            vec![parameters.clone()],
        ]);
        let mut store = MemoryArtifactStore::default();
        let mut recorder = TranscriptRecorder::default();
        let recorded = execute_with_config(
            &validated,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Record(&mut recorder),
            ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(2).expect("nonzero wave size"),
            },
        )
        .expect("batched execution");
        assert_eq!(backend.preimage_batch_calls(), 4);
        let replayer = recorder.into_replayer();
        let mut replay_backend =
            CpuDcrtBackend::new_with_placements(vec![vec![parameters.clone()], vec![parameters]]);
        let replayed = execute(
            &validated,
            &mut replay_backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Replay(&replayer),
        )
        .expect("batched replay");
        assert_eq!(replay_backend.preimage_batch_calls(), 0);
        for index in [0, 1, 2, 4] {
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
}
