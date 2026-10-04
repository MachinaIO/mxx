//! Common stage orchestration for Exponent-LUT.
//!
//! The API in this module owns only the production lifecycle. Program
//! traversal remains in `program::lowering`; concrete schemes provide the
//! stage builders and their evaluator state. Encryption is intentionally
//! input-only and never receives a whole Program.

use crate::{
    noise_refresh::{NoiseRefresh, RefreshDeclaration, prf::PrfLabel},
    program::{
        Environment, ExponentLUTProgramLowering, ExponentLutProgram, FamilyExecution,
        ProgramLowerError, RhsId, WireId, lower_program,
    },
};
use mxx_backends::{
    ArtifactHandle, ArtifactKey, ArtifactPayload, ArtifactStore, ExecutionConfig, ExecutionError,
    FileArtifactError, FileArtifactStore, MemoryArtifactStore, RuntimeValue,
    SessionAliasDescriptor, SessionDescriptor, SessionStatus, SessionStore,
    artifact::MemoryArtifactError,
    authority::CpuExecution,
    execute_in_session,
    matrix::dcrt_poly::DCRTPolyMatrix,
    poly::Poly,
    transcript::{DrawSite, RecordedValue},
};
use mxx_dsl::BuiltGraph;
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::{ArtifactAvailability, ArtifactType, Manifest, ManifestArtifact, ProductionId},
    encoding::spec_hash,
};
use num_bigint::BigInt;
use std::{collections::BTreeMap, error::Error, path::PathBuf};
use thiserror::Error as ThisError;

/// Where stages keep the artifacts they hand to later stages.
#[derive(Clone, Debug)]
pub enum StageStorage {
    /// GPU memory: a GPU stage exports matrix artifacts without leaving the
    /// device and the next GPU stage imports them by a device copy. Other
    /// artifacts, and host readers, use host memory.
    #[cfg(feature = "gpu")]
    Device,
    /// Host memory.
    Host,
    /// Files under this directory.
    File(PathBuf),
}

impl StageStorage {
    pub fn open(&self) -> Result<StageStore, ApiError> {
        Ok(match self {
            #[cfg(feature = "gpu")]
            Self::Device => StageStore::Memory(MemoryArtifactStore::on_device()),
            Self::Host => StageStore::Memory(MemoryArtifactStore::default()),
            Self::File(root) => {
                StageStore::File(FileArtifactStore::new(root).map_err(|error| {
                    ApiError::Runtime(ExecutionError::Artifact(error.to_string()))
                })?)
            }
        })
    }
}

/// The session store a [`StageStorage`] opens.
pub enum StageStore {
    Memory(MemoryArtifactStore),
    File(FileArtifactStore),
}

/// An artifact a [`StageStore`] encoded and has not yet published, of the
/// store kind that encoded it.
pub enum StageEncoded {
    Memory(mxx_backends::artifact::MemoryEncodedArtifact),
    File(mxx_backends::artifact::FileEncodedArtifact),
}

#[derive(Debug, ThisError)]
pub enum StageStoreError {
    #[error(transparent)]
    Memory(MemoryArtifactError),
    #[error(transparent)]
    File(FileArtifactError),
    #[error("an encoded artifact was published to another kind of store")]
    EncodedStoreMismatch,
}

macro_rules! delegate_store {
    ($self:ident, $store:ident => $call:expr) => {
        match $self {
            StageStore::Memory($store) => $call.map_err(StageStoreError::Memory),
            StageStore::File($store) => $call.map_err(StageStoreError::File),
        }
    };
}

impl ArtifactStore for StageStore {
    type Error = StageStoreError;
    type Encoded = StageEncoded;

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
        delegate_store!(self, store => store.load_manifest(production))
    }

    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        delegate_store!(self, store => store.load(key, descriptor))
    }

    fn load_payload_size(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<usize, Self::Error> {
        delegate_store!(self, store => store.load_payload_size(key, descriptor))
    }

    fn store(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.store(key, artifact_type, availability, layout, payload))
    }

    fn stage_raw_chunk(
        &mut self,
        key: ArtifactKey,
        total_raw_bytes: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<bool, Self::Error> {
        delegate_store!(self, store => store.stage_raw_chunk(key, total_raw_bytes, offset, bytes))
    }

    fn encode_staged(
        &mut self,
        key: &ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload_kind: u8,
        encode: mxx_backends::artifact::StagedEncoder,
    ) -> Result<mxx_backends::artifact::EncodeJob<Self::Encoded, Self::Error>, Self::Error> {
        Ok(match self {
            StageStore::Memory(store) => {
                let job = store
                    .encode_staged(key, artifact_type, availability, layout, payload_kind, encode)
                    .map_err(StageStoreError::Memory)?;
                Box::new(move || {
                    job()
                        .map(|(encoded, bytes)| (StageEncoded::Memory(encoded), bytes))
                        .map_err(StageStoreError::Memory)
                })
            }
            StageStore::File(store) => {
                let job = store
                    .encode_staged(key, artifact_type, availability, layout, payload_kind, encode)
                    .map_err(StageStoreError::File)?;
                Box::new(move || {
                    job()
                        .map(|(encoded, bytes)| (StageEncoded::File(encoded), bytes))
                        .map_err(StageStoreError::File)
                })
            }
        })
    }

    fn publish_encoded(
        &mut self,
        key: ArtifactKey,
        encoded: Self::Encoded,
    ) -> Result<(), Self::Error> {
        match (self, encoded) {
            (StageStore::Memory(store), StageEncoded::Memory(encoded)) => {
                store.publish_encoded(key, encoded).map_err(StageStoreError::Memory)
            }
            (StageStore::File(store), StageEncoded::File(encoded)) => {
                store.publish_encoded(key, encoded).map_err(StageStoreError::File)
            }
            _ => Err(StageStoreError::EncodedStoreMismatch),
        }
    }

    fn load_staged(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        delegate_store!(self, store => store.load_staged(key, descriptor))
    }

    fn remove_staged(&mut self, key: &ArtifactKey) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.remove_staged(key))
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.store_manifest(manifest))
    }

    fn evict_cached(&mut self, key: &ArtifactKey) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.evict_cached(key))
    }

    #[cfg(feature = "gpu")]
    fn device_artifacts(&mut self) -> Option<&mut mxx_backends::device_artifact::DeviceArtifacts> {
        match self {
            StageStore::Memory(store) => store.device_artifacts(),
            StageStore::File(store) => store.device_artifacts(),
        }
    }
}

impl SessionStore for StageStore {
    fn resolve_session_nonce(
        &mut self,
        descriptor: &SessionAliasDescriptor,
    ) -> Result<[u8; 32], Self::Error> {
        delegate_store!(self, store => store.resolve_session_nonce(descriptor))
    }

    fn open_session(
        &mut self,
        descriptor: &SessionDescriptor,
    ) -> Result<SessionStatus, Self::Error> {
        delegate_store!(self, store => store.open_session(descriptor))
    }

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.release_session(production))
    }

    fn discard_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.discard_session(production))
    }

    fn transcript_entry(
        &mut self,
        production: &ProductionId,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, Self::Error> {
        delegate_store!(self, store => store.transcript_entry(production, site))
    }

    fn record_transcript_batch(
        &mut self,
        production: &ProductionId,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.record_transcript_batch(production, entries))
    }

    fn commit_artifact(&mut self, handle: &ArtifactHandle) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.commit_artifact(handle))
    }

    fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        delegate_store!(self, store => store.finalize_session(manifest))
    }

    fn load_finalized_manifest(
        &mut self,
        production: &ProductionId,
    ) -> Result<Manifest, Self::Error> {
        delegate_store!(self, store => store.load_finalized_manifest(production))
    }

    fn load_finalized_named_manifest(
        &mut self,
        expected: &SessionAliasDescriptor,
    ) -> Result<Manifest, Self::Error> {
        delegate_store!(self, store => store.load_finalized_named_manifest(expected))
    }
}

/// Errors raised while composing the common refresh declaration with a
/// concrete stage refresh implementation.
#[derive(Debug, ThisError)]
pub enum RefreshApplicationError<RE>
where
    RE: Error + Send + Sync + 'static,
{
    #[error("refresh implementation failed: {0}")]
    Refresh(RE),
    #[error("refresh declaration and lowered PRF outputs do not agree")]
    OutputMismatch,
}

/// Plaintext values bound to declared Program inputs.
pub type PlaintextInputs<I, P> = BTreeMap<I, crate::program::InputBinding<P, Vec<P>>>;

/// The complete graph and runtime inputs produced by one stage builder.
///
/// `BuiltGraph` is deliberately unvalidated here. Stage orchestration
/// validates it against the invocation's actual parameter environment and the
/// application-owned manifest map before execution.
pub struct StageGraph {
    pub(crate) graph: BuiltGraph,
    pub(crate) runtime_inputs: BTreeMap<String, RuntimeValue>,
    pub(crate) manifests: BTreeMap<ProductionId, Manifest>,
}

impl StageGraph {
    pub fn new(
        graph: BuiltGraph,
        runtime_inputs: BTreeMap<String, RuntimeValue>,
        manifests: BTreeMap<ProductionId, Manifest>,
    ) -> Self {
        Self { graph, runtime_inputs, manifests }
    }
}

/// Owned outputs of one non-producing stage execution.
///
/// Outputs are copied out of executor-owned storage, so they remain valid
/// after the executor runs another graph and can be bound as inputs of a
/// later graph on the same executor.
pub struct StageOutputs {
    pub outputs: BTreeMap<String, RuntimeValue>,
}

/// The executor boundary used by every Exponent-LUT stage.
///
/// CPU execution and compiled GPU execution return differently owned
/// results; this trait keeps the stage contract local to Exponent-LUT:
/// producing stages report their finalized production, evaluating stages
/// return owned outputs, and verification downloads single matrices.
pub trait StageExecution<S: SessionStore> {
    /// Execute an artifact-producing graph in the session selected by
    /// `execution_nonce` and return its production identity. An execution
    /// that only estimates stages may plan the graph without executing it and
    /// return the identity its execution would have.
    fn produce(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<ProductionId, ApiError>;

    /// Execute a graph and return owned copies of all of its outputs. An
    /// execution that only estimates stages may plan a graph that reads or
    /// writes artifacts without executing it and return no outputs.
    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<StageOutputs, ApiError>;

    /// Download one matrix output to host memory.
    fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, ApiError>;

    /// Download one integer output, or the members of an integer family, to
    /// host memory.
    fn download_integers(&self, value: &RuntimeValue) -> Result<Vec<BigInt>, ApiError>;
}

impl<S: SessionStore> StageExecution<S> for CpuExecution {
    fn produce(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<ProductionId, ApiError> {
        let result = execute_in_session(
            &validated,
            self.backend_mut(),
            inputs,
            store,
            execution_nonce,
            ExecutionConfig::default(),
        )?;
        result.production_id.ok_or_else(|| {
            ApiError::input("artifact execution did not produce a production identity")
        })
    }

    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<StageOutputs, ApiError> {
        let mut result = execute_in_session(
            &validated,
            self.backend_mut(),
            inputs,
            store,
            execution_nonce,
            ExecutionConfig::default(),
        )?;
        let names = result.outputs.keys().cloned().collect::<Vec<_>>();
        let mut outputs = BTreeMap::new();
        for name in names {
            let value = result.materialize_output(&name, self.backend(), store)?.clone();
            outputs.insert(name, value);
        }
        result.cleanup_staged(store)?;
        Ok(StageOutputs { outputs })
    }

    fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, ApiError> {
        let RuntimeValue::Matrix(matrix) = value else {
            return Err(ApiError::input("stage output is not a matrix"));
        };
        matrix
            .cpu_full_arc()
            .map(|matrix| matrix.as_ref().clone())
            .ok_or_else(|| ApiError::input("CPU stage output is not a full host matrix"))
    }

    fn download_integers(&self, value: &RuntimeValue) -> Result<Vec<BigInt>, ApiError> {
        let integer = |value: &RuntimeValue| match value {
            RuntimeValue::Int(value) => Ok(value.clone()),
            _ => Err(ApiError::input("stage output is not an integer")),
        };
        match value {
            RuntimeValue::IndexedFamily { values, .. } => values.iter().map(integer).collect(),
            value => Ok(vec![integer(value)?]),
        }
    }
}

/// Connection metadata for one completed artifact production.
///
/// The application owns the manifest and descriptor. Large payloads remain in
/// the `SessionStore` and are imported lazily by concrete builders.
pub struct StageArtifacts<D> {
    pub(crate) production: ProductionId,
    pub(crate) manifest: Manifest,
    pub(crate) descriptor: D,
    pub(crate) session: SessionAliasDescriptor,
}

impl<D> StageArtifacts<D> {
    fn new(
        production: ProductionId,
        manifest: Manifest,
        descriptor: D,
        session: SessionAliasDescriptor,
    ) -> Self {
        Self { production, manifest, descriptor, session }
    }

    /// Returns the finalized production identity recorded for this stage.
    pub fn production(&self) -> &ProductionId {
        &self.production
    }

    /// Returns the manifest loaded for this stage.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Returns the application-owned descriptor for this stage.
    pub fn descriptor(&self) -> &D {
        &self.descriptor
    }

    /// Returns the immutable named-session identity for this artifact stage.
    pub fn session(&self) -> &SessionAliasDescriptor {
        &self.session
    }
}

/// Errors exposed by the common production API.
#[derive(Debug, ThisError)]
pub enum ApiError {
    #[error("invalid stage input: {message}")]
    Input { message: String },
    #[error("incompatible stage input or artifact: {message}")]
    Compatibility { message: String },
    #[error("program lowering failed: {source}")]
    Lowering { source: Box<dyn Error + Send + Sync> },
    #[error("scheme stage construction failed: {source}")]
    Scheme { source: Box<dyn Error + Send + Sync> },
    #[error("graph validation failed: {source}")]
    Graph { source: Box<dyn Error + Send + Sync> },
    #[error("runtime execution failed: {0}")]
    Runtime(#[from] ExecutionError),
}

impl ApiError {
    pub(crate) fn lowering<E>(error: ProgramLowerError<E>) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::Lowering { source: Box::new(error) }
    }

    pub(crate) fn input(message: impl Into<String>) -> Self {
        Self::Input { message: message.into() }
    }

    pub(crate) fn compatibility(message: impl Into<String>) -> Self {
        Self::Compatibility { message: message.into() }
    }

    pub(crate) fn scheme<E>(error: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::Scheme { source: Box::new(error) }
    }

    fn graph<E>(error: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::Graph { source: Box::new(error) }
    }
}

/// Builder contract for setup/preprocess graph construction.
pub trait PreprocessBuild<P: Poly> {
    type Evaluator: ExponentLUTProgramLowering + crate::program::ProgramLoweringContext;
    type Families: FamilyExecution<Self::Evaluator>;
    type PayloadSecret;
    type EncryptDescriptor;
    type OnlineEvalDescriptor;
    type Encrypt: EncryptBuild<
            P,
            PayloadSecret = Self::PayloadSecret,
            PreprocessDescriptor = Self::EncryptDescriptor,
        >;
    type OnlineEval: OnlineEvalBuild<
            PreprocessDescriptor = Self::OnlineEvalDescriptor,
            EncryptDescriptor = <Self::Encrypt as EncryptBuild<P>>::Descriptor,
        >;

    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                <Self::Evaluator as ExponentLUTProgramLowering>::Wire,
                <Self::Evaluator as ExponentLUTProgramLowering>::Rhs,
                <Self::Families as FamilyExecution<Self::Evaluator>>::WireFamily,
                <Self::Families as FamilyExecution<Self::Evaluator>>::RhsFamily,
            >,
        ),
        ApiError,
    >;

    /// Bind the Program, its RHS plaintexts, and the payload secret. A scheme
    /// whose RHS values are chosen only at encryption receives no RHS and no
    /// secret here.
    fn bind_inputs(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &ParamEnv,
        rhs: &PlaintextInputs<RhsId, P>,
        payload: Option<&Self::PayloadSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError>;

    fn artifact_identities(&self) -> ([u8; 32], [u8; 32]);

    /// Finalize descriptors after execution has assigned the shared
    /// production nonce.  Concrete schemes may use this hook to attach
    /// dependency edges that cannot be known during graph construction.
    fn finalize_descriptors(
        production: &mxx_ir_core::artifact::ProductionId,
        descriptors: (Self::EncryptDescriptor, Self::OnlineEvalDescriptor),
    ) -> Result<(Self::EncryptDescriptor, Self::OnlineEvalDescriptor), ApiError>;

    fn build_graph(
        self,
        outputs: Vec<<Self::Evaluator as ExponentLUTProgramLowering>::Wire>,
    ) -> Result<(StageGraph, (Self::EncryptDescriptor, Self::OnlineEvalDescriptor)), ApiError>;
}

/// Builder contract for the input-only encryption graph.
pub trait EncryptBuild<P: Poly> {
    type PayloadSecret;
    type MaskSecret;
    type PreprocessDescriptor;
    type Descriptor;

    /// Bind the input plaintexts and secrets. A scheme that encrypts under
    /// one secret takes no separate mask.
    fn bind_inputs(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &ParamEnv,
        lhs: &PlaintextInputs<WireId, P>,
        payload: &Self::PayloadSecret,
        mask: Option<&Self::MaskSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError>;

    fn bind_artifacts(
        &mut self,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
    ) -> Result<(), ApiError>;

    fn artifact_identity(&self) -> [u8; 32];

    fn build_graph(self) -> Result<(StageGraph, Self::Descriptor), ApiError>;
}

/// Builder contract for online evaluation: Program lowering and scheme refresh graph
/// construction.
pub trait OnlineEvalBuild {
    type Evaluator: ExponentLUTProgramLowering + crate::program::ProgramLoweringContext;
    type Families: FamilyExecution<Self::Evaluator>;
    type PreprocessDescriptor;
    type EncryptDescriptor;

    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                <Self::Evaluator as ExponentLUTProgramLowering>::Wire,
                <Self::Evaluator as ExponentLUTProgramLowering>::Rhs,
                <Self::Families as FamilyExecution<Self::Evaluator>>::WireFamily,
                <Self::Families as FamilyExecution<Self::Evaluator>>::RhsFamily,
            >,
        ),
        ApiError,
    >;

    fn bind_artifacts(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &ParamEnv,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
        encryption: &StageArtifacts<Self::EncryptDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError>;

    fn build_graph(
        self,
        outputs: Vec<<Self::Evaluator as ExponentLUTProgramLowering>::Wire>,
    ) -> Result<StageGraph, ApiError>;
}

/// Builder contract for Program-based noise simulation.
pub trait NoiseBuild {
    type Evaluator: ExponentLUTProgramLowering + crate::program::ProgramLoweringContext;
    type Families: FamilyExecution<Self::Evaluator>;
    type Report;

    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                <Self::Evaluator as ExponentLUTProgramLowering>::Wire,
                <Self::Evaluator as ExponentLUTProgramLowering>::Rhs,
                <Self::Families as FamilyExecution<Self::Evaluator>>::WireFamily,
                <Self::Families as FamilyExecution<Self::Evaluator>>::RhsFamily,
            >,
        ),
        ApiError,
    >;

    fn build_report(
        self,
        outputs: Vec<<Self::Evaluator as ExponentLUTProgramLowering>::Wire>,
    ) -> Result<Self::Report, ApiError>;
}

/// Execute one artifact-producing graph and capture its finalized manifest.
/// Validate, execute, and persist one already-built stage graph.
///
/// This is the common persistence boundary used when a refreshed graph must
/// resume through the normal `StageArtifacts` lifecycle. It does not create a
/// second executor or bypass graph validation or observers.
pub(crate) fn execute_artifacts<E, S, D>(
    graph: StageGraph,
    parameters: &ParamEnv,
    descriptor: D,
    identity: [u8; 32],
    execution: &mut E,
    store: &mut S,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<StageArtifacts<D>, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let started = std::time::Instant::now();
    let validated = validate_stage(&graph, parameters)?;
    before_execute(&validated)?;
    let (execution_nonce, session) = fresh_execution_session(&validated, store, identity)?;
    let (production, manifest) = stage_production(&validated, execution_nonce)?;
    let validated_at = std::time::Instant::now();
    let produced = execution.produce(validated, graph.runtime_inputs, store, execution_nonce)?;
    check_production(&produced, &production)?;
    let produced_at = std::time::Instant::now();
    tracing::info!(
        validate_ms = (validated_at - started).as_millis() as u64,
        produce_ms = (produced_at - validated_at).as_millis() as u64,
        manifest_ms = produced_at.elapsed().as_millis() as u64,
        "artifact stage timing"
    );
    Ok(StageArtifacts::new(production, manifest, descriptor, session))
}

fn validate_stage(graph: &StageGraph, parameters: &ParamEnv) -> Result<ValidatedGraph, ApiError> {
    graph.graph.validate_with_manifests(parameters, &graph.manifests).map_err(ApiError::graph)
}

/// The production a stage executed with `execution_nonce` publishes and its
/// manifest, both determined by the validated graph alone. Stage records are
/// built from them before execution, so a later stage can be built and
/// planned whether or not this one has executed; a consumer's store read
/// checks that the production was finalized.
fn stage_production(
    validated: &ValidatedGraph,
    execution_nonce: [u8; 32],
) -> Result<(ProductionId, Manifest), ApiError> {
    let specification =
        spec_hash(&validated.source, &validated.bindings).map_err(ApiError::graph)?;
    let production = mxx_ir_core::artifact::production_id(specification, execution_nonce);
    let manifest = mxx_ir_core::artifact::export_validated_manifest(production.clone(), validated)
        .map_err(ApiError::graph)?;
    Ok((production, manifest))
}

fn check_production(produced: &ProductionId, expected: &ProductionId) -> Result<(), ApiError> {
    if produced == expected {
        Ok(())
    } else {
        Err(ApiError::compatibility("stage execution returned a different production identity"))
    }
}

/// Reload a finalized stage from its named session after an application
/// restart.
///
/// The descriptor is application-owned and must be reconstructed or loaded by
/// the caller.  The runtime is the authority for the manifest: this helper
/// resolves the existing alias with the side-effect-free finalized-session
/// read, then checks that the alias and manifest agree on the IR/specification
/// identity before rebuilding the ordinary [`StageArtifacts`] carrier.
pub fn reload_stage_artifacts<S, D>(
    store: &mut S,
    session: SessionAliasDescriptor,
    descriptor: D,
) -> Result<StageArtifacts<D>, ApiError>
where
    S: SessionStore,
{
    let manifest = store
        .load_finalized_named_manifest(&session)
        .map_err(|error| ApiError::Runtime(ExecutionError::Artifact(error.to_string())))?;
    if manifest.ir_version != session.ir_version {
        return Err(ApiError::compatibility(
            "finalized named session manifest has a different IR version",
        ));
    }
    if manifest.production_id.spec_hash != session.spec_hash {
        return Err(ApiError::compatibility(
            "finalized named session manifest has a different specification identity",
        ));
    }
    let production = manifest.production_id.clone();
    Ok(StageArtifacts::new(production, manifest, descriptor, session))
}

/// Execute one caller-owned graph without publishing application artifacts.
///
/// The graph is validated against its manifest map and the observer runs
/// before execution. The returned outputs are owned copies, so callers can
/// download verification residuals or bind them into a later graph.
pub(crate) fn execute_graph<E, S>(
    graph: StageGraph,
    parameters: &ParamEnv,
    execution: &mut E,
    store: &mut S,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<StageOutputs, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let validated = validate_stage(&graph, parameters)?;
    before_execute(&validated)?;
    let (execution_nonce, _) = fresh_execution_session(&validated, store, rand::random())?;
    execution.evaluate(validated, graph.runtime_inputs, store, execution_nonce)
}

/// Dispatch one refresh PRF evaluation and finish the concrete stage.
///
/// The declaration supplies the canonical ordered labels. This function owns
/// exactly-once callback dispatch and common exact-key-set validation; a
/// [`NoiseRefresh`] implementation only consumes the validated outputs.
pub(crate) fn apply_refresh<R, F>(
    declaration: &RefreshDeclaration,
    refresh: &mut R,
    evaluate_prf: F,
) -> Result<R::Output, RefreshApplicationError<R::Error>>
where
    R: NoiseRefresh,
    F: FnOnce(
        &RefreshDeclaration,
        &[PrfLabel],
    ) -> Result<BTreeMap<PrfLabel, R::PrfOutput>, R::Error>,
{
    let labels = declaration.prf_labels();
    let mut unique = std::collections::BTreeSet::new();
    if labels.is_empty() || labels.iter().any(|label| !unique.insert(label.clone())) {
        return Err(RefreshApplicationError::OutputMismatch);
    }

    let outputs = evaluate_prf(declaration, labels).map_err(RefreshApplicationError::Refresh)?;
    if outputs.len() != labels.len() || labels.iter().any(|label| !outputs.contains_key(label)) {
        return Err(RefreshApplicationError::OutputMismatch);
    }

    refresh.finish_noise_refresh(declaration, outputs).map_err(RefreshApplicationError::Refresh)
}

pub(crate) fn domain_identity(base: [u8; 32], domain: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"mxx.exponent-lut/stage/v1");
    digest.update(domain);
    digest.update(base);
    digest.finalize().into()
}

/// Hash sampling keys are exactly 32 bytes; reject other keys before any
/// graph is built.
fn validate_hash_key(hash_key: &[u8]) -> Result<(), ApiError> {
    if hash_key.len() == 32 {
        Ok(())
    } else {
        Err(ApiError::input(format!("hash key must be 32 bytes, got {}", hash_key.len())))
    }
}

fn fresh_execution_session<S: SessionStore>(
    validated: &ValidatedGraph,
    store: &mut S,
    request_digest: [u8; 32],
) -> Result<([u8; 32], SessionAliasDescriptor), ApiError> {
    let alias_name = request_digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    let specification =
        spec_hash(&validated.source, &validated.bindings).map_err(ApiError::graph)?;
    let alias = SessionAliasDescriptor::new(
        format!("mxx.exponent-lut/{alias_name}"),
        validated.source.name(),
        specification,
        request_digest,
    );
    let nonce = store
        .resolve_session_nonce(&alias)
        .map_err(|error| ApiError::Runtime(ExecutionError::Artifact(error.to_string())))?;
    Ok((nonce, alias))
}

/// Build and execute preprocessing once, then publish two typed stage records
/// over that same finalized production: encryption preparation first and
/// online evaluation preparation second. `rhs` and `payload` are absent when the scheme
/// chooses RHS values and the secret only at encryption.
pub fn preprocess<E, S, P, T>(
    program: &ExponentLutProgram,
    parameters: &ParamEnv,
    rhs: &PlaintextInputs<RhsId, P>,
    payload: Option<&T::PayloadSecret>,
    hash_key: &[u8],
    mut build: T,
    execution: &mut E,
    store: &mut S,
    compose: impl FnOnce(
        &mut T,
        Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>,
    )
        -> Result<Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>, ApiError>,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<(StageArtifacts<T::EncryptDescriptor>, StageArtifacts<T::OnlineEvalDescriptor>), ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
    T: PreprocessBuild<P>,
{
    let started = std::time::Instant::now();
    validate_hash_key(hash_key)?;
    build.bind_inputs(program, parameters, rhs, payload, hash_key)?;
    let (encryption_identity, _) = build.artifact_identities();
    let bound_at = std::time::Instant::now();
    let outputs = {
        let (evaluator, families, environment) = build.lowering_state()?;
        lower_program(program, evaluator, families, environment).map_err(ApiError::lowering)?
    };
    let outputs = compose(&mut build, outputs)?;
    let lowered_at = std::time::Instant::now();
    let (graph, (encrypt_descriptor, online_eval_descriptor)) = build.build_graph(outputs)?;
    let built_at = std::time::Instant::now();

    // Validate and execute once. Both stage records share the same finalized
    // production and manifest.
    let validated = validate_stage(&graph, parameters)?;
    before_execute(&validated)?;
    let (execution_nonce, session) = fresh_execution_session(
        &validated,
        store,
        domain_identity(encryption_identity, b"preprocess"),
    )?;
    let (production, manifest) = stage_production(&validated, execution_nonce)?;
    let validated_at = std::time::Instant::now();
    let produced = execution.produce(validated, graph.runtime_inputs, store, execution_nonce)?;
    check_production(&produced, &production)?;
    let produced_at = std::time::Instant::now();
    let (encrypt_descriptor, online_eval_descriptor) =
        T::finalize_descriptors(&production, (encrypt_descriptor, online_eval_descriptor))?;
    tracing::info!(
        bind_ms = (bound_at - started).as_millis() as u64,
        lower_ms = (lowered_at - bound_at).as_millis() as u64,
        build_graph_ms = (built_at - lowered_at).as_millis() as u64,
        validate_ms = (validated_at - built_at).as_millis() as u64,
        produce_ms = (produced_at - validated_at).as_millis() as u64,
        finalize_ms = produced_at.elapsed().as_millis() as u64,
        "preprocess stage timing"
    );
    let encrypt_stage = StageArtifacts::new(
        production.clone(),
        manifest.clone(),
        encrypt_descriptor,
        session.clone(),
    );
    let online_eval_stage =
        StageArtifacts::new(production, manifest, online_eval_descriptor, session);
    Ok((encrypt_stage, online_eval_stage))
}

/// Build and execute the input-only encryption graph. `mask` is absent when
/// the scheme encrypts under the single secret `payload`.
pub fn encrypt<E, S, P, T>(
    inputs: &crate::program::EncryptionInputs,
    parameters: &ParamEnv,
    lhs: &PlaintextInputs<WireId, P>,
    preprocess: &StageArtifacts<T::PreprocessDescriptor>,
    payload: &T::PayloadSecret,
    mask: Option<&T::MaskSecret>,
    hash_key: &[u8],
    mut build: T,
    execution: &mut E,
    store: &mut S,
    compose: impl FnOnce(&mut T) -> Result<(), ApiError>,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<StageArtifacts<T::Descriptor>, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
    T: EncryptBuild<P>,
{
    validate_hash_key(hash_key)?;
    build.bind_inputs(inputs, parameters, lhs, payload, mask, hash_key)?;
    build.bind_artifacts(preprocess)?;
    compose(&mut build)?;
    let identity = build.artifact_identity();
    let (graph, descriptor) = build.build_graph()?;
    execute_artifacts(graph, parameters, descriptor, identity, execution, store, before_execute)
}

/// Lower the Program, append scheme-owned refresh operations, and execute the
/// resulting online evaluation graph. The returned outputs are owned copies.
pub fn online_eval<E, S, T>(
    program: &ExponentLutProgram,
    parameters: &ParamEnv,
    preprocess: &StageArtifacts<T::PreprocessDescriptor>,
    encryption: &StageArtifacts<T::EncryptDescriptor>,
    hash_key: &[u8],
    mut build: T,
    execution: &mut E,
    store: &mut S,
    compose: impl FnOnce(
        &mut T,
        Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>,
    )
        -> Result<Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>, ApiError>,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<StageOutputs, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    T: OnlineEvalBuild,
{
    validate_hash_key(hash_key)?;
    build.bind_artifacts(program, parameters, preprocess, encryption, hash_key)?;
    let outputs = {
        let (evaluator, families, environment) = build.lowering_state()?;
        lower_program(program, evaluator, families, environment).map_err(ApiError::lowering)?
    };
    let outputs = compose(&mut build, outputs)?;
    let graph = build.build_graph(outputs)?;
    execute_graph(graph, parameters, execution, store, before_execute)
}

/// Lower the Program once through the noise evaluator and produce its report.
pub fn noise_simulate<T>(
    program: &ExponentLutProgram,
    mut build: T,
    compose: impl FnOnce(
        &mut T,
        Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>,
    )
        -> Result<Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>, ApiError>,
) -> Result<T::Report, ApiError>
where
    T: NoiseBuild,
{
    let outputs = {
        let (evaluator, families, environment) = build.lowering_state()?;
        lower_program(program, evaluator, families, environment).map_err(ApiError::lowering)?
    };
    let outputs = compose(&mut build, outputs)?;
    build.build_report(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_backends::{MemoryArtifactStore, backend::poly::cpu_backend};
    use mxx_dsl::{DslContext, Int};

    fn execution() -> CpuExecution {
        CpuExecution::new(cpu_backend([]))
    }
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    fn observer_test_graph() -> StageGraph {
        let graph = DslContext::new("api-observer")
            .transferred_output("value", Int::constant(1))
            .expect("artifact output")
            .build()
            .expect("build graph");
        StageGraph::new(graph, BTreeMap::new(), BTreeMap::new())
    }

    #[test]
    fn stage_identity_is_domain_separated() {
        let base = [7; 32];
        assert_ne!(
            domain_identity(base, b"encrypt-preparation"),
            domain_identity(base, b"online-eval-preparation")
        );
    }

    #[test]
    fn stage_identity_is_sensitive_to_builder_identity() {
        assert_ne!(
            domain_identity([1; 32], b"online-eval-preparation"),
            domain_identity([2; 32], b"online-eval-preparation")
        );
    }

    #[test]
    fn online_eval_observer_runs_once_before_execution() {
        let mut execution = execution();
        let mut store = MemoryArtifactStore::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        execute_graph(
            observer_test_graph(),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            move |_| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .expect("constant graph executes");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn online_eval_observer_error_prevents_execution() {
        let mut execution = execution();
        let mut store = MemoryArtifactStore::default();
        let result = execute_graph(
            observer_test_graph(),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Err(ApiError::input("observer rejected graph")),
        );
        let error = match result {
            Ok(_) => panic!("observer error must stop execution"),
            Err(error) => error,
        };
        assert!(
            matches!(error, ApiError::Input { message } if message == "observer rejected graph")
        );
    }

    #[test]
    fn encrypt_observer_error_prevents_artifact_publication() {
        let mut execution = execution();
        let mut store = MemoryArtifactStore::default();
        let result = execute_artifacts(
            observer_test_graph(),
            &ParamEnv::default(),
            "descriptor".to_owned(),
            [7; 32],
            &mut execution,
            &mut store,
            |_| Err(ApiError::input("observer rejected artifact")),
        );
        let error = match result {
            Ok(_) => panic!("observer error must stop artifact publication"),
            Err(error) => error,
        };
        assert!(
            matches!(error, ApiError::Input { message } if message == "observer rejected artifact")
        );
    }

    #[test]
    fn finalized_named_stage_can_be_reloaded_with_a_reconstructed_descriptor() {
        let mut execution = execution();
        let mut store = MemoryArtifactStore::default();
        let stage = execute_artifacts(
            observer_test_graph(),
            &ParamEnv::default(),
            "descriptor".to_owned(),
            [11; 32],
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .expect("artifact graph executes");
        let session = stage.session().clone();
        let reloaded = reload_stage_artifacts(&mut store, session, "reconstructed".to_owned())
            .expect("finalized named stage reloads");

        assert_eq!(reloaded.production(), stage.production());
        assert_eq!(reloaded.manifest(), stage.manifest());
        assert_eq!(reloaded.descriptor(), "reconstructed");
    }
}
