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
use mxx_dsl::BuiltGraph;
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::{Manifest, ProductionId},
    encoding::spec_hash,
};
use mxx_primitives::poly::Poly;
use mxx_runtime::{
    Backend, ExecutionConfig, ExecutionError, ExecutionResult, RuntimeValue,
    SessionAliasDescriptor, SessionStore, execute_in_session_with_config,
};
use std::{collections::BTreeMap, error::Error};
use thiserror::Error as ThisError;

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
pub struct StageGraph<B: Backend> {
    pub(crate) graph: BuiltGraph,
    pub(crate) runtime_inputs: BTreeMap<String, RuntimeValue<B>>,
    pub(crate) manifests: BTreeMap<ProductionId, Manifest>,
}

impl<B: Backend> StageGraph<B> {
    pub fn new(
        graph: BuiltGraph,
        runtime_inputs: BTreeMap<String, RuntimeValue<B>>,
        manifests: BTreeMap<ProductionId, Manifest>,
    ) -> Self {
        Self { graph, runtime_inputs, manifests }
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
pub trait PreprocessBuild<B: Backend, P: Poly> {
    type Evaluator: ExponentLUTProgramLowering + crate::program::ProgramLoweringContext;
    type Families: FamilyExecution<Self::Evaluator>;
    type PayloadSecret;
    type EncryptDescriptor;
    type OnlineDescriptor;
    type Encrypt: EncryptBuild<
            B,
            P,
            PayloadSecret = Self::PayloadSecret,
            PreprocessDescriptor = Self::EncryptDescriptor,
        >;
    type Online: OnlineBuild<
            B,
            PreprocessDescriptor = Self::OnlineDescriptor,
            EncryptDescriptor = <Self::Encrypt as EncryptBuild<B, P>>::Descriptor,
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

    fn bind_inputs(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &ParamEnv,
        rhs: &PlaintextInputs<RhsId, P>,
        payload: &Self::PayloadSecret,
        hash_key: &[u8],
    ) -> Result<(), ApiError>;

    fn artifact_identities(&self) -> ([u8; 32], [u8; 32]);

    /// Finalize descriptors after execution has assigned the shared
    /// production nonce.  Concrete schemes may use this hook to attach
    /// dependency edges that cannot be known during graph construction.
    fn finalize_descriptors(
        production: &mxx_ir_core::artifact::ProductionId,
        descriptors: (Self::EncryptDescriptor, Self::OnlineDescriptor),
    ) -> Result<(Self::EncryptDescriptor, Self::OnlineDescriptor), ApiError>;

    fn build_graph(
        self,
        outputs: Vec<<Self::Evaluator as ExponentLUTProgramLowering>::Wire>,
    ) -> Result<(StageGraph<B>, (Self::EncryptDescriptor, Self::OnlineDescriptor)), ApiError>;
}

/// Builder contract for the input-only encryption graph.
pub trait EncryptBuild<B: Backend, P: Poly> {
    type PayloadSecret;
    type MaskSecret;
    type PreprocessDescriptor;
    type Descriptor;

    fn bind_inputs(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &ParamEnv,
        lhs: &PlaintextInputs<WireId, P>,
        payload: &Self::PayloadSecret,
        mask: &Self::MaskSecret,
        hash_key: &[u8],
    ) -> Result<(), ApiError>;

    fn bind_artifacts(
        &mut self,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
    ) -> Result<(), ApiError>;

    fn artifact_identity(&self) -> [u8; 32];

    fn build_graph(self) -> Result<(StageGraph<B>, Self::Descriptor), ApiError>;
}

/// Builder contract for online Program lowering and scheme refresh graph
/// construction.
pub trait OnlineBuild<B: Backend> {
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
    ) -> Result<StageGraph<B>, ApiError>;
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
/// second executor or bypass graph validation, observers, or execution
/// configuration.
pub(crate) fn execute_artifacts<B, S, D>(
    graph: StageGraph<B>,
    parameters: &ParamEnv,
    descriptor: D,
    identity: [u8; 32],
    backend: &mut B,
    store: &mut S,
    config: ExecutionConfig,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<StageArtifacts<D>, ApiError>
where
    B: Backend,
    S: SessionStore,
{
    let validated = graph
        .graph
        .validate_with_manifests(parameters, &graph.manifests)
        .map_err(ApiError::graph)?;
    before_execute(&validated)?;
    let (execution_nonce, session) = fresh_execution_session(&validated, store, identity)?;
    let result = execute_in_session_with_config(
        &validated,
        backend,
        graph.runtime_inputs,
        store,
        execution_nonce,
        config,
    )?;
    let production = result.production_id.ok_or_else(|| {
        ApiError::input("artifact execution did not produce a production identity")
    })?;
    let manifest = store
        .load_finalized_named_manifest(&session)
        .map_err(|error| ApiError::Runtime(ExecutionError::Artifact(error.to_string())))?;
    if manifest.production_id != production {
        return Err(ApiError::compatibility(
            "finalized named session returned a different production identity",
        ));
    }
    Ok(StageArtifacts::new(production, manifest, descriptor, session))
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
/// The graph is validated against its manifest map, the observer runs before
/// execution, and the supplied execution configuration is preserved. The
/// returned runtime result retains the leases needed by callers that
/// materialize verification residuals or other outputs.
pub(crate) fn execute_graph<B, S>(
    graph: StageGraph<B>,
    parameters: &ParamEnv,
    backend: &mut B,
    store: &mut S,
    config: ExecutionConfig,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<ExecutionResult<B>, ApiError>
where
    B: Backend,
    S: SessionStore,
{
    let validated = graph
        .graph
        .validate_with_manifests(parameters, &graph.manifests)
        .map_err(ApiError::graph)?;
    before_execute(&validated)?;
    let (execution_nonce, _) = fresh_execution_session(&validated, store, rand::random())?;
    Ok(execute_in_session_with_config(
        &validated,
        backend,
        graph.runtime_inputs,
        store,
        execution_nonce,
        config,
    )?)
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
/// online preparation second.
pub fn preprocess<B, S, P, T>(
    program: &ExponentLutProgram,
    parameters: &ParamEnv,
    rhs: &PlaintextInputs<RhsId, P>,
    payload: &T::PayloadSecret,
    hash_key: &[u8],
    mut build: T,
    backend: &mut B,
    store: &mut S,
    config: ExecutionConfig,
    compose: impl FnOnce(
        &mut T,
        Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>,
    )
        -> Result<Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>, ApiError>,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<(StageArtifacts<T::EncryptDescriptor>, StageArtifacts<T::OnlineDescriptor>), ApiError>
where
    B: Backend,
    S: SessionStore,
    P: Poly,
    T: PreprocessBuild<B, P>,
{
    build.bind_inputs(program, parameters, rhs, payload, hash_key)?;
    let (encryption_identity, _) = build.artifact_identities();
    let outputs = {
        let (evaluator, families, environment) = build.lowering_state()?;
        lower_program(program, evaluator, families, environment).map_err(ApiError::lowering)?
    };
    let outputs = compose(&mut build, outputs)?;
    let (graph, (encrypt_descriptor, online_descriptor)) = build.build_graph(outputs)?;

    // Validate and execute once. Both stage records share the same finalized
    // production and manifest.
    let validated = graph
        .graph
        .validate_with_manifests(parameters, &graph.manifests)
        .map_err(ApiError::graph)?;
    before_execute(&validated)?;
    let (execution_nonce, session) = fresh_execution_session(
        &validated,
        store,
        domain_identity(encryption_identity, b"preprocess"),
    )?;
    let result = execute_in_session_with_config(
        &validated,
        backend,
        graph.runtime_inputs,
        store,
        execution_nonce,
        config,
    )?;
    let (encrypt_descriptor, online_descriptor) = T::finalize_descriptors(
        result.production_id.as_ref().ok_or_else(|| {
            ApiError::input("preprocess execution did not produce an artifact identity")
        })?,
        (encrypt_descriptor, online_descriptor),
    )?;
    let production = result.production_id.ok_or_else(|| {
        ApiError::input("preprocess execution did not produce an artifact identity")
    })?;
    let manifest = store
        .load_finalized_named_manifest(&session)
        .map_err(|error| ApiError::Runtime(ExecutionError::Artifact(error.to_string())))?;
    if manifest.production_id != production {
        return Err(ApiError::compatibility(
            "finalized named session returned a different production identity",
        ));
    }
    let encrypt_stage = StageArtifacts::new(
        production.clone(),
        manifest.clone(),
        encrypt_descriptor,
        session.clone(),
    );
    let online_stage = StageArtifacts::new(production, manifest, online_descriptor, session);
    Ok((encrypt_stage, online_stage))
}

/// Build and execute the input-only encryption graph.
pub fn encrypt<B, S, P, T>(
    inputs: &crate::program::EncryptionInputs,
    parameters: &ParamEnv,
    lhs: &PlaintextInputs<WireId, P>,
    preprocess: &StageArtifacts<T::PreprocessDescriptor>,
    payload: &T::PayloadSecret,
    mask: &T::MaskSecret,
    hash_key: &[u8],
    mut build: T,
    backend: &mut B,
    store: &mut S,
    config: ExecutionConfig,
    compose: impl FnOnce(&mut T) -> Result<(), ApiError>,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<StageArtifacts<T::Descriptor>, ApiError>
where
    B: Backend,
    S: SessionStore,
    P: Poly,
    T: EncryptBuild<B, P>,
{
    build.bind_inputs(inputs, parameters, lhs, payload, mask, hash_key)?;
    build.bind_artifacts(preprocess)?;
    compose(&mut build)?;
    let identity = build.artifact_identity();
    let (graph, descriptor) = build.build_graph()?;
    execute_artifacts(
        graph,
        parameters,
        descriptor,
        identity,
        backend,
        store,
        config,
        before_execute,
    )
}

/// Lower the Program, append scheme-owned refresh operations, and execute the
/// resulting online graph. The returned runtime result retains all leases.
pub fn online_eval<B, S, T>(
    program: &ExponentLutProgram,
    parameters: &ParamEnv,
    preprocess: &StageArtifacts<T::PreprocessDescriptor>,
    encryption: &StageArtifacts<T::EncryptDescriptor>,
    hash_key: &[u8],
    mut build: T,
    backend: &mut B,
    store: &mut S,
    config: ExecutionConfig,
    compose: impl FnOnce(
        &mut T,
        Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>,
    )
        -> Result<Vec<<T::Evaluator as ExponentLUTProgramLowering>::Wire>, ApiError>,
    before_execute: impl FnOnce(&ValidatedGraph) -> Result<(), ApiError>,
) -> Result<ExecutionResult<B>, ApiError>
where
    B: Backend,
    S: SessionStore,
    T: OnlineBuild<B>,
{
    build.bind_artifacts(program, parameters, preprocess, encryption, hash_key)?;
    let outputs = {
        let (evaluator, families, environment) = build.lowering_state()?;
        lower_program(program, evaluator, families, environment).map_err(ApiError::lowering)?
    };
    let outputs = compose(&mut build, outputs)?;
    let graph = build.build_graph(outputs)?;
    execute_graph(graph, parameters, backend, store, config, before_execute)
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
    use mxx_dsl::{DslContext, Int};
    use mxx_runtime::{MemoryArtifactStore, backend::poly::cpu_backend};
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    fn observer_test_graph() -> StageGraph<mxx_runtime::backend::poly::CpuDcrtBackend> {
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
            domain_identity(base, b"online-preparation")
        );
    }

    #[test]
    fn stage_identity_is_sensitive_to_builder_identity() {
        assert_ne!(
            domain_identity([1; 32], b"online-preparation"),
            domain_identity([2; 32], b"online-preparation")
        );
    }

    #[test]
    fn online_observer_runs_once_before_execution() {
        let mut backend = cpu_backend([]);
        let mut store = MemoryArtifactStore::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        execute_graph(
            observer_test_graph(),
            &ParamEnv::default(),
            &mut backend,
            &mut store,
            ExecutionConfig::default(),
            move |_| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .expect("constant graph executes");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn online_observer_error_prevents_execution() {
        let mut backend = cpu_backend([]);
        let mut store = MemoryArtifactStore::default();
        let result = execute_graph(
            observer_test_graph(),
            &ParamEnv::default(),
            &mut backend,
            &mut store,
            ExecutionConfig::default(),
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
        let mut backend = cpu_backend([]);
        let mut store = MemoryArtifactStore::default();
        let result = execute_artifacts(
            observer_test_graph(),
            &ParamEnv::default(),
            "descriptor".to_owned(),
            [7; 32],
            &mut backend,
            &mut store,
            ExecutionConfig::default(),
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
        let mut backend = cpu_backend([]);
        let mut store = MemoryArtifactStore::default();
        let stage = execute_artifacts(
            observer_test_graph(),
            &ParamEnv::default(),
            "descriptor".to_owned(),
            [11; 32],
            &mut backend,
            &mut store,
            ExecutionConfig::default(),
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
