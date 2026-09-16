//! Application-facing, durable RNS-BGG execution.
//!
//! This layer owns phase composition, not cryptographic parameter selection.
//! Parameters must already have been audited by the caller. The DSL remains
//! independent of the filesystem, runtime, and device backend.
#![deny(missing_docs)]

use super::{
    program_setup::{GateRhsIdentity, ProgramDescriptor},
    *,
};
use crate::{
    noise_refresh::{
        pbc::{
            PbcActiveCellIndex, PbcCell, PbcEncodedPublicVector, PbcGeneratedKeyLayout,
            PbcParameters, PbcPublicVectorFamilyBinding, PbcRootSeed, PbcTrustedSelectorBits,
            generate_key_layout, generate_key_layout_from_layout_seed,
        },
        prf::{
            PreparedSparseLwrPrf, PreparedSparseLwrPrfDescriptor, PrfError, PrfSelectionInputs,
            SparseLwrPrfProfile, SparseLwrPrfProgram as NativeSparseLwrPrfProgram,
        },
        simulation::AverageCaseConfig,
    },
    program::{ExponentLutProgramId, Gate, ProgramBindings, ProgramInputId, ProgramStep, WireId},
    readout::ValueReadout,
    rns_bgg::noise::{
        RnsBggGateNoiseMaterial, RnsBggNoiseRhs, RnsBggNoiseValue, RnsBggTerminalNoise,
        stable_source_id,
    },
    scheme::{EncodingBindings, PreparedProgram},
};
use mxx_ir_core::{
    ValidatedGraph,
    artifact::{ArtifactCatalog, Manifest, export_validated_manifest},
};
use mxx_runtime::{
    Backend, ExecutionConfig, ExecutionResult, FileArtifactStore, FileArtifactStoreRouter,
    RuntimeValue,
    artifact::{ArtifactKey, ArtifactStore},
    transcript::SamplingMode,
};
use num_traits::Signed;
use serde::de::DeserializeOwned;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    cell::RefCell,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

/// A failed phase never returns an artifact reference claiming completion.
#[derive(Debug, Error)]
#[error("Exponent-LUT: {0}")]
pub struct ExecutionError(String);
impl ExecutionError {
    pub(crate) fn invalid(message: &str) -> Self {
        Self(message.into())
    }
}

fn failed(error: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(error.to_string())
}
type Result<T> = std::result::Result<T, ExecutionError>;

fn notify_graph_observer(
    observer: &mut Option<Box<dyn FnMut(&str, &ValidatedGraph) -> std::result::Result<(), String>>>,
    stage: &str,
    graph: &ValidatedGraph,
) -> Result<()> {
    if let Some(observer) = observer.as_mut() {
        observer(stage, graph).map_err(failed)?;
    }
    Ok(())
}

/// Backend and execution policy shared by all phases. Device selection remains
/// explicit at construction; there is no CPU fallback or implicit security audit.
pub struct RuntimeConfiguration<B: Backend> {
    backend: B,
    execution: ExecutionConfig,
    working_directory: ArtifactWorkingDirectory,
    graph_observer:
        Option<Box<dyn FnMut(&str, &ValidatedGraph) -> std::result::Result<(), String>>>,
    #[cfg(test)]
    backend_activity: std::sync::Arc<AtomicUsize>,
    #[cfg(test)]
    panic_on_backend_activity: bool,
}
#[derive(Clone, Debug)]
/// Selects where finalized artifact stores live after a facade operation.
pub enum ArtifactWorkingDirectory {
    /// Keep the output directory until the caller removes it.
    Persistent(PathBuf),
    /// Use an owned temporary directory whose lifetime follows the result.
    Temporary,
}
impl<B: Backend> RuntimeConfiguration<B> {
    /// Creates a facade runtime with the supplied backend and execution policy.
    /// No graph is executed or security estimate is performed by construction.
    pub fn new(backend: B, execution: ExecutionConfig) -> Self {
        Self {
            backend,
            execution,
            working_directory: ArtifactWorkingDirectory::Temporary,
            graph_observer: None,
            #[cfg(test)]
            backend_activity: std::sync::Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            panic_on_backend_activity: false,
        }
    }
    /// Selects persistent storage for intermediate runtime artifacts.
    pub fn with_working_directory(mut self, directory: ArtifactWorkingDirectory) -> Self {
        self.working_directory = directory;
        self
    }

    /// Installs an optional advanced observer for the exact validated graphs
    /// that the facade executes. The observer is called immediately before
    /// execution, once for each stage, and is omitted entirely when unset.
    /// This hook is intended for benchmark instrumentation; it does not alter
    /// graph construction or execution semantics.
    pub fn with_graph_observer<F>(mut self, observer: F) -> Self
    where
        F: FnMut(&str, &ValidatedGraph) -> std::result::Result<(), String> + 'static,
    {
        self.graph_observer = Some(Box::new(observer));
        self
    }
    #[cfg(test)]
    fn with_backend_activity_panic(mut self) -> Self {
        self.panic_on_backend_activity = true;
        self
    }
}

/// The normal application entry point. Independent instances can use different
/// backends while the artifact contract stays scheme/parameter specific.
pub struct ExponentLut<B: Backend> {
    scheme: RnsBggScheme,
    runtime: RefCell<RuntimeConfiguration<B>>,
    #[cfg(test)]
    graph_identity: Option<TestGraphIdentity>,
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct TestGraphIdentity {
    setup: [u8; 32],
    seed: [u8; 32],
    preprocess_nonce: [u8; 32],
    encrypt_nonce: [u8; 32],
}

/// Exact validated graphs built by the facade for benchmark observation.
/// Planning does not create artifact directories or execute a backend.
pub struct FacadeBenchmarkGraphs {
    /// The setup graph that produces public preprocessing artifacts.
    pub preprocess: ValidatedGraph,
    /// The mask-dependent encryption graph.
    pub encrypt: ValidatedGraph,
    /// The secret-free online evaluation graph.
    pub online_eval: ValidatedGraph,
}

/// Stable secret coefficients, sampled once and never serialized by this API.
/// Rebuilding DSL graphs from this material does not resample a different key.
pub struct PayloadSecret {
    coefficients: Arc<Vec<i8>>,
    context: [u8; 32],
}
/// Secret mask coefficients used only by the encryption phase.
pub struct MaskSecret {
    coefficients: Arc<Vec<i8>>,
    context: [u8; 32],
}
/// NAND uses independent payload/mask keys. PRF setup can deliberately share the
/// same mathematical key without relying on replayed random sampler nodes.
pub struct EncodingSecrets {
    payload: PayloadSecret,
    mask: MaskSecret,
}
impl EncodingSecrets {
    /// Returns the payload secret for caller-owned correctness checks.
    pub fn payload(&self) -> &PayloadSecret {
        &self.payload
    }
    /// Returns the mask secret for the encryption phase.
    pub fn mask(&self) -> &MaskSecret {
        &self.mask
    }
}

/// Application-validated plaintext bindings. Circuit compilers construct this
/// value; the caller never supplies gate IDs, levels, or sampling identities.
pub struct FixedRhs {
    pub(crate) program: ExponentLutProgramId,
    pub(crate) inputs: BTreeMap<ProgramInputId, RnsBggProgramInput>,
    pub(crate) rhs: BTreeMap<crate::program::RhsId, BigInt>,
    pub(crate) sparse_lwr: Option<SparseLwrFixedRhs>,
}

/// Internal fixed-RHS payload for the sparse-LWR facade. Public callers only
/// obtain it from [`SparseLwrPrfProgram::fixed_rhs`].
#[derive(Clone)]
pub(crate) struct SparseLwrFixedRhs {
    pub(crate) public_vector: Vec<usize>,
    pub(crate) selector: SparseLwrPrfSelectorKey,
    pub(crate) program: NativeSparseLwrPrfProgram,
}

/// Opaque selector key for one validated sparse-LWR PBC layout and schedule.
/// The private matching schedule is retained here and is never serialized or
/// exposed as arbitrary bucket/slot indices.
#[derive(Clone)]
pub struct SparseLwrPrfSelectorKey {
    generated: PbcGeneratedKeyLayout,
}

impl SparseLwrPrfSelectorKey {
    /// Generates a validated selector key from the public PBC parameters and
    /// sparse support. The support is fixed before layout sampling, as required
    /// by the PBC key-generation contract.
    fn generate(
        parameters: &PbcParameters,
        root_seed: PbcRootSeed,
        support: &[usize],
    ) -> std::result::Result<Self, PrfError> {
        Ok(Self {
            generated: generate_key_layout(parameters, root_seed, support)
                .map_err(|_| PrfError::InvalidSparseLwrBlock)?,
        })
    }

    fn layout(&self) -> &crate::noise_refresh::pbc::PbcPublicLayout {
        &self.generated.public_layout
    }
}

/// Application-facing sparse-LWR PRF compiler. It owns the native accumulation
/// Program and accepts public residues plus an opaque selector key; PBC and
/// fixed-product bindings remain implementation details.
#[derive(Clone, Debug)]
pub struct SparseLwrPrfProgram {
    native: NativeSparseLwrPrfProgram,
    selector_capacity: usize,
}

/// The readout contract shared by graph construction and noise auditing.
/// `input_level` is the accumulation terminal; the actual readout is evaluated
/// at the next level and uses the canonical table's centered coefficient bound.
struct SparseLwrReadoutContract {
    output_level: usize,
    digit_scale: BigUint,
    digit_bound: BigUint,
    target_bound: BigUint,
    branches: usize,
}

fn sparse_lwr_readout_contract(
    parameters: &RnsBggParameters,
    readout: &ValueReadout,
    output_modulus: usize,
    input_level: usize,
) -> std::result::Result<SparseLwrReadoutContract, RnsBggError> {
    let output_level =
        input_level.checked_add(1).ok_or(RnsBggError::Budget("sparse-LWR readout level"))?;
    let modulus = parameters.level(output_level)?.modulus().as_ref().clone();
    let digit_scale = &modulus / BigUint::from(2usize * output_modulus);
    if digit_scale.is_zero() {
        return Err(RnsBggError::Parameters("sparse-LWR digit scale is zero"));
    }
    let weight = [vec![BigInt::from(digit_scale.clone())], vec![]];
    let (_, digit_bound) = readout
        .weighted_centered_table(&weight, &BigInt::from(modulus.clone()))
        .map_err(|_| RnsBggError::Lut)?;
    let target_bound = &digit_scale * &digit_bound;
    Ok(SparseLwrReadoutContract {
        output_level,
        digit_scale,
        digit_bound,
        target_bound,
        branches: readout.layout.width(),
    })
}

impl SparseLwrPrfProgram {
    /// Creates the sparse-LWR facade for one accepted profile, selector
    /// capacity, and accumulation block count. This only validates structure;
    /// it does not sample a selector, secret, or artifact.
    pub fn new(
        profile: SparseLwrPrfProfile,
        selector_capacity: usize,
        block_count: usize,
    ) -> std::result::Result<Self, PrfError> {
        if selector_capacity == 0 {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(Self {
            native: NativeSparseLwrPrfProgram::new(profile, selector_capacity, block_count)?,
            selector_capacity,
        })
    }

    /// Generates an opaque selector key from public support and a caller-owned
    /// root seed. The support and generated layout are authenticated by later
    /// fixed-RHS validation.
    pub fn selector_key(
        &self,
        universe_size: usize,
        support: &[usize],
        root_seed: [u8; 32],
    ) -> std::result::Result<SparseLwrPrfSelectorKey, PrfError> {
        let parameters = PbcParameters::paper_evaluation(universe_size, support.len());
        if parameters.bucket_count != self.native.block_count ||
            parameters.bucket_width_limit.is_some_and(|limit| limit > self.selector_capacity)
        {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let key = SparseLwrPrfSelectorKey::generate(&parameters, PbcRootSeed(root_seed), support)?;
        if key.layout().bucket_width > self.selector_capacity {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(key)
    }

    /// Reconstructs a selector key from a persisted accepted layout seed.
    /// Reports store the accepted attempt seed, not the root seed consumed by
    /// ordinary key generation, so replaying through [`Self::selector_key`]
    /// would derive a different public layout.  The resulting key remains
    /// opaque; callers can authenticate its public layout identity separately.
    /// Reconstructs a selector from an accepted layout seed and attempt index.
    /// This is the deterministic replay counterpart of [`Self::selector_key`].
    pub fn selector_key_from_layout_seed(
        &self,
        universe_size: usize,
        support: &[usize],
        layout_seed: [u8; 32],
        accepted_attempt: u32,
    ) -> std::result::Result<SparseLwrPrfSelectorKey, PrfError> {
        let parameters = PbcParameters::paper_evaluation(universe_size, support.len());
        if parameters.bucket_count != self.native.block_count {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let generated = generate_key_layout_from_layout_seed(
            &parameters,
            crate::noise_refresh::pbc::PbcLayoutSeed(layout_seed),
            accepted_attempt,
            support,
        )
        .map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        if generated.public_layout.bucket_width > self.selector_capacity {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(SparseLwrPrfSelectorKey { generated })
    }

    /// Binds public PRF input residues to the opaque selector schedule. Values
    /// are residues, not output digits, so they are checked modulo the native
    /// exponent modulus `Q`, while `P` only describes the output alphabet.
    /// Binds public input residues to the selector schedule. Values are checked
    /// modulo the profile's input modulus `Q`; output modulus `P` is not used
    /// for this validation.
    pub fn fixed_rhs(
        &self,
        public_vector: &[usize],
        selector: &SparseLwrPrfSelectorKey,
    ) -> std::result::Result<FixedRhs, PrfError> {
        let profile = self.native.profile();
        let layout = selector.layout();
        let modulus = profile.q()?;
        if layout.parameters.bucket_count != self.native.block_count ||
            layout.bucket_width > self.selector_capacity ||
            public_vector.len() != layout.parameters.universe_size ||
            public_vector.iter().any(|value| *value >= modulus)
        {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(FixedRhs {
            program: self.native.id(),
            inputs: BTreeMap::new(),
            rhs: BTreeMap::new(),
            sparse_lwr: Some(SparseLwrFixedRhs {
                public_vector: public_vector.to_vec(),
                selector: SparseLwrPrfSelectorKey { generated: selector.generated.clone() },
                program: self.native.clone(),
            }),
        })
    }

    /// Samples the facade's caller-owned encoding secrets. The sparse PRF
    /// contract intentionally shares the mathematical payload and mask key.
    pub fn sample_secrets<B: Backend>(&self, engine: &ExponentLut<B>) -> EncodingSecrets {
        engine.sample_secrets(true)
    }

    /// Returns the exact native accumulation Program used by production.
    pub fn program(&self) -> &ExponentLutProgram {
        &self.native.accumulation
    }

    /// Selects this PRF's sole output from a completed evaluation without
    /// exposing the internal [`WireId`] used by the DSL program.
    /// Selects this PRF's sole output from facade results and checks its Program identity.
    pub fn output<'a>(&self, outputs: &'a EncodedOutputs) -> Result<&'a EncodedOutput> {
        outputs.for_program(self.native.id())
    }

    /// Returns the accepted sparse-LWR profile.
    pub fn profile(&self) -> SparseLwrPrfProfile {
        self.native.profile()
    }

    /// Returns the maximum active selector width accepted by this facade.
    pub fn selector_capacity(&self) -> usize {
        self.selector_capacity
    }

    /// Returns the number of accumulation blocks and terminal level steps.
    pub fn block_count(&self) -> usize {
        self.native.block_count()
    }

    /// Noise evidence for the exact native accumulation Program and selector
    /// schedule used by the facade. This keeps binding construction beside
    /// the production builder so tests cannot silently audit another graph.
    /// Audits the exact accumulation and canonical readout chain before execution.
    /// The report distinguishes the pre-readout terminal level from its output level.
    pub fn audit_noise(
        &self,
        parameters: &RnsBggParameters,
        fixed: &FixedRhs,
        average_case: bool,
        failure_exponent: u32,
    ) -> std::result::Result<SparseLwrPrfNoiseAudit, RnsBggError> {
        let fixed = fixed.sparse_lwr.as_ref().ok_or(RnsBggError::Context)?;
        if fixed.program.id() != self.native.id() {
            return Err(RnsBggError::Context);
        }
        let active =
            PbcActiveCellIndex::build(fixed.selector.layout()).map_err(|_| RnsBggError::Context)?;
        let counts = active.bucket_active_widths().collect::<Vec<_>>();
        let input = RnsBggNoiseValue {
            level: 0,
            bound: RnsBggNoiseBound {
                error: parameters.encoding_error_bound().clone(),
                payload_l1: BigUint::one(),
            },
            payload_l2_squared: BigUint::one(),
            moments: Some(crate::noise_utils::NoiseMoments {
                mean_numerator: BigInt::zero(),
                mean_denominator: BigUint::one(),
                mean_radius: crate::noise_utils::NonnegativeRational::zero(),
                variance: parameters.encoding_error_variance(),
            }),
            public: false,
            range: None,
            one_hot: false,
        };
        let mut rhs_values = BTreeMap::new();
        let mut gate_helpers = BTreeMap::new();
        for (step, operation) in self.native.accumulation.steps().iter().enumerate() {
            let ProgramStep::Gate(Gate::MulPrivate { rhs, .. }) = operation else {
                return Err(RnsBggError::Parameters(
                    "sparse-LWR Program must contain private products only",
                ));
            };
            let count = counts.get(step).copied().ok_or(RnsBggError::Context)?;
            let actions = (0..count)
                .map(|index| {
                    crate::rns_bgg::noise::RnsBggNoiseAction::new(
                        stable_source_id(&("sparse-lwr-rhs", step, rhs, index)),
                        crate::noise_utils::RightActionNoiseGain::unit(),
                    )
                })
                .collect::<Vec<_>>();
            let (material, bound) =
                RnsBggGateNoiseMaterial::from_one_hot_labels(step, None, &actions)?;
            rhs_values.insert(
                *rhs,
                RnsBggNoiseRhs {
                    payload_l1: bound.payload_l1.clone(),
                    payload_l2_squared: bound.payload_l2_squared.clone(),
                    one_hot: None,
                },
            );
            gate_helpers.insert(ProgramGateInstance::Gate { step }, material);
        }
        if rhs_values.len() != self.native.accumulation.rhs_inputs().len() {
            return Err(RnsBggError::Context);
        }
        let bindings = ProgramBindings::new()
            .inputs([(self.native.initial_input(), input)])
            .rhs_values(rhs_values)
            .gate_helpers(gate_helpers);
        let ranges = ProgramFamilyBindings::new();
        let output = *self.native.accumulation.outputs().first().ok_or(RnsBggError::Context)?;
        let terminal_bounds = BTreeMap::from([(output, BigUint::one())]);
        // Split the requested failure budget between the accumulation audit
        // and the separate readout tail: each receives k+1 bits, so their two
        // failure probabilities sum to at most 2^-k.
        let split_failure_exponent = average_case
            .then(|| {
                failure_exponent
                    .checked_add(1)
                    .ok_or(RnsBggError::Parameters("average failure exponent overflow"))
            })
            .transpose()?;
        let average = split_failure_exponent.map(|failure_exponent| AverageCaseConfig {
            failure_exponent,
            allow_average_acceptance: true,
            ..Default::default()
        });
        let trace = parameters.audit_program(
            &self.native.accumulation,
            &bindings,
            &ranges,
            &terminal_bounds,
            average.as_ref(),
        )?;
        let terminal = trace.wires.get(&output).ok_or(RnsBggError::Context)?;
        let readout = sparse_lwr_readout_contract(
            parameters,
            &self.native.readout,
            self.native
                .profile()
                .p()
                .map_err(|_| RnsBggError::Parameters("invalid sparse-LWR output modulus"))?,
            terminal.level,
        )?;
        let terminal_moments =
            parameters.terminal_noise_moments(terminal.level, [(terminal, readout.branches)])?;
        let worst_terminal_bound = parameters.terminal_noise_bound(
            terminal.level,
            [(&terminal.bound, readout.branches, &BigUint::one())],
            readout.target_bound.clone(),
        )?;
        let readout_events = BigUint::from(readout.branches) *
            parameters.level(readout.output_level)?.ring_dimension() *
            2u8 *
            parameters.component_columns(readout.output_level)?;
        let total_event_count = &trace.joint_event_count + &readout_events;
        let readout_average = split_failure_exponent.map(|failure_exponent| AverageCaseConfig {
            failure_exponent,
            allow_average_acceptance: true,
            ..Default::default()
        });
        let z_squared = readout_average
            .as_ref()
            .map(|config| config.z_squared_for_events(&readout_events))
            .transpose()
            .map_err(|_| RnsBggError::Parameters("invalid average-case event budget"))?;
        let selected_error = if let Some(z_squared) = z_squared {
            terminal_moments.tail_bound(&z_squared)
        } else {
            worst_terminal_bound.error.clone()
        };
        let final_terminal_bound = RnsBggTerminalNoise {
            error: selected_error.clone(),
            coefficient_bound: readout.target_bound.clone(),
            branches: readout.branches,
        };
        let strict_terminal_fit =
            parameters.terminal_noise_fits(readout.output_level, &final_terminal_bound)?;
        let worst_case_terminal_fit =
            parameters.terminal_noise_fits(readout.output_level, &worst_terminal_bound)?;
        Ok(SparseLwrPrfNoiseAudit {
            terminal_level: terminal.level,
            output_level: readout.output_level,
            terminal_moments,
            terminal_bound: worst_terminal_bound,
            final_terminal_bound,
            selected_error,
            digit_scale: readout.digit_scale,
            digit_bound: readout.digit_bound,
            target_bound: readout.target_bound,
            branches: readout.branches,
            strict_terminal_fit,
            worst_case_terminal_fit,
            accumulation_failure_exponent: split_failure_exponent,
            readout_failure_exponent: split_failure_exponent,
            joint_event_count: total_event_count,
        })
    }
}

/// Audited terminal noise and production PRF readout spacing for one facade.
pub struct SparseLwrPrfNoiseAudit {
    /// Accumulation terminal level before the readout modulus switch.
    pub terminal_level: usize,
    /// Readout output level, exactly one level above the accumulation terminal.
    pub output_level: usize,
    /// Average-case moments of the audited accumulation trace.
    pub terminal_moments: crate::noise_utils::NoiseMoments,
    /// Deterministic pre-readout bound produced by the terminal transfer.
    pub terminal_bound: RnsBggTerminalNoise,
    /// Final output-level bound, using the selected average or worst error.
    pub final_terminal_bound: RnsBggTerminalNoise,
    /// Error selected by the requested worst- or average-case audit.
    pub selected_error: BigUint,
    /// Native output digit spacing `q/(2P)`.
    pub digit_scale: BigUint,
    /// Canonical centered readout digit bound.
    pub digit_bound: BigUint,
    /// Weighted target bound used by the production readout.
    pub target_bound: BigUint,
    /// Canonical readout branch multiplicity.
    pub branches: usize,
    /// Whether the selected error fits strictly below the output threshold.
    pub strict_terminal_fit: bool,
    /// Whether the deterministic audited bound fits the output threshold.
    pub worst_case_terminal_fit: bool,
    /// Each half of an average-case union bound uses requested k+1 bits.
    pub accumulation_failure_exponent: Option<u32>,
    pub readout_failure_exponent: Option<u32>,
    pub joint_event_count: BigUint,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct Identity {
    version: u32,
    parameter: RnsBggArtifactBinding,
    program: ExponentLutProgramId,
    setup: [u8; 32],
}

/// Names the application-owned artifact namespace represented by a token.
/// Keeping this in the authenticated metadata prevents an encryption token
/// from being accepted as an evaluation or input token after a cross-mix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactNamespace {
    /// Public preprocessing keys, helper banks, and prepared evaluation data.
    Encryption,
    /// Prepared Program artifacts consumed by online evaluation.
    Evaluation,
    /// Encrypted input contributions produced from an encryption namespace.
    Input,
}

/// A reference contains only a path and independently trusted identity digests.
/// Reading expected identities from the metadata itself would not authenticate it.
struct ArtifactReference {
    directory: PathBuf,
    completion: PathBuf,
    production: ProductionId,
    identity: Identity,
    metadata_digest: [u8; 32],
    manifest_digest: [u8; 32],
    namespace: ArtifactNamespace,
}
/// Opaque, small trusted anchor for reopening one durable artifact namespace.
/// It contains identities and digests only; it never contains artifact payloads
/// or either RNS-BGG secret.
#[derive(Clone, Serialize, Deserialize)]
/// Small, payload-free trusted anchor for reopening a preprocessing namespace.
pub struct ArtifactReferenceToken {
    directory: PathBuf,
    completion: PathBuf,
    production: ProductionId,
    identity: Identity,
    metadata_digest: [u8; 32],
    manifest_digest: [u8; 32],
    namespace: ArtifactNamespace,
}
/// Authenticated preprocessing artifacts used to encrypt inputs.
pub struct EncryptionArtifacts(ArtifactReference);
/// Authenticated prepared evaluation artifacts used by online evaluation.
pub struct EvaluationArtifacts(ArtifactReference);
/// Authenticated encrypted input artifacts used by online evaluation.
pub struct InputArtifacts(ArtifactReference);
/// The two preprocessing namespaces returned only after both are finalized.
pub struct PreprocessedProgram {
    /// Public setup artifacts for the encryption phase.
    pub encryption: EncryptionArtifacts,
    /// Prepared evaluation artifacts for the online phase.
    pub evaluation: EvaluationArtifacts,
}

impl ArtifactReference {
    fn reopen(
        token: ArtifactReferenceToken,
        expected_namespace: ArtifactNamespace,
        scheme: &RnsBggScheme,
    ) -> Result<Self> {
        let reference = Self::from_token(token);
        if reference.namespace != expected_namespace {
            return Err(failed("artifact token namespace mismatch"));
        }
        reference.open::<serde_json::Value>(expected_namespace, scheme)?;
        Ok(reference)
    }

    fn token(&self) -> ArtifactReferenceToken {
        ArtifactReferenceToken {
            directory: self.directory.clone(),
            completion: self.completion.clone(),
            production: self.production.clone(),
            identity: self.identity.clone(),
            metadata_digest: self.metadata_digest,
            manifest_digest: self.manifest_digest,
            namespace: self.namespace,
        }
    }

    fn from_token(token: ArtifactReferenceToken) -> Self {
        Self {
            directory: token.directory,
            completion: token.completion,
            production: token.production,
            identity: token.identity,
            metadata_digest: token.metadata_digest,
            manifest_digest: token.manifest_digest,
            namespace: token.namespace,
        }
    }
}
impl EncryptionArtifacts {
    /// Returns a small payload-free token for later encryption-namespace reopen.
    pub fn token(&self) -> ArtifactReferenceToken {
        self.0.token()
    }
}
impl EvaluationArtifacts {
    /// Returns a small payload-free token for later evaluation-namespace reopen.
    pub fn token(&self) -> ArtifactReferenceToken {
        self.0.token()
    }
}
impl InputArtifacts {
    /// Returns a small payload-free token for later input-namespace reopen.
    pub fn token(&self) -> ArtifactReferenceToken {
        self.0.token()
    }
}

type KeySchema = <RnsBggPublicKey as GraphValue>::Schema;
type RowSchema = <RnsBggRow as GraphValue>::Schema;
type PublicSchema = <Option<Mat> as GraphValue>::Schema;
type SparsePreparedDescriptor = PreparedSparseLwrPrfDescriptor<
    <RnsBggScheme as ExponentLUTFixedRhsScheme>::FixedProductBinding,
    <<RnsBggScheme as ExponentLUTFixedRhsScheme>::PreparedFixedProduct as GraphValue>::Schema,
>;

struct PreprocessPlan {
    validated: ValidatedGraph,
    encrypt: ValidatedGraph,
    online: ValidatedGraph,
    identity: Identity,
    setup_seed: [u8; 32],
    nonce: [u8; 32],
    encryption: EncryptionDescriptor,
    evaluation: EvaluationDescriptor,
}
#[derive(Clone, Serialize, Deserialize)]
struct HelperSchema {
    binding: RnsBggLutBinding,
    id: [u8; 32],
    level: usize,
    context: [u8; 32],
    rows: Vec<(usize, RowSchema)>,
}
impl HelperSchema {
    fn of(value: &RnsBggOnlineLutHelpers) -> Self {
        let schema = value.schema();
        Self {
            binding: schema.binding,
            id: schema.masks.id,
            level: schema.masks.level,
            context: schema.masks.context,
            rows: schema.masks.rows.iter().map(|(id, row)| (*id, row.clone())).collect(),
        }
    }
    fn import(self, production: ProductionId, prefix: &str) -> Result<RnsBggOnlineLutHelpers> {
        let count = self.rows.len();
        let rows = self.rows.into_iter().collect::<BTreeMap<_, _>>();
        if rows.len() != count {
            return Err(failed("duplicate online mask branch"));
        }
        Ok(RnsBggOnlineLutHelpers {
            binding: self.binding,
            masks: RnsBggOnlineMasks {
                id: self.id,
                level: self.level,
                context: self.context,
                rows: Arc::new(rows),
            },
        }
        .artifact_input(production, prefix, ArtifactConfidentiality::Public)?)
    }
}

#[derive(Serialize, Deserialize)]
struct Metadata<D> {
    identity: Identity,
    production: ProductionId,
    namespace: ArtifactNamespace,
    descriptor: D,
}
#[derive(Clone, Serialize, Deserialize)]
struct EncryptionInput {
    input: ProgramInputId,
    key: KeySchema,
    payload: RowSchema,
    public: PublicSchema,
}
#[derive(Clone, Serialize, Deserialize)]
struct EncryptionBank {
    gate: ProgramGateInstance,
    binding: RnsBggLutBinding,
    bank: [u8; 32],
    level: usize,
    branches: Vec<MaskBranch<<RnsBggMatrix as GraphValue>::Schema>>,
}
#[derive(Clone, Serialize, Deserialize)]
struct EncryptionDescriptor {
    inputs: Vec<EncryptionInput>,
    banks: Vec<EncryptionBank>,
    readout: Option<EncryptionBank>,
    payload: RowSchema,
}
#[derive(Clone, Serialize, Deserialize)]
struct EvaluationDescriptor {
    prepared: Option<ProgramDescriptor>,
    sparse_prepared: Option<SparsePreparedDescriptor>,
    sparse_output: Option<<RnsBggPublicProgramValue as GraphValue>::Schema>,
    sparse_readout: Option<RnsBggPreparedReadoutDescriptor>,
    sparse_readout_binding: Option<RnsBggLutBinding>,
    sparse_readout_id: Option<[u8; 32]>,
    sparse_readout_input_id: Option<[u8; 32]>,
    sparse_profile: Option<SparseLwrPrfProfile>,
    sparse_selector_capacity: Option<usize>,
    sparse_block_count: Option<usize>,
    rhs: Vec<GateRhsIdentity>,
}
#[derive(Clone, Serialize, Deserialize)]
struct InputDescriptor {
    /// Binds an encrypted input to the exact preprocessing production, not just
    /// another independently sampled setup of the same Program.
    preprocessing: ProductionId,
    inputs: Vec<(ProgramInputId, <RnsBggEncoding as GraphValue>::Schema, PublicSchema)>,
    helpers: Vec<(ProgramGateInstance, HelperSchema)>,
    readout: Option<HelperSchema>,
}

/// Final encodings and their public keys; no payload/mask secret is retained.
pub struct EncodedOutputs {
    program: ExponentLutProgramId,
    values: BTreeMap<WireId, EncodedOutput>,
}
struct FinalizedOutputStore {
    directory: PathBuf,
    remove_on_drop: bool,
    manifest_digest: [u8; 32],
}
impl Drop for FinalizedOutputStore {
    fn drop(&mut self) {
        if self.remove_on_drop {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }
}
/// One finalized output encoding and its authenticated persistent/temporary store.
pub struct EncodedOutput {
    identity: Identity,
    wire: WireId,
    level: usize,
    production: ProductionId,
    store: Arc<FinalizedOutputStore>,
    prf: Option<PrfOutputMetadata>,
    metadata: PathBuf,
    metadata_digest: [u8; 32],
    schema_digest: [u8; 32],
    completion: PathBuf,
    completion_store_count: usize,
    completion_manifest_digest: [u8; 32],
    completion_writable_stores: Vec<usize>,
}
#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct PrfOutputMetadata {
    profile: SparseLwrPrfProfile,
    digit_scale: BigUint,
}
#[derive(Clone, Serialize, Deserialize)]
struct OutputMetadata {
    version: u32,
    identity: Identity,
    production: ProductionId,
    program: ExponentLutProgramId,
    wire: WireId,
    level: usize,
    manifest_digest: [u8; 32],
    schema_digest: [u8; 32],
    completion: PathBuf,
    completion_store_count: usize,
    completion_manifest_digest: [u8; 32],
    completion_writable_stores: Vec<usize>,
    prf: Option<PrfOutputMetadata>,
}
/// Opaque trusted anchor for reopening one persistent online output.
/// It contains no encoded payload and is valid only while the finalized store
/// and its manifest remain unchanged.
#[derive(Clone, Serialize, Deserialize)]
pub struct OutputReferenceToken {
    directory: PathBuf,
    metadata: PathBuf,
    metadata_digest: [u8; 32],
    production: ProductionId,
    identity: Identity,
    wire: WireId,
    level: usize,
    manifest_digest: [u8; 32],
    schema_digest: [u8; 32],
    completion: PathBuf,
    completion_store_count: usize,
    completion_manifest_digest: [u8; 32],
    completion_writable_stores: Vec<usize>,
    prf: Option<PrfOutputMetadata>,
}
impl EncodedOutput {
    /// Returns a payload-free token for persistent output reopen.
    pub fn token(&self) -> OutputReferenceToken {
        OutputReferenceToken {
            directory: self.store.directory.clone(),
            metadata: self.metadata.clone(),
            metadata_digest: self.metadata_digest,
            production: self.production.clone(),
            identity: self.identity.clone(),
            wire: self.wire,
            level: self.level,
            manifest_digest: self.store.manifest_digest,
            schema_digest: self.schema_digest,
            completion: self.completion.clone(),
            completion_store_count: self.completion_store_count,
            completion_manifest_digest: self.completion_manifest_digest,
            completion_writable_stores: self.completion_writable_stores.clone(),
            prf: self.prf.clone(),
        }
    }
}
impl EncodedOutputs {
    /// Require exactly one declared output, without exposing internal wire IDs.
    pub fn single(&self) -> Result<&EncodedOutput> {
        if self.values.len() != 1 {
            return Err(failed("expected exactly one output"));
        }
        Ok(self.values.values().next().expect("length checked"))
    }
    pub(crate) fn for_program(&self, program: ExponentLutProgramId) -> Result<&EncodedOutput> {
        if self.program != program {
            return Err(failed("output Program mismatch"));
        }
        self.single()
    }
}
/// Assertions are explicit offline operations; ordinary evaluation never decodes
/// or imports oracle secrets. Exponents use the native X^b representation.
pub enum ExpectedEncoding {
    /// Expect the native exponent `X^b` relation.
    Exponent(BigInt),
    /// Expect a canonical sparse-PRF digit relation.
    PrfDigit(BigInt),
}

fn digest(value: &impl Serialize) -> Result<[u8; 32]> {
    mxx_ir_core::encoding::hash_canonical(value).map_err(failed)
}
fn manifest_digest(manifest: &Manifest) -> Result<[u8; 32]> {
    let mut catalog = ArtifactCatalog::default();
    let production = manifest.production_id.clone();
    catalog.insert(production, manifest.clone()).map_err(failed)?;
    digest(&catalog.manifests().iter().collect::<Vec<_>>())
}

/// Resolve artifact directories before opening stores.  Resolving after
/// creation also closes the symlink-alias gap for callers that pass one
/// existing directory and one not-yet-created directory.
fn canonical_artifact_directory(path: &Path) -> Result<PathBuf> {
    fs::create_dir_all(path).map_err(failed)?;
    fs::canonicalize(path).map_err(failed)
}

fn validate_artifact_directories(paths: &[&Path]) -> Result<Vec<PathBuf>> {
    let canonical =
        paths.iter().map(|path| canonical_artifact_directory(path)).collect::<Result<Vec<_>>>()?;
    for (index, left) in canonical.iter().enumerate() {
        for right in canonical.iter().skip(index + 1) {
            if left == right || left.starts_with(right) || right.starts_with(left) {
                return Err(failed("artifact directories must be distinct, non-nested paths"));
            }
        }
    }
    Ok(canonical)
}

#[derive(Serialize, Deserialize)]
struct ApplicationCompletion {
    version: u32,
    production: ProductionId,
    identity: Identity,
    entries: Vec<(PathBuf, [u8; 32], [u8; 32])>,
}

fn completion_path(root: &Path, production: &ProductionId) -> PathBuf {
    let digest = digest(production).expect("ProductionId is serializable");
    root.join(format!(".exponent-lut-complete-{}.json", hex_bytes(&digest)))
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut output, byte| {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
        output
    })
}

/// Persist a small authenticated record with a synced temporary inode,
/// no-clobber same-directory publication, and parent-directory
/// synchronization. Artifact payloads remain in the runtime store; this helper
/// is only for metadata and completion records. If cleanup or the final parent
/// sync fails after publication, it returns an error even though the final file
/// may exist; callers must discard that session because no completion/token is
/// issued, and a retry in the same destination is intentionally rejected.
fn atomic_write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!("tmp-{:032x}", rand::random::<u128>()));
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut file =
        OpenOptions::new().write(true).create_new(true).open(&temporary).map_err(failed)?;
    if let Err(write_error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        drop(file);
        let cleanup = fs::remove_file(&temporary);
        return match cleanup {
            Ok(()) => Err(failed(write_error)),
            Err(cleanup_error) => Err(failed(format!(
                "writing metadata failed: {write_error}; cleaning temporary file failed: {cleanup_error}"
            ))),
        };
    }
    drop(file);
    // A hard link publishes the already-synced inode without replacing an
    // existing destination. Unlike an existence check followed by rename,
    // this remains no-clobber when writers race on the same metadata path.
    let publish = fs::hard_link(&temporary, path);
    let cleanup = fs::remove_file(&temporary);
    let sync_parent = || {
        fs::File::open(parent)
            .map_err(failed)
            .and_then(|directory| directory.sync_all().map_err(failed))
    };
    match publish {
        Ok(()) => {
            cleanup.map_err(failed)?;
            sync_parent()
        }
        Err(publish_error) => {
            cleanup.map_err(|cleanup_error| {
                failed(format!("publishing metadata failed: {publish_error}; cleaning temporary file failed: {cleanup_error}"))
            })?;
            sync_parent().map_err(|sync_error| {
                failed(format!("publishing metadata failed: {publish_error}; syncing parent failed: {sync_error}"))
            })?;
            if publish_error.kind() == std::io::ErrorKind::AlreadyExists {
                Err(failed("artifact metadata or completion record already exists"))
            } else {
                Err(failed(publish_error))
            }
        }
    }
}

fn write_completion(
    path: &Path,
    production: ProductionId,
    identity: Identity,
    references: &[&ArtifactReference],
) -> Result<()> {
    let entries = references
        .iter()
        .map(|reference| {
            (reference.directory.clone(), reference.metadata_digest, reference.manifest_digest)
        })
        .collect::<Vec<_>>();
    let metadata = ApplicationCompletion { version: 1, production, identity, entries };
    let bytes = serde_json::to_vec(&metadata).map_err(failed)?;
    atomic_write_new(path, &bytes)
}
fn write_metadata<D: Serialize>(
    directory: &Path,
    production: ProductionId,
    identity: Identity,
    namespace: ArtifactNamespace,
    descriptor: D,
    manifest: &Manifest,
) -> Result<ArtifactReference> {
    let metadata = Metadata {
        identity: identity.clone(),
        production: production.clone(),
        namespace,
        descriptor,
    };
    fs::create_dir_all(directory).map_err(failed)?;
    let metadata_path = directory.join("exponent-lut.json");
    atomic_write_new(&metadata_path, &serde_json::to_vec(&metadata).map_err(failed)?)?;
    Ok(ArtifactReference {
        directory: directory.into(),
        production,
        identity,
        completion: PathBuf::new(),
        metadata_digest: digest(&metadata)?,
        manifest_digest: manifest_digest(manifest)?,
        namespace,
    })
}

fn output_schema_digest(manifest: &Manifest, wire: WireId) -> Result<[u8; 32]> {
    let prefix = format!("output.{}", wire.index());
    let names = [
        format!("{prefix}.encoding.0"),
        format!("{prefix}.encoding.1"),
        format!("{prefix}.key.0"),
        format!("{prefix}.key.1"),
        format!("{prefix}.key.2"),
        format!("{prefix}.key.3"),
    ];
    let mut schema = BTreeMap::new();
    for name in names {
        let descriptor = manifest
            .artifacts
            .get(&name)
            .ok_or_else(|| failed(format!("output manifest is missing {name}")))?;
        schema.insert(name, descriptor.clone());
    }
    digest(&schema)
}

fn write_output_metadata(path: &Path, metadata: &OutputMetadata) -> Result<[u8; 32]> {
    atomic_write_new(path, &serde_json::to_vec(metadata).map_err(failed)?)?;
    digest(metadata)
}

impl ArtifactReference {
    fn open<D: DeserializeOwned + Serialize>(
        &self,
        expected_namespace: ArtifactNamespace,
        scheme: &RnsBggScheme,
    ) -> Result<(FileArtifactStore, Manifest, D)> {
        if self.namespace != expected_namespace {
            return Err(failed("artifact namespace mismatch"));
        }
        if self.identity.version != 1 || self.identity.parameter != scheme.artifact_binding() {
            return Err(failed("artifact parameter/context mismatch"));
        }
        if self.completion.as_os_str().is_empty() {
            return Err(failed("artifact completion record is missing"));
        }
        let completion: ApplicationCompletion =
            serde_json::from_slice(&fs::read(&self.completion).map_err(failed)?).map_err(failed)?;
        let entry = completion
            .entries
            .iter()
            .find(|entry| entry.0 == self.directory)
            .ok_or_else(|| failed("artifact is not covered by the completion record"))?;
        if completion.version != 1 ||
            completion.production != self.production ||
            completion.identity != self.identity ||
            entry.1 != self.metadata_digest ||
            entry.2 != self.manifest_digest
        {
            return Err(failed("artifact completion record authentication failed"));
        }
        let metadata: Metadata<D> = serde_json::from_slice(
            &std::fs::read(self.directory.join("exponent-lut.json")).map_err(failed)?,
        )
        .map_err(failed)?;
        if metadata.identity != self.identity ||
            metadata.production != self.production ||
            metadata.namespace != self.namespace ||
            digest(&metadata)? != self.metadata_digest
        {
            return Err(failed("artifact metadata authentication failed"));
        }
        let mut store = FileArtifactStore::open(&self.directory).map_err(failed)?;
        let manifest = store.load_finalized_manifest(&self.production).map_err(failed)?;
        if manifest_digest(&manifest)? != self.manifest_digest {
            return Err(failed("artifact manifest authentication failed"));
        }
        Ok((store, manifest, metadata.descriptor))
    }
}

impl From<RnsBggError> for ExecutionError {
    fn from(value: RnsBggError) -> Self {
        failed(value)
    }
}
impl From<DslError> for ExecutionError {
    fn from(value: DslError) -> Self {
        failed(value)
    }
}

impl<B: Backend> ExponentLut<B> {
    /// Creates the application facade around audited parameters and an explicit
    /// runtime backend. This constructor does not sample secrets, create
    /// artifact directories, execute a graph, or run a lattice estimator.
    pub fn new(
        parameters: RnsBggParameters,
        context: [u8; 32],
        runtime: RuntimeConfiguration<B>,
    ) -> Self {
        Self {
            scheme: RnsBggScheme::new(parameters, context),
            runtime: RefCell::new(runtime),
            #[cfg(test)]
            graph_identity: None,
        }
    }
    #[cfg(test)]
    fn with_test_graph_identity_for_testing(
        mut self,
        setup: [u8; 32],
        seed: [u8; 32],
        preprocess_nonce: [u8; 32],
        encrypt_nonce: [u8; 32],
    ) -> Self {
        self.graph_identity =
            Some(TestGraphIdentity { setup, seed, preprocess_nonce, encrypt_nonce });
        self
    }
    fn graph_setup_nonce(&self) -> [u8; 32] {
        #[cfg(test)]
        if let Some(identity) = self.graph_identity {
            return identity.setup;
        }
        rand::random()
    }
    fn graph_setup_seed(&self) -> [u8; 32] {
        #[cfg(test)]
        if let Some(identity) = self.graph_identity {
            return identity.seed;
        }
        rand::random()
    }
    fn graph_preprocess_nonce(&self) -> [u8; 32] {
        #[cfg(test)]
        if let Some(identity) = self.graph_identity {
            return identity.preprocess_nonce;
        }
        rand::random()
    }
    fn graph_encrypt_nonce(&self) -> [u8; 32] {
        #[cfg(test)]
        if let Some(identity) = self.graph_identity {
            return identity.encrypt_nonce;
        }
        rand::random()
    }
    #[cfg(test)]
    fn backend_activity_count(&self) -> usize {
        self.runtime.borrow().backend_activity.load(Ordering::SeqCst)
    }
    #[cfg(test)]
    fn set_backend_activity_panic(&self, enabled: bool) {
        self.runtime.borrow_mut().panic_on_backend_activity = enabled;
    }
    pub(crate) fn sample_secrets(&self, shared: bool) -> EncodingSecrets {
        use rand::Rng;
        let n = self.scheme.parameters.level(0).expect("validated parameters").ring_dimension();
        let sample =
            || Arc::new((0..n).map(|_| rand::rng().random_range(-1i8..=1)).collect::<Vec<_>>());
        let payload = sample();
        let mask = if shared { payload.clone() } else { sample() };
        EncodingSecrets {
            payload: PayloadSecret { coefficients: payload, context: self.scheme.context },
            mask: MaskSecret { coefficients: mask, context: self.scheme.context },
        }
    }

    /// Build the exact three graphs used by benchmark estimation without
    /// opening an artifact store or invoking the backend. Sparse and generic
    /// programs use the same phase builders as production; artifact imports
    /// are represented by typed graph inputs during planning.
    fn benchmark_graphs_internal(
        &self,
        program: &ExponentLutProgram,
        fixed: &FixedRhs,
        payload: &PayloadSecret,
    ) -> Result<FacadeBenchmarkGraphs> {
        if fixed.program != program.id() {
            return Err(failed("fixed RHS Program mismatch"));
        }
        let plan = match fixed.sparse_lwr.clone() {
            Some(sparse) => self.build_sparse_preprocess_plan(program, sparse, payload)?,
            None => self.build_generic_preprocess_plan(program, fixed, payload)?,
        };
        Ok(FacadeBenchmarkGraphs {
            preprocess: plan.validated,
            encrypt: plan.encrypt,
            online_eval: plan.online,
        })
    }

    /// Reopen an encryption namespace in a fresh process using an independently
    /// retained token. The token is only an identity/digest anchor; the on-disk
    /// metadata is never treated as its own authentication authority.
    pub fn reopen_encryption(&self, token: ArtifactReferenceToken) -> Result<EncryptionArtifacts> {
        Ok(EncryptionArtifacts(ArtifactReference::reopen(
            token,
            ArtifactNamespace::Encryption,
            &self.scheme,
        )?))
    }

    /// Reopen a preprocessing evaluation namespace using a trusted token.
    pub fn reopen_evaluation(&self, token: ArtifactReferenceToken) -> Result<EvaluationArtifacts> {
        Ok(EvaluationArtifacts(ArtifactReference::reopen(
            token,
            ArtifactNamespace::Evaluation,
            &self.scheme,
        )?))
    }

    /// Reopen encrypted input artifacts using a trusted token.
    pub fn reopen_input(&self, token: ArtifactReferenceToken) -> Result<InputArtifacts> {
        Ok(InputArtifacts(ArtifactReference::reopen(
            token,
            ArtifactNamespace::Input,
            &self.scheme,
        )?))
    }

    fn validate_output_reference(
        &self,
        token: &OutputReferenceToken,
        expected_program: Option<&ExponentLutProgram>,
    ) -> Result<Manifest> {
        if token.identity.parameter != self.scheme.artifact_binding() {
            return Err(failed("output parameter/context mismatch"));
        }
        if let Some(program) = expected_program {
            if token.identity.program != program.id() || !program.outputs().contains(&token.wire) {
                return Err(failed("output Program or parameter/context mismatch"));
            }
        }
        if token.metadata.parent() != Some(token.directory.as_path()) {
            return Err(failed("output metadata is outside its artifact directory"));
        }
        let metadata: OutputMetadata =
            serde_json::from_slice(&fs::read(&token.metadata).map_err(failed)?).map_err(failed)?;
        if metadata.version != 1 ||
            digest(&metadata)? != token.metadata_digest ||
            metadata.identity != token.identity ||
            metadata.production != token.production ||
            metadata.program != token.identity.program ||
            metadata.wire != token.wire ||
            metadata.level != token.level ||
            metadata.manifest_digest != token.manifest_digest ||
            metadata.schema_digest != token.schema_digest ||
            metadata.completion != token.completion ||
            metadata.completion_store_count != token.completion_store_count ||
            metadata.completion_manifest_digest != token.completion_manifest_digest ||
            metadata.completion_writable_stores != token.completion_writable_stores ||
            metadata.prf != token.prf
        {
            return Err(failed("output provenance authentication failed"));
        }
        let proof =
            mxx_runtime::FileArtifactStoreRouter::read_completion_proof(token.completion.clone())
                .map_err(failed)?;
        if proof.path != token.completion ||
            proof.production != token.production ||
            proof.manifest_digest != token.completion_manifest_digest ||
            proof.store_count != token.completion_store_count ||
            proof.writable_stores != token.completion_writable_stores
        {
            return Err(failed("output completion marker authentication failed"));
        }
        let mut store = FileArtifactStore::open(&token.directory).map_err(failed)?;
        let manifest = store.load_finalized_manifest(&token.production).map_err(failed)?;
        if manifest_digest(&manifest)? != token.manifest_digest {
            return Err(failed("output manifest authentication failed"));
        }
        self.scheme.parameters.level(token.level)?;
        if let Some(prf) = &token.prf {
            self.validate_prf_output_metadata(token.level, prf)?;
        }
        if output_schema_digest(&manifest, token.wire)? != token.schema_digest {
            return Err(failed("output schema authentication failed"));
        }
        let output_prefix = format!("output.{}", token.wire.index());
        let required = [
            format!("{output_prefix}.encoding.0"),
            format!("{output_prefix}.encoding.1"),
            format!("{output_prefix}.key.0"),
            format!("{output_prefix}.key.1"),
            format!("{output_prefix}.key.2"),
            format!("{output_prefix}.key.3"),
        ];
        if required.iter().any(|name| !manifest.artifacts.contains_key(name)) {
            return Err(failed("output manifest is missing the selected output"));
        }
        Ok(manifest)
    }

    /// Reopen one persistent online output using a trusted token and the
    /// caller's expected Program. Temporary output stores are removed when
    /// their owning result is dropped and therefore cannot be reopened.
    pub fn reopen_output(
        &self,
        token: OutputReferenceToken,
        expected_program: &ExponentLutProgram,
    ) -> Result<EncodedOutput> {
        self.validate_output_reference(&token, Some(expected_program))?;
        Ok(EncodedOutput {
            identity: token.identity,
            wire: token.wire,
            level: token.level,
            production: token.production,
            store: Arc::new(FinalizedOutputStore {
                directory: token.directory,
                remove_on_drop: false,
                manifest_digest: token.manifest_digest,
            }),
            prf: token.prf,
            metadata: token.metadata,
            metadata_digest: token.metadata_digest,
            schema_digest: token.schema_digest,
            completion: token.completion,
            completion_store_count: token.completion_store_count,
            completion_manifest_digest: token.completion_manifest_digest,
            completion_writable_stores: token.completion_writable_stores,
        })
    }

    /// Construct an explicitly noiseless fixture for tests that need to
    /// isolate graph semantics. Production callers must use [`Self::new`].
    #[cfg(feature = "test-utils")]
    pub fn new_noiseless_for_testing(
        parameters: RnsBggParameters,
        context: [u8; 32],
        runtime: RuntimeConfiguration<B>,
    ) -> Self {
        Self {
            scheme: RnsBggScheme::new_noiseless_for_testing(parameters, context),
            runtime: RefCell::new(runtime),
            #[cfg(test)]
            graph_identity: None,
        }
    }
    fn secret_row(&self, coefficients: &[i8], context: [u8; 32]) -> Result<RnsBggRow> {
        let ring = self.scheme.ring(0)?;
        if context != self.scheme.context ||
            coefficients.len() != self.scheme.parameters.level(0)?.ring_dimension() as usize
        {
            return Err(failed("secret context mismatch"));
        }
        Ok(RnsBggRow::new([[
            ring.polynomial(coefficients.iter().map(|x| IntExpr::constant(*x))),
            ring.identity(1),
        ]])?)
    }
    fn payload(&self, secret: &PayloadSecret) -> Result<RnsBggPayloadSecret> {
        Ok(RnsBggPayloadSecret {
            row: self.secret_row(&secret.coefficients, secret.context)?,
            context: secret.context,
        })
    }
    fn mask(&self, secret: &MaskSecret) -> Result<RnsBggMaskSecret> {
        Ok(RnsBggMaskSecret {
            row: self.secret_row(&secret.coefficients, secret.context)?,
            context: secret.context,
        })
    }

    fn validate_prf_output_metadata(
        &self,
        level: usize,
        metadata: &PrfOutputMetadata,
    ) -> Result<()> {
        metadata.profile.validate().map_err(failed)?;
        let digit_scale = self.scheme.parameters.level(level)?.modulus().as_ref() /
            BigUint::from(2usize * metadata.profile.p().map_err(failed)?);
        if digit_scale.is_zero() || digit_scale != metadata.digit_scale {
            return Err(failed("PRF output digit scale does not match its profile"));
        }
        Ok(())
    }

    fn validate_output_provenance(&self, output: &EncodedOutput) -> Result<Manifest> {
        self.validate_output_reference(&output.token(), None)
    }

    fn observe_graph(&self, stage: &str, graph: &ValidatedGraph) -> Result<()> {
        let mut runtime = self.runtime.borrow_mut();
        notify_graph_observer(&mut runtime.graph_observer, stage, graph)
    }

    fn build_sparse_preprocess_plan(
        &self,
        program: &ExponentLutProgram,
        fixed: SparseLwrFixedRhs,
        payload: &PayloadSecret,
    ) -> Result<PreprocessPlan> {
        let native = fixed.program;
        let native_for_readout = native.clone();
        let canonical_readout = native.readout.clone();
        let sparse_profile = native.profile();
        let sparse_selector_capacity = native.bucket_width();
        let sparse_block_count = native.block_count();
        if native.id() != program.id() {
            return Err(failed("sparse-LWR Program identity mismatch"));
        }
        let layout = &fixed.selector.generated.public_layout;
        if layout.parameters.bucket_count != native.block_count() ||
            fixed.public_vector.len() != layout.parameters.universe_size
        {
            return Err(failed("sparse-LWR layout and Program mismatch"));
        }
        let identity = Identity {
            version: 1,
            parameter: self.scheme.artifact_binding(),
            program: program.id(),
            setup: self.graph_setup_nonce(),
        };
        let payload = self.payload(payload)?;
        let seed = self.graph_setup_seed();
        let ring = self.scheme.ring(0)?;
        let setup_seed = ring.bytes_input("sparse-lwr-setup-seed", 32);
        let input_id = native.initial_input();
        let requirements = self.validate_sparse_requirements(program, &native)?;
        let terminal_level = requirements
            .outputs
            .iter()
            .find(|(wire, _)| program.outputs().contains(wire))
            .and_then(|(_, level)| *level)
            .ok_or_else(|| failed("sparse-LWR terminal level is missing from requirements"))?;
        if terminal_level != sparse_block_count {
            return Err(failed("sparse-LWR terminal level does not match block chain"));
        }
        let input_key_id = digest(&(
            "rns-bgg-sparse-lwr-input-key-v1",
            identity.setup,
            program.id(),
            self.scheme.artifact_binding(),
        ))?;
        let input_key = self.scheme.sample_input_key(
            0,
            setup_seed.clone(),
            input_key_id.as_slice(),
            input_key_id,
        )?;
        let input_attribute = native.initial_attribute(&ring).map_err(failed)?;
        let input_material = self.scheme.prepare_input(&payload, &input_key, input_attribute)?;

        // Route public residues and build one structural selector/label family
        // per block.  Each family is consumed immediately by preparation; no
        // block's host values are retained in the returned artifacts.
        let encoded = PbcEncodedPublicVector::route_usize(
            layout,
            &fixed.public_vector,
            native.profile().q().map_err(|_| failed("invalid PRF input modulus"))?,
        )
        .map_err(failed)?;
        PbcPublicVectorFamilyBinding::from_encoded(layout, &encoded).map_err(failed)?;
        let selector_bits =
            PbcTrustedSelectorBits::from_schedule(&fixed.selector.generated, &ring, input_key_id)
                .map_err(failed)?;
        let active = PbcActiveCellIndex::build(layout).map_err(failed)?;
        let mut selections = Vec::with_capacity(native.block_count());
        for block in 0..native.block_count() {
            let positions = active
                .bucket_iter(block)
                .ok_or_else(|| failed("sparse-LWR layout has no block"))?
                .collect::<Vec<_>>();
            if positions.is_empty() {
                return Err(failed("sparse-LWR layout contains an empty block"));
            }
            let level = block;
            let block_ring = self.scheme.ring(level + 1)?;
            let labels = Family::pack(
                positions
                    .iter()
                    .map(|(_, slot, _)| match layout.cells[block][*slot] {
                        PbcCell::Real { .. } => encoded.values[block][*slot]
                            .try_into()
                            .ok()
                            .and_then(|value: usize| {
                                sparse_profile.layout.mono(&block_ring, &BigInt::from(value)).ok()
                            })
                            .ok_or_else(|| mxx_dsl::DslError::Schema),
                        PbcCell::Dummy | PbcCell::Padding => sparse_profile
                            .layout
                            .mono(&block_ring, &BigInt::zero())
                            .map_err(|_| mxx_dsl::DslError::Schema),
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            )?;
            let selector_plain = Family::pack(
                positions
                    .iter()
                    .map(|(_, _, flat)| {
                        self.scheme.scalar(level + 1, selector_bits.runtime_bits()[*flat])
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            )?;
            let rhs_id = digest(&("rns-bgg-sparse-lwr-selector-v1", identity.setup, block))?;
            let sampled = self.scheme.sample_private_rhs_family(
                &payload,
                level,
                &RnsBggRhsPlaintextFamily { payloads: selector_plain, id: rhs_id },
                setup_seed.clone(),
                format!("sparse-lwr-selector-{block}").into_bytes(),
                rhs_id,
            )?;
            let range = FamilyRange::bounded(0, positions.len(), positions.len())
                .map_err(|_| failed("sparse-LWR block range is invalid"))?;
            selections.push(PrfSelectionInputs::new(sampled, &labels, range, rhs_id)?);
        }
        let initial = RnsBggPublicProgramValue::Scalar((input_key.clone(), None));
        let (output_key, prepared) =
            PreparedSparseLwrPrf::new(&self.scheme, native, initial, &selections)
                .map_err(failed)?;
        let readout_level = terminal_level;
        let readout_bank = self.scheme.sample_mask_bank(
            readout_level,
            sparse_profile.layout,
            setup_seed.clone(),
            b"sparse-lwr-readout-bank".as_slice(),
            digest(&("sparse-lwr-readout-bank", identity.setup))?,
        )?;
        let readout_contract = sparse_lwr_readout_contract(
            &self.scheme.parameters,
            &canonical_readout,
            sparse_profile.p().map_err(|_| failed("invalid PRF output modulus"))?,
            readout_bank.level,
        )
        .map_err(failed)?;
        let readout_material = self.scheme.sample_readout_material(
            &payload,
            &canonical_readout,
            &readout_bank,
            setup_seed.clone(),
            b"sparse-lwr-readout".as_slice(),
            digest(&("sparse-lwr-readout", identity.setup))?,
            Some(RnsBggReadoutWeights::Shared([
                vec![BigInt::from(readout_contract.digit_scale.clone())],
                vec![],
            ])),
            None,
        )?;
        if readout_material.target_bound() != &readout_contract.target_bound {
            return Err(failed("sparse-LWR readout target bound contract mismatch"));
        }
        let prepared_readout = native_for_readout
            .prepare_readout(&self.scheme, &output_key, &readout_material)
            .map_err(failed)?;
        let readout_binding = prepared_readout.binding();
        let readout_id = prepared_readout.id();
        let readout_input_id = output_key.at(0).0.id;
        let (graph, sparse_prepared) = prepared
            .export(&self.scheme, DslContext::new("exponent-lut-sparse-preprocess"), "eval.sparse")
            .map_err(failed)?;
        let (mut graph, sparse_readout) = self
            .scheme
            .export_prepared_readout(&prepared_readout, graph, "eval.sparse.readout")
            .map_err(failed)?;
        graph = graph.public_output("enc.readout.bank", readout_bank.branches.clone())?;
        let prefix = format!("enc.input.{}", input_id.index());
        graph = graph
            .public_output(format!("{prefix}.key"), input_key.clone())?
            .private_output(format!("{prefix}.payload"), input_material.row.clone())?;
        let enc_descriptor = EncryptionDescriptor {
            inputs: vec![EncryptionInput {
                input: input_id,
                key: input_key.schema(),
                payload: input_material.row.schema(),
                public: None,
            }],
            banks: Vec::new(),
            readout: Some(EncryptionBank {
                gate: ProgramGateInstance::Gate { step: 0 },
                binding: readout_binding.clone(),
                bank: readout_bank.id,
                level: readout_bank.level,
                branches: readout_bank.branches.schema(),
            }),
            payload: payload.row.schema(),
        };
        let output_key_schema = output_key.schema();
        let eval_descriptor = EvaluationDescriptor {
            prepared: None,
            sparse_prepared: Some(sparse_prepared),
            sparse_output: Some(output_key_schema),
            sparse_readout: Some(sparse_readout),
            sparse_readout_binding: Some(readout_binding.clone()),
            sparse_readout_id: Some(readout_id),
            sparse_readout_input_id: Some(readout_input_id),
            sparse_profile: Some(sparse_profile),
            sparse_selector_capacity: Some(sparse_selector_capacity),
            sparse_block_count: Some(sparse_block_count),
            rhs: Vec::new(),
        };
        graph = graph.public_output("eval.sparse.output-key", output_key.clone())?;
        for column in 0..2 {
            graph = graph.private_output(
                format!("enc.payload-secret.{column}"),
                payload.row.entry(0, column).clone(),
            )?;
        }
        let built = graph.build()?;
        let validated = built.validate(&ParamEnv::default()).map_err(failed)?;
        let nonce = self.graph_preprocess_nonce();
        let preprocessing = mxx_ir_core::artifact::production_id(
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .map_err(failed)?,
            nonce,
        );
        let manifest =
            export_validated_manifest(preprocessing.clone(), &validated).map_err(failed)?;
        let benchmark_mask =
            RnsBggMaskSecret { row: payload.row.clone(), context: self.scheme.context };
        let (encrypt_graph, input_descriptor) = self.build_encrypt_graph(
            preprocessing.clone(),
            &manifest,
            enc_descriptor.clone(),
            &benchmark_mask,
        )?;
        let encrypt_production = mxx_ir_core::artifact::production_id(
            mxx_ir_core::encoding::spec_hash(&encrypt_graph.source, &encrypt_graph.bindings)
                .map_err(failed)?,
            self.graph_encrypt_nonce(),
        );
        let encrypt_manifest =
            export_validated_manifest(encrypt_production.clone(), &encrypt_graph)
                .map_err(failed)?;
        let (online_graph, online_catalog) = self.build_sparse_online_graph(
            program,
            &eval_descriptor,
            &input_descriptor,
            &manifest,
            &encrypt_manifest,
            preprocessing,
            encrypt_production,
        )?;
        let online_graph = online_graph
            .build()?
            .validate_with_catalog(&ParamEnv::default(), &online_catalog)
            .map_err(failed)?;
        Ok(PreprocessPlan {
            validated,
            encrypt: encrypt_graph,
            online: online_graph,
            identity,
            setup_seed: seed,
            nonce,
            encryption: enc_descriptor,
            evaluation: eval_descriptor,
        })
    }

    fn build_generic_preprocess_plan(
        &self,
        program: &ExponentLutProgram,
        fixed: &FixedRhs,
        payload: &PayloadSecret,
    ) -> Result<PreprocessPlan> {
        let identity = Identity {
            version: 1,
            parameter: self.scheme.artifact_binding(),
            program: program.id(),
            setup: self.graph_setup_nonce(),
        };
        let payload = self.payload(payload)?;
        let seed = self.graph_setup_seed();
        let seed_input = self.scheme.ring(0)?.bytes_input("setup-seed", 32);
        let input_levels = fixed.inputs.iter().map(|(id, input)| (*id, input.level)).collect();
        let requirements = crate::rns_bgg::derive_program_requirements(
            program,
            &input_levels,
            &ProgramFamilyBindings::new(),
            &crate::rns_bgg::ProgramFamilyInputMetadata::default(),
        )
        .map_err(failed)?;
        let prepared = self.scheme.prepare_program(
            program.clone(),
            &payload,
            RnsBggProgramSetup {
                inputs: fixed.inputs.clone(),
                rhs: fixed.rhs.clone(),
                domain: RnsBggSetupDomain {
                    seed: seed_input,
                    label: b"exponent-lut-preprocess-v1".to_vec(),
                    nonce: identity.setup,
                },
            },
        )?;
        let (mut graph, descriptor) = prepared.prepared.export(
            self.scheme.fixed_product_binding(),
            DslContext::new("exponent-lut-preprocess"),
            "eval.prepared",
        )?;
        let mut inputs = Vec::new();
        for (id, (key, material)) in &prepared.inputs {
            let public = prepared.public_values()[&program.input_wires()[id]].1.clone();
            let prefix = format!("enc.input.{}", id.index());
            graph = graph
                .public_output(format!("{prefix}.key"), key.clone())?
                .private_output(format!("{prefix}.payload"), material.row.clone())?;
            if let Some(value) = &public {
                graph = graph.public_output(format!("{prefix}.public"), value.clone())?;
            }
            inputs.push(EncryptionInput {
                input: *id,
                key: key.schema(),
                payload: material.row.schema(),
                public: public.schema(),
            });
        }
        let mut banks = Vec::new();
        for gate in requirements.gates.iter().filter_map(|record| {
            matches!(record.gate, crate::rns_bgg::GateRequirement::Lut { .. })
                .then_some(record.instance)
        }) {
            let PreparedGate::Lut(value) = prepared
                .prepared
                .cache
                .gates
                .get(&gate)
                .ok_or_else(|| failed("LUT helper is missing from prepared requirements"))?
            else {
                return Err(failed("prepared LUT requirement has a non-LUT helper"));
            };
            let bank = value.mask_bank();
            let prefix = format!("enc.bank.{}", banks.len());
            graph = graph.public_output(prefix, bank.branches.clone())?;
            banks.push(EncryptionBank {
                gate,
                binding: value.binding.clone(),
                bank: bank.id,
                level: bank.level,
                branches: bank.branches.schema(),
            });
        }
        for column in 0..2 {
            graph = graph.private_output(
                format!("enc.payload-secret.{column}"),
                payload.row.entry(0, column).clone(),
            )?;
        }
        let enc_descriptor =
            EncryptionDescriptor { inputs, banks, readout: None, payload: payload.row.schema() };
        let eval_descriptor = EvaluationDescriptor {
            prepared: Some(descriptor),
            sparse_prepared: None,
            sparse_output: None,
            sparse_readout: None,
            sparse_readout_binding: None,
            sparse_readout_id: None,
            sparse_readout_input_id: None,
            sparse_profile: None,
            sparse_selector_capacity: None,
            sparse_block_count: None,
            rhs: prepared.rhs,
        };
        let validated = graph.build()?.validate(&ParamEnv::default()).map_err(failed)?;
        let nonce = self.graph_preprocess_nonce();
        let preprocessing = mxx_ir_core::artifact::production_id(
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .map_err(failed)?,
            nonce,
        );
        let manifest =
            export_validated_manifest(preprocessing.clone(), &validated).map_err(failed)?;
        let mask = RnsBggMaskSecret { row: payload.row.clone(), context: self.scheme.context };
        let (encrypt_graph, input_descriptor) = self.build_encrypt_graph(
            preprocessing.clone(),
            &manifest,
            enc_descriptor.clone(),
            &mask,
        )?;
        let encrypt_production = mxx_ir_core::artifact::production_id(
            mxx_ir_core::encoding::spec_hash(&encrypt_graph.source, &encrypt_graph.bindings)
                .map_err(failed)?,
            self.graph_encrypt_nonce(),
        );
        let encrypt_manifest =
            export_validated_manifest(encrypt_production.clone(), &encrypt_graph)
                .map_err(failed)?;
        let (online_graph, online_catalog, _) = self.build_prepared_online_graph(
            program,
            &eval_descriptor,
            &input_descriptor,
            &manifest,
            &encrypt_manifest,
            preprocessing,
            encrypt_production,
        )?;
        let online = online_graph
            .build()?
            .validate_with_catalog(&ParamEnv::default(), &online_catalog)
            .map_err(failed)?;
        Ok(PreprocessPlan {
            validated,
            encrypt: encrypt_graph,
            online,
            identity,
            setup_seed: seed,
            nonce,
            encryption: enc_descriptor,
            evaluation: eval_descriptor,
        })
    }

    fn preprocess_artifacts(
        &self,
        encryption_path: &Path,
        evaluation_path: &Path,
        plan: PreprocessPlan,
        setup_input_name: &'static str,
    ) -> Result<PreprocessedProgram> {
        let routes = plan
            .validated
            .source
            .outputs()
            .keys()
            .map(|name| (name.clone(), usize::from(name.starts_with("eval."))))
            .collect();
        self.observe_graph("preprocess", &plan.validated)?;
        let mut router = FileArtifactStoreRouter::new(
            vec![
                FileArtifactStore::open(&encryption_path).map_err(failed)?,
                FileArtifactStore::open(&evaluation_path).map_err(failed)?,
            ],
            routes,
            0,
            &encryption_path,
        )
        .map_err(failed)?;
        let result = self.execute_durable(
            &plan.validated,
            &mut router,
            [(setup_input_name.into(), RuntimeValue::Bytes(plan.setup_seed.to_vec()))].into(),
            plan.nonce,
        )?;
        let production =
            result.production_id.clone().ok_or_else(|| failed("missing production"))?;
        let union = router.verify_completed(&production).map_err(failed)?;
        mxx_runtime::validate_artifact_execution(&result, &union).map_err(failed)?;
        let enc_manifest =
            router.stores_mut()[0].load_finalized_manifest(&production).map_err(failed)?;
        let eval_manifest =
            router.stores_mut()[1].load_finalized_manifest(&production).map_err(failed)?;
        let completion = completion_path(
            encryption_path.parent().unwrap_or_else(|| Path::new(".")),
            &production,
        );
        let mut encryption = write_metadata(
            &encryption_path,
            production.clone(),
            plan.identity.clone(),
            ArtifactNamespace::Encryption,
            plan.encryption,
            &enc_manifest,
        )?;
        let mut evaluation = write_metadata(
            &evaluation_path,
            production.clone(),
            plan.identity.clone(),
            ArtifactNamespace::Evaluation,
            plan.evaluation,
            &eval_manifest,
        )?;
        encryption.completion = completion.clone();
        evaluation.completion = completion.clone();
        write_completion(&completion, production, plan.identity, &[&encryption, &evaluation])?;
        Ok(PreprocessedProgram {
            encryption: EncryptionArtifacts(encryption),
            evaluation: EvaluationArtifacts(evaluation),
        })
    }

    /// Prepare and stream two disjoint artifact namespaces. No reference escapes
    /// until both manifests and the router completion marker are finalized.
    pub fn preprocess(
        &self,
        program: &ExponentLutProgram,
        fixed: FixedRhs,
        payload: &PayloadSecret,
        encryption_path: impl AsRef<Path>,
        evaluation_path: impl AsRef<Path>,
    ) -> Result<PreprocessedProgram> {
        if fixed.program != program.id() {
            return Err(failed("fixed RHS Program mismatch"));
        }
        if let Some(sparse) = fixed.sparse_lwr.as_ref() {
            let [encryption_path, evaluation_path] = validate_artifact_directories(&[
                encryption_path.as_ref(),
                evaluation_path.as_ref(),
            ])?
            .try_into()
            .expect("two artifact directories");
            let plan = self.build_sparse_preprocess_plan(program, sparse.clone(), payload)?;
            return self.preprocess_artifacts(
                &encryption_path,
                &evaluation_path,
                plan,
                "sparse-lwr-setup-seed",
            );
        }
        let [encryption_path, evaluation_path] =
            validate_artifact_directories(&[encryption_path.as_ref(), evaluation_path.as_ref()])?
                .try_into()
                .expect("two artifact directories");
        let plan = self.build_generic_preprocess_plan(program, &fixed, payload)?;
        self.preprocess_artifacts(&encryption_path, &evaluation_path, plan, "setup-seed")
    }

    fn execute_durable<S: mxx_runtime::SessionStore>(
        &self,
        graph: &mxx_ir_core::ValidatedGraph,
        store: &mut S,
        inputs: BTreeMap<String, RuntimeValue<B>>,
        nonce: [u8; 32],
    ) -> Result<ExecutionResult<B>> {
        let mut runtime = self.runtime.borrow_mut();
        #[cfg(test)]
        assert!(!runtime.panic_on_backend_activity, "benchmark planning invoked backend admission");
        #[cfg(test)]
        runtime.backend_activity.fetch_add(1, Ordering::SeqCst);
        let config = runtime.execution;
        drop(
            runtime
                .backend
                .prepare_graph_admission(
                    graph,
                    false,
                    &inputs,
                    config.max_parallel_instances.get(),
                    true,
                )
                .map_err(failed)?,
        );
        mxx_runtime::execute_in_session_with_config(
            graph,
            &mut runtime.backend,
            inputs,
            store,
            nonce,
            config,
        )
        .map_err(failed)
    }

    fn build_encrypt_graph(
        &self,
        preprocessing: ProductionId,
        manifest: &Manifest,
        descriptor: EncryptionDescriptor,
        mask: &RnsBggMaskSecret,
    ) -> Result<(ValidatedGraph, InputDescriptor)> {
        let mut graph = DslContext::new("exponent-lut-encrypt");
        let mut inputs = Vec::new();
        for input in descriptor.inputs {
            let prefix = format!("enc.input.{}", input.input.index());
            let key = input.key.artifact_input(
                preprocessing.clone(),
                &format!("{prefix}.key"),
                ArtifactConfidentiality::Public,
            )?;
            self.scheme.check_key(&key)?;
            let row = input.payload.artifact_input(
                preprocessing.clone(),
                &format!("{prefix}.payload"),
                ArtifactConfidentiality::Private,
            )?;
            let public = input.public.artifact_input(
                preprocessing.clone(),
                &format!("{prefix}.public"),
                ArtifactConfidentiality::Public,
            )?;
            let material = RnsBggInputMaterial {
                row,
                level: key.level,
                context: self.scheme.context,
                key_id: key.id,
            };
            let value = self.scheme.encode(mask, &key, &material)?;
            let target = format!("input.{}", input.input.index());
            graph = graph.public_output(format!("{target}.encoding"), value.clone())?;
            if let Some(value) = &public {
                graph = graph.public_output(format!("{target}.public"), value.clone())?;
            }
            inputs.push((input.input, value.schema(), public.schema()));
        }
        let mut helpers = Vec::new();
        for (index, bank) in descriptor.banks.into_iter().enumerate() {
            let branches = bank.branches.artifact_input(
                preprocessing.clone(),
                &format!("enc.bank.{index}"),
                ArtifactConfidentiality::Public,
            )?;
            let masks = self.scheme.prepare_online_masks(
                mask,
                &RnsBggMaskBank {
                    id: bank.bank,
                    level: bank.level,
                    context: self.scheme.context,
                    branches,
                },
            )?;
            let helper = RnsBggOnlineLutHelpers { binding: bank.binding, masks };
            graph = graph.public_output(format!("helper.{index}"), helper.clone())?;
            helpers.push((bank.gate, HelperSchema::of(&helper)));
        }
        let readout = if let Some(bank) = descriptor.readout {
            let EncryptionBank { binding, bank: bank_id, level, branches: branch_schemas, .. } =
                bank;
            let branches = branch_schemas.artifact_input(
                preprocessing.clone(),
                "enc.readout.bank",
                ArtifactConfidentiality::Public,
            )?;
            let bank_value =
                RnsBggMaskBank { id: bank_id, level, context: self.scheme.context, branches };
            let masks = self.scheme.prepare_online_masks(mask, &bank_value)?;
            let helper = RnsBggOnlineLutHelpers { binding, masks };
            graph = graph.public_output("helper.readout", helper.clone())?;
            Some(HelperSchema::of(&helper))
        } else {
            None
        };
        let input_descriptor = InputDescriptor { preprocessing, inputs, helpers, readout };
        let mut catalog = ArtifactCatalog::default();
        catalog.insert(manifest.production_id.clone(), manifest.clone()).map_err(failed)?;
        let built = graph.build()?;
        let validated =
            built.validate_with_catalog(&ParamEnv::default(), &catalog).map_err(failed)?;
        Ok((validated, input_descriptor))
    }

    /// Validate the imported generic program against the same structural
    /// requirements used during preprocessing. Artifact descriptors remain
    /// authenticated separately; this check prevents an online builder from
    /// inventing a different gate/RHS/helper order.
    fn validate_generic_requirements(
        &self,
        program: &ExponentLutProgram,
        prepared: &PreparedProgram<RnsBggScheme>,
        eval: &EvaluationDescriptor,
        input_descriptor: &InputDescriptor,
    ) -> Result<crate::rns_bgg::ProgramRequirements> {
        let input_levels = program
            .input_wires()
            .iter()
            .map(|(id, wire)| {
                let value = prepared
                    .public_values()
                    .get(wire)
                    .ok_or_else(|| failed("prepared Program input wire is missing"))?;
                Ok((*id, value.0.level))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let requirements = crate::rns_bgg::derive_program_requirements(
            program,
            &input_levels,
            &ProgramFamilyBindings::new(),
            &crate::rns_bgg::ProgramFamilyInputMetadata::default(),
        )
        .map_err(failed)?;
        if !requirements.families.is_empty() {
            return Err(failed("generic online evaluation does not support structural families"));
        }
        let expected_rhs = requirements
            .gates
            .iter()
            .filter_map(|record| match record.gate {
                crate::rns_bgg::GateRequirement::MulPrivate { rhs, .. } => {
                    Some((record.instance, rhs))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let actual_rhs = eval.rhs.iter().map(|rhs| (rhs.gate, rhs.rhs)).collect::<Vec<_>>();
        if actual_rhs != expected_rhs {
            return Err(failed("generic evaluation RHS order does not match Program requirements"));
        }
        let expected_helpers = requirements
            .gates
            .iter()
            .filter_map(|record| match record.gate {
                crate::rns_bgg::GateRequirement::Lut { .. } => Some(record.instance),
                _ => None,
            })
            .collect::<Vec<_>>();
        let actual_helpers =
            input_descriptor.helpers.iter().map(|(gate, _)| *gate).collect::<Vec<_>>();
        if actual_helpers != expected_helpers {
            return Err(failed(
                "generic evaluation helper order does not match Program requirements",
            ));
        }
        Ok(requirements)
    }

    fn validate_sparse_requirements(
        &self,
        program: &ExponentLutProgram,
        native: &NativeSparseLwrPrfProgram,
    ) -> Result<crate::rns_bgg::ProgramRequirements> {
        let input_levels = BTreeMap::from([(native.initial_input(), 0usize)]);
        let requirements = crate::rns_bgg::derive_program_requirements(
            program,
            &input_levels,
            &ProgramFamilyBindings::new(),
            &crate::rns_bgg::ProgramFamilyInputMetadata::default(),
        )
        .map_err(failed)?;
        if !requirements.families.is_empty() ||
            requirements.gates.len() != native.accumulation.steps().len() ||
            requirements.gates.iter().any(|record| {
                !matches!(record.gate, crate::rns_bgg::GateRequirement::MulPrivate { .. })
            })
        {
            return Err(failed("sparse-LWR Program gate requirements do not match accumulation"));
        }
        Ok(requirements)
    }

    fn build_prepared_online_graph(
        &self,
        program: &ExponentLutProgram,
        eval: &EvaluationDescriptor,
        input_descriptor: &InputDescriptor,
        eval_manifest: &Manifest,
        input_manifest: &Manifest,
        evaluation_production: ProductionId,
        input_production: ProductionId,
    ) -> Result<(DslContext, ArtifactCatalog, BTreeMap<WireId, usize>)> {
        let descriptor =
            eval.prepared.as_ref().ok_or_else(|| failed("prepared descriptor is missing"))?;
        let prepared = PreparedProgram::import(
            self.scheme.fixed_product_binding(),
            self.scheme.clone(),
            program.clone(),
            ProgramFamilyBindings::new(),
            descriptor,
            evaluation_production.clone(),
            descriptor.identity().map_err(failed)?,
        )
        .map_err(failed)?;
        let requirements =
            self.validate_generic_requirements(program, &prepared, eval, input_descriptor)?;
        let mut bindings = EncodingBindings::<RnsBggScheme>::new();
        for (id, encoding, public) in &input_descriptor.inputs {
            let prefix = format!("input.{}", id.index());
            let value = encoding.artifact_input(
                input_production.clone(),
                &format!("{prefix}.encoding"),
                ArtifactConfidentiality::Public,
            )?;
            let public = public.artifact_input(
                input_production.clone(),
                &format!("{prefix}.public"),
                ArtifactConfidentiality::Public,
            )?;
            bindings = bindings.input(*id, (value, public));
        }
        for rhs in &eval.rhs {
            bindings =
                bindings.gate_rhs(rhs.gate, rhs.rhs, RnsBggPublicRhs::Identity(rhs.identity));
        }
        for (index, (gate, schema)) in input_descriptor.helpers.iter().cloned().enumerate() {
            let helper = schema.import(input_production.clone(), &format!("helper.{index}"))?;
            bindings = bindings.gate_helper(gate, ProgramValue::Scalar(helper));
        }
        let values = prepared.evaluate(&bindings).map_err(failed)?;
        let mut graph = DslContext::new("exponent-lut-online");
        let mut output_levels = BTreeMap::new();
        for wire in program.outputs() {
            let value = &values[wire].0;
            let key = &prepared.public_values()[wire].0;
            let expected_level = requirements
                .outputs
                .iter()
                .find(|(output, _)| output == wire)
                .and_then(|(_, level)| *level)
                .ok_or_else(|| {
                    failed("generic output level is missing from Program requirements")
                })?;
            if value.level != expected_level {
                return Err(failed("generic output level does not match Program requirements"));
            }
            output_levels.insert(*wire, value.level);
            graph = graph
                .public_output(format!("output.{}.encoding", wire.index()), value.row.clone())?
                .public_output(format!("output.{}.key", wire.index()), key.matrix.clone())?;
        }
        let mut catalog = ArtifactCatalog::default();
        catalog
            .insert(eval_manifest.production_id.clone(), eval_manifest.clone())
            .map_err(failed)?;
        catalog
            .insert(input_manifest.production_id.clone(), input_manifest.clone())
            .map_err(failed)?;
        Ok((graph, catalog, output_levels))
    }

    fn build_sparse_online_graph(
        &self,
        program: &ExponentLutProgram,
        eval: &EvaluationDescriptor,
        input_descriptor: &InputDescriptor,
        eval_manifest: &Manifest,
        input_manifest: &Manifest,
        evaluation_production: ProductionId,
        input_production: ProductionId,
    ) -> Result<(DslContext, ArtifactCatalog)> {
        let descriptor = eval
            .sparse_prepared
            .as_ref()
            .ok_or_else(|| failed("sparse-LWR prepared descriptor is missing"))?;
        let profile = eval.sparse_profile.ok_or_else(|| failed("sparse-LWR profile is missing"))?;
        let capacity = eval
            .sparse_selector_capacity
            .ok_or_else(|| failed("sparse-LWR selector capacity is missing"))?;
        let block_count =
            eval.sparse_block_count.ok_or_else(|| failed("sparse-LWR block count is missing"))?;
        let native =
            NativeSparseLwrPrfProgram::new(profile, capacity, block_count).map_err(failed)?;
        let requirements = self.validate_sparse_requirements(program, &native)?;
        let terminal_level = requirements
            .outputs
            .iter()
            .find(|(wire, _)| program.outputs().contains(wire))
            .and_then(|(_, level)| *level)
            .ok_or_else(|| failed("sparse-LWR terminal level is missing from requirements"))?;
        if terminal_level != block_count {
            return Err(failed("sparse-LWR terminal level does not match block chain"));
        }
        let initial_input = native.initial_input();
        let prepared = PreparedSparseLwrPrf::import(
            &self.scheme,
            native,
            descriptor,
            evaluation_production.clone(),
            descriptor.identity().map_err(failed)?,
        )
        .map_err(failed)?;
        let (input_id, encoding, public) = input_descriptor
            .inputs
            .first()
            .cloned()
            .ok_or_else(|| failed("sparse-LWR input descriptor is missing"))?;
        if public.is_some() || input_id != initial_input {
            return Err(failed("sparse-LWR input descriptor mismatch"));
        }
        let initial = encoding.artifact_input(
            input_production.clone(),
            &format!("input.{}.encoding", input_id.index()),
            ArtifactConfidentiality::Public,
        )?;
        let evaluated = prepared
            .evaluate(&self.scheme, RnsBggEncodingProgramValue::Scalar((initial, None)))
            .map_err(failed)?;
        let readout_descriptor = eval
            .sparse_readout
            .as_ref()
            .ok_or_else(|| failed("sparse-LWR readout descriptor is missing"))?;
        let readout_binding = eval
            .sparse_readout_binding
            .as_ref()
            .ok_or_else(|| failed("sparse-LWR readout binding is missing"))?;
        let readout_id = eval
            .sparse_readout_id
            .ok_or_else(|| failed("sparse-LWR readout identity is missing"))?;
        let readout_input_id = eval
            .sparse_readout_input_id
            .ok_or_else(|| failed("sparse-LWR readout input identity is missing"))?;
        let readout = profile.canonical_readout().map_err(failed)?;
        let prepared_readout = self.scheme.import_prepared_readout(
            readout_descriptor,
            evaluation_production.clone(),
            &readout,
            readout_binding,
            readout_input_id,
            readout_id,
        )?;
        let readout_helper = input_descriptor
            .readout
            .clone()
            .ok_or_else(|| failed("sparse-LWR readout masks are missing"))?
            .import(input_production.clone(), "helper.readout")?;
        let public_input = eval
            .sparse_output
            .as_ref()
            .ok_or_else(|| failed("sparse-LWR output key is missing"))?
            .artifact_input(
                evaluation_production,
                "eval.sparse.output-key",
                ArtifactConfidentiality::Public,
            )?;
        let public_output =
            self.scheme.project_readout(&readout, &public_input, &prepared_readout, &())?;
        let readout_value = self.scheme.evaluate_readout(
            &readout,
            &evaluated,
            &prepared_readout,
            &readout_helper.masks,
        )?;
        if readout_value.level() != terminal_level + 1 {
            return Err(failed("sparse-LWR readout level does not follow the terminal chain"));
        }
        let value = match readout_value.values {
            ProgramValue::Scalar(value) => value,
            ProgramValue::Family((values, _)) => values.at(Int::constant(0)),
        };
        let wire =
            *program.outputs().first().ok_or_else(|| failed("sparse-LWR Program has no output"))?;
        let output_key = match public_output.values {
            ProgramValue::Scalar(value) => value,
            ProgramValue::Family((values, _)) => values.at(Int::constant(0)),
        };
        let graph = DslContext::new("exponent-lut-online")
            .public_output(format!("output.{}.encoding", wire.index()), value)?
            .public_output(format!("output.{}.key", wire.index()), output_key)?;
        let mut catalog = ArtifactCatalog::default();
        catalog
            .insert(eval_manifest.production_id.clone(), eval_manifest.clone())
            .map_err(failed)?;
        catalog
            .insert(input_manifest.production_id.clone(), input_manifest.clone())
            .map_err(failed)?;
        Ok((graph, catalog))
    }

    /// Encrypt already prepared input contributions and generate mask-dependent
    /// helpers. The mask secret is neither exported nor retained by the result.
    pub fn encrypt(
        &self,
        artifacts: &EncryptionArtifacts,
        mask: &MaskSecret,
        input_path: impl AsRef<Path>,
    ) -> Result<InputArtifacts> {
        let input_path = input_path.as_ref();
        let directories = validate_artifact_directories(&[&artifacts.0.directory, input_path])?;
        let input_path = directories[1].clone();
        let (mut source, manifest, descriptor): (_, _, EncryptionDescriptor) =
            artifacts.0.open(ArtifactNamespace::Encryption, &self.scheme)?;
        for column in 0..2 {
            let payload_key = ArtifactKey {
                production: artifacts.0.production.clone(),
                name: format!("enc.payload-secret.{column}"),
                index: None,
            };
            let payload_descriptor = manifest
                .artifacts
                .get(&payload_key.name)
                .ok_or_else(|| failed("preprocessing payload-secret artifact is missing"))?;
            if payload_descriptor.confidentiality != ArtifactConfidentiality::Private ||
                payload_descriptor.family_count.is_some()
            {
                return Err(failed("preprocessing payload-secret artifact descriptor mismatch"));
            }
            source.load(&payload_key, payload_descriptor).map_err(failed)?;
        }
        let mask = self.mask(mask)?;
        let (validated, descriptor) =
            self.build_encrypt_graph(artifacts.0.production.clone(), &manifest, descriptor, &mask)?;
        self.observe_graph("encrypt", &validated)?;
        let output_names = validated.source.outputs().keys().cloned().collect::<Vec<_>>();
        let nonce = self.graph_encrypt_nonce();
        let production = mxx_ir_core::artifact::production_id(
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .map_err(failed)?,
            nonce,
        );
        let routes = output_names
            .iter()
            .cloned()
            .map(|name| (production.clone(), name.clone(), 1))
            .chain(
                manifest
                    .artifacts
                    .keys()
                    .map(|name| (artifacts.0.production.clone(), name.clone(), 0)),
            )
            .collect::<Vec<_>>();
        let mut router = FileArtifactStoreRouter::new_for_production(
            vec![source, FileArtifactStore::open(&input_path).map_err(failed)?],
            routes,
            1,
            &input_path,
            vec![false, true],
        )
        .map_err(failed)?;
        let result = self.execute_durable(&validated, &mut router, BTreeMap::new(), nonce)?;
        let production =
            result.production_id.clone().ok_or_else(|| failed("missing production"))?;
        let union = router.verify_completed(&production).map_err(failed)?;
        mxx_runtime::validate_artifact_execution(&result, &union).map_err(failed)?;
        let manifest =
            router.stores_mut()[1].load_finalized_manifest(&production).map_err(failed)?;
        let mut reference = write_metadata(
            &input_path,
            production.clone(),
            artifacts.0.identity.clone(),
            ArtifactNamespace::Input,
            descriptor,
            &manifest,
        )?;
        let completion = completion_path(&input_path, &production);
        reference.completion = completion.clone();
        write_completion(&completion, production, artifacts.0.identity.clone(), &[&reference])?;
        Ok(InputArtifacts(reference))
    }

    /// Evaluate using only public evaluation/input artifacts. No secret is an
    /// argument or an artifact import, and no decoding/oracle is implicit.
    pub fn online_eval(
        &self,
        program: &ExponentLutProgram,
        evaluation: &EvaluationArtifacts,
        input: &InputArtifacts,
    ) -> Result<EncodedOutputs> {
        if evaluation.0.identity != input.0.identity ||
            program.id() != evaluation.0.identity.program
        {
            return Err(failed("evaluation/input setup mismatch"));
        }
        validate_artifact_directories(&[&evaluation.0.directory, &input.0.directory])?;
        let (eval_store, eval_manifest, eval): (_, _, EvaluationDescriptor) =
            evaluation.0.open(ArtifactNamespace::Evaluation, &self.scheme)?;
        let (input_store, input_manifest, input_descriptor): (_, _, InputDescriptor) =
            input.0.open(ArtifactNamespace::Input, &self.scheme)?;
        if input_descriptor.preprocessing != evaluation.0.production {
            return Err(failed("input preprocessing provenance mismatch"));
        }
        let planned_eval = eval.clone();
        let planned_input = input_descriptor.clone();
        let mut graph = DslContext::new("exponent-lut-online");
        let mut output_levels = BTreeMap::new();
        let mut prf_output = None;
        if eval.prepared.is_some() {
            // Generic online evaluation uses the same pure builder as benchmark planning.
            // It imports only public artifacts and never receives a secret value.
            let (prepared_graph, _, prepared_levels) = self.build_prepared_online_graph(
                program,
                &eval,
                &input_descriptor,
                &eval_manifest,
                &input_manifest,
                evaluation.0.production.clone(),
                input.0.production.clone(),
            )?;
            graph = prepared_graph;
            output_levels = prepared_levels;
        } else {
            let descriptor = eval
                .sparse_prepared
                .as_ref()
                .ok_or_else(|| failed("sparse-LWR prepared descriptor is missing"))?;
            let profile =
                eval.sparse_profile.ok_or_else(|| failed("sparse-LWR profile is missing"))?;
            let capacity = eval
                .sparse_selector_capacity
                .ok_or_else(|| failed("sparse-LWR selector capacity is missing"))?;
            let block_count = eval
                .sparse_block_count
                .ok_or_else(|| failed("sparse-LWR block count is missing"))?;
            let native =
                NativeSparseLwrPrfProgram::new(profile, capacity, block_count).map_err(failed)?;
            let requirements = self.validate_sparse_requirements(program, &native)?;
            let terminal_level = requirements
                .outputs
                .iter()
                .find(|(wire, _)| program.outputs().contains(wire))
                .and_then(|(_, level)| *level)
                .ok_or_else(|| failed("sparse-LWR terminal level is missing from requirements"))?;
            if terminal_level != block_count {
                return Err(failed("sparse-LWR terminal level does not match block chain"));
            }
            let initial_input = native.initial_input();
            let prepared = PreparedSparseLwrPrf::import(
                &self.scheme,
                native,
                descriptor,
                evaluation.0.production.clone(),
                descriptor.identity().map_err(failed)?,
            )
            .map_err(failed)?;
            let (input_id, encoding, public) = input_descriptor
                .inputs
                .into_iter()
                .next()
                .ok_or_else(|| failed("sparse-LWR input descriptor is missing"))?;
            if public.is_some() || input_id != initial_input {
                return Err(failed("sparse-LWR input descriptor mismatch"));
            }
            let initial = encoding.artifact_input(
                input.0.production.clone(),
                &format!("input.{}.encoding", input_id.index()),
                ArtifactConfidentiality::Public,
            )?;
            let evaluated = prepared
                .evaluate(&self.scheme, RnsBggEncodingProgramValue::Scalar((initial, None)))
                .map_err(failed)?;
            let readout_descriptor = eval
                .sparse_readout
                .as_ref()
                .ok_or_else(|| failed("sparse-LWR readout descriptor is missing"))?;
            let readout_binding = eval
                .sparse_readout_binding
                .as_ref()
                .ok_or_else(|| failed("sparse-LWR readout binding is missing"))?;
            let readout_id = eval
                .sparse_readout_id
                .ok_or_else(|| failed("sparse-LWR readout identity is missing"))?;
            let readout_input_id = eval
                .sparse_readout_input_id
                .ok_or_else(|| failed("sparse-LWR readout input identity is missing"))?;
            let readout = profile.canonical_readout().map_err(failed)?;
            let prepared_readout = self
                .scheme
                .import_prepared_readout(
                    readout_descriptor,
                    evaluation.0.production.clone(),
                    &readout,
                    readout_binding,
                    readout_input_id,
                    readout_id,
                )
                .map_err(failed)?;
            let readout_helper = input_descriptor
                .readout
                .ok_or_else(|| failed("sparse-LWR readout masks are missing"))?
                .import(input.0.production.clone(), "helper.readout")?;
            let public_input = eval
                .sparse_output
                .as_ref()
                .ok_or_else(|| failed("sparse-LWR output key is missing"))?
                .artifact_input(
                    evaluation.0.production.clone(),
                    "eval.sparse.output-key",
                    ArtifactConfidentiality::Public,
                )?;
            let public_output = self
                .scheme
                .project_readout(&readout, &public_input, &prepared_readout, &())
                .map_err(failed)?;
            let readout_value = self
                .scheme
                .evaluate_readout(&readout, &evaluated, &prepared_readout, &readout_helper.masks)
                .map_err(failed)?;
            if readout_value.level() != terminal_level + 1 {
                return Err(failed("sparse-LWR readout level does not follow the terminal chain"));
            }
            let output_level = readout_value.level();
            let modulus = self.scheme.parameters.level(output_level)?.modulus().clone();
            let digit_scale =
                modulus.as_ref() / BigUint::from(2usize * profile.p().map_err(failed)?);
            if digit_scale.is_zero() {
                return Err(failed("sparse-LWR readout digit scale is zero"));
            }
            prf_output = Some(PrfOutputMetadata { profile, digit_scale });
            let readout_value = readout_value.values;
            let value = match readout_value {
                ProgramValue::Scalar(value) => value,
                ProgramValue::Family((values, _)) => values.at(Int::constant(0)),
            };
            let wire = *program
                .outputs()
                .first()
                .ok_or_else(|| failed("sparse-LWR Program has no output"))?;
            output_levels.insert(wire, output_level);
            let output_key = match public_output.values {
                ProgramValue::Scalar(value) => value,
                ProgramValue::Family((values, _)) => values.at(Int::constant(0)),
            };
            graph = graph
                .public_output(format!("output.{}.encoding", wire.index()), value)?
                .public_output(format!("output.{}.key", wire.index()), output_key)?;
        }
        if eval.prepared.is_none() {
            let (planned_graph, _) = self.build_sparse_online_graph(
                program,
                &planned_eval,
                &planned_input,
                &eval_manifest,
                &input_manifest,
                evaluation.0.production.clone(),
                input.0.production.clone(),
            )?;
            graph = planned_graph;
        }
        let mut catalog = ArtifactCatalog::default();
        catalog.insert(evaluation.0.production.clone(), eval_manifest.clone()).map_err(failed)?;
        catalog.insert(input.0.production.clone(), input_manifest.clone()).map_err(failed)?;
        let built = graph.build()?;
        let output_names = built.graph.outputs().keys().cloned().collect::<Vec<_>>();
        let graph = built.validate_with_catalog(&ParamEnv::default(), &catalog).map_err(failed)?;
        self.observe_graph("online_eval", &graph)?;
        let nonce = rand::random::<[u8; 32]>();
        let production = mxx_ir_core::artifact::production_id(
            mxx_ir_core::encoding::spec_hash(&graph.source, &graph.bindings).map_err(failed)?,
            nonce,
        );
        let routes = output_names
            .iter()
            .map(|name| (production.clone(), name.clone(), 2))
            .chain(
                eval_manifest
                    .artifacts
                    .keys()
                    .map(|name| (evaluation.0.production.clone(), name.clone(), 0))
                    .chain(
                        input_manifest
                            .artifacts
                            .keys()
                            .map(|name| (input.0.production.clone(), name.clone(), 1)),
                    ),
            )
            .collect::<Vec<_>>();
        let (output_directory, remove_on_drop) = {
            let runtime = self.runtime.borrow();
            let root = match &runtime.working_directory {
                ArtifactWorkingDirectory::Persistent(path) => path.clone(),
                ArtifactWorkingDirectory::Temporary => std::env::temp_dir(),
            };
            fs::create_dir_all(&root).map_err(failed)?;
            let directory =
                root.join(format!("mxx-exponent-lut-output-{:032x}", rand::random::<u128>()));
            fs::create_dir(&directory).map_err(failed)?;
            (directory, matches!(runtime.working_directory, ArtifactWorkingDirectory::Temporary))
        };
        let mut store = FileArtifactStoreRouter::new_for_production(
            vec![
                eval_store,
                input_store,
                FileArtifactStore::open(&output_directory).map_err(failed)?,
            ],
            routes,
            2,
            &output_directory,
            vec![false, false, true],
        )
        .map_err(failed)?;
        let mut runtime = self.runtime.borrow_mut();
        #[cfg(test)]
        assert!(!runtime.panic_on_backend_activity, "benchmark planning invoked backend admission");
        #[cfg(test)]
        runtime.backend_activity.fetch_add(1, Ordering::SeqCst);
        let config = runtime.execution;
        drop(
            runtime
                .backend
                .prepare_graph_admission(
                    &graph,
                    false,
                    &BTreeMap::new(),
                    config.max_parallel_instances.get(),
                    true,
                )
                .map_err(failed)?,
        );
        let result = mxx_runtime::execute_in_session_with_config(
            &graph,
            &mut runtime.backend,
            BTreeMap::new(),
            &mut store,
            nonce,
            config,
        )
        .map_err(failed)?;
        let union = store.verify_completed(&production).map_err(failed)?;
        mxx_runtime::validate_artifact_execution(&result, &union).map_err(failed)?;
        let completion_proof = store.completion_proof(&production).map_err(failed)?;
        let output_manifest_digest = manifest_digest(&union)?;
        let output_store = Arc::new(FinalizedOutputStore {
            directory: output_directory,
            remove_on_drop,
            manifest_digest: output_manifest_digest,
        });
        let mut outputs = BTreeMap::new();
        for wire in program.outputs() {
            let level = *output_levels.get(wire).ok_or_else(|| failed("missing output level"))?;
            let schema_digest = output_schema_digest(&union, *wire)?;
            let metadata = OutputMetadata {
                version: 1,
                identity: evaluation.0.identity.clone(),
                production: production.clone(),
                program: program.id(),
                wire: *wire,
                level,
                manifest_digest: output_manifest_digest,
                schema_digest,
                completion: completion_proof.path.clone(),
                completion_store_count: completion_proof.store_count,
                completion_manifest_digest: completion_proof.manifest_digest,
                completion_writable_stores: completion_proof.writable_stores.clone(),
                prf: prf_output.clone(),
            };
            let metadata_path =
                output_store.directory.join(format!("output.{}.json", wire.index()));
            let metadata_digest = write_output_metadata(&metadata_path, &metadata)?;
            outputs.insert(
                *wire,
                EncodedOutput {
                    identity: evaluation.0.identity.clone(),
                    wire: *wire,
                    level,
                    production: production.clone(),
                    store: output_store.clone(),
                    prf: prf_output.clone(),
                    metadata: metadata_path,
                    metadata_digest,
                    schema_digest,
                    completion: completion_proof.path.clone(),
                    completion_store_count: completion_proof.store_count,
                    completion_manifest_digest: completion_proof.manifest_digest,
                    completion_writable_stores: completion_proof.writable_stores.clone(),
                },
            );
        }
        Ok(EncodedOutputs { program: program.id(), values: outputs })
    }

    /// Check the selected encoding relation with strict `|centered(e)| < q/4`.
    /// Exponent outputs use the native `c - sA + X^b rG` relation, while PRF
    /// digits use the readout relation `c - sA - digit_scale*d`.
    /// This assertion does not modify runtime outputs or weaken the noise model.
    pub fn assert_encoding(
        &self,
        output: &EncodedOutput,
        expected: ExpectedEncoding,
        secrets: &EncodingSecrets,
    ) -> Result<()> {
        if output.identity.parameter != self.scheme.artifact_binding() {
            return Err(failed("output parameter/context mismatch"));
        }
        self.validate_output_provenance(output)?;
        let ring = self.scheme.ring(output.level)?;
        let columns = match &expected {
            ExpectedEncoding::Exponent(_) => {
                self.scheme.parameters.component_columns(output.level)?
            }
            // A terminal PRF readout is a single-column RNS-BGG row.  Its
            // digit spacing is represented by the scalar target below; it is
            // not a decomposed state value and therefore has no next-level
            // gadget columns.
            ExpectedEncoding::PrfDigit(_) => 1,
        };
        let row = RnsBggRow::new([[
            ring.artifact_input(
                output.production.clone(),
                format!("output.{}.encoding.0", output.wire.index()),
                (1, columns),
                ArtifactConfidentiality::Public,
            ),
            ring.artifact_input(
                output.production.clone(),
                format!("output.{}.encoding.1", output.wire.index()),
                (1, columns),
                ArtifactConfidentiality::Public,
            ),
        ]])?;
        let key = RnsBggMatrix::new(std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                ring.artifact_input(
                    output.production.clone(),
                    format!("output.{}.key.{}", output.wire.index(), r * 2 + c),
                    (1, columns),
                    ArtifactConfidentiality::Public,
                )
            })
        }))?;
        let mask =
            self.scheme.reduce_mask_secret_modulus(&self.mask(secrets.mask())?, output.level)?;
        let residual = match expected {
            ExpectedEncoding::Exponent(exponent) => {
                let payload = self.scheme.reduce_payload_secret_modulus(
                    &self.payload(secrets.payload())?,
                    output.level,
                )?;
                let target = ExpLayout { stride_log2: 0, width_log2: 1 }
                    .mono(&ring, &exponent)
                    .map_err(failed)?;
                let gadget = self.scheme.state_gadget(output.level)?;
                row - mask.clone() * key + (target * payload).map(|entry| entry * gadget.clone())
            }
            ExpectedEncoding::PrfDigit(digit) => {
                let metadata =
                    output.prf.as_ref().ok_or_else(|| failed("PRF digit metadata is missing"))?;
                self.validate_prf_output_metadata(output.level, metadata)?;
                let target = self
                    .scheme
                    .scalar(output.level, digit * BigInt::from(metadata.digit_scale.clone()))?;
                let expected = RnsBggRow::new([[target, ring.zero((1, 1))]]).map_err(failed)?;
                row - mask * key - expected
            }
        };
        let mut artifact_store =
            FileArtifactStore::open(&output.store.directory).map_err(failed)?;
        let manifest =
            artifact_store.load_finalized_manifest(&output.production).map_err(failed)?;
        let mut catalog = ArtifactCatalog::default();
        catalog.insert(output.production.clone(), manifest).map_err(failed)?;
        let graph = DslContext::new("exponent-lut-encoding-assertion")
            .output("residual", residual)?
            .build()?
            .validate_with_catalog(&ParamEnv::default(), &catalog)
            .map_err(failed)?;
        let mut runtime = self.runtime.borrow_mut();
        #[cfg(test)]
        assert!(!runtime.panic_on_backend_activity, "benchmark planning invoked backend execution");
        #[cfg(test)]
        runtime.backend_activity.fetch_add(1, Ordering::SeqCst);
        let config = runtime.execution;
        drop(
            runtime
                .backend
                .prepare_graph_admission(
                    &graph,
                    false,
                    &BTreeMap::new(),
                    config.max_parallel_instances.get(),
                    true,
                )
                .map_err(failed)?,
        );
        let result = mxx_runtime::execute_with_config(
            &graph,
            &mut runtime.backend,
            BTreeMap::new(),
            &mut artifact_store,
            SamplingMode::Fresh,
            config,
        )
        .map_err(failed)?;
        let q =
            BigInt::from(self.scheme.parameters.level(output.level)?.modulus().as_ref().clone());
        for value in result.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else {
                return Err(failed("residual is not resident matrix"));
            };
            for column in 0..columns {
                let slice = runtime
                    .backend
                    .slice(
                        matrix,
                        None,
                        Some(&mxx_runtime::backend::IndexRange { start: column, end: column + 1 }),
                    )
                    .map_err(failed)?;
                for coefficient in
                    runtime.backend.polynomial_values(&slice, false).map_err(failed)?
                {
                    let coefficient = ((coefficient % &q) + &q) % &q;
                    let centered =
                        if coefficient > &q / 2u8 { coefficient - &q } else { coefficient };
                    if centered.abs() * 4u8 >= q {
                        return Err(failed(format!(
                            "encoding residual reaches q/4 (centered coefficient {})",
                            centered
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

impl<B: Backend> ExponentLut<B> {
    /// Build the exact preprocess, encrypt, and online graphs used by the
    /// facade without executing them or writing artifact directories.
    pub fn benchmark_graphs(
        &self,
        program: &ExponentLutProgram,
        fixed: &FixedRhs,
        payload: &PayloadSecret,
    ) -> std::result::Result<FacadeBenchmarkGraphs, ExecutionError> {
        self.benchmark_graphs_internal(program, fixed, payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Int};
    use mxx_runtime::SessionStore;

    fn metadata_fixture(
        namespace: ArtifactNamespace,
    ) -> (tempfile::TempDir, RnsBggScheme, ArtifactReferenceToken) {
        let request = RnsBggPlanRequest {
            ring_dimension: 16,
            widths: vec![1],
            encoding_error_sigma: 1.into(),
            crt_bits: 16,
            terminal_depth: 1,
            max_crt_depth: 16,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 1,
        };
        let (parameters, _) = RnsBggParameters::plan_fixture(&request).unwrap();
        let scheme = RnsBggScheme::new(parameters, [4; 32]);
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("artifact");
        let production = mxx_ir_core::artifact::ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([5; 32]),
            execution_nonce: [6; 32],
        };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::new(),
        };
        let mut store = FileArtifactStore::new(&directory).unwrap();
        store
            .open_session(&mxx_runtime::SessionDescriptor::new(
                production.clone(),
                "fixture",
                [7; 32],
            ))
            .unwrap();
        store.finalize_session(manifest.clone()).unwrap();
        let identity = Identity {
            version: 1,
            parameter: scheme.artifact_binding(),
            program: ExponentLutProgramId([8; 32]),
            setup: [9; 32],
        };
        let mut reference = write_metadata(
            &directory,
            production.clone(),
            identity.clone(),
            namespace,
            serde_json::json!({"fixture": true}),
            &manifest,
        )
        .unwrap();
        reference.completion = completion_path(root.path(), &production);
        write_completion(&reference.completion, production, identity, &[&reference]).unwrap();
        (root, scheme, reference.token())
    }

    #[test]
    fn graph_observer_is_optional_ordered_and_receives_exact_graph() {
        let graph = DslContext::new("observer-test")
            .output("value", Int::constant(1))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let expected = &graph as *const ValidatedGraph;
        let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_stages = Arc::clone(&stages);
        let mut observer: Option<
            Box<dyn FnMut(&str, &ValidatedGraph) -> std::result::Result<(), String>>,
        > = Some(Box::new(move |stage, observed| {
            assert!(std::ptr::eq(expected, observed));
            observed_stages.lock().unwrap().push(stage.to_owned());
            Ok(())
        }));
        notify_graph_observer(&mut observer, "preprocess", &graph).unwrap();
        notify_graph_observer(&mut observer, "encrypt", &graph).unwrap();
        notify_graph_observer(&mut observer, "online_eval", &graph).unwrap();
        assert_eq!(*stages.lock().unwrap(), ["preprocess", "encrypt", "online_eval"]);

        let mut unset = None;
        notify_graph_observer(&mut unset, "unused", &graph).unwrap();
        let mut failing: Option<
            Box<dyn FnMut(&str, &ValidatedGraph) -> std::result::Result<(), String>>,
        > = Some(Box::new(|_, _| Err("observer failure".into())));
        let error = notify_graph_observer(&mut failing, "preprocess", &graph).unwrap_err();
        assert!(error.to_string().contains("observer failure"));
    }

    #[test]
    fn sparse_lwr_facade_keeps_selector_opaque_and_rejects_non_residues() {
        let profile =
            SparseLwrPrfProfile::new(crate::ExpLayout { stride_log2: 2, width_log2: 2 }, 2)
                .unwrap();
        let program = SparseLwrPrfProgram::new(profile, 16, 8).unwrap();
        let selector = program.selector_key(8, &[1, 3, 4, 6, 7], [7; 32]).unwrap();
        let fixed = program.fixed_rhs(&[0, 1, 2, 3, 4, 5, 6, 7], &selector).unwrap();
        assert_eq!(fixed.program, program.program().id());
        assert!(program.fixed_rhs(&[0, 1, 2], &selector).is_err());
        assert!(program.fixed_rhs(&[0, 1, 2, 3, 4, 5, 6, 8], &selector).is_err());
    }

    #[test]
    fn sparse_lwr_sampling_reuses_one_mathematical_secret() {
        let (_root, scheme, _) = metadata_fixture(ArtifactNamespace::Encryption);
        let backend =
            mxx_runtime::backend::poly::cpu_backend(scheme.parameters.levels().iter().cloned());
        let engine = ExponentLut::new(
            (*scheme.parameters).clone(),
            scheme.context,
            RuntimeConfiguration::new(backend, ExecutionConfig::default()),
        );
        let profile =
            SparseLwrPrfProfile::new(crate::ExpLayout { stride_log2: 2, width_log2: 2 }, 2)
                .unwrap();
        let program = SparseLwrPrfProgram::new(profile, 4, 8).unwrap();
        let secrets = program.sample_secrets(&engine);
        assert!(Arc::ptr_eq(&secrets.payload.coefficients, &secrets.mask.coefficients));
    }

    #[test]
    fn sparse_lwr_noise_audit_uses_readout_level_and_weighted_bound() {
        let request = RnsBggPlanRequest {
            ring_dimension: 16,
            widths: vec![1; 8],
            encoding_error_sigma: 3.into(),
            crt_bits: 16,
            terminal_depth: 1,
            max_crt_depth: 16,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 1,
        };
        let (parameters, _) = RnsBggParameters::plan_fixture(&request).unwrap();
        assert_eq!(parameters.encoding_error_sigma(), &BigDecimal::from(3u8));
        assert_eq!(parameters.encoding_error_bound(), &BigUint::from(19u8));
        let profile =
            SparseLwrPrfProfile::new(crate::ExpLayout { stride_log2: 2, width_log2: 2 }, 2)
                .unwrap();
        let program = SparseLwrPrfProgram::new(profile, 4, 4).unwrap();
        let selector = program.selector_key(1, &[0], [8; 32]).unwrap();
        let fixed = program.fixed_rhs(&[0], &selector).unwrap();
        let audit = program.audit_noise(&parameters, &fixed, false, 1).unwrap();
        assert_eq!(audit.output_level, audit.terminal_level + 1);
        assert_eq!(audit.branches, profile.layout.width());
        assert_eq!(audit.target_bound, &audit.digit_scale * &audit.digit_bound);
        assert!(audit.worst_case_terminal_fit);
        assert_eq!(audit.accumulation_failure_exponent, None);
        assert_eq!(audit.readout_failure_exponent, None);
        let average = program.audit_noise(&parameters, &fixed, true, 1).unwrap();
        assert_ne!(average.selected_error, audit.selected_error);
        assert!(average.strict_terminal_fit);
        assert_eq!(average.accumulation_failure_exponent, Some(2));
        assert_eq!(average.readout_failure_exponent, Some(2));
        assert!(program.audit_noise(&parameters, &fixed, true, u32::MAX).is_err());
    }

    #[test]
    fn sparse_lwr_preprocess_uses_the_public_facade() {
        let request = RnsBggPlanRequest {
            ring_dimension: 16,
            widths: vec![1; 8],
            encoding_error_sigma: 1.into(),
            crt_bits: 16,
            terminal_depth: 1,
            max_crt_depth: 16,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 1,
        };
        let (parameters, _) = RnsBggParameters::plan_fixture(&request).unwrap();
        let scheme = RnsBggScheme::new(parameters, [4; 32]);
        let backend =
            mxx_runtime::backend::poly::cpu_backend(scheme.parameters.levels().iter().cloned());
        let observed = Arc::new(std::sync::Mutex::new(Vec::<(String, ValidatedGraph)>::new()));
        let observed_output = Arc::clone(&observed);
        let planning_root = tempfile::tempdir().unwrap();
        let before_planning = directory_snapshot(planning_root.path());
        let engine = ExponentLut::new(
            (*scheme.parameters).clone(),
            scheme.context,
            RuntimeConfiguration::new(backend, ExecutionConfig::default())
                .with_working_directory(ArtifactWorkingDirectory::Persistent(
                    planning_root.path().to_path_buf(),
                ))
                .with_graph_observer(move |stage, graph| {
                    observed_output.lock().unwrap().push((stage.to_owned(), graph.clone()));
                    Ok(())
                })
                .with_backend_activity_panic(),
        )
        .with_test_graph_identity_for_testing([1; 32], [2; 32], [3; 32], [4; 32]);
        let profile =
            SparseLwrPrfProfile::new(crate::ExpLayout { stride_log2: 2, width_log2: 2 }, 2)
                .unwrap();
        let compiled = SparseLwrPrfProgram::new(profile, 4, 4).unwrap();
        let selector = compiled.selector_key(1, &[0], [8; 32]).unwrap();
        // Q is the input residue modulus (8 here), while P=4 is only the
        // output alphabet.  This nonzero residue also exercises the signed
        // upper half of the canonical readout with a nonzero layout stride.
        let fixed = compiled.fixed_rhs(&[6], &selector).unwrap();
        let secrets = compiled.sample_secrets(&engine);
        let activity_before = engine.backend_activity_count();
        let planned =
            engine.benchmark_graphs(compiled.program(), &fixed, secrets.payload()).unwrap();
        engine.set_backend_activity_panic(false);
        assert_eq!(engine.backend_activity_count(), activity_before);
        assert_eq!(directory_snapshot(planning_root.path()), before_planning);
        assert!(observed.lock().unwrap().is_empty());
        assert_eq!(planned.preprocess.source.name(), "exponent-lut-sparse-preprocess");
        assert_eq!(planned.encrypt.source.name(), "exponent-lut-encrypt");
        assert_eq!(planned.online_eval.source.name(), "exponent-lut-online");
        let enc_dir = tempfile::tempdir().unwrap();
        let eval_dir = tempfile::tempdir().unwrap();
        let artifacts = engine
            .preprocess(
                compiled.program(),
                fixed,
                secrets.payload(),
                enc_dir.path(),
                eval_dir.path(),
            )
            .unwrap();
        let input_dir = tempfile::tempdir().unwrap();
        let encrypted =
            engine.encrypt(&artifacts.encryption, secrets.mask(), input_dir.path()).unwrap();
        let outputs =
            engine.online_eval(compiled.program(), &artifacts.evaluation, &encrypted).unwrap();
        let output = compiled.output(&outputs).unwrap();
        engine
            .assert_encoding(&output, ExpectedEncoding::PrfDigit(BigInt::from(-1)), &secrets)
            .unwrap();
        // The signed canonical output is -1 for this upper-half residue.  A
        // sufficiently distant positive digit is a wrong answer and must fail
        // the strict residual check, rather than being accepted by an
        // unsigned-only contract.  The adjacent +1 spacing can be just below
        // q/4 because the digit spacing is floor(q/(2P)).
        assert!(
            engine
                .assert_encoding(&output, ExpectedEncoding::PrfDigit(BigInt::from(2u8)), &secrets)
                .is_err()
        );
        let observed = observed.lock().unwrap();
        assert_eq!(
            observed.iter().map(|(stage, _)| stage.as_str()).collect::<Vec<_>>(),
            ["preprocess", "encrypt", "online_eval"]
        );
        let planned_graphs = [&planned.preprocess, &planned.encrypt, &planned.online_eval];
        let observed_graphs = observed.iter().map(|(_, graph)| graph).collect::<Vec<_>>();
        assert_eq!(planned_graphs.len(), observed_graphs.len());
        for (stage, planned, observed) in [
            ("preprocess", planned_graphs[0], observed_graphs[0]),
            ("encrypt", planned_graphs[1], observed_graphs[1]),
            ("online_eval", planned_graphs[2], observed_graphs[2]),
        ] {
            let planned_fingerprint = graph_fingerprint(planned);
            let observed_fingerprint = graph_fingerprint(observed);
            if planned_fingerprint != observed_fingerprint {
                panic!(
                    "graph stage {stage} fingerprint mismatch at {}",
                    first_graph_difference(&planned_fingerprint, &observed_fingerprint, "$"),
                );
            }
        }
    }

    fn first_graph_difference(
        planned: &serde_json::Value,
        observed: &serde_json::Value,
        path: &str,
    ) -> String {
        match (planned, observed) {
            (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
                for key in left.keys().chain(right.keys()) {
                    let next = format!("{path}.{key}");
                    match (left.get(key), right.get(key)) {
                        (Some(left), Some(right)) if left == right => {}
                        (Some(left), Some(right)) => {
                            return first_graph_difference(left, right, &next);
                        }
                        _ => return next,
                    }
                }
                path.to_owned()
            }
            (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
                if left.len() != right.len() {
                    return format!("{path} length {} != {}", left.len(), right.len());
                }
                for (index, (left, right)) in left.iter().zip(right).enumerate() {
                    if left != right {
                        return first_graph_difference(left, right, &format!("{path}[{index}]"));
                    }
                }
                path.to_owned()
            }
            _ => format!("{path} values differ: {planned:?} != {observed:?}"),
        }
    }

    fn directory_snapshot(path: &std::path::Path) -> Vec<String> {
        let mut entries = std::fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        entries.sort();
        entries
    }

    /// Compare the complete graph semantics, not only output names.  The
    /// production nonce and artifact IDs are intentionally fresh for each
    /// execution, so only those authenticated identity fields are normalized;
    /// node kinds, bindings, artifact names, catalog references, and routing
    /// schemas remain part of the fingerprint.
    fn graph_fingerprint(graph: &ValidatedGraph) -> serde_json::Value {
        let source_scopes = graph
            .source
            .scopes()
            .iter()
            .map(|(id, scope)| {
                let nodes = scope
                    .nodes()
                    .iter()
                    .map(|node| {
                        serde_json::json!({
                            "kind": node.kind(),
                            "arguments": node.arguments().iter().map(|argument| {
                                scope.wire_ref(argument).expect("validated argument wire")
                            }).collect::<Vec<_>>(),
                            "output_types": node.output_types(),
                            "benchmark_role": node.benchmark_role(),
                        })
                    })
                    .collect::<Vec<_>>();
                serde_json::json!({
                    "id": id,
                    "inputs": scope.inputs(),
                    "outputs": scope.outputs(),
                    "nodes": nodes,
                })
            })
            .collect::<Vec<_>>();
        let validated_scopes = graph
            .scopes
            .iter()
            .map(|(id, scope)| {
                serde_json::json!({
                    "id": id,
                    "last_use": scope
                        .liveness
                        .last_use
                        .iter()
                        .map(|(wire, index)| serde_json::json!([wire, index]))
                        .collect::<Vec<_>>(),
                    "retained": scope.liveness.retained.iter().collect::<Vec<_>>(),
                    "wire_types": scope
                        .wire_types
                        .iter()
                        .map(|(wire, ty)| serde_json::json!([wire, ty]))
                        .collect::<Vec<_>>(),
                    "artifact_inputs": scope
                        .artifact_inputs
                        .iter()
                        .map(|(wire, artifact)| serde_json::json!([wire, artifact]))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        let mut fingerprint = serde_json::json!({
            "name": graph.source.name(),
            "parameters": graph.source.parameters(),
            "outputs": graph.source.outputs(),
            "effect_roots": graph.source.effect_roots(),
            "real_constants": graph.source.real_constants(),
            "source_scopes": source_scopes,
            "bindings": graph.bindings,
            "validated_scopes": validated_scopes,
            "warnings": graph.warnings,
        });
        normalize_graph_identity(&mut fingerprint, None);
        if let serde_json::Value::Object(fields) = &mut fingerprint {
            for field in ["source_scopes", "validated_scopes"] {
                fields
                    .get_mut(field)
                    .and_then(serde_json::Value::as_array_mut)
                    .expect("graph fingerprint scope arrays")
                    .sort_by_key(|scope| serde_json::to_string(scope).expect("graph scope JSON"));
            }
        }
        fingerprint
    }

    fn normalize_graph_identity(value: &mut serde_json::Value, field: Option<&str>) {
        // Only explicit production/artifact identity fields are normalized.
        // Hash sampler labels, prefixes, components, node kinds, bindings,
        // artifact names, schemas, and routing fields remain exact. The
        // planner test injects nonces explicitly so no byte-level heuristic is
        // needed here.
        if matches!(field, Some("production_id" | "content_hash")) {
            *value = serde_json::Value::String("<fresh-production>".into());
            return;
        }
        if matches!(field, Some("execution_nonce" | "artifact_id")) {
            *value = serde_json::Value::String("<fresh>".into());
            return;
        }
        match value {
            serde_json::Value::Object(fields) => {
                for (name, child) in fields {
                    normalize_graph_identity(child, Some(name));
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    normalize_graph_identity(child, None);
                }
            }
            serde_json::Value::Null |
            serde_json::Value::Bool(_) |
            serde_json::Value::Number(_) |
            serde_json::Value::String(_) => {}
        }
    }

    #[test]
    fn graph_fingerprint_preserves_hash_domain_labels() {
        let mut stable_a = serde_json::json!({
            "tag_prefix": vec![0xa5u8; 32],
            "tag_components": [{
                "tag": "Bytes",
                "value": b"stable-component-a".to_vec(),
            }],
        });
        let mut stable_b = stable_a.clone();
        stable_b["tag_prefix"][0] = serde_json::json!(0xc7u8);
        stable_b["tag_components"][0]["value"] = serde_json::json!(b"stable-component-b".to_vec());
        normalize_graph_identity(&mut stable_a, None);
        normalize_graph_identity(&mut stable_b, None);
        assert_ne!(stable_a, stable_b);

        let mut fresh_a = serde_json::json!({"execution_nonce": vec![0u8; 32]});
        let mut fresh_b = serde_json::json!({"execution_nonce": vec![1u8; 32]});
        normalize_graph_identity(&mut fresh_a, None);
        normalize_graph_identity(&mut fresh_b, None);
        assert_eq!(fresh_a, fresh_b);
    }

    #[test]
    fn artifact_provenance_rejects_tampering_cross_mix_and_missing_completion() {
        let (root, scheme, token) = metadata_fixture(ArtifactNamespace::Encryption);
        assert!(root.path().is_dir());
        let reference = ArtifactReference::from_token(token.clone());
        assert!(
            reference.open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme).is_ok()
        );
        let backend =
            mxx_runtime::backend::poly::cpu_backend(scheme.parameters.levels().iter().cloned());
        let engine = ExponentLut::new(
            (*scheme.parameters).clone(),
            scheme.context,
            RuntimeConfiguration::new(backend, ExecutionConfig::default()),
        );
        assert!(engine.reopen_evaluation(token.clone()).is_err());
        assert!(engine.reopen_input(token.clone()).is_err());

        let mut wrong = token.clone();
        wrong.namespace = ArtifactNamespace::Evaluation;
        assert!(
            ArtifactReference::from_token(wrong)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );
        let mut wrong = token.clone();
        wrong.identity.program = ExponentLutProgramId([10; 32]);
        assert!(
            ArtifactReference::from_token(wrong)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );
        let mut wrong = token.clone();
        wrong.identity.parameter.parameters[0] ^= 1;
        assert!(
            ArtifactReference::from_token(wrong)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );
        let mut wrong = token.clone();
        wrong.identity.parameter.context[0] ^= 1;
        assert!(
            ArtifactReference::from_token(wrong)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );
        let mut wrong = token.clone();
        wrong.production.execution_nonce[0] ^= 1;
        assert!(
            ArtifactReference::from_token(wrong)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );
        let other_context = RnsBggScheme::new((*scheme.parameters).clone(), [5; 32]);
        assert!(
            reference
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &other_context)
                .is_err()
        );
        let different_parameters = RnsBggParameters::new(
            scheme.parameters.levels().to_vec(),
            2,
            scheme.parameters.operations_per_drop(),
        )
        .unwrap();
        let other_parameters = RnsBggScheme::new(different_parameters, scheme.context);
        assert!(
            reference
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &other_parameters)
                .is_err()
        );
        let mut wrong = token.clone();
        wrong.metadata_digest[0] ^= 1;
        assert!(
            ArtifactReference::from_token(wrong)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );
        let mut wrong = token.clone();
        wrong.manifest_digest[0] ^= 1;
        assert!(
            ArtifactReference::from_token(wrong)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );

        let metadata_path = token.directory.join("exponent-lut.json");
        let metadata = fs::read(&metadata_path).unwrap();
        let mut changed_metadata: serde_json::Value = serde_json::from_slice(&metadata).unwrap();
        changed_metadata["descriptor"]["fixture"] = serde_json::json!(false);
        fs::write(&metadata_path, serde_json::to_vec(&changed_metadata).unwrap()).unwrap();
        let mut forged = token.clone();
        forged.metadata_digest = digest(&changed_metadata).unwrap();
        assert!(
            ArtifactReference::from_token(forged)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );
        fs::write(&metadata_path, b"not metadata").unwrap();
        assert!(
            reference.open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme).is_err()
        );
        fs::write(&metadata_path, metadata).unwrap();

        let production_directory = fs::read_dir(token.directory.join("productions"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let manifest_path = production_directory.join("manifest.bin");
        let manifest = fs::read(&manifest_path).unwrap();
        let mut tampered: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
        tampered["ir_version"] = serde_json::json!(0);
        fs::write(&manifest_path, serde_json::to_vec(&tampered).unwrap()).unwrap();
        assert!(
            reference.open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme).is_err()
        );
        fs::write(&manifest_path, manifest).unwrap();

        let completion = fs::read(&token.completion).unwrap();
        fs::write(&token.completion, b"{").unwrap();
        assert!(
            reference.open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme).is_err()
        );
        fs::write(&token.completion, completion).unwrap();
        fs::remove_file(&token.completion).unwrap();
        assert!(
            reference.open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme).is_err()
        );

        let mut missing = token.clone();
        missing.directory = root.path().join("missing-artifact");
        assert!(
            ArtifactReference::from_token(missing)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );

        let (other_root, _, other) = metadata_fixture(ArtifactNamespace::Evaluation);
        assert!(other_root.path().is_dir());
        let mut cross_mix = token;
        cross_mix.directory = other.directory;
        assert!(
            ArtifactReference::from_token(cross_mix)
                .open::<serde_json::Value>(ArtifactNamespace::Encryption, &scheme)
                .is_err()
        );
    }

    #[test]
    fn preprocess_rejects_fixed_rhs_from_another_program_before_execution() {
        let circuit = crate::NandCircuit::new(2, vec![[0, 1]], 2).unwrap();
        let program =
            circuit.compile(crate::ExpLayout { stride_log2: 0, width_log2: 4 }, 16, 4).unwrap();
        let other_program = crate::NandCircuit::new(1, Vec::new(), 0)
            .unwrap()
            .compile(crate::ExpLayout { stride_log2: 0, width_log2: 4 }, 16, 1)
            .unwrap();
        let request = RnsBggPlanRequest {
            ring_dimension: 16,
            widths: vec![1],
            encoding_error_sigma: 1.into(),
            crt_bits: 16,
            terminal_depth: 1,
            max_crt_depth: 16,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 1,
        };
        let (parameters, _) = RnsBggParameters::plan_fixture(&request).unwrap();
        let backend = mxx_runtime::backend::poly::cpu_backend(parameters.levels().iter().cloned());
        let engine = ExponentLut::new(
            parameters,
            [11; 32],
            RuntimeConfiguration::new(backend, ExecutionConfig::default()),
        );
        let secrets = engine.sample_secrets(false);
        let enc_root = tempfile::tempdir().unwrap();
        let eval_root = tempfile::tempdir().unwrap();
        let error = match engine.preprocess(
            program.program(),
            other_program.fixed_rhs(&[false]).unwrap(),
            secrets.payload(),
            enc_root.path().join("enc"),
            eval_root.path().join("eval"),
        ) {
            Ok(_) => panic!("mismatched FixedRhs unexpectedly executed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("fixed RHS Program mismatch"));
    }

    #[test]
    fn small_cpu_nand_facade_noise_round_trip() {
        let circuit = crate::NandCircuit::new(2, vec![[0, 1]], 2).unwrap();
        let compiled =
            circuit.compile(crate::ExpLayout { stride_log2: 0, width_log2: 4 }, 16, 4).unwrap();
        let request = RnsBggPlanRequest {
            ring_dimension: 16,
            widths: (0..4).flat_map(|_| [8, 16]).collect(),
            encoding_error_sigma: "0.16".parse().unwrap(),
            crt_bits: 32,
            terminal_depth: 1,
            max_crt_depth: 8,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 4,
        };
        let started = std::time::Instant::now();
        eprintln!("small CPU facade: planning started");
        let (planned, _) = RnsBggParameters::plan_fixture(&request).unwrap();
        eprintln!("small CPU facade: planning {:?}", started.elapsed());
        let parameters = RnsBggParameters::new(
            planned.levels().to_vec(),
            request.encoding_error_sigma.clone(),
            request.operations_per_drop,
        )
        .unwrap();
        eprintln!(
            "small CPU facade: levels={:?}, columns={:?}",
            parameters.levels().iter().map(|level| level.crt_depth()).collect::<Vec<_>>(),
            (0..parameters.levels().len() - 1)
                .map(|level| parameters.component_columns(level).unwrap())
                .collect::<Vec<_>>()
        );
        let backend = mxx_runtime::backend::poly::cpu_backend(parameters.levels().iter().cloned());
        let root = tempfile::tempdir().unwrap();
        let engine = ExponentLut::new(
            parameters,
            [9; 32],
            RuntimeConfiguration::new(
                backend,
                ExecutionConfig { prepared_gpu_admission: false, ..ExecutionConfig::default() },
            )
            .with_working_directory(ArtifactWorkingDirectory::Persistent(
                root.path().join("runtime"),
            )),
        );
        let fixed = compiled.fixed_rhs(&[true, false]).unwrap();
        let secrets = compiled.sample_secrets(&engine).unwrap();
        eprintln!("small CPU facade: preprocess started");
        let prepared = engine
            .preprocess(
                compiled.program(),
                fixed,
                secrets.payload(),
                root.path().join("enc"),
                root.path().join("eval"),
            )
            .unwrap();
        eprintln!("small CPU facade: preprocess {:?}", started.elapsed());
        let encryption = engine.reopen_encryption(prepared.encryption.token()).unwrap();
        let evaluation = engine.reopen_evaluation(prepared.evaluation.token()).unwrap();
        drop(prepared);
        eprintln!("small CPU facade: encrypt started");
        let input = engine.encrypt(&encryption, secrets.mask(), root.path().join("input")).unwrap();
        eprintln!("small CPU facade: encrypt {:?}", started.elapsed());
        let input = engine.reopen_input(input.token()).unwrap();
        eprintln!("small CPU facade: online started");
        let outputs = engine.online_eval(compiled.program(), &evaluation, &input).unwrap();
        eprintln!("small CPU facade: online {:?}", started.elapsed());
        let output_token = compiled.output(&outputs).unwrap().token();
        let output_token_bytes = serde_json::to_vec(&output_token).unwrap();
        assert!(output_token_bytes.len() < 4096);
        assert!(!String::from_utf8(output_token_bytes).unwrap().contains("coefficients"));
        drop(outputs);
        let other_program = crate::NandCircuit::new(1, Vec::new(), 0)
            .unwrap()
            .compile(crate::ExpLayout { stride_log2: 0, width_log2: 4 }, 16, 1)
            .unwrap();
        assert!(engine.reopen_output(output_token.clone(), other_program.program()).is_err());
        let mut wrong_output = output_token.clone();
        wrong_output.manifest_digest[0] ^= 1;
        assert!(engine.reopen_output(wrong_output, compiled.program()).is_err());
        let mut wrong_output = output_token.clone();
        wrong_output.identity.parameter.parameters[0] ^= 1;
        assert!(engine.reopen_output(wrong_output, compiled.program()).is_err());
        let mut wrong_output = output_token.clone();
        wrong_output.wire = WireId::from_index(output_token.wire.index() + 1);
        assert!(engine.reopen_output(wrong_output, compiled.program()).is_err());
        let mut wrong_output = output_token.clone();
        wrong_output.level += 1;
        assert!(engine.reopen_output(wrong_output, compiled.program()).is_err());
        let mut missing_output = output_token.clone();
        missing_output.directory = root.path().join("missing-output");
        assert!(engine.reopen_output(missing_output, compiled.program()).is_err());
        let output_metadata = fs::read(&output_token.metadata).unwrap();
        let mut tampered_output_metadata: serde_json::Value =
            serde_json::from_slice(&output_metadata).unwrap();
        tampered_output_metadata["level"] = serde_json::json!(0);
        fs::write(&output_token.metadata, serde_json::to_vec(&tampered_output_metadata).unwrap())
            .unwrap();
        assert!(engine.reopen_output(output_token.clone(), compiled.program()).is_err());
        fs::write(&output_token.metadata, output_metadata).unwrap();
        let output_production_directory = fs::read_dir(output_token.directory.join("productions"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let output_manifest_path = output_production_directory.join("manifest.bin");
        let output_manifest = fs::read(&output_manifest_path).unwrap();
        let mut tampered_output_manifest: serde_json::Value =
            serde_json::from_slice(&output_manifest).unwrap();
        tampered_output_manifest["ir_version"] = serde_json::json!(0);
        fs::write(&output_manifest_path, serde_json::to_vec(&tampered_output_manifest).unwrap())
            .unwrap();
        assert!(engine.reopen_output(output_token.clone(), compiled.program()).is_err());
        fs::write(&output_manifest_path, output_manifest).unwrap();
        let completion = fs::read(&output_token.completion).unwrap();
        let mut tampered_completion: serde_json::Value =
            serde_json::from_slice(&completion).unwrap();
        tampered_completion["writable_stores"] = serde_json::json!([0]);
        fs::write(&output_token.completion, serde_json::to_vec(&tampered_completion).unwrap())
            .unwrap();
        assert!(engine.reopen_output(output_token.clone(), compiled.program()).is_err());
        fs::write(&output_token.completion, &completion).unwrap();
        fs::remove_file(&output_token.completion).unwrap();
        assert!(engine.reopen_output(output_token.clone(), compiled.program()).is_err());
        fs::write(&output_token.completion, completion).unwrap();
        let output = engine.reopen_output(output_token, compiled.program()).unwrap();
        engine
            .assert_encoding(&output, ExpectedEncoding::Exponent(BigInt::from(1u8)), &secrets)
            .unwrap();
    }

    #[test]
    fn reference_token_is_small_and_contains_no_secret_payload() {
        let token = ArtifactReferenceToken {
            directory: PathBuf::from("/tmp/enc"),
            completion: PathBuf::from("/tmp/complete"),
            production: ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([1; 32]),
                execution_nonce: [2; 32],
            },
            identity: Identity {
                version: 1,
                parameter: RnsBggArtifactBinding { parameters: [3; 32], context: [4; 32] },
                program: ExponentLutProgramId([5; 32]),
                setup: [6; 32],
            },
            metadata_digest: [7; 32],
            manifest_digest: [8; 32],
            namespace: ArtifactNamespace::Encryption,
        };
        let encoded = serde_json::to_vec(&token).unwrap();
        assert!(encoded.len() < 1024);
        assert!(!String::from_utf8(encoded.clone()).unwrap().contains("coefficients"));
        assert_eq!(
            serde_json::from_slice::<ArtifactReferenceToken>(&encoded).unwrap().metadata_digest,
            [7; 32]
        );
    }

    #[test]
    fn artifact_directory_validation_rejects_aliases_and_nesting() {
        let root = tempfile::tempdir().unwrap();
        let left = root.path().join("left");
        let right = root.path().join("right");
        assert!(validate_artifact_directories(&[&left, &right]).is_ok());
        assert!(validate_artifact_directories(&[&left, &left]).is_err());
        assert!(validate_artifact_directories(&[root.path(), &left]).is_err());
        #[cfg(unix)]
        {
            let alias = root.path().join("alias");
            std::os::unix::fs::symlink(&left, &alias).unwrap();
            assert!(validate_artifact_directories(&[&left, &alias]).is_err());
        }
    }

    #[test]
    fn atomic_write_new_publishes_one_racing_record_without_clobbering() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("record.json");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let writers = (0..8)
            .map(|index| {
                let barrier = std::sync::Arc::clone(&barrier);
                let path = path.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    atomic_write_new(&path, format!("record-{index}").as_bytes()).is_ok()
                })
            })
            .collect::<Vec<_>>();
        let successes =
            writers.into_iter().map(|writer| writer.join().unwrap()).filter(|won| *won).count();
        assert_eq!(successes, 1);
        let published = fs::read_to_string(&path).unwrap();
        assert!(published.starts_with("record-"));
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);

        let existing = root.path().join("existing.json");
        fs::write(&existing, b"original").unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let writers = (0..8)
            .map(|index| {
                let barrier = std::sync::Arc::clone(&barrier);
                let existing = existing.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    atomic_write_new(&existing, format!("replacement-{index}").as_bytes()).is_ok()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            writers.into_iter().map(|writer| writer.join().unwrap()).filter(|won| *won).count(),
            0
        );
        assert_eq!(fs::read(&existing).unwrap(), b"original");
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 2);
    }
}
