#![cfg(feature = "gpu")]

//! The predicate encryption round trip on the GPU runtime, at parameters whose noise guarantees
//! correctness and whose Ring-LWE instances reach a lattice security target.
//!
//! The parameter search tries the ring dimensions in order and, for each, the CRT depths through
//! [`PeParameters::select`]: a candidate is accepted when [`PeParameters::at_depth`] accepts it (the
//! regularity, truncation, and numeric-range conditions of the statistical trapdoor in
//! `mxx_pe::budget`), the simulation of the class's longest program
//! decrypts under
//! the selected noise model (`worst`: `4 E_PE + 2 < q`; `average`: failure probability at most
//! `2^-FAILURE_EXPONENT`, see `mxx_pe::noise::simulate_average`), and the Ring-LWE instance of the
//! encryption secret ([`PeParameters::security_bits`]) reaches the target according to
//! `lattice-estimator-cli`, which must be on `PATH` with its Sage environment. `chi` is
//! truncated at the budget's multiplier, and the key at the default preimage cutoff.
//!
//! On the accepted parameters, every stage runs on the GPU: setup, one encryption of a random
//! attribute and a random binary message, and keys for two random branching programs of the
//! class's maximal length, adjusted to output `0` (authorized) and `1` on the attribute. The
//! authorized key must decrypt the message; the unauthorized decryption must complete, and its
//! rounded phase is not constrained.
//!
//! Every workflow (also with `EXPORT_ONLY`) prints, from its recorded graphs, the sizes of the
//! public key, the ciphertext, the key's secret part (preimages in small matrix form) and its
//! public evaluation part (`PE_SIZE`, packed at `ceil(log2 q)` bits per coefficient and as
//! stored), and the GPU work of decryption that `mxx_backends::gpu_cost_model` predicts from the
//! online evaluation and decoding graphs (`PE_DECRYPT_GPU_COUNT`: `mul64_instructions`, the
//! weighted 64-bit multiplications, `gadget_decomposed_words`, the RNS residues decomposed into
//! gadget digits, and the other counters).
//!
//! `test_gpu_pe_time_window` runs the time window predicate of the implementation specification
//! (`BranchingProgram::time_window` for `[0900, 1759]`, class `PeClass::decimal_range(4)`) on its
//! test vectors with the same parameter search, and exports the same claim with `LEAN_EXPORT`
//! (and `EXPORT_ONLY`).
//!
//! With `MXX_PE_ROUND_TRIP_LEAN_EXPORT=<parameter id>`, `test_gpu_pe_round_trip` exports the Lean
//! correctness claim of the executed graphs to `lean/<parameter id>/generated`. The claim does not
//! depend on the predicate: the program is an external input of key generation (preprocessing) and
//! online evaluation, and the requirement is that the attribute and the program form a valid
//! instance with `F(x) = 0`.
//!
//! | Variable (`MXX_PE_ROUND_TRIP_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `LEAN_EXPORT` | unset | Export the Lean correctness claim to `lean/<value>/generated` |
//! | `EXPORT_ONLY` | 0 | With `LEAN_EXPORT`, 1 builds and validates the authorized workflow's graphs without executing them (only the payload secret is sampled, on the CPU) and exports the claim; no GPU is needed |
//! | `ATTRIBUTE_LENGTH`, `ALPHABET_SIZE`, `STATE_WIDTH`, `MAX_STEPS` | 1, 2, 1, 1 | The class |
//! | `MIN_LOG_RING_DIMENSION`, `MAX_LOG_RING_DIMENSION` | 12, 15 | Ring-dimension candidates |
//! | `MAX_CRT_DEPTH` | 12 | CRT-depth candidates `1..=` |
//! | `CRT_BITS`, `BASE_BITS` | 50, `ceil(CRT_BITS / 2)` | CRT prime and gadget base widths |
//! | `ERROR_SIGMA` | 3.19 | Sigma of `chi`, the contract of every fresh error |
//! | `COMPACT_ENCODINGS` | 1 | 1 publishes the compressed digit encodings over `pq` and the anchor over `pq` (`PeParameters::with_compact_encodings`), 0 everything at `q` in full gadget width |
//! | `LUT_RADIX` | `PeClass::default_lut_radix` | Radix `B` of the LUT automorphism decomposition (`PeParameters::with_lut_radix`); 0 switches every automorphism directly (`B = W`). The default takes 3 or 2 switches per branch when they divide `log2 W` (`W = 4096`: `B = 16`; `W = 2`: direct) |
//! | `TRAPDOOR_SIGMA` | 4.578 | TrapGen sigma |
//! | `NOISE_MODEL`, `FAILURE_EXPONENT` | `worst`, 128 | Noise model of the correctness check |
//! | `SECURITY_BITS` | 128 | Required lattice security of the encryption secret |
//! | `STORAGE` | `device` | Stage artifacts: `device` (matrices in GPU memory, others in host memory), `host`, `file` (a temporary directory), or `file:<directory>` |
//! | `CRT_DEPTH` | unset | `test_gpu_pe_time_window`: the parameters at this CRT depth and the smallest ring dimension, without the noise simulation and the security estimate (parameters a search already accepted) |
//! | `SINGLE_TIME` | 0 | `test_gpu_pe_time_window`: 1 encrypts and decrypts once, at a random accepted time of `[0900, 1759]`, instead of every vector of `TIME_VECTORS` |
//! | `ESTIMATE_FROM` | unset | `test_gpu_pe_time_window`: `encrypt` runs setup and key generation, then only plans the encryption and the online evaluation (`MXX_EXPONENT_LUT_ESTIMATE_ONLY` from then on); `keygen` runs setup only and plans every later stage |
//! | `DISCARD_CIPHERTEXTS` | 0 | `test_gpu_pe_time_window` with `file` storage: 1 deletes each vector's ciphertext productions after its decryption, so the store holds one ciphertext at a time |
//!
//! With `MXX_EXPONENT_LUT_ESTIMATE_ONLY=1` (see `mxx_exponent_lut::gpu_execution`),
//! `test_gpu_pe_time_window` plans every stage and logs its estimate without executing the stages
//! with artifact I/O; it stops after planning the first vector's online evaluation.

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime, RuntimeValue, SessionStore,
    backend::poly_gpu::gpu_backend_on,
    gpu_cost_model::gpu_operation_counts,
    matrix::dcrt_poly::DCRTPolyMatrix,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
};
use mxx_dsl::{DslContext, IdealSpec, Int};
use mxx_exponent_lut::{
    api::{ApiError, StageExecution, StageOutputs, StageStorage, StageStore},
    program::ProgramData,
    utils::{env_usize, ring_dimension_candidates},
};
use mxx_ir_core::{
    FrozenGraphScopeId, Graph, ParamEnv, ValidatedGraph, WireRef,
    artifact::ProductionId,
    concretize_wire_type,
    node::NodeKind,
    protocol::{
        ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorEndpointBinding,
        ComparatorSpec, EndpointBinding, EndpointBindings, EndpointSemanticBinding, EndpointSpecId,
        InputContract, InputContractEntry, InputValueContract, OutputRef, ProtocolDecl,
        ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec,
        ProtocolStage, PurePredicateSpec, StageId, StageInputName, Workflow,
    },
    types::{CoefficientBoundDomain, ConcreteMatrixType, ConcreteWireType},
};
use mxx_pe::{
    BranchingProgram, PeClass, PeNoiseReport, PeParameterRequest, PeParameters, decrypt, encrypt,
    keygen, setup,
};
use num_bigint::BigInt;
use serial_test::serial as sequential;
use std::{collections::BTreeMap, env, path::PathBuf, time::Instant};

const PREFIX: &str = "MXX_PE_ROUND_TRIP_";
fn variable(name: &str) -> String {
    format!("{PREFIX}{name}")
}

fn decimal(name: &str, default: &str) -> BigDecimal {
    let name = variable(name);
    env::var(&name)
        .unwrap_or_else(|_| default.to_owned())
        .parse()
        .unwrap_or_else(|error| panic!("{name} must be a decimal: {error}"))
}

fn count(name: &str, default: usize) -> usize {
    env_usize(&variable(name), default).unwrap()
}

/// The failure exponent of the configured noise model: `None` for `worst`.
fn noise_model() -> Option<u32> {
    match env::var(variable("NOISE_MODEL")).as_deref() {
        Err(env::VarError::NotPresent) | Ok("worst") => None,
        Ok("average") => Some(u32::try_from(count("FAILURE_EXPONENT", 128)).unwrap()),
        Ok(value) => {
            panic!("{} must be `worst` or `average`, got `{value}`", variable("NOISE_MODEL"))
        }
        Err(error) => panic!("{} is not valid Unicode: {error}", variable("NOISE_MODEL")),
    }
}

fn lean_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lean")
}

/// The parameter request of `class` from the environment, with the ring-dimension candidates.
fn request(class: PeClass) -> (PeParameterRequest, Vec<u32>) {
    let crt_bits = count("CRT_BITS", 50);
    let base_bits = u32::try_from(count("BASE_BITS", crt_bits.div_ceil(2))).unwrap();
    let max_depth = count("MAX_CRT_DEPTH", 12);
    let security = u64::try_from(count("SECURITY_BITS", 128)).unwrap();
    assert!(security > 0, "{} must be positive", variable("SECURITY_BITS"));
    let trapdoor_sigma = decimal("TRAPDOOR_SIGMA", "4.578").to_string().parse::<f64>().unwrap();
    let dimensions = ring_dimension_candidates(
        &variable("MIN_LOG_RING_DIMENSION"),
        &variable("MAX_LOG_RING_DIMENSION"),
        12,
        15,
    )
    .unwrap();
    let request = PeParameterRequest {
        class,
        ring_dimension: dimensions[0],
        crt_bits,
        base_bits,
        crt_depths: 1..=max_depth,
        trapdoor_sigma,
        error_sigma: decimal("ERROR_SIGMA", "3.19"),
        lut_radix: match env::var(variable("LUT_RADIX")) {
            Err(_) => class.default_lut_radix(),
            Ok(_) => Some(count("LUT_RADIX", 0)).filter(|radix| *radix > 0),
        },
        compact_encodings: count("COMPACT_ENCODINGS", 1) != 0,
        failure_exponent: noise_model(),
        lattice_security_bits: Some(security),
    };
    (request, dimensions)
}

/// The first accepted candidate over the ring dimensions, with its noise report.
fn search(class: PeClass) -> (PeParameters, PeNoiseReport) {
    let (request, dimensions) = request(class);
    PeParameters::select_dimension(&request, dimensions)
        .unwrap_or_else(|error| panic!("no parameter candidate was accepted: {error}"))
}

/// With `CRT_DEPTH`, the parameters at that depth and the first ring dimension, built without
/// the noise simulation and the security estimate (`PeParameters::at_depth`): a run of
/// parameters a search already accepted.
fn fixed_parameters(class: PeClass) -> Option<PeParameters> {
    let depth = env_usize(&variable("CRT_DEPTH"), 0).unwrap();
    (depth > 0).then(|| {
        let (request, _) = request(class);
        PeParameters::at_depth(&request, depth).expect("parameters at CRT_DEPTH")
    })
}

/// A random program of the class's maximal length whose output on `attribute` is `value`:
/// random reads and intermediate tables, and a random final table with its reached entry set.
fn program(class: PeClass, attribute: &[usize], value: usize) -> BranchingProgram {
    let random = |bound: usize| rand::random::<u32>() as usize % bound;
    let entries = class.state_width() * class.alphabet_size();
    let steps = class.max_steps();
    let reads = (0..steps).map(|_| random(class.attribute_length())).collect::<Vec<_>>();
    let mut tables = (0..steps)
        .map(|step| {
            let bound = if step + 1 == steps { 2 } else { class.state_width() };
            (0..entries).map(|_| random(bound)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut state = 0;
    for (read, table) in reads.iter().zip(&tables).take(steps - 1) {
        state = table[state + class.state_width() * attribute[*read]];
    }
    tables[steps - 1][state + class.state_width() * attribute[reads[steps - 1]]] = value;
    let program = BranchingProgram::new(class, reads, tables).unwrap();
    assert_eq!(program.evaluate(attribute).unwrap(), value);
    program
}

/// One executed graph: its stage graph, its compile-parameter bindings, and, for a producing
/// stage, its production.
struct Recorded {
    graph: Graph,
    bindings: ParamEnv,
    production: Option<ProductionId>,
}

/// An execution that records every graph it runs, in order.
struct Recording<'a, E> {
    inner: &'a mut E,
    graphs: Vec<Recorded>,
}

impl<S: SessionStore + Send, E: StageExecution<S>> StageExecution<S> for Recording<'_, E> {
    fn produce(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<ProductionId, ApiError> {
        let (graph, bindings) = (validated.source.clone(), validated.bindings.clone());
        let production = self.inner.produce(validated, inputs, store, execution_nonce)?;
        self.graphs.push(Recorded { graph, bindings, production: Some(production.clone()) });
        Ok(production)
    }

    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<StageOutputs, ApiError> {
        let (graph, bindings) = (validated.source.clone(), validated.bindings.clone());
        let outputs = self.inner.evaluate(validated, inputs, store, execution_nonce)?;
        self.graphs.push(Recorded { graph, bindings, production: None });
        Ok(outputs)
    }

    fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, ApiError> {
        StageExecution::<S>::download_matrix(&*self.inner, value)
    }

    fn download_integers(&self, value: &RuntimeValue) -> Result<Vec<BigInt>, ApiError> {
        StageExecution::<S>::download_integers(&*self.inner, value)
    }
}

/// `EXPORT_ONLY=1`: build and export the claim without executing the workflow.
fn export_only() -> bool {
    env::var(variable("EXPORT_ONLY")).is_ok_and(|value| value == "1")
}

/// An execution that runs only the payload secret's sampling, on the CPU. Every other graph is
/// validated by its caller and not executed: a producing stage returns the production its
/// execution would have (which depends only on the validated graph and the nonce), and an
/// evaluated stage returns the integer `0` for each of its outputs, so the stages after it can
/// build their graphs.
struct GraphOnly(mxx_backends::authority::CpuExecution);

impl StageExecution<StageStore> for GraphOnly {
    fn produce(
        &mut self,
        validated: ValidatedGraph,
        _: BTreeMap<String, RuntimeValue>,
        _: &mut StageStore,
        execution_nonce: [u8; 32],
    ) -> Result<ProductionId, ApiError> {
        let specification =
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .expect("stage specification hash");
        Ok(mxx_ir_core::artifact::production_id(specification, execution_nonce))
    }

    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut StageStore,
        execution_nonce: [u8; 32],
    ) -> Result<StageOutputs, ApiError> {
        if validated.source.name() == "rns-bgg-secret" {
            return self.0.evaluate(validated, inputs, store, execution_nonce);
        }
        Ok(StageOutputs {
            outputs: validated
                .source
                .outputs()
                .keys()
                .map(|name| (name.clone(), RuntimeValue::Int(BigInt::from(0))))
                .collect(),
        })
    }

    fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, ApiError> {
        StageExecution::<StageStore>::download_matrix(&self.0, value)
    }

    fn download_integers(&self, value: &RuntimeValue) -> Result<Vec<BigInt>, ApiError> {
        StageExecution::<StageStore>::download_integers(&self.0, value)
    }
}

/// The stage of every graph the authorized workflow executes, by graph name.
const STAGES: [(&str, &str); 9] = [
    ("mxx-pe/setup-trapdoor", "setup_trapdoor"),
    ("mxx-pe/setup-public", "setup_public"),
    ("rns-bgg-secret", "payload"),
    ("mxx-pe/encrypt-prelude", "prelude"),
    ("rns-bgg-encrypt", "encrypt"),
    ("rns-bgg-preprocess", "preprocess"),
    ("mxx-pe/keygen", "keygen"),
    ("rns-bgg-online-eval", "online_eval"),
    ("mxx-pe/decrypt", "decode"),
];

/// The size of artifacts over the scheme's rings: `packed_bits` at `ceil(log2 q)` bits per `R_q`
/// coefficient (`ceil(log2(2B + 1))` for a bounded matrix of coefficient bound `B`), and
/// `stored_bytes` as the backends store them: one 64-bit word per CRT residue of a matrix, and the
/// small matrix form of a bounded matrix (a sign byte and the bound's magnitude bytes per
/// coefficient, per CRT limb for a per-limb bound). Preimages are bounded matrices.
#[derive(Clone, Copy, Debug, Default)]
struct ArtifactSize {
    packed_bits: u128,
    stored_bytes: u128,
}

impl std::ops::AddAssign for ArtifactSize {
    fn add_assign(&mut self, other: Self) {
        self.packed_bits += other.packed_bits;
        self.stored_bytes += other.stored_bytes;
    }
}

impl std::ops::Mul<u128> for ArtifactSize {
    type Output = Self;
    fn mul(self, count: u128) -> Self {
        Self { packed_bits: self.packed_bits * count, stored_bytes: self.stored_bytes * count }
    }
}

/// The [`ArtifactSize`] of one value of type `ty`. Trapdoors and typed blobs count zero: only the
/// master secret key holds a trapdoor, and no measured object holds a blob.
fn wire_size(ty: &ConcreteWireType) -> ArtifactSize {
    let coefficients = |matrix: &ConcreteMatrixType| {
        (matrix.rows * matrix.columns) as u128 * u128::from(matrix.ring.ring_dimension())
    };
    match ty {
        ConcreteWireType::Matrix(matrix) => {
            let coefficients = coefficients(matrix);
            ArtifactSize {
                packed_bits: coefficients * u128::from(matrix.ring.modulus().bits()),
                stored_bytes: coefficients * matrix.ring.crt_depth() as u128 * 8,
            }
        }
        ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound, bound_domain } |
        ConcreteWireType::Preimage { matrix, max_coefficient_bound, bound_domain } => {
            let limbs = match bound_domain {
                CoefficientBoundDomain::Global => 1,
                CoefficientBoundDomain::PerCrtLimb => matrix.ring.crt_depth() as u128,
            };
            let coefficients = coefficients(matrix) * limbs;
            let bound = max_coefficient_bound.magnitude();
            ArtifactSize {
                packed_bits: coefficients * u128::from((bound * 2u8 + 1u8).bits()),
                stored_bytes: coefficients * (1 + bound.bits().div_ceil(8).max(1) as u128),
            }
        }
        ConcreteWireType::IndexedFamily { element, count } => wire_size(element) * *count as u128,
        ConcreteWireType::DependentFamily { members } => {
            members.iter().fold(ArtifactSize::default(), |mut total, member| {
                total += wire_size(member);
                total
            })
        }
        ConcreteWireType::Bytes { length } => {
            ArtifactSize { packed_bits: 8 * *length as u128, stored_bytes: *length as u128 }
        }
        ConcreteWireType::Trapdoor { .. } | ConcreteWireType::TypedBlob { .. } => {
            ArtifactSize::default()
        }
        _ => ArtifactSize { packed_bits: 64, stored_bytes: 8 },
    }
}

/// The total [`ArtifactSize`] of the outputs of `record`'s graph.
fn output_size(record: &Recorded) -> ArtifactSize {
    let scope = record.graph.root_scope();
    let mut total = ArtifactSize::default();
    for root in record.graph.outputs().values() {
        let WireRef { node, port } = root.value;
        let ty = &scope.nodes()[node.0 as usize].output_types()[port.0 as usize];
        let concrete = concretize_wire_type(ty, &record.bindings, &FrozenGraphScopeId::Root, node)
            .unwrap_or_else(|error| panic!("output type of {}: {error}", record.graph.name()));
        total += wire_size(&concrete);
    }
    total
}

/// The stage of `record` in [`STAGES`].
fn stage_of(record: &Recorded) -> &'static str {
    STAGES
        .iter()
        .find(|(name, _)| *name == record.graph.name())
        .map(|(_, stage)| *stage)
        .unwrap_or_else(|| panic!("unexpected executed graph {}", record.graph.name()))
}

/// Print, from the recorded graphs of one authorized workflow, the sizes of the public key (the
/// public setup production; the public seeds add 64 bytes), the ciphertext (the prelude and
/// encryption productions), the key's secret part (the key generation production, preimages in
/// small matrix form) and its public evaluation part (the preprocessing production), and the GPU
/// work of decryption (online evaluation and decoding) that [`gpu_operation_counts`] predicts
/// from the graphs (`MXX_GPU_NTT_RADIX`, 4 by default, one wave per parallel loop). The totals of
/// `mul64_instructions` and `gadget_decomposed_words` (the 64-bit RNS residues decomposed into
/// gadget digits: decomposed ring elements times `n` times the RNS primes the decomposition
/// keeps) are the "Mul" and "Decomp." of decryption.
fn report_measurements(graphs: &[Recorded]) {
    for (object, stages) in [
        ("public_key", &["setup_public"][..]),
        ("ciphertext", &["prelude", "encrypt"][..]),
        ("secret_key", &["keygen"][..]),
        ("evaluation_key", &["preprocess"][..]),
    ] {
        let mut size = ArtifactSize::default();
        for record in graphs.iter().filter(|record| stages.contains(&stage_of(record))) {
            size += output_size(record);
        }
        eprintln!(
            "PE_SIZE {object} packed_bytes={} stored_bytes={}",
            size.packed_bits.div_ceil(8),
            size.stored_bytes
        );
    }
    let radix =
        env::var("MXX_GPU_NTT_RADIX").ok().and_then(|value| value.parse().ok()).unwrap_or(4);
    let mut total = BTreeMap::<String, u128>::new();
    for record in
        graphs.iter().filter(|record| ["online_eval", "decode"].contains(&stage_of(record)))
    {
        let counts = gpu_operation_counts(&record.graph, &record.bindings, None, radix)
            .unwrap_or_else(|error| panic!("GPU counts of {}: {error}", record.graph.name()));
        eprintln!(
            "PE_DECRYPT_GPU_COUNT stage={} mul64_instructions={} gadget_decomposed_words={} \
             gadget_decompositions={}",
            stage_of(record),
            counts.get("mul64_instructions").copied().unwrap_or(0),
            counts.get("gadget_decomposed_words").copied().unwrap_or(0),
            counts.get("gadget_decompositions").copied().unwrap_or(0),
        );
        for (name, count) in counts {
            *total.entry(name).or_default() += count;
        }
    }
    for (name, count) in &total {
        eprintln!("PE_DECRYPT_GPU_COUNT {name}={count}");
    }
}

fn root_inputs(graph: &Graph) -> Vec<String> {
    graph
        .root_scope()
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::Input { name, artifact: None, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// The executed authorized workflow as a closed protocol: setup, payload sampling, the prelude
/// and the Dynamic encryption, the predicate's preprocessing and key generation, online
/// evaluation, and decryption, compared with the message. External inputs are the hash key, the
/// attribute exponents `Bz x_t`, the message bits, the fixed initial-state plaintext `X^0`, and
/// the program as the class Program's data (its reads and its `T_max` tables of `W` entries,
/// read by preprocessing and online evaluation); the requirement checks that the exponents
/// encode an attribute in the alphabet, that the program is one of the class (intermediate
/// outputs below `Bz`, final outputs bits), and that it outputs `0` on the attribute.
fn pe_protocol(
    recorded: Vec<Recorded>,
    parameters: &PeParameters,
    failure_exponent: Option<u32>,
) -> ProtocolDecl {
    let id = |name: &str| StageId(name.to_owned());
    let mut graphs = BTreeMap::new();
    let mut productions = BTreeMap::new();
    for record in recorded {
        let (_, stage) = STAGES
            .iter()
            .find(|(name, _)| *name == record.graph.name())
            .unwrap_or_else(|| panic!("unexpected executed graph {}", record.graph.name()));
        if let Some(production) = record.production {
            productions.insert(production, id(stage));
        }
        assert!(graphs.insert(*stage, record.graph).is_none(), "stage {stage} executed twice");
    }
    assert_eq!(graphs.len(), STAGES.len(), "every stage must execute once");

    let class = parameters.class();
    let dimension = parameters.level().ring_dimension() as usize;
    let attributes = class.attribute_length();
    let width = class.state_width();
    let lookup = class.lookup_width();
    let steps = class.max_steps();
    let compiled = class.compile().expect("class Program");
    let reads_input = ProgramData::input_name(compiled.reads());
    let tables_input = ProgramData::input_name(compiled.tables());
    let mut reads = Vec::new();
    let mut tables = Vec::new();
    let mut hash_key = Vec::new();
    let mut exponents = Vec::new();
    let mut message = Vec::new();
    let mut unit = Vec::new();
    let mut state = Vec::new();
    let mut stages = Vec::new();
    for (_, stage) in STAGES {
        let graph = &graphs[stage];
        let to = |input: &str| ProtocolInputDestination::WorkflowStage {
            stage: id(stage),
            input: StageInputName(input.to_owned()),
        };
        let mut bindings = Vec::new();
        for node in graph.root_scope().nodes() {
            if let NodeKind::Input { name, artifact: Some(artifact), .. } = node.kind() {
                bindings.push(ArtifactBinding {
                    consumer_input: StageInputName(name.clone()),
                    producer_stage: productions
                        .get(&artifact.production_id)
                        .unwrap_or_else(|| {
                            panic!("{stage} input {name} reads an unknown production")
                        })
                        .clone(),
                    producer_output: ArtifactName(artifact.artifact_name.clone()),
                });
            }
        }
        for name in root_inputs(graph) {
            let link = |producer: &str, output: String| ArtifactBinding {
                consumer_input: StageInputName(name.clone()),
                producer_stage: id(producer),
                producer_output: ArtifactName(output),
            };
            if name.starts_with("rns-bgg/secret/") {
                bindings.push(link("payload", "secret".to_owned()));
            } else if let Some(port) = name.strip_prefix("mxx-pe/output-row.") {
                bindings.push(link("online_eval", format!("output-0.{port}")));
            } else if name.ends_with("hash-key") {
                hash_key.push(to(&name));
            } else if name == "mxx-pe/attribute-exponents" {
                exponents.push(to(&name));
            } else if name == "mxx-pe/message-coefficients" {
                message.push(to(&name));
            } else if name.ends_with(".coefficient") {
                unit.push(to(&name));
            } else if name.ends_with(".exponent") {
                state.push(to(&name));
            } else if name == reads_input {
                reads.push(to(&name));
            } else if name == tables_input {
                tables.push(to(&name));
            } else {
                panic!("{stage} input {name} has no source");
            }
        }
        stages.push(ProtocolStage { id: id(stage), graph: graph.clone(), bindings });
    }

    // The requirement: every exponent is `Bz x` for a symbol `x` of the alphabet (the contract
    // bounds its range), every intermediate table entry is a state below `Bz` and every final
    // one a bit (the contracts bound the reads and the entries), and the program, evaluated on
    // the exponents, outputs `0`.
    let requirement = DslContext::new("pe-authorized-attribute");
    let family = requirement.int_family_input("exponents", attributes);
    let read_family = requirement.int_family_input("reads", steps);
    let table_family = requirement.int_family_input("tables", steps * lookup);
    // Each check is one loop, so the requirement does not grow with the class.
    let all = |count: usize, check: &dyn Fn(Int) -> mxx_dsl::Bool| {
        mxx_dsl::iterate(count, mxx_dsl::Bool::constant(true), |index, valid| {
            Ok(valid & check(index))
        })
        .unwrap()
    };
    let symbols =
        all(attributes, &|index| (family.at(index) % (width as i64)).equal(Int::constant(0)));
    let intermediate = all((steps - 1) * lookup, &|entry| {
        table_family.at(entry).less(Int::constant(width as i64))
    });
    let last = all(lookup, &|entry| {
        table_family.at(Int::constant(((steps - 1) * lookup) as i64) + entry).less(Int::constant(2))
    });
    let output = mxx_dsl::iterate(steps, Int::constant(0), |step, current| {
        let index = current + family.at(read_family.at(step.clone()));
        Ok(table_family.at(step * Int::constant(lookup as i64) + index))
    })
    .unwrap();
    let valid = symbols & intermediate & last & output.equal(Int::constant(0));
    let requirement =
        PurePredicateSpec::new(requirement.output("valid", valid).unwrap().build().unwrap().graph)
            .unwrap();
    // The ideal decryption is the message itself.
    let ideal = DslContext::new("pe-message");
    let bits = ideal.int_family_input("message", dimension);
    let ideal =
        IdealSpec::new(ideal.output("message", bits).unwrap().build().unwrap().graph).unwrap();

    let range = |lower: i64, upper: i64| InputValueContract::IntegerRange {
        lower: lower.into(),
        upper: upper.into(),
    };
    let external = |name: &str, value: InputValueContract, destinations| {
        (
            InputContractEntry { id: ProtocolInputId::from(name), name: name.to_owned(), value },
            ProtocolInputBinding { input: ProtocolInputId::from(name), destinations },
        )
    };
    exponents.push(ProtocolInputDestination::Requirement {
        requirement: 0,
        input: "exponents".to_owned(),
    });
    reads.push(ProtocolInputDestination::Requirement { requirement: 0, input: "reads".to_owned() });
    tables
        .push(ProtocolInputDestination::Requirement { requirement: 0, input: "tables".to_owned() });
    message.push(ProtocolInputDestination::Ideal { input: "message".to_owned() });
    let (contracts, bindings): (Vec<_>, Vec<_>) = [
        external("hash_key", InputValueContract::Bytes { length: 32.into() }, hash_key),
        external(
            "attribute_exponents",
            InputValueContract::Family {
                count: attributes.into(),
                element: Box::new(range(0, (width * (class.alphabet_size() - 1)) as i64)),
            },
            exponents,
        ),
        external(
            "message",
            InputValueContract::Family { count: dimension.into(), element: Box::new(range(0, 1)) },
            message,
        ),
        external("unit_coefficient", range(1, 1), unit),
        external("state_exponent", range(0, 0), state),
        external(
            "program_reads",
            InputValueContract::Family {
                count: steps.into(),
                element: Box::new(range(0, attributes as i64 - 1)),
            },
            reads,
        ),
        external(
            "program_tables",
            InputValueContract::Family {
                count: (steps * lookup).into(),
                element: Box::new(range(0, lookup as i64 - 1)),
            },
            tables,
        ),
    ]
    .into_iter()
    .unzip();

    let endpoint = EndpointSpecId::Exact;
    ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        bindings: Default::default(),
        failure_probability_log2: failure_exponent,
        bundle: ClosedProtocolBundle {
            workflow: Workflow { stages, entrypoint: id("decode") },
            ideal,
            requirements: vec![requirement],
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "mxx-pe/message".to_owned(),
                    ideal_input: "message".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef {
                        stage: id("decode"),
                        output: "mxx-pe/message".to_owned(),
                    },
                    ideal_output: "message".to_owned(),
                }],
            },
            operational_decoder_targets: Vec::new(),
            endpoint_specs: vec![endpoint],
            input_contract: InputContract { inputs: contracts },
            input_bindings: bindings,
            precondition_spec: ProtocolPreconditionSpec {
                requirement_outputs: vec!["valid".to_owned()],
            },
        },
    })
    .expect("valid PE protocol")
}

/// The stage storage selected by `STORAGE`, with the temporary directory a `file` store lives in.
fn storage() -> (StageStorage, Option<tempfile::TempDir>) {
    match env::var(variable("STORAGE")).as_deref() {
        Err(_) | Ok("device") => (StageStorage::Device, None),
        Ok("host") => (StageStorage::Host, None),
        Ok("file") => {
            let directory = tempfile::tempdir().expect("artifact directory");
            (StageStorage::File(directory.path().into()), Some(directory))
        }
        Ok(value) => match value.strip_prefix("file:") {
            Some(directory) => (StageStorage::File(directory.into()), None),
            None => panic!("unknown {} {value}", variable("STORAGE")),
        },
    }
}

/// Print the bytes of the artifacts a `device` store holds in GPU memory.
fn report_device_bytes(store: &StageStore) {
    if let StageStore::Memory(memory) = store {
        eprintln!("  artifacts in GPU memory: {} bytes", memory.device_resident_bytes());
    }
}

#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_pe_round_trip() {
    let class = PeClass::new(
        count("ATTRIBUTE_LENGTH", 1),
        count("ALPHABET_SIZE", 2),
        count("STATE_WIDTH", 1),
        count("MAX_STEPS", 1),
    )
    .unwrap();
    let (graphs, parameters, failure_exponent) =
        if export_only() { export_graphs(class, &program) } else { round_trip(class, &program) };
    if let Ok(id) = env::var(variable("LEAN_EXPORT")) {
        let directory = lean_directory().join(&id).join("generated");
        let protocol = pe_protocol(graphs, &parameters, failure_exponent);
        mxx_ir_core::lean::protocol::export(&protocol, &directory, true).expect("PE claim export");
        eprintln!("exported the PE correctness claim to {}", directory.display());
    }
}

/// The time window predicate of the implementation specification (`impl-spec-time-bp.md`): the key
/// of [`BranchingProgram::time_window`] for `[0900, 1759]` over [`PeClass::decimal_range`] with
/// four digits, and one encryption of a random message under each of the specification's test
/// vectors (see [`time_window_round_trip`]). The class is fixed, so the class variables are
/// ignored; the parameter variables apply. With `LEAN_EXPORT`, it exports the Lean correctness
/// claim of the authorized workflow (setup, the window's key generation, and the encryption and
/// decryption of the first vector, `0900`), as [`test_gpu_pe_round_trip`] does; with
/// `EXPORT_ONLY`, it builds the graphs without executing them.
#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_pe_time_window() {
    let window = BranchingProgram::time_window(900, 1759).unwrap();
    for (attribute, expected) in TIME_VECTORS {
        assert_eq!(window.evaluate(&attribute).unwrap(), expected, "{attribute:?}");
    }
    let (graphs, parameters, failure_exponent) = if export_only() {
        export_graphs(*window.class(), &|_, _, _| window.clone())
    } else {
        time_window_round_trip(&window)
    };
    if let Ok(id) = env::var(variable("LEAN_EXPORT")) {
        let directory = lean_directory().join(&id).join("generated");
        let protocol = pe_protocol(graphs, &parameters, failure_exponent);
        mxx_ir_core::lean::protocol::export(&protocol, &directory, true).expect("PE claim export");
        eprintln!("exported the PE correctness claim to {}", directory.display());
    }
}

/// The test vectors of the time window `[0900, 1759]` and their outputs `F(x)`: three accepted
/// times, then times below and above the window, minutes `60` and `75`, the symbol `A`, and hour
/// `24`.
const TIME_VECTORS: [([usize; 4], usize); 9] = [
    ([0, 9, 0, 0], 0),
    ([1, 7, 5, 9], 0),
    ([1, 2, 3, 0], 0),
    ([0, 8, 5, 9], 1),
    ([1, 8, 0, 0], 1),
    ([1, 2, 6, 0], 1),
    ([1, 0, 7, 5], 1),
    ([1, 10, 0, 0], 1),
    ([2, 4, 0, 0], 1),
];

/// Search the parameters of `window`'s class, then run on the GPU setup, key generation for
/// `window`, and one encryption of a random message under each vector of [`TIME_VECTORS`] with
/// its decryption. The accepted times must decrypt their messages; the rejected ones must
/// complete, and their rounded phases are not constrained. Returns the first execution of every
/// stage graph (setup, key generation, and the first vector's encryption and decryption: the
/// authorized workflow), the parameters, and the failure exponent of the noise model.
fn time_window_round_trip(window: &BranchingProgram) -> (Vec<Recorded>, PeParameters, Option<u32>) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let class = *window.class();
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "the PE time window round trip needs a CUDA device");
    let started = Instant::now();
    let (parameters, failure_exponent) = match fixed_parameters(class) {
        Some(parameters) => {
            eprintln!("fixed PE parameters (CRT_DEPTH): no noise simulation or security estimate");
            (parameters, noise_model())
        }
        None => {
            let (parameters, report) = search(class);
            assert!(report.decrypts(), "the accepted parameters must guarantee correctness");
            eprintln!(
                "noise: log2 E_F = {}, log2 E_PE = {}",
                report.evaluation.bits(),
                report.decryption.bits()
            );
            (parameters, report.average.as_ref().map(|(exponent, ..)| *exponent))
        }
    };
    let level = parameters.level();
    eprintln!(
        "PE parameters in {:?}: n = {}, CRT {} x {} bits, base 2^{}, m_B = {}, W = {}, \
         T_max = {}; noise model {}",
        started.elapsed(),
        level.ring_dimension(),
        level.to_crt().0.len(),
        level.crt_bits(),
        level.base_bits(),
        parameters.anchor_columns(),
        class.lookup_width(),
        class.max_steps(),
        failure_exponent.map_or("worst".to_owned(), |exponent| format!("average 2^-{exponent}")),
    );
    let estimate = env::var("MXX_EXPONENT_LUT_ESTIMATE_ONLY").is_ok_and(|value| value == "1");
    let mut runtime = GpuRuntime::new(gpu_backend_on(
        parameters.rings().unwrap().into_iter().map(|ring| {
            GpuDCRTPolyParams::new(ring.ring_dimension(), ring.to_crt().0, ring.base_bits(), None)
        }),
        devices,
    ))
    .expect("GPU runtime");
    let (storage, _directory) = storage();
    let discard = count("DISCARD_CIPHERTEXTS", 0) != 0;
    let productions = match (&storage, discard) {
        (StageStorage::File(root), true) => Some(root.join("productions")),
        (_, false) => None,
        _ => panic!("{} needs file storage", variable("DISCARD_CIPHERTEXTS")),
    };
    let mut store = storage.open().expect("stage storage");
    let mut execution = Recording { inner: &mut runtime, graphs: Vec::new() };

    let stage = Instant::now();
    let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
    eprintln!("setup: {:?}", stage.elapsed());
    report_device_bytes(&store);
    // `ESTIMATE_FROM=keygen`: setup runs, so key generation, which reads its artifacts, can be
    // planned; it and every later stage are then only planned.
    let estimate_from = env::var(variable("ESTIMATE_FROM"));
    if estimate_from.as_deref() == Ok("keygen") {
        // SAFETY: set before the key generation stages read it; this test is the only thread
        // touching the environment.
        unsafe { env::set_var("MXX_EXPONENT_LUT_ESTIMATE_ONLY", "1") };
        eprintln!("estimate mode from key generation on: the remaining stages are only planned");
    }
    let stage = Instant::now();
    let key = keygen(&public_key, &master, window, &mut execution, &mut store).unwrap();
    eprintln!("key generation ([0900, 1759]): {:?}", stage.elapsed());
    report_device_bytes(&store);
    // `ESTIMATE_FROM=encrypt`: setup and key generation run, so the encryption and online
    // evaluation that read their artifacts can be planned; those are then only planned.
    let estimate = estimate ||
        match estimate_from.as_deref() {
            Err(_) => false,
            Ok("keygen") => true,
            Ok("encrypt") => {
                // SAFETY: set before the encryption stages read it; this test is the only thread
                // touching the environment.
                unsafe { env::set_var("MXX_EXPONENT_LUT_ESTIMATE_ONLY", "1") };
                eprintln!(
                    "estimate mode from encryption on: the remaining stages are only planned"
                );
                true
            }
            Ok(value) => {
                panic!("{} must be `keygen` or `encrypt`, got `{value}`", variable("ESTIMATE_FROM"))
            }
        };
    let listing = |directory: &PathBuf| -> std::collections::BTreeSet<PathBuf> {
        std::fs::read_dir(directory)
            .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
            .unwrap_or_default()
    };
    // `SINGLE_TIME=1`: one encryption and decryption, at a random accepted time of the window.
    let vectors = if count("SINGLE_TIME", 0) != 0 {
        let minute = 9 * 60 + rand::random::<u32>() as usize % (18 * 60 - 9 * 60);
        let (hour, minute) = (minute / 60, minute % 60);
        let attribute = [hour / 10, hour % 10, minute / 10, minute % 10];
        assert_eq!(window.evaluate(&attribute).unwrap(), 0, "{attribute:?} is accepted");
        vec![(attribute, 0)]
    } else {
        TIME_VECTORS.to_vec()
    };
    for (attribute, expected) in vectors {
        let message =
            (0..level.ring_dimension()).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
        let before = productions.as_ref().map(listing);
        let stage = Instant::now();
        let ciphertext =
            encrypt(&public_key, &attribute, &message, &mut execution, &mut store).unwrap();
        eprintln!("encryption of {attribute:?}: {:?}", stage.elapsed());
        let stage = Instant::now();
        let decrypted = decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store);
        if estimate {
            // The online evaluation is planned but not executed, so decoding has no inputs.
            eprintln!("estimate mode: decryption stopped after planning: {:?}", decrypted.err());
            break;
        }
        let decrypted = decrypted.unwrap();
        eprintln!("decryption of {attribute:?} (F(x) = {expected}): {:?}", stage.elapsed());
        report_device_bytes(&store);
        if expected == 0 {
            assert_eq!(decrypted, message, "the time {attribute:?} must decrypt the message");
        }
        // `DISCARD_CIPHERTEXTS=1`: delete the productions this vector's encryption wrote.
        if let (Some(directory), Some(before)) = (&productions, before) {
            for created in listing(directory).difference(&before) {
                std::fs::remove_dir_all(created).expect("discard a ciphertext production");
            }
        }
    }
    // The first execution of every graph: the later vectors repeat encryption and decryption.
    let mut seen = std::collections::BTreeSet::new();
    let graphs: Vec<Recorded> = std::mem::take(&mut execution.graphs)
        .into_iter()
        .filter(|record| seen.insert(record.graph.name().to_owned()))
        .collect();
    report_measurements(&graphs);
    (graphs, parameters, failure_exponent)
}

/// Search the parameters of `class`, then run setup, the encryption of a random attribute, and
/// key generation and decryption for the programs `program(class, x, 0)` (which must decrypt)
/// and `program(class, x, 1)` on the GPU. Returns the graphs of the authorized round trip, the
/// parameters, and the failure exponent of the noise model.
fn round_trip(
    class: PeClass,
    program: &dyn Fn(PeClass, &[usize], usize) -> BranchingProgram,
) -> (Vec<Recorded>, PeParameters, Option<u32>) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "the PE round trip needs a CUDA device");
    let started = Instant::now();
    let (parameters, report) = search(class);
    let failure_exponent = report.average.as_ref().map(|(exponent, ..)| *exponent);
    assert!(report.decrypts(), "the accepted parameters must guarantee correctness");
    let security_bits = parameters.security_bits().unwrap();
    let level = parameters.level();
    eprintln!(
        "accepted PE parameters in {:?}: n = {}, CRT {} x {} bits (log2 q = {}), base 2^{}, \
         m = {}, m_B = {}, W = {}, T_max = {}; noise model {}; log2 E_F = {}, log2 E_PE = {}; \
         security {security_bits} bits (encryption); \
         budget {:?}",
        started.elapsed(),
        level.ring_dimension(),
        level.to_crt().0.len(),
        level.crt_bits(),
        report.modulus.bits(),
        level.base_bits(),
        parameters.state_columns(),
        parameters.anchor_columns(),
        class.lookup_width(),
        class.max_steps(),
        failure_exponent.map_or("worst".to_owned(), |exponent| format!("average 2^-{exponent}")),
        report.evaluation.bits(),
        report.decryption.bits(),
        parameters.budget(),
    );

    let mut runtime = GpuRuntime::new(gpu_backend_on(
        parameters.rings().unwrap().into_iter().map(|ring| {
            GpuDCRTPolyParams::new(ring.ring_dimension(), ring.to_crt().0, ring.base_bits(), None)
        }),
        devices,
    ))
    .expect("GPU runtime");
    let (storage, _directory) = storage();
    let mut store = storage.open().expect("stage storage");
    let mut execution = Recording { inner: &mut runtime, graphs: Vec::new() };

    let stage = Instant::now();
    let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
    eprintln!("setup: {:?}", stage.elapsed());
    report_device_bytes(&store);
    let attribute = (0..class.attribute_length())
        .map(|_| rand::random::<u32>() as usize % class.alphabet_size())
        .collect::<Vec<_>>();
    let message = (0..level.ring_dimension()).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
    let stage = Instant::now();
    let ciphertext =
        encrypt(&public_key, &attribute, &message, &mut execution, &mut store).unwrap();
    eprintln!("encryption: {:?}", stage.elapsed());
    report_device_bytes(&store);
    let authorized = program(class, &attribute, 0);
    let stage = Instant::now();
    let key = keygen(&public_key, &master, &authorized, &mut execution, &mut store).unwrap();
    eprintln!("key generation (F(x) = 0): {:?}", stage.elapsed());
    report_device_bytes(&store);
    let stage = Instant::now();
    let decrypted = decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap();
    eprintln!("decryption (F(x) = 0): {:?}", stage.elapsed());
    report_device_bytes(&store);
    assert_eq!(decrypted, message, "the authorized key must decrypt the message");
    let graphs = std::mem::take(&mut execution.graphs);

    let unauthorized = program(class, &attribute, 1);
    let stage = Instant::now();
    let key = keygen(&public_key, &master, &unauthorized, &mut execution, &mut store).unwrap();
    eprintln!("key generation (F(x) = 1): {:?}", stage.elapsed());
    report_device_bytes(&store);
    let stage = Instant::now();
    decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap();
    eprintln!("decryption (F(x) = 1): {:?}", stage.elapsed());
    report_device_bytes(&store);
    report_measurements(&graphs);
    (graphs, parameters, failure_exponent)
}

/// [`round_trip`]'s authorized workflow as graphs only (see [`GraphOnly`]): the parameter search,
/// then setup, the encryption of a random attribute, and key generation and decryption for an
/// authorized program, none of them executed. The decryption of the placeholder outputs fails
/// after its graph is recorded.
fn export_graphs(
    class: PeClass,
    program: &dyn Fn(PeClass, &[usize], usize) -> BranchingProgram,
) -> (Vec<Recorded>, PeParameters, Option<u32>) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    assert!(
        env::var(variable("LEAN_EXPORT")).is_ok(),
        "{} needs {}",
        variable("EXPORT_ONLY"),
        variable("LEAN_EXPORT")
    );
    let (parameters, report) = search(class);
    let failure_exponent = report.average.as_ref().map(|(exponent, ..)| *exponent);
    assert!(report.decrypts(), "the accepted parameters must guarantee correctness");
    let level = parameters.level();
    eprintln!(
        "accepted PE parameters: n = {}, CRT {} x {} bits, base 2^{}, m_B = {}, W = {}, T_max = {}; \
         log2 E_F = {}, log2 E_PE = {}",
        level.ring_dimension(),
        level.to_crt().0.len(),
        level.crt_bits(),
        level.base_bits(),
        parameters.anchor_columns(),
        class.lookup_width(),
        class.max_steps(),
        report.evaluation.bits(),
        report.decryption.bits(),
    );
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = StageStorage::File(directory.path().into()).open().expect("stage storage");
    let mut graph_only = GraphOnly(mxx_backends::authority::CpuExecution::new(
        mxx_backends::backend::poly::cpu_backend(parameters.rings().unwrap()),
    ));
    let mut execution = Recording { inner: &mut graph_only, graphs: Vec::new() };
    let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
    let attribute = (0..class.attribute_length())
        .map(|_| rand::random::<u32>() as usize % class.alphabet_size())
        .collect::<Vec<_>>();
    let message = (0..level.ring_dimension()).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
    let ciphertext =
        encrypt(&public_key, &attribute, &message, &mut execution, &mut store).unwrap();
    let authorized = program(class, &attribute, 0);
    let key = keygen(&public_key, &master, &authorized, &mut execution, &mut store).unwrap();
    assert!(
        decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store).is_err(),
        "the export-only decryption graph is not executed"
    );
    let graphs = std::mem::take(&mut execution.graphs);
    report_measurements(&graphs);
    (graphs, parameters, failure_exponent)
}
