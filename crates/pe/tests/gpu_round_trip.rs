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

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime, RuntimeValue, SessionStore,
    backend::poly_gpu::gpu_backend_on,
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
    Graph, ValidatedGraph,
    artifact::ProductionId,
    node::NodeKind,
    protocol::{
        ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorEndpointBinding,
        ComparatorSpec, EndpointBinding, EndpointBindings, EndpointSemanticBinding, EndpointSpecId,
        InputContract, InputContractEntry, InputValueContract, OutputRef, ProtocolDecl,
        ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec,
        ProtocolStage, PurePredicateSpec, StageId, StageInputName, Workflow,
    },
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

/// The first accepted candidate over the ring dimensions, with its noise report.
fn search(class: PeClass) -> (PeParameters, PeNoiseReport) {
    let crt_bits = count("CRT_BITS", 50);
    let base_bits = u32::try_from(count("BASE_BITS", crt_bits.div_ceil(2))).unwrap();
    let max_depth = count("MAX_CRT_DEPTH", 12);
    let security = u64::try_from(count("SECURITY_BITS", 128)).unwrap();
    assert!(security > 0, "{} must be positive", variable("SECURITY_BITS"));
    let trapdoor_sigma = decimal("TRAPDOOR_SIGMA", "4.578").to_string().parse::<f64>().unwrap();
    let failure_exponent = noise_model();
    let dimensions = ring_dimension_candidates(
        &variable("MIN_LOG_RING_DIMENSION"),
        &variable("MAX_LOG_RING_DIMENSION"),
        12,
        15,
    )
    .unwrap();
    {
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
            failure_exponent,
            lattice_security_bits: Some(security),
        };
        PeParameters::select_dimension(&request, dimensions)
            .unwrap_or_else(|error| panic!("no parameter candidate was accepted: {error}"))
    }
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

/// One executed graph: its stage graph and, for a producing stage, its production.
struct Recorded {
    graph: Graph,
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
        let graph = validated.source.clone();
        let production = self.inner.produce(validated, inputs, store, execution_nonce)?;
        self.graphs.push(Recorded { graph, production: Some(production.clone()) });
        Ok(production)
    }

    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<StageOutputs, ApiError> {
        let graph = validated.source.clone();
        let outputs = self.inner.evaluate(validated, inputs, store, execution_nonce)?;
        self.graphs.push(Recorded { graph, production: None });
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
    (std::mem::take(&mut execution.graphs), parameters, failure_exponent)
}
