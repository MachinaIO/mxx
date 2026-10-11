#![cfg(feature = "gpu")]

//! The predicate encryption round trip on the GPU runtime, at parameters whose noise guarantees
//! correctness and whose Ring-LWE instances reach a lattice security target.
//!
//! The parameter search tries the ring dimensions in order and, for each, the CRT depths through
//! [`PeParameters::select`]: a candidate is accepted when [`PeParameters::at_depth`] accepts it (for
//! the statistical instantiation, the budget, regularity, flooding, key-width, and numeric-range
//! conditions of `mxx_pe::budget`), the simulation of the class's longest program decrypts under
//! the selected noise model (`worst`: `4 E_PE + 2 < q`; `average`: failure probability at most
//! `2^-FAILURE_EXPONENT`, see `mxx_pe::noise::simulate_average`), and the Ring-LWE instances of
//! [`PeParameters::security_bits`] (the encryption secret, and TrapGen for the computational
//! instantiation) reach the target according to `lattice-estimator-cli`, which must be on `PATH`
//! with its Sage environment. The computational instantiation uses the samplers' default cutoffs
//! (`6.5 sigma`); the statistical one truncates `chi` and `chi_fl` at the budget's
//! multiplier. Both use the default preimage cutoff for the key.
//!
//! On the accepted parameters, every stage runs on the GPU: setup, one encryption of a random
//! attribute and a random binary message, and keys for two random branching programs of the
//! class's maximal length, adjusted to output `0` (authorized) and `1` on the attribute. The
//! authorized key must decrypt the message; the unauthorized decryption must complete, and its
//! rounded phase is not constrained.
//!
//! `test_gpu_pe_hamming_round_trip` runs the same search and round trip for the Hamming
//! threshold programs (`BranchingProgram::hamming_threshold`) of the class `PeClass::hamming` over
//! `HAMMING_BITS` attribute bits in symbols of `HAMMING_SYMBOL_BITS` bits, whose steps but the
//! last only add `popcount(x_i xor z_i)` to the state exponent, without a table, and whose last
//! step compares the sum with the threshold: the authorized key's threshold exceeds the distance
//! of the attribute to a random template, and the unauthorized key's equals it. The class variables
//! below do not apply to it.
//!
//! With `MXX_PE_ROUND_TRIP_LEAN_EXPORT=<parameter id>`, `test_gpu_pe_round_trip` exports the Lean
//! correctness claim of the executed graphs to `lean/<parameter id>/generated`. The claim does not
//! depend on the predicate: the program is an external input of key generation (preprocessing) and
//! online evaluation, and the requirement is that the attribute and the program form a valid
//! instance with `F(x) = 0`.
//!
//! The computational instantiation is checked only for correctness and security: the
//! manuscript's leftover-hash, noise-flooding, and key-width conditions are enforced only by the
//! statistical instantiation (see the `mxx-pe` crate documentation).
//!
//! | Variable (`MXX_PE_ROUND_TRIP_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `LEAN_EXPORT` | unset | Export the Lean correctness claim to `lean/<value>/generated` |
//! | `ATTRIBUTE_LENGTH`, `ALPHABET_SIZE`, `STATE_WIDTH`, `MAX_STEPS` | 1, 2, 1, 1 | The class |
//! | `HAMMING_BITS`, `HAMMING_SYMBOL_BITS` | 2, 1 | The attribute bits of the Hamming round trip and the bits per symbol |
//! | `MIN_LOG_RING_DIMENSION`, `MAX_LOG_RING_DIMENSION` | 12, 15 | Ring-dimension candidates |
//! | `MAX_CRT_DEPTH` | 12 | CRT-depth candidates `1..=` |
//! | `CRT_BITS`, `BASE_BITS` | 50, `ceil(CRT_BITS / 2)` | CRT prime and gadget base widths |
//! | `TRAPDOOR_INSTANTIATION` | `statistical` | `statistical` or `computational` |
//! | `KEY_WIDTH_POLICY` | `practical` | `practical` or `strict` (statistical only) |
//! | `ERROR_SIGMA`, `FLOODING_SIGMA` | 3.19 | Sigmas of `chi` and of `chi_fl` (computational, or statistical with `FLOODING=none`) |
//! | `COMPACT_ENCODINGS` | 1 | 1 publishes the fresh encodings in two columns over `pq` (`PeParameters::with_compact_encodings`), 0 at `q` in full gadget width |
//! | `LUT_RADIX` | `PeClass::default_lut_radix` | Radix `B` of the LUT automorphism decomposition (`PeParameters::with_lut_radix`); 0 switches every automorphism directly (`B = W`). The default takes 3 or 2 switches per branch when they divide `log2 W` (`W = 4096`: `B = 16`; `W = 2`: direct) |
//! | `FLOODING` | `enforce` | `enforce` or `none`: whether the statistical instantiation imposes the flooding width |
//! | `RANDOMNESS_SIGMA` | 64 | Sigma of the proof's `R` (statistical only) |
//! | `STATISTICAL_BITS`, `FLOODING_BITS` | 30, 30 | `log(1/ε_total)` and the per-coefficient flooding exponent |
//! | `KEY_QUERIES`, `MAX_ANCHOR_COLUMNS` | `2^20`, 1024 | `Q_key` and `m_B,max` |
//! | `TRAPDOOR_SIGMA` | 4.578 | TrapGen sigma |
//! | `NOISE_MODEL`, `FAILURE_EXPONENT` | `worst`, 128 | Noise model of the correctness check |
//! | `SECURITY_BITS` | 128 | Required lattice security of both instances |
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
    BranchingProgram, PeClass, PeNoiseReport, PeParameterRequest, PeParameters,
    budget::{DEFAULT_FLOODING_BITS, DEFAULT_STATISTICAL_BITS, KeyWidthPolicy},
    decrypt, encrypt, keygen,
    params::TrapdoorMode,
    setup,
};
use num_bigint::BigInt;
use serial_test::serial as sequential;
use std::{collections::BTreeMap, env, path::PathBuf, time::Instant};

const PREFIX: &str = "MXX_PE_ROUND_TRIP_";
/// `Q_key` of the `Strict` budget.
const DEFAULT_KEY_QUERIES: u64 = 1 << 20;
/// `m_B,max`, which fixes the truncation multiplier.
const DEFAULT_MAX_ANCHOR_COLUMNS: usize = 1024;

fn key_width_policy(value: &str) -> KeyWidthPolicy {
    match value {
        "practical" => KeyWidthPolicy::Practical,
        "strict" => KeyWidthPolicy::Strict,
        other => panic!("unknown key width policy `{other}`"),
    }
}

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
    let trapdoor_instantiation = match env::var(variable("TRAPDOOR_INSTANTIATION")).as_deref() {
        Err(env::VarError::NotPresent) | Ok("statistical") => TrapdoorMode::Statistical,
        Ok("computational") => TrapdoorMode::Computational,
        Ok(value) => panic!(
            "{} must be `statistical` or `computational`, got `{value}`",
            variable("TRAPDOOR_INSTANTIATION")
        ),
        Err(error) => {
            panic!("{} is not valid Unicode: {error}", variable("TRAPDOOR_INSTANTIATION"))
        }
    };
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
            trapdoor_instantiation,
            key_width_policy: key_width_policy(
                &env::var(variable("KEY_WIDTH_POLICY")).unwrap_or_else(|_| "practical".into()),
            ),
            trapdoor_sigma,
            error_sigma: decimal("ERROR_SIGMA", "3.19"),
            flooding_sigma: decimal("FLOODING_SIGMA", "3.19"),
            enforce_flooding: match env::var(variable("FLOODING")).as_deref() {
                Err(_) | Ok("enforce") => true,
                Ok("none") => false,
                Ok(value) => {
                    panic!("{} must be `enforce` or `none`, got `{value}`", variable("FLOODING"))
                }
            },
            randomness_sigma: decimal("RANDOMNESS_SIGMA", "64"),
            statistical_security_bits: u32::try_from(count(
                "STATISTICAL_BITS",
                DEFAULT_STATISTICAL_BITS as usize,
            ))
            .unwrap(),
            flooding_statistical_bits: u32::try_from(count(
                "FLOODING_BITS",
                DEFAULT_FLOODING_BITS as usize,
            ))
            .unwrap(),
            key_queries: count("KEY_QUERIES", DEFAULT_KEY_QUERIES as usize) as u64,
            max_anchor_columns: count("MAX_ANCHOR_COLUMNS", DEFAULT_MAX_ANCHOR_COLUMNS),
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
struct Recording<'a> {
    inner: &'a mut GpuRuntime,
    graphs: Vec<Recorded>,
}

impl<S: SessionStore + Send> StageExecution<S> for Recording<'_> {
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
    let mut valid = (0..attributes)
        .map(|index| (family.at(index) % (width as i64)).equal(Int::constant(0)))
        .reduce(|left, right| left & right)
        .expect("a class has an attribute");
    for step in 0..steps {
        let bound = if step + 1 == steps { 2 } else { width };
        for entry in 0..lookup {
            valid =
                valid & table_family.at(step * lookup + entry).less(Int::constant(bound as i64));
        }
    }
    let mut current = Int::constant(0);
    for step in 0..steps {
        let index = current + family.at(read_family.at(step));
        current = table_family.at(Int::constant((step * lookup) as i64) + index);
    }
    valid = valid & current.equal(Int::constant(0));
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
    let (graphs, parameters, failure_exponent) = round_trip(class, &program);
    if let Ok(id) = env::var(variable("LEAN_EXPORT")) {
        let directory = lean_directory().join(&id).join("generated");
        let protocol = pe_protocol(graphs, &parameters, failure_exponent);
        mxx_ir_core::lean::protocol::export(&protocol, &directory).expect("PE claim export");
        eprintln!("exported the PE correctness claim to {}", directory.display());
    }
}

/// The round trip of the Hamming threshold programs ([`BranchingProgram::hamming_threshold`])
/// over `HAMMING_BITS` attribute bits in symbols of `HAMMING_SYMBOL_BITS` bits: every step but
/// the last only adds `popcount(x_i xor z_i)` to the exponent, and the last compares the distance
/// with the threshold. The authorized key's threshold exceeds the distance of the attribute to a
/// random template; the unauthorized key's equals it.
#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_pe_hamming_round_trip() {
    let symbol_bits = count("HAMMING_SYMBOL_BITS", 1);
    let bits = count("HAMMING_BITS", 2);
    assert_eq!(bits % symbol_bits, 0, "HAMMING_BITS must be a multiple of HAMMING_SYMBOL_BITS");
    let symbol_bits = u32::try_from(symbol_bits).unwrap();
    let class = PeClass::hamming(bits / symbol_bits as usize, symbol_bits).unwrap();
    let program = move |attribute: &[usize], value: usize| {
        let template = attribute
            .iter()
            .map(|_| rand::random::<u32>() as usize % (1 << symbol_bits))
            .collect::<Vec<_>>();
        let distance =
            attribute.iter().zip(&template).map(|(x, z)| (x ^ z).count_ones() as usize).sum();
        let threshold = if value == 0 { distance + 1 } else { distance };
        let program =
            BranchingProgram::hamming_threshold(symbol_bits, &template, threshold).unwrap();
        assert_eq!(program.evaluate(attribute).unwrap(), value);
        program
    };
    round_trip(class, &move |_, attribute, value| program(attribute, value));
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
    let (encryption_bits, trapdoor_bits) = parameters.security_bits().unwrap();
    let level = parameters.level();
    eprintln!(
        "accepted PE parameters in {:?}: n = {}, CRT {} x {} bits (log2 q = {}), base 2^{}, \
         m = {}, m_B = {}, W = {}, T_max = {}; noise model {}; log2 E_F = {}, log2 E_PE = {}; \
         security {encryption_bits} bits (encryption), {trapdoor_bits:?} bits (trapdoor); \
         statistical selection {:?}",
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
        parameters.statistical(),
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
