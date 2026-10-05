#![cfg(feature = "gpu")]

//! The application-facing RNS public inner-product lifecycle.
//!
//! The Program of `PublicInnerProductProgram` multiplies the public exponent encodings of the
//! coordinates of `a in Z_Q^m` selected by the hash-sampled `t in {0,1}^m`, so its public output
//! encodes `X^(stride <a, t>)`. With `MXX_EXPONENT_LUT_PUBLIC_IP_PARAMETER_ID`, the workflow runs
//! the recorded parameter set of that ID in `lean/public_ip/parameters.csv` instead of searching,
//! and exports the Lean correctness claim of the executed graphs to
//! `lean/public_ip/<parameter id>/generated`.

mod common;

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime,
    backend::poly_gpu::{GpuDcrtBackend, gpu_backend_on},
    poly::{
        PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
};
use mxx_dsl::{DslContext, Family, FamilyType, IdealSpec, Int, IntType};
use mxx_exponent_lut::{
    ExpLayout, api,
    noise_refresh::simulation::AverageCaseConfig,
    program::{AttributeKnowledge, FamilyRange, InputBinding},
    program_gadgets::public_ip::PublicInnerProductProgram,
    rns_bgg::{
        RhsMode, RnsBggDerivedWorkload, RnsBggMaskSecret, RnsBggNoiseBindings, RnsBggNoiseFamily,
        RnsBggNoiseValue, RnsBggParameters, RnsBggPayloadSecret, RnsBggPlanRequest,
        RnsEncryptBuild, RnsNoiseBuild, RnsOnlineEvalBuild, RnsPreprocessBuild,
        decode_public_output, verify_output,
    },
    utils::{
        audit_program_at_crt_depth, env_usize, lattice_security_bits, noise_model,
        ring_dimension_candidates,
    },
};
use mxx_ir_core::{
    Graph, IntExpr, ParamEnv, ValidatedGraph,
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputValueContract, OutputRef, ProtocolDecl, ProtocolInputDestination,
        ProtocolPreconditionSpec, PurePredicateSpec, StageId, Workflow,
    },
};
use num_bigint::{BigInt, BigUint};
use rand::{Rng, SeedableRng};
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, env};

const HASH_KEY: &[u8; 32] = b"public-ip-rns-public-api-hashkey";
const LABEL: &[u8] = b"public-ip-integration";
const PREFIX: &str = "MXX_EXPONENT_LUT_PUBLIC_IP_";

type ParameterRow = BTreeMap<String, String>;

/// The recorded parameter set selected by `MXX_EXPONENT_LUT_PUBLIC_IP_PARAMETER_ID`, if any.
fn recorded_parameters() -> Option<(String, ParameterRow)> {
    env::var(format!("{PREFIX}PARAMETER_ID"))
        .ok()
        .map(|id| (id.clone(), common::parameter_row("public_ip", &id)))
}

/// Column `column` of the recorded row, or else the variable `{PREFIX}{name}` with `default`.
fn setting(recorded: Option<&ParameterRow>, column: &str, name: &str, default: usize) -> usize {
    match recorded {
        Some(row) => common::column(row, column),
        None => {
            let name = format!("{PREFIX}{name}");
            env_usize(&name, default).unwrap_or_else(|error| panic!("{name}: {error}"))
        }
    }
}

/// Graphs of the executed stages, recorded for the exported claim.
type StageGraphs = RefCell<BTreeMap<&'static str, Graph>>;

/// A stage observer that records the validated graph of `stage`.
fn record<'a>(
    graphs: &'a StageGraphs,
    stage: &'static str,
) -> impl FnOnce(&ValidatedGraph) -> Result<(), api::ApiError> + 'a {
    move |validated| {
        graphs.borrow_mut().insert(stage, validated.source.clone());
        Ok(())
    }
}

fn error_sigma(recorded: Option<&ParameterRow>) -> BigDecimal {
    if let Some(row) = recorded {
        return common::column(row, "error_sigma");
    }
    let sigma = env::var(format!("{PREFIX}ERROR_SIGMA"))
        .unwrap_or_else(|_| "4".to_owned())
        .parse()
        .expect("MXX_EXPONENT_LUT_PUBLIC_IP_ERROR_SIGMA must be a decimal");
    assert!(sigma > 0u8, "the encoding error width must be positive");
    sigma
}

/// The public inner-product Program for ring dimension `ring_dimension`: the input modulus `Q`
/// fixes the exponent layout (`X^(2n / Q * z)` adds exponents modulo `Q`), and the seed fixes `a`
/// and the hash seed of `t`.
fn build_public_ip(
    ring_dimension: u32,
    recorded: Option<&ParameterRow>,
) -> PublicInnerProductProgram {
    let input_modulus = setting(recorded, "input_modulus", "INPUT_MODULUS", 32);
    assert!(
        input_modulus >= 4 && input_modulus.is_power_of_two(),
        "the input modulus must be a power of two at least 4"
    );
    let width_log2 = input_modulus.ilog2() - 1;
    let stride_log2 = ring_dimension.ilog2().checked_sub(width_log2).unwrap_or_else(|| {
        panic!("input modulus {input_modulus} exceeds twice the ring dimension {ring_dimension}")
    });
    let dimension = setting(recorded, "dimension", "DIMENSION", 8);
    assert!(dimension > 0, "the vector dimension must be positive");
    // `a` and the hash seed of `t` come from the recorded seed, `{PREFIX}SEED`, or a fresh seed
    // that is reported, so that any run can be repeated.
    let seed = match (recorded, env::var(format!("{PREFIX}SEED"))) {
        (Some(row), _) => common::column::<u64>(row, "seed"),
        (None, Ok(value)) => value.parse::<u64>().expect("the seed must be a u64"),
        (None, Err(_)) => rand::random(),
    };
    eprintln!("public inner-product seed: {PREFIX}SEED={seed}");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let vector = (0..dimension).map(|_| rng.random_range(0..input_modulus)).collect();
    let hash_seed = rng.random::<[u8; 32]>();
    let public_ip = PublicInnerProductProgram::new(
        ExpLayout { stride_log2, width_log2 },
        vector,
        hash_seed,
        LABEL,
    )
    .expect("public inner-product program");
    eprintln!(
        "public inner-product program: n={ring_dimension} Q={input_modulus} m={dimension} a={:?} t={:?} <a,t>={}",
        public_ip.vector(),
        public_ip.selection().iter().map(|bit| u8::from(*bit)).collect::<Vec<_>>(),
        public_ip.inner_product()
    );
    public_ip
}

/// Every member of the public input family is a separate encryption.
fn noise_bindings(
    public_ip: &PublicInnerProductProgram,
    sigma: &BigDecimal,
) -> RnsBggNoiseBindings {
    let mut inputs = BTreeMap::new();
    for (index, (wire, declaration)) in
        public_ip.program().encryption_inputs().inputs().iter().enumerate()
    {
        let InputBinding::Family(count) = declaration.shape() else {
            panic!("the public inner product reads one input family");
        };
        assert_eq!(declaration.knowledge(), AttributeKnowledge::PublicPayload);
        let value = RnsBggNoiseValue::sampled_encoding_sigma(
            sigma,
            format!("public-ip-input-{index}").as_bytes(),
            &[BigInt::from(1u8)],
        )
        .expect("valid encoding sigma");
        inputs.insert(
            *wire,
            InputBinding::Family(RnsBggNoiseFamily {
                element: value.independent_family_members(),
                range: FamilyRange::full(count.clone()).expect("valid input family range"),
                max_nonzero_payloads: 1u8.into(),
            }),
        );
    }
    RnsBggNoiseBindings::new(inputs, BTreeMap::new()).expect("declaration-order noise bindings")
}

/// Search the configured candidates, or check the recorded parameter set `recorded`.
fn search(
    sigma: &BigDecimal,
    recorded: Option<&ParameterRow>,
) -> (RnsBggParameters, DCRTPolyParams, PublicInnerProductProgram) {
    let required_security = env_usize(&format!("{PREFIX}SECURITY_BITS"), 1)
        .expect("security bits configuration") as u64;
    assert!(required_security > 0, "the security target must be positive");
    let crt_bits = setting(recorded, "crt_bits", "CRT_BITS", 40);
    let base_bits = setting(recorded, "base_bits", "BASE_BITS", crt_bits.div_ceil(2));
    assert!((1..=31).contains(&base_bits), "the base bits must be in 1..=31");
    let max_crt_depth =
        env_usize(&format!("{PREFIX}MAX_CRT_DEPTH"), 16).expect("maximum CRT depth configuration");
    let average = match recorded {
        Some(row) => match row["noise_model"].as_str() {
            "worst" => None,
            "average" => Some(AverageCaseConfig {
                failure_exponent: common::column(row, "failure_exponent"),
                allow_average_acceptance: true,
                ..AverageCaseConfig::default()
            }),
            model => panic!("recorded noise model {model} is neither worst nor average"),
        },
        None => noise_model(PREFIX).expect("noise-model configuration"),
    };
    let (candidates, depths) = match recorded {
        Some(row) => (
            vec![common::column::<u32>(row, "ring_dimension")],
            vec![common::column::<usize>(row, "crt_depth")],
        ),
        None => (
            ring_dimension_candidates(
                &format!("{PREFIX}MIN_LOG_RING_DIMENSION"),
                &format!("{PREFIX}MAX_LOG_RING_DIMENSION"),
                5,
                16,
            )
            .expect("ring-dimension configuration"),
            (1..=max_crt_depth).collect(),
        ),
    };
    let mut failures = Vec::new();
    for ring_dimension in candidates {
        let public_ip = build_public_ip(ring_dimension, recorded);
        let noise = noise_bindings(&public_ip, sigma);
        // The output plaintext is a signed monomial.
        let terminal_bounds = vec![BigUint::from(1u8)];
        for &crt_depth in &depths {
            let request = RnsBggPlanRequest {
                ring_dimension,
                encoding_error_sigma: sigma.clone(),
                crt_bits,
                base_bits: base_bits as u32,
                terminal_depth: 1,
                max_crt_depth: crt_depth,
                terminal_value_bound: BigUint::from(1u8),
                operations_per_drop: 1,
                primes_per_drop: 1,
                rhs_mode: RhsMode::Static,
            };
            let parameters = match audit_program_at_crt_depth(
                &request,
                public_ip.program(),
                &noise,
                &terminal_bounds,
                average.as_ref(),
                0,
            ) {
                Ok((parameters, _)) => parameters,
                Err(error) => {
                    failures.push(format!("n={ring_dimension} crt_depth={crt_depth}: {error}"));
                    continue;
                }
            };
            assert_eq!(parameters.levels().len(), 1, "a public Program has one level");
            let workload = RnsBggDerivedWorkload::from_program_with_parameters(
                &parameters,
                public_ip.program(),
            )
            .expect("workload of the accepted parameters");
            let trace = api::noise_simulate(
                public_ip.program(),
                RnsNoiseBuild::new(&parameters, public_ip.program(), &workload, &noise)
                    .expect("noise lowering state"),
                |_, outputs| Ok(outputs),
            )
            .expect("noise simulation");
            assert_eq!(trace.outputs.len(), 1, "noise simulation must finalize the output");
            let source = parameters.level(0).expect("source level").clone();
            let achieved = lattice_security_bits(&source, sigma)
                .expect("lattice security estimator must be available");
            if let Some(row) = recorded {
                assert_eq!(
                    achieved,
                    common::column::<u64>(row, "security_bits"),
                    "the recorded security estimate must be reproduced"
                );
            }
            if achieved < required_security {
                failures.push(format!(
                    "n={ring_dimension} crt_depth={crt_depth}: security {achieved} < {required_security}"
                ));
                continue;
            }
            eprintln!(
                "public inner-product parameters accepted: ring_dimension={ring_dimension} crt_depth={} security_bits={achieved}",
                source.to_crt().0.len()
            );
            return (parameters, source, public_ip);
        }
    }
    panic!("no public inner-product parameter candidate was accepted: {}", failures.join("; "));
}

fn gpu_backend(parameters: &RnsBggParameters, devices: Vec<i32>) -> GpuDcrtBackend {
    let levels = parameters.distinct_levels().into_iter().map(|parameters| {
        GpuDCRTPolyParams::new(
            parameters.ring_dimension(),
            parameters.to_crt().0,
            parameters.base_bits(),
            None,
        )
    });
    gpu_backend_on(levels, devices)
}

#[test]
#[sequential(dcrt_runtime)]
fn public_ip_end_to_end() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let recorded = recorded_parameters();
    let row = recorded.as_ref().map(|(_, row)| row);
    let sigma = error_sigma(row);
    if env::var(format!("{PREFIX}SIMULATION_ONLY")).is_ok_and(|value| value == "1") {
        search(&sigma, row);
        return;
    }
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping the public inner-product round trip: no CUDA device detected");
        return;
    }
    let (parameters, poly_parameters, public_ip) = search(&sigma, row);
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = api::StageStorage::File(directory.path().into()).open().expect("stage storage");
    let mut runtime = GpuRuntime::new(gpu_backend(&parameters, devices)).expect("GPU runtime");
    let runtime = &mut runtime;
    let graphs = StageGraphs::default();
    let program = public_ip.program();

    // The Static secrets are sampled before preprocessing and stay on the runtime.
    let payload = RnsEncryptBuild::new(&parameters, [7; 32])
        .sample_payload_secret(runtime, &mut store, record(&graphs, "payload"))
        .expect("payload secret");
    let mask = RnsEncryptBuild::new(&parameters, [7; 32])
        .sample_mask_secret(runtime, &mut store, record(&graphs, "mask"))
        .expect("mask secret");
    // Every input is public: preprocessing receives no RHS, and encryption receives `a`
    // followed by the encoding of zero.
    let lhs = public_ip.plaintext_inputs::<DCRTPoly>(&poly_parameters).expect("plaintext inputs");
    let (preprocess, online_eval) = api::preprocess(
        program,
        &ParamEnv::default(),
        &api::PlaintextInputs::<_, DCRTPoly>::new(),
        Some(&payload),
        HASH_KEY,
        RnsPreprocessBuild::new(&parameters, [7; 32]),
        runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        record(&graphs, "preprocess"),
    )
    .expect("RNS preprocessing");
    let encrypted = api::encrypt(
        &program.encryption_inputs(),
        &ParamEnv::default(),
        &lhs,
        &preprocess,
        &payload,
        Some(&mask),
        HASH_KEY,
        RnsEncryptBuild::new(&parameters, [7; 32]),
        runtime,
        &mut store,
        |_| Ok(()),
        record(&graphs, "encrypt"),
    )
    .expect("RNS encryption");
    let result = api::online_eval(
        program,
        &ParamEnv::default(),
        &online_eval,
        &encrypted,
        HASH_KEY,
        RnsOnlineEvalBuild::new(&parameters, [7; 32]),
        runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        record(&graphs, "online_eval"),
    )
    .expect("RNS online evaluation");

    // The independent clear inner product, then the output against it and its successor.
    let expected = public_ip
        .encode::<DCRTPoly>(&poly_parameters, public_ip.inner_product())
        .expect("expected");
    let wrong = public_ip
        .encode::<DCRTPoly>(&poly_parameters, public_ip.inner_product() + 1)
        .expect("wrong plaintext");
    verify_output(
        program,
        &result,
        0,
        &expected,
        &parameters,
        &payload,
        Some(&mask),
        runtime,
        &mut store,
    )
    .expect("public inner-product verification");
    assert!(
        verify_output(
            program,
            &result,
            0,
            &wrong,
            &parameters,
            &payload,
            Some(&mask),
            runtime,
            &mut store
        )
        .is_err(),
        "the output must not verify as the successor"
    );
    let wrong_residual = decode_public_output(
        program,
        &result,
        0,
        &parameters,
        &payload,
        Some(&mask),
        &wrong,
        runtime,
        &mut store,
        |_| Ok(()),
    )
    .expect("public output check of the successor");
    assert_ne!(wrong_residual, BigInt::from(0), "the successor must leave a residual");
    let residual = decode_public_output(
        program,
        &result,
        0,
        &parameters,
        &payload,
        Some(&mask),
        &expected,
        runtime,
        &mut store,
        record(&graphs, "decode"),
    )
    .expect("public output check");
    assert_eq!(residual, BigInt::from(0), "the output encodes the inner product");

    if let Some((id, row)) = recorded {
        let directory = common::generated_directory("public_ip", &id);
        let protocol = public_ip_protocol(
            &public_ip,
            graphs.into_inner(),
            &payload,
            &mask,
            (preprocess.production().clone(), encrypted.production().clone()),
            &row,
        );
        mxx_ir_core::lean::protocol::export(&protocol, &directory).expect("claim export");
        eprintln!("exported the public inner-product claim to {}", directory.display());
    }
}

/// The executed workflow as a closed protocol: payload and mask sampling, preprocessing,
/// encryption of `a` and the zero member, online evaluation, and the check of the output against
/// the claimed plaintext `c X^e`, whose residual count must be the ideal `0`. External inputs are
/// the hash key, `a` (with the zero member last), its plaintext monomials `±X^e`, and the claimed
/// plaintext. A requirement ties every member's monomial to `X^(stride a_k)`, makes the last
/// member zero, and makes the claimed plaintext `X^(stride (<a, t> mod Q))`, with `t` fixed by
/// the Program.
fn public_ip_protocol(
    public_ip: &PublicInnerProductProgram,
    graphs: BTreeMap<&'static str, Graph>,
    payload: &RnsBggPayloadSecret,
    mask: &RnsBggMaskSecret,
    (preprocess, encrypt): (
        mxx_ir_core::artifact::ProductionId,
        mxx_ir_core::artifact::ProductionId,
    ),
    row: &ParameterRow,
) -> ProtocolDecl {
    let id = |name: &str| StageId(name.to_owned());
    let productions = BTreeMap::from([(preprocess, id("preprocess")), (encrypt, id("encrypt"))]);
    let payload_name = payload.input_name().expect("sampled payload").to_owned();
    let mask_name = mask.input_name().expect("sampled mask").to_owned();
    let mut links = BTreeMap::from([
        (payload_name, (id("payload"), "secret".to_owned())),
        (mask_name, (id("mask"), "secret".to_owned())),
    ]);
    for port in 0..6 {
        let input = if port < 4 {
            format!("decode-key-{port}")
        } else {
            format!("decode-row-{}", port - 4)
        };
        links.insert(input, (id("online_eval"), format!("output-0.{port}")));
    }
    let graph = |name: &str| &graphs[name];
    // Only the stages the decode stage depends on are part of the workflow: with public inputs
    // only, online evaluation derives every public key from the hash key and reads no
    // preprocessing artifact, and no stage reads the payload secret.
    let all = ["payload", "mask", "preprocess", "encrypt", "online_eval", "decode"]
        .into_iter()
        .map(|name| (id(name), common::stage(name, graph(name), &productions, &links)))
        .collect::<BTreeMap<_, _>>();
    let mut reached = std::collections::BTreeSet::new();
    let mut pending = vec![id("decode")];
    while let Some(stage) = pending.pop() {
        if reached.insert(stage.clone()) {
            pending
                .extend(all[&stage].bindings.iter().map(|binding| binding.producer_stage.clone()));
        }
    }
    // In execution order.
    let order = ["payload", "mask", "preprocess", "encrypt", "online_eval", "decode"];
    let mut all = all;
    let stages = order
        .into_iter()
        .filter(|name| reached.contains(&id(name)))
        .map(|name| all.remove(&id(name)).unwrap())
        .collect::<Vec<_>>();

    let layout = public_ip.layout();
    let ring_dimension = 1i64 << (layout.stride_log2 + layout.width_log2);
    let stride = layout.stride() as i64;
    let q = public_ip.q() as i64;
    let members = public_ip.vector().len() + 1;
    let selected = public_ip
        .selection()
        .iter()
        .enumerate()
        .filter_map(|(coordinate, bit)| bit.then_some(coordinate))
        .collect::<Vec<_>>();
    let family = |count: usize| FamilyType { element: IntType, count: IntExpr::constant(count) };

    // The ideal residual count is zero, as a runtime integer like the decoded count.
    let ideal = DslContext::new("public-ip-ideal");
    let zero = Int::constant(0).add(Int::constant(0));
    let ideal =
        IdealSpec::new(ideal.output("residual", zero).unwrap().build().unwrap().graph).unwrap();

    // The requirement: `±X^e = X^(stride a)` member by member, a zero last member, and the
    // claimed plaintext `X^(stride z)` for `z = <a, t> mod Q`.
    let requirement = DslContext::new("public-ip-inputs");
    let a = requirement.input::<Family<Int>>("a", family(members)).unwrap();
    let coefficient = requirement.input::<Family<Int>>("a_coefficient", family(members)).unwrap();
    let exponent = requirement.input::<Family<Int>>("a_exponent", family(members)).unwrap();
    let expected_coefficient = requirement.input::<Int>("expected_coefficient", IntType).unwrap();
    let expected_exponent = requirement.input::<Int>("expected_exponent", IntType).unwrap();
    // `2 e + n (1 - c) = 2 stride x` with `c = ±1`.
    let monomial = |sign: Int, exponent: Int, value: Int| {
        sign.clone().mul(sign.clone()).equal(1) &
            exponent
                .mul(2)
                .add(Int::constant(1).sub(sign).mul(ring_dimension))
                .equal(value.mul(2 * stride))
    };
    let mut valid = mxx_dsl::iterate(members, Int::constant(0).equal(0), |member, valid| {
        Ok(valid &
            monomial(coefficient.at(member.clone()), exponent.at(member.clone()), a.at(member)))
    })
    .unwrap();
    valid = valid & a.at(Int::constant(members as i64 - 1)).equal(0);
    let z = selected
        .iter()
        .fold(Int::constant(0), |sum, coordinate| sum.add(a.at(Int::constant(*coordinate as i64))))
        .rem(q);
    valid = valid & monomial(expected_coefficient, expected_exponent, z);
    let requirement =
        PurePredicateSpec::new(requirement.output("valid", valid).unwrap().build().unwrap().graph)
            .unwrap();

    let only = |stage: &str, suffix: String| -> String {
        let [name] = <[String; 1]>::try_from(common::inputs_ending_with(graph(stage), &suffix))
            .unwrap_or_else(|_| panic!("{stage} must declare one input ending with {suffix}"));
        name
    };
    let range = |lower: i64, upper: i64| InputValueContract::IntegerRange {
        lower: lower.into(),
        upper: upper.into(),
    };
    let family_range = |count: usize, lower: i64, upper: i64| InputValueContract::Family {
        count: IntExpr::constant(count),
        element: Box::new(range(lower, upper)),
    };
    let vector = public_ip.input_vector();
    let encrypt_input = |field: &str| {
        common::to_stage("encrypt", &only("encrypt", format!("/input/{vector:?}.{field}")))
    };
    let requirement_input =
        |input: &str| ProtocolInputDestination::Requirement { requirement: 0, input: input.into() };
    let hash_key = stages
        .iter()
        .map(|stage| stage.id.0.as_str())
        .filter(|stage| !common::inputs_ending_with(graph(stage), "rns-bgg/hash-key").is_empty())
        .map(|stage| common::to_stage(stage, "rns-bgg/hash-key"))
        .collect();
    let externals = vec![
        common::external("hash_key", InputValueContract::Bytes { length: 32.into() }, hash_key),
        common::external("a", family_range(members, 0, q - 1), vec![requirement_input("a")]),
        common::external(
            "a_coefficient",
            family_range(members, -1, 1),
            vec![encrypt_input("coefficient"), requirement_input("a_coefficient")],
        ),
        common::external(
            "a_exponent",
            family_range(members, 0, ring_dimension - 1),
            vec![encrypt_input("exponent"), requirement_input("a_exponent")],
        ),
        common::external(
            "expected_coefficient",
            range(-1, 1),
            vec![
                common::to_stage("decode", "decode-expected.coefficient"),
                requirement_input("expected_coefficient"),
            ],
        ),
        common::external(
            "expected_exponent",
            range(0, ring_dimension - 1),
            vec![
                common::to_stage("decode", "decode-expected.exponent"),
                requirement_input("expected_exponent"),
            ],
        ),
    ];
    let (contracts, bindings): (Vec<_>, Vec<_>) = externals.into_iter().unzip();

    let endpoint = EndpointSpecId::Exact;
    let failure_probability_log2 = match row["noise_model"].as_str() {
        "worst" => None,
        _ => Some(common::column(row, "failure_exponent")),
    };
    ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        bindings: Default::default(),
        failure_probability_log2,
        bundle: ClosedProtocolBundle {
            workflow: Workflow { stages, entrypoint: id("decode") },
            ideal,
            requirements: vec![requirement],
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "decoded-residual".to_owned(),
                    ideal_input: "residual".to_owned(),
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
                        output: "decoded-residual".to_owned(),
                    },
                    ideal_output: "residual".to_owned(),
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
    .expect("valid public inner-product protocol")
}
