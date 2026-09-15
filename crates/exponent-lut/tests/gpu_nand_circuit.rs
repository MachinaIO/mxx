#![cfg(feature = "gpu")]

//! GPU round trip for the NAND-to-unary-LUT compiler. The test deliberately uses a
//! real RNS-BGG Program, including every MulPrivate/LUT pair emitted by the
//! compiler.  Noise acceptance is performed on that same Program before any
//! graph is built.

use bigdecimal::BigDecimal;
use mxx_bench_estimator::{
    CostReport, estimate, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig,
};
use mxx_dsl::DslContext;
use mxx_exponent_lut::{
    CompiledNandCircuit, ExpLayout, NandCircuit, ProgramInputId,
    noise_refresh::simulation::AverageCaseConfig,
    program::{
        Gate, LutId, ProgramBindings, ProgramFamilyBindings, ProgramGateInstance, ProgramValue,
    },
    rns_bgg::{
        RnsBggMatrix, RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest, RnsBggPublicRhs,
        RnsBggRow, RnsBggScheme,
        noise::{RnsBggGateNoiseMaterial, RnsBggNoiseRhs, RnsBggNoiseValue},
    },
    scheme::{ExponentLUTFixedRhsScheme, PreparedProgram},
};
use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality};
use mxx_primitives::poly::{
    Poly, PolyParams,
    dcrt::{
        gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
        params::DCRTPolyParams,
    },
};
use mxx_runtime::{
    ExecutionConfig, RuntimeValue,
    artifact::{ArtifactStore, FileArtifactStore},
    backend::poly::gpu::gpu_backend_on,
    execute_with_config,
    transcript::SamplingMode,
};
use num_bigint::{BigInt, BigUint};
use num_traits::One;
use std::{collections::BTreeMap, path::PathBuf, process::Command, time::Instant};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.parse().expect("integer environment value"))
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name).as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        Err(error) => panic!("invalid {name}: {error}"),
        Ok(value) => panic!("{name} must be either 0 or 1, got {value:?}"),
    }
}

/// Run the same ternary-secret/Gaussian-error model used by the Exponent-LUT
/// parameter search against the selected level-zero (largest) RNS modulus.
fn lattice_security_bits(parameters: &DCRTPolyParams, sigma: &BigDecimal) -> u64 {
    let output = Command::new("lattice-estimator-cli")
        .arg(parameters.ring_dimension().to_string())
        .arg(parameters.modulus().as_ref().to_string())
        .arg("--s-dist")
        .arg(r#"{"name":"Ternary"}"#)
        .arg("--e-dist")
        .arg(format!(r#"{{"name":"DiscreteGaussian","stddev":{sigma}}}"#))
        .output()
        .expect("lattice-estimator-cli and Sage must be available on PATH");
    assert!(
        output.status.success(),
        "lattice-estimator-cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("lattice-estimator output must be UTF-8");
    let estimates = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && line.bytes().all(|byte| byte.is_ascii_digit()))
        .collect::<Vec<_>>();
    let [estimate] = estimates.as_slice() else {
        panic!("lattice-estimator-cli must print exactly one integer estimate: {stdout}");
    };
    estimate.parse().expect("lattice security estimate must fit u64")
}

fn log_cost_report(stage: &str, report: &CostReport) {
    println!(
        "nand_cost_estimate stage={stage} total_work_seconds={} total_time_seconds={} critical_path_seconds={} peak_memory_bytes={} transmitted_bytes={} oracle_excluded=true",
        report.total_work_seconds,
        report.total_time_seconds,
        report.critical_path_seconds,
        report.peak_memory_bytes,
        report.transmitted_bytes,
    );
}

fn balanced_circuit(depth: usize) -> NandCircuit {
    assert!((1..=3).contains(&depth), "NAND depth must be between 1 and 3");
    let leaves = 1usize << depth;
    let mut nodes: Vec<usize> = (0..leaves).collect();
    let mut gates = Vec::with_capacity(leaves - 1);
    while nodes.len() > 1 {
        let next = nodes
            .chunks_exact(2)
            .map(|pair| {
                let index = leaves + gates.len();
                gates.push([pair[0], pair[1]]);
                index
            })
            .collect();
        nodes = next;
    }
    NandCircuit::new(leaves, gates, nodes[0]).unwrap()
}

fn oracle(circuit: &NandCircuit, inputs: &[usize]) -> usize {
    let mut values = inputs.to_vec();
    for [lhs, rhs] in circuit.gates() {
        values.push(1 ^ (values[*lhs] & values[*rhs]));
    }
    values[circuit.output()]
}

fn noise_bindings(
    compiled: &CompiledNandCircuit,
    encoding_error_bound: &BigUint,
) -> (
    BTreeMap<ProgramInputId, RnsBggNoiseValue>,
    BTreeMap<mxx_exponent_lut::RhsId, RnsBggNoiseRhs>,
    BTreeMap<LutId, RnsBggGateNoiseMaterial>,
    BTreeMap<ProgramGateInstance, RnsBggGateNoiseMaterial>,
) {
    let program = compiled.program();
    let input = RnsBggNoiseValue {
        level: 0,
        bound: RnsBggNoiseBound { error: encoding_error_bound.clone(), payload_l1: 1u8.into() },
        payload_l2_squared: 1u8.into(),
        moments: Some(mxx_exponent_lut::noise_utils::NoiseMoments {
            mean_numerator: 0.into(),
            mean_denominator: BigUint::one(),
            mean_radius: mxx_exponent_lut::noise_utils::NonnegativeRational::zero(),
            variance: mxx_exponent_lut::noise_utils::NonnegativeRational::new(
                16u8.into(),
                1u8.into(),
            )
            .unwrap(),
        }),
        public: false,
        range: None,
        one_hot: false,
    };
    let inputs = BTreeMap::from([(compiled.state_input(), input)]);
    let rhs = compiled
        .rhs_ids()
        .iter()
        .copied()
        .map(|id| {
            (
                id,
                RnsBggNoiseRhs {
                    payload_l1: 1u8.into(),
                    payload_l2_squared: 1u8.into(),
                    one_hot: None,
                },
            )
        })
        .collect();
    let mut helpers = BTreeMap::new();
    let mut gate_helpers = BTreeMap::new();
    for (step, operation) in program.steps().iter().enumerate() {
        let mxx_exponent_lut::program::ProgramStep::Gate(operation) = operation else { continue };
        let material = match operation {
            Gate::MulPrivate { .. } => RnsBggGateNoiseMaterial {
                level: step,
                table: None,
                fresh_error_gain: mxx_exponent_lut::noise_utils::RightActionNoiseGain::unit(),
            },
            Gate::Lut { lut_id, .. } => RnsBggGateNoiseMaterial {
                level: step,
                table: Some(program.lut(*lut_id).unwrap().commitment()),
                fresh_error_gain: mxx_exponent_lut::noise_utils::RightActionNoiseGain::unit(),
            },
            _ => continue,
        };
        gate_helpers.insert(ProgramGateInstance::Gate { step }, material.clone());
        if let Gate::Lut { lut_id, .. } = operation {
            helpers.insert(*lut_id, material);
        }
    }
    (inputs, rhs, helpers, gate_helpers)
}

#[test]
fn test_gpu_nand_circuit_noise_search_and_artifact_round_trip() {
    // Test-shape controls. The small default ring keeps this local GPU integration
    // test practical; it is deliberately capped at 2^8 by the test contract.
    let depth = env_usize("MXX_EXPONENT_LUT_NAND_DEPTH", 1);
    let n = env_usize("MXX_PRIMITIVE_TEST_RING_DIMENSION", 32);
    let large_parameter_run = env_flag("MXX_EXPONENT_LUT_NAND_LARGE_PARAMETER_RUN");
    assert!(
        n <= 1 << 8 || large_parameter_run,
        "ring dimensions above 2^8 require MXX_EXPONENT_LUT_NAND_LARGE_PARAMETER_RUN=1"
    );
    let operations_per_drop = env_usize("MXX_EXPONENT_LUT_NAND_OPERATIONS_PER_DROP", 2);
    assert!(operations_per_drop > 1, "this test must exercise multi-operation cadence");
    let circuit = balanced_circuit(depth);
    // The compiler represents a leaf by one input-controlled matrix shear. A
    // NAND node evaluates its left and right subcircuits twice each; its leading
    // fixed shear is fused into the first child operation. Hence T(0)=1 and
    // T(d)=4*T(d-1), so a balanced depth-d tree has 4^d rounds.
    let rounds = 1usize.checked_shl((2 * depth) as u32).expect("round count overflow");
    let compiled = circuit.compile(ExpLayout { stride_log2: 0, width_log2: 4 }, n, rounds).unwrap();
    // Every round consumes exactly one original-bit RHS. A binary LUT is
    // represented by one MulPrivate followed by one unary LUT, hence two steps.
    assert_eq!(compiled.rhs_ids().len(), rounds);
    assert_eq!(compiled.program().steps().len(), rounds * 2);

    // Public workload contract used by the arithmetic planner:
    // - ring_dimension: polynomial degree (small and configurable for this test).
    // - widths: active exponent-domain width used by each planner stage. A round alternates the
    //   8-state domain with its packed 8x2=16 LUT domain. The final LUT itself returns the Boolean
    //   exponent domain {X^0, X^1}; no separate readout stage is planned.
    // - slots: independently materialized ciphertext values processed by a stage (not polynomial
    //   SIMD slots). This serial scalar Program has one.
    // - encoding_error_sigma: reviewed runtime/estimator sigma. The runtime derives
    //   floor(6.5*sigma)=26 and applies that Gaussian hard cutoff.
    // - crt_bits: bits per CRT prime. The requested 32-bit test contract makes the planner derive
    //   base_bits=ceil(32/2)=16.
    // - terminal_depth/max_crt_depth: candidate terminal prefix depth and the exhaustive search
    //   ceiling for the CRT chain.
    // - terminal_value_bound: the decoded Boolean coefficient is in {0,1}.
    // - operations_per_drop: the MulPrivate/LUT pair shares one physical level.
    let request = RnsBggPlanRequest {
        ring_dimension: n as u32,
        widths: (0..rounds).flat_map(|_| [8, 16]).collect(),
        slots: vec![1; rounds * 2],
        encoding_error_sigma: BigDecimal::from(4u8),
        crt_bits: env_usize("MXX_EXPONENT_LUT_NAND_CRT_BITS", 32),
        terminal_depth: 1,
        max_crt_depth: env_usize("MXX_EXPONENT_LUT_NAND_MAX_CRT_DEPTH", 32),
        terminal_value_bound: BigUint::one(),
        operations_per_drop,
    };
    if !large_parameter_run {
        assert_eq!(request.crt_bits, 32, "local NAND integration requires 32-bit CRT primes");
    }
    // Execution noise and analytical acceptance are independent explicit
    // selectors. Noiseless execution is test-only and is never run unless the
    // flag is exactly 1. Exactly one noise model runs; average is the default.
    let noiseless = env_flag("MXX_EXPONENT_LUT_NAND_NOISELESS");
    let failure_exponent = u32::try_from(env_usize("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", 1))
        .expect("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS must fit u32");
    assert!(failure_exponent > 0, "noise failure exponent must be positive");
    let average = match std::env::var("MXX_EXPONENT_LUT_NAND_NOISE_MODEL").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("average") => Some(AverageCaseConfig {
            failure_exponent,
            allow_average_acceptance: true,
            ..Default::default()
        }),
        Ok("worst") => None,
        Err(error) => panic!("invalid MXX_EXPONENT_LUT_NAND_NOISE_MODEL: {error}"),
        Ok(value) => {
            panic!("MXX_EXPONENT_LUT_NAND_NOISE_MODEL must be `average` or `worst`, got {value:?}")
        }
    };
    run_nand_case(&circuit, &compiled, &request, average.as_ref(), noiseless);
}

fn run_nand_case(
    circuit: &NandCircuit,
    compiled: &CompiledNandCircuit,
    request: &RnsBggPlanRequest,
    average: Option<&AverageCaseConfig>,
    noiseless: bool,
) {
    let rounds = compiled.rhs_ids().len();
    let operations_per_drop = request.operations_per_drop;
    let encoding_error_bound = mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(
        &request.encoding_error_sigma,
    );
    let (inputs, rhs, helpers, gate_helpers) = noise_bindings(compiled, &encoding_error_bound);
    let ranges = ProgramFamilyBindings::new();
    let output = compiled.program().outputs()[0];
    let terminal_bounds = BTreeMap::from([(output, BigUint::one())]);
    let empty_rhs_families = BTreeMap::new();
    let empty_wire_families = BTreeMap::new();
    let bindings = ProgramBindings::new(
        &inputs,
        &rhs,
        &empty_rhs_families,
        &empty_wire_families,
        &helpers,
        Some(&gate_helpers),
    );
    // `plan` constructs a concrete nested CRT chain from the public widths,
    // slot counts, cadence, noise sigma, and candidate terminal depth while
    // reporting, but not imposing, its estimated peak storage. `audit_program`
    // then interprets the actual compiled Program DAG and
    // checks its deterministic or explicitly enabled average-case noise trace.
    // Neither function runs a lattice estimator or certifies RLWE security.
    // `MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS` is a tail-failure exponent for
    // average-case acceptance, not a lattice-security target.
    //
    // Search the actual CRT depths exhaustively: prime-chain candidates are
    // not assumed to have monotone acceptance. Each candidate must pass the
    // Program noise audit for the terminal X^0/X^1 encoding.
    let evaluate_candidate = |depth: usize| {
        let mut candidate = request.clone();
        candidate.terminal_depth = depth;
        RnsBggParameters::plan(&candidate).ok().and_then(|(parameters, _manifest)| {
            let expected_base_bits = request.crt_bits.div_ceil(2).min(31) as u32;
            assert!(
                parameters.levels().iter().all(|level| level.base_bits() == expected_base_bits),
                "planner selected a base other than ceil(crt_bits/2)"
            );
            let trace = parameters
                .audit_program(&compiled.program(), &bindings, &ranges, &terminal_bounds, average)
                .ok()?;
            Some((parameters, trace))
        })
    };
    let mut accepted = None;
    for depth in 1..=request.max_crt_depth {
        if let Some(result) = evaluate_candidate(depth) {
            accepted = Some((depth, result.0, result.1));
            break;
        }
    }
    let selected_depth = accepted.as_ref().unwrap().0;
    assert!(
        evaluate_candidate(selected_depth).is_some(),
        "accepted NAND candidate changed on recheck"
    );
    let (_, parameters, trace) = accepted.expect("no CRT depth passed the Program noise audit");
    // Arithmetic/noise acceptance is not a security estimate. Certify the
    // actually selected ring dimension and largest q with the external lattice
    // estimator before constructing either GPU graph.
    let required_security_bits = u64::try_from(env_usize("MXX_EXPONENT_LUT_NAND_SECURITY_BITS", 1))
        .expect("security target must fit u64");
    assert!(required_security_bits > 0, "lattice security target must be positive");
    // Remote measurement may reuse a security result established by the prior
    // parameter-search run. Supplying it explicitly avoids installing Sage on
    // a disposable benchmark pod while preserving the target comparison and
    // making the evidence source visible in the log.
    let (achieved_security_bits, security_evidence) =
        match std::env::var("MXX_EXPONENT_LUT_NAND_PREVALIDATED_SECURITY_BITS") {
            Ok(value) => (
                value.parse::<u64>().expect("prevalidated security bits must be an integer"),
                "prevalidated",
            ),
            Err(std::env::VarError::NotPresent) => (
                lattice_security_bits(parameters.level(0).unwrap(), &request.encoding_error_sigma),
                "lattice-estimator-cli",
            ),
            Err(error) => panic!("invalid prevalidated security bits: {error}"),
        };
    println!(
        "nand_security_candidate ring_dimension={} crt_depth={} modulus_bits={} security_bits={} evidence={}",
        parameters.levels()[0].ring_dimension(),
        parameters.levels()[0].crt_depth(),
        parameters.levels()[0].modulus().as_ref().bits(),
        achieved_security_bits,
        security_evidence,
    );
    assert!(
        achieved_security_bits >= required_security_bits,
        "selected RNS-BGG parameters provide {achieved_security_bits} lattice-security bits, below the required {required_security_bits}"
    );
    println!(
        "nand_noise_audit model={} failure_exponent={} rounds={rounds} cadence={operations_per_drop} crt_depth={} base_bits={} events={} bound={} lattice_security_bits={} required_lattice_security_bits={}",
        if average.is_some() { "average-case" } else { "worst-case" },
        average.map_or(0, |config| config.failure_exponent),
        parameters.levels()[0].crt_depth(),
        parameters.levels()[0].base_bits(),
        trace.joint_event_count,
        trace.wires[&output].bound.error,
        achieved_security_bits,
        required_security_bits,
    );
    let assignment =
        (0..circuit.input_count()).map(|_| usize::from(rand::random::<bool>())).collect::<Vec<_>>();
    let expected = oracle(&circuit, &assignment);
    let context = rand::random();
    // RnsBggScheme groups the public API used below into four phases:
    // (1) sample secrets/keys and prepare input/RHS/LUT material,
    // (2) export that public preprocessing into an artifact-producing graph,
    // (3) import the manifest-validated descriptors and evaluate online,
    // and (4) verify the terminal exponent-encoding relation. `new_noiseless_for_testing` changes
    // only fresh encoding-error nodes and is unavailable without `test-utils`.
    #[cfg(feature = "test-utils")]
    let scheme = if noiseless {
        RnsBggScheme::new_noiseless_for_testing(parameters.clone(), context)
    } else {
        RnsBggScheme::new(parameters.clone(), context)
    };
    #[cfg(not(feature = "test-utils"))]
    let scheme = {
        if noiseless {
            panic!(
                "MXX_EXPONENT_LUT_NAND_NOISELESS=1 requires mxx-exponent-lut feature `test-utils`"
            );
        }
        RnsBggScheme::new(parameters.clone(), context)
    };
    // Secret setup and input encoding. The hash-key input is public runtime
    // entropy; domain-separated tags give each sampled object a distinct site.
    let payload_secret = scheme.sample_payload_secret().unwrap();
    let mask_secret = scheme.sample_mask_secret().unwrap();
    let high = scheme.ring(0).unwrap();
    let seed = high.bytes_input("hash-key", 32);
    let key =
        scheme.sample_input_key(0, seed.clone(), b"nand-state".as_slice(), rand::random()).unwrap();
    let input_material = scheme
        .prepare_input(
            &payload_secret,
            &key,
            ExpLayout { stride_log2: 0, width_log2: 4 }.mono(&high, &BigInt::from(2)).unwrap(),
        )
        .unwrap();
    // Prepare one private RHS package per compiled matrix-shear round. Each
    // package encrypts the original circuit bit selected by that round, never
    // an intermediate value.
    let mut rhs_public = BTreeMap::new();
    for (round, rhs_id) in compiled.rhs_ids().iter().copied().enumerate() {
        let level = round * 2;
        let bit = assignment[compiled.circuit_input_indices()[round]];
        let payload = ExpLayout { stride_log2: 0, width_log2: 4 }
            .mono(&scheme.ring(level + 1).unwrap(), &BigInt::from(8 * bit))
            .unwrap();
        let sampled = scheme
            .sample_private_rhs(
                &payload_secret,
                level,
                mxx_exponent_lut::rns_bgg::RnsBggRhsPlaintext { payload, id: rand::random() },
                seed.clone(),
                format!("nand-rhs-{round}").into_bytes(),
                rand::random(),
            )
            .unwrap();
        let prepared = scheme.prepare_private_rhs(&sampled).unwrap();
        rhs_public.insert(rhs_id, RnsBggPublicRhs::Fixed(prepared));
    }
    // `LutId` identifies only the logical unary table, and this NAND lowering
    // keeps the private RHS in the preceding MulPrivate gate (`rhs=None` below).
    // The current RNS-BGG *cryptographic* helper is nevertheless not table-only:
    // `RnsBggLutMaterial` records its level and mask-bank identity, and
    // `prepare_lut` rejects a helper whose level differs from the input key.
    // Bind that material by gate instance until RNS-BGG exposes a genuinely
    // level-independent table helper; reusing one object solely by LutId would
    // be invalid when the same table occurs at different levels.
    let mut lut_gate_helpers = BTreeMap::new();
    for (step, operation) in compiled.program().steps().iter().enumerate() {
        let mxx_exponent_lut::program::ProgramStep::Gate(Gate::Lut { lut_id, .. }) = operation
        else {
            continue
        };
        let bank = scheme
            .sample_mask_bank(
                step,
                ExpLayout { stride_log2: 0, width_log2: 4 },
                seed.clone(),
                format!("nand-bank-{step}").into_bytes(),
                rand::random(),
            )
            .unwrap();
        let material = scheme
            .sample_lut_material(
                &payload_secret,
                Some(compiled.program().lut(*lut_id).unwrap()),
                None,
                &bank,
                seed.clone(),
                format!("nand-lut-{step}").into_bytes(),
                rand::random(),
            )
            .unwrap();
        lut_gate_helpers.insert(ProgramGateInstance::Gate { step }, ProgramValue::Scalar(material));
    }
    let lut_helpers = BTreeMap::new();
    let public_inputs = BTreeMap::from([(compiled.state_input(), (key.clone(), None))]);
    let prepared = PreparedProgram::new(
        scheme.clone(),
        compiled.program().clone(),
        &ProgramBindings::new(
            &public_inputs,
            &rhs_public,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &lut_helpers,
            Some(&lut_gate_helpers),
        ),
        ProgramFamilyBindings::new(),
        None,
    )
    .unwrap();
    // `producer` is the preprocessing graph under construction. Each export
    // appends nodes that persist one class of reusable public material and
    // returns its typed descriptor. Building freezes the combined graph.
    let binding = scheme.fixed_product_binding();
    let (producer, descriptor) =
        prepared.export(binding.clone(), DslContext::new("nand-preprocess"), "nand").unwrap();
    let (producer, input_descriptor) =
        scheme.export_input_material(&key, &input_material, producer, "nand-input").unwrap();
    let terminal_level = rounds * 2;
    let terminal_payload = scheme.payload_at(&payload_secret, terminal_level).unwrap();
    let producer = (0..2).fold(producer, |producer, column| {
        producer
            .private_output(
                format!("nand-oracle-payload.{column}"),
                terminal_payload.entry(0, column).clone(),
            )
            .unwrap()
    });
    let producer = producer.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "NAND test requires a detected GPU");
    let make_estimator = || {
        let backends = devices
            .iter()
            .map(|device| {
                let rings = parameters.levels().iter().map(|p| {
                    GpuDCRTPolyParams::new(p.ring_dimension(), p.to_crt().0, p.base_bits(), None)
                });
                (gpu_backend_on(rings, [*device]), *device)
            })
            .collect();
        GpuNodeMeasurementBackend::new(
            backends,
            MeasurementHarnessConfig {
                warm_up_iterations: env_usize("MXX_EXPONENT_LUT_BENCH_WARMUPS", 1),
                measured_iterations: env_usize("MXX_EXPONENT_LUT_BENCH_ITERATIONS", 1),
                ..Default::default()
            },
        )
    };
    // The estimator owns device-side observations. Measure and drop it before
    // creating the production backend on the same devices.
    let mut estimator = make_estimator();
    estimate(&producer, &mut estimator).expect("collect NAND preprocessing graph");
    estimator.measure_collected().expect("measure NAND preprocessing graph");
    let preprocess_report =
        estimate(&producer, &mut estimator).expect("estimate NAND preprocessing graph");
    log_cost_report("preprocess", &preprocess_report);
    drop(estimator);

    let rings = parameters
        .levels()
        .iter()
        .map(|p| GpuDCRTPolyParams::new(p.ring_dimension(), p.to_crt().0, p.base_bits(), None));
    let mut backend = gpu_backend_on(rings, devices.iter().copied());
    // A configured directory keeps artifacts for inspection/reuse. Otherwise a
    // test-owned temporary directory under test_data is removed automatically.
    let configured_directory =
        std::env::var_os("MXX_EXPONENT_LUT_NAND_ARTIFACT_DIR").map(PathBuf::from);
    let temporary_directory =
        configured_directory.is_none().then(|| tempfile::tempdir_in("test_data").unwrap());
    let directory = configured_directory.as_deref().unwrap_or_else(|| {
        temporary_directory.as_ref().expect("temporary artifact directory").path()
    });
    let mut store = FileArtifactStore::new(directory).unwrap();
    let started = Instant::now();
    let produced = execute_with_config(
        &producer,
        &mut backend,
        BTreeMap::from([(
            "hash-key".into(),
            RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
        )]),
        &mut store,
        // Production uses Fresh. Record/Replay are generic runtime diagnostics
        // for reproducing non-hash random nodes; deterministic hash nodes are
        // already reproduced by supplying the same key and domain-separated tag.
        SamplingMode::Fresh,
        ExecutionConfig::default(),
    )
    .unwrap();
    println!("nand_preprocess_seconds={}", started.elapsed().as_secs_f64());
    let production = produced.production_id.unwrap();
    drop(produced.outputs);
    drop(store);
    let mut store = FileArtifactStore::open(directory).unwrap();
    // The manifest records the IR version, production/spec identity, and each
    // artifact's type, family cardinality, confidentiality, and optional public
    // content hash. Loading validates that metadata; payload hashes are checked
    // later when the corresponding public artifacts are loaded.
    let manifest = store.load_manifest(&production).unwrap();
    let descriptor = serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
    let reopened = PreparedProgram::import(
        binding,
        scheme.clone(),
        compiled.program().clone(),
        ProgramFamilyBindings::new(),
        &descriptor,
        production.clone(),
        descriptor.identity().unwrap(),
    )
    .unwrap();
    let (key, material) = scheme
        .import_input_material(
            &serde_json::from_slice(&serde_json::to_vec(&input_descriptor).unwrap()).unwrap(),
            production.clone(),
            *key.id(),
            0,
        )
        .unwrap();
    let encoded = scheme.encode(&mask_secret, &key, &material).unwrap();
    // Derive the mask-secret-dependent online rows for every imported gate
    // helper. The public heavy preprocessing itself remains in `reopened`.
    let online_helpers = reopened.prepare_online_helpers(&mask_secret).unwrap();
    let online_rhs =
        rhs_public.into_iter().map(|(id, rhs)| (id, RnsBggPublicRhs::Identity(rhs.id()))).collect();
    let online_inputs = BTreeMap::from([(compiled.state_input(), (encoded, None))]);
    let values = reopened
        .evaluate(&ProgramBindings::new(
            &online_inputs,
            &online_rhs,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            Some(&online_helpers),
        ))
        .unwrap();
    // `evaluate` interprets the compiled Program using the encoded input,
    // identity-only references to the already prepared RHS artifacts, and the
    // online mask rows. The selected output wire is the encrypted NAND result.
    let terminal_encoding = values[&output].0.clone();
    let terminal_key = reopened.public_values()[&output].0.clone();
    let level = terminal_encoding.level();
    assert_eq!(level, terminal_level);
    assert_eq!(compiled.program().wire_type(output).unwrap().stride_log2, 0);
    let mask = scheme.mask_at(&mask_secret, level).unwrap();
    // The online graph exports all values needed to verify this exact execution:
    // the terminal encoding is still X^0 or X^1, `mask` is its matching secret
    // row, and `public-key` is the matrix used in c - mask*A. There is no
    // canonical-value readout or q/2 scaling stage in this test.
    let online = DslContext::new("nand-online")
        .output("encoding", terminal_encoding)
        .unwrap()
        .output("mask", mask.clone())
        .unwrap()
        .output("public-key", terminal_key)
        .unwrap()
        .build()
        .unwrap()
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([(production.clone(), manifest.clone())]),
        )
        .unwrap();
    // Release production device ownership before creating the second
    // measurement backend for the imported online graph.
    drop(backend);
    let mut estimator = make_estimator();
    estimate(&online, &mut estimator).expect("collect NAND online graph");
    estimator.measure_collected().expect("measure NAND online graph");
    log_cost_report(
        "online",
        &estimate(&online, &mut estimator).expect("estimate NAND online graph"),
    );
    drop(estimator);

    let rings = parameters
        .levels()
        .iter()
        .map(|p| GpuDCRTPolyParams::new(p.ring_dimension(), p.to_crt().0, p.base_bits(), None));
    let mut backend = gpu_backend_on(rings, devices.iter().copied());

    // The estimator's warmup backend is a separate fleet. Trace dynamic
    // polynomial readback claims on this production backend before execution.
    drop(backend.prepare_graph_admission(&online, false, &BTreeMap::new(), 1, true).unwrap());

    let started = Instant::now();
    let mut online_outputs = execute_with_config(
        &online,
        &mut backend,
        BTreeMap::new(),
        &mut store,
        SamplingMode::Fresh,
        ExecutionConfig::default(),
    )
    .unwrap();
    println!("nand_online_seconds={}", started.elapsed().as_secs_f64());
    // RnsBggRow is flattened into one named output per leaf by GraphValue.
    let mut online_values = BTreeMap::new();
    for name in [
        "encoding.0",
        "encoding.1",
        "mask.0",
        "mask.1",
        "public-key.0",
        "public-key.1",
        "public-key.2",
        "public-key.3",
    ] {
        let value = online_outputs.materialize_output(name, &mut backend, &mut store).unwrap();
        assert!(matches!(value, RuntimeValue::Matrix(_)), "{name} must be a matrix");
        online_values.insert(name.to_owned(), value.clone());
    }
    let expected_bit = expected;
    let oracle_ring = scheme.ring(level).unwrap();
    let expected_monomial = ExpLayout { stride_log2: 0, width_log2: 1 }
        .mono(&oracle_ring, &expected_bit.into())
        .unwrap();
    // Consume the first online execution's exported ciphertext, mask, and public
    // key. The payload secret is a private preprocessing artifact, so this
    // deterministic oracle checks the terminal BGG relation without rebuilding
    // any stochastic producer or converting the exponent through a readout.
    let oracle_context = DslContext::new("nand-decryption-oracle");
    let oracle_encoding = RnsBggRow::new([[
        oracle_ring.input("encoding.0", (1, 1)),
        oracle_ring.input("encoding.1", (1, 1)),
    ]])
    .unwrap();
    let oracle_mask = RnsBggRow::new([[
        oracle_ring.input("mask.0", (1, 1)),
        oracle_ring.input("mask.1", (1, 1)),
    ]])
    .unwrap();
    let oracle_public = RnsBggMatrix::new([
        [oracle_ring.input("public-key.0", (1, 1)), oracle_ring.input("public-key.1", (1, 1))],
        [oracle_ring.input("public-key.2", (1, 1)), oracle_ring.input("public-key.3", (1, 1))],
    ])
    .unwrap();
    let oracle_payload = RnsBggRow::new([[
        oracle_ring.artifact_input(
            production.clone(),
            "nand-oracle-payload.0",
            (1, 1),
            ArtifactConfidentiality::Private,
        ),
        oracle_ring.artifact_input(
            production.clone(),
            "nand-oracle-payload.1",
            (1, 1),
            ArtifactConfidentiality::Private,
        ),
    ]])
    .unwrap();
    // This residual check is deliberately test-only and is not part of either
    // cost report. It confirms that the reopened artifact and online ciphertext
    // satisfy the known-answer BGG relation for the independently computed
    // circuit result. At the terminal level the state gadget is identity(1),
    // but it is kept explicit so this verifier mirrors the full relation.
    let terminal_gadget = oracle_ring.identity(1);
    let expected_payload = RnsBggRow::new([[
        expected_monomial.clone() * oracle_payload.entry(0, 0).clone() * terminal_gadget.clone(),
        expected_monomial * oracle_payload.entry(0, 1).clone() * terminal_gadget,
    ]])
    .unwrap();
    let error = oracle_encoding - oracle_mask * oracle_public + expected_payload;
    let consumer = oracle_context
        .output("error", error)
        .unwrap()
        .build()
        .unwrap()
        .validate_with_manifests(&ParamEnv::default(), &BTreeMap::from([(production, manifest)]))
        .unwrap();
    // The estimator's warmup backend is a separate fleet. Trace this oracle's
    // dynamic readback claims on the production backend before execution.
    drop(backend.prepare_graph_admission(&consumer, false, &BTreeMap::new(), 1, true).unwrap());
    let mut oracle_outputs = execute_with_config(
        &consumer,
        &mut backend,
        online_values,
        &mut store,
        SamplingMode::Fresh,
        ExecutionConfig::default(),
    )
    .unwrap();
    // The terminal output remains an exponent encoding. Its direct correctness
    // contract is c - s*A + X^bit*r = error, with no separate readout or
    // Boolean threshold decoder inserted after the compiled Program.
    let modulus = parameters.level(level).unwrap().modulus().as_ref().clone();
    let bound = &trace.wires[&output].bound.error;
    let mut fits_bound = |prefix: &str| {
        let mut fits = true;
        for column in 0..2 {
            let name = format!("{prefix}.{column}");
            let RuntimeValue::Matrix(matrix) =
                oracle_outputs.materialize_output(&name, &mut backend, &mut store).unwrap()
            else {
                panic!("NAND output {name}")
            };
            for shard in matrix.shards() {
                shard.value.wait_until_ready();
                let host = shard.value.to_cpu_matrix();
                for value in host.entry(0, 0).coeffs_biguints() {
                    let centered = value.clone().min(&modulus - value);
                    assert!(
                        &centered * 4u8 < modulus,
                        "known-answer residual is not strictly below q/4: {centered}"
                    );
                    fits &= centered <= *bound;
                }
            }
        }
        fits
    };
    let bounded_residual = fits_bound("error");
    assert!(
        bounded_residual,
        "reopened native encoding residual exceeded the audited bound for expected X^{expected_bit}"
    );
    println!(
        "nand_round_trip_ok=true bounded_residual_ok=true strict_q_over_4=true audited_bound=true expected_exponent={expected_bit} stride_log2=0 readout_applied=false"
    );
}
