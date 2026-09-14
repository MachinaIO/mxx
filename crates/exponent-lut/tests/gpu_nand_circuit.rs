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
    readout::{ExponentLUTReadoutScheme, ValueReadout},
    rns_bgg::{
        RnsBggMatrix, RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest, RnsBggPublicRhs,
        RnsBggReadoutWeights, RnsBggRow, RnsBggScheme,
        noise::{RnsBggGateNoiseMaterial, RnsBggNoiseRhs, RnsBggNoiseValue},
    },
    scheme::{ExponentLUTFixedRhsScheme, PreparedProgram},
};
use mxx_ir_core::ParamEnv;
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
    assert!(n <= 1 << 8, "NAND integration ring dimension must be at most 2^8");
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
    //   8-state domain with its packed 8x2=16 LUT domain; the final width-one entry accounts for
    //   the terminal Boolean.
    // - slots: independently materialized ciphertext values processed by a stage (not polynomial
    //   SIMD slots). This serial scalar Program has one.
    // - encoding_error_sigma: reviewed runtime/estimator sigma. The runtime derives
    //   floor(6.5*sigma)=26 and applies that Gaussian hard cutoff.
    // - crt_bits: bits per CRT prime. The requested 32-bit test contract makes the planner derive
    //   base_bits=ceil(32/2)=16.
    // - terminal_depth/max_crt_depth: candidate terminal prefix depth and the exhaustive search
    //   ceiling for the CRT chain.
    // - memory_budget_bytes: cap on the planner's modeled peak live LUT graph values. It excludes
    //   allocator/NTT workspaces and is not total GPU VRAM; 4 GiB is a conservative
    //   integration-test planning cap.
    // - terminal_value_bound: the decoded Boolean coefficient is in {0,1}.
    // - operations_per_drop: the MulPrivate/LUT pair shares one physical level.
    let request = RnsBggPlanRequest {
        ring_dimension: n as u32,
        widths: (0..rounds).flat_map(|_| [8, 16]).chain([1]).collect(),
        slots: vec![1; rounds * 2 + 1],
        encoding_error_sigma: BigDecimal::from(4u8),
        crt_bits: env_usize("MXX_EXPONENT_LUT_NAND_CRT_BITS", 32),
        terminal_depth: 1,
        max_crt_depth: env_usize("MXX_EXPONENT_LUT_NAND_MAX_CRT_DEPTH", 32),
        memory_budget_bytes: 1 << 32,
        terminal_value_bound: BigUint::one(),
        operations_per_drop,
    };
    assert_eq!(request.crt_bits, 32, "NAND integration requires 32-bit CRT primes");
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
    let readout_width = request.ring_dimension as usize;
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
    // slot counts, cadence, noise sigma, memory cap, and candidate terminal
    // depth. `audit_program` then interprets the actual compiled Program DAG and
    // checks its deterministic or explicitly enabled average-case noise trace.
    // Neither function runs a lattice estimator or certifies RLWE security.
    // `MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS` is a tail-failure exponent for
    // average-case acceptance, not a lattice-security target.
    //
    // Search the actual CRT depths exhaustively: prime-chain candidates are
    // not assumed to have monotone acceptance. Each candidate must pass both
    // the Program noise audit and the decoder-spacing check at the terminal level.
    let evaluate_candidate = |depth: usize| {
        let mut candidate = request.clone();
        candidate.terminal_depth = depth;
        RnsBggParameters::plan(&candidate).ok().and_then(|(parameters, _manifest)| {
            assert!(
                parameters.levels().iter().all(|level| level.base_bits() == 16),
                "planner selected a base other than ceil(32/2)=16"
            );
            let trace = parameters
                .audit_program(&compiled.program(), &bindings, &ranges, &terminal_bounds, average)
                .ok()?;
            let readout_level = parameters.levels().len().checked_sub(2)?;
            let output_level = readout_level + 1;
            let scale = parameters.level(output_level).ok()?.modulus().as_ref() / 2u8;
            let terminal = parameters
                .terminal_noise_bound(
                    readout_level,
                    [(&trace.wires[&output].bound, readout_width, &BigUint::one())],
                    scale.clone(),
                )
                .ok()?;
            // The terminal readout embeds Boolean 0 at 0 and Boolean 1 at q/2.
            // `threshold_decode(2)` places both decision boundaries q/4 from
            // those centers, so the exact strict decoding condition is
            // |error| < q/4 = scale/2. The generic `terminal_noise_fits` uses a
            // stricter q/8 policy and is intentionally not applied here.
            let decoder_radius = scale / 2u8;
            (terminal.error < decoder_radius).then_some((parameters, trace, terminal))
        })
    };
    let mut accepted = None;
    for depth in 1..=request.max_crt_depth {
        if let Some(result) = evaluate_candidate(depth) {
            accepted = Some((depth, result.0, result.1, result.2));
            break;
        }
    }
    let selected_depth = accepted.as_ref().unwrap().0;
    assert!(
        evaluate_candidate(selected_depth).is_some(),
        "accepted NAND candidate changed on recheck"
    );
    let (_, parameters, trace, terminal_noise) =
        accepted.expect("no CRT depth passed eval_ok and decoder decryption_ok");
    // Arithmetic/noise acceptance is not a security estimate. Certify the
    // actually selected ring dimension and largest q with the external lattice
    // estimator before constructing either GPU graph.
    let required_security_bits = u64::try_from(env_usize("MXX_EXPONENT_LUT_NAND_SECURITY_BITS", 1))
        .expect("security target must fit u64");
    assert!(required_security_bits > 0, "lattice security target must be positive");
    let achieved_security_bits =
        lattice_security_bits(parameters.level(0).unwrap(), &request.encoding_error_sigma);
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
        terminal_noise.error,
        achieved_security_bits,
        required_security_bits,
    );

    let assignment =
        (0..circuit.input_count()).map(|_| usize::from(rand::random::<bool>())).collect::<Vec<_>>();
    let expected = oracle(&circuit, &assignment);
    let context = rand::random();
    // RnsBggScheme groups the public API used below into four phases:
    // (1) sample secrets/keys and prepare input/RHS/LUT/readout material,
    // (2) export that public preprocessing into an artifact-producing graph,
    // (3) import the manifest-validated descriptors and evaluate online,
    // and (4) project/decrypt the readout. `new_noiseless_for_testing` changes
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
    // Readout preprocessing maps the terminal Boolean to 0 or q/2 and prepares
    // the public projection needed to remove the online mask after evaluation.
    let readout_layout = ExpLayout { stride_log2: 0, width_log2: readout_width.trailing_zeros() };
    let readout = ValueReadout::from_canonical_values(
        readout_layout,
        (0..readout_layout.width()).map(|value| value % 2).collect(),
    )
    .unwrap();
    // The readout is one production transition after the final program gate.
    let readout_level = rounds * 2;
    let readout_q = parameters.level(readout_level + 1).unwrap().modulus().as_ref().clone();
    let readout_scale = &readout_q / 2u8;
    let readout_bank = scheme
        .sample_mask_bank(
            readout_level,
            readout_layout,
            seed.clone(),
            b"nand-readout-bank".as_slice(),
            rand::random(),
        )
        .unwrap();
    let readout_material = scheme
        .sample_readout_material(
            &payload_secret,
            &readout,
            &readout_bank,
            seed.clone(),
            b"nand-readout".as_slice(),
            rand::random(),
            Some(RnsBggReadoutWeights::Shared([vec![BigInt::from(readout_scale.clone())], vec![]])),
            None,
        )
        .unwrap();
    let public_readout_input = ProgramValue::Scalar(prepared.public_values()[&output].clone());
    let prepared_readout =
        scheme.prepare_readout(&readout, &public_readout_input, &readout_material).unwrap();
    let readout_binding = prepared_readout.binding();
    let readout_id = prepared_readout.id();
    let input_key_id = public_readout_input.scalar().unwrap().0.id().to_owned();
    // `producer` is the preprocessing graph under construction. Each export
    // appends nodes that persist one class of reusable public material and
    // returns its typed descriptor. Building freezes the combined graph.
    let binding = scheme.fixed_product_binding();
    let (producer, descriptor) =
        prepared.export(binding.clone(), DslContext::new("nand-preprocess"), "nand").unwrap();
    let (producer, input_descriptor) =
        scheme.export_input_material(&key, &input_material, producer, "nand-input").unwrap();
    let (producer, readout_descriptor) =
        scheme.export_prepared_readout(&prepared_readout, producer, "nand-readout").unwrap();
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
    let readout_descriptor =
        serde_json::from_slice(&serde_json::to_vec(&readout_descriptor).unwrap()).unwrap();
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
    let prepared_readout = scheme
        .import_prepared_readout(
            &readout_descriptor,
            production.clone(),
            &readout,
            &readout_binding,
            input_key_id,
            readout_id,
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
    let accumulation = values[&output].0.clone();
    let online_masks =
        scheme.prepare_online_masks(&mask_secret, &prepared_readout.mask_bank()).unwrap();
    // Readout evaluation converts the terminal encoding into its masked
    // canonical-value representation; projection computes the matching public
    // matrix used by the trusted test oracle below.
    let readout_value = scheme
        .evaluate_readout(
            &readout,
            &ProgramValue::Scalar((accumulation, None)),
            &prepared_readout,
            &online_masks,
        )
        .unwrap();
    let level = readout_value.level();
    let mask = scheme.mask_at(&mask_secret, level).unwrap();
    let public_readout =
        scheme.project_readout(&readout, &public_readout_input, &prepared_readout, &()).unwrap();
    // The online graph exports all values needed to verify this exact execution:
    // `readout` is the masked encoding result, `mask` is its matching secret row,
    // and `public-readout` is the public matrix used in c - mask*A. Exporting all
    // three prevents the oracle from rebuilding a second stochastic online run.
    let online = DslContext::new("nand-online")
        .output("readout", readout_value.values.scalar().unwrap().clone())
        .unwrap()
        .output("mask", mask.clone())
        .unwrap()
        .output("public-readout", public_readout.values.scalar().unwrap().clone())
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
    log_cost_report("preprocess", &preprocess_report);
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
        "readout.0",
        "readout.1",
        "mask.0",
        "mask.1",
        "public-readout.0",
        "public-readout.1",
        "public-readout.2",
        "public-readout.3",
    ] {
        let value = online_outputs.materialize_output(name, &mut backend, &mut store).unwrap();
        assert!(matches!(value, RuntimeValue::Matrix(_)), "{name} must be a matrix");
        online_values.insert(name.to_owned(), value.clone());
    }
    let expected_bit = expected;
    let expected = scheme
        .ring(level)
        .unwrap()
        .polynomial([(BigInt::from(expected) * BigInt::from(readout_scale.clone())).into()]);
    let expected = mxx_exponent_lut::rns_bgg::RnsBggRow::new([[
        expected,
        scheme.ring(level).unwrap().zero((1, 1)),
    ]])
    .unwrap();
    // Consume the first online execution's exported ciphertext, mask, and public
    // readout. The oracle graph contains only these inputs and deterministic
    // decryption/decoding; it must not rerun the stochastic producer graph.
    let oracle_context = DslContext::new("nand-decryption-oracle");
    let oracle_ring = scheme.ring(level).unwrap();
    let oracle_readout = RnsBggRow::new([[
        oracle_ring.input("readout.0", (1, 1)),
        oracle_ring.input("readout.1", (1, 1)),
    ]])
    .unwrap();
    let oracle_mask = RnsBggRow::new([[
        oracle_ring.input("mask.0", (1, 1)),
        oracle_ring.input("mask.1", (1, 1)),
    ]])
    .unwrap();
    let oracle_public = RnsBggMatrix::new([
        [
            oracle_ring.input("public-readout.0", (1, 1)),
            oracle_ring.input("public-readout.1", (1, 1)),
        ],
        [
            oracle_ring.input("public-readout.2", (1, 1)),
            oracle_ring.input("public-readout.3", (1, 1)),
        ],
    ])
    .unwrap();
    let decrypted = oracle_readout - oracle_mask * oracle_public;
    let decoded = decrypted
        .entry(0, 0)
        .clone()
        .threshold_decode_bools(2, 1)
        .into_iter()
        .next()
        .expect("decrypted readout decoder output");
    let error = decrypted - expected;
    let consumer = oracle_context
        .output("decoded", decoded)
        .unwrap()
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
    let decoded =
        match oracle_outputs.materialize_output("decoded", &mut backend, &mut store).unwrap() {
            RuntimeValue::Bool(decoded) => *decoded,
            _ => panic!("decryption oracle must produce a Boolean decoder output"),
        };
    assert_eq!(decoded, expected_bit == 1, "production readout decoded the wrong Boolean");
    // The residual is a secondary noise diagnostic only.  The production
    // decryption oracle above is authoritative; fresh BGG masks and LUT
    // randomness make an exact-zero residual neither expected nor sufficient.
    let modulus = parameters.level(level).unwrap().modulus().as_ref().clone();
    let bound = &terminal_noise.error;
    // The RnsBggRow residual follows the same per-leaf flattening as readout.
    for name in ["error.0", "error.1"] {
        let RuntimeValue::Matrix(matrix) =
            oracle_outputs.materialize_output(name, &mut backend, &mut store).unwrap()
        else {
            panic!("NAND output {name}")
        };
        for shard in matrix.shards() {
            shard.value.wait_until_ready();
            let host = shard.value.to_cpu_matrix();
            for value in host.entry(0, 0).coeffs_biguints() {
                let centered = value.clone().min(&modulus - value);
                assert!(
                    centered <= *bound,
                    "decryption oracle failed: centered residual {centered} exceeds audited bound {bound}"
                );
            }
        }
    }
    println!("nand_decryption_ok=true expected_bit={expected_bit}");
}
