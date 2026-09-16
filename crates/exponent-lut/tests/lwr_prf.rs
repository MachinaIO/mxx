#![cfg(feature = "gpu")]

//! Reviewed sparse-LWR PRF integration.
//!
//! The test validates the reviewed report and noise model, estimates the exact
//! three production graphs without executing them, and then runs those stages.

use bigdecimal::BigDecimal;
use mxx_exponent_lut::{
    EncodingSecrets, ExpLayout, ExpectedEncoding, ExponentLut, RuntimeConfiguration,
    SparseLwrPrfProgram,
    noise_refresh::{
        pbc::{
            PbcActiveCellIndex, PbcLayoutSeed, PbcParameters, PbcPublicLayout, PbcRootSeed,
            generate_key_layout,
        },
        prf::SparseLwrPrfProfile,
        simulation::NoiseModelKind,
    },
    rns_bgg::{RnsBggParameters, RnsBggPlanRequest},
};
use mxx_ir_core::ValidatedGraph;
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{
        gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
        params::DCRTPolyParams,
    },
};
use mxx_runtime::ExecutionConfig;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::{SeedableRng, rngs::StdRng, seq::index::sample};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    str::FromStr,
    sync::{Arc, Mutex},
    time::Instant,
};
use tempfile::tempdir;

#[derive(Clone, Copy, Debug)]
struct TestSettings {
    log_n: usize,
    depth: usize,
    crt_bits: usize,
    universe: usize,
    weight: usize,
    q_native: usize,
    p: usize,
    security: u64,
}

impl TestSettings {
    fn from_env() -> Self {
        let value = |name: &str, default: usize| {
            std::env::var(format!("MXX_EXPONENT_LUT_REFRESH_{name}"))
                .map(|value| value.parse().expect("integer refresh setting"))
                .unwrap_or(default)
        };
        Self {
            log_n: value("MIN_LOG_RING_DIMENSION", 5),
            depth: value("MIN_CRT_DEPTH", 7),
            crt_bits: value("CRT_BITS", 50),
            universe: value("SPARSE_LWR_UNIVERSE", 64),
            weight: value("SPARSE_LWR_WEIGHT", 1),
            q_native: value("SPARSE_LWR_MODULUS", 8),
            p: value("SPARSE_LWR_OUTPUT_MODULUS", 4),
            security: value("SECURITY_BITS", 1) as u64,
        }
    }
}

#[derive(Clone)]
struct ReviewedScenario {
    parameters: RnsBggParameters,
    ring_dimension: u32,
    q_native: usize,
    p: usize,
    universe: usize,
    weight: usize,
    security_bits: u64,
    pbc_seed: [u8; 32],
    pbc_attempt: u32,
    public_layout: PbcPublicLayout,
    encoding_bound: BigUint,
}

fn required_u64(object: &Value, field: &str) -> u64 {
    object[field].as_u64().unwrap_or_else(|| panic!("accepted report must contain integer {field}"))
}

fn security_floor(object: &Value, field: &str) -> u64 {
    if let Some(bits) = object.get("integer_floor").and_then(Value::as_u64) {
        return bits;
    }
    let minimum = object
        .get("minimum_classical_bits")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("accepted report must contain {field}.minimum_classical_bits"));
    assert!(minimum.is_finite() && minimum >= 0.0, "invalid {field} security");
    minimum.floor() as u64
}

fn report_encoding_noise(report: &Value) -> (BigDecimal, BigUint) {
    let phase2 = report.get("phase2").filter(|value| value.is_object());
    let sigma = phase2
        .and_then(|value| value.get("error_sigma"))
        .or_else(|| report.get("error_sigma"))
        .and_then(Value::as_str)
        .and_then(|value| BigDecimal::from_str(value).ok())
        .expect("accepted report must contain a valid error_sigma");
    let bound = phase2
        .and_then(|value| value.get("encoding_error_bound"))
        .or_else(|| report.get("encoding_error_bound"))
        .and_then(Value::as_str)
        .and_then(|value| BigUint::from_str(value).ok())
        .expect("accepted report must contain a valid encoding_error_bound");
    let expected = mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma);
    assert_eq!(bound, expected, "accepted report sigma/bound mismatch");
    (sigma, bound)
}

/// The current refresh search writer stores its accepted point in a typed
/// `phase1`/`phase2`/`result` envelope.  Keep this reader strict about the
/// fields needed to reconstruct the reviewed scenario; it must not silently
/// reinterpret an unrelated JSON document as an accepted report.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CurrentRefreshReport {
    phase1: CurrentPhase1Report,
    phase2: CurrentPhase2Report,
    result: CurrentSearchResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CurrentCandidate {
    crt_depth: usize,
    log_ring_dimension: usize,
    crt_bits: usize,
    base_bits: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CurrentPhase1Report {
    security_target_bits: u64,
    selected_universe: usize,
    selected_q_l: usize,
    selected_p: usize,
    selected_weight: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CurrentPhase2Report {
    error_sigma: String,
    encoding_error_bound: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CurrentSearchResult {
    candidate: CurrentCandidate,
    achieved_security_bits: u64,
    ring_dimension: usize,
    bucket_width: usize,
    pbc_attempts_used: u32,
    layout_id: String,
    checker_accepted: bool,
    sparse_lwr_q_l: usize,
    sparse_lwr_p: usize,
    sparse_lwr_universe: usize,
    sparse_lwr_weight: usize,
}

fn current_reviewed_scenario(
    settings: TestSettings,
    report: CurrentRefreshReport,
) -> ReviewedScenario {
    let phase1 = report.phase1;
    let phase2 = report.phase2;
    let result = report.result;
    assert!(result.checker_accepted, "current refresh report result is not accepted");
    assert_eq!(
        (phase1.selected_q_l, phase1.selected_p, phase1.selected_universe, phase1.selected_weight),
        (
            result.sparse_lwr_q_l,
            result.sparse_lwr_p,
            result.sparse_lwr_universe,
            result.sparse_lwr_weight,
        ),
        "current refresh report Phase-1 and result selections disagree"
    );
    let target = settings.security.max(phase1.security_target_bits);
    assert!(result.achieved_security_bits >= target);
    let sigma = BigDecimal::from_str(&phase2.error_sigma)
        .expect("current refresh report must contain a valid error_sigma");
    let bound = BigUint::from_str(&phase2.encoding_error_bound)
        .expect("current refresh report must contain a valid encoding_error_bound");
    assert_eq!(
        bound,
        mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma),
        "current refresh report sigma/bound mismatch"
    );
    let expected_ring = 1usize
        .checked_shl(result.candidate.log_ring_dimension as u32)
        .expect("current refresh report ring dimension overflow");
    assert_eq!(expected_ring, result.ring_dimension);
    assert_eq!(
        result.candidate.base_bits as usize,
        result.candidate.crt_bits.div_ceil(2),
        "current refresh report uses an unsupported CRT/base-bit pair"
    );
    let pbc = PbcParameters::paper_evaluation(result.sparse_lwr_universe, result.sparse_lwr_weight);
    // These are the fixed, public derivation inputs used by the current
    // refresh producer when it constructs the accepted PBC layout. Reusing
    // them makes the layout identity a checked consequence of the report,
    // rather than an unchecked fallback chosen by this reader.
    let mut support_rng = StdRng::from_seed([0x6a; 32]);
    let support =
        sample(&mut support_rng, result.sparse_lwr_universe, result.sparse_lwr_weight).into_vec();
    let generated = generate_key_layout(&pbc, PbcRootSeed([0x19; 32]), &support)
        .expect("current refresh report PBC layout");
    assert_eq!(
        generated.public_layout.accepted_attempt + 1,
        result.pbc_attempts_used,
        "current refresh report PBC retry count disagrees with its deterministic layout"
    );
    assert_eq!(
        generated
            .public_layout
            .layout_id
            .0
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        result.layout_id,
        "current refresh report layout identity disagrees with its deterministic layout"
    );
    assert_eq!(generated.public_layout.bucket_width, result.bucket_width);
    PbcActiveCellIndex::build(&generated.public_layout)
        .expect("current refresh report active-cell index");
    let request = RnsBggPlanRequest {
        ring_dimension: result.ring_dimension as u32,
        widths: vec![1; pbc.bucket_count + 2],
        encoding_error_sigma: sigma.clone(),
        crt_bits: result.candidate.crt_bits,
        terminal_depth: 1,
        max_crt_depth: result.candidate.crt_depth.max(32),
        terminal_value_bound: BigUint::one(),
        operations_per_drop: 1,
    };
    let (parameters, _) = RnsBggParameters::plan_fixture(&request)
        .expect("current refresh report RNS-BGG parameters");
    assert_eq!(parameters.encoding_error_sigma(), &sigma);
    assert_eq!(parameters.encoding_error_bound(), &bound);
    ReviewedScenario {
        parameters,
        ring_dimension: result.ring_dimension as u32,
        q_native: result.sparse_lwr_q_l,
        p: result.sparse_lwr_p,
        universe: result.sparse_lwr_universe,
        weight: result.sparse_lwr_weight,
        security_bits: result.achieved_security_bits,
        pbc_seed: generated.public_layout.seed.0,
        pbc_attempt: generated.public_layout.accepted_attempt,
        public_layout: generated.public_layout,
        encoding_bound: bound,
    }
}

fn reviewed_scenario(settings: TestSettings) -> ReviewedScenario {
    if let Some(path) = std::env::var_os("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT") {
        let report: Value =
            serde_json::from_slice(&std::fs::read(path).expect("accepted report readable"))
                .expect("accepted report JSON");
        if report.get("phase1").is_some() || report.get("phase2").is_some() {
            let current: CurrentRefreshReport = serde_json::from_value(report)
                .unwrap_or_else(|error| panic!("invalid current refresh report schema: {error}"));
            return current_reviewed_scenario(settings, current);
        }
        let meta = report.get("numerical_run").expect("accepted report must contain numerical_run");
        let candidate = report
            .get("selected_candidate")
            .expect("accepted report must contain selected_candidate");
        assert_eq!(
            candidate.get("correctness_accepted").and_then(Value::as_bool),
            Some(true),
            "selected candidate is not correctness-accepted"
        );
        let target = settings.security.max(
            meta.get("security_target_bits")
                .or_else(|| report.get("security_target_bits"))
                .and_then(Value::as_u64)
                .unwrap_or(settings.security),
        );
        let achieved = report
            .get("achieved_security_bits")
            .or_else(|| candidate.get("security_bits"))
            .and_then(Value::as_u64)
            .unwrap_or_else(|| {
                let lwe = report.get("lwe_security").expect("missing lwe security");
                let lwr = report.get("lwr_security").expect("missing lwr security");
                security_floor(lwe, "lwe_security").min(security_floor(lwr, "lwr_security"))
            });
        assert!(achieved >= target, "accepted report security {achieved} is below target {target}");
        let selector_counts: Vec<usize> =
            serde_json::from_value(meta["selector_counts"].clone()).expect("selector_counts");
        let pbc_blocks = required_u64(meta, "pbc_blocks") as usize;
        assert_eq!(selector_counts.len(), pbc_blocks);
        let pbc_seed: [u8; 32] =
            serde_json::from_value(meta["pbc_seed"].clone()).expect("pbc_seed");
        let pbc_attempt =
            meta.get("pbc_accepted_attempt").and_then(Value::as_u64).unwrap_or(0) as u32;
        let pbc = PbcParameters::paper_evaluation(
            required_u64(meta, "lwr_dimension") as usize,
            required_u64(meta, "lwr_weight") as usize,
        );
        let public_layout = PbcPublicLayout::build(&pbc, PbcLayoutSeed(pbc_seed), pbc_attempt)
            .expect("accepted PBC layout");
        assert_eq!(
            public_layout.layout_id.0,
            serde_json::from_value::<[u8; 32]>(meta["pbc_layout_id"].clone())
                .expect("pbc_layout_id")
        );
        assert_eq!(
            PbcActiveCellIndex::build(&public_layout)
                .unwrap()
                .bucket_active_widths()
                .collect::<Vec<_>>(),
            selector_counts
        );
        let (sigma, bound) = report_encoding_noise(&report);
        let n = required_u64(meta, "n") as u32;
        let primes: Vec<Vec<u64>> =
            serde_json::from_value(candidate["level_primes"].clone()).expect("level_primes");
        let parameters = RnsBggParameters::new(
            primes
                .into_iter()
                .map(|primes| {
                    DCRTPolyParams::new(
                        n,
                        primes.len(),
                        required_u64(meta, "crt_bits") as usize,
                        required_u64(meta, "base_bits") as u32,
                        Some(primes),
                        None,
                    )
                })
                .collect(),
            sigma,
            1,
        )
        .expect("accepted RNS-BGG parameters");
        assert_eq!(parameters.encoding_error_bound(), &bound);
        return ReviewedScenario {
            parameters,
            ring_dimension: n,
            q_native: required_u64(meta, "lwr_q") as usize,
            p: required_u64(meta, "lwr_p") as usize,
            universe: required_u64(meta, "lwr_dimension") as usize,
            weight: required_u64(meta, "lwr_weight") as usize,
            security_bits: achieved,
            pbc_seed,
            pbc_attempt,
            public_layout,
            encoding_bound: bound,
        };
    }

    let ring_dimension = 1u32 << settings.log_n;
    let pbc = PbcParameters::paper_evaluation(settings.universe, settings.weight);
    let request = RnsBggPlanRequest {
        ring_dimension,
        widths: vec![1; pbc.bucket_count + 2],
        encoding_error_sigma: BigDecimal::from(4u8),
        crt_bits: settings.crt_bits,
        terminal_depth: 1,
        max_crt_depth: settings.depth.max(32),
        terminal_value_bound: BigUint::one(),
        operations_per_drop: 1,
    };
    let (parameters, manifest) = RnsBggParameters::plan_fixture(&request).unwrap();
    let generated = generate_key_layout(
        &pbc,
        PbcRootSeed([0x91; 32]),
        &(0..settings.weight).collect::<Vec<_>>(),
    )
    .unwrap();
    // Persist the accepted layout seed and retry attempt, not the root seed;
    // selector_key_from_layout_seed reconstructs exactly this public layout.
    let pbc_seed = generated.public_layout.seed.0;
    let pbc_attempt = generated.public_layout.accepted_attempt;
    println!(
        "prf_facade_parameter_report ring_dimension={ring_dimension} encoding_error_sigma={} encoding_error_bound={} estimated_bytes={} security_bits={}",
        parameters.encoding_error_sigma(),
        parameters.encoding_error_bound(),
        manifest.estimated_bytes,
        settings.security
    );
    ReviewedScenario {
        encoding_bound: parameters.encoding_error_bound().clone(),
        parameters,
        ring_dimension,
        q_native: settings.q_native,
        p: settings.p,
        universe: settings.universe,
        weight: settings.weight,
        security_bits: settings.security,
        pbc_seed,
        pbc_attempt,
        public_layout: generated.public_layout,
    }
}

/// Audit the exact facade Program before any graph estimate or execution.
fn simulate_noise(
    scenario: &ReviewedScenario,
    program: &SparseLwrPrfProgram,
    fixed: &mxx_exponent_lut::FixedRhs,
) {
    let model = NoiseModelKind::from_env().unwrap();
    let audit = program
        .audit_noise(
            &scenario.parameters,
            fixed,
            model == NoiseModelKind::AverageCase,
            scenario.security_bits as u32,
        )
        .unwrap();
    assert_eq!(scenario.parameters.encoding_error_bound(), &scenario.encoding_bound);
    assert!(
        audit.strict_terminal_fit,
        "selected terminal noise bound does not fit the readout modulus"
    );
    assert!(!audit.selected_error.is_zero());
    assert!(
        &audit.selected_error * 2u8 < audit.digit_scale,
        "terminal error must be strictly below half of the production PRF digit spacing"
    );
    println!(
        "prf_noise_simulation model={model:?} failure_exponent={} split_accumulation_exponent={:?} split_readout_exponent={:?} coefficient_events={} terminal_level={} output_level={} branches={} selected_error={} worst_error={} digit_bound={} target_bound={} digit_scale={} worst_fit={} sigma={} bound={}",
        scenario.security_bits,
        audit.accumulation_failure_exponent,
        audit.readout_failure_exponent,
        audit.joint_event_count,
        audit.terminal_level,
        audit.output_level,
        audit.branches,
        audit.selected_error,
        audit.terminal_bound.error,
        audit.digit_bound,
        audit.target_bound,
        audit.digit_scale,
        audit.worst_case_terminal_fit,
        scenario.parameters.encoding_error_sigma(),
        scenario.encoding_bound
    );
}

fn gpu_rings(parameters: &RnsBggParameters) -> impl Iterator<Item = GpuDCRTPolyParams> + '_ {
    parameters.levels().iter().map(|level| {
        GpuDCRTPolyParams::new(level.ring_dimension(), level.to_crt().0, level.base_bits(), None)
    })
}

/// Estimate the exact facade graphs. This plans only; it does not create an
/// artifact directory or execute a backend. Q is the native PRF modulus, P is
/// the output alphabet, and stride maps a PBC cell to a ring monomial.
fn estimate_graphs(
    scenario: &ReviewedScenario,
    program: &SparseLwrPrfProgram,
    fixed: &mxx_exponent_lut::FixedRhs,
    devices: &[i32],
) {
    let runtime = RuntimeConfiguration::new(
        mxx_runtime::backend::poly::gpu::gpu_backend_on(
            gpu_rings(&scenario.parameters),
            devices.to_vec(),
        ),
        ExecutionConfig::default(),
    );
    let engine = ExponentLut::new(scenario.parameters.clone(), [0x53; 32], runtime);
    let secrets: EncodingSecrets = program.sample_secrets(&engine);
    let graphs = engine.benchmark_graphs(program.program(), fixed, secrets.payload()).unwrap();
    let warmups = std::env::var("MXX_EXPONENT_LUT_BENCH_WARMUPS")
        .map(|value| value.parse().expect("warmup count"))
        .unwrap_or(1);
    let iterations = std::env::var("MXX_EXPONENT_LUT_BENCH_ITERATIONS")
        .map(|value| value.parse().expect("iteration count"))
        .unwrap_or(1);
    let mut estimator = mxx_bench_estimator::gpu::GpuNodeMeasurementBackend::new(
        devices
            .iter()
            .map(|device| {
                (
                    mxx_runtime::backend::poly::gpu::gpu_backend_on(
                        gpu_rings(&scenario.parameters),
                        [*device],
                    ),
                    *device,
                )
            })
            .collect(),
        mxx_bench_estimator::harness::MeasurementHarnessConfig {
            warm_up_iterations: warmups,
            measured_iterations: iterations,
            ..Default::default()
        },
    );
    for (stage, graph) in [
        ("preprocess", &graphs.preprocess),
        ("encrypt", &graphs.encrypt),
        ("online_eval", &graphs.online_eval),
    ] {
        let report = mxx_bench_estimator::measure_graphs(&[graph], &mut estimator)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        println!(
            "prf_one_output_estimate stage={stage} total_work_seconds={} cumulative_wave_seconds={} ideal_dag_seconds={} warmups={warmups} iterations={iterations}",
            report.total_work_seconds, report.total_time_seconds, report.critical_path_seconds,
        );
    }
}

/// Execute the same three facade stages and retain only typed output handles.
fn execute_round_trip(
    scenario: ReviewedScenario,
    program: SparseLwrPrfProgram,
    fixed: mxx_exponent_lut::FixedRhs,
    devices: Vec<i32>,
) {
    let observed = Arc::new(Mutex::new(Vec::<(String, ValidatedGraph)>::new()));
    let observer_output = observed.clone();
    let runtime = RuntimeConfiguration::new(
        mxx_runtime::backend::poly::gpu::gpu_backend_on(gpu_rings(&scenario.parameters), devices),
        ExecutionConfig::default(),
    )
    .with_graph_observer(move |stage, graph| {
        observer_output.lock().unwrap().push((stage.to_owned(), graph.clone()));
        Ok(())
    });
    let engine = ExponentLut::new(scenario.parameters, [0x53; 32], runtime);
    let secrets: EncodingSecrets = program.sample_secrets(&engine);
    let encryption_dir = tempdir().unwrap();
    let evaluation_dir = tempdir().unwrap();
    let started = Instant::now();
    let prepared = engine
        .preprocess(
            program.program(),
            fixed,
            secrets.payload(),
            encryption_dir.path(),
            evaluation_dir.path(),
        )
        .unwrap();
    println!("prf_roundtrip_stage=preprocess seconds={}", started.elapsed().as_secs_f64());
    let input_dir = tempdir().unwrap();
    let started = Instant::now();
    let input = engine.encrypt(&prepared.encryption, secrets.mask(), input_dir.path()).unwrap();
    println!("prf_roundtrip_stage=encrypt seconds={}", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let outputs = engine.online_eval(program.program(), &prepared.evaluation, &input).unwrap();
    println!("prf_roundtrip_stage=online_eval seconds={}", started.elapsed().as_secs_f64());
    let output = program.output(&outputs).unwrap();
    engine.assert_encoding(output, ExpectedEncoding::PrfDigit(1.into()), &secrets).unwrap();
    let observed = observed.lock().unwrap();
    let stages = observed.iter().map(|(stage, _)| stage.as_str()).collect::<Vec<_>>();
    assert_eq!(stages, ["preprocess", "encrypt", "online_eval"]);
}

#[test]
fn current_refresh_report_round_trips_through_accepted_reader() {
    let universe = 16;
    let weight = 1;
    let pbc = PbcParameters::paper_evaluation(universe, weight);
    let mut support_rng = StdRng::from_seed([0x6a; 32]);
    let support = sample(&mut support_rng, universe, weight).into_vec();
    let layout = generate_key_layout(&pbc, PbcRootSeed([0x19; 32]), &support).unwrap();
    let layout_id = layout
        .public_layout
        .layout_id
        .0
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let report = CurrentRefreshReport {
        phase1: CurrentPhase1Report {
            security_target_bits: 1,
            selected_universe: universe,
            selected_q_l: 8,
            selected_p: 4,
            selected_weight: weight,
        },
        phase2: CurrentPhase2Report {
            error_sigma: "4".to_owned(),
            encoding_error_bound: "26".to_owned(),
        },
        result: CurrentSearchResult {
            candidate: CurrentCandidate {
                crt_depth: 2,
                log_ring_dimension: 5,
                crt_bits: 32,
                base_bits: 16,
            },
            achieved_security_bits: 128,
            ring_dimension: 32,
            bucket_width: layout.public_layout.bucket_width,
            pbc_attempts_used: layout.public_layout.accepted_attempt + 1,
            layout_id,
            checker_accepted: true,
            sparse_lwr_q_l: 8,
            sparse_lwr_p: 4,
            sparse_lwr_universe: universe,
            sparse_lwr_weight: weight,
        },
    };
    let serialized = serde_json::to_value(&report).unwrap();
    let parsed: CurrentRefreshReport = serde_json::from_value(serialized).unwrap();
    let settings = TestSettings {
        log_n: 5,
        depth: 2,
        crt_bits: 32,
        universe,
        weight,
        q_native: 8,
        p: 4,
        security: 1,
    };
    let scenario = current_reviewed_scenario(settings, parsed);
    assert_eq!(scenario.ring_dimension, 32);
    assert_eq!(scenario.q_native, 8);
    assert_eq!(scenario.p, 4);
}

#[test]
#[serial_test::serial]
fn test_gpu_lwr_prf_one_output_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    // Load the accepted parameter/PBC report, or construct the small fixture;
    // this is the simulation authority for the rest of the public facade test.
    let settings = TestSettings::from_env();
    let scenario = reviewed_scenario(settings);
    // Build the native sparse-LWR profile: Q is the input residue modulus and
    // P is the output alphabet, while the layout maps cells to ring monomials.
    assert!(scenario.q_native.is_power_of_two() && scenario.p.is_power_of_two());
    assert!(scenario.q_native >= 4 && scenario.p >= 4);
    let width_log2 = scenario.q_native.trailing_zeros() - 1;
    assert!(width_log2 <= scenario.ring_dimension.trailing_zeros());
    let profile = SparseLwrPrfProfile::new(
        ExpLayout {
            stride_log2: scenario.ring_dimension.trailing_zeros() - width_log2,
            width_log2,
        },
        scenario.p / 2,
    )
    .unwrap();
    let support = (0..scenario.weight).collect::<Vec<_>>();
    assert_eq!(
        PbcActiveCellIndex::build(&scenario.public_layout).unwrap().bucket_active_widths().count(),
        scenario.public_layout.parameters.bucket_count
    );
    let program = SparseLwrPrfProgram::new(
        profile,
        scenario.public_layout.bucket_width,
        scenario.public_layout.parameters.bucket_count,
    )
    .unwrap();
    // Recompute the Gaussian noise bound before any GPU work, then bind the
    // opaque selector schedule and public residues to the frozen Program.
    let selector = program
        .selector_key_from_layout_seed(
            scenario.universe,
            &support,
            scenario.pbc_seed,
            scenario.pbc_attempt,
        )
        .unwrap();
    let fixed = program.fixed_rhs(&vec![0; scenario.universe], &selector).unwrap();
    simulate_noise(&scenario, &program, &fixed);
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU round trip requires a detected device");
    // Estimate the exact preprocess/encrypt/online graphs without executing
    // them, then run the same three public facade phases once each.
    estimate_graphs(&scenario, &program, &fixed, &devices);
    execute_round_trip(scenario, program, fixed, devices);
}
