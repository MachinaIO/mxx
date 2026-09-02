//! GPU benchmark estimation for one already accepted Exponent-LUT refresh profile.
//!
//! This target deliberately does not search parameters, invoke Sage or a
//! lattice estimator, run a noise simulation, execute the graphs, or perform
//! a round trip.  It rebuilds the accepted symbolic graphs from a completed
//! Phase-2 checkpoint and measures representative GPU node shapes only.

use crate::parameter_search_test_support::{
    Candidate, Phase1Checkpoint, SearchConfig, hex, prepare_candidate_with_fixed_digits,
};
use mxx_bench_estimator::{
    CostReport, EstimateConfig, estimate, gpu::GpuNodeMeasurementBackend,
    harness::MeasurementHarnessConfig,
};
use mxx_ir_core::ParamEnv;
use mxx_primitives::{
    env::mul_decompose_column_chunk_width,
    poly::{
        PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
        },
    },
};
use serde::Deserialize;
use std::{env, fs, num::NonZeroUsize, path::PathBuf, time::Instant};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Deserialize)]
struct AcceptedReport {
    phase1: AcceptedPhase1,
    phase2: AcceptedPhase2,
    result: AcceptedResult,
}

#[derive(Deserialize)]
struct AcceptedPhase1 {
    selected_universe: usize,
    selected_q_l: usize,
    selected_p: usize,
    selected_weight: usize,
    selected_tier: String,
    estimator_commit: String,
    estimator_cost_model: String,
    estimator_shape_model: String,
}

#[derive(Deserialize)]
struct AcceptedPhase2 {
    security_target_bits: u64,
    noise_model: String,
}

#[derive(Deserialize)]
struct AcceptedResult {
    candidate: Candidate,
    achieved_security_bits: u64,
    sparse_lwr_universe: usize,
    sparse_lwr_weight: usize,
    sparse_lwr_q_l: usize,
    sparse_lwr_p: usize,
    sparse_lwr_tier: String,
    estimator_commit: String,
    estimator_cost_model: String,
    estimator_shape_model: String,
    layout_id: String,
    program_id: String,
    checker_accepted: bool,
    average_evidence: Option<AcceptedAverageEvidence>,
}

#[derive(Deserialize)]
struct AcceptedAverageEvidence {
    setup_identity: String,
    layout_id: String,
    program_id: String,
    mask_digit_count: usize,
    fresh_error_digit_count: usize,
    hard_authority_accepted: bool,
    correctness_accepted: bool,
    accepted: bool,
}

fn optional_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .map(|value| value.parse().unwrap_or_else(|_| panic!("{name} must be a positive integer")))
        .unwrap_or(default)
}

fn log_report(stage: &'static str, report: &CostReport) {
    info!(
        stage,
        total_work_seconds = report.total_work_seconds,
        preimage_sampling_work_seconds = report.preimage_sampling_work_seconds,
        critical_path_seconds = report.critical_path_seconds,
        maximum_parallelism = report.maximum_parallelism,
        workspace_high_water_bytes = report.workspace_high_water_bytes,
        peak_memory_bytes = report.peak_memory_bytes,
        "Exponent-LUT GPU benchmark estimate"
    );
    log_subgraph_breakdown(stage, report);
}

fn log_subgraph_breakdown(stage: &'static str, report: &CostReport) {
    let mut by_work = report.per_subgraph.iter().collect::<Vec<_>>();
    by_work.sort_by(|(_, left), (_, right)| {
        let left_total = left.work_seconds_per_invocation * left.invocations as f64;
        let right_total = right.work_seconds_per_invocation * right.invocations as f64;
        right_total.total_cmp(&left_total)
    });
    for (rank, (scope, cost)) in by_work.into_iter().take(32).enumerate() {
        info!(
            stage,
            rank,
            scope,
            invocations = cost.invocations,
            work_seconds_per_invocation = cost.work_seconds_per_invocation,
            total_work_seconds = cost.work_seconds_per_invocation * cost.invocations as f64,
            latency_seconds_per_invocation = cost.latency_seconds_per_invocation,
            peak_memory_bytes = cost.peak_memory_bytes,
            workspace_high_water_bytes = cost.workspace_high_water_bytes,
            maximum_parallelism = cost.maximum_parallelism,
            "Exponent-LUT GPU benchmark subgraph work breakdown"
        );
    }

    let mut by_memory = report.per_subgraph.iter().collect::<Vec<_>>();
    by_memory.sort_by_key(|(_, cost)| std::cmp::Reverse(cost.peak_memory_bytes));
    for (rank, (scope, cost)) in by_memory.into_iter().take(16).enumerate() {
        info!(
            stage,
            rank,
            scope,
            invocations = cost.invocations,
            peak_memory_bytes = cost.peak_memory_bytes,
            workspace_high_water_bytes = cost.workspace_high_water_bytes,
            "Exponent-LUT GPU benchmark subgraph memory breakdown"
        );
    }
}

fn log_preprocess_report(select_setup: &CostReport, preprocessing: &CostReport) {
    info!(
        stage = "preprocess",
        total_work_seconds = select_setup.total_work_seconds + preprocessing.total_work_seconds,
        preimage_sampling_work_seconds = select_setup.preimage_sampling_work_seconds +
            preprocessing.preimage_sampling_work_seconds,
        critical_path_seconds =
            select_setup.critical_path_seconds + preprocessing.critical_path_seconds,
        maximum_parallelism =
            select_setup.maximum_parallelism.max(preprocessing.maximum_parallelism),
        workspace_high_water_bytes =
            select_setup.workspace_high_water_bytes.max(preprocessing.workspace_high_water_bytes),
        peak_memory_bytes = select_setup.peak_memory_bytes.max(preprocessing.peak_memory_bytes),
        select_setup_work_seconds = select_setup.total_work_seconds,
        refresh_preprocessing_work_seconds = preprocessing.total_work_seconds,
        "Exponent-LUT GPU benchmark estimate"
    );
    log_subgraph_breakdown("select_setup", select_setup);
    log_subgraph_breakdown("refresh_preprocessing", preprocessing);
}

fn log_prf_work_per_output_bit(
    stage: &'static str,
    report: &CostReport,
    prf_label_count: usize,
    output_bits_per_label: usize,
) {
    assert!(prf_label_count > 0 && output_bits_per_label > 0);
    let (scope, cost) = report
        .per_subgraph
        .iter()
        .filter(|(_, cost)| cost.invocations == prf_label_count)
        .max_by(|(_, left), (_, right)| {
            left.work_seconds_per_invocation.total_cmp(&right.work_seconds_per_invocation)
        })
        .unwrap_or_else(|| {
            panic!("{stage} must contain one complete label-major PRF evaluation subgraph")
        });
    let total_output_bits =
        prf_label_count.checked_mul(output_bits_per_label).expect("PRF output-bit count");
    let total_prf_work_seconds = cost.work_seconds_per_invocation * prf_label_count as f64;
    info!(
        stage,
        scope,
        prf_label_count,
        output_bits_per_label,
        total_output_bits,
        total_prf_work_seconds,
        work_seconds_per_prf_output_bit = total_prf_work_seconds / total_output_bits as f64,
        "Exponent-LUT PRF work per output bit"
    );
}

#[test]
#[ignore = "accepted Exponent-LUT GPU benchmark estimate; no simulation or round trip"]
fn test_gpu_exponent_lut_benchmark_estimate() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("mxx_exponent_lut=debug,mxx_bench_estimator=debug,mxx_runtime=info,info")
    });
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();

    let checkpoint_path =
        env::var_os("MXX_EXPONENT_LUT_REFRESH_PHASE1_CHECKPOINT").map(PathBuf::from).expect(
            "MXX_EXPONENT_LUT_REFRESH_PHASE1_CHECKPOINT must name a completed Phase-2 checkpoint",
        );
    let config = SearchConfig::from_env().expect("valid accepted Exponent-LUT profile overlay");
    let checkpoint: Phase1Checkpoint = serde_json::from_slice(
        &fs::read(&checkpoint_path).expect("read accepted Exponent-LUT checkpoint"),
    )
    .expect("parse accepted Exponent-LUT checkpoint");
    checkpoint.validate(&config).expect("validate accepted Exponent-LUT checkpoint");
    let accepted = checkpoint
        .accepted_phase2
        .as_ref()
        .expect("Exponent-LUT checkpoint must contain an accepted Phase-2 candidate");
    assert_eq!(
        accepted.tuple, checkpoint.selected.tuple,
        "accepted tuple must be the selected Phase-1 tuple"
    );

    let report_path = env::var_os("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT")
        .map(PathBuf::from)
        .expect("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT must name the accepted AverageCase report");
    let report: AcceptedReport =
        serde_json::from_slice(&fs::read(&report_path).expect("read accepted Exponent-LUT report"))
            .expect("parse accepted Exponent-LUT report");
    assert_eq!(report.phase2.noise_model, "AverageCase");
    assert_eq!(report.phase2.security_target_bits, config.security_bits);
    assert_eq!(report.result.achieved_security_bits, config.security_bits);
    assert_eq!(report.result.candidate, accepted.candidate);
    assert_eq!(report.phase1.selected_universe, accepted.tuple.nu);
    assert_eq!(report.phase1.selected_q_l, accepted.tuple.q_l);
    assert_eq!(report.phase1.selected_p, accepted.tuple.p);
    assert_eq!(report.phase1.selected_weight, accepted.tuple.h);
    assert_eq!(report.result.sparse_lwr_universe, accepted.tuple.nu);
    assert_eq!(report.result.sparse_lwr_weight, accepted.tuple.h);
    assert_eq!(report.result.sparse_lwr_q_l, accepted.tuple.q_l);
    assert_eq!(report.result.sparse_lwr_p, accepted.tuple.p);
    let selected_tier = format!("{:?}", checkpoint.selected.tier);
    assert_eq!(report.phase1.selected_tier, selected_tier);
    assert_eq!(report.result.sparse_lwr_tier, selected_tier);
    assert_eq!(report.phase1.estimator_commit, checkpoint.selected.estimator_commit);
    assert_eq!(report.phase1.estimator_cost_model, checkpoint.selected.estimator_cost_model);
    assert_eq!(report.phase1.estimator_shape_model, checkpoint.selected.estimator_shape_model);
    assert_eq!(report.result.estimator_commit, checkpoint.selected.estimator_commit);
    assert_eq!(report.result.estimator_cost_model, checkpoint.selected.estimator_cost_model);
    assert_eq!(report.result.estimator_shape_model, checkpoint.selected.estimator_shape_model);
    assert!(report.result.checker_accepted, "accepted report must pass its checker");
    let evidence = report
        .result
        .average_evidence
        .as_ref()
        .expect("benchmark requires an accepted AverageCase report");
    assert!(
        evidence.accepted && evidence.hard_authority_accepted && evidence.correctness_accepted,
        "AverageCase report must be fully accepted"
    );
    assert_eq!(evidence.layout_id, report.result.layout_id);
    assert_eq!(evidence.program_id, report.result.program_id);
    let mask_digits = evidence.mask_digit_count;
    let fresh_error_digits = evidence.fresh_error_digit_count;
    let graph_started = Instant::now();
    let prepared = prepare_candidate_with_fixed_digits(
        &config,
        accepted.candidate,
        mask_digits,
        fresh_error_digits,
    )
    .expect("build accepted Exponent-LUT graphs without simulation");
    let bundle = prepared.bundle.as_ref().expect("accepted candidate bundle");
    assert_eq!(hex(prepared.layout_id), report.result.layout_id);
    assert_eq!(hex(prepared.program_id), report.result.program_id);
    assert_eq!(hex(*bundle.public_identity()), evidence.setup_identity);
    let bindings = ParamEnv::default();
    let stages = [
        (
            "select_setup",
            bundle.selector_graph().validate(&bindings).expect("validate selector graph"),
        ),
        (
            "preprocessing",
            bundle
                .benchmark_preprocessing_graph()
                .validate_with_manifests(&bindings, bundle.preprocessing_manifests())
                .expect("validate preprocessing graph"),
        ),
        (
            "online_eval",
            bundle
                .benchmark_online_graph()
                .validate_with_manifests(&bindings, bundle.verification_manifests())
                .expect("validate online evaluation graph"),
        ),
    ];
    info!(
        checkpoint = %checkpoint_path.display(),
        accepted_report = %report_path.display(),
        crt_depth = accepted.candidate.crt_depth,
        log_ring_dimension = accepted.candidate.log_ring_dimension,
        crt_bits = accepted.candidate.crt_bits,
        base_bits = accepted.candidate.base_bits,
        mask_digits,
        fresh_error_digits,
        prf_crt_slot_count = accepted.candidate.crt_depth,
        prf_component_count = prepared.prf_component_count,
        prf_coefficient_count = prepared.prf_coefficient_count,
        prf_active_count = prepared.prf_active_count,
        prf_label_count = prepared.prf_label_count,
        prf_value_count = prepared.prf_value_count,
        prf_mask_label_count = accepted.candidate.crt_depth *
            prepared.prf_component_count *
            prepared.prf_coefficient_count *
            mask_digits,
        prf_fresh_label_count = prepared.prf_component_count *
            prepared.prf_coefficient_count *
            fresh_error_digits,
        graph_construction_seconds = graph_started.elapsed().as_secs_f64(),
        parameter_simulation = false,
        lattice_estimator = false,
        round_trip = false,
        runtime_execution = false,
        "built accepted Exponent-LUT benchmark graphs"
    );

    let ring_dimension = 1usize
        .checked_shl(accepted.candidate.log_ring_dimension as u32)
        .expect("accepted ring dimension");
    let parameters = DCRTPolyParams::new(
        ring_dimension as u32,
        accepted.candidate.crt_depth,
        accepted.candidate.crt_bits,
        accepted.candidate.base_bits,
    );
    let (moduli, _, crt_depth) = parameters.to_crt();
    let device_ids = detected_gpu_device_ids();
    assert!(!device_ids.is_empty(), "Exponent-LUT benchmark estimate requires a CUDA GPU");
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits());
    let warm_up_iterations = optional_usize("MXX_EXPONENT_LUT_BENCH_WARMUPS", 1);
    let measured_iterations = optional_usize("MXX_EXPONENT_LUT_BENCH_ITERATIONS", 2);
    assert!(measured_iterations > 0, "benchmark iterations must be positive");
    let parallel_instances = NonZeroUsize::new(optional_usize(
        "MXX_EXPONENT_LUT_BENCH_PARALLEL_INSTANCES",
        device_ids.len(),
    ))
    .expect("parallel instance count must be positive");
    let column_chunk_width = mul_decompose_column_chunk_width();
    let estimate_config =
        EstimateConfig { device_pool_size: parallel_instances.get(), per_instance_occupancy: 1 };
    let mut backend = GpuNodeMeasurementBackend::new(
        &gpu_parameters,
        device_ids.clone(),
        MeasurementHarnessConfig {
            warm_up_iterations,
            measured_iterations,
            ..MeasurementHarnessConfig::default()
        },
        crt_depth,
        column_chunk_width,
        2,
    );
    info!(
        gpu_count = device_ids.len(),
        ?device_ids,
        parallel_instances = parallel_instances.get(),
        column_chunk_width,
        warm_up_iterations,
        measured_iterations,
        "Exponent-LUT benchmark estimator configuration"
    );

    for (stage, graph) in &stages {
        estimate(graph, &mut backend, &estimate_config)
            .unwrap_or_else(|error| panic!("collect {stage} GPU shapes: {error}"));
    }
    let measurement_started = Instant::now();
    backend.measure_collected().expect("measure Exponent-LUT GPU node shapes");
    info!(
        elapsed_seconds = measurement_started.elapsed().as_secs_f64(),
        "Exponent-LUT GPU representative measurement collection complete"
    );

    let reports = stages
        .iter()
        .map(|(stage, graph)| {
            let report = estimate(graph, &mut backend, &estimate_config)
                .unwrap_or_else(|error| panic!("estimate {stage} Exponent-LUT graph: {error}"));
            assert!(report.total_work_seconds > 0.0, "{stage} estimate must contain GPU work");
            report
        })
        .collect::<Vec<_>>();
    log_preprocess_report(&reports[0], &reports[1]);
    log_report("online_eval", &reports[2]);
    // The accepted profile has p=2, so every canonical label produces one PRF output bit.
    log_prf_work_per_output_bit("public_key_prf", &reports[1], prepared.prf_label_count, 1);
    log_prf_work_per_output_bit("encoding_prf", &reports[2], prepared.prf_label_count, 1);
}
