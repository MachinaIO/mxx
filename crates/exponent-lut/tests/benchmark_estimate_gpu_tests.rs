//! GPU benchmark estimation for one already accepted Exponent-LUT refresh profile.
//!
//! This target deliberately does not search parameters, invoke Sage or a
//! lattice estimator, run a noise simulation, execute the graphs, or perform
//! a round trip.  It rebuilds the accepted symbolic graphs from a completed
//! Phase-2 checkpoint and measures representative GPU node shapes only.

#![cfg(feature = "gpu")]

#[allow(dead_code)]
mod test_utils;

use mxx_bench_estimator::{
    CostReport, estimate, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig,
};
use mxx_exponent_lut::ExponentLutStorageReport;
use mxx_ir_core::{BenchmarkRole, Graph, ParamEnv};
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{
        gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
        params::DCRTPolyParams,
    },
};
use mxx_runtime::backend::poly::gpu::gpu_backend_on;
use serde::Deserialize;
use std::{env, fs, path::PathBuf, time::Instant};
use test_utils::{
    Candidate, Phase1Checkpoint, SearchConfig, hex, prepare_candidate_with_fixed_digits,
};
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

fn graph_scope_and_node_counts(graph: &Graph) -> (usize, usize) {
    let scope_count = graph.scopes().len();
    let node_count = graph.scopes().values().map(|scope| scope.nodes().len()).sum();
    (scope_count, node_count)
}

fn log_report(stage: &'static str, report: &CostReport) {
    info!(
        stage,
        total_work_seconds = report.total_work_seconds,
        total_time_seconds = report.total_time_seconds,
        estimated_dag_latency_seconds = report.critical_path_seconds,
        preimage_sampling_work_seconds = report.preimage_sampling_work_seconds,
        critical_path_seconds = report.critical_path_seconds,
        maximum_parallelism = report.maximum_parallelism,
        chunk_count = report.chunk_count,
        transmitted_bytes = report.transmitted_bytes,
        cache_bytes = report.cache_bytes,
        persistent_storage_bytes = report.persistent_storage_bytes,
        resident_vram_bytes = report.resident_vram_bytes,
        expanded_workspace_bytes = report.expanded_workspace_bytes,
        output_bytes = report.output_bytes,
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
            total_time_seconds = cost.total_time_seconds_per_invocation * cost.invocations as f64,
            latency_seconds_per_invocation = cost.latency_seconds_per_invocation,
            peak_memory_bytes = cost.peak_memory_bytes,
            workspace_high_water_bytes = cost.workspace_high_water_bytes,
            maximum_parallelism = cost.maximum_parallelism,
            chunk_count = cost.chunk_count_per_invocation,
            transmitted_bytes = cost.transmitted_bytes_per_invocation,
            cache_bytes = cost.cache_bytes_per_invocation,
            persistent_storage_bytes = cost.persistent_storage_bytes_per_invocation,
            resident_vram_bytes = cost.resident_vram_bytes,
            expanded_workspace_bytes = cost.expanded_workspace_bytes,
            output_bytes = cost.output_bytes_per_invocation,
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

fn log_preprocess_report(preprocessing: &CostReport) {
    info!(
        stage = "preprocess",
        total_work_seconds = preprocessing.total_work_seconds,
        total_time_seconds = preprocessing.total_time_seconds,
        estimated_dag_latency_seconds = preprocessing.critical_path_seconds,
        preimage_sampling_work_seconds = preprocessing.preimage_sampling_work_seconds,
        critical_path_seconds = preprocessing.critical_path_seconds,
        maximum_parallelism = preprocessing.maximum_parallelism,
        chunk_count = preprocessing.chunk_count,
        transmitted_bytes = preprocessing.transmitted_bytes,
        cache_bytes = preprocessing.cache_bytes,
        persistent_storage_bytes = preprocessing.persistent_storage_bytes,
        resident_vram_bytes = preprocessing.resident_vram_bytes,
        expanded_workspace_bytes = preprocessing.expanded_workspace_bytes,
        output_bytes = preprocessing.output_bytes,
        workspace_high_water_bytes = preprocessing.workspace_high_water_bytes,
        peak_memory_bytes = preprocessing.peak_memory_bytes,
        includes_select_setup = true,
        includes_public_key_program = true,
        includes_decoder_preimages = true,
        component_work_is_not_separately_measured = true,
        "Exponent-LUT GPU benchmark estimate"
    );
    log_subgraph_breakdown("preprocess", preprocessing);
}

fn log_storage_report(storage: &ExponentLutStorageReport) {
    info!(
        stage = "storage",
        canonical_public_artifact_bytes = storage.canonical_public_artifact_bytes,
        private_label_input_bytes = storage.private_label_input_bytes,
        decoder_preimage_bytes = storage.decoder_preimage_bytes,
        decomposition_cache_artifact_bytes = storage.decomposition_cache_artifact_bytes,
        peak_loaded_host_bytes = storage.peak_loaded_host_bytes,
        peak_loaded_device_bytes_per_gpu = storage.peak_loaded_device_bytes_per_gpu,
        peak_workspace_bytes_per_gpu = storage.peak_workspace_bytes_per_gpu,
        "Exponent-LUT storage accounting"
    );
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PrfCost {
    accumulation_work_seconds: f64,
    accumulation_latency_seconds: f64,
    value_readout_work_seconds: f64,
    value_readout_latency_seconds: f64,
}

impl PrfCost {
    fn total_work_seconds(self) -> f64 {
        self.accumulation_work_seconds + self.value_readout_work_seconds
    }

    fn total_latency_seconds(self) -> f64 {
        self.accumulation_latency_seconds + self.value_readout_latency_seconds
    }
}

/// Sums only costs explicitly tagged by the graph producer as PRF accumulation
/// or value readout. Missing role tags are an error rather than an estimate;
/// decoder, refresh, and other untagged work therefore cannot enter the PRF
/// numerator accidentally.
fn sum_explicit_prf_roles(
    report: &CostReport,
    accumulation_role: BenchmarkRole,
    readout_role: BenchmarkRole,
) -> Result<PrfCost, &'static str> {
    if !report.benchmark_roles.contains_key(&accumulation_role) ||
        !report.benchmark_roles.contains_key(&readout_role) ||
        report
            .benchmark_roles
            .keys()
            .any(|role| *role != accumulation_role && *role != readout_role)
    {
        return Err("estimator report is missing the exact stage PRF role tags");
    }
    let accumulation =
        report.benchmark_roles.get(&accumulation_role).expect("validated accumulation role");
    let readout = report.benchmark_roles.get(&readout_role).expect("validated readout role");
    let mut cost = PrfCost {
        accumulation_work_seconds: 0.0,
        accumulation_latency_seconds: 0.0,
        value_readout_work_seconds: 0.0,
        value_readout_latency_seconds: 0.0,
    };
    cost.accumulation_work_seconds = accumulation.work_seconds;
    cost.accumulation_latency_seconds = accumulation.total_time_seconds;
    cost.value_readout_work_seconds = readout.work_seconds;
    cost.value_readout_latency_seconds = readout.total_time_seconds;
    Ok(cost)
}

fn native_information_bits_per_output(base_p: usize) -> usize {
    assert!(base_p >= 2 && base_p.is_power_of_two(), "native P must be a power of two");
    base_p.trailing_zeros() as usize
}

fn log_prf_cost_per_output_and_information_bit(
    stage: &'static str,
    cost: PrfCost,
    prf_output_count: usize,
    base_p: usize,
) {
    assert!(prf_output_count > 0, "PRF output count must be positive");
    let information_bits_per_output = native_information_bits_per_output(base_p);
    let total_information_bits = prf_output_count
        .checked_mul(information_bits_per_output)
        .expect("PRF information-bit count");
    let total_work_seconds = cost.total_work_seconds();
    let total_latency_seconds = cost.total_latency_seconds();
    info!(
        stage,
        base_p,
        information_bits_per_output,
        prf_output_count,
        total_information_bits,
        total_prf_work_seconds = total_work_seconds,
        total_prf_latency_seconds = total_latency_seconds,
        work_seconds_per_prf_output = total_work_seconds / prf_output_count as f64,
        // Retained raw fields sum calibrated wave latencies. They do not represent
        // DAG wall completion or isolated-output latency under unlimited scheduling.
        latency_accounting = "cumulative_wave_latencies",
        cumulative_wave_seconds_per_prf_output = total_latency_seconds / prf_output_count as f64,
        latency_seconds_per_prf_output = total_latency_seconds / prf_output_count as f64,
        amortized_latency_seconds_per_prf_output = total_latency_seconds / prf_output_count as f64,
        work_seconds_per_information_bit = total_work_seconds / total_information_bits as f64,
        latency_seconds_per_information_bit = total_latency_seconds / total_information_bits as f64,
        accumulation_work_seconds = cost.accumulation_work_seconds,
        accumulation_latency_seconds = cost.accumulation_latency_seconds,
        value_readout_work_seconds = cost.value_readout_work_seconds,
        value_readout_latency_seconds = cost.value_readout_latency_seconds,
        "Exponent-LUT native PRF accounting"
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
    let mut config = SearchConfig::from_env().expect("valid accepted Exponent-LUT profile overlay");
    let checkpoint: Phase1Checkpoint = serde_json::from_slice(
        &fs::read(&checkpoint_path).expect("read accepted Exponent-LUT checkpoint"),
    )
    .expect("parse accepted Exponent-LUT checkpoint");
    checkpoint.validate(&config).expect("validate accepted Exponent-LUT checkpoint");
    let accepted = checkpoint
        .accepted_phase2
        .as_ref()
        .expect("Exponent-LUT checkpoint must contain an accepted Phase-2 candidate");
    // Phase 1's `selected` profile is only the first qualifying tuple under
    // the Phase-1 security policy.  Phase 2 evaluates every qualifying tuple
    // and may accept a different one when it has the globally smallest CRT
    // depth.  The checkpoint validator already proves that the accepted tuple
    // is declared, qualified, and consistent with the retained Phase-1 rows;
    // it must not be confused with the Phase-1 representative.

    let report_path = env::var_os("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT")
        .map(PathBuf::from)
        .expect("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT must name the accepted AverageCase report");
    let report: AcceptedReport =
        serde_json::from_slice(&fs::read(&report_path).expect("read accepted Exponent-LUT report"))
            .expect("parse accepted Exponent-LUT report");
    assert_eq!(report.phase2.noise_model, "AverageCase");
    assert_eq!(report.phase2.security_target_bits, config.security_bits);
    assert!(
        report.result.achieved_security_bits >= config.security_bits,
        "accepted report must achieve at least the configured security target"
    );
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
    // Reconstruct the graph for the Phase-2 winner, rather than for the
    // Phase-1 representative or for an environment default that happens to
    // differ from the winner's native (Q, P, nu, h) tuple.
    config.sparse_lwr_universe = accepted.tuple.nu;
    config.sparse_lwr_weight = accepted.tuple.h;
    config.sparse_lwr_modulus = accepted.tuple.q_l;
    config.sparse_lwr_output_modulus = accepted.tuple.p;
    info!(
        crt_depth = accepted.candidate.crt_depth,
        log_ring_dimension = accepted.candidate.log_ring_dimension,
        crt_bits = accepted.candidate.crt_bits,
        base_bits = accepted.candidate.base_bits,
        native_q = accepted.tuple.q_l,
        native_p = accepted.tuple.p,
        sparse_lwr_universe = accepted.tuple.nu,
        sparse_lwr_weight = accepted.tuple.h,
        mask_digits,
        fresh_error_digits,
        "starting accepted Exponent-LUT candidate graph preparation"
    );
    let graph_started = Instant::now();
    let prepared = prepare_candidate_with_fixed_digits(
        &config,
        accepted.candidate,
        mask_digits,
        fresh_error_digits,
    )
    .expect("build accepted Exponent-LUT graphs without simulation");
    info!(
        elapsed_seconds = graph_started.elapsed().as_secs_f64(),
        "accepted Exponent-LUT candidate graph preparation complete"
    );
    let bundle_started = Instant::now();
    let bundle = prepared.bundle.as_ref().expect("accepted candidate bundle");
    let preprocessing_graph = bundle.benchmark_preprocessing_graph();
    let online_graph = bundle.benchmark_online_graph();
    let (preprocessing_scope_count, preprocessing_node_count) =
        graph_scope_and_node_counts(&preprocessing_graph.graph);
    let (online_scope_count, online_node_count) = graph_scope_and_node_counts(&online_graph.graph);
    info!(
        elapsed_seconds = bundle_started.elapsed().as_secs_f64(),
        preprocessing_scope_count,
        preprocessing_node_count,
        online_scope_count,
        online_node_count,
        "acquired accepted Exponent-LUT graph bundle"
    );
    let identity_started = Instant::now();
    info!("starting accepted Exponent-LUT graph identity checks");
    let current_layout_id = hex(prepared.layout_id);
    let current_program_id = hex(prepared.program_id);
    let current_setup_identity = hex(*bundle.public_identity());
    assert_eq!(
        current_layout_id, report.result.layout_id,
        "reconstructed layout identity must match the accepted report"
    );
    assert_eq!(
        current_program_id, report.result.program_id,
        "reconstructed program identity must match the accepted report"
    );
    assert_eq!(
        current_setup_identity, evidence.setup_identity,
        "reconstructed setup identity must match the accepted report"
    );
    info!(
        elapsed_seconds = identity_started.elapsed().as_secs_f64(),
        "accepted Exponent-LUT graph identity checks complete"
    );
    let bindings = ParamEnv::default();
    // Selector setup is part of the producer preprocessing graph. Keep one
    // stage entry so its t-dependent work and outputs are never estimated
    // twice under separate `select_setup` and `preprocessing` labels.
    let preprocessing_validation_started = Instant::now();
    info!(
        scope_count = preprocessing_scope_count,
        node_count = preprocessing_node_count,
        "starting Exponent-LUT preprocessing graph validation"
    );
    let preprocessing = preprocessing_graph
        .validate_with_manifests(&bindings, bundle.preprocessing_manifests())
        .expect("validate preprocessing graph");
    let (validated_preprocessing_scope_count, validated_preprocessing_node_count) =
        graph_scope_and_node_counts(&preprocessing.source);
    info!(
        elapsed_seconds = preprocessing_validation_started.elapsed().as_secs_f64(),
        scope_count = validated_preprocessing_scope_count,
        node_count = validated_preprocessing_node_count,
        "Exponent-LUT preprocessing graph validation complete"
    );
    let online_validation_started = Instant::now();
    info!(
        scope_count = online_scope_count,
        node_count = online_node_count,
        "starting Exponent-LUT online graph validation"
    );
    let online = online_graph
        .validate_with_manifests(&bindings, bundle.verification_manifests())
        .expect("validate online evaluation graph");
    let (validated_online_scope_count, validated_online_node_count) =
        graph_scope_and_node_counts(&online.source);
    info!(
        elapsed_seconds = online_validation_started.elapsed().as_secs_f64(),
        scope_count = validated_online_scope_count,
        node_count = validated_online_node_count,
        "Exponent-LUT online graph validation complete"
    );
    let stages = [("preprocess", preprocessing), ("oneline_eval", online)];
    info!(
        checkpoint = %checkpoint_path.display(),
        accepted_report = %report_path.display(),
        crt_depth = accepted.candidate.crt_depth,
        log_ring_dimension = accepted.candidate.log_ring_dimension,
        crt_bits = accepted.candidate.crt_bits,
        base_bits = accepted.candidate.base_bits,
        native_p = config.sparse_lwr_output_modulus,
        native_p_information_bits = native_information_bits_per_output(config.sparse_lwr_output_modulus),
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
        accepted_layout_id = report.result.layout_id,
        current_layout_id,
        layout_id_matches = current_layout_id == report.result.layout_id,
        accepted_program_id = report.result.program_id,
        current_program_id,
        program_id_matches = current_program_id == report.result.program_id,
        accepted_setup_identity = evidence.setup_identity,
        current_setup_identity,
        setup_identity_matches = current_setup_identity == evidence.setup_identity,
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
    let backends = device_ids
        .iter()
        .copied()
        .map(|device_id| (gpu_backend_on([gpu_parameters.clone()], [device_id]), device_id))
        .collect();
    let mut backend = GpuNodeMeasurementBackend::new(
        backends,
        MeasurementHarnessConfig {
            warm_up_iterations,
            measured_iterations,
            ..MeasurementHarnessConfig::default()
        },
        crt_depth,
    );
    info!(
        gpu_count = device_ids.len(),
        ?device_ids,
        warm_up_iterations,
        measured_iterations,
        "Exponent-LUT benchmark estimator configuration"
    );

    for (stage, graph) in &stages {
        let collection_started = Instant::now();
        let (scope_count, node_count) = graph_scope_and_node_counts(&graph.source);
        info!(stage, scope_count, node_count, "starting first Exponent-LUT estimate collection");
        let collected = estimate(graph, &mut backend)
            .unwrap_or_else(|error| panic!("collect {stage} GPU shapes: {error}"));
        info!(
            stage,
            elapsed_seconds = collection_started.elapsed().as_secs_f64(),
            collected_total_work_seconds = collected.total_work_seconds,
            collected_total_time_seconds = collected.total_time_seconds,
            "first Exponent-LUT estimate collection complete"
        );
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
            let report = estimate(graph, &mut backend)
                .unwrap_or_else(|error| panic!("estimate {stage} Exponent-LUT graph: {error}"));
            assert!(report.total_work_seconds > 0.0, "{stage} estimate must contain GPU work");
            report
        })
        .collect::<Vec<_>>();
    log_preprocess_report(&reports[0]);
    log_report("oneline_eval", &reports[1]);
    let peak_device_bytes =
        reports.iter().map(|report| report.resident_vram_bytes).max().unwrap_or_default();
    let peak_workspace_bytes =
        reports.iter().map(|report| report.expanded_workspace_bytes).max().unwrap_or_default();
    // Manifests account for logical protocol/cache storage and bounded host
    // member loading. Device/workspace peaks come from the same measured
    // reports used by the estimator. Do not replace the manifest-derived host
    // peak with a whole-graph persistent byte count.
    let storage = ExponentLutStorageReport::from_metadata(
        bundle.preprocessing_manifests().values().chain(bundle.verification_manifests().values()),
        stages.iter().map(|(_, graph)| graph),
    )
    .with_residency(0, peak_device_bytes, peak_workspace_bytes);
    log_storage_report(&storage);
    let public_cost = sum_explicit_prf_roles(
        &reports[0],
        BenchmarkRole::PublicPrfAccumulation,
        BenchmarkRole::PublicReadout,
    )
    .unwrap_or_else(|error| panic!("public-key PRF role accounting unavailable: {error}"));
    log_prf_cost_per_output_and_information_bit(
        "public_key_prf",
        public_cost,
        prepared.prf_label_count,
        config.sparse_lwr_output_modulus,
    );
    let encoding_cost = sum_explicit_prf_roles(
        &reports[1],
        BenchmarkRole::EncodingPrfAccumulation,
        BenchmarkRole::EncodingReadout,
    )
    .unwrap_or_else(|error| panic!("encoding PRF role accounting unavailable: {error}"));
    log_prf_cost_per_output_and_information_bit(
        "encoding_prf",
        encoding_cost,
        prepared.prf_label_count,
        config.sparse_lwr_output_modulus,
    );
}

#[cfg(test)]
mod accounting_tests {
    use super::*;
    use mxx_bench_estimator::SubgraphCost;

    #[test]
    fn native_p_derives_information_bits_without_binary_prf_multiplier() {
        assert_eq!(native_information_bits_per_output(16), 4);
        assert_eq!(native_information_bits_per_output(32), 5);
    }

    #[test]
    fn explicit_prf_roles_sum_work_and_latency_only() {
        let mut report = CostReport::default();
        report.per_subgraph.insert(
            "accumulation".to_owned(),
            SubgraphCost {
                invocations: 2,
                work_seconds_per_invocation: 3.0,
                total_time_seconds_per_invocation: 5.0,
                ..SubgraphCost::default()
            },
        );
        report.per_subgraph.insert(
            "readout".to_owned(),
            SubgraphCost {
                invocations: 4,
                work_seconds_per_invocation: 7.0,
                total_time_seconds_per_invocation: 11.0,
                ..SubgraphCost::default()
            },
        );
        report.per_subgraph.insert(
            "decoder".to_owned(),
            SubgraphCost {
                invocations: 100,
                work_seconds_per_invocation: 1000.0,
                total_time_seconds_per_invocation: 1000.0,
                ..SubgraphCost::default()
            },
        );
        report.benchmark_roles.insert(
            BenchmarkRole::PublicPrfAccumulation,
            mxx_bench_estimator::BenchmarkRoleCost { work_seconds: 6.0, total_time_seconds: 10.0 },
        );
        report.benchmark_roles.insert(
            BenchmarkRole::PublicReadout,
            mxx_bench_estimator::BenchmarkRoleCost { work_seconds: 28.0, total_time_seconds: 44.0 },
        );
        let cost = sum_explicit_prf_roles(
            &report,
            BenchmarkRole::PublicPrfAccumulation,
            BenchmarkRole::PublicReadout,
        )
        .unwrap();
        assert_eq!(cost.total_work_seconds(), 34.0);
        assert_eq!(cost.total_latency_seconds(), 54.0);
        log_prf_cost_per_output_and_information_bit("test", cost, 2, 16);
        report.benchmark_roles.clear();
        assert!(
            sum_explicit_prf_roles(
                &report,
                BenchmarkRole::PublicPrfAccumulation,
                BenchmarkRole::PublicReadout,
            )
            .is_err()
        );
    }
}
