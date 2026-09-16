#![cfg(feature = "gpu")]

//! Noise-refresh simulation and benchmark estimation for the selected encoding scheme.

use std::{env, path::PathBuf};
use tracing_subscriber::EnvFilter;

fn use_rns_bgg() -> bool {
    match std::env::var("MXX_EXPONENT_LUT_TEST_SCHEME").as_deref().unwrap_or("bgg") {
        "bgg" => false,
        "rns-bgg" => true,
        other => panic!("MXX_EXPONENT_LUT_TEST_SCHEME must be bgg or rns-bgg, got {other}"),
    }
}

fn install_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("mxx_exponent_lut=debug,mxx_runtime=debug,info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
}

fn simulate_refresh() -> (SearchConfig, serde_json::Value) {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sparse_lwr_estimator.sage.py");
    let mut config = SearchConfig::from_env().expect("valid Exponent-LUT refresh profile");
    config.pbc_max_attempts = 128;
    config.noise_model = NoiseSearchModel::from_env().expect("noise simulation model");
    let checkpoint = env::var_os(PHASE1_CHECKPOINT_ENV).map(PathBuf::from);
    // Reviewed tuples carry the exact result of the pinned estimator run.
    // Keep Sage inputs lazy: the normal reviewed search never needs an
    // estimator process, while an evidence-free custom tuple still requires
    // the explicitly configured external estimator.
    let mut estimator_inputs = None;
    let selected = load_or_search_phase1(&config, checkpoint.as_deref(), |u, h, q, p| {
        let tuple = config
            .sparse_lwr_phase1_grid
            .iter()
            .find(|t| t.nu == u && t.h == h && t.q_l == q && t.p == p)
            .ok_or_else(|| "estimator callback received undeclared tuple".to_owned())?;
        if let Some(security_bits) = pinned_estimator_security_bits(tuple)? {
            return Ok(security_bits);
        }

        if estimator_inputs.is_none() {
            let estimator_root = env::var_os("MXX_LATTICE_ESTIMATOR_ROOT")
                .map(PathBuf::from)
                .ok_or_else(|| {
                    "MXX_LATTICE_ESTIMATOR_ROOT must point to pinned checkout for an evidence-free Phase-1 tuple"
                        .to_owned()
                })?;
            let sage_binary = env::var_os("MXX_SAGE_BINARY")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("sage"));
            estimator_inputs = Some((estimator_root, sage_binary));
        }
        let (estimator_root, sage_binary) = estimator_inputs
            .as_ref()
            .expect("estimator inputs initialized for evidence-free tuple");
        let report = run_estimator(
            &tuple.candidate()?,
            &script,
            estimator_root,
            sage_binary,
            Some(&config.phase1_estimator_commit),
        )?;
        if report.cost_model != config.phase1_estimator_cost_model ||
            report.shape_model != config.phase1_estimator_shape_model
        {
            return Err("estimator result disagrees with the configured cost or shape model".to_owned());
        }
        Ok(report.minimum_classical_bits.floor() as u64)
    })
    .expect("Phase-1 search must select a profile");
    let profile_results = search_all_qualified_profiles(&selected, |profile| {
        config.sparse_lwr_universe = profile.universe;
        config.sparse_lwr_weight = profile.weight;
        config.sparse_lwr_modulus = profile.q_l;
        config.sparse_lwr_output_modulus = profile.p;
        search_with_hooks(
            &config,
            &profile,
            |candidate| prepare_candidate(&config, candidate),
            |candidate| {
                if config.bgg_security_envelope.is_some() {
                    return reviewed_bgg_security_bits(&config, candidate);
                }
                let n = 1usize
                    .checked_shl(candidate.log_ring_dimension as u32)
                    .ok_or_else(|| "ring dimension overflow".to_owned())?;
                let dcrt = DCRTPolyParams::new(
                    n as u32,
                    candidate.crt_depth,
                    candidate.crt_bits,
                    candidate.base_bits,
                    None,
                    None,
                );
                lattice_security_bits(n, dcrt.modulus().as_ref(), config.error_sigma)
            },
            |prepared| evaluate_refresh_bundle(&config, prepared.candidate, prepared),
        )
    })
    .expect("Phase-2 search must produce candidates or canonical no-candidate rows");
    let (selected, result) = select_best_profile_result(profile_results.clone())
        .expect("Phase-2 search must produce a candidate");
    if let Some(path) = checkpoint.as_deref() {
        let evidence = if config.noise_model == NoiseSearchModel::WorstCase {
            Some(
                Phase2CertificationEvidence::certify(&config, &selected, result.candidate)
                    .expect("certify exact persisted Phase-2 profile"),
            )
        } else {
            None
        };
        persist_accepted_phase2_profile(
            path,
            &config,
            &selected,
            result.candidate,
            evidence.as_ref(),
        )
        .expect("persist Phase-2 profile");
    }
    config.crt_bits = result.candidate.crt_bits;
    config.base_bits = result.candidate.base_bits;
    let report = serde_json::to_value(search_report_with_profiles(
        &config,
        &selected,
        result,
        profile_results,
    ))
    .unwrap();
    if let Some(path) = env::var_os("MXX_EXPONENT_LUT_REFRESH_SEARCH_OUTPUT") {
        fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).expect("write search report");
    } else {
        println!("{report}");
    }
    (config, report)
}

// GPU benchmark estimation for one already accepted Exponent-LUT refresh profile.
//
// The integration first simulates the selected setup, then measures node
// shapes from its preprocessing and online graphs. Measurement uses the
// same production graphs, excluding correctness-oracle arithmetic.

use bigdecimal::{BigDecimal, FromPrimitive};
use mxx_bench_estimator::{
    CostReport, estimate, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig,
};
use mxx_exponent_lut::noise_refresh::simulation::ExponentLutStorageReport;
use mxx_ir_core::{BenchmarkRole, Graph, ParamEnv};
use mxx_primitives::poly::dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids};
use mxx_runtime::backend::poly::gpu::gpu_backend_on;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct AcceptedReport {
    result: AcceptedResult,
}

#[derive(Deserialize)]
struct AcceptedResult {
    candidate: Candidate,
    achieved_security_bits: u64,
    sparse_lwr_universe: usize,
    sparse_lwr_weight: usize,
    sparse_lwr_q_l: usize,
    sparse_lwr_p: usize,
    layout_id: String,
    program_id: String,
    checker_accepted: bool,
    average_evidence: Option<AcceptedAverageEvidence>,
    production_report: Option<ProductionNoiseReport>,
}

#[derive(Deserialize)]
struct AcceptedAverageEvidence {
    setup_identity: String,
    switch_moduli: Vec<String>,
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

fn selected_average_config() -> Option<AverageCaseConfig> {
    use mxx_exponent_lut::noise_refresh::simulation::NoiseModelKind;
    (NoiseModelKind::from_env().unwrap() == NoiseModelKind::AverageCase).then(|| {
        AverageCaseConfig {
            failure_exponent: optional_usize("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", 100) as u32,
            allow_average_acceptance: true,
            ..Default::default()
        }
    })
}

/// Host-only numerical search: no ciphertext, artifact, or benchmark graph is built.
#[test]
fn test_rns_bgg_average_refresh_noise_simulation() {
    use mxx_exponent_lut::{
        layout::ExpLayout,
        noise_refresh::{
            pbc::{PbcActiveCellIndex, PbcLayoutSeed, PbcParameters, PbcPublicLayout},
            prf::{SparseLwrPrfProfile, SparseLwrPrfProgram},
        },
        noise_utils::{NonnegativeRational, RightActionNoiseGain},
        rns_bgg::{RnsBggParameters, RnsBggRefreshPlanRequest},
    };
    use std::collections::BTreeMap;
    let n = optional_usize("MXX_PRIMITIVE_TEST_RING_DIMENSION", 65536) as u32;
    let bits = optional_usize("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", 50);
    let max_depth = optional_usize("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH", 26);
    let universe = optional_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE", 8192);
    let weight = optional_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT", 18);
    let q_lwr = optional_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS", 4096);
    let encoding_error_sigma = bigdecimal::BigDecimal::from(4u8);
    let encoding_error_variance = NonnegativeRational::new(16u8.into(), 1u8.into()).unwrap();
    let average = AverageCaseConfig {
        failure_exponent: optional_usize("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", 100) as u32,
        allow_average_acceptance: true,
        ..Default::default()
    };
    let layout = PbcPublicLayout::build(
        &PbcParameters::paper_evaluation(universe, weight),
        PbcLayoutSeed(rand::random()),
        0,
    )
    .unwrap();
    let active = PbcActiveCellIndex::build(&layout).unwrap();
    let selector_counts = active.bucket_active_widths().collect::<Vec<_>>();
    let stages = selector_counts.len();
    let width_log2 = q_lwr.trailing_zeros() - 1;
    let program = SparseLwrPrfProgram::new(
        SparseLwrPrfProfile::new(
            ExpLayout { stride_log2: n.trailing_zeros() - width_log2, width_log2 },
            16,
        )
        .unwrap(),
        layout.bucket_width,
        stages,
    )
    .unwrap();
    let pool = DCRTPolyParams::try_new(n, max_depth, bits, (bits / 2) as u32, None, None).unwrap();
    let mut modulus = BigUint::one();
    let levels = pool
        .to_crt()
        .0
        .iter()
        .map(|prime| {
            modulus *= *prime;
            pool.select_modulus(&modulus).unwrap()
        })
        .collect::<Vec<_>>();
    // The minimum variance at each (step, remaining depth) dominates every
    // larger variance: the production transfer is affine and increasing.
    // Start at every allowed initial depth. Ignore intermediate acceptance
    // constraints, making rejection a necessary-condition result for ALL
    // nested prefix chains in this pool, not just a failed greedy allocation.
    let mut states = BTreeMap::new();
    for depth in stages + 3..=max_depth {
        if levels[depth - 1].modulus().bits() > 1400 {
            continue;
        }
        states.insert(depth, (encoding_error_variance.clone(), vec![depth]));
    }
    for (step, count) in selector_counts.iter().enumerate() {
        let mut next_states: BTreeMap<usize, (NonnegativeRational, Vec<usize>)> = BTreeMap::new();
        let gain = RightActionNoiseGain { l1: (*count).into(), l2_squared: (*count).into() };
        for (depth, (variance, path)) in states {
            for next in stages - step + 2..depth {
                let parameters = RnsBggParameters::new(
                    vec![levels[depth - 1].clone(), levels[next - 1].clone()],
                    encoding_error_sigma.clone(),
                    1,
                )
                .unwrap();
                let candidate = parameters
                    .private_mul_noise_variance(0, &variance, &RightActionNoiseGain::unit(), &gain)
                    .unwrap();
                let better = next_states.get(&next).is_none_or(|(current, _)| {
                    &candidate.numerator * &current.denominator <
                        &current.numerator * &candidate.denominator
                });
                if better {
                    let mut next_path = path.clone();
                    next_path.push(next);
                    next_states.insert(next, (candidate, next_path));
                }
            }
        }
        states = next_states;
    }
    let z = average.z_squared_for_events(&BigUint::one()).unwrap();
    let mut reports = Vec::new();
    for (depth, (variance, mut path)) in states {
        // Keep the remaining chain: continuation is a reusable state, not
        // merely the terminal two-column endpoint.
        path.extend((1..depth).rev());
        let parameters = RnsBggParameters::new(
            path.iter().map(|d| levels[d - 1].clone()).collect(),
            encoding_error_sigma.clone(),
            1,
        )
        .unwrap();
        let trace = rns_bgg::trace_prf(&parameters, &program.accumulation, &selector_counts);
        let output = &trace.wires[&program.accumulation.outputs()[0]];
        assert_eq!(output.moments.as_ref().unwrap().variance, variance);
        let optimistic_fits = output.moments.as_ref().unwrap().fits(
            &NonnegativeRational::new(levels[depth - 1].modulus().as_ref().clone(), 8u8.into())
                .unwrap(),
            &z,
        );
        // Refresh data with the same error envelope as a completed application
        // PRF, rather than assuming optimistically fresh data at this level.
        let data = output.clone();
        let observations = BigUint::from(2 * n as usize) *
            parameters.component_columns(stages + 1).unwrap() *
            parameters.levels()[stages + 1].crt_depth();
        let refresh = parameters.plan_refresh(
            &RnsBggRefreshPlanRequest {
                data_branches: n as usize,
                prf_branches: q_lwr / 2,
                base: 32,
                fresh_support: 32u8.into(),
                fresh_bound: 16u8.into(),
                decoder_eta: 2,
                observations,
                statistical_bits: average.failure_exponent as usize,
            },
            &data,
            output,
            Some(&average),
        );
        let log2 = |value: &BigUint| {
            let shift = value.bits().saturating_sub(53) as usize;
            (value >> shift).to_f64().unwrap().log2() + shift as f64
        };
        let joint_z = refresh.as_ref().ok().and_then(|plan| plan.z_squared.as_ref());
        let audit_z = joint_z.unwrap_or(&z);
        let first_failure = trace.gates.iter().find_map(|(gate, value)| {
            let moments = value.moments.as_ref().unwrap();
            let q = parameters.levels()[value.level].modulus();
            let scale = parameters.scale(value.level).unwrap();
            let modulus_fits = moments.fits(&NonnegativeRational::new(q.as_ref().clone(), 8u8.into()).unwrap(), audit_z);
            let scale_fits = moments.fits(&NonnegativeRational::new(scale.clone(), 2u8.into()).unwrap(), audit_z);
            (!modulus_fits || !scale_fits).then(|| serde_json::json!({
                "gate": format!("{gate:?}"), "output_level": value.level,
                "input_log2_q": log2(parameters.levels()[value.level - 1].modulus().as_ref()),
                "moddown_log2_divisor": log2(parameters.scale(value.level - 1).unwrap()),
                "output_log2_q": log2(q.as_ref()), "output_log2_scale": log2(scale),
                "unreduced_model_log2_stddev": (log2(&moments.variance.numerator) - log2(&moments.variance.denominator)) / 2.0,
                "optimistic_log2_tail_bound": log2(&moments.tail_bound(audit_z)),
                "modulus_fits": modulus_fits, "scale_fits": scale_fits,
            }))
        });
        let input_moments = trace.wires
            [program.accumulation.input_wires().values().next().unwrap()]
        .moments
        .as_ref()
        .unwrap();
        let input_fits = input_moments.fits(
            &NonnegativeRational::new(parameters.scale(0).unwrap().clone(), 2u8.into()).unwrap(),
            audit_z,
        );
        let correctness_accepted =
            refresh.is_ok() && joint_z.is_some() && first_failure.is_none() && input_fits;
        let continuation = parameters
            .lut_noise_bound(
                stages + 1,
                &mxx_exponent_lut::rns_bgg::RnsBggNoiseBound {
                    error: 16u8.into(),
                    payload_l1: 1u8.into(),
                },
                q_lwr / 2,
                &BigUint::one(),
                BigUint::one(),
            )
            .unwrap();
        let continuation_fits = parameters.noise_fits(stages + 2, &continuation).unwrap();
        reports.push(serde_json::json!({
            "correctness_accepted": correctness_accepted && continuation_fits,
            "continuation_fits": continuation_fits,
            "continuation_worst_error_bound": continuation.error.to_string(),
            "level_primes": parameters.levels().iter().map(|level| level.to_crt().0).collect::<Vec<_>>(),
            "initial_modulus": parameters.levels()[0].modulus().to_string(),
            "state_columns": (0..parameters.levels().len()).map(|level| 2 * parameters.component_columns(level).unwrap()).collect::<Vec<_>>(),
            "joint_prf_and_refresh_audit": joint_z.is_some(),
            "chain_depths": path, "initial_modulus_bits": parameters.levels()[0].modulus().bits(),
            "prf_output_log2_q": log2(levels[depth - 1].modulus().as_ref()),
            "unreduced_model_log2_stddev": (log2(&variance.numerator) - log2(&variance.denominator)) / 2.0,
            "first_state_failure": first_failure,
            "prf_variance_numerator_bits": variance.numerator.bits(),
            "prf_variance_denominator_bits": variance.denominator.bits(),
            "optimistic_single_event_prf_fits": optimistic_fits,
            "refresh": refresh.map_err(|e| e.to_string()),
        }));
    }
    let report = serde_json::json!({
        "n": n, "crt_bits": bits, "base_bits": bits / 2, "max_crt_depth": max_depth,
        "error_sigma": encoding_error_sigma.normalized().to_string(),
        "encoding_error_bound": mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&encoding_error_sigma).to_string(),
        "initial_log2_q_limit": 1400,
        "lwr_q": q_lwr, "lwr_p": 32, "lwr_dimension": universe, "lwr_weight": weight,
        "pbc_blocks": stages, "pbc_seed": layout.seed, "pbc_layout_id": layout.layout_id, "selector_counts": selector_counts,
        "noise_model": "average-case", "failure_exponent": average.failure_exponent,
        "security_verified": false, "candidates": reports,
    });
    println!("rns_bgg_noise_simulation={report}");
    if let Some(path) = env::var_os("MXX_EXPONENT_LUT_REFRESH_SEARCH_OUTPUT") {
        fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}

#[test]
fn test_gpu_exponent_lut_benchmark_estimate() {
    install_tracing();

    if use_rns_bgg() {
        let graph_started = std::time::Instant::now();
        let average = selected_average_config();
        let (parameters, residue_level, preprocessing, encrypt, online, noise_report) =
            if let Some(path) = std::env::var_os("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT") {
                rns_bgg::benchmark_refresh(std::path::Path::new(&path))
            } else {
                let fixture = rns_bgg::refresh(average.as_ref());
                (
                    fixture.parameters,
                    fixture.residue_level.unwrap(),
                    fixture.benchmark_preprocessing,
                    fixture.benchmark_encrypt,
                    fixture.benchmark_online,
                    fixture.noise_report,
                )
            };
        let (preprocess_scopes, preprocess_nodes) =
            graph_scope_and_node_counts(&preprocessing.source);
        let (encrypt_scopes, encrypt_nodes) = graph_scope_and_node_counts(&encrypt.source);
        let (online_scopes, online_nodes) = graph_scope_and_node_counts(&online.source);
        info!(
            elapsed_seconds = graph_started.elapsed().as_secs_f64(),
            preprocess_scopes,
            preprocess_nodes,
            encrypt_scopes,
            encrypt_nodes,
            online_scopes,
            online_nodes,
            "RNS-BGG benchmark graphs validated"
        );
        let devices = detected_gpu_device_ids();
        assert!(!devices.is_empty(), "benchmark estimate requires CUDA");
        let mut levels = parameters.levels().to_vec();
        levels.extend(parameters.refresh_residue_parameters(residue_level).unwrap());
        let backends = devices
            .into_iter()
            .map(|device| {
                let rings = levels.iter().map(|p| {
                    GpuDCRTPolyParams::new(p.ring_dimension(), p.to_crt().0, p.base_bits(), None)
                });
                (gpu_backend_on(rings, [device]), device)
            })
            .collect();
        let mut backend = GpuNodeMeasurementBackend::new(
            backends,
            MeasurementHarnessConfig {
                warm_up_iterations: optional_usize("MXX_EXPONENT_LUT_BENCH_WARMUPS", 1),
                measured_iterations: optional_usize("MXX_EXPONENT_LUT_BENCH_ITERATIONS", 2),
                ..MeasurementHarnessConfig::default()
            },
        );
        println!("rns_bgg_noise_simulation={noise_report}");
        // Always estimate the complete lifecycle in its production order.
        let stages =
            [("preprocess", &preprocessing), ("encrypt", &encrypt), ("online_eval", &online)];
        for ((stage, _), report) in stages.iter().zip(measure_stages(&stages, &mut backend)) {
            log_report(stage, &report);
        }
        return;
    }
    let (mut config, report) = simulate_refresh();
    let report: AcceptedReport = serde_json::from_value(report).expect("current simulation report");
    let accepted = &report.result;
    assert!(accepted.achieved_security_bits >= config.security_bits);
    assert!(accepted.checker_accepted, "selected noise model must accept the candidate");
    let evidence = accepted.average_evidence.as_ref();
    if let Some(evidence) = evidence {
        assert!(
            evidence.accepted && evidence.hard_authority_accepted && evidence.correctness_accepted
        );
    }
    config.sparse_lwr_universe = accepted.sparse_lwr_universe;
    config.sparse_lwr_weight = accepted.sparse_lwr_weight;
    config.sparse_lwr_modulus = accepted.sparse_lwr_q_l;
    config.sparse_lwr_output_modulus = accepted.sparse_lwr_p;
    info!(
        gadget_dropped_moduli = config.gadget_dropped_moduli,
        "preparing universally approximate gadget graph; only trapdoor preimages retain exact row dimensions"
    );
    let graph_started = Instant::now();
    let prepared = if let Some(evidence) = evidence {
        prepare_candidate_with_fixed_digits(
            &config,
            accepted.candidate,
            evidence.mask_digit_count,
            evidence.fresh_error_digit_count,
            &evidence.switch_moduli.iter().map(|p| p.parse().unwrap()).collect::<Vec<_>>(),
        )
    } else {
        assert_eq!(config.noise_model, NoiseSearchModel::WorstCase);
        prepare_candidate(&config, accepted.candidate)
    }
    .expect("build accepted simulation graphs");
    let mask_digits = prepared.mask_base_p_digit_count;
    let fresh_error_digits = prepared.fresh_error_base_p_digit_count;
    if evidence.is_none() {
        assert_eq!(
            prepared.production_report, accepted.production_report,
            "reconstructed worst-case noise evidence must match the selected candidate"
        );
    }
    info!(
        elapsed_seconds = graph_started.elapsed().as_secs_f64(),
        "accepted Exponent-LUT candidate graph preparation complete"
    );
    let bundle_started = Instant::now();
    let bundle = prepared.bundle.as_ref().expect("accepted candidate bundle");
    let preprocessing_graph = bundle.benchmark_preprocessing_graph();
    let encrypt_graph = bundle.benchmark_encrypt_graph();
    let online_graph = bundle.benchmark_online_graph();
    let (preprocessing_scope_count, preprocessing_node_count) =
        graph_scope_and_node_counts(&preprocessing_graph.graph);
    let (encrypt_scope_count, encrypt_node_count) =
        graph_scope_and_node_counts(&encrypt_graph.graph);
    let (online_scope_count, online_node_count) = graph_scope_and_node_counts(&online_graph.graph);
    info!(
        elapsed_seconds = bundle_started.elapsed().as_secs_f64(),
        preprocessing_scope_count,
        preprocessing_node_count,
        encrypt_scope_count,
        encrypt_node_count,
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
    if let Some(evidence) = evidence {
        assert_eq!(
            current_setup_identity, evidence.setup_identity,
            "reconstructed setup identity must match the accepted report"
        );
    }
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
    let encrypt_validation_started = Instant::now();
    info!(
        scope_count = encrypt_scope_count,
        node_count = encrypt_node_count,
        "starting Exponent-LUT encryption graph validation"
    );
    let encrypt = encrypt_graph
        .validate_with_manifests(&bindings, bundle.verification_manifests())
        .expect("validate encryption graph");
    let (validated_encrypt_scope_count, validated_encrypt_node_count) =
        graph_scope_and_node_counts(&encrypt.source);
    info!(
        elapsed_seconds = encrypt_validation_started.elapsed().as_secs_f64(),
        scope_count = validated_encrypt_scope_count,
        node_count = validated_encrypt_node_count,
        "Exponent-LUT encryption graph validation complete"
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
    // Measure the same independent production graphs that make up the runtime
    // lifecycle. The online graph consumes the private artifact emitted by
    // encryption; correctness-only oracle work is not counted as production.
    let stages = [("preprocess", &preprocessing), ("encrypt", &encrypt), ("online_eval", &online)];
    info!(
        report_origin = "current simulation",
        crt_depth = accepted.candidate.crt_depth,
        log_ring_dimension = accepted.candidate.log_ring_dimension,
        crt_bits = accepted.candidate.crt_bits,
        base_bits = accepted.candidate.base_bits,
        native_p = config.sparse_lwr_output_modulus,
        native_p_information_bits =
            native_information_bits_per_output(config.sparse_lwr_output_modulus),
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
        prf_fresh_label_count =
            prepared.prf_component_count * prepared.prf_coefficient_count * fresh_error_digits,
        accepted_layout_id = report.result.layout_id,
        current_layout_id,
        layout_id_matches = current_layout_id == report.result.layout_id,
        accepted_program_id = report.result.program_id,
        current_program_id,
        program_id_matches = current_program_id == report.result.program_id,
        accepted_setup_identity = ?evidence.map(|e| &e.setup_identity),
        current_setup_identity,

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
        None,
        None,
    );
    let (moduli, _, _) = parameters.to_crt();
    let device_ids = detected_gpu_device_ids();
    assert!(!device_ids.is_empty(), "Exponent-LUT benchmark estimate requires a CUDA GPU");
    let gpu_parameters =
        GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits(), None);
    let warm_up_iterations = optional_usize("MXX_EXPONENT_LUT_BENCH_WARMUPS", 1);
    let measured_iterations = optional_usize("MXX_EXPONENT_LUT_BENCH_ITERATIONS", 2);
    assert!(measured_iterations > 0, "benchmark iterations must be positive");
    let backends = device_ids
        .iter()
        .copied()
        .map(|device_id| {
            (
                gpu_backend_on(
                    bundle.ring_parameters(gpu_parameters.clone()).expect("refresh rings"),
                    [device_id],
                ),
                device_id,
            )
        })
        .collect();
    let mut backend = GpuNodeMeasurementBackend::new(
        backends,
        MeasurementHarnessConfig {
            warm_up_iterations,
            measured_iterations,
            ..MeasurementHarnessConfig::default()
        },
    );
    info!(
        gpu_count = device_ids.len(),
        ?device_ids,
        warm_up_iterations,
        measured_iterations,
        "Exponent-LUT benchmark estimator configuration"
    );

    let reports = measure_stages(&stages, &mut backend);
    log_preprocess_report(&reports[0]);
    log_report("encrypt", &reports[1]);
    log_report("online_eval", &reports[2]);
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
        stages.iter().map(|(_, graph)| *graph),
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
        &reports[2],
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
fn measure_stages(
    stages: &[(&str, &mxx_ir_core::ValidatedGraph)],
    backend: &mut GpuNodeMeasurementBackend,
) -> Vec<CostReport> {
    for (_, graph) in stages {
        estimate(graph, backend).expect("collect production graph node shapes");
    }
    backend.measure_collected().expect("measure collected GPU node shapes");
    stages
        .iter()
        .map(|(_, graph)| {
            let report = estimate(graph, backend).expect("estimate measured production graph");
            assert!(report.total_work_seconds > 0.0, "measured work must be positive");
            report
        })
        .collect()
}

// Small, deterministic helpers for the Section-7 parameter search.
//
// The search is intentionally kept as test-support source. It is a measurement
// harness, not a new Exponent-LUT parameter abstraction: production code continues
// to use the ordinary `BggRefreshParameters`, PBC, PRF program, and checker APIs.

use std::{fs, io::ErrorKind, ops::RangeInclusive, path::Path, process::Command};

pub mod sparse_lwr_parameter_support {
    // Read-only parameter and estimator support for the native mod-Q sparse-LWR PRF.
    //
    // This file is deliberately independent of the refresh search harness. It
    // contains the paper-anchored LWR candidate and glue for the pinned estimator.

    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::{
        collections::{BTreeMap, BTreeSet},
        path::Path,
        process::Command,
    };

    pub const LATTICE_ESTIMATOR_REPOSITORY: &str = "https://github.com/malb/lattice-estimator.git";
    pub const REVIEWED_ESTIMATOR_COMMIT: &str = "53da5982597709ba0fdf94ea37a84d822310fd84";
    pub const PBC_REPLICATION_FACTOR: usize = 3;
    pub const PBC_BUCKET_OFFSET: usize = 3;
    pub const EXPECTED_ATTACKS: &[&str] =
        &["arora-gb", "bkw", "usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid", "dual", "dual_hybrid"];

    /// Derives the only exponent layout compatible with native modulus `Q` and a
    /// concrete encoding ring dimension `N`. `Q` is not an independent LUT width:
    /// `Q = 2N / 2^m`, while the operation width is `log2(Q/2)`.
    pub fn native_exp_layout(
        q: usize,
        ring_dimension: usize,
    ) -> Result<mxx_exponent_lut::ExpLayout, String> {
        if q < 2 || !q.is_power_of_two() || ring_dimension == 0 || !ring_dimension.is_power_of_two()
        {
            return Err(
                "native Q and encoding ring dimension must be positive powers of two".to_owned()
            );
        }
        let twice_ring = ring_dimension
            .checked_mul(2)
            .ok_or_else(|| "encoding ring dimension overflows 2N".to_owned())?;
        if twice_ring < q || twice_ring % q != 0 {
            return Err("native Q must divide 2N and satisfy Q <= 2N".to_owned());
        }
        let stride = twice_ring / q;
        if !stride.is_power_of_two() {
            return Err("native exponent stride must be a power of two".to_owned());
        }
        let stride_log2 = stride.trailing_zeros();
        let width_log2 = (q / 2).trailing_zeros();
        let layout = mxx_exponent_lut::ExpLayout { stride_log2, width_log2 };
        layout.validate(ring_dimension).map_err(|error| error.to_string())?;
        if layout.effective_modulus(ring_dimension) != q {
            return Err("derived exponent layout does not have native modulus Q".to_owned());
        }
        Ok(layout)
    }

    /// Whether a tuple meets the explicitly requested computational target.
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub enum SparseLwrSecurityTier {
        Requested,
        Rejected,
    }

    /// One reviewed Phase-1 point.  The tuple is ordered as `(Q, P, nu, h)`;
    /// The estimator result is part of the declaration so a checkpoint cannot
    /// silently substitute a different model or security policy. Tier is assigned
    /// from the reviewed primary/fallback thresholds after evaluating this row.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct SparseLwrParameterTuple {
        pub q_l: usize,
        pub p: usize,
        pub nu: usize,
        pub h: usize,
        /// Exact finite minimum over the requested attack set, when supplied by
        /// an external reviewed estimator run or a validated checkpoint. Search
        /// configuration never fabricates this evidence.
        pub estimator_minimum_classical_bits: Option<f64>,
        /// Conservative floor corresponding to the external exact minimum.
        pub estimator_security_bits: Option<u64>,
    }

    /// Returns the security floor carried by a reviewed tuple, if the tuple has
    /// complete pinned estimator evidence.  A tuple with no evidence is an
    /// explicit custom point and returns `None`; a partially populated or
    /// inconsistent declaration is rejected before any estimator process is
    /// started.
    pub fn pinned_estimator_security_bits(
        tuple: &SparseLwrParameterTuple,
    ) -> Result<Option<u64>, String> {
        match (tuple.estimator_minimum_classical_bits, tuple.estimator_security_bits) {
            (None, None) => Ok(None),
            (Some(minimum), Some(floor))
                if minimum.is_finite() && minimum >= 0.0 && minimum.floor() as u64 == floor =>
            {
                Ok(Some(floor))
            }
            (Some(_), Some(_)) => Err("Phase-1 tuple estimator evidence is invalid".to_owned()),
            _ => Err("Phase-1 tuple estimator evidence is incomplete".to_owned()),
        }
    }

    impl PartialEq for SparseLwrParameterTuple {
        fn eq(&self, other: &Self) -> bool {
            self.q_l == other.q_l &&
                self.p == other.p &&
                self.nu == other.nu &&
                self.h == other.h &&
                self.estimator_minimum_classical_bits == other.estimator_minimum_classical_bits &&
                self.estimator_security_bits == other.estimator_security_bits
        }
    }

    impl SparseLwrParameterTuple {
        pub fn candidate(&self) -> Result<SparseLwrCandidate, String> {
            let minimum_ring = self
                .q_l
                .checked_div(2)
                .filter(|minimum| *minimum > 0)
                .ok_or_else(|| "native Q must be an even positive modulus".to_owned())?;
            let mut n_enc_candidates = Vec::new();
            for exponent in (0..=16).map(|index| minimum_ring.checked_shl(index as u32)) {
                let Some(n) = exponent else { continue };
                if n <= (1usize << 16) {
                    n_enc_candidates.push(n);
                }
            }
            SparseLwrCandidate::new(
                format!("q{}_p{}_nu{}_h{}", self.q_l, self.p, self.nu, self.h),
                self.nu,
                self.h,
                self.q_l,
                self.p,
                n_enc_candidates,
            )
        }
    }

    fn reviewed_tuple(
        q_l: usize,
        p: usize,
        nu: usize,
        h: usize,
        estimator_minimum_classical_bits: f64,
        estimator_security_bits: u64,
    ) -> SparseLwrParameterTuple {
        SparseLwrParameterTuple {
            q_l,
            p,
            nu,
            h,
            estimator_minimum_classical_bits: Some(estimator_minimum_classical_bits),
            estimator_security_bits: Some(estimator_security_bits),
        }
    }

    /// The finite, ordered primary Phase-1 grid reviewed for integration.  It is
    /// intentionally a tuple list, not a generated Cartesian product.
    pub fn reviewed_phase1_tuple_grid() -> Vec<SparseLwrParameterTuple> {
        vec![
            // The adjacent lower-ν point is retained as explicit minimality
            // evidence and must reject the 100-bit floor.
            reviewed_tuple(2048, 16, 16384, 17, 100.5636531151896, 100),
            reviewed_tuple(4096, 32, 16384, 17, 100.56464330713312, 100),
            reviewed_tuple(4096, 32, 8192, 18, 103.18615790798486, 103),
        ]
    }

    /// A concrete native sparse-LWR candidate. `q` is the native exponent modulus
    /// `Q`, and `p` is the complete balanced digit base `P=2p'`.
    /// `n_enc_candidates` belongs to the encoding layer; the estimator dimension
    /// is always `nu` and is independent of the encoding ring dimension.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct SparseLwrCandidate {
        pub candidate_id: String,
        pub nu: usize,
        pub h: usize,
        pub q: usize,
        pub p: usize,
        pub n_enc_candidates: Vec<usize>,
    }

    impl SparseLwrCandidate {
        pub fn new(
            candidate_id: impl Into<String>,
            nu: usize,
            h: usize,
            q: usize,
            p: usize,
            n_enc_candidates: Vec<usize>,
        ) -> Result<Self, String> {
            let candidate =
                Self { candidate_id: candidate_id.into(), nu, h, q, p, n_enc_candidates };
            candidate.derived()?;
            Ok(candidate)
        }

        pub fn derived(&self) -> Result<SparseLwrDerived, String> {
            if self.nu == 0 || self.h == 0 || self.h > self.nu {
                return Err("native sparse-LWR candidate requires 0 < h <= nu".to_owned());
            }
            if self.q == 0 || !self.q.is_power_of_two() {
                return Err("native Q must be a positive power of two".to_owned());
            }
            if self.p < 2 || !self.p.is_power_of_two() || self.p > self.q || self.q % self.p != 0 {
                return Err(
                    "native P must be a power of two, satisfy 2 <= P <= Q, and divide Q".to_owned()
                );
            }
            if self.n_enc_candidates.is_empty() {
                return Err("at least one encoding ring dimension is required".to_owned());
            }

            // P=2p' is the complete balanced digit base.  Requiring P to be an
            // even power of two makes p'=P/2 integral and keeps the native
            // exponent/error interval aligned with the native power-of-two modulus.
            let delta = self.q / self.p;
            let shift =
                i64::try_from(delta / 2).map_err(|_| "LWR error shift overflows i64".to_owned())?;
            let error_lower = -shift;
            let error_upper = i64::try_from(delta - 1)
                .map_err(|_| "LWR error upper bound overflows i64".to_owned())?
                .checked_sub(shift)
                .ok_or_else(|| "LWR error upper bound overflows i64".to_owned())?;
            let maximum_ring_dimension = 1usize << 16;
            if self
                .n_enc_candidates
                .iter()
                .any(|&n| native_exp_layout(self.q, n).is_err() || n > maximum_ring_dimension)
            {
                return Err(
                "every encoding ring dimension must derive a valid native layout with Q <= 2^16"
                    .to_owned(),
            );
            }

            let delta_squared = delta
                .checked_mul(delta)
                .ok_or_else(|| "native Q/P error variance overflows usize".to_owned())?;
            Ok(SparseLwrDerived {
                p_prime: self.p / 2,
                delta,
                error_lower,
                error_upper,
                error_stddev: ((delta_squared.saturating_sub(1) as f64) / 12.0).sqrt(),
                h_prime: self
                    .h
                    .checked_add(PBC_BUCKET_OFFSET)
                    .ok_or_else(|| "h' overflows usize".to_owned())?,
                total_slots: PBC_REPLICATION_FACTOR
                    .checked_mul(self.nu)
                    .ok_or_else(|| "PBC total slot count overflows usize".to_owned())?,
                average_bucket_size: PBC_REPLICATION_FACTOR as f64 * self.nu as f64 /
                    (self.h + PBC_BUCKET_OFFSET) as f64,
                raw_key_entropy_bits: log2_binomial(self.nu, self.h),
                gcd_condition: true,
            })
        }
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct SparseLwrDerived {
        /// Positive half-width of the balanced native output (`P = 2p'`).
        pub p_prime: usize,
        pub delta: usize,
        pub error_lower: i64,
        pub error_upper: i64,
        pub error_stddev: f64,
        pub h_prime: usize,
        pub total_slots: usize,
        pub average_bucket_size: f64,
        pub raw_key_entropy_bits: f64,
        pub gcd_condition: bool,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct EstimatorReport {
        pub schema_version: u32,
        pub repository_url: String,
        /// Commit of the imported lattice-estimator source checkout (not its wrapper).
        pub git_commit: String,
        pub sage_version: String,
        pub python_version: String,
        pub cost_model: String,
        pub shape_model: String,
        pub quantum: bool,
        pub sample_count: String,
        pub attacks: BTreeMap<String, Value>,
        pub failures: Vec<String>,
        pub infinite_attacks: Vec<String>,
        pub minimum_classical_bits: f64,
    }

    impl EstimatorReport {
        pub(crate) fn validate(&self) -> Result<(), String> {
            if self.schema_version != 1 {
                return Err("unsupported estimator report schema".to_owned());
            }
            if self.repository_url != LATTICE_ESTIMATOR_REPOSITORY {
                return Err("estimator repository is not the official malb repository".to_owned());
            }
            if self.git_commit.len() != 40 ||
                !self.git_commit.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return Err("estimator report does not contain a pinned Git commit".to_owned());
            }
            if self.sage_version.is_empty() || self.python_version.is_empty() {
                return Err("estimator report is missing runtime versions".to_owned());
            }
            if self.cost_model != "MATZOV" || self.shape_model != "gsa" {
                return Err("estimator report does not use RC.MATZOV with GSA".to_owned());
            }
            if self.quantum || self.sample_count != "infinity" {
                return Err(
                    "estimator report is not the requested classical m=infinity run".to_owned()
                );
            }
            if !self.failures.is_empty() {
                return Err("estimator failures are fail-closed".to_owned());
            }
            let infinite = self.infinite_attacks.iter().cloned().collect::<BTreeSet<_>>();
            if infinite.len() != self.infinite_attacks.len() ||
                infinite
                    .iter()
                    .any(|name| !EXPECTED_ATTACKS.iter().any(|attack| *attack == name.as_str()))
            {
                return Err("estimator infinite attack list is unknown or duplicated".to_owned());
            }
            for attack in EXPECTED_ATTACKS {
                let fields = self
                    .attacks
                    .get(*attack)
                    .ok_or_else(|| format!("full estimator result is missing attack {attack}"))?;
                let listed_infinite = infinite.contains(*attack);
                if listed_infinite && fields.get("rop_log2") != Some(&Value::Null) {
                    return Err(format!("infinite attack {attack} must have null rop_log2"));
                }
                let Some(bits) = fields.get("rop_log2").and_then(Value::as_f64) else {
                    if listed_infinite {
                        continue;
                    }
                    return Err(format!("attack {attack} has no finite rop_log2"));
                };
                if !bits.is_finite() {
                    return Err(format!("attack {attack} has non-finite rop_log2"));
                }
            }
            if !self.minimum_classical_bits.is_finite() {
                return Err("minimum classical security is non-finite".to_owned());
            }
            let minimum = EXPECTED_ATTACKS
                .iter()
                .filter_map(|attack| self.attacks[*attack].get("rop_log2").and_then(Value::as_f64))
                .fold(f64::INFINITY, f64::min);
            if !minimum.is_finite() {
                return Err("estimator has no finite attack cost".to_owned());
            }
            if (minimum - self.minimum_classical_bits).abs() > 1e-6 {
                return Err("minimum classical security does not match attack results".to_owned());
            }
            Ok(())
        }
    }

    pub fn parse_estimator_report(bytes: &[u8]) -> Result<EstimatorReport, String> {
        let report: EstimatorReport = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid sparse-LWR estimator JSON: {error}"))?;
        report.validate()?;
        Ok(report)
    }

    /// Invoke the Sage script and parse its machine-readable full estimate.
    pub fn run_estimator(
        candidate: &SparseLwrCandidate,
        script: &Path,
        estimator_root: &Path,
        sage_binary: &Path,
        expected_commit: Option<&str>,
    ) -> Result<EstimatorReport, String> {
        let output = Command::new(sage_binary)
            .args(["-python"])
            .arg(script)
            .args([
                "--nu",
                &candidate.nu.to_string(),
                "--h",
                &candidate.h.to_string(),
                "--q",
                &candidate.q.to_string(),
                "--p",
                &candidate.p.to_string(),
            ])
            .env("MXX_LATTICE_ESTIMATOR_ROOT", estimator_root)
            .env("DOT_SAGE", "/tmp/mxx-sage-cache")
            .output()
            .map_err(|error| format!("failed to start Sage estimator: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "Sage estimator failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let report = parse_estimator_report(&output.stdout)?;
        if let Some(expected) = expected_commit {
            if report.git_commit != expected {
                return Err(format!(
                    "estimator source commit mismatch: expected {expected}, got {}",
                    report.git_commit
                ));
            }
        }
        Ok(report)
    }

    fn log2_binomial(n: usize, k: usize) -> f64 {
        (0..k).map(|index| ((n - index) as f64).ln() - ((index + 1) as f64).ln()).sum::<f64>() /
            2.0_f64.ln()
    }
}

use self::sparse_lwr_parameter_support::{
    REVIEWED_ESTIMATOR_COMMIT, SparseLwrParameterTuple, SparseLwrSecurityTier, native_exp_layout,
    reviewed_phase1_tuple_grid,
};
#[allow(unused_imports)]
pub use self::sparse_lwr_parameter_support::{pinned_estimator_security_bits, run_estimator};
use mxx_bgg::BggSamplerLayout;
use mxx_exponent_lut::{
    bgg::{
        BggRefreshParameters, RefreshCompiler,
        noise::{BggNoiseSnapshot, ProgramNoiseReport, RefreshNoiseGainDecompositions},
    },
    noise_refresh::{
        pbc::{
            PbcActiveCellIndex, PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed,
            generate_key_layout,
        },
        prf::{SparseLwrPrfProfile, SparseLwrPrfProgram},
        setup::{RefreshParameterSimulationBundle, RefreshParameterSimulationRequest},
        simulation::{
            AverageCaseConfig, ExponentLutAverageNoiseReport, ExponentLutNoiseReport,
            PreRefreshNoiseEvaluation, fresh_error_below_every_prime_factor,
        },
    },
    noise_utils::{NonnegativeRational, RightActionNoiseGain},
};
use mxx_ir_core::{IntExpr, RealExpr};
use mxx_primitives::{
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::bounds::TrapdoorPreimageCutoffPolicy,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use rand::{SeedableRng, rngs::StdRng, seq::index::sample};
use serde::Serialize;
use tracing::info;

const REVIEWED_ESTIMATOR_COST_MODEL: &str = "MATZOV";
const REVIEWED_ESTIMATOR_SHAPE_MODEL: &str = "gsa";
pub const NO_CANDIDATE_ERROR: &str =
    "no candidate in the declared finite grid passed security and correctness";

/// Preparation can reject one candidate for exact arithmetic infeasibility,
/// while malformed setup/configuration remains a fatal search error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CandidatePreparationError {
    Infeasible(String),
    Fatal(String),
}

impl From<String> for CandidatePreparationError {
    fn from(error: String) -> Self {
        Self::Fatal(error)
    }
}

impl std::fmt::Display for CandidatePreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infeasible(error) | Self::Fatal(error) => formatter.write_str(error),
        }
    }
}

/// Validates the frozen refresh bundle and runs the application-specific
/// native Exponent-LUT noise authority.  Generic graph-noise conversion is
/// intentionally not part of the search boundary.
pub fn check_refresh_bundle(
    config: &SearchConfig,
    candidate: Candidate,
    prepared: &mut PreparedCandidate,
) -> Result<bool, String> {
    let accepted = evaluate_refresh_bundle(config, candidate, prepared)?;
    if !accepted || config.noise_model == NoiseSearchModel::AverageCase {
        return Ok(accepted);
    }
    if candidate != prepared.candidate {
        return Err("certificate candidate parameters differ from prepared parameters".into());
    }
    let bundle = prepared
        .bundle
        .as_ref()
        .ok_or("worst-case acceptance requires the selected candidate's full PRE/VER bundle")?;
    let parameters = DCRTPolyParams::new(
        prepared.ring_dimension as u32,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
        None,
        None,
    );
    let rings = bundle.ring_parameters(parameters).map_err(|error| error.to_string())?;
    let layouts =
        mxx_runtime::lean::export_dcrt_layouts(&rings).map_err(|error| error.to_string())?;
    let backend = mxx_runtime::lean::render_backend_context(&layouts, "Backend", "RefreshBackend")
        .map_err(|error| error.to_string())?;
    mxx_exponent_lut::lean::certify::certify(
        bundle,
        &mxx_ir_core::ParamEnv::default(),
        &backend,
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/exponent_refresh_certificates"),
        mxx_exponent_lut::lean::certify::check_timeout().map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("worst-case final certificate rejected: {error}"))?;
    Ok(true)
}

fn evaluate_refresh_bundle(
    config: &SearchConfig,
    _candidate: Candidate,
    prepared: &mut PreparedCandidate,
) -> Result<bool, String> {
    if config.noise_model == NoiseSearchModel::AverageCase {
        let average = if let Some(bundle) = prepared.bundle.as_ref() {
            bundle.simulate_average_noise(&average_case_config(config)?)
        } else if let Some(snapshot) = prepared.noise_snapshot.as_ref() {
            if let Some(pre_refresh) = prepared.pre_refresh.as_ref() {
                snapshot.simulate_average_with_pre_refresh(
                    &average_case_config(config)?,
                    pre_refresh.clone(),
                )
            } else {
                snapshot.simulate_average(&average_case_config(config)?)
            }
        } else {
            return Err("candidate has neither a simulation bundle nor a noise snapshot".to_owned());
        }
        .map_err(|error| format!("AverageCase simulation: {error}"))?;
        info!(
            steps = average.program.steps.len(),
            accepted = average.accepted,
            hard_authority_accepted = average.hard_authority_accepted,
            correctness_accepted = average.correctness_accepted,
            "native AverageCase noise report"
        );
        prepared.production_report = Some(ProductionNoiseReport {
            anchor_security_inputs: prepared.anchor_security_inputs.clone(),
            worst_case: None,
            average_case: Some(average.clone()),
            fresh_error_margins: fresh_error_margins(config, _candidate, prepared)?,
        });
        log_average_noise_report(_candidate, &average);
        prepared.average_report = Some(average.clone());
        return Ok(average.accepted);
    }
    let report = if let Some(bundle) = prepared.bundle.as_ref() {
        bundle.simulate_noise()
    } else if let Some(snapshot) = prepared.noise_snapshot.as_ref() {
        if let Some(pre_refresh) = prepared.pre_refresh.as_ref() {
            snapshot.simulate_worst_from_pre_refresh(pre_refresh.clone())
        } else {
            snapshot.simulate()
        }
    } else {
        return Err("candidate has neither a simulation bundle nor a noise snapshot".to_owned());
    }
    .map_err(|error| error.to_string())?;
    info!(
        steps = report.program.steps.len(),
        readout_bits = report.readout_bound.bits(),
        "native Exponent-LUT noise report"
    );
    let accepted = report.refresh.accepted;
    log_worst_case_noise_report(_candidate, &report);
    prepared.production_report = Some(ProductionNoiseReport {
        anchor_security_inputs: prepared.anchor_security_inputs.clone(),
        worst_case: Some(report),
        average_case: None,
        fresh_error_margins: fresh_error_margins(config, _candidate, prepared)?,
    });
    Ok(accepted)
}

fn log_worst_case_noise_report(
    candidate: Candidate,
    report: &ExponentLutNoiseReport<ProgramNoiseReport>,
) {
    for step in &report.program.steps {
        info!(
            crt_depth = candidate.crt_depth,
            step = step.step_index,
            inherited_bits = step.inherited_bits,
            fresh_bits = step.fresh_bits,
            output_bits = step.output_bits,
            bit_growth = step.output_bits.saturating_sub(step.inherited_bits),
            family_member_count = step.family_member_count,
            family_representative_rows = step.family_body.len(),
            one_hot = !step.family_body.is_empty(),
            "production worst-case gate noise growth"
        );
        for body_gate in &step.family_body {
            info!(
                step = step.step_index,
                body_gate = body_gate.body_gate_index,
                operation = ?body_gate.operation,
                inherited_bits = body_gate.inherited_bits,
                fresh_bits = body_gate.fresh_bits,
                output_bits = body_gate.output_bits,
                bit_growth = body_gate.bit_growth,
                "production worst-case family body noise growth"
            );
        }
        if step.family_member_count > 0 {
            info!(
                step = step.step_index,
                family_member_count = step.family_member_count,
                constituent_count = step.constituent_bounds.len(),
                output_bits = step.output_bits,
                "production worst-case family aggregate noise growth"
            );
        }
    }
    info!(readout_bits = report.readout_bound.bits(), "production worst-case readout noise bound");
    for slot in &report.refresh.slots {
        info!(
            slot = slot.slot,
            rounding_lhs_bits = slot.rounding_lhs.bits(),
            rounding_margin = %slot.rounding_margin,
            fresh_error_accepted = slot.fresh_error_accepted,
            "production worst-case refresh slot margins"
        );
    }
}

fn log_average_noise_report(
    candidate: Candidate,
    report: &ExponentLutAverageNoiseReport<ProgramNoiseReport>,
) {
    for step in &report.program.steps {
        info!(
            crt_depth = candidate.crt_depth,
            step = step.step_index,
            inherited_bits = step.inherited_bits,
            fresh_bits = step.fresh_bits,
            output_bits = step.output_bits,
            bit_growth = step.output_bits.saturating_sub(step.inherited_bits),
            family_member_count = step.family_member_count,
            family_representative_rows = step.family_body.len(),
            one_hot = !step.family_body.is_empty(),
            "production average-case gate noise growth"
        );
        for body_gate in &step.family_body {
            info!(
                step = step.step_index,
                body_gate = body_gate.body_gate_index,
                operation = ?body_gate.operation,
                inherited_bits = body_gate.inherited_bits,
                fresh_bits = body_gate.fresh_bits,
                output_bits = body_gate.output_bits,
                bit_growth = body_gate.bit_growth,
                "production average-case family body noise growth"
            );
        }
    }
    info!(
        readout_bits = report.program.output_state().bound.bits(),
        accepted = report.accepted,
        "production average-case readout noise bound"
    );
    for slot in &report.refresh.slots {
        info!(
            slot = slot.slot,
            rounding_lhs_bits = slot.rounding_lhs.bits(),
            rounding_margin = %slot.rounding_margin,
            rounding_deficit_bits = slot.rounding_deficit.bits(),
            fresh_error_accepted = slot.fresh_error_accepted,
            "production average-case refresh slot margins"
        );
    }
}

fn fresh_error_margins(
    config: &SearchConfig,
    candidate: Candidate,
    prepared: &PreparedCandidate,
) -> Result<Vec<String>, String> {
    let parameters = DCRTPolyParams::new(
        prepared.ring_dimension as u32,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
        None,
        None,
    );
    let bound = BigUint::from(config.sparse_lwr_output_modulus)
        .pow(prepared.fresh_error_base_p_digit_count as u32) -
        BigUint::one();
    parameters
        .to_crt()
        .0
        .into_iter()
        .map(|modulus| {
            let factors =
                mxx_exponent_lut::noise_refresh::simulation::prime_factors_of_crt_modulus(
                    &BigUint::from(modulus),
                )
                .map_err(|error| error.to_string())?;
            let smallest_factor = factors
                .into_iter()
                .next()
                .ok_or_else(|| "CRT modulus has no prime factors".to_owned())?;
            Ok::<String, String>(
                (BigInt::from(smallest_factor) - BigInt::from(bound.clone())).to_string(),
            )
        })
        .collect()
}

/// The reviewed finite profile grid used by the Exponent-LUT refresh search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Test-harness noise authority; production defaults to WorstCase.
    pub noise_model: NoiseSearchModel,
    /// Inclusive CRT-depth range. The outer search order is ascending depth.
    pub crt_depths: RangeInclusive<usize>,
    /// Inclusive log2(ring-dimension) range. The inner order is ascending.
    pub log_ring_dimensions: RangeInclusive<usize>,
    pub security_bits: u64,
    /// Statistical security target for the joint mask transcript.
    pub mask_statistical_security_bits: usize,
    pub crt_bits: usize,
    pub base_bits: u32,
    /// Ordered Phase-2 gadget parameter grid.  For each CRT-prime width the
    /// base widths are listed from largest to smallest, as required by the
    /// ell-column minimization policy.
    pub crt_base_bits_grid: Vec<(usize, u32)>,
    pub secret_dimension: usize,
    /// Ordinary gadget towers omitted everywhere; trapdoor preimages stay exact.
    pub gadget_dropped_moduli: usize,
    pub error_sigma: f64,
    pub decoder_sigma: f64,
    /// Selected decoder cutoff policy. It is resolved and compared against
    /// the default trapdoor-sampler cutoff at concrete candidate preparation time.
    pub decoder_preimage_bound: TrapdoorPreimageCutoffPolicy,
    pub sparse_lwr_universe: usize,
    pub sparse_lwr_weight: usize,
    /// Explicit ascending universe-size grid used by Phase 1.  The selected
    /// value is frozen before any DCRT candidate is prepared.
    pub sparse_lwr_universe_grid: Vec<usize>,
    pub sparse_lwr_modulus: usize,
    pub sparse_lwr_output_modulus: usize,
    /// Explicit ordered Phase-1 tuple grid.  This is the source of truth for
    /// sparse-LWR selection; the scalar fields above are the selected tuple
    /// projected into Phase 2.
    pub sparse_lwr_phase1_grid: Vec<SparseLwrParameterTuple>,
    pub phase1_estimator_commit: String,
    pub phase1_estimator_cost_model: String,
    pub phase1_estimator_shape_model: String,
    pub pbc_max_attempts: u32,
    pub one_nontrivial_refresh_round: bool,
    pub plaintext_modulus: usize,
    /// Optional reviewed BGG/RLWE monotone security envelope.  When absent,
    /// Phase 2 uses the external lattice estimator as before.
    pub bgg_security_envelope: Option<String>,
}

pub use mxx_exponent_lut::noise_refresh::simulation::NoiseModelKind as NoiseSearchModel;

impl SearchConfig {
    /// Returns the reviewed finite grid. Phase 1 uses the ordered native
    /// `(Q, P, nu, h)` tuple points selected for the integration search. Phase 2 is
    /// searched through the reviewed depth and ring-dimension ranges. Gadget
    /// bases are considered globally in descending order, with CRT width as a
    /// descending tie-break.
    pub fn reviewed() -> Self {
        Self {
            noise_model: NoiseSearchModel::WorstCase,
            crt_depths: 1..=62,
            log_ring_dimensions: 16..=16,
            security_bits: 100,
            mask_statistical_security_bits: 100,
            crt_bits: 50,
            base_bits: 25,
            crt_base_bits_grid: vec![(50, 25)],
            secret_dimension: 2,
            gadget_dropped_moduli: 0,
            error_sigma: 4.0,
            decoder_sigma: 4.578,
            decoder_preimage_bound: TrapdoorPreimageCutoffPolicy::Default,
            sparse_lwr_universe: 16384,
            sparse_lwr_weight: 17,
            sparse_lwr_universe_grid: vec![16384],
            sparse_lwr_modulus: 2048,
            sparse_lwr_output_modulus: 16,
            sparse_lwr_phase1_grid: reviewed_phase1_tuple_grid(),
            phase1_estimator_commit: REVIEWED_ESTIMATOR_COMMIT.to_owned(),
            phase1_estimator_cost_model: REVIEWED_ESTIMATOR_COST_MODEL.to_owned(),
            phase1_estimator_shape_model: REVIEWED_ESTIMATOR_SHAPE_MODEL.to_owned(),
            pbc_max_attempts: 128,
            one_nontrivial_refresh_round: true,
            plaintext_modulus: 2,
            bgg_security_envelope: None,
        }
    }

    /// Reads the optional test-only parameter overlay from the process
    /// environment.  The canonical reviewed profile remains the fallback for
    /// every unset variable; production code does not consume this harness
    /// configuration.
    pub fn from_env() -> Result<Self, String> {
        Self::from_lookup(|name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{name} must contain valid UTF-8"))
            }
        })
    }

    /// Applies an environment-like lookup without reading process-global
    /// state. Keeping the lookup injectable makes parser tests deterministic
    /// and race-free while preserving the exact environment schema used by
    /// the simulation: security, native sparse-LWR tuple, CRT/base
    /// bits, and MIN/MAX CRT-depth and ring-dimension bounds.
    /// Every schema name uses the `MXX_EXPONENT_LUT_REFRESH_` prefix.
    pub fn from_lookup<L>(lookup: L) -> Result<Self, String>
    where
        L: Fn(&str) -> Result<Option<String>, String>,
    {
        let mut config = Self::reviewed();
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS")? {
            config.security_bits =
                parse_lookup_u64("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", &value)?;
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL")? {
            config.noise_model =
                match value.as_str() {
                    "worst-case" => NoiseSearchModel::WorstCase,
                    "average-case" => NoiseSearchModel::AverageCase,
                    _ => return Err(
                        "MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL must be worst-case or average-case"
                            .to_owned(),
                    ),
                };
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_BGG_SECURITY_ENVELOPE")? {
            if value != "reviewed_monotone_envelope" {
                return Err(
                    "MXX_EXPONENT_LUT_REFRESH_BGG_SECURITY_ENVELOPE must be reviewed_monotone_envelope"
                        .to_owned(),
                );
            }
            config.bgg_security_envelope = Some(value);
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS")? {
            config.mask_statistical_security_bits = parse_lookup_usize(
                "MXX_EXPONENT_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS",
                &value,
            )?;
        }
        let explicit_nu = lookup("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE")?;
        let explicit_h = lookup("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT")?;
        let explicit_q = lookup("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS")?;
        let explicit_p = lookup("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS")?;
        match (explicit_nu, explicit_h, explicit_q, explicit_p) {
            (Some(nu), Some(h), Some(q), Some(p)) => {
                let nu = parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE", &nu)?;
                let h = parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT", &h)?;
                let q = parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS", &q)?;
                let p =
                    parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", &p)?;
                config.sparse_lwr_universe = nu;
                config.sparse_lwr_weight = h;
                config.sparse_lwr_universe_grid = vec![nu];
                config.sparse_lwr_modulus = q;
                config.sparse_lwr_output_modulus = p;
                // The native tuple is the sole Phase-1 source of truth for an
                // explicit profile. Reuse reviewed evidence when this exact
                // tuple is selected; an unknown tuple remains evidence-free
                // and therefore requires an external estimator/checkpoint.
                config.sparse_lwr_phase1_grid = vec![
                    reviewed_phase1_tuple_grid()
                        .into_iter()
                        .find(|tuple| {
                            tuple.q_l == q && tuple.p == p && tuple.nu == nu && tuple.h == h
                        })
                        .unwrap_or(SparseLwrParameterTuple {
                            q_l: q,
                            p,
                            nu,
                            h,
                            estimator_minimum_classical_bits: None,
                            estimator_security_bits: None,
                        }),
                ];
            }
            (None, None, None, None) => {}
            _ => {
                return Err(
                    "MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE, _WEIGHT, _MODULUS, and _OUTPUT_MODULUS must be supplied together"
                        .to_owned(),
                );
            }
        }
        let explicit_crt_bits = lookup("MXX_EXPONENT_LUT_REFRESH_CRT_BITS")?;
        let explicit_base_bits = lookup("MXX_EXPONENT_LUT_REFRESH_BASE_BITS")?;
        match (explicit_crt_bits, explicit_base_bits) {
            (Some(crt_bits), Some(base_bits)) => {
                config.crt_bits =
                    parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", &crt_bits)?;
                config.base_bits =
                    parse_lookup_u32("MXX_EXPONENT_LUT_REFRESH_BASE_BITS", &base_bits)?;
                // An explicit pair is an explicit singleton, not a request to
                // regenerate the historical 32/30/28 search grid.
                config.crt_base_bits_grid = vec![(config.crt_bits, config.base_bits)];
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                return Err(
                    "MXX_EXPONENT_LUT_REFRESH_CRT_BITS and MXX_EXPONENT_LUT_REFRESH_BASE_BITS must be supplied together"
                        .to_owned(),
                );
            }
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MIN_CRT_DEPTH")? {
            config.crt_depths =
                parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_MIN_CRT_DEPTH", &value)?..=
                    *config.crt_depths.end();
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH")? {
            config.crt_depths = *config.crt_depths.start()..=
                parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH", &value)?;
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MIN_LOG_RING_DIMENSION")? {
            config.log_ring_dimensions =
                parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_MIN_LOG_RING_DIMENSION", &value)?..=
                    *config.log_ring_dimensions.end();
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MAX_LOG_RING_DIMENSION")? {
            config.log_ring_dimensions = *config.log_ring_dimensions.start()..=
                parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_MAX_LOG_RING_DIMENSION", &value)?;
        }
        config.validate()
    }

    fn validate(&self) -> Result<Self, String> {
        let crt_start = *self.crt_depths.start();
        let crt_end = *self.crt_depths.end();
        if crt_start == 0 || crt_start > crt_end {
            return Err("CRT depth range must be positive and ascending".to_owned());
        }
        let log_ring_start = *self.log_ring_dimensions.start();
        let log_ring_end = *self.log_ring_dimensions.end();
        if log_ring_start > log_ring_end || log_ring_end >= usize::BITS as usize {
            return Err("log ring-dimension range must be ascending and shiftable".to_owned());
        }
        let ring_dimensions = [log_ring_start, log_ring_end]
            .into_iter()
            .map(|log_ring| {
                1usize
                    .checked_shl(log_ring as u32)
                    .ok_or_else(|| "log ring-dimension range is not representable".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if self.security_bits == 0 {
            return Err("security bits must be positive".to_owned());
        }
        if self.crt_base_bits_grid.is_empty() ||
            self.crt_base_bits_grid.iter().any(|(crt_bits, base_bits)| {
                ring_dimensions.iter().any(|&ring_dimension| {
                    u32::try_from(ring_dimension)
                        .ok()
                        .and_then(|ring_dimension| {
                            DCRTPolyParams::validate_capability(
                                ring_dimension,
                                crt_start,
                                *crt_bits,
                                *base_bits,
                            )
                            .ok()
                        })
                        .is_none()
                })
            }) ||
            self.crt_base_bits_grid.windows(2).any(|window| {
                window[0].1 < window[1].1 ||
                    (window[0].1 == window[1].1 && window[0].0 <= window[1].0)
            })
        {
            return Err(
                "CRT/base bit grid must be ordered and supported by the DCRT backend".to_owned()
            );
        }
        if self.mask_statistical_security_bits == 0 {
            return Err("mask statistical security must be positive".to_owned());
        }
        if self.sparse_lwr_universe_grid.is_empty() ||
            self.sparse_lwr_universe_grid.iter().any(|&universe| universe == 0) ||
            self.sparse_lwr_universe_grid.windows(2).any(|window| window[0] >= window[1])
        {
            return Err(
                "SPARSE_LWR_UNIVERSE_GRID must be positive and strictly ascending".to_owned()
            );
        }
        if self.sparse_lwr_weight == 0 ||
            self.sparse_lwr_universe_grid
                .iter()
                .any(|&universe| self.sparse_lwr_weight > universe)
        {
            return Err("SPARSE_LWR_WEIGHT must be positive and fit every grid universe".to_owned());
        }
        if self.sparse_lwr_phase1_grid.is_empty() {
            return Err("Phase-1 sparse-LWR tuple grid must not be empty".to_owned());
        }
        for tuple in &self.sparse_lwr_phase1_grid {
            tuple.candidate().map_err(|error| format!("invalid Phase-1 tuple: {error}"))?;
            pinned_estimator_security_bits(tuple)?;
        }
        if self.sparse_lwr_phase1_grid.windows(2).any(|window| {
            window[0].q_l > window[1].q_l ||
                (window[0].q_l == window[1].q_l && window[0].p > window[1].p) ||
                window[0] == window[1]
        }) {
            return Err("Phase-1 tuple grid must be in reviewed ascending order".to_owned());
        }
        if self.security_bits == 0 {
            return Err("security target must be positive".to_owned());
        }
        if self.phase1_estimator_commit.len() != 40 ||
            !self.phase1_estimator_commit.bytes().all(|byte| byte.is_ascii_hexdigit()) ||
            self.phase1_estimator_cost_model.is_empty() ||
            self.phase1_estimator_shape_model.is_empty()
        {
            return Err("Phase-1 estimator identity is incomplete".to_owned());
        }
        sparse_lwr_error_bounds(self.sparse_lwr_modulus, self.sparse_lwr_output_modulus)?;
        let minimum_ring_dimension = 1usize
            .checked_shl(log_ring_start as u32)
            .ok_or_else(|| "minimum log ring dimension shift overflows usize".to_owned())?;
        native_exp_layout(self.sparse_lwr_modulus, minimum_ring_dimension)
            .map_err(|error| format!("native exponent layout: {error}"))?;
        Ok(self.clone())
    }
}

fn parse_lookup_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|error| format!("{name} must be an unsigned integer: {error}"))
}

fn parse_lookup_u32(name: &str, value: &str) -> Result<u32, String> {
    value
        .trim()
        .parse::<u32>()
        .map_err(|error| format!("{name} must be an unsigned 32-bit integer: {error}"))
}

fn parse_lookup_u64(name: &str, value: &str) -> Result<u64, String> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("{name} must be an unsigned 64-bit integer: {error}"))
}

/// Returns the reviewed CRT/base search order: base widths descend globally,
/// with CRT widths descending as the tie-break. Only backend-supported pairs
/// are generated; the exact sampler requires `base_bits <= crt_bits / 2`.
pub fn descending_crt_base_bits_grid(max_crt_bits: usize) -> Vec<(usize, u32)> {
    let mut widths = [max_crt_bits, 32, 30, 28]
        .into_iter()
        .filter(|&crt_bits| crt_bits != 0 && crt_bits <= max_crt_bits)
        .collect::<Vec<_>>();
    widths.sort_unstable_by(|left, right| right.cmp(left));
    widths.dedup();
    let max_base_bits = widths.iter().copied().max().unwrap_or(0).saturating_sub(1);
    (1..=max_base_bits as u32)
        .rev()
        .flat_map(|base_bits| widths.iter().copied().map(move |crt_bits| (crt_bits, base_bits)))
        .filter(|&(crt_bits, base_bits)| {
            DCRTPolyParams::validate_capability(1, 1, crt_bits, base_bits).is_ok()
        })
        .collect()
}

/// One point in the finite search grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
pub struct Candidate {
    pub crt_depth: usize,
    pub log_ring_dimension: usize,
    pub crt_bits: usize,
    pub base_bits: u32,
}

/// The public, non-secret facts retained for one prepared candidate.
pub struct PreparedCandidate {
    pub anchor_security_inputs: Vec<mxx_exponent_lut::bgg::BggRefreshAnchorSecurityInputs>,
    pub candidate: Candidate,
    pub ring_dimension: usize,
    pub bucket_width: usize,
    /// The concrete cutoff resolved from a final setup clone carrying the
    /// Default policy. This is the value reported as the official bound.
    pub official_preimage_bound: BigInt,
    /// The concrete cutoff selected by SearchConfig. It may exceed the
    /// official value after the explicit >= comparison.
    pub selected_preimage_bound: BigInt,
    pub layout_id: [u8; 32],
    pub program_id: [u8; 32],
    /// One-based accepted PBC layout retry count (public diagnostics only).
    pub pbc_attempts_used: u32,
    pub mask_base_p_digit_count: usize,
    pub fresh_error_base_p_digit_count: usize,
    /// Exact structural factors of the deferred refresh-PRF batch, retained for
    /// benchmark diagnostics without expanding the family on the host.
    pub prf_component_count: usize,
    pub prf_coefficient_count: usize,
    pub prf_active_count: usize,
    pub prf_label_count: usize,
    pub prf_value_count: usize,
    /// Present for the production search; omitted by mocked ordering tests.
    pub bundle: Option<RefreshParameterSimulationBundle>,
    /// Compact production-equivalent noise authority used during ordinary
    /// search preparation. It contains no executable graph or label family.
    pub noise_snapshot: Option<BggNoiseSnapshot>,
    /// Cached program/readout evaluation shared by all `d_m` candidates.
    pub pre_refresh: Option<PreRefreshNoiseEvaluation<ProgramNoiseReport>>,
    /// The complete production report used to accept this candidate.  It is
    /// populated by `check_refresh_bundle`, never by a shape-only estimate.
    pub production_report: Option<ProductionNoiseReport>,
    pub average_report: Option<ExponentLutAverageNoiseReport<ProgramNoiseReport>>,
}

/// Complete setup-owned evidence for one candidate.  The nested program
/// report retains every gate/family member bit transition; the refresh report
/// retains the exact per-slot rounding and fresh-error margins.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct ProductionNoiseReport {
    /// Marginal inputs and explicitly unresolved joint-view security. Noise
    /// acceptance and existing high-q estimates do not discharge this status.
    pub anchor_security_inputs: Vec<mxx_exponent_lut::bgg::BggRefreshAnchorSecurityInputs>,
    pub worst_case: Option<ExponentLutNoiseReport<ProgramNoiseReport>>,
    pub average_case: Option<ExponentLutAverageNoiseReport<ProgramNoiseReport>>,
    /// Exact smallest-prime-factor minus `(P^d_e-1)` margin for every CRT slot.
    pub fresh_error_margins: Vec<String>,
}

/// Security values and application-specific noise result for the selected point.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SearchResult {
    pub candidate: Candidate,
    /// Minimum of the original high-q BGG and sparse-LWR marginal estimates;
    /// not the security of the switched construction or its joint view.
    pub achieved_security_bits: u64,
    pub switched_security_status: String,
    pub bgg_rlwe_security_bits: u64,
    pub sparse_lwr_security_bits: u64,
    pub raw_key_entropy_bits: f64,
    pub sparse_lwr_universe: usize,
    pub sparse_lwr_weight: usize,
    pub official_preimage_bound: String,
    pub ring_dimension: usize,
    pub bucket_width: usize,
    pub pbc_attempts_used: u32,
    pub layout_id: String,
    pub program_id: String,
    /// Acceptance by the selected application noise authority only.
    pub checker_accepted: bool,
    pub sparse_lwr_q_l: usize,
    pub sparse_lwr_p: usize,
    pub sparse_lwr_tier: SparseLwrSecurityTier,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
    pub average_evidence: Option<AverageAcceptanceEvidence>,
    pub production_report: Option<ProductionNoiseReport>,
    pub bgg_security_authority: String,
    pub bgg_security_evidence_source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AverageAcceptanceEvidence {
    pub setup_identity: String,
    pub layout_id: String,
    pub program_id: String,
    pub switch_moduli: Vec<String>,
    pub mask_digit_count: usize,
    pub fresh_error_digit_count: usize,
    pub event_budget_log2: u32,
    pub tail_correction_bits: u32,
    pub joint_event_count: String,
    pub z_squared: String,
    pub mask_smudging_max_favg: String,
    pub hard_authority_accepted: bool,
    pub correctness_accepted: bool,
    pub accepted: bool,
    pub security_authority: String,
    pub correctness_authority: String,
    pub heuristics: Vec<String>,
    pub slots: Vec<AverageSlotEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AverageSlotEvidence {
    pub slot: usize,
    pub favg: String,
    pub kappa: String,
    pub spacing: String,
    pub deterministic_mask_bound: String,
    pub rounding_lhs: String,
    pub rounding_margin: String,
    pub rounding_deficit: String,
    pub rounding_accepted: bool,
    pub fresh_error_below_prime_factors: bool,
    pub fresh_error_margin: String,
    pub variance: String,
    pub readout_mask_variance: String,
    pub readout_fresh_variance: String,
    pub decoder_variance: String,
    pub state_variance: String,
    pub mask_uniform_moments: mxx_exponent_lut::noise_utils::NoiseMoments,
    pub fresh_uniform_moments: mxx_exponent_lut::noise_utils::NoiseMoments,
}

/// One aggregate, non-secret row persisted for a Phase-1 universe that was
/// actually evaluated.  It records enough evidence to audit the first
/// qualifying grid point without exposing the sparse support or schedule.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct SparseLwrEvaluation {
    pub q_l: usize,
    pub p: usize,
    pub universe: usize,
    pub weight: usize,
    pub tier: SparseLwrSecurityTier,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
    pub error_lower: i64,
    pub error_upper: i64,
    pub sparse_lwr_security_bits: u64,
    pub estimator_security_bits: u64,
    pub raw_key_entropy_bits: f64,
    pub minimum_security_bits: u64,
    pub qualified: bool,
}

/// The Phase-1 result that is reused unchanged by every DCRT candidate.
///
/// Keeping the estimator outputs with the selected universe size makes the
/// minimality claim auditable: Phase 2 cannot accidentally rerun the sparse
/// estimator with a candidate-dependent value.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct SelectedSparseLwrProfile {
    pub tuple: SparseLwrParameterTuple,
    pub parameter_grid: Vec<SparseLwrParameterTuple>,
    pub q_l: usize,
    pub p: usize,
    pub universe: usize,
    pub weight: usize,
    pub tier: SparseLwrSecurityTier,
    pub error_lower: i64,
    pub error_upper: i64,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
    pub sparse_lwr_security_bits: u64,
    pub raw_key_entropy_bits: f64,
    pub evaluations: Vec<SparseLwrEvaluation>,
}

/// Environment variable naming an explicit Phase-1 checkpoint file.
///
/// A checkpoint is never discovered implicitly.  When this variable is
/// absent, the test performs a fresh Phase-1 search and keeps the result only
/// in memory.  When it names a missing file, the fresh result is written to
/// that exact path before Phase 2 starts.  An existing file is reused only
/// after its declaration has an exact match with the current search model.
pub const PHASE1_CHECKPOINT_ENV: &str = "MXX_EXPONENT_LUT_REFRESH_PHASE1_CHECKPOINT";

const PHASE1_CHECKPOINT_VERSION: u32 = 1;

/// The public declaration against which a persisted Phase-1 result is checked.
///
/// These fields describe both the input grid and the security model used to
/// score each row.  They intentionally contain no sparse support, schedule,
/// key material, or protocol/IR value.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct Phase1Declaration {
    pub parameter_grid: Vec<SparseLwrParameterTuple>,
    pub selected: SparseLwrParameterTuple,
    pub raw_key_entropy_bits: Vec<f64>,
    pub security_target_bits: u64,
    pub sparse_secret_model: String,
    pub sparse_error_model: String,
    pub error_intervals: Vec<(usize, usize, i64, i64)>,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
    pub exact_estimator: bool,
    pub crt_base_bits_grid: Vec<(usize, u32)>,
    pub phase2_search_order: String,
}

impl Phase1Declaration {
    /// Derive the complete public declaration from the current test config.
    pub fn from_config(
        config: &SearchConfig,
        selected: &SelectedSparseLwrProfile,
    ) -> Result<Self, String> {
        Ok(Self {
            parameter_grid: config.sparse_lwr_phase1_grid.clone(),
            selected: selected.tuple.clone(),
            raw_key_entropy_bits: config
                .sparse_lwr_phase1_grid
                .iter()
                .map(|tuple| raw_key_entropy_bits(tuple.nu, tuple.h))
                .collect(),
            security_target_bits: config.security_bits,
            sparse_secret_model: "SparseBinary".to_owned(),
            sparse_error_model: "Uniform".to_owned(),
            error_intervals: config
                .sparse_lwr_phase1_grid
                .iter()
                .map(|tuple| {
                    let (lower, upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p)?;
                    Ok((tuple.q_l, tuple.p, lower, upper))
                })
                .collect::<Result<Vec<_>, String>>()?,
            estimator_commit: config.phase1_estimator_commit.clone(),
            estimator_cost_model: config.phase1_estimator_cost_model.clone(),
            estimator_shape_model: config.phase1_estimator_shape_model.clone(),
            exact_estimator: true,
            crt_base_bits_grid: config.crt_base_bits_grid.clone(),
            phase2_search_order: "all_qualified_sparse_profiles; crt_depth_ascending_then_log_ring_dimension_ascending_then_base_bits_descending_then_crt_bits_descending; profile_tuple_q_l_p_nu_h_ascending".to_owned(),
        })
    }
}

/// Public data written immediately after Phase 1 and before any DCRT work.
///
/// The evaluated rows are duplicated in `selected.evaluations` because the
/// selected profile is also the value consumed by Phase 2.  Keeping both
/// named fields makes the checkpoint self-describing and lets validation
/// reject a partially or manually edited checkpoint without reconstructing a
/// security estimate.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct Phase1Checkpoint {
    pub version: u32,
    pub declaration: Phase1Declaration,
    pub evaluated: Vec<SparseLwrEvaluation>,
    pub selected: SelectedSparseLwrProfile,
    pub accepted_phase2: Option<AcceptedPhase2Profile>,
}

#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct AcceptedPhase2Profile {
    pub tuple: SparseLwrParameterTuple,
    pub candidate: Candidate,
}

impl Phase1Checkpoint {
    pub fn from_selection(
        config: &SearchConfig,
        selected: &SelectedSparseLwrProfile,
    ) -> Result<Self, String> {
        let declaration = Phase1Declaration::from_config(config, selected)?;
        let checkpoint = Self {
            version: PHASE1_CHECKPOINT_VERSION,
            declaration,
            evaluated: selected.evaluations.clone(),
            selected: selected.clone(),
            accepted_phase2: None,
        };
        checkpoint.validate(config)?;
        Ok(checkpoint)
    }

    /// Validate an on-disk checkpoint before allowing it to skip Phase 1.
    pub fn validate(&self, config: &SearchConfig) -> Result<(), String> {
        if self.version != PHASE1_CHECKPOINT_VERSION {
            return Err(format!("unsupported Phase-1 checkpoint version {}", self.version));
        }
        let expected = Phase1Declaration::from_config(config, &self.selected)?;
        if self.declaration != expected {
            return Err(
                "Phase-1 checkpoint declaration does not match the current search model".to_owned()
            );
        }
        if self.evaluated != self.selected.evaluations {
            return Err(
                "Phase-1 checkpoint evaluated rows do not match the selected profile".to_owned()
            );
        }
        if let Some(accepted) = &self.accepted_phase2 {
            let row = self
                .evaluated
                .iter()
                .find(|row| {
                    row.q_l == accepted.tuple.q_l &&
                        row.p == accepted.tuple.p &&
                        row.universe == accepted.tuple.nu &&
                        row.weight == accepted.tuple.h
                })
                .ok_or_else(|| "accepted Phase-2 profile is not an evaluated tuple".to_owned())?;
            if accepted.tuple !=
                self.selected
                    .parameter_grid
                    .iter()
                    .find(|tuple| **tuple == accepted.tuple)
                    .cloned()
                    .ok_or_else(|| "accepted Phase-2 tuple is not declared".to_owned())? ||
                !row.qualified ||
                row.tier != self.selected.tier
            {
                return Err(
                    "accepted Phase-2 profile is not a qualified selected-tier row".to_owned()
                );
            }
            if !candidates(config).any(|candidate| candidate == accepted.candidate) {
                return Err(
                    "accepted Phase-2 candidate is outside the declared search grid".to_owned()
                );
            }
        }
        if self.evaluated.len() != config.sparse_lwr_phase1_grid.len() {
            return Err("Phase-1 checkpoint must exhaust the declared tuple grid".to_owned());
        }
        for (index, row) in self.evaluated.iter().enumerate() {
            let tuple = &config.sparse_lwr_phase1_grid[index];
            let (error_lower, error_upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p)?;
            let expected_tier = if row.minimum_security_bits >= config.security_bits {
                SparseLwrSecurityTier::Requested
            } else {
                SparseLwrSecurityTier::Rejected
            };
            if row.q_l != tuple.q_l ||
                row.p != tuple.p ||
                row.universe != tuple.nu ||
                row.weight != tuple.h ||
                row.tier != expected_tier ||
                row.estimator_commit != config.phase1_estimator_commit ||
                row.estimator_cost_model != config.phase1_estimator_cost_model ||
                row.estimator_shape_model != config.phase1_estimator_shape_model ||
                (row.error_lower, row.error_upper) != (error_lower, error_upper) ||
                row.sparse_lwr_security_bits != row.estimator_security_bits ||
                (row.raw_key_entropy_bits - raw_key_entropy_bits(row.universe, row.weight)).abs() >
                    1e-9 ||
                row.qualified != (row.minimum_security_bits >= config.security_bits) ||
                row.minimum_security_bits != row.sparse_lwr_security_bits
            {
                return Err("Phase-1 checkpoint contains an inconsistent evaluated row".to_owned());
            }
        }
        let selected_index = self
            .evaluated
            .iter()
            .position(|row| {
                row.q_l == self.selected.q_l &&
                    row.p == self.selected.p &&
                    row.universe == self.selected.universe &&
                    row.weight == self.selected.weight
            })
            .ok_or_else(|| "Phase-1 checkpoint selected tuple is not evaluated".to_owned())?;
        let selected_row = &self.evaluated[selected_index];
        let earlier_primary = self.evaluated[..selected_index].iter().any(|row| row.qualified);
        if !selected_row.qualified ||
            earlier_primary ||
            self.selected.tuple != config.sparse_lwr_phase1_grid[selected_index] ||
            self.selected.tier != selected_row.tier ||
            self.selected.q_l != selected_row.q_l ||
            self.selected.p != selected_row.p ||
            self.selected.universe != selected_row.universe ||
            self.selected.weight != selected_row.weight ||
            (self.selected.error_lower, self.selected.error_upper) !=
                (selected_row.error_lower, selected_row.error_upper) ||
            self.selected.estimator_commit != selected_row.estimator_commit ||
            self.selected.estimator_cost_model != selected_row.estimator_cost_model ||
            self.selected.estimator_shape_model != selected_row.estimator_shape_model ||
            self.selected.sparse_lwr_security_bits != selected_row.sparse_lwr_security_bits ||
            self.selected
                .tuple
                .estimator_security_bits
                .is_some_and(|bits| bits != selected_row.estimator_security_bits) ||
            (self.selected.raw_key_entropy_bits - selected_row.raw_key_entropy_bits).abs() > 1e-9
        {
            return Err(
                "Phase-1 checkpoint selected profile does not match its final row".to_owned()
            );
        }
        Ok(())
    }
}

/// Opaque persistence authority. Construction rebuilds the exact selected profile
/// and completes its full worst-case certificate; mutable prepared snapshots are
/// never accepted as evidence at this boundary.
mod phase2_certification {
    use super::*;

    pub struct Evidence {
        config: SearchConfig,
        selected: SelectedSparseLwrProfile,
        candidate: Candidate,
    }

    impl Evidence {
        pub fn certify(
            config: &SearchConfig,
            selected: &SelectedSparseLwrProfile,
            candidate: Candidate,
        ) -> Result<Self, String> {
            if config.noise_model != NoiseSearchModel::WorstCase {
                return Err(
                    "worst-case certification evidence requires worst-case configuration".into()
                );
            }
            if (selected.tuple.nu, selected.tuple.h, selected.tuple.q_l, selected.tuple.p) !=
                (selected.universe, selected.weight, selected.q_l, selected.p)
            {
                return Err(
                    "selected profile tuple differs from certified profile parameters".into()
                );
            }
            let mut exact_config = config.clone();
            exact_config.sparse_lwr_universe = selected.universe;
            exact_config.sparse_lwr_weight = selected.weight;
            exact_config.sparse_lwr_modulus = selected.q_l;
            exact_config.sparse_lwr_output_modulus = selected.p;
            prepare_candidate(&exact_config, candidate).map_err(|error| error.to_string())?;
            Ok(Self { config: config.clone(), selected: selected.clone(), candidate })
        }

        pub fn matches(
            &self,
            config: &SearchConfig,
            selected: &SelectedSparseLwrProfile,
            candidate: Candidate,
        ) -> bool {
            self.config == *config && self.selected == *selected && self.candidate == candidate
        }
    }
}
pub use phase2_certification::Evidence as Phase2CertificationEvidence;

/// Persist the accepted Phase-2 profile while retaining and revalidating all
/// Phase-1 evidence. The replacement is written beside the checkpoint and
/// renamed into place only after complete validation.
pub fn persist_accepted_phase2_profile(
    path: &Path,
    config: &SearchConfig,
    selected: &SelectedSparseLwrProfile,
    candidate: Candidate,
    evidence: Option<&Phase2CertificationEvidence>,
) -> Result<(), String> {
    if config.noise_model == NoiseSearchModel::WorstCase &&
        !evidence.is_some_and(|evidence| evidence.matches(config, selected, candidate))
    {
        return Err("worst-case persistence requires certification evidence for the exact candidate/profile/configuration".into());
    }
    let bytes = fs::read(path).map_err(|error| format!("read Phase-1 checkpoint: {error}"))?;
    let mut checkpoint: Phase1Checkpoint = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse Phase-1 checkpoint: {error}"))?;
    checkpoint.validate(config)?;
    let row = selected
        .evaluations
        .iter()
        .find(|row| {
            row.q_l == selected.tuple.q_l &&
                row.p == selected.tuple.p &&
                row.universe == selected.tuple.nu &&
                row.weight == selected.tuple.h
        })
        .ok_or_else(|| "selected Phase-1 profile has no evaluated row".to_owned())?;
    if !row.qualified || row.tier != selected.tier {
        return Err("selected Phase-1 profile is not qualified for acceptance".to_owned());
    }
    checkpoint.accepted_phase2 =
        Some(AcceptedPhase2Profile { tuple: selected.tuple.clone(), candidate });
    checkpoint.validate(config)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&checkpoint).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write Phase-1 checkpoint: {error}"))?;
    fs::rename(&temporary, path).map_err(|error| format!("replace Phase-1 checkpoint: {error}"))
}

/// Read an explicitly named checkpoint, or perform and persist a fresh
/// Phase-1 search when that path does not exist.  A missing path is the only
/// condition that starts a fresh search when a checkpoint path was supplied;
/// malformed files and declaration mismatches fail closed.
pub fn load_or_search_phase1<Security>(
    config: &SearchConfig,
    checkpoint_path: Option<&Path>,
    sparse_security: Security,
) -> Result<SelectedSparseLwrProfile, String>
where
    Security: FnMut(usize, usize, usize, usize) -> Result<u64, String>,
{
    if let Some(path) = checkpoint_path {
        match fs::read(path) {
            Ok(bytes) => {
                let checkpoint: Phase1Checkpoint = serde_json::from_slice(&bytes)
                    .map_err(|error| format!("read Phase-1 checkpoint: {error}"))?;
                checkpoint.validate(config)?;
                info!(path = %path.display(), "reused Phase-1 checkpoint");
                return Ok(checkpoint.selected);
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!("read Phase-1 checkpoint {}: {error}", path.display()));
            }
        }
    }

    let selected = select_sparse_lwr_profile(config, sparse_security)?;
    if let Some(path) = checkpoint_path {
        let checkpoint = Phase1Checkpoint::from_selection(config, &selected)?;
        let bytes = serde_json::to_vec_pretty(&checkpoint)
            .map_err(|error| format!("encode Phase-1 checkpoint: {error}"))?;
        fs::write(path, bytes)
            .map_err(|error| format!("write Phase-1 checkpoint {}: {error}", path.display()))?;
        info!(path = %path.display(), "persisted Phase-1 checkpoint before Phase 2");
    }
    Ok(selected)
}

/// The complete public declaration of the Phase-1 search.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Phase1SearchReport {
    /// Complete public input and security-model declaration for these rows.
    pub declaration: Phase1Declaration,
    pub parameter_grid: Vec<SparseLwrParameterTuple>,
    pub universe_grid: Vec<usize>,
    pub support_weight: usize,
    pub q_l: usize,
    pub output_modulus: usize,
    pub security_target_bits: u64,
    pub evaluated: Vec<SparseLwrEvaluation>,
    pub selected_universe: usize,
    pub selected_q_l: usize,
    pub selected_p: usize,
    pub selected_weight: usize,
    pub selected_tier: SparseLwrSecurityTier,
    pub selected_raw_key_entropy_bits: f64,
    pub selected_error_lower: i64,
    pub selected_error_upper: i64,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
}

/// The complete public declaration of the Phase-2 search order and bounds.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Phase2SearchReport {
    pub crt_depth_min: usize,
    pub crt_depth_max: usize,
    pub log_ring_dimension_min: usize,
    pub log_ring_dimension_max: usize,
    pub order: &'static str,
    pub security_target_bits: u64,
    pub overall_target_policy: String,
    pub crt_base_bits_grid: Vec<(usize, u32)>,
    pub noise_model: NoiseSearchModel,
    /// Canonical encoding-error distribution shared with RNS-BGG runtime.
    pub error_sigma: String,
    pub encoding_error_bound: String,
}

/// Persisted report wrapper for the parameter-search test.
///
/// `result` is deliberately accompanied by the declared search domains and
/// all evaluated Phase-1 rows.  This prevents a reader from mistaking the
/// selected point for a global minimum when it is only minimal in the
/// explicitly declared finite grids.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RefreshParameterSearchReport {
    pub phase1: Phase1SearchReport,
    pub phase2: Phase2SearchReport,
    pub result: SearchResult,
    /// Independent Phase-2 outcome for every qualified Phase-1 profile.  A
    /// missing result records the canonical finite-grid exhaustion for that
    /// profile; fatal construction errors abort the search instead.
    pub profile_results: Vec<ProfileSearchOutcome>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProfileSearchOutcome {
    pub profile: SelectedSparseLwrProfile,
    pub result: Option<SearchResult>,
    pub error: Option<String>,
}

/// Builds the public report metadata without retaining any secret-dependent
/// layout, support, selector, or schedule information.
pub fn search_report(
    config: &SearchConfig,
    sparse_profile: &SelectedSparseLwrProfile,
    result: SearchResult,
) -> RefreshParameterSearchReport {
    RefreshParameterSearchReport {
        phase1: Phase1SearchReport {
            declaration: Phase1Declaration::from_config(config, sparse_profile)
                .expect("reviewed Phase-1 config must have a valid error model"),
            parameter_grid: config.sparse_lwr_phase1_grid.clone(),
            universe_grid: config
                .sparse_lwr_phase1_grid
                .iter()
                .map(|tuple| tuple.nu)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            support_weight: sparse_profile.weight,
            q_l: sparse_profile.q_l,
            output_modulus: sparse_profile.p,
            security_target_bits: config.security_bits,
            evaluated: sparse_profile.evaluations.clone(),
            selected_universe: sparse_profile.universe,
            selected_q_l: sparse_profile.q_l,
            selected_p: sparse_profile.p,
            selected_weight: sparse_profile.weight,
            selected_tier: sparse_profile.tier,
            selected_raw_key_entropy_bits: sparse_profile.raw_key_entropy_bits,
            selected_error_lower: sparse_profile.error_lower,
            selected_error_upper: sparse_profile.error_upper,
            estimator_commit: sparse_profile.estimator_commit.clone(),
            estimator_cost_model: sparse_profile.estimator_cost_model.clone(),
            estimator_shape_model: sparse_profile.estimator_shape_model.clone(),
        },
        phase2: Phase2SearchReport {
            crt_depth_min: *config.crt_depths.start(),
            crt_depth_max: *config.crt_depths.end(),
            log_ring_dimension_min: *config.log_ring_dimensions.start(),
            log_ring_dimension_max: *config.log_ring_dimensions.end(),
            order: "crt_depth_ascending_then_log_ring_dimension_ascending_then_base_bits_descending_then_crt_bits_descending; profile_tuple_q_l_p_nu_h_ascending",
            security_target_bits: config.security_bits,
            overall_target_policy: "requested_target".to_owned(),
            crt_base_bits_grid: config.crt_base_bits_grid.clone(),
            noise_model: config.noise_model,
            error_sigma: BigDecimal::from_f64(config.error_sigma)
                .expect("finite SearchConfig.error_sigma")
                .normalized()
                .to_string(),
            encoding_error_bound: mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(
                &BigDecimal::from_f64(config.error_sigma).expect("finite SearchConfig.error_sigma"),
            )
            .to_string(),
        },
        result,
        profile_results: Vec::new(),
    }
}

/// Builds a report retaining the independently evaluated result for every
/// qualifying Phase-1 profile.  The selected result remains duplicated in
/// `result` for stable consumers, while `profile_results` carries the global
/// minimality evidence.
pub fn search_report_with_profiles(
    config: &SearchConfig,
    selected: &SelectedSparseLwrProfile,
    result: SearchResult,
    profile_results: Vec<ProfileSearchOutcome>,
) -> RefreshParameterSearchReport {
    let mut report = search_report(config, selected, result);
    report.profile_results = profile_results;
    report
}

/// Selects a qualifying sparse-LWR tuple from the explicit ordered Phase-1
/// grid. The callback is invoked exactly once per row and the whole grid is
/// exhausted before applying the primary-100-over-fallback-100 policy.
pub fn select_sparse_lwr_profile<Security>(
    config: &SearchConfig,
    mut sparse_security: Security,
) -> Result<SelectedSparseLwrProfile, String>
where
    Security: FnMut(usize, usize, usize, usize) -> Result<u64, String>,
{
    if config.sparse_lwr_phase1_grid.is_empty() {
        return Err("sparse-LWR Phase-1 tuple grid must not be empty".to_owned());
    }
    for tuple in &config.sparse_lwr_phase1_grid {
        tuple.candidate().map_err(|error| format!("invalid Phase-1 tuple: {error}"))?;
    }
    if config.sparse_lwr_phase1_grid.windows(2).any(|window| {
        window[0].q_l > window[1].q_l ||
            (window[0].q_l == window[1].q_l && window[0].p > window[1].p) ||
            window[0] == window[1]
    }) {
        return Err("sparse-LWR Phase-1 tuple grid must be in reviewed ascending order".to_owned());
    }
    let mut evaluations = Vec::new();
    let mut primary_selection = None;
    for tuple in &config.sparse_lwr_phase1_grid {
        let universe = tuple.nu;
        let weight = tuple.h;
        let q_l = tuple.q_l;
        let p = tuple.p;
        let sparse_bits = sparse_security(universe, weight, q_l, p)?;
        if let Some(expected_floor) = pinned_estimator_security_bits(tuple)? {
            if sparse_bits != expected_floor {
                return Err(format!(
                    "estimator result for (Q_L={}, p={}, nu={}, h={}) was {} bits, expected declared floor {}",
                    tuple.q_l, tuple.p, tuple.nu, tuple.h, sparse_bits, expected_floor
                ));
            }
        }
        if sparse_bits == 0 {
            return Err(format!(
                "estimator result for (Q_L={}, p={}, nu={}, h={}) was zero bits",
                tuple.q_l, tuple.p, tuple.nu, tuple.h
            ));
        }
        let exact_minimum = tuple.estimator_minimum_classical_bits.unwrap_or(sparse_bits as f64);
        let tier = if exact_minimum >= config.security_bits as f64 {
            SparseLwrSecurityTier::Requested
        } else {
            SparseLwrSecurityTier::Rejected
        };
        let qualified = exact_minimum >= config.security_bits as f64;
        let (error_lower, error_upper) = sparse_lwr_error_bounds(q_l, p)?;
        evaluations.push(SparseLwrEvaluation {
            q_l,
            p,
            universe,
            weight,
            tier,
            estimator_commit: config.phase1_estimator_commit.clone(),
            estimator_cost_model: config.phase1_estimator_cost_model.clone(),
            estimator_shape_model: config.phase1_estimator_shape_model.clone(),
            error_lower,
            error_upper,
            sparse_lwr_security_bits: sparse_bits,
            estimator_security_bits: sparse_bits,
            raw_key_entropy_bits: raw_key_entropy_bits(universe, weight),
            minimum_security_bits: sparse_bits,
            qualified,
        });
        info!(
            sparse_lwr_q_l = q_l,
            sparse_lwr_p = p,
            sparse_lwr_universe = universe,
            sparse_lwr_weight = weight,
            sparse_lwr_security_bits = sparse_bits,
            tier = ?tier,
            "evaluated sparse-LWR Phase-1 profile"
        );
        if qualified && primary_selection.is_none() {
            primary_selection = Some((tuple.clone(), sparse_bits));
        }
    }
    let (tuple, sparse_bits) = primary_selection.ok_or_else(|| {
        format!(
            "no sparse-LWR profile in the declared tuple grid passed the requested {}-bit floor",
            config.security_bits
        )
    })?;
    let (error_lower, error_upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p)?;
    Ok(SelectedSparseLwrProfile {
        tuple: tuple.clone(),
        parameter_grid: config.sparse_lwr_phase1_grid.clone(),
        q_l: tuple.q_l,
        p: tuple.p,
        universe: tuple.nu,
        weight: tuple.h,
        tier: SparseLwrSecurityTier::Requested,
        error_lower,
        error_upper,
        estimator_commit: config.phase1_estimator_commit.clone(),
        estimator_cost_model: config.phase1_estimator_cost_model.clone(),
        estimator_shape_model: config.phase1_estimator_shape_model.clone(),
        sparse_lwr_security_bits: sparse_bits,
        raw_key_entropy_bits: raw_key_entropy_bits(tuple.nu, tuple.h),
        evaluations,
    })
}

/// Returns every qualifying Phase-1 profile in deterministic tuple order.
/// The estimator evidence has already been collected by
/// `select_sparse_lwr_profile`; this projection never invokes it again.
pub fn qualified_sparse_lwr_profiles(
    selected: &SelectedSparseLwrProfile,
) -> Vec<SelectedSparseLwrProfile> {
    selected
        .parameter_grid
        .iter()
        .filter_map(|tuple| {
            let row = selected.evaluations.iter().find(|row| {
                row.q_l == tuple.q_l &&
                    row.p == tuple.p &&
                    row.universe == tuple.nu &&
                    row.weight == tuple.h
            })?;
            if !row.qualified || row.tier != selected.tier {
                return None;
            }
            Some(SelectedSparseLwrProfile {
                tuple: tuple.clone(),
                parameter_grid: selected.parameter_grid.clone(),
                q_l: tuple.q_l,
                p: tuple.p,
                universe: tuple.nu,
                weight: tuple.h,
                tier: row.tier,
                error_lower: row.error_lower,
                error_upper: row.error_upper,
                estimator_commit: row.estimator_commit.clone(),
                estimator_cost_model: row.estimator_cost_model.clone(),
                estimator_shape_model: row.estimator_shape_model.clone(),
                sparse_lwr_security_bits: row.sparse_lwr_security_bits,
                raw_key_entropy_bits: row.raw_key_entropy_bits,
                evaluations: selected.evaluations.clone(),
            })
        })
        .collect()
}

/// Evaluate every qualified Phase-1 profile. Only the canonical finite-grid
/// exhaustion result advances to the next profile; setup, estimator, and
/// checker failures remain fatal. Keeping all outcomes is required for a
/// global CRT-depth minimum rather than first-success selection.
pub fn search_all_qualified_profiles<Run>(
    selected: &SelectedSparseLwrProfile,
    mut run: Run,
) -> Result<Vec<ProfileSearchOutcome>, String>
where
    Run: FnMut(&SelectedSparseLwrProfile) -> Result<SearchResult, String>,
{
    let mut outcomes = Vec::new();
    for profile in qualified_sparse_lwr_profiles(selected) {
        match run(&profile) {
            Ok(result) => {
                outcomes.push(ProfileSearchOutcome { profile, result: Some(result), error: None })
            }
            Err(error) if error == NO_CANDIDATE_ERROR => {
                outcomes.push(ProfileSearchOutcome { profile, result: None, error: Some(error) });
            }
            Err(error) => return Err(error),
        }
    }
    if outcomes.iter().all(|outcome| outcome.result.is_none()) {
        return Err("no qualified sparse-LWR profile produced a valid Phase-2 candidate".to_owned());
    }
    Ok(outcomes)
}

/// Advance through all qualified Phase-1 profiles and select the global
/// minimum. The documented tie-break is depth, ring dimension, descending
/// base width, descending CRT width, then the reviewed tuple order.
pub fn search_qualified_profiles<Run>(
    selected: &SelectedSparseLwrProfile,
    run: Run,
) -> Result<(SelectedSparseLwrProfile, SearchResult), String>
where
    Run: FnMut(&SelectedSparseLwrProfile) -> Result<SearchResult, String>,
{
    let outcomes = search_all_qualified_profiles(selected, run)?;
    select_best_profile_result(outcomes)
}

/// Selects the best successful outcome without rerunning any production
/// report. This is kept separate so callers can persist all profile outcomes
/// and the selected result from one evaluation pass.
pub fn select_best_profile_result(
    outcomes: Vec<ProfileSearchOutcome>,
) -> Result<(SelectedSparseLwrProfile, SearchResult), String> {
    outcomes
        .into_iter()
        .filter_map(|outcome| outcome.result.map(|result| (outcome.profile, result)))
        .min_by(|(left_profile, left), (right_profile, right)| {
            let left_key = (
                left.candidate.crt_depth,
                left.candidate.log_ring_dimension,
                std::cmp::Reverse(left.candidate.base_bits),
                std::cmp::Reverse(left.candidate.crt_bits),
                left_profile.q_l,
                left_profile.p,
                left_profile.universe,
                left_profile.weight,
            );
            let right_key = (
                right.candidate.crt_depth,
                right.candidate.log_ring_dimension,
                std::cmp::Reverse(right.candidate.base_bits),
                std::cmp::Reverse(right.candidate.crt_bits),
                right_profile.q_l,
                right_profile.p,
                right_profile.universe,
                right_profile.weight,
            );
            left_key.cmp(&right_key)
        })
        .ok_or_else(|| {
            "no qualified sparse-LWR profile produced a valid Phase-2 candidate".to_owned()
        })
}

fn exact_reconstruction_coefficients(dcrt: &DCRTPolyParams) -> Vec<IntExpr> {
    dcrt.reconst_coeffs()
        .into_iter()
        .map(|value| BigInt::from_biguint(num_bigint::Sign::Plus, value).into())
        .collect()
}

/// Enumerates the grid in the exact order used for minimality claims: CRT
/// depth first, then ring dimension, then the reviewed descending base/CRT
/// tie-break. There is no hidden total-modulus-bit cap here: backend
/// feasibility is established when the concrete candidate is prepared.
pub fn candidates(config: &SearchConfig) -> impl Iterator<Item = Candidate> + '_ {
    config.crt_depths.clone().flat_map(move |crt_depth| {
        config.log_ring_dimensions.clone().flat_map(move |log_ring_dimension| {
            config.crt_base_bits_grid.iter().copied().map(move |(crt_bits, base_bits)| Candidate {
                crt_depth,
                log_ring_dimension,
                crt_bits,
                base_bits,
            })
        })
    })
}

fn final_average_evidence(
    config: &SearchConfig,
    prepared: &PreparedCandidate,
    report: &ExponentLutAverageNoiseReport<ProgramNoiseReport>,
) -> Result<Option<AverageAcceptanceEvidence>, String> {
    if config.noise_model != NoiseSearchModel::AverageCase {
        return Ok(None);
    }
    let policy = average_case_config(config)?;
    let refresh = &report.refresh;
    let fresh_margins = fresh_error_margins(config, prepared.candidate, prepared)?;
    let topology = refresh.joint_event_count.clone();
    let event_budget_log2 = {
        let bits = topology.bits();
        if topology == (BigUint::one() << (bits - 1)) {
            u32::try_from(bits - 1)
                .map_err(|_| "AverageCase event budget is too large".to_owned())?
        } else {
            u32::try_from(bits).map_err(|_| "AverageCase event budget is too large".to_owned())?
        }
    };
    let z_squared = &refresh.z_squared;
    Ok(Some(AverageAcceptanceEvidence {
        setup_identity: hex(report.snapshot_identity),
        layout_id: hex(prepared.layout_id),
        program_id: hex(prepared.program_id),
        switch_moduli: prepared
            .noise_snapshot
            .as_ref()
            .ok_or("missing selected switching rings")?
            .refresh
            .switch_moduli
            .iter()
            .map(ToString::to_string)
            .collect(),
        mask_digit_count: prepared.mask_base_p_digit_count,
        fresh_error_digit_count: prepared.fresh_error_base_p_digit_count,
        event_budget_log2,
        tail_correction_bits: policy.tail_correction_bits,
        joint_event_count: topology.to_string(),
        z_squared: format!("{}/{}", z_squared.numerator, z_squared.denominator),
        mask_smudging_max_favg: refresh.max_f_avg.to_string(),
        hard_authority_accepted: report.hard_authority_accepted,
        correctness_accepted: report.correctness_accepted,
        accepted: report.accepted,
        security_authority: format!("{:?}", report.security_authority),
        correctness_authority: format!("{:?}", report.correctness_authority),
        heuristics: Vec::new(),
        slots: refresh
            .slots
            .iter()
            .map(|slot| {
                Ok::<AverageSlotEvidence, String>(AverageSlotEvidence {
                    slot: slot.slot,
                    favg: slot.f_avg.to_string(),
                    kappa: slot.kappa.to_string(),
                    spacing: slot.spacing.to_string(),
                    deterministic_mask_bound: slot.deterministic_mask_bound.to_string(),
                    rounding_lhs: slot.rounding_lhs.to_string(),
                    rounding_margin: slot.rounding_margin.to_string(),
                    rounding_deficit: slot.rounding_deficit.to_string(),
                    rounding_accepted: slot.rounding_margin.sign() == num_bigint::Sign::Plus,
                    fresh_error_below_prime_factors: slot.fresh_error_accepted,
                    fresh_error_margin: fresh_margins
                        .get(slot.slot)
                        .cloned()
                        .ok_or_else(|| "average report slot lacks fresh-error margin".to_owned())?,
                    variance: format!("{}/{}", slot.variance.numerator, slot.variance.denominator),
                    readout_mask_variance: rational_string(&slot.variance_breakdown.readout_mask),
                    readout_fresh_variance: rational_string(&slot.variance_breakdown.readout_fresh),
                    decoder_variance: rational_string(&slot.variance_breakdown.decoder),
                    state_variance: rational_string(&slot.variance_breakdown.state),
                    mask_uniform_moments: slot.variance_breakdown.mask_uniform_moments.clone(),
                    fresh_uniform_moments: slot.variance_breakdown.fresh_uniform_moments.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    }))
}

fn rational_string(value: &NonnegativeRational) -> String {
    format!("{}/{}", value.numerator, value.denominator)
}

/// Runs a finite search with injectable security and checker functions.
///
/// The hooks make the ordering and minimality tests independent of Sage and
/// the full operational-noise graph.  The production integration test supplies
/// the lattice-estimator wrapper and a checker hook after building the actual
/// PBC/PRF graph metadata for the candidate.
pub fn search_with_hooks<Prepare, Security, Checker>(
    config: &SearchConfig,
    sparse_profile: &SelectedSparseLwrProfile,
    mut prepare: Prepare,
    mut security: Security,
    mut checker: Checker,
) -> Result<SearchResult, String>
where
    Prepare: FnMut(Candidate) -> Result<PreparedCandidate, CandidatePreparationError>,
    Security: FnMut(Candidate) -> Result<u64, String>,
    Checker: FnMut(&mut PreparedCandidate) -> Result<bool, String>,
{
    if sparse_profile.q_l != config.sparse_lwr_modulus ||
        sparse_profile.p != config.sparse_lwr_output_modulus ||
        sparse_profile.universe != config.sparse_lwr_universe ||
        sparse_profile.weight != config.sparse_lwr_weight
    {
        return Err(
            "selected sparse-LWR profile does not match the frozen search configuration".to_owned()
        );
    }
    let overall_security_target = config.security_bits;
    for candidate in candidates(config) {
        info!(
            crt_depth = candidate.crt_depth,
            log_ring_dimension = candidate.log_ring_dimension,
            crt_bits = candidate.crt_bits,
            base_bits = candidate.base_bits,
            "evaluating Exponent-LUT refresh search candidate"
        );
        if let Err(error) = validate_backend_candidate(candidate) {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                crt_bits = candidate.crt_bits,
                base_bits = candidate.base_bits,
                reason = %error,
                "candidate rejected by backend CRT/base capability"
            );
            continue;
        }
        let bgg = security(candidate)?;
        let achieved = bgg.min(sparse_profile.sparse_lwr_security_bits);
        if achieved < overall_security_target {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                crt_bits = candidate.crt_bits,
                base_bits = candidate.base_bits,
                achieved_security_bits = achieved,
                "candidate rejected by security floor"
            );
            continue;
        }
        let prepared = match prepare(candidate) {
            Ok(prepared) => prepared,
            Err(CandidatePreparationError::Infeasible(error)) => {
                info!(
                    crt_depth = candidate.crt_depth,
                    log_ring_dimension = candidate.log_ring_dimension,
                    crt_bits = candidate.crt_bits,
                    base_bits = candidate.base_bits,
                    reason = %error,
                    "candidate rejected by exact preparation bounds"
                );
                continue;
            }
            Err(CandidatePreparationError::Fatal(error)) => return Err(error),
        };
        let mut prepared = prepared;
        if !checker(&mut prepared)? {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                crt_bits = candidate.crt_bits,
                base_bits = candidate.base_bits,
                "candidate rejected by Exponent-LUT noise threshold"
            );
            continue;
        }
        return Ok(SearchResult {
            candidate,
            achieved_security_bits: achieved,
            bgg_rlwe_security_bits: bgg,
            sparse_lwr_security_bits: sparse_profile.sparse_lwr_security_bits,
            raw_key_entropy_bits: sparse_profile.raw_key_entropy_bits,
            sparse_lwr_universe: sparse_profile.universe,
            sparse_lwr_weight: sparse_profile.weight,
            official_preimage_bound: prepared.official_preimage_bound.to_string(),
            ring_dimension: prepared.ring_dimension,
            bucket_width: prepared.bucket_width,
            pbc_attempts_used: prepared.pbc_attempts_used,
            layout_id: hex(prepared.layout_id),
            program_id: hex(prepared.program_id),
            checker_accepted: true,
            switched_security_status: "low-P marginal estimates not supplied; cross-modulus joint-view security unresolved".into(),
            sparse_lwr_q_l: sparse_profile.q_l,
            sparse_lwr_p: sparse_profile.p,
            sparse_lwr_tier: sparse_profile.tier,
            estimator_commit: sparse_profile.estimator_commit.clone(),
            estimator_cost_model: sparse_profile.estimator_cost_model.clone(),
            estimator_shape_model: sparse_profile.estimator_shape_model.clone(),
            average_evidence: prepared
                .average_report
                .as_ref()
                .map(|report| final_average_evidence(config, &prepared, report))
                .transpose()?
                .flatten(),
            production_report: prepared.production_report.clone(),
            bgg_security_authority: bgg_security_authority(config).to_owned(),
            bgg_security_evidence_source: bgg_security_evidence_source(config).to_owned(),
        });
    }
    Err(NO_CANDIDATE_ERROR.to_owned())
}

const REVIEWED_BGG_ENVELOPE: &str = "reviewed_monotone_envelope";
const REVIEWED_BGG_ENVELOPE_SOURCE: &str = "reviewed_monotone_envelope:q27:136";

fn bgg_security_authority(config: &SearchConfig) -> &'static str {
    if config.bgg_security_envelope.as_deref() == Some(REVIEWED_BGG_ENVELOPE) {
        REVIEWED_BGG_ENVELOPE
    } else {
        "external_lattice_estimator"
    }
}

fn bgg_security_evidence_source(config: &SearchConfig) -> &'static str {
    if config.bgg_security_envelope.as_deref() == Some(REVIEWED_BGG_ENVELOPE) {
        REVIEWED_BGG_ENVELOPE_SOURCE
    } else {
        "external_lattice_estimator"
    }
}

/// Returns the reviewed monotone BGG/RLWE envelope floor.  This is deliberately
/// narrower than the external estimator path: the envelope is valid only for
/// the exact reviewed N/CRT/base/sigma/distribution domain and up to q_27.
// This helper is imported by the parameter-search integration tests; Cargo also
// may include this helper without using the reviewed parameter envelope.
#[allow(dead_code)]
pub(crate) fn reviewed_bgg_security_bits(
    config: &SearchConfig,
    candidate: Candidate,
) -> Result<u64, String> {
    if config.bgg_security_envelope.as_deref() != Some(REVIEWED_BGG_ENVELOPE) {
        return Err("reviewed BGG envelope is not selected".to_owned());
    }
    if candidate.log_ring_dimension != 16 ||
        candidate.crt_bits != 50 ||
        candidate.base_bits != 25 ||
        config.error_sigma.to_bits() != 4.0_f64.to_bits()
    {
        return Err(format!("{REVIEWED_BGG_ENVELOPE_SOURCE} does not cover candidate domain"));
    }
    if candidate.crt_depth == 0 || candidate.crt_depth > 27 {
        return Err(format!("{REVIEWED_BGG_ENVELOPE_SOURCE} covers CRT depth 1..=27 only"));
    }
    let q27 = DCRTPolyParams::new(1 << 16, 27, 50, 25, None, None).modulus();
    let candidate_q = DCRTPolyParams::new(
        1 << candidate.log_ring_dimension,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
        None,
        None,
    )
    .modulus();
    if candidate_q.as_ref() > q27.as_ref() {
        return Err(format!("{REVIEWED_BGG_ENVELOPE_SOURCE} rejects modulus above q_27"));
    }
    Ok(136)
}

/// Applies the backend-owned capability check before any security callback or
/// OpenFHE constructor can observe a candidate. The search deliberately keeps
/// no duplicate CRT/base limits of its own.
fn validate_backend_candidate(candidate: Candidate) -> Result<(), String> {
    let ring_dimension = 1usize
        .checked_shl(candidate.log_ring_dimension as u32)
        .ok_or_else(|| "ring dimension shift overflow".to_owned())?;
    let ring_dimension = u32::try_from(ring_dimension)
        .map_err(|_| "ring dimension does not fit the DCRT backend API".to_owned())?;
    DCRTPolyParams::validate_capability(
        ring_dimension,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
    )
    .map_err(|error| error.to_string())
}

/// Builds the concrete public setup and ordinary sparse-LWR program for one
/// candidate.  It does not execute a backend, sample artifacts, or perform a
/// round trip; those are intentionally outside this symbolic search.
pub fn prepare_candidate(
    config: &SearchConfig,
    candidate: Candidate,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    prepare_candidate_with_digits_inner(config, candidate, None, false, None)
}

pub fn prepare_candidate_with_fixed_digits(
    config: &SearchConfig,
    candidate: Candidate,
    mask_digits: usize,
    fresh_error_digits: usize,
    switch_moduli: &[BigUint],
) -> Result<PreparedCandidate, CandidatePreparationError> {
    if mask_digits == 0 || fresh_error_digits == 0 {
        return Err("fixed d_m and d_e must be positive".to_owned().into());
    }
    prepare_candidate_with_digits_inner(
        config,
        candidate,
        Some((mask_digits, fresh_error_digits)),
        true,
        Some(switch_moduli),
    )
}

struct CandidatePreparationContext {
    candidate: Candidate,
    ring_dimension: usize,
    layout: mxx_exponent_lut::ExpLayout,
    program: SparseLwrPrfProgram,
    crt_moduli: Vec<u64>,
    switch_moduli: Vec<BigUint>,
    sampler_layout: BggSamplerLayout,
    refresh: RefreshCompiler,
    generated_layout: PbcGeneratedKeyLayout,
    active_cells: PbcActiveCellIndex,
    pbc_active_count: usize,
    base_p: usize,
    q_l: BigUint,
    derived_fresh_error_digits: usize,
}

fn build_candidate_context(
    config: &SearchConfig,
    candidate: Candidate,
) -> Result<CandidatePreparationContext, CandidatePreparationError> {
    let ring_dimension =
        1usize.checked_shl(candidate.log_ring_dimension as u32).ok_or_else(|| {
            CandidatePreparationError::Fatal("ring dimension shift overflow".to_owned())
        })?;
    validate_backend_candidate(candidate).map_err(CandidatePreparationError::Infeasible)?;
    let layout = native_exp_layout(config.sparse_lwr_modulus, ring_dimension)
        .map_err(CandidatePreparationError::Infeasible)?;
    let profile = SparseLwrPrfProfile::new(layout, config.sparse_lwr_output_modulus / 2)
        .map_err(|error| CandidatePreparationError::Infeasible(error.to_string()))?;
    let dcrt = DCRTPolyParams::new(
        ring_dimension as u32,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
        None,
        None,
    );
    let (crt_moduli, _, _) = dcrt.to_crt();
    let modulus = BigInt::from_biguint(num_bigint::Sign::Plus, dcrt.modulus().as_ref().clone());
    let sampler_layout = BggSamplerLayout {
        modulus: modulus.clone().into(),
        ring_dimension: ring_dimension.into(),
        secret_dimension: config.secret_dimension,
        digit_count: dcrt
            .modulus_digits()
            .checked_div(candidate.crt_depth)
            .and_then(|per_tower| {
                candidate
                    .crt_depth
                    .checked_sub(config.gadget_dropped_moduli)
                    .filter(|retained| *retained > 0)
                    .map(|retained| retained * per_tower)
            })
            .ok_or_else(|| {
                CandidatePreparationError::Infeasible("invalid dropped gadget towers".to_owned())
            })?,
        gadget_base: (BigInt::from(1_u8) << candidate.base_bits).into(),
    };
    let refresh = RefreshCompiler {
        full_modulus: modulus.clone().into(),
        crt_plaintext_moduli: crt_moduli.iter().map(|value| BigInt::from(*value).into()).collect(),
        reconstruction_coefficients: exact_reconstruction_coefficients(&dcrt),
    };
    let pbc_parameters =
        PbcParameters::paper_evaluation(config.sparse_lwr_universe, config.sparse_lwr_weight);
    if pbc_parameters.max_seed_attempts != config.pbc_max_attempts {
        return Err(CandidatePreparationError::Fatal(
            "paper-evaluation PBC profile does not use the reviewed retry count".to_owned(),
        ));
    }
    let mut support_rng = StdRng::from_seed([0x6a; 32]);
    let support = sample(&mut support_rng, config.sparse_lwr_universe, config.sparse_lwr_weight)
        .into_iter()
        .collect::<Vec<_>>();
    let generated = generate_key_layout(&pbc_parameters, PbcRootSeed([0x19; 32]), &support)
        .map_err(|error| {
            CandidatePreparationError::Fatal(format!("PBC layout generation: {error}"))
        })?;
    generated.private_schedule().validate(&generated.public_layout).map_err(|error| {
        CandidatePreparationError::Fatal(format!("PBC schedule validation: {error}"))
    })?;
    let active_cells = PbcActiveCellIndex::build(&generated.public_layout).map_err(|error| {
        CandidatePreparationError::Fatal(format!("PBC active-cell index: {error}"))
    })?;
    let pbc_active_count = active_cells.len();
    let base_p = config.sparse_lwr_output_modulus;
    let q_l = BigUint::from(config.sparse_lwr_modulus);
    let derived_fresh_error_digits = fresh_error_base_p_digit_count(&base_p.into(), &crt_moduli)
        .map_err(|error| CandidatePreparationError::Infeasible(error.to_string()))?;
    let program = SparseLwrPrfProgram::new(
        profile,
        generated.public_layout.bucket_width,
        generated.public_layout.parameters.bucket_count,
    )
    .map_err(|error| CandidatePreparationError::Fatal(format!("PRF Program: {error}")))?;
    Ok(CandidatePreparationContext {
        candidate,
        ring_dimension,
        layout,
        program,
        switch_moduli: vec![dcrt.modulus().as_ref().clone(); crt_moduli.len()],
        crt_moduli,
        sampler_layout,
        refresh,
        generated_layout: generated,
        active_cells,
        pbc_active_count,
        base_p,
        q_l,
        derived_fresh_error_digits,
    })
}

fn setup_for_candidate_context(
    config: &SearchConfig,
    context: &CandidatePreparationContext,
    mask_digits: usize,
    fresh_error_digits: usize,
) -> Result<BggRefreshParameters, CandidatePreparationError> {
    let sigma_decoder = RealExpr::from_f64_exact(config.decoder_sigma)
        .map_err(|error| CandidatePreparationError::Fatal(format!("decoder sigma: {error}")))?;
    let sigma_encoding = RealExpr::from_f64_exact(config.error_sigma)
        .map_err(|error| CandidatePreparationError::Fatal(format!("encoding sigma: {error}")))?;
    let mut setup = BggRefreshParameters::new(
        [0x52; 32],
        context.base_p,
        config.secret_dimension,
        context.ring_dimension,
        mask_digits,
        fresh_error_digits,
        config.mask_statistical_security_bits,
        context.layout.width(),
        context.sampler_layout.clone(),
        context.refresh.clone(),
        sigma_decoder,
        sigma_encoding,
        "refresh-parameter-search",
    )
    .with_decoder_preimage_bound(config.decoder_preimage_bound.clone());
    setup.switch_moduli = context.switch_moduli.clone();
    Ok(setup)
}

/// Builds the actual verification program with fresh test-owned PBC randomness.
/// Search fixtures remain deterministic; runtime correctness never reuses their seeds.
pub fn round_trip_bundle(
    config: &SearchConfig,
    candidate: Candidate,
    mask_digits: usize,
    fresh_digits: usize,
    plaintext_exponent: usize,
) -> Result<(RefreshParameterSimulationBundle, PbcGeneratedKeyLayout, SparseLwrPrfProfile), String>
{
    let mut context = build_candidate_context(config, candidate).map_err(|e| e.to_string())?;
    let mut rng = rand::rng();
    let support = sample(&mut rng, config.sparse_lwr_universe, config.sparse_lwr_weight).into_vec();
    let root_seed = rand::Rng::random(&mut rng);
    context.generated_layout = generate_key_layout(
        &PbcParameters::paper_evaluation(config.sparse_lwr_universe, config.sparse_lwr_weight),
        PbcRootSeed(root_seed),
        &support,
    )
    .map_err(|e| e.to_string())?;
    let setup = setup_for_candidate_context(config, &context, mask_digits, fresh_digits)
        .map_err(|e| e.to_string())?;
    let plaintext = setup.layout.ring().constant(
        (1, 1),
        mxx_ir_core::node::ConstantMatrix::Rotation { exponent: plaintext_exponent.into() },
    );
    let bundle = RefreshParameterSimulationRequest::new(
        setup,
        context.program.clone(),
        context.generated_layout.clone(),
        rand::Rng::random(&mut rng),
        plaintext,
        1,
    )
    .map_err(|e| e.to_string())?
    .build()
    .map_err(|e| e.to_string())?;
    Ok((bundle, context.generated_layout, context.program.profile()))
}

fn prepare_candidate_with_digits_inner(
    config: &SearchConfig,
    candidate: Candidate,
    fixed_digits: Option<(usize, usize)>,
    build_full_bundle: bool,
    selected_switch_moduli: Option<&[BigUint]>,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    let mut context = build_candidate_context(config, candidate)?;
    if let Some(selected) = selected_switch_moduli {
        context.switch_moduli = selected.to_vec();
        let prepared =
            prepare_candidate_from_ring_context(config, &context, fixed_digits, build_full_bundle)?;
        return certify_selected_candidate(config, &context, prepared);
    }
    let mut rejections = Vec::new();
    // Deterministic coarse search: increasing retained width, shared contiguous
    // groups, and an overlapping full-width final group. Full q is evaluated
    // last as an ordinary candidate, not as an evaluator fallback.
    for retained in 1..=context.crt_moduli.len() {
        context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, retained)?;
        match prepare_candidate_from_ring_context(config, &context, fixed_digits, build_full_bundle)
        {
            Ok(prepared) => return certify_selected_candidate(config, &context, prepared),
            Err(CandidatePreparationError::Infeasible(reason)) => {
                rejections.push(format!("retained={retained}: {reason}"))
            }
            Err(error) => return Err(error),
        }
    }
    Err(CandidatePreparationError::Infeasible(format!(
        "coarse switched-ring scan rejected: {}",
        rejections.join("; ")
    )))
}

// Rebuild only the selected tuple, using the identical setup, public layout and
// deterministic search key. Cached profiles also enter through this boundary.
fn certify_selected_candidate(
    config: &SearchConfig,
    context: &CandidatePreparationContext,
    mut prepared: PreparedCandidate,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    if config.noise_model == NoiseSearchModel::AverageCase {
        return Ok(prepared);
    }
    if !evaluate_refresh_bundle(config, context.candidate, &mut prepared)
        .map_err(CandidatePreparationError::Fatal)?
    {
        return Err(CandidatePreparationError::Infeasible(
            "selected worst-case bounds reject".into(),
        ));
    }
    let expected_report = prepared.production_report.clone();
    let setup = setup_for_candidate_context(
        config,
        context,
        prepared.mask_base_p_digit_count,
        prepared.fresh_error_base_p_digit_count,
    )?;
    let plaintext = setup.layout.ring().polynomial([BigInt::zero().into()]);
    let bundle = RefreshParameterSimulationRequest::new(
        setup,
        context.program.clone(),
        context.generated_layout.clone(),
        [0x4b; 32],
        plaintext,
        1,
    )
    .and_then(|request| request.build())
    .map_err(|error| CandidatePreparationError::Fatal(format!("selected proof graph: {error}")))?;
    if let Some(snapshot) = prepared
        .noise_snapshot
        .as_ref()
        .or_else(|| prepared.bundle.as_ref().map(|bundle| bundle.noise_snapshot()))
    {
        let full = bundle.noise_snapshot();
        if snapshot.setup_identity() != full.setup_identity() ||
            snapshot.authority != full.authority ||
            snapshot.layout != full.layout ||
            snapshot.program.id() != full.program.id() ||
            snapshot.refresh != full.refresh ||
            snapshot.gate_gains != full.gate_gains ||
            snapshot.readout_gains != full.readout_gains ||
            snapshot.initial_state_bound != full.initial_state_bound ||
            snapshot.initial_state_doubled_variance != full.initial_state_doubled_variance ||
            snapshot.helper_error_cutoff != full.helper_error_cutoff ||
            snapshot.helper_doubled_variance != full.helper_doubled_variance ||
            snapshot.inputs != full.inputs ||
            snapshot.gain_plan != full.gain_plan ||
            snapshot.family_active_counts != full.family_active_counts
        {
            return Err(CandidatePreparationError::Fatal(
                "selected compact and proof snapshots differ".into(),
            ));
        }
    }
    prepared.bundle = Some(bundle);
    evaluate_refresh_bundle(config, context.candidate, &mut prepared)
        .map_err(CandidatePreparationError::Fatal)?;
    if prepared.production_report != expected_report {
        return Err(CandidatePreparationError::Fatal(
            "selected compact and proof reports differ".into(),
        ));
    }
    // Parameter preparation returns noise evidence. Explicit certificate checks
    // remain in check_refresh_bundle; simulation/measurement never runs Lean.
    Ok(prepared)
}

fn candidate_switch_moduli(
    crt_moduli: &[u64],
    retained: usize,
) -> Result<Vec<BigUint>, CandidatePreparationError> {
    if retained == 0 || retained > crt_moduli.len() {
        return Err(CandidatePreparationError::Fatal("invalid retained CRT width".into()));
    }
    Ok((0..crt_moduli.len())
        .map(|slot| {
            let start = ((slot / retained) * retained).min(crt_moduli.len() - retained);
            crt_moduli[start..start + retained]
                .iter()
                .fold(BigUint::one(), |product, prime| product * *prime)
        })
        .collect())
}

fn prepare_candidate_from_ring_context(
    config: &SearchConfig,
    context: &CandidatePreparationContext,
    fixed_digits: Option<(usize, usize)>,
    build_full_bundle: bool,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    let candidate = context.candidate;
    match fixed_digits {
        Some((mask_digits, fresh_error_digits)) => prepare_candidate_from_context(
            config,
            &context,
            mask_digits,
            fresh_error_digits,
            build_full_bundle,
            None,
            None,
            None,
        ),
        None => {
            let base_p_big = BigUint::from(context.base_p);
            let (first_digits, last_digits) = mask_digit_search_bounds(
                &base_p_big,
                &context.q_l,
                &context.crt_moduli,
                &context.switch_moduli,
            )?;
            let invariant_setup = setup_for_candidate_context(
                config,
                &context,
                first_digits,
                context.derived_fresh_error_digits,
            )?;
            let (gain_plan, invariant_routes) =
                BggRefreshParameters::noise_gain_and_route_invariants_from_public_layout(
                    &invariant_setup,
                    &context.program,
                    &context.generated_layout.public_layout,
                )
                .map_err(|error| CandidatePreparationError::Infeasible(error.to_string()))?;
            let mut first_prepared = prepare_candidate_from_context(
                config,
                &context,
                first_digits,
                context.derived_fresh_error_digits,
                false,
                Some(&gain_plan),
                Some(&invariant_routes),
                None,
            )?;
            let pre_refresh = first_prepared.pre_refresh.clone();
            let scan_base = first_prepared
                .noise_snapshot
                .as_ref()
                .ok_or_else(|| {
                    CandidatePreparationError::Fatal("compact scan snapshot missing".to_owned())
                })?
                .clone();
            if config.noise_model == NoiseSearchModel::WorstCase {
                // State and shared-fresh routes do not depend on mask digit
                // count. Even dropping all mask-route, rounding and decoder
                // errors cannot repair 2 H >= kappa, since
                // (H/J)/Delta = H/kappa for every selected P.
                let routes = scan_base.refresh.route_gains.as_ref().ok_or_else(|| {
                    CandidatePreparationError::Fatal("missing route gains".into())
                })?;
                let readout = &pre_refresh
                    .as_ref()
                    .ok_or_else(|| {
                        CandidatePreparationError::Fatal("missing pre-refresh report".into())
                    })?
                    .readout_bound;
                let q = scan_base.refresh.global_modulus.as_ref().ok_or_else(|| {
                    CandidatePreparationError::Fatal("missing global modulus".into())
                })?;
                for slot in 0..scan_base.refresh.slot_count {
                    let unavoidable = &routes.state_scaling[slot].l1 *
                        &scan_base.initial_state_bound +
                        &routes.fresh[slot].l1 * readout;
                    let kappa = q / &scan_base.refresh.plaintext_moduli[slot];
                    if &unavoidable * 2u8 >= kappa {
                        return Err(CandidatePreparationError::Infeasible(format!(
                            "no mask digit count passed the complete production report in d_m={first_digits}..={last_digits}: exact necessary bound rejects slot {slot}, 2*H_unavoidable={} >= kappa={kappa}",
                            &unavoidable * 2u8
                        )));
                    }
                }
            }
            let mask_gains = (first_digits..=last_digits)
                .map(|mask_digits| {
                    BggRefreshParameters::mask_route_gain_for_digit_count(
                        &invariant_setup,
                        mask_digits,
                    )
                    .map_err(|error| CandidatePreparationError::Infeasible(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (offset, mask_digits) in (first_digits..=last_digits).enumerate() {
                let accepted = if offset == 0 {
                    evaluate_refresh_bundle(config, candidate, &mut first_prepared)
                        .map_err(CandidatePreparationError::Fatal)?
                } else {
                    let scan_snapshot =
                        snapshot_with_mask_gain(&scan_base, mask_digits, &mask_gains[offset])
                            .map_err(CandidatePreparationError::Infeasible)?;
                    if config.noise_model == NoiseSearchModel::AverageCase {
                        scan_snapshot
                            .average_refresh_accepted_from_pre_refresh(
                                &average_case_config(config)
                                    .map_err(CandidatePreparationError::Fatal)?,
                                pre_refresh.as_ref().ok_or_else(|| {
                                    CandidatePreparationError::Fatal(
                                        "compact scan pre-refresh missing".to_owned(),
                                    )
                                })?,
                            )
                            .map_err(|error| {
                                CandidatePreparationError::Infeasible(error.to_string())
                            })?
                    } else {
                        scan_snapshot
                            .worst_refresh_accepted_from_pre_refresh(
                                pre_refresh.as_ref().ok_or_else(|| {
                                    CandidatePreparationError::Fatal(
                                        "compact scan pre-refresh missing".to_owned(),
                                    )
                                })?,
                            )
                            .map_err(|error| {
                                CandidatePreparationError::Infeasible(error.to_string())
                            })?
                    }
                };
                if accepted {
                    if offset == 0 {
                        return Ok(first_prepared);
                    }
                    let mut prepared = prepare_candidate_from_context(
                        config,
                        &context,
                        mask_digits,
                        context.derived_fresh_error_digits,
                        false,
                        Some(&gain_plan),
                        Some(&invariant_routes),
                        pre_refresh.as_ref(),
                    )?;
                    let checked = evaluate_refresh_bundle(config, candidate, &mut prepared)
                        .map_err(CandidatePreparationError::Fatal)?;
                    if checked {
                        return Ok(prepared);
                    }
                    return Err(CandidatePreparationError::Fatal(
                        "cached scan and materialized snapshot disagree".to_owned(),
                    ));
                }
            }
            Err(CandidatePreparationError::Infeasible(format!(
                "no mask digit count passed the complete production report in d_m={first_digits}..={last_digits}"
            )))
        }
    }
}

pub(crate) fn snapshot_with_mask_gain(
    base: &BggNoiseSnapshot,
    mask_digits: usize,
    mask_gain: &[RightActionNoiseGain],
) -> Result<BggNoiseSnapshot, String> {
    if base.refresh.route_bias_bounds.iter().any(|bias| !bias.is_zero()) {
        return Err("approximate mask scan must rebuild corrected route bias for each digit count"
            .to_owned());
    }
    let mut snapshot = base.clone();
    let mut refresh = snapshot.refresh.clone();
    if mask_gain.len() != refresh.slot_count {
        return Err("compact scan mask gain slot count differs".to_owned());
    }
    refresh.mask_digit_counts.fill(mask_digits);
    let topology = refresh
        .transcript_topology
        .as_mut()
        .ok_or_else(|| "compact snapshot transcript topology is missing".to_owned())?;
    topology.mask_digit_counts.fill(mask_digits);
    let transcript_coordinates = topology
        .crt_slot_count
        .checked_mul(topology.component_columns)
        .and_then(|value| value.checked_mul(topology.coefficient_count))
        .ok_or_else(|| "compact scan mask coordinate count overflows".to_owned())?;
    refresh
        .smudging
        .as_mut()
        .ok_or_else(|| "compact snapshot smudging parameters are missing".to_owned())?
        .transcript_coordinates = transcript_coordinates;
    let mut route_gains = refresh
        .route_gains
        .clone()
        .ok_or_else(|| "compact snapshot route gains are missing".to_owned())?;
    route_gains.mask = mask_gain.to_vec();
    refresh.route_gains = Some(route_gains);
    snapshot.refresh = refresh;
    Ok(snapshot)
}

fn prepare_candidate_from_context(
    config: &SearchConfig,
    context: &CandidatePreparationContext,
    mask_digits: usize,
    fresh_error_digits: usize,
    build_full_bundle: bool,
    cached_gain_plan: Option<&mxx_exponent_lut::bgg::noise::NoiseGainPlan>,
    cached_invariant_routes: Option<&RefreshNoiseGainDecompositions>,
    cached_pre_refresh: Option<&PreRefreshNoiseEvaluation<ProgramNoiseReport>>,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    let candidate = context.candidate;
    let ring_dimension = context.ring_dimension;
    let pbc_active_count = context.pbc_active_count;
    let generated_layout = &context.generated_layout;
    if fresh_error_digits != context.derived_fresh_error_digits {
        return Err(CandidatePreparationError::Infeasible(
            "fixed fresh-error digit count is not the setup-derived injective count".to_owned(),
        ));
    }
    if mask_digits == 0 || fresh_error_digits == 0 {
        return Err(CandidatePreparationError::Fatal(
            "native digit counts must be positive".to_owned(),
        ));
    }
    let setup = setup_for_candidate_context(config, context, mask_digits, fresh_error_digits)?;
    let selected_preimage_bound_expr = setup.resolve_decoder_preimage_bound().map_err(|error| {
        CandidatePreparationError::Infeasible(format!("selected decoder bound: {error}"))
    })?;
    let official_preimage_bound_expr = setup
        .clone()
        .with_decoder_preimage_bound(TrapdoorPreimageCutoffPolicy::Default)
        .resolve_decoder_preimage_bound()
        .map_err(|error| {
            CandidatePreparationError::Infeasible(format!("official decoder bound: {error}"))
        })?;
    let selected_value =
        selected_preimage_bound_expr.evaluate(&mxx_ir_core::ParamEnv::default()).map_err(
            |error| CandidatePreparationError::Fatal(format!("selected bound evaluation: {error}")),
        )?;
    let official_value =
        official_preimage_bound_expr.evaluate(&mxx_ir_core::ParamEnv::default()).map_err(
            |error| CandidatePreparationError::Fatal(format!("official bound evaluation: {error}")),
        )?;
    validate_decoder_preimage_cutoff(
        &selected_value.to_biguint().ok_or_else(|| {
            CandidatePreparationError::Fatal(
                "selected decoder bound must be non-negative".to_owned(),
            )
        })?,
        &official_value.to_biguint().ok_or_else(|| {
            CandidatePreparationError::Fatal(
                "official decoder bound must be non-negative".to_owned(),
            )
        })?,
    )
    .map_err(CandidatePreparationError::Fatal)?;
    let anchor_security_inputs = setup
        .anchor_security_inputs(1)
        .map_err(|error| CandidatePreparationError::Fatal(error.to_string()))?;
    let prf_component_count = setup.prf_component_count();
    let prf_label_count = setup
        .refresh
        .crt_plaintext_moduli
        .len()
        .checked_mul(prf_component_count)
        .and_then(|value| value.checked_mul(setup.coefficient_count))
        .and_then(|value| value.checked_mul(mask_digits))
        .and_then(|mask| {
            prf_component_count
                .checked_mul(setup.coefficient_count)
                .and_then(|value| value.checked_mul(fresh_error_digits))
                .and_then(|fresh| mask.checked_add(fresh))
        })
        .ok_or_else(|| {
            CandidatePreparationError::Fatal("refresh PRF label count overflows usize".to_owned())
        })?;
    let prf_value_count = prf_label_count.checked_mul(pbc_active_count).ok_or_else(|| {
        CandidatePreparationError::Fatal("refresh PRF value count overflows usize".to_owned())
    })?;
    let pbc_layout = &generated_layout.public_layout;
    let layout_id = pbc_layout.layout_id.0;
    let pbc_attempts_used = pbc_layout.accepted_attempt + 1;
    let bucket_width = pbc_layout.bucket_width;
    let prf_coefficient_count = setup.coefficient_count;
    let program_id = *context.program.id().as_bytes();
    let (bundle, noise_snapshot) = if build_full_bundle {
        // Fixed-digit callers are selected candidates used by benchmark
        // estimation; retain the complete production benchmark graph there.
        let expected_plaintext = setup.layout.ring().polynomial([BigInt::zero().into()]);
        let request = RefreshParameterSimulationRequest::new(
            setup,
            context.program.clone(),
            generated_layout.clone(),
            [0x4b; 32],
            expected_plaintext,
            1,
        )
        .map_err(|error| {
            CandidatePreparationError::Fatal(format!("refresh simulation request: {error}"))
        })?;
        let bundle = request.build_benchmark().map_err(|error| {
            CandidatePreparationError::Infeasible(format!("refresh benchmark graph: {error}"))
        })?;
        (Some(bundle), None)
    } else {
        // Ordinary search uses the same setup-owned snapshot construction but
        // does not expand the label-major executable graph for every mask
        // digit candidate.
        let snapshot = setup
            .build_noise_snapshot_from_public_layout(
                &context.program,
                pbc_layout,
                &context.active_cells,
                cached_gain_plan.map(|gain_plan| (gain_plan, cached_invariant_routes)),
            )
            .map_err(|error| {
                CandidatePreparationError::Infeasible(format!("compact noise snapshot: {error}"))
            })?;
        (None, Some(snapshot))
    };
    let pre_refresh = noise_snapshot.as_ref().map(|snapshot| {
        cached_pre_refresh.cloned().unwrap_or_else(|| {
            snapshot
                .evaluate_pre_refresh()
                .expect("validated compact snapshot pre-refresh evaluation")
        })
    });
    Ok(PreparedCandidate {
        anchor_security_inputs,
        candidate,
        ring_dimension,
        bucket_width,
        official_preimage_bound: official_value,
        selected_preimage_bound: selected_value,
        layout_id,
        program_id,
        pbc_attempts_used,
        mask_base_p_digit_count: mask_digits,
        fresh_error_base_p_digit_count: fresh_error_digits,
        prf_component_count,
        prf_coefficient_count,
        prf_active_count: pbc_active_count,
        prf_label_count,
        prf_value_count,
        bundle,
        noise_snapshot,
        pre_refresh,
        production_report: None,
        average_report: None,
    })
}

/// Accepts a selected decoder cutoff only when it is at least the resolved
/// official cutoff. This keeps a larger explicitly reviewed cutoff possible
/// while rejecting the unsafe below-official boundary.
pub fn validate_decoder_preimage_cutoff(
    selected: &BigUint,
    official: &BigUint,
) -> Result<(), String> {
    if selected < official {
        return Err("selected decoder preimage cutoff is below the official bound".to_owned());
    }
    Ok(())
}

fn minimum_mask_base_p_digit_count(q_l: usize, base_p: usize) -> Result<usize, String> {
    if q_l == 0 || base_p < 2 {
        return Err("invalid base-p digit-count inputs".to_owned());
    }
    let mut covered = 1usize;
    let mut digits = 0usize;
    while covered < q_l {
        covered =
            covered.checked_mul(base_p).ok_or_else(|| "base-p digit-count overflow".to_owned())?;
        digits = digits.checked_add(1).ok_or_else(|| "base-p digit-count overflow".to_owned())?;
    }
    Ok(digits)
}

/// Returns the finite `d_m` interval that may satisfy the strict deterministic
/// CRT precondition.  The lower endpoint is exactly `ceil_log_P(Q_L)`; the
/// upper endpoint is the last digit before `2 floor(M/2) >= min Delta`. The exact
/// report checks `2(B_m + F) < Delta`; this weaker monotone stopping condition
/// cannot skip a valid candidate and leaves the final decision to production
/// report evaluation.
fn mask_digit_search_bounds(
    base_p: &BigUint,
    q_l: &BigUint,
    crt_moduli: &[u64],
    switch_moduli: &[BigUint],
) -> Result<(usize, usize), CandidatePreparationError> {
    if base_p < &BigUint::from(2u8) ||
        q_l.is_zero() ||
        crt_moduli.is_empty() ||
        switch_moduli.len() != crt_moduli.len()
    {
        return Err(CandidatePreparationError::Fatal(
            "invalid mask digit-search boundary inputs".to_owned(),
        ));
    }
    let first = minimum_mask_base_p_digit_count(
        q_l.to_usize()
            .ok_or_else(|| CandidatePreparationError::Fatal("Q_L does not fit usize".to_owned()))?,
        base_p.to_usize().ok_or_else(|| {
            CandidatePreparationError::Fatal("base-p modulus does not fit usize".to_owned())
        })?,
    )?;
    let moduli = crt_moduli.iter().copied().map(BigUint::from).collect::<Vec<_>>();
    if moduli.iter().zip(switch_moduli).any(|(qt, p)| p.is_zero() || p % qt != BigUint::zero()) {
        return Err(CandidatePreparationError::Fatal(
            "selected ring does not contain its CRT slot".into(),
        ));
    }
    let min_spacing = moduli
        .iter()
        .zip(switch_moduli)
        .map(|(modulus, selected)| selected / modulus)
        .min()
        .ok_or_else(|| CandidatePreparationError::Fatal("missing CRT spacing".to_owned()))?;
    let mut last = first;
    loop {
        let exponent = u32::try_from(last).map_err(|_| {
            CandidatePreparationError::Fatal("mask digit count overflow".to_owned())
        })?;
        let bound = (base_p.pow(exponent) / 2u8) * 2u8;
        if bound >= min_spacing {
            if last == first {
                return Err(CandidatePreparationError::Infeasible(
                    "strict deterministic mask boundary is below the domain minimum".to_owned(),
                ));
            }
            return Ok((first, last - 1));
        }
        last = last.checked_add(1).ok_or_else(|| {
            CandidatePreparationError::Fatal("mask digit count overflow".to_owned())
        })?;
    }
}

/// Chooses the smallest fresh-error digit count for which B_e=p^d_e-1 is
/// strictly below every prime factor of every authoritative CRT tower.
fn fresh_error_base_p_digit_count(
    base_p: &BigUint,
    crt_moduli: &[u64],
) -> Result<usize, CandidatePreparationError> {
    if base_p < &BigUint::from(2u8) ||
        crt_moduli.is_empty() ||
        crt_moduli.iter().any(|modulus| *modulus <= 1)
    {
        return Err(CandidatePreparationError::Fatal(
            "invalid fresh-error digit-count inputs".to_owned(),
        ));
    }
    let fresh_bound = base_p - BigUint::one();
    for modulus in crt_moduli {
        let accepted = fresh_error_below_every_prime_factor(&fresh_bound, &BigUint::from(*modulus))
            .map_err(|error| CandidatePreparationError::Fatal(error.to_string()))?;
        if !accepted {
            return Err(CandidatePreparationError::Infeasible(
                "no fresh-error digit count satisfies B_e < every prime factor of every CRT modulus"
                    .to_owned(),
            ));
        }
    }
    Ok(1)
}

/// Builds the single AverageCase policy used by both d_m selection and the
/// final bundle check.  Keeping this projection centralized prevents a
/// selector from silently evaluating a different event/failure budget than
/// the accepted graph.
pub fn average_case_config(config: &SearchConfig) -> Result<AverageCaseConfig, String> {
    Ok(AverageCaseConfig {
        failure_exponent: u32::try_from(config.security_bits)
            .map_err(|_| "AverageCase security target does not fit u32".to_owned())?,
        allow_average_acceptance: true,
        ..Default::default()
    })
}

/// Builds the arguments for the BGG ternary/Gaussian estimator invocation.
fn bgg_estimator_args(ring_dimension: usize, modulus: &BigUint, sigma: f64) -> Vec<String> {
    vec![
        ring_dimension.to_string(),
        modulus.to_string(),
        "--s-dist".to_owned(),
        r#"{"name":"Ternary"}"#.to_owned(),
        "--e-dist".to_owned(),
        format!(r#"{{"name":"DiscreteGaussian","stddev":{sigma}}}"#),
    ]
}

/// Derives the centered rounding-error interval for a lifted LWR sample.
///
/// The sparse-LWR estimator models the marginal error after lifting the
/// output modulus.  With `Delta = Q_L / p`, write the transformed sample as
/// `x = Delta * floor(x / Delta) + r`, where `0 <= r < Delta`; after the
/// known centered shift `floor(Delta / 2)`, the estimator error is
/// `e = r - floor(Delta / 2)`, giving the returned interval `[a, b]`.  This
/// is only a marginal surrogate: it does not model correlations between the
/// public coefficient and its rounding error.  The separate sparse-support
/// MITM floor remains part of the final minimum and is not folded into this
/// estimate.
pub fn sparse_lwr_error_bounds(q_l: usize, p: usize) -> Result<(i64, i64), String> {
    if q_l == 0 || !q_l.is_power_of_two() {
        return Err("native Q must be a positive power of two".to_owned());
    }
    if p < 2 || !p.is_power_of_two() || p > q_l || q_l % p != 0 {
        return Err("native P must be a power of two, satisfy 2 <= P <= Q, and divide Q".to_owned());
    }
    let delta = q_l / p;
    let shift = delta / 2;
    let delta = i64::try_from(delta).map_err(|_| "sparse-LWR Delta overflows i64".to_owned())?;
    let shift = i64::try_from(shift).map_err(|_| "sparse-LWR shift overflows i64".to_owned())?;
    let upper = delta
        .checked_sub(1)
        .and_then(|value| value.checked_sub(shift))
        .ok_or_else(|| "sparse-LWR error interval overflows i64".to_owned())?;
    Ok((-shift, upper))
}

/// Builds the exact sparse-LWR estimator invocation for the reviewed model.
/// `--m` is intentionally omitted, so the CLI uses its default infinite
/// lifted dimension.  This does not ask the CLI to derive a finite sample
/// count or replace the transformed-error model documented above.
pub fn sparse_lwr_estimator_args(
    universe: usize,
    weight: usize,
    q_l: usize,
    p: usize,
) -> Result<Vec<String>, String> {
    let (a, b) = sparse_lwr_error_bounds(q_l, p)?;
    Ok(vec![
        universe.to_string(),
        q_l.to_string(),
        "--s-dist".to_owned(),
        format!(r#"{{"name":"SparseBinary","hw":{weight},"n":{universe}}}"#),
        "--e-dist".to_owned(),
        format!(r#"{{"name":"Uniform","a":{a},"b":{b}}}"#),
        "--exact".to_owned(),
    ])
}

/// Parses the estimator output while keeping the security result unambiguous.
///
/// The bundled CLI prints human-readable algorithm diagnostics followed by
/// one unsigned-decimal security estimate.  Diagnostics are allowed, but the
/// parser rejects output containing zero or multiple decimal result lines so
/// that a changed CLI format cannot silently select an arbitrary estimate.
pub fn parse_security_bits(stdout: &[u8]) -> Result<u64, String> {
    let text = std::str::from_utf8(stdout)
        .map_err(|error| format!("lattice-estimator-cli output is not UTF-8: {error}"))?;
    let result_lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| line.bytes().all(|byte| byte.is_ascii_digit()))
        .collect::<Vec<_>>();
    let line = match result_lines.as_slice() {
        [line] => line,
        [] => {
            return Err(
                "lattice-estimator-cli output contains no unsigned-decimal security estimate"
                    .to_owned(),
            );
        }
        _ => {
            return Err(
                "lattice-estimator-cli output contains multiple security estimates".to_owned()
            );
        }
    };
    line.parse::<u64>().map_err(|error| format!("invalid security estimate: {error}"))
}

fn run_lattice_estimator(args: Vec<String>) -> Result<u64, String> {
    let output = Command::new("lattice-estimator-cli")
        .args(args)
        .output()
        .map_err(|error| format!("failed to start lattice-estimator-cli: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "lattice-estimator-cli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_security_bits(&output.stdout)
}

/// Calls the existing BGG estimator wrapper used by the parameter-search
/// tests.  The result is parsed as one unambiguous security-bits line.
pub fn lattice_security_bits(
    ring_dimension: usize,
    modulus: &BigUint,
    sigma: f64,
) -> Result<u64, String> {
    run_lattice_estimator(bgg_estimator_args(ring_dimension, modulus, sigma))
}

/// Raw support-key entropy, retained as a diagnostic rather than a separate
/// hard security floor.
fn raw_key_entropy_bits(universe: usize, weight: usize) -> f64 {
    (0..weight).map(|index| ((universe - index) as f64).log2() - ((index + 1) as f64).log2()).sum()
}

pub fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Small real-graph fixture shared by runtime smoke tests.
pub fn tiny_native_refresh_bundle() -> (DCRTPolyParams, RefreshParameterSimulationBundle, usize) {
    let poly_parameters = DCRTPolyParams::new(32, 2, 10, 5, None, None);
    let modulus = poly_parameters.modulus().as_ref().clone();
    let (crt_moduli, _, _) = poly_parameters.to_crt();
    let ring = mxx_dsl::Ring::new(
        BigInt::from_biguint(num_bigint::Sign::Plus, modulus.clone()),
        poly_parameters.ring_dimension() as usize,
    );
    let setup = BggRefreshParameters::new(
        [0x91; 32],
        4,
        2,
        32,
        3,
        1,
        1,
        16,
        BggSamplerLayout {
            modulus: BigInt::from_biguint(num_bigint::Sign::Plus, modulus.clone()).into(),
            ring_dimension: 32.into(),
            secret_dimension: 2,
            digit_count: poly_parameters.modulus_digits(),
            gadget_base: (1usize << poly_parameters.base_bits()).into(),
        },
        RefreshCompiler {
            full_modulus: BigInt::from_biguint(num_bigint::Sign::Plus, modulus).into(),
            crt_plaintext_moduli: crt_moduli
                .iter()
                .copied()
                .map(|value| IntExpr::constant(BigInt::from(value)))
                .collect(),
            reconstruction_coefficients: poly_parameters
                .reconst_coeffs()
                .into_iter()
                .map(|value| IntExpr::constant(BigInt::from_biguint(num_bigint::Sign::Plus, value)))
                .collect(),
        },
        1.into(),
        7.into(),
        "tiny-native-refresh",
    );
    let pbc = PbcParameters::custom(1, 1, 2, 2, 1, None);
    let generated =
        generate_key_layout(&pbc, PbcRootSeed([0x92; 32]), &[0]).expect("tiny native PBC layout");
    let active_count = PbcActiveCellIndex::build(&generated.public_layout)
        .expect("tiny native active cells")
        .len();
    let profile =
        SparseLwrPrfProfile::new(mxx_exponent_lut::ExpLayout { stride_log2: 1, width_log2: 4 }, 2)
            .expect("tiny native profile");
    let online_label_count = 3 * setup.prf_component_count() * setup.coefficient_count;
    let total_labels = online_label_count * active_count;
    let request = RefreshParameterSimulationRequest::new(
        setup,
        SparseLwrPrfProgram::new(
            profile,
            generated.public_layout.bucket_width,
            generated.public_layout.parameters.bucket_count,
        )
        .expect("PRF Program"),
        generated,
        [0x93; 32],
        ring.identity(1),
        1,
    )
    .expect("tiny native simulation request");
    (poly_parameters, request.build().expect("tiny native refresh bundle"), total_labels)
}

#[cfg(test)]
pub mod parameter_search_tests {
    use super::*;
    mod parameter_tests {
        use super::*;

        #[test]
        fn estimator_gaussian_stddev_matches_reviewed_encoding_sigma() {
            let config = SearchConfig::reviewed();
            let args =
                super::super::bgg_estimator_args(1 << 16, &BigUint::from(2u8), config.error_sigma);
            let error_distribution = args
                .windows(2)
                .find_map(|window| (window[0] == "--e-dist").then(|| &window[1]))
                .expect("estimator error distribution");
            let error_distribution: serde_json::Value =
                serde_json::from_str(error_distribution).unwrap();
            assert_eq!(error_distribution["stddev"].as_f64(), Some(config.error_sigma),);
        }

        fn evaluate_universal_depth_case(
            config: &SearchConfig,
            directory: &Path,
            depth: usize,
            dropped: usize,
        ) -> serde_json::Value {
            let mut config = config.clone();
            config.gadget_dropped_moduli = dropped;
            let candidate =
                Candidate { crt_depth: depth, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
            let result = (|| -> Result<serde_json::Value, CandidatePreparationError> {
                let mut context = build_candidate_context(&config, candidate)?;
                context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, 5)?;
                let prepared =
                    prepare_candidate_from_ring_context(&config, &context, Some((38, 1)), false)?;
                let snapshot = prepared.noise_snapshot.as_ref().expect("production snapshot");
                let average =
                    snapshot.simulate_average(&average_case_config(&config).unwrap()).unwrap();
                let row = serde_json::json!({
                    "depth":depth,"dropped":dropped,"retained":depth-dropped,
                    "gadget_columns":4*(depth-dropped),"accepted":average.accepted,
                    "full_modulus":snapshot.refresh.global_modulus.as_ref().unwrap().to_string(),
                    "crt_moduli":context.crt_moduli,
                    "switch_moduli":context.switch_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "max_f_avg":average.refresh.max_f_avg.to_string(),
                    "minimum_rounding_margin":average.refresh.slots.iter().map(|s| &s.rounding_margin).min().unwrap().to_string(),
                    "minimum_smudging_margin":average.refresh.slots.iter().map(|s| &s.smudging_margin).min().unwrap().to_string(),
                    "route_bias_bounds":snapshot.refresh.route_bias_bounds.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "average":average,
                    "setup_identity":hex(*snapshot.setup_identity()),
                });
                Ok(row)
            })();
            let row = match result {
                Ok(row) => row,
                Err(error) => serde_json::json!({"depth":depth,"dropped":dropped,
                "accepted":false,"preparation_error":error.to_string()}),
            };
            fs::write(
                directory.join(format!("depth-{depth}-drop-{dropped}.json")),
                serde_json::to_vec_pretty(&row).unwrap(),
            )
            .unwrap();
            eprintln!(
                "DEPTH_WINDOW_CASE depth={depth} dropped={dropped} accepted={} F={} error={}",
                row["accepted"], row["max_f_avg"], row["preparation_error"]
            );
            row
        }

        #[test]
        fn universal_approximate_minimum_depth_window() {
            use rayon::prelude::*;
            let mut config = SearchConfig::reviewed();
            config.noise_model = NoiseSearchModel::AverageCase;
            config.sparse_lwr_universe = 8192;
            config.sparse_lwr_weight = 18;
            config.sparse_lwr_modulus = 4096;
            config.sparse_lwr_output_modulus = 32;
            let run = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/universal_approximate_depth_window")
                .join(run.to_string());
            fs::create_dir_all(&directory).unwrap();
            eprintln!("DEPTH_WINDOW_DIRECTORY {}", directory.display());
            let evaluate =
                |depth, dropped| evaluate_universal_depth_case(&config, &directory, depth, dropped);
            let mut baseline = (1..5)
                .map(|depth| {
                    serde_json::json!({
                        "depth":depth,"dropped":0,"accepted":false,
                        "exclusion":"fixed exact decoder switch width five exceeds full CRT depth"
                    })
                })
                .collect::<Vec<_>>();
            let mut minimum = None;
            for depth in 5..=33 {
                let row = evaluate(depth, 0);
                let accepted = row["accepted"] == true;
                baseline.push(row);
                if accepted {
                    minimum = Some(depth);
                    break;
                }
            }
            let minimum =
                minimum.expect("exact accepted baseline inside conventional security cap");
            assert!(minimum + 3 <= 33, "depth window extends beyond verified security envelope");
            let candidates = (minimum..=minimum + 3)
                .flat_map(|depth| (0..depth).map(move |drop| (depth, drop)))
                .collect::<Vec<_>>();
            let mut rows = Vec::new();
            // Four concurrent independent snapshots cap peak program-state memory.
            for chunk in candidates.chunks(4) {
                rows.extend(
                    chunk
                        .par_iter()
                        .map(|&(depth, drop)| evaluate(depth, drop))
                        .collect::<Vec<_>>(),
                );
            }
            let best = rows
                .iter()
                .filter(|row| row["accepted"] == true)
                .map(|row| row["dropped"].as_u64().unwrap())
                .max()
                .unwrap();
            // Exact failure does not imply approximate failure: separately exclude
            // every lower-depth configuration that could improve the incumbent.
            // Depth <= best+1 cannot retain even one tower while dropping more.
            let lower_candidates = (5.max(best as usize + 2)..minimum)
                .flat_map(|depth| ((best as usize + 1)..depth).map(move |drop| (depth, drop)))
                .collect::<Vec<_>>();
            let mut lower_rows = Vec::new();
            for chunk in lower_candidates.chunks(4) {
                lower_rows.extend(
                    chunk
                        .par_iter()
                        .map(|&(depth, drop)| evaluate(depth, drop))
                        .collect::<Vec<_>>(),
                );
            }
            assert!(
                baseline
                    .iter()
                    .chain(&rows)
                    .chain(&lower_rows)
                    .all(|row| row.get("preparation_error").is_none()),
                "unexpected preparation errors require explicit audit, not rejection by default"
            );
            let best = rows
                .iter()
                .chain(&lower_rows)
                .filter(|row| row["accepted"] == true)
                .map(|row| row["dropped"].as_u64().unwrap())
                .max()
                .unwrap();
            let winners = rows
                .iter()
                .chain(&lower_rows)
                .filter(|row| row["accepted"] == true && row["dropped"] == best)
                .map(|row| serde_json::json!({"depth":row["depth"],"dropped":row["dropped"]}))
                .collect::<Vec<_>>();
            fs::write(directory.join("search.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "scope":"Current universal approximate production model; fixed N65536, CRT50/base25, dm38/df1, five-tower exact decoder; conventional security envelope through depth33; not a joint-security proof",
        "minimum_exact_depth":minimum,"maximum_dropped":best,"winners":winners,
        "baseline":baseline,"window":rows,"lower_depth_exclusions":lower_rows
    })).unwrap()).unwrap();
            eprintln!(
                "DEPTH_WINDOW_RESULT minimum_exact_depth={minimum} maximum_dropped={best} winners={winners:?}"
            );
        }

        #[test]
        fn universal_approximate_complete_lower_depths() {
            use rayon::prelude::*;
            let mut config = SearchConfig::reviewed();
            config.noise_model = NoiseSearchModel::AverageCase;
            config.sparse_lwr_universe = 8192;
            config.sparse_lwr_weight = 18;
            config.sparse_lwr_modulus = 4096;
            config.sparse_lwr_output_modulus = 32;
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/universal_approximate_depth_window");
            let directory = fs::read_dir(root)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.join("search.json").is_file())
                .max()
                .expect("completed first-stage sweep");
            let mut search: serde_json::Value =
                serde_json::from_slice(&fs::read(directory.join("search.json")).unwrap()).unwrap();
            assert!(
                search["lower_depth_exclusions"].is_array(),
                "first-stage maximum proof must be complete"
            );
            let minimum = search["minimum_exact_depth"].as_u64().unwrap() as usize;
            let best = search["maximum_dropped"].as_u64().unwrap() as usize;
            let candidates = (5..minimum)
                .flat_map(|depth| (1..=best.min(depth - 1)).map(move |drop| (depth, drop)))
                .collect::<Vec<_>>();
            let mut rows = Vec::new();
            for chunk in candidates.chunks(4) {
                rows.extend(
                    chunk
                        .par_iter()
                        .map(|&(depth, drop)| {
                            evaluate_universal_depth_case(&config, &directory, depth, drop)
                        })
                        .collect::<Vec<_>>(),
                );
            }
            assert!(rows.iter().all(|row| row.get("preparation_error").is_none()));
            let accepted = rows.iter().filter(|row| row["accepted"] == true).count();
            search["lower_depth_small_drops"] = serde_json::Value::Array(rows);
            fs::write(
                directory.join("complete-search.json"),
                serde_json::to_vec_pretty(&search).unwrap(),
            )
            .unwrap();
            eprintln!(
                "DEPTH_WINDOW_SUPPLEMENT directory={} cases={} accepted={accepted}",
                directory.display(),
                candidates.len()
            );
        }

        #[test]
        fn universal_approximate_selected_noise() {
            export_selected_noise_and_graph(
                33,
                7,
                102,
                "universal_approximate_selected",
                "test_data/approximate_gadget_maximum_drop/1788656642909207371/mxx-approximate-n16-exact-security-cap.log",
            );
        }

        #[test]
        fn exact_minimum_depth_selected_noise() {
            export_selected_noise_and_graph(
                26,
                0,
                143,
                "exact_minimum_depth_selected",
                "test_data/universal_approximate_depth_window/1788660855725121579/security-depth26.log",
            );
        }

        fn export_selected_noise_and_graph(
            depth: usize,
            dropped: usize,
            security: u64,
            output_name: &str,
            security_evidence: &str,
        ) {
            let mut config = SearchConfig::reviewed();
            config.noise_model = NoiseSearchModel::AverageCase;
            config.sparse_lwr_universe = 8192;
            config.sparse_lwr_weight = 18;
            config.sparse_lwr_modulus = 4096;
            config.sparse_lwr_output_modulus = 32;
            config.gadget_dropped_moduli = dropped;
            let candidate =
                Candidate { crt_depth: depth, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
            let mut context = build_candidate_context(&config, candidate).unwrap();
            context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, 5).unwrap();
            let prepared =
                prepare_candidate_from_ring_context(&config, &context, Some((38, 1)), false)
                    .unwrap();
            let snapshot = prepared.noise_snapshot.as_ref().unwrap();
            let average =
                snapshot.simulate_average(&average_case_config(&config).unwrap()).unwrap();
            let directory =
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data").join(output_name);
            fs::create_dir_all(&directory).unwrap();
            fs::write(directory.join("noise.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "scope":"Universal approximate gadget, exact rectangular trapdoor preimages, even-corrected routes; AverageCase analytical acceptance only",
        "candidate":candidate,"gadget_dropped_moduli":dropped,"gadget_columns":4*(depth-dropped),
        "crt_moduli":context.crt_moduli,"switch_moduli":context.switch_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "route_bias_bounds":snapshot.refresh.route_bias_bounds.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "average":average,"setup_identity":hex(*snapshot.setup_identity()),
        "layout_id":hex(prepared.layout_id),"program_id":hex(prepared.program_id),
    })).unwrap()).unwrap();
            eprintln!(
                "UNIVERSAL_APPROXIMATE_RESULT {} accepted={} max_F={}",
                directory.display(),
                average.accepted,
                average.refresh.max_f_avg
            );
            assert_eq!(context.sampler_layout.public_key_columns(), 4 * (depth - dropped));
            assert_eq!(snapshot.inputs.operation_residuals.is_empty(), dropped == 0);
            if dropped == 0 {
                assert!(snapshot.readout_residuals.is_none());
                assert!(snapshot.refresh.route_bias_bounds.iter().all(BigUint::is_zero));
            }
            assert!(
                average.accepted,
                "new universal approximate noise model must pass independently"
            );
            fs::write(directory.join("benchmark-input.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "security_scope":"Conventional full-q RLWE rough estimate and pinned sparse LWR 103.186; correlated joint-view security not established",
        "security_evidence":security_evidence,
        "result":{
            "candidate":candidate,"gadget_dropped_moduli":dropped,"achieved_security_bits":security.min(103),
            "bgg_security_bits":security,
            "sparse_lwr_universe":8192,"sparse_lwr_weight":18,"sparse_lwr_q_l":4096,"sparse_lwr_p":32,
            "layout_id":hex(prepared.layout_id),"program_id":hex(prepared.program_id),
            "average_evidence":{
                "setup_identity":hex(*snapshot.setup_identity()),
                "switch_moduli":context.switch_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "mask_digit_count":38,"fresh_error_digit_count":1,
                "hard_authority_accepted":average.hard_authority_accepted,
                "correctness_accepted":average.correctness_accepted,"accepted":average.accepted
            }
        }
    })).unwrap()).unwrap();
            let graph_prepared = prepare_candidate_with_fixed_digits(
                &config,
                candidate,
                38,
                1,
                &context.switch_moduli,
            )
            .unwrap();
            let bundle = graph_prepared.bundle.as_ref().unwrap();
            assert_eq!(bundle.public_identity(), snapshot.setup_identity());
            let preprocessing = bundle.benchmark_preprocessing_graph();
            preprocessing
                .validate_with_manifests(
                    &mxx_ir_core::ParamEnv::default(),
                    bundle.preprocessing_manifests(),
                )
                .unwrap();
            bundle
                .benchmark_online_graph()
                .validate_with_manifests(
                    &mxx_ir_core::ParamEnv::default(),
                    bundle.verification_manifests(),
                )
                .unwrap();
            eprintln!(
                "UNIVERSAL_APPROXIMATE_GRAPH validated short ordinary gadget and exact rectangular decoder graph"
            );
        }

        #[test]
        fn profile_b_approximate_gadget_noise_simulation() {
            approximate_gadget_noise_cases(27, 16, 38, 5, false, None);
        }

        #[test]
        fn approximate_gadget_maximum_drop_pilot() {
            let retained = (1..=33usize).collect::<Vec<_>>();
            for digits in [38usize, 39] {
                let rows = approximate_gadget_noise_cases(33, 16, digits, 5, true, Some(&retained));
                let selected =
                    rows.iter().find(|row| row["modeled_refresh_accepted"] == true).unwrap();
                eprintln!(
                    "MAXIMUM_DROP_PILOT dm={digits} dropped={} retained={} F_avg={}",
                    selected["dropped_towers"],
                    selected["retained_towers"],
                    selected["conditional_refresh_max_F_avg"]
                );
            }
        }

        #[test]
        fn approximate_gadget_maximum_drop_frontier() {
            let mut config = SearchConfig::reviewed();
            config.noise_model = NoiseSearchModel::AverageCase;
            config.sparse_lwr_universe = 8192;
            config.sparse_lwr_weight = 18;
            config.sparse_lwr_modulus = 4096;
            config.sparse_lwr_output_modulus = 32;
            let policy = average_case_config(&config).unwrap();
            let run = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/approximate_gadget_maximum_drop")
                .join(run.to_string());
            fs::create_dir_all(&directory).unwrap();
            eprintln!("MAXIMUM_DROP_DIRECTORY {}", directory.display());
            let mut records = Vec::new();
            let mut unresolved = Vec::new();
            let max_mask = BigUint::from(32u8).pow(39);
            let ceil_div = |n: BigUint, d: &BigUint| (n + d - 1u8) / d;
            // Incumbent k=7 was independently found at depth33. Depth<=8 cannot
            // beat it since at least one tower must remain. This is a count bound,
            // not a monotonicity assumption about the noise model.
            for depth in 9..=33usize {
                let candidate = Candidate {
                    crt_depth: depth,
                    log_ring_dimension: 16,
                    crt_bits: 50,
                    base_bits: 25,
                };
                let mut context = build_candidate_context(&config, candidate).unwrap();
                context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, 5).unwrap();
                let setup = setup_for_candidate_context(&config, &context, 3, 1).unwrap();
                let prepared =
                    prepare_candidate_from_ring_context(&config, &context, Some((3, 1)), false)
                        .unwrap();
                let baseline = prepared.noise_snapshot.unwrap();
                for dropped in 8..depth {
                    let retained = depth - dropped;
                    let (snapshot, pre_refresh, _, _, _, _, _, _) =
                        modeled_approximate_pre_refresh(&baseline, &context, retained, true);
                    let average = snapshot
                        .simulate_average_with_pre_refresh(&policy, pre_refresh.clone())
                        .unwrap();
                    let routes = snapshot.refresh.route_gains.as_ref().unwrap();
                    let smudging = snapshot.refresh.smudging.as_ref().unwrap();
                    let smudge_factor = (BigUint::one() << (smudging.lambda_mask + 1)) *
                        BigUint::from(smudging.transcript_coordinates);
                    // Drop every nonnegative mask error contribution, use the
                    // smallest joint-tail event count (dm3) and largest M (dm39).
                    // This is optimistic for every admissible mask digit count.
                    let witness=average.refresh.slots.iter().find_map(|slot| {
                let variance=slot.variance_breakdown.readout_fresh.add(&slot.variance_breakdown.state).add(&slot.variance_breakdown.decoder);
                let lower=mxx_exponent_lut::noise_refresh::simulation::average_favg(&average.refresh.z_squared,&variance,0).unwrap()
                    +ceil_div(slot.rounding_numerator.clone(),&slot.rounding_denominator)
                    +ceil_div(&pre_refresh.readout_bias_bound*&routes.fresh[slot.slot].l1,&slot.lift);
                let smudge_lhs=&lower*&smudge_factor;
                let round_lhs=&lower*2u8;
                (smudge_lhs>max_mask || round_lhs>=slot.spacing).then(||serde_json::json!({"slot":slot.slot,"F_lower":lower.to_string(),"smudging_lhs":smudge_lhs.to_string(),"M_max":max_mask.to_string(),"rounding_lhs":round_lhs.to_string(),"spacing":slot.spacing.to_string()}))
            });
                    if let Some(witness) = witness {
                        records.push(serde_json::json!({"depth":depth,"dropped":dropped,"status":"optimistic_lower_bound_rejects_all_dm3_to39","witness":witness}));
                    } else {
                        let mut accepted = None;
                        for digits in 3..=39 {
                            let gains = BggRefreshParameters::mask_route_gain_for_digit_count(
                                &setup, digits,
                            )
                            .unwrap();
                            let changed =
                                snapshot_with_mask_gain(&snapshot, digits, &gains).unwrap();
                            if changed
                                .average_refresh_accepted_from_pre_refresh(&policy, &pre_refresh)
                                .unwrap()
                            {
                                accepted = Some(digits);
                                break;
                            }
                        }
                        unresolved.push(
                    serde_json::json!({"depth":depth,"dropped":dropped,"accepted_digits":accepted}),
                );
                        records.push(serde_json::json!({"depth":depth,"dropped":dropped,"status":"exhaustive_dm3_to39","accepted_digits":accepted}));
                    }
                }
                fs::write(
                    directory.join("frontier.json"),
                    serde_json::to_vec_pretty(&records).unwrap(),
                )
                .unwrap();
                eprintln!(
                    "MAXIMUM_DROP_DEPTH depth={depth} checked={} unresolved={}",
                    records.len(),
                    unresolved.len()
                );
            }
            assert!(
                unresolved.iter().all(|row| row["accepted_digits"].is_null()),
                "incumbent exceeded; inspect persisted grid"
            );
            fs::write(directory.join("search.json"),serde_json::to_vec_pretty(&serde_json::json!({"ring_dimension":65536,"security_depth_cap":33,"incumbent_dropped":7,"switch_width":5,"mask_digit_range":[3,39],"checked":records.len(),"exhaustive_fallback":unresolved,"records":records,"scope":"Maximum absolute dropped count within fixed50/25, fixed five-tower switching model and security-qualified depths; explicit analytical even-residual bridge, not runtime or joint-security proof"})).unwrap()).unwrap();
            eprintln!("MAXIMUM_DROP_COMPLETE {}", directory.display());
        }

        #[test]
        fn approximate_gadget_depth_search() {
            let config = SearchConfig::from_env().unwrap();
            let log_ring = *config.log_ring_dimensions.start();
            let started = std::time::Instant::now();
            let mut selected = [None, None];
            let mut rows = Vec::new();
            for depth in 27..=90 {
                for mask_digits in [38usize, 39] {
                    let cases =
                        approximate_gadget_noise_cases(depth, log_ring, mask_digits, 5, true, None);
                    for (ratio, case) in
                        [cases[1].clone(), cases[2].clone()].into_iter().enumerate()
                    {
                        if selected[ratio].is_none() && case["modeled_refresh_accepted"] == true {
                            selected[ratio] = Some(case.clone());
                            eprintln!(
                                "APPROXIMATE_DEPTH_FIRST_PASS ratio={ratio} depth={depth} dm={mask_digits} elapsed={:.3}",
                                started.elapsed().as_secs_f64()
                            );
                        }
                        rows.push(case);
                    }
                }
                if selected.iter().all(Option::is_some) {
                    break;
                }
            }
            let run = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/approximate_gadget_depth_search")
                .join(run.to_string());
            fs::create_dir_all(&directory).unwrap();
            fs::write(directory.join("search.json"),serde_json::to_vec_pretty(&serde_json::json!({"log_ring_dimension":log_ring,"selected":selected,"cases":rows,"elapsed_seconds":started.elapsed().as_secs_f64(),"scope":"AverageCase noise model with explicit even-residual bridge; not runtime or security acceptance"})).unwrap()).unwrap();
            eprintln!("APPROXIMATE_DEPTH_COMPLETE {}", directory.display());
        }

        fn modeled_approximate_pre_refresh(
            baseline: &BggNoiseSnapshot,
            context: &CandidatePreparationContext,
            retained: usize,
            include_bridge: bool,
        ) -> (
            BggNoiseSnapshot,
            PreRefreshNoiseEvaluation<ProgramNoiseReport>,
            BigUint,
            BigUint,
            mxx_exponent_lut::bgg::noise::ProgramNoiseState,
            RightActionNoiseGain,
            BigUint,
            BigUint,
        ) {
            use mxx_exponent_lut::bgg::noise::{
                GateNoiseDecomposition, GateNoiseOperation, NoiseDecompositionPlan,
                NoiseGainPlanBuilder, ProgramNoiseState,
            };
            let depth = context.candidate.crt_depth;
            let ring = context.ring_dimension;
            let mut snapshot = baseline.clone();
            let mut inputs = snapshot.inputs.clone();
            let dropped = depth - retained;
            let params = DCRTPolyParams::new(
                ring as u32,
                depth,
                50,
                25,
                Some(context.crt_moduli.clone()),
                Some(dropped),
            );
            let epsilon = params.gadget_error_bound(None);
            let low: BigUint =
                context.crt_moduli[retained..].iter().map(|p| BigUint::from(*p)).product();
            // Proposition 1 residual is a sum of centered dropped CRT residues.
            // Uniform, independent residues and augmented ternary secret are the
            // explicit AverageCase assumptions. The stored convention is 4 Var.
            let residual_sum =
                context.crt_moduli[retained..].iter().fold(BigUint::zero(), |sum, p| {
                    let p = BigUint::from(*p);
                    sum + (&low / &p).pow(2) * (p.pow(2) - 1u8)
                });
            let residual = ProgramNoiseState {
                bound: &epsilon * BigUint::from(ring + 1),
                doubled_variance: NonnegativeRational::new(
                    residual_sum * BigUint::from(3 + 2 * ring),
                    9u8.into(),
                )
                .unwrap(),
            };
            if dropped != 0 {
                // Every retained balanced digit is bounded by beta/2. Retain the
                // true secret, digit, and convolution axes instead of scaling a
                // random-matrix variance without an operator bound.
                let rows = 2 * 2 * retained;
                let cap = BigUint::one() << 24usize;
                let direct = RightActionNoiseGain {
                    l1: &cap * BigUint::from(rows * ring),
                    l2_squared: cap.pow(2) * BigUint::from(rows * ring),
                };
                let mut gates = snapshot.gate_gains.clone();
                for gate in &mut gates {
                    if gate.operation == GateNoiseOperation::Add {
                        continue;
                    }
                    gate.action = direct.clone();
                    let multiplicity = if gate.operation == GateNoiseOperation::Lut {
                        gate.helper_action = Some(direct.clone());
                        2u8
                    } else {
                        1u8
                    };
                    inputs.operation_residuals.insert(
                        gate.operation_index,
                        ProgramNoiseState {
                            bound: &residual.bound * multiplicity,
                            doubled_variance: residual
                                .doubled_variance
                                .scaled(&BigUint::from(multiplicity)),
                        },
                    );
                }
                snapshot.gate_gains = gates.clone();
                snapshot.readout_gains.target = direct.clone();
                snapshot.readout_gains.normalization = direct.clone();
                for trace in &mut snapshot.readout_gains.nonidentity_trace {
                    trace.action = direct.clone();
                    trace.helper_action = direct.clone();
                }
                let plan = NoiseDecompositionPlan::bounded_matrix(
                    BigUint::one() << 25usize,
                    rows,
                    rows,
                    ring,
                    cap,
                )
                .unwrap();
                let add_plan = NoiseDecompositionPlan::unit(
                    BigUint::one() << 25usize,
                    2 * depth,
                    2,
                    4 * depth,
                    ring,
                    0,
                    0,
                    0,
                )
                .unwrap()
                .bind_crt_layout(context.crt_moduli.iter().map(|p| BigUint::from(*p)).collect(), 2)
                .unwrap();
                snapshot.gain_plan = NoiseGainPlanBuilder::new(
                    gates
                        .iter()
                        .map(|gate| GateNoiseDecomposition {
                            action: if gate.operation == GateNoiseOperation::Add {
                                add_plan.clone()
                            } else {
                                plan.clone()
                            },
                            helper_action: gate.helper_action.as_ref().map(|_| plan.clone()),
                        })
                        .collect(),
                    plan.clone(),
                    vec![
                        (plan.clone(), plan.clone());
                        snapshot.readout_gains.nonidentity_trace.len()
                    ],
                    plan,
                )
                .unwrap()
                .build(gates, snapshot.readout_gains.clone())
                .unwrap();
                let zero = ProgramNoiseState {
                    bound: BigUint::zero(),
                    doubled_variance: NonnegativeRational::zero(),
                };
                let trace_residual = ProgramNoiseState {
                    bound: &residual.bound * (&snapshot.target_plaintext_gain().l1 + 1u8),
                    doubled_variance: residual
                        .doubled_variance
                        .scaled(&(&snapshot.target_plaintext_gain().l2_squared + 1u8)),
                };
                // Scalar multiples of the shortened gadget have zero discarded
                // residual; the two generic trace actions do not.
                snapshot.readout_residuals = Some((
                    zero.clone(),
                    vec![trace_residual; snapshot.readout_gains.nonidentity_trace.len()],
                    zero,
                ));
            }
            let mut pre_refresh = snapshot.evaluate_pre_refresh_with_inputs(&inputs).unwrap();
            let (bridge_gain, bridge_residual) =
                mxx_exponent_lut::bgg::noise::even_residual_gadget_bridge(
                    &context.crt_moduli,
                    retained,
                    &(BigUint::one() << 25usize),
                    2,
                )
                .unwrap();
            let bridge_bias = &bridge_residual * 31u8 / 2u8;
            if include_bridge {
                pre_refresh.apply_gadget_bridge(&bridge_gain, &bridge_bias);
            }
            (
                snapshot,
                pre_refresh,
                epsilon,
                low,
                residual,
                bridge_gain,
                bridge_residual,
                bridge_bias,
            )
        }

        fn approximate_gadget_noise_cases(
            depth: usize,
            log_ring: usize,
            mask_digits: usize,
            switch_width: usize,
            include_bridge: bool,
            retained_override: Option<&[usize]>,
        ) -> Vec<serde_json::Value> {
            let mut config = SearchConfig::reviewed();
            config.noise_model = NoiseSearchModel::AverageCase;
            config.sparse_lwr_universe = 8192;
            config.sparse_lwr_weight = 18;
            config.sparse_lwr_modulus = 4096;
            config.sparse_lwr_output_modulus = 32;
            assert_eq!(config.secret_dimension, 2);
            let candidate = Candidate {
                crt_depth: depth,
                log_ring_dimension: log_ring,
                crt_bits: 50,
                base_bits: 25,
            };
            let ring = 1usize << log_ring;
            let mut context = build_candidate_context(&config, candidate).unwrap();
            context.switch_moduli =
                candidate_switch_moduli(&context.crt_moduli, switch_width).unwrap();
            let prepared = prepare_candidate_from_ring_context(
                &config,
                &context,
                Some((mask_digits, 1)),
                false,
            )
            .unwrap();
            let baseline = prepared.noise_snapshot.unwrap();
            let run = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/profile_b_approximate_gadget_noise")
                .join(run.to_string());
            fs::create_dir_all(&directory).unwrap();
            eprintln!("APPROXIMATE_NOISE_DIRECTORY {}", directory.display());
            let mut rows = Vec::new();
            let retained_cases =
                retained_override.map(|values| values.to_vec()).unwrap_or_else(|| {
                    vec![depth, (2 * depth).div_ceil(3), depth.div_ceil(2), depth / 2]
                });
            for retained in retained_cases {
                let dropped = depth - retained;
                let (
                    snapshot,
                    pre_refresh,
                    epsilon,
                    low,
                    residual,
                    bridge_gain,
                    bridge_residual,
                    bridge_bias,
                ) = modeled_approximate_pre_refresh(&baseline, &context, retained, include_bridge);
                let average = snapshot
                    .simulate_average_with_pre_refresh(
                        &average_case_config(&config).unwrap(),
                        pre_refresh.clone(),
                    )
                    .unwrap();
                let program_variance = &pre_refresh.program.output_state().doubled_variance;
                let program_favg = mxx_exponent_lut::noise_refresh::simulation::average_favg(
                    &average.refresh.z_squared,
                    program_variance,
                    0,
                )
                .unwrap();
                let readout_stochastic_favg =
                    mxx_exponent_lut::noise_refresh::simulation::average_favg(
                        &average.refresh.z_squared,
                        &pre_refresh.readout_variance,
                        0,
                    )
                    .unwrap();
                let readout_favg = &readout_stochastic_favg + &pre_refresh.readout_bias_bound;
                let row = serde_json::json!({
                    "depth":depth,"ring_dimension":ring,"mask_digits":mask_digits,"switch_width":switch_width,
                    "retained_towers":retained,"dropped_towers":dropped,"gadget_columns":4*retained,
                    "full_modulus":snapshot.refresh.global_modulus.as_ref().unwrap().to_string(),
                    "crt_moduli":context.crt_moduli,"low_modulus":low.to_string(),
                    "coefficient_residual_bound":epsilon.to_string(),"secret_weighted_residual":residual,
                    "program_bound":pre_refresh.program.output_state().bound.to_string(),
                    "program_bound_bits":pre_refresh.program.output_state().bound.bits(),
                    "readout_bound":pre_refresh.readout_bound.to_string(),"readout_bound_bits":pre_refresh.readout_bound.bits(),
                    "readout_doubled_variance":pre_refresh.readout_variance,
                    "bridge_included":include_bridge,"bridge_gain":bridge_gain,"bridge_residual":bridge_residual.to_string(),"bridge_bias":bridge_bias.to_string(),
                    "modeled_refresh_accepted":if include_bridge {Some(average.refresh.accepted)}else{None},
                    "program_F_avg":program_favg.to_string(),"program_F_avg_bits":program_favg.bits(),
                    "readout_F_avg":readout_favg.to_string(),"readout_F_avg_bits":readout_favg.bits(),
                    "readout_stochastic_F_avg":readout_stochastic_favg.to_string(),"readout_bias_bound":pre_refresh.readout_bias_bound.to_string(),
                    "joint_event_count":average.refresh.joint_event_count.to_string(),"z_squared":average.refresh.z_squared,
                    "conditional_refresh_F_avg_bits":average.refresh.max_f_avg.bits(),
                    "conditional_minimum_rounding_margin":average.refresh.slots.iter().map(|s| &s.rounding_margin).min().unwrap().to_string(),
                    "conditional_minimum_smudging_margin":average.refresh.slots.iter().map(|s| &s.smudging_margin).min().unwrap().to_string(),
                    "baseline_first_action":baseline.gate_gains.first().map(|g|&g.action),
                    "modeled_first_action":snapshot.gate_gains.first().map(|g|&g.action),
                    "conditional_refresh_max_F_avg":average.refresh.max_f_avg.to_string(),
                    "conditional_refresh_accepted":average.refresh.accepted,
                    "full_approximate_refresh_accepted":if dropped==0 { Some(average.refresh.accepted) } else { None },
                    "scope":if dropped==0 { "exact baseline" } else if include_bridge { "Approximate gadget noise model with explicit even-residual scalar bridge; not runtime or security acceptance" } else { "shortened-gadget PRF/readout model; full-width refresh figures conditional on a zero-error bridge, not executable acceptance" },
                    "program_steps":pre_refresh.program.steps,
                });
                fs::write(
                    directory.join(format!("retained-{retained}.json")),
                    serde_json::to_vec_pretty(&row).unwrap(),
                )
                .unwrap();
                eprintln!(
                    "APPROXIMATE_NOISE depth={depth} logN={log_ring} dm={mask_digits} retained={retained} columns={} program_average_bits={} readout_average_bits={} conditional_F_avg={}",
                    4 * retained,
                    program_favg.bits(),
                    readout_favg.bits(),
                    average.refresh.max_f_avg
                );
                rows.push(row);
            }
            fs::write(directory.join("summary.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "parameters":{"N":ring,"crt_depth":depth,"crt_bits":50,"base_bits":25,"security_bits":100,"v":8192,"h":18,"Q":4096,"p":32,"mask_digits":mask_digits,"fresh_digits":1,"refresh_switch_retained_towers":switch_width,"secret_dimension":2},
        "assumptions":"Uniform independent centered dropped CRT residues, augmented iid ternary secret, independent residual sources and TFHE-style propagation; stored variance is four times physical variance. See bridge_included per case: the optional custom even-residual bridge has deterministic bias propagated with l1 gains. Neither case establishes runtime or security acceptance.",
        "cases":rows,
    })).unwrap()).unwrap();
            rows
        }

        #[test]
        fn test_round_trip_artifact_inventory() {
            let config = SearchConfig::from_env().unwrap();
            let report: serde_json::Value = serde_json::from_slice(
                &std::fs::read(std::env::var("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT").unwrap())
                    .unwrap(),
            )
            .unwrap();
            let result = &report["result"];
            let candidate = serde_json::from_value(result["candidate"].clone()).unwrap();
            let evidence = &result["average_evidence"];
            let (bundle, _, _) = round_trip_bundle(
                &config,
                candidate,
                evidence["mask_digit_count"].as_u64().unwrap() as usize,
                evidence["fresh_error_digit_count"].as_u64().unwrap() as usize,
                1,
            )
            .unwrap();
            for manifest in bundle.preprocessing_manifests().values() {
                println!("artifact_inventory={}", serde_json::to_string(manifest).unwrap());
            }
        }

        fn reviewed_estimator_bits(universe: usize, q_l: usize) -> u64 {
            match (universe, q_l) {
                (16384, 2048) => 100,
                (16384, 4096) => 100,
                (8192, 4096) => 103,
                other => panic!("unexpected reviewed tuple {other:?}"),
            }
        }

        fn mock_prepared(candidate: Candidate) -> PreparedCandidate {
            PreparedCandidate {
                anchor_security_inputs: Vec::new(),
                candidate,
                ring_dimension: 1 << candidate.log_ring_dimension,
                bucket_width: 3,
                official_preimage_bound: 10.into(),
                selected_preimage_bound: 10.into(),
                layout_id: [candidate.crt_depth as u8; 32],
                program_id: [candidate.log_ring_dimension as u8; 32],
                pbc_attempts_used: 1,
                mask_base_p_digit_count: 1,
                fresh_error_base_p_digit_count: 1,
                prf_component_count: 2,
                prf_coefficient_count: 1 << candidate.log_ring_dimension,
                prf_active_count: 1,
                prf_label_count: 1,
                prf_value_count: 1,
                bundle: None,
                noise_snapshot: None,
                pre_refresh: None,
                production_report: None,
                average_report: None,
            }
        }

        fn mock_sparse_profile(config: &SearchConfig) -> SelectedSparseLwrProfile {
            let tuple = config.sparse_lwr_phase1_grid[0].clone();
            let (error_lower, error_upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p).unwrap();
            SelectedSparseLwrProfile {
                tuple: tuple.clone(),
                parameter_grid: config.sparse_lwr_phase1_grid.clone(),
                q_l: config.sparse_lwr_modulus,
                p: config.sparse_lwr_output_modulus,
                universe: config.sparse_lwr_universe,
                weight: config.sparse_lwr_weight,
                tier: SparseLwrSecurityTier::Requested,
                error_lower,
                error_upper,
                estimator_commit: config.phase1_estimator_commit.clone(),
                estimator_cost_model: config.phase1_estimator_cost_model.clone(),
                estimator_shape_model: config.phase1_estimator_shape_model.clone(),
                sparse_lwr_security_bits: 200,
                raw_key_entropy_bits: raw_key_entropy_bits(
                    config.sparse_lwr_universe,
                    config.sparse_lwr_weight,
                ),
                evaluations: Vec::new(),
            }
        }

        #[test]
        fn grid_is_lexicographic_depth_then_ring_dimension_when_base_is_fixed() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 2..=3;
            config.log_ring_dimensions = 5..=6;
            config.crt_base_bits_grid = vec![(32, 16)];
            assert_eq!(
                candidates(&config).collect::<Vec<_>>(),
                vec![
                    Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                    Candidate { crt_depth: 2, log_ring_dimension: 6, crt_bits: 32, base_bits: 16 },
                    Candidate { crt_depth: 3, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                    Candidate { crt_depth: 3, log_ring_dimension: 6, crt_bits: 32, base_bits: 16 },
                ]
            );
        }

        #[test]
        fn reviewed_grid_declares_phase_two_minimality_scope() {
            let config = SearchConfig::reviewed();
            assert_eq!(config.crt_depths, 1..=62);
            assert_eq!(config.log_ring_dimensions, 16..=16);
            assert_eq!(config.crt_bits, 50);
            assert_eq!(config.base_bits, 25);
            assert_eq!(config.crt_base_bits_grid, vec![(50, 25)]);
            assert_eq!(config.crt_base_bits_grid[0], (50, 25));
        }

        #[test]
        fn crt_base_grid_descends_base_width_for_each_crt_width() {
            let grid = descending_crt_base_bits_grid(6);
            assert_eq!(grid, vec![(6, 3), (6, 2), (6, 1)]);
        }

        #[test]
        fn phase_two_candidates_keep_depth_ring_order_and_descend_base_bits() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 1..=1;
            config.log_ring_dimensions = 5..=5;
            config.crt_base_bits_grid = vec![(5, 2), (4, 2), (5, 1), (4, 1)];
            assert_eq!(
                candidates(&config).collect::<Vec<_>>(),
                vec![
                    Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 5, base_bits: 2 },
                    Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 4, base_bits: 2 },
                    Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 5, base_bits: 1 },
                    Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 4, base_bits: 1 },
                ]
            );
        }

        #[test]
        fn phase_two_candidates_exhaust_largest_base_before_smaller_base() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 1..=2;
            config.log_ring_dimensions = 5..=5;
            config.crt_base_bits_grid = vec![(32, 16), (30, 15), (32, 15)];
            let candidates = candidates(&config).collect::<Vec<_>>();
            assert_eq!(
                candidates,
                vec![
                    Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                    Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 30, base_bits: 15 },
                    Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 15 },
                    Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                    Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 30, base_bits: 15 },
                    Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 15 },
                ]
            );
        }

        #[test]
        fn average_policy_is_derived_once_from_search_config() {
            let mut config = SearchConfig::reviewed();
            config.security_bits = 37;
            let policy = average_case_config(&config).unwrap();
            assert_eq!(policy.failure_exponent, 37);
            assert!(policy.allow_average_acceptance);
            assert_eq!(policy, average_case_config(&config).unwrap());
        }

        #[test]
        fn average_boundary_enumerates_only_strictly_valid_doubled_masks() {
            let base = BigUint::from(2u8);
            let spacing = BigUint::from(6u8);
            let valid = (1..=4)
                .map(|digits| (digits, base.pow(digits) - BigUint::one()))
                .take_while(|(_, d2)| d2 * 2u8 < spacing.clone() * 2u8)
                .map(|(digits, _)| digits)
                .collect::<Vec<_>>();
            assert_eq!(valid, vec![1, 2]);
            assert!(base.pow(3) - BigUint::one() >= spacing);
        }

        #[test]
        fn phase_two_does_not_apply_a_hidden_total_modulus_bit_cap() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 62..=62;
            config.log_ring_dimensions = 16..=16;
            config.crt_base_bits_grid = vec![(32, 16), (30, 15), (28, 14)];
            assert_eq!(
                candidates(&config).collect::<Vec<_>>(),
                vec![
                    Candidate {
                        crt_depth: 62,
                        log_ring_dimension: 16,
                        crt_bits: 32,
                        base_bits: 16
                    },
                    Candidate {
                        crt_depth: 62,
                        log_ring_dimension: 16,
                        crt_bits: 30,
                        base_bits: 15
                    },
                    Candidate {
                        crt_depth: 62,
                        log_ring_dimension: 16,
                        crt_bits: 28,
                        base_bits: 14
                    },
                ]
            );
            config.crt_depths = 63..=63;
            config.crt_base_bits_grid = vec![(32, 16)];
            assert_eq!(
                candidates(&config).collect::<Vec<_>>(),
                vec![Candidate {
                    crt_depth: 63,
                    log_ring_dimension: 16,
                    crt_bits: 32,
                    base_bits: 16
                }]
            );
        }

        #[test]
        fn lookup_overlay_applies_the_configured_tiny_profile() {
            let values = std::collections::BTreeMap::from([
                ("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", "1"),
                ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE", "4"),
                ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT", "1"),
                ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS", "4"),
                ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", "2"),
                ("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", "4"),
                ("MXX_EXPONENT_LUT_REFRESH_BASE_BITS", "2"),
                ("MXX_EXPONENT_LUT_REFRESH_MIN_CRT_DEPTH", "1"),
                ("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH", "1"),
                ("MXX_EXPONENT_LUT_REFRESH_MIN_LOG_RING_DIMENSION", "5"),
                ("MXX_EXPONENT_LUT_REFRESH_MAX_LOG_RING_DIMENSION", "5"),
            ]);
            let config = SearchConfig::from_lookup(|name| {
                Ok(values.get(name).map(|value| (*value).to_owned()))
            })
            .unwrap();
            assert_eq!(config.security_bits, 1);
            assert_eq!(config.sparse_lwr_universe_grid, vec![4]);
            assert_eq!(config.sparse_lwr_weight, 1);
            assert_eq!(config.sparse_lwr_modulus, 4);
            assert_eq!(config.sparse_lwr_output_modulus, 2);
            assert_eq!(config.sparse_lwr_phase1_grid.len(), 1);
            assert_eq!(
                (
                    &config.sparse_lwr_phase1_grid[0].nu,
                    &config.sparse_lwr_phase1_grid[0].h,
                    &config.sparse_lwr_phase1_grid[0].q_l,
                    &config.sparse_lwr_phase1_grid[0].p
                ),
                (&4, &1, &4, &2)
            );
            assert_eq!(config.crt_bits, 4);
            assert_eq!(config.base_bits, 2);
            assert_eq!(config.crt_depths, 1..=1);
            assert_eq!(config.log_ring_dimensions, 5..=5);
        }

        #[test]
        fn lookup_overlay_keeps_unset_values_from_reviewed_profile() {
            let config = SearchConfig::from_lookup(|name| {
                Ok((name == "MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS").then(|| "1".to_owned()))
            })
            .unwrap();
            assert_eq!(config.security_bits, 1);
            assert_eq!(config.crt_bits, SearchConfig::reviewed().crt_bits);
            assert_eq!(config.base_bits, SearchConfig::reviewed().base_bits);
            assert_eq!(
                config.sparse_lwr_universe_grid,
                SearchConfig::reviewed().sparse_lwr_universe_grid
            );
        }

        #[test]
        fn lookup_overlay_preserves_an_explicit_crt_base_singleton() {
            let values = std::collections::BTreeMap::from([
                ("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", "50"),
                ("MXX_EXPONENT_LUT_REFRESH_BASE_BITS", "25"),
            ]);
            let config = SearchConfig::from_lookup(|name| {
                Ok(values.get(name).map(|value| (*value).to_owned()))
            })
            .unwrap();
            assert_eq!(config.crt_bits, 50);
            assert_eq!(config.base_bits, 25);
            assert_eq!(config.crt_base_bits_grid, vec![(50, 25)]);
            let mut bounded = config.clone();
            bounded.crt_depths = 1..=2;
            bounded.log_ring_dimensions = 16..=16;
            assert!(
                candidates(&bounded)
                    .all(|candidate| { candidate.crt_bits == 50 && candidate.base_bits == 25 })
            );
        }

        #[test]
        fn lookup_overlay_accepts_reviewed_log_ring_dimension_and_crt_base_pair() {
            let values = std::collections::BTreeMap::from([
                ("MXX_EXPONENT_LUT_REFRESH_MIN_LOG_RING_DIMENSION", "16"),
                ("MXX_EXPONENT_LUT_REFRESH_MAX_LOG_RING_DIMENSION", "16"),
                ("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", "50"),
                ("MXX_EXPONENT_LUT_REFRESH_BASE_BITS", "25"),
            ]);
            let config = SearchConfig::from_lookup(|name| {
                Ok(values.get(name).map(|value| (*value).to_owned()))
            })
            .unwrap();
            assert_eq!(config.log_ring_dimensions, 16..=16);
            assert_eq!(config.crt_base_bits_grid, vec![(50, 25)]);
        }

        #[test]
        fn backend_capability_accepts_the_exact_50_25_singleton() {
            assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 50, 25).is_ok());
            assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 60, 30).is_ok());
            assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 60, 31).is_err());
            assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 61, 25).is_err());
            assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 50, 32).is_err());
        }

        #[test]
        fn reviewed_bgg_envelope_accepts_only_its_bound_domain() {
            let mut config = SearchConfig::reviewed();
            config.bgg_security_envelope = Some("reviewed_monotone_envelope".to_owned());
            let accepted =
                Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
            assert_eq!(reviewed_bgg_security_bits(&config, accepted), Ok(136));
            assert_eq!(bgg_security_authority(&config), "reviewed_monotone_envelope");
            assert_eq!(bgg_security_evidence_source(&config), "reviewed_monotone_envelope:q27:136");

            for candidate in [
                Candidate { crt_depth: 28, ..accepted },
                Candidate { log_ring_dimension: 15, ..accepted },
                Candidate { crt_bits: 49, ..accepted },
                Candidate { base_bits: 24, ..accepted },
            ] {
                assert!(reviewed_bgg_security_bits(&config, candidate).is_err());
            }

            let mut wrong_sigma = config.clone();
            wrong_sigma.error_sigma = 4.000000000000001;
            assert!(reviewed_bgg_security_bits(&wrong_sigma, accepted).is_err());
        }

        #[test]
        fn reviewed_bgg_envelope_requires_explicit_selection() {
            let config = SearchConfig::reviewed();
            let candidate =
                Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
            assert!(reviewed_bgg_security_bits(&config, candidate).is_err());
            let error = SearchConfig::from_lookup(|name| {
                Ok((name == "MXX_EXPONENT_LUT_REFRESH_BGG_SECURITY_ENVELOPE")
                    .then(|| "unreviewed".to_owned()))
            })
            .unwrap_err();
            assert!(error.contains("reviewed_monotone_envelope"));
        }

        #[test]
        fn explicit_reviewed_tuple_reuses_pinned_estimator_evidence() {
            let values = std::collections::BTreeMap::from([
                ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE", "16384"),
                ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT", "17"),
                ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS", "2048"),
                ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", "16"),
            ]);
            let config = SearchConfig::from_lookup(|name| {
                Ok(values.get(name).map(|value| (*value).to_owned()))
            })
            .unwrap();
            let tuple = &config.sparse_lwr_phase1_grid[0];
            assert_eq!(tuple.estimator_security_bits, Some(100));
            assert_eq!(tuple.estimator_minimum_classical_bits, Some(100.5636531151896));
        }

        #[test]
        fn decoder_cutoff_acceptance_is_monotone_from_the_official_bound_in_setup() {
            let baseline = BggRefreshParameters::new(
                [0x91; 32],
                2,
                2,
                4,
                1,
                1,
                32,
                4,
                BggSamplerLayout {
                    modulus: 6.into(),
                    ring_dimension: 4.into(),
                    secret_dimension: 2,
                    digit_count: 2,
                    gadget_base: 2.into(),
                },
                RefreshCompiler {
                    full_modulus: 6.into(),
                    crt_plaintext_moduli: vec![2.into(), 3.into()],
                    reconstruction_coefficients: vec![1.into(), 1.into()],
                },
                1.into(),
                1.into(),
                "cutoff-boundary-test",
            );
            let official = baseline
                .resolve_decoder_preimage_bound()
                .unwrap()
                .evaluate(&mxx_ir_core::ParamEnv::default())
                .unwrap();
            let official = official.to_biguint().unwrap();
            for (offset, accepted) in [(-1_i32, false), (0, true), (1, true)] {
                let selected = if offset < 0 {
                    &official - BigUint::from((-offset) as u32)
                } else {
                    &official + BigUint::from(offset as u32)
                };
                assert_eq!(
                    validate_decoder_preimage_cutoff(&selected, &official).is_ok(),
                    accepted
                );
                if offset < 0 {
                    let policy = TrapdoorPreimageCutoffPolicy::Explicit(selected);
                    assert!(
                        baseline
                            .clone()
                            .with_decoder_preimage_bound(policy)
                            .resolve_decoder_preimage_bound()
                            .is_err()
                    );
                    continue;
                }
                let policy = TrapdoorPreimageCutoffPolicy::Explicit(selected.clone());
                let resolved = baseline
                    .clone()
                    .with_decoder_preimage_bound(policy)
                    .resolve_decoder_preimage_bound()
                    .unwrap()
                    .evaluate(&mxx_ir_core::ParamEnv::default())
                    .unwrap()
                    .to_biguint()
                    .unwrap();
                assert_eq!(resolved, selected);
                if offset == 1 {
                    // This is an actual setup clone carrying Explicit(Default+1),
                    // rather than a detached comparison of two integers.
                    assert!(resolved > official);
                }
            }
        }

        #[test]
        fn lookup_overlay_rejects_a_partial_crt_base_pair() {
            let error = SearchConfig::from_lookup(|name| {
                Ok((name == "MXX_EXPONENT_LUT_REFRESH_CRT_BITS").then(|| "50".to_owned()))
            })
            .unwrap_err();
            assert!(error.contains("supplied together"));
        }

        #[test]
        fn lookup_overlay_rejects_a_partial_native_phase_one_tuple() {
            let error = SearchConfig::from_lookup(|name| {
                Ok((name == "MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE")
                    .then(|| "16384".to_owned()))
            })
            .unwrap_err();
            assert!(error.contains("UNIVERSE, _WEIGHT, _MODULUS, and _OUTPUT_MODULUS"));
        }

        #[test]
        fn lookup_overlay_rejects_invalid_ranges() {
            let invalid_range = std::collections::BTreeMap::from([
                ("MXX_EXPONENT_LUT_REFRESH_MIN_CRT_DEPTH", "2"),
                ("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH", "1"),
            ]);
            let error = SearchConfig::from_lookup(|name| {
                Ok(invalid_range.get(name).map(|value| (*value).to_owned()))
            })
            .unwrap_err();
            assert!(error.contains("CRT depth range"));
        }

        #[test]
        fn lookup_overlay_propagates_non_unicode_lookup_failure() {
            let error = SearchConfig::from_lookup(|name| {
                if name == "MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS" {
                    Err("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS must contain valid UTF-8"
                        .to_owned())
                } else {
                    Ok(None)
                }
            })
            .unwrap_err();
            assert!(error.contains("valid UTF-8"));
        }

        #[test]
        fn mocked_security_and_checker_select_first_minimal_candidate() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 1..=3;
            config.log_ring_dimensions = 5..=5;
            config.crt_base_bits_grid = vec![(32, 16)];
            let sparse_profile = mock_sparse_profile(&config);
            let result = search_with_hooks(
                &config,
                &sparse_profile,
                |candidate| Ok(mock_prepared(candidate)),
                |candidate| Ok(candidate.crt_depth as u64 * 100),
                |prepared| Ok(prepared.candidate.crt_depth >= 2),
            )
            .unwrap();
            assert_eq!(
                result.candidate,
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 }
            );
        }

        #[test]
        fn requested_one_bit_target_controls_both_phases_and_checkpoint_identity() {
            let mut config = SearchConfig::reviewed();
            config.security_bits = 1;
            config.mask_statistical_security_bits = 1;
            config.sparse_lwr_universe = 64;
            config.sparse_lwr_weight = 1;
            config.sparse_lwr_modulus = 8;
            config.sparse_lwr_output_modulus = 4;
            config.sparse_lwr_universe_grid = vec![64];
            config.sparse_lwr_phase1_grid = vec![SparseLwrParameterTuple {
                q_l: 8,
                p: 4,
                nu: 64,
                h: 1,
                estimator_minimum_classical_bits: None,
                estimator_security_bits: None,
            }];
            config.crt_depths = 1..=1;
            config.log_ring_dimensions = 5..=5;
            let selected = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(16)).unwrap();
            assert_eq!(selected.tier, SparseLwrSecurityTier::Requested);
            let checkpoint = Phase1Checkpoint::from_selection(&config, &selected).unwrap();
            checkpoint.validate(&config).unwrap();
            let mut different_target = config.clone();
            different_target.security_bits = 2;
            assert!(checkpoint.validate(&different_target).is_err());
            let result = search_with_hooks(
                &config,
                &selected,
                |c| Ok(mock_prepared(c)),
                |_| Ok(1),
                |_| Ok(true),
            )
            .unwrap();
            assert_eq!(result.achieved_security_bits, 1);
            let report = search_report(&config, &selected, result);
            assert_eq!(report.phase1.security_target_bits, 1);
            assert_eq!(report.phase2.security_target_bits, 1);
            assert!(
                search_with_hooks(
                    &config,
                    &selected,
                    |c| Ok(mock_prepared(c)),
                    |_| Ok(0),
                    |_| Ok(true)
                )
                .is_err()
            );
        }

        #[test]
        fn phase_one_exhausts_ordered_grid_before_selecting_primary_profile() {
            let mut config = SearchConfig::reviewed();
            config.security_bits = 100;
            let mut calls = Vec::new();
            let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
                calls.push(universe);
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            assert_eq!(calls, vec![16384, 16384, 8192]);
            assert_eq!(selected.universe, 16384);
            assert_eq!(selected.q_l, 2048);
            assert_eq!(selected.tier, SparseLwrSecurityTier::Requested);
            assert_eq!(selected.sparse_lwr_security_bits, 100);
        }

        #[test]
        fn phase_one_rejects_an_estimator_result_that_does_not_match_declared_evidence() {
            let config = SearchConfig::reviewed();
            let error = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(128)).unwrap_err();
            assert!(error.contains("expected declared"));
        }

        #[test]
        fn qualified_profiles_preserve_all_primary_rows_for_phase_two_advancement() {
            let config = SearchConfig::reviewed();
            let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            let profiles = qualified_sparse_lwr_profiles(&selected);
            assert_eq!(
                profiles
                    .iter()
                    .map(|profile| (profile.q_l, profile.p, profile.universe))
                    .collect::<Vec<_>>(),
                [(2048, 16, 16384), (4096, 32, 16384), (4096, 32, 8192)]
            );
            assert!(
                profiles.iter().all(|profile| profile.tier == SparseLwrSecurityTier::Requested)
            );
        }

        #[test]
        fn phase_two_caches_sparse_security_and_defers_prepare_until_bgg_floor() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 1..=1;
            config.log_ring_dimensions = 5..=6;
            config.crt_base_bits_grid = vec![(32, 16)];
            let sparse_profile = mock_sparse_profile(&config);
            let mut security_calls = Vec::new();
            let mut prepare_calls = Vec::new();
            let mut checker_calls = Vec::new();
            let result = search_with_hooks(
                &config,
                &sparse_profile,
                |candidate| {
                    prepare_calls.push(candidate);
                    Ok(mock_prepared(candidate))
                },
                |candidate| {
                    security_calls.push(candidate);
                    Ok(if candidate.log_ring_dimension == 5 { 99 } else { 100 })
                },
                |prepared| {
                    checker_calls.push(prepared.candidate);
                    Ok(true)
                },
            )
            .unwrap();
            assert_eq!(security_calls.len(), 2);
            assert_eq!(
                prepare_calls,
                vec![Candidate {
                    crt_depth: 1,
                    log_ring_dimension: 6,
                    crt_bits: 32,
                    base_bits: 16
                }]
            );
            assert_eq!(checker_calls, prepare_calls);
            assert_eq!(result.sparse_lwr_universe, config.sparse_lwr_universe);
            assert_eq!(result.sparse_lwr_weight, config.sparse_lwr_weight);
        }

        #[test]
        fn phase_two_skips_exactly_infeasible_preparation_but_propagates_fatal_errors() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 1..=2;
            config.log_ring_dimensions = 5..=5;
            config.crt_base_bits_grid = vec![(32, 16)];
            let sparse_profile = mock_sparse_profile(&config);
            let mut prepared_candidates = Vec::new();
            let result = search_with_hooks(
                &config,
                &sparse_profile,
                |candidate| {
                    prepared_candidates.push(candidate);
                    if candidate.crt_depth == 1 {
                        Err(CandidatePreparationError::Infeasible(
                            "strict rounding is infeasible".to_owned(),
                        ))
                    } else {
                        Ok(mock_prepared(candidate))
                    }
                },
                |_| Ok(128),
                |_| Ok(true),
            )
            .unwrap();
            assert_eq!(
                prepared_candidates,
                vec![
                    Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                    Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                ]
            );
            assert_eq!(
                result.candidate,
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 }
            );

            let fatal = search_with_hooks(
                &config,
                &sparse_profile,
                |_| Err(CandidatePreparationError::Fatal("invalid setup identity".to_owned())),
                |_| Ok(128),
                |_| Ok(true),
            )
            .unwrap_err();
            assert_eq!(fatal, "invalid setup identity");
        }

        #[test]
        fn phase_one_rejects_a_non_ascending_grid() {
            let mut config = SearchConfig::reviewed();
            config.sparse_lwr_phase1_grid[1] = config.sparse_lwr_phase1_grid[0].clone();
            let error = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(200)).unwrap_err();
            assert!(error.contains("tuple grid"));
        }

        fn checkpoint_test_path(name: &str) -> std::path::PathBuf {
            std::env::temp_dir()
                .join(format!("mxx-exponent-lut-{name}-{}.json", std::process::id()))
        }

        #[test]
        fn phase_one_checkpoint_reuses_an_exact_declaration() {
            let mut config = SearchConfig::reviewed();
            config.sparse_lwr_phase1_grid = config.sparse_lwr_phase1_grid[..2].to_vec();
            let path = checkpoint_test_path("phase1-reuse");
            let _ = std::fs::remove_file(&path);

            let mut first_calls = 0;
            let first = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
                first_calls += 1;
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();

            let mut second_calls = 0;
            let second = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
                second_calls += 1;
                Err("an exact checkpoint must not rerun Phase 1".to_owned())
            })
            .unwrap();
            assert_eq!(first_calls, 2);
            assert_eq!(second_calls, 0);
            assert_eq!(first.universe, second.universe);
            assert_eq!(second.universe, 16384);
            let _ = std::fs::remove_file(path);
        }

        #[test]
        fn phase_one_checkpoint_rejects_mismatch_and_missing_path_starts_fresh_search() {
            let mut config = SearchConfig::reviewed();
            config.sparse_lwr_phase1_grid = config.sparse_lwr_phase1_grid[..2].to_vec();
            let path = checkpoint_test_path("phase1-mismatch");
            let _ = std::fs::remove_file(&path);

            let mut calls = 0;
            let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
                calls += 1;
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            assert_eq!(calls, 2);
            assert_eq!(selected.universe, 16384);

            let mut changed = config.clone();
            changed.phase1_estimator_cost_model = "ALT".to_owned();
            let error = load_or_search_phase1(&changed, Some(&path), |_, _, _, _| {
                Err("a declaration mismatch must not fall back to a fresh search".to_owned())
            })
            .unwrap_err();
            assert!(error.contains("does not match"));
            let _ = std::fs::remove_file(path);
        }

        #[test]
        fn phase_one_checkpoint_binds_tuple_tier_error_and_estimator_identity() {
            let config = SearchConfig::reviewed();
            let path = checkpoint_test_path("phase1-identity");
            let _ = std::fs::remove_file(&path);
            load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            let mut changed = config.clone();
            // Keep the tampered tuple structurally valid so this exercises the
            // checkpoint's source-of-truth identity check rather than native-Q
            // validation itself.
            changed.sparse_lwr_phase1_grid[0].q_l = 1024;
            assert!(
                load_or_search_phase1(&changed, Some(&path), |universe, _, q_l, _| Ok(
                    reviewed_estimator_bits(universe, q_l)
                ))
                .unwrap_err()
                .contains("does not match")
            );
            let mut checkpoint: Phase1Checkpoint =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            checkpoint.selected.tier = SparseLwrSecurityTier::Rejected;
            std::fs::write(&path, serde_json::to_vec_pretty(&checkpoint).unwrap()).unwrap();
            assert!(
                load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| Ok(
                    reviewed_estimator_bits(universe, q_l)
                ))
                .unwrap_err()
                .contains("final row")
            );
            std::fs::remove_file(&path).unwrap();
            load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            let mut changed = config.clone();
            changed.phase1_estimator_shape_model = "primal".to_owned();
            assert!(
                load_or_search_phase1(&changed, Some(&path), |universe, _, q_l, _| Ok(
                    reviewed_estimator_bits(universe, q_l)
                ))
                .unwrap_err()
                .contains("does not match")
            );
            let mut changed = config.clone();
            changed.security_bits = 101;
            assert!(
                load_or_search_phase1(&changed, Some(&path), |_, _, _, _| Ok(128))
                    .unwrap_err()
                    .contains("does not match")
            );
            let _ = std::fs::remove_file(path);
        }

        #[test]
        fn phase_one_checkpoint_round_trips_the_boundary_tuple() {
            for selected_universe in [16384, 8192] {
                let mut config = SearchConfig::reviewed();
                config.sparse_lwr_phase1_grid = config
                    .sparse_lwr_phase1_grid
                    .into_iter()
                    .filter(|tuple| tuple.nu == selected_universe)
                    .collect();
                let path = checkpoint_test_path(&format!("phase1-selected-{selected_universe}"));
                let _ = std::fs::remove_file(&path);
                let selected =
                    load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
                        Ok(reviewed_estimator_bits(universe, q_l))
                    })
                    .unwrap();
                assert_eq!(selected.universe, selected_universe);
                let reused = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
                    Err("an exact checkpoint must not rerun Phase 1".to_owned())
                })
                .unwrap();
                assert_eq!(reused.tuple, selected.tuple);
                let _ = std::fs::remove_file(path);
            }
        }

        #[test]
        fn accepted_phase_two_profile_persists_native_q_and_rejects_out_of_grid_tampering() {
            let config = SearchConfig::reviewed();
            let path = checkpoint_test_path("phase2-accepted");
            let _ = std::fs::remove_file(&path);
            let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            let mut order = Vec::new();
            let (selected_profile, result) = search_qualified_profiles(&selected, |profile| {
        order.push(profile.q_l);
        Ok(SearchResult {
            candidate: Candidate {
                crt_depth: match profile.q_l {
                    2048 => 30,
                    4096 if profile.universe == 16384 => 28,
                    4096 => 29,
                    _ => 31,
                },
                log_ring_dimension: 16,
                crt_bits: 50,
                base_bits: 25,
            },
            achieved_security_bits: 128,
            bgg_rlwe_security_bits: 128,
            sparse_lwr_security_bits: profile.sparse_lwr_security_bits,
            raw_key_entropy_bits: profile.raw_key_entropy_bits,
            sparse_lwr_universe: profile.universe,
            sparse_lwr_weight: profile.weight,
            official_preimage_bound: "1".to_owned(),
            ring_dimension: 1 << 16,
            bucket_width: 1,
            pbc_attempts_used: 1,
            layout_id: "aa".repeat(32),
            program_id: "bb".repeat(32),
            checker_accepted: true,
            switched_security_status: "low-P marginal estimates not supplied; cross-modulus joint-view security unresolved".into(),
            sparse_lwr_q_l: profile.q_l,
            sparse_lwr_p: profile.p,
            sparse_lwr_tier: profile.tier,
            estimator_commit: profile.estimator_commit.clone(),
            estimator_cost_model: profile.estimator_cost_model.clone(),
            estimator_shape_model: profile.estimator_shape_model.clone(),
            average_evidence: None,
            production_report: None,
            bgg_security_authority: "external_lattice_estimator".to_owned(),
            bgg_security_evidence_source: "external_lattice_estimator".to_owned(),
        })
    })
    .unwrap();
            // Every reviewed profile is evaluated, and global selection chooses the
            // later profile with the smallest CRT depth.
            assert_eq!(order, vec![2048, 4096, 4096]);
            assert_eq!(selected_profile.q_l, 4096);
            assert_eq!(selected_profile.universe, 16384);
            assert_eq!(result.candidate.crt_depth, 28);
            let candidate = result.candidate;
            assert!(
                persist_accepted_phase2_profile(&path, &config, &selected_profile, candidate, None)
                    .unwrap_err()
                    .contains("requires certification evidence")
            );
            let evidence =
                Phase2CertificationEvidence::certify(&config, &selected_profile, candidate)
                    .unwrap();
            let mut mismatched = candidate;
            mismatched.crt_depth += 1;
            assert!(
                persist_accepted_phase2_profile(
                    &path,
                    &config,
                    &selected_profile,
                    mismatched,
                    Some(&evidence)
                )
                .unwrap_err()
                .contains("exact candidate/profile/configuration")
            );
            assert!(
                persist_accepted_phase2_profile(
                    &path,
                    &config,
                    &selected,
                    candidate,
                    Some(&evidence)
                )
                .unwrap_err()
                .contains("exact candidate/profile/configuration")
            );
            persist_accepted_phase2_profile(
                &path,
                &config,
                &selected_profile,
                candidate,
                Some(&evidence),
            )
            .unwrap();
            let checkpoint: Phase1Checkpoint =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            checkpoint.validate(&config).unwrap();
            // The Phase-2 global winner is intentionally allowed to differ from the
            // Phase-1 representative stored in `checkpoint.selected`.
            assert_ne!(checkpoint.selected.tuple, selected_profile.tuple);
            assert_eq!(checkpoint.accepted_phase2.as_ref().unwrap().tuple.q_l, 4096);
            assert_eq!(checkpoint.accepted_phase2.as_ref().unwrap().candidate, candidate);

            let mut tampered = checkpoint;
            tampered.accepted_phase2.as_mut().unwrap().candidate.crt_bits = 31;
            std::fs::write(&path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
            assert!(
                Phase1Checkpoint::validate(&tampered, &config)
                    .unwrap_err()
                    .contains("outside the declared search grid")
            );
            let _ = std::fs::remove_file(path);
        }

        #[test]
        fn profile_advancement_propagates_noncanonical_errors() {
            let config = SearchConfig::reviewed();
            let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            let error =
                search_qualified_profiles(&selected, |_| Err("fatal setup error".to_owned()))
                    .unwrap_err();
            assert_eq!(error, "fatal setup error");
            let error = search_qualified_profiles(&selected, |_| {
                Err(format!("{NO_CANDIDATE_ERROR}: extra diagnostic"))
            })
            .unwrap_err();
            assert_eq!(error, format!("{NO_CANDIDATE_ERROR}: extra diagnostic"));
        }

        #[test]
        fn malformed_phase_one_checkpoint_fails_before_security_callback() {
            let config = SearchConfig::reviewed();
            let path = checkpoint_test_path("phase1-malformed");
            std::fs::write(&path, b"{not valid json").unwrap();

            let mut calls = 0;
            let error = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
                calls += 1;
                Ok(129)
            })
            .unwrap_err();
            assert!(error.contains("read Phase-1 checkpoint"));
            assert_eq!(calls, 0);
            let _ = std::fs::remove_file(path);
        }

        #[test]
        fn result_serialization_is_redacted() {
            let result = SearchResult {
        candidate: Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
        achieved_security_bits: 128,
        bgg_rlwe_security_bits: 130,
        sparse_lwr_security_bits: 140,
        raw_key_entropy_bits: raw_key_entropy_bits(512, 64),
        sparse_lwr_universe: 512,
        sparse_lwr_weight: 64,
        official_preimage_bound: "123".to_owned(),
        ring_dimension: 32,
        bucket_width: 4,
        pbc_attempts_used: 1,
        layout_id: "aa".repeat(32),
        program_id: "bb".repeat(32),
        checker_accepted: true,
        switched_security_status:
            "low-P marginal estimates not supplied; cross-modulus joint-view security unresolved"
                .into(),
        sparse_lwr_q_l: 8,
        sparse_lwr_p: 2,
        sparse_lwr_tier: SparseLwrSecurityTier::Requested,
        estimator_commit: REVIEWED_ESTIMATOR_COMMIT.to_owned(),
        estimator_cost_model: REVIEWED_ESTIMATOR_COST_MODEL.to_owned(),
        estimator_shape_model: REVIEWED_ESTIMATOR_SHAPE_MODEL.to_owned(),
        average_evidence: None,
        production_report: None,
        bgg_security_authority: "external_lattice_estimator".to_owned(),
        bgg_security_evidence_source: "external_lattice_estimator".to_owned(),
    };
            let json = serde_json::to_string(&result).unwrap();
            assert!(!json.contains("support"));
            assert!(!json.contains("selected"));
            assert!(json.contains("official_preimage_bound"));
        }

        #[test]
        fn persisted_report_declares_minimality_scope_and_evaluated_phase_one_rows() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 2..=3;
            config.log_ring_dimensions = 5..=6;
            let sparse_profile = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            let result = SearchResult {
        candidate: Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
        achieved_security_bits: 128,
        bgg_rlwe_security_bits: 128,
        sparse_lwr_security_bits: sparse_profile.sparse_lwr_security_bits,
        raw_key_entropy_bits: sparse_profile.raw_key_entropy_bits,
        sparse_lwr_universe: sparse_profile.universe,
        sparse_lwr_weight: sparse_profile.weight,
        official_preimage_bound: "123".to_owned(),
        ring_dimension: 32,
        bucket_width: 4,
        pbc_attempts_used: 1,
        layout_id: "aa".repeat(32),
        program_id: "bb".repeat(32),
        checker_accepted: true,
        switched_security_status:
            "low-P marginal estimates not supplied; cross-modulus joint-view security unresolved"
                .into(),
        sparse_lwr_q_l: sparse_profile.q_l,
        sparse_lwr_p: sparse_profile.p,
        sparse_lwr_tier: sparse_profile.tier,
        estimator_commit: sparse_profile.estimator_commit.clone(),
        estimator_cost_model: sparse_profile.estimator_cost_model.clone(),
        estimator_shape_model: sparse_profile.estimator_shape_model.clone(),
        average_evidence: None,
        production_report: None,
        bgg_security_authority: "external_lattice_estimator".to_owned(),
        bgg_security_evidence_source: "external_lattice_estimator".to_owned(),
    };
            let json =
                serde_json::to_string(&search_report(&config, &sparse_profile, result)).unwrap();

            assert!(json.contains("\"parameter_grid\":["));
            assert!(json.contains("\"universe_grid\":[8192,16384]"));
            assert!(json.contains(&format!("\"support_weight\":{}", sparse_profile.weight)));
            assert!(json.contains("\"q_l\":2048"));
            assert!(json.contains("\"output_modulus\":16"));
            assert!(json.contains("\"security_target_bits\":100"));
            assert!(json.contains("\"sparse_secret_model\":\"SparseBinary\""));
            assert!(json.contains("\"sparse_error_model\":\"Uniform\""));
            assert!(json.contains("\"selected_error_lower\":-64"));
            assert!(json.contains("\"selected_error_upper\":63"));
            assert!(json.contains("\"exact_estimator\":true"));
            assert!(json.contains("\"evaluated\":["));
            assert!(json.contains("\"universe\":16384"));
            assert!(json.contains("\"minimum_security_bits\""));
            assert!(json.contains("\"qualified\":true"));
            assert!(json.contains("\"selected_universe\":16384"));
            assert!(json.contains("\"crt_depth_min\":2"));
            assert!(json.contains("\"crt_depth_max\":3"));
            assert!(json.contains("\"log_ring_dimension_min\":5"));
            assert!(json.contains("\"log_ring_dimension_max\":6"));
            assert!(
        json.contains("\"order\":\"crt_depth_ascending_then_log_ring_dimension_ascending_then_base_bits_descending_then_crt_bits_descending; profile_tuple_q_l_p_nu_h_ascending\"")
        );
            assert!(json.contains("\"crt_base_bits_grid\":[[50,25]"));
            assert!(json.contains("\"error_sigma\":\"4\""));
            assert!(json.contains("\"encoding_error_bound\":\"26\""));
            assert!(!json.contains("selected_slots"));
            assert!(!json.contains("support_coordinates"));
        }

        #[test]
        fn sparse_lwr_estimator_arguments_use_exact_reviewed_model() {
            assert_eq!(sparse_lwr_error_bounds(2048, 16).unwrap(), (-64, 63));
            assert_eq!(
                sparse_lwr_estimator_args(16384, 17, 2048, 16).unwrap(),
                vec![
                    "16384",
                    "2048",
                    "--s-dist",
                    r#"{"name":"SparseBinary","hw":17,"n":16384}"#,
                    "--e-dist",
                    r#"{"name":"Uniform","a":-64,"b":63}"#,
                    "--exact",
                ]
            );
            assert!(sparse_lwr_error_bounds(15, 2).is_err());
            assert!(sparse_lwr_error_bounds(2048, 0).is_err());
        }

        #[test]
        fn reconstruction_coefficients_come_from_the_dcrt_parameters() {
            let dcrt = DCRTPolyParams::new(32, 2, 28, 14, None, None);
            let coefficients = exact_reconstruction_coefficients(&dcrt);
            assert_eq!(coefficients.len(), 2);
            assert!(dcrt.reconst_coeffs().iter().any(|value| value != &BigUint::from(1_u8)));
        }

        #[test]
        fn security_output_parser_is_strict_and_fail_closed() {
            assert_eq!(parse_security_bits(b"128\n"), Ok(128));
            assert!(parse_security_bits(b"\n128\n\n").is_ok());
            assert_eq!(
                parse_security_bits(b"Algorithm ... failed\nalgorithm-result\n128\n"),
                Ok(128)
            );
            assert!(parse_security_bits(b"128\n129\n").is_err());
            assert!(parse_security_bits(b"security: 128\n").is_err());
            assert!(parse_security_bits(b"+128\n").is_err());
            assert!(parse_security_bits(b"18446744073709551616\n").is_err());
        }

        #[test]
        fn fresh_digit_selection_is_strict_and_finite() {
            assert_eq!(fresh_error_base_p_digit_count(&BigUint::from(32u8), &[37, 41]), Ok(1));
            assert!(fresh_error_base_p_digit_count(&BigUint::from(32u8), &[31, 32]).is_err());
        }

        #[test]
        fn fresh_digit_selection_checks_prime_factors_of_composite_towers() {
            // B_e = 7 is below the composite modulus 35, but not below its prime
            // factor 5; the modulus-level comparison must therefore reject it.
            assert!(fresh_error_base_p_digit_count(&BigUint::from(8u8), &[35]).is_err());
            assert_eq!(fresh_error_base_p_digit_count(&BigUint::from(4u8), &[35]), Ok(1));
        }

        #[test]
        fn coarse_switch_ring_candidates_preserve_slots_and_evaluate_real_noise() {
            assert_eq!(
                candidate_switch_moduli(&[3, 5, 7], 2).unwrap(),
                vec![15u8.into(), 15u8.into(), 35u8.into()]
            );
            assert_eq!(candidate_switch_moduli(&[3, 5, 7], 3).unwrap(), vec![105u8.into(); 3]);
            assert!(candidate_switch_moduli(&[3, 5, 7], 0).is_err());
            assert!(
                mask_digit_search_bounds(
                    &4u8.into(),
                    &4u8.into(),
                    &[3, 5, 7],
                    &candidate_switch_moduli(&[3, 5, 7], 1).unwrap()
                )
                .is_err()
            );
            let mut config = SearchConfig::reviewed();
            config.sparse_lwr_universe = 10;
            config.sparse_lwr_weight = 4;
            config.sparse_lwr_modulus = 32;
            config.sparse_lwr_output_modulus = 4;
            let candidate =
                Candidate { crt_depth: 3, log_ring_dimension: 5, crt_bits: 40, base_bits: 4 };
            match prepare_candidate(&config, candidate) {
                Ok(mut prepared) => {
                    assert!(check_refresh_bundle(&config, candidate, &mut prepared).unwrap());
                    let snapshot = prepared.noise_snapshot.as_ref().unwrap();
                    assert_eq!(snapshot.refresh.switch_moduli.len(), 3);
                    eprintln!(
                        "coarse switched candidate accepted by configured noise authority: P={:?}, report={}",
                        snapshot.refresh.switch_moduli,
                        serde_json::to_string(prepared.production_report.as_ref().unwrap())
                            .unwrap()
                    );
                }
                Err(CandidatePreparationError::Infeasible(reason)) => {
                    assert!(reason.contains("coarse switched-ring scan rejected"));
                    assert!(
                        (1..=3).all(|retained| reason.contains(&format!("retained={retained}:")))
                    );
                    eprintln!("coarse switched candidate concrete rejection: {reason}");
                }
                Err(error) => panic!("unexpected candidate construction failure: {error:?}"),
            }
        }

        #[test]
        fn profile_b_switched_refresh_named_average_authority_component_search() {
            // Pure public-layout/noise interpretation: no backend, artifact sampling,
            // external estimator, integration execution, or benchmark is invoked.
            let mut config = SearchConfig::reviewed();
            config.noise_model = NoiseSearchModel::AverageCase;
            config.crt_depths = 27..=27;
            config.log_ring_dimensions = 16..=16;
            config.sparse_lwr_universe = 8192;
            config.sparse_lwr_weight = 18;
            config.sparse_lwr_modulus = 4096;
            config.sparse_lwr_output_modulus = 32;
            let candidate =
                Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
            eprintln!(
                "Profile B: n=65536, depth=27, crt_bits=50, base_bits=25, Q=4096, p=32, v=8192, h=18; security target=100; authority={:?}; coarse contiguous retained-width groups and uniform mask digits; security acceptance is NOT inferred from noise acceptance",
                average_case_config(&config).unwrap()
            );
            match prepare_candidate(&config, candidate) {
                Ok(mut prepared) => {
                    assert!(check_refresh_bundle(&config, candidate, &mut prepared).unwrap());
                    let report = prepared.production_report.as_ref().unwrap();
                    assert!(report.average_case.as_ref().unwrap().accepted);
                    assert_eq!(report.anchor_security_inputs.len(), 6);
                    assert!(report.anchor_security_inputs.iter().all(
                        |group| group.modulus.bits() == 250 &&
                            group.ring_dimension == 65536 &&
                            group.anchor_columns == 24 &&
                            group.execution_budget == 1 &&
                            group.estimated_security_bits.is_none()
                    ));
                    assert_eq!(prepared.mask_base_p_digit_count, 38);
                    eprintln!(
                        "PROFILE_B_NOISE_ACCEPTED {}",
                        serde_json::to_string(report).unwrap()
                    );
                }
                Err(CandidatePreparationError::Infeasible(reason)) => {
                    panic!("previously accepted Profile B noise candidate regressed: {reason}");
                }
                Err(error) => panic!("Profile B component construction failed: {error:?}"),
            }
        }

        #[test]
        fn profile_b_minimum_mask_retained_width_search() {
            let mut config = SearchConfig::reviewed();
            config.noise_model = NoiseSearchModel::AverageCase;
            config.crt_depths = 27..=27;
            config.log_ring_dimensions = 16..=16;
            config.sparse_lwr_universe = 8192;
            config.sparse_lwr_weight = 18;
            config.sparse_lwr_modulus = 4096;
            config.sparse_lwr_output_modulus = 32;
            let candidate =
                Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
            let mut context = build_candidate_context(&config, candidate).unwrap();
            let started = std::time::Instant::now();
            let run = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/profile_b_minimum_mask_search")
                .join(run.to_string());
            fs::create_dir_all(&directory).unwrap();
            eprintln!("SEARCH_DIRECTORY {}", directory.display());
            let mut rows = Vec::new();
            let mut selected: Option<(usize, usize)> = None;
            for retained in 1..=candidate.crt_depth {
                eprintln!(
                    "SEARCH_WIDTH_START retained={retained} elapsed={:.3}s",
                    started.elapsed().as_secs_f64()
                );
                context.switch_moduli =
                    candidate_switch_moduli(&context.crt_moduli, retained).unwrap();
                match prepare_candidate_from_ring_context(&config, &context, None, false) {
                    Ok(mut prepared) => {
                        assert!(check_refresh_bundle(&config, candidate, &mut prepared).unwrap());
                        let report = prepared.production_report.as_ref().unwrap();
                        let average = report.average_case.as_ref().unwrap();
                        let snapshot = prepared.noise_snapshot.as_ref().unwrap();
                        let mask_digits = prepared.mask_base_p_digit_count;
                        let score = (candidate.crt_depth * mask_digits, retained);
                        if selected.is_none_or(|best| score < best) {
                            selected = Some(score);
                        }
                        let row = serde_json::json!({
                            "retained": retained, "accepted": true,
                            "mask_digits": mask_digits, "sum_mask_digits": score.0,
                            "fresh_digits": prepared.fresh_error_base_p_digit_count,
                            "label_count": prepared.prf_label_count,
                            "q": snapshot.refresh.global_modulus.as_ref().unwrap().to_string(),
                            "q_t": snapshot.refresh.plaintext_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
                            "P_t": snapshot.refresh.switch_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
                            "M": snapshot.refresh.base_p.pow(mask_digits as u32).to_string(),
                            "max_F_avg": average.refresh.max_f_avg.to_string(),
                            "minimum_rounding_margin": average.refresh.slots.iter().map(|slot| &slot.rounding_margin).min().unwrap().to_string(),
                            "minimum_smudging_margin": average.refresh.slots.iter().map(|slot| &slot.smudging_margin).min().unwrap().to_string(),
                            "joint_event_count": average.refresh.joint_event_count.to_string(),
                            "z_squared": format!("{}/{}", average.refresh.z_squared.numerator, average.refresh.z_squared.denominator),
                            "anchor_group_count": report.anchor_security_inputs.len(),
                            "decoder_columns": report.anchor_security_inputs.iter().map(|group| group.anchor_columns).collect::<Vec<_>>(),
                            "official_preimage_bound": prepared.official_preimage_bound.to_string(),
                            "selected_preimage_bound": prepared.selected_preimage_bound.to_string(),
                            "fresh_margins": report.fresh_error_margins,
                            "slots": average.refresh.slots.iter().map(|slot| serde_json::json!({
                                "slot": slot.slot, "P": slot.switch_modulus.to_string(), "J": slot.lift.to_string(),
                                "Delta": slot.spacing.to_string(), "F_avg": slot.f_avg.to_string(),
                                "B_m": slot.deterministic_mask_bound.to_string(),
                                "rounding_margin": slot.rounding_margin.to_string(),
                                "smudging_margin": slot.smudging_margin.to_string(),
                            })).collect::<Vec<_>>(),
                        });
                        fs::write(
                            directory.join(format!("retained-{retained}-full-report.json")),
                            serde_json::to_vec_pretty(report).unwrap(),
                        )
                        .unwrap();
                        eprintln!(
                            "SEARCH_WIDTH_ACCEPT retained={retained} dm={mask_digits} labels={} elapsed={:.3}s",
                            prepared.prf_label_count,
                            started.elapsed().as_secs_f64()
                        );
                        rows.push(row);
                    }
                    Err(CandidatePreparationError::Infeasible(reason)) => {
                        eprintln!(
                            "SEARCH_WIDTH_REJECT retained={retained} reason={reason} elapsed={:.3}s",
                            started.elapsed().as_secs_f64()
                        );
                        rows.push(
                    serde_json::json!({"retained": retained, "accepted": false, "reason": reason}),
                );
                    }
                    Err(error) => {
                        panic!("unexpected search error at retained={retained}: {error:?}")
                    }
                }
                fs::write(directory.join("widths.json"), serde_json::to_vec_pretty(&rows).unwrap())
                    .unwrap();
            }
            let (sum_digits, retained) =
                selected.expect("no accepted candidate in the declared grid");
            let summary = serde_json::json!({
                "scope": "minimum sum of mask digits in all retained widths 1..27, contiguous groups, uniform digits; not exhaustive subsets or ragged optimum",
                "baseline": {"n":65536,"crt_depth":27,"crt_bits":50,"base_bits":25,"Q":4096,"p":32,"v":8192,"h":18,"encoding_sigma":4.0,"decoder_sigma":4.578},
                "noise_authority":"AverageCase TFHE-style heuristic with conservative Add",
                "failure_exponent":100,"mask_statistical_security_bits":100,"tail_correction_bits":0,
                "selected_retained":retained,"sum_mask_digits":sum_digits,
                "joint_security":"unresolved; noise acceptance is not a whole-construction security claim",
                "elapsed_seconds":started.elapsed().as_secs_f64(), "candidates":rows,
            });
            fs::write(directory.join("search.json"), serde_json::to_vec_pretty(&summary).unwrap())
                .unwrap();
            eprintln!(
                "SEARCH_COMPLETE retained={retained} sum_mask_digits={sum_digits} report={}",
                directory.join("search.json").display()
            );
        }

        #[test]
        fn export_profile_b_modswitch_benchmark_inputs() {
            let mut config = SearchConfig::from_env().unwrap();
            assert_eq!(config.noise_model, NoiseSearchModel::AverageCase);
            assert_eq!(
                (
                    config.sparse_lwr_modulus,
                    config.sparse_lwr_output_modulus,
                    config.sparse_lwr_universe,
                    config.sparse_lwr_weight
                ),
                (4096, 32, 8192, 18)
            );
            assert_eq!((config.security_bits, config.mask_statistical_security_bits), (100, 100));
            config.bgg_security_envelope = Some("reviewed_monotone_envelope".into());
            let candidate =
                Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/profile_b_minimum_mask_search/1788588283213876339");
            let saved: ProductionNoiseReport = serde_json::from_slice(
                &fs::read(directory.join("retained-5-full-report.json")).unwrap(),
            )
            .unwrap();
            let average = saved.average_case.as_ref().expect("saved average authority");
            assert!(
                average.accepted && average.hard_authority_accepted && average.correctness_accepted
            );
            let selected: serde_json::Value =
                serde_json::from_slice(&fs::read(directory.join("search.json")).unwrap()).unwrap();
            assert_eq!(selected["selected_retained"].as_u64(), Some(5));
            let row = selected["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["retained"].as_u64() == Some(5))
                .unwrap();
            let mask_digits = row["mask_digits"].as_u64().unwrap() as usize;
            let fresh_digits = row["fresh_digits"].as_u64().unwrap() as usize;
            let moduli = row["P_t"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().parse::<BigUint>().unwrap())
                .collect::<Vec<_>>();
            let mut context = build_candidate_context(&config, candidate).unwrap();
            context.switch_moduli = moduli.clone();
            let setup =
                setup_for_candidate_context(&config, &context, mask_digits, fresh_digits).unwrap();
            // Reconstruct actual public identities and route metadata, without
            // evaluate_pre_refresh/simulate/check_refresh_bundle or an estimator.
            let snapshot = BggRefreshParameters::build_noise_snapshot_from_public_layout(
                &setup,
                &context.program,
                &context.generated_layout.public_layout,
                &context.active_cells,
                None,
            )
            .unwrap();
            assert_eq!(snapshot.setup_identity(), &average.snapshot_identity);
            assert_eq!(snapshot.refresh.switch_moduli, moduli);
            let mut prepared = prepare_candidate_with_fixed_digits(
                &config,
                candidate,
                mask_digits,
                fresh_digits,
                &moduli,
            )
            .unwrap();
            assert_eq!(
                prepared.bundle.as_ref().unwrap().public_identity(),
                &average.snapshot_identity
            );
            assert_eq!(prepared.anchor_security_inputs, saved.anchor_security_inputs);
            assert_eq!(prepared.prf_label_count as u64, row["label_count"].as_u64().unwrap());
            prepared.noise_snapshot = Some(snapshot);
            let phase1 = select_sparse_lwr_profile(&config, |nu, h, q, p| {
                let tuple = config
                    .sparse_lwr_phase1_grid
                    .iter()
                    .find(|tuple| (tuple.nu, tuple.h, tuple.q_l, tuple.p) == (nu, h, q, p))
                    .ok_or("tuple is not declared")?;
                sparse_lwr_parameter_support::pinned_estimator_security_bits(tuple)?
                    .ok_or_else(|| "missing pinned sparse-LWR evidence".into())
            })
            .unwrap();
            let bgg = reviewed_bgg_security_bits(&config, candidate).unwrap();
            let result = SearchResult {
        candidate,
        achieved_security_bits: bgg.min(phase1.sparse_lwr_security_bits),
        switched_security_status: "selected low-P marginal evidence is external; cross-modulus joint-view security unresolved".into(),
        bgg_rlwe_security_bits: bgg,
        sparse_lwr_security_bits: phase1.sparse_lwr_security_bits,
        raw_key_entropy_bits: phase1.raw_key_entropy_bits,
        sparse_lwr_universe: phase1.universe,
        sparse_lwr_weight: phase1.weight,
        official_preimage_bound: prepared.official_preimage_bound.to_string(),
        ring_dimension: prepared.ring_dimension,
        bucket_width: prepared.bucket_width,
        pbc_attempts_used: prepared.pbc_attempts_used,
        layout_id: hex(prepared.layout_id),
        program_id: hex(prepared.program_id),
        checker_accepted: average.accepted,
        sparse_lwr_q_l: phase1.q_l,
        sparse_lwr_p: phase1.p,
        sparse_lwr_tier: phase1.tier,
        estimator_commit: phase1.estimator_commit.clone(),
        estimator_cost_model: phase1.estimator_cost_model.clone(),
        estimator_shape_model: phase1.estimator_shape_model.clone(),
        average_evidence: final_average_evidence(&config, &prepared, average).unwrap(),
        production_report: Some(saved.clone()),
        bgg_security_authority: bgg_security_authority(&config).into(),
        bgg_security_evidence_source: bgg_security_evidence_source(&config).into(),
    };
            let mut checkpoint = Phase1Checkpoint::from_selection(&config, &phase1).unwrap();
            checkpoint.accepted_phase2 =
                Some(AcceptedPhase2Profile { tuple: phase1.tuple.clone(), candidate });
            checkpoint.validate(&config).unwrap();
            let checkpoint_path =
                std::env::var(PHASE1_CHECKPOINT_ENV).expect("output checkpoint path");
            let report_path = std::env::var("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT")
                .expect("output report path");
            let report = search_report(&config, &phase1, result);
            for (path, value) in [
                (checkpoint_path, serde_json::to_value(&checkpoint).unwrap()),
                (report_path, serde_json::to_value(&report).unwrap()),
            ] {
                let output =
                    fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
                serde_json::to_writer_pretty(output, &value).unwrap();
                eprintln!("EXPORTED {path}");
            }
            eprintln!(
                "EXPORT_COMPLETE layout={} program={} setup={} dm={mask_digits} de={fresh_digits} labels={} simulation=false lattice_estimator=false backend_execution=false",
                hex(prepared.layout_id),
                hex(prepared.program_id),
                hex(average.snapshot_identity),
                prepared.prf_label_count
            );
        }

        #[test]
        fn mask_digit_search_starts_at_domain_coverage_and_stops_at_safe_bound() {
            assert_eq!(
                mask_digit_search_bounds(
                    &BigUint::from(2u8),
                    &BigUint::from(8u8),
                    &[100, 200],
                    &[20000u32.into(), 20000u32.into()]
                ),
                Ok((3, 6))
            );
            assert_eq!(
                mask_digit_search_bounds(
                    &BigUint::from(32u8),
                    &BigUint::from(32u8),
                    &[32, 37],
                    &[1184u32.into(), 1184u32.into()]
                ),
                Err(CandidatePreparationError::Infeasible(
                    "strict deterministic mask boundary is below the domain minimum".into()
                ))
            );
        }

        #[test]
        fn emits_large_profile_sparse_prf_stage_diagnostics() {
            let mut config = SearchConfig::reviewed();
            config.crt_depths = 62..=62;
            config.log_ring_dimensions = 16..=16;
            config.crt_bits = 32;
            config.base_bits = 1;
            let ring_dimension = 1usize << 16;
            let dcrt = DCRTPolyParams::new(ring_dimension as u32, 62, 32, 1, None, None);
            let q_bits = dcrt.modulus().bits();
            info!(q_bits, ring_dimension, "large exact-noise diagnostic profile");
            let result = prepare_candidate(
                &config,
                Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 32, base_bits: 1 },
            );
            info!(accepted = result.is_ok(), "large exact-noise diagnostic completed");
        }

        #[test]
        fn diag_free_prepare_and_check_bundle_smoke() {
            let config = SearchConfig::reviewed();
            let result = prepare_candidate(
                &config,
                Candidate { crt_depth: 30, log_ring_dimension: 15, crt_bits: 32, base_bits: 16 },
            );
            match result {
                Ok(mut prepared) => {
                    let checked = check_refresh_bundle(&config, prepared.candidate, &mut prepared)
                        .expect("compact noise checker must complete");
                    assert!(
                        checked,
                        "reviewed fixed candidate must pass canonical compact acceptance"
                    );
                }
                Err(CandidatePreparationError::Infeasible(reason)) => {
                    assert!(reason.starts_with("coarse switched-ring scan rejected: retained=1:"));
                    assert!(
                        reason
                            .contains("no mask digit count passed the complete production report")
                    );
                    assert!(
                        (1..=30).all(|retained| reason.contains(&format!("retained={retained}:")))
                    );
                }
                Err(CandidatePreparationError::Fatal(reason)) => {
                    panic!("compact preparation failed fatally: {reason}")
                }
            }
        }
    }
    mod sparse_lwr_tests {
        use super as test_utils;

        use serde_json::Value;
        use std::collections::BTreeMap;
        use test_utils::sparse_lwr_parameter_support::*;

        #[test]
        fn reviewed_primary_tuple_derives_reviewed_values() {
            for (tuple, expected_minimum, expected_floor) in reviewed_phase1_tuple_grid()
                .into_iter()
                .zip([100.5636531151896, 100.56464330713312, 103.18615790798486])
                .zip([100_u64, 100, 103])
                .map(|((tuple, minimum), floor)| (tuple, minimum, floor))
            {
                assert_eq!(tuple.estimator_minimum_classical_bits, Some(expected_minimum));
                assert_eq!(tuple.estimator_security_bits, Some(expected_floor));
                assert_eq!(pinned_estimator_security_bits(&tuple).unwrap(), Some(expected_floor));
                let candidate = tuple.candidate().unwrap();
                let derived = candidate.derived().unwrap();
                assert_eq!(candidate.nu, tuple.nu);
                assert_eq!(candidate.h, tuple.h);
                assert_eq!(derived.p_prime, tuple.p / 2);
                assert_eq!(derived.delta, tuple.q_l / tuple.p);
                assert_eq!((derived.error_lower, derived.error_upper), (-64, 63));
            }
            assert_eq!(
                native_exp_layout(2048, 1usize << 16).unwrap(),
                mxx_exponent_lut::ExpLayout { stride_log2: 6, width_log2: 10 }
            );
            assert_eq!(
                native_exp_layout(4096, 1usize << 16).unwrap(),
                mxx_exponent_lut::ExpLayout { stride_log2: 5, width_log2: 11 }
            );
        }

        #[test]
        fn estimator_evidence_is_optional_but_must_be_complete_when_present() {
            let mut tuple = reviewed_phase1_tuple_grid().remove(0);
            tuple.estimator_minimum_classical_bits = None;
            tuple.estimator_security_bits = None;
            assert_eq!(pinned_estimator_security_bits(&tuple).unwrap(), None);

            tuple.estimator_minimum_classical_bits = Some(100.5);
            assert!(pinned_estimator_security_bits(&tuple).unwrap_err().contains("incomplete"));

            tuple.estimator_security_bits = Some(99);
            assert!(pinned_estimator_security_bits(&tuple).unwrap_err().contains("invalid"));
        }

        #[test]
        fn non_dividing_output_modulus_is_rejected() {
            assert!(SparseLwrCandidate::new("bad", 32, 4, 8, 31, vec![16]).is_err());
        }

        #[test]
        fn incompatible_encoding_ring_is_rejected() {
            assert!(SparseLwrCandidate::new("bad", 32, 4, 8, 2, vec![10]).is_err());
        }

        #[test]
        fn phase_one_grid_is_ordered_and_accepts_native_nonbinary_parameters() {
            let grid = reviewed_phase1_tuple_grid();
            assert_eq!(
                grid.iter()
                    .map(|tuple| (tuple.q_l, tuple.p, tuple.nu, tuple.h))
                    .collect::<Vec<_>>(),
                vec![(2048, 16, 16384, 17), (4096, 32, 16384, 17), (4096, 32, 8192, 18)]
            );
            assert!(
                SparseLwrCandidate::new("native-p16", 16384, 17, 2048, 16, vec![65536]).is_ok()
            );
            assert!(SparseLwrCandidate::new("native-p32", 8192, 18, 4096, 32, vec![65536]).is_ok());
            assert!(SparseLwrCandidate::new("bad", 896, 24, 8, 4, vec![10]).is_err());
            assert!(SparseLwrCandidate::new("bad", 896, 24, 9, 2, vec![32]).is_err());
        }

        #[test]
        fn native_candidate_derives_exact_n_2_16_layout_independently_of_nu() {
            let candidate = SparseLwrCandidate::new(
                "native-q4096-p64",
                16384,
                17,
                4096,
                64,
                vec![8192, 16384, 32768, 65536],
            )
            .unwrap();
            let derived = candidate.derived().unwrap();
            assert_eq!(derived.p_prime, 32);
            assert_eq!(derived.delta, 64);
            assert_eq!((derived.error_lower, derived.error_upper), (-32, 31));
            let layout = native_exp_layout(4096, 1usize << 16).unwrap();
            assert_eq!(layout.stride_log2, 5);
            assert_eq!(layout.width_log2, 11);
            assert_eq!(layout.effective_modulus(1usize << 16), 4096);
            assert_eq!(candidate.n_enc_candidates.last().copied(), Some(1usize << 16));

            let changed_universe = SparseLwrCandidate::new(
                "native-q4096-p64-small-universe",
                32,
                8,
                4096,
                64,
                vec![1usize << 16],
            )
            .unwrap();
            assert_eq!(native_exp_layout(4096, 1usize << 16).unwrap(), layout);
            assert_eq!(changed_universe.derived().unwrap().delta, derived.delta);
        }

        #[test]
        fn infinity_is_not_treated_as_security() {
            let mut attacks = BTreeMap::new();
            for attack in EXPECTED_ATTACKS {
                attacks.insert((*attack).to_owned(), serde_json::json!({"rop_log2": 200.0}));
            }
            attacks.get_mut("bkw").unwrap()["rop_log2"] = Value::Null;
            let report = EstimatorReport {
                schema_version: 1,
                repository_url: LATTICE_ESTIMATOR_REPOSITORY.to_owned(),
                git_commit: "e35f45b7976a90a79c3c6625a45bbc344c1abc67".to_owned(),
                sage_version: "SageMath version 10.7".to_owned(),
                python_version: "3.12.12".to_owned(),
                cost_model: "MATZOV".to_owned(),
                shape_model: "gsa".to_owned(),
                quantum: false,
                sample_count: "infinity".to_owned(),
                attacks,
                failures: Vec::new(),
                infinite_attacks: vec!["bkw".to_owned()],
                minimum_classical_bits: 200.0,
            };
            assert!(report.validate().is_ok());
            let mut numeric_infinite = report.clone();
            numeric_infinite.attacks.get_mut("bkw").unwrap()["rop_log2"] = serde_json::json!(200.0);
            assert!(numeric_infinite.validate().is_err());
            let mut unknown = report.clone();
            unknown.infinite_attacks.push("unknown".to_owned());
            assert!(unknown.validate().is_err());
            let mut duplicate = report;
            duplicate.infinite_attacks.push("bkw".to_owned());
            assert!(duplicate.validate().is_err());
        }
    }
}

mod bgg_refresh_artifact_tests {
    use super as test_utils;
    use mxx_exponent_lut::{layout::ExpLayout, noise_refresh::prf::SparseLwrPrfProfile};
    use mxx_ir_core::{
        ParamEnv, artifact::ProductionId, encoding::spec_hash, types::ConcreteMatrixType,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
    };
    use mxx_runtime::{
        ExecutionConfig, RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::gpu::gpu_backend_on,
        execute_in_session, execute_with_config,
        transcript::{SamplingMode, TranscriptRecorder},
    };
    use std::{collections::BTreeMap, num::NonZeroUsize};
    use test_utils::tiny_native_refresh_bundle;

    /// Build and validate all BGG lifecycle graphs without a backend.  This
    /// guards the artifact boundary used by the GPU test: encryption must be
    /// an independent graph between preprocessing and online evaluation.
    #[test]
    fn bgg_refresh_graphs_have_three_stage_boundary() {
        let (_parameters, bundle, _labels) = tiny_native_refresh_bundle();
        let bindings = ParamEnv::default();
        let preprocessing = bundle
            .preprocessing_graph()
            .validate_with_manifests(&bindings, bundle.preprocessing_manifests())
            .expect("preprocessing graph validation");
        let benchmark_preprocessing = bundle
            .benchmark_preprocessing_graph()
            .validate_with_manifests(&bindings, bundle.preprocessing_manifests())
            .expect("benchmark preprocessing graph validation");
        assert!(
            benchmark_preprocessing
                .source
                .outputs()
                .keys()
                .all(|name| !name.starts_with("oracle-")),
            "BGG benchmark preprocessing must exclude correctness-only oracle work"
        );

        // Rebind with IDs that do not occur in the deterministic planning
        // template. This exercises the same factory path used after runtime
        // execution, rather than merely adding the new IDs to a manifest map.
        let producer_spec_hash = spec_hash(&preprocessing.source, &preprocessing.bindings).unwrap();
        let producer =
            ProductionId { spec_hash: producer_spec_hash.clone(), execution_nonce: [0xa1; 32] };
        let mut producer_manifest = bundle
            .preprocessing_manifests()
            .values()
            .find(|manifest| manifest.production_id.spec_hash == producer_spec_hash)
            .cloned()
            .unwrap_or_else(|| {
                bundle
                    .preprocessing_manifests()
                    .values()
                    .next()
                    .expect("preprocessing manifest")
                    .clone()
            });
        producer_manifest.production_id = producer.clone();
        let encrypt_binding = bundle
            .encrypt_graph_for(producer.clone(), producer_manifest.clone())
            .expect("encryption graph validation");
        let encrypt = encrypt_binding.graph();
        let encryption = ProductionId {
            spec_hash: spec_hash(&encrypt.source, &encrypt.bindings).unwrap(),
            execution_nonce: [0xa2; 32],
        };
        let encryption_manifest =
            mxx_ir_core::artifact::export_validated_manifest(encryption.clone(), &encrypt)
                .expect("encryption manifest");
        let online = bundle
            .online_graph_for(
                producer.clone(),
                producer_manifest.clone(),
                encryption.clone(),
                encryption_manifest.clone(),
                &encrypt_binding,
            )
            .expect("online graph validation");

        // A manifest cannot self-authorize a different encryption graph: even
        // when its production ID is changed consistently, the trusted binding
        // returned by encrypt_graph_for must reject the altered spec hash.
        let tampered_encryption = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([0x99; 32]),
            execution_nonce: encryption.execution_nonce,
        };
        let mut tampered_manifest = encryption_manifest;
        tampered_manifest.production_id = tampered_encryption.clone();
        assert!(
            bundle
                .online_graph_for(
                    producer,
                    producer_manifest,
                    tampered_encryption,
                    tampered_manifest,
                    &encrypt_binding,
                )
                .is_err(),
            "tampered encryption spec hash must be rejected by the trusted binding"
        );

        assert_ne!(preprocessing.source.name(), encrypt.source.name());
        assert_ne!(encrypt.source.name(), online.source.name());
        assert!(
            encrypt.source.outputs().contains_key("encrypted-input-vector"),
            "encryption must export a private input artifact"
        );
        assert!(
            online.source.scopes().values().flat_map(|scope| scope.nodes()).all(|node| !matches!(
                node.kind(),
                mxx_ir_core::node::NodeKind::UniformIntervalSample { .. }
            )),
            "online graph must not resample the mask secret"
        );
    }

    /// Real noisy execution, deliberately separate from the benchmark estimator.
    #[test]
    #[serial_test::serial]
    fn test_gpu_noise_refresh_round_trip() {
        use mxx_exponent_lut::noise_refresh::{
            pbc::{PbcActiveCellIndex, PbcCell, clear_pbc_inner_product},
            refresh::{RefreshPrfLabel, RefreshPrfLabelIndex},
        };
        use mxx_primitives::{element::PolyElem, poly::dcrt::params::DCRTPolyParams};
        use mxx_runtime::{
            artifact::{ArtifactStore, FileArtifactStore},
            backend::Backend,
        };
        use num_bigint::{BigInt, BigUint};
        use num_traits::{One, Zero};
        use rand::Rng;
        use rayon::prelude::*;

        let config = test_utils::SearchConfig::from_env().expect("round-trip parameters");
        assert!(
            config.error_sigma > 0.0 && config.decoder_sigma > 0.0,
            "real Gaussian noise required"
        );
        let report: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::env::var("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT")
                    .expect("accepted report required"),
            )
            .expect("read accepted report"),
        )
        .expect("parse accepted report");
        let selected = &report["result"];
        assert!(selected["bgg_rlwe_security_bits"].as_u64().unwrap() >= config.security_bits);
        assert!(selected["sparse_lwr_security_bits"].as_u64().unwrap() >= config.security_bits);
        assert_eq!(
            report["phase1"]["security_target_bits"].as_u64().unwrap(),
            config.security_bits
        );
        assert_eq!(
            report["phase2"]["security_target_bits"].as_u64().unwrap(),
            config.security_bits
        );
        assert_eq!(selected["estimator_commit"].as_str().unwrap(), config.phase1_estimator_commit);
        assert_eq!(
            selected["estimator_cost_model"].as_str().unwrap(),
            config.phase1_estimator_cost_model
        );
        assert_eq!(
            selected["estimator_shape_model"].as_str().unwrap(),
            config.phase1_estimator_shape_model
        );
        assert_eq!(selected["checker_accepted"], true);
        assert!(selected["achieved_security_bits"].as_u64().unwrap() >= config.security_bits);
        for (name, expected) in [
            ("sparse_lwr_universe", config.sparse_lwr_universe),
            ("sparse_lwr_weight", config.sparse_lwr_weight),
            ("sparse_lwr_q_l", config.sparse_lwr_modulus),
            ("sparse_lwr_p", config.sparse_lwr_output_modulus),
        ] {
            assert_eq!(selected[name].as_u64().unwrap(), expected as u64, "{name}");
        }
        let candidate: test_utils::Candidate =
            serde_json::from_value(selected["candidate"].clone()).unwrap();
        let evidence = &selected["average_evidence"];
        assert_eq!(evidence["accepted"], true);
        let mask_digits = evidence["mask_digit_count"].as_u64().unwrap() as usize;
        let fresh_digits = evidence["fresh_error_digit_count"].as_u64().unwrap() as usize;
        let params = DCRTPolyParams::new(
            1 << candidate.log_ring_dimension,
            candidate.crt_depth,
            candidate.crt_bits,
            candidate.base_bits,
            None,
            None,
        );
        let n = params.ring_dimension() as usize;
        println!(
            "round_trip_security bgg_rlwe_bits={} sparse_lwr_bits={} N={n} q_bits={} sigma={}",
            selected["bgg_rlwe_security_bits"],
            selected["sparse_lwr_security_bits"],
            params.modulus().bits(),
            config.error_sigma
        );
        let mut rng = rand::rng();
        let plaintext_exponent = rng.random_range(1..n);
        let (bundle, generated, profile) = test_utils::round_trip_bundle(
            &config,
            candidate,
            mask_digits,
            fresh_digits,
            plaintext_exponent,
        )
        .expect("build real refresh verification bundle");
        println!(
            "round_trip_config security_bits={} mask_security_bits={} correctness_failure_exponent={} encoding_sigma={} decoder_sigma={} candidate={candidate:?}",
            config.security_bits,
            config.mask_statistical_security_bits,
            config.security_bits,
            config.error_sigma,
            config.decoder_sigma
        );
        let noise = bundle
            .simulate_average_noise(&test_utils::average_case_config(&config).unwrap())
            .expect("randomized layout noise authority");
        assert!(noise.accepted, "randomized runtime layout must independently pass authority");
        let columns = config.secret_dimension * params.modulus_digits();
        let labels = RefreshPrfLabelIndex::new(
            [0; 32],
            &mxx_exponent_lut::noise_refresh::simulation::RefreshTranscriptTopology {
                crt_slot_count: candidate.crt_depth,
                component_columns: columns,
                coefficient_count: n,
                mask_digit_counts: vec![mask_digits; candidate.crt_depth],
                fresh_digit_count: fresh_digits,
            },
        )
        .unwrap();
        let active = PbcActiveCellIndex::build(&generated.public_layout).unwrap();
        let table = profile.canonical_readout().unwrap().canonical_coefficients().unwrap().to_vec();
        let q = profile.q().unwrap();
        let scalar_type = ConcreteMatrixType::scalar(params.modulus().as_ref().clone().into(), n);
        // Share immutable staging bytes for equal public monomials. Every label
        // still has its independently sampled exponent; only Q distinct matrices
        // exist, so their storage need not grow with the label count.
        let public_monomials = (0..q)
            .into_par_iter()
            .map(|z| {
                let exponent = z << profile.layout.stride_log2;
                let poly = DCRTPoly::const_rotate_poly(&params, exponent % n);
                let poly = if exponent >= n { -poly } else { poly };
                RuntimeValue::HostMatrix {
                    matrix_type: scalar_type.clone(),
                    bytes: DCRTPolyMatrix::from_poly_vec_row(&params, vec![poly])
                        .to_cpu_staging_bytes()
                        .into(),
                }
            })
            .collect::<Vec<_>>();
        let mut expected_coefficients = vec![vec![BigInt::zero(); n]; columns];
        let mut monomials = Vec::with_capacity(labels.len() * active.len());
        for index in 0..labels.len() {
            let public = (0..config.sparse_lwr_universe)
                .map(|_| rng.random_range(0..q as u64))
                .collect::<Vec<_>>();
            let z = clear_pbc_inner_product(
                &generated.public_layout,
                generated.private_schedule(),
                &public,
                q as u64,
            )
            .unwrap() as usize;
            if let RefreshPrfLabel::FreshError { component, coefficient, digit, .. } =
                labels.label(index).unwrap()
            {
                let f = &table[z % (q / 2)];
                let d = if z < q / 2 { f.clone() } else { BigInt::one() - f };
                // BGG payload has negative sign: routing -g and adding 1/2 gives 1-d.
                expected_coefficients[component][coefficient] += (BigInt::one() - d) *
                    BigInt::from(config.sparse_lwr_output_modulus).pow(digit as u32);
            }
            monomials.extend(active.iter().map(|(bucket, slot, _)| {
                let z = match generated.public_layout.cells[bucket][slot] {
                    PbcCell::Real { coordinate, .. } => public[coordinate] as usize,
                    PbcCell::Dummy => 0,
                    PbcCell::Padding => unreachable!("active index excludes padding"),
                };
                public_monomials[z].clone()
            }));
        }
        let expected = DCRTPolyMatrix::from_poly_vec_row(
            &params,
            expected_coefficients
                .par_iter()
                .map(|values| {
                    DCRTPoly::from_coeffs(
                        &params,
                        &values
                            .iter()
                            .map(|v| {
                                mxx_primitives::element::finite_ring::FinRingElem::new(
                                    v.clone(),
                                    params.modulus(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        );
        let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
        assert!(!devices.is_empty(), "real GPU required");
        let (moduli, _, _) = params.to_crt();
        let mut backend = gpu_backend_on(
            [mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams::new(
                params.ring_dimension(),
                moduli,
                params.base_bits(),
                None,
            )],
            devices,
        );
        let preprocessing = bundle
            .preprocessing_graph()
            .validate_with_manifests(&ParamEnv::default(), bundle.preprocessing_manifests())
            .unwrap();
        let mut inputs = bundle.selector_runtime_inputs(&mut backend).unwrap();
        inputs.insert(
            "refresh-parameter-search-hash-key".into(),
            RuntimeValue::Bytes(rng.random::<[u8; 32]>().to_vec()),
        );
        inputs.insert(
            "refresh-parameter-search-native-monomials".into(),
            RuntimeValue::IndexedFamily(monomials),
        );
        std::fs::create_dir_all("test_data/noise_refresh_round_trip").unwrap();
        let directory = tempfile::tempdir_in("test_data/noise_refresh_round_trip").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let production =
            execute_in_session(&preprocessing, &mut backend, inputs, &mut store, rng.random())
                .expect("real preprocessing")
                .production_id
                .unwrap();
        drop(store);
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        let manifest = store.load_manifest(&production).unwrap();
        // Encryption is a separate artifact boundary.  The online graph is
        // intentionally validated and executed only after this private input
        // artifact has been produced.
        let encrypt_binding =
            bundle.encrypt_graph_for(production.clone(), manifest.clone()).unwrap();
        let encrypt = encrypt_binding.graph();
        let encrypted =
            execute_in_session(&encrypt, &mut backend, BTreeMap::new(), &mut store, rng.random())
                .expect("real encryption");
        let encrypt_production = encrypted.production_id.expect("encryption production");
        let encrypt_manifest = store.load_manifest(&encrypt_production).unwrap();
        let online = bundle
            .online_graph_for(
                production,
                manifest,
                encrypt_production,
                encrypt_manifest,
                &encrypt_binding,
            )
            .unwrap();
        let mut previous_vector = None;
        for online_run in 0..3 {
            let result = execute_with_config(
                &online,
                &mut backend,
                BTreeMap::new(),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .expect("real online refresh");
            let matrix = |name: &str| {
                let RuntimeValue::Matrix(value) = &result.outputs[name] else {
                    panic!("matrix output required")
                };
                DCRTPolyMatrix::from_compact_bytes(
                    &params,
                    &backend.matrix_to_bytes(value).unwrap(),
                )
            };
            let residual = matrix("residual");
            assert_eq!(
                residual, expected,
                "every fresh-error coefficient must match the independent clear PRF oracle"
            );
            let vector = matrix("refresh-verification-vector");
            assert_eq!(
                vector,
                matrix("refresh-verification-anchor") + expected.clone(),
                "refreshed vector must equal plaintext anchor plus actual fresh error"
            );
            if let Some(previous) = previous_vector.replace(vector.clone()) {
                assert_ne!(previous, vector, "fresh online sampling must change the encoding mask");
            }
            let maximum = residual
                .get_row(0)
                .par_iter()
                .flat_map_iter(|p| p.coeffs())
                .map(|v| v.value().min(&(params.modulus().as_ref() - v.value())).clone())
                .max()
                .unwrap();
            let bound = BigUint::from(config.sparse_lwr_output_modulus).pow(fresh_digits as u32) -
                BigUint::one();
            assert!(!maximum.is_zero(), "refresh must contain nonzero fresh noise");
            assert!(
                maximum < bound,
                "actual coefficient norm {maximum} must be strictly below B_e={bound}"
            );
            let decoded = result
                .outputs
                .iter()
                .filter(|(name, _)| name.starts_with("decoded_"))
                .collect::<Vec<_>>();
            assert_eq!(decoded.len(), columns);
            assert!(decoded.iter().all(|(_, v)| matches!(v, RuntimeValue::Bool(false))));
            println!(
                "round_trip_passed online_run={online_run} N={n} depth={} plaintext_exponent={plaintext_exponent} max_noise={maximum} strict_fresh_bound={bound} mask_digits={mask_digits} fresh_digits={fresh_digits}",
                candidate.crt_depth
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn gpu_setup_uses_derived_native_q_p_and_readout() {
        let profile = SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8)
            .expect("native PRF profile");
        assert_eq!(profile.q().expect("Q"), 64);
        assert_eq!(profile.p().expect("P"), 16);
        let program = mxx_exponent_lut::noise_refresh::prf::SparseLwrPrfProgram::new(profile, 1, 1)
            .expect("native PRF program");
        assert!(program.has_only_native_accumulation());
    }

    #[test]
    #[serial_test::serial]
    fn gpu_artifact_backed_refresh_replays_across_wave_settings() {
        let (poly_parameters, bundle, total_labels) = tiny_native_refresh_bundle();
        let noise = bundle.simulate_noise().expect("tiny fixture noise authority");
        assert!(noise.refresh.fresh_error_accepted, "strict fresh-error bound must hold");
        assert!(noise.refresh.mask_coverage_accepted, "tiny mask digits must cover native Q");
        // This deliberately tiny arithmetic fixture is not a security parameter
        // profile. Selected-profile authority acceptance is exercised by the
        // parameter-search integration test; this runtime test covers the same
        // artifact-backed execution path and the strict fresh-error invariant.
        let preprocessing = bundle
            .preprocessing_graph()
            .validate_with_manifests(&ParamEnv::default(), bundle.preprocessing_manifests())
            .expect("validate preprocessing graph");
        let (moduli, _, _) = poly_parameters.to_crt();
        let gpu_parameters = mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams::new(
            poly_parameters.ring_dimension(),
            moduli,
            poly_parameters.base_bits(),
            None,
        );
        let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
        assert!(!devices.is_empty(), "GPU required for artifact-backed smoke test");
        let mut backend =
            gpu_backend_on(bundle.ring_parameters(gpu_parameters).expect("refresh rings"), devices);
        let mut inputs = bundle.selector_runtime_inputs(&mut backend).expect("selector inputs");
        inputs.insert(
            "refresh-parameter-search-hash-key".to_owned(),
            RuntimeValue::Bytes(vec![0x94; 32]),
        );
        let monomials = (0..total_labels)
            .map(|index| {
                let exponent = 2 * (index % 32);
                let monomial = DCRTPoly::const_rotate_poly(
                    &poly_parameters,
                    if exponent >= 32 { exponent - 32 } else { exponent },
                );
                let monomial = if exponent >= 32 { -monomial } else { monomial };
                let matrix = DCRTPolyMatrix::from_poly_vec_row(&poly_parameters, vec![monomial]);
                RuntimeValue::HostMatrix {
                    matrix_type: ConcreteMatrixType::scalar(
                        poly_parameters.modulus().as_ref().clone().into(),
                        poly_parameters.ring_dimension() as usize,
                    ),
                    bytes: matrix.to_cpu_staging_bytes().into(),
                }
            })
            .collect();
        inputs.insert(
            "refresh-parameter-search-native-monomials".to_owned(),
            RuntimeValue::IndexedFamily(monomials),
        );
        let mut store = MemoryArtifactStore::default();
        let result =
            execute_in_session(&preprocessing, &mut backend, inputs, &mut store, [0x53; 32])
                .expect("GPU preprocessing execution");
        let production = result.production_id.expect("preprocessing production");
        let manifest =
            store.manifest(&production).expect("preprocessing artifact manifest").clone();
        let encrypt_binding = bundle
            .encrypt_graph_for(production.clone(), manifest.clone())
            .expect("artifact-backed encryption graph");
        let encrypt = encrypt_binding.graph();
        let encrypted =
            execute_in_session(&encrypt, &mut backend, BTreeMap::new(), &mut store, [0x54; 32])
                .expect("GPU encryption execution");
        let encrypt_production = encrypted.production_id.expect("encryption production");
        let encrypt_manifest =
            store.manifest(&encrypt_production).expect("encryption artifact manifest").clone();
        let online = bundle
            .online_graph_for(
                production,
                manifest,
                encrypt_production,
                encrypt_manifest,
                &encrypt_binding,
            )
            .expect("artifact-backed online graph");
        let online_inputs = BTreeMap::new();
        let mut transcript = TranscriptRecorder::default();
        let first = execute_with_config(
            &online,
            &mut backend,
            online_inputs.clone(),
            &mut store,
            SamplingMode::Record(&mut transcript),
            ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(1).unwrap(),
                ..Default::default()
            },
        )
        .expect("GPU online execution at narrow wave");
        let replayer = transcript.into_replayer();
        let second = execute_with_config(
            &online,
            &mut backend,
            online_inputs,
            &mut store,
            SamplingMode::Replay(&replayer),
            ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(2).unwrap(),
                ..Default::default()
            },
        )
        .expect("GPU online replay at wider wave");
        let decoded = |result: &mxx_runtime::ExecutionResult<_>| {
            result
                .outputs
                .iter()
                .filter(|(name, _)| name.starts_with("decoded_"))
                .map(|(_, value)| matches!(value, RuntimeValue::Bool(false)))
                .collect::<Vec<_>>()
        };
        assert!(!decoded(&first).is_empty());
        assert!(decoded(&first).iter().all(|value| *value));
        assert_eq!(decoded(&first), decoded(&second), "wave setting changed decoded output");
    }
}

mod rns_bgg {
    //! Shared graph fixtures for RNS-BGG integration and backend regression tests.
    use bigdecimal::BigDecimal;
    use mxx_dsl::{DslContext, Family, GraphValue, GraphValueSchema, Mat};
    use mxx_exponent_lut::{
        layout::ExpLayout,
        noise_refresh::{
            pbc::{PbcLayoutId, derive_lwr_vector},
            prf::{
                PreparedSparseLwrPrf, PrfSelectionInputs, SparseLwrPrfProfile, SparseLwrPrfProgram,
            },
            refresh::{DirectRefreshChannel, PreparedDirectRefresh, RefreshPrfLabelIndex},
        },
        program::{FamilyRange, LutTable, ProgramValue},
        rns_bgg::{
            RnsBggEncoding, RnsBggEncodingProgramValue, RnsBggNoiseBound, RnsBggOnlineMasks,
            RnsBggParameters, RnsBggPlanRequest, RnsBggPublicProgramValue, RnsBggRefreshLevel,
            RnsBggRhsPlaintextFamily, RnsBggRow, RnsBggScheme,
        },
    };
    use mxx_ir_core::{
        ParamEnv, ValidatedGraph,
        artifact::{ArtifactConfidentiality, Manifest, ProductionId},
    };
    use mxx_primitives::poly::PolyParams;
    use mxx_runtime::{
        Backend, RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    type RnsBggEncodingSchema = <RnsBggEncoding as GraphValue>::Schema;
    type RnsBggOnlineMasksSchema = <RnsBggOnlineMasks as GraphValue>::Schema;
    type RnsBggRowSchema = <RnsBggRow as GraphValue>::Schema;
    type RnsBggDecoderSchema = <Family<RnsBggRow> as GraphValue>::Schema;

    pub struct RnsBggRoundTrip {
        pub parameters: RnsBggParameters,
        pub producer: ValidatedGraph,
        pub benchmark_preprocessing: ValidatedGraph,
        pub benchmark_encrypt: ValidatedGraph,
        pub benchmark_online: ValidatedGraph,
        pub encrypt: Box<
            dyn Fn(
                ProductionId,
                Manifest,
            ) -> (
                ValidatedGraph,
                ValidatedGraph,
                (
                    RnsBggEncodingSchema,
                    RnsBggEncodingSchema,
                    RnsBggOnlineMasksSchema,
                    RnsBggOnlineMasksSchema,
                    RnsBggOnlineMasksSchema,
                    RnsBggDecoderSchema,
                    RnsBggRowSchema,
                    RnsBggRowSchema,
                ),
            ),
        >,
        pub online: Box<
            dyn Fn(
                ProductionId,
                Manifest,
                ProductionId,
                Manifest,
            ) -> (ValidatedGraph, ValidatedGraph),
        >,
        pub bounds: BTreeMap<String, (BigUint, BigUint)>,
        pub residue_level: Option<usize>,
        pub noise_report: serde_json::Value,
    }

    /// Construct the production refresh graphs from an accepted numerical search.
    /// Public label polynomials and one-hot bits are structural inputs, as in the
    /// BGG estimator; no host vector is allocated per PRF output or coefficient.
    pub fn benchmark_refresh(
        path: &std::path::Path,
    ) -> (RnsBggParameters, usize, ValidatedGraph, ValidatedGraph, ValidatedGraph, serde_json::Value)
    {
        use mxx_exponent_lut::rns_bgg::RnsBggRefreshPlan;
        use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let meta = report.get("numerical_run").unwrap_or(&report);
        let candidate = &report["selected_candidate"];
        assert_eq!(candidate["correctness_accepted"], true);
        let n = meta["n"].as_u64().unwrap() as u32;
        let bits = meta["crt_bits"].as_u64().unwrap() as usize;
        let base_bits = meta["base_bits"].as_u64().unwrap() as u32;
        let sigma_text = report["phase2"]["error_sigma"]
            .as_str()
            .or_else(|| report["error_sigma"].as_str())
            .expect("accepted report must contain phase2.error_sigma");
        let bound_text = report["phase2"]["encoding_error_bound"]
            .as_str()
            .or_else(|| report["encoding_error_bound"].as_str())
            .expect("accepted report must contain phase2.encoding_error_bound");
        let encoding_error_sigma: BigDecimal =
            sigma_text.parse().expect("accepted report must contain a valid error_sigma");
        let encoding_error_bound: BigUint =
            bound_text.parse().expect("accepted report must contain a valid encoding_error_bound");
        assert_eq!(
            encoding_error_bound,
            mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&encoding_error_sigma),
            "accepted report sigma/bound mismatch"
        );
        let primes: Vec<Vec<u64>> =
            serde_json::from_value(candidate["level_primes"].clone()).unwrap();
        let parameters = RnsBggParameters::new(
            primes
                .into_iter()
                .map(|p| DCRTPolyParams::new(n, p.len(), bits, base_bits, Some(p), None))
                .collect(),
            encoding_error_sigma,
            1,
        )
        .unwrap();
        assert_eq!(parameters.encoding_error_bound(), &encoding_error_bound);
        let counts: Vec<usize> = serde_json::from_value(meta["selector_counts"].clone()).unwrap();
        let terminal = counts.len();
        let output_level = terminal + 1;
        let q_lwr = meta["lwr_q"].as_u64().unwrap() as usize;
        let p_lwr = meta["lwr_p"].as_u64().unwrap() as usize;
        let width_log2 = q_lwr.trailing_zeros() - 1;
        let layout = ExpLayout { stride_log2: n.trailing_zeros() - width_log2, width_log2 };
        let data_layout = ExpLayout { stride_log2: 0, width_log2: n.trailing_zeros() };
        let profile = SparseLwrPrfProfile::new(layout, p_lwr / 2).unwrap();
        let program =
            SparseLwrPrfProgram::new(profile, *counts.iter().max().unwrap(), terminal).unwrap();
        let refresh_plan: RnsBggRefreshPlan =
            serde_json::from_value(candidate["refresh"]["Ok"].clone()).unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let seed = scheme.ring(0).unwrap().bytes_input("hash-key", 32);
        let r = scheme.sample_payload_secret().unwrap();
        let initial_key = scheme
            .sample_input_key(0, seed.clone(), b"initial".as_slice(), rand::random())
            .unwrap();
        let initial_material =
            scheme.prepare_input(&r, &initial_key, scheme.scalar(0, 1).unwrap()).unwrap();
        let data_key = scheme
            .sample_input_key(terminal, seed.clone(), b"data".as_slice(), rand::random())
            .unwrap();
        let data_material = scheme
            .prepare_input(
                &r,
                &data_key,
                scheme.ring(terminal).unwrap().polynomial([0.into(), 1.into()]),
            )
            .unwrap();
        let output_key = scheme
            .sample_input_key(output_level, seed.clone(), b"output".as_slice(), rand::random())
            .unwrap();
        let topology = refresh_plan
            .digits
            .topology(&scheme, &(output_key.clone(), None), &ParamEnv::default())
            .unwrap();
        let labels = RefreshPrfLabelIndex::new(rand::random(), &topology).unwrap();
        assert_eq!(BigUint::from(labels.len()), refresh_plan.distinct_prf_outputs);
        let blocks = counts
            .iter()
            .enumerate()
            .map(|(step, count)| {
                let ring = scheme.ring(step + 1).unwrap();
                let rhs = RnsBggRhsPlaintextFamily {
                    payloads: ring.input_family(format!("selector-bits-{step}"), *count, (1, 1)),
                    id: rand::random(),
                };
                let selectors = scheme
                    .sample_private_rhs_family(
                        &r,
                        step,
                        &rhs,
                        seed.clone(),
                        format!("selectors-{step}").into_bytes(),
                        rand::random(),
                    )
                    .unwrap();
                PrfSelectionInputs {
                    selectors,
                    labels: ring.input_family(
                        format!("prf-labels-{step}"),
                        labels.len() * count,
                        (1, 1),
                    ),
                    active: FamilyRange::bounded(0, *count, *count).unwrap(),
                    id: rand::random(),
                }
            })
            .collect::<Vec<_>>();
        let (prf_public, prf) = PreparedSparseLwrPrf::new(
            &scheme,
            program.clone(),
            RnsBggPublicProgramValue::Scalar((initial_key.clone(), None)),
            &blocks,
        )
        .unwrap();
        let data_bank = scheme
            .sample_mask_bank(
                terminal,
                data_layout,
                seed.clone(),
                b"data-bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let prf_bank = scheme
            .sample_mask_bank(
                terminal,
                layout,
                seed.clone(),
                b"prf-bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let q = parameters.level(output_level).unwrap().modulus();
        let channels = parameters
            .level(output_level)
            .unwrap()
            .to_crt()
            .0
            .into_iter()
            .map(|prime| {
                let h = q.as_ref() / prime;
                DirectRefreshChannel {
                    level: RnsBggRefreshLevel { output_level, prime: prime.into() },
                    output_scale: scheme.scalar(output_level, h.clone()).unwrap(),
                    data: scheme
                        .sample_refresh_data_material(
                            &r,
                            data_layout,
                            &h,
                            &data_bank,
                            seed.clone(),
                            format!("data-{prime}").into_bytes(),
                            rand::random(),
                        )
                        .unwrap(),
                    spacing: h,
                }
            })
            .collect();
        let prepared = PreparedDirectRefresh::new(
            &scheme,
            &RnsBggPublicProgramValue::Scalar((data_key.clone(), None)),
            &prf_public,
            &program.readout,
            &refresh_plan.digits,
            labels,
            &r,
            &prf_bank,
            channels,
            output_key.matrix().clone(),
            scheme.prepare_refresh_recomposition(output_key).unwrap(),
            seed,
            b"refresh".as_slice().into(),
            rand::random(),
        )
        .unwrap();
        let (producer, refresh_descriptor) = prepared
            .export(&scheme, DslContext::new("rns-bgg-refresh-benchmark"), "refresh")
            .unwrap();
        let (producer, prf_descriptor) = prf.export(&scheme, producer, "prf").unwrap();
        let (producer, initial_descriptor) = scheme
            .export_input_material(&initial_key, &initial_material, producer, "initial")
            .unwrap();
        let (producer, data_descriptor) =
            scheme.export_input_material(&data_key, &data_material, producer, "data").unwrap();
        let producer = producer.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::encoding::spec_hash(&producer.source, &producer.bindings)
                .unwrap(),
            execution_nonce: rand::random(),
        };
        let manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &producer)
                .unwrap();
        let refresh = PreparedDirectRefresh::import(
            &scheme,
            &refresh_descriptor,
            production.clone(),
            refresh_descriptor.identity().unwrap(),
        )
        .unwrap();
        let prf = PreparedSparseLwrPrf::import(
            &scheme,
            program,
            &prf_descriptor,
            production.clone(),
            prf_descriptor.identity().unwrap(),
        )
        .unwrap();
        let (initial_key, initial_material) = scheme
            .import_input_material(&initial_descriptor, production.clone(), *initial_key.id(), 0)
            .unwrap();
        let (data_key, data_material) = scheme
            .import_input_material(&data_descriptor, production.clone(), *data_key.id(), terminal)
            .unwrap();
        // Encryption is its own graph: both the initial PRF input and the data
        // payload are encoded here, then imported as private artifacts by the
        // online graph below.  This keeps the benchmark and round-trip stage
        // boundaries identical.
        let s = scheme.sample_mask_secret().unwrap();
        let encrypted_initial = scheme.encode(&s, &initial_key, &initial_material).unwrap();
        let encrypted_data = scheme.encode(&s, &data_key, &data_material).unwrap();
        let initial_schema = encrypted_initial.schema();
        let data_schema = encrypted_data.schema();
        let bank = |index: usize| match &refresh.slots()[0].terms()[index] {
            ProgramValue::Scalar(term) => term.mask_bank(),
            ProgramValue::Family(terms) => terms.at(0).mask_bank(),
        };
        let data_helpers = scheme.prepare_online_masks(&s, &bank(0)).unwrap();
        let prf_helpers = scheme.prepare_online_masks(&s, &bank(1)).unwrap();
        let decoders = Family::pack(refresh.prepare_decoders(&scheme, &s, &2)).unwrap();
        let encrypt = DslContext::new("rns-bgg-refresh-encrypt-benchmark")
            .private_output("data-helpers", data_helpers.clone())
            .unwrap()
            .private_output("prf-helpers", prf_helpers.clone())
            .unwrap()
            .private_output("decoders", decoders.clone())
            .unwrap()
            .private_output("encrypted-initial", encrypted_initial.row().clone())
            .unwrap()
            .private_output("encrypted-data", encrypted_data.row().clone())
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .unwrap();
        let encrypt_production = ProductionId {
            spec_hash: mxx_ir_core::encoding::spec_hash(&encrypt.source, &encrypt.bindings)
                .unwrap(),
            execution_nonce: rand::random(),
        };
        let encrypt_manifest =
            mxx_ir_core::artifact::export_validated_manifest(encrypt_production.clone(), &encrypt)
                .unwrap();
        let initial = initial_schema
            .artifact_input(
                encrypt_production.clone(),
                "encrypted-initial",
                ArtifactConfidentiality::Private,
            )
            .unwrap();
        let digits =
            prf.evaluate(&scheme, RnsBggEncodingProgramValue::Scalar((initial, None))).unwrap();
        let data = data_schema
            .artifact_input(
                encrypt_production.clone(),
                "encrypted-data",
                ArtifactConfidentiality::Private,
            )
            .unwrap();
        let data = RnsBggEncodingProgramValue::Scalar((data, None));
        let data_helpers = data_helpers
            .schema()
            .artifact_input(
                encrypt_production.clone(),
                "data-helpers",
                ArtifactConfidentiality::Private,
            )
            .unwrap();
        let prf_helpers = prf_helpers
            .schema()
            .artifact_input(
                encrypt_production.clone(),
                "prf-helpers",
                ArtifactConfidentiality::Private,
            )
            .unwrap();
        let decoders = decoders
            .schema()
            .artifact_input(
                encrypt_production.clone(),
                "decoders",
                ArtifactConfidentiality::Private,
            )
            .unwrap();
        let (refreshed, _) = refresh
            .evaluate(
                &scheme,
                &data,
                &digits,
                &data_helpers,
                &prf_helpers,
                (0..refresh.slots().len()).map(|index| decoders.at(index)).collect(),
            )
            .unwrap();
        let online = DslContext::new("rns-bgg-refresh-online-benchmark")
            .output("refreshed", refreshed.row().clone())
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest), (encrypt_production, encrypt_manifest)]),
            )
            .unwrap();
        (parameters, output_level, producer, encrypt, online, candidate.clone())
    }

    pub fn refresh(
        average: Option<&mxx_exponent_lut::noise_refresh::simulation::AverageCaseConfig>,
    ) -> RnsBggRoundTrip {
        let n: u32 = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(4);
        let layout = ExpLayout { stride_log2: 0, width_log2: n.trailing_zeros() };
        let encoding_error_sigma = BigDecimal::from(4u8);
        let profile = SparseLwrPrfProfile::new(layout, 2).unwrap();
        let program = SparseLwrPrfProgram::new(profile, 2, 1).unwrap();
        // Trace the exact frozen product schedule consumed by preparation and
        // cached evaluation, including every ciphertext in the composed RHS.
        let execution = program.accumulation.clone();
        let data_noise = mxx_exponent_lut::rns_bgg::RnsBggNoiseValue {
            level: 1,
            bound: RnsBggNoiseBound { error: 26u8.into(), payload_l1: 1u8.into() },
            payload_l2_squared: 1u8.into(),
            moments: Some(mxx_exponent_lut::noise_utils::NoiseMoments {
                mean_numerator: 0.into(),
                mean_denominator: 1u8.into(),
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
        let refresh_request = mxx_exponent_lut::rns_bgg::RnsBggRefreshPlanRequest {
            data_branches: n as usize,
            prf_branches: n as usize,
            base: 4,
            fresh_support: 4u8.into(),
            fresh_bound: 2u8.into(),
            decoder_eta: 2,
            observations: 8u8.into(),
            statistical_bits: std::env::var(
                "MXX_EXPONENT_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS",
            )
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16),
        };
        let mut chosen = None;
        let mut last_error = None;
        // The refresh output must fit its full wide-coordinate mask fixed point,
        // in addition to the preceding PRF and following LUT. Choose before
        // constructing ciphertext graphs so rejected candidates allocate no graph.
        for terminal_depth in 2..=29 {
            let candidate = RnsBggParameters::plan_fixture(&RnsBggPlanRequest {
                ring_dimension: n,
                widths: vec![1, n as usize, 1],
                encoding_error_sigma: encoding_error_sigma.clone(),
                crt_bits: 30,
                terminal_depth,
                max_crt_depth: 32,
                terminal_value_bound: 32u8.into(),
                operations_per_drop: 1,
            });
            let (parameters, _) = match candidate {
                Ok(value) => value,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            let trace = trace_prf(&parameters, &execution, &[2]);
            match parameters.plan_refresh(
                &refresh_request,
                &data_noise,
                &trace.wires[&execution.outputs()[0]],
                average,
            ) {
                Ok(refresh_plan) => {
                    chosen = Some((parameters, refresh_plan));
                    break;
                }
                Err(error) => last_error = Some(error),
            }
        }
        let (parameters, refresh_plan) = chosen.unwrap_or_else(|| {
            panic!("no RNS-BGG refresh chain within the fixture budget: {last_error:?}")
        });
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let high = scheme.ring(0).unwrap();
        let terminal_ring = scheme.ring(1).unwrap();
        let output_ring = scheme.ring(2).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let r = scheme.sample_payload_secret().unwrap();
        let initial_key = scheme
            .sample_input_key(0, seed.clone(), b"initial".as_slice(), rand::random())
            .unwrap();
        let initial_material =
            scheme.prepare_input(&r, &initial_key, scheme.scalar(0, 1).unwrap()).unwrap();
        let data_key =
            scheme.sample_input_key(1, seed.clone(), b"data".as_slice(), rand::random()).unwrap();
        let data_material = scheme
            .prepare_input(&r, &data_key, terminal_ring.polynomial([0.into(), 1.into()]))
            .unwrap();
        let output_key =
            scheme.sample_input_key(2, seed.clone(), b"output".as_slice(), rand::random()).unwrap();
        let plan = &refresh_plan.digits;
        let data_bound = &data_noise.bound;
        let topology =
            plan.topology(&scheme, &(output_key.clone(), None), &ParamEnv::default()).unwrap();
        let labels = RefreshPrfLabelIndex::new(rand::random(), &topology).unwrap();
        assert_eq!(BigUint::from(labels.len()), refresh_plan.distinct_prf_outputs);
        let selected = usize::from(rand::random::<bool>());
        let layout_id = PbcLayoutId(rand::random());
        let vectors = (0..labels.len())
            .map(|index| {
                derive_lwr_vector(
                    layout_id,
                    &labels.label(index).unwrap().canonical_bytes(),
                    2,
                    profile.q().unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let labels_family = Family::pack(
            vectors
                .iter()
                .flat_map(|vector| {
                    vector.iter().map(|v| layout.mono(&terminal_ring, &BigInt::from(*v)).unwrap())
                })
                .collect(),
        )
        .unwrap();
        let rhs = RnsBggRhsPlaintextFamily {
            payloads: Family::pack(
                (0..2).map(|slot| scheme.scalar(1, u8::from(slot == selected)).unwrap()).collect(),
            )
            .unwrap(),
            id: rand::random(),
        };
        let selectors = scheme
            .sample_private_rhs_family(
                &r,
                0,
                &rhs,
                seed.clone(),
                b"selectors".as_slice(),
                rand::random(),
            )
            .unwrap();
        let (prf_public, prepared_prf) = PreparedSparseLwrPrf::new(
            &scheme,
            program.clone(),
            RnsBggPublicProgramValue::Scalar((initial_key.clone(), None)),
            &[PrfSelectionInputs {
                selectors,
                labels: labels_family,
                active: FamilyRange::bounded(0, 2, 2).unwrap(),
                id: rand::random(),
            }],
        )
        .unwrap();
        let terminal_bank = scheme
            .sample_mask_bank(1, layout, seed.clone(), b"terminal-bank".as_slice(), rand::random())
            .unwrap();
        let q = parameters.level(2).unwrap().modulus();
        let channels = parameters
            .level(2)
            .unwrap()
            .to_crt()
            .0
            .into_iter()
            .map(|prime| {
                let h = q.as_ref() / prime;
                DirectRefreshChannel {
                    level: RnsBggRefreshLevel { output_level: 2, prime: prime.into() },
                    output_scale: scheme.scalar(2, h.clone()).unwrap(),
                    data: scheme
                        .sample_refresh_data_material(
                            &r,
                            layout,
                            &h,
                            &terminal_bank,
                            seed.clone(),
                            format!("data-{prime}").into_bytes(),
                            rand::random(),
                        )
                        .unwrap(),
                    spacing: h,
                }
            })
            .collect();
        let public_data = RnsBggPublicProgramValue::Scalar((data_key.clone(), None));
        let prepared = PreparedDirectRefresh::new(
            &scheme,
            &public_data,
            &prf_public,
            &program.readout,
            plan,
            labels.clone(),
            &r,
            &terminal_bank,
            channels,
            output_key.matrix().clone(),
            scheme.prepare_refresh_recomposition(output_key.clone()).unwrap(),
            seed.clone(),
            b"refresh".as_slice().into(),
            rand::random(),
        )
        .unwrap();
        assert_eq!(prepared_prf.program().id(), execution.id());
        for (slot, error) in prepared.slots().iter().zip(&refresh_plan.exposed_errors) {
            let h = q.as_ref() / slot.level.prime.magnitude();
            assert!(BigUint::from(8u8) + error < h / 2u8);
        }
        let next_bank = scheme
            .sample_mask_bank(
                2,
                ExpLayout { stride_log2: 0, width_log2: 0 },
                seed.clone(),
                b"next-bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        // The public helper contract requires a concrete unary table. A
        // width-one table with output value zero is the identity polynomial;
        // its output width is the PRF's declared output modulus.
        let identity_table = LutTable::unary(1, program.profile().p().unwrap(), vec![0]).unwrap();
        let next_material = scheme
            .sample_lut_material(
                &r,
                &identity_table,
                &next_bank,
                seed,
                b"next".as_slice(),
                rand::random(),
            )
            .unwrap();
        let next = scheme.prepare_lut(&output_key, &next_material, rand::random()).unwrap();
        let next_binding = next.binding().clone();
        let next_input_id = *output_key.id();
        let next_output_id = *next.output_key().id();
        let (producer, refresh_descriptor) = prepared
            .export(&scheme, DslContext::new("rns_bgg-refresh-preprocess"), "refresh")
            .unwrap();
        let (producer, prf_descriptor) = prepared_prf.export(&scheme, producer, "prf").unwrap();
        let (producer, initial_descriptor) = scheme
            .export_input_material(&initial_key, &initial_material, producer, "initial")
            .unwrap();
        let (producer, data_descriptor) =
            scheme.export_input_material(&data_key, &data_material, producer, "data").unwrap();
        let (mut producer, next_descriptor) =
            scheme.export_prepared_lut(&next, producer, "next").unwrap();
        // Freeze the production preprocessing graph before adding the
        // correctness-only oracle outputs below. The benchmark must measure
        // only the cryptographic preprocessing work.
        let benchmark_preprocessing =
            producer.clone().build().unwrap().validate(&ParamEnv::default()).unwrap();
        // Private correctness oracle, separate from the public refresh bundle.
        for level in [2, 3] {
            let ring = scheme.ring(level).unwrap();
            let payload = scheme.reduce_payload_secret_modulus(&r, level).unwrap();
            let gadget = if level == 2 {
                let low = parameters.level(level + 1).unwrap();
                scheme
                    .ring(level + 1)
                    .unwrap()
                    .gadget(1, 1u64 << low.base_bits(), low.modulus_digits())
                    .centered_extend(BigInt::from(
                        parameters.level(level).unwrap().modulus().as_ref().clone(),
                    ))
            } else {
                ring.identity(1)
            };
            let expected = scheme.scalar(level, parameters.scale(level).unwrap().clone()).unwrap() *
                ring.polynomial([0.into(), 1.into()]) *
                RnsBggRow::new([std::array::from_fn(|c| {
                    payload.entry(0, c).clone() * gadget.clone()
                })])
                .unwrap();
            for c in 0..2 {
                producer = producer
                    .private_output(format!("oracle-{level}-{c}"), expected.entry(0, c).clone())
                    .unwrap();
            }
        }
        let producer = producer.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::encoding::spec_hash(&producer.source, &producer.bindings)
                .unwrap(),
            execution_nonce: rand::random(),
        };
        let manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &producer)
                .unwrap();
        let oracle_columns =
            [parameters.component_columns(2).unwrap(), parameters.component_columns(3).unwrap()];
        let fresh_count = topology.fresh_digit_count;
        let fresh_weights = refresh_plan.digits.weights(fresh_count);
        let encryption_secret = scheme.sample_mask_secret().unwrap();
        let encrypt_secret = encryption_secret.clone();
        let encrypt_scheme = scheme.clone();
        let encrypt_initial_descriptor = initial_descriptor.clone();
        let encrypt_data_descriptor = data_descriptor.clone();
        let encrypt_initial_key_id = *initial_key.id();
        let encrypt_data_key_id = *data_key.id();
        let encrypt_refresh_descriptor = refresh_descriptor.clone();
        let encrypt_next_descriptor = next_descriptor.clone();
        let encrypt_next_binding = next_binding.clone();
        let encrypt = Box::new(move |production: ProductionId, manifest: Manifest| {
            let (initial_key, initial_material) = encrypt_scheme
                .import_input_material(
                    &encrypt_initial_descriptor,
                    production.clone(),
                    encrypt_initial_key_id,
                    0,
                )
                .unwrap();
            let (data_key, data_material) = encrypt_scheme
                .import_input_material(
                    &encrypt_data_descriptor,
                    production.clone(),
                    encrypt_data_key_id,
                    1,
                )
                .unwrap();
            let initial =
                encrypt_scheme.encode(&encrypt_secret, &initial_key, &initial_material).unwrap();
            let data = encrypt_scheme.encode(&encrypt_secret, &data_key, &data_material).unwrap();
            let initial_schema = initial.schema();
            let data_schema = data.schema();
            let encrypt_refresh = PreparedDirectRefresh::import(
                &encrypt_scheme,
                &encrypt_refresh_descriptor,
                production.clone(),
                encrypt_refresh_descriptor.identity().unwrap(),
            )
            .unwrap();
            let encrypt_next = encrypt_scheme
                .import_prepared_lut(
                    &encrypt_next_descriptor,
                    production.clone(),
                    &encrypt_next_binding,
                    next_input_id,
                    next_output_id,
                )
                .unwrap();
            let encrypt_bank0 = match &encrypt_refresh.slots()[0].terms()[0] {
                ProgramValue::Scalar(term) => term.mask_bank(),
                ProgramValue::Family(terms) => terms.at(0).mask_bank(),
            };
            let encrypt_bank1 = match &encrypt_refresh.slots()[0].terms()[1] {
                ProgramValue::Scalar(term) => term.mask_bank(),
                ProgramValue::Family(terms) => terms.at(0).mask_bank(),
            };
            let encrypt_decoders =
                encrypt_refresh.prepare_decoders(&encrypt_scheme, &encrypt_secret, &2);
            let data_helpers =
                encrypt_scheme.prepare_online_masks(&encrypt_secret, &encrypt_bank0).unwrap();
            let prf_helpers =
                encrypt_scheme.prepare_online_masks(&encrypt_secret, &encrypt_bank1).unwrap();
            let next_helpers = encrypt_scheme
                .prepare_online_masks(&encrypt_secret, &encrypt_next.mask_bank())
                .unwrap();
            let decoders = Family::pack(encrypt_decoders.clone()).unwrap();
            let oracle_mask_2 =
                encrypt_scheme.reduce_mask_secret_modulus(&encrypt_secret, 2).unwrap();
            let oracle_mask_3 =
                encrypt_scheme.reduce_mask_secret_modulus(&encrypt_secret, 3).unwrap();
            let graph = DslContext::new("rns-bgg-refresh-encrypt")
                .private_output("data-helpers", data_helpers.clone())
                .unwrap()
                .private_output("prf-helpers", prf_helpers.clone())
                .unwrap()
                .private_output("next-helpers", next_helpers.clone())
                .unwrap()
                .private_output("decoders", decoders.clone())
                .unwrap()
                .private_output("oracle-mask-2", oracle_mask_2.clone())
                .unwrap()
                .private_output("oracle-mask-3", oracle_mask_3.clone())
                .unwrap()
                .private_output("encrypted-initial", initial.row().clone())
                .unwrap()
                .private_output("encrypted-data", data.row().clone())
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production.clone(), manifest.clone())]),
                )
                .unwrap();
            let benchmark_graph = DslContext::new("rns-bgg-refresh-encrypt-benchmark")
                .private_output("data-helpers", data_helpers.clone())
                .unwrap()
                .private_output("prf-helpers", prf_helpers.clone())
                .unwrap()
                .private_output("next-helpers", next_helpers.clone())
                .unwrap()
                .private_output("decoders", decoders.clone())
                .unwrap()
                .private_output("encrypted-initial", initial.row().clone())
                .unwrap()
                .private_output("encrypted-data", data.row().clone())
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production, manifest)]),
                )
                .unwrap();
            (
                graph,
                benchmark_graph,
                (
                    initial_schema,
                    data_schema,
                    data_helpers.schema(),
                    prf_helpers.schema(),
                    next_helpers.schema(),
                    decoders.schema(),
                    oracle_mask_2.schema(),
                    oracle_mask_3.schema(),
                ),
            )
        });
        let (
            _,
            benchmark_encrypt,
            (
                initial_schema,
                data_schema,
                data_helpers_schema,
                prf_helpers_schema,
                next_helpers_schema,
                decoders_schema,
                oracle_mask_2_schema,
                oracle_mask_3_schema,
            ),
        ) = encrypt(production.clone(), manifest.clone());
        let benchmark_encrypt_production = ProductionId {
            spec_hash: mxx_ir_core::encoding::spec_hash(
                &benchmark_encrypt.source,
                &benchmark_encrypt.bindings,
            )
            .unwrap(),
            execution_nonce: rand::random(),
        };
        let benchmark_encrypt_manifest = mxx_ir_core::artifact::export_validated_manifest(
            benchmark_encrypt_production.clone(),
            &benchmark_encrypt,
        )
        .unwrap();
        let online_scheme = scheme.clone();
        let online = Box::new(
            move |production: ProductionId,
                  manifest: Manifest,
                  encrypt_production: ProductionId,
                  encrypt_manifest: Manifest| {
                for (name, artifact) in &manifest.artifacts {
                    if name.starts_with("refresh") || name.starts_with("prf") {
                        assert_eq!(artifact.confidentiality, ArtifactConfidentiality::Public);
                    }
                }
                let descriptor =
                    serde_json::from_slice(&serde_json::to_vec(&refresh_descriptor).unwrap())
                        .unwrap();
                let refresh = PreparedDirectRefresh::import(
                    &online_scheme,
                    &descriptor,
                    production.clone(),
                    refresh_descriptor.identity().unwrap(),
                )
                .unwrap();
                let prf = PreparedSparseLwrPrf::import(
                    &online_scheme,
                    program.clone(),
                    &prf_descriptor,
                    production.clone(),
                    prf_descriptor.identity().unwrap(),
                )
                .unwrap();
                let next = online_scheme
                    .import_prepared_lut(
                        &next_descriptor,
                        production.clone(),
                        &next_binding,
                        *output_key.id(),
                        next_output_id,
                    )
                    .unwrap();
                let initial = initial_schema
                    .artifact_input(
                        encrypt_production.clone(),
                        "encrypted-initial",
                        ArtifactConfidentiality::Private,
                    )
                    .unwrap();
                let initial = RnsBggEncodingProgramValue::Scalar((initial, None));
                let digits = prf.evaluate(&online_scheme, initial).unwrap();
                let data = data_schema
                    .artifact_input(
                        encrypt_production.clone(),
                        "encrypted-data",
                        ArtifactConfidentiality::Private,
                    )
                    .unwrap();
                let data = RnsBggEncodingProgramValue::Scalar((data, None));
                let data_helpers = data_helpers_schema
                    .artifact_input(
                        encrypt_production.clone(),
                        "data-helpers",
                        ArtifactConfidentiality::Private,
                    )
                    .unwrap();
                let prf_helpers = prf_helpers_schema
                    .artifact_input(
                        encrypt_production.clone(),
                        "prf-helpers",
                        ArtifactConfidentiality::Private,
                    )
                    .unwrap();
                let next_helpers = next_helpers_schema
                    .artifact_input(
                        encrypt_production.clone(),
                        "next-helpers",
                        ArtifactConfidentiality::Private,
                    )
                    .unwrap();
                let decoders = decoders_schema
                    .artifact_input(
                        encrypt_production.clone(),
                        "decoders",
                        ArtifactConfidentiality::Private,
                    )
                    .unwrap();
                let oracle_mask_2 = oracle_mask_2_schema
                    .artifact_input(
                        encrypt_production.clone(),
                        "oracle-mask-2",
                        ArtifactConfidentiality::Private,
                    )
                    .unwrap();
                let oracle_mask_3 = oracle_mask_3_schema
                    .artifact_input(
                        encrypt_production.clone(),
                        "oracle-mask-3",
                        ArtifactConfidentiality::Private,
                    )
                    .unwrap();
                let refreshed = refresh
                    .evaluate(
                        &online_scheme,
                        &data,
                        &digits,
                        &data_helpers,
                        &prf_helpers,
                        (0..refresh.slots().len()).map(|index| decoders.at(index)).collect(),
                    )
                    .unwrap();
                let (refreshed, _) = refreshed;
                let refreshed = refreshed;
                let (fresh_start, _) = labels.group_range(None);
                let table = program.readout.canonical_coefficients().unwrap();
                let columns = oracle_columns[0];
                let expected_fresh = (0..2)
                    .map(|block| {
                        Mat::concat(
                            mxx_ir_core::node::ConcatAxis::Columns,
                            (0..columns)
                                .map(|column| {
                                    let component = block * columns + column;
                                    output_ring.polynomial((0..n as usize).map(|coefficient| {
                                        fresh_weights
                                            .iter()
                                            .enumerate()
                                            .map(|(digit, weight)| {
                                                let index = fresh_start +
                                                    (component * n as usize + coefficient) *
                                                        fresh_count +
                                                    digit;
                                                let exponent = vectors[index][selected];
                                                let value = &table[exponent % n as usize];
                                                let value = if exponent < n as usize {
                                                    value.clone()
                                                } else {
                                                    BigInt::from(1u8) - value
                                                };
                                                value * BigInt::from(weight.clone())
                                            })
                                            .sum::<BigInt>()
                                            .into()
                                    }))
                                })
                                .collect(),
                        )
                    })
                    .collect::<Vec<_>>();
                let continued =
                    online_scheme.evaluate_lut(&refreshed, &next, &next_helpers).unwrap();
                // The estimate covers the complete production computation:
                // refresh followed by the continuation LUT.  The correctness
                // consumer below adds only private oracle residuals, which are
                // test instrumentation and are intentionally not benchmarked.
                let benchmark_online = DslContext::new("rns_bgg-refresh-online-benchmark")
                    .output("continued", continued.row().clone())
                    .unwrap()
                    .build()
                    .unwrap()
                    .validate_with_manifests(
                        &ParamEnv::default(),
                        &BTreeMap::from([
                            (production.clone(), manifest.clone()),
                            (encrypt_production.clone(), encrypt_manifest.clone()),
                        ]),
                    )
                    .unwrap();
                let mut consumer = DslContext::new("rns_bgg-refresh-online");
                let has_oracle_artifacts =
                    encrypt_manifest.artifacts.keys().any(|name| name.starts_with("oracle-mask-2"));
                if has_oracle_artifacts {
                    for (level, encoding, key) in [
                        (2, &refreshed, &refresh.recomposition().output_key),
                        (3, &continued, next.output_key()),
                    ] {
                        let ring = online_scheme.ring(level).unwrap();
                        let expected = RnsBggRow::new([std::array::from_fn(|c| {
                            ring.artifact_input(
                                production.clone(),
                                format!("oracle-{level}-{c}"),
                                (1, oracle_columns[level - 2]),
                                ArtifactConfidentiality::Private,
                            )
                        })])
                        .unwrap();
                        let oracle_mask =
                            if level == 2 { oracle_mask_2.clone() } else { oracle_mask_3.clone() };
                        let error =
                            encoding.row().clone() - oracle_mask * key.matrix().clone() + expected;
                        for c in 0..2 {
                            let residual = if level == 2 {
                                error.entry(0, c).clone() - expected_fresh[c].clone()
                            } else {
                                error.entry(0, c).clone()
                            };
                            consumer =
                                consumer.output(format!("error-{level}-{c}"), residual).unwrap();
                        }
                    }
                } else {
                    // Benchmark-only encryption deliberately omits the private
                    // correctness oracle, so its companion consumer exports the
                    // production continuation without oracle residuals.
                    consumer = consumer.output("continued", continued.row().clone()).unwrap();
                }
                let consumer = consumer
                    .build()
                    .unwrap()
                    .validate_with_manifests(
                        &ParamEnv::default(),
                        &BTreeMap::from([
                            (production.clone(), manifest),
                            (encrypt_production, encrypt_manifest),
                        ]),
                    )
                    .unwrap();
                (consumer, benchmark_online)
            },
        );
        let (_, benchmark_online) = online(
            production.clone(),
            manifest.clone(),
            benchmark_encrypt_production,
            benchmark_encrypt_manifest,
        );
        let next_bound =
            parameters.lut_noise_bound(2, data_bound, 1, &1u8.into(), 1u8.into()).unwrap();
        let bounds = (0..2)
            .flat_map(|c| {
                [
                    (
                        format!("error-2-{c}"),
                        (parameters.level(2).unwrap().modulus().as_ref().clone(), 0u8.into()),
                    ),
                    (
                        format!("error-3-{c}"),
                        (
                            parameters.level(3).unwrap().modulus().as_ref().clone(),
                            next_bound.error.clone(),
                        ),
                    ),
                ]
            })
            .collect();
        let mut noise_report = serde_json::to_value(&refresh_plan).unwrap();
        noise_report["levels"] = serde_json::Value::Array(
            parameters
                .levels()
                .iter()
                .enumerate()
                .map(|(level, ring)| {
                    let mut report = serde_json::json!({
                        "level": level,
                        "ring_dimension": ring.ring_dimension(),
                        "modulus_bits": ring.modulus().bits(),
                        "crt_depth": ring.crt_depth(),
                        "base_bits": ring.base_bits(),
                        "state_columns": 2 * parameters.component_columns(level).unwrap(),
                    });
                    if let Some(low) = parameters.levels().get(level + 1) {
                        // Fixed-chain algebraic polynomial-product counts, not kernel
                        // launches or an end-to-end timing ratio. The terminal endpoint
                        // has one column per secret component in both constructions.
                        report["private_row_product_terms"] = (4 *
                            low.modulus_digits() *
                            parameters.component_columns(level + 1).unwrap())
                        .into();
                        report["one_prime_digit_row_product_terms"] = (4 *
                            low.crt_depth() *
                            parameters
                                .levels()
                                .get(level + 2)
                                .map_or(1, |next| next.crt_depth()))
                        .into();
                    }
                    report
                })
                .collect(),
        );
        RnsBggRoundTrip {
            parameters,
            producer,
            online,
            encrypt,
            bounds,
            benchmark_preprocessing,
            benchmark_encrypt,
            benchmark_online,
            residue_level: Some(2),
            noise_report,
        }
    }

    pub fn run<B: Backend>(
        fixture: RnsBggRoundTrip,
        make_backend: impl FnOnce(&RnsBggParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &str, &BigUint, &BigUint),
    ) {
        std::fs::create_dir_all("test_data/rns_bgg_round_trip").unwrap();
        let directory = tempfile::tempdir_in("test_data/rns_bgg_round_trip").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut backend = make_backend(&fixture.parameters);
        let produced = execute(
            &fixture.producer,
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        drop(store);
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        let manifest = store.load_manifest(&production).unwrap();
        // Encryption is a distinct artifact-producing phase.  The online
        // graph can only consume the private ciphertext artifacts produced by
        // this phase, so a successful preprocess alone is not sufficient.
        let (encrypt, _, _) = (fixture.encrypt)(production.clone(), manifest.clone());
        let encrypted =
            execute(&encrypt, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        let encrypt_production = encrypted.production_id.unwrap();
        drop(store);
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        let encrypt_manifest = store.load_manifest(&encrypt_production).unwrap();
        let (consumer, _) =
            (fixture.online)(production, manifest, encrypt_production, encrypt_manifest);
        let result =
            execute(&consumer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        for (name, value) in result.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial output") };
            let (q, bound) = &fixture.bounds[&name];
            check_matrix(&matrix, &name, q, bound);
        }
    }

    pub fn trace_prf(
        parameters: &RnsBggParameters,
        execution: &mxx_exponent_lut::program::ExponentLutProgram,
        selector_counts: &[usize],
    ) -> mxx_exponent_lut::rns_bgg::RnsBggNoiseTrace {
        let inputs = BTreeMap::from([(
            *execution.inputs().keys().next().unwrap(),
            mxx_exponent_lut::rns_bgg::RnsBggNoiseValue {
                level: 0,
                bound: RnsBggNoiseBound {
                    error: parameters.encoding_error_bound().clone(),
                    payload_l1: 1u8.into(),
                },
                payload_l2_squared: 1u8.into(),
                moments: Some(mxx_exponent_lut::noise_utils::NoiseMoments {
                    mean_numerator: 0.into(),
                    mean_denominator: 1u8.into(),
                    mean_radius: mxx_exponent_lut::noise_utils::NonnegativeRational::zero(),
                    variance: mxx_exponent_lut::noise_utils::NonnegativeRational::new(
                        parameters.encoding_error_variance().numerator.clone(),
                        parameters.encoding_error_variance().denominator.clone(),
                    )
                    .unwrap(),
                }),
                public: false,
                range: None,
                one_hot: false,
            },
        )]);
        let mut rhs_bounds = BTreeMap::new();
        let mut materials = BTreeMap::new();
        for (step, gate) in execution.steps().iter().enumerate() {
            let mxx_exponent_lut::program::ProgramStep::Gate(
                mxx_exponent_lut::program::Gate::MulPrivate { rhs, .. },
            ) = gate
            else {
                panic!("composed PRF must contain direct products")
            };
            let (material, bound) =
                mxx_exponent_lut::rns_bgg::RnsBggGateNoiseMaterial::from_one_hot_labels(
                    step,
                    None,
                    &(0..selector_counts[step])
                        .map(|index| {
                            mxx_exponent_lut::rns_bgg::RnsBggNoiseAction::new(
                                mxx_exponent_lut::rns_bgg::stable_source_id(&(
                                    "noise-refresh-rhs",
                                    step,
                                    rhs,
                                    index,
                                )),
                                mxx_exponent_lut::noise_utils::RightActionNoiseGain::unit(),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            rhs_bounds.insert(*rhs, bound);
            materials
                .insert(mxx_exponent_lut::program::ProgramGateInstance::Gate { step }, material);
        }
        mxx_exponent_lut::rns_bgg::trace_program(
            parameters,
            execution,
            &mxx_exponent_lut::program::ProgramBindings::new()
                .inputs(inputs.iter().map(|(&id, value)| (id, value.clone())))
                .rhs_values(rhs_bounds.iter().map(|(&id, value)| (id, value.clone())))
                .gate_helpers(materials.iter().map(|(&id, value)| (id, value.clone()))),
            &mxx_exponent_lut::program::ProgramFamilyBindings::new(),
        )
        .unwrap()
    }
}

#[test]
fn test_rns_bgg_three_stage_graph_construction() {
    // This is intentionally backend-free: invoking all three fixture
    // closures catches missing artifact manifests during graph validation
    // before the much slower GPU round trip is attempted.
    let average = selected_average_config();
    let fixture = rns_bgg::refresh(average.as_ref());
    let benchmark_preprocessing_production = mxx_ir_core::artifact::ProductionId {
        spec_hash: mxx_ir_core::encoding::spec_hash(
            &fixture.benchmark_preprocessing.source,
            &fixture.benchmark_preprocessing.bindings,
        )
        .unwrap(),
        execution_nonce: [0x10; 32],
    };
    let benchmark_preprocessing_manifest = mxx_ir_core::artifact::export_validated_manifest(
        benchmark_preprocessing_production,
        &fixture.benchmark_preprocessing,
    )
    .unwrap();
    assert!(
        benchmark_preprocessing_manifest.artifacts.keys().all(|name| !name.starts_with("oracle-")),
        "benchmark preprocessing must exclude correctness-only oracle artifacts"
    );
    let preprocessing_production = mxx_ir_core::artifact::ProductionId {
        spec_hash: mxx_ir_core::encoding::spec_hash(
            &fixture.producer.source,
            &fixture.producer.bindings,
        )
        .unwrap(),
        execution_nonce: [0x11; 32],
    };
    let preprocessing_manifest = mxx_ir_core::artifact::export_validated_manifest(
        preprocessing_production.clone(),
        &fixture.producer,
    )
    .unwrap();
    let (encrypt, _, _) =
        (fixture.encrypt)(preprocessing_production.clone(), preprocessing_manifest.clone());
    assert_eq!(encrypt.source.name(), "rns-bgg-refresh-encrypt");
    let encrypt_production = mxx_ir_core::artifact::ProductionId {
        spec_hash: mxx_ir_core::encoding::spec_hash(&encrypt.source, &encrypt.bindings).unwrap(),
        execution_nonce: [0x22; 32],
    };
    let encrypt_manifest =
        mxx_ir_core::artifact::export_validated_manifest(encrypt_production.clone(), &encrypt)
            .unwrap();
    let (online, benchmark_online) = (fixture.online)(
        preprocessing_production,
        preprocessing_manifest,
        encrypt_production,
        encrypt_manifest,
    );
    assert_eq!(online.source.name(), "rns_bgg-refresh-online");
    assert_eq!(benchmark_online.source.name(), "rns_bgg-refresh-online-benchmark");
    assert!(
        online.source.scopes().values().flat_map(|scope| scope.nodes()).all(|node| !matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::UniformIntervalSample { .. }
        )),
        "online graph must import secret-dependent artifacts instead of sampling a secret"
    );
    assert!(
        benchmark_online.source.scopes().values().flat_map(|scope| scope.nodes()).all(
            |node| !matches!(
                node.kind(),
                mxx_ir_core::node::NodeKind::UniformIntervalSample { .. }
            )
        ),
        "benchmark online graph must not resample the mask secret"
    );
    assert!(
        benchmark_online.source.outputs().keys().any(|name| name.starts_with("continued")),
        "online estimate must retain the continuation LUT computation"
    );
}

#[test]
fn test_gpu_rns_bgg_refresh_artifacts_with_native_prf_digits() {
    use mxx_primitives::{element::PolyElem, matrix::PolyMatrix, poly::Poly};
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    let average = selected_average_config();
    println!("rns_bgg_refresh_average_case_config={average:?}");
    for device in devices {
        let fixture = rns_bgg::refresh(average.as_ref());
        println!("rns_bgg_noise_simulation={}", fixture.noise_report);
        rns_bgg::run(
            fixture,
            |parameters| {
                let levels = parameters
                    .levels()
                    .iter()
                    .cloned()
                    .chain(parameters.refresh_residue_parameters(2).unwrap())
                    .map(|level| {
                        GpuDCRTPolyParams::new_with_gpu(
                            level.ring_dimension(),
                            level.to_crt().0,
                            level.base_bits(),
                            vec![device],
                            None,
                            None,
                            None,
                        )
                    });
                gpu_backend_on(levels, [device])
            },
            |matrix, name, q, bound| {
                for shard in matrix.shards() {
                    let cpu = shard.value.to_cpu_matrix();
                    for poly in cpu
                        .block_entries(0..cpu.row_size(), 0..cpu.col_size())
                        .into_iter()
                        .flatten()
                    {
                        for coefficient in poly.coeffs() {
                            let magnitude =
                                coefficient.value().clone().min(q - coefficient.value());
                            assert!(
                                &magnitude <= bound,
                                "{name} GPU {device}: {magnitude} > {bound}"
                            );
                        }
                    }
                }
            },
        );
    }
}
