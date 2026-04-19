#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use mxx::{
    circuit::PolyCircuit,
    gadgets::arith::{DEFAULT_MAX_UNREDUCED_MULS, NestedRnsPoly, NestedRnsPolyContext},
    poly::{
        PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    simulator::{SimulatorContext, error_norm::NormPltGGH15Evaluator},
    utils::bigdecimal_bits_ceil,
};
use num_bigint::BigUint;
use std::{env, sync::Arc};
use tracing::info;

const DEFAULT_RING_DIM: u32 = 1 << 14;
const DEFAULT_CRT_BITS: usize = 24;
const DEFAULT_P_MODULI_BITS: usize = 6;
const DEFAULT_MAX_UNREDUCED_MULS_BUDGET: usize = DEFAULT_MAX_UNREDUCED_MULS;
const DEFAULT_SCALE: u64 = 1 << 7;
const DEFAULT_BASE_BITS: u32 = 12;
const DEFAULT_MAX_CRT_DEPTH: usize = 32;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_HEIGHT: usize = 1;

#[derive(Debug, Clone)]
struct ModqArithConfig {
    ring_dim: u32,
    crt_bits: usize,
    p_moduli_bits: usize,
    scale: u64,
    base_bits: u32,
    max_crt_depth: usize,
    error_sigma: f64,
    d_secret: usize,
    height: usize,
}

fn env_or_parse_u32(key: &str, default: u32) -> u32 {
    match env::var(key) {
        Ok(raw) => raw.parse::<u32>().unwrap_or_else(|e| panic!("{key} must be a valid u32: {e}")),
        Err(_) => default,
    }
}

fn env_or_parse_u64(key: &str, default: u64) -> u64 {
    match env::var(key) {
        Ok(raw) => raw.parse::<u64>().unwrap_or_else(|e| panic!("{key} must be a valid u64: {e}")),
        Err(_) => default,
    }
}

fn env_or_parse_usize(key: &str, default: usize) -> usize {
    match env::var(key) {
        Ok(raw) => {
            raw.parse::<usize>().unwrap_or_else(|e| panic!("{key} must be a valid usize: {e}"))
        }
        Err(_) => default,
    }
}

fn env_or_parse_f64(key: &str, default: f64) -> f64 {
    match env::var(key) {
        Ok(raw) => raw.parse::<f64>().unwrap_or_else(|e| panic!("{key} must be a valid f64: {e}")),
        Err(_) => default,
    }
}

impl ModqArithConfig {
    fn from_env() -> Self {
        let ring_dim = env_or_parse_u32("GGH15_MODQ_ARITH_RING_DIM", DEFAULT_RING_DIM);
        let crt_bits = env_or_parse_usize("GGH15_MODQ_ARITH_CRT_BITS", DEFAULT_CRT_BITS);
        let p_moduli_bits =
            env_or_parse_usize("GGH15_MODQ_ARITH_P_MODULI_BITS", DEFAULT_P_MODULI_BITS);
        let scale = env_or_parse_u64("GGH15_MODQ_ARITH_SCALE", DEFAULT_SCALE);
        let base_bits = env_or_parse_u32("GGH15_MODQ_ARITH_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth =
            env_or_parse_usize("GGH15_MODQ_ARITH_MAX_CRT_DEPTH", DEFAULT_MAX_CRT_DEPTH);
        let error_sigma = env_or_parse_f64("GGH15_MODQ_ARITH_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret = env_or_parse_usize("GGH15_MODQ_ARITH_D_SECRET", DEFAULT_D_SECRET);
        let height = env_or_parse_usize("GGH15_MODQ_ARITH_HEIGHT", DEFAULT_HEIGHT);

        assert!(ring_dim > 0, "GGH15_MODQ_ARITH_RING_DIM must be > 0");
        assert!(crt_bits > 0, "GGH15_MODQ_ARITH_CRT_BITS must be > 0");
        assert!(p_moduli_bits > 0, "GGH15_MODQ_ARITH_P_MODULI_BITS must be > 0");
        assert!(scale > 0, "GGH15_MODQ_ARITH_SCALE must be > 0");
        assert!(base_bits > 0, "GGH15_MODQ_ARITH_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "GGH15_MODQ_ARITH_MAX_CRT_DEPTH must be > 0");
        assert!(error_sigma > 0.0, "GGH15_MODQ_ARITH_ERROR_SIGMA must be > 0");
        assert!(d_secret > 0, "GGH15_MODQ_ARITH_D_SECRET must be > 0");

        Self {
            ring_dim,
            crt_bits,
            p_moduli_bits,
            scale,
            base_bits,
            max_crt_depth,
            error_sigma,
            d_secret,
            height,
        }
    }
}

fn q_level_from_env() -> Option<usize> {
    std::env::var("GGH15_MODQ_ARITH_Q_LEVEL").ok().map(|raw| {
        let level =
            raw.parse::<usize>().expect("GGH15_MODQ_ARITH_Q_LEVEL must be a positive integer");
        assert!(level > 0, "GGH15_MODQ_ARITH_Q_LEVEL must be greater than or equal to 1");
        level
    })
}

fn active_q_moduli_and_modulus<T: PolyParams>(
    params: &T,
    q_level: Option<usize>,
) -> (Vec<u64>, BigUint, usize) {
    let (q_moduli, _, max_q_level) = params.to_crt();
    let active_q_level = q_level.unwrap_or(max_q_level);
    assert!(
        active_q_level <= max_q_level,
        "q_level exceeds CRT depth: q_level={}, crt_depth={}",
        active_q_level,
        max_q_level
    );
    let active_q_moduli = q_moduli.into_iter().take(active_q_level).collect::<Vec<_>>();
    let active_q =
        active_q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
    (active_q_moduli, active_q, active_q_level)
}

fn build_modq_arith_circuit_cpu(
    params: &DCRTPolyParams,
    q_level: Option<usize>,
    p_moduli_bits: usize,
    scale: u64,
    height: usize,
) -> (PolyCircuit<DCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        p_moduli_bits,
        DEFAULT_MAX_UNREDUCED_MULS_BUDGET,
        scale,
        false,
        q_level,
    ));

    NestedRnsPoly::benchmark_multiplication_tree(ctx.clone(), &mut circuit, height, q_level);
    (circuit, ctx)
}

fn build_modq_arith_circuit_gpu(
    params: &GpuDCRTPolyParams,
    q_level: Option<usize>,
    p_moduli_bits: usize,
    scale: u64,
    height: usize,
) -> (PolyCircuit<GpuDCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        p_moduli_bits,
        DEFAULT_MAX_UNREDUCED_MULS_BUDGET,
        scale,
        false,
        q_level,
    ));

    NestedRnsPoly::benchmark_multiplication_tree(ctx.clone(), &mut circuit, height, q_level);
    (circuit, ctx)
}

fn find_crt_depth_for_modq_arith(
    cfg: &ModqArithConfig,
    q_level: Option<usize>,
) -> (usize, DCRTPolyParams) {
    let ring_dim_sqrt = BigDecimal::from_u32(cfg.ring_dim).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << cfg.base_bits, 0);
    let error_sigma = BigDecimal::from_f64(cfg.error_sigma).expect("valid error sigma");
    let input_bound = BigDecimal::from((1u64 << cfg.p_moduli_bits) - 1);
    let e_init_norm = &error_sigma * BigDecimal::from_f32(6.5).unwrap();

    for crt_depth in 1..=cfg.max_crt_depth {
        let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
        let (active_q_moduli, _, _) = active_q_moduli_and_modulus(&params, q_level);
        let full_q = params.modulus();
        let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
        let (_, _, crt_depth) = params.to_crt();
        let (circuit, _ctx) = build_modq_arith_circuit_cpu(
            &params,
            q_level,
            cfg.p_moduli_bits,
            cfg.scale,
            cfg.height,
        );

        let log_base_q = params.modulus_digits();
        let log_base_q_small = log_base_q / crt_depth;
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt.clone(),
            base.clone(),
            cfg.d_secret,
            log_base_q,
            log_base_q_small,
        ));
        let plt_evaluator =
            NormPltGGH15Evaluator::new(ctx.clone(), &error_sigma, &error_sigma, None);

        let out_errors = circuit.simulate_max_error_norm(
            ctx,
            input_bound.clone(),
            circuit.num_input(),
            &e_init_norm,
            Some(&plt_evaluator),
            None,
        );

        assert_eq!(out_errors.len(), 1);

        let threshold = full_q.as_ref() / BigUint::from(2u64 * q_max);
        let error = &out_errors[0].matrix_norm.poly_norm.norm;
        let max_error_bits = bigdecimal_bits_ceil(error);
        let all_ok = *error < BigDecimal::from_biguint(threshold, 0);

        info!(
            "crt_depth={} q_bits={} max_error_bits={}",
            crt_depth,
            params.modulus_bits(),
            max_error_bits
        );

        if all_ok {
            return (crt_depth, params);
        }
    }

    panic!(
        "crt_depth satisfying error < q/(2*q_i) for all CRT moduli not found up to MAX_CRT_DEPTH ({})",
        cfg.max_crt_depth
    );
}

#[test]
fn test_gpu_ggh15_nested_rns_modq_arith() {
    let _ = tracing_subscriber::fmt::try_init();
    gpu_device_sync();
    let cfg = ModqArithConfig::from_env();
    info!("nested_rns modq arith test config: {:?}", cfg);

    let q_level = q_level_from_env();
    let num_inputs =
        1usize.checked_shl(cfg.height as u32).expect("GGH15_MODQ_ARITH_HEIGHT is too large");
    let (crt_depth, cpu_params) = find_crt_depth_for_modq_arith(&cfg, q_level);
    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli.clone(), cpu_params.base_bits());
    let single_gpu_id = *detected_gpu_params
        .gpu_ids()
        .first()
        .expect("at least one GPU device is required for test_gpu_ggh15_nested_rns_modq_arith");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
    );
    let (all_q_moduli, _, _) = params.to_crt();
    let (active_q_moduli, active_q, active_q_level) = active_q_moduli_and_modulus(&params, q_level);
    let (circuit, _ctx) =
        build_modq_arith_circuit_gpu(&params, q_level, cfg.p_moduli_bits, cfg.scale, cfg.height);
    let gate_counts = circuit.count_gates_by_type_vec();
    let total_gates = circuit.num_gates();

    info!("forcing single GPU for this test: gpu_id={}", single_gpu_id);
    info!("found crt_depth={}", crt_depth);
    info!(
        "selected crt_depth={} ring_dim={} crt_bits={} base_bits={} q_level={:?} q_moduli={:?}",
        crt_depth,
        params.ring_dimension(),
        cfg.crt_bits,
        cfg.base_bits,
        q_level,
        all_q_moduli
    );
    info!(
        "active_q_level={} active_q_bits={} active_q_moduli_len={}",
        active_q_level,
        active_q.bits(),
        active_q_moduli.len()
    );
    info!("multiplication tree config: height={} num_inputs={}", cfg.height, num_inputs);
    let non_free_depth_contributions = circuit.non_free_depth_contributions();
    info!(
        "circuit total_gates={} non_free_depth_contributions={:?} gate_counts={:?}",
        total_gates, non_free_depth_contributions, gate_counts
    );

    assert_eq!(params.modulus(), cpu_params.modulus());
    assert_eq!(circuit.num_output(), 1, "multiplication tree should emit one output gate");
    assert!(circuit.num_input() > 0, "circuit must expose inputs");
    assert!(total_gates > 0, "circuit must contain gates");

    gpu_device_sync();
}
