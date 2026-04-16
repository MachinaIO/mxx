#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bench_estimator::{
        BenchEstimator, BggPolyEncodingBenchEstimator, BggPolyEncodingBenchSamples,
        BggPublicKeyBenchEstimator, BggPublicKeyBenchSamples,
    },
    bgg::{
        public_key::BggPublicKey,
        sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
    },
    circuit::{PolyCircuit, evaluable::PolyVec, gate::GateId},
    element::PolyElem,
    gadgets::{
        arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly_compact_bytes},
        conv_mul::negacyclic_conv_mul,
        ntt::encode_nested_rns_poly_vec,
    },
    lookup::{
        PltEvaluator, PublicLut,
        ggh15_eval::{GGH15BGGPolyEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator},
        poly::PolyPltEvaluator,
        poly_vec::PolyVecPltEvaluator,
    },
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{
                GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_count, detected_gpu_device_ids,
                gpu_device_sync,
            },
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    sampler::{
        DistType, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    simulator::{
        SimulatorContext,
        error_norm::{ErrorNorm, NormBggPolyEncodingSTEvaluator, NormPltGGH15Evaluator},
        poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::{
        BggPolyEncodingSTEvaluator, PolyVecSlotTransferEvaluator, SlotTransferEvaluator,
        bgg_pubkey::BggPublicKeySTEvaluator,
    },
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::{bigdecimal_bits_ceil, gen_biguint_for_modulus},
};
use num_bigint::BigUint;
use std::{env, fs, path::Path, sync::Arc, time::Instant};
use tracing::info;

const DEFAULT_RING_DIM: u32 = 1 << 2;
const DEFAULT_NUM_SLOTS: usize = 1;
const DEFAULT_CRT_BITS: usize = 12;
const DEFAULT_P_MODULI_BITS: usize = 5;
const DEFAULT_MAX_UNREDUCED_MULS: usize = 2;
const DEFAULT_SCALE: u64 = 1;
const DEFAULT_BASE_BITS: u32 = 12;
const DEFAULT_MAX_CRT_DEPTH: usize = 16;
const DEFAULT_ERROR_SIGMA: f64 = 0.0;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_BENCH_ITERATIONS: usize = 1;
const DEFAULT_Q_LEVEL: usize = 1;
const TRAPDOOR_SIGMA: f64 = 4.578;

type GpuMatrix = GpuDCRTPolyMatrix;
type GpuHashSampler = GpuDCRTPolyHashSampler<Keccak256>;
type GpuPubKeyPltEvaluator = GGH15BGGPubKeyPltEvaluator<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
>;
type GpuPubKeySlotEvaluator = BggPublicKeySTEvaluator<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
>;
type GpuPolyPltEvaluator = GGH15BGGPolyEncodingPltEvaluator<GpuMatrix, GpuHashSampler>;
type GpuPolySlotEvaluator = BggPolyEncodingSTEvaluator<GpuMatrix, GpuHashSampler>;

#[derive(Debug, Clone)]
struct NegacyclicConvMulConfig {
    ring_dim: u32,
    num_slots: usize,
    crt_bits: usize,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    scale: u64,
    base_bits: u32,
    max_crt_depth: usize,
    error_sigma: f64,
    d_secret: usize,
    bench_iterations: usize,
    dir_name_override: Option<String>,
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

impl NegacyclicConvMulConfig {
    fn from_env() -> Self {
        let ring_dim = env_or_parse_u32("GGH15_NEGACYCLIC_CONV_MUL_RING_DIM", DEFAULT_RING_DIM);
        let num_slots =
            env_or_parse_usize("GGH15_NEGACYCLIC_CONV_MUL_NUM_SLOTS", DEFAULT_NUM_SLOTS);
        let crt_bits = env_or_parse_usize("GGH15_NEGACYCLIC_CONV_MUL_CRT_BITS", DEFAULT_CRT_BITS);
        let p_moduli_bits =
            env_or_parse_usize("GGH15_NEGACYCLIC_CONV_MUL_P_MODULI_BITS", DEFAULT_P_MODULI_BITS);
        let max_unreduced_muls = env_or_parse_usize(
            "GGH15_NEGACYCLIC_CONV_MUL_MAX_UNREDUCED_MULS",
            DEFAULT_MAX_UNREDUCED_MULS,
        );
        let scale = env_or_parse_u64("GGH15_NEGACYCLIC_CONV_MUL_SCALE", DEFAULT_SCALE);
        let base_bits = env_or_parse_u32("GGH15_NEGACYCLIC_CONV_MUL_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth =
            env_or_parse_usize("GGH15_NEGACYCLIC_CONV_MUL_MAX_CRT_DEPTH", DEFAULT_MAX_CRT_DEPTH);
        let error_sigma =
            env_or_parse_f64("GGH15_NEGACYCLIC_CONV_MUL_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret = env_or_parse_usize("GGH15_NEGACYCLIC_CONV_MUL_D_SECRET", DEFAULT_D_SECRET);
        let bench_iterations = env_or_parse_usize(
            "GGH15_NEGACYCLIC_CONV_MUL_BENCH_ITERATIONS",
            DEFAULT_BENCH_ITERATIONS,
        );
        let dir_name_override = env::var("GGH15_NEGACYCLIC_CONV_MUL_DIR_NAME")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        assert!(ring_dim > 0, "GGH15_NEGACYCLIC_CONV_MUL_RING_DIM must be > 0");
        assert!(num_slots > 0, "GGH15_NEGACYCLIC_CONV_MUL_NUM_SLOTS must be > 0");
        assert!(
            num_slots <= ring_dim as usize,
            "GGH15_NEGACYCLIC_CONV_MUL_NUM_SLOTS must not exceed ring dimension"
        );
        assert!(crt_bits > 0, "GGH15_NEGACYCLIC_CONV_MUL_CRT_BITS must be > 0");
        assert!(p_moduli_bits > 0, "GGH15_NEGACYCLIC_CONV_MUL_P_MODULI_BITS must be > 0");
        assert!(max_unreduced_muls > 0, "GGH15_NEGACYCLIC_CONV_MUL_MAX_UNREDUCED_MULS must be > 0");
        assert!(scale > 0, "GGH15_NEGACYCLIC_CONV_MUL_SCALE must be > 0");
        assert!(base_bits > 0, "GGH15_NEGACYCLIC_CONV_MUL_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "GGH15_NEGACYCLIC_CONV_MUL_MAX_CRT_DEPTH must be > 0");
        assert!(error_sigma >= 0.0, "GGH15_NEGACYCLIC_CONV_MUL_ERROR_SIGMA must be >= 0");
        assert!(d_secret > 0, "GGH15_NEGACYCLIC_CONV_MUL_D_SECRET must be > 0");
        assert!(bench_iterations > 0, "GGH15_NEGACYCLIC_CONV_MUL_BENCH_ITERATIONS must be > 0");

        Self {
            ring_dim,
            num_slots,
            crt_bits,
            p_moduli_bits,
            max_unreduced_muls,
            scale,
            base_bits,
            max_crt_depth,
            error_sigma,
            d_secret,
            bench_iterations,
            dir_name_override,
        }
    }

    fn dir_name(&self, active_q_level: usize) -> String {
        self.dir_name_override.clone().unwrap_or_else(|| {
            format!(
                "test_data/test_gpu_ggh15_negacyclic_conv_mul_ring{}_slots{}_crt{}_p{}_scale{}_qlevel_{}",
                self.ring_dim,
                self.num_slots,
                self.crt_bits,
                self.p_moduli_bits,
                self.scale,
                active_q_level
            )
        })
    }
}

fn q_level_from_env() -> Option<usize> {
    match env::var("GGH15_NEGACYCLIC_CONV_MUL_Q_LEVEL") {
        Ok(raw) => {
            let normalized = raw.trim().to_ascii_lowercase();
            if normalized.is_empty() || normalized == "full" || normalized == "none" {
                None
            } else {
                let level = normalized.parse::<usize>().expect(
                    "GGH15_NEGACYCLIC_CONV_MUL_Q_LEVEL must be a positive integer or `full`",
                );
                assert!(
                    level > 0,
                    "GGH15_NEGACYCLIC_CONV_MUL_Q_LEVEL must be greater than or equal to 1"
                );
                Some(level)
            }
        }
        Err(_) => Some(DEFAULT_Q_LEVEL),
    }
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

fn assert_value_matches_q_level(
    value: &BigUint,
    expected_mod_active_q: &BigUint,
    active_q: &BigUint,
    active_q_moduli: &[u64],
    all_q_moduli: &[u64],
) {
    if active_q_moduli.len() == all_q_moduli.len() {
        assert_eq!(
            value, expected_mod_active_q,
            "value must match expected modulo full q when q_level covers all CRT levels"
        );
        return;
    }

    assert_eq!(
        value % active_q,
        expected_mod_active_q.clone(),
        "value modulo active q must match expected modulo active q"
    );
}

fn build_negacyclic_conv_circuit_with_ctx<P: Poly + 'static>(
    params: &P::Params,
    cfg: &NegacyclicConvMulConfig,
    q_level: Option<usize>,
) -> (PolyCircuit<P>, Arc<NestedRnsPolyContext>) {
    let build_start = Instant::now();
    let mut circuit = PolyCircuit::<P>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        cfg.p_moduli_bits,
        cfg.max_unreduced_muls,
        cfg.scale,
        false,
        q_level,
    ));
    let lhs = NestedRnsPoly::input(ctx.clone(), q_level, None, &mut circuit);
    let rhs = NestedRnsPoly::input(ctx.clone(), q_level, None, &mut circuit);
    let product = negacyclic_conv_mul(params, &mut circuit, &lhs, &rhs, cfg.num_slots);
    let output = product.reconstruct(&mut circuit);
    circuit.output(vec![output]);
    info!(
        "negacyclic conv circuit build elapsed_ms={:.3}",
        build_start.elapsed().as_secs_f64() * 1000.0
    );
    (circuit, ctx)
}

fn build_negacyclic_conv_circuit<P: Poly + 'static>(
    params: &P::Params,
    cfg: &NegacyclicConvMulConfig,
    q_level: Option<usize>,
) -> PolyCircuit<P> {
    build_negacyclic_conv_circuit_with_ctx(params, cfg, q_level).0
}

fn simulate_max_error_norm_with_slot_transfer<P: PltEvaluator<ErrorNorm>>(
    circuit: &PolyCircuit<DCRTPoly>,
    ctx: Arc<SimulatorContext>,
    input_norm_bound: BigDecimal,
    input_size: usize,
    e_init_norm: &BigDecimal,
    plt_evaluator: &P,
    slot_transfer_evaluator: &NormBggPolyEncodingSTEvaluator,
) -> Vec<ErrorNorm> {
    let one_error = ErrorNorm::new(
        PolyNorm::one(ctx.clone()),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
    );
    let input_error = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), input_norm_bound),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
    );
    let input_errors = vec![input_error; input_size];
    circuit.eval(
        &(),
        one_error,
        input_errors,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator),
        None,
    )
}

fn find_crt_depth_for_negacyclic_conv_mul(
    cfg: &NegacyclicConvMulConfig,
    q_level: Option<usize>,
) -> (usize, DCRTPolyParams) {
    let ring_dim_sqrt = BigDecimal::from_u32(cfg.ring_dim).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << cfg.base_bits, 0);
    let error_sigma = BigDecimal::from_f64(cfg.error_sigma).expect("valid error sigma");
    let input_bound = BigDecimal::from((1u64 << cfg.p_moduli_bits) - 1);
    let e_init_norm = &error_sigma * BigDecimal::from_f32(6.5).unwrap();

    for crt_depth in 1..=cfg.max_crt_depth {
        let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
        let (active_q_moduli, active_q, active_q_level) =
            active_q_moduli_and_modulus(&params, q_level);
        let full_q = params.modulus();
        let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
        let (_, _, actual_crt_depth) = params.to_crt();
        let circuit = build_negacyclic_conv_circuit::<DCRTPoly>(&params, cfg, q_level);

        let log_base_q = params.modulus_digits();
        let log_base_q_small = log_base_q / actual_crt_depth;
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt.clone(),
            base.clone(),
            cfg.d_secret,
            log_base_q,
            log_base_q_small,
        ));
        let plt_evaluator =
            NormPltGGH15Evaluator::new(ctx.clone(), &error_sigma, &error_sigma, None);
        let slot_transfer_evaluator =
            NormBggPolyEncodingSTEvaluator::new(ctx.clone(), cfg.error_sigma, &error_sigma, None);
        let out_errors = simulate_max_error_norm_with_slot_transfer(
            &circuit,
            ctx,
            input_bound.clone(),
            circuit.num_input(),
            &e_init_norm,
            &plt_evaluator,
            &slot_transfer_evaluator,
        );
        assert_eq!(out_errors.len(), 1, "convolution circuit must output one error bound");

        let threshold = full_q.as_ref() / BigUint::from(2u64 * q_max);
        let threshold_bd = BigDecimal::from_biguint(threshold.clone(), 0);
        let error = &out_errors[0].matrix_norm.poly_norm.norm;
        let max_error_bits = bigdecimal_bits_ceil(error);
        let all_ok = *error < threshold_bd;

        info!(
            "crt_depth={} active_q_level={} active_q_bits={} q_bits={} max_error_bits={} threshold_bits={}",
            crt_depth,
            active_q_level,
            active_q.bits(),
            full_q.bits(),
            max_error_bits,
            bigdecimal_bits_ceil(&BigDecimal::from_biguint(threshold, 0))
        );

        if all_ok {
            return (crt_depth, params);
        }
    }

    panic!(
        "crt_depth satisfying error < q/(2*q_max) for negacyclic convolution was not found up to GGH15_NEGACYCLIC_CONV_MUL_MAX_CRT_DEPTH ({})",
        cfg.max_crt_depth
    );
}

fn eval_reference_negacyclic_conv_output(
    params: &DCRTPolyParams,
    cfg: &NegacyclicConvMulConfig,
    q_level: Option<usize>,
    lhs_coeffs: &[BigUint],
    rhs_coeffs: &[BigUint],
) -> Vec<Vec<BigUint>> {
    let reference_eval_start = Instant::now();
    let (circuit, ctx) = build_negacyclic_conv_circuit_with_ctx::<DCRTPoly>(params, cfg, q_level);
    let lhs_inputs =
        encode_nested_rns_poly_vec::<DCRTPoly>(params, ctx.as_ref(), lhs_coeffs, q_level);
    let rhs_inputs =
        encode_nested_rns_poly_vec::<DCRTPoly>(params, ctx.as_ref(), rhs_coeffs, q_level);
    let one = PolyVec::new(vec![DCRTPoly::const_one(params); cfg.num_slots]);
    let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
    let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
    let mut out = circuit.eval(
        params,
        one,
        [lhs_inputs, rhs_inputs].concat(),
        Some(&plt_evaluator),
        Some(&slot_transfer_evaluator),
        None,
    );
    assert_eq!(out.len(), 1, "reference CPU evaluation must produce one output");
    let out = out.pop().expect("reference CPU output must exist");
    info!(
        "cpu reference eval elapsed_ms={:.3}",
        reference_eval_start.elapsed().as_secs_f64() * 1000.0
    );
    out.as_slice().iter().map(|slot_poly| slot_poly.coeffs_biguints()).collect()
}

fn build_plaintext_rows_for_poly_inputs(
    params: &GpuDCRTPolyParams,
    cfg: &NegacyclicConvMulConfig,
    q_level: Option<usize>,
    lhs_coeffs: &[BigUint],
    rhs_coeffs: &[BigUint],
    expected_input_count: usize,
) -> Vec<Vec<Arc<[u8]>>> {
    let input_coeff_sets = [lhs_coeffs, rhs_coeffs];
    let mut rows = Vec::new();
    for coeffs in input_coeff_sets {
        let mut per_gate_rows: Vec<Vec<Arc<[u8]>>> = Vec::new();
        for coeff in coeffs {
            let encoded = encode_nested_rns_poly_compact_bytes::<GpuDCRTPoly>(
                cfg.p_moduli_bits,
                cfg.max_unreduced_muls,
                params,
                coeff,
                q_level,
            );
            if per_gate_rows.is_empty() {
                per_gate_rows = (0..encoded.len())
                    .map(|_| Vec::with_capacity(cfg.num_slots))
                    .collect::<Vec<_>>();
            }
            assert_eq!(
                encoded.len(),
                per_gate_rows.len(),
                "encoded row count must stay constant across coefficients"
            );
            for (row_idx, bytes) in encoded.into_iter().enumerate() {
                per_gate_rows[row_idx].push(Arc::<[u8]>::from(bytes));
            }
        }
        rows.extend(per_gate_rows);
    }
    assert_eq!(
        rows.len(),
        expected_input_count,
        "plaintext row count must match circuit input count"
    );
    assert!(
        rows.iter().all(|row| row.len() == cfg.num_slots),
        "each plaintext row must contain exactly num_slots entries"
    );
    rows
}

fn build_benchmark_public_lut(params: &GpuDCRTPolyParams) -> PublicLut<GpuDCRTPoly> {
    PublicLut::<GpuDCRTPoly>::new(
        params,
        16,
        move |params, x| {
            if x >= 16 {
                return None;
            }
            let y_elem = <<GpuDCRTPoly as Poly>::Elem as PolyElem>::constant(
                &params.modulus(),
                (x + 1) % 16,
            );
            Some((x, y_elem))
        },
        None,
    )
}

fn build_benchmark_public_lookup_gate(
    params: &GpuDCRTPolyParams,
) -> (PublicLut<GpuDCRTPoly>, usize, GateId) {
    let lut = build_benchmark_public_lut(params);
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let input = circuit.input(1).at(0);
    let lut_id = circuit.register_public_lookup(lut.clone());
    let gate_id = circuit.public_lookup_gate(input, lut_id).as_single_wire();
    (lut, lut_id, gate_id)
}

fn build_benchmark_slot_transfer_gate(src_slots: &[(u32, Option<u32>)]) -> GateId {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let input = circuit.input(1).at(0);
    circuit.slot_transfer_gate(input, src_slots).as_single_wire()
}

fn identity_slot_transfer_plan(num_slots: usize) -> Vec<(u32, Option<u32>)> {
    (0..num_slots).map(|slot| (slot as u32, None)).collect()
}

fn single_slot_plaintext(value: usize, params: &GpuDCRTPolyParams) -> Arc<[u8]> {
    Arc::<[u8]>::from(GpuDCRTPoly::from_usize_to_constant(params, value).to_compact_bytes())
}

fn single_slot_secret_mats(slot_secret_mats: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![slot_secret_mats.first().expect("slot secret mats must not be empty").clone()]
}

fn sample_benchmark_pubkeys(
    params: &GpuDCRTPolyParams,
    seed: [u8; 32],
    d_secret: usize,
    count: usize,
    tag: &[u8],
) -> Vec<BggPublicKey<GpuMatrix>> {
    let sampler = BGGPublicKeySampler::<_, GpuHashSampler>::new(seed, d_secret);
    let reveal_plaintexts = vec![true; count];
    sampler.sample(params, tag, &reveal_plaintexts)
}

#[tokio::test]
async fn test_gpu_ggh15_negacyclic_conv_mul() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    gpu_device_sync();

    let cfg = NegacyclicConvMulConfig::from_env();
    let q_level = q_level_from_env();
    info!("negacyclic conv test config: {:?}, q_level={:?}", cfg, q_level);

    let (crt_depth, cpu_params) = find_crt_depth_for_negacyclic_conv_mul(&cfg, q_level);
    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_ids = detected_gpu_device_ids();
    let detected_gpu_count = detected_gpu_device_count();
    assert_eq!(
        detected_gpu_count,
        detected_gpu_ids.len(),
        "detected GPU count and ids length must match"
    );
    let single_gpu_id = *detected_gpu_ids
        .first()
        .expect("at least one GPU device is required for test_gpu_ggh15_negacyclic_conv_mul");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
    );
    info!(
        "forcing single GPU for eval path: eval_gpu_id={} detected_gpu_count={} detected_gpu_ids={:?}",
        single_gpu_id, detected_gpu_count, detected_gpu_ids
    );
    assert_eq!(params.modulus(), cpu_params.modulus());

    let (all_q_moduli, _, _) = params.to_crt();
    let (active_q_moduli, active_q, active_q_level) = active_q_moduli_and_modulus(&params, q_level);
    let circuit = build_negacyclic_conv_circuit::<GpuDCRTPoly>(&params, &cfg, q_level);
    info!(
        "selected crt_depth={} ring_dim={} num_slots={} crt_bits={} base_bits={} q_moduli={:?}",
        crt_depth,
        params.ring_dimension(),
        cfg.num_slots,
        cfg.crt_bits,
        cfg.base_bits,
        all_q_moduli
    );
    info!(
        "circuit non_free_depth={} gate_counts={:?} num_inputs={}",
        circuit.non_free_depth(),
        circuit.count_gates_by_type_vec(),
        circuit.num_input()
    );
    info!(
        "active_q_level={} active_q_bits={} active_q_moduli_len={}",
        active_q_level,
        active_q.bits(),
        active_q_moduli.len()
    );

    let mut rng = rand::rng();
    let lhs_coeffs = (0..cfg.num_slots)
        .map(|_| gen_biguint_for_modulus(&mut rng, &active_q))
        .collect::<Vec<_>>();
    let rhs_coeffs = (0..cfg.num_slots)
        .map(|_| gen_biguint_for_modulus(&mut rng, &active_q))
        .collect::<Vec<_>>();
    info!("sampled lhs/rhs coefficient vectors for negacyclic convolution");
    let reference_output_coeffs =
        eval_reference_negacyclic_conv_output(&cpu_params, &cfg, q_level, &lhs_coeffs, &rhs_coeffs);
    assert_eq!(
        reference_output_coeffs.len(),
        cfg.num_slots,
        "reference CPU output slot count must match configured num_slots"
    );

    let seed: [u8; 32] = [0u8; 32];
    let dir_name = cfg.dir_name(active_q_level);
    let root_dir = Path::new(&dir_name);
    let actual_dir = root_dir.join("actual");
    let pubkey_bench_dir = root_dir.join("pubkey_bench");
    let poly_bench_dir = root_dir.join("poly_bench");
    for dir in [&actual_dir, &pubkey_bench_dir, &poly_bench_dir] {
        if !dir.exists() {
            fs::create_dir_all(dir).expect("failed to create test directory");
        }
    }
    init_storage_system(actual_dir.clone());

    let input_setup_start = Instant::now();
    let plaintext_rows = build_plaintext_rows_for_poly_inputs(
        &params,
        &cfg,
        q_level,
        &lhs_coeffs,
        &rhs_coeffs,
        circuit.num_input(),
    );
    info!(
        "plaintext row materialization elapsed_ms={:.3}",
        input_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets =
        uniform_sampler.sample_uniform(&params, 1, cfg.d_secret, DistType::TernaryDist).get_row(0);
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
    info!("sampled secret vector with {} polynomials", secrets.len());

    let pk_sampler = BGGPublicKeySampler::<_, GpuHashSampler>::new(seed, cfg.d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    let pubkey_sampling_start = Instant::now();
    let mut sampled_pubkeys =
        pk_sampler.sample(&params, b"GGH15_NEGACYCLIC_CONV", &reveal_plaintexts);
    info!(
        "pubkey sampling elapsed_ms={:.3} sampled_pubkeys={}",
        pubkey_sampling_start.elapsed().as_secs_f64() * 1000.0,
        sampled_pubkeys.len()
    );
    let input_pubkeys = sampled_pubkeys.split_off(1);
    let one_pubkey = sampled_pubkeys.pop().expect("const-one public key must exist");

    let plt_pubkey_evaluator = GpuPubKeyPltEvaluator::new(
        seed,
        cfg.d_secret,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        actual_dir.clone(),
    );
    let slot_pubkey_evaluator = GpuPubKeySlotEvaluator::new(
        seed,
        cfg.d_secret,
        cfg.num_slots,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        actual_dir.clone(),
    );

    let pubkey_eval_start = Instant::now();
    let pubkey_out = circuit.eval(
        &params,
        one_pubkey,
        input_pubkeys,
        Some(&plt_pubkey_evaluator),
        Some(&slot_pubkey_evaluator),
        None,
    );
    info!("pubkey eval elapsed_ms={:.3}", pubkey_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(pubkey_out.len(), 1, "pubkey evaluation must produce one output");

    let actual_b0_sampling_start = Instant::now();
    plt_pubkey_evaluator.sample_aux_matrices(&params);
    slot_pubkey_evaluator.sample_aux_matrices(&params);
    info!(
        "actual sample_aux_matrices elapsed_ms={:.3}",
        actual_b0_sampling_start.elapsed().as_secs_f64() * 1000.0
    );

    let wait_writes_start = Instant::now();
    wait_for_all_writes(actual_dir.clone()).await.expect("storage writes should complete");
    info!(
        "actual wait_for_all_writes elapsed_ms={:.3}",
        wait_writes_start.elapsed().as_secs_f64() * 1000.0
    );

    let slot_secret_mats = slot_pubkey_evaluator
        .load_slot_secret_mats_checkpoint(&params)
        .expect("slot secret matrix checkpoints should exist after auxiliary sampling");
    let poly_pubkey_sampling_start = Instant::now();
    let pubkeys_for_poly_encodings =
        pk_sampler.sample(&params, b"GGH15_NEGACYCLIC_CONV", &reveal_plaintexts);
    info!(
        "poly encoding pubkey resampling elapsed_ms={:.3} sampled_pubkeys={}",
        poly_pubkey_sampling_start.elapsed().as_secs_f64() * 1000.0,
        pubkeys_for_poly_encodings.len()
    );
    let encoding_sampler = BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(
        &params,
        &secrets,
        Some(cfg.error_sigma),
    );
    let poly_encoding_setup_start = Instant::now();
    let mut poly_encodings = encoding_sampler.sample(
        &params,
        &pubkeys_for_poly_encodings,
        &plaintext_rows,
        Some(&slot_secret_mats),
    );
    info!(
        "poly encoding sampling elapsed_ms={:.3}",
        poly_encoding_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let plt_b0_matrix = plt_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("public-lookup b0 checkpoint should exist after auxiliary sampling");
    let slot_b0_matrix = slot_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("slot-transfer b0 checkpoint should exist after auxiliary sampling");
    let plt_c_b0_compact_bytes_by_slot =
        GpuPolyPltEvaluator::build_c_b0_compact_bytes_by_slot::<GpuDCRTPolyUniformSampler>(
            &params,
            &s_vec,
            &plt_b0_matrix,
            &slot_secret_mats,
            Some(cfg.error_sigma),
        );
    let mut slot_c_b0 = s_vec.clone() * &slot_b0_matrix;
    if cfg.error_sigma != 0.0 {
        let slot_c_b0_error = uniform_sampler.sample_uniform(
            &params,
            slot_c_b0.row_size(),
            slot_c_b0.col_size(),
            DistType::GaussDist { sigma: cfg.error_sigma },
        );
        slot_c_b0 = slot_c_b0 + slot_c_b0_error;
    }
    let plt_poly_evaluator = GpuPolyPltEvaluator::new(
        seed,
        actual_dir.clone(),
        plt_pubkey_evaluator.checkpoint_prefix(&params),
        plt_c_b0_compact_bytes_by_slot,
    );
    let slot_poly_evaluator = GpuPolySlotEvaluator::new(
        seed,
        actual_dir.clone(),
        slot_pubkey_evaluator.checkpoint_prefix(&params),
        slot_c_b0.to_compact_bytes(),
    );

    let input_poly_encodings = poly_encodings.split_off(1);
    let one_poly_encoding = poly_encodings.pop().expect("const-one poly encoding must exist");
    let poly_eval_start = Instant::now();
    let poly_out = circuit.eval(
        &params,
        one_poly_encoding,
        input_poly_encodings,
        Some(&plt_poly_evaluator),
        Some(&slot_poly_evaluator),
        None,
    );
    info!("poly encoding eval elapsed_ms={:.3}", poly_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(poly_out.len(), 1, "poly encoding evaluation must produce one output");

    let poly_out = &poly_out[0];
    assert_eq!(poly_out.pubkey, pubkey_out[0]);
    assert_eq!(poly_out.num_slots(), cfg.num_slots);

    for (slot_idx, expected_poly_coeffs) in reference_output_coeffs.iter().enumerate() {
        let result_plaintext = poly_out
            .plaintext_for_params(&params, slot_idx)
            .expect("poly output should reveal plaintexts");
        let result_coeffs = result_plaintext.coeffs();
        assert!(
            result_coeffs.len() >= expected_poly_coeffs.len(),
            "result plaintext polynomial degree must cover the reference polynomial coefficients"
        );
        for (coeff_idx, expected_coeff) in expected_poly_coeffs.iter().enumerate() {
            let result_coeff = result_coeffs
                .get(coeff_idx)
                .expect("result plaintext polynomial coefficient must exist")
                .value();
            let expected_mod_active_q = expected_coeff % &active_q;
            assert_value_matches_q_level(
                result_coeff,
                &expected_mod_active_q,
                &active_q,
                &active_q_moduli,
                &all_q_moduli,
            );
        }
    }

    init_storage_system(pubkey_bench_dir.clone());
    let (bench_lut, pubkey_bench_public_lut_id, pubkey_bench_public_lut_gate_id) =
        build_benchmark_public_lookup_gate(&params);
    let bench_slot_transfer_src_slots = identity_slot_transfer_plan(cfg.num_slots);
    let pubkey_bench_slot_transfer_gate_id =
        build_benchmark_slot_transfer_gate(&bench_slot_transfer_src_slots);
    let pubkey_bench_public_keys =
        sample_benchmark_pubkeys(&params, seed, cfg.d_secret, 10, b"GGH15_NEGACYCLIC_CONV_PKBENCH");
    let pubkey_bench_plt_evaluator = GpuPubKeyPltEvaluator::new(
        seed,
        cfg.d_secret,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        pubkey_bench_dir.clone(),
    );
    let pubkey_bench_slot_evaluator = GpuPubKeySlotEvaluator::new(
        seed,
        cfg.d_secret,
        cfg.num_slots,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        pubkey_bench_dir.clone(),
    );
    let pubkey_bench_estimator_owner = BggPublicKeyBenchEstimator::benchmark(
        &BggPublicKeyBenchSamples {
            params: &params,
            add_lhs: &pubkey_bench_public_keys[1],
            add_rhs: &pubkey_bench_public_keys[2],
            sub_lhs: &pubkey_bench_public_keys[3],
            sub_rhs: &pubkey_bench_public_keys[4],
            mul_lhs: &pubkey_bench_public_keys[5],
            mul_rhs: &pubkey_bench_public_keys[6],
            small_scalar_input: &pubkey_bench_public_keys[7],
            small_scalar: &[3u32, 5u32],
            large_scalar_input: &pubkey_bench_public_keys[8],
            large_scalar: &[BigUint::from(7u32)],
            public_lut_one: &pubkey_bench_public_keys[0],
            public_lut_input: &pubkey_bench_public_keys[9],
            public_lut: &bench_lut,
            public_lut_gate_id: pubkey_bench_public_lut_gate_id,
            public_lut_id: pubkey_bench_public_lut_id,
            slot_transfer_input: &pubkey_bench_public_keys[10],
            slot_transfer_src_slots: &bench_slot_transfer_src_slots,
            slot_transfer_gate_id: pubkey_bench_slot_transfer_gate_id,
        },
        &pubkey_bench_plt_evaluator,
        &pubkey_bench_slot_evaluator,
        GpuPubKeyPltEvaluator::new(
            seed,
            cfg.d_secret,
            TRAPDOOR_SIGMA,
            cfg.error_sigma,
            pubkey_bench_dir.clone(),
        ),
        GpuPubKeySlotEvaluator::new(
            seed,
            cfg.d_secret,
            cfg.num_slots,
            TRAPDOOR_SIGMA,
            cfg.error_sigma,
            pubkey_bench_dir.clone(),
        ),
        cfg.bench_iterations,
    );
    let pubkey_circuit_bench = pubkey_bench_estimator_owner.estimate_circuit_bench(&circuit);
    info!(
        "pubkey circuit bench estimate: total_time={:.6} latency={:.6} max_parallelism={}",
        pubkey_circuit_bench.total_time,
        pubkey_circuit_bench.latency,
        pubkey_circuit_bench.max_parallelism
    );

    init_storage_system(poly_bench_dir.clone());
    let one_slot_bench_pubkeys = sample_benchmark_pubkeys(
        &params,
        seed,
        cfg.d_secret,
        10,
        b"GGH15_NEGACYCLIC_CONV_POLYBENCH",
    );
    let poly_bench_pubkey_plt_evaluator = GpuPubKeyPltEvaluator::new(
        seed,
        cfg.d_secret,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        poly_bench_dir.clone(),
    );
    let poly_bench_pubkey_slot_evaluator = GpuPubKeySlotEvaluator::new(
        seed,
        cfg.d_secret,
        1,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        poly_bench_dir.clone(),
    );
    let poly_bench_slot_transfer_src_slots = vec![(0u32, None)];
    let (_poly_bench_lut, poly_bench_public_lut_id, poly_bench_public_lut_gate_id) =
        build_benchmark_public_lookup_gate(&params);
    let poly_bench_slot_transfer_gate_id =
        build_benchmark_slot_transfer_gate(&poly_bench_slot_transfer_src_slots);
    let _ = poly_bench_pubkey_plt_evaluator.public_lookup(
        &params,
        &bench_lut,
        &one_slot_bench_pubkeys[0],
        &one_slot_bench_pubkeys[9],
        poly_bench_public_lut_gate_id,
        poly_bench_public_lut_id,
    );
    let _ = poly_bench_pubkey_slot_evaluator.slot_transfer(
        &params,
        &one_slot_bench_pubkeys[10],
        &poly_bench_slot_transfer_src_slots,
        poly_bench_slot_transfer_gate_id,
    );
    let poly_bench_aux_start = Instant::now();
    poly_bench_pubkey_plt_evaluator.sample_aux_matrices(&params);
    poly_bench_pubkey_slot_evaluator.sample_aux_matrices(&params);
    info!(
        "poly bench sample_aux_matrices elapsed_ms={:.3}",
        poly_bench_aux_start.elapsed().as_secs_f64() * 1000.0
    );
    wait_for_all_writes(poly_bench_dir.clone())
        .await
        .expect("benchmark storage writes should complete");

    let bench_secrets =
        uniform_sampler.sample_uniform(&params, 1, cfg.d_secret, DistType::TernaryDist).get_row(0);
    let bench_s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, bench_secrets.clone());
    let bench_slot_secret_mats = single_slot_secret_mats(
        &poly_bench_pubkey_slot_evaluator
            .load_slot_secret_mats_checkpoint(&params)
            .expect("one-slot slot secret matrix checkpoints must exist for poly benchmark"),
    );
    let bench_plaintext_rows = vec![
        vec![single_slot_plaintext(2, &params)],
        vec![single_slot_plaintext(3, &params)],
        vec![single_slot_plaintext(4, &params)],
        vec![single_slot_plaintext(1, &params)],
        vec![single_slot_plaintext(5, &params)],
        vec![single_slot_plaintext(7, &params)],
        vec![single_slot_plaintext(3, &params)],
        vec![single_slot_plaintext(6, &params)],
        vec![single_slot_plaintext(2, &params)],
        vec![single_slot_plaintext(4, &params)],
    ];
    let bench_encoding_sampler = BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(
        &params,
        &bench_secrets,
        Some(cfg.error_sigma),
    );
    let bench_poly_encodings = bench_encoding_sampler.sample(
        &params,
        &one_slot_bench_pubkeys,
        &bench_plaintext_rows,
        Some(&bench_slot_secret_mats),
    );
    assert_eq!(
        bench_poly_encodings.len(),
        11,
        "one-slot benchmark encoding set must contain const one plus ten sample encodings"
    );

    let bench_plt_b0_matrix = poly_bench_pubkey_plt_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("one-slot public-lookup b0 checkpoint must exist for poly benchmark");
    let bench_slot_b0_matrix = poly_bench_pubkey_slot_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("one-slot slot-transfer b0 checkpoint must exist for poly benchmark");
    let bench_plt_c_b0_compact_bytes_by_slot =
        GpuPolyPltEvaluator::build_c_b0_compact_bytes_by_slot::<GpuDCRTPolyUniformSampler>(
            &params,
            &bench_s_vec,
            &bench_plt_b0_matrix,
            &bench_slot_secret_mats,
            Some(cfg.error_sigma),
        );
    let mut bench_slot_c_b0 = bench_s_vec.clone() * &bench_slot_b0_matrix;
    if cfg.error_sigma != 0.0 {
        let slot_c_b0_error = uniform_sampler.sample_uniform(
            &params,
            bench_slot_c_b0.row_size(),
            bench_slot_c_b0.col_size(),
            DistType::GaussDist { sigma: cfg.error_sigma },
        );
        bench_slot_c_b0 = bench_slot_c_b0 + slot_c_b0_error;
    }
    let bench_poly_plt_evaluator = GpuPolyPltEvaluator::new(
        seed,
        poly_bench_dir.clone(),
        poly_bench_pubkey_plt_evaluator.checkpoint_prefix(&params),
        bench_plt_c_b0_compact_bytes_by_slot,
    );
    let bench_poly_slot_evaluator = GpuPolySlotEvaluator::new(
        seed,
        poly_bench_dir.clone(),
        poly_bench_pubkey_slot_evaluator.checkpoint_prefix(&params),
        bench_slot_c_b0.to_compact_bytes(),
    );
    let poly_bench_estimator = BggPolyEncodingBenchEstimator::<GpuMatrix>::benchmark_with_samples(
        &BggPolyEncodingBenchSamples {
            num_slots: cfg.num_slots,
            params: &params,
            add_lhs: &bench_poly_encodings[1],
            add_rhs: &bench_poly_encodings[2],
            sub_lhs: &bench_poly_encodings[3],
            sub_rhs: &bench_poly_encodings[4],
            mul_lhs: &bench_poly_encodings[5],
            mul_rhs: &bench_poly_encodings[6],
            small_scalar_input: &bench_poly_encodings[7],
            small_scalar: &[3u32, 5u32],
            large_scalar_input: &bench_poly_encodings[8],
            large_scalar: &[BigUint::from(7u32)],
            public_lut_one: &bench_poly_encodings[0],
            public_lut_input: &bench_poly_encodings[9],
            public_lut: &bench_lut,
            public_lut_gate_id: poly_bench_public_lut_gate_id,
            public_lut_id: poly_bench_public_lut_id,
            slot_transfer_input: &bench_poly_encodings[10],
            slot_transfer_src_slots: &poly_bench_slot_transfer_src_slots,
            slot_transfer_gate_id: poly_bench_slot_transfer_gate_id,
        },
        &bench_poly_plt_evaluator,
        &bench_poly_slot_evaluator,
        cfg.bench_iterations,
    );
    let poly_circuit_bench = poly_bench_estimator.estimate_circuit_bench(&circuit);
    info!(
        "bgg poly encoding circuit bench estimate: total_time={:.6} latency={:.6} max_parallelism={}",
        poly_circuit_bench.total_time,
        poly_circuit_bench.latency,
        poly_circuit_bench.max_parallelism
    );

    wait_for_all_writes(poly_bench_dir.clone())
        .await
        .expect("final storage writes should complete");
    gpu_device_sync();
}
