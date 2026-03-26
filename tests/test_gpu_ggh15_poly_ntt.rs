#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    gadgets::{
        arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly_compact_bytes},
        ntt::inverse_ntt,
    },
    lookup::{
        PltEvaluator, PublicLut,
        ggh15_eval::{GGH15BGGPolyEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator},
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
        error_norm::{ErrorNorm, NormBggPolyEncodingSTEvaluator},
        poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::{BggPolyEncodingSTEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::bigdecimal_bits_ceil,
};
use num_bigint::BigUint;
use rand::Rng;
use rayon::prelude::*;
use std::{env, fs, path::Path, sync::Arc, time::Instant};
use tracing::{debug, info};

const DEFAULT_RING_DIM: u32 = 2;
const DEFAULT_NUM_SLOTS: usize = 2;
const DEFAULT_CRT_BITS: usize = 14;
const DEFAULT_P_MODULI_BITS: usize = 6;
const DEFAULT_SCALE: u64 = 1;
const DEFAULT_BASE_BITS: u32 = 6;
const DEFAULT_MAX_CRT_DEPTH: usize = 8;
const DEFAULT_MAX_UNREDUCED_MULS: usize = 2;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;
const TRAPDOOR_SIGMA: f64 = 4.578;

#[derive(Debug, Clone)]
struct NttConfig {
    ring_dim: u32,
    num_slots: usize,
    crt_bits: usize,
    p_moduli_bits: usize,
    scale: u64,
    base_bits: u32,
    max_crt_depth: usize,
    max_unreduced_muls: usize,
    error_sigma: f64,
    d_secret: usize,
    dir_name_override: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct ExactPltErrorNormEvaluator;

impl PltEvaluator<ErrorNorm> for ExactPltErrorNormEvaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        ErrorNorm {
            plaintext_norm: PolyNorm::new(input.clone_ctx(), plaintext_bd),
            matrix_norm: input.matrix_norm.clone(),
        }
    }
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

impl NttConfig {
    fn from_env() -> Self {
        let ring_dim = env_or_parse_u32("GGH15_POLY_NTT_RING_DIM", DEFAULT_RING_DIM);
        let num_slots = env_or_parse_usize("GGH15_POLY_NTT_NUM_SLOTS", DEFAULT_NUM_SLOTS);
        let crt_bits = env_or_parse_usize("GGH15_POLY_NTT_CRT_BITS", DEFAULT_CRT_BITS);
        let p_moduli_bits =
            env_or_parse_usize("GGH15_POLY_NTT_P_MODULI_BITS", DEFAULT_P_MODULI_BITS);
        let scale = env_or_parse_u64("GGH15_POLY_NTT_SCALE", DEFAULT_SCALE);
        let base_bits = env_or_parse_u32("GGH15_POLY_NTT_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth =
            env_or_parse_usize("GGH15_POLY_NTT_MAX_CRT_DEPTH", DEFAULT_MAX_CRT_DEPTH);
        let max_unreduced_muls =
            env_or_parse_usize("GGH15_POLY_NTT_MAX_UNREDUCED_MULS", DEFAULT_MAX_UNREDUCED_MULS);
        let error_sigma = env_or_parse_f64("GGH15_POLY_NTT_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret = env_or_parse_usize("GGH15_POLY_NTT_D_SECRET", DEFAULT_D_SECRET);
        let dir_name_override = env::var("GGH15_POLY_NTT_DIR_NAME")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        assert!(ring_dim > 0, "GGH15_POLY_NTT_RING_DIM must be > 0");
        assert!(num_slots > 0, "GGH15_POLY_NTT_NUM_SLOTS must be > 0");
        assert!(num_slots.is_power_of_two(), "GGH15_POLY_NTT_NUM_SLOTS must be a power of two");
        assert!(
            ring_dim as usize == num_slots,
            "GGH15_POLY_NTT_RING_DIM must equal GGH15_POLY_NTT_NUM_SLOTS for this test"
        );
        assert!(crt_bits > 0, "GGH15_POLY_NTT_CRT_BITS must be > 0");
        assert!(p_moduli_bits > 0, "GGH15_POLY_NTT_P_MODULI_BITS must be > 0");
        assert!(scale > 0, "GGH15_POLY_NTT_SCALE must be > 0");
        assert!(base_bits > 0, "GGH15_POLY_NTT_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "GGH15_POLY_NTT_MAX_CRT_DEPTH must be > 0");
        assert!(max_unreduced_muls > 0, "GGH15_POLY_NTT_MAX_UNREDUCED_MULS must be > 0");
        assert!(error_sigma > 0.0, "GGH15_POLY_NTT_ERROR_SIGMA must be > 0");
        assert!(d_secret > 0, "GGH15_POLY_NTT_D_SECRET must be > 0");

        Self {
            ring_dim,
            num_slots,
            crt_bits,
            p_moduli_bits,
            scale,
            base_bits,
            max_crt_depth,
            max_unreduced_muls,
            error_sigma,
            d_secret,
            dir_name_override,
        }
    }

    fn dir_name(&self, active_q_level: usize) -> String {
        self.dir_name_override.clone().unwrap_or_else(|| {
            format!("test_data/test_gpu_ggh15_poly_ntt_qlevel_{}", active_q_level)
        })
    }
}

fn q_level_from_env() -> Option<usize> {
    env::var("GGH15_POLY_NTT_Q_LEVEL").ok().map(|raw| {
        let level =
            raw.parse::<usize>().expect("GGH15_POLY_NTT_Q_LEVEL must be a positive integer");
        assert!(level > 0, "GGH15_POLY_NTT_Q_LEVEL must be greater than or equal to 1");
        level
    })
}

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
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
    for &q_i in all_q_moduli.iter().skip(active_q_moduli.len()) {
        assert_eq!(
            value % BigUint::from(q_i),
            BigUint::from(0u64),
            "inactive CRT residues must be zero when q_level is limited"
        );
    }
}

fn build_ntt_circuit_cpu(
    params: &DCRTPolyParams,
    q_level: Option<usize>,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    scale: u64,
    num_slots: usize,
) -> (PolyCircuit<DCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        p_moduli_bits,
        max_unreduced_muls,
        scale,
        false,
        q_level,
    ));
    let input = NestedRnsPoly::input(ctx.clone(), q_level, None, &mut circuit);
    let inverse = inverse_ntt(params, &mut circuit, &input, num_slots);
    let (reconstructed, approx_error) = inverse.reconstruct(&mut circuit);
    assert_eq!(approx_error, BigUint::ZERO);
    circuit.output(vec![reconstructed]);
    (circuit, ctx)
}

fn build_ntt_circuit_gpu(
    params: &GpuDCRTPolyParams,
    q_level: Option<usize>,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    scale: u64,
    num_slots: usize,
) -> (PolyCircuit<GpuDCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        p_moduli_bits,
        max_unreduced_muls,
        scale,
        false,
        q_level,
    ));
    let input = NestedRnsPoly::input(ctx.clone(), q_level, None, &mut circuit);
    let inverse = inverse_ntt(params, &mut circuit, &input, num_slots);
    let (reconstructed, approx_error) = inverse.reconstruct(&mut circuit);
    assert_eq!(approx_error, BigUint::ZERO);
    circuit.output(vec![reconstructed]);
    (circuit, ctx)
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

fn find_crt_depth_for_ntt(cfg: &NttConfig, q_level: Option<usize>) -> (usize, DCRTPolyParams) {
    let ring_dim_sqrt = BigDecimal::from_u32(cfg.ring_dim).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << cfg.base_bits, 0);
    let error_sigma = BigDecimal::from_f64(cfg.error_sigma).expect("valid error sigma");
    let e_init_norm = &error_sigma * BigDecimal::from(6u64);

    for crt_depth in 1..=cfg.max_crt_depth {
        let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
        let (active_q_moduli, _, _) = active_q_moduli_and_modulus(&params, q_level);
        let full_q = params.modulus();
        let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
        let (circuit, _ctx) = build_ntt_circuit_cpu(
            &params,
            q_level,
            cfg.p_moduli_bits,
            cfg.max_unreduced_muls,
            cfg.scale,
            cfg.num_slots,
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
        let plt_evaluator = ExactPltErrorNormEvaluator;
        let c_b0_error_norm =
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, BigDecimal::from(0u64), None);
        let slot_transfer_evaluator =
            NormBggPolyEncodingSTEvaluator::new(ctx.clone(), c_b0_error_norm, &error_sigma, None);
        let input_bound = BigDecimal::from((1u64 << cfg.p_moduli_bits) - 1);

        let out_errors = simulate_max_error_norm_with_slot_transfer(
            &circuit,
            ctx,
            input_bound,
            circuit.num_input(),
            &e_init_norm,
            &plt_evaluator,
            &slot_transfer_evaluator,
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

#[tokio::test]
async fn test_gpu_ggh15_poly_ntt() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    gpu_device_sync();
    let cfg = NttConfig::from_env();
    info!("poly ntt test config: {:?}", cfg);

    let q_level = q_level_from_env();
    let (crt_depth, cpu_params) = find_crt_depth_for_ntt(&cfg, q_level);
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
        .expect("at least one GPU device is required for test_gpu_ggh15_poly_ntt");
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
    let full_q = params.modulus();

    let (all_q_moduli, _, _) = params.to_crt();
    let (active_q_moduli, active_q, active_q_level) = active_q_moduli_and_modulus(&params, q_level);
    let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
    let (circuit, _ctx) = build_ntt_circuit_gpu(
        &params,
        q_level,
        cfg.p_moduli_bits,
        cfg.max_unreduced_muls,
        cfg.scale,
        cfg.num_slots,
    );
    info!("found crt_depth={}", crt_depth);
    info!(
        "selected crt_depth={} ring_dim={} num_slots={} crt_bits={} base_bits={} q_level={:?} q_moduli={:?}",
        crt_depth,
        params.ring_dimension(),
        cfg.num_slots,
        cfg.crt_bits,
        cfg.base_bits,
        q_level,
        all_q_moduli
    );
    info!(
        "circuit non_free_depth={} gate_counts={:?}",
        circuit.non_free_depth(),
        circuit.count_gates_by_type_vec()
    );
    info!(
        "active_q_level={} active_q_bits={} active_q_moduli_len={}",
        active_q_level,
        active_q.bits(),
        active_q_moduli.len()
    );

    let mut rng = rand::rng();
    let expected_coeffs_u64 =
        (0..cfg.num_slots).map(|_| rng.random_range(0..q_max)).collect::<Vec<_>>();
    let expected_coeffs =
        expected_coeffs_u64.iter().copied().map(BigUint::from).collect::<Vec<_>>();
    let input_eval_slots = DCRTPoly::from_biguints(&cpu_params, &expected_coeffs).eval_slots();
    assert_eq!(input_eval_slots.len(), cfg.num_slots, "evaluation slot count must match num_slots");

    let slot_input_parallelism = detected_gpu_count.max(1).min(cfg.num_slots);
    let flattened_input_count = circuit.num_input();
    let mut plaintext_inputs = (0..flattened_input_count)
        .map(|_| Vec::with_capacity(cfg.num_slots))
        .collect::<Vec<Vec<Arc<[u8]>>>>();

    let input_materialization_start = Instant::now();
    debug!(
        "slot input materialization started: num_slots={} chunk_size={} flattened_input_count={} q_level={:?}",
        cfg.num_slots, slot_input_parallelism, flattened_input_count, q_level
    );
    for slot_start in (0..cfg.num_slots).step_by(slot_input_parallelism) {
        let chunk_len = (slot_start + slot_input_parallelism).min(cfg.num_slots) - slot_start;
        let chunk_gpu_ids = &detected_gpu_ids[..chunk_len];
        let chunk_results = (0..chunk_len)
            .into_par_iter()
            .map(|offset| {
                let slot_idx = slot_start + offset;
                let gpu_id = chunk_gpu_ids[offset];
                let local_params = params.params_for_device(gpu_id);
                let encoded_inputs = encode_nested_rns_poly_compact_bytes::<GpuDCRTPoly>(
                    cfg.p_moduli_bits,
                    cfg.max_unreduced_muls,
                    &local_params,
                    &input_eval_slots[slot_idx],
                    q_level,
                )
                .into_iter()
                .map(Arc::<[u8]>::from)
                .collect::<Vec<_>>();
                (slot_idx, gpu_id, encoded_inputs)
            })
            .collect::<Vec<_>>();

        for (slot_idx, gpu_id, encoded_inputs) in chunk_results {
            assert_eq!(
                encoded_inputs.len(),
                flattened_input_count,
                "flattened encoded inputs must match circuit input count (slot_idx={}, gpu_id={})",
                slot_idx,
                gpu_id
            );
            for (input_idx, plaintext_bytes) in encoded_inputs.into_iter().enumerate() {
                plaintext_inputs[input_idx].push(plaintext_bytes);
            }
        }
    }
    info!(
        "slot input materialization elapsed_ms={:.3}",
        input_materialization_start.elapsed().as_secs_f64() * 1000.0
    );
    assert!(
        plaintext_inputs.iter().all(|slots| slots.len() == cfg.num_slots),
        "each plaintext input row must contain one encoded polynomial per slot"
    );

    let seed: [u8; 32] = [0u8; 32];
    let dir_name = cfg.dir_name(active_q_level);
    let dir = Path::new(&dir_name);
    if !dir.exists() {
        fs::create_dir_all(dir).expect("failed to create test directory");
    }
    init_storage_system(dir.to_path_buf());

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets =
        uniform_sampler.sample_uniform(&params, 1, cfg.d_secret, DistType::TernaryDist).get_row(0);
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
    info!("sampled secret vector with {} polynomials", secrets.len());

    let pk_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(seed, cfg.d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    info!("sampling public keys");
    let mut pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);
    info!("sampled {} public keys", pubkeys.len());

    let enc_setup_start = Instant::now();
    let encoding_sampler =
        BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    drop(secrets);
    let slot_secret_mats_start = Instant::now();
    let slot_secret_mats = encoding_sampler.sample_slot_secret_mats(&params, cfg.num_slots);
    info!(
        "sample_slot_secret_mats elapsed_ms={:.3}",
        slot_secret_mats_start.elapsed().as_secs_f64() * 1000.0
    );
    let mut poly_encodings =
        encoding_sampler.sample(&params, &pubkeys, &plaintext_inputs, Some(&slot_secret_mats));
    drop(plaintext_inputs);
    drop(encoding_sampler);
    info!(
        "poly encoding sampling elapsed_ms={:.3}",
        enc_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let pk_evaluator_setup_start = Instant::now();
    let plt_pubkey_evaluator =
        GGH15BGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(seed, cfg.d_secret, TRAPDOOR_SIGMA, cfg.error_sigma, dir.to_path_buf());
    let slot_pubkey_evaluator = BggPublicKeySTEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
    >::new(
        seed,
        cfg.d_secret,
        cfg.num_slots,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        dir.to_path_buf(),
    );
    info!(
        "pk evaluator setup elapsed_ms={:.3}",
        pk_evaluator_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let input_pubkeys = pubkeys.split_off(1);
    let one_pubkey = pubkeys.pop().expect("pubkeys must contain one entry for const one");
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
    assert_eq!(pubkey_out.len(), 1);

    let plt_sample_aux_start = Instant::now();
    plt_pubkey_evaluator.sample_aux_matrices(&params);
    info!(
        "plt_pubkey_sample_aux_matrices elapsed_ms={:.3}",
        plt_sample_aux_start.elapsed().as_secs_f64() * 1000.0
    );
    let slot_sample_aux_start = Instant::now();
    slot_pubkey_evaluator.sample_aux_matrices(&params, slot_secret_mats.clone());
    info!(
        "slot_pubkey_sample_aux_matrices elapsed_ms={:.3}",
        slot_sample_aux_start.elapsed().as_secs_f64() * 1000.0
    );
    let wait_writes_start = Instant::now();
    wait_for_all_writes(dir.to_path_buf()).await.expect("storage writes should complete");
    info!(
        "wait_for_all_writes elapsed_ms={:.3}",
        wait_writes_start.elapsed().as_secs_f64() * 1000.0
    );

    let plt_b0_matrix = plt_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("b0 matrix checkpoint should exist after public lookup auxiliary sampling");
    let slot_b0_matrix = slot_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("b0 matrix checkpoint should exist after slot-transfer auxiliary sampling");
    let plt_c_b0_compact_bytes_by_slot = GGH15BGGPolyEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::build_c_b0_compact_bytes_by_slot(
        &params, &s_vec, &plt_b0_matrix, &slot_secret_mats
    );
    let c_b0 = s_vec.clone() * &slot_b0_matrix;
    let plt_poly_evaluator = GGH15BGGPolyEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::new(
        seed,
        dir.to_path_buf(),
        plt_pubkey_evaluator.checkpoint_prefix(&params),
        plt_c_b0_compact_bytes_by_slot,
    );
    let slot_poly_evaluator =
        BggPolyEncodingSTEvaluator::<GpuDCRTPolyMatrix, GpuDCRTPolyHashSampler<Keccak256>>::new(
            seed,
            dir.to_path_buf(),
            slot_pubkey_evaluator.checkpoint_prefix(&params),
            c_b0.to_compact_bytes(),
        );

    let input_poly_encodings = poly_encodings.split_off(1);
    let one_poly_encoding =
        poly_encodings.pop().expect("poly encodings must contain one entry for const one");
    let poly_encoding_eval_start = Instant::now();
    let poly_out = circuit.eval(
        &params,
        one_poly_encoding,
        input_poly_encodings,
        Some(&plt_poly_evaluator),
        Some(&slot_poly_evaluator),
        Some(1),
    );
    info!(
        "poly encoding eval elapsed_ms={:.3}",
        poly_encoding_eval_start.elapsed().as_secs_f64() * 1000.0
    );
    assert_eq!(poly_out.len(), 1);

    let poly_out = &poly_out[0];
    assert_eq!(poly_out.pubkey, pubkey_out[0]);
    assert_eq!(poly_out.num_slots(), cfg.num_slots);
    let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, cfg.d_secret);
    let q_over_qmax = full_q.as_ref() / BigUint::from(q_max);

    for slot in 0..cfg.num_slots {
        let result_plaintext = poly_out
            .plaintext_for_params(&params, slot)
            .expect("poly output should reveal plaintexts");
        let plaintext_coeffs = result_plaintext.coeffs();
        let result_const_term = plaintext_coeffs
            .first()
            .expect("result plaintext polynomial must have at least one coefficient")
            .value()
            .clone();
        assert_value_matches_q_level(
            &result_const_term,
            &expected_coeffs[slot],
            &active_q,
            &active_q_moduli,
            &all_q_moduli,
        );
        let zero = BigUint::from(0u64);
        assert!(
            plaintext_coeffs.iter().skip(1).all(|coeff| coeff.value() == &zero),
            "result plaintext polynomial must remain constant across non-zero coefficients"
        );

        let expected_poly =
            GpuDCRTPoly::from_biguint_to_constant(&params, expected_coeffs[slot].clone());
        let slot_secret_mat =
            GpuDCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[slot].as_ref());
        let transformed_secret_vec = s_vec.clone() * &slot_secret_mat;
        let expected_times_gadget =
            transformed_secret_vec.clone() * (gadget.clone() * expected_poly);
        let s_times_pk = transformed_secret_vec.clone() * &poly_out.pubkey.matrix;
        let diff = poly_out.vector(slot) - s_times_pk + expected_times_gadget;
        let coeff = diff
            .entry(0, 0)
            .coeffs()
            .into_iter()
            .next()
            .expect("diff poly must have at least one coefficient")
            .value()
            .clone();

        let random_int: u64 = rand::random::<u64>() % q_max;
        let randomized_coeff = coeff + q_over_qmax.clone() * BigUint::from(random_int);
        let rounded = round_div_biguint(&randomized_coeff, &q_over_qmax);
        let decoded_random: u64 = (&rounded % BigUint::from(q_max))
            .try_into()
            .expect("decoded random coefficient must fit u64");
        assert_eq!(decoded_random, random_int);
    }

    gpu_device_sync();
}
