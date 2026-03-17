#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    element::PolyElem,
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly},
    lookup::ggh15_eval::{GGH15BGGPolyEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator},
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{
                GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_count, detected_gpu_device_ids,
                gpu_device_sync,
            },
            params::DCRTPolyParams,
        },
    },
    sampler::{
        DistType, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    simulator::{SimulatorContext, error_norm::NormPltGGH15Evaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::{bigdecimal_bits_ceil, gen_biguint_for_modulus},
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{env, fs, path::Path, sync::Arc, time::Instant};
use tracing::{debug, info};

const DEFAULT_RING_DIM: u32 = 1 << 14;
const DEFAULT_CRT_BITS: usize = 24;
const DEFAULT_P_MODULI_BITS: usize = 6;
const DEFAULT_SCALE: u64 = 1 << 7;
const DEFAULT_BASE_BITS: u32 = 12;
const DEFAULT_MAX_CRT_DEPTH: usize = 32;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_HEIGHT: usize = 1;
const DEFAULT_NUM_SLOTS: usize = 1024;

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
    num_slots: usize,
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
        let num_slots = env_or_parse_usize("GGH15_POLY_MODQ_ARITH_NUM_SLOTS", DEFAULT_NUM_SLOTS);
        let dir_name_override = env::var("GGH15_POLY_MODQ_ARITH_DIR_NAME")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        assert!(ring_dim > 0, "GGH15_MODQ_ARITH_RING_DIM must be > 0");
        assert!(crt_bits > 0, "GGH15_MODQ_ARITH_CRT_BITS must be > 0");
        assert!(p_moduli_bits > 0, "GGH15_MODQ_ARITH_P_MODULI_BITS must be > 0");
        assert!(scale > 0, "GGH15_MODQ_ARITH_SCALE must be > 0");
        assert!(base_bits > 0, "GGH15_MODQ_ARITH_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "GGH15_MODQ_ARITH_MAX_CRT_DEPTH must be > 0");
        assert!(error_sigma > 0.0, "GGH15_MODQ_ARITH_ERROR_SIGMA must be > 0");
        assert!(d_secret > 0, "GGH15_MODQ_ARITH_D_SECRET must be > 0");
        assert!(num_slots > 0, "GGH15_POLY_MODQ_ARITH_NUM_SLOTS must be > 0");

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
            num_slots,
            dir_name_override,
        }
    }

    fn dir_name(&self, active_q_level: usize) -> String {
        self.dir_name_override.clone().unwrap_or_else(|| {
            format!("test_data/test_gpu_ggh15_poly_modq_arith_qlevel_{}", active_q_level)
        })
    }
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

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
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
) -> (PolyCircuit<mxx::poly::dcrt::poly::DCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<mxx::poly::dcrt::poly::DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        p_moduli_bits,
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
    let e_init_norm = &error_sigma * BigDecimal::from(6u64);

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
async fn test_gpu_ggh15_poly_modq_arith() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    gpu_device_sync();
    let cfg = ModqArithConfig::from_env();
    info!("poly modq arith test config: {:?}", cfg);

    let q_level = q_level_from_env();
    let logical_num_inputs =
        1usize.checked_shl(cfg.height as u32).expect("GGH15_MODQ_ARITH_HEIGHT is too large");
    let (crt_depth, cpu_params) = find_crt_depth_for_modq_arith(&cfg, q_level);
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
        .expect("at least one GPU device is required for test_gpu_ggh15_poly_modq_arith");
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
    let (circuit, _ctx) =
        build_modq_arith_circuit_gpu(&params, q_level, cfg.p_moduli_bits, cfg.scale, cfg.height);
    info!("found crt_depth={}", crt_depth);
    info!(
        "selected crt_depth={} ring_dim={} crt_bits={} base_bits={} q_level={:?} q_modulo={:?}",
        crt_depth,
        params.ring_dimension(),
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
    info!(
        "multiplication tree config: height={} num_inputs={} num_slots={}",
        cfg.height, logical_num_inputs, cfg.num_slots
    );

    let slot_input_parallelism = detected_gpu_count.max(1).min(cfg.num_slots);
    info!(
        "slot input materialization parallelism={} detected_gpu_count={} eval_gpu_ids={:?}",
        slot_input_parallelism,
        detected_gpu_count,
        params.gpu_ids()
    );

    let flattened_input_count = circuit.num_input();
    let mut expected_by_slot = Vec::with_capacity(cfg.num_slots);
    let mut plaintext_inputs = (0..flattened_input_count)
        .map(|_| Vec::with_capacity(cfg.num_slots))
        .collect::<Vec<Vec<GpuDCRTPoly>>>();

    debug!(
        "slot input materialization started: num_slots={} chunk_size={} flattened_input_count={} logical_num_inputs={} q_level={:?}",
        cfg.num_slots, slot_input_parallelism, flattened_input_count, logical_num_inputs, q_level
    );
    for slot_start in (0..cfg.num_slots).step_by(slot_input_parallelism) {
        let chunk_len = (slot_start + slot_input_parallelism).min(cfg.num_slots) - slot_start;
        let chunk_gpu_ids = &detected_gpu_ids[..chunk_len];
        debug!(
            "slot input chunk started: slot_range=[{}, {}), chunk_len={}, gpu_ids={:?}, expected_slots_before={}",
            slot_start,
            slot_start + chunk_len,
            chunk_len,
            chunk_gpu_ids,
            expected_by_slot.len()
        );
        let chunk_results = (0..chunk_len)
            .into_par_iter()
            .map(|offset| {
                let slot_idx = slot_start + offset;
                let gpu_id = chunk_gpu_ids[offset];
                debug!(
                    "slot input slot started: slot_idx={} gpu_id={} logical_num_inputs={}",
                    slot_idx,
                    gpu_id,
                    logical_num_inputs
                );
                let local_params = params.params_for_device(gpu_id);
                debug!(
                    "slot input slot params prepared: slot_idx={} gpu_id={} local_gpu_ids={:?}",
                    slot_idx,
                    gpu_id,
                    local_params.gpu_ids()
                );
                let mut rng = rand::rng();
                let input_values = (0..logical_num_inputs)
                    .map(|_| gen_biguint_for_modulus(&mut rng, &active_q))
                    .collect::<Vec<_>>();
                debug!(
                    "slot input values sampled: slot_idx={} gpu_id={} sampled_inputs={}",
                    slot_idx,
                    gpu_id,
                    input_values.len()
                );
                let expected = input_values
                    .iter()
                    .fold(BigUint::from(1u64), |acc, value| (acc * value) % &active_q);
                let encoded_inputs = input_values
                    .iter()
                    .enumerate()
                    .flat_map(|(input_idx, value)| {
                        debug!(
                            "slot input encode started: slot_idx={} gpu_id={} input_idx={} q_level={:?}",
                            slot_idx,
                            gpu_id,
                            input_idx,
                            q_level
                        );
                        let encoded = encode_nested_rns_poly::<GpuDCRTPoly>(
                            cfg.p_moduli_bits,
                            &local_params,
                            value,
                            q_level,
                        );
                        debug!(
                            "slot input encode finished: slot_idx={} gpu_id={} input_idx={} encoded_polys={}",
                            slot_idx,
                            gpu_id,
                            input_idx,
                            encoded.len()
                        );
                        encoded
                    })
                    .collect::<Vec<_>>();
                debug!(
                    "slot input slot finished: slot_idx={} gpu_id={} encoded_inputs_len={}",
                    slot_idx,
                    gpu_id,
                    encoded_inputs.len()
                );
                (slot_idx, gpu_id, expected, encoded_inputs)
            })
            .collect::<Vec<_>>();
        debug!(
            "slot input chunk encoded: slot_range=[{}, {}), completed_slots={}",
            slot_start,
            slot_start + chunk_len,
            chunk_results.len()
        );

        for (slot_idx, gpu_id, expected, encoded_inputs) in chunk_results {
            assert_eq!(
                encoded_inputs.len(),
                flattened_input_count,
                "flattened encoded inputs must match circuit input count (slot_idx={}, gpu_id={})",
                slot_idx,
                gpu_id
            );
            expected_by_slot.push(expected);
            for (input_idx, poly) in encoded_inputs.into_iter().enumerate() {
                plaintext_inputs[input_idx].push(poly);
            }
        }
        debug!(
            "slot input chunk committed: slot_range=[{}, {}), expected_slots_after={}",
            slot_start,
            slot_start + chunk_len,
            expected_by_slot.len()
        );
    }
    debug!(
        "slot input materialization finished: total_slots={} flattened_input_count={}",
        expected_by_slot.len(),
        flattened_input_count
    );

    assert!(
        plaintext_inputs.iter().all(|slots| slots.len() == cfg.num_slots),
        "each plaintext input row must contain one encoded polynomial per slot"
    );

    let seed: [u8; 32] = [0u8; 32];
    let trapdoor_sigma = 4.578;
    let d_secret = cfg.d_secret;

    let dir_name = cfg.dir_name(active_q_level);
    let dir = Path::new(&dir_name);
    if !dir.exists() {
        fs::create_dir_all(dir).expect("failed to create test directory");
    }
    init_storage_system(dir.to_path_buf());

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets =
        uniform_sampler.sample_uniform(&params, 1, d_secret, DistType::TernaryDist).get_row(0);
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
    info!("sampled secret vector with {} polynomials", secrets.len());

    let pk_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(seed, d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    info!("sampling public keys");
    let mut pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);
    info!("sampled {} public keys", pubkeys.len());

    let enc_setup_start = Instant::now();
    let encoding_sampler =
        BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    drop(secrets);
    let slot_secret_mats = encoding_sampler.sample_slot_secret_mats(&params, cfg.num_slots);
    let poly_encodings = encoding_sampler.sample_with_slot_secret_mats(
        &params,
        &pubkeys,
        &plaintext_inputs,
        &slot_secret_mats,
    );
    drop(plaintext_inputs);
    drop(encoding_sampler);
    info!(
        "poly encoding sampling elapsed_ms={:.3}",
        enc_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let pk_evaluator_setup_start = Instant::now();
    let pk_evaluator =
        GGH15BGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(seed, d_secret, trapdoor_sigma, cfg.error_sigma, dir.to_path_buf());
    info!(
        "pk evaluator setup elapsed_ms={:.3}",
        pk_evaluator_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let input_pubkeys = pubkeys.split_off(1);
    let one_pubkey = pubkeys.pop().expect("pubkeys must contain one entry for const one");
    let pubkey_eval_start = Instant::now();
    let pubkey_out = circuit.eval(&params, one_pubkey, input_pubkeys, Some(&pk_evaluator), None);
    info!("pubkey eval elapsed_ms={:.3}", pubkey_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(pubkey_out.len(), 1);

    let sample_aux_start = Instant::now();
    pk_evaluator.sample_aux_matrices(&params);
    info!(
        "sample_aux_matrices elapsed_ms={:.3}",
        sample_aux_start.elapsed().as_secs_f64() * 1000.0
    );
    let wait_writes_start = Instant::now();
    wait_for_all_writes(dir.to_path_buf()).await.expect("storage writes should complete");
    info!(
        "wait_for_all_writes elapsed_ms={:.3}",
        wait_writes_start.elapsed().as_secs_f64() * 1000.0
    );

    let b0_matrix = pk_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
    let checkpoint_prefix = pk_evaluator.checkpoint_prefix(&params);
    let poly_evaluator = GGH15BGGPolyEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::new(
        seed,
        dir.to_path_buf(),
        checkpoint_prefix,
        &params,
        s_vec.clone(),
        b0_matrix,
        slot_secret_mats.clone(),
    );

    let mut poly_encodings = poly_encodings;
    let input_poly_encodings = poly_encodings.split_off(1);
    let one_poly_encoding =
        poly_encodings.pop().expect("poly encodings must contain one entry for const one");
    let poly_encoding_eval_start = Instant::now();
    let poly_out = circuit.eval(
        &params,
        one_poly_encoding,
        input_poly_encodings,
        Some(&poly_evaluator),
        Some(1),
    );
    info!(
        "poly encoding eval elapsed_ms={:.3}",
        poly_encoding_eval_start.elapsed().as_secs_f64() * 1000.0
    );
    assert_eq!(poly_out.len(), 1);

    let poly_out = &poly_out[0];
    assert_eq!(poly_out.pubkey, pubkey_out[0]);
    let result_plaintexts =
        poly_out.plaintexts.as_ref().expect("poly lookup result should reveal plaintexts");
    assert_eq!(result_plaintexts.len(), cfg.num_slots);
    assert_eq!(poly_out.num_slots(), cfg.num_slots);
    let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, d_secret);

    for slot in 0..cfg.num_slots {
        let plaintext_coeffs = result_plaintexts[slot].coeffs();
        let result_const_term = plaintext_coeffs
            .first()
            .expect("result plaintext polynomial must have at least one coefficient")
            .value()
            .clone();
        assert_value_matches_q_level(
            &result_const_term,
            &expected_by_slot[slot],
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
            GpuDCRTPoly::from_biguint_to_constant(&params, expected_by_slot[slot].clone());
        let transformed_secret_vec = s_vec.clone() * &slot_secret_mats[slot];
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
        let q_over_qmax = full_q.as_ref() / BigUint::from(q_max);
        let randomized_coeff = coeff + q_over_qmax.clone() * BigUint::from(random_int);
        let rounded = round_div_biguint(&randomized_coeff, &q_over_qmax);
        let decoded_random: u64 = (&rounded % BigUint::from(q_max))
            .try_into()
            .expect("decoded random coefficient must fit u64");
        assert_eq!(decoded_random, random_int);
    }

    gpu_device_sync();
}
