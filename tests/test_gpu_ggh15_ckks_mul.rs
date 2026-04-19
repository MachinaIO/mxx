#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    element::PolyElem,
    gadgets::{
        arith::encode_nested_rns_poly_with_offset,
        fhe::ckks::{CKKSCiphertext, CKKSContext, sample_relinearization_eval_key_slots},
    },
    lookup::{
        PltEvaluator,
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
        uniform::DCRTPolyUniformSampler,
    },
    simulator::{
        SimulatorContext,
        error_norm::{ErrorNorm, NormBggPolyEncodingSTEvaluator, NormPltGGH15Evaluator},
        poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::{BggPolyEncodingSTEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::{bigdecimal_bits_ceil, gen_biguint_for_modulus, mod_inverse},
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::{env, fs, path::Path, sync::Arc, time::Instant};
use tracing::{debug, info};

const DEFAULT_RING_DIM: u32 = 1 << 14;
const DEFAULT_NUM_SLOTS: usize = 1 << 14;
const DEFAULT_CRT_BITS: usize = 24;
const DEFAULT_P_MODULI_BITS: usize = 7;
const DEFAULT_SCALE: u64 = 256;
const DEFAULT_BASE_BITS: u32 = 12;
const DEFAULT_MAX_CRT_DEPTH: usize = 32;
const DEFAULT_RELIN_EXTRA_LEVELS: usize = 1;
const DEFAULT_MAX_UNREDUCED_MULS: usize = 2;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_PLAINTEXT_BITS: usize = DEFAULT_CRT_BITS / 2 - 1;
const TRAPDOOR_SIGMA: f64 = 4.578;

#[derive(Debug, Clone)]
struct CkksMulConfig {
    ring_dim: u32,
    num_slots: usize,
    crt_bits: usize,
    p_moduli_bits: usize,
    scale: u64,
    base_bits: u32,
    max_crt_depth: usize,
    relinearization_extra_levels: usize,
    max_unreduced_muls: usize,
    error_sigma: f64,
    d_secret: usize,
    plaintext_bits: usize,
    dir_name_override: Option<String>,
}

struct BuiltMulRescaleCircuit<P: Poly> {
    circuit: PolyCircuit<P>,
    ctx: Arc<CKKSContext<P>>,
    output_error_bounds: (BigUint, BigUint),
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

impl CkksMulConfig {
    fn from_env() -> Self {
        let ring_dim = env_or_parse_u32("GGH15_CKKS_MUL_RING_DIM", DEFAULT_RING_DIM);
        let num_slots = env_or_parse_usize("GGH15_CKKS_MUL_NUM_SLOTS", DEFAULT_NUM_SLOTS);
        let crt_bits = env_or_parse_usize("GGH15_CKKS_MUL_CRT_BITS", DEFAULT_CRT_BITS);
        let p_moduli_bits =
            env_or_parse_usize("GGH15_CKKS_MUL_P_MODULI_BITS", DEFAULT_P_MODULI_BITS);
        let scale = env_or_parse_u64("GGH15_CKKS_MUL_SCALE", DEFAULT_SCALE);
        let base_bits = env_or_parse_u32("GGH15_CKKS_MUL_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth =
            env_or_parse_usize("GGH15_CKKS_MUL_MAX_CRT_DEPTH", DEFAULT_MAX_CRT_DEPTH);
        let relinearization_extra_levels =
            env_or_parse_usize("GGH15_CKKS_MUL_RELIN_EXTRA_LEVELS", DEFAULT_RELIN_EXTRA_LEVELS);
        let max_unreduced_muls =
            env_or_parse_usize("GGH15_CKKS_MUL_MAX_UNREDUCED_MULS", DEFAULT_MAX_UNREDUCED_MULS);
        let error_sigma = env_or_parse_f64("GGH15_CKKS_MUL_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret = env_or_parse_usize("GGH15_CKKS_MUL_D_SECRET", DEFAULT_D_SECRET);
        let plaintext_bits =
            env_or_parse_usize("GGH15_CKKS_MUL_PLAINTEXT_BITS", DEFAULT_PLAINTEXT_BITS);
        let dir_name_override = env::var("GGH15_CKKS_MUL_DIR_NAME")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        assert!(ring_dim > 0, "GGH15_CKKS_MUL_RING_DIM must be > 0");
        assert!(num_slots > 0, "GGH15_CKKS_MUL_NUM_SLOTS must be > 0");
        assert!(num_slots.is_power_of_two(), "GGH15_CKKS_MUL_NUM_SLOTS must be a power of two");
        assert!(crt_bits > 0, "GGH15_CKKS_MUL_CRT_BITS must be > 0");
        assert!(p_moduli_bits > 0, "GGH15_CKKS_MUL_P_MODULI_BITS must be > 0");
        assert!(scale > 0, "GGH15_CKKS_MUL_SCALE must be > 0");
        assert!(base_bits > 0, "GGH15_CKKS_MUL_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "GGH15_CKKS_MUL_MAX_CRT_DEPTH must be > 0");
        assert!(relinearization_extra_levels > 0, "GGH15_CKKS_MUL_RELIN_EXTRA_LEVELS must be > 0");
        assert!(
            max_crt_depth > relinearization_extra_levels + 1,
            "GGH15_CKKS_MUL_MAX_CRT_DEPTH must exceed GGH15_CKKS_MUL_RELIN_EXTRA_LEVELS + 1 for mul+rescale"
        );
        assert!(max_unreduced_muls > 0, "GGH15_CKKS_MUL_MAX_UNREDUCED_MULS must be > 0");
        assert!(error_sigma >= 0.0, "GGH15_CKKS_MUL_ERROR_SIGMA must be >= 0");
        assert!(d_secret > 0, "GGH15_CKKS_MUL_D_SECRET must be > 0");
        assert!(plaintext_bits > 0, "GGH15_CKKS_MUL_PLAINTEXT_BITS must be > 0");

        Self {
            ring_dim,
            num_slots,
            crt_bits,
            p_moduli_bits,
            scale,
            base_bits,
            max_crt_depth,
            relinearization_extra_levels,
            max_unreduced_muls,
            error_sigma,
            d_secret,
            plaintext_bits,
            dir_name_override,
        }
    }

    fn min_crt_depth(&self) -> usize {
        self.relinearization_extra_levels + 2
    }

    fn active_levels_for_crt_depth(&self, crt_depth: usize) -> usize {
        assert!(
            crt_depth >= self.min_crt_depth(),
            "crt_depth {} must be at least {} for mul+rescale with relinearization_extra_levels={}",
            crt_depth,
            self.min_crt_depth(),
            self.relinearization_extra_levels
        );
        crt_depth - self.relinearization_extra_levels
    }

    fn plaintext_bound(&self) -> BigUint {
        BigUint::from(1u64) << self.plaintext_bits
    }

    fn dir_name(&self, crt_depth: usize) -> String {
        let active_levels = self.active_levels_for_crt_depth(crt_depth);
        self.dir_name_override.clone().unwrap_or_else(|| {
            format!(
                "test_data/test_gpu_ggh15_ckks_mul_active_{}_relin_{}",
                active_levels, self.relinearization_extra_levels
            )
        })
    }
}

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
}

fn q_window_modulus(q_moduli: &[u64], level_offset: usize, active_levels: usize) -> BigUint {
    q_moduli
        .iter()
        .skip(level_offset)
        .take(active_levels)
        .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
}

fn q_window_moduli(q_moduli: &[u64], level_offset: usize, active_levels: usize) -> Vec<u64> {
    assert!(
        level_offset + active_levels <= q_moduli.len(),
        "q-window [{}, {}) exceeds CRT depth {}",
        level_offset,
        level_offset + active_levels,
        q_moduli.len()
    );
    q_moduli[level_offset..level_offset + active_levels].to_vec()
}

fn crt_value_from_residues(moduli: &[u64], residues: &[u64]) -> BigUint {
    assert_eq!(moduli.len(), residues.len(), "CRT residues must match modulus count");
    let modulus = moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
    moduli.iter().zip(residues.iter()).fold(BigUint::from(0u64), |acc, (&q_i, &residue)| {
        let q_i_big = BigUint::from(q_i);
        let q_hat = &modulus / &q_i_big;
        let q_hat_mod_q_i = (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in u64");
        let q_hat_inv = mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must exist");
        (acc + BigUint::from(residue) * q_hat * BigUint::from(q_hat_inv)) % &modulus
    })
}

fn coeffs_from_eval_slots_for_q_window(
    params: &DCRTPolyParams,
    q_moduli: &[u64],
    slots: &[BigUint],
    level_offset: usize,
    active_levels: usize,
) -> Vec<BigUint> {
    let active_moduli = q_window_moduli(q_moduli, level_offset, active_levels);
    if level_offset == 0 && active_levels == q_moduli.len() {
        return DCRTPoly::from_biguints_eval(params, slots).coeffs_biguints();
    }

    let coeffs_by_tower = (level_offset..level_offset + active_levels)
        .map(|crt_idx| {
            DCRTPoly::from_biguints_eval_single_mod(params, crt_idx, slots).coeffs_biguints()
        })
        .collect::<Vec<_>>();
    let coeff_count = coeffs_by_tower.first().map(|coeffs| coeffs.len()).unwrap_or(0);
    assert!(
        coeffs_by_tower.iter().all(|coeffs| coeffs.len() == coeff_count),
        "single-mod coefficient vectors must have matching lengths"
    );

    (0..coeff_count)
        .map(|coeff_idx| {
            let residues = coeffs_by_tower
                .iter()
                .map(|coeffs| {
                    coeffs[coeff_idx]
                        .to_u64()
                        .expect("single-mod coefficient residue must fit in u64")
                })
                .collect::<Vec<_>>();
            crt_value_from_residues(&active_moduli, &residues)
        })
        .collect()
}

fn reduce_coeffs_modulo(coeffs: &[BigUint], modulus: &BigUint) -> Vec<BigUint> {
    coeffs.iter().map(|coeff| coeff % modulus).collect()
}

fn centered_modular_distance(actual: &BigUint, expected: &BigUint, modulus: &BigUint) -> BigUint {
    let forward = if actual >= expected { actual - expected } else { actual + modulus - expected };
    let backward = if expected >= actual { expected - actual } else { expected + modulus - actual };
    forward.min(backward)
}

fn random_slots(modulus: &BigUint, num_slots: usize) -> Vec<BigUint> {
    let mut rng = rand::rng();
    (0..num_slots).map(|_| gen_biguint_for_modulus(&mut rng, modulus)).collect()
}

fn sample_ternary_secret_key(params: &DCRTPolyParams) -> DCRTPoly {
    let sampler = DCRTPolyUniformSampler::new();
    sampler.sample_poly(params, &DistType::TernaryDist)
}

fn encrypt_zero_error_ciphertext(
    params: &DCRTPolyParams,
    q_moduli: &[u64],
    level_offset: usize,
    num_slots: usize,
    plaintext_slots: &[BigUint],
    secret_key: &DCRTPoly,
    active_levels: usize,
) -> (DCRTPoly, DCRTPoly) {
    let modulus = q_window_modulus(q_moduli, level_offset, active_levels);
    let c1_slots = random_slots(&modulus, num_slots);
    let c1 = DCRTPoly::from_biguints_eval(params, &c1_slots);
    let plaintext = DCRTPoly::from_biguints_eval(params, plaintext_slots);
    let c0 = plaintext - &(c1.clone() * secret_key);
    (c0, c1)
}

fn encode_eval_slots_with_offset_gpu(
    params: &GpuDCRTPolyParams,
    ctx: &CKKSContext<GpuDCRTPoly>,
    slots: &[BigUint],
    level_offset: usize,
    active_levels: usize,
) -> Vec<Vec<Vec<u8>>> {
    let slot_parallelism = detected_gpu_device_count();
    let device_ids = detected_gpu_device_ids();
    assert!(
        slot_parallelism > 0,
        "at least one GPU device is required for CKKS GPU input encoding"
    );
    assert!(
        slot_parallelism <= device_ids.len(),
        "detected GPU count must not exceed detected GPU id count: count={}, ids={}",
        slot_parallelism,
        device_ids.len()
    );

    let mut encoded_slot_bytes = Vec::with_capacity(slots.len());
    for (batch_idx, slot_batch) in slots.chunks(slot_parallelism).enumerate() {
        let batch_device_ids = &device_ids[..slot_batch.len()];
        debug!(
            "encoding slot batch {}: slot_count={} device_ids={:?}",
            batch_idx,
            slot_batch.len(),
            batch_device_ids
        );
        let batch_results = slot_batch
            .par_iter()
            .enumerate()
            .map(|(offset, slot)| {
                let device_id = batch_device_ids[offset];
                let local_params = params.params_for_device(device_id);
                encode_nested_rns_poly_with_offset::<GpuDCRTPoly>(
                    ctx.nested_rns.p_moduli_bits,
                    ctx.nested_rns.max_unreduced_muls,
                    &local_params,
                    slot,
                    level_offset,
                    Some(active_levels),
                )
                .into_iter()
                .map(|poly| poly.to_compact_bytes())
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        encoded_slot_bytes.extend(batch_results);
    }

    encoded_slot_bytes
}

fn ciphertext_inputs_from_polys(
    params: &GpuDCRTPolyParams,
    ctx: &CKKSContext<GpuDCRTPoly>,
    c0: &DCRTPoly,
    c1: &DCRTPoly,
    level_offset: usize,
    active_levels: usize,
) -> Vec<Vec<Vec<u8>>> {
    let c0_slots = c0.eval_slots();
    let c1_slots = c1.eval_slots();
    assert!(
        ctx.num_slots <= c0_slots.len(),
        "c0 eval slot count must cover num_slots: eval_slots={} num_slots={}",
        c0_slots.len(),
        ctx.num_slots
    );
    assert!(
        ctx.num_slots <= c1_slots.len(),
        "c1 eval slot count must cover num_slots: eval_slots={} num_slots={}",
        c1_slots.len(),
        ctx.num_slots
    );
    let c0_inputs = encode_eval_slots_with_offset_gpu(
        params,
        ctx,
        &c0_slots[..ctx.num_slots],
        level_offset,
        active_levels,
    );
    let c1_inputs = encode_eval_slots_with_offset_gpu(
        params,
        ctx,
        &c1_slots[..ctx.num_slots],
        level_offset,
        active_levels,
    );
    assert_eq!(c0_inputs.len(), c1_inputs.len(), "c0 and c1 encoded slot counts must match");
    c0_inputs
        .into_iter()
        .zip(c1_inputs)
        .map(|(mut c0_slot_inputs, c1_slot_inputs)| {
            c0_slot_inputs.extend(c1_slot_inputs);
            c0_slot_inputs
        })
        .collect()
}

fn slot_major_compact_bytes_to_plaintext_rows(
    encoded_slot_inputs: Vec<Vec<Vec<u8>>>,
    num_slots: usize,
) -> Vec<Vec<Arc<[u8]>>> {
    assert_eq!(
        encoded_slot_inputs.len(),
        num_slots,
        "encoded slot input count must match num_slots"
    );
    let flattened_input_count =
        encoded_slot_inputs.first().map(|slot_inputs| slot_inputs.len()).unwrap_or(0);
    assert!(
        encoded_slot_inputs.iter().all(|slot_inputs| slot_inputs.len() == flattened_input_count),
        "each encoded slot must contain the same flattened input count"
    );

    let mut plaintext_rows =
        (0..flattened_input_count).map(|_| Vec::with_capacity(num_slots)).collect::<Vec<_>>();
    for slot_inputs in encoded_slot_inputs {
        for (input_idx, bytes) in slot_inputs.into_iter().enumerate() {
            plaintext_rows[input_idx].push(Arc::<[u8]>::from(bytes));
        }
    }
    plaintext_rows
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

fn build_ckks_mul_rescale_circuit<P: Poly + 'static>(
    params: &P::Params,
    cfg: &CkksMulConfig,
    active_levels: usize,
) -> BuiltMulRescaleCircuit<P> {
    let build_start = Instant::now();
    let mut circuit = PolyCircuit::<P>::new();

    // Match the zero-input-error CKKS reference test; the configurable Gaussian noise is for the
    // outer GGH15 encoding samplers only.
    let ctx = Arc::new(CKKSContext::setup(
        &mut circuit,
        params,
        cfg.num_slots,
        cfg.p_moduli_bits,
        cfg.max_unreduced_muls,
        cfg.scale,
        false,
        None,
        cfg.relinearization_extra_levels,
        None,
    ));
    let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
    let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
    let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
    let product = lhs.mul(&rhs, &eval_keys, &mut circuit);
    let rescaled = product.rescale(&mut circuit);
    let output_error_bounds = rescaled.error_bounds.clone();
    let rescaled_c0_poly = rescaled.c0.reconstruct(&mut circuit);
    let rescaled_c1_poly = rescaled.c1.reconstruct(&mut circuit);
    circuit.output(vec![rescaled_c0_poly, rescaled_c1_poly]);

    info!(
        "ckks mul-rescale circuit build elapsed_ms={:.3}",
        build_start.elapsed().as_secs_f64() * 1000.0
    );

    BuiltMulRescaleCircuit { circuit, ctx, output_error_bounds }
}

fn find_crt_depth_for_ckks_mul(cfg: &CkksMulConfig) -> (usize, DCRTPolyParams) {
    let ring_dim_sqrt = BigDecimal::from_u32(cfg.ring_dim).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << cfg.base_bits, 0);
    let error_sigma = BigDecimal::from_f64(cfg.error_sigma).expect("valid error sigma");
    let e_init_norm = &error_sigma * BigDecimal::from_f32(6.5).unwrap();
    let input_bound = BigDecimal::from((1u64 << cfg.p_moduli_bits) - 1);

    for crt_depth in cfg.min_crt_depth()..=cfg.max_crt_depth {
        let active_levels = cfg.active_levels_for_crt_depth(crt_depth);
        let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
        let (all_q_moduli, _, actual_crt_depth) = params.to_crt();
        let kept_q_moduli =
            q_window_moduli(&all_q_moduli, cfg.relinearization_extra_levels, active_levels - 1);
        let kept_modulus =
            q_window_modulus(&all_q_moduli, cfg.relinearization_extra_levels, active_levels - 1);
        let q_max = *kept_q_moduli.iter().max().expect("kept CRT modulus list must not be empty");
        let threshold = &kept_modulus / BigUint::from(2u64 * q_max);
        let threshold_bd = BigDecimal::from_biguint(threshold.clone(), 0);
        let BuiltMulRescaleCircuit { circuit, .. } =
            build_ckks_mul_rescale_circuit::<DCRTPoly>(&params, cfg, active_levels);

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
        assert_eq!(out_errors.len(), 2, "mul-then-rescale circuit must output two error bounds");

        let max_error_bits = out_errors
            .iter()
            .map(|error| bigdecimal_bits_ceil(&error.matrix_norm.poly_norm.norm))
            .max()
            .expect("mul-then-rescale circuit must produce output error bounds");
        let all_ok = out_errors.iter().all(|error| error.matrix_norm.poly_norm.norm < threshold_bd);

        info!(
            "crt_depth={} active_levels={} q_bits={} max_error_bits={} threshold_bits={}",
            crt_depth,
            active_levels,
            kept_modulus.bits(),
            max_error_bits,
            bigdecimal_bits_ceil(&BigDecimal::from_biguint(threshold, 0))
        );

        if all_ok {
            info!(
                "selected crt_depth={} active_levels={} for CKKS mul-rescale with max_error_bits={}",
                crt_depth, active_levels, max_error_bits
            );
            return (crt_depth, params);
        }
    }

    panic!(
        "crt_depth satisfying error < q/(2*q_max) for the CKKS mul-rescale circuit was not found up to GGH15_CKKS_MUL_MAX_CRT_DEPTH ({})",
        cfg.max_crt_depth
    );
}

#[tokio::test]
async fn test_gpu_ggh15_ckks_mul() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    gpu_device_sync();

    let cfg = CkksMulConfig::from_env();
    info!("ckks mul test config: {:?}", cfg);

    let (crt_depth, cpu_params) = find_crt_depth_for_ckks_mul(&cfg);
    let active_levels = cfg.active_levels_for_crt_depth(crt_depth);
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
        .expect("at least one GPU device is required for test_gpu_ggh15_ckks_mul");
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
    let q_max = *all_q_moduli.iter().max().expect("CRT modulus list must not be empty");
    let kept_modulus =
        q_window_modulus(&all_q_moduli, cfg.relinearization_extra_levels, active_levels - 1);

    let BuiltMulRescaleCircuit { circuit, ctx, output_error_bounds } =
        build_ckks_mul_rescale_circuit::<GpuDCRTPoly>(&params, &cfg, active_levels);
    info!(
        "selected crt_depth={} ring_dim={} num_slots={} crt_bits={} base_bits={} active_levels={} relin_extra_levels={} q_moduli={:?}",
        crt_depth,
        params.ring_dimension(),
        cfg.num_slots,
        cfg.crt_bits,
        cfg.base_bits,
        active_levels,
        cfg.relinearization_extra_levels,
        all_q_moduli
    );
    let non_free_depth_contributions = circuit.non_free_depth_contributions();
    info!(
        "circuit non_free_depth_contributions={:?} gate_counts={:?}",
        non_free_depth_contributions,
        circuit.count_gates_by_type_vec()
    );

    let input_setup_start = Instant::now();
    let plaintext_bound = cfg.plaintext_bound();
    let scale = BigUint::from(cfg.scale);
    let secret_key = sample_ternary_secret_key(&cpu_params);
    let eval_key_polys = sample_relinearization_eval_key_slots(
        &cpu_params,
        &secret_key,
        cfg.relinearization_extra_levels,
        cfg.error_sigma,
    );
    let lhs_plain_slots = random_slots(&plaintext_bound, cfg.num_slots);
    let rhs_plain_slots = random_slots(&plaintext_bound, cfg.num_slots);
    let lhs_scaled_slots = lhs_plain_slots.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
    let rhs_scaled_slots = rhs_plain_slots.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
    let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
        &cpu_params,
        &all_q_moduli,
        cfg.relinearization_extra_levels,
        cfg.num_slots,
        &lhs_scaled_slots,
        &secret_key,
        active_levels,
    );
    let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
        &cpu_params,
        &all_q_moduli,
        cfg.relinearization_extra_levels,
        cfg.num_slots,
        &rhs_scaled_slots,
        &secret_key,
        active_levels,
    );
    info!(
        "ckks input synthesis elapsed_ms={:.3}",
        input_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let flattened_input_count = circuit.num_input();
    let input_materialization_start = Instant::now();
    debug!(
        "slot input materialization started: num_slots={} flattened_input_count={} active_levels={} relin_extra_levels={}",
        cfg.num_slots, flattened_input_count, active_levels, cfg.relinearization_extra_levels
    );
    let lhs_input_bytes_by_slot = ciphertext_inputs_from_polys(
        &params,
        ctx.as_ref(),
        &lhs_c0,
        &lhs_c1,
        cfg.relinearization_extra_levels,
        active_levels,
    );
    let rhs_input_bytes_by_slot = ciphertext_inputs_from_polys(
        &params,
        ctx.as_ref(),
        &rhs_c0,
        &rhs_c1,
        cfg.relinearization_extra_levels,
        active_levels,
    );
    let eval_key_input_bytes_by_slot = ciphertext_inputs_from_polys(
        &params,
        ctx.as_ref(),
        &eval_key_polys.b0,
        &eval_key_polys.b1,
        0,
        crt_depth,
    );
    assert_eq!(
        lhs_input_bytes_by_slot.len(),
        cfg.num_slots,
        "lhs encoded slot count must match num_slots"
    );
    assert_eq!(
        rhs_input_bytes_by_slot.len(),
        cfg.num_slots,
        "rhs encoded slot count must match num_slots"
    );
    assert_eq!(
        eval_key_input_bytes_by_slot.len(),
        cfg.num_slots,
        "eval key encoded slot count must match num_slots"
    );
    let encoded_input_bytes_by_slot = lhs_input_bytes_by_slot
        .into_iter()
        .zip(rhs_input_bytes_by_slot)
        .zip(eval_key_input_bytes_by_slot)
        .map(|((mut lhs_slot_inputs, rhs_slot_inputs), eval_key_slot_inputs)| {
            lhs_slot_inputs.extend(rhs_slot_inputs);
            lhs_slot_inputs.extend(eval_key_slot_inputs);
            lhs_slot_inputs
        })
        .collect::<Vec<_>>();
    assert_eq!(
        encoded_input_bytes_by_slot.first().map(|slot_inputs| slot_inputs.len()).unwrap_or(0),
        flattened_input_count,
        "flattened encoded inputs per slot must match circuit input count"
    );
    let plaintext_inputs =
        slot_major_compact_bytes_to_plaintext_rows(encoded_input_bytes_by_slot, cfg.num_slots);
    info!(
        "slot input materialization elapsed_ms={:.3}",
        input_materialization_start.elapsed().as_secs_f64() * 1000.0
    );
    debug!(
        "slot input materialization finished: flattened_input_count={} plaintext_rows={}",
        flattened_input_count,
        plaintext_inputs.len()
    );
    assert!(
        plaintext_inputs.iter().all(|slots| slots.len() == cfg.num_slots),
        "each plaintext input row must contain one encoded polynomial per slot"
    );

    let seed: [u8; 32] = [0u8; 32];
    let dir_name = cfg.dir_name(crt_depth);
    let dir = Path::new(&dir_name);
    if !dir.exists() {
        fs::create_dir_all(dir).expect("failed to create test directory");
    }
    init_storage_system(dir.to_path_buf());

    let secret_sampling_start = Instant::now();
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets =
        uniform_sampler.sample_uniform(&params, 1, cfg.d_secret, DistType::TernaryDist).get_row(0);
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
    info!(
        "ggh secret sampling elapsed_ms={:.3}",
        secret_sampling_start.elapsed().as_secs_f64() * 1000.0
    );
    info!("sampled secret vector with {} polynomials", secrets.len());

    let pk_sampling_start = Instant::now();
    let pk_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(seed, cfg.d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    let mut pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);
    let pubkeys_for_poly_encodings = pubkeys.clone();
    info!(
        "public key sampling elapsed_ms={:.3}",
        pk_sampling_start.elapsed().as_secs_f64() * 1000.0
    );
    info!("sampled {} public keys", pubkeys.len());

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
    assert_eq!(pubkey_out.len(), 2, "mul-then-rescale circuit must output two public keys");

    let plt_sample_aux_start = Instant::now();
    plt_pubkey_evaluator.sample_aux_matrices(&params);
    info!(
        "plt_pubkey_sample_aux_matrices elapsed_ms={:.3}",
        plt_sample_aux_start.elapsed().as_secs_f64() * 1000.0
    );
    let slot_sample_aux_start = Instant::now();
    slot_pubkey_evaluator.sample_aux_matrices(&params);
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
    let slot_secret_mats_load_start = Instant::now();
    let slot_secret_mats = slot_pubkey_evaluator.load_slot_secret_mats_checkpoint(&params).expect(
        "slot secret matrix checkpoints should exist after slot-transfer auxiliary sampling",
    );
    info!(
        "load_slot_secret_mats_checkpoint elapsed_ms={:.3}",
        slot_secret_mats_load_start.elapsed().as_secs_f64() * 1000.0
    );

    let poly_encoding_sampling_start = Instant::now();
    let encoding_sampler = BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(
        &params,
        &secrets,
        Some(cfg.error_sigma),
    );
    drop(secrets);
    let mut poly_encodings = encoding_sampler.sample(
        &params,
        &pubkeys_for_poly_encodings,
        &plaintext_inputs,
        Some(&slot_secret_mats),
    );
    drop(plaintext_inputs);
    drop(encoding_sampler);
    info!(
        "poly encoding sampling elapsed_ms={:.3}",
        poly_encoding_sampling_start.elapsed().as_secs_f64() * 1000.0
    );

    let b0_load_start = Instant::now();
    let plt_b0_matrix = plt_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("b0 matrix checkpoint should exist after public lookup auxiliary sampling");
    let slot_b0_matrix = slot_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("b0 matrix checkpoint should exist after slot-transfer auxiliary sampling");
    info!("load_b0_checkpoints elapsed_ms={:.3}", b0_load_start.elapsed().as_secs_f64() * 1000.0);

    let poly_evaluator_setup_start = Instant::now();
    let plt_c_b0_compact_bytes_by_slot = GGH15BGGPolyEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::build_c_b0_compact_bytes_by_slot::<
        GpuDCRTPolyUniformSampler,
    >(
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
            slot_c_b0.to_compact_bytes(),
        );
    info!(
        "poly evaluator setup elapsed_ms={:.3}",
        poly_evaluator_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let input_poly_encodings = poly_encodings.split_off(1);
    let one_poly_encoding =
        poly_encodings.pop().expect("poly encodings must contain one entry for const one");
    let poly_encoding_eval_start = Instant::now();
    let output_encodings = circuit.eval(
        &params,
        one_poly_encoding,
        input_poly_encodings,
        Some(&plt_poly_evaluator),
        Some(&slot_poly_evaluator),
        None,
    );
    info!(
        "poly encoding eval elapsed_ms={:.3}",
        poly_encoding_eval_start.elapsed().as_secs_f64() * 1000.0
    );
    assert_eq!(
        output_encodings.len(),
        2,
        "mul-then-rescale circuit must output two poly encodings"
    );

    let plaintext_assert_start = Instant::now();
    let mut output_eval_slots = Vec::with_capacity(output_encodings.len());
    for (output_idx, output_encoding) in output_encodings.iter().enumerate() {
        assert_eq!(output_encoding.pubkey, pubkey_out[output_idx]);
        assert_eq!(output_encoding.num_slots(), cfg.num_slots);

        let mut slots = Vec::with_capacity(cfg.num_slots);
        for slot in 0..cfg.num_slots {
            let result_plaintext = output_encoding
                .plaintext_for_params(&params, slot)
                .expect("poly output should reveal plaintexts");
            let plaintext_coeffs = result_plaintext.coeffs();
            let constant_coeff = plaintext_coeffs
                .first()
                .expect("result plaintext polynomial must have at least one coefficient")
                .value()
                .clone();
            let zero = BigUint::from(0u64);
            assert!(
                plaintext_coeffs.iter().skip(1).all(|coeff| coeff.value() == &zero),
                "output {} slot {} plaintext polynomial must remain constant across non-zero coefficients",
                output_idx,
                slot
            );
            slots.push(constant_coeff);
        }
        output_eval_slots.push(slots);
    }

    let rescaled_c0_coeffs = coeffs_from_eval_slots_for_q_window(
        &cpu_params,
        &all_q_moduli,
        &output_eval_slots[0],
        cfg.relinearization_extra_levels,
        active_levels - 1,
    );
    let rescaled_c1_coeffs = coeffs_from_eval_slots_for_q_window(
        &cpu_params,
        &all_q_moduli,
        &output_eval_slots[1],
        cfg.relinearization_extra_levels,
        active_levels - 1,
    );
    let rescaled_c0_coeff_poly = DCRTPoly::from_biguints(&cpu_params, &rescaled_c0_coeffs);
    let rescaled_c1_coeff_poly = DCRTPoly::from_biguints(&cpu_params, &rescaled_c1_coeffs);

    let secret_key_coeffs_kept = reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &kept_modulus);
    let secret_key_coeff_poly_kept = DCRTPoly::from_biguints(&cpu_params, &secret_key_coeffs_kept);
    let decrypted_coeff_poly =
        rescaled_c0_coeff_poly + &(rescaled_c1_coeff_poly * &secret_key_coeff_poly_kept);
    let actual_coeffs =
        reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &kept_modulus);

    let expected_scaled_product_slots = lhs_plain_slots
        .iter()
        .zip(rhs_plain_slots.iter())
        .map(|(lhs, rhs)| lhs * rhs * &scale * &scale)
        .collect::<Vec<_>>();
    let expected_pre_rescale_coeffs = coeffs_from_eval_slots_for_q_window(
        &cpu_params,
        &all_q_moduli,
        &expected_scaled_product_slots,
        cfg.relinearization_extra_levels,
        active_levels,
    );
    let removed_modulus_u64 = all_q_moduli[cfg.relinearization_extra_levels + active_levels - 1];
    let removed_modulus = BigUint::from(removed_modulus_u64);
    let expected_coeffs = expected_pre_rescale_coeffs
        .iter()
        .map(|coeff| coeff / &removed_modulus)
        .collect::<Vec<_>>();

    let secret_key_bound = BigUint::from(params.ring_dimension());
    let total_bound = &output_error_bounds.0 + (&secret_key_bound * &output_error_bounds.1);
    info!("mul-then-rescale total_bound={}", total_bound);
    actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
        |(coeff_idx, (actual, expected))| {
            let diff = centered_modular_distance(actual, expected, &kept_modulus);
            if diff > total_bound {
                info!(
                    "coeff_idx={} actual={} expected={} diff={} bound={}",
                    coeff_idx, actual, expected, diff, total_bound
                );
            }
            assert!(
                diff <= total_bound,
                "mul-then-rescale decrypted coefficient {} error {} exceeds bound {}",
                coeff_idx,
                diff,
                total_bound
            );
        },
    );
    info!(
        "plaintext decode and coeff bound check elapsed_ms={:.3}",
        plaintext_assert_start.elapsed().as_secs_f64() * 1000.0
    );

    let vector_assert_start = Instant::now();
    let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, cfg.d_secret);
    let q_over_qmax = full_q.as_ref() / BigUint::from(q_max);
    for (output_idx, output_encoding) in output_encodings.iter().enumerate() {
        for slot in 0..cfg.num_slots {
            info!("verifying output {} slot {} of {}", output_idx, slot + 1, cfg.num_slots);
            let slot_secret_mat =
                GpuDCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[slot].as_ref());
            let transformed_secret_vec = s_vec.clone() * &slot_secret_mat;
            let expected_times_gadget = transformed_secret_vec.clone() *
                (gadget.clone() * output_encoding.plaintext(slot).unwrap());
            let s_times_pk = transformed_secret_vec.clone() * &output_encoding.pubkey.matrix;
            let diff = output_encoding.vector(slot) - s_times_pk + expected_times_gadget;
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
    }
    info!(
        "vector consistency checks elapsed_ms={:.3}",
        vector_assert_start.elapsed().as_secs_f64() * 1000.0
    );

    gpu_device_sync();
}
