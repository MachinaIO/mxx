#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    element::PolyElem,
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly},
    lookup::{
        ggh15_eval::{GGH15BGGEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator},
        poly::PolyPltEvaluator,
    },
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
            params::DCRTPolyParams,
            poly::DCRTPoly,
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
use std::{env, fs, path::Path, sync::Arc, time::Instant};
use tracing::info;

const DEFAULT_RING_DIM: u32 = 1 << 14;
const DEFAULT_CRT_BITS: usize = 24;
const DEFAULT_P_MODULI_BITS: usize = 6;
const DEFAULT_SCALE: u64 = 1 << 7;
const DEFAULT_BASE_BITS: u32 = 12;
const DEFAULT_MAX_CRT_DEPTH: usize = 32;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;

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
        let dir_name_override = env::var("GGH15_MODQ_ARITH_DIR_NAME")
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

        Self {
            ring_dim,
            crt_bits,
            p_moduli_bits,
            scale,
            base_bits,
            max_crt_depth,
            error_sigma,
            d_secret,
            dir_name_override,
        }
    }

    fn dir_name(&self, active_q_level: usize) -> String {
        self.dir_name_override.clone().unwrap_or_else(|| {
            format!("test_data/test_gpu_ggh15_modq_arith_qlevel_{}", active_q_level)
        })
    }
}

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
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
) -> (PolyCircuit<DCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        p_moduli_bits,
        scale,
        false,
        q_level,
    ));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);
    circuit.output(vec![out]);
    (circuit, ctx)
}

fn build_modq_arith_circuit_gpu(
    params: &GpuDCRTPolyParams,
    q_level: Option<usize>,
    p_moduli_bits: usize,
    scale: u64,
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

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);
    circuit.output(vec![out]);
    (circuit, ctx)
}

fn build_modq_arith_value_circuit_gpu(
    params: &GpuDCRTPolyParams,
    q_level: Option<usize>,
    p_moduli_bits: usize,
    scale: u64,
) -> PolyCircuit<GpuDCRTPoly> {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        p_moduli_bits,
        scale,
        false,
        q_level,
    ));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);
    circuit.output(vec![out]);
    circuit
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
        let (circuit, _ctx) =
            build_modq_arith_circuit_cpu(&params, q_level, cfg.p_moduli_bits, cfg.scale);

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
async fn test_gpu_ggh15_modq_arith() {
    let _ = tracing_subscriber::fmt::try_init();
    gpu_device_sync();
    let cfg = ModqArithConfig::from_env();
    info!("modq arith test config: {:?}", cfg);

    let q_level = q_level_from_env();
    let (crt_depth, cpu_params) = find_crt_depth_for_modq_arith(&cfg, q_level);
    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli.clone(), cpu_params.base_bits());
    let single_gpu_id = *detected_gpu_params
        .gpu_ids()
        .first()
        .expect("at least one GPU device is required for test_gpu_ggh15_modq_arith");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
        detected_gpu_params.batch(),
    );
    info!("forcing single GPU for this test: gpu_id={}", single_gpu_id);
    assert_eq!(params.modulus(), cpu_params.modulus());
    let full_q = params.modulus();

    let (all_q_moduli, _, _) = params.to_crt();
    let (active_q_moduli, active_q, active_q_level) = active_q_moduli_and_modulus(&params, q_level);
    let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
    let (circuit, _ctx) =
        build_modq_arith_circuit_gpu(&params, q_level, cfg.p_moduli_bits, cfg.scale);
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

    let mut rng = rand::rng();
    let a_value: BigUint = gen_biguint_for_modulus(&mut rng, &active_q);
    let b_value: BigUint = gen_biguint_for_modulus(&mut rng, &active_q);
    let expected = (&a_value * &b_value) % &active_q;

    let a_inputs: Vec<GpuDCRTPoly> =
        encode_nested_rns_poly(cfg.p_moduli_bits, &params, &a_value, q_level);
    let b_inputs: Vec<GpuDCRTPoly> =
        encode_nested_rns_poly(cfg.p_moduli_bits, &params, &b_value, q_level);
    let plaintext_inputs = [a_inputs.clone(), b_inputs.clone()].concat();

    let dry_circuit =
        build_modq_arith_value_circuit_gpu(&params, q_level, cfg.p_moduli_bits, cfg.scale);
    let dry_plt_evaluator = PolyPltEvaluator::new();
    let dry_eval_start = Instant::now();
    let dry_one = GpuDCRTPoly::const_one(&params);
    let dry_inputs = plaintext_inputs.clone();
    let dry_out = dry_circuit.eval(&params, dry_one, dry_inputs, Some(&dry_plt_evaluator));
    info!("dry eval elapsed_ms={:.3}", dry_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(dry_out.len(), 1, "dry-run should output one value polynomial");
    let dry_const_term = dry_out[0]
        .coeffs()
        .into_iter()
        .next()
        .expect("dry-run output polynomial must have at least one coefficient")
        .value()
        .clone();
    assert_value_matches_q_level(
        &dry_const_term,
        &expected,
        &active_q,
        &active_q_moduli,
        &all_q_moduli,
    );
    info!("dry-run succeeded with expected constant term");
    drop(dry_out);
    drop(dry_circuit);
    drop(dry_plt_evaluator);

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
        BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    drop(secrets);
    let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintext_inputs);
    drop(plaintext_inputs);
    drop(a_inputs);
    drop(b_inputs);
    info!("encoding sampling elapsed_ms={:.3}", enc_setup_start.elapsed().as_secs_f64() * 1000.0);
    let encodings_compact_start = Instant::now();
    let encodings_compact: Vec<_> = encodings
        .into_iter()
        .map(|encoding| {
            (
                encoding.vector.into_compact_bytes(),
                encoding.pubkey.matrix.into_compact_bytes(),
                encoding.pubkey.reveal_plaintext,
                encoding.plaintext,
            )
        })
        .collect();
    info!(
        "encoding compact serialization elapsed_ms={:.3}",
        encodings_compact_start.elapsed().as_secs_f64() * 1000.0
    );

    let pk_evaluator_setup_start = Instant::now();
    let pk_evaluator =
        GGH15BGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(seed, d_secret, trapdoor_sigma, cfg.error_sigma, dir.to_path_buf(), false);
    info!(
        "pk evaluator setup elapsed_ms={:.3}",
        pk_evaluator_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let input_pubkeys = pubkeys.split_off(1);
    let one_pubkey = pubkeys.pop().expect("pubkeys must contain one entry for const one");
    let pubkey_eval_start = Instant::now();
    let pubkey_out = circuit.eval(&params, one_pubkey, input_pubkeys, Some(&pk_evaluator));
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
    let c_b0 = s_vec.clone() * &b0_matrix;
    drop(b0_matrix);
    let checkpoint_prefix = pk_evaluator.checkpoint_prefix(&params);
    drop(pk_evaluator);

    let enc_evaluator = GGH15BGGEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::new(seed, dir.to_path_buf(), checkpoint_prefix, &params, c_b0);

    let encodings_restore_start = Instant::now();
    let mut encodings: Vec<_> = encodings_compact
        .into_iter()
        .map(|(vector_bytes, pubkey_bytes, reveal_plaintext, plaintext)| {
            mxx::bgg::encoding::BggEncoding::new(
                GpuDCRTPolyMatrix::from_compact_bytes(&params, &vector_bytes),
                mxx::bgg::public_key::BggPublicKey::new(
                    GpuDCRTPolyMatrix::from_compact_bytes(&params, &pubkey_bytes),
                    reveal_plaintext,
                ),
                plaintext,
            )
        })
        .collect();
    info!(
        "encoding restore elapsed_ms={:.3}",
        encodings_restore_start.elapsed().as_secs_f64() * 1000.0
    );

    let input_encodings = encodings.split_off(1);
    let one_encoding = encodings.pop().expect("encodings must contain one entry for const one");
    let encoding_eval_start = Instant::now();
    let encoding_out = circuit.eval(&params, one_encoding, input_encodings, Some(&enc_evaluator));
    info!("encoding eval elapsed_ms={:.3}", encoding_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(encoding_out.len(), 1);

    assert_eq!(encoding_out[0].pubkey, pubkey_out[0]);
    let expected_poly = GpuDCRTPoly::from_biguint_to_constant(&params, expected.clone());
    let expected_times_gadget = s_vec.clone() *
        (GpuDCRTPolyMatrix::gadget_matrix(&params, cfg.d_secret) * expected_poly.clone());
    let s_times_pk = s_vec.clone() * &pubkey_out[0].matrix;
    let diff = encoding_out[0].vector.clone() - s_times_pk + expected_times_gadget;
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

    gpu_device_sync();
}
