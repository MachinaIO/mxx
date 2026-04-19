#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    element::PolyElem,
    gadgets::arith::{MontgomeryPoly, MontgomeryPolyContext, encode_montgomery_poly},
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
use rayon::prelude::*;
use std::{env, fs, path::Path, sync::Arc, time::Instant};
use tracing::info;

const DEFAULT_RING_DIM: u32 = 1 << 8;
const DEFAULT_CRT_BITS: usize = 24;
const DEFAULT_BASE_BITS: u32 = 12;
const DEFAULT_MAX_CRT_DEPTH: usize = 64;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_HEIGHT: usize = 1;
const DEFAULT_LIMB_BIT_SIZE: usize = 6;

#[derive(Debug, Clone)]
struct MontgomeryModqArithConfig {
    ring_dim: u32,
    crt_bits: usize,
    base_bits: u32,
    max_crt_depth: usize,
    error_sigma: f64,
    d_secret: usize,
    height: usize,
    limb_bit_size: usize,
    dir_name_override: Option<String>,
}

fn env_or_parse_u32(key: &str, default: u32) -> u32 {
    match env::var(key) {
        Ok(raw) => raw.parse::<u32>().unwrap_or_else(|e| panic!("{key} must be a valid u32: {e}")),
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

impl MontgomeryModqArithConfig {
    fn from_env() -> Self {
        let ring_dim = env_or_parse_u32("GGH15_MONTGOMERY_MODQ_ARITH_RING_DIM", DEFAULT_RING_DIM);
        let crt_bits = env_or_parse_usize("GGH15_MONTGOMERY_MODQ_ARITH_CRT_BITS", DEFAULT_CRT_BITS);
        let base_bits =
            env_or_parse_u32("GGH15_MONTGOMERY_MODQ_ARITH_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth =
            env_or_parse_usize("GGH15_MONTGOMERY_MODQ_ARITH_MAX_CRT_DEPTH", DEFAULT_MAX_CRT_DEPTH);
        let error_sigma =
            env_or_parse_f64("GGH15_MONTGOMERY_MODQ_ARITH_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret = env_or_parse_usize("GGH15_MONTGOMERY_MODQ_ARITH_D_SECRET", DEFAULT_D_SECRET);
        let height = env_or_parse_usize("GGH15_MONTGOMERY_MODQ_ARITH_HEIGHT", DEFAULT_HEIGHT);
        let limb_bit_size =
            env_or_parse_usize("GGH15_MONTGOMERY_MODQ_ARITH_LIMB_BIT_SIZE", DEFAULT_LIMB_BIT_SIZE);
        let dir_name_override = env::var("GGH15_MONTGOMERY_MODQ_ARITH_DIR_NAME")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        assert!(ring_dim > 0, "GGH15_MONTGOMERY_MODQ_ARITH_RING_DIM must be > 0");
        assert!(crt_bits > 0, "GGH15_MONTGOMERY_MODQ_ARITH_CRT_BITS must be > 0");
        assert!(base_bits > 0, "GGH15_MONTGOMERY_MODQ_ARITH_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "GGH15_MONTGOMERY_MODQ_ARITH_MAX_CRT_DEPTH must be > 0");
        assert!(error_sigma > 0.0, "GGH15_MONTGOMERY_MODQ_ARITH_ERROR_SIGMA must be > 0");
        assert!(d_secret > 0, "GGH15_MONTGOMERY_MODQ_ARITH_D_SECRET must be > 0");
        assert!(height > 0, "GGH15_MONTGOMERY_MODQ_ARITH_HEIGHT must be > 0");
        assert!(
            limb_bit_size > 0 && limb_bit_size < 32,
            "GGH15_MONTGOMERY_MODQ_ARITH_LIMB_BIT_SIZE must be in 1..32"
        );

        Self {
            ring_dim,
            crt_bits,
            base_bits,
            max_crt_depth,
            error_sigma,
            d_secret,
            height,
            limb_bit_size,
            dir_name_override,
        }
    }

    fn dir_name(&self) -> String {
        self.dir_name_override
            .clone()
            .unwrap_or_else(|| "test_data/test_gpu_ggh15_montgomery_modq_arith".to_string())
    }
}

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
}

fn build_montgomery_multiplication_tree<P: Poly + 'static>(
    ctx: Arc<MontgomeryPolyContext<P>>,
    circuit: &mut PolyCircuit<P>,
    height: usize,
) {
    let num_inputs =
        1usize.checked_shl(height as u32).expect("height is too large to represent 2^h inputs");
    let mut current_layer: Vec<MontgomeryPoly<P>> =
        (0..num_inputs).map(|_| MontgomeryPoly::input(ctx.clone(), circuit)).collect();

    while current_layer.len() > 1 {
        debug_assert!(current_layer.len().is_multiple_of(2), "layer size must stay even");
        let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
        for pair in current_layer.chunks(2) {
            let parent = pair[0].mul(&pair[1], circuit);
            next_layer.push(parent);
        }
        current_layer = next_layer;
    }

    let root = current_layer.pop().expect("multiplication tree must contain at least one node");
    let out = root.finalize(circuit);
    circuit.output(vec![out]);
}

fn build_modq_arith_circuit_cpu(
    params: &DCRTPolyParams,
    cfg: &MontgomeryModqArithConfig,
) -> (PolyCircuit<DCRTPoly>, Arc<MontgomeryPolyContext<DCRTPoly>>) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx =
        Arc::new(MontgomeryPolyContext::setup(&mut circuit, params, cfg.limb_bit_size, false));
    build_montgomery_multiplication_tree(ctx.clone(), &mut circuit, cfg.height);
    (circuit, ctx)
}

fn build_modq_arith_circuit_gpu(
    params: &GpuDCRTPolyParams,
    cfg: &MontgomeryModqArithConfig,
) -> (PolyCircuit<GpuDCRTPoly>, Arc<MontgomeryPolyContext<GpuDCRTPoly>>) {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx =
        Arc::new(MontgomeryPolyContext::setup(&mut circuit, params, cfg.limb_bit_size, false));
    build_montgomery_multiplication_tree(ctx.clone(), &mut circuit, cfg.height);
    (circuit, ctx)
}

fn build_modq_arith_value_circuit_gpu(
    params: &GpuDCRTPolyParams,
    cfg: &MontgomeryModqArithConfig,
) -> PolyCircuit<GpuDCRTPoly> {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx =
        Arc::new(MontgomeryPolyContext::setup(&mut circuit, params, cfg.limb_bit_size, false));
    build_montgomery_multiplication_tree(ctx, &mut circuit, cfg.height);
    circuit
}

fn find_crt_depth_for_modq_arith(cfg: &MontgomeryModqArithConfig) -> (usize, DCRTPolyParams) {
    let ring_dim_sqrt = BigDecimal::from_u32(cfg.ring_dim).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << cfg.base_bits, 0);
    let error_sigma = BigDecimal::from_f64(cfg.error_sigma).expect("valid error sigma");
    let input_bound = BigDecimal::from((1u64 << cfg.limb_bit_size) - 1);
    let e_init_norm = &error_sigma * BigDecimal::from_f32(6.5).unwrap();

    for crt_depth in 1..=cfg.max_crt_depth {
        let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
        let full_q = params.modulus();
        let (q_moduli, _, crt_depth) = params.to_crt();
        let q_max = *q_moduli.iter().max().expect("q_moduli must not be empty");
        let (circuit, _ctx) = build_modq_arith_circuit_cpu(&params, cfg);

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
        let all_ok = *error < BigDecimal::from_biguint(threshold.clone(), 0);

        info!(
            "crt_depth={} q_bits={} max_error_bits={} threshold_bits={}",
            crt_depth,
            params.modulus_bits(),
            max_error_bits,
            threshold.bits()
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
async fn test_gpu_ggh15_montgomery_modq_arith() {
    let _ = tracing_subscriber::fmt::try_init();
    gpu_device_sync();
    let cfg = MontgomeryModqArithConfig::from_env();
    info!("montgomery modq arith test config: {:?}", cfg);

    let num_inputs = 1usize
        .checked_shl(cfg.height as u32)
        .expect("GGH15_MONTGOMERY_MODQ_ARITH_HEIGHT is too large");
    let depth_search_start = Instant::now();
    let (crt_depth, cpu_params) = find_crt_depth_for_modq_arith(&cfg);
    info!("crt depth search elapsed_ms={:.3}", depth_search_start.elapsed().as_secs_f64() * 1000.0);

    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli.clone(), cpu_params.base_bits());
    let single_gpu_id = *detected_gpu_params
        .gpu_ids()
        .first()
        .expect("at least one GPU device is required for test_gpu_ggh15_montgomery_modq_arith");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
    );
    let full_q = params.modulus();
    let (all_q_moduli, _, _) = params.to_crt();
    let q_max = *all_q_moduli.iter().max().expect("q_moduli must not be empty");
    let (circuit, ctx) = build_modq_arith_circuit_gpu(&params, &cfg);
    let gate_counts = circuit.count_gates_by_type_vec();
    let total_gates = circuit.num_gates();

    info!("forcing single GPU for this test: gpu_id={}", single_gpu_id);
    info!("found crt_depth={}", crt_depth);
    info!(
        "selected crt_depth={} ring_dim={} crt_bits={} base_bits={} limb_bit_size={} q_moduli={:?}",
        crt_depth,
        params.ring_dimension(),
        cfg.crt_bits,
        cfg.base_bits,
        cfg.limb_bit_size,
        all_q_moduli
    );
    info!("multiplication tree config: height={} num_inputs={}", cfg.height, num_inputs);
    let non_free_depth_contributions = circuit.non_free_depth_contributions();
    info!(
        "circuit total_gates={} non_free_depth_contributions={:?} gate_counts={:?}",
        total_gates, non_free_depth_contributions, gate_counts
    );

    assert_eq!(params.modulus(), cpu_params.modulus());
    assert_eq!(circuit.num_output(), 1, "multiplication tree should emit one output gate");
    assert_eq!(
        circuit.num_input(),
        num_inputs * ctx.num_limbs,
        "Montgomery inputs should contribute one limb polynomial per input limb"
    );
    assert!(total_gates > 0, "circuit must contain gates");

    let mut rng = rand::rng();
    let input_values: Vec<BigUint> =
        (0..num_inputs).map(|_| gen_biguint_for_modulus(&mut rng, &full_q)).collect();
    let expected =
        input_values.iter().fold(BigUint::from(1u64), |acc, value| (acc * value) % full_q.as_ref());
    let plaintext_inputs: Vec<GpuDCRTPoly> = input_values
        .par_iter()
        .flat_map(|value| encode_montgomery_poly(cfg.limb_bit_size, &params, value))
        .collect();

    let dry_circuit = build_modq_arith_value_circuit_gpu(&params, &cfg);
    let dry_plt_evaluator = PolyPltEvaluator::new();
    let dry_eval_start = Instant::now();
    let dry_one = GpuDCRTPoly::const_one(&params);
    let dry_inputs = plaintext_inputs.clone();
    info!("starting dry-run evaluation to verify correctness before GPU evaluation");
    let dry_out =
        dry_circuit.eval(&params, dry_one, dry_inputs, Some(&dry_plt_evaluator), None, None);
    info!("dry eval elapsed_ms={:.3}", dry_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(dry_out.len(), 1, "dry-run should output one value polynomial");
    let dry_const_term = dry_out[0]
        .coeffs()
        .into_iter()
        .next()
        .expect("dry-run output polynomial must have at least one coefficient")
        .value()
        .clone();
    assert_eq!(dry_const_term, expected, "dry-run output must match the expected product mod q");
    info!("dry-run succeeded with expected constant term");
    drop(dry_out);
    drop(dry_circuit);
    drop(dry_plt_evaluator);

    let seed: [u8; 32] = [0u8; 32];
    let trapdoor_sigma = 4.578;

    let dir_name = cfg.dir_name();
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
    let encoding_sampler = BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(
        &params,
        &secrets,
        Some(cfg.error_sigma),
    );
    drop(secrets);
    let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintext_inputs);
    drop(plaintext_inputs);
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
        >::new(seed, cfg.d_secret, trapdoor_sigma, cfg.error_sigma, dir.to_path_buf());
    info!(
        "pk evaluator setup elapsed_ms={:.3}",
        pk_evaluator_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let input_pubkeys = pubkeys.split_off(1);
    let one_pubkey = pubkeys.pop().expect("pubkeys must contain one entry for const one");
    let pubkey_eval_start = Instant::now();
    let pubkey_out =
        circuit.eval(&params, one_pubkey, input_pubkeys, Some(&pk_evaluator), None, None);
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
    let c_b0_base = s_vec.clone() * &b0_matrix;
    let c_b0_error = uniform_sampler.sample_uniform(
        &params,
        1,
        c_b0_base.col_size(),
        DistType::GaussDist { sigma: cfg.error_sigma },
    );
    let c_b0 = c_b0_base + c_b0_error;
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
    let encoding_out =
        circuit.eval(&params, one_encoding, input_encodings, Some(&enc_evaluator), None, None);
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
