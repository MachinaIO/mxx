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
    utils::{bigdecimal_bits_ceil, gen_biguint_for_modulus, mod_inverse},
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::{fs, path::Path, sync::Arc, time::Instant};
use tracing::info;

const RING_DIM: u32 = 1 << 8;
const CRT_BITS: usize = 24;
const P_MODULI_BITS: usize = 6;
const SCALE: u64 = 1 << 7;
const BASE_BITS: u32 = 8;
const MAX_CRT_DEPTH: usize = 32;
const ERROR_SIGMA: f64 = 4.0;
const D_SECRET: usize = 2;

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
}

fn decode_residue_from_scaled_coeff(coeff: &BigUint, q_i: u64, q: &BigUint) -> BigUint {
    let scaled = coeff * BigUint::from(q_i);
    let rounded = round_div_biguint(&scaled, q);
    rounded % BigUint::from(q_i)
}

fn reconstruct_from_residues(q_moduli: &[u64], q: &BigUint, residues: &[BigUint]) -> BigUint {
    assert_eq!(residues.len(), q_moduli.len());
    let mut value = BigUint::from(0u64);
    for (residue, &q_i) in residues.iter().zip(q_moduli.iter()) {
        let q_i_big = BigUint::from(q_i);
        let q_over_qi = q / &q_i_big;
        let q_over_qi_mod_qi =
            (&q_over_qi % &q_i_big).to_u64().expect("CRT residue must fit in u64");
        let inv = mod_inverse(q_over_qi_mod_qi, q_i).expect("CRT moduli must be pairwise coprime");
        let coeff = (&q_over_qi * BigUint::from(inv)) % q;
        value += residue * coeff;
    }
    value % q
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
) -> (PolyCircuit<DCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        P_MODULI_BITS,
        SCALE,
        false,
        q_level,
    ));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);

    let (active_q_moduli, _, _) = active_q_moduli_and_modulus(params, q_level);
    let q = params.modulus();
    let mut unit_vector = vec![BigUint::from(0u64); params.ring_dimension() as usize];
    unit_vector[0] = BigUint::from(1u64);

    let mut outputs = Vec::with_capacity(active_q_moduli.len());
    for &q_i in active_q_moduli.iter() {
        let q_over_qi = q.as_ref() / BigUint::from(q_i);
        let scalar = unit_vector.iter().map(|u| u * &q_over_qi).collect::<Vec<_>>();
        outputs.push(circuit.large_scalar_mul(out, &scalar));
    }
    circuit.output(outputs);
    (circuit, ctx)
}

fn build_modq_arith_circuit_gpu(
    params: &GpuDCRTPolyParams,
    q_level: Option<usize>,
) -> (PolyCircuit<GpuDCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        P_MODULI_BITS,
        SCALE,
        false,
        q_level,
    ));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);

    let (active_q_moduli, _, _) = active_q_moduli_and_modulus(params, q_level);
    let q = params.modulus();
    let mut unit_vector = vec![BigUint::from(0u64); params.ring_dimension() as usize];
    unit_vector[0] = BigUint::from(1u64);

    let mut outputs = Vec::with_capacity(active_q_moduli.len());
    for &q_i in active_q_moduli.iter() {
        let q_over_qi = q.as_ref() / BigUint::from(q_i);
        let scalar = unit_vector.iter().map(|u| u * &q_over_qi).collect::<Vec<_>>();
        outputs.push(circuit.large_scalar_mul(out, &scalar));
    }
    circuit.output(outputs);
    (circuit, ctx)
}

fn build_modq_arith_value_circuit_gpu(
    params: &GpuDCRTPolyParams,
    q_level: Option<usize>,
) -> PolyCircuit<GpuDCRTPoly> {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        P_MODULI_BITS,
        SCALE,
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

fn find_crt_depth_for_modq_arith(q_level: Option<usize>) -> (usize, DCRTPolyParams) {
    let ring_dim_sqrt = BigDecimal::from_u32(RING_DIM).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << BASE_BITS, 0);
    let error_sigma = BigDecimal::from_f64(ERROR_SIGMA).expect("valid error sigma");
    let input_bound = BigDecimal::from((1u64 << P_MODULI_BITS) - 1);
    let trapdoor_sigma = BigDecimal::from_f64(4.578).expect("valid trapdoor sigma");
    let e_init_norm = &error_sigma * BigDecimal::from(6u64);

    for crt_depth in 1..=MAX_CRT_DEPTH {
        let params = DCRTPolyParams::new(RING_DIM, crt_depth, CRT_BITS, BASE_BITS);
        let (active_q_moduli, _, _) = active_q_moduli_and_modulus(&params, q_level);
        let q = params.modulus();
        let (_, _, crt_depth) = params.to_crt();
        let (circuit, _ctx) = build_modq_arith_circuit_cpu(&params, q_level);

        let log_base_q = params.modulus_digits();
        let log_base_q_small = log_base_q / crt_depth;
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt.clone(),
            base.clone(),
            D_SECRET,
            log_base_q,
            log_base_q_small,
        ));
        let plt_evaluator = NormPltGGH15Evaluator::new(
            ctx.clone(),
            &error_sigma,
            &error_sigma,
            &trapdoor_sigma,
            None,
        );

        let out_errors = circuit.simulate_max_error_norm(
            ctx,
            input_bound.clone(),
            circuit.num_input(),
            &e_init_norm,
            Some(&plt_evaluator),
        );

        assert_eq!(out_errors.len(), active_q_moduli.len());

        let mut all_ok = true;
        let mut max_error_bits = 0u64;
        for (idx, &q_i) in active_q_moduli.iter().enumerate() {
            let threshold = q.as_ref() / BigUint::from(2u64 * q_i);
            let error = &out_errors[idx].matrix_norm.poly_norm.norm;
            max_error_bits = max_error_bits.max(bigdecimal_bits_ceil(error));
            if *error >= BigDecimal::from_biguint(threshold, 0) {
                all_ok = false;
            }
        }

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
        "crt_depth satisfying error < q/(2*q_i) for all CRT moduli not found up to MAX_CRT_DEPTH"
    );
}

#[tokio::test]
async fn test_gpu_ggh15_modq_arith() {
    let _ = tracing_subscriber::fmt::try_init();
    gpu_device_sync();

    let q_level = q_level_from_env();
    let (crt_depth, cpu_params) = find_crt_depth_for_modq_arith(q_level);
    let (moduli, _, _) = cpu_params.to_crt();
    let params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli, cpu_params.base_bits());
    assert_eq!(params.modulus(), cpu_params.modulus());

    let (all_q_moduli, _, _) = params.to_crt();
    let full_q = params.modulus();
    let (active_q_moduli, q, active_q_level) = active_q_moduli_and_modulus(&params, q_level);
    let (circuit, _ctx) = build_modq_arith_circuit_gpu(&params, q_level);
    info!("found crt_depth={}", crt_depth);
    info!(
        "selected crt_depth={} ring_dim={} crt_bits={} base_bits={} q_level={:?}",
        crt_depth,
        params.ring_dimension(),
        CRT_BITS,
        BASE_BITS,
        q_level
    );
    info!(
        "circuit non_free_depth={} gate_counts={:?}",
        circuit.non_free_depth(),
        circuit.count_gates_by_type_vec()
    );
    info!(
        "active_q_level={} active_q_bits={} active_q_moduli_len={}",
        active_q_level,
        q.bits(),
        active_q_moduli.len()
    );

    let mut rng = rand::rng();
    let a_value: BigUint = gen_biguint_for_modulus(&mut rng, &q);
    let b_value: BigUint = gen_biguint_for_modulus(&mut rng, &q);
    let expected = (&a_value * &b_value) % &q;

    let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value, q_level);
    let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value, q_level);
    let plaintext_inputs = [a_inputs.clone(), b_inputs.clone()].concat();

    let dry_circuit = build_modq_arith_value_circuit_gpu(&params, q_level);
    let dry_plt_evaluator = PolyPltEvaluator::new();
    let dry_eval_start = Instant::now();
    let dry_out = dry_circuit.eval(
        &params,
        &GpuDCRTPoly::const_one(&params),
        &plaintext_inputs,
        Some(&dry_plt_evaluator),
    );
    info!("dry eval elapsed_ms={:.3}", dry_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(dry_out.len(), 1, "dry-run should output one value polynomial");
    let dry_const_term = dry_out[0]
        .coeffs()
        .into_iter()
        .next()
        .expect("dry-run output polynomial must have at least one coefficient")
        .value()
        .clone();
    assert_value_matches_q_level(&dry_const_term, &expected, &q, &active_q_moduli, &all_q_moduli);
    info!("dry-run succeeded with expected constant term");

    let plt_evaluator = PolyPltEvaluator::new();
    let plain_out = circuit.eval(
        &params,
        &GpuDCRTPoly::const_one(&params),
        &plaintext_inputs,
        Some(&plt_evaluator),
    );
    assert_eq!(plain_out.len(), active_q_moduli.len());

    let mut plain_residues = Vec::with_capacity(active_q_moduli.len());
    for (idx, &q_i) in active_q_moduli.iter().enumerate() {
        let coeff = plain_out[idx]
            .coeffs()
            .into_iter()
            .next()
            .expect("output poly must have at least one coefficient")
            .value()
            .clone();
        let decoded = decode_residue_from_scaled_coeff(&coeff, q_i, full_q.as_ref());
        plain_residues.push(decoded);
    }
    let plain_reconstructed =
        reconstruct_from_residues(&active_q_moduli, full_q.as_ref(), &plain_residues);
    assert_value_matches_q_level(
        &plain_reconstructed,
        &expected,
        &q,
        &active_q_moduli,
        &all_q_moduli,
    );

    let seed: [u8; 32] = [0u8; 32];
    let trapdoor_sigma = 4.578;
    let d_secret = D_SECRET;

    let dir_name = format!("test_data/test_gpu_ggh15_modq_arith_qlevel_{}", active_q_level);
    let dir = Path::new(&dir_name);
    if !dir.exists() {
        fs::create_dir_all(dir).expect("failed to create test directory");
    }
    init_storage_system(dir.to_path_buf());

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let sampled =
        uniform_sampler.sample_uniform(&params, 1, d_secret - 1, DistType::TernaryDist).get_row(0);
    let mut secrets = sampled;
    secrets.push(GpuDCRTPoly::const_minus_one(&params));
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.to_vec());

    let pk_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(seed, d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    let pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);

    let enc_setup_start = Instant::now();
    let encoding_sampler =
        BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintext_inputs);
    let enc_one = encodings[0].clone();

    let pk_evaluator =
        GGH15BGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(seed, d_secret, trapdoor_sigma, ERROR_SIGMA, dir.to_path_buf(), false);
    info!(
        "encoding sampler + pk evaluator setup elapsed_ms={:.3}",
        enc_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let pubkey_eval_start = Instant::now();
    let pubkey_out = circuit.eval(&params, &enc_one.pubkey, &pubkeys[1..], Some(&pk_evaluator));
    info!("pubkey eval elapsed_ms={:.3}", pubkey_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(pubkey_out.len(), active_q_moduli.len());

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
    let checkpoint_prefix = pk_evaluator.checkpoint_prefix(&params);

    let enc_evaluator = GGH15BGGEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::new(seed, dir.to_path_buf(), checkpoint_prefix, c_b0);

    let encoding_eval_start = Instant::now();
    let encoding_out = circuit.eval(&params, &enc_one, &encodings[1..], Some(&enc_evaluator));
    info!("encoding eval elapsed_ms={:.3}", encoding_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(encoding_out.len(), active_q_moduli.len());

    let unit_column = GpuDCRTPolyMatrix::unit_column_vector(&params, d_secret, d_secret - 1);
    let mut decoded_residues = Vec::with_capacity(active_q_moduli.len());
    for (idx, &q_i) in active_q_moduli.iter().enumerate() {
        assert_eq!(encoding_out[idx].pubkey, pubkey_out[idx]);

        let s_times_pk = s_vec.clone() * &pubkey_out[idx].matrix;
        let diff = encoding_out[idx].vector.clone() - s_times_pk;
        let projected = diff.mul_decompose(&unit_column);
        assert_eq!(projected.row_size(), 1);
        assert_eq!(projected.col_size(), 1);

        let coeff = projected
            .entry(0, 0)
            .coeffs()
            .into_iter()
            .next()
            .expect("projected poly must have at least one coefficient")
            .value()
            .clone();

        let decoded = decode_residue_from_scaled_coeff(&coeff, q_i, full_q.as_ref());
        let expected_residue = &expected % BigUint::from(q_i);
        assert_eq!(decoded, expected_residue);
        decoded_residues.push(decoded);
    }

    let reconstructed =
        reconstruct_from_residues(&active_q_moduli, full_q.as_ref(), &decoded_residues);
    assert_value_matches_q_level(&reconstructed, &expected, &q, &active_q_moduli, &all_q_moduli);

    gpu_device_sync();
}
