use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    element::PolyElem,
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly},
    lookup::{
        lwe_eval::{LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator},
        poly::PolyPltEvaluator,
    },
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler, hash::DCRTPolyHashSampler,
        trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
    },
    simulator::{SimulatorContext, error_norm::NormPltLWEEvaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::{bigdecimal_bits_ceil, gen_biguint_for_modulus},
};
use num_bigint::BigUint;
use std::{fs, path::Path, sync::Arc};
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

fn reconstruct_from_residues(params: &DCRTPolyParams, residues: &[BigUint]) -> BigUint {
    let reconst_coeffs = params.reconst_coeffs();
    assert_eq!(residues.len(), reconst_coeffs.len());
    let q = params.modulus();
    let mut value = BigUint::from(0u64);
    for (residue, coeff) in residues.iter().zip(reconst_coeffs.iter()) {
        value += residue * coeff;
    }
    value % q.as_ref()
}

fn build_modq_arith_circuit(
    params: &DCRTPolyParams,
) -> (PolyCircuit<DCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        P_MODULI_BITS,
        SCALE,
        false,
        None,
    ));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);

    let (q_moduli, _, _) = params.to_crt();
    let q = params.modulus();
    let mut unit_vector = vec![BigUint::from(0u64); params.ring_dimension() as usize];
    unit_vector[0] = BigUint::from(1u64);

    let mut outputs = Vec::with_capacity(q_moduli.len());
    for &q_i in q_moduli.iter() {
        let q_over_qi = q.as_ref() / BigUint::from(q_i);
        let scalar = unit_vector.iter().map(|u| u * &q_over_qi).collect::<Vec<_>>();
        outputs.push(circuit.large_scalar_mul(out, &scalar));
    }
    circuit.output(outputs);
    (circuit, ctx)
}

fn build_modq_arith_value_circuit(params: &DCRTPolyParams) -> PolyCircuit<DCRTPoly> {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        P_MODULI_BITS,
        SCALE,
        false,
        None,
    ));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);
    circuit.output(vec![out]);
    circuit
}

fn find_crt_depth_for_modq_arith() -> (usize, DCRTPolyParams, PolyCircuit<DCRTPoly>) {
    let ring_dim_sqrt = BigDecimal::from_u32(RING_DIM).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << BASE_BITS, 0);
    let error_sigma = BigDecimal::from_f64(ERROR_SIGMA).expect("valid error sigma");
    let input_bound = BigDecimal::from((1u64 << P_MODULI_BITS) - 1);
    let e_init_norm = &error_sigma * BigDecimal::from(6u64);

    for crt_depth in 1..=MAX_CRT_DEPTH {
        let params = DCRTPolyParams::new(RING_DIM, crt_depth, CRT_BITS, BASE_BITS);
        let (q_moduli, _, _) = params.to_crt();
        let q = params.modulus();
        let (circuit, _ctx) = build_modq_arith_circuit(&params);

        let log_base_q = params.modulus_digits();
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt.clone(),
            base.clone(),
            D_SECRET,
            log_base_q,
        ));
        let plt_evaluator = NormPltLWEEvaluator::new(ctx.clone(), &error_sigma);

        let out_errors = circuit.simulate_max_error_norm(
            ctx,
            input_bound.clone(),
            circuit.num_input(),
            &e_init_norm,
            Some(&plt_evaluator),
        );

        assert_eq!(out_errors.len(), q_moduli.len());

        let mut all_ok = true;
        let mut max_error_bits = 0u64;
        for (idx, &q_i) in q_moduli.iter().enumerate() {
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
            return (crt_depth, params, circuit);
        }
    }

    panic!(
        "crt_depth satisfying error < q/(2*q_i) for all CRT moduli not found up to MAX_CRT_DEPTH"
    );
}

#[tokio::test]
async fn test_lwe_modq_arith() {
    let _ = tracing_subscriber::fmt::try_init();

    let (crt_depth, params, circuit) = find_crt_depth_for_modq_arith();
    info!(
        "selected crt_depth={} ring_dim={} crt_bits={} base_bits={}",
        crt_depth,
        params.ring_dimension(),
        CRT_BITS,
        BASE_BITS
    );
    info!(
        "circuit non_free_depth={} gate_counts={:?}",
        circuit.non_free_depth(),
        circuit.count_gates_by_type_vec()
    );

    let q = params.modulus();
    let (q_moduli, _, _) = params.to_crt();

    let mut rng = rand::rng();
    let a_value: BigUint = gen_biguint_for_modulus(&mut rng, q.as_ref());
    let b_value: BigUint = gen_biguint_for_modulus(&mut rng, q.as_ref());
    let expected = (&a_value * &b_value) % q.as_ref();

    let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value, None);
    let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value, None);
    let plaintext_inputs = [a_inputs.clone(), b_inputs.clone()].concat();

    let dry_circuit = build_modq_arith_value_circuit(&params);
    let dry_plt_evaluator = PolyPltEvaluator::new();
    let dry_out = dry_circuit.eval(
        &params,
        &DCRTPoly::const_one(&params),
        &plaintext_inputs,
        Some(&dry_plt_evaluator),
    );
    assert_eq!(dry_out.len(), 1, "plain PolyCircuit dry-run should output one value polynomial");
    let dry_const_term = dry_out[0]
        .coeffs()
        .into_iter()
        .next()
        .expect("dry-run output polynomial must have at least one coefficient")
        .value()
        .clone();
    assert_eq!(
        dry_const_term, expected,
        "dry-run output polynomial constant term must match expected mod q output"
    );
    info!("plain PolyCircuit dry-run succeeded with expected constant term");

    let plt_evaluator = PolyPltEvaluator::new();
    let plain_out = circuit.eval(
        &params,
        &DCRTPoly::const_one(&params),
        &plaintext_inputs,
        Some(&plt_evaluator),
    );
    assert_eq!(plain_out.len(), q_moduli.len());

    let mut plain_residues = Vec::with_capacity(q_moduli.len());
    for (idx, &q_i) in q_moduli.iter().enumerate() {
        let coeff = plain_out[idx]
            .coeffs()
            .into_iter()
            .next()
            .expect("output poly must have at least one coefficient")
            .value()
            .clone();
        let decoded = decode_residue_from_scaled_coeff(&coeff, q_i, q.as_ref());
        plain_residues.push(decoded);
    }
    let plain_reconstructed = reconstruct_from_residues(&params, &plain_residues);
    assert_eq!(plain_reconstructed, expected);

    let seed: [u8; 32] = [0u8; 32];
    let trapdoor_sigma = 4.578;
    let d_secret = D_SECRET;

    let dir = Path::new("test_data/test_lwe_modq_arith");
    if !dir.exists() {
        fs::create_dir_all(dir).unwrap();
    }
    init_storage_system(dir.to_path_buf());

    let uniform_sampler = DCRTPolyUniformSampler::new();
    let sampled =
        uniform_sampler.sample_uniform(&params, 1, d_secret - 1, DistType::TernaryDist).get_row(0);
    let mut secrets = sampled;
    secrets.push(DCRTPoly::const_minus_one(&params));
    let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets.to_vec());

    let pk_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(seed, d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    let pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);

    let encoding_sampler =
        BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
    let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintext_inputs);
    let enc_one = encodings[0].clone();

    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, trapdoor_sigma);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(&params, d_secret);
    let trapdoor = Arc::new(trapdoor);
    let pub_matrix = Arc::new(pub_matrix);

    let pk_evaluator =
        LWEBGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(
            seed, trapdoor_sampler.clone(), pub_matrix.clone(), trapdoor, dir.to_path_buf()
        );

    let pubkey_out = circuit.eval(&params, &enc_one.pubkey, &pubkeys[1..], Some(&pk_evaluator));
    assert_eq!(pubkey_out.len(), q_moduli.len());

    pk_evaluator.sample_aux_matrices(&params);
    wait_for_all_writes(dir.to_path_buf()).await.unwrap();

    let p = s_vec.clone() * pub_matrix.as_ref();
    let enc_evaluator =
        LWEBGGEncodingPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::new(
            seed,
            dir.to_path_buf(),
            p,
        );

    let encoding_out = circuit.eval(&params, &enc_one, &encodings[1..], Some(&enc_evaluator));
    assert_eq!(encoding_out.len(), q_moduli.len());

    let unit_column = DCRTPolyMatrix::unit_column_vector(&params, d_secret, d_secret - 1);
    let mut decoded_residues = Vec::with_capacity(q_moduli.len());
    for (idx, &q_i) in q_moduli.iter().enumerate() {
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

        let decoded = decode_residue_from_scaled_coeff(&coeff, q_i, q.as_ref());
        let expected_residue = &expected % BigUint::from(q_i);
        assert_eq!(decoded, expected_residue);
        decoded_residues.push(decoded);
    }

    let reconstructed = reconstruct_from_residues(&params, &decoded_residues);
    assert_eq!(reconstructed, expected);
}
