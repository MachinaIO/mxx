#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    element::PolyElem,
    lookup::{
        PublicLut,
        ggh15_eval::{GGH15BGGEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator},
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
    utils::bigdecimal_bits_ceil,
};
use num_bigint::BigUint;
use rand::Rng;
use std::{fs, path::Path, sync::Arc, time::Instant};
use tracing::info;

const CRT_BITS: usize = 51;
const RING_DIM: u32 = 1 << 8;
const ERROR_SIGMA: f64 = 4.0;
const BASE_BITS: u32 = 17;
const MAX_CRT_DEPTH: usize = 12;
const P: u64 = 7;
const D_SECRET: usize = 2;

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
}

fn decode_mod_p_from_scaled_plaintext(plaintext: &GpuDCRTPoly, q_over_p: &BigUint, p: u64) -> u64 {
    let coeff = plaintext
        .coeffs()
        .into_iter()
        .next()
        .expect("plaintext polynomial must have at least one coefficient");
    let rounded = round_div_biguint(coeff.value(), q_over_p);
    (&rounded % BigUint::from(p)).try_into().expect("decoded value must fit u64")
}

fn max_output_row_from_biguint_cpu(
    params: &DCRTPolyParams,
    idx: usize,
    value: BigUint,
) -> (usize, <DCRTPoly as Poly>::Elem) {
    let poly = DCRTPoly::from_biguint_to_constant(params, value);
    let coeff = poly
        .coeffs()
        .into_iter()
        .max()
        .expect("max_output_row_from_biguint_cpu requires coefficients");
    (idx, coeff)
}

fn build_mod_p_lut_cpu(params: &DCRTPolyParams, p: u64) -> PublicLut<DCRTPoly> {
    let lut_len = (p * p) as usize;
    let max_row = max_output_row_from_biguint_cpu(params, (p - 1) as usize, BigUint::from(p - 1));
    PublicLut::<DCRTPoly>::new_from_usize_range(
        params,
        lut_len,
        move |params, t| {
            let output = BigUint::from((t as u64) % p);
            (t, DCRTPoly::from_biguint_to_constant(params, output))
        },
        Some(max_row),
    )
}

fn build_modp_chain_circuit_cpu(
    params: &DCRTPolyParams,
    p: u64,
    q_over_p: &BigUint,
) -> PolyCircuit<DCRTPoly> {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(3);

    let lut_id = circuit.register_public_lookup(build_mod_p_lut_cpu(params, p));

    let t1 = circuit.mul_gate(inputs[0], inputs[1]);
    let t1_mod = circuit.public_lookup_gate(t1, lut_id);

    let t2 = circuit.mul_gate(t1_mod, inputs[2]);
    let t2_mod = circuit.public_lookup_gate(t2, lut_id);

    let scalar = DCRTPoly::from_biguint_to_constant(params, q_over_p.clone());
    let scaled = circuit.poly_scalar_mul(t2_mod, &scalar);
    circuit.output(vec![scaled]);
    circuit
}

fn find_crt_depth_for_modp_chain() -> (usize, DCRTPolyParams, BigUint) {
    let ring_dim_sqrt = BigDecimal::from_u32(RING_DIM)
        .expect("ring dim should convert to BigDecimal")
        .sqrt()
        .expect("ring dim sqrt should exist");
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << BASE_BITS, 0);
    let error_sigma = BigDecimal::from_f64(ERROR_SIGMA).expect("valid error sigma");

    for crt_depth in 1..=MAX_CRT_DEPTH {
        let params = DCRTPolyParams::new(RING_DIM, crt_depth, CRT_BITS, BASE_BITS);
        let (q_moduli, _, crt_depth) = params.to_crt();
        let q_moduli_min = *q_moduli.iter().min().expect("q_moduli must not be empty");
        assert!(
            (P as u128) * (P as u128) < q_moduli_min as u128,
            "p^2 must be smaller than any CRT modulus"
        );

        let q = params.modulus();
        let q_over_p = q.as_ref() / BigUint::from(P);
        let q_over_2p = q.as_ref() / BigUint::from(2 * P);
        let circuit = build_modp_chain_circuit_cpu(&params, P, &q_over_p);

        let log_base_q = params.modulus_digits();
        let log_base_q_small = log_base_q / crt_depth;
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt.clone(),
            base.clone(),
            D_SECRET,
            log_base_q,
            log_base_q_small,
        ));
        let plt_evaluator =
            NormPltGGH15Evaluator::new(ctx.clone(), &error_sigma, &error_sigma, None);
        let e_init_norm = &error_sigma * BigDecimal::from(6u64);
        let input_bound = BigDecimal::from((P - 1) as u64);

        let out_errors = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound,
            circuit.num_input(),
            &e_init_norm,
            Some(&plt_evaluator),
        );

        let max_error = out_errors
            .into_iter()
            .map(|e| e.matrix_norm.poly_norm.norm)
            .max_by(|a, b| a.partial_cmp(b).expect("comparable BigDecimal"))
            .expect("non-empty output");
        let q_over_2p_bd = BigDecimal::from_biguint(q_over_2p.clone(), 0);
        info!(
            "crt_depth={} q_over_p_bits={} q_over_2p_bits={} max_error_bits={}",
            crt_depth,
            q_over_p.bits(),
            q_over_2p.bits(),
            bigdecimal_bits_ceil(&max_error)
        );
        if max_error < q_over_2p_bd {
            return (crt_depth, params, q_over_p);
        }
    }

    panic!("crt_depth satisfying error < q/(2p) not found up to MAX_CRT_DEPTH");
}

fn max_output_row_from_biguint_gpu(
    params: &GpuDCRTPolyParams,
    idx: usize,
    value: BigUint,
) -> (usize, <GpuDCRTPoly as Poly>::Elem) {
    let poly = GpuDCRTPoly::from_biguint_to_constant(params, value);
    let coeff = poly
        .coeffs()
        .into_iter()
        .max()
        .expect("max_output_row_from_biguint_gpu requires coefficients");
    (idx, coeff)
}

fn build_mod_p_lut_gpu(params: &GpuDCRTPolyParams, p: u64) -> PublicLut<GpuDCRTPoly> {
    let lut_len = (p * p) as usize;
    let max_row = max_output_row_from_biguint_gpu(params, (p - 1) as usize, BigUint::from(p - 1));
    PublicLut::<GpuDCRTPoly>::new_from_usize_range(
        params,
        lut_len,
        move |params, t| {
            let output = BigUint::from((t as u64) % p);
            (t, GpuDCRTPoly::from_biguint_to_constant(params, output))
        },
        Some(max_row),
    )
}

fn build_modp_chain_circuit_gpu(
    params: &GpuDCRTPolyParams,
    p: u64,
    q_over_p: &BigUint,
) -> PolyCircuit<GpuDCRTPoly> {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let inputs = circuit.input(3);

    let lut_id = circuit.register_public_lookup(build_mod_p_lut_gpu(params, p));

    let t1 = circuit.mul_gate(inputs[0], inputs[1]);
    let t1_mod = circuit.public_lookup_gate(t1, lut_id);

    let t2 = circuit.mul_gate(t1_mod, inputs[2]);
    let t2_mod = circuit.public_lookup_gate(t2, lut_id);

    let scalar = GpuDCRTPoly::from_biguint_to_constant(params, q_over_p.clone());
    let scaled = circuit.poly_scalar_mul(t2_mod, &scalar);
    circuit.output(vec![scaled]);
    circuit
}

#[tokio::test]
async fn test_gpu_ggh15_modp_chain_rounding() {
    let _ = tracing_subscriber::fmt::try_init();
    gpu_device_sync();

    let (_crt_depth, cpu_params, q_over_p) = find_crt_depth_for_modp_chain();
    let (moduli, _, _) = cpu_params.to_crt();
    let params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli, cpu_params.base_bits());
    assert_eq!(params.modulus(), cpu_params.modulus());

    let circuit = build_modp_chain_circuit_gpu(&params, P, &q_over_p);
    info!(
        "selected crt_depth={} ring_dim={} base_bits={}",
        params.crt_depth(),
        params.ring_dimension(),
        params.base_bits()
    );

    let mut rng = rand::rng();
    let a: u64 = rng.random_range(0..P);
    let b: u64 = rng.random_range(0..P);
    let c: u64 = rng.random_range(0..P);
    let expected_mod_p = ((a * b) % P) * c % P;

    let plaintexts = vec![
        GpuDCRTPoly::from_usize_to_constant(&params, a as usize),
        GpuDCRTPoly::from_usize_to_constant(&params, b as usize),
        GpuDCRTPoly::from_usize_to_constant(&params, c as usize),
    ];

    let key: [u8; 32] = [
        0x5f, 0x92, 0x10, 0x6a, 0xa0, 0xd4, 0x55, 0xf2, 0x64, 0x96, 0x74, 0x8f, 0xc6, 0xdb, 0xed,
        0x20, 0x1e, 0x0c, 0xd6, 0x6a, 0xe4, 0xa4, 0x3a, 0x9a, 0x41, 0xb5, 0xf8, 0x30, 0xe7, 0x3a,
        0x08, 0xd9,
    ];

    let bgg_pubkey_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(key, D_SECRET);
    let tag_bytes: &[u8] = b"bgg_pubkey";

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let s =
        uniform_sampler.sample_uniform(&params, 1, D_SECRET - 1, DistType::TernaryDist).get_row(0);
    let mut secrets = s;
    secrets.push(GpuDCRTPoly::const_minus_one(&params));
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.to_vec());

    let reveal_plaintexts = vec![true; circuit.num_input()];
    let pubkeys = bgg_pubkey_sampler.sample(&params, tag_bytes, &reveal_plaintexts);
    let bgg_encoding_sampler =
        BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
    let enc_one = encodings[0].clone();
    let input_pubkeys = pubkeys[1..].to_vec();
    let input_encodings = encodings[1..].to_vec();
    let input_pubkeys_shared: Vec<Arc<_>> = input_pubkeys.iter().cloned().map(Arc::new).collect();
    let input_encodings_shared: Vec<Arc<_>> =
        input_encodings.iter().cloned().map(Arc::new).collect();

    let trapdoor_sigma = 4.578;
    let dir = Path::new("test_data/gpu_ggh15_modp_chain_rounding");
    if !dir.exists() {
        fs::create_dir_all(dir).expect("failed to create test directory");
    }
    init_storage_system(dir.to_path_buf());

    info!("plt pubkey evaluator setup start");
    let insert_1_to_s = false;
    let plt_pubkey_evaluator =
        GGH15BGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(key, D_SECRET, trapdoor_sigma, ERROR_SIGMA, dir.to_path_buf(), insert_1_to_s);
    info!("plt pubkey evaluator setup done");

    info!("circuit eval pubkey start");
    let enc_one_pubkey = Arc::new(enc_one.pubkey.clone());
    let result_pubkey =
        circuit.eval(&params, &enc_one_pubkey, &input_pubkeys_shared, Some(&plt_pubkey_evaluator));
    info!("circuit eval pubkey done");
    assert_eq!(result_pubkey.len(), 1);
    let sample_aux_start = Instant::now();
    plt_pubkey_evaluator.sample_aux_matrices(&params);
    info!(
        "plt_pubkey_evaluator.sample_aux_matrices elapsed_ms={:.3}",
        sample_aux_start.elapsed().as_secs_f64() * 1000.0
    );

    let wait_writes_start = Instant::now();
    wait_for_all_writes(dir.to_path_buf()).await.expect("storage writes should complete");
    info!(
        "wait_for_all_writes elapsed_ms={:.3}",
        wait_writes_start.elapsed().as_secs_f64() * 1000.0
    );

    let b0_matrix = plt_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
    let c_b0 = s_vec.clone() * &b0_matrix;
    let checkpoint_prefix = plt_pubkey_evaluator.checkpoint_prefix(&params);
    info!("plt encoding evaluator setup start");
    let plt_encoding_evaluator = GGH15BGGEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::new(key, dir.to_path_buf(), checkpoint_prefix, c_b0);
    info!("plt encoding evaluator setup done");

    info!("circuit eval encoding start");
    let enc_one_shared = Arc::new(enc_one.clone());
    let result_encoding = circuit.eval(
        &params,
        &enc_one_shared,
        &input_encodings_shared,
        Some(&plt_encoding_evaluator),
    );
    info!("circuit eval encoding done");
    assert_eq!(result_encoding.len(), 1);

    let encoding = &result_encoding[0];
    assert_eq!(encoding.pubkey, result_pubkey[0]);

    let expected = BigUint::from(expected_mod_p);
    let expected_plaintext =
        GpuDCRTPoly::from_biguint_to_constant(&params, expected * q_over_p.clone());
    let plaintext =
        encoding.plaintext.as_ref().expect("encoding plaintext should be available for this test");
    assert_eq!(
        plaintext, &expected_plaintext,
        "symbolic plaintext must match expected scaled mod-p value"
    );

    let decoded_from_plaintext = decode_mod_p_from_scaled_plaintext(plaintext, &q_over_p, P);
    assert_eq!(
        decoded_from_plaintext, expected_mod_p,
        "decoded value from encoding plaintext must match mod p result"
    );

    let s_times_pk = s_vec.clone() * &encoding.pubkey.matrix;
    let diff = encoding.vector.clone() - s_times_pk;
    let unit_vector = GpuDCRTPolyMatrix::unit_column_vector(&params, D_SECRET, D_SECRET - 1);
    let projected = diff.mul_decompose(&unit_vector);
    assert_eq!(projected.row_size(), 1);
    assert_eq!(projected.col_size(), 1);

    let projected_poly = projected.entry(0, 0);
    let coeff = projected_poly
        .coeffs()
        .into_iter()
        .next()
        .expect("projected poly must have at least one coefficient");
    let rounded = round_div_biguint(coeff.value(), &q_over_p);
    let decoded: u64 =
        (&rounded % BigUint::from(P)).try_into().expect("decoded coefficient must fit u64");
    assert_eq!(
        decoded, expected_mod_p,
        "decoded value from projected encoding.vector must match mod p result"
    );

    gpu_device_sync();
}
