use keccak_asm::Keccak256;
use mxx::{
    arithmetic::circuit::{ArithGateId, ArithmeticCircuit},
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler, hash::DCRTPolyHashSampler,
        trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
    },
    utils::gen_biguint_for_modulus,
};
use num_bigint::BigUint;
use std::sync::Arc;
use tempfile::tempdir;
use tracing::info;

fn init_tracing() {
    // Install a global tracing subscriber once; ignore if already set by another test.
    let _ = tracing_subscriber::fmt::try_init();
}

#[tokio::test]
async fn test_arithmetic_circuit_operations() {
    // Test mixed operations in single circuit: (a + b) * c - a.
    init_tracing();
    let params = DCRTPolyParams::new(4, 2, 28, 17);
    let (_, crt_bits, _) = params.to_crt();
    info!("crt_bits={}", crt_bits);
    let n = 3;
    let limb_bit_size = 3;
    let mut rng = rand::rng();
    let a_vec: Vec<BigUint> = (0..n)
        .map(|_| gen_biguint_for_modulus(&mut rng, limb_bit_size, &params.modulus()))
        .collect::<Vec<_>>();
    let b_vec: Vec<BigUint> = (0..n)
        .map(|_| gen_biguint_for_modulus(&mut rng, limb_bit_size, &params.modulus()))
        .collect::<Vec<_>>();
    let c_vec: Vec<BigUint> = (0..n)
        .map(|_| gen_biguint_for_modulus(&mut rng, limb_bit_size, &params.modulus()))
        .collect::<Vec<_>>();
    let inputs = vec![&a_vec[..], &b_vec[..], &c_vec[..]];
    let mut mixed_circuit =
        ArithmeticCircuit::<DCRTPoly>::setup(&params, limb_bit_size, n, inputs.len(), true, true);
    let add_idx = mixed_circuit.add(ArithGateId::new(0), ArithGateId::new(1)); // a + b
    let mul_idx = mixed_circuit.mul(add_idx, ArithGateId::new(2)); // (a + b) * c
    let final_idx = mixed_circuit.sub(mul_idx, ArithGateId::new(0)); // (a + b) * c - a
    mixed_circuit.output(final_idx);

    // Test with polynomial evaluation
    info!("start evaluate_with_poly");
    let mixed_poly_result = &mixed_circuit.evaluate_with_poly(&params, &inputs)[0];
    info!("end evaluate_with_poly");
    let q = params.modulus();
    let q = q.as_ref();
    let expected_slots = a_vec
        .iter()
        .zip(b_vec.iter().zip(c_vec.iter()))
        .map(|(a, (b, c))| {
            // Compute ((a + b) * c - a) mod q without underflow
            let aa = a % q;
            let bb = b % q;
            let cc = c % q;
            let t = (&aa + &bb) % q; // (a + b) mod q
            let t = (t * &cc) % q; // ((a + b) * c) mod q
            (t + (q - &aa)) % q // subtract a mod q safely
        })
        .collect::<Vec<_>>();
    let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_slots);
    assert_eq!(mixed_poly_result, &expected_poly, "Mixed operations should be correct");

    // Test with BGG public key evaluation
    let tmp_dir = tempdir().unwrap();
    let seed: [u8; 32] = [0u8; 32];
    let d = 1;
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, 1.0);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(&params, d);
    let trapdoor = Arc::new(trapdoor);
    let pub_matrix = Arc::new(pub_matrix);
    info!("start evaluate_with_bgg_pubkey");
    let mixed_pubkey_result = mixed_circuit.evaluate_with_bgg_pubkey::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        DCRTPolyUniformSampler,
    >(&params,  seed, tmp_dir.path().to_path_buf(), d, pub_matrix.clone(),  trapdoor.clone(),
trapdoor_sampler.clone()).await;
    info!("end evaluate_with_bgg_pubkey");
    assert_eq!(mixed_pubkey_result.len(), 1);
    // Test with BGG encoding evaluation
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
    let s = DCRTPolyMatrix::from_poly_vec_row(&params, secrets.to_vec());
    let p = s.clone() * pub_matrix.as_ref();
    info!("start evaluate_with_bgg_encoding");
    let mixed_encoding_result = mixed_circuit.evaluate_with_bgg_encoding::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyUniformSampler,
    >(&params, seed, tmp_dir.path().to_path_buf(),  &inputs, &secrets, p, 0.0);
    info!("end evaluate_with_bgg_encoding");
    assert_eq!(mixed_encoding_result.len(), 1);
    assert_eq!(mixed_encoding_result[0].plaintext.as_ref().unwrap(), mixed_poly_result);
    assert_eq!(mixed_encoding_result[0].pubkey, mixed_pubkey_result[0]);
    let mixed_encoding_expected = s.clone() * &mixed_pubkey_result[0].matrix -
        s * (DCRTPolyMatrix::gadget_matrix(&params, d) * mixed_poly_result);
    assert_eq!(mixed_encoding_result[0].vector, mixed_encoding_expected);
}

#[tokio::test]
async fn test_arithmetic_circuit_no_crt_limb1() {
    init_tracing();

    // this is curated to be have same CRT bit as test_arithmetic_circuit_operations
    // (crt_depth*crt_bit)
    let params = DCRTPolyParams::new(4, 1, 56, 17);

    let (moduli, _, crt_depth) = params.to_crt();
    assert_eq!(moduli.len(), 1, "Should have only one modulus for non-CRT");
    assert_eq!(crt_depth, 1, "CRT depth should be 1");
    info!("Non-CRT mode: single modulus = {}", moduli[0]);
    let n = 3;
    let limb_bit_size = 1;
    let mut rng = rand::rng();
    let a_vec: Vec<BigUint> = (0..n)
        .map(|_| gen_biguint_for_modulus(&mut rng, limb_bit_size, &params.modulus()))
        .collect::<Vec<_>>();
    let b_vec: Vec<BigUint> = (0..n)
        .map(|_| gen_biguint_for_modulus(&mut rng, limb_bit_size, &params.modulus()))
        .collect::<Vec<_>>();
    let c_vec: Vec<BigUint> = (0..n)
        .map(|_| gen_biguint_for_modulus(&mut rng, limb_bit_size, &params.modulus()))
        .collect::<Vec<_>>();
    let inputs = vec![&a_vec[..], &b_vec[..], &c_vec[..]];

    // Test mixed operations in single circuit: (a + b) * c - a.
    let mut mixed_circuit =
        ArithmeticCircuit::<DCRTPoly>::setup(&params, limb_bit_size, n, inputs.len(), false, true);
    let add_idx = mixed_circuit.add(ArithGateId::new(0), ArithGateId::new(1)); // a + b
    let mul_idx = mixed_circuit.mul(add_idx, ArithGateId::new(2)); // (a + b) * c
    let final_idx = mixed_circuit.sub(mul_idx, ArithGateId::new(0)); // (a + b) * c - a
    mixed_circuit.output(final_idx);

    // Test with polynomial evaluation
    info!("start evaluate_with_poly");
    let mixed_poly_result = &mixed_circuit.evaluate_with_poly(&params, &inputs)[0];
    info!("end evaluate_with_poly");
    let q = params.modulus();
    let q = q.as_ref();
    let expected_slots = a_vec
        .iter()
        .zip(b_vec.iter().zip(c_vec.iter()))
        .map(|(a, (b, c))| {
            // Compute ((a + b) * c - a) mod q without underflow
            let aa = a % q;
            let bb = b % q;
            let cc = c % q;
            let t = (&aa + &bb) % q; // (a + b) mod q
            let t = (t * &cc) % q; // ((a + b) * c) mod q
            (t + (q - &aa)) % q // subtract a mod q safely
        })
        .collect::<Vec<_>>();
    let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_slots);
    assert_eq!(mixed_poly_result, &expected_poly, "Mixed operations should be correct");

    // Test with BGG public key evaluation.
    let tmp_dir = tempdir().unwrap();
    let seed: [u8; 32] = [0u8; 32];
    let d = 1;
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, 1.0);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(&params, d);
    let trapdoor = Arc::new(trapdoor);
    let pub_matrix = Arc::new(pub_matrix);
    info!("start evaluate_with_bgg_pubkey");
    let mixed_pubkey_result = mixed_circuit.evaluate_with_bgg_pubkey::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        DCRTPolyUniformSampler,
    >(&params,  seed, tmp_dir.path().to_path_buf(), d, pub_matrix.clone(),  trapdoor.clone(),
trapdoor_sampler.clone()).await;
    info!("Non-CRT: end evaluate_with_bgg_pubkey");
    assert_eq!(mixed_pubkey_result.len(), 1);

    // Test with BGG encoding evaluation.
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
    let s = DCRTPolyMatrix::from_poly_vec_row(&params, secrets.to_vec());
    let p = s.clone() * pub_matrix.as_ref();
    info!("Non-CRT: start evaluate_with_bgg_encoding");
    let mixed_encoding_result = mixed_circuit.evaluate_with_bgg_encoding::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyUniformSampler,
    >(&params, seed, tmp_dir.path().to_path_buf(), &inputs, &secrets, p, 0.0);
    info!("Non-CRT: end evaluate_with_bgg_encoding");
    assert_eq!(mixed_encoding_result.len(), 1);
    assert_eq!(mixed_encoding_result[0].plaintext.as_ref().unwrap(), mixed_poly_result);
    assert_eq!(mixed_encoding_result[0].pubkey, mixed_pubkey_result[0]);
    let mixed_encoding_expected = s.clone() * &mixed_pubkey_result[0].matrix -
        s * (DCRTPolyMatrix::gadget_matrix(&params, d) * mixed_poly_result);
    assert_eq!(mixed_encoding_result[0].vector, mixed_encoding_expected);
}
