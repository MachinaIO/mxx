use keccak_asm::Keccak256;
use mxx::{
    arithmetic::circuit::{ArithGateId, ArithmeticCircuit},
    element::PolyElem,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler, hash::DCRTPolyHashSampler,
        trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
    },
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use tempfile::tempdir;
use tokio;
use tracing::info;

fn init_tracing() {
    tracing_subscriber::fmt::init();
}

#[tokio::test]
async fn test_arithmetic_circuit_operations() {
    init_tracing();
    let params = DCRTPolyParams::new(4, 2, 31, 17);
    let (_, crt_bits, _) = params.to_crt();
    info!("crt_bits={}", crt_bits);
    let large_a = BigUint::from(140000u64);
    let large_b = BigUint::from(132000u64);
    let large_c = BigUint::from(50000u64);
    let inputs = vec![large_a.clone(), large_b.clone(), large_c.clone()];

    let limb_bit_size = 3;

    // Test mixed operations in single circuit: (a + b) * c - a.
    let mut mixed_circuit =
        ArithmeticCircuit::<DCRTPoly>::setup(&params, limb_bit_size, inputs.len(), false, true);
    let add_idx = mixed_circuit.add(ArithGateId::new(0), ArithGateId::new(1)); // a + b
    let mul_idx = mixed_circuit.mul(add_idx, ArithGateId::new(2)); // (a + b) * c
    let final_idx = mixed_circuit.sub(mul_idx, ArithGateId::new(0)); // (a + b) * c - a
    mixed_circuit.output(final_idx);

    // Test with polynomial evaluation
    info!("start evaluate_with_poly");
    let mixed_poly_result = &mixed_circuit.evaluate_with_poly(&params, &inputs)[0];
    info!("end evaluate_with_poly");
    let mixed_expected =
        (((&large_a + &large_b) * &large_c) - &large_a) % params.modulus().as_ref();
    println!("coeffs {:?}", mixed_poly_result.coeffs());
    assert_eq!(
        mixed_poly_result.coeffs()[0].value().clone(),
        mixed_expected,
        "Mixed operations should be correct"
    );

    // Test with BGG public key evaluation
    let tmp_dir = tempdir().unwrap();
    let seed: [u8; 32] = [0u8; 32];
    let d = 1;
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, 1.0);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(&params, d + 1);
    info!("start evaluate_with_bgg_pubkey");
    let mixed_pubkey_result = mixed_circuit.evaluate_with_bgg_pubkey::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        DCRTPolyUniformSampler,
    >(&params,  seed, tmp_dir.path().to_path_buf(), d, pub_matrix.clone(), trapdoor, trapdoor_sampler.clone()).await;
    info!("end evaluate_with_bgg_pubkey");
    assert_eq!(mixed_pubkey_result.len(), 1);
    // Test with BGG encoding evaluation
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
    let s = {
        let minus_one_poly = DCRTPoly::const_minus_one(&params);
        let mut secrets = secrets.to_vec();
        secrets.push(minus_one_poly);
        DCRTPolyMatrix::from_poly_vec_row(&params, secrets)
    };
    let p = s.clone() * &pub_matrix;
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
        s * (DCRTPolyMatrix::gadget_matrix(&params, d + 1) * mixed_poly_result);
    assert_eq!(mixed_encoding_result[0].vector, mixed_encoding_expected);
}

#[tokio::test]
#[ignore]
async fn test_arithmetic_circuit_no_crt_limb1() {
    init_tracing();

    // this is curated to be have same CRT bit as test_arithmetic_circuit_operations
    // (crt_depth*crt_bit)
    let params = DCRTPolyParams::new(4, 1, 62, 17);

    let (moduli, _, crt_depth) = params.to_crt();
    assert_eq!(moduli.len(), 1, "Should have only one modulus for non-CRT");
    assert_eq!(crt_depth, 1, "CRT depth should be 1");

    info!("Non-CRT mode: single modulus = {}", moduli[0]);

    let large_a = BigUint::from(140000u64);
    let large_b = BigUint::from(132000u64);
    let large_c = BigUint::from(50000u64);

    let inputs = vec![large_a.clone(), large_b.clone(), large_c.clone()];
    let limb_bit_size = 1;

    // Test mixed operations in single circuit: (a + b) * c - a.
    let mut circuit = ArithmeticCircuit::<DCRTPoly>::setup(&params, limb_bit_size, 3, false, true);
    info!("Non-CRT: setup");

    let add_idx = circuit.add(ArithGateId::new(0), ArithGateId::new(1)); // a + b
    info!("Non-CRT: add");
    let mul_idx = circuit.mul(add_idx, ArithGateId::new(2)); // (a + b) * c
    info!("Non-CRT: mul");
    let final_idx = circuit.sub(mul_idx, ArithGateId::new(0)); // (a + b) * c - a
    info!("Non-CRT: sub");
    circuit.output(final_idx);

    // Test with polynomial evaluation.
    info!("Non-CRT: start evaluate_with_poly");
    let mixed_poly_result = &circuit.evaluate_with_poly(&params, &inputs)[0];
    info!("Non-CRT: end evaluate_with_poly");

    let expected = ((&large_a + &large_b) * &large_c) - &large_a;
    let single_modulus = moduli[0];
    let expected_mod = (&expected % single_modulus).to_u64().unwrap() as usize;

    info!("Expected full result: {}", expected);
    info!("Expected result mod {}: {}", single_modulus, expected_mod);
    info!("Actual result: {}", mixed_poly_result.to_const_int());

    assert_eq!(
        mixed_poly_result.to_const_int(),
        expected_mod,
        "Non-CRT mixed operations should be correct"
    );

    // Test with BGG public key evaluation.
    let tmp_dir = tempdir().unwrap();
    let seed: [u8; 32] = [0u8; 32];
    let d = 1;
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, 1.0);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(&params, d + 1);

    info!("Non-CRT: start evaluate_with_bgg_pubkey");
    let mixed_pubkey_result = circuit.evaluate_with_bgg_pubkey::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        DCRTPolyUniformSampler,
    >(&params, seed, tmp_dir.path().to_path_buf(), d, pub_matrix.clone(), trapdoor, trapdoor_sampler.clone()).await;
    info!("Non-CRT: end evaluate_with_bgg_pubkey");
    assert_eq!(mixed_pubkey_result.len(), 1);

    // Test with BGG encoding evaluation.
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
    let s = {
        let minus_one_poly = DCRTPoly::const_minus_one(&params);
        let mut secrets = secrets.to_vec();
        secrets.push(minus_one_poly);
        DCRTPolyMatrix::from_poly_vec_row(&params, secrets)
    };
    let p = s.clone() * &pub_matrix;

    info!("Non-CRT: start evaluate_with_bgg_encoding");
    let mixed_encoding_result = circuit.evaluate_with_bgg_encoding::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyUniformSampler,
    >(&params, seed, tmp_dir.path().to_path_buf(), &inputs, &secrets, p, 0.0);
    assert_eq!(mixed_encoding_result.len(), 1);
    assert_eq!(mixed_encoding_result[0].plaintext.as_ref().unwrap(), mixed_poly_result);
    assert_eq!(mixed_encoding_result[0].pubkey, mixed_pubkey_result[0]);
    let mixed_encoding_expected = s.clone() * &mixed_pubkey_result[0].matrix -
        s * (DCRTPolyMatrix::gadget_matrix(&params, d + 1) * mixed_poly_result);
    assert_eq!(mixed_encoding_result[0].vector, mixed_encoding_expected);
}
