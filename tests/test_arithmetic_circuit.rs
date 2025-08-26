use keccak_asm::Keccak256;
use mxx::{
    arithmetic::circuit::ArithmeticCircuit,
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
use tracing::info;

pub fn init_tracing() {
    tracing_subscriber::fmt::init();
}

#[test]
fn test_arithmetic_circuit_operations() {
    init_tracing();

    let params = DCRTPolyParams::default();
    let (moduli, _, _) = params.to_crt();
    let large_a = BigUint::from(140000u64);
    let large_b = BigUint::from(132000u64);
    let large_c = BigUint::from(50000u64);

    // Expected results for each operation.
    let add_expected = &large_a + &large_b;
    let mul_expected = &large_a * &large_c;
    let sub_expected = &large_a - &large_c;

    // Verify modular arithmetic correctness for all operations.
    for (i, &qi) in moduli.iter().enumerate() {
        let a_mod_qi = (&large_a % qi as u64).to_u64().unwrap();
        let b_mod_qi = (&large_b % qi as u64).to_u64().unwrap();
        let c_mod_qi = (&large_c % qi as u64).to_u64().unwrap();

        let add_mod_qi = (a_mod_qi + b_mod_qi) % qi as u64;
        let mul_mod_qi = (a_mod_qi * c_mod_qi) % qi as u64;
        let sub_mod_qi = if a_mod_qi >= c_mod_qi {
            a_mod_qi - c_mod_qi
        } else {
            qi as u64 - (c_mod_qi - a_mod_qi)
        };

        let add_expected_mod_qi = (&add_expected % qi as u64).to_u64().unwrap();
        let mul_expected_mod_qi = (&mul_expected % qi as u64).to_u64().unwrap();
        let sub_expected_mod_qi = (&sub_expected % qi as u64).to_u64().unwrap();

        assert_eq!(add_mod_qi, add_expected_mod_qi, "Addition should be consistent in slot {}", i);
        assert_eq!(
            mul_mod_qi, mul_expected_mod_qi,
            "Multiplication should be consistent in slot {}",
            i
        );
        assert_eq!(
            sub_mod_qi, sub_expected_mod_qi,
            "Subtraction should be consistent in slot {}",
            i
        );
    }

    let inputs = vec![large_a.clone(), large_b.clone(), large_c.clone()];
    let (_, crt_bits, _) = params.to_crt();
    let limb_bit_size = 5;

    // Test mixed operations in single circuit: (a + b) * c - a.
    let mut mixed_circuit = ArithmeticCircuit::<DCRTPoly>::setup(
        &params,
        crt_bits.div_ceil(limb_bit_size),
        limb_bit_size,
        1,
        &inputs,
        true,
    );
    let add_idx = mixed_circuit.add(0, 1); // a + b
    let mul_idx = mixed_circuit.mul(add_idx, 2); // (a + b) * c
    let final_idx = mixed_circuit.sub(mul_idx, 0); // (a + b) * c - a
    mixed_circuit.finalize(final_idx);

    // Test with polynomial evaluation
    let mixed_result = &mixed_circuit.evaluate_with_poly(&params, &inputs)[0];
    let mixed_expected = ((&large_a + &large_b) * &large_c) - &large_a;
    assert_eq!(
        mixed_result.to_const_int(),
        mixed_expected.to_u64().unwrap() as usize,
        "Mixed operations should be correct"
    );

    // Test with BGG public key evaluation
    let tmp_dir = tempdir().unwrap();
    let seed: [u8; 32] = [0u8; 32];
    let d = 1;
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, 1.0);
    let (b_epsilon_trapdoor, b_epsilon) = trapdoor_sampler.trapdoor(&params, d + 1);
    info!("start evaluate_with_bgg_pubkey");
    let pubkey_results = mixed_circuit.evaluate_with_bgg_pubkey::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        DCRTPolyUniformSampler,
    >(&params, inputs.len(), seed, tmp_dir.path().to_path_buf(), d, b_epsilon.clone(), b_epsilon_trapdoor, trapdoor_sampler.clone());
    info!("end evaluate_with_bgg_pubkey");
    // Test with BGG encoding evaluation
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
    let s_x_l = {
        let minus_one_poly = DCRTPoly::const_minus_one(&params);
        let mut secrets = secrets.to_vec();
        secrets.push(minus_one_poly);
        DCRTPolyMatrix::from_poly_vec_row(&params, secrets)
    };
    let p = s_x_l * &b_epsilon;
    info!("start evaluate_with_bgg_encoding");
    let encoding_results = mixed_circuit.evaluate_with_bgg_encoding::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyUniformSampler,
    >(&params, inputs.len(), seed, tmp_dir.path().to_path_buf(), d, &inputs, &secrets, p, 0.0);
    info!("end evaluate_with_bgg_encoding");
    // Verify that both BGG methods produce results
    assert_eq!(pubkey_results.len(), 1, "BGG pubkey evaluation should produce one result");
    assert_eq!(encoding_results.len(), 1, "BGG encoding evaluation should produce one result");

    // The actual values will be different due to encryption, but both should complete without
    // error
    println!("BGG public key evaluation completed successfully");
    println!("BGG encoding evaluation completed successfully");
}
