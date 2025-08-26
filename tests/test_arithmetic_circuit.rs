use mxx::{
    arithmetic::circuit::ArithmeticCircuit,
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;

#[test]
fn test_arithmetic_circuit_operations() {
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
    let mixed_result = &mixed_circuit.evaluate_with_poly(&params, &inputs)[0];

    let mixed_expected = ((&large_a + &large_b) * &large_c) - &large_a;
    assert_eq!(
        mixed_result.to_const_int(),
        mixed_expected.to_u64().unwrap() as usize,
        "Mixed operations should be correct"
    );
}
