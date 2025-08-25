use mxx::{
    arithmetic::circuit::ArithmeticCircuit,
    gadgets::crt::biguint_to_crt_poly,
    lookup::poly::PolyPltEvaluator,
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use num_bigint::BigUint;

#[test]
fn test_arithmetic_circuit_chaining() {
    let params = DCRTPolyParams::default();
    // Test case: (10 + 20) * 5 - 3 = 30 * 5 - 3 = 150 - 3 = 147
    let inputs =
        vec![BigUint::from(10u64), BigUint::from(20u64), BigUint::from(5u64), BigUint::from(3u64)];

    let (_, crt_bits, _) = params.to_crt();
    let limb_bit_size = 5;
    let mut circuit = ArithmeticCircuit::<DCRTPoly>::setup(
        &params,
        crt_bits.div_ceil(limb_bit_size),
        limb_bit_size,
        1,
        &inputs,
        true,
    );
    // (input[0] + input[1]) * input[2] - input[3]
    // Initial indices: 0=10, 1=20, 2=5, 3=3
    let sum_index = circuit.add(0, 1); // index 4: 10 + 20 = 30
    let product_index = circuit.mul(sum_index, 2); // index 5: 30 * 5 = 150  
    let final_index = circuit.sub(product_index, 3); // index 6: 150 - 3 = 147
    circuit.finalize(final_index);

    // Evaluate the circuit
    let one = DCRTPoly::const_one(&params);
    let mut all_input_polys = Vec::new();
    for input in &inputs {
        let crt_limbs = biguint_to_crt_poly(limb_bit_size, &params, input);
        all_input_polys.extend(crt_limbs);
    }

    let plt_evaluator = PolyPltEvaluator::new();
    let results = circuit.poly_circuit.eval(&params, &one, &all_input_polys, Some(plt_evaluator));
    assert_eq!(results.len(), 1);
    let actual_result = results[0].to_const_int();

    println!("Chained operation result: {}", actual_result);
    println!(
        "Sum index: {}, Product index: {}, Final index: {}",
        sum_index, product_index, final_index
    );

    // Verify: (10 + 20) * 5 - 3 = 30 * 5 - 3 = 150 - 3 = 147
    assert_eq!(actual_result, 147, "Circuit should compute (10+20)*5-3 = 147");
}
