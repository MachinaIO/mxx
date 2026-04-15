use super::*;
use crate::{
    element::PolyElem,
    lookup::{PltEvaluator, poly::PolyPltEvaluator},
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    rlwe_enc::rlwe_encrypt,
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    utils::{create_bit_random_poly, create_random_poly},
};
use num_bigint::BigUint;
fn eval_with_const_one<PE>(
    circuit: &PolyCircuit<DCRTPoly>,
    params: &DCRTPolyParams,
    inputs: &[DCRTPoly],
    plt_evaluator: Option<&PE>,
) -> Vec<DCRTPoly>
where
    PE: PltEvaluator<DCRTPoly>,
{
    let one = DCRTPoly::const_one(params);
    let eval_inputs = inputs.to_vec();
    circuit.eval(params, one, eval_inputs, plt_evaluator, None, None)
}

#[test]
fn test_eval_add() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials using UniformSampler
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);

    // Create a circuit with an Add operation
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let add_gate = circuit.add_gate(inputs[0], inputs[1]);
    circuit.output(vec![add_gate]);

    // Evaluate the circuit
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );

    // Expected result: poly1 + poly2
    let expected = poly1 + poly2;

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].coeffs(), expected.coeffs());
}

#[test]
fn test_eval_sub() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials using UniformSampler
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);

    // Create a circuit with a Sub operation
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let sub_gate = circuit.sub_gate(inputs[0], inputs[1]);
    circuit.output(vec![sub_gate]);

    // Evaluate the circuit
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );

    // Expected result: poly1 - poly2
    let expected = poly1 - poly2;

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_eval_mul() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials using UniformSampler
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);

    // Create a circuit with a Mul operation
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let mul_gate = circuit.mul_gate(inputs[0], inputs[1]);
    circuit.output(vec![mul_gate]);

    // Evaluate the circuit
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );

    // Expected result: poly1 * poly2
    let expected = poly1 * poly2;

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_rotate_gate_uses_small_scalar_mul() {
    let params = DCRTPolyParams::default();
    let input = create_random_poly(&params);

    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(1).to_vec();
    let rotated = circuit.rotate_gate(inputs[0], 3);
    circuit.output(vec![rotated]);

    let result = eval_with_const_one(
        &circuit,
        &params,
        std::slice::from_ref(&input),
        None::<&PolyPltEvaluator>,
    );
    let expected = input * DCRTPoly::from_u32s(&params, &[0, 0, 0, 1]);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
    assert_eq!(circuit.count_gates_by_type_vec().get(&PolyGateKind::SmallScalarMul), Some(&1));
}

#[test]
fn test_const_digits() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create a circuit with a const_bit_poly gate
    let mut circuit = PolyCircuit::new();
    // We need to call input() to initialize the circuit
    circuit.input(1);

    // Define a specific bit pattern
    // This will create a polynomial with coefficients:
    // [1, 0, 1, 1]
    // (where 1 is at positions 0, 2, 3, and 4)
    let digits = vec![1u32, 0u32, 1u32, 1u32];
    let digits_poly_gate = circuit.const_digits(&digits);
    circuit.output(vec![digits_poly_gate]);

    // Evaluate the circuit with any input (it won't be used)
    let dummy_input = create_random_poly(&params);
    let result = eval_with_const_one(&circuit, &params, &[dummy_input], None::<&PolyPltEvaluator>);

    // Verify the result
    assert_eq!(result.len(), 1);

    // Check that the coefficients match the bit pattern
    let coeffs = result[0].coeffs();
    for (i, digit) in digits.iter().enumerate() {
        if digit != &0 {
            assert_eq!(
                coeffs[i].value(),
                &BigUint::from(1u8),
                "Coefficient at position {} should be 1",
                i
            );
        } else {
            assert_eq!(
                coeffs[i].value(),
                &BigUint::from(0u8),
                "Coefficient at position {} should be 0",
                i
            );
        }
    }

    // Check that remaining coefficients are 0
    for (i, _) in coeffs.iter().enumerate().skip(digits.len()) {
        assert_eq!(
            coeffs[i].value(),
            &BigUint::from(0u8),
            "Coefficient at position {} should be 0",
            i
        );
    }
}

#[test]
fn test_eval_complex() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials using UniformSampler
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);
    let poly3 = create_random_poly(&params);

    // Create a complex circuit: (poly1 + poly2) - poly3
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(3).to_vec();

    // poly1 + poly2
    let add_gate = circuit.add_gate(inputs[0], inputs[1]);

    // (poly1 + poly2) - poly3
    let sub_gate = circuit.sub_gate(add_gate, inputs[2]);

    circuit.output(vec![sub_gate]);

    // Evaluate the circuit
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone(), poly3.clone()],
        None::<&PolyPltEvaluator>,
    );

    // Expected result: (poly1 + poly2) - poly3
    let expected = (poly1 + poly2) - poly3;

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_eval_multiple_outputs() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials using UniformSampler
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);

    // Create a circuit with multiple outputs
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();

    // poly1 + poly2
    let add_gate = circuit.add_gate(inputs[0], inputs[1]);

    // poly1 - poly2
    let sub_gate = circuit.sub_gate(inputs[0], inputs[1]);

    // poly1 * poly2
    let mul_gate = circuit.mul_gate(inputs[0], inputs[1]);

    circuit.output(vec![add_gate, sub_gate, mul_gate]);

    // Evaluate the circuit
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );

    // Expected results
    let expected_add = poly1.clone() + poly2.clone();
    let expected_sub = poly1.clone() - poly2.clone();
    let expected_mul = poly1 * poly2;

    // Verify the results
    assert_eq!(result.len(), 3);
    assert_eq!(result[0], expected_add);
    assert_eq!(result[1], expected_sub);
    assert_eq!(result[2], expected_mul);
}

#[test]
fn test_multiple_input_calls_with_nonconsecutive_gate_ids() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials using UniformSampler
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);

    // Create a circuit
    let mut circuit = PolyCircuit::new();

    // First input call: creates GateId(1) as input
    let inputs_first = circuit.input(1).to_vec();
    assert_eq!(inputs_first.len(), 1);

    // Insert a gate between input calls so next input gate is non-consecutive
    // Use a const-digits gate which introduces a new gate with no inputs
    circuit.const_digits(&[1u32, 0u32, 1u32]);

    // Second input call: creates a new input gate with a higher, non-consecutive GateId
    let inputs_second = circuit.input(1).to_vec();
    assert_eq!(inputs_second.len(), 1);

    // Ensure non-consecutive input GateIds (there should be a gap)
    assert_ne!(inputs_second[0].0, inputs_first[0].0 + 1);

    // Build a simple circuit that adds the two inputs together
    let add_gate = circuit.add_gate(inputs_first[0], inputs_second[0]);
    circuit.output(vec![add_gate]);

    // Evaluate the circuit: inputs are assigned in ascending input GateId order
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );

    let expected = poly1 + poly2;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_eval_deep_complex() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials using UniformSampler
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);
    let poly3 = create_random_poly(&params);
    let poly4 = create_random_poly(&params);

    // Create a complex circuit with depth = 4
    // Circuit structure:
    // Level 1: a = poly1 + poly2, b = poly3 * poly4, d = poly1 - poly3
    // Level 2: c = a * b
    // Level 3: e = c + d
    // Level 4: f = e * e
    // Output: f
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(4).to_vec();

    // Level 1
    let a = circuit.add_gate(inputs[0], inputs[1]); // poly1 + poly2
    let b = circuit.mul_gate(inputs[2], inputs[3]); // poly3 * poly4
    let d = circuit.sub_gate(inputs[0], inputs[2]); // poly1 - poly3

    // Level 2
    let c = circuit.mul_gate(a, b); // (poly1 + poly2) * (poly3 * poly4)

    // Level 3
    let e = circuit.add_gate(c, d); // ((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3)

    // Level 4
    let f = circuit.mul_gate(e, e); // (((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3))^2

    circuit.output(vec![f]);

    // Evaluate the circuit
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone(), poly3.clone(), poly4.clone()],
        None::<&PolyPltEvaluator>,
    );

    // Expected result: (((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3))^2
    let expected = (((poly1.clone() + poly2.clone()) * (poly3.clone() * poly4.clone())) +
        (poly1.clone() - poly3.clone())) *
        (((poly1.clone() + poly2) * (poly3.clone() * poly4)) + (poly1 - poly3));

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_boolean_gate_and() {
    let params = DCRTPolyParams::default();
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let and_result = circuit.and_gate(inputs[0], inputs[1]);
    circuit.output(vec![and_result]);
    let poly1 = create_bit_random_poly(&params);
    let poly2 = create_bit_random_poly(&params);
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );
    let expected = poly1.clone() * poly2;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].coeffs(), expected.coeffs());
}

#[test]
fn test_boolean_gate_not() {
    let params = DCRTPolyParams::default();
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(1).to_vec();
    let not_result = circuit.not_gate(inputs[0]);
    circuit.output(vec![not_result]);
    let poly1 = create_bit_random_poly(&params);
    let result = eval_with_const_one(
        &circuit,
        &params,
        std::slice::from_ref(&poly1),
        None::<&PolyPltEvaluator>,
    );
    let expected = DCRTPoly::const_one(&params) - poly1.clone();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].coeffs(), expected.coeffs());
}

#[test]
fn test_boolean_gate_or() {
    let params = DCRTPolyParams::default();
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let or_result = circuit.or_gate(inputs[0], inputs[1]);
    circuit.output(vec![or_result]);
    let poly1 = create_bit_random_poly(&params);
    let poly2 = create_bit_random_poly(&params);
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );
    let expected = (poly1.clone() + poly2.clone()) - (poly1 * poly2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].coeffs(), expected.coeffs());
}

#[test]
fn test_boolean_gate_nand() {
    let params = DCRTPolyParams::default();
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let nand_result = circuit.nand_gate(inputs[0], inputs[1]);
    circuit.output(vec![nand_result]);
    let poly1 = create_bit_random_poly(&params);
    let poly2 = create_bit_random_poly(&params);
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );
    let expected = DCRTPoly::const_one(&params) - (poly1 * poly2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].coeffs(), expected.coeffs());
}

#[test]
fn test_boolean_gate_nor() {
    let params = DCRTPolyParams::default();
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let nor_result = circuit.nor_gate(inputs[0], inputs[1]); // poly1 AND poly2
    circuit.output(vec![nor_result]);
    let poly1 = create_bit_random_poly(&params);
    let poly2 = create_bit_random_poly(&params);
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );
    let expected =
        DCRTPoly::const_one(&params) - ((poly1.clone() + poly2.clone()) - (poly1 * poly2));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].coeffs(), expected.coeffs());
}

#[test]
fn test_boolean_gate_xor() {
    let params = DCRTPolyParams::default();
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let nor_result = circuit.xor_gate(inputs[0], inputs[1]);
    circuit.output(vec![nor_result]);
    let poly1 = create_bit_random_poly(&params);
    let poly2 = create_bit_random_poly(&params);
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );
    let expected = (poly1.clone() + poly2.clone()) -
        (DCRTPoly::from_usize_to_constant(&params, 2) * poly1 * poly2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].coeffs(), expected.coeffs());
}

#[test]
fn test_boolean_gate_xnor() {
    let params = DCRTPolyParams::default();
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(2).to_vec();
    let xnor_result = circuit.xnor_gate(inputs[0], inputs[1]);
    circuit.output(vec![xnor_result]);
    let poly1 = create_bit_random_poly(&params);
    let poly2 = create_bit_random_poly(&params);
    let result = eval_with_const_one(
        &circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );
    let expected = DCRTPoly::const_one(&params) -
        ((poly1.clone() + poly2.clone()) -
            (DCRTPoly::from_usize_to_constant(&params, 2) * poly1 * poly2));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].coeffs(), expected.coeffs());
}

#[test]
fn test_mul_fhe_poly_bits_mul_by_poly_circuit() {
    let mut circuit = PolyCircuit::new();
    let params = DCRTPolyParams::default();
    let sampler = DCRTPolyUniformSampler::new();
    let sigma = 3.0;
    let log_q = params.modulus_bits();

    // encrypt a polynomial m using FHE secret key encryption
    // Generate random message bits
    let m = sampler.sample_poly(&params, &DistType::BitDist);

    // Encrypt the message
    let a = sampler.sample_poly(&params, &DistType::BitDist);
    let t = sampler.sample_poly(&params, &DistType::BitDist);

    let m_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![m.clone()]);
    let a_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![a.clone()]);
    let t_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![t.clone()]);
    let b_mat = rlwe_encrypt(&params, &sampler, &t_mat, &a_mat, &m_mat, sigma);
    let b = b_mat.entry(0, 0);

    // ct = (a, b)
    let a_bits = a.decompose_base(&params);
    let b_bits = b.decompose_base(&params);

    let x = DCRTPoly::const_one(&params);

    let inputs = circuit.input(a_bits.len() + b_bits.len() + 1).to_vec();
    assert_eq!(inputs.len(), params.modulus_bits() * 2 + 1);

    // Input: ct[bits], x
    // Output: ct[bits] * x
    let x_id = inputs[inputs.len() - 1];
    let output_ids: Vec<_> = inputs
        .iter()
        .take(inputs.len() - 1)
        .map(|&input_id| circuit.mul_gate(input_id, x_id))
        .collect();

    circuit.output(output_ids);

    // concatenate decomposed_c0 and decomposed_c1 and x
    let input = [a_bits, b_bits, vec![x.clone()]].concat();
    let result = eval_with_const_one(&circuit, &params, &input, None::<&PolyPltEvaluator>);

    assert_eq!(result.len(), log_q * 2);

    let a_bits_eval = result[..params.modulus_bits()].to_vec();
    let b_bits_eval = result[params.modulus_bits()..].to_vec();

    let a_eval = DCRTPoly::from_decomposed(&params, &a_bits_eval);
    let b_eval = DCRTPoly::from_decomposed(&params, &b_bits_eval);

    assert_eq!(a_eval, &a * &x);
    assert_eq!(b_eval, &b * &x);

    // decrypt the result
    let plaintext = b_eval - (a_eval * t);
    // recover the message bits
    let plaintext_bits = plaintext.extract_bits_with_threshold(&params);
    assert_eq!(plaintext_bits, (m * x).to_bool_vec());
}

#[test]
fn test_register_and_call_sub_circuit() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials using UniformSampler
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);

    // Create a sub-circuit that performs addition and multiplication
    let mut sub_circuit = PolyCircuit::new();
    let sub_inputs = sub_circuit.input(2).to_vec();

    // Add operation: poly1 + poly2
    let add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);

    // Mul operation: poly1 * poly2
    let mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);

    // Set the outputs of the sub-circuit
    sub_circuit.output(vec![add_gate, mul_gate]);

    // Create the main circuit
    let mut main_circuit = PolyCircuit::new();
    let main_inputs = main_circuit.input(2).to_vec();

    // Register the sub-circuit and get its ID
    let sub_circuit_id = main_circuit.register_sub_circuit(sub_circuit);

    // Call the sub-circuit with the main circuit's inputs
    let sub_outputs =
        main_circuit.call_sub_circuit(sub_circuit_id, &[main_inputs[0], main_inputs[1]]);

    // Verify we got two outputs from the sub-circuit
    assert_eq!(sub_outputs.len(), 2);

    // Use the sub-circuit outputs for further operations
    // For example, subtract the multiplication result from the addition result
    let final_gate = main_circuit.sub_gate(sub_outputs[0], sub_outputs[1]);

    // Set the output of the main circuit
    main_circuit.output(vec![final_gate]);

    // Evaluate the main circuit
    let result = eval_with_const_one(
        &main_circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );

    // Expected result: (poly1 + poly2) - (poly1 * poly2)
    let expected = (poly1.clone() + poly2.clone()) - (poly1 * poly2);

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_batched_wire_range_helpers() {
    let batch = BatchedWire::from_start_len(GateId(5), 4);
    let (left, right) = batch.split_at(2);

    assert_eq!(left, BatchedWire::new(GateId(5), GateId(7)));
    assert_eq!(right, BatchedWire::new(GateId(7), GateId(9)));
    assert_eq!(left.at(1), BatchedWire::single(GateId(6)));
    assert_eq!(batch.slice(1..3), BatchedWire::new(GateId(6), GateId(8)));
    assert_eq!(BatchedWire::from_batches([left, right]), batch);
}

#[test]
fn test_sub_circuit_call_info_preserves_batched_input_ranges() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(3);
    sub_circuit.output([sub_inputs]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let main_inputs = main_circuit.input(4);
    let (shared_prefix, remainder) = main_inputs.split_at(2);
    let suffix = remainder.at(0);
    let prefix_set_id = main_circuit.intern_input_set([shared_prefix]);
    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let outputs = main_circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
        sub_id,
        prefix_set_id,
        [suffix],
        &[],
    );
    main_circuit.output(outputs);

    let call = main_circuit.sub_circuit_call_info(0);
    assert_eq!(call.inputs, vec![shared_prefix, suffix]);
    assert_eq!(main_circuit.input_set(prefix_set_id).as_ref(), &[shared_prefix]);
    assert_eq!(main_circuit.non_free_depth(), 0);
}

#[test]
fn test_register_and_call_parameterized_sub_circuit() {
    let params = DCRTPolyParams::default();
    let input_poly = create_random_poly(&params);

    let mut sub_circuit = PolyCircuit::new();
    let scalar_param = sub_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
    let sub_inputs = sub_circuit.input(1).to_vec();
    let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
    sub_circuit.output(vec![scaled]);

    let mut main_circuit = PolyCircuit::new();
    let main_inputs = main_circuit.input(1).to_vec();
    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let doubled = main_circuit.call_sub_circuit_with_bindings(
        sub_id,
        &[main_inputs[0]],
        &[SubCircuitParamValue::SmallScalarMul(vec![2])],
    );
    let tripled = main_circuit.call_sub_circuit_with_bindings(
        sub_id,
        &[main_inputs[0]],
        &[SubCircuitParamValue::SmallScalarMul(vec![3])],
    );
    main_circuit.output(vec![doubled[0], tripled[0]]);

    let result = eval_with_const_one(
        &main_circuit,
        &params,
        std::slice::from_ref(&input_poly),
        None::<&PolyPltEvaluator>,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], input_poly.small_scalar_mul(&params, &[2]));
    assert_eq!(result[1], input_poly.small_scalar_mul(&params, &[3]));
    assert_eq!(main_circuit.non_free_depth(), 0);
}

#[test]
fn test_parameterized_sub_circuit_reuses_identical_binding_sets() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let scalar_param = sub_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
    let sub_inputs = sub_circuit.input(1).to_vec();
    let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
    sub_circuit.output(vec![scaled]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let main_inputs = main_circuit.input(1).to_vec();
    let sub_id = main_circuit.register_sub_circuit(sub_circuit);

    let _ = main_circuit.call_sub_circuit_with_bindings(
        sub_id,
        &[main_inputs[0]],
        &[SubCircuitParamValue::SmallScalarMul(vec![7])],
    );
    let _ = main_circuit.call_sub_circuit_with_bindings(
        sub_id,
        &[main_inputs[0]],
        &[SubCircuitParamValue::SmallScalarMul(vec![7])],
    );
    let _ = main_circuit.call_sub_circuit_with_bindings(
        sub_id,
        &[main_inputs[0]],
        &[SubCircuitParamValue::SmallScalarMul(vec![9])],
    );

    let binding_set_ids =
        main_circuit.sub_circuit_calls.values().map(|call| call.binding_set_id).collect::<Vec<_>>();
    assert_eq!(binding_set_ids.len(), 3);
    assert_eq!(binding_set_ids[0], binding_set_ids[1]);
    assert_ne!(binding_set_ids[0], binding_set_ids[2]);
    assert_eq!(main_circuit.binding_registry.binding_sets.len(), 2);
}

#[test]
fn test_register_sub_circuit_reuses_child_binding_sets_without_duplication() {
    let mut leaf_circuit = PolyCircuit::<DCRTPoly>::new();
    let scalar_param = leaf_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
    let leaf_inputs = leaf_circuit.input(1).to_vec();
    let scaled = leaf_circuit.small_scalar_mul_param(leaf_inputs[0], scalar_param);
    leaf_circuit.output(vec![scaled]);

    let mut middle_circuit = PolyCircuit::<DCRTPoly>::new();
    let middle_inputs = middle_circuit.input(1).to_vec();
    let leaf_id = middle_circuit.register_sub_circuit(leaf_circuit);
    let _ = middle_circuit.call_sub_circuit_with_bindings(
        leaf_id,
        &[middle_inputs[0]],
        &[SubCircuitParamValue::SmallScalarMul(vec![11])],
    );
    let _ = middle_circuit.call_sub_circuit_with_bindings(
        leaf_id,
        &[middle_inputs[0]],
        &[SubCircuitParamValue::SmallScalarMul(vec![11])],
    );
    assert_eq!(middle_circuit.binding_registry.binding_sets.len(), 1);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let main_inputs = main_circuit.input(1).to_vec();
    let middle_id = main_circuit.register_sub_circuit(middle_circuit);
    let _ = main_circuit.call_sub_circuit(middle_id, &[main_inputs[0]]);

    assert_eq!(main_circuit.binding_registry.binding_sets.len(), 2);
    let registered_middle =
        main_circuit.sub_circuits.get(&middle_id).expect("middle circuit missing");
    assert!(Arc::ptr_eq(&registered_middle.binding_registry, &main_circuit.binding_registry));
}

#[test]
fn test_register_and_call_summed_parameterized_sub_circuit() {
    let params = DCRTPolyParams::default();
    let input_poly = create_random_poly(&params);

    let mut sub_circuit = PolyCircuit::new();
    let scalar_param = sub_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
    let sub_inputs = sub_circuit.input(1).to_vec();
    let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
    sub_circuit.output(vec![scaled]);

    let mut main_circuit = PolyCircuit::new();
    let main_inputs = main_circuit.input(1).to_vec();
    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let double_id =
        main_circuit.intern_binding_set(&[SubCircuitParamValue::SmallScalarMul(vec![2])]);
    let triple_id =
        main_circuit.intern_binding_set(&[SubCircuitParamValue::SmallScalarMul(vec![3])]);
    let input_set_id = main_circuit.intern_input_set(&[main_inputs[0]]);
    let outputs = main_circuit.call_sub_circuit_sum_many_with_binding_set_ids(
        sub_id,
        vec![input_set_id, input_set_id, input_set_id],
        vec![double_id, double_id, triple_id],
    );
    main_circuit.output(outputs.clone());

    let result = eval_with_const_one(
        &main_circuit,
        &params,
        std::slice::from_ref(&input_poly),
        None::<&PolyPltEvaluator>,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], input_poly.small_scalar_mul(&params, &[7]));
    assert_eq!(main_circuit.non_free_depth(), 0);
    assert_eq!(main_circuit.binding_registry.binding_sets.len(), 2);
    assert_eq!(main_circuit.input_set_registry.input_sets.len(), 1);
    assert_eq!(main_circuit.summed_sub_circuit_calls.len(), 1);
}

#[test]
fn test_summed_sub_circuit_non_free_depth_uses_max_inner_call_depth() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(2).to_vec();
    let product = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
    sub_circuit.output(vec![product]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = main_circuit.input(3).to_vec();
    let precomputed = main_circuit.mul_gate(inputs[0], inputs[1]);
    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let direct_input_set_id = main_circuit.intern_input_set(&[inputs[0], inputs[2]]);
    let precomputed_input_set_id = main_circuit.intern_input_set(&[precomputed, inputs[2].into()]);
    let outputs = main_circuit.call_sub_circuit_sum_many_with_binding_set_ids(
        sub_id,
        vec![direct_input_set_id, precomputed_input_set_id],
        vec![main_circuit.intern_binding_set(&[]), main_circuit.intern_binding_set(&[])],
    );
    main_circuit.output(outputs);

    assert_eq!(main_circuit.non_free_depth(), 2);
}

#[test]
fn test_nested_sub_circuits_with_disk_api_compatibility() {
    let params = DCRTPolyParams::default();

    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);
    let poly3 = create_random_poly(&params);

    let mut inner_circuit = PolyCircuit::new();
    let inner_inputs = inner_circuit.input(2).to_vec();
    let mul_gate = inner_circuit.mul_gate(inner_inputs[0], inner_inputs[1]);
    inner_circuit.output(vec![mul_gate]);

    let mut middle_circuit = PolyCircuit::new();
    let middle_inputs = middle_circuit.input(3).to_vec();
    let inner_circuit_id = middle_circuit.register_sub_circuit(inner_circuit);
    let inner_outputs =
        middle_circuit.call_sub_circuit(inner_circuit_id, &[middle_inputs[0], middle_inputs[1]]);
    let add_gate = middle_circuit.add_gate(inner_outputs[0], middle_inputs[2]);
    middle_circuit.output(vec![add_gate]);

    let mut main_circuit = PolyCircuit::new();
    main_circuit.enable_subcircuits_in_disk("unused");
    let main_inputs = main_circuit.input(3).to_vec();
    let middle_circuit_id = main_circuit.register_sub_circuit(middle_circuit);
    let middle_outputs = main_circuit
        .call_sub_circuit(middle_circuit_id, &[main_inputs[0], main_inputs[1], main_inputs[2]]);
    let square_gate = main_circuit.mul_gate(middle_outputs[0], middle_outputs[0]);
    main_circuit.output(vec![square_gate]);

    assert!(
        main_circuit
            .sub_circuits
            .values()
            .all(|sub| Arc::ptr_eq(&sub.lookup_registry, &main_circuit.lookup_registry))
    );

    let gate_counts = main_circuit.count_gates_by_type_vec();
    assert_eq!(main_circuit.lut_vector_len_with_subcircuits(), 0);
    assert_eq!(gate_counts.get(&PolyGateKind::Mul).copied().unwrap_or(0), 2);
    assert_eq!(main_circuit.non_free_depth(), 2);

    let result = eval_with_const_one(
        &main_circuit,
        &params,
        &[poly1.clone(), poly2.clone(), poly3.clone()],
        None::<&PolyPltEvaluator>,
    );

    let expected = ((poly1.clone() * poly2.clone()) + poly3.clone()) * ((poly1 * poly2) + poly3);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_enable_subcircuits_in_disk_is_noop_for_in_memory_storage() {
    let params = DCRTPolyParams::default();
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);

    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(2).to_vec();
    let add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
    sub_circuit.output(vec![add_gate]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    main_circuit.enable_subcircuits_in_disk("unused");
    let main_inputs = main_circuit.input(2).to_vec();
    let sub_circuit_id = main_circuit.register_sub_circuit(sub_circuit);
    let outputs = main_circuit.call_sub_circuit(sub_circuit_id, &[main_inputs[0], main_inputs[1]]);
    main_circuit.output(outputs);

    assert!(
        main_circuit
            .sub_circuits
            .values()
            .all(|sub| Arc::ptr_eq(&sub.lookup_registry, &main_circuit.lookup_registry))
    );

    let result = eval_with_const_one(
        &main_circuit,
        &params,
        &[poly1.clone(), poly2.clone()],
        None::<&PolyPltEvaluator>,
    );
    assert_eq!(result[0], poly1 + poly2);
}

#[test]
fn test_nested_sub_circuits() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create input polynomials
    let poly1 = create_random_poly(&params);
    let poly2 = create_random_poly(&params);
    let poly3 = create_random_poly(&params);

    // Create the innermost sub-circuit that performs multiplication
    let mut inner_circuit = PolyCircuit::new();
    let inner_inputs = inner_circuit.input(2).to_vec();
    let mul_gate = inner_circuit.mul_gate(inner_inputs[0], inner_inputs[1]);
    inner_circuit.output(vec![mul_gate]);

    // Create a middle sub-circuit that uses the inner sub-circuit
    let mut middle_circuit = PolyCircuit::new();
    let middle_inputs = middle_circuit.input(3).to_vec();

    // Register the inner circuit
    let inner_circuit_id = middle_circuit.register_sub_circuit(inner_circuit);

    // Call the inner circuit with the first two inputs
    let inner_outputs =
        middle_circuit.call_sub_circuit(inner_circuit_id, &[middle_inputs[0], middle_inputs[1]]);

    // Add the result of the inner circuit with the third input
    let add_gate = middle_circuit.add_gate(inner_outputs[0], middle_inputs[2]);
    middle_circuit.output(vec![add_gate]);

    // Create the main circuit
    let mut main_circuit = PolyCircuit::new();
    let main_inputs = main_circuit.input(3).to_vec();

    // Register the middle circuit
    let middle_circuit_id = main_circuit.register_sub_circuit(middle_circuit);

    // Call the middle circuit with all inputs
    let middle_outputs = main_circuit
        .call_sub_circuit(middle_circuit_id, &[main_inputs[0], main_inputs[1], main_inputs[2]]);

    let scalar_mul_gate = main_circuit.mul_gate(middle_outputs[0], middle_outputs[0]);

    // Set the output of the main circuit
    main_circuit.output(vec![scalar_mul_gate]);

    // Evaluate the main circuit
    let result = eval_with_const_one(
        &main_circuit,
        &params,
        &[poly1.clone(), poly2.clone(), poly3.clone()],
        None::<&PolyPltEvaluator>,
    );

    // Expected result: ((poly1 * poly2) + poly3)^2
    let expected = ((poly1.clone() * poly2.clone()) + poly3.clone()) * ((poly1 * poly2) + poly3);

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_const_zero_gate() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create a circuit with a const_zero_gate
    let mut circuit = PolyCircuit::new();
    // We need to call input() to initialize the circuit
    circuit.input(1);
    let zero_gate = circuit.const_zero_gate();
    circuit.output(vec![zero_gate]);

    // Evaluate the circuit with any input (it won't be used)
    let dummy_input = create_random_poly(&params);
    let result = eval_with_const_one(&circuit, &params, &[dummy_input], None::<&PolyPltEvaluator>);

    // Expected result: 0
    let expected = DCRTPoly::const_zero(&params);

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_const_one_gate() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create a circuit with a const_one_gate
    let mut circuit = PolyCircuit::new();
    // We need to call input() to initialize the circuit
    circuit.input(1);
    let one_gate = circuit.const_one_gate();
    circuit.output(vec![one_gate]);

    // Evaluate the circuit with any input (it won't be used)
    let dummy_input = create_random_poly(&params);
    let result = eval_with_const_one(&circuit, &params, &[dummy_input], None::<&PolyPltEvaluator>);

    // Expected result: 1
    let expected = DCRTPoly::const_one(&params);

    // Verify the result
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_const_minus_one_gate() {
    // Create parameters for testing
    let params = DCRTPolyParams::default();

    // Create a circuit with a const_minus_one_gate
    let mut circuit = PolyCircuit::new();
    // We need to call input() to initialize the circuit
    circuit.input(1);
    let minus_one_gate = circuit.const_minus_one_gate();
    circuit.output(vec![minus_one_gate]);

    // Evaluate the circuit with any input (it won't be used)
    let dummy_input = create_random_poly(&params);
    let result = eval_with_const_one(&circuit, &params, &[dummy_input], None::<&PolyPltEvaluator>);

    // Expected result: -1
    // We can compute -1 as 0 - 1
    let expected = DCRTPoly::const_zero(&params) - DCRTPoly::const_one(&params);

    // verify
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected);
}

#[test]
fn test_depth_zero_with_direct_input_output() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    circuit.output(vec![inputs[0]]);
    assert_eq!(circuit.depth(), 0);
}

#[test]
fn test_depth_one_with_add() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(2).to_vec();
    let add = circuit.add_gate(inputs[0], inputs[1]);
    circuit.output(vec![add]);
    assert_eq!(circuit.depth(), 1);
}

#[test]
fn test_depth_two_with_chain() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(3).to_vec();
    let add = circuit.add_gate(inputs[0], inputs[1]);
    let mul = circuit.mul_gate(add, inputs[2]);
    circuit.output(vec![mul]);
    assert_eq!(circuit.depth(), 2);
}

#[test]
fn test_non_free_depth_counts_sub_circuit() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(1).to_vec();
    let mul1 = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
    let mul2 = sub_circuit.mul_gate(mul1, sub_inputs[0]);
    sub_circuit.output(vec![mul2]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let main_inputs = main_circuit.input(1).to_vec();
    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let sub_outputs = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0]]);
    main_circuit.output(vec![sub_outputs[0]]);

    assert_eq!(main_circuit.non_free_depth(), 2);
}

#[test]
fn test_non_free_depth_respects_sub_circuit_inputs() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(2).to_vec();
    sub_circuit.output(vec![sub_inputs[0]]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let main_inputs = main_circuit.input(3).to_vec();
    let mul1 = main_circuit.mul_gate(main_inputs[1], main_inputs[2]);
    let mul2 = main_circuit.mul_gate(mul1, main_inputs[1]);

    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let sub_outputs = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0].into(), mul2]);
    main_circuit.output(vec![sub_outputs[0]]);

    assert_eq!(main_circuit.non_free_depth(), 0);
}

#[test]
fn test_non_free_depth_counts_slot_transfer_as_non_free() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1).to_vec();
    let transferred = circuit.slot_transfer_gate(inputs[0], &[(0, None)]);
    circuit.output(vec![transferred]);

    assert_eq!(circuit.non_free_depth(), 1);
}

#[test]
fn test_non_free_depth_ignores_add_chains() {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(4).to_vec();
    let add1 = circuit.add_gate(inputs[0], inputs[1]);
    let add2 = circuit.add_gate(add1, inputs[2]);
    let mul = circuit.mul_gate(add2, inputs[3]);
    circuit.output(vec![mul]);

    assert_eq!(circuit.non_free_depth(), 1);
}

#[test]
fn test_non_free_depth_handles_multi_output_sub_circuit_call() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(1).to_vec();
    let add = sub_circuit.add_gate(sub_inputs[0], sub_inputs[0]);
    let mul = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
    sub_circuit.output(vec![add, mul]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let main_inputs = main_circuit.input(1).to_vec();
    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let sub_outputs = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0]]);
    let sum = main_circuit.add_gate(sub_outputs[0], sub_outputs[1]);
    main_circuit.output(vec![sum]);

    assert_eq!(main_circuit.non_free_depth(), 1);
}

#[test]
fn test_non_free_depth_handles_repeated_sub_circuit_calls_with_different_input_levels() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(2).to_vec();
    let mul = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
    sub_circuit.output(vec![mul]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let main_inputs = main_circuit.input(3).to_vec();
    let precomputed = main_circuit.mul_gate(main_inputs[0], main_inputs[1]);

    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let direct_call = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0], main_inputs[2]]);
    let nested_call = main_circuit.call_sub_circuit(sub_id, &[precomputed, main_inputs[2].into()]);
    let output = main_circuit.add_gate(direct_call[0], nested_call[0]);
    main_circuit.output(vec![output]);

    assert_eq!(main_circuit.non_free_depth(), 2);
}

#[test]
fn test_non_free_depth_batches_multiple_ready_sub_circuit_calls() {
    let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
    let sub_inputs = sub_circuit.input(2).to_vec();
    let mul = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
    sub_circuit.output(vec![mul]);

    let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
    let main_inputs = main_circuit.input(4).to_vec();
    let precomputed = main_circuit.mul_gate(main_inputs[0], main_inputs[1]);

    let sub_id = main_circuit.register_sub_circuit(sub_circuit);
    let call1 = main_circuit.call_sub_circuit(sub_id, &[precomputed, main_inputs[2].into()]);
    let call2 = main_circuit.call_sub_circuit(sub_id, &[precomputed, main_inputs[3].into()]);
    let output = main_circuit.add_gate(call1[0], call2[0]);
    main_circuit.output(vec![output]);

    assert_eq!(main_circuit.non_free_depth(), 2);
}
