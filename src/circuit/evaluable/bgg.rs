use crate::{
    bgg::{encoding::BGGEncoding, public_key::BGGPublicKey},
    circuit::evaluable::Evaluable,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};

impl<M: PolyMatrix> Evaluable for BGGEncoding<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self {
        let pubkey = self.pubkey.rotate(params, shift);
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.unsigned_abs() as usize
        };
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        let vector = self.vector.clone() * &rotate_poly;
        let plaintext = self.plaintext.clone().map(|plaintext| plaintext * rotate_poly);
        Self { vector, pubkey, plaintext }
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        let scalar = Self::P::from_u32s(params, scalar);
        let vector = self.vector.clone() * &scalar;
        let pubkey_matrix = self.pubkey.matrix.clone() * &scalar;
        let pubkey = BGGPublicKey::new(pubkey_matrix, self.pubkey.reveal_plaintext);
        let plaintext = self.plaintext.clone().map(|p| p * scalar);
        Self { vector, pubkey, plaintext }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[num_bigint::BigUint]) -> Self {
        let scalar = Self::P::from_biguints(params, scalar);
        let row_size = self.pubkey.matrix.row_size();
        let scalar_gadget = M::gadget_matrix(params, row_size) * &scalar;
        let decomposed = scalar_gadget.decompose();
        let vector = self.vector.clone() * &decomposed;
        let pubkey_matrix = self.pubkey.matrix.clone() * &decomposed;
        let pubkey = BGGPublicKey::new(pubkey_matrix, self.pubkey.reveal_plaintext);
        let plaintext = self.plaintext.clone().map(|p| p * scalar);
        Self { vector, pubkey, plaintext }
    }
}

impl<M: PolyMatrix> Evaluable for BGGPublicKey<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self {
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.unsigned_abs() as usize
        };
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        let matrix = self.matrix.clone() * rotate_poly;
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        let scalar = Self::P::from_u32s(params, scalar);
        let matrix = self.matrix.clone() * scalar;
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[num_bigint::BigUint]) -> Self {
        let scalar = Self::P::from_biguints(params, scalar);
        let row_size = self.matrix.row_size();
        let scalar_gadget = M::gadget_matrix(params, row_size) * scalar;
        let decomposed = scalar_gadget.decompose();
        let matrix = self.matrix.clone() * decomposed;
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circuit::PolyCircuit, lookup::lwe_eval::LweBGGEncodingPltEvaluator,
        matrix::dcrt_poly::DCRTPolyMatrix, poly::dcrt::params::DCRTPolyParams,
        sampler::hash::DCRTPolyHashSampler, utils::random_bgg_encodings,
    };
    use keccak_asm::Keccak256;

    #[test]
    fn test_encoding_add() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        // Create a simple circuit with an Add operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);
        circuit.output(vec![add_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &enc_one.clone(),
            &[enc1.clone(), enc2.clone()],
            None::<LweBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );

        // Expected result
        let expected = enc1.clone() + enc2.clone();

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[test]
    fn test_encoding_sub() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        // Create a simple circuit with a Sub operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let sub_gate = circuit.sub_gate(inputs[0], inputs[1]);
        circuit.output(vec![sub_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &enc_one,
            &[enc1.clone(), enc2.clone()],
            None::<LweBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );

        // Expected result
        let expected = enc1.clone() - enc2.clone();

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[test]
    fn test_encoding_mul() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        // Create a simple circuit with a Mul operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let mul_gate = circuit.mul_gate(inputs[0], inputs[1]);
        circuit.output(vec![mul_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &enc_one,
            &[enc1.clone(), enc2.clone()],
            None::<LweBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );

        // Expected result
        let expected = enc1.clone() * enc2.clone();

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[test]
    fn test_encoding_circuit_operations() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 3;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();
        let enc3 = encodings[3].clone();

        // Create a circuit: ((enc1 + enc2)^2) - enc3
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(3);

        // enc1 + enc2
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);

        // (enc1 + enc2)^2
        let square_gate = circuit.mul_gate(add_gate, add_gate);

        // ((enc1 + enc2)^2) - enc3
        let sub_gate = circuit.sub_gate(square_gate, inputs[2]);

        circuit.output(vec![sub_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &enc_one,
            &[enc1.clone(), enc2.clone(), enc3.clone()],
            None::<LweBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );

        // Expected result: ((enc1 + enc2)^2) - enc3
        let expected =
            ((enc1.clone() + enc2.clone()) * (enc1.clone() + enc2.clone())) - enc3.clone();

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[test]
    fn test_encoding_complex_circuit() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 4;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();
        let enc3 = encodings[3].clone();
        let enc4 = encodings[4].clone();

        // Create a complex circuit with depth = 4
        // Circuit structure:
        // Level 1: a = enc1 + enc2, b = enc3 * enc4
        // Level 2: c = a * b, d = enc1 - enc3
        // Level 3: e = c + d
        // Level 4: f = e * e
        // Output: f
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(4);

        // Level 1
        let a = circuit.add_gate(inputs[0], inputs[1]); // enc1 + enc2
        let b = circuit.mul_gate(inputs[2], inputs[3]); // enc3 * enc4

        // Level 2
        let c = circuit.mul_gate(a, b); // (enc1 + enc2) * (enc3 * enc4)
        let d = circuit.sub_gate(inputs[0], inputs[2]); // enc1 - enc3

        // Level 3
        let e = circuit.add_gate(c, d); // ((enc1 + enc2) * (enc3 * enc4)) + (enc1 - enc3)

        // Level 4
        let f = circuit.mul_gate(e, e); // (((enc1 + enc2) * (enc3 * enc4)) + (enc1 - enc3))^2

        circuit.output(vec![f]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &enc_one,
            &[enc1.clone(), enc2.clone(), enc3.clone(), enc4.clone()],
            None::<LweBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );

        // Expected result: (((enc1 + enc2) * (enc3 * enc4)) + (enc1 - enc3)) * scalar
        let sum1 = enc1.clone() + enc2.clone();
        let prod1 = enc3.clone() * enc4.clone();
        let prod2 = sum1.clone() * prod1;
        let diff = enc1.clone() - enc3.clone();
        let sum2 = prod2 + diff;
        let expected = sum2.clone() * sum2;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[test]
    fn test_encoding_multiple_input_calls_with_nonconsecutive_gate_ids() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        // encodings[0] is the constant-one encoding used as `one` in eval
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        // Create a circuit and call input() twice with a gate in between
        let mut circuit = PolyCircuit::new();
        let first_inputs = circuit.input(1);
        assert_eq!(first_inputs.len(), 1);

        // Insert a const gate between input() calls to force a gap in GateIds
        let _const_gate = circuit.const_digits(&[1u32, 0u32, 1u32]);

        // Second input call: next input gate id should not be consecutive to the first
        let second_inputs = circuit.input(1);
        assert_eq!(second_inputs.len(), 1);
        assert_ne!(second_inputs[0].0, first_inputs[0].0 + 1);

        // Build a simple circuit that adds the two inputs together
        let add_gate = circuit.add_gate(first_inputs[0], second_inputs[0]);
        circuit.output(vec![add_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &enc_one,
            &[enc1.clone(), enc2.clone()],
            None::<LweBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );

        // Expected: enc1 + enc2
        let expected = enc1 + enc2;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[test]
    fn test_encoding_register_and_call_sub_circuit() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        // Create a sub-circuit that performs addition and multiplication
        let mut sub_circuit = PolyCircuit::new();
        let sub_inputs = sub_circuit.input(2);

        // Add operation: enc1 + enc2
        let add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);

        // Mul operation: enc1 * enc2
        let mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);

        // Set the outputs of the sub-circuit
        sub_circuit.output(vec![add_gate, mul_gate]);

        // Create the main circuit
        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(2);

        // Register the sub-circuit and get its ID
        let sub_circuit_id = main_circuit.register_sub_circuit(sub_circuit);

        // Call the sub-circuit with the main circuit's inputs
        let sub_outputs =
            main_circuit.call_sub_circuit(sub_circuit_id, &[main_inputs[0], main_inputs[1]]);

        // Verify we got two outputs from the sub-circuit
        assert_eq!(sub_outputs.len(), 2);

        // Use the sub-circuit outputs for further computation
        // For example, subtract the multiplication result from the addition result
        let final_gate = main_circuit.sub_gate(sub_outputs[0], sub_outputs[1]);

        // Set the output of the main circuit
        main_circuit.output(vec![final_gate]);

        // Evaluate the main circuit
        let result = main_circuit.eval(
            &params,
            &enc_one,
            &[enc1.clone(), enc2.clone()],
            None::<LweBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );

        // Expected result: (enc1 + enc2) - (enc1 * enc2)
        let expected = (enc1.clone() + enc2.clone()) - (enc1.clone() * enc2.clone());

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[test]
    fn test_encoding_nested_sub_circuits() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 3;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();
        let enc3 = encodings[3].clone();

        // Create the innermost sub-circuit that performs multiplication
        let mut inner_circuit = PolyCircuit::new();
        let inner_inputs = inner_circuit.input(2);
        let mul_gate = inner_circuit.mul_gate(inner_inputs[0], inner_inputs[1]);
        inner_circuit.output(vec![mul_gate]);

        // Create a middle sub-circuit that uses the inner sub-circuit
        let mut middle_circuit = PolyCircuit::new();
        let middle_inputs = middle_circuit.input(3);

        // Register the inner circuit
        let inner_circuit_id = middle_circuit.register_sub_circuit(inner_circuit);

        // Call the inner circuit with the first two inputs
        let inner_outputs = middle_circuit
            .call_sub_circuit(inner_circuit_id, &[middle_inputs[0], middle_inputs[1]]);

        // Add the result of the inner circuit with the third input
        let add_gate = middle_circuit.add_gate(inner_outputs[0], middle_inputs[2]);
        middle_circuit.output(vec![add_gate]);

        // Create the main circuit
        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(3);

        // Register the middle circuit
        let middle_circuit_id = main_circuit.register_sub_circuit(middle_circuit);

        // Call the middle circuit with all inputs
        let middle_outputs = main_circuit
            .call_sub_circuit(middle_circuit_id, &[main_inputs[0], main_inputs[1], main_inputs[2]]);

        // Use the output for square
        let scalar_mul_gate = main_circuit.mul_gate(middle_outputs[0], middle_outputs[0]);

        // Set the output of the main circuit
        main_circuit.output(vec![scalar_mul_gate]);

        // Evaluate the main circuit
        let result = main_circuit.eval(
            &params,
            &enc_one,
            &[enc1.clone(), enc2.clone(), enc3.clone()],
            None::<LweBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );

        // Expected result: ((enc1 * enc2) + enc3)^2
        let expected = ((enc1.clone() * enc2.clone()) + enc3.clone()) *
            ((enc1.clone() * enc2.clone()) + enc3.clone());

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }
}
