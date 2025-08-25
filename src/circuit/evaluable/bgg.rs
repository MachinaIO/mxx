use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::evaluable::Evaluable,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};

impl<M: PolyMatrix> Evaluable for BggEncoding<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;

    fn rotate(self, params: &Self::Params, shift: i32) -> Self {
        let pubkey = self.pubkey.rotate(params, shift);
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.abs() as usize
        };
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        let vector = self.vector.clone() * &rotate_poly;
        let plaintext = self.plaintext.clone().map(|plaintext| plaintext * rotate_poly);
        Self { vector, pubkey, plaintext }
    }

    fn from_digits(params: &Self::Params, one: &Self, digits: &[u32]) -> Self {
        let const_poly =
            <M::P as Evaluable>::from_digits(params, &<M::P>::const_one(params), digits);
        let vector = one.vector.clone() * &const_poly;
        let pubkey = BggPublicKey::from_digits(params, &one.pubkey, digits);
        let plaintext = one.plaintext.clone().map(|plaintext| plaintext * const_poly);
        Self { vector, pubkey, plaintext }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[num_bigint::BigUint]) -> Self {
        let scalar = Self::P::from_biguints(params, scalar);
        let row_size = self.pubkey.matrix.row_size();
        let scalar_gadget = M::gadget_matrix(params, row_size) * &scalar;
        let decomposed = scalar_gadget.decompose();
        let vector = self.vector.clone() * &decomposed;
        let pubkey_matrix = self.pubkey.matrix.clone() * &decomposed;
        let pubkey = BggPublicKey::new(pubkey_matrix, self.pubkey.reveal_plaintext);
        let plaintext = self.plaintext.clone().map(|p| p * scalar);
        Self { vector, pubkey, plaintext }
    }
}

impl<M: PolyMatrix> Evaluable for BggPublicKey<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;

    fn rotate(self, params: &Self::Params, shift: i32) -> Self {
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.abs() as usize
        };
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        let matrix = self.matrix.clone() * rotate_poly;
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }

    fn from_digits(params: &Self::Params, one: &Self, digits: &[u32]) -> Self {
        let const_poly =
            <M::P as Evaluable>::from_digits(params, &<M::P>::const_one(params), digits);
        let matrix = one.matrix.clone() * const_poly;
        Self { matrix, reveal_plaintext: one.reveal_plaintext }
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
        circuit::PolyCircuit, lookup::lwe_eval::LweBggEncodingPltEvaluator,
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
            None::<LweBggEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
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
            None::<LweBggEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
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
            None::<LweBggEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
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
            None::<LweBggEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
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
            None::<LweBggEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
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
        let _const_gate = circuit.const_digits_poly(&[1u32, 0u32, 1u32]);

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
            None::<LweBggEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
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
            None::<LweBggEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
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
            None::<LweBggEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
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

    // #[tokio::test]
    // #[ignore]
    // async fn test_encoding_plt_for_dio() {
    //     init_tracing();
    //     init_storage_system();

    //     let tmp_dir = tempdir().unwrap();
    //     let tmp_dir_path = tmp_dir.path().to_path_buf();

    //     /* Setup */
    //     let params = DCRTPolyParams::default();
    //     let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);

    //     /* Obfuscation Step */
    //     let key: [u8; 32] = rand::random();
    //     let d = 1;
    //     let uni = DCRTPolyUniformSampler::new();
    //     let bgg_pubkey_sampler =
    //         BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
    //     let uniform_sampler = DCRTPolyUniformSampler::new();
    //     let (b_l_plus_one_trapdoor, b_l_plus_one) = trapdoor_sampler.trapdoor(&params, d + 1);
    //     info!("b_l_plus_one ({},{})", b_l_plus_one.row_size(), b_l_plus_one.col_size());
    //     /* BGG+ encoding setup */
    //     let secrets = uni.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
    //     // in reality there should be input insertion step that updates the secret s_init to
    //     let s_x_l = {
    //         let minus_one_poly = DCRTPoly::const_minus_one(&params);
    //         let mut secrets = secrets.to_vec();
    //         secrets.push(minus_one_poly);
    //         DCRTPolyMatrix::from_poly_vec_row(&params, secrets)
    //     };
    //     info!("s_x_L ({},{})", s_x_l.row_size(), s_x_l.col_size());
    //     let p = s_x_l.clone() * &b_l_plus_one;
    //     let tag: u64 = rand::random();
    //     let tag_bytes = tag.to_le_bytes();
    //     let plt = setup_constant_plt(8, &params);

    //     // Create a simple circuit with an plt operation
    //     let mut circuit = PolyCircuit::new();
    //     {
    //         let inputs = circuit.input(1);
    //         let plt_id = circuit.register_public_lookup(plt.clone());
    //         let plt_gate = circuit.public_lookup_gate(inputs[0], plt_id);
    //         circuit.output(vec![plt_gate]);
    //     }

    //     // Create random public keys
    //     let reveal_plaintexts = [true, false];
    //     let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
    //     assert_eq!(pubkeys.len(), 2);
    //     let pubkey_plt_evaluator = BggPubKeyPltEvaluator::<
    //         DCRTPolyMatrix,
    //         DCRTPolyHashSampler<Keccak256>,
    //         DCRTPolyUniformSampler,
    //         DCRTPolyTrapdoorSampler,
    //     >::new(
    //         key,
    //         trapdoor_sampler,
    //         Arc::new(b_l_plus_one),
    //         Arc::new(b_l_plus_one_trapdoor),
    //         tmp_dir_path.clone(),
    //     );

    //     let expected_pubkey_output =
    //         &circuit.eval(&params, &pubkeys[0], &[pubkeys[1].clone()],
    // Some(pubkey_plt_evaluator))             [0];

    //     wait_for_all_writes().await.unwrap();

    //     // Create secret and plaintexts
    //     let k = 2;
    //     let plaintexts = vec![DCRTPoly::from_usize_to_constant(&params, k)];

    //     // Create encoding sampler and encodings
    //     let bgg_encoding_sampler = BGGEncodingSampler::new(&params, &secrets, uniform_sampler,
    // 0.0);     let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
    //     let enc_one = encodings[0].clone();
    //     let enc1 = encodings[1].clone();

    //     assert_eq!(*enc1.plaintext.as_ref().unwrap(), plaintexts[0]);

    //     // Evaluate the circuit
    //     let bgg_encoding_plt_evaluator = BggEncodingPltEvaluator::<
    //         DCRTPolyMatrix,
    //         DCRTPolyHashSampler<Keccak256>,
    //     >::new(key, tmp_dir_path.clone(), p);
    //     let result = circuit.eval(&params, &enc_one, &[enc1], Some(bgg_encoding_plt_evaluator));
    //     let (_, y_k) = plt.f[&plaintexts[0]].clone();
    //     let expected_encodings = bgg_encoding_sampler.sample(
    //         &params,
    //         &[pubkeys[0].clone(), expected_pubkey_output.clone()],
    //         &[y_k.clone()],
    //     );
    //     let expected_enc1 = expected_encodings[1].clone();

    //     // Verify the result
    //     assert_eq!(result.len(), 1);
    //     assert_eq!(result[0].vector, expected_enc1.vector);
    //     assert_eq!(result[0].pubkey.matrix, expected_enc1.pubkey.matrix);
    //     assert_eq!(*result[0].plaintext.as_ref().unwrap(), y_k.clone());
    // }
}
