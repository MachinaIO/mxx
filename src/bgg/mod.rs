pub mod digits_to_int;
pub mod encoding;
pub mod poly_encoding;
pub mod public_key;
pub mod sampler;

#[cfg(test)]
mod tests {
    use crate::{
        __PAIR, __TestState,
        bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
        circuit::PolyCircuit,
        lookup::lwe_eval::LWEBGGEncodingPltEvaluator,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{hash::DCRTPolyHashSampler, uniform::DCRTPolyUniformSampler},
        utils::{create_random_poly, create_ternary_random_poly, random_bgg_encodings},
    };
    use keccak_asm::Keccak256;

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_pub_key_addition() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = 2;
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let log_base_q = params.modulus_digits();
        let columns = d * log_base_q;

        for pair in sampled_pub_keys[1..].chunks(2) {
            if let [a, b] = pair {
                let addition = a.clone() + b.clone();
                assert_eq!(addition.matrix.row_size(), d);
                assert_eq!(addition.matrix.col_size(), columns);
                assert_eq!(addition.matrix, a.matrix.clone() + b.matrix.clone());
            } else {
                panic!("pair should have two public keys");
            }
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_pub_key_multiplication() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = 2;
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let log_base_q = params.modulus_digits();
        let columns = d * log_base_q;

        for pair in sampled_pub_keys[1..].chunks(2) {
            if let [a, b] = pair {
                let multiplication = a.clone() * b.clone();
                assert_eq!(multiplication.matrix.row_size(), d);
                assert_eq!(multiplication.matrix.col_size(), columns);
                assert_eq!(multiplication.matrix, (&a.matrix).mul_decompose(&b.matrix))
            } else {
                panic!("pair should have two public keys");
            }
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_encoding_sampling() {
        let input_size = 10_usize;
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = input_size.div_ceil(params.ring_dimension().try_into().unwrap());
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let secrets = vec![create_ternary_random_poly(&params); d];
        let plaintexts = vec![DCRTPoly::const_one(&params); packed_input_size];
        let bgg_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let bgg_encodings = bgg_sampler.sample(&params, &sampled_pub_keys, &plaintexts);
        let g = DCRTPolyMatrix::gadget_matrix(&params, d);
        assert_eq!(bgg_encodings.len(), packed_input_size + 1);
        assert_eq!(
            bgg_encodings[0].vector,
            bgg_sampler.secret_vec.clone() * bgg_encodings[0].pubkey.matrix.clone() -
                bgg_sampler.secret_vec.clone() *
                    (g.clone() * bgg_encodings[0].plaintext.clone().unwrap())
        );
        assert_eq!(
            bgg_encodings[1].vector,
            bgg_sampler.secret_vec.clone() * bgg_encodings[1].pubkey.matrix.clone() -
                bgg_sampler.secret_vec.clone() *
                    (g * bgg_encodings[1].plaintext.clone().unwrap())
        )
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_encoding_addition() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = 2;
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let secrets = vec![create_ternary_random_poly(&params); d];
        let plaintexts = vec![create_random_poly(&params); packed_input_size];
        // TODO: set the standard deviation to a non-zero value
        let bgg_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let bgg_encodings = bgg_sampler.sample(&params, &sampled_pub_keys, &plaintexts);

        for pair in bgg_encodings[1..].chunks(2) {
            if let [a, b] = pair {
                let addition = a.clone() + b.clone();
                assert_eq!(addition.pubkey, a.pubkey.clone() + b.pubkey.clone());
                assert_eq!(
                    addition.clone().plaintext.unwrap(),
                    a.plaintext.clone().unwrap() + b.plaintext.clone().unwrap()
                );
                let g = DCRTPolyMatrix::gadget_matrix(&params, d);
                assert_eq!(addition.vector, a.clone().vector + b.clone().vector);
                assert_eq!(
                    addition.vector,
                    bgg_sampler.secret_vec.clone() *
                        (addition.pubkey.matrix - (g * addition.plaintext.unwrap()))
                )
            } else {
                panic!("pair should have two encodings");
            }
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_encoding_multiplication() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = 2;
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let secrets = vec![create_ternary_random_poly(&params); d];
        let plaintexts = vec![create_random_poly(&params); packed_input_size];
        let bgg_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let bgg_encodings = bgg_sampler.sample(&params, &sampled_pub_keys, &plaintexts);

        for pair in bgg_encodings[1..].chunks(2) {
            if let [a, b] = pair {
                let multiplication = a.clone() * b.clone();
                assert_eq!(multiplication.pubkey, (a.clone().pubkey * b.clone().pubkey));
                assert_eq!(
                    multiplication.clone().plaintext.unwrap(),
                    a.clone().plaintext.unwrap() * b.clone().plaintext.unwrap()
                );
                let g = DCRTPolyMatrix::gadget_matrix(&params, d);
                assert_eq!(
                    multiplication.vector,
                    (bgg_sampler.secret_vec.clone() *
                        (multiplication.pubkey.matrix - (g * multiplication.plaintext.unwrap())))
                )
            } else {
                panic!("pair should have two encodings");
            }
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_evaluable_bgg_add() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2).to_vec();
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);
        circuit.output(vec![add_gate]);

        let eval_inputs = vec![enc1.clone(), enc2.clone()];
        let result = circuit.eval(
            &params,
            enc_one,
            eval_inputs,
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
            None,
            None,
        );

        let expected = enc1.clone() + enc2.clone();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[sequential_test::sequential]
    #[test]
    fn test_evaluable_bgg_sub() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2).to_vec();
        let sub_gate = circuit.sub_gate(inputs[0], inputs[1]);
        circuit.output(vec![sub_gate]);

        let eval_inputs = vec![enc1.clone(), enc2.clone()];
        let result = circuit.eval(
            &params,
            enc_one,
            eval_inputs,
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
            None,
            None,
        );

        let expected = enc1.clone() - enc2.clone();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[sequential_test::sequential]
    #[test]
    fn test_evaluable_bgg_mul() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2).to_vec();
        let mul_gate = circuit.mul_gate(inputs[0], inputs[1]);
        circuit.output(vec![mul_gate]);

        let eval_inputs = vec![enc1.clone(), enc2.clone()];
        let result = circuit.eval(
            &params,
            enc_one,
            eval_inputs,
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
            None,
            None,
        );

        let expected = enc1.clone() * enc2.clone();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[sequential_test::sequential]
    #[test]
    fn test_evaluable_bgg_circuit_operations() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 3;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();
        let enc3 = encodings[3].clone();

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(3).to_vec();
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);
        let square_gate = circuit.mul_gate(add_gate, add_gate);
        let sub_gate = circuit.sub_gate(square_gate, inputs[2]);
        circuit.output(vec![sub_gate]);

        let eval_inputs = vec![enc1.clone(), enc2.clone(), enc3.clone()];
        let result = circuit.eval(
            &params,
            enc_one,
            eval_inputs,
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
            None,
            None,
        );

        let expected =
            ((enc1.clone() + enc2.clone()) * (enc1.clone() + enc2.clone())) - enc3.clone();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[sequential_test::sequential]
    #[test]
    fn test_evaluable_bgg_complex_circuit() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 4;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();
        let enc3 = encodings[3].clone();
        let enc4 = encodings[4].clone();

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(4).to_vec();
        let a = circuit.add_gate(inputs[0], inputs[1]);
        let b = circuit.mul_gate(inputs[2], inputs[3]);
        let c = circuit.mul_gate(a, b);
        let d = circuit.sub_gate(inputs[0], inputs[2]);
        let e = circuit.add_gate(c, d);
        let f = circuit.mul_gate(e, e);
        circuit.output(vec![f]);

        let eval_inputs = vec![enc1.clone(), enc2.clone(), enc3.clone(), enc4.clone()];
        let result = circuit.eval(
            &params,
            enc_one,
            eval_inputs,
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
            None,
            None,
        );

        let sum1 = enc1.clone() + enc2.clone();
        let prod1 = enc3.clone() * enc4.clone();
        let prod2 = sum1.clone() * prod1;
        let diff = enc1.clone() - enc3.clone();
        let sum2 = prod2 + diff;
        let expected = sum2.clone() * sum2;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[sequential_test::sequential]
    #[test]
    fn test_evaluable_bgg_multiple_input_calls_with_nonconsecutive_gate_ids() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        let mut circuit = PolyCircuit::new();
        let first_inputs = circuit.input(1).to_vec();
        assert_eq!(first_inputs.len(), 1);
        let _const_gate = circuit.const_digits(&[1u32, 0u32, 1u32]);
        let second_inputs = circuit.input(1).to_vec();
        assert_eq!(second_inputs.len(), 1);
        assert_ne!(second_inputs[0].0, first_inputs[0].0 + 1);

        let add_gate = circuit.add_gate(first_inputs[0], second_inputs[0]);
        circuit.output(vec![add_gate]);

        let eval_inputs = vec![enc1.clone(), enc2.clone()];
        let result = circuit.eval(
            &params,
            enc_one,
            eval_inputs,
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
            None,
            None,
        );

        let expected = enc1.clone() + enc2.clone();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[sequential_test::sequential]
    #[test]
    fn test_evaluable_bgg_register_and_call_sub_circuit() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 2;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();

        let mut sub_circuit = PolyCircuit::new();
        let sub_inputs = sub_circuit.input(2).to_vec();
        let add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        let mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![add_gate, mul_gate]);

        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(2).to_vec();
        let sub_circuit_id = main_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs =
            main_circuit.call_sub_circuit(sub_circuit_id, &[main_inputs[0], main_inputs[1]]);
        assert_eq!(sub_outputs.len(), 2);
        let final_gate = main_circuit.sub_gate(sub_outputs[0], sub_outputs[1]);
        main_circuit.output(vec![final_gate]);

        let eval_inputs = vec![enc1.clone(), enc2.clone()];
        let result = main_circuit.eval(
            &params,
            enc_one,
            eval_inputs,
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
            None,
            None,
        );

        let expected = (enc1.clone() + enc2.clone()) - (enc1.clone() * enc2.clone());

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }

    #[sequential_test::sequential]
    #[test]
    fn test_evaluable_bgg_nested_sub_circuits() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let input_size = 3;
        let encodings = random_bgg_encodings(input_size, d, &params);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();
        let enc2 = encodings[2].clone();
        let enc3 = encodings[3].clone();

        let mut inner_circuit = PolyCircuit::new();
        let inner_inputs = inner_circuit.input(2).to_vec();
        let mul_gate = inner_circuit.mul_gate(inner_inputs[0], inner_inputs[1]);
        inner_circuit.output(vec![mul_gate]);

        let mut middle_circuit = PolyCircuit::new();
        let middle_inputs = middle_circuit.input(3).to_vec();
        let inner_circuit_id = middle_circuit.register_sub_circuit(inner_circuit);
        let inner_outputs = middle_circuit
            .call_sub_circuit(inner_circuit_id, &[middle_inputs[0], middle_inputs[1]]);
        let add_gate = middle_circuit.add_gate(inner_outputs[0], middle_inputs[2]);
        middle_circuit.output(vec![add_gate]);

        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(3).to_vec();
        let middle_circuit_id = main_circuit.register_sub_circuit(middle_circuit);
        let middle_outputs = main_circuit
            .call_sub_circuit(middle_circuit_id, &[main_inputs[0], main_inputs[1], main_inputs[2]]);
        let scalar_mul_gate = main_circuit.mul_gate(middle_outputs[0], middle_outputs[0]);
        main_circuit.output(vec![scalar_mul_gate]);

        let eval_inputs = vec![enc1.clone(), enc2.clone(), enc3.clone()];
        let result = main_circuit.eval(
            &params,
            enc_one,
            eval_inputs,
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
            None,
            None,
        );

        let expected = ((enc1.clone() * enc2.clone()) + enc3.clone()) *
            ((enc1.clone() * enc2.clone()) + enc3.clone());

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, expected.vector);
        assert_eq!(result[0].pubkey.matrix, expected.pubkey.matrix);
        assert_eq!(result[0].plaintext.as_ref().unwrap(), expected.plaintext.as_ref().unwrap());
    }
}
