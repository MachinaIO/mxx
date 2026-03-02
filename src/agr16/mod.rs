pub mod encoding;
pub mod public_key;
pub mod sampler;

#[cfg(test)]
mod tests {
    use crate::{
        agr16::{
            encoding::Agr16Encoding,
            public_key::Agr16PublicKey,
            sampler::{AGR16EncodingSampler, AGR16PublicKeySampler},
        },
        circuit::{PolyCircuit, gate::GateId},
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{hash::DCRTPolyHashSampler, uniform::DCRTPolyUniformSampler},
        utils::{create_random_poly, create_ternary_random_poly},
    };
    use keccak_asm::Keccak256;

    const AUXILIARY_DEPTH: usize = 8;

    struct NoopAgr16PkPlt;

    impl PltEvaluator<Agr16PublicKey<DCRTPolyMatrix>> for NoopAgr16PkPlt {
        fn public_lookup(
            &self,
            _params: &<Agr16PublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::Params,
            _plt: &PublicLut<
                <Agr16PublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::P,
            >,
            _one: &Agr16PublicKey<DCRTPolyMatrix>,
            _input: &Agr16PublicKey<DCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> Agr16PublicKey<DCRTPolyMatrix> {
            panic!("NoopAgr16PkPlt should not be called in these tests");
        }
    }

    struct NoopAgr16EncPlt;

    impl PltEvaluator<Agr16Encoding<DCRTPolyMatrix>> for NoopAgr16EncPlt {
        fn public_lookup(
            &self,
            _params: &<Agr16Encoding<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::Params,
            _plt: &PublicLut<
                <Agr16Encoding<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::P,
            >,
            _one: &Agr16Encoding<DCRTPolyMatrix>,
            _input: &Agr16Encoding<DCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> Agr16Encoding<DCRTPolyMatrix> {
            panic!("NoopAgr16EncPlt should not be called in these tests");
        }
    }

    fn sample_fixture(
        input_size: usize,
        params: &DCRTPolyParams,
    ) -> (
        Vec<Agr16PublicKey<DCRTPolyMatrix>>,
        Vec<Agr16Encoding<DCRTPolyMatrix>>,
        Vec<DCRTPoly>,
        DCRTPoly,
    ) {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();

        let pubkey_sampler =
            AGR16PublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, AUXILIARY_DEPTH);
        let reveal_plaintexts = vec![true; input_size];
        let pubkeys = pubkey_sampler.sample(params, &tag_bytes, &reveal_plaintexts);

        let secret = create_ternary_random_poly(params);
        let secrets = vec![secret.clone()];
        let plaintexts = (0..input_size).map(|_| create_random_poly(params)).collect::<Vec<_>>();
        let encoding_sampler =
            AGR16EncodingSampler::<DCRTPolyUniformSampler>::new(params, &secrets, None);
        let encodings = encoding_sampler.sample(params, &pubkeys, &plaintexts);

        (pubkeys, encodings, plaintexts, encoding_sampler.secret)
    }

    fn scalar_matrix(params: &DCRTPolyParams, value: DCRTPoly) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec_row(params, vec![value])
    }

    fn assert_primary_auxiliary_invariants(
        encoding: &Agr16Encoding<DCRTPolyMatrix>,
        secret: &DCRTPoly,
    ) {
        assert!(
            !encoding.pubkey.c_times_s_pubkeys.is_empty() &&
                !encoding.c_times_s_encodings.is_empty(),
            "AGR16 encoding must keep at least one recursive c_times_s level"
        );
        let expected_c_times_s = (encoding.pubkey.c_times_s_pubkeys[0].clone() * secret) +
            (encoding.vector.clone() * secret);
        assert_eq!(
            encoding.c_times_s_encodings[0], expected_c_times_s,
            "AGR16 c_times_s invariant must hold"
        );
    }

    fn assert_full_auxiliary_invariants(
        params: &DCRTPolyParams,
        encoding: &Agr16Encoding<DCRTPolyMatrix>,
        secret: &DCRTPoly,
    ) {
        let secret_matrix = scalar_matrix(params, secret.clone());
        assert_eq!(
            encoding.pubkey.c_times_s_pubkeys.len(),
            encoding.c_times_s_encodings.len(),
            "AGR16 c_times_s invariant depth mismatch between key and encoding"
        );
        assert_eq!(
            encoding.pubkey.s_power_pubkeys.len(),
            encoding.s_power_encodings.len(),
            "AGR16 s-power advice depth mismatch between key and encoding"
        );

        let mut current_c_level = encoding.vector.clone();
        for level in 0..encoding.c_times_s_encodings.len() {
            let expected = (encoding.pubkey.c_times_s_pubkeys[level].clone() * secret) +
                (current_c_level.clone() * secret);
            assert_eq!(
                encoding.c_times_s_encodings[level], expected,
                "AGR16 c_times_s recursive invariant must hold at level {level}"
            );
            current_c_level = encoding.c_times_s_encodings[level].clone();
        }

        let mut current_s_level = secret_matrix;
        for level in 0..encoding.s_power_encodings.len() {
            let expected = (encoding.pubkey.s_power_pubkeys[level].clone() * secret) +
                (current_s_level.clone() * secret);
            assert_eq!(
                encoding.s_power_encodings[level], expected,
                "AGR16 s-power recursive invariant must hold at level {level}"
            );
            current_s_level = encoding.s_power_encodings[level].clone();
        }
    }

    fn assert_eval_output_matches_equation_5_1(
        params: &DCRTPolyParams,
        secret: &DCRTPoly,
        pk_out: &Agr16PublicKey<DCRTPolyMatrix>,
        enc_out: &Agr16Encoding<DCRTPolyMatrix>,
        expected_plain: DCRTPoly,
        context: &str,
    ) {
        assert_eq!(enc_out.pubkey, *pk_out);
        let expected_ct = (scalar_matrix(params, secret.clone()) * pk_out.matrix.clone()) +
            scalar_matrix(params, expected_plain.clone());
        assert_eq!(enc_out.vector, expected_ct, "{context}");
        assert_primary_auxiliary_invariants(enc_out, secret);
        assert_eq!(enc_out.plaintext, Some(expected_plain));
    }

    #[test]
    fn test_agr16_sampling_satisfies_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let input_size = 3;
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(input_size, &params);

        let secret_matrix = scalar_matrix(&params, secret.clone());

        // Slot 0 is the constant-1 encoding.
        let all_plaintexts = [&[DCRTPoly::const_one(&params)], plaintexts.as_slice()].concat();
        for idx in 0..encodings.len() {
            let expected = (secret_matrix.clone() * pubkeys[idx].matrix.clone()) +
                scalar_matrix(&params, all_plaintexts[idx].clone());
            assert_eq!(
                encodings[idx].vector, expected,
                "AGR16 base encoding must satisfy Equation 5.1 with zero injected error"
            );
            assert_full_auxiliary_invariants(&params, &encodings[idx], &secret);
        }
    }

    #[test]
    fn test_agr16_circuit_eval_matches_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(3, &params);

        // f(x1,x2,x3) = (x1 + x2) * x3 + x1
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(3);
        let add = circuit.add_gate(inputs[0], inputs[1]);
        let mul = circuit.mul_gate(add, inputs[2]);
        let out = circuit.add_gate(mul, inputs[0]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![pubkeys[1].clone(), pubkeys[2].clone(), pubkeys[3].clone()],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![encodings[1].clone(), encodings[2].clone(), encodings[3].clone()],
            None::<&NoopAgr16EncPlt>,
        );

        let pk_out = &pk_outputs[0];
        let enc_out = &enc_outputs[0];
        let expected_plain = (plaintexts[0].clone() + plaintexts[1].clone()) *
            plaintexts[2].clone() +
            plaintexts[0].clone();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            pk_out,
            enc_out,
            expected_plain,
            "Evaluated AGR16 ciphertext must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_nested_multiplication_preserves_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(3, &params);

        // f(x1,x2,x3) = ((x1 * x2) + x3) * x2
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(3);
        let mul1 = circuit.mul_gate(inputs[0], inputs[1]);
        let add = circuit.add_gate(mul1, inputs[2]);
        let out = circuit.mul_gate(add, inputs[1]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![pubkeys[1].clone(), pubkeys[2].clone(), pubkeys[3].clone()],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![encodings[1].clone(), encodings[2].clone(), encodings[3].clone()],
            None::<&NoopAgr16EncPlt>,
        );

        let pk_out = &pk_outputs[0];
        let enc_out = &enc_outputs[0];
        let expected_plain = ((plaintexts[0].clone() * plaintexts[1].clone()) +
            plaintexts[2].clone()) *
            plaintexts[1].clone();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            pk_out,
            enc_out,
            expected_plain,
            "Nested AGR16 multiplication output must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_depth3_multiplication_preserves_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(4, &params);

        // f(x1,x2,x3,x4) = (((x1 * x2) * x3) * x4)
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(4);
        let mul1 = circuit.mul_gate(inputs[0], inputs[1]);
        let mul2 = circuit.mul_gate(mul1, inputs[2]);
        let out = circuit.mul_gate(mul2, inputs[3]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![pubkeys[1].clone(), pubkeys[2].clone(), pubkeys[3].clone(), pubkeys[4].clone()],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![
                encodings[1].clone(),
                encodings[2].clone(),
                encodings[3].clone(),
                encodings[4].clone(),
            ],
            None::<&NoopAgr16EncPlt>,
        );

        let expected_plain = ((plaintexts[0].clone() * plaintexts[1].clone()) *
            plaintexts[2].clone()) *
            plaintexts[3].clone();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            &pk_outputs[0],
            &enc_outputs[0],
            expected_plain,
            "Depth-3 AGR16 multiplication output must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    fn test_agr16_depth4_composed_circuit_preserves_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(8, &params);

        // f(x1..x8) = ((((x1 * x2) + x3) * (x4 * x5)) * (x6 + x7)) * x8
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(8);
        let mul12 = circuit.mul_gate(inputs[0], inputs[1]);
        let add123 = circuit.add_gate(mul12, inputs[2]);
        let mul45 = circuit.mul_gate(inputs[3], inputs[4]);
        let mul_left = circuit.mul_gate(add123, mul45);
        let add67 = circuit.add_gate(inputs[5], inputs[6]);
        let mul_deep = circuit.mul_gate(mul_left, add67);
        let out = circuit.mul_gate(mul_deep, inputs[7]);
        circuit.output(vec![out]);

        let pk_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![
                pubkeys[1].clone(),
                pubkeys[2].clone(),
                pubkeys[3].clone(),
                pubkeys[4].clone(),
                pubkeys[5].clone(),
                pubkeys[6].clone(),
                pubkeys[7].clone(),
                pubkeys[8].clone(),
            ],
            None::<&NoopAgr16PkPlt>,
        );
        let enc_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![
                encodings[1].clone(),
                encodings[2].clone(),
                encodings[3].clone(),
                encodings[4].clone(),
                encodings[5].clone(),
                encodings[6].clone(),
                encodings[7].clone(),
                encodings[8].clone(),
            ],
            None::<&NoopAgr16EncPlt>,
        );

        let expected_plain = ((((plaintexts[0].clone() * plaintexts[1].clone()) +
            plaintexts[2].clone()) *
            (plaintexts[3].clone() * plaintexts[4].clone())) *
            (plaintexts[5].clone() + plaintexts[6].clone())) *
            plaintexts[7].clone();
        assert_eval_output_matches_equation_5_1(
            &params,
            &secret,
            &pk_outputs[0],
            &enc_outputs[0],
            expected_plain,
            "Depth-4 AGR16 composed output must satisfy Equation 5.1 when error=0",
        );
    }

    #[test]
    #[should_panic(expected = "AGR16EncodingSampler::new requires at least one secret polynomial")]
    fn test_agr16_sampler_rejects_empty_secret_input() {
        let params = DCRTPolyParams::default();
        let empty_secrets: Vec<DCRTPoly> = Vec::new();
        let _ = AGR16EncodingSampler::<DCRTPolyUniformSampler>::new(&params, &empty_secrets, None);
    }
}
