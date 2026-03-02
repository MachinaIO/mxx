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
            AGR16PublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, 1);
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

    #[test]
    fn test_agr16_sampling_satisfies_equation_5_1_without_error() {
        let params = DCRTPolyParams::default();
        let input_size = 3;
        let (pubkeys, encodings, plaintexts, secret) = sample_fixture(input_size, &params);

        let secret_matrix = scalar_matrix(&params, secret);

        // Slot 0 is the constant-1 encoding.
        let all_plaintexts = [&[DCRTPoly::const_one(&params)], plaintexts.as_slice()].concat();
        for idx in 0..encodings.len() {
            let expected = (secret_matrix.clone() * pubkeys[idx].matrix.clone()) +
                scalar_matrix(&params, all_plaintexts[idx].clone());
            assert_eq!(
                encodings[idx].vector, expected,
                "AGR16 base encoding must satisfy Equation 5.1 with zero injected error"
            );
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

        assert_eq!(enc_out.pubkey.matrix, pk_out.matrix);

        let expected_ct = (scalar_matrix(&params, secret) * pk_out.matrix.clone()) +
            scalar_matrix(&params, expected_plain.clone());
        assert_eq!(
            enc_out.vector, expected_ct,
            "Evaluated AGR16 ciphertext must satisfy Equation 5.1 when error=0"
        );
        assert_eq!(enc_out.plaintext, Some(expected_plain));
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

        assert_eq!(enc_out.pubkey.matrix, pk_out.matrix);

        let expected_ct = (scalar_matrix(&params, secret) * pk_out.matrix.clone()) +
            scalar_matrix(&params, expected_plain.clone());
        assert_eq!(
            enc_out.vector, expected_ct,
            "Nested AGR16 multiplication output must satisfy Equation 5.1 when error=0"
        );
        assert_eq!(enc_out.plaintext, Some(expected_plain));
    }

    #[test]
    #[should_panic(expected = "AGR16EncodingSampler::new requires at least one secret polynomial")]
    fn test_agr16_sampler_rejects_empty_secret_input() {
        let params = DCRTPolyParams::default();
        let empty_secrets: Vec<DCRTPoly> = Vec::new();
        let _ = AGR16EncodingSampler::<DCRTPolyUniformSampler>::new(&params, &empty_secrets, None);
    }
}
