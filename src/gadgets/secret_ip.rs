use crate::{
    circuit::{PolyCircuit, gate::GateId},
    poly::Poly,
};

pub fn secret_inner_product<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    public_vec: &[GateId],
    secret_vec: &[GateId],
) -> GateId {
    assert_eq!(public_vec.len(), secret_vec.len(), "vector lengths must match");
    if public_vec.is_empty() {
        return circuit.const_zero_gate();
    }

    // Multiply with public input on the left to keep BGG encoding semantics.
    let mut acc = circuit.mul_gate(public_vec[0], secret_vec[0]);
    for (&public_id, &secret_id) in public_vec.iter().zip(secret_vec.iter()).skip(1) {
        let prod = circuit.mul_gate(public_id, secret_id);
        acc = circuit.add_gate(acc, prod);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::secret_inner_product;
    use crate::{
        bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
        circuit::PolyCircuit,
        lookup::{
            lwe_eval::{LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator},
            poly::PolyPltEvaluator,
        },
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        rlwe_enc::rlwe_encrypt,
        sampler::{
            DistType, PolyUniformSampler, hash::DCRTPolyHashSampler,
            trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
        },
    };
    use keccak_asm::Keccak256;

    #[test]
    fn test_secret_ip_with_rlwe_vectors() {
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyUniformSampler::new();
        let sigma = 3.0;

        let a = sampler.sample_poly(&params, &DistType::BitDist);
        let t = sampler.sample_poly(&params, &DistType::BitDist);
        let m = sampler.sample_poly(&params, &DistType::BitDist);

        let a_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![a.clone()]);
        let t_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![t.clone()]);
        let m_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![m]);
        let b_mat = rlwe_encrypt(&params, &sampler, &t_mat, &a_mat, &m_mat, sigma);
        let b = b_mat.entry(0, 0);
        let neg_t = -&t;

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let public_inputs = circuit.input(2);
        let secret_inputs = circuit.input(2);
        let out = secret_inner_product(&mut circuit, &public_inputs, &secret_inputs);
        circuit.output(vec![out]);

        let poly_inputs = vec![a.clone(), b.clone(), neg_t.clone(), DCRTPoly::const_one(&params)];
        let poly_out = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &poly_inputs,
            None::<&PolyPltEvaluator>,
        );
        assert_eq!(poly_out.len(), 1);
        let expected_poly = b.clone() + (a.clone() * &neg_t);
        assert_eq!(poly_out[0], expected_poly);

        let seed: [u8; 32] = [0u8; 32];
        let d = 1usize;
        let reveal_plaintexts = vec![true, true, false, false];
        let pk_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(seed, d);
        let pubkeys = pk_sampler.sample(&params, b"SECRET_IP_TEST", &reveal_plaintexts);
        let pk_out = circuit.eval(
            &params,
            &pubkeys[0],
            &pubkeys[1..],
            None::<
                &LWEBGGPubKeyPltEvaluator<
                    DCRTPolyMatrix,
                    DCRTPolyHashSampler<Keccak256>,
                    DCRTPolyTrapdoorSampler,
                >,
            >,
        );
        assert_eq!(pk_out.len(), 1);
        let pk_expected =
            (pubkeys[1].clone() * pubkeys[3].clone()) + (pubkeys[2].clone() * pubkeys[4].clone());
        assert_eq!(pk_out[0], pk_expected);

        let secrets = sampler.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
        let encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let plaintexts = vec![a.clone(), b.clone(), neg_t.clone(), DCRTPoly::const_one(&params)];
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let enc_out = circuit.eval(
            &params,
            &encodings[0],
            &encodings[1..],
            None::<&LWEBGGEncodingPltEvaluator<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>>,
        );
        assert_eq!(enc_out.len(), 1);
        let enc_expected = (encodings[1].clone() * encodings[3].clone()) +
            (encodings[2].clone() * encodings[4].clone());
        assert_eq!(enc_out[0].vector, enc_expected.vector);
        assert_eq!(enc_out[0].pubkey, enc_expected.pubkey);
        assert_eq!(enc_out[0].plaintext, enc_expected.plaintext);
    }
}
