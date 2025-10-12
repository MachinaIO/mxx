use crate::{
    arithmetic::circuit::ArithmeticCircuit,
    bgg::{
        encoding::BggEncoding,
        public_key::BggPublicKey,
        sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    },
    gadgets::crt::encode_modulo_poly,
    lookup::{
        lwe_eval::{LweBggEncodingPltEvaluator, LweBggPubKeyEvaluator},
        poly::PolyPltEvaluator,
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::write::{init_storage_system, wait_for_all_writes},
};
use num_bigint::BigUint;
use std::{path::PathBuf, sync::Arc};
use tracing::info;

const TAG_BGG_PUBKEY: &[u8] = b"BGG_PUBKEY";

impl<P: Poly> ArithmeticCircuit<P> {
    pub fn evaluate_with_poly(&self, params: &P::Params, inputs: &[&[BigUint]]) -> Vec<P> {
        let one = P::const_one(params);
        let input_polys = inputs
            .iter()
            .flat_map(|input| encode_modulo_poly(self.limb_bit_size, params, input))
            .collect::<Vec<_>>();
        let plt_evaluator =
            if self.limb_bit_size > 1 { Some(PolyPltEvaluator::new()) } else { None };
        self.poly_circuit.eval(params, &one, &input_polys, plt_evaluator)
    }

    pub async fn evaluate_with_bgg_pubkey<M, SH, ST, SU>(
        &self,
        params: &P::Params,
        seed: [u8; 32],
        dir_path: PathBuf,
        d: usize,
        pub_matrix: Arc<M>,
        trapdoor: Arc<ST::Trapdoor>,
        trapdoor_sampler: ST,
    ) -> Vec<BggPublicKey<M>>
    where
        M: PolyMatrix<P = P> + Clone + Send + Sync + 'static,
        SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        ST: PolyTrapdoorSampler<M = M> + Clone + Send + Sync,
        SU: PolyUniformSampler<M = M> + Send + Sync,
    {
        init_storage_system();
        let pubkeys = self.sample_input_pubkeys::<M, SH>(params, seed, d);
        info!("sampled all pubkeys {}", pubkeys.len());
        let plt_evaluator = if self.limb_bit_size > 1 {
            Some(LweBggPubKeyEvaluator::<M, SH, ST>::new(
                seed,
                trapdoor_sampler,
                pub_matrix.clone(),
                trapdoor.clone(),
                dir_path.clone(),
            ))
        } else {
            None
        };
        let outputs = self.poly_circuit.eval(params, &pubkeys[0], &pubkeys[1..], plt_evaluator);
        info!("finished evaluation of pubkeys");
        wait_for_all_writes(dir_path).await.unwrap();
        info!("finished write files");
        outputs
    }

    pub fn evaluate_with_bgg_encoding<M, SH, SU>(
        &self,
        params: &P::Params,
        seed: [u8; 32],
        dir_path: PathBuf,
        inputs: &[&[BigUint]],
        secret: &[P],
        p: M,
        error_sigma: f64,
    ) -> Vec<BggEncoding<M>>
    where
        M: PolyMatrix<P = P> + Clone + Send + Sync + 'static,
        SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        SU: PolyUniformSampler<M = M> + Send + Sync,
    {
        // If error_sigma <= 0.0, disable noise to allow exact equality in tests.
        let gauss_sigma = if error_sigma > 0.0 { Some(error_sigma) } else { None };
        let bgg_encoding_sampler = BGGEncodingSampler::<SU>::new(params, secret, gauss_sigma);
        let plaintexts = inputs
            .iter()
            .flat_map(|input| encode_modulo_poly(self.limb_bit_size, params, input))
            .collect::<Vec<_>>();
        let pubkeys = self.sample_input_pubkeys::<M, SH>(params, seed, secret.len());
        let encodings = bgg_encoding_sampler.sample(params, &pubkeys, &plaintexts);
        let bgg_evaluator = LweBggEncodingPltEvaluator::<M, SH>::new(seed, dir_path, p);
        self.poly_circuit.eval(params, &encodings[0], &encodings[1..], Some(bgg_evaluator))
    }

    fn sample_input_pubkeys<M, SH>(
        &self,
        params: &P::Params,
        seed: [u8; 32],
        d: usize,
    ) -> Vec<BggPublicKey<M>>
    where
        M: PolyMatrix<P = P> + Clone + Send + Sync + 'static,
        SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    {
        let num_given_input_polys = {
            let (_, crt_bits, _) = params.to_crt();
            let num_limbs_per_slot = crt_bits.div_ceil(self.limb_bit_size);
            self.num_inputs * num_limbs_per_slot
        };
        let reveal_plaintexts = vec![true; num_given_input_polys + 1];
        let bgg_pubkey_sampler = BGGPublicKeySampler::<_, SH>::new(seed, d);
        bgg_pubkey_sampler.sample(params, TAG_BGG_PUBKEY, &reveal_plaintexts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly};
    use num_bigint::BigUint;
    use num_traits::One;

    #[test]
    fn test_benchmark_multiplication_tree_evaluation() {
        let params = DCRTPolyParams::default();
        let ring_degree = params.ring_dimension() as usize;
        let height = 4;
        let limb_bit_size = 3usize;
        let circuit = ArithmeticCircuit::<DCRTPoly>::benchmark_multiplication_tree(
            &params,
            limb_bit_size,
            ring_degree,
            height,
        );

        let ring_n = params.ring_dimension() as usize;
        let num_inputs = 1usize << height;
        let inputs: Vec<Vec<BigUint>> = (0..num_inputs)
            .map(|i| {
                (0..ring_n).map(|slot| BigUint::from(((i + slot + 1) % 13 + 1) as u64)).collect()
            })
            .collect();

        let modulus = params.modulus();
        let modulus_ref = modulus.as_ref();
        let expected_slots = (0..ring_n)
            .map(|slot| {
                inputs.iter().fold(BigUint::one(), |acc, input| (acc * &input[slot]) % modulus_ref)
            })
            .collect::<Vec<_>>();
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_slots);

        let input_refs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let result = circuit.evaluate_with_poly(&params, &input_refs);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected_poly);
    }
}
