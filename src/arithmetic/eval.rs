use crate::{
    arithmetic::circuit::ArithmeticCircuit,
    bgg::{
        encoding::BggEncoding,
        public_key::BggPublicKey,
        sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    },
    gadgets::crt::biguint_to_crt_poly,
    lookup::{
        poly::PolyPltEvaluator,
        simple_eval::{SimpleBggEncodingPltEvaluator, SimpleBggPubKeyEvaluator},
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
};
use num_bigint::BigUint;
use std::path::PathBuf;
use tracing::info;

const TAG_BGG_PUBKEY: &[u8] = b"BGG_PUBKEY";

impl<P: Poly> ArithmeticCircuit<P> {
    pub fn evaluate_with_poly(&self, params: &P::Params, inputs: &[BigUint]) -> Vec<P> {
        let one = P::const_one(params);
        let mut all_input_polys = Vec::new();

        for input in inputs {
            let crt_limbs = biguint_to_crt_poly(self.limb_bit_size, params, input);
            all_input_polys.extend(crt_limbs);
        }

        let plt_evaluator = PolyPltEvaluator::new();
        
        self.poly_circuit.eval(params, &one, &all_input_polys, Some(plt_evaluator))
    }

    pub fn evaluate_with_bgg_pubkey<M, SH, ST, SU>(
        &self,
        params: &P::Params,
        num_inputs: usize,
        seed: [u8; 32],
        dir_path: PathBuf,
        d: usize,
        pub_matrix: M,
        trapdoor: ST::Trapdoor,
        trapdoor_sampler: ST,
    ) -> Vec<BggPublicKey<M>>
    where
        M: PolyMatrix<P = P> + Clone + Send + Sync + 'static,
        SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        ST: PolyTrapdoorSampler<M = M> + Clone + Send + Sync,
        SU: PolyUniformSampler<M = M> + Send + Sync,
    {
        let (_, _, crt_depth) = params.to_crt();
        let total_limbs = self.num_limbs * crt_depth * num_inputs;
        let num_packed_poly_inputs = total_limbs.div_ceil(self.packed_limbs);
        let reveal_plaintexts = vec![true; num_packed_poly_inputs + 1];
        let bgg_pubkey_sampler = BGGPublicKeySampler::<_, SH>::new(seed, d);
        let pubkeys = bgg_pubkey_sampler.sample(params, TAG_BGG_PUBKEY, &reveal_plaintexts);
        info!("sampled all pubkeys {}", pubkeys.len());
        let bgg_evaluator = SimpleBggPubKeyEvaluator::<M, SH, SU, ST>::new(
            seed,
            trapdoor_sampler,
            std::sync::Arc::new(pub_matrix),
            std::sync::Arc::new(trapdoor),
            dir_path,
        );
        
        self.poly_circuit.eval(params, &pubkeys[0], &pubkeys[1..], Some(bgg_evaluator))
    }

    pub fn evaluate_with_bgg_encoding<M, SH, SU>(
        &self,
        params: &P::Params,
        seed: [u8; 32],
        dir_path: PathBuf,
        inputs: &[BigUint],
        secret: &[P],
        p: M,
        error_sigma: f64,
    ) -> Vec<BggEncoding<M>>
    where
        M: PolyMatrix<P = P> + Clone + Send + Sync + 'static,
        SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        SU: PolyUniformSampler<M = M> + Send + Sync,
    {
        let (_, _, crt_depth) = params.to_crt();
        let total_limbs = self.num_limbs * crt_depth * inputs.len();
        let num_packed_poly_inputs = total_limbs.div_ceil(self.packed_limbs);
        let reveal_plaintexts = vec![true; num_packed_poly_inputs + 1];
        let bgg_pubkey_sampler = BGGPublicKeySampler::<_, SH>::new(seed, secret.len());
        let pubkeys = bgg_pubkey_sampler.sample(params, TAG_BGG_PUBKEY, &reveal_plaintexts);
        info!("sampled all pubkeys {}", pubkeys.len());
        let mut packed_inputs: Vec<P> = vec![];
        for input in inputs {
            let crt_limbs = biguint_to_crt_poly(self.limb_bit_size, params, input);
            packed_inputs.extend(crt_limbs);
        }

        let uniform_sampler = SU::new();
        let bgg_encoding_sampler =
            BGGEncodingSampler::new(params, secret, uniform_sampler, error_sigma);
        let encodings = bgg_encoding_sampler.sample(params, &pubkeys, &packed_inputs);
        let bgg_evaluator = SimpleBggEncodingPltEvaluator::<M, SH>::new(seed, dir_path, p);
        
        self.poly_circuit.eval(params, &encodings[0], &encodings[1..], Some(bgg_evaluator))
    }
}
