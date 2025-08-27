use crate::{
    arithmetic::circuit::ArithmeticCircuit,
    bgg::{
        encoding::BggEncoding,
        public_key::BggPublicKey,
        sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    },
    gadgets::{
        crt::{biguint_to_crt_poly, num_limbs_of_crt_poly},
        packed_crt::{biguint_to_packed_crt_polys, num_packed_crt_poly},
    },
    lookup::{
        lwe_eval::{LweBggEncodingPltEvaluator, LweBggPubKeyEvaluator},
        poly::PolyPltEvaluator,
    },
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::{init_storage_system, wait_for_all_writes},
};
use num_bigint::BigUint;
use std::path::PathBuf;
use tracing::info;

const TAG_BGG_PUBKEY: &[u8] = b"BGG_PUBKEY";

impl<P: Poly> ArithmeticCircuit<P> {
    pub fn evaluate_with_poly(&self, params: &P::Params, inputs: &[BigUint]) -> Vec<P> {
        let one = P::const_one(params);
        let input_polys = if self.use_packing {
            biguint_to_packed_crt_polys::<P>(self.limb_bit_size, params, inputs)
        } else {
            inputs
                .iter()
                .flat_map(|input| biguint_to_crt_poly(self.limb_bit_size, params, input))
                .collect()
        };
        let plt_evaluator = if self.use_packing || self.limb_bit_size > 1 {
            Some(PolyPltEvaluator::new())
        } else {
            None
        };
        self.poly_circuit.eval(params, &one, &input_polys, plt_evaluator)
    }

    pub async fn evaluate_with_bgg_pubkey<M, SH, ST, SU>(
        &self,
        params: &P::Params,
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
        init_storage_system();
        let pubkeys = self.sample_input_pubkeys::<M, SH>(params, seed, d);
        info!("sampled all pubkeys {}", pubkeys.len());
        let plt_evaluator = if self.use_packing || self.limb_bit_size > 1 {
            Some(LweBggPubKeyEvaluator::<M, SH, ST>::new(
                seed,
                trapdoor_sampler,
                std::sync::Arc::new(pub_matrix),
                std::sync::Arc::new(trapdoor),
                dir_path,
            ))
        } else {
            None
        };

        let outputs = self.poly_circuit.eval(params, &pubkeys[0], &pubkeys[1..], plt_evaluator);
        wait_for_all_writes().await.unwrap();
        outputs
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
        let uniform_sampler = SU::new();
        let bgg_encoding_sampler =
            BGGEncodingSampler::new(params, secret, uniform_sampler, error_sigma);
        let plaintexts = if self.use_packing {
            biguint_to_packed_crt_polys(self.limb_bit_size, params, inputs)
        } else {
            inputs
                .into_iter()
                .flat_map(|input| biguint_to_crt_poly(self.limb_bit_size, params, input))
                .collect()
        };
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
        let num_given_input_polys = if self.use_packing {
            num_packed_crt_poly::<P>(self.limb_bit_size, params, self.num_inputs)
        } else {
            self.num_inputs * num_limbs_of_crt_poly::<P>(self.limb_bit_size, params)
        };
        let reveal_plaintexts = vec![true; num_given_input_polys + 1];
        let bgg_pubkey_sampler = BGGPublicKeySampler::<_, SH>::new(seed, d);
        bgg_pubkey_sampler.sample(params, TAG_BGG_PUBKEY, &reveal_plaintexts)
    }
}
