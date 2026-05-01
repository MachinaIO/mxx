use crate::{
    bench_estimator::{PublicLutSampleAuxBenchEstimator, SampleAuxBenchEstimate},
    bgg::naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{PolyHashSampler, PolyTrapdoorSampler},
};
use rayon::prelude::*;

use super::{LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator};

pub struct NaiveLWEBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub inner: LWEBGGPubKeyPltEvaluator<M, SH, ST>,
}

impl<M, SH, ST> NaiveLWEBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn new(inner: LWEBGGPubKeyPltEvaluator<M, SH, ST>) -> Self {
        Self { inner }
    }

    pub fn sample_aux_matrices(&self, params: &<M::P as Poly>::Params) {
        self.inner.sample_aux_matrices(params);
    }
}

impl<M, SH, ST> PltEvaluator<NaiveBGGPublicKeyVec<M>>
    for NaiveLWEBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    fn public_lookup(
        &self,
        params: &<NaiveBGGPublicKeyVec<M> as Evaluable>::Params,
        plt: &PublicLut<<NaiveBGGPublicKeyVec<M> as Evaluable>::P>,
        one: &NaiveBGGPublicKeyVec<M>,
        input: &NaiveBGGPublicKeyVec<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> NaiveBGGPublicKeyVec<M> {
        input.assert_compatible(one);
        let num_slots = input.num_slots();
        NaiveBGGPublicKeyVec::new(
            params,
            (0..num_slots)
                .into_par_iter()
                .map(|slot_idx| {
                    let input_key = input.key(slot_idx);
                    self.inner.public_lookup_for_slot(
                        params,
                        plt,
                        &input_key,
                        gate_id,
                        lut_id,
                        Some(slot_idx),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl<M, SH, ST> PublicLutSampleAuxBenchEstimator<M>
    for NaiveLWEBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    type Params = <M::P as Poly>::Params;

    fn estimate_public_lut_sample_aux_matrices(
        &self,
        params: &Self::Params,
        total_lut_entries: usize,
        total_lut_gates: usize,
    ) -> SampleAuxBenchEstimate {
        self.inner.estimate_public_lut_sample_aux_matrices(
            params,
            total_lut_entries,
            total_lut_gates,
        )
    }

    fn write_dummy_aux_for_poly_encode_bench(
        &self,
        params: &Self::Params,
        plt: &PublicLut<M::P>,
        used_inputs: &[u64],
        lut_id: usize,
        gate_id: GateId,
        error_sigma: f64,
    ) {
        // Only the slot-0 dummy auxiliary data is needed for benchmark warmup because the naive
        // vector benchmark estimator measures a single scalar slot and scales the estimate by the
        // logical slot count.
        self.inner.write_dummy_aux_for_poly_encode_bench(
            params,
            plt,
            used_inputs,
            lut_id,
            gate_id,
            error_sigma,
        );
    }
}

#[derive(Debug, Clone)]
pub struct NaiveLWEBGGEncodingVecPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub inner: LWEBGGEncodingPltEvaluator<M, SH>,
}

impl<M, SH> NaiveLWEBGGEncodingVecPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(inner: LWEBGGEncodingPltEvaluator<M, SH>) -> Self {
        Self { inner }
    }
}

impl<M, SH> PltEvaluator<NaiveBGGEncodingVec<M>> for NaiveLWEBGGEncodingVecPltEvaluator<M, SH>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    fn public_lookup(
        &self,
        params: &<NaiveBGGEncodingVec<M> as Evaluable>::Params,
        plt: &PublicLut<<NaiveBGGEncodingVec<M> as Evaluable>::P>,
        one: &NaiveBGGEncodingVec<M>,
        input: &NaiveBGGEncodingVec<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> NaiveBGGEncodingVec<M> {
        input.assert_compatible(one);
        let num_slots = input.num_slots();
        NaiveBGGEncodingVec::new(
            params,
            (0..num_slots)
                .into_par_iter()
                .map(|slot_idx| {
                    let input_encoding = input.encoding(slot_idx);
                    self.inner.public_lookup_for_slot(
                        params,
                        plt,
                        &input_encoding,
                        gate_id,
                        lut_id,
                        Some(slot_idx),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{NaiveLWEBGGEncodingVecPltEvaluator, NaiveLWEBGGPublicKeyVecPltEvaluator};
    use crate::{
        __PAIR, __TestState,
        bgg::{
            encoding::BggEncoding,
            naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
            sampler::BGGPublicKeySampler,
        },
        circuit::PolyCircuit,
        element::PolyElem,
        lookup::{
            PublicLut,
            lwe::{LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator},
        },
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            PolyTrapdoorSampler, hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        utils::create_bit_random_poly,
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path, sync::Arc};

    const SIGMA: f64 = 4.578;

    fn lsb_lut(params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::new(
            params,
            16,
            |params: &DCRTPolyParams, input| {
                Some((input, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), input & 1)))
            },
            Some((1, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 1))),
        )
    }

    fn prepare_clean_storage(dir_path: &str) {
        let dir = Path::new(dir_path);
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_naive_lwe_bgg_vec_lookup_output_vectors_match_plaintext_relation() {
        let _storage_lock = storage_test_lock().await;
        let params = DCRTPolyParams::default();
        let plt = lsb_lut(&params);

        let mut circuit = PolyCircuit::new();
        let input = circuit.input(1).as_single_wire();
        let plt_id = circuit.register_public_lookup(plt);
        let output = circuit.public_lookup_gate(input, plt_id);
        circuit.output(vec![output]);

        let d = 2;
        let num_slots = 2;
        let hash_key = [0x46u8; 32];
        let pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, d);
        let secrets = vec![create_bit_random_poly(&params); d];
        let plaintexts = vec![
            DCRTPoly::from_usize_to_constant(&params, 5),
            DCRTPoly::from_usize_to_constant(&params, 6),
        ];
        let pubkeys = pubkey_sampler.sample(&params, b"naive-lwe-output-relation", &[true, true]);
        let secret_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);
        let gadget = DCRTPolyMatrix::gadget_matrix(&params, d);
        let encoding_plaintexts = [&[DCRTPoly::const_one(&params)], plaintexts.as_slice()].concat();
        let encodings = pubkeys
            .iter()
            .zip(encoding_plaintexts)
            .map(|(pubkey, plaintext)| {
                BggEncoding::new(
                    secret_vec.clone() *
                        (pubkey.matrix.clone() - &(gadget.clone() * plaintext.clone())),
                    pubkey.clone(),
                    Some(plaintext),
                )
            })
            .collect::<Vec<_>>();

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
        let c_b = secret_vec.clone() * &b;
        let dir_path = "test_data/test_naive_lwe_bgg_vec_lookup_output_relation";
        prepare_clean_storage(dir_path);

        let pubkey_evaluator =
            NaiveLWEBGGPublicKeyVecPltEvaluator::new(LWEBGGPubKeyPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(
                hash_key,
                trapdoor_sampler,
                Arc::new(b),
                Arc::new(b_trapdoor),
                dir_path.into(),
            ));
        let one_pubkey = NaiveBGGPublicKeyVec::new(vec![pubkeys[0].clone(); num_slots]);
        let input_pubkey = NaiveBGGPublicKeyVec::new(pubkeys[1..].to_vec());
        let result_pubkey = circuit.eval(
            &params,
            one_pubkey,
            vec![input_pubkey],
            Some(&pubkey_evaluator),
            None,
            None,
        );
        pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(Path::new(dir_path).to_path_buf()).await.unwrap();

        let encoding_evaluator =
            NaiveLWEBGGEncodingVecPltEvaluator::new(LWEBGGEncodingPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyHashSampler<Keccak256>,
            >::new(
                hash_key, dir_path.into(), c_b
            ));
        let one_encoding = NaiveBGGEncodingVec::new(vec![encodings[0].clone(); num_slots]);
        let input_encoding = NaiveBGGEncodingVec::new(encodings[1..].to_vec());
        let result_encoding = circuit.eval(
            &params,
            one_encoding,
            vec![input_encoding],
            Some(&encoding_evaluator),
            None,
            None,
        );

        assert_eq!(result_pubkey.len(), 1);
        assert_eq!(result_encoding.len(), 1);
        for slot_idx in 0..num_slots {
            let output_encoding = &result_encoding[0].encodings[slot_idx];
            let output_pubkey = &result_pubkey[0].keys[slot_idx];
            assert_eq!(output_encoding.pubkey, output_pubkey.clone());

            let expected_bit = plaintexts[slot_idx].const_coeff_u64() & 1;
            let expected_output = DCRTPoly::from_usize_to_constant(&params, expected_bit as usize);
            assert_eq!(
                output_encoding
                    .plaintext
                    .as_ref()
                    .expect("lookup output plaintext should be revealed"),
                &expected_output
            );

            let expected_vector = secret_vec.clone() * output_pubkey.matrix.clone() -
                (secret_vec.clone() * (gadget.clone() * expected_output));
            assert!(output_encoding.vector == expected_vector, "output slot {slot_idx}");
        }
    }
}
