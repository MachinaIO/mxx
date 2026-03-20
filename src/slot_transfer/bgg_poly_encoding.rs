#[cfg(feature = "gpu")]
#[path = "bgg_poly_encoding_gpu.rs"]
mod gpu;

use crate::{
    bgg::{poly_encoding::BggPolyEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler},
    slot_transfer::SlotTransferEvaluator,
    storage::read::read_bytes_from_multi_batch,
};
#[cfg(not(feature = "gpu"))]
use rayon::prelude::*;
use std::{marker::PhantomData, ops::Mul, path::PathBuf, sync::Arc};

#[derive(Debug, Clone)]
pub struct BggPolyEncodingSTEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub checkpoint_prefix: String,
    pub c_b0_bytes: Vec<u8>,
    _hs: PhantomData<HS>,
}

impl<M, HS> BggPolyEncodingSTEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(
        hash_key: [u8; 32],
        dir_path: PathBuf,
        checkpoint_prefix: String,
        c_b0_bytes: Vec<u8>,
    ) -> Self {
        Self { hash_key, dir_path, checkpoint_prefix, c_b0_bytes, _hs: PhantomData }
    }

    fn slot_preimage_b0_id_prefix(&self, slot_idx: usize) -> String {
        format!("{}_slot_preimage_b0_{}", self.checkpoint_prefix, slot_idx)
    }

    fn slot_preimage_b1_id_prefix(&self, slot_idx: usize) -> String {
        format!("{}_slot_preimage_b1_{}", self.checkpoint_prefix, slot_idx)
    }

    fn gate_preimage_id_prefix(&self, gate_id: GateId, dst_slot: usize) -> String {
        format!("{}_gate_preimage_{}_dst{}", self.checkpoint_prefix, gate_id, dst_slot)
    }

    fn load_checkpoint_bytes(&self, id_prefix: &str) -> Vec<u8> {
        read_bytes_from_multi_batch(self.dir_path.as_path(), id_prefix, 0)
            .unwrap_or_else(|| panic!("missing slot-transfer checkpoint bytes for {id_prefix}"))
    }

    fn slot_a_for_params(
        &self,
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        dst_slot: usize,
    ) -> M {
        HS::new().sample_hash(
            params,
            self.hash_key,
            format!("slot_transfer_slot_a_{}", dst_slot),
            secret_size,
            secret_size * params.modulus_digits(),
            DistType::FinRingDist,
        )
    }

    fn output_pubkey_for_params(
        &self,
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        gate_id: GateId,
    ) -> BggPublicKey<M> {
        BggPublicKey::new(
            HS::new().sample_hash(
                params,
                self.hash_key,
                format!("slot_transfer_gate_a_out_{}", gate_id),
                secret_size,
                secret_size * params.modulus_digits(),
                DistType::FinRingDist,
            ),
            true,
        )
    }

    fn output_slot_for_params(
        &self,
        params: &<M::P as Poly>::Params,
        c_b0: &M,
        input: &BggPolyEncoding<M>,
        plaintext_bytes_by_slot: &[Arc<[u8]>],
        src_slots: &[(u32, Option<u32>)],
        out_pubkey_matrix: &M,
        gate_id: GateId,
        dst_slot: usize,
    ) -> (Arc<[u8]>, Arc<[u8]>)
    where
        for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
    {
        let (src_slot_u32, scalar) = src_slots[dst_slot];
        let src_slot = usize::try_from(src_slot_u32).expect("source slot index must fit in usize");
        assert!(
            src_slot < input.num_slots(),
            "source slot index {} out of range for input slot count {}",
            src_slot,
            input.num_slots()
        );

        let input_vector = M::from_compact_bytes(params, input.vector_bytes[src_slot].as_ref());

        let src_plaintext =
            M::P::from_compact_bytes(params, plaintext_bytes_by_slot[src_slot].as_ref());
        let constant_term = src_plaintext
            .coeffs_biguints()
            .into_iter()
            .next()
            .expect("plaintext polynomial must contain a constant coefficient");
        drop(src_plaintext);
        let mut output_plaintext = M::P::from_biguint_to_constant(params, constant_term);

        let slot_preimage_b0_bytes =
            self.load_checkpoint_bytes(&self.slot_preimage_b0_id_prefix(src_slot));
        let slot_preimage_b0 = M::from_compact_bytes(params, &slot_preimage_b0_bytes);
        drop(slot_preimage_b0_bytes);

        let slot_preimage_b1_bytes =
            self.load_checkpoint_bytes(&self.slot_preimage_b1_id_prefix(dst_slot));
        let slot_preimage_b1 = M::from_compact_bytes(params, &slot_preimage_b1_bytes);
        drop(slot_preimage_b1_bytes);

        let c_transfer = (c_b0 * &slot_preimage_b0) * &slot_preimage_b1;
        drop(slot_preimage_b0);
        drop(slot_preimage_b1);

        let slot_a = self.slot_a_for_params(params, out_pubkey_matrix.row_size(), dst_slot);
        let slot_a_decomposed = slot_a.decompose();
        drop(slot_a);

        let transfer_plaintext_term = c_transfer.clone() * &output_plaintext;
        let mut pre_out = (&input_vector * &slot_a_decomposed) + &transfer_plaintext_term;
        drop(input_vector);
        drop(slot_a_decomposed);
        drop(transfer_plaintext_term);
        drop(c_transfer);

        if let Some(scalar) = scalar {
            let scalar_poly = M::P::from_usize_to_constant(params, scalar as usize);
            pre_out = pre_out * &scalar_poly;
            output_plaintext = output_plaintext * &scalar_poly;
        }

        let gate_preimage_bytes =
            self.load_checkpoint_bytes(&self.gate_preimage_id_prefix(gate_id, dst_slot));
        let gate_preimage = M::from_compact_bytes(params, &gate_preimage_bytes);
        drop(gate_preimage_bytes);

        let out_vector = (c_b0 * &gate_preimage) + &pre_out;
        drop(gate_preimage);
        drop(pre_out);

        (
            Arc::<[u8]>::from(out_vector.into_compact_bytes()),
            Arc::<[u8]>::from(output_plaintext.to_compact_bytes()),
        )
    }
}

impl<M, HS> SlotTransferEvaluator<BggPolyEncoding<M>> for BggPolyEncodingSTEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    fn slot_transfer(
        &self,
        params: &<BggPolyEncoding<M> as Evaluable>::Params,
        input: &BggPolyEncoding<M>,
        src_slots: &[(u32, Option<u32>)],
        gate_id: GateId,
    ) -> BggPolyEncoding<M> {
        let num_slots = input.num_slots();
        assert_eq!(
            src_slots.len(),
            num_slots,
            "source slot count {} does not match input slot count {}",
            src_slots.len(),
            num_slots
        );
        let plaintext_bytes_by_slot = input
            .plaintext_bytes
            .as_ref()
            .expect("BggPolyEncoding slot transfer requires plaintext_bytes");
        assert_eq!(
            plaintext_bytes_by_slot.len(),
            num_slots,
            "BggPolyEncoding slot transfer requires plaintext_bytes.len() == num_slots"
        );

        let secret_size = input.pubkey.matrix.row_size();
        let out_pubkey = self.output_pubkey_for_params(params, secret_size, gate_id);
        let requested_parallelism =
            crate::env::slot_transfer_slot_parallelism().max(1).min(num_slots.max(1));
        #[cfg(feature = "gpu")]
        let configured_parallelism =
            gpu::effective_gpu_slot_parallelism(requested_parallelism).min(num_slots.max(1));
        #[cfg(not(feature = "gpu"))]
        let configured_parallelism = requested_parallelism;

        #[cfg(feature = "gpu")]
        let (output_vector_bytes, output_plaintext_bytes) =
            gpu::evaluate_slot_transfer_slots_gpu::<M, HS>(
                self,
                params,
                input,
                plaintext_bytes_by_slot,
                src_slots,
                gate_id,
                &out_pubkey,
                configured_parallelism,
            );

        #[cfg(not(feature = "gpu"))]
        let (output_vector_bytes, output_plaintext_bytes) = {
            let c_b0 = M::from_compact_bytes(params, &self.c_b0_bytes);
            let mut output_vector_bytes = Vec::with_capacity(num_slots);
            let mut output_plaintext_bytes = Vec::with_capacity(num_slots);
            for slot_start in (0..num_slots).step_by(configured_parallelism) {
                let chunk_len = (slot_start + configured_parallelism).min(num_slots) - slot_start;
                let chunk_outputs = (0..chunk_len)
                    .into_par_iter()
                    .map(|offset| {
                        self.output_slot_for_params(
                            params,
                            &c_b0,
                            input,
                            plaintext_bytes_by_slot,
                            src_slots,
                            &out_pubkey.matrix,
                            gate_id,
                            slot_start + offset,
                        )
                    })
                    .collect::<Vec<_>>();
                let (chunk_vector_bytes, chunk_plaintext_bytes): (Vec<_>, Vec<_>) =
                    chunk_outputs.into_iter().unzip();
                output_vector_bytes.extend(chunk_vector_bytes);
                output_plaintext_bytes.extend(chunk_plaintext_bytes);
            }
            (output_vector_bytes, output_plaintext_bytes)
        };

        BggPolyEncoding::new(
            params.clone(),
            output_vector_bytes,
            out_pubkey,
            Some(output_plaintext_bytes),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::BggPolyEncodingSTEvaluator;
    use crate::{
        __PAIR, __TestState,
        bgg::{
            poly_encoding::BggPolyEncoding,
            public_key::BggPublicKey,
            sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
        },
        circuit::{PolyCircuit, evaluable::Evaluable, gate::GateId},
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            DistType, PolyHashSampler, PolyTrapdoorSampler, hash::DCRTPolyHashSampler,
            trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
        },
        slot_transfer::bgg_pubkey::BggPublicKeySTEvaluator,
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path, sync::Arc};

    const SIGMA: f64 = 4.578;

    struct DummyPubKeyPltEvaluator;

    impl PltEvaluator<BggPublicKey<DCRTPolyMatrix>> for DummyPubKeyPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPublicKey<DCRTPolyMatrix> as Evaluable>::Params,
            _plt: &PublicLut<<BggPublicKey<DCRTPolyMatrix> as Evaluable>::P>,
            _one: &BggPublicKey<DCRTPolyMatrix>,
            _input: &BggPublicKey<DCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> BggPublicKey<DCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer tests")
        }
    }

    struct DummyPolyEncodingPltEvaluator;

    impl PltEvaluator<BggPolyEncoding<DCRTPolyMatrix>> for DummyPolyEncodingPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPolyEncoding<DCRTPolyMatrix> as Evaluable>::Params,
            _plt: &PublicLut<<BggPolyEncoding<DCRTPolyMatrix> as Evaluable>::P>,
            _one: &BggPolyEncoding<DCRTPolyMatrix>,
            _input: &BggPolyEncoding<DCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> BggPolyEncoding<DCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer tests")
        }
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_slot_transfer_bgg_poly_encoding_matches_slot_relation() {
        let _storage_lock = storage_test_lock().await;

        let params = DCRTPolyParams::default();
        let hash_key = [0x31u8; 32];
        let secret_size = 2usize;
        let num_slots = 3usize;
        let dir_path = "test_data/test_slot_transfer_bgg_poly_encoding";
        let dir = Path::new(dir_path);
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());

        let tag: u64 = 7;
        let reveal_plaintexts = [true];
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, secret_size);
        let public_keys =
            bgg_pubkey_sampler.sample(&params, &tag.to_le_bytes(), &reveal_plaintexts);

        let secrets = (0..secret_size)
            .map(|idx| DCRTPoly::from_usize_to_constant(&params, idx + 2))
            .collect::<Vec<_>>();
        let plaintext_rows = vec![vec![
            Arc::<[u8]>::from(DCRTPoly::from_usize_to_constant(&params, 5).to_compact_bytes()),
            Arc::<[u8]>::from(DCRTPoly::from_usize_to_constant(&params, 9).to_compact_bytes()),
            Arc::<[u8]>::from(DCRTPoly::from_usize_to_constant(&params, 4).to_compact_bytes()),
        ]];
        let encoding_sampler =
            BGGPolyEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let slot_secret_mats = encoding_sampler.sample_slot_secret_mats(&params, num_slots);
        let encodings = encoding_sampler.sample(
            &params,
            &public_keys,
            &plaintext_rows,
            Some(&slot_secret_mats),
        );
        let one = encodings[0].clone();
        let input = encodings[1].clone();

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&params, secret_size);
        let pubkey_evaluator =
            BggPublicKeySTEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(hash_key, secret_size, num_slots, SIGMA, 0.0, dir.to_path_buf());

        let src_slots = [(2, None), (0, Some(3)), (1, Some(2))];
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let transferred = circuit.slot_transfer_gate(inputs[0], &src_slots);
        circuit.output(vec![transferred]);

        let result_pubkey = circuit.eval(
            &params,
            one.pubkey.clone(),
            vec![input.pubkey.clone()],
            None::<&DummyPubKeyPltEvaluator>,
            Some(&pubkey_evaluator),
            None,
        );
        assert_eq!(result_pubkey.len(), 1);

        pubkey_evaluator.sample_aux_matrices(
            &params,
            &b0_matrix,
            &b0_trapdoor,
            slot_secret_mats.clone(),
        );
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();

        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
        let c_b0 = s_vec.clone() * &b0_matrix;
        let evaluator =
            BggPolyEncodingSTEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::new(
                hash_key,
                dir.to_path_buf(),
                pubkey_evaluator.checkpoint_prefix(&params),
                c_b0.to_compact_bytes(),
            );

        let result = circuit.eval(
            &params,
            one,
            vec![input.clone()],
            None::<&DummyPolyEncodingPltEvaluator>,
            Some(&evaluator),
            Some(1),
        );
        assert_eq!(result.len(), 1);
        let output = &result[0];
        assert_eq!(output.pubkey, result_pubkey[0]);

        let a_out = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            format!("slot_transfer_gate_a_out_{}", transferred),
            secret_size,
            secret_size * params.modulus_digits(),
            DistType::FinRingDist,
        );
        let gadget_matrix = DCRTPolyMatrix::gadget_matrix(&params, secret_size);

        for (dst_slot, (src_slot_u32, scalar)) in src_slots.into_iter().enumerate() {
            let src_slot = src_slot_u32 as usize;
            let s_dst =
                DCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[dst_slot].as_ref());
            let source_plaintext = input
                .plaintext(src_slot)
                .expect("input encoding should reveal plaintext constants");
            let constant_term = source_plaintext
                .coeffs_biguints()
                .into_iter()
                .next()
                .expect("plaintext must have a constant coefficient");
            let mut scaled_plaintext = DCRTPoly::from_biguint_to_constant(&params, constant_term);
            if let Some(scalar) = scalar {
                let scalar_poly = DCRTPoly::from_usize_to_constant(&params, scalar as usize);
                scaled_plaintext = scaled_plaintext * scalar_poly;
            }
            let expected_vector = (s_vec.clone() * &s_dst) *
                &(a_out.clone() - (gadget_matrix.clone() * &scaled_plaintext));
            assert_eq!(output.vector(dst_slot), expected_vector);

            let output_plaintext = output
                .plaintext(dst_slot)
                .expect("slot-transfer output should reveal plaintext constants");
            assert_eq!(output_plaintext, scaled_plaintext);
        }
    }
}
