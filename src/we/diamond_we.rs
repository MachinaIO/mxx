use std::{marker::PhantomData, path::PathBuf, sync::Arc};

use num_bigint::BigUint;

use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey, sampler::BGGPublicKeySampler},
    circuit::{Evaluable, PolyCircuit},
    func_enc::NoCircuitEvaluator,
    input_injector::{
        DIAMOND_SECRET_SIZE, DiamondInjector, DiamondInjectorPreprocessOut, InputInjector,
    },
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::SlotTransferEvaluator,
};

use super::WitnessEnc;

pub mod bench_estimator;
pub mod simulation;
pub use bench_estimator::{DiamondWEBenchEstimate, DiamondWEBenchEstimator};
pub use simulation::{
    DiamondWECrtDepthSearchResult, DiamondWEErrorSimulation, diamond_we_find_crt_depth,
};

#[derive(Debug, Clone)]
pub struct DiamondWECiphertext<M, T>
where
    M: PolyMatrix,
{
    pub circuit: PolyCircuit<M::P>,
    pub instance: Vec<bool>,
    pub hash_key: [u8; 32],
    pub preprocess_out: DiamondInjectorPreprocessOut<M, T>,
}

#[derive(Clone)]
pub struct DiamondWE<
    M,
    US,
    HS,
    TS,
    PKPE = NoCircuitEvaluator,
    PKST = NoCircuitEvaluator,
    ENCPE = NoCircuitEvaluator,
    ENCST = NoCircuitEvaluator,
> where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub injector: DiamondInjector<M, US, HS, TS>,
    pub witness_size: usize,
    pub artifact_dir: PathBuf,
    pub bgg_tag: Vec<u8>,
    pub pk_lookup_evaluator: Option<PKPE>,
    pub pk_slot_transfer_evaluator: Option<PKST>,
    pub enc_lookup_base_matrix: Option<M>,
    pub enc_lookup_evaluator_factory: Option<Arc<dyn Fn(M) -> ENCPE + Send + Sync>>,
    pub enc_lookup_evaluator: Option<ENCPE>,
    pub enc_slot_transfer_evaluator: Option<ENCST>,
    _m: PhantomData<M>,
}

impl<M, US, HS, TS>
    DiamondWE<
        M,
        US,
        HS,
        TS,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
    >
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        injector: DiamondInjector<M, US, HS, TS>,
        witness_size: usize,
        artifact_dir: impl Into<PathBuf>,
        bgg_tag: Vec<u8>,
    ) -> Self {
        Self {
            injector,
            witness_size,
            artifact_dir: artifact_dir.into(),
            bgg_tag,
            pk_lookup_evaluator: None,
            pk_slot_transfer_evaluator: None,
            enc_lookup_base_matrix: None,
            enc_lookup_evaluator_factory: None,
            enc_lookup_evaluator: None,
            enc_slot_transfer_evaluator: None,
            _m: PhantomData,
        }
    }
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    #[allow(clippy::too_many_arguments)]
    pub fn with_evaluators(
        injector: DiamondInjector<M, US, HS, TS>,
        witness_size: usize,
        artifact_dir: impl Into<PathBuf>,
        bgg_tag: Vec<u8>,
        pk_lookup_evaluator: Option<PKPE>,
        pk_slot_transfer_evaluator: Option<PKST>,
        enc_lookup_base_matrix: Option<M>,
        enc_lookup_evaluator_factory: Option<Arc<dyn Fn(M) -> ENCPE + Send + Sync>>,
        enc_lookup_evaluator: Option<ENCPE>,
        enc_slot_transfer_evaluator: Option<ENCST>,
    ) -> Self {
        Self {
            injector,
            witness_size,
            artifact_dir: artifact_dir.into(),
            bgg_tag,
            pk_lookup_evaluator,
            pk_slot_transfer_evaluator,
            enc_lookup_base_matrix,
            enc_lookup_evaluator_factory,
            enc_lookup_evaluator,
            enc_slot_transfer_evaluator,
            _m: PhantomData,
        }
    }

    fn matrix_path(&self, id: &str) -> PathBuf {
        self.artifact_dir.join(format!("{id}.matrixbin"))
    }

    fn write_matrix(&self, id: &str, matrix: &M) {
        std::fs::create_dir_all(&self.artifact_dir).unwrap_or_else(|err| {
            panic!(
                "DiamondWE failed to create artifact directory {}: {err}",
                self.artifact_dir.display()
            )
        });
        std::fs::write(self.matrix_path(id), matrix.to_compact_bytes())
            .unwrap_or_else(|err| panic!("DiamondWE failed to write matrix {id}: {err}"));
    }

    fn read_matrix(&self, id: &str) -> M {
        let bytes = std::fs::read(self.matrix_path(id))
            .unwrap_or_else(|err| panic!("DiamondWE failed to read matrix {id}: {err}"));
        M::from_compact_bytes(&self.injector.params, &bytes)
    }

    fn one_preimage_id() -> &'static str {
        "we_one_preimage"
    }

    fn witness_preimage_id(bit_idx: usize) -> String {
        format!("we_witness_preimage_{bit_idx}")
    }

    fn k_preimage_id() -> &'static str {
        "we_k_preimage"
    }

    fn decoder_preimage_id() -> &'static str {
        "we_decoder_preimage"
    }

    fn enc_lookup_base_preimage_id() -> &'static str {
        "we_enc_lookup_base_preimage"
    }

    fn sample_w_block(
        &self,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        block_idx: usize,
        cols: usize,
    ) -> M {
        HS::new().sample_hash(
            &self.injector.params,
            preprocess_out.hash_key,
            format!("diamond_w_{}_{}", block_idx, self.injector.input_count),
            2usize
                .checked_mul(DIAMOND_SECRET_SIZE)
                .expect("DiamondWE final state row count overflow"),
            cols,
            DistType::FinRingDist,
        )
    }

    fn sample_bgg_public_keys(
        &self,
        hash_key: [u8; 32],
    ) -> (BggPublicKey<M>, BggPublicKey<M>, Vec<BggPublicKey<M>>) {
        let params = &self.injector.params;
        let mut tag = self.bgg_tag.clone();
        tag.extend_from_slice(b":witness_public_keys");
        let reveal_plaintexts = vec![true; self.witness_size];
        let sampled = BGGPublicKeySampler::<[u8; 32], HS>::new(hash_key, DIAMOND_SECRET_SIZE)
            .sample(params, &tag, &reveal_plaintexts);
        assert_eq!(
            sampled.len(),
            self.witness_size + 1,
            "DiamondWE public-key sampler must return one and all witness keys"
        );
        let one = sampled[0].clone();
        let witness_pubkeys = sampled[1..].to_vec();

        let mut k_tag = self.bgg_tag.clone();
        k_tag.extend_from_slice(b":k_public_key");
        let k_matrix = HS::new().sample_hash(params, hash_key, &k_tag, 1, 1, DistType::FinRingDist);
        let k_pubkey = BggPublicKey::new(k_matrix, false);
        (one, k_pubkey, witness_pubkeys)
    }

    fn sample_output_preimage(
        &self,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        block_idx: usize,
        pubkey: &BggPublicKey<M>,
        top_gadget_plaintext: Option<&M::P>,
        bottom_gadget_plaintext: Option<&M::P>,
    ) -> M {
        let params = &self.injector.params;
        let gadget = M::gadget_matrix(params, DIAMOND_SECRET_SIZE);
        let cols = DIAMOND_SECRET_SIZE
            .checked_mul(params.modulus_digits())
            .expect("DiamondWE final gadget column count overflow");
        let mut top = pubkey.matrix.clone();
        if let Some(plaintext) = top_gadget_plaintext {
            top = top - &(gadget.clone() * plaintext);
        }
        let mut bottom = M::zero(params, DIAMOND_SECRET_SIZE, cols);
        if let Some(plaintext) = bottom_gadget_plaintext {
            bottom = bottom - &(gadget * plaintext);
        }
        let target = top.concat_rows(&[&bottom]);
        TS::new(params, self.injector.trapdoor_sigma).preimage_extend(
            params,
            &preprocess_out.final_trapdoor,
            &preprocess_out.final_pub_matrix,
            &self.sample_w_block(preprocess_out, block_idx, cols),
            &target,
        )
    }

    fn sample_k_preimage(
        &self,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        k_pubkey: &BggPublicKey<M>,
    ) -> M {
        let params = &self.injector.params;
        assert_eq!(
            k_pubkey.matrix.size(),
            (DIAMOND_SECRET_SIZE, DIAMOND_SECRET_SIZE),
            "DiamondWE k public key must be a 1 x 1 public matrix"
        );
        let identity = M::identity(params, DIAMOND_SECRET_SIZE, None);
        let target = k_pubkey.matrix.concat_rows(&[&identity]);
        let ext_cols = DIAMOND_SECRET_SIZE
            .checked_mul(params.modulus_digits())
            .expect("DiamondWE final W block column count overflow");
        TS::new(params, self.injector.trapdoor_sigma).preimage_extend(
            params,
            &preprocess_out.final_trapdoor,
            &preprocess_out.final_pub_matrix,
            &self.sample_w_block(preprocess_out, 0, ext_cols),
            &target,
        )
    }

    fn sample_decoder_preimage(
        &self,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        dec_pubkey_matrix: &M,
    ) -> M {
        let params = &self.injector.params;
        let bottom = M::zero(params, DIAMOND_SECRET_SIZE, dec_pubkey_matrix.col_size());
        let target = dec_pubkey_matrix.concat_rows(&[&bottom]);
        let ext_cols = DIAMOND_SECRET_SIZE
            .checked_mul(params.modulus_digits())
            .expect("DiamondWE final W block column count overflow");
        TS::new(params, self.injector.trapdoor_sigma).preimage_extend(
            params,
            &preprocess_out.final_trapdoor,
            &preprocess_out.final_pub_matrix,
            &self.sample_w_block(preprocess_out, 0, ext_cols),
            &target,
        )
    }

    fn sample_lookup_base_preimage(
        &self,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        lookup_base_matrix: &M,
    ) -> M {
        assert_eq!(
            lookup_base_matrix.row_size(),
            DIAMOND_SECRET_SIZE,
            "DiamondWE lookup base matrix row count must match the Diamond secret size"
        );
        let params = &self.injector.params;
        let lookup_bottom = M::zero(params, DIAMOND_SECRET_SIZE, lookup_base_matrix.col_size());
        let target = lookup_base_matrix.concat_rows(&[&lookup_bottom]);
        TS::new(params, self.injector.trapdoor_sigma).preimage_extend(
            params,
            &preprocess_out.final_trapdoor,
            &preprocess_out.final_pub_matrix,
            &self.sample_w_block(preprocess_out, 0, target.col_size()),
            &target,
        )
    }

    fn sample_r(&self, hash_key: [u8; 32]) -> M {
        let mut tag = self.bgg_tag.clone();
        tag.extend_from_slice(b":r");
        HS::new().sample_hash(&self.injector.params, hash_key, &tag, 1, 1, DistType::FinRingDist)
    }

    fn instance_pubkeys(&self, one: &BggPublicKey<M>, instance: &[bool]) -> Vec<BggPublicKey<M>> {
        instance
            .iter()
            .map(|bit| one.small_scalar_mul(&self.injector.params, &[*bit as u32]))
            .collect()
    }

    fn instance_encodings(&self, one: &BggEncoding<M>, instance: &[bool]) -> Vec<BggEncoding<M>> {
        instance
            .iter()
            .map(|bit| one.small_scalar_mul(&self.injector.params, &[*bit as u32]))
            .collect()
    }

    fn pack_witness_digits(&self, witness: &[bool]) -> Vec<u32> {
        let batch_bits = self.injector.batch_bits();
        assert_eq!(
            witness.len(),
            self.witness_size,
            "DiamondWE witness length must match witness_size"
        );
        assert_eq!(
            witness.len() % batch_bits,
            0,
            "DiamondWE witness length must be divisible by the injector digit bit width"
        );
        let digits = witness
            .chunks_exact(batch_bits)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |digit, (bit_idx, bit)| digit | ((*bit as u32) << bit_idx))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            digits.len(),
            self.injector.input_count,
            "DiamondWE packed witness digit count must match the injector input count"
        );
        digits
    }

    fn decode_noisy_bool(&self, noisy_plaintext: &M) -> bool {
        let q = self.injector.params.modulus();
        let q: std::sync::Arc<BigUint> = q.into();
        let coeff = noisy_plaintext.entry(0, 0).coeffs_biguints()[0].clone();
        let quarter_q = q.as_ref() / 4u32;
        let three_quarter_q = &quarter_q * 3u32;
        !(coeff < quarter_q || coeff > three_quarter_q)
    }
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> WitnessEnc<M::P>
    for DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    PKPE: PltEvaluator<BggPublicKey<M>>,
    PKST: SlotTransferEvaluator<BggPublicKey<M>>,
    ENCPE: PltEvaluator<BggEncoding<M>>,
    ENCST: SlotTransferEvaluator<BggEncoding<M>>,
{
    type Msg = bool;
    type Inst = Vec<bool>;
    type Wtns = Vec<bool>;
    type Ciphertext = DiamondWECiphertext<M, TS::Trapdoor>;

    fn enc(
        &self,
        msg: &Self::Msg,
        circuit: PolyCircuit<M::P>,
        instance: &Self::Inst,
    ) -> Self::Ciphertext {
        assert_eq!(circuit.num_output(), 1, "DiamondWE currently requires one circuit output");
        assert_eq!(
            self.witness_size + instance.len(),
            circuit.num_input(),
            "DiamondWE witness_size + instance length must match the circuit input size"
        );
        assert_eq!(
            self.witness_size,
            self.injector.input_count * self.injector.batch_bits(),
            "DiamondWE witness_size must match the DiamondInjector bit input count"
        );

        let params = &self.injector.params;
        let k = if *msg {
            let q: std::sync::Arc<BigUint> = params.modulus().into();
            M::P::from_biguint_to_constant(params, q.as_ref() / 2u32)
        } else {
            M::P::const_zero(params)
        };
        let preprocess_out = self.injector.preprocess(&self.artifact_dir, &k);
        let hash_key = rand::random::<[u8; 32]>();
        let (one_pubkey, k_pubkey, witness_pubkeys) = self.sample_bgg_public_keys(hash_key);
        let instance_pubkeys = self.instance_pubkeys(&one_pubkey, instance);
        let mut input_pubkeys = witness_pubkeys.clone();
        input_pubkeys.extend(instance_pubkeys);

        let out_pubkey = circuit
            .eval(
                params,
                one_pubkey.clone(),
                input_pubkeys,
                self.pk_lookup_evaluator.as_ref(),
                self.pk_slot_transfer_evaluator
                    .as_ref()
                    .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<BggPublicKey<M>>),
                None,
            )
            .into_iter()
            .next()
            .expect("DiamondWE circuit must produce one output public key");

        let one_plaintext = M::P::const_one(params);
        let one_preimage = self.sample_output_preimage(
            &preprocess_out,
            0,
            &one_pubkey,
            Some(&one_plaintext),
            None,
        );
        self.write_matrix(Self::one_preimage_id(), &one_preimage);
        for (bit_idx, pubkey) in witness_pubkeys.iter().enumerate() {
            let digit_idx = bit_idx / self.injector.batch_bits();
            let bit_in_digit = bit_idx % self.injector.batch_bits();
            let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
            let preimage = self.sample_output_preimage(
                &preprocess_out,
                state_idx,
                pubkey,
                None,
                Some(&one_plaintext),
            );
            self.write_matrix(&Self::witness_preimage_id(bit_idx), &preimage);
        }
        let k_preimage = self.sample_k_preimage(&preprocess_out, &k_pubkey);
        self.write_matrix(Self::k_preimage_id(), &k_preimage);
        if let Some(lookup_base_matrix) = self.enc_lookup_base_matrix.as_ref() {
            let lookup_base_preimage =
                self.sample_lookup_base_preimage(&preprocess_out, lookup_base_matrix);
            self.write_matrix(Self::enc_lookup_base_preimage_id(), &lookup_base_preimage);
        }

        let r = self.sample_r(hash_key);
        let dec_term = one_pubkey - &out_pubkey;
        let dec_pubkey_matrix = k_pubkey.matrix + &dec_term.matrix.mul_decompose(&r);
        let decoder_preimage = self.sample_decoder_preimage(&preprocess_out, &dec_pubkey_matrix);
        self.write_matrix(Self::decoder_preimage_id(), &decoder_preimage);

        DiamondWECiphertext { circuit, instance: instance.clone(), hash_key, preprocess_out }
    }

    fn dec(&self, ct: &Self::Ciphertext, witness: &Self::Wtns) -> Self::Msg {
        assert_eq!(ct.circuit.num_output(), 1, "DiamondWE currently requires one circuit output");
        assert_eq!(
            self.witness_size + ct.instance.len(),
            ct.circuit.num_input(),
            "DiamondWE witness_size + instance length must match the circuit input size"
        );
        let params = &self.injector.params;
        let witness_digits = self.pack_witness_digits(witness);
        let states =
            self.injector.online_eval(&self.artifact_dir, &ct.preprocess_out, &witness_digits);
        assert_eq!(
            states.len(),
            1 + self.witness_size,
            "DiamondWE final Diamond state count mismatch"
        );

        let (one_pubkey, k_pubkey, witness_pubkeys) = self.sample_bgg_public_keys(ct.hash_key);
        let one_encoding = self.injector.build_output_encoding(
            states[0].clone() * &self.read_matrix(Self::one_preimage_id()),
            one_pubkey,
            Some(M::P::const_one(params)),
        );
        let k_encoding = self.injector.build_output_encoding(
            states[0].clone() * &self.read_matrix(Self::k_preimage_id()),
            k_pubkey,
            None,
        );
        let mut input_encodings = witness_pubkeys
            .into_iter()
            .enumerate()
            .map(|(bit_idx, pubkey)| {
                let digit_idx = bit_idx / self.injector.batch_bits();
                let bit_in_digit = bit_idx % self.injector.batch_bits();
                let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
                let plaintext = M::P::from_usize_to_constant(
                    params,
                    self.injector.digit_bit_value(witness_digits[digit_idx] as usize, bit_in_digit),
                );
                self.injector.build_output_encoding(
                    states[state_idx].clone() *
                        &self.read_matrix(&Self::witness_preimage_id(bit_idx)),
                    pubkey,
                    Some(plaintext),
                )
            })
            .collect::<Vec<_>>();
        input_encodings.extend(self.instance_encodings(&one_encoding, &ct.instance));

        let owned_enc_lookup_evaluator =
            self.enc_lookup_evaluator_factory.as_ref().map(|factory| {
                let lookup_base_preimage = self.read_matrix(Self::enc_lookup_base_preimage_id());
                let c_b0 = states[0].clone() * &lookup_base_preimage;
                factory(c_b0)
            });
        let enc_lookup_evaluator =
            owned_enc_lookup_evaluator.as_ref().or(self.enc_lookup_evaluator.as_ref());

        let out_encoding = ct
            .circuit
            .eval(
                params,
                one_encoding.clone(),
                input_encodings,
                enc_lookup_evaluator,
                self.enc_slot_transfer_evaluator
                    .as_ref()
                    .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<BggEncoding<M>>),
                None,
            )
            .into_iter()
            .next()
            .expect("DiamondWE circuit must produce one output encoding");
        let r = self.sample_r(ct.hash_key);
        let dec_term = one_encoding - &out_encoding;
        let dec_encoding_vector = k_encoding.vector + &dec_term.vector.mul_decompose(&r);
        let decoder = states[0].clone() * &self.read_matrix(Self::decoder_preimage_id());
        let noisy_plaintext = decoder - &dec_encoding_vector;
        self.decode_noisy_bool(&noisy_plaintext)
    }
}

#[cfg(test)]
mod tests {
    use keccak_asm::Keccak256;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        circuit::PolyCircuit,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };

    #[test]
    fn test_diamond_we_constant_one_circuit_round_trip() {
        let params = DCRTPolyParams::default();
        let witness_size = 2;
        let instance = vec![true];
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        circuit.input(witness_size + instance.len());
        let one = circuit.const_one_gate();
        circuit.output(vec![one]);

        let witness = vec![false, true];

        for msg in [false, true] {
            let injector = DiamondInjector::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(params.clone(), 1, 4, 4.578, 0.0);
            let dir = tempdir().expect("temporary DiamondWE artifact directory should be created");
            let we =
                DiamondWE::new(injector, witness_size, dir.path(), b"diamond_we_test".to_vec());
            let ct = we.enc(&msg, circuit.clone(), &instance);
            assert_eq!(we.dec(&ct, &witness), msg);
        }
    }

    #[test]
    fn test_diamond_we_witness_dependent_circuit_round_trip() {
        let params = DCRTPolyParams::default();
        let witness_size = 2;
        let instance = vec![false];
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(witness_size + instance.len()).to_vec();
        let output = circuit.or_gate(inputs[1], inputs[2]);
        circuit.output(vec![output]);

        let witness = vec![false, true];

        for msg in [false, true] {
            let injector = DiamondInjector::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(params.clone(), 1, 4, 4.578, 0.0);
            let dir = tempdir().expect("temporary DiamondWE artifact directory should be created");
            let we = DiamondWE::new(
                injector,
                witness_size,
                dir.path(),
                b"diamond_we_witness_test".to_vec(),
            );
            let ct = we.enc(&msg, circuit.clone(), &instance);
            assert_eq!(we.dec(&ct, &witness), msg);
        }
    }
}
