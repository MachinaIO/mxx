use std::{marker::PhantomData, path::PathBuf, sync::Arc};

use num_bigint::BigUint;
use tracing::info;

use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
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

    fn remove_matrix_if_exists(&self, id: &str) {
        match std::fs::remove_file(self.matrix_path(id)) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => panic!("DiamondWE failed to remove stale matrix {id}: {err}"),
        }
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

    fn preimage_chunk_id(id: &str, chunk_idx: usize) -> String {
        format!("{id}_chunk{chunk_idx}")
    }

    fn preimage_chunk_cols(total_cols: usize) -> Option<usize> {
        let chunk_cols = crate::env::aux_sampling_chunk_width().min(total_cols);
        (chunk_cols < total_cols).then_some(chunk_cols)
    }

    fn preimage_chunk_count(total_cols: usize, chunk_cols: usize) -> usize {
        assert!(total_cols > 0, "DiamondWE preimage total column count must be positive");
        total_cols.div_ceil(chunk_cols)
    }

    fn preimage_chunk_bounds(
        total_cols: usize,
        chunk_cols: usize,
        chunk_idx: usize,
    ) -> (usize, usize) {
        assert!(chunk_cols > 0, "DiamondWE preimage chunk column count must be positive");
        let col_start =
            chunk_idx.checked_mul(chunk_cols).expect("DiamondWE preimage chunk start overflow");
        assert!(
            col_start < total_cols,
            "DiamondWE preimage chunk index out of range: total_cols={}, chunk_cols={}, chunk_idx={}",
            total_cols,
            chunk_cols,
            chunk_idx
        );
        let col_len = (total_cols - col_start).min(chunk_cols);
        (col_start, col_len)
    }

    fn sample_target_preimage(
        &self,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        state_idx: usize,
        target: &M,
    ) -> M {
        let (trapdoor, public_matrix) = preprocess_out.final_checkpoint(state_idx);
        TS::new(&self.injector.params, self.injector.trapdoor_sigma).preimage(
            &self.injector.params,
            trapdoor,
            public_matrix,
            target,
        )
    }

    fn sample_and_write_preimage(
        &self,
        id: &str,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        state_idx: usize,
        target: &M,
    ) {
        let total_cols = target.col_size();
        if let Some(chunk_cols) = Self::preimage_chunk_cols(total_cols) {
            let chunk_count = Self::preimage_chunk_count(total_cols, chunk_cols);
            self.remove_matrix_if_exists(id);
            info!(
                id,
                total_cols, chunk_cols, chunk_count, "diamond we final preimage: sampling chunks"
            );
            for chunk_idx in 0..chunk_count {
                let (col_start, col_len) =
                    Self::preimage_chunk_bounds(total_cols, chunk_cols, chunk_idx);
                info!(
                    id,
                    chunk_idx = chunk_idx + 1,
                    chunk_count,
                    col_start,
                    col_len,
                    "diamond we final preimage: sampling chunk"
                );
                let target_chunk = target.slice_columns(col_start, col_start + col_len);
                let preimage =
                    self.sample_target_preimage(preprocess_out, state_idx, &target_chunk);
                self.write_matrix(&Self::preimage_chunk_id(id, chunk_idx), &preimage);
            }
        } else {
            let preimage = self.sample_target_preimage(preprocess_out, state_idx, target);
            self.write_matrix(id, &preimage);
        }
    }

    fn left_mul_preimage_columns(
        &self,
        lhs: &M,
        id: &str,
        total_cols: usize,
        col_start: usize,
        col_len: usize,
    ) -> M {
        assert!(col_len > 0, "DiamondWE preimage projection must request columns");
        let col_end = col_start
            .checked_add(col_len)
            .expect("DiamondWE preimage projection column range overflow");
        assert!(
            col_end <= total_cols,
            "DiamondWE preimage projection out of range: total_cols={total_cols}, col_start={col_start}, col_len={col_len}"
        );

        if self.matrix_path(id).exists() {
            let preimage = self.read_matrix(id);
            assert_eq!(
                preimage.col_size(),
                total_cols,
                "DiamondWE full preimage artifact {id} has unexpected column count"
            );
            let preimage = if col_start == 0 && col_len == total_cols {
                preimage
            } else {
                preimage.slice_columns(col_start, col_end)
            };
            return lhs.clone() * &preimage;
        }

        let chunk_cols = Self::preimage_chunk_cols(total_cols).unwrap_or_else(|| {
            panic!(
                "DiamondWE missing full preimage artifact {id} and chunking is disabled for total_cols={total_cols}"
            )
        });
        let first_chunk = col_start / chunk_cols;
        let last_chunk = (col_end - 1) / chunk_cols;
        let mut out = M::zero(&self.injector.params, lhs.row_size(), col_len);
        for chunk_idx in first_chunk..=last_chunk {
            let (chunk_col_start, expected_cols) =
                Self::preimage_chunk_bounds(total_cols, chunk_cols, chunk_idx);
            let chunk_col_end = chunk_col_start + expected_cols;
            let overlap_start = col_start.max(chunk_col_start);
            let overlap_end = col_end.min(chunk_col_end);
            let overlap_len = overlap_end - overlap_start;
            let preimage_chunk = self.read_matrix(&Self::preimage_chunk_id(id, chunk_idx));
            assert_eq!(
                preimage_chunk.col_size(),
                expected_cols,
                "DiamondWE preimage chunk {chunk_idx} for {id} has unexpected column count"
            );
            let local_start = overlap_start - chunk_col_start;
            let preimage_chunk = if local_start == 0 && overlap_len == expected_cols {
                preimage_chunk
            } else {
                preimage_chunk.slice_columns(local_start, local_start + overlap_len)
            };
            let projected_chunk = lhs.clone() * &preimage_chunk;
            out.copy_block_from(
                &projected_chunk,
                0,
                overlap_start - col_start,
                0,
                0,
                lhs.row_size(),
                overlap_len,
            );
        }
        out
    }

    fn left_mul_preimage(&self, lhs: &M, id: &str, total_cols: usize) -> M {
        self.left_mul_preimage_columns(lhs, id, total_cols, 0, total_cols)
    }

    fn bgg_public_key_columns(&self) -> usize {
        DIAMOND_SECRET_SIZE
            .checked_mul(self.injector.params.modulus_digits())
            .expect("DiamondWE public-key column count overflow")
    }

    fn sample_bgg_public_key(&self, hash_key: [u8; 32], idx: usize) -> BggPublicKey<M> {
        let params = &self.injector.params;
        let mut tag = self.bgg_tag.clone();
        tag.extend_from_slice(b":witness_public_keys");
        let columns = self.bgg_public_key_columns();
        let total_keys =
            self.witness_size.checked_add(1).expect("DiamondWE public-key count overflow");
        assert!(
            idx < total_keys,
            "DiamondWE public-key index out of bounds: idx={}, total={}",
            idx,
            total_keys
        );
        let total_cols = columns
            .checked_mul(total_keys)
            .expect("DiamondWE public-key total column count overflow");
        let col_start =
            columns.checked_mul(idx).expect("DiamondWE public-key column start overflow");
        let matrix = HS::new().sample_hash_columns(
            params,
            hash_key,
            &tag,
            DIAMOND_SECRET_SIZE,
            total_cols,
            col_start,
            columns,
            DistType::FinRingDist,
        );
        BggPublicKey::new(matrix, true)
    }

    fn sample_k_public_key(&self, hash_key: [u8; 32]) -> BggPublicKey<M> {
        let params = &self.injector.params;
        let mut k_tag = self.bgg_tag.clone();
        k_tag.extend_from_slice(b":k_public_key");
        let k_matrix = HS::new().sample_hash(params, hash_key, &k_tag, 1, 1, DistType::FinRingDist);
        BggPublicKey::new(k_matrix, false)
    }

    fn sample_bgg_public_keys(
        &self,
        hash_key: [u8; 32],
    ) -> (BggPublicKey<M>, BggPublicKey<M>, Vec<BggPublicKey<M>>) {
        let one = self.sample_bgg_public_key(hash_key, 0);
        let witness_pubkeys = (0..self.witness_size)
            .map(|bit_idx| self.sample_bgg_public_key(hash_key, bit_idx + 1))
            .collect();
        let k_pubkey = self.sample_k_public_key(hash_key);
        (one, k_pubkey, witness_pubkeys)
    }

    fn output_preimage_target(
        &self,
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
        top.concat_rows(&[&bottom])
    }

    fn k_preimage_target(&self, k_pubkey: &BggPublicKey<M>) -> M {
        let params = &self.injector.params;
        assert_eq!(
            k_pubkey.matrix.size(),
            (DIAMOND_SECRET_SIZE, DIAMOND_SECRET_SIZE),
            "DiamondWE k public key must be a 1 x 1 public matrix"
        );
        let identity = M::identity(params, DIAMOND_SECRET_SIZE, None);
        k_pubkey.matrix.concat_rows(&[&identity])
    }

    fn decoder_preimage_target(&self, dec_pubkey_matrix: &M) -> M {
        let params = &self.injector.params;
        let bottom = M::zero(params, DIAMOND_SECRET_SIZE, dec_pubkey_matrix.col_size());
        dec_pubkey_matrix.concat_rows(&[&bottom])
    }

    fn lookup_base_preimage_target(&self, lookup_base_matrix: &M) -> M {
        assert_eq!(
            lookup_base_matrix.row_size(),
            DIAMOND_SECRET_SIZE,
            "DiamondWE lookup base matrix row count must match the Diamond secret size"
        );
        let params = &self.injector.params;
        let lookup_bottom = M::zero(params, DIAMOND_SECRET_SIZE, lookup_base_matrix.col_size());
        lookup_base_matrix.concat_rows(&[&lookup_bottom])
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
        let one_preimage_target =
            self.output_preimage_target(&one_pubkey, Some(&one_plaintext), None);
        self.sample_and_write_preimage(
            Self::one_preimage_id(),
            &preprocess_out,
            0,
            &one_preimage_target,
        );
        for (bit_idx, pubkey) in witness_pubkeys.iter().enumerate() {
            let digit_idx = bit_idx / self.injector.batch_bits();
            let bit_in_digit = bit_idx % self.injector.batch_bits();
            let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
            let preimage_target = self.output_preimage_target(pubkey, None, Some(&one_plaintext));
            self.sample_and_write_preimage(
                &Self::witness_preimage_id(bit_idx),
                &preprocess_out,
                state_idx,
                &preimage_target,
            );
        }
        let k_preimage_target = self.k_preimage_target(&k_pubkey);
        self.sample_and_write_preimage(
            Self::k_preimage_id(),
            &preprocess_out,
            0,
            &k_preimage_target,
        );
        if let Some(lookup_base_matrix) = self.enc_lookup_base_matrix.as_ref() {
            let lookup_base_preimage_target = self.lookup_base_preimage_target(lookup_base_matrix);
            self.sample_and_write_preimage(
                Self::enc_lookup_base_preimage_id(),
                &preprocess_out,
                0,
                &lookup_base_preimage_target,
            );
        }

        let r = self.sample_r(hash_key);
        let dec_term = one_pubkey - &out_pubkey;
        let dec_pubkey_matrix = k_pubkey.matrix + &dec_term.matrix.mul_decompose(&r);
        let decoder_preimage_target = self.decoder_preimage_target(&dec_pubkey_matrix);
        self.sample_and_write_preimage(
            Self::decoder_preimage_id(),
            &preprocess_out,
            0,
            &decoder_preimage_target,
        );

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
        let mut states = states.into_iter().map(Some).collect::<Vec<_>>();
        let root_state = states[0].take().expect("DiamondWE root state must be present");

        let one_pubkey = self.sample_bgg_public_key(ct.hash_key, 0);
        let mut input_encodings =
            Vec::with_capacity(self.witness_size.saturating_add(ct.instance.len()));
        for bit_idx in 0..self.witness_size {
            let pubkey = self.sample_bgg_public_key(ct.hash_key, bit_idx + 1);
            let digit_idx = bit_idx / self.injector.batch_bits();
            let bit_in_digit = bit_idx % self.injector.batch_bits();
            let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
            let plaintext = M::P::from_usize_to_constant(
                params,
                self.injector.digit_bit_value(witness_digits[digit_idx] as usize, bit_in_digit),
            );
            let state = states[state_idx]
                .take()
                .expect("DiamondWE witness state must be present exactly once");
            let vector = self.left_mul_preimage(
                &state,
                &Self::witness_preimage_id(bit_idx),
                pubkey.matrix.col_size(),
            );
            drop(state);
            input_encodings.push(self.injector.build_output_encoding(
                vector,
                pubkey,
                Some(plaintext),
            ));
        }
        drop(states);

        let one_encoding = self.injector.build_output_encoding(
            self.left_mul_preimage(
                &root_state,
                Self::one_preimage_id(),
                one_pubkey.matrix.col_size(),
            ),
            one_pubkey,
            Some(M::P::const_one(params)),
        );
        input_encodings.extend(self.instance_encodings(&one_encoding, &ct.instance));

        let out_encoding = {
            let owned_enc_lookup_evaluator = if let Some(factory) =
                self.enc_lookup_evaluator_factory.as_ref()
            {
                let lookup_base_cols = self
                    .enc_lookup_base_matrix
                    .as_ref()
                    .expect("DiamondWE encoding lookup-base preimage requires a lookup base matrix")
                    .col_size();
                let c_b0 = self.left_mul_preimage(
                    &root_state,
                    Self::enc_lookup_base_preimage_id(),
                    lookup_base_cols,
                );
                Some(factory(c_b0))
            } else {
                None
            };
            let enc_lookup_evaluator =
                owned_enc_lookup_evaluator.as_ref().or(self.enc_lookup_evaluator.as_ref());

            ct.circuit
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
                .expect("DiamondWE circuit must produce one output encoding")
        };
        let k_pubkey = self.sample_k_public_key(ct.hash_key);
        let k_encoding = self.injector.build_output_encoding(
            self.left_mul_preimage(&root_state, Self::k_preimage_id(), k_pubkey.matrix.col_size()),
            k_pubkey,
            None,
        );
        let r = self.sample_r(ct.hash_key);
        let r_decode_col = r.slice_columns(0, 1);
        drop(r);
        let BggEncoding { vector: one_vector, .. } = one_encoding;
        let BggEncoding { vector: out_vector, .. } = out_encoding;
        let dec_term_vector = one_vector - &out_vector;
        drop(out_vector);
        let dec_term_projection = dec_term_vector.mul_decompose(&r_decode_col);
        drop(dec_term_vector);
        drop(r_decode_col);
        let BggEncoding { vector: k_vector, .. } = k_encoding;
        let decoder_total_cols = k_vector.col_size();
        let mut dec_encoding_vector = k_vector.slice_columns(0, 1);
        drop(k_vector);
        dec_encoding_vector.add_in_place(&dec_term_projection);
        drop(dec_term_projection);
        let decoder = self.left_mul_preimage_columns(
            &root_state,
            Self::decoder_preimage_id(),
            decoder_total_cols,
            0,
            1,
        );
        drop(root_state);
        let noisy_plaintext = decoder - &dec_encoding_vector;
        drop(dec_encoding_vector);
        self.decode_noisy_bool(&noisy_plaintext)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use keccak_asm::Keccak256;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        bgg::sampler::BGGPublicKeySampler,
        circuit::PolyCircuit,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };

    fn diamond_we_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        key: &'static str,
        old_value: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let old_value = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            Self { key, old_value }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.old_value {
                unsafe { std::env::set_var(self.key, value) };
            } else {
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    type TestDiamondWE = DiamondWE<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
    >;

    #[test]
    fn test_diamond_we_column_window_public_keys_match_contiguous_sampler() {
        let params = DCRTPolyParams::default();
        let witness_size = 3;
        let hash_key = [7u8; 32];
        let tag = b"diamond_we_public_key_columns_test".to_vec();

        let injector = DiamondInjector::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(params.clone(), 1, 4, 2, 4.578, 0.0);
        let dir = tempdir().expect("temporary DiamondWE artifact directory should be created");
        let we = DiamondWE::new(injector, witness_size, dir.path(), tag.clone());

        let mut full_tag = tag;
        full_tag.extend_from_slice(b":witness_public_keys");
        let expected = BGGPublicKeySampler::<[u8; 32], DCRTPolyHashSampler<Keccak256>>::new(
            hash_key,
            DIAMOND_SECRET_SIZE,
        )
        .sample(&params, &full_tag, &vec![true; witness_size]);

        let actual_one = we.sample_bgg_public_key(hash_key, 0);
        assert_eq!(actual_one, expected[0]);
        for bit_idx in 0..witness_size {
            let actual = we.sample_bgg_public_key(hash_key, bit_idx + 1);
            assert_eq!(actual, expected[bit_idx + 1]);
        }

        let (one, k_pubkey, witness_pubkeys) = we.sample_bgg_public_keys(hash_key);
        assert_eq!(one, expected[0]);
        assert_eq!(witness_pubkeys, expected[1..].to_vec());

        let mut k_tag = we.bgg_tag.clone();
        k_tag.extend_from_slice(b":k_public_key");
        let expected_k = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            &k_tag,
            1,
            1,
            DistType::FinRingDist,
        );
        assert_eq!(k_pubkey, BggPublicKey::new(expected_k, false));
    }

    #[test]
    fn test_diamond_we_constant_one_circuit_round_trip() {
        let _env_lock = diamond_we_env_lock().lock().expect("DiamondWE env lock poisoned");
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
            >::new(params.clone(), 1, 4, 2, 4.578, 0.0);
            let dir = tempdir().expect("temporary DiamondWE artifact directory should be created");
            let we =
                DiamondWE::new(injector, witness_size, dir.path(), b"diamond_we_test".to_vec());
            let ct = we.enc(&msg, circuit.clone(), &instance);
            assert_eq!(we.dec(&ct, &witness), msg);
        }
    }

    #[test]
    fn test_diamond_we_chunked_preimages_round_trip_and_artifacts() {
        let _env_lock = diamond_we_env_lock().lock().expect("DiamondWE env lock poisoned");
        let _chunk_cols = EnvVarGuard::set("AUX_SAMPLING_CHUNK_WIDTH", "20");

        let params = DCRTPolyParams::default();
        let witness_size = 2;
        let instance = vec![true];
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        circuit.input(witness_size + instance.len());
        let one = circuit.const_one_gate();
        circuit.output(vec![one]);

        let witness = vec![false, true];
        let injector = DiamondInjector::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(params.clone(), 1, 4, 2, 4.578, 0.0);
        let dir = tempdir().expect("temporary DiamondWE artifact directory should be created");
        let we = DiamondWE::new(
            injector,
            witness_size,
            dir.path(),
            b"diamond_we_chunked_preimage_test".to_vec(),
        );

        let ct = we.enc(&true, circuit, &instance);
        assert_eq!(we.dec(&ct, &witness), true);

        let total_cols = params.modulus_digits();
        let chunk_cols = crate::env::aux_sampling_chunk_width().min(total_cols);
        let chunk_count = TestDiamondWE::preimage_chunk_count(total_cols, chunk_cols);
        assert_eq!(chunk_cols, 20);
        assert_eq!(chunk_count, 2);
        assert!(
            !we.matrix_path(TestDiamondWE::one_preimage_id()).exists(),
            "chunked preimage mode should not write the legacy full one-preimage artifact"
        );
        assert!(
            !dir.path()
                .join(format!("{}.preimage_chunks.json", TestDiamondWE::one_preimage_id()))
                .exists(),
            "chunked preimage mode should not write extra JSON metadata"
        );
        for chunk_idx in 0..chunk_count {
            let chunk_id =
                TestDiamondWE::preimage_chunk_id(TestDiamondWE::one_preimage_id(), chunk_idx);
            assert!(
                we.matrix_path(&chunk_id).exists(),
                "chunked one-preimage artifact {chunk_id} should exist"
            );
        }
    }

    #[test]
    fn test_diamond_we_witness_dependent_circuit_round_trip() {
        let _env_lock = diamond_we_env_lock().lock().expect("DiamondWE env lock poisoned");
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
            >::new(params.clone(), 1, 4, 2, 4.578, 0.0);
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
