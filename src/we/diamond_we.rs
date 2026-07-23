use digest::Digest;
use keccak_asm::Keccak256;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, path::PathBuf, sync::Arc};

use num_bigint::BigUint;
use tracing::info;

use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{Evaluable, PolyCircuit, serde::SerializablePolyCircuit},
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DiamondWEEncCheckpoint {
    msg: bool,
    instance: Vec<bool>,
    target_witness: Option<Vec<bool>>,
    circuit_json: String,
    hash_key: [u8; 32],
    preimage_chunk_width: usize,
    bgg_tag: Vec<u8>,
    witness_size: usize,
    injector_input_count: usize,
    injector_base: usize,
    injector_batch_bits: usize,
    ring_dimension: u32,
    base_bits: u32,
    modulus_bits: usize,
    modulus_digits: usize,
    crt_moduli: Vec<u64>,
    crt_modulus_bits: usize,
    crt_depth: usize,
    trapdoor_sigma_bits: u64,
    error_sigma_bits: u64,
    lookup_base_hash: Option<[u8; 32]>,
    matrix_type: String,
    uniform_sampler_type: String,
    hash_sampler_type: String,
    trapdoor_sampler_type: String,
    pk_lookup_evaluator_type: String,
    pk_slot_transfer_evaluator_type: String,
    enc_lookup_evaluator_type: String,
    enc_slot_transfer_evaluator_type: String,
    evaluator_checkpoint_id: Vec<u8>,
    pk_lookup_evaluator_enabled: bool,
    pk_slot_transfer_evaluator_enabled: bool,
    enc_lookup_evaluator_factory_enabled: bool,
    enc_lookup_evaluator_enabled: bool,
    enc_slot_transfer_evaluator_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DiamondWEStorageAccounting {
    transition_preimage_bytes: u128,
    final_preimage_bytes: u128,
    total_preimage_bytes: u128,
}

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
    pub evaluator_checkpoint_id: Vec<u8>,
    pub pk_lookup_evaluator: Option<PKPE>,
    pub pk_slot_transfer_evaluator: Option<PKST>,
    pub target_witness: Option<Vec<bool>>,
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
        target_witness: Option<Vec<bool>>,
    ) -> Self {
        Self {
            injector,
            witness_size,
            artifact_dir: artifact_dir.into(),
            bgg_tag,
            evaluator_checkpoint_id: Vec::new(),
            pk_lookup_evaluator: None,
            pk_slot_transfer_evaluator: None,
            enc_lookup_base_matrix: None,
            enc_lookup_evaluator_factory: None,
            target_witness,
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
        evaluator_checkpoint_id: Vec<u8>,
        target_witness: Option<Vec<bool>>,
        pk_lookup_evaluator: Option<PKPE>,
        pk_slot_transfer_evaluator: Option<PKST>,
        enc_lookup_base_matrix: Option<M>,
        enc_lookup_evaluator_factory: Option<Arc<dyn Fn(M) -> ENCPE + Send + Sync>>,
        enc_lookup_evaluator: Option<ENCPE>,
        enc_slot_transfer_evaluator: Option<ENCST>,
    ) -> Self {
        let has_evaluator_state = pk_lookup_evaluator.is_some() ||
            pk_slot_transfer_evaluator.is_some() ||
            enc_lookup_evaluator_factory.is_some() ||
            enc_lookup_evaluator.is_some() ||
            enc_slot_transfer_evaluator.is_some();
        assert!(
            !has_evaluator_state || !evaluator_checkpoint_id.is_empty(),
            "DiamondWE evaluator_checkpoint_id must identify evaluator configuration and state"
        );
        Self {
            injector,
            witness_size,
            artifact_dir: artifact_dir.into(),
            bgg_tag,
            evaluator_checkpoint_id,
            pk_lookup_evaluator,
            pk_slot_transfer_evaluator,
            enc_lookup_base_matrix,
            target_witness,
            enc_lookup_evaluator_factory,
            enc_lookup_evaluator,
            enc_slot_transfer_evaluator,
            _m: PhantomData,
        }
    }

    fn matrix_path(&self, id: &str) -> PathBuf {
        self.artifact_dir.join(format!("{id}.matrixbin"))
    }

    fn enc_checkpoint_path(&self) -> PathBuf {
        self.artifact_dir.join("diamond_we_enc_checkpoint.json")
    }

    fn enc_complete_path(&self) -> PathBuf {
        self.artifact_dir.join("diamond_we_enc_complete")
    }

    fn current_enc_checkpoint(
        &self,
        msg: bool,
        circuit: &PolyCircuit<M::P>,
        instance: &[bool],
        hash_key: [u8; 32],
    ) -> DiamondWEEncCheckpoint {
        let params = &self.injector.params;
        let (crt_moduli, crt_modulus_bits, crt_depth) = params.to_crt();
        let lookup_base_hash = self.enc_lookup_base_matrix.as_ref().map(|matrix| {
            let digest = Keccak256::digest(matrix.to_compact_bytes());
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&digest);
            hash
        });
        DiamondWEEncCheckpoint {
            msg,
            instance: instance.to_vec(),
            target_witness: self.target_witness.clone(),
            circuit_json: SerializablePolyCircuit::from_circuit(circuit.clone()).to_json_str(),
            hash_key,
            preimage_chunk_width: crate::env::aux_sampling_chunk_width(),
            bgg_tag: self.bgg_tag.clone(),
            witness_size: self.witness_size,
            injector_input_count: self.injector.input_count,
            injector_base: self.injector.base,
            injector_batch_bits: self.injector.batch_bits,
            ring_dimension: params.ring_dimension(),
            base_bits: params.base_bits(),
            modulus_bits: params.modulus_bits(),
            modulus_digits: params.modulus_digits(),
            crt_moduli,
            crt_modulus_bits,
            crt_depth,
            trapdoor_sigma_bits: self.injector.trapdoor_sigma.to_bits(),
            error_sigma_bits: self.injector.error_sigma.to_bits(),
            lookup_base_hash,
            matrix_type: std::any::type_name::<M>().to_owned(),
            uniform_sampler_type: std::any::type_name::<US>().to_owned(),
            hash_sampler_type: std::any::type_name::<HS>().to_owned(),
            trapdoor_sampler_type: std::any::type_name::<TS>().to_owned(),
            pk_lookup_evaluator_type: std::any::type_name::<PKPE>().to_owned(),
            pk_slot_transfer_evaluator_type: std::any::type_name::<PKST>().to_owned(),
            enc_lookup_evaluator_type: std::any::type_name::<ENCPE>().to_owned(),
            enc_slot_transfer_evaluator_type: std::any::type_name::<ENCST>().to_owned(),
            evaluator_checkpoint_id: self.evaluator_checkpoint_id.clone(),
            pk_lookup_evaluator_enabled: self.pk_lookup_evaluator.is_some(),
            pk_slot_transfer_evaluator_enabled: self.pk_slot_transfer_evaluator.is_some(),
            enc_lookup_evaluator_factory_enabled: self.enc_lookup_evaluator_factory.is_some(),
            enc_lookup_evaluator_enabled: self.enc_lookup_evaluator.is_some(),
            enc_slot_transfer_evaluator_enabled: self.enc_slot_transfer_evaluator.is_some(),
        }
    }

    fn prepare_enc_checkpoint(
        &self,
        msg: bool,
        circuit: &PolyCircuit<M::P>,
        instance: &[bool],
    ) -> [u8; 32] {
        std::fs::create_dir_all(&self.artifact_dir).unwrap_or_else(|err| {
            panic!(
                "DiamondWE failed to create artifact directory {}: {err}",
                self.artifact_dir.display()
            )
        });
        assert!(
            !self.enc_complete_path().exists(),
            "DiamondWE encryption session is already complete; use a new artifact directory for a fresh end-to-end measurement"
        );
        let path = self.enc_checkpoint_path();
        if path.exists() {
            let bytes = std::fs::read(&path).unwrap_or_else(|err| {
                panic!("DiamondWE failed to read checkpoint {}: {err}", path.display())
            });
            let checkpoint: DiamondWEEncCheckpoint = serde_json::from_slice(&bytes)
                .expect("DiamondWE encryption checkpoint should decode");
            let expected = self.current_enc_checkpoint(msg, circuit, instance, checkpoint.hash_key);
            assert_eq!(
                checkpoint, expected,
                "DiamondWE encryption checkpoint configuration mismatch; use a separate artifact directory"
            );
            return checkpoint.hash_key;
        }
        let checkpoint = self.current_enc_checkpoint(msg, circuit, instance, rand::random());
        let bytes = serde_json::to_vec_pretty(&checkpoint)
            .expect("DiamondWE encryption checkpoint should serialize");
        crate::utils::write_bytes_atomic(&path, &bytes);
        checkpoint.hash_key
    }

    fn session_prefix(hash_key: [u8; 32]) -> String {
        let hash = hash_key.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        format!("diamond_we_session_{hash}")
    }

    fn session_preimage_id(hash_key: [u8; 32], id: &str) -> String {
        format!("{}_{}", Self::session_prefix(hash_key), id)
    }

    fn storage_accounting_path(&self, hash_key: [u8; 32]) -> PathBuf {
        self.artifact_dir
            .join(format!("{}_storage_accounting.json", Self::session_prefix(hash_key)))
    }

    fn mark_enc_complete(&self) {
        crate::utils::write_bytes_atomic(&self.enc_complete_path(), b"complete\n");
    }

    fn write_matrix(&self, id: &str, matrix: &M) -> u64 {
        std::fs::create_dir_all(&self.artifact_dir).unwrap_or_else(|err| {
            panic!(
                "DiamondWE failed to create artifact directory {}: {err}",
                self.artifact_dir.display()
            )
        });
        let bytes = matrix.to_compact_bytes();
        let stored_bytes =
            u64::try_from(bytes.len()).expect("DiamondWE matrix artifact size must fit u64");
        crate::utils::write_bytes_atomic(&self.matrix_path(id), &bytes);
        stored_bytes
    }

    fn read_matrix(&self, id: &str) -> M {
        let bytes = std::fs::read(self.matrix_path(id))
            .unwrap_or_else(|err| panic!("DiamondWE failed to read matrix {id}: {err}"));
        M::from_compact_bytes(&self.injector.params, &bytes)
    }

    fn matrix_from_staging_bytes(&self, bytes: &[u8]) -> M {
        M::from_cpu_staging_bytes(&self.injector.params, bytes)
    }

    fn online_eval_state_bytes(
        &self,
        ct: &DiamondWECiphertext<M, TS::Trapdoor>,
        witness_digits: &[u32],
    ) -> Vec<Vec<u8>> {
        let states = self.injector.online_eval_staging_bytes(
            &self.artifact_dir,
            &ct.preprocess_out,
            witness_digits,
        );
        assert_eq!(
            states.len(),
            1 + self.witness_size,
            "DiamondWE final Diamond state count mismatch"
        );
        states
    }

    fn remove_matrix_if_exists(&self, id: &str) {
        let path = self.matrix_path(id);
        #[cfg(test)]
        if path.exists() {
            crate::utils::observe_file_before_delete(&path);
        }
        match std::fs::remove_file(path) {
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
        let trapdoor = TS::trapdoor_from_bytes(
            &self.injector.params,
            preprocess_out.final_trapdoor_bytes(state_idx),
        )
        .expect("DiamondInjector final trapdoor checkpoint must decode");
        let public_matrix = preprocess_out.final_public_matrix(&self.injector.params, state_idx);
        TS::new(&self.injector.params, self.injector.trapdoor_sigma).preimage(
            &self.injector.params,
            &trapdoor,
            &public_matrix,
            target,
        )
    }

    fn sample_and_write_preimage_columns(
        &self,
        id: &str,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        state_idx: usize,
        total_cols: usize,
        mut target_columns: impl FnMut(usize, usize) -> M,
    ) {
        if let Some(chunk_cols) = Self::preimage_chunk_cols(total_cols) {
            let chunk_count = Self::preimage_chunk_count(total_cols, chunk_cols);
            self.remove_matrix_if_exists(id);
            info!(
                id,
                total_cols, chunk_cols, chunk_count, "diamond we final preimage: sampling chunks"
            );
            for chunk_idx in 0..chunk_count {
                let chunk_id = Self::preimage_chunk_id(id, chunk_idx);
                if self.matrix_path(&chunk_id).exists() {
                    info!(
                        id,
                        chunk_idx = chunk_idx + 1,
                        chunk_count,
                        "diamond we final preimage: resumed chunk"
                    );
                    continue;
                }
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
                let target_chunk = target_columns(col_start, col_len);
                let preimage =
                    self.sample_target_preimage(preprocess_out, state_idx, &target_chunk);
                self.write_matrix(&chunk_id, &preimage);
            }
        } else {
            if self.matrix_path(id).exists() {
                return;
            }
            let target = target_columns(0, total_cols);
            let preimage = self.sample_target_preimage(preprocess_out, state_idx, &target);
            self.write_matrix(id, &preimage);
        }
    }

    fn write_storage_accounting(
        &self,
        hash_key: [u8; 32],
        transition_preimage_bytes: u128,
    ) -> DiamondWEStorageAccounting {
        let prefix = Self::session_prefix(hash_key);
        let retained_final_bytes = std::fs::read_dir(&self.artifact_dir)
            .unwrap_or_else(|err| {
                panic!(
                    "DiamondWE failed to list artifact directory {}: {err}",
                    self.artifact_dir.display()
                )
            })
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with(&prefix) && name.ends_with(".matrixbin")
            })
            .map(|entry| {
                u128::from(
                    entry
                        .metadata()
                        .unwrap_or_else(|err| {
                            panic!("DiamondWE failed to stat {}: {err}", entry.path().display())
                        })
                        .len(),
                )
            })
            .sum::<u128>();
        let final_preimage_bytes = retained_final_bytes;
        let accounting = DiamondWEStorageAccounting {
            transition_preimage_bytes,
            final_preimage_bytes,
            total_preimage_bytes: transition_preimage_bytes
                .checked_add(final_preimage_bytes)
                .expect("DiamondWE total preimage byte count overflow"),
        };
        let bytes = serde_json::to_vec_pretty(&accounting)
            .expect("DiamondWE storage accounting should serialize");
        crate::utils::write_bytes_atomic(&self.storage_accounting_path(hash_key), &bytes);
        accounting
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

    fn k_public_key_columns(&self) -> usize {
        self.bgg_public_key_columns()
    }

    fn sample_k_public_key(&self, hash_key: [u8; 32]) -> BggPublicKey<M> {
        let params = &self.injector.params;
        let mut k_tag = self.bgg_tag.clone();
        k_tag.extend_from_slice(b":k_public_key");
        let k_matrix = HS::new().sample_hash(
            params,
            hash_key,
            &k_tag,
            1,
            self.k_public_key_columns(),
            DistType::FinRingDist,
        );
        BggPublicKey::new(k_matrix, false)
    }

    #[cfg(test)]
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

    fn gadget_row_columns(params: &<M::P as Poly>::Params, col_start: usize, col_len: usize) -> M {
        assert_eq!(
            DIAMOND_SECRET_SIZE, 1,
            "DiamondWE gadget row chunks assume the current one-row Diamond secret"
        );
        let col_end =
            col_start.checked_add(col_len).expect("DiamondWE gadget column range overflow");
        assert!(
            col_end <= params.modulus_digits(),
            "DiamondWE gadget column range out of bounds: col_start={col_start}, col_len={col_len}, modulus_digits={}",
            params.modulus_digits()
        );
        // Full DCRT gadget columns are tower-local CRT idempotents. A global
        // base power is not equivalent once the modulus has multiple towers.
        M::gadget_matrix(params, DIAMOND_SECRET_SIZE).slice_columns(col_start, col_end)
    }

    fn output_preimage_target_columns(
        &self,
        pubkey: &BggPublicKey<M>,
        top_gadget_plaintext: Option<&M::P>,
        bottom_gadget_plaintext: Option<&M::P>,
        col_start: usize,
        col_len: usize,
    ) -> M {
        let params = &self.injector.params;
        let col_end =
            col_start.checked_add(col_len).expect("DiamondWE target column range overflow");
        let mut top = pubkey.matrix.slice_columns(col_start, col_end);
        if let Some(plaintext) = top_gadget_plaintext {
            let gadget = Self::gadget_row_columns(params, col_start, col_len);
            top = top - &(gadget * plaintext);
        }
        let mut bottom = M::zero(params, DIAMOND_SECRET_SIZE, col_len);
        if let Some(plaintext) = bottom_gadget_plaintext {
            let gadget = Self::gadget_row_columns(params, col_start, col_len);
            bottom = bottom - &(gadget * plaintext);
        }
        top.concat_rows_owned(vec![bottom])
    }

    fn k_preimage_target_columns(
        &self,
        k_pubkey: &BggPublicKey<M>,
        col_start: usize,
        col_len: usize,
    ) -> M {
        let params = &self.injector.params;
        let columns = self.k_public_key_columns();
        assert_eq!(
            k_pubkey.matrix.size(),
            (DIAMOND_SECRET_SIZE, columns),
            "DiamondWE k public key must match k_public_key_columns"
        );
        let col_end =
            col_start.checked_add(col_len).expect("DiamondWE k target column range overflow");
        assert!(
            col_end <= columns,
            "DiamondWE k target column range out of bounds: col_start={col_start}, col_len={col_len}, columns={columns}"
        );
        let q: Arc<BigUint> = params.modulus().into();
        let scale = M::P::from_biguint_to_constant(params, q.as_ref() / 2u32);
        let zero = M::P::const_zero(params);
        let selector = M::from_poly_vec_row(
            params,
            (col_start..col_end)
                .map(|column| if column == 0 { scale.clone() } else { zero.clone() })
                .collect(),
        );
        k_pubkey.matrix.slice_columns(col_start, col_end).concat_rows_owned(vec![selector])
    }

    fn decoder_preimage_target_columns(
        &self,
        dec_pubkey_matrix: &M,
        col_start: usize,
        col_len: usize,
    ) -> M {
        let params = &self.injector.params;
        let col_end =
            col_start.checked_add(col_len).expect("DiamondWE decoder target range overflow");
        let top = dec_pubkey_matrix.slice_columns(col_start, col_end);
        let bottom = M::zero(params, DIAMOND_SECRET_SIZE, col_len);
        top.concat_rows_owned(vec![bottom])
    }

    fn lookup_base_preimage_target_columns(
        &self,
        lookup_base_matrix: &M,
        col_start: usize,
        col_len: usize,
    ) -> M {
        assert_eq!(
            lookup_base_matrix.row_size(),
            DIAMOND_SECRET_SIZE,
            "DiamondWE lookup base matrix row count must match the Diamond secret size"
        );
        let params = &self.injector.params;
        let col_end =
            col_start.checked_add(col_len).expect("DiamondWE lookup target range overflow");
        let top = lookup_base_matrix.slice_columns(col_start, col_end);
        let bottom = M::zero(params, DIAMOND_SECRET_SIZE, col_len);
        top.concat_rows_owned(vec![bottom])
    }

    fn sample_r_columns(&self, hash_key: [u8; 32], col_start: usize, col_len: usize) -> M {
        let mut tag = self.bgg_tag.clone();
        tag.extend_from_slice(b":r");
        HS::new().sample_hash_columns(
            &self.injector.params,
            hash_key,
            &tag,
            1,
            self.k_public_key_columns(),
            col_start,
            col_len,
            DistType::FinRingDist,
        )
    }

    fn sample_r(&self, hash_key: [u8; 32]) -> M {
        self.sample_r_columns(hash_key, 0, self.k_public_key_columns())
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

    fn centered_distance_to_zero(
        value: &BigUint,
        modulus: &BigUint,
        half_modulus: &BigUint,
    ) -> BigUint {
        if value > half_modulus { modulus - value } else { value.clone() }
    }

    fn log_noisy_plaintext_error_diagnostic(&self, noisy_plaintext: &M) {
        let Some(expected_msg) =
            std::env::var("DIAMOND_WE_GPU_BENCH_EXPECTED_MSG").ok().and_then(|raw| {
                match raw.as_str() {
                    "true" => Some(true),
                    "false" => Some(false),
                    _ => None,
                }
            })
        else {
            return;
        };
        let selected_bound_bits =
            std::env::var("DIAMOND_WE_GPU_BENCH_SELECTED_NOISY_PLAINTEXT_ERROR_BITS")
                .ok()
                .and_then(|raw| raw.parse::<u64>().ok());
        let q = self.injector.params.modulus();
        let q: std::sync::Arc<BigUint> = q.into();
        let q = q.as_ref();
        let half_q = q / 2u32;
        let coeffs = noisy_plaintext.entry(0, 0).coeffs_biguints();
        let mut max_error = BigUint::from(0u32);
        let mut decode_coeff_error = BigUint::from(0u32);
        for (coeff_idx, coeff) in coeffs.iter().enumerate() {
            let error = if expected_msg && coeff_idx == 0 {
                if coeff >= &half_q { coeff - &half_q } else { &half_q - coeff }
            } else {
                Self::centered_distance_to_zero(coeff, q, &half_q)
            };
            if coeff_idx == 0 {
                decode_coeff_error = error.clone();
            }
            if error > max_error {
                max_error = error;
            }
        }
        let decode_coeff_error_bits = decode_coeff_error.bits();
        let max_poly_error_bits = max_error.bits();
        let within_selected_bound =
            selected_bound_bits.map(|bound_bits| max_poly_error_bits <= bound_bits);
        info!(
            expected_msg,
            decode_coeff_error_bits,
            max_poly_error_bits,
            selected_bound_bits,
            within_selected_bound,
            "diamond we dec: noisy plaintext error diagnostic"
        );
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
        if let Some(target_witness) = self.target_witness.as_ref() {
            assert_eq!(
                target_witness.len(),
                self.witness_size,
                "DiamondWE target witness length must match witness_size"
            );
        }
        let retained_input_digits =
            self.target_witness.as_ref().map(|witness| self.pack_witness_digits(witness));
        let hash_key = self.prepare_enc_checkpoint(*msg, &circuit, instance);
        let params = &self.injector.params;
        let k = if *msg { M::P::const_one(params) } else { M::P::const_zero(params) };
        let preprocess_out =
            self.injector.preprocess(&self.artifact_dir, &k, retained_input_digits.as_deref());
        let one_pubkey = self.sample_bgg_public_key(hash_key, 0);
        let one_pubkey_compact = one_pubkey.clone().to_compact();
        let zero_pubkey_compact = one_pubkey.small_scalar_mul(params, &[0]).to_compact();
        let mut input_pubkey_compacts = Vec::with_capacity(circuit.num_input());
        for bit_idx in 0..self.witness_size {
            input_pubkey_compacts
                .push(self.sample_bgg_public_key(hash_key, bit_idx + 1).to_compact());
        }
        input_pubkey_compacts.extend(instance.iter().map(|bit| {
            if *bit { one_pubkey_compact.clone() } else { zero_pubkey_compact.clone() }
        }));

        let out_pubkey = circuit
            .eval_from_compacts(
                params,
                one_pubkey_compact,
                input_pubkey_compacts,
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
        let one_preimage_id = Self::session_preimage_id(hash_key, Self::one_preimage_id());
        self.sample_and_write_preimage_columns(
            &one_preimage_id,
            &preprocess_out,
            0,
            one_pubkey.matrix.col_size(),
            |col_start, col_len| {
                self.output_preimage_target_columns(
                    &one_pubkey,
                    Some(&one_plaintext),
                    None,
                    col_start,
                    col_len,
                )
            },
        );
        for bit_idx in 0..self.witness_size {
            let pubkey = self.sample_bgg_public_key(hash_key, bit_idx + 1);
            let digit_idx = bit_idx / self.injector.batch_bits();
            let bit_in_digit = bit_idx % self.injector.batch_bits();
            let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
            let witness_preimage_id =
                Self::session_preimage_id(hash_key, &Self::witness_preimage_id(bit_idx));
            self.sample_and_write_preimage_columns(
                &witness_preimage_id,
                &preprocess_out,
                state_idx,
                pubkey.matrix.col_size(),
                |col_start, col_len| {
                    self.output_preimage_target_columns(
                        &pubkey,
                        None,
                        Some(&one_plaintext),
                        col_start,
                        col_len,
                    )
                },
            );
        }
        let k_pubkey = self.sample_k_public_key(hash_key);
        let k_preimage_id = Self::session_preimage_id(hash_key, Self::k_preimage_id());
        self.sample_and_write_preimage_columns(
            &k_preimage_id,
            &preprocess_out,
            0,
            k_pubkey.matrix.col_size(),
            |col_start, col_len| self.k_preimage_target_columns(&k_pubkey, col_start, col_len),
        );
        if let Some(lookup_base_matrix) = self.enc_lookup_base_matrix.as_ref() {
            let lookup_preimage_id =
                Self::session_preimage_id(hash_key, Self::enc_lookup_base_preimage_id());
            self.sample_and_write_preimage_columns(
                &lookup_preimage_id,
                &preprocess_out,
                0,
                lookup_base_matrix.col_size(),
                |col_start, col_len| {
                    self.lookup_base_preimage_target_columns(lookup_base_matrix, col_start, col_len)
                },
            );
        }

        let r = self.sample_r(hash_key);
        let dec_term = one_pubkey - &out_pubkey;
        let dec_pubkey_matrix = k_pubkey.matrix + &dec_term.matrix.mul_decompose(&r);
        let decoder_preimage_id = Self::session_preimage_id(hash_key, Self::decoder_preimage_id());
        self.sample_and_write_preimage_columns(
            &decoder_preimage_id,
            &preprocess_out,
            0,
            dec_pubkey_matrix.col_size(),
            |col_start, col_len| {
                self.decoder_preimage_target_columns(&dec_pubkey_matrix, col_start, col_len)
            },
        );
        let transition_preimage_bytes = self.injector.generated_transition_preimage_bytes(
            &self.artifact_dir,
            retained_input_digits.as_deref(),
        );
        let storage_accounting = self.write_storage_accounting(hash_key, transition_preimage_bytes);
        info!(
            transition_preimage_bytes = %storage_accounting.transition_preimage_bytes,
            final_preimage_bytes = %storage_accounting.final_preimage_bytes,
            total_preimage_bytes = %storage_accounting.total_preimage_bytes,
            "diamond we enc: actual saved preimage bytes"
        );
        self.mark_enc_complete();

        DiamondWECiphertext { circuit, instance: instance.clone(), hash_key, preprocess_out }
    }

    fn dec(&self, ct: &Self::Ciphertext, witness: &Self::Wtns) -> Self::Msg {
        assert_eq!(ct.circuit.num_output(), 1, "DiamondWE currently requires one circuit output");
        if let Some(target_witness) = self.target_witness.as_ref() {
            assert_eq!(
                witness, target_witness,
                "DiamondWE fixed-witness artifacts can only decrypt the witness selected at encryption time"
            );
        }
        assert_eq!(
            self.witness_size + ct.instance.len(),
            ct.circuit.num_input(),
            "DiamondWE witness_size + instance length must match the circuit input size"
        );
        let started = std::time::Instant::now();
        let params = &self.injector.params;
        let one_preimage_id = Self::session_preimage_id(ct.hash_key, Self::one_preimage_id());
        let witness_digits = self.pack_witness_digits(witness);
        info!(
            witness_size = self.witness_size,
            instance_len = ct.instance.len(),
            circuit_inputs = ct.circuit.num_input(),
            "diamond we dec: begin online_eval"
        );
        let mut state_bytes = self
            .online_eval_state_bytes(ct, &witness_digits)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let root_state_bytes =
            state_bytes[0].take().expect("DiamondWE root state bytes must be present");
        let state_byte_lens = state_bytes
            .iter()
            .enumerate()
            .filter_map(|(idx, bytes)| bytes.as_ref().map(|bytes| (idx, bytes.len())))
            .collect::<Vec<_>>();
        info!(
            elapsed_ms = started.elapsed().as_millis(),
            state_count = state_bytes.len(),
            root_state_bytes = root_state_bytes.len(),
            ?state_byte_lens,
            bgg_public_key_columns = self.bgg_public_key_columns(),
            k_public_key_columns = self.k_public_key_columns(),
            "diamond we dec: online_eval states staged"
        );

        info!(
            rows = DIAMOND_SECRET_SIZE,
            cols = self.bgg_public_key_columns(),
            "diamond we dec: sample one public key begin"
        );
        let one_pubkey = self.sample_bgg_public_key(ct.hash_key, 0);
        info!(
            rows = one_pubkey.matrix.row_size(),
            cols = one_pubkey.matrix.col_size(),
            "diamond we dec: sample one public key done"
        );
        let mut input_encoding_compacts =
            Vec::with_capacity(self.witness_size.saturating_add(ct.instance.len()));
        for bit_idx in 0..self.witness_size {
            info!(
                bit_idx,
                rows = DIAMOND_SECRET_SIZE,
                cols = self.bgg_public_key_columns(),
                "diamond we dec: sample witness public key begin"
            );
            let pubkey = self.sample_bgg_public_key(ct.hash_key, bit_idx + 1);
            info!(
                bit_idx,
                rows = pubkey.matrix.row_size(),
                cols = pubkey.matrix.col_size(),
                "diamond we dec: sample witness public key done"
            );
            let digit_idx = bit_idx / self.injector.batch_bits();
            let bit_in_digit = bit_idx % self.injector.batch_bits();
            let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
            let plaintext = M::P::from_usize_to_constant(
                params,
                self.injector.digit_bit_value(witness_digits[digit_idx] as usize, bit_in_digit),
            );
            let staged = state_bytes[state_idx]
                .take()
                .expect("DiamondWE witness state bytes must be present exactly once");
            info!(
                bit_idx,
                state_idx,
                staged_bytes = staged.len(),
                "diamond we dec: reload witness state begin"
            );
            let state = self.matrix_from_staging_bytes(staged.as_ref());
            info!(
                bit_idx,
                state_idx,
                rows = state.row_size(),
                cols = state.col_size(),
                "diamond we dec: reload witness state done"
            );
            info!(
                bit_idx,
                preimage_id = %Self::session_preimage_id(
                    ct.hash_key,
                    &Self::witness_preimage_id(bit_idx),
                ),
                total_cols = pubkey.matrix.col_size(),
                "diamond we dec: witness left_mul_preimage begin"
            );
            let witness_preimage_id =
                Self::session_preimage_id(ct.hash_key, &Self::witness_preimage_id(bit_idx));
            let vector =
                self.left_mul_preimage(&state, &witness_preimage_id, pubkey.matrix.col_size());
            info!(
                bit_idx,
                rows = vector.row_size(),
                cols = vector.col_size(),
                "diamond we dec: witness left_mul_preimage done"
            );
            drop(state);
            info!(bit_idx, "diamond we dec: build witness output encoding begin");
            let encoding = self.injector.build_output_encoding(vector, pubkey, Some(plaintext));
            info!(bit_idx, "diamond we dec: compact witness output encoding begin");
            input_encoding_compacts.push(encoding.to_compact());
            info!(
                bit_idx,
                input_encoding_count = input_encoding_compacts.len(),
                "diamond we dec: build witness output encoding done"
            );
        }
        drop(state_bytes);
        info!(
            input_encoding_count = input_encoding_compacts.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "diamond we dec: witness encodings complete"
        );

        info!(staged_bytes = root_state_bytes.len(), "diamond we dec: reload root state begin");
        let root_state = self.matrix_from_staging_bytes(&root_state_bytes);
        info!(
            rows = root_state.row_size(),
            cols = root_state.col_size(),
            "diamond we dec: reload root state done"
        );
        info!(
            total_cols = one_pubkey.matrix.col_size(),
            "diamond we dec: one left_mul_preimage begin"
        );
        let one_vector =
            self.left_mul_preimage(&root_state, &one_preimage_id, one_pubkey.matrix.col_size());
        info!(
            rows = one_vector.row_size(),
            cols = one_vector.col_size(),
            "diamond we dec: one left_mul_preimage done"
        );
        let owned_enc_lookup_evaluator = if let Some(factory) =
            self.enc_lookup_evaluator_factory.as_ref()
        {
            let lookup_base_cols = self
                .enc_lookup_base_matrix
                .as_ref()
                .expect("DiamondWE encoding lookup-base preimage requires a lookup base matrix")
                .col_size();
            info!(lookup_base_cols, "diamond we dec: lookup-base left_mul_preimage begin");
            let lookup_preimage_id =
                Self::session_preimage_id(ct.hash_key, Self::enc_lookup_base_preimage_id());
            let c_b0 = self.left_mul_preimage(&root_state, &lookup_preimage_id, lookup_base_cols);
            info!(
                rows = c_b0.row_size(),
                cols = c_b0.col_size(),
                "diamond we dec: lookup-base left_mul_preimage done"
            );
            info!("diamond we dec: build owned encoding lookup evaluator begin");
            let evaluator = factory(c_b0);
            info!("diamond we dec: build owned encoding lookup evaluator done");
            Some(evaluator)
        } else {
            info!("diamond we dec: no owned encoding lookup evaluator needed");
            None
        };
        let k_public_key_columns = self.k_public_key_columns();
        info!(
            total_cols = k_public_key_columns,
            col_start = 0usize,
            col_len = 1usize,
            "diamond we dec: k left_mul_preimage_columns begin"
        );
        let k_preimage_id = Self::session_preimage_id(ct.hash_key, Self::k_preimage_id());
        let k_vector =
            self.left_mul_preimage_columns(&root_state, &k_preimage_id, k_public_key_columns, 0, 1);
        info!(
            rows = k_vector.row_size(),
            cols = k_vector.col_size(),
            "diamond we dec: k left_mul_preimage_columns done"
        );
        info!(
            total_cols = k_public_key_columns,
            col_start = 0usize,
            col_len = 1usize,
            "diamond we dec: decoder left_mul_preimage_columns begin"
        );
        let decoder_preimage_id =
            Self::session_preimage_id(ct.hash_key, Self::decoder_preimage_id());
        let decoder = self.left_mul_preimage_columns(
            &root_state,
            &decoder_preimage_id,
            k_public_key_columns,
            0,
            1,
        );
        info!(
            rows = decoder.row_size(),
            cols = decoder.col_size(),
            "diamond we dec: decoder left_mul_preimage_columns done"
        );
        drop(root_state);
        info!("diamond we dec: root state dropped");

        info!("diamond we dec: build one output encoding begin");
        let one_encoding = self.injector.build_output_encoding(
            one_vector,
            one_pubkey,
            Some(M::P::const_one(params)),
        );
        info!(
            vector_rows = one_encoding.vector.row_size(),
            vector_cols = one_encoding.vector.col_size(),
            pubkey_rows = one_encoding.pubkey.matrix.row_size(),
            pubkey_cols = one_encoding.pubkey.matrix.col_size(),
            "diamond we dec: build one output encoding done"
        );
        info!(
            instance_len = ct.instance.len(),
            input_encoding_count = input_encoding_compacts.len(),
            "diamond we dec: compact input encodings begin"
        );
        let zero_encoding_compact = one_encoding.small_scalar_mul(params, &[0]).to_compact();
        let one_encoding_compact = one_encoding.to_compact();
        let witness_input_count = input_encoding_compacts.len();
        input_encoding_compacts.extend(ct.instance.iter().map(|bit| {
            if *bit { one_encoding_compact.clone() } else { zero_encoding_compact.clone() }
        }));
        info!(
            witness_input_count,
            instance_len = ct.instance.len(),
            input_encoding_count = input_encoding_compacts.len(),
            "diamond we dec: compact input encodings done"
        );

        let enc_lookup_evaluator =
            owned_enc_lookup_evaluator.as_ref().or(self.enc_lookup_evaluator.as_ref());
        info!(
            input_encoding_count = input_encoding_compacts.len(),
            has_enc_lookup_evaluator = enc_lookup_evaluator.is_some(),
            has_enc_slot_transfer_evaluator = self.enc_slot_transfer_evaluator.is_some(),
            elapsed_ms = started.elapsed().as_millis(),
            "diamond we dec: circuit eval begin"
        );
        let out_encoding = ct
            .circuit
            .eval_from_compacts(
                params,
                one_encoding_compact.clone(),
                input_encoding_compacts,
                enc_lookup_evaluator,
                self.enc_slot_transfer_evaluator
                    .as_ref()
                    .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<BggEncoding<M>>),
                None,
            )
            .into_iter()
            .next()
            .expect("DiamondWE circuit must produce one output encoding");
        let one_encoding = BggEncoding::<M>::from_compact(params, &one_encoding_compact);
        info!(
            vector_rows = out_encoding.vector.row_size(),
            vector_cols = out_encoding.vector.col_size(),
            pubkey_rows = out_encoding.pubkey.matrix.row_size(),
            pubkey_cols = out_encoding.pubkey.matrix.col_size(),
            elapsed_ms = started.elapsed().as_millis(),
            "diamond we dec: circuit eval done"
        );
        drop(owned_enc_lookup_evaluator);
        info!("diamond we dec: owned lookup evaluator dropped");

        info!("diamond we dec: sample r decode column begin");
        let r_decode_col = self.sample_r_columns(ct.hash_key, 0, 1);
        info!(
            rows = r_decode_col.row_size(),
            cols = r_decode_col.col_size(),
            "diamond we dec: sample r decode column done"
        );
        let BggEncoding { vector: mut dec_term_vector, .. } = one_encoding;
        let BggEncoding { vector: out_vector, .. } = out_encoding;
        info!(
            dec_term_rows = dec_term_vector.row_size(),
            dec_term_cols = dec_term_vector.col_size(),
            out_rows = out_vector.row_size(),
            out_cols = out_vector.col_size(),
            "diamond we dec: subtract output vector begin"
        );
        dec_term_vector.sub_in_place(&out_vector);
        info!("diamond we dec: subtract output vector done");
        drop(out_vector);
        info!("diamond we dec: dropped output vector");
        info!("diamond we dec: dec term mul_decompose begin");
        let dec_term_projection = dec_term_vector.mul_decompose(&r_decode_col);
        info!(
            rows = dec_term_projection.row_size(),
            cols = dec_term_projection.col_size(),
            "diamond we dec: dec term mul_decompose done"
        );
        drop(dec_term_vector);
        drop(r_decode_col);
        info!("diamond we dec: dropped dec term vector and r decode column");
        let mut dec_encoding_vector = k_vector;
        info!(
            rows = dec_encoding_vector.row_size(),
            cols = dec_encoding_vector.col_size(),
            "diamond we dec: add dec term projection begin"
        );
        dec_encoding_vector.add_in_place(&dec_term_projection);
        info!("diamond we dec: add dec term projection done");
        drop(dec_term_projection);
        let mut noisy_plaintext = decoder;
        info!(
            rows = noisy_plaintext.row_size(),
            cols = noisy_plaintext.col_size(),
            "diamond we dec: subtract dec encoding from decoder begin"
        );
        noisy_plaintext.sub_in_place(&dec_encoding_vector);
        info!("diamond we dec: subtract dec encoding from decoder done");
        drop(dec_encoding_vector);
        info!(
            elapsed_ms = started.elapsed().as_millis(),
            "diamond we dec: decode noisy bool begin"
        );
        self.log_noisy_plaintext_error_diagnostic(&noisy_plaintext);
        let decoded = self.decode_noisy_bool(&noisy_plaintext);
        info!(
            decoded,
            elapsed_ms = started.elapsed().as_millis(),
            "diamond we dec: decode noisy bool done"
        );
        decoded
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use keccak_asm::Keccak256;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        __PAIR, __TestState,
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
    fn test_diamond_we_gadget_row_chunks_match_full_dcrt_gadget() {
        let params = DCRTPolyParams::default();
        let full = DCRTPolyMatrix::gadget_matrix(&params, DIAMOND_SECRET_SIZE);

        for col_start in 0..params.modulus_digits() {
            let col_len = (params.modulus_digits() - col_start).min(3);
            assert_eq!(
                TestDiamondWE::gadget_row_columns(&params, col_start, col_len),
                full.slice_columns(col_start, col_start + col_len),
                "DiamondWE gadget chunk must preserve tower-local DCRT columns at {col_start}"
            );
        }
    }

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
        let we = DiamondWE::new(injector, witness_size, dir.path(), tag.clone(), None);

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
            we.k_public_key_columns(),
            DistType::FinRingDist,
        );
        assert_eq!(k_pubkey, BggPublicKey::new(expected_k, false));
    }

    #[test]
    fn test_diamond_we_k_preimage_target_uses_hidden_plaintext_selector() {
        let params = DCRTPolyParams::default();
        let witness_size = 2;
        let hash_key = [9u8; 32];

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
            b"diamond_we_k_preimage_target_test".to_vec(),
            None,
        );

        let k_pubkey = we.sample_k_public_key(hash_key);
        let columns = we.k_public_key_columns();
        let target = we.k_preimage_target_columns(&k_pubkey, 0, columns);
        assert_eq!(target.size(), (2 * DIAMOND_SECRET_SIZE, columns));
        assert_eq!(target.slice_rows(0, DIAMOND_SECRET_SIZE), k_pubkey.matrix);

        let q: std::sync::Arc<BigUint> = params.modulus().into();
        let scale = DCRTPoly::from_biguint_to_constant(&params, q.as_ref() / 2u32);
        let expected_bottom = DCRTPolyMatrix::unit_row_vector(&params, columns, 0) * &scale;
        assert_eq!(
            target.slice_rows(DIAMOND_SECRET_SIZE, 2 * DIAMOND_SECRET_SIZE),
            expected_bottom
        );
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
            let we = DiamondWE::new(
                injector,
                witness_size,
                dir.path(),
                b"diamond_we_test".to_vec(),
                Some(witness.clone()),
            );
            crate::utils::start_pre_delete_file_length_observer();
            let ct = we.enc(&msg, circuit.clone(), &instance);
            let observed_deleted_bytes = crate::utils::take_pre_delete_file_lengths()
                .into_iter()
                .map(|(path, bytes)| {
                    let id = path
                        .file_stem()
                        .and_then(|name| name.to_str())
                        .expect("observed deleted artifact must have a UTF-8 stem")
                        .to_owned();
                    (id, bytes)
                })
                .collect::<std::collections::HashMap<_, _>>();
            let final_columns = we.k_public_key_columns();
            for id in [
                TestDiamondWE::session_preimage_id(ct.hash_key, TestDiamondWE::k_preimage_id()),
                TestDiamondWE::session_preimage_id(
                    ct.hash_key,
                    TestDiamondWE::decoder_preimage_id(),
                ),
            ] {
                if let Some(chunk_cols) = TestDiamondWE::preimage_chunk_cols(final_columns) {
                    for chunk_idx in
                        0..TestDiamondWE::preimage_chunk_count(final_columns, chunk_cols)
                    {
                        assert!(
                            we.matrix_path(&TestDiamondWE::preimage_chunk_id(&id, chunk_idx))
                                .exists(),
                            "full final preimage chunk must remain checkpointed"
                        );
                    }
                } else {
                    assert!(
                        we.matrix_path(&id).exists(),
                        "full final preimage must remain checkpointed"
                    );
                }
            }
            assert!(
                std::fs::read_dir(dir.path())
                    .expect("artifact directory should be readable")
                    .filter_map(Result::ok)
                    .all(|entry| !entry.file_name().to_string_lossy().contains("retained_col0")),
                "fixed-witness mode must not create first-column-only artifacts"
            );
            let session_prefix = TestDiamondWE::session_prefix(ct.hash_key);
            let observed_deleted_transition_bytes = observed_deleted_bytes
                .iter()
                .filter(|(id, _)| id.starts_with("diamond_transition_tensor_"))
                .map(|(_, bytes)| u128::from(*bytes))
                .sum::<u128>();
            assert!(
                observed_deleted_bytes
                    .keys()
                    .all(|id| id.starts_with("diamond_transition_tensor_")),
                "only unselected transition preimages should be deleted"
            );
            let mut retained_transition_bytes = 0u128;
            let mut retained_final_bytes = 0u128;
            for entry in std::fs::read_dir(dir.path())
                .expect("artifact directory should be readable")
                .filter_map(Result::ok)
            {
                let name = entry.file_name().to_string_lossy().into_owned();
                let bytes = u128::from(
                    entry.metadata().expect("retained artifact metadata should be readable").len(),
                );
                if name.starts_with("diamond_transition_tensor_") && name.ends_with(".matrixbin") {
                    retained_transition_bytes += bytes;
                } else if name.starts_with(&session_prefix) && name.ends_with(".matrixbin") {
                    retained_final_bytes += bytes;
                }
            }
            assert_eq!(
                observed_deleted_bytes.len(),
                observed_deleted_bytes
                    .keys()
                    .filter(|id| {
                        id.starts_with("diamond_transition_tensor_") ||
                            id.starts_with(&session_prefix)
                    })
                    .count(),
                "the observer should capture only intentionally pruned preimages"
            );
            let accounting: DiamondWEStorageAccounting = serde_json::from_slice(
                &std::fs::read(we.storage_accounting_path(ct.hash_key))
                    .expect("storage accounting should be readable"),
            )
            .expect("storage accounting should decode");
            assert_eq!(
                accounting.transition_preimage_bytes,
                observed_deleted_transition_bytes + retained_transition_bytes,
                "transition accounting must include exact deleted and retained file lengths"
            );
            assert_eq!(
                accounting.final_preimage_bytes, retained_final_bytes,
                "final accounting must equal the exact retained final preimage file lengths"
            );
            assert_eq!(
                accounting.total_preimage_bytes,
                observed_deleted_transition_bytes +
                    retained_transition_bytes +
                    retained_final_bytes
            );
            assert_eq!(we.dec(&ct, &witness), msg);
        }
    }

    #[test]
    #[sequential_test::sequential]
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
            None,
        );

        let ct = we.enc(&true, circuit.clone(), &instance);
        assert_eq!(we.dec(&ct, &witness), true);

        let total_cols = params.modulus_digits();
        let chunk_cols = crate::env::aux_sampling_chunk_width().min(total_cols);
        let chunk_count = TestDiamondWE::preimage_chunk_count(total_cols, chunk_cols);
        assert_eq!(chunk_cols, 20);
        assert_eq!(chunk_count, 2);
        let one_preimage_id =
            TestDiamondWE::session_preimage_id(ct.hash_key, TestDiamondWE::one_preimage_id());
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
            let chunk_id = TestDiamondWE::preimage_chunk_id(&one_preimage_id, chunk_idx);
            assert!(
                we.matrix_path(&chunk_id).exists(),
                "chunked one-preimage artifact {chunk_id} should exist"
            );
        }
        let retained_chunk_id = TestDiamondWE::preimage_chunk_id(&one_preimage_id, 0);
        let missing_chunk_id = TestDiamondWE::preimage_chunk_id(&one_preimage_id, 1);
        let retained_bytes = std::fs::read(we.matrix_path(&retained_chunk_id))
            .expect("retained final-preimage chunk should be readable");
        std::fs::remove_file(we.matrix_path(&missing_chunk_id))
            .expect("test should remove one final-preimage chunk");
        std::fs::remove_file(we.enc_complete_path())
            .expect("test should remove the encryption completion marker");

        for (tag, error_sigma) in [
            (b"diamond_we_mismatched_tag".to_vec(), 0.0),
            (b"diamond_we_chunked_preimage_test".to_vec(), 1.0),
        ] {
            let mismatched_injector =
                DiamondInjector::<
                    DCRTPolyMatrix,
                    DCRTPolyUniformSampler,
                    DCRTPolyHashSampler<Keccak256>,
                    DCRTPolyTrapdoorSampler,
                >::new(params.clone(), 1, 4, 2, 4.578, error_sigma);
            let mismatched_we =
                DiamondWE::new(mismatched_injector, witness_size, dir.path(), tag, None);
            let mismatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                mismatched_we.enc(&true, circuit.clone(), &instance)
            }));
            assert!(
                mismatch.is_err(),
                "resume must reject a changed BGG tag or injector parameter"
            );
        }
        let mismatched_injector = DiamondInjector::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(params.clone(), 1, 4, 2, 4.578, 0.0);
        let evaluator_mismatch_we = TestDiamondWE::with_evaluators(
            mismatched_injector,
            witness_size,
            dir.path(),
            b"diamond_we_chunked_preimage_test".to_vec(),
            b"different-evaluator-state".to_vec(),
            None,
            Some(NoCircuitEvaluator),
            None,
            None,
            None,
            None,
            None,
        );
        let evaluator_mismatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluator_mismatch_we.enc(&true, circuit.clone(), &instance)
        }));
        assert!(
            evaluator_mismatch.is_err(),
            "resume must reject changed evaluator presence or checkpoint identity"
        );

        let resumed_ct = we.enc(&true, circuit.clone(), &instance);
        assert_eq!(resumed_ct.hash_key, ct.hash_key, "resume must reuse the session hash key");
        assert_eq!(
            std::fs::read(we.matrix_path(&retained_chunk_id))
                .expect("retained final-preimage chunk should remain readable"),
            retained_bytes,
            "resume must not overwrite completed final-preimage chunks"
        );
        assert!(we.matrix_path(&missing_chunk_id).exists());
        assert!(we.enc_complete_path().exists());
        assert_eq!(we.dec(&resumed_ct, &witness), true);
        let completed_rerun = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            we.enc(&true, circuit, &instance)
        }));
        assert!(
            completed_rerun.is_err(),
            "a completed encryption session must require a fresh artifact directory"
        );
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
                Some(witness.clone()),
            );
            let ct = we.enc(&msg, circuit.clone(), &instance);
            assert_eq!(we.dec(&ct, &witness), msg);
        }
    }
}
