use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::bgg_pubkey::{column_chunk_bounds, column_chunk_count},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, fs::OpenOptions, io::Write, marker::PhantomData, path::Path};

pub(crate) mod bench_estimator;
#[cfg(feature = "gpu")]
#[path = "diamond_gpu.rs"]
mod gpu;
mod simulation;

pub use simulation::DiamondInputErrorSimulation;

const DIAMOND_PREFIX_SIZE: usize = 2;
pub(crate) const DIAMOND_SECRET_SIZE: usize = 1;

pub trait InputInjector<P> {
    type PreprocessOut;
    type State;

    /// Precompute and persist the transition matrices needed to advance the
    /// Diamond state for every possible input digit.
    fn preprocess(
        &self,
        dir_path: &Path,
        k: &P,
        retained_input_digits: Option<&[u32]>,
    ) -> Self::PreprocessOut;

    /// Rebuild the final Diamond states for the chosen input digits from the
    /// persisted transition matrices.
    fn online_eval(
        &self,
        dir_path: &Path,
        preprocess_out: &Self::PreprocessOut,
        input_digits: &[u32],
    ) -> Vec<Self::State>;
}

#[derive(Debug, Clone)]
/// Disk-backed implementation of the Diamond iO input insertion procedure for
/// BGG public keys and encodings. Preprocessing samples and stores the
/// transition/output preimages, while online evaluation only reads them back
/// and threads the selected digits through the stored transition graph.
pub struct DiamondInjector<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub params: <M::P as Poly>::Params,
    pub gpu_device_ids: Vec<i32>,
    pub input_count: usize,
    pub base: usize,
    pub batch_bits: usize,
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DiamondInjectorMetadata {
    input_count: usize,
    base: usize,
    batch_bits: usize,
    retained_input_digits: Option<Vec<u32>>,
    ring_dimension: u32,
    base_bits: u32,
    modulus_bits: usize,
    modulus_digits: usize,
    crt_moduli: Vec<u64>,
    crt_modulus_bits: usize,
    crt_depth: usize,
    trapdoor_sigma_bits: u64,
    error_sigma_bits: u64,
    transition_chunk_width: usize,
}

#[derive(Debug, Clone)]
/// Compact in-memory data returned by Diamond preprocessing.
///
/// Callers use the final state-specific trapdoor public bases to sample their
/// own final output projection preimages.
pub struct DiamondInjectorPreprocessOut<M, T>
where
    M: PolyMatrix,
{
    final_trapdoor_bytes: Vec<Vec<u8>>,
    final_pub_matrix_bytes: Vec<Vec<u8>>,
    _marker: PhantomData<(M, T)>,
}

impl<M, T> DiamondInjectorPreprocessOut<M, T>
where
    M: PolyMatrix,
{
    pub fn final_state_count(&self) -> usize {
        debug_assert_eq!(
            self.final_trapdoor_bytes.len(),
            self.final_pub_matrix_bytes.len(),
            "DiamondInjector final checkpoint vector length mismatch"
        );
        self.final_pub_matrix_bytes.len()
    }

    pub fn final_trapdoor_bytes(&self, state_idx: usize) -> &[u8] {
        assert!(
            state_idx < self.final_state_count(),
            "DiamondInjector final state index out of range: {} >= {}",
            state_idx,
            self.final_state_count()
        );
        &self.final_trapdoor_bytes[state_idx]
    }

    pub fn final_public_matrix(&self, params: &<M::P as Poly>::Params, state_idx: usize) -> M {
        assert!(
            state_idx < self.final_state_count(),
            "DiamondInjector final state index out of range: {} >= {}",
            state_idx,
            self.final_state_count()
        );
        M::from_compact_bytes(params, &self.final_pub_matrix_bytes[state_idx])
    }
}

impl<M, US, HS, TS> DiamondInjector<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub(crate) fn online_eval_staging_bytes(
        &self,
        dir_path: &Path,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        input_digits: &[u32],
    ) -> Vec<Vec<u8>> {
        #[cfg(feature = "gpu")]
        {
            return self.online_eval_gpu_staging_bytes(dir_path, preprocess_out, input_digits);
        }

        #[cfg(not(feature = "gpu"))]
        {
            self.online_eval(dir_path, preprocess_out, input_digits)
                .into_iter()
                .map(M::into_cpu_staging_bytes)
                .collect()
        }
    }

    pub fn new(
        params: <M::P as Poly>::Params,
        input_count: usize,
        base: usize,
        batch_bits: usize,
        trapdoor_sigma: f64,
        error_sigma: f64,
    ) -> Self {
        assert!(base > 0, "DiamondInjector base must be positive");
        assert!(batch_bits > 0, "DiamondInjector batch_bits must be positive");
        assert!(
            batch_bits <= u32::BITS as usize,
            "DiamondInjector batch_bits must fit into u32 input digits"
        );
        let required_base = 1usize
            .checked_shl(
                batch_bits
                    .try_into()
                    .expect("DiamondInjector batch_bits must fit into u32 for base validation"),
            )
            .expect("DiamondInjector batch_bits overflowed usize base validation");
        assert!(base >= required_base, "DiamondInjector base must be at least 2^batch_bits");
        assert!(error_sigma >= 0.0, "DiamondInjector error_sigma must be nonnegative");
        #[cfg(feature = "gpu")]
        let gpu_device_ids = params.device_ids();
        #[cfg(not(feature = "gpu"))]
        let gpu_device_ids = Vec::new();
        Self {
            params,
            gpu_device_ids,
            input_count,
            base,
            batch_bits,
            trapdoor_sigma,
            error_sigma,
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }

    /// Keep ordinary matrix sampling/loading on the base params while letting
    /// the GPU helpers distribute independent work across explicit single-GPU
    /// local params derived from these device ids.
    pub fn with_gpu_device_ids(mut self, gpu_device_ids: Vec<i32>) -> Self {
        self.gpu_device_ids = gpu_device_ids;
        self
    }

    fn ensure_dir(&self, dir_path: &Path) {
        fs::create_dir_all(dir_path).unwrap_or_else(|err| {
            panic!(
                "DiamondInjector failed to create preprocessing directory {}: {err}",
                dir_path.display()
            )
        });
    }

    fn metadata_path(&self, dir_path: &Path) -> std::path::PathBuf {
        dir_path.join("diamond_injector_metadata.json")
    }

    fn matrix_path(&self, dir_path: &Path, id: &str) -> std::path::PathBuf {
        dir_path.join(format!("{id}.matrixbin"))
    }

    fn bytes_path(&self, dir_path: &Path, id: &str) -> std::path::PathBuf {
        dir_path.join(format!("{id}.bytesbin"))
    }

    fn write_metadata(&self, dir_path: &Path, metadata: &DiamondInjectorMetadata) {
        let bytes =
            serde_json::to_vec_pretty(metadata).expect("DiamondInjector metadata should serialize");
        crate::utils::write_bytes_atomic(&self.metadata_path(dir_path), &bytes);
    }

    fn read_metadata(&self, dir_path: &Path) -> DiamondInjectorMetadata {
        let bytes = fs::read(self.metadata_path(dir_path))
            .expect("DiamondInjector metadata should have been written");
        serde_json::from_slice(&bytes).expect("DiamondInjector metadata should decode")
    }

    fn current_metadata(&self, retained_input_digits: Option<&[u32]>) -> DiamondInjectorMetadata {
        let (crt_moduli, crt_modulus_bits, crt_depth) = self.params.to_crt();
        DiamondInjectorMetadata {
            input_count: self.input_count,
            base: self.base,
            batch_bits: self.batch_bits,
            retained_input_digits: retained_input_digits.map(<[u32]>::to_vec),
            ring_dimension: self.params.ring_dimension(),
            base_bits: self.params.base_bits(),
            modulus_bits: self.params.modulus_bits(),
            modulus_digits: self.params.modulus_digits(),
            crt_moduli,
            crt_modulus_bits,
            crt_depth,
            trapdoor_sigma_bits: self.trapdoor_sigma.to_bits(),
            error_sigma_bits: self.error_sigma.to_bits(),
            transition_chunk_width: crate::env::aux_sampling_chunk_width(),
        }
    }

    fn prepare_metadata(&self, dir_path: &Path, retained_input_digits: Option<&[u32]>) {
        if let Some(input_digits) = retained_input_digits {
            self.validate_digits(input_digits);
        }
        let current = self.current_metadata(retained_input_digits);
        if self.metadata_path(dir_path).exists() {
            let stored = self.read_metadata(dir_path);
            assert_eq!(
                stored, current,
                "DiamondInjector checkpoint configuration mismatch; use a separate artifact directory"
            );
        } else {
            self.write_metadata(dir_path, &current);
        }
    }

    fn assert_current_metadata(&self, metadata: &DiamondInjectorMetadata) {
        assert_eq!(
            metadata.input_count, self.input_count,
            "DiamondInjector metadata input count mismatch"
        );
        assert_eq!(metadata.base, self.base, "DiamondInjector metadata base mismatch");
        assert_eq!(
            metadata.batch_bits, self.batch_bits,
            "DiamondInjector metadata batch_bits mismatch"
        );
    }

    fn write_matrix(&self, dir_path: &Path, id: &str, matrix: &M) {
        self.write_matrix_bytes(dir_path, id, &matrix.to_compact_bytes());
    }

    fn write_matrix_bytes(&self, dir_path: &Path, id: &str, bytes: &[u8]) {
        crate::utils::write_bytes_atomic(&self.matrix_path(dir_path, id), bytes);
    }

    #[cfg(any(not(feature = "gpu"), test))]
    fn read_matrix(&self, dir_path: &Path, id: &str) -> M {
        let bytes = self.read_matrix_bytes(dir_path, id);
        M::from_compact_bytes(&self.params, &bytes)
    }

    fn read_matrix_bytes(&self, dir_path: &Path, id: &str) -> Vec<u8> {
        fs::read(self.matrix_path(dir_path, id))
            .unwrap_or_else(|err| panic!("DiamondInjector failed to read matrix {id}: {err}"))
    }

    fn write_bytes(&self, dir_path: &Path, id: &str, bytes: &[u8]) {
        crate::utils::write_bytes_atomic(&self.bytes_path(dir_path, id), bytes);
    }

    fn read_bytes(&self, dir_path: &Path, id: &str) -> Vec<u8> {
        fs::read(self.bytes_path(dir_path, id))
            .unwrap_or_else(|err| panic!("DiamondInjector failed to read bytes {id}: {err}"))
    }

    fn matrix_exists(&self, dir_path: &Path, id: &str) -> bool {
        self.matrix_path(dir_path, id).exists()
    }

    fn bytes_exists(&self, dir_path: &Path, id: &str) -> bool {
        self.bytes_path(dir_path, id).exists()
    }

    fn state_row_size(&self) -> usize {
        DIAMOND_PREFIX_SIZE
            .checked_mul(DIAMOND_SECRET_SIZE)
            .expect("DiamondInjector state row size overflow")
    }

    fn gadget_col_size(&self, params: &<M::P as Poly>::Params) -> usize {
        DIAMOND_SECRET_SIZE
            .checked_mul(params.modulus_digits())
            .expect("DiamondInjector gadget column count overflow")
    }

    fn b_public_col_size(&self, params: &<M::P as Poly>::Params) -> usize {
        self.state_row_size()
            .checked_mul(params.modulus_digits() + 2)
            .expect("DiamondInjector B public column count overflow")
    }

    fn state_col_size(&self, params: &<M::P as Poly>::Params) -> usize {
        self.b_public_col_size(params)
    }

    fn chunk_id(&self, id: &str, chunk_idx: usize) -> String {
        format!("{id}_chunk{chunk_idx}")
    }

    fn secret_epsilon_id(&self) -> &'static str {
        "diamond_secret_epsilon_tensor"
    }

    fn digit_secret_id(&self, level: usize, digit_value: usize) -> String {
        format!("diamond_secret_tensor_{level}_{digit_value}")
    }

    fn b_matrix_id(&self, level: usize, state_idx: usize) -> String {
        format!("diamond_b_tensor_{level}_{state_idx}")
    }

    fn b_trapdoor_id(&self, level: usize, state_idx: usize) -> String {
        format!("diamond_b_tensor_{level}_{state_idx}_trapdoor")
    }

    fn b_checkpoint_complete_id(&self, level: usize, state_idx: usize) -> String {
        format!("diamond_b_tensor_{level}_{state_idx}_pair_complete")
    }

    fn p_epsilon_id(&self) -> &'static str {
        "diamond_initial_state_tensor"
    }

    fn k_plaintext_id(&self) -> &'static str {
        "diamond_k_plaintext"
    }

    fn k_id(&self, level: usize, digit_value: usize, state_idx: usize) -> String {
        format!("diamond_transition_tensor_{level}_{digit_value}_{state_idx}")
    }

    fn discard_journal_path(&self, dir_path: &Path) -> std::path::PathBuf {
        dir_path.join("diamond_discarded_transition_chunks.journal")
    }

    fn transition_task_index(
        &self,
        level: usize,
        digit_value: usize,
        state_idx: usize,
        chunk_idx: usize,
        chunk_count: usize,
    ) -> u64 {
        let previous_states = (1..level)
            .map(|previous_level| self.state_count_at_level(previous_level))
            .sum::<usize>();
        let task_index = previous_states
            .checked_mul(self.base)
            .and_then(|value| {
                digit_value
                    .checked_mul(self.state_count_at_level(level))
                    .and_then(|level_offset| value.checked_add(level_offset))
            })
            .and_then(|value| value.checked_add(state_idx))
            .and_then(|value| value.checked_mul(chunk_count))
            .and_then(|value| value.checked_add(chunk_idx))
            .expect("DiamondInjector transition task index overflow");
        u64::try_from(task_index).expect("DiamondInjector transition task index must fit u64")
    }

    fn load_discarded_transition_tasks(&self, dir_path: &Path) -> HashMap<u64, u64> {
        const RECORD_SIZE: usize = 2 * size_of::<u64>();
        let path = self.discard_journal_path(dir_path);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return HashMap::new(),
            Err(err) => panic!("DiamondInjector failed to read {}: {err}", path.display()),
        };
        let complete_len = bytes.len() / RECORD_SIZE * RECORD_SIZE;
        if complete_len != bytes.len() {
            let file = OpenOptions::new().write(true).open(&path).unwrap_or_else(|err| {
                panic!("DiamondInjector failed to repair {}: {err}", path.display())
            });
            file.set_len(complete_len as u64).unwrap_or_else(|err| {
                panic!("DiamondInjector failed to truncate {}: {err}", path.display())
            });
            file.sync_all().unwrap_or_else(|err| {
                panic!("DiamondInjector failed to sync repaired {}: {err}", path.display())
            });
        }
        bytes[..complete_len]
            .chunks_exact(RECORD_SIZE)
            .map(|record| {
                let task_index = u64::from_le_bytes(record[..8].try_into().unwrap());
                let stored_bytes = u64::from_le_bytes(record[8..].try_into().unwrap());
                (task_index, stored_bytes)
            })
            .collect()
    }

    fn record_discarded_transition_tasks(&self, dir_path: &Path, tasks: &[(u64, u64)]) {
        if tasks.is_empty() {
            return;
        }
        let path = self.discard_journal_path(dir_path);
        let mut file =
            OpenOptions::new().create(true).append(true).open(&path).unwrap_or_else(|err| {
                panic!("DiamondInjector failed to open {}: {err}", path.display())
            });
        for (task_index, stored_bytes) in tasks {
            file.write_all(&task_index.to_le_bytes()).unwrap_or_else(|err| {
                panic!("DiamondInjector failed to append {}: {err}", path.display())
            });
            file.write_all(&stored_bytes.to_le_bytes()).unwrap_or_else(|err| {
                panic!("DiamondInjector failed to append {}: {err}", path.display())
            });
        }
        file.sync_all().unwrap_or_else(|err| {
            panic!("DiamondInjector failed to sync {}: {err}", path.display())
        });
    }

    pub(crate) fn generated_transition_preimage_bytes(
        &self,
        dir_path: &Path,
        retained_input_digits: Option<&[u32]>,
    ) -> u128 {
        let discarded_bytes = self
            .load_discarded_transition_tasks(dir_path)
            .into_values()
            .map(u128::from)
            .sum::<u128>();
        let state_cols = self.state_col_size(&self.params);
        let chunk_count = column_chunk_count(state_cols);
        let mut retained_bytes = 0u128;
        for level in 1..=self.input_count {
            for digit_value in 0..self.base {
                let retained = retained_input_digits
                    .map(|digits| digits[level - 1] as usize == digit_value)
                    .unwrap_or(true);
                if !retained {
                    continue;
                }
                for state_idx in 0..self.state_count_at_level(level) {
                    let id = self.k_id(level, digit_value, state_idx);
                    for chunk_idx in 0..chunk_count {
                        let chunk_id = self.chunk_id(&id, chunk_idx);
                        let bytes = fs::metadata(self.matrix_path(dir_path, &chunk_id))
                            .unwrap_or_else(|err| {
                                panic!(
                                    "DiamondInjector failed to stat retained matrix {chunk_id}: {err}"
                                )
                            })
                            .len();
                        retained_bytes = retained_bytes
                            .checked_add(u128::from(bytes))
                            .expect("DiamondInjector generated byte count overflow");
                    }
                }
            }
        }
        discarded_bytes
            .checked_add(retained_bytes)
            .expect("DiamondInjector generated byte count overflow")
    }

    fn remove_matrix_checkpoint(&self, dir_path: &Path, id: &str) {
        let path = self.matrix_path(dir_path, id);
        #[cfg(test)]
        if path.exists() {
            crate::utils::observe_file_before_delete(&path);
        }
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => panic!("DiamondInjector failed to remove matrix {id}: {err}"),
        }
    }

    fn remove_bytes_checkpoint(&self, dir_path: &Path, id: &str) {
        match fs::remove_file(self.bytes_path(dir_path, id)) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => panic!("DiamondInjector failed to remove bytes {id}: {err}"),
        }
    }

    fn sample_secret_epsilon_with_params(&self, params: &<M::P as Poly>::Params) -> M {
        let s = US::new().sample_uniform(params, 1, 1, DistType::TernaryDist).entry(0, 0);
        M::from_poly_vec_row(params, vec![s])
    }

    fn sample_digit_secret_mask_with_params(&self, params: &<M::P as Poly>::Params) -> M {
        let s_prime = US::new().sample_uniform(params, 1, 1, DistType::TernaryDist).entry(0, 0);
        M::from_poly_vec_row(params, vec![s_prime])
    }

    fn sample_error_matrix_with_dims(
        &self,
        params: &<M::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
    ) -> M {
        if self.error_sigma == 0.0 {
            M::zero(params, nrow, ncol)
        } else {
            US::new().sample_uniform(
                params,
                nrow,
                ncol,
                DistType::GaussDist { sigma: self.error_sigma },
            )
        }
    }

    pub fn batch_bits(&self) -> usize {
        self.batch_bits
    }

    fn state_count_at_level(&self, level: usize) -> usize {
        1usize
            .checked_add(
                level
                    .checked_mul(self.batch_bits())
                    .expect("DiamondInjector expanded state count overflow"),
            )
            .expect("DiamondInjector expanded state count overflow")
    }

    fn first_bit_state_idx_for_level(&self, level: usize) -> usize {
        assert!(level > 0, "DiamondInjector level must be positive for bit state indexing");
        1usize
            .checked_add(
                (level - 1)
                    .checked_mul(self.batch_bits())
                    .expect("DiamondInjector bit state index overflow"),
            )
            .expect("DiamondInjector bit state index overflow")
    }

    pub fn bit_state_idx(&self, input_idx: usize, bit_idx: usize) -> usize {
        assert!(bit_idx < self.batch_bits(), "DiamondInjector bit index out of range");
        1usize
            .checked_add(
                input_idx
                    .checked_mul(self.batch_bits())
                    .expect("DiamondInjector bit state index overflow"),
            )
            .and_then(|idx| idx.checked_add(bit_idx))
            .expect("DiamondInjector bit state index overflow")
    }

    pub fn bit_pubkey_idx(&self, input_idx: usize, bit_idx: usize) -> usize {
        assert!(bit_idx < self.batch_bits(), "DiamondInjector bit index out of range");
        input_idx
            .checked_mul(self.batch_bits())
            .and_then(|idx| idx.checked_add(bit_idx))
            .expect("DiamondInjector bit public key index overflow")
    }

    fn new_bit_idx_for_state(&self, level: usize, state_idx: usize) -> Option<usize> {
        let first = self.first_bit_state_idx_for_level(level);
        let end =
            first.checked_add(self.batch_bits()).expect("DiamondInjector bit state index overflow");
        if (first..end).contains(&state_idx) { Some(state_idx - first) } else { None }
    }

    fn transition_source_state_idx(&self, level: usize, state_idx: usize) -> usize {
        assert!(level > 0, "DiamondInjector transition level must be positive");
        if self.new_bit_idx_for_state(level, state_idx).is_some() { 0 } else { state_idx }
    }

    pub fn digit_bit_value(&self, digit_value: usize, bit_idx: usize) -> usize {
        assert!(bit_idx < self.batch_bits(), "DiamondInjector bit index out of range");
        (digit_value >> bit_idx) & 1
    }

    fn validate_digits(&self, input_digits: &[u32]) {
        assert_eq!(
            input_digits.len(),
            self.input_count,
            "DiamondInjector online_eval expected {} input digits but received {}",
            self.input_count,
            input_digits.len()
        );
        for (digit_idx, digit_value) in input_digits.iter().copied().enumerate() {
            assert!(
                (digit_value as usize) < self.base,
                "DiamondInjector input digit at position {} out of range: {} >= {}",
                digit_idx,
                digit_value,
                self.base
            );
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn load_or_sample_secret_epsilon(&self, dir_path: &Path, id: &str) -> M {
        if self.matrix_exists(dir_path, id) {
            self.read_matrix(dir_path, id)
        } else {
            let secret = self.sample_secret_epsilon_with_params(&self.params);
            self.write_matrix(dir_path, id, &secret);
            secret
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn load_or_sample_digit_secret_mask(&self, dir_path: &Path, id: &str) -> M {
        if self.matrix_exists(dir_path, id) {
            self.read_matrix(dir_path, id)
        } else {
            let secret = self.sample_digit_secret_mask_with_params(&self.params);
            self.write_matrix(dir_path, id, &secret);
            secret
        }
    }

    #[cfg(feature = "gpu")]
    fn load_or_sample_secret_epsilon_bytes(&self, dir_path: &Path, id: &str) -> Vec<u8> {
        if self.matrix_exists(dir_path, id) {
            self.read_matrix_bytes(dir_path, id)
        } else {
            let secret = self.sample_secret_epsilon_with_params(&self.params);
            let bytes = secret.to_compact_bytes();
            self.write_matrix_bytes(dir_path, id, &bytes);
            bytes
        }
    }

    #[cfg(feature = "gpu")]
    fn load_or_sample_digit_secret_mask_bytes(&self, dir_path: &Path, id: &str) -> Vec<u8> {
        if self.matrix_exists(dir_path, id) {
            self.read_matrix_bytes(dir_path, id)
        } else {
            let secret = self.sample_digit_secret_mask_with_params(&self.params);
            let bytes = secret.to_compact_bytes();
            self.write_matrix_bytes(dir_path, id, &bytes);
            bytes
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn load_or_sample_b_checkpoint(
        &self,
        dir_path: &Path,
        level: usize,
        state_idx: usize,
    ) -> (TS::Trapdoor, M) {
        // Checkpoint one trapdoor public matrix per level/state. Later
        // preimage sampling uses this stored pair directly.
        let matrix_id = self.b_matrix_id(level, state_idx);
        let trapdoor_id = self.b_trapdoor_id(level, state_idx);
        let complete_id = self.b_checkpoint_complete_id(level, state_idx);
        let pair_is_complete = self.matrix_exists(dir_path, &matrix_id) &&
            self.bytes_exists(dir_path, &trapdoor_id) &&
            self.bytes_exists(dir_path, &complete_id);
        if pair_is_complete {
            let trapdoor =
                TS::trapdoor_from_bytes(&self.params, &self.read_bytes(dir_path, &trapdoor_id))
                    .unwrap_or_else(|| {
                        panic!(
                            "DiamondInjector failed to decode trapdoor checkpoint for level {level}, state {state_idx}"
                        )
                    });
            return (trapdoor, self.read_matrix(dir_path, &matrix_id));
        }
        self.remove_matrix_checkpoint(dir_path, &matrix_id);
        self.remove_bytes_checkpoint(dir_path, &trapdoor_id);
        self.remove_bytes_checkpoint(dir_path, &complete_id);

        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let (trapdoor, matrix) = trap_sampler.trapdoor(&self.params, self.state_row_size());
        self.write_bytes(dir_path, &trapdoor_id, &TS::trapdoor_to_bytes(&trapdoor));
        self.write_matrix(dir_path, &matrix_id, &matrix);
        self.write_bytes(dir_path, &complete_id, b"complete\n");
        (trapdoor, matrix)
    }

    #[cfg(feature = "gpu")]
    fn load_or_sample_b_checkpoint_bytes(
        &self,
        dir_path: &Path,
        level: usize,
        state_idx: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let matrix_id = self.b_matrix_id(level, state_idx);
        let trapdoor_id = self.b_trapdoor_id(level, state_idx);
        let complete_id = self.b_checkpoint_complete_id(level, state_idx);
        let pair_is_complete = self.matrix_exists(dir_path, &matrix_id) &&
            self.bytes_exists(dir_path, &trapdoor_id) &&
            self.bytes_exists(dir_path, &complete_id);
        if pair_is_complete {
            return (
                self.read_matrix_bytes(dir_path, &matrix_id),
                self.read_bytes(dir_path, &trapdoor_id),
            );
        }
        self.remove_matrix_checkpoint(dir_path, &matrix_id);
        self.remove_bytes_checkpoint(dir_path, &trapdoor_id);
        self.remove_bytes_checkpoint(dir_path, &complete_id);

        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let (trapdoor, matrix) = trap_sampler.trapdoor(&self.params, self.state_row_size());
        let trapdoor_bytes = TS::trapdoor_to_bytes(&trapdoor);
        let matrix_bytes = matrix.to_compact_bytes();
        self.write_bytes(dir_path, &trapdoor_id, &trapdoor_bytes);
        self.write_matrix_bytes(dir_path, &matrix_id, &matrix_bytes);
        self.write_bytes(dir_path, &complete_id, b"complete\n");
        (matrix_bytes, trapdoor_bytes)
    }

    fn state_public_chunk_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        b_matrix: &M,
        chunk_idx: usize,
    ) -> M {
        let total_cols = self.state_col_size(params);
        let (col_start, col_len) = column_chunk_bounds(total_cols, chunk_idx);
        let col_end = col_start + col_len;
        debug_assert_eq!(
            b_matrix.col_size(),
            total_cols,
            "DiamondInjector B matrix column count must equal state_col_size"
        );
        b_matrix.slice_columns(col_start, col_end)
    }

    fn transition_selector_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        secret_mask: &M,
    ) -> M {
        let zero_block = M::zero(params, DIAMOND_SECRET_SIZE, DIAMOND_SECRET_SIZE);
        let top = secret_mask.concat_columns(&[&zero_block]);
        let bottom = zero_block.concat_columns(&[secret_mask]);
        top.concat_rows(&[&bottom])
    }

    fn k_transition_selector_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        secret_mask: &M,
    ) -> M {
        let zero = M::P::const_zero(params);
        M::from_poly_vec(
            params,
            vec![vec![secret_mask.entry(0, 0), zero.clone()], vec![zero, M::P::const_one(params)]],
        )
    }

    fn special_transition_selector_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        bit_value: usize,
        secret_mask: &M,
    ) -> M {
        let bit = M::P::from_usize_to_constant(params, bit_value);
        // Newly born bit branches use H_x tensor s': the empty prefix
        // component becomes (s', x * s') while the lower row remains zero.
        let zero_block = M::zero(params, DIAMOND_SECRET_SIZE, DIAMOND_SECRET_SIZE);
        let bit_mask = secret_mask.clone() * &bit;
        let top = secret_mask.concat_columns(&[&bit_mask]);
        let bottom = zero_block.clone().concat_columns(&[&zero_block]);
        top.concat_rows(&[&bottom])
    }

    fn build_initial_encoding(&self, b0_matrix: &M, secret_epsilon: &M, k: &M::P) -> M {
        // Build the state that represents the empty input prefix. It is the
        // only online-evaluation seed that exists before any digit is chosen.
        let selector =
            M::from_poly_vec_row(&self.params, vec![secret_epsilon.entry(0, 0), k.clone()]);
        let mut p_epsilon = selector * b0_matrix;
        p_epsilon.add_in_place(&self.sample_error_matrix_with_dims(
            &self.params,
            1,
            self.state_col_size(&self.params),
        ));
        p_epsilon
    }

    fn build_k_target_chunk_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        level: usize,
        digit_value: usize,
        state_idx: usize,
        secret_mask: &M,
        b_matrix: &M,
        chunk_idx: usize,
    ) -> M {
        // Build one chunk of the target matrix whose preimage becomes a
        // transition matrix. Existing branches use the identity-style selector,
        // while each newly born branch for the current digit uses the special
        // selector above so one chosen bit is embedded into that path.
        let public_chunk = self.state_public_chunk_with_params(params, b_matrix, chunk_idx);
        let selector = if let Some(bit_idx) = self.new_bit_idx_for_state(level, state_idx) {
            let bit_value = self.digit_bit_value(digit_value, bit_idx);
            self.special_transition_selector_with_params(params, bit_value, secret_mask)
        } else if state_idx == 0 {
            self.k_transition_selector_with_params(params, secret_mask)
        } else {
            self.transition_selector_with_params(params, secret_mask)
        };
        let mut target = selector * public_chunk;
        let (_, col_len) = column_chunk_bounds(self.state_col_size(params), chunk_idx);
        target.add_in_place(&self.sample_error_matrix_with_dims(
            params,
            self.state_row_size(),
            col_len,
        ));
        target
    }

    #[cfg(not(feature = "gpu"))]
    fn left_mul_checkpointed_cpu(
        &self,
        dir_path: &Path,
        lhs: &M,
        id: &str,
        total_cols: usize,
    ) -> M {
        let mut chunk_iter = (0..column_chunk_count(total_cols)).map(|chunk_idx| {
            lhs.clone() * &self.read_matrix(dir_path, &self.chunk_id(id, chunk_idx))
        });
        let first = chunk_iter.next().expect("chunked artifact should have at least one chunk");
        let rest = chunk_iter.collect::<Vec<_>>();
        if rest.is_empty() { first } else { first.concat_columns_owned(rest) }
    }

    pub fn read_preprocessed_k(&self, dir_path: &Path) -> M::P {
        M::P::from_compact_bytes(&self.params, &self.read_bytes(dir_path, self.k_plaintext_id()))
    }

    pub fn build_output_encoding(
        &self,
        vector: M,
        pubkey: BggPublicKey<M>,
        plaintext: Option<M::P>,
    ) -> BggEncoding<M> {
        let plaintext = if pubkey.reveal_plaintext { plaintext } else { None };
        BggEncoding::new(vector, pubkey, plaintext)
    }

    #[cfg(test)]
    pub fn debug_final_secret_matrix(&self, dir_path: &Path, input_digits: &[u32]) -> M {
        self.validate_digits(input_digits);
        let mut secret_matrix = self.read_matrix(dir_path, self.secret_epsilon_id());
        for (digit_idx, digit_value) in input_digits.iter().copied().enumerate() {
            let secret_mask = self
                .read_matrix(dir_path, &self.digit_secret_id(digit_idx + 1, digit_value as usize));
            secret_matrix = secret_matrix * secret_mask;
        }
        secret_matrix
    }
}

impl<M, US, HS, TS> InputInjector<M::P> for DiamondInjector<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    type PreprocessOut = DiamondInjectorPreprocessOut<M, TS::Trapdoor>;
    type State = M;

    fn preprocess(
        &self,
        dir_path: &Path,
        k: &M::P,
        retained_input_digits: Option<&[u32]>,
    ) -> Self::PreprocessOut {
        #[cfg(feature = "gpu")]
        {
            self.preprocess_gpu(dir_path, k, retained_input_digits);
            let mut final_trapdoor_bytes =
                Vec::with_capacity(self.state_count_at_level(self.input_count));
            let mut final_pub_matrix_bytes =
                Vec::with_capacity(self.state_count_at_level(self.input_count));
            for state_idx in 0..self.state_count_at_level(self.input_count) {
                let (public_matrix_bytes, trapdoor_bytes) =
                    self.load_or_sample_b_checkpoint_bytes(dir_path, self.input_count, state_idx);
                final_trapdoor_bytes.push(trapdoor_bytes);
                final_pub_matrix_bytes.push(public_matrix_bytes);
            }
            return DiamondInjectorPreprocessOut {
                final_trapdoor_bytes,
                final_pub_matrix_bytes,
                _marker: PhantomData,
            };
        }

        #[cfg(not(feature = "gpu"))]
        {
            self.ensure_dir(dir_path);
            self.prepare_metadata(dir_path, retained_input_digits);
            let mut discarded_tasks = self.load_discarded_transition_tasks(dir_path);
            self.write_bytes(dir_path, self.k_plaintext_id(), &k.to_compact_bytes());

            let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
            let mut b_checkpoints = Vec::with_capacity(self.input_count + 1);
            let mut trapdoors = Vec::with_capacity(self.input_count + 1);
            for level in 0..=self.input_count {
                let state_count = self.state_count_at_level(level);
                let mut level_b = Vec::with_capacity(state_count);
                let mut level_t = Vec::with_capacity(state_count);
                for state_idx in 0..state_count {
                    let (trapdoor, b_matrix) =
                        self.load_or_sample_b_checkpoint(dir_path, level, state_idx);
                    level_t.push(trapdoor);
                    level_b.push(b_matrix);
                }
                trapdoors.push(level_t);
                b_checkpoints.push(level_b);
            }

            // The empty-prefix seed embeds the encrypted message k. Rebuild it
            // for every encryption while reusing the message-independent
            // secret and B checkpoint.
            let secret_epsilon =
                self.load_or_sample_secret_epsilon(dir_path, self.secret_epsilon_id());
            self.write_matrix(
                dir_path,
                self.p_epsilon_id(),
                &self.build_initial_encoding(&b_checkpoints[0][0], &secret_epsilon, k),
            );

            let state_cols = self.state_col_size(&self.params);

            // For each level, each digit value, and each active branch, sample
            // the transition preimage that advances the state machine by one
            // more chosen digit. Each column chunk is written immediately so we
            // never need to keep the full transition matrix in memory.
            for level in 1..=self.input_count {
                for digit_value in 0..self.base {
                    let secret_mask = self.load_or_sample_digit_secret_mask(
                        dir_path,
                        &self.digit_secret_id(level, digit_value),
                    );
                    for state_idx in 0..self.state_count_at_level(level) {
                        let k_id = self.k_id(level, digit_value, state_idx);
                        let source_state_idx = self.transition_source_state_idx(level, state_idx);
                        let chunk_count = column_chunk_count(state_cols);
                        let retain_transition = retained_input_digits
                            .map(|digits| digits[level - 1] as usize == digit_value)
                            .unwrap_or(true);
                        for chunk_idx in 0..chunk_count {
                            let chunk_id = self.chunk_id(&k_id, chunk_idx);
                            let task_index = self.transition_task_index(
                                level,
                                digit_value,
                                state_idx,
                                chunk_idx,
                                chunk_count,
                            );
                            if !retain_transition && discarded_tasks.contains_key(&task_index) {
                                self.remove_matrix_checkpoint(dir_path, &chunk_id);
                                continue;
                            }
                            if self.matrix_exists(dir_path, &chunk_id) {
                                if !retain_transition {
                                    let stored_bytes = fs::metadata(
                                        self.matrix_path(dir_path, &chunk_id),
                                    )
                                    .unwrap_or_else(|err| {
                                        panic!("DiamondInjector failed to stat matrix {chunk_id}: {err}")
                                    })
                                    .len();
                                    self.record_discarded_transition_tasks(
                                        dir_path,
                                        &[(task_index, stored_bytes)],
                                    );
                                    discarded_tasks.insert(task_index, stored_bytes);
                                    self.remove_matrix_checkpoint(dir_path, &chunk_id);
                                }
                                continue;
                            }
                            let target_chunk = self.build_k_target_chunk_with_params(
                                &self.params,
                                level,
                                digit_value,
                                state_idx,
                                &secret_mask,
                                &b_checkpoints[level][state_idx],
                                chunk_idx,
                            );
                            let k_chunk = trap_sampler.preimage(
                                &self.params,
                                &trapdoors[level - 1][source_state_idx],
                                &b_checkpoints[level - 1][source_state_idx],
                                &target_chunk,
                            );
                            let k_chunk_bytes = k_chunk.to_compact_bytes();
                            let stored_bytes = u64::try_from(k_chunk_bytes.len())
                                .expect("DiamondInjector transition chunk size must fit u64");
                            self.write_matrix_bytes(dir_path, &chunk_id, &k_chunk_bytes);
                            if !retain_transition {
                                self.record_discarded_transition_tasks(
                                    dir_path,
                                    &[(task_index, stored_bytes)],
                                );
                                discarded_tasks.insert(task_index, stored_bytes);
                                self.remove_matrix_checkpoint(dir_path, &chunk_id);
                            }
                        }
                    }
                }
            }
            DiamondInjectorPreprocessOut {
                final_trapdoor_bytes: trapdoors
                    .pop()
                    .expect("DiamondInjector must keep final trapdoor checkpoints")
                    .into_iter()
                    .map(|trapdoor| TS::trapdoor_to_bytes(&trapdoor))
                    .collect(),
                final_pub_matrix_bytes: b_checkpoints
                    .pop()
                    .expect("DiamondInjector must keep final public matrix checkpoints")
                    .into_iter()
                    .map(|matrix| matrix.to_compact_bytes())
                    .collect(),
                _marker: PhantomData,
            }
        }
    }

    fn online_eval(
        &self,
        dir_path: &Path,
        preprocess_out: &Self::PreprocessOut,
        input_digits: &[u32],
    ) -> Vec<M> {
        #[cfg(feature = "gpu")]
        {
            return self.online_eval_gpu(dir_path, preprocess_out, input_digits);
        }

        #[cfg(not(feature = "gpu"))]
        {
            self.validate_digits(input_digits);
            assert_eq!(
                preprocess_out.final_state_count(),
                self.state_count_at_level(self.input_count),
                "DiamondInjector final checkpoint count mismatch"
            );
            let metadata = self.read_metadata(dir_path);
            self.assert_current_metadata(&metadata);

            // Start from the persisted empty-prefix seed.
            let mut states = vec![self.read_matrix(dir_path, self.p_epsilon_id())];
            let state_cols = self.state_col_size(&self.params);
            for (digit_idx, digit_value) in input_digits.iter().copied().enumerate() {
                let level = digit_idx + 1;
                let prev_states = std::mem::take(&mut states);
                let prev_p0 = prev_states[0].clone();
                let mut next_states = Vec::with_capacity(self.state_count_at_level(level));
                // Advance every currently alive branch through the transition
                // matrix for the chosen digit, and spawn the new branch that
                // records each bit of the current digit.
                for state_idx in 0..self.state_count_at_level(level) {
                    let lhs = if self.new_bit_idx_for_state(level, state_idx).is_some() {
                        prev_p0.clone()
                    } else {
                        prev_states[state_idx].clone()
                    };
                    let rhs_id = self.k_id(level, digit_value as usize, state_idx);
                    next_states
                        .push(self.left_mul_checkpointed_cpu(dir_path, &lhs, &rhs_id, state_cols));
                }
                states = next_states;
            }
            states
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DIAMOND_SECRET_SIZE, DiamondInjector, InputInjector};
    use crate::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        simulator::{
            SimulatorContext, error_norm::compute_preimage_sigma, poly_matrix_norm::PolyMatrixNorm,
        },
        utils::bigdecimal_bits_ceil,
    };
    use bigdecimal::BigDecimal;
    use keccak_asm::Keccak256;
    use num_bigint::{BigInt, BigUint};
    use num_traits::FromPrimitive;
    use std::{io::Write as _, sync::Arc};
    use tempfile::tempdir;

    type TestInjector = DiamondInjector<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
    >;
    type TestPoly = <DCRTPolyMatrix as PolyMatrix>::P;

    fn assert_poly_matrix_bound_eq(actual: &PolyMatrixNorm, expected: &PolyMatrixNorm) {
        assert_eq!(actual.nrow, expected.nrow);
        assert_eq!(actual.ncol, expected.ncol);
        assert_eq!(actual.ncol_sqrt, expected.ncol_sqrt);
        assert_eq!(actual.poly_norm, expected.poly_norm);
        assert_eq!(actual.zero_rows, expected.zero_rows);
    }

    fn assert_poly_matrix_bounds_eq(actual: &[PolyMatrixNorm], expected: &[PolyMatrixNorm]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_poly_matrix_bound_eq(actual, expected);
        }
    }

    #[serial_test::serial]
    #[test]
    fn test_diamond_injector_online_eval_returns_exact_bgg_relations() {
        let params = DCRTPolyParams::default();
        let input_count = 3;
        let base = 4;
        let batch_bits = 2;
        let dir = tempdir().expect("temporary directory should be created");

        let injector = TestInjector::new(params.clone(), input_count, base, batch_bits, 4.578, 0.0);

        let k = TestPoly::from_usize_to_constant(&params, 3);

        let preprocess_out = injector.preprocess(dir.path(), &k, None);

        let digits = vec![1u32, 3u32, 2u32];
        let states = injector.online_eval(dir.path(), &preprocess_out, &digits);

        assert_eq!(states.len(), 1 + input_count * batch_bits);

        let mut secret_matrix = injector.read_matrix(dir.path(), injector.secret_epsilon_id());
        assert_eq!(secret_matrix.size(), (1, DIAMOND_SECRET_SIZE));
        for (digit_idx, digit_value) in digits.iter().copied().enumerate() {
            let secret_mask = injector.read_matrix(
                dir.path(),
                &injector.digit_secret_id(digit_idx + 1, digit_value as usize),
            );
            assert_eq!(secret_mask.size(), (DIAMOND_SECRET_SIZE, DIAMOND_SECRET_SIZE));
            secret_matrix = secret_matrix * secret_mask;
        }
        let base_public_matrix = preprocess_out.final_public_matrix(&injector.params, 0);
        let base_selector =
            DCRTPolyMatrix::from_poly_vec_row(&params, vec![secret_matrix.entry(0, 0), k.clone()]);
        assert_eq!(states[0], base_selector * base_public_matrix);

        for digit_idx in 0..input_count {
            for bit_idx in 0..batch_bits {
                let state_idx = injector.bit_state_idx(digit_idx, bit_idx);
                let bit_value = ((digits[digit_idx] as usize) >> bit_idx) & 1;
                let bit_plaintext = TestPoly::from_usize_to_constant(&params, bit_value);
                let bit_public_matrix =
                    preprocess_out.final_public_matrix(&injector.params, state_idx);
                let bit_selector = DCRTPolyMatrix::from_poly_vec_row(
                    &params,
                    vec![secret_matrix.entry(0, 0), secret_matrix.entry(0, 0) * &bit_plaintext],
                );
                assert_eq!(states[state_idx], bit_selector * bit_public_matrix);
            }
        }
    }

    #[serial_test::serial]
    #[test]
    fn test_diamond_injector_retains_selected_transitions_and_journals_deleted_paths() {
        let params = DCRTPolyParams::default();
        let input_count = 1;
        let base = 4;
        let batch_bits = 2;
        let retained_digits = [2u32];
        let dir = tempdir().expect("temporary directory should be created");
        let injector = TestInjector::new(params.clone(), input_count, base, batch_bits, 4.578, 0.0);
        let k = TestPoly::const_one(&params);

        crate::utils::start_pre_delete_file_length_observer();
        let preprocess_out = injector.preprocess(dir.path(), &k, Some(&retained_digits));
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
        let state_cols = injector.state_col_size(&params);
        let chunk_count = crate::slot_transfer::bgg_pubkey::column_chunk_count(state_cols);
        let state_count = injector.state_count_at_level(1);
        let journal_records = injector.load_discarded_transition_tasks(dir.path());
        let mut retained_bytes = 0u128;
        for digit_value in 0..base {
            for state_idx in 0..state_count {
                for chunk_idx in 0..chunk_count {
                    let chunk_id =
                        injector.chunk_id(&injector.k_id(1, digit_value, state_idx), chunk_idx);
                    let task_index = injector.transition_task_index(
                        1,
                        digit_value,
                        state_idx,
                        chunk_idx,
                        chunk_count,
                    );
                    assert_eq!(
                        injector.matrix_exists(dir.path(), &chunk_id),
                        digit_value == retained_digits[0] as usize,
                        "only the selected transition should remain on disk"
                    );
                    if digit_value == retained_digits[0] as usize {
                        retained_bytes += u128::from(
                            std::fs::metadata(injector.matrix_path(dir.path(), &chunk_id))
                                .expect("retained transition artifact should exist")
                                .len(),
                        );
                    } else {
                        assert_eq!(
                            journal_records[&task_index], observed_deleted_bytes[&chunk_id],
                            "journal bytes must equal the same file's observed pre-deletion length"
                        );
                    }
                }
            }
        }
        let journal_path = injector.discard_journal_path(dir.path());
        let expected_discarded = (base - 1) * state_count * chunk_count;
        assert_eq!(
            std::fs::read(&journal_path).expect("discard journal should exist").len(),
            expected_discarded * 2 * size_of::<u64>()
        );
        let generated_bytes =
            injector.generated_transition_preimage_bytes(dir.path(), Some(&retained_digits));
        assert_eq!(
            generated_bytes,
            retained_bytes + observed_deleted_bytes.values().copied().map(u128::from).sum::<u128>(),
            "accounting must equal independently observed pre-deletion transition file lengths"
        );

        let mut journal = std::fs::OpenOptions::new()
            .append(true)
            .open(&journal_path)
            .expect("discard journal should open for torn-tail simulation");
        journal.write_all(&[0xa5; 7]).expect("torn discard-journal tail should be written");
        journal.sync_all().expect("torn discard-journal tail should sync");
        drop(journal);

        injector.preprocess(dir.path(), &k, Some(&retained_digits));
        assert_eq!(
            std::fs::read(&journal_path).expect("discard journal should remain readable").len(),
            expected_discarded * 2 * size_of::<u64>(),
            "resume must repair the torn tail without regenerating or re-journaling deleted transitions"
        );
        assert_eq!(
            injector.generated_transition_preimage_bytes(dir.path(), Some(&retained_digits)),
            generated_bytes,
            "resume must preserve the actual generated byte total"
        );
        let states = injector.online_eval(dir.path(), &preprocess_out, &retained_digits);
        assert_eq!(states.len(), 1 + input_count * batch_bits);
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_diamond_injector_repairs_incomplete_b_checkpoint_pair() {
        let params = DCRTPolyParams::default();
        let dir = tempdir().expect("temporary directory should be created");
        let injector = TestInjector::new(params.clone(), 1, 2, 1, 4.578, 0.0);
        let k = TestPoly::const_one(&params);
        injector.preprocess(dir.path(), &k, None);

        let matrix_id = injector.b_matrix_id(1, 0);
        let trapdoor_id = injector.b_trapdoor_id(1, 0);
        let complete_id = injector.b_checkpoint_complete_id(1, 0);
        assert!(injector.matrix_exists(dir.path(), &matrix_id));
        assert!(injector.bytes_exists(dir.path(), &trapdoor_id));
        assert!(injector.bytes_exists(dir.path(), &complete_id));

        injector.remove_bytes_checkpoint(dir.path(), &trapdoor_id);
        injector.remove_bytes_checkpoint(dir.path(), &complete_id);
        assert!(
            injector.matrix_exists(dir.path(), &matrix_id),
            "test setup must leave only the public matrix from an interrupted pair"
        );

        let (_trapdoor, matrix) = injector.load_or_sample_b_checkpoint(dir.path(), 1, 0);
        assert_eq!(
            injector.read_matrix(dir.path(), &matrix_id),
            matrix,
            "the returned public matrix must be the fully persisted replacement"
        );
        assert!(injector.bytes_exists(dir.path(), &trapdoor_id));
        assert!(
            injector.bytes_exists(dir.path(), &complete_id),
            "the pair completion marker must be written after both components"
        );
    }

    #[test]
    fn test_diamond_injector_simulate_output_error_bounds_matches_repeated_preimage_bound() {
        let params = DCRTPolyParams::default();
        let injector = TestInjector::new(params.clone(), 3, 4, 2, 6.0, 3.0);
        let batch_bits = injector.batch_bits();

        let simulated = injector.simulate_output_error_bounds();
        let state_cols = injector.state_col_size(&params);
        let gadget_cols = injector.gadget_col_size(&params);
        let ring_dim_sqrt = BigDecimal::from(params.ring_dimension() as u64)
            .sqrt()
            .expect("sqrt(ring_dimension) failed");
        let base = BigDecimal::from(BigInt::from(BigUint::from(1u64) << params.base_bits()));
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt,
            base,
            DIAMOND_SECRET_SIZE,
            params.modulus_digits(),
            params.modulus_digits(),
        ));
        let initial_sigma =
            BigDecimal::from_f64(injector.error_sigma).expect("error_sigma must be finite");
        let expected_initial =
            PolyMatrixNorm::sample_gauss(ctx.clone(), 1, state_cols, initial_sigma);
        let expected_transition_target_error = PolyMatrixNorm::sample_gauss(
            ctx.clone(),
            injector.state_row_size(),
            state_cols,
            BigDecimal::from_f64(injector.error_sigma).expect("error_sigma must be finite"),
        );
        let expected_preimage_sigma = compute_preimage_sigma(
            &ctx.ring_dim_sqrt,
            ctx.m_g as u64,
            &ctx.base,
            Some(injector.state_row_size() / DIAMOND_SECRET_SIZE),
            Some(injector.trapdoor_sigma),
        );
        let expected_transition = PolyMatrixNorm::fresh_preimage(
            ctx.clone(),
            state_cols,
            state_cols,
            expected_preimage_sigma.clone(),
            None,
        );
        let expected_output = PolyMatrixNorm::fresh_preimage(
            ctx.clone(),
            state_cols,
            gadget_cols,
            expected_preimage_sigma,
            None,
        );
        let expected_regular_selector = PolyMatrixNorm::new(
            ctx.clone(),
            injector.state_row_size(),
            injector.state_row_size(),
            BigDecimal::from(1u64),
            None,
        );
        let expected_base_selector = PolyMatrixNorm::new(
            ctx.clone(),
            injector.state_row_size(),
            injector.state_row_size(),
            BigDecimal::from(1u64),
            None,
        );
        let expected_special_selector = PolyMatrixNorm::new(
            ctx.clone(),
            injector.state_row_size(),
            injector.state_row_size(),
            BigDecimal::from(1u64),
            Some(DIAMOND_SECRET_SIZE),
        );
        let expected_initial_secret_factor = PolyMatrixNorm::new(
            ctx.clone(),
            1,
            injector.state_row_size(),
            BigDecimal::from(1u64),
            None,
        );

        let mut expected_secret_factors = vec![expected_initial_secret_factor];
        let mut expected_state_errors = vec![expected_initial];
        let advance_expected_state =
            |prev_secret: &[PolyMatrixNorm], prev_state_errors: &[PolyMatrixNorm]| {
                let mut next_secret = prev_secret
                    .iter()
                    .enumerate()
                    .map(|(state_idx, secret)| {
                        let selector = if state_idx == 0 {
                            &expected_base_selector
                        } else {
                            &expected_regular_selector
                        };
                        secret.clone() * selector
                    })
                    .collect::<Vec<_>>();
                let mut next_state_errors = prev_secret
                    .iter()
                    .zip(prev_state_errors.iter())
                    .map(|(secret, state_error)| {
                        state_error.clone() * &expected_transition +
                            secret.clone() * &expected_transition_target_error
                    })
                    .collect::<Vec<_>>();

                for _ in 0..batch_bits {
                    let born_secret = prev_secret[0].clone() * &expected_special_selector;
                    let born_state_error = prev_state_errors[0].clone() * &expected_transition +
                        prev_secret[0].clone() * &expected_transition_target_error;
                    next_secret.push(born_secret);
                    next_state_errors.push(born_state_error);
                }

                (next_secret, next_state_errors)
            };

        for _ in 1..=injector.input_count {
            let (next_secret, next_state_errors) =
                advance_expected_state(&expected_secret_factors, &expected_state_errors);
            expected_secret_factors = next_secret;
            expected_state_errors = next_state_errors;
        }

        assert_poly_matrix_bounds_eq(&simulated.state_errors, &expected_state_errors);
        assert_poly_matrix_bounds_eq(&simulated.secret_state_factors, &expected_secret_factors);
        assert_poly_matrix_bound_eq(&simulated.output_preimage, &expected_output);
    }

    #[test]
    fn test_diamond_injector_preprocess_refreshes_message_dependent_initial_state() {
        let params = DCRTPolyParams::default();
        let injector = TestInjector::new(params.clone(), 1, 2, 1, 4.578, 0.0);
        let dir = tempdir().expect("temporary directory should be created");
        let zero = TestPoly::const_zero(&params);
        let one = TestPoly::const_one(&params);

        injector.preprocess(dir.path(), &one, None);
        let initial_one = injector.read_matrix(dir.path(), injector.p_epsilon_id());
        injector.preprocess(dir.path(), &zero, None);
        let initial_zero = injector.read_matrix(dir.path(), injector.p_epsilon_id());

        let secret = injector.read_matrix(dir.path(), injector.secret_epsilon_id());
        let b0 = injector.read_matrix(dir.path(), &injector.b_matrix_id(0, 0));
        let expected_zero =
            DCRTPolyMatrix::from_poly_vec_row(&params, vec![secret.entry(0, 0), zero]) * &b0;
        let expected_one =
            DCRTPolyMatrix::from_poly_vec_row(&params, vec![secret.entry(0, 0), one]) * &b0;

        assert_eq!(initial_one, expected_one);
        assert_eq!(initial_zero, expected_zero);
        assert_ne!(initial_one, initial_zero);
    }

    #[test]
    #[ignore = "metrics-style reporting test; run with --ignored --nocapture"]
    fn test_diamond_injector_large_output_error_metrics() {
        let ring_dim = 1u32 << 16;
        let crt_depth = 60usize;
        let crt_bits = 28usize;
        let base_bits = 14u32;
        let input_count = 32usize;
        let digit_bits = 8u32;
        let input_base = 1usize << digit_bits;
        let params = DCRTPolyParams::new(ring_dim, crt_depth, crt_bits, base_bits);
        let injector = TestInjector::new(
            params.clone(),
            input_count,
            input_base,
            digit_bits as usize,
            4.578,
            4.578,
        );

        let simulated = injector.simulate_output_error_bounds();
        let projected_errors = simulated
            .state_errors
            .iter()
            .map(|state_error| state_error.clone() * &simulated.output_preimage)
            .collect::<Vec<_>>();
        let max_error = projected_errors
            .iter()
            .map(|norm| norm.maximum_coefficient_bound())
            .max()
            .expect("state error list must be non-empty");

        println!(
            "diamond injector output-error metrics: ring_dim={ring_dim}, crt_depth={crt_depth}, crt_bits={crt_bits}, base_bits={base_bits}, digit_bits={digit_bits}, input_base={input_base}, input_count={input_count}, output_secret_size={DIAMOND_SECRET_SIZE}, state_row_size={}",
            injector.state_row_size()
        );
        println!(
            "diamond injector output-error bits: max_projected={}",
            bigdecimal_bits_ceil(&max_error),
        );
    }
}
