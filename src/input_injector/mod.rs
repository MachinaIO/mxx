#[cfg(not(feature = "gpu"))]
use crate::slot_transfer::bgg_pubkey::column_chunk_count;
use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::bgg_pubkey::column_chunk_bounds,
};
use serde::{Deserialize, Serialize};
use std::{fs, marker::PhantomData, path::Path};

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
    fn preprocess(&self, dir_path: &Path, k: &P) -> Self::PreprocessOut;

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
}

#[derive(Debug, Clone)]
/// Compact in-memory data returned by Diamond preprocessing.
///
/// `hash_key` identifies the hash-derived transition public matrices, while
/// `final_trapdoor` and `final_pub_matrix` are the trapdoor pair for the final
/// Diamond state basis. Callers use that pair to sample their own final output
/// projection preimages.
pub struct DiamondInjectorPreprocessOut<M, T>
where
    M: PolyMatrix,
{
    pub hash_key: [u8; 32],
    pub final_trapdoor: T,
    pub final_pub_matrix: M,
}

impl<M, US, HS, TS> DiamondInjector<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        params: <M::P as Poly>::Params,
        input_count: usize,
        base: usize,
        trapdoor_sigma: f64,
        error_sigma: f64,
    ) -> Self {
        assert!(base > 0, "DiamondInjector base must be positive");
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
        fs::write(self.metadata_path(dir_path), bytes)
            .expect("DiamondInjector metadata should be written");
    }

    fn read_metadata(&self, dir_path: &Path) -> DiamondInjectorMetadata {
        let bytes = fs::read(self.metadata_path(dir_path))
            .expect("DiamondInjector metadata should have been written");
        serde_json::from_slice(&bytes).expect("DiamondInjector metadata should decode")
    }

    fn write_matrix(&self, dir_path: &Path, id: &str, matrix: &M) {
        self.write_matrix_bytes(dir_path, id, &matrix.to_compact_bytes());
    }

    fn write_matrix_bytes(&self, dir_path: &Path, id: &str, bytes: &[u8]) {
        fs::write(self.matrix_path(dir_path, id), bytes)
            .unwrap_or_else(|err| panic!("DiamondInjector failed to write matrix {id}: {err}"));
    }

    fn read_matrix(&self, dir_path: &Path, id: &str) -> M {
        let bytes = self.read_matrix_bytes(dir_path, id);
        M::from_compact_bytes(&self.params, &bytes)
    }

    fn read_matrix_bytes(&self, dir_path: &Path, id: &str) -> Vec<u8> {
        fs::read(self.matrix_path(dir_path, id))
            .unwrap_or_else(|err| panic!("DiamondInjector failed to read matrix {id}: {err}"))
    }

    fn write_bytes(&self, dir_path: &Path, id: &str, bytes: &[u8]) {
        fs::write(self.bytes_path(dir_path, id), bytes)
            .unwrap_or_else(|err| panic!("DiamondInjector failed to write bytes {id}: {err}"));
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
            .checked_add(self.gadget_col_size(params))
            .expect("DiamondInjector state column count overflow")
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

    fn b_matrix_id(&self, level: usize) -> String {
        format!("diamond_b_tensor_{level}")
    }

    fn b_trapdoor_id(&self, level: usize) -> String {
        format!("diamond_b_tensor_{level}_trapdoor")
    }

    fn p_epsilon_id(&self) -> &'static str {
        "diamond_p_epsilon_tensor_0"
    }

    fn k_plaintext_id(&self) -> &'static str {
        "diamond_k_plaintext"
    }

    fn preprocess_hash_key_id(&self) -> &'static str {
        "diamond_preprocess_hash_key"
    }

    fn k_id(&self, level: usize, digit_value: usize, state_idx: usize) -> String {
        format!("diamond_k_bit_tensor_{level}_{digit_value}_{state_idx}")
    }

    fn load_or_sample_preprocess_hash_key(&self, dir_path: &Path) -> [u8; 32] {
        let id = self.preprocess_hash_key_id();
        if self.bytes_exists(dir_path, id) {
            let bytes = self.read_bytes(dir_path, id);
            return bytes
                .as_slice()
                .try_into()
                .unwrap_or_else(|_| panic!("DiamondInjector preprocess hash key must be 32 bytes"));
        }
        let hash_key = rand::random::<[u8; 32]>();
        self.write_bytes(dir_path, id, &hash_key);
        hash_key
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
        assert!(
            self.base >= 2 && self.base.is_power_of_two(),
            "DiamondInjector base must be a power of two greater than one for bit batching"
        );
        self.base.trailing_zeros() as usize
    }

    fn expanded_state_count_after_level(&self, level: usize) -> usize {
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
    fn load_or_sample_b_checkpoint(&self, dir_path: &Path, level: usize) -> (TS::Trapdoor, M) {
        // Checkpoint one trapdoor public matrix per level. Later preimage
        // sampling uses this stored pair as the source side of a
        // preimage_extend call, while the hashed W blocks are reconstructed on
        // demand.
        let matrix_id = self.b_matrix_id(level);
        let trapdoor_id = self.b_trapdoor_id(level);
        if self.matrix_exists(dir_path, &matrix_id) && self.bytes_exists(dir_path, &trapdoor_id) {
            let trapdoor =
                TS::trapdoor_from_bytes(&self.params, &self.read_bytes(dir_path, &trapdoor_id))
                    .unwrap_or_else(|| {
                        panic!(
                            "DiamondInjector failed to decode trapdoor checkpoint for level {level}"
                        )
                    });
            let matrix = self.read_matrix(dir_path, &matrix_id);
            return (trapdoor, matrix);
        }

        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let (trapdoor, matrix) = trap_sampler.trapdoor(&self.params, self.state_row_size());
        self.write_bytes(dir_path, &trapdoor_id, &TS::trapdoor_to_bytes(&trapdoor));
        self.write_matrix(dir_path, &matrix_id, &matrix);
        (trapdoor, matrix)
    }

    #[cfg(feature = "gpu")]
    fn load_or_sample_b_checkpoint_bytes(
        &self,
        dir_path: &Path,
        level: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let matrix_id = self.b_matrix_id(level);
        let trapdoor_id = self.b_trapdoor_id(level);
        if self.matrix_exists(dir_path, &matrix_id) && self.bytes_exists(dir_path, &trapdoor_id) {
            return (
                self.read_matrix_bytes(dir_path, &matrix_id),
                self.read_bytes(dir_path, &trapdoor_id),
            );
        }

        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let (trapdoor, matrix) = trap_sampler.trapdoor(&self.params, self.state_row_size());
        let trapdoor_bytes = TS::trapdoor_to_bytes(&trapdoor);
        let matrix_bytes = matrix.to_compact_bytes();
        self.write_bytes(dir_path, &trapdoor_id, &trapdoor_bytes);
        self.write_matrix_bytes(dir_path, &matrix_id, &matrix_bytes);
        (matrix_bytes, trapdoor_bytes)
    }

    fn sample_w_block_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        hash_key: [u8; 32],
        block_idx: usize,
        level: usize,
    ) -> M {
        // Reconstruct one deterministic hashed extension block W_{block_idx,
        // level}. This keeps preprocessing storage focused on preimages rather
        // than on the hash-derived public side.
        HS::new().sample_hash(
            params,
            hash_key,
            format!("diamond_w_{block_idx}_{level}"),
            self.state_row_size(),
            self.gadget_col_size(params),
            DistType::FinRingDist,
        )
    }

    fn sample_w_block_columns_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        hash_key: [u8; 32],
        block_idx: usize,
        level: usize,
        col_start: usize,
        col_len: usize,
    ) -> M {
        HS::new().sample_hash_columns(
            params,
            hash_key,
            format!("diamond_w_{block_idx}_{level}"),
            self.state_row_size(),
            self.gadget_col_size(params),
            col_start,
            col_len,
            DistType::FinRingDist,
        )
    }

    fn state_public_chunk_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        hash_key: [u8; 32],
        b_matrix: &M,
        block_idx: usize,
        level: usize,
        chunk_idx: usize,
    ) -> M {
        // Materialize a column slice of the effective public matrix
        // (B_level | W_{block_idx, level}). Transition matrices, one-outputs,
        // bit-outputs, and decoder outputs are all sampled against this
        // concatenated view.
        let total_cols = self.state_col_size(params);
        let b_cols = self.b_public_col_size(params);
        let (col_start, col_len) = column_chunk_bounds(total_cols, chunk_idx);
        let col_end = col_start + col_len;
        if col_end <= b_cols {
            return b_matrix.slice_columns(col_start, col_end);
        }
        if col_start >= b_cols {
            return self.sample_w_block_columns_with_params(
                params,
                hash_key,
                block_idx,
                level,
                col_start - b_cols,
                col_len,
            );
        }
        let b_chunk = b_matrix.slice_columns(col_start, b_cols);
        let w_chunk = self.sample_w_block_columns_with_params(
            params,
            hash_key,
            block_idx,
            level,
            0,
            col_end - b_cols,
        );
        b_chunk.concat_columns(&[&w_chunk])
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

    fn build_initial_encoding(
        &self,
        hash_key: [u8; 32],
        b0_matrix: &M,
        secret_epsilon: &M,
        k: &M::P,
    ) -> M {
        // Build the state that represents the empty input prefix. It is the
        // only online-evaluation seed that exists before any digit is chosen.
        let selector =
            M::from_poly_vec_row(&self.params, vec![secret_epsilon.entry(0, 0), k.clone()]);
        let w00 = self.sample_w_block_with_params(&self.params, hash_key, 0, 0);
        let mut p_epsilon = selector * b0_matrix.concat_columns(&[&w00]);
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
        hash_key: [u8; 32],
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
        let public_chunk = self.state_public_chunk_with_params(
            params, hash_key, b_matrix, state_idx, level, chunk_idx,
        );
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
    fn has_all_chunks(&self, dir_path: &Path, id: &str, total_cols: usize) -> bool {
        (0..column_chunk_count(total_cols))
            .all(|chunk_idx| self.matrix_exists(dir_path, &self.chunk_id(id, chunk_idx)))
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

    fn preprocess(&self, dir_path: &Path, k: &M::P) -> Self::PreprocessOut {
        let hash_key = self.load_or_sample_preprocess_hash_key(dir_path);
        #[cfg(feature = "gpu")]
        {
            self.preprocess_gpu(dir_path, hash_key, k);
            let (final_pub_matrix_bytes, final_trapdoor_bytes) =
                self.load_or_sample_b_checkpoint_bytes(dir_path, self.input_count);
            return DiamondInjectorPreprocessOut {
                hash_key,
                final_trapdoor: TS::trapdoor_from_bytes(&self.params, &final_trapdoor_bytes)
                    .expect("DiamondInjector final trapdoor checkpoint must decode"),
                final_pub_matrix: M::from_compact_bytes(&self.params, &final_pub_matrix_bytes),
            };
        }

        #[cfg(not(feature = "gpu"))]
        {
            self.ensure_dir(dir_path);
            self.write_metadata(
                dir_path,
                &DiamondInjectorMetadata { input_count: self.input_count, base: self.base },
            );
            self.write_bytes(dir_path, self.k_plaintext_id(), &k.to_compact_bytes());

            let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
            let mut b_checkpoints = Vec::with_capacity(self.input_count + 1);
            let mut trapdoors = Vec::with_capacity(self.input_count + 1);
            // Load or sample the per-level trapdoor checkpoints that all later
            // preimage samplers will reference.
            for level in 0..=self.input_count {
                let (trapdoor, b_matrix) = self.load_or_sample_b_checkpoint(dir_path, level);
                trapdoors.push(trapdoor);
                b_checkpoints.push(b_matrix);
            }

            // Sample the empty-prefix seed once and persist it if it does not
            // already exist.
            let secret_epsilon =
                self.load_or_sample_secret_epsilon(dir_path, self.secret_epsilon_id());
            if !self.matrix_exists(dir_path, self.p_epsilon_id()) {
                self.write_matrix(
                    dir_path,
                    self.p_epsilon_id(),
                    &self.build_initial_encoding(hash_key, &b_checkpoints[0], &secret_epsilon, k),
                );
            }

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
                    let prev_ext_w0 =
                        self.sample_w_block_with_params(&self.params, hash_key, 0, level - 1);
                    for state_idx in 0..self.expanded_state_count_after_level(level) {
                        let k_id = self.k_id(level, digit_value, state_idx);
                        if self.has_all_chunks(dir_path, &k_id, state_cols) {
                            continue;
                        }
                        let ext_matrix = if self.new_bit_idx_for_state(level, state_idx).is_some() {
                            &prev_ext_w0
                        } else {
                            &self.sample_w_block_with_params(
                                &self.params,
                                hash_key,
                                state_idx,
                                level - 1,
                            )
                        };
                        for chunk_idx in 0..column_chunk_count(state_cols) {
                            let chunk_id = self.chunk_id(&k_id, chunk_idx);
                            if self.matrix_exists(dir_path, &chunk_id) {
                                continue;
                            }
                            let target_chunk = self.build_k_target_chunk_with_params(
                                &self.params,
                                hash_key,
                                level,
                                digit_value,
                                state_idx,
                                &secret_mask,
                                &b_checkpoints[level],
                                chunk_idx,
                            );
                            let k_chunk = trap_sampler.preimage_extend(
                                &self.params,
                                &trapdoors[level - 1],
                                &b_checkpoints[level - 1],
                                ext_matrix,
                                &target_chunk,
                            );
                            self.write_matrix(dir_path, &chunk_id, &k_chunk);
                        }
                    }
                }
            }
            DiamondInjectorPreprocessOut {
                hash_key,
                final_trapdoor: trapdoors
                    .pop()
                    .expect("DiamondInjector must keep final trapdoor checkpoint"),
                final_pub_matrix: b_checkpoints
                    .pop()
                    .expect("DiamondInjector must keep final public matrix checkpoint"),
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
                self.read_bytes(dir_path, self.preprocess_hash_key_id()).as_slice(),
                &preprocess_out.hash_key,
                "DiamondInjector online_eval preprocess hash key mismatch"
            );
            let metadata = self.read_metadata(dir_path);
            assert_eq!(
                metadata.input_count, self.input_count,
                "DiamondInjector metadata input count mismatch"
            );
            assert_eq!(metadata.base, self.base, "DiamondInjector metadata base mismatch");

            // Start from the persisted empty-prefix seed.
            let mut states = vec![self.read_matrix(dir_path, self.p_epsilon_id())];
            let state_cols = self.state_col_size(&self.params);
            for (digit_idx, digit_value) in input_digits.iter().copied().enumerate() {
                let level = digit_idx + 1;
                let prev_states = std::mem::take(&mut states);
                let prev_p0 = prev_states[0].clone();
                let mut next_states =
                    Vec::with_capacity(self.expanded_state_count_after_level(level));
                // Advance every currently alive branch through the transition
                // matrix for the chosen digit, and spawn the new branch that
                // records each bit of the current digit.
                for state_idx in 0..self.expanded_state_count_after_level(level) {
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
        __PAIR, __TestState,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        simulator::{
            SimulatorContext, error_norm::compute_preimage_norm, poly_matrix_norm::PolyMatrixNorm,
        },
        utils::bigdecimal_bits_ceil,
    };
    use bigdecimal::BigDecimal;
    use keccak_asm::Keccak256;
    use num_bigint::{BigInt, BigUint};
    use num_traits::FromPrimitive;
    use std::sync::Arc;
    use tempfile::tempdir;

    type TestInjector = DiamondInjector<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
    >;

    #[sequential_test::sequential]
    #[test]
    fn test_diamond_injector_online_eval_returns_exact_bgg_relations() {
        type TestPoly = <DCRTPolyMatrix as PolyMatrix>::P;

        let params = DCRTPolyParams::default();
        let input_count = 3;
        let base = 4;
        let batch_bits = 2;
        let dir = tempdir().expect("temporary directory should be created");

        let injector = TestInjector::new(params.clone(), input_count, base, 4.578, 0.0);

        let k = TestPoly::from_usize_to_constant(&params, 3);

        let preprocess_out = injector.preprocess(dir.path(), &k);

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
        let base_public_matrix =
            preprocess_out.final_pub_matrix.concat_columns(&[&injector
                .sample_w_block_with_params(&params, preprocess_out.hash_key, 0, input_count)]);
        let base_selector =
            DCRTPolyMatrix::from_poly_vec_row(&params, vec![secret_matrix.entry(0, 0), k.clone()]);
        assert_eq!(states[0], base_selector * base_public_matrix);

        for digit_idx in 0..input_count {
            for bit_idx in 0..batch_bits {
                let state_idx = injector.bit_state_idx(digit_idx, bit_idx);
                let bit_value = ((digits[digit_idx] as usize) >> bit_idx) & 1;
                let bit_plaintext = TestPoly::from_usize_to_constant(&params, bit_value);
                let bit_public_matrix =
                    preprocess_out.final_pub_matrix.concat_columns(&[&injector
                        .sample_w_block_with_params(
                            &params,
                            preprocess_out.hash_key,
                            state_idx,
                            input_count,
                        )]);
                let bit_selector = DCRTPolyMatrix::from_poly_vec_row(
                    &params,
                    vec![secret_matrix.entry(0, 0), secret_matrix.entry(0, 0) * &bit_plaintext],
                );
                assert_eq!(states[state_idx], bit_selector * bit_public_matrix);
            }
        }
    }

    #[test]
    fn test_diamond_injector_simulate_output_error_bounds_matches_repeated_preimage_bound() {
        let params = DCRTPolyParams::default();
        let injector = TestInjector::new(params.clone(), 3, 4, 4.578, 3.0);
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
        let expected_preimage_norm = compute_preimage_norm(
            &ctx.ring_dim_sqrt,
            ctx.m_g as u64,
            &ctx.base,
            Some(injector.state_row_size() / DIAMOND_SECRET_SIZE),
            None,
        );
        let expected_transition = PolyMatrixNorm::new(
            ctx.clone(),
            state_cols,
            state_cols,
            expected_preimage_norm.clone(),
            None,
        );
        let expected_output =
            PolyMatrixNorm::new(ctx.clone(), state_cols, gadget_cols, expected_preimage_norm, None);
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

        assert_eq!(simulated.state_errors, expected_state_errors);
        assert_eq!(simulated.output_preimage, expected_output);
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
        let injector = TestInjector::new(params.clone(), input_count, input_base, 4.578, 4.578);

        let simulated = injector.simulate_output_error_bounds();
        let projected_errors = simulated
            .state_errors
            .iter()
            .map(|state_error| state_error.clone() * &simulated.output_preimage)
            .collect::<Vec<_>>();
        let max_error = projected_errors
            .iter()
            .map(|norm| norm.poly_norm.norm.clone())
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
