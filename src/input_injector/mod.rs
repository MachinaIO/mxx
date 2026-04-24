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
use std::{fs, marker::PhantomData, path::PathBuf};

#[cfg(feature = "gpu")]
#[path = "diamond_gpu.rs"]
mod gpu;
mod simulation;

pub use simulation::DiamondInputErrorSimulation;

const DIAMOND_SECRET_SIZE: usize = 1;

pub trait InputInjector<K, E> {
    /// Precompute and persist the auxiliary matrices needed to inject one
    /// constant value, every input digit position, and every decoder output.
    fn preprocess(&self, one: &K, input_digits: &[K], decoders: &[K]);
    /// Rebuild the final injected encodings for one, the chosen input digits,
    /// and the decoder outputs from the persisted preprocessing artifacts.
    fn online_eval(
        &self,
        input_digits: &[u32],
        one: &K,
        input_digit_pubkeys: &[K],
        decoders: &[K],
    ) -> (E, Vec<E>, Vec<E>);
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
    pub hash_key: [u8; 32],
    pub input_count: usize,
    pub base: usize,
    pub decoder_count: usize,
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    pub dir_path: PathBuf,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DiamondInjectorMetadata {
    input_count: usize,
    base: usize,
    decoder_count: usize,
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
        hash_key: [u8; 32],
        input_count: usize,
        base: usize,
        decoder_count: usize,
        trapdoor_sigma: f64,
        error_sigma: f64,
        dir_path: PathBuf,
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
            hash_key,
            input_count,
            base,
            decoder_count,
            trapdoor_sigma,
            error_sigma,
            dir_path,
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

    fn ensure_dir(&self) {
        fs::create_dir_all(&self.dir_path).unwrap_or_else(|err| {
            panic!(
                "DiamondInjector failed to create preprocessing directory {}: {err}",
                self.dir_path.display()
            )
        });
    }

    fn metadata_path(&self) -> PathBuf {
        self.dir_path.join("diamond_injector_metadata.json")
    }

    fn matrix_path(&self, id: &str) -> PathBuf {
        self.dir_path.join(format!("{id}.matrixbin"))
    }

    fn bytes_path(&self, id: &str) -> PathBuf {
        self.dir_path.join(format!("{id}.bytesbin"))
    }

    fn write_metadata(&self, metadata: &DiamondInjectorMetadata) {
        let bytes =
            serde_json::to_vec_pretty(metadata).expect("DiamondInjector metadata should serialize");
        fs::write(self.metadata_path(), bytes).expect("DiamondInjector metadata should be written");
    }

    fn read_metadata(&self) -> DiamondInjectorMetadata {
        let bytes = fs::read(self.metadata_path())
            .expect("DiamondInjector metadata should have been written");
        serde_json::from_slice(&bytes).expect("DiamondInjector metadata should decode")
    }

    fn write_matrix(&self, id: &str, matrix: &M) {
        self.write_matrix_bytes(id, &matrix.to_compact_bytes());
    }

    fn write_matrix_bytes(&self, id: &str, bytes: &[u8]) {
        fs::write(self.matrix_path(id), bytes)
            .unwrap_or_else(|err| panic!("DiamondInjector failed to write matrix {id}: {err}"));
    }

    fn read_matrix(&self, id: &str) -> M {
        let bytes = self.read_matrix_bytes(id);
        M::from_compact_bytes(&self.params, &bytes)
    }

    fn read_matrix_bytes(&self, id: &str) -> Vec<u8> {
        fs::read(self.matrix_path(id))
            .unwrap_or_else(|err| panic!("DiamondInjector failed to read matrix {id}: {err}"))
    }

    fn write_bytes(&self, id: &str, bytes: &[u8]) {
        fs::write(self.bytes_path(id), bytes)
            .unwrap_or_else(|err| panic!("DiamondInjector failed to write bytes {id}: {err}"));
    }

    fn read_bytes(&self, id: &str) -> Vec<u8> {
        fs::read(self.bytes_path(id))
            .unwrap_or_else(|err| panic!("DiamondInjector failed to read bytes {id}: {err}"))
    }

    fn matrix_exists(&self, id: &str) -> bool {
        self.matrix_path(id).exists()
    }

    fn bytes_exists(&self, id: &str) -> bool {
        self.bytes_path(id).exists()
    }

    fn state_row_size(&self) -> usize {
        DIAMOND_SECRET_SIZE.checked_mul(2).expect("DiamondInjector state row size overflow")
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
        "diamond_secret_epsilon"
    }

    fn digit_secret_id(&self, level: usize, digit_value: usize) -> String {
        format!("diamond_secret_{level}_{digit_value}")
    }

    fn b_matrix_id(&self, level: usize) -> String {
        format!("diamond_b_{level}")
    }

    fn b_trapdoor_id(&self, level: usize) -> String {
        format!("diamond_b_{level}_trapdoor")
    }

    fn p_epsilon_id(&self) -> &'static str {
        "diamond_p_epsilon_0"
    }

    fn k_id(&self, level: usize, digit_value: usize, state_idx: usize) -> String {
        format!("diamond_k_{level}_{digit_value}_{state_idx}")
    }

    fn l_id(&self, input_idx: usize) -> String {
        format!("diamond_l_{input_idx}")
    }

    fn m_id(&self, decoder_idx: usize) -> String {
        format!("diamond_m_{decoder_idx}")
    }

    fn sample_secret_mask_with_params(&self, params: &<M::P as Poly>::Params) -> M {
        US::new().sample_uniform(
            params,
            DIAMOND_SECRET_SIZE,
            DIAMOND_SECRET_SIZE,
            DistType::TernaryDist,
        )
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

    fn validate_lengths(&self, input_pubkeys: &[BggPublicKey<M>], decoders: &[BggPublicKey<M>]) {
        assert_eq!(
            input_pubkeys.len(),
            self.input_count,
            "DiamondInjector expected {} input public keys but received {}",
            self.input_count,
            input_pubkeys.len()
        );
        assert_eq!(
            decoders.len(),
            self.decoder_count,
            "DiamondInjector expected {} decoder public keys but received {}",
            self.decoder_count,
            decoders.len()
        );
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
    fn load_or_sample_secret_mask(&self, id: &str) -> M {
        // Checkpoint the small ternary matrix that drives either the initial
        // state or one digit/value-specific transition.
        if self.matrix_exists(id) {
            self.read_matrix(id)
        } else {
            let secret = self.sample_secret_mask_with_params(&self.params);
            self.write_matrix(id, &secret);
            secret
        }
    }

    #[cfg(feature = "gpu")]
    fn load_or_sample_secret_mask_bytes(&self, id: &str) -> Vec<u8> {
        if self.matrix_exists(id) {
            self.read_matrix_bytes(id)
        } else {
            let secret = self.sample_secret_mask_with_params(&self.params);
            let bytes = secret.to_compact_bytes();
            self.write_matrix_bytes(id, &bytes);
            bytes
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn load_or_sample_b_checkpoint(&self, level: usize) -> (TS::Trapdoor, M) {
        // Checkpoint one trapdoor public matrix per level. Later preimage
        // sampling uses this stored pair as the source side of a
        // preimage_extend call, while the hashed W blocks are reconstructed on
        // demand.
        let matrix_id = self.b_matrix_id(level);
        let trapdoor_id = self.b_trapdoor_id(level);
        if self.matrix_exists(&matrix_id) && self.bytes_exists(&trapdoor_id) {
            let trapdoor = TS::trapdoor_from_bytes(&self.params, &self.read_bytes(&trapdoor_id))
                .unwrap_or_else(|| {
                    panic!("DiamondInjector failed to decode trapdoor checkpoint for level {level}")
                });
            let matrix = self.read_matrix(&matrix_id);
            return (trapdoor, matrix);
        }

        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let (trapdoor, matrix) = trap_sampler.trapdoor(&self.params, self.state_row_size());
        self.write_bytes(&trapdoor_id, &TS::trapdoor_to_bytes(&trapdoor));
        self.write_matrix(&matrix_id, &matrix);
        (trapdoor, matrix)
    }

    #[cfg(feature = "gpu")]
    fn load_or_sample_b_checkpoint_bytes(&self, level: usize) -> (Vec<u8>, Vec<u8>) {
        let matrix_id = self.b_matrix_id(level);
        let trapdoor_id = self.b_trapdoor_id(level);
        if self.matrix_exists(&matrix_id) && self.bytes_exists(&trapdoor_id) {
            return (self.read_matrix_bytes(&matrix_id), self.read_bytes(&trapdoor_id));
        }

        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let (trapdoor, matrix) = trap_sampler.trapdoor(&self.params, self.state_row_size());
        let trapdoor_bytes = TS::trapdoor_to_bytes(&trapdoor);
        let matrix_bytes = matrix.to_compact_bytes();
        self.write_bytes(&trapdoor_id, &trapdoor_bytes);
        self.write_matrix_bytes(&matrix_id, &matrix_bytes);
        (matrix_bytes, trapdoor_bytes)
    }

    fn sample_w_block_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        block_idx: usize,
        level: usize,
    ) -> M {
        // Reconstruct one deterministic hashed extension block W_{block_idx,
        // level}. This keeps preprocessing storage focused on preimages rather
        // than on the hash-derived public side.
        HS::new().sample_hash(
            params,
            self.hash_key,
            format!("diamond_w_{block_idx}_{level}"),
            self.state_row_size(),
            self.gadget_col_size(params),
            DistType::FinRingDist,
        )
    }

    fn sample_w_block_columns_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        block_idx: usize,
        level: usize,
        col_start: usize,
        col_len: usize,
    ) -> M {
        HS::new().sample_hash_columns(
            params,
            self.hash_key,
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
        b_matrix: &M,
        block_idx: usize,
        level: usize,
        chunk_idx: usize,
    ) -> M {
        // Materialize a column slice of the effective public matrix
        // (B_level | W_{block_idx, level}). Transition matrices, one-outputs,
        // digit-outputs, and decoder outputs are all sampled against this
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
                block_idx,
                level,
                col_start - b_cols,
                col_len,
            );
        }
        let b_chunk = b_matrix.slice_columns(col_start, b_cols);
        let w_chunk =
            self.sample_w_block_columns_with_params(params, block_idx, level, 0, col_end - b_cols);
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

    fn special_transition_selector_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        digit_value: usize,
        secret_mask: &M,
    ) -> M {
        let digit = M::P::from_usize_to_constant(params, digit_value);
        // The newly created branch for the current digit must carry both a
        // constant term and the chosen digit value. Because the code multiplies
        // states from the left, the selector is written in row-vector form.
        let zero_block = M::zero(params, DIAMOND_SECRET_SIZE, DIAMOND_SECRET_SIZE);
        let top = secret_mask.concat_columns(&[&(secret_mask.clone() * &digit)]);
        let bottom = zero_block.clone().concat_columns(&[&zero_block]);
        top.concat_rows(&[&bottom])
    }

    fn initial_selector_with_params(&self, params: &<M::P as Poly>::Params, secret_mask: &M) -> M {
        let zero_block = M::zero(params, DIAMOND_SECRET_SIZE, DIAMOND_SECRET_SIZE);
        secret_mask.concat_columns(&[&zero_block])
    }

    fn build_initial_encoding(&self, b0_matrix: &M, secret_epsilon: &M) -> M {
        // Build the state that represents the empty input prefix. It is the
        // only online-evaluation seed that exists before any digit is chosen.
        let selector = self.initial_selector_with_params(&self.params, secret_epsilon);
        let w00 = self.sample_w_block_with_params(&self.params, 0, 0);
        let mut p_epsilon = selector * b0_matrix.concat_columns(&[&w00]);
        p_epsilon.add_in_place(&self.sample_error_matrix_with_dims(
            &self.params,
            DIAMOND_SECRET_SIZE,
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
        // while the newly born branch for the current digit uses the special
        // selector above so the chosen digit value is embedded into that path.
        let public_chunk = self.state_public_chunk_with_params(
            params,
            b_matrix,
            state_idx.min(level),
            level,
            chunk_idx,
        );
        let selector = if state_idx == level {
            self.special_transition_selector_with_params(params, digit_value, secret_mask)
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

    fn build_one_target_chunk_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        one: &BggPublicKey<M>,
        gadget: &M,
        chunk_idx: usize,
    ) -> M {
        // Build one chunk of the final matrix that turns the surviving base
        // branch into an encoding of the constant one value.
        let total_cols = self.gadget_col_size(params);
        let (col_start, col_len) = column_chunk_bounds(total_cols, chunk_idx);
        let col_end = col_start + col_len;
        let target = one.matrix.slice_columns(col_start, col_end) -
            &gadget.slice_columns(col_start, col_end);
        let zero_block = M::zero(params, DIAMOND_SECRET_SIZE, col_len);
        target.concat_rows(&[&zero_block])
    }

    fn build_input_target_chunk_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        pubkey: &BggPublicKey<M>,
        gadget: &M,
        chunk_idx: usize,
    ) -> M {
        // Build one chunk of the final matrix that turns a digit-specific
        // branch into a BGG encoding under that digit position's public key.
        let total_cols = self.gadget_col_size(params);
        let (col_start, col_len) = column_chunk_bounds(total_cols, chunk_idx);
        let col_end = col_start + col_len;
        let pubkey_chunk = pubkey.matrix.slice_columns(col_start, col_end);
        let gadget_chunk = gadget.slice_columns(col_start, col_end);
        let neg_gadget_chunk = -gadget_chunk;
        pubkey_chunk.concat_rows(&[&neg_gadget_chunk])
    }

    fn build_decoder_target_chunk_with_params(
        &self,
        params: &<M::P as Poly>::Params,
        pubkey: &BggPublicKey<M>,
        chunk_idx: usize,
    ) -> M {
        // Build one chunk of the final matrix that maps the surviving base
        // branch to one decoder output public key.
        let total_cols = self.gadget_col_size(params);
        let (col_start, col_len) = column_chunk_bounds(total_cols, chunk_idx);
        let col_end = col_start + col_len;
        let pubkey_chunk = pubkey.matrix.slice_columns(col_start, col_end);
        let zero_block = M::zero(params, DIAMOND_SECRET_SIZE, col_len);
        pubkey_chunk.concat_rows(&[&zero_block])
    }

    #[cfg(not(feature = "gpu"))]
    fn has_all_chunks(&self, id: &str, total_cols: usize) -> bool {
        (0..column_chunk_count(total_cols))
            .all(|chunk_idx| self.matrix_exists(&self.chunk_id(id, chunk_idx)))
    }

    #[cfg(not(feature = "gpu"))]
    fn left_mul_checkpointed_cpu(&self, lhs: &M, id: &str, total_cols: usize) -> M {
        let mut chunk_iter = (0..column_chunk_count(total_cols))
            .map(|chunk_idx| lhs.clone() * &self.read_matrix(&self.chunk_id(id, chunk_idx)));
        let first = chunk_iter.next().expect("chunked artifact should have at least one chunk");
        let rest = chunk_iter.collect::<Vec<_>>();
        if rest.is_empty() { first } else { first.concat_columns_owned(rest) }
    }

    fn build_output_encoding(
        &self,
        vector: M,
        pubkey: BggPublicKey<M>,
        plaintext: Option<M::P>,
    ) -> BggEncoding<M> {
        let plaintext = if pubkey.reveal_plaintext { plaintext } else { None };
        BggEncoding::new(vector, pubkey, plaintext)
    }
}

impl<M, US, HS, TS> InputInjector<BggPublicKey<M>, BggEncoding<M>>
    for DiamondInjector<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    fn preprocess(
        &self,
        one: &BggPublicKey<M>,
        input_digits: &[BggPublicKey<M>],
        decoders: &[BggPublicKey<M>],
    ) {
        #[cfg(feature = "gpu")]
        {
            self.preprocess_gpu(one, input_digits, decoders);
            return;
        }

        #[cfg(not(feature = "gpu"))]
        {
            self.validate_lengths(input_digits, decoders);
            self.ensure_dir();
            self.write_metadata(&DiamondInjectorMetadata {
                input_count: self.input_count,
                base: self.base,
                decoder_count: self.decoder_count,
            });

            let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
            let gadget = M::gadget_matrix(&self.params, DIAMOND_SECRET_SIZE);
            let mut b_checkpoints = Vec::with_capacity(self.input_count + 1);
            let mut trapdoors = Vec::with_capacity(self.input_count + 1);
            // Load or sample the per-level trapdoor checkpoints that all later
            // preimage samplers will reference.
            for level in 0..=self.input_count {
                let (trapdoor, b_matrix) = self.load_or_sample_b_checkpoint(level);
                trapdoors.push(trapdoor);
                b_checkpoints.push(b_matrix);
            }

            // Sample the empty-prefix seed once and persist it if it does not
            // already exist.
            let secret_epsilon = self.load_or_sample_secret_mask(self.secret_epsilon_id());
            if !self.matrix_exists(self.p_epsilon_id()) {
                self.write_matrix(
                    self.p_epsilon_id(),
                    &self.build_initial_encoding(&b_checkpoints[0], &secret_epsilon),
                );
            }

            let state_cols = self.state_col_size(&self.params);
            let output_cols = self.gadget_col_size(&self.params);

            // For each level, each digit value, and each active branch, sample
            // the transition preimage that advances the state machine by one
            // more chosen digit. Each column chunk is written immediately so we
            // never need to keep the full transition matrix in memory.
            for level in 1..=self.input_count {
                for digit_value in 0..self.base {
                    let secret_mask =
                        self.load_or_sample_secret_mask(&self.digit_secret_id(level, digit_value));
                    let prev_ext_w0 = self.sample_w_block_with_params(&self.params, 0, level - 1);
                    for state_idx in 0..=level {
                        let k_id = self.k_id(level, digit_value, state_idx);
                        if self.has_all_chunks(&k_id, state_cols) {
                            continue;
                        }
                        let ext_matrix = if state_idx == level {
                            &prev_ext_w0
                        } else {
                            &self.sample_w_block_with_params(&self.params, state_idx, level - 1)
                        };
                        for chunk_idx in 0..column_chunk_count(state_cols) {
                            let chunk_id = self.chunk_id(&k_id, chunk_idx);
                            if self.matrix_exists(&chunk_id) {
                                continue;
                            }
                            let target_chunk = self.build_k_target_chunk_with_params(
                                &self.params,
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
                            self.write_matrix(&chunk_id, &k_chunk);
                        }
                    }
                }
            }

            // Sample the final projection matrices that turn the completed
            // branches into one-output and digit-output encodings.
            for input_idx in 0..=self.input_count {
                let l_id = self.l_id(input_idx);
                if self.has_all_chunks(&l_id, output_cols) {
                    continue;
                }
                let ext_matrix =
                    self.sample_w_block_with_params(&self.params, input_idx, self.input_count);
                for chunk_idx in 0..column_chunk_count(output_cols) {
                    let chunk_id = self.chunk_id(&l_id, chunk_idx);
                    if self.matrix_exists(&chunk_id) {
                        continue;
                    }
                    let target_chunk = if input_idx == 0 {
                        self.build_one_target_chunk_with_params(
                            &self.params,
                            one,
                            &gadget,
                            chunk_idx,
                        )
                    } else {
                        self.build_input_target_chunk_with_params(
                            &self.params,
                            &input_digits[input_idx - 1],
                            &gadget,
                            chunk_idx,
                        )
                    };
                    let l_chunk = trap_sampler.preimage_extend(
                        &self.params,
                        &trapdoors[self.input_count],
                        &b_checkpoints[self.input_count],
                        &ext_matrix,
                        &target_chunk,
                    );
                    self.write_matrix(&chunk_id, &l_chunk);
                }
            }

            // Sample the final projection matrices that turn the surviving base
            // branch into each decoder output.
            let ext_w0_final = self.sample_w_block_with_params(&self.params, 0, self.input_count);
            for (decoder_idx, decoder) in decoders.iter().enumerate() {
                let m_id = self.m_id(decoder_idx);
                if self.has_all_chunks(&m_id, output_cols) {
                    continue;
                }
                for chunk_idx in 0..column_chunk_count(output_cols) {
                    let chunk_id = self.chunk_id(&m_id, chunk_idx);
                    if self.matrix_exists(&chunk_id) {
                        continue;
                    }
                    let target_chunk = self.build_decoder_target_chunk_with_params(
                        &self.params,
                        decoder,
                        chunk_idx,
                    );
                    let m_chunk = trap_sampler.preimage_extend(
                        &self.params,
                        &trapdoors[self.input_count],
                        &b_checkpoints[self.input_count],
                        &ext_w0_final,
                        &target_chunk,
                    );
                    self.write_matrix(&chunk_id, &m_chunk);
                }
            }
        }
    }

    fn online_eval(
        &self,
        input_digits: &[u32],
        one: &BggPublicKey<M>,
        input_digit_pubkeys: &[BggPublicKey<M>],
        decoders: &[BggPublicKey<M>],
    ) -> (BggEncoding<M>, Vec<BggEncoding<M>>, Vec<BggEncoding<M>>) {
        #[cfg(feature = "gpu")]
        {
            return self.online_eval_gpu(input_digits, one, input_digit_pubkeys, decoders);
        }

        #[cfg(not(feature = "gpu"))]
        {
            self.validate_lengths(input_digit_pubkeys, decoders);
            self.validate_digits(input_digits);
            let metadata = self.read_metadata();
            assert_eq!(
                metadata.input_count, self.input_count,
                "DiamondInjector metadata input count mismatch"
            );
            assert_eq!(metadata.base, self.base, "DiamondInjector metadata base mismatch");
            assert_eq!(
                metadata.decoder_count, self.decoder_count,
                "DiamondInjector metadata decoder count mismatch"
            );

            // Start from the persisted empty-prefix seed.
            let mut states = vec![self.read_matrix(self.p_epsilon_id())];
            let state_cols = self.state_col_size(&self.params);
            for (digit_idx, digit_value) in input_digits.iter().copied().enumerate() {
                let level = digit_idx + 1;
                let prev_states = std::mem::take(&mut states);
                let prev_p0 = prev_states[0].clone();
                let mut next_states = Vec::with_capacity(level + 1);
                // Advance every currently alive branch through the transition
                // matrix for the chosen digit, and spawn the new branch that
                // records the current digit position.
                for state_idx in 0..=level {
                    let lhs = if state_idx == level {
                        prev_p0.clone()
                    } else {
                        prev_states[state_idx].clone()
                    };
                    let rhs_id = self.k_id(level, digit_value as usize, state_idx);
                    next_states.push(self.left_mul_checkpointed_cpu(&lhs, &rhs_id, state_cols));
                }
                states = next_states;
            }

            let output_cols = self.gadget_col_size(&self.params);
            // Turn the surviving base branch into the encoding of one.
            let one_vector = self.left_mul_checkpointed_cpu(&states[0], &self.l_id(0), output_cols);
            let one_output = self.build_output_encoding(
                one_vector,
                one.clone(),
                Some(M::P::const_one(&self.params)),
            );

            // Turn each digit-specific branch into the encoding for the digit
            // that was chosen at that position.
            let digit_outputs = input_digits
                .iter()
                .copied()
                .enumerate()
                .map(|(digit_idx, digit_value)| {
                    let vector = self.left_mul_checkpointed_cpu(
                        &states[digit_idx + 1],
                        &self.l_id(digit_idx + 1),
                        output_cols,
                    );
                    self.build_output_encoding(
                        vector,
                        input_digit_pubkeys[digit_idx].clone(),
                        Some(M::P::from_usize_to_constant(&self.params, digit_value as usize)),
                    )
                })
                .collect::<Vec<_>>();

            // Turn the surviving base branch into every decoder output.
            let decoder_outputs = decoders
                .iter()
                .cloned()
                .enumerate()
                .map(|(decoder_idx, pubkey)| {
                    let vector = self.left_mul_checkpointed_cpu(
                        &states[0],
                        &self.m_id(decoder_idx),
                        output_cols,
                    );
                    self.build_output_encoding(vector, pubkey, Some(M::P::const_zero(&self.params)))
                })
                .collect::<Vec<_>>();

            (one_output, digit_outputs, decoder_outputs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DIAMOND_SECRET_SIZE, DiamondInjector, InputInjector};
    use crate::{
        __PAIR, __TestState,
        bgg::public_key::BggPublicKey,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{
            DistType, PolyHashSampler, hash::DCRTPolyHashSampler,
            trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
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

    fn sample_pubkey(
        params: &DCRTPolyParams,
        hash_key: [u8; 32],
        tag: &str,
    ) -> BggPublicKey<DCRTPolyMatrix> {
        let matrix = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            params,
            hash_key,
            tag,
            DIAMOND_SECRET_SIZE,
            DIAMOND_SECRET_SIZE * params.modulus_digits(),
            DistType::FinRingDist,
        );
        BggPublicKey::new(matrix, true)
    }

    #[sequential_test::sequential]
    #[test]
    fn test_diamond_injector_online_eval_returns_exact_bgg_relations() {
        type TestPoly = <DCRTPolyMatrix as PolyMatrix>::P;

        let params = DCRTPolyParams::default();
        let hash_key = [7u8; 32];
        let input_count = 3;
        let base = 4;
        let decoder_count = 2;
        let dir = tempdir().expect("temporary directory should be created");

        let injector = TestInjector::new(
            params.clone(),
            hash_key,
            input_count,
            base,
            decoder_count,
            4.578,
            0.0,
            dir.path().to_path_buf(),
        );

        let one_pubkey = sample_pubkey(&params, hash_key, "diamond_one_pubkey");
        let input_pubkeys = (0..input_count)
            .map(|digit_idx| {
                sample_pubkey(&params, hash_key, &format!("diamond_input_pubkey_{digit_idx}"))
            })
            .collect::<Vec<_>>();
        let decoder_pubkeys = (0..decoder_count)
            .map(|decoder_idx| {
                sample_pubkey(&params, hash_key, &format!("diamond_decoder_pubkey_{decoder_idx}"))
            })
            .collect::<Vec<_>>();

        injector.preprocess(&one_pubkey, &input_pubkeys, &decoder_pubkeys);

        let digits = vec![1u32, 3u32, 2u32];
        let (one_output, digit_outputs, decoder_outputs) =
            injector.online_eval(&digits, &one_pubkey, &input_pubkeys, &decoder_pubkeys);

        assert_eq!(digit_outputs.len(), input_count);
        assert_eq!(decoder_outputs.len(), decoder_count);

        let mut secret_matrix = injector.read_matrix(injector.secret_epsilon_id());
        for (digit_idx, digit_value) in digits.iter().copied().enumerate() {
            secret_matrix = secret_matrix *
                injector
                    .read_matrix(&injector.digit_secret_id(digit_idx + 1, digit_value as usize));
        }
        let gadget = DCRTPolyMatrix::gadget_matrix(&params, DIAMOND_SECRET_SIZE);
        let secret_times_gadget = secret_matrix.clone() * &gadget;

        assert_eq!(one_output.vector, secret_matrix.clone() * (&one_pubkey.matrix - &gadget));
        assert_eq!(one_output.plaintext, Some(TestPoly::const_one(&params)));

        for (digit_idx, output) in digit_outputs.iter().enumerate() {
            let plaintext = TestPoly::from_usize_to_constant(&params, digits[digit_idx] as usize);
            let expected = secret_matrix.clone() * &input_pubkeys[digit_idx].matrix -
                (secret_times_gadget.clone() * &plaintext);
            assert_eq!(output.vector, expected);
            assert_eq!(output.plaintext, Some(plaintext));
        }

        for (decoder_idx, output) in decoder_outputs.iter().enumerate() {
            assert_eq!(output.vector, secret_matrix.clone() * &decoder_pubkeys[decoder_idx].matrix);
            assert_eq!(output.plaintext, Some(TestPoly::const_zero(&params)));
        }
    }

    #[test]
    fn test_diamond_injector_simulate_output_error_bounds_matches_repeated_preimage_bound() {
        let params = DCRTPolyParams::default();
        let injector = TestInjector::new(
            params.clone(),
            [19u8; 32],
            3,
            4,
            2,
            4.578,
            3.0,
            std::env::temp_dir(),
        );

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
        let expected_initial = PolyMatrixNorm::sample_gauss(
            ctx.clone(),
            DIAMOND_SECRET_SIZE,
            state_cols,
            initial_sigma,
        );
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
        let expected_special_selector = PolyMatrixNorm::new(
            ctx.clone(),
            injector.state_row_size(),
            injector.state_row_size(),
            BigDecimal::from(injector.base.saturating_sub(1).max(1) as u64),
            Some(DIAMOND_SECRET_SIZE),
        );
        let expected_initial_secret_factor = PolyMatrixNorm::new(
            ctx.clone(),
            DIAMOND_SECRET_SIZE,
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
                    .map(|secret| secret.clone() * &expected_regular_selector)
                    .collect::<Vec<_>>();
                let mut next_state_errors = prev_secret
                    .iter()
                    .zip(prev_state_errors.iter())
                    .map(|(secret, state_error)| {
                        state_error.clone() * &expected_transition +
                            secret.clone() * &expected_transition_target_error
                    })
                    .collect::<Vec<_>>();

                let born_secret = prev_secret[0].clone() * &expected_special_selector;
                let born_state_error = prev_state_errors[0].clone() * &expected_transition +
                    prev_secret[0].clone() * &expected_transition_target_error;
                next_secret.push(born_secret);
                next_state_errors.push(born_state_error);

                (next_secret, next_state_errors)
            };

        for _ in 1..=injector.input_count {
            let (next_secret, next_state_errors) =
                advance_expected_state(&expected_secret_factors, &expected_state_errors);
            expected_secret_factors = next_secret;
            expected_state_errors = next_state_errors;
        }

        let expected_inputs = (0..injector.input_count)
            .map(|input_idx| expected_state_errors[input_idx + 1].clone() * &expected_output)
            .collect::<Vec<_>>();
        assert_eq!(simulated.input_errors, expected_inputs);

        let expected_decoders =
            vec![expected_state_errors[0].clone() * &expected_output; injector.decoder_count];
        assert_eq!(simulated.decoder_errors, expected_decoders);
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
        let decoder_count = 1usize;
        let params = DCRTPolyParams::new(ring_dim, crt_depth, crt_bits, base_bits);
        let injector = TestInjector::new(
            params.clone(),
            [29u8; 32],
            input_count,
            input_base,
            decoder_count,
            4.578,
            4.578,
            std::env::temp_dir(),
        );

        let simulated = injector.simulate_output_error_bounds();
        let max_input_error = simulated
            .input_errors
            .iter()
            .map(|norm| norm.poly_norm.norm.clone())
            .max()
            .expect("input error list must be non-empty");
        let max_decoder_error = simulated
            .decoder_errors
            .iter()
            .map(|norm| norm.poly_norm.norm.clone())
            .max()
            .expect("decoder error list must be non-empty");
        let max_error = std::cmp::max(max_input_error.clone(), max_decoder_error.clone());

        println!(
            "diamond injector output-error metrics: ring_dim={ring_dim}, crt_depth={crt_depth}, crt_bits={crt_bits}, base_bits={base_bits}, digit_bits={digit_bits}, input_base={input_base}, input_count={input_count}, decoder_count={decoder_count}, fixed_secret_size=1"
        );
        println!(
            "diamond injector output-error bits: max_input={}, max_decoder={}, max_total={}",
            bigdecimal_bits_ceil(&max_input_error),
            bigdecimal_bits_ceil(&max_decoder_error),
            bigdecimal_bits_ceil(&max_error),
        );
    }
}
