use num_bigint::BigUint;

use crate::{
    bench_estimator::CircuitBenchSummary,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::bgg_pubkey::{column_chunk_bounds, column_chunk_count},
};

use super::{
    DIAMOND_SECRET_SIZE, DiamondIO,
    bench_estimator_native::DiamondIONativeBenchEstimator,
    bench_estimator_utils::{
        estimate_summary, parallel_summaries, scale_summary, sequential_summaries,
    },
};
use crate::io::utils::bench_estimator as bench_utils;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiamondIOStorageEstimate {
    pub(super) input_injection_metadata_and_seed_bytes: BigUint,
    pub(super) input_injection_bytes: BigUint,
    pub(super) final_projection_preimage_bytes: BigUint,
    pub(super) prf_refresh_preimage_bytes: BigUint,
    pub(super) public_lut_aux_bytes: BigUint,
    pub(super) total_bytes: BigUint,
}

#[derive(Debug, Clone)]
pub(super) struct DiamondIOBenchShape {
    pub(super) ring_dim: usize,
    pub(super) input_size: usize,
    pub(super) prf_round_count: usize,
    pub(super) prf_branch_count: usize,
    pub(super) output_size: usize,
    pub(super) seed_bits: usize,
    pub(super) prf_mask_output_coeff_bits: usize,
    pub(super) crt_depth: usize,
    pub(super) modulus_digits: usize,
    pub(super) modulus_bits: u16,
    pub(super) state_row_size: usize,
    pub(super) state_col_size: usize,
    pub(super) b_public_col_size: usize,
    pub(super) checkpoint_count: usize,
    pub(super) transition_matrix_count: usize,
    pub(super) transition_preimage_chunk_count: usize,
    pub(super) transition_ext_w_hash_count: usize,
    pub(super) transition_target_w_hash_chunk_count: usize,
    pub(super) transition_w_hash_max_col_len: usize,
    pub(super) input_injection_transition_preimage_bytes: usize,
    pub(super) online_level_state_counts: Vec<usize>,
    pub(super) transition_chunk_col_lens: Vec<usize>,
    pub(super) output_preimage_bytes: usize,
    pub(super) final_decoder_preimage_bytes: usize,
    pub(super) lookup_bridge_cols: usize,
    pub(super) lookup_bridge_preimage_bytes: usize,
    pub(super) ring_gsw_wire_count: usize,
}

impl DiamondIOBenchShape {
    pub(super) fn from_diamond<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        function_output_bits: usize,
    ) -> Self
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        let params = &diamond.injector.params;
        let ring_dim = params.ring_dimension() as usize;
        let prf_round_count = diamond.injector.input_count;
        let prf_branch_count = 1usize
            .checked_shl(diamond.injector.batch_bits() as u32)
            .expect("DiamondIO PRF digit branch count overflow");
        let (_, _, crt_depth) = params.to_crt();
        let modulus_digits = params.modulus_digits();
        let modulus_bits = params
            .modulus_bits()
            .try_into()
            .expect("DiamondIO modulus bits must fit in u16 for compact-byte estimates");
        let state_row_size =
            2usize.checked_mul(DIAMOND_SECRET_SIZE).expect("DiamondIO state row size overflow");
        let gadget_col_size = DIAMOND_SECRET_SIZE
            .checked_mul(modulus_digits)
            .expect("DiamondIO gadget column count overflow");
        let b_public_col_size = state_row_size
            .checked_mul(modulus_digits + 2)
            .expect("DiamondIO B public column count overflow");
        let state_col_size =
            b_public_col_size.checked_add(gadget_col_size).expect("DiamondIO state cols overflow");
        let chunk_count = column_chunk_count(state_col_size);
        let transition_chunk_col_lens = (0..chunk_count)
            .map(|chunk_idx| column_chunk_bounds(state_col_size, chunk_idx).1)
            .collect::<Vec<_>>();
        let transition_chunk_bytes = transition_chunk_col_lens
            .iter()
            .map(|&col_len| {
                matrix_compact_bytes::<M>(params, state_col_size, col_len, modulus_bits)
            })
            .collect::<Vec<_>>();
        let transition_preimage_bytes_per_matrix = transition_chunk_bytes.iter().sum::<usize>();
        let transition_w_chunk_lens = (0..chunk_count)
            .filter_map(|chunk_idx| {
                let (col_start, col_len) = column_chunk_bounds(state_col_size, chunk_idx);
                let col_end = col_start + col_len;
                let w_start = b_public_col_size;
                (col_end > w_start)
                    .then_some(col_end - col_start.max(w_start))
                    .filter(|&w_len| w_len > 0)
            })
            .collect::<Vec<_>>();
        let transition_w_hash_chunks_per_matrix = transition_w_chunk_lens.len();
        let transition_w_hash_max_col_len =
            transition_w_chunk_lens.iter().copied().max().unwrap_or(1);

        let mut transition_preimage_chunk_count = 0usize;
        let mut transition_matrix_count = 0usize;
        let mut transition_ext_w_hash_count = 0usize;
        let mut online_level_state_counts = Vec::with_capacity(diamond.injector.input_count);
        for level in 1..=diamond.injector.input_count {
            let state_count = expanded_state_count_after_level(diamond, level);
            let old_state_count = state_count.saturating_sub(diamond.injector.batch_bits());
            transition_matrix_count = transition_matrix_count
                .checked_add(
                    diamond
                        .injector
                        .base
                        .checked_mul(state_count)
                        .expect("DiamondIO transition matrix count overflow"),
                )
                .expect("DiamondIO transition matrix count overflow");
            transition_preimage_chunk_count = transition_preimage_chunk_count
                .checked_add(
                    diamond
                        .injector
                        .base
                        .checked_mul(state_count)
                        .and_then(|count| count.checked_mul(chunk_count))
                        .expect("DiamondIO transition chunk count overflow"),
                )
                .expect("DiamondIO transition chunk count overflow");
            // `DiamondInjector::preprocess` materializes one full hashed W block for the previous
            // empty-prefix state before iterating states, then one more full W block for every
            // non-new state. New states reuse that first previous-level block. These matrices are
            // deterministic hash-derived public data, so storage does not count them, but their
            // sampling work is part of preprocessing latency and total work.
            transition_ext_w_hash_count = transition_ext_w_hash_count
                .checked_add(
                    diamond
                        .injector
                        .base
                        .checked_mul(
                            old_state_count
                                .checked_add(1)
                                .expect("DiamondIO transition W hash count overflow"),
                        )
                        .expect("DiamondIO transition W hash count overflow"),
                )
                .expect("DiamondIO transition W hash count overflow");
            online_level_state_counts.push(state_count);
        }
        let transition_target_w_hash_chunk_count = transition_matrix_count
            .checked_mul(transition_w_hash_chunks_per_matrix)
            .expect("DiamondIO transition target W hash chunk count overflow");

        let input_injection_transition_preimage_bytes = transition_preimage_bytes_per_matrix
            .checked_mul(transition_matrix_count)
            .expect("DiamondIO transition preimage byte count overflow");
        let output_preimage_bytes =
            matrix_compact_bytes::<M>(params, state_col_size, modulus_digits, modulus_bits);
        let final_decoder_preimage_bytes =
            matrix_compact_bytes::<M>(params, state_col_size, 1, modulus_bits);
        let lookup_cols = diamond
            .enc_lookup_base_matrix
            .as_ref()
            .map(PolyMatrix::col_size)
            .unwrap_or(modulus_digits);
        let lookup_bridge_preimage_bytes =
            matrix_compact_bytes::<M>(params, state_col_size, lookup_cols, modulus_bits);
        let ring_gsw_active_levels = diamond.ring_gsw_enable_levels.unwrap_or_else(|| {
            diamond
                .ring_gsw_context
                .q_moduli_depth
                .checked_sub(diamond.ring_gsw_level_offset)
                .expect("DiamondIO Ring-GSW level offset exceeds q-moduli depth")
        });
        assert!(
            diamond.ring_gsw_level_offset + ring_gsw_active_levels <=
                diamond.ring_gsw_context.q_moduli_depth,
            "DiamondIO Ring-GSW active-level range exceeds q-moduli depth"
        );
        // Shape calculation only needs the flattened wire count of one Ring-GSW ciphertext.
        // Avoid constructing a representative Goldreich PRG circuit here; the actual PRG circuit
        // is still built and measured by `estimate_prf_path`.
        let ring_gsw_wire_count = 2usize
            .checked_mul(diamond.ring_gsw_width)
            .and_then(|count| count.checked_mul(ring_gsw_active_levels))
            .and_then(|count| count.checked_mul(diamond.ring_gsw_context.p_moduli.len()))
            .expect("DiamondIO Ring-GSW wire count overflow");

        Self {
            ring_dim,
            input_size: diamond.input_size,
            prf_round_count,
            prf_branch_count,
            output_size: function_output_bits,
            seed_bits: diamond.seed_bits,
            prf_mask_output_coeff_bits: diamond.prf_mask_output_coeff_bits,
            crt_depth,
            modulus_digits,
            modulus_bits,
            state_row_size,
            state_col_size,
            b_public_col_size,
            checkpoint_count: diamond.injector.input_count + 1,
            transition_matrix_count,
            transition_preimage_chunk_count,
            transition_ext_w_hash_count,
            transition_target_w_hash_chunk_count,
            transition_w_hash_max_col_len,
            input_injection_transition_preimage_bytes,
            online_level_state_counts,
            transition_chunk_col_lens,
            output_preimage_bytes,
            final_decoder_preimage_bytes,
            lookup_bridge_cols: lookup_cols,
            lookup_bridge_preimage_bytes,
            ring_gsw_wire_count,
        }
    }

    pub(super) fn input_injection_metadata_and_seed_bytes(&self) -> BigUint {
        // This term covers the small non-matrix files plus the persisted empty-prefix state used
        // to start `DiamondInjector::online_eval`.
        //
        // * `metadata_estimate` is a fixed conservative allowance for the JSON metadata written by
        //   `DiamondInjector::write_metadata`; it stores only `input_count` and `base`, so the
        //   actual encoded file is tiny and independent of the polynomial parameters.
        // * `hash_key_bytes` is the 32-byte `diamond_preprocess_hash_key`. It is needed at eval
        //   time to check that the selected preprocessed transition chunks belong to the same
        //   deterministic hash-derived public side.
        // * `p_epsilon_bytes` estimates `diamond_p_epsilon_tensor_0`, the initial empty-prefix
        //   state read before the first input digit is applied. Its shape is `(state_row_size / 2)
        //   x state_col_size`, matching the secret-side half-state produced by
        //   `build_initial_encoding`.
        let metadata_estimate = 128usize;
        let hash_key_bytes = 32usize;
        let p_epsilon_bytes = bench_utils::matrix_compact_bytes_for_shape(
            self.state_row_size / 2,
            self.state_col_size,
            self.ring_dim,
            self.modulus_bits,
        );
        BigUint::from(metadata_estimate + hash_key_bytes + p_epsilon_bytes)
    }

    pub(super) fn input_injection_public_checkpoint_bytes(&self) -> usize {
        // This counts the public `B` checkpoint matrices persisted by
        // `DiamondInjector::load_or_sample_b_checkpoint` for levels `0..=input_count`. The matching
        // trapdoors are intentionally not counted because they are preprocessing secrets rather
        // than obfuscated-circuit data. Online eval does not multiply by these checkpoints
        // directly; they are counted here only because the current preprocessing implementation
        // persists the public matrices as reusable checkpoint artifacts.
        bench_utils::matrix_compact_bytes_for_shape(
            self.state_row_size,
            self.b_public_col_size,
            self.ring_dim,
            self.modulus_bits,
        )
        .checked_mul(self.checkpoint_count)
        .expect("DiamondIO B checkpoint byte count overflow")
    }

    pub(super) fn input_injection_bytes(&self) -> BigUint {
        // Input-injection persisted bytes are grouped by the artifacts that survive preprocessing:
        //
        // 1. Small metadata, the 32-byte hash key, and the empty-prefix seed state `p_epsilon`.
        // 2. Public `B` checkpoint matrices for the per-level trapdoor chain, excluding trapdoors.
        // 3. Chunked transition preimages `diamond_k_bit_tensor_*`, which are the large artifacts
        //    read by online eval to advance the selected input-dependent branch states.
        self.input_injection_metadata_and_seed_bytes() +
            BigUint::from(self.input_injection_public_checkpoint_bytes()) +
            BigUint::from(self.input_injection_transition_preimage_bytes)
    }

    pub(super) fn estimate_transition_target_building<NBE>(
        &self,
        native_estimator: &NBE,
    ) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator,
    {
        let per_transition_matrix = self
            .transition_chunk_col_lens
            .iter()
            .map(|&col_len| {
                sequential_summaries(&[
                    native_estimator.estimate_row_parallel_matrix_product(
                        self.state_row_size,
                        self.state_row_size,
                        col_len,
                    ),
                    estimate_summary(
                        native_estimator
                            .estimate_vector_add(self.state_row_size.saturating_mul(col_len)),
                    ),
                ])
            })
            .collect::<Vec<_>>();
        let per_transition_matrix = parallel_summaries(per_transition_matrix.as_slice());
        scale_summary(per_transition_matrix, self.transition_matrix_count)
    }

    pub(super) fn estimate_online_input_injection<NBE>(
        &self,
        native_estimator: &NBE,
    ) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator,
    {
        let per_state = self
            .transition_chunk_col_lens
            .iter()
            .map(|&col_len| {
                native_estimator.estimate_vector_matrix_product(self.state_col_size, col_len)
            })
            .collect::<Vec<_>>();
        let per_state = parallel_summaries(per_state.as_slice());
        let levels = self
            .online_level_state_counts
            .iter()
            .map(|&state_count| scale_summary(per_state.clone(), state_count))
            .collect::<Vec<_>>();
        sequential_summaries(levels.as_slice())
    }

    pub(super) fn estimate_input_encoding_projection<NBE>(
        &self,
        native_estimator: &NBE,
        input_size: usize,
    ) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator,
    {
        let ordinary_projection_count =
            input_size.checked_add(2).expect("DiamondIO ordinary input projection count overflow");
        parallel_summaries(&[
            scale_summary(
                native_estimator
                    .estimate_vector_matrix_product(self.state_col_size, self.modulus_digits),
                ordinary_projection_count,
            ),
            native_estimator
                .estimate_vector_matrix_product(self.state_col_size, self.lookup_bridge_cols),
        ])
    }

    pub(super) fn estimate_final_projection_target_building<NBE>(
        &self,
        native_estimator: &NBE,
    ) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator,
    {
        let ordinary_count = 2usize
            .checked_add(self.input_size)
            .expect("DiamondIO ordinary projection count overflow");
        let ordinary_targets = scale_summary(
            sequential_summaries(&[
                // `sample_final_output_preimage` subtracts a one-row gadget term from the top or
                // bottom half for the one/k/input projection targets. This is native polynomial
                // matrix arithmetic performed before the trapdoor preimage sampler sees the
                // target.
                native_estimator.estimate_row_parallel_matrix_product(
                    DIAMOND_SECRET_SIZE,
                    DIAMOND_SECRET_SIZE,
                    self.modulus_digits,
                ),
                estimate_summary(
                    native_estimator.estimate_vector_sub(
                        DIAMOND_SECRET_SIZE
                            .checked_mul(self.modulus_digits)
                            .expect("DiamondIO ordinary final target size overflow"),
                    ),
                ),
            ]),
            ordinary_count,
        );
        let decoder_targets = scale_summary(
            sequential_summaries(&[
                // Decoder targets first add function and PRF-mask public keys, then project the
                // secret-dependent evaluated vector through the identity selector by
                // `mul_decompose`. The zero bottom row and concatenation are bookkeeping rather
                // than arithmetic-heavy GPU work.
                estimate_summary(
                    native_estimator.estimate_vector_add(
                        DIAMOND_SECRET_SIZE
                            .checked_mul(self.modulus_digits)
                            .expect("DiamondIO decoder public-key matrix size overflow"),
                    ),
                ),
                native_estimator
                    .estimate_vector_matrix_product(self.modulus_digits, DIAMOND_SECRET_SIZE),
            ]),
            self.final_decoder_count(),
        );
        parallel_summaries(&[ordinary_targets, decoder_targets])
    }

    pub(super) fn final_projection_preimage_count(&self) -> usize {
        // one, k, every explicit input bit, the lookup bridge, and every final decoder slot.
        3usize
            .checked_add(self.input_size)
            .expect("DiamondIO final projection preimage count overflow")
            .checked_add(self.final_decoder_count())
            .expect("DiamondIO final projection preimage count overflow")
    }

    pub(super) fn final_projection_standard_preimage_count(&self) -> usize {
        // Ordinary final projection preimages have `modulus_digits` columns: one, k, and each
        // explicit input bit. The lookup bridge has lookup-specific width, and final decoder
        // preimages have a one-column target induced by the identity selector.
        self.final_projection_preimage_count()
            .checked_sub(1)
            .expect("DiamondIO final projection count must include the lookup bridge")
            .checked_sub(self.final_decoder_count())
            .expect("DiamondIO final projection count must include final decoder preimages")
    }

    pub(super) fn final_projection_preimage_bytes(&self) -> usize {
        self.lookup_bridge_preimage_bytes
            .checked_add(
                self.output_preimage_bytes
                    .checked_mul(self.final_projection_standard_preimage_count())
                    .expect("DiamondIO output preimage byte count overflow"),
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    self.final_decoder_preimage_bytes
                        .checked_mul(self.final_decoder_count())
                        .expect("DiamondIO final decoder preimage byte count overflow"),
                )
            })
            .expect("DiamondIO final projection byte count overflow")
    }

    pub(super) fn final_decoder_count(&self) -> usize {
        self.output_size.checked_mul(self.ring_dim).expect("DiamondIO final decoder count overflow")
    }

    pub(super) fn final_mask_prg_output_count(&self) -> usize {
        self.final_decoder_count()
            .checked_mul(self.prf_mask_output_coeff_bits)
            .expect("DiamondIO final mask PRG output count overflow")
    }

    pub(super) fn goldreich_prf_output_count(&self) -> usize {
        self.output_size
    }

    pub(super) fn final_prg_output_count(&self) -> usize {
        self.final_mask_prg_output_count()
            .checked_add(self.goldreich_prf_output_count())
            .expect("DiamondIO final mask/function PRG output count overflow")
    }

    pub(super) fn selected_prg_output_count(&self) -> usize {
        self.prf_round_count
            .checked_mul(self.prf_branch_count)
            .and_then(|count| count.checked_mul(self.seed_bits))
            .expect("DiamondIO selected-branch PRG output count overflow")
    }

    pub(super) fn sparse_noise_refresh_material_uniform_prg_output_count(
        &self,
        input_size: usize,
        v_bits: usize,
        cbd_n: usize,
    ) -> BigUint {
        let error_ciphertexts = self
            .ring_dim
            .checked_mul(self.modulus_digits)
            .and_then(|count| count.checked_mul(self.ring_dim))
            .expect("DiamondIO noise-refresh CBD ciphertext count overflow");
        let mask_ciphertexts = error_ciphertexts
            .checked_mul(self.crt_depth)
            .and_then(|count| count.checked_mul(v_bits))
            .expect("DiamondIO noise-refresh mask ciphertext count overflow");
        let uniform_outputs = error_ciphertexts
            .checked_mul(2)
            .and_then(|count| count.checked_mul(cbd_n))
            .and_then(|count| count.checked_add(mask_ciphertexts))
            .expect("DiamondIO noise-refresh uniform PRG output count overflow");
        BigUint::from(uniform_outputs) * BigUint::from(input_size)
    }

    pub(super) fn prf_refresh_decoder_preimage_count(&self) -> BigUint {
        BigUint::from(self.prf_round_count) *
            BigUint::from(self.prf_branch_count) *
            BigUint::from(self.seed_bits) *
            BigUint::from(self.ring_gsw_wire_count) *
            BigUint::from(self.ring_dim) *
            BigUint::from(self.crt_depth)
    }

    pub(super) fn prf_branch_rebase_preimage_count(&self) -> BigUint {
        BigUint::from(self.prf_round_count) *
            BigUint::from(self.prf_branch_count) *
            BigUint::from(self.seed_bits) *
            BigUint::from(self.ring_gsw_wire_count) *
            BigUint::from(self.ring_dim)
    }

    pub(super) fn prf_refresh_preimage_bytes(&self) -> BigUint {
        BigUint::from(self.output_preimage_bytes) *
            (self.prf_refresh_decoder_preimage_count() + self.prf_branch_rebase_preimage_count())
    }
}

fn expanded_state_count_after_level<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
    diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
    level: usize,
) -> usize
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    1usize
        .checked_add(
            level
                .checked_mul(diamond.injector.batch_bits())
                .expect("DiamondIO expanded state count overflow"),
        )
        .expect("DiamondIO expanded state count overflow")
}

fn matrix_compact_bytes<M>(
    params: &<M::P as Poly>::Params,
    nrow: usize,
    ncol: usize,
    modulus_bits: u16,
) -> usize
where
    M: PolyMatrix,
{
    // This is only a shape/size estimate. Calling `M::zero_compact_bytes(...).len()` would
    // materialize the full compact-byte payload just to ask for its length, which can exceed the
    // H200 CPU cgroup for large DiamondIO shapes before any benchmark unit is measured.
    bench_utils::matrix_compact_bytes_for_shape(
        nrow,
        ncol,
        params.ring_dimension() as usize,
        modulus_bits,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prf_refresh_preimage_bytes_scale_with_digit_branch_count() {
        let shape = DiamondIOBenchShape {
            ring_dim: 7,
            input_size: 6,
            prf_round_count: 3,
            prf_branch_count: 4,
            output_size: 2,
            seed_bits: 5,
            prf_mask_output_coeff_bits: 2,
            crt_depth: 4,
            modulus_digits: 6,
            modulus_bits: 60,
            state_row_size: 2,
            state_col_size: 3,
            b_public_col_size: 4,
            checkpoint_count: 0,
            transition_matrix_count: 0,
            transition_preimage_chunk_count: 0,
            transition_ext_w_hash_count: 0,
            transition_target_w_hash_chunk_count: 0,
            transition_w_hash_max_col_len: 0,
            input_injection_transition_preimage_bytes: 0,
            online_level_state_counts: Vec::new(),
            transition_chunk_col_lens: Vec::new(),
            output_preimage_bytes: 11,
            final_decoder_preimage_bytes: 13,
            lookup_bridge_cols: 0,
            lookup_bridge_preimage_bytes: 0,
            ring_gsw_wire_count: 9,
        };
        let noise_refresh_count = BigUint::from(3usize * 4 * 5 * 9 * 7 * 4);
        let branch_rebase_count = BigUint::from(3usize * 4 * 5 * 9 * 7);
        assert_eq!(shape.prf_refresh_decoder_preimage_count(), noise_refresh_count);
        assert_eq!(shape.prf_branch_rebase_preimage_count(), branch_rebase_count);
        assert_eq!(
            shape.prf_refresh_preimage_bytes(),
            BigUint::from(11usize) * (noise_refresh_count + branch_rebase_count)
        );
    }
}
