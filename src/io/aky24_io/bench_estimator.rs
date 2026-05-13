use std::sync::Arc;

use num_bigint::BigUint;
use num_traits::Zero;

use crate::{
    bench_estimator::{
        BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary, PublicKeyAuxBenchEstimator,
        estimate_public_key_circuit_bench_with_aux,
    },
    bgg::naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
    circuit::PolyCircuit,
    decoder::{
        bench::{
            bit_decomposed_mask_reduce_add_count, bit_decomposed_polynomial_mask_reduction_summary,
            bit_decomposed_refresh_material_counts, bit_decomposed_refresh_material_summary,
            goldreich_cbd_error_prg_summary,
            scale_bit_decomposed_polynomial_mask_decrypt_contributions,
        },
        mask_circuit::append_one_ciphertext_bit_decrypt,
    },
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner, NestedRnsPoly},
        fhe::{ring_gsw::RingGswCiphertext, ring_gsw_nested_rns::NestedRnsRingGswContext},
        fhe_prg::goldreich::{GoldreichEdge, GoldreichFhePrg, GoldreichGraph},
    },
    matrix::PolyMatrix,
    poly::PolyParams,
};

use super::{Aky24IO, Aky24IOFuncType};
use crate::io::diamond_io::DiamondIONativeBenchEstimator;

/// Combined AKY24 iO benchmark estimate for obfuscation and eval.
#[derive(Debug, Clone, PartialEq)]
pub struct Aky24IOBenchEstimate {
    /// Estimated cost of conventional FE-to-iO obfuscation.
    pub obfuscate: CircuitBenchSummary,
    /// Estimated cost of conventional FE-to-iO online evaluation.
    pub eval: CircuitBenchSummary,
    /// Compact bytes for persisted obfuscated-circuit material modeled by this estimator.
    pub obfuscated_circuit_bytes: BigUint,
}

/// Maintained IO-level AKY24 benchmark estimator.
///
/// This estimator intentionally follows the DiamondIO benchmark decomposition but omits all
/// Diamond input-injection preprocessing and online-evaluation costs. It composes representative
/// BGG circuit work, seed-ciphertext lifting, PRF/noise-refresh work, final projection, and native
/// final decode costs without depending on the unmaintained `func_enc::aky24` module.
#[derive(Debug, Clone)]
pub struct Aky24IOBenchEstimator<'a, PKBE, EncBE, NBE> {
    pub public_key_estimator: &'a PKBE,
    pub encoding_estimator: &'a EncBE,
    pub native_estimator: &'a NBE,
    pub bgg_public_key_sample: CircuitBenchEstimate,
    pub ring_gsw_public_key_sample: CircuitBenchEstimate,
    pub ring_gsw_encrypt_bit: CircuitBenchEstimate,
    pub full_w_block_hash_sample: CircuitBenchEstimate,
    pub a_prime_hash_sample: CircuitBenchSummary,
    pub final_output_preimage_extend: CircuitBenchEstimate,
    pub final_decoder_preimage_extend: CircuitBenchEstimate,
}

impl<'a, PKBE, EncBE, NBE> Aky24IOBenchEstimator<'a, PKBE, EncBE, NBE> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        public_key_estimator: &'a PKBE,
        encoding_estimator: &'a EncBE,
        native_estimator: &'a NBE,
        bgg_public_key_sample: CircuitBenchEstimate,
        ring_gsw_public_key_sample: CircuitBenchEstimate,
        ring_gsw_encrypt_bit: CircuitBenchEstimate,
        full_w_block_hash_sample: CircuitBenchEstimate,
        a_prime_hash_sample: CircuitBenchSummary,
        final_output_preimage_extend: CircuitBenchEstimate,
        final_decoder_preimage_extend: CircuitBenchEstimate,
    ) -> Self {
        Self {
            public_key_estimator,
            encoding_estimator,
            native_estimator,
            bgg_public_key_sample,
            ring_gsw_public_key_sample,
            ring_gsw_encrypt_bit,
            full_w_block_hash_sample,
            a_prime_hash_sample,
            final_output_preimage_extend,
            final_decoder_preimage_extend,
        }
    }

    pub fn estimate<M, PKPE, PKST, ENCPE, ENCST>(
        &self,
        scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
        func: Aky24IOFuncType,
    ) -> Aky24IOBenchEstimate
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NBE: DiamondIONativeBenchEstimator,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        let shape = Aky24IOBenchShape::from_scheme(scheme, func);
        let obfuscate = self.estimate_obfuscate(scheme, &shape);
        let eval = self.estimate_eval(scheme, &shape);
        let public_lut_aux_bytes = self.estimate_public_lut_aux_storage_bytes(scheme, &shape);
        Aky24IOBenchEstimate {
            obfuscate,
            eval,
            obfuscated_circuit_bytes: shape.obfuscated_circuit_bytes(public_lut_aux_bytes),
        }
    }

    fn estimate_obfuscate<M, PKPE, PKST, ENCPE, ENCST>(
        &self,
        scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
        shape: &Aky24IOBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NBE: DiamondIONativeBenchEstimator,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        let scalar_one = vec![BigUint::from(1u32); shape.ring_dim];
        let bgg_public_keys = scale_estimate(
            self.bgg_public_key_sample.clone(),
            shape.input_size.checked_add(2).expect("AKY24IO public-key count overflow"),
        );
        let ring_gsw_public_key_sampling =
            estimate_summary(self.ring_gsw_public_key_sample.clone());
        let seed_encrypt = scale_estimate(self.ring_gsw_encrypt_bit.clone(), shape.seed_bits);
        let seed_lift = scale_estimate(
            self.public_key_estimator.estimate_large_scalar_mul(scalar_one.as_slice()),
            shape
                .seed_bits
                .checked_mul(shape.ring_gsw_wire_count)
                .expect("AKY24IO seed lift count overflow"),
        );
        let seed_public_side =
            sequential_summaries(&[ring_gsw_public_key_sampling, seed_encrypt, seed_lift]);
        let prf_path = self.estimate_prf_path::<M, PKPE, PKST, ENCPE, ENCST>(
            scheme,
            shape,
            PrfBenchMode::PublicKeyPreprocess,
        );
        let final_projection = self.estimate_final_projection_preprocess(shape);
        sequential_summaries(&[
            bgg_public_keys,
            seed_public_side,
            prf_path.compute_without_refresh_decoder,
            parallel_summaries(&[prf_path.refresh_decoder_work, final_projection]),
        ])
    }

    fn estimate_eval<M, PKPE, PKST, ENCPE, ENCST>(
        &self,
        _scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
        shape: &Aky24IOBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NBE: DiamondIONativeBenchEstimator,
    {
        let scalar_one = vec![BigUint::from(1u32); shape.ring_dim];
        let input_projection = scale_summary(
            self.native_estimator
                .estimate_vector_matrix_product(shape.state_col_size, shape.modulus_digits),
            shape.input_size.checked_add(2).expect("AKY24IO input projection count overflow"),
        );
        let seed_lift = scale_estimate(
            self.encoding_estimator.estimate_large_scalar_mul(scalar_one.as_slice()),
            shape
                .seed_bits
                .checked_mul(shape.ring_gsw_wire_count)
                .expect("AKY24IO eval seed lift count overflow"),
        );
        let prf_path = self.estimate_prf_path::<M, PKPE, PKST, ENCPE, ENCST>(
            _scheme,
            shape,
            PrfBenchMode::EncodingOnline,
        );
        let decoder_projection = scale_summary(
            self.native_estimator.estimate_vector_matrix_product(shape.state_col_size, 1),
            shape.final_decoder_count(),
        );
        let final_decode_unit = sequential_summaries(&[
            estimate_summary(
                self.native_estimator.estimate_vector_inner_product(shape.modulus_digits),
            ),
            estimate_summary(self.native_estimator.estimate_vector_add(1)),
            estimate_summary(self.native_estimator.estimate_vector_sub(1)),
        ]);
        let final_decode = scale_summary(final_decode_unit, shape.final_decoder_count());
        sequential_summaries(&[
            input_projection,
            seed_lift,
            prf_path.total,
            decoder_projection,
            final_decode,
        ])
    }

    fn estimate_prf_path<M, PKPE, PKST, ENCPE, ENCST>(
        &self,
        scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
        shape: &Aky24IOBenchShape,
        mode: PrfBenchMode,
    ) -> Aky24IOPrfBenchEstimateParts
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NBE: DiamondIONativeBenchEstimator,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        let final_mask_decrypt_unit = self
            .estimate_prf_mask_decrypt_one_ciphertext_bit_unit::<M, PKPE, PKST, ENCPE, ENCST>(
                scheme, mode,
            );
        let prg_circuit = build_representative_goldreich_prg_one_output_circuit(scheme);
        let prg_unit = match mode {
            PrfBenchMode::PublicKeyPreprocess => {
                estimate_public_key_circuit_bench_with_aux::<NaiveBGGPublicKeyVec<M>, PKBE>(
                    self.public_key_estimator,
                    &scheme.params,
                    &prg_circuit,
                )
            }
            PrfBenchMode::EncodingOnline => {
                self.encoding_estimator.estimate_circuit_bench(&prg_circuit)
            }
        };
        let (add, sub, mul) = match mode {
            PrfBenchMode::PublicKeyPreprocess => (
                self.public_key_estimator.estimate_add(),
                self.public_key_estimator.estimate_sub(),
                self.public_key_estimator.estimate_mul(),
            ),
            PrfBenchMode::EncodingOnline => (
                self.encoding_estimator.estimate_add(),
                self.encoding_estimator.estimate_sub(),
                self.encoding_estimator.estimate_mul(),
            ),
        };
        let noise_refresh_mask_prg_unit = prg_unit.clone();
        let noise_refresh_error_prg_unit = goldreich_cbd_error_prg_summary(
            prg_unit.clone(),
            add.clone(),
            sub.clone(),
            shape.cbd_n,
        );
        let selected_branch_prg_output_bits = match mode {
            PrfBenchMode::PublicKeyPreprocess => shape
                .prf_branch_count
                .checked_mul(shape.seed_bits)
                .expect("AKY24IO public PRF branch PRG output count overflow"),
            PrfBenchMode::EncodingOnline => shape.seed_bits,
        };
        let selected_branch_prg_per_round =
            scale_summary(prg_unit.clone(), selected_branch_prg_output_bits);
        let selected_branch_prg =
            repeat_sequential_summary(selected_branch_prg_per_round.clone(), shape.prf_round_count);
        let selected_wire_count_per_round = shape
            .seed_bits
            .checked_mul(shape.ring_gsw_wire_count)
            .expect("AKY24IO selected PRG wire count overflow");
        let branch_rebase_branch_count = match mode {
            PrfBenchMode::PublicKeyPreprocess => shape.prf_branch_count,
            PrfBenchMode::EncodingOnline => 1usize,
        };
        let selected_branch_rebase_unit = sequential_summaries(&[
            estimate_summary(sub.clone()),
            estimate_summary(mul.clone()),
            estimate_summary(add.clone()),
        ]);
        let selected_branch_rebase_per_round = scale_summary(
            selected_branch_rebase_unit,
            selected_wire_count_per_round
                .checked_mul(branch_rebase_branch_count)
                .expect("AKY24IO selected-branch rebase work count overflow"),
        );
        let selected_rebase = repeat_sequential_summary(
            selected_branch_rebase_per_round.clone(),
            shape.prf_round_count,
        );
        let refresh_parts = self.estimate_noise_refresh_sparse::<M, PKPE, PKST, ENCPE, ENCST>(
            scheme,
            shape,
            mode,
            noise_refresh_error_prg_unit,
            noise_refresh_mask_prg_unit,
            final_mask_decrypt_unit.clone(),
        );
        let noise_refresh_branch_count = match mode {
            PrfBenchMode::PublicKeyPreprocess => shape.prf_branch_count,
            PrfBenchMode::EncodingOnline => 1usize,
        };
        let noise_refresh_decoder_count_per_round = BigUint::from(selected_wire_count_per_round) *
            BigUint::from(noise_refresh_branch_count) *
            BigUint::from(shape.ring_dim) *
            BigUint::from(shape.crt_depth);
        let branch_rebase_decoder_count_per_round = BigUint::from(selected_wire_count_per_round) *
            BigUint::from(shape.ring_dim) *
            BigUint::from(branch_rebase_branch_count);
        let refresh_decoder_work_per_round = match mode {
            PrfBenchMode::PublicKeyPreprocess => sequential_summaries(&[
                scale_estimate_biguint(
                    self.full_w_block_hash_sample.clone(),
                    &(&noise_refresh_decoder_count_per_round +
                        &branch_rebase_decoder_count_per_round),
                ),
                scale_estimate_biguint(
                    self.final_output_preimage_extend.clone(),
                    &(&noise_refresh_decoder_count_per_round +
                        &branch_rebase_decoder_count_per_round),
                ),
            ]),
            PrfBenchMode::EncodingOnline => {
                let decoder_unit = self
                    .native_estimator
                    .estimate_vector_matrix_product(shape.state_col_size, shape.modulus_digits);
                sequential_summaries(&[
                    scale_summary_biguint(
                        decoder_unit.clone(),
                        &branch_rebase_decoder_count_per_round,
                    ),
                    scale_summary_biguint(decoder_unit, &noise_refresh_decoder_count_per_round),
                ])
            }
        };
        let noise_refresh_per_round = sequential_summaries(&[
            scale_summary(refresh_parts.material.clone(), noise_refresh_branch_count),
            scale_summary(
                refresh_parts.per_refresh.clone(),
                selected_wire_count_per_round
                    .checked_mul(noise_refresh_branch_count)
                    .expect("AKY24IO noise-refresh branch work count overflow"),
            ),
        ]);
        let noise_refresh =
            repeat_sequential_summary(noise_refresh_per_round.clone(), shape.prf_round_count);
        let final_prg = scale_summary(prg_unit.clone(), shape.final_prg_output_count());
        let (final_mask_decrypt_contributions, _final_mask_decrypt_contribution_count) =
            scale_bit_decomposed_polynomial_mask_decrypt_contributions(
                final_mask_decrypt_unit.clone(),
                shape.ring_dim,
                shape.prf_mask_output_coeff_bits,
                shape.output_size,
            );
        let _final_mask_reduce_add_count =
            bit_decomposed_mask_reduce_add_count(shape.prf_mask_output_coeff_bits);
        let final_mask_reduction = bit_decomposed_polynomial_mask_reduction_summary(
            add.clone(),
            None,
            Some(sub.clone()),
            shape.prf_mask_output_coeff_bits,
            shape.output_size,
        );
        let final_mask_decrypt =
            sequential_summaries(&[final_mask_decrypt_contributions, final_mask_reduction]);
        let final_function_decrypt =
            scale_summary(final_mask_decrypt_unit.clone(), shape.output_size);
        let final_output_decrypt =
            parallel_summaries(&[final_mask_decrypt.clone(), final_function_decrypt.clone()]);
        let round_compute_per_round = match mode {
            PrfBenchMode::PublicKeyPreprocess => sequential_summaries(&[
                selected_branch_prg_per_round.clone(),
                selected_branch_rebase_per_round.clone(),
                noise_refresh_per_round.clone(),
            ]),
            PrfBenchMode::EncodingOnline => sequential_summaries(&[
                selected_branch_prg_per_round.clone(),
                selected_branch_rebase_per_round.clone(),
                refresh_decoder_work_per_round.clone(),
                noise_refresh_per_round.clone(),
            ]),
        };
        let round_summary_per_round = sequential_summaries(&[
            selected_branch_prg_per_round,
            selected_branch_rebase_per_round,
            refresh_decoder_work_per_round.clone(),
            noise_refresh_per_round,
        ]);
        let round_compute =
            repeat_sequential_summary(round_compute_per_round, shape.prf_round_count);
        let round_summary =
            repeat_sequential_summary(round_summary_per_round, shape.prf_round_count);
        let refresh_decoder_work = match mode {
            PrfBenchMode::PublicKeyPreprocess => {
                scale_summary(refresh_decoder_work_per_round, shape.prf_round_count)
            }
            PrfBenchMode::EncodingOnline => {
                repeat_sequential_summary(refresh_decoder_work_per_round, shape.prf_round_count)
            }
        };
        let final_summary = sequential_summaries(&[final_prg.clone(), final_output_decrypt]);
        let compute_without_refresh_decoder =
            sequential_summaries(&[round_compute, final_summary.clone()]);
        let total = match mode {
            PrfBenchMode::PublicKeyPreprocess => parallel_summaries(&[
                compute_without_refresh_decoder.clone(),
                refresh_decoder_work.clone(),
            ]),
            PrfBenchMode::EncodingOnline => sequential_summaries(&[round_summary, final_summary]),
        };
        Aky24IOPrfBenchEstimateParts {
            selected_branch_prg,
            selected_branch_rebase: selected_rebase,
            refresh_decoder_work,
            noise_refresh,
            final_prg,
            final_mask_decrypt,
            final_function_decrypt,
            compute_without_refresh_decoder,
            total,
        }
    }

    fn estimate_prf_mask_decrypt_one_ciphertext_bit_unit<M, PKPE, PKST, ENCPE, ENCST>(
        &self,
        scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
        mode: PrfBenchMode,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        let circuit = prf_mask_decrypt_one_ciphertext_bit_circuit(scheme);
        match mode {
            PrfBenchMode::PublicKeyPreprocess => {
                estimate_public_key_circuit_bench_with_aux::<NaiveBGGPublicKeyVec<M>, PKBE>(
                    self.public_key_estimator,
                    &scheme.params,
                    &circuit,
                )
            }
            PrfBenchMode::EncodingOnline => {
                self.encoding_estimator.estimate_circuit_bench(&circuit)
            }
        }
    }

    fn estimate_noise_refresh_sparse<M, PKPE, PKST, ENCPE, ENCST>(
        &self,
        _scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
        shape: &Aky24IOBenchShape,
        mode: PrfBenchMode,
        error_prg_unit: CircuitBenchSummary,
        mask_prg_unit: CircuitBenchSummary,
        decrypt_contribution_unit: CircuitBenchSummary,
    ) -> NoiseRefreshBenchEstimateParts
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NBE: DiamondIONativeBenchEstimator,
    {
        let material_counts = bit_decomposed_refresh_material_counts(
            shape.ring_dim,
            shape.modulus_digits,
            shape.crt_depth,
            shape.noise_refresh_v_bits,
            false,
        );
        let add = match mode {
            PrfBenchMode::PublicKeyPreprocess => self.public_key_estimator.estimate_add(),
            PrfBenchMode::EncodingOnline => self.encoding_estimator.estimate_add(),
        };
        let material = bit_decomposed_refresh_material_summary(
            error_prg_unit,
            mask_prg_unit,
            decrypt_contribution_unit,
            add.clone(),
            material_counts,
            shape.noise_refresh_v_bits,
        );
        let scalar_target = [BigUint::from(1u32)];
        let combine_task_count = shape
            .ring_dim
            .checked_mul(shape.crt_depth)
            .expect("AKY24IO noise-refresh combine task count overflow");
        let a_prime_sampling_stage =
            scale_summary(self.a_prime_hash_sample.clone(), shape.ring_dim);
        let collapse_add_count = shape
            .modulus_digits
            .checked_mul(shape.ring_dim.saturating_sub(1))
            .expect("AKY24IO noise-refresh collapse add count overflow");
        let per_refresh = match mode {
            PrfBenchMode::PublicKeyPreprocess => {
                let pk_matrix_mul =
                    self.public_key_estimator.estimate_large_scalar_mul(&scalar_target);
                let pk_add = self.public_key_estimator.estimate_add();
                let pk_sub = self.public_key_estimator.estimate_sub();
                let combine_unit = sequential_summaries(&[
                    estimate_summary(pk_matrix_mul.clone()),
                    estimate_summary(pk_matrix_mul.clone()),
                    scale_estimate(pk_matrix_mul, shape.modulus_digits),
                    scale_estimate(pk_add.clone(), collapse_add_count),
                    estimate_summary(pk_add),
                    estimate_summary(pk_sub),
                ]);
                sequential_summaries(&[
                    a_prime_sampling_stage.clone(),
                    scale_summary(combine_unit, combine_task_count),
                ])
            }
            PrfBenchMode::EncodingOnline => {
                let enc_matrix_mul =
                    self.encoding_estimator.estimate_large_scalar_mul(&scalar_target);
                let enc_add = self.encoding_estimator.estimate_add();
                let enc_sub = self.encoding_estimator.estimate_sub();
                let crt_recompose = self.native_estimator.estimate_vector_add(shape.modulus_digits);
                let combine_unit = sequential_summaries(&[
                    estimate_summary(enc_matrix_mul.clone()),
                    estimate_summary(enc_matrix_mul.clone()),
                    scale_estimate(enc_matrix_mul, shape.modulus_digits),
                    scale_estimate(enc_add.clone(), collapse_add_count),
                    estimate_summary(enc_add),
                    estimate_summary(enc_sub.clone()),
                    estimate_summary(enc_sub),
                    estimate_summary(crt_recompose),
                ]);
                sequential_summaries(&[
                    a_prime_sampling_stage,
                    scale_summary(combine_unit, combine_task_count),
                ])
            }
        };
        let total = sequential_summaries(&[material.clone(), per_refresh.clone()]);
        NoiseRefreshBenchEstimateParts { material, per_refresh, total }
    }

    fn estimate_final_projection_preprocess(&self, shape: &Aky24IOBenchShape) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator,
    {
        let standard_preimages =
            shape.input_size.checked_add(2).expect("AKY24IO projection count overflow");
        let decoder_preimages = shape.final_decoder_count();
        let final_projection_hash_sampling = scale_estimate(
            self.full_w_block_hash_sample.clone(),
            standard_preimages
                .checked_add(decoder_preimages)
                .expect("AKY24IO final projection hash count overflow"),
        );
        let target_building = parallel_summaries(&[
            scale_summary(
                self.native_estimator.estimate_vector_matrix_product(shape.modulus_digits, 1),
                standard_preimages,
            ),
            scale_summary(
                self.native_estimator.estimate_vector_matrix_product(shape.modulus_digits, 1),
                decoder_preimages,
            ),
        ]);
        let preimages = parallel_summaries(&[
            scale_estimate(self.final_output_preimage_extend.clone(), standard_preimages),
            scale_estimate(self.final_decoder_preimage_extend.clone(), decoder_preimages),
        ]);
        let inputs = parallel_summaries(&[final_projection_hash_sampling, target_building]);
        sequential_summaries(&[inputs, preimages])
    }

    fn estimate_public_lut_aux_storage_bytes<M, PKPE, PKST, ENCPE, ENCST>(
        &self,
        scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
        shape: &Aky24IOBenchShape,
    ) -> BigUint
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
    {
        let prg_circuit = build_representative_goldreich_prg_one_output_circuit(scheme);
        let prg_aux = self
            .public_key_estimator
            .estimate_public_lut_sample_aux_matrices_for_circuit(&scheme.params, &prg_circuit);
        if prg_aux.compact_bytes == BigUint::default() {
            return BigUint::default();
        }

        prg_aux.compact_bytes.clone() * shape.public_lut_prg_output_count()
    }
}

#[derive(Debug, Clone, Copy)]
enum PrfBenchMode {
    PublicKeyPreprocess,
    EncodingOnline,
}

#[derive(Debug, Clone, PartialEq)]
struct NoiseRefreshBenchEstimateParts {
    material: CircuitBenchSummary,
    per_refresh: CircuitBenchSummary,
    total: CircuitBenchSummary,
}

#[derive(Debug, Clone, PartialEq)]
struct Aky24IOPrfBenchEstimateParts {
    selected_branch_prg: CircuitBenchSummary,
    selected_branch_rebase: CircuitBenchSummary,
    refresh_decoder_work: CircuitBenchSummary,
    noise_refresh: CircuitBenchSummary,
    final_prg: CircuitBenchSummary,
    final_mask_decrypt: CircuitBenchSummary,
    final_function_decrypt: CircuitBenchSummary,
    compute_without_refresh_decoder: CircuitBenchSummary,
    total: CircuitBenchSummary,
}

fn build_representative_goldreich_prg_one_output_circuit<M, PKPE, PKST, ENCPE, ENCST>(
    scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
) -> PolyCircuit<M::P>
where
    M: PolyMatrix,
    M::P: 'static,
{
    let mut circuit = PolyCircuit::new();
    let ring_gsw_context = build_ring_gsw_circuit_context(scheme, &mut circuit);
    let seed_ciphertexts = (0..5)
        .map(|_| {
            RingGswCiphertext::input(
                ring_gsw_context.clone(),
                Some(BigUint::from(1u64)),
                &mut circuit,
            )
        })
        .collect::<Vec<_>>();
    let graph = GoldreichGraph::from_edges(
        5,
        vec![GoldreichEdge::new([0, 1, 2], [3, 4])],
        Default::default(),
    );
    let goldreich = GoldreichFhePrg::from_public_graph(&mut circuit, ring_gsw_context, graph);
    let outputs = goldreich.evaluate_uniform(&seed_ciphertexts, &mut circuit);
    circuit.output(outputs.iter().flat_map(|output| output.sub_circuit_wires()));
    circuit
}

fn prf_mask_decrypt_one_ciphertext_bit_circuit<M, PKPE, PKST, ENCPE, ENCST>(
    scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
) -> PolyCircuit<M::P>
where
    M: PolyMatrix,
    M::P: 'static,
    NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
{
    let mut circuit = PolyCircuit::new();
    let ring_gsw_context = build_ring_gsw_circuit_context(scheme, &mut circuit);
    let decrypted = append_one_ciphertext_bit_decrypt::<M::P, NestedRnsPoly<M::P>, M>(
        &mut circuit,
        ring_gsw_context,
        BigUint::from(2u64),
    );
    circuit.output(vec![decrypted.secret_dependent, decrypted.public_bottom]);
    circuit
}

fn build_ring_gsw_circuit_context<M, PKPE, PKST, ENCPE, ENCST>(
    scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
    circuit: &mut PolyCircuit<M::P>,
) -> Arc<NestedRnsRingGswContext<M::P>>
where
    M: PolyMatrix,
    M::P: 'static,
{
    let nested_rns_context = Arc::new(crate::gadgets::arith::NestedRnsPolyContext::setup(
        circuit,
        &scheme.params,
        scheme.ring_gsw_context.p_moduli_bits,
        scheme.ring_gsw_context.max_unreduced_muls,
        scheme.ring_gsw_context.scale,
        false,
        scheme.ring_gsw_enable_levels,
    ));
    Arc::new(NestedRnsRingGswContext::<M::P>::from_arith_context(
        circuit,
        &scheme.params,
        scheme.params.ring_dimension() as usize,
        nested_rns_context,
        scheme.ring_gsw_enable_levels,
        Some(scheme.ring_gsw_level_offset),
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Aky24IOBenchShape {
    ring_dim: usize,
    input_size: usize,
    output_size: usize,
    seed_bits: usize,
    prf_round_count: usize,
    prf_branch_count: usize,
    prf_mask_output_coeff_bits: usize,
    noise_refresh_v_bits: usize,
    cbd_n: usize,
    crt_depth: usize,
    modulus_digits: usize,
    modulus_bits: u16,
    state_col_size: usize,
    ring_gsw_wire_count: usize,
}

impl Aky24IOBenchShape {
    fn from_scheme<M, PKPE, PKST, ENCPE, ENCST>(
        scheme: &Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
        func: Aky24IOFuncType,
    ) -> Self
    where
        M: PolyMatrix,
    {
        let params = &scheme.params;
        let ring_dim = params.ring_dimension() as usize;
        let (_, _, crt_depth) = params.to_crt();
        let modulus_digits = params.modulus_digits();
        let modulus_bits = params
            .modulus_bits()
            .try_into()
            .expect("AKY24IO modulus bits must fit in u16 for compact-byte estimates");
        let state_col_size = modulus_digits + 2;
        let ring_gsw_active_levels = scheme.ring_gsw_enable_levels.unwrap_or_else(|| {
            scheme
                .ring_gsw_context
                .q_moduli_depth
                .checked_sub(scheme.ring_gsw_level_offset)
                .expect("AKY24IO Ring-GSW level offset exceeds q-moduli depth")
        });
        let ring_gsw_wire_count = 2usize
            .checked_mul(scheme.ring_gsw_width)
            .and_then(|count| count.checked_mul(ring_gsw_active_levels))
            .and_then(|count| count.checked_mul(scheme.ring_gsw_context.p_moduli.len()))
            .expect("AKY24IO Ring-GSW wire count overflow");
        Self {
            ring_dim,
            input_size: scheme.input_size,
            output_size: func.output_bits(),
            seed_bits: scheme.seed_bits,
            prf_round_count: scheme.public_prf_seed_bits,
            prf_branch_count: 2,
            prf_mask_output_coeff_bits: scheme.prf_mask_output_coeff_bits,
            noise_refresh_v_bits: scheme.noise_refresh_v_bits,
            cbd_n: scheme.noise_refresh_cbd_n,
            crt_depth,
            modulus_digits,
            modulus_bits,
            state_col_size,
            ring_gsw_wire_count,
        }
    }

    fn final_decoder_count(&self) -> usize {
        self.output_size.checked_mul(self.ring_dim).expect("AKY24IO final decoder count overflow")
    }

    fn final_mask_prg_output_count(&self) -> usize {
        self.final_decoder_count()
            .checked_mul(self.prf_mask_output_coeff_bits)
            .expect("AKY24IO final mask PRG output count overflow")
    }

    fn final_prg_output_count(&self) -> usize {
        self.final_mask_prg_output_count()
            .checked_add(self.output_size)
            .expect("AKY24IO final PRG output count overflow")
    }

    fn obfuscated_circuit_bytes(&self, public_lut_aux_bytes: BigUint) -> BigUint {
        BigUint::from(self.final_projection_preimage_bytes()) +
            self.prf_refresh_preimage_bytes() +
            public_lut_aux_bytes
    }

    fn final_projection_preimage_bytes(&self) -> usize {
        let standard_preimages =
            self.input_size.checked_add(2).expect("AKY24IO projection count overflow");
        let output_preimage_bytes = matrix_compact_bytes_for_shape(
            self.state_col_size,
            self.modulus_digits,
            self.ring_dim,
            self.modulus_bits,
        );
        let final_decoder_preimage_bytes = matrix_compact_bytes_for_shape(
            self.state_col_size,
            1,
            self.ring_dim,
            self.modulus_bits,
        );
        output_preimage_bytes
            .checked_mul(standard_preimages)
            .and_then(|bytes| {
                bytes.checked_add(
                    final_decoder_preimage_bytes
                        .checked_mul(self.final_decoder_count())
                        .expect("AKY24IO final decoder preimage byte count overflow"),
                )
            })
            .expect("AKY24IO final projection byte count overflow")
    }

    fn selected_prg_output_count(&self) -> usize {
        self.prf_round_count
            .checked_mul(self.prf_branch_count)
            .and_then(|count| count.checked_mul(self.seed_bits))
            .expect("AKY24IO selected-branch PRG output count overflow")
    }

    fn sparse_noise_refresh_material_uniform_prg_output_count(&self, input_size: usize) -> BigUint {
        let error_ciphertexts = self
            .ring_dim
            .checked_mul(self.modulus_digits)
            .and_then(|count| count.checked_mul(self.ring_dim))
            .expect("AKY24IO noise-refresh CBD ciphertext count overflow");
        let mask_ciphertexts = error_ciphertexts
            .checked_mul(self.crt_depth)
            .and_then(|count| count.checked_mul(self.noise_refresh_v_bits))
            .expect("AKY24IO noise-refresh mask ciphertext count overflow");
        let uniform_outputs = error_ciphertexts
            .checked_mul(2)
            .and_then(|count| count.checked_mul(self.cbd_n))
            .and_then(|count| count.checked_add(mask_ciphertexts))
            .expect("AKY24IO noise-refresh uniform PRG output count overflow");
        BigUint::from(uniform_outputs) * BigUint::from(input_size)
    }

    fn prf_refresh_decoder_preimage_count(&self) -> BigUint {
        BigUint::from(self.prf_round_count) *
            BigUint::from(self.prf_branch_count) *
            BigUint::from(self.seed_bits) *
            BigUint::from(self.ring_gsw_wire_count) *
            BigUint::from(self.ring_dim) *
            BigUint::from(self.crt_depth)
    }

    fn prf_branch_rebase_preimage_count(&self) -> BigUint {
        BigUint::from(self.prf_round_count) *
            BigUint::from(self.prf_branch_count) *
            BigUint::from(self.seed_bits) *
            BigUint::from(self.ring_gsw_wire_count) *
            BigUint::from(self.ring_dim)
    }

    fn prf_refresh_preimage_bytes(&self) -> BigUint {
        let output_preimage_bytes = matrix_compact_bytes_for_shape(
            self.state_col_size,
            self.modulus_digits,
            self.ring_dim,
            self.modulus_bits,
        );
        BigUint::from(output_preimage_bytes) *
            (self.prf_refresh_decoder_preimage_count() + self.prf_branch_rebase_preimage_count())
    }

    fn public_lut_prg_output_count(&self) -> BigUint {
        let selected_prg_outputs = BigUint::from(self.selected_prg_output_count());
        let noise_refresh_material_prg_outputs = self
            .sparse_noise_refresh_material_uniform_prg_output_count(
                self.prf_round_count
                    .checked_mul(self.prf_branch_count)
                    .expect("AKY24IO branch-specific noise-refresh material count overflow"),
            );
        let final_prg_outputs = BigUint::from(self.final_prg_output_count());
        selected_prg_outputs + noise_refresh_material_prg_outputs + final_prg_outputs
    }
}

fn matrix_compact_bytes_for_shape(
    nrow: usize,
    ncol: usize,
    ring_dim: usize,
    modulus_bits: u16,
) -> usize {
    let coeff_count =
        nrow.checked_mul(ncol).and_then(|count| count.checked_mul(ring_dim)).unwrap_or(usize::MAX);
    let payload = coeff_count.checked_mul(modulus_bits as usize).unwrap_or(usize::MAX).div_ceil(8);
    payload + 128
}

fn estimate_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
    let summary = CircuitBenchSummary::from_nanos(
        estimate.total_time,
        estimate.latency,
        estimate.max_parallelism,
    );
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(estimate.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn scale_estimate(estimate: CircuitBenchEstimate, count: usize) -> CircuitBenchSummary {
    scale_summary(estimate_summary(estimate), count)
}

fn scale_estimate_biguint(estimate: CircuitBenchEstimate, count: &BigUint) -> CircuitBenchSummary {
    scale_summary_biguint(estimate_summary(estimate), count)
}

fn scale_summary(summary: CircuitBenchSummary, count: usize) -> CircuitBenchSummary {
    let count = BigUint::from(count);
    let scaled = CircuitBenchSummary::from_nanos(
        summary.total_time.clone() * &count,
        summary.latency,
        summary.max_parallelism.clone() * count,
    );
    #[cfg(feature = "gpu")]
    {
        scaled.with_peak_vram(summary.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

fn scale_summary_biguint(summary: CircuitBenchSummary, count: &BigUint) -> CircuitBenchSummary {
    let scaled = CircuitBenchSummary::from_nanos(
        summary.total_time.clone() * count,
        summary.latency,
        summary.max_parallelism.clone() * count,
    );
    #[cfg(feature = "gpu")]
    {
        scaled.with_peak_vram(summary.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

fn repeat_sequential_summary(summary: CircuitBenchSummary, count: usize) -> CircuitBenchSummary {
    let total_time = summary.total_time * BigUint::from(count);
    let latency = summary.latency * count as f64;
    let repeated = CircuitBenchSummary::from_nanos(total_time, latency, summary.max_parallelism);
    #[cfg(feature = "gpu")]
    {
        repeated.with_peak_vram(summary.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        repeated
    }
}

fn parallel_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts.iter().map(|part| part.total_time.clone()).sum::<BigUint>();
    let latency = parts.iter().map(|part| part.latency).fold(0.0f64, f64::max);
    let max_parallelism = parts.iter().map(|part| part.max_parallelism.clone()).sum::<BigUint>();
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.iter().map(|part| part.peak_vram).sum::<usize>())
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts.iter().map(|part| part.total_time.clone()).sum::<BigUint>();
    let latency = parts.iter().map(|part| part.latency).sum::<f64>();
    let max_parallelism =
        parts.iter().map(|part| part.max_parallelism.clone()).max().unwrap_or_else(BigUint::zero);
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.iter().map(|part| part.peak_vram).max().unwrap_or(0))
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}
