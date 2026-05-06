use std::{fs, marker::PhantomData, path::Path, sync::Arc, time::Instant};

use digest::Digest;
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use tracing::{debug, info};

use crate::{
    bgg::{
        naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
        public_key::BggPublicKey,
        sampler::BGGPublicKeySampler,
    },
    circuit::{Evaluable, PolyCircuit},
    func_enc::aky24::NoCircuitEvaluator,
    gadgets::{
        arith::{NestedRnsPoly, NestedRnsPolyContext},
        fhe::{
            ring_gsw::RingGswCiphertext,
            ring_gsw_nested_rns::{
                NativeRingGswCiphertext, NestedRnsRingGswContext, ciphertext_inputs_from_native,
                encrypt_plaintext_bit, sample_public_key,
            },
        },
        fhe_prg::goldreich::evaluate_goldreich_uniform_range,
    },
    input_injector::{
        DIAMOND_SECRET_SIZE, DiamondInjector, DiamondInjectorPreprocessOut, InputInjector,
    },
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    noise_refresh::{
        NoiseRefresherNaiveVec,
        circuit_decrypt::{
            decrypt_bit_decomposed_polynomial_parts, mask_plaintext_moduli_from_full_modulus,
        },
    },
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::SlotTransferEvaluator,
};

use super::Obfuscation;

pub mod bench_estimator;
mod circuits;
pub mod simulation;
mod utils;

#[cfg(feature = "gpu")]
pub use bench_estimator::GpuDCRTPolyMatrixNativeBenchEstimator;
pub use bench_estimator::{
    DiamondIOBenchEstimate, DiamondIOBenchEstimator, DiamondIONativeBenchEstimator,
};
pub use simulation::{
    DiamondIOCrtDepthSearchResult, DiamondIOErrorSimulation,
    DiamondIOPrfMaskOutputCoeffBitsSearchResult, DiamondIOPrfRoundErrorSimulation,
    diamond_io_find_crt_depth,
};

/// Decode one coefficient that should contain `q/2 * bit` plus a centered mask.
///
/// The final DiamondIO decoder cancels the secret-dependent BGG public-key
/// part and leaves a single noisy plaintext coefficient. PRF masking is
/// supposed to keep that coefficient inside the rounding interval around
/// either `0` or `q/2`, so decoding is ordinary rounding modulo plaintext
/// modulus two.
fn decode_centered_masked_boolean_coeff(coeff: BigUint, q_modulus: &BigUint) -> bool {
    let half_q = q_modulus / 2u32;
    let rounded = (BigUint::from(2u32) * coeff + half_q) / q_modulus;
    (&rounded % 2u32) == BigUint::from(1u32)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Function families supported by the Diamond iO wrapper.
pub enum DiamondIOFuncType {
    /// Debug circuit that decrypts private seed ciphertexts and returns only
    /// those decrypted seed bits. Explicit public input bits are still used as
    /// Diamond/PRF inputs, but they are not function outputs.
    DebugDecryption,
}

/// Small in-memory handle returned by `DiamondIO::obfuscation`.
///
/// Large Diamond preimage matrices are stored under the `dir_path` passed to
/// `obfuscation`; this handle keeps only the compact values needed to identify
/// and later evaluate that persisted preprocessing state.
#[derive(Debug, Clone)]
pub struct DiamondIOObf<M, T>
where
    M: PolyMatrix,
{
    /// Hash key sampled by `DiamondInjector::preprocess` and required by
    /// `DiamondInjector::online_eval` to identify the persisted preimages.
    pub preprocess_out: DiamondInjectorPreprocessOut<M, T>,
    /// Hash key used to deterministically sample the BGG public keys in this
    /// obfuscation instance.
    pub bgg_hash_key: [u8; 32],
    /// Function descriptor that determined the evaluated decoder circuit.
    pub func_type: DiamondIOFuncType,
    /// Native Ring-GSW public key for the sampled ternary decryption key `k`.
    pub ring_gsw_public_key: NativeRingGswCiphertext,
    /// Native Ring-GSW encryptions of the private PRF seed bits. Circuit input
    /// encodings can be derived again from these ciphertexts and the public
    /// Ring-GSW parameters, so no derived inputs are stored here.
    pub seed_ciphertexts: Vec<NativeRingGswCiphertext>,
    /// Original private seed bits, kept only in tests so debug circuits can
    /// assert that the obfuscated decryption path returns the sampled bits.
    #[cfg(test)]
    pub original_seed_bits: Vec<bool>,
    /// Test-only native Ring-GSW ciphertexts used to replace expensive
    /// Goldreich PRG circuit outputs. They are stored so eval can use the exact
    /// same random PRG-output wires as obfuscation.
    #[cfg(test)]
    pub debug_prg_ciphertexts: Vec<NativeRingGswCiphertext>,
}

/// Diamond iO frontend that turns a high-level function type into Diamond input
/// injection preprocessing artifacts.
///
/// The struct owns only reusable parameters and evaluators. Per-obfuscation
/// disk state is supplied through `Obfuscation::obfuscation` and
/// `Obfuscation::eval` as an explicit `dir_path`.
pub struct DiamondIO<
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
    /// Diamond input injector used to preprocess the final decoder public keys.
    pub injector: DiamondInjector<M, US, HS, TS>,
    /// Number of boolean input bits accepted by the selected iO function.
    pub input_size: usize,
    /// Number of boolean output bits produced by the selected iO function.
    pub output_size: usize,
    /// Native DCRT parameters used for Ring-GSW encryption before conversion
    /// into the BGG polynomial domain.
    pub native_poly_params: DCRTPolyParams,
    /// Nested-RNS arithmetic context used by Ring-GSW ciphertext conversion.
    pub ring_gsw_context: Arc<NestedRnsPolyContext>,
    /// Number of native Ring-GSW public-key columns.
    pub ring_gsw_width: usize,
    /// Level offset used when converting native Ring-GSW ciphertext entries
    /// into nested-RNS polynomial circuit inputs.
    pub ring_gsw_level_offset: usize,
    /// Optional number of nested-RNS levels enabled during ciphertext
    /// conversion and circuit construction.
    pub ring_gsw_enable_levels: Option<usize>,
    /// Optional Gaussian error used when sampling the native Ring-GSW public key.
    pub ring_gsw_public_key_error_sigma: Option<f64>,
    /// Domain-separation prefix for BGG public-key sampling tags.
    pub bgg_tag: Vec<u8>,
    /// Number of private PRF seed bits encrypted into Ring-GSW ciphertexts.
    pub seed_bits: usize,
    /// Number of bit-decomposed PRF mask output coefficients to compute.
    /// DiamondIO requires this to be positive because every decoder targets
    /// `function_output + prf_mask`.
    pub prf_mask_output_coeff_bits: usize,
    /// Test-only debug knob that replaces Goldreich PRG circuit evaluation
    /// with encrypted random PRG-output wires.
    #[cfg(test)]
    debug_encrypt_random_prg_wires: bool,
    /// Number of low bits retained in the noise-refresh rounding material.
    pub noise_refresh_v_bits: usize,
    /// Centered-binomial sample count used by noise-refresh material PRGs.
    pub noise_refresh_cbd_n: usize,
    /// Hash key used by the noise-refresh public material sampler.
    pub noise_refresh_hash_key: [u8; 32],
    /// Base seed from which per-round Goldreich PRG graphs are derived.
    pub goldreich_graph_seed: [u8; 32],
    /// Public-key lookup evaluator used when evaluating the selected circuit
    /// over `NaiveBGGPublicKeyVec` wires.
    pub pk_lookup_evaluator: Option<PKPE>,
    /// Public-key slot-transfer evaluator used when the selected circuit
    /// contains slot movement gates.
    pub pk_slot_transfer_evaluator: Option<PKST>,
    /// Public LUT base matrix `b0` used to bridge Diamond's final `(s, k) * B`
    /// state into the `s * b0` state expected by encoding lookup evaluators.
    pub enc_lookup_base_matrix: Option<M>,
    /// Factory for the encoding lookup evaluator used during `eval`.
    ///
    /// Encoding lookup needs the input-specific `c_b0 = s * b0`, so the
    /// evaluator is constructed lazily after Diamond online evaluation and the
    /// saved bridge preimage produce that state.
    pub enc_lookup_evaluator_factory: Option<Arc<dyn Fn(M) -> ENCPE + Send + Sync>>,
    /// Encoding slot-transfer evaluator used by PRF and function circuits
    /// during `eval`.
    pub enc_slot_transfer_evaluator: Option<ENCST>,
    _m: PhantomData<M>,
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> Obfuscation
    for DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    PKPE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
    PKST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
    ENCPE: PltEvaluator<NaiveBGGEncodingVec<M>>,
    ENCST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
{
    type FuncType = DiamondIOFuncType;
    type Obf = DiamondIOObf<M, TS::Trapdoor>;
    type Input = Vec<bool>;
    type Output = Vec<bool>;

    /// Create a Diamond obfuscation for `func`.
    ///
    /// The returned object keeps only compact public data. Large preimage
    /// matrices and sampled Diamond state are written under `dir_path`.
    fn obfuscation(&self, dir_path: &Path, func: Self::FuncType) -> Self::Obf {
        let obfuscation_started = Instant::now();
        info!(
            ?func,
            input_size = self.input_size,
            output_size = self.output_size,
            seed_bits = self.seed_bits,
            prf_mask_output_coeff_bits = self.prf_mask_output_coeff_bits,
            "DiamondIO obfuscation started"
        );
        // Validate shape agreements that are independent of the selected
        // function family.
        assert!(
            self.injector.base >= 2 && self.injector.base.is_power_of_two(),
            "DiamondIO requires a power-of-two DiamondInjector base"
        );
        let injector_input_bit_count = self
            .injector
            .input_count
            .checked_mul(self.injector.base.trailing_zeros() as usize)
            .expect("DiamondIO injector input bit count overflow");
        assert_eq!(
            self.input_size, injector_input_bit_count,
            "DiamondIO input_size must match the DiamondInjector bit input count"
        );
        let params = &self.injector.params;
        let ring_dim = params.ring_dimension() as usize;
        debug!(
            ring_dim,
            injector_input_count = self.injector.input_count,
            injector_base = self.injector.base,
            "DiamondIO obfuscation shape validation finished"
        );

        // Function-specific work is restricted to selecting and validating the
        // circuit. Key sampling, seed encryption, circuit evaluation, and
        // Diamond preprocessing below are shared across function types.
        let circuit_started = Instant::now();
        let circuit = match func {
            DiamondIOFuncType::DebugDecryption => {
                assert_eq!(
                    self.output_size, self.seed_bits,
                    "DebugDecryption output_size must be seed_bits"
                );
                self.build_debug_decryption_circuit()
            }
        };
        debug!(
            elapsed_ms = circuit_started.elapsed().as_millis(),
            "DiamondIO obfuscation selected function circuit"
        );

        // Sample the ternary Ring-GSW decryption key `k` and the BGG hash key
        // used to derive all scalar public keys for this obfuscation.
        let key_started = Instant::now();
        let uniform_sampler = US::new();
        let k = uniform_sampler.sample_poly(params, &DistType::TernaryDist);
        let native_k = DCRTPoly::from_biguints(&self.native_poly_params, &k.coeffs_biguints());
        let bgg_hash_key = rand::random::<[u8; 32]>();

        // Sample scalar BGG public keys for one, k, and every input bit. The
        // Diamond injector consumes the scalar keys directly.
        let (one, k_pubkey, input_digits) = self.sample_bgg_public_keys(params, bgg_hash_key);
        debug!(
            input_public_key_count = input_digits.len(),
            elapsed_ms = key_started.elapsed().as_millis(),
            "DiamondIO obfuscation sampled secret and scalar public keys"
        );

        // Duplicate scalar keys across all ring slots so the same public keys
        // can be used as `NaiveBGGPublicKeyVec` circuit inputs.
        let duplicate_started = Instant::now();
        let one_vec = Self::duplicate_public_key(params, &one, ring_dim);
        let k_vec = Self::duplicate_public_key(params, &k_pubkey, ring_dim);
        let input_digit_vecs = input_digits
            .iter()
            .map(|key| Self::duplicate_public_key(params, key, ring_dim))
            .collect::<Vec<_>>();
        debug!(
            duplicated_input_key_count = input_digit_vecs.len(),
            slot_count = ring_dim,
            elapsed_ms = duplicate_started.elapsed().as_millis(),
            "DiamondIO obfuscation duplicated public keys across slots"
        );

        // Sample the native Ring-GSW public key for `k`.
        let ring_gsw_started = Instant::now();
        let ring_gsw_public_key = sample_public_key(
            &self.native_poly_params,
            self.ring_gsw_width,
            &native_k,
            bgg_hash_key,
            b"diamond_io_ring_gsw_public_key",
            self.ring_gsw_public_key_error_sigma,
        );
        debug!(
            ring_gsw_width = self.ring_gsw_width,
            elapsed_ms = ring_gsw_started.elapsed().as_millis(),
            "DiamondIO obfuscation sampled Ring-GSW public key"
        );

        // Encrypt each private seed bit natively, convert the ciphertext to
        // nested-RNS plaintext inputs, and lift those inputs into public-key
        // wires by slot-wise multiplying the constant-one BGG key.
        let seed_started = Instant::now();
        let mut seed_ciphertexts = Vec::with_capacity(self.seed_bits);
        #[cfg(test)]
        let mut original_seed_bits = Vec::with_capacity(self.seed_bits);
        let mut enc_seed_public_keys = Vec::new();
        for seed_bit_idx in 0..self.seed_bits {
            let per_seed_started = Instant::now();
            let seed_bit = rand::random::<bool>();
            #[cfg(test)]
            original_seed_bits.push(seed_bit);
            let native_seed_ciphertext = encrypt_plaintext_bit(
                &self.native_poly_params,
                self.ring_gsw_context.as_ref(),
                &ring_gsw_public_key,
                seed_bit,
            );
            let seed_inputs = ciphertext_inputs_from_native::<M::P>(
                params,
                self.ring_gsw_context.as_ref(),
                &native_seed_ciphertext,
                self.ring_gsw_level_offset,
                self.ring_gsw_enable_levels,
            );
            enc_seed_public_keys.extend(seed_inputs.iter().map(|input| {
                assert_eq!(
                    one_vec.num_slots(),
                    input.len(),
                    "DiamondIO slot-wise scalar multiplication requires matching slot counts"
                );
                NaiveBGGPublicKeyVec::new(
                    params,
                    (0..one_vec.num_slots())
                        .map(|slot_idx| {
                            one_vec.key(slot_idx).large_scalar_mul(
                                params,
                                &input.as_slice()[slot_idx].coeffs_biguints(),
                            )
                        })
                        .collect(),
                )
            }));
            seed_ciphertexts.push(native_seed_ciphertext);
            debug!(
                seed_bit_idx,
                total_seed_wire_count = enc_seed_public_keys.len(),
                elapsed_ms = per_seed_started.elapsed().as_millis(),
                "DiamondIO obfuscation encrypted seed bit and lifted public wires"
            );
        }
        info!(
            seed_ciphertext_count = seed_ciphertexts.len(),
            seed_wire_count = enc_seed_public_keys.len(),
            elapsed_ms = seed_started.elapsed().as_millis(),
            "DiamondIO obfuscation seed encryption finished"
        );

        // Diamond preprocessing now produces only the final state trapdoor
        // basis. PRF refresh preimages and output projection preimages are
        // sampled here because they depend on the selected function's public
        // keys and the PRF mask public key.
        let preprocess_started = Instant::now();
        let preprocess_out = self.injector.preprocess(dir_path, &k);
        info!(
            elapsed_ms = preprocess_started.elapsed().as_millis(),
            "DiamondIO obfuscation Diamond preprocessing finished"
        );
        let lookup_bridge_started = Instant::now();
        let enc_lookup_base_matrix = self
            .enc_lookup_base_matrix
            .as_ref()
            .expect("DiamondIO obfuscation requires an encoding lookup base matrix");
        assert_eq!(
            enc_lookup_base_matrix.row_size(),
            DIAMOND_SECRET_SIZE,
            "DiamondIO encoding lookup base matrix row count must match the Diamond secret size"
        );
        let lookup_top = enc_lookup_base_matrix.clone();
        let lookup_bottom = M::zero(params, DIAMOND_SECRET_SIZE, lookup_top.col_size());
        let lookup_target = lookup_top.concat_rows(&[&lookup_bottom]);
        let final_w_block_col_size = DIAMOND_SECRET_SIZE
            .checked_mul(params.modulus_digits())
            .expect("DiamondIO final W block column count overflow");
        let lookup_ext_matrix = HS::new().sample_hash(
            params,
            preprocess_out.hash_key,
            format!("diamond_w_{}_{}", 0, self.injector.input_count),
            2usize
                .checked_mul(DIAMOND_SECRET_SIZE)
                .expect("DiamondIO lookup bridge state row count overflow"),
            final_w_block_col_size,
            DistType::FinRingDist,
        );
        let lookup_base_preimage = TS::new(params, self.injector.trapdoor_sigma).preimage_extend(
            params,
            &preprocess_out.final_trapdoor,
            &preprocess_out.final_pub_matrix,
            &lookup_ext_matrix,
            &lookup_target,
        );
        Self::write_io_matrix(dir_path, Self::enc_lookup_base_preimage_id(), &lookup_base_preimage);
        info!(
            target_rows = lookup_target.row_size(),
            target_cols = lookup_target.col_size(),
            elapsed_ms = lookup_bridge_started.elapsed().as_millis(),
            "DiamondIO obfuscation persisted encoding lookup bridge preimage"
        );

        // Precompute the PRF public-key path. The concrete PRF
        // seed is supplied later as the obfuscated circuit input, so the
        // selection is represented symbolically over public keys here. While
        // doing so, persist refresh preimages against the final Diamond state.
        let prf_started = Instant::now();
        #[cfg(test)]
        let mut debug_prg_ciphertexts = Vec::new();
        #[cfg(test)]
        let debug_prg_ciphertext_sink =
            self.debug_encrypt_random_prg_wires().then_some(&mut debug_prg_ciphertexts);
        #[cfg(not(test))]
        let debug_prg_ciphertext_sink: Option<&mut Vec<NativeRingGswCiphertext>> = None;
        let final_prf_mask_public_key_vecs = self.compute_prf_mask_public_key(
            Some(dir_path),
            Some(&preprocess_out),
            &one_vec,
            &k_vec,
            &input_digit_vecs,
            &enc_seed_public_keys,
            self.debug_encrypt_random_prg_wires().then_some(&ring_gsw_public_key),
            debug_prg_ciphertext_sink,
        );
        assert_eq!(
            final_prf_mask_public_key_vecs.len(),
            self.output_size,
            "DiamondIO must sample one final PRF mask public key per output"
        );
        info!(
            final_prf_mask_count = final_prf_mask_public_key_vecs.len(),
            elapsed_ms = prf_started.elapsed().as_millis(),
            "DiamondIO obfuscation PRF public-key path finished"
        );

        // Evaluate the selected function circuit over public keys. The inputs
        // are ordered as the circuit expects: decryption key followed by seed
        // ciphertext wires. Explicit public input bits are consumed by the
        // PRF path above, not by `DebugDecryption`.
        let function_eval_started = Instant::now();
        let mut function_inputs = Vec::with_capacity(1 + enc_seed_public_keys.len());
        function_inputs.push(NaiveBGGPublicKeyVec::new(params, vec![k_vec.key(0)]));
        function_inputs.extend(enc_seed_public_keys.iter().cloned());
        let pk_slot_transfer_evaluator = self
            .pk_slot_transfer_evaluator
            .as_ref()
            .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>);
        let evaluated_public_keys = circuit.eval(
            params,
            one_vec.clone(),
            function_inputs,
            self.pk_lookup_evaluator.as_ref(),
            pk_slot_transfer_evaluator,
            None,
        );
        assert_eq!(
            evaluated_public_keys.len(),
            2 * self.output_size,
            "DiamondIO evaluated public-key output count mismatch"
        );
        info!(
            function_input_count = 1 + enc_seed_public_keys.len(),
            evaluated_output_count = evaluated_public_keys.len(),
            elapsed_ms = function_eval_started.elapsed().as_millis(),
            "DiamondIO obfuscation function public-key evaluation finished"
        );
        let projection_started = Instant::now();
        let one_plaintext = M::P::const_one(params);
        let one_preimage =
            self.sample_final_output_preimage(&preprocess_out, 0, &one, Some(&one_plaintext), None);
        Self::write_io_matrix(dir_path, Self::one_preimage_id(), &one_preimage);
        let k_preimage = self.sample_final_output_preimage(
            &preprocess_out,
            0,
            &k_pubkey,
            None,
            Some(&one_plaintext),
        );
        Self::write_io_matrix(dir_path, Self::k_preimage_id(), &k_preimage);
        for (bit_idx, pubkey) in input_digits.iter().enumerate() {
            let digit_idx = bit_idx / self.injector.batch_bits();
            let bit_in_digit = bit_idx % self.injector.batch_bits();
            let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
            let preimage = self.sample_final_output_preimage(
                &preprocess_out,
                state_idx,
                pubkey,
                None,
                Some(&one_plaintext),
            );
            Self::write_io_matrix(dir_path, &Self::input_preimage_id(bit_idx), &preimage);
        }
        let identity_selector = M::identity(params, DIAMOND_SECRET_SIZE, None).slice_columns(0, 1);
        let mut decoder_idx = 0usize;
        for ((output_idx, public_key_pair), (mask_secret_vec, _mask_public_bottom_vec)) in
            evaluated_public_keys
                .chunks_exact(2)
                .enumerate()
                .zip(final_prf_mask_public_key_vecs.iter())
        {
            let function_outputs = public_key_pair[0].keys();
            let mask_outputs = mask_secret_vec.keys();
            assert!(
                function_outputs.len() == 1 || function_outputs.len() == mask_outputs.len(),
                "DiamondIO function output {output_idx} must be scalar or match the PRF mask slot count"
            );
            for slot_idx in 0..mask_outputs.len() {
                // Decoder preimages target the masked secret-dependent branch.
                // The split Ring-GSW public-bottom branch is not decoded here;
                // eval later adds its revealed plaintext directly.
                let function_output = if function_outputs.len() == 1 {
                    function_outputs[0].clone()
                } else {
                    function_outputs[slot_idx].clone()
                };
                let masked_output = function_output + &mask_outputs[slot_idx];
                let top = masked_output.matrix.mul_decompose(&identity_selector);
                let bottom = M::zero(params, DIAMOND_SECRET_SIZE, top.col_size());
                let target = top.concat_rows(&[&bottom]);
                let final_w_block_col_size = DIAMOND_SECRET_SIZE
                    .checked_mul(params.modulus_digits())
                    .expect("DiamondIO final W block column count overflow");
                let ext_matrix = HS::new().sample_hash(
                    params,
                    preprocess_out.hash_key,
                    format!("diamond_w_{}_{}", 0, self.injector.input_count),
                    2usize
                        .checked_mul(DIAMOND_SECRET_SIZE)
                        .expect("DiamondIO final state row count overflow"),
                    final_w_block_col_size,
                    DistType::FinRingDist,
                );
                let preimage = TS::new(params, self.injector.trapdoor_sigma).preimage_extend(
                    params,
                    &preprocess_out.final_trapdoor,
                    &preprocess_out.final_pub_matrix,
                    &ext_matrix,
                    &target,
                );
                Self::write_io_matrix(dir_path, &Self::decoder_preimage_id(decoder_idx), &preimage);
                decoder_idx += 1;
            }
        }
        assert_eq!(
            decoder_idx,
            final_prf_mask_public_key_vecs
                .iter()
                .map(|(secret_dependent, _public_bottom)| secret_dependent.num_slots())
                .sum::<usize>(),
            "DiamondIO decoder preimage count must match secret-dependent evaluated public-key slots"
        );
        info!(
            input_preimage_count = input_digits.len(),
            decoder_preimage_count = decoder_idx,
            elapsed_ms = projection_started.elapsed().as_millis(),
            "DiamondIO obfuscation final projection preimages persisted"
        );
        info!(
            elapsed_ms = obfuscation_started.elapsed().as_millis(),
            "DiamondIO obfuscation finished"
        );

        DiamondIOObf {
            preprocess_out,
            bgg_hash_key,
            func_type: func,
            ring_gsw_public_key,
            seed_ciphertexts,
            #[cfg(test)]
            original_seed_bits,
            #[cfg(test)]
            debug_prg_ciphertexts,
        }
    }

    /// Evaluate an obfuscation on boolean input bits.
    ///
    /// This rebuilds the same public-key circuit inputs used during
    /// obfuscation, asks `DiamondInjector` for the input-specific BGG
    /// encodings, evaluates the function circuit over those encodings, and
    /// cancels each evaluated output against the matching decoder encoding.
    fn eval(&self, dir_path: &Path, obf: &Self::Obf, input: Self::Input) -> Self::Output {
        let eval_started = Instant::now();
        info!(
            input_len = input.len(),
            output_size = self.output_size,
            seed_bits = self.seed_bits,
            "DiamondIO eval started"
        );
        assert_eq!(
            input.len(),
            self.input_size,
            "DiamondIO eval input length must match input_size"
        );
        assert_eq!(
            obf.seed_ciphertexts.len(),
            self.seed_bits,
            "DiamondIO obfuscation seed ciphertext count mismatch"
        );
        let params = &self.injector.params;
        let ring_dim = params.ring_dimension() as usize;
        let batch_bits = {
            assert!(
                self.injector.base >= 2 && self.injector.base.is_power_of_two(),
                "DiamondIO requires a power-of-two DiamondInjector base"
            );
            self.injector.base.trailing_zeros() as usize
        };
        assert_eq!(
            input.len() % batch_bits,
            0,
            "DiamondIO input length must be divisible by the injector digit bit width"
        );
        let input_digits = input
            .chunks_exact(batch_bits)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |digit, (bit_idx, bit)| digit | ((*bit as u32) << bit_idx))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            input_digits.len(),
            self.injector.input_count,
            "DiamondIO packed input digit count must match the injector input count"
        );
        debug!(
            batch_bits,
            packed_digit_count = input_digits.len(),
            ring_dim,
            "DiamondIO eval packed input digits"
        );

        // Rebuild only scalar public keys that identify the already persisted
        // DiamondIO projection artifacts. Online evaluation does not rerun the
        // function or PRF circuits over public keys.
        let public_key_started = Instant::now();
        let circuit = match obf.func_type {
            DiamondIOFuncType::DebugDecryption => self.build_debug_decryption_circuit(),
        };
        let (one, k_pubkey, input_digit_pubkeys) =
            self.sample_bgg_public_keys(params, obf.bgg_hash_key);
        debug!(
            input_public_key_count = input_digit_pubkeys.len(),
            elapsed_ms = public_key_started.elapsed().as_millis(),
            "DiamondIO eval rebuilt scalar public keys"
        );

        // The injector stops at the final input-dependent branch states. The
        // output projection preimages were sampled during obfuscation and are
        // applied below after the encoding circuit produces output public keys.
        let online_eval_started = Instant::now();
        let states = self.injector.online_eval(dir_path, &obf.preprocess_out, &input_digits);
        assert_eq!(
            states.len(),
            1 + self.input_size,
            "DiamondIO final Diamond state count mismatch"
        );
        info!(
            state_count = states.len(),
            elapsed_ms = online_eval_started.elapsed().as_millis(),
            "DiamondIO eval injector online evaluation finished"
        );

        // Build online encodings for the default one input, hidden key k, and
        // selected public input bits from the persisted final-state projection
        // preimages. Even in test debug mode these vectors come from the same
        // Diamond states as production eval; debug PRG wires are replayed from
        // the native ciphertexts sampled during obfuscation.
        let input_encoding_started = Instant::now();
        #[cfg(test)]
        let debug_final_secret_matrix = if crate::env::diamond_io_eval_relation_asserts() {
            debug!("DiamondIO eval reconstructing final secret matrix for relation diagnostics");
            Some(self.injector.debug_final_secret_matrix(dir_path, &input_digits))
        } else {
            debug!("DiamondIO eval skipping final secret relation diagnostics");
            None
        };
        debug!("DiamondIO eval loading one output preimage");
        let one_preimage = self.read_io_matrix(dir_path, Self::one_preimage_id());
        debug!("DiamondIO eval building one output encoding");
        let one_output = self.injector.build_output_encoding(
            states[0].clone() * &one_preimage,
            one,
            Some(M::P::const_one(params)),
        );
        debug!("DiamondIO eval built one output encoding");
        #[cfg(test)]
        if let Some(debug_final_secret_matrix) = debug_final_secret_matrix.as_ref() {
            let gadget = M::gadget_matrix(params, DIAMOND_SECRET_SIZE);
            let one_plaintext = M::P::const_one(params);
            let expected_one_vector = debug_final_secret_matrix.clone() *
                (one_output.pubkey.matrix.clone() - &(gadget * one_plaintext));
            assert_eq!(
                one_output.vector, expected_one_vector,
                "DiamondIO one encoding must satisfy s_final * (A_1 - G)"
            );
        }
        debug!("DiamondIO eval loading k output preimage");
        let k_preimage = self.read_io_matrix(dir_path, Self::k_preimage_id());
        debug!("DiamondIO eval building k output encoding");
        let k_output =
            self.injector.build_output_encoding(states[0].clone() * &k_preimage, k_pubkey, None);
        debug!("DiamondIO eval built k output encoding");
        #[cfg(test)]
        if let Some(debug_final_secret_matrix) = debug_final_secret_matrix.as_ref() {
            let gadget = M::gadget_matrix(params, DIAMOND_SECRET_SIZE);
            let k_plaintext = self.injector.read_preprocessed_k(dir_path);
            let expected_k_vector = debug_final_secret_matrix.clone() *
                k_output.pubkey.matrix.clone() -
                &(gadget * k_plaintext);
            assert_eq!(
                k_output.vector, expected_k_vector,
                "DiamondIO k encoding must satisfy s_final * A_k - k * G"
            );
        }
        let digit_outputs = input_digit_pubkeys
            .into_iter()
            .enumerate()
            .map(|(bit_idx, pubkey)| {
                debug!(bit_idx, "DiamondIO eval building input digit output encoding");
                let digit_idx = bit_idx / batch_bits;
                let bit_in_digit = bit_idx % batch_bits;
                let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
                let plaintext = M::P::from_usize_to_constant(
                    params,
                    self.injector.digit_bit_value(input_digits[digit_idx] as usize, bit_in_digit),
                );
                let output = self.injector.build_output_encoding(
                    states[state_idx].clone() *
                        &self.read_io_matrix(dir_path, &Self::input_preimage_id(bit_idx)),
                    pubkey,
                    Some(plaintext),
                );
                #[cfg(test)]
                if let Some(debug_final_secret_matrix) = debug_final_secret_matrix.as_ref() {
                    let gadget = M::gadget_matrix(params, DIAMOND_SECRET_SIZE);
                    let expected_input_vector = debug_final_secret_matrix.clone() *
                        (output.pubkey.matrix.clone() -
                            &(gadget *
                                output.plaintext.clone().expect(
                                    "DiamondIO test input public keys must reveal plaintexts",
                                )));
                    assert_eq!(
                        output.vector, expected_input_vector,
                        "DiamondIO input encoding {bit_idx} must satisfy s_final * (A_x - x * G)"
                    );
                }
                debug!(bit_idx, "DiamondIO eval built input digit output encoding");
                output
            })
            .collect::<Vec<_>>();
        debug!(
            digit_output_count = digit_outputs.len(),
            "DiamondIO eval built all input digit output encodings"
        );
        let one_encoding_vec = NaiveBGGEncodingVec::new(params, vec![one_output.clone(); ring_dim]);
        let k_encoding_vec = NaiveBGGEncodingVec::new(params, vec![k_output.clone(); ring_dim]);
        debug!(ring_dim, "DiamondIO eval built one/k full-slot encoding vectors");
        let seed_lift_started = Instant::now();
        let mut seed_encoding_inputs = Vec::new();
        let mut seed_wire_idx = 0usize;
        for (ciphertext_idx, ciphertext) in obf.seed_ciphertexts.iter().enumerate() {
            let seed_ciphertext_started = Instant::now();
            debug!(
                ciphertext_idx,
                "DiamondIO eval converting native seed ciphertext to PolyVec wires"
            );
            let seed_plaintext_inputs = ciphertext_inputs_from_native::<M::P>(
                params,
                self.ring_gsw_context.as_ref(),
                ciphertext,
                self.ring_gsw_level_offset,
                self.ring_gsw_enable_levels,
            );
            debug!(
                ciphertext_idx,
                seed_plaintext_input_count = seed_plaintext_inputs.len(),
                elapsed_ms = seed_ciphertext_started.elapsed().as_millis(),
                "DiamondIO eval converted one native seed ciphertext"
            );
            seed_encoding_inputs.reserve(seed_plaintext_inputs.len());
            for input in seed_plaintext_inputs {
                debug!(
                    wire_idx = seed_wire_idx,
                    slot_count = input.len(),
                    "DiamondIO eval lifting seed ciphertext PolyVec wire to BGG slots"
                );
                assert_eq!(
                    one_encoding_vec.num_slots(),
                    input.len(),
                    "DiamondIO slot-wise scalar multiplication requires matching slot counts"
                );
                seed_encoding_inputs.push(NaiveBGGEncodingVec::new(
                    params,
                    (0..one_encoding_vec.num_slots())
                        .map(|slot_idx| {
                            debug!(
                                wire_idx = seed_wire_idx,
                                slot_idx,
                                "DiamondIO eval large-scalar-multiplying one encoding slot for seed wire"
                            );
                            one_encoding_vec.encoding(slot_idx).large_scalar_mul(
                                params,
                                &input.as_slice()[slot_idx].coeffs_biguints(),
                            )
                        })
                        .collect(),
                ));
                #[cfg(test)]
                {
                    // Check the conversion/lifting order immediately so the
                    // native PolyVec wire can be dropped before the next
                    // ciphertext is materialized. Keeping all native wires
                    // alive at once creates a large number of tiny GPU matrix
                    // allocations and was the observed eval-side OOM source.
                    let lifted_input = seed_encoding_inputs
                        .last()
                        .expect("DiamondIO seed encoding input was just pushed");
                    for slot_idx in 0..lifted_input.num_slots() {
                        let expected_plaintext = input.as_slice()[slot_idx].coeffs_biguints();
                        let actual_plaintext = lifted_input
                            .encoding(slot_idx)
                            .plaintext
                            .as_ref()
                            .expect("DiamondIO lifted seed ciphertext wires must reveal plaintext")
                            .coeffs_biguints();
                        assert_eq!(
                            actual_plaintext, expected_plaintext,
                            "DiamondIO lifted seed ciphertext wire {seed_wire_idx} slot {slot_idx} plaintext must match the native PolyVec input"
                        );
                    }
                }
                seed_wire_idx += 1;
            }
            debug!(
                ciphertext_idx,
                total_seed_encoding_input_count = seed_encoding_inputs.len(),
                elapsed_ms = seed_ciphertext_started.elapsed().as_millis(),
                "DiamondIO eval lifted one seed ciphertext and released native wires"
            );
        }
        debug!(
            seed_encoding_input_count = seed_encoding_inputs.len(),
            elapsed_ms = seed_lift_started.elapsed().as_millis(),
            "DiamondIO eval lifted seed ciphertext wires"
        );
        #[cfg(test)]
        {
            assert_eq!(
                circuit.num_input(),
                1 + seed_encoding_inputs.len(),
                "DebugDecryption circuit input order must be decryption key followed by seed ciphertext wires"
            );
        }
        let digit_encoding_inputs = digit_outputs
            .iter()
            .enumerate()
            .map(|(bit_idx, encoding)| {
                debug!(
                    bit_idx,
                    "DiamondIO eval expanding input digit encoding to full-slot vector"
                );
                NaiveBGGEncodingVec::new(params, vec![encoding.clone(); ring_dim])
            })
            .collect::<Vec<_>>();
        debug!(
            digit_encoding_input_count = digit_encoding_inputs.len(),
            "DiamondIO eval expanded input digit encodings to full-slot vectors"
        );
        info!(
            seed_encoding_input_count = seed_encoding_inputs.len(),
            digit_encoding_input_count = digit_encoding_inputs.len(),
            elapsed_ms = input_encoding_started.elapsed().as_millis(),
            "DiamondIO eval built input encodings"
        );
        let lookup_started = Instant::now();
        let enc_lookup_base_preimage =
            self.read_io_matrix(dir_path, Self::enc_lookup_base_preimage_id());
        let c_b0 = states[0].clone() * &enc_lookup_base_preimage;
        #[cfg(test)]
        if let Some(debug_final_secret_matrix) = debug_final_secret_matrix.as_ref() {
            let enc_lookup_base_matrix = self
                .enc_lookup_base_matrix
                .as_ref()
                .expect("DiamondIO test requires the encoding lookup base matrix");
            let expected_c_b0 = debug_final_secret_matrix.clone() * enc_lookup_base_matrix.clone();
            assert_eq!(c_b0, expected_c_b0, "DiamondIO c_b0 must equal s_final * b0");
        }
        let enc_lookup_evaluator = (self
            .enc_lookup_evaluator_factory
            .as_ref()
            .expect("DiamondIO eval requires an encoding lookup evaluator factory"))(
            c_b0
        );
        let enc_slot_transfer_evaluator = self
            .enc_slot_transfer_evaluator
            .as_ref()
            .expect("DiamondIO eval requires an encoding slot-transfer evaluator");
        debug!(
            elapsed_ms = lookup_started.elapsed().as_millis(),
            "DiamondIO eval initialized encoding lookup and slot-transfer evaluators"
        );
        #[cfg(test)]
        let debug_prg_ciphertexts = obf.debug_prg_ciphertexts.as_slice();
        #[cfg(not(test))]
        let debug_prg_ciphertexts: &[NativeRingGswCiphertext] = &[];
        let prf_eval_started = Instant::now();
        let final_prf_mask_encodings = self.compute_prf_mask_encoding(
            dir_path,
            &states,
            &one_encoding_vec,
            &k_encoding_vec,
            &digit_encoding_inputs,
            seed_encoding_inputs.clone(),
            debug_prg_ciphertexts,
            &enc_lookup_evaluator,
            enc_slot_transfer_evaluator,
        );
        assert_eq!(
            final_prf_mask_encodings.len(),
            self.output_size,
            "DiamondIO eval must compute one final PRF mask encoding per output"
        );
        info!(
            final_prf_mask_count = final_prf_mask_encodings.len(),
            elapsed_ms = prf_eval_started.elapsed().as_millis(),
            "DiamondIO eval PRF mask encoding path finished"
        );
        let function_eval_started = Instant::now();
        let mut encoding_function_inputs = Vec::with_capacity(1 + seed_encoding_inputs.len());
        encoding_function_inputs.push(NaiveBGGEncodingVec::new(params, vec![k_output]));
        encoding_function_inputs.extend(seed_encoding_inputs.iter().cloned());
        let evaluated_encodings = circuit
            .eval(
                params,
                one_encoding_vec.clone(),
                encoding_function_inputs,
                Some(&enc_lookup_evaluator),
                Some(
                    enc_slot_transfer_evaluator
                        as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
                ),
                None,
            )
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            evaluated_encodings.len(),
            2 * self.output_size,
            "DiamondIO evaluated encoding count mismatch"
        );
        let evaluated_encodings = evaluated_encodings
            .chunks_exact(2)
            .enumerate()
            .flat_map(|(output_idx, output_pair)| {
                let (mask_secret_dependent, mask_public_bottom) =
                    &final_prf_mask_encodings[output_idx];
                let function_encodings = output_pair[0].encodings();
                let public_bottom_encodings = output_pair[1].encodings();
                assert!(
                    function_encodings.len() == 1 ||
                        function_encodings.len() == mask_secret_dependent.num_slots(),
                    "DiamondIO function output {output_idx} must be scalar or match the PRF mask slot count"
                );
                assert!(
                    public_bottom_encodings.len() == 1 ||
                        public_bottom_encodings.len() == mask_public_bottom.num_slots(),
                    "DiamondIO public-bottom output {output_idx} must be scalar or match the PRF mask slot count"
                );
                let masked_secret_dependent = NaiveBGGEncodingVec::new(
                    params,
                    (0..mask_secret_dependent.num_slots())
                        .map(|slot_idx| {
                            let output = if function_encodings.len() == 1 {
                                function_encodings[0].clone()
                            } else {
                                function_encodings[slot_idx].clone()
                            };
                            output + &mask_secret_dependent.encoding(slot_idx)
                        })
                        .collect(),
                );
                let masked_public_bottom = NaiveBGGEncodingVec::new(
                    params,
                    (0..mask_public_bottom.num_slots())
                        .map(|slot_idx| {
                            let output = if public_bottom_encodings.len() == 1 {
                                public_bottom_encodings[0].clone()
                            } else {
                                public_bottom_encodings[slot_idx].clone()
                            };
                            output + &mask_public_bottom.encoding(slot_idx)
                        })
                        .collect(),
                );
                [masked_secret_dependent, masked_public_bottom]
            })
            .collect::<Vec<_>>();
        // Decoder cancellation only needs the projected vector
        // `state * preimage`. The matching public key was already fixed when
        // the preimage was sampled during obfuscation, so eval does not rebuild
        // or otherwise depend on output public keys here.
        let decoder_started = Instant::now();
        let decoder_slot_counts = evaluated_encodings
            .iter()
            .step_by(2)
            .map(NaiveBGGEncodingVec::num_slots)
            .collect::<Vec<_>>();
        let identity_selector = M::identity(params, DIAMOND_SECRET_SIZE, None).slice_columns(0, 1);
        let decoder_count = decoder_slot_counts.iter().sum::<usize>();
        let decoder_outputs = (0..decoder_count)
            .map(|decoder_idx| {
                let preimage =
                    self.read_io_matrix(dir_path, &Self::decoder_preimage_id(decoder_idx));
                states[0].clone() * &preimage
            })
            .collect::<Vec<_>>();
        assert_eq!(decoder_outputs.len(), decoder_count, "DiamondIO decoder output count mismatch");
        #[cfg(test)]
        if let Some(debug_final_secret_matrix) = debug_final_secret_matrix.as_ref() {
            let mut decoder_idx = 0usize;
            for evaluated_vec in evaluated_encodings.iter().step_by(2) {
                for slot_idx in 0..evaluated_vec.num_slots() {
                    let projected_output_public_key = evaluated_vec
                        .encoding(slot_idx)
                        .pubkey
                        .matrix
                        .clone()
                        .mul_decompose(&identity_selector);
                    let expected_decoder_output =
                        debug_final_secret_matrix.clone() * projected_output_public_key;
                    assert_eq!(
                        decoder_outputs[decoder_idx], expected_decoder_output,
                        "DiamondIO decoder output {decoder_idx} must equal s_final * output public key projection"
                    );
                    decoder_idx += 1;
                }
            }
            assert_eq!(
                decoder_idx, decoder_count,
                "DiamondIO decoder relation check must cover every decoder output"
            );
        }
        debug!(
            decoder_count,
            elapsed_ms = decoder_started.elapsed().as_millis(),
            "DiamondIO eval loaded decoder outputs"
        );
        #[cfg(test)]
        if let Some(debug_final_secret_matrix) = debug_final_secret_matrix.as_ref() {
            let gadget = M::gadget_matrix(params, DIAMOND_SECRET_SIZE);
            for (output_idx, evaluated_vec) in evaluated_encodings.iter().step_by(2).enumerate() {
                let evaluated = evaluated_vec.encoding(0);
                if let Some(actual_plaintext) = evaluated.plaintext.as_ref() {
                    let secret_times_pubkey =
                        debug_final_secret_matrix.clone() * evaluated.pubkey.matrix.clone();
                    let expected_secret_dependent =
                        secret_times_pubkey - &(gadget.clone() * actual_plaintext.clone());
                    let matches_secret_dependent = evaluated.vector == expected_secret_dependent;
                    debug!(
                        output_idx,
                        matches_secret_dependent,
                        "DiamondIO evaluated secret-dependent output relation check"
                    );
                    assert!(
                        matches_secret_dependent,
                        "DiamondIO evaluated secret-dependent output {output_idx} must equal s_final * A_y - G * y"
                    );
                } else {
                    debug!(
                        output_idx,
                        "DiamondIO evaluated secret-dependent output hides plaintext metadata"
                    );
                }
            }
        }
        info!(
            function_input_count = 1 + seed_encoding_inputs.len(),
            evaluated_output_count = evaluated_encodings.len(),
            elapsed_ms = function_eval_started.elapsed().as_millis(),
            "DiamondIO eval function encoding evaluation finished"
        );

        let rounding_started = Instant::now();
        let q: Arc<BigUint> = params.modulus().into();
        let outputs =
            evaluated_encodings
                .chunks_exact(2)
                .enumerate()
                .zip(decoder_slot_counts.iter().scan(0usize, |offset, slots| {
                    let current = *offset;
                    *offset += *slots;
                    Some(current)
                }))
                .map(|((_output_idx, output_pair), decoder_offset)| {
                    let evaluated = output_pair[0].encoding(0);
                    let public_bottom_encoding = output_pair[1].encoding(0);
                    let public_bottom = public_bottom_encoding
                        .plaintext
                        .as_ref()
                        .expect("DiamondIO public Ring-GSW bottom output must reveal plaintext");
                    let decoder = decoder_outputs[decoder_offset].clone();
                    let noisy_plaintext = decoder -
                        &evaluated.vector.mul_decompose(&identity_selector) +
                        &M::from_poly_vec_row(params, vec![public_bottom.clone()]);
                    assert_eq!(
                        noisy_plaintext.size(),
                        (1, 1),
                        "DiamondIO decoded output must be a 1x1 noisy plaintext matrix"
                    );
                    let coeffs = noisy_plaintext.entry(0, 0).coeffs_biguints();
                    info!(
                        output_idx = _output_idx,
                        decoder_offset,
                        coeffs = ?coeffs,
                        "DiamondIO noisy plaintext coefficients"
                    );
                    coeffs
                        .into_iter()
                        .next()
                        .map(|coeff| decode_centered_masked_boolean_coeff(coeff, q.as_ref()))
                        .expect("DiamondIO output plaintext polynomial must have one coefficient")
                })
                .collect::<Vec<_>>();
        info!(
            output_count = outputs.len(),
            rounding_elapsed_ms = rounding_started.elapsed().as_millis(),
            elapsed_ms = eval_started.elapsed().as_millis(),
            "DiamondIO eval finished"
        );
        outputs
    }
}

#[cfg(test)]
mod scalar_tests {
    use super::decode_centered_masked_boolean_coeff;
    use num_bigint::BigUint;

    #[test]
    fn test_diamond_io_noisy_boolean_decoder_recovers_centered_masked_bits() {
        let q = BigUint::from(1_048_583u64);
        let half_q = &q / 2u32;
        let safe_radius = &q / 4u32;
        let masks = [
            BigUint::ZERO,
            BigUint::from(1u32),
            &safe_radius - 1u32,
            &q - 1u32,
            &q - (&safe_radius - 1u32),
        ];

        for bit in [false, true] {
            for mask in masks.iter().cloned() {
                let coeff = if bit { (&half_q + mask) % &q } else { mask };
                assert_eq!(
                    decode_centered_masked_boolean_coeff(coeff.clone(), &q),
                    bit,
                    "failed to decode bit={bit} from centered masked coefficient {coeff}"
                );
            }
        }
    }

    #[test]
    fn test_diamond_io_raw_mask_must_be_centered_by_subtracting_midpoint() {
        let q = BigUint::from(1_034_273u64);
        let mask_bits = 18usize;
        let midpoint = BigUint::from(1u64) << (mask_bits - 1);
        let raw_mask = BigUint::from(222_541u64);
        let wrongly_centered = (&raw_mask + &midpoint) % &q;
        let centered = (&raw_mask + &q - &midpoint) % &q;

        assert!(
            decode_centered_masked_boolean_coeff(wrongly_centered.clone(), &q),
            "adding the midpoint to a raw mask leaves a coefficient near q/2: {wrongly_centered}"
        );
        assert!(
            !decode_centered_masked_boolean_coeff(centered.clone(), &q),
            "subtracting the midpoint centers the same mask so it rounds away: {centered}"
        );
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {

    use std::sync::Arc;

    use keccak_asm::Keccak256;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        __PAIR, __TestState,
        bgg::{encoding::BggEncoding, sampler::BGGEncodingSampler},
        circuit::evaluable::PolyVec,
        gadgets::{
            arith::{ModularArithmeticContext, NestedRnsPolyContext},
            fhe::ring_gsw_nested_rns::{decrypt_ciphertext, sample_secret_key},
        },
        input_injector::DiamondInjector,
        lookup::{
            debug::{DebugNaiveBGGEncodingVecPltEvaluator, DebugNaiveBGGPublicKeyVecPltEvaluator},
            poly::PolyPltEvaluator,
            poly_vec::PolyVecPltEvaluator,
        },
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::{
            PolyParams,
            dcrt::{
                gpu::{GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::{
            PolyTrapdoorSampler,
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::{NaiveBGGVecSlotTransferEvaluator, PolyVecSlotTransferEvaluator},
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use num_traits::ToPrimitive;

    type TestInjector = DiamondInjector<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
    >;
    type TestDiamondIO = DiamondIO<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
        DebugNaiveBGGPublicKeyVecPltEvaluator<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >,
        NaiveBGGVecSlotTransferEvaluator,
        DebugNaiveBGGEncodingVecPltEvaluator<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >,
        NaiveBGGVecSlotTransferEvaluator,
    >;

    fn rounded_coeffs<P: Poly>(
        decrypted: &P,
        plaintext_modulus: u64,
        q_modulus: &BigUint,
    ) -> Vec<u64> {
        let half_q = q_modulus / BigUint::from(2u64);
        decrypted
            .coeffs_biguints()
            .into_iter()
            .map(|slot| {
                let rounded = (BigUint::from(plaintext_modulus) * slot + &half_q) / q_modulus;
                rounded.to_u64().expect("rounded plaintext slot must fit in u64") %
                    plaintext_modulus
            })
            .collect()
    }

    fn expected_coeffs(ring_dim: usize, expected: u64) -> Vec<u64> {
        let mut coeffs = vec![0u64; ring_dim];
        coeffs[0] = expected;
        coeffs
    }

    #[sequential_test::sequential]
    #[test]
    fn test_gpu_diamond_io_debug_decryption_polyvec_eval_returns_seed_bits() {
        let gpu_ids = detected_gpu_device_ids();
        assert!(!gpu_ids.is_empty(), "DiamondIO GPU test requires at least one GPU");
        gpu_device_sync();

        let native_poly_params = DCRTPolyParams::new(2, 2, 10, 5);
        let (moduli, _, _) = native_poly_params.to_crt();
        let poly_params = GpuDCRTPolyParams::new_with_gpu(
            native_poly_params.ring_dimension(),
            moduli,
            native_poly_params.base_bits(),
            vec![gpu_ids[0]],
            Some(2),
        );

        let active_levels = 2usize;
        let mut setup_circuit = PolyCircuit::<GpuDCRTPoly>::new();
        let ring_gsw_context = Arc::new(NestedRnsPolyContext::setup(
            &mut setup_circuit,
            &poly_params,
            5,
            2,
            1 << 11,
            false,
            Some(active_levels),
        ));
        let ring_gsw_level_offset = 0usize;
        let ring_gsw_enable_levels = Some(active_levels);
        let ring_gsw_width = 2 *
            <NestedRnsPolyContext as ModularArithmeticContext<GpuDCRTPoly>>::gadget_len(
                ring_gsw_context.as_ref(),
                ring_gsw_enable_levels,
                Some(ring_gsw_level_offset),
            );

        let input_count = 2usize;
        let input_base = 2usize;
        let input_size = 2usize;
        let seed_bits = vec![true];
        let injector = TestInjector::new(poly_params.clone(), input_count, input_base, 4.578, 0.0)
            .with_gpu_device_ids(vec![gpu_ids[0]]);
        let scheme = TestDiamondIO::new(
            injector,
            input_size,
            seed_bits.len(),
            native_poly_params.clone(),
            ring_gsw_context.clone(),
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            Some(0.0),
            b"diamond_io_gpu_polyvec_debug_test".to_vec(),
            seed_bits.len(),
            1,
            1,
            1,
            [0x24; 32],
            [0x42; 32],
            None,
            None,
            None,
            None,
            None,
        );
        let circuit = scheme.build_debug_decryption_circuit();

        let secret_key = sample_secret_key(&native_poly_params);
        let gpu_secret_key =
            GpuDCRTPoly::from_biguints(&poly_params, &secret_key.coeffs_biguints());
        let public_key = sample_public_key(
            &native_poly_params,
            ring_gsw_width,
            &secret_key,
            [0x51; 32],
            b"diamond_io_debug_decryption_polyvec_ring_gsw_public_key",
            Some(0.0),
        );

        let mut inputs = Vec::new();
        inputs.push(PolyVec::new(vec![gpu_secret_key]));
        for &seed_bit in seed_bits.iter() {
            let ciphertext = encrypt_plaintext_bit(
                &native_poly_params,
                ring_gsw_context.as_ref(),
                &public_key,
                seed_bit,
            );
            inputs.extend(ciphertext_inputs_from_native::<GpuDCRTPoly>(
                &poly_params,
                ring_gsw_context.as_ref(),
                &ciphertext,
                ring_gsw_level_offset,
                ring_gsw_enable_levels,
            ));
        }

        let one = PolyVec::new(vec![
            GpuDCRTPoly::const_one(&poly_params);
            poly_params.ring_dimension() as usize
        ]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        let outputs = circuit.eval(
            &poly_params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            None,
        );

        let q_modulus = ring_gsw_context
            .q_moduli()
            .iter()
            .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
        let ring_dim = poly_params.ring_dimension() as usize;
        for (output_idx, (output_pair, &seed_bit)) in
            outputs.chunks_exact(2).zip(seed_bits.iter()).enumerate()
        {
            assert_eq!(
                output_pair[0].len(),
                1,
                "DebugDecryption PolyVec secret-dependent output must be scalar"
            );
            assert_eq!(
                output_pair[1].len(),
                1,
                "DebugDecryption PolyVec public-bottom output must be scalar"
            );
            let output_poly = output_pair[0].as_slice().first().expect("scalar output missing") +
                output_pair[1].as_slice().first().expect("scalar output missing");
            let raw_coeffs = output_poly.coeffs_biguints();
            let mut canonical_scaled_coeffs = vec![BigUint::ZERO; ring_dim];
            if seed_bit {
                canonical_scaled_coeffs[0] = q_modulus.clone() / 2u32;
            }
            println!(
                "DebugDecryption direct PolyVec output {output_idx} raw plaintext: actual={:?}, canonical={:?}",
                raw_coeffs, canonical_scaled_coeffs
            );
            assert_eq!(
                rounded_coeffs(&output_poly, 2, &q_modulus),
                expected_coeffs(ring_dim, seed_bit as u64),
                "DebugDecryption direct PolyVec output {output_idx} should match Ring-GSW decrypt_batch rounded coefficients"
            );
        }
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_gpu_diamond_io_debug_decryption_bgg_eval_returns_expected_vectors() {
        let _storage_lock = storage_test_lock().await;
        let gpu_ids = detected_gpu_device_ids();
        assert!(!gpu_ids.is_empty(), "DiamondIO GPU test requires at least one GPU");
        gpu_device_sync();

        let active_levels_u32 = 2u32;
        let active_levels = active_levels_u32 as usize;
        let native_poly_params = DCRTPolyParams::new(2, active_levels, 10, 5);
        let (moduli, _, _) = native_poly_params.to_crt();
        let poly_params = GpuDCRTPolyParams::new_with_gpu(
            native_poly_params.ring_dimension(),
            moduli,
            native_poly_params.base_bits(),
            vec![gpu_ids[0]],
            Some(active_levels_u32),
        );

        let mut setup_circuit = PolyCircuit::<GpuDCRTPoly>::new();
        let ring_gsw_context = Arc::new(NestedRnsPolyContext::setup(
            &mut setup_circuit,
            &poly_params,
            5,
            2,
            1 << 11,
            false,
            Some(active_levels),
        ));
        let ring_gsw_level_offset = 0usize;
        let ring_gsw_enable_levels = Some(active_levels);
        let ring_gsw_width = 2 *
            <NestedRnsPolyContext as ModularArithmeticContext<GpuDCRTPoly>>::gadget_len(
                ring_gsw_context.as_ref(),
                ring_gsw_enable_levels,
                Some(ring_gsw_level_offset),
            );

        let input_count = 2usize;
        let input_base = 2usize;
        let input_size = 2usize;
        let seed_bits = vec![true];
        let injector = TestInjector::new(poly_params.clone(), input_count, input_base, 4.578, 0.0)
            .with_gpu_device_ids(vec![gpu_ids[0]]);
        let scheme = TestDiamondIO::new(
            injector,
            input_size,
            seed_bits.len(),
            native_poly_params.clone(),
            ring_gsw_context.clone(),
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            Some(0.0),
            b"diamond_io_gpu_bgg_debug_test".to_vec(),
            seed_bits.len(),
            1,
            1,
            1,
            [0x24; 32],
            [0x42; 32],
            None,
            None,
            None,
            None,
            None,
        );
        let circuit = scheme.build_debug_decryption_circuit();

        let ring_gsw_secret_key = sample_secret_key(&native_poly_params);
        let k = GpuDCRTPoly::from_biguints(&poly_params, &ring_gsw_secret_key.coeffs_biguints());
        let ring_gsw_public_key = sample_public_key(
            &native_poly_params,
            ring_gsw_width,
            &ring_gsw_secret_key,
            [0x51; 32],
            b"diamond_io_debug_decryption_bgg_ring_gsw_public_key",
            Some(0.0),
        );

        let (one_pubkey, k_pubkey, _) = scheme.sample_bgg_public_keys(&poly_params, [0x71; 32]);
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let bgg_secret =
            uniform_sampler.sample_uniform(&poly_params, 1, 1, DistType::TernaryDist).entry(0, 0);
        let bgg_secret_matrix =
            GpuDCRTPolyMatrix::from_poly_vec_row(&poly_params, vec![bgg_secret]);
        let one_encoding = BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(
            &poly_params,
            &[bgg_secret_matrix.entry(0, 0)],
            None,
        )
        .sample(&poly_params, &[one_pubkey.clone()], &[])
        .into_iter()
        .next()
        .expect("BGG one encoding sampler must produce the constant-one encoding");
        let k_gadget = GpuDCRTPolyMatrix::gadget_matrix(&poly_params, k_pubkey.matrix.row_size());
        let k_encoding = BggEncoding::new(
            bgg_secret_matrix.clone() * k_pubkey.matrix.clone() - &(k_gadget * k.clone()),
            k_pubkey,
            Some(k.clone()),
        );

        let ring_dim = poly_params.ring_dimension() as usize;
        let one_encoding_vec =
            NaiveBGGEncodingVec::new(&poly_params, vec![one_encoding.clone(); ring_dim]);
        let q_modulus = ring_gsw_context
            .q_moduli()
            .iter()
            .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
        let seed_ciphertexts = seed_bits
            .iter()
            .map(|&seed_bit| {
                encrypt_plaintext_bit(
                    &native_poly_params,
                    ring_gsw_context.as_ref(),
                    &ring_gsw_public_key,
                    seed_bit,
                )
            })
            .collect::<Vec<_>>();
        for (&seed_bit, ciphertext) in seed_bits.iter().zip(seed_ciphertexts.iter()) {
            let native_decrypted = decrypt_ciphertext(
                &native_poly_params,
                ring_gsw_context.as_ref(),
                &ciphertext,
                &ring_gsw_secret_key,
                2,
            );
            assert_eq!(
                rounded_coeffs(&native_decrypted, 2, &q_modulus),
                expected_coeffs(ring_dim, seed_bit as u64),
                "native Ring-GSW decrypt should recover the plaintext after rounding when error sigma is zero"
            );
        }

        let mut direct_inputs = Vec::new();
        direct_inputs.push(PolyVec::new(vec![k.clone()]));
        for ciphertext in seed_ciphertexts.iter() {
            direct_inputs.extend(ciphertext_inputs_from_native::<GpuDCRTPoly>(
                &poly_params,
                ring_gsw_context.as_ref(),
                ciphertext,
                ring_gsw_level_offset,
                ring_gsw_enable_levels,
            ));
        }
        let direct_outputs = circuit.eval(
            &poly_params,
            PolyVec::new(vec![GpuDCRTPoly::const_one(&poly_params); ring_dim]),
            direct_inputs,
            Some(&PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() }),
            Some(&PolyVecSlotTransferEvaluator::new()),
            Some(1),
        );

        let mut inputs = Vec::new();
        inputs.push(NaiveBGGEncodingVec::new(&poly_params, vec![k_encoding]));
        for ciphertext in seed_ciphertexts.iter() {
            let ciphertext_inputs = ciphertext_inputs_from_native::<GpuDCRTPoly>(
                &poly_params,
                ring_gsw_context.as_ref(),
                ciphertext,
                ring_gsw_level_offset,
                ring_gsw_enable_levels,
            );
            for input in ciphertext_inputs.iter() {
                let scalar_base = &one_encoding_vec;
                assert_eq!(
                    input.len(),
                    scalar_base.num_slots(),
                    "DebugDecryption BGG-lifted ciphertext wire must match the one encoding slot count"
                );
                inputs.push(NaiveBGGEncodingVec::new(
                    &poly_params,
                    (0..scalar_base.num_slots())
                        .map(|slot_idx| {
                            scalar_base.encoding(slot_idx).large_scalar_mul(
                                &poly_params,
                                &input.as_slice()[slot_idx].coeffs_biguints(),
                            )
                        })
                        .collect(),
                ));
            }
        }

        let dir = tempdir().expect("DebugDecryption BGG test must create a tempdir");
        init_storage_system(dir.path().to_path_buf());
        let lookup_hash_key = [0x91; 32];
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&poly_params, 4.578);
        let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&poly_params, 1);
        let checkpoint_prefix = "DEBUG_DIAMOND_IO_BGG_DECRYPTION_TEST".to_string();
        let pk_lookup_evaluator = DebugNaiveBGGPublicKeyVecPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::with_checkpoint_prefix(
            lookup_hash_key,
            b0_trapdoor,
            b0_matrix.clone(),
            dir.path().to_path_buf(),
            checkpoint_prefix.clone(),
        );
        pk_lookup_evaluator.sample_aux_matrices(&poly_params);
        wait_for_all_writes(dir.path().to_path_buf()).await.unwrap();
        let c_b0 = bgg_secret_matrix.clone() * b0_matrix;
        let lookup_evaluator = DebugNaiveBGGEncodingVecPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::with_checkpoint_prefix(
            lookup_hash_key,
            dir.path().to_path_buf(),
            checkpoint_prefix,
            c_b0,
        );
        let slot_transfer_evaluator = NaiveBGGVecSlotTransferEvaluator::new();
        let outputs = circuit.eval(
            &poly_params,
            one_encoding_vec,
            inputs,
            Some(&lookup_evaluator),
            Some(&slot_transfer_evaluator),
            Some(1),
        );

        for (output_idx, (output_pair, &seed_bit)) in
            outputs.chunks_exact(2).zip(seed_bits.iter()).enumerate()
        {
            assert_eq!(
                output_pair[0].num_slots(),
                1,
                "DebugDecryption BGG secret-dependent output must be scalar"
            );
            assert_eq!(
                output_pair[1].num_slots(),
                1,
                "DebugDecryption BGG public-bottom output must be scalar"
            );
            let output = output_pair[0].encoding(0);
            let public_bottom = output_pair[1].encoding(0);
            let direct_secret_plaintext = direct_outputs[2 * output_idx].as_slice()[0].clone();
            let direct_bottom_plaintext = direct_outputs[2 * output_idx + 1].as_slice()[0].clone();
            let direct_plaintext =
                direct_secret_plaintext.clone() + direct_bottom_plaintext.clone();
            let actual_secret_plaintext = output.plaintext.as_ref();
            let actual_bottom_plaintext = public_bottom.plaintext.as_ref().expect(
                "DebugDecryption BGG public-bottom plaintext must be revealed in this test",
            );
            if let Some(actual_secret_plaintext) = actual_secret_plaintext {
                assert_eq!(
                    actual_secret_plaintext.coeffs_biguints(),
                    direct_secret_plaintext.coeffs_biguints(),
                    "DebugDecryption BGG secret-dependent output {output_idx} plaintext must match direct PolyVec circuit evaluation"
                );
            }
            assert_eq!(
                actual_bottom_plaintext.coeffs_biguints(),
                direct_bottom_plaintext.coeffs_biguints(),
                "DebugDecryption BGG public-bottom output {output_idx} plaintext must match direct PolyVec circuit evaluation"
            );
            let direct_raw_coeffs = direct_plaintext.coeffs_biguints();
            let mut canonical_scaled_coeffs = vec![BigUint::ZERO; ring_dim];
            if seed_bit {
                canonical_scaled_coeffs[0] = q_modulus.clone() / 2u32;
            }
            println!(
                "DebugDecryption direct PolyVec output {output_idx} raw plaintext: actual={:?}, canonical={:?}",
                direct_raw_coeffs, canonical_scaled_coeffs
            );
            println!(
                "DebugDecryption direct PolyVec output {output_idx} rounded plaintext: actual={:?}, expected={:?}",
                rounded_coeffs(&direct_plaintext, 2, &q_modulus),
                expected_coeffs(ring_dim, seed_bit as u64)
            );
            assert_eq!(
                rounded_coeffs(&direct_plaintext, 2, &q_modulus),
                expected_coeffs(ring_dim, seed_bit as u64),
                "DebugDecryption direct PolyVec output {output_idx} should match Ring-GSW decrypt_batch rounded coefficients"
            );
            println!(
                "DebugDecryption BGG output {output_idx} rounded plaintext: actual={:?}, expected={:?}",
                rounded_coeffs(
                    &(direct_secret_plaintext.clone() + actual_bottom_plaintext.clone()),
                    2,
                    &q_modulus
                ),
                expected_coeffs(ring_dim, seed_bit as u64)
            );
            assert_eq!(
                rounded_coeffs(
                    &(direct_secret_plaintext.clone() + actual_bottom_plaintext.clone()),
                    2,
                    &q_modulus
                ),
                expected_coeffs(ring_dim, seed_bit as u64),
                "DebugDecryption BGG combined plaintext output {output_idx} should match Ring-GSW decrypt_batch rounded coefficients"
            );
            let secret_times_pubkey = bgg_secret_matrix.clone() * output.pubkey.matrix.clone();
            if let Some(actual_secret_plaintext) = actual_secret_plaintext {
                let output_gadget =
                    GpuDCRTPolyMatrix::gadget_matrix(&poly_params, output.pubkey.matrix.row_size());
                let gadget_plaintext = output_gadget.clone() * actual_secret_plaintext.clone();
                let matches_hybrid_minus =
                    output.vector == secret_times_pubkey.clone() - &gadget_plaintext;
                println!(
                    "DebugDecryption BGG output {output_idx} vector relation: expected_bit={}, hybrid_minus={}, actual_plaintext_coeffs={:?}",
                    seed_bit,
                    matches_hybrid_minus,
                    actual_secret_plaintext.coeffs_biguints()
                );
                assert!(
                    matches_hybrid_minus,
                    "DebugDecryption BGG secret-dependent output {output_idx} must satisfy s * A_y - G * y for its revealed output plaintext"
                );
            }
            let identity_selector = GpuDCRTPolyMatrix::identity(&poly_params, 1, None);
            let projected_vector = output.vector.mul_decompose(&identity_selector);
            let projected_secret_times_pubkey =
                secret_times_pubkey.mul_decompose(&identity_selector);
            let decoded_with_public_bottom = projected_secret_times_pubkey - &projected_vector +
                &GpuDCRTPolyMatrix::from_poly_vec_row(
                    &poly_params,
                    vec![actual_bottom_plaintext.clone()],
                );
            assert_eq!(
                decoded_with_public_bottom.size(),
                (1, 1),
                "DebugDecryption decoded BGG output must be scalar"
            );
            assert_eq!(
                rounded_coeffs(&decoded_with_public_bottom.entry(0, 0), 2, &q_modulus),
                expected_coeffs(ring_dim, seed_bit as u64),
                "DebugDecryption BGG output {output_idx} should round correctly after projection, public-bottom addition, and public-key cancellation"
            );
        }
    }

    #[tokio::test]
    #[ignore = "full DiamondIO PRF-mask GPU roundtrip is expensive"]
    #[sequential_test::sequential]
    async fn test_gpu_diamond_io_debug_decryption_eval_returns_seed_bits() {
        let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
        let _storage_lock = storage_test_lock().await;
        let gpu_ids = detected_gpu_device_ids();
        assert!(!gpu_ids.is_empty(), "DiamondIO GPU test requires at least one GPU");
        gpu_device_sync();

        let native_poly_params = DCRTPolyParams::new(2, 2, 10, 5);
        let (moduli, _, _) = native_poly_params.to_crt();
        let full_q = moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * q_i);
        let q_max = moduli.iter().copied().max().expect("CRT moduli must be nonempty");
        let prf_mask_output_coeff_bits = ((&full_q / BigUint::from(2u64)).bits() - 1) as usize;
        let noise_refresh_v_bits = ((&full_q / q_max).bits() - 1) as usize;
        let poly_params = GpuDCRTPolyParams::new_with_gpu(
            native_poly_params.ring_dimension(),
            moduli,
            native_poly_params.base_bits(),
            vec![gpu_ids[0]],
            Some(1),
        );

        let active_levels = 2usize;
        let mut setup_circuit = PolyCircuit::<GpuDCRTPoly>::new();
        let ring_gsw_context = Arc::new(NestedRnsPolyContext::setup(
            &mut setup_circuit,
            &poly_params,
            5,
            2,
            1 << 8,
            false,
            Some(active_levels),
        ));
        let ring_gsw_level_offset = 0usize;
        let ring_gsw_enable_levels = Some(active_levels);
        let ring_gsw_width = 2 *
            <NestedRnsPolyContext as ModularArithmeticContext<GpuDCRTPoly>>::gadget_len(
                ring_gsw_context.as_ref(),
                ring_gsw_enable_levels,
                Some(ring_gsw_level_offset),
            );

        let input_count = 2usize;
        let input_base = 2usize;
        let input_size = 2usize;
        let seed_bits = 5usize;
        let output_size = seed_bits;
        let injector = TestInjector::new(poly_params.clone(), input_count, input_base, 4.578, 0.0)
            .with_gpu_device_ids(vec![gpu_ids[0]]);
        let dir = tempdir().expect("DiamondIO GPU test must create a tempdir");
        init_storage_system(dir.path().to_path_buf());
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&poly_params, 4.578);
        let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&poly_params, 1);
        let lookup_hash_key = [0x91; 32];
        let pk_lookup_evaluator =
            DebugNaiveBGGPublicKeyVecPltEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyHashSampler<Keccak256>,
                GpuDCRTPolyTrapdoorSampler,
            >::new(
                lookup_hash_key, b0_trapdoor, b0_matrix.clone(), dir.path().to_path_buf()
            );
        pk_lookup_evaluator.sample_aux_matrices(&poly_params);
        wait_for_all_writes(dir.path().to_path_buf()).await.unwrap();
        let enc_lookup_dir = dir.path().to_path_buf();
        let scheme = TestDiamondIO::new(
            injector,
            input_size,
            output_size,
            native_poly_params,
            ring_gsw_context,
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            Some(0.0),
            b"diamond_io_gpu_test".to_vec(),
            seed_bits,
            prf_mask_output_coeff_bits,
            noise_refresh_v_bits,
            1,
            [0x24; 32],
            [0x42; 32],
            Some(pk_lookup_evaluator),
            Some(NaiveBGGVecSlotTransferEvaluator::new()),
            Some(b0_matrix),
            Some(Arc::new(move |c_b0| {
                DebugNaiveBGGEncodingVecPltEvaluator::<
                    GpuDCRTPolyMatrix,
                    GpuDCRTPolyHashSampler<Keccak256>,
                    GpuDCRTPolyTrapdoorSampler,
                >::new(lookup_hash_key, enc_lookup_dir.clone(), c_b0)
            })),
            Some(NaiveBGGVecSlotTransferEvaluator::new()),
        );
        let obf = scheme.obfuscation(dir.path(), DiamondIOFuncType::DebugDecryption);
        let input = vec![true, false];
        let output = scheme.eval(dir.path(), &obf, input.clone());

        assert_eq!(output.len(), output_size);
        assert_eq!(output.as_slice(), obf.original_seed_bits.as_slice());
    }
}
