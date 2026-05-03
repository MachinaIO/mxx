use std::{fs, marker::PhantomData, path::Path, sync::Arc};

use num_bigint::BigUint;

use crate::{
    bgg::{
        encoding::BggEncoding,
        naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
        public_key::BggPublicKey,
        sampler::BGGPublicKeySampler,
    },
    circuit::{Evaluable, PolyCircuit},
    func_enc::aky24::NoCircuitEvaluator,
    gadgets::{
        arith::NestedRnsPolyContext,
        fhe::{
            ring_gsw::RingGswCiphertext,
            ring_gsw_nested_rns::{
                NativeRingGswCiphertext, NestedRnsRingGswContext, ciphertext_inputs_from_native,
                encrypt_plaintext_bit, sample_public_key,
            },
        },
    },
    input_injector::{
        DIAMOND_SECRET_SIZE, DiamondInjector, DiamondInjectorPreprocessOut, InputInjector,
    },
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::SlotTransferEvaluator,
};

use super::Obfuscation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Function families supported by the Diamond iO wrapper.
pub enum DiamondIOFuncType {
    /// Debug circuit that decrypts private seed ciphertexts and returns those
    /// seed bits followed by the public input bits unchanged.
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
}

/// Diamond iO frontend that turns a high-level function type into Diamond input
/// injection preprocessing artifacts.
///
/// The struct owns only reusable parameters and evaluators. Per-obfuscation
/// disk state is supplied through `Obfuscation::obfuscation` and
/// `Obfuscation::eval` as an explicit `dir_path`.
pub struct DiamondIO<M, US, HS, TS, PKPE = NoCircuitEvaluator, PKST = NoCircuitEvaluator>
where
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
    /// Public-key lookup evaluator used when evaluating the selected circuit
    /// over `NaiveBGGPublicKeyVec` wires.
    pub pk_lookup_evaluator: Option<PKPE>,
    /// Public-key slot-transfer evaluator used when the selected circuit
    /// contains slot movement gates.
    pub pk_slot_transfer_evaluator: Option<PKST>,
    _m: PhantomData<M>,
}

impl<M, US, HS, TS, PKPE, PKST> DiamondIO<M, US, HS, TS, PKPE, PKST>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    #[allow(clippy::too_many_arguments)]
    /// Build a reusable Diamond iO frontend from explicit cryptographic
    /// parameters and circuit evaluators.
    ///
    /// `injector` owns the Diamond input-injection parameters, `input_size` and
    /// `output_size` describe the selected boolean circuit shape,
    /// `native_poly_params` and `ring_gsw_context` describe the native
    /// Ring-GSW layer, `bgg_tag` is the domain-separation prefix for BGG public
    /// keys, and the optional evaluators are used when evaluating the function
    /// circuit over public keys.
    pub fn new(
        injector: DiamondInjector<M, US, HS, TS>,
        input_size: usize,
        output_size: usize,
        native_poly_params: DCRTPolyParams,
        ring_gsw_context: Arc<NestedRnsPolyContext>,
        ring_gsw_width: usize,
        ring_gsw_level_offset: usize,
        ring_gsw_enable_levels: Option<usize>,
        ring_gsw_public_key_error_sigma: Option<f64>,
        bgg_tag: Vec<u8>,
        seed_bits: usize,
        pk_lookup_evaluator: Option<PKPE>,
        pk_slot_transfer_evaluator: Option<PKST>,
    ) -> Self {
        Self {
            injector,
            input_size,
            output_size,
            native_poly_params,
            ring_gsw_context,
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            ring_gsw_public_key_error_sigma,
            bgg_tag,
            seed_bits,
            pk_lookup_evaluator,
            pk_slot_transfer_evaluator,
            _m: PhantomData,
        }
    }

    /// Sample the scalar BGG public keys for one, k, and every input bit with a
    /// single hash-sampler call.
    fn sample_bgg_public_keys(
        &self,
        params: &<M::P as Poly>::Params,
        bgg_hash_key: [u8; 32],
    ) -> (BggPublicKey<M>, BggPublicKey<M>, Vec<BggPublicKey<M>>) {
        let mut bgg_public_key_tag = self.bgg_tag.clone();
        bgg_public_key_tag.extend_from_slice(b":public_keys");
        let mut public_keys = BGGPublicKeySampler::<[u8; 32], HS>::new(bgg_hash_key, 1).sample(
            params,
            &bgg_public_key_tag,
            &vec![true; self.input_size + 1],
        );
        assert_eq!(
            public_keys.len(),
            self.input_size + 2,
            "BGG public-key sampler must return one, k, and all input-bit public keys"
        );
        let input_digits = public_keys.split_off(2);
        let k_pubkey =
            public_keys.pop().expect("BGG public-key sampler must return a k public key");
        let one = public_keys.pop().expect("BGG public-key sampler must return a one public key");
        (one, k_pubkey, input_digits)
    }

    /// Expand a scalar BGG public key into one identical key per ring slot.
    fn duplicate_public_key(
        params: &<M::P as Poly>::Params,
        public_key: &BggPublicKey<M>,
        num_slots: usize,
    ) -> NaiveBGGPublicKeyVec<M> {
        NaiveBGGPublicKeyVec::new(params, vec![public_key.clone(); num_slots])
    }

    /// Build the debug circuit that decrypts private seed ciphertext inputs and
    /// appends the explicit public input bits to the output.
    fn build_debug_decryption_circuit(&self) -> PolyCircuit<M::P>
    where
        M::P: 'static,
    {
        let mut circuit = PolyCircuit::new();
        let nested_rns_context = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &self.injector.params,
            self.ring_gsw_context.p_moduli_bits,
            self.ring_gsw_context.max_unreduced_muls,
            self.ring_gsw_context.scale,
            false,
            self.ring_gsw_enable_levels,
        ));
        let ring_gsw_context = Arc::new(NestedRnsRingGswContext::<M::P>::from_arith_context(
            &mut circuit,
            &self.injector.params,
            self.injector.params.ring_dimension() as usize,
            nested_rns_context,
            self.ring_gsw_enable_levels,
            Some(self.ring_gsw_level_offset),
        ));
        let decryption_key = circuit.input(1).at(0).as_single_wire();
        let mut outputs = Vec::with_capacity(self.output_size);
        for _ in 0..self.seed_bits {
            let ciphertext = RingGswCiphertext::input(
                ring_gsw_context.clone(),
                Some(BigUint::from(1u64)),
                &mut circuit,
            );
            outputs.push(ciphertext.decrypt::<M>(
                decryption_key,
                BigUint::from(2u64),
                &mut circuit,
            ));
        }
        outputs.extend(circuit.input(self.input_size).to_vec());
        circuit.output(outputs);
        circuit
    }

    fn io_matrix_path(dir_path: &Path, id: &str) -> std::path::PathBuf {
        dir_path.join(format!("diamond_io_{id}.matrixbin"))
    }

    fn io_bytes_path(dir_path: &Path, id: &str) -> std::path::PathBuf {
        dir_path.join(format!("diamond_io_{id}.bytesbin"))
    }

    /// Persist a DiamondIO-owned matrix artifact next to the injector
    /// preprocessing directory.
    fn write_io_matrix(dir_path: &Path, id: &str, matrix: &M) {
        fs::write(Self::io_matrix_path(dir_path, id), matrix.to_compact_bytes())
            .unwrap_or_else(|err| panic!("DiamondIO failed to write matrix {id}: {err}"));
    }

    /// Load a DiamondIO-owned matrix artifact produced by `obfuscation`.
    fn read_io_matrix(&self, dir_path: &Path, id: &str) -> M {
        let bytes = fs::read(Self::io_matrix_path(dir_path, id))
            .unwrap_or_else(|err| panic!("DiamondIO failed to read matrix {id}: {err}"));
        M::from_compact_bytes(&self.injector.params, &bytes)
    }

    /// Persist a flattened decoder public key, including whether plaintext
    /// metadata should be revealed when building BGG encodings from it.
    fn write_io_public_key(dir_path: &Path, id: &str, public_key: &BggPublicKey<M>) {
        Self::write_io_matrix(dir_path, id, &public_key.matrix);
        fs::write(
            Self::io_bytes_path(dir_path, &format!("{id}_reveal")),
            [public_key.reveal_plaintext as u8],
        )
        .unwrap_or_else(|err| panic!("DiamondIO failed to write public-key metadata {id}: {err}"));
    }

    /// Load a flattened decoder public key produced during `obfuscation`.
    fn read_io_public_key(&self, dir_path: &Path, id: &str) -> BggPublicKey<M> {
        let reveal_bytes = fs::read(Self::io_bytes_path(dir_path, &format!("{id}_reveal")))
            .unwrap_or_else(|err| {
                panic!("DiamondIO failed to read public-key metadata {id}: {err}")
            });
        assert_eq!(
            reveal_bytes.len(),
            1,
            "DiamondIO public-key metadata must contain exactly one reveal byte"
        );
        BggPublicKey::new(self.read_io_matrix(dir_path, id), reveal_bytes[0] != 0)
    }

    fn one_preimage_id() -> &'static str {
        "one_preimage"
    }

    fn k_preimage_id() -> &'static str {
        "k_preimage"
    }

    fn input_preimage_id(bit_idx: usize) -> String {
        format!("input_preimage_{bit_idx}")
    }

    fn decoder_preimage_id(decoder_idx: usize) -> String {
        format!("decoder_preimage_{decoder_idx}")
    }

    fn decoder_public_key_id(decoder_idx: usize) -> String {
        format!("decoder_public_key_{decoder_idx}")
    }

    fn final_gadget_col_size(params: &<M::P as Poly>::Params) -> usize {
        DIAMOND_SECRET_SIZE
            .checked_mul(params.modulus_digits())
            .expect("DiamondIO final gadget column count overflow")
    }

    fn final_state_row_size() -> usize {
        2usize.checked_mul(DIAMOND_SECRET_SIZE).expect("DiamondIO final state row count overflow")
    }

    fn sample_final_w_block(
        &self,
        params: &<M::P as Poly>::Params,
        hash_key: [u8; 32],
        block_idx: usize,
    ) -> M {
        HS::new().sample_hash(
            params,
            hash_key,
            format!("diamond_w_{}_{}", block_idx, self.injector.input_count),
            Self::final_state_row_size(),
            Self::final_gadget_col_size(params),
            DistType::FinRingDist,
        )
    }

    /// Sample a final projection preimage from the injector's final trapdoor
    /// basis to a two-row target. The top row consumes the `s` component and
    /// the bottom row consumes the fixed `k` or bit-carrier component.
    fn sample_final_output_preimage(
        &self,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        block_idx: usize,
        pubkey: &BggPublicKey<M>,
        top_gadget_plaintext: Option<&M::P>,
        bottom_gadget_plaintext: Option<&M::P>,
    ) -> M {
        let params = &self.injector.params;
        let gadget = M::gadget_matrix(params, DIAMOND_SECRET_SIZE);
        let mut top = pubkey.matrix.clone();
        if let Some(plaintext) = top_gadget_plaintext {
            top = top - &(gadget.clone() * plaintext);
        }
        let mut bottom = M::zero(params, DIAMOND_SECRET_SIZE, Self::final_gadget_col_size(params));
        if let Some(plaintext) = bottom_gadget_plaintext {
            bottom = bottom - &(gadget * plaintext);
        }
        let target = top.concat_rows(&[&bottom]);
        let ext_matrix = self.sample_final_w_block(params, preprocess_out.hash_key, block_idx);
        TS::new(params, self.injector.trapdoor_sigma).preimage_extend(
            params,
            &preprocess_out.final_trapdoor,
            &preprocess_out.final_pub_matrix,
            &ext_matrix,
            &target,
        )
    }
}

impl<M, US, HS, TS, PKPE, PKST> Obfuscation for DiamondIO<M, US, HS, TS, PKPE, PKST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    PKPE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
    PKST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
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

        // Function-specific work is restricted to selecting and validating the
        // circuit. Key sampling, seed encryption, circuit evaluation, and
        // Diamond preprocessing below are shared across function types.
        let circuit = match func {
            DiamondIOFuncType::DebugDecryption => {
                assert_eq!(
                    self.output_size,
                    self.seed_bits + self.input_size,
                    "DebugDecryption output_size must be seed_bits + input_size"
                );
                self.build_debug_decryption_circuit()
            }
        };

        // Sample the ternary Ring-GSW decryption key `k` and the BGG hash key
        // used to derive all scalar public keys for this obfuscation.
        let uniform_sampler = US::new();
        let k = uniform_sampler.sample_poly(params, &DistType::TernaryDist);
        let native_k = DCRTPoly::from_biguints(&self.native_poly_params, &k.coeffs_biguints());
        let bgg_hash_key = rand::random::<[u8; 32]>();

        // Sample scalar BGG public keys for one, k, and every input bit. The
        // Diamond injector consumes these scalar keys directly.
        let (one, k_pubkey, input_digits) = self.sample_bgg_public_keys(params, bgg_hash_key);

        // Duplicate scalar keys across all ring slots so the same public keys
        // can be used as `NaiveBGGPublicKeyVec` circuit inputs.
        let one_vec = Self::duplicate_public_key(params, &one, ring_dim);
        let k_vec = Self::duplicate_public_key(params, &k_pubkey, ring_dim);
        let input_digit_vecs = input_digits
            .iter()
            .map(|key| Self::duplicate_public_key(params, key, ring_dim))
            .collect::<Vec<_>>();

        // Sample the native Ring-GSW public key for `k`.
        let ring_gsw_public_key = sample_public_key(
            &self.native_poly_params,
            self.ring_gsw_width,
            &native_k,
            bgg_hash_key,
            b"diamond_io_ring_gsw_public_key",
            self.ring_gsw_public_key_error_sigma,
        );

        // Encrypt each private seed bit natively, convert the ciphertext to
        // nested-RNS plaintext inputs, and lift those inputs into public-key
        // wires by slot-wise multiplying the constant-one BGG key.
        let mut seed_ciphertexts = Vec::with_capacity(self.seed_bits);
        let mut enc_seed_public_keys = Vec::new();
        for _ in 0..self.seed_bits {
            let seed_bit = rand::random::<bool>();
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
        }

        // Evaluate the selected function circuit over public keys. The inputs
        // are ordered as the circuit expects: decryption key, seed ciphertext
        // wires, then explicit public input bits.
        let mut function_inputs =
            Vec::with_capacity(1 + enc_seed_public_keys.len() + input_digit_vecs.len());
        function_inputs.push(k_vec);
        function_inputs.extend(enc_seed_public_keys);
        function_inputs.extend(input_digit_vecs);
        let pk_slot_transfer_evaluator = self
            .pk_slot_transfer_evaluator
            .as_ref()
            .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>);
        let evaluated_public_keys = circuit.eval(
            params,
            one_vec,
            function_inputs,
            self.pk_lookup_evaluator.as_ref(),
            pk_slot_transfer_evaluator,
            None,
        );
        let decoders =
            evaluated_public_keys.iter().flat_map(NaiveBGGPublicKeyVec::keys).collect::<Vec<_>>();

        // Diamond preprocessing now produces only the final state trapdoor
        // basis. The output projection preimages are sampled here because they
        // depend on the selected function's public decoder keys.
        let preprocess_out = self.injector.preprocess(dir_path, &k);
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
        let input_preimages = input_digits
            .iter()
            .enumerate()
            .map(|(bit_idx, pubkey)| {
                let digit_idx = bit_idx / self.injector.batch_bits();
                let bit_in_digit = bit_idx % self.injector.batch_bits();
                let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
                self.sample_final_output_preimage(
                    &preprocess_out,
                    state_idx,
                    pubkey,
                    None,
                    Some(&one_plaintext),
                )
            })
            .collect::<Vec<_>>();
        for (bit_idx, preimage) in input_preimages.iter().enumerate() {
            Self::write_io_matrix(dir_path, &Self::input_preimage_id(bit_idx), preimage);
        }
        let decoder_preimages = decoders
            .iter()
            .enumerate()
            .map(|(decoder_idx, decoder)| {
                Self::write_io_public_key(
                    dir_path,
                    &Self::decoder_public_key_id(decoder_idx),
                    decoder,
                );
                self.sample_final_output_preimage(&preprocess_out, 0, decoder, None, None)
            })
            .collect::<Vec<_>>();
        for (decoder_idx, preimage) in decoder_preimages.iter().enumerate() {
            Self::write_io_matrix(dir_path, &Self::decoder_preimage_id(decoder_idx), preimage);
        }

        DiamondIOObf {
            preprocess_out,
            bgg_hash_key,
            func_type: func,
            ring_gsw_public_key,
            seed_ciphertexts,
        }
    }

    /// Evaluate an obfuscation on boolean input bits.
    ///
    /// This rebuilds the same public-key circuit inputs used during
    /// obfuscation, asks `DiamondInjector` for the input-specific BGG
    /// encodings, evaluates the function circuit over those encodings, and
    /// cancels each evaluated output against the matching decoder encoding.
    fn eval(&self, dir_path: &Path, obf: &Self::Obf, input: Self::Input) -> Self::Output {
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

        let circuit = match obf.func_type {
            DiamondIOFuncType::DebugDecryption => self.build_debug_decryption_circuit(),
        };
        let (one, k_pubkey, input_digit_pubkeys) =
            self.sample_bgg_public_keys(params, obf.bgg_hash_key);
        let decoder_count =
            self.output_size.checked_mul(ring_dim).expect("DiamondIO decoder count overflow");
        let decoders = (0..decoder_count)
            .map(|decoder_idx| {
                self.read_io_public_key(dir_path, &Self::decoder_public_key_id(decoder_idx))
            })
            .collect::<Vec<_>>();
        let input_preimages = (0..input_digit_pubkeys.len())
            .map(|bit_idx| self.read_io_matrix(dir_path, &Self::input_preimage_id(bit_idx)))
            .collect::<Vec<_>>();
        let decoder_preimages = (0..decoder_count)
            .map(|decoder_idx| {
                self.read_io_matrix(dir_path, &Self::decoder_preimage_id(decoder_idx))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            decoder_preimages.len(),
            decoders.len(),
            "DiamondIO decoder projection preimage count mismatch"
        );
        let states = self.injector.online_eval(dir_path, &obf.preprocess_out, &input_digits);
        assert_eq!(
            states.len(),
            1 + self.input_size,
            "DiamondIO final Diamond state count mismatch"
        );
        let one_output = self.injector.build_output_encoding(
            states[0].clone() * &self.read_io_matrix(dir_path, Self::one_preimage_id()),
            one,
            Some(M::P::const_one(params)),
        );
        let k_output = self.injector.build_output_encoding(
            states[0].clone() * &self.read_io_matrix(dir_path, Self::k_preimage_id()),
            k_pubkey,
            Some(self.injector.read_preprocessed_k(dir_path)),
        );
        let digit_outputs = input_digit_pubkeys
            .into_iter()
            .enumerate()
            .map(|(bit_idx, pubkey)| {
                let digit_idx = bit_idx / batch_bits;
                let bit_in_digit = bit_idx % batch_bits;
                let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
                let plaintext = M::P::from_usize_to_constant(
                    params,
                    self.injector.digit_bit_value(input_digits[digit_idx] as usize, bit_in_digit),
                );
                self.injector.build_output_encoding(
                    states[state_idx].clone() * &input_preimages[bit_idx],
                    pubkey,
                    Some(plaintext),
                )
            })
            .collect::<Vec<_>>();
        let decoder_outputs = decoders
            .into_iter()
            .zip(decoder_preimages.iter())
            .map(|(decoder, preimage)| {
                self.injector.build_output_encoding(
                    states[0].clone() * preimage,
                    decoder,
                    Some(M::P::const_zero(params)),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            decoder_outputs.len(),
            self.output_size
                .checked_mul(ring_dim)
                .expect("DiamondIO decoder output count overflow"),
            "DiamondIO decoder output count mismatch"
        );

        let one_encoding_vec = NaiveBGGEncodingVec::new(params, vec![one_output.clone(); ring_dim]);
        let k_encoding_vec = NaiveBGGEncodingVec::new(params, vec![k_output; ring_dim]);
        let seed_encoding_inputs = obf
            .seed_ciphertexts
            .iter()
            .flat_map(|ciphertext| {
                ciphertext_inputs_from_native::<M::P>(
                    params,
                    self.ring_gsw_context.as_ref(),
                    ciphertext,
                    self.ring_gsw_level_offset,
                    self.ring_gsw_enable_levels,
                )
            })
            .map(|input| {
                assert_eq!(
                    one_encoding_vec.num_slots(),
                    input.len(),
                    "DiamondIO slot-wise scalar multiplication requires matching slot counts"
                );
                NaiveBGGEncodingVec::new(
                    params,
                    (0..one_encoding_vec.num_slots())
                        .map(|slot_idx| {
                            one_encoding_vec.encoding(slot_idx).large_scalar_mul(
                                params,
                                &input.as_slice()[slot_idx].coeffs_biguints(),
                            )
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let digit_encoding_inputs = digit_outputs
            .iter()
            .map(|encoding| NaiveBGGEncodingVec::new(params, vec![encoding.clone(); ring_dim]))
            .collect::<Vec<_>>();

        let mut encoding_function_inputs =
            Vec::with_capacity(1 + seed_encoding_inputs.len() + digit_encoding_inputs.len());
        encoding_function_inputs.push(k_encoding_vec);
        encoding_function_inputs.extend(seed_encoding_inputs);
        encoding_function_inputs.extend(digit_encoding_inputs);
        let evaluated_encodings = circuit.eval(
            params,
            one_encoding_vec,
            encoding_function_inputs,
            None::<&NoCircuitEvaluator>,
            None::<&dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>>,
            None,
        );
        assert_eq!(
            evaluated_encodings.len(),
            self.output_size,
            "DiamondIO evaluated encoding count mismatch"
        );

        let q: Arc<BigUint> = params.modulus().into();
        let quarter_q = q.as_ref() / 4u32;
        let three_quarter_q = (&quarter_q) * 3u32;
        let one_coeff = BigUint::from(1u64);
        let zero = M::P::const_zero(params);
        evaluated_encodings
            .iter()
            .enumerate()
            .map(|(output_idx, evaluated_vec)| {
                let evaluated = evaluated_vec.encoding(0);
                let decoder = decoder_outputs[output_idx * ring_dim].clone();
                let zero_decoder =
                    BggEncoding::new(decoder.vector, decoder.pubkey, Some(zero.clone()));
                let canceled = evaluated - &zero_decoder;
                let plaintext = canceled
                    .plaintext
                    .expect("DiamondIO canceled output encoding must retain plaintext metadata");
                plaintext
                    .coeffs_biguints()
                    .into_iter()
                    .next()
                    .map(|coeff| {
                        coeff == one_coeff || (coeff > quarter_q && coeff < three_quarter_q)
                    })
                    .expect("DiamondIO output plaintext polynomial must have one coefficient")
            })
            .collect()
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
        gadgets::arith::{ModularArithmeticContext, NestedRnsPolyContext},
        input_injector::DiamondInjector,
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::{
            PolyParams,
            dcrt::{
                gpu::{GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::{
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
    };

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
        NoCircuitEvaluator,
        NoCircuitEvaluator,
    >;

    #[sequential_test::sequential]
    #[test]
    fn test_gpu_diamond_io_debug_decryption_eval_returns_input_suffix() {
        let gpu_ids = detected_gpu_device_ids();
        assert!(!gpu_ids.is_empty(), "DiamondIO GPU test requires at least one GPU");
        gpu_device_sync();

        let native_poly_params = DCRTPolyParams::new(2, 1, 10, 5);
        let (moduli, _, _) = native_poly_params.to_crt();
        let poly_params = GpuDCRTPolyParams::new_with_gpu(
            native_poly_params.ring_dimension(),
            moduli,
            native_poly_params.base_bits(),
            vec![gpu_ids[0]],
            Some(1),
        );

        let active_levels = 1usize;
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

        let input_count = 1usize;
        let input_base = 2usize;
        let input_size = 1usize;
        let seed_bits = 0usize;
        let output_size = seed_bits + input_size;
        let injector = TestInjector::new(poly_params.clone(), input_count, input_base, 4.578, 0.0)
            .with_gpu_device_ids(vec![gpu_ids[0]]);
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
            None,
            None,
        );
        let dir = tempdir().expect("DiamondIO GPU test must create a tempdir");
        let obf = scheme.obfuscation(dir.path(), DiamondIOFuncType::DebugDecryption);

        let input = vec![true];
        let output = scheme.eval(dir.path(), &obf, input.clone());

        assert_eq!(output.len(), output_size);
        assert_eq!(&output[seed_bits..], input.as_slice());
    }
}
