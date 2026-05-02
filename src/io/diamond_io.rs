use std::{marker::PhantomData, path::Path, sync::Arc};

use num_bigint::BigUint;

use crate::{
    bgg::{
        naive_vec::NaiveBGGPublicKeyVec, public_key::BggPublicKey, sampler::BGGPublicKeySampler,
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
    input_injector::{DiamondInjector, InputInjector},
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
pub struct DiamondIOObf {
    /// Hash key sampled by `DiamondInjector::preprocess` and required by
    /// `DiamondInjector::online_eval` to identify the persisted preimages.
    pub preprocess_out: [u8; 32],
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
    type Obf = DiamondIOObf;
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

        // Flattened output public keys become the Diamond decoder targets.
        let preprocess_out =
            self.injector.preprocess(dir_path, &one, &k_pubkey, &input_digits, &decoders, &k);

        DiamondIOObf {
            preprocess_out,
            bgg_hash_key,
            func_type: func,
            ring_gsw_public_key,
            seed_ciphertexts,
        }
    }

    /// Online evaluation for Diamond iO.
    ///
    /// This is intentionally left unimplemented until the evaluation path is
    /// specified. The `dir_path` argument is reserved for reading the same
    /// Diamond preprocessing artifacts written by `obfuscation`.
    fn eval(&self, _dir_path: &Path, _obf: &Self::Obf, _input: Self::Input) -> Self::Output {
        unimplemented!("DiamondIO::eval is not implemented yet")
    }
}
