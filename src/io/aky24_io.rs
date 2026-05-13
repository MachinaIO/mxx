use std::{marker::PhantomData, sync::Arc};

use crate::{
    func_enc::NoCircuitEvaluator,
    gadgets::arith::NestedRnsPolyContext,
    matrix::PolyMatrix,
    poly::{Poly, dcrt::params::DCRTPolyParams},
};

pub mod bench_estimator;
pub mod simulation;

pub use bench_estimator::{Aky24IOBenchEstimate, Aky24IOBenchEstimator};
pub use simulation::{
    Aky24IOCrtDepthSearchResult, Aky24IOErrorSimulation, Aky24IOPrfMaskOutputCoeffBitsSearchResult,
    Aky24IOPrfRoundErrorSimulation, aky24_io_find_crt_depth,
    aky24_io_max_noise_refresh_v_bits_without_pre_rounding_error, minimum_aky24_io_prf_seed_bits,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Function families supported by the conventional AKY24 FE-to-iO wrapper.
pub enum Aky24IOFuncType {
    /// Debug PRF circuit matching the maintained DiamondIO benchmark and
    /// simulation shape: expand the final refreshed private seed with a
    /// Goldreich PRG, decrypt `output_bits` Ring-GSW ciphertext outputs, and
    /// decode them as boolean function outputs.
    GoldreichPRF { output_bits: usize },
}

impl Aky24IOFuncType {
    pub(crate) fn output_bits(self) -> usize {
        match self {
            Self::GoldreichPRF { output_bits } => output_bits,
        }
    }
}

/// Conventional AKY24 FE-to-iO parameter carrier for maintained IO-level
/// simulation and benchmark estimators.
///
/// This type intentionally does not depend on `func_enc::aky24`. It keeps only
/// the parameters needed by the DiamondIO-derived PRF, noise-refresh, final-mask,
/// and final decode models, while replacing Diamond input-injection state with a
/// fresh Gaussian encoding error in `simulation`.
pub struct Aky24IO<
    M,
    PKPE = NoCircuitEvaluator,
    PKST = NoCircuitEvaluator,
    ENCPE = NoCircuitEvaluator,
    ENCST = NoCircuitEvaluator,
> where
    M: PolyMatrix,
{
    /// Polynomial parameters for BGG encodings and IO-level circuits.
    pub params: <M::P as Poly>::Params,
    /// Native DCRT parameters used by the Ring-GSW ciphertext layer.
    pub native_poly_params: DCRTPolyParams,
    /// Nested-RNS arithmetic context used by Ring-GSW ciphertext conversion.
    pub ring_gsw_context: Arc<NestedRnsPolyContext>,
    /// Number of native Ring-GSW public-key columns.
    pub ring_gsw_width: usize,
    /// Level offset used when converting native Ring-GSW ciphertext entries.
    pub ring_gsw_level_offset: usize,
    /// Optional number of nested-RNS levels enabled during conversion.
    pub ring_gsw_enable_levels: Option<usize>,
    /// Optional Gaussian error used when sampling the native Ring-GSW public key.
    pub ring_gsw_public_key_error_sigma: Option<f64>,
    /// Domain-separation prefix for BGG public-key sampling tags.
    pub bgg_tag: Vec<u8>,
    /// Number of boolean input bits accepted by the selected iO function.
    pub input_size: usize,
    /// Number of boolean output bits produced by the selected iO function.
    pub output_size: usize,
    /// Number of private PRF seed bits encrypted into Ring-GSW ciphertexts.
    pub seed_bits: usize,
    /// Number of public PRF seed bits, hence PRF seed-refresh rounds.
    pub public_prf_seed_bits: usize,
    /// Number of bit-decomposed PRF mask output coefficients to compute.
    pub prf_mask_output_coeff_bits: usize,
    /// Number of low bits retained in the noise-refresh rounding material.
    pub noise_refresh_v_bits: usize,
    /// Centered-binomial sample count used by noise-refresh material PRGs.
    pub noise_refresh_cbd_n: usize,
    /// Hash key used by the noise-refresh public material sampler.
    pub noise_refresh_hash_key: [u8; 32],
    /// Base seed from which per-round Goldreich PRG graphs are derived.
    pub goldreich_graph_seed: [u8; 32],
    /// Public-key lookup evaluator used by benchmark callers.
    pub pk_lookup_evaluator: Option<PKPE>,
    /// Public-key slot-transfer evaluator used by benchmark callers.
    pub pk_slot_transfer_evaluator: Option<PKST>,
    /// Encoding lookup evaluator used by benchmark callers.
    pub enc_lookup_evaluator: Option<ENCPE>,
    /// Encoding slot-transfer evaluator used by benchmark callers.
    pub enc_slot_transfer_evaluator: Option<ENCST>,
    _m: PhantomData<(M, ENCPE)>,
}

impl<M, PKPE, PKST, ENCPE, ENCST> Aky24IO<M, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: <M::P as Poly>::Params,
        native_poly_params: DCRTPolyParams,
        ring_gsw_context: Arc<NestedRnsPolyContext>,
        ring_gsw_width: usize,
        ring_gsw_level_offset: usize,
        ring_gsw_enable_levels: Option<usize>,
        ring_gsw_public_key_error_sigma: Option<f64>,
        bgg_tag: Vec<u8>,
        input_size: usize,
        output_size: usize,
        seed_bits: usize,
        public_prf_seed_bits: usize,
        prf_mask_output_coeff_bits: usize,
        noise_refresh_v_bits: usize,
        noise_refresh_cbd_n: usize,
        noise_refresh_hash_key: [u8; 32],
        goldreich_graph_seed: [u8; 32],
        pk_lookup_evaluator: Option<PKPE>,
        pk_slot_transfer_evaluator: Option<PKST>,
        enc_lookup_evaluator: Option<ENCPE>,
        enc_slot_transfer_evaluator: Option<ENCST>,
    ) -> Self {
        assert!(input_size > 0, "AKY24IO input_size must be positive");
        assert!(output_size > 0, "AKY24IO output_size must be positive");
        assert!(seed_bits > 0, "AKY24IO seed_bits must be positive");
        assert!(public_prf_seed_bits > 0, "AKY24IO public_prf_seed_bits must be positive");
        assert!(
            prf_mask_output_coeff_bits > 0,
            "AKY24IO prf_mask_output_coeff_bits must be positive"
        );
        assert!(noise_refresh_v_bits > 0, "AKY24IO noise_refresh_v_bits must be positive");
        assert!(noise_refresh_cbd_n > 0, "AKY24IO noise_refresh_cbd_n must be positive");
        Self {
            params,
            native_poly_params,
            ring_gsw_context,
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            ring_gsw_public_key_error_sigma,
            bgg_tag,
            input_size,
            output_size,
            seed_bits,
            public_prf_seed_bits,
            prf_mask_output_coeff_bits,
            noise_refresh_v_bits,
            noise_refresh_cbd_n,
            noise_refresh_hash_key,
            goldreich_graph_seed,
            pk_lookup_evaluator,
            pk_slot_transfer_evaluator,
            enc_lookup_evaluator,
            enc_slot_transfer_evaluator,
            _m: PhantomData,
        }
    }

    pub(crate) fn prf_final_round_idx(&self) -> usize {
        self.public_prf_seed_bits
    }
}
