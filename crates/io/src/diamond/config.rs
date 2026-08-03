use mxx_gadgets::{
    circuit_gadgets::fhe_prg::goldreich::{
        goldreich_output_bound_holds, minimum_goldreich_input_size,
    },
    input_injector::{DiamondInputConfig, DiamondInputConfigError},
    noise_refresh::circuit_prg::goldreich_noise_refresh_uniform_output_bits,
};
use mxx_ir_core::{IntExpr, ParamEnv, RealExpr};
use num_bigint::BigInt;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiamondIoFunction {
    /// Evaluate an additional Goldreich suffix from the final refreshed,
    /// privately sampled PRF seed and return the decrypted suffix bits.
    GoldreichPrf { output_bits: usize },
}

impl DiamondIoFunction {
    pub fn output_bits(&self) -> usize {
        match self {
            Self::GoldreichPrf { output_bits } => *output_bits,
        }
    }
}

/// Parameters shared by the Diamond iO preprocessing and evaluation graphs.
///
/// Native DCRT parameters and the nested-RNS arithmetic context are runtime
/// backend objects and intentionally do not live in this serializable graph
/// configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondIoConfig {
    pub modulus: BigInt,
    pub ring_dimension: usize,
    pub input_count: usize,
    pub digit_base: usize,
    pub batch_bits: usize,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub trapdoor_sigma: RealExpr,
    pub error_sigma: RealExpr,
    pub bgg_tag: Vec<u8>,
    pub seed_bits: usize,
    pub prf_mask_output_coeff_bits: usize,
    pub noise_refresh_v_bits: usize,
    pub noise_refresh_cbd_n: usize,
    pub noise_refresh_hash_key: [u8; 32],
    pub goldreich_graph_seed: [u8; 32],
    pub ring_gsw_width: usize,
    pub ring_gsw_public_key_error_sigma: Option<RealExpr>,
    pub refresh_crt_scale_factors: Vec<IntExpr>,
    pub refresh_crt_plaintext_moduli: Vec<IntExpr>,
    pub refresh_reconstruction_coefficients: Vec<IntExpr>,
    pub refresh_decoder_public_columns: usize,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DiamondIoConfigError {
    #[error(transparent)]
    Input(#[from] DiamondInputConfigError),
    #[error("Diamond iO requires a nonempty Goldreich suffix")]
    EmptyOutput,
    #[error("Diamond iO requires at least five private PRF seed bits")]
    SeedTooShort,
    #[error("Diamond iO mask and noise-refresh dimensions must be positive")]
    EmptyRefreshLayout,
    #[error("Diamond iO refresh CRT parameter vectors must have the same nonzero length")]
    RefreshCrtLayout,
    #[error("Diamond iO Ring-GSW dimensions are inconsistent")]
    RingGswLayout,
    #[error("Diamond iO requires a strictly positive native Ring-GSW public-key error sigma")]
    NativeRingGswNoiseRequired,
    #[error("Diamond iO requires a strictly positive BGG encryption error sigma")]
    ProtocolNoiseRequired,
    #[error("Diamond iO requires a finite, evaluable, and strictly positive trapdoor sigma")]
    TrapdoorSigmaRequired,
    #[error("Diamond iO digit_base must equal 2^batch_bits")]
    DigitBranchLayout,
    #[error(
        "Diamond iO seed has {seed_bits} bits, but its largest Goldreich stream requires at least {minimum_seed_bits}"
    )]
    GoldreichOutputBound { seed_bits: usize, minimum_seed_bits: usize },
}

impl DiamondIoConfig {
    pub fn validate(&self, function: &DiamondIoFunction) -> Result<(), DiamondIoConfigError> {
        self.input_config().validate()?;
        if self.digit_base != self.branch_count()? {
            return Err(DiamondIoConfigError::DigitBranchLayout);
        }
        if function.output_bits() == 0 {
            return Err(DiamondIoConfigError::EmptyOutput);
        }
        if self.seed_bits < 5 {
            return Err(DiamondIoConfigError::SeedTooShort);
        }
        if self.prf_mask_output_coeff_bits == 0 ||
            self.noise_refresh_v_bits == 0 ||
            self.noise_refresh_cbd_n == 0 ||
            self.refresh_decoder_public_columns == 0
        {
            return Err(DiamondIoConfigError::EmptyRefreshLayout);
        }
        let depth = self.refresh_crt_plaintext_moduli.len();
        if depth == 0 ||
            self.refresh_crt_scale_factors.len() != depth ||
            self.refresh_reconstruction_coefficients.len() != depth
        {
            return Err(DiamondIoConfigError::RefreshCrtLayout);
        }
        if self.ring_gsw_width == 0 {
            return Err(DiamondIoConfigError::RingGswLayout);
        }
        let native_sigma = self
            .ring_gsw_public_key_error_sigma
            .as_ref()
            .ok_or(DiamondIoConfigError::NativeRingGswNoiseRequired)?
            .evaluate_f64(&ParamEnv::default())
            .map_err(|_| DiamondIoConfigError::NativeRingGswNoiseRequired)?;
        if !native_sigma.is_finite() || native_sigma <= 0.0 {
            return Err(DiamondIoConfigError::NativeRingGswNoiseRequired);
        }
        let protocol_sigma = self
            .error_sigma
            .evaluate_f64(&ParamEnv::default())
            .map_err(|_| DiamondIoConfigError::ProtocolNoiseRequired)?;
        if !protocol_sigma.is_finite() || protocol_sigma <= 0.0 {
            return Err(DiamondIoConfigError::ProtocolNoiseRequired);
        }
        let trapdoor_sigma = self
            .trapdoor_sigma
            .evaluate_f64(&ParamEnv::default())
            .map_err(|_| DiamondIoConfigError::TrapdoorSigmaRequired)?;
        if !trapdoor_sigma.is_finite() || trapdoor_sigma <= 0.0 {
            return Err(DiamondIoConfigError::TrapdoorSigmaRequired);
        }
        let minimum_seed_bits = self.minimum_goldreich_seed_bits(function)?;
        if self.seed_bits < minimum_seed_bits {
            return Err(DiamondIoConfigError::GoldreichOutputBound {
                seed_bits: self.seed_bits,
                minimum_seed_bits,
            });
        }
        Ok(())
    }

    /// Smallest seed length satisfying all three Goldreich stream bounds.
    ///
    /// The branch stream has `branch_count * seed_bits` outputs, so its bound
    /// cannot be obtained by applying `minimum_goldreich_input_size` once.
    /// This exact integer binary search solves that fixed-point condition.
    pub fn minimum_goldreich_seed_bits(
        &self,
        function: &DiamondIoFunction,
    ) -> Result<usize, DiamondIoConfigError> {
        let [_, refresh, final_stream] = self.goldreich_stream_sizes(function)?;
        let fixed_minimum = minimum_goldreich_input_size(refresh)
            .max(minimum_goldreich_input_size(final_stream))
            .max(5);
        let branch_count = self.branch_count()?;
        let branch_bound_holds = |seed_bits: usize| {
            branch_count
                .checked_mul(seed_bits)
                .is_some_and(|outputs| goldreich_output_bound_holds(seed_bits, outputs))
        };
        let mut high = fixed_minimum;
        while !branch_bound_holds(high) {
            high = high.checked_mul(2).ok_or(DiamondInputConfigError::LayoutOverflow)?;
        }
        let mut low = 5usize;
        while low < high {
            let mid = low + (high - low) / 2;
            if branch_bound_holds(mid) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        Ok(fixed_minimum.max(low))
    }

    /// Conceptual output lengths of the three independent Goldreich streams:
    /// per-round branch expansion, noise-refresh material, and the final
    /// function-plus-mask stream. Each range circuit preserves the full
    /// conceptual graph, so the safety bound applies to these totals rather
    /// than to a single material slice.
    pub fn goldreich_stream_sizes(
        &self,
        function: &DiamondIoFunction,
    ) -> Result<[usize; 3], DiamondIoConfigError> {
        let branch_seed = self
            .branch_count()?
            .checked_mul(self.seed_bits)
            .ok_or(DiamondInputConfigError::LayoutOverflow)?;
        let refresh = goldreich_noise_refresh_uniform_output_bits(
            self.ring_dimension,
            self.digit_count,
            self.refresh_crt_plaintext_moduli.len(),
            self.noise_refresh_v_bits,
            self.noise_refresh_cbd_n,
        );
        let final_per_output = self
            .ring_dimension
            .checked_mul(self.prf_mask_output_coeff_bits)
            .and_then(|mask| mask.checked_add(1))
            .ok_or(DiamondInputConfigError::LayoutOverflow)?;
        let final_stream = function
            .output_bits()
            .checked_mul(final_per_output)
            .ok_or(DiamondInputConfigError::LayoutOverflow)?;
        Ok([branch_seed, refresh, final_stream])
    }

    pub fn input_config(&self) -> DiamondInputConfig {
        DiamondInputConfig {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension,
            input_count: self.input_count,
            digit_base: self.digit_base,
            batch_bits: self.batch_bits,
            gadget_base: self.gadget_base.clone(),
            digit_count: self.digit_count,
            trapdoor_sigma: self.trapdoor_sigma.clone(),
            error_sigma: self.error_sigma.clone(),
        }
    }

    pub fn input_bits(&self) -> Result<usize, DiamondInputConfigError> {
        self.input_config().witness_size()
    }

    pub fn round_count(&self) -> usize {
        self.input_count
    }

    pub fn branch_count(&self) -> Result<usize, DiamondInputConfigError> {
        1usize.checked_shl(self.batch_bits as u32).ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn public_key_columns(&self) -> usize {
        self.digit_count
    }

    pub fn gadget_base_expr(&self) -> IntExpr {
        self.gadget_base.clone().into()
    }

    pub fn digit_count_expr(&self) -> IntExpr {
        self.digit_count.into()
    }

    /// Conservative physical nested-RNS bound for native Ring-GSW ciphertext
    /// error. The polynomial error `eR` contributes at most
    /// `6.5 * sigma * width * ring_dimension` per coefficient. Encoding a
    /// coefficient as `(c mod q_i) mod p_j` can additionally cross a `q_i`
    /// boundary, so the concrete residue difference needs `max(p_j) - 1`.
    pub fn native_ring_gsw_ciphertext_error_norm(
        &self,
        max_p_modulus: u64,
    ) -> Result<RealExpr, DiamondIoConfigError> {
        let sigma = self
            .ring_gsw_public_key_error_sigma
            .clone()
            .ok_or(DiamondIoConfigError::NativeRingGswNoiseRequired)?;
        let term_count = self
            .ring_gsw_width
            .checked_mul(self.ring_dimension)
            .and_then(|value| value.checked_mul(13))
            .ok_or(DiamondIoConfigError::RingGswLayout)?;
        let polynomial_error = RealExpr::Div(
            Box::new(RealExpr::Mul(Box::new(sigma), Box::new(RealExpr::from_integer(term_count)))),
            Box::new(RealExpr::from_integer(2)),
        );
        let residue_wrap =
            max_p_modulus.checked_sub(1).ok_or(DiamondIoConfigError::RingGswLayout)?;
        Ok(RealExpr::Add(
            Box::new(polynomial_error),
            Box::new(RealExpr::from_integer(residue_wrap as usize)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DiamondIoConfig {
        DiamondIoConfig {
            modulus: 65_537.into(),
            ring_dimension: 8,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: 4.into(),
            digit_count: 4,
            trapdoor_sigma: RealExpr::from_integer(5),
            error_sigma: RealExpr::from_integer(1),
            bgg_tag: b"diamond-io-test".to_vec(),
            seed_bits: 64,
            prf_mask_output_coeff_bits: 1,
            noise_refresh_v_bits: 1,
            noise_refresh_cbd_n: 2,
            noise_refresh_hash_key: [2; 32],
            goldreich_graph_seed: [3; 32],
            ring_gsw_width: 4,
            ring_gsw_public_key_error_sigma: Some(RealExpr::from_integer(1)),
            refresh_crt_scale_factors: vec![1.into()],
            refresh_crt_plaintext_moduli: vec![2.into()],
            refresh_reconstruction_coefficients: vec![1.into()],
            refresh_decoder_public_columns: 4,
        }
    }

    #[test]
    fn accepts_the_full_private_seed_and_refresh_layout() {
        let config = config();
        let function = DiamondIoFunction::GoldreichPrf { output_bits: 1 };
        assert_eq!(config.goldreich_stream_sizes(&function).unwrap(), [128, 160, 9]);
        assert_eq!(config.minimum_goldreich_seed_bits(&function).unwrap(), 38);
        config.validate(&function).unwrap();
    }

    #[test]
    fn rejects_unused_injector_digits_without_matching_prf_branches() {
        let mut config = config();
        config.digit_base = 4;
        assert_eq!(
            config.validate(&DiamondIoFunction::GoldreichPrf { output_bits: 1 }),
            Err(DiamondIoConfigError::DigitBranchLayout)
        );
    }

    #[test]
    fn rejects_a_seed_that_is_too_short_for_noise_refresh() {
        let mut config = config();
        config.seed_bits = 5;
        assert!(matches!(
            config.validate(&DiamondIoFunction::GoldreichPrf { output_bits: 1 }),
            Err(DiamondIoConfigError::GoldreichOutputBound { seed_bits: 5, minimum_seed_bits: 38 })
        ));
    }

    #[test]
    fn requires_positive_native_ring_gsw_noise_and_derives_its_worst_case_bound() {
        let mut config = config();
        config.ring_gsw_public_key_error_sigma = None;
        assert_eq!(
            config.validate(&DiamondIoFunction::GoldreichPrf { output_bits: 1 }),
            Err(DiamondIoConfigError::NativeRingGswNoiseRequired)
        );
        config.ring_gsw_public_key_error_sigma = Some(RealExpr::from_integer(0));
        assert_eq!(
            config.validate(&DiamondIoFunction::GoldreichPrf { output_bits: 1 }),
            Err(DiamondIoConfigError::NativeRingGswNoiseRequired)
        );
        config.ring_gsw_public_key_error_sigma = Some(RealExpr::from_integer(2));
        assert_eq!(
            config
                .native_ring_gsw_ciphertext_error_norm(17)
                .unwrap()
                .evaluate_f64(&ParamEnv::default())
                .unwrap(),
            6.5 * 2.0 * 4.0 * 8.0 + 16.0
        );
    }

    #[test]
    fn nested_residue_wrap_can_exceed_the_raw_polynomial_error() {
        let q = 13u64;
        let p = 7u64;
        let noiseless = q - 1;
        let raw_error = 2u64;
        let noisy = (noiseless + raw_error) % q;
        let residue_delta = (noiseless % p).abs_diff(noisy % p);
        assert!(residue_delta > raw_error);
        assert!(residue_delta <= raw_error + (p - 1));
    }

    #[test]
    fn trapdoor_sigma_must_be_strictly_positive_and_evaluable() {
        for invalid in [RealExpr::from_integer(0), RealExpr::from_integer(-1)] {
            let mut config = config();
            config.trapdoor_sigma = invalid;
            assert_eq!(
                config.validate(&DiamondIoFunction::GoldreichPrf { output_bits: 1 }),
                Err(DiamondIoConfigError::TrapdoorSigmaRequired)
            );
        }

        let mut config = config();
        config.trapdoor_sigma =
            RealExpr::Div(Box::new(RealExpr::from_integer(1)), Box::new(RealExpr::from_integer(0)));
        assert_eq!(
            config.validate(&DiamondIoFunction::GoldreichPrf { output_bits: 1 }),
            Err(DiamondIoConfigError::TrapdoorSigmaRequired)
        );
    }
}
