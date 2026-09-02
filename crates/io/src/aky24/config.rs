use mxx_ir_core::{ParamEnv, RealExpr};
use num_bigint::BigInt;
use thiserror::Error;

/// Goldreich-PRF function descriptor supported by AKY24 iO.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Aky24GoldreichPrf {
    pub output_bits: usize,
    pub graph_seed: [u8; 32],
}

/// Parameters shared by every private prFE layer in the AKY24 prMIFE cascade.
///
/// A uniform parameter set is used across layers. This is conservative and
/// avoids giving the same matrix shape
/// different meanings at different points in the linked graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Aky24IoConfig {
    pub modulus: BigInt,
    pub ring_dimension: usize,
    pub input_size: usize,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    /// Inclusive coefficient bound for compact preimages produced by B.1.
    pub preimage_max_coefficient_bound: BigInt,
    /// The Appendix B.1 high/low decomposition divisor `M`.
    pub modulus_split: BigInt,
    pub trapdoor_sigma: RealExpr,
    pub secret_sigma: RealExpr,
    /// Appendix B.1 error distribution for `e_B`.
    pub b_error_sigma: RealExpr,
    /// Appendix B.1 error distribution for `e_fhe`.
    pub fhe_error_sigma: RealExpr,
    /// Appendix B.1 error distribution for every `e_att` block.
    pub attribute_error_sigma: RealExpr,
    pub security_parameter_bits: usize,
    pub cascade_randomness_bits: usize,
    /// Random bits used by each bounded inverse-CDF discrete-Gaussian sample
    /// inside a prescribed-randomness cascade function.
    pub gaussian_sample_bits: usize,
    /// Extra bits used before reducing prescribed `A_bar` coefficients modulo
    /// `q`, bounding the fixed-width reduction bias.
    pub uniform_statistical_bits: usize,
    pub function: Aky24GoldreichPrf,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum Aky24ConfigError {
    #[error("AKY24 iO dimensions and security parameters must be positive")]
    NonPositiveParameter,
    #[error("all AKY24 Gaussian sigmas must be finite, evaluable, and strictly positive")]
    InvalidGaussianSigma,
    #[error("the Goldreich predicate requires at least five input bits")]
    GoldreichInputTooSmall,
    #[error("the Goldreich output count does not satisfy m < n^1.4")]
    GoldreichOutputBound,
    #[error("the AKY24 prMIFE arity overflowed")]
    ArityOverflow,
    #[error("the AKY24 circuit-compatible SKE bit layout is inconsistent")]
    SkeLayout,
    #[error("the AKY24 cascade function input layout overflowed")]
    CascadeLayoutOverflow,
    #[error("the private-prFE modulus split must be positive and divide the ciphertext modulus")]
    InvalidModulusSplit,
    #[error("the AKY24 attribute encoding requires a binary gadget covering the modulus")]
    InvalidBinaryGadget,
    #[error("the AKY24 compact preimage coefficient bound must be non-negative")]
    InvalidPreimageBound,
}

impl Aky24IoConfig {
    pub fn validate(&self) -> Result<(), Aky24ConfigError> {
        if self.modulus <= BigInt::from(0) ||
            self.ring_dimension == 0 ||
            self.input_size == 0 ||
            self.digit_count == 0 ||
            self.security_parameter_bits == 0 ||
            self.cascade_randomness_bits == 0 ||
            self.gaussian_sample_bits == 0 ||
            self.gaussian_sample_bits > 52 ||
            self.uniform_statistical_bits == 0 ||
            self.function.output_bits == 0
        {
            return Err(Aky24ConfigError::NonPositiveParameter);
        }
        if self.preimage_max_coefficient_bound < BigInt::from(0) {
            return Err(Aky24ConfigError::InvalidPreimageBound);
        }
        let bindings = ParamEnv::default();
        if [
            &self.trapdoor_sigma,
            &self.secret_sigma,
            &self.b_error_sigma,
            &self.fhe_error_sigma,
            &self.attribute_error_sigma,
        ]
        .into_iter()
        .any(|sigma| {
            sigma.evaluate_f64(&bindings).map_or(true, |value| !value.is_finite() || value <= 0.0)
        }) {
            return Err(Aky24ConfigError::InvalidGaussianSigma);
        }
        if self.modulus_split <= BigInt::from(0) ||
            &self.modulus % &self.modulus_split != BigInt::from(0)
        {
            return Err(Aky24ConfigError::InvalidModulusSplit);
        }
        if self.gadget_base != BigInt::from(2) ||
            BigInt::from(2).pow(self.digit_count as u32) < self.modulus
        {
            return Err(Aky24ConfigError::InvalidBinaryGadget);
        }
        if self.input_size < 5 {
            return Err(Aky24ConfigError::GoldreichInputTooSmall);
        }
        if !mxx_gadgets::circuit_gadgets::fhe_prg::goldreich::goldreich_output_bound_holds(
            self.input_size,
            self.function.output_bits,
        ) {
            return Err(Aky24ConfigError::GoldreichOutputBound);
        }
        self.prmife_arity()?;
        Ok(())
    }

    /// Section 5.2 uses one prMIFE input for every user bit and one final input
    /// for the circuit description.
    pub fn prmife_arity(&self) -> Result<usize, Aky24ConfigError> {
        self.input_size.checked_add(1).ok_or(Aky24ConfigError::ArityOverflow)
    }

    pub fn public_key_columns(&self) -> usize {
        self.digit_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Aky24IoConfig {
        Aky24IoConfig {
            modulus: 257.into(),
            ring_dimension: 8,
            input_size: 8,
            gadget_base: 2.into(),
            digit_count: 9,
            preimage_max_coefficient_bound: 1_000_000.into(),
            modulus_split: 1.into(),
            trapdoor_sigma: RealExpr::from_integer(4),
            secret_sigma: RealExpr::from_integer(2),
            b_error_sigma: RealExpr::from_integer(1),
            fhe_error_sigma: RealExpr::from_integer(1),
            attribute_error_sigma: RealExpr::from_integer(1),
            security_parameter_bits: 128,
            cascade_randomness_bits: 128,
            gaussian_sample_bits: 16,
            uniform_statistical_bits: 16,
            function: Aky24GoldreichPrf { output_bits: 2, graph_seed: [9; 32] },
        }
    }

    #[test]
    fn circuit_description_occupies_the_last_prmife_slot() {
        let config = config();
        config.validate().unwrap();
        assert_eq!(config.prmife_arity().unwrap(), config.input_size + 1);
    }

    #[test]
    fn invalid_goldreich_domain_is_rejected() {
        let mut config = config();
        config.input_size = 4;
        assert_eq!(config.validate(), Err(Aky24ConfigError::GoldreichInputTooSmall));
    }

    #[test]
    fn modulus_split_must_divide_the_ciphertext_modulus() {
        let mut config = config();
        config.modulus_split = 4.into();
        assert_eq!(config.validate(), Err(Aky24ConfigError::InvalidModulusSplit));
    }

    #[test]
    fn attribute_encoding_requires_binary_digits_covering_the_modulus() {
        let mut nonbinary = config();
        nonbinary.gadget_base = 4.into();
        assert_eq!(nonbinary.validate(), Err(Aky24ConfigError::InvalidBinaryGadget));

        let mut too_short = config();
        too_short.digit_count = 8;
        assert_eq!(too_short.validate(), Err(Aky24ConfigError::InvalidBinaryGadget));
    }

    #[test]
    fn modulus_and_all_gaussian_sigmas_must_be_strictly_positive() {
        let mut zero_modulus = config();
        zero_modulus.modulus = BigInt::from(0);
        assert_eq!(zero_modulus.validate(), Err(Aky24ConfigError::NonPositiveParameter));

        let setters: [fn(&mut Aky24IoConfig, RealExpr); 5] = [
            |config: &mut Aky24IoConfig, sigma| config.trapdoor_sigma = sigma,
            |config: &mut Aky24IoConfig, sigma| config.secret_sigma = sigma,
            |config: &mut Aky24IoConfig, sigma| config.b_error_sigma = sigma,
            |config: &mut Aky24IoConfig, sigma| config.fhe_error_sigma = sigma,
            |config: &mut Aky24IoConfig, sigma| config.attribute_error_sigma = sigma,
        ];
        for (index, setter) in setters.into_iter().enumerate() {
            let mut zero = config();
            setter(&mut zero, RealExpr::from_integer(0));
            assert_eq!(
                zero.validate(),
                Err(Aky24ConfigError::InvalidGaussianSigma),
                "sigma field {index} accepted zero"
            );

            let mut negative = config();
            setter(&mut negative, RealExpr::from_integer(-1));
            assert_eq!(
                negative.validate(),
                Err(Aky24ConfigError::InvalidGaussianSigma),
                "sigma field {index} accepted a negative value"
            );
        }

        let mut unevaluable = config();
        unevaluable.trapdoor_sigma =
            RealExpr::Div(Box::new(RealExpr::from_integer(1)), Box::new(RealExpr::from_integer(0)));
        assert_eq!(unevaluable.validate(), Err(Aky24ConfigError::InvalidGaussianSigma));
    }

    #[test]
    fn preimage_bound_is_explicit_and_non_negative() {
        let mut invalid = config();
        invalid.preimage_max_coefficient_bound = (-1).into();
        assert_eq!(invalid.validate(), Err(Aky24ConfigError::InvalidPreimageBound));
        assert_eq!(config().preimage_max_coefficient_bound, 1_000_000.into());
    }
}
