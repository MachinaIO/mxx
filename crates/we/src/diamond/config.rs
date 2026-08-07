use mxx_dsl::Ring;
use mxx_gadgets::input_injector::{
    DIAMOND_PREFIX_DIMENSION, DIAMOND_SECRET_DIMENSION, DiamondInputConfig, DiamondInputConfigError,
};
use mxx_ir_core::{IntExpr, ParamEnv, RealExpr};
use num_bigint::BigInt;
use num_integer::Integer;
use thiserror::Error;

pub type DiamondConfigError = DiamondInputConfigError;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DiamondSamplerBoundError {
    #[error("a sampler sigma bound is not a closed exact expression")]
    Expression,
    #[error("a sampler-bound dimension calculation overflowed")]
    DimensionOverflow,
}

/// Returns the default deterministic cutoff `floor(6.5 * sigma)` without an `f64` round trip.
pub fn default_error_max_coefficient_bound(
    sigma: &RealExpr,
) -> Result<BigInt, DiamondSamplerBoundError> {
    let value = sigma
        .evaluate_rational(&ParamEnv::default())
        .map_err(|_| DiamondSamplerBoundError::Expression)?;
    Ok((value.numerator() * BigInt::from(13)).div_floor(&(value.denominator() * BigInt::from(2))))
}

/// Returns `floor(6.5 * sigma_preimage)` using a rational upper bound for the existing sampler's
/// preimage sigma formula. Integer square-root ceilings keep this a hard upper bound.
pub fn default_preimage_max_coefficient_bound(
    trapdoor_sigma: &RealExpr,
    ring_dimension: usize,
    digit_count: usize,
    gadget_base: &BigInt,
) -> Result<BigInt, DiamondSamplerBoundError> {
    let tau = trapdoor_sigma
        .evaluate_rational(&ParamEnv::default())
        .map_err(|_| DiamondSamplerBoundError::Expression)?;
    let root_rkn = ceil_sqrt(
        DIAMOND_PREFIX_DIMENSION
            .checked_mul(digit_count)
            .and_then(|value| value.checked_mul(ring_dimension))
            .ok_or(DiamondSamplerBoundError::DimensionOverflow)?,
    );
    let root_2n = ceil_sqrt(
        2usize.checked_mul(ring_dimension).ok_or(DiamondSamplerBoundError::DimensionOverflow)?,
    );
    let bracket_num = BigInt::from(10 * (root_rkn + root_2n) + 47);
    let numerator = BigInt::from(9) *
        (gadget_base + BigInt::from(1)) *
        tau.numerator() *
        tau.numerator() *
        bracket_num;
    let denominator = BigInt::from(50) * tau.denominator() * tau.denominator();
    Ok((numerator * BigInt::from(13)).div_floor(&(denominator * BigInt::from(2))))
}

fn ceil_sqrt(value: usize) -> usize {
    if value <= 1 {
        return value;
    }
    let mut low = 1usize;
    let mut high = value;
    while low < high {
        let mid = low + (high - low) / 2;
        if mid >= value.div_ceil(mid) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondWeConfig {
    pub modulus: BigInt,
    pub ring_dimension: usize,
    pub input_count: usize,
    pub digit_base: usize,
    pub batch_bits: usize,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub trapdoor_sigma: RealExpr,
    pub error_sigma: RealExpr,
    pub error_max_coefficient_bound: BigInt,
    pub preimage_max_coefficient_bound: BigInt,
    pub bgg_tag: Vec<u8>,
}

impl DiamondWeConfig {
    pub fn validate(&self) -> Result<(), DiamondConfigError> {
        self.input_config().validate()
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
            error_max_coefficient_bound: self.error_max_coefficient_bound.clone(),
            preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
        }
    }

    pub fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension)
    }

    pub fn witness_size(&self) -> Result<usize, DiamondConfigError> {
        self.input_config().witness_size()
    }

    pub fn state_rows(&self) -> usize {
        self.input_config().state_rows()
    }

    pub fn state_columns(&self) -> Result<usize, DiamondConfigError> {
        self.input_config().state_columns()
    }

    pub fn public_key_columns(&self) -> Result<usize, DiamondConfigError> {
        DIAMOND_SECRET_DIMENSION
            .checked_mul(self.digit_count)
            .ok_or(DiamondConfigError::LayoutOverflow)
    }

    pub fn state_count_at_level(&self, level: usize) -> Result<usize, DiamondConfigError> {
        self.input_config().state_count_at_level(level)
    }

    pub fn bit_state_index(
        &self,
        digit_index: usize,
        bit_index: usize,
    ) -> Result<usize, DiamondConfigError> {
        self.input_config().bit_state_index(digit_index, bit_index)
    }

    pub fn gadget_base_expr(&self) -> IntExpr {
        self.gadget_base.clone().into()
    }

    pub fn digit_count_expr(&self) -> IntExpr {
        self.digit_count.into()
    }
}
