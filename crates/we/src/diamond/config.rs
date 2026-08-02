use mxx_dsl::Ring;
use mxx_gadgets::input_injector::{
    DIAMOND_SECRET_DIMENSION, DiamondInputConfig, DiamondInputConfigError,
};
use mxx_ir_core::{IntExpr, RealExpr};
use num_bigint::BigInt;

pub type DiamondConfigError = DiamondInputConfigError;

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
