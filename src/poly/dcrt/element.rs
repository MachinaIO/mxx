use crate::element::{PolyElem, finite_ring::FinRingElem};
use num_bigint::BigUint;
use std::sync::Arc;

impl PolyElem for FinRingElem {
    type Modulus = Arc<BigUint>;
    fn zero(modulus: &Self::Modulus) -> Self {
        Self::new(0, modulus.clone())
    }

    fn one(modulus: &Self::Modulus) -> Self {
        Self::new(1, modulus.clone())
    }

    fn minus_one(modulus: &Self::Modulus) -> Self {
        Self::new(modulus.as_ref() - &BigUint::from(1u8), modulus.clone())
    }

    fn constant(modulus: &Self::Modulus, value: u64) -> Self {
        Self::new(value, modulus.clone())
    }

    fn to_bit(&self) -> bool {
        if self.value == BigUint::ZERO {
            false
        } else if self.value == BigUint::from(1u8) {
            true
        } else {
            panic!("Cannot convert non-zero or non-one value to bit");
        }
    }

    fn half_q(modulus: &Self::Modulus) -> Self {
        let bits = modulus.bits();
        let value = BigUint::from(1u8) << (bits - 1);
        Self::new(value, modulus.clone())
    }

    fn max_q(modulus: &Self::Modulus) -> Self {
        Self::new(modulus.as_ref() - &BigUint::from(1u8), modulus.clone())
    }

    fn modulus(&self) -> &Self::Modulus {
        &self.modulus
    }

    fn from_bytes(modulus: &Self::Modulus, bytes: &[u8]) -> Self {
        let value = BigUint::from_bytes_le(bytes);
        Self::new(value, modulus.clone())
    }

    fn to_bytes(&self) -> Vec<u8> {
        let log_q_bytes = self.modulus.bits().div_ceil(8) as usize;
        let mut bytes = self.value.to_bytes_le();
        bytes.resize(log_q_bytes, 0);
        bytes
    }

    fn to_biguint(&self) -> &num_bigint::BigUint {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_zero() {
        let modulus = Arc::new(BigUint::from(17u8));
        let zero = FinRingElem::zero(&modulus);
        assert_eq!(zero.value(), &BigUint::from(0u8));
        assert_eq!(zero.modulus(), modulus.as_ref());

        let modulus = Arc::new(BigUint::from(10000usize));
        let zero = FinRingElem::zero(&modulus);
        assert_eq!(zero.value(), &BigUint::from(0u8));
        assert_eq!(zero.modulus(), modulus.as_ref());
    }

    #[test]
    fn test_element_one() {
        let modulus = Arc::new(BigUint::from(17u8));
        let one = FinRingElem::one(&modulus);
        assert_eq!(one.value(), &BigUint::from(1u8));
        assert_eq!(one.modulus(), modulus.as_ref());

        let modulus = Arc::new(BigUint::from(10000usize));
        let one = FinRingElem::one(&modulus);
        assert_eq!(one.value(), &BigUint::from(1u8));
        assert_eq!(one.modulus(), modulus.as_ref());
    }

    #[test]
    fn test_element_minus_one() {
        let modulus = Arc::new(BigUint::from(17u8));
        let minus_one = FinRingElem::minus_one(&modulus);
        assert_eq!(minus_one.value(), &BigUint::from(16u8));
        assert_eq!(minus_one.modulus(), modulus.as_ref());

        let modulus = Arc::new(BigUint::from(10000usize));
        let minus_one = FinRingElem::minus_one(&modulus);
        assert_eq!(minus_one.value(), &BigUint::from((10000 - 1) as usize));
        assert_eq!(minus_one.modulus(), modulus.as_ref());
    }
}
