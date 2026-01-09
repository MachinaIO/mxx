use crate::element::{PolyElem, finite_ring::FinRingElem};
use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::Signed;
use std::sync::Arc;

impl PolyElem for FinRingElem {
    type Modulus = Arc<BigUint>;

    fn new<V: Into<BigInt>>(value: V, modulus: Self::Modulus) -> Self {
        let value = value.into();
        let modulus_bigint = modulus.as_ref().to_bigint().unwrap();
        let value = if value.is_negative() {
            ((value % &modulus_bigint) + &modulus_bigint) % &modulus_bigint
        } else {
            value % &modulus_bigint
        };
        let reduced_value = value.to_biguint().unwrap();
        Self { value: reduced_value, modulus }
    }

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

    fn from_int64(value: i64, modulus: &Self::Modulus) -> Self {
        if value < 0 {
            let abs_rem = value.unsigned_abs() % modulus.as_ref();
            let value = modulus.as_ref() - abs_rem;
            Self::new(value, modulus.clone())
        } else {
            Self::new(value, modulus.clone())
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let log_q_bytes = self.modulus.bits().div_ceil(8) as usize;
        let mut bytes = self.value.to_bytes_le();
        bytes.resize(log_q_bytes, 0);
        bytes
    }

    fn value(&self) -> &num_bigint::BigUint {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::{__PAIR, __TestState};
    use super::*;

    #[test]
    #[sequential_test::sequential]
    fn test_element_zero() {
        let modulus = Arc::new(BigUint::from(17u8));
        let zero = FinRingElem::zero(&modulus);
        assert_eq!(zero.value(), &BigUint::from(0u8));
        assert_eq!(zero.modulus(), &modulus);

        let modulus = Arc::new(BigUint::from(10000usize));
        let zero = FinRingElem::zero(&modulus);
        assert_eq!(zero.value(), &BigUint::from(0u8));
        assert_eq!(zero.modulus(), &modulus);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_element_one() {
        let modulus = Arc::new(BigUint::from(17u8));
        let one = FinRingElem::one(&modulus);
        assert_eq!(one.value(), &BigUint::from(1u8));
        assert_eq!(one.modulus(), &modulus);

        let modulus = Arc::new(BigUint::from(10000usize));
        let one = FinRingElem::one(&modulus);
        assert_eq!(one.value(), &BigUint::from(1u8));
        assert_eq!(one.modulus(), &modulus);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_element_minus_one() {
        let modulus = Arc::new(BigUint::from(17u8));
        let minus_one = FinRingElem::minus_one(&modulus);
        assert_eq!(minus_one.value(), &BigUint::from(16u8));
        assert_eq!(minus_one.modulus(), &modulus);

        let modulus = Arc::new(BigUint::from(10000usize));
        let minus_one = FinRingElem::minus_one(&modulus);
        assert_eq!(minus_one.value(), &BigUint::from((10000 - 1) as usize));
        assert_eq!(minus_one.modulus(), &modulus);
    }
}
