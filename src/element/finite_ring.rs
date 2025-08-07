use crate::element::PolyElem;
use num_bigint::{BigInt, BigUint, ParseBigIntError};
use std::{
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
    sync::Arc,
};

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct FinRingElem {
    pub(crate) value: BigUint,
    pub(crate) modulus: Arc<BigUint>,
}

impl FinRingElem {
    pub fn from_str(value: &str, modulus: &str) -> Result<Self, ParseBigIntError> {
        let value = BigInt::from_str(value)?;
        let modulus = BigUint::from_str(modulus)?.into();
        Ok(Self::new(value, modulus))
    }

    pub fn modulus_switch(&self, new_modulus: Arc<BigUint>) -> Self {
        let value =
            ((&self.value * new_modulus.as_ref()) / self.modulus.as_ref()) % new_modulus.as_ref();
        Self { value, modulus: self.modulus.clone() }
    }
}

impl Add for FinRingElem {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<'a> Add<&'a FinRingElem> for FinRingElem {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        Self::new(self.value + &rhs.value, self.modulus)
    }
}

impl AddAssign for FinRingElem {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self::new(&self.value + rhs.value, self.modulus.clone());
    }
}

impl<'a> AddAssign<&'a FinRingElem> for FinRingElem {
    fn add_assign(&mut self, rhs: &'a Self) {
        *self = Self::new(&self.value + &rhs.value, self.modulus.clone());
    }
}

impl Mul for FinRingElem {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a> Mul<&'a FinRingElem> for FinRingElem {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self::new(self.value * &rhs.value, self.modulus)
    }
}

impl MulAssign for FinRingElem {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self::new(&self.value * rhs.value, self.modulus.clone());
    }
}

impl<'a> MulAssign<&'a FinRingElem> for FinRingElem {
    fn mul_assign(&mut self, rhs: &'a Self) {
        *self = Self::new(&self.value * &rhs.value, self.modulus.clone());
    }
}

impl Sub for FinRingElem {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a> Sub<&'a FinRingElem> for FinRingElem {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self + (&-rhs.clone())
    }
}

impl SubAssign for FinRingElem {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<'a> SubAssign<&'a FinRingElem> for FinRingElem {
    fn sub_assign(&mut self, rhs: &'a Self) {
        *self = self.clone() - rhs;
    }
}

impl Neg for FinRingElem {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.value == BigUint::ZERO {
            self
        } else {
            Self::new(self.modulus.as_ref() - &self.value, self.modulus)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_new_str() {
        let modulus = "17";
        let elem = FinRingElem::from_str("5", modulus).unwrap();
        assert_eq!(elem.value(), &BigUint::from(5u8));
        assert_eq!(elem.modulus().as_ref(), &BigUint::from(17u8));
        let elem = FinRingElem::from_str("-5", modulus).unwrap();
        assert_eq!(elem.value(), &BigUint::from(12u8));
        assert_eq!(elem.modulus().as_ref(), &BigUint::from(17u8));
        let elem = FinRingElem::from_str("-20", modulus).unwrap();
        assert_eq!(elem.value(), &BigUint::from(14u8));
        assert_eq!(elem.modulus().as_ref(), &BigUint::from(17u8));
        let is_err = FinRingElem::from_str("0xabc", modulus).is_err();
        assert!(is_err)
    }

    #[test]
    fn test_element_new() {
        let modulus = Arc::new(BigUint::from(17u8));
        let elem = FinRingElem::new(5, modulus.clone());
        assert_eq!(elem.value(), &BigUint::from(5u8));
        assert_eq!(elem.modulus().as_ref(), &BigUint::from(17u8));
        let elem = FinRingElem::new(-5, modulus.clone());
        assert_eq!(elem.value(), &BigUint::from(12u8));
        assert_eq!(elem.modulus().as_ref(), &BigUint::from(17u8));
        let elem = FinRingElem::new(-20, modulus.clone());
        assert_eq!(elem.value(), &BigUint::from(14u8));
        assert_eq!(elem.modulus().as_ref(), &BigUint::from(17u8));
        let elem = FinRingElem::new(20, modulus.clone());
        assert_eq!(elem.value(), &BigUint::from(3u8));
    }

    #[test]
    fn test_element_add() {
        let modulus = Arc::new(BigUint::from(17u8));
        let a = FinRingElem::new(19, modulus.clone());
        let b = FinRingElem::new(16, modulus.clone());
        let c = a + b;
        assert_eq!(c.value(), &BigUint::from(1u8));
        assert_eq!(c.modulus(), &modulus);

        let modulus = Arc::new(BigUint::from(10000usize));
        let a = FinRingElem::new(19 + 10000, modulus.clone());
        let b = FinRingElem::new(16 + 10000, modulus.clone());
        let c = a + b;
        assert_eq!(c.value(), &BigUint::from(35u8));
        assert_eq!(c.modulus(), &modulus);
    }

    #[test]
    fn test_element_sub() {
        let modulus = Arc::new(BigUint::from(17u8));
        let a = FinRingElem::new(-1, modulus.clone());
        let b = FinRingElem::new(4, modulus.clone());
        let c = a - b;
        assert_eq!(c.value(), &BigUint::from(12u8));
        assert_eq!(c.modulus(), &modulus);

        let modulus = Arc::new(BigUint::from(10000usize));
        let a = FinRingElem::new(-19, modulus.clone());
        let b = FinRingElem::new(16 + 10000, modulus.clone());
        let c = a - b;
        assert_eq!(c.value(), &BigUint::from(9965usize));
        assert_eq!(c.modulus(), &modulus);
    }

    #[test]
    fn test_element_mul() {
        let modulus = Arc::new(BigUint::from(17u8));
        let a = FinRingElem::new(3, modulus.clone());
        let b = FinRingElem::new(5, modulus.clone());
        let c = a * b;
        assert_eq!(c.value(), &BigUint::from(15u8)); // 3 * 5 ≡ 15 (mod 17)
        assert_eq!(c.modulus(), &modulus);

        let modulus = Arc::new(BigUint::from(10000usize));
        let a = FinRingElem::new(200, modulus.clone());
        let b = FinRingElem::new(50, modulus.clone());
        let c = a * b;
        assert_eq!(c.value(), &BigUint::from(0usize)); // 200 * 50 ≡ 0 (mod 10000)
        assert_eq!(c.modulus(), &modulus);
    }

    #[test]
    fn test_element_neg() {
        let modulus = Arc::new(BigUint::from(17u8));
        let a = FinRingElem::new(5, modulus.clone());
        let neg_a = -a;
        assert_eq!(neg_a.value(), &BigUint::from(12u8)); // -5 ≡ 12 (mod 17)
        assert_eq!(neg_a.modulus(), &modulus);

        let modulus = Arc::new(BigUint::from(10000usize));
        let a = FinRingElem::new(200, modulus.clone());
        let neg_a = -a;
        assert_eq!(neg_a.value(), &BigUint::from(9800usize)); // -200 ≡ 9800 (mod 10000)
        assert_eq!(neg_a.modulus(), &modulus);
    }
}
