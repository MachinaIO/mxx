use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::sign::Signed;
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Clone, PartialEq, Eq)]
pub struct FiniteRing {
    modulus: BigUint,
}

impl Debug for FiniteRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FiniteRing")
            .field("modulus", &self.modulus)
            .finish()
    }
}

impl FiniteRing {
    pub fn new(modulus: BigUint) -> Self {
        Self { modulus }
    }

    pub fn elem<V: Into<BigInt>>(&self, value: V) -> FinRingElem {
        FinRingElem::new(value, &self)
    }

    pub fn elem_from_i64<V: Into<i64>>(&self, value: V) -> FinRingElem {
        let value: i64 = value.into();
        if value < 0 {
            let abs_rem = value.unsigned_abs() % &self.modulus;
            let value = &self.modulus - abs_rem;
            FinRingElem::new(value, &self)
        } else {
            FinRingElem::new(value as u64, &self)
        }
    }
}

#[derive(Clone)]
pub struct FinRingElem<'r> {
    value: BigUint,
    ring: &'r FiniteRing,
}

impl<'r> FinRingElem<'r> {
    pub fn new<V>(value: V, ring: &'r FiniteRing) -> Self
    where
        V: Into<BigInt>,
    {
        let value: BigInt = value.into();
        let modulus_bigint = ring.modulus.to_bigint().unwrap();
        let value = if value.is_negative() {
            ((value % &modulus_bigint) + &modulus_bigint) % &modulus_bigint
        } else {
            value % &modulus_bigint
        };
        let reduced_value = value.to_biguint().unwrap();
        assert!((BigUint::ZERO <= reduced_value) && (reduced_value < ring.modulus));
        Self {
            value: reduced_value,
            ring,
        }
    }

    pub fn value(&self) -> &BigUint {
        &self.value
    }

    pub fn modulus(&self) -> &BigUint {
        &self.ring.modulus
    }

    pub fn modulus_switch<'a>(&self, dst_ring: &'a FiniteRing) -> FinRingElem<'a> {
        let q_prime = &dst_ring.modulus;
        let q = &self.ring.modulus;
        let new_value = ((&self.value * q_prime) / q) % q_prime;
        FinRingElem {
            value: new_value,
            ring: dst_ring,
        }
    }
}

impl<'r> FinRingElem<'r> {
    #[inline]
    fn from_sum(a: &Self, b: &Self) -> Self {
        assert_eq!(a.modulus(), b.modulus());
        Self::new((&a.value + &b.value) % a.modulus(), a.ring)
    }

    #[inline]
    fn from_prod(a: &Self, b: &Self) -> Self {
        assert_eq!(a.modulus(), b.modulus());
        Self::new((&a.value * &b.value) % a.modulus(), a.ring)
    }
}

impl<'r> core::ops::Neg for &FinRingElem<'r> {
    type Output = FinRingElem<'r>;

    #[inline]
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl<'r> Add<FinRingElem<'r>> for FinRingElem<'r> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}
impl<'r, 'a> Add<&'a FinRingElem<'r>> for FinRingElem<'r> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        &self + rhs
    }
}
impl<'r, 'a> Add<FinRingElem<'r>> for &'a FinRingElem<'r> {
    type Output = FinRingElem<'r>;
    fn add(self, rhs: FinRingElem<'r>) -> Self::Output {
        self + &rhs
    }
}

impl<'r, 'a, 'b> Add<&'b FinRingElem<'r>> for &'a FinRingElem<'r> {
    type Output = FinRingElem<'r>;
    fn add(self, rhs: &'b FinRingElem<'r>) -> Self::Output {
        FinRingElem::from_sum(self, rhs)
    }
}

impl<'r> AddAssign<FinRingElem<'r>> for FinRingElem<'r> {
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}
impl<'r, 'a> AddAssign<&'a FinRingElem<'r>> for FinRingElem<'r> {
    fn add_assign(&mut self, rhs: &'a Self) {
        *self = &*self + rhs;
    }
}

impl<'r> Mul<FinRingElem<'r>> for FinRingElem<'r> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}
impl<'r, 'a> Mul<&'a FinRingElem<'r>> for FinRingElem<'r> {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        &self * rhs
    }
}
impl<'r, 'a> Mul<FinRingElem<'r>> for &'a FinRingElem<'r> {
    type Output = FinRingElem<'r>;
    fn mul(self, rhs: FinRingElem<'r>) -> Self::Output {
        self * &rhs
    }
}
impl<'r, 'a, 'b> Mul<&'b FinRingElem<'r>> for &'a FinRingElem<'r> {
    type Output = FinRingElem<'r>;
    fn mul(self, rhs: &'b FinRingElem<'r>) -> Self::Output {
        FinRingElem::from_prod(self, rhs)
    }
}

impl<'r> MulAssign<FinRingElem<'r>> for FinRingElem<'r> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = &*self * &rhs;
    }
}
impl<'r, 'a> MulAssign<&'a FinRingElem<'r>> for FinRingElem<'r> {
    fn mul_assign(&mut self, rhs: &'a Self) {
        *self = &*self * rhs;
    }
}

impl<'r> Sub<FinRingElem<'r>> for FinRingElem<'r> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<'r, 'a> Sub<&'a FinRingElem<'r>> for FinRingElem<'r> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        &self - rhs
    }
}

impl<'r, 'a> Sub<FinRingElem<'r>> for &'a FinRingElem<'r> {
    type Output = FinRingElem<'r>;
    fn sub(self, rhs: FinRingElem<'r>) -> Self::Output {
        self - &rhs
    }
}

impl<'r, 'a, 'b> Sub<&'b FinRingElem<'r>> for &'a FinRingElem<'r> {
    type Output = FinRingElem<'r>;
    fn sub(self, rhs: &'b FinRingElem<'r>) -> Self::Output {
        self + &-&*rhs
    }
}

impl<'r> SubAssign<FinRingElem<'r>> for FinRingElem<'r> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl<'r, 'a> SubAssign<&'a FinRingElem<'r>> for FinRingElem<'r> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        *self = &*self - rhs;
    }
}

impl<'r> Neg for FinRingElem<'r> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        if self.value == BigUint::ZERO {
            self
        } else {
            Self {
                value: self.modulus() - &self.value,
                ring: self.ring,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_add() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let a = ring.elem_from_i64(19);
        let b = ring.elem_from_i64(16);
        let c = a + b;
        assert_eq!(c.value(), &BigUint::from(1u8));
        assert_eq!(*c.modulus(), ring.modulus);

        let ring = FiniteRing::new(BigUint::from(10000usize));
        let a = ring.elem_from_i64(19 + 10000);
        let b = ring.elem_from_i64(16 + 10000);
        let c = a + b;
        assert_eq!(c.value(), &BigUint::from(35u8));
        assert_eq!(*c.modulus(), ring.modulus);
    }

    #[test]
    fn test_element_sub() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let a = ring.elem_from_i64(-1);
        let b = ring.elem_from_i64(4);
        let c = a - b;
        assert_eq!(c.value(), &BigUint::from(12u8));
        assert_eq!(*c.modulus(), ring.modulus);

        let ring = FiniteRing::new(BigUint::from(10000u32));
        let a = ring.elem_from_i64(-19);
        let b = ring.elem_from_i64(16 + 10000);
        let c = a - b;
        assert_eq!(c.value(), &BigUint::from(9965u32));
        assert_eq!(*c.modulus(), ring.modulus);
    }

    #[test]
    fn test_element_mul() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let a = ring.elem_from_i64(3);
        let b = ring.elem_from_i64(5);
        let c = a * b;
        assert_eq!(c.value(), &BigUint::from(15u8));
        assert_eq!(*c.modulus(), ring.modulus);

        let ring = FiniteRing::new(BigUint::from(10000u32));
        let a = ring.elem_from_i64(200);
        let b = ring.elem_from_i64(50);
        let c = a * b;
        assert_eq!(c.value(), &BigUint::from(0u8));
        assert_eq!(*c.modulus(), ring.modulus);
    }

    #[test]
    fn test_element_neg() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let a = ring.elem_from_i64(5);
        let neg_a = -a;
        assert_eq!(neg_a.value(), &BigUint::from(12u8));
        assert_eq!(*neg_a.modulus(), ring.modulus);

        let ring = FiniteRing::new(BigUint::from(10000u32));
        let a = ring.elem_from_i64(200);
        let neg_a = -a;
        assert_eq!(neg_a.value(), &BigUint::from(9800u32));
        assert_eq!(*neg_a.modulus(), ring.modulus);
    }
}
