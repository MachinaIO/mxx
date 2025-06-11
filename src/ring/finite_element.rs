use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::sign::Signed;
use std::{
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    sync::Arc,
};

use crate::ring::FiniteRing;

#[derive(Clone)]
pub struct FinRingElem {
    value: BigUint,
    ring: Arc<FiniteRing>,
}

impl FinRingElem {
    pub fn new<V>(value: V, ring: Arc<FiniteRing>) -> Self
    where
        V: Into<BigInt>,
    {
        let value: BigInt = value.into();
        let modulus_bigint = ring.modulus().to_bigint().unwrap();
        let value = if value.is_negative() {
            ((value % &modulus_bigint) + &modulus_bigint) % &modulus_bigint
        } else {
            value % &modulus_bigint
        };
        let reduced_value = value.to_biguint().unwrap();
        assert!((BigUint::ZERO <= reduced_value) && (reduced_value < *ring.modulus()));
        Self {
            value: reduced_value,
            ring,
        }
    }

    pub fn value(&self) -> &BigUint {
        &self.value
    }

    pub fn modulus(&self) -> &BigUint {
        &self.ring.modulus()
    }

    pub fn modulus_switch<'a>(&self, dst_ring: Arc<FiniteRing>) -> FinRingElem {
        let q_prime = dst_ring.modulus();
        let q = &self.ring.modulus();
        let new_value = ((&self.value * q_prime) / *q) % q_prime;
        FinRingElem {
            value: new_value,
            ring: dst_ring,
        }
    }
}

impl FinRingElem {
    #[inline]
    fn from_sum(a: &Self, b: &Self) -> Self {
        assert_eq!(a.modulus(), b.modulus());
        Self::new((&a.value + &b.value) % a.modulus(), a.ring.clone())
    }

    #[inline]
    fn from_prod(a: &Self, b: &Self) -> Self {
        assert_eq!(a.modulus(), b.modulus());
        Self::new((&a.value * &b.value) % a.modulus(), a.ring.clone())
    }
}

impl core::ops::Neg for &FinRingElem {
    type Output = FinRingElem;

    #[inline]
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl Add<FinRingElem> for FinRingElem {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}
impl<'r, 'a> Add<&'a FinRingElem> for FinRingElem {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        &self + rhs
    }
}
impl<'r, 'a> Add<FinRingElem> for &'a FinRingElem {
    type Output = FinRingElem;
    fn add(self, rhs: FinRingElem) -> Self::Output {
        self + &rhs
    }
}

impl<'r, 'a, 'b> Add<&'b FinRingElem> for &'a FinRingElem {
    type Output = FinRingElem;
    fn add(self, rhs: &'b FinRingElem) -> Self::Output {
        FinRingElem::from_sum(self, rhs)
    }
}

impl AddAssign<FinRingElem> for FinRingElem {
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}
impl<'r, 'a> AddAssign<&'a FinRingElem> for FinRingElem {
    fn add_assign(&mut self, rhs: &'a Self) {
        *self = &*self + rhs;
    }
}

impl Mul<FinRingElem> for FinRingElem {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}
impl<'r, 'a> Mul<&'a FinRingElem> for FinRingElem {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        &self * rhs
    }
}
impl<'r, 'a> Mul<FinRingElem> for &'a FinRingElem {
    type Output = FinRingElem;
    fn mul(self, rhs: FinRingElem) -> Self::Output {
        self * &rhs
    }
}
impl<'r, 'a, 'b> Mul<&'b FinRingElem> for &'a FinRingElem {
    type Output = FinRingElem;
    fn mul(self, rhs: &'b FinRingElem) -> Self::Output {
        FinRingElem::from_prod(self, rhs)
    }
}

impl MulAssign<FinRingElem> for FinRingElem {
    fn mul_assign(&mut self, rhs: Self) {
        *self = &*self * &rhs;
    }
}
impl<'r, 'a> MulAssign<&'a FinRingElem> for FinRingElem {
    fn mul_assign(&mut self, rhs: &'a Self) {
        *self = &*self * rhs;
    }
}

impl Sub<FinRingElem> for FinRingElem {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<'r, 'a> Sub<&'a FinRingElem> for FinRingElem {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        &self - rhs
    }
}

impl<'r, 'a> Sub<FinRingElem> for &'a FinRingElem {
    type Output = FinRingElem;
    fn sub(self, rhs: FinRingElem) -> Self::Output {
        self - &rhs
    }
}

impl<'r, 'a, 'b> Sub<&'b FinRingElem> for &'a FinRingElem {
    type Output = FinRingElem;
    fn sub(self, rhs: &'b FinRingElem) -> Self::Output {
        self + &-&*rhs
    }
}

impl SubAssign<FinRingElem> for FinRingElem {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl<'r, 'a> SubAssign<&'a FinRingElem> for FinRingElem {
    fn sub_assign(&mut self, rhs: &'a Self) {
        *self = &*self - rhs;
    }
}

impl Neg for FinRingElem {
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
    fn round_trip_elem_from_i64() {
        let ring = FiniteRing::new(BigUint::from(97u8));
        for x in [-1, -97, -98, 0, 1, 96, 97, 98] {
            let e = ring.elem_from_i64(x);
            assert!(e.value() < &ring.modulus());
        }
    }

    #[test]
    fn add_assign_self_alias() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let mut x = ring.elem_from_i64(5);
        x += x.clone();
        assert_eq!(x.value(), &BigUint::from(10u8));
    }

    #[test]
    fn test_element_add() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let a = ring.elem_from_i64(19);
        let b = ring.elem_from_i64(16);
        let c = a + b;
        assert_eq!(c.value(), &BigUint::from(1u8));
        assert_eq!(c.modulus(), ring.modulus());

        let ring = FiniteRing::new(BigUint::from(10000usize));
        let a = ring.elem_from_i64(19 + 10000);
        let b = ring.elem_from_i64(16 + 10000);
        let c = a + b;
        assert_eq!(c.value(), &BigUint::from(35u8));
        assert_eq!(c.modulus(), ring.modulus());
    }

    #[test]
    fn test_element_sub() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let a = ring.elem_from_i64(-1);
        let b = ring.elem_from_i64(4);
        let c = a - b;
        assert_eq!(c.value(), &BigUint::from(12u8));
        assert_eq!(c.modulus(), ring.modulus());

        let ring = FiniteRing::new(BigUint::from(10000u32));
        let a = ring.elem_from_i64(-19);
        let b = ring.elem_from_i64(16 + 10000);
        let c = a - b;
        assert_eq!(c.value(), &BigUint::from(9965u32));
        assert_eq!(c.modulus(), ring.modulus());
    }

    #[test]
    fn test_element_mul() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let a = ring.elem_from_i64(3);
        let b = ring.elem_from_i64(5);
        let c = a * b;
        assert_eq!(c.value(), &BigUint::from(15u8));
        assert_eq!(c.modulus(), ring.modulus());

        let ring = FiniteRing::new(BigUint::from(10000u32));
        let a = ring.elem_from_i64(200);
        let b = ring.elem_from_i64(50);
        let c = a * b;
        assert_eq!(c.value(), &BigUint::from(0u8));
        assert_eq!(c.modulus(), ring.modulus());
    }

    #[test]
    fn test_element_neg() {
        let ring = FiniteRing::new(BigUint::from(17u8));
        let a = ring.elem_from_i64(5);
        let neg_a = -a;
        assert_eq!(neg_a.value(), &BigUint::from(12u8));
        assert_eq!(neg_a.modulus(), ring.modulus());

        let ring = FiniteRing::new(BigUint::from(10000u32));
        let a = ring.elem_from_i64(200);
        let neg_a = -a;
        assert_eq!(neg_a.value(), &BigUint::from(9800u32));
        assert_eq!(neg_a.modulus(), ring.modulus());
    }
}
