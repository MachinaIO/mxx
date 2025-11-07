use super::params::IntRingPolyParams;
use crate::{
    element::{PolyElem, finite_ring::FinRingElem},
    impl_binop_with_refs,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntRingPoly {
    inner: FinRingElem,
}

impl IntRingPoly {
    pub fn new(inner: FinRingElem) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &FinRingElem {
        &self.inner
    }

    fn from_biguint(params: &IntRingPolyParams, value: BigUint) -> Self {
        Self::from_elem_to_constant(params, &FinRingElem::new(value, params.modulus()))
    }
}

impl Poly for IntRingPoly {
    type Elem = FinRingElem;
    type Params = IntRingPolyParams;

    fn from_bool_vec(params: &Self::Params, coeffs: &[bool]) -> Self {
        debug_assert!(
            coeffs.len() <= 1,
            "IntRingPoly expects at most one coefficient, got {}",
            coeffs.len()
        );
        let bit = coeffs[0];
        let elem = FinRingElem::constant(&params.modulus(), bit as u64);
        Self { inner: elem }
    }

    fn from_coeffs(params: &Self::Params, coeffs: &[Self::Elem]) -> Self {
        assert_eq!(coeffs.len(), 1, "IntRingPoly expects exactly one coefficient");
        let coeff = coeffs[0].clone();
        debug_assert_eq!(coeff.modulus(), &params.modulus());
        Self { inner: coeff }
    }

    fn from_u32s(params: &Self::Params, coeffs: &[u32]) -> Self {
        let value = coeffs.first().copied().unwrap_or_default();
        let elem = FinRingElem::constant(&params.modulus(), value as u64);
        Self { inner: elem }
    }

    fn from_biguints(params: &Self::Params, coeffs: &[BigUint]) -> Self {
        assert_eq!(coeffs.len(), 1, "IntRingPoly expects exactly one coefficient");
        Self::from_biguint_to_constant(params, coeffs[0].clone())
    }

    fn from_biguints_eval(params: &Self::Params, slots: &[BigUint]) -> Self {
        // Evaluation and coefficient domains coincide for degree-1 polynomials.
        Self::from_biguints(params, slots)
    }

    fn from_decomposed(params: &Self::Params, decomposed: &[Self]) -> Self {
        let base = BigUint::from(1u32) << params.base_bits();
        let modulus = params.modulus();
        let mut acc = BigUint::zero();
        let mut power = BigUint::from(1u32);
        for digit_poly in decomposed {
            acc += digit_poly.inner.value() * &power;
            power *= &base;
        }
        Self::from_biguint_to_constant(params, acc % modulus.as_ref())
    }

    fn from_compact_bytes(params: &Self::Params, bytes: &[u8]) -> Self {
        let expected_len = params.modulus_bits().div_ceil(8);
        assert_eq!(bytes.len(), expected_len, "unexpected byte length for IntRingPoly");
        let elem = FinRingElem::from_bytes(&params.modulus(), bytes);
        Self { inner: elem }
    }

    fn coeffs(&self) -> Vec<Self::Elem> {
        vec![self.inner.clone()]
    }

    fn const_zero(params: &Self::Params) -> Self {
        Self { inner: FinRingElem::zero(&params.modulus()) }
    }

    fn const_one(params: &Self::Params) -> Self {
        Self { inner: FinRingElem::one(&params.modulus()) }
    }

    fn const_minus_one(params: &Self::Params) -> Self {
        Self { inner: FinRingElem::minus_one(&params.modulus()) }
    }

    fn from_power_of_base_to_constant(params: &Self::Params, k: usize) -> Self {
        let base = BigUint::from(1u32) << params.base_bits();
        Self::from_biguint_to_constant(params, base.pow(k as u32))
    }

    fn from_elem_to_constant(params: &Self::Params, constant: &Self::Elem) -> Self {
        debug_assert_eq!(constant.modulus(), &params.modulus());
        Self { inner: constant.clone() }
    }

    fn from_biguint_to_constant(params: &Self::Params, int: BigUint) -> Self {
        let elem = FinRingElem::new(int, params.modulus());
        Self { inner: elem }
    }

    fn from_usize_to_constant(params: &Self::Params, int: usize) -> Self {
        let elem = FinRingElem::constant(&params.modulus(), int as u64);
        Self { inner: elem }
    }

    fn from_usize_to_lsb(params: &Self::Params, int: usize) -> Self {
        let bit = (int & 1) as u64;
        let elem = FinRingElem::constant(&params.modulus(), bit);
        Self { inner: elem }
    }

    fn const_max(params: &Self::Params) -> Self {
        Self { inner: FinRingElem::max_q(&params.modulus()) }
    }

    fn from_biguints_eval_single_mod(
        _params: &Self::Params,
        _crt_idx: usize,
        _slots: &[BigUint],
    ) -> Self {
        unimplemented!("CRT decomposition is not supported for IntRingPoly")
    }

    fn extract_bits_with_threshold(&self, params: &Self::Params) -> Vec<bool> {
        let modulus = params.modulus();
        let half_q = FinRingElem::half_q(&modulus);
        let quarter_q = half_q.value() >> 1;
        let three_quarter_q = &quarter_q * 3u32;
        let coeff = self.inner.value();
        vec![coeff >= &quarter_q && coeff < &three_quarter_q]
    }

    fn decompose_base(&self, params: &Self::Params) -> Vec<Self> {
        let coeff = self.inner.value();
        let num_digits = params.modulus_digits();
        let base_bits = params.base_bits() as usize;
        let base_mask = (&BigUint::from(1u32) << base_bits) - 1u32;
        let last_mask = params.decompose_last_mask().map(BigUint::from);
        (0..num_digits)
            .map(|digit_idx| {
                let mask = if digit_idx == num_digits - 1 {
                    last_mask.as_ref().cloned().unwrap_or_else(|| base_mask.clone())
                } else {
                    base_mask.clone()
                };
                let shift = digit_idx * base_bits;
                let mut digit_value = coeff >> shift;
                digit_value &= mask;
                Self::from_biguint(params, digit_value)
            })
            .collect()
    }

    fn to_bool_vec(&self) -> Vec<bool> {
        vec![self.inner.to_bit()]
    }

    fn to_compact_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    fn to_const_int(&self) -> usize {
        self.inner.value().try_into().unwrap_or(usize::MAX)
    }
}

impl Neg for IntRingPoly {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { inner: -self.inner }
    }
}

impl Neg for &IntRingPoly {
    type Output = IntRingPoly;

    fn neg(self) -> Self::Output {
        IntRingPoly { inner: -self.inner.clone() }
    }
}

impl AddAssign for IntRingPoly {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl AddAssign<&IntRingPoly> for IntRingPoly {
    fn add_assign(&mut self, rhs: &IntRingPoly) {
        self.inner += &rhs.inner;
    }
}

impl SubAssign for IntRingPoly {
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl SubAssign<&IntRingPoly> for IntRingPoly {
    fn sub_assign(&mut self, rhs: &IntRingPoly) {
        self.inner -= &rhs.inner;
    }
}

impl MulAssign for IntRingPoly {
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl MulAssign<&IntRingPoly> for IntRingPoly {
    fn mul_assign(&mut self, rhs: &IntRingPoly) {
        self.inner *= &rhs.inner;
    }
}

impl_binop_with_refs!(IntRingPoly => Add::add(self, rhs: &IntRingPoly) -> IntRingPoly {
    IntRingPoly { inner: self.inner.clone() + &rhs.inner }
});

impl_binop_with_refs!(IntRingPoly => Sub::sub(self, rhs: &IntRingPoly) -> IntRingPoly {
    IntRingPoly { inner: self.inner.clone() - &rhs.inner }
});

impl_binop_with_refs!(IntRingPoly => Mul::mul(self, rhs: &IntRingPoly) -> IntRingPoly {
    IntRingPoly { inner: self.inner.clone() * &rhs.inner }
});

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    fn params() -> IntRingPolyParams {
        IntRingPolyParams::new(BigUint::from(257u32), 4)
    }

    #[test]
    fn test_integer_ring_basic_constructors() {
        let params = params();
        let poly = IntRingPoly::from_u32s(&params, &[42]);
        assert_eq!(poly.coeffs()[0].value(), &BigUint::from(42u32));

        let zero = IntRingPoly::const_zero(&params);
        assert_eq!(zero.coeffs()[0].value(), &BigUint::from(0u32));
    }

    #[test]
    fn test_integer_ring_decompose_recompose() {
        let params = params();
        let value = BigUint::from(1337u32);
        let poly = IntRingPoly::from_biguint_to_constant(&params, value.clone());
        let digits = poly.decompose_base(&params);
        let recomposed = IntRingPoly::from_decomposed(&params, &digits);
        assert_eq!(recomposed.coeffs()[0].value(), &(value % params.modulus().as_ref()));
    }

    #[test]
    fn test_integer_ring_arithmetic() {
        let params = params();
        let a = IntRingPoly::from_u32s(&params, &[10]);
        let b = IntRingPoly::from_u32s(&params, &[20]);
        assert_eq!((a.clone() + b.clone()).coeffs()[0].value(), &BigUint::from(30u32));
        assert_eq!((a.clone() * b.clone()).coeffs()[0].value(), &BigUint::from(200u32));
        assert_eq!((a - b).coeffs()[0].value(), &BigUint::from(257u32 - 10));
    }
}
