//! Pure conservative coefficient-bound arithmetic.

use crate::state::MatrixState;
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProductGeometry {
    pub inner_dimension: usize,
    pub ring_dimension: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum BoundError {
    #[error("modulus must be positive")]
    NonPositiveModulus,
    #[error("ring dimension must be positive")]
    ZeroRingDimension,
    #[error("matrix inner dimension must be positive")]
    ZeroInnerDimension,
}

/// Returns `floor(q / 2)` for a positive concrete modulus.
pub fn centered_residue_bound(modulus: &BigInt) -> Result<BigUint, BoundError> {
    let modulus = modulus.to_biguint().ok_or(BoundError::NonPositiveModulus)?;
    Ok(modulus >> 1)
}

pub fn convolution_factor(
    left_is_constant: bool,
    right_is_constant: bool,
    geometry: ProductGeometry,
) -> Result<BigUint, BoundError> {
    if geometry.ring_dimension == 0 {
        return Err(BoundError::ZeroRingDimension);
    }
    Ok(if left_is_constant || right_is_constant {
        BigUint::from(1u8)
    } else {
        BigUint::from(geometry.ring_dimension)
    })
}

pub fn product_bound(
    left: &BigUint,
    right: &BigUint,
    geometry: ProductGeometry,
    left_is_constant: bool,
    right_is_constant: bool,
) -> Result<BigUint, BoundError> {
    if geometry.inner_dimension == 0 {
        return Err(BoundError::ZeroInnerDimension);
    }
    let factor = convolution_factor(left_is_constant, right_is_constant, geometry)?;
    // This bounds one multiplicative term such as actual(A)e_B or
    // e_A nominal(B).  A matrix product sums `inner_dimension` coefficient
    // products, and each nonconstant polynomial product can contribute up to
    // `ring_dimension` convolution terms; a constant factor removes that
    // ring-wide sum.
    Ok(left * right * geometry.inner_dimension * factor)
}

pub fn left_action_gain(
    left: &MatrixState,
    geometry: ProductGeometry,
) -> Result<BigUint, BoundError> {
    let factor = convolution_factor(left.is_constant_polynomial, false, geometry)?;
    if geometry.inner_dimension == 0 {
        return Err(BoundError::ZeroInnerDimension);
    }
    // This is the operator norm used when a left matrix multiplies a right
    // error: |L * e| <= |L| * inner_dimension * convolution_factor.
    Ok(&left.coefficient_magnitude_bound * geometry.inner_dimension * factor)
}

pub fn right_action_gain(
    right: &MatrixState,
    geometry: ProductGeometry,
) -> Result<BigUint, BoundError> {
    let factor = convolution_factor(false, right.is_constant_polynomial, geometry)?;
    if geometry.inner_dimension == 0 {
        return Err(BoundError::ZeroInnerDimension);
    }
    // Symmetrically, this gain bounds the right action on a left error:
    // |e * R| <= |R| * inner_dimension * convolution_factor.
    Ok(&right.coefficient_magnitude_bound * geometry.inner_dimension * factor)
}

pub fn cap_by_centered_residue(value: BigUint, modulus: &BigInt) -> Result<BigUint, BoundError> {
    let cap = centered_residue_bound(modulus)?;
    Ok(value.min(cap))
}

pub fn max_abs_interval(minimum: &BigInt, maximum: &BigInt) -> BigUint {
    // Every value in a closed interval is bounded in magnitude by the larger
    // endpoint magnitude, including intervals entirely below zero.
    minimum.abs().max(maximum.abs()).to_biguint().unwrap_or_else(BigUint::zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::MatrixState;

    fn matrix(error: u64, magnitude: u64, constant: bool) -> MatrixState {
        MatrixState::new(error.into(), magnitude.into(), constant).unwrap()
    }

    #[test]
    fn zero_and_constant_products_have_expected_convolution_factor() {
        let geometry = ProductGeometry { inner_dimension: 3, ring_dimension: 8 };
        assert_eq!(
            product_bound(&0u8.into(), &99u8.into(), geometry, false, false).unwrap(),
            0u8.into()
        );
        assert_eq!(
            product_bound(&2u8.into(), &3u8.into(), geometry, true, false).unwrap(),
            18u8.into()
        );
        assert_eq!(
            product_bound(&2u8.into(), &3u8.into(), geometry, false, false).unwrap(),
            144u8.into()
        );
    }

    #[test]
    fn rectangular_inner_dimension_is_counted_once() {
        let geometry = ProductGeometry { inner_dimension: 5, ring_dimension: 4 };
        assert_eq!(
            product_bound(&2u8.into(), &7u8.into(), geometry, true, true).unwrap(),
            70u8.into()
        );
    }

    #[test]
    fn residue_bound_caps_represented_magnitude_but_not_error() {
        assert_eq!(centered_residue_bound(&BigInt::from(17)).unwrap(), 8u8.into());
        assert_eq!(cap_by_centered_residue(99u8.into(), &BigInt::from(17)).unwrap(), 8u8.into());
        let a = matrix(90, 5, true);
        let b = matrix(4, 7, true);
        let geometry = ProductGeometry { inner_dimension: 1, ring_dimension: 8 };
        let error =
            product_bound(&a.coefficient_magnitude_bound, &b.error_bound, geometry, true, false)
                .unwrap() +
                product_bound(
                    &a.error_bound,
                    &b.coefficient_magnitude_bound,
                    geometry,
                    false,
                    true,
                )
                .unwrap();
        assert_eq!(error, 650u64.into());
    }

    #[test]
    fn interval_magnitude_covers_negative_mixed_and_positive_ranges() {
        assert_eq!(max_abs_interval(&(-10).into(), &(-5).into()), 10u8.into());
        assert_eq!(max_abs_interval(&(-10).into(), &5.into()), 10u8.into());
        assert_eq!(max_abs_interval(&5.into(), &10.into()), 10u8.into());
    }

    #[test]
    fn two_term_product_identity_does_not_add_error_times_error() {
        let a = matrix(11, 13, false);
        let b = matrix(17, 19, false);
        let geometry = ProductGeometry { inner_dimension: 2, ring_dimension: 4 };
        // The exact identity is
        // actual(A)actual(B)-nominal(A)nominal(B)
        //   = actual(A)e_B + e_A nominal(B).
        let actual_a_times_error_b =
            product_bound(&a.coefficient_magnitude_bound, &b.error_bound, geometry, false, false)
                .unwrap();
        let error_a_times_nominal_b =
            product_bound(&a.error_bound, &b.coefficient_magnitude_bound, geometry, false, false)
                .unwrap();
        let error = &actual_a_times_error_b + &error_a_times_nominal_b;
        assert_eq!(error, BigUint::from(13u64 * 17 * 2 * 4 + 11 * 19 * 2 * 4));
        assert_eq!(error, actual_a_times_error_b + error_a_times_nominal_b);
        // actual(A)e_B already contains e_A e_B, so a third cross term would
        // double-count it rather than represent a new error contribution.
        assert!(error < BigUint::from(11u64 * 17 * 2 * 4 + 13 * 19 * 2 * 4 + 11 * 17));
    }

    #[test]
    fn action_gains_use_known_multiplier_constantness() {
        let geometry = ProductGeometry { inner_dimension: 2, ring_dimension: 8 };
        let left = matrix(0, 3, true);
        let right = matrix(0, 5, false);
        assert_eq!(left_action_gain(&left, geometry).unwrap(), 6u8.into());
        assert_eq!(right_action_gain(&right, geometry).unwrap(), 80u8.into());
    }
}
