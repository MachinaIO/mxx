//! Hard coefficient cutoffs and checks shared by samplers and runtimes.

use crate::{
    element::PolyElem,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use bigdecimal::BigDecimal;
use num_bigint::{BigUint, ToBigInt};
use num_traits::{FromPrimitive, Zero};
use rayon::prelude::*;

/// Returns the authoritative integer cutoff `floor(6.5 * sigma_bound)`.
///
/// `sigma_bound` is an exact rational upper bound supplied by configuration. Runtime sampling and
/// emitted correctness statements consume the resulting integer rather than independently
/// re-evaluating a floating-point sigma.
pub fn hard_cutoff_from_sigma_bound(sigma_bound: &BigDecimal) -> BigUint {
    assert!(*sigma_bound >= BigDecimal::zero(), "sigma bound must be nonnegative");
    (sigma_bound * BigDecimal::from(13u64) / BigDecimal::from(2u64))
        .to_bigint()
        .expect("nonnegative finite BigDecimal must convert to BigInt")
        .to_biguint()
        .expect("nonnegative cutoff must convert to BigUint")
}

/// Returns the authoritative minimum inclusive bound accepted by preimage
/// sampling for the supplied concrete sampler parameters.
pub fn default_preimage_cutoff(
    ring_dimension: u32,
    public_rows: usize,
    modulus_digits: usize,
    base: u32,
    sigma: f64,
) -> Option<BigUint> {
    if ring_dimension == 0 ||
        public_rows == 0 ||
        modulus_digits == 0 ||
        base == 0 ||
        !sigma.is_finite() ||
        sigma <= 0.0
    {
        return None;
    }
    let m_g = public_rows.checked_mul(modulus_digits)?.try_into().ok()?;
    let ring_dim_sqrt = BigDecimal::from_u32(ring_dimension)?.sqrt()?;
    let base = BigDecimal::from_u32(base)?;
    let preimage_sigma = compute_preimage_sigma(&ring_dim_sqrt, m_g, &base, None, Some(sigma));
    Some(hard_cutoff_from_sigma_bound(&preimage_sigma))
}

pub fn centered_coefficient_abs(value: &BigUint, modulus: &BigUint) -> BigUint {
    debug_assert!(value < modulus, "ring residue must be reduced");
    let negative_magnitude = modulus - value;
    value.min(&negative_magnitude).clone()
}

pub fn matrix_within_coefficient_bound<M: PolyMatrix>(
    matrix: &M,
    max_coefficient_bound: &BigUint,
) -> bool {
    let modulus = matrix.params().modulus().into();
    (0..matrix.row_size()).into_par_iter().all(|row| {
        (0..matrix.col_size()).all(|column| {
            matrix.entry(row, column).coeffs().into_iter().all(|coefficient| {
                centered_coefficient_abs(coefficient.value(), modulus.as_ref()) <=
                    *max_coefficient_bound
            })
        })
    })
}

pub fn compute_preimage_sigma(
    ring_dim_sqrt: &BigDecimal,
    m_g: u64,
    base: &BigDecimal,
    b_nrow: Option<usize>,
    sigma: Option<f64>,
) -> BigDecimal {
    let c_0 = BigDecimal::from_f64(1.8).unwrap();
    let c_1 = BigDecimal::from_f64(4.7).unwrap();
    let sigma = BigDecimal::from_f64(sigma.unwrap_or(4.578)).unwrap();
    let two_sqrt = BigDecimal::from(2).sqrt().unwrap();
    let m_g_sqrt = BigDecimal::from(m_g).sqrt().expect("sqrt(m_g) failed");
    let b_nrow = b_nrow.unwrap_or(1);
    let term = BigDecimal::from(b_nrow as u64).sqrt().unwrap() * ring_dim_sqrt.clone() * m_g_sqrt +
        two_sqrt * ring_dim_sqrt +
        c_1;
    c_0 * sigma.clone() * ((base + 1) * sigma) * term
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn hard_cutoff_uses_exact_floor_of_thirteen_halves_sigma() {
        assert_eq!(
            hard_cutoff_from_sigma_bound(&BigDecimal::from_str("4.578").unwrap()),
            BigUint::from(29u8)
        );
        assert_eq!(hard_cutoff_from_sigma_bound(&BigDecimal::zero()), BigUint::zero());
    }
}
