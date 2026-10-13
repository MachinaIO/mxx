//! Hard coefficient cutoffs and checks shared by samplers and runtimes.

use crate::{
    element::PolyElem,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use bigdecimal::BigDecimal;
use mxx_ir_core::{Rational, types::TrapdoorInstantiation};
use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use rayon::prelude::*;
use thiserror::Error;

/// Constant of the heuristic singular-value estimate `s_1(X) ≈ C·σ·(√rows + √columns + C')` of a
/// Gaussian matrix of width `σ` (MP12, GM18). The same constants size every trapdoor preimage
/// width.
pub const SPECTRAL_CONSTANT: f64 = 1.8;
/// Additive constant `C'` of the singular-value estimate.
pub const SPECTRAL_OFFSET: f64 = 4.7;

/// Selects the hard coefficient cutoff for a trapdoor-sampled preimage.
///
/// This policy applies only to randomized trapdoor preimage sampling. A
/// deterministic gadget decomposition has a different, exact digit bound and
/// must not be resolved through this policy.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum TrapdoorPreimageCutoffPolicy {
    /// Use the default cutoff `floor(6.5 σ_pre)` of the requested preimage width `σ_pre`.
    #[default]
    Default,
    /// Use an explicitly selected cutoff, provided it is not below the default cutoff of the
    /// requested preimage width.
    Explicit(BigUint),
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum TrapdoorPreimageCutoffError {
    #[error("invalid trapdoor preimage sampler parameters")]
    InvalidParameters,
    #[error("explicit trapdoor preimage cutoff {selected} is below the default {minimum}")]
    ExplicitBelowDefault { selected: BigUint, minimum: BigUint },
    #[error("requested preimage width {requested} is below the trapdoor's minimum {minimum}")]
    WidthBelowMinimum { requested: BigDecimal, minimum: BigDecimal },
}

impl From<TrapdoorPreimageCutoffError> for crate::matrix::SmallMatrixError {
    fn from(error: TrapdoorPreimageCutoffError) -> Self {
        match error {
            TrapdoorPreimageCutoffError::InvalidParameters => Self::InvalidConfig,
            TrapdoorPreimageCutoffError::ExplicitBelowDefault { selected, minimum } => {
                Self::PreimageBoundTooSmall { requested: selected, minimum }
            }
            TrapdoorPreimageCutoffError::WidthBelowMinimum { requested, minimum } => {
                Self::PreimageWidthTooSmall { requested, minimum }
            }
        }
    }
}

impl TrapdoorPreimageCutoffPolicy {
    /// Resolves the cutoff of preimages sampled with width `preimage_sigma`.
    pub fn resolve(
        &self,
        preimage_sigma: &Rational,
    ) -> Result<BigUint, TrapdoorPreimageCutoffError> {
        let minimum = hard_cutoff_from_rational(preimage_sigma)
            .ok_or(TrapdoorPreimageCutoffError::InvalidParameters)?;
        match self {
            Self::Default => Ok(minimum),
            Self::Explicit(selected) if selected < &minimum => {
                Err(TrapdoorPreimageCutoffError::ExplicitBelowDefault {
                    selected: selected.clone(),
                    minimum,
                })
            }
            Self::Explicit(selected) => Ok(selected.clone()),
        }
    }
}

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

/// [`hard_cutoff_from_sigma_bound`] of an exact rational width: `floor(13 p / (2 q))` for
/// `σ = p / q`, or `None` for a negative width.
pub fn hard_cutoff_from_rational(sigma: &Rational) -> Option<BigUint> {
    (sigma.numerator() * 13u8 / (sigma.denominator() * 2u8)).to_biguint()
}

/// The exact rational value of a finite decimal.
pub fn rational_from_decimal(value: &BigDecimal) -> Rational {
    let (integer, exponent) = value.as_bigint_and_exponent();
    let power = BigInt::from(10u8).pow(exponent.unsigned_abs() as u32);
    let (numerator, denominator) =
        if exponent < 0 { (integer * power, BigInt::one()) } else { (integer, power) };
    Rational::new(numerator, denominator).expect("a power of ten is nonzero")
}

/// The decimal value of an exact rational, rounded to the default `BigDecimal` precision.
pub fn decimal_from_rational(value: &Rational) -> BigDecimal {
    BigDecimal::from(value.numerator().clone()) / BigDecimal::from(value.denominator().clone())
}

/// Width `c_G = (b + 1) σ` of the gadget sampler of a trapdoor with TrapGen width `σ` and gadget
/// base `b`.
pub fn gadget_sampler_sigma(base: u32, sigma: f64) -> f64 {
    (base as f64 + 1.0) * sigma
}

/// The heuristic estimate `ŝ_1` of the largest singular value of the top block `T'` of the
/// trapdoor `[T'; I]` with TrapGen width `σ`, `n` the ring dimension and `D` the digits per row:
///
/// - `Computational` (`T' = [R; E]`, `2·rows x rows·D`): `1.8 σ (√(n·rows·D) + √(2n) + 4.7)`, the
///   estimate of the existing sampler (the `√(2n)` term is not scaled by `rows`).
/// - `Statistical { m_bar }` (`T`, `m_bar x D`, one row): `1.8 σ (√(n D) + √(n m_bar) + 4.7)`.
///
/// Returns `None` for invalid parameters, including a statistical trapdoor with other than one
/// row.
pub fn trapdoor_singular_value_estimate(
    ring_dimension: u32,
    public_rows: usize,
    modulus_digits: usize,
    sigma: f64,
    instantiation: TrapdoorInstantiation,
) -> Option<BigDecimal> {
    if ring_dimension == 0 ||
        public_rows == 0 ||
        modulus_digits == 0 ||
        !sigma.is_finite() ||
        sigma <= 0.0
    {
        return None;
    }
    let ring_dim_sqrt = BigDecimal::from_u32(ring_dimension)?.sqrt()?;
    let m_g = BigDecimal::from(u64::try_from(public_rows.checked_mul(modulus_digits)?).ok()?);
    let other = match instantiation {
        TrapdoorInstantiation::Computational => BigDecimal::from(2u8),
        TrapdoorInstantiation::Statistical { m_bar } if public_rows == 1 && m_bar > 0 => {
            BigDecimal::from(u64::try_from(m_bar).ok()?)
        }
        TrapdoorInstantiation::Statistical { .. } => return None,
    };
    let term = ring_dim_sqrt.clone() * m_g.sqrt()? +
        other.sqrt()? * ring_dim_sqrt +
        BigDecimal::from_f64(SPECTRAL_OFFSET)?;
    Some(BigDecimal::from_f64(SPECTRAL_CONSTANT)? * BigDecimal::from_f64(sigma)? * term)
}

/// The smallest preimage width (a standard deviation) the trapdoor sampler accepts.
///
/// With the gadget sampler width `c_G = (b + 1) σ` and the estimate `ŝ_1` of
/// [`trapdoor_singular_value_estimate`]:
///
/// - `Computational`: `c_G ŝ_1 = 1.8 (b + 1) σ² (√(n·rows·D) + √(2n) + 4.7)`, the width of the
///   existing sampler.
/// - `Statistical`: `√(c_G² (1 + 2 ŝ_1²) + r²)` with the rounding width `r = σ`, about `√2` times
///   `c_G ŝ_1`. It makes the isotropic part `s² - r² - 2 α ŝ_1²` of the matrix-free perturbation
///   sampler nonnegative for `α = c_G² s² / (s² - c_G²)`.
///
/// The estimate is heuristic in the same way as the existing sampler's.
pub fn minimum_preimage_sigma(
    ring_dimension: u32,
    public_rows: usize,
    modulus_digits: usize,
    base: u32,
    sigma: f64,
    instantiation: TrapdoorInstantiation,
) -> Option<BigDecimal> {
    if base < 2 {
        return None;
    }
    let s1 = trapdoor_singular_value_estimate(
        ring_dimension,
        public_rows,
        modulus_digits,
        sigma,
        instantiation,
    )?;
    let sigma = BigDecimal::from_f64(sigma)?;
    let c = (BigDecimal::from(base) + BigDecimal::from(1u8)) * sigma.clone();
    match instantiation {
        TrapdoorInstantiation::Computational => {
            Some(sigma * (BigDecimal::from(base) + BigDecimal::from(1u8)) * s1)
        }
        TrapdoorInstantiation::Statistical { .. } => {
            let variance = c.square() *
                (BigDecimal::from(1u8) + BigDecimal::from(2u8) * s1.square()) +
                sigma.square();
            variance.sqrt()
        }
    }
}

/// [`minimum_preimage_sigma`] as an exact rational graph expression, or `None` for invalid
/// parameters.
pub fn minimum_preimage_sigma_expr(
    ring_dimension: u32,
    public_rows: usize,
    modulus_digits: usize,
    base: u32,
    sigma: f64,
    instantiation: TrapdoorInstantiation,
) -> Option<mxx_ir_core::RealExpr> {
    minimum_preimage_sigma(ring_dimension, public_rows, modulus_digits, base, sigma, instantiation)
        .map(|width| mxx_ir_core::RealExpr::Rational(rational_from_decimal(&width)))
}

/// Returns the default hard cutoff, `floor(6.5 σ_min)`, of a preimage sampled with the minimum
/// width [`minimum_preimage_sigma`].
pub fn default_preimage_cutoff(
    ring_dimension: u32,
    public_rows: usize,
    modulus_digits: usize,
    base: u32,
    sigma: f64,
    instantiation: TrapdoorInstantiation,
) -> Option<BigUint> {
    minimum_preimage_sigma(ring_dimension, public_rows, modulus_digits, base, sigma, instantiation)
        .map(|width| hard_cutoff_from_sigma_bound(&width))
}

/// Checks a requested preimage width and cutoff against a trapdoor: the width is at least
/// [`minimum_preimage_sigma`], and the cutoff at least the default cutoff `floor(6.5 σ_pre)` of the
/// width.
#[allow(clippy::too_many_arguments)]
pub fn validate_preimage_width(
    ring_dimension: u32,
    public_rows: usize,
    modulus_digits: usize,
    base: u32,
    sigma: f64,
    instantiation: TrapdoorInstantiation,
    preimage_sigma: &Rational,
    cutoff: &BigUint,
) -> Result<(), TrapdoorPreimageCutoffError> {
    let minimum = minimum_preimage_sigma(
        ring_dimension,
        public_rows,
        modulus_digits,
        base,
        sigma,
        instantiation,
    )
    .ok_or(TrapdoorPreimageCutoffError::InvalidParameters)?;
    if BigDecimal::from(preimage_sigma.numerator().clone()) <
        minimum.clone() * BigDecimal::from(preimage_sigma.denominator().clone())
    {
        return Err(TrapdoorPreimageCutoffError::WidthBelowMinimum {
            requested: decimal_from_rational(preimage_sigma),
            minimum,
        });
    }
    TrapdoorPreimageCutoffPolicy::Explicit(cutoff.clone()).resolve(preimage_sigma).map(|_| ())
}

/// The width a preimage is sampled with: `preimage_sigma`, or the minimum width when it is
/// `None`, after [`validate_preimage_width`] of it and `cutoff`.
#[allow(clippy::too_many_arguments)]
pub fn checked_preimage_sigma(
    ring_dimension: u32,
    public_rows: usize,
    modulus_digits: usize,
    base: u32,
    sigma: f64,
    instantiation: TrapdoorInstantiation,
    preimage_sigma: Option<&Rational>,
    cutoff: &BigUint,
) -> Result<f64, TrapdoorPreimageCutoffError> {
    let width = match preimage_sigma {
        Some(width) => width.clone(),
        None => rational_from_decimal(
            &minimum_preimage_sigma(
                ring_dimension,
                public_rows,
                modulus_digits,
                base,
                sigma,
                instantiation,
            )
            .ok_or(TrapdoorPreimageCutoffError::InvalidParameters)?,
        ),
    };
    validate_preimage_width(
        ring_dimension,
        public_rows,
        modulus_digits,
        base,
        sigma,
        instantiation,
        &width,
        cutoff,
    )?;
    decimal_from_rational(&width)
        .to_f64()
        .filter(|width| width.is_finite())
        .ok_or(TrapdoorPreimageCutoffError::InvalidParameters)
}

/// Whether integer samples bounded by `cutoff` are represented exactly by the samplers: below
/// `2^52`, the integers a binary64 carries exactly with margin, and below `q / 2`, so the centered
/// residue recovers the integer.
pub fn cutoff_is_representable(cutoff: &BigUint, modulus: &BigUint) -> bool {
    cutoff < &(BigUint::one() << 52u32) && cutoff * 2u8 < *modulus
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

    #[test]
    fn trapdoor_preimage_cutoff_policy_rejects_values_below_default() {
        let width = Rational::new(BigInt::from(9), BigInt::from(2)).unwrap();
        let minimum = TrapdoorPreimageCutoffPolicy::Default.resolve(&width).unwrap();
        assert_eq!(minimum, BigUint::from(29u8));
        assert_eq!(
            TrapdoorPreimageCutoffPolicy::Explicit(&minimum - BigUint::from(1_u8)).resolve(&width),
            Err(TrapdoorPreimageCutoffError::ExplicitBelowDefault {
                selected: &minimum - BigUint::from(1_u8),
                minimum: minimum.clone(),
            })
        );
        assert_eq!(
            TrapdoorPreimageCutoffPolicy::Explicit(minimum.clone()).resolve(&width),
            Ok(minimum)
        );
    }

    #[test]
    fn computational_minimum_width_is_the_existing_formula() {
        let (n, rows, digits, base, sigma) = (16u32, 2usize, 6usize, 8u32, 4.578f64);
        let width = minimum_preimage_sigma(
            n,
            rows,
            digits,
            base,
            sigma,
            TrapdoorInstantiation::Computational,
        )
        .unwrap();
        let expected = 1.8 *
            (base as f64 + 1.0) *
            sigma *
            sigma *
            (((rows * digits * n as usize) as f64).sqrt() + ((2 * n) as f64).sqrt() + 4.7);
        assert!((width.to_f64().unwrap() - expected).abs() < 1e-9 * expected);
    }

    #[test]
    fn statistical_minimum_width_admits_the_perturbation_decomposition() {
        let (n, digits, base, sigma, m_bar) = (16u32, 4usize, 4u32, 4.578f64, 7usize);
        let instantiation = TrapdoorInstantiation::Statistical { m_bar };
        let s1 = trapdoor_singular_value_estimate(n, 1, digits, sigma, instantiation)
            .unwrap()
            .to_f64()
            .unwrap();
        let expected_s1 = 1.8 * sigma * ((16.0f64 * 4.0).sqrt() + (16.0f64 * 7.0).sqrt() + 4.7);
        assert!((s1 - expected_s1).abs() < 1e-9 * expected_s1);
        let s = minimum_preimage_sigma(n, 1, digits, base, sigma, instantiation)
            .unwrap()
            .to_f64()
            .unwrap();
        let c = gadget_sampler_sigma(base, sigma);
        let alpha = c * c * s * s / (s * s - c * c);
        assert!(s * s - sigma * sigma - 2.0 * alpha * s1 * s1 > -1e-6 * s * s);
        assert!(s > c * s1);
        assert!(minimum_preimage_sigma(n, 2, digits, base, sigma, instantiation).is_none());
    }

    #[test]
    fn requested_width_and_cutoff_are_checked_against_the_trapdoor() {
        let instantiation = TrapdoorInstantiation::Computational;
        let minimum = minimum_preimage_sigma(16, 1, 4, 4, 4.578, instantiation).unwrap();
        let width = rational_from_decimal(&minimum);
        let cutoff = hard_cutoff_from_sigma_bound(&minimum);
        assert_eq!(
            validate_preimage_width(16, 1, 4, 4, 4.578, instantiation, &width, &cutoff),
            Ok(())
        );
        let below =
            Rational::new(width.numerator() - BigInt::one(), width.denominator().clone()).unwrap();
        assert!(matches!(
            validate_preimage_width(16, 1, 4, 4, 4.578, instantiation, &below, &cutoff),
            Err(TrapdoorPreimageCutoffError::WidthBelowMinimum { .. })
        ));
        assert!(matches!(
            validate_preimage_width(
                16,
                1,
                4,
                4,
                4.578,
                instantiation,
                &width,
                &(&cutoff - BigUint::one())
            ),
            Err(TrapdoorPreimageCutoffError::ExplicitBelowDefault { .. })
        ));
    }

    #[test]
    fn representable_cutoffs_stay_below_two_to_the_fifty_two_and_half_the_modulus() {
        let modulus = BigUint::one() << 60u32;
        assert!(cutoff_is_representable(&((BigUint::one() << 52u32) - 1u8), &modulus));
        assert!(!cutoff_is_representable(&(BigUint::one() << 52u32), &modulus));
        assert!(cutoff_is_representable(&BigUint::from(49u8), &BigUint::from(99u8)));
        assert!(!cutoff_is_representable(&BigUint::from(50u8), &BigUint::from(99u8)));
    }
}
