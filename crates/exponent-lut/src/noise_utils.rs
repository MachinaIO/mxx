//! Exact noise arithmetic shared by encoding schemes.
//!
//! These values have no implicit ciphertext coordinate units. Coordinate
//! conventions belong to the scheme adapter that consumes them.
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;

use crate::program::{FamilyRange, LutId, WireId};

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum NoiseArithmeticError {
    #[error("rational denominator must be positive")]
    ZeroDenominator,
}

/// Exact nonnegative rational. Coordinate units belong to its owning noise trace.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NonnegativeRational {
    pub numerator: BigUint,
    pub denominator: BigUint,
}

impl NonnegativeRational {
    pub fn new(numerator: BigUint, denominator: BigUint) -> Result<Self, NoiseArithmeticError> {
        if denominator.is_zero() {
            return Err(NoiseArithmeticError::ZeroDenominator);
        }
        let divisor = gcd(&numerator, &denominator);
        Ok(Self { numerator: numerator / &divisor, denominator: denominator / divisor })
    }

    pub fn zero() -> Self {
        Self { numerator: BigUint::zero(), denominator: BigUint::one() }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            &self.numerator * &other.denominator + &other.numerator * &self.denominator,
            &self.denominator * &other.denominator,
        )
        .expect("positive variance denominator")
    }

    pub fn scaled(&self, gain: &BigUint) -> Self {
        Self::new(&self.numerator * gain, self.denominator.clone())
            .expect("positive variance denominator")
    }

    /// The least multiple of `2^-fraction_bits` not below this value, when
    /// the denominator is longer than `fraction_bits`; otherwise unchanged.
    /// An outward bound that keeps propagated denominators short.
    pub fn rounded_up(&self, fraction_bits: u32) -> Self {
        if self.denominator.bits() <= u64::from(fraction_bits) {
            return self.clone();
        }
        let scaled = &self.numerator << fraction_bits;
        let quotient = &scaled / &self.denominator;
        let numerator =
            if &quotient * &self.denominator == scaled { quotient } else { quotient + 1u8 };
        Self::new(numerator, BigUint::one() << fraction_bits).expect("positive power of two")
    }

    /// Scale this exact rational by a (possibly fractional) nonnegative
    /// rational. Modulus-alignment transfers use this for the `1/alpha`
    /// action; no floating-point approximation is involved.
    pub fn scaled_ratio(&self, numerator: &BigUint, denominator: &BigUint) -> Self {
        assert!(!denominator.is_zero(), "a rational scale cannot have zero denominator");
        Self::new(&self.numerator * numerator, &self.denominator * denominator)
            .expect("positive rational scale denominator")
    }

    /// Bound a heterogeneous sum without assuming independence. Zero-variance
    /// terms introduce no covariance, including when their means are nonzero.
    /// Compose pairwise Cauchy bounds and cap outward-rounding accumulation by
    /// the Cauchy bound r * sum(V_i), counting only nonzero variances. Equal
    /// variances and rational proportional standard deviations remain exact.
    pub fn sum_variance_bounds<'a>(variances: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut total = Self::zero();
        let mut bound = Self::zero();
        let mut count = 0usize;
        for variance in variances {
            if variance.numerator.is_zero() {
                continue;
            }
            total = total.add(variance);
            bound = bound.sum_variance_bound(variance);
            count += 1;
        }
        let cap = total.scaled(&BigUint::from(count));
        if &bound.numerator * &cap.denominator <= &cap.numerator * &bound.denominator {
            bound
        } else {
            cap
        }
    }

    /// Upper bound on the variance of a sum with unknown covariance.
    /// Cauchy--Schwarz gives V(X+Y) <= V(X)+V(Y)+2 sqrt(V(X)V(Y)).
    /// Round the square root outward in the common rational denominator;
    /// cap that rounding by 2(V(X)+V(Y)), which is always another valid bound.
    /// Equal variances retain the exact coherent bound 4V, and a zero-variance
    /// operand leaves the other variance unchanged. No independence is assumed.
    pub fn sum_variance_bound(&self, other: &Self) -> Self {
        let denominator = &self.denominator * &other.denominator;
        let left = &self.numerator * &other.denominator;
        let right = &other.numerator * &self.denominator;
        let product = &left * &right;
        let floor = product.sqrt();
        let ceil = if &floor * &floor == product { floor } else { floor + 1u8 };
        let sum = left + right;
        let numerator = (&sum + ceil * 2u8).min(&sum * 2u8);
        Self::new(numerator, denominator).expect("positive variance denominator")
    }
}

#[cfg(test)]
mod covariance_tests {
    use super::*;

    #[test]
    fn tail_envelope_rounds_the_combined_mean_and_deviation_once() {
        let moments = NoiseMoments::uniform_interval((-1).into(), 4u8.into());
        assert_eq!(moments.tail_bound(&NonnegativeRational::zero()), 1u8.into());
        assert_eq!(
            moments.tail_bound(&NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap()),
            2u8.into()
        );
        assert_eq!(
            moments.tail_bound(&NonnegativeRational::new(4u8.into(), 1u8.into()).unwrap()),
            3u8.into()
        );
    }

    #[test]
    fn test_mean_enclosure_preserves_native_bias_and_linear_images() {
        let native = NoiseMoments::uniform_interval((-1).into(), 4u8.into());
        assert_eq!(native.mean_radius, NonnegativeRational::zero());
        let doubled = native.add(&native);
        assert_eq!(doubled.mean_numerator, BigInt::one());
        assert_eq!(doubled.mean_denominator, BigUint::one());
        assert_eq!(doubled, native.scaled(&2u8.into()));
        let image = native.linear_image(
            &NonnegativeRational::new(3u8.into(), 1u8.into()).unwrap(),
            native.variance.scaled(&9u8.into()),
        );
        // The wrapped/nonwrapped coefficients of multiplication by 2-X
        // contain 2*d0+d1 or 2*d0-d1. Enumerate the actual finite digit law.
        for sign in [-1i64, 1] {
            let values = (-1i64..=2)
                .flat_map(|a| (-1i64..=2).map(move |b| 2 * a + sign * b))
                .collect::<Vec<_>>();
            let sum = values.iter().sum::<i64>();
            let squares = values.iter().map(|x| x * x).sum::<i64>();
            let count = values.len() as i64;
            assert!(
                BigUint::from(sum.unsigned_abs()) * &image.mean_radius.denominator <=
                    &image.mean_radius.numerator * BigUint::from(count as u64)
            );
            let variance_numerator = BigUint::from((squares * count - sum * sum) as u64);
            assert!(
                variance_numerator * &image.variance.denominator <=
                    &image.variance.numerator * BigUint::from((count * count) as u64)
            );
        }
        let zero = native.linear_image(&NonnegativeRational::zero(), NonnegativeRational::zero());
        assert!(zero.mean_radius.numerator.is_zero());
        assert!(zero.variance.numerator.is_zero());
    }

    #[test]
    fn test_moment_clearance_includes_bias_and_strict_boundary() {
        let unit = NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap();
        let moments = NoiseMoments {
            mean_numerator: (-1).into(),
            mean_denominator: 2u8.into(),
            mean_radius: NonnegativeRational::new(1u8.into(), 4u8.into()).unwrap(),
            variance: NonnegativeRational::new(9u8.into(), 16u8.into()).unwrap(),
        };
        // |mean| + radius + sigma = 1/2 + 1/4 + 3/4 = 3/2.
        assert!(!moments.fits(&NonnegativeRational::new(3u8.into(), 2u8.into()).unwrap(), &unit));
        assert!(moments.fits(&NonnegativeRational::new(7u8.into(), 4u8.into()).unwrap(), &unit));
        assert!(!moments.fits(&NonnegativeRational::new(1u8.into(), 2u8.into()).unwrap(), &unit));
        let deterministic = NoiseMoments { variance: NonnegativeRational::zero(), ..moments };
        assert!(
            !deterministic.fits(&NonnegativeRational::new(3u8.into(), 4u8.into()).unwrap(), &unit)
        );
        assert!(deterministic.fits(&unit, &unit));
    }

    #[test]
    fn test_sum_variance_bound_preserves_coherent_and_zero_variance() {
        let variance = NonnegativeRational::new(7u8.into(), 3u8.into()).unwrap();
        assert_eq!(variance.sum_variance_bound(&variance), variance.scaled(&4u8.into()));
        assert_eq!(variance.sum_variance_bound(&NonnegativeRational::zero()), variance);
        assert_eq!(NonnegativeRational::zero().sum_variance_bound(&variance), variance);
        let amplified = variance.scaled(&9u8.into());
        assert_eq!(variance.sum_variance_bound(&amplified), variance.scaled(&16u8.into()));
    }

    #[test]
    fn test_family_variance_bound_preserves_shared_sources_and_constant_terms() {
        let variance = NonnegativeRational::new(7u8.into(), 3u8.into()).unwrap();
        let zero = NonnegativeRational::zero();
        assert_eq!(NonnegativeRational::sum_variance_bounds([]), zero);
        assert_eq!(NonnegativeRational::sum_variance_bounds([&zero, &variance, &zero]), variance);
        assert_eq!(
            NonnegativeRational::sum_variance_bounds([&variance; 3]),
            variance.scaled(&9u8.into())
        );
        // If all errors are multiples of the same E, the tight worst
        // covariance is attained: Var(E + 2E + 3E) = 36 Var(E).
        let twice = variance.scaled(&4u8.into());
        let thrice = variance.scaled(&9u8.into());
        assert_eq!(
            NonnegativeRational::sum_variance_bounds([&variance, &twice, &zero, &thrice]),
            variance.scaled(&36u8.into())
        );
        assert_eq!(
            NonnegativeRational::sum_variance_bounds([&thrice, &zero, &twice, &variance]),
            variance.scaled(&36u8.into())
        );
    }

    #[test]
    fn test_sum_variance_bound_rounds_covariance_outward() {
        for left in 1u32..12 {
            for right in 1u32..12 {
                let x = NonnegativeRational::new(left.into(), 3u8.into()).unwrap();
                let y = NonnegativeRational::new(right.into(), 5u8.into()).unwrap();
                let bound = x.sum_variance_bound(&y);
                let sum = x.add(&y);
                // Verify the defining Cauchy inequality exactly after clearing
                // denominators; no floating-point reference or tail model.
                let excess =
                    &bound.numerator * &sum.denominator - &sum.numerator * &bound.denominator;
                let excess_denominator = &bound.denominator * &sum.denominator;
                assert!(
                    &excess * &excess * &x.denominator * &y.denominator >=
                        &x.numerator *
                            &y.numerator *
                            4u8 *
                            &excess_denominator *
                            &excess_denominator
                );
                assert!(
                    &bound.numerator * &sum.denominator <=
                        &sum.numerator * 2u8 * &bound.denominator
                );
            }
        }
    }

    #[test]
    fn discrete_rounding_keeps_even_ratio_bias_and_exact_variance() {
        let odd = NoiseMoments::discrete_rounding_residue(&5u8.into()).unwrap();
        assert_eq!(odd.mean_numerator, BigInt::zero());
        assert_eq!(odd.variance, NonnegativeRational::new(2u8.into(), 25u8.into()).unwrap());

        let even = NoiseMoments::discrete_rounding_residue(&4u8.into()).unwrap();
        assert_eq!(even.mean_numerator, BigInt::one());
        assert_eq!(even.mean_denominator, BigUint::from(8u8));
        assert_eq!(even.variance, NonnegativeRational::new(5u8.into(), 64u8.into()).unwrap());
        assert_eq!(
            NoiseMoments::discrete_rounding_residue(&1u8.into()).unwrap().variance,
            NonnegativeRational::zero()
        );
    }

    #[test]
    fn continuous_rounding_is_explicit_and_not_discrete_default() {
        let continuous = NoiseMomentPolicy::Continuous.rounding_moments().unwrap();
        let discrete =
            NoiseMomentPolicy::Discrete { alpha: 4u8.into() }.rounding_moments().unwrap();
        assert_eq!(continuous.variance, NonnegativeRational::new(1u8.into(), 12u8.into()).unwrap());
        assert_ne!(continuous, discrete);
    }

    #[test]
    fn second_moment_keeps_nonzero_mean_and_interval_radius() {
        let moments = NoiseMoments {
            mean_numerator: 1.into(),
            mean_denominator: 2u8.into(),
            mean_radius: NonnegativeRational::new(1u8.into(), 4u8.into()).unwrap(),
            variance: NonnegativeRational::new(1u8.into(), 4u8.into()).unwrap(),
        };
        assert_eq!(
            moments.second_moment_bound(),
            NonnegativeRational::new(13u8.into(), 16u8.into()).unwrap()
        );
    }
}

/// Physical moment enclosure, conditional on the caller's stated distribution
/// model. Each coefficient mean lies in center +/- mean_radius; a zero radius
/// records an exact scalar mean. Variance is a centered variance bound, not a
/// second moment. Neither field implies independence or a tail law.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseMoments {
    pub mean_numerator: BigInt,
    pub mean_denominator: BigUint,
    pub mean_radius: NonnegativeRational,
    pub variance: NonnegativeRational,
}

impl NoiseMoments {
    /// Outward bound with denominators of at most `fraction_bits` bits: a
    /// long mean center joins the mean radius, and the radius and variance
    /// round up. Short exact moments are unchanged.
    pub fn rounded_up(&self, fraction_bits: u32) -> Self {
        let limit = u64::from(fraction_bits);
        if self.mean_denominator.bits() <= limit &&
            self.mean_radius.denominator.bits() <= limit &&
            self.variance.denominator.bits() <= limit
        {
            return self.clone();
        }
        let (mean_numerator, mean_denominator, mean_radius) =
            if self.mean_denominator.bits() <= limit {
                (
                    self.mean_numerator.clone(),
                    self.mean_denominator.clone(),
                    self.mean_radius.rounded_up(fraction_bits),
                )
            } else {
                let center = NonnegativeRational::new(
                    self.mean_numerator.magnitude().clone(),
                    self.mean_denominator.clone(),
                )
                .expect("positive mean denominator");
                (
                    BigInt::zero(),
                    BigUint::one(),
                    center.add(&self.mean_radius).rounded_up(fraction_bits),
                )
            };
        Self {
            mean_numerator,
            mean_denominator,
            mean_radius,
            variance: self.variance.rounded_up(fraction_bits),
        }
    }

    /// Exact moments of the centered nearest-rounding residue for an integer
    /// modulus ratio. The support is `-floor(alpha/2)..ceil(alpha/2)-1`, and
    /// the residue is divided by `alpha`. Even ratios retain their real tie
    /// bias (`1/(2*alpha)`) instead of silently using a continuous `1/12`
    /// approximation.
    pub fn discrete_rounding_residue(alpha: &BigUint) -> Result<Self, NoiseArithmeticError> {
        if alpha.is_zero() {
            return Err(NoiseArithmeticError::ZeroDenominator);
        }
        if *alpha == BigUint::one() {
            return Ok(Self {
                mean_numerator: BigInt::zero(),
                mean_denominator: BigUint::one(),
                mean_radius: NonnegativeRational::zero(),
                variance: NonnegativeRational::zero(),
            });
        }
        let alpha_squared = alpha.pow(2u32);
        let variance =
            NonnegativeRational::new(&alpha_squared - 1u8, BigUint::from(12u8) * &alpha_squared)
                .expect("positive discrete rounding variance");
        let (mean_numerator, mean_denominator) = if !alpha.bit(0) {
            // The centered integer support has sum -alpha/2. The rounding
            // residue is its negative divided by alpha.
            (BigInt::one(), BigUint::from(2u8) * alpha)
        } else {
            (BigInt::zero(), BigUint::one())
        };
        Ok(Self {
            mean_numerator,
            mean_denominator,
            mean_radius: NonnegativeRational::zero(),
            variance,
        })
    }

    /// Continuous centered-uniform approximation on `[-1/2, 1/2]`.
    /// Callers must select this explicitly; integer modulus alignment uses
    /// [`Self::discrete_rounding_residue`] when exact support is available.
    pub fn continuous_rounding_residue() -> Self {
        Self {
            mean_numerator: BigInt::zero(),
            mean_denominator: BigUint::one(),
            mean_radius: NonnegativeRational::zero(),
            variance: NonnegativeRational::new(1u8.into(), 12u8.into())
                .expect("positive continuous rounding denominator"),
        }
    }

    /// Outward integer envelope for the caller's explicit tail model.
    pub fn tail_bound(&self, z_squared: &NonnegativeRational) -> BigUint {
        let bias = NonnegativeRational::new(
            self.mean_numerator.magnitude().clone(),
            self.mean_denominator.clone(),
        )
        .expect("positive mean denominator")
        .add(&self.mean_radius);
        let numerator = &z_squared.numerator * &self.variance.numerator;
        let denominator = &z_squared.denominator * &self.variance.denominator;
        let squared = (&numerator + &denominator - 1u8) / denominator;
        let floor = squared.sqrt();
        let tail = if &floor * &floor == squared { floor } else { floor + 1u8 };
        let upper = (&bias.numerator + &bias.denominator - 1u8) / &bias.denominator + tail;
        if upper.is_zero() {
            return upper;
        }
        // The separate ceilings exceed ceil(bias + sqrt(V*z^2)) by at most
        // one. Test that final integer exactly, without a floating-point tail.
        let lower = &upper - 1u8;
        let scaled = &lower * &bias.denominator;
        if scaled >= bias.numerator &&
            (&scaled - &bias.numerator).pow(2) *
                &z_squared.denominator *
                &self.variance.denominator >=
                &z_squared.numerator * &self.variance.numerator * bias.denominator.pow(2)
        {
            lower
        } else {
            upper
        }
    }

    /// Strict clearance for a mean interval plus a caller-selected tail
    /// multiplier: |center| + radius + sqrt(z_squared * variance) < limit.
    /// The multiplier's distribution assumption and event budget belong to
    /// the audit, not to this exact rational comparison.
    pub fn fits(&self, limit: &NonnegativeRational, z_squared: &NonnegativeRational) -> bool {
        let bias = NonnegativeRational::new(
            self.mean_numerator.magnitude().clone(),
            self.mean_denominator.clone(),
        )
        .expect("positive mean denominator")
        .add(&self.mean_radius);
        let available = &limit.numerator * &bias.denominator;
        let occupied = &bias.numerator * &limit.denominator;
        if available <= occupied {
            return false;
        }
        let margin = available - occupied;
        let denominator = &limit.denominator * &bias.denominator;
        margin.pow(2) * &z_squared.denominator * &self.variance.denominator >
            &z_squared.numerator * &self.variance.numerator * denominator.pow(2)
    }

    /// Conservative centered-plus-bias second-moment enclosure. The stored
    /// variance is centered; reporting only it would drop a nonzero mean or
    /// interval radius from an MSE calculation.
    pub fn second_moment_bound(&self) -> NonnegativeRational {
        let bias = NonnegativeRational::new(
            self.mean_numerator.magnitude().clone(),
            self.mean_denominator.clone(),
        )
        .expect("positive mean denominator")
        .add(&self.mean_radius);
        let bias_squared =
            NonnegativeRational::new(bias.numerator.pow(2u32), bias.denominator.pow(2u32))
                .expect("positive bias denominator");
        self.variance.add(&bias_squared)
    }

    /// Enclose a linear image with a known row-l1 bound. The caller derives
    /// its centered variance transfer under the applicable coefficient model.
    /// A uniform scalar mean need not stay uniform after negacyclic wrapping;
    /// retain its image as an interval instead of assigning it to every output.
    pub fn linear_image(&self, l1: &NonnegativeRational, variance: NonnegativeRational) -> Self {
        let center = NonnegativeRational::new(
            self.mean_numerator.magnitude().clone(),
            self.mean_denominator.clone(),
        )
        .expect("positive mean denominator");
        let magnitude = center.add(&self.mean_radius);
        Self {
            mean_numerator: BigInt::zero(),
            mean_denominator: BigUint::one(),
            mean_radius: NonnegativeRational::new(
                magnitude.numerator * &l1.numerator,
                magnitude.denominator * &l1.denominator,
            )
            .expect("positive mean and action denominators"),
            variance,
        }
    }

    /// Apply a positive scalar exactly to this mean interval and supply the
    /// centered variance of its linear image.
    pub fn affine_scale(
        &self,
        factor: &NonnegativeRational,
        variance: NonnegativeRational,
    ) -> Self {
        let mean_numerator = &self.mean_numerator * BigInt::from(factor.numerator.clone());
        let mean_denominator = &self.mean_denominator * &factor.denominator;
        let divisor = gcd(mean_numerator.magnitude(), &mean_denominator);
        Self {
            mean_numerator: mean_numerator / BigInt::from(divisor.clone()),
            mean_denominator: mean_denominator / divisor,
            mean_radius: self.mean_radius.scaled_ratio(&factor.numerator, &factor.denominator),
            variance,
        }
    }

    /// Negate this enclosed error, preserving its signed mean and centered
    /// variance. The interval radius is unchanged.
    pub fn negated(&self) -> Self {
        Self {
            mean_numerator: -&self.mean_numerator,
            mean_denominator: self.mean_denominator.clone(),
            mean_radius: self.mean_radius.clone(),
            variance: self.variance.clone(),
        }
    }

    /// Subtract moments while retaining the unknown/shared covariance bound.
    pub fn subtract(&self, rhs: &Self) -> Self {
        self.add(&rhs.negated())
    }

    /// Sum mean intervals and retain unknown/shared-source covariance.
    pub fn add(&self, rhs: &Self) -> Self {
        self.add_with_variance(rhs, self.variance.sum_variance_bound(&rhs.variance))
    }

    /// Add moments under an explicitly justified correlation classification.
    /// Only `Independent` permits a linear variance sum; shared or unknown
    /// sources retain the Cauchy covariance envelope.
    pub fn add_with_covariance(&self, rhs: &Self, correlation: NoiseCorrelation) -> Self {
        let variance = match correlation {
            NoiseCorrelation::Independent => self.variance.add(&rhs.variance),
            NoiseCorrelation::SharedSource | NoiseCorrelation::Unknown => {
                self.variance.sum_variance_bound(&rhs.variance)
            }
        };
        self.add_with_variance(rhs, variance)
    }

    fn add_with_variance(&self, rhs: &Self, variance: NonnegativeRational) -> Self {
        let numerator = &self.mean_numerator * BigInt::from(rhs.mean_denominator.clone()) +
            &rhs.mean_numerator * BigInt::from(self.mean_denominator.clone());
        let denominator = &self.mean_denominator * &rhs.mean_denominator;
        let divisor = gcd(numerator.magnitude(), &denominator);
        Self {
            mean_numerator: numerator / BigInt::from(divisor.clone()),
            mean_denominator: denominator / divisor,
            mean_radius: self.mean_radius.add(&rhs.mean_radius),
            variance,
        }
    }

    /// Scale one source, including repeated family captures. This is not an
    /// independent sum of newly sampled errors.
    pub fn scaled(&self, count: &BigUint) -> Self {
        let numerator = &self.mean_numerator * BigInt::from(count.clone());
        let divisor = gcd(numerator.magnitude(), &self.mean_denominator);
        Self {
            mean_numerator: numerator / BigInt::from(divisor.clone()),
            mean_denominator: &self.mean_denominator / divisor,
            mean_radius: self.mean_radius.scaled(count),
            variance: self.variance.scaled(&count.pow(2)),
        }
    }

    /// Uniform integer interval starting at `lower` with positive `size`.
    /// The support contract is established by parameter construction.
    pub fn uniform_interval(lower: BigInt, size: BigUint) -> Self {
        let numerator = lower * 2u8 + BigInt::from(&size - 1u8);
        let denominator = BigUint::from(2u8);
        let divisor = gcd(numerator.magnitude(), &denominator);
        Self {
            mean_numerator: numerator / BigInt::from(divisor.clone()),
            mean_denominator: denominator / divisor,
            mean_radius: NonnegativeRational::zero(),
            variance: NonnegativeRational::new(&size * &size - 1u8, 12u8.into())
                .expect("positive constant denominator"),
        }
    }
}

/// Coefficient norms of one matrix action on a noise row.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RightActionNoiseGain {
    pub l1: BigUint,
    pub l2_squared: BigUint,
}

impl RightActionNoiseGain {
    pub fn unit() -> Self {
        Self { l1: BigUint::one(), l2_squared: BigUint::one() }
    }
}

/// Distribution policy for a modulus-switch rounding residue.
///
/// The discrete policy is the default for an integer source/target ratio.
/// Continuous rounding is retained as an explicit approximation for models
/// that document that assumption; it is never selected by a missing value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseMomentPolicy {
    /// Exact centered integer-residue law for the supplied ratio.
    Discrete { alpha: BigUint },
    /// Centered continuous-uniform approximation on `[-1/2, 1/2]`.
    Continuous,
}

impl NoiseMomentPolicy {
    /// Construct the exact or approximate rounding moments selected by this
    /// policy. A zero discrete ratio is rejected rather than interpreted as a
    /// noiseless conversion.
    pub fn rounding_moments(&self) -> Result<NoiseMoments, NoiseArithmeticError> {
        match self {
            Self::Discrete { alpha } => NoiseMoments::discrete_rounding_residue(alpha),
            Self::Continuous => Ok(NoiseMoments::continuous_rounding_residue()),
        }
    }
}

/// Named assumptions attached to an average-case transfer.
///
/// These labels are diagnostic only. The deterministic bound channel remains
/// authoritative unless the caller's existing acceptance policy explicitly
/// opts into the average model.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum NoiseHeuristicId {
    /// State coefficient/error decorrelation.
    H1StateUncorrelated,
    /// Centered-uniform rounding residues, independent per coefficient and
    /// of the errors they round. Decomposition digits are not assumed
    /// uniform; they keep their bounded energy.
    H2ResidueUniform,
    /// Independent setup randomness across flat branches.
    H3BranchIndependence,
    /// Fixed-RHS conditional independence.
    H4RhsIndependence,
    /// Distinct routed-label independence.
    H5RouteIndependence,
    /// Gaussian/subgaussian tail closure.
    H6GaussianTail,
    /// Exact under the selected Gaussian or discrete law.
    ExactDistribution,
    /// An unknown covariance was bounded, not set to zero.
    UnknownCovarianceBound,
}

/// Deduplicated, serializable heuristic ledger for one transfer.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseHeuristicLedger {
    ids: Vec<NoiseHeuristicId>,
}

impl NoiseHeuristicLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a label once, preserving deterministic enum order in the result.
    pub fn insert(&mut self, id: NoiseHeuristicId) {
        if !self.ids.contains(&id) {
            self.ids.push(id);
            self.ids.sort_unstable();
        }
    }

    /// Add several labels and return the updated ledger.
    pub fn with(mut self, ids: impl IntoIterator<Item = NoiseHeuristicId>) -> Self {
        for id in ids {
            self.insert(id);
        }
        self
    }

    /// Return the deduplicated labels.
    pub fn ids(&self) -> &[NoiseHeuristicId] {
        &self.ids
    }
}

/// Stable source and route identity carried by alignment/moment reports.
/// Reusing a source on a different route does not make it independent.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NoiseSourceProvenance {
    /// Sampler/source identity.
    pub source: [u8; 32],
    /// Ordered route/site indices used to reach this term.
    pub route: Vec<usize>,
}

/// Correlation classification for a pair of transfer terms.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum NoiseCorrelation {
    /// Same sampled source or a known shared ancestor.
    SharedSource,
    /// Distinct samples with a declared independence assumption.
    Independent,
    /// No relationship was established; use a Cauchy bound.
    Unknown,
}

fn gcd(left: &BigUint, right: &BigUint) -> BigUint {
    let mut left = left.clone();
    let mut right = right.clone();
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    left
}

/// An index into one trace's flat dependency arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseNodeId(pub usize);

/// Program binding provenance used by symbolic noise traces.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum NoiseBinding {
    Input(WireId),
    RhsSite { gate: Option<WireId>, indices: Vec<mxx_ir_core::IntExpr> },
    Helper(LutId),
    GateHelper(WireId),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseDependencyNode {
    Binding(NoiseBinding),
    Add {
        gate: Option<WireId>,
        lhs: NoiseNodeId,
        rhs: NoiseNodeId,
    },
    MulPublic {
        gate: Option<WireId>,
        lhs: NoiseNodeId,
        rhs: NoiseNodeId,
    },
    MulPrivate {
        gate: Option<WireId>,
        input: NoiseNodeId,
        rhs: NoiseNodeId,
        helper: Option<NoiseNodeId>,
    },
    Lut {
        gate: Option<WireId>,
        table: LutId,
        input: NoiseNodeId,
        helper: Option<NoiseNodeId>,
    },
    Reduction {
        gate: Option<WireId>,
        input: NoiseNodeId,
        range: FamilyRange,
    },
}

/// Dependencies recorded by a symbolic noise trace.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseDependencyGraph {
    pub nodes: Vec<NoiseDependencyNode>,
    pub wires: BTreeMap<WireId, NoiseNodeId>,
}

// Source-aware average-case noise contributions.
//
// A noise value's moments are the image of the sampled errors and rounding
// residuals that produced it. The breakdown below keeps those images apart,
// keyed by the sampled object they came from, so that composition can add
// the variances of genuinely independent samples while retaining the Cauchy
// (coherent) bound for repeated or unknown ones. See
// `docs/correctness/average-case-noise-implementation-design.md`.
//
// Independence rules, all conservative:
//
// * Contributions of the same sampled object are combined coherently, and so are those whose member
//   coordinates are not provably different: distinct coordinate expressions may alias.
// * Sampled errors of distinct objects are uncorrelated. Every transfer that produces them applies
//   a fixed public or prepared action (the fixed-RHS conditional model, H4), and the caller records
//   that assumption.
// * A source whose members were summed over a lane, or read through an unknown index relation,
//   keeps all of its contributions in one coherent group: overlapping sums of the same members are
//   not independent.
// * A rounding residual is identified by its operation and operands, so repeating an operation
//   repeats its residual. Residuals of distinct operations are uncorrelated with each other and
//   with sampled errors (H2).
// * An opaque contribution (a boundary that only has a total envelope) is coherent with everything.

/// Coordinate of a sampled member.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum SampleCoordinate {
    /// Member `index`, read inside `depth` enclosing parallel families.
    Index { depth: usize, index: Arc<IntExpr> },
    /// The sum over the members addressed by the lanes of a family at
    /// `depth`.
    Reduced { depth: usize },
    /// A member related to other reads in an unknown way.
    Unknown,
}

/// What produced a contribution. Independence also depends on the sampling
/// contract and the transfer; the kind alone does not establish it.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ContributionKind {
    SampledError,
    RoundingResidual,
    Opaque,
}

/// The image of one sampled object (or residual) in a noise value.
#[derive(Clone, Debug)]
pub(crate) struct NoiseContribution {
    /// Sampling namespace of the original object, never the consuming gate.
    pub source: [u8; 32],
    /// Diagnostic name of the producing transfer; not part of the identity.
    pub origin: &'static str,
    /// Member coordinates of the sampled object.
    pub sample: Vec<SampleCoordinate>,
    pub kind: ContributionKind,
    /// Every member of the enclosing input family has its own sample, so a
    /// member read appends its index to `sample`.
    pub per_member: bool,
    /// The coefficients are uncorrelated with a common variance (a fresh
    /// sample, or its image under signed monomials), so a fixed action may
    /// use its coefficient energy instead of its squared L1 norm (H1/H4).
    pub white: bool,
    pub moments: NoiseMoments,
}

impl NoiseContribution {
    fn summarized(&self) -> bool {
        self.sample.iter().any(|coordinate| !matches!(coordinate, SampleCoordinate::Index { .. }))
    }
}

/// Aggregate reduction counts, for diagnostics.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReductionCounts {
    pub coherent: usize,
    pub independent: usize,
    pub histogram: usize,
    pub unknown_index: usize,
}

/// The contributions of one noise value.
#[derive(Clone, Debug, Default)]
pub(crate) struct NoiseBreakdown {
    terms: Vec<NoiseContribution>,
    /// Depth of the family lane whose coordinates this element's terms
    /// carry, when the element was produced by a parallel body.
    lane_depth: Option<usize>,
}

/// Fraction bits kept by the outward rounding of propagated moments. The
/// exact alignment and scale ratios otherwise grow every denominator along a
/// modulus chain; `2^-128` is far below every noise budget.
const PRECISION: u32 = 128;

fn zero_moments() -> NoiseMoments {
    NoiseMoments {
        mean_numerator: num_bigint::BigInt::zero(),
        mean_denominator: BigUint::one(),
        mean_radius: NonnegativeRational::zero(),
        variance: NonnegativeRational::zero(),
    }
}

impl NoiseBreakdown {
    /// One fresh sampled object.
    pub(crate) fn sampled(
        source: [u8; 32],
        origin: &'static str,
        moments: NoiseMoments,
        white: bool,
    ) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source,
                origin,
                sample: Vec::new(),
                kind: ContributionKind::SampledError,
                per_member: false,
                white,
                moments,
            }],
        }
    }

    /// One sampled member of a sampled family, at coordinate `index`.
    pub(crate) fn sampled_member(
        source: [u8; 32],
        origin: &'static str,
        depth: usize,
        index: IntExpr,
        moments: NoiseMoments,
        white: bool,
    ) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source,
                origin,
                sample: vec![SampleCoordinate::Index { depth, index: Arc::new(index) }],
                kind: ContributionKind::SampledError,
                per_member: false,
                white,
                moments,
            }],
        }
    }

    /// One rounding residual of a gate.
    pub(crate) fn rounding(source: [u8; 32], origin: &'static str, moments: NoiseMoments) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source,
                origin,
                sample: Vec::new(),
                kind: ContributionKind::RoundingResidual,
                per_member: false,
                white: false,
                moments,
            }],
        }
    }

    /// A total envelope with no known decomposition, coherent with every
    /// other contribution.
    /// A rounding residual of one sampled object, with that object's member
    /// coordinates: residuals of distinct objects are uncorrelated (H2).
    pub(crate) fn object_residual(
        source: [u8; 32],
        origin: &'static str,
        sample: Vec<SampleCoordinate>,
        moments: NoiseMoments,
        white: bool,
    ) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source,
                origin,
                sample,
                kind: ContributionKind::RoundingResidual,
                per_member: false,
                white,
                moments,
            }],
        }
    }

    /// The source and member coordinates when these contributions are those
    /// of exactly one sampled object.
    pub(crate) fn single_sample(&self) -> Option<([u8; 32], &[SampleCoordinate])> {
        match self.terms.as_slice() {
            [term] if term.kind == ContributionKind::SampledError && !term.per_member => {
                Some((term.source, &term.sample))
            }
            _ => None,
        }
    }

    pub(crate) fn opaque(origin: &'static str, moments: NoiseMoments) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source: [0; 32],
                origin,
                sample: Vec::new(),
                kind: ContributionKind::Opaque,
                per_member: false,
                white: false,
                moments,
            }],
        }
    }

    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// Mark every sampled error as the sample of one family member each.
    pub(crate) fn independent_members(mut self) -> Self {
        for term in &mut self.terms {
            if term.kind == ContributionKind::SampledError {
                term.per_member = true;
            }
        }
        self
    }

    /// The contributions of one member read at `index` inside `depth`
    /// enclosing families. Per-member samples take the member's coordinate.
    /// Coordinates of a produced family's lane keep their meaning only when
    /// the member is read by the same lane (`identity`) at the same depth;
    /// otherwise they are no longer related to the reading lane and become
    /// unknown.
    pub(crate) fn member(&self, depth: usize, index: &IntExpr, identity: bool) -> Self {
        let lane_depth = self.lane_depth;
        let related = |d: usize| match lane_depth {
            Some(lane_depth) => d < lane_depth || (d == depth && lane_depth == depth && identity),
            None => d <= depth,
        };
        let terms = self
            .terms
            .iter()
            .map(|term| {
                let mut term = term.clone();
                for coordinate in &mut term.sample {
                    if matches!(coordinate, SampleCoordinate::Index { depth: d, .. } if !related(*d))
                    {
                        *coordinate = SampleCoordinate::Unknown;
                    }
                }
                if term.per_member {
                    term.sample
                        .push(SampleCoordinate::Index { depth, index: Arc::new(index.clone()) });
                    term.per_member = false;
                }
                term
            })
            .collect();
        Self { terms, lane_depth: None }
    }

    /// Mark this element of a family produced by a parallel body at `depth`.
    pub(crate) fn with_lane_depth(mut self, depth: usize) -> Self {
        self.lane_depth = Some(depth);
        self
    }

    /// Apply a fixed linear action to every contribution. `action` returns
    /// the mean gain, the variance of the image, and whether the image stays
    /// white, from the contribution before the action.
    pub(crate) fn transform(
        &self,
        mut action: impl FnMut(&NoiseContribution) -> (NonnegativeRational, NonnegativeRational, bool),
    ) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|term| {
                let (mean_gain, variance, white) = action(term);
                NoiseContribution {
                    moments: term.moments.linear_image(&mean_gain, variance).rounded_up(PRECISION),
                    white,
                    ..term.clone()
                }
            })
            .collect();
        Self { terms, lane_depth: self.lane_depth }
    }

    /// Both collections; contributions of the same object are combined
    /// coherently.
    pub(crate) fn merge(&self, other: &Self) -> Self {
        let mut merged =
            Self { terms: self.terms.clone(), lane_depth: self.lane_depth.or(other.lane_depth) };
        merged.terms.extend(other.terms.iter().cloned());
        merged.normalize()
    }

    fn normalize(self) -> Self {
        let mut grouped = BTreeMap::<
            (ContributionKind, [u8; 32], Vec<SampleCoordinate>),
            NoiseContribution,
        >::new();
        let lane_depth = self.lane_depth;
        for term in self.terms {
            let key = (term.kind, term.source, term.sample.clone());
            match grouped.get_mut(&key) {
                Some(existing) => {
                    existing.moments = existing
                        .moments
                        .add_with_covariance(&term.moments, NoiseCorrelation::SharedSource)
                        .rounded_up(PRECISION);
                    existing.white &= term.white;
                    existing.per_member &= term.per_member;
                }
                None => {
                    let mut term = term;
                    term.moments = term.moments.rounded_up(PRECISION);
                    grouped.insert(key, term);
                }
            }
        }
        Self { terms: grouped.into_values().collect(), lane_depth }
    }

    /// Sum a family of `count` lanes at `depth` whose lane index is `lane`.
    pub(crate) fn reduce(
        &self,
        depth: usize,
        lane: &IntExpr,
        count: &BigUint,
        counts: &mut ReductionCounts,
    ) -> Self {
        let lane = lane.clone().canonicalize();
        let terms = self
            .terms
            .iter()
            .map(|term| {
                let at_depth = term
                    .sample
                    .iter()
                    .filter(|coordinate| {
                        matches!(coordinate, SampleCoordinate::Index { depth: d, .. } if *d == depth)
                    })
                    .collect::<Vec<_>>();
                let lane_dependent = at_depth.iter().any(|coordinate| match coordinate {
                    SampleCoordinate::Index { index, .. } => index.contains_loop_index(),
                    _ => false,
                });
                let mut reduced = term.clone();
                // This lane's coordinates are summed away.
                for coordinate in &mut reduced.sample {
                    if matches!(coordinate, SampleCoordinate::Index { depth: d, .. } if *d == depth) {
                        *coordinate = SampleCoordinate::Reduced { depth };
                    }
                }
                // Only a contribution carrying member coordinates can be a
                // distinct object per lane; a gate residual has none.
                let independent =
                    term.kind != ContributionKind::Opaque && lane_dependent && at_depth.len() == 1;
                let multiplicity = match (independent, at_depth.first()) {
                    (true, Some(SampleCoordinate::Index { index, .. })) => {
                        lane_multiplicity(index, &lane, count)
                    }
                    _ => None,
                };
                match multiplicity {
                    Some(LaneMultiplicity::Injective) => {
                        counts.independent += 1;
                        reduced.moments = independent_sum(&term.moments, count, count);
                    }
                    Some(LaneMultiplicity::Histogram(squares)) => {
                        counts.histogram += 1;
                        reduced.moments = independent_sum(&term.moments, count, &squares);
                    }
                    None => {
                        if lane_dependent && term.kind != ContributionKind::Opaque {
                            counts.unknown_index += 1;
                            for coordinate in &mut reduced.sample {
                                if matches!(coordinate, SampleCoordinate::Reduced { depth: d } if *d == depth) {
                                    *coordinate = SampleCoordinate::Unknown;
                                }
                            }
                        } else {
                            counts.coherent += 1;
                        }
                        reduced.moments = term.moments.scaled(count);
                    }
                }
                reduced.white = false;
                reduced
            })
            .collect::<Vec<_>>();
        Self { terms, lane_depth: None }.normalize()
    }

    /// Total moments. Contributions are grouped into classes of possible
    /// covariance: within a class variances combine coherently; distinct
    /// sampled and residual classes add; the opaque class is coherent with
    /// the rest. Contributions of one source share a class unless their
    /// member coordinates provably address different members: different
    /// coordinate expressions may alias, so they stay coherent.
    pub(crate) fn total(&self) -> NoiseMoments {
        let mut opaque = None::<NoiseMoments>;
        let mut groups = BTreeMap::<(ContributionKind, [u8; 32]), Vec<&NoiseContribution>>::new();
        for term in &self.terms {
            if term.kind == ContributionKind::Opaque {
                opaque = Some(match opaque {
                    None => term.moments.clone(),
                    Some(previous) => {
                        previous.add_with_covariance(&term.moments, NoiseCorrelation::Unknown)
                    }
                });
                continue;
            }
            groups.entry((term.kind, term.source)).or_default().push(term);
        }
        let mut independent = zero_moments();
        for terms in groups.values() {
            // A source summed over a lane or read at an unknown member may
            // overlap every other read of it.
            let summarized = terms.iter().any(|term| term.summarized());
            let mut parent = (0..terms.len()).collect::<Vec<_>>();
            fn root(parent: &mut [usize], mut index: usize) -> usize {
                while parent[index] != index {
                    parent[index] = parent[parent[index]];
                    index = parent[index];
                }
                index
            }
            for left in 0..terms.len() {
                for right in left + 1..terms.len() {
                    if summarized || !provably_distinct(&terms[left].sample, &terms[right].sample) {
                        let (left, right) = (root(&mut parent, left), root(&mut parent, right));
                        parent[right] = left;
                    }
                }
            }
            let mut classes = BTreeMap::<usize, NoiseMoments>::new();
            for (index, term) in terms.iter().enumerate() {
                let class = root(&mut parent, index);
                classes
                    .entry(class)
                    .and_modify(|moments| {
                        *moments = moments
                            .add_with_covariance(&term.moments, NoiseCorrelation::SharedSource)
                    })
                    .or_insert_with(|| term.moments.clone());
            }
            for class in classes.into_values() {
                independent =
                    independent.add_with_covariance(&class, NoiseCorrelation::Independent);
            }
        }
        match opaque {
            None => independent,
            Some(opaque) => independent.add_with_covariance(&opaque, NoiseCorrelation::Unknown),
        }
    }

    /// A digest of the contributions' identities, naming the value that an
    /// operation consumes: a deterministic operation on equal identities
    /// produces the same residual. `None` when an opaque contribution hides
    /// the value's identity.
    pub(crate) fn identity(&self) -> Option<[u8; 32]> {
        use sha2::{Digest, Sha256};
        if self.terms.iter().any(|term| term.kind == ContributionKind::Opaque) {
            return None;
        }
        let mut keys = self
            .terms
            .iter()
            .map(|term| format!("{:?}/{:?}/{:?}", term.kind, term.source, term.sample))
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        let mut hasher = Sha256::new();
        for key in keys {
            hasher.update((key.len() as u64).to_le_bytes());
            hasher.update(key.as_bytes());
        }
        Some(hasher.finalize().into())
    }

    /// Variance by contribution kind, for diagnostics.
    /// The contribution with the largest variance, for diagnostics.
    pub(crate) fn dominant(&self) -> Option<&NoiseContribution> {
        self.terms.iter().max_by(|left, right| {
            let (left, right) = (&left.moments.variance, &right.moments.variance);
            (&left.numerator * &right.denominator).cmp(&(&right.numerator * &left.denominator))
        })
    }

    pub(crate) fn variance_by_kind(&self) -> BTreeMap<ContributionKind, NonnegativeRational> {
        let mut by_kind = BTreeMap::<ContributionKind, NonnegativeRational>::new();
        for term in &self.terms {
            let entry = by_kind.entry(term.kind).or_insert_with(NonnegativeRational::zero);
            *entry = entry.add(&term.moments.variance);
        }
        by_kind
    }
}

enum LaneMultiplicity {
    /// Every lane reads a distinct member.
    Injective,
    /// Lanes read members with multiplicities `m_s`; the value is `sum m_s²`.
    Histogram(BigUint),
}

/// How the lanes of a family read the members addressed by `index`.
fn lane_multiplicity(index: &IntExpr, lane: &IntExpr, count: &BigUint) -> Option<LaneMultiplicity> {
    let index = index.clone().canonicalize();
    if let IntExpr::Select { selector, branches } = &index {
        if selector.clone().canonicalize() != *lane || BigUint::from(branches.len()) != *count {
            return None;
        }
        let mut histogram = BTreeMap::new();
        for branch in branches {
            *histogram.entry(branch.evaluate(&ParamEnv::default()).ok()?).or_insert(0u64) += 1;
        }
        if histogram.len() == branches.len() {
            return Some(LaneMultiplicity::Injective);
        }
        return Some(LaneMultiplicity::Histogram(
            histogram.values().map(|multiplicity| BigUint::from(multiplicity * multiplicity)).sum(),
        ));
    }
    // A constant offset of the lane index reads distinct members.
    IntExpr::Sub(Box::new(index), Box::new(lane.clone()))
        .canonicalize()
        .evaluate(&ParamEnv::default())
        .ok()
        .map(|_| LaneMultiplicity::Injective)
}

/// Sum of `count` equally distributed members whose variances combine as
/// `squares` copies of one member's (the count itself for independent
/// members, `sum m_s²` for repeated groups).
/// Whether two member coordinate tuples address different members in every
/// lane. Only a proof counts; an unknown relation may alias.
fn provably_distinct(left: &[SampleCoordinate], right: &[SampleCoordinate]) -> bool {
    left.len() == right.len() &&
        left.iter().zip(right).any(|pair| match pair {
            (
                SampleCoordinate::Index { depth: left_depth, index: left },
                SampleCoordinate::Index { depth: right_depth, index: right },
            ) => indices_distinct(*left_depth, left, *right_depth, right),
            _ => false,
        })
}

fn indices_distinct(
    left_depth: usize,
    left: &IntExpr,
    right_depth: usize,
    right: &IntExpr,
) -> bool {
    let environment = ParamEnv::default();
    // Constants name fixed members, whatever the scope that reads them.
    if let (Ok(left), Ok(right)) = (left.evaluate(&environment), right.evaluate(&environment)) {
        return left != right;
    }
    // Lane-dependent indices only relate within one scope.
    if left_depth != right_depth {
        return false;
    }
    let difference = IntExpr::Sub(Box::new(left.clone()), Box::new(right.clone())).canonicalize();
    if let Ok(difference) = difference.evaluate(&environment) {
        return !difference.is_zero();
    }
    match (left.clone().canonicalize(), right.clone().canonicalize()) {
        (
            IntExpr::Select { selector: left_selector, branches: left },
            IntExpr::Select { selector: right_selector, branches: right },
        ) if left_selector == right_selector && left.len() == right.len() => {
            left.iter().zip(&right).all(|(left, right)| {
                matches!(
                    (left.evaluate(&environment), right.evaluate(&environment)),
                    (Ok(left), Ok(right)) if left != right
                )
            })
        }
        _ => false,
    }
}

fn independent_sum(moments: &NoiseMoments, count: &BigUint, squares: &BigUint) -> NoiseMoments {
    let mean = moments.scaled(count);
    NoiseMoments { variance: moments.variance.scaled(squares), ..mean }
}

#[cfg(test)]
mod breakdown_tests {
    use super::*;

    fn moments(variance: u64) -> NoiseMoments {
        NoiseMoments {
            variance: NonnegativeRational::new(variance.into(), 1u8.into()).unwrap(),
            ..zero_moments()
        }
    }

    fn variance(breakdown: &NoiseBreakdown) -> NonnegativeRational {
        breakdown.total().variance
    }

    fn rational(value: u64) -> NonnegativeRational {
        NonnegativeRational::new(value.into(), 1u8.into()).unwrap()
    }

    #[test]
    fn identical_sources_add_coherently_and_distinct_samples_independently() {
        let a = NoiseBreakdown::sampled([1; 32], "test", moments(3), true);
        let b = NoiseBreakdown::sampled([2; 32], "test", moments(3), true);
        assert_eq!(variance(&a.merge(&a)), rational(12));
        assert_eq!(variance(&a.merge(&b)), rational(6));
    }

    #[test]
    fn a_family_sum_keeps_shared_coherent_and_fresh_independent() {
        let lane = IntExpr::LoopIndex(0);
        let shared = NoiseBreakdown::sampled([1; 32], "test", moments(2), true);
        let fresh =
            NoiseBreakdown::sampled_member([2; 32], "test", 1, lane.clone(), moments(5), true);
        let mut counts = ReductionCounts::default();
        let reduced = shared.merge(&fresh).reduce(1, &lane, &BigUint::from(4u8), &mut counts);
        // 4² * 2 + 4 * 5.
        assert_eq!(variance(&reduced), rational(52));
        assert_eq!((counts.coherent, counts.independent), (1, 1));
    }

    #[test]
    fn repeated_selections_use_their_multiplicity_histogram() {
        let lane = IntExpr::LoopIndex(0);
        let index = IntExpr::Select {
            selector: Box::new(lane.clone()),
            branches: vec![IntExpr::constant(0), IntExpr::constant(0), IntExpr::constant(1)],
        };
        let member = NoiseBreakdown::sampled_member([1; 32], "test", 1, index, moments(1), true);
        let mut counts = ReductionCounts::default();
        let reduced = member.reduce(1, &lane, &BigUint::from(3u8), &mut counts);
        // 2² + 1², not 3.
        assert_eq!(variance(&reduced), rational(5));
        let unknown = IntExpr::Mul(Box::new(lane.clone()), Box::new(lane.clone()));
        let member = NoiseBreakdown::sampled_member([1; 32], "test", 1, unknown, moments(1), true);
        let reduced = member.reduce(1, &lane, &BigUint::from(3u8), &mut counts);
        assert_eq!(variance(&reduced), rational(9), "an unknown index map stays coherent");
    }

    #[test]
    fn reused_members_stay_shared_and_distinct_members_are_independent() {
        let first = NoiseBreakdown::sampled_member(
            [1; 32],
            "test",
            0,
            IntExpr::constant(3),
            moments(1),
            true,
        );
        let other = NoiseBreakdown::sampled_member(
            [1; 32],
            "test",
            0,
            IntExpr::constant(4),
            moments(1),
            true,
        );
        assert_eq!(variance(&first.merge(&first)), rational(4));
        assert_eq!(variance(&first.merge(&other)), rational(2));
    }

    #[test]
    fn overlapping_family_sums_of_one_source_stay_coherent() {
        let lane = IntExpr::LoopIndex(0);
        let member =
            NoiseBreakdown::sampled_member([1; 32], "test", 1, lane.clone(), moments(1), true);
        let mut counts = ReductionCounts::default();
        let reduced = member.reduce(1, &lane, &BigUint::from(4u8), &mut counts);
        // Two sums over the same members: (sqrt 4 + sqrt 4)² = 16, not 8.
        assert_eq!(variance(&reduced.merge(&reduced)), rational(16));
        // Another single member of the same source joins the coherent class.
        let single = NoiseBreakdown::sampled_member(
            [1; 32],
            "test",
            0,
            IntExpr::constant(0),
            moments(4),
            true,
        );
        assert_eq!(variance(&reduced.merge(&single)), rational(16));
    }

    #[test]
    fn only_provably_different_coordinates_are_independent() {
        let lane = IntExpr::LoopIndex(0);
        let read = |depth, index: IntExpr| {
            NoiseBreakdown::sampled_member([1; 32], "test", depth, index, moments(1), true)
        };
        // `i` and `Select(i, [0, 0])` alias in lane zero.
        let aliased = IntExpr::Select {
            selector: Box::new(lane.clone()),
            branches: vec![IntExpr::constant(0), IntExpr::constant(0)],
        };
        assert_eq!(variance(&read(1, lane.clone()).merge(&read(1, aliased))), rational(4));
        // `i` and `i + 1` never address the same member.
        let next = IntExpr::Add(Box::new(lane.clone()), Box::new(IntExpr::constant(1)));
        assert_eq!(variance(&read(1, lane.clone()).merge(&read(1, next))), rational(2));
        // Selections that differ in every lane are distinct.
        let select = |branches: [i64; 2]| IntExpr::Select {
            selector: Box::new(lane.clone()),
            branches: branches.into_iter().map(IntExpr::constant).collect(),
        };
        assert_eq!(variance(&read(1, select([0, 1])).merge(&read(1, select([1, 0])))), rational(2));
        assert_eq!(variance(&read(1, select([0, 1])).merge(&read(1, select([0, 2])))), rational(4));
        // Lane indices of different scopes are unrelated, so they may alias;
        // constants name the same members in every scope.
        assert_eq!(variance(&read(1, lane.clone()).merge(&read(2, lane))), rational(4));
        assert_eq!(
            variance(&read(1, IntExpr::constant(0)).merge(&read(2, IntExpr::constant(0)))),
            rational(4)
        );
        assert_eq!(
            variance(&read(1, IntExpr::constant(0)).merge(&read(2, IntExpr::constant(1)))),
            rational(2)
        );
    }

    #[test]
    fn identity_names_equal_operands_and_hides_opaque_ones() {
        let a = NoiseBreakdown::sampled([1; 32], "a", moments(1), true);
        let renamed = NoiseBreakdown::sampled([1; 32], "b", moments(9), true);
        let b = NoiseBreakdown::sampled([2; 32], "a", moments(1), true);
        assert_eq!(a.identity(), renamed.identity());
        assert_ne!(a.identity(), b.identity());
        assert_eq!(a.merge(&NoiseBreakdown::opaque("test", moments(1))).identity(), None);
    }

    #[test]
    fn per_member_samples_take_the_read_coordinate() {
        let family =
            NoiseBreakdown::sampled([1; 32], "test", moments(1), true).independent_members();
        let a = family.member(0, &IntExpr::constant(0), false);
        let b = family.member(0, &IntExpr::constant(1), false);
        assert_eq!(variance(&a.merge(&b)), rational(2));
        let shared = NoiseBreakdown::sampled([1; 32], "test", moments(1), true);
        let a = shared.member(0, &IntExpr::constant(0), false);
        let b = shared.member(0, &IntExpr::constant(1), false);
        assert_eq!(
            variance(&a.merge(&b)),
            rational(4),
            "a family without the contract is one sample"
        );
    }

    #[test]
    fn rounding_residuals_are_coherent_across_lanes_and_opaque_terms_with_everything() {
        let lane = IntExpr::LoopIndex(0);
        let rounding = NoiseBreakdown::rounding([7; 32], "test", moments(1));
        let mut counts = ReductionCounts::default();
        assert_eq!(
            variance(&rounding.reduce(1, &lane, &BigUint::from(3u8), &mut counts)),
            rational(9)
        );
        let fresh = NoiseBreakdown::sampled([1; 32], "test", moments(4), true);
        assert_eq!(variance(&fresh.merge(&rounding)), rational(5));
        let opaque = NoiseBreakdown::opaque("test", moments(4));
        assert_eq!(variance(&fresh.merge(&opaque)), rational(16));
    }
}
