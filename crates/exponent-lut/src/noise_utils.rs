//! Exact noise arithmetic shared by encoding schemes.
//!
//! These values have no implicit ciphertext coordinate units. Coordinate
//! conventions belong to the scheme adapter that consumes them.
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
    /// Centered digit or residue approximation.
    H2DigitUniform,
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
