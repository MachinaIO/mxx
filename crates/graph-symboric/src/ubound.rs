use crate::serde_support;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fmt};
use thiserror::Error;

/// Binary fractional precision used by every [`UBound`] operation.
pub const UBOUND_FRACTION_BITS: usize = 128;

/// A nonnegative fixed-precision upper bound.
///
/// `scaled` represents `value * 2^UBOUND_FRACTION_BITS`. Every operation rounds
/// toward positive infinity, so a value can never under-approximate its exact
/// mathematical input.
#[derive(Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct UBound {
    #[serde(default, with = "serde_support::biguint")]
    lower_scaled: BigUint,
    #[serde(with = "serde_support::biguint")]
    scaled: BigUint,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum UBoundError {
    #[error("UBound cannot represent a negative value")]
    Negative,
    #[error("UBound denominator must be nonzero")]
    DivisionByZero,
    #[error("UBound denominator is too small to derive a positive lower bound")]
    DenominatorBelowPrecision,
    #[error("UBound subtraction would be negative")]
    NegativeSubtraction,
}

impl UBound {
    fn scale() -> BigUint {
        BigUint::one() << UBOUND_FRACTION_BITS
    }

    pub fn zero() -> Self {
        Self { lower_scaled: BigUint::zero(), scaled: BigUint::zero() }
    }

    pub fn one() -> Self {
        let scale = Self::scale();
        Self { lower_scaled: scale.clone(), scaled: scale }
    }

    pub fn from_integer(value: &BigInt) -> Result<Self, UBoundError> {
        let (sign, magnitude) = value.to_bytes_be();
        if sign == Sign::Minus {
            return Err(UBoundError::Negative);
        }
        let scaled = BigUint::from_bytes_be(&magnitude) * Self::scale();
        Ok(Self { lower_scaled: scaled.clone(), scaled })
    }

    pub fn from_u64(value: u64) -> Self {
        let scaled = BigUint::from(value) * Self::scale();
        Self { lower_scaled: scaled.clone(), scaled }
    }

    pub fn from_ratio(numerator: &BigInt, denominator: &BigInt) -> Result<Self, UBoundError> {
        if numerator.sign() == Sign::Minus {
            return Err(UBoundError::Negative);
        }
        if denominator <= &BigInt::zero() {
            return if denominator.is_zero() {
                Err(UBoundError::DivisionByZero)
            } else {
                Err(UBoundError::Negative)
            };
        }
        let numerator = numerator.to_biguint().expect("nonnegative numerator");
        let denominator = denominator.to_biguint().expect("positive denominator");
        let scaled_numerator = numerator * Self::scale();
        Ok(Self {
            lower_scaled: &scaled_numerator / &denominator,
            scaled: div_ceil(&scaled_numerator, &denominator),
        })
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            lower_scaled: &self.lower_scaled + &rhs.lower_scaled,
            scaled: &self.scaled + &rhs.scaled,
        }
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, UBoundError> {
        if self.scaled < rhs.lower_scaled {
            return Err(UBoundError::NegativeSubtraction);
        }
        let lower_scaled = if self.lower_scaled > rhs.scaled {
            &self.lower_scaled - &rhs.scaled
        } else {
            BigUint::zero()
        };
        Ok(Self { lower_scaled, scaled: &self.scaled - &rhs.lower_scaled })
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let scale = Self::scale();
        Self {
            lower_scaled: (&self.lower_scaled * &rhs.lower_scaled) / &scale,
            scaled: div_ceil(&(&self.scaled * &rhs.scaled), &scale),
        }
    }

    pub fn div(&self, rhs: &Self) -> Result<Self, UBoundError> {
        if rhs.scaled.is_zero() {
            return Err(UBoundError::DivisionByZero);
        }
        if rhs.lower_scaled.is_zero() {
            return Err(UBoundError::DenominatorBelowPrecision);
        }
        let scale = Self::scale();
        Ok(Self {
            lower_scaled: (&self.lower_scaled * &scale) / &rhs.scaled,
            scaled: div_ceil(&(&self.scaled * scale), &rhs.lower_scaled),
        })
    }

    pub fn sqrt(&self) -> Self {
        let scale = Self::scale();
        Self {
            lower_scaled: sqrt_floor(&(&self.lower_scaled * &scale)),
            scaled: sqrt_ceil(&(&self.scaled * scale)),
        }
    }

    pub fn max(&self, rhs: &Self) -> Self {
        if self >= rhs { self.clone() } else { rhs.clone() }
    }

    pub fn min(&self, rhs: &Self) -> Self {
        if self <= rhs { self.clone() } else { rhs.clone() }
    }

    pub fn is_zero(&self) -> bool {
        self.scaled.is_zero()
    }

    pub fn scaled(&self) -> &BigUint {
        &self.scaled
    }

    /// Converts this value for legacy backend APIs while retaining an upper bound.
    pub fn to_f64_upper(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        let value =
            self.scaled.to_f64().unwrap_or(f64::INFINITY) / 2f64.powi(UBOUND_FRACTION_BITS as i32);
        if value.is_finite() { f64::from_bits(value.to_bits().saturating_add(1)) } else { value }
    }
}

impl Ord for UBound {
    fn cmp(&self, other: &Self) -> Ordering {
        self.scaled.cmp(&other.scaled).then_with(|| self.lower_scaled.cmp(&other.lower_scaled))
    }
}

impl PartialOrd for UBound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for UBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UBound([{}/2^{}, {}/2^{}])",
            self.lower_scaled, UBOUND_FRACTION_BITS, self.scaled, UBOUND_FRACTION_BITS
        )
    }
}

fn div_ceil(numerator: &BigUint, denominator: &BigUint) -> BigUint {
    debug_assert!(!denominator.is_zero());
    let quotient = numerator / denominator;
    if numerator % denominator == BigUint::zero() { quotient } else { quotient + BigUint::one() }
}

fn sqrt_ceil(value: &BigUint) -> BigUint {
    if value.is_zero() {
        return BigUint::zero();
    }
    let mut low = BigUint::zero();
    let mut high = BigUint::one() << ((value.bits() as usize).div_ceil(2) + 1);
    while &low + BigUint::one() < high {
        let mid = (&low + &high) >> 1usize;
        if &mid * &mid < *value {
            low = mid;
        } else {
            high = mid;
        }
    }
    high
}

fn sqrt_floor(value: &BigUint) -> BigUint {
    let ceiling = sqrt_ceil(value);
    if &ceiling * &ceiling == *value { ceiling } else { ceiling - BigUint::one() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_rounds_up() {
        let one_third =
            UBound::from_ratio(&BigInt::from(1), &BigInt::from(3)).expect("valid ratio");
        let product = one_third.mul(&UBound::from_u64(3));
        assert!(product >= UBound::one());
        assert!(one_third.scaled() * BigUint::from(3u8) >= UBound::scale());
    }

    #[test]
    fn sqrt_rounds_up() {
        let two = UBound::from_u64(2);
        let root = two.sqrt();
        assert!(root.mul(&root) >= two);
    }

    #[test]
    fn rational_operations_never_under_approximate() {
        let scale = UBound::scale();
        for seed in 1u64..=128 {
            let a = BigUint::from(seed * 17 % 101 + 1);
            let b = BigUint::from(seed * 29 % 97 + 1);
            let c = BigUint::from(seed * 43 % 103 + 1);
            let d = BigUint::from(seed * 61 % 89 + 1);
            let lhs = UBound::from_ratio(&BigInt::from(a.clone()), &BigInt::from(b.clone()))
                .expect("positive ratio");
            let rhs = UBound::from_ratio(&BigInt::from(c.clone()), &BigInt::from(d.clone()))
                .expect("positive ratio");

            let product = lhs.mul(&rhs);
            assert!(
                product.scaled() * &b * &d >= &a * &c * &scale,
                "multiplication under-approximated at seed {seed}"
            );
            let quotient = lhs.div(&rhs).expect("nonzero divisor");
            assert!(
                quotient.scaled() * &b * &c >= &a * &d * &scale,
                "division under-approximated at seed {seed}"
            );
            if &a * &d >= &c * &b {
                let difference = lhs.sub(&rhs).expect("nonnegative difference");
                assert!(
                    difference.scaled() * &b * &d >= (&a * &d - &c * &b) * &scale,
                    "subtraction under-approximated at seed {seed}"
                );
            }
            let root = lhs.sqrt();
            assert!(
                root.scaled() * root.scaled() * &b >= &a * &scale * &scale,
                "square root under-approximated at seed {seed}"
            );
        }
    }
}
