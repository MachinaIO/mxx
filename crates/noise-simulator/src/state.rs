//! Abstract numeric states and primitive initial-state constructors.

use crate::{bound, identity::SourceId};
use mxx_ir_core::types::ConcreteMatrixType;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MatrixState {
    /// Bound on actual minus implicit nominal.
    pub error_bound: BigUint,
    /// Bound on both represented actual and implicit nominal values.
    pub coefficient_magnitude_bound: BigUint,
    /// True only when all polynomial entries are constant polynomials.
    pub is_constant_polynomial: bool,
    /// Bound for the one nominal rightmost source retained for preimage use.
    pub right_carrier: Option<RightCarrier>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RightCarrier {
    pub source: SourceId,
    pub left_gain: BigUint,
}

impl MatrixState {
    pub fn new(
        error_bound: BigUint,
        coefficient_magnitude_bound: BigUint,
        is_constant_polynomial: bool,
    ) -> Result<Self, StateError> {
        Ok(Self {
            error_bound,
            coefficient_magnitude_bound,
            is_constant_polynomial,
            right_carrier: None,
        })
    }

    pub fn with_carrier(mut self, source: SourceId, left_gain: BigUint) -> Self {
        self.right_carrier = Some(RightCarrier { source, left_gain });
        self
    }

    pub fn ordinary_product(
        &self,
        right: &Self,
        geometry: bound::ProductGeometry,
        modulus: &BigInt,
    ) -> Result<Self, StateError> {
        // Let actual(A) = nominal(A) + e_A and actual(B) = nominal(B) + e_B.
        // The exact product error is
        // actual(A)actual(B) - nominal(A)nominal(B)
        //   = actual(A)e_B + e_A nominal(B).
        // Because actual(A)e_B already contains e_A e_B, adding a separate
        // e_A e_B term would double-count that cross term.
        // The first bound below represents actual(A)e_B; the second represents
        // e_A nominal(B).
        let error = bound::product_bound(
            &self.coefficient_magnitude_bound,
            &right.error_bound,
            geometry,
            self.is_constant_polynomial,
            false,
        )? + bound::product_bound(
            &self.error_bound,
            &right.coefficient_magnitude_bound,
            geometry,
            false,
            right.is_constant_polynomial,
        )?;
        let magnitude = bound::product_bound(
            &self.coefficient_magnitude_bound,
            &right.coefficient_magnitude_bound,
            geometry,
            self.is_constant_polynomial,
            right.is_constant_polynomial,
        )?;
        let left_gain = bound::left_action_gain(self, geometry)?;
        // The rightmost source is the carrier of a product.  Its gain is
        // amplified by the left action, so B's source survives as L*B.
        let right_carrier = right.right_carrier.as_ref().map(|carrier| RightCarrier {
            source: carrier.source,
            left_gain: &left_gain * &carrier.left_gain,
        });
        Ok(Self {
            error_bound: error,
            coefficient_magnitude_bound: bound::cap_by_centered_residue(magnitude, modulus)?,
            is_constant_polynomial: self.is_constant_polynomial && right.is_constant_polynomial,
            right_carrier,
        })
    }

    pub fn add(&self, right: &Self, modulus: &BigInt) -> Result<Self, StateError> {
        let magnitude = bound::cap_by_centered_residue(
            &self.coefficient_magnitude_bound + &right.coefficient_magnitude_bound,
            modulus,
        )?;
        // Addition can keep a source witness only when both terms refer to
        // the same source; distinct sources would require a multi-carrier
        // state, so the witness is conservatively removed.
        let right_carrier = match (&self.right_carrier, &right.right_carrier) {
            (Some(a), Some(b)) if a.source == b.source => {
                Some(RightCarrier { source: a.source, left_gain: &a.left_gain + &b.left_gain })
            }
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
            _ => None,
        };
        Ok(Self {
            error_bound: &self.error_bound + &right.error_bound,
            coefficient_magnitude_bound: magnitude,
            is_constant_polynomial: self.is_constant_polynomial && right.is_constant_polynomial,
            right_carrier,
        })
    }

    /// Subtraction has the same conservative numeric and carrier transfer as
    /// addition; carrier gains are absolute amplification bounds.
    pub fn subtract(&self, right: &Self, modulus: &BigInt) -> Result<Self, StateError> {
        self.add(right, modulus)
    }

    pub fn negate(&self, modulus: &BigInt) -> Result<Self, StateError> {
        let mut result = self.clone();
        result.coefficient_magnitude_bound =
            bound::cap_by_centered_residue(result.coefficient_magnitude_bound.clone(), modulus)?;
        Ok(result)
    }

    pub fn scale(&self, scalar: &BigInt, modulus: &BigInt) -> Result<Self, StateError> {
        let absolute = scalar.abs().to_biguint().unwrap_or_default();
        let mut result = self.clone();
        result.error_bound *= &absolute;
        result.coefficient_magnitude_bound = bound::cap_by_centered_residue(
            result.coefficient_magnitude_bound * absolute.clone(),
            modulus,
        )?;
        if let Some(carrier) = result.right_carrier.as_mut() {
            carrier.left_gain *= &absolute;
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntegerState {
    pub minimum: BigInt,
    pub maximum_inclusive: BigInt,
}

impl IntegerState {
    pub fn new(minimum: BigInt, maximum_inclusive: BigInt) -> Result<Self, StateError> {
        if minimum > maximum_inclusive {
            return Err(StateError::InvalidIntegerRange);
        }
        Ok(Self { minimum, maximum_inclusive })
    }

    pub fn singleton(value: impl Into<BigInt>) -> Self {
        let value = value.into();
        Self { minimum: value.clone(), maximum_inclusive: value }
    }

    pub fn join(&self, other: &Self) -> Self {
        Self {
            minimum: self.minimum.clone().min(other.minimum.clone()),
            maximum_inclusive: self.maximum_inclusive.clone().max(other.maximum_inclusive.clone()),
        }
    }

    pub fn contains(&self, value: &BigInt) -> bool {
        self.minimum <= *value && *value <= self.maximum_inclusive
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            minimum: &self.minimum + &other.minimum,
            maximum_inclusive: &self.maximum_inclusive + &other.maximum_inclusive,
        }
    }

    pub fn subtract(&self, other: &Self) -> Self {
        Self {
            minimum: &self.minimum - &other.maximum_inclusive,
            maximum_inclusive: &self.maximum_inclusive - &other.minimum,
        }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let candidates = [
            &self.minimum * &other.minimum,
            &self.minimum * &other.maximum_inclusive,
            &self.maximum_inclusive * &other.minimum,
            &self.maximum_inclusive * &other.maximum_inclusive,
        ];
        Self {
            minimum: candidates.iter().min().expect("four products").clone(),
            maximum_inclusive: candidates.iter().max().expect("four products").clone(),
        }
    }

    pub fn divide(&self, other: &Self) -> Result<Self, StateError> {
        if other.minimum <= BigInt::zero() && other.maximum_inclusive >= BigInt::zero() {
            return Err(StateError::DivisionRangeContainsZero);
        }
        let candidates = [
            &self.minimum / &other.minimum,
            &self.minimum / &other.maximum_inclusive,
            &self.maximum_inclusive / &other.minimum,
            &self.maximum_inclusive / &other.maximum_inclusive,
        ];
        Ok(Self {
            minimum: candidates.iter().min().expect("four quotients").clone(),
            maximum_inclusive: candidates.iter().max().expect("four quotients").clone(),
        })
    }

    pub fn remainder(&self, other: &Self) -> Result<Self, StateError> {
        if other.minimum <= BigInt::zero() && other.maximum_inclusive >= BigInt::zero() {
            return Err(StateError::DivisionRangeContainsZero);
        }
        let maximum_abs = other.minimum.abs().max(other.maximum_inclusive.abs());
        if self.minimum >= BigInt::zero() && other.minimum > BigInt::zero() {
            return Ok(Self {
                minimum: BigInt::zero(),
                maximum_inclusive: &maximum_abs - BigInt::one(),
            });
        }
        Ok(Self {
            minimum: -(&maximum_abs - BigInt::one()),
            maximum_inclusive: &maximum_abs - BigInt::one(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FamilyState {
    pub shape: Vec<usize>,
    pub element: Box<AbstractValue>,
}

impl FamilyState {
    pub fn new(shape: Vec<usize>, element: AbstractValue) -> Result<Self, StateError> {
        if shape.is_empty() || shape.contains(&0) || matches!(element, AbstractValue::Family(_)) {
            return Err(StateError::InvalidFamilyShape);
        }
        shape
            .iter()
            .try_fold(1usize, |product, extent| product.checked_mul(*extent))
            .ok_or(StateError::InvalidFamilyShape)?;
        Ok(Self { shape, element: Box::new(element) })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrapdoorState {
    pub matrix: ConcreteMatrixType,
    pub sigma: mxx_ir_core::expr::RealExpr,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub preimage_max_coefficient_bound: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AbstractValue {
    Matrix(MatrixState),
    Integer(IntegerState),
    Boolean(BooleanState),
    Bytes,
    TypedBlob { type_name: String, schema_hash: [u8; 32] },
    Trapdoor(TrapdoorState),
    Family(FamilyState),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BooleanState {
    FalseOnly,
    TrueOnly,
    Either,
}

impl BooleanState {
    pub fn join(self, other: Self) -> Self {
        if self == other { self } else { Self::Either }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum StateError {
    #[error(transparent)]
    Bound(#[from] bound::BoundError),
    #[error("integer interval minimum exceeds maximum")]
    InvalidIntegerRange,
    #[error("division or remainder divisor interval contains zero")]
    DivisionRangeContainsZero,
    #[error("family shape must be nonempty, nonzero, and cannot nest families")]
    InvalidFamilyShape,
    #[error("sample interval minimum exceeds maximum")]
    InvalidSampleRange,
    #[error("sample cutoff must be nonnegative")]
    NegativeCutoff,
    #[error("gadget base must be a power of two greater than one")]
    InvalidGadgetBase,
    #[error("gadget digit count must be positive")]
    InvalidDigitCount,
    #[error("preimage relation is unavailable for this value")]
    RelationUnavailable,
    #[error(
        "preimage relation source does not match left carrier (expected {expected:?}, actual {actual:?})"
    )]
    RelationSourceMismatch { expected: SourceId, actual: SourceId },
}

fn modulus_cap(matrix: &ConcreteMatrixType, value: BigUint) -> Result<BigUint, StateError> {
    Ok(value.min(bound::centered_residue_bound(&matrix.modulus)?))
}

fn ring_constant(matrix: &ConcreteMatrixType) -> bool {
    matrix.ring_dimension == 1
}

pub fn zero_matrix(matrix: &ConcreteMatrixType) -> Result<MatrixState, StateError> {
    bound::centered_residue_bound(&matrix.modulus)?;
    MatrixState::new(0u8.into(), 0u8.into(), true)
}

pub fn exact_matrix(
    matrix: &ConcreteMatrixType,
    magnitude: BigUint,
    constant: bool,
) -> Result<MatrixState, StateError> {
    MatrixState::new(0u8.into(), modulus_cap(matrix, magnitude)?, constant)
}

pub fn uniform_interval_sample(
    matrix: &ConcreteMatrixType,
    minimum: &BigInt,
    maximum: &BigInt,
) -> Result<MatrixState, StateError> {
    if minimum > maximum {
        return Err(StateError::InvalidSampleRange);
    }
    MatrixState::new(
        0u8.into(),
        modulus_cap(matrix, bound::max_abs_interval(minimum, maximum))?,
        ring_constant(matrix),
    )
}

pub fn gaussian_sample(
    matrix: &ConcreteMatrixType,
    cutoff: &BigInt,
) -> Result<MatrixState, StateError> {
    let cutoff = cutoff.to_biguint().ok_or(StateError::NegativeCutoff)?;
    MatrixState::new(cutoff.clone(), modulus_cap(matrix, cutoff)?, ring_constant(matrix))
}

pub fn uniform_residue_sample(matrix: &ConcreteMatrixType) -> Result<MatrixState, StateError> {
    MatrixState::new(0u8.into(), bound::centered_residue_bound(&matrix.modulus)?, false)
}

pub fn plain_hash_sample(matrix: &ConcreteMatrixType) -> Result<MatrixState, StateError> {
    uniform_residue_sample(matrix)
}

pub fn trapdoor_public_matrix(matrix: &ConcreteMatrixType) -> Result<MatrixState, StateError> {
    MatrixState::new(0u8.into(), bound::centered_residue_bound(&matrix.modulus)?, false)
}

pub fn preimage_sample(
    matrix: &ConcreteMatrixType,
    cutoff: &BigInt,
) -> Result<MatrixState, StateError> {
    let cutoff = cutoff.to_biguint().ok_or(StateError::NegativeCutoff)?;
    MatrixState::new(0u8.into(), modulus_cap(matrix, cutoff)?, false)
}

fn validate_gadget(base: &BigInt, digit_count: usize) -> Result<BigUint, StateError> {
    let base = base.to_biguint().ok_or(StateError::InvalidGadgetBase)?;
    if base <= BigUint::one() || (&base & (&base - BigUint::one())) != BigUint::zero() {
        return Err(StateError::InvalidGadgetBase);
    }
    if digit_count == 0 {
        return Err(StateError::InvalidDigitCount);
    }
    Ok(base)
}

pub fn gadget_matrix(
    matrix: &ConcreteMatrixType,
    base: &BigInt,
    digit_count: usize,
) -> Result<MatrixState, StateError> {
    let base = validate_gadget(base, digit_count)?;
    let magnitude = base.pow((digit_count - 1) as u32);
    // Gadget entries are explicit constant coefficients, even when the ring
    // contains several polynomial coefficients.
    exact_matrix(matrix, magnitude, true)
}

pub fn gadget_decomposition(
    matrix: &ConcreteMatrixType,
    base: &BigInt,
    small: bool,
    digit_count: usize,
) -> Result<MatrixState, StateError> {
    let base = validate_gadget(base, digit_count)?;
    let bits = base.trailing_zeros().expect("positive power of two");
    let raw = if small { &base - BigUint::one() } else { BigUint::one() << (bits - 1) };
    MatrixState::new(0u8.into(), modulus_cap(matrix, raw)?, ring_constant(matrix))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ty(ring_dimension: usize) -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: 17.into(), ring_dimension, rows: 2, columns: 3 }
    }

    #[test]
    fn short_samples_are_exact_and_gaussian_is_zero_nominal() {
        assert_eq!(
            uniform_interval_sample(&ty(8), &(-1).into(), &1.into()).unwrap().error_bound,
            0u8.into()
        );
        assert_eq!(
            uniform_interval_sample(&ty(8), &(-1).into(), &1.into())
                .unwrap()
                .coefficient_magnitude_bound,
            1u8.into()
        );
        let binary = uniform_interval_sample(&ty(8), &0.into(), &1.into()).unwrap();
        assert_eq!(binary.error_bound, 0u8.into());
        assert_eq!(binary.coefficient_magnitude_bound, 1u8.into());
        let gaussian = gaussian_sample(&ty(8), &100.into()).unwrap();
        assert_eq!(gaussian.error_bound, 100u8.into());
        assert_eq!(gaussian.coefficient_magnitude_bound, 8u8.into());
    }

    #[test]
    fn signed_interval_samples_use_both_endpoint_magnitudes() {
        let matrix =
            ConcreteMatrixType { modulus: 101.into(), ring_dimension: 1, rows: 1, columns: 1 };
        for (minimum, maximum) in [(-10, -5), (-10, 5), (5, 10)] {
            assert_eq!(
                uniform_interval_sample(&matrix, &minimum.into(), &maximum.into())
                    .unwrap()
                    .coefficient_magnitude_bound,
                10u8.into()
            );
        }
    }

    #[test]
    fn residue_hash_and_public_sources_use_full_centered_bound() {
        assert_eq!(uniform_residue_sample(&ty(8)).unwrap().coefficient_magnitude_bound, 8u8.into());
        assert_eq!(plain_hash_sample(&ty(8)).unwrap().coefficient_magnitude_bound, 8u8.into());
        assert_eq!(trapdoor_public_matrix(&ty(8)).unwrap().error_bound, 0u8.into());
    }

    #[test]
    fn gadget_bounds_distinguish_matrix_entries_from_digits() {
        let matrix = gadget_matrix(&ty(8), &4.into(), 3).unwrap();
        assert_eq!(matrix.coefficient_magnitude_bound, 8u8.into());
        assert!(matrix.is_constant_polynomial);
        let regular = gadget_decomposition(&ty(8), &4.into(), false, 3).unwrap();
        let small = gadget_decomposition(&ty(8), &4.into(), true, 3).unwrap();
        assert_eq!(regular.coefficient_magnitude_bound, 2u8.into());
        assert_eq!(small.coefficient_magnitude_bound, 3u8.into());
    }

    #[test]
    fn ordinary_product_keeps_rightmost_carrier_with_left_action_gain() {
        let left = MatrixState::new(2u8.into(), 3u8.into(), true).unwrap();
        let right = MatrixState::new(5u8.into(), 7u8.into(), false)
            .unwrap()
            .with_carrier(SourceId(9), 11u8.into());
        let result = left
            .ordinary_product(
                &right,
                crate::ProductGeometry { inner_dimension: 2, ring_dimension: 8 },
                &17.into(),
            )
            .unwrap();
        assert_eq!(
            result.right_carrier,
            Some(RightCarrier { source: SourceId(9), left_gain: 66u8.into() })
        );
    }

    #[test]
    fn zero_left_factor_keeps_rightmost_carrier_with_zero_gain() {
        let zero = MatrixState::new(0u8.into(), 0u8.into(), true).unwrap();
        let gadget = MatrixState::new(0u8.into(), 8u8.into(), true)
            .unwrap()
            .with_carrier(SourceId(9), 1u8.into());
        let result = zero
            .ordinary_product(
                &gadget,
                crate::ProductGeometry { inner_dimension: 2, ring_dimension: 8 },
                &17.into(),
            )
            .unwrap();
        assert_eq!(
            result.right_carrier,
            Some(RightCarrier { source: SourceId(9), left_gain: 0u8.into() })
        );
    }
}
