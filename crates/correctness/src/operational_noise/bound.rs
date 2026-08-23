//! Pure coefficient-bound transfer rules used by normal-form constructors.
//!
//! This module intentionally contains no expression graph, extracted-node, or evaluator
//! abstraction.  Bounds are attached to typed normal-form factors and are combined by the
//! deterministic constructors in `normal_form`.

use mxx_ir_core::types::ConcreteMatrixType;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

/// Whether a matrix has a numeric centered-coefficient bound.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BoundClass {
    ExactZero,
    Bounded { maximum_absolute_coefficient: BigUint },
    Large,
}

impl BoundClass {
    /// Preserve the unique representation of an exactly-zero matrix.
    pub fn bounded(maximum_absolute_coefficient: BigUint) -> Self {
        if maximum_absolute_coefficient.is_zero() {
            Self::ExactZero
        } else {
            Self::Bounded { maximum_absolute_coefficient }
        }
    }

    pub fn maximum_absolute_coefficient(&self) -> Option<BigUint> {
        match self {
            Self::ExactZero => Some(BigUint::ZERO),
            Self::Bounded { maximum_absolute_coefficient } => {
                Some(maximum_absolute_coefficient.clone())
            }
            Self::Large => None,
        }
    }
}

/// Facts about the represented matrix value, independent of its noise bound.
///
/// A coefficient cap here is a semantic contract for coefficient extraction;
/// it is not the bound used by noise arithmetic.  Keeping this as a distinct
/// typed value prevents a finite noise summary from being accidentally reused
/// as a canonical selector range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixValueMetadata {
    /// Authoritative exclusive upper bound for nonnegative coefficients.
    pub canonical_coefficient_exclusive_upper: Option<BigUint>,
    /// Whether every polynomial coefficient is a constant polynomial.
    pub is_constant_polynomial: bool,
    /// Number of rows proved to be identically zero.
    pub known_zero_rows: Option<BigUint>,
    /// Facts about the coefficient support of scalar polynomials.  Absence is
    /// deliberately conservative: a caller must use the ring dimension.
    pub polynomial: Option<PolynomialFacts>,
}

/// Typed facts for a polynomial value.  `support_upper` is always validated
/// against the owning ring dimension before it enters a normal-form factor.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PolynomialFacts {
    pub support_upper: usize,
}

impl PolynomialFacts {
    pub fn new(support_upper: usize, ring_dimension: usize) -> Result<Self, BoundArithmeticError> {
        if support_upper > ring_dimension {
            return Err(BoundArithmeticError::InvalidSupportUpper { support_upper, ring_dimension });
        }
        Ok(Self { support_upper })
    }

    pub fn conservative(ring_dimension: usize) -> Self {
        Self { support_upper: ring_dimension }
    }
}

impl MatrixValueMetadata {
    pub const fn unknown() -> Self {
        Self {
            canonical_coefficient_exclusive_upper: None,
            is_constant_polynomial: false,
            known_zero_rows: None,
            polynomial: None,
        }
    }

    /// Join value facts from alternative representations of one value.
    ///
    /// A family/packed value is sound only when a fact is true for every
    /// reachable alternative.  In particular, a coefficient cap is retained
    /// only when every alternative supplies one; the common cap is the
    /// maximum, never the first child's cap.  Polynomial support is joined in
    /// the same (upper-bound) lattice.  `known_zero_rows` is a count-only
    /// legacy contract, so the only information that can be carried across
    /// alternatives without row identities is the conservative minimum.
    pub fn join<'a>(metadata: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut iter = metadata.into_iter();
        let Some(first) = iter.next() else { return Self::unknown() };
        let mut joined = first.clone();
        for next in iter {
            joined.canonical_coefficient_exclusive_upper = match (
                joined.canonical_coefficient_exclusive_upper.take(),
                next.canonical_coefficient_exclusive_upper.clone(),
            ) {
                (Some(left), Some(right)) => Some(left.max(right)),
                _ => None,
            };
            joined.is_constant_polynomial &= next.is_constant_polynomial;
            joined.known_zero_rows =
                match (joined.known_zero_rows.take(), next.known_zero_rows.clone()) {
                    (Some(left), Some(right)) => Some(left.min(right)),
                    _ => None,
                };
            joined.polynomial = match (joined.polynomial.take(), next.polynomial.clone()) {
                (Some(left), Some(right)) => Some(PolynomialFacts {
                    support_upper: left.support_upper.max(right.support_upper),
                }),
                _ => None,
            };
        }
        joined
    }

    /// Transfer facts through addition.  This is deliberately separate from
    /// the noise-bound transfer: a finite noise cap is not a value contract.
    pub fn add(&self, other: &Self, ring_dimension: usize) -> Self {
        let polynomial =
            self.polynomial.as_ref().zip(other.polynomial.as_ref()).map(|(left, right)| {
                PolynomialFacts {
                    support_upper: left
                        .support_upper
                        .saturating_add(right.support_upper)
                        .min(ring_dimension),
                }
            });
        Self {
            // Addition can wrap/reduce in the owning ring.  Unless a
            // dedicated canonical-residue proof is supplied, no input cap
            // remains an authoritative extraction contract.
            canonical_coefficient_exclusive_upper: None,
            is_constant_polynomial: self.is_constant_polynomial && other.is_constant_polynomial,
            known_zero_rows: self
                .known_zero_rows
                .as_ref()
                .zip(other.known_zero_rows.as_ref())
                .map(|(left, right)| left.min(right).clone()),
            polynomial,
        }
    }

    pub fn negate(&self) -> Self {
        self.clone()
    }

    pub fn transpose(&self) -> Self {
        let mut result = self.clone();
        // The metadata has only a row count, not a row-position map.  A
        // transpose turns rows into columns, so retaining this field would
        // claim facts about unrelated output rows.
        result.known_zero_rows = None;
        result
    }

    pub fn product(&self, other: &Self, ring_dimension: usize) -> Self {
        let polynomial =
            self.polynomial.as_ref().zip(other.polynomial.as_ref()).map(|(left, right)| {
                PolynomialFacts {
                    support_upper: left
                        .support_upper
                        .checked_mul(right.support_upper)
                        .unwrap_or(ring_dimension)
                        .min(ring_dimension),
                }
            });
        Self {
            // Matrix products use convolution and inner-dimension sums; a
            // coefficient cap alone is not a canonical residue range.
            canonical_coefficient_exclusive_upper: None,
            is_constant_polynomial: self.is_constant_polynomial && other.is_constant_polynomial,
            known_zero_rows: None,
            polynomial,
        }
    }

    pub fn tensor(&self, other: &Self, ring_dimension: usize) -> Self {
        let mut result = self.product(other, ring_dimension);
        result.known_zero_rows = None;
        result
    }

    pub fn concat<'a>(metadata: impl IntoIterator<Item = &'a Self>, ring_dimension: usize) -> Self {
        let mut result = Self::join(metadata);
        result.polynomial = result.polynomial.map(|facts| PolynomialFacts {
            support_upper: facts.support_upper.min(ring_dimension),
        });
        // A concat changes row positions; without an explicit row-position
        // map no zero-row claim can be transferred soundly.
        result.known_zero_rows = None;
        result
    }
}

/// Short internal name retained while the remaining operation table is
/// migrated to the explicit value-metadata terminology.
pub type MatrixMetadata = MatrixValueMetadata;

/// A coefficient bound and the concrete matrix shape to which it applies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixBound {
    pub matrix_type: ConcreteMatrixType,
    pub coefficient_class: BoundClass,
}

/// The value facts that are relevant to a matrix-product bound transfer.
///
/// This is deliberately passed separately from [`MatrixBound`].  A noise
/// bound is not a proof about the represented value, and must never be reused
/// as one.  Callers that do not have value facts use the conservative default.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatrixProductFacts {
    pub left_is_constant_polynomial: bool,
    pub right_is_constant_polynomial: bool,
    pub right_known_zero_rows: Option<BigUint>,
    pub left_support_upper: Option<usize>,
    pub right_support_upper: Option<usize>,
}

/// A resolved matrix constant consumed by the normal-form operation table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedMatrixConstant {
    Zero,
    Identity,
    UnitRow { index: BigUint },
    UnitColumn { index: BigUint },
    Gadget { base: BigInt, small: bool },
    PowerOfBase { base: BigInt, exponent: BigUint },
    Rotation { exponent: BigInt },
    Polynomial { coefficients: Box<[BigInt]> },
}

/// Errors from pure matrix-bound arithmetic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoundArithmeticError {
    IncompatibleMatrixProduct { left: ConcreteMatrixType, right: ConcreteMatrixType },
    InvalidKnownZeroRows { known_zero_rows: BigUint, row_count: BigUint },
    InvalidSupportUpper { support_upper: usize, ring_dimension: usize },
}

/// The sole deterministic matrix-product transfer helper.
pub fn product_bound(
    left: &MatrixBound,
    right: &MatrixBound,
) -> Result<MatrixBound, BoundArithmeticError> {
    product_bound_with_facts(left, right, &MatrixProductFacts::default())
}

/// Product transfer with explicitly supplied value facts.
pub fn product_bound_with_facts(
    left: &MatrixBound,
    right: &MatrixBound,
    facts: &MatrixProductFacts,
) -> Result<MatrixBound, BoundArithmeticError> {
    if left.matrix_type.modulus != right.matrix_type.modulus ||
        left.matrix_type.ring_dimension != right.matrix_type.ring_dimension
    {
        return Err(BoundArithmeticError::IncompatibleMatrixProduct {
            left: left.matrix_type.clone(),
            right: right.matrix_type.clone(),
        });
    }
    for support_upper in [facts.left_support_upper, facts.right_support_upper].into_iter().flatten()
    {
        if support_upper > left.matrix_type.ring_dimension {
            return Err(BoundArithmeticError::InvalidSupportUpper {
                support_upper,
                ring_dimension: left.matrix_type.ring_dimension,
            });
        }
    }
    let left_scalar = left.matrix_type.rows == 1 && left.matrix_type.columns == 1;
    let right_scalar = right.matrix_type.rows == 1 && right.matrix_type.columns == 1;
    let left_support = facts
        .left_support_upper
        .map(BigUint::from)
        .unwrap_or_else(|| BigUint::from(left.matrix_type.ring_dimension));
    let right_support = facts
        .right_support_upper
        .map(BigUint::from)
        .unwrap_or_else(|| BigUint::from(right.matrix_type.ring_dimension));
    let left_support =
        if facts.left_is_constant_polynomial { BigUint::one() } else { left_support };
    let right_support =
        if facts.right_is_constant_polynomial { BigUint::one() } else { right_support };
    let ring_dimension = BigUint::from(left.matrix_type.ring_dimension);
    let (matrix_type, coefficient_factor) = if left_scalar && right_scalar {
        // The left scalar acts on the right scalar, so its proved support is
        // part of this coefficient transfer.  Canonical monomial folding
        // applies central factors one at a time and therefore remains
        // association-independent even when the support product is capped.
        (right.matrix_type.clone(), left_support)
    } else if left_scalar {
        (right.matrix_type.clone(), left_support)
    } else if right_scalar {
        (left.matrix_type.clone(), right_support)
    } else {
        if left.matrix_type.columns != right.matrix_type.rows {
            return Err(BoundArithmeticError::IncompatibleMatrixProduct {
                left: left.matrix_type.clone(),
                right: right.matrix_type.clone(),
            });
        }
        let rows = BigUint::from(right.matrix_type.rows);
        let known_zero_rows = facts.right_known_zero_rows.clone().unwrap_or_default();
        if known_zero_rows > rows {
            return Err(BoundArithmeticError::InvalidKnownZeroRows {
                known_zero_rows,
                row_count: rows,
            });
        }
        let inner = BigUint::from(left.matrix_type.columns) - known_zero_rows;
        let ring_factor = if facts.left_is_constant_polynomial || facts.right_is_constant_polynomial
        {
            BigUint::one()
        } else {
            ring_dimension
        };
        (
            ConcreteMatrixType {
                modulus: left.matrix_type.modulus.clone(),
                ring_dimension: left.matrix_type.ring_dimension,
                rows: left.matrix_type.rows,
                columns: right.matrix_type.columns,
            },
            inner * ring_factor,
        )
    };
    Ok(MatrixBound {
        matrix_type,
        coefficient_class: multiply_classes(
            &left.coefficient_class,
            &right.coefficient_class,
            &coefficient_factor,
        ),
    })
}

/// Transfer the coefficient cap of a Kronecker/tensor product.  A tensor
/// multiplies every coefficient pair, and therefore carries one ring-size
/// factor unless either operand is proved constant in the polynomial ring.
/// This is the sole tensor-bound rule used by both normal-form evaluation and
/// family-contract propagation.
pub fn tensor_bound_with_facts(
    left: &MatrixBound,
    right: &MatrixBound,
    facts: &MatrixProductFacts,
) -> Result<MatrixBound, BoundArithmeticError> {
    if left.matrix_type.modulus != right.matrix_type.modulus ||
        left.matrix_type.ring_dimension != right.matrix_type.ring_dimension
    {
        return Err(BoundArithmeticError::IncompatibleMatrixProduct {
            left: left.matrix_type.clone(),
            right: right.matrix_type.clone(),
        });
    }
    let coefficient_class = match (&left.coefficient_class, &right.coefficient_class) {
        (BoundClass::ExactZero, _) | (_, BoundClass::ExactZero) => BoundClass::ExactZero,
        (
            BoundClass::Bounded { maximum_absolute_coefficient: left_coeff },
            BoundClass::Bounded { maximum_absolute_coefficient: right_coeff },
        ) => BoundClass::bounded(
            left_coeff *
                right_coeff *
                if facts.left_is_constant_polynomial || facts.right_is_constant_polynomial {
                    BigUint::from(1_u8)
                } else {
                    BigUint::from(left.matrix_type.ring_dimension)
                },
        ),
        _ => BoundClass::Large,
    };
    let matrix_type = ConcreteMatrixType {
        modulus: left.matrix_type.modulus.clone(),
        ring_dimension: left.matrix_type.ring_dimension,
        rows: left.matrix_type.rows.checked_mul(right.matrix_type.rows).ok_or(
            BoundArithmeticError::IncompatibleMatrixProduct {
                left: left.matrix_type.clone(),
                right: right.matrix_type.clone(),
            },
        )?,
        columns: left.matrix_type.columns.checked_mul(right.matrix_type.columns).ok_or(
            BoundArithmeticError::IncompatibleMatrixProduct {
                left: left.matrix_type.clone(),
                right: right.matrix_type.clone(),
            },
        )?,
    };
    Ok(MatrixBound { matrix_type, coefficient_class })
}

fn multiply_classes(left: &BoundClass, right: &BoundClass, factor: &BigUint) -> BoundClass {
    match (left, right) {
        (BoundClass::ExactZero, _) | (_, BoundClass::ExactZero) => BoundClass::ExactZero,
        (
            BoundClass::Bounded { maximum_absolute_coefficient: left },
            BoundClass::Bounded { maximum_absolute_coefficient: right },
        ) => BoundClass::bounded(factor * left * right),
        _ => BoundClass::Large,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize) -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: 17.into(), ring_dimension: 1, rows, columns }
    }

    fn matrix_ring(ring_dimension: usize, rows: usize, columns: usize) -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: 17.into(), ring_dimension, rows, columns }
    }

    fn bounded(matrix_type: ConcreteMatrixType, value: u64) -> MatrixBound {
        MatrixBound { matrix_type, coefficient_class: BoundClass::bounded(value.into()) }
    }

    #[test]
    fn product_uses_proved_zero_rows() {
        let left = bounded(matrix(2, 3), 2);
        let right = bounded(matrix(3, 4), 5);
        let actual = product_bound_with_facts(
            &left,
            &right,
            &MatrixProductFacts { right_known_zero_rows: Some(1_u8.into()), ..Default::default() },
        )
        .unwrap();
        assert_eq!(actual.coefficient_class, BoundClass::bounded(20_u8.into()));
    }

    #[test]
    fn product_preserves_large_and_zero_annihilation() {
        let left = MatrixBound { matrix_type: matrix(1, 1), coefficient_class: BoundClass::Large };
        let right = bounded(matrix(1, 1), 3);
        assert_eq!(product_bound(&left, &right).unwrap().coefficient_class, BoundClass::Large);

        let zero = bounded(matrix(1, 1), 0);
        assert_eq!(product_bound(&zero, &left).unwrap().coefficient_class, BoundClass::ExactZero);
    }

    #[test]
    fn scalar_product_uses_scalar_runtime_semantics() {
        let scalar = bounded(matrix(1, 1), 3);
        let row = bounded(matrix(1, 2), 5);
        assert_eq!(product_bound(&row, &scalar).unwrap().matrix_type, row.matrix_type);
        assert_eq!(
            product_bound(&row, &scalar).unwrap().coefficient_class,
            BoundClass::bounded(15_u8.into())
        );
    }

    #[test]
    fn scalar_support_uses_one_or_proved_support_instead_of_ring_dimension() {
        let scalar = bounded(matrix_ring(4, 1, 1), 3);
        let row = bounded(matrix_ring(4, 1, 2), 5);
        let support_one = product_bound_with_facts(
            &scalar,
            &row,
            &MatrixProductFacts { left_support_upper: Some(1), ..Default::default() },
        )
        .unwrap();
        assert_eq!(support_one.coefficient_class, BoundClass::bounded(15_u8.into()));
        let conservative = product_bound(&scalar, &row).unwrap();
        assert_eq!(conservative.coefficient_class, BoundClass::bounded(60_u8.into()));
    }

    #[test]
    fn scalar_by_scalar_multiplies_both_support_upper_bounds() {
        let left = bounded(matrix_ring(8, 1, 1), 2);
        let right = bounded(matrix_ring(8, 1, 1), 3);
        let product = product_bound_with_facts(
            &left,
            &right,
            &MatrixProductFacts {
                left_support_upper: Some(2),
                right_support_upper: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(product.coefficient_class, BoundClass::bounded(12_u8.into()));
    }

    #[test]
    fn ordinary_product_uses_capped_support_geometry_and_rejects_invalid_support() {
        let left = bounded(matrix_ring(4, 2, 3), 2);
        let right = bounded(matrix_ring(4, 3, 2), 3);
        let ordinary = product_bound(&left, &right).unwrap();
        assert_eq!(ordinary.coefficient_class, BoundClass::bounded(72_u8.into()));
        let error = product_bound_with_facts(
            &left,
            &right,
            &MatrixProductFacts { right_support_upper: Some(5), ..Default::default() },
        )
        .unwrap_err();
        assert_eq!(
            error,
            BoundArithmeticError::InvalidSupportUpper { support_upper: 5, ring_dimension: 4 }
        );
    }

    #[test]
    fn ordinary_row_column_product_uses_inner_geometry() {
        let left = bounded(matrix_ring(4, 1, 3), 2);
        let right = bounded(matrix_ring(4, 3, 1), 3);
        let product = product_bound_with_facts(
            &left,
            &right,
            &MatrixProductFacts {
                left_support_upper: Some(2),
                right_support_upper: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((product.matrix_type.rows, product.matrix_type.columns), (1, 1));
        assert_eq!(product.coefficient_class, BoundClass::bounded(72_u8.into()));
    }

    #[test]
    fn tensor_uses_ring_factor_without_constant_polynomial_fact() {
        let left = bounded(matrix_ring(3, 2, 1), 2);
        let right = bounded(matrix_ring(3, 1, 2), 5);
        let conservative =
            tensor_bound_with_facts(&left, &right, &MatrixProductFacts::default()).unwrap();
        assert_eq!((conservative.matrix_type.rows, conservative.matrix_type.columns), (2, 2));
        assert_eq!(conservative.coefficient_class, BoundClass::bounded(30_u8.into()));
        let constant = tensor_bound_with_facts(
            &left,
            &right,
            &MatrixProductFacts { left_is_constant_polynomial: true, ..Default::default() },
        )
        .unwrap();
        assert_eq!(constant.coefficient_class, BoundClass::bounded(10_u8.into()));
    }

    #[test]
    fn polynomial_facts_are_bounded_and_zero_large_are_preserved() {
        assert_eq!(PolynomialFacts::new(3, 4).unwrap().support_upper, 3);
        assert_eq!(
            PolynomialFacts::new(5, 4).unwrap_err(),
            BoundArithmeticError::InvalidSupportUpper { support_upper: 5, ring_dimension: 4 }
        );
        let large =
            MatrixBound { matrix_type: matrix_ring(4, 1, 1), coefficient_class: BoundClass::Large };
        let zero = bounded(matrix_ring(4, 1, 1), 0);
        assert_eq!(product_bound(&zero, &large).unwrap().coefficient_class, BoundClass::ExactZero);
    }

    #[test]
    fn metadata_join_is_conservative_across_alternatives() {
        let first = MatrixMetadata {
            canonical_coefficient_exclusive_upper: Some(7_u8.into()),
            is_constant_polynomial: true,
            known_zero_rows: Some(3_u8.into()),
            polynomial: Some(PolynomialFacts { support_upper: 2 }),
        };
        let second = MatrixMetadata {
            canonical_coefficient_exclusive_upper: Some(11_u8.into()),
            is_constant_polynomial: false,
            known_zero_rows: Some(1_u8.into()),
            polynomial: Some(PolynomialFacts { support_upper: 4 }),
        };
        assert_eq!(
            MatrixMetadata::join([&first, &second]),
            MatrixMetadata {
                canonical_coefficient_exclusive_upper: Some(11_u8.into()),
                is_constant_polynomial: false,
                known_zero_rows: Some(1_u8.into()),
                polynomial: Some(PolynomialFacts { support_upper: 4 }),
            }
        );
        let missing = MatrixMetadata { canonical_coefficient_exclusive_upper: None, ..first };
        assert_eq!(
            MatrixMetadata::join([&missing, &second]).canonical_coefficient_exclusive_upper,
            None
        );
    }

    #[test]
    fn operation_metadata_does_not_reuse_an_input_canonical_cap() {
        let left = MatrixMetadata {
            canonical_coefficient_exclusive_upper: Some(7_u8.into()),
            is_constant_polynomial: true,
            known_zero_rows: Some(1_u8.into()),
            polynomial: Some(PolynomialFacts { support_upper: 1 }),
        };
        let right = MatrixMetadata {
            canonical_coefficient_exclusive_upper: Some(9_u8.into()),
            is_constant_polynomial: true,
            known_zero_rows: Some(1_u8.into()),
            polynomial: Some(PolynomialFacts { support_upper: 2 }),
        };
        let product = left.product(&right, 8);
        assert_eq!(product.canonical_coefficient_exclusive_upper, None);
        assert_eq!(product.polynomial, Some(PolynomialFacts { support_upper: 2 }));
        assert_eq!(left.transpose().known_zero_rows, None);
    }
}
