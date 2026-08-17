//! Pure coefficient-bound transfer rules used by normal-form constructors.
//!
//! This module intentionally contains no expression graph, extracted-node, or evaluator
//! abstraction.  Bounds are attached to typed normal-form factors and are combined by the
//! deterministic constructors in `normal_form` and `normal_form_product`.

use mxx_ir_core::types::ConcreteMatrixType;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

/// Whether a matrix has a numeric centered-coefficient bound.
#[derive(Clone, Debug, Eq, PartialEq)]
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

/// The exact coefficient cap of one gadget-decomposition digit.
pub(crate) fn gadget_digit_bound(base: &BigInt, small: bool) -> Option<BoundClass> {
    if base <= &BigInt::one() {
        return None;
    }
    let absolute = base.to_biguint().expect("positive gadget base");
    Some(BoundClass::bounded(if small {
        absolute - BigUint::one()
    } else {
        (absolute / BigUint::from(2_u8)).max(BigUint::one())
    }))
}

/// The coefficient class of a gadget matrix.  Regular gadget matrices are exact signal factors;
/// only the explicitly small representation has a finite digit-range cap.
pub(crate) fn gadget_matrix_bound(base: &BigInt, small: bool) -> Option<BoundClass> {
    if base <= &BigInt::one() {
        return None;
    }
    if !small {
        return Some(BoundClass::Large);
    }
    Some(BoundClass::bounded(base.to_biguint().expect("positive gadget base") - BigUint::one()))
}

/// Metadata that has a proof-preserving matrix transfer rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixMetadata {
    pub is_constant_polynomial: bool,
    pub known_zero_rows: Option<BigUint>,
}

impl MatrixMetadata {
    pub const fn unknown() -> Self {
        Self { is_constant_polynomial: false, known_zero_rows: None }
    }
}

/// A coefficient bound and the concrete matrix shape to which it applies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixBound {
    pub matrix_type: ConcreteMatrixType,
    pub coefficient_class: BoundClass,
    pub metadata: MatrixMetadata,
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
}

/// The sole deterministic matrix-product transfer helper.
pub fn product_bound(
    left: &MatrixBound,
    right: &MatrixBound,
) -> Result<MatrixBound, BoundArithmeticError> {
    if left.matrix_type.modulus != right.matrix_type.modulus ||
        left.matrix_type.ring_dimension != right.matrix_type.ring_dimension
    {
        return Err(BoundArithmeticError::IncompatibleMatrixProduct {
            left: left.matrix_type.clone(),
            right: right.matrix_type.clone(),
        });
    }
    let left_scalar = left.matrix_type.rows == 1 && left.matrix_type.columns == 1;
    let right_scalar = right.matrix_type.rows == 1 && right.matrix_type.columns == 1;
    let (matrix_type, effective_inner) = if left_scalar {
        (right.matrix_type.clone(), BigUint::one())
    } else if right_scalar {
        (left.matrix_type.clone(), BigUint::one())
    } else {
        if left.matrix_type.columns != right.matrix_type.rows {
            return Err(BoundArithmeticError::IncompatibleMatrixProduct {
                left: left.matrix_type.clone(),
                right: right.matrix_type.clone(),
            });
        }
        let rows = BigUint::from(right.matrix_type.rows);
        let known_zero_rows = right.metadata.known_zero_rows.clone().unwrap_or_default();
        if known_zero_rows > rows {
            return Err(BoundArithmeticError::InvalidKnownZeroRows {
                known_zero_rows,
                row_count: rows,
            });
        }
        (
            ConcreteMatrixType {
                modulus: left.matrix_type.modulus.clone(),
                ring_dimension: left.matrix_type.ring_dimension,
                rows: left.matrix_type.rows,
                columns: right.matrix_type.columns,
            },
            BigUint::from(left.matrix_type.columns) - known_zero_rows,
        )
    };
    let ring_factor =
        if left.metadata.is_constant_polynomial || right.metadata.is_constant_polynomial {
            BigUint::one()
        } else {
            BigUint::from(left.matrix_type.ring_dimension)
        };
    Ok(MatrixBound {
        matrix_type,
        coefficient_class: multiply_classes(
            &left.coefficient_class,
            &right.coefficient_class,
            &(effective_inner * ring_factor),
        ),
        metadata: MatrixMetadata {
            is_constant_polynomial: left.metadata.is_constant_polynomial &&
                right.metadata.is_constant_polynomial,
            known_zero_rows: None,
        },
    })
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

    fn bounded(matrix_type: ConcreteMatrixType, value: u64) -> MatrixBound {
        MatrixBound {
            matrix_type,
            coefficient_class: BoundClass::bounded(value.into()),
            metadata: MatrixMetadata::unknown(),
        }
    }

    #[test]
    fn product_uses_proved_zero_rows() {
        let left = bounded(matrix(2, 3), 2);
        let mut right = bounded(matrix(3, 4), 5);
        right.metadata.known_zero_rows = Some(1_u8.into());
        let actual = product_bound(&left, &right).unwrap();
        assert_eq!(actual.coefficient_class, BoundClass::bounded(20_u8.into()));
    }

    #[test]
    fn product_preserves_large_and_zero_annihilation() {
        let left = MatrixBound {
            matrix_type: matrix(1, 1),
            coefficient_class: BoundClass::Large,
            metadata: MatrixMetadata::unknown(),
        };
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
    fn gadget_bounds_reject_invalid_bases() {
        for small in [false, true] {
            for base in [-2, 0, 1] {
                assert_eq!(gadget_digit_bound(&base.into(), small), None);
                assert_eq!(gadget_matrix_bound(&base.into(), small), None);
            }
        }
        assert_eq!(gadget_matrix_bound(&4.into(), false), Some(BoundClass::Large));
        assert_eq!(gadget_matrix_bound(&4.into(), true), Some(BoundClass::bounded(3_u8.into())));
        assert_eq!(gadget_digit_bound(&4.into(), false), Some(BoundClass::bounded(2_u8.into())));
    }
}
