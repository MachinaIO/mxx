//! Uniform family transfer helpers.

use crate::{AbstractValue, FamilyState, state::StateError};

pub(crate) fn pack(
    shape: Vec<usize>,
    values: &[AbstractValue],
) -> Result<AbstractValue, StateError> {
    let first = values.first().ok_or(StateError::InvalidFamilyShape)?;
    let mut joined = first.clone();
    for value in &values[1..] {
        joined = join(&joined, value)?;
    }
    Ok(AbstractValue::Family(FamilyState::new(shape, joined)?))
}

pub(crate) fn join(
    left: &AbstractValue,
    right: &AbstractValue,
) -> Result<AbstractValue, StateError> {
    match (left, right) {
        (AbstractValue::Matrix(a), AbstractValue::Matrix(b)) => {
            Ok(AbstractValue::Matrix(crate::MatrixState {
                error_bound: a.error_bound.clone().max(b.error_bound.clone()),
                coefficient_magnitude_bound: a
                    .coefficient_magnitude_bound
                    .clone()
                    .max(b.coefficient_magnitude_bound.clone()),
                is_constant_polynomial: a.is_constant_polynomial && b.is_constant_polynomial,
                right_carrier: match (&a.right_carrier, &b.right_carrier) {
                    (Some(a), Some(b)) if a.source == b.source => {
                        Some(crate::state::RightCarrier {
                            source: a.source,
                            left_gain: a.left_gain.clone().max(b.left_gain.clone()),
                        })
                    }
                    (Some(carrier), None)
                        if b.error_bound == 0u8.into() &&
                            b.coefficient_magnitude_bound == 0u8.into() =>
                    {
                        Some(carrier.clone())
                    }
                    (None, Some(carrier))
                        if a.error_bound == 0u8.into() &&
                            a.coefficient_magnitude_bound == 0u8.into() =>
                    {
                        Some(carrier.clone())
                    }
                    (None, None) => None,
                    _ => return Err(StateError::InvalidFamilyShape),
                },
            }))
        }
        (AbstractValue::Integer(a), AbstractValue::Integer(b)) => {
            Ok(AbstractValue::Integer(a.join(b)))
        }
        (AbstractValue::Boolean(left), AbstractValue::Boolean(right)) => {
            Ok(AbstractValue::Boolean(left.join(*right)))
        }
        (AbstractValue::Bytes, AbstractValue::Bytes) => Ok(AbstractValue::Bytes),
        (AbstractValue::Trapdoor(left), AbstractValue::Trapdoor(right)) if left == right => {
            Ok(AbstractValue::Trapdoor(left.clone()))
        }
        (AbstractValue::Family(left), AbstractValue::Family(right))
            if left.shape == right.shape =>
        {
            Ok(AbstractValue::Family(FamilyState::new(
                left.shape.clone(),
                join(left.element.as_ref(), right.element.as_ref())?,
            )?))
        }
        (
            AbstractValue::TypedBlob { type_name, schema_hash },
            AbstractValue::TypedBlob { type_name: other, schema_hash: other_hash },
        ) if type_name == other && schema_hash == other_hash => Ok(left.clone()),
        _ => Err(StateError::InvalidFamilyShape),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        SourceId,
        state::{MatrixState, TrapdoorState},
    };
    use mxx_ir_core::types::ConcreteMatrixType;
    use num_bigint::{BigInt, BigUint};

    fn value_with_gain(source: Option<u32>, error: u32, gain: u32) -> AbstractValue {
        AbstractValue::Matrix(MatrixState {
            error_bound: BigUint::from(error),
            coefficient_magnitude_bound: BigUint::from(2u8),
            is_constant_polynomial: true,
            right_carrier: source.map(|source| crate::state::RightCarrier {
                source: SourceId(source),
                left_gain: BigUint::from(gain),
            }),
        })
    }

    fn value(source: Option<u32>, error: u32) -> AbstractValue {
        value_with_gain(source, error, 1)
    }

    #[test]
    fn pack_joins_uniform_shared_source() {
        let packed =
            pack(vec![2], &[value_with_gain(Some(4), 2, 3), value_with_gain(Some(4), 7, 11)])
                .unwrap();
        let AbstractValue::Family(family) = packed else { panic!("family expected") };
        let AbstractValue::Matrix(matrix) = family.element.as_ref() else {
            panic!("matrix expected")
        };
        assert_eq!(matrix.error_bound, BigUint::from(7u8));
        assert_eq!(matrix.right_carrier.as_ref().map(|x| x.source), Some(SourceId(4)));
        assert_eq!(matrix.right_carrier.as_ref().map(|x| x.left_gain.clone()), Some(11u8.into()));
    }

    #[test]
    fn pack_rejects_mismatched_sources() {
        let packed = pack(vec![2], &[value(Some(4), 2), value(Some(5), 7)]);
        assert!(packed.is_err());
    }

    #[test]
    fn pack_joins_equal_trapdoor_states_and_rejects_mismatches() {
        let trapdoor = || {
            AbstractValue::Trapdoor(TrapdoorState {
                matrix: ConcreteMatrixType::scalar(BigInt::from(17), 4),
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                gadget_base: 2.into(),
                digit_count: 3,
                preimage_max_coefficient_bound: 4.into(),
            })
        };
        assert!(pack(vec![2], &[trapdoor(), trapdoor()]).is_ok());
        let different = AbstractValue::Trapdoor(TrapdoorState {
            matrix: ConcreteMatrixType::scalar(BigInt::from(17), 4),
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            gadget_base: 4.into(),
            digit_count: 3,
            preimage_max_coefficient_bound: 4.into(),
        });
        assert!(pack(vec![2], &[trapdoor(), different]).is_err());
    }
}
