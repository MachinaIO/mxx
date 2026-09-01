//! Exact right-preimage relation bookkeeping.

use crate::{FamilyViewId, SelectorId, SourceId, state::MatrixState};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RightPreimage {
    pub source: SourceId,
    pub target: FamilyViewId,
    pub view: Option<FamilyViewId>,
    pub selector: Option<SelectorId>,
}

/// Consume `value * preimage` using the one exact relation attached to the
/// direct preimage wire.  The target state is deliberately copied from the
/// ordinary evaluator state; it is not a second numeric domain.
pub(crate) fn consume(
    value: &MatrixState,
    preimage: &MatrixState,
    target: &MatrixState,
    relation: &RightPreimage,
    preimage_view: FamilyViewId,
    target_view: FamilyViewId,
    geometry: crate::ProductGeometry,
    modulus: &num_bigint::BigInt,
) -> Result<MatrixState, crate::state::StateError> {
    // The canonical views are part of the proof obligation.  A relation
    // copied onto a different wire (or specialized with a different target)
    // must never be consumed merely because its numeric state happens to fit.
    if relation.view != Some(preimage_view) || relation.target != target_view {
        return Err(crate::state::StateError::RelationUnavailable);
    }
    let left_gain = if let Some(value_carrier) = value.right_carrier.as_ref() {
        if value_carrier.source != relation.source {
            return Err(crate::state::StateError::RelationSourceMismatch {
                expected: relation.source,
                actual: value_carrier.source,
            });
        }
        value_carrier.left_gain.clone()
    } else {
        return Err(crate::state::StateError::RelationUnavailable);
    };
    // The relation is B*K = T + eT, while the left input is X = L*B + eX.
    // Expanding X*K gives L*T + L*eT + eX*K; the returned error keeps exactly
    // the two observable error contributions shown below.
    let error = &left_gain * &target.error_bound +
        crate::right_action_gain(preimage, geometry)? * &value.error_bound;
    let magnitude = crate::product_bound(
        &value.coefficient_magnitude_bound,
        &preimage.coefficient_magnitude_bound,
        geometry,
        value.is_constant_polynomial,
        preimage.is_constant_polynomial,
    )?;
    // K consumes the source carrier for B, but T's carrier is preserved as
    // the output carrier and receives the same left-action gain L.
    Ok(MatrixState {
        error_bound: error,
        coefficient_magnitude_bound: crate::bound::cap_by_centered_residue(magnitude, modulus)?,
        is_constant_polynomial: value.is_constant_polynomial && preimage.is_constant_polynomial,
        right_carrier: target.right_carrier.as_ref().map(|target_carrier| {
            crate::state::RightCarrier {
                source: target_carrier.source,
                left_gain: &left_gain * &target_carrier.left_gain,
            }
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FamilyViewId, SourceId};
    use num_bigint::{BigInt, BigUint};

    fn matrix(error: u64, gain: u64, source: u32) -> MatrixState {
        MatrixState {
            error_bound: BigUint::from(error),
            coefficient_magnitude_bound: BigUint::from(3u8),
            is_constant_polynomial: true,
            right_carrier: Some(crate::state::RightCarrier {
                source: SourceId(source),
                left_gain: BigUint::from(gain),
            }),
        }
    }

    #[test]
    fn noisy_target_contributes_left_gain_times_target_error() {
        // Model T = M*C + E: the target carrier C must survive consumption,
        // with its gain multiplied by the left carrier gain.
        let value = matrix(5, 2, 7);
        let preimage = matrix(3, 1, 9);
        let target = matrix(11, 4, 8);
        let result = consume(
            &value,
            &preimage,
            &target,
            &RightPreimage {
                source: SourceId(7),
                target: FamilyViewId(1),
                view: Some(FamilyViewId(0)),
                selector: None,
            },
            FamilyViewId(0),
            FamilyViewId(1),
            crate::ProductGeometry { inner_dimension: 1, ring_dimension: 1 },
            &BigInt::from(97),
        )
        .unwrap();
        assert_eq!(result.error_bound, BigUint::from(37u8));
        assert_eq!(result.right_carrier.as_ref().map(|x| x.source), Some(SourceId(8)));
    }

    #[test]
    fn exact_target_leaves_only_left_error_times_preimage_gain() {
        let value = matrix(5, 2, 7);
        let preimage = matrix(3, 1, 9);
        let mut target = matrix(0, 4, 8);
        target.error_bound = BigUint::from(0u8);
        let result = consume(
            &value,
            &preimage,
            &target,
            &RightPreimage {
                source: SourceId(7),
                target: FamilyViewId(1),
                view: Some(FamilyViewId(0)),
                selector: None,
            },
            FamilyViewId(0),
            FamilyViewId(1),
            crate::ProductGeometry { inner_dimension: 1, ring_dimension: 1 },
            &BigInt::from(97),
        )
        .unwrap();
        assert_eq!(result.error_bound, BigUint::from(15u8));
        assert_eq!(result.right_carrier.as_ref().map(|x| x.source), Some(SourceId(8)));
    }

    #[test]
    fn relation_rejects_source_mismatch_and_missing_carrier() {
        let value = matrix(1, 1, 3);
        let target = matrix(1, 1, 4);
        let relation = RightPreimage {
            source: SourceId(9),
            target: FamilyViewId(1),
            view: Some(FamilyViewId(u32::MAX)),
            selector: None,
        };
        let mismatch = consume(
            &value,
            &target,
            &value,
            &relation,
            FamilyViewId(u32::MAX),
            FamilyViewId(1),
            crate::ProductGeometry { inner_dimension: 1, ring_dimension: 1 },
            &BigInt::from(97),
        );
        assert!(matches!(
            mismatch,
            Err(crate::state::StateError::RelationSourceMismatch {
                expected: SourceId(9),
                actual: SourceId(3),
            })
        ));
        let mut no_carrier = value.clone();
        no_carrier.right_carrier = None;
        let relation = RightPreimage {
            source: SourceId(3),
            target: FamilyViewId(1),
            view: Some(FamilyViewId(u32::MAX)),
            selector: None,
        };
        assert!(matches!(
            consume(
                &no_carrier,
                &no_carrier,
                &value,
                &relation,
                FamilyViewId(u32::MAX),
                FamilyViewId(1),
                crate::ProductGeometry { inner_dimension: 1, ring_dimension: 1 },
                &BigInt::from(97)
            ),
            Err(crate::state::StateError::RelationUnavailable)
        ));
    }

    #[test]
    fn relation_rejects_preimage_or_target_view_mismatch() {
        let value = matrix(1, 1, 3);
        let target = matrix(1, 1, 3);
        let relation = RightPreimage {
            source: SourceId(3),
            target: FamilyViewId(7),
            view: Some(FamilyViewId(6)),
            selector: None,
        };
        assert!(matches!(
            consume(
                &value,
                &target,
                &target,
                &relation,
                FamilyViewId(5),
                FamilyViewId(7),
                crate::ProductGeometry { inner_dimension: 1, ring_dimension: 1 },
                &BigInt::from(97),
            ),
            Err(crate::state::StateError::RelationUnavailable)
        ));
        assert!(matches!(
            consume(
                &value,
                &target,
                &target,
                &relation,
                FamilyViewId(6),
                FamilyViewId(8),
                crate::ProductGeometry { inner_dimension: 1, ring_dimension: 1 },
                &BigInt::from(97),
            ),
            Err(crate::state::StateError::RelationUnavailable)
        ));
    }
}
