//! Native exponent refresh: CRT routing, public preparation and online evaluation.
//!
//! Trait-based witness and direct-decoder paths keep aggregate-before-switch
//! ordering explicit. Payload-secret/public work belongs to preprocessing;
//! online preparation binds helpers/decoders to the current mask secret.
//! BGG-specific witness binding and compilation live in the backend.

use super::{
    ExponentLUTRefreshDirectScheme, ExponentLUTRefreshScheme, ExponentLUTRefreshWitnessScheme,
};
use crate::{
    noise_refresh::{prf::SparseLwrPrfProgram, simulation::RefreshTranscriptTopology},
    noise_utils::NoiseMoments,
    program::ProgramValue,
    readout::{TerminalError, ValueReadout},
};
use mxx_dsl::{Bytes, DslError, Family, GraphValue, HashTag, Int};
use mxx_ir_core::IntExpr;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

/// Group canonical slot positions by encoding level, in ascending level order.
/// The same ordering binds preprocessing anchors, online shared errors and
/// evaluation batches; family members are never expanded here.
pub(crate) fn group_refresh_levels<L: Ord>(
    levels: impl IntoIterator<Item = L>,
) -> BTreeMap<L, Vec<usize>> {
    let mut groups = BTreeMap::<L, Vec<usize>>::new();
    for (slot, level) in levels.into_iter().enumerate() {
        groups.entry(level).or_default().push(slot);
    }
    groups
}

#[derive(Debug, Error, Eq, PartialEq)]
/// Validation and graph-linkage failures in the RNS refresh protocol.
pub enum RefreshError {
    #[error("refresh CRT layout is empty or inconsistent")]
    /// CRT slots or reconstruction coefficients are empty or inconsistent.
    InvalidLayout,
    #[error("refresh slot packages must be in exact CRT slot order")]
    /// Setup packages are not in canonical zero-based slot order.
    SlotOrderMismatch,
    #[error("refresh decoder is not bound to the supplied anchor")]
    /// A decoder package is bound to a different anchor or target.
    AnchorMismatch,
    #[error("all CRT slots must carry the same fresh-error identity")]
    /// CRT slots do not share one fresh-error source identity.
    FreshErrorMismatch,
    #[error("refresh PRF output is not bound to its canonical program label or output wire")]
    /// A mask or fresh-error result came from the wrong PRF program or label.
    PrfOutputMismatch,
    #[error("refresh target public matrix does not match its setup binding")]
    /// The setup target does not match the supplied public matrix.
    TargetMismatch,
    #[error("refresh graph linkage failed at {0}")]
    /// A required graph attachment or handle relationship is absent.
    GraphLink(&'static str),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// One CRT slot's source, switched, and plaintext moduli.
pub struct RefreshSlot {
    /// Zero-based CRT slot index.
    pub slot: usize,
    /// Full ciphertext modulus for this slot.
    pub q: BigInt,
    /// Exact intermediate ciphertext modulus selected for this slot.
    pub p: BigInt,
    /// Plaintext modulus represented by this slot.
    pub q_t: BigInt,
}

/// Canonical public label attached to a sparse-LWR PRF result used by refresh.
///
/// The label contains only refresh instance and coefficient coordinates. It
/// never contains sparse support, selected buckets, or selector material.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefreshPrfLabel {
    /// A per-CRT-slot mask digit.
    Mask {
        /// Refresh instance identity chosen by the caller.
        refresh_id: [u8; 32],
        /// Zero-based CRT slot.
        slot: usize,
        /// Vector component.
        component: usize,
        /// Polynomial coefficient.
        coefficient: usize,
        /// Base-`p` digit.
        digit: usize,
    },
    /// The one fresh-error digit shared by all CRT slots.
    FreshError {
        /// Refresh instance identity chosen by the caller.
        refresh_id: [u8; 32],
        /// Vector component.
        component: usize,
        /// Polynomial coefficient.
        coefficient: usize,
        /// Base-`p` digit.
        digit: usize,
    },
}

impl RefreshPrfLabel {
    /// Returns the canonical bytes used for the sparse-LWR PRF output identity.
    pub fn canonical_bytes(self) -> Vec<u8> {
        let mut raw = Vec::with_capacity(1 + 32 + 4 * std::mem::size_of::<u64>());
        match self {
            Self::Mask { refresh_id, slot, component, coefficient, digit } => {
                raw.push(0);
                raw.extend_from_slice(&refresh_id);
                raw.extend((slot as u64).to_le_bytes());
                raw.extend((component as u64).to_le_bytes());
                raw.extend((coefficient as u64).to_le_bytes());
                raw.extend((digit as u64).to_le_bytes());
            }
            Self::FreshError { refresh_id, component, coefficient, digit } => {
                raw.push(1);
                raw.extend_from_slice(&refresh_id);
                raw.extend((component as u64).to_le_bytes());
                raw.extend((coefficient as u64).to_le_bytes());
                raw.extend((digit as u64).to_le_bytes());
            }
        }
        raw
    }
}

/// Canonical flat index for all PRF labels consumed by one refresh.
///
/// The order is mask slots first, followed by the shared fresh-error group;
/// inside each group it is component-major, coefficient-major, then
/// base-`p`-digit-major. Mask and fresh-error groups intentionally have
/// independent digit cardinalities: `d_m` is used for every mask group and
/// `d_e` for the shared fresh-error group. The component count is the selected
/// scheme's encoding polynomial count, taken from the same topology used for
/// noise accounting. This is public metadata only and contains no
/// selector, support, schedule, or plaintext material.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RefreshPrfLabelIndex {
    refresh_id: [u8; 32],
    topology: RefreshTranscriptTopology,
    mask_offsets: Vec<usize>,
}

impl RefreshPrfLabelIndex {
    /// Creates a canonical label index for a refresh instance.
    pub fn new(
        refresh_id: [u8; 32],
        topology: &RefreshTranscriptTopology,
    ) -> Result<Self, RefreshError> {
        // The topology checks the complete cardinality once; its positive
        // group sizes make every prefix operation below fit as well.
        topology.joint_event_count().map_err(|_| RefreshError::InvalidLayout)?;
        let coordinates = topology.component_columns * topology.coefficient_count;
        let mut mask_offsets = Vec::with_capacity(topology.crt_slot_count + 1);
        mask_offsets.push(0);
        for digits in &topology.mask_digit_counts {
            mask_offsets.push(mask_offsets.last().unwrap() + coordinates * digits);
        }
        Ok(Self { refresh_id, topology: topology.clone(), mask_offsets })
    }

    /// Returns the total number of mask and fresh-error labels.
    pub fn len(&self) -> usize {
        self.mask_offsets[self.topology.crt_slot_count] + self.fresh_group_size()
    }

    /// Returns whether the index contains no labels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Public slice of the canonical label-major output family. `None` is the
    /// single fresh-error group, shared unchanged by every CRT channel.
    pub fn group_range(&self, slot: Option<usize>) -> (usize, usize) {
        match slot {
            Some(slot) => {
                (self.mask_offsets[slot], self.mask_offsets[slot + 1] - self.mask_offsets[slot])
            }
            None => (self.mask_offsets[self.topology.crt_slot_count], self.fresh_group_size()),
        }
    }

    /// Resolve a digit group structurally; this preserves the original graph
    /// handles and shares the fresh group across CRT channels.
    fn group<T: GraphValue>(
        &self,
        values: &Family<T>,
        slot: Option<usize>,
    ) -> Result<ProgramValue<T, Family<T>>, DslError> {
        let (start, count) = self.group_range(slot);
        Ok(ProgramValue::Family(mxx_dsl::parallel(count, |index| Ok(values.at(index.add(start))))?))
    }

    /// Returns the canonical label at `index`.
    pub fn label(&self, index: usize) -> Option<RefreshPrfLabel> {
        if index >= self.len() {
            return None;
        }
        let mask_total = self.mask_offsets[self.topology.crt_slot_count];
        if index < mask_total {
            let group = self.mask_offsets.partition_point(|offset| *offset <= index) - 1;
            let within = index - self.mask_offsets[group];
            let digits = self.topology.mask_digit_counts[group];
            let digit = within % digits;
            let coefficient = (within / digits) % self.topology.coefficient_count;
            let component = within / (digits * self.topology.coefficient_count);
            Some(RefreshPrfLabel::Mask {
                refresh_id: self.refresh_id,
                slot: group,
                component,
                coefficient,
                digit,
            })
        } else {
            let within = index - mask_total;
            let digit = within % self.topology.fresh_digit_count;
            let coefficient =
                (within / self.topology.fresh_digit_count) % self.topology.coefficient_count;
            let component =
                within / (self.topology.fresh_digit_count * self.topology.coefficient_count);
            Some(RefreshPrfLabel::FreshError {
                refresh_id: self.refresh_id,
                component,
                coefficient,
                digit,
            })
        }
    }

    /// Returns the canonical flat index of `label`, rejecting a label from a
    /// different refresh instance or outside this index's dimensions.
    pub fn index_of(&self, label: RefreshPrfLabel) -> Option<usize> {
        let (group_offset, component, coefficient, digit, is_mask) = match label {
            RefreshPrfLabel::Mask { refresh_id, slot, component, coefficient, digit }
                if refresh_id == self.refresh_id && slot < self.topology.crt_slot_count =>
            {
                (slot, component, coefficient, digit, true)
            }
            RefreshPrfLabel::FreshError { refresh_id, component, coefficient, digit }
                if refresh_id == self.refresh_id =>
            {
                (0, component, coefficient, digit, false)
            }
            _ => return None,
        };
        let digit_count = if is_mask {
            self.topology.mask_digit_counts[group_offset]
        } else {
            self.topology.fresh_digit_count
        };
        if component >= self.topology.component_columns ||
            coefficient >= self.topology.coefficient_count ||
            digit >= digit_count
        {
            return None;
        }
        let offset = component * self.topology.coefficient_count * digit_count +
            coefficient * digit_count +
            digit;
        Some(if is_mask {
            self.mask_offsets[group_offset] + offset
        } else {
            self.mask_offsets[self.topology.crt_slot_count] + offset
        })
    }

    fn fresh_group_size(&self) -> usize {
        self.topology.component_columns *
            self.topology.coefficient_count *
            self.topology.fresh_digit_count
    }
}

/// Immutable sparse-LWR contract expected by one refresh setup.
///
/// The contract is derived from an independently constructed
/// [`SparseLwrPrfProgram`], rather than from the outputs being validated.  It
/// therefore prevents a producer from making its own program identity or
/// terminal form authoritative after the fact.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct RefreshPrfContract {
    program_id: crate::program::ExponentLutProgramId,
    output_wire: crate::program::WireId,
    profile: super::prf::SparseLwrPrfProfile,
}

impl RefreshPrfContract {
    /// Derives the contract from the expected canonical sparse-LWR program.
    pub(crate) fn from_program(program: &SparseLwrPrfProgram) -> Self {
        let profile = program.profile();
        Self { program_id: program.id(), output_wire: program.output(), profile }
    }

    /// Checks that one output descriptor is exactly the contracted terminal.
    pub(crate) const fn program_id(&self) -> crate::program::ExponentLutProgramId {
        self.program_id
    }

    pub(crate) const fn q(&self) -> usize {
        // Construction and profile deserialization establish these ranges.
        // Accessors need no repeated validation or cached dimension copies.
        2 * self.profile.native_width()
    }

    pub(crate) const fn p(&self) -> usize {
        2 * self.profile.p_prime
    }

    pub(crate) const fn native_width(&self) -> usize {
        self.profile.native_width()
    }

    pub(crate) const fn ring_dimension(&self) -> usize {
        1usize << self.profile.native_ring_log2()
    }

    pub(crate) const fn output_wire(&self) -> crate::program::WireId {
        self.output_wire
    }
}

/// A PRF output family in canonical refresh-label order.
///
/// Setup binds the PRF contract and layout before constructing this routing
/// value. Only fixed-size mask/fresh slices are gathered here; the production
/// path never materializes a per-label wrapper.
#[derive(Clone)]
pub(crate) struct RefreshPrfFamilyMaterial<V: GraphValue> {
    labels: RefreshPrfLabelIndex,
    vectors: Family<V>,
}

impl<V: GraphValue> RefreshPrfFamilyMaterial<V> {
    pub(crate) fn new(
        vectors: Family<V>,
        labels: RefreshPrfLabelIndex,
    ) -> Result<Self, RefreshError> {
        if !family_has_count(&vectors, labels.len())? {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(Self { labels, vectors })
    }
}

fn family_has_count<V: GraphValue>(
    family: &Family<V>,
    expected: usize,
) -> Result<bool, RefreshError> {
    Ok(family
        .count()
        .evaluate(&Default::default())
        .map_err(|_| RefreshError::InvalidLayout)?
        .to_usize() ==
        Some(expected))
}

/// Resolve a canonical flat digit label into slot, digit, coefficient and
/// component coordinates. This metadata is independent of encoding layout.
pub(crate) fn route_coordinates(
    flat: IntExpr,
    component_count: usize,
    coefficient_count: usize,
    digit_counts: &[usize],
) -> Result<(IntExpr, IntExpr, IntExpr, IntExpr), RefreshError> {
    if digit_counts.is_empty() || digit_counts.iter().any(|digits| *digits == 0) {
        return Err(RefreshError::InvalidLayout);
    }
    let coordinates = component_count
        .checked_mul(coefficient_count)
        .filter(|count| *count > 0)
        .ok_or(RefreshError::InvalidLayout)?;
    let mut offsets = vec![0usize];
    for digits in digit_counts {
        let next = coordinates
            .checked_mul(*digits)
            .and_then(|count| offsets.last().unwrap().checked_add(count))
            .ok_or(RefreshError::InvalidLayout)?;
        offsets.push(next);
    }
    let total = IntExpr::constant(*offsets.last().unwrap());
    let mut slot = IntExpr::constant(0);
    for offset in offsets.iter().skip(1).take(digit_counts.len() - 1) {
        let above = IntExpr::FloorDiv(
            Box::new(IntExpr::Sub(
                Box::new(IntExpr::Add(Box::new(flat.clone()), Box::new(total.clone()))),
                Box::new(IntExpr::constant(*offset)),
            )),
            Box::new(total.clone()),
        );
        slot = IntExpr::Add(Box::new(slot), Box::new(above));
    }
    let slot = slot.canonicalize();
    let offset = IntExpr::Select {
        selector: Box::new(slot.clone()),
        branches: offsets.into_iter().take(digit_counts.len()).map(IntExpr::constant).collect(),
    };
    let digits = IntExpr::Select {
        selector: Box::new(slot.clone()),
        branches: digit_counts.iter().copied().map(IntExpr::constant).collect(),
    };
    let local = IntExpr::Sub(Box::new(flat), Box::new(offset));
    let digit = IntExpr::Rem(Box::new(local.clone()), Box::new(digits.clone())).canonicalize();
    let quotient = IntExpr::FloorDiv(Box::new(local), Box::new(digits));
    let coefficient =
        IntExpr::Rem(Box::new(quotient.clone()), Box::new(IntExpr::constant(coefficient_count)))
            .canonicalize();
    let component =
        IntExpr::FloorDiv(Box::new(quotient), Box::new(IntExpr::constant(coefficient_count)))
            .canonicalize();
    Ok((slot, digit, coefficient, component))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prf_labels_cover_all_public_key_columns_with_separate_digit_groups() {
        let index = RefreshPrfLabelIndex::new(
            [7; 32],
            &crate::noise_refresh::simulation::RefreshTranscriptTopology {
                crt_slot_count: 2,
                component_columns: 6,
                coefficient_count: 2,
                mask_digit_counts: vec![3; 2],
                fresh_digit_count: 1,
            },
        )
        .expect("label index");
        let mask_group = 6 * 2 * 3;
        let fresh_group = 6 * 2;
        assert_eq!(index.len(), 2 * mask_group + fresh_group);
        for flat in 0..index.len() {
            let label = index.label(flat).expect("label");
            assert_eq!(index.index_of(label), Some(flat));
            match label {
                RefreshPrfLabel::Mask { component, digit, .. } => {
                    assert!(component < 6);
                    assert!(digit < 3);
                }
                RefreshPrfLabel::FreshError { component, digit, .. } => {
                    assert!(component < 6);
                    assert!(digit < 1);
                }
            }
        }
        assert!(matches!(index.label(2 * mask_group), Some(RefreshPrfLabel::FreshError { .. })));
        assert!(
            index
                .index_of(RefreshPrfLabel::Mask {
                    refresh_id: [7; 32],
                    slot: 0,
                    component: 6,
                    coefficient: 0,
                    digit: 0,
                })
                .is_none()
        );
    }

    #[test]
    fn ragged_label_prefixes_match_structural_route_coordinates() {
        let refresh_id = rand::random();
        let digits = vec![1, 3, 2];
        let index = RefreshPrfLabelIndex::new(
            refresh_id,
            &crate::noise_refresh::simulation::RefreshTranscriptTopology {
                crt_slot_count: 3,
                component_columns: 2,
                coefficient_count: 3,
                mask_digit_counts: digits.clone(),
                fresh_digit_count: 2,
            },
        )
        .unwrap();
        let index: RefreshPrfLabelIndex =
            serde_json::from_slice(&serde_json::to_vec(&index).unwrap()).unwrap();
        assert_eq!(index.len(), index.topology.joint_event_count().unwrap());
        assert_eq!(index.mask_offsets, vec![0, 6, 24, 36]);
        assert_eq!(index.len(), 48);
        for flat in 0..index.len() {
            let label = index.label(flat).unwrap();
            assert_eq!(index.index_of(label), Some(flat));
            if let RefreshPrfLabel::Mask { slot, digit, coefficient, component, .. } = label {
                let (actual_slot, actual_digit, actual_coefficient, actual_component) =
                    route_coordinates(IntExpr::constant(flat), 2, 3, &digits).unwrap();
                for (expression, expected) in [
                    (actual_slot, slot),
                    (actual_digit, digit),
                    (actual_coefficient, coefficient),
                    (actual_component, component),
                ] {
                    assert_eq!(
                        expression.evaluate(&Default::default()).unwrap(),
                        BigInt::from(expected)
                    );
                }
            } else {
                assert!(flat >= 36);
            }
        }
        assert!(index.label(48).is_none());
        assert!(
            index
                .index_of(RefreshPrfLabel::Mask {
                    refresh_id,
                    slot: 0,
                    component: 0,
                    coefficient: 0,
                    digit: 1,
                })
                .is_none()
        );
        assert!(
            RefreshPrfLabelIndex::new(
                refresh_id,
                &crate::noise_refresh::simulation::RefreshTranscriptTopology {
                    crt_slot_count: 2,
                    component_columns: 2,
                    coefficient_count: 3,
                    mask_digit_counts: digits,
                    fresh_digit_count: 2,
                }
            )
            .is_err()
        );
        assert!(route_coordinates(0.into(), 2, 3, &[0, 2]).is_err());
        assert!(route_coordinates(0.into(), usize::MAX, 3, &[1, 2]).is_err());
    }

    #[test]
    fn prf_label_boundaries_are_mixed_radix_for_varying_family_shapes() {
        for (slot_count, component_count, coefficient_count, dm, de) in
            [(2, 2, 1, 3, 1), (3, 4, 2, 1, 3), (2, 6, 3, 2, 4)]
        {
            let index = RefreshPrfLabelIndex::new(
                [0x31; 32],
                &crate::noise_refresh::simulation::RefreshTranscriptTopology {
                    crt_slot_count: slot_count,
                    component_columns: component_count,
                    coefficient_count,
                    mask_digit_counts: vec![dm; slot_count],
                    fresh_digit_count: de,
                },
            )
            .expect("mixed-radix label index");
            let mask_group = component_count * coefficient_count * dm;
            let fresh_group = component_count * coefficient_count * de;
            let fresh_start = slot_count * mask_group;
            assert_eq!(index.len(), fresh_start + fresh_group);
            for slot in 0..slot_count {
                for component in 0..component_count {
                    for coefficient in 0..coefficient_count {
                        for digit in 0..dm {
                            let flat = ((slot * component_count + component) * coefficient_count +
                                coefficient) *
                                dm +
                                digit;
                            assert!(
                                matches!(index.label(flat), Some(RefreshPrfLabel::Mask { slot: s, component: c, coefficient: n, digit: d, .. }) if s == slot && c == component && n == coefficient && d == digit)
                            );
                            assert_eq!(index.index_of(index.label(flat).unwrap()), Some(flat));
                        }
                    }
                }
            }
            for component in 0..component_count {
                for coefficient in 0..coefficient_count {
                    for digit in 0..de {
                        let offset = (component * coefficient_count + coefficient) * de + digit;
                        let flat = fresh_start + offset;
                        assert!(
                            matches!(index.label(flat), Some(RefreshPrfLabel::FreshError { component: c, coefficient: n, digit: d, .. }) if c == component && n == coefficient && d == digit)
                        );
                        assert_eq!(index.index_of(index.label(flat).unwrap()), Some(flat));
                    }
                }
            }
            assert!(index.label(fresh_start - 1).is_some());
            assert!(matches!(index.label(fresh_start), Some(RefreshPrfLabel::FreshError { .. })));
        }
    }
}

/// A setup-bound CRT slot. Equal levels promise homogeneous graph schemas;
/// mask and fresh-error labels and decoder equations are bound by setup.
pub struct RefreshSlotValues<S: ExponentLUTRefreshScheme> {
    pub level: S::RefreshLevel,
    pub scaled_state: S::RefreshRow,
    pub mask: S::RefreshRow,
    pub fresh_error: S::RefreshRow,
    pub decoder: S::RefreshDecoder,
}

impl<S: ExponentLUTRefreshScheme> Clone for RefreshSlotValues<S> {
    fn clone(&self) -> Self {
        Self {
            level: self.level.clone(),
            scaled_state: self.scaled_state.clone(),
            mask: self.mask.clone(),
            fresh_error: self.fresh_error.clone(),
            decoder: self.decoder.clone(),
        }
    }
}

/// Per-slot preparation inputs. Scaling and the decoder witness come from
/// preprocessing; the decoder base is generated for the current online secret.
pub struct RefreshSlotPreparation<S: ExponentLUTRefreshWitnessScheme> {
    pub level: S::RefreshLevel,
    pub scaling: S::RefreshScaling,
    pub decoder_base: S::RefreshDecoderBase,
    pub witness: S::RefreshDecoderWitness,
}

pub fn prepare_refresh_slot<S: ExponentLUTRefreshWitnessScheme>(
    scheme: &S,
    state: &S::Encoding,
    mask: S::RefreshRow,
    fresh_error: S::RefreshRow,
    preparation: RefreshSlotPreparation<S>,
) -> RefreshSlotValues<S> {
    RefreshSlotValues {
        level: preparation.level,
        scaled_state: scheme.scale_refresh_input(state, &preparation.scaling),
        mask,
        fresh_error,
        decoder: scheme.prepare_refresh_decoder(preparation.decoder_base, preparation.witness),
    }
}

/// Evaluate one symbolic body per homogeneous level. Slot count affects the
/// family extent, not the number of arithmetic nodes built on the host.
pub fn evaluate_refresh<'a, S: ExponentLUTRefreshScheme + 'a>(
    scheme: &S,
    slots: impl IntoIterator<Item = &'a RefreshSlotValues<S>>,
    recomposition: &S::RefreshRecomposition,
) -> Result<S::Encoding, DslError> {
    let slots = slots.into_iter().collect::<Vec<_>>();
    let count = slots.len();
    let groups = group_refresh_levels(slots.iter().map(|slot| slot.level.clone()));
    if count == 0 {
        return Err(DslError::Schema);
    }
    let mut ordered = vec![None; count];
    for (level, members) in groups {
        let scaled =
            Family::pack(members.iter().map(|index| slots[*index].scaled_state.clone()).collect())?;
        let masks = Family::pack(members.iter().map(|index| slots[*index].mask.clone()).collect())?;
        let fresh =
            Family::pack(members.iter().map(|index| slots[*index].fresh_error.clone()).collect())?;
        let decoders =
            Family::pack(members.iter().map(|index| slots[*index].decoder.clone()).collect())?;
        let results = mxx_dsl::parallel(members.len(), |index| {
            let masked = scheme.add_refresh_rows(scaled.at(&index), masks.at(&index));
            let combined = scheme.add_refresh_rows(masked, fresh.at(&index));
            let switched = scheme.switch_refresh_row(combined, &level);
            let decoded = scheme.apply_refresh_decoder(switched, decoders.at(&index));
            Ok(scheme.extract_refresh_residue(decoded, &level))
        })?;
        for (local, index) in members.into_iter().enumerate() {
            ordered[index] = Some(results.at(local));
        }
    }
    let slots = ordered
        .into_iter()
        .map(|row| row.expect("each slot belongs to one level"))
        .collect::<Vec<_>>();
    Ok(scheme.recompose_refresh_rows(slots, recomposition))
}

/// Route public PRF outputs using the same prepared witnesses consumed online.
/// The canonical label index owns ragged mask ranges and the single fresh
/// source range. Only routes are slot-specific; fresh PRF outputs are shared.
/// No target construction, decomposition or sampling occurs in this traversal.
pub fn prepare_refresh_public_routes<S: ExponentLUTRefreshWitnessScheme>(
    scheme: &S,
    labels: &RefreshPrfLabelIndex,
    outputs: Family<S::RefreshPublicRow>,
    mask_routes: Family<S::RefreshPublicScaling>,
    fresh_routes: Family<S::RefreshPublicScaling>,
) -> Result<(Vec<S::RefreshPublicRow>, Vec<S::RefreshPublicRow>), DslError> {
    aggregate_refresh_routes(
        labels,
        outputs,
        mask_routes,
        fresh_routes,
        |row, route| scheme.scale_refresh_public_input(row, route),
        |left, right| scheme.add_refresh_public(left, right),
    )
}

/// Evaluate the same canonical routing schedule on online rows, then add the
/// backend-prepared public offsets. Fresh inputs remain shared across slots.
pub(crate) fn aggregate_refresh_prf<S: ExponentLUTRefreshWitnessScheme>(
    scheme: &S,
    family: &RefreshPrfFamilyMaterial<S::RefreshRow>,
    mask_routes: Family<S::RefreshPublicScaling>,
    fresh_routes: Family<S::RefreshPublicScaling>,
    mask_offsets: Vec<S::RefreshRow>,
    fresh_offsets: Vec<S::RefreshRow>,
) -> Result<(Vec<S::RefreshRow>, Vec<S::RefreshRow>), DslError> {
    if mask_offsets.len() != family.labels.topology.crt_slot_count ||
        fresh_offsets.len() != family.labels.topology.crt_slot_count
    {
        return Err(DslError::FamilyCountMismatch);
    }
    let (masks, fresh) = aggregate_refresh_routes(
        &family.labels,
        family.vectors.clone(),
        mask_routes,
        fresh_routes,
        |row, route| scheme.route_refresh_row(row, route),
        |left, right| scheme.add_refresh_rows(left, right),
    )?;
    Ok((
        masks
            .into_iter()
            .zip(mask_offsets)
            .map(|(row, offset)| scheme.add_refresh_rows(row, offset))
            .collect(),
        fresh
            .into_iter()
            .zip(fresh_offsets)
            .map(|(row, offset)| scheme.add_refresh_rows(row, offset))
            .collect(),
    ))
}

fn aggregate_refresh_routes<V: GraphValue, R: GraphValue>(
    labels: &RefreshPrfLabelIndex,
    outputs: Family<V>,
    mask_routes: Family<R>,
    fresh_routes: Family<R>,
    route: impl Fn(V, R) -> V,
    add: impl Fn(V, V) -> V,
) -> Result<(Vec<V>, Vec<V>), DslError> {
    let (fresh_start, fresh_count) = labels.group_range(None);
    let slot_count = labels.topology.crt_slot_count;
    if outputs.count() != &IntExpr::constant(labels.len()) ||
        mask_routes.count() != &IntExpr::constant(fresh_start) ||
        fresh_routes.count() != &IntExpr::constant(fresh_count * slot_count)
    {
        return Err(DslError::FamilyCountMismatch);
    }
    let routed_masks = mxx_dsl::parallel(fresh_start, |index| {
        Ok(route(outputs.at(&index), mask_routes.at(&index)))
    })?;
    let mut masks = Vec::with_capacity(slot_count);
    for slot in 0..slot_count {
        let (start, count) = labels.group_range(Some(slot));
        let slice =
            mxx_dsl::parallel(count, |index| Ok(routed_masks.at(Int::constant(start).add(index))))?;
        let reduced =
            crate::utils::reduce_family_segments(slice, 1, count, |left, right| add(left, right))
                .map_err(|_| DslError::Schema)?;
        masks.push(reduced.at(0));
    }
    let fresh_indices = mxx_dsl::parallel(fresh_count * slot_count, |index| {
        Ok(Int::constant(fresh_start).add(index.rem(Int::constant(fresh_count))))
    })?;
    let repeated_fresh = mxx_dsl::parallel(fresh_indices.count().clone(), |index| {
        Ok(outputs.at(fresh_indices.at(&index)))
    })?;
    let routed_fresh = mxx_dsl::parallel(fresh_routes.count().clone(), |index| {
        Ok(route(repeated_fresh.at(&index), fresh_routes.at(&index)))
    })?;
    let fresh = crate::utils::reduce_family_segments(
        routed_fresh,
        slot_count,
        fresh_count,
        |left, right| add(left, right),
    )
    .map_err(|_| DslError::Schema)?;
    Ok((masks, (0..slot_count).map(|slot| fresh.at(slot)).collect()))
}

/// Prepare scaled public inputs and add routed mask/fresh-error keys before
/// any destination-level switch. Both passes remain structural families.
pub fn prepare_refresh_public_sums<S: ExponentLUTRefreshWitnessScheme>(
    scheme: &S,
    state: S::RefreshPublicRow,
    scaling: Family<S::RefreshPublicScaling>,
    masks: Family<S::RefreshPublicRow>,
    fresh: Family<S::RefreshPublicRow>,
) -> Result<Family<S::RefreshPublicRow>, DslError> {
    if scaling.count() != masks.count() || scaling.count() != fresh.count() {
        return Err(DslError::FamilyCountMismatch);
    }
    let scaled = mxx_dsl::parallel(scaling.count().clone(), |index| {
        Ok(scheme.scale_refresh_public_input(state.clone(), scaling.at(&index)))
    })?;
    mxx_dsl::parallel(scaled.count().clone(), |index| {
        let masked = scheme.add_refresh_public(scaled.at(&index), masks.at(&index));
        Ok(scheme.add_refresh_public(masked, fresh.at(&index)))
    })
}

/// Preprocess the setup-owned CRT groups and restore canonical slot order.
/// Each tuple supplies a level, its slots, target scales and anchor parameters.
/// Setup guarantees that groups partition the slots and each level occurs once.
/// Public anchors stay in group order; targets and witnesses return in slot order.
/// The online secret is absent from this API and from the generated graph.
pub fn prepare_refresh_decoders<S: ExponentLUTRefreshWitnessScheme>(
    scheme: &S,
    groups: &[(
        S::RefreshLevel,
        Vec<usize>,
        Family<S::RefreshTargetScale>,
        S::RefreshAnchorParameters,
    )],
    sums: Family<S::RefreshPublicRow>,
    output: S::RefreshPublicRow,
) -> Result<
    (Vec<S::RefreshPublicRow>, Vec<S::RefreshPublicRow>, Vec<S::RefreshDecoderWitness>),
    DslError,
> {
    let mut public_anchors = Vec::with_capacity(groups.len());
    let mut targets_by_slot = BTreeMap::new();
    let mut witnesses_by_slot = BTreeMap::new();
    for (level, slots, scales, parameters) in groups {
        let sums = Family::pack(slots.iter().map(|slot| sums.at(*slot)).collect())?;
        if sums.count() != scales.count() {
            return Err(DslError::FamilyCountMismatch);
        }
        let anchor = scheme.sample_refresh_anchor(level, parameters);
        public_anchors.push(scheme.refresh_anchor_public(&anchor));
        let reduced = scheme.reduce_refresh_public(output.clone(), level);
        let prepared = mxx_dsl::parallel(sums.count().clone(), |index| {
            let switched = scheme.switch_refresh_public(sums.at(&index), level);
            let scaled = scheme.scale_refresh_public(reduced.clone(), scales.at(&index));
            let target = scheme.subtract_refresh_public(switched, scaled);
            let witness = scheme.sample_refresh_witness(&anchor, target.clone());
            Ok((target, witness))
        })?;
        for (local, slot) in slots.iter().enumerate() {
            let (target, witness) = prepared.at(local);
            targets_by_slot.insert(*slot, target);
            witnesses_by_slot.insert(*slot, witness);
        }
    }
    Ok((
        public_anchors,
        targets_by_slot.into_values().collect(),
        witnesses_by_slot.into_values().collect(),
    ))
}

/// Prepare one online decoder base per public anchor group and restore slot
/// order. Setup guarantees that groups partition the CRT slots and that each
/// level occurs once. Cloning a base shares its sampled error across the group;
/// a new call creates fresh bases for the next online execution.
pub fn prepare_shared_decoder_bases<S: crate::noise_refresh::ExponentLUTRefreshWitnessScheme>(
    groups: &[(S::RefreshLevel, Vec<usize>)],
    public: &[S::RefreshPublicRow],
    secret: &S::RefreshSecret,
    parameters: &S::RefreshBaseParameters,
) -> Result<(Vec<S::RefreshPublicRow>, Vec<S::RefreshDecoderBase>), DslError> {
    if groups.len() != public.len() {
        return Err(DslError::Schema);
    }
    let mut public_by_slot = BTreeMap::new();
    let mut bases_by_slot = BTreeMap::new();
    for ((level, slots), anchor) in groups.iter().zip(public) {
        let base = S::sample_refresh_decoder_base(secret, anchor, level, parameters);
        for slot in slots {
            public_by_slot.insert(*slot, anchor.clone());
            bases_by_slot.insert(*slot, base.clone());
        }
    }
    Ok((public_by_slot.into_values().collect(), bases_by_slot.into_values().collect()))
}

/// Construct the decoder target using the actual finalized public aggregate.
/// No online secret is needed in this pass.
pub fn prepare_direct_refresh_target<S: ExponentLUTRefreshScheme>(
    scheme: &S,
    raw_sum: S::RefreshPublicRow,
    output_key: S::RefreshPublicRow,
    scale: S::RefreshTargetScale,
    level: &S::RefreshLevel,
) -> S::RefreshPublicRow {
    let public = scheme.switch_refresh_public(raw_sum, level);
    let output = scheme.reduce_refresh_public(output_key, level);
    let scaled = scheme.scale_refresh_public(output, scale);
    scheme.subtract_refresh_public(public, scaled)
}

/// A direct-decoder refresh slot compiled entirely before the online secret
/// exists. Data, mask digits and fresh digits are all ordinary terminal terms.
pub struct PreparedDirectRefreshSlot<S: crate::noise_refresh::ExponentLUTRefreshDirectScheme> {
    pub level: S::RefreshLevel,
    pub decoder_target: S::RefreshPublicRow,
    pub(crate) terms: Vec<ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>>>,
    pub(crate) offset: S::RefreshRow,
}

/// One slot's ordered terminal artifacts and decoding level. Keeping them in
/// one record preserves their association through serialization and import.
#[derive(Clone, Serialize, Deserialize)]
pub struct PreparedDirectRefreshSlotDescriptor<L, D> {
    pub(crate) level: L,
    pub(crate) terms: Vec<D>,
}

impl<S: crate::noise_refresh::ExponentLUTRefreshDirectScheme> PreparedDirectRefreshSlot<S> {
    pub(crate) fn export(
        &self,
        graph: mxx_dsl::DslContext,
        prefix: &str,
    ) -> Result<(mxx_dsl::DslContext, PreparedDirectRefreshSlotDescriptor<S::RefreshLevel,
    <ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>> as GraphValue>::Schema>), DslError>{
        let mut graph = graph
            .public_output(format!("{prefix}.target"), self.decoder_target.clone())?
            .public_output(format!("{prefix}.offset"), self.offset.clone())?;
        let mut terms = Vec::with_capacity(self.terms.len());
        for (index, term) in self.terms.iter().enumerate() {
            graph = graph.public_output(format!("{prefix}.term.{index}"), term.clone())?;
            terms.push(term.schema());
        }
        Ok((graph, PreparedDirectRefreshSlotDescriptor { level: self.level.clone(), terms }))
    }

    /// Restore cached raw terms and target directly, without public arithmetic.
    pub(crate) fn import(
        descriptor: &PreparedDirectRefreshSlotDescriptor<
            S::RefreshLevel,
            <ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>> as GraphValue>::Schema,
        >,
        production: &mxx_ir_core::artifact::ProductionId,
        prefix: &str,
        target: &<S::RefreshPublicRow as GraphValue>::Schema,
        offset: &<S::RefreshRow as GraphValue>::Schema,
    ) -> Result<Self, DslError> {
        use mxx_dsl::GraphValueSchema;
        use mxx_ir_core::artifact::ArtifactConfidentiality;
        Ok(Self {
            level: descriptor.level.clone(),
            terms: descriptor
                .terms
                .iter()
                .enumerate()
                .map(|(index, term)| {
                    term.artifact_input(
                        production.clone(),
                        format!("{prefix}.term.{index}"),
                        ArtifactConfidentiality::Public,
                    )
                })
                .collect::<Result<_, _>>()?,
            decoder_target: target.artifact_input(
                production.clone(),
                format!("{prefix}.target"),
                ArtifactConfidentiality::Public,
            )?,
            offset: offset.artifact_input(
                production.clone(),
                format!("{prefix}.offset"),
                ArtifactConfidentiality::Public,
            )?,
        })
    }

    pub fn terms(&self) -> &[ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>>] {
        &self.terms
    }

    pub fn new(
        scheme: &S,
        inputs: &[(
            &crate::scheme::PublicProgramValue<S>,
            &ProgramValue<S::TerminalMaterial, Family<S::TerminalMaterial>>,
        )],
        output_key: S::RefreshPublicRow,
        scale: S::RefreshTargetScale,
        level: S::RefreshLevel,
        offset: S::RefreshRow,
    ) -> Result<Self, crate::readout::TerminalError<S::Error>> {
        use crate::readout::TerminalError;
        // Inputs are local arrays after digit routing/slicing. A scalar is
        // captured directly, and paired families have the same local count.
        let mut terms = Vec::with_capacity(inputs.len());
        let raw = crate::readout::sum_terms(
            inputs.iter().map(|(input, material)| {
                let (cache, raw) =
                    crate::readout::prepare_terminal_value(scheme, input, material, true, Ok)?;
                terms.push(cache);
                Ok(raw.at(0))
            }),
            |left, right| scheme.add_terminal_public(left, right).map_err(TerminalError::Scheme),
        )?;
        let public = scheme.refresh_terminal_public(raw);
        let decoder_target =
            prepare_direct_refresh_target(scheme, public, output_key, scale, &level);
        Ok(Self { level, decoder_target, terms, offset })
    }

    /// Online preparation only. Offsets convert native affine digits to the
    /// specified integer mask/fresh distributions. Subtracting d-offset later
    /// adds this public constant after ModDown, without changing any error.
    pub fn prepare_decoder(
        &self,
        scheme: &S,
        secret: &S::RefreshSecret,
        parameters: &S::RefreshDecoderParameters,
    ) -> S::RefreshDecoder {
        let decoder = scheme.prepare_direct_refresh_decoder(
            secret,
            self.decoder_target.clone(),
            &self.level,
            parameters,
        );
        scheme.offset_refresh_decoder(decoder, self.offset.clone())
    }

    /// Evaluate a bound combined target and only then release its CRT residue.
    /// The evaluator receives helpers and a decoder, never the online secret.
    pub fn evaluate(
        &self,
        scheme: &S,
        inputs: &[(&crate::scheme::EncodingProgramValue<S>, &S::TerminalOnlineHelpers)],
        decoder: S::RefreshDecoder,
    ) -> Result<S::RefreshRow, crate::readout::TerminalError<S::Error>> {
        use crate::readout::TerminalError;
        if inputs.len() != self.terms.len() {
            return Err(TerminalError::InputCount);
        }
        let raw = crate::readout::sum_terms(
            inputs.iter().zip(&self.terms).map(|((input, helpers), term)| {
                crate::readout::evaluate_terminal_value(scheme, input, term, helpers, true, Ok)
                    .map(|raw| raw.at(0))
            }),
            |left, right| scheme.add_terminal_encoding(left, right).map_err(TerminalError::Scheme),
        )?;
        let combined = scheme.refresh_terminal_encoding(raw);
        let switched = scheme.switch_refresh_row(combined, &self.level);
        let decoded = scheme.apply_refresh_decoder(switched, decoder);
        Ok(scheme.extract_refresh_residue(decoded, &self.level))
    }
}

/// Fixed terminal data target and output-key scale for one CRT channel.
/// `spacing` is H_t; the target already contains H_t times the desired scale.
pub struct DirectRefreshChannel<S: ExponentLUTRefreshDirectScheme> {
    pub level: S::RefreshLevel,
    pub spacing: BigUint,
    pub output_scale: S::RefreshTargetScale,
    pub data: ProgramValue<S::TerminalMaterial, Family<S::TerminalMaterial>>,
}

/// Public preparation for a complete refresh. Its terminal terms and decoder
/// targets are cached once; only helpers and decoders depend on the online s.
pub struct PreparedDirectRefresh<S: ExponentLUTRefreshDirectScheme> {
    pub(crate) labels: RefreshPrfLabelIndex,
    pub(crate) slots: Vec<PreparedDirectRefreshSlot<S>>,
    pub(crate) recomposition: S::RefreshRecomposition,
}

/// Ordered refresh artifact declaration. Concrete bindings and schemas come
/// from the adapter; transport and channel ordering are scheme-independent.
#[derive(Clone, Serialize, Deserialize)]
pub struct PreparedDirectRefreshDescriptor<B, K, L, D> {
    prefix: String,
    scheme: B,
    output_key: K,
    labels: RefreshPrfLabelIndex,
    slots: Vec<PreparedDirectRefreshSlotDescriptor<L, D>>,
}

impl<B: Serialize, K: Serialize, L: Serialize, D: Serialize>
    PreparedDirectRefreshDescriptor<B, K, L, D>
{
    /// Bind this digest independently of the transported descriptor.
    pub fn identity(&self) -> Result<[u8; 32], serde_json::Error> {
        let mut hash = Sha256::new();
        hash.update(b"exponent-lut-direct-refresh");
        hash.update(serde_json::to_vec(self)?);
        Ok(hash.finalize().into())
    }
}

impl<S: ExponentLUTRefreshDirectScheme> PreparedDirectRefresh<S> {
    pub fn export(
        &self,
        scheme: &S,
        graph: mxx_dsl::DslContext,
        prefix: &str,
    ) -> Result<(mxx_dsl::DslContext, PreparedDirectRefreshDescriptor<
        S::RefreshArtifactBinding,
        <S::RefreshOutputKey as GraphValue>::Schema,
        S::RefreshLevel,
        <ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>> as GraphValue>::Schema,
    >), DslError>{
        let key = S::refresh_output_key(&self.recomposition);
        let mut graph = graph.public_output(format!("{prefix}.output-key"), key.clone())?;
        let mut slots = Vec::with_capacity(self.slots.len());
        for (index, slot) in self.slots.iter().enumerate() {
            let (next, descriptor) = slot.export(graph, &format!("{prefix}.slot.{index}"))?;
            graph = next;
            slots.push(descriptor);
        }
        Ok((
            graph,
            PreparedDirectRefreshDescriptor {
                prefix: prefix.into(),
                scheme: scheme.refresh_artifact_binding(),
                output_key: key.schema(),
                labels: self.labels.clone(),
                slots,
            },
        ))
    }

    /// Reopen caches without repeating public sums or ModDown. Secret-dependent
    /// helpers and decoders remain in online preparation.
    pub fn import(
        scheme: &S,
        descriptor: &PreparedDirectRefreshDescriptor<
            S::RefreshArtifactBinding,
            <S::RefreshOutputKey as GraphValue>::Schema,
            S::RefreshLevel,
            <ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>> as GraphValue>::Schema,
        >,
        production: mxx_ir_core::artifact::ProductionId,
        expected_identity: [u8; 32],
    ) -> Result<Self, super::setup::RefreshSetupError>
    where
        S::RefreshLevel: Serialize,
        <ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>> as GraphValue>::Schema:
            Serialize,
    {
        use super::setup::RefreshSetupError;
        use mxx_dsl::GraphValueSchema;
        use mxx_ir_core::artifact::ArtifactConfidentiality;
        let identity = descriptor.identity().map_err(|error| RefreshSetupError::SetupStage {
            stage: "serialize refresh descriptor",
            detail: error.to_string(),
        })?;
        if identity != expected_identity || descriptor.scheme != scheme.refresh_artifact_binding() {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let key = descriptor.output_key.artifact_input(
            production.clone(),
            format!("{}.output-key", descriptor.prefix),
            ArtifactConfidentiality::Public,
        )?;
        let (recomposition, target, offset) = scheme
            .import_refresh_recomposition(key)
            .map_err(|error| RefreshSetupError::Backend(Box::new(error)))?;
        let slots = descriptor
            .slots
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                PreparedDirectRefreshSlot::import(
                    slot,
                    &production,
                    &format!("{}.slot.{index}", descriptor.prefix),
                    &target,
                    &offset,
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { labels: descriptor.labels.clone(), slots, recomposition })
    }

    pub fn recomposition(&self) -> &S::RefreshRecomposition {
        &self.recomposition
    }

    pub fn slots(&self) -> &[PreparedDirectRefreshSlot<S>] {
        &self.slots
    }

    pub fn new(
        scheme: &S,
        data: &crate::scheme::PublicProgramValue<S>,
        digits: &crate::scheme::PublicProgramValue<S>,
        readout: &ValueReadout,
        plan: &RefreshDigitPlan,
        labels: RefreshPrfLabelIndex,
        secret: &S::RefreshDigitSecret,
        bank: &S::RefreshDigitMaskBank,
        channels: Vec<DirectRefreshChannel<S>>,
        output_key: S::RefreshPublicRow,
        recomposition: S::RefreshRecomposition,
        hash_key: Bytes,
        tag: HashTag,
        id: [u8; 32],
    ) -> Result<Self, TerminalError<S::Error>> {
        if channels.len() != plan.mask_counts.len() {
            return Err(TerminalError::InputCount);
        }
        let ProgramValue::Family(digits) = digits else {
            return Err(TerminalError::InputCount);
        };
        let fresh = labels.group(digits, None)?;
        let mut slots = Vec::with_capacity(channels.len());
        // This loop visits heterogeneous CRT moduli, never individual digits.
        for (slot, channel) in channels.into_iter().enumerate() {
            let mask = labels.group(digits, Some(slot))?;
            let prepare = |mask: bool| {
                let routing = RefreshDigitRouting {
                    base: plan.base,
                    count: if mask { plan.mask_counts[slot] } else { plan.fresh_count },
                    scale: if mask { 1u8.into() } else { channel.spacing.clone() },
                    mask,
                };
                let mut tag = tag.clone();
                tag.push(mxx_ir_core::IntExpr::constant(slot));
                tag.push(mxx_ir_core::IntExpr::constant(usize::from(mask)));
                let mut identity = Sha256::new();
                identity.update(b"exponent-lut-refresh-digit-material");
                identity.update(id);
                identity.update((slot as u64).to_le_bytes());
                identity.update([u8::from(mask)]);
                scheme
                    .sample_refresh_digit_material(
                        secret,
                        readout,
                        bank,
                        &routing,
                        hash_key.clone(),
                        tag,
                        identity.finalize().into(),
                    )
                    .map_err(TerminalError::Scheme)
            };
            let (mask_material, mask_offset) = prepare(true)?;
            let (fresh_material, fresh_offset) = prepare(false)?;
            let prepared = PreparedDirectRefreshSlot::new(
                scheme,
                &[(data, &channel.data), (&mask, &mask_material), (&fresh, &fresh_material)],
                output_key.clone(),
                channel.output_scale,
                channel.level,
                scheme.add_refresh_rows(mask_offset, fresh_offset),
            )?;
            slots.push(prepared);
        }
        Ok(Self { labels, slots, recomposition })
    }

    pub fn prepare_decoders(
        &self,
        scheme: &S,
        secret: &S::RefreshSecret,
        parameters: &S::RefreshDecoderParameters,
    ) -> Vec<S::RefreshDecoder> {
        self.slots.iter().map(|slot| slot.prepare_decoder(scheme, secret, parameters)).collect()
    }

    pub fn evaluate(
        &self,
        scheme: &S,
        data: &crate::scheme::EncodingProgramValue<S>,
        digits: &crate::scheme::EncodingProgramValue<S>,
        data_helpers: &S::TerminalOnlineHelpers,
        digit_helpers: &S::TerminalOnlineHelpers,
        decoders: Vec<S::RefreshDecoder>,
    ) -> Result<S::Encoding, TerminalError<S::Error>> {
        if decoders.len() != self.slots.len() {
            return Err(TerminalError::InputCount);
        }
        let ProgramValue::Family(digits) = digits else {
            return Err(TerminalError::InputCount);
        };
        let fresh = self.labels.group(digits, None)?;
        let residues = self
            .slots
            .iter()
            .zip(decoders)
            .enumerate()
            .map(|(index, (slot, decoder))| {
                let mask = self.labels.group(digits, Some(index))?;
                slot.evaluate(
                    scheme,
                    &[(data, data_helpers), (&mask, digit_helpers), (&fresh, digit_helpers)],
                    decoder,
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(scheme.recompose_refresh_rows(residues, &self.recomposition))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefreshDigitPlan {
    pub base: usize,
    pub mask_counts: Vec<usize>,
    pub fresh_count: usize,
}

/// One structural group in component/coefficient/digit order. Fresh groups
/// reuse their PRF inputs across CRT channels, with channel-specific scale H.
pub struct RefreshDigitRouting {
    pub base: usize,
    pub count: usize,
    pub scale: BigUint,
    pub mask: bool,
}

impl RefreshDigitRouting {
    pub fn offset_numerator(&self) -> BigInt {
        let support = (0..self.count).fold(BigUint::one(), |value, _| value * self.base);
        let numerator = if self.mask {
            -BigInt::one()
        } else {
            BigInt::from((support - 1u8) / (self.base - 1))
        };
        numerator * BigInt::from(self.scale.clone())
    }
}

#[derive(Debug, Error)]
pub enum DigitPlanError {
    #[error("native LWR requires a power-of-two digit base and nonempty support requirements")]
    Parameters,
    #[error("the mask interval and exposed error do not fit the CRT spacing")]
    MaskRange,
    #[error("the requested fresh digit distribution exceeds the next-state error budget")]
    FreshRange,
}

/// Physical payload moments conditional on independent uniform native digits.
/// `mask` centers their consecutive support; fresh digits retain the native
/// signed representatives. Parameters establish a positive count and even base.
pub(crate) fn native_digit_uniform_moments(
    base: &BigUint,
    count: usize,
    mask: bool,
) -> NoiseMoments {
    let support = (0..count).fold(BigUint::one(), |value, _| value * base);
    let lower = if mask {
        -BigInt::from(&support / 2u8)
    } else {
        BigInt::from((&support - 1u8) / (base - 1u8)) * (BigInt::one() - BigInt::from(base / 2u8))
    };
    NoiseMoments::uniform_interval(lower, support)
}

impl RefreshDigitPlan {
    /// Smallest positive digit counts covering the requested support sizes.
    /// Public parameter selection uses exact products rather than float logs.
    pub fn new(
        base: usize,
        mask_supports: &[BigUint],
        fresh_support: &BigUint,
    ) -> Result<Self, DigitPlanError> {
        if base < 2 ||
            !base.is_power_of_two() ||
            mask_supports.is_empty() ||
            mask_supports.iter().any(BigUint::is_zero) ||
            fresh_support.is_zero()
        {
            return Err(DigitPlanError::Parameters);
        }
        let count = |required: &BigUint| {
            let mut support = BigUint::from(base);
            let mut digits = 1;
            while &support < required {
                support *= base;
                digits += 1;
            }
            digits
        };
        Ok(Self {
            base,
            mask_counts: mask_supports.iter().map(count).collect(),
            fresh_count: count(fresh_support),
        })
    }

    pub fn weights(&self, count: usize) -> Vec<BigUint> {
        let mut weight = BigUint::one();
        (0..count)
            .map(|_| {
                let value = weight.clone();
                weight *= self.base;
                value
            })
            .collect()
    }

    pub fn support(&self, count: usize) -> BigUint {
        (0..count).fold(BigUint::one(), |value, _| value * self.base)
    }

    /// Native integer digit representatives are [1-p/2, p/2]. This preserves
    /// their actual support and mean; it is not a centered-binomial sampler.
    pub fn fresh_interval(&self) -> (BigInt, BigInt) {
        let sum = (self.support(self.fresh_count) - 1u8) / (self.base - 1);
        let upper = BigInt::from(&sum * (self.base / 2));
        let lower = BigInt::from(sum) * (BigInt::one() - BigInt::from(self.base / 2));
        (lower, upper)
    }

    /// Conditional on distinct native PRF digits being independent and uniform.
    /// Positional digits enumerate one consecutive interval bijectively. This
    /// describes one fresh source shared across CRT channels, not one per slot.
    pub fn fresh_uniform_moments(&self) -> NoiseMoments {
        native_digit_uniform_moments(&self.base.into(), self.fresh_count, false)
    }

    /// Conditional uniform centered mask model, with physical mean -1/2.
    pub fn mask_uniform_moments(&self, slot: usize) -> NoiseMoments {
        native_digit_uniform_moments(&self.base.into(), self.mask_counts[slot], true)
    }

    /// Solve the mask-width/error fixed point at a fixed output modulus.
    /// The supplied error function includes all digit emissions and the
    /// decoder; the arithmetic scheme supplies those bounds, not this planner.
    pub fn fit_masks(
        &mut self,
        spacings: &[BigUint],
        observations: &BigUint,
        statistical_bits: usize,
        fresh_bound: &BigUint,
        exposed_errors: impl Fn(&Self) -> Result<Vec<BigUint>, DigitPlanError>,
    ) -> Result<Vec<BigUint>, DigitPlanError> {
        if spacings.len() != self.mask_counts.len() || observations.is_zero() {
            return Err(DigitPlanError::Parameters);
        }
        let (lower, upper) = self.fresh_interval();
        if lower.magnitude().max(upper.magnitude()) > fresh_bound {
            return Err(DigitPlanError::FreshRange);
        }
        loop {
            let errors = exposed_errors(self)?;
            if errors.len() != spacings.len() {
                return Err(DigitPlanError::Parameters);
            }
            let mut changed = false;
            for index in 0..spacings.len() {
                let required = (observations * &errors[index] * 2u8) << statistical_bits;
                let mut support = self.support(self.mask_counts[index]);
                while support < required {
                    support *= self.base;
                    self.mask_counts[index] += 1;
                    changed = true;
                }
                if support + &errors[index] * 2u8 >= spacings[index] {
                    return Err(DigitPlanError::MaskRange);
                }
            }
            if !changed {
                return Ok(errors);
            }
        }
    }

    pub fn topology<S: crate::scheme::ExponentLUTEncodingScheme>(
        &self,
        scheme: &S,
        key: &S::PublicKey,
        env: &mxx_ir_core::ParamEnv,
    ) -> Result<super::simulation::RefreshTranscriptTopology, super::simulation::NoiseSimulationError>
    {
        super::simulation::RefreshTranscriptTopology::new(
            scheme,
            key,
            env,
            self.mask_counts.clone(),
            self.fresh_count,
        )
    }
}

#[cfg(test)]
mod digit_tests {
    use super::*;

    #[test]
    fn test_digit_plan_nonbinary_support_offsets_and_counts() {
        let plan = RefreshDigitPlan::new(4, &[16u8.into(), 17u8.into()], &5u8.into()).unwrap();
        assert_eq!(plan.mask_counts, vec![2, 3]);
        assert_eq!(plan.fresh_count, 2);
        assert_eq!(plan.weights(3), vec![1u8.into(), 4u8.into(), 16u8.into()]);
        assert_eq!(plan.fresh_interval(), ((-5).into(), 10.into()));
        assert_eq!(
            RefreshDigitRouting { base: plan.base, count: 2, scale: 1u8.into(), mask: true }
                .offset_numerator(),
            (-1).into()
        );
        assert_eq!(
            RefreshDigitRouting { base: plan.base, count: 2, scale: 1u8.into(), mask: false }
                .offset_numerator(),
            5.into()
        );
        // Enumerate the actual independent digit distribution as an oracle.
        let sums = (-1..=2)
            .flat_map(|low| (-1..=2).map(move |high| low + 4 * high))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(sums, (-5..=10).collect());
        let moments = plan.fresh_uniform_moments();
        let count = BigInt::from(sums.len());
        let sum: BigInt = sums.iter().copied().map(BigInt::from).sum();
        let squares: BigInt = sums.iter().map(|value| BigInt::from(value * value)).sum();
        assert_eq!(
            &moments.mean_numerator * &count,
            &sum * BigInt::from(moments.mean_denominator.clone()),
        );
        assert_eq!(
            BigInt::from(moments.variance.numerator) * &count * &count,
            (squares * count - &sum * &sum) * BigInt::from(moments.variance.denominator),
        );
        for slot in 0..plan.mask_counts.len() {
            let mask = plan.mask_uniform_moments(slot);
            assert_eq!(mask.mean_numerator, (-1).into());
            assert_eq!(mask.mean_denominator, 2u8.into());
            let support = plan.support(plan.mask_counts[slot]);
            assert_eq!(
                mask.variance.scaled(&12u8.into()),
                crate::noise_utils::NonnegativeRational::new(
                    &support * &support - 1u8, 1u8.into(),
                ).unwrap(),
            );
        }
    }

    #[test]
    fn test_digit_plan_recomputes_error_after_mask_count_growth() {
        let mut plan = RefreshDigitPlan::new(16, &[1u8.into()], &17u8.into()).unwrap();
        let result = plan
            .fit_masks(&[BigUint::from(1u8) << 64], &16u8.into(), 8, &200u8.into(), |current| {
                Ok(vec![BigUint::from(current.mask_counts[0] + current.fresh_count)])
            })
            .unwrap();
        assert_eq!(result[0], BigUint::from(plan.mask_counts[0] + plan.fresh_count));
        assert!(plan.support(plan.mask_counts[0]) >= (&result[0] * 32u8) << 8);
        assert!(plan.support(plan.mask_counts[0] - 1) < (&result[0] * 32u8) << 8);
    }
}
