//! Native exponent refresh: CRT routing and noise-model utilities.
use crate::{
    noise_refresh::{
        prf::{PreparedSparseLwrPrf, PrfError, PrfLabel},
        simulation::RefreshTranscriptTopology,
    },
    noise_utils::{NoiseMomentPolicy, NoiseMoments},
    program::{ExponentLutProgram, ExponentLutProgramId, Operand, WireId},
};
use mxx_ir_core::types::WireType;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

/// Finish one concrete refresh stage from already validated PRF outputs.
///
/// [`RefreshDeclaration`] is the sole immutable owner of refresh metadata and
/// canonical PRF labels. The common `api::apply_refresh` orchestration invokes
/// the PRF exactly once and validates its output key set before calling this
/// stage-specific operation.
pub trait NoiseRefresh {
    type PrfOutput;
    type Output;
    type Error: std::error::Error + Send + Sync + 'static;

    fn finish_noise_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        outputs: BTreeMap<PrfLabel, Self::PrfOutput>,
    ) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, Error, Eq, PartialEq)]
/// Validation and graph-linkage failures in the refresh protocol.
pub enum RefreshError {
    #[error("refresh CRT layout is empty or inconsistent")]
    /// CRT slots or reconstruction coefficients are empty or inconsistent.
    InvalidLayout,
    #[error("refresh decoder is not bound to the supplied anchor")]
    /// A decoder package is bound to a different anchor or target.
    AnchorMismatch,
    #[error("refresh PRF output is not bound to its canonical program label or output wire")]
    /// A mask or fresh-error result came from the wrong PRF program or label.
    PrfOutputMismatch,
    #[error("refresh target public matrix does not match its setup binding")]
    /// The setup target does not match the supplied public matrix.
    TargetMismatch,
    #[error("refresh graph linkage failed at {0}")]
    /// A required graph attachment or handle relationship is absent.
    GraphLink(&'static str),
    #[error("refresh source output is not a declared exponent output")]
    InvalidSourceOutput,
    #[error("refresh output schema is empty or inconsistent")]
    InvalidOutputSchema,
    #[error("refresh modulus relations are inconsistent")]
    InvalidModulusRelation,
    #[error("refresh metadata commitment failed")]
    Commitment,
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
pub(crate) enum RefreshPrfLabel {
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
    pub(crate) fn canonical_bytes(self) -> Vec<u8> {
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

    /// Returns the canonical label at `index`.
    pub(crate) fn label(&self, index: usize) -> Option<RefreshPrfLabel> {
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
    pub(crate) fn index_of(&self, label: RefreshPrfLabel) -> Option<usize> {
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

    /// Returns the PRF labels corresponding to this index in canonical
    /// label-major order.  The application-owned coordinates are encoded as
    /// one opaque component so that the sparse-LWR declaration never needs to
    /// parse refresh semantics.
    pub fn prf_labels(&self) -> Result<Vec<PrfLabel>, PrfError> {
        (0..self.len())
            .map(|index| {
                let label =
                    self.label(index).ok_or(PrfError::InvalidSparseLwrBlock)?.canonical_bytes();
                PrfLabel::from_components(&[label.as_slice()])
            })
            .collect()
    }

    fn fresh_group_size(&self) -> usize {
        self.topology.component_columns *
            self.topology.coefficient_count *
            self.topology.fresh_digit_count
    }
}

/// Immutable, checked declaration shared by preparation, online evaluation,
/// and noise simulation for one refresh operation.
///
/// The declaration owns only protocol metadata and the structural PRF
/// preparation.  It does not contain a scheme carrier, plaintext, sampled
/// matrix, or runtime graph handle.  Concrete stages bind those values through
/// their `NoiseRefresh` implementation when the common refresh dispatch is
/// called.
#[derive(Clone, Debug)]
pub struct RefreshDeclaration {
    source_program: ExponentLutProgramId,
    source_output: WireId,
    output_schema: Vec<WireType>,
    slots: Vec<RefreshSlot>,
    digits: RefreshDigitPlan,
    labels: RefreshPrfLabelIndex,
    prepared: PreparedSparseLwrPrf,
    target_level: usize,
    moment_policy: NoiseMomentPolicy,
    prf_labels: Vec<PrfLabel>,
    commitment: [u8; 32],
}

impl RefreshDeclaration {
    /// Checks and freezes one refresh declaration.
    pub fn new(
        source_program: &ExponentLutProgram,
        source_output: WireId,
        output_schema: Vec<WireType>,
        slots: Vec<RefreshSlot>,
        digits: RefreshDigitPlan,
        labels: RefreshPrfLabelIndex,
        prepared: PreparedSparseLwrPrf,
        target_level: usize,
        moment_policy: NoiseMomentPolicy,
    ) -> Result<Self, RefreshError> {
        validate_refresh_source(source_program, source_output, &output_schema)?;
        validate_refresh_layout(&slots, &digits, &labels)?;
        validate_modulus_relations(&slots)?;
        validate_refresh_prf_binding(&digits, &prepared)?;
        let prf_labels = labels.prf_labels().map_err(|_| RefreshError::PrfOutputMismatch)?;
        let declared_outputs = prepared
            .declaration()
            .ordered_outputs()
            .map(|(label, _)| label.clone())
            .collect::<Vec<_>>();
        if declared_outputs != prf_labels {
            return Err(RefreshError::PrfOutputMismatch);
        }
        let source_program_id = source_program.id();
        let commitment = refresh_commitment(
            source_program_id,
            source_output,
            &output_schema,
            &slots,
            &digits,
            &labels,
            &prepared,
            target_level,
            &moment_policy,
        )?;
        Ok(Self {
            source_program: source_program_id,
            source_output,
            output_schema,
            slots,
            digits,
            labels,
            prepared,
            target_level,
            moment_policy,
            prf_labels,
            commitment,
        })
    }

    pub const fn source_program(&self) -> ExponentLutProgramId {
        self.source_program
    }

    pub const fn source_output(&self) -> WireId {
        self.source_output
    }

    pub fn output_schema(&self) -> &[WireType] {
        &self.output_schema
    }

    pub fn slots(&self) -> &[RefreshSlot] {
        &self.slots
    }

    pub fn digits(&self) -> &RefreshDigitPlan {
        &self.digits
    }

    pub fn labels(&self) -> &RefreshPrfLabelIndex {
        &self.labels
    }

    pub fn prepared(&self) -> &PreparedSparseLwrPrf {
        &self.prepared
    }

    pub const fn target_level(&self) -> usize {
        self.target_level
    }

    pub fn moment_policy(&self) -> &NoiseMomentPolicy {
        &self.moment_policy
    }

    pub const fn commitment(&self) -> &[u8; 32] {
        &self.commitment
    }

    pub(crate) fn prf_labels(&self) -> &[PrfLabel] {
        &self.prf_labels
    }
}

fn validate_refresh_source(
    program: &ExponentLutProgram,
    source_output: WireId,
    output_schema: &[WireType],
) -> Result<(), RefreshError> {
    if output_schema.is_empty() {
        return Err(RefreshError::InvalidOutputSchema);
    }
    let is_declared_output = program
        .outputs()
        .iter()
        .any(|operand| matches!(operand, Operand::Scalar(output) if *output == source_output));
    if !is_declared_output ||
        program
            .wire_is_coefficient_output(source_output)
            .map_err(|_| RefreshError::InvalidSourceOutput)?
    {
        return Err(RefreshError::InvalidSourceOutput);
    }
    Ok(())
}

fn validate_modulus_relations(slots: &[RefreshSlot]) -> Result<(), RefreshError> {
    for slot in slots {
        // q is the preterminal modulus.  p is the exact intermediate
        // modulus M_t and q_t is the plaintext prime pi_t.
        if slot.q <= slot.p ||
            (&slot.q % &slot.p) != BigInt::zero() ||
            (&slot.p % &slot.q_t) != BigInt::zero()
        {
            return Err(RefreshError::InvalidModulusRelation);
        }
        // The slot's intermediate modulus is M_t.  With p=M_t, the
        // state-switch factor is H_t = M_t/q_t and must be invertible modulo
        // the plaintext modulus; otherwise CRT recomposition cannot recover
        // this channel.
        let h_t = &slot.p / &slot.q_t;
        if positive_gcd(&h_t, &slot.q_t) != BigInt::one() {
            return Err(RefreshError::InvalidModulusRelation);
        }
    }
    for (index, left) in slots.iter().enumerate() {
        for right in slots.iter().skip(index + 1) {
            if positive_gcd(&left.q_t, &right.q_t) != BigInt::one() {
                return Err(RefreshError::InvalidModulusRelation);
            }
        }
    }
    Ok(())
}

fn positive_gcd(left: &BigInt, right: &BigInt) -> BigInt {
    let mut left = left.clone();
    let mut right = right.clone();
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    left
}

fn refresh_commitment(
    source_program: ExponentLutProgramId,
    source_output: WireId,
    output_schema: &[WireType],
    slots: &[RefreshSlot],
    digits: &RefreshDigitPlan,
    labels: &RefreshPrfLabelIndex,
    prepared: &PreparedSparseLwrPrf,
    target_level: usize,
    moment_policy: &NoiseMomentPolicy,
) -> Result<[u8; 32], RefreshError> {
    let metadata = (
        source_program,
        source_output,
        output_schema,
        slots,
        digits,
        labels,
        prepared.descriptor(),
        target_level,
        moment_policy,
    );
    let bytes = serde_json::to_vec(&metadata).map_err(|_| RefreshError::Commitment)?;
    let mut digest = Sha256::new();
    digest.update(b"mxx-exponent-lut/refresh-declaration");
    digest.update(bytes);
    Ok(digest.finalize().into())
}

fn validate_refresh_layout(
    slots: &[RefreshSlot],
    digits: &RefreshDigitPlan,
    labels: &RefreshPrfLabelIndex,
) -> Result<(), RefreshError> {
    if slots.is_empty() ||
        slots.iter().enumerate().any(|(index, slot)| {
            slot.slot != index || slot.q <= 0.into() || slot.p <= 0.into() || slot.q_t <= 0.into()
        })
    {
        return Err(RefreshError::InvalidLayout);
    }
    if digits.base < 2 ||
        !digits.base.is_power_of_two() ||
        digits.mask_counts.len() != slots.len() ||
        digits.mask_counts.iter().any(|count| *count == 0) ||
        digits.fresh_count == 0 ||
        labels.is_empty()
    {
        return Err(RefreshError::InvalidLayout);
    }

    let mut coordinates = std::collections::BTreeSet::new();
    let mut mask_coordinates = vec![std::collections::BTreeSet::new(); slots.len()];
    let mut mask_digits = vec![std::collections::BTreeSet::new(); slots.len()];
    let mut fresh_digits = std::collections::BTreeSet::new();
    for index in 0..labels.len() {
        let label = labels.label(index).ok_or(RefreshError::InvalidLayout)?;
        if labels.index_of(label) != Some(index) {
            return Err(RefreshError::InvalidLayout);
        }
        match label {
            RefreshPrfLabel::Mask { slot, component, coefficient, digit, .. } => {
                if slot >= slots.len() || digit >= digits.mask_counts[slot] {
                    return Err(RefreshError::InvalidLayout);
                }
                let coordinate = (component, coefficient);
                coordinates.insert(coordinate);
                mask_coordinates[slot].insert(coordinate);
                mask_digits[slot].insert(digit);
            }
            RefreshPrfLabel::FreshError { component, coefficient, digit, .. } => {
                if digit >= digits.fresh_count {
                    return Err(RefreshError::InvalidLayout);
                }
                coordinates.insert((component, coefficient));
                fresh_digits.insert(digit);
            }
        }
    }
    if coordinates.is_empty() ||
        fresh_digits.len() != digits.fresh_count ||
        (0..slots.len()).any(|slot| {
            mask_coordinates[slot] != coordinates ||
                mask_digits[slot].len() != digits.mask_counts[slot]
        })
    {
        return Err(RefreshError::InvalidLayout);
    }
    Ok(())
}

fn validate_refresh_prf_binding(
    digits: &RefreshDigitPlan,
    prepared: &PreparedSparseLwrPrf,
) -> Result<(), RefreshError> {
    let program = prepared.declaration();
    let profile = program.profile();
    if profile.p().map_err(|_| RefreshError::InvalidLayout)? != digits.base {
        return Err(RefreshError::InvalidLayout);
    }
    crate::noise_refresh::prf::validate_profile_program(profile, program)
        .map_err(|_| RefreshError::PrfOutputMismatch)
}

/// Resolve a canonical flat digit label into slot, digit, coefficient and
#[cfg(test)]
mod tests {
    use super::*;

    fn source_program() -> (ExponentLutProgram, WireId) {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let output = builder.input(crate::program::AttributeKnowledge::PublicPayload).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        (program, output)
    }

    #[derive(Debug)]
    struct CallbackError(PrfError);

    impl std::fmt::Display for CallbackError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(formatter)
        }
    }

    impl std::error::Error for CallbackError {}

    impl From<PrfError> for CallbackError {
        fn from(error: PrfError) -> Self {
            Self(error)
        }
    }

    struct CallbackRefresh {
        finished: bool,
    }

    impl NoiseRefresh for CallbackRefresh {
        type PrfOutput = usize;
        type Output = usize;
        type Error = CallbackError;

        fn finish_noise_refresh(
            &mut self,
            _declaration: &RefreshDeclaration,
            outputs: BTreeMap<PrfLabel, Self::PrfOutput>,
        ) -> Result<Self::Output, Self::Error> {
            self.finished = true;
            Ok(outputs.values().sum())
        }
    }

    #[test]
    fn apply_refresh_invokes_the_prf_callback_once() {
        let (_, _, declaration) = test_declaration();
        let labels = declaration.prf_labels().to_vec();
        let mut refresh = CallbackRefresh { finished: false };
        let mut callbacks = 0;
        let output = crate::api::apply_refresh(
            &declaration,
            &mut refresh,
            |callback_declaration, callback_labels| {
                callbacks += 1;
                assert!(std::ptr::eq(callback_declaration, &declaration));
                assert_eq!(callback_labels, labels.as_slice());
                Ok(callback_labels.iter().cloned().map(|label| (label, 7)).collect())
            },
        )
        .unwrap();
        assert_eq!(callbacks, 1);
        assert!(refresh.finished);
        assert_eq!(output, labels.len() * 7);
    }

    #[test]
    fn apply_refresh_rejects_inexact_prf_output_keys() {
        let (_, _, declaration) = test_declaration();
        let labels = declaration.prf_labels().to_vec();

        let mut missing = CallbackRefresh { finished: false };
        let result = crate::api::apply_refresh(&declaration, &mut missing, |_, _| {
            Ok(labels.iter().take(1).cloned().map(|label| (label, 1)).collect())
        });
        assert!(matches!(result, Err(crate::api::RefreshApplicationError::OutputMismatch)));
        assert!(!missing.finished);

        let mut extra = CallbackRefresh { finished: false };
        let undeclared = PrfLabel::from_components(&[b"undeclared"]).unwrap();
        let result = crate::api::apply_refresh(&declaration, &mut extra, |_, _| {
            Ok(BTreeMap::from([(labels[0].clone(), 1), (undeclared, 1)]))
        });
        assert!(matches!(result, Err(crate::api::RefreshApplicationError::OutputMismatch)));
        assert!(!extra.finished);

        let mut empty = CallbackRefresh { finished: false };
        let result =
            crate::api::apply_refresh(&declaration, &mut empty, |_, _| Ok(BTreeMap::new()));
        assert!(matches!(result, Err(crate::api::RefreshApplicationError::OutputMismatch)));
        assert!(!empty.finished);
    }

    fn test_declaration() -> (ExponentLutProgram, WireId, RefreshDeclaration) {
        let topology = RefreshTranscriptTopology {
            crt_slot_count: 1,
            component_columns: 1,
            coefficient_count: 1,
            mask_digit_counts: vec![1],
            fresh_digit_count: 1,
        };
        let labels = RefreshPrfLabelIndex::new([3; 32], &topology).unwrap();
        let prf_labels = labels.prf_labels().unwrap();
        let parameters = crate::noise_refresh::pbc::PbcParameters::custom(3, 2, 2, 2, 32, None);
        let generated = (0..32_u8)
            .find_map(|seed| {
                crate::noise_refresh::pbc::generate_key_layout(
                    &parameters,
                    crate::noise_refresh::pbc::PbcRootSeed([seed; 32]),
                    &[0, 1],
                )
                .ok()
            })
            .expect("toy setup must find a PBC schedule");
        let profile = crate::noise_refresh::prf::SparseLwrPrfProfile::new(
            crate::program::ExpLayout { stride_log2: 1, width_log2: 5 },
            1,
        )
        .unwrap();
        let sparse = crate::noise_refresh::prf::SparseLwrPrfProgram::new_structural(
            profile,
            generated.public_layout.bucket_width,
            generated.public_layout.parameters.bucket_count,
            b"refresh-test",
            &prf_labels,
        )
        .unwrap();
        let prepared =
            PreparedSparseLwrPrf::new(sparse, &generated.public_layout, [9; 32]).unwrap();
        let (source, source_output) = source_program();
        let declaration = RefreshDeclaration::new(
            &source,
            source_output,
            vec![WireType::Int],
            vec![RefreshSlot { slot: 0, q: 20.into(), p: 10.into(), q_t: 2.into() }],
            RefreshDigitPlan { base: 2, mask_counts: vec![1], fresh_count: 1 },
            labels,
            prepared,
            0,
            NoiseMomentPolicy::Continuous,
        )
        .unwrap();
        (source, source_output, declaration)
    }

    #[test]
    fn refresh_declaration_freezes_exact_prf_outputs() {
        let (source, source_output, declaration) = test_declaration();
        let prf_labels = declaration.prf_labels().to_vec();
        assert_eq!(declaration.prf_labels(), prf_labels.as_slice());
        assert_eq!(declaration.slots()[0].slot, 0);
        assert_eq!(declaration.source_program(), source.id());
        assert_eq!(declaration.source_output(), source_output);
        assert_eq!(declaration.output_schema(), &[WireType::Int]);
        assert_eq!(declaration.moment_policy(), &NoiseMomentPolicy::Continuous);
        assert_ne!(declaration.commitment(), &[0; 32]);

        assert_eq!(
            RefreshDeclaration::new(
                &source,
                source_output,
                vec![WireType::Int],
                declaration.slots().to_vec(),
                RefreshDigitPlan { base: 4, mask_counts: vec![1], fresh_count: 1 },
                declaration.labels().clone(),
                declaration.prepared().clone(),
                declaration.target_level(),
                declaration.moment_policy().clone(),
            )
            .unwrap_err(),
            RefreshError::InvalidLayout
        );

        let duplicate = RefreshDeclaration::new(
            &source,
            source_output,
            vec![WireType::Int],
            declaration.slots().to_vec(),
            declaration.digits().clone(),
            declaration.labels().clone(),
            declaration.prepared().clone(),
            declaration.target_level(),
            declaration.moment_policy().clone(),
        )
        .unwrap();
        assert_eq!(declaration.commitment(), duplicate.commitment());
    }

    #[test]
    fn refresh_declaration_rejects_non_coprime_or_nondividing_moduli() {
        let topology = RefreshTranscriptTopology {
            crt_slot_count: 2,
            component_columns: 1,
            coefficient_count: 1,
            mask_digit_counts: vec![1, 1],
            fresh_digit_count: 1,
        };
        let labels = RefreshPrfLabelIndex::new([5; 32], &topology).unwrap();
        let (source, source_output) = source_program();
        let invalid = validate_modulus_relations(&[
            RefreshSlot { slot: 0, q: 20.into(), p: 10.into(), q_t: 2.into() },
            RefreshSlot { slot: 1, q: 30.into(), p: 10.into(), q_t: 4.into() },
        ]);
        assert_eq!(invalid, Err(RefreshError::InvalidModulusRelation));
        assert_eq!(
            validate_modulus_relations(&[RefreshSlot {
                slot: 0,
                q: 10.into(),
                p: 10.into(),
                q_t: 2.into(),
            }]),
            Err(RefreshError::InvalidModulusRelation)
        );
        let parameters = crate::noise_refresh::pbc::PbcParameters::custom(3, 2, 2, 2, 32, None);
        let generated = (0..32_u8)
            .find_map(|seed| {
                crate::noise_refresh::pbc::generate_key_layout(
                    &parameters,
                    crate::noise_refresh::pbc::PbcRootSeed([seed; 32]),
                    &[0, 1],
                )
                .ok()
            })
            .expect("toy setup must find a PBC schedule");
        let profile = crate::noise_refresh::prf::SparseLwrPrfProfile::new(
            crate::program::ExpLayout { stride_log2: 1, width_log2: 5 },
            1,
        )
        .unwrap();
        let sparse = crate::noise_refresh::prf::SparseLwrPrfProgram::new_structural(
            profile,
            generated.public_layout.bucket_width,
            generated.public_layout.parameters.bucket_count,
            b"refresh-test",
            &labels.prf_labels().unwrap(),
        )
        .unwrap();
        let prepared =
            PreparedSparseLwrPrf::new(sparse, &generated.public_layout, [9; 32]).unwrap();
        assert_eq!(
            RefreshDeclaration::new(
                &source,
                source_output,
                vec![WireType::Int],
                vec![
                    RefreshSlot { slot: 0, q: 20.into(), p: 10.into(), q_t: 2.into() },
                    RefreshSlot { slot: 1, q: 30.into(), p: 10.into(), q_t: 4.into() },
                ],
                RefreshDigitPlan { base: 2, mask_counts: vec![1, 1], fresh_count: 1 },
                labels,
                prepared,
                0,
                NoiseMomentPolicy::Continuous,
            )
            .unwrap_err(),
            RefreshError::InvalidModulusRelation
        );
    }

    #[test]
    fn refresh_declaration_rejects_incomplete_label_topology() {
        let topology = RefreshTranscriptTopology {
            crt_slot_count: 1,
            component_columns: 1,
            coefficient_count: 1,
            mask_digit_counts: vec![1],
            fresh_digit_count: 1,
        };
        let labels = RefreshPrfLabelIndex::new([4; 32], &topology).unwrap();
        let parameters = crate::noise_refresh::pbc::PbcParameters::custom(3, 2, 2, 2, 32, None);
        let generated = (0..32_u8)
            .find_map(|seed| {
                crate::noise_refresh::pbc::generate_key_layout(
                    &parameters,
                    crate::noise_refresh::pbc::PbcRootSeed([seed; 32]),
                    &[0, 1],
                )
                .ok()
            })
            .expect("toy setup must find a PBC schedule");
        let profile = crate::noise_refresh::prf::SparseLwrPrfProfile::new(
            crate::program::ExpLayout { stride_log2: 1, width_log2: 5 },
            1,
        )
        .unwrap();
        let sparse = crate::noise_refresh::prf::SparseLwrPrfProgram::new_structural(
            profile,
            generated.public_layout.bucket_width,
            generated.public_layout.parameters.bucket_count,
            b"refresh-test",
            &[PrfLabel::from_components(&[b"foreign"]).unwrap()],
        )
        .unwrap();
        let prepared =
            PreparedSparseLwrPrf::new(sparse, &generated.public_layout, [9; 32]).unwrap();
        let (source, source_output) = source_program();
        assert!(
            RefreshDeclaration::new(
                &source,
                source_output,
                vec![WireType::Int],
                vec![RefreshSlot { slot: 0, q: 20.into(), p: 10.into(), q_t: 2.into() }],
                RefreshDigitPlan { base: 2, mask_counts: vec![1], fresh_count: 1 },
                labels,
                prepared,
                0,
                NoiseMomentPolicy::Continuous,
            )
            .is_err()
        );
    }

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefreshDigitPlan {
    pub base: usize,
    pub mask_counts: Vec<usize>,
    pub fresh_count: usize,
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

/// Return the affine offset for positive canonical digits `1..=base`.
///
/// For `count = d`, the unshifted positional sum is the interval
/// `S_d..base*S_d`, where `S_d = (base^d - 1)/(base - 1)`.  Mask digits are
/// centered with `-S_d - base^d/2`; fresh digits retain their native signed
/// representatives with `-(base/2)*S_d`.  The scheme applies any channel
/// factor such as `H` after this common offset; callers must not supply a
/// pre-scaled offset.
pub(crate) fn canonical_digit_offset(
    base: &BigUint,
    count: usize,
    mask: bool,
) -> Result<BigInt, DigitPlanError> {
    if *base < BigUint::from(2u8) || base % 2u8 != BigUint::zero() || count == 0 {
        return Err(DigitPlanError::Parameters);
    }
    let support = (0..count).fold(BigUint::one(), |value, _| value * base);
    let digit_sum = (&support - 1u8) / (base - 1u8);
    let offset = if mask {
        -BigInt::from(digit_sum) - BigInt::from(&support / 2u8)
    } else {
        -BigInt::from(base / 2u8) * BigInt::from(digit_sum)
    };
    Ok(offset)
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
    let digit_sum = (&support - 1u8) / (base - 1u8);
    let offset = canonical_digit_offset(base, count, mask)
        .expect("native digit moment parameters must be a positive even base");
    let lower = offset + BigInt::from(digit_sum);
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

    pub fn topology(
        &self,
        component_columns: usize,
        coefficient_count: usize,
    ) -> Result<super::simulation::RefreshTranscriptTopology, super::simulation::NoiseSimulationError>
    {
        super::simulation::RefreshTranscriptTopology::new(
            component_columns,
            coefficient_count,
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
            canonical_digit_offset(&BigUint::from(plan.base), 2, true).unwrap(),
            (-13).into()
        );
        assert_eq!(
            canonical_digit_offset(&BigUint::from(plan.base), 2, false).unwrap(),
            (-10).into()
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
    fn canonical_offsets_match_exhaustive_positive_digit_moments() {
        fn positive_digit_sums(base: usize, count: usize) -> std::collections::BTreeSet<BigInt> {
            let mut sums = std::collections::BTreeSet::from([BigInt::zero()]);
            let mut weight = BigInt::one();
            for _ in 0..count {
                let current_weight = weight.clone();
                let mut next = std::collections::BTreeSet::new();
                for sum in &sums {
                    for digit in 1..=base {
                        next.insert(sum + &current_weight * BigInt::from(digit));
                    }
                }
                sums = next;
                weight *= base;
            }
            sums
        }

        for base in [2usize, 4, 8] {
            for count in [1usize, 2] {
                let positive = positive_digit_sums(base, count);
                let support = BigUint::from(base).pow(count as u32);
                assert_eq!(BigUint::from(positive.len()), support);
                for mask in [true, false] {
                    let offset = canonical_digit_offset(&BigUint::from(base), count, mask).unwrap();
                    let values = positive
                        .iter()
                        .map(|value| value + &offset)
                        .collect::<std::collections::BTreeSet<_>>();
                    let moments = native_digit_uniform_moments(&BigUint::from(base), count, mask);
                    let cardinality = BigInt::from(values.len());
                    let sum: BigInt = values.iter().cloned().sum();
                    let squares: BigInt = values.iter().map(|value| value * value).sum();
                    assert_eq!(
                        &moments.mean_numerator * &cardinality,
                        &sum * BigInt::from(moments.mean_denominator.clone()),
                    );
                    let centered = squares * &cardinality - &sum * &sum;
                    assert_eq!(
                        BigInt::from(moments.variance.numerator.clone()) *
                            &cardinality *
                            &cardinality,
                        centered * BigInt::from(moments.variance.denominator.clone()),
                    );
                    let first = values.first().unwrap();
                    let last = values.last().unwrap();
                    assert_eq!(last - first + 1u8, cardinality);
                    if base == 2 && count == 1 {
                        assert_eq!(
                            values,
                            if mask {
                                std::collections::BTreeSet::from([BigInt::from(-1), BigInt::zero()])
                            } else {
                                std::collections::BTreeSet::from([BigInt::zero(), BigInt::one()])
                            }
                        );
                    }
                }
            }
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
