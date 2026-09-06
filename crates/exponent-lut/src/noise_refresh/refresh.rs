//! The direct RNS refresh protocol.
//!
//! This is deliberately separate from mxx_bgg::noise_refresh: it operates
//! on scalar BGG encodings and follows the defined order
//! coefficient-switch(scaled state + mask + fresh error) - low-P decoder before CRT
//! recomposition. Sparse-LWR PRF evaluation produces the mask and fresh-error
//! wires consumed here; this module owns only CRT routing, scaling, and
//! aggregation, so it does not duplicate the PRF program.  Quantitative noise
//! correctness is supplied by the application-specific Exponent-LUT noise
//! visitor; this protocol declaration checks only structural equations and
//! identities.
//!
//! For CRT slot `t`, let `q_t` be its plaintext modulus and `mu_t=q/q_t`.
//! Refresh scales an encoding with `u_t = c G^{-1}(mu_t G)` and its public
//! matrix with `A_t = A G^{-1}(mu_t G)`. It forms
//! `A_{sum,t} = A_t + A_{m,t} + A_{e,t}`, asks for `K_t` satisfying
//! `B_P K_t = MS(A_{sum,t}) - (P_t/q_t) reduce(A')`, and applies `d_t = b_P K_t`.
//! The final scalar relation is `s A' - X^w t G + e'`; the implementation
//! below keeps these operations in CRT-slot order and recombines only after
//! each slot has been decoded.

use crate::{
    ExponentLutEncodingCompiler, ExponentLutError,
    noise_refresh::{
        pbc::PbcLayoutId,
        prf::{NativePrfEncodingBatchOutputs, SparseLwrPrfProgram},
    },
};
use mxx_bgg::{BggEncodingWire, BggPublicKeyWire};
use mxx_dsl::{Family, Int, Mat, Preimage, Ring};
use mxx_ir_core::IntExpr;
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

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
    #[error("refresh anchor has no graph-checked B*K equation")]
    /// The anchor lacks the graph-checked `B*K` equation required by refresh.
    MissingAnchorEquation,
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
    #[error(transparent)]
    /// A lower-level BGG operation failed.
    Bgg(#[from] mxx_bgg::EncodingCompileError),
    #[error(transparent)]
    /// A generic Exponent-LUT operation failed.
    ExponentLut(#[from] ExponentLutError),
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
/// `d_e` for the shared fresh-error group. The component count is the BGG
/// public-key column count (`2 * ell_beta`), not the secret dimension.
/// These counts are independent of the BGG gadget digit count carried by
/// [`mxx_bgg::BggSamplerLayout`]. This is public metadata only and contains no
/// selector, support, schedule, or plaintext material.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshPrfLabelIndex {
    refresh_id: [u8; 32],
    mask_slot_count: usize,
    component_count: usize,
    coefficient_count: usize,
    mask_base_p_digit_counts: Vec<usize>,
    mask_offsets: Vec<usize>,
    fresh_error_base_p_digit_count: usize,
}

impl RefreshPrfLabelIndex {
    /// Creates a canonical label index for a refresh instance.
    pub fn new(
        refresh_id: [u8; 32],
        mask_slot_count: usize,
        component_count: usize,
        coefficient_count: usize,
        mask_base_p_digit_counts: Vec<usize>,
        fresh_error_base_p_digit_count: usize,
    ) -> Result<Self, RefreshError> {
        if component_count == 0 ||
            coefficient_count == 0 ||
            mask_base_p_digit_counts.len() != mask_slot_count ||
            mask_base_p_digit_counts.iter().any(|count| *count == 0) ||
            fresh_error_base_p_digit_count == 0
        {
            return Err(RefreshError::InvalidLayout);
        }
        // Validate the complete cardinality up front so index arithmetic is
        // exact and cannot wrap while building structural loop bounds.
        let coordinates =
            component_count.checked_mul(coefficient_count).ok_or(RefreshError::InvalidLayout)?;
        let mut mask_offsets = vec![0usize];
        for digits in &mask_base_p_digit_counts {
            let end = coordinates
                .checked_mul(*digits)
                .and_then(|count| mask_offsets.last().unwrap().checked_add(count))
                .ok_or(RefreshError::InvalidLayout)?;
            mask_offsets.push(end);
        }
        let fresh_group_size = component_count
            .checked_mul(coefficient_count)
            .and_then(|value| value.checked_mul(fresh_error_base_p_digit_count))
            .ok_or(RefreshError::InvalidLayout)?;
        mask_offsets
            .last()
            .unwrap()
            .checked_add(fresh_group_size)
            .ok_or(RefreshError::InvalidLayout)?;
        Ok(Self {
            refresh_id,
            mask_slot_count,
            component_count,
            coefficient_count,
            mask_base_p_digit_counts,
            mask_offsets,
            fresh_error_base_p_digit_count,
        })
    }

    /// Returns the total number of mask and fresh-error labels.
    pub fn len(&self) -> usize {
        self.mask_offsets[self.mask_slot_count] + self.fresh_group_size()
    }

    /// Returns whether the index contains no labels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the canonical label at `index`.
    pub fn label(&self, index: usize) -> Option<RefreshPrfLabel> {
        if index >= self.len() {
            return None;
        }
        let mask_total = self.mask_offsets[self.mask_slot_count];
        if index < mask_total {
            let group = self.mask_offsets.partition_point(|offset| *offset <= index) - 1;
            let within = index - self.mask_offsets[group];
            let digits = self.mask_base_p_digit_counts[group];
            let digit = within % digits;
            let coefficient = (within / digits) % self.coefficient_count;
            let component = within / (digits * self.coefficient_count);
            Some(RefreshPrfLabel::Mask {
                refresh_id: self.refresh_id,
                slot: group,
                component,
                coefficient,
                digit,
            })
        } else {
            let within = index - mask_total;
            let digit = within % self.fresh_error_base_p_digit_count;
            let coefficient =
                (within / self.fresh_error_base_p_digit_count) % self.coefficient_count;
            let component = within / (self.fresh_error_base_p_digit_count * self.coefficient_count);
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
                if refresh_id == self.refresh_id && slot < self.mask_slot_count =>
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
            self.mask_base_p_digit_counts[group_offset]
        } else {
            self.fresh_error_base_p_digit_count
        };
        if component >= self.component_count ||
            coefficient >= self.coefficient_count ||
            digit >= digit_count
        {
            return None;
        }
        let offset =
            component * self.coefficient_count * digit_count + coefficient * digit_count + digit;
        Some(if is_mask {
            self.mask_offsets[group_offset] + offset
        } else {
            self.mask_offsets[self.mask_slot_count] + offset
        })
    }

    fn fresh_group_size(&self) -> usize {
        self.component_count * self.coefficient_count * self.fresh_error_base_p_digit_count
    }
}

/// Immutable sparse-LWR contract expected by one refresh setup.
///
/// The contract is derived from an independently constructed
/// [`SparseLwrPrfProgram`], rather than from the outputs being validated.  It
/// therefore prevents a producer from making its own program identity or
/// terminal form authoritative after the fact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RefreshPrfContract {
    program_id: crate::program::ExponentLutProgramId,
    output_wire: crate::program::WireId,
    native_width: usize,
    q: usize,
    p: usize,
    ring_dimension: usize,
}

impl RefreshPrfContract {
    /// Derives the contract from the expected canonical sparse-LWR program.
    pub(crate) fn from_program(program: &SparseLwrPrfProgram) -> Self {
        let profile = program.profile();
        Self {
            program_id: program.id(),
            output_wire: program.output(),
            native_width: profile.native_width(),
            q: profile.q().expect("validated PRF profile"),
            p: profile.p().expect("validated PRF profile"),
            ring_dimension: profile.ring_dimension().expect("validated PRF profile"),
        }
    }

    pub(crate) const fn from_parts(
        program_id: crate::program::ExponentLutProgramId,
        output_wire: crate::program::WireId,
        native_width: usize,
        q: usize,
        p: usize,
        ring_dimension: usize,
    ) -> Self {
        Self { program_id, output_wire, native_width, q, p, ring_dimension }
    }

    /// Validates the contract against the refresh setup dimensions.
    pub(crate) fn validate_for(
        &self,
        parameters: &crate::noise_refresh::setup::RefreshSetupParameters,
    ) -> Result<(), RefreshError> {
        let ring_dimension = parameters
            .layout
            .ring_dimension
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?
            .to_usize()
            .ok_or(RefreshError::InvalidLayout)?;
        if self.native_width == 0 ||
            self.q == 0 ||
            self.p < 2 ||
            self.p != parameters.base_p ||
            self.ring_dimension != ring_dimension
        {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(())
    }

    /// Checks that one output descriptor is exactly the contracted terminal.
    pub(crate) const fn program_id(&self) -> crate::program::ExponentLutProgramId {
        self.program_id
    }

    pub(crate) const fn q(&self) -> usize {
        self.q
    }

    pub(crate) const fn p(&self) -> usize {
        self.p
    }

    pub(crate) const fn native_width(&self) -> usize {
        self.native_width
    }

    pub(crate) const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }

    pub(crate) const fn output_wire(&self) -> crate::program::WireId {
        self.output_wire
    }
}

/// Exact finite base-`p` label coverage required for one refresh PRF aggregate.
///
/// This type is intentionally opaque and carries no schedule, selector, or
/// artifact information. It is created only by the crate's PRF binder.
#[allow(dead_code)]
#[derive(Clone, Eq, PartialEq)]
pub struct RefreshPrfCoverage {
    refresh_id: [u8; 32],
    component_count: usize,
    coefficient_count: usize,
    mask_base_p_digit_counts: Vec<usize>,
    fresh_error_base_p_digit_count: usize,
}

#[allow(dead_code)]
impl RefreshPrfCoverage {
    pub(crate) fn new(
        refresh_id: [u8; 32],
        component_count: usize,
        coefficient_count: usize,
        mask_base_p_digit_counts: Vec<usize>,
        fresh_error_base_p_digit_count: usize,
    ) -> Result<Self, RefreshError> {
        if component_count == 0 ||
            coefficient_count == 0 ||
            mask_base_p_digit_counts.iter().any(|count| *count == 0) ||
            fresh_error_base_p_digit_count == 0
        {
            return Err(RefreshError::InvalidLayout);
        }
        Ok(Self {
            refresh_id,
            component_count,
            coefficient_count,
            mask_base_p_digit_counts,
            fresh_error_base_p_digit_count,
        })
    }
}

/// Sealed family-level PRF provenance and coverage.
///
/// The vector and public-key families are kept in the exact canonical order
/// committed by [`RefreshPrfLabelIndex`]. Only fixed-size mask/fresh slices
/// are gathered from this value; the production path never materializes a
/// per-label wrapper.
#[derive(Clone)]
pub(crate) struct RefreshPrfFamilyMaterial {
    coverage: RefreshPrfCoverage,
    contract: RefreshPrfContract,
    layout_id: PbcLayoutId,
    vectors: Family<Mat>,
    slot_count: usize,
    family_identity: [u8; 32],
    family_pair_id: [u8; 32],
}

impl RefreshPrfFamilyMaterial {
    pub(crate) fn from_native_encoding_batch_outputs(
        output: &NativePrfEncodingBatchOutputs,
        index: &RefreshPrfLabelIndex,
        coverage: RefreshPrfCoverage,
        contract: RefreshPrfContract,
        layout_id: PbcLayoutId,
    ) -> Result<Self, RefreshError> {
        if coverage.refresh_id != index.refresh_id ||
            coverage.component_count != index.component_count ||
            coverage.coefficient_count != index.coefficient_count ||
            coverage.mask_base_p_digit_counts != index.mask_base_p_digit_counts ||
            coverage.fresh_error_base_p_digit_count != index.fresh_error_base_p_digit_count ||
            output.label_count() != index.len() ||
            !family_has_count(output.vectors(), index.len())?
        {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(Self {
            coverage,
            contract,
            layout_id,
            vectors: output.vectors().clone(),
            slot_count: index.mask_slot_count,
            family_identity: contract.program_id().0,
            family_pair_id: layout_id.0,
        })
    }

    pub(crate) const fn family_identity(&self) -> [u8; 32] {
        self.family_identity
    }

    pub(crate) const fn family_pair_id(&self) -> [u8; 32] {
        self.family_pair_id
    }

    fn mask_family(&self) -> Result<Family<Mat>, RefreshError> {
        let digits =
            self.coverage.mask_base_p_digit_counts.iter().try_fold(0usize, |sum, count| {
                sum.checked_add(*count).ok_or(RefreshError::InvalidLayout)
            })?;
        let count = self
            .coverage
            .component_count
            .checked_mul(self.coverage.coefficient_count)
            .and_then(|value| value.checked_mul(digits))
            .ok_or(RefreshError::InvalidLayout)?;
        let indices = mxx_dsl::parallel(count, |index| Ok(index.clone()))
            .map_err(|_| RefreshError::InvalidLayout)?;
        let vectors = {
            let source_family = self.vectors.clone();
            let gather_indices = indices.clone();
            mxx_dsl::parallel(gather_indices.count().clone(), |family_index| {
                Ok(source_family.at(gather_indices.at(&family_index)))
            })
        }
        .map_err(|_| RefreshError::InvalidLayout)?;
        Ok(vectors)
    }

    pub(crate) fn fresh_error(&self) -> Result<Family<Mat>, RefreshError> {
        let digits =
            self.coverage.mask_base_p_digit_counts.iter().try_fold(0usize, |sum, count| {
                sum.checked_add(*count).ok_or(RefreshError::InvalidLayout)
            })?;
        let start = self
            .coverage
            .component_count
            .checked_mul(self.coverage.coefficient_count)
            .and_then(|value| value.checked_mul(digits))
            .ok_or(RefreshError::InvalidLayout)?;
        let fresh_group = self
            .coverage
            .component_count
            .checked_mul(self.coverage.coefficient_count)
            .and_then(|value| value.checked_mul(self.coverage.fresh_error_base_p_digit_count))
            .ok_or(RefreshError::InvalidLayout)?;
        let indices = mxx_dsl::parallel(fresh_group, |index| {
            Ok(mxx_dsl::Int::constant(start).add(index.clone()))
        })
        .map_err(|_| RefreshError::InvalidLayout)?;
        let vectors = {
            let source_family = self.vectors.clone();
            let gather_indices = indices.clone();
            mxx_dsl::parallel(gather_indices.count().clone(), |family_index| {
                Ok(source_family.at(gather_indices.at(&family_index)))
            })
        }
        .map_err(|_| RefreshError::InvalidLayout)?;
        Ok(vectors)
    }

    fn coverage_slot_count(&self) -> usize {
        // The family-level value is created with the complete label index. The
        // slot count is carried by the family cardinality only at this layer;
        // callers pass the authoritative count through `set_slot_count`.
        self.slot_count
    }
}

fn family_has_count(family: &Family<Mat>, expected: usize) -> Result<bool, RefreshError> {
    Ok(family
        .count()
        .evaluate(&Default::default())
        .map_err(|_| RefreshError::InvalidLayout)?
        .to_usize() ==
        Some(expected))
}

/// Opaque, completely covered per-slot mask material.
#[allow(dead_code)]
#[derive(Clone)]
pub struct RefreshMaskMaterial {
    coverage: RefreshPrfCoverage,
    slot: usize,
    contract: RefreshPrfContract,
    layout_id: PbcLayoutId,
    family: Option<RefreshPrfFamilyMaterial>,
}

#[allow(dead_code)]
impl RefreshMaskMaterial {
    pub(crate) fn from_family(
        coverage: RefreshPrfCoverage,
        slot: usize,
        family: RefreshPrfFamilyMaterial,
    ) -> Result<Self, RefreshError> {
        if family.coverage != coverage || family.slot_count <= slot {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(Self {
            coverage,
            slot,
            contract: family.contract,
            layout_id: family.layout_id,
            family: Some(family),
        })
    }

    pub(crate) fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }

    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    pub(crate) fn family_material(&self) -> Option<&RefreshPrfFamilyMaterial> {
        self.family.as_ref()
    }

    pub(crate) fn program_id(&self) -> crate::program::ExponentLutProgramId {
        self.contract.program_id()
    }

    pub(crate) const fn contract(&self) -> RefreshPrfContract {
        self.contract
    }

    pub(crate) fn coverage_matches(&self, coverage: &RefreshPrfCoverage) -> bool {
        self.coverage == *coverage
    }

    pub(crate) fn validate(&self) -> Result<(), RefreshError> {
        if let Some(family) = &self.family {
            if family.coverage != self.coverage ||
                family.contract != self.contract ||
                family.layout_id != self.layout_id ||
                family.family_pair_id() == [0; 32]
            {
                return Err(RefreshError::PrfOutputMismatch);
            }
            return Ok(());
        }
        Ok(())
    }
}

/// Opaque, completely covered fresh-error material shared by all CRT slots.
#[allow(dead_code)]
#[derive(Clone)]
pub struct RefreshFreshErrorMaterial {
    coverage: RefreshPrfCoverage,
    contract: RefreshPrfContract,
    layout_id: PbcLayoutId,
    family: Option<RefreshPrfFamilyMaterial>,
}

#[allow(dead_code)]
impl RefreshFreshErrorMaterial {
    pub(crate) fn family_material(&self) -> Result<RefreshPrfFamilyMaterial, RefreshError> {
        self.family.clone().ok_or(RefreshError::PrfOutputMismatch)
    }

    pub(crate) fn family_identity(&self) -> Result<[u8; 32], RefreshError> {
        self.family
            .as_ref()
            .map(RefreshPrfFamilyMaterial::family_identity)
            .ok_or(RefreshError::PrfOutputMismatch)
    }

    pub(crate) fn from_family(
        coverage: RefreshPrfCoverage,
        family: RefreshPrfFamilyMaterial,
    ) -> Result<Self, RefreshError> {
        if family.coverage != coverage {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(Self {
            coverage,
            contract: family.contract,
            layout_id: family.layout_id,
            family: Some(family),
        })
    }

    pub(crate) fn program_id(&self) -> crate::program::ExponentLutProgramId {
        self.contract.program_id()
    }

    pub(crate) const fn contract(&self) -> RefreshPrfContract {
        self.contract
    }

    pub(crate) fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }

    pub(crate) fn coverage_matches(&self, coverage: &RefreshPrfCoverage) -> bool {
        self.coverage == *coverage
    }

    pub(crate) fn validate(&self) -> Result<(), RefreshError> {
        if let Some(family) = &self.family {
            if family.coverage != self.coverage ||
                family.contract != self.contract ||
                family.layout_id != self.layout_id
            {
                return Err(RefreshError::PrfOutputMismatch);
            }
            return Ok(());
        }
        Ok(())
    }
}

/// Routes the shared raw fresh-error family once for all CRT slots, applying
/// each slot's exact `kappa_t` inside the same symbolic route body. The output
/// is reduced to one scaled fresh encoding per slot.
pub(crate) fn aggregate_refresh_fresh_error_per_slot(
    base_p: usize,
    material: &RefreshFreshErrorMaterial,
    scales: Vec<Mat>,
    route_preimages: Family<Preimage>,
) -> Result<Vec<BggEncodingWire>, RefreshError> {
    let family = material.family.as_ref().ok_or(RefreshError::PrfOutputMismatch)?;
    let vectors = family.fresh_error()?;
    let component_count = family.coverage.component_count;
    let coefficient_count = family.coverage.coefficient_count;
    let digit_count = family.coverage.fresh_error_base_p_digit_count;
    let group = component_count
        .checked_mul(coefficient_count)
        .and_then(|value| value.checked_mul(digit_count))
        .ok_or(RefreshError::InvalidLayout)?;
    let slot_count = scales.len();
    if slot_count == 0 || !family_has_count(&vectors, group)? {
        return Err(RefreshError::PrfOutputMismatch);
    }
    let total = group.checked_mul(slot_count).ok_or(RefreshError::InvalidLayout)?;
    let raw_indices = mxx_dsl::parallel(total, |index| {
        Ok({
            let flat = index.clone();
            let quotient = flat.clone().div(Int::constant(group));
            flat.sub(quotient.mul(Int::constant(group)))
        })
    })
    .map_err(|_| RefreshError::InvalidLayout)?;
    let repeated_vectors = {
        let source_family = vectors;
        let gather_indices = raw_indices;
        mxx_dsl::parallel(gather_indices.count().clone(), |family_index| {
            Ok(source_family.at(gather_indices.at(&family_index)))
        })
    }
    .map_err(|_| RefreshError::InvalidLayout)?;
    let scales_for_offset = scales.clone();
    let routed_vectors = route_prf_family(repeated_vectors, route_preimages)?;
    let vectors = reduce_family_segments(routed_vectors, slot_count, group)?;
    let columns = vectors
        .element_type()
        .columns
        .evaluate(&Default::default())
        .map_err(|_| RefreshError::InvalidLayout)?
        .to_usize()
        .ok_or(RefreshError::InvalidLayout)?;
    let ring = Ring::new(
        vectors.element_type().modulus.clone(),
        vectors.element_type().ring_dimension.clone(),
    );
    let offset = public_half_offset_vector(&ring, base_p, digit_count, columns, false)?;
    (0..slot_count)
        .map(|slot| {
            let scale = scales_for_offset.get(slot).ok_or(RefreshError::InvalidLayout)?;
            Ok(BggEncodingWire {
                vector: vectors.at(slot) + scale.clone() * offset.clone(),
                plaintext: None,
            })
        })
        .collect()
}

/// Routes a complete canonical family with one symbolic body. Callers may
/// then perform one or more structural balanced reductions over its output;
/// no label is projected to a host value during this phase.
fn route_prf_family(
    vectors: Family<Mat>,
    route_preimages: Family<Preimage>,
) -> Result<Family<Mat>, RefreshError> {
    {
        let source_families = (vectors, route_preimages);
        if source_families.0.count() != source_families.1.count() {
            Err(mxx_dsl::DslError::Schema)
        } else {
            mxx_dsl::parallel(source_families.0.count().clone(), |family_index| {
                let (vector, rhs) =
                    (source_families.0.at(&family_index), source_families.1.at(&family_index));
                let _ = family_index;
                Ok(rhs.mul_small_rhs(vector))
            })
        }
    }
    .map_err(|_| RefreshError::InvalidLayout)
}

fn reduce_family_segments(
    family: Family<Mat>,
    segment_count: usize,
    segment_size: usize,
) -> Result<Family<Mat>, RefreshError> {
    if segment_count == 0 || segment_size == 0 {
        return Err(RefreshError::InvalidLayout);
    }
    crate::encoding::reduce_encoding_family_segments(family, segment_count, segment_size)
        .map_err(|_| RefreshError::InvalidLayout)
}

/// Returns the public vector added after centered native PRF routing.
///
/// Value readout returns `g = d - 1/2` for each base-`P` digit.  The routed
/// digit therefore needs the public offset
/// `1/2 * sum_c P^c * (1 + X + ... + X^(n-1))` in every public-key column.
/// This is a public ring expression and contributes no operational noise.
pub(crate) fn public_half_offset_vector(
    ring: &Ring,
    base_p: usize,
    digit_count: usize,
    columns: usize,
    normalize_mask: bool,
) -> Result<Mat, RefreshError> {
    if base_p < 2 || digit_count == 0 || columns == 0 {
        return Err(RefreshError::InvalidLayout);
    }
    let ring_type = ring.matrix_type((1, 1));
    let modulus =
        ring_type.modulus.evaluate(&Default::default()).map_err(|_| RefreshError::InvalidLayout)?;
    if modulus <= BigInt::from(1) || (&modulus & BigInt::from(1)) == BigInt::from(0) {
        return Err(RefreshError::InvalidLayout);
    }
    let ring_dimension = ring_type
        .ring_dimension
        .evaluate(&Default::default())
        .map_err(|_| RefreshError::InvalidLayout)?
        .to_usize()
        .ok_or(RefreshError::InvalidLayout)?;
    if ring_dimension == 0 {
        return Err(RefreshError::InvalidLayout);
    }
    let inverse_two = (&modulus + BigInt::from(1)) / BigInt::from(2);
    let mut powers = BigInt::from(1);
    let mut digit_sum = BigInt::from(0);
    let base = BigInt::from(base_p);
    for _ in 0..digit_count {
        digit_sum += &powers;
        powers *= &base;
    }
    // Native digits d=g+1/2 span [1-p/2, p/2]. Masks shift this
    // consecutive interval to exactly [-floor(p^digits/2), ..., ceil(p^digits/2)-1].
    let scalar = if normalize_mask {
        let minimum_digit = BigInt::from(1) - BigInt::from(base_p / 2);
        digit_sum * (inverse_two - minimum_digit) - powers / BigInt::from(2)
    } else {
        digit_sum * inverse_two
    };
    let all_ones = ring.polynomial((0..ring_dimension).map(|_| IntExpr::constant(1)));
    let offset = ring.polynomial([IntExpr::constant(scalar)]) * all_ones;
    Ok(Mat::concat(mxx_dsl::ConcatAxis::Columns, (0..columns).map(|_| offset.clone()).collect()))
}

/// Routes all mask slots once and reduces each canonical slot segment to one
/// encoding. Only the resulting fixed-size CRT-slot vector is materialized.
pub(crate) fn aggregate_refresh_masks(
    base_p: usize,
    family: &RefreshPrfFamilyMaterial,
    slot_count: usize,
    route_preimages: Family<Preimage>,
    lifts: Vec<Mat>,
) -> Result<Vec<BggEncodingWire>, RefreshError> {
    if slot_count != family.coverage_slot_count() || lifts.len() != slot_count {
        return Err(RefreshError::SlotOrderMismatch);
    }
    let mask_vectors = family.mask_family()?;
    let routed_vectors = route_prf_family(mask_vectors, route_preimages)?;
    let columns = routed_vectors
        .element_type()
        .columns
        .evaluate(&Default::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(RefreshError::InvalidLayout)?;
    let ring = Ring::new(
        routed_vectors.element_type().modulus.clone(),
        routed_vectors.element_type().ring_dimension.clone(),
    );
    let coordinates = family
        .coverage
        .component_count
        .checked_mul(family.coverage.coefficient_count)
        .ok_or(RefreshError::InvalidLayout)?;
    let mut start = 0usize;
    family
        .coverage
        .mask_base_p_digit_counts
        .iter()
        .enumerate()
        .map(|(slot, digits)| {
            let count = coordinates.checked_mul(*digits).ok_or(RefreshError::InvalidLayout)?;
            let indices = mxx_dsl::parallel(count, |index| {
                Ok(mxx_dsl::Int::constant(start).add(index.clone()))
            })
            .map_err(|_| RefreshError::InvalidLayout)?;
            let slice = {
                let source_family = routed_vectors.clone();
                let gather_indices = indices;
                mxx_dsl::parallel(gather_indices.count().clone(), |family_index| {
                    Ok(source_family.at(gather_indices.at(&family_index)))
                })
            }
            .map_err(|_| RefreshError::InvalidLayout)?;
            let vector = reduce_family_segments(slice, 1, count)?.at(0);
            let offset = public_half_offset_vector(&ring, base_p, *digits, columns, true)?;
            start = start.checked_add(count).ok_or(RefreshError::InvalidLayout)?;
            Ok(BggEncodingWire { vector: vector + lifts[slot].clone() * offset, plaintext: None })
        })
        .collect()
}

fn symbolic_prf_route_matrix(
    ring: &Ring,
    public_key: &Mat,
    base_p: usize,
    digit: IntExpr,
    coefficient: IntExpr,
    component: IntExpr,
) -> Result<Mat, RefreshError> {
    let public_key_type = public_key.matrix_type();
    if public_key_type.rows.evaluate(&Default::default()).ok().and_then(|value| value.to_usize()) !=
        Some(2)
    {
        return Err(RefreshError::InvalidLayout);
    }
    let columns = public_key_type.columns.clone();
    let scalar = ring.constant(
        (1, 1),
        mxx_ir_core::node::ConstantMatrix::PowerOfBase {
            base: IntExpr::constant(base_p),
            exponent: digit,
        },
    ) * ring
        .constant((1, 1), mxx_ir_core::node::ConstantMatrix::Rotation { exponent: coefficient });
    let secret_dimension = public_key_type.rows.clone();
    Ok(-scalar *
        ring.constant(
            (secret_dimension, 1),
            mxx_ir_core::node::ConstantMatrix::UnitColumn { index: IntExpr::constant(1) },
        ) *
        ring.constant(
            (1, columns),
            mxx_ir_core::node::ConstantMatrix::UnitRow { index: component },
        ))
}

/// Builds the deterministic decomposition family for a complete PRF route.
/// The route target depends only on the public plaintext digit base `P`,
/// mixed-radix coordinates, and the optional CRT scale; it is independent of
/// the online `s`-dependent carrier. Consequently the family is safe to
/// persist as public typed preimages and can be consumed online directly.
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

pub(crate) fn route_preimage_family(
    ring: &Ring,
    public_key: &Mat,
    base_p: usize,
    component_count: usize,
    coefficient_count: usize,
    digit_counts: &[usize],
    count: usize,
    scales: Option<Family<Mat>>,
    corrected_scalars: Option<Family<Mat>>,
    decomposition_base: IntExpr,
    decomposition_digit_count: IntExpr,
) -> Result<Family<Preimage>, RefreshError> {
    if count == 0 || component_count == 0 || coefficient_count == 0 {
        return Err(RefreshError::InvalidLayout);
    }
    let total_digits = digit_counts.iter().try_fold(0usize, |sum, digits| {
        sum.checked_add(*digits).ok_or(RefreshError::InvalidLayout)
    })?;
    if component_count
        .checked_mul(coefficient_count)
        .and_then(|coordinates| coordinates.checked_mul(total_digits)) !=
        Some(count)
    {
        return Err(RefreshError::InvalidLayout);
    }
    // Route targets are constructed from the public layout only.  The online
    // carrier is deliberately absent. The supplied setup public key provides
    // the two-row BGG+ target shape needed by `u_2`; only its public type is
    // used, and its value is not copied into the route target.
    let targets = {
        let captured_values = Vec::<Mat>::new();
        mxx_dsl::parallel(count, |index| {
            let _ = captured_values;
            {
                let (_, digit, coefficient, component) = route_coordinates(
                    index.expression()?,
                    component_count,
                    coefficient_count,
                    digit_counts,
                )
                .map_err(|_| mxx_dsl::DslError::Schema)?;
                symbolic_prf_route_matrix(
                    ring,
                    public_key,
                    if corrected_scalars.is_some() { 1 } else { base_p },
                    digit,
                    coefficient,
                    component,
                )
                .map_err(|_| mxx_dsl::DslError::Schema)
            }
        })
    }
    .map_err(|_| RefreshError::InvalidLayout)?;
    let targets = if let Some(scales) = corrected_scalars.or(scales) {
        {
            let source_families = (targets, scales);
            if source_families.0.count() != source_families.1.count() {
                Err(mxx_dsl::DslError::Schema)
            } else {
                mxx_dsl::parallel(source_families.0.count().clone(), |family_index| {
                    let (target, scale) =
                        (source_families.0.at(&family_index), source_families.1.at(&family_index));
                    let _ = family_index;
                    Ok(scale * target)
                })
            }
        }
        .map_err(|_| RefreshError::InvalidLayout)?
    } else {
        targets
    };
    {
        let source_family = targets;
        mxx_dsl::parallel(source_family.count().clone(), |family_index| {
            let target = source_family.at(&family_index);
            let _ = family_index;
            Ok(target.decompose(decomposition_base.clone(), decomposition_digit_count.clone()))
        })
    }
    .map_err(|_| RefreshError::InvalidLayout)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Protocol declaration used to validate an RNS refresh graph.
pub struct RefreshDeclaration {
    /// Stable declaration identity.
    pub identity: [u8; 32],
    /// Identity of the setup that produced this declaration.
    pub setup_identity: [u8; 32],
    /// Owning protocol/component name.
    pub owner: String,
    /// Named refresh plan.
    pub plan: String,
    /// Declared value type.
    pub value_type: String,
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub columns: usize,
    /// CRT slots in canonical order.
    pub slots: Vec<RefreshSlot>,
    /// CRT recomposition coefficients in slot order.
    pub reconstruction_coefficients: Vec<BigInt>,
}

#[derive(Clone)]
/// Output of refresh, retaining the declaration beside the resulting wire.
pub struct RefreshResult {
    encoding: BggEncodingWire,
    declaration: RefreshDeclaration,
}

impl RefreshResult {
    /// Borrows the refreshed BGG encoding wire.
    pub fn encoding(&self) -> &BggEncodingWire {
        &self.encoding
    }
    /// Borrows the declaration checked for this result.
    pub fn declaration(&self) -> &RefreshDeclaration {
        &self.declaration
    }
    /// Extracts the refreshed encoding and discards the declaration.
    pub fn into_encoding(self) -> BggEncodingWire {
        self.encoding
    }
}

impl RefreshDeclaration {
    fn validate_parameters(&self) -> Result<(), RefreshError> {
        if self.owner != "mxx-exponent-lut" ||
            self.setup_identity == [0; 32] ||
            self.plan != "section-7-refresh" ||
            self.value_type != "BggEncodingVector" ||
            self.rows == 0 ||
            self.columns == 0 ||
            self.slots.is_empty() ||
            self.reconstruction_coefficients.len() != self.slots.len()
        {
            return Err(RefreshError::InvalidLayout);
        }
        for (expected, slot) in self.slots.iter().enumerate() {
            if slot.slot != expected ||
                slot.q <= 0.into() ||
                slot.q_t <= 0.into() ||
                slot.p <= 1.into() ||
                &slot.p % 2 != 1.into() ||
                &slot.q % &slot.p != 0.into() ||
                &slot.p % &slot.q_t != 0.into()
            {
                return Err(RefreshError::InvalidLayout);
            }
        }
        Ok(())
    }
}

/// The graph-checked `B*K` equation used to validate decoder preimages.
///
/// The anchor stores only the graph expression needed for identity checks.
/// It does not expose a secret key or a decoder result; callers can bind a
/// preimage only to the exact product constructed here.
#[derive(Clone)]
struct RefreshAnchor {
    equation: Option<RefreshAnchorEquation>,
}

#[derive(Clone)]
struct RefreshAnchorEquation {
    /// The graph expression `B * K` that decoder preimages must reference.
    target: Mat,
}

impl RefreshAnchor {
    /// Builds an anchor from the issuer's decoder matrix `B` and preimage
    /// typed preimage `K`, retaining the exact graph product `B*K` as target.
    pub(crate) fn with_equation(b: Mat, k: Preimage) -> Self {
        // Construct the issuer target here so callers cannot attach an
        // unrelated matrix while claiming it is B*K.
        let target = k.mul_small_rhs(b);
        Self { equation: Some(RefreshAnchorEquation { target }) }
    }
}

/// One scalar decoder preimage bound to an anchor equation and target matrix.
///
/// The stored value handles tie the decoder wire to both the exact `B*K`
/// anchor and the exact combined public target. This prevents setup code from
/// substituting a decoder from another slot or another refresh instance.
#[derive(Clone)]
struct RefreshDecoderPreimage {
    /// Decoder encoding supplied by setup binding.
    encoding: BggEncodingWire,
    /// Identity of the anchor's graph expression `B*K`.
    anchor_equation: mxx_ir_core::ValueHandle,
    /// Identity of the combined public target decoded by this preimage.
    target_equation: mxx_ir_core::ValueHandle,
}

impl RefreshDecoderPreimage {
    /// Binds a decoder wire to an existing anchor and combined target.
    ///
    /// Only graph identities are retained here; no private support or
    /// selector information is copied into the decoder package.
    fn bind(
        anchor: &RefreshAnchor,
        encoding: BggEncodingWire,
        combined_target: &Mat,
    ) -> Result<Self, RefreshError> {
        let equation = anchor.equation.as_ref().ok_or(RefreshError::MissingAnchorEquation)?;
        Ok(Self {
            encoding,
            anchor_equation: equation.target.value_handle().clone(),
            target_equation: combined_target.value_handle().clone(),
        })
    }

    /// Checks that this preimage still references the supplied anchor and
    /// target before it is admitted to a scalar refresh package.
    fn validate(&self, anchor: &RefreshAnchor, target: &Mat) -> Result<(), RefreshError> {
        if anchor.equation.as_ref().map(|equation| equation.target.value_handle()) !=
            Some(&self.anchor_equation) ||
            self.target_equation != *target.value_handle()
        {
            return Err(RefreshError::AnchorMismatch);
        }
        Ok(())
    }
}

/// Typed per-slot package for the RNS refresh equation.
///
/// The package keeps the old state, scaled state target, mask, fresh-error
/// source, and decoder preimage together with their graph identities. Mask and
/// fresh-error material remain separate fields so callers cannot accidentally
/// decode before combining them. The online `b`, preprocessing-fixed `k`, and
/// anchor product are linkage evidence, not additional artifact outputs.
#[derive(Clone)]
struct RefreshScalarPackage {
    slot: usize,
    state_handle: mxx_ir_core::ValueHandle,
    a_prime_handle: mxx_ir_core::ValueHandle,
    scale_target: Mat,
    scaled_state: BggEncodingWire,
    mask: BggEncodingWire,
    fresh_error_source: BggEncodingWire,
    decoder_base_handle: mxx_ir_core::ValueHandle,
    decoder: RefreshDecoderPreimage,
    target_public_matrix: Mat,
    b: Mat,
    k: Preimage,
    anchor_product_handle: mxx_ir_core::ValueHandle,
}

impl RefreshScalarPackage {
    /// Validates and assembles one CRT-slot refresh package.
    ///
    /// The constructor checks the decoder's anchor/target linkage and records
    /// the state and fresh-error graph handles used by later validation. It
    /// does not expose the private setup matrices to callers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        slot: usize,
        state: &BggEncodingWire,
        a_prime: &Mat,
        scale_target: Mat,
        scaled_state: BggEncodingWire,
        mask: BggEncodingWire,
        fresh_error_source: BggEncodingWire,
        decoder: RefreshDecoderPreimage,
        decoder_base_handle: mxx_ir_core::ValueHandle,
        target_public_matrix: Mat,
        anchor: &RefreshAnchor,
        b: Mat,
        k: Preimage,
    ) -> Result<Self, RefreshError> {
        decoder.validate(anchor, &target_public_matrix)?;
        let low = target_public_matrix.matrix_type();
        let high = state.vector.matrix_type();
        if !same_matrix_type(
            b.matrix_type(),
            &mxx_ir_core::types::MatrixType {
                modulus: low.modulus.clone(),
                ring_dimension: low.ring_dimension.clone(),
                ..b.matrix_type().clone()
            },
        ) || !same_matrix_type(
            k.matrix_type(),
            &mxx_ir_core::types::MatrixType {
                rows: b.matrix_type().columns.clone(),
                columns: high.columns.clone(),
                ..low.clone()
            },
        ) || !same_matrix_type(
            low,
            &mxx_ir_core::types::MatrixType {
                rows: b.matrix_type().rows.clone(),
                columns: high.columns.clone(),
                ring_dimension: high.ring_dimension.clone(),
                ..low.clone()
            },
        ) || !same_matrix_type(
            decoder.encoding.vector.matrix_type(),
            &mxx_ir_core::types::MatrixType { modulus: low.modulus.clone(), ..high.clone() },
        ) {
            return Err(RefreshError::InvalidLayout);
        }
        let anchor_product_handle = anchor
            .equation
            .as_ref()
            .ok_or(RefreshError::MissingAnchorEquation)?
            .target
            .value_handle()
            .clone();
        Ok(Self {
            slot,
            state_handle: state.vector.value_handle().clone(),
            a_prime_handle: a_prime.value_handle().clone(),
            scale_target,
            scaled_state,
            mask,
            fresh_error_source,
            decoder_base_handle,
            decoder,
            target_public_matrix,
            b,
            k,
            anchor_product_handle,
        })
    }
}

/// Opaque, validated setup for one refresh invocation. It owns the exact input
/// state and refreshed public matrix, so evaluation cannot substitute either
/// after setup binding.
#[derive(Clone)]
pub struct RefreshSetupManifest {
    identity: [u8; 32],
    state: BggEncodingWire,
    a_prime: Mat,
    packages: Vec<RefreshScalarPackage>,
}

impl RefreshSetupManifest {
    fn bind(
        state: BggEncodingWire,
        a_prime: Mat,
        packages: Vec<RefreshScalarPackage>,
    ) -> Result<Self, RefreshError> {
        if packages.is_empty() {
            return Err(RefreshError::InvalidLayout);
        }
        let mut anchors = std::collections::BTreeMap::new();
        for (slot, package) in packages.iter().enumerate() {
            if package.slot != slot ||
                package.state_handle != *state.vector.value_handle() ||
                package.a_prime_handle != *a_prime.value_handle() ||
                !same_matrix_type(package.mask.vector.matrix_type(), state.vector.matrix_type()) ||
                !same_matrix_type(
                    package.fresh_error_source.vector.matrix_type(),
                    state.vector.matrix_type(),
                )
            {
                return Err(if package.slot != slot {
                    RefreshError::SlotOrderMismatch
                } else {
                    RefreshError::InvalidLayout
                });
            }
            let modulus = package.b.matrix_type().modulus.clone();
            let binding = (package.b.value_handle().clone(), package.decoder_base_handle.clone());
            if anchors.get(&modulus).is_some_and(|previous| previous != &binding) {
                return Err(RefreshError::AnchorMismatch);
            }
            anchors.insert(modulus, binding);
        }
        let mut hash = Sha256::new();
        hash.update(b"mxx.exponent-lut.refresh-setup");
        hash.update(format!("{:?}", state.vector.value_handle()).as_bytes());
        hash.update(format!("{:?}", a_prime.value_handle()).as_bytes());
        for package in &packages {
            hash.update(package.slot.to_le_bytes());
            hash.update(format!("{:?}", package.target_public_matrix.value_handle()).as_bytes());
            hash.update(format!("{:?}", package.b.value_handle()).as_bytes());
            hash.update(format!("{:?}", package.decoder_base_handle).as_bytes());
            hash.update(format!("{:?}", package.k.value_handle()).as_bytes());
        }
        Ok(Self { identity: hash.finalize().into(), state, a_prime, packages })
    }

    /// Returns the identity of this validated setup.
    pub fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    /// Returns the exact state encoding sealed into the setup.
    pub fn state(&self) -> &BggEncodingWire {
        &self.state
    }

    /// Returns the public matrix targeted by refresh.
    pub fn refreshed_public_matrix(&self) -> &Mat {
        &self.a_prime
    }
}

/// Configuration for exact q/q_t gadget scaling and CRT recomposition.
#[derive(Clone)]
pub struct RefreshCompiler {
    /// Full ciphertext modulus `q`.
    pub full_modulus: IntExpr,
    /// CRT plaintext moduli `q_t`, in slot order.
    pub crt_plaintext_moduli: Vec<IntExpr>,
    /// Coefficients used to recombine CRT slot outputs.
    pub reconstruction_coefficients: Vec<IntExpr>,
}

impl RefreshCompiler {
    /// Binds setup material imported from the canonical Phase-2 manifest.
    ///
    /// The online shared decoder base and imported per-slot preimages are
    /// deliberately plain wires at this boundary. This helper is the only
    /// crate-internal adapter that turns them into typed per-slot packages:
    /// each target is recomputed from the imported state/mask/fresh material
    /// and the decoder is rebuilt as the shared `base.vector * K_t`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_imported_wires(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        state: BggEncodingWire,
        state_public: BggPublicKeyWire,
        a_prime: Mat,
        public_b: Vec<Mat>,
        refresh_targets: Vec<Mat>,
        scaled_freshes: Vec<BggEncodingWire>,
        masks: Vec<BggEncodingWire>,
        state_scaling: Vec<Preimage>,
        decoder_bases: Vec<BggEncodingWire>,
        preimages: Vec<Preimage>,
    ) -> Result<RefreshSetupManifest, RefreshError> {
        self.validate_layout()?;
        if masks.len() != self.crt_plaintext_moduli.len() ||
            scaled_freshes.len() != masks.len() ||
            state_scaling.len() != masks.len() ||
            preimages.len() != masks.len() ||
            public_b.len() != masks.len() ||
            decoder_bases.len() != masks.len() ||
            refresh_targets.len() != masks.len() ||
            !same_matrix_type(a_prime.matrix_type(), state_public.matrix.matrix_type())
        {
            return Err(RefreshError::InvalidLayout);
        }
        // Slots sharing P use the same B_P and online anchor handles. Distinct
        // rings remain separate, including when their matrix shapes coincide.
        let mut packages = Vec::with_capacity(masks.len());
        for (slot, (((mask, scaled_fresh), scale_decomposition), k)) in masks
            .into_iter()
            .zip(scaled_freshes.into_iter())
            .zip(state_scaling.into_iter())
            .zip(preimages.into_iter())
            .enumerate()
        {
            let base = decoder_bases[slot].clone();
            let decoder_base_handle = base.vector.value_handle().clone();
            // `scale = mu_t = q/q_t`; multiplying by its gadget decomposition
            // implements `u_t=c G^{-1}(mu_t G)` and `A_t=A G^{-1}(mu_t G)`.
            let scale = mxx_dsl::Ring::new(
                state.vector.matrix_type().modulus.clone(),
                state.vector.matrix_type().ring_dimension.clone(),
            )
            .polynomial([self.scale_expression(slot)?]);
            let scaled_state =
                compiler.bgg.large_scalar_mul(&state, &scale, scale_decomposition.clone());
            let target = refresh_targets.get(slot).cloned().ok_or(RefreshError::InvalidLayout)?;
            let p = target
                .matrix_type()
                .modulus
                .evaluate(&Default::default())
                .map_err(|_| RefreshError::InvalidLayout)?;
            let q = self
                .full_modulus
                .evaluate(&Default::default())
                .map_err(|_| RefreshError::InvalidLayout)?;
            let qt = self.crt_plaintext_moduli[slot]
                .evaluate(&Default::default())
                .map_err(|_| RefreshError::InvalidLayout)?;
            if p <= 1.into() || &p % 2 != 1.into() || &q % &p != 0.into() || &p % &qt != 0.into() {
                return Err(RefreshError::InvalidLayout);
            }
            // K_t targets MS(A_sum,t) - Delta_t reduce(A') in R_P.
            let decoder =
                BggEncodingWire { vector: k.clone().mul_small_rhs(base.vector), plaintext: None };
            let anchor = RefreshAnchor::with_equation(public_b[slot].clone(), k.clone());
            let decoder = RefreshDecoderPreimage::bind(&anchor, decoder, &target)?;
            packages.push(RefreshScalarPackage::new(
                slot,
                &state,
                &a_prime,
                scale,
                scaled_state,
                mask,
                scaled_fresh,
                decoder,
                decoder_base_handle,
                target,
                &anchor,
                public_b[slot].clone(),
                k,
            )?);
        }
        RefreshSetupManifest::bind(state, a_prime, packages)
    }

    /// Build the canonical event from the actual refresh output and equation
    /// operands. The event is attached to the output graph value below.
    fn refresh_declaration(
        &self,
        setup: &RefreshSetupManifest,
        output: &Mat,
    ) -> Result<RefreshDeclaration, RefreshError> {
        let state = &setup.state;
        let rows = state
            .vector
            .matrix_type()
            .rows
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?
            .to_usize()
            .ok_or(RefreshError::InvalidLayout)?;
        let columns = state
            .vector
            .matrix_type()
            .columns
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?
            .to_usize()
            .ok_or(RefreshError::InvalidLayout)?;
        let q = self
            .full_modulus
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?;
        let slots = self
            .crt_plaintext_moduli
            .iter()
            .enumerate()
            .map(|(slot, q_t)| {
                Ok(RefreshSlot {
                    slot,
                    q: q.clone(),
                    p: setup.packages[slot]
                        .target_public_matrix
                        .matrix_type()
                        .modulus
                        .evaluate(&Default::default())
                        .map_err(|_| RefreshError::InvalidLayout)?,
                    q_t: q_t
                        .evaluate(&Default::default())
                        .map_err(|_| RefreshError::InvalidLayout)?,
                })
            })
            .collect::<Result<Vec<_>, RefreshError>>()?;
        let declaration = RefreshDeclaration {
            identity: matrix_identity(output),
            setup_identity: setup.identity,
            owner: "mxx-exponent-lut".into(),
            plan: "section-7-refresh".into(),
            value_type: "BggEncodingVector".into(),
            rows,
            columns,
            slots,
            reconstruction_coefficients: self
                .reconstruction_coefficients
                .iter()
                .map(|coefficient| {
                    coefficient
                        .evaluate(&Default::default())
                        .map_err(|_| RefreshError::InvalidLayout)
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        if output.matrix_type().rows.evaluate(&Default::default()).ok().and_then(|v| v.to_usize()) !=
            Some(rows) ||
            output
                .matrix_type()
                .columns
                .evaluate(&Default::default())
                .ok()
                .and_then(|v| v.to_usize()) !=
                Some(columns) ||
            setup.packages.is_empty()
        {
            return Err(RefreshError::InvalidLayout);
        }
        declaration.validate_parameters()?;
        Ok(declaration)
    }
    /// Validates CRT divisibility and coefficient count.
    pub fn validate_layout(&self) -> Result<(), RefreshError> {
        // CRT-backed refresh requires at least two plaintext-modulus slots;
        // a single slot is not a supported CRT layout.
        if self.crt_plaintext_moduli.len() < 2 ||
            self.crt_plaintext_moduli.len() != self.reconstruction_coefficients.len()
        {
            return Err(RefreshError::InvalidLayout);
        }
        let q = self
            .full_modulus
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?;
        if q <= BigInt::from(0) {
            return Err(RefreshError::InvalidLayout);
        }
        let q_t = self
            .crt_plaintext_moduli
            .iter()
            .map(|qt| qt.evaluate(&Default::default()).map_err(|_| RefreshError::InvalidLayout))
            .collect::<Result<Vec<_>, _>>()?;
        if q_t.iter().any(|value| *value <= BigInt::from(1) || &q % value != BigInt::from(0)) {
            return Err(RefreshError::InvalidLayout);
        }
        for (index, left) in q_t.iter().enumerate() {
            for right in q_t.iter().skip(index + 1) {
                if bigint_gcd(left, right) != BigInt::from(1) {
                    return Err(RefreshError::InvalidLayout);
                }
            }
        }
        let product = q_t.iter().fold(BigInt::from(1), |product, value| product * value);
        if product != q {
            return Err(RefreshError::InvalidLayout);
        }
        Ok(())
    }

    /// Returns the exact `q / q_t` scale expression for one CRT slot.
    pub fn scale_expression(&self, slot: usize) -> Result<IntExpr, RefreshError> {
        self.crt_plaintext_moduli
            .get(slot)
            .cloned()
            .map(|q_t| IntExpr::Div(Box::new(self.full_modulus.clone()), Box::new(q_t)))
            .ok_or(RefreshError::InvalidLayout)
    }

    /// Refreshes a state using one typed package for each RNS slot. The
    /// fresh-error identity is checked before any graph is built, and all
    /// masks/errors are added before their combined decoder is subtracted.
    pub fn refresh(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        setup: &RefreshSetupManifest,
    ) -> Result<RefreshResult, RefreshError> {
        self.validate_layout()?;
        let state = &setup.state;
        let packages = &setup.packages;
        if packages.len() != self.crt_plaintext_moduli.len() {
            return Err(RefreshError::InvalidLayout);
        }
        // One homogeneous family per selected ring performs the slot equation.
        // This keeps CRT-slot work independent while preserving the caller's
        // declared slot order for the final recomposition.
        let mut groups = std::collections::BTreeMap::<IntExpr, Vec<usize>>::new();
        for (slot, package) in packages.iter().enumerate() {
            groups
                .entry(package.target_public_matrix.matrix_type().modulus.clone())
                .or_default()
                .push(slot);
        }
        let mut levels_by_slot = std::collections::BTreeMap::new();
        for (modulus, slots) in groups {
            let scaled = Family::pack(
                slots.iter().map(|slot| packages[*slot].scaled_state.vector.clone()).collect(),
            )
            .map_err(|_| RefreshError::InvalidLayout)?;
            let masks = Family::pack(
                slots.iter().map(|slot| packages[*slot].mask.vector.clone()).collect(),
            )
            .map_err(|_| RefreshError::InvalidLayout)?;
            let fresh = Family::pack(
                slots
                    .iter()
                    .map(|slot| packages[*slot].fresh_error_source.vector.clone())
                    .collect(),
            )
            .map_err(|_| RefreshError::InvalidLayout)?;
            let decoders = Family::pack(
                slots.iter().map(|slot| packages[*slot].decoder.encoding.vector.clone()).collect(),
            )
            .map_err(|_| RefreshError::InvalidLayout)?;
            let family = {
                let source_families = vec![scaled, masks, fresh, decoders];
                if source_families.is_empty() ||
                    source_families
                        .iter()
                        .any(|family| family.count() != source_families[0].count())
                {
                    Err(mxx_dsl::DslError::Schema)
                } else {
                    mxx_dsl::parallel(source_families[0].count().clone(), |index| {
                        let mut values = source_families
                            .iter()
                            .map(|family| family.at(&index))
                            .collect::<Vec<_>>();
                        Ok({
                            let decoder = values.pop().expect("low-P decoder");
                            let fresh = BggEncodingWire {
                                vector: values.pop().expect("fresh"),
                                plaintext: None,
                            };
                            let mask = BggEncodingWire {
                                vector: values.pop().expect("mask"),
                                plaintext: None,
                            };
                            let scaled = BggEncodingWire {
                                vector: values.pop().expect("scaled state"),
                                plaintext: None,
                            };
                            let combined =
                                compiler.bgg.add(&scaled, &mask).expect("validated high-q add");
                            let sum =
                                compiler.bgg.add(&combined, &fresh).expect("validated high-q add");
                            // The complete high-q sum is switched exactly once. No high-q
                            // combined decoder is provided to this online graph.
                            sum.vector.modulus_switch(modulus.clone()) - decoder
                        })
                    })
                }
            }
            .map_err(|_| RefreshError::InvalidLayout)?;
            for (local, slot) in slots.into_iter().enumerate() {
                levels_by_slot.insert(slot, family.at(local));
            }
        }
        let mut levels = Vec::with_capacity(packages.len());
        let mut slot_roles = Vec::with_capacity(packages.len() * 8);
        for (slot, package) in packages.iter().enumerate() {
            if package.slot != slot {
                return Err(RefreshError::SlotOrderMismatch);
            }
            if package.state_handle != *state.vector.value_handle() ||
                package.a_prime_handle != *setup.a_prime.value_handle()
            {
                return Err(RefreshError::TargetMismatch);
            }
            if package.decoder.anchor_equation != package.anchor_product_handle ||
                package.decoder.target_equation != *package.target_public_matrix.value_handle()
            {
                return Err(RefreshError::AnchorMismatch);
            }
            let level = levels_by_slot.remove(&slot).ok_or(RefreshError::SlotOrderMismatch)?;
            // These names are retained as a compact linkage manifest. The
            // actual arithmetic lives in the one parallel body above.
            slot_roles.extend([
                (format!("slot-{slot}-scale-target"), package.scale_target.value_handle().clone()),
                (format!("slot-{slot}-scaled-state"), state.vector.value_handle().clone()),
                (format!("slot-{slot}-mask"), package.mask.vector.value_handle().clone()),
                (
                    format!("slot-{slot}-scaled-fresh-error"),
                    package.fresh_error_source.vector.value_handle().clone(),
                ),
                (format!("slot-{slot}-masked"), package.mask.vector.value_handle().clone()),
                (
                    format!("slot-{slot}-combined"),
                    package.fresh_error_source.vector.value_handle().clone(),
                ),
                (
                    format!("slot-{slot}-decoder"),
                    package.decoder.encoding.vector.value_handle().clone(),
                ),
                (
                    format!("slot-{slot}-decoder-public"),
                    package.target_public_matrix.value_handle().clone(),
                ),
                (format!("slot-{slot}-level"), level.value_handle().clone()),
                (
                    format!("slot-{slot}-target"),
                    package.target_public_matrix.value_handle().clone(),
                ),
                (format!("slot-{slot}-b"), package.b.value_handle().clone()),
                (format!("slot-{slot}-k"), package.k.value_handle().clone()),
                (
                    format!("slot-{slot}-b-k-product"),
                    package.decoder.encoding.vector.value_handle().clone(),
                ),
            ]);
            levels.push(level);
        }
        let vector = Mat::crt_recompose(
            levels,
            self.crt_plaintext_moduli.clone(),
            self.reconstruction_coefficients.clone(),
            self.full_modulus.clone(),
        );
        let declaration = self.refresh_declaration(setup, &vector)?;
        Ok(RefreshResult { encoding: BggEncodingWire { vector, plaintext: None }, declaration })
    }
}

fn bigint_gcd(left: &BigInt, right: &BigInt) -> BigInt {
    let mut a = left.clone().abs();
    let mut b = right.clone().abs();
    while !b.is_zero() {
        let remainder = a % &b;
        a = b;
        b = remainder;
    }
    a
}

fn matrix_identity(matrix: &Mat) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(format!("{:?}", matrix.value_handle()).as_bytes());
    h.finalize().into()
}

fn same_matrix_type(
    left: &mxx_ir_core::types::MatrixType,
    right: &mxx_ir_core::types::MatrixType,
) -> bool {
    let environment = mxx_ir_core::ParamEnv::default();
    [
        (&left.modulus, &right.modulus),
        (&left.ring_dimension, &right.ring_dimension),
        (&left.rows, &right.rows),
        (&left.columns, &right.columns),
    ]
    .into_iter()
    .all(|(left, right)| matches!((left.evaluate(&environment), right.evaluate(&environment)), (Ok(left), Ok(right)) if left == right))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prf_labels_cover_all_public_key_columns_with_separate_digit_groups() {
        let index =
            RefreshPrfLabelIndex::new([7; 32], 2, 6, 2, vec![3; 2], 1).expect("label index");
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
        let index = RefreshPrfLabelIndex::new(refresh_id, 3, 2, 3, digits.clone(), 2).unwrap();
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
        assert!(RefreshPrfLabelIndex::new(refresh_id, 2, 2, 3, digits, 2).is_err());
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
                slot_count,
                component_count,
                coefficient_count,
                vec![dm; slot_count],
                de,
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

    #[test]
    fn refresh_layout_rejects_duplicate_or_non_coprime_crt_moduli() {
        let duplicate = RefreshCompiler {
            full_modulus: 12.into(),
            crt_plaintext_moduli: vec![2.into(), 2.into(), 3.into()],
            reconstruction_coefficients: vec![1.into(), 1.into(), 1.into()],
        };
        assert!(matches!(duplicate.validate_layout(), Err(RefreshError::InvalidLayout)));

        let non_coprime = RefreshCompiler {
            full_modulus: 24.into(),
            crt_plaintext_moduli: vec![4.into(), 6.into()],
            reconstruction_coefficients: vec![1.into(), 1.into()],
        };
        assert!(matches!(non_coprime.validate_layout(), Err(RefreshError::InvalidLayout)));

        let length_mismatch = RefreshCompiler {
            full_modulus: 6.into(),
            crt_plaintext_moduli: vec![2.into(), 3.into()],
            reconstruction_coefficients: vec![1.into()],
        };
        assert!(matches!(length_mismatch.validate_layout(), Err(RefreshError::InvalidLayout)));
    }

    #[test]
    fn refresh_layout_rejects_crt_product_mismatch_and_unit_modulus() {
        let mismatch = RefreshCompiler {
            full_modulus: 24.into(),
            crt_plaintext_moduli: vec![3.into(), 5.into()],
            reconstruction_coefficients: vec![1.into(), 1.into()],
        };
        assert!(matches!(mismatch.validate_layout(), Err(RefreshError::InvalidLayout)));

        let unit = RefreshCompiler {
            full_modulus: 6.into(),
            crt_plaintext_moduli: vec![1.into(), 6.into()],
            reconstruction_coefficients: vec![1.into(), 1.into()],
        };
        assert!(matches!(unit.validate_layout(), Err(RefreshError::InvalidLayout)));
    }

    #[test]
    fn refresh_layout_rejects_single_crt_slot() {
        let layout = RefreshCompiler {
            full_modulus: 97.into(),
            crt_plaintext_moduli: vec![97.into()],
            reconstruction_coefficients: vec![1.into()],
        };
        assert!(matches!(layout.validate_layout(), Err(RefreshError::InvalidLayout)));
    }

    #[test]
    fn grouped_low_p_refresh_preserves_slot_order_and_anchor_binding() {
        let ring = Ring::new(105, 4);
        let compiler = ExponentLutEncodingCompiler::with_layout(
            mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: 2.into(),
            },
            crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 },
        );
        for (moduli, plaintext) in [([15, 15, 21], [3, 5, 7]), ([21, 15, 15], [7, 3, 5])] {
            let refresh = RefreshCompiler {
                full_modulus: 105.into(),
                crt_plaintext_moduli: plaintext.into_iter().map(IntExpr::constant).collect(),
                reconstruction_coefficients: vec![1.into(); 3],
            };
            let state = BggEncodingWire { vector: ring.input("state", (1, 4)), plaintext: None };
            let state_public =
                BggPublicKeyWire { matrix: ring.input("A", (2, 4)), reveal_plaintext: false };
            let a_prime = ring.input("A-prime", (2, 4));
            let groups = [15, 21]
                .map(|p| {
                    let low = Ring::new(p, 4);
                    (
                        p,
                        (
                            low.input(format!("B-{p}"), (2, 6)),
                            BggEncodingWire {
                                vector: low.input(format!("b-{p}"), (1, 6)),
                                plaintext: None,
                            },
                        ),
                    )
                })
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            let public_b = moduli.map(|p| groups[&p].0.clone()).to_vec();
            let bases = moduli.map(|p| groups[&p].1.clone()).to_vec();
            let targets = moduli
                .iter()
                .enumerate()
                .map(|(slot, p)| Ring::new(*p, 4).input(format!("target-{slot}"), (2, 4)))
                .collect::<Vec<_>>();
            let preimages = moduli
                .iter()
                .enumerate()
                .map(|(slot, p)| Ring::new(*p, 4).preimage_input(format!("K-{slot}"), (6, 4), 1))
                .collect::<Vec<_>>();
            let masks = (0..3)
                .map(|slot| BggEncodingWire {
                    vector: ring.input(format!("mask-{slot}"), (1, 4)),
                    plaintext: None,
                })
                .collect::<Vec<_>>();
            let fresh = (0..3)
                .map(|slot| BggEncodingWire {
                    vector: ring.input(format!("fresh-{slot}"), (1, 4)),
                    plaintext: None,
                })
                .collect::<Vec<_>>();
            let scaling = (0..3)
                .map(|slot| ring.preimage_input(format!("scale-{slot}"), (4, 4), 1))
                .collect::<Vec<_>>();
            let bind = |bases| {
                refresh.bind_imported_wires(
                    &compiler,
                    state.clone(),
                    state_public.clone(),
                    a_prime.clone(),
                    public_b.clone(),
                    targets.clone(),
                    fresh.clone(),
                    masks.clone(),
                    scaling.clone(),
                    bases,
                    preimages.clone(),
                )
            };
            let setup = bind(bases.clone()).unwrap();
            let result = refresh.refresh(&compiler, &setup).unwrap();
            assert_eq!(result.encoding().vector.matrix_type(), state.vector.matrix_type());
            assert_eq!(
                result.declaration().slots.iter().map(|slot| slot.p.clone()).collect::<Vec<_>>(),
                moduli.map(BigInt::from)
            );
            let graph = mxx_dsl::DslContext::new("grouped-refresh")
                .output("result", result.into_encoding().vector)
                .unwrap()
                .build()
                .unwrap();
            graph.validate(&Default::default()).unwrap();
            let switches = graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .filter(|node| {
                    matches!(node.kind(), mxx_ir_core::node::NodeKind::ModulusSwitch { .. })
                })
                .count();
            assert_eq!(switches, 2, "one symbolic complete-sum switch body per distinct P");
            let shared_slot = moduli.iter().position(|p| *p == 15).unwrap();
            let mut tampered = bases;
            tampered[shared_slot].vector = Ring::new(15, 4).input("substituted-anchor", (1, 6));
            assert!(matches!(bind(tampered), Err(RefreshError::AnchorMismatch)));
        }
    }

    #[test]
    fn public_half_offset_supports_native_p16_and_p32() {
        let ring = Ring::new(97, 8);
        for (base_p, digit_count) in [(16, 2), (32, 1)] {
            let offset = public_half_offset_vector(&ring, base_p, digit_count, 2, false)
                .expect("valid public half offset");
            assert_eq!(
                offset.matrix_type().columns.evaluate(&mxx_ir_core::ParamEnv::default()),
                Ok(BigInt::from(2))
            );
            let graph = mxx_dsl::DslContext::new("public-half-offset")
                .public_output("offset", offset)
                .expect("offset output")
                .build()
                .expect("offset graph");
            graph.validate(&mxx_ir_core::ParamEnv::default()).expect("valid offset graph");
        }
    }

    #[test]
    fn route_preimages_use_the_two_row_public_key_shape() {
        let ring = Ring::new(97, 8);
        let public_key = ring.input("route-public-key", (2, 4));
        let routes = route_preimage_family(
            &ring,
            &public_key,
            4,
            4,
            2,
            &[2],
            16,
            None,
            None,
            IntExpr::constant(2),
            IntExpr::constant(3),
        )
        .expect("valid route family");
        let graph = mxx_dsl::DslContext::new("route-preimages-two-row-target")
            .public_output("routes", routes)
            .expect("route output")
            .build()
            .expect("route graph");
        graph
            .validate(&mxx_ir_core::ParamEnv::default())
            .expect("u_2 must be valid for the two-row public-key target");

        let one_row_public_key = ring.input("invalid-route-public-key", (1, 4));
        assert!(matches!(
            route_preimage_family(
                &ring,
                &one_row_public_key,
                4,
                4,
                2,
                &[2],
                16,
                None,
                None,
                IntExpr::constant(2),
                IntExpr::constant(3),
            ),
            Err(RefreshError::InvalidLayout)
        ));
    }
}
