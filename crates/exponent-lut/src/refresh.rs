//! The direct RNS refresh protocol.
//!
//! This is deliberately separate from mxx_bgg::noise_refresh: it operates
//! on scalar BGG encodings and follows the defined order
//! scaled state + mask + fresh error - combined decoder before CRT
//! recomposition. Sparse-LWR PRF evaluation produces the mask and fresh-error
//! wires consumed here; this module owns only CRT routing, scaling, and
//! aggregation, so it does not duplicate the PRF program.  Noise correctness
//! is intentionally not modeled by this protocol declaration: the generic
//! operational-noise checker analyzes the actual graph, while this module
//! checks only structural equations and identities.
//!
//! For CRT slot `t`, let `q_t` be its plaintext modulus and `mu_t=q/q_t`.
//! Refresh scales an encoding with `u_t = c G^{-1}(mu_t G)` and its public
//! matrix with `A_t = A G^{-1}(mu_t G)`. It forms
//! `A_{sum,t} = A_t + A_{m,t} + A_{e,t}`, asks for `K_t` satisfying
//! `B K_t = A_{sum,t} - mu_t A'`, and sets the decoder target `d_t = b K_t`.
//! The final scalar relation is `s A' - X^w t G + e'`; the implementation
//! below keeps these operations in CRT-slot order and recombines only after
//! each slot has been decoded.

use crate::{
    ExponentLutEncodingCompiler, ExponentLutError,
    pbc::PbcLayoutId,
    prf::{PbcSparseLwrEncodingOutputs, SparseLwrPrfProgram, SparseLwrPrfTerminalForm},
};
use mxx_bgg::{BggEncodingWire, BggPublicKeyWire};
use mxx_dsl::{Family, Int, Mat, Parallel, Preimage};
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
/// One CRT slot's full modulus and plaintext modulus.
pub struct RefreshSlot {
    /// Zero-based CRT slot index.
    pub slot: usize,
    /// Full ciphertext modulus for this slot.
    pub q: BigInt,
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
    mask_base_p_digit_count: usize,
    fresh_error_base_p_digit_count: usize,
}

impl RefreshPrfLabelIndex {
    pub(crate) const fn refresh_id(&self) -> [u8; 32] {
        self.refresh_id
    }
    pub(crate) const fn mask_slot_count(&self) -> usize {
        self.mask_slot_count
    }
    pub(crate) const fn component_count(&self) -> usize {
        self.component_count
    }
    pub(crate) const fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }
    pub(crate) const fn mask_base_p_digit_count(&self) -> usize {
        self.mask_base_p_digit_count
    }
    pub(crate) const fn fresh_error_base_p_digit_count(&self) -> usize {
        self.fresh_error_base_p_digit_count
    }
    /// Creates a canonical label index for a refresh instance.
    pub fn new(
        refresh_id: [u8; 32],
        mask_slot_count: usize,
        component_count: usize,
        coefficient_count: usize,
        mask_base_p_digit_count: usize,
        fresh_error_base_p_digit_count: usize,
    ) -> Result<Self, RefreshError> {
        if component_count == 0 ||
            coefficient_count == 0 ||
            mask_base_p_digit_count == 0 ||
            fresh_error_base_p_digit_count == 0
        {
            return Err(RefreshError::InvalidLayout);
        }
        // Validate the complete cardinality up front so index arithmetic is
        // exact and cannot wrap while building structural loop bounds.
        let mask_group_size = component_count
            .checked_mul(coefficient_count)
            .and_then(|value| value.checked_mul(mask_base_p_digit_count))
            .ok_or(RefreshError::InvalidLayout)?;
        let fresh_group_size = component_count
            .checked_mul(coefficient_count)
            .and_then(|value| value.checked_mul(fresh_error_base_p_digit_count))
            .ok_or(RefreshError::InvalidLayout)?;
        mask_slot_count
            .checked_mul(mask_group_size)
            .and_then(|value| value.checked_add(fresh_group_size))
            .ok_or(RefreshError::InvalidLayout)?;
        Ok(Self {
            refresh_id,
            mask_slot_count,
            component_count,
            coefficient_count,
            mask_base_p_digit_count,
            fresh_error_base_p_digit_count,
        })
    }

    /// Returns the total number of mask and fresh-error labels.
    pub fn len(&self) -> usize {
        self.mask_slot_count * self.mask_group_size() + self.fresh_group_size()
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
        let mask_total = self.mask_slot_count * self.mask_group_size();
        if index < mask_total {
            let group = index / self.mask_group_size();
            let within = index % self.mask_group_size();
            let digit = within % self.mask_base_p_digit_count;
            let coefficient = (within / self.mask_base_p_digit_count) % self.coefficient_count;
            let component = within / (self.mask_base_p_digit_count * self.coefficient_count);
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
            self.mask_base_p_digit_count
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
            group_offset * self.mask_group_size() + offset
        } else {
            self.mask_slot_count * self.mask_group_size() + offset
        })
    }

    fn mask_group_size(&self) -> usize {
        self.component_count * self.coefficient_count * self.mask_base_p_digit_count
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
    output_wire: crate::program::ProgramWireId,
    terminal_form: SparseLwrPrfTerminalForm,
    q_l: usize,
    p: usize,
    lut_width: usize,
    ring_dimension: usize,
}

impl RefreshPrfContract {
    /// Derives the contract from the expected canonical sparse-LWR program.
    pub(crate) fn from_program(program: &SparseLwrPrfProgram) -> Self {
        let profile = program.profile();
        Self {
            program_id: program.id(),
            output_wire: program.terminal_output_wire(),
            terminal_form: SparseLwrPrfTerminalForm::RawScalar,
            q_l: profile.q_l(),
            p: profile.p(),
            lut_width: profile.lut_width(),
            ring_dimension: profile.ring_dimension(),
        }
    }

    pub(crate) const fn from_parts(
        program_id: crate::program::ExponentLutProgramId,
        output_wire: crate::program::ProgramWireId,
        terminal_form: SparseLwrPrfTerminalForm,
        q_l: usize,
        p: usize,
        lut_width: usize,
        ring_dimension: usize,
    ) -> Self {
        Self { program_id, output_wire, terminal_form, q_l, p, lut_width, ring_dimension }
    }

    /// Validates the contract against the refresh setup dimensions.
    pub(crate) fn validate_for(
        &self,
        parameters: &crate::refresh_setup::RefreshSetupParameters,
    ) -> Result<(), RefreshError> {
        let ring_dimension = parameters
            .layout
            .ring_dimension
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?
            .to_usize()
            .ok_or(RefreshError::InvalidLayout)?;
        if self.terminal_form != SparseLwrPrfTerminalForm::RawScalar ||
            self.q_l == 0 ||
            self.q_l > self.lut_width ||
            self.p < 2 ||
            self.p != parameters.base_p ||
            self.lut_width != parameters.lut_width ||
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

    pub(crate) const fn q_l(&self) -> usize {
        self.q_l
    }

    pub(crate) const fn p(&self) -> usize {
        self.p
    }

    pub(crate) const fn lut_width(&self) -> usize {
        self.lut_width
    }

    pub(crate) const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }

    pub(crate) const fn output_wire(&self) -> crate::program::ProgramWireId {
        self.output_wire
    }

    pub(crate) const fn terminal_form(&self) -> SparseLwrPrfTerminalForm {
        self.terminal_form
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
    mask_base_p_digit_count: usize,
    fresh_error_base_p_digit_count: usize,
}

#[allow(dead_code)]
impl RefreshPrfCoverage {
    pub(crate) fn new(
        refresh_id: [u8; 32],
        component_count: usize,
        coefficient_count: usize,
        mask_base_p_digit_count: usize,
        fresh_error_base_p_digit_count: usize,
    ) -> Result<Self, RefreshError> {
        if component_count == 0 ||
            coefficient_count == 0 ||
            mask_base_p_digit_count == 0 ||
            fresh_error_base_p_digit_count == 0
        {
            return Err(RefreshError::InvalidLayout);
        }
        Ok(Self {
            refresh_id,
            component_count,
            coefficient_count,
            mask_base_p_digit_count,
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
    public_keys: Family<Mat>,
    slot_count: usize,
    family_identity: [u8; 32],
    family_pair_id: [u8; 32],
    reduction_helper_commitment: [u8; 32],
    terminal_helper_commitment: [u8; 32],
}

impl RefreshPrfFamilyMaterial {
    pub(crate) fn from_pbc_family_outputs(
        output: &PbcSparseLwrEncodingOutputs,
        index: &RefreshPrfLabelIndex,
        coverage: RefreshPrfCoverage,
        contract: RefreshPrfContract,
    ) -> Result<Self, RefreshError> {
        let (program_id, output_wire, layout_id, metadata_count) = output.family_metadata();
        if program_id != contract.program_id() ||
            output_wire != contract.output_wire() ||
            metadata_count != index.len() ||
            !family_has_count(output.vectors(), index.len())? ||
            !family_has_count(output.public_keys(), index.len())?
        {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(Self {
            coverage,
            contract,
            layout_id,
            vectors: output.vectors().clone(),
            public_keys: output.public_keys().clone(),
            slot_count: index.mask_slot_count,
            family_identity: output.batch_id().0,
            family_pair_id: output.family_pair_id(),
            reduction_helper_commitment: output.helper_commitments().0,
            terminal_helper_commitment: output.helper_commitments().1,
        })
    }

    pub(crate) const fn family_identity(&self) -> [u8; 32] {
        self.family_identity
    }

    pub(crate) const fn family_pair_id(&self) -> [u8; 32] {
        self.family_pair_id
    }

    pub(crate) const fn helper_commitments(&self) -> ([u8; 32], [u8; 32]) {
        (self.reduction_helper_commitment, self.terminal_helper_commitment)
    }

    fn mask_family(&self) -> Result<(Family<Mat>, Family<Mat>), RefreshError> {
        let group = self
            .coverage
            .component_count
            .checked_mul(self.coverage.coefficient_count)
            .and_then(|value| value.checked_mul(self.coverage.mask_base_p_digit_count))
            .ok_or(RefreshError::InvalidLayout)?;
        let count =
            self.coverage_slot_count().checked_mul(group).ok_or(RefreshError::InvalidLayout)?;
        let indices = Parallel::range(count)
            .map_values(|index| index.as_int())
            .map_err(|_| RefreshError::InvalidLayout)?;
        let vectors = self
            .vectors
            .clone()
            .parallel_gather(indices.clone())
            .map_err(|_| RefreshError::InvalidLayout)?;
        let public_keys = self
            .public_keys
            .clone()
            .parallel_gather(indices)
            .map_err(|_| RefreshError::InvalidLayout)?;
        Ok((vectors, public_keys))
    }

    pub(crate) fn fresh_error(&self) -> Result<(Family<Mat>, Family<Mat>), RefreshError> {
        let mask_group = self
            .coverage
            .component_count
            .checked_mul(self.coverage.coefficient_count)
            .and_then(|value| value.checked_mul(self.coverage.mask_base_p_digit_count))
            .ok_or(RefreshError::InvalidLayout)?;
        let fresh_group = self
            .coverage
            .component_count
            .checked_mul(self.coverage.coefficient_count)
            .and_then(|value| value.checked_mul(self.coverage.fresh_error_base_p_digit_count))
            .ok_or(RefreshError::InvalidLayout)?;
        let slots = self.coverage_slot_count();
        let start = slots.checked_mul(mask_group).ok_or(RefreshError::InvalidLayout)?;
        let indices = Parallel::range(fresh_group)
            .map_values(|index| mxx_dsl::Int::constant(start).add(index.as_int()))
            .map_err(|_| RefreshError::InvalidLayout)?;
        let vectors = self
            .vectors
            .clone()
            .parallel_gather(indices.clone())
            .map_err(|_| RefreshError::InvalidLayout)?;
        let public_keys = self
            .public_keys
            .clone()
            .parallel_gather(indices)
            .map_err(|_| RefreshError::InvalidLayout)?;
        Ok((vectors, public_keys))
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
                family.family_pair_id() == [0; 32] ||
                family.helper_commitments().0 == [0; 32] ||
                family.helper_commitments().1 == [0; 32]
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
    compiler: &ExponentLutEncodingCompiler,
    base_p: usize,
    material: &RefreshFreshErrorMaterial,
    scales: Vec<Mat>,
) -> Result<Vec<BggEncodingWire>, RefreshError> {
    let family = material.family.as_ref().ok_or(RefreshError::PrfOutputMismatch)?;
    let (vectors, public_keys) = family.fresh_error()?;
    let component_count = family.coverage.component_count;
    let coefficient_count = family.coverage.coefficient_count;
    let digit_count = family.coverage.fresh_error_base_p_digit_count;
    let group = component_count
        .checked_mul(coefficient_count)
        .and_then(|value| value.checked_mul(digit_count))
        .ok_or(RefreshError::InvalidLayout)?;
    let slot_count = scales.len();
    if slot_count == 0 ||
        !family_has_count(&vectors, group)? ||
        !family_has_count(&public_keys, group)?
    {
        return Err(RefreshError::PrfOutputMismatch);
    }
    let total = group.checked_mul(slot_count).ok_or(RefreshError::InvalidLayout)?;
    let raw_indices = Parallel::range(total)
        .map_values(|index| {
            let flat = index.as_int();
            let quotient = flat.clone().div(Int::constant(group));
            flat.sub(quotient.mul(Int::constant(group)))
        })
        .map_err(|_| RefreshError::InvalidLayout)?;
    let slot_indices = Parallel::range(total)
        .map_values(|index| index.as_int().div(Int::constant(group)))
        .map_err(|_| RefreshError::InvalidLayout)?;
    let repeated_vectors =
        vectors.parallel_gather(raw_indices.clone()).map_err(|_| RefreshError::InvalidLayout)?;
    let repeated_public_keys =
        public_keys.parallel_gather(raw_indices).map_err(|_| RefreshError::InvalidLayout)?;
    let scale_family = Family::pack(scales).map_err(|_| RefreshError::InvalidLayout)?;
    let repeated_scales =
        scale_family.parallel_gather(slot_indices).map_err(|_| RefreshError::InvalidLayout)?;
    let routed = route_prf_family(
        compiler,
        base_p,
        repeated_vectors,
        repeated_public_keys,
        component_count,
        coefficient_count,
        digit_count,
        Some(repeated_scales),
    )?;
    let vectors = reduce_family_segments(routed.0, slot_count, group)?;
    let public_keys = reduce_family_segments(routed.1, slot_count, group)?;
    (0..slot_count)
        .map(|slot| {
            Ok(BggEncodingWire {
                vector: vectors.get_static(slot),
                pubkey: BggPublicKeyWire {
                    matrix: public_keys.get_static(slot),
                    reveal_plaintext: false,
                },
                plaintext: None,
            })
        })
        .collect()
}

/// Routes a complete canonical family with one symbolic body. Callers may
/// then perform one or more structural balanced reductions over its output;
/// no label is projected to a host value during this phase.
fn route_prf_family(
    compiler: &ExponentLutEncodingCompiler,
    base_p: usize,
    vectors: Family<Mat>,
    public_keys: Family<Mat>,
    component_count: usize,
    coefficient_count: usize,
    digit_count: usize,
    scales: Option<Family<Mat>>,
) -> Result<(Family<Mat>, Family<Mat>), RefreshError> {
    let has_scales = scales.is_some();
    let mut families = vec![vectors, public_keys];
    if let Some(scales) = scales {
        families.push(scales);
    }
    Family::<Mat>::try_parallel_zip_many_values(families, |index, mut inputs| {
        let scale = has_scales.then(|| inputs.pop().expect("scale family"));
        let public_key = inputs.pop().ok_or(mxx_dsl::DslError::Schema)?;
        let vector = inputs.pop().ok_or(mxx_dsl::DslError::Schema)?;
        let wire = BggEncodingWire {
            vector,
            pubkey: BggPublicKeyWire { matrix: public_key, reveal_plaintext: false },
            plaintext: None,
        };
        let flat = index.expression();
        let group_size = component_count
            .checked_mul(coefficient_count)
            .and_then(|value| value.checked_mul(digit_count))
            .ok_or(mxx_dsl::DslError::Schema)?;
        let group_quotient =
            IntExpr::Div(Box::new(flat.clone()), Box::new(IntExpr::constant(group_size)));
        let within_group = IntExpr::Sub(
            Box::new(flat.clone()),
            Box::new(IntExpr::Mul(
                Box::new(group_quotient),
                Box::new(IntExpr::constant(group_size)),
            )),
        )
        .canonicalize();
        // `IntExpr` deliberately has no remainder node. Express the
        // mixed-radix decode using quotient and subtraction while keeping
        // the loop index symbolic in the generated routing body.
        let digit_quotient =
            IntExpr::Div(Box::new(within_group.clone()), Box::new(IntExpr::constant(digit_count)));
        let digit = IntExpr::Sub(
            Box::new(within_group.clone()),
            Box::new(IntExpr::Mul(
                Box::new(digit_quotient.clone()),
                Box::new(IntExpr::constant(digit_count)),
            )),
        )
        .canonicalize();
        let coefficient_quotient = IntExpr::Div(
            Box::new(digit_quotient.clone()),
            Box::new(IntExpr::constant(coefficient_count)),
        );
        let coefficient = IntExpr::Sub(
            Box::new(digit_quotient),
            Box::new(IntExpr::Mul(
                Box::new(coefficient_quotient),
                Box::new(IntExpr::constant(coefficient_count)),
            )),
        )
        .canonicalize();
        let component = IntExpr::Div(
            Box::new(within_group),
            Box::new(IntExpr::constant(
                coefficient_count.checked_mul(digit_count).ok_or(mxx_dsl::DslError::Schema)?,
            )),
        )
        .canonicalize();
        let route = symbolic_prf_route_matrix(&wire, base_p, digit, coefficient, component)
            .map_err(|_| mxx_dsl::DslError::Schema)?;
        let route = match scale {
            Some(scale) => scale * route,
            None => route,
        };
        let output = compiler.bgg.matrix_mul(&wire, &route);
        Ok((output.vector, output.pubkey.matrix))
    })
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
    Parallel::range(segment_count)
        .try_map_values({
            let family = family.clone();
            move |segment| {
                let start = segment.as_int().mul(Int::constant(segment_size));
                let indices = Parallel::range(segment_size)
                    .map_values(|index| start.clone().add(index.as_int()))?;
                let values = family
                    .clone()
                    .parallel_gather(indices)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                crate::encoding::balanced_sum_family(values).map_err(|_| mxx_dsl::DslError::Schema)
            }
        })
        .map_err(|_| RefreshError::InvalidLayout)
}

/// Routes all mask slots once and reduces each canonical slot segment to one
/// encoding. Only the resulting fixed-size CRT-slot vector is materialized.
pub(crate) fn aggregate_refresh_masks(
    compiler: &ExponentLutEncodingCompiler,
    base_p: usize,
    family: &RefreshPrfFamilyMaterial,
    slot_count: usize,
) -> Result<Vec<BggEncodingWire>, RefreshError> {
    if slot_count != family.coverage_slot_count() {
        return Err(RefreshError::SlotOrderMismatch);
    }
    let group = family
        .coverage
        .component_count
        .checked_mul(family.coverage.coefficient_count)
        .and_then(|value| value.checked_mul(family.coverage.mask_base_p_digit_count))
        .ok_or(RefreshError::InvalidLayout)?;
    let mask_total = group.checked_mul(slot_count).ok_or(RefreshError::InvalidLayout)?;
    let fresh_group = family
        .coverage
        .component_count
        .checked_mul(family.coverage.coefficient_count)
        .and_then(|value| value.checked_mul(family.coverage.fresh_error_base_p_digit_count))
        .ok_or(RefreshError::InvalidLayout)?;
    let total = mask_total.checked_add(fresh_group).ok_or(RefreshError::InvalidLayout)?;
    if !family_has_count(&family.vectors, total)? || !family_has_count(&family.public_keys, total)?
    {
        return Err(RefreshError::PrfOutputMismatch);
    }
    let (mask_vectors, mask_public_keys) = family.mask_family()?;
    let routed = route_prf_family(
        compiler,
        base_p,
        mask_vectors,
        mask_public_keys,
        family.coverage.component_count,
        family.coverage.coefficient_count,
        family.coverage.mask_base_p_digit_count,
        None,
    )?;
    let vectors = reduce_family_segments(routed.0, slot_count, group)?;
    let public_keys = reduce_family_segments(routed.1, slot_count, group)?;
    if !family_has_count(&vectors, slot_count)? || !family_has_count(&public_keys, slot_count)? {
        return Err(RefreshError::PrfOutputMismatch);
    }
    Ok((0..slot_count)
        .map(|slot| BggEncodingWire {
            vector: vectors.get_static(slot),
            pubkey: BggPublicKeyWire {
                matrix: public_keys.get_static(slot),
                reveal_plaintext: false,
            },
            plaintext: None,
        })
        .collect())
}

fn symbolic_prf_route_matrix(
    value: &BggEncodingWire,
    base_p: usize,
    digit: IntExpr,
    coefficient: IntExpr,
    component: IntExpr,
) -> Result<Mat, RefreshError> {
    let vector_type = value.vector.matrix_type();
    let public_key_type = value.pubkey.matrix.matrix_type();
    if public_key_type.rows.evaluate(&Default::default()).ok().and_then(|value| value.to_usize()) !=
        Some(2)
    {
        return Err(RefreshError::InvalidLayout);
    }
    let columns = public_key_type.columns.clone();
    let ring = mxx_dsl::Ring::new(vector_type.modulus.clone(), vector_type.ring_dimension.clone());
    let scalar = ring.constant(
        (1, 1),
        mxx_ir_core::node::ConstantMatrix::PowerOfBase {
            base: IntExpr::constant(base_p),
            exponent: digit,
        },
    ) * ring
        .constant((1, 1), mxx_ir_core::node::ConstantMatrix::Rotation { exponent: coefficient });
    let secret_dimension = public_key_type.rows.clone();
    Ok(scalar *
        ring.constant(
            (secret_dimension, 1),
            mxx_ir_core::node::ConstantMatrix::UnitColumn { index: IntExpr::constant(1) },
        ) *
        ring.constant(
            (1, columns),
            mxx_ir_core::node::ConstantMatrix::UnitRow { index: component },
        ))
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
                &slot.q % &slot.q_t != 0.into()
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
        let target = b.mul_small_rhs(k);
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
        crate::ensure_ciphertext_only(&encoding).map_err(RefreshError::ExponentLut)?;
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
/// decode before combining them. The `b`, `k`, and anchor product are setup
/// evidence for fail-closed linkage checks, not additional public outputs.
#[derive(Clone)]
struct RefreshScalarPackage {
    slot: usize,
    state_handle: mxx_ir_core::ValueHandle,
    a_prime_handle: mxx_ir_core::ValueHandle,
    a_sum_t: Mat,
    scale_target: Mat,
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
        a_sum_t: Mat,
        scale_target: Mat,
        mask: BggEncodingWire,
        fresh_error_source: BggEncodingWire,
        decoder: RefreshDecoderPreimage,
        decoder_base_handle: mxx_ir_core::ValueHandle,
        target_public_matrix: Mat,
        anchor: &RefreshAnchor,
        b: Mat,
        k: Preimage,
    ) -> Result<Self, RefreshError> {
        crate::ensure_ciphertext_only(state).map_err(RefreshError::ExponentLut)?;
        crate::ensure_ciphertext_only(&mask).map_err(RefreshError::ExponentLut)?;
        crate::ensure_ciphertext_only(&fresh_error_source).map_err(RefreshError::ExponentLut)?;
        crate::ensure_ciphertext_only(&decoder.encoding).map_err(RefreshError::ExponentLut)?;
        decoder.validate(anchor, &target_public_matrix)?;
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
            a_sum_t,
            scale_target,
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
        let decoder_base_handle = packages[0].decoder_base_handle.clone();
        for (slot, package) in packages.iter().enumerate() {
            if package.slot != slot ||
                package.state_handle != *state.vector.value_handle() ||
                package.a_prime_handle != *a_prime.value_handle() ||
                package.mask.vector.matrix_type() != state.vector.matrix_type() ||
                package.fresh_error_source.vector.matrix_type() != state.vector.matrix_type() ||
                package.decoder_base_handle != decoder_base_handle
            {
                return Err(if package.slot != slot {
                    RefreshError::SlotOrderMismatch
                } else {
                    RefreshError::InvalidLayout
                });
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"mxx.exponent-lut.refresh-setup");
        hash.update(format!("{:?}", state.vector.value_handle()).as_bytes());
        hash.update(format!("{:?}", a_prime.value_handle()).as_bytes());
        for package in &packages {
            hash.update(package.slot.to_le_bytes());
            hash.update(format!("{:?}", package.a_sum_t.value_handle()).as_bytes());
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
    /// The imported shared decoder base and per-slot preimages are deliberately plain wires
    /// at this boundary.  This helper is the only crate-internal adapter that
    /// turns them back into the typed per-slot packages: each target is
    /// recomputed from the imported state/mask/fresh material and the exact
    /// decoder is rebuilt as the shared `base.vector * K_t`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_imported_wires(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        state: BggEncodingWire,
        a_prime: Mat,
        public_b: Mat,
        scaled_freshes: Vec<BggEncodingWire>,
        masks: Vec<BggEncodingWire>,
        decoder_base: BggEncodingWire,
        preimages: Vec<Preimage>,
    ) -> Result<RefreshSetupManifest, RefreshError> {
        self.validate_layout()?;
        crate::ensure_ciphertext_only(&state).map_err(RefreshError::ExponentLut)?;
        if masks.len() != self.crt_plaintext_moduli.len() ||
            scaled_freshes.len() != masks.len() ||
            preimages.len() != masks.len() ||
            !same_matrix_type(a_prime.matrix_type(), state.pubkey.matrix.matrix_type())
        {
            return Err(RefreshError::InvalidLayout);
        }
        // `public_b` is the common trapdoor public matrix B. Every slot below
        // must use the same B handle; only the target and preimage K_t vary.
        let base_handle = public_b.value_handle().clone();
        let mut packages = Vec::with_capacity(masks.len());
        for (slot, ((mask, scaled_fresh), k)) in
            masks.into_iter().zip(scaled_freshes.into_iter()).zip(preimages.into_iter()).enumerate()
        {
            let base = decoder_base.clone();
            let decoder_base_handle = base.vector.value_handle().clone();
            crate::ensure_ciphertext_only(&mask).map_err(RefreshError::ExponentLut)?;
            crate::ensure_ciphertext_only(&base).map_err(RefreshError::ExponentLut)?;
            if base.pubkey.reveal_plaintext || base.pubkey.matrix.value_handle() != &base_handle {
                return Err(RefreshError::TargetMismatch);
            }
            // `scale = mu_t = q/q_t`; multiplying by its gadget decomposition
            // implements `u_t=c G^{-1}(mu_t G)` and `A_t=A G^{-1}(mu_t G)`.
            let scale = mxx_dsl::Ring::new(
                state.vector.matrix_type().modulus.clone(),
                state.vector.matrix_type().ring_dimension.clone(),
            )
            .polynomial([self.scale_expression(slot)?]);
            let scaled_state = compiler.bgg.large_scalar_mul(&state, &scale);
            let combined =
                compiler.bgg.add(&compiler.bgg.add(&scaled_state, &mask)?, &scaled_fresh)?;
            // Aggregate the scaled state, mask, and fresh-error public terms:
            // `A_{sum,t} = A_t + A_{m,t} + A_{e,t}`.
            let a_sum_t = combined.pubkey.matrix;
            let target = a_sum_t.clone() - scale.clone() * a_prime.clone();
            // `target = A_{sum,t} - mu_t A'`; the imported preimage K_t must
            // satisfy `B K_t = target` and is checked by RefreshDecoderPreimage.
            let decoder = BggEncodingWire {
                vector: base.vector.mul_small_rhs(k.clone()),
                pubkey: BggPublicKeyWire { matrix: target.clone(), reveal_plaintext: false },
                plaintext: None,
            };
            let anchor = RefreshAnchor::with_equation(public_b.clone(), k.clone());
            let decoder = RefreshDecoderPreimage::bind(&anchor, decoder, &target)?;
            packages.push(RefreshScalarPackage::new(
                slot,
                &state,
                &a_prime,
                a_sum_t,
                scale,
                mask,
                scaled_fresh,
                decoder,
                decoder_base_handle,
                target,
                &anchor,
                public_b.clone(),
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
        let _ = packages.first().ok_or(RefreshError::InvalidLayout)?;
        // Package fields are exposed as indexed families once, then one
        // The structural family operation performs the complete per-slot equation.
        // This keeps CRT-slot work independent while preserving the caller's
        // declared slot order for the final recomposition.
        let scales =
            Family::pack(packages.iter().map(|package| package.scale_target.clone()).collect())
                .map_err(|_| RefreshError::InvalidLayout)?;
        let masks_vector =
            Family::pack(packages.iter().map(|package| package.mask.vector.clone()).collect())
                .map_err(|_| RefreshError::InvalidLayout)?;
        let masks_pubkey = Family::pack(
            packages.iter().map(|package| package.mask.pubkey.matrix.clone()).collect(),
        )
        .map_err(|_| RefreshError::InvalidLayout)?;
        let fresh_vectors = Family::pack(
            packages.iter().map(|package| package.fresh_error_source.vector.clone()).collect(),
        )
        .map_err(|_| RefreshError::InvalidLayout)?;
        let fresh_pubkeys = Family::pack(
            packages
                .iter()
                .map(|package| package.fresh_error_source.pubkey.matrix.clone())
                .collect(),
        )
        .map_err(|_| RefreshError::InvalidLayout)?;
        let decoder_vectors = Family::pack(
            packages.iter().map(|package| package.decoder.encoding.vector.clone()).collect(),
        )
        .map_err(|_| RefreshError::InvalidLayout)?;
        let decoder_targets = Family::pack(
            packages.iter().map(|package| package.target_public_matrix.clone()).collect(),
        )
        .map_err(|_| RefreshError::InvalidLayout)?;

        let levels_family = Family::parallel_zip_many_values(
            vec![
                scales,
                masks_vector,
                masks_pubkey,
                fresh_vectors,
                fresh_pubkeys,
                decoder_vectors,
                decoder_targets,
            ],
            |_slot, mut values| {
                let decoder_target = values.pop().expect("decoder target family");
                let decoder_vector = values.pop().expect("decoder vector family");
                let fresh_pubkey = values.pop().expect("fresh public family");
                let fresh_vector = values.pop().expect("fresh vector family");
                let mask_pubkey = values.pop().expect("mask public family");
                let mask_vector = values.pop().expect("mask vector family");
                let scale = values.pop().expect("scale family");
                let fresh = BggEncodingWire {
                    vector: fresh_vector,
                    pubkey: BggPublicKeyWire { matrix: fresh_pubkey, reveal_plaintext: false },
                    plaintext: None,
                };
                let mask = BggEncodingWire {
                    vector: mask_vector,
                    pubkey: BggPublicKeyWire { matrix: mask_pubkey, reveal_plaintext: false },
                    plaintext: None,
                };
                let decoder = BggEncodingWire {
                    vector: decoder_vector,
                    pubkey: BggPublicKeyWire { matrix: decoder_target, reveal_plaintext: false },
                    plaintext: None,
                };
                let scaled = compiler.bgg.large_scalar_mul(state, &scale);
                let combined = compiler.bgg.add(&scaled, &mask).expect("validated refresh add");
                let combined_full =
                    compiler.bgg.add(&combined, &fresh).expect("validated refresh add");
                compiler
                    .bgg
                    .sub(&combined_full, &decoder)
                    .expect("validated refresh subtract")
                    .vector
            },
        )
        .map_err(|_| RefreshError::InvalidLayout)?;
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
            let level = levels_family.get_static(slot);
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
                (format!("slot-{slot}-a-sum"), package.a_sum_t.value_handle().clone()),
                (
                    format!("slot-{slot}-target"),
                    package.target_public_matrix.value_handle().clone(),
                ),
                (format!("slot-{slot}-b"), package.b.value_handle().clone()),
                (format!("slot-{slot}-k"), package.k.value_handle().clone()),
                (
                    format!("slot-{slot}-b-k-product"),
                    package.b.clone().mul_small_rhs(package.k.clone()).value_handle().clone(),
                ),
            ]);
            levels.push(level);
        }
        let vector = Mat::crt_recompose(
            levels,
            self.crt_plaintext_moduli.clone(),
            self.reconstruction_coefficients.clone(),
        );
        let declaration = self.refresh_declaration(setup, &vector)?;
        Ok(RefreshResult {
            encoding: BggEncodingWire {
                vector,
                pubkey: BggPublicKeyWire { matrix: setup.a_prime.clone(), reveal_plaintext: false },
                plaintext: None,
            },
            declaration,
        })
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
    .all(|(left, right)| left.evaluate(&environment).ok() == right.evaluate(&environment).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prf_labels_cover_all_public_key_columns_with_separate_digit_groups() {
        let index = RefreshPrfLabelIndex::new([7; 32], 2, 6, 2, 3, 1).expect("label index");
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
                slot_count,
                component_count,
                coefficient_count,
                dm,
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
}
