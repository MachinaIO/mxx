//! The direct section 7 RNS refresh protocol.
//!
//! This is deliberately separate from mxx_bgg::noise_refresh: it operates
//! on scalar BGG encodings and models the manuscript order
//! scaled state + mask + fresh error - combined decoder before CRT
//! recomposition. Sparse-LWR PRF evaluation produces the mask and fresh-error
//! wires consumed here; this module owns only CRT routing, scaling, and
//! aggregation, so it does not duplicate the PRF program.  Noise correctness
//! is intentionally not modeled by this protocol declaration: the generic
//! operational-noise checker analyzes the actual graph, while this module
//! checks only structural equations and identities.

use crate::{
    PowerLutEncodingCompiler, PowerLutError,
    pbc::PbcLayoutId,
    prf::{PbcSparseLwrEncodingOutput, SparseLwrPrfOutput},
};
use mxx_bgg::{BggEncodingWire, BggPublicKeyWire};
use mxx_dsl::{BuiltGraph, DerivationAttachmentValue, Family, FrozenDerivationAttachment, Mat};
use mxx_ir_core::{
    FrozenGraphScopeId, Graph, GraphScope, IntExpr, ScopedWireRef, WireRef, WireType,
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
/// Validation and graph-linkage failures in the Section 7 refresh protocol.
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
    /// A generic Power-LUT operation failed.
    Power(#[from] PowerLutError),
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
/// digit-major.  This is public metadata only and contains no selector,
/// support, schedule, or plaintext material.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshPrfLabelIndex {
    refresh_id: [u8; 32],
    mask_slot_count: usize,
    component_count: usize,
    coefficient_count: usize,
    digit_count: usize,
}

impl RefreshPrfLabelIndex {
    /// Creates a canonical label index for a refresh instance.
    pub fn new(
        refresh_id: [u8; 32],
        mask_slot_count: usize,
        component_count: usize,
        coefficient_count: usize,
        digit_count: usize,
    ) -> Result<Self, RefreshError> {
        if component_count == 0 || coefficient_count == 0 || digit_count == 0 {
            return Err(RefreshError::InvalidLayout);
        }
        // Validate the complete cardinality up front so index arithmetic is
        // exact and cannot wrap while building structural loop bounds.
        let group_size = component_count
            .checked_mul(coefficient_count)
            .and_then(|value| value.checked_mul(digit_count))
            .ok_or(RefreshError::InvalidLayout)?;
        mask_slot_count
            .checked_add(1)
            .and_then(|groups| groups.checked_mul(group_size))
            .ok_or(RefreshError::InvalidLayout)?;
        Ok(Self { refresh_id, mask_slot_count, component_count, coefficient_count, digit_count })
    }

    /// Returns the total number of mask and fresh-error labels.
    pub fn len(&self) -> usize {
        (self.mask_slot_count + 1) * self.group_size()
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
        let group_size = self.group_size();
        let group = index / group_size;
        let within = index % group_size;
        let digit = within % self.digit_count;
        let coefficient = (within / self.digit_count) % self.coefficient_count;
        let component = within / (self.digit_count * self.coefficient_count);
        if group < self.mask_slot_count {
            Some(RefreshPrfLabel::Mask {
                refresh_id: self.refresh_id,
                slot: group,
                component,
                coefficient,
                digit,
            })
        } else {
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
        let (group, component, coefficient, digit) = match label {
            RefreshPrfLabel::Mask { refresh_id, slot, component, coefficient, digit }
                if refresh_id == self.refresh_id && slot < self.mask_slot_count =>
            {
                (slot, component, coefficient, digit)
            }
            RefreshPrfLabel::FreshError { refresh_id, component, coefficient, digit }
                if refresh_id == self.refresh_id =>
            {
                (self.mask_slot_count, component, coefficient, digit)
            }
            _ => return None,
        };
        if component >= self.component_count ||
            coefficient >= self.coefficient_count ||
            digit >= self.digit_count
        {
            return None;
        }
        Some(
            group * self.group_size() +
                component * self.coefficient_count * self.digit_count +
                coefficient * self.digit_count +
                digit,
        )
    }

    fn group_size(&self) -> usize {
        self.component_count * self.coefficient_count * self.digit_count
    }
}

/// A sparse-LWR PRF output paired with its lowered BGG wire and public label.
///
/// The descriptor is created by the sparse-LWR program lowerer. Its
/// [`crate::program::ProgramWireId`] identifies the declared output of a
/// particular canonical program, while the label digest identifies the
/// domain-separated refresh purpose. Refresh consumes this typed boundary and
/// therefore does not accept an unclassified mask or error wire.
#[derive(Clone)]
pub struct RefreshPrfOutput {
    descriptor: SparseLwrPrfOutput,
    encoding: BggEncodingWire,
    label: RefreshPrfLabel,
    layout_id: PbcLayoutId,
}

impl RefreshPrfOutput {
    /// Binds the wire and descriptor returned by the real PBC lowering.
    ///
    /// The descriptor must already identify a declared program output. This
    /// constructor additionally recomputes the canonical digest of `label`;
    /// a descriptor from another refresh purpose is rejected before the wire
    /// enters the refresh setup.
    fn from_pbc_evaluation(
        output: PbcSparseLwrEncodingOutput,
        label: RefreshPrfLabel,
    ) -> Result<Self, RefreshError> {
        let encoding = output.encoding().clone();
        let descriptor = output.descriptor();
        crate::ensure_ciphertext_only(&encoding).map_err(RefreshError::Power)?;
        if descriptor.label_digest() != &refresh_prf_label_digest(label) {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(Self { descriptor, encoding, label, layout_id: output.layout_id() })
    }

    /// Returns the typed sparse-LWR descriptor.
    pub fn descriptor(&self) -> SparseLwrPrfOutput {
        self.descriptor
    }

    /// Returns the lowered BGG encoding wire.
    pub fn encoding(&self) -> &BggEncodingWire {
        &self.encoding
    }

    /// Returns the canonical public refresh label.
    pub fn label(&self) -> RefreshPrfLabel {
        self.label
    }

    pub(crate) fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }
}

/// Typed per-slot sparse-LWR mask output.
#[derive(Clone)]
pub struct RefreshMaskPrfOutput(RefreshPrfOutput);

impl RefreshMaskPrfOutput {
    /// Binds a real PBC evaluation to a canonical per-slot mask label.
    pub fn from_pbc_evaluation(
        output: PbcSparseLwrEncodingOutput,
        refresh_id: [u8; 32],
        slot: usize,
        component: usize,
        coefficient: usize,
        digit: usize,
    ) -> Result<Self, RefreshError> {
        let label = RefreshPrfLabel::Mask { refresh_id, slot, component, coefficient, digit };
        Ok(Self(RefreshPrfOutput::from_pbc_evaluation(output, label)?))
    }

    fn encoding(&self) -> &BggEncodingWire {
        self.0.encoding()
    }
}

/// Typed sparse-LWR fresh-error output shared by every CRT slot.
#[derive(Clone)]
pub struct RefreshFreshErrorPrfOutput(RefreshPrfOutput);

impl RefreshFreshErrorPrfOutput {
    /// Binds a real PBC evaluation to the canonical shared fresh-error label.
    pub fn from_pbc_evaluation(
        output: PbcSparseLwrEncodingOutput,
        refresh_id: [u8; 32],
        component: usize,
        coefficient: usize,
        digit: usize,
    ) -> Result<Self, RefreshError> {
        let label = RefreshPrfLabel::FreshError { refresh_id, component, coefficient, digit };
        Ok(Self(RefreshPrfOutput::from_pbc_evaluation(output, label)?))
    }

    fn encoding(&self) -> &BggEncodingWire {
        self.0.encoding()
    }
}

/// Exact finite label coverage required for one refresh PRF aggregate.
///
/// This type is intentionally opaque and carries no schedule, selector, or
/// artifact information. It is created only by the crate's PRF binder.
#[allow(dead_code)]
#[derive(Clone, Eq, PartialEq)]
pub struct RefreshPrfCoverage {
    refresh_id: [u8; 32],
    component_count: usize,
    coefficient_count: usize,
    digit_count: usize,
}

#[allow(dead_code)]
impl RefreshPrfCoverage {
    pub(crate) fn new(
        refresh_id: [u8; 32],
        component_count: usize,
        coefficient_count: usize,
        digit_count: usize,
    ) -> Result<Self, RefreshError> {
        if component_count == 0 || coefficient_count == 0 || digit_count == 0 {
            return Err(RefreshError::InvalidLayout);
        }
        Ok(Self { refresh_id, component_count, coefficient_count, digit_count })
    }
}

/// Opaque, completely covered per-slot mask material.
#[allow(dead_code)]
#[derive(Clone)]
pub struct RefreshMaskMaterial {
    coverage: RefreshPrfCoverage,
    slot: usize,
    program_id: crate::program::PowerLutProgramId,
    layout_id: PbcLayoutId,
    outputs: Vec<RefreshMaskPrfOutput>,
}

#[allow(dead_code)]
impl RefreshMaskMaterial {
    pub(crate) fn new(
        coverage: RefreshPrfCoverage,
        slot: usize,
        program_id: crate::program::PowerLutProgramId,
        outputs: Vec<RefreshMaskPrfOutput>,
    ) -> Result<Self, RefreshError> {
        validate_mask_coverage(&coverage, slot, program_id, &outputs)?;
        let layout_id = outputs.first().ok_or(RefreshError::PrfOutputMismatch)?.0.layout_id();
        if outputs.iter().any(|output| output.0.layout_id() != layout_id) {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(Self { coverage, slot, program_id, layout_id, outputs })
    }

    pub(crate) fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }

    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    pub(crate) fn program_id(&self) -> crate::program::PowerLutProgramId {
        self.program_id
    }

    pub(crate) fn coverage_matches(&self, coverage: &RefreshPrfCoverage) -> bool {
        self.coverage == *coverage
    }

    pub(crate) fn validate(&self) -> Result<(), RefreshError> {
        validate_mask_coverage(&self.coverage, self.slot, self.program_id, &self.outputs)?;
        if self.outputs.iter().any(|output| output.0.layout_id() != self.layout_id) {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(())
    }
}

/// Opaque, completely covered fresh-error material shared by all CRT slots.
#[allow(dead_code)]
#[derive(Clone)]
pub struct RefreshFreshErrorMaterial {
    coverage: RefreshPrfCoverage,
    program_id: crate::program::PowerLutProgramId,
    layout_id: PbcLayoutId,
    outputs: Vec<RefreshFreshErrorPrfOutput>,
}

#[allow(dead_code)]
impl RefreshFreshErrorMaterial {
    pub(crate) fn new(
        coverage: RefreshPrfCoverage,
        program_id: crate::program::PowerLutProgramId,
        outputs: Vec<RefreshFreshErrorPrfOutput>,
    ) -> Result<Self, RefreshError> {
        validate_fresh_error_coverage(&coverage, program_id, &outputs)?;
        let layout_id = outputs.first().ok_or(RefreshError::PrfOutputMismatch)?.0.layout_id();
        if outputs.iter().any(|output| output.0.layout_id() != layout_id) {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(Self { coverage, program_id, layout_id, outputs })
    }

    pub(crate) fn program_id(&self) -> crate::program::PowerLutProgramId {
        self.program_id
    }

    pub(crate) fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }

    pub(crate) fn coverage_matches(&self, coverage: &RefreshPrfCoverage) -> bool {
        self.coverage == *coverage
    }

    pub(crate) fn validate(&self) -> Result<(), RefreshError> {
        validate_fresh_error_coverage(&self.coverage, self.program_id, &self.outputs)?;
        if self.outputs.iter().any(|output| output.0.layout_id() != self.layout_id) {
            return Err(RefreshError::PrfOutputMismatch);
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn expected_coverage_size(coverage: &RefreshPrfCoverage) -> Option<usize> {
    coverage
        .component_count
        .checked_mul(coverage.coefficient_count)
        .and_then(|count| count.checked_mul(coverage.digit_count))
}

#[allow(dead_code)]
fn validate_mask_coverage(
    coverage: &RefreshPrfCoverage,
    slot: usize,
    program_id: crate::program::PowerLutProgramId,
    outputs: &[RefreshMaskPrfOutput],
) -> Result<(), RefreshError> {
    let expected = expected_coverage_size(coverage).ok_or(RefreshError::InvalidLayout)?;
    if outputs.len() != expected {
        return Err(RefreshError::PrfOutputMismatch);
    }
    let mut labels = BTreeMap::new();
    for output in outputs {
        if output.0.descriptor().program_id() != program_id {
            return Err(RefreshError::PrfOutputMismatch);
        }
        let RefreshPrfLabel::Mask { refresh_id, slot: output_slot, component, coefficient, digit } =
            output.0.label()
        else {
            return Err(RefreshError::PrfOutputMismatch);
        };
        if refresh_id != coverage.refresh_id ||
            output_slot != slot ||
            component >= coverage.component_count ||
            coefficient >= coverage.coefficient_count ||
            digit >= coverage.digit_count ||
            labels.insert((component, coefficient, digit), ()).is_some()
        {
            return Err(RefreshError::PrfOutputMismatch);
        }
    }
    if labels.len() != expected {
        return Err(RefreshError::PrfOutputMismatch);
    }
    Ok(())
}

#[allow(dead_code)]
fn validate_fresh_error_coverage(
    coverage: &RefreshPrfCoverage,
    program_id: crate::program::PowerLutProgramId,
    outputs: &[RefreshFreshErrorPrfOutput],
) -> Result<(), RefreshError> {
    let expected = expected_coverage_size(coverage).ok_or(RefreshError::InvalidLayout)?;
    if outputs.len() != expected {
        return Err(RefreshError::PrfOutputMismatch);
    }
    let mut labels = BTreeMap::new();
    for output in outputs {
        if output.0.descriptor().program_id() != program_id {
            return Err(RefreshError::PrfOutputMismatch);
        }
        let RefreshPrfLabel::FreshError { refresh_id, component, coefficient, digit } =
            output.0.label()
        else {
            return Err(RefreshError::PrfOutputMismatch);
        };
        if refresh_id != coverage.refresh_id ||
            component >= coverage.component_count ||
            coefficient >= coverage.coefficient_count ||
            digit >= coverage.digit_count ||
            labels.insert((component, coefficient, digit), ()).is_some()
        {
            return Err(RefreshError::PrfOutputMismatch);
        }
    }
    if labels.len() != expected {
        return Err(RefreshError::PrfOutputMismatch);
    }
    Ok(())
}

fn aggregate_prf_digits(
    compiler: &PowerLutEncodingCompiler,
    base_p: usize,
    values: impl IntoIterator<Item = (usize, usize, usize, BggEncodingWire)>,
) -> Result<BggEncodingWire, RefreshError> {
    let values = values.into_iter().collect::<Vec<_>>();
    let first = values.first().ok_or(RefreshError::InvalidLayout)?;
    if values.iter().any(|(_, _, _, value)| {
        value.vector.matrix_type() != first.3.vector.matrix_type() ||
            value.pubkey.matrix.matrix_type() != first.3.pubkey.matrix.matrix_type()
    }) {
        return Err(RefreshError::InvalidLayout);
    }
    for (_, _, _, value) in &values {
        crate::ensure_ciphertext_only(value).map_err(RefreshError::Power)?;
    }

    // Route every label in one reusable structural body. The route matrices
    // are public constants in canonical label order; no per-label addition
    // chain or host-side wire aggregation is emitted.
    let vectors =
        Family::pack(values.iter().map(|(_, _, _, value)| value.vector.clone()).collect())
            .map_err(|_| RefreshError::InvalidLayout)?;
    let public_keys =
        Family::pack(values.iter().map(|(_, _, _, value)| value.pubkey.matrix.clone()).collect())
            .map_err(|_| RefreshError::InvalidLayout)?;
    let routes = Family::pack(
        values
            .iter()
            .map(|(digit, coefficient, component, value)| {
                prf_route_matrix(value, base_p, *digit, *coefficient, *component)
            })
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(|_| RefreshError::InvalidLayout)?;
    let routed = Family::<Mat>::parallel_zip_many_values(
        vec![vectors, public_keys, routes],
        |_index, mut items| {
            let route = items.pop().expect("route family element");
            let public_key = items.pop().expect("public-key family element");
            let vector = items.pop().expect("vector family element");
            let value = BggEncodingWire {
                vector,
                pubkey: BggPublicKeyWire { matrix: public_key, reveal_plaintext: false },
                plaintext: None,
            };
            let routed = compiler.bgg.matrix_mul(&value, &route);
            (routed.vector, routed.pubkey.matrix)
        },
    )
    .map_err(|_| RefreshError::InvalidLayout)?;
    Ok(BggEncodingWire {
        vector: crate::encoding::balanced_sum_family(routed.0).map_err(RefreshError::Power)?,
        pubkey: BggPublicKeyWire {
            matrix: crate::encoding::balanced_sum_family(routed.1).map_err(RefreshError::Power)?,
            reveal_plaintext: false,
        },
        plaintext: None,
    })
}

/// Aggregates a completely covered mask digit set using the canonical route.
#[allow(dead_code)]
pub(crate) fn aggregate_refresh_mask(
    compiler: &PowerLutEncodingCompiler,
    base_p: usize,
    material: &RefreshMaskMaterial,
) -> Result<BggEncodingWire, RefreshError> {
    // Validation is repeated at the consumption boundary so an opaque value
    // cannot be used after an internal mutation or an accidental replacement.
    validate_mask_coverage(
        &material.coverage,
        material.slot,
        material.program_id,
        &material.outputs,
    )?;
    if material.outputs.iter().any(|output| output.0.layout_id() != material.layout_id) {
        return Err(RefreshError::PrfOutputMismatch);
    }
    let mut ordered = BTreeMap::new();
    for output in &material.outputs {
        let RefreshPrfLabel::Mask { component, coefficient, digit, .. } = output.0.label() else {
            unreachable!("validated mask coverage")
        };
        ordered.insert((component, coefficient, digit), output.encoding().clone());
    }
    aggregate_prf_digits(
        compiler,
        base_p,
        ordered
            .into_iter()
            .map(|((component, coefficient, digit), value)| (digit, coefficient, component, value)),
    )
}

/// Aggregates the one shared fresh-error digit set. The returned plain BGG
/// wire is intended to be cloned into every CRT slot package, preserving one
/// graph handle for the Section 7 evaluator's single per-slot scale.
#[allow(dead_code)]
pub(crate) fn aggregate_refresh_fresh_error(
    compiler: &PowerLutEncodingCompiler,
    base_p: usize,
    material: &RefreshFreshErrorMaterial,
) -> Result<BggEncodingWire, RefreshError> {
    validate_fresh_error_coverage(&material.coverage, material.program_id, &material.outputs)?;
    if material.outputs.iter().any(|output| output.0.layout_id() != material.layout_id) {
        return Err(RefreshError::PrfOutputMismatch);
    }
    let mut ordered = BTreeMap::new();
    for output in &material.outputs {
        let RefreshPrfLabel::FreshError { component, coefficient, digit, .. } = output.0.label()
        else {
            unreachable!("validated fresh-error coverage")
        };
        ordered.insert((component, coefficient, digit), output.encoding().clone());
    }
    aggregate_prf_digits(
        compiler,
        base_p,
        ordered
            .into_iter()
            .map(|((component, coefficient, digit), value)| (digit, coefficient, component, value)),
    )
}

fn refresh_prf_label_digest(label: RefreshPrfLabel) -> [u8; 32] {
    let raw = label.canonical_bytes();
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/sparse-lwr/prf-label/v1");
    digest.update((raw.len() as u64).to_le_bytes());
    digest.update(raw);
    digest.finalize().into()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Protocol declaration used to validate a Section 7 refresh graph.
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

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
/// Collection of refresh declarations attached to one protocol graph.
pub struct PowerLutProtocolDeclaration {
    refreshes: Vec<RefreshDeclaration>,
}

impl PowerLutProtocolDeclaration {
    /// Creates a declaration set and rejects an empty protocol.
    pub fn new(refreshes: Vec<RefreshDeclaration>) -> Result<Self, RefreshError> {
        if refreshes.is_empty() {
            return Err(RefreshError::InvalidLayout);
        }
        Ok(Self { refreshes })
    }

    /// Validates all Section 7 attachments in a built graph.
    pub fn validate(&self, built: &BuiltGraph) -> Result<(), RefreshError> {
        let matching = built
            .derivation_attachments
            .iter()
            .filter(|attachment| {
                attachment.namespace == "mxx-power-lut" && attachment.rule == "section-7-refresh"
            })
            .count();
        if matching != self.refreshes.len() {
            return Err(RefreshError::InvalidLayout);
        }
        for declaration in &self.refreshes {
            declaration.validate_built(built)?;
        }
        Ok(())
    }
}

fn body_ancestry_contains(scope: &GraphScope, wire: WireRef, target: WireRef) -> bool {
    if wire == target {
        return true;
    }
    scope
        .node(wire.node)
        .and_then(|node| scope.arguments(node))
        .into_iter()
        .flatten()
        .any(|argument| body_ancestry_contains(scope, argument, target))
}

fn validate_parallel_refresh(
    loop_spec: &mxx_ir_core::node::ParallelLoop,
    slots: usize,
    argument_count: usize,
    output_count: usize,
) -> Result<(), RefreshError> {
    if loop_spec.count.evaluate(&Default::default()).ok() != Some(BigInt::from(slots)) ||
        loop_spec.minimum_count != 0 ||
        !loop_spec.bindings.is_empty() ||
        loop_spec.input_modes !=
            [
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Broadcast,
            ] ||
        argument_count != 8 ||
        output_count != 1
    {
        return Err(RefreshError::GraphLink("parallel loop schema"));
    }
    Ok(())
}

impl RefreshDeclaration {
    fn validate_parameters(&self) -> Result<(), RefreshError> {
        if self.owner != "mxx-power-lut" ||
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

    /// Checks this declaration against the graph attachments and slot equations.
    pub fn validate_built(&self, built: &BuiltGraph) -> Result<(), RefreshError> {
        self.validate_parameters()?;
        let instance_role = format!("refresh-instance-{}", hex_identity(&self.identity));
        let mut candidates = built.derivation_attachments.iter().filter(|attachment| {
            attachment.namespace == self.owner &&
                attachment.rule == self.plan &&
                attachment.roles.iter().any(|(role, _)| role == &instance_role)
        });
        let attachment = candidates.next().ok_or(RefreshError::InvalidLayout)?;
        if candidates.next().is_some() {
            return Err(RefreshError::GraphLink("role set"));
        }
        self.validate_graph(attachment, &built.graph)
    }

    fn validate_graph(
        &self,
        attachment: &FrozenDerivationAttachment,
        graph: &Graph,
    ) -> Result<(), RefreshError> {
        let role = |name: &str| {
            let mut found = attachment.roles.iter().filter(|(role, _)| role == name);
            let value = found.next().map(|(_, wire)| wire);
            (found.next().is_none()).then_some(value).flatten()
        };
        let required = ["refresh-output", "shared-fresh-error", "refresh-state", "a-prime"];
        if required.iter().any(|name| role(name).is_none()) ||
            role(&format!("refresh-instance-{}", hex_identity(&self.identity))).is_none() ||
            role(&format!("refresh-setup-{}", hex_identity(&self.setup_identity))).is_none() ||
            attachment.roles.len() != 6 + 14 * self.slots.len()
        {
            return Err(RefreshError::GraphLink("role set"));
        }
        let output = role("refresh-output").ok_or(RefreshError::InvalidLayout)?;
        let instance = role(&format!("refresh-instance-{}", hex_identity(&self.identity)))
            .ok_or(RefreshError::InvalidLayout)?;
        let setup = role(&format!("refresh-setup-{}", hex_identity(&self.setup_identity)))
            .ok_or(RefreshError::InvalidLayout)?;
        let a_prime = role("a-prime").ok_or(RefreshError::InvalidLayout)?;
        if output != instance || setup != a_prime {
            return Err(RefreshError::GraphLink("output identity"));
        }
        let root_node = |wire: &ScopedWireRef| {
            if wire.scope != FrozenGraphScopeId::Root {
                return None;
            }
            graph.root_scope().node(wire.wire.node)
        };
        let output_node = root_node(output).ok_or(RefreshError::InvalidLayout)?;
        let NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } =
            output_node.kind()
        else {
            return Err(RefreshError::GraphLink("CRT output kind"));
        };
        if output_node.arguments().len() != self.slots.len() ||
            plaintext_moduli.iter().zip(&self.slots).any(|(modulus, slot)| {
                modulus.evaluate(&Default::default()).ok().as_ref() != Some(&slot.q_t)
            }) ||
            reconstruction_coefficients
                .iter()
                .map(|coefficient| coefficient.evaluate(&Default::default()).ok())
                .ne(self.reconstruction_coefficients.iter().cloned().map(Some))
        {
            return Err(RefreshError::GraphLink("CRT parameters or arity"));
        }
        let Some(mxx_ir_core::types::WireType::Matrix(output_type)) =
            output_node.output_types().first()
        else {
            return Err(RefreshError::GraphLink("output matrix type"));
        };
        if output_type.rows.evaluate(&Default::default()).ok().and_then(|v| v.to_usize()) !=
            Some(self.rows) ||
            output_type.columns.evaluate(&Default::default()).ok().and_then(|v| v.to_usize()) !=
                Some(self.columns)
        {
            return Err(RefreshError::GraphLink("output matrix shape"));
        }
        // Section 7 slot arithmetic is built once in an indexed ParallelLoop.
        // Each declared level must be a static projection of the one loop's
        // sole output.  This is deliberately checked before inspecting the
        // arithmetic so a second loop, a swapped projection, or a nonzero
        // output port cannot be accepted as a look-alike.
        let role_root = |wire: &ScopedWireRef| {
            (wire.scope == FrozenGraphScopeId::Root && wire.wire.port.0 == 0).then_some(wire.wire)
        };
        let get_role = |slot: usize, suffix: &str| {
            role(&format!("slot-{slot}-{suffix}")).ok_or(RefreshError::InvalidLayout)
        };
        let first_level = get_role(0, "level")?;
        let first_level_node = root_node(first_level).ok_or(RefreshError::InvalidLayout)?;
        let NodeKind::FamilyGetStatic { index } = first_level_node.kind() else {
            return Err(RefreshError::GraphLink("parallel slot projection"));
        };
        if first_level_node.arguments().len() != 1 ||
            index.evaluate(&Default::default()).ok() != Some(BigInt::zero())
        {
            return Err(RefreshError::GraphLink("parallel slot projection"));
        }
        let first_family = graph
            .root_scope()
            .wire_ref(&first_level_node.arguments()[0])
            .ok_or(RefreshError::InvalidLayout)?;
        if first_family.port.0 != 0 {
            return Err(RefreshError::GraphLink("parallel family port"));
        }
        let first_family_node =
            graph.root_scope().node(first_family.node).ok_or(RefreshError::InvalidLayout)?;
        let NodeKind::ParallelLoop(loop_spec) = first_family_node.kind() else {
            return Err(RefreshError::GraphLink("parallel slot body"));
        };
        let loop_node_id = first_family.node;
        validate_parallel_refresh(
            loop_spec,
            self.slots.len(),
            first_family_node.arguments().len(),
            first_family_node.output_types().len(),
        )?;
        let loop_args =
            graph.root_scope().arguments(first_family_node).ok_or(RefreshError::InvalidLayout)?;
        let body_id = graph
            .child_scope_id(&FrozenGraphScopeId::Root, loop_node_id)
            .ok_or(RefreshError::GraphLink("parallel body scope"))?;
        let body = graph.scope(&body_id).ok_or(RefreshError::InvalidLayout)?;
        if body.inputs().len() != 8 || body.outputs().len() != 1 || body.outputs()[0].port.0 != 0 {
            return Err(RefreshError::GraphLink("parallel body schema"));
        }
        let body_output = body.outputs()[0];
        let body_output_type = body
            .node(body_output.node)
            .and_then(|node| node.output_types().get(body_output.port.0 as usize))
            .ok_or(RefreshError::InvalidLayout)?;
        let loop_output_type =
            first_family_node.output_types().first().ok_or(RefreshError::InvalidLayout)?;
        let WireType::IndexedFamily { element, count } = loop_output_type else {
            return Err(RefreshError::GraphLink("parallel output family"));
        };
        if count.evaluate(&Default::default()).ok() != Some(BigInt::from(self.slots.len())) ||
            element.as_ref() != body_output_type
        {
            return Err(RefreshError::GraphLink("parallel output alignment"));
        }
        let body_input = |index: usize| body.inputs()[index];
        for input_index in 0..8 {
            let parent_node = graph
                .root_scope()
                .node(loop_args[input_index].node)
                .ok_or(RefreshError::InvalidLayout)?;
            let parent_type = parent_node
                .output_types()
                .get(loop_args[input_index].port.0 as usize)
                .ok_or(RefreshError::InvalidLayout)?;
            let expected_type = if input_index < 7 {
                let WireType::IndexedFamily { element, .. } = parent_type else {
                    return Err(RefreshError::GraphLink("parallel family input type"));
                };
                element.as_ref()
            } else {
                parent_type
            };
            let input = body_input(input_index);
            let body_input_type = body
                .node(input.node)
                .and_then(|node| node.output_types().get(input.port.0 as usize))
                .ok_or(RefreshError::InvalidLayout)?;
            if body_input_type != expected_type {
                return Err(RefreshError::GraphLink("parallel input alignment"));
            }
        }
        let body_args =
            |wire: WireRef, operation: MatrixBinaryOp| -> Result<[WireRef; 2], RefreshError> {
                let node = body.node(wire.node).ok_or(RefreshError::InvalidLayout)?;
                if wire.port.0 != 0 ||
                    node.output_types().len() != 1 ||
                    !matches!(node.kind(), NodeKind::MatrixBinary(found) if *found == operation) ||
                    node.arguments().len() != 2
                {
                    return Err(RefreshError::GraphLink("parallel body arithmetic"));
                }
                Ok(body.arguments(node).ok_or(RefreshError::InvalidLayout)?.try_into().unwrap())
            };
        let validate_large = |source: WireRef, scale: WireRef, result: WireRef| {
            let result_args = body_args(result, MatrixBinaryOp::Multiply)?;
            if result_args[0] != source {
                return Err(RefreshError::GraphLink("parallel large source"));
            }
            let wrapper = body.node(result_args[1].node).ok_or(RefreshError::InvalidLayout)?;
            if result_args[1].port.0 != 0 || wrapper.output_types().len() != 1 {
                return Err(RefreshError::GraphLink("parallel large wrapper"));
            }
            let decomposition_wire = match wrapper.kind() {
                NodeKind::MatrixScale { scalar }
                    if wrapper.arguments().len() == 1 &&
                        scalar.evaluate(&Default::default()).ok() == Some(BigInt::from(1)) =>
                {
                    body.arguments(wrapper).ok_or(RefreshError::InvalidLayout)?[0]
                }
                _ => return Err(RefreshError::GraphLink("parallel large wrapper")),
            };
            let decomposition =
                body.node(decomposition_wire.node).ok_or(RefreshError::InvalidLayout)?;
            let NodeKind::GadgetDecompose { base, small, digit_count } = decomposition.kind()
            else {
                return Err(RefreshError::GraphLink("parallel decomposition"));
            };
            if decomposition_wire.port.0 != 0 || decomposition.arguments().len() != 1 || *small {
                return Err(RefreshError::GraphLink("parallel decomposition"));
            }
            let decomposition_arg =
                body.arguments(decomposition).ok_or(RefreshError::InvalidLayout)?[0];
            let scale_product = body_args(decomposition_arg, MatrixBinaryOp::Multiply)?;
            if scale_product[1] != scale {
                return Err(RefreshError::GraphLink("parallel scale ancestry"));
            }
            let gadget = body.node(scale_product[0].node).ok_or(RefreshError::InvalidLayout)?;
            let NodeKind::ConstantMatrix {
                value:
                    mxx_ir_core::node::ConstantMatrix::Gadget { base: gadget_base, small: gadget_small },
                ..
            } = gadget.kind()
            else {
                return Err(RefreshError::GraphLink("parallel gadget"));
            };
            if scale_product[0].port.0 != 0 ||
                gadget.arguments().len() != 0 ||
                *gadget_small ||
                gadget_base != base
            {
                return Err(RefreshError::GraphLink("parallel gadget relation"));
            }
            Ok((base.clone(), digit_count.clone()))
        };
        let mut large_parameters = None;
        for slot in 0..self.slots.len() {
            let level = get_role(slot, "level")?;
            let level_node = root_node(level).ok_or(RefreshError::InvalidLayout)?;
            let NodeKind::FamilyGetStatic { index } = level_node.kind() else {
                return Err(RefreshError::GraphLink("parallel slot projection"));
            };
            if level_node.arguments().len() != 1 ||
                level.wire.port.0 != 0 ||
                index.evaluate(&Default::default()).ok() != Some(BigInt::from(slot)) ||
                graph.root_scope().wire_ref(&level_node.arguments()[0]) != Some(first_family) ||
                graph.root_scope().wire_ref(&output_node.arguments()[slot]) != Some(level.wire)
            {
                return Err(RefreshError::GraphLink("parallel slot projection"));
            }
            for suffix in [
                "scale-target",
                "mask",
                "scaled-fresh-error",
                "decoder",
                "decoder-public",
                "target",
            ] {
                if role_root(get_role(slot, suffix)?).is_none() {
                    return Err(RefreshError::GraphLink("parallel role scope"));
                }
            }
            let expected = [
                get_role(slot, "scale-target")?,
                get_role(slot, "mask")?,
                get_role(slot, "scaled-fresh-error")?,
                get_role(slot, "decoder")?,
                get_role(slot, "decoder-public")?,
            ];
            let family_indices = [0usize, 1, 3, 5, 6];
            for (family_index, expected_role) in family_indices.into_iter().zip(expected) {
                let family_wire = loop_args[family_index];
                let family_node =
                    graph.root_scope().node(family_wire.node).ok_or(RefreshError::InvalidLayout)?;
                let NodeKind::FamilyPack { count } = family_node.kind() else {
                    return Err(RefreshError::GraphLink("parallel family pack"));
                };
                if family_wire.port.0 != 0 ||
                    family_node.arguments().len() != self.slots.len() ||
                    family_node.output_types().len() != 1 ||
                    count.evaluate(&Default::default()).ok() !=
                        Some(BigInt::from(self.slots.len())) ||
                    graph.root_scope().wire_ref(&family_node.arguments()[slot]) !=
                        role_root(expected_role)
                {
                    return Err(RefreshError::GraphLink("parallel family order"));
                }
            }
            for family_index in [2usize, 4] {
                let family_node = graph
                    .root_scope()
                    .node(loop_args[family_index].node)
                    .ok_or(RefreshError::InvalidLayout)?;
                let NodeKind::FamilyPack { count } = family_node.kind() else {
                    return Err(RefreshError::GraphLink("parallel family pack"));
                };
                if loop_args[family_index].port.0 != 0 ||
                    family_node.arguments().len() != self.slots.len() ||
                    family_node.output_types().len() != 1 ||
                    count.evaluate(&Default::default()).ok() !=
                        Some(BigInt::from(self.slots.len()))
                {
                    return Err(RefreshError::GraphLink("parallel family shape"));
                }
            }
            if loop_args[7] !=
                role_root(role("refresh-state").ok_or(RefreshError::InvalidLayout)?)
                    .ok_or(RefreshError::GraphLink("parallel state"))?
            {
                return Err(RefreshError::GraphLink("parallel state"));
            }
            let vector_types = [1usize, 3, 5]
                .map(|index| {
                    body.node(body_input(index).node).and_then(|node| node.output_types().first())
                })
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or(RefreshError::InvalidLayout)?;
            let public_types = [2usize, 4, 6]
                .map(|index| {
                    body.node(body_input(index).node).and_then(|node| node.output_types().first())
                })
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or(RefreshError::InvalidLayout)?;
            if vector_types.iter().any(|ty| !matches!(ty, WireType::Matrix(_))) ||
                public_types.iter().any(|ty| !matches!(ty, WireType::Matrix(_))) ||
                vector_types[1] != vector_types[0] ||
                vector_types[2] != vector_types[0] ||
                public_types[1] != public_types[0] ||
                public_types[2] != public_types[0]
            {
                return Err(RefreshError::GraphLink("parallel input shapes"));
            }
            if body_ancestry_contains(body, body_output, body_input(2)) ||
                body_ancestry_contains(body, body_output, body_input(4)) ||
                body_ancestry_contains(body, body_output, body_input(6))
            {
                return Err(RefreshError::GraphLink("parallel unused inputs"));
            }
            let subtract = body_args(body_output, MatrixBinaryOp::Subtract)?;
            if subtract[1] != body_input(5) {
                return Err(RefreshError::GraphLink("parallel decoder order"));
            }
            let add_mask = body_args(subtract[0], MatrixBinaryOp::Add)?;
            let add_state = body_args(add_mask[0], MatrixBinaryOp::Add)?;
            if add_state[1] != body_input(1) {
                return Err(RefreshError::GraphLink("parallel mask order"));
            }
            let state_large = validate_large(body_input(7), body_input(0), add_state[0])?;
            let fresh_large = validate_large(body_input(3), body_input(0), add_mask[1])?;
            if state_large != fresh_large {
                return Err(RefreshError::GraphLink("parallel large parameters"));
            }
            if let Some(previous) = &large_parameters {
                if previous != &state_large {
                    return Err(RefreshError::GraphLink("parallel producer"));
                }
            } else {
                large_parameters = Some(state_large);
            }
        }
        if role("refresh-state").ok_or(RefreshError::InvalidLayout)?.scope !=
            FrozenGraphScopeId::Root
        {
            return Err(RefreshError::GraphLink("parallel state scope"));
        }
        let binary_args = |wire: &ScopedWireRef,
                           operation: MatrixBinaryOp|
         -> Result<[ScopedWireRef; 2], RefreshError> {
            let node = root_node(wire).ok_or(RefreshError::InvalidLayout)?;
            if !matches!(node.kind(), NodeKind::MatrixBinary(found) if *found == operation) ||
                node.arguments().len() != 2
            {
                return Err(RefreshError::GraphLink("slot add/subtract chain"));
            }
            let left = graph
                .root_scope()
                .wire_ref(&node.arguments()[0])
                .ok_or(RefreshError::InvalidLayout)?;
            let right = graph
                .root_scope()
                .wire_ref(&node.arguments()[1])
                .ok_or(RefreshError::InvalidLayout)?;
            Ok([
                ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: left },
                ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: right },
            ])
        };
        for slot in 0..self.slots.len() {
            let get = |suffix: &str| {
                role(&format!("slot-{slot}-{suffix}")).ok_or(RefreshError::InvalidLayout)
            };
            let scale_target = get("scale-target")?;
            let decoder_public = get("decoder-public")?;
            let a_sum = get("a-sum")?;
            let target = get("target")?;
            let b = get("b")?;
            let k = get("k")?;
            let scale_node = root_node(scale_target).ok_or(RefreshError::InvalidLayout)?;
            let NodeKind::ConstantMatrix {
                value: mxx_ir_core::node::ConstantMatrix::Polynomial { coefficients },
                ..
            } = scale_node.kind()
            else {
                return Err(RefreshError::InvalidLayout);
            };
            if coefficients.len() != 1 ||
                coefficients[0].evaluate(&Default::default()).ok() !=
                    Some(&self.slots[slot].q / &self.slots[slot].q_t)
            {
                return Err(RefreshError::GraphLink("scale constant"));
            }
            let target_args = binary_args(target, MatrixBinaryOp::Subtract)?;
            if target_args[0] != *a_sum {
                return Err(RefreshError::GraphLink("decoder target minuend"));
            }
            let scaled_a_prime_args = binary_args(&target_args[1], MatrixBinaryOp::Multiply)?;
            if scaled_a_prime_args != [scale_target.clone(), a_prime.clone()] {
                return Err(RefreshError::GraphLink("decoder target scale"));
            }
            let b_k_product = get("b-k-product")?;
            let b_k = binary_args(b_k_product, MatrixBinaryOp::Multiply)?;
            if b_k != [b.clone(), k.clone()] || decoder_public != target {
                return Err(RefreshError::GraphLink("decoder preimage equation"));
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
    /// matrix `K`, retaining their graph product as the trusted target.
    pub(crate) fn with_equation(b: Mat, k: Mat) -> Self {
        // Construct the issuer target here so callers cannot attach an
        // unrelated matrix while claiming it is B*K.
        let target = b.clone() * k.clone();
        Self { equation: Some(RefreshAnchorEquation { target }) }
    }

    #[cfg(test)]
    fn equation(&self) -> Option<&RefreshAnchorEquation> {
        self.equation.as_ref()
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
        crate::ensure_ciphertext_only(&encoding).map_err(RefreshError::Power)?;
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

/// Typed per-slot package for the Section 7 refresh equation.
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
    fresh_error_source_handle: mxx_ir_core::ValueHandle,
    decoder: RefreshDecoderPreimage,
    target_public_matrix: Mat,
    b: Mat,
    k: Mat,
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
        target_public_matrix: Mat,
        anchor: &RefreshAnchor,
        b: Mat,
        k: Mat,
    ) -> Result<Self, RefreshError> {
        crate::ensure_ciphertext_only(state).map_err(RefreshError::Power)?;
        crate::ensure_ciphertext_only(&mask).map_err(RefreshError::Power)?;
        crate::ensure_ciphertext_only(&fresh_error_source).map_err(RefreshError::Power)?;
        crate::ensure_ciphertext_only(&decoder.encoding).map_err(RefreshError::Power)?;
        decoder.validate(anchor, &target_public_matrix)?;
        let anchor_product_handle = anchor
            .equation
            .as_ref()
            .ok_or(RefreshError::MissingAnchorEquation)?
            .target
            .value_handle()
            .clone();
        let fresh_error_source_handle = fresh_error_source.vector.value_handle().clone();
        Ok(Self {
            slot,
            state_handle: state.vector.value_handle().clone(),
            a_prime_handle: a_prime.value_handle().clone(),
            a_sum_t,
            scale_target,
            mask,
            fresh_error_source,
            fresh_error_source_handle,
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

/// Artifacts for one canonical CRT slot. The sparse-LWR PRF supplies the mask
/// wire and the one shared fresh-error wire; this setup layer derives the slot
/// scale and decoder target from those wires and the refresh config.
#[derive(Clone)]
#[allow(dead_code)]
#[cfg(test)]
pub(crate) struct RefreshSetupSlotArtifacts {
    /// Mask encoding for this CRT slot.
    pub mask: RefreshMaskPrfOutput,
    /// Fresh-error encoding shared by every CRT slot.
    pub fresh_error_source: RefreshFreshErrorPrfOutput,
    /// Decoder preimage bound to the slot's refresh anchor.
    pub decoder: BggEncodingWire,
    /// Public `B` matrix in the slot's anchor equation.
    pub b: Mat,
    /// Public `K` matrix in the slot's anchor equation.
    pub k: Mat,
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
        for (slot, package) in packages.iter().enumerate() {
            if package.slot != slot ||
                package.state_handle != *state.vector.value_handle() ||
                package.a_prime_handle != *a_prime.value_handle() ||
                package.mask.vector.matrix_type() != state.vector.matrix_type() ||
                package.fresh_error_source.vector.matrix_type() != state.vector.matrix_type()
            {
                return Err(if package.slot != slot {
                    RefreshError::SlotOrderMismatch
                } else {
                    RefreshError::InvalidLayout
                });
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"mxx.power-lut.refresh-setup");
        hash.update(format!("{:?}", state.vector.value_handle()).as_bytes());
        hash.update(format!("{:?}", a_prime.value_handle()).as_bytes());
        for package in &packages {
            hash.update(package.slot.to_le_bytes());
            hash.update(format!("{:?}", package.a_sum_t.value_handle()).as_bytes());
            hash.update(format!("{:?}", package.target_public_matrix.value_handle()).as_bytes());
            hash.update(format!("{:?}", package.b.value_handle()).as_bytes());
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
    /// The imported decoder bases and preimages are deliberately plain wires
    /// at this boundary.  This helper is the only crate-internal adapter that
    /// turns them back into the typed Section-7 packages: each target is
    /// recomputed from the imported state/mask/fresh material and the exact
    /// decoder is rebuilt as `base.vector * K`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_imported_wires(
        &self,
        compiler: &PowerLutEncodingCompiler,
        state: BggEncodingWire,
        a_prime: Mat,
        public_b: Mat,
        fresh: BggEncodingWire,
        masks: Vec<BggEncodingWire>,
        decoder_bases: Vec<BggEncodingWire>,
        preimages: Vec<Mat>,
    ) -> Result<RefreshSetupManifest, RefreshError> {
        self.validate_layout()?;
        crate::ensure_ciphertext_only(&state).map_err(RefreshError::Power)?;
        crate::ensure_ciphertext_only(&fresh).map_err(RefreshError::Power)?;
        if masks.len() != self.crt_plaintext_moduli.len() ||
            decoder_bases.len() != masks.len() ||
            preimages.len() != masks.len() ||
            !same_matrix_type(a_prime.matrix_type(), state.pubkey.matrix.matrix_type())
        {
            return Err(RefreshError::InvalidLayout);
        }
        let base_handle = public_b.value_handle().clone();
        let mut packages = Vec::with_capacity(masks.len());
        for (slot, ((mask, base), k)) in
            masks.into_iter().zip(decoder_bases.into_iter()).zip(preimages.into_iter()).enumerate()
        {
            crate::ensure_ciphertext_only(&mask).map_err(RefreshError::Power)?;
            crate::ensure_ciphertext_only(&base).map_err(RefreshError::Power)?;
            if base.pubkey.reveal_plaintext || base.pubkey.matrix.value_handle() != &base_handle {
                return Err(RefreshError::TargetMismatch);
            }
            let scale = mxx_dsl::Ring::new(
                state.vector.matrix_type().modulus.clone(),
                state.vector.matrix_type().ring_dimension.clone(),
            )
            .polynomial([self.scale_expression(slot)?]);
            let scaled_state = compiler.bgg.large_scalar_mul(&state, &scale);
            let scaled_fresh = compiler.bgg.large_scalar_mul(&fresh, &scale);
            let combined =
                compiler.bgg.add(&compiler.bgg.add(&scaled_state, &mask)?, &scaled_fresh)?;
            let a_sum_t = combined.pubkey.matrix;
            let target = a_sum_t.clone() - scale.clone() * a_prime.clone();
            let decoder = BggEncodingWire {
                vector: base.vector * k.clone(),
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
                fresh.clone(),
                decoder,
                target,
                &anchor,
                public_b.clone(),
                k,
            )?);
        }
        let fresh_handle =
            packages.first().ok_or(RefreshError::InvalidLayout)?.fresh_error_source_handle.clone();
        if packages.iter().any(|package| package.fresh_error_source_handle != fresh_handle) {
            return Err(RefreshError::FreshErrorMismatch);
        }
        RefreshSetupManifest::bind(state, a_prime, packages)
    }

    /// Validates and seals setup-time artifacts into an opaque refresh
    /// manifest. Evaluation subsequently consumes only this manifest.
    #[allow(dead_code)]
    #[cfg(test)]
    pub(crate) fn bind_setup(
        &self,
        compiler: &PowerLutEncodingCompiler,
        state: BggEncodingWire,
        a_prime: Mat,
        slots: Vec<RefreshSetupSlotArtifacts>,
    ) -> Result<RefreshSetupManifest, RefreshError> {
        self.validate_layout()?;
        crate::ensure_ciphertext_only(&state).map_err(RefreshError::Power)?;
        if slots.len() != self.crt_plaintext_moduli.len() ||
            !same_matrix_type(a_prime.matrix_type(), state.pubkey.matrix.matrix_type())
        {
            return Err(RefreshError::InvalidLayout);
        }
        let packages = slots
            .into_iter()
            .enumerate()
            .map(|(slot, artifacts)| {
                crate::ensure_ciphertext_only(artifacts.mask.encoding())
                    .map_err(RefreshError::Power)?;
                crate::ensure_ciphertext_only(artifacts.fresh_error_source.encoding())
                    .map_err(RefreshError::Power)?;
                crate::ensure_ciphertext_only(&artifacts.decoder).map_err(RefreshError::Power)?;
                if !same_matrix_type(
                    artifacts.mask.encoding().vector.matrix_type(),
                    state.vector.matrix_type(),
                ) || !same_matrix_type(
                    artifacts.fresh_error_source.encoding().vector.matrix_type(),
                    state.vector.matrix_type(),
                ) || !same_matrix_type(
                    artifacts.decoder.vector.matrix_type(),
                    state.vector.matrix_type(),
                ) || !same_matrix_type(
                    artifacts.mask.encoding().pubkey.matrix.matrix_type(),
                    a_prime.matrix_type(),
                ) || !same_matrix_type(
                    artifacts.fresh_error_source.encoding().pubkey.matrix.matrix_type(),
                    a_prime.matrix_type(),
                ) {
                    return Err(RefreshError::InvalidLayout);
                }
                let ring = mxx_dsl::Ring::new(
                    state.vector.matrix_type().modulus.clone(),
                    state.vector.matrix_type().ring_dimension.clone(),
                );
                let scale_target = ring.polynomial([self.scale_expression(slot)?]);
                let scaled_state = compiler.bgg.large_scalar_mul(&state, &scale_target);
                let scaled_fresh = compiler
                    .bgg
                    .large_scalar_mul(artifacts.fresh_error_source.encoding(), &scale_target);
                let masked = compiler.bgg.add(&scaled_state, artifacts.mask.encoding())?;
                let combined = compiler.bgg.add(&masked, &scaled_fresh)?;
                let a_sum_t = combined.pubkey.matrix;
                let target_public_matrix = a_sum_t.clone() - scale_target.clone() * a_prime.clone();
                if !same_matrix_type(
                    (artifacts.b.clone() * artifacts.k.clone()).matrix_type(),
                    target_public_matrix.matrix_type(),
                ) {
                    return Err(RefreshError::InvalidLayout);
                }
                let decoder = BggEncodingWire {
                    vector: artifacts.decoder.vector.clone(),
                    pubkey: BggPublicKeyWire {
                        matrix: target_public_matrix.clone(),
                        reveal_plaintext: false,
                    },
                    plaintext: None,
                };
                let anchor = RefreshAnchor::with_equation(artifacts.b.clone(), artifacts.k.clone());
                let decoder =
                    RefreshDecoderPreimage::bind(&anchor, decoder, &target_public_matrix)?;
                RefreshScalarPackage::new(
                    slot,
                    &state,
                    &a_prime,
                    a_sum_t,
                    scale_target,
                    artifacts.mask.encoding().clone(),
                    artifacts.fresh_error_source.encoding().clone(),
                    decoder,
                    target_public_matrix,
                    &anchor,
                    artifacts.b,
                    artifacts.k,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shared_fresh = packages[0].fresh_error_source_handle.clone();
        if packages.iter().any(|package| package.fresh_error_source_handle != shared_fresh) {
            return Err(RefreshError::FreshErrorMismatch);
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
            owner: "mxx-power-lut".into(),
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
        if self.crt_plaintext_moduli.is_empty() ||
            self.crt_plaintext_moduli.len() != self.reconstruction_coefficients.len()
        {
            return Err(RefreshError::InvalidLayout);
        }
        let q = self
            .full_modulus
            .evaluate(&Default::default())
            .map_err(|_| RefreshError::InvalidLayout)?;
        if q <= BigInt::from(0) ||
            self.crt_plaintext_moduli.iter().any(|qt| {
                qt.evaluate(&Default::default())
                    .map(|value| value <= BigInt::from(0) || &q % value != BigInt::from(0))
                    .unwrap_or(true)
            })
        {
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
        compiler: &PowerLutEncodingCompiler,
        setup: &RefreshSetupManifest,
    ) -> Result<RefreshResult, RefreshError> {
        self.validate_layout()?;
        let state = &setup.state;
        let packages = &setup.packages;
        if packages.len() != self.crt_plaintext_moduli.len() {
            return Err(RefreshError::InvalidLayout);
        }
        let _ = packages.first().ok_or(RefreshError::InvalidLayout)?;
        let fresh_handle = packages[0].fresh_error_source_handle.clone();
        // Package fields are exposed as indexed families once, then one
        // structural ParallelLoop performs the complete per-slot equation.
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
                let scaled_fresh = compiler.bgg.large_scalar_mul(&fresh, &scale);
                let combined = compiler.bgg.add(&scaled, &mask).expect("validated refresh add");
                let combined_full =
                    compiler.bgg.add(&combined, &scaled_fresh).expect("validated refresh add");
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
            if package.fresh_error_source_handle != fresh_handle {
                return Err(RefreshError::FreshErrorMismatch);
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
                    (package.b.clone() * package.k.clone()).value_handle().clone(),
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
        let mut roles = vec![
            ("refresh-output".to_owned(), vector.value_handle().clone()),
            ("shared-fresh-error".to_owned(), fresh_handle),
            ("refresh-state".to_owned(), state.vector.value_handle().clone()),
            ("a-prime".to_owned(), setup.a_prime.value_handle().clone()),
            (
                format!("refresh-instance-{}", hex_identity(&declaration.identity)),
                vector.value_handle().clone(),
            ),
            (
                format!("refresh-setup-{}", hex_identity(&setup.identity)),
                setup.a_prime.value_handle().clone(),
            ),
        ];
        roles.extend(slot_roles);
        let vector = vector
            .derivation_attachment("mxx-power-lut", "section-7-refresh", roles)
            .map_err(|_| RefreshError::InvalidLayout)?;
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

/// Routes one PRF base-`p` digit to one coefficient and public-key component.
///
/// In the Section 7 shorthand, `p^c X^j u_2 delta_k^T` means the following
/// concrete matrix product: `p^c` is the digit scale, `X^j` is the negacyclic
/// rotation by `coefficient`, `u_2` is the unit column selecting the final
/// secret coordinate, and `delta_k^T` is the unit row selecting `component`.
/// The implementation multiplies this route into `value`, preserving the
/// BGG encoding shape. CRT scaling is deliberately not part of this route;
/// Section 7 applies the exact `q / q_t` factor once to the aggregated fresh
/// error in its evaluator.
pub fn route_prf_digit(
    compiler: &PowerLutEncodingCompiler,
    value: &BggEncodingWire,
    base_p: usize,
    digit: usize,
    coefficient: usize,
    component: usize,
) -> Result<BggEncodingWire, RefreshError> {
    crate::ensure_ciphertext_only(value).map_err(RefreshError::Power)?;
    let route = prf_route_matrix(value, base_p, digit, coefficient, component)?;
    Ok(compiler.bgg.matrix_mul(value, &route))
}

fn prf_route_matrix(
    value: &BggEncodingWire,
    base_p: usize,
    digit: usize,
    coefficient: usize,
    component: usize,
) -> Result<Mat, RefreshError> {
    crate::ensure_ciphertext_only(value).map_err(RefreshError::Power)?;
    let vector_type = value.vector.matrix_type();
    let public_key_type = value.pubkey.matrix.matrix_type();
    let columns = public_key_type
        .columns
        .evaluate(&Default::default())
        .map_err(|_| RefreshError::InvalidLayout)?
        .to_usize()
        .ok_or(RefreshError::InvalidLayout)?;
    let secret_dimension = public_key_type
        .rows
        .evaluate(&Default::default())
        .map_err(|_| RefreshError::InvalidLayout)?
        .to_usize()
        .ok_or(RefreshError::InvalidLayout)?;
    let ring_dimension = vector_type
        .ring_dimension
        .evaluate(&Default::default())
        .map_err(|_| RefreshError::InvalidLayout)?
        .to_usize()
        .ok_or(RefreshError::InvalidLayout)?;
    if base_p < 2 || component >= columns || coefficient >= ring_dimension || secret_dimension < 2 {
        return Err(RefreshError::InvalidLayout);
    }
    let ring = mxx_dsl::Ring::new(vector_type.modulus.clone(), vector_type.ring_dimension.clone());
    let digit = u32::try_from(digit).map_err(|_| RefreshError::InvalidLayout)?;
    let scale = BigInt::from(base_p).pow(digit);
    let scalar = ring.constant(
        (1, 1),
        mxx_ir_core::node::ConstantMatrix::Rotation { exponent: coefficient.into() },
    ) * ring.polynomial([scale.into()]);
    let route = scalar *
        ring.constant(
            (secret_dimension, 1),
            mxx_ir_core::node::ConstantMatrix::UnitColumn { index: (secret_dimension - 1).into() },
        ) *
        ring.constant(
            (1, columns),
            mxx_ir_core::node::ConstantMatrix::UnitRow { index: component.into() },
        );
    // Keep the route as a matrix so a family of labels can be routed in one
    // structural loop and then reduced through a balanced tree.
    Ok(route)
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

fn hex_identity(identity: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in identity {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0xf) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        encoding::{AutomorphismHelper, EncodingSelectorFamily, PowerLutEncodingSampler},
        pbc::PbcSelectorArtifactNames,
    };
    use serial_test::serial;

    #[test]
    fn refresh_prf_label_index_is_canonical_and_round_trips() {
        let index = RefreshPrfLabelIndex::new([3; 32], 2, 2, 2, 3).unwrap();
        assert_eq!(index.len(), (2 + 1) * 2 * 2 * 3);
        let expected = [
            RefreshPrfLabel::Mask {
                refresh_id: [3; 32],
                slot: 0,
                component: 0,
                coefficient: 0,
                digit: 0,
            },
            RefreshPrfLabel::Mask {
                refresh_id: [3; 32],
                slot: 0,
                component: 0,
                coefficient: 0,
                digit: 1,
            },
            RefreshPrfLabel::Mask {
                refresh_id: [3; 32],
                slot: 0,
                component: 0,
                coefficient: 0,
                digit: 2,
            },
            RefreshPrfLabel::Mask {
                refresh_id: [3; 32],
                slot: 1,
                component: 1,
                coefficient: 1,
                digit: 2,
            },
            RefreshPrfLabel::FreshError {
                refresh_id: [3; 32],
                component: 0,
                coefficient: 0,
                digit: 0,
            },
        ];
        assert_eq!(index.label(0), Some(expected[0]));
        assert_eq!(index.label(1), Some(expected[1]));
        assert_eq!(index.label(2), Some(expected[2]));
        assert_eq!(index.label(23), Some(expected[3]));
        assert_eq!(index.label(24), Some(expected[4]));
        for flat in 0..index.len() {
            let label = index.label(flat).unwrap();
            assert_eq!(index.index_of(label), Some(flat));
        }
        assert!(index.label(index.len()).is_none());
    }

    #[test]
    fn aggregate_prf_digits_has_one_structural_routing_loop() {
        let ring = mxx_dsl::Ring::new(17, 2);
        let compiler = PowerLutEncodingCompiler::from_public_key(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 2.into(),
        });
        let values = (0..4)
            .map(|index| {
                let value = BggEncodingWire {
                    vector: ring.input(format!("aggregate-vector-{index}"), (1, 4)),
                    pubkey: BggPublicKeyWire {
                        matrix: ring.input(format!("aggregate-public-{index}"), (2, 4)),
                        reveal_plaintext: false,
                    },
                    plaintext: None,
                };
                (index % 2, 0, index % 2, value)
            })
            .collect::<Vec<_>>();
        let aggregate = aggregate_prf_digits(&compiler, 2, values).unwrap();
        let built = mxx_dsl::DslContext::new("refresh-aggregate-structural")
            .output("aggregate", aggregate.vector)
            .unwrap()
            .build()
            .unwrap();
        let routing_loops = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                NodeKind::ParallelLoop(spec)
                    if spec.input_modes ==
                        [LoopInputMode::Zip, LoopInputMode::Zip, LoopInputMode::Zip] =>
                {
                    Some(spec)
                }
                _ => None,
            })
            .count();
        assert_eq!(routing_loops, 1);
        assert!(built.graph.root_scope().nodes().len() < 64);
    }

    fn config() -> RefreshCompiler {
        RefreshCompiler {
            full_modulus: 323.into(),
            crt_plaintext_moduli: vec![17.into(), 19.into()],
            reconstruction_coefficients: vec![1.into(), 1.into()],
        }
    }

    struct RealPbcTestContext {
        layout: crate::pbc::PbcPublicLayout,
        public_vector: crate::pbc::PbcPublicVectorFamilyBinding,
        selectors: EncodingSelectorFamily,
        values: Family<Mat>,
        helpers: Vec<AutomorphismHelper>,
        selector_graph: mxx_dsl::BuiltGraph,
        selector_production: mxx_ir_core::artifact::ProductionId,
        selector_bit_input_name: String,
        selector_bit_values: Vec<u8>,
    }

    impl RealPbcTestContext {
        fn new(
            compiler: &PowerLutEncodingCompiler,
            input: &BggEncodingWire,
            seed: [u8; 32],
            key_instance_id: [u8; 32],
        ) -> Self {
            let ring = compiler.bgg.public_key.ring.clone();
            let pbc_parameters = crate::pbc::PbcParameters::custom(1, 1, 2, 2, 4, None);
            let generated = crate::pbc::generate_key_layout(
                &pbc_parameters,
                crate::pbc::PbcRootSeed(seed),
                &[0],
            )
            .unwrap();
            let layout = generated.public_layout.clone();
            assert_eq!(layout.bucket_width, 2);
            let routed =
                crate::pbc::PbcEncodedPublicVector::route_usize(&layout, &[3], 17).unwrap();
            let public_vector =
                crate::pbc::PbcPublicVectorFamilyBinding::from_encoded(&layout, &routed).unwrap();
            let count = public_vector.family_count;
            let columns = input
                .pubkey
                .matrix
                .matrix_type()
                .columns
                .evaluate(&mxx_ir_core::ParamEnv::default())
                .unwrap()
                .to_usize()
                .unwrap();
            let rows = input
                .pubkey
                .matrix
                .matrix_type()
                .rows
                .evaluate(&mxx_ir_core::ParamEnv::default())
                .unwrap()
                .to_usize()
                .unwrap();
            let digit_count = compiler
                .bgg
                .public_key
                .digit_count
                .evaluate(&mxx_ir_core::ParamEnv::default())
                .unwrap()
                .to_usize()
                .unwrap();
            let packed_columns = columns * digit_count;
            let sampler = PowerLutEncodingSampler {
                layout: mxx_bgg::BggSamplerLayout {
                    modulus: input.pubkey.matrix.matrix_type().modulus.clone(),
                    ring_dimension: input.pubkey.matrix.matrix_type().ring_dimension.clone(),
                    secret_dimension: rows,
                    digit_count,
                    gadget_base: compiler.bgg.public_key.base.clone(),
                },
                gaussian_sigma: None,
                gaussian_max_coefficient_bound: None,
            };
            let secret = ring.zero((1, rows));
            let selector_key = ring.bytes_input("real-pbc-selector-key", 32);
            let selector_bits = crate::pbc::PbcTrustedSelectorBits::from_schedule(
                &generated,
                &ring,
                key_instance_id,
            )
            .unwrap();
            debug_assert_eq!(selector_bits.runtime_bits().len(), count);
            let structural_families = crate::pbc::build_structural_selector_families(
                &sampler,
                selector_bits.family().clone(),
                secret.clone(),
                secret,
                selector_key,
                &layout,
                key_instance_id,
            )
            .unwrap();
            let names = PbcSelectorArtifactNames::canonicalize_schema(
                &layout,
                key_instance_id,
                rows,
                columns,
            )
            .unwrap();
            let artifacts =
                crate::pbc::PbcSelectorArtifacts::from_structural(&layout, key_instance_id, names)
                    .unwrap();
            let selector_graph = artifacts
                .add_structural_family_outputs(
                    mxx_dsl::DslContext::new("real-pbc-selector-producer"),
                    &layout,
                    structural_families,
                )
                .unwrap()
                .public_family_output(
                    "real-pbc-values",
                    Family::pack(
                        public_vector
                            .values_usize()
                            .unwrap()
                            .into_iter()
                            .map(|value| ring.polynomial([value.into()]))
                            .collect(),
                    )
                    .unwrap(),
                )
                .unwrap()
                .build()
                .unwrap();
            let selector_production = mxx_ir_core::artifact::ProductionId {
                spec_hash: mxx_ir_core::encoding::spec_hash(
                    &selector_graph.graph,
                    &Default::default(),
                )
                .unwrap(),
                execution_nonce: [55; 32],
            };
            let production_id = selector_production.clone();
            let companions = (0..rows * columns)
                .map(|column| {
                    (
                        ring.family_artifact_input(
                            production_id.clone(),
                            crate::pbc::selector_family_artifact_name(
                                &artifacts,
                                &format!("vector-{column}"),
                            ),
                            count,
                            (1, packed_columns),
                            mxx_ir_core::artifact::ArtifactConfidentiality::Private,
                        ),
                        ring.family_artifact_input(
                            production_id.clone(),
                            crate::pbc::selector_family_artifact_name(
                                &artifacts,
                                &format!("public-{column}"),
                            ),
                            count,
                            (rows, packed_columns),
                            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
                        ),
                    )
                })
                .collect();
            let selectors = EncodingSelectorFamily::new(
                ring.family_artifact_input(
                    production_id.clone(),
                    crate::pbc::selector_family_artifact_name(&artifacts, "gsw"),
                    count,
                    (rows, columns),
                    mxx_ir_core::artifact::ArtifactConfidentiality::Private,
                ),
                companions,
            )
            .unwrap();
            let values = ring.family_artifact_input(
                production_id,
                "real-pbc-values",
                count,
                (1, 1),
                mxx_ir_core::artifact::ArtifactConfidentiality::Public,
            );
            let helpers = sampler
                .sample_automorphism_helpers(
                    ring.zero((1, rows)),
                    ring.bytes_input("real-pbc-helper-key", 32),
                    b"real-pbc-helpers".to_vec(),
                    4,
                )
                .unwrap();
            // The production fixture uses W = 4, so the mandatory
            // coefficient sieve requires log2(W) = 2 helper rounds.
            assert_eq!(helpers.len(), 2);
            Self {
                layout,
                public_vector,
                selectors,
                values,
                helpers,
                selector_graph,
                selector_production,
                selector_bit_input_name: selector_bits.input_name().to_owned(),
                selector_bit_values: selector_bits.runtime_bits().to_vec(),
            }
        }

        fn lower(
            &self,
            compiler: &PowerLutEncodingCompiler,
            input: BggEncodingWire,
            body: &crate::prf::SparseLwrPrfProgram,
            label: &[u8],
        ) -> crate::prf::PbcSparseLwrEncodingOutput {
            body.compile_pbc_encoding(
                compiler,
                input,
                &self.layout,
                &self.public_vector,
                self.selectors.clone(),
                self.values.clone(),
                &self.helpers,
                label,
            )
            .unwrap()
        }
    }

    fn real_mask(
        context: &RealPbcTestContext,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        refresh_id: [u8; 32],
        slot: usize,
        component: usize,
        coefficient: usize,
        digit: usize,
    ) -> RefreshMaskPrfOutput {
        let label = RefreshPrfLabel::Mask { refresh_id, slot, component, coefficient, digit };
        let body = crate::prf::test_sparse_lwr_program(context.layout.bucket_width, 4).unwrap();
        let output = context.lower(compiler, input, &body, &label.canonical_bytes());
        RefreshMaskPrfOutput::from_pbc_evaluation(
            output,
            refresh_id,
            slot,
            component,
            coefficient,
            digit,
        )
        .unwrap()
    }

    fn real_fresh_error(
        context: &RealPbcTestContext,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        refresh_id: [u8; 32],
        component: usize,
        coefficient: usize,
        digit: usize,
    ) -> RefreshFreshErrorPrfOutput {
        let label = RefreshPrfLabel::FreshError { refresh_id, component, coefficient, digit };
        let body = crate::prf::test_sparse_lwr_program(context.layout.bucket_width, 4).unwrap();
        let output = context.lower(compiler, input, &body, &label.canonical_bytes());
        RefreshFreshErrorPrfOutput::from_pbc_evaluation(
            output,
            refresh_id,
            component,
            coefficient,
            digit,
        )
        .unwrap()
    }

    struct StructuralRefreshFixture {
        declaration: RefreshDeclaration,
        graph: Graph,
        attachment: FrozenDerivationAttachment,
    }

    fn structural_refresh_fixture() -> StructuralRefreshFixture {
        use mxx_primitives::poly::PolyParams;

        let parameters = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(4, 2, 17, 16);
        let q = BigInt::from(parameters.modulus().as_ref().clone());
        let moduli = parameters.to_crt().0;
        let ring = mxx_dsl::Ring::new(q.clone(), 4);
        let compiler =
            crate::PowerLutEncodingCompiler::from_public_key(mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 2.into(),
                digit_count: 4.into(),
            });
        let encoding = |name: &str| BggEncodingWire {
            vector: ring.input(format!("{name}-vector"), (1, 8)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{name}-public"), (2, 8)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let state = encoding("state");
        let a_prime = ring.zero((2, 8));
        let decoder_encoding = BggEncodingWire {
            vector: ring.input("decoder-vector", (1, 8)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("decoder-public", (2, 8)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let fresh = encoding("fresh");
        let refresh = RefreshCompiler {
            full_modulus: q.clone().into(),
            crt_plaintext_moduli: moduli.iter().map(|value| BigInt::from(*value).into()).collect(),
            reconstruction_coefficients: parameters
                .reconst_coeffs()
                .into_iter()
                .map(|value| BigInt::from_biguint(num_bigint::Sign::Plus, value).into())
                .collect(),
        };
        let pbc_context = RealPbcTestContext::new(&compiler, &state, [19; 32], [8; 32]);
        let pbc_context_ref = &pbc_context;
        let shared_fresh_error =
            real_fresh_error(pbc_context_ref, &compiler, fresh.clone(), [7; 32], 0, 0, 0);
        let slots = moduli
            .iter()
            .enumerate()
            .map(|(slot, modulus)| {
                let scale_target = ring.polynomial([(q.clone() / BigInt::from(*modulus)).into()]);
                let mask = encoding(&format!("mask-{slot}"));
                let scaled_state = compiler.bgg.large_scalar_mul(&state, &scale_target);
                let scaled_fresh = compiler.bgg.large_scalar_mul(&fresh, &scale_target);
                let masked = compiler.bgg.add(&scaled_state, &mask).unwrap();
                let combined = compiler.bgg.add(&masked, &scaled_fresh).unwrap();
                let target = combined.pubkey.matrix - scale_target.clone() * a_prime.clone();
                let decoder = BggEncodingWire {
                    vector: decoder_encoding.vector.clone(),
                    pubkey: BggPublicKeyWire { matrix: target.clone(), reveal_plaintext: false },
                    plaintext: None,
                };
                RefreshSetupSlotArtifacts {
                    mask: real_mask(pbc_context_ref, &compiler, mask, [7; 32], slot, 0, 0, 0),
                    fresh_error_source: shared_fresh_error.clone(),
                    decoder,
                    b: ring.identity(1),
                    k: target,
                }
            })
            .collect();
        let setup = refresh.bind_setup(&compiler, state, a_prime, slots).unwrap();
        let result = refresh.refresh(&compiler, &setup).unwrap();
        let built = mxx_dsl::DslContext::new("refresh-structural-negative")
            .output("refreshed", result.encoding().vector.clone())
            .unwrap()
            .build()
            .unwrap();
        let attachment = built
            .derivation_attachments
            .iter()
            .find(|attachment| attachment.namespace == "mxx-power-lut")
            .unwrap()
            .clone();
        StructuralRefreshFixture {
            declaration: result.declaration().clone(),
            graph: built.graph,
            attachment,
        }
    }

    fn mutate_structural_graph(
        fixture: &StructuralRefreshFixture,
        mutate: impl FnOnce(&mut serde_json::Value, &Graph, &FrozenDerivationAttachment),
    ) -> Graph {
        let mut encoded = serde_json::to_value(&fixture.graph).unwrap();
        mutate(&mut encoded, &fixture.graph, &fixture.attachment);
        serde_json::from_value(encoded).unwrap()
    }

    fn root_nodes_json(encoded: &mut serde_json::Value) -> &mut Vec<serde_json::Value> {
        scope_nodes_json(encoded, &FrozenGraphScopeId::Root)
    }

    fn scope_nodes_json<'a>(
        encoded: &'a mut serde_json::Value,
        scope_id: &FrozenGraphScopeId,
    ) -> &'a mut Vec<serde_json::Value> {
        let scope_json = serde_json::to_value(scope_id).unwrap();
        encoded["scopes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["id"] == scope_json)
            .map(|entry| entry["scope"]["nodes"].as_array_mut().unwrap())
            .unwrap()
    }

    fn node_role(attachment: &FrozenDerivationAttachment, name: &str) -> WireRef {
        attachment.roles.iter().find(|(role, _)| role == name).unwrap().1.wire
    }

    fn loop_wire(graph: &Graph, attachment: &FrozenDerivationAttachment) -> WireRef {
        let level = node_role(attachment, "slot-0-level");
        let level_node = graph.root_scope().node(level.node).unwrap();
        graph.root_scope().wire_ref(&level_node.arguments()[0]).unwrap()
    }

    fn body_id(graph: &Graph, attachment: &FrozenDerivationAttachment) -> FrozenGraphScopeId {
        graph.child_scope_id(&FrozenGraphScopeId::Root, loop_wire(graph, attachment).node).unwrap()
    }

    fn expect_structural_graph_link(
        mutate: impl FnOnce(&mut serde_json::Value, &Graph, &FrozenDerivationAttachment),
    ) {
        let fixture = structural_refresh_fixture();
        let graph = mutate_structural_graph(&fixture, mutate);
        assert!(matches!(
            fixture.declaration.validate_graph(&fixture.attachment, &graph),
            Err(RefreshError::GraphLink(_))
        ));
    }

    fn shift_root_node_references(value: &mut serde_json::Value, insertion: u64) {
        if let Some(node) = value.get("node").and_then(serde_json::Value::as_u64) {
            if node >= insertion {
                *value.get_mut("node").unwrap() = serde_json::json!(node + 1);
            }
        }
        if let Some(array) = value.as_array_mut() {
            for child in array {
                shift_root_node_references(child, insertion);
            }
        } else if let Some(object) = value.as_object_mut() {
            for child in object.values_mut() {
                shift_root_node_references(child, insertion);
            }
        }
    }

    fn shift_attachment_root_nodes(attachment: &mut FrozenDerivationAttachment, insertion: u64) {
        for (_, wire) in &mut attachment.roles {
            if wire.scope == FrozenGraphScopeId::Root && wire.wire.node.0 >= insertion {
                wire.wire.node.0 += 1;
            }
        }
    }

    fn replace_node_kind(nodes: &mut [serde_json::Value], node: WireRef, kind: NodeKind) {
        nodes[node.node.0 as usize]["kind"] = serde_json::to_value(kind).unwrap();
    }

    #[test]
    fn structural_validator_rejects_same_shaped_altered_body_arithmetic() {
        expect_structural_graph_link(|encoded, graph, attachment| {
            let body = graph.scope(&body_id(graph, attachment)).unwrap();
            let output = body.outputs()[0];
            replace_node_kind(
                scope_nodes_json(encoded, &body_id(graph, attachment)),
                output,
                NodeKind::MatrixBinary(MatrixBinaryOp::Add),
            );
        });
    }

    #[test]
    fn structural_validator_rejects_omitted_scaled_fresh_same_shape() {
        expect_structural_graph_link(|encoded, graph, attachment| {
            let body = graph.scope(&body_id(graph, attachment)).unwrap();
            let output = body.outputs()[0];
            let subtract = body.node(output.node).unwrap();
            let add = body.arguments(subtract).unwrap()[0];
            let add_node = body.node(add.node).unwrap();
            let args = body.arguments(add_node).unwrap();
            let nodes = scope_nodes_json(encoded, &body_id(graph, attachment));
            nodes[add.node.0 as usize]["arguments"][1] = serde_json::to_value(args[0]).unwrap();
        });
    }

    #[test]
    fn structural_validator_rejects_swapped_mask_and_fresh_family_bindings() {
        expect_structural_graph_link(|encoded, graph, attachment| {
            let loop_ref = loop_wire(graph, attachment);
            let nodes = root_nodes_json(encoded);
            let args = nodes[loop_ref.node.0 as usize]["arguments"].as_array_mut().unwrap();
            args.swap(1, 3);
        });
    }

    #[test]
    fn structural_validator_rejects_wrong_slot_projection_index() {
        expect_structural_graph_link(|encoded, _graph, attachment| {
            let level = node_role(attachment, "slot-0-level");
            replace_node_kind(
                root_nodes_json(encoded),
                level,
                NodeKind::FamilyGetStatic { index: 1.into() },
            );
        });
    }

    #[test]
    fn structural_validator_rejects_mixed_same_shaped_parallel_loop_producers() {
        let fixture = structural_refresh_fixture();
        let graph = mutate_structural_graph(&fixture, |encoded, graph, attachment| {
            let original = loop_wire(graph, attachment);
            let body_scope_id = body_id(graph, attachment);
            let level = node_role(attachment, "slot-1-level");
            let body_json = serde_json::to_value(&body_scope_id).unwrap();
            let duplicate_body = encoded["scopes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["id"] == body_json)
                .unwrap()
                .clone();
            let insertion = original.node.0 + 1;
            shift_root_node_references(&mut encoded["outputs"], insertion);
            shift_root_node_references(&mut encoded["effect_roots"], insertion);
            let scopes = encoded["scopes"].as_array_mut().unwrap();
            let root_index = scopes
                .iter()
                .position(|entry| {
                    let id = &entry["id"];
                    id.as_str() == Some("Root") || id["tag"].as_str() == Some("Root")
                })
                .unwrap();
            shift_root_node_references(&mut scopes[root_index]["scope"]["outputs"], insertion);
            {
                let root_nodes = scopes[root_index]["scope"]["nodes"].as_array_mut().unwrap();
                for node in root_nodes.iter_mut() {
                    let id = node["id"].as_u64().unwrap();
                    if id >= insertion {
                        node["id"] = serde_json::json!(id + 1);
                    }
                    shift_root_node_references(&mut node["arguments"], insertion);
                }
                let new_id = mxx_ir_core::NodeId(insertion);
                let mut duplicate = root_nodes[original.node.0 as usize].clone();
                duplicate["id"] = serde_json::json!(insertion);
                root_nodes.insert(insertion as usize, duplicate);
                root_nodes[level.node.0 as usize + 1]["arguments"][0] =
                    serde_json::to_value(WireRef { node: new_id, port: mxx_ir_core::Port(0) })
                        .unwrap();
            }
            let new_id = mxx_ir_core::NodeId(insertion);
            let new_body_id = FrozenGraphScopeId::ParallelBody {
                parent: Box::new(FrozenGraphScopeId::Root),
                owner: new_id,
            };
            let mut duplicate_body = duplicate_body;
            duplicate_body["id"] = serde_json::to_value(new_body_id).unwrap();
            scopes.push(duplicate_body);
        });
        let mut attachment = fixture.attachment.clone();
        shift_attachment_root_nodes(
            &mut attachment,
            loop_wire(&fixture.graph, &fixture.attachment).node.0 + 1,
        );
        assert!(matches!(
            fixture.declaration.validate_graph(&attachment, &graph),
            Err(RefreshError::GraphLink(_))
        ));
    }

    #[test]
    fn structural_validator_rejects_wrong_parallel_count_mode_and_schema() {
        for mutate in [
            0usize, // wrong count
            1usize, // wrong mode
            2usize, // wrong body schema
        ] {
            expect_structural_graph_link(move |encoded, graph, attachment| {
                let loop_ref = loop_wire(graph, attachment);
                let nodes = root_nodes_json(encoded);
                let node = &mut nodes[loop_ref.node.0 as usize];
                let mut kind: NodeKind = serde_json::from_value(node["kind"].clone()).unwrap();
                let NodeKind::ParallelLoop(ref mut spec) = kind else { unreachable!() };
                match mutate {
                    0 => spec.count = 3.into(),
                    1 => spec.input_modes[0] = LoopInputMode::Broadcast,
                    _ => {
                        spec.input_modes.pop();
                    }
                };
                node["kind"] = serde_json::to_value(kind).unwrap();
            });
        }
    }

    #[test]
    fn refresh_scale_is_exact_q_over_qt() {
        let compiler = config();
        assert_eq!(
            compiler.scale_expression(0).unwrap().evaluate(&Default::default()).unwrap(),
            19.into()
        );
        assert_eq!(
            compiler.scale_expression(1).unwrap().evaluate(&Default::default()).unwrap(),
            17.into()
        );
    }

    #[test]
    fn routed_prf_digit_has_no_rns_scale_and_refresh_body_has_one() {
        let ring = mxx_dsl::Ring::new(257, 4);
        let compiler = PowerLutEncodingCompiler::from_public_key(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 2.into(),
        });
        let value = BggEncodingWire {
            vector: ring.input("route-vector", (1, 4)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("route-public", (2, 4)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let routed = route_prf_digit(&compiler, &value, 3, 1, 2, 1).unwrap();
        let refresh = config();
        let scale = ring.polynomial([refresh.scale_expression(0).unwrap()]);
        let scaled = compiler.bgg.large_scalar_mul(&value, &scale);
        let graph = mxx_dsl::DslContext::new("refresh-route-scale-structure")
            .output("routed", routed.vector)
            .unwrap()
            .output("scaled", scaled.vector)
            .unwrap()
            .build()
            .unwrap();
        let encoded = serde_json::to_string(&graph.graph).unwrap();
        let division_count =
            encoded.matches("\"tag\":\"Div\"").count() + encoded.matches("\"Divide\"").count();
        assert_eq!(division_count, 1, "only the refresh body carries q/q_t scaling");
    }

    #[test]
    fn decoder_binds_the_actual_anchor_product_not_a_handle_label() {
        let ring = mxx_dsl::Ring::new(257, 4);
        let encoding = |name: &str| mxx_bgg::BggEncodingWire {
            vector: ring.input(name, (1, 1)),
            pubkey: mxx_bgg::BggPublicKeyWire {
                matrix: ring.input(format!("{name}-public"), (1, 1)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let b = ring.input("anchor-b", (1, 1));
        let k = ring.input("anchor-k", (1, 1));
        let anchor = RefreshAnchor::with_equation(b, k);
        let target = anchor.equation().unwrap().target.clone();
        let decoder = RefreshDecoderPreimage::bind(&anchor, encoding("decoder"), &target).unwrap();
        let wrong_target = ring.input("wrong-target", (1, 1));
        let different_anchor = RefreshAnchor::with_equation(
            ring.input("different-b", (1, 1)),
            ring.input("different-k", (1, 1)),
        );
        assert_eq!(
            decoder.validate(&different_anchor, &wrong_target),
            Err(RefreshError::AnchorMismatch)
        );
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn multi_slot_refresh_executes_combined_order_and_crt_recomposition() {
        use mxx_dsl::DslContext;
        use mxx_ir_core::ParamEnv;
        use mxx_primitives::{
            matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
            poly::{
                Poly, PolyParams,
                dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
            },
        };
        use mxx_runtime::{
            RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
            execute_in_session, transcript::SamplingMode,
        };

        let parameters = DCRTPolyParams::new(4, 2, 17, 16);
        assert_eq!(parameters.crt_depth(), 2);
        assert_eq!(parameters.crt_bits().div_ceil(parameters.base_bits() as usize), 2);
        assert_eq!(parameters.modulus_digits(), 4);
        let q = BigInt::from(parameters.modulus().as_ref().clone());
        let (moduli, _, _) = parameters.to_crt();
        assert_eq!(moduli.len(), 2);
        let ring = mxx_dsl::Ring::new(q.clone(), 4);
        let compiler =
            crate::PowerLutEncodingCompiler::from_public_key(mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 65_536.into(),
                digit_count: 4.into(),
            });
        let encoding = |name: &str| mxx_bgg::BggEncodingWire {
            vector: ring.input(format!("{name}-vector"), (1, 8)),
            pubkey: mxx_bgg::BggPublicKeyWire {
                matrix: ring.input(format!("{name}-public"), (2, 8)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let state = encoding("state");
        let a_prime = ring.zero((2, 8));
        let fresh = encoding("fresh");
        let refresh = RefreshCompiler {
            full_modulus: q.clone().into(),
            crt_plaintext_moduli: moduli.iter().map(|value| BigInt::from(*value).into()).collect(),
            reconstruction_coefficients: parameters
                .reconst_coeffs()
                .into_iter()
                .map(|value| BigInt::from_biguint(num_bigint::Sign::Plus, value).into())
                .collect(),
        };
        let pbc_context = RealPbcTestContext::new(&compiler, &state, [19; 32], [8; 32]);
        let pbc_context_ref = &pbc_context;
        let shared_fresh_error =
            real_fresh_error(pbc_context_ref, &compiler, fresh.clone(), [7; 32], 0, 0, 0);
        let slot_artifacts = moduli
            .iter()
            .enumerate()
            .map(|(slot, modulus)| {
                let scale = q.clone() / BigInt::from(*modulus);
                let scale_target = ring.polynomial([scale.clone().into()]);
                let mask_wire = encoding(&format!("mask-{slot}"));
                let scaled_state = compiler.bgg.large_scalar_mul(&state, &scale_target);
                let scaled_fresh = compiler.bgg.large_scalar_mul(&fresh, &scale_target);
                let masked = compiler.bgg.add(&scaled_state, &mask_wire).unwrap();
                let combined = compiler.bgg.add(&masked, &scaled_fresh).unwrap();
                let a_sum_t = combined.pubkey.matrix;
                let target = a_sum_t.clone() - scale_target.clone() * a_prime.clone();
                let b = ring.identity(1);
                let k = target.clone();
                let decoder = mxx_bgg::BggEncodingWire {
                    vector: ring.input(format!("decoder-{slot}-vector"), (1, 8)),
                    pubkey: mxx_bgg::BggPublicKeyWire { matrix: target, reveal_plaintext: false },
                    plaintext: None,
                };
                assert_eq!(scale, q.clone() / BigInt::from(*modulus));
                RefreshSetupSlotArtifacts {
                    mask: real_mask(pbc_context_ref, &compiler, mask_wire, [7; 32], slot, 0, 0, 0),
                    fresh_error_source: shared_fresh_error.clone(),
                    decoder,
                    b,
                    k,
                }
            })
            .collect::<Vec<_>>();
        let mut malformed = slot_artifacts.clone();
        malformed[0].fresh_error_source =
            real_fresh_error(&pbc_context, &compiler, encoding("anchor-fresh"), [7; 32], 0, 0, 0);
        assert_eq!(
            refresh.bind_setup(&compiler, state.clone(), a_prime.clone(), malformed).err(),
            Some(RefreshError::FreshErrorMismatch),
            "each CRT slot must use the same fresh-error source"
        );
        let public_invariance_slots = slot_artifacts.clone();
        let setup =
            refresh.bind_setup(&compiler, state.clone(), a_prime.clone(), slot_artifacts).unwrap();
        let changed_state = encoding("anchor-state");
        let invariant_setup = refresh
            .bind_setup(&compiler, changed_state, a_prime.clone(), public_invariance_slots)
            .unwrap();
        assert_ne!(
            setup.packages[0].target_public_matrix.value_handle(),
            setup.packages[1].target_public_matrix.value_handle(),
            "RNS slots must retain distinct decoder targets"
        );
        let refreshed = refresh.refresh(&compiler, &setup).unwrap();
        assert_eq!(
            refreshed.encoding().pubkey.matrix.value_handle(),
            a_prime.value_handle(),
            "the refreshed owner is A', not a per-slot decoder target"
        );
        // Build an independent sequential reference graph from the
        // authoritative BGG primitives and CRT operation. CrtRecompose
        // rounds t*v/q, so raw decoder constants 1 and 2 are below its
        // quantization step and disappear. The runtime probes are therefore
        // alpha_t*r_t, where alpha_t=q/q_t and r_t is 1 or 2.
        let expected_levels: Vec<Mat> = setup
            .packages
            .iter()
            .enumerate()
            .map(|(slot, package)| {
                let scale = ring.polynomial([(q.clone() / BigInt::from(moduli[slot])).into()]);
                let scaled = compiler.bgg.large_scalar_mul(&state, &scale);
                let scaled_fresh =
                    compiler.bgg.large_scalar_mul(&package.fresh_error_source, &scale);
                let combined = compiler.bgg.add(&scaled, &package.mask).unwrap();
                let combined = compiler.bgg.add(&combined, &scaled_fresh).unwrap();
                compiler.bgg.sub(&combined, &package.decoder.encoding).unwrap().vector
            })
            .collect();
        let expected = Mat::crt_recompose(
            expected_levels.clone(),
            refresh.crt_plaintext_moduli.clone(),
            refresh.reconstruction_coefficients.clone(),
        );
        let swapped = Mat::crt_recompose(
            expected_levels.into_iter().rev().collect(),
            refresh.crt_plaintext_moduli.clone(),
            refresh.reconstruction_coefficients.clone(),
        );
        let context = DslContext::new("power-lut-refresh-runtime")
            .output("refreshed", refreshed.encoding().vector.clone())
            .unwrap()
            .output("expected", expected)
            .unwrap()
            .output("swapped", swapped)
            .unwrap();
        let context =
            setup.packages.iter().enumerate().fold(context, |context, (slot, package)| {
                context
                    .output(format!("target-{slot}"), package.target_public_matrix.clone())
                    .unwrap()
                    .output(format!("b-k-{slot}"), package.b.clone() * package.k.clone())
                    .unwrap()
            });
        let context = invariant_setup.packages.iter().enumerate().fold(
            context,
            |context, (slot, package)| {
                context
                    .output(
                        format!("invariant-target-{slot}"),
                        package.target_public_matrix.clone(),
                    )
                    .unwrap()
            },
        );
        let built = context.build().unwrap();
        let selector_graph = pbc_context.selector_graph.validate(&ParamEnv::default()).unwrap();
        let mut artifact_store = MemoryArtifactStore::default();
        let selector_bits = RuntimeValue::IndexedFamily(
            pbc_context
                .selector_bit_values
                .iter()
                .map(|bit| {
                    RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                        &parameters,
                        vec![DCRTPoly::from_usize_to_constant(&parameters, *bit as usize)],
                    ))
                })
                .collect(),
        );
        let selector_result = execute_in_session(
            &selector_graph,
            &mut cpu_backend([parameters.clone()]),
            std::collections::BTreeMap::from([
                ("real-pbc-selector-key".to_owned(), RuntimeValue::Bytes(vec![0; 32])),
                (pbc_context.selector_bit_input_name.clone(), selector_bits),
            ]),
            &mut artifact_store,
            pbc_context.selector_production.execution_nonce,
        )
        .unwrap();
        assert_eq!(selector_result.production_id.as_ref(), Some(&pbc_context.selector_production));
        let event = refreshed.declaration();
        assert_eq!(
            (event.owner.as_str(), event.plan.as_str(), event.value_type.as_str()),
            ("mxx-power-lut", "section-7-refresh", "BggEncodingVector")
        );
        assert_eq!((event.rows, event.columns), (1, 8));
        assert_eq!(event.slots.len(), 2);
        let encoded = serde_json::to_vec(event).unwrap();
        let decoded: RefreshDeclaration = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(&decoded, event, "canonical refresh ABI must round-trip");
        let attachment = built
            .derivation_attachments
            .iter()
            .find(|attachment| {
                attachment.namespace == "mxx-power-lut" && attachment.rule == "section-7-refresh"
            })
            .expect("event attachment");
        assert_eq!(event.validate_graph(attachment, &built.graph), Ok(()));
        assert!(
            PowerLutProtocolDeclaration::new(vec![event.clone()]).unwrap().validate(&built).is_ok()
        );
        let mut tampered = attachment.clone();
        tampered.roles.pop();
        assert!(event.validate_graph(&tampered, &built.graph).is_err());
        let mut tampered_target = attachment.clone();
        tampered_target.roles[3].1 = tampered_target.roles[1].1.clone();
        assert!(
            event.validate_graph(&tampered_target, &built.graph).is_err(),
            "tampered B*K target must fail closed"
        );
        let mut tampered_output = attachment.clone();
        tampered_output.roles[0].1 = tampered_output.roles[1].1.clone();
        assert!(
            event.validate_graph(&tampered_output, &built.graph).is_err(),
            "tampered refresh output handle must fail closed"
        );
        let mut tampered_fresh = attachment.clone();
        tampered_fresh.roles[5].1 = tampered_fresh.roles[4].1.clone();
        assert!(event.validate_graph(&tampered_fresh, &built.graph).is_err());
        let mut tampered_slots = event.clone();
        tampered_slots.slots.swap(0, 1);
        assert!(
            tampered_slots.validate_graph(attachment, &built.graph).is_err(),
            "tampered slot order must fail closed"
        );
        let mut tampered_coefficients = event.clone();
        tampered_coefficients.reconstruction_coefficients[0] += 1;
        assert!(
            tampered_coefficients.validate_graph(attachment, &built.graph).is_err(),
            "tampered CRT reconstruction coefficients must fail closed"
        );
        let selector_manifest =
            artifact_store.manifest(&pbc_context.selector_production).cloned().unwrap();
        let graph = built
            .validate_with_manifests(
                &ParamEnv::default(),
                &std::collections::BTreeMap::from([(
                    pbc_context.selector_production.clone(),
                    selector_manifest,
                )]),
            )
            .unwrap();
        let decoder_one = mxx_primitives::poly::dcrt::poly::DCRTPoly::const_one(&parameters);
        let decoder_two = &decoder_one + &decoder_one;
        let state_polynomial = decoder_two.clone();
        let decoder_polynomials = moduli
            .iter()
            .enumerate()
            .map(|(slot, modulus)| {
                let alpha_r = (q.clone() / BigInt::from(*modulus)) * BigInt::from(slot + 1);
                mxx_primitives::poly::dcrt::poly::DCRTPoly::from_biguint_to_constant(
                    &parameters,
                    alpha_r.to_biguint().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let state_matrix = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            (0..8).map(|_| state_polynomial.clone()).collect(),
        );
        let zero_vector = DCRTPolyMatrix::zero(&parameters, 1, 8);
        let zero_public = DCRTPolyMatrix::zero(&parameters, 2, 8);
        let runtime_inputs = std::collections::BTreeMap::from([
            ("state-vector".to_owned(), RuntimeValue::matrix(state_matrix)),
            ("state-public".to_owned(), RuntimeValue::matrix(zero_public.clone())),
            ("fresh-vector".to_owned(), RuntimeValue::matrix(zero_vector.clone())),
            ("fresh-public".to_owned(), RuntimeValue::matrix(zero_public.clone())),
            ("mask-0-vector".to_owned(), RuntimeValue::matrix(zero_vector.clone())),
            ("mask-0-public".to_owned(), RuntimeValue::matrix(zero_public.clone())),
            ("mask-1-vector".to_owned(), RuntimeValue::matrix(zero_vector.clone())),
            ("mask-1-public".to_owned(), RuntimeValue::matrix(zero_public.clone())),
            (
                "decoder-0-vector".to_owned(),
                RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    (0..8).map(|_| decoder_polynomials[0].clone()).collect(),
                )),
            ),
            (
                "decoder-1-vector".to_owned(),
                RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    (0..8).map(|_| decoder_polynomials[1].clone()).collect(),
                )),
            ),
            ("anchor-state-vector".to_owned(), RuntimeValue::matrix(zero_vector)),
            ("anchor-state-public".to_owned(), RuntimeValue::matrix(zero_public)),
            ("real-pbc-helper-key".to_owned(), RuntimeValue::Bytes(vec![0; 32])),
        ]);
        let result = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            runtime_inputs,
            &mut artifact_store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(actual) = &result.outputs["refreshed"] else { panic!("matrix") };
        let RuntimeValue::Matrix(expected) = &result.outputs["expected"] else { panic!("matrix") };
        let RuntimeValue::Matrix(swapped) = &result.outputs["swapped"] else { panic!("matrix") };
        assert_eq!(actual, expected, "full CRT refresh output and ordering");
        assert_ne!(actual.as_ref(), &DCRTPolyMatrix::zero(&parameters, 1, 8));
        assert_ne!(actual, swapped, "CRT slot order must affect recomposition");
        for slot in 0..setup.packages.len() {
            let RuntimeValue::Matrix(target) = &result.outputs[&format!("target-{slot}")] else {
                panic!("target matrix")
            };
            let RuntimeValue::Matrix(product) = &result.outputs[&format!("b-k-{slot}")] else {
                panic!("B*K matrix")
            };
            let RuntimeValue::Matrix(invariant) =
                &result.outputs[&format!("invariant-target-{slot}")]
            else {
                panic!("invariant target matrix")
            };
            assert_eq!(product, target, "slot {slot} decoder preimage equation");
            assert_eq!(
                invariant, target,
                "slot {slot} public target must be independent of the private state vector"
            );
        }
    }

    #[test]
    fn refresh_rejects_non_dividing_qt_before_graph_construction() {
        let mut compiler = config();
        compiler.full_modulus = 320.into();
        assert_eq!(compiler.validate_layout(), Err(RefreshError::InvalidLayout));
    }

    fn boundary_parameters() -> (
        crate::refresh_setup::RefreshSetupParameters,
        PowerLutEncodingCompiler,
        crate::program::PowerLutProgramId,
    ) {
        let ring = mxx_dsl::Ring::new(323, 4);
        let compiler = PowerLutEncodingCompiler::from_public_key(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 3.into(),
        });
        let body = crate::prf::test_sparse_lwr_program(2, 4).unwrap();
        let refresh = config();
        let parameters = crate::refresh_setup::RefreshSetupParameters {
            refresh_id: [7; 32],
            base_p: 2,
            component_count: 2,
            coefficient_count: 1,
            digit_count: 3,
            lut_width: 4,
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 323.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 3,
                gadget_base: 2.into(),
            },
            refresh,
            decoder_sigma: 1.into(),
            decoder_preimage_bound: mxx_bgg::PreimageCoefficientBound::Official,
            name: "pbc-boundary".to_owned(),
        };
        (parameters, compiler, body.id())
    }

    fn boundary_wire(compiler: &PowerLutEncodingCompiler, name: &str) -> BggEncodingWire {
        let ring = compiler.bgg.public_key.ring.clone();
        BggEncodingWire {
            vector: ring.input(format!("{name}-vector"), (1, 6)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{name}-public"), (2, 6)),
                reveal_plaintext: false,
            },
            plaintext: None,
        }
    }

    fn complete_boundary_inputs() -> (
        crate::refresh_setup::RefreshSetupParameters,
        crate::program::PowerLutProgramId,
        Vec<Vec<RefreshMaskPrfOutput>>,
        Vec<RefreshFreshErrorPrfOutput>,
    ) {
        let (parameters, compiler, program_id) = boundary_parameters();
        let context = RealPbcTestContext::new(
            &compiler,
            &boundary_wire(&compiler, "context"),
            [19; 32],
            [8; 32],
        );
        let context_ref = &context;
        let compiler_ref = &compiler;
        let masks = (0..2)
            .map(|slot| {
                (0..2)
                    .flat_map(|component| {
                        (0..3).map(move |digit| {
                            real_mask(
                                context_ref,
                                compiler_ref,
                                boundary_wire(
                                    compiler_ref,
                                    &format!("mask-{slot}-{component}-{digit}"),
                                ),
                                parameters.refresh_id,
                                slot,
                                component,
                                0,
                                digit,
                            )
                        })
                    })
                    .collect()
            })
            .collect();
        let fresh = (0..2)
            .flat_map(|component| {
                (0..3).map(move |digit| {
                    real_fresh_error(
                        context_ref,
                        compiler_ref,
                        boundary_wire(compiler_ref, &format!("fresh-{component}-{digit}")),
                        parameters.refresh_id,
                        component,
                        0,
                        digit,
                    )
                })
            })
            .collect();
        (parameters, program_id, masks, fresh)
    }

    #[test]
    fn refresh_prf_inputs_accepts_complete_real_pbc_coverage() {
        let (parameters, program_id, masks, fresh) = complete_boundary_inputs();
        assert!(
            crate::refresh_setup::RefreshPrfInputs::from_pbc_outputs(
                &parameters,
                program_id,
                masks,
                fresh,
            )
            .is_ok()
        );
    }

    #[test]
    fn refresh_prf_inputs_rejects_wrong_refresh_id_slot_order_duplicate_and_missing() {
        let (parameters, program_id, masks, fresh) = complete_boundary_inputs();
        let (_, compiler, _) = boundary_parameters();
        let context = RealPbcTestContext::new(
            &compiler,
            &boundary_wire(&compiler, "rejection-context"),
            [19; 32],
            [8; 32],
        );
        let wrong_id = real_mask(
            &context,
            &compiler,
            boundary_wire(&compiler, "wrong-id"),
            [8; 32],
            0,
            0,
            0,
            0,
        );
        let mut wrong_id_masks = masks.clone();
        wrong_id_masks[0][0] = wrong_id;
        assert!(
            crate::refresh_setup::RefreshPrfInputs::from_pbc_outputs(
                &parameters,
                program_id,
                wrong_id_masks,
                fresh.clone(),
            )
            .is_err()
        );

        let mut wrong_slot = masks.clone();
        wrong_slot[0][0] = real_mask(
            &context,
            &compiler,
            boundary_wire(&compiler, "wrong-slot"),
            parameters.refresh_id,
            1,
            0,
            0,
            0,
        );
        assert!(
            crate::refresh_setup::RefreshPrfInputs::from_pbc_outputs(
                &parameters,
                program_id,
                wrong_slot,
                fresh.clone(),
            )
            .is_err()
        );

        let mut swapped = masks.clone();
        swapped.swap(0, 1);
        assert!(
            crate::refresh_setup::RefreshPrfInputs::from_pbc_outputs(
                &parameters,
                program_id,
                swapped,
                fresh.clone(),
            )
            .is_err()
        );

        let mut duplicate = masks.clone();
        duplicate[0][2] = duplicate[0][0].clone();
        assert!(
            crate::refresh_setup::RefreshPrfInputs::from_pbc_outputs(
                &parameters,
                program_id,
                duplicate,
                fresh.clone(),
            )
            .is_err()
        );

        let mut missing = masks;
        missing[0].pop();
        assert!(
            crate::refresh_setup::RefreshPrfInputs::from_pbc_outputs(
                &parameters,
                program_id,
                missing,
                fresh,
            )
            .is_err()
        );
    }

    #[test]
    fn refresh_prf_inputs_rejects_mixed_program_and_layout() {
        let (parameters, program_id, mut masks, fresh) = complete_boundary_inputs();
        // Use a genuinely different, but still valid, sparse-LWR profile for
        // the first branch.  Reusing the boundary helper here would construct
        // the same composite program identity; the seed is intentionally kept
        // equal so this assertion isolates program binding from layout
        // binding.
        let alt_ring = mxx_dsl::Ring::new(323, 8);
        let alt_compiler =
            PowerLutEncodingCompiler::from_public_key(mxx_bgg::BggPublicKeyCompiler {
                ring: alt_ring.clone(),
                base: 2.into(),
                digit_count: 2.into(),
            });
        let alt_context = RealPbcTestContext::new(
            &alt_compiler,
            &BggEncodingWire {
                vector: alt_ring.input("alternate-program-context-vector", (1, 4)),
                pubkey: BggPublicKeyWire {
                    matrix: alt_ring.input("alternate-program-context-public", (2, 4)),
                    reveal_plaintext: false,
                },
                plaintext: None,
            },
            [19; 32],
            [9; 32],
        );
        let mut alt_context = alt_context;
        // W = 8 needs three ClearCoeff helper rounds.  The regular boundary
        // fixture uses W = 4, so replace only this test context's helpers.
        let alt_sampler = crate::encoding::PowerLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 323.into(),
                ring_dimension: 8.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 2.into(),
            },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        alt_context.helpers = alt_sampler
            .sample_automorphism_helpers(
                alt_ring.zero((1, 2)),
                alt_ring.bytes_input("alternate-program-helper-key", 32),
                b"alternate-program-helpers".to_vec(),
                8,
            )
            .unwrap();
        let alt_body = crate::prf::SparseLwrPrfProgram::new(
            crate::prf::SparseLwrPrfProfile::new(3, 2, 8, 8).unwrap(),
            2,
        )
        .unwrap();
        let alt_output = alt_context.lower(
            &alt_compiler,
            BggEncodingWire {
                vector: alt_ring.input("alternate-program-vector", (1, 4)),
                pubkey: BggPublicKeyWire {
                    matrix: alt_ring.input("alternate-program-public", (2, 4)),
                    reveal_plaintext: false,
                },
                plaintext: None,
            },
            &alt_body,
            &RefreshPrfLabel::Mask {
                refresh_id: parameters.refresh_id,
                slot: 0,
                component: 0,
                coefficient: 0,
                digit: 0,
            }
            .canonical_bytes(),
        );
        masks[0][0] = RefreshMaskPrfOutput::from_pbc_evaluation(
            alt_output,
            parameters.refresh_id,
            0,
            0,
            0,
            0,
        )
        .unwrap();
        assert!(
            crate::refresh_setup::RefreshPrfInputs::from_pbc_outputs(
                &parameters,
                program_id,
                masks,
                fresh.clone(),
            )
            .is_err()
        );

        let (_, layout_compiler, _) = boundary_parameters();
        let layout_context = RealPbcTestContext::new(
            &layout_compiler,
            &boundary_wire(&layout_compiler, "alternate-layout-context"),
            [20; 32],
            [8; 32],
        );
        let layout_body =
            crate::prf::test_sparse_lwr_program(layout_context.layout.bucket_width, 4).unwrap();
        let layout_output = layout_context.lower(
            &layout_compiler,
            boundary_wire(&layout_compiler, "alternate-layout"),
            &layout_body,
            &RefreshPrfLabel::Mask {
                refresh_id: parameters.refresh_id,
                slot: 0,
                component: 0,
                coefficient: 0,
                digit: 0,
            }
            .canonical_bytes(),
        );
        let (_, _, mut layout_masks, layout_fresh) = complete_boundary_inputs();
        layout_masks[0][0] = RefreshMaskPrfOutput::from_pbc_evaluation(
            layout_output,
            parameters.refresh_id,
            0,
            0,
            0,
            0,
        )
        .unwrap();
        assert!(
            crate::refresh_setup::RefreshPrfInputs::from_pbc_outputs(
                &parameters,
                program_id,
                layout_masks,
                layout_fresh,
            )
            .is_err()
        );
    }
}
