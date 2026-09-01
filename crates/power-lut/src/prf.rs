//! Sparse-LWR PRF application built on the generic Power-LUT program and
//! Fuse/LUT core.
//!
//! This module owns the sparse-LWR application descriptor and its reusable
//! program. PBC layout, matching, selector artifacts, and clear inner-product
//! oracles live in the sibling [`crate::pbc`] module. Keeping these layers
//! separate ensures that the public-key path sees only public PBC projections,
//! while the same program description can be lowered privately or publicly.
//!
//! The program builders below describe one structural bucket body. They carry
//! no sparse support, schedule, selected slot, selector bit, or private RHS
//! material; those values are supplied only through explicit lowering bindings.
//!
//! For a bucket with public factors `X^{a_i}`, selector ciphertexts `C_i`, and
//! one-hot bits `b_i`, the accumulator update is
//! `z' = z + sum_i b_i Fuse(z, C_i) * X^{a_i}`. The public projection performs
//! the same update on matrices `A`, consuming only concrete public `C_i` and
//! `X^{a_i}`. After all buckets, scalar LWR rounding is applied once as
//! `floor(p * z / Q_L)` (with the centered representative used by the ring).

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use mxx_bgg::{BggEncodingWire, BggPublicKeyWire};
use mxx_dsl::{Family, Int, Mat, Parallel, Sequential};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    PowerLutEncodingCompiler, PowerLutError,
    encoding::{EncodingSelectorFamily, FlatLutHelper, FlatLutHelperMap, FlatLutHelperSet},
    program::{
        FamilyRange, LutTable, PBC_MONOMIAL_FAMILY_PROVENANCE, PowerLutMonomialFamily,
        PowerLutProgram, PowerLutProgramBuilder, PowerLutProgramId, ProgramFamilyRanges,
        ProgramInputId, ProgramValidationError, ProgramWireId, PublicValueFamilyId, RhsFamilyId,
    },
    public_key::{
        FlatLutPublicHelper, FlatLutPublicHelperMap, FlatLutPublicHelperSet,
        PowerLutPublicKeyCompiler, PublicSelectorFamily,
    },
    rhs::PowerRhsPackage,
};

/// Trusted, public description of the refresh PRF batch order.
///
/// Construction accepts the layout, concrete sparse-LWR profile, and the
/// canonical refresh label index together. It derives the immutable label
/// digests and public-vector identities in that order; it never accepts a
/// caller-provided label list and stores no support, schedule, or selector
/// material. Runtime RHS and encoding families are bound separately by the
/// trusted key provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshPrfBatchInputs {
    layout: crate::pbc::PbcPublicLayout,
    layout_id: crate::pbc::PbcLayoutId,
    profile: SparseLwrPrfProfile,
    batch_id: PbcPublicValueBatchId,
    active_count: usize,
    value_count: usize,
    label_count: usize,
    program_id: PowerLutProgramId,
    output_wire: ProgramWireId,
}

/// Stable identity for the deferred public-value family payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PbcPublicValueBatchId(pub [u8; 32]);

impl RefreshPrfBatchInputs {
    /// Derives a canonical batch description from public layout metadata and a
    /// refresh label index. Public vectors are routed through the same PBC
    /// layout used by the evaluator, then committed over public ordered data.
    pub fn new(
        layout: &crate::pbc::PbcPublicLayout,
        profile: SparseLwrPrfProfile,
        labels: &crate::refresh::RefreshPrfLabelIndex,
        program: &SparseLwrPrfProgram,
    ) -> Result<Self, PowerLutError> {
        layout.validate().map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        if program.profile() != &profile {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let active_count = crate::pbc::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .len();
        let value_count =
            labels.len().checked_mul(active_count).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let batch_id =
            public_value_batch_id(layout, &profile, labels, active_count, value_count, program)?;
        Ok(Self {
            layout: layout.clone(),
            layout_id: layout.layout_id,
            profile,
            batch_id,
            active_count,
            value_count,
            label_count: labels.len(),
            program_id: program.id(),
            output_wire: program.terminal_output_wire(),
        })
    }

    /// Returns the number of canonical label positions.
    pub fn len(&self) -> usize {
        self.label_count
    }

    /// Returns whether the batch contains no labels.
    pub fn is_empty(&self) -> bool {
        self.label_count == 0
    }

    /// Returns the layout identity bound into every batch member.
    pub const fn layout_id(&self) -> crate::pbc::PbcLayoutId {
        self.layout_id
    }

    /// Returns the validated public layout used for all batch members.
    pub fn layout(&self) -> &crate::pbc::PbcPublicLayout {
        &self.layout
    }

    /// Returns the concrete profile used for vector derivation.
    pub const fn profile(&self) -> &SparseLwrPrfProfile {
        &self.profile
    }

    /// Returns the stable identity of the deferred public-value family.
    pub const fn batch_id(&self) -> PbcPublicValueBatchId {
        self.batch_id
    }

    /// Returns the number of active (non-padding) layout cells per label.
    pub const fn active_count(&self) -> usize {
        self.active_count
    }

    /// Returns the total number of public values in the flattened batch.
    pub const fn value_count(&self) -> usize {
        self.value_count
    }

    pub const fn program_id(&self) -> PowerLutProgramId {
        self.program_id
    }
    pub const fn output_wire(&self) -> ProgramWireId {
        self.output_wire
    }

    /// Returns the stable input name used for this deferred public family.
    ///
    /// Runtime values for this input are ring matrices containing the
    /// monomial `X^a` for each host residue `a`; they are not constant
    /// polynomials with coefficient `a`.
    pub fn public_input_name(&self) -> String {
        format!(
            "pbc-public-values-{:x}",
            u128::from_le_bytes(self.batch_id.0[..16].try_into().unwrap())
        )
    }
}

fn public_value_batch_id(
    layout: &crate::pbc::PbcPublicLayout,
    profile: &SparseLwrPrfProfile,
    labels: &crate::refresh::RefreshPrfLabelIndex,
    active_count: usize,
    value_count: usize,
    program: &SparseLwrPrfProgram,
) -> Result<PbcPublicValueBatchId, PowerLutError> {
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/pbc-public-value-batch/v1");
    digest.update(b"order=mask-slot-component-coefficient-digit/fresh-component-coefficient-digit");
    digest.update(b"public-values=canonical-label-pbc-encoding-v1");
    digest.update(layout.layout_id.0);
    digest.update((profile.q_l as u64).to_le_bytes());
    digest.update((profile.p as u64).to_le_bytes());
    digest.update((profile.lut_width as u64).to_le_bytes());
    digest.update((profile.ring_dimension as u64).to_le_bytes());
    digest.update(labels.refresh_id());
    digest.update((labels.mask_slot_count() as u64).to_le_bytes());
    digest.update((labels.component_count() as u64).to_le_bytes());
    digest.update((labels.coefficient_count() as u64).to_le_bytes());
    digest.update((labels.mask_base_p_digit_count() as u64).to_le_bytes());
    digest.update((labels.fresh_error_base_p_digit_count() as u64).to_le_bytes());
    digest.update((active_count as u64).to_le_bytes());
    digest.update((labels.len() as u64).to_le_bytes());
    digest.update((value_count as u64).to_le_bytes());
    digest.update((layout.bucket_width as u64).to_le_bytes());
    digest.update((layout.parameters.bucket_count as u64).to_le_bytes());
    digest.update(program.id().0);
    digest.update(format!("{:?}", program.terminal_output_wire()).as_bytes());
    digest.update(b"terminal=raw-scalar-v1");
    Ok(PbcPublicValueBatchId(digest.finalize().into()))
}

/// Declares the batch's deferred label-major public-value family. The runtime
/// payload provider is intentionally outside this graph-construction layer.
/// Host PBC materialization remains residue-valued, while each runtime family
/// element bound to this DSL input must be the corresponding ring monomial
/// `X^a`. This distinction is part of the one-hot gate's public-factor
/// contract and is therefore documented at the single family declaration
/// boundary used by both private and public-key lowering.
type SparseLwrPublicMonomialFamily = PowerLutMonomialFamily;

fn batch_public_values_family(
    batch: &RefreshPrfBatchInputs,
    ring: &mxx_dsl::Ring,
) -> Result<SparseLwrPublicMonomialFamily, PowerLutError> {
    let ring_type = ring.matrix_type((1, 1));
    let ring_dimension = ring_type
        .ring_dimension
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
        .to_usize()
        .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
    if ring_dimension != batch.profile.ring_dimension {
        return Err(PowerLutError::InvalidSparseLwrBlock);
    }

    batch.layout.validate().map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
    let active_count = crate::pbc::PbcActiveCellIndex::build(&batch.layout)
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
        .len();
    if batch.active_count != active_count ||
        batch.value_count !=
            batch
                .len()
                .checked_mul(active_count)
                .ok_or(PowerLutError::InvalidSparseLwrBlock)?
    {
        return Err(PowerLutError::InvalidSparseLwrBlock);
    }
    PowerLutMonomialFamily::from_trusted(
        ring.input_family(batch.public_input_name(), batch.value_count, (1, 1)),
        ring,
        PBC_MONOMIAL_FAMILY_PROVENANCE,
    )
    .map_err(|_| PowerLutError::InvalidSparseLwrBlock)
}

/// Concrete sparse-LWR parameters used to construct a PRF program.
///
/// The profile is intentionally value-owned rather than an `IntExpr`: it is
/// validated before any DSL lowering starts. `q_l` is the plaintext modulus,
/// `p` is the LWR output modulus, `lut_width` is the logical Power-LUT domain
/// width `W`, and `ring_dimension` is the concrete ring dimension `N`.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SparseLwrPrfProfile {
    q_l: usize,
    p: usize,
    lut_width: usize,
    ring_dimension: usize,
}

/// Immutable schedule and LUT-width contract for grouped sparse-LWR
/// reduction.  The schedule always leaves a non-empty terminal group, so
/// every intermediate group has exactly `k` buckets and the terminal group
/// has `1..=k` buckets.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SparseLwrReductionPlan {
    q_l: usize,
    bucket_count: usize,
    ring_dimension: usize,
    k: usize,
    intermediate_groups: usize,
    terminal_start: usize,
    terminal_len: usize,
    lut_width: usize,
}

impl SparseLwrReductionPlan {
    /// Derives the maximal valid cadence under the ring and 4096-width
    /// limits.  Width is computed from the largest possible unreduced sum in
    /// one group, `(k + 1) * (Q_L - 1)`, and must divide `N`.
    pub fn derive(
        q_l: usize,
        bucket_count: usize,
        ring_dimension: usize,
    ) -> Result<Self, PowerLutError> {
        if q_l == 0 || bucket_count == 0 || ring_dimension == 0 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let cap = ring_dimension.min(4096);
        let mut k = 1usize;
        let mut width = 0usize;
        // The width grows monotonically in k.  Search the finite integer
        // domain below the width cap and retain the maximal divisible one.
        let mut candidate = 1usize;
        // Keep one bucket for the terminal scan.  The cadence is therefore
        // maximal subject to both the LUT/ring constraints and `k < B`;
        // without this bound a large ring could absorb the entire batch into
        // one nominal group and silently erase the terminal transfer.
        let max_candidate = bucket_count.saturating_sub(1).max(1);
        while let Some(term) = candidate
            .checked_add(1)
            .and_then(|value| value.checked_mul(q_l.saturating_sub(1)))
            .and_then(|value| value.checked_add(1))
        {
            let Some(candidate_width) = term.checked_next_power_of_two() else { break };
            if candidate > max_candidate || candidate_width > cap {
                break;
            }
            if ring_dimension % candidate_width == 0 {
                k = candidate;
                width = candidate_width;
            }
            candidate = candidate.saturating_add(1);
        }
        if width == 0 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let intermediate_groups = (bucket_count - 1) / k;
        let terminal_start =
            intermediate_groups.checked_mul(k).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let terminal_len =
            bucket_count.checked_sub(terminal_start).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        if terminal_len == 0 || terminal_len > k {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Ok(Self {
            q_l,
            bucket_count,
            ring_dimension,
            k,
            intermediate_groups,
            terminal_start,
            terminal_len,
            lut_width: width,
        })
    }

    pub const fn q_l(&self) -> usize {
        self.q_l
    }
    pub const fn bucket_count(&self) -> usize {
        self.bucket_count
    }
    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    pub const fn k(&self) -> usize {
        self.k
    }
    pub const fn intermediate_groups(&self) -> usize {
        self.intermediate_groups
    }
    pub const fn terminal_start(&self) -> usize {
        self.terminal_start
    }
    pub const fn terminal_len(&self) -> usize {
        self.terminal_len
    }
    pub const fn lut_width(&self) -> usize {
        self.lut_width
    }
}

/// Private reduction helper set. The wrapper prevents accidental role swaps.
#[derive(Clone)]
pub struct SparseLwrReductionHelpers(FlatLutHelperSet);

/// Private terminal helper set. It is intentionally a distinct type from the
/// reduction set even though both contain flat helper branches.
#[derive(Clone)]
pub struct SparseLwrTerminalHelpers(FlatLutHelperSet);

impl SparseLwrReductionHelpers {
    pub fn new(set: FlatLutHelperSet) -> Self {
        Self(set)
    }
    fn as_set(&self) -> &FlatLutHelperSet {
        &self.0
    }
}
impl SparseLwrTerminalHelpers {
    pub fn new(set: FlatLutHelperSet) -> Self {
        Self(set)
    }
    fn as_set(&self) -> &FlatLutHelperSet {
        &self.0
    }
}

/// Role-typed helper material for the two reductions in a grouped PRF.
#[derive(Clone)]
pub struct SparseLwrPrfHelperBundle {
    pub reduction: SparseLwrReductionHelpers,
    pub terminal: SparseLwrTerminalHelpers,
}

/// Public reduction helper set.
#[derive(Clone)]
pub struct SparseLwrPublicReductionHelpers(FlatLutPublicHelperSet);

/// Public terminal helper set.
#[derive(Clone)]
pub struct SparseLwrPublicTerminalHelpers(FlatLutPublicHelperSet);

impl SparseLwrPublicReductionHelpers {
    pub fn new(set: FlatLutPublicHelperSet) -> Self {
        Self(set)
    }
    fn as_set(&self) -> &FlatLutPublicHelperSet {
        &self.0
    }
}
impl SparseLwrPublicTerminalHelpers {
    pub fn new(set: FlatLutPublicHelperSet) -> Self {
        Self(set)
    }
    fn as_set(&self) -> &FlatLutPublicHelperSet {
        &self.0
    }
}

/// Public-key counterpart of [`SparseLwrPrfHelperBundle`].
#[derive(Clone)]
pub struct SparseLwrPrfPublicHelperBundle {
    pub reduction: SparseLwrPublicReductionHelpers,
    pub terminal: SparseLwrPublicTerminalHelpers,
}

impl SparseLwrPrfProfile {
    /// Validates and creates a concrete sparse-LWR profile.
    pub fn new(
        q_l: usize,
        p: usize,
        lut_width: usize,
        ring_dimension: usize,
    ) -> Result<Self, PowerLutError> {
        // The bucket reduction and final rounding tables require a domain
        // wide enough for every residue modulo `q_l`.  Check this relation
        // with checked arithmetic so a malformed host parameter cannot wrap
        // into an apparently valid profile.
        let minimum_lut_width = q_l
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        if q_l == 0 ||
            p < 2 ||
            p > q_l ||
            lut_width == 0 ||
            !lut_width.is_power_of_two() ||
            ring_dimension == 0 ||
            lut_width > ring_dimension ||
            ring_dimension % lut_width != 0 ||
            q_l > lut_width ||
            lut_width < minimum_lut_width
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Ok(Self { q_l, p, lut_width, ring_dimension })
    }

    /// Plaintext modulus `Q_L`.
    pub const fn q_l(&self) -> usize {
        self.q_l
    }

    /// Output modulus `p`.
    pub const fn p(&self) -> usize {
        self.p
    }

    /// Logical LUT width `W`.
    pub const fn lut_width(&self) -> usize {
        self.lut_width
    }

    /// Concrete ring dimension `N`.
    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
}
/// The reusable program body for one sparse-LWR/PBC bucket update.
///
/// `input` is the accumulator wire supplied to the bucket body.  The
/// selector and public-value family identifiers are declarations in
/// `program`; they do not contain selector bits, support coordinates, GSW
/// ciphertexts, or artifact names.  `output` is the selected and LUT-mapped
/// bucket contribution.  The caller performs any sequential loop bookkeeping
/// around this body.  In particular, the selector is not represented as a
/// computed wire: it is an explicit family input at lowering time.
#[derive(Clone, Debug)]
pub struct SparseLwrBucketProgram {
    /// Canonical shared program consumed by both lowerers.
    pub program: PowerLutProgram,
    /// Program input carrying the current bucket accumulator.
    pub input: ProgramInputId,
    /// Hidden one-hot selector family declaration.
    pub selector_family: RhsFamilyId,
    /// Public encoded values for the rectangular bucket row.
    pub public_value_family: PublicValueFamilyId,
    /// Output wire containing the selected bucket value.
    pub output: ProgramWireId,
}

/// A complete sparse-LWR PRF program for one structural bucket iteration.
///
/// The selection program contains only the one-hot family selection. The
/// reduction program is a separate unary `z % Q_L` LUT applied once per full
/// group, while the terminal scalar LUT is applied once to the remainder.
#[derive(Clone, Debug)]
pub struct SparseLwrPrfProgram {
    /// Canonical program shared by private and public-key lowering.
    pub program: PowerLutProgram,
    /// Program input carrying the current bucket accumulator.
    pub input: ProgramInputId,
    /// Hidden selector family declaration.
    pub selector_family: RhsFamilyId,
    /// Public bucket-value family declaration.
    pub public_value_family: PublicValueFamilyId,
    /// Output wire of one bucket body. Lowering applies the mandatory final
    /// rounding table to the completed sequential state after this wire.
    pub output: ProgramWireId,
    /// Logical Power-LUT domain width `W`. This is independent of the number
    /// of cells in one rectangular PBC bucket.
    pub lut_width: usize,
    /// Public rectangular PBC bucket width. This controls family ranges, not
    /// the logical LUT domain.
    pub bucket_width: usize,
    pub bucket_count: usize,
    /// Concrete profile from which both mandatory LUTs were derived.
    profile: SparseLwrPrfProfile,
    /// Shared program for the final LWR rounding operation. Its output is a
    /// raw scalar terminal, not an ordinary monomial PRF value.
    pub rounding_program: PowerLutProgram,
    /// Composite identity of the bucket and final-rounding programs.
    composite_id: PowerLutProgramId,
    /// Input declaration used when invoking the final-rounding program.
    rounding_input: ProgramInputId,
    rounding_output: ProgramWireId,
    /// The final table is cached only as immutable lowering data. Its
    /// declaration is also present in `rounding_program`.
    rounding_lut: Vec<usize>,
    /// Immutable grouped cadence derived from the concrete profile and PBC
    /// bucket count.
    pub plan: SparseLwrReductionPlan,
    /// Selection-only program and unary intermediate-reduction program.
    pub selection_program: PowerLutProgram,
    pub reduction_program: PowerLutProgram,
    pub selection_input: ProgramInputId,
    pub selection_output: ProgramWireId,
    pub reduction_input: ProgramInputId,
    pub reduction_output: ProgramWireId,
}

#[allow(dead_code)]
#[allow(dead_code)]
#[allow(dead_code)]
impl SparseLwrPrfProgram {
    fn bind_output(&self, label: &[u8]) -> Result<SparseLwrPrfOutput, PowerLutError> {
        SparseLwrPrfOutput::bind_with_id(self.id(), label, self.rounding_output)
            .map_err(|_| PowerLutError::InvalidLut)
    }

    /// Returns the logical Power-LUT domain width `W`.
    pub const fn lut_width(&self) -> usize {
        self.lut_width
    }

    /// Returns the rectangular PBC bucket width used by family bindings.
    pub const fn bucket_width(&self) -> usize {
        self.bucket_width
    }

    /// Returns the validated concrete sparse-LWR profile.
    pub const fn profile(&self) -> &SparseLwrPrfProfile {
        &self.profile
    }

    /// Returns the ordinary shared program used for the final rounding step.
    pub const fn rounding_program(&self) -> &PowerLutProgram {
        &self.rounding_program
    }

    /// Returns the final-rounding program's declared input identifier.
    pub const fn rounding_input(&self) -> ProgramInputId {
        self.rounding_input
    }

    /// Returns the output wire of the scalar terminal program.
    ///
    /// Refresh setup commits this wire together with the composite program
    /// identity.  This prevents an output from an intermediate bucket body
    /// from being accepted as the raw-scalar PRF result.
    pub const fn terminal_output_wire(&self) -> ProgramWireId {
        self.rounding_output
    }

    /// Returns the immutable final rounding table derived from the profile.
    ///
    /// The table is exposed only for diagnostics and tests; lowering always
    /// executes the [`Self::rounding_program`] containing this same table.
    pub fn rounding_lut(&self) -> &[usize] {
        &self.rounding_lut
    }

    /// Returns the composite identity covering both ordinary programs.
    pub const fn id(&self) -> PowerLutProgramId {
        self.composite_id
    }

    /// Checks the complete batch/program binding before any lowering work is
    /// performed.  Keeping this in one predicate prevents one typed lowering
    /// entry point from accidentally accepting a batch for another program.
    fn validate_batch(&self, batch: &RefreshPrfBatchInputs) -> Result<(), PowerLutError> {
        if batch.layout_id != batch.layout.layout_id ||
            batch.profile != self.profile ||
            batch.program_id != self.id() ||
            batch.output_wire != self.terminal_output_wire()
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Ok(())
    }

    fn validate_widths(&self, ring_dimension: usize) -> Result<(), PowerLutError> {
        if self.profile.ring_dimension != ring_dimension ||
            self.lut_width == 0 ||
            !self.lut_width.is_power_of_two() ||
            self.lut_width > ring_dimension ||
            ring_dimension % self.lut_width != 0 ||
            self.bucket_width == 0
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Ok(())
    }

    /// Validates the profile against the concrete modulus and ring used by a
    /// lowering boundary. The profile itself has already checked all ordinary
    /// inequalities; this method additionally rejects a caller that tries to
    /// bind a different `Q_L` or ring dimension.
    pub fn validate_lwr_profile(
        &self,
        lwr_modulus: &mxx_ir_core::IntExpr,
        ring_dimension: usize,
    ) -> Result<(), PowerLutError> {
        self.validate_widths(ring_dimension)?;
        let modulus = lwr_modulus
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        if modulus != BigInt::from(self.profile.q_l) {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        if self.profile.q_l > self.profile.lut_width {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Ok(())
    }
}

impl SparseLwrPrfProgram {
    /// Lowers this program through the private encoding API and creates the
    /// typed output descriptor from the returned wire map.
    ///
    /// The caller supplies selector RHS packages and public bucket values as
    /// explicit family bindings. No support, schedule, or selected slot is
    /// copied into the program or inferred by the lowerer.
    fn lower_private_grouped(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        offsets: Vec<usize>,
        widths: Vec<usize>,
        selector_flat_count: usize,
        capacity: usize,
        invariants: Vec<Family<Mat>>,
        helpers: &FlatLutHelperMap,
        label: &[u8],
        rounding_helpers: &FlatLutHelperSet,
    ) -> Result<(BggEncodingWire, SparseLwrPrfOutput), PowerLutError> {
        self.validate_widths(concrete_ring_dimension(&input)?)?;
        let plan = &self.plan;
        let lowering_error = RefCell::new(None);
        let grouped_state = if plan.intermediate_groups() == 0 {
            input
        } else {
            Sequential::range(plan.intermediate_groups())
                .scan(input, invariants.clone(), |group, state, invariants| {
                    let grouped = Sequential::range(plan.k())
                        .scan(state, invariants, |bucket, state, mut flat| {
                            if flat.len() != selector_flat_count + 1 {
                                *lowering_error.borrow_mut() =
                                    Some(PowerLutError::InvalidSparseLwrBlock);
                                return Err(mxx_dsl::DslError::Schema);
                            }
                            let public_values = PowerLutMonomialFamily::from_trusted(
                                flat.pop().ok_or(mxx_dsl::DslError::Schema)?,
                                &compiler.bgg.public_key.ring,
                                PBC_MONOMIAL_FAMILY_PROVENANCE,
                            )
                            .map_err(|_| mxx_dsl::DslError::Schema)?;
                            let selectors =
                                EncodingSelectorFamily::from_flattened(flat).map_err(|error| {
                                    *lowering_error.borrow_mut() = Some(error);
                                    mxx_dsl::DslError::Schema
                                })?;
                            let index = mxx_ir_core::IntExpr::Add(
                                Box::new(mxx_ir_core::IntExpr::Mul(
                                    Box::new(mxx_ir_core::IntExpr::constant(plan.k())),
                                    Box::new(group.expression()),
                                )),
                                Box::new(bucket.expression()),
                            )
                            .canonicalize();
                            let start =
                                interpolate_lookup(&offsets, index.clone()).map_err(|e| {
                                    *lowering_error.borrow_mut() = Some(e);
                                    mxx_dsl::DslError::Schema
                                })?;
                            let count = interpolate_lookup(&widths, index).map_err(|e| {
                                *lowering_error.borrow_mut() = Some(e);
                                mxx_dsl::DslError::Schema
                            })?;
                            let range =
                                FamilyRange::bounded(start, count, capacity).map_err(|_| {
                                    *lowering_error.borrow_mut() =
                                        Some(PowerLutError::InvalidSparseLwrBlock);
                                    mxx_dsl::DslError::Schema
                                })?;
                            let mut ranges = ProgramFamilyRanges::new();
                            ranges.selector(self.selector_family, range.clone());
                            ranges.public_values(self.public_value_family, range);
                            let selected = compiler
                                .compile_program_with_ranges(
                                    &self.selection_program,
                                    &BTreeMap::from([(self.selection_input, state)]),
                                    &BTreeMap::new(),
                                    &BTreeMap::from([(self.selector_family, selectors)]),
                                    &BTreeMap::from([(self.public_value_family, public_values)]),
                                    &ranges,
                                    &BTreeMap::new(),
                                )
                                .map_err(|e| {
                                    *lowering_error.borrow_mut() = Some(e);
                                    mxx_dsl::DslError::Schema
                                })?
                                .remove(&self.selection_output)
                                .ok_or_else(|| {
                                    *lowering_error.borrow_mut() =
                                        Some(PowerLutError::InvalidSparseLwrBlock);
                                    mxx_dsl::DslError::Schema
                                })?;
                            Ok(selected)
                        })
                        .map_err(|_| mxx_dsl::DslError::Schema)?;
                    let reduced = compiler
                        .compile_program(
                            &self.reduction_program,
                            &BTreeMap::from([(self.reduction_input, grouped)]),
                            &BTreeMap::new(),
                            &BTreeMap::new(),
                            &BTreeMap::new(),
                            helpers,
                        )
                        .map_err(|e| {
                            *lowering_error.borrow_mut() = Some(e);
                            mxx_dsl::DslError::Schema
                        })?;
                    reduced.get(&self.reduction_output).cloned().ok_or_else(|| {
                        *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                        mxx_dsl::DslError::Schema
                    })
                })
                .map_err(|_| {
                    lowering_error
                        .borrow_mut()
                        .take()
                        .unwrap_or(PowerLutError::InvalidSparseLwrBlock)
                })?
        };
        let terminal_state = Sequential::range(plan.terminal_len())
            .scan(grouped_state, invariants, |bucket, state, mut flat| {
                if flat.len() != selector_flat_count + 1 {
                    return Err(mxx_dsl::DslError::Schema);
                }
                let public_values = PowerLutMonomialFamily::from_trusted(
                    flat.pop().ok_or(mxx_dsl::DslError::Schema)?,
                    &compiler.bgg.public_key.ring,
                    PBC_MONOMIAL_FAMILY_PROVENANCE,
                )
                .map_err(|_| mxx_dsl::DslError::Schema)?;
                let selectors = EncodingSelectorFamily::from_flattened(flat)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let index = mxx_ir_core::IntExpr::Add(
                    Box::new(mxx_ir_core::IntExpr::constant(plan.terminal_start())),
                    Box::new(bucket.expression()),
                )
                .canonicalize();
                let start = interpolate_lookup(&offsets, index.clone())
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let count =
                    interpolate_lookup(&widths, index).map_err(|_| mxx_dsl::DslError::Schema)?;
                let range = FamilyRange::bounded(start, count, capacity)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let mut ranges = ProgramFamilyRanges::new();
                ranges.selector(self.selector_family, range.clone());
                ranges.public_values(self.public_value_family, range);
                let selected = compiler
                    .compile_program_with_ranges(
                        &self.selection_program,
                        &BTreeMap::from([(self.selection_input, state)]),
                        &BTreeMap::new(),
                        &BTreeMap::from([(self.selector_family, selectors)]),
                        &BTreeMap::from([(self.public_value_family, public_values)]),
                        &ranges,
                        &BTreeMap::new(),
                    )
                    .map_err(|_| mxx_dsl::DslError::Schema)?
                    .remove(&self.selection_output)
                    .ok_or(mxx_dsl::DslError::Schema)?;
                Ok(selected)
            })
            .map_err(|_| {
                lowering_error.into_inner().unwrap_or(PowerLutError::InvalidSparseLwrBlock)
            })?;
        let output =
            self.apply_final_rounding_encoding(compiler, terminal_state, rounding_helpers)?;
        Ok((output, self.bind_output(label)?))
    }

    /// Retained only as an internal migration boundary.
    ///
    /// The binding closure can expose only public selector projections and
    /// public routed values.  It cannot receive a sparse support, schedule,
    /// selected slot, or private GSW family.  Both methods construct the same
    /// structural loop and invoke the same immutable program body.
    fn lower_public_grouped(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input: BggPublicKeyWire,
        offsets: Vec<usize>,
        widths: Vec<usize>,
        selector_flat_count: usize,
        capacity: usize,
        invariants: Vec<Family<Mat>>,
        helpers: &FlatLutPublicHelperMap,
        label: &[u8],
        rounding_helpers: &FlatLutPublicHelperSet,
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        self.validate_widths(concrete_ring_dimension_public(&input)?)?;
        let plan = &self.plan;
        let lowering_error = RefCell::new(None);
        let grouped_state = if plan.intermediate_groups() == 0 {
            input
        } else {
            Sequential::range(plan.intermediate_groups())
                .scan(input, invariants.clone(), |group, state, invariants| {
                    let grouped = Sequential::range(plan.k())
                        .scan(state, invariants, |bucket, state, mut flat| {
                            if flat.len() != selector_flat_count + 1 {
                                return Err(mxx_dsl::DslError::Schema);
                            }
                            let public_values = PowerLutMonomialFamily::from_trusted(
                                flat.pop().ok_or(mxx_dsl::DslError::Schema)?,
                                &compiler.public_key.ring,
                                PBC_MONOMIAL_FAMILY_PROVENANCE,
                            )
                            .map_err(|_| mxx_dsl::DslError::Schema)?;
                            let selectors = PublicSelectorFamily::from_flattened(flat)
                                .map_err(|_| mxx_dsl::DslError::Schema)?;
                            let index = mxx_ir_core::IntExpr::Add(
                                Box::new(mxx_ir_core::IntExpr::Mul(
                                    Box::new(mxx_ir_core::IntExpr::constant(plan.k())),
                                    Box::new(group.expression()),
                                )),
                                Box::new(bucket.expression()),
                            )
                            .canonicalize();
                            let start = interpolate_lookup(&offsets, index.clone())
                                .map_err(|_| mxx_dsl::DslError::Schema)?;
                            let count = interpolate_lookup(&widths, index)
                                .map_err(|_| mxx_dsl::DslError::Schema)?;
                            let range = FamilyRange::bounded(start, count, capacity)
                                .map_err(|_| mxx_dsl::DslError::Schema)?;
                            let mut ranges = ProgramFamilyRanges::new();
                            ranges.selector(self.selector_family, range.clone());
                            ranges.public_values(self.public_value_family, range);
                            let selected = compiler
                                .compile_program_with_ranges(
                                    &self.selection_program,
                                    &BTreeMap::from([(self.selection_input, state)]),
                                    &BTreeMap::new(),
                                    &BTreeMap::from([(self.selector_family, selectors)]),
                                    &BTreeMap::from([(self.public_value_family, public_values)]),
                                    &ranges,
                                    &BTreeMap::new(),
                                )
                                .map_err(|_| mxx_dsl::DslError::Schema)?
                                .remove(&self.selection_output)
                                .ok_or(mxx_dsl::DslError::Schema)?;
                            Ok(selected)
                        })
                        .map_err(|_| mxx_dsl::DslError::Schema)?;
                    let reduced = compiler
                        .compile_program(
                            &self.reduction_program,
                            &BTreeMap::from([(self.reduction_input, grouped)]),
                            &BTreeMap::new(),
                            &BTreeMap::new(),
                            &BTreeMap::new(),
                            helpers,
                        )
                        .map_err(|_| mxx_dsl::DslError::Schema)?;
                    reduced.get(&self.reduction_output).cloned().ok_or(mxx_dsl::DslError::Schema)
                })
                .map_err(|_| {
                    lowering_error
                        .borrow_mut()
                        .take()
                        .unwrap_or(PowerLutError::InvalidSparseLwrBlock)
                })?
        };
        let terminal_state = Sequential::range(plan.terminal_len())
            .scan(grouped_state, invariants, |bucket, state, mut flat| {
                if flat.len() != selector_flat_count + 1 {
                    return Err(mxx_dsl::DslError::Schema);
                }
                let public_values = PowerLutMonomialFamily::from_trusted(
                    flat.pop().ok_or(mxx_dsl::DslError::Schema)?,
                    &compiler.public_key.ring,
                    PBC_MONOMIAL_FAMILY_PROVENANCE,
                )
                .map_err(|_| mxx_dsl::DslError::Schema)?;
                let selectors = PublicSelectorFamily::from_flattened(flat)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let index = mxx_ir_core::IntExpr::Add(
                    Box::new(mxx_ir_core::IntExpr::constant(plan.terminal_start())),
                    Box::new(bucket.expression()),
                )
                .canonicalize();
                let start = interpolate_lookup(&offsets, index.clone())
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let count =
                    interpolate_lookup(&widths, index).map_err(|_| mxx_dsl::DslError::Schema)?;
                let range = FamilyRange::bounded(start, count, capacity)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let mut ranges = ProgramFamilyRanges::new();
                ranges.selector(self.selector_family, range.clone());
                ranges.public_values(self.public_value_family, range);
                compiler
                    .compile_program_with_ranges(
                        &self.selection_program,
                        &BTreeMap::from([(self.selection_input, state)]),
                        &BTreeMap::new(),
                        &BTreeMap::from([(self.selector_family, selectors)]),
                        &BTreeMap::from([(self.public_value_family, public_values)]),
                        &ranges,
                        &BTreeMap::new(),
                    )
                    .map_err(|_| mxx_dsl::DslError::Schema)?
                    .remove(&self.selection_output)
                    .ok_or(mxx_dsl::DslError::Schema)
            })
            .map_err(|_| {
                lowering_error.into_inner().unwrap_or(PowerLutError::InvalidSparseLwrBlock)
            })?;
        let output =
            self.apply_final_rounding_public(compiler, terminal_state, rounding_helpers)?;
        Ok((output, self.bind_output(label)?))
    }

    /// Batched PBC lowering with one explicit terminal rounding helper set.
    fn lower_private_batch(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &FlatLutHelperMap,
        rounding_helpers: &FlatLutHelperSet,
    ) -> Result<(Family<Mat>, Family<Mat>), PowerLutError> {
        self.lower_private_batch_impl(
            compiler,
            input_vectors,
            input_public_keys,
            batch,
            selectors,
            helpers,
            rounding_helpers,
        )
    }

    fn lower_private_batch_impl(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &FlatLutHelperMap,
        rounding_helpers: &FlatLutHelperSet,
    ) -> Result<(Family<Mat>, Family<Mat>), PowerLutError> {
        // For label `ell`, public values are
        // `(X^{a'_{\ell}[0]},...,X^{a'_{\ell}[m-1]})`. The nested bucket scan keeps
        // `X^acc` as state; OneHot selects C_i and updates it by
        // `X^{(acc+a'_{\ell}[i]) mod Q}` before applying the bucket LUT.
        let label_count = batch.len();
        if label_count == 0 ||
            input_vectors.count().evaluate(&mxx_ir_core::ParamEnv::default()).ok() !=
                Some(BigInt::from(label_count)) ||
            input_public_keys.count().evaluate(&mxx_ir_core::ParamEnv::default()).ok() !=
                Some(BigInt::from(label_count))
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let layout = &batch.layout;
        let public_values = batch_public_values_family(batch, &compiler.bgg.public_key.ring)?;
        let active = crate::pbc::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let widths = active.bucket_active_widths().collect::<Vec<_>>();
        let mut offsets = Vec::with_capacity(widths.len());
        let mut offset = 0usize;
        for width in widths.iter().copied() {
            offsets.push(offset);
            offset = offset.checked_add(width).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        }
        let active_count = active.len();
        let expected_values =
            label_count.checked_mul(active_count).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        ensure_family_count(public_values.as_family(), expected_values)?;

        let selector_flat = selectors.flattened();
        let selector_flat_count = selector_flat.len();
        let (helper_flat, helper_arities) = flatten_private_helper_families(helpers)?;
        let (rounding_flat, rounding_arities) =
            flatten_private_helper_families(&FlatLutHelperMap::from([(
                crate::program::LutId::from_index(0),
                rounding_helpers.clone(),
            )]))?;
        let mut broadcasts = selector_flat;
        broadcasts.push(public_values.into_family());
        let helper_flat_count = helper_flat.len();
        broadcasts.extend(helper_flat);
        let rounding_flat_count = rounding_flat.len();
        broadcasts.extend(rounding_flat);
        let lowering_error = Rc::new(RefCell::new(None));
        let lowering_error_for_loop = lowering_error.clone();
        let outputs = Family::<Mat>::parallel_zip_many_with_broadcast_values(
            vec![input_vectors, input_public_keys],
            broadcasts,
            move |label, mut inputs, mut broadcasts| {
                let helper_start = selector_flat_count + 1;
                if broadcasts.len() != helper_start + helper_flat_count + rounding_flat_count {
                    return Err(mxx_dsl::DslError::Schema);
                }
                let mut helper_broadcasts = broadcasts.split_off(helper_start);
                let rounding_broadcasts = helper_broadcasts.split_off(helper_flat_count);
                let label_values = broadcasts.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let selectors = EncodingSelectorFamily::from_flattened(broadcasts)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let helpers = rebuild_private_helper_families(helper_broadcasts, &helper_arities)?;
                let rounding_map =
                    rebuild_private_helper_families(rounding_broadcasts, &rounding_arities)?;
                let rounding_helpers = rounding_map
                    .get(&crate::program::LutId::from_index(0))
                    .ok_or(mxx_dsl::DslError::Schema)?;
                let public_offset = mxx_ir_core::IntExpr::Mul(
                    Box::new(label.expression()),
                    Box::new(mxx_ir_core::IntExpr::constant(active_count)),
                )
                .canonicalize();
                let label_start = Int::evaluate(public_offset);
                let label_indices = Parallel::range(active_count)
                    .map_values(|index| label_start.clone().add(index.as_int()))
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let label_values = label_values
                    .parallel_gather(label_indices)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let public_key = inputs.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let vector = inputs.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let input = BggEncodingWire {
                    vector,
                    pubkey: BggPublicKeyWire { matrix: public_key, reveal_plaintext: false },
                    plaintext: None,
                };
                let (output, _) = self
                    .lower_private_grouped(
                        compiler,
                        input,
                        offsets,
                        widths,
                        selector_flat_count,
                        layout.bucket_width,
                        {
                            let mut invariants = selectors.flattened();
                            invariants.push(label_values);
                            invariants
                        },
                        &helpers,
                        &[],
                        rounding_helpers,
                    )
                    .map_err(|error| {
                        *lowering_error_for_loop.borrow_mut() = Some(error);
                        mxx_dsl::DslError::Schema
                    })?;
                Ok((output.vector, output.pubkey.matrix))
            },
        )
        .map_err(|_| lowering_error.take().unwrap_or(PowerLutError::InvalidSparseLwrBlock))?;
        Ok(outputs)
    }

    /// Typed batched private lowering with the setup-fixed terminal rounding
    /// helpers.  The resulting family has the scalar terminal operation
    /// already executed in the structural label loop.
    fn lower_private_batch_typed(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &FlatLutHelperMap,
        rounding_helpers: &FlatLutHelperSet,
    ) -> Result<PbcSparseLwrEncodingOutputs, PowerLutError> {
        self.validate_batch(batch)?;
        let (vectors, public_keys) = self.lower_private_batch(
            compiler,
            input_vectors,
            input_public_keys,
            batch,
            selectors,
            helpers,
            rounding_helpers,
        )?;
        let reduction_helper_commitment = helpers
            .get(&crate::program::LutId::from_index(0))
            .ok_or(PowerLutError::InvalidLut)?
            .metadata()
            .0;
        let terminal_helper_commitment = rounding_helpers.metadata().0;
        Ok(PbcSparseLwrEncodingOutputs::new(
            vectors,
            public_keys,
            self.id(),
            self.rounding_output,
            batch,
            reduction_helper_commitment,
            terminal_helper_commitment,
        ))
    }

    /// Typed batched private lowering with an explicit role-typed helper
    /// bundle.  Reduction and terminal tables are intentionally separate
    /// even though both programs use local LUT identifier zero.
    pub fn compile_pbc_encoding_family_typed_with_batch_and_helpers(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &SparseLwrPrfHelperBundle,
    ) -> Result<PbcSparseLwrEncodingOutputs, PowerLutError> {
        self.lower_private_batch_typed(
            compiler,
            input_vectors,
            input_public_keys,
            batch,
            selectors,
            &BTreeMap::from([(
                crate::program::LutId::from_index(0),
                helpers.reduction.as_set().clone(),
            )]),
            helpers.terminal.as_set(),
        )
    }

    /// Public batched PBC lowering with explicit terminal rounding helpers.
    fn lower_public_batch(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &FlatLutPublicHelperMap,
        rounding_helpers: &FlatLutPublicHelperSet,
    ) -> Result<Family<Mat>, PowerLutError> {
        self.lower_public_batch_impl(
            compiler,
            input_keys,
            batch,
            selectors,
            helpers,
            rounding_helpers,
        )
    }

    fn lower_public_batch_impl(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &FlatLutPublicHelperMap,
        rounding_helpers: &FlatLutPublicHelperSet,
    ) -> Result<Family<Mat>, PowerLutError> {
        let label_count = batch.len();
        if label_count == 0 ||
            input_keys.count().evaluate(&mxx_ir_core::ParamEnv::default()).ok() !=
                Some(BigInt::from(label_count))
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let layout = &batch.layout;
        let public_values = batch_public_values_family(batch, &compiler.public_key.ring)?;
        let active = crate::pbc::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let widths = active.bucket_active_widths().collect::<Vec<_>>();
        let mut offsets = Vec::with_capacity(widths.len());
        let mut offset = 0usize;
        for width in widths.iter().copied() {
            offsets.push(offset);
            offset = offset.checked_add(width).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        }
        let active_count = active.len();
        let expected_values =
            label_count.checked_mul(active_count).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        ensure_family_count(public_values.as_family(), expected_values)?;
        let selector_flat = selectors.flattened();
        let selector_flat_count = selector_flat.len();
        let (helper_flat, helper_arities) = flatten_public_helper_families(helpers)?;
        let (rounding_flat, rounding_arities) =
            flatten_public_helper_families(&FlatLutPublicHelperMap::from([(
                crate::program::LutId::from_index(0),
                rounding_helpers.clone(),
            )]))?;
        let mut broadcasts = selector_flat;
        broadcasts.push(public_values.into_family());
        let helper_flat_count = helper_flat.len();
        broadcasts.extend(helper_flat);
        let rounding_flat_count = rounding_flat.len();
        broadcasts.extend(rounding_flat);
        let lowering_error = Rc::new(RefCell::new(None));
        let lowering_error_for_loop = lowering_error.clone();
        let outputs = Family::<Mat>::parallel_zip_many_with_broadcast_values(
            vec![input_keys],
            broadcasts,
            move |label, mut inputs, mut broadcasts| {
                let helper_start = selector_flat_count + 1;
                if broadcasts.len() != helper_start + helper_flat_count + rounding_flat_count {
                    return Err(mxx_dsl::DslError::Schema);
                }
                let mut helper_broadcasts = broadcasts.split_off(helper_start);
                let rounding_broadcasts = helper_broadcasts.split_off(helper_flat_count);
                let label_values = broadcasts.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let selectors = PublicSelectorFamily::from_flattened(broadcasts)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let helpers = rebuild_public_helper_families(helper_broadcasts, &helper_arities)?;
                let rounding_map =
                    rebuild_public_helper_families(rounding_broadcasts, &rounding_arities)?;
                let rounding_helpers = rounding_map
                    .get(&crate::program::LutId::from_index(0))
                    .ok_or(mxx_dsl::DslError::Schema)?;
                let public_offset = mxx_ir_core::IntExpr::Mul(
                    Box::new(label.expression()),
                    Box::new(mxx_ir_core::IntExpr::constant(active_count)),
                )
                .canonicalize();
                let label_start = Int::evaluate(public_offset);
                let label_indices = Parallel::range(active_count)
                    .map_values(|index| label_start.clone().add(index.as_int()))
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let label_values = label_values
                    .parallel_gather(label_indices)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let public_key = inputs.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let input = BggPublicKeyWire { matrix: public_key, reveal_plaintext: false };
                let (output, _) = self
                    .lower_public_grouped(
                        compiler,
                        input,
                        offsets,
                        widths,
                        selector_flat_count,
                        layout.bucket_width,
                        {
                            let mut invariants = selectors.flattened();
                            invariants.push(label_values);
                            invariants
                        },
                        &helpers,
                        &[],
                        rounding_helpers,
                    )
                    .map_err(|error| {
                        *lowering_error_for_loop.borrow_mut() = Some(error);
                        mxx_dsl::DslError::Schema
                    })?;
                Ok(output.matrix)
            },
        )
        .map_err(|_| lowering_error.take().unwrap_or(PowerLutError::InvalidSparseLwrBlock))?;
        Ok(outputs)
    }

    /// Typed public batched lowering with the explicit terminal rounding
    /// helper family supplied by setup.
    fn lower_public_batch_typed(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &FlatLutPublicHelperMap,
        rounding_helpers: &FlatLutPublicHelperSet,
    ) -> Result<Family<Mat>, PowerLutError> {
        self.validate_batch(batch)?;
        self.lower_public_batch(compiler, input_keys, batch, selectors, helpers, rounding_helpers)
    }

    /// Typed batched public-key lowering with role-typed helper material.
    pub fn compile_pbc_public_key_family_with_batch_and_helpers(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &SparseLwrPrfPublicHelperBundle,
    ) -> Result<Family<Mat>, PowerLutError> {
        self.lower_public_batch_typed(
            compiler,
            input_keys,
            batch,
            selectors,
            &BTreeMap::from([(
                crate::program::LutId::from_index(0),
                helpers.reduction.as_set().clone(),
            )]),
            helpers.terminal.as_set(),
        )
    }

    /// Applies the mandatory LWR rounding table once to the completed bucket
    /// accumulator. Keeping this after the sequential scan is essential:
    /// rounding each bucket would change `floor(p * sum(z_b) / Q_L)` into a
    /// sum of rounded partial values.
    fn apply_final_rounding_encoding(
        &self,
        compiler: &PowerLutEncodingCompiler,
        state: BggEncodingWire,
        rounding_helpers: &FlatLutHelperSet,
    ) -> Result<BggEncodingWire, PowerLutError> {
        let table = self
            .rounding_program
            .lut(crate::program::LutId::from_index(0))
            .ok_or(PowerLutError::InvalidLut)?;
        let helpers = FlatLutHelperMap::from([(
            crate::program::LutId::from_index(0),
            rounding_helpers.clone(),
        )]);
        rounding_helpers.resolve(table)?;
        let outputs = compiler.compile_program(
            &self.rounding_program,
            &BTreeMap::from([(self.rounding_input, state)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &helpers,
        )?;
        outputs.get(&self.rounding_output).cloned().ok_or(PowerLutError::InvalidLut)
    }

    /// Public-key counterpart of [`Self::apply_final_rounding_encoding`].
    fn apply_final_rounding_public(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        state: BggPublicKeyWire,
        rounding_helpers: &FlatLutPublicHelperSet,
    ) -> Result<BggPublicKeyWire, PowerLutError> {
        let table = self
            .rounding_program
            .lut(crate::program::LutId::from_index(0))
            .ok_or(PowerLutError::InvalidLut)?;
        let helpers = FlatLutPublicHelperMap::from([(
            crate::program::LutId::from_index(0),
            rounding_helpers.clone(),
        )]);
        rounding_helpers.resolve(table)?;
        let outputs = compiler.compile_program(
            &self.rounding_program,
            &BTreeMap::from([(self.rounding_input, state)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &helpers,
        )?;
        outputs.get(&self.rounding_output).cloned().ok_or(PowerLutError::InvalidLut)
    }
}

fn concrete_ring_dimension(input: &BggEncodingWire) -> Result<usize, PowerLutError> {
    input
        .pubkey
        .matrix
        .matrix_type()
        .ring_dimension
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(PowerLutError::InvalidSparseLwrBlock)
}

fn concrete_ring_dimension_public(input: &BggPublicKeyWire) -> Result<usize, PowerLutError> {
    input
        .matrix
        .matrix_type()
        .ring_dimension
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(PowerLutError::InvalidSparseLwrBlock)
}

fn ensure_family_count(family: &Family<Mat>, expected: usize) -> Result<(), PowerLutError> {
    let count = family
        .count()
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
    if count != BigInt::from(expected) {
        return Err(PowerLutError::InvalidSparseLwrBlock);
    }
    Ok(())
}

/// Lifts imported scalar helper artifacts into singleton families so an
/// outer label loop can receive every executable graph value explicitly.
fn flatten_private_helper_families(
    helpers: &FlatLutHelperMap,
) -> Result<(Vec<Family<Mat>>, Vec<(crate::program::LutId, [u8; 32], Vec<usize>)>), PowerLutError> {
    let mut flat = Vec::new();
    let mut arities = Vec::with_capacity(helpers.len());
    for (lut, lut_helpers) in helpers {
        let (commitment, _width) = lut_helpers.metadata();
        arities.push((*lut, commitment, lut_helpers.iter().map(|h| h.sigma()).collect()));
        for helper in lut_helpers.iter() {
            for matrix in [
                helper.switch().gsw_ciphertext().clone(),
                helper.mask().vector.clone(),
                helper.mask().pubkey.matrix.clone(),
            ] {
                flat.push(Family::pack(vec![matrix]).map_err(|_| PowerLutError::InvalidLut)?);
            }
        }
    }
    Ok((flat, arities))
}

fn rebuild_private_helper_families(
    flat: Vec<Family<Mat>>,
    arities: &[(crate::program::LutId, [u8; 32], Vec<usize>)],
) -> Result<FlatLutHelperMap, mxx_dsl::DslError> {
    let expected = arities.iter().try_fold(0usize, |total, (_, _, sigmas)| {
        total
            .checked_add(sigmas.len().checked_mul(3).ok_or(mxx_dsl::DslError::Schema)?)
            .ok_or(mxx_dsl::DslError::Schema)
    })?;
    if flat.len() != expected {
        return Err(mxx_dsl::DslError::Schema);
    }
    let mut cursor = 0usize;
    let mut rebuilt = FlatLutHelperMap::new();
    for (lut, commitment, sigmas) in arities.iter() {
        let mut lut_helpers = Vec::with_capacity(sigmas.len());
        for sigma in sigmas {
            let take = |cursor: &mut usize| -> Result<Mat, mxx_dsl::DslError> {
                let family = flat.get(*cursor).ok_or(mxx_dsl::DslError::Schema)?;
                *cursor += 1;
                Ok(family.get_static(0))
            };
            let gsw = take(&mut cursor)?;
            let mask_vector = take(&mut cursor)?;
            let mask_public = take(&mut cursor)?;
            let switch = PowerRhsPackage::new(gsw).map_err(|_| mxx_dsl::DslError::Schema)?;
            let mask = BggEncodingWire {
                vector: mask_vector,
                pubkey: BggPublicKeyWire { matrix: mask_public, reveal_plaintext: false },
                plaintext: None,
            };
            // The loop preserves the imported helper order; indexes are the
            // trusted metadata attached to those artifacts, not loop-derived data.
            let helper =
                FlatLutHelper::new(*sigma, switch, mask).map_err(|_| mxx_dsl::DslError::Schema)?;
            lut_helpers.push(helper);
        }
        let set = FlatLutHelperSet::from_parts(*commitment, sigmas.len(), lut_helpers)
            .map_err(|_| mxx_dsl::DslError::Schema)?;
        rebuilt.insert(*lut, set);
    }
    Ok(rebuilt)
}

fn flatten_public_helper_families(
    helpers: &FlatLutPublicHelperMap,
) -> Result<(Vec<Family<Mat>>, Vec<(crate::program::LutId, [u8; 32], Vec<usize>)>), PowerLutError> {
    let mut flat = Vec::new();
    let mut arities = Vec::with_capacity(helpers.len());
    for (lut, lut_helpers) in helpers {
        // Public and private helper sets use the same table commitment. The
        // public set exposes only the fixed matrices, never plaintext data.
        let commitment = lut_helpers.table_commitment();
        arities.push((*lut, commitment, lut_helpers.iter().map(|h| h.sigma()).collect()));
        for helper in lut_helpers.iter() {
            for matrix in [helper.switch().gsw_ciphertext().clone(), helper.mask().clone()] {
                flat.push(Family::pack(vec![matrix]).map_err(|_| PowerLutError::InvalidLut)?);
            }
        }
    }
    Ok((flat, arities))
}

fn rebuild_public_helper_families(
    flat: Vec<Family<Mat>>,
    arities: &[(crate::program::LutId, [u8; 32], Vec<usize>)],
) -> Result<FlatLutPublicHelperMap, mxx_dsl::DslError> {
    let expected = arities.iter().try_fold(0usize, |total, (_, _, sigmas)| {
        total
            .checked_add(sigmas.len().checked_mul(2).ok_or(mxx_dsl::DslError::Schema)?)
            .ok_or(mxx_dsl::DslError::Schema)
    })?;
    if flat.len() != expected {
        return Err(mxx_dsl::DslError::Schema);
    }
    let mut cursor = 0usize;
    let mut rebuilt = FlatLutPublicHelperMap::new();
    for (lut, commitment, sigmas) in arities.iter() {
        let mut lut_helpers = Vec::with_capacity(sigmas.len());
        for sigma in sigmas {
            let take = |cursor: &mut usize| -> Result<Mat, mxx_dsl::DslError> {
                let family = flat.get(*cursor).ok_or(mxx_dsl::DslError::Schema)?;
                *cursor += 1;
                Ok(family.get_static(0))
            };
            let switch =
                PowerRhsPackage::new(take(&mut cursor)?).map_err(|_| mxx_dsl::DslError::Schema)?;
            let mask = take(&mut cursor)?;
            lut_helpers.push(FlatLutPublicHelper::new(*sigma, switch, mask));
        }
        let set = FlatLutPublicHelperSet::from_parts(*commitment, sigmas.len(), lut_helpers)
            .map_err(|_| mxx_dsl::DslError::Schema)?;
        rebuilt.insert(*lut, set);
    }
    Ok(rebuilt)
}

/// Builds an exact integer expression for a finite lookup table over
/// `x = 0..values.len()-1` using Newton's forward formula.  Every division is
/// exact at those integer points; `IntExpr` evaluation retains that invariant
/// and rejects any accidental inexact division.
fn interpolate_lookup(
    values: &[usize],
    x: mxx_ir_core::IntExpr,
) -> Result<mxx_ir_core::IntExpr, PowerLutError> {
    if values.is_empty() {
        return Err(PowerLutError::InvalidSparseLwrBlock);
    }
    let mut differences = values.iter().map(|&value| BigInt::from(value)).collect::<Vec<_>>();
    let mut factorial = BigInt::from(1u8);
    let mut falling = mxx_ir_core::IntExpr::constant(1u8);
    let mut result = mxx_ir_core::IntExpr::constant(0u8);
    for degree in 0..values.len() {
        let coefficient = mxx_ir_core::IntExpr::constant(differences[0].clone());
        let numerator = mxx_ir_core::IntExpr::Mul(Box::new(coefficient), Box::new(falling.clone()));
        let term = mxx_ir_core::IntExpr::Div(
            Box::new(numerator),
            Box::new(mxx_ir_core::IntExpr::constant(factorial.clone())),
        );
        result = mxx_ir_core::IntExpr::Add(Box::new(result), Box::new(term)).canonicalize();
        if degree + 1 == values.len() {
            break;
        }
        differences =
            differences.windows(2).map(|pair| pair[1].clone() - pair[0].clone()).collect();
        factorial *= BigInt::from(degree + 1);
        let next = mxx_ir_core::IntExpr::Sub(
            Box::new(x.clone()),
            Box::new(mxx_ir_core::IntExpr::constant(degree)),
        );
        falling = mxx_ir_core::IntExpr::Mul(Box::new(falling), Box::new(next)).canonicalize();
    }
    Ok(result.canonicalize())
}

/// Builds the reusable one-hot bucket body. The bucket body is selection-only;
/// the sole non-identity LUT is the terminal scalar rounding table.
///
/// The selector and public-value families stay as runtime inputs; this helper
/// only declares the structural program shared by every bucket.
fn build_bucket_program(
    lut_width: usize,
    bucket_width: usize,
) -> Result<SparseLwrBucketProgram, ProgramValidationError> {
    if lut_width == 0 || !lut_width.is_power_of_two() || bucket_width == 0 {
        return Err(ProgramValidationError::WidthMismatch);
    }
    let mut builder = PowerLutProgramBuilder::new();
    let input = builder.input(lut_width)?;
    let selector_family = builder.rhs_family(lut_width)?;
    let public_value_family = builder.public_value_family(lut_width)?;
    let input_wire = builder.input_wire(input)?;
    let output = builder.one_hot_select(input_wire, selector_family, public_value_family)?;
    builder.output(output)?;
    let program = builder.build()?;
    Ok(SparseLwrBucketProgram { program, input, selector_family, public_value_family, output })
}

/// Builds the unary monomial reduction `z -> z mod Q_L`.
fn build_reduction_program(
    lut_width: usize,
    q_l: usize,
) -> Result<(PowerLutProgram, ProgramInputId, ProgramWireId), ProgramValidationError> {
    let mut builder = PowerLutProgramBuilder::new();
    let input = builder.input(lut_width)?;
    let input_wire = builder.input_wire(input)?;
    let table = builder.lut(LutTable::unary(
        lut_width,
        lut_width,
        (0..lut_width).map(|z| z % q_l).collect(),
    )?)?;
    let output = builder.unary(input_wire, table)?;
    builder.output(output)?;
    Ok((builder.build()?, input, output))
}

/// Builds the final scalar LUT that rounds a reduced sparse-LWR value.
fn build_rounding_program(
    lut_width: usize,
    output_width: usize,
    rounding_lut: &[usize],
) -> Result<(PowerLutProgram, ProgramInputId, ProgramWireId), ProgramValidationError> {
    let mut builder = PowerLutProgramBuilder::new();
    let input = builder.input(lut_width)?;
    let input_wire = builder.input_wire(input)?;
    // Rounding produces a scalar digit, not a ring monomial.  The terminal
    // output form is committed in the program and disallows further gates.
    let lut =
        builder.lut(LutTable::unary_scalar(lut_width, output_width, rounding_lut.to_vec())?)?;
    let output = builder.unary(input_wire, lut)?;
    builder.output(output)?;
    Ok((builder.build()?, input, output))
}

/// Derives the PRF identity from its profile and both canonical subprograms.
///
/// Including both program identities prevents a change to either the bucket
/// reduction or final rounding table from being treated as the same output
/// contract.
fn composite_prf_id(
    profile: &SparseLwrPrfProfile,
    plan: &SparseLwrReductionPlan,
    bucket_program: &PowerLutProgram,
    reduction_program: &PowerLutProgram,
    rounding_program: &PowerLutProgram,
) -> PowerLutProgramId {
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/sparse-lwr/prf-program/v4-grouped-reduction");
    digest.update((profile.q_l as u64).to_le_bytes());
    digest.update((profile.p as u64).to_le_bytes());
    digest.update((profile.lut_width as u64).to_le_bytes());
    digest.update((profile.ring_dimension as u64).to_le_bytes());
    digest.update(bucket_program.id().as_bytes());
    digest.update(reduction_program.id().as_bytes());
    digest.update((plan.k as u64).to_le_bytes());
    digest.update((plan.intermediate_groups as u64).to_le_bytes());
    digest.update((plan.terminal_start as u64).to_le_bytes());
    digest.update((plan.terminal_len as u64).to_le_bytes());
    digest.update((plan.lut_width as u64).to_le_bytes());
    digest.update(rounding_program.id().as_bytes());
    PowerLutProgramId::from_digest(digest.finalize().into())
}

impl SparseLwrPrfProgram {
    /// Constructs the selection, grouped reduction, and terminal programs.
    ///
    /// The PBC `bucket_width` is only a rectangular family shape. It is not a
    /// LUT domain and never determines either table. The reduction table is
    /// `z mod Q_L`; the final table is `floor(p * (z mod Q_L) / Q_L)` and is
    /// applied exactly once after the sequential bucket scan.
    pub fn new(
        profile: SparseLwrPrfProfile,
        bucket_width: usize,
        bucket_count: usize,
    ) -> Result<Self, PowerLutError> {
        if bucket_width == 0 || bucket_count == 0 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let plan =
            SparseLwrReductionPlan::derive(profile.q_l, bucket_count, profile.ring_dimension)?;
        let w = plan.lut_width;
        let rounding_lut = (0..w)
            .map(|z| {
                profile
                    .p
                    .checked_mul(z % profile.q_l)
                    .map(|value| value / profile.q_l)
                    .ok_or(PowerLutError::InvalidSparseLwrBlock)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bucket = build_bucket_program(w, bucket_width).map_err(PowerLutError::from)?;
        let (reduction_program, reduction_input, reduction_output) =
            build_reduction_program(w, profile.q_l).map_err(PowerLutError::from)?;
        let (rounding_program, rounding_input, rounding_output) =
            build_rounding_program(w, profile.p, &rounding_lut).map_err(PowerLutError::from)?;
        let composite_id = composite_prf_id(
            &profile,
            &plan,
            &bucket.program,
            &reduction_program,
            &rounding_program,
        );
        let selection_program = bucket.program.clone();
        Ok(Self {
            program: bucket.program,
            input: bucket.input,
            selector_family: bucket.selector_family,
            public_value_family: bucket.public_value_family,
            output: bucket.output,
            lut_width: w,
            bucket_width,
            bucket_count,
            profile,
            rounding_program,
            composite_id,
            rounding_input,
            rounding_output,
            rounding_lut,
            plan,
            selection_program,
            reduction_program,
            selection_input: bucket.input,
            selection_output: bucket.output,
            reduction_input,
            reduction_output,
        })
    }
}

/// Algebraic contract of a sparse-LWR PRF descriptor.
///
/// Ordinary Power-LUT gates return monomial encodings. The sparse-LWR
/// rounding terminal is intentionally different: it returns the integer
/// rounding value as a constant ring polynomial. Keeping this distinction in
/// the descriptor prevents a caller from treating a raw scalar as a normal
/// monomial PRF output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SparseLwrPrfTerminalForm {
    /// A conventional PRF value represented by the monomial `X^v`.
    Monomial,
    /// The sparse-LWR terminal represented by the constant `v`.
    RawScalar,
}

/// Typed identity of one sparse-LWR PRF result.
///
/// The value itself is produced by a lowerer. This descriptor binds the result
/// to the exact composite program, public label domain, and final output wire.
/// It intentionally stores no sparse support, selected slot, selector bit, or
/// private RHS material.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseLwrPrfOutput {
    program_id: PowerLutProgramId,
    label_digest: [u8; 32],
    output_wire: ProgramWireId,
    terminal_form: SparseLwrPrfTerminalForm,
}

/// Opaque family result of one structural PBC lowering.
///
/// The family is produced by
/// [`SparseLwrPrfProgram::compile_pbc_encoding_family_typed_with_batch_and_helpers`]
/// and remains in canonical label order. Production refresh consumes these sealed
/// families directly; no per-label projection API exists.
pub struct PbcSparseLwrEncodingOutputs {
    vectors: Family<Mat>,
    public_keys: Family<Mat>,
    program_id: PowerLutProgramId,
    output_wire: ProgramWireId,
    layout_id: crate::pbc::PbcLayoutId,
    batch_id: PbcPublicValueBatchId,
    batch_count: usize,
    family_pair_id: [u8; 32],
    reduction_helper_commitment: [u8; 32],
    terminal_helper_commitment: [u8; 32],
}

impl PbcSparseLwrEncodingOutputs {
    fn new(
        vectors: Family<Mat>,
        public_keys: Family<Mat>,
        program_id: PowerLutProgramId,
        output_wire: ProgramWireId,
        batch: &RefreshPrfBatchInputs,
        reduction_helper_commitment: [u8; 32],
        terminal_helper_commitment: [u8; 32],
    ) -> Self {
        let mut pair_digest = Sha256::new();
        pair_digest.update(b"mxx-power-lut/pbc-output-family-pair/v1");
        pair_digest.update(batch.batch_id.0);
        pair_digest.update(batch.layout_id.0);
        pair_digest.update(program_id.0);
        pair_digest.update(format!("{output_wire:?}").as_bytes());
        pair_digest.update(reduction_helper_commitment);
        pair_digest.update(terminal_helper_commitment);
        let family_pair_id = pair_digest.finalize().into();
        Self {
            vectors,
            public_keys,
            program_id,
            output_wire,
            layout_id: batch.layout_id,
            batch_id: batch.batch_id,
            batch_count: batch.len(),
            family_pair_id,
            reduction_helper_commitment,
            terminal_helper_commitment,
        }
    }

    /// Borrows the complete lowered vector family. The family remains sealed
    /// to the canonical batch order; callers must not project it into a
    /// label-sized host collection.
    pub(crate) fn vectors(&self) -> &Family<Mat> {
        &self.vectors
    }

    /// Borrows the complete lowered public-key family in the same canonical
    /// batch order as [`Self::vectors`].
    pub(crate) fn public_keys(&self) -> &Family<Mat> {
        &self.public_keys
    }

    /// Returns the family-level batch metadata needed by a typed consumer.
    pub(crate) fn family_metadata(
        &self,
    ) -> (PowerLutProgramId, ProgramWireId, crate::pbc::PbcLayoutId, usize) {
        (self.program_id, self.output_wire, self.layout_id, self.batch_count)
    }

    /// Returns the immutable public-value batch identity bound to this family.
    pub(crate) const fn batch_id(&self) -> PbcPublicValueBatchId {
        self.batch_id
    }

    pub(crate) const fn family_pair_id(&self) -> [u8; 32] {
        self.family_pair_id
    }

    pub const fn helper_commitments(&self) -> ([u8; 32], [u8; 32]) {
        (self.reduction_helper_commitment, self.terminal_helper_commitment)
    }
}

impl SparseLwrPrfOutput {
    fn bind_with_id(
        program_id: PowerLutProgramId,
        label: &[u8],
        output_wire: ProgramWireId,
    ) -> Result<Self, PowerLutError> {
        let mut digest = Sha256::new();
        digest.update(b"mxx-power-lut/sparse-lwr/prf-label/v1");
        digest.update((label.len() as u64).to_le_bytes());
        digest.update(label);
        Ok(Self {
            program_id,
            label_digest: digest.finalize().into(),
            output_wire,
            terminal_form: SparseLwrPrfTerminalForm::RawScalar,
        })
    }

    /// Returns the shared program identity.
    pub const fn program_id(&self) -> PowerLutProgramId {
        self.program_id
    }

    /// Returns the domain-separated public label digest.
    pub const fn label_digest(&self) -> &[u8; 32] {
        &self.label_digest
    }

    /// Returns the final-rounding program's declared output wire.
    pub const fn output_wire(&self) -> ProgramWireId {
        self.output_wire
    }

    /// Returns the algebraic form promised by this descriptor.
    pub const fn terminal_form(&self) -> SparseLwrPrfTerminalForm {
        self.terminal_form
    }

    /// Returns whether this is the raw-scalar sparse-LWR terminal contract.
    pub const fn is_raw_scalar(&self) -> bool {
        matches!(self.terminal_form, SparseLwrPrfTerminalForm::RawScalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_bgg::BggPublicKeyCompiler;
    use mxx_dsl::{DslContext, Ring};

    fn batch_fixture() -> (
        crate::pbc::PbcPublicLayout,
        SparseLwrPrfProfile,
        SparseLwrPrfProgram,
        crate::refresh::RefreshPrfLabelIndex,
    ) {
        let parameters = crate::pbc::PbcParameters::custom(1, 1, 2, 2, 1, None);
        let generated =
            crate::pbc::generate_key_layout(&parameters, crate::pbc::PbcRootSeed([0x41; 32]), &[0])
                .expect("small PBC fixture layout");
        let profile = SparseLwrPrfProfile::new(2, 2, 4, 8).expect("test profile");
        let program = SparseLwrPrfProgram::new(
            profile.clone(),
            generated.public_layout.bucket_width,
            generated.public_layout.parameters.bucket_count,
        )
        .expect("test PRF program");
        let labels = crate::refresh::RefreshPrfLabelIndex::new([0x52; 32], 1, 2, 1, 1, 1)
            .expect("test label index");
        (generated.public_layout, profile, program, labels)
    }

    #[test]
    fn selection_and_reduction_are_separate_programs() {
        let program = SparseLwrPrfProgram::new(
            SparseLwrPrfProfile::new(2, 2, 4, 4).expect("test profile"),
            1,
            1,
        )
        .expect("test PRF program");

        assert!(program.selection_program.lut(crate::program::LutId::from_index(0)).is_none());
        assert_eq!(program.selection_program.gates().len(), 1);
        assert!(matches!(
            program.selection_program.gates()[0],
            crate::program::ProgramGate::OneHot { .. }
        ));
        assert_eq!(program.reduction_program.gates().len(), 1);
        assert!(program.reduction_program.lut(crate::program::LutId::from_index(0)).is_some());
    }

    #[test]
    fn grouped_plan_q512_b32_is_maximal_and_reserves_terminal_group() {
        let plan = SparseLwrReductionPlan::derive(512, 32, 4096).expect("grouped plan");
        assert_eq!(plan.k(), 7);
        assert_eq!(plan.intermediate_groups(), 4);
        assert_eq!(plan.terminal_start(), 28);
        assert_eq!(plan.terminal_len(), 4);
        assert_eq!(plan.lut_width(), 4096);
        let next = (plan.k() + 2) * (plan.q_l() - 1) + 1;
        assert!(next.next_power_of_two() > 4096);
    }

    #[test]
    fn grouped_plan_selected_profiles_have_one_reduction_and_terminal_transfer() {
        let cases = [(8, 27, 26, 256), (16, 27, 26, 512), (32, 35, 34, 2048)];
        for (q_l, bucket_count, k, width) in cases {
            let plan = SparseLwrReductionPlan::derive(q_l, bucket_count, width)
                .expect("selected grouped profile plan");
            assert_eq!(plan.k(), k);
            assert_eq!(plan.intermediate_groups(), 1);
            assert_eq!(plan.terminal_start(), k);
            assert_eq!(plan.terminal_len(), bucket_count - k);
            assert_eq!(plan.lut_width(), width);
        }
    }

    #[test]
    fn grouped_plan_q16_b34_uses_maximal_cadence_and_width() {
        let plan = SparseLwrReductionPlan::derive(16, 34, 1 << 16).expect("Q16/B34 plan");
        assert_eq!(plan.k(), 33);
        assert_eq!(plan.intermediate_groups(), 1);
        assert_eq!(plan.terminal_start(), 33);
        assert_eq!(plan.terminal_len(), 1);
        assert_eq!(plan.lut_width(), 512);

        // k=34 would consume every bucket and leave no terminal transfer;
        // its unreduced sum would otherwise require the next power-of-two
        // width, 1024.
        assert_eq!(plan.k() + 1, plan.bucket_count());
        let next_width = ((plan.k() + 2) * (plan.q_l() - 1) + 1).next_power_of_two();
        assert_eq!(next_width, 1024);
    }

    #[test]
    fn grouped_plan_rejects_overflow_and_nondivisible_width() {
        assert!(SparseLwrReductionPlan::derive(usize::MAX, 2, 4096).is_err());
        assert!(SparseLwrReductionPlan::derive(512, 32, 3000).is_err());
    }

    #[test]
    fn grouped_plan_handles_terminal_only_and_partial_terminal_cases() {
        let small = SparseLwrReductionPlan::derive(2, 2, 8).expect("small plan");
        assert!(small.terminal_len() > 0 && small.terminal_len() <= small.k());
        for buckets in [small.k(), small.k() + 1, small.k() * 2 + 1] {
            let plan = SparseLwrReductionPlan::derive(2, buckets, 8).expect("plan");
            assert!(plan.terminal_len() > 0 && plan.terminal_len() <= plan.k());
            assert_eq!(plan.terminal_start(), plan.intermediate_groups() * plan.k());
        }
    }

    #[test]
    fn terminal_rounding_rejects_a_helper_set_bound_to_another_table() {
        let ring = Ring::new(97, 4);
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 2.into(),
        });
        let program = SparseLwrPrfProgram::new(
            SparseLwrPrfProfile::new(2, 2, 4, 4).expect("test profile"),
            1,
            1,
        )
        .expect("test PRF program");
        let wrong_table = LutTable::unary(4, 4, vec![0, 1, 2, 3]).expect("wrong LUT");
        let helpers = (0..4)
            .map(|index| {
                FlatLutHelper::new(
                    1 + index * 2,
                    PowerRhsPackage::new(ring.zero((2, 2))).expect("test RHS"),
                    BggEncodingWire {
                        vector: ring.zero((1, 2)),
                        pubkey: BggPublicKeyWire {
                            matrix: ring.zero((2, 2)),
                            reveal_plaintext: false,
                        },
                        plaintext: None,
                    },
                )
                .expect("test helper")
            })
            .collect();
        let wrong_set = FlatLutHelperSet::new(&wrong_table, helpers).expect("wrong helper set");
        let state = BggEncodingWire {
            vector: ring.zero((1, 2)),
            pubkey: BggPublicKeyWire { matrix: ring.zero((2, 2)), reveal_plaintext: false },
            plaintext: None,
        };

        assert!(matches!(
            program.apply_final_rounding_encoding(&compiler, state, &wrong_set),
            Err(PowerLutError::InvalidLut)
        ));
    }

    #[test]
    fn batch_constructor_and_lowering_reject_mismatched_profile_or_program() {
        let (layout, profile_a, program_a, labels) = batch_fixture();
        let profile_b = SparseLwrPrfProfile::new(3, 2, 8, 8).expect("different profile");

        // The public constructor binds the program profile, rather than
        // allowing a caller to pair a descriptor with a different program.
        assert!(matches!(
            RefreshPrfBatchInputs::new(&layout, profile_b.clone(), &labels, &program_a),
            Err(PowerLutError::InvalidSparseLwrBlock)
        ));

        let program_b = SparseLwrPrfProgram::new(
            profile_b.clone(),
            program_a.bucket_width(),
            program_a.bucket_count,
        )
        .expect("different program identity");
        let mut batch = RefreshPrfBatchInputs::new(&layout, profile_a, &labels, &program_a)
            .expect("matching batch");
        batch.profile = profile_b;
        // This is the common guard used by every typed private/public
        // lowering entry point, so a forged batch cannot cross that boundary.
        assert!(matches!(
            program_a.validate_batch(&batch),
            Err(PowerLutError::InvalidSparseLwrBlock)
        ));
        batch.profile = program_a.profile().clone();
        batch.program_id = program_b.id();
        assert!(matches!(
            program_a.validate_batch(&batch),
            Err(PowerLutError::InvalidSparseLwrBlock)
        ));
    }

    #[test]
    fn huge_compact_batch_uses_deferred_family_without_host_enumeration() {
        let (layout, profile, program, small_labels) = batch_fixture();
        let active_count =
            crate::pbc::PbcActiveCellIndex::build(&layout).expect("active cells").len();
        let small = RefreshPrfBatchInputs::new(&layout, profile.clone(), &small_labels, &program)
            .expect("small batch");

        // Leave ample room for the label-to-value multiplication while still
        // making the synthetic domain far larger than any host materializer
        // could accidentally enumerate.
        let huge_slots = (usize::MAX / active_count / 4).max(1 << 40);
        let huge_labels =
            crate::refresh::RefreshPrfLabelIndex::new([0x53; 32], huge_slots, 2, 1, 1, 1)
                .expect("huge checked label index");
        let huge = RefreshPrfBatchInputs::new(&layout, profile.clone(), &huge_labels, &program)
            .expect("huge compact batch");
        assert!(huge.len() > (1usize << 40));
        assert_eq!(huge.value_count(), huge.len() * active_count);
        assert!(huge.public_input_name().len() <= 64);

        let ring = Ring::new(17, profile.ring_dimension);
        let small_family = batch_public_values_family(&small, &ring).expect("small family");
        let huge_family = batch_public_values_family(&huge, &ring).expect("huge family");
        assert_eq!(
            small_family.as_family().count().evaluate(&mxx_ir_core::ParamEnv::default()).unwrap(),
            BigInt::from(small.value_count())
        );
        assert_eq!(
            huge_family.as_family().count().evaluate(&mxx_ir_core::ParamEnv::default()).unwrap(),
            BigInt::from(huge.value_count())
        );

        // A deferred family contributes no label-sized host graph structure:
        // only the input declaration is emitted for either cardinality.
        let small_graph = DslContext::new("compact-small")
            .family_output("values", small_family.into_family())
            .expect("small output")
            .build()
            .expect("small graph");
        let huge_graph = DslContext::new("compact-huge")
            .family_output("values", huge_family.into_family())
            .expect("huge output")
            .build()
            .expect("huge graph");
        assert_eq!(small_graph.graph.scopes().len(), huge_graph.graph.scopes().len());
        assert_eq!(
            small_graph.graph.root_scope().nodes().len(),
            huge_graph.graph.root_scope().nodes().len()
        );

        assert!(
            crate::refresh::RefreshPrfLabelIndex::new([0x54; 32], usize::MAX, usize::MAX, 2, 2, 2,)
                .is_err()
        );
    }
}
