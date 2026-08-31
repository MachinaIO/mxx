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

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use mxx_bgg::{BggEncodingWire, BggPublicKeyWire};
use mxx_dsl::{Bool, Family, Int, LoopIndex, Mat, Parallel, Sequential};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use sha2::{Digest, Sha256};

use crate::{
    PowerLutEncodingCompiler, PowerLutError,
    encoding::{AutomorphismHelper, EncodingSelectorFamily},
    program::{
        FamilyRange, LutTable, PowerLutProgram, PowerLutProgramBuilder, PowerLutProgramId,
        ProgramFamilyRanges, ProgramInputId, ProgramValidationError, ProgramWireId,
        PublicValueFamilyId, RhsFamilyId,
    },
    public_key::{AutomorphismPublicHelper, PowerLutPublicKeyCompiler, PublicSelectorFamily},
    rhs::{PowerLutPublicRhsPackage, PowerRhsCompanionBlock, PowerRhsPackage},
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
    label_digests: Vec<[u8; 32]>,
    public_vector_identities: Vec<[u8; 32]>,
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
    ) -> Result<Self, PowerLutError> {
        layout.validate().map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let mut label_digests = Vec::with_capacity(labels.len());
        let mut public_vector_identities = Vec::with_capacity(labels.len());
        for index in 0..labels.len() {
            let label =
                labels.label(index).ok_or(PowerLutError::InvalidSparseLwrBlock)?.canonical_bytes();
            let encoded =
                crate::pbc::PbcEncodedPublicVector::from_label(layout, &label, profile.q_l)
                    .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
            let vector = crate::pbc::PbcPublicVectorFamilyBinding::from_encoded(layout, &encoded)
                .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
            label_digests.push(label_digest(&label));
            public_vector_identities.push(public_vector_identity(layout, &vector));
        }
        let active_count = crate::pbc::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .len();
        let value_count =
            labels.len().checked_mul(active_count).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let batch_id =
            public_value_batch_id(layout, &profile, &label_digests, &public_vector_identities);
        Ok(Self {
            layout: layout.clone(),
            layout_id: layout.layout_id,
            profile,
            batch_id,
            active_count,
            value_count,
            label_digests,
            public_vector_identities,
        })
    }

    /// Returns the number of canonical label positions.
    pub fn len(&self) -> usize {
        self.label_digests.len()
    }

    /// Returns whether the batch contains no labels.
    pub fn is_empty(&self) -> bool {
        self.label_digests.is_empty()
    }

    /// Returns the public label digest at a canonical index.
    pub fn label_digest(&self, index: usize) -> Option<[u8; 32]> {
        self.label_digests.get(index).copied()
    }

    /// Returns the public routed-vector identity at a canonical index.
    pub fn public_vector_identity(&self, index: usize) -> Option<[u8; 32]> {
        self.public_vector_identities.get(index).copied()
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

    /// Returns the stable input name used for this deferred public family.
    pub fn public_input_name(&self) -> String {
        format!(
            "pbc-public-values-{:x}",
            u128::from_le_bytes(self.batch_id.0[..16].try_into().unwrap())
        )
    }

    /// Materializes canonical residues for small trusted fixtures. Production
    /// execution payload streaming is intentionally deferred; this helper
    /// rederives and validates the same identity used by the runtime family.
    pub fn materialize_public_values(
        &self,
        labels: &crate::refresh::RefreshPrfLabelIndex,
    ) -> Result<Vec<u64>, PowerLutError> {
        if labels.len() != self.len() {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let mut values = Vec::with_capacity(self.value_count);
        let mut label_digests = Vec::with_capacity(self.len());
        let mut vector_identities = Vec::with_capacity(self.len());
        for index in 0..labels.len() {
            let label =
                labels.label(index).ok_or(PowerLutError::InvalidSparseLwrBlock)?.canonical_bytes();
            let encoded = crate::pbc::PbcEncodedPublicVector::from_label(
                &self.layout,
                &label,
                self.profile.q_l,
            )
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
            let vector =
                crate::pbc::PbcPublicVectorFamilyBinding::from_encoded(&self.layout, &encoded)
                    .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
            label_digests.push(label_digest(&label));
            vector_identities.push(public_vector_identity(&self.layout, &vector));
            values.extend(vector.values_u64());
        }
        if label_digests != self.label_digests ||
            vector_identities != self.public_vector_identities ||
            public_value_batch_id(
                &self.layout,
                &self.profile,
                &label_digests,
                &vector_identities,
            ) != self.batch_id ||
            values.len() != self.value_count
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Ok(values)
    }
}

fn public_value_batch_id(
    layout: &crate::pbc::PbcPublicLayout,
    profile: &SparseLwrPrfProfile,
    labels: &[[u8; 32]],
    vectors: &[[u8; 32]],
) -> PbcPublicValueBatchId {
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/pbc-public-value-batch/v1");
    digest.update(layout.layout_id.0);
    digest.update((profile.q_l as u64).to_le_bytes());
    digest.update((profile.ring_dimension as u64).to_le_bytes());
    for value in labels.iter().chain(vectors) {
        digest.update(value);
    }
    PbcPublicValueBatchId(digest.finalize().into())
}

fn label_digest(label: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/sparse-lwr/prf-label/v1");
    digest.update((label.len() as u64).to_le_bytes());
    digest.update(label);
    digest.finalize().into()
}

fn public_vector_identity(
    layout: &crate::pbc::PbcPublicLayout,
    vector: &crate::pbc::PbcPublicVectorFamilyBinding,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/sparse-lwr/public-vector/v1");
    digest.update(layout.layout_id.0);
    digest.update((vector.modulus as u64).to_le_bytes());
    for value in vector.values_u64() {
        digest.update(value.to_le_bytes());
    }
    digest.finalize().into()
}

/// Declares the batch's deferred label-major public-value family. The runtime
/// payload provider is intentionally outside this graph-construction layer.
fn batch_public_values_family(
    batch: &RefreshPrfBatchInputs,
    ring: &mxx_dsl::Ring,
) -> Result<Family<Mat>, PowerLutError> {
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
    Ok(ring.input_family(batch.public_input_name(), batch.value_count, (1, 1)))
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
/// The mandatory reduction table is an ordinary unary LUT gate following the
/// shared one-hot selection. The mandatory rounding table is a second
/// ordinary program applied once after the bucket scan and never per bucket.
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
    /// Concrete profile from which both mandatory LUTs were derived.
    profile: SparseLwrPrfProfile,
    /// Ordinary shared program for the final LWR rounding operation.
    pub rounding_program: PowerLutProgram,
    /// Composite identity of the bucket and final-rounding programs.
    composite_id: PowerLutProgramId,
    rounding_input: ProgramInputId,
    rounding_output: ProgramWireId,
    /// The final table is cached only as immutable lowering data. Its
    /// declaration is also present in `rounding_program`.
    rounding_lut: Vec<usize>,
}

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
    pub fn compile_encoding(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        selectors: BTreeMap<RhsFamilyId, EncodingSelectorFamily>,
        public_values: BTreeMap<PublicValueFamilyId, mxx_dsl::Family<Mat>>,
        helpers: &[AutomorphismHelper],
        label: &[u8],
    ) -> Result<(BggEncodingWire, SparseLwrPrfOutput), PowerLutError> {
        self.validate_widths(concrete_ring_dimension(&input)?)?;
        let outputs = compiler.compile_program(
            &self.program,
            &BTreeMap::from([(self.input, input)]),
            &BTreeMap::new(),
            &selectors,
            &public_values,
            helpers,
        )?;
        let output = outputs.get(&self.output).cloned().ok_or(PowerLutError::InvalidLut)?;
        let output = self.apply_final_rounding_encoding(compiler, output, helpers)?;
        let descriptor = self.bind_output(label)?;
        Ok((output, descriptor))
    }

    /// Lowers this same program through the public-key API.
    ///
    /// Only public input keys, RHS projections, and public value matrices are
    /// accepted. This method cannot receive a sparse support or private
    /// selector package, which keeps the public derivation independent of the
    /// secret schedule.
    pub fn compile_public_key(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input: BggPublicKeyWire,
        selectors: BTreeMap<RhsFamilyId, PublicSelectorFamily>,
        public_values: BTreeMap<PublicValueFamilyId, mxx_dsl::Family<Mat>>,
        helpers: &[AutomorphismPublicHelper],
        label: &[u8],
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        self.validate_widths(concrete_ring_dimension_public(&input)?)?;
        let outputs = compiler.compile_program(
            &self.program,
            &BTreeMap::from([(self.input, input)]),
            &BTreeMap::new(),
            &selectors,
            &public_values,
            helpers,
        )?;
        let output = outputs.get(&self.output).cloned().ok_or(PowerLutError::InvalidLut)?;
        let output = self.apply_final_rounding_public(compiler, output, helpers)?;
        Ok((output, self.bind_output(label)?))
    }

    /// Evaluates the same bucket body over a structural sequential loop.
    ///
    /// The binding closure is invoked once while the DSL graph is being
    /// constructed with the loop index.  It must return the selector and
    /// public-value families plus one identical contiguous range for that
    /// bucket.  The range must exclude padding cells rather than
    /// manufacturing private RHS packages for them.  The loop state is the
    /// current monomial accumulator, so the body output becomes the next
    /// accumulator without a host-side bucket loop.
    pub fn compile_encoding_sequential(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        bucket_count: impl Into<mxx_ir_core::IntExpr>,
        bucket_bindings: impl FnOnce(
            LoopIndex,
        ) -> Result<
            (EncodingSelectorFamily, Family<Mat>, FamilyRange),
            PowerLutError,
        >,
        helpers: &[AutomorphismHelper],
        label: &[u8],
    ) -> Result<(BggEncodingWire, SparseLwrPrfOutput), PowerLutError> {
        self.validate_widths(concrete_ring_dimension(&input)?)?;
        let lowering_error = RefCell::new(None);
        let final_state = Sequential::range(bucket_count)
            .scan(input, Bool::constant(true), |bucket, state, _| {
                let (selectors, public_values, range) =
                    bucket_bindings(bucket).map_err(|error| {
                        *lowering_error.borrow_mut() = Some(error);
                        mxx_dsl::DslError::Schema
                    })?;
                let mut ranges = ProgramFamilyRanges::new();
                ranges.selector(self.selector_family, range.clone());
                ranges.public_values(self.public_value_family, range);
                let outputs = compiler
                    .compile_program_with_ranges(
                        &self.program,
                        &BTreeMap::from([(self.input, state)]),
                        &BTreeMap::new(),
                        &BTreeMap::from([(self.selector_family, selectors)]),
                        &BTreeMap::from([(self.public_value_family, public_values)]),
                        &ranges,
                        helpers,
                    )
                    .map_err(|error| {
                        *lowering_error.borrow_mut() = Some(error);
                        mxx_dsl::DslError::Schema
                    })?;
                outputs.get(&self.output).cloned().ok_or_else(|| {
                    *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                    mxx_dsl::DslError::Schema
                })
            })
            .map_err(|_| {
                lowering_error.into_inner().unwrap_or(PowerLutError::InvalidSparseLwrBlock)
            })?;
        let output = self.apply_final_rounding_encoding(compiler, final_state, helpers)?;
        let descriptor = self.bind_output(label)?;
        Ok((output, descriptor))
    }

    /// Sequential bucket lowering with all dynamic selector inputs supplied
    /// as explicit loop invariants.  This is used by the batched PBC path so
    /// the outer label loop does not rely on captured executable families.
    fn compile_encoding_sequential_with_invariants(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        bucket_count: impl Into<mxx_ir_core::IntExpr>,
        offsets: Vec<usize>,
        widths: Vec<usize>,
        selector_flat_count: usize,
        capacity: usize,
        invariants: Vec<Family<Mat>>,
        helpers: &[AutomorphismHelper],
        label: &[u8],
    ) -> Result<(BggEncodingWire, SparseLwrPrfOutput), PowerLutError> {
        self.validate_widths(concrete_ring_dimension(&input)?)?;
        let lowering_error = RefCell::new(None);
        let final_state = Sequential::range(bucket_count)
            .scan(input, invariants, |bucket, state, mut flat| {
                if flat.len() != selector_flat_count + 1 {
                    *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                    return Err(mxx_dsl::DslError::Schema);
                }
                let public_values = flat.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let selectors = EncodingSelectorFamily::from_flattened(flat).map_err(|error| {
                    *lowering_error.borrow_mut() = Some(error);
                    mxx_dsl::DslError::Schema
                })?;
                let index = bucket.expression();
                let start = interpolate_lookup(&offsets, index.clone()).map_err(|error| {
                    *lowering_error.borrow_mut() = Some(error);
                    mxx_dsl::DslError::Schema
                })?;
                let count = interpolate_lookup(&widths, index).map_err(|error| {
                    *lowering_error.borrow_mut() = Some(error);
                    mxx_dsl::DslError::Schema
                })?;
                let range = FamilyRange::bounded(start, count, capacity).map_err(|_| {
                    *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                    mxx_dsl::DslError::Schema
                })?;
                let mut ranges = ProgramFamilyRanges::new();
                ranges.selector(self.selector_family, range.clone());
                ranges.public_values(self.public_value_family, range);
                let outputs = compiler
                    .compile_program_with_ranges(
                        &self.program,
                        &BTreeMap::from([(self.input, state)]),
                        &BTreeMap::new(),
                        &BTreeMap::from([(self.selector_family, selectors)]),
                        &BTreeMap::from([(self.public_value_family, public_values)]),
                        &ranges,
                        helpers,
                    )
                    .map_err(|error| {
                        *lowering_error.borrow_mut() = Some(error);
                        mxx_dsl::DslError::Schema
                    })?;
                outputs.get(&self.output).cloned().ok_or_else(|| {
                    *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                    mxx_dsl::DslError::Schema
                })
            })
            .map_err(|_| {
                lowering_error.into_inner().unwrap_or(PowerLutError::InvalidSparseLwrBlock)
            })?;
        let output = self.apply_final_rounding_encoding(compiler, final_state, helpers)?;
        let descriptor = self.bind_output(label)?;
        Ok((output, descriptor))
    }

    /// Public-key counterpart of [`Self::compile_encoding_sequential`].
    ///
    /// The binding closure can expose only public selector projections and
    /// public routed values.  It cannot receive a sparse support, schedule,
    /// selected slot, or private GSW family.  Both methods construct the same
    /// structural loop and invoke the same immutable program body.
    pub fn compile_public_key_sequential(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input: BggPublicKeyWire,
        bucket_count: impl Into<mxx_ir_core::IntExpr>,
        bucket_bindings: impl FnOnce(
            LoopIndex,
        ) -> Result<
            (PublicSelectorFamily, Family<Mat>, FamilyRange),
            PowerLutError,
        >,
        helpers: &[AutomorphismPublicHelper],
        label: &[u8],
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        self.validate_widths(concrete_ring_dimension_public(&input)?)?;
        let lowering_error = RefCell::new(None);
        let final_state = Sequential::range(bucket_count)
            .scan(input, Bool::constant(true), |bucket, state, _| {
                let (selectors, public_values, range) =
                    bucket_bindings(bucket).map_err(|error| {
                        *lowering_error.borrow_mut() = Some(error);
                        mxx_dsl::DslError::Schema
                    })?;
                let mut ranges = ProgramFamilyRanges::new();
                ranges.selector(self.selector_family, range.clone());
                ranges.public_values(self.public_value_family, range);
                let outputs = compiler
                    .compile_program_with_ranges(
                        &self.program,
                        &BTreeMap::from([(self.input, state)]),
                        &BTreeMap::new(),
                        &BTreeMap::from([(self.selector_family, selectors)]),
                        &BTreeMap::from([(self.public_value_family, public_values)]),
                        &ranges,
                        helpers,
                    )
                    .map_err(|error| {
                        *lowering_error.borrow_mut() = Some(error);
                        mxx_dsl::DslError::Schema
                    })?;
                outputs.get(&self.output).cloned().ok_or_else(|| {
                    *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                    mxx_dsl::DslError::Schema
                })
            })
            .map_err(|_| {
                lowering_error.into_inner().unwrap_or(PowerLutError::InvalidSparseLwrBlock)
            })?;
        let output = self.apply_final_rounding_public(compiler, final_state, helpers)?;
        Ok((output, self.bind_output(label)?))
    }

    /// Public-key sequential bucket lowering with explicit selector and value
    /// invariants.  The invariant representation mirrors the private path,
    /// while containing only public companion families.
    fn compile_public_key_sequential_with_invariants(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input: BggPublicKeyWire,
        bucket_count: impl Into<mxx_ir_core::IntExpr>,
        offsets: Vec<usize>,
        widths: Vec<usize>,
        selector_flat_count: usize,
        capacity: usize,
        invariants: Vec<Family<Mat>>,
        helpers: &[AutomorphismPublicHelper],
        label: &[u8],
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        self.validate_widths(concrete_ring_dimension_public(&input)?)?;
        let lowering_error = RefCell::new(None);
        let final_state = Sequential::range(bucket_count)
            .scan(input, invariants, |bucket, state, mut flat| {
                if flat.len() != selector_flat_count + 1 {
                    *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                    return Err(mxx_dsl::DslError::Schema);
                }
                let public_values = flat.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let selectors = PublicSelectorFamily::from_flattened(flat).map_err(|error| {
                    *lowering_error.borrow_mut() = Some(error);
                    mxx_dsl::DslError::Schema
                })?;
                let index = bucket.expression();
                let start = interpolate_lookup(&offsets, index.clone()).map_err(|error| {
                    *lowering_error.borrow_mut() = Some(error);
                    mxx_dsl::DslError::Schema
                })?;
                let count = interpolate_lookup(&widths, index).map_err(|error| {
                    *lowering_error.borrow_mut() = Some(error);
                    mxx_dsl::DslError::Schema
                })?;
                let range = FamilyRange::bounded(start, count, capacity).map_err(|_| {
                    *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                    mxx_dsl::DslError::Schema
                })?;
                let mut ranges = ProgramFamilyRanges::new();
                ranges.selector(self.selector_family, range.clone());
                ranges.public_values(self.public_value_family, range);
                let outputs = compiler
                    .compile_program_with_ranges(
                        &self.program,
                        &BTreeMap::from([(self.input, state)]),
                        &BTreeMap::new(),
                        &BTreeMap::from([(self.selector_family, selectors)]),
                        &BTreeMap::from([(self.public_value_family, public_values)]),
                        &ranges,
                        helpers,
                    )
                    .map_err(|error| {
                        *lowering_error.borrow_mut() = Some(error);
                        mxx_dsl::DslError::Schema
                    })?;
                outputs.get(&self.output).cloned().ok_or_else(|| {
                    *lowering_error.borrow_mut() = Some(PowerLutError::InvalidSparseLwrBlock);
                    mxx_dsl::DslError::Schema
                })
            })
            .map_err(|_| {
                lowering_error.into_inner().unwrap_or(PowerLutError::InvalidSparseLwrBlock)
            })?;
        let output = self.apply_final_rounding_public(compiler, final_state, helpers)?;
        Ok((output, self.bind_output(label)?))
    }

    /// Evaluates a flattened PBC selector/value family over all layout
    /// buckets.  The active cells of a bucket are contiguous in the
    /// canonical [`crate::pbc::PbcActiveCellIndex`] order, but their offsets
    /// and counts may differ.  We therefore derive public lookup expressions
    /// for both quantities and bind one exact `FamilyRange` inside the single
    /// outer sequential loop.
    ///
    /// The interpolation metadata is host-side and contains no secret
    /// schedule.  Its size is quadratic in the public bucket count; Power-LUT
    /// parameter sets keep this count small.  Padding is outside every range,
    /// so no hidden RHS package is allocated for it.
    #[cfg(test)]
    pub(crate) fn compile_pbc_encoding(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        layout: &crate::pbc::PbcPublicLayout,
        public_vector: &crate::pbc::PbcPublicVectorFamilyBinding,
        selectors: EncodingSelectorFamily,
        public_values: Family<Mat>,
        helpers: &[AutomorphismHelper],
        label: &[u8],
    ) -> Result<PbcSparseLwrEncodingOutput, PowerLutError> {
        if self.bucket_width != layout.bucket_width ||
            self.program.inputs().get(&self.input) != Some(&self.lut_width) ||
            self.program.rhs_family_width(self.selector_family) != Some(self.lut_width) ||
            self.program.public_value_family_width(self.public_value_family) !=
                Some(self.lut_width)
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        public_vector.validate(layout).map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        self.compile_pbc_encoding_with_family(
            compiler,
            input,
            layout,
            selectors,
            public_values,
            helpers,
            label,
        )
    }

    /// Internal structural implementation for [`Self::compile_pbc_encoding`].
    ///
    /// The family must have been declared from the validated
    /// [`crate::pbc::PbcPublicVectorFamilyBinding`] by the public wrapper.
    /// Keeping this raw-family operation crate-private prevents callers from
    /// accidentally pairing an unrelated family with a PBC layout.
    #[cfg(test)]
    pub(crate) fn compile_pbc_encoding_with_family(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        layout: &crate::pbc::PbcPublicLayout,
        selectors: EncodingSelectorFamily,
        public_values: Family<Mat>,
        helpers: &[AutomorphismHelper],
        label: &[u8],
    ) -> Result<PbcSparseLwrEncodingOutput, PowerLutError> {
        if self.bucket_width != layout.bucket_width ||
            self.program.inputs().get(&self.input) != Some(&self.lut_width) ||
            self.program.rhs_family_width(self.selector_family) != Some(self.lut_width) ||
            self.program.public_value_family_width(self.public_value_family) !=
                Some(self.lut_width)
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let active = crate::pbc::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        ensure_family_count(&public_values, active.len())?;
        let widths = active.bucket_active_widths().collect::<Vec<_>>();
        let mut offsets = Vec::with_capacity(widths.len());
        let mut offset = 0usize;
        for width in widths.iter().copied() {
            offsets.push(offset);
            offset = offset.checked_add(width).ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        }
        let bucket_count = layout.parameters.bucket_count;
        let (encoding, descriptor) = self.compile_encoding_sequential(
            compiler,
            input,
            bucket_count,
            move |bucket| {
                let index = bucket.expression();
                let start = interpolate_lookup(&offsets, index.clone())?;
                let count = interpolate_lookup(&widths, index)?;
                let range = FamilyRange::bounded(start, count, layout.bucket_width)
                    .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
                Ok((selectors.clone(), public_values.clone(), range))
            },
            helpers,
            label,
        )?;
        Ok(PbcSparseLwrEncodingOutput::new(encoding, descriptor, layout.layout_id))
    }

    /// Lowers a family of independent PBC labels through one outer
    /// structural parallel loop. The input families are zipped by label;
    /// selector families and label-major public values are explicit outer
    /// broadcast inputs, then become invariants of the nested sequential
    /// bucket loop. Public values are flattened in
    /// canonical label-major, active-cell order, and each bucket iteration
    /// selects only its own contiguous active-cell range.
    ///
    /// This is the batch counterpart of [`Self::compile_pbc_encoding`].  It
    /// deliberately returns families rather than a host vector of wires, so
    /// callers can continue routing and reducing the labels structurally.
    pub(crate) fn compile_pbc_encoding_family(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &[AutomorphismHelper],
    ) -> Result<(Family<Mat>, Family<Mat>), PowerLutError> {
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
        ensure_family_count(&public_values, expected_values)?;

        let selector_flat = selectors.flattened();
        let selector_flat_count = selector_flat.len();
        let (helper_flat, helper_arities) = flatten_private_helper_families(helpers)?;
        let mut broadcasts = selector_flat;
        broadcasts.push(public_values);
        let helper_flat_count = helper_flat.len();
        broadcasts.extend(helper_flat);
        let lowering_error = Rc::new(RefCell::new(None));
        let lowering_error_for_loop = lowering_error.clone();
        let outputs = Family::<Mat>::parallel_zip_many_with_broadcast_values(
            vec![input_vectors, input_public_keys],
            broadcasts,
            move |label, mut inputs, mut broadcasts| {
                let helper_start = selector_flat_count + 1;
                if broadcasts.len() != helper_start + helper_flat_count {
                    return Err(mxx_dsl::DslError::Schema);
                }
                let helper_broadcasts = broadcasts.split_off(helper_start);
                let label_values = broadcasts.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let selectors = EncodingSelectorFamily::from_flattened(broadcasts)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let helpers = rebuild_private_helper_families(helper_broadcasts, &helper_arities)?;
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
                    .compile_encoding_sequential_with_invariants(
                        compiler,
                        input,
                        layout.parameters.bucket_count,
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

    /// Lowers a trusted canonical refresh batch through one structural
    /// parallel loop. The output family is already bound to the batch's
    /// ordered labels and routed public-vector identities.
    pub fn compile_pbc_encoding_family_typed_with_batch(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &[AutomorphismHelper],
    ) -> Result<PbcSparseLwrEncodingOutputs, PowerLutError> {
        if batch.layout_id != batch.layout.layout_id || batch.profile != self.profile {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let (vectors, public_keys) = self.compile_pbc_encoding_family(
            compiler,
            input_vectors,
            input_public_keys,
            batch,
            selectors,
            helpers,
        )?;
        Ok(PbcSparseLwrEncodingOutputs::new(
            vectors,
            public_keys,
            self.id(),
            self.rounding_output,
            batch,
        ))
    }

    /// Public-key counterpart of [`Self::compile_pbc_encoding_family`].  The
    /// label inputs, public values, selector projections, and helper masks
    /// are all public; no private selector or schedule can enter this path.
    pub(crate) fn compile_pbc_public_key_family(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &[AutomorphismPublicHelper],
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
        ensure_family_count(&public_values, expected_values)?;
        let selector_flat = selectors.flattened();
        let selector_flat_count = selector_flat.len();
        let (helper_flat, helper_arities) = flatten_public_helper_families(helpers)?;
        let mut broadcasts = selector_flat;
        broadcasts.push(public_values);
        let helper_flat_count = helper_flat.len();
        broadcasts.extend(helper_flat);
        let lowering_error = Rc::new(RefCell::new(None));
        let lowering_error_for_loop = lowering_error.clone();
        let outputs = Family::<Mat>::parallel_zip_many_with_broadcast_values(
            vec![input_keys],
            broadcasts,
            move |label, mut inputs, mut broadcasts| {
                let helper_start = selector_flat_count + 1;
                if broadcasts.len() != helper_start + helper_flat_count {
                    return Err(mxx_dsl::DslError::Schema);
                }
                let helper_broadcasts = broadcasts.split_off(helper_start);
                let label_values = broadcasts.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let selectors = PublicSelectorFamily::from_flattened(broadcasts)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                let helpers = rebuild_public_helper_families(helper_broadcasts, &helper_arities)?;
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
                    .compile_public_key_sequential_with_invariants(
                        compiler,
                        input,
                        layout.parameters.bucket_count,
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

    /// Lowers a canonical refresh batch through the public-key PBC path.
    ///
    /// The batch owns the label order and the routed-vector identities. This
    /// is the production entry point for public-key sparse-LWR evaluation;
    /// the raw family lowerer remains crate-private so callers cannot attach
    /// replacement labels or vectors after construction.
    pub fn compile_pbc_public_key_family_with_batch(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &[AutomorphismPublicHelper],
    ) -> Result<Family<Mat>, PowerLutError> {
        if batch.layout_id != batch.layout.layout_id || batch.profile != self.profile {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        self.compile_pbc_public_key_family(compiler, input_keys, batch, selectors, helpers)
    }

    /// Applies the mandatory LWR rounding table once to the completed bucket
    /// accumulator. Keeping this after the sequential scan is essential:
    /// rounding each bucket would change `floor(p * sum(z_b) / Q_L)` into a
    /// sum of rounded partial values.
    fn apply_final_rounding_encoding(
        &self,
        compiler: &PowerLutEncodingCompiler,
        state: BggEncodingWire,
        helpers: &[AutomorphismHelper],
    ) -> Result<BggEncodingWire, PowerLutError> {
        let outputs = compiler.compile_program(
            &self.rounding_program,
            &BTreeMap::from([(self.rounding_input, state)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            helpers,
        )?;
        outputs.get(&self.rounding_output).cloned().ok_or(PowerLutError::InvalidSparseLwrBlock)
    }

    /// Public-key counterpart of [`Self::apply_final_rounding_encoding`].
    fn apply_final_rounding_public(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        state: BggPublicKeyWire,
        helpers: &[AutomorphismPublicHelper],
    ) -> Result<BggPublicKeyWire, PowerLutError> {
        let outputs = compiler.compile_program(
            &self.rounding_program,
            &BTreeMap::from([(self.rounding_input, state)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            helpers,
        )?;
        outputs.get(&self.rounding_output).cloned().ok_or(PowerLutError::InvalidSparseLwrBlock)
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
    helpers: &[AutomorphismHelper],
) -> Result<(Vec<Family<Mat>>, Vec<(usize, usize)>), PowerLutError> {
    let mut flat = Vec::new();
    let mut arities = Vec::with_capacity(helpers.len());
    for helper in helpers {
        let switch = helper.switch();
        let mut companions = Vec::new();
        while let Some(companion) = switch.companion_at(companions.len()) {
            companions.push(companion);
        }
        if companions.is_empty() {
            return Err(PowerLutError::InvalidAutomorphismHelper);
        }
        arities.push((helper.index(), companions.len()));
        let mut matrices = Vec::with_capacity(1 + companions.len() * 2 + 2);
        matrices.push(switch.gsw_ciphertext().clone());
        for companion in companions {
            matrices.push(companion.vector.clone());
            matrices.push(companion.public_matrix.clone());
        }
        matrices.push(helper.mask().vector.clone());
        matrices.push(helper.mask().pubkey.matrix.clone());
        for matrix in matrices {
            flat.push(
                Family::pack(vec![matrix]).map_err(|_| PowerLutError::InvalidAutomorphismHelper)?,
            );
        }
    }
    Ok((flat, arities))
}

fn rebuild_private_helper_families(
    flat: Vec<Family<Mat>>,
    arities: &[(usize, usize)],
) -> Result<Vec<AutomorphismHelper>, mxx_dsl::DslError> {
    let expected = arities.iter().try_fold(0usize, |total, (_, count)| {
        total.checked_add(1 + count * 2 + 2).ok_or(mxx_dsl::DslError::Schema)
    })?;
    if flat.len() != expected {
        return Err(mxx_dsl::DslError::Schema);
    }
    let mut cursor = 0usize;
    let mut rebuilt = Vec::with_capacity(arities.len());
    for (helper_index, companion_count) in arities.iter().copied() {
        let take = |cursor: &mut usize| -> Result<Mat, mxx_dsl::DslError> {
            let family = flat.get(*cursor).ok_or(mxx_dsl::DslError::Schema)?;
            *cursor += 1;
            Ok(family.get_static(0))
        };
        let gsw = take(&mut cursor)?;
        let companions = (0..companion_count)
            .map(|_| {
                Ok(PowerRhsCompanionBlock {
                    vector: take(&mut cursor)?,
                    public_matrix: take(&mut cursor)?,
                })
            })
            .collect::<Result<Vec<_>, mxx_dsl::DslError>>()?;
        let mask_vector = take(&mut cursor)?;
        let mask_public = take(&mut cursor)?;
        let switch =
            PowerRhsPackage::new(gsw, companions).map_err(|_| mxx_dsl::DslError::Schema)?;
        let mask = BggEncodingWire {
            vector: mask_vector,
            pubkey: BggPublicKeyWire { matrix: mask_public, reveal_plaintext: false },
            plaintext: None,
        };
        // The loop preserves the imported helper order; indexes are the
        // trusted metadata attached to those artifacts, not loop-derived data.
        rebuilt.push(
            AutomorphismHelper::new(helper_index, switch, mask)
                .map_err(|_| mxx_dsl::DslError::Schema)?,
        );
    }
    Ok(rebuilt)
}

fn flatten_public_helper_families(
    helpers: &[AutomorphismPublicHelper],
) -> Result<(Vec<Family<Mat>>, Vec<(usize, usize)>), PowerLutError> {
    let mut flat = Vec::new();
    let mut arities = Vec::with_capacity(helpers.len());
    for helper in helpers {
        let companions = helper.switch().companions();
        if companions.is_empty() {
            return Err(PowerLutError::InvalidAutomorphismHelper);
        }
        arities.push((helper.index(), companions.len()));
        for matrix in companions.iter().chain(std::iter::once(helper.mask())) {
            flat.push(
                Family::pack(vec![matrix.clone()])
                    .map_err(|_| PowerLutError::InvalidAutomorphismHelper)?,
            );
        }
    }
    Ok((flat, arities))
}

fn rebuild_public_helper_families(
    flat: Vec<Family<Mat>>,
    arities: &[(usize, usize)],
) -> Result<Vec<AutomorphismPublicHelper>, mxx_dsl::DslError> {
    let expected = arities.iter().try_fold(0usize, |total, (_, count)| {
        total.checked_add(count + 1).ok_or(mxx_dsl::DslError::Schema)
    })?;
    if flat.len() != expected {
        return Err(mxx_dsl::DslError::Schema);
    }
    let mut cursor = 0usize;
    let mut rebuilt = Vec::with_capacity(arities.len());
    for (helper_index, companion_count) in arities.iter().copied() {
        let take = |cursor: &mut usize| -> Result<Mat, mxx_dsl::DslError> {
            let family = flat.get(*cursor).ok_or(mxx_dsl::DslError::Schema)?;
            *cursor += 1;
            Ok(family.get_static(0))
        };
        let companions =
            (0..companion_count).map(|_| take(&mut cursor)).collect::<Result<_, _>>()?;
        let mask = take(&mut cursor)?;
        let switch =
            PowerLutPublicRhsPackage::new(companions).map_err(|_| mxx_dsl::DslError::Schema)?;
        rebuilt.push(AutomorphismPublicHelper::new(helper_index, switch, mask));
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

/// Builds the reusable one-hot bucket body and its reduction LUT.
///
/// The selector and public-value families stay as runtime inputs; this helper
/// only declares the structural program shared by every bucket.
fn build_bucket_program(
    lut_width: usize,
    bucket_width: usize,
    reduction_lut: &[usize],
) -> Result<SparseLwrBucketProgram, ProgramValidationError> {
    if lut_width == 0 || !lut_width.is_power_of_two() || bucket_width == 0 {
        return Err(ProgramValidationError::WidthMismatch);
    }
    let mut builder = PowerLutProgramBuilder::new();
    let input = builder.input(lut_width)?;
    let selector_family = builder.rhs_family(lut_width)?;
    let public_value_family = builder.public_value_family(lut_width)?;
    let select_lut =
        builder.lut(LutTable::unary(lut_width, lut_width, (0..lut_width).collect())?)?;
    let input_wire = builder.input_wire(input)?;
    let selected = builder.one_hot(input_wire, selector_family, public_value_family, select_lut)?;
    let reduction = builder.lut(LutTable::unary(lut_width, lut_width, reduction_lut.to_vec())?)?;
    let output = builder.unary(selected, reduction)?;
    builder.output(output)?;
    let program = builder.build()?;
    Ok(SparseLwrBucketProgram { program, input, selector_family, public_value_family, output })
}

/// Builds the final scalar LUT that rounds a reduced sparse-LWR value.
fn build_rounding_program(
    lut_width: usize,
    rounding_lut: &[usize],
) -> Result<(PowerLutProgram, ProgramInputId, ProgramWireId), ProgramValidationError> {
    let mut builder = PowerLutProgramBuilder::new();
    let input = builder.input(lut_width)?;
    let input_wire = builder.input_wire(input)?;
    let lut = builder.lut(LutTable::unary(lut_width, lut_width, rounding_lut.to_vec())?)?;
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
    bucket_program: &PowerLutProgram,
    rounding_program: &PowerLutProgram,
) -> PowerLutProgramId {
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/sparse-lwr/prf-program/v2");
    digest.update((profile.q_l as u64).to_le_bytes());
    digest.update((profile.p as u64).to_le_bytes());
    digest.update((profile.lut_width as u64).to_le_bytes());
    digest.update((profile.ring_dimension as u64).to_le_bytes());
    digest.update(bucket_program.id().as_bytes());
    digest.update(rounding_program.id().as_bytes());
    PowerLutProgramId::from_digest(digest.finalize().into())
}

impl SparseLwrPrfProgram {
    /// Constructs the mandatory sparse-LWR bucket and final-rounding programs.
    ///
    /// The PBC `bucket_width` is only a rectangular family shape. It is not a
    /// LUT domain and never determines either table. The reduction table is
    /// `z mod Q_L`; the final table is `floor(p * (z mod Q_L) / Q_L)` and is
    /// applied exactly once after the sequential bucket scan.
    pub fn new(profile: SparseLwrPrfProfile, bucket_width: usize) -> Result<Self, PowerLutError> {
        if bucket_width == 0 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let w = profile.lut_width;
        let reduction_lut = (0..w).map(|z| z % profile.q_l).collect::<Vec<_>>();
        let rounding_lut = (0..w)
            .map(|z| {
                profile
                    .p
                    .checked_mul(z % profile.q_l)
                    .map(|value| value / profile.q_l)
                    .ok_or(PowerLutError::InvalidSparseLwrBlock)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bucket =
            build_bucket_program(w, bucket_width, &reduction_lut).map_err(PowerLutError::from)?;
        let (rounding_program, rounding_input, rounding_output) =
            build_rounding_program(w, &rounding_lut).map_err(PowerLutError::from)?;
        let composite_id = composite_prf_id(&profile, &bucket.program, &rounding_program);
        Ok(Self {
            program: bucket.program,
            input: bucket.input,
            selector_family: bucket.selector_family,
            public_value_family: bucket.public_value_family,
            output: bucket.output,
            lut_width: w,
            bucket_width,
            profile,
            rounding_program,
            composite_id,
            rounding_input,
            rounding_output,
            rounding_lut,
        })
    }
}

/// Generic one-hot fixture retained only for crate tests. Production callers
/// must use [`SparseLwrPrfProgram::new`]. This type intentionally contains no
/// sparse-LWR profile or output identity, so generic OneHot/LUT tests cannot
/// bypass the production PRF invariants.
#[cfg(test)]
pub(crate) struct TestOneHotProgram {
    pub(crate) program: PowerLutProgram,
    pub(crate) input: ProgramInputId,
    pub(crate) selector_family: RhsFamilyId,
    pub(crate) public_value_family: PublicValueFamilyId,
    pub(crate) output: ProgramWireId,
    lut_width: usize,
    bucket_width: usize,
    rounding_lut: Vec<usize>,
}

#[cfg(test)]
impl TestOneHotProgram {
    fn lut_width(&self) -> usize {
        self.lut_width
    }

    fn bucket_width(&self) -> usize {
        self.bucket_width
    }

    fn rounding_lut(&self) -> &[usize] {
        &self.rounding_lut
    }
}

#[cfg(test)]
pub(crate) fn test_one_hot_fixture(
    lut_width: usize,
    bucket_width: usize,
    reduction_lut: Option<Vec<usize>>,
    rounding_lut: Option<Vec<usize>>,
) -> Result<TestOneHotProgram, ProgramValidationError> {
    let mut builder = PowerLutProgramBuilder::new();
    let input = builder.input(lut_width)?;
    let selector_family = builder.rhs_family(lut_width)?;
    let public_value_family = builder.public_value_family(lut_width)?;
    let select_lut =
        builder.lut(LutTable::unary(lut_width, lut_width, (0..lut_width).collect())?)?;
    let selected = builder.one_hot(
        builder.input_wire(input)?,
        selector_family,
        public_value_family,
        select_lut,
    )?;
    let output = if let Some(values) = reduction_lut {
        let lut = builder.lut(LutTable::unary(lut_width, lut_width, values)?)?;
        builder.unary(selected, lut)?
    } else {
        selected
    };
    builder.output(output)?;
    let program = builder.build()?;
    let rounding_lut = rounding_lut.unwrap_or_else(|| vec![0; lut_width]);
    Ok(TestOneHotProgram {
        program,
        input,
        selector_family,
        public_value_family,
        output,
        lut_width,
        bucket_width,
        rounding_lut,
    })
}

/// Creates the smallest valid production sparse-LWR program for tests that
/// exercise refresh output identities. Unlike [`test_one_hot_fixture`], this
/// path uses the checked profile constructor and the mandatory PRF LUTs.
#[cfg(test)]
pub(crate) fn test_sparse_lwr_program(
    bucket_width: usize,
    ring_dimension: usize,
) -> Result<SparseLwrPrfProgram, PowerLutError> {
    SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(2, 2, 4, ring_dimension)?, bucket_width)
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
}

/// Opaque output of the real private PBC lowering.
///
/// The layout identity is retained alongside the evaluated wire and its
/// program descriptor.  Keeping all three values together prevents refresh
/// setup from accepting an evaluated wire that was produced for another PBC
/// layout.  The constructor and accessors are crate-private: external callers
/// can only obtain this value by invoking the corresponding private lowering
/// or the public typed batch entry point
/// [`SparseLwrPrfProgram::compile_pbc_encoding_family_typed_with_batch`].
#[derive(Clone)]
pub struct PbcSparseLwrEncodingOutput {
    encoding: BggEncodingWire,
    descriptor: SparseLwrPrfOutput,
    layout_id: crate::pbc::PbcLayoutId,
}

/// Opaque family result of one structural PBC lowering.
///
/// The family is produced by
/// [`SparseLwrPrfProgram::compile_pbc_encoding_family_typed_with_batch`] and
/// can only be projected at a canonical static label index. The
/// projection keeps the output wire and layout identity from that lowering.
pub struct PbcSparseLwrEncodingOutputs {
    vectors: Family<Mat>,
    public_keys: Family<Mat>,
    program_id: PowerLutProgramId,
    output_wire: ProgramWireId,
    layout_id: crate::pbc::PbcLayoutId,
    label_digests: Vec<[u8; 32]>,
    public_vector_identities: Vec<[u8; 32]>,
}

impl PbcSparseLwrEncodingOutputs {
    fn new(
        vectors: Family<Mat>,
        public_keys: Family<Mat>,
        program_id: PowerLutProgramId,
        output_wire: ProgramWireId,
        batch: &RefreshPrfBatchInputs,
    ) -> Self {
        Self {
            vectors,
            public_keys,
            program_id,
            output_wire,
            layout_id: batch.layout_id,
            label_digests: batch.label_digests.clone(),
            public_vector_identities: batch.public_vector_identities.clone(),
        }
    }

    /// Projects one member of the already-lowered family at its immutable
    /// canonical index. The index, rather than a caller-supplied label, is
    /// part of the trusted batch binding.
    pub fn project(&self, index: usize) -> Result<PbcSparseLwrEncodingOutput, PowerLutError> {
        let digest =
            self.label_digests.get(index).copied().ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        // A projection is valid only after both canonical label and routed
        // public-vector identities were attached by the trusted builder.
        // Keeping the two ordered lists in lockstep prevents a descriptor
        // from being produced for an incompletely bound family.
        if self.public_vector_identities.get(index).is_none() {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        self.project_with_digest(index, digest)
    }

    fn project_with_digest(
        &self,
        index: usize,
        digest: [u8; 32],
    ) -> Result<PbcSparseLwrEncodingOutput, PowerLutError> {
        let count = self
            .vectors
            .count()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        if index >= count || self.public_keys.count() != self.vectors.count() {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let encoding = BggEncodingWire {
            vector: self.vectors.get_static(index),
            pubkey: BggPublicKeyWire {
                matrix: self.public_keys.get_static(index),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let descriptor = SparseLwrPrfOutput {
            program_id: self.program_id,
            label_digest: digest,
            output_wire: self.output_wire,
        };
        Ok(PbcSparseLwrEncodingOutput::new(encoding, descriptor, self.layout_id))
    }

    /// Returns the public identity of the routed vector at `index`.
    pub fn public_vector_identity(&self, index: usize) -> Option<[u8; 32]> {
        self.public_vector_identities.get(index).copied()
    }
}

impl PbcSparseLwrEncodingOutput {
    fn new(
        encoding: BggEncodingWire,
        descriptor: SparseLwrPrfOutput,
        layout_id: crate::pbc::PbcLayoutId,
    ) -> Self {
        Self { encoding, descriptor, layout_id }
    }

    pub(crate) fn encoding(&self) -> &BggEncodingWire {
        &self.encoding
    }

    pub(crate) fn descriptor(&self) -> SparseLwrPrfOutput {
        self.descriptor
    }

    pub(crate) fn layout_id(&self) -> crate::pbc::PbcLayoutId {
        self.layout_id
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
        Ok(Self { program_id, label_digest: digest.finalize().into(), output_wire })
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::ProgramGate;

    #[test]
    fn typed_prf_output_binds_program_label_and_output_wire() {
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(2, 2, 4, 4).unwrap(), 2).unwrap();
        let output = program.bind_output(b"label").unwrap();
        assert_eq!(output.program_id(), program.id());
        assert_eq!(output.output_wire(), program.rounding_output);
        assert_ne!(output.output_wire(), program.output);
        assert_ne!(output.label_digest(), program.bind_output(b"other").unwrap().label_digest());
    }

    #[test]
    fn typed_prf_output_uses_composite_final_program_identity() {
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(2, 2, 4, 4).unwrap(), 2).unwrap();
        let output = program.bind_output(b"label").unwrap();
        assert_eq!(output.program_id(), program.id());
        assert_eq!(output.output_wire(), program.rounding_output);
    }

    #[test]
    fn bucket_program_has_one_shared_one_hot_body() {
        let body = test_one_hot_fixture(4, 4, Some(vec![0, 1, 2, 3]), None).unwrap();
        assert_eq!(body.program.outputs(), &[body.output]);
        assert!(matches!(
            body.program.gates().first(),
            Some(ProgramGate::OneHot { selector_family, public_value_family, .. })
                if *selector_family == body.selector_family
                   && *public_value_family == body.public_value_family
        ));
        assert!(matches!(body.program.gates().get(1), Some(ProgramGate::Unary { .. })));
    }

    #[test]
    fn bucket_program_id_is_independent_of_private_schedule() {
        let first = test_one_hot_fixture(4, 4, None, None).unwrap();
        let second = test_one_hot_fixture(4, 4, None, None).unwrap();
        assert_eq!(first.program.id(), second.program.id());
        assert_eq!(first.program.gates().len(), 1);
    }

    #[test]
    fn prf_program_keeps_selection_and_rounding_in_one_description() {
        let body =
            test_one_hot_fixture(4, 4, Some(vec![0, 1, 2, 3]), Some(vec![0, 0, 1, 1])).unwrap();
        assert_eq!(body.program.outputs(), &[body.output]);
        assert_eq!(body.program.gates().len(), 2);
        assert_eq!(body.rounding_lut(), [0, 0, 1, 1]);
        assert!(body.program.gates().iter().any(|gate| matches!(gate, ProgramGate::OneHot { .. })));
    }

    #[test]
    fn lut_width_is_independent_of_rectangular_bucket_width() {
        let body = test_one_hot_fixture(2, 7, Some(vec![0, 1]), Some(vec![0, 1])).unwrap();
        assert_eq!(body.lut_width(), 2);
        assert_eq!(body.bucket_width(), 7);
        assert_eq!(body.program.inputs().get(&body.input), Some(&2));
        assert_eq!(body.program.rhs_family_width(body.selector_family), Some(2));
        assert_eq!(body.program.public_value_family_width(body.public_value_family), Some(2));
        assert_eq!(body.program.gates().len(), 2);
        assert_eq!(body.rounding_lut(), [0, 1]);
    }

    #[test]
    fn concrete_profile_checks_all_reviewed_invariants() {
        let profile = SparseLwrPrfProfile::new(2, 2, 4, 8).unwrap();
        assert_eq!(profile.q_l(), 2);
        assert_eq!(profile.p(), 2);
        assert_eq!(profile.lut_width(), 4);
        assert_eq!(profile.ring_dimension(), 8);
        for invalid in [
            (0, 2, 4, 8),
            (2, 1, 4, 8),
            (2, 3, 4, 8),
            (2, 2, 3, 8),
            (2, 2, 16, 8),
            (2, 2, 8, 6),
            (2, 2, 8, 0),
            (3, 2, 2, 8),
        ] {
            assert!(
                SparseLwrPrfProfile::new(invalid.0, invalid.1, invalid.2, invalid.3).is_err(),
                "invalid profile {invalid:?} was accepted"
            );
        }
        assert!(SparseLwrPrfProfile::new(3, 2, 8, 8).is_ok());
        assert!(SparseLwrPrfProfile::new(3, 2, 4, 8).is_err());
    }

    #[test]
    fn constructor_derives_mandatory_reduction_and_final_rounding_programs() {
        let profile = SparseLwrPrfProfile::new(2, 2, 4, 8).unwrap();
        let body = SparseLwrPrfProgram::new(profile, 7).unwrap();
        assert_eq!(body.program.gates().len(), 2);
        assert_eq!(body.rounding_program().gates().len(), 1);
        let reduction = match body.program.gates()[1] {
            ProgramGate::Unary { lut, .. } => body.program.lut(lut).unwrap().values(),
            _ => panic!("missing mandatory reduction gate"),
        };
        assert_eq!(reduction, &[0, 1, 0, 1]);
        let rounding = match body.rounding_program().gates()[0] {
            ProgramGate::Unary { lut, .. } => body.rounding_program().lut(lut).unwrap().values(),
            _ => panic!("missing mandatory rounding gate"),
        };
        assert_eq!(rounding, &[0, 1, 0, 1]);
        assert_ne!(body.id(), body.program.id());
    }

    #[test]
    fn refresh_fixture_uses_the_checked_profile_for_both_lut_stages() {
        // This is the concrete profile used by the small refresh fixtures.
        // Keeping the assertion at the fixture constructor catches accidental
        // reintroduction of a bucket-width-only or optional LUT path.
        let body = test_sparse_lwr_program(2, 4).unwrap();
        assert_eq!(body.profile().q_l(), 2);
        assert_eq!(body.profile().p(), 2);
        assert_eq!(body.profile().lut_width(), 4);
        assert_eq!(body.profile().ring_dimension(), 4);
        assert_eq!(body.rounding_lut(), [0, 1, 0, 1]);
        let reduction = match body.program.gates()[1] {
            ProgramGate::Unary { lut, .. } => body.program.lut(lut).unwrap().values(),
            _ => panic!("missing mandatory per-bucket reduction gate"),
        };
        assert_eq!(reduction, &[0, 1, 0, 1]);
    }

    #[test]
    fn refresh_batch_derives_immutable_public_identities_in_canonical_order() {
        let parameters = crate::pbc::PbcParameters::custom(4, 1, 2, 2, 1, None);
        let layout =
            crate::pbc::PbcPublicLayout::build(&parameters, crate::pbc::PbcLayoutSeed([9; 32]), 0)
                .unwrap();
        let profile = SparseLwrPrfProfile::new(3, 2, 8, 8).unwrap();
        let labels = crate::refresh::RefreshPrfLabelIndex::new([3; 32], 1, 1, 1, 2).unwrap();
        let first = RefreshPrfBatchInputs::new(&layout, profile.clone(), &labels).unwrap();
        let second = RefreshPrfBatchInputs::new(&layout, profile, &labels).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), labels.len());
        assert!(matches!(labels.label(0), Some(crate::refresh::RefreshPrfLabel::Mask { .. })));
        assert!(matches!(
            labels.label(2),
            Some(crate::refresh::RefreshPrfLabel::FreshError { .. })
        ));
        assert!(first.label_digest(0).is_some());
        assert!(first.public_vector_identity(0).is_some());

        // A different refresh identity changes both the canonical label
        // bytes and the derived public vector, while preserving the same
        // structural index order.
        let other_labels = crate::refresh::RefreshPrfLabelIndex::new([4; 32], 1, 1, 1, 2).unwrap();
        let other = RefreshPrfBatchInputs::new(
            &layout,
            SparseLwrPrfProfile::new(3, 2, 8, 8).unwrap(),
            &other_labels,
        )
        .unwrap();
        assert_ne!(first.label_digest(0), other.label_digest(0));
        assert_ne!(first.public_vector_identity(0), other.public_vector_identity(0));
        assert_ne!(first.batch_id, other.batch_id);
    }

    #[test]
    fn batch_public_values_are_flattened_label_major_in_active_cell_order() {
        let parameters = crate::pbc::PbcParameters::custom(4, 1, 2, 2, 1, None);
        let layout =
            crate::pbc::PbcPublicLayout::build(&parameters, crate::pbc::PbcLayoutSeed([17; 32]), 0)
                .unwrap();
        let profile = SparseLwrPrfProfile::new(2, 2, 4, 4).unwrap();
        let labels = crate::refresh::RefreshPrfLabelIndex::new([5; 32], 1, 1, 1, 2).unwrap();
        let batch = RefreshPrfBatchInputs::new(&layout, profile, &labels).unwrap();
        let ring = mxx_dsl::Ring::new(17, 4);
        let family = batch_public_values_family(&batch, &ring).unwrap();

        let mxx_ir_core::node::NodeKind::Input { name, wire_type, .. } =
            family.value_handle().node().kind()
        else {
            panic!("the helper must produce one runtime public-value family")
        };
        assert_eq!(name, &batch.public_input_name());
        assert_eq!(
            wire_type,
            &mxx_ir_core::types::WireType::IndexedFamily {
                element: Box::new(mxx_ir_core::types::WireType::Matrix(ring.matrix_type((1, 1)))),
                count: mxx_ir_core::IntExpr::constant(batch.value_count()),
            }
        );
        assert!(!family.value_handle().node().arguments().iter().any(|argument| matches!(
            argument.node().kind(),
            mxx_ir_core::node::NodeKind::ConstantMatrix { .. }
        )));
    }

    #[test]
    fn interpolation_lookup_is_exact_for_irregular_bucket_metadata() {
        let values = [0usize, 3, 1, 4, 2];
        let expression = interpolate_lookup(&values, mxx_ir_core::IntExpr::LoopIndex(7)).unwrap();
        for (index, expected) in values.iter().copied().enumerate() {
            let environment = mxx_ir_core::ParamEnv {
                loop_indices: std::collections::BTreeMap::from([(7u32, BigInt::from(index))]),
                ..mxx_ir_core::ParamEnv::default()
            };
            assert_eq!(expression.evaluate(&environment).unwrap(), BigInt::from(expected));
        }
    }

    #[test]
    fn batched_pbc_lowering_uses_one_outer_label_loop() {
        let layout = crate::pbc::PbcPublicLayout::build(
            &crate::pbc::PbcParameters::custom(1, 1, 2, 2, 1, None),
            crate::pbc::PbcLayoutSeed([31; 32]),
            0,
        )
        .unwrap();
        let ring = mxx_dsl::Ring::new(17, 4);
        let compiler = PowerLutEncodingCompiler::from_public_key(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 2.into(),
        });
        let input_vectors = Family::pack(
            (0..2).map(|index| ring.input(format!("batch-vector-{index}"), (1, 4))).collect(),
        )
        .unwrap();
        let input_public_keys = Family::pack(
            (0..2).map(|index| ring.input(format!("batch-public-{index}"), (2, 4))).collect(),
        )
        .unwrap();
        let selector_count = crate::pbc::PbcActiveCellIndex::build(&layout).unwrap().len();
        let gsw = Family::pack(
            (0..selector_count)
                .map(|index| ring.input(format!("batch-gsw-{index}"), (2, 4)))
                .collect(),
        )
        .unwrap();
        let companions = (0..8)
            .map(|index| {
                (
                    Family::pack(
                        (0..selector_count)
                            .map(|item| {
                                ring.input(format!("batch-companion-v-{index}-{item}"), (1, 8))
                            })
                            .collect(),
                    )
                    .unwrap(),
                    Family::pack(
                        (0..selector_count)
                            .map(|item| {
                                ring.input(format!("batch-companion-p-{index}-{item}"), (2, 8))
                            })
                            .collect(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let selectors = EncodingSelectorFamily::new(gsw, companions).unwrap();
        // The production body includes the mandatory W = 4 coefficient
        // reduction LUT.  Its lowering therefore needs the two sampled
        // automorphism rounds used by ClearCoeff, even though this test does
        // not declare an explicit automorphism gate of its own.
        let sampler = crate::encoding::PowerLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 17.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 2.into(),
            },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let helpers = sampler
            .sample_automorphism_helpers(
                ring.zero((1, 2)),
                ring.bytes_input("batch-helper-key", 32),
                &b"batch-helpers"[..],
                4,
            )
            .unwrap();
        assert_eq!(helpers.len(), 2);
        let body = test_sparse_lwr_program(layout.bucket_width, 4).unwrap();
        let labels = crate::refresh::RefreshPrfLabelIndex::new([3; 32], 0, 1, 1, 2).unwrap();
        let batch = RefreshPrfBatchInputs::new(&layout, body.profile().clone(), &labels).unwrap();
        let wrong_input_vectors =
            Family::pack(vec![ring.input("wrong-batch-vector", (1, 4))]).unwrap();
        assert!(
            body.compile_pbc_encoding_family_typed_with_batch(
                &compiler,
                wrong_input_vectors,
                input_public_keys.clone(),
                &batch,
                selectors.clone(),
                &helpers,
            )
            .is_err()
        );
        let outputs = body
            .compile_pbc_encoding_family_typed_with_batch(
                &compiler,
                input_vectors,
                input_public_keys,
                &batch,
                selectors,
                &helpers,
            )
            .unwrap();
        let projected = outputs.project(0).unwrap();
        assert_eq!(projected.descriptor().program_id(), body.id());
        assert_eq!(projected.descriptor().output_wire(), body.rounding_output);
        assert_eq!(projected.layout_id(), layout.layout_id);
        assert_eq!(outputs.public_vector_identity(0), batch.public_vector_identity(0));
        assert!(outputs.project(2).is_err());
        let built = mxx_dsl::DslContext::new("batched-pbc-labels")
            .output("vector", projected.encoding().vector.clone())
            .unwrap()
            .build()
            .unwrap();
        // Mandatory reduction and final-rounding LUTs add structural nodes;
        // keep the assertion focused on preventing per-cell graph expansion.
        assert!(built.graph.root_scope().nodes().len() < 512);
        let (outer_position, outer_node) = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .enumerate()
            .find(|(_, node)| {
                matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::ParallelLoop(spec)
                        if spec.count == mxx_ir_core::IntExpr::constant(2)
                )
            })
            .expect("outer label loop");
        let mxx_ir_core::node::NodeKind::ParallelLoop(outer_spec) = outer_node.kind() else {
            unreachable!()
        };
        assert_eq!(
            outer_spec.input_modes[0..2],
            [mxx_ir_core::node::LoopInputMode::Zip, mxx_ir_core::node::LoopInputMode::Zip,]
        );
        assert!(
            outer_spec.input_modes[2..]
                .iter()
                .all(|mode| *mode == mxx_ir_core::node::LoopInputMode::Broadcast)
        );
        let outer_body_id = built
            .graph
            .child_scope_id(
                &mxx_ir_core::FrozenGraphScopeId::Root,
                mxx_ir_core::NodeId(outer_position as u64),
            )
            .expect("outer label body");
        let outer_body = built.graph.scope(&outer_body_id).expect("outer body scope");
        let (bucket_position, _bucket_node) = outer_body
            .nodes()
            .iter()
            .enumerate()
            .find(|(_, node)| matches!(node.kind(), mxx_ir_core::node::NodeKind::SequentialLoop(_)))
            .expect("nested sequential bucket loop");
        let bucket_body_id = built
            .graph
            .child_scope_id(&outer_body_id, mxx_ir_core::NodeId(bucket_position as u64))
            .expect("bucket body");
        assert!(
            built
                .graph
                .scope(&bucket_body_id)
                .expect("bucket body scope")
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(_)))
        );
    }

    #[test]
    #[serial_test::serial(dcrt_runtime)]
    fn nested_label_loop_executes_on_cpu_with_bounded_waves() {
        use mxx_ir_core::ParamEnv;
        use mxx_primitives::{
            matrix::dcrt_poly::DCRTPolyMatrix,
            poly::{PolyParams, dcrt::params::DCRTPolyParams},
        };
        use mxx_runtime::{
            RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend,
            execute_with_config, executor::ExecutionConfig, transcript::SamplingMode,
        };
        use std::num::NonZeroUsize;

        let parameters = DCRTPolyParams::new(4, 2, 17, 1);
        let ring = mxx_dsl::Ring::new(
            BigInt::from_biguint(num_bigint::Sign::Plus, parameters.modulus().as_ref().clone()),
            4,
        );
        let labels = ring.input_family("runtime-labels", 2, (1, 1));
        let cells = ring.input_family("runtime-cells", 2, (1, 1));
        let output = Family::<Mat>::parallel_zip_many_with_broadcast_values(
            vec![labels],
            vec![cells],
            |_label, mut zipped, mut broadcast| {
                let state = zipped.pop().ok_or(mxx_dsl::DslError::Schema)?;
                let cells = broadcast.pop().ok_or(mxx_dsl::DslError::Schema)?;
                Sequential::range(2).scan(state, cells, |bucket, state, cells| {
                    let mapped = cells.parallel_map_values(|_, value| value);
                    let index = Family::<mxx_dsl::Int>::pack(vec![bucket.as_int()])?;
                    Ok(state + mapped?.parallel_gather(index)?.get_static(0))
                })
            },
        )
        .unwrap();
        let built = mxx_dsl::DslContext::new("nested-label-runtime")
            .output("result", output.get_static(0))
            .unwrap()
            .build()
            .unwrap();
        let graph = built.validate(&ParamEnv::default()).unwrap();
        let zero = DCRTPolyMatrix::zero(&parameters, 1, 1);
        let inputs = std::collections::BTreeMap::from([
            (
                "runtime-labels".to_owned(),
                RuntimeValue::IndexedFamily(vec![
                    RuntimeValue::matrix(zero.clone()),
                    RuntimeValue::matrix(zero.clone()),
                ]),
            ),
            (
                "runtime-cells".to_owned(),
                RuntimeValue::IndexedFamily(vec![
                    RuntimeValue::matrix(zero.clone()),
                    RuntimeValue::matrix(zero),
                ]),
            ),
        ]);
        for cap in [1, 2] {
            let result = execute_with_config(
                &graph,
                &mut cpu_backend([parameters.clone()]),
                inputs.clone(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
                ExecutionConfig {
                    max_parallel_instances: NonZeroUsize::new(cap).unwrap(),
                    ..ExecutionConfig::default()
                },
            )
            .unwrap();
            assert!(matches!(result.outputs.get("result"), Some(RuntimeValue::Matrix(_))));
        }
    }

    #[test]
    fn pbc_compiler_rejects_a_family_not_bound_to_the_public_vector() {
        let parameters = crate::pbc::PbcParameters::custom(2, 1, 2, 2, 1, None);
        let layout =
            crate::pbc::PbcPublicLayout::build(&parameters, crate::pbc::PbcLayoutSeed([19; 32]), 0)
                .expect("toy PBC layout");
        let encoded = crate::pbc::PbcEncodedPublicVector::route_usize(&layout, &[3, 5], 17)
            .expect("route public vector");
        let binding = crate::pbc::PbcPublicVectorFamilyBinding::from_encoded(&layout, &encoded)
            .expect("bind public vector");
        let ring = mxx_dsl::Ring::new(17, 4);
        let wrong_family =
            ring.input_family("unrelated-pbc-values", binding.family_count - 1, (1, 1));
        let selectors = EncodingSelectorFamily::new(
            ring.input_family("selector-gsw", 1, (1, 1)),
            vec![(
                ring.input_family("selector-vector", 1, (1, 1)),
                ring.input_family("selector-public", 1, (1, 1)),
            )],
        )
        .expect("toy selector family");
        let body = test_sparse_lwr_program(layout.bucket_width, 4).expect("toy sparse-LWR program");
        let public_key_compiler = mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: mxx_ir_core::IntExpr::constant(2),
            digit_count: mxx_ir_core::IntExpr::constant(2),
        };
        let compiler = PowerLutEncodingCompiler::from_public_key(public_key_compiler);
        let input = BggEncodingWire {
            vector: ring.zero((1, 1)),
            pubkey: BggPublicKeyWire { matrix: ring.zero((1, 1)), reveal_plaintext: false },
            plaintext: None,
        };

        let result = body.compile_pbc_encoding(
            &compiler,
            input,
            &layout,
            &binding,
            selectors,
            wrong_family,
            &[],
            b"binding-test",
        );
        assert!(matches!(result, Err(PowerLutError::InvalidSparseLwrBlock)));
    }
}
