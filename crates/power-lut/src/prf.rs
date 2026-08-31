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
use mxx_dsl::{Bool, Family, Int, LoopIndex, Mat, Parallel, Sequential};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    PowerLutEncodingCompiler, PowerLutError,
    encoding::{EncodingSelectorFamily, FlatLutHelper, FlatLutHelperMap, FlatLutHelperSet},
    program::{
        FamilyRange, LutTable, PowerLutProgram, PowerLutProgramBuilder, PowerLutProgramId,
        ProgramFamilyRanges, ProgramInputId, ProgramValidationError, ProgramWireId,
        PublicValueFamilyId, RhsFamilyId,
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
/// Host PBC materialization remains residue-valued, while each runtime family
/// element bound to this DSL input must be the corresponding ring monomial
/// `X^a`. This distinction is part of the one-hot gate's public-factor
/// contract and is therefore documented at the single family declaration
/// boundary used by both private and public-key lowering.
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
/// The mandatory reduction table is the one-hot gate's single unary LUT. The
/// mandatory rounding table is a second shared program applied once after the
/// bucket scan and never per bucket. Its terminal LUT has the raw-scalar form,
/// so the result is a constant polynomial rather than a monomial encoding.
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
        helpers: &FlatLutHelperMap,
        label: &[u8],
    ) -> Result<(BggEncodingWire, SparseLwrPrfOutput), PowerLutError> {
        self.compile_encoding_inner(compiler, input, selectors, public_values, helpers, label, None)
    }

    /// Private lowering entry point with the separately sampled terminal
    /// rounding helper set.  Bucket helpers and rounding helpers are kept as
    /// distinct arguments because their local LUT identifiers are independent.
    pub fn compile_encoding_with_rounding_helpers(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        selectors: BTreeMap<RhsFamilyId, EncodingSelectorFamily>,
        public_values: BTreeMap<PublicValueFamilyId, mxx_dsl::Family<Mat>>,
        helpers: &FlatLutHelperMap,
        rounding_helpers: &FlatLutHelperSet,
        label: &[u8],
    ) -> Result<(BggEncodingWire, SparseLwrPrfOutput), PowerLutError> {
        self.compile_encoding_inner(
            compiler,
            input,
            selectors,
            public_values,
            helpers,
            label,
            Some(rounding_helpers),
        )
    }

    fn compile_encoding_inner(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input: BggEncodingWire,
        selectors: BTreeMap<RhsFamilyId, EncodingSelectorFamily>,
        public_values: BTreeMap<PublicValueFamilyId, mxx_dsl::Family<Mat>>,
        helpers: &FlatLutHelperMap,
        label: &[u8],
        rounding_helpers: Option<&FlatLutHelperSet>,
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
        let output = self.apply_final_rounding_encoding(compiler, output, rounding_helpers)?;
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
        helpers: &FlatLutPublicHelperMap,
        label: &[u8],
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        self.compile_public_key_inner(
            compiler,
            input,
            selectors,
            public_values,
            helpers,
            label,
            None,
        )
    }

    /// Public-key lowering entry point with the setup-fixed terminal helper
    /// set supplied separately from bucket helper material.
    pub fn compile_public_key_with_rounding_helpers(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input: BggPublicKeyWire,
        selectors: BTreeMap<RhsFamilyId, PublicSelectorFamily>,
        public_values: BTreeMap<PublicValueFamilyId, mxx_dsl::Family<Mat>>,
        helpers: &FlatLutPublicHelperMap,
        rounding_helpers: &FlatLutPublicHelperSet,
        label: &[u8],
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        self.compile_public_key_inner(
            compiler,
            input,
            selectors,
            public_values,
            helpers,
            label,
            Some(rounding_helpers),
        )
    }

    fn compile_public_key_inner(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input: BggPublicKeyWire,
        selectors: BTreeMap<RhsFamilyId, PublicSelectorFamily>,
        public_values: BTreeMap<PublicValueFamilyId, mxx_dsl::Family<Mat>>,
        helpers: &FlatLutPublicHelperMap,
        label: &[u8],
        rounding_helpers: Option<&FlatLutPublicHelperSet>,
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
        let output = self.apply_final_rounding_public(compiler, output, rounding_helpers)?;
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
        helpers: &FlatLutHelperMap,
        label: &[u8],
    ) -> Result<(BggEncodingWire, SparseLwrPrfOutput), PowerLutError> {
        self.compile_encoding_sequential_inner(
            compiler,
            input,
            bucket_count,
            bucket_bindings,
            helpers,
            label,
            None,
        )
    }

    /// Structural sequential lowering with the explicitly supplied terminal
    /// scalar-rounding helper set.
    pub fn compile_encoding_sequential_with_rounding_helpers(
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
        helpers: &FlatLutHelperMap,
        rounding_helpers: &FlatLutHelperSet,
        label: &[u8],
    ) -> Result<(BggEncodingWire, SparseLwrPrfOutput), PowerLutError> {
        self.compile_encoding_sequential_inner(
            compiler,
            input,
            bucket_count,
            bucket_bindings,
            helpers,
            label,
            Some(rounding_helpers),
        )
    }

    fn compile_encoding_sequential_inner(
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
        helpers: &FlatLutHelperMap,
        label: &[u8],
        rounding_helpers: Option<&FlatLutHelperSet>,
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
        let output = self.apply_final_rounding_encoding(compiler, final_state, rounding_helpers)?;
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
        helpers: &FlatLutHelperMap,
        label: &[u8],
        rounding_helpers: Option<&FlatLutHelperSet>,
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
        let output = self.apply_final_rounding_encoding(compiler, final_state, rounding_helpers)?;
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
        helpers: &FlatLutPublicHelperMap,
        label: &[u8],
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        self.compile_public_key_sequential_inner(
            compiler,
            input,
            bucket_count,
            bucket_bindings,
            helpers,
            label,
            None,
        )
    }

    /// Public structural sequential lowering with the setup-fixed terminal
    /// rounding helpers supplied separately from bucket helpers.
    pub fn compile_public_key_sequential_with_rounding_helpers(
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
        helpers: &FlatLutPublicHelperMap,
        rounding_helpers: &FlatLutPublicHelperSet,
        label: &[u8],
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        self.compile_public_key_sequential_inner(
            compiler,
            input,
            bucket_count,
            bucket_bindings,
            helpers,
            label,
            Some(rounding_helpers),
        )
    }

    fn compile_public_key_sequential_inner(
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
        helpers: &FlatLutPublicHelperMap,
        label: &[u8],
        rounding_helpers: Option<&FlatLutPublicHelperSet>,
    ) -> Result<(BggPublicKeyWire, SparseLwrPrfOutput), PowerLutError> {
        // Public lowering carries only A. Each selected concrete C_i applies
        // `A^\sigma G^{-1}(C_i)`, and the public factor `X^{a'_{b}[i]}` updates
        // the same accumulator exponent as in the private recurrence.
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
        let output = self.apply_final_rounding_public(compiler, final_state, rounding_helpers)?;
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
        helpers: &FlatLutPublicHelperMap,
        label: &[u8],
        rounding_helpers: Option<&FlatLutPublicHelperSet>,
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
        let output = self.apply_final_rounding_public(compiler, final_state, rounding_helpers)?;
        Ok((output, self.bind_output(label)?))
    }

    /// Lowers a family of independent PBC labels through one outer
    /// structural parallel loop. The input families are zipped by label;
    /// selector families and label-major public values are explicit outer
    /// broadcast inputs, then become invariants of the nested sequential
    /// bucket loop. Public values are flattened in
    /// canonical label-major, active-cell order, and each bucket iteration
    /// selects only its own contiguous active-cell range.
    ///
    /// It deliberately returns families rather than a host vector of wires,
    /// so callers can continue routing and reducing the labels structurally.
    pub(crate) fn compile_pbc_encoding_family(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &FlatLutHelperMap,
    ) -> Result<(Family<Mat>, Family<Mat>), PowerLutError> {
        self.compile_pbc_encoding_family_inner(
            compiler,
            input_vectors,
            input_public_keys,
            batch,
            selectors,
            helpers,
            None,
        )
    }

    /// Batched PBC lowering with one explicit terminal rounding helper set.
    pub(crate) fn compile_pbc_encoding_family_with_rounding_helpers(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &FlatLutHelperMap,
        rounding_helpers: &FlatLutHelperSet,
    ) -> Result<(Family<Mat>, Family<Mat>), PowerLutError> {
        self.compile_pbc_encoding_family_inner(
            compiler,
            input_vectors,
            input_public_keys,
            batch,
            selectors,
            helpers,
            Some(rounding_helpers),
        )
    }

    fn compile_pbc_encoding_family_inner(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &FlatLutHelperMap,
        rounding_helpers: Option<&FlatLutHelperSet>,
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
        ensure_family_count(&public_values, expected_values)?;

        let selector_flat = selectors.flattened();
        let selector_flat_count = selector_flat.len();
        let (helper_flat, helper_arities) = flatten_private_helper_families(helpers)?;
        let (rounding_flat, rounding_arities) = if let Some(rounding_helpers) = rounding_helpers {
            flatten_private_helper_families(&FlatLutHelperMap::from([(
                crate::program::LutId::from_index(0),
                rounding_helpers.clone(),
            )]))?
        } else {
            (Vec::new(), Vec::new())
        };
        let mut broadcasts = selector_flat;
        broadcasts.push(public_values);
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
                let rounding_map = if rounding_arities.is_empty() {
                    None
                } else {
                    Some(rebuild_private_helper_families(rounding_broadcasts, &rounding_arities)?)
                };
                let rounding_helpers = rounding_map
                    .as_ref()
                    .map(|map| {
                        map.get(&crate::program::LutId::from_index(0))
                            .ok_or(mxx_dsl::DslError::Schema)
                    })
                    .transpose()?;
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
        helpers: &FlatLutHelperMap,
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

    /// Typed batched private lowering with the setup-fixed terminal rounding
    /// helpers.  The resulting family has the scalar terminal operation
    /// already executed in the structural label loop.
    pub fn compile_pbc_encoding_family_typed_with_batch_and_rounding_helpers(
        &self,
        compiler: &PowerLutEncodingCompiler,
        input_vectors: Family<Mat>,
        input_public_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: EncodingSelectorFamily,
        helpers: &FlatLutHelperMap,
        rounding_helpers: &FlatLutHelperSet,
    ) -> Result<PbcSparseLwrEncodingOutputs, PowerLutError> {
        if batch.layout_id != batch.layout.layout_id || batch.profile != self.profile {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let (vectors, public_keys) = self.compile_pbc_encoding_family_with_rounding_helpers(
            compiler,
            input_vectors,
            input_public_keys,
            batch,
            selectors,
            helpers,
            rounding_helpers,
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
        helpers: &FlatLutPublicHelperMap,
    ) -> Result<Family<Mat>, PowerLutError> {
        self.compile_pbc_public_key_family_inner(
            compiler, input_keys, batch, selectors, helpers, None,
        )
    }

    /// Public batched PBC lowering with explicit terminal rounding helpers.
    pub(crate) fn compile_pbc_public_key_family_with_rounding_helpers(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &FlatLutPublicHelperMap,
        rounding_helpers: &FlatLutPublicHelperSet,
    ) -> Result<Family<Mat>, PowerLutError> {
        self.compile_pbc_public_key_family_inner(
            compiler,
            input_keys,
            batch,
            selectors,
            helpers,
            Some(rounding_helpers),
        )
    }

    fn compile_pbc_public_key_family_inner(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &FlatLutPublicHelperMap,
        rounding_helpers: Option<&FlatLutPublicHelperSet>,
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
        let (rounding_flat, rounding_arities) = if let Some(rounding_helpers) = rounding_helpers {
            flatten_public_helper_families(&FlatLutPublicHelperMap::from([(
                crate::program::LutId::from_index(0),
                rounding_helpers.clone(),
            )]))?
        } else {
            (Vec::new(), Vec::new())
        };
        let mut broadcasts = selector_flat;
        broadcasts.push(public_values);
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
                let rounding_map = if rounding_arities.is_empty() {
                    None
                } else {
                    Some(rebuild_public_helper_families(rounding_broadcasts, &rounding_arities)?)
                };
                let rounding_helpers = rounding_map
                    .as_ref()
                    .map(|map| {
                        map.get(&crate::program::LutId::from_index(0))
                            .ok_or(mxx_dsl::DslError::Schema)
                    })
                    .transpose()?;
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
        helpers: &FlatLutPublicHelperMap,
    ) -> Result<Family<Mat>, PowerLutError> {
        if batch.layout_id != batch.layout.layout_id || batch.profile != self.profile {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        self.compile_pbc_public_key_family(compiler, input_keys, batch, selectors, helpers)
    }

    /// Typed public batched lowering with the explicit terminal rounding
    /// helper family supplied by setup.
    pub fn compile_pbc_public_key_family_with_batch_and_rounding_helpers(
        &self,
        compiler: &PowerLutPublicKeyCompiler,
        input_keys: Family<Mat>,
        batch: &RefreshPrfBatchInputs,
        selectors: PublicSelectorFamily,
        helpers: &FlatLutPublicHelperMap,
        rounding_helpers: &FlatLutPublicHelperSet,
    ) -> Result<Family<Mat>, PowerLutError> {
        if batch.layout_id != batch.layout.layout_id || batch.profile != self.profile {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        self.compile_pbc_public_key_family_with_rounding_helpers(
            compiler,
            input_keys,
            batch,
            selectors,
            helpers,
            rounding_helpers,
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
        rounding_helpers: Option<&FlatLutHelperSet>,
    ) -> Result<BggEncodingWire, PowerLutError> {
        let rounding_helpers = rounding_helpers.ok_or(PowerLutError::MissingRoundingHelpers)?;
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
        rounding_helpers: Option<&FlatLutPublicHelperSet>,
    ) -> Result<BggPublicKeyWire, PowerLutError> {
        let rounding_helpers = rounding_helpers.ok_or(PowerLutError::MissingRoundingHelpers)?;
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
    let input_wire = builder.input_wire(input)?;
    let reduction = builder.lut(LutTable::unary(lut_width, lut_width, reduction_lut.to_vec())?)?;
    // The public family already supplies the selected ring factor.  Applying
    // the reduction table directly in `one_hot` keeps the single LUT's
    // monomial output bound to the bucket's modulo-Q semantics; an identity
    // LUT followed by a second unary gate would rebind the factor and obscure
    // that contract.
    let output = builder.one_hot(input_wire, selector_family, public_value_family, reduction)?;
    builder.output(output)?;
    let program = builder.build()?;
    Ok(SparseLwrBucketProgram { program, input, selector_family, public_value_family, output })
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
            build_rounding_program(w, profile.p, &rounding_lut).map_err(PowerLutError::from)?;
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
            terminal_form: SparseLwrPrfTerminalForm::RawScalar,
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
    use mxx_dsl::Ring;

    #[test]
    fn bucket_program_uses_one_reduction_lut_in_one_hot() {
        let program = SparseLwrPrfProgram::new(
            SparseLwrPrfProfile::new(2, 2, 4, 4).expect("test profile"),
            1,
        )
        .expect("test PRF program");

        assert!(program.program.lut(crate::program::LutId::from_index(0)).is_some());
        assert!(program.program.lut(crate::program::LutId::from_index(1)).is_none());
        assert_eq!(program.program.gates().len(), 1);
        assert!(matches!(
            program.program.gates()[0],
            crate::program::ProgramGate::OneHot { lut, .. }
                if lut == crate::program::LutId::from_index(0)
        ));
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
            program.apply_final_rounding_encoding(&compiler, state, Some(&wrong_set)),
            Err(PowerLutError::InvalidLut)
        ));
    }
}
