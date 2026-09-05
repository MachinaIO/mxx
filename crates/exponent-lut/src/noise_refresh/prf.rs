//! Native sparse-key LWR PRF declarations.
//!
//! The PRF accumulator is a native mod-Q exponent program. Its cryptographic
//! steps are the frozen family `MulPrivate`/`MulPublic` and `Add` operations
//! exposed by the shared Exponent-LUT builder; no terminal LUT is present.
//! PBC setup material and online BGG+ wires are bound by callers; no label
//! value is stored in this module's program identity.

use crate::{
    ExponentLutError,
    encoding::{EncodingRhsFamily, ExponentLutEncodingCompiler, NativeReadoutHelperFamilies},
    layout::ExpLayout,
    noise_refresh::{
        pbc::{PbcActiveCellIndex, PbcLayoutId, PbcPublicLayout},
        readout::{
            NormalizationTargetDescriptor, ReadoutPreimageSchema, ValueReadout, ValueReadoutError,
        },
    },
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, ExponentLutProgramId,
        FamilyRange, Gate, ProgramFamilyBindings, ProgramInputId, ProgramStep, ProgramWireType,
        RhsFamilyId, WireFamilyId, WireId,
    },
    public_key::{ProgramCacheBatchBindings, ProgramCacheBindings},
};
use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{Family, Mat, Parallel, ParallelZipTuple, Preimage, Ring};
use mxx_ir_core::{BenchmarkRole, IntExpr, with_benchmark_role};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(test)]
use crate::encoding::LutHelperSet;
#[cfg(test)]
use crate::noise_refresh::{artifacts::ReadoutCacheFamily, readout::ValueReadoutHelper};
#[cfg(test)]
use mxx_ir_core::artifact::Manifest;

/// Native sparse-LWR profile.
///
/// The ring dimension is derived from the native layout as
/// `n = 2^(m + width_log2)`. Consequently the exponent modulus is
/// `Q = 2^(width_log2 + 1)` and the balanced digit base is `P = 2*p_prime`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SparseLwrPrfProfile {
    pub layout: ExpLayout,
    pub p_prime: usize,
}

impl SparseLwrPrfProfile {
    /// Validates a deserialized or otherwise externally assembled profile.
    /// Public fields are intentional for declarative manifests, therefore all
    /// consuming boundaries must call this check instead of trusting them.
    pub fn validate(&self) -> Result<(), ExponentLutError> {
        if self.p_prime == 0 {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let ring_log2 = self
            .layout
            .stride_log2
            .checked_add(self.layout.width_log2)
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let ring_dimension =
            1usize.checked_shl(ring_log2).ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        self.layout
            .validate(ring_dimension)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let q = 1usize
            .checked_shl(self.layout.width_log2 + 1)
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let p = self.p_prime.checked_mul(2).ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        if p < 2 || p > q || q % p != 0 {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(())
    }

    /// Validates a native profile and derives its ring from the full layout.
    pub fn new(layout: ExpLayout, p_prime: usize) -> Result<Self, ExponentLutError> {
        let profile = Self { layout, p_prime };
        profile.validate()?;
        Ok(profile)
    }

    pub const fn native_ring_log2(&self) -> u32 {
        self.layout.stride_log2 + self.layout.width_log2
    }

    pub fn ring_dimension(&self) -> Result<usize, ExponentLutError> {
        1usize.checked_shl(self.native_ring_log2()).ok_or(ExponentLutError::InvalidSparseLwrBlock)
    }

    pub fn q(&self) -> Result<usize, ExponentLutError> {
        1usize
            .checked_shl(self.layout.width_log2 + 1)
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)
    }

    pub fn p(&self) -> Result<usize, ExponentLutError> {
        self.p_prime.checked_mul(2).ok_or(ExponentLutError::InvalidSparseLwrBlock)
    }

    pub const fn native_width(&self) -> usize {
        1usize << self.layout.width_log2
    }

    /// Returns the canonical sparse-LWR value table
    /// `f(k) = floor(p' * k / n') + 1` for `0 <= k < n'`.
    pub fn canonical_readout(&self) -> Result<ValueReadout, ExponentLutError> {
        let n_prime = self.native_width();
        let values = (0..n_prime)
            .map(|k| self.p_prime.checked_mul(k).map(|value| value / n_prime + 1))
            .collect::<Option<Vec<_>>>()
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        ValueReadout::from_canonical_values(self.layout, values)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)
    }
}

impl<'de> Deserialize<'de> for SparseLwrPrfProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawProfile {
            layout: ExpLayout,
            p_prime: usize,
        }
        let raw = RawProfile::deserialize(deserializer)?;
        let profile = Self { layout: raw.layout, p_prime: raw.p_prime };
        profile.validate().map_err(serde::de::Error::custom)?;
        Ok(profile)
    }
}

/// Online label input for the native PRF.
///
/// The only clear value channel is `BggEncodingWire::plaintext`. In
/// particular this type does not carry a public-key matrix. Setup public-key
/// decompositions are consumed by the program cache at each multiplication.
#[derive(Clone)]
pub(crate) struct NativePrfLabelInputs {
    vectors: Family<mxx_dsl::Mat>,
    plaintexts: Family<mxx_dsl::Mat>,
}

/// Setup-fixed public inputs for native PRF preprocessing.
///
/// The selector family is supplied separately to [`SparseLwrPrfProgram::compile_public`];
/// this value contains only the setup public-key matrix family `A[S]`.  There
/// are no label vectors or plaintexts in the public preprocessing boundary.
#[derive(Clone)]
pub struct NativePrfPublicInputs {
    public_keys: Family<mxx_dsl::Mat>,
    output_count: usize,
    layout_id: Option<PbcLayoutId>,
}

impl NativePrfPublicInputs {
    pub fn new(
        public_keys: Family<mxx_dsl::Mat>,
        output_count: usize,
    ) -> Result<Self, ExponentLutError> {
        let count = public_keys
            .count()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize());
        if output_count == 0 || count == Some(0) {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let count = count.ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        if count % output_count != 0 {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(Self { public_keys, output_count, layout_id: None })
    }

    pub fn new_for_layout(
        public_keys: Family<mxx_dsl::Mat>,
        output_count: usize,
        layout_id: PbcLayoutId,
    ) -> Result<Self, ExponentLutError> {
        let mut inputs = Self::new(public_keys, output_count)?;
        inputs.layout_id = Some(layout_id);
        Ok(inputs)
    }

    pub fn count(&self) -> &mxx_ir_core::IntExpr {
        self.public_keys.count()
    }

    pub fn layout_id(&self) -> Option<PbcLayoutId> {
        self.layout_id
    }

    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    fn active_count(&self) -> Result<usize, ExponentLutError> {
        self.public_keys
            .count()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .and_then(|count| count.checked_div(self.output_count))
            .filter(|count| *count != 0)
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)
    }

    pub(crate) fn public_family(
        &self,
    ) -> Result<crate::public_key::PublicKeyWireFamily, ExponentLutError> {
        let public_keys = self.public_keys.clone();
        let secret_dimension = public_keys
            .element_type()
            .rows
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let public_key_columns = public_keys
            .element_type()
            .columns
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let ring = mxx_dsl::Ring::new(
            public_keys.element_type().modulus.clone(),
            public_keys.element_type().ring_dimension.clone(),
        );
        crate::public_key::PublicKeyWireFamily::from_matrices(
            public_keys,
            &ring,
            secret_dimension,
            public_key_columns,
        )
    }
}

struct PublicProgramBindings {
    inputs: BTreeMap<ProgramInputId, BggPublicKeyWire>,
    selectors: BTreeMap<RhsFamilyId, crate::public_key::PublicRhsFamily>,
    values: BTreeMap<WireFamilyId, crate::public_key::PublicKeyWireFamily>,
    ranges: ProgramFamilyBindings,
}

/// Label-major online inputs for native PRF encoding.
///
/// `vectors` and `plaintexts` are flattened `[label][active-slot]` families.
/// The compiler gathers one active-slot row for each label and returns one
/// terminal output wire per label, without materializing PBC padding.
#[derive(Clone)]
pub struct NativePrfOnlineBatchInputs {
    vectors: Family<mxx_dsl::Mat>,
    plaintexts: Family<mxx_dsl::Mat>,
    label_count: usize,
    layout_id: Option<PbcLayoutId>,
}

/// Terminal native PRF batch output.
///
/// The native accumulator carries only online vectors and the label count.
#[derive(Clone)]
pub struct NativePrfEncodingBatchOutputs {
    vectors: Family<Mat>,
    label_count: usize,
}

impl NativePrfEncodingBatchOutputs {
    pub(crate) fn new(vectors: Family<Mat>, label_count: usize) -> Result<Self, ExponentLutError> {
        if label_count == 0 || *vectors.count() != mxx_ir_core::IntExpr::constant(label_count) {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let vector_type = vectors.element_type();
        if vector_type.rows != mxx_ir_core::IntExpr::constant(1) {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(Self { vectors, label_count })
    }

    pub fn label_count(&self) -> usize {
        self.label_count
    }

    pub fn vectors(&self) -> &Family<Mat> {
        &self.vectors
    }
}

/// Explicit cache artifacts consumed by one value-readout batch.
///
/// Every decomposition is supplied by preprocessing (or imported through
/// [`crate::noise_refresh::artifacts::ReadoutCacheFamily::member`]). The online path
/// only clones these typed graph handles while constructing each label branch;
/// it never calls decomposition or public-key compilation as a fallback.
#[derive(Clone)]
#[cfg(test)]
pub struct NativePrfReadoutArtifacts {
    target_decomposition: Preimage,
    target_schema: ReadoutPreimageSchema,
    helpers: Vec<NativePrfReadoutHelper>,
    normalization_decomposition: Preimage,
    normalization_schema: ReadoutPreimageSchema,
}

/// Production readout cache.  The non-identity helper witnesses stay as two
/// aligned structural families, so setup and online lowering never enumerate
/// or statically select helper members on the host.
#[derive(Clone)]
pub(crate) struct NativePrfReadoutFamilyArtifacts {
    target_decompositions: Family<Preimage>,
    target_schema: ReadoutPreimageSchema,
    helper_sigmas: Vec<usize>,
    helper_switch_decompositions: Family<Preimage>,
    helper_public_decompositions: Family<Preimage>,
    normalization_decomposition: Preimage,
    normalization_schema: ReadoutPreimageSchema,
    output_count: usize,
}

impl NativePrfReadoutFamilyArtifacts {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        target_decompositions: Family<Preimage>,
        target_schema: ReadoutPreimageSchema,
        helper_sigmas: Vec<usize>,
        helper_switch_decompositions: Family<Preimage>,
        helper_public_decompositions: Family<Preimage>,
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
        output_count: usize,
    ) -> Result<Self, ValueReadoutError> {
        let count = mxx_ir_core::IntExpr::constant(helper_sigmas.len());
        let public_count = output_count
            .checked_mul(helper_sigmas.len())
            .ok_or(ValueReadoutError::InvalidTraceSet)?;
        if output_count == 0 ||
            helper_sigmas.is_empty() ||
            *target_decompositions.count() != IntExpr::constant(output_count) ||
            *helper_switch_decompositions.count() != count ||
            *helper_public_decompositions.count() != IntExpr::constant(public_count)
        {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        Ok(Self {
            target_decompositions,
            target_schema,
            helper_sigmas,
            helper_switch_decompositions,
            helper_public_decompositions,
            normalization_decomposition,
            normalization_schema,
            output_count,
        })
    }

    pub(crate) fn target_decompositions(&self) -> &Family<Preimage> {
        &self.target_decompositions
    }
    pub(crate) fn helper_sigmas(&self) -> &[usize] {
        &self.helper_sigmas
    }
    pub(crate) fn helper_switch_decompositions(&self) -> &Family<Preimage> {
        &self.helper_switch_decompositions
    }
    pub(crate) fn helper_public_decompositions(&self) -> &Family<Preimage> {
        &self.helper_public_decompositions
    }
    pub(crate) fn normalization_decomposition(&self) -> &Preimage {
        &self.normalization_decomposition
    }

    pub(crate) fn project_public_batch(
        &self,
        readout: &ValueReadout,
        compiler: &BggPublicKeyCompiler,
        inputs: Family<Mat>,
        public_masks: Family<Mat>,
    ) -> Result<Family<Mat>, ValueReadoutError> {
        readout.project_public_batch_with_preprocessed_family_artifacts(
            compiler,
            inputs,
            self.output_count,
            self.target_decompositions.clone(),
            self.target_schema.clone(),
            &self.helper_sigmas,
            self.helper_public_decompositions.clone(),
            public_masks,
            self.normalization_decomposition.clone(),
            self.normalization_schema.clone(),
        )
    }
}

/// Cache-only description of one readout branch.  The s-dependent mask
/// encoding is deliberately absent and is supplied by the online caller.
#[derive(Clone)]
#[cfg(test)]
pub(crate) struct NativePrfReadoutHelper {
    layout: ExpLayout,
    sigma: usize,
    gadget_base: BigInt,
    digit_count: usize,
    max_coefficient_bound: BigInt,
    switch_decomposition: Preimage,
    public_decomposition: Preimage,
}

#[cfg(test)]
impl NativePrfReadoutHelper {
    pub(crate) fn new(
        layout: ExpLayout,
        sigma: usize,
        gadget_base: BigInt,
        digit_count: usize,
        max_coefficient_bound: BigInt,
        switch_decomposition: Preimage,
        public_decomposition: Preimage,
    ) -> Self {
        Self {
            layout,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
            switch_decomposition,
            public_decomposition,
        }
    }

    fn with_online_mask(
        &self,
        mask: BggEncodingWire,
    ) -> Result<ValueReadoutHelper, ValueReadoutError> {
        ValueReadoutHelper::new(
            self.layout,
            self.sigma,
            self.gadget_base.clone(),
            self.digit_count,
            self.max_coefficient_bound.clone(),
            self.switch_decomposition.clone(),
            self.public_decomposition.clone(),
            mask,
        )
    }

    pub(crate) fn sigma(&self) -> usize {
        self.sigma
    }
}

#[cfg(test)]
impl NativePrfReadoutArtifacts {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        target_decomposition: Preimage,
        target_schema: ReadoutPreimageSchema,
        helpers: Vec<NativePrfReadoutHelper>,
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Self {
        Self {
            target_decomposition,
            target_schema,
            helpers,
            normalization_decomposition,
            normalization_schema,
        }
    }

    /// Imports one setup-produced readout artifact set without loading cache
    /// payloads eagerly. Each family declaration is a distinct association;
    /// `member` creates the lazy typed artifact handle used by the online graph.
    pub fn import_from_cache(
        layout: ExpLayout,
        ring: &Ring,
        manifest: &Manifest,
        target_family: &ReadoutCacheFamily,
        helpers: &[(usize, ReadoutCacheFamily, ReadoutCacheFamily)],
        normalization_family: &ReadoutCacheFamily,
        target_schema: ReadoutPreimageSchema,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<Self, ValueReadoutError> {
        let target_decomposition = target_family.member(ring, manifest, 0)?;
        let mut imported_helpers = Vec::with_capacity(helpers.len());
        for (sigma, switch_family, public_family) in helpers {
            let switch_decomposition = switch_family.member(ring, manifest, 0)?;
            let public_decomposition = public_family.member(ring, manifest, 0)?;
            imported_helpers.push(NativePrfReadoutHelper::new(
                layout,
                *sigma,
                target_schema.gadget_base.clone(),
                target_schema.digit_count,
                target_schema.max_coefficient_bound.clone(),
                switch_decomposition,
                public_decomposition,
            ));
        }
        let normalization_decomposition = normalization_family.member(ring, manifest, 0)?;
        Ok(Self::new(
            target_decomposition.clone(),
            target_schema,
            imported_helpers,
            normalization_decomposition.clone(),
            normalization_schema,
        ))
    }
}

/// Public preprocessing declaration for a native PRF readout.
///
/// `accumulator` is the single public-key PRF result.  `target` and
/// `normalization` are exact graph declarations from the same `ValueReadout`;
/// callers use them to request the corresponding typed cache preimages.
#[derive(Clone)]
pub struct NativePrfPublicReadoutPlan {
    accumulator: BggPublicKeyWire,
    target: Mat,
    normalization: NormalizationTargetDescriptor,
}

impl NativePrfPublicReadoutPlan {
    pub fn accumulator(&self) -> &BggPublicKeyWire {
        &self.accumulator
    }

    pub fn target(&self) -> &Mat {
        &self.target
    }

    pub fn normalization(&self) -> &NormalizationTargetDescriptor {
        &self.normalization
    }
}

impl NativePrfOnlineBatchInputs {
    pub(crate) fn new(
        vectors: Family<mxx_dsl::Mat>,
        plaintexts: Family<mxx_dsl::Mat>,
        label_count: usize,
    ) -> Result<Self, ExponentLutError> {
        if label_count == 0 || vectors.count() != plaintexts.count() {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let vector_count = vectors
            .count()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        if vector_count % label_count != 0 {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(Self { vectors, plaintexts, label_count, layout_id: None })
    }

    pub(crate) fn new_for_layout(
        vectors: Family<mxx_dsl::Mat>,
        plaintexts: Family<mxx_dsl::Mat>,
        label_count: usize,
        layout_id: PbcLayoutId,
    ) -> Result<Self, ExponentLutError> {
        let mut inputs = Self::new(vectors, plaintexts, label_count)?;
        inputs.layout_id = Some(layout_id);
        Ok(inputs)
    }

    /// Constructs online inputs from the public LWR exponents `a_j[i]`.
    /// Plaintext members are created here as the canonical `mono(a_j[i])`;
    /// callers cannot substitute arbitrary plaintext matrices on the
    /// production online boundary.
    pub fn from_public_exponents(
        ring: &Ring,
        exponent_layout: ExpLayout,
        vectors: Family<mxx_dsl::Mat>,
        exponents: &[BigInt],
        label_count: usize,
        layout_id: Option<PbcLayoutId>,
    ) -> Result<Self, ExponentLutError> {
        if exponents.is_empty() {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let ring_dimension = ring
            .matrix_type((1, 1))
            .ring_dimension
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?
            .to_usize()
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        exponent_layout
            .validate(ring_dimension)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let modulus = BigInt::from(exponent_layout.effective_modulus(ring_dimension));
        if exponents.iter().any(|exponent| exponent.is_negative() || exponent >= &modulus) {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let plaintexts = Family::pack(
            exponents
                .iter()
                .map(|exponent| exponent_layout.mono(ring, exponent))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
        )
        .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let mut inputs = Self::new(vectors, plaintexts, label_count)?;
        inputs.layout_id = layout_id;
        Ok(inputs)
    }

    pub fn label_count(&self) -> usize {
        self.label_count
    }

    pub fn vectors(&self) -> &Family<mxx_dsl::Mat> {
        &self.vectors
    }

    pub fn plaintexts(&self) -> &Family<mxx_dsl::Mat> {
        &self.plaintexts
    }

    pub fn layout_id(&self) -> Option<PbcLayoutId> {
        self.layout_id
    }

    fn label(&self, label: mxx_dsl::LoopIndex) -> Result<NativePrfLabelInputs, ExponentLutError> {
        let vector_count = self.vectors.count().clone();
        let slot_count = mxx_ir_core::IntExpr::Div(
            Box::new(vector_count),
            Box::new(mxx_ir_core::IntExpr::constant(self.label_count)),
        )
        .canonicalize();
        let base = label.as_int().mul(mxx_dsl::Int::evaluate(slot_count.clone()));
        let indices = Parallel::range(slot_count)
            .map_values(|slot| base.clone().add(slot.as_int()))
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        NativePrfLabelInputs::new(
            self.vectors
                .clone()
                .parallel_gather(indices.clone())
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
            self.plaintexts
                .clone()
                .parallel_gather(indices)
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
        )
        .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)
    }
}

impl NativePrfLabelInputs {
    /// Constructs the transport form when `Family<BggEncodingWire>` is not a
    /// DSL family element. The three families must remain label-major and
    /// have identical cardinality; plaintexts are the sole clear attribute
    /// channel.
    pub(crate) fn new(
        vectors: Family<mxx_dsl::Mat>,
        plaintexts: Family<mxx_dsl::Mat>,
    ) -> Result<Self, ExponentLutError> {
        if vectors.count() != plaintexts.count() {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(Self { vectors, plaintexts })
    }

    pub(crate) fn count(&self) -> &mxx_ir_core::IntExpr {
        self.vectors.count()
    }

    pub(crate) fn public_family(
        &self,
    ) -> Result<crate::encoding::BggEncodingFamily, ExponentLutError> {
        let vectors = self.vectors.clone();
        let plaintexts = self.plaintexts.clone();
        let public_key_columns = vectors
            .element_type()
            .columns
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let ring = mxx_dsl::Ring::new(
            self.vectors.element_type().modulus.clone(),
            self.vectors.element_type().ring_dimension.clone(),
        );
        crate::encoding::BggEncodingFamily::from_components(
            vectors,
            plaintexts,
            &ring,
            public_key_columns,
        )
        .map_err(ExponentLutError::from)
    }
}

/// A native accumulator plus its value-form readout.
///
/// `initial_input` is the accumulator state initialized by the evaluator to
/// `mono_m(0)=1`; it is represented as a typed program input rather than a
/// label-dependent declaration. Each entry in `selector_families` and
/// `wire_families` is consumed by exactly one structural `select`.
#[derive(Clone)]
pub struct SparseLwrPrfProgram {
    pub accumulation: ExponentLutProgram,
    pub readout: ValueReadout,
    initial_input: ProgramInputId,
    output: WireId,
    selector_families: Vec<RhsFamilyId>,
    wire_families: Vec<WireFamilyId>,
    profile: SparseLwrPrfProfile,
    bucket_width: usize,
    block_count: usize,
}

impl std::fmt::Debug for SparseLwrPrfProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SparseLwrPrfProgram")
            .field("id", &self.id())
            .field("profile", &self.profile)
            .field("bucket_width", &self.bucket_width)
            .field("block_count", &self.block_count)
            .field("readout", &self.readout.identity())
            .finish()
    }
}

impl SparseLwrPrfProgram {
    pub(crate) fn execute_value_readout_batch_with_mask_family(
        &self,
        compiler: &mxx_bgg::BggEncodingCompiler,
        inputs: &NativePrfEncodingBatchOutputs,
        artifacts: &NativePrfReadoutFamilyArtifacts,
        online_masks: Family<Mat>,
    ) -> Result<NativePrfEncodingBatchOutputs, ValueReadoutError> {
        let vectors = with_benchmark_role(BenchmarkRole::EncodingReadout, || {
            self.readout.execute_batch_with_preprocessed_family_artifacts(
                compiler,
                inputs.vectors.clone(),
                inputs.label_count,
                artifacts.target_decompositions.clone(),
                artifacts.target_schema.clone(),
                &artifacts.helper_sigmas,
                artifacts.helper_switch_decompositions.clone(),
                artifacts.helper_public_decompositions.clone(),
                online_masks,
                artifacts.normalization_decomposition.clone(),
                artifacts.normalization_schema.clone(),
            )
        })?;
        NativePrfEncodingBatchOutputs::new(vectors, inputs.label_count)
            .map_err(ValueReadoutError::Graph)
    }

    /// Builds one native accumulator select per PBC block.
    pub fn new(
        profile: SparseLwrPrfProfile,
        bucket_width: usize,
        block_count: usize,
    ) -> Result<Self, ExponentLutError> {
        profile.validate()?;
        if bucket_width == 0 || block_count == 0 {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let readout = profile.canonical_readout()?;
        let ring_dimension = profile.ring_dimension()?;
        readout
            .layout
            .validate(ring_dimension)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;

        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(profile.layout, ring_dimension).map_err(ExponentLutError::from)?;
        let initial_input = builder
            .input(
                profile.native_width(),
                ProgramWireType {
                    stride_log2: profile.layout.stride_log2,
                    knowledge: AttributeKnowledge::Private,
                },
            )
            .map_err(ExponentLutError::from)?;
        let mut state = builder.input_wire(initial_input).map_err(ExponentLutError::from)?;
        let mut selector_families = Vec::with_capacity(block_count);
        let mut wire_families = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let selectors =
                builder.rhs_family(profile.native_width()).map_err(ExponentLutError::from)?;
            let values =
                builder.wire_family(profile.native_width()).map_err(ExponentLutError::from)?;
            state = builder
                .select(state, selectors, values, bucket_width)
                .map_err(ExponentLutError::from)?;
            selector_families.push(selectors);
            wire_families.push(values);
        }
        builder.output(state).map_err(ExponentLutError::from)?;
        let accumulation = builder.build().map_err(ExponentLutError::from)?;
        if accumulation
            .steps()
            .iter()
            .any(|step| matches!(step, ProgramStep::Gate(Gate::Lut { .. })))
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(Self {
            accumulation,
            readout,
            initial_input,
            output: state,
            selector_families,
            wire_families,
            profile,
            bucket_width,
            block_count,
        })
    }

    pub const fn profile(&self) -> SparseLwrPrfProfile {
        self.profile
    }

    pub const fn initial_input(&self) -> ProgramInputId {
        self.initial_input
    }

    /// Returns the accumulator's fixed initial native attribute `mono_m(0)`.
    pub fn initial_attribute(
        &self,
        ring: &mxx_dsl::Ring,
    ) -> Result<mxx_dsl::Mat, ExponentLutError> {
        self.profile
            .layout
            .mono(ring, &num_bigint::BigInt::from(0))
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)
    }

    pub const fn output(&self) -> WireId {
        self.output
    }

    pub const fn bucket_width(&self) -> usize {
        self.bucket_width
    }

    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    pub fn selector_families(&self) -> &[RhsFamilyId] {
        &self.selector_families
    }

    pub fn wire_families(&self) -> &[WireFamilyId] {
        &self.wire_families
    }

    pub const fn id(&self) -> ExponentLutProgramId {
        self.accumulation.id()
    }

    /// The native accumulator has no reduction or terminal LUT stage.
    pub fn has_only_native_accumulation(&self) -> bool {
        !self
            .accumulation
            .steps()
            .iter()
            .any(|step| matches!(step, ProgramStep::Gate(Gate::Lut { .. })))
    }

    /// Cache-required online accumulator lowering. Every multiplication is
    /// supplied its typed `K_rhs` witness from public preprocessing.
    pub(crate) fn compile_encoding_with_preprocessed_artifacts(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        initial: BggEncodingWire,
        selectors: &EncodingRhsFamily,
        labels: &NativePrfLabelInputs,
        layout: &PbcPublicLayout,
        cache: &ProgramCacheBindings,
    ) -> Result<BggEncodingWire, ExponentLutError> {
        self.compile_encoding_inner(
            compiler,
            initial,
            selectors,
            labels,
            layout,
            Some(cache),
            BenchmarkRole::EncodingPrfAccumulation,
        )
    }

    fn compile_encoding_inner(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        initial: BggEncodingWire,
        selectors: &EncodingRhsFamily,
        labels: &NativePrfLabelInputs,
        layout: &PbcPublicLayout,
        cache: Option<&ProgramCacheBindings>,
        benchmark_role: BenchmarkRole,
    ) -> Result<BggEncodingWire, ExponentLutError> {
        validate_profile_program(self.profile, self)?;
        let active = PbcActiveCellIndex::build(layout)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        if self.block_count != layout.parameters.bucket_count ||
            self.bucket_width != layout.bucket_width
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        if selectors.layout_id() != Some(layout.layout_id) {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let expected = active.len();
        let concrete_count = |count: &mxx_ir_core::IntExpr| {
            count.evaluate(&mxx_ir_core::ParamEnv::default()).ok().and_then(|v| v.to_usize())
        };
        if concrete_count(labels.count()) != Some(expected) ||
            concrete_count(selectors.count()) != Some(expected)
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let mut inputs = BTreeMap::new();
        inputs.insert(self.initial_input, initial);
        let mut selector_bindings = BTreeMap::new();
        let mut value_bindings = BTreeMap::new();
        let mut ranges = ProgramFamilyBindings::new();
        let public_family = labels.public_family()?;
        for block in 0..self.block_count {
            let (start, active_width) = active
                .bucket_iter(block)
                .and_then(|mut entries| {
                    entries.next().map(|(_, _, offset)| {
                        (offset, active.bucket_active_width(block).unwrap_or_default())
                    })
                })
                .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
            let selector = selectors.clone();
            let selector_id = self.selector_families[block];
            let value_id = self.wire_families[block];
            selector_bindings.insert(selector_id, selector);
            value_bindings.insert(value_id, public_family.clone());
            let range = FamilyRange::bounded(start, active_width, self.bucket_width)
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            ranges.bind(crate::program::ProgramFamilyId::from_index(block), range);
        }
        let outputs = with_benchmark_role(benchmark_role, || match cache {
            Some(cache) => compiler.compile_program_with_ranges_and_cache(
                &self.accumulation,
                &inputs,
                &BTreeMap::new(),
                &selector_bindings,
                &value_bindings,
                &ranges,
                &BTreeMap::new(),
                cache,
            ),
            None => compiler.compile_program_with_ranges(
                &self.accumulation,
                &inputs,
                &BTreeMap::new(),
                &selector_bindings,
                &value_bindings,
                &ranges,
                &BTreeMap::new(),
            ),
        })?;
        outputs.get(&self.output).cloned().ok_or(ExponentLutError::InvalidSparseLwrBlock)
    }

    /// Batch form of cache-required online accumulation. The outer label
    /// family is structural; each member consumes the same imported public
    /// cache declarations and no member emits `GadgetDecompose`.
    pub(crate) fn compile_encoding_batch_with_preprocessed_artifacts(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        initial: BggEncodingWire,
        selectors: &EncodingRhsFamily,
        inputs: &NativePrfOnlineBatchInputs,
        layout: &PbcPublicLayout,
        cache: &ProgramCacheBatchBindings,
    ) -> Result<NativePrfEncodingBatchOutputs, ExponentLutError> {
        if inputs.layout_id() != Some(layout.layout_id) ||
            selectors.layout_id() != Some(layout.layout_id) ||
            inputs.label_count != cache.output_count()
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let construction_error = std::cell::RefCell::new(None);
        let vectors_result = with_benchmark_role(BenchmarkRole::EncodingPrfAccumulation, || {
            Parallel::range(inputs.label_count).try_map_values(
                (inputs.vectors.clone(), inputs.plaintexts.clone(), selectors.gsw().clone()),
                |label, (vectors, plaintexts, gsw)| {
                    let inputs = NativePrfOnlineBatchInputs {
                        vectors,
                        plaintexts,
                        label_count: inputs.label_count,
                        layout_id: inputs.layout_id,
                    };
                    let selectors = EncodingRhsFamily::new_for_layout(gsw, layout.layout_id)
                        .map_err(|_| mxx_dsl::DslError::Schema)?;
                    let output_cache = cache.output(label.clone()).map_err(|error| {
                        construction_error.replace(Some(error));
                        mxx_dsl::DslError::Schema
                    })?;
                    let labels = inputs.label(label).map_err(|error| {
                        construction_error.replace(Some(ExponentLutError::OnlinePrfStage {
                            stage: "select label family",
                            detail: error.to_string(),
                        }));
                        mxx_dsl::DslError::Schema
                    })?;
                    let output = self
                        .compile_encoding_with_preprocessed_artifacts(
                            compiler,
                            initial.clone(),
                            &selectors,
                            &labels,
                            layout,
                            &output_cache,
                        )
                        .map_err(|error| {
                            construction_error.replace(Some(error));
                            mxx_dsl::DslError::Schema
                        })?;
                    Ok(output.vector)
                },
            )
        });
        let vectors = match vectors_result {
            Ok(vectors) => vectors,
            Err(error) => {
                return Err(construction_error.into_inner().unwrap_or(
                    ExponentLutError::OnlinePrfStage {
                        stage: "construct label-major family",
                        detail: error.to_string(),
                    },
                ));
            }
        };
        NativePrfEncodingBatchOutputs::new(vectors, inputs.label_count)
    }

    /// Online readout variant which combines setup-fixed decomposition
    /// witnesses with freshly sampled online mask encodings. Switch/GSW and
    /// public decomposition artifacts remain setup-owned; no online helper
    /// switch or GSW ciphertext is sampled here.
    #[cfg(test)]
    pub fn execute_value_readout_batch_with_masks(
        &self,
        compiler: &mxx_bgg::BggEncodingCompiler,
        inputs: &NativePrfEncodingBatchOutputs,
        artifacts: &NativePrfReadoutArtifacts,
        online_masks: &[(usize, BggEncodingWire)],
    ) -> Result<NativePrfEncodingBatchOutputs, ValueReadoutError> {
        self.execute_value_readout_batch_inner_with_online_masks(
            compiler,
            inputs,
            artifacts,
            online_masks,
        )
    }

    #[cfg(test)]
    fn execute_value_readout_batch_inner_with_online_masks(
        &self,
        compiler: &mxx_bgg::BggEncodingCompiler,
        inputs: &NativePrfEncodingBatchOutputs,
        artifacts: &NativePrfReadoutArtifacts,
        online_masks: &[(usize, BggEncodingWire)],
    ) -> Result<NativePrfEncodingBatchOutputs, ValueReadoutError> {
        if online_masks.len() != artifacts.helpers.len() {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        let helpers = artifacts
            .helpers
            .iter()
            .zip(online_masks)
            .map(|(cached, (sigma, mask))| {
                if cached.sigma() != *sigma {
                    return Err(ValueReadoutError::InvalidTraceSet);
                }
                cached.with_online_mask(mask.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.execute_value_readout_batch_core(compiler, inputs, artifacts, &helpers)
    }

    #[cfg(test)]
    fn execute_value_readout_batch_core(
        &self,
        compiler: &mxx_bgg::BggEncodingCompiler,
        inputs: &NativePrfEncodingBatchOutputs,
        artifacts: &NativePrfReadoutArtifacts,
        helpers: &[ValueReadoutHelper],
    ) -> Result<NativePrfEncodingBatchOutputs, ValueReadoutError> {
        let vectors = with_benchmark_role(BenchmarkRole::EncodingReadout, || {
            Family::try_parallel_zip_many_values(vec![inputs.vectors.clone()], |_, mut values| {
                let input = BggEncodingWire {
                    vector: values.pop().ok_or(mxx_dsl::DslError::Schema)?,
                    plaintext: None,
                };
                let output = self
                    .readout
                    .execute_with_preprocessed_artifacts(
                        compiler,
                        &input,
                        artifacts.target_decomposition.clone(),
                        artifacts.target_schema.clone(),
                        &helpers,
                        artifacts.normalization_decomposition.clone(),
                        artifacts.normalization_schema.clone(),
                    )
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                Ok(output.vector)
            })
            .map_err(|error| {
                ValueReadoutError::Graph(ExponentLutError::OnlinePrfStage {
                    stage: "compile readout batch",
                    detail: error.to_string(),
                })
            })
        })?;
        NativePrfEncodingBatchOutputs::new(vectors, inputs.label_count)
            .map_err(ValueReadoutError::Graph)
    }

    /// Executes the native accumulator against public BGG matrices. This is
    /// the preprocessing-side counterpart of [`Self::compile_encoding`].
    pub fn compile_public(
        &self,
        compiler: &crate::public_key::ExponentLutPublicKeyCompiler,
        initial: BggPublicKeyWire,
        selectors: &crate::public_key::PublicRhsFamily,
        public: &NativePrfPublicInputs,
        layout: &PbcPublicLayout,
    ) -> Result<BggPublicKeyWire, ExponentLutError> {
        let bindings = self.public_program_bindings(initial, selectors, public, layout)?;
        let outputs = with_benchmark_role(BenchmarkRole::PublicPrfAccumulation, || {
            compiler.compile_program_with_ranges(
                &self.accumulation,
                &bindings.inputs,
                &BTreeMap::new(),
                &bindings.selectors,
                &bindings.values,
                &bindings.ranges,
                &BTreeMap::new(),
            )
        })?;
        outputs.get(&self.output).cloned().ok_or(ExponentLutError::InvalidSparseLwrBlock)
    }

    /// Compiles one setup-fixed public-key PRF per logical output. The outer
    /// output axis remains structural and every cache family is label-major.
    pub(crate) fn compile_public_batch_with_preprocessing(
        &self,
        compiler: &crate::public_key::ExponentLutPublicKeyCompiler,
        initial: BggPublicKeyWire,
        selectors: &crate::public_key::PublicRhsFamily,
        public: &NativePrfPublicInputs,
        layout: &PbcPublicLayout,
    ) -> Result<crate::public_key::PublicProgramBatchPreprocessing, ExponentLutError> {
        validate_profile_program(self.profile, self)?;
        let active = PbcActiveCellIndex::build(layout)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let active_count = active.len();
        if self.block_count != layout.parameters.bucket_count ||
            self.bucket_width != layout.bucket_width ||
            public.layout_id() != Some(layout.layout_id) ||
            selectors.layout_id() != Some(layout.layout_id) ||
            public.active_count()? != active_count
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let ranges = (0..self.block_count)
            .map(|block| {
                active
                    .bucket_iter(block)
                    .and_then(|mut entries| {
                        entries.next().map(|(_, _, offset)| {
                            (offset, active.bucket_active_width(block).unwrap_or_default())
                        })
                    })
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)
            })
            .collect::<Result<Vec<_>, _>>()?;
        with_benchmark_role(BenchmarkRole::PublicPrfAccumulation, || {
            compiler.compile_select_batch_with_preprocessing(
                initial,
                selectors,
                public.public_keys.clone(),
                public.output_count,
                active_count,
                &ranges,
            )
        })
    }

    fn public_program_bindings(
        &self,
        initial: BggPublicKeyWire,
        selectors: &crate::public_key::PublicRhsFamily,
        public: &NativePrfPublicInputs,
        layout: &PbcPublicLayout,
    ) -> Result<PublicProgramBindings, ExponentLutError> {
        validate_profile_program(self.profile, self)?;
        let active = PbcActiveCellIndex::build(layout)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        if self.block_count != layout.parameters.bucket_count ||
            self.bucket_width != layout.bucket_width
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        if public.layout_id() != Some(layout.layout_id) ||
            selectors.layout_id() != Some(layout.layout_id)
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let expected = active.len();
        let concrete_count = |count: &mxx_ir_core::IntExpr| {
            count.evaluate(&mxx_ir_core::ParamEnv::default()).ok().and_then(|v| v.to_usize())
        };
        if concrete_count(public.count()) != Some(expected) ||
            concrete_count(selectors.count()) != Some(expected)
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let mut inputs = BTreeMap::new();
        inputs.insert(
            self.initial_input,
            BggPublicKeyWire { matrix: initial.matrix, reveal_plaintext: false },
        );
        let mut selector_bindings = BTreeMap::new();
        let mut value_bindings = BTreeMap::new();
        let mut ranges = ProgramFamilyBindings::new();
        let public_family = public.public_family()?;
        for block in 0..self.block_count {
            let (start, active_width) = active
                .bucket_iter(block)
                .and_then(|mut entries| {
                    entries.next().map(|(_, _, offset)| {
                        (offset, active.bucket_active_width(block).unwrap_or_default())
                    })
                })
                .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
            let selector_id = self.selector_families[block];
            let value_id = self.wire_families[block];
            selector_bindings.insert(selector_id, selectors.clone());
            value_bindings.insert(value_id, public_family.clone());
            let range = FamilyRange::bounded(start, active_width, self.bucket_width)
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            ranges.bind(crate::program::ProgramFamilyId::from_index(block), range);
        }
        Ok(PublicProgramBindings {
            inputs,
            selectors: selector_bindings,
            values: value_bindings,
            ranges,
        })
    }

    /// Compiles the setup-fixed public accumulator once and returns the exact
    /// readout target declarations from the same ring/layout.  The returned
    /// target and normalization descriptors are the declarations used to
    /// request the typed preprocessing cache artifacts; no witness is sampled
    /// here and no online fallback is introduced.
    pub fn compile_public_readout_plan(
        &self,
        compiler: &crate::public_key::ExponentLutPublicKeyCompiler,
        initial: BggPublicKeyWire,
        selectors: &crate::public_key::PublicRhsFamily,
        public: &NativePrfPublicInputs,
        layout: &PbcPublicLayout,
        normalization_gadget: Mat,
    ) -> Result<NativePrfPublicReadoutPlan, ValueReadoutError> {
        let accumulator = self
            .compile_public(compiler, initial, selectors, public, layout)
            .map_err(ValueReadoutError::Graph)?;
        let ring = compiler.public_key.ring.clone();
        let target = self.readout.target(&ring)?;
        let normalization =
            self.readout.normalization_target_descriptor(&ring, &normalization_gadget)?;
        Ok(NativePrfPublicReadoutPlan { accumulator, target, normalization })
    }

    /// Produces the complete setup-side readout artifact set.  All
    /// decompositions are ordinary typed `Preimage` graph outputs; online
    /// execution imports these handles and never performs decomposition or
    /// public-key recomputation.
    #[cfg(test)]
    pub(crate) fn produce_readout_artifacts(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        input: &BggEncodingWire,
        helper_set: &LutHelperSet,
        normalization_gadget: Mat,
        max_coefficient_bound: BigInt,
    ) -> Result<NativePrfReadoutArtifacts, ValueReadoutError> {
        with_benchmark_role(BenchmarkRole::PublicReadout, || {
            self.produce_readout_artifacts_inner(
                compiler,
                input,
                helper_set,
                normalization_gadget,
                max_coefficient_bound,
            )
        })
    }

    /// Produces the native readout cache without enumerating helper branches.
    pub(crate) fn produce_public_readout_family_artifacts(
        &self,
        compiler: &crate::public_key::ExponentLutPublicKeyCompiler,
        inputs: Family<Mat>,
        helpers: &NativeReadoutHelperFamilies,
        normalization_gadget: Mat,
    ) -> Result<NativePrfReadoutFamilyArtifacts, ValueReadoutError> {
        let ring = compiler.public_key.ring.clone();
        self.readout.validate(&ring)?;
        let ring_dimension = ring
            .matrix_type((1, 1))
            .ring_dimension
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
        let plan = self.readout.trace_plan(ring_dimension)?;
        let output_count = inputs
            .count()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .filter(|count| *count != 0)
            .ok_or(ValueReadoutError::InvalidTraceSet)?;
        if helpers.sigmas.as_slice() != plan.non_identity_helpers() {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        let gadget_base = compiler
            .public_key
            .base
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let digit_count = compiler
            .public_key
            .digit_count
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
        let actual_bound = &gadget_base / BigInt::from(2u8);
        let schema = ReadoutPreimageSchema::new(gadget_base.clone(), digit_count, actual_bound)?;
        let target = self.readout.target(&ring)?;
        let target_for_outputs = target.clone();
        let target_compiler = compiler.public_key.clone();
        let target_decompositions = inputs
            .clone()
            .parallel_map_values(move |_, input| {
                target_compiler.large_scalar_decomposition(
                    &BggPublicKeyWire { matrix: input, reveal_plaintext: false },
                    &target_for_outputs,
                )
            })
            .map_err(|_| ValueReadoutError::Graph(ExponentLutError::InvalidLut))?;
        let transformed = (inputs, target_decompositions.clone())
            .parallel_zip_tuple(|_, (input, target)| target.mul_small_rhs(input))
            .map_err(|_| ValueReadoutError::Graph(ExponentLutError::InvalidLut))?;
        let switch_decompositions = helpers
            .switch_ciphertexts
            .clone()
            .parallel_map_values(|_, switch| {
                switch.decompose(compiler.public_key.base.clone(), digit_count)
            })
            .map_err(|_| ValueReadoutError::Graph(ExponentLutError::InvalidLut))?;
        let helper_count = helpers.sigmas.len();
        let flat_count =
            output_count.checked_mul(helper_count).ok_or(ValueReadoutError::InvalidTraceSet)?;
        let label_indices = Parallel::range(flat_count)
            .map_values(|index| index.as_int().div(mxx_dsl::Int::constant(helper_count)))
            .map_err(|_| ValueReadoutError::Graph(ExponentLutError::InvalidLut))?;
        let flat_transformed = transformed
            .parallel_gather(label_indices)
            .map_err(|_| ValueReadoutError::Graph(ExponentLutError::InvalidLut))?;
        let flat_switches = switch_decompositions
            .clone()
            .parallel_gather(
                Parallel::range(flat_count)
                    .map_values(|index| index.as_int().rem(mxx_dsl::Int::constant(helper_count)))
                    .map_err(|_| ValueReadoutError::Graph(ExponentLutError::InvalidLut))?,
            )
            .map_err(|_| ValueReadoutError::Graph(ExponentLutError::InvalidLut))?;
        let sigma_branches =
            helpers.sigmas.iter().copied().map(IntExpr::constant).collect::<Vec<_>>();
        let public_decompositions = (flat_transformed, flat_switches)
            .parallel_zip_tuple(move |index, (transformed, switch)| {
                let sigma = IntExpr::Select {
                    selector: Box::new(IntExpr::Sub(
                        Box::new(index.expression()),
                        Box::new(IntExpr::Mul(
                            Box::new(IntExpr::Div(
                                Box::new(index.expression()),
                                Box::new(IntExpr::constant(helper_count)),
                            )),
                            Box::new(IntExpr::constant(helper_count)),
                        )),
                    )),
                    branches: sigma_branches.clone(),
                };
                switch
                    .mul_small_rhs(transformed.ring_automorphism(sigma))
                    .decompose(compiler.public_key.base.clone(), digit_count)
            })
            .map_err(|_| ValueReadoutError::Graph(ExponentLutError::InvalidLut))?;
        let normalization_target =
            self.readout.normalization_target(&ring, normalization_gadget)?;
        let normalization_decomposition =
            normalization_target.decompose(compiler.public_key.base.clone(), digit_count);
        NativePrfReadoutFamilyArtifacts::new(
            target_decompositions,
            schema.clone(),
            helpers.sigmas.as_ref().clone(),
            switch_decompositions,
            public_decompositions,
            normalization_decomposition,
            schema,
            output_count,
        )
    }

    #[cfg(test)]
    fn produce_readout_artifacts_inner(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        input: &BggEncodingWire,
        helper_set: &LutHelperSet,
        normalization_gadget: Mat,
        max_coefficient_bound: BigInt,
    ) -> Result<NativePrfReadoutArtifacts, ValueReadoutError> {
        let ring = compiler.ring.clone();
        let ring_dimension = ring
            .matrix_type((1, 1))
            .ring_dimension
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
        let plan = self.readout.trace_plan(ring_dimension)?;
        let helper_sigmas =
            helper_set.as_slice().iter().map(|helper| helper.sigma()).collect::<Vec<_>>();
        if helper_sigmas != plan.automorphisms() {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        let gadget_base = compiler
            .gadget_base
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let digit_count = compiler
            .digit_count
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
        let target = self.readout.target(&ring)?;
        let gadget = ring.gadget(
            input.vector.matrix_type().rows.clone(),
            compiler.gadget_base.clone(),
            compiler.digit_count.clone(),
        );
        let target_decomposition = (target.clone() * gadget)
            .decompose(compiler.gadget_base.clone(), compiler.digit_count.clone());
        let actual_bound = target_decomposition
            .max_coefficient_bound()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        // B_chi bounds helper/anchor Gaussian errors; the cache schema bound
        // is the deterministic decomposition bound floor(beta/2).
        let _ = max_coefficient_bound;
        let target_schema =
            ReadoutPreimageSchema::new(gadget_base.clone(), digit_count, actual_bound.clone())?;
        let mut helpers = Vec::with_capacity(plan.helper_count());
        for sigma in plan.non_identity_helpers() {
            let source_helper = helper_set
                .as_slice()
                .iter()
                .find(|helper| helper.sigma() == *sigma)
                .ok_or(ValueReadoutError::InvalidTraceSet)?;
            let switch_decomposition = source_helper
                .switch()
                .gsw_ciphertext()
                .clone()
                .decompose(gadget_base.clone(), digit_count);
            let public_decomposition = switch_decomposition.clone();
            let helper_bound = public_decomposition
                .max_coefficient_bound()
                .evaluate(&mxx_ir_core::ParamEnv::default())
                .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
            let switch_bound = switch_decomposition
                .max_coefficient_bound()
                .evaluate(&mxx_ir_core::ParamEnv::default())
                .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
            if helper_bound != actual_bound || switch_bound != actual_bound {
                return Err(ValueReadoutError::InvalidPreimageSchema);
            }
            let helper = NativePrfReadoutHelper::new(
                self.readout.layout,
                *sigma,
                gadget_base.clone(),
                digit_count,
                actual_bound.clone(),
                switch_decomposition,
                public_decomposition,
            );
            helpers.push(helper);
        }
        let normalization_target =
            self.readout.normalization_target(&ring, normalization_gadget)?;
        let normalization_schema =
            ReadoutPreimageSchema::new(gadget_base.clone(), digit_count, actual_bound.clone())?;
        let normalization_decomposition =
            normalization_target.decompose(gadget_base.clone(), digit_count);
        let normalization_bound = normalization_decomposition
            .max_coefficient_bound()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        if normalization_bound != actual_bound {
            return Err(ValueReadoutError::InvalidPreimageSchema);
        }
        Ok(NativePrfReadoutArtifacts::new(
            target_decomposition,
            target_schema,
            helpers,
            normalization_decomposition,
            normalization_schema,
        ))
    }
}

/// Validates the profile associated with a caller-provided program.
pub fn validate_profile_program(
    profile: SparseLwrPrfProfile,
    program: &SparseLwrPrfProgram,
) -> Result<(), ExponentLutError> {
    profile.validate()?;
    program.profile.validate()?;
    if program.profile != profile || program.readout.layout != profile.layout {
        return Err(ExponentLutError::InvalidSparseLwrBlock);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{Int, Ring};

    #[test]
    fn profile_derives_q_and_p_for_nonbinary_digit_bases() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        assert_eq!(profile.native_width(), 32);
        assert_eq!(profile.q().unwrap(), 64);
        assert_eq!(profile.p().unwrap(), 16);
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 6 }, 16).unwrap();
        assert_eq!(profile.q().unwrap(), 128);
        assert_eq!(profile.p().unwrap(), 32);
    }

    #[test]
    fn profile_rejects_nondividing_digit_base() {
        assert!(SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 12).is_err());
    }

    #[test]
    fn canonical_sparse_lwr_table_has_full_q_complement_and_uniform_p_values() {
        for (ring_dimension_log2, stride_log2) in
            [(3_u32, 0_u32), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]
        {
            let width_log2 = ring_dimension_log2 - stride_log2;
            let n_prime = 1usize << width_log2;
            let p_prime = n_prime / 2;
            let profile =
                SparseLwrPrfProfile::new(ExpLayout { stride_log2, width_log2 }, p_prime).unwrap();
            let readout = profile.canonical_readout().unwrap();
            let expected = (0..n_prime).map(|k| p_prime * k / n_prime + 1).collect::<Vec<_>>();
            let coefficients = readout.canonical_coefficients().unwrap();
            let expected_coefficients =
                expected.iter().map(|value| num_bigint::BigInt::from(*value)).collect::<Vec<_>>();
            assert_eq!(coefficients, expected_coefficients.as_slice());
            for k in 0..n_prime {
                let complement = 1_i64 - expected[k] as i64;
                assert_eq!(expected[k] as i64 + complement, 1);
            }
            for value in 1..=p_prime {
                assert_eq!(expected.iter().filter(|entry| **entry == value).count(), 2);
            }
            assert_eq!(profile.q().unwrap(), 2 * n_prime);
        }
    }

    #[test]
    fn profile_deserialization_rechecks_derived_moduli() {
        let encoded = serde_json::json!({
            "layout": { "stride_log2": 1, "width_log2": 5 },
            "p_prime": 12,
        });
        assert!(serde_json::from_value::<SparseLwrPrfProfile>(encoded).is_err());
    }

    #[test]
    fn native_program_has_one_select_per_block_and_no_lut() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let program = SparseLwrPrfProgram::new(profile, 2, 3).unwrap();
        assert!(program.has_only_native_accumulation());
        assert_eq!(program.selector_families().len(), 3);
        assert_eq!(program.wire_families().len(), 3);
        assert_eq!(program.accumulation.steps().len(), 3);
        assert_eq!(program.accumulation.gates().len(), 0);
    }

    #[test]
    fn native_batch_keeps_per_label_vectors_and_monomials_distinct() {
        let ring = Ring::new(97, 8);
        let active_count = 2usize;
        let label_count = 2usize;
        let total = active_count * label_count;
        let vectors = ring.input_family("native-label-vectors", total, (1, 1));
        let monomials = ring.input_family("native-label-monomials", total, (1, 1));
        let inputs = NativePrfOnlineBatchInputs::new(vectors, monomials, label_count)
            .expect("native online inputs");
        assert_ne!(
            inputs.vectors().get(Int::constant(0)).value_handle(),
            inputs.vectors().get(Int::constant(active_count)).value_handle()
        );
        assert_ne!(
            inputs.plaintexts().get(Int::constant(0)).value_handle(),
            inputs.plaintexts().get(Int::constant(active_count)).value_handle()
        );
    }

    #[test]
    fn online_public_exponent_boundary_constructs_canonical_monomials() {
        let ring = Ring::new(97, 8);
        let exponent_layout = ExpLayout { stride_log2: 1, width_log2: 2 };
        let vectors = ring.input_family("online-canonical-vectors", 2, (1, 2));
        let inputs = NativePrfOnlineBatchInputs::from_public_exponents(
            &ring,
            exponent_layout,
            vectors,
            &[BigInt::from(1), BigInt::from(3)],
            1,
            None,
        )
        .unwrap();
        let expected = exponent_layout.mono(&ring, &BigInt::from(3)).unwrap();
        assert_eq!(
            inputs.plaintexts().value_handle().node().arguments()[1].node().kind(),
            expected.value_handle().node().kind()
        );
        assert!(
            NativePrfOnlineBatchInputs::from_public_exponents(
                &ring,
                exponent_layout,
                ring.input_family("online-invalid-vectors", 1, (1, 2)),
                &[BigInt::from(-1)],
                1,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn profile_program_mismatch_is_rejected() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let program = SparseLwrPrfProgram::new(profile, 2, 1).unwrap();
        let other =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 4).unwrap();
        assert!(validate_profile_program(other, &program).is_err());
    }

    #[test]
    fn native_public_accumulator_binds_one_structural_selector_block() {
        let ring = Ring::new(97, 16);
        let layout = ExpLayout { stride_log2: 1, width_log2: 3 };
        let profile = SparseLwrPrfProfile::new(layout, 1).unwrap();
        let pbc = crate::noise_refresh::pbc::PbcParameters::custom(4, 1, 2, 3, 1, None);
        let pbc_layout = crate::noise_refresh::pbc::PbcPublicLayout::build(
            &pbc,
            crate::noise_refresh::pbc::derive_attempt_seed(
                crate::noise_refresh::pbc::PbcRootSeed([3; 32]),
                0,
            ),
            0,
        )
        .unwrap();
        let program = SparseLwrPrfProgram::new(
            profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )
        .unwrap();
        assert_ne!(program.bucket_width(), 4);
        assert_ne!(program.bucket_width(), 2);
        assert_ne!(profile.native_width(), 4);
        let active_count =
            crate::noise_refresh::pbc::PbcActiveCellIndex::build(&pbc_layout).unwrap().len();
        assert!(active_count < pbc_layout.bucket_width * pbc_layout.parameters.bucket_count);

        let public_compiler = crate::public_key::ExponentLutPublicKeyCompiler::with_layout(
            mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 2.into(),
                digit_count: 2.into(),
            },
            layout,
        );
        let public_selector = crate::public_key::PublicRhsFamily::new_for_layout(
            ring.input_family("native-public-selector", active_count, (2, 2)),
            pbc_layout.layout_id,
        )
        .unwrap();
        let public_inputs = NativePrfPublicInputs::new_for_layout(
            ring.input_family("native-public-values", active_count, (2, 2)),
            1,
            pbc_layout.layout_id,
        )
        .unwrap();
        let public_output = program
            .compile_public(
                &public_compiler,
                BggPublicKeyWire {
                    matrix: ring.input("native-public-initial", (2, 2)),
                    reveal_plaintext: false,
                },
                &public_selector,
                &public_inputs,
                &pbc_layout,
            )
            .unwrap();
        assert_eq!(public_output.matrix.matrix_type(), &ring.matrix_type((2, 2)));
        let public_graph = mxx_dsl::DslContext::new("native-public-stage")
            .public_output("public-output", public_output.matrix.clone())
            .unwrap()
            .build()
            .unwrap();
        assert!(
            public_graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| node.benchmark_role() == Some(BenchmarkRole::PublicPrfAccumulation))
        );
        let wrong_layout = crate::noise_refresh::pbc::PbcPublicLayout::build(
            &crate::noise_refresh::pbc::PbcParameters::custom(4, 1, 2, 2, 1, None),
            crate::noise_refresh::pbc::derive_attempt_seed(
                crate::noise_refresh::pbc::PbcRootSeed([3; 32]),
                0,
            ),
            0,
        )
        .unwrap();
        assert!(
            program
                .compile_public(
                    &public_compiler,
                    BggPublicKeyWire {
                        matrix: ring.input("native-public-wrong-initial", (2, 2)),
                        reveal_plaintext: false,
                    },
                    &public_selector,
                    &public_inputs,
                    &wrong_layout,
                )
                .is_err()
        );
    }

    #[test]
    fn native_batch_readout_consumes_supplied_artifacts_for_two_labels() {
        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let batch = NativePrfEncodingBatchOutputs::new(
            ring.input_family("readout-batch-vectors", 2, (1, 4)),
            2,
        )
        .unwrap();
        let helpers = [3usize, 5, 7]
            .into_iter()
            .map(|sigma| {
                NativePrfReadoutHelper::new(
                    layout,
                    sigma,
                    4.into(),
                    2,
                    7.into(),
                    ring.preimage_input(format!("batch-switch-{sigma}"), (4, 4), 7),
                    ring.preimage_input(format!("batch-public-{sigma}"), (4, 4), 7),
                )
            })
            .collect::<Vec<_>>();
        let artifacts = NativePrfReadoutArtifacts::new(
            ring.preimage_input("batch-target", (4, 4), 7),
            ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
            helpers,
            ring.preimage_input("batch-normalization", (4, 4), 7),
            ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
        );
        let compiler = mxx_bgg::BggEncodingCompiler;
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(layout, 1).unwrap(), 1, 1).unwrap();
        let online_masks = [3usize, 5, 7]
            .into_iter()
            .map(|sigma| (sigma, BggEncodingWire { vector: ring.zero((1, 4)), plaintext: None }))
            .collect::<Vec<_>>();
        let batched = program
            .execute_value_readout_batch_with_masks(&compiler, &batch, &artifacts, &online_masks)
            .unwrap();
        assert_eq!(batched.label_count(), 2);
        let built = mxx_dsl::DslContext::new("native-batch-readout")
            .family_output("vectors", batched.vectors().clone())
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::GadgetDecompose { .. }
                ))
                .count(),
            0
        );
        assert!(
            nodes
                .iter()
                .any(|node| { node.benchmark_role() == Some(BenchmarkRole::EncodingReadout) })
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::SequentialLoop(_)
                ))
                .count(),
            0
        );
    }

    #[test]
    fn preprocessing_produces_ordered_readout_artifacts_from_mask_sampler() {
        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let sampler = crate::encoding::ExponentLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            exponent_layout: layout,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let bank = sampler
            .sample_mask_bank(
                ring.input("readout-producer-mask-secret", (1, 2)),
                ring.bytes_input("readout-producer-hash", 32),
                4,
                b"readout-producer-mask-bank".as_slice(),
            )
            .unwrap();
        let table = crate::program::LutTable::unary(4, 2, vec![1, 1, 1, 1]).unwrap();
        let sampled_helpers = sampler
            .sample_helpers_for_lut(
                ring.input("readout-producer-mask-secret", (1, 2)),
                Some(ring.input("readout-producer-payload-secret", (1, 2))),
                ring.bytes_input("readout-producer-helper-hash", 32),
                &table,
                bank.as_ref(),
                b"readout-producer-helpers".as_slice(),
            )
            .unwrap();
        let helper_set = LutHelperSet::new(&table, sampled_helpers).unwrap();
        let compiler = ExponentLutEncodingCompiler::with_layout(
            mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: 2.into(),
            },
            layout,
        );
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(layout, 1).unwrap(), 1, 1).unwrap();
        let input = BggEncodingWire {
            vector: ring.input("readout-producer-vector", (1, 4)),
            plaintext: None,
        };
        let artifacts = program
            .produce_readout_artifacts(
                &compiler,
                &input,
                &helper_set,
                ring.gadget(2, 4, 2),
                2.into(),
            )
            .unwrap();
        assert_eq!(artifacts.helpers.len(), 3);
        assert_eq!(
            artifacts.helpers.iter().map(NativePrfReadoutHelper::sigma).collect::<Vec<_>>(),
            vec![3, 5, 7]
        );
        let built = mxx_dsl::DslContext::new("readout-producer")
            .private_output("target", artifacts.target_decomposition.clone())
            .unwrap()
            .private_output("normalization", artifacts.normalization_decomposition.clone())
            .unwrap()
            .build()
            .unwrap();
        let producer_id = mxx_ir_core::artifact::ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([0x31; 32]),
            execution_nonce: [0x32; 32],
        };
        let validated = built.validate(&mxx_ir_core::ParamEnv::default()).unwrap();
        let exported =
            mxx_ir_core::artifact::export_validated_manifest(producer_id, &validated).unwrap();
        assert!(!exported.artifacts.values().any(|artifact| matches!(
            artifact.artifact_type,
            mxx_ir_core::artifact::ArtifactType::Matrix(_)
        )));
        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert!(nodes.iter().any(|node| matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::GadgetDecompose { .. }
        )));
        assert!(
            !nodes
                .iter()
                .any(|node| matches!(node.kind(), mxx_ir_core::node::NodeKind::SequentialLoop(_)))
        );
    }

    #[test]
    fn imported_readout_families_feed_online_without_decomposition_nodes() {
        use crate::noise_refresh::artifacts::{
            DecompositionCacheArtifact, DecompositionCacheKey, DecompositionOperandRole,
            ReadoutCacheFamily,
        };
        use mxx_ir_core::{
            artifact::{ArtifactConfidentiality, Manifest, ProductionId, SpecHash},
            types::ConcreteMatrixType,
        };
        use std::collections::BTreeMap;

        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let production_id = ProductionId { spec_hash: SpecHash([8; 32]), execution_nonce: [9; 32] };
        let readout_identity =
            SparseLwrPrfProfile::new(layout, 1).unwrap().canonical_readout().unwrap().identity();
        let mut manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production_id.clone(),
            artifacts: BTreeMap::new(),
        };
        let family = |role: DecompositionOperandRole, coordinate: usize| {
            let key = DecompositionCacheKey {
                layout,
                readout_identity,
                setup_identity: [1; 32],
                program_identity: [2; 32],
                gate_or_readout_location: format!("readout/{coordinate}"),
                family_coordinate: vec![coordinate],
                operand_role: role,
            };
            let declaration = DecompositionCacheArtifact::preimage(
                key,
                ConcreteMatrixType { modulus: 97.into(), ring_dimension: 4, rows: 4, columns: 4 },
                2.into(),
                Some(vec![1]),
            );
            ReadoutCacheFamily::new(
                production_id.clone(),
                declaration,
                ArtifactConfidentiality::Public,
            )
            .unwrap()
        };
        let target_family = family(DecompositionOperandRole::ReadoutNormalization, 0);
        let normalization_family = family(DecompositionOperandRole::ReadoutNormalization, 99);
        let helper_families = (0..3)
            .map(|index| {
                let sigma = [3, 5, 7][index];
                let switch = family(DecompositionOperandRole::ReadoutAutomorphism, index * 2);
                let public = family(DecompositionOperandRole::ReadoutAutomorphism, index * 2 + 1);
                manifest
                    .artifacts
                    .insert(switch.artifact_name().to_owned(), switch.manifest_artifact());
                manifest
                    .artifacts
                    .insert(public.artifact_name().to_owned(), public.manifest_artifact());
                (sigma, switch, public)
            })
            .collect::<Vec<_>>();
        manifest
            .artifacts
            .insert(target_family.artifact_name().to_owned(), target_family.manifest_artifact());
        manifest.artifacts.insert(
            normalization_family.artifact_name().to_owned(),
            normalization_family.manifest_artifact(),
        );
        let schemas = ReadoutPreimageSchema::new(4, 2, 2).unwrap();
        let artifacts = NativePrfReadoutArtifacts::import_from_cache(
            layout,
            &ring,
            &manifest,
            &target_family,
            &helper_families,
            &normalization_family,
            schemas.clone(),
            schemas,
        )
        .unwrap();
        let batch =
            NativePrfEncodingBatchOutputs::new(ring.input_family("imported-vectors", 1, (1, 4)), 1)
                .unwrap();
        let compiler = mxx_bgg::BggEncodingCompiler;
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(layout, 1).unwrap(), 1, 1).unwrap();
        let online_masks = [3usize, 5, 7]
            .into_iter()
            .map(|sigma| {
                (
                    sigma,
                    BggEncodingWire {
                        vector: ring.input(format!("online-mask-{sigma}"), (1, 4)),
                        plaintext: None,
                    },
                )
            })
            .collect::<Vec<_>>();
        let output = program
            .execute_value_readout_batch_with_masks(&compiler, &batch, &artifacts, &online_masks)
            .unwrap();
        let graph = mxx_dsl::DslContext::new("imported-readout-online")
            .family_output("vectors", output.vectors().clone())
            .unwrap()
            .build()
            .unwrap();
        assert!(!graph.graph.scopes().values().flat_map(|scope| scope.nodes()).any(
            |node| matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
        ));
    }
}
