//! Public-key projection of the setup-fixed Exponent-LUT evaluator.
//!
//! This path receives only public BGG matrices, fixed GSW ciphertexts, and
//! public helper matrices. It mirrors the private evaluator without
//! knowing encoding vectors, plaintexts, secrets, or selector schedules.
//!
//! If a private branch is `(v_jD_j,A_jD_j)`, this path retains `A_jD_j`:
//! public Fuse is `A -> A G^{-1}(C)`, and a public LUT sums
//! `M_j (A^{\sigma_j} G^{-1}(C_j))` in the same odd-automorphism order.

use super::*;
use crate::noise_refresh::{prf::PrfPublicSetup, setup::RefreshSetupError};
use std::{collections::BTreeMap, sync::Arc};

#[cfg(test)]
use crate::program::{
    FamilyRange, ProgramBindings, ProgramFamilyBindings, ProgramInputId, RhsId, WireId,
};
use crate::{
    bgg::{BggError, ExponentRhsPackage, encoding::LutPublicMaskBank},
    program::{
        ExponentLutProgram, ExponentProgramLowering, LutTable, ProgramGateInstance,
        ProgramLoweringTypes,
    },
    utils::balanced_sum_family,
};
use mxx_bgg::BggPublicKeyWire;
use mxx_dsl::{Bytes, Family, GraphValue, HashTag, Int, Mat, Preimage};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_traits::ToPrimitive;

impl PrfPublicSetup<BggScheme> {
    /// Samples the setup-fixed public key and terminal helpers. The caller
    /// supplies the existing payload-secret sampler, so these artifacts share
    /// its secret and preserve the preprocessing/online separation.
    pub(crate) fn sample(
        parameters: &BggRefreshParameters,
        sampler: &super::encoding::ExponentLutEncodingSampler,
        readout_width: usize,
        setup_secret: Mat,
        hash_key: Bytes,
    ) -> Result<Self, RefreshSetupError> {
        let ring = parameters.layout.ring();
        let helpers = sampler.sample_native_readout_helper_families(
            setup_secret,
            hash_key.clone(),
            readout_width,
            b"refresh-parameter-search-readout-mask-bank".as_slice(),
            b"refresh-parameter-search-readout-helpers".as_slice(),
        )?;
        let initial = mxx_bgg::BggPublicKeySampler { layout: sampler.layout.clone() }
            .sample(hash_key, b"refresh-parameter-search-initial".as_slice(), &[false])
            .into_iter()
            .nth(1)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        Ok(Self {
            initial,
            readout_preparation: (
                helpers.sigmas,
                helpers.switch_ciphertexts,
                ring.gadget(
                    parameters.component_count,
                    parameters.layout.gadget_base.clone(),
                    parameters.layout.digit_count,
                ),
            ),
            readout_public_masks: helpers.public_masks,
        })
    }

    /// Validates the public PRF setup against the refresh dimensions and
    /// ring. The checks intentionally inspect only setup-fixed
    /// public matrices and declarations.
    pub fn validate(&self, parameters: &BggRefreshParameters) -> Result<(), RefreshSetupError> {
        parameters.validate_dimensions()?;
        let ring = parameters.layout.ring();
        if self.initial.reveal_plaintext ||
            *self.initial.matrix.matrix_type() !=
                ring.matrix_type((
                    parameters.component_count,
                    parameters.layout.public_key_columns(),
                ))
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        if self.readout_preparation.2.matrix_type().modulus != ring.matrix_type((1, 1)).modulus ||
            self.readout_preparation.2.matrix_type().ring_dimension !=
                ring.matrix_type((1, 1)).ring_dimension
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        Ok(())
    }
}

/// Preprocessing-produced gadget decompositions consumed by online encoding
/// evaluation.  Only generic typed preimages are stored; no encoding or
/// secret-dependent public matrix is materialized as an artifact.
#[derive(Clone, PartialEq)]
pub struct ProgramCacheBindings<P = Preimage, L = ProgramLutCache> {
    gate: BTreeMap<ProgramGateInstance, P>,
    lut: BTreeMap<ProgramGateInstance, L>,
}
// Structured gate identities are serialized as ordered entries, not JSON keys.
impl<P: Serialize, L: Serialize> Serialize for ProgramCacheBindings<P, L> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (self.gate.iter().collect::<Vec<_>>(), self.lut.iter().collect::<Vec<_>>())
            .serialize(serializer)
    }
}
impl<'de, P: Deserialize<'de>, L: Deserialize<'de>> Deserialize<'de>
    for ProgramCacheBindings<P, L>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (gate, lut) =
            <(Vec<(ProgramGateInstance, P)>, Vec<(ProgramGateInstance, L)>)>::deserialize(
                deserializer,
            )?;
        Ok(Self { gate: gate.into_iter().collect(), lut: lut.into_iter().collect() })
    }
}

impl<P, L> Default for ProgramCacheBindings<P, L> {
    fn default() -> Self {
        Self { gate: BTreeMap::new(), lut: BTreeMap::new() }
    }
}
impl<P: mxx_dsl::GraphValue, L: mxx_dsl::GraphValue> mxx_dsl::GraphValue
    for ProgramCacheBindings<P, L>
{
    type Schema = ProgramCacheBindings<P::Schema, L::Schema>;
    fn flatten(&self) -> Vec<mxx_ir_core::ValueHandle> {
        let mut values = self.gate.flatten();
        values.extend(self.lut.flatten());
        values
    }
    fn schema(&self) -> Self::Schema {
        ProgramCacheBindings { gate: self.gate.schema(), lut: self.lut.schema() }
    }
    fn from_values(
        schema: &Self::Schema,
        values: &[mxx_ir_core::ValueHandle],
    ) -> Result<Self, mxx_dsl::DslError> {
        let (gate, lut) =
            <(BTreeMap<ProgramGateInstance, P>, BTreeMap<ProgramGateInstance, L>)>::from_values(
                &(schema.gate.clone(), schema.lut.clone()),
                values,
            )?;
        Ok(Self { gate, lut })
    }
}
impl<P: mxx_dsl::GraphValueSchema, L: mxx_dsl::GraphValueSchema> mxx_dsl::GraphValueSchema
    for ProgramCacheBindings<P, L>
{
    type Value = ProgramCacheBindings<P::Value, L::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        ProgramCacheBindings {
            gate: self.gate.placeholders_from(next),
            lut: self.lut.placeholders_from(next),
        }
    }
    fn wire_types(&self) -> Vec<mxx_ir_core::types::WireType> {
        let mut types = self.gate.wire_types();
        types.extend(self.lut.wire_types());
        types
    }
}

/// Static LUT branches are separate graph fields. The enclosing Program
/// family supplies the indexed axis, avoiding unsupported nested families.
/// Evaluation packs these fields once for the existing parallel branch loop.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramLutCache<P = Vec<Preimage>> {
    pub(crate) switch: P,
    pub(crate) branch: P,
}

impl<P: mxx_dsl::GraphValue> mxx_dsl::GraphValue for ProgramLutCache<P> {
    type Schema = ProgramLutCache<P::Schema>;

    fn flatten(&self) -> Vec<mxx_ir_core::ValueHandle> {
        let mut values = self.switch.flatten();
        values.extend(self.branch.flatten());
        values
    }

    fn schema(&self) -> Self::Schema {
        ProgramLutCache { switch: self.switch.schema(), branch: self.branch.schema() }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[mxx_ir_core::ValueHandle],
    ) -> Result<Self, mxx_dsl::DslError> {
        let (switch, branch) =
            <(P, P)>::from_values(&(schema.switch.clone(), schema.branch.clone()), values)?;
        Ok(Self { switch, branch })
    }
}

impl<P: mxx_dsl::GraphValueSchema> mxx_dsl::GraphValueSchema for ProgramLutCache<P> {
    type Value = ProgramLutCache<P::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        ProgramLutCache {
            switch: self.switch.placeholders_from(next),
            branch: self.branch.placeholders_from(next),
        }
    }

    fn wire_types(&self) -> Vec<mxx_ir_core::types::WireType> {
        let mut types = self.switch.wire_types();
        types.extend(self.branch.wire_types());
        types
    }
}

impl ProgramCacheBindings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a setup-fixed RHS before entering a structural body. The gate
    /// identity binds the RHS; later scalar preparation reuses this exact node.
    pub(crate) fn prepare_private_rhs(
        &mut self,
        gate: ProgramGateInstance,
        rhs: &ExponentRhsPackage,
        compiler: &mxx_bgg::BggPublicKeyCompiler,
    ) -> Preimage {
        self.gate
            .entry(gate)
            .or_insert_with(|| {
                rhs.gsw_ciphertext()
                    .clone()
                    .decompose(compiler.base.clone(), compiler.digit_count.clone())
            })
            .clone()
    }

    pub fn gate(mut self, step: ProgramGateInstance, rhs: Preimage) -> Self {
        self.gate.insert(step, rhs);
        self
    }

    pub fn lut(
        mut self,
        step: ProgramGateInstance,
        switch: Vec<Preimage>,
        branch: Vec<Preimage>,
    ) -> Self {
        self.lut.insert(step, ProgramLutCache { switch, branch });
        self
    }

    pub(crate) fn gate_entry(&self, step: ProgramGateInstance) -> Option<&Preimage> {
        self.gate.get(&step)
    }

    pub(crate) fn lut_entry(&self, step: ProgramGateInstance) -> Option<&ProgramLutCache> {
        self.lut.get(&step)
    }

    fn insert_gate(&mut self, step: ProgramGateInstance, rhs: Preimage) {
        self.gate.insert(step, rhs);
    }

    fn insert_lut(&mut self, step: ProgramGateInstance, cache: ProgramLutCache) {
        self.lut.insert(step, cache);
    }
}

/// Public-key-only family consumed by the public accumulator lowerer.
/// Unlike the encoding backend's carrier/plaintext family, this type contains no
/// fabricated vector or plaintext carriers: only setup public matrices are
/// accepted and exposed to `MulPublic`.
#[cfg(test)]
#[derive(Clone)]
pub struct PublicKeyWireFamily {
    matrices: Family<Mat>,
}

#[cfg(test)]
impl PublicKeyWireFamily {
    pub(crate) fn from_matrices(
        matrices: Family<Mat>,
        ring: &mxx_dsl::Ring,
        secret_dimension: usize,
        public_key_columns: usize,
    ) -> Result<Self, BggError> {
        let ring_type = ring.matrix_type((1, 1));
        let matrix_type = matrices.element_type();
        let count = matrices
            .count()
            .evaluate(&ParamEnv::default())
            .map_err(|_| BggError::InvalidSparseLwrBlock)?;
        if secret_dimension == 0 ||
            public_key_columns == 0 ||
            count <= 0.into() ||
            matrix_type.modulus != ring_type.modulus ||
            matrix_type.ring_dimension != ring_type.ring_dimension ||
            matrix_type.rows != IntExpr::constant(secret_dimension) ||
            matrix_type.columns != IntExpr::constant(public_key_columns)
        {
            return Err(BggError::InvalidSparseLwrBlock);
        }
        Ok(Self { matrices })
    }

    pub(crate) fn matrices(&self) -> &Family<Mat> {
        &self.matrices
    }
}

/// Public portion of one setup-fixed LUT helper.
#[derive(Clone)]
pub struct LutPublicHelper {
    sigma: usize,
    switch: ExponentRhsPackage,
    mask_bank: Arc<LutPublicMaskBank>,
    mask_index: usize,
}

/// Public helper branches bound to one concrete LUT table.
#[derive(Clone, PartialEq)]
pub struct LutPublicHelperSet<H = LutPublicHelper> {
    table_commitment: [u8; 32],
    width: usize,
    helpers: Vec<H>,
}

impl mxx_dsl::GraphValue for LutPublicHelperSet {
    type Schema =
        LutPublicHelperSet<(usize, mxx_dsl::MatType, Option<ExponentRhsPackage<mxx_dsl::MatType>>)>;
    fn flatten(&self) -> Vec<mxx_ir_core::ValueHandle> {
        self.helpers
            .iter()
            .flat_map(|helper| {
                let switch: Option<ExponentRhsPackage> = Some(helper.switch.clone());
                mxx_dsl::GraphValue::flatten(&(helper.mask().clone(), switch))
            })
            .collect()
    }
    fn schema(&self) -> Self::Schema {
        LutPublicHelperSet {
            table_commitment: self.table_commitment,
            width: self.width,
            helpers: self
                .helpers
                .iter()
                .map(|helper| {
                    let switch: Option<ExponentRhsPackage> = Some(helper.switch.clone());
                    (
                        helper.sigma(),
                        mxx_dsl::GraphValue::schema(helper.mask()),
                        mxx_dsl::GraphValue::schema(&switch),
                    )
                })
                .collect(),
        }
    }
    fn from_values(
        schema: &Self::Schema,
        values: &[mxx_ir_core::ValueHandle],
    ) -> Result<Self, mxx_dsl::DslError> {
        use mxx_dsl::GraphValueSchema;
        let mut offset = 0;
        let helpers = schema
            .helpers
            .iter()
            .map(|(sigma, mask, switch)| {
                let fields = (mask.clone(), switch.clone());
                let count = fields.wire_types().len();
                let (mask, switch) = <(Mat, Option<ExponentRhsPackage>)>::from_values(
                    &fields,
                    values.get(offset..offset + count).ok_or(mxx_dsl::DslError::Schema)?,
                )?;
                offset += count;
                Ok(LutPublicHelper::new(*sigma, switch.ok_or(mxx_dsl::DslError::Schema)?, mask))
            })
            .collect::<Result<Vec<_>, mxx_dsl::DslError>>()?;
        if offset != values.len() {
            return Err(mxx_dsl::DslError::Schema);
        }
        Ok(Self { table_commitment: schema.table_commitment, width: schema.width, helpers })
    }
}
impl mxx_dsl::GraphValueSchema
    for LutPublicHelperSet<(usize, mxx_dsl::MatType, Option<ExponentRhsPackage<mxx_dsl::MatType>>)>
{
    type Value = LutPublicHelperSet;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        use mxx_dsl::GraphValue;
        let values = self
            .helpers
            .iter()
            .flat_map(|(_, mask, switch)| {
                (mask.clone(), switch.clone()).placeholders_from(next).flatten()
            })
            .collect::<Vec<_>>();
        LutPublicHelperSet::from_values(self, &values).expect("helper schema placeholders")
    }
    fn wire_types(&self) -> Vec<mxx_ir_core::types::WireType> {
        self.helpers
            .iter()
            .flat_map(|(_, mask, switch)| (mask.clone(), switch.clone()).wire_types())
            .collect()
    }
}

impl LutPublicHelperSet {
    pub fn new(
        table: &crate::program::LutTable,
        helpers: Vec<LutPublicHelper>,
    ) -> Result<Self, BggError> {
        if table.values().len() != helpers.len() {
            return Err(BggError::InvalidLut);
        }
        Ok(Self { table_commitment: table.commitment(), width: helpers.len(), helpers })
    }
    pub(crate) fn resolve(
        &self,
        table: &crate::program::LutTable,
    ) -> Result<&[LutPublicHelper], BggError> {
        if self.width != table.values().len() || self.table_commitment != table.commitment() {
            return Err(BggError::InvalidLut);
        }
        Ok(&self.helpers)
    }
}

impl LutPublicHelper {
    pub fn new(sigma: usize, switch: ExponentRhsPackage, mask: Mat) -> Self {
        let mask_bank = Arc::new(LutPublicMaskBank::single(sigma, mask));
        Self::with_mask_bank(sigma, switch, mask_bank).expect("single public mask branch")
    }
    pub fn with_mask_bank(
        sigma: usize,
        switch: ExponentRhsPackage,
        mask_bank: Arc<LutPublicMaskBank>,
    ) -> Result<Self, BggError> {
        let mask_index = mask_bank.index_for_sigma(sigma).ok_or(BggError::InvalidLut)?;
        Ok(Self { sigma, switch, mask_bank, mask_index })
    }
    pub fn sigma(&self) -> usize {
        self.sigma
    }
    pub fn switch(&self) -> &ExponentRhsPackage {
        &self.switch
    }
    pub fn mask(&self) -> &Mat {
        self.mask_bank.matrix(self.mask_index).expect("validated public mask index")
    }
}

pub type LutPublicHelperMap = BTreeMap<crate::program::LutId, LutPublicHelperSet>;

#[derive(Debug, thiserror::Error)]
pub enum ExponentLutPublicSamplingError {
    #[error("invalid public Exponent-LUT sampler configuration: {0}")]
    InvalidConfiguration(&'static str),
}

/// Public setup sampler. Public matrices are derived from the existing BGG
/// hash sampler and contain no plaintext or private-schedule state. Concrete
/// selector-C setup material may affect public evaluation; its issuer need not
/// know selector plaintext, private schedule, or sampler encoding.
#[derive(Clone)]
pub struct ExponentLutPublicKeySampler {
    pub layout: mxx_bgg::BggSamplerLayout,
    pub exponent_layout: crate::layout::ExpLayout,
}
impl ExponentLutPublicKeySampler {
    /// Samples `count` ciphertext-only input public keys from one indexed BGG
    /// family. The sampler's reserved index zero is discarded.
    pub fn sample_input_keys(
        &self,
        hash_key: Bytes,
        base_tag: impl Into<HashTag>,
        count: usize,
    ) -> Result<Vec<BggPublicKeyWire>, ExponentLutPublicSamplingError> {
        if count == 0 {
            return Err(ExponentLutPublicSamplingError::InvalidConfiguration(
                "input-key count must be positive",
            ));
        }
        let keys = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() }.sample(
            hash_key,
            base_tag,
            &vec![false; count],
        );
        Ok(keys.into_iter().skip(1).collect())
    }

    /// Samples the public projection of a reusable canonical mask bank. This
    /// is intentionally independent of private helper construction: the same
    /// deterministic public-key family is regenerated from the hash key and
    /// mask-bank tag.
    pub fn sample_mask_bank(
        &self,
        hash_key: Bytes,
        max_width: usize,
        tag: impl Into<HashTag>,
    ) -> Result<Arc<LutPublicMaskBank>, ExponentLutPublicSamplingError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutPublicSamplingError::InvalidConfiguration(
                "ring dimension must be concrete",
            ))?;
        if max_width == 0 ||
            !max_width.is_power_of_two() ||
            max_width > n ||
            n % max_width != 0 ||
            max_width > self.exponent_layout.width()
        {
            return Err(ExponentLutPublicSamplingError::InvalidConfiguration(
                "mask-bank width must be a power of two dividing the ring dimension",
            ));
        }
        let root = tag.into();
        let sampler = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() };
        let mut branches = Vec::with_capacity(max_width);
        let bank_layout = crate::layout::ExpLayout {
            width_log2: max_width.trailing_zeros(),
            ..self.exponent_layout
        };
        let sigmas = bank_layout.automorphism_indices(n).map_err(|_| {
            ExponentLutPublicSamplingError::InvalidConfiguration("invalid mask-bank sigma")
        })?;
        for sigma in sigmas {
            let key = sampler
                .sample(
                    hash_key.clone(),
                    crate::bgg::encoding::canonical_lut_mask_branch_tag(&root, sigma),
                    &[false],
                )
                .into_iter()
                .nth(1)
                .ok_or(ExponentLutPublicSamplingError::InvalidConfiguration(
                    "mask sample is empty",
                ))?;
            branches.push((sigma, key.matrix));
        }
        Ok(Arc::new(LutPublicMaskBank::from_branches(branches)))
    }
}

#[derive(Clone)]
pub struct ExponentLutPublicKeyCompiler {
    pub public_key: mxx_bgg::BggPublicKeyCompiler,
    pub exponent_layout: crate::layout::ExpLayout,
}
impl ExponentLutPublicKeyCompiler {
    pub fn new(public_key: mxx_bgg::BggPublicKeyCompiler) -> Self {
        Self {
            public_key,
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 },
        }
    }
    pub fn with_layout(
        public_key: mxx_bgg::BggPublicKeyCompiler,
        exponent_layout: crate::layout::ExpLayout,
    ) -> Self {
        Self { public_key, exponent_layout }
    }
    #[cfg(test)]
    pub(crate) fn compile_program(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggPublicKeyWire>,
        rhs_inputs: &BTreeMap<RhsId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, PublicRhsFamily>,
        values: &BTreeMap<crate::program::WireFamilyId, PublicKeyWireFamily>,
        helpers: &LutPublicHelperMap,
    ) -> Result<BTreeMap<WireId, BggPublicKeyWire>, BggError> {
        let mut ranges = ProgramFamilyBindings::new();
        for (index, family) in selectors.values().enumerate() {
            ranges.bind(
                crate::program::ProgramFamilyId::from_index(index),
                FamilyRange::full(family.gsw().count().clone())
                    .map_err(|_| BggError::InvalidSparseLwrBlock)?,
            );
        }
        Ok(self
            .compile_program_with_preprocessing(
                program, inputs, rhs_inputs, selectors, values, &ranges, helpers,
            )?
            .public_values)
    }

    /// Result of public-program preprocessing. The ordinary public wire map
    /// is retained for protocol outputs, while every multiplication and LUT's
    /// typed preimages are recorded by the same traversal.
    #[cfg(test)]
    pub(crate) fn compile_program_with_preprocessing(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggPublicKeyWire>,
        rhs_inputs: &BTreeMap<RhsId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, PublicRhsFamily>,
        values: &BTreeMap<crate::program::WireFamilyId, PublicKeyWireFamily>,
        ranges: &ProgramFamilyBindings,
        helpers: &LutPublicHelperMap,
    ) -> Result<crate::scheme::PreparedProgram<BggScheme>, BggError> {
        let selectors = selectors
            .iter()
            .map(|(id, family)| {
                Ok((
                    *id,
                    (
                        family
                            .gsw()
                            .field(|gsw_ciphertext| ExponentRhsPackage { gsw_ciphertext })?,
                        FamilyRange::full(family.gsw().count().clone())
                            .map_err(|_| mxx_dsl::DslError::Schema)?,
                    ),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, mxx_dsl::DslError>>()?;
        let values = values
            .iter()
            .map(|(id, family)| {
                Ok((
                    *id,
                    (
                        family
                            .matrices()
                            .field(|matrix| BggPublicKeyWire { matrix, reveal_plaintext: true })?,
                        FamilyRange::full(family.matrices().count().clone())
                            .map_err(|_| mxx_dsl::DslError::Schema)?,
                    ),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, mxx_dsl::DslError>>()?;
        let helpers = helpers
            .iter()
            .map(|(id, value)| (*id, crate::program::ProgramValue::Scalar(value.clone())))
            .collect();
        Ok(crate::scheme::PreparedProgram::new(
            BggScheme::new(self.public_key.clone(), self.exponent_layout),
            program.clone(),
            &ProgramBindings::new(inputs, rhs_inputs, &selectors, &values, &helpers, None),
            ranges.clone(),
            None,
        )?)
    }

    /// Setup-fixed Fuse is the public matrix product `A G^{-1}(C)`.
    #[cfg(test)]
    pub(crate) fn fuse_public(
        &self,
        input: &Mat,
        rhs: &ExponentRhsPackage,
    ) -> Result<Mat, BggError> {
        // Public projection of Fuse: multiply `A` by the digit matrix of C.
        Ok(rhs
            .gsw_ciphertext()
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            .mul_small_rhs(input.clone()))
    }

    /// Cache-required public Fuse boundary. The typed RHS decomposition is
    /// consumed directly; no public-output matrix artifact is transported.
    pub fn fuse_public_with_preprocessed_artifacts(
        &self,
        input: &Mat,
        decomposition: Preimage,
    ) -> Result<Mat, BggError> {
        Ok(decomposition.mul_small_rhs(input.clone()))
    }
    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn single_input_lut(
        &self,
        input: &Mat,
        table: &[usize],
        helpers: &[LutPublicHelper],
    ) -> Result<Mat, BggError> {
        let width = table.len();
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(BggError::InvalidLut)?;
        let table =
            LutTable::unary(width, self.exponent_layout.effective_modulus(n), table.to_vec())
                .map_err(BggError::from)?;
        self.single_input_lut_table(input, &table, helpers)
    }

    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn single_input_lut_table(
        &self,
        input: &Mat,
        table: &LutTable,
        helpers: &[LutPublicHelper],
    ) -> Result<Mat, BggError> {
        let width = table.values().len();
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(BggError::InvalidLut)?;
        let lut_layout = crate::utils::derive_lut_layout(&self.exponent_layout, width, n)?;
        if width == 0 ||
            !width.is_power_of_two() ||
            width > n ||
            n % width != 0 ||
            helpers.len() != width ||
            table.values().iter().any(|value| *value >= lut_layout.effective_modulus(n))
        {
            return Err(BggError::InvalidLut);
        }
        let sigmas = lut_layout.automorphism_indices(n).map_err(|_| BggError::InvalidLut)?;
        // The branch body computes `mask_j * (A^{\sigma_j} G^{-1}(C_j))`, the
        // public component corresponding to the private Fuse branch.
        let raw_publics = Family::pack(
            helpers
                .iter()
                .enumerate()
                .map(|(j, helper)| {
                    let expected = sigmas.get(j).copied().ok_or(BggError::InvalidLut)?;
                    if helper.sigma() != expected {
                        return Err(BggError::InvalidLut);
                    }
                    Ok(input.clone().ring_automorphism(expected))
                })
                .collect::<Result<Vec<_>, BggError>>()?,
        )
        .map_err(|_| BggError::InvalidLut)?;
        let switches = Family::pack(
            helpers.iter().map(|helper| helper.switch().gsw_ciphertext().clone()).collect(),
        )
        .map_err(|_| BggError::InvalidLut)?;
        let masks = Family::pack(helpers.iter().map(|helper| helper.mask().clone()).collect())
            .map_err(|_| BggError::InvalidLut)?;
        let branches = {
            let source_families = vec![raw_publics, switches, masks];
            if source_families.is_empty() ||
                source_families.iter().any(|family| family.count() != source_families[0].count())
            {
                Err(mxx_dsl::DslError::Schema)
            } else {
                mxx_dsl::parallel(source_families[0].count().clone(), |_index| {
                    let mut items =
                        source_families.iter().map(|family| family.at(&_index)).collect::<Vec<_>>();
                    {
                        // The families were zipped as `[raw_publics, switches, masks]`;
                        // `pop` therefore yields the public mask, the concrete
                        // setup-fixed selector ciphertext `C_{sigma,L}`, then the
                        // automorphed public-key matrix `A^{\sigma}`.
                        let mask = items.pop().ok_or(mxx_dsl::DslError::Schema)?;
                        // `mask` is the public matrix from the mask-alignment helper
                        // encoding. Its private vector switches the mask secret from
                        // `sigma(s)` back to `s` after the automorphed fixed-C Fuse;
                        // this public closure computes only that helper's projection.
                        let switch = items.pop().ok_or(mxx_dsl::DslError::Schema)?;
                        // `switch` is the concrete setup-fixed GSW ciphertext
                        // `C_{sigma,L}`, wrapped below for fixed-RHS shape validation.
                        let raw = items.pop().ok_or(mxx_dsl::DslError::Schema)?;
                        // `raw = A^\sigma` is the automorphed public-key matrix; no
                        // payload or plaintext term is formed on this public path.
                        let rhs = ExponentRhsPackage::new(switch)
                            .map_err(|_| mxx_dsl::DslError::Schema)?;
                        // `fused = A^{\sigma} G^{-1}(C_{sigma,L})` is still public.
                        let fused = rhs
                            .gsw_ciphertext()
                            .clone()
                            .decompose(
                                self.public_key.base.clone(),
                                self.public_key.digit_count.clone(),
                            )
                            .mul_small_rhs(raw);
                        // `decomp = G^{-1}(fused)` has the gadget-digit shape needed
                        // by `mask`; `mask * decomp` is the helper's public projection
                        // that accompanies the private vector-side alignment.
                        Ok(fused
                            .decompose(
                                self.public_key.base.clone(),
                                self.public_key.digit_count.clone(),
                            )
                            .mul_small_rhs(mask))
                    }
                })
            }
        }
        .map_err(|_| BggError::InvalidLut)?;
        balanced_sum_family(branches).map_err(Into::into)
    }

    /// Public LUT lowering with all branch decompositions supplied by the
    /// current preprocessing traversal. This keeps the public matrix result
    /// materialized while ensuring each switch and switched branch is
    /// decomposed exactly once.
    fn single_input_lut_table_with_cache(
        &self,
        input: &Mat,
        table: &LutTable,
        helpers: &[LutPublicHelper],
        cache: &ProgramLutCache,
    ) -> Result<Mat, BggError> {
        let width = table.values().len();
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(BggError::InvalidLut)?;
        let lut_layout = crate::utils::derive_lut_layout(&self.exponent_layout, width, n)?;
        if width == 0 ||
            !width.is_power_of_two() ||
            width > n ||
            n % width != 0 ||
            helpers.len() != width ||
            cache.switch.len() != width ||
            cache.branch.len() != width ||
            table.values().iter().any(|value| *value >= lut_layout.effective_modulus(n))
        {
            return Err(BggError::InvalidLut);
        }
        let sigmas = lut_layout.automorphism_indices(n).map_err(|_| BggError::InvalidLut)?;
        for (index, helper) in helpers.iter().enumerate() {
            if helper.sigma() != sigmas[index] {
                return Err(BggError::InvalidLut);
            }
        }
        let masks = Family::pack(helpers.iter().map(|helper| helper.mask().clone()).collect())
            .map_err(|_| BggError::InvalidLut)?;
        let aligned = masks
            .zip_map(Family::pack(cache.branch.clone())?, |mask, decomposition| {
                Ok(decomposition.mul_small_rhs(mask))
            })
            .map_err(|_| BggError::InvalidLut)?;
        let _ = input;
        balanced_sum_family(aligned).map_err(Into::into)
    }

    pub(crate) fn build_lut_cache(
        &self,
        input: &Mat,
        table: &LutTable,
        helpers: &[LutPublicHelper],
    ) -> Result<ProgramLutCache, BggError> {
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(BggError::InvalidLut)?;
        let sigmas =
            crate::utils::derive_lut_layout(&self.exponent_layout, table.values().len(), n)?
                .automorphism_indices(n)
                .map_err(|_| BggError::InvalidLut)?;
        if helpers.len() != sigmas.len() {
            return Err(BggError::InvalidLut);
        }
        let mut switch = Vec::with_capacity(sigmas.len());
        let mut branch = Vec::with_capacity(sigmas.len());
        for (index, helper) in helpers.iter().enumerate() {
            if helper.sigma() != sigmas[index] {
                return Err(BggError::InvalidLut);
            }
            let switch_decomposition = helper
                .switch()
                .gsw_ciphertext()
                .clone()
                .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone());
            let switched = switch_decomposition
                .clone()
                .mul_small_rhs(input.clone().ring_automorphism(sigmas[index]));
            switch.push(switch_decomposition);
            branch.push(
                switched
                    .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone()),
            );
        }
        Ok(ProgramLutCache { switch, branch })
    }
}

pub struct PublicProgramLowering<'a> {
    pub(crate) shared_cache: Option<&'a ProgramCacheBindings>,
    pub(crate) compiler: &'a ExponentLutPublicKeyCompiler,
    pub(crate) cache: Option<&'a mut ProgramCacheBindings>,
}

impl ProgramLoweringTypes for PublicProgramLowering<'_> {
    type Wire = BggPublicKeyWire;
    type Rhs = ExponentRhsPackage;
    type HelperSet = LutPublicHelperSet;
    type Error = BggError;
}

impl ExponentProgramLowering for PublicProgramLowering<'_> {
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let decomposition = private.matrix.clone().decompose(
            self.compiler.public_key.base.clone(),
            self.compiler.public_key.digit_count.clone(),
        );
        if let Some(cache) = &mut self.cache {
            cache.insert_gate(gate, decomposition.clone());
        }
        Ok(self.compiler.public_key.mul_with_decomposition(public, private, decomposition))
    }

    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        _: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(self.compiler.public_key.add(lhs, rhs))
    }

    fn mul_private(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Rhs,
        _: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let decomposition = if let Some(value) = self
            .cache
            .as_ref()
            .and_then(|cache| cache.gate_entry(gate))
            .or_else(|| self.shared_cache.and_then(|cache| cache.gate_entry(gate)))
        {
            value.clone()
        } else {
            match &mut self.cache {
                Some(cache) => cache.prepare_private_rhs(gate, rhs, &self.compiler.public_key),
                None => rhs.gsw_ciphertext().clone().decompose(
                    self.compiler.public_key.base.clone(),
                    self.compiler.public_key.digit_count.clone(),
                ),
            }
        };
        Ok(BggPublicKeyWire {
            matrix: decomposition.mul_small_rhs(lhs.matrix.clone()),
            reveal_plaintext: false,
        })
    }

    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: crate::program::LutId,
        input: &Self::Wire,
        helpers: Option<&Self::HelperSet>,
        _: Option<&Self::Rhs>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let helpers = helpers.ok_or(BggError::InvalidLut)?;
        let table = program.lut(lut).ok_or(BggError::InvalidLut)?;
        let helpers = helpers.resolve(table)?;
        let cache = self.compiler.build_lut_cache(&input.matrix, table, helpers)?;
        let matrix = self.compiler.single_input_lut_table_with_cache(
            &input.matrix,
            table,
            helpers,
            &cache,
        )?;
        if let Some(bindings) = &mut self.cache {
            bindings.insert_lut(gate, cache);
        }
        Ok(BggPublicKeyWire { matrix, reveal_plaintext: false })
    }
}

pub type PublicRhsFamily = crate::bgg::encoding::SetupRhsFamily;

#[cfg(test)]
mod tests {
    use super::{
        ExponentLutPublicKeyCompiler, ExponentLutPublicKeySampler, LutPublicHelper,
        LutPublicHelperSet,
    };
    use crate::{bgg::ExponentRhsPackage, program::LutTable};
    use mxx_bgg::{BggPublicKeyCompiler, BggPublicKeyWire};
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::ParamEnv;
    use std::collections::BTreeMap;

    #[test]
    fn public_input_sampler_batches_and_rejects_empty_count() {
        let sampler = ExponentLutPublicKeySampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 2 },
        };
        let ring = Ring::new(97, 4);
        let hash_key = ring.bytes_input("public-batch-key", 32);
        assert!(
            sampler.sample_input_keys(hash_key.clone(), b"public-batch".as_slice(), 0).is_err()
        );
        let batch =
            sampler.sample_input_keys(hash_key.clone(), b"public-batch".as_slice(), 2).unwrap();
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|key| !key.reveal_plaintext));
        let one = sampler.sample_input_keys(hash_key, b"public-batch".as_slice(), 1).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].matrix.matrix_type(), batch[0].matrix.matrix_type());
    }

    #[test]
    fn public_fuse_uses_the_setup_fixed_ciphertext() {
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let input = ring.zero((2, 2));
        let rhs = ExponentRhsPackage::new(ring.zero((2, 2))).unwrap();
        let output = compiler.fuse_public(&input, &rhs).unwrap();
        let expected = rhs.gsw_ciphertext().clone().decompose(4, 2).mul_small_rhs(input);
        assert_eq!(output.matrix_type(), expected.matrix_type());
        assert!(matches!(
            output.value_handle().node().kind(),
            mxx_ir_core::node::NodeKind::MatrixMulSmallRhs
        ));
    }

    #[test]
    fn public_projection_accepts_a_typed_public_input() {
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 }, 4).unwrap();
        let input = builder
            .input(
                2,
                crate::program::ProgramWireType {
                    stride_log2: 0,
                    knowledge: crate::program::AttributeKnowledge::Public,
                },
            )
            .unwrap();
        let wire = builder.input_wire(input).unwrap();
        builder.output(wire).unwrap();
        let program = builder.build().unwrap();
        let public = BggPublicKeyWire {
            matrix: ring.input("typed-public", (2, 2)),
            reveal_plaintext: false,
        };
        let outputs = compiler
            .compile_program(
                &program,
                &BTreeMap::from([(input, public.clone())]),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(outputs.get(&wire).unwrap().matrix.matrix_type(), public.matrix.matrix_type());
    }

    #[test]
    fn cache_required_public_fuse_rejects_missing_artifacts() {
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let input = ring.zero((2, 2));
        let error = compiler
            .fuse_public_with_preprocessed_artifacts(
                &input,
                ring.preimage_input("cached-public-fuse", (4, 2), 7),
            )
            .expect("cache-required public fuse");
        assert_eq!(error.matrix_type(), &ring.matrix_type((2, 2)));
    }

    #[test]
    fn public_helper_set_rejects_a_different_lut_table() {
        let ring = Ring::new(97, 4);
        let rhs = ExponentRhsPackage::new(ring.zero((2, 2))).unwrap();
        let helper = LutPublicHelper::new(1, rhs, ring.zero((2, 2)));
        let table = LutTable::unary(1, 2, vec![0]).unwrap();
        let other = LutTable::unary(1, 2, vec![1]).unwrap();
        let set = LutPublicHelperSet::new(&table, vec![helper]).unwrap();
        assert!(set.resolve(&table).is_ok());
        assert!(set.resolve(&other).is_err());
    }

    #[test]
    fn public_prf_batch_keeps_output_dependent_cache_structural() {
        use crate::{
            bgg::BggScheme,
            layout::ExpLayout,
            noise_refresh::prf::{
                PreparedSparseLwrPrf, PrfSelectionInputs, SparseLwrPrfProfile, SparseLwrPrfProgram,
            },
            program::{FamilyRange, ProgramValue},
        };
        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let scheme = BggScheme::new(
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() },
            layout,
        );
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(layout, 1).unwrap(), 2, 1).unwrap();
        let selectors =
            crate::bgg::encoding::SetupRhsFamily::new(ring.input_family("selectors", 2, (2, 4)))
                .unwrap();
        let block = PrfSelectionInputs::new(
            selectors,
            &ring.input_family("labels", 34, (1, 1)),
            FamilyRange::full(2).unwrap(),
            rand::random(),
        )
        .unwrap();
        let (public, prepared) = PreparedSparseLwrPrf::new(
            &scheme,
            program,
            ProgramValue::Scalar(BggPublicKeyWire {
                matrix: ring.input("initial", (2, 4)),
                reveal_plaintext: false,
            }),
            &[block],
        )
        .unwrap();
        let ProgramValue::Family(public) = public else { panic!("public family") };
        assert_eq!(public.count(), &17.into());
        assert_eq!(prepared.steps().len(), 1);
        assert_eq!(prepared.steps()[0].count(), &17.into());
        let context =
            DslContext::new("public-prf-output-axis").public_output("outputs", public).unwrap();
        let (context, _) = prepared.export(&scheme, context, "composed-cache").unwrap();
        let graph = context.build().unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        assert_eq!(
            graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .filter(|node| matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::GadgetDecompose { .. }
                ))
                .count(),
            1
        );
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
    ) -> Result<Self, BggReadoutError> {
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

impl SparseLwrPrfProgram {
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
    ) -> Result<NativePrfReadoutArtifacts, BggReadoutError> {
        with_benchmark_role(BenchmarkRole::PublicReadout, || {
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
                return Err(ValueReadoutError::InvalidTraceSet.into());
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
                    return Err(BggReadoutError::InvalidPreimageSchema.into());
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
                return Err(BggReadoutError::InvalidPreimageSchema.into());
            }
            Ok(NativePrfReadoutArtifacts::new(
                target_decomposition,
                target_schema,
                helpers,
                normalization_decomposition,
                normalization_schema,
            ))
        })
    }
}

impl ReadoutPreimageSchema {
    pub fn new(
        gadget_base: impl Into<BigInt>,
        digit_count: usize,
        max_coefficient_bound: impl Into<BigInt>,
    ) -> Result<Self, BggReadoutError> {
        let schema = Self {
            gadget_base: gadget_base.into(),
            digit_count,
            max_coefficient_bound: max_coefficient_bound.into(),
        };
        if schema.gadget_base <= BigInt::one() {
            return Err(BggReadoutError::InvalidGadgetBase.into());
        }
        if schema.digit_count == 0 {
            return Err(BggReadoutError::InvalidDigitCount.into());
        }
        if schema.max_coefficient_bound < BigInt::zero() {
            return Err(BggReadoutError::InvalidPreimageBound.into());
        }
        Ok(schema)
    }
}

impl ValueReadoutPublicHelper {
    pub fn new(
        layout: ExpLayout,
        sigma: usize,
        gadget_base: BigInt,
        digit_count: usize,
        max_coefficient_bound: BigInt,
        public_decomposition: Preimage,
        mask: Mat,
    ) -> Result<Self, BggReadoutError> {
        if sigma == 0 || sigma % 2 == 0 || sigma == 1 {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        if gadget_base <= BigInt::one() {
            return Err(BggReadoutError::InvalidGadgetBase.into());
        }
        if digit_count == 0 {
            return Err(BggReadoutError::InvalidDigitCount.into());
        }
        if max_coefficient_bound < BigInt::zero() {
            return Err(BggReadoutError::InvalidPreimageBound.into());
        }
        Ok(Self {
            layout,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
            public_decomposition,
            mask,
        })
    }

    pub fn sigma(&self) -> usize {
        self.sigma
    }

    pub fn layout(&self) -> ExpLayout {
        self.layout
    }
}

impl ValueReadout {
    /// Executes the public-key projection of value readout with the same
    /// single-layer trace and explicit target/normalization preimages as the
    /// encoding path. Cache-time output-key attachment is kept separate from
    /// this algebraic projection.
    pub fn project_public_with_preprocessed_artifacts(
        &self,
        compiler: &BggPublicKeyCompiler,
        input: &BggPublicKeyWire,
        target_decomposition: Preimage,
        target_schema: ReadoutPreimageSchema,
        helpers: &[ValueReadoutPublicHelper],
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<Mat, BggReadoutError> {
        let ring = compiler.ring.clone();
        self.validate(&ring)?;
        let ring_dimension = concrete_ring_dimension(&ring)?;
        let plan = self.trace_plan(ring_dimension)?;
        if helpers.len() != plan.helper_count() ||
            helpers.iter().map(ValueReadoutPublicHelper::sigma).collect::<Vec<_>>() !=
                plan.non_identity_helpers()
        {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        if helpers.iter().any(|helper| helper.layout != self.layout) {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        validate_preimage_ring(&target_decomposition, &ring)?;
        validate_preimage_ring(&normalization_decomposition, &ring)?;
        let (gadget_base, digit_count) = public_bgg_schema(compiler)?;
        if target_schema.gadget_base != gadget_base ||
            target_schema.digit_count != digit_count ||
            normalization_schema.gadget_base != gadget_base ||
            normalization_schema.digit_count != digit_count
        {
            return Err(BggReadoutError::InvalidPreimageSchema.into());
        }
        validate_preimage_schema(
            &target_decomposition,
            &input.matrix,
            &target_schema.gadget_base,
            target_schema.digit_count,
            Some(&target_schema.max_coefficient_bound),
        )?;
        for helper in helpers {
            if helper.gadget_base != gadget_base || helper.digit_count != digit_count {
                return Err(BggReadoutError::InvalidPreimageSchema.into());
            }
            validate_preimage_schema(
                &helper.public_decomposition,
                &input.matrix,
                &gadget_base,
                digit_count,
                Some(&helper.max_coefficient_bound),
            )?;
            validate_output_shape(&helper.mask, &input.matrix, &ring)?;
        }
        // Public projection is the preprocessing graph itself: consume the
        // supplied target decomposition and retain the resulting matrix
        // expression. Cache-time output-key attachment belongs to the
        // encoding path and must not disconnect this public algebra.
        let transformed =
            compiler.large_scalar_mul_with_decomposition(input, target_decomposition).matrix;
        let mut branches = vec![transformed.clone()];
        for helper in helpers {
            // Public projection is exactly the public component of the BGG
            // helper relation: A_mask * D(A_switched). The switched vector
            // term is private ciphertext data and has no public matrix
            // counterpart, so it must not be added here.
            let aligned = helper.public_decomposition.clone().mul_small_rhs(helper.mask.clone());
            branches.push(aligned);
        }
        let traced = crate::utils::balanced_sum_family(
            Family::pack(branches).map_err(|_| crate::bgg::BggError::InvalidLut)?,
        )
        .map_err(crate::bgg::BggError::from)?;
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let _inverse = modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(BggReadoutError::NonInvertibleNormalization)?;
        validate_preimage_schema(
            &normalization_decomposition,
            &traced,
            &normalization_schema.gadget_base,
            normalization_schema.digit_count,
            Some(&normalization_schema.max_coefficient_bound),
        )?;
        let normalized = compiler.large_scalar_mul_with_decomposition(
            &BggPublicKeyWire { matrix: traced, reveal_plaintext: false },
            normalization_decomposition,
        );
        Ok(normalized.matrix)
    }
    /// Public-key projection for an output-indexed PRF batch. The public
    /// helper decomposition family is flattened as `[output][helper]`, while
    /// setup-fixed masks and normalization remain shared.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_public_batch_with_preprocessed_family_artifacts(
        &self,
        compiler: &BggPublicKeyCompiler,
        inputs: Family<Mat>,
        output_count: usize,
        target_decompositions: Family<Preimage>,
        target_schema: ReadoutPreimageSchema,
        helper_sigmas: &[usize],
        public_decompositions: Family<Preimage>,
        masks: Family<Mat>,
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<Family<Mat>, BggReadoutError> {
        let ring = compiler.ring.clone();
        self.validate(&ring)?;
        let ring_dimension = concrete_ring_dimension(&ring)?;
        let plan = self.trace_plan(ring_dimension)?;
        let helper_count = helper_sigmas.len();
        let flat_count =
            output_count.checked_mul(helper_count).ok_or(ValueReadoutError::InvalidTraceSet)?;
        if output_count == 0 ||
            *inputs.count() != IntExpr::constant(output_count) ||
            helper_sigmas != plan.non_identity_helpers() ||
            *public_decompositions.count() != IntExpr::constant(flat_count) ||
            *masks.count() != IntExpr::constant(helper_count)
        {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        let (gadget_base, digit_count) = public_bgg_schema(compiler)?;
        if target_schema.gadget_base != gadget_base ||
            target_schema.digit_count != digit_count ||
            normalization_schema.gadget_base != gadget_base ||
            normalization_schema.digit_count != digit_count
        {
            return Err(BggReadoutError::InvalidPreimageSchema.into());
        }
        validate_preimage_schema(
            &target_decompositions.at(0),
            &inputs.at(0),
            &gadget_base,
            digit_count,
            Some(&target_schema.max_coefficient_bound),
        )?;
        validate_family_preimage_shape(
            public_decompositions.element_type(),
            &inputs.at(0),
            digit_count,
        )?;
        validate_family_output_shape(masks.element_type(), &inputs.at(0), &ring)?;

        if *target_decompositions.count() != IntExpr::constant(output_count) {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        let transformed = inputs
            .zip_map(target_decompositions, |input, target| Ok(target.mul_small_rhs(input)))
            .map_err(|_| BggReadoutError::Graph(crate::bgg::BggError::InvalidLut))?;
        let helper_indices = mxx_dsl::parallel(flat_count, |index| {
            Ok(index.clone().rem(Int::constant(helper_count)))
        })
        .map_err(|_| BggReadoutError::Graph(crate::bgg::BggError::InvalidLut))?;
        let flat_masks = masks
            .gather(helper_indices)
            .map_err(|_| BggReadoutError::Graph(crate::bgg::BggError::InvalidLut))?;
        let helper_branches = public_decompositions
            .zip_map(flat_masks, |public, mask| Ok(public.mul_small_rhs(mask)))
            .map_err(|_| BggReadoutError::Graph(crate::bgg::BggError::InvalidLut))?;
        let helper_sums =
            reduce_matrix_family_segments(helper_branches, output_count, helper_count)?;
        let traced = transformed
            .zip_map(helper_sums, |identity, helpers| Ok(identity + helpers))
            .map_err(|_| BggReadoutError::Graph(crate::bgg::BggError::InvalidLut))?;
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(BggReadoutError::NonInvertibleNormalization)?;
        validate_preimage_schema(
            &normalization_decomposition,
            &traced.at(0),
            &normalization_schema.gadget_base,
            normalization_schema.digit_count,
            Some(&normalization_schema.max_coefficient_bound),
        )?;
        {
            let source_family = traced;
            mxx_dsl::parallel(source_family.count().clone(), |family_index| {
                let matrix = source_family.at(&family_index);
                let _ = family_index;
                Ok(normalization_decomposition.clone().mul_small_rhs(matrix))
            })
        }
        .map_err(|_| BggReadoutError::Graph(crate::bgg::BggError::InvalidLut))
    }
    /// Returns the normalization descriptor `(2*n')^-1` for this ring.
    pub fn normalization_target_descriptor(
        &self,
        ring: &Ring,
        gadget: &Mat,
    ) -> Result<NormalizationTargetDescriptor, BggReadoutError> {
        self.validate(ring)?;
        if gadget.matrix_type().modulus != ring.matrix_type((1, 1)).modulus ||
            gadget.matrix_type().ring_dimension != ring.matrix_type((1, 1)).ring_dimension
        {
            return Err(ValueReadoutError::RingTypeMismatch.into());
        }
        let matrix = concrete_matrix_type(gadget.matrix_type())?;
        let ring_dimension = concrete_ring_dimension(ring)?;
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        Ok(NormalizationTargetDescriptor {
            matrix,
            layout: self.layout,
            numerator: BigInt::one(),
            denominator,
        })
    }
    /// Describes the bounded preimage needed for normalization.
    pub fn normalization_preimage_requirement(
        &self,
        ring: &Ring,
        gadget: Mat,
        max_coefficient_bound: impl Into<BigInt>,
        gadget_base: impl Into<BigInt>,
        digit_count: usize,
    ) -> Result<NormalizationPreimageRequirement, BggReadoutError> {
        let gadget_base = gadget_base.into();
        if gadget_base <= BigInt::one() {
            return Err(BggReadoutError::InvalidGadgetBase.into());
        }
        if digit_count == 0 {
            return Err(BggReadoutError::InvalidDigitCount.into());
        }
        let max_coefficient_bound = max_coefficient_bound.into();
        if max_coefficient_bound < BigInt::zero() {
            return Err(BggReadoutError::InvalidPreimageBound.into());
        }
        validate_canonical_gadget(&gadget, &gadget_base, digit_count)?;
        let target = self.normalization_target_descriptor(ring, &gadget)?;
        let artifact_type = ArtifactType::Preimage {
            matrix: target.matrix.clone(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        };
        Ok(NormalizationPreimageRequirement {
            target,
            artifact_type,
            role: NormalizationPreimageRole::InverseNativeTraceGadget,
            max_coefficient_bound,
            gadget_base,
            digit_count,
            readout_identity: self.identity(),
        })
    }
    /// Scales a caller-supplied gadget matrix by `(2*n')^-1` in the ring.
    ///
    /// This only constructs the public target expression.  Callers must still
    /// request a typed bounded preimage through the DSL artifact path.
    pub fn normalization_target(&self, ring: &Ring, gadget: Mat) -> Result<Mat, BggReadoutError> {
        self.validate(ring)?;
        if gadget.matrix_type().modulus != ring.matrix_type((1, 1)).modulus ||
            gadget.matrix_type().ring_dimension != ring.matrix_type((1, 1)).ring_dimension
        {
            return Err(ValueReadoutError::RingTypeMismatch.into());
        }
        let ring_dimension = concrete_ring_dimension(ring)?;
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let inverse = modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(BggReadoutError::NonInvertibleNormalization)?;
        Ok(ring.polynomial([inverse.into()]) * gadget)
    }
}

impl DecompositionCacheArtifact {
    pub fn key(&self) -> &DecompositionCacheKey {
        &self.key
    }

    pub fn artifact_name(&self) -> &str {
        &self.artifact_name
    }

    pub fn artifact_schema(&self) -> &str {
        &self.artifact_schema
    }

    pub fn family_shape(&self) -> Option<&[usize]> {
        self.family_shape.as_deref()
    }

    pub fn preimage(
        key: DecompositionCacheKey,
        matrix: ConcreteMatrixType,
        max_coefficient_bound: BigInt,
        family_shape: Option<Vec<usize>>,
    ) -> Self {
        let artifact_schema = "exponent-lut/preimage/small-rhs-row-major/v1".to_owned();
        let artifact_type = ArtifactType::Preimage { matrix, max_coefficient_bound };
        let artifact_name =
            canonical_artifact_name(&key, &artifact_schema, &artifact_type, &family_shape);
        Self { key, artifact_name, artifact_schema, artifact_type, family_shape }
    }

    pub fn is_preimage(&self) -> bool {
        matches!(self.artifact_type, ArtifactType::Preimage { .. })
    }

    pub fn artifact_type(&self) -> &ArtifactType {
        &self.artifact_type
    }

    /// Converts this declaration to the shared manifest schema.  The payload
    /// remains owned by the artifact store; this object contains only the
    /// application-owned association and exact typed preimage schema.
    pub fn manifest_artifact(&self, confidentiality: ArtifactConfidentiality) -> ManifestArtifact {
        ManifestArtifact {
            artifact_type: self.artifact_type.clone(),
            family_count: self.family_shape.as_ref().map(|shape| {
                shape
                    .iter()
                    .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                    .expect("cache family count overflow")
            }),
            confidentiality,
            content_hash: None,
            layout: None,
        }
    }
}

impl ReadoutCacheFamily {
    pub fn new(
        production_id: ProductionId,
        declaration: DecompositionCacheArtifact,
        confidentiality: ArtifactConfidentiality,
    ) -> Result<Self, ReadoutCacheImportError> {
        if !declaration.is_preimage() {
            return Err(ReadoutCacheImportError::DescriptorMismatch);
        }
        Ok(Self { production_id, declaration, confidentiality })
    }

    pub fn production_id(&self) -> &ProductionId {
        &self.production_id
    }

    pub fn declaration(&self) -> &DecompositionCacheArtifact {
        &self.declaration
    }

    pub fn artifact_name(&self) -> &str {
        self.declaration.artifact_name()
    }

    /// Returns the exact producer-manifest entry for this family.
    pub fn manifest_artifact(&self) -> ManifestArtifact {
        self.declaration.manifest_artifact(self.confidentiality)
    }

    /// Builds the producer manifest fragment consumed by
    /// `FileArtifactStore::store_manifest`. Callers may merge this entry with
    /// the other public/secret outputs of the preprocessing producer.
    pub fn manifest(&self) -> Manifest {
        Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: self.production_id.clone(),
            artifacts: BTreeMap::from([(
                self.artifact_name().to_owned(),
                self.manifest_artifact(),
            )]),
        }
    }

    /// Publishes this declaration through the shared runtime store. The
    /// preprocessing executor stores the family members separately under the
    /// same name; this method only commits the typed manifest boundary.
    pub fn store_manifest<S: ArtifactStore>(&self, store: &mut S) -> Result<(), S::Error> {
        store.store_manifest(self.manifest())
    }

    /// Loads the declaration manifest without loading any family member.
    pub fn load_manifest<S: ArtifactStore>(&self, store: &mut S) -> Result<Manifest, S::Error> {
        store.load_manifest(&self.production_id)
    }

    /// Adds the typed preprocessing family to a producer graph. The output
    /// is deliberately restricted to `Preimage` values, so ordinary public
    /// matrices cannot be accidentally registered as decoder-cache payloads.
    pub fn add_preimage_family_output(
        &self,
        context: DslContext,
        family: Family<Preimage>,
    ) -> Result<DslContext, mxx_dsl::DslError> {
        let schema = family.schema();
        let ArtifactType::Preimage { matrix, max_coefficient_bound } =
            self.declaration.artifact_type()
        else {
            return Err(mxx_dsl::DslError::Schema);
        };
        let expected_element = mxx_ir_core::types::MatrixType {
            modulus: matrix.modulus.clone().into(),
            ring_dimension: matrix.ring_dimension.into(),
            rows: matrix.rows.into(),
            columns: matrix.columns.into(),
        };
        if schema.element.matrix != expected_element ||
            schema.element.max_coefficient_bound != max_coefficient_bound.clone().into() ||
            schema.count !=
                self.declaration
                    .family_shape()
                    .ok_or(mxx_dsl::DslError::Schema)?
                    .iter()
                    .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                    .ok_or(mxx_dsl::DslError::Schema)?
                    .into()
        {
            return Err(mxx_dsl::DslError::Schema);
        }
        match self.confidentiality {
            ArtifactConfidentiality::Public => context.public_output(self.artifact_name(), family),
            ArtifactConfidentiality::Private => {
                context.private_output(self.artifact_name(), family)
            }
        }
    }

    /// Validates the manifest declaration without loading any payload.
    pub fn validate_manifest(&self, manifest: &Manifest) -> Result<(), ReadoutCacheImportError> {
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION {
            return Err(ReadoutCacheImportError::DescriptorMismatch);
        }
        if manifest.production_id != self.production_id {
            return Err(ReadoutCacheImportError::ProductionMismatch);
        }
        let actual = manifest
            .artifacts
            .get(self.artifact_name())
            .ok_or(ReadoutCacheImportError::MissingArtifact)?;
        let expected = self.manifest_artifact();
        // Public family content hashes are an optional store integrity
        // commitment and are checked by FileArtifactStore; they do not alter
        // the typed preimage declaration.
        if actual.artifact_type != expected.artifact_type ||
            actual.family_count != expected.family_count ||
            actual.confidentiality != expected.confidentiality ||
            actual.layout != expected.layout
        {
            return Err(ReadoutCacheImportError::DescriptorMismatch);
        }
        Ok(())
    }

    /// Imports one requested family member as a typed lazy artifact input.
    /// The returned `Preimage` is relation-bearing and can be consumed by the
    /// cache-required BGG operations without decomposition or public-key
    /// recomputation.
    pub fn member(
        &self,
        ring: &Ring,
        manifest: &Manifest,
        index: usize,
    ) -> Result<Preimage, ReadoutCacheImportError> {
        self.validate_manifest(manifest)?;
        let descriptor = self.declaration.artifact_type();
        let ArtifactType::Preimage { matrix, max_coefficient_bound } = descriptor else {
            return Err(ReadoutCacheImportError::DescriptorMismatch);
        };
        let ring_type = ring.matrix_type((1, 1));
        let modulus = ring_type
            .modulus
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ReadoutCacheImportError::InvalidMatrixType)?;
        let ring_dimension = ring_type
            .ring_dimension
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ReadoutCacheImportError::InvalidMatrixType)?
            .to_usize()
            .ok_or(ReadoutCacheImportError::InvalidMatrixType)?;
        if modulus != matrix.modulus || ring_dimension != matrix.ring_dimension {
            return Err(ReadoutCacheImportError::InvalidMatrixType);
        }
        let family_shape = self.declaration.family_shape();
        let count = family_shape
            .map(|shape| {
                shape
                    .iter()
                    .try_fold(1usize, |product, extent| product.checked_mul(*extent))
                    .ok_or(ReadoutCacheImportError::InvalidFamilyShape)
            })
            .transpose()?
            .unwrap_or(1);
        if index >= count {
            return Err(ReadoutCacheImportError::MemberOutOfRange);
        }
        // The ring is checked by the DSL/runtime graph validation against the
        // manifest's concrete ArtifactType.  Keep this construction purely
        // typed and lazy: no store load occurs here.
        let _ = matrix;
        let Some(_) = family_shape else {
            return Ok(ring.preimage_artifact_input(
                self.production_id.clone(),
                self.artifact_name().to_owned(),
                (matrix.rows, matrix.columns),
                max_coefficient_bound.clone(),
                self.confidentiality,
            ));
        };
        Ok(ring
            .preimage_family_artifact_input(
                self.production_id.clone(),
                self.artifact_name().to_owned(),
                count,
                (matrix.rows, matrix.columns),
                max_coefficient_bound.clone(),
                self.confidentiality,
            )
            .at(index))
    }
}

#[cfg(test)]
mod readout_cache_tests {
    use super::*;

    #[test]
    fn cache_key_is_stable_and_content_separating() {
        let key = DecompositionCacheKey {
            layout: ExpLayout { stride_log2: 1, width_log2: 4 },
            readout_identity: ValueReadoutIdentity::from_bytes([3; 32]),
            setup_identity: [1; 32],
            program_identity: [2; 32],
            gate_or_readout_location: "readout/0".to_owned(),
            family_coordinate: vec![3, 4],
            operand_role: DecompositionOperandRole::ReadoutNormalization,
        };
        let mut other = key.clone();
        other.family_coordinate = vec![3, 5];
        assert_ne!(key, other);
        let mut other_layout = key.clone();
        other_layout.layout.width_log2 = 5;
        assert_ne!(key, other_layout);
        let artifact = DecompositionCacheArtifact::preimage(
            key.clone(),
            ConcreteMatrixType::scalar(BigInt::from(97), 32),
            BigInt::from(7),
            Some(vec![8, 9]),
        );
        assert!(artifact.is_preimage());
        assert!(matches!(artifact.artifact_type(), ArtifactType::Preimage { .. }));
        assert!(artifact.artifact_name().contains("exponent-lut/decomposition/"));
        let manifest = artifact.manifest_artifact(ArtifactConfidentiality::Public);
        assert_eq!(manifest.layout.as_deref(), None);
        let roundtrip: DecompositionCacheArtifact =
            serde_json::from_slice(&serde_json::to_vec(&artifact).unwrap()).unwrap();
        assert_eq!(roundtrip, artifact);

        let different_bound = DecompositionCacheArtifact::preimage(
            key.clone(),
            ConcreteMatrixType::scalar(BigInt::from(97), 32),
            BigInt::from(8),
            Some(vec![8, 9]),
        );
        assert_ne!(artifact.artifact_name(), different_bound.artifact_name());

        let different_family = DecompositionCacheArtifact::preimage(
            key,
            ConcreteMatrixType::scalar(BigInt::from(97), 32),
            BigInt::from(7),
            Some(vec![8, 10]),
        );
        assert_ne!(artifact.artifact_name(), different_family.artifact_name());
    }

    #[test]
    fn family_import_validates_manifest_and_keeps_member_lazy() {
        use mxx_dsl::DslContext;
        use mxx_ir_core::{ParamEnv, node::NodeKind};
        use mxx_runtime::{ArtifactKey, ArtifactPayload, ArtifactStore};

        let key = DecompositionCacheKey {
            layout: ExpLayout { stride_log2: 0, width_log2: 2 },
            readout_identity: ValueReadoutIdentity::from_bytes([7; 32]),
            setup_identity: [1; 32],
            program_identity: [2; 32],
            gate_or_readout_location: "readout/0".to_owned(),
            family_coordinate: vec![0],
            operand_role: DecompositionOperandRole::ReadoutAutomorphism,
        };
        let declaration = DecompositionCacheArtifact::preimage(
            key,
            ConcreteMatrixType {
                modulus: BigInt::from(97),
                ring_dimension: 8,
                rows: 4,
                columns: 1,
            },
            BigInt::from(7),
            Some(vec![3]),
        );
        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([3; 32]),
            execution_nonce: [4; 32],
        };
        let family = ReadoutCacheFamily::new(
            production_id.clone(),
            declaration,
            ArtifactConfidentiality::Private,
        )
        .unwrap();
        let manifest = family.manifest();
        let root = std::env::temp_dir().join(format!(
            "mxx-exponent-lut-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let mut store = mxx_runtime::FileArtifactStore::new(&root).unwrap();
        family.store_manifest(&mut store).unwrap();
        assert_eq!(family.load_manifest(&mut store).unwrap(), manifest);
        let ring = Ring::new(97, 8);
        let family_wire = ring.preimage_family_artifact_input(
            production_id.clone(),
            family.artifact_name().to_owned(),
            3usize,
            (4, 1),
            7,
            ArtifactConfidentiality::Private,
        );
        let producer_graph = family
            .add_preimage_family_output(DslContext::new("readout-cache-producer"), family_wire)
            .unwrap()
            .build()
            .unwrap();
        assert!(producer_graph.graph.outputs().contains_key(family.artifact_name()));
        let mut payload = Vec::new();
        payload.extend_from_slice(b"SMR1");
        payload.push(1);
        payload.extend_from_slice(&(4u64).to_le_bytes());
        payload.extend_from_slice(&(1u64).to_le_bytes());
        payload.extend_from_slice(&(8u64).to_le_bytes());
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.push(7);
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.extend_from_slice(&(32u64).to_le_bytes());
        payload.extend(std::iter::repeat(0u8).take(64));
        for index in 0..3 {
            store
                .store(
                    ArtifactKey {
                        production: production_id.clone(),
                        name: family.artifact_name().to_owned(),
                        index: Some(index),
                    },
                    family.declaration().artifact_type(),
                    ArtifactConfidentiality::Private,
                    None,
                    ArtifactPayload::SmallMatrix(payload.clone()),
                )
                .unwrap();
        }
        let member = family.member(&ring, &manifest, 2).unwrap();
        assert_eq!(member.matrix_type().rows, 4.into());
        assert_eq!(member.matrix_type().columns, 1.into());
        let selected_key = ArtifactKey {
            production: production_id.clone(),
            name: family.artifact_name().to_owned(),
            index: Some(2),
        };
        assert_eq!(
            store.load(&selected_key, &family.manifest_artifact()).unwrap(),
            ArtifactPayload::SmallMatrix(payload)
        );
        assert_eq!(store.load_count(&selected_key), 1);
        assert_eq!(
            store.load_count(&ArtifactKey {
                production: production_id.clone(),
                name: family.artifact_name().to_owned(),
                index: Some(0),
            }),
            0
        );
        let graph_nodes = producer_graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .collect::<Vec<_>>();
        assert!(graph_nodes.iter().any(|node| matches!(node.kind(), NodeKind::Input { .. })));
        let validated = producer_graph
            .validate_with_manifests(
                &ParamEnv::default(),
                &std::collections::BTreeMap::from([(production_id.clone(), manifest.clone())]),
            )
            .unwrap();
        let exported =
            mxx_ir_core::artifact::export_validated_manifest(production_id, &validated).unwrap();
        assert_eq!(
            exported.artifacts.get(family.artifact_name()),
            Some(&family.manifest_artifact())
        );
        assert!(matches!(
            family.member(&Ring::new(101, 8), &manifest, 0),
            Err(ReadoutCacheImportError::InvalidMatrixType)
        ));
        assert!(matches!(
            family.member(&ring, &manifest, 3),
            Err(ReadoutCacheImportError::MemberOutOfRange)
        ));
        let mut mismatched = manifest.clone();
        mismatched.production_id.execution_nonce = [9; 32];
        assert_eq!(
            family.validate_manifest(&mismatched),
            Err(ReadoutCacheImportError::ProductionMismatch)
        );
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_prepared_readout_round_trips_through_file_store() {
        use mxx_dsl::{DslContext, GraphValueSchema};
        use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
        use mxx_runtime::{
            ArtifactStore, RuntimeValue, backend::poly::cpu_backend, execute,
            transcript::SamplingMode,
        };

        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let output_count = 2usize;
        let parameters = DCRTPolyParams::new(n, 1, 17, 2, None, None);
        let layout = mxx_bgg::BggSamplerLayout {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: (n as usize).into(),
            secret_dimension: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: 4.into(),
        };
        let exponent = ExpLayout { stride_log2: n.trailing_zeros() - 1, width_log2: 1 };
        let ring = layout.ring();
        let scheme = BggScheme::new(
            mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: layout.gadget_base.clone(),
                digit_count: layout.digit_count.into(),
            },
            exponent,
        );
        let readout = ValueReadout::from_canonical_values(exponent, vec![0, 1]).unwrap();
        let sigmas = readout.trace_plan(n as usize).unwrap().non_identity_helpers().to_vec();
        let shape = (layout.secret_dimension, layout.public_key_columns());
        let inputs = ProgramValue::Family(
            mxx_dsl::parallel(output_count, |_| {
                Ok(BggPublicKeyWire {
                    matrix: ring.uniform_residue(shape),
                    reveal_plaintext: false,
                })
            })
            .unwrap(),
        );
        let switches =
            mxx_dsl::parallel(sigmas.len(), |_| Ok(ring.uniform_residue(shape))).unwrap();
        let prepared = scheme
            .prepare_readout(
                &readout,
                &inputs,
                &(
                    Arc::new(sigmas.clone()),
                    switches,
                    ring.gadget(
                        layout.secret_dimension,
                        layout.gadget_base.clone(),
                        layout.digit_count,
                    ),
                ),
            )
            .unwrap();
        let schema =
            BggPreparedReadout::artifact_schema(&layout, 2, layout.secret_dimension, output_count)
                .unwrap();
        assert_eq!(prepared.schema(), schema);
        assert_eq!(schema.wire_types().len(), 4);
        let schema: <BggPreparedReadout as GraphValue>::Schema =
            serde_json::from_slice(&serde_json::to_vec(&schema).unwrap()).unwrap();
        let matrix_outputs = |mut context: DslContext, value: &BggPreparedReadout| {
            for (name, family, count) in [
                ("target", &value.target_decompositions, output_count),
                ("switch", &value.helper_switch_decompositions, sigmas.len()),
                ("public", &value.helper_public_decompositions, output_count * sigmas.len()),
            ] {
                for index in 0..count {
                    let preimage = family.at(index);
                    let identity = ring.identity(preimage.matrix_type().rows.clone());
                    context = context
                        .output(format!("{name}-{index}"), preimage.mul_small_rhs(identity))
                        .unwrap();
                }
            }
            let normalization = value.normalization_decomposition.clone();
            let identity = ring.identity(normalization.matrix_type().rows.clone());
            context.output("normalization", normalization.mul_small_rhs(identity)).unwrap()
        };
        let producer = matrix_outputs(
            DslContext::new("prepared-readout-producer")
                .public_output("readout", prepared.clone())
                .unwrap(),
            &prepared,
        )
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
        std::fs::create_dir_all("test_data/bgg_prepared_readout_artifacts").unwrap();
        let directory = tempfile::tempdir_in("test_data/bgg_prepared_readout_artifacts").unwrap();
        let mut store = mxx_runtime::FileArtifactStore::new(directory.path()).unwrap();
        let produced = execute(
            &producer,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.clone().unwrap();
        drop(store);
        let mut store = mxx_runtime::FileArtifactStore::open(directory.path()).unwrap();
        let manifests =
            BTreeMap::from([(production.clone(), store.load_manifest(&production).unwrap())]);
        let imported = GraphValueSchema::artifact_input(
            &schema,
            production,
            "readout",
            ArtifactConfidentiality::Public,
        )
        .unwrap();
        let consumer = matrix_outputs(DslContext::new("prepared-readout-consumer"), &imported)
            .build()
            .unwrap()
            .validate_with_manifests(&ParamEnv::default(), &manifests)
            .unwrap();
        assert!(!consumer.source.scopes().values().flat_map(|scope| scope.nodes()).any(
            |node| matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
        ));
        let reopened = execute(
            &consumer,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        for (name, value) in &reopened.outputs {
            let RuntimeValue::Matrix(actual) = value else { panic!("matrix output") };
            let RuntimeValue::Matrix(expected) = &produced.outputs[name] else {
                panic!("matrix output")
            };
            assert_eq!(actual.as_ref(), expected.as_ref(), "{name}");
        }
    }

    #[test]
    fn scalar_preimage_round_trips_through_file_store() {
        use mxx_runtime::{ArtifactKey, ArtifactPayload, ArtifactStore};

        let key = DecompositionCacheKey {
            layout: ExpLayout { stride_log2: 0, width_log2: 1 },
            readout_identity: ValueReadoutIdentity::from_bytes([11; 32]),
            setup_identity: [12; 32],
            program_identity: [13; 32],
            gate_or_readout_location: "gate/0".to_owned(),
            family_coordinate: Vec::new(),
            operand_role: DecompositionOperandRole::LutHelperRhs,
        };
        let declaration = DecompositionCacheArtifact::preimage(
            key,
            ConcreteMatrixType { modulus: 97.into(), ring_dimension: 8, rows: 4, columns: 1 },
            7.into(),
            None,
        );
        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([14; 32]),
            execution_nonce: [15; 32],
        };
        let family = ReadoutCacheFamily::new(
            production_id.clone(),
            declaration,
            ArtifactConfidentiality::Public,
        )
        .unwrap();
        let root = std::env::temp_dir().join(format!(
            "mxx-exponent-lut-scalar-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let mut store = mxx_runtime::FileArtifactStore::new(&root).unwrap();
        family.store_manifest(&mut store).unwrap();
        let mut payload = Vec::new();
        payload.extend_from_slice(b"SMR1");
        payload.push(1);
        payload.extend_from_slice(&(4u64).to_le_bytes());
        payload.extend_from_slice(&(1u64).to_le_bytes());
        payload.extend_from_slice(&(8u64).to_le_bytes());
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.push(7);
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.extend_from_slice(&(32u64).to_le_bytes());
        payload.extend(std::iter::repeat(0u8).take(64));
        let artifact_key = ArtifactKey {
            production: production_id.clone(),
            name: family.artifact_name().to_owned(),
            index: None,
        };
        store
            .store(
                artifact_key.clone(),
                family.declaration().artifact_type(),
                ArtifactConfidentiality::Public,
                None,
                ArtifactPayload::SmallMatrix(payload.clone()),
            )
            .unwrap();
        assert_eq!(
            store.load(&artifact_key, &family.manifest_artifact()).unwrap(),
            ArtifactPayload::SmallMatrix(payload)
        );
        assert_eq!(store.load_count(&artifact_key), 1);
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }
}

impl<F: mxx_dsl::GraphValue, P: mxx_dsl::GraphValue> mxx_dsl::GraphValue
    for BggPreparedReadout<F, P>
{
    type Schema = BggPreparedReadout<F::Schema, P::Schema>;

    fn flatten(&self) -> Vec<mxx_ir_core::ValueHandle> {
        let mut values = self.target_decompositions.flatten();
        values.extend(self.helper_switch_decompositions.flatten());
        values.extend(self.helper_public_decompositions.flatten());
        values.extend(self.normalization_decomposition.flatten());
        values
    }

    fn schema(&self) -> Self::Schema {
        BggPreparedReadout {
            target_decompositions: self.target_decompositions.schema(),
            target_schema: self.target_schema.clone(),
            helper_sigmas: self.helper_sigmas.clone(),
            helper_switch_decompositions: self.helper_switch_decompositions.schema(),
            helper_public_decompositions: self.helper_public_decompositions.schema(),
            normalization_decomposition: self.normalization_decomposition.schema(),
            normalization_schema: self.normalization_schema.clone(),
            output_count: self.output_count,
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[mxx_ir_core::ValueHandle],
    ) -> Result<Self, mxx_dsl::DslError> {
        let (target, switch, public, normalization) = <(F, F, F, P)>::from_values(
            &(
                schema.target_decompositions.clone(),
                schema.helper_switch_decompositions.clone(),
                schema.helper_public_decompositions.clone(),
                schema.normalization_decomposition.clone(),
            ),
            values,
        )?;
        Ok(Self {
            target_decompositions: target,
            target_schema: schema.target_schema.clone(),
            helper_sigmas: schema.helper_sigmas.clone(),
            helper_switch_decompositions: switch,
            helper_public_decompositions: public,
            normalization_decomposition: normalization,
            normalization_schema: schema.normalization_schema.clone(),
            output_count: schema.output_count,
        })
    }
}

impl<F: mxx_dsl::GraphValueSchema, P: mxx_dsl::GraphValueSchema> mxx_dsl::GraphValueSchema
    for BggPreparedReadout<F, P>
{
    type Value = BggPreparedReadout<F::Value, P::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        BggPreparedReadout {
            target_decompositions: self.target_decompositions.placeholders_from(next),
            target_schema: self.target_schema.clone(),
            helper_sigmas: self.helper_sigmas.clone(),
            helper_switch_decompositions: self.helper_switch_decompositions.placeholders_from(next),
            helper_public_decompositions: self.helper_public_decompositions.placeholders_from(next),
            normalization_decomposition: self.normalization_decomposition.placeholders_from(next),
            normalization_schema: self.normalization_schema.clone(),
            output_count: self.output_count,
        }
    }

    fn wire_types(&self) -> Vec<mxx_ir_core::types::WireType> {
        let mut types = self.target_decompositions.wire_types();
        types.extend(self.helper_switch_decompositions.wire_types());
        types.extend(self.helper_public_decompositions.wire_types());
        types.extend(self.normalization_decomposition.wire_types());
        types
    }
}

impl BggPreparedReadout {
    /// Computes the exact readout artifact shape from setup geometry without
    /// creating graph nodes or requiring public matrices.
    pub(crate) fn artifact_schema(
        layout: &mxx_bgg::BggSamplerLayout,
        native_width: usize,
        component_count: usize,
        output_count: usize,
    ) -> Result<
        BggPreparedReadout<mxx_dsl::FamilyType<mxx_dsl::PreimageType>, mxx_dsl::PreimageType>,
        BggReadoutError,
    > {
        let ring_dimension = layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
        if native_width == 0 ||
            !native_width.is_power_of_two() ||
            ring_dimension == 0 ||
            !ring_dimension.is_power_of_two() ||
            native_width > ring_dimension ||
            output_count == 0
        {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        let exponent_layout = ExpLayout {
            stride_log2: ring_dimension.trailing_zeros() - native_width.trailing_zeros(),
            width_log2: native_width.trailing_zeros(),
        };
        let helper_sigmas = exponent_layout
            .automorphism_indices(ring_dimension)
            .map_err(ValueReadoutError::from)?
            .into_iter()
            .filter(|sigma| *sigma != 1)
            .collect::<Vec<_>>();
        let gadget_base = layout
            .gadget_base
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let bound = &gadget_base / BigInt::from(2u8);
        let metadata = ReadoutPreimageSchema::new(gadget_base, layout.digit_count, bound.clone())?;
        let preimage = mxx_dsl::PreimageType {
            matrix: layout
                .ring()
                .matrix_type((component_count * layout.digit_count, layout.public_key_columns())),
            max_coefficient_bound: IntExpr::RoundDiv(
                Box::new(layout.gadget_base.clone()),
                Box::new(IntExpr::constant(2)),
            )
            .canonicalize(),
        };
        let family = |count| mxx_dsl::FamilyType {
            element: preimage.clone(),
            count: IntExpr::constant(count),
            shared: vec![false],
        };
        Ok(BggPreparedReadout {
            target_decompositions: family(output_count),
            target_schema: metadata.clone(),
            helper_switch_decompositions: family(helper_sigmas.len()),
            helper_public_decompositions: family(output_count * helper_sigmas.len()),
            helper_sigmas,
            normalization_decomposition: preimage,
            normalization_schema: metadata,
            output_count,
        })
    }

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
    ) -> Result<Self, BggReadoutError> {
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
            return Err(ValueReadoutError::InvalidTraceSet.into());
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
}

impl BggRefreshParameters {
    /// Validates every exported artifact's canonical name, shape, and
    /// confidentiality. Private values may never carry a public hash.
    /// Returns the validated decoder bound for typed witness import.
    pub(crate) fn validate_refresh_manifest<B, D: mxx_dsl::GraphValueSchema>(
        &self,
        declaration: &crate::noise_refresh::setup::RefreshPreprocessingDeclaration<B, D>,
        manifest: &mxx_ir_core::artifact::Manifest,
    ) -> Result<(), RefreshSetupError> {
        self.validate()?;
        self.validate_prf(&declaration.prf_contract)?;
        if declaration.prf_contract.program_id() ==
            crate::program::ExponentLutProgramId::from_digest([0; 32])
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let decoder_preimage_bound = self.resolve_decoder_preimage_bound()?;
        let identity = setup_identity_digest(
            self,
            &declaration.prf_contract,
            declaration.pbc_layout_id,
            &decoder_preimage_bound,
            declaration.readout_identity,
        );
        if declaration.identity != identity ||
            declaration.names !=
                crate::noise_refresh::setup::RefreshPreprocessingArtifactNames::new(
                    &identity,
                    self.refresh.crt_plaintext_moduli.len(),
                    crate::noise_refresh::refresh::group_refresh_levels(
                        self.switch_moduli.iter().cloned(),
                    )
                    .len(),
                    declaration.names.selector_gsw.clone(),
                ) ||
            manifest.ir_version != mxx_ir_core::encoding::IR_VERSION
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        // The PBC boundary binds selector dimensions separately. Preserve
        // its nonempty matrix-family contract in the BGG adapter.
        let selector = manifest
            .artifacts
            .get(&declaration.names.selector_gsw)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        if !matches!(selector.family_count, Some(count) if count != 0) ||
            !matches!(selector.artifact_type, mxx_ir_core::artifact::ArtifactType::Matrix(_))
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let mut fields =
            self.refresh_artifact_types(&declaration.names, &decoder_preimage_bound)?;
        fields.push((
            declaration.names.selector_gsw.clone(),
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
            selector.artifact_type.clone(),
            selector.family_count,
        ));
        crate::noise_refresh::setup::validate_manifest(
            manifest,
            &declaration.names.readout,
            &declaration.prepared_prf,
            fields,
            self.refresh_readout_schema(
                self.transcript_topology()
                    .joint_event_count()
                    .map_err(|_| RefreshSetupError::InvalidManifest)?,
            )?,
        )
    }

    /// Builds compiler metadata from the authoritative resolved setup.
    pub(crate) fn public_key_compiler(&self) -> mxx_bgg::BggPublicKeyCompiler {
        mxx_bgg::BggPublicKeyCompiler {
            ring: self.layout.ring(),
            base: self.layout.gadget_base.clone(),
            digit_count: self.layout.digit_count.into(),
        }
    }

    /// Backend field schema for the setup's prepared terminal readout.
    /// The application supplies its logical PRF output count.
    pub(crate) fn refresh_readout_schema(
        &self,
        output_count: usize,
    ) -> Result<<BggPreparedReadout as GraphValue>::Schema, RefreshSetupError> {
        BggPreparedReadout::artifact_schema(
            &self.layout,
            self.lut_width,
            self.component_count,
            output_count,
        )
        .map_err(|error| RefreshSetupError::Backend(Box::new(error)))
    }

    /// Schema of deterministic gadget decompositions persisted for refresh.
    /// The bound is floor(beta / 2), independent of Gaussian helper cutoffs.
    pub(crate) fn refresh_cache_schema(&self) -> Result<mxx_dsl::PreimageType, RefreshSetupError> {
        let rows = self
            .component_count
            .checked_mul(self.layout.digit_count)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let base = self
            .layout
            .gadget_base
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        Ok(mxx_dsl::PreimageType {
            matrix: self.layout.ring().matrix_type((rows, self.layout.public_key_columns())),
            max_coefficient_bound: (base / BigInt::from(2u8)).into(),
        })
    }

    /// Concrete fields of the witness-based refresh setup. Shared transport
    /// validates these declarations without reconstructing gadget dimensions.
    pub(crate) fn refresh_artifact_types(
        &self,
        names: &crate::noise_refresh::setup::RefreshPreprocessingArtifactNames,
        decoder_bound: &IntExpr,
    ) -> Result<
        Vec<(
            String,
            mxx_ir_core::artifact::ArtifactConfidentiality,
            mxx_ir_core::artifact::ArtifactType,
            Option<usize>,
        )>,
        RefreshSetupError,
    > {
        use mxx_dsl::GraphValueSchema;
        use mxx_ir_core::{
            artifact::{
                ArtifactConfidentiality::{Private, Public},
                ArtifactType,
            },
            types::ConcreteMatrixType,
        };
        let env = ParamEnv::default();
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&env)
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let modulus =
            self.layout.modulus.evaluate(&env).map_err(|_| RefreshSetupError::InvalidManifest)?;
        let bound = decoder_bound.evaluate(&env).map_err(|_| RefreshSetupError::InvalidManifest)?;
        let cols = self.layout.public_key_columns();
        let matrix = |modulus: BigInt, rows, columns| ConcreteMatrixType {
            modulus,
            ring_dimension,
            rows,
            columns,
        };
        let mut fields = vec![
            (
                names.setup_secret_t.clone(),
                Private,
                ArtifactType::Matrix(matrix(modulus.clone(), 1, self.layout.secret_dimension)),
                None,
            ),
            (
                names.state_public_matrix.clone(),
                Public,
                ArtifactType::Matrix(matrix(modulus.clone(), self.component_count, cols)),
                None,
            ),
            (
                names.a_prime.clone(),
                Public,
                ArtifactType::Matrix(matrix(modulus, self.component_count, cols)),
                None,
            ),
        ];
        let groups =
            crate::noise_refresh::refresh::group_refresh_levels(self.switch_moduli.iter().cloned());
        let slots = self.refresh.crt_plaintext_moduli.len();
        if names.public_matrix_b.len() != groups.len() ||
            names.refresh_targets.len() != slots ||
            names.preimages.len() != slots
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        for (group, (modulus, members)) in groups.iter().enumerate() {
            let columns = self.component_count * (self.decoder_digit_count(members[0])? + 2);
            fields.push((
                names.public_matrix_b[group].clone(),
                Public,
                ArtifactType::Matrix(matrix(modulus.clone().into(), self.component_count, columns)),
                None,
            ));
        }
        for slot in 0..slots {
            let modulus: BigInt = self.switch_moduli[slot].clone().into();
            let rows = self.component_count * (self.decoder_digit_count(slot)? + 2);
            fields.push((
                names.refresh_targets[slot].clone(),
                Public,
                ArtifactType::Matrix(matrix(modulus.clone(), self.component_count, cols)),
                None,
            ));
            fields.push((
                names.preimages[slot].clone(),
                Private,
                ArtifactType::Preimage {
                    matrix: matrix(modulus, rows, cols),
                    max_coefficient_bound: bound.clone(),
                },
                None,
            ));
        }
        let cache = mxx_ir_core::validate::concretize_wire_type(
            &self.refresh_cache_schema()?.wire_types()[0],
            &env,
            &mxx_ir_core::graph::FrozenGraphScopeId::Root,
            mxx_ir_core::types::NodeId(0),
        )
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let cache =
            ArtifactType::from_wire_type(&cache).ok_or(RefreshSetupError::InvalidManifest)?;
        for name in &names.state_scaling {
            fields.push((name.clone(), Public, cache.clone(), None));
        }
        for (name, count) in [
            (
                &names.mask_route_rhs_decompositions,
                self.refresh_route_family_count(&self.mask_base_p_digit_counts)?,
            ),
            (
                &names.fresh_route_rhs_decompositions,
                self.refresh_route_family_count(&vec![self.fresh_error_base_p_digit_count; slots])?,
            ),
        ] {
            fields.push((name.clone(), Public, cache.clone(), Some(count)));
        }
        Ok(fields)
    }

    /// Prepare state scaling and weighted route decompositions from public setup.
    /// Public projection and online routing reuse these same typed witnesses.
    pub(crate) fn prepare_refresh_route_witnesses(
        &self,
        readout_type: &mxx_ir_core::types::MatrixType,
    ) -> Result<(Vec<Preimage>, Family<Preimage>, Family<Preimage>), RefreshSetupError> {
        let p = self;
        let ring = p.layout.ring();
        let mut scales = Vec::new();
        for slot in 0..p.refresh.crt_plaintext_moduli.len() {
            // State scale mu_t = q/q_t for CRT slot t.
            let scale = ring.polynomial([p.refresh.scale_expression(slot)?]);
            scales.push(scale);
        }
        let decomposition_base = p.layout.gadget_base.clone();
        let decomposition_digit_count = p.layout.digit_count;
        let state_gadget =
            ring.gadget(p.component_count, decomposition_base.clone(), decomposition_digit_count);
        let state_scaling = scales
            .iter()
            .map(|scale| {
                (scale.clone() * state_gadget.clone())
                    .decompose(decomposition_base.clone(), decomposition_digit_count)
            })
            .collect::<Vec<_>>();
        let fresh_digits = vec![p.fresh_error_base_p_digit_count; p.switch_moduli.len()];
        // Corrected scalars already contain the digit powers and CRT weights;
        // route construction therefore uses base one with this single scale.
        let mask_route_rhs_decompositions = crate::bgg::route_preimage_family(
            readout_type,
            1,
            p.prf_component_count(),
            p.coefficient_count,
            &p.mask_base_p_digit_counts,
            Some(p.corrected_route_scalar_family(&ring, false)?),
            decomposition_base.clone(),
            IntExpr::constant(decomposition_digit_count),
        )?;
        let fresh_route_rhs_decompositions = crate::bgg::route_preimage_family(
            readout_type,
            1,
            p.prf_component_count(),
            p.coefficient_count,
            &fresh_digits,
            Some(p.corrected_route_scalar_family(&ring, true)?),
            decomposition_base.clone(),
            IntExpr::constant(decomposition_digit_count),
        )?;
        Ok((state_scaling, mask_route_rhs_decompositions, fresh_route_rhs_decompositions))
    }

    fn corrected_route_scalar_family(
        &self,
        ring: &mxx_dsl::Ring,
        fresh: bool,
    ) -> Result<Family<Mat>, RefreshSetupError> {
        let p = self;
        let (values, _) = super::noise::corrected_route_scalars(p, fresh)?;
        let digits = values.iter().map(Vec::len).collect::<Vec<_>>();
        let mut offsets = Vec::new();
        let mut scalars = Vec::new();
        for slot in values {
            offsets.push(scalars.len());
            scalars
                .extend(slot.into_iter().map(|value| ring.polynomial([IntExpr::constant(value)])));
        }
        let count = p.refresh_route_family_count(&digits)?;
        let offsets = Family::pack(offsets.into_iter().map(Int::constant).collect())?;
        let scalars = Family::pack(scalars)?;
        Ok(mxx_dsl::parallel(count, |index| {
            let (slot, digit, _, _) = crate::noise_refresh::refresh::route_coordinates(
                index.expression()?,
                p.prf_component_count(),
                p.coefficient_count,
                &digits,
            )
            .map_err(|_| mxx_dsl::DslError::Schema)?;
            let offset = offsets.at(Int::evaluate(slot));
            Ok(scalars.at(offset.add(Int::evaluate(digit))))
        })?)
    }

    pub(crate) fn refresh_route_family_count(
        &self,
        digit_counts: &[usize],
    ) -> Result<usize, RefreshSetupError> {
        let p = self;
        if digit_counts.len() != p.refresh.crt_plaintext_moduli.len() {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let digits = digit_counts.iter().try_fold(0usize, |sum, count| {
            sum.checked_add(*count).ok_or(RefreshSetupError::InvalidManifest)
        })?;
        digits
            .checked_mul(p.prf_component_count())
            .and_then(|value| value.checked_mul(p.coefficient_count))
            .ok_or(RefreshSetupError::InvalidManifest)
    }

    /// Samples the preprocessing-fixed low-P targets and decoder families.
    pub(crate) fn sample_refresh_decoders(
        &self,
        scheme: &BggScheme,
        sum_family: Family<Mat>,
        a_prime: &Mat,
        decoder_preimage_bound: &IntExpr,
    ) -> Result<(Vec<Mat>, Vec<Mat>, Vec<mxx_dsl::Preimage>), RefreshSetupError> {
        let p = self;
        let mut groups = Vec::new();
        for (modulus, slots) in group_refresh_levels(p.switch_moduli.iter().cloned()) {
            let low_ring = p.switch_ring(slots[0])?;
            let modulus = IntExpr::constant(BigInt::from(modulus));
            let digits = p.decoder_digit_count(slots[0])?;
            let deltas = Family::pack(
                slots
                    .iter()
                    .map(|slot| {
                        low_ring.polynomial([IntExpr::Div(
                            Box::new(modulus.clone()),
                            Box::new(p.refresh.crt_plaintext_moduli[*slot].clone()),
                        )])
                    })
                    .collect(),
            )?;
            groups.push((
                modulus,
                slots,
                deltas,
                crate::bgg::BggRefreshAnchorParameters {
                    components: p.component_count,
                    gadget_base: p.layout.gadget_base.clone(),
                    digits,
                    sigma: p.decoder_sigma.clone(),
                    preimage_bound: decoder_preimage_bound.clone(),
                },
            ));
        }
        Ok(crate::noise_refresh::refresh::prepare_refresh_decoders(
            scheme,
            &groups,
            sum_family,
            a_prime.clone(),
        )?)
    }
}
