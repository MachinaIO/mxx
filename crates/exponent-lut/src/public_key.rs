//! Public-key projection of the setup-fixed Exponent-LUT evaluator.
//!
//! This path receives only public BGG matrices, fixed GSW ciphertexts, and
//! public helper matrices. It mirrors the private flat evaluator without
//! knowing encoding vectors, plaintexts, secrets, or selector schedules.
//!
//! If a private branch is `(v_jD_j,A_jD_j)`, this path retains `A_jD_j`:
//! public Fuse is `A -> A G^{-1}(C)`, and a public LUT sums
//! `M_j (A^{\sigma_j} G^{-1}(C_j))` in the same odd-automorphism order.

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    ExponentLutError,
    encoding::{
        FlatLutPublicMaskBank, balanced_sum_family, canonical_flat_mask_branch_tag,
        canonical_sigma, one_hot_indices_and_masks,
    },
    program::{
        ExponentLutMonomialFamily, ExponentLutProgram, FamilyRange, LutOutputForm, LutTable,
        ProgramBindings, ProgramFamilyRanges, ProgramInputId, ProgramLoweringBackend,
        ProgramWireId, RhsInputId, lower_program,
    },
    rhs::ExponentRhsPackage,
};
use mxx_bgg::BggPublicKeyWire;
use mxx_dsl::{Bytes, DslError, Family, HashTag, Mat};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_traits::ToPrimitive;

/// Public portion of one setup-fixed flat LUT helper.
#[derive(Clone)]
pub struct FlatLutPublicHelper {
    sigma: usize,
    switch: ExponentRhsPackage,
    mask_bank: Arc<FlatLutPublicMaskBank>,
    mask_index: usize,
}

/// Public helper branches bound to one concrete LUT table.
#[derive(Clone)]
pub struct FlatLutPublicHelperSet {
    table_commitment: [u8; 32],
    width: usize,
    helpers: Vec<FlatLutPublicHelper>,
}

impl FlatLutPublicHelperSet {
    pub fn new(
        table: &crate::program::LutTable,
        helpers: Vec<FlatLutPublicHelper>,
    ) -> Result<Self, ExponentLutError> {
        if table.values().len() != helpers.len() {
            return Err(ExponentLutError::InvalidLut);
        }
        Ok(Self { table_commitment: table.commitment(), width: helpers.len(), helpers })
    }
    pub(crate) fn resolve(
        &self,
        table: &crate::program::LutTable,
    ) -> Result<&[FlatLutPublicHelper], ExponentLutError> {
        if self.width != table.values().len() || self.table_commitment != table.commitment() {
            return Err(ExponentLutError::InvalidLut);
        }
        Ok(&self.helpers)
    }
    pub(crate) fn from_parts(
        table_commitment: [u8; 32],
        width: usize,
        helpers: Vec<FlatLutPublicHelper>,
    ) -> Result<Self, ExponentLutError> {
        if width == 0 || width != helpers.len() {
            return Err(ExponentLutError::InvalidLut);
        }
        Ok(Self { table_commitment, width, helpers })
    }
    pub(crate) fn as_slice(&self) -> &[FlatLutPublicHelper] {
        &self.helpers
    }
    pub(crate) fn table_commitment(&self) -> [u8; 32] {
        self.table_commitment
    }
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, FlatLutPublicHelper> {
        self.helpers.iter()
    }
}

impl FlatLutPublicHelper {
    pub fn new(sigma: usize, switch: ExponentRhsPackage, mask: Mat) -> Self {
        let mask_bank = Arc::new(FlatLutPublicMaskBank::single(sigma, mask));
        Self::with_mask_bank(sigma, switch, mask_bank).expect("single public mask branch")
    }
    pub fn with_mask_bank(
        sigma: usize,
        switch: ExponentRhsPackage,
        mask_bank: Arc<FlatLutPublicMaskBank>,
    ) -> Result<Self, ExponentLutError> {
        let mask_index = mask_bank.index_for_sigma(sigma).ok_or(ExponentLutError::InvalidLut)?;
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

/// Public helper registry. It is keyed by LUT identity at the application
/// boundary; this type only stores the ordered helper branch values.
#[derive(Clone, Default)]
pub struct FlatLutPublicHelperRegistry {
    helpers: BTreeMap<crate::program::LutId, FlatLutPublicHelperSet>,
}
pub type FlatLutPublicHelperMap = BTreeMap<crate::program::LutId, FlatLutPublicHelperSet>;
impl FlatLutPublicHelperRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(
        &mut self,
        lut: crate::program::LutId,
        helpers: FlatLutPublicHelperSet,
    ) -> Result<(), ExponentLutError> {
        if self.helpers.insert(lut, helpers).is_some() {
            Err(ExponentLutError::InvalidLut)
        } else {
            Ok(())
        }
    }
    pub fn get(&self, lut: crate::program::LutId) -> Option<&[FlatLutPublicHelper]> {
        self.helpers.get(&lut).map(FlatLutPublicHelperSet::as_slice)
    }
}

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
    pub fn sample_flat_mask_bank(
        &self,
        hash_key: Bytes,
        max_width: usize,
        tag: impl Into<HashTag>,
    ) -> Result<Arc<FlatLutPublicMaskBank>, ExponentLutPublicSamplingError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutPublicSamplingError::InvalidConfiguration(
                "ring dimension must be concrete",
            ))?;
        if max_width == 0 || !max_width.is_power_of_two() || max_width > n || n % max_width != 0 {
            return Err(ExponentLutPublicSamplingError::InvalidConfiguration(
                "mask-bank width must be a power of two dividing the ring dimension",
            ));
        }
        let root = tag.into();
        let sampler = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() };
        let mut branches = Vec::with_capacity(max_width);
        for j in 0..max_width {
            let sigma = canonical_sigma(j, n, max_width).ok_or(
                ExponentLutPublicSamplingError::InvalidConfiguration("invalid mask-bank sigma"),
            )?;
            let key = sampler
                .sample(hash_key.clone(), canonical_flat_mask_branch_tag(&root, sigma), &[false])
                .into_iter()
                .nth(1)
                .ok_or(ExponentLutPublicSamplingError::InvalidConfiguration(
                    "mask sample is empty",
                ))?;
            branches.push((sigma, key.matrix));
        }
        Ok(Arc::new(FlatLutPublicMaskBank::from_branches(branches)))
    }
}

#[derive(Clone)]
pub struct ExponentLutPublicKeyCompiler {
    pub public_key: mxx_bgg::BggPublicKeyCompiler,
}
impl ExponentLutPublicKeyCompiler {
    pub fn new(public_key: mxx_bgg::BggPublicKeyCompiler) -> Self {
        Self { public_key }
    }
    pub(crate) fn compile_program(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggPublicKeyWire>,
        rhs: &BTreeMap<RhsInputId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, PublicSelectorFamily>,
        values: &BTreeMap<crate::program::PublicValueFamilyId, ExponentLutMonomialFamily>,
        helpers: &FlatLutPublicHelperMap,
    ) -> Result<BTreeMap<ProgramWireId, BggPublicKeyWire>, ExponentLutError> {
        let mut ranges = ProgramFamilyRanges::new();
        for (id, family) in selectors {
            ranges.selector(
                *id,
                FamilyRange::full(family.count().clone())
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
            );
        }
        for (id, family) in values {
            ranges.public_values(
                *id,
                FamilyRange::full(family.count().clone())
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
            );
        }
        let bindings = ProgramBindings::new(inputs, rhs, selectors, values, helpers);
        lower_program(program, &bindings, &ranges, self)
    }
    pub(crate) fn compile_program_with_ranges(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggPublicKeyWire>,
        rhs: &BTreeMap<RhsInputId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, PublicSelectorFamily>,
        values: &BTreeMap<crate::program::PublicValueFamilyId, ExponentLutMonomialFamily>,
        ranges: &ProgramFamilyRanges,
        helpers: &FlatLutPublicHelperMap,
    ) -> Result<BTreeMap<ProgramWireId, BggPublicKeyWire>, ExponentLutError> {
        let bindings = ProgramBindings::new(inputs, rhs, selectors, values, helpers);
        lower_program(program, &bindings, ranges, self)
    }
    /// Setup-fixed Fuse is the public matrix product `A G^{-1}(C)`.
    pub fn fuse_public(
        &self,
        input: &Mat,
        rhs: &ExponentRhsPackage,
    ) -> Result<Mat, ExponentLutError> {
        // Public projection of Fuse: multiply `A` by the digit matrix of C.
        Ok(input.clone().mul_small_rhs(
            rhs.gsw_ciphertext()
                .clone()
                .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone()),
        ))
    }
    pub fn single_input_lut(
        &self,
        input: &Mat,
        table: &[usize],
        helpers: &[FlatLutPublicHelper],
    ) -> Result<Mat, ExponentLutError> {
        let width = table.len();
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let table = LutTable::unary(width, n, table.to_vec()).map_err(ExponentLutError::from)?;
        self.single_input_lut_table(input, &table, helpers)
    }

    pub(crate) fn single_input_lut_table(
        &self,
        input: &Mat,
        table: &LutTable,
        helpers: &[FlatLutPublicHelper],
    ) -> Result<Mat, ExponentLutError> {
        let width = table.values().len();
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        if width == 0 ||
            !width.is_power_of_two() ||
            width > n ||
            n % width != 0 ||
            helpers.len() != width ||
            table.values().iter().any(|value| *value >= n)
        {
            return Err(ExponentLutError::InvalidLut);
        }
        // The branch body computes `mask_j * (A^{\sigma_j} G^{-1}(C_j))`, the
        // public component corresponding to the private Fuse branch.
        let raw_publics = Family::pack(
            helpers
                .iter()
                .enumerate()
                .map(|(j, helper)| {
                    let expected =
                        canonical_sigma(j, n, width).ok_or(ExponentLutError::InvalidLut)?;
                    if helper.sigma() != expected {
                        return Err(ExponentLutError::InvalidLut);
                    }
                    Ok(input.clone().ring_automorphism(expected))
                })
                .collect::<Result<Vec<_>, ExponentLutError>>()?,
        )
        .map_err(|_| ExponentLutError::InvalidLut)?;
        let switches = Family::pack(
            helpers.iter().map(|helper| helper.switch().gsw_ciphertext().clone()).collect(),
        )
        .map_err(|_| ExponentLutError::InvalidLut)?;
        let masks = Family::pack(helpers.iter().map(|helper| helper.mask().clone()).collect())
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let branches =
            Family::try_parallel_zip_many_values(
                vec![raw_publics, switches, masks],
                move |_index, mut items| {
                    // The families were zipped as `[raw_publics, switches, masks]`;
                    // `pop` therefore yields the public mask, the concrete
                    // setup-fixed selector ciphertext `C_{sigma,L}`, then the
                    // automorphed public-key matrix `A^{\sigma}`.
                    let mask = items.pop().ok_or(DslError::Schema)?;
                    // `mask` is the public matrix from the mask-alignment helper
                    // encoding. Its private vector switches the mask secret from
                    // `sigma(s)` back to `s` after the automorphed fixed-C Fuse;
                    // this public closure computes only that helper's projection.
                    let switch = items.pop().ok_or(DslError::Schema)?;
                    // `switch` is the concrete setup-fixed GSW ciphertext
                    // `C_{sigma,L}`, wrapped below for fixed-RHS shape validation.
                    let raw = items.pop().ok_or(DslError::Schema)?;
                    // `raw = A^\sigma` is the automorphed public-key matrix; no
                    // payload or plaintext term is formed on this public path.
                    let rhs = ExponentRhsPackage::new(switch).map_err(|_| DslError::Schema)?;
                    // `fused = A^{\sigma} G^{-1}(C_{sigma,L})` is still public.
                    let fused = raw.mul_small_rhs(rhs.gsw_ciphertext().clone().decompose(
                        self.public_key.base.clone(),
                        self.public_key.digit_count.clone(),
                    ));
                    // `decomp = G^{-1}(fused)` has the gadget-digit shape needed
                    // by `mask`; `mask * decomp` is the helper's public projection
                    // that accompanies the private vector-side alignment.
                    Ok(mask.mul_small_rhs(fused.decompose(
                        self.public_key.base.clone(),
                        self.public_key.digit_count.clone(),
                    )))
                },
            )
            .map_err(|_| ExponentLutError::InvalidLut)?;
        balanced_sum_family(branches)
    }
    pub fn two_input_lut(
        &self,
        lhs: &Mat,
        rhs: &ExponentRhsPackage,
        lhs_width: usize,
        rhs_width: usize,
        table: &[usize],
        helpers: &[FlatLutPublicHelper],
    ) -> Result<Mat, ExponentLutError> {
        if table.len() != lhs_width.checked_mul(rhs_width).ok_or(ExponentLutError::InvalidLut)? {
            return Err(ExponentLutError::InvalidLut);
        }
        self.single_input_lut(&self.fuse_public(lhs, rhs)?, table, helpers)
    }

    fn two_input_lut_table(
        &self,
        lhs: &Mat,
        rhs: &ExponentRhsPackage,
        table: &LutTable,
        helpers: &[FlatLutPublicHelper],
    ) -> Result<Mat, ExponentLutError> {
        let rhs_width = table.rhs_width().ok_or(ExponentLutError::InvalidLut)?;
        if table.input_width() == 0 ||
            rhs_width == 0 ||
            table.output_form() != LutOutputForm::Monomial
        {
            return Err(ExponentLutError::InvalidLut);
        }
        let fused = self.fuse_public(lhs, rhs)?;
        self.single_input_lut_table(&fused, table, helpers)
    }
}

impl ProgramLoweringBackend for ExponentLutPublicKeyCompiler {
    type Wire = BggPublicKeyWire;
    type Rhs = ExponentRhsPackage;
    type SelectorFamily = PublicSelectorFamily;
    type PublicValueFamily = ExponentLutMonomialFamily;
    type Helper = FlatLutPublicHelper;
    type HelperSet = FlatLutPublicHelperSet;

    fn resolve_helpers<'a>(
        &self,
        helpers: &'a Self::HelperSet,
        table: &crate::program::LutTable,
    ) -> Result<&'a [Self::Helper], ExponentLutError> {
        helpers.resolve(table)
    }
    fn unary(
        &self,
        input: Self::Wire,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, ExponentLutError> {
        if table.rhs_width().is_some() {
            return Err(ExponentLutError::InvalidLut);
        }
        Ok(BggPublicKeyWire {
            matrix: self.single_input_lut_table(&input.matrix, table, helpers)?,
            reveal_plaintext: false,
        })
    }
    fn binary(
        &self,
        lhs: Self::Wire,
        rhs: &Self::Rhs,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, ExponentLutError> {
        Ok(BggPublicKeyWire {
            matrix: self.two_input_lut_table(&lhs.matrix, rhs, table, helpers)?,
            reveal_plaintext: false,
        })
    }
    fn one_hot_select(
        &self,
        input: Self::Wire,
        selectors: &Self::SelectorFamily,
        values: &Self::PublicValueFamily,
        selector_range: &FamilyRange,
        value_range: &FamilyRange,
    ) -> Result<Self::Wire, ExponentLutError> {
        // Public projection of one-hot selection is
        // `sum_i m_i * (A G^{-1}(C_i)) * v_i`: `m_i` is the public structural
        // active-lane mask (1 for a real or dummy active cell, 0 for padding
        // or an inactive lane), `v_i = X^{a_i}` is the public monomial made
        // from routed residue `a_i` (and `v_i = X^0 = 1` for a dummy), and
        // `C_i` is the setup-fixed public GSW ciphertext whose hidden
        // plaintext relation carries selector bit `z_i`. The factor
        // `A G^{-1}(C_i)` is public Fuse projection, multiplication by `v_i`
        // adds the routed exponent, and `m_i` suppresses padding; in the
        // underlying payload exactly one `z_i` is one, so the sum selects one
        // branch without reconstructing that bit here.
        // Here active/inactive is public structure, not secretly
        // selected/unselected: for a rectangular PBC layout, `m_i` is derived
        // solely from the accepted public layout and contains no support
        // coordinate, private schedule, or selected slot. The accepted
        // layout's seed/attempt may nevertheless have been conditioned on
        // scheduling success for the fixed support by the retry procedure. A
        // dummy stays active because an unmatched bucket may be privately
        // selected; hidden `z_i` exists only in the plaintext relation carried
        // by `C_i`.
        // Selector C and public values are gathered in lockstep; no private
        // selector bit is reconstructed on this path.
        if !values.has_provenance(crate::program::PBC_MONOMIAL_FAMILY_PROVENANCE) ||
            selector_range != value_range ||
            selectors.count() != values.count()
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let capacity = selector_range
            .capacity()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let count = mxx_dsl::Int::evaluate(selector_range.count().clone());
        let start = mxx_dsl::Int::evaluate(selector_range.start().clone());
        let (indices, masks) =
            one_hot_indices_and_masks(capacity, count, start, values.element_type().clone())
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let cs = selectors
            .gsw()
            .clone()
            .parallel_gather(indices.clone())
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let vals = values
            .as_family()
            .clone()
            .parallel_gather(indices)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let weighted = Family::try_parallel_zip_many_values(
            vec![cs, vals, masks],
            move |_index, mut items| {
                let mask = items.pop().ok_or(DslError::Schema)?;
                let value = items.pop().ok_or(DslError::Schema)?;
                let c = items.pop().ok_or(DslError::Schema)?;
                let rhs = ExponentRhsPackage::new(c).map_err(|_| DslError::Schema)?;
                Ok(self.fuse_public(&input.matrix, &rhs).map_err(|_| DslError::Schema)? *
                    (value * mask))
            },
        )
        .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        Ok(BggPublicKeyWire {
            matrix: balanced_sum_family(weighted)
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
            reveal_plaintext: false,
        })
    }
}

#[derive(Clone)]
pub struct PublicSelectorFamily {
    gsw: Family<Mat>,
}
impl PublicSelectorFamily {
    pub fn new(gsw: Family<Mat>) -> Result<Self, ExponentLutError> {
        if *gsw.count() == IntExpr::constant(0) {
            Err(ExponentLutError::InvalidSparseLwrBlock)
        } else {
            Ok(Self { gsw })
        }
    }
    pub(crate) fn gsw(&self) -> &Family<Mat> {
        &self.gsw
    }
    pub(crate) fn count(&self) -> &IntExpr {
        self.gsw.count()
    }
    pub(crate) fn flattened(&self) -> Vec<Family<Mat>> {
        vec![self.gsw.clone()]
    }
    pub(crate) fn from_flattened(mut values: Vec<Family<Mat>>) -> Result<Self, ExponentLutError> {
        if values.len() != 1 {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Self::new(values.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExponentLutPublicKeyCompiler, ExponentLutPublicKeySampler, FlatLutPublicHelper,
        FlatLutPublicHelperSet, PublicSelectorFamily,
    };
    use crate::{
        program::{
            ExponentLutMonomialFamily, FamilyRange, LutTable, PBC_MONOMIAL_FAMILY_PROVENANCE,
            ProgramLoweringBackend,
        },
        rhs::ExponentRhsPackage,
    };
    use mxx_bgg::BggPublicKeyCompiler;
    use mxx_dsl::{Family, Ring};

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
        let expected = input.mul_small_rhs(rhs.gsw_ciphertext().clone().decompose(4, 2));
        assert_eq!(output.matrix_type(), expected.matrix_type());
        assert!(matches!(
            output.value_handle().node().kind(),
            mxx_ir_core::node::NodeKind::MatrixMulSmallRhs
        ));
    }

    #[test]
    fn public_one_hot_selection_returns_the_balanced_sum_without_lut_helpers() {
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let input = super::BggPublicKeyWire {
            matrix: ring.input("public-selection-input", (2, 2)),
            reveal_plaintext: false,
        };
        let selectors = PublicSelectorFamily::new(
            Family::pack(vec![ring.zero((2, 2)), ring.zero((2, 2))]).unwrap(),
        )
        .unwrap();
        let values = ExponentLutMonomialFamily::from_trusted(
            Family::pack(vec![ring.polynomial([0.into()]), ring.polynomial([1.into()])]).unwrap(),
            &ring,
            PBC_MONOMIAL_FAMILY_PROVENANCE,
        )
        .unwrap();
        let range = FamilyRange::full(2).unwrap();

        let wrong_provenance = ExponentLutMonomialFamily::from_trusted(
            values.as_family().clone(),
            &ring,
            PBC_MONOMIAL_FAMILY_PROVENANCE + 1,
        )
        .unwrap();
        assert!(
            compiler
                .one_hot_select(input.clone(), &selectors, &wrong_provenance, &range, &range)
                .is_err()
        );
        let wrong_count = ExponentLutMonomialFamily::from_trusted(
            Family::pack(vec![ring.polynomial([0.into()])]).unwrap(),
            &ring,
            PBC_MONOMIAL_FAMILY_PROVENANCE,
        )
        .unwrap();
        assert!(
            compiler
                .one_hot_select(input.clone(), &selectors, &wrong_count, &range, &range)
                .is_err()
        );

        // The selection-only interface has no LUT/helper arguments. Its
        // result is already the balanced public projection.
        let output = compiler.one_hot_select(input, &selectors, &values, &range, &range).unwrap();
        assert!(!output.reveal_plaintext);
        assert_eq!(
            output.matrix.matrix_type(),
            ring.input("public-selection-type", (2, 2)).matrix_type()
        );
    }

    #[test]
    fn public_helper_set_rejects_a_different_lut_table() {
        let ring = Ring::new(97, 4);
        let rhs = ExponentRhsPackage::new(ring.zero((2, 2))).unwrap();
        let helper = FlatLutPublicHelper::new(1, rhs, ring.zero((2, 2)));
        let table = LutTable::unary(1, 2, vec![0]).unwrap();
        let other = LutTable::unary(1, 2, vec![1]).unwrap();
        let set = FlatLutPublicHelperSet::new(&table, vec![helper]).unwrap();
        assert!(set.resolve(&table).is_ok());
        assert!(set.resolve(&other).is_err());
    }
}
