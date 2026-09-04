//! Public-key projection of the setup-fixed Exponent-LUT evaluator.
//!
//! This path receives only public BGG matrices, fixed GSW ciphertexts, and
//! public helper matrices. It mirrors the private evaluator without
//! knowing encoding vectors, plaintexts, secrets, or selector schedules.
//!
//! If a private branch is `(v_jD_j,A_jD_j)`, this path retains `A_jD_j`:
//! public Fuse is `A -> A G^{-1}(C)`, and a public LUT sums
//! `M_j (A^{\sigma_j} G^{-1}(C_j))` in the same odd-automorphism order.

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    ExponentLutError,
    encoding::{LutPublicMaskBank, balanced_sum_family},
    program::{
        ExponentLutProgram, ExponentProgramLowering, FamilyRange, Gate, LutTable, ProgramBindings,
        ProgramFamilyBindings, ProgramGateInstance, ProgramInputId, ProgramLoweringTypes,
        ProgramStep, RhsId, WireId, lower_program,
    },
    rhs::ExponentRhsPackage,
};
use mxx_bgg::BggPublicKeyWire;
use mxx_dsl::{Bytes, Family, HashTag, Int, Mat, Parallel, ParallelZipTuple, Preimage};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

/// Kind of setup-fixed public decomposition required by an evaluator gate.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum ProgramCacheRole {
    K_rhs,
    LutSwitch,
    LutBranch,
}

/// Producer-owned location of one cached public decomposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum ProgramCacheLocation {
    Gate { step: usize },
    Lut { step: usize, branch: usize },
    Family { step: usize, body_gate: usize },
    FamilyLut { step: usize, body_gate: usize, branch: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ProgramCacheDeclaration {
    pub location: ProgramCacheLocation,
    pub role: ProgramCacheRole,
}

/// Preprocessing-produced gadget decompositions consumed by online encoding
/// evaluation.  Only generic typed preimages are stored; no encoding or
/// secret-dependent public matrix is materialized as an artifact.
#[derive(Clone, Default)]
pub struct ProgramCacheBindings {
    gate: BTreeMap<usize, Preimage>,
    family: BTreeMap<(usize, usize), Family<Preimage>>,
    lut: BTreeMap<usize, ProgramLutCache>,
    family_lut: BTreeMap<(usize, usize), ProgramLutCache>,
}

/// Label-major preprocessing cache for a batch of independent program
/// evaluations.  Every member is still a generic typed [`Preimage`]; this
/// wrapper only records how a flat family is partitioned by logical output.
#[derive(Clone)]
pub(crate) struct ProgramCacheBatchBindings {
    output_count: usize,
    family: BTreeMap<(usize, usize), ProgramCacheBatchFamily>,
}

#[derive(Clone)]
enum ProgramCacheBatchFamily {
    Shared { count: usize, values: Family<Preimage> },
    OutputIndexed { count: usize, values: Family<Preimage> },
}

impl ProgramCacheBatchBindings {
    pub(crate) fn new(output_count: usize) -> Result<Self, ExponentLutError> {
        if output_count == 0 {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(Self { output_count, family: BTreeMap::new() })
    }

    pub(crate) const fn output_count(&self) -> usize {
        self.output_count
    }

    pub(crate) fn insert_family(
        &mut self,
        step: usize,
        body_gate: usize,
        per_output_count: usize,
        rhs: Family<Preimage>,
    ) -> Result<(), ExponentLutError> {
        let expected = self
            .output_count
            .checked_mul(per_output_count)
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        if per_output_count == 0 || concrete_preimage_family_count(&rhs)? != expected {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        self.family.insert(
            (step, body_gate),
            ProgramCacheBatchFamily::OutputIndexed { count: per_output_count, values: rhs },
        );
        Ok(())
    }

    pub(crate) fn insert_shared_family(
        &mut self,
        step: usize,
        body_gate: usize,
        count: usize,
        rhs: Family<Preimage>,
    ) -> Result<(), ExponentLutError> {
        if count == 0 || concrete_preimage_family_count(&rhs)? != count {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        self.family
            .insert((step, body_gate), ProgramCacheBatchFamily::Shared { count, values: rhs });
        Ok(())
    }

    pub(crate) fn family_entry(
        &self,
        step: usize,
        body_gate: usize,
    ) -> Option<(usize, bool, &Family<Preimage>)> {
        self.family.get(&(step, body_gate)).map(|entry| match entry {
            ProgramCacheBatchFamily::Shared { count, values } => (*count, false, values),
            ProgramCacheBatchFamily::OutputIndexed { count, values } => (*count, true, values),
        })
    }

    /// Selects one logical output without statically materializing any family
    /// member.  Nested family caches are sliced from the flat label-major
    /// storage using the symbolic outer-loop index.
    pub(crate) fn output(
        &self,
        output: mxx_dsl::LoopIndex,
    ) -> Result<ProgramCacheBindings, ExponentLutError> {
        let mut selected = ProgramCacheBindings::new();
        for (&(step, body_gate), entry) in &self.family {
            let (count, values) = match entry {
                ProgramCacheBatchFamily::Shared { values, .. } => {
                    selected.insert_family(step, body_gate, values.clone());
                    continue;
                }
                ProgramCacheBatchFamily::OutputIndexed { count, values } => (*count, values),
            };
            let base = output.as_int().mul(Int::constant(count));
            let values = values.clone();
            selected.insert_family(
                step,
                body_gate,
                values
                    .parallel_gather(
                        Parallel::range(count)
                            .map_values(move |index| base.clone().add(index.as_int()))
                            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
                    )
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
            );
        }
        Ok(selected)
    }
}

#[derive(Clone)]
pub struct ProgramLutCache {
    pub(crate) switch: Family<Preimage>,
    pub(crate) branch: Family<Preimage>,
}

impl ProgramCacheBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn gate(mut self, step: usize, rhs: Preimage) -> Self {
        self.gate.insert(step, rhs);
        self
    }

    pub fn family(mut self, step: usize, body_gate: usize, rhs: Family<Preimage>) -> Self {
        self.family.insert((step, body_gate), rhs);
        self
    }

    pub fn lut(mut self, step: usize, switch: Family<Preimage>, branch: Family<Preimage>) -> Self {
        self.lut.insert(step, ProgramLutCache { switch, branch });
        self
    }

    pub fn family_lut(
        mut self,
        step: usize,
        body_gate: usize,
        switch: Family<Preimage>,
        branch: Family<Preimage>,
    ) -> Self {
        self.family_lut.insert((step, body_gate), ProgramLutCache { switch, branch });
        self
    }

    pub(crate) fn gate_entry(&self, step: usize) -> Option<&Preimage> {
        self.gate.get(&step)
    }

    pub(crate) fn family_entry(&self, step: usize, body_gate: usize) -> Option<&Family<Preimage>> {
        self.family.get(&(step, body_gate))
    }

    pub(crate) fn lut_entry(&self, step: usize) -> Option<&ProgramLutCache> {
        self.lut.get(&step)
    }

    pub(crate) fn family_lut_entry(
        &self,
        step: usize,
        body_gate: usize,
    ) -> Option<&ProgramLutCache> {
        self.family_lut.get(&(step, body_gate))
    }

    fn insert_gate(&mut self, step: usize, rhs: Preimage) {
        self.gate.insert(step, rhs);
    }

    fn insert_family(&mut self, step: usize, body_gate: usize, rhs: Family<Preimage>) {
        self.family.insert((step, body_gate), rhs);
    }

    fn insert_lut(&mut self, step: usize, cache: ProgramLutCache) {
        self.lut.insert(step, cache);
    }

    fn insert_family_lut(&mut self, step: usize, body_gate: usize, cache: ProgramLutCache) {
        self.family_lut.insert((step, body_gate), cache);
    }
}

pub fn program_cache_declarations(program: &ExponentLutProgram) -> Vec<ProgramCacheDeclaration> {
    let mut declarations = Vec::new();
    for (step, item) in program.steps().iter().enumerate() {
        match item {
            ProgramStep::Gate(Gate::MulPrivate { .. }) |
            ProgramStep::Gate(Gate::MulPublic { .. }) => {
                declarations.push(ProgramCacheDeclaration {
                    location: ProgramCacheLocation::Gate { step },
                    role: ProgramCacheRole::K_rhs,
                });
            }
            ProgramStep::Gate(Gate::Lut { lut, .. }) => {
                let width = program.lut(*lut).map_or(0, |table| table.values().len());
                for branch in 0..width {
                    for role in [ProgramCacheRole::LutSwitch, ProgramCacheRole::LutBranch] {
                        declarations.push(ProgramCacheDeclaration {
                            location: ProgramCacheLocation::Lut { step, branch },
                            role,
                        });
                    }
                }
            }
            ProgramStep::Family { body, .. } => {
                for (body_gate, gate) in body.gates().iter().enumerate() {
                    match gate {
                        crate::program::Gate::MulPrivate { .. } |
                        crate::program::Gate::MulPublic { .. } => {
                            declarations.push(ProgramCacheDeclaration {
                                location: ProgramCacheLocation::Family { step, body_gate },
                                role: ProgramCacheRole::K_rhs,
                            })
                        }
                        crate::program::Gate::Lut { lut, .. } => {
                            let width = program.lut(*lut).map_or(0, |table| table.values().len());
                            for branch in 0..width {
                                for role in
                                    [ProgramCacheRole::LutSwitch, ProgramCacheRole::LutBranch]
                                {
                                    declarations.push(ProgramCacheDeclaration {
                                        location: ProgramCacheLocation::FamilyLut {
                                            step,
                                            body_gate,
                                            branch,
                                        },
                                        role,
                                    });
                                }
                            }
                        }
                        crate::program::Gate::Add { .. } => {}
                    }
                }
            }
            _ => {}
        }
    }
    declarations
}

/// Public-key-only family consumed by the public accumulator lowerer.
/// Unlike the encoding backend's carrier/plaintext family, this type contains no
/// fabricated vector or plaintext carriers: only setup public matrices are
/// accepted and exposed to `MulPublic`.
#[derive(Clone)]
pub(crate) struct PublicKeyWireFamily {
    matrices: Family<Mat>,
    count: IntExpr,
}

impl PublicKeyWireFamily {
    pub(crate) fn from_matrices(
        matrices: Family<Mat>,
        ring: &mxx_dsl::Ring,
        secret_dimension: usize,
        public_key_columns: usize,
    ) -> Result<Self, ExponentLutError> {
        let ring_type = ring.matrix_type((1, 1));
        let matrix_type = matrices.element_type();
        let count = matrices
            .count()
            .evaluate(&ParamEnv::default())
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        if secret_dimension == 0 ||
            public_key_columns == 0 ||
            count <= 0.into() ||
            matrix_type.modulus != ring_type.modulus ||
            matrix_type.ring_dimension != ring_type.ring_dimension ||
            matrix_type.rows != IntExpr::constant(secret_dimension) ||
            matrix_type.columns != IntExpr::constant(public_key_columns)
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let family_count = matrices.count().clone();
        Ok(Self { matrices, count: family_count })
    }

    pub(crate) fn count(&self) -> &IntExpr {
        &self.count
    }

    pub(crate) fn matrices(&self) -> &Family<Mat> {
        &self.matrices
    }
}

/// Public-program outputs together with the typed deterministic cache
/// witnesses generated from the same public-key graph.
#[derive(Clone)]
#[cfg(test)]
pub(crate) struct PublicProgramPreprocessing {
    pub cache: ProgramCacheBindings,
}

/// Output-indexed public preprocessing result used by native PRF batches.
#[derive(Clone)]
pub(crate) struct PublicProgramBatchPreprocessing {
    pub outputs: Family<Mat>,
    pub cache: ProgramCacheBatchBindings,
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
#[derive(Clone)]
pub struct LutPublicHelperSet {
    table_commitment: [u8; 32],
    width: usize,
    helpers: Vec<LutPublicHelper>,
}

impl LutPublicHelperSet {
    pub fn new(
        table: &crate::program::LutTable,
        helpers: Vec<LutPublicHelper>,
    ) -> Result<Self, ExponentLutError> {
        if table.values().len() != helpers.len() {
            return Err(ExponentLutError::InvalidLut);
        }
        Ok(Self { table_commitment: table.commitment(), width: helpers.len(), helpers })
    }
    pub(crate) fn resolve(
        &self,
        table: &crate::program::LutTable,
    ) -> Result<&[LutPublicHelper], ExponentLutError> {
        if self.width != table.values().len() || self.table_commitment != table.commitment() {
            return Err(ExponentLutError::InvalidLut);
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
                    crate::encoding::canonical_lut_mask_branch_tag(&root, sigma),
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
    ) -> Result<BTreeMap<WireId, BggPublicKeyWire>, ExponentLutError> {
        let mut ranges = ProgramFamilyBindings::new();
        for (index, family) in selectors.values().enumerate() {
            ranges.bind(
                crate::program::ProgramFamilyId::from_index(index),
                FamilyRange::full(family.count().clone())
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
            );
        }
        self.compile_program_wrapped(
            program, inputs, rhs_inputs, selectors, values, &ranges, helpers, None,
        )
    }
    pub(crate) fn compile_program_with_ranges(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggPublicKeyWire>,
        rhs: &BTreeMap<RhsId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, PublicRhsFamily>,
        values: &BTreeMap<crate::program::WireFamilyId, PublicKeyWireFamily>,
        ranges: &ProgramFamilyBindings,
        helpers: &LutPublicHelperMap,
    ) -> Result<BTreeMap<WireId, BggPublicKeyWire>, ExponentLutError> {
        self.compile_program_wrapped(program, inputs, rhs, selectors, values, ranges, helpers, None)
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
    ) -> Result<PublicProgramPreprocessing, ExponentLutError> {
        let mut cache = ProgramCacheBindings::new();
        self.compile_program_wrapped(
            program,
            inputs,
            rhs_inputs,
            selectors,
            values,
            ranges,
            helpers,
            Some(&mut cache),
        )?;
        Ok(PublicProgramPreprocessing { cache })
    }

    /// Evaluates the selection-only native PRF for every logical output while
    /// retaining one structural label axis.  Public values are flattened as
    /// `[output][active cell]`; selector GSWs are setup-fixed and shared.
    /// A decomposition whose source varies with the public output carries the
    /// output axis. Setup-fixed selector decompositions remain shared.
    pub(crate) fn compile_select_batch_with_preprocessing(
        &self,
        initial: BggPublicKeyWire,
        selectors: &PublicRhsFamily,
        public_values: Family<Mat>,
        output_count: usize,
        active_count: usize,
        ranges: &[(usize, usize)],
    ) -> Result<PublicProgramBatchPreprocessing, ExponentLutError> {
        let expected_values = output_count
            .checked_mul(active_count)
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        if output_count == 0 ||
            active_count == 0 ||
            concrete_mat_family_count(&public_values)? != expected_values
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let mut state = Parallel::range(output_count)
            .map_values(|_| initial.matrix.clone())
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let mut cache = ProgramCacheBatchBindings::new(output_count)?;
        for (step, &(start, count)) in ranges.iter().enumerate() {
            if count == 0 || start.checked_add(count).is_none_or(|end| end > active_count) {
                return Err(ExponentLutError::InvalidSparseLwrBlock);
            }
            let flat_count =
                output_count.checked_mul(count).ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
            let output_indices = Parallel::range(flat_count)
                .map_values(|index| index.as_int().div(Int::constant(count)))
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let local_indices = Parallel::range(flat_count)
                .map_values(|index| index.as_int().rem(Int::constant(count)))
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let selector_indices = local_indices
                .clone()
                .parallel_map(|_, index| index.add(Int::constant(start)))
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let lhs = state
                .clone()
                .parallel_gather(output_indices.clone())
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            // Selector GSW material is setup-fixed and independent of the
            // output index. Keep one cache family and broadcast it online.
            let shared_selector_decompositions = selectors
                .gsw()
                .clone()
                .parallel_gather(
                    Parallel::range(count)
                        .map_values(|index| index.as_int().add(Int::constant(start)))
                        .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
                )
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?
                .parallel_map_values(|_, matrix| {
                    matrix.decompose(
                        self.public_key.base.clone(),
                        self.public_key.digit_count.clone(),
                    )
                })
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let shared_for_outputs = shared_selector_decompositions.clone();
            let selector_decompositions = shared_for_outputs
                .parallel_gather(
                    Parallel::range(flat_count)
                        .map_values(move |index| index.as_int().rem(Int::constant(count)))
                        .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
                )
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let selected = (lhs, selector_decompositions)
                .parallel_zip_tuple(|_, (lhs, decomposition)| decomposition.mul_small_rhs(lhs))
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            cache.insert_shared_family(step, 0, count, shared_selector_decompositions)?;

            let value_indices = (output_indices, selector_indices)
                .parallel_zip_tuple(move |_, (output, source)| {
                    output.mul(Int::constant(active_count)).add(source)
                })
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let values = public_values
                .clone()
                .parallel_gather(value_indices)
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let selected_decompositions = selected
                .clone()
                .parallel_map_values(|_, matrix| {
                    matrix.decompose(
                        self.public_key.base.clone(),
                        self.public_key.digit_count.clone(),
                    )
                })
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let weighted = (values, selected_decompositions.clone())
                .parallel_zip_tuple(|_, (public, decomposition)| {
                    decomposition.mul_small_rhs(public)
                })
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            cache.insert_family(step, 1, count, selected_decompositions)?;
            state = reduce_family_segments(weighted, output_count, count)?;
        }
        Ok(PublicProgramBatchPreprocessing { outputs: state, cache })
    }

    fn compile_program_wrapped(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggPublicKeyWire>,
        rhs_inputs: &BTreeMap<RhsId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, PublicRhsFamily>,
        values: &BTreeMap<crate::program::WireFamilyId, PublicKeyWireFamily>,
        ranges: &ProgramFamilyBindings,
        helpers: &LutPublicHelperMap,
        cache: Option<&mut ProgramCacheBindings>,
    ) -> Result<BTreeMap<WireId, BggPublicKeyWire>, ExponentLutError> {
        let wrapped_inputs = inputs
            .iter()
            .map(|(id, wire)| (*id, PublicProgramWire::Scalar(wire.clone())))
            .collect();
        let wrapped_rhs = rhs_inputs
            .iter()
            .map(|(id, rhs)| (*id, PublicProgramRhs::Scalar(rhs.clone())))
            .collect();
        let wrapped_selectors = selectors
            .iter()
            .map(|(id, family)| (*id, PublicProgramRhs::Members(family.clone())))
            .collect();
        let wrapped_values = values
            .iter()
            .map(|(id, family)| (*id, PublicProgramWire::Members(family.clone())))
            .collect();
        let bindings = ProgramBindings::new(
            &wrapped_inputs,
            &wrapped_rhs,
            &wrapped_selectors,
            &wrapped_values,
            helpers,
        );
        let mut backend = PublicProgramLowering { compiler: self, cache, ranges };
        let lowered = lower_program(program, &bindings, ranges, &mut backend)?;
        let outputs = lowered
            .into_iter()
            .map(|(id, wire)| match wire {
                PublicProgramWire::Scalar(value) => Ok((id, value)),
                _ => Err(ExponentLutError::InvalidSparseLwrBlock),
            })
            .collect::<Result<_, _>>()?;
        Ok(outputs)
    }
    /// Setup-fixed Fuse is the public matrix product `A G^{-1}(C)`.
    #[cfg(test)]
    pub(crate) fn fuse_public(
        &self,
        input: &Mat,
        rhs: &ExponentRhsPackage,
    ) -> Result<Mat, ExponentLutError> {
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
    ) -> Result<Mat, ExponentLutError> {
        Ok(decomposition.mul_small_rhs(input.clone()))
    }
    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn single_input_lut(
        &self,
        input: &Mat,
        table: &[usize],
        helpers: &[LutPublicHelper],
    ) -> Result<Mat, ExponentLutError> {
        let width = table.len();
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let table =
            LutTable::unary(width, self.exponent_layout.effective_modulus(n), table.to_vec())
                .map_err(ExponentLutError::from)?;
        self.single_input_lut_table(input, &table, helpers)
    }

    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn single_input_lut_table(
        &self,
        input: &Mat,
        table: &LutTable,
        helpers: &[LutPublicHelper],
    ) -> Result<Mat, ExponentLutError> {
        let width = table.values().len();
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let lut_layout = crate::encoding::derive_lut_layout(&self.exponent_layout, width, n)?;
        if width == 0 ||
            !width.is_power_of_two() ||
            width > n ||
            n % width != 0 ||
            helpers.len() != width ||
            table.values().iter().any(|value| *value >= lut_layout.effective_modulus(n))
        {
            return Err(ExponentLutError::InvalidLut);
        }
        let sigmas =
            lut_layout.automorphism_indices(n).map_err(|_| ExponentLutError::InvalidLut)?;
        // The branch body computes `mask_j * (A^{\sigma_j} G^{-1}(C_j))`, the
        // public component corresponding to the private Fuse branch.
        let raw_publics = Family::pack(
            helpers
                .iter()
                .enumerate()
                .map(|(j, helper)| {
                    let expected = sigmas.get(j).copied().ok_or(ExponentLutError::InvalidLut)?;
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
        let branches = Family::try_parallel_zip_many_values(
            vec![raw_publics, switches, masks],
            move |_index, mut items| {
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
                let rhs = ExponentRhsPackage::new(switch).map_err(|_| mxx_dsl::DslError::Schema)?;
                // `fused = A^{\sigma} G^{-1}(C_{sigma,L})` is still public.
                let fused = rhs
                    .gsw_ciphertext()
                    .clone()
                    .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
                    .mul_small_rhs(raw);
                // `decomp = G^{-1}(fused)` has the gadget-digit shape needed
                // by `mask`; `mask * decomp` is the helper's public projection
                // that accompanies the private vector-side alignment.
                Ok(fused
                    .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
                    .mul_small_rhs(mask))
            },
        )
        .map_err(|_| ExponentLutError::InvalidLut)?;
        balanced_sum_family(branches)
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
    ) -> Result<Mat, ExponentLutError> {
        let width = table.values().len();
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let lut_layout = crate::encoding::derive_lut_layout(&self.exponent_layout, width, n)?;
        if width == 0 ||
            !width.is_power_of_two() ||
            width > n ||
            n % width != 0 ||
            helpers.len() != width ||
            cache.switch.count().evaluate(&ParamEnv::default()).ok().and_then(|v| v.to_usize()) !=
                Some(width) ||
            cache.branch.count().evaluate(&ParamEnv::default()).ok().and_then(|v| v.to_usize()) !=
                Some(width) ||
            table.values().iter().any(|value| *value >= lut_layout.effective_modulus(n))
        {
            return Err(ExponentLutError::InvalidLut);
        }
        let sigmas =
            lut_layout.automorphism_indices(n).map_err(|_| ExponentLutError::InvalidLut)?;
        for (index, helper) in helpers.iter().enumerate() {
            if helper.sigma() != sigmas[index] {
                return Err(ExponentLutError::InvalidLut);
            }
        }
        let masks = Family::pack(helpers.iter().map(|helper| helper.mask().clone()).collect())
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let aligned = (masks, cache.branch.clone())
            .parallel_zip_tuple_result(|_, (mask, decomposition)| {
                Ok(decomposition.mul_small_rhs(mask))
            })
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let _ = input;
        balanced_sum_family(aligned)
    }

    fn build_lut_cache(
        &self,
        input: &Mat,
        table: &LutTable,
        helpers: &[LutPublicHelper],
    ) -> Result<ProgramLutCache, ExponentLutError> {
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let sigmas =
            crate::encoding::derive_lut_layout(&self.exponent_layout, table.values().len(), n)?
                .automorphism_indices(n)
                .map_err(|_| ExponentLutError::InvalidLut)?;
        if helpers.len() != sigmas.len() {
            return Err(ExponentLutError::InvalidLut);
        }
        let mut switch = Vec::with_capacity(sigmas.len());
        let mut branch = Vec::with_capacity(sigmas.len());
        for (index, helper) in helpers.iter().enumerate() {
            if helper.sigma() != sigmas[index] {
                return Err(ExponentLutError::InvalidLut);
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
        Ok(ProgramLutCache {
            switch: Family::pack(switch).map_err(|_| ExponentLutError::InvalidLut)?,
            branch: Family::pack(branch).map_err(|_| ExponentLutError::InvalidLut)?,
        })
    }

    fn build_family_lut_cache(
        &self,
        inputs: &Family<Mat>,
        table: &LutTable,
        helpers: &[LutPublicHelper],
    ) -> Result<ProgramLutCache, ExponentLutError> {
        let member_count = concrete_mat_family_count(inputs)?;
        let width = table.values().len();
        let first = inputs.get(Int::constant(0));
        let n = first
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let sigmas = crate::encoding::derive_lut_layout(&self.exponent_layout, width, n)?
            .automorphism_indices(n)
            .map_err(|_| ExponentLutError::InvalidLut)?;
        if helpers.len() != width || sigmas.len() != width || member_count == 0 {
            return Err(ExponentLutError::InvalidLut);
        }
        let switch_base = Family::pack(
            helpers
                .iter()
                .zip(&sigmas)
                .map(|(helper, sigma)| {
                    if helper.sigma() != *sigma {
                        return Err(ExponentLutError::InvalidLut);
                    }
                    Ok(helper.switch().gsw_ciphertext().clone().decompose(
                        self.public_key.base.clone(),
                        self.public_key.digit_count.clone(),
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|_| ExponentLutError::InvalidLut)?;
        let flat_count = member_count.checked_mul(width).ok_or(ExponentLutError::InvalidLut)?;
        let member_indices = Parallel::range(flat_count)
            .map_values(|index| index.as_int().div(Int::constant(width)))
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let flat_inputs = inputs
            .clone()
            .parallel_gather(member_indices)
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let flat_switches = switch_base
            .parallel_gather(
                Parallel::range(flat_count)
                    .map_values(|index| index.as_int().rem(Int::constant(width)))
                    .map_err(|_| ExponentLutError::InvalidLut)?,
            )
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let sigma_values = sigmas.iter().copied().map(IntExpr::constant).collect::<Vec<_>>();
        let branches = (flat_inputs, flat_switches.clone())
            .parallel_zip_tuple(move |index, (input, switch)| {
                let flat = index.expression();
                let divisor = IntExpr::constant(width);
                let branch = IntExpr::Sub(
                    Box::new(flat.clone()),
                    Box::new(IntExpr::Mul(
                        Box::new(IntExpr::Div(Box::new(flat), Box::new(divisor.clone()))),
                        Box::new(divisor),
                    )),
                );
                let sigma =
                    IntExpr::Select { selector: Box::new(branch), branches: sigma_values.clone() };
                switch
                    .mul_small_rhs(input.ring_automorphism(sigma))
                    .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            })
            .map_err(|_| ExponentLutError::InvalidLut)?;
        Ok(ProgramLutCache { switch: flat_switches, branch: branches })
    }

    fn family_lut_with_cache(
        &self,
        inputs: &Family<Mat>,
        table: &LutTable,
        helpers: &[LutPublicHelper],
        cache: &ProgramLutCache,
    ) -> Result<Family<Mat>, ExponentLutError> {
        let member_count = concrete_mat_family_count(inputs)?;
        let width = table.values().len();
        let flat_count = member_count.checked_mul(width).ok_or(ExponentLutError::InvalidLut)?;
        if concrete_preimage_family_count(&cache.switch)? != flat_count ||
            concrete_preimage_family_count(&cache.branch)? != flat_count
        {
            return Err(ExponentLutError::InvalidLut);
        }
        let branch_indices = Parallel::range(flat_count)
            .map_values(|index| index.as_int().rem(Int::constant(width)))
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let masks = Family::pack(helpers.iter().map(|helper| helper.mask().clone()).collect())
            .map_err(|_| ExponentLutError::InvalidLut)?
            .parallel_gather(branch_indices)
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let aligned = (masks, cache.branch.clone())
            .parallel_zip_tuple_result(|_, (mask, branch)| Ok(branch.mul_small_rhs(mask)))
            .map_err(|_| ExponentLutError::InvalidLut)?;
        reduce_family_segments(aligned, member_count, width)
    }

    fn family_mul_private_with_cache(
        &self,
        input: BggPublicKeyWire,
        selectors: &PublicRhsFamily,
        selector_range: &FamilyRange,
        decompositions: Family<Preimage>,
    ) -> Result<PublicFamilyIntermediate, ExponentLutError> {
        let selector_count = selectors
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let start = selector_range
            .start()
            .evaluate(&ParamEnv::default())
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?
            .to_usize()
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let count = selector_range
            .count()
            .evaluate(&ParamEnv::default())
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?
            .to_usize()
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        if count == 0 ||
            start.checked_add(count).is_none() ||
            start + count > selector_count ||
            decompositions.count() != selector_range.count()
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let inputs = Parallel::range(count)
            .map_values(|_| input.matrix.clone())
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let matrices = (inputs, decompositions)
            .parallel_zip_tuple_result(|_, (matrix, decomposition)| {
                Ok(decomposition.mul_small_rhs(matrix))
            })
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        Ok(PublicFamilyIntermediate { matrices })
    }

    fn family_mul_public_with_cache(
        &self,
        mapped: PublicFamilyIntermediate,
        values: &PublicKeyWireFamily,
        value_range: &FamilyRange,
        decompositions: Family<Preimage>,
    ) -> Result<PublicFamilyIntermediate, ExponentLutError> {
        let value_count = values
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let start = value_range
            .start()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let count = value_range
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        if count == 0 ||
            start.checked_add(count).is_none() ||
            start + count > value_count ||
            decompositions.count() != value_range.count() ||
            mapped.matrices.count() != value_range.count()
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let indices = Parallel::range(count)
            .map_values(|index| index.as_int().add(mxx_dsl::Int::constant(start)))
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let values = values
            .matrices()
            .clone()
            .parallel_gather(indices)
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        let matrices = (values, decompositions)
            .parallel_zip_tuple_result(|_, (value, decomposition)| {
                Ok(decomposition.mul_small_rhs(value))
            })
            .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
        Ok(PublicFamilyIntermediate { matrices })
    }

    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn two_input_lut(
        &self,
        lhs: &Mat,
        rhs: &ExponentRhsPackage,
        lhs_width: usize,
        rhs_width: usize,
        table: &[usize],
        helpers: &[LutPublicHelper],
    ) -> Result<Mat, ExponentLutError> {
        if table.len() != lhs_width.checked_mul(rhs_width).ok_or(ExponentLutError::InvalidLut)? {
            return Err(ExponentLutError::InvalidLut);
        }
        self.single_input_lut(&self.fuse_public(lhs, rhs)?, table, helpers)
    }
}

#[derive(Clone)]
pub(crate) struct PublicFamilyIntermediate {
    matrices: Family<Mat>,
}

#[derive(Clone)]
pub(crate) enum PublicProgramWire {
    Scalar(BggPublicKeyWire),
    Broadcast { value: BggPublicKeyWire, range: FamilyRange },
    Members(PublicKeyWireFamily),
    Materialized { value: PublicFamilyIntermediate, range: FamilyRange },
}

#[derive(Clone)]
pub(crate) enum PublicProgramRhs {
    Scalar(ExponentRhsPackage),
    Members(PublicRhsFamily),
}

struct PublicProgramLowering<'a> {
    compiler: &'a ExponentLutPublicKeyCompiler,
    cache: Option<&'a mut ProgramCacheBindings>,
    ranges: &'a ProgramFamilyBindings,
}

impl ProgramLoweringTypes for PublicProgramLowering<'_> {
    type Wire = PublicProgramWire;
    type Rhs = PublicProgramRhs;
    type HelperSet = LutPublicHelperSet;
    type Error = ExponentLutError;
}

impl ExponentProgramLowering for PublicProgramLowering<'_> {
    fn mul_private(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Rhs,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        match (lhs, rhs, gate) {
            (
                PublicProgramWire::Scalar(lhs),
                PublicProgramRhs::Scalar(rhs),
                ProgramGateInstance::Gate { step },
            ) => {
                let decomposition = rhs.gsw_ciphertext().clone().decompose(
                    self.compiler.public_key.base.clone(),
                    self.compiler.public_key.digit_count.clone(),
                );
                if let Some(cache) = &mut self.cache {
                    cache.insert_gate(step, decomposition.clone());
                }
                Ok(PublicProgramWire::Scalar(BggPublicKeyWire {
                    matrix: decomposition.mul_small_rhs(lhs.matrix.clone()),
                    reveal_plaintext: false,
                }))
            }
            (
                PublicProgramWire::Broadcast { value, range },
                PublicProgramRhs::Members(selectors),
                ProgramGateInstance::FamilyGate { step, gate },
            ) => {
                let start = range
                    .start()
                    .evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
                let count = range
                    .count()
                    .evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
                let indices = Parallel::range(count)
                    .map_values(|index| index.as_int().add(mxx_dsl::Int::constant(start)))
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                let decompositions = selectors
                    .gsw()
                    .clone()
                    .parallel_gather(indices)
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?
                    .parallel_map_values(|_, matrix| {
                        matrix.decompose(
                            self.compiler.public_key.base.clone(),
                            self.compiler.public_key.digit_count.clone(),
                        )
                    })
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                if let Some(cache) = &mut self.cache {
                    cache.insert_family(step, gate, decompositions.clone());
                }
                let value = self.compiler.family_mul_private_with_cache(
                    value.clone(),
                    selectors,
                    range,
                    decompositions,
                )?;
                Ok(PublicProgramWire::Materialized { value, range: range.clone() })
            }
            _ => Err(ExponentLutError::InvalidSparseLwrBlock),
        }
    }

    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        match (private, public, gate) {
            (
                PublicProgramWire::Scalar(private),
                PublicProgramWire::Scalar(public),
                ProgramGateInstance::Gate { step },
            ) => {
                let decomposition = private.matrix.clone().decompose(
                    self.compiler.public_key.base.clone(),
                    self.compiler.public_key.digit_count.clone(),
                );
                if let Some(cache) = &mut self.cache {
                    cache.insert_gate(step, decomposition.clone());
                }
                Ok(PublicProgramWire::Scalar(self.compiler.public_key.mul_with_decomposition(
                    public,
                    private,
                    decomposition,
                )))
            }
            (
                PublicProgramWire::Materialized { value: private, range },
                PublicProgramWire::Members(values),
                ProgramGateInstance::FamilyGate { step, gate },
            ) => {
                let decompositions = private
                    .matrices
                    .clone()
                    .parallel_map_values(|_, matrix| {
                        matrix.decompose(
                            self.compiler.public_key.base.clone(),
                            self.compiler.public_key.digit_count.clone(),
                        )
                    })
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                if let Some(cache) = &mut self.cache {
                    cache.insert_family(step, gate, decompositions.clone());
                }
                let value = self.compiler.family_mul_public_with_cache(
                    private.clone(),
                    values,
                    range,
                    decompositions,
                )?;
                Ok(PublicProgramWire::Materialized { value, range: range.clone() })
            }
            _ => Err(ExponentLutError::InvalidSparseLwrBlock),
        }
    }

    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        _gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        match (lhs, rhs) {
            (PublicProgramWire::Scalar(lhs), PublicProgramWire::Scalar(rhs)) => {
                Ok(PublicProgramWire::Scalar(self.compiler.public_key.add(lhs, rhs)))
            }
            (
                PublicProgramWire::Materialized { value: lhs, range },
                PublicProgramWire::Materialized { value: rhs, range: rhs_range },
            ) if range == rhs_range => {
                let matrices = (lhs.matrices.clone(), rhs.matrices.clone())
                    .parallel_zip_tuple_result(|_, (lhs, rhs)| Ok(lhs + rhs))
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                Ok(PublicProgramWire::Materialized {
                    value: PublicFamilyIntermediate { matrices },
                    range: range.clone(),
                })
            }
            _ => Err(ExponentLutError::InvalidSparseLwrBlock),
        }
    }

    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: crate::program::LutId,
        input: &Self::Wire,
        helpers: &Self::HelperSet,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let table = program.lut(lut).ok_or(ExponentLutError::InvalidLut)?;
        let helpers = helpers.resolve(table)?;
        match (input, gate) {
            (PublicProgramWire::Scalar(input), ProgramGateInstance::Gate { step }) => {
                let cache = self.compiler.build_lut_cache(&input.matrix, table, helpers)?;
                let matrix = self.compiler.single_input_lut_table_with_cache(
                    &input.matrix,
                    table,
                    helpers,
                    &cache,
                )?;
                if let Some(bindings) = &mut self.cache {
                    bindings.insert_lut(step, cache);
                }
                Ok(PublicProgramWire::Scalar(BggPublicKeyWire { matrix, reveal_plaintext: false }))
            }
            (family_input, ProgramGateInstance::FamilyGate { step, gate }) => {
                let range = match program.steps().get(step) {
                    Some(ProgramStep::Family { id, .. }) => {
                        self.ranges.range(*id).ok_or(ExponentLutError::InvalidSparseLwrBlock)?
                    }
                    _ => return Err(ExponentLutError::InvalidLut),
                };
                let matrices = active_public_family(family_input, range)?;
                let cache = self.compiler.build_family_lut_cache(&matrices, table, helpers)?;
                let matrices =
                    self.compiler.family_lut_with_cache(&matrices, table, helpers, &cache)?;
                if let Some(bindings) = &mut self.cache {
                    bindings.insert_family_lut(step, gate, cache);
                }
                Ok(PublicProgramWire::Materialized {
                    value: PublicFamilyIntermediate { matrices },
                    range: range.clone(),
                })
            }
            _ => Err(ExponentLutError::InvalidLut),
        }
    }

    fn backend_layout(&self) -> Option<crate::program::ProgramLayout> {
        let ring_dimension = self
            .compiler
            .public_key
            .ring
            .matrix_type((1, 1))
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())?;
        Some(crate::program::ProgramLayout {
            exponent: self.compiler.exponent_layout,
            ring_dimension_log2: ring_dimension.trailing_zeros(),
        })
    }

    fn public_attribute_available(&self, wire: &Self::Wire) -> bool {
        matches!(wire, PublicProgramWire::Scalar(_))
    }

    fn broadcast_wire(
        &mut self,
        wire: &Self::Wire,
        range: &FamilyRange,
    ) -> Result<Self::Wire, Self::Error> {
        let PublicProgramWire::Scalar(value) = wire else {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        };
        Ok(PublicProgramWire::Broadcast { value: value.clone(), range: range.clone() })
    }

    fn reduce_add(
        &mut self,
        value: &Self::Wire,
        range: &FamilyRange,
        _gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let PublicProgramWire::Materialized { value, range: value_range } = value else {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        };
        if value_range != range {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(PublicProgramWire::Scalar(BggPublicKeyWire {
            matrix: balanced_sum_family(value.matrices.clone())?,
            reveal_plaintext: false,
        }))
    }
}

fn active_public_family(
    input: &PublicProgramWire,
    expected_range: &FamilyRange,
) -> Result<Family<Mat>, ExponentLutError> {
    let count = expected_range
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .filter(|count| *count != 0)
        .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
    let start = expected_range
        .start()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
    match input {
        PublicProgramWire::Materialized { value, range } if range == expected_range => {
            Ok(value.matrices.clone())
        }
        PublicProgramWire::Broadcast { value, range } if range == expected_range => {
            Parallel::range(count)
                .map_values(|_| value.matrix.clone())
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)
        }
        PublicProgramWire::Members(values) => {
            let indices = Parallel::range(count)
                .map_values(|index| index.as_int().add(Int::constant(start)))
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            values
                .matrices()
                .clone()
                .parallel_gather(indices)
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)
        }
        _ => Err(ExponentLutError::InvalidSparseLwrBlock),
    }
}

pub type PublicRhsFamily = crate::encoding::SetupRhsFamily;

fn concrete_mat_family_count(family: &Family<Mat>) -> Result<usize, ExponentLutError> {
    family
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .filter(|count| *count != 0)
        .ok_or(ExponentLutError::InvalidLut)
}

fn concrete_preimage_family_count(family: &Family<Preimage>) -> Result<usize, ExponentLutError> {
    family
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .filter(|count| *count != 0)
        .ok_or(ExponentLutError::InvalidLut)
}

fn reduce_family_segments(
    family: Family<Mat>,
    segment_count: usize,
    segment_size: usize,
) -> Result<Family<Mat>, ExponentLutError> {
    crate::encoding::reduce_encoding_family_segments(family, segment_count, segment_size)
}

#[cfg(test)]
mod tests {
    use super::{
        ExponentLutPublicKeyCompiler, ExponentLutPublicKeySampler, LutPublicHelper,
        LutPublicHelperSet,
    };
    use crate::{program::LutTable, rhs::ExponentRhsPackage};
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
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let output_count = 17;
        let active_count = 2;
        let selectors = crate::encoding::SetupRhsFamily::new(ring.input_family(
            "batch-selectors",
            active_count,
            (2, 4),
        ))
        .unwrap();
        let preprocessing = compiler
            .compile_select_batch_with_preprocessing(
                BggPublicKeyWire {
                    matrix: ring.input("batch-initial", (2, 4)),
                    reveal_plaintext: false,
                },
                &selectors,
                ring.input_family("batch-public-values", output_count * active_count, (2, 4)),
                output_count,
                active_count,
                &[(0, active_count)],
            )
            .unwrap();
        assert_eq!(
            preprocessing.outputs.count().evaluate(&ParamEnv::default()).unwrap(),
            output_count.into()
        );
        let (shared_count, shared_indexed, shared) =
            preprocessing.cache.family_entry(0, 0).unwrap();
        assert_eq!(shared_count, active_count);
        assert!(!shared_indexed);
        assert_eq!(shared.count().evaluate(&ParamEnv::default()).unwrap(), active_count.into());
        let (private_count, private_indexed, private) =
            preprocessing.cache.family_entry(0, 1).unwrap();
        assert_eq!(private_count, active_count);
        assert!(private_indexed);
        assert_eq!(
            private.count().evaluate(&ParamEnv::default()).unwrap(),
            (output_count * active_count).into()
        );
        let graph = DslContext::new("public-prf-output-axis")
            .public_output("outputs", preprocessing.outputs)
            .unwrap()
            .public_output("shared-selector-cache", shared.clone())
            .unwrap()
            .public_output("output-private-cache", private.clone())
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
    }
}
