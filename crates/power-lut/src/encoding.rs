//! Private Power-LUT evaluation over ordinary `BggEncodingWire` values.
//!
//! This module implements only the setup-fixed RHS interface. A fixed GSW
//! ciphertext is gadget-decomposed once and multiplied by the input vector;
//! no encoding of individual GSW digits is constructed. LUT evaluation uses
//! the flat specialization: one setup-fixed switch and one reusable mask
//! alignment per canonical automorphism branch, followed by a balanced sum.
//!
//! With `c = s A - μ t G + e` and a fixed GSW ciphertext satisfying
//! `t C = y v G + e_C`, setup-fixed Fuse computes
//! `D = G^{-1}(C)` and `cD = s(A D) - μ y v G + e'`. For unary table `f`,
//! branch `j` uses `sigma_j = 1 + j(2n/W)` and
//! `D_{sigma_j,L} = W^{-1} sum_k L(k) X^{-k sigma_j}` (a scalar output uses
//! the constant polynomial `L(k)`, while a monomial output uses `X^{L(k)}`).
//! The branch is mask-aligned and the `W` branches are added; this is the
//! meaning of the `sigma`, `coefficient`, `decomposed`, and balanced-family
//! values below.

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    PowerLutError,
    program::{
        FamilyRange, LutOutputForm, LutTable, PowerLutMonomialFamily, PowerLutProgram,
        ProgramBindings, ProgramFamilyRanges, ProgramInputId, ProgramLoweringBackend,
        ProgramWireId, RhsInputId, lower_program,
    },
    rhs::{
        ManifestSecretMetadata, PowerRhsPackage, PowerRhsPackageArtifactNames, PowerRhsPackageError,
    },
};
use mxx_bgg::{BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{Bytes, DslError, Family, HashTag, Mat, Parallel};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    node::{ConcatAxis, ConstantMatrix, IndexRange},
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A setup-fixed mask-alignment branch, reusable across LUT tables.
#[derive(Clone)]
pub struct FlatLutMaskBank {
    branches: Arc<Vec<FlatLutMaskBranch>>,
}

#[derive(Clone)]
struct FlatLutMaskBranch {
    sigma: usize,
    mask: BggEncodingWire,
}

/// Public projections of a reusable [`FlatLutMaskBank`].
#[derive(Clone)]
pub struct FlatLutPublicMaskBank {
    branches: Vec<(usize, Mat)>,
}

impl FlatLutMaskBank {
    fn single(sigma: usize, mask: BggEncodingWire) -> Self {
        Self { branches: Arc::new(vec![FlatLutMaskBranch { sigma, mask }]) }
    }

    pub(crate) fn index_for_sigma(&self, sigma: usize) -> Option<usize> {
        self.branches.iter().position(|branch| branch.sigma == sigma)
    }

    fn branch(&self, index: usize) -> Option<&FlatLutMaskBranch> {
        self.branches.get(index)
    }

    /// Returns the ordered odd automorphisms covered by this bank.
    pub fn sigmas(&self) -> impl Iterator<Item = usize> + '_ {
        self.branches.iter().map(|branch| branch.sigma)
    }
}

impl FlatLutPublicMaskBank {
    pub(crate) fn from_branches(branches: Vec<(usize, Mat)>) -> Self {
        Self { branches }
    }

    pub(crate) fn single(sigma: usize, matrix: Mat) -> Self {
        Self { branches: vec![(sigma, matrix)] }
    }

    pub(crate) fn index_for_sigma(&self, sigma: usize) -> Option<usize> {
        self.branches.iter().position(|(branch_sigma, _)| *branch_sigma == sigma)
    }

    pub(crate) fn matrix(&self, index: usize) -> Option<&Mat> {
        self.branches.get(index).map(|(_, matrix)| matrix)
    }
}

/// Constructs the shared domain for one canonical mask branch. Both private
/// and public samplers feed this exact tag to the BGG public-key sampler.
pub(crate) fn canonical_flat_mask_branch_tag(root: &HashTag, sigma: usize) -> HashTag {
    let mut tag = root.clone();
    tag.push("power-lut-flat-mask-bank-v1");
    tag.push("mask");
    tag.push(IntExpr::constant(sigma));
    tag
}

/// Setup-fixed helper for one flat LUT branch.
#[derive(Clone)]
pub struct FlatLutHelper {
    sigma: usize,
    switch: PowerRhsPackage,
    mask_bank: Arc<FlatLutMaskBank>,
    mask_index: usize,
}

/// All setup-fixed branches for one concrete LUT table.
///
/// The commitment is metadata for artifact binding only; it is never carried
/// by a BGG encoding wire or used as a cryptographic provenance check during
/// ordinary lowering.
#[derive(Clone)]
pub struct FlatLutHelperSet {
    table_commitment: [u8; 32],
    width: usize,
    helpers: Vec<FlatLutHelper>,
}

impl FlatLutHelperSet {
    pub fn new(
        table: &crate::program::LutTable,
        helpers: Vec<FlatLutHelper>,
    ) -> Result<Self, PowerLutError> {
        if table.values().len() != helpers.len() {
            return Err(PowerLutError::InvalidLut);
        }
        Ok(Self { table_commitment: table.commitment(), width: helpers.len(), helpers })
    }

    pub(crate) fn from_parts(
        table_commitment: [u8; 32],
        width: usize,
        helpers: Vec<FlatLutHelper>,
    ) -> Result<Self, PowerLutError> {
        if width == 0 || width != helpers.len() {
            return Err(PowerLutError::InvalidLut);
        }
        Ok(Self { table_commitment, width, helpers })
    }

    pub(crate) fn resolve(
        &self,
        table: &crate::program::LutTable,
    ) -> Result<&[FlatLutHelper], PowerLutError> {
        if self.width != table.values().len() || self.table_commitment != table.commitment() {
            return Err(PowerLutError::InvalidLut);
        }
        Ok(&self.helpers)
    }

    pub(crate) fn as_slice(&self) -> &[FlatLutHelper] {
        &self.helpers
    }
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, FlatLutHelper> {
        self.helpers.iter()
    }
    pub(crate) fn metadata(&self) -> ([u8; 32], usize) {
        (self.table_commitment, self.width)
    }
}

impl FlatLutHelper {
    pub(crate) fn new(
        sigma: usize,
        switch: PowerRhsPackage,
        mask: BggEncodingWire,
    ) -> Result<Self, PowerLutError> {
        crate::ensure_ciphertext_only(&mask)?;
        if sigma == 0 || sigma % 2 == 0 {
            return Err(PowerLutError::InvalidLut);
        }
        Self::with_mask_bank(sigma, switch, Arc::new(FlatLutMaskBank::single(sigma, mask)))
    }
    pub(crate) fn with_mask_bank(
        sigma: usize,
        switch: PowerRhsPackage,
        mask_bank: Arc<FlatLutMaskBank>,
    ) -> Result<Self, PowerLutError> {
        let mask_index = mask_bank.index_for_sigma(sigma).ok_or(PowerLutError::InvalidLut)?;
        Ok(Self { sigma, switch, mask_bank, mask_index })
    }
    pub(crate) fn sigma(&self) -> usize {
        self.sigma
    }
    pub(crate) fn switch(&self) -> &PowerRhsPackage {
        &self.switch
    }
    pub(crate) fn mask(&self) -> &BggEncodingWire {
        &self.mask_bank.branch(self.mask_index).expect("validated mask index").mask
    }
}

/// Names of independently stored flat helper components.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FlatLutHelperArtifactNames {
    pub switch: PowerRhsPackageArtifactNames,
    pub mask: BggEncodingArtifactNames,
}

/// A helper registry keyed by the program LUT identity. The registry prevents
/// a fixed switch generated for one table from being accidentally reused for
/// another table while keeping the evaluator free of provenance wrappers.
#[derive(Clone, Default)]
pub struct FlatLutHelperRegistry {
    helpers: BTreeMap<crate::program::LutId, FlatLutHelperSet>,
}

/// Runtime helper bindings keyed by the exact LUT identity used by a gate.
pub type FlatLutHelperMap = BTreeMap<crate::program::LutId, FlatLutHelperSet>;

impl FlatLutHelperRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(
        &mut self,
        lut: crate::program::LutId,
        helpers: FlatLutHelperSet,
    ) -> Result<(), PowerLutError> {
        if self.helpers.insert(lut, helpers).is_some() {
            return Err(PowerLutError::InvalidLut);
        }
        Ok(())
    }
    pub fn get(&self, lut: crate::program::LutId) -> Option<&[FlatLutHelper]> {
        self.helpers.get(&lut).map(FlatLutHelperSet::as_slice)
    }
}

/// Errors raised while constructing setup-time Power-LUT inputs and helpers.
#[derive(Debug, Error)]
pub enum PowerLutSamplingError {
    #[error(transparent)]
    Bgg(#[from] mxx_bgg::BggSampleError),
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error(transparent)]
    PowerLut(#[from] PowerLutError),
    #[error(transparent)]
    Rhs(#[from] PowerRhsPackageError),
    #[error("invalid Power-LUT sampler configuration: {0}")]
    InvalidConfiguration(&'static str),
}

/// Setup sampler for input encodings and flat LUT helpers.
#[derive(Clone)]
pub struct PowerLutEncodingSampler {
    pub layout: mxx_bgg::BggSamplerLayout,
    pub gaussian_sigma: Option<mxx_ir_core::RealExpr>,
    pub gaussian_max_coefficient_bound: Option<IntExpr>,
}

/// Compiler for private Power-LUT graphs over ordinary BGG+ wires.
pub struct PowerLutEncodingCompiler {
    pub bgg: BggEncodingCompiler,
}

impl PowerLutEncodingSampler {
    /// Samples an augmented secret `(s_bar,1)` using private randomness.
    pub fn sample_secret(&self) -> Result<Mat, PowerLutSamplingError> {
        if self.layout.secret_dimension < 2 {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "secret dimension must be at least two",
            ));
        }
        // `s_bar = (s,1)` is the augmented BGG+ secret: small random entries
        // form `s`, and the final identity column supplies the affine constant.
        let ring = self.layout.ring();
        Ok(Mat::concat(
            ConcatAxis::Columns,
            vec![
                ring.uniform_interval((1, self.layout.secret_dimension - 1), -1, 1),
                ring.identity(1),
            ],
        ))
    }

    /// Samples a ciphertext-only batch of inputs from one indexed public-key
    /// family, then delegates the actual BGG+ construction to the common
    /// public-matrix core.
    pub fn sample_input_encodings(
        &self,
        mask_secret: Mat,
        payload_secret: Option<Mat>,
        hash_key: Bytes,
        base_tag: impl Into<HashTag>,
        plaintexts: &[Mat],
    ) -> Result<Vec<BggEncodingWire>, PowerLutSamplingError> {
        if plaintexts.is_empty() {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "input-encoding count must be positive",
            ));
        }
        let keys = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() }.sample(
            hash_key,
            base_tag,
            &vec![false; plaintexts.len()],
        );
        let public_keys = keys.into_iter().skip(1).collect::<Vec<_>>();
        self.sample_encodings_for_public_matrices(
            mask_secret,
            payload_secret,
            &public_keys,
            plaintexts,
        )
    }

    /// Samples under existing public matrices, with an optional payload
    /// secret for the separate-secret BGG+ relation. The leading constant
    /// public key required by the packed BGG+ sampler is supplied internally.
    pub fn sample_encodings_for_public_matrices(
        &self,
        mask_secret: Mat,
        payload_secret: Option<Mat>,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &[Mat],
    ) -> Result<Vec<BggEncodingWire>, PowerLutSamplingError> {
        if public_keys.is_empty() || public_keys.len() != plaintexts.len() {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "public-key and plaintext counts must be equal and positive",
            ));
        }
        if public_keys.iter().any(|key| key.reveal_plaintext) {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "public keys must be ciphertext-only",
            ));
        }
        let ring = self.layout.ring();
        let constant = BggPublicKeyWire {
            matrix: ring.zero((self.layout.secret_dimension, self.layout.public_key_columns())),
            reveal_plaintext: false,
        };
        let mut all_public_keys = Vec::with_capacity(public_keys.len() + 1);
        all_public_keys.push(constant);
        all_public_keys.extend(public_keys.iter().cloned());
        let values = mxx_bgg::BggEncodingSampler {
            layout: self.layout.clone(),
            gaussian_sigma: self.gaussian_sigma.clone(),
            gaussian_max_coefficient_bound: self.gaussian_max_coefficient_bound.clone(),
        }
        .sample(mask_secret, payload_secret, &all_public_keys, plaintexts)?;
        Ok(values.into_iter().skip(1).collect())
    }

    /// Samples the reusable mask-alignment bank for the largest supported LUT.
    /// Mask branches are keyed only by their canonical automorphism, so the
    /// same bank can serve every smaller compatible LUT width.
    pub fn sample_flat_mask_bank(
        &self,
        mask_secret: Mat,
        hash_key: Bytes,
        max_width: usize,
        tag: impl Into<HashTag>,
    ) -> Result<Arc<FlatLutMaskBank>, PowerLutSamplingError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutSamplingError::InvalidConfiguration(
                "ring dimension must be concrete",
            ))?;
        if max_width == 0 || !max_width.is_power_of_two() || max_width > n || n % max_width != 0 {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "mask-bank width must be a power of two dividing the ring dimension",
            ));
        }
        let bank_tag = tag.into();
        let mut branches = Vec::with_capacity(max_width);
        for j in 0..max_width {
            let sigma = 1usize
                .checked_add(
                    j.checked_mul(2 * n / max_width)
                        .ok_or(PowerLutSamplingError::InvalidConfiguration("sigma overflow"))?,
                )
                .ok_or(PowerLutSamplingError::InvalidConfiguration("sigma overflow"))?;
            let mask_tag = canonical_flat_mask_branch_tag(&bank_tag, sigma);
            let mask_key = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() }
                .sample(hash_key.clone(), mask_tag, &[false])
                .into_iter()
                .nth(1)
                .ok_or(PowerLutSamplingError::InvalidConfiguration("mask sample is empty"))?;
            let mask_sigma = mask_secret.clone().ring_automorphism(sigma);
            let constant = BggPublicKeyWire {
                matrix: self
                    .layout
                    .ring()
                    .zero((self.layout.secret_dimension, self.layout.public_key_columns())),
                reveal_plaintext: false,
            };
            let mask = mxx_bgg::BggEncodingSampler {
                layout: self.layout.clone(),
                gaussian_sigma: self.gaussian_sigma.clone(),
                gaussian_max_coefficient_bound: self.gaussian_max_coefficient_bound.clone(),
            }
            .sample(
                mask_secret.clone(),
                Some(mask_sigma),
                &[constant, mask_key],
                &[self.layout.ring().identity(1)],
            )?
            .into_iter()
            .nth(1)
            .ok_or(PowerLutSamplingError::InvalidConfiguration("mask sample is empty"))?;
            branches.push(FlatLutMaskBranch { sigma, mask });
        }
        Ok(Arc::new(FlatLutMaskBank { branches: Arc::new(branches) }))
    }

    /// Samples setup-fixed helpers for a complete LUT declaration using an
    /// explicitly shared mask bank. The bank must cover every canonical sigma
    /// required by this table; it is never sampled implicitly here.
    ///
    /// The declaration, including its output form, is the source of truth for
    /// both helper commitments and the branch coefficient.  A scalar table
    /// therefore receives constant-polynomial coefficients, while a monomial
    /// table receives the corresponding rotations.
    pub fn sample_flat_helpers_for_lut(
        &self,
        mask_secret: Mat,
        payload_secret: Option<Mat>,
        hash_key: Bytes,
        table: &LutTable,
        mask_bank: &FlatLutMaskBank,
        tag: impl Into<HashTag>,
    ) -> Result<Vec<FlatLutHelper>, PowerLutSamplingError> {
        let width = table.values().len();
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutSamplingError::InvalidConfiguration(
                "ring dimension must be concrete",
            ))?;
        if width == 0 ||
            !width.is_power_of_two() ||
            table.input_width() != width ||
            width > n ||
            n % width != 0 ||
            table.values().iter().any(|value| *value >= n)
        {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "LUT width or output exponent is invalid",
            ));
        }
        let payload = payload_secret.unwrap_or_else(|| mask_secret.clone());
        let shared_mask_bank = Arc::new(mask_bank.clone());
        let mut tag = tag.into();
        tag.push("power-lut-flat-v1");
        let mut helpers = Vec::with_capacity(width);
        for j in 0..width {
            // `j` chooses one Fourier branch; `sigma` is odd so its ring
            // automorphism is invertible and follows the canonical LUT order.
            let sigma = 1usize
                .checked_add(
                    j.checked_mul(2 * n / width)
                        .ok_or(PowerLutSamplingError::InvalidConfiguration("sigma overflow"))?,
                )
                .ok_or(PowerLutSamplingError::InvalidConfiguration("sigma overflow"))?;
            let coefficient = lut_coefficient(
                &self.layout.ring(),
                n,
                width,
                sigma,
                table.values(),
                table.output_form(),
            )?;
            // The switch encrypts the rotated payload relation; the separate
            // mask encoding supplies the compensating public alignment.
            // The fixed switch is generated under the payload secret after
            // automorphism. The mask alignment is a separate encoding under
            // the mask secret and its automorphed image.
            let payload_sigma = payload.clone().ring_automorphism(sigma);
            let switch = self.sample_fixed_rhs(
                payload_sigma,
                payload.clone(),
                coefficient,
                hash_key.clone(),
                {
                    let mut t = tag.clone();
                    t.push("switch");
                    t.push(IntExpr::constant(j));
                    t
                },
            )?;
            if mask_bank.index_for_sigma(sigma).is_none() {
                return Err(PowerLutSamplingError::InvalidConfiguration(
                    "mask bank does not cover LUT canonical sigma",
                ));
            }
            helpers.push(FlatLutHelper::with_mask_bank(sigma, switch, shared_mask_bank.clone())?);
        }
        Ok(helpers)
    }

    fn sample_fixed_rhs(
        &self,
        source: Mat,
        target: Mat,
        payload: Mat,
        hash_key: Bytes,
        tag: HashTag,
    ) -> Result<PowerRhsPackage, PowerLutSamplingError> {
        let ring = self.layout.ring();
        let columns = self.layout.public_key_columns();
        // For mask secret `s`, payload secret `t`, and scalar `y`, construct
        // `C = [R; y*t*G - s*R + e_C]`, so `[s,1]C = y*t*G + e_C`.
        let mut top_tag = tag.clone();
        top_tag.push("power-lut/fixed-rhs/top/v1");
        let top = ring.hash_matrix(
            hash_key.clone(),
            top_tag,
            (self.layout.secret_dimension - 1, columns),
        );
        let source_prefix = source.clone().slice(
            None,
            Some(IndexRange { start: 0.into(), end: (self.layout.secret_dimension - 1).into() }),
        );
        let error = match (&self.gaussian_sigma, &self.gaussian_max_coefficient_bound) {
            (Some(sigma), Some(bound)) => ring.gaussian((1, columns), sigma.clone(), bound.clone()),
            (None, None) => ring.zero((1, columns)),
            _ => {
                return Err(PowerLutSamplingError::InvalidConfiguration(
                    "Gaussian sigma and cutoff must be paired",
                ))
            }
        };
        let gadget = ring.gadget(
            self.layout.secret_dimension,
            self.layout.gadget_base.clone(),
            self.layout.digit_count,
        );
        let last = payload * (target * gadget) - source_prefix * top.clone() + error;
        let ciphertext = Mat::concat(ConcatAxis::Rows, vec![top, last]);
        PowerRhsPackage::new(ciphertext).map_err(Into::into)
    }

    /// Samples one fixed RHS ciphertext for the PBC selector producer.
    /// The selector value is setup-fixed payload data, not an encoded digit
    /// family, so this returns the ciphertext package directly.
    pub(crate) fn sample_cross_secret_rhs(
        &self,
        source: Mat,
        target: Mat,
        payload: Mat,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
    ) -> Result<PowerRhsPackage, PowerLutSamplingError> {
        self.sample_fixed_rhs(source, target, payload, hash_key, tag.into())
    }
}

impl PowerLutEncodingCompiler {
    pub fn new(bgg: BggEncodingCompiler) -> Self {
        Self { bgg }
    }
    pub fn from_public_key(public_key: BggPublicKeyCompiler) -> Self {
        Self::new(BggEncodingCompiler { public_key })
    }

    pub(crate) fn compile_program(
        &self,
        program: &PowerLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggEncodingWire>,
        rhs: &BTreeMap<RhsInputId, PowerRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, EncodingSelectorFamily>,
        values: &BTreeMap<crate::program::PublicValueFamilyId, PowerLutMonomialFamily>,
        helpers: &FlatLutHelperMap,
    ) -> Result<BTreeMap<ProgramWireId, BggEncodingWire>, PowerLutError> {
        let mut ranges = ProgramFamilyRanges::new();
        for (id, family) in selectors {
            ranges.selector(
                *id,
                FamilyRange::full(family.count().clone())
                    .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?,
            );
        }
        for (id, family) in values {
            ranges.public_values(
                *id,
                FamilyRange::full(family.count().clone())
                    .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?,
            );
        }
        let bindings = ProgramBindings::new(inputs, rhs, selectors, values, helpers);
        lower_program(program, &bindings, &ranges, self)
    }

    pub(crate) fn compile_program_with_ranges(
        &self,
        program: &PowerLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggEncodingWire>,
        rhs: &BTreeMap<RhsInputId, PowerRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, EncodingSelectorFamily>,
        values: &BTreeMap<crate::program::PublicValueFamilyId, PowerLutMonomialFamily>,
        ranges: &ProgramFamilyRanges,
        helpers: &FlatLutHelperMap,
    ) -> Result<BTreeMap<ProgramWireId, BggEncodingWire>, PowerLutError> {
        let bindings = ProgramBindings::new(inputs, rhs, selectors, values, helpers);
        lower_program(program, &bindings, ranges, self)
    }

    /// Performs setup-fixed Fuse: `c * G^{-1}(C)`.
    pub fn fuse(
        &self,
        lhs: &BggEncodingWire,
        rhs: &PowerRhsPackage,
    ) -> Result<BggEncodingWire, PowerLutError> {
        crate::ensure_ciphertext_only(lhs)?;
        // `decomposed = G^{-1}(C)` is applied to both the vector and public
        // matrix, preserving the BGG relation while switching the payload.
        let decomposed = rhs
            .gsw_ciphertext()
            .clone()
            .decompose(self.bgg.public_key.base.clone(), self.bgg.public_key.digit_count.clone());
        let public = lhs.pubkey.matrix.clone().mul_decomposed(decomposed.clone());
        Ok(BggEncodingWire {
            vector: lhs.vector.clone().mul_decomposed(decomposed),
            pubkey: BggPublicKeyWire { matrix: public, reveal_plaintext: false },
            plaintext: None,
        })
    }

    /// Evaluates one LUT with the flat helper set. Helpers are in canonical
    /// `sigma_j = 1 + j * (2n/W)` order.
    pub fn single_input_lut(
        &self,
        input: &BggEncodingWire,
        table: &[usize],
        helpers: &[FlatLutHelper],
    ) -> Result<BggEncodingWire, PowerLutError> {
        let width = table.len();
        let n = input
            .pubkey
            .matrix
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let table = LutTable::unary(width, n, table.to_vec()).map_err(PowerLutError::from)?;
        self.single_input_lut_table(input, &table, helpers)
    }

    /// Lowers a complete unary table, preserving its monomial/scalar output
    /// form.  The public slice API above is intentionally monomial-only.
    pub(crate) fn single_input_lut_table(
        &self,
        input: &BggEncodingWire,
        table: &LutTable,
        helpers: &[FlatLutHelper],
    ) -> Result<BggEncodingWire, PowerLutError> {
        crate::ensure_ciphertext_only(input)?;
        let width = table.values().len();
        let n = input
            .pubkey
            .matrix
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        if width == 0 ||
            !width.is_power_of_two() ||
            width > n ||
            n % width != 0 ||
            helpers.len() != width ||
            table.values().iter().any(|value| *value >= n)
        {
            return Err(PowerLutError::InvalidLut);
        }
        // `raw_vectors[j]` and `raw_publics[j]` are the two input components
        // after `X -> X^{\sigma_j}`, before applying branch `C_j` and its mask.
        let raw_vectors = Family::pack(
            helpers
                .iter()
                .enumerate()
                .map(|(j, helper)| {
                    let expected = canonical_sigma(j, n, width).ok_or(PowerLutError::InvalidLut)?;
                    if helper.sigma() != expected {
                        return Err(PowerLutError::InvalidLut);
                    }
                    Ok(input.vector.clone().ring_automorphism(expected))
                })
                .collect::<Result<Vec<_>, PowerLutError>>()?,
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let raw_publics = Family::pack(
            helpers
                .iter()
                .enumerate()
                .map(|(j, helper)| {
                    let expected = canonical_sigma(j, n, width).ok_or(PowerLutError::InvalidLut)?;
                    if helper.sigma() != expected {
                        return Err(PowerLutError::InvalidLut);
                    }
                    Ok(input.pubkey.matrix.clone().ring_automorphism(expected))
                })
                .collect::<Result<Vec<_>, PowerLutError>>()?,
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let switches = Family::pack(
            helpers.iter().map(|helper| helper.switch().gsw_ciphertext().clone()).collect(),
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let mask_vectors =
            Family::pack(helpers.iter().map(|helper| helper.mask().vector.clone()).collect())
                .map_err(|_| PowerLutError::InvalidLut)?;
        let mask_publics = Family::pack(
            helpers.iter().map(|helper| helper.mask().pubkey.matrix.clone()).collect(),
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let (vectors, publics) = Family::try_parallel_zip_many_values(
            vec![raw_vectors, raw_publics, switches, mask_vectors, mask_publics],
            move |_index, mut items| {
                let mask_public = items.pop().ok_or(DslError::Schema)?;
                let mask_vector = items.pop().ok_or(DslError::Schema)?;
                let switch = items.pop().ok_or(DslError::Schema)?;
                let raw_public = items.pop().ok_or(DslError::Schema)?;
                let raw_vector = items.pop().ok_or(DslError::Schema)?;
                let rhs = PowerRhsPackage::new(switch).map_err(|_| DslError::Schema)?;
                // The same digit matrix is used in both components of Fuse.
                let c_decomposition = rhs.gsw_ciphertext().clone().decompose(
                    self.bgg.public_key.base.clone(),
                    self.bgg.public_key.digit_count.clone(),
                );
                let switched_vector = raw_vector.mul_decomposed(c_decomposition.clone());
                let switched_public = raw_public.mul_decomposed(c_decomposition);
                let a_decomposition = switched_public.decompose(
                    self.bgg.public_key.base.clone(),
                    self.bgg.public_key.digit_count.clone(),
                );
                Ok((
                    mask_vector.mul_decomposed(a_decomposition.clone()) + switched_vector,
                    mask_public.mul_decomposed(a_decomposition),
                ))
            },
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        Ok(BggEncodingWire {
            vector: balanced_sum_family(vectors)?,
            pubkey: BggPublicKeyWire {
                matrix: balanced_sum_family(publics)?,
                reveal_plaintext: false,
            },
            plaintext: None,
        })
    }

    pub fn two_input_lut(
        &self,
        lhs: &BggEncodingWire,
        rhs: &PowerRhsPackage,
        lhs_width: usize,
        rhs_width: usize,
        table: &[usize],
        helpers: &[FlatLutHelper],
    ) -> Result<BggEncodingWire, PowerLutError> {
        if lhs_width == 0 ||
            rhs_width == 0 ||
            table.len() != lhs_width.checked_mul(rhs_width).ok_or(PowerLutError::InvalidLut)?
        {
            return Err(PowerLutError::InvalidLut);
        }
        self.single_input_lut(&self.fuse(lhs, rhs)?, table, helpers)
    }

    fn two_input_lut_table(
        &self,
        lhs: &BggEncodingWire,
        rhs: &PowerRhsPackage,
        table: &LutTable,
        helpers: &[FlatLutHelper],
    ) -> Result<BggEncodingWire, PowerLutError> {
        let rhs_width = table.rhs_width().ok_or(PowerLutError::InvalidLut)?;
        if table.input_width() == 0 ||
            rhs_width == 0 ||
            table.output_form() != LutOutputForm::Monomial
        {
            return Err(PowerLutError::InvalidLut);
        }
        let fused = self.fuse(lhs, rhs)?;
        self.single_input_lut_table(&fused, table, helpers)
    }
}

impl ProgramLoweringBackend for PowerLutEncodingCompiler {
    type Wire = BggEncodingWire;
    type Rhs = PowerRhsPackage;
    type SelectorFamily = EncodingSelectorFamily;
    type PublicValueFamily = PowerLutMonomialFamily;
    type Helper = FlatLutHelper;
    type HelperSet = FlatLutHelperSet;

    fn resolve_helpers<'a>(
        &self,
        helpers: &'a Self::HelperSet,
        table: &crate::program::LutTable,
    ) -> Result<&'a [Self::Helper], PowerLutError> {
        helpers.resolve(table)
    }
    fn unary(
        &self,
        input: Self::Wire,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError> {
        if table.rhs_width().is_some() {
            return Err(PowerLutError::InvalidLut);
        }
        self.single_input_lut_table(&input, table, helpers)
    }
    fn binary(
        &self,
        lhs: Self::Wire,
        rhs: &Self::Rhs,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError> {
        self.two_input_lut_table(&lhs, rhs, table, helpers)
    }
    fn one_hot_select(
        &self,
        input: Self::Wire,
        selectors: &Self::SelectorFamily,
        public_values: &Self::PublicValueFamily,
        selector_range: &FamilyRange,
        public_value_range: &FamilyRange,
    ) -> Result<Self::Wire, PowerLutError> {
        // For selector entries `C_i` and public values `v_i`, compute
        // `sum_i m_i * Fuse(input,C_i) * v_i`; `indices` and `masks` relocate
        // the logical selector range into one-hot coefficients.
        if !public_values.has_provenance(crate::program::PBC_MONOMIAL_FAMILY_PROVENANCE) ||
            selector_range != public_value_range ||
            selectors.count() != public_values.count()
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        crate::ensure_ciphertext_only(&input)?;
        let capacity = selector_range
            .capacity()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let count = mxx_dsl::Int::evaluate(selector_range.count().clone());
        let start = mxx_dsl::Int::evaluate(selector_range.start().clone());
        let (indices, masks) =
            one_hot_indices_and_masks(capacity, count, start, public_values.element_type().clone())
                .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let cs = selectors
            .gsw()
            .clone()
            .parallel_gather(indices.clone())
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let vals = public_values
            .as_family()
            .clone()
            .parallel_gather(indices)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let weighted = Family::try_parallel_zip_many_values(
            vec![cs, vals, masks],
            move |_index, mut items| {
                let mask = items.pop().ok_or(DslError::Schema)?;
                let value = items.pop().ok_or(DslError::Schema)?;
                let c = items.pop().ok_or(DslError::Schema)?;
                let rhs = PowerRhsPackage::new(c).map_err(|_| DslError::Schema)?;
                let fused = self.fuse(&input, &rhs).map_err(|_| DslError::Schema)?;
                let weighted = value * mask;
                Ok((fused.vector * weighted.clone(), fused.pubkey.matrix * weighted))
            },
        )
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let (vectors, publics) = weighted;
        let vector = balanced_sum_family(vectors)?;
        let public = balanced_sum_family(publics)?;
        Ok(BggEncodingWire {
            vector,
            pubkey: BggPublicKeyWire { matrix: public, reveal_plaintext: false },
            plaintext: None,
        })
    }
}

/// Family of fixed GSW ciphertexts used by the one-hot lowering.
#[derive(Clone)]
pub struct EncodingSelectorFamily {
    gsw: Family<Mat>,
}
impl EncodingSelectorFamily {
    pub fn new(gsw: Family<Mat>) -> Result<Self, PowerLutError> {
        if *gsw.count() == IntExpr::constant(0) {
            Err(PowerLutError::InvalidSparseLwrBlock)
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
    pub(crate) fn from_flattened(mut values: Vec<Family<Mat>>) -> Result<Self, PowerLutError> {
        if values.len() != 1 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Self::new(values.remove(0))
    }
}

pub fn flattened_lut_index(
    u: usize,
    v: usize,
    lhs_width: usize,
    rhs_width: usize,
) -> Option<usize> {
    (lhs_width > 0 && rhs_width > 0 && u < lhs_width && v < rhs_width).then(|| u + lhs_width * v)
}

/// Returns the canonical odd automorphism used by flat branch `j`.
pub(crate) fn canonical_sigma(j: usize, ring_dimension: usize, width: usize) -> Option<usize> {
    (width != 0 && width.is_power_of_two() && ring_dimension % width == 0 && j < width)
        .then(|| 1 + j * (2 * ring_dimension / width))
}

fn rotation_power(ring: &mxx_dsl::Ring, exponent: usize, ring_dimension: usize) -> Mat {
    // In R_q = Z_q[X]/(X^n+1), `X^k` is a signed permutation. Reducing the
    // exponent modulo `2n` records the sign flip for exponents crossing n.
    let exponent = exponent % (2 * ring_dimension);
    let reduced = exponent % ring_dimension;
    let rotation = ring.constant((1, 1), ConstantMatrix::Rotation { exponent: reduced.into() });
    if exponent < ring_dimension { rotation } else { -rotation }
}

fn lut_coefficient(
    ring: &mxx_dsl::Ring,
    n: usize,
    width: usize,
    sigma: usize,
    table: &[usize],
    output_form: LutOutputForm,
) -> Result<Mat, PowerLutSamplingError> {
    // The inverse `W^{-1}` is in `Z_q`; the output form decides whether each
    // table entry contributes a scalar or a monomial rotation.
    let modulus =
        ring.zero((1, 1)).matrix_type().modulus.evaluate(&ParamEnv::default()).map_err(|_| {
            PowerLutSamplingError::InvalidConfiguration("ring modulus must be concrete")
        })?;
    let inverse = modular_inverse(&(BigInt::from(width) % &modulus), &modulus)
        .ok_or(PowerLutSamplingError::InvalidConfiguration("LUT width is not invertible"))?;
    let mut result = ring.zero((1, 1));
    for (k, output) in table.iter().copied().enumerate() {
        let left = match output_form {
            LutOutputForm::Monomial => rotation_power(ring, output, n),
            LutOutputForm::Scalar => ring.polynomial([IntExpr::constant(output)]),
        };
        let neg_k = (2 * n - k % (2 * n)) % (2 * n);
        let right = rotation_power(ring, (neg_k * sigma) % (2 * n), n);
        result = result + left * right;
    }
    Ok(ring.polynomial([inverse.into()]) * result)
}

pub(crate) fn one_hot_indices_and_masks(
    capacity: usize,
    count: mxx_dsl::Int,
    start: mxx_dsl::Int,
    mask_type: mxx_ir_core::types::MatrixType,
) -> Result<(Family<mxx_dsl::Int>, Family<Mat>), DslError> {
    Parallel::range(capacity).try_map_values(|index| {
        let offset = index.as_int();
        let active = offset.clone().less_equal(count.clone().sub(mxx_dsl::Int::constant(1)));
        Ok((
            start.clone().add(offset.clone().rem(count.clone())),
            active.to_int().lift_to_constant_polynomial(mask_type.clone()),
        ))
    })
}

pub(crate) fn modular_inverse(value: &BigInt, modulus: &BigInt) -> Option<BigInt> {
    use num_traits::{One, Zero};
    let mut old_r = value.clone();
    let mut r = modulus.clone();
    let mut old_s = BigInt::one();
    let mut s = BigInt::zero();
    while !r.is_zero() {
        let quotient = &old_r / &r;
        (old_r, r) = (r.clone(), old_r - &quotient * &r);
        (old_s, s) = (s.clone(), old_s - quotient * &s);
    }
    (old_r == BigInt::one()).then(|| ((old_s % modulus) + modulus) % modulus)
}

pub(crate) fn balanced_sum_family(family: Family<Mat>) -> Result<Mat, PowerLutError> {
    let count = family
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(PowerLutError::InvalidLut)?;
    if count == 0 {
        return Err(PowerLutError::InvalidLut);
    }
    let mut current = family;
    let mut count = count;
    while count > 1 {
        let next = count.div_ceil(2);
        let left = current
            .clone()
            .parallel_gather(
                Parallel::range(next)
                    .map_values(|i| i.as_int().mul(mxx_dsl::Int::constant(2)))
                    .map_err(|_| PowerLutError::InvalidLut)?,
            )
            .map_err(|_| PowerLutError::InvalidLut)?;
        let right = current
            .clone()
            .parallel_gather(
                Parallel::range(next)
                    .map_values(|i| {
                        i.as_int()
                            .mul(mxx_dsl::Int::constant(2))
                            .add(mxx_dsl::Int::constant(1))
                            .rem(mxx_dsl::Int::constant(count))
                    })
                    .map_err(|_| PowerLutError::InvalidLut)?,
            )
            .map_err(|_| PowerLutError::InvalidLut)?;
        current =
            Family::try_parallel_zip_many_values(vec![left, right], move |index, mut items| {
                let l = items.pop().ok_or(DslError::Schema)?;
                let r = items.pop().ok_or(DslError::Schema)?;
                if count % 2 == 1 {
                    let last = index.as_int().equal(mxx_dsl::Int::constant(next - 1)).to_int();
                    last.select(vec![l.clone() + r, l])
                } else {
                    Ok(l + r)
                }
            })
            .map_err(|_| PowerLutError::InvalidLut)?;
        count = next;
    }
    Ok(current.get_static(0))
}

// Artifact import boundary for ordinary encoding wires.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BggEncodingArtifactNames {
    pub vector: String,
    pub public_matrix: String,
}
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PowerArtifactImportError {
    #[error("artifact production mismatch")]
    ProductionMismatch,
    #[error("artifact missing")]
    MissingArtifact,
    #[error("artifact confidentiality mismatch")]
    ConfidentialityMismatch,
    #[error("artifact matrix type mismatch")]
    MatrixTypeMismatch,
    #[error("artifact metadata invalid")]
    InvalidMetadata,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestEncodingMetadata {
    secret: ManifestSecretMetadata,
    role: serde_json::Value,
}
pub fn artifact_input(
    production_id: mxx_ir_core::artifact::ProductionId,
    manifest: &mxx_ir_core::artifact::Manifest,
    names: BggEncodingArtifactNames,
) -> Result<BggEncodingWire, PowerArtifactImportError> {
    artifact_input_with_role(production_id, manifest, names, None::<&serde_json::Value>)
}
pub(crate) fn artifact_input_with_role<R: Serialize>(
    production_id: mxx_ir_core::artifact::ProductionId,
    manifest: &mxx_ir_core::artifact::Manifest,
    names: BggEncodingArtifactNames,
    expected_role: Option<&R>,
) -> Result<BggEncodingWire, PowerArtifactImportError> {
    if manifest.production_id != production_id {
        return Err(PowerArtifactImportError::ProductionMismatch);
    }
    let vector =
        manifest.artifacts.get(&names.vector).ok_or(PowerArtifactImportError::MissingArtifact)?;
    let public = manifest
        .artifacts
        .get(&names.public_matrix)
        .ok_or(PowerArtifactImportError::MissingArtifact)?;
    let metadata: ManifestEncodingMetadata = serde_json::from_str(
        vector.layout.as_deref().ok_or(PowerArtifactImportError::InvalidMetadata)?,
    )
    .map_err(|_| PowerArtifactImportError::InvalidMetadata)?;
    let public_metadata: ManifestEncodingMetadata = serde_json::from_str(
        public.layout.as_deref().ok_or(PowerArtifactImportError::InvalidMetadata)?,
    )
    .map_err(|_| PowerArtifactImportError::InvalidMetadata)?;
    if expected_role
        .map(serde_json::to_value)
        .transpose()
        .map_err(|_| PowerArtifactImportError::InvalidMetadata)?
        .is_some_and(|role| metadata.role != role || public_metadata.role != role) ||
        metadata.secret.identity != public_metadata.secret.identity
    {
        return Err(PowerArtifactImportError::InvalidMetadata);
    }
    let layout = metadata.secret.sampler();
    let modulus = layout
        .modulus
        .evaluate(&ParamEnv::default())
        .map_err(|_| PowerArtifactImportError::MatrixTypeMismatch)?;
    let n = layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(PowerArtifactImportError::MatrixTypeMismatch)?;
    if vector.confidentiality != mxx_ir_core::artifact::ArtifactConfidentiality::Private ||
        public.confidentiality != mxx_ir_core::artifact::ArtifactConfidentiality::Public ||
        vector.artifact_type !=
            mxx_ir_core::artifact::ArtifactType::Matrix(
                mxx_ir_core::types::ConcreteMatrixType {
                    modulus: modulus.clone(),
                    ring_dimension: n,
                    rows: 1,
                    columns: layout.public_key_columns(),
                },
            ) ||
        public.artifact_type !=
            mxx_ir_core::artifact::ArtifactType::Matrix(
                mxx_ir_core::types::ConcreteMatrixType {
                    modulus,
                    ring_dimension: n,
                    rows: layout.secret_dimension,
                    columns: layout.public_key_columns(),
                },
            )
    {
        return Err(PowerArtifactImportError::MatrixTypeMismatch);
    }
    Ok(BggEncodingWire {
        vector: layout.ring().artifact_input(
            production_id.clone(),
            names.vector,
            (1, layout.public_key_columns()),
            mxx_ir_core::artifact::ArtifactConfidentiality::Private,
        ),
        pubkey: BggPublicKeyWire {
            matrix: layout.ring().artifact_input(
                production_id,
                names.public_matrix,
                (layout.secret_dimension, layout.public_key_columns()),
                mxx_ir_core::artifact::ArtifactConfidentiality::Public,
            ),
            reveal_plaintext: false,
        },
        plaintext: None,
    })
}
#[cfg(test)]
mod tests {
    use super::{
        FlatLutHelper, FlatLutHelperSet, PowerLutEncodingCompiler, PowerLutEncodingSampler,
        canonical_sigma,
    };
    use crate::{program::LutTable, public_key::PowerLutPublicKeySampler, rhs::PowerRhsPackage};
    use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
    use mxx_dsl::{DslContext, HashTag, Ring};
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[test]
    fn canonical_sigma_has_expected_order() {
        assert_eq!(
            (0..4).filter_map(|j| canonical_sigma(j, 16, 4)).collect::<Vec<_>>(),
            [1, 9, 17, 25]
        );
        assert!(canonical_sigma(4, 16, 4).is_none());
        assert!(canonical_sigma(0, 12, 4).is_some());
        assert!(canonical_sigma(0, 10, 4).is_none());
    }

    #[test]
    fn encoding_sampler_batches_and_core_rejects_invalid_bindings() {
        let ring = Ring::new(97, 4);
        let sampler = PowerLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let mask = ring.input("batch-mask", (1, 2));
        let plaintexts = [ring.identity(1), ring.polynomial([1.into()])];
        let hash_key = ring.bytes_input("batch-hash", 32);
        let batch = sampler
            .sample_input_encodings(
                mask.clone(),
                None,
                hash_key.clone(),
                b"batch-inputs".as_slice(),
                &plaintexts,
            )
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|encoding| encoding.plaintext.is_none()));
        let public_keys = mxx_bgg::BggPublicKeySampler { layout: sampler.layout.clone() }
            .sample(hash_key.clone(), b"batch-inputs".as_slice(), &[false, false])
            .into_iter()
            .skip(1)
            .collect::<Vec<_>>();
        let independently_bound = sampler
            .sample_encodings_for_public_matrices(mask, None, &public_keys, &plaintexts)
            .unwrap();
        assert_eq!(
            batch[0].pubkey.matrix.matrix_type(),
            independently_bound[0].pubkey.matrix.matrix_type()
        );
        let public_keys_again = PowerLutPublicKeySampler { layout: sampler.layout.clone() }
            .sample_input_keys(hash_key, b"batch-inputs".as_slice(), 2)
            .unwrap();
        for index in 0..2 {
            assert_eq!(
                batch[index].pubkey.matrix.value_handle().node().kind(),
                independently_bound[index].pubkey.matrix.value_handle().node().kind()
            );
            assert_eq!(
                batch[index].pubkey.matrix.value_handle().node().kind(),
                public_keys_again[index].matrix.value_handle().node().kind()
            );
        }
        assert!(
            sampler
                .sample_encodings_for_public_matrices(
                    ring.input("mismatch-mask", (1, 2)),
                    None,
                    &public_keys,
                    &plaintexts[..1],
                )
                .is_err()
        );
        let revealing_key = mxx_bgg::BggPublicKeySampler { layout: sampler.layout.clone() }
            .sample(ring.bytes_input("reveal-hash", 32), b"reveal".as_slice(), &[true])
            .into_iter()
            .nth(1)
            .unwrap();
        assert!(
            sampler
                .sample_encodings_for_public_matrices(
                    ring.input("reveal-mask", (1, 2)),
                    None,
                    std::slice::from_ref(&revealing_key),
                    std::slice::from_ref(&plaintexts[0]),
                )
                .is_err()
        );
    }

    #[test]
    fn input_batch_public_matrices_match_in_runtime_order() {
        let parameters = DCRTPolyParams::new(4, 1, 20, 4);
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let layout = mxx_bgg::BggSamplerLayout {
            modulus: modulus.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: 16.into(),
        };
        let ring = layout.ring();
        let sampler = PowerLutEncodingSampler {
            layout: layout.clone(),
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let public_sampler = PowerLutPublicKeySampler { layout };
        let hash_key = ring.bytes_input("batch-runtime-hash", 32);
        let plaintexts = [ring.polynomial([1.into()]), ring.polynomial([2.into()])];
        let mask_secret = ring.input("batch-runtime-mask", (1, 2));
        let encoded = sampler
            .sample_input_encodings(
                mask_secret,
                None,
                hash_key.clone(),
                b"batch-runtime".as_slice(),
                &plaintexts,
            )
            .unwrap();
        let public =
            public_sampler.sample_input_keys(hash_key, b"batch-runtime".as_slice(), 2).unwrap();
        let mut context = DslContext::new("batch-runtime-public-equivalence");
        for index in 0..2 {
            context = context
                .output(format!("encoded-{index}"), encoded[index].pubkey.matrix.clone())
                .unwrap()
                .output(format!("public-{index}"), public[index].matrix.clone())
                .unwrap();
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::from([(
                "batch-runtime-hash".to_owned(),
                RuntimeValue::Bytes(vec![0x63; 32]),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for index in 0..2 {
            let RuntimeValue::Matrix(encoded_matrix) = &result.outputs[&format!("encoded-{index}")]
            else {
                panic!("encoded matrix output")
            };
            let RuntimeValue::Matrix(public_matrix) = &result.outputs[&format!("public-{index}")]
            else {
                panic!("public matrix output")
            };
            assert_eq!(
                encoded_matrix.as_ref(),
                public_matrix.as_ref(),
                "batch member {index} differs"
            );
        }
    }

    #[test]
    fn fixed_rhs_top_is_domain_separated_and_hash_derived() {
        let ring = Ring::new(97, 4);
        let sampler = PowerLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let secret = ring.input("fixed-rhs-secret", (1, 2));
        let target = ring.input("fixed-rhs-target", (1, 2));
        let payload = ring.input("fixed-rhs-payload", (1, 2));
        let hash_key = ring.bytes_input("fixed-rhs-hash", 32);
        let first = sampler
            .sample_cross_secret_rhs(
                secret.clone(),
                target.clone(),
                payload.clone(),
                hash_key.clone(),
                b"fixed-rhs-test".as_slice(),
            )
            .unwrap();
        let repeated = sampler
            .sample_cross_secret_rhs(
                secret.clone(),
                target.clone(),
                payload.clone(),
                hash_key.clone(),
                b"fixed-rhs-test".as_slice(),
            )
            .unwrap();
        let different_tag = sampler
            .sample_cross_secret_rhs(
                secret,
                target,
                payload,
                hash_key.clone(),
                b"fixed-rhs-other".as_slice(),
            )
            .unwrap();
        let top_kind = |rhs: &PowerRhsPackage| {
            rhs.gsw_ciphertext()
                .value_handle()
                .node()
                .arguments()
                .first()
                .expect("fixed-RHS top row")
                .node()
                .kind()
                .clone()
        };
        let mut expected_tag = HashTag::from(b"fixed-rhs-test".as_slice());
        expected_tag.push("power-lut/fixed-rhs/top/v1");
        let expected = ring.hash_matrix(
            hash_key,
            expected_tag,
            (sampler.layout.secret_dimension - 1, sampler.layout.public_key_columns()),
        );
        assert_eq!(top_kind(&first), expected.value_handle().node().kind().clone());
        assert_eq!(top_kind(&first), top_kind(&repeated));
        assert_ne!(top_kind(&first), top_kind(&different_tag));
    }

    #[test]
    fn fixed_fuse_uses_one_c_decomposition_for_vector_and_public_matrix() {
        let ring = Ring::new(97, 4);
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let lhs = BggEncodingWire {
            vector: ring.zero((1, 2)),
            pubkey: BggPublicKeyWire { matrix: ring.zero((2, 2)), reveal_plaintext: false },
            plaintext: None,
        };
        let rhs = PowerRhsPackage::new(ring.zero((2, 2))).unwrap();
        let output = compiler.fuse(&lhs, &rhs).unwrap();
        let c_decomposition = rhs.gsw_ciphertext().clone().decompose(4, 2);
        assert_eq!(
            output.vector.matrix_type(),
            ring.zero((1, 2)).mul_decomposed(c_decomposition.clone()).matrix_type()
        );
        assert_eq!(
            output.pubkey.matrix.matrix_type(),
            ring.zero((2, 2)).mul_decomposed(c_decomposition).matrix_type()
        );
        assert!(matches!(
            output.vector.value_handle().node().kind(),
            mxx_ir_core::node::NodeKind::ApplyPreimage
        ));
        assert!(matches!(
            output.pubkey.matrix.value_handle().node().kind(),
            mxx_ir_core::node::NodeKind::ApplyPreimage
        ));
    }

    #[test]
    fn helper_set_rejects_a_different_lut_table() {
        let ring = Ring::new(97, 4);
        let rhs = PowerRhsPackage::new(ring.zero((2, 2))).unwrap();
        let mask = BggEncodingWire {
            vector: ring.zero((1, 2)),
            pubkey: BggPublicKeyWire { matrix: ring.zero((2, 2)), reveal_plaintext: false },
            plaintext: None,
        };
        let helper = FlatLutHelper::new(1, rhs, mask).unwrap();
        let table = LutTable::unary(1, 2, vec![0]).unwrap();
        let other = LutTable::unary(1, 2, vec![1]).unwrap();
        let same_values_scalar = LutTable::unary_scalar(1, 2, vec![0]).unwrap();
        let set = FlatLutHelperSet::new(&table, vec![helper]).unwrap();
        assert!(set.resolve(&table).is_ok());
        assert!(set.resolve(&other).is_err());
        // The output algebra is part of the commitment: a scalar helper set
        // must never be accepted for an otherwise identical monomial table.
        assert!(set.resolve(&same_values_scalar).is_err());
    }

    #[test]
    fn flat_sampler_supports_shared_and_distinct_payload_secrets() {
        let ring = Ring::new(97, 4);
        let sampler = super::PowerLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let mask = ring.input("flat-mask", (1, 2));
        let payload = ring.input("flat-payload", (1, 2));
        let hash_key = ring.bytes_input("flat-hash", 32);
        let table = LutTable::unary(2, 4, vec![0, 1]).unwrap();
        let bank = sampler
            .sample_flat_mask_bank(mask.clone(), hash_key.clone(), 2, b"flat-bank".as_slice())
            .unwrap();
        let shared = sampler
            .sample_flat_helpers_for_lut(
                mask.clone(),
                None,
                hash_key.clone(),
                &table,
                bank.as_ref(),
                b"shared".as_slice(),
            )
            .unwrap();
        let distinct = sampler
            .sample_flat_helpers_for_lut(
                mask,
                Some(payload),
                hash_key,
                &table,
                bank.as_ref(),
                b"distinct".as_slice(),
            )
            .unwrap();
        assert_eq!(shared.len(), 2);
        assert_eq!(distinct.len(), 2);
        assert_ne!(
            shared[0].switch().gsw_ciphertext().value_handle().node(),
            distinct[0].switch().gsw_ciphertext().value_handle().node()
        );
    }

    #[test]
    fn flat_mask_bank_reuses_branches_across_lut_widths() {
        let ring = Ring::new(97, 16);
        let sampler = super::PowerLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 16.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let mask = ring.input("mask-bank-secret", (1, 2));
        let hash_key = ring.bytes_input("mask-bank-hash", 32);
        let bank = sampler
            .sample_flat_mask_bank(mask.clone(), hash_key.clone(), 4, b"mask-bank".as_slice())
            .unwrap();
        assert_eq!(bank.sigmas().collect::<Vec<_>>(), [1, 9, 17, 25]);
        let public_bank = PowerLutPublicKeySampler { layout: sampler.layout.clone() }
            .sample_flat_mask_bank(hash_key.clone(), 4, b"mask-bank".as_slice())
            .unwrap();
        for (index, sigma) in bank.sigmas().enumerate() {
            let private_matrix = &bank.branch(index).unwrap().mask.pubkey.matrix;
            let public_matrix = public_bank.matrix(index).unwrap();
            assert_eq!(
                private_matrix.value_handle().node().kind(),
                public_matrix.value_handle().node().kind()
            );
            assert_eq!(private_matrix.matrix_type(), public_matrix.matrix_type());
            assert_eq!(sigma, bank.branch(index).unwrap().sigma);
        }
        let wide = LutTable::unary(4, 16, vec![0, 1, 2, 3]).unwrap();
        let same_width = LutTable::unary(4, 16, vec![3, 2, 1, 0]).unwrap();
        let narrow = LutTable::unary(2, 16, vec![0, 1]).unwrap();
        let wide_helpers = sampler
            .sample_flat_helpers_for_lut(
                mask.clone(),
                None,
                hash_key.clone(),
                &wide,
                bank.as_ref(),
                b"wide".as_slice(),
            )
            .unwrap();
        let same_width_helpers = sampler
            .sample_flat_helpers_for_lut(
                mask.clone(),
                None,
                hash_key.clone(),
                &same_width,
                bank.as_ref(),
                b"same-width".as_slice(),
            )
            .unwrap();
        let narrow_helpers = sampler
            .sample_flat_helpers_for_lut(
                mask,
                None,
                hash_key,
                &narrow,
                bank.as_ref(),
                b"narrow".as_slice(),
            )
            .unwrap();
        assert_eq!(
            wide_helpers[0].mask().pubkey.matrix.value_handle().node(),
            narrow_helpers[0].mask().pubkey.matrix.value_handle().node()
        );
        assert_eq!(
            wide_helpers[2].mask().pubkey.matrix.value_handle().node(),
            narrow_helpers[1].mask().pubkey.matrix.value_handle().node()
        );
        assert_eq!(
            wide_helpers[0].mask().pubkey.matrix.value_handle().node(),
            same_width_helpers[0].mask().pubkey.matrix.value_handle().node()
        );
        assert_ne!(
            wide_helpers[0].switch().gsw_ciphertext().value_handle().node(),
            same_width_helpers[0].switch().gsw_ciphertext().value_handle().node()
        );
        let wide_set = FlatLutHelperSet::new(&wide, wide_helpers).unwrap();
        let same_width_set = FlatLutHelperSet::new(&same_width, same_width_helpers).unwrap();
        assert_ne!(wide_set.metadata().0, same_width_set.metadata().0);
        assert_ne!(
            wide_set.metadata().0,
            FlatLutHelperSet::new(&narrow, narrow_helpers).unwrap().metadata().0
        );
    }

    #[test]
    fn flat_lut_branches_are_lowered_in_one_structural_parallel_loop() {
        let ring = Ring::new(97, 4);
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let input = BggEncodingWire {
            vector: ring.input("flat-input-vector", (1, 2)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("flat-input-public", (2, 2)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let make_helper = |sigma| {
            FlatLutHelper::new(
                sigma,
                PowerRhsPackage::new(ring.zero((2, 2))).unwrap(),
                BggEncodingWire {
                    vector: ring.zero((1, 2)),
                    pubkey: BggPublicKeyWire { matrix: ring.zero((2, 2)), reveal_plaintext: false },
                    plaintext: None,
                },
            )
            .unwrap()
        };
        let output =
            compiler.single_input_lut(&input, &[0, 1], &[make_helper(1), make_helper(5)]).unwrap();
        let graph = DslContext::new("flat-lut-structural-loop")
            .output("vector", output.vector)
            .unwrap()
            .build()
            .unwrap();
        let nodes = graph.graph.scopes().values().flat_map(|scope| scope.nodes());
        let mut parallel_loops = 0;
        for node in nodes {
            match node.kind() {
                mxx_ir_core::node::NodeKind::ParallelGrid(_) => {
                    parallel_loops += 1;
                }
                _ => {}
            }
        }
        assert!(parallel_loops >= 1, "flat branches must use a structural family loop");
    }
}
