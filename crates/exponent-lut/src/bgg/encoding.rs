//! Private Exponent-LUT evaluation over ordinary `BggEncodingWire` values.
//!
//! This module implements only the setup-fixed RHS interface. A fixed GSW
//! ciphertext is gadget-decomposed once and multiplied by the input vector;
//! no encoding of individual GSW digits is constructed. LUT evaluation uses
//! one setup-fixed switch and one reusable mask alignment per canonical
//! automorphism branch, followed by one balanced sum.
//!
//! With `c = s A - μ t G + e` and a fixed GSW ciphertext satisfying
//! `t C = y v G + e_C`, setup-fixed Fuse computes
//! `D = G^{-1}(C)` and `cD = s(A D) - μ y v G + e'`. For unary table `f`,
//! branch `j` uses the layout-derived odd automorphism and
//! `D_{sigma_j,L} = W'^{-1} sum_k mono_m(L(k)) sigma_j(mono_m(-k))`.
//! The branch is mask-aligned and the `W'` branches are added.

use super::*;
use std::{collections::BTreeMap, sync::Arc};

#[cfg(test)]
use crate::program::{
    FamilyRange, ProgramBindings, ProgramFamilyBindings, ProgramInputId, ProgramValidationError,
    RhsId, WireId,
};
use crate::{
    bgg::{
        BggError, ExponentRhsPackage, ExponentRhsPackageError,
        public_key::{ProgramCacheBindings, ProgramLutCache},
    },
    noise_refresh::{
        prf::SparseLwrPrfProgram,
        refresh::{RefreshPrfContract, RefreshPrfFamilyMaterial, aggregate_refresh_prf},
        setup::{ImportedRefreshSetup, RefreshSetupError},
    },
    program::{
        ExponentLutProgram, ExponentProgramLowering, LutTable, ProgramGateInstance,
        ProgramLoweringTypes,
    },
    utils::{balanced_sum_family, derive_lut_layout, lut_coefficient, same_matrix_type},
};
use mxx_bgg::{BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{Bytes, DslError, Family, HashTag, Int, Mat, Preimage};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    node::{ConcatAxis, IndexRange},
};
use num_traits::ToPrimitive;
use thiserror::Error;

/// Validated online BGG carrier/plaintext family consumed by family lowering.
///
/// This is encoding-domain runtime data, not part of the host-side program
/// representation. Public-key matrices never cross this boundary because
/// multiplication consumes preprocessing-produced decomposition artifacts.
#[cfg(test)]
#[derive(Clone)]
pub struct BggEncodingFamily {
    vector: Family<Mat>,
    plaintext: Family<Mat>,
}

#[cfg(test)]
impl BggEncodingFamily {
    pub(crate) fn from_components(
        vector: Family<Mat>,
        plaintext: Family<Mat>,
        ring: &mxx_dsl::Ring,
        public_key_columns: usize,
    ) -> Result<Self, ProgramValidationError> {
        let ring_type = ring.matrix_type((1, 1));
        let vector_type = vector.element_type();
        let count = vector
            .count()
            .evaluate(&ParamEnv::default())
            .map_err(|_| ProgramValidationError::InvalidWireFamily)?;
        let plaintext_count = plaintext
            .count()
            .evaluate(&ParamEnv::default())
            .map_err(|_| ProgramValidationError::InvalidWireFamily)?;
        if public_key_columns == 0 ||
            vector_type.modulus != ring_type.modulus ||
            vector_type.ring_dimension != ring_type.ring_dimension ||
            vector_type.rows != IntExpr::constant(1) ||
            vector_type.columns != IntExpr::constant(public_key_columns) ||
            *plaintext.element_type() != ring_type ||
            count <= 0.into() ||
            plaintext_count != count ||
            vector.count() != plaintext.count()
        {
            return Err(ProgramValidationError::InvalidWireFamily);
        }
        Ok(Self { vector, plaintext })
    }

    pub(crate) fn plaintext_family(&self) -> &Family<Mat> {
        &self.plaintext
    }

    pub(crate) fn vector_family(&self) -> &Family<Mat> {
        &self.vector
    }
}

/// A setup-fixed mask-alignment branch, reusable across LUT tables.
#[derive(Clone)]
pub struct LutMaskBank {
    branches: Arc<Vec<LutMaskBranch>>,
}

#[derive(Clone)]
struct LutMaskBranch {
    sigma: usize,
    mask: BggEncodingWire,
}

/// Public projections of a reusable [`LutMaskBank`].
#[derive(Clone)]
pub struct LutPublicMaskBank {
    branches: Vec<(usize, Mat)>,
}

impl LutMaskBank {
    fn single(sigma: usize, mask: BggEncodingWire) -> Self {
        Self { branches: Arc::new(vec![LutMaskBranch { sigma, mask }]) }
    }

    pub(crate) fn index_for_sigma(&self, sigma: usize) -> Option<usize> {
        self.branches.iter().position(|branch| branch.sigma == sigma)
    }

    fn branch(&self, index: usize) -> Option<&LutMaskBranch> {
        self.branches.get(index)
    }

    /// Returns the ordered odd automorphisms covered by this bank.
    pub fn sigmas(&self) -> impl Iterator<Item = usize> + '_ {
        self.branches.iter().map(|branch| branch.sigma)
    }
}

impl LutPublicMaskBank {
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
pub(crate) fn canonical_lut_mask_branch_tag(root: &HashTag, sigma: impl Into<IntExpr>) -> HashTag {
    let mut tag = root.clone();
    tag.push("exponent-lut-native-mask-bank-v1");
    tag.push("mask");
    tag.push(sigma.into());
    tag
}

/// Setup-fixed helper for one native LUT branch.
#[derive(Clone)]
pub struct LutHelper {
    sigma: usize,
    #[cfg(test)]
    switch: ExponentRhsPackage,
    mask_bank: Arc<LutMaskBank>,
    mask_index: usize,
}

/// All setup-fixed branches for one concrete LUT table.
///
/// The commitment is metadata for artifact binding only; it is never carried
/// by a BGG encoding wire or used as a cryptographic provenance check during
/// ordinary lowering.
#[derive(Clone, PartialEq)]
pub struct LutHelperSet<H = LutHelper> {
    table_commitment: [u8; 32],
    width: usize,
    helpers: Vec<H>,
}

impl mxx_dsl::GraphValue for LutHelperSet {
    type Schema = LutHelperSet<(
        usize,
        mxx_bgg::BggEncodingType,
        Option<ExponentRhsPackage<mxx_dsl::MatType>>,
    )>;
    fn flatten(&self) -> Vec<mxx_ir_core::ValueHandle> {
        self.helpers
            .iter()
            .flat_map(|helper| {
                let switch: Option<ExponentRhsPackage> = {
                    #[cfg(test)]
                    {
                        Some(helper.switch.clone())
                    }
                    #[cfg(not(test))]
                    {
                        None
                    }
                };
                mxx_dsl::GraphValue::flatten(&(helper.mask().clone(), switch))
            })
            .collect()
    }
    fn schema(&self) -> Self::Schema {
        LutHelperSet {
            table_commitment: self.table_commitment,
            width: self.width,
            helpers: self
                .helpers
                .iter()
                .map(|helper| {
                    let switch: Option<ExponentRhsPackage> = {
                        #[cfg(test)]
                        {
                            Some(helper.switch.clone())
                        }
                        #[cfg(not(test))]
                        {
                            None
                        }
                    };
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
                let (mask, switch) = <(BggEncodingWire, Option<ExponentRhsPackage>)>::from_values(
                    &fields,
                    values.get(offset..offset + count).ok_or(mxx_dsl::DslError::Schema)?,
                )?;
                offset += count;
                #[cfg(not(test))]
                let _ = switch;
                Ok(LutHelper {
                    sigma: *sigma,
                    #[cfg(test)]
                    switch: switch.ok_or(mxx_dsl::DslError::Schema)?,
                    mask_bank: Arc::new(LutMaskBank::single(*sigma, mask)),
                    mask_index: 0,
                })
            })
            .collect::<Result<Vec<_>, mxx_dsl::DslError>>()?;
        if offset != values.len() {
            return Err(mxx_dsl::DslError::Schema);
        }
        Ok(Self { table_commitment: schema.table_commitment, width: schema.width, helpers })
    }
}
impl mxx_dsl::GraphValueSchema
    for LutHelperSet<(
        usize,
        mxx_bgg::BggEncodingType,
        Option<ExponentRhsPackage<mxx_dsl::MatType>>,
    )>
{
    type Value = LutHelperSet;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        use mxx_dsl::GraphValue;
        let values = self
            .helpers
            .iter()
            .flat_map(|(_, mask, switch)| {
                (mask.clone(), switch.clone()).placeholders_from(next).flatten()
            })
            .collect::<Vec<_>>();
        LutHelperSet::from_values(self, &values).expect("helper schema placeholders")
    }
    fn wire_types(&self) -> Vec<mxx_ir_core::types::WireType> {
        self.helpers
            .iter()
            .flat_map(|(_, mask, switch)| (mask.clone(), switch.clone()).wire_types())
            .collect()
    }
}

impl LutHelperSet {
    pub fn new(
        table: &crate::program::LutTable,
        helpers: Vec<LutHelper>,
    ) -> Result<Self, BggError> {
        if table.values().len() != helpers.len() {
            return Err(BggError::InvalidLut);
        }
        Ok(Self { table_commitment: table.commitment(), width: helpers.len(), helpers })
    }

    pub(crate) fn resolve(
        &self,
        table: &crate::program::LutTable,
    ) -> Result<&[LutHelper], BggError> {
        if self.width != table.values().len() || self.table_commitment != table.commitment() {
            return Err(BggError::InvalidLut);
        }
        Ok(&self.helpers)
    }

    #[cfg(test)]
    pub(crate) fn as_slice(&self) -> &[LutHelper] {
        &self.helpers
    }
    #[cfg(test)]
    pub(crate) fn metadata(&self) -> ([u8; 32], usize) {
        (self.table_commitment, self.width)
    }
}

impl LutHelper {
    #[cfg(test)]
    pub(crate) fn new(
        sigma: usize,
        switch: ExponentRhsPackage,
        mask: BggEncodingWire,
    ) -> Result<Self, BggError> {
        if sigma == 0 || sigma % 2 == 0 {
            return Err(BggError::InvalidLut);
        }
        Self::with_mask_bank(sigma, switch, Arc::new(LutMaskBank::single(sigma, mask)))
    }
    pub(crate) fn with_mask_bank(
        sigma: usize,
        switch: ExponentRhsPackage,
        mask_bank: Arc<LutMaskBank>,
    ) -> Result<Self, BggError> {
        let mask_index = mask_bank.index_for_sigma(sigma).ok_or(BggError::InvalidLut)?;
        #[cfg(not(test))]
        let _ = switch;
        Ok(Self {
            sigma,
            #[cfg(test)]
            switch,
            mask_bank,
            mask_index,
        })
    }
    pub(crate) fn sigma(&self) -> usize {
        self.sigma
    }
    #[cfg(test)]
    pub(crate) fn switch(&self) -> &ExponentRhsPackage {
        &self.switch
    }
    pub(crate) fn mask(&self) -> &BggEncodingWire {
        &self.mask_bank.branch(self.mask_index).expect("validated mask index").mask
    }
}

/// Runtime helper bindings keyed by the exact LUT identity used by a gate.
pub type LutHelperMap = BTreeMap<crate::program::LutId, LutHelperSet>;

/// Structural public setup material for non-identity native readout branches.
/// Both families share canonical sigma order. Public masks are independent of
/// payload-secret switches, so online preparation captures no switch sampler.
#[derive(Clone)]
pub struct NativeReadoutHelperFamilies {
    pub sigmas: Arc<Vec<usize>>,
    pub switch_ciphertexts: Family<Mat>,
    pub public_masks: Family<Mat>,
}

/// Errors raised while constructing setup-time Exponent-LUT inputs and helpers.
#[derive(Debug, Error)]
pub enum ExponentLutSamplingError {
    #[error(transparent)]
    Bgg(#[from] mxx_bgg::BggSampleError),
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error(transparent)]
    ExponentLut(#[from] BggError),
    #[error(transparent)]
    Rhs(#[from] ExponentRhsPackageError),
    #[error("invalid Exponent-LUT sampler configuration: {0}")]
    InvalidConfiguration(&'static str),
}

/// Setup sampler for input encodings and native LUT helpers.
#[derive(Clone)]
pub struct ExponentLutEncodingSampler {
    pub layout: mxx_bgg::BggSamplerLayout,
    pub exponent_layout: crate::layout::ExpLayout,
    pub gaussian_sigma: Option<mxx_ir_core::RealExpr>,
    pub gaussian_max_coefficient_bound: Option<IntExpr>,
}

/// Compiler for private Exponent-LUT graphs over ordinary BGG+ wires.
#[derive(Clone)]
pub struct ExponentLutEncodingCompiler {
    pub bgg: BggEncodingCompiler,
    pub exponent_layout: crate::layout::ExpLayout,
    pub ring: mxx_dsl::Ring,
    pub gadget_base: IntExpr,
    pub digit_count: IntExpr,
}

impl ExponentLutEncodingSampler {
    pub fn sample_native_readout_helper_families(
        &self,
        setup_secret: Mat,
        hash_key: Bytes,
        width: usize,
        mask_tag: impl Into<HashTag>,
        helper_tag: impl Into<HashTag>,
    ) -> Result<NativeReadoutHelperFamilies, ExponentLutSamplingError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutSamplingError::InvalidConfiguration(
                "ring dimension must be concrete",
            ))?;
        let lut_layout =
            derive_lut_layout(&self.exponent_layout, width, n).map_err(BggError::from)?;
        if width <= 1 {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
                "native readout requires at least two branches",
            ));
        }
        let sigmas = lut_layout.automorphism_indices(n).map_err(|_| {
            ExponentLutSamplingError::InvalidConfiguration("invalid automorphism layout")
        })?;
        if sigmas.first() != Some(&1) {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
                "canonical native readout must begin with identity",
            ));
        }
        let helper_sigmas = Arc::new(sigmas[1..].to_vec());
        let mask_root = mask_tag.into();
        let mut helper_root = helper_tag.into();
        helper_root.push("exponent-lut-lut-v1");
        let sampler = self.clone();
        let helper_count = helper_sigmas.len();
        let public_masks = mxx_dsl::parallel(helper_count, |index| {
            Ok({
                let sigma = lut_layout
                    .automorphism_index_expression(
                        n,
                        IntExpr::Add(Box::new(index.expression()?), Box::new(IntExpr::constant(1))),
                    )
                    .expect("validated canonical readout automorphisms");
                let mask_key = mxx_bgg::BggPublicKeySampler { layout: sampler.layout.clone() }
                    .sample(
                        hash_key.clone(),
                        canonical_lut_mask_branch_tag(&mask_root, sigma.clone()),
                        &[false],
                    )
                    .into_iter()
                    .nth(1)
                    .expect("validated structural mask sample");
                mask_key.matrix
            })
        })?;
        let switch_ciphertexts = mxx_dsl::parallel(helper_count, |index| {
            Ok({
                let sigma = lut_layout
                    .automorphism_index_expression(
                        n,
                        IntExpr::Add(Box::new(index.expression()?), Box::new(IntExpr::constant(1))),
                    )
                    .expect("validated canonical readout automorphisms");
                // ValueReadout already multiplies the input by T'. Its trace
                // must sum unweighted automorphism branches, so this helper
                // changes the payload secret from sigma(t) to t with value 1.
                // Ordinary LUT helpers separately retain their LUT coefficients.
                let mut switch_tag = helper_root.clone();
                switch_tag.push("switch");
                switch_tag.push(IntExpr::Add(
                    Box::new(index.expression()?),
                    Box::new(IntExpr::constant(1)),
                ));
                let switch = sampler
                    .sample_fixed_rhs(
                        setup_secret.clone().ring_automorphism(sigma),
                        setup_secret.clone(),
                        sampler.layout.ring().identity(1),
                        hash_key.clone(),
                        switch_tag,
                    )
                    .expect("validated structural readout switch");
                switch.gsw_ciphertext().clone()
            })
        })
        .map_err(ExponentLutSamplingError::Dsl)?;
        Ok(NativeReadoutHelperFamilies { sigmas: helper_sigmas, switch_ciphertexts, public_masks })
    }

    /// Online preparation consumes the fixed public mask family and s only.
    /// Neither payload-secret switch ciphertexts nor their sampler are captured.
    pub(crate) fn sample_native_readout_masks(
        &self,
        online_secret: Mat,
        public_masks: &Family<Mat>,
    ) -> Result<Family<Mat>, ExponentLutSamplingError> {
        let concrete = |value: &IntExpr| {
            value.evaluate(&ParamEnv::default()).ok().and_then(|value| value.to_usize()).ok_or(
                ExponentLutSamplingError::InvalidConfiguration(
                    "native readout dimensions must be concrete",
                ),
            )
        };
        let n = concrete(&self.layout.ring_dimension)?;
        let width = concrete(public_masks.count())? + 1;
        let lut_layout =
            derive_lut_layout(&self.exponent_layout, width, n).map_err(BggError::from)?;
        Ok(mxx_dsl::parallel(public_masks.count().clone(), |index| {
            let sigma = lut_layout
                .automorphism_index_expression(
                    n,
                    IntExpr::Add(Box::new(index.expression()?), Box::new(IntExpr::constant(1))),
                )
                .expect("validated canonical readout automorphisms");
            let constant = BggPublicKeyWire {
                matrix: self
                    .layout
                    .ring()
                    .zero((self.layout.secret_dimension, self.layout.public_key_columns())),
                reveal_plaintext: false,
            };
            let mask_key =
                BggPublicKeyWire { matrix: public_masks.at(index), reveal_plaintext: false };
            let mask = mxx_bgg::BggEncodingSampler {
                layout: self.layout.clone(),
                gaussian_sigma: self.gaussian_sigma.clone(),
                gaussian_max_coefficient_bound: self.gaussian_max_coefficient_bound.clone(),
            }
            .sample(
                online_secret.clone(),
                Some(online_secret.clone().ring_automorphism(sigma)),
                &[constant, mask_key],
                &[self.layout.ring().identity(1)],
            )
            .expect("validated structural online mask sample")
            .into_iter()
            .nth(1)
            .expect("validated structural online mask");
            Ok(mask.vector)
        })?)
    }

    /// Samples an augmented secret `(s_bar,1)` using private randomness.
    pub fn sample_secret(&self) -> Result<Mat, ExponentLutSamplingError> {
        if self.layout.secret_dimension < 2 {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
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
    ) -> Result<Vec<BggEncodingWire>, ExponentLutSamplingError> {
        if plaintexts.is_empty() {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
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
    ) -> Result<Vec<BggEncodingWire>, ExponentLutSamplingError> {
        if public_keys.is_empty() || public_keys.len() != plaintexts.len() {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
                "public-key and plaintext counts must be equal and positive",
            ));
        }
        if public_keys.iter().any(|key| key.reveal_plaintext) {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
                "public keys must be ciphertext-only",
            ));
        }
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutSamplingError::InvalidConfiguration(
                "ring dimension must be concrete",
            ))?;
        self.exponent_layout.validate(ring_dimension).map_err(|_| {
            ExponentLutSamplingError::InvalidConfiguration("invalid exponent layout")
        })?;
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
    pub fn sample_mask_bank(
        &self,
        mask_secret: Mat,
        hash_key: Bytes,
        max_width: usize,
        tag: impl Into<HashTag>,
    ) -> Result<Arc<LutMaskBank>, ExponentLutSamplingError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutSamplingError::InvalidConfiguration(
                "ring dimension must be concrete",
            ))?;
        self.exponent_layout.validate(n).map_err(|_| {
            ExponentLutSamplingError::InvalidConfiguration("invalid exponent layout")
        })?;
        if max_width == 0 ||
            !max_width.is_power_of_two() ||
            max_width > n ||
            n % max_width != 0 ||
            self.exponent_layout.width() < max_width
        {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
                "mask-bank width must be a power of two dividing the ring dimension",
            ));
        }
        let bank_tag = tag.into();
        let mut branches = Vec::with_capacity(max_width);
        let bank_layout = crate::layout::ExpLayout {
            width_log2: max_width.trailing_zeros(),
            ..self.exponent_layout
        };
        let sigmas = bank_layout.automorphism_indices(n).map_err(|_| {
            ExponentLutSamplingError::InvalidConfiguration("invalid automorphism layout")
        })?;
        for (_j, sigma) in sigmas.into_iter().enumerate() {
            let mask_tag = canonical_lut_mask_branch_tag(&bank_tag, sigma);
            let mask_key = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() }
                .sample(hash_key.clone(), mask_tag, &[false])
                .into_iter()
                .nth(1)
                .ok_or(ExponentLutSamplingError::InvalidConfiguration("mask sample is empty"))?;
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
            .ok_or(ExponentLutSamplingError::InvalidConfiguration("mask sample is empty"))?;
            branches.push(LutMaskBranch { sigma, mask });
        }
        Ok(Arc::new(LutMaskBank { branches: Arc::new(branches) }))
    }

    /// Samples setup-fixed helpers for a complete LUT declaration using an
    /// explicitly shared mask bank. The bank must cover every canonical sigma
    /// required by this table; it is never sampled implicitly here.
    ///
    /// The declaration is the source of truth for both helper commitments and
    /// the branch coefficient. Every table entry is a native strided monomial.
    /// For a fused LUT, `rhs` enters each plaintext target as
    /// `d_sigma * sigma(rhs)` before sampling its fresh switch ciphertext.
    /// The switch error is sampled afterward and is not multiplied by `rhs`.
    pub fn sample_helpers_for_lut(
        &self,
        mask_secret: Mat,
        payload_secret: Option<Mat>,
        hash_key: Bytes,
        table: &LutTable,
        mask_bank: &LutMaskBank,
        tag: impl Into<HashTag>,
        rhs: Option<Mat>,
    ) -> Result<Vec<LutHelper>, ExponentLutSamplingError> {
        if table.rhs_width().is_some() != rhs.is_some() {
            return Err(ExponentLutSamplingError::InvalidConfiguration("LUT RHS arity mismatch"));
        }
        if rhs
            .as_ref()
            .is_some_and(|rhs| rhs.matrix_type() != &self.layout.ring().matrix_type((1, 1)))
        {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
                "LUT RHS ring or shape mismatch",
            ));
        }
        let width = table.values().len();
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutSamplingError::InvalidConfiguration(
                "ring dimension must be concrete",
            ))?;
        self.exponent_layout.validate(n).map_err(|_| {
            ExponentLutSamplingError::InvalidConfiguration("invalid exponent layout")
        })?;
        if width == 0 ||
            !width.is_power_of_two() ||
            width > n ||
            n % width != 0 ||
            table
                .values()
                .iter()
                .any(|value| *value >= self.exponent_layout.effective_modulus(n))
        {
            return Err(ExponentLutSamplingError::InvalidConfiguration(
                "LUT width or output exponent is invalid",
            ));
        }
        let payload = payload_secret.unwrap_or_else(|| mask_secret.clone());
        let shared_mask_bank = Arc::new(mask_bank.clone());
        let mut tag = tag.into();
        tag.push("exponent-lut-lut-v1");
        let mut helpers = Vec::with_capacity(width);
        let lut_layout =
            crate::layout::ExpLayout { width_log2: width.trailing_zeros(), ..self.exponent_layout };
        let sigmas = lut_layout.automorphism_indices(n).map_err(|_| {
            ExponentLutSamplingError::InvalidConfiguration("invalid automorphism layout")
        })?;
        for (j, sigma) in sigmas.into_iter().enumerate() {
            // `j` chooses one Fourier branch; `sigma` is odd so its ring
            // automorphism is invertible and follows the canonical LUT order.
            let coefficient =
                lut_coefficient(&self.layout.ring(), &lut_layout, sigma, table.values())
                    .map_err(BggError::from)?;
            let coefficient = match &rhs {
                Some(rhs) => coefficient * rhs.clone().ring_automorphism(sigma),
                None => coefficient,
            };
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
                return Err(ExponentLutSamplingError::InvalidConfiguration(
                    "mask bank does not cover LUT canonical sigma",
                ));
            }
            helpers.push(LutHelper::with_mask_bank(sigma, switch, shared_mask_bank.clone())?);
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
    ) -> Result<ExponentRhsPackage, ExponentLutSamplingError> {
        let ring = self.layout.ring();
        let columns = self.layout.public_key_columns();
        // For mask secret `s`, payload secret `t`, and scalar `y`, construct
        // `C = [R; y*t*G - s*R + e_C]`, so `[s,1]C = y*t*G + e_C`.
        let mut top_tag = tag.clone();
        top_tag.push("exponent-lut/fixed-rhs/top/v1");
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
                return Err(ExponentLutSamplingError::InvalidConfiguration(
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
        ExponentRhsPackage::new(ciphertext).map_err(Into::into)
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
    ) -> Result<ExponentRhsPackage, ExponentLutSamplingError> {
        self.sample_fixed_rhs(source, target, payload, hash_key, tag.into())
    }
}

impl ExponentLutEncodingCompiler {
    pub fn new(public_key: BggPublicKeyCompiler) -> Self {
        Self {
            bgg: BggEncodingCompiler,
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 },
            ring: public_key.ring,
            gadget_base: public_key.base,
            digit_count: public_key.digit_count,
        }
    }
    pub fn with_layout(
        public_key: BggPublicKeyCompiler,
        exponent_layout: crate::layout::ExpLayout,
    ) -> Self {
        Self { exponent_layout, ..Self::new(public_key) }
    }
    pub fn from_public_key(public_key: BggPublicKeyCompiler) -> Self {
        Self::new(public_key)
    }

    /// Compiles a program using only the supplied typed public preimage
    /// bindings for multiplication. A non-empty production cache is therefore
    /// consumed directly by the graph; no online decomposition is emitted.
    #[cfg(test)]
    pub(crate) fn compile_program_with_ranges_and_cache(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggEncodingWire>,
        rhs: &BTreeMap<RhsId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, EncodingRhsFamily>,
        values: &BTreeMap<crate::program::WireFamilyId, BggEncodingFamily>,
        ranges: &ProgramFamilyBindings,
        helpers: &LutHelperMap,
        cache: &ProgramCacheBindings,
    ) -> Result<BTreeMap<WireId, BggEncodingWire>, BggError> {
        self.compile_program_wrapped(
            program,
            inputs,
            rhs,
            selectors,
            values,
            ranges,
            helpers,
            Some(cache),
        )
    }

    #[cfg(test)]
    fn compile_program_wrapped(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggEncodingWire>,
        rhs: &BTreeMap<RhsId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, EncodingRhsFamily>,
        values: &BTreeMap<crate::program::WireFamilyId, BggEncodingFamily>,
        ranges: &ProgramFamilyBindings,
        helpers: &LutHelperMap,
        cache: Option<&ProgramCacheBindings>,
    ) -> Result<BTreeMap<WireId, BggEncodingWire>, BggError> {
        let rhs = rhs.keys().map(|id| (*id, ())).collect();
        let selectors = selectors
            .iter()
            .map(|(id, family)| {
                Ok((
                    *id,
                    (
                        family.gsw().field(|_| ())?,
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
                        family.vector_family().clone().zip_map(
                            family.plaintext_family().clone(),
                            |vector, plaintext| {
                                Ok(BggEncodingWire { vector, plaintext: Some(plaintext) })
                            },
                        )?,
                        FamilyRange::full(family.vector_family().count().clone())
                            .map_err(|_| mxx_dsl::DslError::Schema)?,
                    ),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, mxx_dsl::DslError>>()?;
        let helpers = helpers
            .iter()
            .map(|(id, value)| (*id, crate::program::ProgramValue::Scalar(value.clone())))
            .collect();
        let prepared = crate::scheme::PreparedProgram {
            scheme: BggScheme::new(
                BggPublicKeyCompiler {
                    ring: self.ring.clone(),
                    base: self.gadget_base.clone(),
                    digit_count: self.digit_count.clone(),
                },
                self.exponent_layout,
            ),
            program: program.clone(),
            ranges: ranges.clone(),
            cache: cache.cloned().unwrap_or_default(),
            shared_cache: Default::default(),
            family_caches: BTreeMap::new(),
            public_values: BTreeMap::new(),
        };
        prepared
            .evaluate(&ProgramBindings::new(inputs, &rhs, &selectors, &values, &helpers, None))
            .map_err(Into::into)
    }

    #[cfg(all(test, feature = "gpu"))]
    fn test_single_input_lut(
        &self,
        input: &BggEncodingWire,
        values: &[usize],
        helpers: &[LutHelper],
        cache: &ProgramLutCache,
    ) -> Result<BggEncodingWire, BggError> {
        let ring_dimension = input
            .vector
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(BggError::InvalidLut)?;
        let table = LutTable::unary(
            values.len(),
            self.exponent_layout.effective_modulus(ring_dimension),
            values.to_vec(),
        )?;
        self.single_input_lut_table_with_cache(input, &table, helpers, Some(cache))
    }

    fn single_input_lut_table_with_cache(
        &self,
        input: &BggEncodingWire,
        table: &LutTable,
        helpers: &[LutHelper],
        cached: Option<&ProgramLutCache>,
    ) -> Result<BggEncodingWire, BggError> {
        self.single_input_lut_table_components(
            &input.vector,
            input.plaintext.as_ref(),
            table,
            helpers,
            cached,
        )
    }

    fn single_input_lut_table_components(
        &self,
        input_vector: &Mat,
        input_plaintext: Option<&Mat>,
        table: &LutTable,
        helpers: &[LutHelper],
        cached: Option<&ProgramLutCache>,
    ) -> Result<BggEncodingWire, BggError> {
        let output = self.single_input_lut_table_parts(
            input_vector,
            input_plaintext,
            table,
            helpers,
            cached,
        )?;
        Ok(BggEncodingWire { vector: output.vector, plaintext: output.plaintext })
    }

    fn single_input_lut_table_parts(
        &self,
        input_vector: &Mat,
        input_plaintext: Option<&Mat>,
        table: &LutTable,
        helpers: &[LutHelper],
        cached: Option<&ProgramLutCache>,
    ) -> Result<LutOutputParts, BggError> {
        let width = table.values().len();
        let n = input_vector
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(BggError::InvalidLut)?;
        let lut_layout =
            derive_lut_layout(&self.exponent_layout, width, n).map_err(BggError::from)?;
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
        let coefficients = sigmas
            .iter()
            .map(|sigma| {
                lut_coefficient(&self.ring, &lut_layout, *sigma, table.values())
                    .map_err(|_| BggError::InvalidLut)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // `raw_vectors[j]` and `raw_publics[j]` are the two input components
        // after `X -> X^{\sigma_j}`, before applying branch `C_j` and its mask.
        let raw_vectors = Family::pack(
            helpers
                .iter()
                .enumerate()
                .map(|(j, helper)| {
                    let expected = sigmas.get(j).copied().ok_or(BggError::InvalidLut)?;
                    if helper.sigma() != expected {
                        return Err(BggError::InvalidLut);
                    }
                    Ok(input_vector.clone().ring_automorphism(expected))
                })
                .collect::<Result<Vec<_>, BggError>>()?,
        )
        .map_err(|_| BggError::InvalidLut)?;
        let cache = cached.ok_or(BggError::InvalidSparseLwrBlock)?;
        if cache.switch.len() != width || cache.branch.len() != width {
            return Err(BggError::InvalidSparseLwrBlock);
        }
        let mask_vectors =
            Family::pack(helpers.iter().map(|helper| helper.mask().vector.clone()).collect())
                .map_err(|_| BggError::InvalidLut)?;
        let vectors = {
            let source_families = (
                raw_vectors,
                Family::pack(cache.switch.clone())?,
                mask_vectors,
                Family::pack(cache.branch.clone())?,
            );
            if source_families.0.count() != source_families.1.count() ||
                source_families.0.count() != source_families.2.count() ||
                source_families.0.count() != source_families.3.count()
            {
                Err(mxx_dsl::DslError::Schema)
            } else {
                mxx_dsl::parallel(source_families.0.count().clone(), |family_index| {
                    let (raw, switch, mask, branch) = (
                        source_families.0.at(&family_index),
                        source_families.1.at(&family_index),
                        source_families.2.at(&family_index),
                        source_families.3.at(&family_index),
                    );
                    let _ = family_index;
                    Ok(branch.mul_small_rhs(mask) + switch.mul_small_rhs(raw))
                })
            }
        }
        .map_err(|_| BggError::InvalidLut)?;
        let plaintext = input_plaintext.cloned().map(|value| {
            let branches = sigmas
                .iter()
                .zip(coefficients.iter())
                .map(|(sigma, coefficient)| {
                    value.clone().ring_automorphism(*sigma) * coefficient.clone()
                })
                .collect::<Vec<_>>();
            balanced_sum_family(Family::pack(branches).expect("validated LUT branch family"))
                .expect("validated LUT plaintext branch family")
        });
        Ok(LutOutputParts { vector: balanced_sum_family(vectors)?, plaintext })
    }
}

pub struct EncodingProgramLowering<'a> {
    pub(crate) shared_cache: Option<&'a ProgramCacheBindings>,
    pub(crate) compiler: &'a ExponentLutEncodingCompiler,
    pub(crate) cache: Option<&'a ProgramCacheBindings>,
}

impl ProgramLoweringTypes for EncodingProgramLowering<'_> {
    type Wire = BggEncodingWire;
    type Rhs = ();
    type HelperSet = LutHelperSet;
    type Error = BggError;
}

impl ExponentProgramLowering for EncodingProgramLowering<'_> {
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let decomposition = self
            .cache
            .and_then(|cache| cache.gate_entry(gate))
            .or_else(|| self.shared_cache.and_then(|cache| cache.gate_entry(gate)))
            .ok_or(BggError::InvalidSparseLwrBlock)?;
        let plaintext = public.plaintext.clone().ok_or(BggError::InvalidSparseLwrBlock)?;
        Ok(BggEncodingWire {
            vector: decomposition.clone().mul_small_rhs(public.vector.clone()) +
                plaintext * private.vector.clone(),
            plaintext: program_binary_plaintext(public, private, |left, right| left * right),
        })
    }

    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        _: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        self.compiler.bgg.add(lhs, rhs).map_err(Into::into)
    }

    fn mul_private(
        &mut self,
        lhs: &Self::Wire,
        _: &Self::Rhs,
        _: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let decomposition = self
            .cache
            .and_then(|cache| cache.gate_entry(gate))
            .or_else(|| self.shared_cache.and_then(|cache| cache.gate_entry(gate)))
            .ok_or(BggError::InvalidSparseLwrBlock)?;
        Ok(BggEncodingWire {
            vector: decomposition.clone().mul_small_rhs(lhs.vector.clone()),
            plaintext: None,
        })
    }

    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: crate::program::LutId,
        input: &Self::Wire,
        helpers: Option<&Self::HelperSet>,
        rhs: Option<&Self::Rhs>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let helpers = helpers.ok_or(BggError::InvalidLut)?;
        let table = program.lut(lut).ok_or(BggError::InvalidLut)?;
        let helpers = helpers.resolve(table)?;
        let cache = self
            .cache
            .and_then(|cache| cache.lut_entry(gate))
            .or_else(|| self.shared_cache.and_then(|cache| cache.lut_entry(gate)))
            .ok_or(BggError::InvalidSparseLwrBlock)?;
        // A fused private RHS suppresses unary public-plaintext metadata.
        let input = BggEncodingWire {
            vector: input.vector.clone(),
            plaintext: if rhs.is_some() { None } else { input.plaintext.clone() },
        };
        self.compiler.single_input_lut_table_with_cache(&input, table, helpers, Some(cache))
    }
}

#[derive(Clone)]
struct LutOutputParts {
    vector: Mat,
    plaintext: Option<Mat>,
}

fn program_binary_plaintext(
    lhs: &BggEncodingWire,
    rhs: &BggEncodingWire,
    operation: impl FnOnce(Mat, Mat) -> Mat,
) -> Option<Mat> {
    lhs.plaintext.clone().zip(rhs.plaintext.clone()).map(|(lhs, rhs)| operation(lhs, rhs))
}

/// Setup-fixed family of GSW ciphertexts consumed by both public-key
/// preprocessing and online encoding evaluation.
#[derive(Clone)]
pub struct SetupRhsFamily {
    gsw: Family<Mat>,
    layout_id: Option<crate::noise_refresh::pbc::PbcLayoutId>,
}
impl SetupRhsFamily {
    pub fn new(gsw: Family<Mat>) -> Result<Self, BggError> {
        if *gsw.count() == IntExpr::constant(0) {
            Err(BggError::InvalidSparseLwrBlock)
        } else {
            Ok(Self { gsw, layout_id: None })
        }
    }

    pub fn new_for_layout(
        gsw: Family<Mat>,
        layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    ) -> Result<Self, BggError> {
        let mut selector = Self::new(gsw)?;
        selector.layout_id = Some(layout_id);
        Ok(selector)
    }

    pub(crate) fn layout_id(&self) -> Option<crate::noise_refresh::pbc::PbcLayoutId> {
        self.layout_id
    }
    pub(crate) fn gsw(&self) -> &Family<Mat> {
        &self.gsw
    }
}

pub type EncodingRhsFamily = SetupRhsFamily;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ExponentArtifactImportError {
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
#[cfg(test)]
mod tests {
    use super::{
        BggEncodingFamily, ExponentLutEncodingCompiler, ExponentLutEncodingSampler, LutHelper,
        LutHelperSet,
    };
    use crate::{
        bgg::{
            ExponentRhsPackage,
            public_key::{
                ExponentLutPublicKeyCompiler, LutPublicHelper, LutPublicHelperSet,
                ProgramCacheBindings, ProgramLutCache, PublicKeyWireFamily,
            },
        },
        program::{
            AttributeKnowledge, ExponentLutProgramBuilder, FamilyProgram, FamilyRange,
            FamilyWireSource, Gate, LutTable, ProgramBindings, ProgramFamilyBindings,
            ProgramFamilyId, ProgramGateInstance, ProgramInputId, ProgramWireType, RhsId, WireId,
        },
    };
    use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler};
    use mxx_dsl::{DslContext, Family, HashTag, Mat, Ring};
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use num_traits::ToPrimitive;
    use std::collections::BTreeMap;

    #[test]
    fn bgg_encoding_family_rejects_wrong_ring_and_empty_count() {
        let ring = Ring::new(97, 4);
        let valid = Family::pack(vec![ring.zero((1, 5))]).unwrap();
        let valid_plaintext = Family::pack(vec![ring.zero((1, 1))]).unwrap();
        assert!(BggEncodingFamily::from_components(valid, valid_plaintext, &ring, 5).is_ok());

        let wrong_ring = Ring::new(97, 8);
        let wrong_ring_family = Family::pack(vec![ring.zero((1, 1))]).unwrap();
        assert!(
            BggEncodingFamily::from_components(
                wrong_ring_family.clone(),
                wrong_ring_family,
                &wrong_ring,
                1,
            )
            .is_err()
        );

        let empty = Family::pack(Vec::<Mat>::new());
        assert!(
            empty.is_err() || {
                let empty = empty.unwrap();
                BggEncodingFamily::from_components(empty.clone(), empty, &ring, 1).is_err()
            }
        );
    }

    /// Mirrors the public-key preprocessing pass for a single LUT without
    /// introducing an online decomposition fallback into the production
    /// backend. The tests below exercise plaintext and structural lowering, so
    /// a zero public-key matrix is sufficient while retaining the exact
    /// production shapes and cache dataflow.
    fn preprocess_lut_cache(
        compiler: &ExponentLutEncodingCompiler,
        helpers: &[LutHelper],
    ) -> ProgramLutCache {
        let public_input = compiler.ring.zero((2, 4));
        let mut switch = Vec::with_capacity(helpers.len());
        let mut branch = Vec::with_capacity(helpers.len());
        for helper in helpers {
            let switch_decomposition = helper
                .switch()
                .gsw_ciphertext()
                .clone()
                .decompose(compiler.gadget_base.clone(), compiler.digit_count.clone());
            let switched = switch_decomposition
                .clone()
                .mul_small_rhs(public_input.clone().ring_automorphism(helper.sigma()));
            switch.push(switch_decomposition);
            branch.push(
                switched.decompose(compiler.gadget_base.clone(), compiler.digit_count.clone()),
            );
        }
        ProgramLutCache { switch, branch }
    }

    #[test]
    fn layout_indices_have_expected_order() {
        assert_eq!(
            crate::layout::ExpLayout { stride_log2: 0, width_log2: 2 }
                .automorphism_indices(16)
                .unwrap(),
            [1, 9, 17, 25]
        );
    }

    #[test]
    fn encoding_sampler_batches_and_core_rejects_invalid_bindings() {
        let sampler = ExponentLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        // The sampler's layout owns the plaintext modulus.  Construct every
        // binding in that same ring so the BGG sampler's typed matrix checks
        // exercise the intended invalid-binding cases below.
        let ring = sampler.layout.ring();
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
        assert_eq!(batch[0].vector.matrix_type(), independently_bound[0].vector.matrix_type());
        for index in 0..2 {
            assert_eq!(
                batch[index].vector.value_handle().node().kind(),
                independently_bound[index].vector.value_handle().node().kind()
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
    fn fixed_rhs_top_is_domain_separated_and_hash_derived() {
        let ring = Ring::new(97, 4);
        let sampler = ExponentLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 },
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
        let top_kind = |rhs: &ExponentRhsPackage| {
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
        expected_tag.push("exponent-lut/fixed-rhs/top/v1");
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
    fn helper_set_rejects_a_different_lut_table() {
        let ring = Ring::new(97, 4);
        let rhs = ExponentRhsPackage::new(ring.zero((2, 2))).unwrap();
        let mask = BggEncodingWire { vector: ring.zero((1, 2)), plaintext: None };
        let helper = LutHelper::new(1, rhs, mask).unwrap();
        let table = LutTable::unary(1, 2, vec![0]).unwrap();
        let other = LutTable::unary(1, 2, vec![1]).unwrap();
        let different_values = LutTable::unary(1, 2, vec![1]).unwrap();
        let set = LutHelperSet::new(&table, vec![helper]).unwrap();
        assert!(set.resolve(&table).is_ok());
        assert!(set.resolve(&other).is_err());
        assert!(set.resolve(&different_values).is_err());
    }

    #[test]
    fn public_lut_lowering_propagates_native_plaintext() {
        let parameters = DCRTPolyParams::new(4, 1, 20, 4, None, None);
        let ring = Ring::new(BigInt::from(parameters.modulus().as_ref().clone()), 4);
        let compiler = ExponentLutEncodingCompiler::with_layout(
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() },
            crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 },
        );
        let input = BggEncodingWire {
            vector: ring.input("public-lut-vector", (1, 4)),
            plaintext: Some(ring.polynomial([1.into()])),
        };
        let table = LutTable::unary(1, 2, vec![1]).unwrap();
        let helper = LutHelper::new(
            1,
            ExponentRhsPackage::new(ring.zero((2, 4))).unwrap(),
            BggEncodingWire { vector: ring.zero((1, 4)), plaintext: None },
        )
        .unwrap();
        let helpers = [helper];
        let cache = preprocess_lut_cache(&compiler, &helpers);
        let output = compiler
            .single_input_lut_table_with_cache(&input, &table, &helpers, Some(&cache))
            .unwrap();
        let expected = crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }
            .mono(&ring, &BigInt::from(1))
            .unwrap();
        let graph = DslContext::new("public-lut-plaintext-exact")
            .output("actual", output.plaintext.expect("public LUT plaintext"))
            .unwrap()
            .output("expected", expected)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(actual) = &result.outputs["actual"] else {
            panic!("public LUT plaintext output")
        };
        let RuntimeValue::Matrix(expected) = &result.outputs["expected"] else {
            panic!("public LUT expected output")
        };
        assert_eq!(actual.as_ref(), expected.as_ref());
    }

    #[test]
    fn native_lut_plaintext_is_exhaustive_for_small_rings_and_strides() {
        for (ring_dimension, stride_log2) in
            [(8usize, 0u32), (8, 1), (8, 2), (16, 0), (16, 1), (16, 2)]
        {
            let parameters = DCRTPolyParams::new(ring_dimension as u32, 1, 20, 4, None, None);
            let modulus = BigInt::from(parameters.modulus().as_ref().clone());
            let ring = Ring::new(modulus, ring_dimension);
            let width = ring_dimension >> stride_log2;
            let layout =
                crate::layout::ExpLayout { stride_log2, width_log2: width.trailing_zeros() };
            let compiler = ExponentLutEncodingCompiler::with_layout(
                BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() },
                layout,
            );
            let sigmas = layout.automorphism_indices(ring_dimension).unwrap();
            let table = (0..width).map(|value| (value + 3) % width).collect::<Vec<_>>();
            let helpers = sigmas
                .iter()
                .map(|sigma| {
                    LutHelper::new(
                        *sigma,
                        ExponentRhsPackage::new(ring.zero((2, 4))).unwrap(),
                        BggEncodingWire { vector: ring.zero((1, 4)), plaintext: None },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let table =
                LutTable::unary(width, layout.effective_modulus(ring_dimension), table).unwrap();
            let cache = preprocess_lut_cache(&compiler, &helpers);
            let mut context =
                DslContext::new(format!("native-lut-plaintext-{ring_dimension}-{stride_log2}"));
            for input_exponent in 0..width {
                let input = BggEncodingWire {
                    vector: ring.zero((1, 4)),
                    plaintext: Some(layout.mono(&ring, &BigInt::from(input_exponent)).unwrap()),
                };
                let output = compiler
                    .single_input_lut_table_with_cache(&input, &table, &helpers, Some(&cache))
                    .unwrap();
                let expected =
                    layout.mono(&ring, &BigInt::from(table.values()[input_exponent])).unwrap();
                context = context
                    .output(
                        format!("output-{input_exponent}"),
                        output.plaintext.expect("public LUT plaintext"),
                    )
                    .unwrap()
                    .output(format!("expected-{input_exponent}"), expected)
                    .unwrap();
            }
            let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
            let result = execute(
                &graph,
                &mut cpu_backend([parameters]),
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            for input_exponent in 0..width {
                let RuntimeValue::Matrix(actual) =
                    &result.outputs[&format!("output-{input_exponent}")]
                else {
                    panic!("native LUT plaintext output")
                };
                let RuntimeValue::Matrix(expected) =
                    &result.outputs[&format!("expected-{input_exponent}")]
                else {
                    panic!("native LUT expected output")
                };
                assert_eq!(
                    actual.as_ref(),
                    expected.as_ref(),
                    "n={ring_dimension}, m={stride_log2}, u={input_exponent}"
                );
            }
        }
    }

    #[test]
    fn cached_program_keeps_public_key_materialization_at_output_boundary() {
        let ring = Ring::new(97, 4);
        let layout = crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 };
        let compiler = ExponentLutEncodingCompiler::with_layout(
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() },
            layout,
        );
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(layout, 4).unwrap();
        let input = builder
            .input(1, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(1).unwrap();
        let rhs_id = builder.rhs_input(family, 1, 1).unwrap();
        let output = builder.private_mul(builder.input_wire(input).unwrap(), rhs_id).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();

        let input_wire = BggEncodingWire {
            vector: ring.input("cached-program-vector", (1, 4)),
            plaintext: None,
        };
        let rhs = ExponentRhsPackage::new(ring.input("cached-program-rhs", (2, 4))).unwrap();
        let k_rhs = ring.preimage_input("cached-program-k-rhs", (4, 4), 7);
        let outputs = compiler
            .compile_program_with_ranges_and_cache(
                &program,
                &BTreeMap::from([(ProgramInputId::from_index(0), input_wire)]),
                &BTreeMap::from([(RhsId::from_index(0), rhs)]),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &ProgramFamilyBindings::new(),
                &BTreeMap::new(),
                &ProgramCacheBindings::new().gate(ProgramGateInstance::Gate { step: 0 }, k_rhs),
            )
            .unwrap();
        let output = outputs.get(&output).unwrap();
        let graph = DslContext::new("cached-program-materialization")
            .output("vector", output.vector.clone())
            .unwrap()
            .build()
            .unwrap();
        let nodes = graph.graph.scopes().values().flat_map(|scope| scope.nodes());
        let nodes = nodes.collect::<Vec<_>>();
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
        // The RHS decomposition is used by carrier multiplication; Add and
        // reduction remain vector-only and never create an output cache.
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::MatrixMulSmallRhs
                ))
                .count(),
            1
        );
    }

    #[test]
    fn cached_lut_consumes_branch_preimages_without_online_decomposition() {
        let ring = Ring::new(97, 4);
        let layout = crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 };
        let compiler = ExponentLutEncodingCompiler::with_layout(
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() },
            layout,
        );
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(layout, 4).unwrap();
        let input_id = builder
            .input(1, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let table = LutTable::unary(1, 2, vec![1]).unwrap();
        let lut = builder.register_lut(table.clone()).unwrap();
        let output = builder.lut(builder.input_wire(input_id).unwrap(), lut, None).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let helper = LutHelper::new(
            1,
            ExponentRhsPackage::new(ring.input("cached-lut-switch", (2, 2))).unwrap(),
            BggEncodingWire {
                vector: ring.input("cached-lut-mask-vector", (1, 2)),
                plaintext: None,
            },
        )
        .unwrap();
        let input = BggEncodingWire {
            vector: ring.input("cached-lut-input-vector", (1, 2)),
            plaintext: None,
        };
        let cache = ProgramCacheBindings::new().lut(
            ProgramGateInstance::Gate { step: 0 },
            vec![ring.preimage_input("cached-lut-switch-k", (4, 2), 7)],
            vec![ring.preimage_input("cached-lut-branch-k", (4, 2), 7)],
        );
        let outputs = compiler
            .compile_program_with_ranges_and_cache(
                &program,
                &BTreeMap::from([(input_id, input)]),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &ProgramFamilyBindings::new(),
                &BTreeMap::from([(lut, LutHelperSet::new(&table, vec![helper]).unwrap())]),
                &cache,
            )
            .unwrap();
        let output = outputs.get(&output).unwrap();
        let graph = DslContext::new("cached-lut-no-decompose")
            .output("vector", output.vector.clone())
            .unwrap()
            .build()
            .unwrap();
        let nodes = graph.graph.scopes().values().flat_map(|scope| scope.nodes());
        assert!(!nodes.clone().any(|node| matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::GadgetDecompose { .. }
        )));
    }

    #[test]
    fn family_lut_reuses_public_preprocessing_cache_in_encoding_lowering() {
        let ring = Ring::new(97, 4);
        let layout = crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 };
        let public_key =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
        let public_compiler = ExponentLutPublicKeyCompiler::with_layout(public_key.clone(), layout);

        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(layout, 4).unwrap();
        let members = builder.wire_family(1).unwrap();
        let table = LutTable::unary(1, 2, vec![1]).unwrap();
        let lut = builder.register_lut(table.clone()).unwrap();
        let input = WireId::from_index(0);
        let transformed = WireId::from_index(1);
        let output = builder
            .family(
                2,
                FamilyProgram::new(
                    BTreeMap::from([(input, FamilyWireSource::Member(members))]),
                    BTreeMap::new(),
                    vec![Gate::Lut { input, lut_id: lut, output: transformed, rhs: None }],
                    transformed,
                )
                .unwrap(),
            )
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(ProgramFamilyId::from_index(0), FamilyRange::bounded(0, 2, 2).unwrap());

        let public_values = PublicKeyWireFamily::from_matrices(
            Family::pack(vec![
                ring.input("family-lut-public-0", (2, 4)),
                ring.input("family-lut-public-1", (2, 4)),
            ])
            .unwrap(),
            &ring,
            2,
            4,
        )
        .unwrap();
        let switch = ExponentRhsPackage::new(ring.input("family-lut-switch", (2, 4))).unwrap();
        let public_helpers = LutPublicHelperSet::new(
            &table,
            vec![LutPublicHelper::new(
                1,
                switch.clone(),
                ring.input("family-lut-public-mask", (2, 4)),
            )],
        )
        .unwrap();
        let preprocessing = public_compiler
            .compile_program_with_preprocessing(
                &program,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::from([(members, public_values)]),
                &ranges,
                &BTreeMap::from([(lut, public_helpers)]),
            )
            .unwrap();
        assert_eq!(
            preprocessing.family_caches[&ProgramFamilyId::from_index(0)]
                .count()
                .evaluate(&ParamEnv::default())
                .unwrap(),
            BigInt::from(2)
        );

        let encoding_values = BggEncodingFamily::from_components(
            Family::pack(vec![
                ring.input("family-lut-vector-0", (1, 4)),
                ring.input("family-lut-vector-1", (1, 4)),
            ])
            .unwrap(),
            Family::pack(vec![ring.identity(1), ring.identity(1)]).unwrap(),
            &ring,
            4,
        )
        .unwrap();
        let private_helpers = LutHelperSet::new(
            &table,
            vec![
                LutHelper::new(
                    1,
                    switch,
                    BggEncodingWire {
                        vector: ring.input("family-lut-private-mask", (1, 4)),
                        plaintext: None,
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let encoding_values = encoding_values
            .vector_family()
            .clone()
            .zip_map(encoding_values.plaintext_family().clone(), |vector, plaintext| {
                Ok(BggEncodingWire { vector, plaintext: Some(plaintext) })
            })
            .unwrap();
        let outputs = preprocessing
            .evaluate(&ProgramBindings::new(
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::from([(members, (encoding_values, FamilyRange::full(2).unwrap()))]),
                &BTreeMap::from([(lut, crate::program::ProgramValue::Scalar(private_helpers))]),
                None,
            ))
            .unwrap();
        DslContext::new("family-lut-public-to-encoding-cache")
            .output("vector", outputs[&output].vector.clone())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
    }

    #[test]
    fn native_sampler_supports_shared_and_distinct_payload_secrets() {
        let ring = Ring::new(97, 4);
        let sampler = super::ExponentLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let mask = ring.input("native-mask", (1, 2));
        let payload = ring.input("native-payload", (1, 2));
        let hash_key = ring.bytes_input("native-hash", 32);
        let table = LutTable::unary(2, 4, vec![0, 1]).unwrap();
        let bank = sampler
            .sample_mask_bank(mask.clone(), hash_key.clone(), 2, b"native-bank".as_slice())
            .unwrap();
        let shared = sampler
            .sample_helpers_for_lut(
                mask.clone(),
                None,
                hash_key.clone(),
                &table,
                bank.as_ref(),
                b"shared".as_slice(),
                None,
            )
            .unwrap();
        let distinct = sampler
            .sample_helpers_for_lut(
                mask,
                Some(payload),
                hash_key,
                &table,
                bank.as_ref(),
                b"distinct".as_slice(),
                None,
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
    fn native_mask_bank_reuses_branches_across_lut_widths() {
        let ring = Ring::new(97, 16);
        let sampler = super::ExponentLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: 16.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2: 2 },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let mask = ring.input("mask-bank-secret", (1, 2));
        let hash_key = ring.bytes_input("mask-bank-hash", 32);
        let bank = sampler
            .sample_mask_bank(mask.clone(), hash_key.clone(), 4, b"mask-bank".as_slice())
            .unwrap();
        assert_eq!(bank.sigmas().collect::<Vec<_>>(), [1, 9, 17, 25]);
        for (index, sigma) in bank.sigmas().enumerate() {
            let private_vector = &bank.branch(index).unwrap().mask.vector;
            assert_eq!(private_vector.matrix_type(), &ring.matrix_type((1, 4)));
            assert_eq!(sigma, bank.branch(index).unwrap().sigma);
        }
        let wide = LutTable::unary(4, 16, vec![0, 1, 2, 3]).unwrap();
        let same_width = LutTable::unary(4, 16, vec![3, 2, 1, 0]).unwrap();
        let narrow = LutTable::unary(2, 16, vec![0, 1]).unwrap();
        let wide_helpers = sampler
            .sample_helpers_for_lut(
                mask.clone(),
                None,
                hash_key.clone(),
                &wide,
                bank.as_ref(),
                b"wide".as_slice(),
                None,
            )
            .unwrap();
        let same_width_helpers = sampler
            .sample_helpers_for_lut(
                mask.clone(),
                None,
                hash_key.clone(),
                &same_width,
                bank.as_ref(),
                b"same-width".as_slice(),
                None,
            )
            .unwrap();
        let narrow_helpers = sampler
            .sample_helpers_for_lut(
                mask,
                None,
                hash_key,
                &narrow,
                bank.as_ref(),
                b"narrow".as_slice(),
                None,
            )
            .unwrap();
        assert_eq!(
            wide_helpers[0].mask().vector.value_handle().node(),
            narrow_helpers[0].mask().vector.value_handle().node()
        );
        assert_eq!(
            wide_helpers[2].mask().vector.value_handle().node(),
            narrow_helpers[1].mask().vector.value_handle().node()
        );
        assert_eq!(
            wide_helpers[0].mask().vector.value_handle().node(),
            same_width_helpers[0].mask().vector.value_handle().node()
        );
        assert_ne!(
            wide_helpers[0].switch().gsw_ciphertext().value_handle().node(),
            same_width_helpers[0].switch().gsw_ciphertext().value_handle().node()
        );
        let wide_set = LutHelperSet::new(&wide, wide_helpers).unwrap();
        let same_width_set = LutHelperSet::new(&same_width, same_width_helpers).unwrap();
        assert_ne!(wide_set.metadata().0, same_width_set.metadata().0);
        assert_ne!(
            wide_set.metadata().0,
            LutHelperSet::new(&narrow, narrow_helpers).unwrap().metadata().0
        );
    }

    #[test]
    fn native_lut_branches_are_lowered_in_one_structural_parallel_loop() {
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutEncodingCompiler::with_layout(
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() },
            crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 },
        );
        let input =
            BggEncodingWire { vector: ring.input("native-input-vector", (1, 4)), plaintext: None };
        let make_helper = |sigma| {
            LutHelper::new(
                sigma,
                ExponentRhsPackage::new(ring.zero((2, 4))).unwrap(),
                BggEncodingWire { vector: ring.zero((1, 4)), plaintext: None },
            )
            .unwrap()
        };
        let helpers = [make_helper(1), make_helper(5)];
        let table = LutTable::unary(2, 2, vec![0, 1]).unwrap();
        let cache = preprocess_lut_cache(&compiler, &helpers);
        let output = compiler
            .single_input_lut_table_with_cache(&input, &table, &helpers, Some(&cache))
            .unwrap();
        let graph = DslContext::new("native-lut-structural-loop")
            .output("vector", output.vector)
            .unwrap()
            .build()
            .unwrap();
        let nodes = graph.graph.scopes().values().flat_map(|scope| scope.nodes());
        let mut private_branch_grids = 0;
        let mut sequential_loops = 0;
        for node in nodes {
            match node.kind() {
                mxx_ir_core::node::NodeKind::ParallelLoop(grid) => {
                    let count =
                        grid.count.evaluate(&ParamEnv::default()).unwrap().to_usize().unwrap();
                    if count == 2 && grid.input_modes.len() == 4 {
                        private_branch_grids += 1;
                    }
                }
                mxx_ir_core::node::NodeKind::SequentialLoop(_) => sequential_loops += 1,
                _ => {}
            }
        }
        assert_eq!(private_branch_grids, 1, "one private automorphism grid of extent W");
        assert_eq!(sequential_loops, 0, "native LUT lowering has no sequential terminal");
    }

    #[test]
    fn native_readout_width_1024_keeps_helpers_in_one_structural_family() {
        let ring = Ring::new(97, 1usize << 16);
        let sampler = super::ExponentLutEncodingSampler {
            layout: mxx_bgg::BggSamplerLayout {
                modulus: 97.into(),
                ring_dimension: (1usize << 16).into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            exponent_layout: crate::layout::ExpLayout { stride_log2: 6, width_log2: 10 },
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let helpers = sampler
            .sample_native_readout_helper_families(
                ring.input("setup-secret", (1, 2)),
                ring.bytes_input("hash-key", 32),
                1024,
                b"native-mask".as_slice(),
                b"native-helper".as_slice(),
            )
            .unwrap();
        assert_eq!(helpers.sigmas.len(), 1023);
        let online_masks = sampler
            .sample_native_readout_masks(ring.input("online-secret", (1, 2)), &helpers.public_masks)
            .unwrap();
        let online_graph = DslContext::new("native-readout-online-masks")
            .private_output("online-masks", online_masks)
            .unwrap()
            .build()
            .unwrap();
        assert!(online_graph.graph.scopes().values().flat_map(|scope| scope.nodes()).all(|node| {
            !matches!(node.kind(), mxx_ir_core::node::NodeKind::Input { name, .. }
                if name == "setup-secret")
        }));
        online_graph.validate(&ParamEnv::default()).expect("online phase validates");
        let graph = DslContext::new("native-readout-helper-family")
            .public_output("switches", helpers.switch_ciphertexts)
            .unwrap()
            .public_output("masks", helpers.public_masks)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        // Public matrices and payload switches each have one structural body.
        // The online graph above contains no payload-secret input.
        assert!(nodes.iter().all(|node| !matches!(node.kind(),
            mxx_ir_core::node::NodeKind::Input { name, .. } if name == "online-secret"
        )));
        assert!(
            nodes.len() < 8 * 1024,
            "helper graph grew beyond one structural family: {} nodes",
            nodes.len()
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::ParallelLoop(grid)
                        if grid.count == mxx_ir_core::IntExpr::constant(1023)
                ))
                .count(),
            2
        );
        assert!(graph.graph.root_scope().nodes().iter().all(|node| !matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::FamilyPack { .. } |
                mxx_ir_core::node::NodeKind::FamilyGetStatic { .. }
        )));
        graph.validate(&ParamEnv::default()).expect("structural helper graph validates");
    }
}

#[cfg(all(test, feature = "gpu"))]
#[path = "gpu.rs"]
mod gpu_tests;

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

    pub(super) fn with_online_mask(
        &self,
        mask: BggEncodingWire,
    ) -> Result<ValueReadoutHelper, BggReadoutError> {
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

impl SparseLwrPrfProgram {
    /// Online readout variant which combines setup-fixed decomposition
    /// witnesses with freshly sampled online mask encodings. Switch/GSW and
    /// public decomposition artifacts remain setup-owned; no online helper
    /// switch or GSW ciphertext is sampled here.
    #[cfg(test)]
    pub fn execute_value_readout_batch_with_masks(
        &self,
        compiler: &mxx_bgg::BggEncodingCompiler,
        inputs: &Family<Mat>,
        artifacts: &NativePrfReadoutArtifacts,
        online_masks: &[(usize, BggEncodingWire)],
    ) -> Result<Family<Mat>, BggReadoutError> {
        if online_masks.len() != artifacts.helpers.len() {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        let helpers = artifacts
            .helpers
            .iter()
            .zip(online_masks)
            .map(|(cached, (sigma, mask))| {
                if cached.sigma() != *sigma {
                    return Err(ValueReadoutError::InvalidTraceSet.into());
                }
                cached.with_online_mask(mask.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let vectors = with_benchmark_role(BenchmarkRole::EncodingReadout, || {
            mxx_dsl::parallel(inputs.count().clone(), |index| {
                let input = BggEncodingWire { vector: inputs.at(index), plaintext: None };
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
                BggReadoutError::Graph(BggError::OnlinePrfStage {
                    stage: "compile readout batch",
                    detail: error.to_string(),
                })
            })
        })?;
        Ok(vectors)
    }
}

/// Validates the profile associated with a caller-provided program.
#[cfg(test)]
mod prf_tests {
    use super::*;
    use mxx_dsl::Ring;

    #[test]
    fn native_batch_imports_one_fixed_product_per_block() {
        use crate::{
            noise_refresh::prf::{PreparedSparseLwrPrf, PrfSelectionInputs},
            program::{FamilyRange, ProgramValue},
        };
        let ring = Ring::new(97, 8);
        let layout = ExpLayout { stride_log2: 1, width_log2: 2 };
        let scheme = BggScheme::new(
            mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 2.into(),
                digit_count: 2.into(),
            },
            layout,
        );
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(layout, 1).unwrap(), 2, 2).unwrap();
        let labels = ring.input_family("labels", 8, (1, 1));
        let selectors = SetupRhsFamily::new(ring.input_family("selectors", 4, (2, 4))).unwrap();
        let blocks = [0usize, 2]
            .into_iter()
            .map(|start| {
                PrfSelectionInputs::new(
                    selectors.clone(),
                    &labels,
                    FamilyRange::bounded(start, 2, 4).unwrap(),
                    rand::random(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let (_, prepared) = PreparedSparseLwrPrf::new(
            &scheme,
            program.clone(),
            ProgramValue::Scalar(BggPublicKeyWire {
                matrix: ring.input("public", (2, 4)),
                reveal_plaintext: false,
            }),
            &blocks,
        )
        .unwrap();
        let (context, descriptor) =
            prepared.export(&scheme, mxx_dsl::DslContext::new("fixed-cache"), "prf").unwrap();
        let producer = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let production = mxx_ir_core::artifact::ProductionId {
            spec_hash: mxx_ir_core::encoding::spec_hash(&producer.source, &producer.bindings)
                .unwrap(),
            execution_nonce: rand::random(),
        };
        let manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &producer)
                .unwrap();
        let identity = descriptor.identity().unwrap();
        let descriptor = serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let prepared = PreparedSparseLwrPrf::import(
            &scheme,
            program,
            &descriptor,
            production.clone(),
            identity,
        )
        .unwrap();
        assert!(prepared.steps().iter().all(|step| step.count() == &2.into()));
        let outputs = prepared
            .evaluate(
                &scheme,
                ProgramValue::Scalar(BggEncodingWire {
                    vector: ring.input("initial", (1, 4)),
                    plaintext: None,
                }),
            )
            .unwrap();
        let ProgramValue::Family(outputs) = outputs else { panic!("output family") };
        assert_eq!(outputs.count(), &2.into());
        let graph = mxx_dsl::DslContext::new("online-fixed-body")
            .public_output("outputs", outputs)
            .unwrap()
            .build()
            .unwrap();
        graph
            .validate_with_manifests(
                &ParamEnv::default(),
                &std::collections::BTreeMap::from([(production, manifest)]),
            )
            .unwrap();
        let products = graph
            .graph
            .scopes()
            .values()
            .map(|scope| {
                assert!(scope.nodes().iter().all(|node| !matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::GadgetDecompose { .. }
                )));
                scope
                    .nodes()
                    .iter()
                    .filter(|node| {
                        matches!(node.kind(), mxx_ir_core::node::NodeKind::MatrixMulSmallRhs)
                    })
                    .count()
            })
            .filter(|count| *count > 0)
            .collect::<Vec<_>>();
        assert_eq!(products, vec![1, 1]);
    }

    #[test]
    fn native_batch_readout_consumes_supplied_artifacts_for_two_labels() {
        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let batch = ring.input_family("readout-batch-vectors", 2, (1, 4));
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
        assert_eq!(batched.count(), &2.into());
        let built = mxx_dsl::DslContext::new("native-batch-readout")
            .output("vectors", batched.clone())
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
        let sampler = crate::bgg::encoding::ExponentLutEncodingSampler {
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
                None,
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
            .produce_readout_artifacts(&compiler, &input, &helper_set, ring.gadget(2, 4, 2))
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
        use crate::bgg::{
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
        let batch = ring.input_family("imported-vectors", 1, (1, 4));
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
            .output("vectors", output.clone())
            .unwrap()
            .build()
            .unwrap();
        assert!(!graph.graph.scopes().values().flat_map(|scope| scope.nodes()).any(
            |node| matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
        ));
    }
}

impl ValueReadoutHelper {
    pub fn new(
        layout: ExpLayout,
        sigma: usize,
        gadget_base: BigInt,
        digit_count: usize,
        max_coefficient_bound: BigInt,
        switch_decomposition: Preimage,
        public_decomposition: Preimage,
        mask: BggEncodingWire,
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
            switch_decomposition,
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
    /// Executes readout for every online PRF output without nesting a helper
    /// loop inside a label loop. The label/helper Cartesian product is one
    /// flat structural family, then each consecutive helper segment is
    /// reduced back to one output per label.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_batch_with_preprocessed_family_artifacts(
        &self,
        compiler: &BggEncodingCompiler,
        inputs: Family<Mat>,
        label_count: usize,
        target_decompositions: Family<Preimage>,
        target_schema: ReadoutPreimageSchema,
        helper_sigmas: &[usize],
        switch_decompositions: Family<Preimage>,
        public_decompositions: Family<Preimage>,
        masks: Family<Mat>,
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<Family<Mat>, BggReadoutError> {
        if label_count == 0 || *inputs.count() != IntExpr::constant(label_count) {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        let input_type = inputs.element_type();
        let ring = Ring::new(input_type.modulus.clone(), input_type.ring_dimension.clone());
        self.validate(&ring)?;
        let ring_dimension = concrete_ring_dimension(&ring)?;
        let plan = self.trace_plan(ring_dimension)?;
        let helper_count = helper_sigmas.len();
        if helper_count == 0 ||
            helper_sigmas != plan.non_identity_helpers() ||
            switch_decompositions.count() != masks.count() ||
            *switch_decompositions.count() != IntExpr::constant(helper_count) ||
            *public_decompositions.count() !=
                IntExpr::constant(
                    label_count
                        .checked_mul(helper_count)
                        .ok_or(ValueReadoutError::InvalidTraceSet)?,
                )
        {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        if *target_decompositions.count() != IntExpr::constant(label_count) {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        validate_preimage_ring(&normalization_decomposition, &ring)?;
        if target_schema.gadget_base != normalization_schema.gadget_base ||
            target_schema.digit_count != normalization_schema.digit_count
        {
            return Err(BggReadoutError::InvalidPreimageSchema.into());
        }
        validate_encoding_action_preimage_schema_type(
            &target_decompositions.at(0),
            input_type,
            &target_schema.gadget_base,
            target_schema.digit_count,
            Some(&target_schema.max_coefficient_bound),
        )?;
        validate_encoding_action_preimage_family_type(
            switch_decompositions.element_type(),
            input_type,
            target_schema.digit_count,
        )?;
        validate_encoding_action_preimage_family_type(
            public_decompositions.element_type(),
            input_type,
            target_schema.digit_count,
        )?;
        validate_family_output_type(masks.element_type(), input_type, &ring)?;

        let target = self.target(&ring)?;
        let target_for_labels = target.clone();
        let label_compiler = compiler.clone();
        let transformed = {
            let source_families = (inputs, target_decompositions);
            if source_families.0.count() != source_families.1.count() {
                Err(mxx_dsl::DslError::Schema)
            } else {
                mxx_dsl::parallel(source_families.0.count().clone(), |family_index| {
                    let (vector, target_decomposition) =
                        (source_families.0.at(&family_index), source_families.1.at(&family_index));
                    let _ = family_index;
                    Ok({
                        label_compiler
                            .large_scalar_mul(
                                &BggEncodingWire { vector, plaintext: None },
                                &target_for_labels,
                                target_decomposition,
                            )
                            .vector
                    })
                })
            }
        }
        .map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "transform readout input family",
                detail: error.to_string(),
            })
        })?;
        let flat_count =
            label_count.checked_mul(helper_count).ok_or(ValueReadoutError::InvalidTraceSet)?;
        let helper_indices = mxx_dsl::parallel(flat_count, |index| {
            Ok(index.clone().rem(Int::constant(helper_count)))
        })
        .map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "index aligned readout masks",
                detail: error.to_string(),
            })
        })?;
        let flat_masks = masks.gather(helper_indices.clone()).map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "gather aligned readout masks",
                detail: error.to_string(),
            })
        })?;
        let aligned_public = flat_masks
            .zip_map(public_decompositions, |mask, public| Ok(public.mul_small_rhs(mask)))
            .map_err(|error| {
                BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                    stage: "align readout public helper family",
                    detail: error.to_string(),
                })
            })?;

        let label_indices = mxx_dsl::parallel(flat_count, |index| {
            Ok(index.clone().div(Int::constant(helper_count)))
        })
        .map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "index readout label family",
                detail: error.to_string(),
            })
        })?;
        let flat_transformed = transformed.clone().gather(label_indices).map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "gather readout labels",
                detail: error.to_string(),
            })
        })?;
        let flat_aligned = aligned_public;
        let flat_switches = {
            let source_family = switch_decompositions;
            let gather_indices = mxx_dsl::parallel(flat_count, |index| {
                Ok(index.clone().rem(Int::constant(helper_count)))
            })
            .map_err(|error| {
                BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                    stage: "index readout switches",
                    detail: error.to_string(),
                })
            })?;
            mxx_dsl::parallel(gather_indices.count().clone(), |family_index| {
                Ok(source_family.at(gather_indices.at(&family_index)))
            })
        }
        .map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "reindex readout switches",
                detail: error.to_string(),
            })
        })?;
        let sigma_branches =
            helper_sigmas.iter().copied().map(IntExpr::constant).collect::<Vec<_>>();
        let helper_vectors = {
            let source_families = (flat_transformed, flat_aligned, flat_switches);
            if source_families.0.count() != source_families.1.count() ||
                source_families.0.count() != source_families.2.count()
            {
                Err(mxx_dsl::DslError::Schema)
            } else {
                mxx_dsl::parallel(source_families.0.count().clone(), |family_index| {
                    let (transformed, aligned, switch) = (
                        source_families.0.at(&family_index),
                        source_families.1.at(&family_index),
                        source_families.2.at(&family_index),
                    );
                    let index = family_index;
                    Ok({
                        let sigma = helper_sigma_for_flat_index(
                            index.expression()?,
                            sigma_branches.clone(),
                        );
                        aligned + switch.mul_small_rhs(transformed.ring_automorphism(sigma))
                    })
                })
            }
        }
        .map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "construct flat readout helper family",
                detail: error.to_string(),
            })
        })?;
        let helper_sums = reduce_matrix_family_segments(helper_vectors, label_count, helper_count)?;
        let traced = transformed
            .zip_map(helper_sums, |identity, helpers| Ok(identity + helpers))
            .map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "add readout identity branches",
                detail: error.to_string(),
            })
        })?;

        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let inverse = modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(BggReadoutError::NonInvertibleNormalization)?;
        validate_encoding_action_preimage_schema_type(
            &normalization_decomposition,
            traced.element_type(),
            &normalization_schema.gadget_base,
            normalization_schema.digit_count,
            Some(&normalization_schema.max_coefficient_bound),
        )?;
        let normalization_scalar = ring.polynomial([inverse.into()]);
        let normalization_compiler = compiler.clone();
        {
            let source_family = traced;
            mxx_dsl::parallel(source_family.count().clone(), |family_index| {
                let vector = source_family.at(&family_index);
                let _ = family_index;
                Ok({
                    normalization_compiler
                        .large_scalar_mul(
                            &BggEncodingWire { vector, plaintext: None },
                            &normalization_scalar,
                            normalization_decomposition.clone(),
                        )
                        .vector
                })
            })
        }
        .map_err(|error| {
            BggReadoutError::Graph(crate::bgg::BggError::OnlinePrfStage {
                stage: "normalize readout output family",
                detail: error.to_string(),
            })
        })
    }
    /// Executes value readout using only explicitly supplied decomposition
    /// artifacts.  The target multiplication is followed by one family trace
    /// containing the identity branch exactly once and one helper branch for
    /// each non-identity automorphism, then by the explicit normalization
    /// preimage.  In particular, this path contains no sampler or fallback
    /// decomposition operation.
    pub fn execute_with_preprocessed_artifacts(
        &self,
        compiler: &BggEncodingCompiler,
        input: &BggEncodingWire,
        target_decomposition: Preimage,
        target_schema: ReadoutPreimageSchema,
        helpers: &[ValueReadoutHelper],
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<BggEncodingWire, BggReadoutError> {
        let matrix_type = input.vector.matrix_type();
        let modulus = matrix_type.modulus.clone();
        let ring = Ring::new(modulus, matrix_type.ring_dimension.clone());
        self.validate(&ring)?;
        let ring_dimension = concrete_ring_dimension(&ring)?;
        let plan = self.trace_plan(ring_dimension)?;
        if helpers.len() != plan.helper_count() ||
            helpers.iter().map(ValueReadoutHelper::sigma).collect::<Vec<_>>() !=
                plan.non_identity_helpers()
        {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        if helpers.iter().any(|helper| helper.layout != self.layout) {
            return Err(ValueReadoutError::InvalidTraceSet.into());
        }
        validate_preimage_ring(&target_decomposition, &ring)?;
        validate_preimage_ring(&normalization_decomposition, &ring)?;
        let gadget_base = target_schema.gadget_base.clone();
        let digit_count = target_schema.digit_count;
        if target_schema.gadget_base != gadget_base ||
            target_schema.digit_count != digit_count ||
            normalization_schema.gadget_base != gadget_base ||
            normalization_schema.digit_count != digit_count
        {
            return Err(BggReadoutError::InvalidPreimageSchema.into());
        }
        validate_encoding_action_preimage_schema(
            &target_decomposition,
            &input.vector,
            &target_schema.gadget_base,
            target_schema.digit_count,
            Some(&target_schema.max_coefficient_bound),
        )?;
        for helper in helpers {
            if helper.gadget_base != gadget_base || helper.digit_count != digit_count {
                return Err(BggReadoutError::InvalidPreimageSchema.into());
            }
            validate_encoding_action_preimage_schema(
                &helper.switch_decomposition,
                &input.vector,
                &gadget_base,
                digit_count,
                Some(&helper.max_coefficient_bound),
            )?;
            validate_encoding_action_preimage_schema(
                &helper.public_decomposition,
                &input.vector,
                &gadget_base,
                digit_count,
                Some(&helper.max_coefficient_bound),
            )?;
            validate_matrix_ring(&helper.mask.vector, &ring)?;
            validate_matrix_ring(&helper.mask.vector, &ring)?;
            validate_output_shape(&helper.mask.vector, &input.vector, &ring)?;
            if helper.mask.plaintext.is_some() {
                return Err(ValueReadoutError::InvalidTraceSet.into());
            }
        }
        // Derive public matrices online from the input relation and typed
        // preimages.  Public output matrices are not preprocessing artifacts.
        let target = self.target(&ring)?;
        let transformed = compiler.large_scalar_mul(input, &target, target_decomposition);
        let mut vectors = Vec::with_capacity(plan.term_count());
        let mut plaintext = transformed.plaintext.clone().map(|value| vec![value]);
        vectors.push(transformed.vector.clone());
        for helper in helpers {
            let raw_vector = transformed.vector.clone().ring_automorphism(helper.sigma);
            let switched_vector = helper.switch_decomposition.clone().mul_small_rhs(raw_vector);
            vectors.push(
                helper.public_decomposition.clone().mul_small_rhs(helper.mask.vector.clone()) +
                    switched_vector,
            );
            if let Some(values) = plaintext.as_mut() {
                values.push(
                    transformed.plaintext.as_ref().unwrap().clone().ring_automorphism(helper.sigma),
                );
            }
        }
        let traced = BggEncodingWire {
            vector: crate::utils::balanced_sum_family(
                Family::pack(vectors).map_err(|_| crate::bgg::BggError::InvalidLut)?,
            )
            .map_err(crate::bgg::BggError::from)?,
            plaintext: plaintext.map(|values| {
                crate::utils::balanced_sum_family(
                    Family::pack(values).expect("validated trace plaintext family"),
                )
                .expect("validated trace plaintext sum")
            }),
        };
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let inverse = modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(BggReadoutError::NonInvertibleNormalization)?;
        validate_encoding_action_preimage_schema(
            &normalization_decomposition,
            &traced.vector,
            &normalization_schema.gadget_base,
            normalization_schema.digit_count,
            Some(&normalization_schema.max_coefficient_bound),
        )?;
        Ok(compiler.large_scalar_mul(
            &traced,
            &ring.polynomial([inverse.into()]),
            normalization_decomposition,
        ))
    }
}

#[cfg(test)]
mod readout_tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{
        IntExpr,
        node::{ConstantMatrix, NodeKind},
    };

    #[test]
    fn target_plaintext_gain_tracks_table_and_monomial_coefficients() {
        // Native readout requires width >= 2. Check the monomial invariant
        // directly on the same convolution-gain primitive without relaxing it.
        for sign in [-1, 1] {
            assert_eq!(
                crate::noise_utils::RightActionNoiseGain::from_polynomial_coefficients([
                    BigInt::zero(),
                    sign.into(),
                    BigInt::zero()
                ],),
                crate::noise_utils::RightActionNoiseGain::unit()
            );
        }
        for stride_log2 in [0, 1, 3] {
            let readout = ValueReadout::from_canonical_values(
                ExpLayout { stride_log2, width_log2: 2 },
                vec![1, 1, 2, 2],
            )
            .unwrap();
            assert_eq!(
                readout.target_plaintext_gain(4 << stride_log2).unwrap(),
                crate::noise_utils::RightActionNoiseGain {
                    l1: 8u8.into(),
                    l2_squared: 20u8.into()
                }
            );
        }
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let ring = Ring::new(97, 4);
        let symbolic = ValueReadout::new(layout, ring.input("table", (1, 1))).unwrap();
        assert!(symbolic.target_plaintext_gain(4).is_err());
        let padded = ValueReadout::new(layout, ring.polynomial([1.into()])).unwrap();
        assert_eq!(
            padded.target_plaintext_gain(4).unwrap(),
            crate::noise_utils::RightActionNoiseGain { l1: 4u8.into(), l2_squared: 4u8.into() }
        );
        let aliased = ValueReadout::new(layout, ring.polynomial(vec![1.into(); 5])).unwrap();
        assert!(aliased.target_plaintext_gain(4).is_err());
    }

    fn independent_automorphisms(ring_dimension: usize, layout: ExpLayout) -> Vec<usize> {
        let modulus = 2 * ring_dimension;
        let mut result = std::collections::BTreeSet::new();
        for subset in 0..layout.width() {
            let mut product = 1usize;
            for bit in 0..layout.width_log2 {
                if subset & (1usize << bit) != 0 {
                    let denominator = 1usize << (layout.stride_log2 + bit + 1);
                    product = product * (modulus / denominator + 1) % modulus;
                }
            }
            result.insert(product);
        }
        result.into_iter().collect()
    }

    fn collect_target_ir(
        value: &mxx_ir_core::graph::ValueHandle,
        extracts: &mut Vec<usize>,
        rotations: &mut Vec<usize>,
        negated_monomials: &mut usize,
    ) {
        let node = value.node();
        match node.kind() {
            NodeKind::ExtractCoefficient { position, .. } => {
                if let IntExpr::Const(position) = position {
                    extracts.push(position.to_usize().unwrap());
                }
            }
            NodeKind::ConstantMatrix { value: ConstantMatrix::Rotation { exponent }, .. } => {
                if let IntExpr::Const(exponent) = exponent {
                    rotations.push(exponent.to_usize().unwrap());
                }
            }
            NodeKind::MatrixNegate => *negated_monomials += 1,
            _ => {}
        }
        for argument in node.arguments() {
            collect_target_ir(argument, extracts, rotations, negated_monomials);
        }
    }

    #[test]
    fn n32_m1_has_exact_j_order_and_one_identity() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 1, width_log2: 4 },
            &ring,
            vec![0; 16],
        )
        .unwrap();
        let plan = readout.trace_plan(32).unwrap();
        assert_eq!(
            plan.automorphisms(),
            &[1, 3, 5, 7, 9, 11, 15, 17, 21, 25, 27, 45, 51, 55, 61, 63]
        );
        assert_eq!(
            plan.non_identity_helpers(),
            &[3, 5, 7, 9, 11, 15, 17, 21, 25, 27, 45, 51, 55, 61, 63]
        );
        assert_eq!(plan.term_count(), 16);
        assert_eq!(plan.helper_count(), 15);
        assert_eq!(plan.automorphisms().iter().filter(|index| **index == 1).count(), 1);
    }

    #[test]
    fn n65536_m5_has_exact_j_order_and_helper_list() {
        let ring = Ring::new(97, 1usize << 16);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 5, width_log2: 11 },
            &ring,
            vec![0; 1usize << 11],
        )
        .unwrap();
        let plan = readout.trace_plan(1usize << 16).unwrap();
        let expected =
            independent_automorphisms(1usize << 16, ExpLayout { stride_log2: 5, width_log2: 11 });
        assert_eq!(plan.automorphisms(), expected.as_slice());
        assert_eq!(plan.non_identity_helpers(), &expected[1..]);
        assert_eq!(plan.term_count(), 2048);
        assert_eq!(plan.helper_count(), 2047);
        assert_eq!(plan.automorphisms().iter().filter(|index| **index == 1).count(), 1);
    }

    #[test]
    fn target_has_canonical_coefficients_and_expected_ir_signs() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 1, width_log2: 4 },
            &ring,
            (0..16).map(|value| value % 2).collect(),
        )
        .unwrap();
        assert_eq!(
            readout.canonical_coefficients().unwrap(),
            (0..16).map(|v| BigInt::from(v % 2)).collect::<Vec<_>>()
        );
        let target = readout.target(&ring).unwrap();
        assert_eq!(target.matrix_type(), &ring.matrix_type((1, 1)));
        let mut extracts = Vec::new();
        let mut rotations = Vec::new();
        let mut negated_monomials = 0;
        collect_target_ir(
            target.value_handle(),
            &mut extracts,
            &mut rotations,
            &mut negated_monomials,
        );
        extracts.sort_unstable();
        assert_eq!(extracts, (0..16).collect::<Vec<_>>());
        assert_eq!(rotations.len(), 16);
        assert_eq!(negated_monomials, 15);
        let expected_signs: Vec<BigInt> =
            (0..16).map(|k| BigInt::from(2 * (k % 2) as i32 - 1)).collect();
        assert_eq!(
            expected_signs,
            vec![
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
            ]
        );
    }

    #[test]
    fn normalization_requirement_is_typed_metadata_and_inverse_target() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 1, width_log2: 4 },
            &ring,
            vec![0; 16],
        )
        .unwrap();
        let gadget = ring.gadget(1, 2, 4);
        let requirement =
            readout.normalization_preimage_requirement(&ring, gadget.clone(), 7, 2, 4).unwrap();
        assert_eq!(requirement.target.denominator, 32);
        assert_eq!(requirement.target.matrix.modulus, BigInt::from(97));
        assert_eq!(requirement.target.matrix.ring_dimension, 32);
        assert_eq!(requirement.target.matrix.rows, 1);
        assert_eq!(requirement.target.matrix.columns, 4);
        assert_eq!(requirement.role, NormalizationPreimageRole::InverseNativeTraceGadget);
        assert_eq!(requirement.max_coefficient_bound, BigInt::from(7));
        assert_eq!(requirement.gadget_base, BigInt::from(2));
        assert_eq!(requirement.digit_count, 4);
        assert_eq!(requirement.readout_identity, readout.identity());
        assert_eq!(modular_inverse(&BigInt::from(32), &BigInt::from(97)), Some(BigInt::from(94)));
        assert_eq!(
            requirement.artifact_type,
            ArtifactType::Preimage {
                matrix: requirement.target.matrix.clone(),
                max_coefficient_bound: BigInt::from(7)
            }
        );
        let target = readout.normalization_target(&ring, ring.identity(1)).unwrap();
        let bytes = canonical_table_declaration(&target).unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("94"));
        let zero_gadget = ring.zero((1, 4));
        assert!(matches!(
            readout.normalization_preimage_requirement(&ring, zero_gadget, 7, 2, 4),
            Err(BggReadoutError::InvalidNormalizationGadget)
        ));
    }

    #[test]
    fn m_zero_fixture_matches_legacy_coefficients_and_monomial_signs() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 0, width_log2: 4 },
            &ring,
            (0..32).map(|k| k % 2).collect(),
        )
        .unwrap();
        assert_eq!(readout.canonical_coefficients().unwrap().len(), 32);
        let target = readout.target(&ring).unwrap();
        let mut extracts = Vec::new();
        let mut rotations = Vec::new();
        let mut negated_monomials = 0;
        collect_target_ir(
            target.value_handle(),
            &mut extracts,
            &mut rotations,
            &mut negated_monomials,
        );
        extracts.sort_unstable();
        assert_eq!(extracts, (0..32).collect::<Vec<_>>());
        assert_eq!(negated_monomials, 31);
        assert!(rotations.contains(&31) && rotations.contains(&1));
    }

    #[test]
    fn stable_identity_is_content_complete_and_layout_typed() {
        let layout = ExpLayout { stride_log2: 1, width_log2: 4 };
        let first = ValueReadout::from_values(layout, &Ring::new(97, 32), vec![0; 16]).unwrap();
        let rebuilt = ValueReadout::from_values(layout, &Ring::new(97, 32), vec![0; 16]).unwrap();
        let changed_table =
            ValueReadout::from_values(layout, &Ring::new(97, 32), vec![1; 16]).unwrap();
        let changed_modulus =
            ValueReadout::from_values(layout, &Ring::new(101, 32), vec![0; 16]).unwrap();
        let changed_layout = ValueReadout::from_values(
            ExpLayout { stride_log2: 0, width_log2: 4 },
            &Ring::new(97, 32),
            vec![0; 32],
        )
        .unwrap();
        assert_eq!(first.identity(), rebuilt.identity());
        assert_ne!(first.identity(), changed_table.identity());
        assert_ne!(first.identity(), changed_modulus.identity());
        assert_ne!(first.identity(), changed_layout.identity());
    }

    #[test]
    fn normalization_rejects_mismatched_and_noninvertible_rings() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 1, width_log2: 4 },
            &ring,
            vec![0; 16],
        )
        .unwrap();
        assert_eq!(readout.validate(&Ring::new(101, 32)), Err(ValueReadoutError::RingTypeMismatch));
        let bad_gadget = Ring::new(101, 32).gadget(1, 2, 4);
        assert_eq!(
            readout.normalization_target_descriptor(&ring, &bad_gadget),
            Err(BggReadoutError::Declaration(ValueReadoutError::RingTypeMismatch))
        );
        let even_ring = Ring::new(32, 32);
        let even_readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 0, width_log2: 4 },
            &even_ring,
            vec![0; 32],
        )
        .unwrap();
        assert!(matches!(
            even_readout.normalization_target(&even_ring, even_ring.identity(1)),
            Err(BggReadoutError::NonInvertibleNormalization)
        ));
    }

    #[test]
    fn cached_readout_uses_supplied_preimages_and_no_decomposition_nodes() {
        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 0 };
        let readout = ValueReadout::from_values(layout, &ring, vec![0, 1, 0, 1]).unwrap();
        let compiler = BggEncodingCompiler;
        let input = BggEncodingWire {
            vector: ring.input("readout-input-vector", (1, 4)),
            plaintext: Some(ring.polynomial([1.into()])),
        };
        let make_helpers = |prefix: &str, rows| {
            [3usize, 5, 7]
                .into_iter()
                .map(|sigma| {
                    ValueReadoutHelper::new(
                        layout,
                        sigma,
                        4.into(),
                        2,
                        7.into(),
                        ring.preimage_input(format!("{prefix}-switch-{sigma}"), (rows, 4), 7),
                        ring.preimage_input(format!("{prefix}-public-{sigma}"), (rows, 4), 7),
                        BggEncodingWire { vector: ring.zero((1, 4)), plaintext: None },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let wrong_helpers = make_helpers("readout-wrong", 2);
        assert!(matches!(
            readout.execute_with_preprocessed_artifacts(
                &compiler,
                &input,
                ring.preimage_input("readout-wrong-target", (2, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
                &wrong_helpers,
                ring.preimage_input("readout-wrong-normalization", (2, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
            ),
            Err(BggReadoutError::InvalidPreimageSchema)
        ));
        let helpers = make_helpers("readout", 4);
        let output = readout
            .execute_with_preprocessed_artifacts(
                &compiler,
                &input,
                ring.preimage_input("readout-target-decomposition", (4, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
                &helpers,
                ring.preimage_input("readout-normalization-decomposition", (4, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
            )
            .unwrap();
        let built = DslContext::new("cached-readout-no-decompose")
            .output("vector", output.vector)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
                .count(),
            0
        );
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::SequentialLoop(_))).count(),
            0
        );
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn public_readout_uses_masked_switched_decomposition_formula() {
        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 0 };
        let readout = ValueReadout::from_values(layout, &ring, vec![0, 1, 0, 1]).unwrap();
        let compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
        let input = BggPublicKeyWire {
            matrix: ring.input("public-readout-input", (2, 4)),
            reveal_plaintext: false,
        };
        let helpers = [3usize, 5, 7]
            .into_iter()
            .map(|sigma| {
                ValueReadoutPublicHelper::new(
                    layout,
                    sigma,
                    4.into(),
                    2,
                    7.into(),
                    ring.preimage_input(format!("public-readout-decomposition-{sigma}"), (4, 4), 7),
                    ring.input(format!("public-readout-mask-{sigma}"), (2, 4)),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let output = readout
            .project_public_with_preprocessed_artifacts(
                &compiler,
                &input,
                ring.preimage_input("public-readout-target-decomposition", (4, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
                &helpers,
                ring.preimage_input("public-readout-normalization-decomposition", (4, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
            )
            .unwrap();
        let built = DslContext::new("public-readout-trace")
            .output("output", output)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs)).count(),
            helpers.len() + 2,
            "public helper branches must be exactly A_mask * D(A_switched)"
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::RingAutomorphism { .. }))
                .count(),
            0,
            "public projection has no switched ciphertext vector term"
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
                .count(),
            0
        );
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::SequentialLoop(_))).count(),
            0
        );
        built.validate(&ParamEnv::default()).unwrap();
    }
}

#[cfg(test)]
mod noiseless_prf_tests {
    // Exact production PRF accumulator/readout execution with encoding errors disabled.
    use super::*;
    use crate::{
        bgg::ReadoutPreimageSchema,
        noise_refresh::{
            pbc::{
                PbcActiveCellIndex, PbcCell, PbcParameters, PbcRootSeed, PbcTrustedSelectorBits,
                clear_pbc_inner_product, generate_key_layout,
            },
            prf::{PreparedSparseLwrPrf, SparseLwrPrfProfile, SparseLwrPrfProgram},
        },
        program::ProgramValue,
    };
    use mxx_bgg::{BggPublicKeyCompiler, BggPublicKeyWire};
    use mxx_ir_core::{
        IntExpr, ParamEnv,
        artifact::{ArtifactConfidentiality, ArtifactType, ProductionId},
        encoding::spec_hash,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        ExecutionConfig, RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        backend::poly::cpu_backend,
        execute_in_session, execute_with_config,
        transcript::SamplingMode,
    };
    use num_traits::{One, Zero};
    use rand::Rng;

    #[test]
    fn test_lwr_prf_noiseless_round_trip() {
        let env = |suffix: &str, default: usize| {
            std::env::var(format!("MXX_EXPONENT_LUT_REFRESH_{suffix}"))
                .map(|v| v.parse::<usize>().expect("integer parameter"))
                .unwrap_or(default)
        };
        let log_n = env("MIN_LOG_RING_DIMENSION", 2);
        let n = 1usize << log_n;
        let q_native = env("SPARSE_LWR_MODULUS", 8);
        let p = env("SPARSE_LWR_OUTPUT_MODULUS", 4);
        let universe = env("SPARSE_LWR_UNIVERSE", 4);
        let base_bits = env("BASE_BITS", 4) as u32;
        let depth = env("MIN_CRT_DEPTH", 2);
        assert!(depth >= 2, "exercise at least two CRT limbs");
        assert!(q_native.is_power_of_two() && q_native >= 4 && q_native <= 2 * n);
        assert!(p.is_power_of_two() && p >= 4 && p <= q_native);
        let params =
            DCRTPolyParams::new(n as u32, depth, env("CRT_BITS", 20), base_bits, None, None);
        let modulus = BigInt::from(params.modulus().as_ref().clone());
        let layout = BggSamplerLayout {
            modulus: modulus.clone().into(),
            ring_dimension: n.into(),
            secret_dimension: 2,
            digit_count: params.modulus_digits(),
            gadget_base: (BigInt::one() << base_bits).into(),
        };
        let width_log2 = q_native.ilog2() - 1;
        let profile = SparseLwrPrfProfile::new(
            crate::ExpLayout { stride_log2: log_n as u32 - width_log2, width_log2 },
            p / 2,
        )
        .unwrap();
        let mut rng = rand::rng();
        let support = [rng.random_range(0..universe)];
        let generated = generate_key_layout(
            &PbcParameters::custom(universe, 1, 2, 3, 1, None),
            PbcRootSeed(rng.random()),
            &support,
        )
        .unwrap();
        let public_layout = &generated.public_layout;
        let active = PbcActiveCellIndex::build(public_layout).unwrap();
        let program = SparseLwrPrfProgram::new(
            profile,
            public_layout.bucket_width,
            public_layout.parameters.bucket_count,
        )
        .unwrap();
        let ring = layout.ring();
        let columns = layout.public_key_columns();
        let sampler = ExponentLutEncodingSampler {
            layout: layout.clone(),
            exponent_layout: profile.layout,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let t = sampler.sample_secret().unwrap();
        let s = sampler.sample_secret().unwrap();
        let hash = ring.bytes_input("prf-hash", 32);
        let key_id = rng.random();
        let bits = PbcTrustedSelectorBits::from_schedule(&generated, &ring, key_id).unwrap();
        let runtime_bits = bits.runtime_binding(&ring).unwrap();
        let selectors = sampler
            .sample_pbc_selectors(
                bits.family().clone(),
                t.clone(),
                t.clone(),
                hash.clone(),
                public_layout,
                key_id,
            )
            .unwrap();
        let initial = mxx_bgg::BggPublicKeySampler { layout: layout.clone() }
            .sample(hash.clone(), b"prf-initial".as_slice(), &[false])
            .remove(1);
        let bgg_public = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: layout.gadget_base.clone(),
            digit_count: layout.digit_count.into(),
        };
        let scheme = crate::bgg::BggScheme::new(bgg_public, profile.layout);
        let labels = ring.input_family("prf-monomials", active.len(), (1, 1));
        let fixed_selectors = crate::bgg::encoding::SetupRhsFamily::new_for_layout(
            selectors.clone(),
            public_layout.layout_id,
        )
        .unwrap();
        let blocks =
            program.selection_inputs(public_layout, &fixed_selectors, &labels, key_id).unwrap();
        let (public, prepared) = PreparedSparseLwrPrf::new(
            &scheme,
            program.clone(),
            ProgramValue::Scalar(initial.clone()),
            &blocks,
        )
        .unwrap();
        let public_outputs = public;
        let helpers = sampler
            .sample_native_readout_helper_families(
                t.clone(),
                hash.clone(),
                profile.native_width(),
                b"prf-mask".as_slice(),
                b"prf-helper".as_slice(),
            )
            .unwrap();
        let gadget = ring.gadget(2, layout.gadget_base.clone(), layout.digit_count);
        let readout = program
            .prepare_readout(
                &scheme,
                &public_outputs,
                &(helpers.sigmas.clone(), helpers.switch_ciphertexts.clone(), gadget.clone()),
            )
            .unwrap();
        let public_output = program
            .project_readout(&scheme, &public_outputs, &readout, &helpers.public_masks)
            .unwrap();
        let context = DslContext::new("one-output-prf-preprocessing")
            .private_output("t", t)
            .unwrap()
            .public_output("initial", initial.matrix)
            .unwrap()
            .public_output("selectors", selectors)
            .unwrap()
            .public_output("output-public", public_output)
            .unwrap()
            .public_output("public-masks", helpers.public_masks.clone())
            .unwrap()
            .public_output("readout-target", readout.target_decompositions().clone())
            .unwrap()
            .public_output("readout-switch", readout.helper_switch_decompositions().clone())
            .unwrap()
            .public_output("readout-public", readout.helper_public_decompositions().clone())
            .unwrap()
            .public_output("readout-normalization", readout.normalization_decomposition().clone())
            .unwrap();
        let (context, descriptor) = prepared.export(&scheme, context, "fixed-prf").unwrap();
        // The exact operation schemas must survive the same serialization boundary
        // as the matrices. Online evaluation cannot reuse producer graph handles.
        let descriptor = serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let preprocessing = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let nonce = rng.random();
        let production = ProductionId {
            spec_hash: spec_hash(&preprocessing.source, &preprocessing.bindings).unwrap(),
            execution_nonce: nonce,
        };
        let manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &preprocessing)
                .unwrap();
        let imported_matrix = |name: &str, rows: usize, cols: usize, private: bool| {
            ring.artifact_input(
                production.clone(),
                name,
                (rows, cols),
                if private {
                    ArtifactConfidentiality::Private
                } else {
                    ArtifactConfidentiality::Public
                },
            )
        };
        let import_preimage_family = |name: &str| {
            let artifact = &manifest.artifacts[name];
            let ArtifactType::Preimage { matrix, max_coefficient_bound } = &artifact.artifact_type
            else {
                panic!("preimage schema")
            };
            ring.preimage_family_artifact_input(
                production.clone(),
                name,
                artifact.family_count.unwrap(),
                (matrix.rows, matrix.columns),
                max_coefficient_bound.clone(),
                ArtifactConfidentiality::Public,
            )
        };
        let import_matrix_family = |name: &str| {
            let artifact = &manifest.artifacts[name];
            let ArtifactType::Matrix(matrix) = &artifact.artifact_type else {
                panic!("matrix schema")
            };
            ring.family_artifact_input(
                production.clone(),
                name,
                artifact.family_count.unwrap(),
                (matrix.rows, matrix.columns),
                ArtifactConfidentiality::Public,
            )
        };
        let imported_t = imported_matrix("t", 1, 2, true);
        let imported_initial = BggPublicKeyWire {
            matrix: imported_matrix("initial", 2, columns, false),
            reveal_plaintext: false,
        };
        let initial_vector = sampler
            .sample_encodings_for_public_matrices(
                s.clone(),
                Some(imported_t.clone()),
                &[imported_initial],
                &[ring.identity(1)],
            )
            .unwrap()
            .remove(0);
        let prepared = PreparedSparseLwrPrf::import(
            &scheme,
            program.clone(),
            &descriptor,
            production.clone(),
            descriptor.identity().unwrap(),
        )
        .unwrap();
        let accumulation =
            prepared.evaluate(&scheme, ProgramValue::Scalar(initial_vector)).unwrap();

        let normalization = &manifest.artifacts["readout-normalization"].artifact_type;
        let ArtifactType::Preimage { matrix, max_coefficient_bound } = normalization else {
            panic!("normalization preimage")
        };
        let schema = ReadoutPreimageSchema::new(
            layout.gadget_base.evaluate(&ParamEnv::default()).unwrap(),
            layout.digit_count,
            max_coefficient_bound.clone(),
        )
        .unwrap();
        let imported_readout = BggPreparedReadout::new(
            import_preimage_family("readout-target"),
            schema.clone(),
            helpers.sigmas.as_ref().clone(),
            import_preimage_family("readout-switch"),
            import_preimage_family("readout-public"),
            ring.preimage_artifact_input(
                production.clone(),
                "readout-normalization",
                (matrix.rows, matrix.columns),
                max_coefficient_bound.clone(),
                ArtifactConfidentiality::Public,
            ),
            schema,
            1,
        )
        .unwrap();
        let mask_publics = import_matrix_family("public-masks");
        let masks = {
            let source_family = mask_publics;
            mxx_dsl::parallel(source_family.count().clone(), |family_index| {
                let matrix = source_family.at(&family_index);
                let index = family_index;
                Ok({
                    let sigma = profile
                        .layout
                        .automorphism_index_expression(
                            n,
                            IntExpr::Add(
                                Box::new(index.expression()?),
                                Box::new(IntExpr::constant(1)),
                            ),
                        )
                        .unwrap();
                    let mask_sigma = s.clone().ring_automorphism(sigma);
                    mxx_bgg::BggEncodingSampler {
                        layout: layout.clone(),
                        gaussian_sigma: sampler.gaussian_sigma.clone(),
                        gaussian_max_coefficient_bound: sampler
                            .gaussian_max_coefficient_bound
                            .clone(),
                    }
                    .sample(
                        s.clone(),
                        Some(mask_sigma),
                        &[
                            BggPublicKeyWire {
                                matrix: ring.zero((2, columns)),
                                reveal_plaintext: false,
                            },
                            BggPublicKeyWire { matrix, reveal_plaintext: false },
                        ],
                        &[ring.identity(1)],
                    )
                    .unwrap()
                    .remove(1)
                    .vector
                })
            })
        }
        .unwrap();
        let output = program
            .evaluate_readout(&scheme, &accumulation, &imported_readout, &masks)
            .unwrap()
            .at(0);

        // The oracle uses the independent signed LWR definition, not the compiler's table.
        let g = ring.input("expected-value", (1, 1));
        let anchor = s * import_matrix_family("output-public").at(0) - g * (imported_t * gadget);
        let online = DslContext::new("noiseless-prf-online")
            .private_output("vector", output)
            .unwrap()
            .private_output("anchor", anchor)
            .unwrap()
            .build()
            .unwrap();
        assert!(!online.graph.scopes().values().flat_map(|scope| scope.nodes()).any(
            |node| matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. })
        ));
        online
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .unwrap();
        let mut backend = cpu_backend([params.clone()]);
        let mut runtime_inputs = runtime_bits.materialize(&mut backend).unwrap();
        runtime_inputs
            .insert("prf-hash".into(), RuntimeValue::Bytes(rng.random::<[u8; 32]>().to_vec()));
        std::fs::create_dir_all("test_data/prf_noiseless_round_trip").unwrap();
        let directory = tempfile::tempdir_in("test_data/prf_noiseless_round_trip").unwrap();
        for z in 0..q_native {
            let mut public_vector =
                (0..universe).map(|_| rng.random_range(0..q_native as u64)).collect::<Vec<_>>();
            public_vector[support[0]] = z as u64;
            assert_eq!(
                clear_pbc_inner_product(
                    public_layout,
                    generated.private_schedule(),
                    &public_vector,
                    q_native as u64
                )
                .unwrap(),
                z as u64
            );
            let magnitude = BigInt::from(2 * (p * (z % (q_native / 2)) / q_native) + 1);
            let doubled = if z < q_native / 2 { magnitude } else { -magnitude };
            let g = (doubled * ((&modulus + BigInt::one()) / BigInt::from(2u8))) % &modulus;
            let g = if g < BigInt::zero() { g + &modulus } else { g };
            let expected = DCRTPolyMatrix::from_poly_vec_row(
                &params,
                vec![DCRTPoly::from_biguints(&params, &[g.to_biguint().unwrap()])],
            );
            let monomials = active
                .iter()
                .map(|(bucket, slot, _)| {
                    let value = match public_layout.cells[bucket][slot] {
                        PbcCell::Real { coordinate, .. } => public_vector[coordinate] as usize,
                        PbcCell::Dummy => 0,
                        PbcCell::Padding => unreachable!(),
                    };
                    let exponent = value << profile.layout.stride_log2;
                    let poly = DCRTPoly::const_rotate_poly(&params, exponent % n);
                    let poly = if exponent / n % 2 == 0 { poly } else { -poly };
                    RuntimeValue::Matrix(
                        DCRTPolyMatrix::from_poly_vec_row(&params, vec![poly]).into(),
                    )
                })
                .collect::<Vec<_>>();
            // Labels are preprocessing-fixed: each exhaustive native-domain case
            // receives its own production and reopened artifact store.
            let case_directory = tempfile::tempdir_in(directory.path()).unwrap();
            let mut store = FileArtifactStore::new(case_directory.path()).unwrap();
            let mut inputs = runtime_inputs.clone();
            inputs.insert("prf-monomials".into(), RuntimeValue::IndexedFamily(monomials));
            execute_in_session(&preprocessing, &mut backend, inputs, &mut store, nonce)
                .expect("production PRF preprocessing");
            drop(store);
            let mut store = FileArtifactStore::open(case_directory.path()).unwrap();
            let manifest = store.load_manifest(&production).unwrap();
            let online = online
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production.clone(), manifest)]),
                )
                .unwrap();
            for run in 0..2 {
                let mut result = execute_with_config(
                    &online,
                    &mut backend,
                    BTreeMap::from([(
                        "expected-value".into(),
                        RuntimeValue::Matrix(expected.clone().into()),
                    )]),
                    &mut store,
                    SamplingMode::Fresh,
                    ExecutionConfig::default(),
                )
                .expect("production PRF online execution");
                result.materialize_output("vector", &mut backend, &mut store).unwrap();
                result.materialize_output("anchor", &mut backend, &mut store).unwrap();
                let RuntimeValue::Matrix(actual) = &result.outputs["vector"] else {
                    panic!("matrix")
                };
                let RuntimeValue::Matrix(expected) = &result.outputs["anchor"] else {
                    panic!("matrix")
                };
                assert_eq!(actual, expected, "exact PRF round trip z={z} fresh_run={run}");
            }
        }
        println!(
            "noiseless_prf_pass N={n} depth={depth} q={modulus} base_bits={base_bits} Q={q_native} P={p} inputs={q_native} fresh_runs=2"
        );
    }
}

impl BggRefreshParameters {
    /// Samples one online anchor per exact P group and restores original slot order.
    pub(crate) fn shared_decoder_bases(
        &self,
        secret: Mat,
        public_b: &[Mat],
    ) -> Result<(Vec<Mat>, Vec<Mat>), crate::noise_refresh::setup::RefreshSetupError> {
        let groups = group_refresh_levels(self.switch_moduli.iter().cloned())
            .into_iter()
            .map(|(modulus, slots)| (IntExpr::constant(BigInt::from(modulus)), slots))
            .collect::<Vec<_>>();
        Ok(crate::noise_refresh::refresh::prepare_shared_decoder_bases::<BggScheme>(
            &groups,
            public_b,
            &secret,
            &(self.encoding_error_sigma.clone(), self.encoding_error_bound.clone()),
        )?)
    }
}

impl RefreshCompiler {
    /// Binds online PRF outputs to an imported preprocessing setup.
    ///
    /// The preprocessing import contains only setup-fixed public projections,
    /// public decoder targets, typed `K_t` preimages, and readout artifacts;
    /// it never contains the online decoder anchor `b`. Label-dependent
    /// encoding vectors and the secret `s` enter here from the online
    /// evaluator. This method performs canonical mask/fresh routing, samples
    /// `b`, and only then constructs the refresh packages.
    pub fn bind_online_outputs(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        setup: &ImportedRefreshSetup,
        online_secret_s: Mat,
        state: BggEncodingWire,
        program: &SparseLwrPrfProgram,
        outputs: &Family<Mat>,
    ) -> Result<RefreshSetupManifest, RefreshSetupError> {
        if setup.parameters().refresh.crt_plaintext_moduli != self.crt_plaintext_moduli ||
            setup.parameters().refresh.reconstruction_coefficients !=
                self.reconstruction_coefficients ||
            setup.parameters().refresh.full_modulus != self.full_modulus ||
            setup.declaration().readout_identity != program.readout.identity()
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        if !same_matrix_type(
            online_secret_s.matrix_type(),
            &setup
                .parameters()
                .layout
                .ring()
                .matrix_type((1, setup.parameters().layout.secret_dimension)),
        ) || !same_matrix_type(
            state.vector.matrix_type(),
            &setup
                .parameters()
                .layout
                .ring()
                .matrix_type((1, setup.parameters().layout.public_key_columns())),
        ) {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        // Sample one fresh error per distinct public B_P in this execution.
        // All anchors reduce the same integer secret; switching or independently
        // resampling the secret would change the defining decoder equation.
        let (public_b_by_slot, decoder_bases_by_slot) =
            setup.parameters().shared_decoder_bases(online_secret_s, &setup.public_b)?;
        // Bind the actual output family once to the setup's canonical label schedule.
        // The immutable setup already owns its sampling parameters; copying them
        // into a temporary wrapper would only compare those copies with themselves.
        let parameters = setup.parameters();
        parameters.validate()?;
        let contract = RefreshPrfContract::from_program(program);
        let labels = crate::noise_refresh::refresh::RefreshPrfLabelIndex::new(
            parameters.refresh_id,
            &parameters.transcript_topology(),
        )?;
        parameters.validate_prf(&contract)?;
        if setup.declaration().pbc_layout_id.0 == [0; 32] {
            return Err(RefreshError::PrfOutputMismatch.into());
        }
        let online = RefreshPrfFamilyMaterial::new(outputs.clone(), labels)?;
        let ring = parameters.layout.ring();
        let columns = parameters.layout.public_key_columns();
        let mask_offsets = parameters
            .mask_base_p_digit_counts
            .iter()
            .zip(parameters.mask_lifts()?)
            .map(|(digits, lift)| {
                Ok(lift *
                    crate::bgg::public_half_offset_vector(
                        &ring,
                        parameters.base_p,
                        *digits,
                        columns,
                        true,
                    )?)
            })
            .collect::<Result<Vec<_>, RefreshError>>()?;
        let fresh_offset = crate::bgg::public_half_offset_vector(
            &ring,
            parameters.base_p,
            parameters.fresh_error_base_p_digit_count,
            columns,
            false,
        )?;
        let fresh_offsets = (0..parameters.refresh.crt_plaintext_moduli.len())
            .map(|slot| Ok(ring.polynomial([self.scale_expression(slot)?]) * fresh_offset.clone()))
            .collect::<Result<Vec<_>, RefreshError>>()?;
        let scheme = crate::bgg::BggScheme::new(
            BggPublicKeyCompiler {
                ring: compiler.ring.clone(),
                base: compiler.gadget_base.clone(),
                digit_count: compiler.digit_count.clone(),
            },
            compiler.exponent_layout,
        );
        let (masks, scaled_fresh) = aggregate_refresh_prf(
            &scheme,
            &online,
            setup.mask_route_rhs_decompositions.clone(),
            setup.fresh_route_rhs_decompositions.clone(),
            mask_offsets,
            fresh_offsets,
        )?;
        Ok(self.bind_imported_wires(
            compiler,
            state,
            setup.state.clone(),
            setup.a_prime.clone(),
            public_b_by_slot,
            setup.refresh_targets.clone(),
            scaled_fresh,
            masks,
            setup.state_scaling.clone(),
            decoder_bases_by_slot,
            setup.preimages.clone(),
        )?)
    }
}

#[cfg(test)]
pub(crate) mod refresh_tests {
    use super::*;
    use crate::{
        bgg::{
            identity_digest,
            noise::{
                compact_mask_routes, compact_noise_snapshot_plans, compact_readout_plan_schema,
            },
        },
        noise_refresh::{
            pbc::{PbcActiveCellIndex, PbcParameters, PbcRootSeed, generate_key_layout},
            prf::SparseLwrPrfProfile,
            setup::*,
            simulation::{
                AverageCaseConfig, MaskSmudgingParameters, RefreshRouteNoiseGains,
                RefreshTranscriptTopology,
            },
        },
        noise_utils::{NonnegativeRational, RightActionNoiseGain},
    };
    use mxx_bgg::BggSamplerLayout;
    use mxx_dsl::DslContext;
    use mxx_ir_core::node::NodeKind;
    use mxx_primitives::sampler::bounds::TrapdoorPreimageCutoffPolicy;
    use num_bigint::{BigInt, BigUint};
    use num_traits::{One, Zero};

    pub(crate) fn parameters(gadget_base: i32, secret_dimension: usize) -> BggRefreshParameters {
        let layout = BggSamplerLayout {
            modulus: 15.into(),
            ring_dimension: 4.into(),
            secret_dimension,
            digit_count: 2,
            gadget_base: gadget_base.into(),
        };
        BggRefreshParameters::new(
            [0x41; 32],
            2,
            secret_dimension,
            4,
            1,
            1,
            32,
            4,
            layout,
            RefreshCompiler {
                full_modulus: 15.into(),
                crt_plaintext_moduli: vec![3.into(), 5.into()],
                reconstruction_coefficients: vec![1.into(), 1.into()],
            },
            1.into(),
            1.into(),
            "setup-test",
        )
    }

    #[test]
    fn base_p_digit_counts_are_independent_of_gadget_digit_count() {
        let setup = parameters(4, 2);
        assert_eq!(setup.mask_base_p_digit_counts, vec![1; 2]);
        assert_eq!(setup.fresh_error_base_p_digit_count, 1);
        assert_eq!(setup.prf_component_count(), 4);
        assert_eq!(setup.layout.digit_count, 2);
        setup.validate_dimensions().expect("independent digit counts are valid");
    }

    #[test]
    fn setup_family_count_contract_accepts_the_exact_output_and_route_shapes() {
        let setup = parameters(4, 2);
        let output_count = setup.transcript_topology().joint_event_count().expect("output count");
        assert_eq!(output_count, 48);
        assert_eq!(
            setup
                .refresh_route_family_count(&setup.mask_base_p_digit_counts)
                .expect("mask route count"),
            32
        );
        assert_eq!(
            setup
                .refresh_route_family_count(&vec![setup.fresh_error_base_p_digit_count; 2])
                .expect("fresh route count"),
            32
        );
    }

    #[test]
    fn setup_route_family_count_rejects_overflow() {
        let mut overflowing = parameters(4, 2);
        overflowing.coefficient_count = usize::MAX;
        assert!(
            overflowing.refresh_route_family_count(&overflowing.mask_base_p_digit_counts).is_err()
        );
    }

    #[test]
    fn decoder_cutoff_policy_is_checked_against_official_on_the_same_setup() {
        let setup = parameters(4, 2);
        let official = setup
            .clone()
            .with_decoder_preimage_bound(TrapdoorPreimageCutoffPolicy::Default)
            .resolve_decoder_preimage_bound()
            .expect("default cutoff")
            .evaluate(&ParamEnv::default())
            .expect("concrete default cutoff")
            .to_biguint()
            .expect("positive default cutoff");
        for (offset, accepted) in [(-1_i32, false), (0, true), (1, true)] {
            let selected = if offset < 0 {
                &official - BigUint::from((-offset) as u32)
            } else {
                &official + BigUint::from(offset as u32)
            };
            let configured = setup.clone().with_decoder_preimage_bound(
                TrapdoorPreimageCutoffPolicy::Explicit(selected.clone()),
            );
            assert_eq!(configured.validate().is_ok(), accepted);
            if offset == 1 {
                assert_eq!(
                    configured
                        .resolve_decoder_preimage_bound()
                        .expect("explicit cutoff above official")
                        .evaluate(&ParamEnv::default())
                        .expect("concrete explicit cutoff")
                        .to_biguint()
                        .expect("positive explicit cutoff"),
                    selected
                );
            }
        }
    }

    #[test]
    fn decoder_cutoff_policy_rejects_zero_explicit_value() {
        let setup = parameters(4, 2);
        let zero = setup
            .with_decoder_preimage_bound(TrapdoorPreimageCutoffPolicy::Explicit(BigUint::zero()));
        assert!(zero.validate().is_err());
    }

    #[test]
    fn simulation_request_accepts_explicit_cutoff_above_official() {
        let setup = parameters(4, 2);
        let official = setup
            .resolve_decoder_preimage_bound()
            .expect("default cutoff")
            .evaluate(&ParamEnv::default())
            .expect("concrete default cutoff")
            .to_biguint()
            .expect("positive default cutoff");
        let selected = &official + BigUint::from(1_u8);
        let setup =
            setup.with_decoder_preimage_bound(TrapdoorPreimageCutoffPolicy::Explicit(selected));
        let pbc_parameters = PbcParameters::paper_evaluation(10, 4);
        let generated =
            generate_key_layout(&pbc_parameters, PbcRootSeed([0x77; 32]), &[0, 1, 2, 3])
                .expect("generated PBC layout");
        let profile =
            SparseLwrPrfProfile::new(crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 }, 1)
                .expect("sparse-LWR profile");
        let expected_plaintext = setup.layout.ring().polynomial([BigInt::from(0_u8).into()]);
        RefreshParameterSimulationRequest::new(
            setup,
            crate::noise_refresh::prf::SparseLwrPrfProgram::new(
                profile,
                generated.public_layout.bucket_width,
                generated.public_layout.parameters.bucket_count,
            )
            .expect("PRF Program"),
            generated.clone(),
            [0x78; 32],
            expected_plaintext,
            1,
        )
        .expect("explicit cutoff above official is accepted");
    }

    #[test]
    fn readout_switch_error_adds_target_plaintext_gains_only() {
        let mut rng = rand::rng();
        let generated = generate_key_layout(
            &PbcParameters::custom(4, 1, 2, 3, 1, None),
            PbcRootSeed(rand::Rng::random(&mut rng)),
            &[0],
        )
        .unwrap();
        let active = PbcActiveCellIndex::build(&generated.public_layout).unwrap();
        let profile =
            SparseLwrPrfProfile::new(crate::ExpLayout { stride_log2: 0, width_log2: 2 }, 2)
                .unwrap();
        let mut setup = parameters(4, 2);
        setup.base_p = 4;
        let program = SparseLwrPrfProgram::new(
            profile,
            generated.public_layout.bucket_width,
            generated.public_layout.parameters.bucket_count,
        )
        .unwrap();
        let snapshot = BggRefreshParameters::build_noise_snapshot_from_public_layout(
            &setup,
            &program,
            &generated.public_layout,
            &active,
            None,
        )
        .unwrap();
        let actual = snapshot.evaluate_pre_refresh().unwrap();
        let gains = &snapshot.readout_gains;
        let (target_residual, trace_residuals, normalization_residual) =
            snapshot.readout_residuals.as_ref().expect("approximate gadget residuals");
        let target_bound =
            &gains.target.l1 * &actual.program.output_state().bound + &target_residual.bound;
        let output_wire = snapshot.program.outputs().last().unwrap();
        let target_variance = actual.program.wire_states[output_wire]
            .scalar()
            .unwrap()
            .doubled_variance
            .scaled(&gains.target.l2_squared)
            .add(&target_residual.doubled_variance);
        let mut old_trace_bound = &gains.identity_trace.l1 * &target_bound;
        let mut old_trace_variance = target_variance.clone();
        for gain in &gains.nonidentity_trace {
            old_trace_bound += &gain.action.l1 * &target_bound +
                (BigUint::one() + &gain.helper_action.l1) * &snapshot.helper_error_cutoff;
            old_trace_variance =
                old_trace_variance.add(&target_variance.scaled(&gain.action.l2_squared)).add(
                    &snapshot
                        .helper_doubled_variance
                        .scaled(&(BigUint::one() + &gain.helper_action.l2_squared)),
                );
        }
        for residual in trace_residuals {
            old_trace_bound += &residual.bound;
            old_trace_variance = old_trace_variance.add(&residual.doubled_variance);
        }
        let branches = BigUint::from(gains.nonidentity_trace.len());
        // T'=[1,-3,-3,-1]: the former unit factor omitted exactly 7 B_chi
        // and 19 V_chi per nonidentity branch, before normalization.
        let worst_delta = &gains.normalization.l1 *
            &branches *
            BigUint::from(7u8) *
            &snapshot.helper_error_cutoff;
        let variance_delta = snapshot
            .helper_doubled_variance
            .scaled(&(&gains.normalization.l2_squared * branches * BigUint::from(19u8)));
        assert!(!worst_delta.is_zero() && !variance_delta.numerator.is_zero());
        assert_eq!(
            actual.readout_bound,
            &gains.normalization.l1 * old_trace_bound + worst_delta + &normalization_residual.bound
        );
        assert_eq!(
            actual.readout_variance,
            old_trace_variance
                .scaled(&gains.normalization.l2_squared)
                .add(&variance_delta)
                .add(&normalization_residual.doubled_variance)
        );
    }

    #[test]
    fn switched_refresh_snapshot_tracks_exact_rounding_and_unscaled_decoder() {
        let mut setup = parameters(4, 2);
        setup.layout.modulus = 105.into();
        setup.layout.digit_count = 6;
        setup.refresh.full_modulus = 105.into();
        setup.refresh.crt_plaintext_moduli = vec![3.into(), 5.into(), 7.into()];
        setup.refresh.reconstruction_coefficients = vec![1.into(); 3];
        setup.switch_moduli = vec![15u8.into(), 15u8.into(), 21u8.into()];
        setup.mask_base_p_digit_counts = vec![2; 3];
        let profile =
            SparseLwrPrfProfile::new(crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 }, 1)
                .unwrap();
        let generated = generate_key_layout(
            &PbcParameters::paper_evaluation(10, 4),
            PbcRootSeed(rand::random()),
            &[0, 1, 2, 3],
        )
        .unwrap();
        let active = PbcActiveCellIndex::build(&generated.public_layout).unwrap();
        let program = SparseLwrPrfProgram::new(
            profile,
            generated.public_layout.bucket_width,
            generated.public_layout.parameters.bucket_count,
        )
        .unwrap();
        let mut snapshot = BggRefreshParameters::build_noise_snapshot_from_public_layout(
            &setup.clone(),
            &program,
            &generated.public_layout,
            &active,
            None,
        )
        .unwrap();
        let candidate = snapshot.simulate().unwrap();
        assert!(!candidate.refresh.accepted);
        eprintln!(
            "small switched-refresh candidate (q=105,n=4,P=[15,15,21],d_m=[2,2,2]): {}",
            serde_json::to_string(&candidate.refresh).unwrap()
        );
        // Real compact routes must depend on the selected J, and low-P decoder
        // rows retain the original number of target columns.
        assert_eq!(setup.decoder_digit_count(0).unwrap(), 4);
        let switched_plans = compact_mask_routes(&setup).unwrap().0;
        let mut unswitched_setup = setup.clone();
        unswitched_setup.switch_moduli.fill(105u8.into());
        assert_ne!(switched_plans, compact_mask_routes(&unswitched_setup).unwrap().0);
        let mut ragged_setup = setup.clone();
        ragged_setup.mask_base_p_digit_counts = vec![1, 3, 2];
        let ragged_snapshot = BggRefreshParameters::build_noise_snapshot_from_public_layout(
            &ragged_setup,
            &program,
            &generated.public_layout,
            &active,
            None,
        )
        .unwrap();
        assert_eq!(ragged_snapshot.refresh.mask_digit_counts, vec![1, 3, 2]);
        assert_ne!(
            ragged_snapshot.refresh.route_gains.as_ref().unwrap().mask[0],
            ragged_snapshot.refresh.route_gains.as_ref().unwrap().mask[1]
        );

        // Isolate the transfer equation after exercising the real setup path.
        snapshot.initial_state_bound = 1u8.into();
        snapshot.helper_error_cutoff = 1u8.into();
        snapshot.initial_state_doubled_variance = NonnegativeRational::zero();
        snapshot.helper_doubled_variance =
            NonnegativeRational::new(4u8.into(), 1u8.into()).unwrap();
        let unit = RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() };
        snapshot.refresh.route_gains = Some(RefreshRouteNoiseGains {
            state_scaling: vec![unit.clone(); 3],
            mask: vec![unit.clone(); 3],
            fresh: vec![unit; 3],
            decoder: vec![RightActionNoiseGain { l1: 5u8.into(), l2_squared: 25u8.into() }; 3],
        });
        let mut pre = snapshot.evaluate_pre_refresh().unwrap();
        pre.readout_bound = 0u8.into();
        pre.readout_variance = NonnegativeRational::zero();
        let report = snapshot.simulate_worst_from_pre_refresh(pre.clone()).unwrap();
        let slot = &report.refresh.slots[1];
        // n=4, J=7: ceil(1/7 + 18/7) + 5 = 8, while separate
        // ceilings would incorrectly produce 9. Decoder noise stays 5.
        assert_eq!(slot.pre_switch_bound, 1u8.into());
        assert_eq!(slot.lift, 7u8.into());
        assert_eq!(slot.spacing, 3u8.into());
        assert_eq!(slot.decoder_bound, 5u8.into());
        assert_eq!(slot.residual_bound, 8u8.into());
        assert_eq!(slot.mask_bound, 2u8.into());
        assert_eq!(slot.rounding_lhs, 20u8.into());
        assert!(!report.refresh.accepted);

        let config = AverageCaseConfig { allow_average_acceptance: true, ..Default::default() };
        let average = snapshot.simulate_average_with_pre_refresh(&config, pre.clone()).unwrap();
        assert_eq!(
            average.refresh.slots[1].variance,
            NonnegativeRational::new(100u8.into(), 1u8.into()).unwrap()
        );
        assert_eq!(
            average.refresh.slots[1].variance_breakdown.mask_uniform_moments,
            snapshot.refresh.mask_uniform_moments(1)
        );
        assert_eq!(
            average.refresh.slots[1].variance_breakdown.fresh_uniform_moments,
            snapshot.refresh.fresh_uniform_moments()
        );

        snapshot.refresh.switch_moduli.fill(105u8.into());
        let comparison = snapshot.simulate_worst_from_pre_refresh(pre.clone()).unwrap();
        assert_eq!(comparison.refresh.slots[1].rounding_numerator, 0u8.into());
        assert_eq!(comparison.refresh.slots[1].residual_bound, 6u8.into());
        snapshot.refresh.mask_digit_counts = vec![1, 2, 3];
        snapshot.refresh.transcript_topology.as_mut().unwrap().mask_digit_counts = vec![1, 2, 3];
        let ragged = snapshot.simulate_worst_from_pre_refresh(pre.clone()).unwrap();
        assert_eq!(
            ragged.refresh.slots.iter().map(|slot| slot.mask_modulus.clone()).collect::<Vec<_>>(),
            vec![2u8.into(), 4u8.into(), 8u8.into()]
        );
        for invalid in [0u8, 10, 25, 7] {
            snapshot.refresh.switch_moduli[1] = invalid.into();
            assert!(snapshot.simulate_worst_from_pre_refresh(pre.clone()).is_err());
        }
    }

    #[test]
    fn graph_free_snapshot_uses_canonical_setup_without_lowering_graph() {
        let setup = parameters(4, 2);
        let pbc_parameters = PbcParameters::paper_evaluation(10, 4);
        let generated =
            generate_key_layout(&pbc_parameters, PbcRootSeed([0x79; 32]), &[0, 1, 2, 3])
                .expect("generated PBC layout");
        let profile =
            SparseLwrPrfProfile::new(crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 }, 1)
                .expect("sparse-LWR profile");
        let expected_plaintext = setup.layout.ring().polynomial([BigInt::from(0_u8).into()]);
        let request = RefreshParameterSimulationRequest::new(
            setup,
            crate::noise_refresh::prf::SparseLwrPrfProgram::new(
                profile,
                generated.public_layout.bucket_width,
                generated.public_layout.parameters.bucket_count,
            )
            .expect("PRF Program"),
            generated.clone(),
            [0x7a; 32],
            expected_plaintext,
            1,
        )
        .expect("simulation request");
        let snapshot = request.build_noise_snapshot_only().expect("compact noise snapshot");
        assert_eq!(snapshot.setup_identity().len(), 32);

        // The graph-free cached-route path must produce the same authoritative
        // snapshot as the ordinary compact builder.  This guards the search
        // optimization against silently changing a route gain or refresh
        // topology while scanning `d_m`.
        let setup = parameters(4, 2);
        let pbc_layout =
            generate_key_layout(&pbc_parameters, PbcRootSeed([0x79; 32]), &[0, 1, 2, 3])
                .expect("generated PBC layout")
                .public_layout;
        let active_cells = PbcActiveCellIndex::build(&pbc_layout).expect("active-cell index");
        // The optimized evaluator's exact Program must also work without
        // inventing structural families or retaining the old selector gates.
        let prf = SparseLwrPrfProgram::new(
            profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )
        .unwrap();
        let direct = prf.accumulation.clone();
        assert_eq!(snapshot.program.id(), direct.id());
        assert!(snapshot.family_active_counts.is_empty());
        assert!(snapshot.inputs.public_family_errors.is_empty());
        let fresh = setup.average_initial_variance().unwrap();
        let bound = setup
            .encoding_error_bound
            .evaluate(&ParamEnv::default())
            .unwrap()
            .to_biguint()
            .unwrap();
        for (rhs, count) in direct.rhs_inputs().keys().zip(active_cells.bucket_active_widths()) {
            let error = &snapshot.inputs.rhs_errors[rhs];
            assert_eq!(error.bound, &bound * BigUint::from(count));
            assert_eq!(error.doubled_variance, fresh.scaled(&BigUint::from(count)));
        }

        let schema = compact_readout_plan_schema(&setup, &prf).unwrap();
        let (direct_plan, _) = compact_noise_snapshot_plans(&setup, &direct, &[], &schema).unwrap();
        assert_eq!(direct_plan.program.gates().len(), prf.block_count());
        direct_plan.validate(&direct, &[]).unwrap();
        assert!(
            direct_plan
                .validate(
                    &prf.accumulation,
                    &active_cells.bucket_active_widths().collect::<Vec<_>>(),
                )
                .is_err()
        );

        let (gain_plan, invariant_routes) =
            BggRefreshParameters::noise_gain_and_route_invariants_from_public_layout(
                &setup,
                &prf,
                &pbc_layout,
            )
            .expect("invariant compact plans");
        let cached = BggRefreshParameters::build_noise_snapshot_from_public_layout(
            &setup,
            &prf,
            &pbc_layout,
            &active_cells,
            Some((&gain_plan, Some(&invariant_routes))),
        )
        .expect("cached compact noise snapshot");
        assert_eq!(snapshot.setup_identity(), cached.setup_identity());
        assert_eq!(snapshot.gain_plan, cached.gain_plan);
        assert_eq!(snapshot.refresh, cached.refresh);
        assert_eq!(snapshot.family_active_counts, cached.family_active_counts);

        // Search's borrowed fast path must make the same hard/average
        // decision as the ordinary full report path.  It reuses the exact
        // primitive-gate visitor result and evaluates only refresh equations.
        let pre_refresh = snapshot.evaluate_pre_refresh().expect("pre-refresh evaluation");
        let worst = snapshot.simulate().expect("worst report");
        assert_eq!(
            worst.refresh.accepted,
            snapshot
                .worst_refresh_accepted_from_pre_refresh(&pre_refresh)
                .expect("cached worst refresh")
        );
        let average_config = AverageCaseConfig {
            failure_exponent: 20,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        };
        let average = snapshot.simulate_average(&average_config).expect("average report");
        assert_eq!(
            average.accepted,
            snapshot
                .average_refresh_accepted_from_pre_refresh(&average_config, &pre_refresh)
                .expect("cached average refresh")
        );

        let baseline_worst = snapshot.simulate().expect("baseline worst report");
        let baseline_average =
            snapshot.simulate_average(&average_config).expect("baseline average report");

        // Repeat the comparison for a second mask digit count.  This checks
        // the actual materialized worst/average reports (including every
        // slot breakdown), not only their acceptance booleans, while reusing
        // the invariant gate/readout routes from the first setup.
        for mask_digits in [1usize, 2usize] {
            let mut setup_for_digits = parameters(4, 2);
            setup_for_digits.mask_base_p_digit_counts.fill(mask_digits);
            let expected_plaintext =
                setup_for_digits.layout.ring().polynomial([BigInt::from(0_u8).into()]);
            let full_request = RefreshParameterSimulationRequest::new(
                setup_for_digits.clone(),
                crate::noise_refresh::prf::SparseLwrPrfProgram::new(
                    profile,
                    generated.public_layout.bucket_width,
                    generated.public_layout.parameters.bucket_count,
                )
                .expect("PRF Program"),
                generated.clone(),
                [0x7a; 32],
                expected_plaintext,
                1,
            )
            .expect("per-digit-count simulation request");
            let full = full_request.build_noise_snapshot_only().expect("ordinary compact snapshot");
            let cached = BggRefreshParameters::build_noise_snapshot_from_public_layout(
                &setup_for_digits,
                &prf,
                &pbc_layout,
                &active_cells,
                Some((&gain_plan, Some(&invariant_routes))),
            )
            .expect("cached compact snapshot");

            let full_worst = full.simulate().expect("ordinary worst report");
            let cached_worst = cached.simulate().expect("cached worst report");
            assert_eq!(full_worst, cached_worst);
            assert_eq!(full_worst.refresh.slots, cached_worst.refresh.slots);
            assert_eq!(
                full_worst.refresh.rounding_accepted,
                cached_worst.refresh.rounding_accepted
            );
            assert_eq!(
                full_worst.refresh.mask_smudging_accepted,
                cached_worst.refresh.mask_smudging_accepted
            );
            let full_average =
                full.simulate_average(&average_config).expect("ordinary average report");
            let cached_average =
                cached.simulate_average(&average_config).expect("cached average report");
            assert_eq!(full_average, cached_average);
            assert_eq!(full_average.refresh.slots, cached_average.refresh.slots);
            assert_eq!(
                full_average.refresh.rounding_accepted,
                cached_average.refresh.rounding_accepted
            );
            assert_eq!(
                full_average.refresh.mask_smudging_accepted,
                cached_average.refresh.mask_smudging_accepted
            );
            if mask_digits == 2 {
                // Ensure that this is a real d_m test rather than two
                // aliases of the same snapshot.  The small fixture is
                // intentionally rejected by rounding/smudging, so the
                // boundary is exercised without depending on acceptance.
                assert_ne!(full_worst.refresh, baseline_worst.refresh);
                assert_ne!(full_average.refresh, baseline_average.refresh);
            }
            let cached_pre_refresh =
                cached.evaluate_pre_refresh().expect("cached per-digit pre-refresh evaluation");
            assert_eq!(
                full_average.accepted,
                cached
                    .average_refresh_accepted_from_pre_refresh(
                        &average_config,
                        &cached_pre_refresh,
                    )
                    .expect("cached average acceptance")
            );
        }
    }

    #[test]
    fn average_initial_variance_is_derived_from_setup_sigma() {
        let setup = parameters(4, 2);
        assert_eq!(
            setup.average_initial_variance().expect("rational setup sigma"),
            NonnegativeRational::new(4u8.into(), 1u8.into()).expect("positive variance")
        );
    }

    #[test]
    fn refresh_noise_parameters_bind_full_q_topology_and_digit_variances() {
        let setup = parameters(4, 2);
        let refresh = setup.refresh_noise_parameters().expect("complete refresh noise schema");
        assert_eq!(refresh.global_modulus, Some(BigUint::from(15u8)));
        assert_eq!(refresh.plaintext_moduli, vec![BigUint::from(3u8), BigUint::from(5u8)]);
        assert_eq!(
            refresh.transcript_topology,
            Some(RefreshTranscriptTopology {
                crt_slot_count: 2,
                component_columns: 4,
                coefficient_count: 4,
                mask_digit_counts: vec![1; 2],
                fresh_digit_count: 1,
            })
        );
        assert_eq!(
            refresh.smudging,
            Some(MaskSmudgingParameters { lambda_mask: 32, transcript_coordinates: 32 })
        );
        assert_eq!(
            refresh
                .transcript_topology
                .as_ref()
                .expect("transcript topology")
                .joint_event_count()
                .expect("joint event count"),
            48
        );
        assert!(refresh.route_gains.is_none());
    }

    #[test]
    fn refresh_family_counts_include_dummy_and_exclude_padding() {
        let parameters = PbcParameters::custom(8, 1, 2, 3, 1, None);
        let layout = PbcPublicLayout::build(
            &parameters,
            crate::noise_refresh::pbc::derive_attempt_seed(PbcRootSeed([0x2a; 32]), 0),
            0,
        )
        .expect("PBC layout");
        let active = PbcActiveCellIndex::build(&layout).expect("active-cell index");
        for (bucket, row) in layout.cells.iter().enumerate() {
            let nonpadding = row
                .iter()
                .filter(|cell| !matches!(cell, crate::noise_refresh::pbc::PbcCell::Padding))
                .count();
            assert_eq!(active.bucket_active_width(bucket), Some(nonpadding));
            assert!(
                row.iter().any(|cell| matches!(cell, crate::noise_refresh::pbc::PbcCell::Dummy))
            );
        }
    }

    #[test]
    fn setup_gadget_changes_route_gain_without_changing_digit_distribution() {
        let pbc_parameters = PbcParameters::custom(4, 1, 2, 2, 1, None);
        let layout = PbcPublicLayout::build(
            &pbc_parameters,
            crate::noise_refresh::pbc::derive_attempt_seed(PbcRootSeed(rand::random()), 0),
            0,
        )
        .unwrap();
        let profile =
            SparseLwrPrfProfile::new(crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 }, 1)
                .unwrap();
        let program =
            SparseLwrPrfProgram::new(profile, layout.bucket_width, layout.parameters.bucket_count)
                .unwrap();
        let snapshot = |base| {
            let setup = parameters(base, 2);
            let (gains, routes) = setup
                .noise_gain_and_route_invariants_from_public_layout(&program, &layout)
                .unwrap();
            setup.build_noise_snapshot(program.clone(), &layout, gains, routes).unwrap()
        };
        let first = snapshot(4);
        let second = snapshot(8);
        assert_ne!(
            first.refresh.route_gains.as_ref().unwrap().mask[0].l2_squared,
            second.refresh.route_gains.as_ref().unwrap().mask[0].l2_squared,
        );
        assert_eq!(first.refresh.mask_uniform_moments(0), second.refresh.mask_uniform_moments(0));
        assert_eq!(first.refresh.fresh_uniform_moments(), second.refresh.fresh_uniform_moments());
    }

    #[test]
    fn setup_requires_secret_dimension_two_and_power_of_two_gadget_base() {
        assert!(parameters(4, 1).validate_dimensions().is_err());
        assert!(parameters(3, 2).validate_dimensions().is_err());
    }

    #[test]
    fn setup_identity_binds_mask_and_fresh_digit_counts_and_security() {
        let setup = parameters(4, 2);
        let profile =
            SparseLwrPrfProfile::new(crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 }, 1)
                .expect("profile");
        let program = SparseLwrPrfProgram::new(profile, 1, 1).expect("program");
        let contract = RefreshPrfContract::from_program(&program);
        let restored: RefreshPrfContract =
            serde_json::from_slice(&serde_json::to_vec(&contract).unwrap()).unwrap();
        assert_eq!(restored, contract);
        assert_eq!(restored.q(), profile.q().unwrap());
        assert_eq!(restored.p(), profile.p().unwrap());
        assert_eq!(restored.ring_dimension(), profile.ring_dimension().unwrap());
        let decoder_bound = 1.into();
        let identity = |setup: &BggRefreshParameters| {
            identity_digest(
                setup,
                &contract,
                crate::noise_refresh::pbc::PbcLayoutId([9; 32]),
                &decoder_bound,
            )
        };
        let baseline = identity(&setup);
        let mut changed_mask = setup.clone();
        changed_mask.mask_base_p_digit_counts.fill(2);
        assert_ne!(baseline, identity(&changed_mask));
        let mut changed_fresh = setup.clone();
        changed_fresh.fresh_error_base_p_digit_count = 2;
        assert_ne!(baseline, identity(&changed_fresh));
        let mut changed_security = setup;
        changed_security.mask_statistical_security_bits = 33;
        assert_ne!(baseline, identity(&changed_security));
    }

    #[test]
    fn online_decoder_base_samples_one_error_without_exporting_the_anchor() {
        let setup = parameters(4, 2);
        let ring = setup.layout.ring();
        let secret = ring.zero((1, 2));
        let public_b = ring.zero((2, 8));
        let (_, bases) = setup.shared_decoder_bases(secret, &[public_b]).unwrap();
        let base = bases[0].clone();
        assert_eq!(base.matrix_type().rows, 1.into());
        assert_eq!(base.matrix_type().columns, 8.into());

        let consumer = ring.input("decoder-anchor-consumer", (8, 1));
        let context = DslContext::new("online-decoder-base-test")
            .private_output("decoder-result", base * consumer)
            .expect("decoder result output")
            .build()
            .expect("online decoder graph");
        let gaussian_nodes = context
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::GaussianSample { .. }))
            .count();
        assert_eq!(gaussian_nodes, 1);
        assert_eq!(context.graph.outputs().len(), 1);
        assert!(context.graph.outputs().contains_key("decoder-result"));
    }

    #[derive(Default)]
    pub(crate) struct NumericRefreshBounds {
        bounds: BTreeMap<String, BigUint>,
        paired: Vec<(BigUint, BigUint)>,
    }

    pub(crate) fn assert_numeric_refresh_outputs(
        outputs: BTreeMap<String, (BigUint, Vec<BigUint>)>,
        bounds: &NumericRefreshBounds,
    ) {
        let centered = outputs
            .into_iter()
            .map(|(name, (modulus, coefficients))| {
                let half = &modulus / 2u8;
                let coefficients = coefficients
                    .into_iter()
                    .map(|value| {
                        if value > half {
                            BigInt::from(value) - BigInt::from(modulus.clone())
                        } else {
                            BigInt::from(value)
                        }
                    })
                    .collect::<Vec<_>>();
                let bound = bounds.bounds.get(&name).cloned().unwrap_or_default();
                assert!(
                    coefficients.iter().all(|value| value.magnitude() <= &bound),
                    "{name}: observed centered operational error exceeds {bound}"
                );
                (name, coefficients)
            })
            .collect::<BTreeMap<_, _>>();
        for (slot, (lift, rounding_numerator)) in bounds.paired.iter().enumerate() {
            let paired = &centered[&format!("paired-error-{slot}")];
            let pre_switch = &centered[&format!("pre-switch-error-{slot}")];
            assert_eq!(paired.len(), pre_switch.len());
            for (paired, pre_switch) in paired.iter().zip(pre_switch) {
                let exact_rounding = paired * BigInt::from(lift.clone()) - pre_switch;
                assert!(
                    exact_rounding.magnitude() * 2u8 <= *rounding_numerator,
                    "slot {slot}: exact paired rounding exceeded (J-1)(n+2)/(2J)"
                );
            }
        }
    }

    pub(crate) fn switched_decoder_numeric_fixture() -> (
        mxx_ir_core::ValidatedGraph,
        Vec<mxx_primitives::poly::dcrt::params::DCRTPolyParams>,
        NumericRefreshBounds,
    ) {
        use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
        let parameter = |name, default| {
            std::env::var(name).ok().map(|value| value.parse::<u32>().unwrap()).unwrap_or(default)
        };
        let n = parameter("MXX_PRIMITIVE_TEST_RING_DIMENSION", 32);
        let depth = parameter("MXX_PRIMITIVE_TEST_CRT_DEPTH", 3) as usize;
        let bits = parameter("MXX_PRIMITIVE_TEST_CRT_BITS", 40) as usize;
        let base_bits = parameter("MXX_PRIMITIVE_TEST_BASE_BITS", 4);
        assert!(depth >= 3, "mixed-P round trip requires at least three towers");
        let parameters = DCRTPolyParams::new(n, depth, bits, base_bits, None, None);
        let q = parameters.modulus().as_ref().clone();
        let (primes, _, _) = parameters.to_crt();
        let ring = mxx_dsl::Ring::new(BigInt::from(q.clone()), n as usize);
        let mut setup = parameters_for_numeric_refresh(&parameters);
        setup.switch_moduli = (0..depth)
            .map(|slot| &q / BigUint::from(primes[if slot + 1 == depth { 0 } else { depth - 1 }]))
            .collect();
        setup.mask_base_p_digit_counts = (0..depth).map(|slot| 1 + slot % 2).collect();
        let columns = setup.layout.public_key_columns();
        let secret = Mat::concat(
            mxx_dsl::ConcatAxis::Columns,
            vec![ring.polynomial([(-1).into()]), ring.polynomial([1.into()])],
        );
        let gadget = ring.gadget(2, setup.layout.gadget_base.clone(), setup.layout.digit_count);
        let payload_secret = Mat::concat(
            mxx_dsl::ConcatAxis::Columns,
            vec![ring.polynomial([1.into()]), ring.polynomial([1.into()])],
        );
        let public_a = ring.uniform_residue((2, columns));
        let a_prime = ring.uniform_residue((2, columns));
        let message = ring.polynomial([1.into(), (-1).into()]);
        // Nonzero coefficient-bounded input error and post-readout helper
        // residual. This fixture tests refresh, not the full PRF evaluator.
        let input_error = Mat::concat(
            mxx_dsl::ConcatAxis::Columns,
            vec![ring.polynomial((0..n).map(|_| 1.into())); columns],
        );
        let state = BggEncodingWire {
            vector: secret.clone() * public_a.clone() -
                message.clone() * (payload_secret.clone() * gadget.clone()) +
                input_error.clone(),
            plaintext: None,
        };
        let fresh = Mat::concat(
            mxx_dsl::ConcatAxis::Columns,
            vec![ring.polynomial((0..n).map(|_| 1.into())); columns],
        );
        let mut scalings = Vec::new();
        let mut sums = Vec::new();
        let mut masks = Vec::new();
        let mut scaled_fresh = Vec::new();
        let mut mask_residuals = Vec::new();
        let mut pre_switch_errors = Vec::new();
        let mut bounds = NumericRefreshBounds::default();
        let inverse_two = (&q + BigUint::one()) / 2u8;
        for slot in 0..depth {
            let kappa = ring.polynomial([setup.refresh.scale_expression(slot).unwrap()]);
            let scaling = (kappa.clone() * gadget.clone())
                .decompose(setup.layout.gadget_base.clone(), setup.layout.digit_count);
            sums.push(scaling.clone().mul_small_rhs(public_a.clone()));
            pre_switch_errors
                .push(scaling.clone().mul_small_rhs(input_error.clone()) + input_error.clone());
            scalings.push(scaling);
            scaled_fresh.push(BggEncodingWire { vector: kappa * fresh.clone(), plaintext: None });
            let digits = setup.mask_base_p_digit_counts[slot];
            let support = 4usize.pow(digits as u32);
            // Enumerate native digit combinations, including endpoints and interior
            // values. The production public normalization must yield v-M/2.
            let raw = Mat::concat(
                mxx_dsl::ConcatAxis::Columns,
                (0..columns)
                    .map(|column| {
                        ring.polynomial((0..n as usize).map(|coefficient| {
                            let mut value = (column * n as usize + coefficient) % support;
                            let mut weight = BigInt::one();
                            let mut encoded = BigInt::zero();
                            for _ in 0..digits {
                                let native_digit = BigInt::from(value % 4) - 1;
                                encoded +=
                                    &weight * (native_digit - BigInt::from(inverse_two.clone()));
                                weight *= 4;
                                value /= 4;
                            }
                            IntExpr::constant(encoded)
                        }))
                    })
                    .collect(),
            );
            let offset =
                crate::bgg::public_half_offset_vector(&ring, 4, digits, columns, true).unwrap();
            let lift =
                ring.polynomial([IntExpr::constant(BigInt::from(&q / &setup.switch_moduli[slot]))]);
            let normalized = raw + offset;
            let expected_mask = Mat::concat(
                mxx_dsl::ConcatAxis::Columns,
                (0..columns)
                    .map(|column| {
                        ring.polynomial((0..n as usize).map(|coefficient| {
                            IntExpr::constant(
                                BigInt::from((column * n as usize + coefficient) % support) -
                                    BigInt::from(support / 2),
                            )
                        }))
                    })
                    .collect(),
            );
            mask_residuals.push(normalized.clone() - expected_mask);
            masks.push(BggEncodingWire {
                vector: lift * normalized + input_error.clone(),
                plaintext: None,
            });
        }
        let cutoff = setup.resolve_decoder_preimage_bound().unwrap();
        let decoder_cutoff = cutoff.evaluate(&ParamEnv::default()).unwrap().to_biguint().unwrap();
        let anchor_cutoff = setup
            .encoding_error_bound
            .evaluate(&ParamEnv::default())
            .unwrap()
            .to_biguint()
            .unwrap();
        // Generic balanced-decomposition bound for the actual D(kappa G)
        // action, plus one post-readout residual. Constants make this safely
        // conservative even though the generic convolution factor is retained.
        let pre_switch_bound =
            BigUint::from(n) * columns * BigUint::from(1u64 << (base_bits - 1)) + 1u8;
        for slot in 0..depth {
            let lift = &q / &setup.switch_moduli[slot];
            let denominator = &lift * 2u8;
            let rounding_numerator = (&lift - BigUint::one()) * (n + 2);
            let decoder_error = &decoder_cutoff *
                &anchor_cutoff *
                n *
                (2 * (setup.decoder_digit_count(slot).unwrap() + 2));
            let mask_bound =
                BigUint::from(4usize.pow(setup.mask_base_p_digit_counts[slot] as u32) / 2);
            let spacing = &setup.switch_moduli[slot] / BigUint::from(primes[slot]);
            let final_bound = crate::noise_refresh::simulation::switched_error_bound(
                &pre_switch_bound,
                &lift,
                n as usize,
                &decoder_error,
            )
            .unwrap();
            let paired_bound =
                (&pre_switch_bound * 2u8 + &rounding_numerator + &denominator - 1u8) / &denominator;
            assert!(
                (mask_bound + &final_bound) * 2u8 < spacing,
                "test parameters must satisfy the strict switched mask-plus-residual margin"
            );
            bounds.bounds.insert(format!("pre-switch-error-{slot}"), pre_switch_bound.clone());
            bounds.bounds.insert(format!("paired-error-{slot}"), paired_bound);
            bounds.bounds.insert(format!("decoder-error-{slot}"), decoder_error);
            bounds.bounds.insert(format!("final-error-{slot}"), final_bound);
            bounds.bounds.insert(format!("anchor-error-{slot}"), anchor_cutoff.clone());
            bounds.paired.push((lift, rounding_numerator));
        }
        let compiler = ExponentLutEncodingCompiler::with_layout(
            mxx_bgg::BggPublicKeyCompiler {
                ring: ring.clone(),
                base: setup.layout.gadget_base.clone(),
                digit_count: setup.layout.digit_count.into(),
            },
            crate::layout::ExpLayout { stride_log2: 1, width_log2: n.ilog2() - 1 },
        );
        let (public_b, targets, preimages) = setup
            .sample_refresh_decoders(
                &crate::bgg::BggScheme::new(
                    mxx_bgg::BggPublicKeyCompiler {
                        ring: compiler.ring.clone(),
                        base: compiler.gadget_base.clone(),
                        digit_count: compiler.digit_count.clone(),
                    },
                    compiler.exponent_layout,
                ),
                Family::pack(sums).unwrap(),
                &a_prime,
                &cutoff,
            )
            .unwrap();
        let (slot_public, bases) = setup.shared_decoder_bases(secret.clone(), &public_b).unwrap();
        let anchor_errors = (0..depth)
            .map(|slot| {
                bases[slot].clone() -
                    secret.clone().reduce_modulus(BigInt::from(setup.switch_moduli[slot].clone())) *
                        slot_public[slot].clone()
            })
            .collect::<Vec<_>>();
        let diagnostic_state = state.vector.clone();
        let diagnostic_masks = masks.clone();
        let diagnostic_fresh = scaled_fresh.clone();
        let diagnostic_scalings = scalings.clone();
        let diagnostic_bases = bases.clone();
        let bound = setup
            .refresh
            .bind_imported_wires(
                &compiler,
                state,
                BggPublicKeyWire { matrix: public_a, reveal_plaintext: false },
                a_prime.clone(),
                slot_public.clone(),
                targets.clone(),
                scaled_fresh.into_iter().map(|wire| wire.vector).collect(),
                masks.into_iter().map(|wire| wire.vector).collect(),
                scalings,
                bases,
                preimages.clone(),
            )
            .unwrap();
        let refreshed = setup.refresh.refresh(&compiler, &bound).unwrap().into_encoding().vector;
        let expected = secret * a_prime - message * (payload_secret * gadget) + fresh;
        let mut context = DslContext::new("numeric-switched-decoder")
            .output("refresh-residual", refreshed - expected.clone())
            .unwrap();
        for slot in 0..depth {
            let modulus = IntExpr::constant(BigInt::from(setup.switch_moduli[slot].clone()));
            let low = setup.switch_ring(slot).unwrap();
            let delta = low.polynomial([IntExpr::constant(BigInt::from(
                &setup.switch_moduli[slot] / primes[slot],
            ))]);
            let support = 4usize.pow(setup.mask_base_p_digit_counts[slot] as u32);
            let nominal_mask = Mat::concat(
                mxx_dsl::ConcatAxis::Columns,
                (0..columns)
                    .map(|column| {
                        low.polynomial((0..n as usize).map(|coefficient| {
                            IntExpr::constant(
                                BigInt::from((column * n as usize + coefficient) % support) -
                                    BigInt::from(support / 2),
                            )
                        }))
                    })
                    .collect(),
            );
            let combined =
                diagnostic_scalings[slot].clone().mul_small_rhs(diagnostic_state.clone()) +
                    diagnostic_masks[slot].vector.clone() +
                    diagnostic_fresh[slot].vector.clone();
            let decoder_error = preimages[slot].clone().mul_small_rhs(anchor_errors[slot].clone());
            let final_error = combined.modulus_switch(modulus.clone()) -
                preimages[slot].clone().mul_small_rhs(diagnostic_bases[slot].clone()) -
                delta * expected.clone().reduce_modulus(modulus) -
                nominal_mask;
            context = context
                .output(format!("pre-switch-error-{slot}"), pre_switch_errors[slot].clone())
                .unwrap()
                .output(format!("paired-error-{slot}"), final_error.clone() + decoder_error.clone())
                .unwrap()
                .output(format!("decoder-error-{slot}"), decoder_error)
                .unwrap()
                .output(format!("final-error-{slot}"), final_error)
                .unwrap()
                .output(format!("anchor-error-{slot}"), anchor_errors[slot].clone())
                .unwrap();
            context = context
                .output(format!("mask-normalization-residual-{slot}"), mask_residuals[slot].clone())
                .unwrap();
            context = context
                .output(
                    format!("preimage-residual-{slot}"),
                    preimages[slot].clone().mul_small_rhs(slot_public[slot].clone()) -
                        targets[slot].clone(),
                )
                .unwrap();
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let mut concrete = vec![parameters.clone()];
        for p in setup.switch_moduli.iter().collect::<std::collections::BTreeSet<_>>() {
            concrete.push(parameters.select_modulus(p).unwrap());
        }
        (graph, concrete, bounds)
    }

    #[test]
    #[cfg_attr(feature = "gpu", serial_test::serial(dcrt_runtime))]
    fn switched_decoder_numeric_round_trip_with_shared_fresh_error() {
        use mxx_primitives::{
            matrix::PolyMatrix,
            poly::{Poly, PolyParams},
        };
        use mxx_runtime::{
            MemoryArtifactStore, RuntimeValue, backend::poly::cpu_backend, transcript::SamplingMode,
        };
        let (graph, concrete, bounds) = switched_decoder_numeric_fixture();
        let mut backend = cpu_backend(concrete);
        // Two fresh execution draws of the same validated graph, with no
        // replay/cached anchor. Structural tests separately bind draw sites.
        for _execution in 0..2 {
            let result = mxx_runtime::execute(
                &graph,
                &mut backend,
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            let mut observations = BTreeMap::new();
            for (name, output) in result.outputs {
                let RuntimeValue::Matrix(matrix) = output else {
                    panic!("matrix output expected");
                };
                let mut coefficients = Vec::new();
                for row in 0..matrix.row_size() {
                    for column in 0..matrix.col_size() {
                        coefficients.extend(matrix.entry(row, column).coeffs_biguints());
                    }
                }
                observations
                    .insert(name, (matrix.params().modulus().as_ref().clone(), coefficients));
            }
            assert_numeric_refresh_outputs(observations, &bounds);
        }
    }

    pub(crate) fn parameters_for_numeric_refresh(
        parameters: &mxx_primitives::poly::dcrt::params::DCRTPolyParams,
    ) -> BggRefreshParameters {
        use mxx_primitives::poly::PolyParams;
        let (primes, _, _) = parameters.to_crt();
        let q = BigInt::from(parameters.modulus().as_ref().clone());
        BggRefreshParameters::new(
            rand::random(),
            4,
            2,
            parameters.ring_dimension() as usize,
            1,
            1,
            32,
            4,
            BggSamplerLayout {
                modulus: q.clone().into(),
                ring_dimension: (parameters.ring_dimension() as usize).into(),
                secret_dimension: 2,
                digit_count: parameters.modulus_digits(),
                gadget_base: BigInt::from(1u64 << parameters.base_bits()).into(),
            },
            RefreshCompiler {
                full_modulus: q.into(),
                crt_plaintext_moduli: primes.into_iter().map(IntExpr::constant).collect(),
                reconstruction_coefficients: parameters
                    .reconst_coeffs()
                    .into_iter()
                    .map(|value| IntExpr::constant(BigInt::from(value)))
                    .collect(),
            },
            1.into(),
            1.into(),
            "numeric-switched-decoder",
        )
    }

    #[test]
    fn switched_preprocessing_artifacts_use_only_selected_decoder_rings() {
        use mxx_primitives::poly::PolyParams;
        let (_, rings, _) = switched_decoder_numeric_fixture();
        let source = &rings[0];
        let mut setup = parameters_for_numeric_refresh(source);
        let primes = source.to_crt().0;
        let q = source.modulus().as_ref().clone();
        setup.switch_moduli = (0..primes.len())
            .map(|slot| {
                &q / BigUint::from(
                    primes[if slot + 1 == primes.len() { 0 } else { primes.len() - 1 }],
                )
            })
            .collect();
        let profile = SparseLwrPrfProfile::new(
            crate::layout::ExpLayout {
                stride_log2: 1,
                width_log2: source.ring_dimension().ilog2() - 1,
            },
            2,
        )
        .unwrap();
        setup.lut_width = profile.layout.width();
        let expected = setup.layout.ring().polynomial([1.into()]);
        let generated = generate_key_layout(
            &PbcParameters::paper_evaluation(10, 4),
            PbcRootSeed(rand::random()),
            &[0, 1, 2, 3],
        )
        .unwrap();
        let bundle = RefreshParameterSimulationRequest::new(
            setup.clone(),
            crate::noise_refresh::prf::SparseLwrPrfProgram::new(
                profile,
                generated.public_layout.bucket_width,
                generated.public_layout.parameters.bucket_count,
            )
            .expect("PRF Program"),
            generated,
            rand::random(),
            expected,
            1,
        )
        .unwrap()
        .build_benchmark()
        .unwrap();
        assert_eq!(bundle.ring_parameters(source.clone()).unwrap().len(), 3);
        let public_label_input = "refresh-parameter-search-native-monomials";
        assert!(bundle.benchmark_preprocessing_graph().graph.root_scope().nodes().iter().any(
            |node| {
                matches!(node.kind(), NodeKind::Input { name, artifact: None, .. }
                if name == public_label_input)
            }
        ));
        assert!(!bundle.benchmark_online_graph().graph.root_scope().nodes().iter().any(|node| {
            matches!(node.kind(), NodeKind::Input { name, artifact: None, .. }
                if name == public_label_input)
        }));
        let nodes = bundle
            .benchmark_preprocessing_graph()
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .collect::<Vec<_>>();
        let trapdoors = nodes
            .iter()
            .filter_map(|node| match node.kind() {
                NodeKind::TrapdoorSample { matrix_type, .. } => {
                    Some(matrix_type.modulus.evaluate(&ParamEnv::default()).unwrap())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(trapdoors.len(), 2);
        assert!(trapdoors.iter().all(|p| *p != BigInt::from(q.clone())));
        assert_eq!(
            trapdoors.into_iter().collect::<std::collections::BTreeSet<_>>(),
            setup.switch_moduli.iter().cloned().map(BigInt::from).collect()
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ModulusSwitch { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn online_anchor_groups_reduce_one_secret_and_sample_once_per_ring() {
        let mut setup = parameters(4, 2);
        setup.layout.modulus = 105.into();
        setup.refresh.full_modulus = 105.into();
        setup.refresh.crt_plaintext_moduli = vec![7.into(), 3.into(), 5.into()];
        setup.refresh.reconstruction_coefficients = vec![1.into(); 3];
        setup.switch_moduli = vec![21u8.into(), 15u8.into(), 15u8.into()];
        setup.mask_base_p_digit_counts = vec![1, 3, 2];
        let security = setup.anchor_security_inputs(3).unwrap();
        assert_eq!(security.len(), 2);
        assert_eq!(security[0].modulus, 15u8.into());
        assert_eq!(security[0].complementary_modulus, 7u8.into());
        assert_eq!(security[0].slots, vec![1, 2]);
        assert_eq!(security[1].modulus, 21u8.into());
        assert_eq!(security[1].complementary_modulus, 5u8.into());
        for group in security {
            assert_eq!(group.ring_dimension, 4);
            assert_eq!(group.random_secret_polynomials, 1);
            assert_eq!(group.fixed_augmented_coordinate, 1);
            assert_eq!(group.ring_samples, 3 * group.anchor_columns);
            assert_eq!(group.error_sigma, setup.encoding_error_sigma);
            assert_eq!(group.error_coefficient_cutoff, setup.encoding_error_bound);
            assert_eq!(group.estimated_security_bits, None);
            assert!(group.status.contains("noise acceptance is not security acceptance"));
        }
        assert!(setup.anchor_security_inputs(0).is_err());
        let ring = setup.layout.ring();
        let secret = ring.input("shared-small-secret", (1, 2));
        let public = group_refresh_levels(setup.switch_moduli.iter().cloned())
            .iter()
            .map(|(p, _)| {
                mxx_dsl::Ring::new(IntExpr::constant(BigInt::from(p.clone())), 4)
                    .input(format!("B-{p}"), (2, 8))
            })
            .collect::<Vec<_>>();
        let (anchors, bases) = setup.shared_decoder_bases(secret.clone(), &public).unwrap();
        assert_eq!(anchors[1].value_handle(), anchors[2].value_handle());
        assert_ne!(anchors[0].value_handle(), anchors[1].value_handle());
        assert_eq!(bases[1].value_handle(), bases[2].value_handle());
        assert_ne!(bases[0].value_handle(), bases[1].value_handle());
        let (_, second) = setup.shared_decoder_bases(secret, &public).unwrap();
        assert_ne!(bases[0].value_handle(), second[0].value_handle());
        let mut context = DslContext::new("grouped-online-anchors");
        for (slot, base) in bases.into_iter().enumerate() {
            let low = setup.switch_ring(slot).unwrap();
            context = context
                .output(
                    format!("decoder-result-{slot}"),
                    base * low.input(format!("consumer-{slot}"), (8, 1)),
                )
                .unwrap();
        }
        let graph = context.build().unwrap();
        graph.validate(&Default::default()).unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GaussianSample { .. }))
                .count(),
            2
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ModulusReduce { .. }))
                .count(),
            2
        );
        assert!(!nodes.iter().any(|node| matches!(
            node.kind(),
            NodeKind::ModulusSwitch { .. } | NodeKind::UniformIntervalSample { .. }
        )));
        assert!(graph.graph.outputs().keys().all(|name| name.starts_with("decoder-result-")));
    }
}
