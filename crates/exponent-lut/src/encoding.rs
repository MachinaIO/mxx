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

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    ExponentLutError,
    program::{
        ExponentLutProgram, ExponentProgramLowering, FamilyRange, LutTable, ProgramBindings,
        ProgramFamilyBindings, ProgramGateInstance, ProgramInputId, ProgramLoweringTypes,
        ProgramValidationError, RhsId, WireId, lower_program,
    },
    public_key::{ProgramCacheBindings, ProgramLutCache},
    rhs::{ExponentRhsPackage, ExponentRhsPackageError},
};
use mxx_bgg::{BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{Bytes, DslError, Family, HashTag, Int, Mat, Parallel, ParallelZipTuple, Preimage};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    node::{ConcatAxis, IndexRange},
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use thiserror::Error;

/// Validated online BGG carrier/plaintext family consumed by family lowering.
///
/// This is encoding-domain runtime data, not part of the host-side program
/// representation. Public-key matrices never cross this boundary because
/// multiplication consumes preprocessing-produced decomposition artifacts.
#[derive(Clone)]
pub(crate) struct BggEncodingFamily {
    vector: Family<Mat>,
    plaintext: Family<Mat>,
}

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
    #[cfg(test)]
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
#[derive(Clone)]
pub struct LutHelperSet {
    table_commitment: [u8; 32],
    width: usize,
    helpers: Vec<LutHelper>,
}

impl LutHelperSet {
    pub fn new(
        table: &crate::program::LutTable,
        helpers: Vec<LutHelper>,
    ) -> Result<Self, ExponentLutError> {
        if table.values().len() != helpers.len() {
            return Err(ExponentLutError::InvalidLut);
        }
        Ok(Self { table_commitment: table.commitment(), width: helpers.len(), helpers })
    }

    pub(crate) fn resolve(
        &self,
        table: &crate::program::LutTable,
    ) -> Result<&[LutHelper], ExponentLutError> {
        if self.width != table.values().len() || self.table_commitment != table.commitment() {
            return Err(ExponentLutError::InvalidLut);
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
    ) -> Result<Self, ExponentLutError> {
        if sigma == 0 || sigma % 2 == 0 {
            return Err(ExponentLutError::InvalidLut);
        }
        Self::with_mask_bank(sigma, switch, Arc::new(LutMaskBank::single(sigma, mask)))
    }
    pub(crate) fn with_mask_bank(
        sigma: usize,
        switch: ExponentRhsPackage,
        mask_bank: Arc<LutMaskBank>,
    ) -> Result<Self, ExponentLutError> {
        let mask_index = mask_bank.index_for_sigma(sigma).ok_or(ExponentLutError::InvalidLut)?;
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

/// Structural setup/online material for the non-identity native readout
/// branches. All three families share canonical sigma order and are produced
/// by one parallel body rather than one Rust graph-construction iteration per
/// branch.
#[derive(Clone)]
pub(crate) struct NativeReadoutHelperFamilies {
    pub(crate) sigmas: Arc<Vec<usize>>,
    pub(crate) switch_ciphertexts: Family<Mat>,
    pub(crate) public_masks: Family<Mat>,
    pub(crate) online_mask_vectors: Family<Mat>,
}

/// Errors raised while constructing setup-time Exponent-LUT inputs and helpers.
#[derive(Debug, Error)]
pub enum ExponentLutSamplingError {
    #[error(transparent)]
    Bgg(#[from] mxx_bgg::BggSampleError),
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error(transparent)]
    ExponentLut(#[from] ExponentLutError),
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
    pub(crate) fn sample_native_readout_helper_families(
        &self,
        setup_secret: Mat,
        online_secret: Mat,
        hash_key: Bytes,
        table: &LutTable,
        mask_tag: impl Into<HashTag>,
        helper_tag: impl Into<HashTag>,
    ) -> Result<NativeReadoutHelperFamilies, ExponentLutSamplingError> {
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
        let lut_layout = derive_lut_layout(&self.exponent_layout, width, n)?;
        if table.input_width() != width || width <= 1 {
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
        let (switch_ciphertexts, public_masks, online_mask_vectors) = Parallel::range(helper_count)
            .map_values(move |index| {
                let sigma = lut_layout
                    .automorphism_index_expression(
                        n,
                        IntExpr::Add(Box::new(index.expression()), Box::new(IntExpr::constant(1))),
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
                let mask_sigma = online_secret.clone().ring_automorphism(sigma.clone());
                let constant = BggPublicKeyWire {
                    matrix: sampler.layout.ring().zero((
                        sampler.layout.secret_dimension,
                        sampler.layout.public_key_columns(),
                    )),
                    reveal_plaintext: false,
                };
                let online_mask = mxx_bgg::BggEncodingSampler {
                    layout: sampler.layout.clone(),
                    gaussian_sigma: sampler.gaussian_sigma.clone(),
                    gaussian_max_coefficient_bound: sampler.gaussian_max_coefficient_bound.clone(),
                }
                .sample(
                    online_secret.clone(),
                    Some(mask_sigma),
                    &[constant, mask_key.clone()],
                    &[sampler.layout.ring().identity(1)],
                )
                .expect("validated structural online mask sample")
                .into_iter()
                .nth(1)
                .expect("validated structural online mask");
                // ValueReadout already multiplies the input by T'. Its trace
                // must sum unweighted automorphism branches, so this helper
                // changes the payload secret from sigma(t) to t with value 1.
                // Ordinary LUT helpers separately retain their LUT coefficients.
                let mut switch_tag = helper_root.clone();
                switch_tag.push("switch");
                switch_tag.push(IntExpr::Add(
                    Box::new(index.expression()),
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
                (switch.gsw_ciphertext().clone(), mask_key.matrix, online_mask.vector)
            })
            .map_err(ExponentLutSamplingError::Dsl)?;
        Ok(NativeReadoutHelperFamilies {
            sigmas: helper_sigmas,
            switch_ciphertexts,
            public_masks,
            online_mask_vectors,
        })
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
    pub fn sample_helpers_for_lut(
        &self,
        mask_secret: Mat,
        payload_secret: Option<Mat>,
        hash_key: Bytes,
        table: &LutTable,
        mask_bank: &LutMaskBank,
        tag: impl Into<HashTag>,
    ) -> Result<Vec<LutHelper>, ExponentLutSamplingError> {
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
            table.input_width() != width ||
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
                lut_coefficient(&self.layout.ring(), &lut_layout, sigma, table.values())?;
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

    #[cfg(test)]
    pub(crate) fn compile_program(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggEncodingWire>,
        rhs: &BTreeMap<RhsId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, EncodingRhsFamily>,
        values: &BTreeMap<crate::program::WireFamilyId, BggEncodingFamily>,
        helpers: &LutHelperMap,
    ) -> Result<BTreeMap<WireId, BggEncodingWire>, ExponentLutError> {
        let mut ranges = ProgramFamilyBindings::new();
        for (index, family) in selectors.values().enumerate() {
            ranges.bind(
                crate::program::ProgramFamilyId::from_index(index),
                FamilyRange::full(family.count().clone())
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?,
            );
        }
        self.compile_program_wrapped(
            program, inputs, rhs, selectors, values, &ranges, helpers, None,
        )
    }

    pub(crate) fn compile_program_with_ranges(
        &self,
        program: &ExponentLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggEncodingWire>,
        rhs: &BTreeMap<RhsId, ExponentRhsPackage>,
        selectors: &BTreeMap<crate::program::RhsFamilyId, EncodingRhsFamily>,
        values: &BTreeMap<crate::program::WireFamilyId, BggEncodingFamily>,
        ranges: &ProgramFamilyBindings,
        helpers: &LutHelperMap,
    ) -> Result<BTreeMap<WireId, BggEncodingWire>, ExponentLutError> {
        self.compile_program_wrapped(program, inputs, rhs, selectors, values, ranges, helpers, None)
    }

    /// Compiles a program using only the supplied typed public preimage
    /// bindings for multiplication. A non-empty production cache is therefore
    /// consumed directly by the graph; no online decomposition is emitted.
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
    ) -> Result<BTreeMap<WireId, BggEncodingWire>, ExponentLutError> {
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
    ) -> Result<BTreeMap<WireId, BggEncodingWire>, ExponentLutError> {
        let wrapped_inputs = inputs
            .iter()
            .map(|(id, value)| (*id, EncodingProgramWire::Scalar(value.clone())))
            .collect();
        let wrapped_rhs =
            rhs.iter().map(|(id, _value)| (*id, EncodingProgramRhs::Scalar)).collect();
        let wrapped_selectors = selectors
            .iter()
            .map(|(id, value)| (*id, EncodingProgramRhs::Members(value.clone())))
            .collect();
        let wrapped_values = values
            .iter()
            .map(|(id, value)| (*id, EncodingProgramWire::Members(value.clone())))
            .collect();
        let bindings = ProgramBindings::new(
            &wrapped_inputs,
            &wrapped_rhs,
            &wrapped_selectors,
            &wrapped_values,
            helpers,
        );
        let mut backend = EncodingProgramLowering { compiler: self, cache, ranges };
        lower_program(program, &bindings, ranges, &mut backend)?
            .into_iter()
            .map(|(id, wire)| match wire {
                EncodingProgramWire::Scalar(value) => Ok((id, value)),
                _ => Err(ExponentLutError::InvalidSparseLwrBlock),
            })
            .collect()
    }

    #[cfg(all(test, feature = "gpu"))]
    fn test_single_input_lut(
        &self,
        input: &BggEncodingWire,
        values: &[usize],
        helpers: &[LutHelper],
        cache: &ProgramLutCache,
    ) -> Result<BggEncodingWire, ExponentLutError> {
        let ring_dimension = input
            .vector
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let table = LutTable::unary(
            values.len(),
            self.exponent_layout.effective_modulus(ring_dimension),
            values.to_vec(),
        )?;
        self.single_input_lut_table_with_cache(input, &table, helpers, Some(cache))
    }

    #[cfg(all(test, feature = "gpu"))]
    fn test_two_input_lut(
        &self,
        lhs: &BggEncodingWire,
        rhs: &ExponentRhsPackage,
        table: &LutTable,
        helpers: &[LutHelper],
        cache: &ProgramLutCache,
    ) -> Result<BggEncodingWire, ExponentLutError> {
        let fused = BggEncodingWire {
            vector: rhs
                .gsw_ciphertext()
                .clone()
                .decompose(self.gadget_base.clone(), self.digit_count.clone())
                .mul_small_rhs(lhs.vector.clone()),
            plaintext: None,
        };
        self.single_input_lut_table_with_cache(&fused, table, helpers, Some(cache))
    }

    fn single_input_lut_table_with_cache(
        &self,
        input: &BggEncodingWire,
        table: &LutTable,
        helpers: &[LutHelper],
        cached: Option<&ProgramLutCache>,
    ) -> Result<BggEncodingWire, ExponentLutError> {
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
    ) -> Result<BggEncodingWire, ExponentLutError> {
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
    ) -> Result<LutOutputParts, ExponentLutError> {
        let width = table.values().len();
        let n = input_vector
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let lut_layout = derive_lut_layout(&self.exponent_layout, width, n)?;
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
        let coefficients = sigmas
            .iter()
            .map(|sigma| {
                lut_coefficient(&self.ring, &lut_layout, *sigma, table.values())
                    .map_err(|_| ExponentLutError::InvalidLut)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // `raw_vectors[j]` and `raw_publics[j]` are the two input components
        // after `X -> X^{\sigma_j}`, before applying branch `C_j` and its mask.
        let raw_vectors = Family::pack(
            helpers
                .iter()
                .enumerate()
                .map(|(j, helper)| {
                    let expected = sigmas.get(j).copied().ok_or(ExponentLutError::InvalidLut)?;
                    if helper.sigma() != expected {
                        return Err(ExponentLutError::InvalidLut);
                    }
                    Ok(input_vector.clone().ring_automorphism(expected))
                })
                .collect::<Result<Vec<_>, ExponentLutError>>()?,
        )
        .map_err(|_| ExponentLutError::InvalidLut)?;
        let cache = cached.ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
        let family_width = |family: &Family<Preimage>| {
            family.count().evaluate(&ParamEnv::default()).ok().and_then(|value| value.to_usize())
        };
        if family_width(&cache.switch) != Some(width) || family_width(&cache.branch) != Some(width)
        {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        let mask_vectors =
            Family::pack(helpers.iter().map(|helper| helper.mask().vector.clone()).collect())
                .map_err(|_| ExponentLutError::InvalidLut)?;
        let vectors = (raw_vectors, cache.switch.clone(), mask_vectors, cache.branch.clone())
            .parallel_zip_tuple(|_, (raw, switch, mask, branch)| {
                branch.mul_small_rhs(mask) + switch.mul_small_rhs(raw)
            })
            .map_err(|_| ExponentLutError::InvalidLut)?;
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

    fn family_lut_with_cache(
        &self,
        input: &EncodingFamilyIntermediate,
        table: &LutTable,
        helpers: &[LutHelper],
        cache: &ProgramLutCache,
    ) -> Result<EncodingFamilyIntermediate, ExponentLutError> {
        let member_count = concrete_mat_family_count(&input.vectors)?;
        let width = table.values().len();
        let flat_count = member_count.checked_mul(width).ok_or(ExponentLutError::InvalidLut)?;
        if concrete_preimage_family_count(&cache.switch)? != flat_count ||
            concrete_preimage_family_count(&cache.branch)? != flat_count ||
            helpers.len() != width
        {
            return Err(ExponentLutError::InvalidLut);
        }
        let n = input
            .vectors
            .element_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentLutError::InvalidLut)?;
        let layout = derive_lut_layout(&self.exponent_layout, width, n)?;
        let sigmas = layout.automorphism_indices(n).map_err(|_| ExponentLutError::InvalidLut)?;
        if sigmas.len() != width ||
            helpers.iter().zip(&sigmas).any(|(helper, sigma)| helper.sigma() != *sigma)
        {
            return Err(ExponentLutError::InvalidLut);
        }
        let member_indices = Parallel::range(flat_count)
            .map_values(|index| index.as_int().div(Int::constant(width)))
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let branch_indices = Parallel::range(flat_count)
            .map_values(|index| index.as_int().rem(Int::constant(width)))
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let flat_vectors = input
            .vectors
            .clone()
            .parallel_gather(member_indices.clone())
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let flat_masks =
            Family::pack(helpers.iter().map(|helper| helper.mask().vector.clone()).collect())
                .map_err(|_| ExponentLutError::InvalidLut)?
                .parallel_gather(branch_indices.clone())
                .map_err(|_| ExponentLutError::InvalidLut)?;
        let sigma_values = sigmas.iter().copied().map(IntExpr::constant).collect::<Vec<_>>();
        let switched = (flat_vectors, cache.switch.clone())
            .parallel_zip_tuple(move |index, (raw, switch)| {
                let flat = index.expression();
                let divisor = IntExpr::constant(width);
                let branch_index = IntExpr::Sub(
                    Box::new(flat.clone()),
                    Box::new(IntExpr::Mul(
                        Box::new(IntExpr::Div(Box::new(flat), Box::new(divisor.clone()))),
                        Box::new(divisor),
                    )),
                );
                let sigma = IntExpr::Select {
                    selector: Box::new(branch_index),
                    branches: sigma_values.clone(),
                };
                switch.mul_small_rhs(raw.ring_automorphism(sigma))
            })
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let aligned = (flat_masks, cache.branch.clone())
            .parallel_zip_tuple(|_, (mask, branch)| branch.mul_small_rhs(mask))
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let branches = (switched, aligned)
            .parallel_zip_tuple(|_, (switched, aligned)| switched + aligned)
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let vectors = reduce_encoding_family_segments(branches, member_count, width)?;
        let plaintexts = if let Some(plaintexts) = &input.plaintexts {
            let coefficients = sigmas
                .iter()
                .map(|sigma| lut_coefficient(&self.ring, &layout, *sigma, table.values()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ExponentLutError::InvalidLut)?;
            let flat_plaintexts = plaintexts
                .clone()
                .parallel_gather(member_indices)
                .map_err(|_| ExponentLutError::InvalidLut)?;
            let flat_coefficients = Family::pack(coefficients)
                .map_err(|_| ExponentLutError::InvalidLut)?
                .parallel_gather(branch_indices)
                .map_err(|_| ExponentLutError::InvalidLut)?;
            let sigma_values = sigmas.iter().copied().map(IntExpr::constant).collect::<Vec<_>>();
            let branches = (flat_plaintexts, flat_coefficients)
                .parallel_zip_tuple(move |index, (value, coefficient)| {
                    let flat = index.expression();
                    let divisor = IntExpr::constant(width);
                    let branch_index = IntExpr::Sub(
                        Box::new(flat.clone()),
                        Box::new(IntExpr::Mul(
                            Box::new(IntExpr::Div(Box::new(flat), Box::new(divisor.clone()))),
                            Box::new(divisor),
                        )),
                    );
                    let sigma = IntExpr::Select {
                        selector: Box::new(branch_index),
                        branches: sigma_values.clone(),
                    };
                    value.ring_automorphism(sigma) * coefficient
                })
                .map_err(|_| ExponentLutError::InvalidLut)?;
            Some(reduce_encoding_family_segments(branches, member_count, width)?)
        } else {
            None
        };
        Ok(EncodingFamilyIntermediate { vectors, plaintexts })
    }
}

#[derive(Clone)]
pub(crate) enum EncodingProgramWire {
    Scalar(BggEncodingWire),
    /// Structural scalar view shared at every active logical family index.
    Broadcast {
        value: BggEncodingWire,
        range: FamilyRange,
    },
    /// Index-dependent external family. The active slice is selected by the
    /// operation's structural range, without cloning up to capacity.
    Members(BggEncodingFamily),
    Materialized {
        value: EncodingFamilyIntermediate,
        range: FamilyRange,
    },
}

#[derive(Clone)]
pub(crate) enum EncodingProgramRhs {
    Scalar,
    /// Index-dependent external RHS family at the current logical body index.
    Members(EncodingRhsFamily),
}

struct EncodingProgramLowering<'a> {
    compiler: &'a ExponentLutEncodingCompiler,
    cache: Option<&'a ProgramCacheBindings>,
    ranges: &'a ProgramFamilyBindings,
}

impl ProgramLoweringTypes for EncodingProgramLowering<'_> {
    type Wire = EncodingProgramWire;
    type Rhs = EncodingProgramRhs;
    type HelperSet = LutHelperSet;
    type Error = ExponentLutError;
}

impl ExponentProgramLowering for EncodingProgramLowering<'_> {
    fn mul_private(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Rhs,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        match (lhs, rhs, gate) {
            (
                EncodingProgramWire::Scalar(lhs),
                EncodingProgramRhs::Scalar,
                ProgramGateInstance::Gate { step },
            ) => {
                let decomposition = self
                    .cache
                    .and_then(|cache| cache.gate_entry(step))
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
                Ok(EncodingProgramWire::Scalar(BggEncodingWire {
                    vector: decomposition.clone().mul_small_rhs(lhs.vector.clone()),
                    plaintext: None,
                }))
            }
            (
                EncodingProgramWire::Broadcast { value: input, range },
                EncodingProgramRhs::Members(selectors),
                ProgramGateInstance::FamilyGate { step, gate },
            ) => {
                let decompositions = self
                    .cache
                    .and_then(|cache| cache.family_entry(step, gate))
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
                if decompositions.count() != range.count() || selectors.count() < range.count() {
                    return Err(ExponentLutError::InvalidSparseLwrBlock);
                }
                let vectors = decompositions
                    .clone()
                    .parallel_map_values(|_, decomposition| {
                        decomposition.mul_small_rhs(input.vector.clone())
                    })
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                Ok(EncodingProgramWire::Materialized {
                    value: EncodingFamilyIntermediate { vectors, plaintexts: None },
                    range: range.clone(),
                })
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
                EncodingProgramWire::Scalar(private),
                EncodingProgramWire::Scalar(public),
                ProgramGateInstance::Gate { step },
            ) => {
                let decomposition = self
                    .cache
                    .and_then(|cache| cache.gate_entry(step))
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
                let plaintext =
                    public.plaintext.clone().ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
                Ok(EncodingProgramWire::Scalar(BggEncodingWire {
                    vector: decomposition.clone().mul_small_rhs(public.vector.clone()) +
                        plaintext * private.vector.clone(),
                    plaintext: program_binary_plaintext(public, private, |left, right| {
                        left * right
                    }),
                }))
            }
            (
                EncodingProgramWire::Materialized { value: mapped, range },
                EncodingProgramWire::Members(values),
                ProgramGateInstance::FamilyGate { step, gate },
            ) => {
                let decompositions = self
                    .cache
                    .and_then(|cache| cache.family_entry(step, gate))
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
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
                if mapped.vectors.count() != range.count() ||
                    decompositions.count() != range.count()
                {
                    return Err(ExponentLutError::InvalidSparseLwrBlock);
                }
                let indices = Parallel::range(count)
                    .map_values(|index| index.as_int().add(mxx_dsl::Int::constant(start)))
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                let values_vector = values
                    .vector_family()
                    .clone()
                    .parallel_gather(indices.clone())
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                let values_plaintext = values
                    .plaintext_family()
                    .clone()
                    .parallel_gather(indices)
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                let first = (values_vector, decompositions.clone())
                    .parallel_zip_tuple_result(|_, (value, decomposition)| {
                        Ok(decomposition.mul_small_rhs(value))
                    })
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                let second = (values_plaintext, mapped.vectors.clone())
                    .parallel_zip_tuple_result(|_, (value, vector)| Ok(value * vector))
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                let vectors = (first, second)
                    .parallel_zip_tuple_result(|_, (left, right)| Ok(left + right))
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                Ok(EncodingProgramWire::Materialized {
                    value: EncodingFamilyIntermediate { vectors, plaintexts: None },
                    range: range.clone(),
                })
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
            (EncodingProgramWire::Scalar(lhs), EncodingProgramWire::Scalar(rhs)) => {
                self.compiler.bgg.add(lhs, rhs).map(EncodingProgramWire::Scalar).map_err(Into::into)
            }
            (
                EncodingProgramWire::Materialized { value: lhs, range },
                EncodingProgramWire::Materialized { value: rhs, range: rhs_range },
            ) if range == rhs_range => {
                let vectors = (lhs.vectors.clone(), rhs.vectors.clone())
                    .parallel_zip_tuple_result(|_, (lhs, rhs)| Ok(lhs + rhs))
                    .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
                Ok(EncodingProgramWire::Materialized {
                    value: EncodingFamilyIntermediate { vectors, plaintexts: None },
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
            (EncodingProgramWire::Scalar(input), ProgramGateInstance::Gate { step }) => {
                let cache = self
                    .cache
                    .and_then(|cache| cache.lut_entry(step))
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
                self.compiler
                    .single_input_lut_table_with_cache(input, table, helpers, Some(cache))
                    .map(EncodingProgramWire::Scalar)
            }
            (family_input, ProgramGateInstance::FamilyGate { step, gate }) => {
                let range = match program.steps().get(step) {
                    Some(crate::program::ProgramStep::Family { id, .. }) => {
                        self.ranges.range(*id).ok_or(ExponentLutError::InvalidSparseLwrBlock)?
                    }
                    _ => return Err(ExponentLutError::InvalidLut),
                };
                let cache = self
                    .cache
                    .and_then(|cache| cache.family_lut_entry(step, gate))
                    .ok_or(ExponentLutError::InvalidSparseLwrBlock)?;
                let value = active_encoding_family(family_input, range)?;
                let value = self.compiler.family_lut_with_cache(&value, table, helpers, cache)?;
                Ok(EncodingProgramWire::Materialized { value, range: range.clone() })
            }
            _ => Err(ExponentLutError::InvalidLut),
        }
    }

    fn backend_layout(&self) -> Option<crate::program::ProgramLayout> {
        let n = self
            .compiler
            .ring
            .matrix_type((1, 1))
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()?
            .to_usize()?;
        Some(crate::program::ProgramLayout {
            exponent: self.compiler.exponent_layout,
            ring_dimension_log2: n.trailing_zeros(),
        })
    }

    fn public_attribute_available(&self, wire: &Self::Wire) -> bool {
        matches!(wire, EncodingProgramWire::Scalar(value) if value.plaintext.is_some())
    }

    fn broadcast_wire(
        &mut self,
        wire: &Self::Wire,
        range: &FamilyRange,
    ) -> Result<Self::Wire, Self::Error> {
        let EncodingProgramWire::Scalar(value) = wire else {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        };
        Ok(EncodingProgramWire::Broadcast { value: value.clone(), range: range.clone() })
    }

    fn reduce_add(
        &mut self,
        value: &Self::Wire,
        range: &FamilyRange,
        _gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let EncodingProgramWire::Materialized { value, range: value_range } = value else {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        };
        if value_range != range {
            return Err(ExponentLutError::InvalidSparseLwrBlock);
        }
        Ok(EncodingProgramWire::Scalar(BggEncodingWire {
            vector: balanced_sum_family(value.vectors.clone())?,
            plaintext: value.plaintexts.clone().map(balanced_sum_family).transpose()?,
        }))
    }
}

#[derive(Clone)]
struct LutOutputParts {
    vector: Mat,
    plaintext: Option<Mat>,
}

#[derive(Clone)]
pub(crate) struct EncodingFamilyIntermediate {
    vectors: Family<Mat>,
    plaintexts: Option<Family<Mat>>,
}

fn program_binary_plaintext(
    lhs: &BggEncodingWire,
    rhs: &BggEncodingWire,
    operation: impl FnOnce(Mat, Mat) -> Mat,
) -> Option<Mat> {
    lhs.plaintext.clone().zip(rhs.plaintext.clone()).map(|(lhs, rhs)| operation(lhs, rhs))
}

fn active_encoding_family(
    input: &EncodingProgramWire,
    expected_range: &FamilyRange,
) -> Result<EncodingFamilyIntermediate, ExponentLutError> {
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
        EncodingProgramWire::Materialized { value, range } if range == expected_range => {
            Ok(value.clone())
        }
        EncodingProgramWire::Broadcast { value, range } if range == expected_range => {
            let vectors = Parallel::range(count)
                .map_values(|_| value.vector.clone())
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let plaintexts = value
                .plaintext
                .as_ref()
                .map(|plaintext| {
                    Parallel::range(count)
                        .map_values(|_| plaintext.clone())
                        .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)
                })
                .transpose()?;
            Ok(EncodingFamilyIntermediate { vectors, plaintexts })
        }
        EncodingProgramWire::Members(values) => {
            let indices = Parallel::range(count)
                .map_values(|index| index.as_int().add(Int::constant(start)))
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let vectors = values
                .vector_family()
                .clone()
                .parallel_gather(indices.clone())
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            let plaintexts = values
                .plaintext_family()
                .clone()
                .parallel_gather(indices)
                .map_err(|_| ExponentLutError::InvalidSparseLwrBlock)?;
            Ok(EncodingFamilyIntermediate { vectors, plaintexts: Some(plaintexts) })
        }
        _ => Err(ExponentLutError::InvalidSparseLwrBlock),
    }
}

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

pub(crate) fn reduce_encoding_family_segments(
    family: Family<Mat>,
    segment_count: usize,
    segment_size: usize,
) -> Result<Family<Mat>, ExponentLutError> {
    if segment_count == 0 ||
        segment_size == 0 ||
        segment_count.checked_mul(segment_size) != Some(concrete_mat_family_count(&family)?)
    {
        return Err(ExponentLutError::InvalidLut);
    }
    let mut current = family;
    let mut width = segment_size;
    while width > 1 {
        let next_width = width.div_ceil(2);
        let count = segment_count.checked_mul(next_width).ok_or(ExponentLutError::InvalidLut)?;
        let indices = |offset| {
            Parallel::range(count).map_values(|index| {
                let flat = index.as_int();
                let segment = flat.clone().div(Int::constant(next_width));
                let member = flat.rem(Int::constant(next_width)).mul(Int::constant(2));
                segment
                    .mul(Int::constant(width))
                    .add(member.add(Int::constant(offset)).rem(Int::constant(width)))
            })
        };
        let left = current
            .clone()
            .parallel_gather(indices(0).map_err(|_| ExponentLutError::InvalidLut)?)
            .map_err(|_| ExponentLutError::InvalidLut)?;
        let right = current
            .parallel_gather(indices(1).map_err(|_| ExponentLutError::InvalidLut)?)
            .map_err(|_| ExponentLutError::InvalidLut)?;
        current = (left, right)
            .parallel_zip_tuple_result(|index, (left, right)| {
                if width % 2 == 0 {
                    Ok(left + right)
                } else {
                    index
                        .as_int()
                        .rem(Int::constant(next_width))
                        .equal(Int::constant(next_width - 1))
                        .to_int()
                        .select(vec![left.clone() + right, left])
                }
            })
            .map_err(|_| ExponentLutError::InvalidLut)?;
        width = next_width;
    }
    Ok(current)
}

/// Setup-fixed family of GSW ciphertexts consumed by both public-key
/// preprocessing and online encoding evaluation.
#[derive(Clone)]
pub struct SetupRhsFamily {
    gsw: Family<Mat>,
    layout_id: Option<crate::noise_refresh::pbc::PbcLayoutId>,
}
impl SetupRhsFamily {
    pub fn new(gsw: Family<Mat>) -> Result<Self, ExponentLutError> {
        if *gsw.count() == IntExpr::constant(0) {
            Err(ExponentLutError::InvalidSparseLwrBlock)
        } else {
            Ok(Self { gsw, layout_id: None })
        }
    }

    pub fn new_for_layout(
        gsw: Family<Mat>,
        layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    ) -> Result<Self, ExponentLutError> {
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

    pub(crate) fn count(&self) -> &IntExpr {
        self.gsw.count()
    }
}

pub type EncodingRhsFamily = SetupRhsFamily;

fn lut_coefficient(
    ring: &mxx_dsl::Ring,
    layout: &crate::layout::ExpLayout,
    sigma: impl Into<IntExpr>,
    table: &[usize],
) -> Result<Mat, ExponentLutSamplingError> {
    // The coefficient is exactly
    // `W'^{-1} sum_k mono_m(L(k)) sigma(mono_m(-k))`.
    let width = table.len();
    if width == 0 || !width.is_power_of_two() || width != layout.width() {
        return Err(ExponentLutSamplingError::InvalidConfiguration("invalid LUT layout"));
    }
    let sigma = sigma.into();
    let modulus =
        ring.zero((1, 1)).matrix_type().modulus.evaluate(&ParamEnv::default()).map_err(|_| {
            ExponentLutSamplingError::InvalidConfiguration("ring modulus must be concrete")
        })?;
    let inverse = modular_inverse(&(BigInt::from(width) % &modulus), &modulus)
        .ok_or(ExponentLutSamplingError::InvalidConfiguration("LUT width is not invertible"))?;
    let mut terms = Vec::with_capacity(width);
    for (k, output) in table.iter().copied().enumerate() {
        let left = layout
            .mono(ring, &BigInt::from(output))
            .map_err(|_| ExponentLutSamplingError::InvalidConfiguration("invalid LUT exponent"))?;
        let right = layout
            .mono(ring, &-BigInt::from(k))
            .map_err(|_| ExponentLutSamplingError::InvalidConfiguration("invalid LUT exponent"))?
            .ring_automorphism(sigma.clone());
        terms.push(left * right);
    }
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let mut iter = std::mem::take(&mut terms).into_iter();
        while let Some(left) = iter.next() {
            next.push(match iter.next() {
                Some(right) => left + right,
                None => left,
            });
        }
        terms = next;
    }
    let result = terms.pop().expect("validated nonempty LUT");
    Ok(ring.polynomial([inverse.into()]) * result)
}

pub(crate) fn derive_lut_layout(
    base: &crate::layout::ExpLayout,
    width: usize,
    ring_dimension: usize,
) -> Result<crate::layout::ExpLayout, ExponentLutError> {
    if width == 0 || !width.is_power_of_two() || width > base.width() {
        return Err(ExponentLutError::InvalidLut);
    }
    let layout = crate::layout::ExpLayout {
        stride_log2: base.stride_log2,
        width_log2: width.trailing_zeros(),
    };
    layout.validate(ring_dimension).map_err(|_| ExponentLutError::InvalidLut)?;
    Ok(layout)
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

pub(crate) fn balanced_sum_family(family: Family<Mat>) -> Result<Mat, ExponentLutError> {
    let count = family
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(ExponentLutError::InvalidLut)?;
    if count == 0 {
        return Err(ExponentLutError::InvalidLut);
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
                    .map_err(|_| ExponentLutError::InvalidLut)?,
            )
            .map_err(|_| ExponentLutError::InvalidLut)?;
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
                    .map_err(|_| ExponentLutError::InvalidLut)?,
            )
            .map_err(|_| ExponentLutError::InvalidLut)?;
        current =
            Family::try_parallel_zip_many_values(vec![left, right], move |index, mut items| {
                let r = items.pop().ok_or(DslError::Schema)?;
                let l = items.pop().ok_or(DslError::Schema)?;
                if count % 2 == 1 {
                    let last = index.as_int().equal(mxx_dsl::Int::constant(next - 1)).to_int();
                    last.select(vec![l.clone() + r, l])
                } else {
                    Ok(l + r)
                }
            })
            .map_err(|_| ExponentLutError::InvalidLut)?;
        count = next;
    }
    Ok(current.get_static(0))
}

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
        ExponentLutError,
        program::{
            AttributeKnowledge, ExponentLutProgramBuilder, FamilyProgram, FamilyRange,
            FamilyWireSource, Gate, LutTable, ProgramFamilyBindings, ProgramFamilyId,
            ProgramInputId, ProgramWireType, RhsId, WireId,
        },
        public_key::{
            ExponentLutPublicKeyCompiler, LutPublicHelper, LutPublicHelperSet,
            ProgramCacheBindings, ProgramLutCache, PublicKeyWireFamily,
        },
        rhs::ExponentRhsPackage,
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
    fn odd_width_segmented_reduction_matches_scalar_addition() {
        let parameters = DCRTPolyParams::new(4, 1, 20, 4, None);
        let ring = Ring::new(BigInt::from(parameters.modulus().as_ref().clone()), 4);
        let mut context = DslContext::new("odd-width-segmented-reduction");
        for width in [1usize, 3, 5] {
            let values = (0..2 * width)
                .map(|index| ring.polynomial([(index + 1).into()]))
                .collect::<Vec<_>>();
            let actual = super::reduce_encoding_family_segments(
                Family::pack(values.clone()).unwrap(),
                2,
                width,
            )
            .unwrap();
            for segment in 0..2 {
                let inputs = &values[segment * width..(segment + 1) * width];
                let expected = inputs.iter().cloned().reduce(|left, right| left + right).unwrap();
                context = context
                    .output(format!("actual-{width}-{segment}"), actual.get_static(segment))
                    .unwrap()
                    .output(format!("expected-{width}-{segment}"), expected)
                    .unwrap()
                    .output(
                        format!("scalar-{width}-{segment}"),
                        super::balanced_sum_family(Family::pack(inputs.to_vec()).unwrap()).unwrap(),
                    )
                    .unwrap();
            }
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
        for width in [1usize, 3, 5] {
            for segment in 0..2 {
                let RuntimeValue::Matrix(expected) =
                    &result.outputs[&format!("expected-{width}-{segment}")]
                else {
                    panic!("expected matrix")
                };
                for prefix in ["actual", "scalar"] {
                    let RuntimeValue::Matrix(actual) =
                        &result.outputs[&format!("{prefix}-{width}-{segment}")]
                    else {
                        panic!("reduced matrix")
                    };
                    assert_eq!(actual.as_ref(), expected.as_ref());
                }
            }
        }
    }

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
        ProgramLutCache {
            switch: Family::pack(switch).expect("validated LUT switch family"),
            branch: Family::pack(branch).expect("validated LUT branch family"),
        }
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
        let parameters = DCRTPolyParams::new(4, 1, 20, 4, None);
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
            let parameters = DCRTPolyParams::new(ring_dimension as u32, 1, 20, 4, None);
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
        let output = builder.mul_private(builder.input_wire(input).unwrap(), rhs_id).unwrap();
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
                &ProgramCacheBindings::new().gate(0, k_rhs),
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
        let output = builder.lut(builder.input_wire(input_id).unwrap(), lut).unwrap();
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
            0,
            Family::pack(vec![ring.preimage_input("cached-lut-switch-k", (4, 2), 7)]).unwrap(),
            Family::pack(vec![ring.preimage_input("cached-lut-branch-k", (4, 2), 7)]).unwrap(),
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
        let encoding_compiler = ExponentLutEncodingCompiler::with_layout(public_key, layout);

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
                    vec![Gate::Lut { input, lut, output: transformed }],
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
            preprocessing
                .cache
                .family_lut_entry(0, 0)
                .unwrap()
                .branch
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
        let outputs = encoding_compiler
            .compile_program_with_ranges_and_cache(
                &program,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::from([(members, encoding_values)]),
                &ranges,
                &BTreeMap::from([(lut, private_helpers)]),
                &preprocessing.cache,
            )
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
    fn public_lut_lowering_fails_closed_when_plaintext_is_missing() {
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 }, 4).unwrap();
        let input_id = builder
            .input(
                1,
                crate::program::ProgramWireType {
                    stride_log2: 0,
                    knowledge: crate::program::AttributeKnowledge::Public,
                },
            )
            .unwrap();
        let table = LutTable::unary(1, 2, vec![1]).unwrap();
        let lut = builder.register_lut(table.clone()).unwrap();
        let output = builder.lut(builder.input_wire(input_id).unwrap(), lut).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let input = BggEncodingWire {
            vector: ring.input("missing-lut-plaintext-vector", (1, 2)),
            plaintext: None,
        };
        let helper = LutHelper::new(
            1,
            ExponentRhsPackage::new(ring.zero((2, 2))).unwrap(),
            BggEncodingWire { vector: ring.zero((1, 2)), plaintext: None },
        )
        .unwrap();
        let error = compiler
            .compile_program(
                &program,
                &BTreeMap::from([(input_id, input)]),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::from([(lut, LutHelperSet::new(&table, vec![helper]).unwrap())]),
            )
            .err()
            .expect("public LUT must fail closed without plaintext");
        assert!(matches!(
            error,
            ExponentLutError::Program(
                crate::program::ProgramValidationError::PublicAttributeRequired(_)
            )
        ));
    }

    #[test]
    fn real_encoding_backend_rejects_program_layout_mismatch() {
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutEncodingCompiler::with_layout(
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() },
            crate::layout::ExpLayout { stride_log2: 1, width_log2: 0 },
        );
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 }, 4).unwrap();
        let input_id = builder
            .input(
                1,
                crate::program::ProgramWireType {
                    stride_log2: 0,
                    knowledge: crate::program::AttributeKnowledge::Private,
                },
            )
            .unwrap();
        builder.output(builder.input_wire(input_id).unwrap()).unwrap();
        let program = builder.build().unwrap();
        let input = BggEncodingWire {
            vector: ring.input("layout-mismatch-vector", (1, 4)),
            plaintext: None,
        };
        let error = compiler
            .compile_program(
                &program,
                &BTreeMap::from([(input_id, input)]),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .err()
            .expect("real backend must reject mismatched program layout");
        assert!(matches!(
            error,
            ExponentLutError::Program(crate::program::ProgramValidationError::WidthMismatch)
        ));
    }

    #[test]
    fn encoding_lowering_rejects_the_same_public_binding_without_plaintext() {
        let ring = Ring::new(97, 4);
        let compiler = ExponentLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 4.into(),
            digit_count: 2.into(),
        });
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 }, 4).unwrap();
        let input_id = builder
            .input(
                2,
                crate::program::ProgramWireType {
                    stride_log2: 0,
                    knowledge: crate::program::AttributeKnowledge::Public,
                },
            )
            .unwrap();
        let wire = builder.input_wire(input_id).unwrap();
        builder.output(wire).unwrap();
        let program = builder.build().unwrap();
        let input = BggEncodingWire {
            vector: ring.input("missing-plaintext-vector", (1, 2)),
            plaintext: None,
        };
        let error = compiler
            .compile_program(
                &program,
                &BTreeMap::from([(input_id, input)]),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .err()
            .expect("encoding must reject public input without plaintext");
        assert!(matches!(
            error,
            ExponentLutError::Program(
                crate::program::ProgramValidationError::PublicAttributeRequired(_)
            )
        ));
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
        let table = LutTable::unary(1024, 2048, (0..1024).collect()).unwrap();
        let helpers = sampler
            .sample_native_readout_helper_families(
                ring.input("setup-secret", (1, 2)),
                ring.input("online-secret", (1, 2)),
                ring.bytes_input("hash-key", 32),
                &table,
                b"native-mask".as_slice(),
                b"native-helper".as_slice(),
            )
            .unwrap();
        assert_eq!(helpers.sigmas.len(), 1023);
        let graph = DslContext::new("native-readout-helper-family")
            .public_output("switches", helpers.switch_ciphertexts)
            .unwrap()
            .public_output("masks", helpers.public_masks)
            .unwrap()
            .private_output("online-masks", helpers.online_mask_vectors)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        // The unweighted helper matrices occur only once in the Parallel
        // body rather than O(width) times on the host.
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
            1
        );
        assert!(graph.graph.root_scope().nodes().iter().all(|node| !matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::FamilyPack { .. } |
                mxx_ir_core::node::NodeKind::FamilyGetStatic { .. }
        )));
        graph.validate(&ParamEnv::default()).expect("structural helper graph validates");
    }

    #[cfg(feature = "gpu")]
    mod gpu_tests {
        // GPU runtime coverage for the setup-fixed native Exponent-LUT evaluator.
        //
        // These tests deliberately build the private encoding and public projection
        // through separate compilers. The expected vector is the concrete BGG+
        // relation `s_mask * A - mu * s_payload * G`; it is not obtained from the
        // private evaluator's output.

        use crate::{
            ExponentLutEncodingCompiler, ExponentLutPublicKeyCompiler,
            encoding::{
                ExponentLutEncodingSampler, LutHelper, LutHelperSet, LutMaskBank, LutPublicMaskBank,
            },
            program::{AttributeKnowledge, ExponentLutProgramBuilder, LutTable, ProgramWireType},
            public_key::{ExponentLutPublicKeySampler, LutPublicHelper, LutPublicHelperSet},
            rhs::ExponentRhsPackage,
        };
        use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
        use mxx_dsl::{DslContext, Mat, Ring};
        use mxx_ir_core::{ParamEnv, node::ConstantMatrix};
        use mxx_primitives::{
            matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
            poly::{
                Poly, PolyParams,
                dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
            },
        };
        use mxx_runtime::{
            RuntimeValue, artifact::MemoryArtifactStore, backend::poly::gpu::gpu_backend, execute,
            transcript::SamplingMode,
        };
        use num_bigint::BigInt;
        use serial_test::serial;
        use std::{collections::BTreeMap, env};

        const RING_DIMENSION_ENV: &str = "MXX_EXPONENT_LUT_GPU_RING_DIMENSION";
        const CRT_BITS_ENV: &str = "MXX_EXPONENT_LUT_GPU_CRT_BITS";
        const BASE_BITS_ENV: &str = "MXX_EXPONENT_LUT_GPU_BASE_BITS";

        fn parameter(name: &str, default: u32) -> u32 {
            env::var(name)
                .map(|value| {
                    value.parse().unwrap_or_else(|_| panic!("{name} must be a positive integer"))
                })
                .unwrap_or(default)
        }

        struct Fixture {
            parameters: DCRTPolyParams,
            gpu_parameters: GpuDCRTPolyParams,
            ring: Ring,
            sampler: ExponentLutEncodingSampler,
            public_sampler: ExponentLutPublicKeySampler,
            compiler: ExponentLutEncodingCompiler,
            public_compiler: ExponentLutPublicKeyCompiler,
            mask_secret: Mat,
            payload_secret: Mat,
            hash_key: mxx_dsl::Bytes,
        }

        struct Relation {
            encoded: BggEncodingWire,
            expected_public: Mat,
            mu: Mat,
        }

        impl Fixture {
            fn new(width_log2: u32) -> Self {
                let ring_dimension = parameter(RING_DIMENSION_ENV, 4);
                let crt_bits = parameter(CRT_BITS_ENV, 17);
                let base_bits = parameter(BASE_BITS_ENV, 9);
                assert!(ring_dimension.is_power_of_two() && ring_dimension >= 2);
                assert!(base_bits > 0 && base_bits < 63 && crt_bits > base_bits);
                let parameters =
                    DCRTPolyParams::new(ring_dimension, 1, crt_bits as usize, base_bits, None);
                assert_eq!(parameters.modulus_digits(), 2, "GPU fixture requires digit_count=2");
                let (moduli, _, crt_depth) = parameters.to_crt();
                assert_eq!(crt_depth, 1);
                let gpu_parameters =
                    GpuDCRTPolyParams::new(ring_dimension, moduli, parameters.base_bits());
                let modulus = BigInt::from(parameters.modulus().as_ref().clone());
                let ring = Ring::new(modulus.clone(), ring_dimension as usize);
                let layout = mxx_bgg::BggSamplerLayout {
                    modulus: modulus.into(),
                    ring_dimension: (ring_dimension as usize).into(),
                    secret_dimension: 2,
                    digit_count: 2,
                    gadget_base: BigInt::from(1u64 << base_bits).into(),
                };
                let bgg = BggPublicKeyCompiler {
                    ring: ring.clone(),
                    base: layout.gadget_base.clone(),
                    digit_count: layout.digit_count.into(),
                };
                Self {
                    parameters,
                    gpu_parameters,
                    ring: ring.clone(),
                    sampler: ExponentLutEncodingSampler {
                        layout: layout.clone(),
                        exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2 },
                        gaussian_sigma: None,
                        gaussian_max_coefficient_bound: None,
                    },
                    public_sampler: ExponentLutPublicKeySampler {
                        layout,
                        exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2 },
                    },
                    compiler: ExponentLutEncodingCompiler::with_layout(
                        bgg.clone(),
                        crate::layout::ExpLayout { stride_log2: 0, width_log2 },
                    ),
                    public_compiler: ExponentLutPublicKeyCompiler::with_layout(
                        bgg,
                        crate::layout::ExpLayout { stride_log2: 0, width_log2 },
                    ),
                    mask_secret: ring.input("exponent-lut-gpu-mask-secret", (1, 2)),
                    payload_secret: ring.input("exponent-lut-gpu-payload-secret", (1, 2)),
                    hash_key: ring.bytes_input("exponent-lut-gpu-hash-key", 32),
                }
            }

            fn rotation(&self, exponent: usize) -> Mat {
                self.ring.constant((1, 1), ConstantMatrix::Rotation { exponent: exponent.into() })
            }

            fn input(&self, tag: impl Into<mxx_dsl::HashTag>, exponent: usize) -> BggEncodingWire {
                let mut values = self
                    .sampler
                    .sample_input_encodings(
                        self.mask_secret.clone(),
                        Some(self.payload_secret.clone()),
                        self.hash_key.clone(),
                        tag,
                        &[self.rotation(exponent)],
                    )
                    .expect("ciphertext-only input sample");
                values.pop().expect("one input encoding")
            }

            fn public_input(&self, tag: impl Into<mxx_dsl::HashTag>) -> BggPublicKeyWire {
                let mut values = self
                    .public_sampler
                    .sample_input_keys(self.hash_key.clone(), tag, 1)
                    .expect("ciphertext-only public input sample");
                values.pop().expect("one public input key")
            }

            fn helpers(
                &self,
                table: &[usize],
                tag: impl Into<mxx_dsl::HashTag>,
            ) -> (Vec<LutHelper>, Vec<LutPublicHelper>) {
                let tag = tag.into();
                let table = LutTable::unary(
                    table.len(),
                    self.parameters.ring_dimension() as usize,
                    table.to_vec(),
                )
                .expect("native LUT table");
                let private_bank = self
                    .sampler
                    .sample_mask_bank(
                        self.mask_secret.clone(),
                        self.hash_key.clone(),
                        table.values().len(),
                        tag.clone(),
                    )
                    .expect("native mask bank");
                let public_bank = self
                    .public_sampler
                    .sample_mask_bank(self.hash_key.clone(), table.values().len(), tag.clone())
                    .expect("public native mask bank");
                self.helpers_with_banks(&table, tag, private_bank.as_ref(), &public_bank)
            }

            fn helpers_with_banks(
                &self,
                table: &LutTable,
                tag: impl Into<mxx_dsl::HashTag>,
                private_bank: &LutMaskBank,
                public_bank: &std::sync::Arc<LutPublicMaskBank>,
            ) -> (Vec<LutHelper>, Vec<LutPublicHelper>) {
                let tag = tag.into();
                let private = self
                    .sampler
                    .sample_helpers_for_lut(
                        self.mask_secret.clone(),
                        Some(self.payload_secret.clone()),
                        self.hash_key.clone(),
                        table,
                        private_bank,
                        tag,
                    )
                    .expect("native setup helpers");
                let public = private
                    .iter()
                    .map(|helper| {
                        LutPublicHelper::with_mask_bank(
                            helper.sigma(),
                            helper.switch().public_projection(),
                            public_bank.clone(),
                        )
                        .expect("public native helper mask branch")
                    })
                    .collect();
                (private, public)
            }

            fn rhs(&self, tag: impl Into<mxx_dsl::HashTag>, exponent: usize) -> ExponentRhsPackage {
                self.sampler
                    .sample_cross_secret_rhs(
                        self.payload_secret.clone(),
                        self.payload_secret.clone(),
                        self.rotation(exponent),
                        self.hash_key.clone(),
                        tag,
                    )
                    .expect("C-only RHS package")
            }

            fn execute(&self, name: &str, relations: Vec<Relation>) {
                let gadget = self.ring.gadget(
                    2,
                    self.compiler.gadget_base.clone(),
                    self.compiler.digit_count.clone(),
                );
                let mut context = DslContext::new(name);
                for (index, relation) in relations.iter().enumerate() {
                    assert!(relation.encoded.plaintext.is_none());
                    let expected_vector = self.mask_secret.clone() *
                        relation.expected_public.clone() -
                        relation.mu.clone() * (self.payload_secret.clone() * gadget.clone());
                    context = context
                        .output(format!("vector-{index}"), relation.encoded.vector.clone())
                        .unwrap()
                        .output(format!("expected-vector-{index}"), expected_vector)
                        .unwrap();
                }
                let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
                let mask = concrete_secret(&self.parameters, [2, 1]);
                let payload = concrete_secret(&self.parameters, [1, 2]);
                let result = execute(
                    &graph,
                    &mut gpu_backend([self.gpu_parameters.clone()]),
                    BTreeMap::from([
                        (
                            "exponent-lut-gpu-mask-secret".to_owned(),
                            RuntimeValue::matrix(
                                GpuDCRTPolyMatrix::from_cpu_matrix(&self.gpu_parameters, &mask)
                                    .into(),
                            ),
                        ),
                        (
                            "exponent-lut-gpu-payload-secret".to_owned(),
                            RuntimeValue::matrix(
                                GpuDCRTPolyMatrix::from_cpu_matrix(&self.gpu_parameters, &payload)
                                    .into(),
                            ),
                        ),
                        (
                            "exponent-lut-gpu-hash-key".to_owned(),
                            RuntimeValue::Bytes(vec![0x91; 32]),
                        ),
                    ]),
                    &mut MemoryArtifactStore::default(),
                    SamplingMode::Fresh,
                )
                .unwrap();
                for index in 0..relations.len() {
                    let RuntimeValue::Matrix(vector) = &result.outputs[&format!("vector-{index}")]
                    else {
                        panic!("vector output")
                    };
                    let RuntimeValue::Matrix(expected_vector) =
                        &result.outputs[&format!("expected-vector-{index}")]
                    else {
                        panic!("expected vector output")
                    };
                    assert_eq!(
                        vector.as_ref(),
                        expected_vector.as_ref(),
                        "relation {index}: vector"
                    );
                }
            }
        }

        fn concrete_secret(parameters: &DCRTPolyParams, coefficients: [u32; 2]) -> DCRTPolyMatrix {
            let mut values = vec![0u32; parameters.ring_dimension() as usize];
            values[..coefficients.len()].copy_from_slice(&coefficients);
            DCRTPolyMatrix::from_poly_vec(
                parameters,
                vec![vec![
                    DCRTPoly::from_u32s(parameters, &values),
                    DCRTPoly::from_usize_to_constant(parameters, 1),
                ]],
            )
        }

        fn unary_table() -> Vec<usize> {
            vec![1, 0]
        }

        #[test]
        #[serial(dcrt_runtime)]
        fn test_gpu_native_unary_exhaustive_inputs_distinct_secrets() {
            let fixture = Fixture::new(1);
            let table = unary_table();
            let (helpers, public_helpers) = fixture.helpers(&table, b"unary-helpers".as_slice());
            let public_input = fixture.public_input(b"unary-input".as_slice());
            let cache = fixture
                .public_compiler
                .build_lut_cache(
                    &public_input.matrix,
                    &LutTable::unary(
                        table.len(),
                        fixture.parameters.ring_dimension() as usize,
                        table.clone(),
                    )
                    .unwrap(),
                    &public_helpers,
                )
                .unwrap();
            let mut relations = Vec::new();
            for exponent in 0..table.len() {
                let encoded = fixture
                    .compiler
                    .test_single_input_lut(
                        &fixture.input(b"unary-input".as_slice(), exponent),
                        &table,
                        &helpers,
                        &cache,
                    )
                    .unwrap();
                let expected_public = fixture
                    .public_compiler
                    .single_input_lut(&public_input.matrix, &table, &public_helpers)
                    .unwrap();
                relations.push(Relation {
                    encoded,
                    expected_public,
                    mu: fixture.rotation(table[exponent]),
                });
            }
            fixture.execute("exponent-lut-native-unary-gpu", relations);
        }

        #[test]
        #[serial(dcrt_runtime)]
        fn test_gpu_native_binary_all_u_plus_bv_pairs() {
            let fixture = Fixture::new(2);
            let table = LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap();
            let (helpers, public_helpers) =
                fixture.helpers(table.values(), b"binary-helpers".as_slice());
            let public_lhs = fixture.public_input(b"binary-lhs".as_slice());
            let mut relations = Vec::new();
            for lhs_exponent in 0..2 {
                for rhs_exponent in 0..2 {
                    let rhs = fixture
                        .rhs(format!("binary-rhs-{rhs_exponent}").into_bytes(), 2 * rhs_exponent);
                    let fused_public = rhs
                        .gsw_ciphertext()
                        .clone()
                        .decompose(
                            fixture.compiler.gadget_base.clone(),
                            fixture.compiler.digit_count.clone(),
                        )
                        .mul_small_rhs(public_lhs.matrix.clone());
                    let cache = fixture
                        .public_compiler
                        .build_lut_cache(&fused_public, &table, &public_helpers)
                        .unwrap();
                    let encoded = fixture
                        .compiler
                        .test_two_input_lut(
                            &fixture.input(b"binary-lhs".as_slice(), lhs_exponent),
                            &rhs,
                            &table,
                            &helpers,
                            &cache,
                        )
                        .unwrap();
                    let expected_public = fixture
                        .public_compiler
                        .two_input_lut(
                            &public_lhs.matrix,
                            &rhs,
                            2,
                            2,
                            table.values(),
                            &public_helpers,
                        )
                        .unwrap();
                    let index = lhs_exponent + 2 * rhs_exponent;
                    relations.push(Relation {
                        encoded,
                        expected_public,
                        mu: fixture.rotation(table.values()[index]),
                    });
                }
            }
            fixture.execute("exponent-lut-native-binary-gpu", relations);
        }

        #[test]
        #[serial(dcrt_runtime)]
        fn test_gpu_native_two_stage_lut_chain() {
            let fixture = Fixture::new(1);
            let first_table = vec![0, 1];
            let second_table = unary_table();
            let private_bank = fixture
                .sampler
                .sample_mask_bank(
                    fixture.mask_secret.clone(),
                    fixture.hash_key.clone(),
                    2,
                    b"chain-mask-bank".as_slice(),
                )
                .expect("chain private mask bank");
            let public_bank = fixture
                .public_sampler
                .sample_mask_bank(fixture.hash_key.clone(), 2, b"chain-mask-bank".as_slice())
                .expect("chain public mask bank");
            let (first_helpers, first_public_helpers) = fixture.helpers_with_banks(
                &LutTable::unary(
                    2,
                    fixture.parameters.ring_dimension() as usize,
                    first_table.clone(),
                )
                .unwrap(),
                b"chain-first".as_slice(),
                private_bank.as_ref(),
                &public_bank,
            );
            let (second_helpers, second_public_helpers) = fixture.helpers_with_banks(
                &LutTable::unary(
                    2,
                    fixture.parameters.ring_dimension() as usize,
                    second_table.clone(),
                )
                .unwrap(),
                b"chain-second".as_slice(),
                private_bank.as_ref(),
                &public_bank,
            );
            let input = fixture.input(b"chain-input".as_slice(), 1);
            let public_input = fixture.public_input(b"chain-input".as_slice());
            let first_cache = fixture
                .public_compiler
                .build_lut_cache(
                    &public_input.matrix,
                    &LutTable::unary(
                        2,
                        fixture.parameters.ring_dimension() as usize,
                        first_table.clone(),
                    )
                    .unwrap(),
                    &first_public_helpers,
                )
                .unwrap();
            let first = fixture
                .compiler
                .test_single_input_lut(&input, &first_table, &first_helpers, &first_cache)
                .unwrap();
            let first_public = fixture
                .public_compiler
                .single_input_lut(&public_input.matrix, &first_table, &first_public_helpers)
                .unwrap();
            let second_cache = fixture
                .public_compiler
                .build_lut_cache(
                    &first_public,
                    &LutTable::unary(
                        2,
                        fixture.parameters.ring_dimension() as usize,
                        second_table.clone(),
                    )
                    .unwrap(),
                    &second_public_helpers,
                )
                .unwrap();
            let second = fixture
                .compiler
                .test_single_input_lut(&first, &second_table, &second_helpers, &second_cache)
                .unwrap();
            let second_public = fixture
                .public_compiler
                .single_input_lut(&first_public, &second_table, &second_public_helpers)
                .unwrap();
            fixture.execute(
                "exponent-lut-native-chain-gpu",
                vec![Relation {
                    encoded: second,
                    expected_public: second_public,
                    mu: fixture.rotation(0),
                }],
            );
        }

        #[test]
        #[serial(dcrt_runtime)]
        fn test_gpu_native_generic_exponent_lut_program() {
            let fixture = Fixture::new(1);
            let table = LutTable::unary(2, 2, unary_table()).unwrap();
            let mut builder = ExponentLutProgramBuilder::new();
            builder
                .set_layout(
                    crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 },
                    fixture.parameters.ring_dimension() as usize,
                )
                .unwrap();
            let input_id = builder
                .input(
                    2,
                    ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private },
                )
                .unwrap();
            let lut_id = builder.register_lut(table.clone()).unwrap();
            let output_id = builder.lut(builder.input_wire(input_id).unwrap(), lut_id).unwrap();
            builder.output(output_id).unwrap();
            let program = builder.build().unwrap();
            let (private_helpers, public_helpers) =
                fixture.helpers(table.values(), b"program-helpers".as_slice());
            let private_set = LutHelperSet::new(&table, private_helpers).unwrap();
            let public_set = LutPublicHelperSet::new(&table, public_helpers).unwrap();
            let encoded = fixture.input(b"program-input".as_slice(), 1);
            let public_input = fixture.public_input(b"program-input".as_slice());
            let ranges = crate::program::ProgramFamilyBindings::new();
            let public_inputs = BTreeMap::from([(input_id, public_input)]);
            let public_sets = BTreeMap::from([(lut_id, public_set)]);
            let preprocessing = fixture
                .public_compiler
                .compile_program_with_preprocessing(
                    &program,
                    &public_inputs,
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &ranges,
                    &public_sets,
                )
                .unwrap();
            let actual = fixture
                .compiler
                .compile_program_with_ranges_and_cache(
                    &program,
                    &BTreeMap::from([(input_id, encoded)]),
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &ranges,
                    &BTreeMap::from([(lut_id, private_set)]),
                    &preprocessing.cache,
                )
                .unwrap();
            let expected = fixture
                .public_compiler
                .compile_program(
                    &program,
                    &public_inputs,
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &public_sets,
                )
                .unwrap();
            fixture.execute(
                "exponent-lut-native-program-gpu",
                vec![Relation {
                    encoded: actual[&output_id].clone(),
                    expected_public: expected[&output_id].matrix.clone(),
                    mu: fixture.rotation(0),
                }],
            );
        }
    }
}
