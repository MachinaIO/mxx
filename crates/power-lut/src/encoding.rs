//! Power-LUT operations over plain BGG+ encoding wires.
//!
//! This module owns the application-level orchestration: fixed-secret
//! automorphism alignment, ClearCoeff, and generic LUT evaluation.
//! The algebraic BGG primitives (addition and gadget products) remain in
//! `mxx-bgg`; this module supplies Power-LUT routing around them.
//!
//! A normal operation consumes and returns [`BggEncodingWire`] values. The
//! public counterpart lives in
//! [`crate::public_key`]; its methods deliberately mirror the formulas here while
//! accepting public matrices, public package projections, and
//! [`crate::public_key::AutomorphismPublicHelper`] values. One-hot families are
//! supplied as explicit
//! selector packages paired with public scalar values; the sparse-LWR
//! application builds those bindings without exposing them in this core.

use std::collections::BTreeMap;

use crate::{
    PowerLutError,
    program::{
        FamilyRange, PowerLutProgram, ProgramBindings, ProgramFamilyRanges, ProgramInputId,
        ProgramLoweringBackend, ProgramWireId, RhsInputId, lower_program,
    },
    rhs::{
        ManifestSecretMetadata, PowerRhsPackage, PowerRhsPackageArtifactNames, PowerRhsPackageError,
    },
};
use mxx_bgg::{BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{Bytes, DslError, Family, HashTag, Mat, Parallel};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    node::{ConcatAxis, IndexRange},
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Validated key-switch and mask material for one non-trivial ring
/// automorphism used by `ClearCoeff`.
#[derive(Clone)]
pub struct AutomorphismHelper {
    index: usize,
    switch: PowerRhsPackage,
    mask: BggEncodingWire,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Public artifact names needed to import an [`AutomorphismHelper`].
pub struct AutomorphismHelperArtifactNames {
    /// Private key-switch RHS package.
    pub switch: PowerRhsPackageArtifactNames,
    /// Producer-bound mask encoding for this automorphism.
    pub mask: BggEncodingArtifactNames,
}

impl AutomorphismHelper {
    /// Internal constructor. Setup callers should use manifest-bound import so
    /// that the switch transition and mask dimensions are checked together.
    pub(crate) fn new(
        index: usize,
        switch: PowerRhsPackage,
        mask: BggEncodingWire,
    ) -> Result<Self, PowerLutError> {
        crate::ensure_ciphertext_only(&mask)?;
        let n = mask
            .pubkey
            .matrix
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|v| v.to_usize())
            .ok_or(PowerLutError::InvalidAutomorphismHelper)?;
        if index == 0 || index >= 2 * n || index % 2 == 0 {
            return Err(PowerLutError::InvalidAutomorphismHelper);
        }
        let mask_type = mask.pubkey.matrix.matrix_type();
        let vector_type = mask.vector.matrix_type();
        // A BGG encoding stores a row vector with the same number of columns
        // as its public matrix. The public matrix rows are the secret
        // dimension; they are not the vector width. Checking the latter
        // relation is the fail-closed shape invariant used by the helper.
        if mask_type.modulus.canonicalize() != vector_type.modulus.canonicalize() ||
            mask_type.ring_dimension.canonicalize() != vector_type.ring_dimension.canonicalize() ||
            mask_type.columns.canonicalize() != vector_type.columns.canonicalize() ||
            vector_type.rows.evaluate(&ParamEnv::default()).ok().and_then(|v| v.to_usize()) !=
                Some(1)
        {
            return Err(PowerLutError::InvalidAutomorphismHelper);
        }
        Ok(Self { index, switch, mask })
    }

    pub(crate) fn switch(&self) -> &PowerRhsPackage {
        &self.switch
    }
    pub(crate) fn mask(&self) -> &BggEncodingWire {
        &self.mask
    }
    pub(crate) fn index(&self) -> usize {
        self.index
    }

    /// Imports and validates helper artifacts for `index` from `manifest`.
    /// Validation checks production, role, secret transition, and shapes
    /// before returning any runtime wires.
    pub fn artifact_input(
        production_id: mxx_ir_core::artifact::ProductionId,
        manifest: &mxx_ir_core::artifact::Manifest,
        index: usize,
        names: AutomorphismHelperArtifactNames,
    ) -> Result<Self, PowerLutError> {
        let expected_role = serde_json::json!({
            "AutomorphismSwitch": { "index": index }
        });
        let actual_role = manifest
            .artifacts
            .get(&names.switch.gsw_ciphertext)
            .and_then(|artifact| artifact.layout.as_deref())
            .and_then(|layout| serde_json::from_str::<serde_json::Value>(layout).ok())
            .and_then(|document| document.get("role").cloned())
            .ok_or(PowerLutError::InvalidAutomorphismHelper)?;
        if actual_role != expected_role {
            return Err(PowerLutError::InvalidAutomorphismHelper);
        }
        let switch_name = names.switch.gsw_ciphertext.clone();
        let target_identity = manifest
            .artifacts
            .get(&switch_name)
            .and_then(|artifact| artifact.layout.as_deref())
            .and_then(|layout| serde_json::from_str::<serde_json::Value>(layout).ok())
            .and_then(|document| document.get("target").cloned())
            .and_then(|target| target.get("identity").cloned())
            .ok_or(PowerLutError::InvalidAutomorphismHelper)?;
        let switch =
            PowerRhsPackage::artifact_input(production_id.clone(), manifest, names.switch)?;
        let expected_mask_role = serde_json::json!({
            "AutomorphismMask": {
                "index": index,
                "source_secret": target_identity,
            }
        });
        let mask = artifact_input_with_role(
            production_id,
            manifest,
            names.mask,
            Some(&expected_mask_role),
        )?;
        Self::new(index, switch, mask)
    }
}

/// Compiler for Power-LUT encoding graphs.
///
/// This wrapper composes generic BGG primitives with Power-LUT Fuse,
/// automorphism, and LUT routing checks. Setup/layout identities are validated
/// only while importing independently stored artifacts; runtime wires remain
/// plain BGG values. It stores the configured generic
/// compiler directly; public-key projection is provided separately by
/// [`crate::PowerLutPublicKeyCompiler`].
pub struct PowerLutEncodingCompiler {
    /// Generic BGG encoding operations used by Power-LUT graph construction.
    pub bgg: BggEncodingCompiler,
}

/// Errors raised while constructing setup-time Power-LUT input/helper data.
#[derive(Debug, Error)]
pub enum PowerLutSamplingError {
    #[error(transparent)]
    /// The underlying BGG sampler rejected a shape or Gaussian configuration.
    Bgg(#[from] mxx_bgg::BggSampleError),
    #[error(transparent)]
    /// DSL family/hash construction failed.
    Dsl(#[from] DslError),
    #[error(transparent)]
    /// The canonical automorphism helper invariant was violated.
    PowerLut(#[from] PowerLutError),
    #[error(transparent)]
    /// The constructed RHS package had invalid material or companion shape.
    Rhs(#[from] PowerRhsPackageError),
    #[error("invalid Power-LUT sampler configuration: {0}")]
    /// Setup inputs are incompatible with the selected BGG layout.
    InvalidConfiguration(&'static str),
}

/// Setup-time sampler for ordinary Power-LUT input encodings and reusable
/// automorphism helpers.
///
/// Public matrices are generated by [`mxx_bgg::BggPublicKeySampler`] from the
/// caller's public hash input and domain-separated tags. Private matrix rows
/// and BGG errors are sampled through the existing uniform/Gaussian DSL
/// samplers; no secret or private randomness is derived from a public hash.
/// `max_lut_width` controls only the number of sign-flag helper indices and is
/// independent of [`PowerLutProgram`].
#[derive(Clone)]
pub struct PowerLutEncodingSampler {
    /// BGG dimensions and gadget parameters shared by all generated wires.
    pub layout: mxx_bgg::BggSamplerLayout,
    /// Optional BGG Gaussian error distribution. Both fields must be present
    /// together; when both are absent, the sampler emits zero error.
    pub gaussian_sigma: Option<mxx_ir_core::RealExpr>,
    /// Explicit coefficient cutoff required when Gaussian errors are enabled.
    pub gaussian_max_coefficient_bound: Option<IntExpr>,
}

struct CrossSecretRhsPublicKeys {
    by_column: Vec<Vec<BggPublicKeyWire>>,
    companions: Vec<Mat>,
}

impl PowerLutEncodingSampler {
    /// Samples the global BGG secret with the conventional augmented-secret
    /// shape `(s_bar, 1)`. The non-constant prefix is sampled from the
    /// private uniform interval sampler; it is deliberately not derived from
    /// the public hash key. Callers that already own a setup secret may pass
    /// it directly to the other methods instead.
    pub fn sample_secret(&self) -> Result<Mat, PowerLutSamplingError> {
        if self.layout.secret_dimension < 2 {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "secret dimension must be at least two for an augmented secret",
            ));
        }
        let ring = self.layout.ring();
        let prefix = ring.uniform_interval((1, self.layout.secret_dimension - 1), -1, 1);
        Ok(Mat::concat(ConcatAxis::Columns, vec![prefix, ring.identity(1)]))
    }

    /// Samples public-key matrices and one ordinary input encoding.
    ///
    /// The returned encoding is the second member of the existing BGG packed
    /// sample: the first member is the conventional constant encoding needed
    /// by [`mxx_bgg::BggEncodingSampler`]. The returned wire is always
    /// ciphertext-only: transient sampling plaintext is never retained.
    pub fn sample_input_encoding(
        &self,
        secret: Mat,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        plaintext: Mat,
    ) -> Result<BggEncodingWire, PowerLutSamplingError> {
        let public_keys = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() }.sample(
            hash_key,
            tag,
            &[false],
        );
        let encodings = self.sample_encodings(secret, &public_keys, &[plaintext])?;
        encodings
            .into_iter()
            .nth(1)
            .ok_or(PowerLutSamplingError::InvalidConfiguration("BGG input sample is empty"))
    }

    /// Samples one ordinary BGG encoding under an independently supplied
    /// public matrix. The generic packed sampler includes the conventional
    /// constant column; this entry point supplies a zero constant public key
    /// and returns only the requested plaintext encoding.
    ///
    /// The supplied key is intentionally required to be ciphertext-only. The
    /// returned wire is also ciphertext-only, even though the underlying BGG
    /// sampler accepts revealable public keys for general callers.
    pub fn sample_encoding_for_public_matrix(
        &self,
        secret: Mat,
        public_key: BggPublicKeyWire,
        plaintext: Mat,
    ) -> Result<BggEncodingWire, PowerLutSamplingError> {
        self.validate_secret(&secret)?;
        if public_key.reveal_plaintext {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "supplied public key must not reveal plaintext",
            ));
        }
        let ring = self.layout.ring();
        let expected_public =
            ring.matrix_type((self.layout.secret_dimension, self.layout.public_key_columns()));
        let expected_plaintext = ring.matrix_type((1, 1));
        if public_key.matrix.matrix_type() != &expected_public {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "supplied public key has the wrong BGG matrix shape",
            ));
        }
        if plaintext.matrix_type() != &expected_plaintext {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "plaintext must be one ring element",
            ));
        }
        let constant = BggPublicKeyWire {
            matrix: ring.zero((self.layout.secret_dimension, self.layout.public_key_columns())),
            reveal_plaintext: false,
        };
        let encodings = mxx_bgg::BggEncodingSampler {
            layout: self.layout.clone(),
            gaussian_sigma: self.gaussian_sigma.clone(),
            gaussian_max_coefficient_bound: self.gaussian_max_coefficient_bound.clone(),
        }
        .sample(secret, &[constant, public_key], &[plaintext])?;
        let encoding = encodings
            .into_iter()
            .nth(1)
            .ok_or(PowerLutSamplingError::InvalidConfiguration("BGG input sample is empty"))?;
        if encoding.plaintext.is_some() || encoding.pubkey.reveal_plaintext {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "BGG sampler returned plaintext metadata",
            ));
        }
        Ok(encoding)
    }

    /// Reuses the ordinary BGG encoding sampler for a caller-supplied public
    /// key set and plaintext set. This is the generic entry point for input
    /// families and keeps public matrix derivation in the BGG sampler.
    fn sample_encodings(
        &self,
        secret: Mat,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &[Mat],
    ) -> Result<Vec<BggEncodingWire>, PowerLutSamplingError> {
        Ok(mxx_bgg::BggEncodingSampler {
            layout: self.layout.clone(),
            gaussian_sigma: self.gaussian_sigma.clone(),
            gaussian_max_coefficient_bound: self.gaussian_max_coefficient_bound.clone(),
        }
        .sample(secret, public_keys, plaintexts)?)
    }

    /// Samples a packed row of BGG encodings in one relation.
    ///
    /// `plaintexts` contains all companion digits for one target GSW column
    /// as a `1 x count` row. Keeping that row intact means the caller never
    /// slices a `GadgetDecompose` result merely to recover individual digits;
    /// the packed BGG relation performs the same canonical column packing as
    /// [`mxx_bgg::BggEncodingSampler`]. Returned wires are ordinary
    /// `BggEncodingWire` values obtained only by slicing the packed arithmetic
    /// result. The public-key list has one additional leading key for the
    /// conventional constant encoding, so `public_keys.len() == count + 1`;
    /// the remaining `count == source_dimension * digit_count` keys are the
    /// packed row's canonical tower-major limbs.
    fn sample_packed_encodings(
        &self,
        secret: Mat,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &Mat,
    ) -> Result<Vec<BggEncodingWire>, PowerLutSamplingError> {
        if public_keys.len() !=
            plaintexts
                .matrix_type()
                .columns
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|value| value.to_usize())
                .map(|count| count + 1)
                .unwrap_or(0)
        {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "packed BGG plaintext count does not match public keys",
            ));
        }
        let columns = self.layout.public_key_columns();
        let all_public_keys = Mat::concat(
            ConcatAxis::Columns,
            public_keys.iter().map(|key| key.matrix.clone()).collect(),
        );
        let encoded_plaintexts = Mat::concat(
            ConcatAxis::Columns,
            vec![self.layout.ring().identity(1), plaintexts.clone()],
        );
        let gadget = self.layout.ring().gadget(
            self.layout.secret_dimension,
            self.layout.gadget_base.clone(),
            self.layout.digit_count,
        );
        let error = self.sample_error((1, columns * public_keys.len()))?;
        let packed_vector =
            secret.clone() * all_public_keys - encoded_plaintexts.tensor(secret * gadget) + error;
        Ok(public_keys
            .iter()
            .enumerate()
            .map(|(index, key)| BggEncodingWire {
                vector: packed_vector.clone().slice(
                    None,
                    Some(IndexRange {
                        start: (columns * index).into(),
                        end: (columns * (index + 1)).into(),
                    }),
                ),
                pubkey: key.clone(),
                plaintext: None,
            })
            .collect())
    }

    /// Returns the canonical ClearCoeff helper indices required for a maximum
    /// LUT width. The sequence is exactly
    /// `2*n/2^(i+1) + 1`, in sieve round order.
    pub fn automorphism_helper_indices(
        &self,
        max_lut_width: usize,
    ) -> Result<Vec<usize>, PowerLutSamplingError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutSamplingError::InvalidConfiguration(
                "ring dimension must be a concrete positive integer",
            ))?;
        if max_lut_width == 0 ||
            !max_lut_width.is_power_of_two() ||
            max_lut_width > n ||
            n % max_lut_width != 0
        {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "maximum LUT width must be a power of two dividing the ring dimension",
            ));
        }
        Ok((0..max_lut_width.trailing_zeros() as usize)
            .map(|round| (2 * n / (1usize << (round + 1))) + 1)
            .collect())
    }

    /// Samples and returns the reusable helper packages for all ClearCoeff
    /// rounds up to `max_lut_width`.
    ///
    /// The helper switch is sampled as a cross-secret GSW relation
    /// `t_k C_k = s G + E`, where `t_k = sigma_k(s)`. Companion public
    /// matrices use domain-separated BGG hash-sampler tags. The mask row is
    /// sampled directly as `s D_k - t_k G + e_h`, with `D_k` public and all
    /// private rows/errors coming from the appropriate private samplers.
    pub fn sample_automorphism_helpers(
        &self,
        secret: Mat,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        max_lut_width: usize,
    ) -> Result<Vec<AutomorphismHelper>, PowerLutSamplingError> {
        self.validate_secret(&secret)?;
        let indices = self.automorphism_helper_indices(max_lut_width)?;
        let mut helpers = Vec::with_capacity(indices.len());
        let mut base_tag = tag.into();
        base_tag.push("power-lut-automorphism");
        for index in indices {
            let source = secret.clone().ring_automorphism(index);
            let switch_tag = canonical_switch_companion_tag(&base_tag, index);
            let switch = self.sample_cross_secret_rhs(
                source.clone(),
                secret.clone(),
                self.layout.ring().identity(1),
                hash_key.clone(),
                switch_tag,
            )?;

            let mask_tag = canonical_mask_tag(&base_tag, index);
            let mut mask_public = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() }
                .sample(hash_key.clone(), mask_tag, &[])
                .into_iter()
                .next()
                .ok_or(PowerLutSamplingError::InvalidConfiguration("mask key sample is empty"))?;
            // BGG reserves the leading member of every sampled family for its
            // conventional constant relation and marks it as revealed.  The
            // Power-LUT helper mask is a ciphertext-only wire, so retain the
            // sampled matrix while clearing that metadata at this API
            // boundary.  This does not change the mask relation.
            mask_public.reveal_plaintext = false;
            let mask_error = self.sample_error((1, self.layout.public_key_columns()))?;
            let gadget = self.layout.ring().gadget(
                self.layout.secret_dimension,
                self.layout.gadget_base.clone(),
                self.layout.digit_count,
            );
            let mask = BggEncodingWire {
                vector: secret.clone() * mask_public.matrix.clone() - source.clone() * gadget +
                    mask_error,
                pubkey: mask_public,
                plaintext: None,
            };
            helpers.push(AutomorphismHelper::new(index, switch, mask)?);
        }
        Ok(helpers)
    }

    fn validate_secret(&self, secret: &Mat) -> Result<(), PowerLutSamplingError> {
        let expected = self.layout.ring().matrix_type((1, self.layout.secret_dimension));
        let actual = secret.matrix_type();
        if actual.modulus.canonicalize() != expected.modulus.canonicalize() ||
            actual.ring_dimension.canonicalize() != expected.ring_dimension.canonicalize() ||
            actual.rows.canonicalize() != expected.rows.canonicalize() ||
            actual.columns.canonicalize() != expected.columns.canonicalize()
        {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "secret must have the BGG sampler layout shape",
            ));
        }
        Ok(())
    }

    fn sample_error(&self, shape: impl mxx_dsl::IntoShape) -> Result<Mat, PowerLutSamplingError> {
        match (&self.gaussian_sigma, &self.gaussian_max_coefficient_bound) {
            (Some(sigma), Some(bound)) => {
                Ok(self.layout.ring().gaussian(shape, sigma.clone(), bound.clone()))
            }
            (None, None) => Ok(self.layout.ring().zero(shape)),
            _ => Err(PowerLutSamplingError::InvalidConfiguration(
                "Gaussian sigma and coefficient bound must be supplied together",
            )),
        }
    }

    /// Samples a ciphertext-assisted RHS package for an arbitrary hidden
    /// payload `y`, satisfying `source * C = y * target * G + E`.
    ///
    /// The payload is used only in the private GSW relation. Companion public
    /// matrices are derived exclusively from `hash_key` and `tag`, so the same
    /// setup namespace produces an identical public projection for every
    /// payload. Returned packages retain neither `payload` nor plaintext
    /// metadata; evaluator-side Fuse sees only ciphertext matrices.
    pub fn sample_cross_secret_rhs(
        &self,
        source: Mat,
        target: Mat,
        payload: Mat,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
    ) -> Result<PowerRhsPackage, PowerLutSamplingError> {
        let public_keys = self.sample_cross_secret_rhs_public_keys(hash_key, tag.into())?;
        self.sample_cross_secret_rhs_with_public_keys(source, target, payload, &public_keys)
    }

    fn sample_cross_secret_rhs_public_keys(
        &self,
        hash_key: Bytes,
        tag: HashTag,
    ) -> Result<CrossSecretRhsPublicKeys, PowerLutSamplingError> {
        let target_columns = self.layout.public_key_columns();
        let sampler = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() };
        let by_column = (0..target_columns)
            .map(|column| {
                let mut column_tag = tag.clone();
                column_tag.push(IntExpr::constant(column));
                sampler.sample(hash_key.clone(), column_tag, &vec![false; target_columns])
            })
            .collect::<Vec<_>>();
        let mut companions = Vec::with_capacity(self.layout.secret_dimension * target_columns);
        for row in 0..self.layout.secret_dimension {
            for column in 0..target_columns {
                let start = row * self.layout.digit_count;
                let end = start + self.layout.digit_count;
                companions.push(Mat::concat(
                    ConcatAxis::Columns,
                    by_column[column][start + 1..end + 1]
                        .iter()
                        .map(|key| key.matrix.clone())
                        .collect(),
                ));
            }
        }
        Ok(CrossSecretRhsPublicKeys { by_column, companions })
    }

    fn sample_cross_secret_rhs_with_public_keys(
        &self,
        source: Mat,
        target: Mat,
        payload: Mat,
        public_keys: &CrossSecretRhsPublicKeys,
    ) -> Result<PowerRhsPackage, PowerLutSamplingError> {
        let ring = self.layout.ring();
        let source_dimension = self.layout.secret_dimension;
        if source_dimension < 2 {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "cross-secret RHS sampling requires at least two secret coordinates",
            ));
        }
        self.validate_secret(&source)?;
        self.validate_secret(&target)?;
        if payload.matrix_type() != &ring.matrix_type((1, 1)) {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "RHS payload must be one ring element",
            ));
        }
        let target_columns = self.layout.public_key_columns();
        if public_keys.by_column.len() != target_columns ||
            public_keys.by_column.iter().any(|keys| keys.len() != target_columns + 1) ||
            public_keys.companions.len() != source_dimension * target_columns
        {
            return Err(PowerLutSamplingError::InvalidConfiguration(
                "cross-secret RHS public-key set has the wrong shape",
            ));
        }
        let top = ring.uniform_residue((source_dimension - 1, target_columns));
        // The augmented secret's final coordinate is the public constant one,
        // so only its private prefix participates in this product. Slice that
        // prefix once before the matrix multiply instead of creating one
        // source/top slice per row.
        let source_prefix = source
            .clone()
            .slice(None, Some(IndexRange { start: 0.into(), end: (source_dimension - 1).into() }));
        let source_product = source_prefix * top.clone();
        let error = self.sample_error((1, target_columns))?;
        let gadget =
            ring.gadget(source_dimension, self.layout.gadget_base.clone(), self.layout.digit_count);
        let last = payload * (target * gadget) - source_product + error;
        let gsw = Mat::concat(ConcatAxis::Rows, vec![top, last]);

        let mut column_companions = Vec::with_capacity(target_columns);
        for column in 0..target_columns {
            // Restrict the original GSW relation to one target column before
            // decomposing it. The complete tower-major decomposition is then
            // transposed and passed to one packed BGG relation; no Slice node
            // consumes the GadgetDecompose output.
            let decomposed_column = gsw
                .clone()
                .slice(None, Some(IndexRange { start: column.into(), end: (column + 1).into() }))
                .decompose(self.layout.gadget_base.clone(), self.layout.digit_count)
                .as_mat();
            let encodings = self.sample_packed_encodings(
                source.clone(),
                &public_keys.by_column[column],
                &decomposed_column.transpose(),
            )?;
            column_companions.push(encodings.into_iter().skip(1).collect::<Vec<_>>());
        }
        // The package ABI is source-row/target-column/digit major. Sampling
        // above is target-column major so each target column can be sliced
        // and decomposed independently. Reorder only the already packed BGG
        // wires, never the decomposition expression itself.
        let mut companions = Vec::with_capacity(source_dimension * target_columns);
        for row in 0..source_dimension {
            for column in 0..target_columns {
                let start = row * self.layout.digit_count;
                let end = start + self.layout.digit_count;
                let limbs = &column_companions[column][start..end];
                let vector = Mat::concat(
                    ConcatAxis::Columns,
                    limbs.iter().map(|limb| limb.vector.clone()).collect(),
                );
                let public_matrix = public_keys.companions[row * target_columns + column].clone();
                companions.push(crate::rhs::PowerRhsCompanionBlock { vector, public_matrix });
            }
        }
        debug_assert_eq!(companions.len(), source_dimension * target_columns);
        PowerRhsPackage::new(gsw, companions).map_err(PowerLutSamplingError::from)
    }
}

/// Builds the canonical public hash domain for switch companion matrices.
/// Both setup paths must use this exact sequence so their independently
/// lowered public-key projections are byte-for-byte identical.
fn canonical_switch_companion_tag(root: &HashTag, index: usize) -> HashTag {
    let mut tag = root.clone();
    tag.push("switch");
    tag.push(IntExpr::constant(index));
    tag.push("companions");
    tag
}

/// Builds the canonical public hash domain for an automorphism mask matrix.
fn canonical_mask_tag(root: &HashTag, index: usize) -> HashTag {
    let mut tag = root.clone();
    tag.push("mask");
    tag.push(IntExpr::constant(index));
    tag
}

/// Family-level private RHS material used by a structural
/// [`ProgramGate::OneHot`](crate::program::ProgramGate::OneHot).
///
/// A family element is a complete `PowerRhsPackage`, represented by parallel
/// DSL families for its GSW matrix and packed companion blocks. Each block is
/// one `(source_row, target_column)` relation whose columns contain all
/// tower-major CRT digits. Keeping the components parallel lets the compiler
/// build one reusable family body; it never allocates one graph node per
/// configured bucket, cell, or CRT limb.
#[derive(Clone)]
pub struct EncodingSelectorFamily {
    gsw: Family<Mat>,
    companions: Vec<(Family<Mat>, Family<Mat>)>,
}

impl EncodingSelectorFamily {
    /// Creates a structural family from parallel GSW/vector/public families.
    pub fn new(
        gsw: Family<Mat>,
        companions: Vec<(Family<Mat>, Family<Mat>)>,
    ) -> Result<Self, PowerLutError> {
        if companions.is_empty() ||
            companions.iter().any(|(vector, public)| {
                vector.count() != gsw.count() || public.count() != gsw.count()
            })
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Ok(Self { gsw, companions })
    }

    fn count(&self) -> &mxx_ir_core::IntExpr {
        self.gsw.count()
    }

    /// Returns the canonical flat family order used by structural label
    /// loops: GSW first, then each vector/public companion pair.
    pub(crate) fn flattened(&self) -> Vec<Family<Mat>> {
        let mut flat = Vec::with_capacity(1 + self.companions.len() * 2);
        flat.push(self.gsw.clone());
        for (vector, public) in &self.companions {
            flat.push(vector.clone());
            flat.push(public.clone());
        }
        flat
    }

    /// Rebuilds a selector family from its canonical flat representation.
    /// The arity, family count, and public matrix domain are checked before
    /// the representation becomes usable by a lowering body.
    pub(crate) fn from_flattened(flat: Vec<Family<Mat>>) -> Result<Self, PowerLutError> {
        if flat.len() < 3 || flat.len() % 2 == 0 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let gsw = flat[0].clone();
        let gsw_type = gsw.element_type();
        if flat.iter().any(|family| {
            family.count() != gsw.count() ||
                family.element_type().modulus.canonicalize() != gsw_type.modulus.canonicalize() ||
                family.element_type().ring_dimension.canonicalize() !=
                    gsw_type.ring_dimension.canonicalize()
        }) {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let companions =
            flat[1..].chunks_exact(2).map(|pair| (pair[0].clone(), pair[1].clone())).collect();
        Self::new(gsw, companions)
    }
}

impl PowerLutEncodingCompiler {
    /// Creates a compiler from a fully configured generic BGG compiler.
    pub fn new(bgg: BggEncodingCompiler) -> Self {
        Self { bgg }
    }

    /// Creates a compiler from public BGG parameters at a setup boundary.
    pub fn from_public_key(public_key: BggPublicKeyCompiler) -> Self {
        Self::new(BggEncodingCompiler { public_key })
    }

    /// Lowers a validated program using plain BGG wires and explicit private
    /// RHS inputs. One-hot selector families and their public weighting values
    /// are explicit runtime bindings. The returned map contains every program
    /// wire.
    pub fn compile_program(
        &self,
        program: &PowerLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggEncodingWire>,
        rhs_inputs: &BTreeMap<RhsInputId, PowerRhsPackage>,
        one_hot_selectors: &BTreeMap<crate::program::RhsFamilyId, EncodingSelectorFamily>,
        public_values: &BTreeMap<crate::program::PublicValueFamilyId, Family<Mat>>,
        helpers: &[AutomorphismHelper],
    ) -> Result<BTreeMap<ProgramWireId, BggEncodingWire>, PowerLutError> {
        let mut ranges = ProgramFamilyRanges::new();
        for (id, family) in one_hot_selectors {
            let range = FamilyRange::full(family.count().clone())
                .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
            ranges.selector(*id, range);
        }
        for (id, family) in public_values {
            let range = FamilyRange::full(family.count().clone())
                .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
            ranges.public_values(*id, range);
        }
        self.compile_program_with_ranges(
            program,
            inputs,
            rhs_inputs,
            one_hot_selectors,
            public_values,
            &ranges,
            helpers,
        )
    }

    /// Lowers a program with explicit contiguous views into flattened
    /// selector/value families. A PBC bucket can bind only its own range,
    /// while the same structural body remains reusable for every bucket.
    pub fn compile_program_with_ranges(
        &self,
        program: &PowerLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggEncodingWire>,
        rhs_inputs: &BTreeMap<RhsInputId, PowerRhsPackage>,
        one_hot_selectors: &BTreeMap<crate::program::RhsFamilyId, EncodingSelectorFamily>,
        public_values: &BTreeMap<crate::program::PublicValueFamilyId, Family<Mat>>,
        family_ranges: &ProgramFamilyRanges,
        helpers: &[AutomorphismHelper],
    ) -> Result<BTreeMap<ProgramWireId, BggEncodingWire>, PowerLutError> {
        for input in inputs.values() {
            crate::ensure_ciphertext_only(input)?;
        }
        let bindings =
            ProgramBindings::new(inputs, rhs_inputs, one_hot_selectors, public_values, helpers);
        lower_program(program, &bindings, family_ranges, self)
    }
}

/// Returns the row-major exponent index `u + lhs_width * v` used by a
/// rectangular flattened LUT.
pub fn flattened_lut_index(
    u: usize,
    v: usize,
    lhs_width: usize,
    rhs_width: usize,
) -> Option<usize> {
    (lhs_width > 0 && rhs_width > 0 && u < lhs_width && v < rhs_width).then(|| u + lhs_width * v)
}

impl PowerLutEncodingCompiler {
    /// Performs ciphertext-assisted multiplication with a typed RHS package.
    ///
    /// The program/lowering caller is responsible for binding an RHS package
    /// to the corresponding input. Secret identities stored in an imported
    /// package are checked at import time, not carried by this wire value.
    pub fn fuse(
        &self,
        lhs: &BggEncodingWire,
        rhs: &PowerRhsPackage,
    ) -> Result<BggEncodingWire, PowerLutError> {
        crate::ensure_ciphertext_only(lhs)?;
        let digits = self
            .bgg
            .public_key
            .digit_count
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|v| v.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let base = self.bgg.public_key.base.clone();
        let lhs_decomp = lhs.pubkey.matrix.clone().decompose(base.clone(), digits).as_mat();
        let source_dimension = lhs
            .pubkey
            .matrix
            .matrix_type()
            .rows
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let target_columns = rhs
            .gsw_ciphertext()
            .matrix_type()
            .columns
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let vector = crate::utils::fuse_columns(
            Some(&lhs.vector),
            &lhs_decomp,
            Some(rhs.gsw_ciphertext()),
            source_dimension,
            target_columns,
            digits,
            &self.bgg.public_key.ring,
            &base,
            |row, column| rhs.companion_block(row, column, target_columns),
        )?;
        let public_matrix =
            crate::public_key::PowerLutPublicKeyCompiler::new(self.bgg.public_key.clone())
                .fuse_public_with_decomposition(
                    &lhs.pubkey.matrix,
                    &lhs_decomp,
                    &rhs.public_projection(),
                )?;
        Ok(BggEncodingWire {
            vector,
            pubkey: mxx_bgg::BggPublicKeyWire { matrix: public_matrix, reveal_plaintext: false },
            plaintext: None,
        })
    }

    /// Applies a setup-time automorphism helper and returns an encoding under
    /// the original secret. The helper's switch and mask are reused, but their
    /// identities and matrix shapes are checked against this input first.
    pub fn automorphism(
        &self,
        input: &BggEncodingWire,
        helper: &AutomorphismHelper,
    ) -> Result<BggEncodingWire, PowerLutError> {
        crate::ensure_ciphertext_only(input)?;
        let raw = BggEncodingWire {
            vector: input.vector.clone().ring_automorphism(helper.index()),
            pubkey: mxx_bgg::BggPublicKeyWire {
                matrix: input.pubkey.matrix.clone().ring_automorphism(helper.index()),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let switched = self.fuse(&raw, helper.switch())?;
        let decomposition = switched
            .pubkey
            .matrix
            .clone()
            .decompose(self.bgg.public_key.base.clone(), self.bgg.public_key.digit_count.clone())
            .as_mat();
        let mask = helper.mask();
        let vector = mask.vector.clone() * decomposition.clone() + switched.vector.clone();
        // The switched vector is added to cancel the transformed-secret term,
        // but its public matrix is not part of the resulting BGG key.  With
        // `h = s*D - t*G` and `r = t*B - mu*s*G`, the output is
        // `h*D(B) + r = s*(D*D(B)) - mu*s*G`; the `t*B` term is consumed by
        // cancellation.  Keeping `+B` in the public matrix would make the
        // declared key one extra `s*B` larger than its vector relation.
        let public_matrix = mask.pubkey.matrix.clone() * decomposition;
        Ok(BggEncodingWire {
            vector,
            pubkey: mxx_bgg::BggPublicKeyWire { matrix: public_matrix, reveal_plaintext: false },
            plaintext: None,
        })
    }

    /// Keeps only coefficients in the zero residue class modulo `width`.
    ///
    /// The helper at round `i` is the automorphism
    /// `r_i = 2n / 2^(i + 1) + 1`. Adding the transformed state cancels the
    /// odd quotient residue classes and doubles the survivors. After the
    /// required `log2(width)` rounds, only positions divisible by `width`
    /// remain, with amplitude `width`. This method intentionally returns that
    /// unnormalised result; [`Self::single_input_lut`] sums all branches first
    /// and applies one final gadget normalization.
    ///
    /// `helpers` must contain the prevalidated automorphism rounds in exactly
    /// the order above. Reordering them changes the sieve and is rejected.
    pub fn clear_coeff(
        &self,
        input: &BggEncodingWire,
        width: usize,
        helpers: &[AutomorphismHelper],
    ) -> Result<BggEncodingWire, PowerLutError> {
        crate::ensure_ciphertext_only(input)?;
        let n = input
            .pubkey
            .matrix
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|v| v.to_usize())
            .ok_or(PowerLutError::InvalidClearCoeffWidth)?;
        if width == 0 ||
            !width.is_power_of_two() ||
            width > n ||
            n % width != 0 ||
            helpers.len() != width.trailing_zeros() as usize
        {
            return Err(PowerLutError::InvalidClearCoeffWidth);
        }
        let mut state = input.clone();
        for (round, helper) in helpers.iter().enumerate() {
            let expected = (2 * n / (1usize << (round + 1))) + 1;
            if helper.index() != expected {
                return Err(PowerLutError::InvalidAutomorphismHelper);
            }
            let transformed = self.automorphism(&state, helper)?;
            state = self.bgg.add(&state, &transformed)?;
        }
        Ok(state)
    }

    /// Evaluates a public single-input LUT over exponent-encoded values.
    ///
    /// Each candidate is sieved without normalization, all rotated branches
    /// are added, and one final `width^-1` gadget product normalizes the sum.
    pub fn single_input_lut(
        &self,
        input: &BggEncodingWire,
        table: &[usize],
        helpers: &[AutomorphismHelper],
    ) -> Result<BggEncodingWire, PowerLutError> {
        crate::ensure_ciphertext_only(input)?;
        let width = table.len();
        if width == 0 || !width.is_power_of_two() {
            return Err(PowerLutError::InvalidLut);
        }
        let n = input
            .pubkey
            .matrix
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|v| v.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let ring = self.bgg.public_key.ring.clone();
        let shifts = Family::pack(
            (0..width)
                .map(|candidate| {
                    crate::utils::rotation_power(&ring, (2 * n - candidate % (2 * n)) % (2 * n), n)
                })
                .collect(),
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let output_rotations = Family::pack(
            table
                .iter()
                .copied()
                .map(|output| crate::utils::rotation_power(&ring, output, n))
                .collect(),
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let input = input.clone();
        let input_for_loop = input.clone();
        let helpers = helpers.to_vec();
        let (branch_vectors, branch_publics) = Family::try_parallel_zip_many_values(
            vec![shifts, output_rotations],
            move |_index, items| {
                let mut items = items.into_iter();
                let shift = items.next().ok_or(DslError::Schema)?;
                let output_rotation = items.next().ok_or(DslError::Schema)?;
                let shifted = self.bgg.small_scalar_mul(&input_for_loop, &shift);
                let selected =
                    self.clear_coeff(&shifted, width, &helpers).map_err(|_| DslError::Schema)?;
                let branch = self.bgg.small_scalar_mul(&selected, &output_rotation);
                Ok((branch.vector, branch.pubkey.matrix))
            },
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let sum = BggEncodingWire {
            vector: balanced_sum_family(branch_vectors)?,
            pubkey: BggPublicKeyWire {
                matrix: balanced_sum_family(branch_publics)?,
                reveal_plaintext: false,
            },
            plaintext: None,
        };

        // ClearCoeff leaves a factor `width` on every surviving coefficient.
        // Conceptually normalize once with `G^-1(width^-1 G)`: the explicit
        // `G` preserves the canonical payload because
        // `G G^-1(width^-1 G) = width^-1 G`. The BGG large-scalar helper
        // inserts that gadget internally, so the scalar passed below is only
        // `width^-1`; this emits exactly one gadget decomposition and avoids
        // direct large-inverse amplification.
        let modulus = input
            .pubkey
            .matrix
            .matrix_type()
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| PowerLutError::InvalidLut)?;
        let inverse = modular_inverse(&(BigInt::from(width) % &modulus), &modulus)
            .ok_or(PowerLutError::InvalidLut)?;
        let ring = self.bgg.public_key.ring.clone();
        let inverse_scalar = ring.polynomial([inverse.into()]);
        Ok(self.bgg.large_scalar_mul(&sum, &inverse_scalar))
    }

    /// Fuses a private RHS monomial and evaluates a flattened two-input LUT.
    /// The RHS exponent is `u + lhs_width*v`; no overlapping exponent encoding
    /// is used for rectangular tables.
    pub fn two_input_lut(
        &self,
        lhs: &BggEncodingWire,
        rhs: &PowerRhsPackage,
        lhs_width: usize,
        rhs_width: usize,
        table: &[usize],
        helpers: &[AutomorphismHelper],
    ) -> Result<BggEncodingWire, PowerLutError> {
        crate::ensure_ciphertext_only(lhs)?;
        if lhs_width == 0 ||
            rhs_width == 0 ||
            !lhs_width.is_power_of_two() ||
            !rhs_width.is_power_of_two() ||
            table.len() != lhs_width.checked_mul(rhs_width).ok_or(PowerLutError::InvalidLut)?
        {
            return Err(PowerLutError::InvalidLut);
        }
        let fused = self.fuse(lhs, rhs)?;
        self.single_input_lut(&fused, table, helpers)
    }
}

impl ProgramLoweringBackend for PowerLutEncodingCompiler {
    type Wire = BggEncodingWire;
    type Rhs = PowerRhsPackage;
    type SelectorFamily = EncodingSelectorFamily;
    type PublicValueFamily = Family<Mat>;
    type Helper = AutomorphismHelper;

    fn unary(
        &self,
        input: Self::Wire,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError> {
        self.single_input_lut(&input, table.values(), helpers)
    }

    fn binary(
        &self,
        lhs: Self::Wire,
        rhs: &Self::Rhs,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError> {
        self.two_input_lut(
            &lhs,
            rhs,
            table.input_width(),
            table.rhs_width().expect("shared traversal validates binary LUT"),
            table.values(),
            helpers,
        )
    }

    fn one_hot(
        &self,
        lhs: Self::Wire,
        selectors: &Self::SelectorFamily,
        public_values: &Self::PublicValueFamily,
        selector_range: &FamilyRange,
        public_value_range: &FamilyRange,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError> {
        if selectors.count() != public_values.count() ||
            selector_range != public_value_range ||
            !selector_range.is_within(selectors.count())
        {
            return Err(crate::program::ProgramValidationError::WidthMismatch.into());
        }

        // Use a fixed structural loop capacity. The logical range may be
        // shorter for a sparse bucket, but its inactive tail is masked before
        // weighting. This keeps all indexed-family wire types independent of
        // the enclosing bucket-loop binder and excludes padding RHS material.
        let capacity = selector_range
            .capacity()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let count = mxx_dsl::Int::evaluate(selector_range.count().clone());
        let start = mxx_dsl::Int::evaluate(selector_range.start().clone());
        let mask_type = public_values.element_type().clone();
        let (safe_indices, active_masks) =
            one_hot_indices_and_masks(capacity, count, start, mask_type)
                .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let gsw = selectors
            .gsw
            .clone()
            .parallel_gather(safe_indices.clone())
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let companions = selectors
            .companions
            .iter()
            .map(|(vector, public)| {
                Ok((
                    vector
                        .clone()
                        .parallel_gather(safe_indices.clone())
                        .map_err(|_| DslError::Schema)?,
                    public
                        .clone()
                        .parallel_gather(safe_indices.clone())
                        .map_err(|_| DslError::Schema)?,
                ))
            })
            .collect::<Result<Vec<_>, DslError>>()
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let values = public_values
            .clone()
            .parallel_gather(safe_indices)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let lhs_vector = lhs.vector.clone();
        let lhs_matrix = lhs.pubkey.matrix.clone();
        let compiler = Self { bgg: self.bgg.clone() };
        let mut zipped = vec![gsw];
        for (vector, public) in companions {
            zipped.push(vector);
            zipped.push(public);
        }
        zipped.push(values);
        zipped.push(active_masks);
        let weighted =
            Family::<Mat>::try_parallel_zip_many_values(zipped, move |_index, mut zipped| {
                let active = zipped.pop().ok_or(DslError::Schema)?;
                let value = zipped.pop().ok_or(DslError::Schema)?;
                let gsw = zipped.remove(0);
                let mut rhs_companions = Vec::with_capacity(zipped.len() / 2);
                for pair in zipped.chunks_exact(2) {
                    rhs_companions.push(crate::rhs::PowerRhsCompanionBlock {
                        vector: pair[0].clone(),
                        public_matrix: pair[1].clone(),
                    });
                }
                let lhs = BggEncodingWire {
                    vector: lhs_vector.clone(),
                    pubkey: BggPublicKeyWire {
                        matrix: lhs_matrix.clone(),
                        reveal_plaintext: false,
                    },
                    plaintext: None,
                };
                let rhs =
                    PowerRhsPackage::new(gsw, rhs_companions).map_err(|_| DslError::Schema)?;
                let fused = compiler.fuse(&lhs, &rhs).map_err(|_| DslError::Schema)?;
                let weighted = compiler.bgg.small_scalar_mul(&fused, &(value * active));
                Ok((weighted.vector, weighted.pubkey.matrix))
            })
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;

        // Reduce the fixed-capacity weighted family with the existing static
        // balanced reduction. Inactive terms are zero, so the result equals
        // the requested logical range without any parent-dependent family.
        let (vector, public) = (balanced_sum_family(weighted.0)?, balanced_sum_family(weighted.1)?);
        let selected = BggEncodingWire {
            vector,
            pubkey: mxx_bgg::BggPublicKeyWire { matrix: public, reveal_plaintext: false },
            plaintext: None,
        };
        self.single_input_lut(&selected, table.values(), helpers)
    }
}

/// Builds fixed-capacity one-hot indices and masks for a logical family range.
///
/// Every lane gathers from the logical range, including inactive capacity
/// lanes, and the corresponding mask removes those inactive contributions.
pub(crate) fn one_hot_indices_and_masks(
    capacity: usize,
    count: mxx_dsl::Int,
    start: mxx_dsl::Int,
    mask_type: mxx_ir_core::types::MatrixType,
) -> Result<(Family<mxx_dsl::Int>, Family<Mat>), DslError> {
    Parallel::range(capacity).try_map_values(|index| {
        let offset = index.as_int();
        let active = offset.clone().less_equal(count.clone().sub(mxx_dsl::Int::constant(1)));
        let selected = offset.rem(count.clone());
        Ok((
            start.clone().add(selected),
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

/// Sums a positive family through logarithmically many structural
/// parallel rounds. The family elements are never host-unrolled into one
/// graph expression per LUT branch; each round contains one reusable loop
/// body that adds adjacent elements.
pub(crate) fn balanced_sum_family(family: Family<Mat>) -> Result<Mat, PowerLutError> {
    let mut count = family
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(PowerLutError::InvalidLut)?;
    if count == 0 {
        return Err(PowerLutError::InvalidLut);
    }
    let mut current = family;
    while count > 1 {
        let source = current.clone();
        let next_count = count.div_ceil(2);
        let odd = count % 2 == 1;
        let last_pair = mxx_dsl::Int::constant(next_count - 1);
        let left_indices = Parallel::range(next_count)
            .map_values(|index| index.as_int().mul(mxx_dsl::Int::constant(2)))
            .map_err(|_| PowerLutError::InvalidLut)?;
        let right_indices = Parallel::range(next_count)
            .map_values(|index| {
                let index_value = index.as_int();
                let candidate = index_value
                    .clone()
                    .mul(mxx_dsl::Int::constant(2))
                    .add(mxx_dsl::Int::constant(1));
                if odd { candidate.rem(mxx_dsl::Int::constant(count)) } else { candidate }
            })
            .map_err(|_| PowerLutError::InvalidLut)?;
        let left =
            source.clone().parallel_gather(left_indices).map_err(|_| PowerLutError::InvalidLut)?;
        let right = source.parallel_gather(right_indices).map_err(|_| PowerLutError::InvalidLut)?;
        current =
            Family::try_parallel_zip_many_values(vec![left, right], move |index, mut items| {
                let left = items.remove(0);
                let right = items.remove(0);
                if odd {
                    let odd_last = index.as_int().equal(last_pair.clone()).to_int();
                    odd_last.select(vec![left.clone() + right, left])
                } else {
                    Ok(left + right)
                }
            })
            .map_err(|_| PowerLutError::InvalidLut)?;
        count = next_count;
    }
    Ok(current.get_static(0))
}

// -------------------------------------------------------------------------
// Artifact import boundary
// -------------------------------------------------------------------------

/// Names of the private vector and public matrix artifacts for one encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggEncodingArtifactNames {
    /// Artifact containing the private encoding vector.
    pub vector: String,
    /// Artifact containing the public key matrix paired with the vector.
    pub public_matrix: String,
}

#[derive(Debug, Error, Eq, PartialEq)]
/// Reasons why a Power-LUT encoding or package cannot be imported.
pub enum PowerArtifactImportError {
    #[error("artifact manifest production does not match the requested production")]
    /// Manifest belongs to another production.
    ProductionMismatch,
    #[error("required Power-LUT artifact is missing")]
    /// A named artifact is absent from the manifest.
    MissingArtifact,
    #[error("Power-LUT artifact has the wrong confidentiality")]
    /// Public/private confidentiality does not match the expected role.
    ConfidentialityMismatch,
    #[error("Power-LUT artifact has the wrong matrix type")]
    /// Artifact matrix dimensions or modulus are incompatible.
    MatrixTypeMismatch,
    #[error("Power-LUT artifact is missing canonical provenance metadata")]
    /// Required serialized provenance metadata is absent or malformed.
    InvalidMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestEncodingMetadata {
    secret: ManifestSecretMetadata,
    role: serde_json::Value,
}

/// Imports a state encoding after validating its manifest metadata and shapes.
pub fn artifact_input(
    production_id: mxx_ir_core::artifact::ProductionId,
    manifest: &mxx_ir_core::artifact::Manifest,
    names: BggEncodingArtifactNames,
) -> Result<BggEncodingWire, PowerArtifactImportError> {
    artifact_input_with_role::<serde_json::Value>(production_id, manifest, names, None)
}

pub(crate) fn artifact_input_with_role<R: Serialize>(
    production_id: mxx_ir_core::artifact::ProductionId,
    manifest: &mxx_ir_core::artifact::Manifest,
    names: BggEncodingArtifactNames,
    expected_role: Option<&R>,
) -> Result<BggEncodingWire, PowerArtifactImportError> {
    artifact_input_with_columns(production_id, manifest, names, expected_role, None)
}

/// Imports an encoding whose columns are a packed sequence of ordinary BGG
/// columns. The packed companion relation uses this to carry all CRT digits
/// in one artifact while retaining the same secret/layout metadata checks as
/// an ordinary encoding.
pub(crate) fn artifact_input_with_columns<R: Serialize>(
    production_id: mxx_ir_core::artifact::ProductionId,
    manifest: &mxx_ir_core::artifact::Manifest,
    names: BggEncodingArtifactNames,
    expected_role: Option<&R>,
    columns: Option<usize>,
) -> Result<BggEncodingWire, PowerArtifactImportError> {
    if manifest.production_id != production_id {
        return Err(PowerArtifactImportError::ProductionMismatch);
    }
    if names.vector == names.public_matrix {
        return Err(PowerArtifactImportError::InvalidMetadata);
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
        .is_some_and(|expected_role| {
            metadata.role != expected_role || public_metadata.role != expected_role
        }) ||
        metadata.secret.identity != public_metadata.secret.identity ||
        metadata.secret.modulus != public_metadata.secret.modulus ||
        metadata.secret.ring_dimension != public_metadata.secret.ring_dimension ||
        metadata.secret.secret_dimension != public_metadata.secret.secret_dimension ||
        metadata.secret.digit_count != public_metadata.secret.digit_count ||
        metadata.secret.gadget_base != public_metadata.secret.gadget_base
    {
        return Err(PowerArtifactImportError::InvalidMetadata);
    }
    let layout = metadata.secret.sampler();
    let columns = columns.unwrap_or_else(|| layout.public_key_columns());
    let modulus = layout
        .modulus
        .evaluate(&Default::default())
        .map_err(|_| PowerArtifactImportError::MatrixTypeMismatch)?;
    let ring_dimension = layout
        .ring_dimension
        .evaluate(&Default::default())
        .map_err(|_| PowerArtifactImportError::MatrixTypeMismatch)?
        .to_usize()
        .ok_or(PowerArtifactImportError::MatrixTypeMismatch)?;
    let vector_type =
        mxx_ir_core::artifact::ArtifactType::Matrix(mxx_ir_core::types::ConcreteMatrixType {
            modulus: modulus.clone(),
            ring_dimension: ring_dimension.clone(),
            rows: 1,
            columns,
        });
    let public_type =
        mxx_ir_core::artifact::ArtifactType::Matrix(mxx_ir_core::types::ConcreteMatrixType {
            modulus,
            ring_dimension,
            rows: layout.secret_dimension,
            columns,
        });
    if vector.confidentiality != mxx_ir_core::artifact::ArtifactConfidentiality::Private ||
        public.confidentiality != mxx_ir_core::artifact::ArtifactConfidentiality::Public ||
        vector.family_count.is_some() ||
        public.family_count.is_some() ||
        vector.artifact_type != vector_type ||
        public.artifact_type != public_type
    {
        return Err(PowerArtifactImportError::MatrixTypeMismatch);
    }
    let ring = layout.ring();
    Ok(BggEncodingWire {
        vector: ring.artifact_input(
            production_id.clone(),
            names.vector,
            (1, columns),
            mxx_ir_core::artifact::ArtifactConfidentiality::Private,
        ),
        pubkey: mxx_bgg::BggPublicKeyWire {
            matrix: ring.artifact_input(
                production_id,
                names.public_matrix,
                (layout.secret_dimension, columns),
                mxx_ir_core::artifact::ArtifactConfidentiality::Public,
            ),
            reveal_plaintext: false,
        },
        plaintext: None,
    })
}

/// Serializes canonical provenance metadata for an encoding artifact.
#[cfg(test)]
pub(crate) fn power_encoding_artifact_layout<R: Serialize>(
    sampler: &mxx_bgg::BggSamplerLayout,
    identity: [u8; 32],
    role: R,
) -> String {
    serde_json::to_string(&ManifestEncodingMetadata {
        secret: ManifestSecretMetadata {
            modulus: sampler.modulus.clone(),
            ring_dimension: sampler.ring_dimension.clone(),
            secret_dimension: sampler.secret_dimension,
            digit_count: sampler.digit_count,
            gadget_base: sampler.gadget_base.clone(),
            identity,
        },
        role: serde_json::to_value(role).expect("Power-LUT encoding role serialization"),
    })
    .expect("Power-LUT encoding metadata serialization")
}
#[cfg(test)]
mod tests {
    use super::{AutomorphismHelper, EncodingSelectorFamily, flattened_lut_index, modular_inverse};
    use crate::{
        PowerLutEncodingCompiler, PowerLutError, PowerLutPublicKeyCompiler,
        program::{FamilyRange, ProgramFamilyRanges},
        public_key::PowerLutPublicKeySampler,
        rhs::{PowerRhsCompanionBlock, PowerRhsPackage},
    };
    use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{ParamEnv, node::NodeKind, types::WireType};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use num_traits::ToPrimitive;
    use serial_test::serial;
    use std::collections::BTreeMap;

    #[test]
    fn clear_coeff_scale_uses_an_exact_modular_inverse() {
        let modulus = BigInt::from(257u32);
        assert_eq!(modular_inverse(&BigInt::from(4u32), &modulus), Some(BigInt::from(193u32)));
        assert_eq!(modular_inverse(&BigInt::from(0u32), &modulus), None);
    }

    #[test]
    fn flattened_two_input_index_is_row_major_for_all_power_of_two_widths() {
        for width in [2usize, 4, 8] {
            for u in 0..width {
                for v in 0..width {
                    assert_eq!(flattened_lut_index(u, v, width, width), Some(u + width * v));
                }
            }
            assert_eq!(flattened_lut_index(width, 0, width, width), None);
        }
    }

    struct RuntimeFixture {
        parameters: DCRTPolyParams,
        ring: Ring,
        compiler: PowerLutEncodingCompiler,
    }

    impl RuntimeFixture {
        fn new() -> Self {
            Self::with_dimension(4)
        }

        fn with_dimension(ring_dimension: u32) -> Self {
            let parameters = DCRTPolyParams::new(ring_dimension, 1, 17, 17);
            let modulus = BigInt::from(parameters.modulus().as_ref().clone());
            let ring = Ring::new(modulus.clone(), ring_dimension as usize);
            let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 131_072.into(),
                digit_count: 1.into(),
            });
            Self { parameters, ring, compiler }
        }

        fn bound(&self, vector: mxx_dsl::Mat, public: mxx_dsl::Mat) -> BggEncodingWire {
            BggEncodingWire {
                vector,
                pubkey: BggPublicKeyWire { matrix: public, reveal_plaintext: false },
                plaintext: None,
            }
        }

        fn helper(&self, index: usize) -> AutomorphismHelper {
            // Nonzero companion/mask vectors exercise the RHS and mask paths;
            // zero public projections keep the expected value a signed
            // automorphism, making the permutation observable directly.
            let zero = self.ring.zero((1, 1));
            let nonzero = self.ring.polynomial([1.into()]);
            let switch = PowerRhsPackage::new(
                self.ring.polynomial([1.into()]),
                vec![crate::rhs::PowerRhsCompanionBlock {
                    vector: nonzero.clone(),
                    public_matrix: zero.clone(),
                }],
            )
            .unwrap();
            AutomorphismHelper::new(index, switch, self.bound(nonzero, zero)).unwrap()
        }

        fn run(&self, name: &str, value: mxx_dsl::Mat) -> DCRTPolyMatrix {
            let graph = DslContext::new(name).output("result", value).unwrap().build().unwrap();
            let graph = graph
                .validate(&ParamEnv::default())
                .unwrap_or_else(|error| panic!("{name}: {error:?}"));
            let result = execute(
                &graph,
                &mut cpu_backend([self.parameters.clone()]),
                std::collections::BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            let RuntimeValue::Matrix(value) = &result.outputs["result"] else { panic!("matrix") };
            value.as_ref().clone()
        }
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn concrete_clear_coeff_and_single_lut_match_authoritative_polynomial_values() {
        let fixture = RuntimeFixture::new();
        let input = fixture.bound(
            fixture.ring.polynomial([1.into(), 2.into(), 3.into(), 4.into()]),
            fixture.ring.zero((1, 1)),
        );
        for (width, indices, table) in [
            (2usize, vec![5usize], vec![0usize, 1]),
            (4usize, vec![5usize, 3], vec![0usize, 1, 2, 3]),
        ] {
            let helpers =
                indices.iter().copied().map(|index| fixture.helper(index)).collect::<Vec<_>>();
            let cleared = fixture.compiler.clear_coeff(&input, width, &helpers).unwrap();
            let actual = fixture.run("power-lut-clear-coeff-runtime", cleared.vector.clone());
            let mut expected = DCRTPolyMatrix::from_poly_vec_row(
                &fixture.parameters,
                vec![DCRTPoly::from_u32s(&fixture.parameters, &[1, 2, 3, 4])],
            );
            for index in indices {
                let transformed = expected.ring_automorphism_out_of_place(index);
                expected = expected + transformed;
            }
            assert_eq!(actual, expected, "unnormalised ClearCoeff width {width}");
            let lut_input =
                fixture.bound(fixture.ring.polynomial([1.into()]), fixture.ring.zero((1, 1)));
            let lut = fixture.compiler.single_input_lut(&lut_input, &table, &helpers).unwrap();
            let actual = fixture.run("power-lut-single-lut-runtime", lut.vector.clone());
            let expected = DCRTPolyMatrix::from_poly_vec_row(
                &fixture.parameters,
                vec![DCRTPoly::const_rotate_poly(&fixture.parameters, table[0])],
            );
            assert_eq!(actual, expected, "single-input LUT width {width}");
        }
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn fixed_secret_automorphism_alignment_matches_public_projection_and_runtime_value() {
        let fixture = RuntimeFixture::new();
        let input = fixture.bound(
            fixture.ring.polynomial([1.into(), 2.into(), 3.into(), 4.into()]),
            fixture.ring.zero((1, 1)),
        );
        let helper = fixture.helper(5);
        let aligned = fixture.compiler.automorphism(&input, &helper).unwrap();
        let public_helper = crate::public_key::AutomorphismPublicHelper::new(
            helper.index(),
            helper.switch().public_projection(),
            helper.mask().pubkey.matrix.clone(),
        );
        let public = PowerLutPublicKeyCompiler::new(fixture.compiler.bgg.public_key.clone())
            .automorphism(&input.pubkey.matrix, &public_helper)
            .unwrap();
        let graph = DslContext::new("power-lut-automorphism-runtime")
            .output("vector", aligned.vector.clone())
            .unwrap()
            .output("public-derived", aligned.pubkey.matrix.clone())
            .unwrap()
            .output("public-only", public)
            .unwrap()
            .build()
            .unwrap();
        let graph = graph.validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([fixture.parameters.clone()]),
            std::collections::BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(vector) = &result.outputs["vector"] else { panic!("matrix") };
        let expected = DCRTPolyMatrix::from_poly_vec_row(
            &fixture.parameters,
            vec![DCRTPoly::from_u32s(&fixture.parameters, &[1, 2, 3, 4])],
        )
        .ring_automorphism_out_of_place(5);
        assert_eq!(vector.as_ref(), &expected);
        let RuntimeValue::Matrix(public_derived) = &result.outputs["public-derived"] else {
            panic!("matrix")
        };
        let RuntimeValue::Matrix(public_only) = &result.outputs["public-only"] else {
            panic!("matrix")
        };
        assert_eq!(public_derived, public_only, "public automorphism projection diverged");
    }

    #[test]
    fn structural_one_hot_graph_does_not_unroll_family_elements() {
        fn graph_node_count(family_count: usize) -> usize {
            let ring = mxx_dsl::Ring::new(257, 4);
            let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 2.into(),
                digit_count: 1.into(),
            });
            let mut builder = crate::program::PowerLutProgramBuilder::new();
            let input = builder.input(1).unwrap();
            let selector_family = builder.rhs_family(1).unwrap();
            let public_value_family = builder.public_value_family(1).unwrap();
            let lut = builder.lut(crate::program::LutTable::unary(1, 1, vec![0]).unwrap()).unwrap();
            let selected = builder
                .one_hot(
                    builder.input_wire(input).unwrap(),
                    selector_family,
                    public_value_family,
                    lut,
                )
                .unwrap();
            builder.output(selected).unwrap();
            let program = builder.build().unwrap();

            let gsw = ring.input_family("structural-gsw", family_count, (1, 1));
            let vectors = ring.input_family("structural-vectors", family_count, (1, 1));
            let publics = ring.input_family("structural-publics", family_count, (1, 1));
            let selectors = EncodingSelectorFamily::new(gsw, vec![(vectors, publics)]).unwrap();
            let values = ring.input_family("structural-values", family_count, (1, 1));
            let mut ranges = ProgramFamilyRanges::new();
            let range = FamilyRange::full(family_count).unwrap();
            ranges.selector(selector_family, range.clone());
            ranges.public_values(public_value_family, range);
            let lhs = BggEncodingWire {
                vector: ring.input("structural-lhs-vector", (1, 1)),
                pubkey: BggPublicKeyWire {
                    matrix: ring.input("structural-lhs-public", (1, 1)),
                    reveal_plaintext: false,
                },
                plaintext: None,
            };
            let wires = compiler
                .compile_program_with_ranges(
                    &program,
                    &BTreeMap::from([(input, lhs)]),
                    &BTreeMap::new(),
                    &BTreeMap::from([(selector_family, selectors)]),
                    &BTreeMap::from([(public_value_family, values)]),
                    &ranges,
                    &[],
                )
                .unwrap();
            let output = wires.get(&selected).unwrap();
            DslContext::new(format!("power-lut-structural-one-hot-{family_count}"))
                .output("vector", output.vector.clone())
                .unwrap()
                .build()
                .unwrap()
                .graph
                .root_scope()
                .nodes()
                .len()
        }

        let small = graph_node_count(1);
        for count in [2, 3, 5, 6, 8] {
            let nodes = graph_node_count(count);
            assert!(nodes >= small, "structural reduction must build for count {count}");
        }
        let large = graph_node_count(64);
        assert!(
            large < small * 8,
            "family cardinality should add structural rounds, not one graph node per element"
        );
    }

    #[test]
    fn one_hot_nonuniform_bucket_range_gathers_before_fuse() {
        let ring = mxx_dsl::Ring::new(257, 4);
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 1.into(),
        });
        let mut builder = crate::program::PowerLutProgramBuilder::new();
        let input = builder.input(1).unwrap();
        let selector_family = builder.rhs_family(1).unwrap();
        let public_value_family = builder.public_value_family(1).unwrap();
        let lut = builder.lut(crate::program::LutTable::unary(1, 1, vec![0]).unwrap()).unwrap();
        let selected = builder
            .one_hot(builder.input_wire(input).unwrap(), selector_family, public_value_family, lut)
            .unwrap();
        builder.output(selected).unwrap();
        let program = builder.build().unwrap();

        let family_count = 16;
        let production = mxx_ir_core::artifact::ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([1; 32]),
            execution_nonce: [2; 32],
        };
        let gsw = ring.family_artifact_input(
            production.clone(),
            "nonuniform-gsw",
            family_count,
            (1, 1),
            mxx_ir_core::artifact::ArtifactConfidentiality::Private,
        );
        let vectors = ring.family_artifact_input(
            production.clone(),
            "nonuniform-vectors",
            family_count,
            (1, 1),
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
        );
        let publics = ring.family_artifact_input(
            production.clone(),
            "nonuniform-publics",
            family_count,
            (1, 1),
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
        );
        let selectors = EncodingSelectorFamily::new(gsw, vec![(vectors, publics)]).unwrap();
        let values = ring.family_artifact_input(
            production,
            "nonuniform-values",
            family_count,
            (1, 1),
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
        );
        let mut ranges = ProgramFamilyRanges::new();
        let range = FamilyRange::bounded(2usize, 3usize, 8).unwrap();
        ranges.selector(selector_family, range.clone());
        ranges.public_values(public_value_family, range);
        let lhs = BggEncodingWire {
            vector: ring.input("nonuniform-lhs-vector", (1, 1)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("nonuniform-lhs-public", (1, 1)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let wires = compiler
            .compile_program_with_ranges(
                &program,
                &BTreeMap::from([(input, lhs)]),
                &BTreeMap::new(),
                &BTreeMap::from([(selector_family, selectors)]),
                &BTreeMap::from([(public_value_family, values)]),
                &ranges,
                &[],
            )
            .unwrap();
        let graph = DslContext::new("power-lut-nonuniform-encoding-one-hot")
            .output("result", wires[&selected].vector.clone())
            .unwrap()
            .build()
            .unwrap();
        let all_nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        let parallel_loops = graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(_)))
            .count();
        assert!(parallel_loops >= 2, "range gathers and Fuse must remain structural loops");
        assert!(graph.graph.root_scope().nodes().iter().any(|node| matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::ParallelLoop(spec)
                if spec.count == mxx_ir_core::IntExpr::constant(8)
        )));
        assert!(all_nodes.iter().any(|node| matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::IntBinary(mxx_ir_core::node::IntBinaryOp::Remainder)
        )));
        assert!(all_nodes.iter().any(|node| matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::LiftIntegerToConstantPolynomial { .. }
        )));
        let weighted_loop_has_no_singleton_pack = graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                mxx_ir_core::node::NodeKind::ParallelLoop(spec)
                    if spec.count == mxx_ir_core::IntExpr::constant(8) &&
                        spec.input_modes
                            .iter()
                            .position(|mode| {
                                matches!(mode, mxx_ir_core::node::LoopInputMode::Broadcast)
                            })
                            .is_some_and(|first_broadcast| {
                                spec.input_modes[..first_broadcast].iter().all(|mode| {
                                    matches!(mode, mxx_ir_core::node::LoopInputMode::Zip)
                                }) && spec.input_modes[first_broadcast..].iter().all(|mode| {
                                    matches!(mode, mxx_ir_core::node::LoopInputMode::Broadcast)
                                })
                            }) =>
                {
                    graph.graph.root_scope().node_id(node).and_then(|id| {
                        graph
                            .graph
                            .child_scope_id(&mxx_ir_core::graph::FrozenGraphScopeId::Root, id)
                    })
                }
                _ => None,
            })
            .any(|scope_id| {
                graph.graph.scope(&scope_id).unwrap().nodes().iter().all(|node| {
                    !matches!(
                        node.kind(),
                        mxx_ir_core::node::NodeKind::FamilyPack { count }
                            if *count == mxx_ir_core::IntExpr::constant(1)
                    )
                })
            });
        assert!(weighted_loop_has_no_singleton_pack);
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn variable_width_one_hot_indices_wrap_and_mask_inactive_lanes() {
        let parameters = DCRTPolyParams::new(4, 1, 17, 4);
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, 4);
        let (indices, masks) = super::one_hot_indices_and_masks(
            8,
            mxx_dsl::Int::constant(3),
            mxx_dsl::Int::constant(2),
            ring.matrix_type((1, 1)),
        )
        .unwrap();
        let mut context = DslContext::new("power-lut-variable-width-one-hot-indices")
            .int_family_output("indices", indices)
            .unwrap();
        for offset in 0..8 {
            context = context.output(format!("mask-{offset}"), masks.get_static(offset)).unwrap();
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::IndexedFamily(indexes) = &result.outputs["indices"] else {
            panic!("one-hot indices output must be a family")
        };
        let indexes = indexes
            .iter()
            .map(|value| {
                let RuntimeValue::Int(value) = value else { panic!("one-hot index must be int") };
                value.to_usize().unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(indexes, vec![2, 3, 4, 2, 3, 4, 2, 3]);
        assert!(indexes.iter().all(|&index| (2..5).contains(&index)));

        let zero = DCRTPolyMatrix::zero(&parameters, 1, 1);
        for offset in 3..8 {
            let mask = &result.outputs[&format!("mask-{offset}")];
            let RuntimeValue::Matrix(mask) = mask else { panic!("one-hot mask must be matrix") };
            assert_eq!(mask.as_ref(), &zero, "inactive offset {offset} must be masked");
        }
    }

    #[test]
    fn clear_coeff_rejects_helpers_in_the_wrong_round_order() {
        let fixture = RuntimeFixture::new();
        let input = fixture.bound(
            fixture.ring.polynomial([1.into(), 2.into(), 3.into(), 4.into()]),
            fixture.ring.zero((1, 1)),
        );
        let first = fixture.helper(5);
        let second = fixture.helper(3);
        assert!(matches!(
            fixture.compiler.clear_coeff(&input, 4, &[second, first]),
            Err(PowerLutError::InvalidAutomorphismHelper)
        ));
    }

    #[test]
    fn automorphism_helper_rejects_incompatible_mask_shapes() {
        let ring = Ring::new(257, 4);
        let zero = ring.zero((1, 1));
        let switch = PowerRhsPackage::new(
            zero.clone(),
            vec![crate::rhs::PowerRhsCompanionBlock {
                vector: zero.clone(),
                public_matrix: zero.clone(),
            }],
        )
        .unwrap();

        let mismatched_width = BggEncodingWire {
            vector: ring.zero((1, 2)),
            pubkey: BggPublicKeyWire { matrix: ring.zero((1, 1)), reveal_plaintext: false },
            plaintext: None,
        };
        assert!(matches!(
            AutomorphismHelper::new(5, switch.clone(), mismatched_width),
            Err(PowerLutError::InvalidAutomorphismHelper)
        ));

        let mismatched_rows = BggEncodingWire {
            vector: ring.zero((2, 1)),
            pubkey: BggPublicKeyWire { matrix: ring.zero((1, 1)), reveal_plaintext: false },
            plaintext: None,
        };
        assert!(matches!(
            AutomorphismHelper::new(5, switch, mismatched_rows),
            Err(PowerLutError::InvalidAutomorphismHelper)
        ));
    }

    #[test]
    fn sampler_reuses_canonical_helper_indices_and_builds_private_material() {
        let layout = mxx_bgg::BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: 2.into(),
        };
        let ring = layout.ring();
        let sampler = super::PowerLutEncodingSampler {
            layout,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        assert_eq!(sampler.automorphism_helper_indices(4).unwrap(), vec![5, 3]);
        let helpers = sampler
            .sample_automorphism_helpers(
                ring.input("global-secret", (1, 2)),
                ring.bytes_input("public-hash-key", 32),
                &b"helper-sampler"[..],
                4,
            )
            .unwrap();
        assert_eq!(helpers.iter().map(AutomorphismHelper::index).collect::<Vec<_>>(), vec![5, 3]);
        let mut context = DslContext::new("power-lut-helper-sampler-shapes");
        for (round, helper) in helpers.iter().enumerate() {
            context = context
                .output(format!("mask-{round}"), helper.mask().vector.clone())
                .unwrap()
                .output(format!("switch-{round}"), helper.switch().gsw_ciphertext().clone())
                .unwrap();
        }
        context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn private_and_public_samplers_produce_the_same_public_setup_matrices() {
        let parameters = DCRTPolyParams::new(4, 1, 17, 17);
        let layout = mxx_bgg::BggSamplerLayout {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: 2.into(),
        };
        let ring = layout.ring();
        let hash_key = ring.bytes_input("sampler-public-key", 32);
        let secret = ring.input("sampler-secret", (1, 2));
        let encoding_sampler = super::PowerLutEncodingSampler {
            layout: layout.clone(),
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let private_input = encoding_sampler
            .sample_input_encoding(
                secret.clone(),
                hash_key.clone(),
                &b"input-setup"[..],
                ring.polynomial([5.into()]),
            )
            .unwrap();
        let private_helpers = encoding_sampler
            .sample_automorphism_helpers(secret, hash_key.clone(), &b"helper-setup"[..], 4)
            .unwrap();
        assert!(private_input.plaintext.is_none());
        assert!(!private_input.pubkey.reveal_plaintext);
        for helper in &private_helpers {
            assert!(helper.mask().plaintext.is_none());
            assert!(!helper.mask().pubkey.reveal_plaintext);
            for companion in 0..helper.switch().companion_count() {
                assert!(
                    helper.switch().companion(0, companion, layout.public_key_columns()).is_some()
                );
            }
        }
        let public_sampler = PowerLutPublicKeySampler { layout: layout.clone() };
        let public_input =
            public_sampler.sample_input_key(hash_key.clone(), &b"input-setup"[..]).unwrap();
        assert!(!public_input.reveal_plaintext);
        let public_helpers =
            public_sampler.sample_automorphism_helpers(hash_key, &b"helper-setup"[..], 4).unwrap();

        let columns = layout.public_key_columns();
        let mut context = DslContext::new("power-lut-sampler-public-setup-equality")
            .output("private-input", private_input.pubkey.matrix)
            .unwrap()
            .output("public-input", public_input.matrix)
            .unwrap();
        for (round, (private, public)) in
            private_helpers.iter().zip(public_helpers.iter()).enumerate()
        {
            context = context
                .output(format!("private-mask-{round}"), private.mask().pubkey.matrix.clone())
                .unwrap()
                .output(format!("public-mask-{round}"), public.mask().clone())
                .unwrap()
                .output(
                    format!("private-switch-{round}"),
                    private.switch().companion(0, 0, columns).unwrap().public_matrix.clone(),
                )
                .unwrap()
                .output(
                    format!("public-switch-{round}"),
                    public.switch().companion(0, 0, columns).unwrap().clone(),
                )
                .unwrap();
            for row in 0..layout.secret_dimension {
                for column in 0..columns {
                    let private_block = private.switch().companion(row, column, columns).unwrap();
                    let public_block = public.switch().companion(row, column, columns).unwrap();
                    let mut block_context = DslContext::new(format!(
                        "power-lut-sampler-public-setup-block-{round}-{row}-{column}"
                    ));
                    block_context = block_context
                        .output("private", private_block.public_matrix.clone())
                        .unwrap()
                        .output("public", public_block.clone())
                        .unwrap();
                    let block_graph =
                        block_context.build().unwrap().validate(&ParamEnv::default()).unwrap();
                    let block_result = execute(
                        &block_graph,
                        &mut cpu_backend([parameters.clone()]),
                        BTreeMap::from([(
                            "sampler-public-key".to_owned(),
                            RuntimeValue::Bytes(vec![0x42; 32]),
                        )]),
                        &mut MemoryArtifactStore::default(),
                        SamplingMode::Fresh,
                    )
                    .unwrap();
                    let RuntimeValue::Matrix(private_value) = &block_result.outputs["private"]
                    else {
                        panic!("private switch companion")
                    };
                    let RuntimeValue::Matrix(public_value) = &block_result.outputs["public"] else {
                        panic!("public switch companion")
                    };
                    assert_eq!(private_value, public_value);
                }
            }
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::from([(
                "sampler-public-key".to_owned(),
                RuntimeValue::Bytes(vec![0x42; 32]),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for name in ["input", "mask-0", "mask-1", "switch-0", "switch-1"] {
            let RuntimeValue::Matrix(private) = &result.outputs[&format!("private-{name}")] else {
                panic!("private sampler output is not a matrix")
            };
            let RuntimeValue::Matrix(public) = &result.outputs[&format!("public-{name}")] else {
                panic!("public sampler output is not a matrix")
            };
            assert_eq!(
                private.as_ref(),
                public.as_ref(),
                "sampler public matrix mismatch for {name}"
            );
        }
    }

    #[test]
    fn rhs_sampling_and_fuse_never_slice_a_decomposition_result() {
        let layout = mxx_bgg::BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: 2.into(),
        };
        let ring = layout.ring();
        let sampler = super::PowerLutEncodingSampler {
            layout,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let helpers = sampler
            .sample_automorphism_helpers(
                ring.input("shape-secret", (1, 2)),
                ring.bytes_input("shape-hash", 32),
                &b"shape-test"[..],
                2,
            )
            .unwrap();
        let mut context = DslContext::new("power-lut-no-decomposition-slices");
        for helper in &helpers {
            context = context
                .output("switch-gsw", helper.switch().gsw_ciphertext().clone())
                .unwrap()
                .output(
                    "switch-companion",
                    helper.switch().companion(0, 0, 4).unwrap().vector.clone(),
                )
                .unwrap();
        }
        let graph = context.build().unwrap();
        let slices_over_decomposition =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
                matches!(node.kind(), NodeKind::Slice { .. }) &&
                    node.arguments().iter().any(|argument| {
                        matches!(argument.node().kind(), NodeKind::GadgetDecompose { .. })
                    })
            });
        assert!(!slices_over_decomposition);
    }

    #[test]
    fn private_fuse_shares_one_lhs_public_decomposition() {
        let ring = Ring::new(257, 4);
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 8.into(),
            digit_count: 2.into(),
        });
        let lhs = BggEncodingWire {
            vector: ring.input("fuse-lhs-vector", (1, 2)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("fuse-lhs-public", (1, 2)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let rhs = PowerRhsPackage::new(
            ring.input("fuse-gsw", (1, 2)),
            vec![
                PowerRhsCompanionBlock {
                    vector: ring.input("fuse-companion-vector", (1, 4)),
                    public_matrix: ring.input("fuse-companion-public", (1, 4)),
                },
                PowerRhsCompanionBlock {
                    vector: ring.input("fuse-companion-vector-1", (1, 4)),
                    public_matrix: ring.input("fuse-companion-public-1", (1, 4)),
                },
            ],
        )
        .unwrap();
        let fused = compiler.fuse(&lhs, &rhs).unwrap();
        let graph = DslContext::new("power-lut-private-fuse-shared-decomposition")
            .output("vector", fused.vector)
            .unwrap()
            .output("public", fused.pubkey.matrix)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        let decompositions = nodes
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
            .collect::<Vec<_>>();
        let lhs_decompositions = decompositions
            .iter()
            .filter(|node| {
                node.arguments().first().is_some_and(|argument| {
                    matches!(argument.node().kind(), NodeKind::Input { name, .. } if name == "fuse-lhs-public")
                })
            })
            .count();
        let gsw_decompositions = decompositions
            .iter()
            .filter(|node| {
                node.arguments().first().is_some_and(|argument| {
                    matches!(argument.node().kind(), NodeKind::Slice { .. })
                })
            })
            .count();
        assert_eq!(decompositions.len(), 2, "one lhs and one structural GSW decomposition");
        assert_eq!(lhs_decompositions, 1, "private vector and public ancestry share D(lhs)");
        assert_eq!(gsw_decompositions, 1, "the loop body decomposes its selected GSW column once");
        assert!(!nodes.iter().any(|node| matches!(node.kind(), NodeKind::Tensor)));
        let lhs_decomposition = decompositions
            .iter()
            .find(|node| {
                node.arguments().first().is_some_and(|argument| {
                    matches!(argument.node().kind(), NodeKind::Input { name, .. } if name == "fuse-lhs-public")
                })
            })
            .expect("complete lhs decomposition");
        assert!(!nodes.iter().any(|node| {
            matches!(node.kind(), NodeKind::Slice { .. }) &&
                node.arguments().iter().any(|argument| {
                    argument.node() == **lhs_decomposition ||
                        (matches!(argument.node().kind(), NodeKind::MatrixScale { .. }) &&
                            argument
                                .node()
                                .arguments()
                                .iter()
                                .any(|nested| nested.node() == **lhs_decomposition))
                })
        }));
        let column_loops = graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.kind(), NodeKind::ParallelLoop(spec)
                    if spec.output_mode == mxx_ir_core::node::ParallelOutputMode::CollectColumns)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            column_loops.len(),
            2,
            "private Fuse has one ordered column sink per vector/public projection"
        );
        for column_loop in column_loops {
            let column_loop_id = graph.graph.root_scope().node_id(column_loop).unwrap();
            let body_id = graph
                .graph
                .child_scope_id(&mxx_ir_core::graph::FrozenGraphScopeId::Root, column_loop_id)
                .unwrap();
            let body = graph.graph.scope(&body_id).unwrap();
            assert!(
                !body.nodes().iter().any(|node| {
                    matches!(node.kind(), NodeKind::Tensor | NodeKind::Concat { .. })
                })
            );
            assert!(matches!(body.outputs().first().and_then(|wire| body.node(wire.node))
                .and_then(|node| node.output_types().first()), Some(WireType::Matrix(matrix))
                if matrix.columns == 1.into()));
            assert!(matches!(column_loop.output_types().first(), Some(WireType::Matrix(matrix))
                if matrix.columns == 2.into()));
        }
    }

    /// Small runtime harness for checking the BGG relation itself. All
    /// public matrices in this harness come from the independent public
    /// compiler; no expected key is copied from an encoding result.
    struct NoiselessFixture {
        parameters: DCRTPolyParams,
        ring: Ring,
        compiler: PowerLutEncodingCompiler,
        public_compiler: PowerLutPublicKeyCompiler,
        sampler: super::PowerLutEncodingSampler,
        public_sampler: PowerLutPublicKeySampler,
        secret: mxx_dsl::Mat,
        hash_key: mxx_dsl::Bytes,
    }

    impl NoiselessFixture {
        fn new() -> Self {
            // Keep two gadget digits per CRT tower, while using the largest
            // base that still satisfies ceil(crt_bits / base_bits) == 2.
            let parameters = DCRTPolyParams::new(4, 1, 17, 9);
            let modulus = BigInt::from(parameters.modulus().as_ref().clone());
            let ring = Ring::new(modulus.clone(), 4);
            let layout = mxx_bgg::BggSamplerLayout {
                modulus: modulus.into(),
                ring_dimension: 4.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 512.into(),
            };
            let bgg = BggPublicKeyCompiler {
                ring: ring.clone(),
                base: layout.gadget_base.clone(),
                digit_count: layout.digit_count.into(),
            };
            Self {
                parameters,
                ring: ring.clone(),
                compiler: PowerLutEncodingCompiler::from_public_key(bgg.clone()),
                public_compiler: PowerLutPublicKeyCompiler::new(bgg),
                sampler: super::PowerLutEncodingSampler {
                    layout: layout.clone(),
                    gaussian_sigma: None,
                    gaussian_max_coefficient_bound: None,
                },
                public_sampler: PowerLutPublicKeySampler { layout },
                secret: ring.input("noiseless-secret", (1, 2)),
                hash_key: ring.bytes_input("noiseless-hash", 32),
            }
        }

        fn rotation(&self, exponent: usize) -> mxx_dsl::Mat {
            self.ring.constant(
                (1, 1),
                mxx_ir_core::node::ConstantMatrix::Rotation { exponent: exponent.into() },
            )
        }

        fn input(&self, tag: &[u8], exponent: usize) -> BggEncodingWire {
            self.sampler
                .sample_input_encoding(
                    self.secret.clone(),
                    self.hash_key.clone(),
                    tag,
                    self.rotation(exponent),
                )
                .expect("noiseless sampled input")
        }

        fn public_input(&self, tag: &[u8]) -> BggPublicKeyWire {
            self.public_sampler
                .sample_input_key(self.hash_key.clone(), tag)
                .expect("noiseless public input")
        }

        fn helpers(&self, tag: &[u8], width: usize) -> Vec<AutomorphismHelper> {
            self.sampler
                .sample_automorphism_helpers(self.secret.clone(), self.hash_key.clone(), tag, width)
                .expect("noiseless sampled helpers")
        }

        fn public_helpers(
            &self,
            tag: &[u8],
            width: usize,
        ) -> Vec<crate::public_key::AutomorphismPublicHelper> {
            self.public_sampler
                .sample_automorphism_helpers(self.hash_key.clone(), tag, width)
                .expect("noiseless public helpers")
        }

        /// Independently evaluates the public operation and checks both
        /// required assertions: public-key equality and the noiseless BGG
        /// equation `c = s*A - mu*(s*G)`.
        fn assert_relation(
            &self,
            name: &str,
            encoded: BggEncodingWire,
            expected_public: mxx_dsl::Mat,
            expected_mu: mxx_dsl::Mat,
        ) {
            let gadget = self.ring.gadget(2, 512, 2);
            let expected_vector = self.secret.clone() * expected_public.clone() -
                expected_mu * (self.secret.clone() * gadget);
            let context = DslContext::new(name)
                .output("encoded-vector", encoded.vector)
                .unwrap()
                .output("encoded-public", encoded.pubkey.matrix)
                .unwrap()
                .output("expected-public", expected_public.clone())
                .unwrap()
                .output("expected-vector", expected_vector)
                .unwrap();
            let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
            let result = execute(
                &graph,
                &mut cpu_backend([self.parameters.clone()]),
                BTreeMap::from([
                    (
                        "noiseless-secret".to_owned(),
                        RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec(
                            &self.parameters,
                            vec![vec![
                                DCRTPoly::from_u32s(&self.parameters, &[2, 1, 0, 0]),
                                DCRTPoly::from_usize_to_constant(&self.parameters, 1),
                            ]],
                        )),
                    ),
                    ("noiseless-hash".to_owned(), RuntimeValue::Bytes(vec![0x91; 32])),
                ]),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            let RuntimeValue::Matrix(encoded_vector) = &result.outputs["encoded-vector"] else {
                panic!("encoded vector")
            };
            let RuntimeValue::Matrix(expected_vector) = &result.outputs["expected-vector"] else {
                panic!("expected vector")
            };
            let RuntimeValue::Matrix(encoded_public) = &result.outputs["encoded-public"] else {
                panic!("encoded public")
            };
            let RuntimeValue::Matrix(expected_public) = &result.outputs["expected-public"] else {
                panic!("expected public")
            };
            assert_eq!(encoded_public, expected_public, "{name}: public key mismatch");
            assert_eq!(encoded_vector, expected_vector, "{name}: noiseless BGG relation");
        }
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn noiseless_sampled_input_and_automorphism_satisfy_bgg_relation() {
        let fixture = NoiselessFixture::new();
        let input = fixture.input(&b"sampled-input"[..], 1);
        let public_input = fixture.public_input(&b"sampled-input"[..]);
        fixture.assert_relation(
            "power-lut-noiseless-sampled-input",
            input,
            public_input.matrix,
            fixture.rotation(1),
        );

        let independent_public = fixture.public_input(&b"independent-public-matrix"[..]);
        let independently_sampled = fixture
            .sampler
            .sample_encoding_for_public_matrix(
                fixture.secret.clone(),
                independent_public.clone(),
                fixture.rotation(1),
            )
            .expect("independently supplied public matrix");
        fixture.assert_relation(
            "power-lut-noiseless-independent-public-matrix",
            independently_sampled,
            independent_public.matrix,
            fixture.rotation(1),
        );

        let helpers = fixture.helpers(&b"sampled-helper"[..], 2);
        let public_helpers = fixture.public_helpers(&b"sampled-helper"[..], 2);
        let transformed = fixture
            .compiler
            .automorphism(&fixture.input(&b"auto-input"[..], 1), &helpers[0])
            .unwrap();
        let public = fixture
            .public_compiler
            .automorphism(&fixture.public_input(&b"auto-input"[..]).matrix, &public_helpers[0])
            .unwrap();
        fixture.assert_relation(
            "power-lut-noiseless-automorphism",
            transformed,
            public,
            fixture.rotation(1).ring_automorphism(helpers[0].index()),
        );
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn noiseless_sampled_automorphism_helper_material_has_its_declared_relations() {
        let fixture = NoiselessFixture::new();
        let helper = &fixture.helpers(&b"helper-relation"[..], 2)[0];
        let gadget = fixture.ring.gadget(2, 512, 2);
        let source = fixture.secret.clone().ring_automorphism(helper.index());
        let switch_left = source.clone() * helper.switch().gsw_ciphertext().clone();
        let switch_right = fixture.secret.clone() * gadget.clone();
        let mask_expected =
            fixture.secret.clone() * helper.mask().pubkey.matrix.clone() - source * gadget;
        let graph = DslContext::new("power-lut-noiseless-helper-material")
            .output("switch-left", switch_left)
            .unwrap()
            .output("switch-right", switch_right)
            .unwrap()
            .output("mask", helper.mask().vector.clone())
            .unwrap()
            .output("mask-expected", mask_expected)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([fixture.parameters.clone()]),
            BTreeMap::from([
                (
                    "noiseless-secret".to_owned(),
                    RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec(
                        &fixture.parameters,
                        vec![vec![
                            DCRTPoly::from_u32s(&fixture.parameters, &[2, 1, 0, 0]),
                            DCRTPoly::from_usize_to_constant(&fixture.parameters, 1),
                        ]],
                    )),
                ),
                ("noiseless-hash".to_owned(), RuntimeValue::Bytes(vec![0x91; 32])),
            ]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(switch_left) = &result.outputs["switch-left"] else {
            panic!("sampled switch relation left")
        };
        let RuntimeValue::Matrix(switch_right) = &result.outputs["switch-right"] else {
            panic!("sampled switch relation right")
        };
        assert_eq!(switch_left, switch_right, "sampled switch relation");
        let RuntimeValue::Matrix(mask) = &result.outputs["mask"] else { panic!("sampled mask") };
        let RuntimeValue::Matrix(mask_expected) = &result.outputs["mask-expected"] else {
            panic!("sampled mask relation")
        };
        assert_eq!(mask, mask_expected, "sampled mask relation");
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn balanced_sum_family_runtime_handles_ragged_counts() {
        let parameters = DCRTPolyParams::new(4, 1, 17, 9);
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, 4);
        for count in [1usize, 2, 3, 5, 6, 7, 9] {
            let values = ring.input_family(format!("balanced-values-{count}"), count, (1, 1));
            let sum = super::balanced_sum_family(values).unwrap();
            let graph = DslContext::new(format!("balanced-runtime-{count}"))
                .output("sum", sum)
                .unwrap()
                .build()
                .unwrap();
            let validated = graph.validate(&ParamEnv::default()).unwrap();
            let matrix = |value: u32| {
                RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_u32s(&parameters, &[value, 0, 0, 0])],
                ))
            };
            let input = RuntimeValue::IndexedFamily((1..=count as u32).map(matrix).collect());
            let result = execute(
                &validated,
                &mut cpu_backend([parameters.clone()]),
                BTreeMap::from([(format!("balanced-values-{count}"), input)]),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            let RuntimeValue::Matrix(actual) = &result.outputs["sum"] else { panic!("matrix") };
            let expected = DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                vec![DCRTPoly::from_u32s(
                    &parameters,
                    &[(count * (count + 1) / 2) as u32, 0, 0, 0],
                )],
            );
            assert_eq!(actual.as_ref(), &expected, "count {count}");
            if count % 2 == 1 && count > 1 {
                let nodes = graph
                    .graph
                    .scopes()
                    .values()
                    .flat_map(|scope| scope.nodes())
                    .collect::<Vec<_>>();
                assert!(nodes.iter().any(|node| matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::IntBinary(
                        mxx_ir_core::node::IntBinaryOp::Remainder
                    ) if node.arguments().get(1).is_some_and(|divisor| matches!(
                        divisor.node().kind(),
                        mxx_ir_core::node::NodeKind::ConstantInt(value)
                            if value == &BigInt::from(count)
                    ))
                )));
                assert!(!nodes.iter().any(|node| matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::IntBinary(
                        mxx_ir_core::node::IntBinaryOp::Subtract
                    )
                )));
            }
        }
    }
}
