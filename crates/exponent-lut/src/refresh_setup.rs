//! RNS refresh setup and artifact boundary.
//!
//! This module owns the setup-only half of refresh: it samples the public and
//! private preprocessing material, builds the preprocessing graph, records an
//! attestation of the exact decoder relations, and imports that material at a
//! later production boundary. The runtime refresh equations themselves live
//! in [`crate::refresh`], while sparse-LWR/PBC evaluation lives in
//! [`crate::prf`] and [`crate::pbc`]. Keeping these layers separate is
//! important: setup may hold secrets and trapdoors, but the public program and
//! graph declarations must not contain private support or selector choices.
//!
//! Setup material is organized per CRT slot `t`. With `mu_t=q/q_t`, the
//! producer constructs `A_{sum,t} = A_t + A_{m,t} + A_{e,t}`, then the preimage
//! target `T_t = A_{sum,t} - mu_t A'` and a matrix `K_t` satisfying `B K_t=T_t`.
//! One shared decoder base `b=sB+e_B` yields `d_t=bK_t` for each slot; imported
//! declarations retain these exact graph relations while keeping secrets and
//! plaintext out of the public identity.

use crate::{
    ExponentLutEncodingCompiler, ExponentLutError,
    encoding::{EncodingSelectorFamily, ExponentLutEncodingSampler, FlatLutHelperSet},
    noise::{
        AverageCaseConfig, AverageVariance, ExponentLutAverageNoiseReport,
        ExponentLutNoiseParameters, ExponentLutNoiseReport, ExponentLutNoiseSnapshot,
    },
    pbc::{
        PbcPublicLayout, PbcSelectorArtifactNames, PbcSelectorArtifacts, PbcTrustedSelectorBits,
        build_structural_selector_families,
    },
    prf::{
        RefreshPrfBatchInputs, SparseLwrPrfHelperBundle, SparseLwrPrfProfile, SparseLwrPrfProgram,
        SparseLwrPrfPublicHelperBundle, SparseLwrPrfTerminalForm, SparseLwrPublicReductionHelpers,
        SparseLwrPublicTerminalHelpers, SparseLwrReductionHelpers, SparseLwrTerminalHelpers,
    },
    program::ExponentLutProgramId,
    public_key::{
        ExponentLutPublicKeyCompiler, ExponentLutPublicKeySampler, FlatLutPublicHelper,
        FlatLutPublicHelperSet, PublicSelectorFamily,
    },
    refresh::{
        RefreshCompiler, RefreshError, RefreshFreshErrorMaterial, RefreshMaskMaterial,
        RefreshPrfContract, RefreshPrfCoverage, RefreshPrfFamilyMaterial, RefreshSetupManifest,
        aggregate_refresh_fresh_error_per_slot, aggregate_refresh_masks,
    },
};
use bigdecimal::BigDecimal;
use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout};
use mxx_dsl::{Bool, BuiltGraph, Bytes, DslContext, Family, HashTag, Int, Mat, Parallel};
use mxx_ir_core::{
    ParamEnv, ScopedWireRef,
    artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId},
    encoding::{IR_VERSION, hash_canonical, spec_hash},
    graph::{FrozenGraphScopeId, Graph},
    types::{ConcreteMatrixType, NodeId, WireType},
};
use num_bigint::{BigUint, ToBigInt};
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
/// Errors raised while building or importing trusted refresh setup.
pub enum RefreshSetupError {
    #[error(transparent)]
    /// The refresh declaration or its CRT equations are invalid.
    Refresh(#[from] RefreshError),
    #[error(transparent)]
    /// A Exponent-LUT boundary rejected the setup inputs.
    ExponentLut(#[from] ExponentLutError),
    #[error(transparent)]
    /// Sparse-LWR encoding setup sampling failed.
    Sampling(#[from] crate::encoding::ExponentLutSamplingError),
    #[error(transparent)]
    /// A BGG encoding operation failed while constructing setup.
    Encoding(#[from] mxx_bgg::EncodingCompileError),
    #[error(transparent)]
    /// BGG trapdoor or matrix sampling failed.
    BggSampling(#[from] mxx_bgg::BggSampleError),
    #[error("PBC construction failed: {0}")]
    /// Public PBC layout generation failed.
    Pbc(String),
    #[error(transparent)]
    /// The DSL rejected graph construction or a family operation.
    Dsl(#[from] mxx_dsl::DslError),
    #[error("invalid refresh setup parameters: {0}")]
    /// Setup dimensions or concrete parameter values are inconsistent.
    InvalidParameters(&'static str),
    #[error("refresh setup manifest is invalid")]
    /// The producer graph or artifact manifest violates its role schema.
    InvalidManifest,
    #[error("refresh setup identity mismatch")]
    /// A setup value belongs to a different refresh, layout, or program.
    IdentityMismatch,
    #[error("refresh verification decoder residuals do not match the bound operands")]
    /// A decoder target is not linked to the residual proved by the graph.
    DecoderResidualMismatch,
}

#[derive(Clone)]
/// Public dimensions and identities used to build one refresh setup.
///
/// The fields describe both the BGG sampler and the CRT refresh.
/// `lut_width` is the Exponent-LUT coefficient-sieve width; it is independent of
/// any PBC bucket width. The constructor sets the authoritative decoder bound
/// policy, and validation is repeated at every producer/import boundary.
pub struct RefreshSetupParameters {
    /// Application-level refresh instance identity.
    pub refresh_id: [u8; 32],
    /// Sparse-LWR modulus used by the PRF layer.
    pub base_p: usize,
    /// Number of secret components represented by each BGG matrix.
    pub component_count: usize,
    /// Number of refresh coefficients processed per CRT slot.
    pub coefficient_count: usize,
    /// Number of base-`p` digits represented by each mask PRF label. This is
    /// independent of `layout.digit_count`, which is the gadget digit count.
    pub mask_base_p_digit_count: usize,
    /// Number of base-`p` digits represented by each fresh-error PRF label.
    pub fresh_error_base_p_digit_count: usize,
    /// Statistical security parameter used for the joint mask transcript.
    pub mask_statistical_security_bits: usize,
    /// Explicit Exponent-LUT sieve width `W`; this is not the PBC bucket width.
    pub lut_width: usize,
    /// BGG sampler dimensions shared by setup artifacts.
    pub layout: BggSamplerLayout,
    /// CRT modulus and reconstruction data for the refresh equations.
    pub refresh: RefreshCompiler,
    /// Gaussian width used when sampling the decoder trapdoor.
    pub decoder_sigma: mxx_ir_core::RealExpr,
    /// Gaussian width used by every setup helper and the shared anchor error.
    pub encoding_error_sigma: mxx_ir_core::RealExpr,
    /// Common hard coefficient cutoff `B_chi` for setup helper errors.
    pub encoding_error_bound: mxx_ir_core::IntExpr,
    /// Policy for the decoder preimage rejection cutoff.  The policy is
    /// resolved to one concrete integer before the producer graph is built.
    pub decoder_preimage_bound: mxx_bgg::PreimageCoefficientBound,
    /// Human-readable setup name used by callers when selecting a production.
    pub name: String,
}

fn resolve_encoding_error_bound(
    sigma: &mxx_ir_core::RealExpr,
) -> Result<mxx_ir_core::IntExpr, RefreshSetupError> {
    let sigma = sigma
        .evaluate_f64(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("encoding sigma must be concrete"))?;
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(RefreshSetupError::InvalidParameters(
            "encoding sigma must be positive and finite",
        ));
    }
    let sigma = BigDecimal::from_f64(sigma)
        .ok_or(RefreshSetupError::InvalidParameters("encoding sigma is not finite"))?;
    mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma)
        .to_bigint()
        .map(Into::into)
        .ok_or(RefreshSetupError::InvalidParameters("encoding cutoff overflow"))
}

fn encoding_error_bound_or_zero(sigma: &mxx_ir_core::RealExpr) -> mxx_ir_core::IntExpr {
    resolve_encoding_error_bound(sigma).unwrap_or_else(|_| 0.into())
}

fn average_variance_from_encoding_sigma(
    sigma: &mxx_ir_core::RealExpr,
) -> Result<AverageVariance, RefreshSetupError> {
    let sigma = sigma
        .evaluate_rational(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("encoding sigma must be rational"))?;
    let numerator = sigma
        .numerator()
        .to_biguint()
        .filter(|value| !value.is_zero())
        .ok_or(RefreshSetupError::InvalidParameters("encoding sigma must be positive"))?;
    let denominator = sigma
        .denominator()
        .to_biguint()
        .filter(|value| !value.is_zero())
        .ok_or(RefreshSetupError::InvalidParameters("encoding sigma denominator invalid"))?;
    AverageVariance::new(BigUint::from(4u8) * &numerator * &numerator, &denominator * &denominator)
        .map_err(|_| RefreshSetupError::InvalidParameters("encoding variance is invalid"))
}

fn is_power_of_two_expr(value: &mxx_ir_core::IntExpr) -> bool {
    value
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_biguint())
        .is_some_and(|value| value > 1u8.into() && (BigUint::one() << (value.bits() - 1)) == value)
}

impl std::fmt::Debug for RefreshSetupParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshSetupParameters")
            .field("refresh_id", &self.refresh_id)
            .field("base_p", &self.base_p)
            .field("component_count", &self.component_count)
            .field("coefficient_count", &self.coefficient_count)
            .field("mask_base_p_digit_count", &self.mask_base_p_digit_count)
            .field("fresh_error_base_p_digit_count", &self.fresh_error_base_p_digit_count)
            .field("mask_statistical_security_bits", &self.mask_statistical_security_bits)
            .field("lut_width", &self.lut_width)
            .field("layout", &self.layout)
            .field("decoder_sigma", &self.decoder_sigma)
            .field("encoding_error_sigma", &self.encoding_error_sigma)
            .field("encoding_error_bound", &self.encoding_error_bound)
            .field("decoder_preimage_bound", &self.decoder_preimage_bound)
            .field("name", &self.name)
            .finish()
    }
}
impl RefreshSetupParameters {
    /// Creates refresh setup parameters with the official preimage cutoff
    /// policy.  Call [`Self::with_decoder_preimage_bound`] only for an
    /// explicitly reviewed alternative cutoff.
    pub fn new(
        refresh_id: [u8; 32],
        base_p: usize,
        component_count: usize,
        coefficient_count: usize,
        mask_base_p_digit_count: usize,
        fresh_error_base_p_digit_count: usize,
        mask_statistical_security_bits: usize,
        lut_width: usize,
        layout: BggSamplerLayout,
        refresh: RefreshCompiler,
        decoder_sigma: mxx_ir_core::RealExpr,
        encoding_error_sigma: mxx_ir_core::RealExpr,
        name: impl Into<String>,
    ) -> Self {
        let encoding_error_bound = encoding_error_bound_or_zero(&encoding_error_sigma);
        Self {
            refresh_id,
            base_p,
            component_count,
            coefficient_count,
            mask_base_p_digit_count,
            fresh_error_base_p_digit_count,
            mask_statistical_security_bits,
            lut_width,
            layout,
            refresh,
            decoder_sigma,
            encoding_error_sigma,
            encoding_error_bound,
            decoder_preimage_bound: mxx_bgg::PreimageCoefficientBound::Official,
            name: name.into(),
        }
    }

    /// Replaces the default official cutoff policy with an explicit policy.
    pub fn with_decoder_preimage_bound(mut self, bound: mxx_bgg::PreimageCoefficientBound) -> Self {
        self.decoder_preimage_bound = bound;
        self
    }

    /// Builds the concrete sparse-LWR PRF profile for a selected `Q_L`.
    ///
    /// Refresh setup owns `p` and `W`; the caller supplies the concrete
    /// plaintext modulus because a refresh may contain several CRT moduli.
    /// Keeping this check explicit prevents a PBC bucket width from being
    /// accidentally reused as the Exponent-LUT domain width.
    pub fn sparse_lwr_profile(
        &self,
        q_l: usize,
    ) -> Result<crate::prf::SparseLwrPrfProfile, RefreshSetupError> {
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?;
        crate::prf::SparseLwrPrfProfile::new(q_l, self.base_p, self.lut_width, ring_dimension)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid sparse-LWR profile"))
    }

    /// Number of PRF components covered by every mask and fresh-error group.
    /// This is the BGG public-key column count (`2 * ell_beta`), not the
    /// secret dimension stored in `component_count`.
    pub fn prf_component_count(&self) -> usize {
        self.layout.public_key_columns()
    }

    /// Returns the setup-derived doubled-coordinate variance of one sampled
    /// encoding-error coefficient. AverageCase cannot start from a caller
    /// supplied variance; unsupported symbolic/non-rational distributions
    /// fail closed.
    pub fn average_initial_variance(&self) -> Result<AverageVariance, RefreshSetupError> {
        self.validate()?;
        average_variance_from_encoding_sigma(&self.encoding_error_sigma)
    }

    /// Rebuilds the deterministic noise snapshot from this complete setup and
    /// a public PRF/layout pair. The same method is used by candidate
    /// evaluation and by graph construction so changing `d_m` necessarily
    /// changes the resulting setup identity/model together.
    pub fn build_noise_snapshot(
        &self,
        prf_program: SparseLwrPrfProgram,
        pbc_layout: PbcPublicLayout,
    ) -> Result<ExponentLutNoiseSnapshot, RefreshSetupError> {
        self.validate()?;
        let decoder_preimage_bound = self.resolve_decoder_preimage_bound()?;
        let contract = RefreshPrfContract::from_program(&prf_program);
        let setup_identity =
            identity_digest(self, &contract, pbc_layout.layout_id, &decoder_preimage_bound);
        let eval_uint = |value: &mxx_ir_core::IntExpr,
                         message: &'static str|
         -> Result<BigUint, RefreshSetupError> {
            value
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters(message))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters(message))
        };
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
            .to_usize()
            .ok_or(RefreshSetupError::InvalidParameters("ring dimension does not fit usize"))?;
        let gadget_base = self
            .layout
            .gadget_base
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
            .to_biguint()
            .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))?;
        let helper_error_bound = eval_uint(
            &self.encoding_error_bound,
            "encoding error bound must be a non-negative concrete integer",
        )?;
        let full_modulus = eval_uint(
            &self.refresh.full_modulus,
            "full modulus must be a positive concrete integer",
        )?;
        let plaintext_moduli = self
            .refresh
            .crt_plaintext_moduli
            .iter()
            .map(|value| eval_uint(value, "plaintext modulus must be a positive concrete integer"))
            .collect::<Result<Vec<_>, _>>()?;
        let decoder_preimage_bound_value = decoder_preimage_bound
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters("decoder bound must be concrete"))?
            .to_biguint()
            .ok_or(RefreshSetupError::InvalidParameters("decoder bound must be positive"))?;
        let noise_model = ExponentLutNoiseParameters::dense(
            ring_dimension,
            gadget_base,
            self.layout.digit_count,
            helper_error_bound.clone(),
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid exact noise model"))?;
        ExponentLutNoiseSnapshot::from_setup(
            setup_identity,
            prf_program,
            pbc_layout,
            noise_model,
            full_modulus,
            BigUint::from(self.base_p),
            self.mask_base_p_digit_count,
            self.fresh_error_base_p_digit_count,
            u64::try_from(self.mask_statistical_security_bits).map_err(|_| {
                RefreshSetupError::InvalidParameters("mask security parameter does not fit u64")
            })?,
            self.refresh.crt_plaintext_moduli.len(),
            self.coefficient_count,
            plaintext_moduli,
            decoder_preimage_bound_value,
            helper_error_bound,
            self.average_initial_variance()?,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid exact noise snapshot"))
    }

    /// Evaluates one AverageCase search candidate using the same complete,
    /// setup-owned snapshot construction used by final graph assembly.
    ///
    /// Only the candidate mask digit count is varied; all other setup fields,
    /// including the fresh-error digit count, remain fixed.  The returned
    /// report carries the canonical snapshot identity and both the independent
    /// WorstCase hard authority and AverageCase correctness result, so a
    /// selector can compare the report with the final bundle without building
    /// a graph or supplying a proxy variance.
    pub fn evaluate_average_candidate(
        &self,
        prf_program: SparseLwrPrfProgram,
        pbc_layout: PbcPublicLayout,
        candidate_mask_base_p_digit_count: usize,
        config: &AverageCaseConfig,
    ) -> Result<ExponentLutAverageNoiseReport, RefreshSetupError> {
        let mut candidate = self.clone();
        candidate.mask_base_p_digit_count = candidate_mask_base_p_digit_count;
        let snapshot = candidate.build_noise_snapshot(prf_program, pbc_layout)?;
        snapshot
            .simulate_average(config)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid AverageCase candidate"))
    }

    fn validate_dimensions(&self) -> Result<(), RefreshSetupError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|x| x.to_usize())
            .ok_or(RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?;
        if self.base_p < 2 ||
            self.component_count != self.layout.secret_dimension ||
            self.coefficient_count == 0 ||
            self.coefficient_count != n ||
            self.lut_width == 0 ||
            !self.lut_width.is_power_of_two() ||
            self.lut_width > n ||
            n % self.lut_width != 0 ||
            self.mask_base_p_digit_count == 0 ||
            self.fresh_error_base_p_digit_count == 0 ||
            self.mask_statistical_security_bits == 0 ||
            self.layout.digit_count == 0 ||
            self.layout.secret_dimension != 2 ||
            !is_power_of_two_expr(&self.layout.gadget_base) ||
            self.refresh.full_modulus.canonicalize() != self.layout.modulus.canonicalize()
        {
            return Err(RefreshSetupError::InvalidParameters("inconsistent dimensions"));
        }
        let expected_prf_component_count = self
            .layout
            .secret_dimension
            .checked_mul(self.layout.digit_count)
            .ok_or(RefreshSetupError::InvalidParameters("public-key column count overflow"))?;
        if self.prf_component_count() != expected_prf_component_count {
            return Err(RefreshSetupError::InvalidParameters("invalid public-key column count"));
        }
        self.refresh.validate_layout()?;
        let expected_bound = resolve_encoding_error_bound(&self.encoding_error_sigma)?;
        if self.encoding_error_bound != expected_bound {
            return Err(RefreshSetupError::InvalidParameters(
                "encoding error cutoff does not match encoding sigma",
            ));
        }
        Ok(())
    }

    /// Resolves the preimage cutoff using the authoritative primitives formula.
    /// All dimensions and sigma must be concrete and valid; symbolic,
    /// non-positive, non-finite, and overflowing inputs fail closed.
    pub fn resolve_decoder_preimage_bound(
        &self,
    ) -> Result<mxx_ir_core::IntExpr, RefreshSetupError> {
        self.validate_dimensions()?;
        self.decoder_preimage_bound
            .resolve(&self.layout, self.layout.secret_dimension, &self.decoder_sigma)
            .map_err(RefreshSetupError::from)
    }

    fn validate(&self) -> Result<(), RefreshSetupError> {
        self.validate_dimensions()?;
        self.resolve_decoder_preimage_bound().map(|_| ())
    }
}

#[derive(Clone)]
/// Validated PRF outputs needed by refresh preprocessing.
///
/// The private vectors are retained only behind this setup boundary. Each
/// output carries the exact PRF program and PBC layout identities, and the
/// constructor checks complete slot/component/coefficient/digit coverage.
pub struct RefreshPrfInputs {
    masks: Vec<RefreshMaskMaterial>,
    fresh_error: RefreshFreshErrorMaterial,
    contract: RefreshPrfContract,
    layout_id: crate::pbc::PbcLayoutId,
    encoding_error_sigma: mxx_ir_core::RealExpr,
    encoding_error_bound: mxx_ir_core::IntExpr,
}

impl RefreshPrfInputs {
    /// Binds the complete, canonical PBC output family produced by the typed
    /// PRF lowerer to the refresh label index. This is the production
    /// honest-builder entry point: it rejects label reordering/splicing and
    /// retains family handles until structural routing and reduction.
    pub fn from_pbc_family_outputs(
        parameters: &RefreshSetupParameters,
        expected_program: &SparseLwrPrfProgram,
        expected_batch: &RefreshPrfBatchInputs,
        outputs: &crate::prf::PbcSparseLwrEncodingOutputs,
    ) -> Result<Self, RefreshSetupError> {
        parameters.validate()?;
        let contract = RefreshPrfContract::from_program(expected_program);
        contract.validate_for(parameters)?;
        let coverage = RefreshPrfCoverage::new(
            parameters.refresh_id,
            parameters.prf_component_count(),
            parameters.coefficient_count,
            parameters.mask_base_p_digit_count,
            parameters.fresh_error_base_p_digit_count,
        )?;
        let labels = crate::refresh::RefreshPrfLabelIndex::new(
            parameters.refresh_id,
            parameters.refresh.crt_plaintext_moduli.len(),
            parameters.prf_component_count(),
            parameters.coefficient_count,
            parameters.mask_base_p_digit_count,
            parameters.fresh_error_base_p_digit_count,
        )?;
        let family = RefreshPrfFamilyMaterial::from_pbc_family_outputs(
            outputs,
            &labels,
            coverage.clone(),
            contract,
        )?;
        if expected_batch.len() != labels.len() ||
            expected_batch.layout_id() != outputs.family_metadata().2 ||
            expected_batch.batch_id() != outputs.batch_id()
        {
            return Err(RefreshSetupError::Refresh(RefreshError::PrfOutputMismatch));
        }
        let slot_count = parameters.refresh.crt_plaintext_moduli.len();
        let masks = (0..slot_count)
            .map(|slot| RefreshMaskMaterial::from_family(coverage.clone(), slot, family.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let fresh_error = RefreshFreshErrorMaterial::from_family(coverage, family)?;
        let layout_id = outputs.family_metadata().2;
        let inputs = Self {
            masks,
            fresh_error,
            contract,
            layout_id,
            encoding_error_sigma: parameters.encoding_error_sigma.clone(),
            encoding_error_bound: parameters.encoding_error_bound.clone(),
        };
        inputs.validate_for(parameters)?;
        Ok(inputs)
    }

    /// Re-runs all producer-side identity and coverage checks at consumption.
    pub(crate) fn validate_for(
        &self,
        parameters: &RefreshSetupParameters,
    ) -> Result<(), RefreshSetupError> {
        parameters.validate()?;
        if self.encoding_error_sigma != parameters.encoding_error_sigma ||
            self.encoding_error_bound != parameters.encoding_error_bound
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        self.contract.validate_for(parameters)?;
        let coverage = RefreshPrfCoverage::new(
            parameters.refresh_id,
            parameters.prf_component_count(),
            parameters.coefficient_count,
            parameters.mask_base_p_digit_count,
            parameters.fresh_error_base_p_digit_count,
        )?;
        if self.masks.len() != parameters.refresh.crt_plaintext_moduli.len() {
            return Err(RefreshSetupError::Refresh(RefreshError::SlotOrderMismatch));
        }
        for (slot, material) in self.masks.iter().enumerate() {
            if material.slot() != slot ||
                material.contract() != self.contract ||
                material.layout_id() != self.layout_id ||
                !material.coverage_matches(&coverage)
            {
                return Err(RefreshSetupError::Refresh(RefreshError::PrfOutputMismatch));
            }
            material.validate()?;
        }
        if self.fresh_error.contract() != self.contract ||
            self.fresh_error.layout_id() != self.layout_id ||
            !self.fresh_error.coverage_matches(&coverage)
        {
            return Err(RefreshSetupError::Refresh(RefreshError::PrfOutputMismatch));
        }
        self.fresh_error.validate()?;
        Ok(())
    }

    /// Aggregates the complete mask family in one symbolic routing body and
    /// one structural reduction per CRT slot. The fixed-size slot vector is
    /// materialized only after those reductions.
    fn aggregate_masks(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        base_p: usize,
    ) -> Result<Vec<BggEncodingWire>, RefreshSetupError> {
        let slot_count = self.masks.len();
        let family = self
            .masks
            .first()
            .and_then(RefreshMaskMaterial::family_material)
            .ok_or(RefreshSetupError::Refresh(RefreshError::PrfOutputMismatch))?;
        if !self.masks.iter().all(|material| material.family_material().is_some()) {
            return Err(RefreshSetupError::Refresh(RefreshError::PrfOutputMismatch));
        }
        aggregate_refresh_masks(compiler, base_p, family, slot_count)
            .map_err(RefreshSetupError::from)
    }
}

/// Inputs for constructing the trusted refresh preprocessing graph.
///
/// `state`, `secret`, and `hash_key` are setup material. The resulting graph
/// exports named artifacts; later execution must import those artifacts and
/// validate the producer attestation rather than reuse live producer wires.
pub struct RefreshPreprocessingRequest {
    /// Validated dimensions and CRT refresh declaration.
    pub parameters: RefreshSetupParameters,
    /// PRF mask and fresh-error outputs bound to the same setup identities.
    pub prf: RefreshPrfInputs,
    /// Private Exponent-LUT compiler used to aggregate PRF digits.
    pub compiler: ExponentLutEncodingCompiler,
    /// Ciphertext state to be refreshed; plaintext metadata is forbidden.
    pub state: BggEncodingWire,
    /// Secret vector used by the preprocessing equations.
    pub secret: Mat,
    /// Hash key used for deterministic public matrix derivation.
    pub hash_key: Bytes,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Names of the private vector and public matrix artifacts for one relation.
pub struct RefreshArtifactPairNames {
    /// Private vector artifact name.
    pub vector: String,
    /// Public matrix artifact name.
    pub public_matrix: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Canonical artifact names emitted by refresh preprocessing.
pub struct RefreshPreprocessingArtifactNames {
    /// Private state vector artifact.
    pub state_vector: String,
    /// Public matrix paired with [`Self::state_vector`].
    pub state_public_matrix: String,
    /// Public random matrix used in the target equation.
    pub a_prime: String,
    /// Public trapdoor matrix `B`.
    pub public_matrix_b: String,
    /// Per-CRT-slot fresh-error vectors after direct `kappa_t` routing.
    pub scaled_fresh: Vec<RefreshArtifactPairNames>,
    /// Per-CRT-slot private mask vectors and public matrices.
    pub masks: Vec<RefreshArtifactPairNames>,
    /// One shared decoder base vector paired with `B`.
    pub decoder_base_vector: String,
    /// Per-slot sampled preimage matrices.
    pub preimages: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct RefreshPreprocessingWires {
    pub(crate) state: BggEncodingWire,
    pub(crate) a_prime: Mat,
    pub(crate) public_b: Mat,
    pub(crate) scaled_fresh: Vec<BggEncodingWire>,
    pub(crate) masks: Vec<BggEncodingWire>,
    pub(crate) decoder_base: BggEncodingWire,
    pub(crate) preimages: Vec<mxx_dsl::Preimage>,
    pub(crate) names: RefreshPreprocessingArtifactNames,
    pub(crate) declaration: RefreshPreprocessingDeclaration,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Public declaration of a preprocessing producer and its artifact schema.
///
/// The identity commits to all dimensions, setup identities, decoder policy,
/// scales, and canonical artifact names. It is checked alongside the frozen
/// producer attestation before imported values become runtime wires.
pub struct RefreshPreprocessingDeclaration {
    /// Digest of this complete declaration.
    pub identity: [u8; 32],
    /// Canonical hash of the frozen producer graph.
    pub producer_spec_hash: mxx_ir_core::artifact::SpecHash,
    /// PBC layout identity used by the PRF outputs.
    pub pbc_layout_id: crate::pbc::PbcLayoutId,
    /// Refresh instance identity.
    pub refresh_id: [u8; 32],
    /// Sparse-LWR/Exponent-LUT program identity.
    pub program_id: ExponentLutProgramId,
    /// Plaintext modulus bound to the sparse-LWR PRF terminal.
    pub prf_q_l: usize,
    /// Raw scalar output modulus bound to the sparse-LWR PRF terminal.
    pub prf_p: usize,
    /// Logical LUT width bound to the sparse-LWR PRF program.
    pub prf_lut_width: usize,
    /// Ring dimension bound to the sparse-LWR PRF program.
    pub prf_ring_dimension: usize,
    /// Algebraic form promised by the PRF terminal.
    pub prf_terminal_form: SparseLwrPrfTerminalForm,
    /// Output wire promised by the PRF terminal.
    pub prf_output_wire: crate::program::ProgramWireId,
    /// Canonical names and roles of all produced artifacts.
    pub names: RefreshPreprocessingArtifactNames,
    /// Number of CRT slots.
    pub slot_count: usize,
    /// Number of secret components.
    pub component_count: usize,
    /// Number of public-key columns covered by each PRF group (`2 * ell_beta`).
    pub prf_component_count: usize,
    /// Number of coefficients per slot.
    pub coefficient_count: usize,
    /// Number of base-`p` digits represented by each mask PRF label.
    pub mask_base_p_digit_count: usize,
    /// Number of base-`p` digits represented by the shared fresh-error label group.
    pub fresh_error_base_p_digit_count: usize,
    /// Statistical security parameter used for the joint mask transcript.
    pub mask_statistical_security_bits: usize,
    /// Number of gadget digits used by the BGG layout.
    pub gadget_digit_count: usize,
    /// Decoder trapdoor Gaussian width.
    pub decoder_sigma: mxx_ir_core::RealExpr,
    /// Concrete maximum coefficient accepted by preimage sampling.
    pub decoder_preimage_bound: mxx_ir_core::IntExpr,
    /// Gaussian cutoff shared by helper and anchor errors.
    pub encoding_error_sigma: mxx_ir_core::RealExpr,
    /// Concrete common helper/anchor cutoff `B_chi`.
    pub encoding_error_bound: mxx_ir_core::IntExpr,
    /// Per-slot CRT scaling polynomials.
    pub slot_scales: Vec<mxx_ir_core::IntExpr>,
    /// Full modulus used by the BGG layout.
    pub layout_modulus: mxx_ir_core::IntExpr,
    /// Ring dimension used by the BGG layout.
    pub layout_ring_dimension: mxx_ir_core::IntExpr,
    /// Gadget base used by the BGG layout.
    pub layout_gadget_base: mxx_ir_core::IntExpr,
}

/// Frozen producer graph and its public specification hash.
///
/// The graph hash identifies the producer declaration.  Preimages remain typed
/// `Preimage` values at the artifact boundary. AverageCase derives its
/// structural model directly from these setup inputs and accepts no detached
/// proof attachment.
#[derive(Clone)]
pub struct RefreshProducerAttestation {
    graph: Graph,
    producer_spec_hash: mxx_ir_core::artifact::SpecHash,
}

impl RefreshProducerAttestation {
    /// Returns the canonical hash of the frozen producer graph.
    pub fn producer_spec_hash(&self) -> &mxx_ir_core::artifact::SpecHash {
        &self.producer_spec_hash
    }

    /// Returns the frozen graph whose outputs are named by the declaration.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

#[derive(Clone)]
/// Runtime setup material imported from a validated producer declaration.
///
/// Private vectors and preimages remain crate-visible only. Consumers receive
/// them through [`RefreshCompiler::bind_imported_setup`] after all manifest,
/// graph, and identity checks succeed.
pub struct ImportedRefreshSetup {
    production_id: ProductionId,
    parameters: RefreshSetupParameters,
    pub(crate) state: BggEncodingWire,
    pub(crate) a_prime: Mat,
    pub(crate) public_b: Mat,
    pub(crate) scaled_fresh: Vec<BggEncodingWire>,
    pub(crate) masks: Vec<BggEncodingWire>,
    pub(crate) decoder_base: BggEncodingWire,
    pub(crate) preimages: Vec<mxx_dsl::Preimage>,
    declaration: RefreshPreprocessingDeclaration,
    attestation: RefreshProducerAttestation,
}
impl std::fmt::Debug for ImportedRefreshSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedRefreshSetup")
            .field("identity", &self.declaration.identity)
            .field("slot_count", &self.masks.len())
            .finish()
    }
}

/// Immutable producer for the trusted refresh preprocessing graph.
///
/// Building this value samples setup artifacts, freezes the graph, and
/// validates the declaration and relation attestation. The producer retains
/// private setup internally; callers should export its declaration and use
/// [`ImportedRefreshSetup::import`] at an execution boundary.
pub struct RefreshPreprocessingProducer {
    request: RefreshPreprocessingRequest,
    wires: RefreshPreprocessingWires,
    built: BuiltGraph,
    declaration: RefreshPreprocessingDeclaration,
    attestation: RefreshProducerAttestation,
}

impl RefreshPreprocessingProducer {
    /// Builds, attests, and validates the immutable preprocessing graph.
    pub fn build(request: RefreshPreprocessingRequest) -> Result<Self, RefreshSetupError> {
        request.parameters.validate_dimensions()?;
        let decoder_preimage_bound = request.parameters.resolve_decoder_preimage_bound()?;
        crate::ensure_ciphertext_only(&request.state)?;
        request.prf.validate_for(&request.parameters)?;
        let p = &request.parameters;
        let ring = p.layout.ring();
        let masks = request.prf.aggregate_masks(&request.compiler, p.base_p)?;
        // `A'` is the public random matrix used in every slot target
        // `T_t = A_{sum,t} - mu_t A'`; its domain tag binds it to this refresh.
        let a_prime = ring.hash_matrix(
            request.hash_key.clone(),
            HashTag::from(
                format!("mxx-exponent-lut/refresh/a-prime/v1/{}", hex(&p.refresh_id)).into_bytes(),
            ),
            (p.layout.secret_dimension, p.layout.public_key_columns()),
        );
        let trapdoor_digits = p.layout.digit_count;
        let trapdoor = ring.sample_trapdoor(
            p.layout.secret_dimension,
            p.decoder_sigma.clone(),
            p.layout.gadget_base.clone(),
            trapdoor_digits,
            decoder_preimage_bound.clone(),
        );
        let public_b = trapdoor.public_matrix();
        let b_columns = p
            .component_count
            .checked_mul(p.layout.digit_count + 2)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let decoder_base = shared_decoder_base(
            &ring,
            request.secret.clone(),
            public_b.clone(),
            p.encoding_error_sigma.clone(),
            p.encoding_error_bound.clone(),
        );
        let mut scales = Vec::new();
        for slot in 0..masks.len() {
            // `scale = mu_t = q/q_t` is the scalar represented by the
            // setup-fixed RNS scaling LUT for CRT slot t.
            let scale = ring.polynomial([p.refresh.scale_expression(slot)?]);
            scales.push(scale);
        }
        let scaled_fresh = aggregate_refresh_fresh_error_per_slot(
            &request.compiler,
            p.base_p,
            &request.prf.fresh_error,
            scales.clone(),
        )?;
        let mask_public_family =
            Family::pack(masks.iter().map(|mask| mask.pubkey.matrix.clone()).collect())?;
        let scale_family = Family::pack(scales)?;
        let scaled_fresh_public_family =
            Family::pack(scaled_fresh.iter().map(|fresh| fresh.pubkey.matrix.clone()).collect())?;
        // Build every slot target in one structural loop, then sample all
        // preimages in a second structural loop. Each family element computes
        // `T_t = A_t + A_{m,t} + A_{e,t} - mu_t A'`; the slot index is structural,
        // never plaintext. The target family is Zip and the trapdoor is
        // captured once as a Broadcast input.
        let target_family = Family::<Mat>::parallel_zip_many_values(
            vec![mask_public_family, scale_family, scaled_fresh_public_family],
            |_, mut items| {
                let fresh_public = items.pop().expect("scaled fresh family");
                let scale = items.pop().expect("scale family");
                let mask_public = items.pop().expect("mask family");
                let scaled_state = request.compiler.bgg.large_scalar_mul(&request.state, &scale);
                scaled_state.pubkey.matrix + mask_public + fresh_public - scale * a_prime.clone()
            },
        )?;
        // Sample one preimage K_t per target; the trapdoor is captured once
        // and broadcast, so the resulting relation is `B K_t = T_t`.
        let preimage_family = target_family.parallel_map_values(|_, target| {
            trapdoor.sample_preimage(target, (b_columns, p.layout.public_key_columns()))
        })?;
        let ks = (0..masks.len()).map(|slot| preimage_family.get_static(slot)).collect();
        let names = canonical_names(
            p,
            &request.prf.contract,
            request.prf.layout_id,
            &decoder_preimage_bound,
        );
        let declaration = make_declaration(
            p,
            &request.prf.contract,
            names.clone(),
            mxx_ir_core::artifact::SpecHash([0; 32]),
            request.prf.layout_id,
            &decoder_preimage_bound,
        );
        let wires = RefreshPreprocessingWires {
            state: request.state.clone(),
            a_prime,
            public_b,
            scaled_fresh,
            masks,
            decoder_base,
            preimages: ks,
            names,
            declaration,
        };
        let names = wires.names.clone();
        let mut context = DslContext::new("mxx-exponent-lut-refresh-setup");
        context = add_setup_outputs(context, &wires, &names)?;
        let built = context.build()?;
        let producer_spec_hash = spec_hash(&built.graph, &ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let declaration = make_declaration(
            p,
            &request.prf.contract,
            names,
            producer_spec_hash.clone(),
            request.prf.layout_id,
            &decoder_preimage_bound,
        );
        let mut wires = wires;
        wires.declaration = declaration.clone();
        let attestation =
            RefreshProducerAttestation { graph: built.graph.clone(), producer_spec_hash };
        let producer = Self { request, wires, built, declaration, attestation };
        producer.validate_built()?;
        Ok(producer)
    }
    /// Returns the frozen preprocessing graph and its artifact outputs.
    pub fn built(&self) -> &BuiltGraph {
        &self.built
    }

    /// Returns the graph attestation used during later import.
    pub fn attestation(&self) -> &RefreshProducerAttestation {
        &self.attestation
    }

    /// Returns the public declaration describing this producer's artifacts.
    pub fn declaration(&self) -> &RefreshPreprocessingDeclaration {
        &self.declaration
    }

    /// Checks the actual setup equations and all canonical output roles.
    pub fn validate_built(&self) -> Result<(), RefreshSetupError> {
        self.declaration.validate_built(&self.wires)?;
        self.declaration.validate_graph(&self.attestation)?;
        let p = &self.request.parameters;
        let decoder_preimage_bound = p.resolve_decoder_preimage_bound()?;
        let w = &self.wires;
        if self.declaration.identity !=
            identity_digest(
                p,
                &self.request.prf.contract,
                self.request.prf.layout_id,
                &decoder_preimage_bound,
            ) ||
            self.declaration.names !=
                canonical_names(
                    p,
                    &self.request.prf.contract,
                    self.request.prf.layout_id,
                    &decoder_preimage_bound,
                )
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        if w.masks.is_empty() ||
            w.masks.len() != p.refresh.crt_plaintext_moduli.len() ||
            w.preimages.len() != w.masks.len() ||
            w.scaled_fresh.len() != w.masks.len()
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        crate::ensure_ciphertext_only(&w.state)?;
        let ring = p.layout.ring();
        let cols = p.layout.public_key_columns();
        let b_columns = p.component_count * (p.layout.digit_count + 2);
        if !same_matrix_type(w.state.vector.matrix_type(), &ring.matrix_type((1, cols))) ||
            !same_matrix_type(
                w.state.pubkey.matrix.matrix_type(),
                &ring.matrix_type((p.component_count, cols)),
            ) ||
            !same_matrix_type(
                w.public_b.matrix_type(),
                &ring.matrix_type((p.component_count, b_columns)),
            ) ||
            !same_matrix_type(
                w.a_prime.matrix_type(),
                &ring.matrix_type((p.component_count, cols)),
            )
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let b_handle = w.public_b.value_handle().clone();
        let base = &w.decoder_base;
        crate::ensure_ciphertext_only(base)?;
        if base.pubkey.matrix.value_handle() != &b_handle ||
            base.pubkey.reveal_plaintext ||
            !same_matrix_type(base.vector.matrix_type(), &ring.matrix_type((1, b_columns))) ||
            !same_matrix_type(
                base.pubkey.matrix.matrix_type(),
                &ring.matrix_type((p.component_count, b_columns)),
            )
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        for (slot, (mask, k)) in w.masks.iter().zip(w.preimages.iter()).enumerate() {
            crate::ensure_ciphertext_only(mask)?;
            if !same_matrix_type(mask.vector.matrix_type(), &ring.matrix_type((1, cols))) ||
                !same_matrix_type(
                    mask.pubkey.matrix.matrix_type(),
                    &ring.matrix_type((p.component_count, cols)),
                ) ||
                !same_matrix_type(k.matrix_type(), &ring.matrix_type((b_columns, cols)))
            {
                return Err(RefreshSetupError::InvalidManifest);
            }
            // Reconstruct the same `mu_t = q/q_t` scaling used to define
            // `T_t`; this check prevents a preimage from being paired with a
            // target from another CRT modulus.
            let scale = p.layout.ring().polynomial([p.refresh.scale_expression(slot)?]);
            let combined = self.request.compiler.bgg.add(
                &self
                    .request
                    .compiler
                    .bgg
                    .add(&self.request.compiler.bgg.large_scalar_mul(&w.state, &scale), mask)?,
                &w.scaled_fresh[slot],
            )?;
            let target = combined.pubkey.matrix - scale * w.a_prime.clone();
            if !same_matrix_type(
                w.public_b.clone().mul_small_rhs(k.clone()).matrix_type(),
                target.matrix_type(),
            ) {
                return Err(RefreshSetupError::InvalidManifest);
            }
        }
        Ok(())
    }

    /// Applies the declaration's role/type checks to an exported manifest.
    pub fn finalize_export_manifest(
        &self,
        manifest: &mut Manifest,
    ) -> Result<(), RefreshSetupError> {
        self.declaration.finalize_export_manifest(manifest, &self.request.parameters)
    }
}

/// Builds the one shared decoder anchor `b=sB+e_B` used by every CRT slot.
fn shared_decoder_base(
    ring: &mxx_dsl::Ring,
    secret: Mat,
    public_b: Mat,
    encoding_error_sigma: mxx_ir_core::RealExpr,
    encoding_error_bound: mxx_ir_core::IntExpr,
) -> BggEncodingWire {
    let b_columns = public_b.matrix_type().columns.clone();
    let anchor_error = ring.gaussian((1, b_columns), encoding_error_sigma, encoding_error_bound);
    BggEncodingWire {
        vector: secret * public_b.clone() + anchor_error,
        pubkey: BggPublicKeyWire { matrix: public_b, reveal_plaintext: false },
        plaintext: None,
    }
}

/// Public-key-only counterpart of the refresh PRF routing performed online on
/// complete encodings.  The canonical label order is
/// `[slot][component][coefficient][base-p digit]`, followed by the shared
/// fresh-error group.  Keeping this separate is essential for benchmark stage
/// fidelity: preprocessing must not pull private encoding-vector work into its
/// graph merely to obtain the public matrices used by the decoder targets.
fn aggregate_public_refresh_prf(
    compiler: &ExponentLutPublicKeyCompiler,
    outputs: Family<Mat>,
    base_p: usize,
    slot_count: usize,
    component_count: usize,
    coefficient_count: usize,
    mask_digit_count: usize,
    fresh_digit_count: usize,
    scales: Vec<Mat>,
) -> Result<(Vec<Mat>, Vec<Mat>), RefreshSetupError> {
    let mask_group = component_count
        .checked_mul(coefficient_count)
        .and_then(|value| value.checked_mul(mask_digit_count))
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let fresh_group = component_count
        .checked_mul(coefficient_count)
        .and_then(|value| value.checked_mul(fresh_digit_count))
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let mask_total =
        slot_count.checked_mul(mask_group).ok_or(RefreshSetupError::InvalidManifest)?;
    let total = mask_total.checked_add(fresh_group).ok_or(RefreshSetupError::InvalidManifest)?;
    if slot_count == 0 || scales.len() != slot_count || !family_count_is(&outputs, total)? {
        return Err(RefreshSetupError::InvalidManifest);
    }

    let mask_indices = Parallel::range(mask_total).map_values(|index| index.as_int())?;
    let mask_outputs = outputs.clone().parallel_gather(mask_indices)?;
    let routed_masks = route_public_prf_family(
        compiler,
        base_p,
        mask_outputs,
        component_count,
        coefficient_count,
        mask_digit_count,
        None,
    )?;
    let masks = reduce_public_family_segments(routed_masks, slot_count, mask_group)?;

    let fresh_indices = Parallel::range(fresh_group)
        .map_values(|index| Int::constant(mask_total).add(index.as_int()))?;
    let fresh_outputs = outputs.parallel_gather(fresh_indices)?;
    let repeated_count =
        fresh_group.checked_mul(slot_count).ok_or(RefreshSetupError::InvalidManifest)?;
    let repeated_fresh_indices = Parallel::range(repeated_count).map_values(|index| {
        let flat = index.as_int();
        let quotient = flat.clone().div(Int::constant(fresh_group));
        flat.sub(quotient.mul(Int::constant(fresh_group)))
    })?;
    let slot_indices = Parallel::range(repeated_count)
        .map_values(|index| index.as_int().div(Int::constant(fresh_group)))?;
    let repeated_fresh = fresh_outputs.parallel_gather(repeated_fresh_indices)?;
    let repeated_scales = Family::pack(scales)?.parallel_gather(slot_indices)?;
    let routed_fresh = route_public_prf_family(
        compiler,
        base_p,
        repeated_fresh,
        component_count,
        coefficient_count,
        fresh_digit_count,
        Some(repeated_scales),
    )?;
    let fresh = reduce_public_family_segments(routed_fresh, slot_count, fresh_group)?;
    if !family_count_is(&masks, slot_count)? || !family_count_is(&fresh, slot_count)? {
        return Err(RefreshSetupError::InvalidManifest);
    }
    Ok((
        (0..slot_count).map(|slot| masks.get_static(slot)).collect(),
        (0..slot_count).map(|slot| fresh.get_static(slot)).collect(),
    ))
}

fn route_public_prf_family(
    compiler: &ExponentLutPublicKeyCompiler,
    base_p: usize,
    public_keys: Family<Mat>,
    component_count: usize,
    coefficient_count: usize,
    digit_count: usize,
    scales: Option<Family<Mat>>,
) -> Result<Family<Mat>, RefreshSetupError> {
    let has_scales = scales.is_some();
    let mut families = vec![public_keys];
    if let Some(scales) = scales {
        families.push(scales);
    }
    Family::<Mat>::try_parallel_zip_many_values(families, |index, mut inputs| {
        let scale = has_scales.then(|| inputs.pop().expect("scale family"));
        let public_key = inputs.pop().ok_or(mxx_dsl::DslError::Schema)?;
        let flat = index.expression();
        let group_size = component_count
            .checked_mul(coefficient_count)
            .and_then(|value| value.checked_mul(digit_count))
            .ok_or(mxx_dsl::DslError::Schema)?;
        let group_quotient = mxx_ir_core::IntExpr::Div(
            Box::new(flat.clone()),
            Box::new(mxx_ir_core::IntExpr::constant(group_size)),
        );
        let within_group = mxx_ir_core::IntExpr::Sub(
            Box::new(flat),
            Box::new(mxx_ir_core::IntExpr::Mul(
                Box::new(group_quotient),
                Box::new(mxx_ir_core::IntExpr::constant(group_size)),
            )),
        )
        .canonicalize();
        let digit_quotient = mxx_ir_core::IntExpr::Div(
            Box::new(within_group.clone()),
            Box::new(mxx_ir_core::IntExpr::constant(digit_count)),
        );
        let digit = mxx_ir_core::IntExpr::Sub(
            Box::new(within_group.clone()),
            Box::new(mxx_ir_core::IntExpr::Mul(
                Box::new(digit_quotient.clone()),
                Box::new(mxx_ir_core::IntExpr::constant(digit_count)),
            )),
        )
        .canonicalize();
        let coefficient_quotient = mxx_ir_core::IntExpr::Div(
            Box::new(digit_quotient.clone()),
            Box::new(mxx_ir_core::IntExpr::constant(coefficient_count)),
        );
        let coefficient = mxx_ir_core::IntExpr::Sub(
            Box::new(digit_quotient),
            Box::new(mxx_ir_core::IntExpr::Mul(
                Box::new(coefficient_quotient),
                Box::new(mxx_ir_core::IntExpr::constant(coefficient_count)),
            )),
        )
        .canonicalize();
        let component = mxx_ir_core::IntExpr::Div(
            Box::new(within_group),
            Box::new(mxx_ir_core::IntExpr::constant(
                coefficient_count.checked_mul(digit_count).ok_or(mxx_dsl::DslError::Schema)?,
            )),
        )
        .canonicalize();
        let public_type = public_key.matrix_type();
        let ring =
            mxx_dsl::Ring::new(public_type.modulus.clone(), public_type.ring_dimension.clone());
        let scalar = ring.constant(
            (1, 1),
            mxx_ir_core::node::ConstantMatrix::PowerOfBase {
                base: mxx_ir_core::IntExpr::constant(base_p),
                exponent: digit,
            },
        ) * ring.constant(
            (1, 1),
            mxx_ir_core::node::ConstantMatrix::Rotation { exponent: coefficient },
        );
        let route = scalar *
            ring.constant(
                (public_type.rows.clone(), 1),
                mxx_ir_core::node::ConstantMatrix::UnitColumn {
                    index: mxx_ir_core::IntExpr::constant(1),
                },
            ) *
            ring.constant(
                (1, public_type.columns.clone()),
                mxx_ir_core::node::ConstantMatrix::UnitRow { index: component },
            );
        let target = match scale {
            Some(scale) => scale * route,
            None => route,
        };
        Ok(compiler
            .public_key
            .matrix_mul(&BggPublicKeyWire { matrix: public_key, reveal_plaintext: false }, &target)
            .matrix)
    })
    .map_err(RefreshSetupError::from)
}

fn reduce_public_family_segments(
    family: Family<Mat>,
    segment_count: usize,
    segment_size: usize,
) -> Result<Family<Mat>, RefreshSetupError> {
    if segment_count == 0 || segment_size == 0 {
        return Err(RefreshSetupError::InvalidManifest);
    }
    Parallel::range(segment_count)
        .try_map_values({
            let family = family.clone();
            move |segment| {
                let start = segment.as_int().mul(Int::constant(segment_size));
                let indices = Parallel::range(segment_size)
                    .map_values(|index| start.clone().add(index.as_int()))?;
                let values = family
                    .clone()
                    .parallel_gather(indices)
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                crate::encoding::balanced_sum_family(values).map_err(|_| mxx_dsl::DslError::Schema)
            }
        })
        .map_err(RefreshSetupError::from)
}

fn family_count_is(family: &Family<Mat>, expected: usize) -> Result<bool, RefreshSetupError> {
    Ok(family
        .count()
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidManifest)?
        .to_usize() ==
        Some(expected))
}
impl RefreshPreprocessingDeclaration {
    /// Rejects declarations that cannot represent the supported CRT refresh
    /// layout. This check is repeated at every declaration boundary so a
    /// depth-one manifest cannot enter through direct import.
    fn validate_slot_count(&self) -> Result<(), RefreshSetupError> {
        if self.slot_count < 2 {
            return Err(RefreshSetupError::InvalidManifest);
        }
        Ok(())
    }

    fn validate_prf_contract(
        &self,
        parameters: &RefreshSetupParameters,
    ) -> Result<(), RefreshSetupError> {
        declaration_contract(self).validate_for(parameters)?;
        if self.prf_terminal_form != SparseLwrPrfTerminalForm::RawScalar ||
            self.program_id == ExponentLutProgramId::from_digest([0; 32])
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        Ok(())
    }

    pub(crate) fn validate_built(
        &self,
        w: &RefreshPreprocessingWires,
    ) -> Result<(), RefreshSetupError> {
        self.validate_slot_count()?;
        if self.identity != w.declaration.identity ||
            self.names != w.names ||
            self.slot_count != w.masks.len() ||
            self.component_count == 0 ||
            self.prf_component_count == 0 ||
            self.coefficient_count == 0 ||
            self.mask_base_p_digit_count == 0 ||
            self.fresh_error_base_p_digit_count == 0 ||
            self.mask_statistical_security_bits == 0 ||
            self.gadget_digit_count == 0 ||
            self.component_count.checked_mul(self.gadget_digit_count) !=
                Some(self.prf_component_count) ||
            self.encoding_error_bound == 0.into()
        {
            Err(RefreshSetupError::IdentityMismatch)
        } else {
            Ok(())
        }
    }

    /// Validates every exported artifact's canonical name, shape, and
    /// confidentiality.  Private values may never carry a public hash.
    pub fn finalize_export_manifest(
        &self,
        manifest: &mut Manifest,
        parameters: &RefreshSetupParameters,
    ) -> Result<(), RefreshSetupError> {
        parameters.validate()?;
        self.validate_slot_count()?;
        self.validate_prf_contract(parameters)?;
        let decoder_preimage_bound = parameters.resolve_decoder_preimage_bound()?;
        if self.identity !=
            identity_digest(
                parameters,
                &declaration_contract(self),
                self.pbc_layout_id,
                &decoder_preimage_bound,
            ) ||
            self.names !=
                canonical_names(
                    parameters,
                    &declaration_contract(self),
                    self.pbc_layout_id,
                    &decoder_preimage_bound,
                ) ||
            self.slot_count != parameters.refresh.crt_plaintext_moduli.len() ||
            manifest.ir_version != IR_VERSION ||
            manifest.production_id.spec_hash != self.producer_spec_hash
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        validate_manifest(manifest, &self.names, parameters)
    }

    /// Validates the frozen graph hash and the declared artifact cardinality.
    pub fn validate_graph(
        &self,
        attestation: &RefreshProducerAttestation,
    ) -> Result<(), RefreshSetupError> {
        self.validate_slot_count()?;
        if self.component_count == 0 ||
            self.prf_component_count == 0 ||
            self.coefficient_count == 0 ||
            self.mask_base_p_digit_count == 0 ||
            self.fresh_error_base_p_digit_count == 0 ||
            self.mask_statistical_security_bits == 0 ||
            self.gadget_digit_count == 0 ||
            self.component_count.checked_mul(self.gadget_digit_count) !=
                Some(self.prf_component_count) ||
            self.encoding_error_bound == 0.into() ||
            self.slot_scales.len() != self.slot_count ||
            self.names.scaled_fresh.len() != self.slot_count ||
            self.names.masks.len() != self.slot_count ||
            self.names.preimages.len() != self.slot_count
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let actual = spec_hash(&attestation.graph, &ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        if actual != self.producer_spec_hash ||
            attestation.producer_spec_hash != self.producer_spec_hash
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        Ok(())
    }
}

impl ImportedRefreshSetup {
    /// The production identifier used for every imported artifact input.
    pub fn production_id(&self) -> &ProductionId {
        &self.production_id
    }

    /// Returns the validated setup parameters used by this import.
    pub fn parameters(&self) -> &RefreshSetupParameters {
        &self.parameters
    }

    /// Returns the declaration that was checked during import.
    pub fn declaration(&self) -> &RefreshPreprocessingDeclaration {
        &self.declaration
    }

    /// Imports preprocessing artifacts after validating their producer graph,
    /// declaration identity, confidentiality, and concrete matrix shapes.
    pub fn import(
        production_id: ProductionId,
        parameters: RefreshSetupParameters,
        declaration: RefreshPreprocessingDeclaration,
        attestation: &RefreshProducerAttestation,
        manifest: &Manifest,
    ) -> Result<Self, RefreshSetupError> {
        parameters.validate()?;
        declaration.validate_slot_count()?;
        declaration.validate_prf_contract(&parameters)?;
        let decoder_preimage_bound = parameters.resolve_decoder_preimage_bound()?;
        if manifest.production_id != production_id ||
            manifest.production_id.spec_hash != declaration.producer_spec_hash ||
            declaration.identity !=
                identity_digest(
                    &parameters,
                    &declaration_contract(&declaration),
                    declaration.pbc_layout_id,
                    &decoder_preimage_bound,
                ) ||
            declaration.refresh_id != parameters.refresh_id ||
            declaration.component_count != parameters.component_count ||
            declaration.prf_component_count != parameters.prf_component_count() ||
            declaration.coefficient_count != parameters.coefficient_count ||
            declaration.mask_base_p_digit_count != parameters.mask_base_p_digit_count ||
            declaration.fresh_error_base_p_digit_count !=
                parameters.fresh_error_base_p_digit_count ||
            declaration.mask_statistical_security_bits !=
                parameters.mask_statistical_security_bits ||
            declaration.gadget_digit_count != parameters.layout.digit_count ||
            declaration.encoding_error_sigma != parameters.encoding_error_sigma ||
            declaration.encoding_error_bound != parameters.encoding_error_bound ||
            declaration.names !=
                canonical_names(
                    &parameters,
                    &declaration_contract(&declaration),
                    declaration.pbc_layout_id,
                    &decoder_preimage_bound,
                )
        {
            return Err(RefreshSetupError::IdentityMismatch)
        };
        declaration.validate_graph(attestation)?;
        validate_manifest(manifest, &declaration.names, &parameters)?;
        let n = &declaration.names;
        let cols = parameters.layout.public_key_columns();
        let b_columns = parameters.component_count * (parameters.layout.digit_count + 2);
        // Reconstruct every imported value from the frozen producer output
        // type.  The manifest checks concrete dimensions, while this helper
        // preserves the producer's symbolic expressions in the new graph.
        // This is the artifact boundary: no live producer wire is reused.
        let imported = |name: &str,
                        confidentiality: ArtifactConfidentiality,
                        expected_family: bool,
                        expected: &mxx_ir_core::types::MatrixType|
         -> Result<Mat, RefreshSetupError> {
            let (matrix, count) = resolve_graph_output_matrix(
                &attestation.graph,
                name,
                confidentiality,
                expected_family,
                expected,
                None,
            )?;
            if count.is_some() {
                return Err(RefreshSetupError::InvalidManifest);
            }
            let ring = mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension.clone());
            Ok(ring.artifact_input(
                production_id.clone(),
                name.to_owned(),
                (matrix.rows.clone(), matrix.columns.clone()),
                confidentiality,
            ))
        };
        let state_vector = imported(
            &n.state_vector,
            ArtifactConfidentiality::Private,
            false,
            &parameters.layout.ring().matrix_type((1, cols)),
        )?;
        let state_public_matrix = imported(
            &n.state_public_matrix,
            ArtifactConfidentiality::Public,
            false,
            &parameters.layout.ring().matrix_type((parameters.component_count, cols)),
        )?;
        let a_prime = imported(
            &n.a_prime,
            ArtifactConfidentiality::Public,
            false,
            &parameters.layout.ring().matrix_type((parameters.component_count, cols)),
        )?;
        let public_b = imported(
            &n.public_matrix_b,
            ArtifactConfidentiality::Public,
            false,
            &parameters.layout.ring().matrix_type((parameters.component_count, b_columns)),
        )?;
        let state = BggEncodingWire {
            vector: state_vector,
            pubkey: BggPublicKeyWire { matrix: state_public_matrix, reveal_plaintext: false },
            plaintext: None,
        };
        let mut scaled_fresh = Vec::new();
        for pair in &n.scaled_fresh {
            let vector = imported(
                &pair.vector,
                ArtifactConfidentiality::Private,
                false,
                &parameters.layout.ring().matrix_type((1, cols)),
            )?;
            let public_matrix = imported(
                &pair.public_matrix,
                ArtifactConfidentiality::Public,
                false,
                &parameters.layout.ring().matrix_type((parameters.component_count, cols)),
            )?;
            scaled_fresh.push(BggEncodingWire {
                vector,
                pubkey: BggPublicKeyWire { matrix: public_matrix, reveal_plaintext: false },
                plaintext: None,
            });
        }
        let mut masks = Vec::new();
        let mut ks = Vec::new();
        for x in &n.masks {
            let vector = imported(
                &x.vector,
                ArtifactConfidentiality::Private,
                false,
                &parameters.layout.ring().matrix_type((1, cols)),
            )?;
            let public_matrix = imported(
                &x.public_matrix,
                ArtifactConfidentiality::Public,
                false,
                &parameters.layout.ring().matrix_type((parameters.component_count, cols)),
            )?;
            masks.push(BggEncodingWire {
                vector,
                pubkey: BggPublicKeyWire { matrix: public_matrix, reveal_plaintext: false },
                plaintext: None,
            });
        }
        let decoder_base = BggEncodingWire {
            vector: imported(
                &n.decoder_base_vector,
                ArtifactConfidentiality::Private,
                false,
                &parameters.layout.ring().matrix_type((1, b_columns)),
            )?,
            pubkey: BggPublicKeyWire { matrix: public_b.clone(), reveal_plaintext: false },
            plaintext: None,
        };
        for x in &n.preimages {
            // Keep K typed as a preimage across the artifact boundary. The
            // consumer therefore applies the bounded witness through `mul_small_rhs`
            // instead of materializing it as an untyped matrix.
            let matrix = resolve_graph_output_preimage(
                &attestation.graph,
                x,
                ArtifactConfidentiality::Private,
                &parameters.layout.ring().matrix_type((b_columns, cols)),
            )?;
            let ring = mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension.clone());
            ks.push(ring.preimage_artifact_input(
                production_id.clone(),
                x.to_owned(),
                (matrix.rows.clone(), matrix.columns.clone()),
                decoder_preimage_bound.clone(),
                ArtifactConfidentiality::Private,
            ));
        }
        Ok(Self {
            production_id,
            parameters,
            state,
            a_prime,
            public_b,
            scaled_fresh,
            masks,
            decoder_base,
            preimages: ks,
            declaration,
            attestation: attestation.clone(),
        })
    }
}

impl RefreshCompiler {
    /// Binds an imported setup to the runtime refresh compiler.
    ///
    /// The compiler's CRT configuration must match the imported declaration;
    /// the returned manifest exposes only validated runtime wiring.
    pub fn bind_imported_setup(
        &self,
        compiler: &ExponentLutEncodingCompiler,
        setup: &ImportedRefreshSetup,
    ) -> Result<RefreshSetupManifest, RefreshSetupError> {
        if setup.parameters.refresh.crt_plaintext_moduli != self.crt_plaintext_moduli ||
            setup.parameters.refresh.reconstruction_coefficients !=
                self.reconstruction_coefficients ||
            setup.parameters.refresh.full_modulus != self.full_modulus
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        setup.declaration.validate_graph(&setup.attestation)?;
        Ok(self.bind_imported_wires(
            compiler,
            setup.state.clone(),
            setup.a_prime.clone(),
            setup.public_b.clone(),
            setup.scaled_fresh.clone(),
            setup.masks.clone(),
            setup.decoder_base.clone(),
            setup.preimages.clone(),
        )?)
    }
}

/// Derives all preprocessing artifact names from the public setup identity.
///
/// Slot-indexed names are emitted in CRT order, so an importer cannot silently
/// exchange two mask or preimage artifacts; the decoder-base vector is shared.
fn canonical_names(
    p: &RefreshSetupParameters,
    contract: &RefreshPrfContract,
    pbc_layout_id: crate::pbc::PbcLayoutId,
    decoder_preimage_bound: &mxx_ir_core::IntExpr,
) -> RefreshPreprocessingArtifactNames {
    let d = identity_digest(p, contract, pbc_layout_id, decoder_preimage_bound);
    let n = |r: &str, i: usize| format!("mxx-refresh-{}-{r}-{i}", hex(&d));
    let s = p.refresh.crt_plaintext_moduli.len();
    RefreshPreprocessingArtifactNames {
        state_vector: n("state-vector", 0),
        state_public_matrix: n("state-public", 0),
        a_prime: n("a-prime", 0),
        public_matrix_b: n("public-b", 0),
        scaled_fresh: (0..s)
            .map(|i| RefreshArtifactPairNames {
                vector: n("scaled-fresh-vector", i),
                public_matrix: n("scaled-fresh-public", i),
            })
            .collect(),
        masks: (0..s)
            .map(|i| RefreshArtifactPairNames {
                vector: n("mask-vector", i),
                public_matrix: n("mask-public", i),
            })
            .collect(),
        decoder_base_vector: n("decoder-base-vector", 0),
        preimages: (0..s).map(|i| n("preimage", i)).collect(),
    }
}
/// Constructs the declaration committed by a preprocessing producer.
fn make_declaration(
    p: &RefreshSetupParameters,
    contract: &RefreshPrfContract,
    n: RefreshPreprocessingArtifactNames,
    producer_spec_hash: mxx_ir_core::artifact::SpecHash,
    pbc_layout_id: crate::pbc::PbcLayoutId,
    decoder_preimage_bound: &mxx_ir_core::IntExpr,
) -> RefreshPreprocessingDeclaration {
    RefreshPreprocessingDeclaration {
        identity: identity_digest(p, contract, pbc_layout_id, decoder_preimage_bound),
        producer_spec_hash,
        pbc_layout_id,
        refresh_id: p.refresh_id,
        program_id: contract.program_id(),
        prf_q_l: contract.q_l(),
        prf_p: contract.p(),
        prf_lut_width: contract.lut_width(),
        prf_ring_dimension: contract.ring_dimension(),
        prf_terminal_form: contract.terminal_form(),
        prf_output_wire: contract.output_wire(),
        names: n,
        slot_count: p.refresh.crt_plaintext_moduli.len(),
        component_count: p.component_count,
        prf_component_count: p.prf_component_count(),
        coefficient_count: p.coefficient_count,
        mask_base_p_digit_count: p.mask_base_p_digit_count,
        fresh_error_base_p_digit_count: p.fresh_error_base_p_digit_count,
        mask_statistical_security_bits: p.mask_statistical_security_bits,
        gadget_digit_count: p.layout.digit_count,
        decoder_sigma: p.decoder_sigma.clone(),
        decoder_preimage_bound: decoder_preimage_bound.clone(),
        encoding_error_sigma: p.encoding_error_sigma.clone(),
        encoding_error_bound: p.encoding_error_bound.clone(),
        slot_scales: (0..p.refresh.crt_plaintext_moduli.len())
            .map(|slot| p.refresh.scale_expression(slot).expect("validated scale"))
            .collect(),
        layout_modulus: p.layout.modulus.clone(),
        layout_ring_dimension: p.layout.ring_dimension.clone(),
        layout_gadget_base: p.layout.gadget_base.clone(),
    }
}

/// Reconstructs the committed PRF contract from an imported declaration.
///
/// This is used only to recompute the declaration identity; runtime output
/// validation still originates from the independently constructed program.
fn declaration_contract(declaration: &RefreshPreprocessingDeclaration) -> RefreshPrfContract {
    RefreshPrfContract::from_parts(
        declaration.program_id,
        declaration.prf_output_wire,
        declaration.prf_terminal_form,
        declaration.prf_q_l,
        declaration.prf_p,
        declaration.prf_lut_width,
        declaration.prf_ring_dimension,
    )
}

/// Exports producer wires under their canonical public/private artifact names.
fn add_setup_outputs(
    mut context: DslContext,
    wires: &RefreshPreprocessingWires,
    names: &RefreshPreprocessingArtifactNames,
) -> Result<DslContext, RefreshSetupError> {
    // Private vectors and K_t remain private artifacts; A', B, and the public
    // projections are exported with the declaration's canonical roles.
    context = context
        .private_output(names.state_vector.clone(), wires.state.vector.clone())?
        .public_output(names.state_public_matrix.clone(), wires.state.pubkey.matrix.clone())?
        .public_output(names.a_prime.clone(), wires.a_prime.clone())?
        .public_output(names.public_matrix_b.clone(), wires.public_b.clone())?;
    for slot in 0..wires.scaled_fresh.len() {
        context = context
            .private_output(
                names.scaled_fresh[slot].vector.clone(),
                wires.scaled_fresh[slot].vector.clone(),
            )?
            .public_output(
                names.scaled_fresh[slot].public_matrix.clone(),
                wires.scaled_fresh[slot].pubkey.matrix.clone(),
            )?;
    }
    context = context
        .private_output(names.decoder_base_vector.clone(), wires.decoder_base.vector.clone())?;
    for slot in 0..wires.masks.len() {
        context = context
            .private_output(names.masks[slot].vector.clone(), wires.masks[slot].vector.clone())?
            .public_output(
                names.masks[slot].public_matrix.clone(),
                wires.masks[slot].pubkey.matrix.clone(),
            )
            .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?
            .private_preimage_output(
                names.preimages[slot].clone(),
                wires.preimages[slot].clone(),
            )?;
    }
    Ok(context)
}

/// Computes the canonical public identity for refresh preprocessing setup.
///
/// The digest includes dimensions, CRT data, PRF identity, decoder policy,
/// and public artifact schema; private vectors, support, schedules, and
/// plaintext values are deliberately excluded.
fn identity_digest(
    p: &RefreshSetupParameters,
    contract: &RefreshPrfContract,
    pbc_layout_id: crate::pbc::PbcLayoutId,
    decoder_preimage_bound: &mxx_ir_core::IntExpr,
) -> [u8; 32] {
    #[derive(Serialize)]
    struct Layout {
        modulus: mxx_ir_core::IntExpr,
        ring_dimension: mxx_ir_core::IntExpr,
        secret_dimension: usize,
        gadget_digit_count: usize,
        gadget_base: mxx_ir_core::IntExpr,
    }
    #[derive(Serialize)]
    struct Refresh {
        full_modulus: mxx_ir_core::IntExpr,
        crt_plaintext_moduli: Vec<mxx_ir_core::IntExpr>,
        reconstruction_coefficients: Vec<mxx_ir_core::IntExpr>,
    }
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        refresh_id: [u8; 32],
        program_id: ExponentLutProgramId,
        prf_q_l: usize,
        prf_p: usize,
        prf_lut_width: usize,
        prf_ring_dimension: usize,
        prf_terminal_form: SparseLwrPrfTerminalForm,
        prf_output_wire: crate::program::ProgramWireId,
        pbc_layout_id: crate::pbc::PbcLayoutId,
        name: &'a str,
        base_p: usize,
        component_count: usize,
        prf_component_count: usize,
        coefficient_count: usize,
        mask_base_p_digit_count: usize,
        fresh_error_base_p_digit_count: usize,
        mask_statistical_security_bits: usize,
        gadget_digit_count: usize,
        lut_width: usize,
        layout: Layout,
        refresh: Refresh,
        decoder_sigma: &'a mxx_ir_core::RealExpr,
        decoder_preimage_bound: &'a mxx_ir_core::IntExpr,
        encoding_error_sigma: &'a mxx_ir_core::RealExpr,
        encoding_error_bound: &'a mxx_ir_core::IntExpr,
    }
    let payload = Payload {
        schema: "mxx-exponent-lut/refresh-setup/v7",
        refresh_id: p.refresh_id,
        program_id: contract.program_id(),
        prf_q_l: contract.q_l(),
        prf_p: contract.p(),
        prf_lut_width: contract.lut_width(),
        prf_ring_dimension: contract.ring_dimension(),
        prf_terminal_form: contract.terminal_form(),
        prf_output_wire: contract.output_wire(),
        pbc_layout_id,
        name: &p.name,
        base_p: p.base_p,
        component_count: p.component_count,
        prf_component_count: p.prf_component_count(),
        coefficient_count: p.coefficient_count,
        mask_base_p_digit_count: p.mask_base_p_digit_count,
        fresh_error_base_p_digit_count: p.fresh_error_base_p_digit_count,
        mask_statistical_security_bits: p.mask_statistical_security_bits,
        gadget_digit_count: p.layout.digit_count,
        lut_width: p.lut_width,
        layout: Layout {
            modulus: p.layout.modulus.clone().canonicalize(),
            ring_dimension: p.layout.ring_dimension.clone().canonicalize(),
            secret_dimension: p.layout.secret_dimension,
            gadget_digit_count: p.layout.digit_count,
            gadget_base: p.layout.gadget_base.clone().canonicalize(),
        },
        refresh: Refresh {
            full_modulus: p.refresh.full_modulus.clone().canonicalize(),
            crt_plaintext_moduli: p
                .refresh
                .crt_plaintext_moduli
                .iter()
                .cloned()
                .map(|value| value.canonicalize())
                .collect(),
            reconstruction_coefficients: p
                .refresh
                .reconstruction_coefficients
                .iter()
                .cloned()
                .map(|value| value.canonicalize())
                .collect(),
        },
        decoder_sigma: &p.decoder_sigma,
        decoder_preimage_bound,
        encoding_error_sigma: &p.encoding_error_sigma,
        encoding_error_bound: &p.encoding_error_bound,
    };
    hash_canonical(&payload).expect("canonical refresh setup identity")
}
fn hex(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
/// Checks the imported artifact manifest against the declaration's role schema.
fn validate_manifest(
    m: &Manifest,
    n: &RefreshPreprocessingArtifactNames,
    p: &RefreshSetupParameters,
) -> Result<(), RefreshSetupError> {
    let decoder_preimage_bound = p
        .resolve_decoder_preimage_bound()?
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
    let ring_dimension = p
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|x| x.to_usize())
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let modulus = p
        .layout
        .modulus
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
    let matrix_type = |rows: usize, columns: usize| {
        ArtifactType::Matrix(ConcreteMatrixType {
            modulus: modulus.clone(),
            ring_dimension,
            rows,
            columns,
        })
    };
    let check_type = |name: &str,
                      confidentiality: ArtifactConfidentiality,
                      artifact_type: ArtifactType|
     -> Result<(), RefreshSetupError> {
        let artifact = m.artifacts.get(name).ok_or(RefreshSetupError::InvalidManifest)?;
        if artifact.confidentiality != confidentiality ||
            artifact.family_shape.is_some() ||
            artifact.artifact_type != artifact_type ||
            artifact.layout.is_some() ||
            (confidentiality == ArtifactConfidentiality::Private &&
                artifact.content_hash.is_some()) ||
            (confidentiality == ArtifactConfidentiality::Public &&
                artifact.content_hash.is_none())
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        Ok(())
    };
    let check =
        |name: &str, confidentiality: ArtifactConfidentiality, rows: usize, columns: usize| {
            check_type(name, confidentiality, matrix_type(rows, columns))
        };
    let check_preimage = |name: &str, rows: usize, columns: usize| {
        check_type(
            name,
            ArtifactConfidentiality::Private,
            ArtifactType::Preimage {
                matrix: ConcreteMatrixType {
                    modulus: modulus.clone(),
                    ring_dimension,
                    rows,
                    columns,
                },
                max_coefficient_bound: decoder_preimage_bound.clone(),
            },
        )
    };
    let slots = p.refresh.crt_plaintext_moduli.len();
    let expected_artifact_count = 5usize
        .checked_add(slots.checked_mul(5).ok_or(RefreshSetupError::InvalidManifest)?)
        .ok_or(RefreshSetupError::InvalidManifest)?;
    if m.artifacts.len() != expected_artifact_count {
        return Err(RefreshSetupError::InvalidManifest);
    }
    if n.scaled_fresh.len() != slots || n.masks.len() != slots || n.preimages.len() != slots {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let cols = p.layout.public_key_columns();
    let b_columns = p
        .component_count
        .checked_mul(p.layout.digit_count + 2)
        .ok_or(RefreshSetupError::InvalidManifest)?;
    check(&n.state_vector, ArtifactConfidentiality::Private, 1, cols)?;
    check(&n.state_public_matrix, ArtifactConfidentiality::Public, p.component_count, cols)?;
    check(&n.a_prime, ArtifactConfidentiality::Public, p.component_count, cols)?;
    check(&n.public_matrix_b, ArtifactConfidentiality::Public, p.component_count, b_columns)?;
    for fresh in &n.scaled_fresh {
        check(&fresh.vector, ArtifactConfidentiality::Private, 1, cols)?;
        check(&fresh.public_matrix, ArtifactConfidentiality::Public, p.component_count, cols)?;
    }
    for mask in &n.masks {
        check(&mask.vector, ArtifactConfidentiality::Private, 1, cols)?;
        check(&mask.public_matrix, ArtifactConfidentiality::Public, p.component_count, cols)?;
    }
    check(&n.decoder_base_vector, ArtifactConfidentiality::Private, 1, b_columns)?;
    for preimage in &n.preimages {
        check_preimage(preimage, b_columns, cols)?;
    }
    mxx_ir_core::artifact::validate_manifest(m).map_err(|_| RefreshSetupError::InvalidManifest)?;
    Ok(())
}

/// The graph values produced by [`build_refresh_verification`].  This is a
/// verification circuit boundary, not a decryption API: the caller supplies
/// the expected plaintext and decides how to consume the decoded outputs.
pub struct RefreshVerification {
    residual: Mat,
    /// The exact scalar matrix passed as the input to each threshold decoder,
    /// in public-key-column order.  Keeping these wires prevents consumers
    /// from reconstructing an equivalent-looking slice and accidentally
    /// binding the correctness checker to a different graph operand.
    decoder_residuals: Vec<Mat>,
    decoded: Vec<Bool>,
}

impl RefreshVerification {
    /// The exact residual `c - sA + mu*(tG)`, where `s` is the mask secret
    /// and `t` is the payload secret.
    pub fn residual(&self) -> &Mat {
        &self.residual
    }

    /// Returns the exact threshold-decoder operands, one 1x1 matrix per
    /// public-key column, in column order.
    ///
    /// These are the same wires consumed by the `ThresholdDecode` nodes that
    /// produce [`Self::decoded`].  The returned slice is borrowed from the
    /// opaque verification object; callers cannot replace the operands used
    /// by the verification graph.
    pub fn decoder_residuals(&self) -> &[Mat] {
        &self.decoder_residuals
    }

    /// Checks that a consumer is binding the decoder targets to these exact
    /// graph operands rather than to merely same-shaped residuals.
    ///
    /// This is intentionally an identity check on DSL wires, not a value
    /// comparison.  A mismatching wire is rejected before a correctness
    /// checker or integration test can attach a threshold-decoder target to
    /// an unrelated expression.
    pub fn validate_decoder_residuals(&self, residuals: &[Mat]) -> Result<(), RefreshSetupError> {
        if residuals.len() != self.decoder_residuals.len() ||
            residuals.iter().zip(&self.decoder_residuals).any(|(candidate, expected)| {
                candidate.value_handle() != expected.value_handle() ||
                    candidate.matrix_type() != expected.matrix_type()
            })
        {
            return Err(RefreshSetupError::DecoderResidualMismatch);
        }
        Ok(())
    }

    /// Threshold-decoded residual indicators (`false` is the expected result).
    ///
    /// The output is ordered column-major: all `decode_length` coefficients
    /// decoded from residual column 0, followed by all coefficients from
    /// column 1, and so on.  Therefore the length is
    /// `public_key_columns * decode_length`.
    pub fn decoded(&self) -> &[Bool] {
        &self.decoded
    }

    /// Adds the residual and each threshold-decoded Boolean indicator to a DSL
    /// context as private outputs.  This is only an output-construction
    /// helper; it does not establish a correctness claim or decrypt a
    /// ciphertext.
    pub fn add_outputs(
        &self,
        context: DslContext,
        residual_name: impl Into<String>,
        decoded_prefix: impl Into<String>,
    ) -> Result<DslContext, RefreshSetupError> {
        let mut context = context.private_output(residual_name, self.residual.clone())?;
        // Export each scalar decoder operand under its own canonical name.  A
        // correctness target must bind to one exact column, not to the
        // aggregate residual matrix (which would make a same-shaped column
        // substitution indistinguishable to a downstream checker).
        for (column, residual) in self.decoder_residuals.iter().enumerate() {
            context = context
                .private_output(format!("refresh-decoder-residual-{column}"), residual.clone())?;
        }
        let decoded_prefix = decoded_prefix.into();
        for (index, value) in self.decoded.iter().enumerate() {
            context = context.bool_output(format!("{decoded_prefix}_{index}"), value.clone())?;
        }
        Ok(context)
    }
}

/// Builds the noiseless refresh verification residual and threshold decoder.
///
/// The expected plaintext is explicit and is never read from
/// `BggEncodingWire::plaintext`.  All arithmetic is ordinary matrix
/// multiplication, matching the separate-secret BGG+ relation
/// `c' = sA' - X^w*(tG) + e'`: the mask secret `s` is used for the public-key
/// term and the payload secret `t` is used for the gadget term. The supplied
/// `expected_plaintext` is the scalar/monomial `X^w` multiplier in that
/// residual, so the noiseless check is `c' - sA' + X^w(tG) = 0`.
///
/// Threshold decoding is scalar-only.  Since a BGG+ residual has one row and
/// one column per public-key column, this function slices each concrete
/// residual column to a 1x1 matrix before decoding it.  The returned Boolean
/// values are flattened in column-major, then coefficient order.  A `true`
/// value means that the corresponding residual decoded as nonzero.
pub fn build_refresh_verification(
    encoding: &BggEncodingWire,
    mask_secret: &Mat,
    payload_secret: &Mat,
    expected_plaintext: &Mat,
    gadget_base: impl Into<mxx_ir_core::IntExpr>,
    gadget_digit_count: usize,
    plaintext_modulus: impl Into<mxx_ir_core::IntExpr>,
    decode_length: usize,
) -> Result<RefreshVerification, RefreshSetupError> {
    crate::ensure_ciphertext_only(encoding)?;
    let scalar = expected_plaintext.matrix_type();
    if scalar.rows != 1.into() || scalar.columns != 1.into() {
        return Err(RefreshSetupError::InvalidParameters("expected plaintext must be 1x1"));
    }
    let gadget = encoding.pubkey.matrix.matrix_type();
    let expected_vector = mxx_ir_core::types::MatrixType {
        modulus: gadget.modulus.clone(),
        ring_dimension: gadget.ring_dimension.clone(),
        rows: 1.into(),
        columns: gadget.columns.clone(),
    };
    let secret_shape_matches = |secret: &Mat| {
        secret.matrix_type().rows == 1.into() &&
            secret.matrix_type().columns.evaluate(&ParamEnv::default()).ok() ==
                gadget.rows.evaluate(&ParamEnv::default()).ok() &&
            secret.matrix_type().modulus.evaluate(&ParamEnv::default()).ok() ==
                gadget.modulus.evaluate(&ParamEnv::default()).ok() &&
            secret.matrix_type().ring_dimension.evaluate(&ParamEnv::default()).ok() ==
                gadget.ring_dimension.evaluate(&ParamEnv::default()).ok()
    };
    if !same_matrix_type(encoding.vector.matrix_type(), &expected_vector) ||
        !secret_shape_matches(mask_secret) ||
        !secret_shape_matches(payload_secret)
    {
        return Err(RefreshSetupError::InvalidParameters(
            "mask and payload secrets must both be 1 x gadget-row matrices compatible with the encoding",
        ));
    }
    let public_key_columns = encoding
        .pubkey
        .matrix
        .matrix_type()
        .columns
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(RefreshSetupError::InvalidParameters(
            "public-key column count must be concrete and fit usize",
        ))?;
    if public_key_columns == 0 || decode_length == 0 {
        return Err(RefreshSetupError::InvalidParameters(
            "verification requires positive public-key columns and decode length",
        ));
    }
    let decoded_count = public_key_columns
        .checked_mul(decode_length)
        .ok_or(RefreshSetupError::InvalidParameters("decoded output count overflow"))?;
    let ring = mxx_dsl::Ring::new(
        encoding.pubkey.matrix.matrix_type().modulus.clone(),
        encoding.pubkey.matrix.matrix_type().ring_dimension.clone(),
    );
    let g = ring.gadget(mask_secret.matrix_type().columns.clone(), gadget_base, gadget_digit_count);
    let residual = encoding.vector.clone() - mask_secret.clone() * encoding.pubkey.matrix.clone() +
        expected_plaintext.clone() * (payload_secret.clone() * g);
    let plaintext_modulus = plaintext_modulus.into();
    let mut decoder_residuals = Vec::with_capacity(public_key_columns);
    let mut decoded = Vec::with_capacity(decoded_count);
    for column in 0..public_key_columns {
        let scalar_residual = residual.clone().slice(
            None,
            Some(mxx_ir_core::node::IndexRange { start: column.into(), end: (column + 1).into() }),
        );
        decoder_residuals.push(scalar_residual.clone());
        decoded.extend(
            scalar_residual.threshold_decode_bools(plaintext_modulus.clone(), decode_length),
        );
    }
    Ok(RefreshVerification { residual, decoder_residuals, decoded })
}

/// Exact matrix facts exported for request-level external-input validation.
/// BuiltGraph, declaration, and attestation validation occurs while the
/// simulation bundle is constructed; these facts do not define operational
/// noise bounds, which are owned by the Exponent-LUT noise snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshSimulationMatrixInputMetadata {
    /// Exclusive coefficient bound inferred from the producer matrix, when
    /// the input is a non-constant polynomial.
    pub canonical_coefficient_exclusive_upper_bound: Option<mxx_ir_core::IntExpr>,
    /// Whether every coefficient in the input polynomial is constant.
    pub is_constant_polynomial: bool,
}

/// One exact residual/decoder linkage produced by the verification graph.
/// The selected decoder operand is retained as a frozen scoped wire, rather
/// than reconstructed from an equivalent-looking slice by a caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshSimulationDecoderTarget {
    /// Stable identifier for this verification target.
    pub target_id: String,
    /// Canonical verification-graph output exporting this target's scalar
    /// residual.  This is intentionally distinct for every public-key column.
    pub residual_output_name: String,
    /// Semantic anchor of the residual output in the verification graph.
    pub residual_anchor: String,
    /// Frozen residual wire selected by the graph attestation.
    pub residual: ScopedWireRef,
    /// Exact type of [`Self::residual`].
    pub residual_type: WireType,
    /// Frozen scalar residual supplied to the threshold decoder.
    pub decoder: ScopedWireRef,
    /// Node implementing the decoder operation.
    pub decoder_node: NodeId,
    /// Exact output type of [`Self::decoder`].
    pub decoder_output_type: WireType,
    /// Canonical verification output containing the decoded indicator.
    pub decoded_output_name: String,
    /// Semantic anchor of the decoder input/output relation.
    pub decoder_anchor: String,
    /// Plaintext modulus used by threshold decoding.
    pub plaintext_modulus: mxx_ir_core::IntExpr,
}

/// Opaque request for the honest parameter-search construction.
///
/// The trusted key provider supplies the validated setup, sparse-LWR profile,
/// generated PBC layout, and key identity.  Sampling, selector construction,
/// preprocessing, refresh, and verification are intentionally kept inside the
/// consuming [`RefreshParameterSimulationBundle`] builder.
pub struct RefreshParameterSimulationRequest {
    setup: RefreshSetupParameters,
    profile: SparseLwrPrfProfile,
    generated_layout: crate::pbc::PbcGeneratedKeyLayout,
    key_instance_id: [u8; 32],
    expected_plaintext: Mat,
    decode_length: usize,
}

impl RefreshParameterSimulationRequest {
    /// Creates a request after checking all cross-layer dimensions and the
    /// mandatory official preimage policy.  The sparse support and one-hot
    /// schedule remain private inside the generated PBC layout.
    pub fn new(
        setup: RefreshSetupParameters,
        profile: SparseLwrPrfProfile,
        generated_layout: crate::pbc::PbcGeneratedKeyLayout,
        key_instance_id: [u8; 32],
        decoder_sigma: mxx_ir_core::RealExpr,
        expected_plaintext: Mat,
        decode_length: usize,
    ) -> Result<Self, RefreshSetupError> {
        setup.validate()?;
        if setup.decoder_preimage_bound != mxx_bgg::PreimageCoefficientBound::Official {
            return Err(RefreshSetupError::InvalidParameters(
                "parameter search requires the official preimage policy",
            ));
        }
        if setup.decoder_sigma != decoder_sigma || decode_length == 0 {
            return Err(RefreshSetupError::InvalidParameters(
                "decoder sigma or decode length does not match setup",
            ));
        }
        generated_layout
            .public_layout
            .validate()
            .map_err(|_| RefreshSetupError::InvalidParameters("generated PBC layout is invalid"))?;
        if profile.ring_dimension() !=
            setup
                .layout
                .ring_dimension
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|value| value.to_usize())
                .ok_or(RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        {
            return Err(RefreshSetupError::InvalidParameters(
                "sparse-LWR and refresh ring dimensions differ",
            ));
        }
        if expected_plaintext.matrix_type().rows != 1.into() ||
            expected_plaintext.matrix_type().columns != 1.into()
        {
            return Err(RefreshSetupError::InvalidParameters("expected plaintext must be 1x1"));
        }
        Ok(Self {
            setup,
            profile,
            generated_layout,
            key_instance_id,
            expected_plaintext,
            decode_length,
        })
    }

    /// Builds and validates the three graph stages and their declaration and
    /// attestation boundary. The resulting Exponent-LUT noise snapshot is the
    /// operational-noise authority for this bundle.
    pub fn build(self) -> Result<RefreshParameterSimulationBundle, RefreshSetupError> {
        self.build_with_mode(false)
    }

    /// Builds only the symbolic graphs needed by the GPU cost estimator.
    /// Simulation/verification graphs and runtime round-trip outputs are not
    /// constructed on this path.
    pub fn build_benchmark(self) -> Result<RefreshParameterSimulationBundle, RefreshSetupError> {
        self.build_with_mode(true)
    }

    fn build_with_mode(
        self,
        benchmark_only: bool,
    ) -> Result<RefreshParameterSimulationBundle, RefreshSetupError> {
        let setup = self.setup;
        let layout = self.generated_layout.public_layout.clone();
        let ring = setup.layout.ring();
        let hash_key = ring.bytes_input("refresh-parameter-search-hash-key", 32);
        let sampler = ExponentLutEncodingSampler {
            layout: setup.layout.clone(),
            gaussian_sigma: Some(setup.encoding_error_sigma.clone()),
            gaussian_max_coefficient_bound: Some(setup.encoding_error_bound.clone()),
        };
        let selector_bits = PbcTrustedSelectorBits::from_schedule(
            &self.generated_layout,
            &ring,
            self.key_instance_id,
        )
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let secret = sampler.sample_secret()?;
        let selector_names = PbcSelectorArtifactNames::canonicalize_schema(
            &layout,
            self.key_instance_id,
            setup.layout.secret_dimension,
            setup.layout.public_key_columns(),
        )
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let selector_artifacts = PbcSelectorArtifacts::from_structural(
            &layout,
            self.key_instance_id,
            selector_names,
            &sampler.layout,
            self.key_instance_id,
            &sampler.layout,
            self.key_instance_id,
        )
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let structural = build_structural_selector_families(
            &sampler,
            selector_bits.family().clone(),
            secret.clone(),
            secret.clone(),
            hash_key.clone(),
            &layout,
            self.key_instance_id,
        )
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let selector_graph = selector_artifacts
            .add_structural_family_outputs(
                DslContext::new("refresh-parameter-search-selector"),
                &layout,
                structural,
            )
            .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?
            .private_output("refresh-secret", secret.clone())?
            .build()?;
        let selector_spec_hash = spec_hash(&selector_graph.graph, &Default::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let selector_production =
            ProductionId { spec_hash: selector_spec_hash, execution_nonce: [0x52; 32] };
        let selector_validated = selector_graph
            .validate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let mut selector_manifest = mxx_ir_core::artifact::export_validated_manifest(
            selector_production.clone(),
            &selector_validated,
        )
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
        for artifact in selector_manifest.artifacts.values_mut() {
            if artifact.confidentiality == ArtifactConfidentiality::Public {
                artifact.content_hash = Some([0x5a; 32]);
            }
        }
        selector_artifacts
            .finalize_export_manifest(&mut selector_manifest)
            .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let selector_manifests = BTreeMap::from([(selector_production.clone(), selector_manifest)]);
        // Resolve each artifact input from the frozen selector producer.  The
        // graph is the source of truth for symbolic dimensions and family
        // counts; the concrete checks below catch an accidentally wired role
        // before it can enter the refresh graph.
        let package_count = layout
            .parameters
            .universe_size
            .checked_mul(layout.parameters.hash_count)
            .and_then(|count| count.checked_add(layout.parameters.bucket_count))
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let (secret_matrix, _) = resolve_selector_output_matrix(
            &selector_graph,
            "refresh-secret",
            ArtifactConfidentiality::Private,
            false,
            &ring.matrix_type((1, setup.layout.secret_dimension)),
            None,
        )?;
        let secret_ring =
            mxx_dsl::Ring::new(secret_matrix.modulus.clone(), secret_matrix.ring_dimension.clone());
        let secret_input = secret_ring.artifact_input(
            selector_production.clone(),
            "refresh-secret",
            (secret_matrix.rows.clone(), secret_matrix.columns.clone()),
            ArtifactConfidentiality::Private,
        );
        let gsw_name = crate::pbc::selector_family_artifact_name(&selector_artifacts, "gsw");
        let (gsw_matrix, gsw_count) = resolve_selector_output_matrix(
            &selector_graph,
            &gsw_name,
            ArtifactConfidentiality::Public,
            true,
            &ring.matrix_type((setup.layout.secret_dimension, setup.layout.public_key_columns())),
            Some(package_count),
        )?;
        let gsw_ring =
            mxx_dsl::Ring::new(gsw_matrix.modulus.clone(), gsw_matrix.ring_dimension.clone());
        let gsw = gsw_ring.family_artifact_input(
            selector_production.clone(),
            gsw_name,
            gsw_count.clone().ok_or(RefreshSetupError::InvalidManifest)?,
            (gsw_matrix.rows.clone(), gsw_matrix.columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let selectors = EncodingSelectorFamily::new(gsw.clone())?;
        let public_selectors = PublicSelectorFamily::new(gsw)?;
        let compiler = ExponentLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: setup.layout.gadget_base.clone(),
            digit_count: setup.layout.digit_count.into(),
        });
        let public_compiler = ExponentLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: setup.layout.gadget_base.clone(),
            digit_count: setup.layout.digit_count.into(),
        });
        let input_encodings = sampler.sample_input_encodings(
            secret_input.clone(),
            None,
            hash_key.clone(),
            b"refresh-parameter-search-inputs".as_slice(),
            &[ring.zero((1, 1)), ring.zero((1, 1))],
        )?;
        let state = input_encodings[0].clone();
        let fresh = input_encodings[1].clone();
        let program = SparseLwrPrfProgram::new(
            self.profile.clone(),
            layout.bucket_width,
            layout.parameters.bucket_count,
        )?;
        let reduction_width = program.plan.lut_width();
        let reduction_table =
            (0..reduction_width).map(|value| value % self.profile.q_l()).collect::<Vec<_>>();
        let reduction_lut = crate::program::LutTable::unary(
            reduction_width,
            reduction_width,
            reduction_table.clone(),
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid reduction LUT"))?;
        let rounding_lut = program
            .rounding_program()
            .lut(crate::program::LutId::from_index(0))
            .ok_or(RefreshSetupError::InvalidParameters("missing rounding LUT"))?;
        if program.rounding_lut() != rounding_lut.values() {
            return Err(RefreshSetupError::InvalidParameters("rounding program mismatch"));
        }
        let mask_bank = sampler.sample_flat_mask_bank(
            secret.clone(),
            hash_key.clone(),
            reduction_width.max(rounding_lut.values().len()),
            b"refresh-parameter-search-mask-bank".as_slice(),
        )?;
        let public_mask_bank = ExponentLutPublicKeySampler { layout: sampler.layout.clone() }
            .sample_flat_mask_bank(
                hash_key.clone(),
                reduction_width.max(rounding_lut.values().len()),
                b"refresh-parameter-search-mask-bank".as_slice(),
            )
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid public PRF mask bank"))?;
        let mut helpers = BTreeMap::from([(
            crate::program::LutId::from_index(0),
            FlatLutHelperSet::new(
                &reduction_lut,
                sampler.sample_flat_helpers_for_lut(
                    secret.clone(),
                    None,
                    hash_key.clone(),
                    &reduction_lut,
                    mask_bank.as_ref(),
                    b"refresh-parameter-search-reduce".as_slice(),
                )?,
            )?,
        )]);
        let rounding_helper_values = sampler.sample_flat_helpers_for_lut(
            secret.clone(),
            None,
            hash_key.clone(),
            rounding_lut,
            mask_bank.as_ref(),
            b"refresh-parameter-search-rounding".as_slice(),
        )?;
        let public_reduction_helpers = FlatLutPublicHelperSet::new(
            &reduction_lut,
            helpers
                .get(&crate::program::LutId::from_index(0))
                .ok_or(RefreshSetupError::InvalidParameters("missing reduction helpers"))?
                .iter()
                .map(|helper| {
                    FlatLutPublicHelper::with_mask_bank(
                        helper.sigma(),
                        helper.switch().public_projection(),
                        public_mask_bank.clone(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let public_rounding_helpers = FlatLutPublicHelperSet::new(
            rounding_lut,
            rounding_helper_values
                .iter()
                .map(|helper| {
                    FlatLutPublicHelper::with_mask_bank(
                        helper.sigma(),
                        helper.switch().public_projection(),
                        public_mask_bank.clone(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let public_helper_bundle = SparseLwrPrfPublicHelperBundle {
            reduction: SparseLwrPublicReductionHelpers::new(public_reduction_helpers),
            terminal: SparseLwrPublicTerminalHelpers::new(public_rounding_helpers),
        };
        let rounding_helpers = FlatLutHelperSet::new(rounding_lut, rounding_helper_values)?;
        let helper_bundle = SparseLwrPrfHelperBundle {
            reduction: SparseLwrReductionHelpers::new(
                helpers
                    .remove(&crate::program::LutId::from_index(0))
                    .ok_or(RefreshSetupError::InvalidParameters("missing reduction helpers"))?,
            ),
            terminal: SparseLwrTerminalHelpers::new(rounding_helpers),
        };
        let labels = crate::refresh::RefreshPrfLabelIndex::new(
            setup.refresh_id,
            setup.refresh.crt_plaintext_moduli.len(),
            setup.prf_component_count(),
            setup.coefficient_count,
            setup.mask_base_p_digit_count,
            setup.fresh_error_base_p_digit_count,
        )?;
        let batch = RefreshPrfBatchInputs::new(&layout, self.profile, &labels, &program)?;
        let mask_count = setup
            .refresh
            .crt_plaintext_moduli
            .len()
            .checked_mul(setup.prf_component_count())
            .and_then(|value| value.checked_mul(setup.coefficient_count))
            .and_then(|value| value.checked_mul(setup.mask_base_p_digit_count))
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let fresh_count = setup
            .prf_component_count()
            .checked_mul(setup.coefficient_count)
            .and_then(|value| value.checked_mul(setup.fresh_error_base_p_digit_count))
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let total =
            mask_count.checked_add(fresh_count).ok_or(RefreshSetupError::InvalidManifest)?;
        let mask_last = mask_count.checked_sub(1).ok_or(RefreshSetupError::InvalidManifest)?;
        // The PRF lowerer consumes one canonical label-major family.  Build
        // the two input families structurally in that same order and select
        // the state/fresh source inside one parallel body; this keeps the
        // label axis out of the host graph as a Vec/Family::pack.
        let selector = Parallel::range(total)
            .map_values(|index| index.as_int().less_equal(Int::constant(mask_last)).to_int())?;
        let state_vectors = Parallel::range(total).map_values({
            let state_vector = state.vector.clone();
            move |_| state_vector.clone()
        })?;
        let fresh_vectors = Parallel::range(total).map_values({
            let fresh_vector = fresh.vector.clone();
            move |_| fresh_vector.clone()
        })?;
        let input_vectors =
            selector.clone().parallel_select_mats(vec![fresh_vectors, state_vectors])?;
        let state_public_keys = Parallel::range(total).map_values({
            let state_public_key = state.pubkey.matrix.clone();
            move |_| state_public_key.clone()
        })?;
        let fresh_public_keys = Parallel::range(total).map_values({
            let fresh_public_key = fresh.pubkey.matrix.clone();
            move |_| fresh_public_key.clone()
        })?;
        let input_public_keys =
            selector.parallel_select_mats(vec![fresh_public_keys, state_public_keys])?;
        let public_outputs = program.compile_pbc_public_key_family_with_batch_and_helpers(
            &public_compiler,
            input_public_keys.clone(),
            &batch,
            public_selectors,
            &public_helper_bundle,
        )?;
        let outputs = program.compile_pbc_encoding_family_typed_with_batch_and_helpers(
            &compiler,
            input_vectors,
            input_public_keys,
            &batch,
            selectors,
            &helper_bundle,
        )?;
        let prf = RefreshPrfInputs::from_pbc_family_outputs(&setup, &program, &batch, &outputs)?;

        // Build the two benchmark stages at the mathematical protocol
        // boundary. Preprocessing independently evaluates the PRF program on
        // public keys and samples decoder preimages. Online evaluation
        // independently evaluates the same program on complete encodings and
        // consumes preimages as external typed inputs. In particular, the
        // online graph does not import PRF-derived masks/fresh errors from the
        // preprocessing graph, and the preprocessing graph contains no
        // encoding-vector PRF work.
        let slot_count = setup.refresh.crt_plaintext_moduli.len();
        let benchmark_scales = (0..slot_count)
            .map(|slot| Ok(ring.polynomial([setup.refresh.scale_expression(slot)?])))
            .collect::<Result<Vec<_>, RefreshSetupError>>()?;
        let (public_masks, public_fresh) = aggregate_public_refresh_prf(
            &public_compiler,
            public_outputs,
            setup.base_p,
            slot_count,
            setup.prf_component_count(),
            setup.coefficient_count,
            setup.mask_base_p_digit_count,
            setup.fresh_error_base_p_digit_count,
            benchmark_scales.clone(),
        )?;
        let benchmark_a_prime = ring.hash_matrix(
            hash_key.clone(),
            HashTag::from(
                format!("mxx-exponent-lut/refresh/a-prime/v1/{}", hex(&setup.refresh_id))
                    .into_bytes(),
            ),
            (setup.layout.secret_dimension, setup.layout.public_key_columns()),
        );
        let decoder_preimage_bound = setup.resolve_decoder_preimage_bound()?;
        let benchmark_trapdoor = ring.sample_trapdoor(
            setup.layout.secret_dimension,
            setup.decoder_sigma.clone(),
            setup.layout.gadget_base.clone(),
            setup.layout.digit_count,
            decoder_preimage_bound.clone(),
        );
        let benchmark_public_b = benchmark_trapdoor.public_matrix();
        let benchmark_decoder_base = shared_decoder_base(
            &ring,
            secret_input.clone(),
            benchmark_public_b.clone(),
            setup.encoding_error_sigma.clone(),
            setup.encoding_error_bound.clone(),
        );
        let b_columns = setup
            .component_count
            .checked_mul(setup.layout.digit_count + 2)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let public_mask_family = Family::pack(public_masks)?;
        let public_fresh_family = Family::pack(public_fresh)?;
        let scale_family = Family::pack(benchmark_scales.clone())?;
        let benchmark_targets = Family::<Mat>::parallel_zip_many_values(
            vec![public_mask_family, public_fresh_family, scale_family],
            |_, mut values| {
                let scale = values.pop().expect("scale family");
                let fresh = values.pop().expect("fresh family");
                let mask = values.pop().expect("mask family");
                let scaled_state = public_compiler.public_key.large_scalar_mul(
                    &BggPublicKeyWire {
                        matrix: state.pubkey.matrix.clone(),
                        reveal_plaintext: false,
                    },
                    &scale,
                );
                scaled_state.matrix + mask + fresh - scale * benchmark_a_prime.clone()
            },
        )?;
        let benchmark_preimages = benchmark_targets.parallel_map_values(|_, target| {
            benchmark_trapdoor
                .sample_preimage(target, (b_columns, setup.layout.public_key_columns()))
        })?;
        let mut benchmark_preprocessing_context =
            DslContext::new("refresh-parameter-benchmark-preprocessing");
        benchmark_preprocessing_context = benchmark_preprocessing_context
            .public_output("benchmark-a-prime", benchmark_a_prime.clone())?
            .public_output("benchmark-public-b", benchmark_public_b.clone())?
            .private_output("benchmark-decoder-base", benchmark_decoder_base.vector.clone())?;
        for slot in 0..slot_count {
            benchmark_preprocessing_context = benchmark_preprocessing_context
                .private_preimage_output(
                    format!("benchmark-decoder-preimage-{slot}"),
                    benchmark_preimages.get_static(slot),
                )?;
        }
        let benchmark_preprocessing_graph = benchmark_preprocessing_context.build()?;

        let online_masks = prf.aggregate_masks(&compiler, setup.base_p)?;
        let online_fresh = aggregate_refresh_fresh_error_per_slot(
            &compiler,
            setup.base_p,
            &prf.fresh_error,
            benchmark_scales,
        )?;
        let online_a_prime = ring
            .input("benchmark-a-prime", (setup.component_count, setup.layout.public_key_columns()));
        let online_public_b = ring.input("benchmark-public-b", (setup.component_count, b_columns));
        let online_decoder_base = BggEncodingWire {
            vector: ring.input("benchmark-decoder-base", (1, b_columns)),
            pubkey: BggPublicKeyWire { matrix: online_public_b.clone(), reveal_plaintext: false },
            plaintext: None,
        };
        let online_preimages = (0..slot_count)
            .map(|slot| {
                ring.preimage_input(
                    format!("benchmark-decoder-preimage-{slot}"),
                    (b_columns, setup.layout.public_key_columns()),
                    decoder_preimage_bound.clone(),
                )
            })
            .collect();
        let online_manifest = setup.refresh.bind_imported_wires(
            &compiler,
            state.clone(),
            online_a_prime,
            online_public_b,
            online_fresh,
            online_masks,
            online_decoder_base,
            online_preimages,
        )?;
        let online_refreshed = setup.refresh.refresh(&compiler, &online_manifest)?;
        let benchmark_online_graph = DslContext::new("refresh-parameter-benchmark-online-eval")
            .private_output(
                "refreshed-encoding-vector",
                online_refreshed.encoding().vector.clone(),
            )?
            .public_output(
                "refreshed-encoding-public-key",
                online_refreshed.encoding().pubkey.matrix.clone(),
            )?
            .build()?;
        if benchmark_only {
            let noise_snapshot = setup.build_noise_snapshot(program, layout)?;
            let public_identity = *noise_snapshot.setup_identity();
            return Ok(RefreshParameterSimulationBundle {
                selector_graph,
                preprocessing_graph: None,
                verification_graph: None,
                benchmark_preprocessing_graph,
                benchmark_online_graph,
                preprocessing_manifests: selector_manifests.clone(),
                verification_manifests: selector_manifests,
                public_identity,
                metadata: BTreeMap::new(),
                targets: Vec::new(),
                noise_snapshot,
            });
        }
        let producer = RefreshPreprocessingProducer::build(RefreshPreprocessingRequest {
            parameters: setup.clone(),
            prf,
            compiler: ExponentLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
                ring: ring.clone(),
                base: setup.layout.gadget_base.clone(),
                digit_count: setup.layout.digit_count.into(),
            }),
            state: state.clone(),
            secret: secret_input.clone(),
            hash_key,
        })?;
        // Cross the preprocessing boundary through an exported manifest.  A
        // simulation must exercise the same producer/import contract as a
        // real refresh consumer; passing `producer.wires` directly would hide
        // missing artifact bindings and accidentally couple the stages by
        // construction-time handles.
        let preprocessing_manifests = selector_manifests;
        let producer_validated = producer
            .built
            .validate_with_manifests(&ParamEnv::default(), &preprocessing_manifests)
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let producer_spec_hash =
            spec_hash(&producer_validated.source, &producer_validated.bindings)
                .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let producer_production = ProductionId {
            spec_hash: producer_spec_hash,
            // Public and fixed for this deterministic simulation namespace;
            // the producer spec hash still commits to the actual graph.
            execution_nonce: [0x53; 32],
        };
        let mut producer_manifest = mxx_ir_core::artifact::export_validated_manifest(
            producer_production.clone(),
            &producer_validated,
        )
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
        for artifact in producer_manifest.artifacts.values_mut() {
            if artifact.confidentiality == ArtifactConfidentiality::Public {
                artifact.content_hash = Some([0xab; 32]);
            }
        }
        producer.finalize_export_manifest(&mut producer_manifest)?;
        let imported = ImportedRefreshSetup::import(
            producer_production.clone(),
            setup.clone(),
            producer.declaration().clone(),
            producer.attestation(),
            &producer_manifest,
        )?;
        let mut verification_manifests = preprocessing_manifests.clone();
        verification_manifests.insert(producer_production, producer_manifest);
        let manifest = setup.refresh.bind_imported_setup(&compiler, &imported)?;
        let refreshed = setup.refresh.refresh(&compiler, &manifest)?;
        let verification = build_refresh_verification(
            refreshed.encoding(),
            &secret_input,
            &secret_input,
            &self.expected_plaintext,
            setup.layout.gadget_base.clone(),
            setup.layout.digit_count,
            setup.base_p,
            self.decode_length,
        )?;
        let verification_graph = verification
            .add_outputs(
                DslContext::new("refresh-parameter-search-verification"),
                "residual",
                "decoded",
            )?
            .build()?;
        let targets = decoder_targets(
            &verification_graph,
            &verification,
            self.decode_length,
            setup.base_p.into(),
        )?;
        let metadata = matrix_input_metadata(
            &selector_graph,
            &producer.built,
            &verification_graph,
            &batch.public_input_name(),
            batch.profile().q_l(),
        )?;
        let public_identity = producer.declaration.identity;
        let noise_snapshot = setup.build_noise_snapshot(program, layout)?;
        if noise_snapshot.setup_identity() != &public_identity {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let preprocessing_graph = producer.built;
        Ok(RefreshParameterSimulationBundle {
            selector_graph,
            preprocessing_graph: Some(preprocessing_graph),
            verification_graph: Some(verification_graph),
            benchmark_preprocessing_graph,
            benchmark_online_graph,
            preprocessing_manifests,
            verification_manifests,
            public_identity,
            metadata,
            targets,
            noise_snapshot,
        })
    }
}

/// Resolves one selector-producer output into the exact symbolic matrix type
/// consumed by a later artifact input.  The output's confidentiality and
/// family shape are checked against the role contract, while every integer
/// expression is also evaluated against the concrete expected dimensions.
/// This keeps producer and consumer schemas coupled by the frozen graph
/// without adding a public schema type or exposing private selector values.
fn resolve_selector_output_matrix(
    graph: &BuiltGraph,
    name: &str,
    expected_confidentiality: ArtifactConfidentiality,
    expected_family: bool,
    expected_matrix: &mxx_ir_core::types::MatrixType,
    expected_count: Option<usize>,
) -> Result<(mxx_ir_core::types::MatrixType, Option<mxx_ir_core::IntExpr>), RefreshSetupError> {
    resolve_graph_output_matrix(
        &graph.graph,
        name,
        expected_confidentiality,
        expected_family,
        expected_matrix,
        expected_count,
    )
}

/// Resolves one canonical graph output to its exact symbolic matrix type.
///
/// The caller supplies the semantic role contract (confidentiality, scalar or
/// family, and expected evaluated dimensions).  The returned matrix retains
/// the producer's original symbolic expressions, which is important when an
/// artifact-backed consumer is rebuilt at a stage boundary.
fn resolve_graph_output_matrix(
    graph: &Graph,
    name: &str,
    expected_confidentiality: ArtifactConfidentiality,
    expected_family: bool,
    expected_matrix: &mxx_ir_core::types::MatrixType,
    expected_count: Option<usize>,
) -> Result<(mxx_ir_core::types::MatrixType, Option<mxx_ir_core::IntExpr>), RefreshSetupError> {
    let output = graph.outputs().get(name).ok_or(RefreshSetupError::InvalidManifest)?;
    if output.confidentiality != Some(expected_confidentiality) {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let root = graph.root_scope();
    let output_type = root
        .node(output.value.node)
        .and_then(|node| node.output_types().get(output.value.port.0 as usize))
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let (matrix, count) = match output_type {
        WireType::Matrix(matrix) if !expected_family => (matrix, None),
        WireType::Family { element, shape } if expected_family => {
            let WireType::Matrix(matrix) = element.as_ref() else {
                return Err(RefreshSetupError::InvalidManifest);
            };
            let count = (shape.len() == 1).then(|| shape[0].clone());
            (matrix, count)
        }
        _ => return Err(RefreshSetupError::InvalidManifest),
    };
    let evaluate = |actual: &mxx_ir_core::IntExpr, expected: &mxx_ir_core::IntExpr| match (
        actual.evaluate(&ParamEnv::default()),
        expected.evaluate(&ParamEnv::default()),
    ) {
        (Ok(actual), Ok(expected)) => actual == expected,
        _ => false,
    };
    if !evaluate(&matrix.modulus, &expected_matrix.modulus) ||
        !evaluate(&matrix.ring_dimension, &expected_matrix.ring_dimension) ||
        !evaluate(&matrix.rows, &expected_matrix.rows) ||
        !evaluate(&matrix.columns, &expected_matrix.columns)
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    match (count.clone(), expected_count) {
        (None, None) => {}
        (Some(actual), Some(expected))
            if actual.evaluate(&ParamEnv::default()).ok().and_then(|v| v.to_usize()) ==
                Some(expected) => {}
        _ => return Err(RefreshSetupError::InvalidManifest),
    }
    Ok((matrix.clone(), count))
}

/// Resolves a private preimage artifact output while retaining its typed
/// witness schema.  A matrix output with the same dimensions is not accepted:
/// the `Preimage` wire type is part of the producer/consumer contract.
fn resolve_graph_output_preimage(
    graph: &Graph,
    name: &str,
    expected_confidentiality: ArtifactConfidentiality,
    expected_matrix: &mxx_ir_core::types::MatrixType,
) -> Result<mxx_ir_core::types::MatrixType, RefreshSetupError> {
    let output = graph.outputs().get(name).ok_or(RefreshSetupError::InvalidManifest)?;
    if output.confidentiality != Some(expected_confidentiality) {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let root = graph.root_scope();
    let output_type = root
        .node(output.value.node)
        .and_then(|node| node.output_types().get(output.value.port.0 as usize))
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let WireType::Preimage { matrix, .. } = output_type else {
        return Err(RefreshSetupError::InvalidManifest);
    };
    if !same_matrix_type(matrix, expected_matrix) {
        return Err(RefreshSetupError::InvalidManifest);
    }
    Ok(matrix.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::node::NodeKind;

    fn parameters(gadget_base: i32, secret_dimension: usize) -> RefreshSetupParameters {
        let layout = BggSamplerLayout {
            modulus: 6.into(),
            ring_dimension: 4.into(),
            secret_dimension,
            digit_count: 2,
            gadget_base: gadget_base.into(),
        };
        RefreshSetupParameters::new(
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
                full_modulus: 6.into(),
                crt_plaintext_moduli: vec![2.into(), 3.into()],
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
        assert_eq!(setup.mask_base_p_digit_count, 1);
        assert_eq!(setup.fresh_error_base_p_digit_count, 1);
        assert_eq!(setup.prf_component_count(), 4);
        assert_eq!(setup.layout.digit_count, 2);
        setup.validate_dimensions().expect("independent digit counts are valid");
    }

    #[test]
    fn average_initial_variance_is_derived_from_setup_sigma() {
        let setup = parameters(4, 2);
        assert_eq!(
            setup.average_initial_variance().expect("rational setup sigma"),
            AverageVariance::new(4u8.into(), 1u8.into()).expect("positive variance")
        );
    }

    #[test]
    fn average_candidate_factory_reuses_canonical_snapshot_identity() {
        let setup = RefreshSetupParameters::new(
            [0x42; 32],
            2,
            2,
            8,
            1,
            1,
            32,
            8,
            BggSamplerLayout {
                modulus: 6.into(),
                ring_dimension: 8.into(),
                secret_dimension: 2,
                digit_count: 2,
                gadget_base: 4.into(),
            },
            RefreshCompiler {
                full_modulus: 6.into(),
                crt_plaintext_moduli: vec![2.into(), 3.into()],
                reconstruction_coefficients: vec![1.into(), 1.into()],
            },
            1.into(),
            1.into(),
            "candidate-factory-test",
        );
        let pbc_parameters = crate::pbc::PbcParameters::paper_evaluation(10, 4);
        let pbc_layout = PbcPublicLayout::build(
            &pbc_parameters,
            crate::pbc::derive_attempt_seed(crate::pbc::PbcRootSeed([14u8; 32]), 0),
            0,
        )
        .expect("PBC layout");
        let program = SparseLwrPrfProgram::new(
            SparseLwrPrfProfile::new(2, 2, 8, 8).expect("PRF profile"),
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )
        .expect("PRF program");
        let config =
            AverageCaseConfig { allow_average_acceptance: true, ..AverageCaseConfig::default() };
        let snapshot = setup
            .build_noise_snapshot(program.clone(), pbc_layout.clone())
            .expect("canonical snapshot");
        let baseline = setup
            .evaluate_average_candidate(program.clone(), pbc_layout.clone(), 1, &config)
            .expect("baseline candidate");
        let changed = setup
            .evaluate_average_candidate(program, pbc_layout, 2, &config)
            .expect("changed candidate");
        assert_eq!(baseline.snapshot_identity, *snapshot.setup_identity());
        assert_ne!(baseline.snapshot_identity, changed.snapshot_identity);
        assert_eq!(
            baseline.accepted,
            baseline.hard_authority_accepted && baseline.correctness_accepted
        );
    }

    #[test]
    fn setup_requires_secret_dimension_two_and_power_of_two_gadget_base() {
        assert!(parameters(4, 1).validate_dimensions().is_err());
        assert!(parameters(3, 2).validate_dimensions().is_err());
    }

    #[test]
    fn setup_identity_binds_mask_and_fresh_digit_counts_and_security() {
        let setup = parameters(4, 2);
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(2, 2, 4, 4).expect("profile"), 1, 1)
                .expect("program");
        let contract = RefreshPrfContract::from_program(&program);
        let decoder_bound = 1.into();
        let identity = |setup: &RefreshSetupParameters| {
            identity_digest(setup, &contract, crate::pbc::PbcLayoutId([9; 32]), &decoder_bound)
        };
        let baseline = identity(&setup);
        let mut changed_mask = setup.clone();
        changed_mask.mask_base_p_digit_count = 2;
        assert_ne!(baseline, identity(&changed_mask));
        let mut changed_fresh = setup.clone();
        changed_fresh.fresh_error_base_p_digit_count = 2;
        assert_ne!(baseline, identity(&changed_fresh));
        let mut changed_security = setup;
        changed_security.mask_statistical_security_bits = 33;
        assert_ne!(baseline, identity(&changed_security));
    }

    #[test]
    fn shared_decoder_base_samples_one_error_and_exports_one_artifact() {
        let setup = parameters(4, 2);
        let ring = setup.layout.ring();
        let secret = ring.zero((1, 2));
        let public_b = ring.zero((2, 8));
        let base = shared_decoder_base(
            &ring,
            secret,
            public_b,
            setup.encoding_error_sigma.clone(),
            setup.encoding_error_bound.clone(),
        );
        assert_eq!(base.vector.matrix_type().rows, 1.into());
        assert_eq!(base.vector.matrix_type().columns, 8.into());

        let context = DslContext::new("shared-decoder-base-test")
            .private_output("decoder-base-vector", base.vector)
            .expect("decoder-base output")
            .build()
            .expect("decoder-base graph");
        let gaussian_nodes = context
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::GaussianSample { .. }))
            .count();
        assert_eq!(gaussian_nodes, 1);
        assert_eq!(context.graph.outputs().len(), 1);
    }
}

/// Compares matrix schemas by their evaluated dimensions.  DSL constructors
/// may leave equivalent expressions such as `1 + 1` symbolic in one producer
/// and `2` in another; those are the same concrete ring matrix and must remain
/// compatible when the producer's exact expressions are reused.
fn same_matrix_type(
    left: &mxx_ir_core::types::MatrixType,
    right: &mxx_ir_core::types::MatrixType,
) -> bool {
    [
        (&left.modulus, &right.modulus),
        (&left.ring_dimension, &right.ring_dimension),
        (&left.rows, &right.rows),
        (&left.columns, &right.columns),
    ]
    .into_iter()
    .all(|(left, right)| {
        match (left.evaluate(&ParamEnv::default()), right.evaluate(&ParamEnv::default())) {
            (Ok(left), Ok(right)) => left == right,
            _ => false,
        }
    })
}

/// The three immutable graph stages plus request-level external-input facts.
/// BuiltGraph, declaration, and attestation validation occurs during bundle
/// construction. No schedules, secrets, or raw constructors are exposed, and
/// the embedded Exponent-LUT snapshot remains the operational-noise authority.
pub struct RefreshParameterSimulationBundle {
    selector_graph: BuiltGraph,
    preprocessing_graph: Option<BuiltGraph>,
    verification_graph: Option<BuiltGraph>,
    benchmark_preprocessing_graph: BuiltGraph,
    benchmark_online_graph: BuiltGraph,
    preprocessing_manifests: std::collections::BTreeMap<ProductionId, Manifest>,
    verification_manifests: std::collections::BTreeMap<ProductionId, Manifest>,
    public_identity: [u8; 32],
    metadata: std::collections::BTreeMap<String, RefreshSimulationMatrixInputMetadata>,
    targets: Vec<RefreshSimulationDecoderTarget>,
    noise_snapshot: ExponentLutNoiseSnapshot,
}

impl RefreshParameterSimulationBundle {
    /// Returns the graph entrypoint used when assembling request-level facts.
    pub fn entrypoint(&self) -> &'static str {
        "verification"
    }
    /// Returns the public identity of the complete setup bundle.
    pub fn public_identity(&self) -> &[u8; 32] {
        &self.public_identity
    }
    /// Returns the structural selector-producer graph.
    pub fn selector_graph(&self) -> &BuiltGraph {
        &self.selector_graph
    }
    /// Returns the trusted preprocessing producer graph.
    pub fn preprocessing_graph(&self) -> &BuiltGraph {
        self.preprocessing_graph
            .as_ref()
            .expect("simulation preprocessing graph is unavailable in a benchmark-only bundle")
    }
    /// Returns the selector manifests required to validate preprocessing.
    pub fn preprocessing_manifests(&self) -> &std::collections::BTreeMap<ProductionId, Manifest> {
        &self.preprocessing_manifests
    }
    /// Returns the verification graph used when assembling request-level facts.
    pub fn verification_graph(&self) -> &BuiltGraph {
        self.verification_graph
            .as_ref()
            .expect("simulation verification graph is unavailable in a benchmark-only bundle")
    }
    /// Returns the benchmark preprocessing graph: public-key PRF evaluation
    /// followed by decoder-target construction and preimage sampling.
    pub fn benchmark_preprocessing_graph(&self) -> &BuiltGraph {
        &self.benchmark_preprocessing_graph
    }
    /// Returns the benchmark online graph: independent encoding PRF
    /// evaluation followed by application of externally supplied decoder
    /// preimages and production of the refreshed encoding.
    pub fn benchmark_online_graph(&self) -> &BuiltGraph {
        &self.benchmark_online_graph
    }
    /// Returns the preprocessing manifests required to validate online evaluation.
    pub fn verification_manifests(&self) -> &std::collections::BTreeMap<ProductionId, Manifest> {
        &self.verification_manifests
    }
    /// Returns exact matrix facts used only for request-level external-input
    /// validation. Operational noise is evaluated by [`Self::simulate_noise`].
    pub fn matrix_input_metadata(
        &self,
    ) -> &std::collections::BTreeMap<String, RefreshSimulationMatrixInputMetadata> {
        &self.metadata
    }
    /// Returns the frozen residual-to-decoder links for every target column.
    pub fn decoder_targets(&self) -> &[RefreshSimulationDecoderTarget] {
        &self.targets
    }

    /// Runs the setup-bound application-specific sparse-PRF and refresh noise
    /// simulation.  The snapshot identity is checked against the public bundle
    /// identity before any bound is evaluated.
    pub fn simulate_noise(
        &self,
    ) -> Result<ExponentLutNoiseReport, crate::noise::NoiseSimulationError> {
        if self.noise_snapshot.setup_identity() != &self.public_identity {
            return Err(crate::noise::NoiseSimulationError::InvalidRefresh(
                "simulation snapshot identity mismatch",
            ));
        }
        self.noise_snapshot.simulate()
    }

    /// Runs the explicitly opted-in AverageCase diagnostic against the same
    /// immutable setup snapshot used by [`Self::simulate_noise`].
    pub fn simulate_average_noise(
        &self,
        config: &AverageCaseConfig,
    ) -> Result<ExponentLutAverageNoiseReport, crate::noise::NoiseSimulationError> {
        if self.noise_snapshot.setup_identity() != &self.public_identity {
            return Err(crate::noise::NoiseSimulationError::AverageIdentityMismatch);
        }
        self.noise_snapshot.simulate_average(config)
    }
}

/// Collects constant-coefficient facts for request-level external-input
/// validation in all graph stages. It does not infer operational noise from a
/// runtime ciphertext or from a private setup vector.
fn matrix_input_metadata(
    selector: &BuiltGraph,
    preprocessing: &BuiltGraph,
    verification: &BuiltGraph,
    public_value_name: &str,
    public_value_modulus: usize,
) -> Result<
    std::collections::BTreeMap<String, RefreshSimulationMatrixInputMetadata>,
    RefreshSetupError,
> {
    let mut result = std::collections::BTreeMap::new();
    let mut public_value_inputs = 0usize;
    for graph in [selector, preprocessing, verification] {
        for node in graph.graph.root_scope().nodes() {
            if let mxx_ir_core::node::NodeKind::Input {
                name,
                wire_type: WireType::Family { element, .. },
                artifact: None,
            } = node.kind() &&
                matches!(element.as_ref(), WireType::Matrix(_))
            {
                if name == public_value_name {
                    public_value_inputs += 1;
                }
                result.entry(name.clone()).or_insert_with(|| {
                    RefreshSimulationMatrixInputMetadata {
                        canonical_coefficient_exclusive_upper_bound: Some(
                            if name == public_value_name {
                                mxx_ir_core::IntExpr::constant(public_value_modulus)
                            } else {
                                mxx_ir_core::IntExpr::constant(2)
                            },
                        ),
                        is_constant_polynomial: true,
                    }
                });
            }
        }
    }
    if public_value_inputs != 1 || !result.contains_key(public_value_name) {
        return Err(RefreshSetupError::InvalidManifest);
    }
    Ok(result)
}

/// Resolves and validates each frozen residual-to-decoder output link.
fn decoder_targets(
    graph: &BuiltGraph,
    verification: &RefreshVerification,
    decode_length: usize,
    plaintext_modulus: mxx_ir_core::IntExpr,
) -> Result<Vec<RefreshSimulationDecoderTarget>, RefreshSetupError> {
    let root = graph.graph.root_scope();
    let mut validated = Vec::with_capacity(verification.decoder_residuals().len());
    for column in 0..verification.decoder_residuals().len() {
        let operand = root
            .wire_ref(verification.decoder_residuals()[column].value_handle())
            .ok_or(RefreshSetupError::DecoderResidualMismatch)?;
        let operand = ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: operand };
        let residual_output_name = format!("refresh-decoder-residual-{column}");
        let residual_output = graph
            .graph
            .outputs()
            .get(&residual_output_name)
            .ok_or(RefreshSetupError::DecoderResidualMismatch)?;
        if residual_output.value != operand.wire {
            return Err(RefreshSetupError::DecoderResidualMismatch);
        }
        let name = format!("decoded_{}", column * decode_length);
        let output =
            graph.graph.outputs().get(&name).ok_or(RefreshSetupError::DecoderResidualMismatch)?;
        let decoder_wire = output.value;
        let decoder_node =
            root.node(decoder_wire.node).ok_or(RefreshSetupError::DecoderResidualMismatch)?;
        if !matches!(
            decoder_node.kind(),
            mxx_ir_core::node::NodeKind::ThresholdDecode {
                plaintext_modulus: actual,
                output_bool: true,
                ..
            } if actual == &plaintext_modulus
        ) {
            return Err(RefreshSetupError::DecoderResidualMismatch);
        }
        let arguments =
            root.arguments(decoder_node).ok_or(RefreshSetupError::DecoderResidualMismatch)?;
        let argument =
            arguments.first().copied().ok_or(RefreshSetupError::DecoderResidualMismatch)?;
        let argument = ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: argument };
        if argument != operand {
            return Err(RefreshSetupError::DecoderResidualMismatch);
        }
        let decoder = ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: decoder_wire };
        let residual_type = root
            .node(residual_output.value.node)
            .and_then(|node| node.output_types().get(residual_output.value.port.0 as usize))
            .cloned()
            .ok_or(RefreshSetupError::DecoderResidualMismatch)?;
        let output_type = decoder_node
            .output_types()
            .get(decoder_wire.port.0 as usize)
            .cloned()
            .ok_or(RefreshSetupError::DecoderResidualMismatch)?;
        validated.push(RefreshSimulationDecoderTarget {
            target_id: format!("refresh-decoder-{column}"),
            residual_output_name,
            residual_anchor: format!("refresh.decoder.residual.{column}"),
            residual: operand,
            residual_type,
            decoder,
            decoder_node: decoder_wire.node,
            decoder_output_type: output_type,
            decoded_output_name: name,
            decoder_anchor: format!("refresh.decoder.output.{}", column * decode_length),
            plaintext_modulus: plaintext_modulus.clone(),
        });
    }

    // The simulation checker consumes one aggregate residual target, while
    // the complete scalar decoder graph remains present and validated above.
    // Use decoder 0 as the executable witness; this metadata coalescing does
    // not modify graph nodes, wires, or outputs.
    let decoder_zero = validated.first().ok_or(RefreshSetupError::DecoderResidualMismatch)?;
    let aggregate_output =
        graph.graph.outputs().get("residual").ok_or(RefreshSetupError::DecoderResidualMismatch)?;
    let aggregate_wire =
        ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: aggregate_output.value };
    let aggregate_type = root
        .node(aggregate_output.value.node)
        .and_then(|node| node.output_types().get(aggregate_output.value.port.0 as usize))
        .cloned()
        .ok_or(RefreshSetupError::DecoderResidualMismatch)?;
    if !matches!(aggregate_type, WireType::Matrix(_)) {
        return Err(RefreshSetupError::DecoderResidualMismatch);
    }
    Ok(vec![RefreshSimulationDecoderTarget {
        target_id: "refresh-decoder-aggregate".to_owned(),
        residual_output_name: "residual".to_owned(),
        residual_anchor: "refresh.residual.aggregate".to_owned(),
        residual: aggregate_wire,
        residual_type: aggregate_type,
        decoder: decoder_zero.decoder.clone(),
        decoder_node: decoder_zero.decoder_node,
        decoder_output_type: decoder_zero.decoder_output_type.clone(),
        decoded_output_name: decoder_zero.decoded_output_name.clone(),
        decoder_anchor: decoder_zero.decoder_anchor.clone(),
        plaintext_modulus: decoder_zero.plaintext_modulus.clone(),
    }])
}
