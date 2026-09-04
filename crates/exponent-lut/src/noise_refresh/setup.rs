//! RNS refresh setup and artifact boundary.
//!
//! This module owns the setup-only half of refresh: it samples the public and
//! private preprocessing material, builds the preprocessing graph, and imports
//! that material at a later production boundary. The runtime refresh equations themselves live
//! in [`crate::noise_refresh::refresh`], while sparse-LWR/PBC evaluation lives in
//! [`crate::noise_refresh::prf`] and [`crate::noise_refresh::pbc`]. Keeping these layers separate
//! is important: setup may hold secrets and trapdoors, but the public program and
//! graph declarations must not contain private support or selector choices.
//!
//! Setup material is organized per CRT slot `t`. With `mu_t=q/q_t`, the
//! producer constructs `A_{sum,t} = A_t + A_{m,t} + A_{e,t}`, then the preimage
//! target `T_t = A_{sum,t} - mu_t A'` and a matrix `K_t` satisfying `B K_t=T_t`.
//! The public matrix `B`, targets, and typed preimages `K_t` are fixed by
//! preprocessing. Online evaluation samples one fresh anchor error and forms
//! one shared `b=sB+e_b`, then uses `d_t=bK_t` for every slot. Neither the
//! online secret `s` nor `b` crosses the artifact boundary.

use crate::{
    ExponentLutEncodingCompiler, ExponentLutError,
    encoding::{EncodingRhsFamily, ExponentLutEncodingSampler},
    noise::{
        AverageVariance, GateNoiseDecomposition, GateNoiseGains, GateNoiseOperation,
        NoiseDecompositionPlan, RightActionNoiseGain,
    },
    noise_refresh::{
        pbc::{
            PbcActiveCellIndex, PbcPublicLayout, PbcSelectorArtifactNames, PbcSelectorArtifacts,
            PbcSelectorFamilyInputs, PbcSelectorRuntimeBinding, PbcTrustedSelectorBits,
            build_structural_selector_families, selector_family_artifact_name,
        },
        prf::{
            NativePrfEncodingBatchOutputs, NativePrfOnlineBatchInputs, NativePrfPublicInputs,
            NativePrfReadoutFamilyArtifacts, SparseLwrPrfProfile, SparseLwrPrfProgram,
        },
        refresh::{
            RefreshCompiler, RefreshError, RefreshFreshErrorMaterial, RefreshMaskMaterial,
            RefreshPrfContract, RefreshPrfCoverage, RefreshPrfFamilyMaterial, RefreshSetupManifest,
            aggregate_refresh_fresh_error_per_slot, aggregate_refresh_masks,
        },
        simulation::{
            AverageCaseConfig, ExponentLutAverageNoiseReport, ExponentLutNoiseReport,
            ExponentLutNoiseSnapshot, MaskSmudgingParameters, NoiseGainPlan, NoiseGainPlanBuilder,
            RefreshNoiseParameters, RefreshRouteNoiseGains, RefreshTranscriptTopology,
            TraceNoiseGains, ValueReadoutNoiseGains, centered_base_p_digit_variance,
        },
    },
    program::{ExponentLutProgramId, Gate, LutTable, ProgramStep},
    public_key::{
        ExponentLutPublicKeyCompiler, ProgramCacheBatchBindings, ProgramCacheLocation,
        ProgramCacheRole, PublicRhsFamily, program_cache_declarations,
    },
};
use bigdecimal::BigDecimal;
use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout};
use mxx_dsl::{
    Bool, BuiltGraph, Bytes, DslContext, Family, HashTag, Int, Mat, Parallel, ParallelZipTuple,
    Preimage,
};
use mxx_ir_core::{
    BenchmarkRole, IntExpr, ParamEnv, ScopedWireRef,
    artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId},
    encoding::{IR_VERSION, hash_canonical, spec_hash},
    graph::FrozenGraphScopeId,
    types::{ConcreteMatrixType, NodeId, WireType},
    with_benchmark_role,
};
use mxx_primitives::sampler::bounds::TrapdoorPreimageCutoffPolicy;
use num_bigint::{BigInt, BigUint, ToBigInt};
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
    /// The producer output or artifact manifest violates its role schema.
    InvalidManifest,
    #[error("refresh setup stage `{stage}` failed: {detail}")]
    /// A setup boundary rejected the graph or manifest; the stage and original
    /// diagnostic are retained so benchmark failures remain actionable.
    SetupStage { stage: &'static str, detail: String },
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
    pub decoder_preimage_bound: TrapdoorPreimageCutoffPolicy,
    /// Human-readable setup name used by callers when selecting a production.
    pub name: String,
}

/// Setup-produced decomposition plans for the refresh routes.
///
/// Each plan must contain either the concrete balanced coefficients or a
/// validated compact source/bound for the corresponding target.  The setup
/// boundary derives all route gains through
/// [`NoiseDecompositionPlan::action_gain`]; callers cannot provide a detached
/// scalar gain in its place.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefreshNoiseGainDecompositions {
    /// One compact decomposition for each CRT slot's `G^-1(kappa_t G)` action.
    pub state_scaling: Vec<NoiseDecompositionPlan>,
    /// One compact decomposition for every P-adic mask digit `P^c`.
    pub mask: Vec<NoiseDecompositionPlan>,
    /// Number of ring-coefficient positions represented by each mask plan.
    /// The compact search path stores one plan per P-adic digit and uses this
    /// multiplicity instead of materializing `coefficient_count` copies.
    #[serde(default = "one_usize")]
    pub mask_multiplicity: usize,
    /// One compact decomposition family for every slot's `kappa_t P^c` fresh digit.
    pub fresh: Vec<Vec<NoiseDecompositionPlan>>,
    /// Per-slot ring-coefficient multiplicity for the compact fresh plans.
    #[serde(default = "one_usize_vec")]
    pub fresh_multiplicity: Vec<usize>,
    pub decoder: Vec<NoiseDecompositionPlan>,
}

fn one_usize() -> usize {
    1
}

fn one_usize_vec() -> Vec<usize> {
    Vec::new()
}

impl RefreshNoiseGainDecompositions {
    fn derive_route_gains(
        &self,
        slot_count: usize,
    ) -> Result<RefreshRouteNoiseGains, RefreshSetupError> {
        if self.state_scaling.len() != slot_count ||
            self.mask.is_empty() ||
            self.fresh.len() != slot_count ||
            self.mask_multiplicity == 0 ||
            (!self.fresh_multiplicity.is_empty() &&
                (self.fresh_multiplicity.len() != slot_count ||
                    self.fresh_multiplicity.iter().any(|value| *value == 0))) ||
            self.fresh.iter().any(|plans| plans.is_empty()) ||
            self.decoder.len() != slot_count
        {
            return Err(RefreshSetupError::InvalidParameters(
                "refresh decomposition slot count mismatch",
            ));
        }
        let derive = |plan: &NoiseDecompositionPlan| {
            NoiseGainPlanBuilder::derive_route_gains(std::slice::from_ref(plan))
                .map(|mut gains| gains.remove(0))
                .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh decomposition"))
        };
        let derive_sum = |plans: &[NoiseDecompositionPlan], multiplicity: usize| {
            let gains = plans.iter().map(derive).collect::<Result<Vec<_>, _>>()?;
            let mut aggregate =
                RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::zero() };
            for gain in gains {
                aggregate.l1 += gain.l1 * multiplicity;
                aggregate.l2_squared += gain.l2_squared * multiplicity;
            }
            if aggregate.l1.is_zero() || aggregate.l2_squared.is_zero() {
                return Err(RefreshSetupError::InvalidParameters("invalid refresh decomposition"));
            }
            Ok(aggregate)
        };
        Ok(RefreshRouteNoiseGains {
            state_scaling: self.state_scaling.iter().map(derive).collect::<Result<Vec<_>, _>>()?,
            mask: derive_sum(&self.mask, self.mask_multiplicity)?,
            fresh: self
                .fresh
                .iter()
                .enumerate()
                .map(|(slot, plans)| {
                    let multiplicity = self
                        .fresh_multiplicity
                        .get(slot)
                        .copied()
                        .unwrap_or(self.mask_multiplicity);
                    derive_sum(plans, multiplicity)
                })
                .collect::<Result<Vec<_>, _>>()?,
            decoder: self.decoder.iter().map(derive).collect::<Result<Vec<_>, _>>()?,
        })
    }
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
            decoder_preimage_bound: TrapdoorPreimageCutoffPolicy::Default,
            name: name.into(),
        }
    }

    /// Replaces the default trapdoor-sampler cutoff policy with an explicit policy.
    pub fn with_decoder_preimage_bound(mut self, bound: TrapdoorPreimageCutoffPolicy) -> Self {
        self.decoder_preimage_bound = bound;
        self
    }

    /// Builds the native sparse-LWR profile for a selected power-of-two `Q`.
    ///
    /// Refresh setup owns `p` and `W`; the caller supplies the concrete
    /// plaintext modulus because a refresh may contain several CRT moduli.
    /// Keeping this check explicit prevents a PBC bucket width from being
    /// accidentally reused as the Exponent-LUT domain width.
    pub fn sparse_lwr_profile(
        &self,
        q_l: usize,
    ) -> Result<crate::noise_refresh::prf::SparseLwrPrfProfile, RefreshSetupError> {
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?;
        if q_l < 2 || !q_l.is_power_of_two() {
            return Err(RefreshSetupError::InvalidParameters("native PRF Q must be a power of two"));
        }
        let width_log2 = q_l
            .trailing_zeros()
            .checked_sub(1)
            .ok_or(RefreshSetupError::InvalidParameters("native PRF Q is too small"))?;
        let stride_log2 = ring_dimension
            .trailing_zeros()
            .checked_sub(width_log2)
            .ok_or(RefreshSetupError::InvalidParameters("native PRF layout exceeds ring"))?;
        let layout = crate::layout::ExpLayout { stride_log2, width_log2 };
        crate::noise_refresh::prf::SparseLwrPrfProfile::new(layout, self.base_p / 2)
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

    /// Constructs the complete refresh-noise schema from setup parameters.
    ///
    /// This is intentionally independent of any PRF evaluator.  It binds the
    /// full modulus and every CRT modulus, derives both the joint tail event
    /// count and the mask-only smudging dimension from the refresh topology.
    /// Route-dependent digit variances remain
    /// unattached until preprocessing supplies the balanced decomposition
    /// plans (explicit or compact), because the route's concrete row `L2²`
    /// controls the transported variance.
    pub fn refresh_noise_parameters(&self) -> Result<RefreshNoiseParameters, RefreshSetupError> {
        self.validate()?;
        let eval_uint = |value: &mxx_ir_core::IntExpr,
                         message: &'static str|
         -> Result<BigUint, RefreshSetupError> {
            value
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters(message))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters(message))
        };
        let full_modulus = eval_uint(
            &self.refresh.full_modulus,
            "full refresh modulus must be a positive concrete integer",
        )?;
        let plaintext_moduli = self
            .refresh
            .crt_plaintext_moduli
            .iter()
            .map(|value| eval_uint(value, "CRT plaintext modulus must be positive"))
            .collect::<Result<Vec<_>, _>>()?;
        let component_columns = self.prf_component_count();
        let topology = RefreshTranscriptTopology {
            crt_slot_count: plaintext_moduli.len(),
            component_columns,
            coefficient_count: self.coefficient_count,
            mask_digit_count: self.mask_base_p_digit_count,
            fresh_digit_count: self.fresh_error_base_p_digit_count,
        };
        // The joint mask/fresh event count belongs only to the average tail
        // budget. Smudging uses the mask transcript dimension D, without the
        // P-adic digit axis or shared fresh labels.
        let smudging_coordinates = topology
            .mask_smudging_coordinates()
            .map_err(|_| RefreshSetupError::InvalidParameters("refresh topology overflows"))?;
        let mut refresh = RefreshNoiseParameters::new(
            plaintext_moduli,
            self.base_p.into(),
            self.mask_base_p_digit_count,
            self.fresh_error_base_p_digit_count,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh noise parameters"))?;
        refresh = refresh
            .with_global_modulus(full_modulus)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid full refresh modulus"))?
            .with_transcript_topology(topology.clone())
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh topology"))?
            .with_mask_smudging_parameters(MaskSmudgingParameters {
                lambda_mask: self.mask_statistical_security_bits,
                transcript_coordinates: smudging_coordinates,
            })
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid mask smudging schema"))?;
        Ok(refresh)
    }

    /// Builds the setup-owned noise snapshot from the exact producer plans.
    /// Every plan is derived from the producer's program/readout/refresh
    /// descriptors; geometry-only gains are rejected by the snapshot schema.
    fn build_noise_snapshot(
        &self,
        prf_program: SparseLwrPrfProgram,
        pbc_layout: &PbcPublicLayout,
        gain_plan: crate::noise_refresh::simulation::NoiseGainPlan,
        route_decompositions: RefreshNoiseGainDecompositions,
    ) -> Result<ExponentLutNoiseSnapshot, RefreshSetupError> {
        let family_active_counts = PbcActiveCellIndex::build(pbc_layout)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid PBC active-cell index"))?
            .bucket_active_widths()
            .collect::<Vec<_>>();
        self.build_noise_snapshot_with_family_active_counts(
            prf_program,
            pbc_layout,
            gain_plan,
            route_decompositions,
            family_active_counts,
        )
    }

    fn build_noise_snapshot_with_family_active_counts(
        &self,
        prf_program: SparseLwrPrfProgram,
        pbc_layout: &PbcPublicLayout,
        gain_plan: crate::noise_refresh::simulation::NoiseGainPlan,
        route_decompositions: RefreshNoiseGainDecompositions,
        family_active_counts: Vec<usize>,
    ) -> Result<ExponentLutNoiseSnapshot, RefreshSetupError> {
        self.validate()?;
        let decoder_preimage_bound = self.resolve_decoder_preimage_bound()?;
        let contract = RefreshPrfContract::from_program(&prf_program);
        // The producer declaration commits to the exact value-readout
        // semantics as well as the geometry.  Use the same readout-inclusive
        // identity here so the production snapshot can be compared directly
        // with that declaration.
        let setup_identity = setup_identity_digest(
            self,
            &contract,
            pbc_layout.layout_id,
            &decoder_preimage_bound,
            prf_program.readout.identity(),
        );
        let helper_error_bound = self
            .encoding_error_bound
            .evaluate(&ParamEnv::default())
            .map_err(|_| {
                RefreshSetupError::InvalidParameters("encoding error bound must be concrete")
            })?
            .to_biguint()
            .ok_or(RefreshSetupError::InvalidParameters(
                "encoding error bound must be non-negative",
            ))?;
        let expected_mask_route_count = Some(self.coefficient_count)
            .and_then(|value| value.checked_mul(self.mask_base_p_digit_count))
            .ok_or(RefreshSetupError::InvalidParameters("mask route count overflow"))?;
        let expected_fresh_route_count = Some(self.coefficient_count)
            .and_then(|value| value.checked_mul(self.fresh_error_base_p_digit_count))
            .ok_or(RefreshSetupError::InvalidParameters("fresh route count overflow"))?;
        let actual_mask_route_count = route_decompositions
            .mask
            .len()
            .checked_mul(route_decompositions.mask_multiplicity)
            .ok_or(RefreshSetupError::InvalidParameters("mask route count overflow"))?;
        let actual_fresh_route_counts = route_decompositions
            .fresh
            .iter()
            .enumerate()
            .map(|(slot, plans)| {
                plans
                    .len()
                    .checked_mul(
                        route_decompositions
                            .fresh_multiplicity
                            .get(slot)
                            .copied()
                            .unwrap_or(route_decompositions.mask_multiplicity),
                    )
                    .ok_or(RefreshSetupError::InvalidParameters("fresh route count overflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if actual_mask_route_count != expected_mask_route_count ||
            route_decompositions.fresh.len() != self.refresh.crt_plaintext_moduli.len() ||
            actual_fresh_route_counts.iter().any(|count| *count != expected_fresh_route_count)
        {
            return Err(RefreshSetupError::InvalidParameters(
                "refresh route count does not match P-adic topology",
            ));
        }
        let route_gains =
            route_decompositions.derive_route_gains(self.refresh.crt_plaintext_moduli.len())?;
        let centered_digit_variance = centered_base_p_digit_variance(&self.base_p.into())
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid centered digit variance"))?;
        let mask_digit_variances = vec![
            centered_digit_variance.scaled(&route_gains.mask.l2_squared);
            self.refresh.crt_plaintext_moduli.len()
        ];
        let fresh_digit_variances = route_gains
            .fresh
            .iter()
            .map(|gain| centered_digit_variance.scaled(&gain.l2_squared))
            .collect();
        let refresh = self
            .refresh_noise_parameters()?
            .with_route_gains(route_gains)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh route gains"))?
            .with_digit_variances(mask_digit_variances, fresh_digit_variances)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid digit variances"))?;
        ExponentLutNoiseSnapshot::from_setup(
            setup_identity,
            prf_program,
            helper_error_bound.clone(),
            self.average_initial_variance()?,
            helper_error_bound,
            self.average_initial_variance()?,
            refresh,
            gain_plan,
            family_active_counts,
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
        // The mask digit count changes the route topology.  Rebuild both the
        // compact route plans and the gain plan from the candidate instead of
        // reusing plans generated for the previous digit count.
        let readout_schema = compact_readout_plan_schema(&candidate, &prf_program)?;
        let active_cells = PbcActiveCellIndex::build(&pbc_layout)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid PBC active-cell index"))?;
        let (gain_plan, route_decompositions) = compact_noise_snapshot_plans(
            &candidate,
            &prf_program,
            &pbc_layout,
            &active_cells,
            &readout_schema,
        )?;
        let snapshot = candidate.build_noise_snapshot(
            prf_program,
            &pbc_layout,
            gain_plan,
            route_decompositions,
        )?;
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
        let ring_dimension = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_u32())
            .ok_or(RefreshSetupError::InvalidParameters(
                "decoder cutoff requires a concrete ring dimension that fits in u32",
            ))?;
        let gadget_base = self
            .layout
            .gadget_base
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_u32())
            .ok_or(RefreshSetupError::InvalidParameters(
                "decoder cutoff requires a concrete gadget base that fits in u32",
            ))?;
        let decoder_sigma =
            self.decoder_sigma.evaluate_f64(&ParamEnv::default()).map_err(|_| {
                RefreshSetupError::InvalidParameters(
                    "decoder cutoff requires a concrete finite sigma",
                )
            })?;
        let selected = self
            .decoder_preimage_bound
            .resolve(
                ring_dimension,
                self.layout.secret_dimension,
                self.layout.digit_count,
                gadget_base,
                decoder_sigma,
            )
            .map_err(|_| {
                RefreshSetupError::InvalidParameters(
                    "invalid decoder trapdoor preimage cutoff policy",
                )
            })?;
        Ok(IntExpr::constant(BigInt::from_biguint(num_bigint::Sign::Plus, selected)))
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
    layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    encoding_error_sigma: mxx_ir_core::RealExpr,
    encoding_error_bound: mxx_ir_core::IntExpr,
}

impl RefreshPrfInputs {
    /// Binds the label-major output of the online native PRF batch. This
    /// boundary has no plaintext family: the public monomials were already
    /// consumed by the PRF program, and only the resulting encoding vectors
    /// are routed into the refresh transcript.
    pub(crate) fn from_native_encoding_batch_outputs(
        parameters: &RefreshSetupParameters,
        expected_program: &SparseLwrPrfProgram,
        outputs: &NativePrfEncodingBatchOutputs,
        layout_id: crate::noise_refresh::pbc::PbcLayoutId,
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
        let labels = crate::noise_refresh::refresh::RefreshPrfLabelIndex::new(
            parameters.refresh_id,
            parameters.refresh.crt_plaintext_moduli.len(),
            parameters.prf_component_count(),
            parameters.coefficient_count,
            parameters.mask_base_p_digit_count,
            parameters.fresh_error_base_p_digit_count,
        )?;
        let family = RefreshPrfFamilyMaterial::from_native_encoding_batch_outputs(
            outputs,
            &labels,
            coverage.clone(),
            contract,
            layout_id,
        )?;
        let slot_count = parameters.refresh.crt_plaintext_moduli.len();
        let masks = (0..slot_count)
            .map(|slot| RefreshMaskMaterial::from_family(coverage.clone(), slot, family.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let fresh_error = RefreshFreshErrorMaterial::from_family(coverage, family)?;
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
        base_p: usize,
        route_preimages: Family<mxx_dsl::Preimage>,
    ) -> Result<Vec<BggEncodingWire>, RefreshSetupError> {
        let family = self
            .masks
            .first()
            .and_then(RefreshMaskMaterial::family_material)
            .ok_or(RefreshSetupError::InvalidParameters("missing native mask family"))?;
        aggregate_refresh_masks(base_p, family, self.masks.len(), route_preimages)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid native mask family"))
    }
}

/// Inputs for constructing the trusted refresh preprocessing graph.
///
/// `state`, `secret`, and `hash_key` are setup material. The resulting graph
/// exports named artifacts; later execution imports typed artifact handles
/// rather than reusing live producer wires.
pub struct RefreshPreprocessingRequest {
    /// Validated dimensions and CRT refresh declaration.
    pub parameters: RefreshSetupParameters,
    /// Setup-fixed public PRF and readout inputs. This value contains no
    /// label-dependent vectors, monomials, or online refresh outputs.
    pub public_prf: RefreshPublicPrfSetup,
    /// Private Exponent-LUT compiler used to aggregate PRF digits.
    pub compiler: ExponentLutEncodingCompiler,
    /// Setup-fixed public matrix of the state encoding. The state vector is
    /// online input and never enters this preprocessing request.
    pub state: BggPublicKeyWire,
    /// Setup secret `t` used by the preprocessing equations. This is not the
    /// online evaluator secret `s`; it crosses the trusted local boundary only
    /// as a typed private setup artifact and is never transmitted.
    pub setup_secret_t: Mat,
    /// The setup-generated selector GSW family. It is emitted by this same
    /// producer graph so the selector and decoder setup share one `t` sample.
    pub selector_gsw: Family<Mat>,
    /// Canonical PBC artifact name for the selector family.
    pub selector_gsw_name: String,
    /// Public typed decompositions of the setup PRF public-key family.  The
    /// corresponding `A[S]` matrices are reconstructed as `G*K` at each
    /// consumer and are never exported as raw artifacts.
    pub setup_public_key_rhs: Family<mxx_dsl::Preimage>,
    /// Hash key used for deterministic public matrix derivation.
    pub hash_key: Bytes,
}

/// Setup-fixed public inputs required by the refresh preprocessing producer.
///
/// The public compiler evaluates the accumulator and value-readout projection
/// once. The helper set may retain setup-private mask vectors for the online
/// readout artifact bundle, but no label-dependent input is accepted here.
pub struct RefreshPublicPrfSetup {
    pub program: SparseLwrPrfProgram,
    pub compiler: ExponentLutPublicKeyCompiler,
    pub initial: BggPublicKeyWire,
    pub selectors: PublicRhsFamily,
    pub inputs: NativePrfPublicInputs,
    pub layout: PbcPublicLayout,
    pub(crate) helpers: crate::encoding::NativeReadoutHelperFamilies,
    pub normalization_gadget: Mat,
}

impl RefreshPublicPrfSetup {
    /// Validates the public PRF setup against the refresh dimensions and
    /// compiler ring. The checks intentionally inspect only setup-fixed
    /// public matrices and declarations.
    pub fn validate(&self, parameters: &RefreshSetupParameters) -> Result<(), RefreshSetupError> {
        parameters.validate_dimensions()?;
        RefreshPrfContract::from_program(&self.program).validate_for(parameters)?;
        self.layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        if self.inputs.layout_id() != Some(self.layout.layout_id) ||
            self.selectors.layout_id() != Some(self.layout.layout_id) ||
            self.program.profile().ring_dimension()? !=
                self.compiler
                    .public_key
                    .ring
                    .matrix_type((1, 1))
                    .ring_dimension
                    .evaluate(&ParamEnv::default())
                    .map_err(|_| {
                        RefreshSetupError::InvalidParameters(
                            "public PRF compiler ring dimension must be concrete",
                        )
                    })?
                    .to_usize()
                    .ok_or(RefreshSetupError::InvalidParameters(
                        "public PRF compiler ring dimension must fit usize",
                    ))?
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        if self.initial.reveal_plaintext ||
            *self.initial.matrix.matrix_type() !=
                self.compiler.public_key.ring.matrix_type((
                    parameters.component_count,
                    parameters.layout.public_key_columns(),
                ))
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        if self.normalization_gadget.matrix_type().modulus !=
            self.compiler.public_key.ring.matrix_type((1, 1)).modulus ||
            self.normalization_gadget.matrix_type().ring_dimension !=
                self.compiler.public_key.ring.matrix_type((1, 1)).ring_dimension
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        Ok(())
    }
}

/// Names crossing the preprocessing/online boundary. The decomposition names
/// below identify deterministic, typed `Preimage` cache entries. The helper
/// protocol pair names identify t-only setup encodings (including their fresh
/// setup error); they are protocol data, not cache entries, and must not be
/// confused with online s-dependent label vectors.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefreshReadoutArtifactNames {
    pub target_decomposition: String,
    pub helper_switch_decompositions: String,
    pub helper_public_decompositions: String,
    pub normalization_decomposition: String,
}

/// Names and structural coordinates of the public deterministic program
/// cache. Each entry names the one `K_rhs` decomposition consumed online.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramCacheFamilyNames {
    pub step: usize,
    pub body_gate: usize,
    pub count: usize,
    /// Whether the flat artifact has shape `[output][member]`; false means
    /// the setup-fixed member family is shared by every output.
    pub output_indexed: bool,
    pub rhs: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramCacheArtifactNames {
    pub gates: Vec<ProgramCacheGateNames>,
    pub luts: Vec<ProgramCacheLutNames>,
    pub families: Vec<ProgramCacheFamilyNames>,
}

/// Artifact names for a LUT gate's per-branch public-key decompositions.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramCacheLutNames {
    pub step: usize,
    pub body_gate: Option<usize>,
    pub count: usize,
    pub width: usize,
    pub switch: String,
    pub branch: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramCacheGateNames {
    pub step: usize,
    pub rhs: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Canonical artifact names emitted by refresh preprocessing.
pub struct RefreshPreprocessingArtifactNames {
    /// Private setup secret `t`, retained locally for online/verification use.
    pub setup_secret_t: String,
    /// Setup-fixed selector GSW family used by online accumulation.
    pub selector_gsw: String,
    /// Public typed `K_rhs = G^-1(A[S])` family for setup PRF keys.
    pub setup_public_key_rhs: String,
    /// Public typed decompositions of each CRT `kappa_t` state scale.
    pub state_scaling: Vec<String>,
    /// Public typed route decompositions for mask and fresh label families.
    pub mask_route_rhs_decompositions: String,
    pub fresh_route_rhs_decompositions: String,
    /// Public matrix carried by the online state encoding.
    pub state_public_matrix: String,
    /// Public random matrix used in the target equation.
    pub a_prime: String,
    /// Public trapdoor matrix `B`.
    pub public_matrix_b: String,
    /// Exact public refresh target matrix for each CRT slot.
    pub refresh_targets: Vec<String>,
    /// Per-slot sampled preimage matrices.
    pub preimages: Vec<String>,
    /// Public value-readout and typed decomposition-cache artifacts.
    pub readout: RefreshReadoutArtifactNames,
    /// Public typed Preimage families for every program multiplication.
    pub program_cache: ProgramCacheArtifactNames,
}

#[derive(Clone)]
pub(crate) struct RefreshPreprocessingWires {
    pub(crate) setup_secret_t: Mat,
    pub(crate) selector_gsw: Family<Mat>,
    pub(crate) setup_public_key_rhs: Family<mxx_dsl::Preimage>,
    pub(crate) state_scaling: Vec<mxx_dsl::Preimage>,
    pub(crate) mask_route_rhs_decompositions: Family<mxx_dsl::Preimage>,
    pub(crate) fresh_route_rhs_decompositions: Family<mxx_dsl::Preimage>,
    pub(crate) state: BggPublicKeyWire,
    pub(crate) a_prime: Mat,
    pub(crate) public_b: Mat,
    pub(crate) refresh_targets: Vec<Mat>,
    pub(crate) scaled_fresh: Vec<Mat>,
    pub(crate) masks: Vec<Mat>,
    pub(crate) preimages: Vec<mxx_dsl::Preimage>,
    pub(crate) readout_artifacts: NativePrfReadoutFamilyArtifacts,
    pub(crate) program_cache: ProgramCacheBatchBindings,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Public declaration of a preprocessing producer and its artifact schema.
///
/// The identity commits to all dimensions, setup identities, decoder policy,
/// scales, and canonical artifact names.
pub struct RefreshPreprocessingDeclaration {
    /// Digest of this complete declaration.
    pub identity: [u8; 32],
    /// PBC layout identity used by the PRF outputs.
    pub pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    /// Refresh instance identity.
    pub refresh_id: [u8; 32],
    /// Sparse-LWR/Exponent-LUT program identity.
    pub program_id: ExponentLutProgramId,
    /// Plaintext modulus bound to the sparse-LWR PRF terminal.
    pub prf_q: usize,
    /// Raw scalar output modulus bound to the sparse-LWR PRF terminal.
    pub prf_p: usize,
    /// Logical LUT width bound to the sparse-LWR PRF program.
    pub prf_native_width: usize,
    /// Ring dimension bound to the sparse-LWR PRF program.
    pub prf_ring_dimension: usize,
    /// Algebraic form promised by the PRF terminal.
    /// Output wire promised by the PRF terminal.
    pub prf_output_wire: crate::program::WireId,
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
    /// Stable identity of the value-form readout used by this setup.
    pub readout_identity: crate::noise_refresh::readout::ValueReadoutIdentity,
    /// Canonical non-identity trace automorphisms for the readout helpers.
    /// Keeping this in the declaration lets an importer reconstruct the
    /// typed helper set without requiring the online program or its labels.
    pub readout_helper_sigmas: Vec<usize>,
}

#[derive(Clone)]
/// Runtime setup material imported from a validated producer declaration.
///
/// Private setup values and preimages remain crate-visible only. Consumers use
/// them through [`RefreshOnlineEvaluationRequest`] after all manifest,
/// identity, and artifact-schema checks succeed.
pub struct ImportedRefreshSetup {
    parameters: RefreshSetupParameters,
    pub(crate) state: BggPublicKeyWire,
    pub(crate) setup_secret_t: Mat,
    pub(crate) a_prime: Mat,
    pub(crate) public_b: Mat,
    pub(crate) refresh_targets: Vec<Mat>,
    pub(crate) readout_artifacts: NativePrfReadoutFamilyArtifacts,
    pub(crate) program_cache: ProgramCacheBatchBindings,
    pub(crate) setup_public_key_rhs: Family<mxx_dsl::Preimage>,
    pub(crate) state_scaling: Vec<mxx_dsl::Preimage>,
    pub(crate) mask_route_rhs_decompositions: Family<mxx_dsl::Preimage>,
    pub(crate) fresh_route_rhs_decompositions: Family<mxx_dsl::Preimage>,
    pub(crate) preimages: Vec<mxx_dsl::Preimage>,
    declaration: RefreshPreprocessingDeclaration,
}
impl std::fmt::Debug for ImportedRefreshSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedRefreshSetup")
            .field("identity", &self.declaration.identity)
            .field("slot_count", &self.state_scaling.len())
            .finish()
    }
}

/// Explicit online evaluation request for one imported refresh setup.
///
/// The state encoding, secret `s`, and label-major online inputs are owned by
/// the caller; preprocessing contributes only imported public projections,
/// readout artifacts, public decoder targets, and typed `K_t` preimages. In
/// particular, preprocessing never contributes the decoder anchor `b`. This is
/// the production boundary used by benchmark graph construction as well as by
/// a runtime evaluator.
pub struct RefreshOnlineEvaluationRequest<'a> {
    pub refresh: &'a RefreshCompiler,
    pub compiler: &'a ExponentLutEncodingCompiler,
    pub setup: &'a ImportedRefreshSetup,
    pub program: &'a SparseLwrPrfProgram,
    /// The fresh online secret `s` shared by the state, PRF inputs, helper
    /// masks, and the one decoder anchor `b = sB + e_b`. It is never exported.
    pub online_secret_s: Mat,
    pub state: BggEncodingWire,
    pub selectors: &'a EncodingRhsFamily,
    pub inputs: &'a NativePrfOnlineBatchInputs,
    /// Fresh `s`-dependent helper mask encodings, keyed by canonical sigma.
    /// Setup-fixed switches and decompositions are loaded from artifacts.
    pub online_masks: Family<Mat>,
    pub layout: &'a PbcPublicLayout,
}

/// Outputs retained at the online stage boundary for routing and tests.
pub struct RefreshOnlineEvaluation {
    pub accumulation: NativePrfEncodingBatchOutputs,
    pub readout: NativePrfEncodingBatchOutputs,
    pub manifest: RefreshSetupManifest,
    pub refreshed: crate::noise_refresh::refresh::RefreshResult,
}

impl<'a> RefreshOnlineEvaluationRequest<'a> {
    /// Runs online encoding accumulation, imported-artifact value readout,
    /// centered mask/fresh routing, and decoder-preimage application.
    pub fn evaluate(self) -> Result<RefreshOnlineEvaluation, RefreshSetupError> {
        let accumulation = self
            .program
            .compile_encoding_batch_with_preprocessed_artifacts(
                self.compiler,
                self.state.clone(),
                self.selectors,
                self.inputs,
                self.layout,
                &self.setup.program_cache,
            )
            .map_err(|error| RefreshSetupError::SetupStage {
                stage: "evaluate online PRF accumulation",
                detail: error.to_string(),
            })?;
        let readout = self
            .program
            .execute_value_readout_batch_with_mask_family(
                &self.compiler.bgg,
                &accumulation,
                &self.setup.readout_artifacts,
                self.online_masks.clone(),
            )
            .map_err(|error| RefreshSetupError::SetupStage {
                stage: "evaluate online PRF readout",
                detail: error.to_string(),
            })?;
        let manifest = self
            .refresh
            .bind_online_outputs(
                self.compiler,
                self.setup,
                self.online_secret_s,
                self.state,
                self.program,
                &readout,
            )
            .map_err(|error| RefreshSetupError::SetupStage {
                stage: "bind online refresh inputs",
                detail: error.to_string(),
            })?;
        let refreshed = self.refresh.refresh(self.compiler, &manifest).map_err(|error| {
            RefreshSetupError::SetupStage {
                stage: "apply online refresh",
                detail: error.to_string(),
            }
        })?;
        Ok(RefreshOnlineEvaluation { accumulation, readout, manifest, refreshed })
    }
}

/// Builds the compact gain plan for a sampled preimage matrix consumed
/// directly by the runtime. The typed preimage bound is checked against the
/// producer schema before its exact matrix shape and bound are retained.
/// Enforces the production online boundary: all deterministic decompositions
/// must have crossed preprocessing as typed artifacts. This scan traverses
/// every frozen scope, including structural family bodies.
fn validate_cache_required_online_graph(graph: &BuiltGraph) -> Result<(), RefreshSetupError> {
    if graph
        .graph
        .scopes()
        .values()
        .flat_map(|scope| scope.nodes())
        .any(|node| matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. }))
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    Ok(())
}

#[cfg(test)]
fn direct_bounded_preimage_plan(
    preimage: &mxx_dsl::Preimage,
    base: &BigUint,
    schema_bound: BigUint,
) -> Result<NoiseDecompositionPlan, RefreshSetupError> {
    let rows = preimage
        .matrix_type()
        .rows
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("preimage rows must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("preimage rows must fit usize"))?;
    let columns = preimage
        .matrix_type()
        .columns
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("preimage columns must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("preimage columns must fit usize"))?;
    let ring = preimage
        .matrix_type()
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| {
            RefreshSetupError::InvalidParameters("preimage ring dimension must be concrete")
        })?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("preimage ring dimension must fit usize"))?;
    let actual_bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("preimage bound must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("preimage bound must be non-negative"))?;
    if actual_bound != schema_bound {
        return Err(RefreshSetupError::IdentityMismatch);
    }
    NoiseDecompositionPlan::bounded_matrix(base.clone(), rows, columns, ring, actual_bound)
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid bounded preimage action"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactPreimageSchema {
    rows: usize,
    columns: usize,
    ring_dimension: usize,
    max_coefficient_bound: BigUint,
}

fn compact_preimage_schema(
    preimage: &mxx_dsl::Preimage,
) -> Result<CompactPreimageSchema, RefreshSetupError> {
    let matrix = preimage.matrix_type();
    let concrete = |value: &IntExpr, what: &'static str| {
        value
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters(what))?
            .to_usize()
            .ok_or(RefreshSetupError::InvalidParameters(what))
    };
    let bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("preimage bound must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("preimage bound must be non-negative"))?;
    Ok(CompactPreimageSchema {
        rows: concrete(&matrix.rows, "preimage rows must be concrete")?,
        columns: concrete(&matrix.columns, "preimage columns must be concrete")?,
        ring_dimension: concrete(
            &matrix.ring_dimension,
            "preimage ring dimension must be concrete",
        )?,
        max_coefficient_bound: bound,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactReadoutPlanSchema {
    target: CompactPreimageSchema,
    normalization: CompactPreimageSchema,
    trace: Vec<(CompactPreimageSchema, CompactPreimageSchema)>,
    decoder: Vec<CompactPreimageSchema>,
}

fn compact_readout_plan_schema(
    parameters: &RefreshSetupParameters,
    program: &SparseLwrPrfProgram,
) -> Result<CompactReadoutPlanSchema, RefreshSetupError> {
    let ring_dimension = parameters
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
    let digit_count = parameters.layout.digit_count;
    let decomposed_rows = parameters
        .layout
        .secret_dimension
        .checked_mul(digit_count)
        .ok_or(RefreshSetupError::InvalidParameters("decomposed readout rows overflow"))?;
    let direct_bound = parameters
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))? /
        2u8;
    let direct = |rows: usize, columns: usize| CompactPreimageSchema {
        rows,
        columns,
        ring_dimension,
        max_coefficient_bound: direct_bound.clone(),
    };
    let normalization_size = parameters
        .component_count
        .checked_mul(digit_count)
        .ok_or(RefreshSetupError::InvalidParameters("normalization shape overflow"))?;
    let trace_count = program
        .readout
        .trace_plan(ring_dimension)
        .map_err(|_| RefreshSetupError::InvalidManifest)?
        .helper_count();
    let decoder_rows = parameters
        .component_count
        .checked_mul(
            digit_count
                .checked_add(2)
                .ok_or(RefreshSetupError::InvalidParameters("decoder shape overflow"))?,
        )
        .ok_or(RefreshSetupError::InvalidParameters("decoder shape overflow"))?;
    let decoder_bound = parameters
        .resolve_decoder_preimage_bound()?
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("decoder bound must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("decoder bound must be non-negative"))?;
    Ok(CompactReadoutPlanSchema {
        target: direct(decomposed_rows, parameters.layout.public_key_columns()),
        normalization: direct(normalization_size, normalization_size),
        trace: (0..trace_count)
            .map(|_| {
                (
                    direct(decomposed_rows, parameters.layout.public_key_columns()),
                    direct(decomposed_rows, parameters.layout.public_key_columns()),
                )
            })
            .collect(),
        decoder: (0..parameters.refresh.crt_plaintext_moduli.len())
            .map(|_| CompactPreimageSchema {
                rows: decoder_rows,
                columns: parameters.layout.public_key_columns(),
                ring_dimension,
                max_coefficient_bound: decoder_bound.clone(),
            })
            .collect(),
    })
}

fn bounded_plan_from_schema(
    base: &BigUint,
    schema: &CompactPreimageSchema,
) -> Result<NoiseDecompositionPlan, RefreshSetupError> {
    NoiseDecompositionPlan::bounded_matrix(
        base.clone(),
        schema.rows,
        schema.columns,
        schema.ring_dimension,
        schema.max_coefficient_bound.clone(),
    )
    .map_err(|_| RefreshSetupError::InvalidParameters("invalid compact preimage action"))
}

fn producer_readout_plan_schema(
    parameters: &RefreshSetupParameters,
    artifacts: &NativePrfReadoutFamilyArtifacts,
    decoder: &[mxx_dsl::Preimage],
) -> Result<CompactReadoutPlanSchema, RefreshSetupError> {
    let direct_bound = parameters
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))? /
        2u8;
    let family_schema = |matrix: &mxx_ir_core::types::MatrixType| {
        let concrete = |value: &IntExpr, what: &'static str| {
            value
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters(what))?
                .to_usize()
                .ok_or(RefreshSetupError::InvalidParameters(what))
        };
        Ok::<_, RefreshSetupError>(CompactPreimageSchema {
            rows: concrete(&matrix.rows, "readout helper rows must be concrete")?,
            columns: concrete(&matrix.columns, "readout helper columns must be concrete")?,
            ring_dimension: concrete(
                &matrix.ring_dimension,
                "readout helper ring dimension must be concrete",
            )?,
            max_coefficient_bound: direct_bound.clone(),
        })
    };
    let switch_schema = family_schema(artifacts.helper_switch_decompositions().element_type())?;
    let public_schema = family_schema(artifacts.helper_public_decompositions().element_type())?;
    let trace = (0..artifacts.helper_sigmas().len())
        .map(|_| (switch_schema.clone(), public_schema.clone()))
        .collect();
    let decoder = decoder.iter().map(compact_preimage_schema).collect::<Result<Vec<_>, _>>()?;
    if decoder.len() != parameters.refresh.crt_plaintext_moduli.len() {
        return Err(RefreshSetupError::InvalidManifest);
    }
    Ok(CompactReadoutPlanSchema {
        target: compact_preimage_schema(&artifacts.target_decompositions().get_static(0))?,
        normalization: compact_preimage_schema(artifacts.normalization_decomposition())?,
        trace,
        decoder,
    })
}

/// Derives the compact noise plans used by the application-specific snapshot
/// without constructing any label-major executable graph.  The dimensions are
/// the same dimensions produced by the public producer: a decomposed BGG
/// matrix has `secret_dimension * gadget_digit_count` rows and the original
/// public-key columns, while normalization has `component_count * digits`
/// rows and columns.  Directly consumed preimages intentionally use the
/// bounded-matrix source with their typed decomposition bound.
fn compact_noise_snapshot_plans(
    parameters: &RefreshSetupParameters,
    program: &SparseLwrPrfProgram,
    pbc_layout: &PbcPublicLayout,
    active_cells: &PbcActiveCellIndex,
    readout_schema: &CompactReadoutPlanSchema,
) -> Result<(NoiseGainPlan, RefreshNoiseGainDecompositions), RefreshSetupError> {
    let p = parameters;
    RefreshPrfContract::from_program(program).validate_for(p)?;
    if program.block_count() != pbc_layout.parameters.bucket_count ||
        program.bucket_width() != pbc_layout.bucket_width
    {
        return Err(RefreshSetupError::InvalidParameters(
            "PRF program topology does not match PBC layout",
        ));
    }
    if active_cells.bucket_active_widths().count() != pbc_layout.cells.len() {
        return Err(RefreshSetupError::InvalidParameters("PBC active-cell index shape mismatch"));
    }
    let ring_dimension = p
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
    let base = p
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))?;
    let digit_count = p.layout.digit_count;
    let target_rows = p.layout.secret_dimension;
    let target_columns = p.layout.public_key_columns();
    let crt_moduli = p
        .refresh
        .crt_plaintext_moduli
        .iter()
        .map(|modulus| {
            modulus
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters("CRT modulus must be concrete"))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("CRT modulus must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let digits_per_tower = digit_count
        .checked_div(crt_moduli.len())
        .filter(|digits| *digits != 0 && *digits * crt_moduli.len() == digit_count)
        .ok_or(RefreshSetupError::InvalidParameters(
            "gadget digits do not match the CRT tower count",
        ))?;
    let bind_crt = |plan: NoiseDecompositionPlan| {
        plan.bind_crt_layout(crt_moduli.clone(), digits_per_tower)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid compact CRT decomposition"))
    };
    let public_bound = p
        .refresh
        .full_modulus
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("full modulus must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("full modulus must be positive"))? /
        2u8;
    let public_action = || {
        NoiseDecompositionPlan::dense_bound(
            base.clone(),
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            public_bound.clone(),
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid public action plan"))
        .and_then(&bind_crt)
    };
    let unit_action = || {
        NoiseDecompositionPlan::unit(
            base.clone(),
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            0,
            0,
            0,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid unit action plan"))
        .and_then(&bind_crt)
    };
    let mut gate_decompositions = Vec::new();
    let mut gates = Vec::new();
    let mut operation_index = 0usize;
    let mut add_gate = |step_index: usize,
                        branch_index: usize,
                        operation: GateNoiseOperation,
                        action: NoiseDecompositionPlan,
                        helper_action: Option<NoiseDecompositionPlan>| {
        gate_decompositions.push(GateNoiseDecomposition { action, helper_action });
        gates.push(GateNoiseGains {
            step_index,
            operation_index,
            branch_index,
            operation,
            gate: "refresh-producer".to_owned(),
            action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
            helper_action: None,
            inherited_bound: BigUint::zero(),
            fresh_bound: BigUint::zero(),
            output_bound: BigUint::zero(),
            inherited_variance: AverageVariance::zero(),
            fresh_variance: AverageVariance::zero(),
            output_variance: AverageVariance::zero(),
        });
        operation_index += 1;
    };
    let mut family_index = 0usize;
    for (step_index, step) in program.accumulation.steps().iter().enumerate() {
        match step {
            ProgramStep::Family { body, .. } => {
                let count = active_cells
                    .bucket_active_width(family_index)
                    .ok_or(RefreshSetupError::InvalidParameters("family layout mismatch"))?;
                family_index += 1;
                if count == 0 {
                    return Err(RefreshSetupError::InvalidParameters("invalid program family body"));
                }
                // All active family members use the same typed public/private
                // action schema.  Keep one pair of gains for the structural
                // body and apply it to `count` members in the simulator;
                // materializing one pair per member makes search scale with
                // the PBC family rather than with the program shape.
                for (body_gate, gate) in body.gates().iter().enumerate() {
                    let (operation, action) = match gate {
                        crate::program::Gate::MulPrivate { .. } => {
                            (GateNoiseOperation::MulPrivate, public_action()?)
                        }
                        crate::program::Gate::MulPublic { .. } => {
                            (GateNoiseOperation::MulPublic, public_action()?)
                        }
                        crate::program::Gate::Add { .. } => {
                            (GateNoiseOperation::Add, unit_action()?)
                        }
                        crate::program::Gate::Lut { .. } => {
                            (GateNoiseOperation::Lut, public_action()?)
                        }
                    };
                    add_gate(step_index, body_gate, operation, action, None);
                }
                add_gate(
                    step_index,
                    body.gates().len(),
                    GateNoiseOperation::Add,
                    unit_action()?,
                    None,
                );
            }
            ProgramStep::Gate(gate) => match gate {
                Gate::MulPrivate { .. } => {
                    add_gate(step_index, 0, GateNoiseOperation::MulPrivate, public_action()?, None)
                }
                Gate::MulPublic { .. } => {
                    add_gate(step_index, 0, GateNoiseOperation::MulPublic, public_action()?, None)
                }
                Gate::Add { .. } => {
                    add_gate(step_index, 0, GateNoiseOperation::Add, unit_action()?, None)
                }
                Gate::Lut { lut, .. } => {
                    let branches = program
                        .accumulation
                        .lut(*lut)
                        .ok_or(RefreshSetupError::InvalidParameters("missing LUT"))?
                        .values()
                        .len();
                    for branch_index in 0..branches {
                        add_gate(
                            step_index,
                            branch_index,
                            GateNoiseOperation::Lut,
                            public_action()?,
                            Some(public_action()?),
                        );
                    }
                }
            },
        }
    }
    if family_index != pbc_layout.cells.len() {
        return Err(RefreshSetupError::InvalidParameters("program family count mismatch"));
    }

    let target_plan = bounded_plan_from_schema(&base, &readout_schema.target)?;
    let normalization_plan = bounded_plan_from_schema(&base, &readout_schema.normalization)?;
    let trace_plan = readout_schema
        .trace
        .iter()
        .map(|(action, helper)| {
            Ok((bounded_plan_from_schema(&base, action)?, bounded_plan_from_schema(&base, helper)?))
        })
        .collect::<Result<Vec<_>, RefreshSetupError>>()?;
    let readout = ValueReadoutNoiseGains {
        target: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
        identity_trace: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
        nonidentity_trace: trace_plan
            .iter()
            .map(|_| TraceNoiseGains {
                action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
                helper_action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
            })
            .collect(),
        normalization: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
        inherited_bound: BigUint::zero(),
        output_bound: BigUint::zero(),
        output_variance: AverageVariance::zero(),
    };
    let gain_plan =
        NoiseGainPlanBuilder::new(gate_decompositions, target_plan, trace_plan, normalization_plan)
            .map_err(|_| {
                RefreshSetupError::InvalidParameters("invalid compact producer gain plan")
            })?
            .build(gates, readout)
            .map_err(|_| {
                RefreshSetupError::InvalidParameters("invalid compact producer gain plan")
            })?;

    let scales = (0..p.refresh.crt_plaintext_moduli.len())
        .map(|slot| {
            let expression = p.refresh.scale_expression(slot)?;
            expression
                .evaluate(&ParamEnv::default())
                .map_err(|_| {
                    RefreshSetupError::InvalidParameters("refresh scale must be concrete")
                })?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("refresh scale must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let state_scaling = scales
        .iter()
        .map(|scale| {
            NoiseDecompositionPlan::gadget_scalar(
                base.clone(),
                digit_count,
                target_rows,
                target_rows * digit_count,
                ring_dimension,
                scale.clone().into(),
            )
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh scale plan"))
            .and_then(bind_crt)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let route_plan = |value: BigInt, ring_coefficient: usize| {
        NoiseDecompositionPlan::sparse_monomial(
            base.clone(),
            digit_count,
            target_rows,
            p.prf_component_count(),
            ring_dimension,
            target_rows.saturating_sub(1),
            0,
            ring_coefficient,
            value,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid PRF route plan"))
        .and_then(bind_crt)
    };
    let mask = (0..p.mask_base_p_digit_count)
        .map(|c| {
            let value = BigUint::from(p.base_p).pow(c as u32);
            route_plan(value.into(), 0)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fresh = scales
        .iter()
        .map(|scale| {
            (0..p.fresh_error_base_p_digit_count)
                .map(|c| {
                    let value = scale * BigUint::from(p.base_p).pow(c as u32);
                    route_plan(value.into(), 0)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let decoder = readout_schema
        .decoder
        .iter()
        .map(|schema| bounded_plan_from_schema(&base, schema))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        gain_plan,
        RefreshNoiseGainDecompositions {
            state_scaling,
            mask,
            mask_multiplicity: p.coefficient_count,
            fresh_multiplicity: vec![p.coefficient_count; fresh.len()],
            fresh,
            decoder,
        },
    ))
}

/// Rebuilds only the mask/fresh route decomposition for a changed `d_m`.
/// Gate/readout plans are independent of the mask digit count and are cached
/// by the parameter-search context.
fn compact_mask_routes(
    parameters: &RefreshSetupParameters,
) -> Result<(Vec<NoiseDecompositionPlan>, usize), RefreshSetupError> {
    let p = parameters;
    let ring_dimension = p
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
    let base = p
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))?;
    let digit_count = p.layout.digit_count;
    let target_rows = p.layout.secret_dimension;
    let crt_moduli = p
        .refresh
        .crt_plaintext_moduli
        .iter()
        .map(|modulus| {
            modulus
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters("CRT modulus must be concrete"))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("CRT modulus must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let digits_per_tower = digit_count
        .checked_div(crt_moduli.len())
        .filter(|digits| *digits != 0 && *digits * crt_moduli.len() == digit_count)
        .ok_or(RefreshSetupError::InvalidParameters(
            "gadget digits do not match the CRT tower count",
        ))?;
    let bind_crt = |plan: NoiseDecompositionPlan| {
        plan.bind_crt_layout(crt_moduli.clone(), digits_per_tower)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid compact CRT decomposition"))
    };
    let route_plan = |value: BigInt| {
        NoiseDecompositionPlan::sparse_monomial(
            base.clone(),
            digit_count,
            target_rows,
            p.prf_component_count(),
            ring_dimension,
            target_rows.saturating_sub(1),
            0,
            0,
            value,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid PRF route plan"))
        .and_then(bind_crt)
    };
    let mask = (0..p.mask_base_p_digit_count)
        .map(|c| route_plan(BigUint::from(p.base_p).pow(c as u32).into()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((mask, p.coefficient_count))
}

fn compact_noise_snapshot_routes(
    parameters: &RefreshSetupParameters,
    readout_schema: &CompactReadoutPlanSchema,
) -> Result<RefreshNoiseGainDecompositions, RefreshSetupError> {
    let p = parameters;
    let ring_dimension = p
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
    let base = p
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))?;
    let digit_count = p.layout.digit_count;
    let target_rows = p.layout.secret_dimension;
    let crt_moduli = p
        .refresh
        .crt_plaintext_moduli
        .iter()
        .map(|modulus| {
            modulus
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters("CRT modulus must be concrete"))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("CRT modulus must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let digits_per_tower = digit_count
        .checked_div(crt_moduli.len())
        .filter(|digits| *digits != 0 && *digits * crt_moduli.len() == digit_count)
        .ok_or(RefreshSetupError::InvalidParameters(
            "gadget digits do not match the CRT tower count",
        ))?;
    let bind_crt = |plan: NoiseDecompositionPlan| {
        plan.bind_crt_layout(crt_moduli.clone(), digits_per_tower)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid compact CRT decomposition"))
    };
    let scales = (0..p.refresh.crt_plaintext_moduli.len())
        .map(|slot| {
            let expression = p.refresh.scale_expression(slot)?;
            expression
                .evaluate(&ParamEnv::default())
                .map_err(|_| {
                    RefreshSetupError::InvalidParameters("refresh scale must be concrete")
                })?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("refresh scale must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let state_scaling = scales
        .iter()
        .map(|scale| {
            NoiseDecompositionPlan::gadget_scalar(
                base.clone(),
                digit_count,
                target_rows,
                target_rows * digit_count,
                ring_dimension,
                scale.clone().into(),
            )
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh scale plan"))
            .and_then(&bind_crt)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let route_plan = |value: BigInt, ring_coefficient: usize| {
        NoiseDecompositionPlan::sparse_monomial(
            base.clone(),
            digit_count,
            target_rows,
            p.prf_component_count(),
            ring_dimension,
            target_rows.saturating_sub(1),
            0,
            ring_coefficient,
            value,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid PRF route plan"))
        .and_then(&bind_crt)
    };
    let mask = (0..p.mask_base_p_digit_count)
        .map(|c| route_plan(BigUint::from(p.base_p).pow(c as u32).into(), 0))
        .collect::<Result<Vec<_>, _>>()?;
    let fresh = scales
        .iter()
        .map(|scale| {
            (0..p.fresh_error_base_p_digit_count)
                .map(|c| route_plan((scale * BigUint::from(p.base_p).pow(c as u32)).into(), 0))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let decoder = readout_schema
        .decoder
        .iter()
        .map(|schema| bounded_plan_from_schema(&base, schema))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RefreshNoiseGainDecompositions {
        state_scaling,
        mask,
        mask_multiplicity: p.coefficient_count,
        fresh_multiplicity: vec![p.coefficient_count; fresh.len()],
        fresh,
        decoder,
    })
}

/// Immutable producer for the trusted refresh preprocessing graph.
///
/// Building this value samples setup artifacts, freezes the graph, and
/// validates the declaration and output schema. The producer retains
/// private setup internally; callers should export its declaration and use
/// [`ImportedRefreshSetup::import`] at an execution boundary.
pub struct RefreshPreprocessingProducer {
    request: RefreshPreprocessingRequest,
    wires: RefreshPreprocessingWires,
    built: BuiltGraph,
    declaration: RefreshPreprocessingDeclaration,
}

impl RefreshPreprocessingProducer {
    /// Builds and validates the immutable preprocessing graph.
    pub fn build(request: RefreshPreprocessingRequest) -> Result<Self, RefreshSetupError> {
        request.parameters.validate_dimensions()?;
        let decoder_preimage_bound = request.parameters.resolve_decoder_preimage_bound()?;
        request.public_prf.validate(&request.parameters)?;
        let p = &request.parameters;
        let ring = p.layout.ring();
        if request.state.reveal_plaintext ||
            request.state.matrix.value_handle() !=
                request.public_prf.initial.matrix.value_handle()
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let output_count = refresh_prf_output_count(p)?;
        if request.public_prf.inputs.output_count() != output_count {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let public_preprocessing =
            request.public_prf.program.compile_public_batch_with_preprocessing(
                &request.public_prf.compiler,
                request.public_prf.initial.clone(),
                &request.public_prf.selectors,
                &request.public_prf.inputs,
                &request.public_prf.layout,
            )?;
        // The readout cache schema is the deterministic `Mat::decompose`
        // bound, not the Gaussian helper cutoff `B_chi`.
        let (readout_artifacts, public_readout_output) =
            with_benchmark_role(BenchmarkRole::PublicReadout, || {
                let readout_artifacts = request
                    .public_prf
                    .program
                    .produce_public_readout_family_artifacts(
                        &request.public_prf.compiler,
                        public_preprocessing.outputs.clone(),
                        &request.public_prf.helpers,
                        request.public_prf.normalization_gadget.clone(),
                    )
                    .map_err(|_| RefreshSetupError::InvalidManifest)?;
                let public_readout_output = readout_artifacts
                    .project_public_batch(
                        &request.public_prf.program.readout,
                        &request.public_prf.compiler.public_key,
                        public_preprocessing.outputs.clone(),
                        request.public_prf.helpers.public_masks.clone(),
                    )
                    .map_err(|_| RefreshSetupError::InvalidManifest)?;
                Ok::<_, RefreshSetupError>((readout_artifacts, public_readout_output))
            })?;
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
        let mut scales = Vec::new();
        for slot in 0..p.refresh.crt_plaintext_moduli.len() {
            // `scale = mu_t = q/q_t` is the scalar represented by the
            // setup-fixed RNS scaling LUT for CRT slot t.
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
        let fresh_group = p
            .prf_component_count()
            .checked_mul(p.coefficient_count)
            .and_then(|value| value.checked_mul(p.fresh_error_base_p_digit_count))
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let mask_route_count = refresh_route_family_count(p, p.mask_base_p_digit_count)?;
        let fresh_route_count = refresh_route_family_count(p, p.fresh_error_base_p_digit_count)?;
        // The route constructor uses only the matrix type. Selecting one
        // symbolic member does not enumerate or share its value.
        let public_key_for_routes = public_readout_output.get_static(0);
        let mask_route_rhs_decompositions = crate::noise_refresh::refresh::route_preimage_family(
            &ring,
            &public_key_for_routes,
            p.base_p,
            p.prf_component_count(),
            p.coefficient_count,
            p.mask_base_p_digit_count,
            mask_route_count,
            None,
            decomposition_base.clone(),
            IntExpr::constant(decomposition_digit_count),
        )?;
        let slot_indices = Parallel::range(fresh_route_count)
            .map_values(|index| index.as_int().div(Int::constant(fresh_group)))?;
        let repeated_scales = Family::pack(scales.clone())?.parallel_gather(slot_indices)?;
        let fresh_route_rhs_decompositions = crate::noise_refresh::refresh::route_preimage_family(
            &ring,
            &public_key_for_routes,
            p.base_p,
            p.prf_component_count(),
            p.coefficient_count,
            p.fresh_error_base_p_digit_count,
            fresh_route_count,
            Some(repeated_scales),
            decomposition_base.clone(),
            IntExpr::constant(decomposition_digit_count),
        )?;
        let active_count = PbcActiveCellIndex::build(&request.public_prf.layout)
            .map_err(|_| RefreshSetupError::InvalidManifest)?
            .len();
        let expected_setup_public_key_count =
            output_count.checked_mul(active_count).ok_or(RefreshSetupError::InvalidManifest)?;
        if concrete_preimage_family_count(&request.setup_public_key_rhs)? !=
            expected_setup_public_key_count ||
            concrete_preimage_family_count(&mask_route_rhs_decompositions)? != mask_route_count ||
            concrete_preimage_family_count(&fresh_route_rhs_decompositions)? != fresh_route_count
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let (public_masks, public_fresh) = aggregate_public_refresh_prf(
            &request.public_prf.compiler,
            public_readout_output,
            p.base_p,
            p.refresh.crt_plaintext_moduli.len(),
            p.prf_component_count(),
            p.coefficient_count,
            p.mask_base_p_digit_count,
            p.fresh_error_base_p_digit_count,
            scales.clone(),
        )?;
        let mask_public_family = Family::pack(public_masks.clone())?;
        let scale_family = Family::pack(scales.clone())?;
        let scaled_fresh_public_family = Family::pack(public_fresh.clone())?;
        let state_public = request.state.clone();
        let state_scaling_family = Family::pack(state_scaling.clone())?;
        let scaled_state_family =
            state_scaling_family.parallel_map_values(|_, decomposition| {
                decomposition.mul_small_rhs(state_public.matrix.clone())
            })?;
        // Build every slot target in one structural loop, then sample all
        // preimages in a second structural loop. Each family element computes
        // `T_t = A_t + A_{m,t} + A_{e,t} - mu_t A'`; the slot index is structural,
        // never plaintext. This path contains only public matrices from the
        // state and PRF/readout projections.
        let target_family = Family::<Mat>::parallel_zip_many_values(
            vec![mask_public_family, scale_family, scaled_fresh_public_family, scaled_state_family],
            |_, mut items| {
                let scaled_state = items.pop().expect("scaled state family");
                let fresh_public = items.pop().expect("scaled fresh family");
                let scale = items.pop().expect("scale family");
                let mask_public = items.pop().expect("mask family");
                scaled_state + mask_public + fresh_public - scale * a_prime.clone()
            },
        )?;
        // Sample one preimage K_t per target; the trapdoor is captured once
        // and broadcast, so the resulting relation is `B K_t = T_t`.
        let refresh_targets =
            (0..public_masks.len()).map(|slot| target_family.get_static(slot)).collect();
        let preimage_family = target_family.parallel_map_values(|_, target| {
            trapdoor.sample_preimage(target, (b_columns, p.layout.public_key_columns()))
        })?;
        let ks = (0..public_masks.len()).map(|slot| preimage_family.get_static(slot)).collect();
        let ring_dimension = p
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
            .to_usize()
            .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
        let readout_helper_sigmas = request
            .public_prf
            .program
            .readout
            .trace_plan(ring_dimension)
            .map_err(|_| RefreshSetupError::InvalidManifest)?
            .non_identity_helpers()
            .to_vec();
        if readout_helper_sigmas.len() != p.lut_width.saturating_sub(1) {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let mut names = canonical_names(
            p,
            &RefreshPrfContract::from_program(&request.public_prf.program),
            request.public_prf.layout.layout_id,
            &decoder_preimage_bound,
            request.public_prf.program.readout.identity(),
            request.selector_gsw_name.clone(),
        );
        names.program_cache = canonical_program_cache_names(
            &request.public_prf.program,
            &request.public_prf.layout,
            setup_identity_digest(
                p,
                &RefreshPrfContract::from_program(&request.public_prf.program),
                request.public_prf.layout.layout_id,
                &decoder_preimage_bound,
                request.public_prf.program.readout.identity(),
            ),
        )?;
        let declaration = make_declaration(
            p,
            &RefreshPrfContract::from_program(&request.public_prf.program),
            names.clone(),
            request.public_prf.layout.layout_id,
            &decoder_preimage_bound,
            request.public_prf.program.readout.identity(),
            readout_helper_sigmas.clone(),
        );
        let wires = RefreshPreprocessingWires {
            setup_secret_t: request.setup_secret_t.clone(),
            selector_gsw: request.selector_gsw.clone(),
            setup_public_key_rhs: request.setup_public_key_rhs.clone(),
            state_scaling,
            mask_route_rhs_decompositions,
            fresh_route_rhs_decompositions,
            state: request.state.clone(),
            a_prime,
            public_b,
            refresh_targets,
            scaled_fresh: public_fresh.clone(),
            masks: public_masks.clone(),
            preimages: ks,
            readout_artifacts,
            program_cache: public_preprocessing.cache,
        };
        let mut context = DslContext::new("mxx-exponent-lut-refresh-setup");
        context = add_setup_outputs(context, &wires, &names)?;
        let built = context.build()?;
        let producer = Self { request, wires, built, declaration };
        producer.validate_built()?;
        Ok(producer)
    }
    /// Returns the frozen preprocessing graph and its artifact outputs.
    pub fn built(&self) -> &BuiltGraph {
        &self.built
    }

    /// Returns the public declaration describing this producer's artifacts.
    pub fn declaration(&self) -> &RefreshPreprocessingDeclaration {
        &self.declaration
    }

    /// Derives the compact operational-noise plans from this producer's
    /// actual program, readout artifacts, and refresh descriptors.  The
    /// returned plans retain exact matrix shapes, ring dimension, CRT towers,
    /// and the enforced preimage bounds; callers must pass them unchanged to
    /// [`RefreshSetupParameters::build_noise_snapshot`].
    pub(crate) fn noise_snapshot_plans(
        &self,
    ) -> Result<(NoiseGainPlan, RefreshNoiseGainDecompositions), RefreshSetupError> {
        let parameters = &self.request.parameters;
        let expected_schema =
            compact_readout_plan_schema(parameters, &self.request.public_prf.program)?;
        let actual_schema = producer_readout_plan_schema(
            parameters,
            &self.wires.readout_artifacts,
            &self.wires.preimages,
        )?;
        if expected_schema != actual_schema {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let active_cells = PbcActiveCellIndex::build(&self.request.public_prf.layout)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid PBC active-cell index"))?;
        compact_noise_snapshot_plans(
            parameters,
            &self.request.public_prf.program,
            &self.request.public_prf.layout,
            &active_cells,
            &actual_schema,
        )
    }
    /// Checks the actual setup equations and all canonical output roles.
    pub fn validate_built(&self) -> Result<(), RefreshSetupError> {
        self.declaration.validate_built(&self.wires)?;
        let p = &self.request.parameters;
        let decoder_preimage_bound = p.resolve_decoder_preimage_bound()?;
        let contract = RefreshPrfContract::from_program(&self.request.public_prf.program);
        let w = &self.wires;
        let active_count = PbcActiveCellIndex::build(&self.request.public_prf.layout)
            .map_err(|_| RefreshSetupError::InvalidManifest)?
            .len();
        let setup_public_key_count = refresh_prf_output_count(p)?
            .checked_mul(active_count)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let mask_route_count = refresh_route_family_count(p, p.mask_base_p_digit_count)?;
        let fresh_route_count = refresh_route_family_count(p, p.fresh_error_base_p_digit_count)?;
        if concrete_preimage_family_count(&w.setup_public_key_rhs)? != setup_public_key_count ||
            concrete_preimage_family_count(&w.mask_route_rhs_decompositions)? != mask_route_count ||
            concrete_preimage_family_count(&w.fresh_route_rhs_decompositions)? !=
                fresh_route_count
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        if self.declaration.identity !=
            setup_identity_digest(
                p,
                &contract,
                self.request.public_prf.layout.layout_id,
                &decoder_preimage_bound,
                self.request.public_prf.program.readout.identity(),
            ) ||
            names_without_program_cache(&self.declaration.names) !=
                canonical_names(
                    p,
                    &contract,
                    self.request.public_prf.layout.layout_id,
                    &decoder_preimage_bound,
                    self.request.public_prf.program.readout.identity(),
                    self.request.selector_gsw_name.clone(),
                )
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let expected_program_cache = canonical_program_cache_names(
            &self.request.public_prf.program,
            &self.request.public_prf.layout,
            self.declaration.identity,
        )?;
        if self.declaration.names.program_cache != expected_program_cache {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        validate_program_cache_names(
            &self.declaration.names.program_cache,
            self.declaration.identity,
        )?;
        if w.masks.is_empty() ||
            w.masks.len() != p.refresh.crt_plaintext_moduli.len() ||
            w.preimages.len() != w.masks.len() ||
            w.scaled_fresh.len() != w.masks.len() ||
            w.refresh_targets.len() != w.masks.len() ||
            w.setup_public_key_rhs
                .count()
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|count| count.to_usize()) ==
                Some(0)
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let ring = p.layout.ring();
        let cols = p.layout.public_key_columns();
        let b_columns = p.component_count * (p.layout.digit_count + 2);
        let setup_key_schema = ring.matrix_type((p.component_count * p.layout.digit_count, cols));
        if w.setup_public_key_rhs.element_type() != &setup_key_schema {
            return Err(RefreshSetupError::InvalidManifest);
        }
        if w.state.reveal_plaintext ||
            !same_matrix_type(
                w.state.matrix.matrix_type(),
                &ring.matrix_type((p.component_count, cols)),
            ) ||
            !same_matrix_type(
                w.state.matrix.matrix_type(),
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
        for (slot, (mask, k)) in w.masks.iter().zip(w.preimages.iter()).enumerate() {
            if !same_matrix_type(mask.matrix_type(), &ring.matrix_type((p.component_count, cols))) ||
                !same_matrix_type(
                    w.scaled_fresh[slot].matrix_type(),
                    &ring.matrix_type((p.component_count, cols)),
                ) ||
                !same_matrix_type(
                    w.refresh_targets[slot].matrix_type(),
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
            let scaled_state = self
                .request
                .public_prf
                .compiler
                .public_key
                .large_scalar_mul(
                    &BggPublicKeyWire { matrix: w.state.matrix.clone(), reveal_plaintext: false },
                    &scale,
                )
                .matrix;
            let target = scaled_state + w.masks[slot].clone() + w.scaled_fresh[slot].clone() -
                scale * w.a_prime.clone();
            if !same_matrix_type(
                k.clone().mul_small_rhs(w.public_b.clone()).matrix_type(),
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

/// Builds the one shared online decoder anchor `b=sB+e_b` used by every CRT slot.
fn shared_decoder_base(
    ring: &mxx_dsl::Ring,
    secret: Mat,
    public_b: Mat,
    encoding_error_sigma: mxx_ir_core::RealExpr,
    encoding_error_bound: mxx_ir_core::IntExpr,
) -> BggEncodingWire {
    let b_columns = public_b.matrix_type().columns.clone();
    let anchor_error = ring.gaussian((1, b_columns), encoding_error_sigma, encoding_error_bound);
    BggEncodingWire { vector: secret * public_b.clone() + anchor_error, plaintext: None }
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
    if slot_count == 0 || scales.len() != slot_count {
        return Err(RefreshSetupError::InvalidManifest);
    }

    if !family_count_is(&outputs, total)? {
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

fn refresh_prf_output_count(p: &RefreshSetupParameters) -> Result<usize, RefreshSetupError> {
    let mask = p
        .refresh
        .crt_plaintext_moduli
        .len()
        .checked_mul(p.prf_component_count())
        .and_then(|value| value.checked_mul(p.coefficient_count))
        .and_then(|value| value.checked_mul(p.mask_base_p_digit_count))
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let fresh = p
        .prf_component_count()
        .checked_mul(p.coefficient_count)
        .and_then(|value| value.checked_mul(p.fresh_error_base_p_digit_count))
        .ok_or(RefreshSetupError::InvalidManifest)?;
    mask.checked_add(fresh).ok_or(RefreshSetupError::InvalidManifest)
}

fn refresh_route_family_count(
    p: &RefreshSetupParameters,
    digit_count: usize,
) -> Result<usize, RefreshSetupError> {
    p.refresh
        .crt_plaintext_moduli
        .len()
        .checked_mul(p.prf_component_count())
        .and_then(|value| value.checked_mul(p.coefficient_count))
        .and_then(|value| value.checked_mul(digit_count))
        .ok_or(RefreshSetupError::InvalidManifest)
}

fn concrete_preimage_family_count(
    family: &Family<mxx_dsl::Preimage>,
) -> Result<usize, RefreshSetupError> {
    family
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|count| count.to_usize())
        .ok_or(RefreshSetupError::InvalidManifest)
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
        let group_quotient = mxx_ir_core::IntExpr::FloorDiv(
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
        let digit_quotient = mxx_ir_core::IntExpr::FloorDiv(
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
        let coefficient_quotient = mxx_ir_core::IntExpr::FloorDiv(
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
        let component = mxx_ir_core::IntExpr::FloorDiv(
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
        if self.program_id == ExponentLutProgramId::from_digest([0; 32]) {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        Ok(())
    }

    pub(crate) fn validate_built(
        &self,
        w: &RefreshPreprocessingWires,
    ) -> Result<(), RefreshSetupError> {
        self.validate_slot_count()?;
        if self.slot_count != w.masks.len() ||
            self.component_count == 0 ||
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
            self.names.preimages.len() != self.slot_count ||
            self.names.state_scaling.len() != self.slot_count ||
            self.names.setup_public_key_rhs.is_empty()
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
            setup_identity_digest(
                parameters,
                &declaration_contract(self),
                self.pbc_layout_id,
                &decoder_preimage_bound,
                self.readout_identity,
            ) ||
            names_without_program_cache(&self.names) !=
                canonical_names(
                    parameters,
                    &declaration_contract(self),
                    self.pbc_layout_id,
                    &decoder_preimage_bound,
                    self.readout_identity,
                    self.names.selector_gsw.clone(),
                ) ||
            self.slot_count != parameters.refresh.crt_plaintext_moduli.len() ||
            manifest.ir_version != IR_VERSION
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        validate_program_cache_names(&self.names.program_cache, self.identity)?;
        validate_manifest(manifest, &self.names, parameters)
    }
}

impl ImportedRefreshSetup {
    /// Returns the validated setup parameters used by this import.
    pub fn parameters(&self) -> &RefreshSetupParameters {
        &self.parameters
    }

    /// Returns the declaration that was checked during import.
    pub fn declaration(&self) -> &RefreshPreprocessingDeclaration {
        &self.declaration
    }

    /// Returns the setup-fixed public state key paired with the online state
    /// vector supplied to [`RefreshOnlineEvaluationRequest`].
    pub fn state_public_key(&self) -> &BggPublicKeyWire {
        &self.state
    }

    /// Returns the fixed private setup secret `t` imported from the producer.
    /// It is a local private artifact, not a transmitted protocol value or a
    /// deterministic decomposition cache.
    pub(crate) fn setup_secret_t(&self) -> &Mat {
        &self.setup_secret_t
    }

    /// Imports preprocessing artifacts after validating their declaration
    /// identity, confidentiality, bounds, and concrete matrix shapes.
    pub fn import(
        parameters: RefreshSetupParameters,
        declaration: RefreshPreprocessingDeclaration,
        manifest: &Manifest,
    ) -> Result<Self, RefreshSetupError> {
        parameters.validate()?;
        declaration.validate_slot_count()?;
        declaration.validate_prf_contract(&parameters)?;
        let decoder_preimage_bound = parameters.resolve_decoder_preimage_bound()?;
        if manifest.ir_version != IR_VERSION ||
            declaration.identity !=
                setup_identity_digest(
                    &parameters,
                    &declaration_contract(&declaration),
                    declaration.pbc_layout_id,
                    &decoder_preimage_bound,
                    declaration.readout_identity,
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
            names_without_program_cache(&declaration.names) !=
                canonical_names(
                    &parameters,
                    &declaration_contract(&declaration),
                    declaration.pbc_layout_id,
                    &decoder_preimage_bound,
                    declaration.readout_identity,
                    declaration.names.selector_gsw.clone(),
                )
        {
            return Err(RefreshSetupError::IdentityMismatch)
        };
        validate_program_cache_names(&declaration.names.program_cache, declaration.identity)?;
        validate_manifest(manifest, &declaration.names, &parameters)?;
        let production_id = manifest.production_id.clone();
        let n = &declaration.names;
        let cols = parameters.layout.public_key_columns();
        let b_columns = parameters.component_count * (parameters.layout.digit_count + 2);
        // The manifest has already checked every concrete type. Preserve the
        // parameter-derived symbolic shape in the consumer graph; no producer
        // graph or live producer wire crosses this artifact boundary.
        let imported = |name: &str,
                        confidentiality: ArtifactConfidentiality,
                        expected: &mxx_ir_core::types::MatrixType|
         -> Result<Mat, RefreshSetupError> {
            let ring =
                mxx_dsl::Ring::new(expected.modulus.clone(), expected.ring_dimension.clone());
            Ok(ring.artifact_input(
                production_id.clone(),
                name.to_owned(),
                (expected.rows.clone(), expected.columns.clone()),
                confidentiality,
            ))
        };
        let state_public_matrix = imported(
            &n.state_public_matrix,
            ArtifactConfidentiality::Public,
            &parameters.layout.ring().matrix_type((parameters.component_count, cols)),
        )?;
        let setup_secret_t = imported(
            &n.setup_secret_t,
            ArtifactConfidentiality::Private,
            &parameters.layout.ring().matrix_type((1, parameters.layout.secret_dimension)),
        )?;
        let a_prime = imported(
            &n.a_prime,
            ArtifactConfidentiality::Public,
            &parameters.layout.ring().matrix_type((parameters.component_count, cols)),
        )?;
        let public_b = imported(
            &n.public_matrix_b,
            ArtifactConfidentiality::Public,
            &parameters.layout.ring().matrix_type((parameters.component_count, b_columns)),
        )?;
        let refresh_targets = n
            .refresh_targets
            .iter()
            .map(|name| {
                imported(
                    name,
                    ArtifactConfidentiality::Public,
                    &parameters.layout.ring().matrix_type((parameters.component_count, cols)),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let state = BggPublicKeyWire { matrix: state_public_matrix, reveal_plaintext: false };
        let mut ks = Vec::new();
        for x in &n.preimages {
            // Keep K typed as a preimage across the artifact boundary. The
            // consumer therefore applies the bounded witness through `mul_small_rhs`
            // instead of materializing it as an untyped matrix.
            let matrix = parameters.layout.ring().matrix_type((b_columns, cols));
            let ring = mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension.clone());
            ks.push(ring.preimage_artifact_input(
                production_id.clone(),
                x.to_owned(),
                (matrix.rows.clone(), matrix.columns.clone()),
                decoder_preimage_bound.clone(),
                ArtifactConfidentiality::Private,
            ));
        }
        let gadget_base = parameters
            .layout
            .gadget_base
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        // Readout Preimage artifacts carry Mat::decompose's deterministic
        // coefficient bound floor(beta/2). B_chi remains the helper/anchor
        // error cutoff and is not the cache schema bound.
        let readout_bound = gadget_base.clone() / BigInt::from(2u8);
        let ring_dimension = parameters
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?
            .to_usize()
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let readout_schema = crate::noise_refresh::readout::ReadoutPreimageSchema::new(
            gadget_base.clone(),
            parameters.layout.digit_count,
            readout_bound.clone(),
        )
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let _readout_layout = {
            let width = declaration.prf_native_width;
            if width == 0 ||
                !width.is_power_of_two() ||
                ring_dimension == 0 ||
                width > ring_dimension
            {
                return Err(RefreshSetupError::InvalidManifest);
            }
            crate::layout::ExpLayout {
                stride_log2: ring_dimension.trailing_zeros() - width.trailing_zeros(),
                width_log2: width.trailing_zeros(),
            }
        };
        let readout_preimage = |name: &str| -> Result<mxx_dsl::Preimage, RefreshSetupError> {
            let matrix = parameters
                .layout
                .ring()
                .matrix_type((parameters.component_count * parameters.layout.digit_count, cols));
            let ring = mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension.clone());
            Ok(ring.preimage_artifact_input(
                production_id.clone(),
                name.to_owned(),
                (matrix.rows.clone(), matrix.columns.clone()),
                readout_bound.clone(),
                ArtifactConfidentiality::Public,
            ))
        };
        let output_count = refresh_prf_output_count(&parameters)?;
        let target_decompositions = {
            let matrix = parameters
                .layout
                .ring()
                .matrix_type((parameters.component_count * parameters.layout.digit_count, cols));
            let ring = mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension.clone());
            ring.preimage_family_artifact_input(
                production_id.clone(),
                n.readout.target_decomposition.clone(),
                output_count,
                (matrix.rows, matrix.columns),
                readout_bound.clone(),
                ArtifactConfidentiality::Public,
            )
        };
        let normalization_decomposition = readout_preimage(&n.readout.normalization_decomposition)?;
        let helper_count = declaration.readout_helper_sigmas.len();
        let readout_preimage_family = |name: &str, count: usize| {
            let matrix = parameters
                .layout
                .ring()
                .matrix_type((parameters.component_count * parameters.layout.digit_count, cols));
            let ring = mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension.clone());
            ring.preimage_family_artifact_input(
                production_id.clone(),
                name.to_owned(),
                count,
                (matrix.rows.clone(), matrix.columns.clone()),
                readout_bound.clone(),
                ArtifactConfidentiality::Public,
            )
        };
        let readout_artifacts = NativePrfReadoutFamilyArtifacts::new(
            target_decompositions,
            readout_schema.clone(),
            declaration.readout_helper_sigmas.clone(),
            readout_preimage_family(&n.readout.helper_switch_decompositions, helper_count),
            readout_preimage_family(
                &n.readout.helper_public_decompositions,
                refresh_prf_output_count(&parameters)?
                    .checked_mul(helper_count)
                    .ok_or(RefreshSetupError::InvalidManifest)?,
            ),
            normalization_decomposition.clone(),
            readout_schema,
            output_count,
        )
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let cache_matrix = parameters
            .layout
            .ring()
            .matrix_type((parameters.component_count * parameters.layout.digit_count, cols));
        let family_count = |name: &str| -> Result<usize, RefreshSetupError> {
            let count = manifest
                .artifacts
                .get(name)
                .and_then(|artifact| artifact.family_count)
                .ok_or(RefreshSetupError::InvalidManifest)?;
            match count {
                count if count > 0 => Ok(count),
                _ => Err(RefreshSetupError::InvalidManifest),
            }
        };
        let setup_public_key_matrix = cache_matrix.clone();
        let setup_public_key_count = family_count(&n.setup_public_key_rhs)?;
        let setup_public_key_ring = mxx_dsl::Ring::new(
            setup_public_key_matrix.modulus.clone(),
            setup_public_key_matrix.ring_dimension.clone(),
        );
        let setup_public_key_rhs = setup_public_key_ring.preimage_family_artifact_input(
            production_id.clone(),
            n.setup_public_key_rhs.clone(),
            setup_public_key_count,
            (setup_public_key_matrix.rows.clone(), setup_public_key_matrix.columns.clone()),
            readout_bound.clone(),
            ArtifactConfidentiality::Public,
        );
        let state_scaling = n
            .state_scaling
            .iter()
            .map(|name| {
                let matrix = cache_matrix.clone();
                let ring =
                    mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension.clone());
                Ok(ring.preimage_artifact_input(
                    production_id.clone(),
                    name.clone(),
                    (matrix.rows.clone(), matrix.columns.clone()),
                    readout_bound.clone(),
                    ArtifactConfidentiality::Public,
                ))
            })
            .collect::<Result<Vec<_>, RefreshSetupError>>()?;
        let import_route_family = |name: &str| -> Result<Family<Preimage>, RefreshSetupError> {
            let matrix = cache_matrix.clone();
            let count = family_count(name)?;
            let ring = mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension.clone());
            Ok(ring.preimage_family_artifact_input(
                production_id.clone(),
                name.to_owned(),
                count,
                (matrix.rows.clone(), matrix.columns.clone()),
                readout_bound.clone(),
                ArtifactConfidentiality::Public,
            ))
        };
        let mask_route_rhs_decompositions = import_route_family(&n.mask_route_rhs_decompositions)?;
        let fresh_route_rhs_decompositions =
            import_route_family(&n.fresh_route_rhs_decompositions)?;
        if !n.program_cache.gates.is_empty() || !n.program_cache.luts.is_empty() {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let output_count = refresh_prf_output_count(&parameters)?;
        let mut program_cache = ProgramCacheBatchBindings::new(output_count)?;
        for entry in &n.program_cache.families {
            let flat_count = if entry.output_indexed {
                output_count.checked_mul(entry.count).ok_or(RefreshSetupError::InvalidManifest)?
            } else {
                entry.count
            };
            if family_count(&entry.rhs)? != flat_count {
                return Err(RefreshSetupError::InvalidManifest);
            }
            let rhs_matrix = cache_matrix.clone();
            let shape = IntExpr::constant(flat_count);
            let rhs_ring =
                mxx_dsl::Ring::new(rhs_matrix.modulus.clone(), rhs_matrix.ring_dimension.clone());
            let family = rhs_ring.preimage_family_artifact_input(
                production_id.clone(),
                entry.rhs.clone(),
                shape.clone(),
                (rhs_matrix.rows.clone(), rhs_matrix.columns.clone()),
                readout_bound.clone(),
                ArtifactConfidentiality::Public,
            );
            if entry.output_indexed {
                program_cache.insert_family(entry.step, entry.body_gate, entry.count, family)?;
            } else {
                program_cache.insert_shared_family(
                    entry.step,
                    entry.body_gate,
                    entry.count,
                    family,
                )?;
            }
        }
        Ok(Self {
            parameters,
            state,
            setup_secret_t,
            a_prime,
            public_b,
            refresh_targets,
            readout_artifacts,
            program_cache,
            setup_public_key_rhs,
            state_scaling,
            mask_route_rhs_decompositions,
            fresh_route_rhs_decompositions,
            preimages: ks,
            declaration,
        })
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
        outputs: &NativePrfEncodingBatchOutputs,
    ) -> Result<RefreshSetupManifest, RefreshSetupError> {
        if setup.parameters.refresh.crt_plaintext_moduli != self.crt_plaintext_moduli ||
            setup.parameters.refresh.reconstruction_coefficients !=
                self.reconstruction_coefficients ||
            setup.parameters.refresh.full_modulus != self.full_modulus ||
            setup.declaration.readout_identity != program.readout.identity()
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        if !same_matrix_type(
            online_secret_s.matrix_type(),
            &setup
                .parameters
                .layout
                .ring()
                .matrix_type((1, setup.parameters.layout.secret_dimension)),
        ) || !same_matrix_type(
            state.vector.matrix_type(),
            &setup
                .parameters
                .layout
                .ring()
                .matrix_type((1, setup.parameters.layout.public_key_columns())),
        ) {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        // The decoder anchor is online material because it depends on the
        // fresh evaluator secret `s`. Sample its one fresh error here, once,
        // and share the resulting wire across every CRT slot.
        let decoder_base = shared_decoder_base(
            &setup.parameters.layout.ring(),
            online_secret_s,
            setup.public_b.clone(),
            setup.parameters.encoding_error_sigma.clone(),
            setup.parameters.encoding_error_bound.clone(),
        );
        let online = RefreshPrfInputs::from_native_encoding_batch_outputs(
            &setup.parameters,
            program,
            outputs,
            setup.declaration.pbc_layout_id,
        )?;
        let masks = online.aggregate_masks(
            setup.parameters.base_p,
            setup.mask_route_rhs_decompositions.clone(),
        )?;
        let scales = (0..setup.parameters.refresh.crt_plaintext_moduli.len())
            .map(|slot| {
                Ok(setup.parameters.layout.ring().polynomial([self.scale_expression(slot)?]))
            })
            .collect::<Result<Vec<_>, RefreshError>>()?;
        let scaled_fresh = aggregate_refresh_fresh_error_per_slot(
            setup.parameters.base_p,
            &online.fresh_error,
            scales,
            setup.fresh_route_rhs_decompositions.clone(),
        )?;
        Ok(self.bind_imported_wires(
            compiler,
            state,
            setup.state.clone(),
            setup.a_prime.clone(),
            setup.public_b.clone(),
            setup.refresh_targets.clone(),
            scaled_fresh,
            masks,
            setup.state_scaling.clone(),
            decoder_base,
            setup.preimages.clone(),
        )?)
    }
}

/// Derives all preprocessing artifact names from the public setup identity.
///
/// Slot-indexed names are emitted in CRT order, so an importer cannot silently
/// exchange two mask or preimage artifacts.
fn canonical_names(
    p: &RefreshSetupParameters,
    contract: &RefreshPrfContract,
    pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    decoder_preimage_bound: &mxx_ir_core::IntExpr,
    readout_identity: crate::noise_refresh::readout::ValueReadoutIdentity,
    selector_gsw_name: String,
) -> RefreshPreprocessingArtifactNames {
    let d =
        setup_identity_digest(p, contract, pbc_layout_id, decoder_preimage_bound, readout_identity);
    let n = |r: &str, i: usize| format!("mxx-refresh-{}-{r}-{i}", hex(&d));
    let s = p.refresh.crt_plaintext_moduli.len();
    RefreshPreprocessingArtifactNames {
        setup_secret_t: n("setup-secret-t", 0),
        selector_gsw: selector_gsw_name,
        setup_public_key_rhs: n("setup-public-key-k-rhs", 0),
        state_scaling: (0..s).map(|i| n("state-scaling-k-rhs", i)).collect(),
        mask_route_rhs_decompositions: n("mask-route-k-rhs", 0),
        fresh_route_rhs_decompositions: n("fresh-route-k-rhs", 0),
        state_public_matrix: n("state-public", 0),
        a_prime: n("a-prime", 0),
        public_matrix_b: n("public-b", 0),
        refresh_targets: (0..s).map(|i| n("refresh-target", i)).collect(),
        preimages: (0..s).map(|i| n("preimage", i)).collect(),
        readout: RefreshReadoutArtifactNames {
            target_decomposition: n("readout-target-decomposition", 0),
            helper_switch_decompositions: n("readout-helper-switch-family", 0),
            helper_public_decompositions: n("readout-helper-public-family", 0),
            normalization_decomposition: n("readout-normalization-decomposition", 0),
        },
        program_cache: ProgramCacheArtifactNames::default(),
    }
}

fn canonical_program_cache_names(
    program: &SparseLwrPrfProgram,
    layout: &PbcPublicLayout,
    setup_identity: [u8; 32],
) -> Result<ProgramCacheArtifactNames, RefreshSetupError> {
    let active =
        PbcActiveCellIndex::build(layout).map_err(|_| RefreshSetupError::InvalidManifest)?;
    let prefix = format!("mxx-refresh-{}-program-cache", hex(&setup_identity));
    let role_name = |role: ProgramCacheRole| match role {
        ProgramCacheRole::K_rhs => "k-rhs",
        ProgramCacheRole::LutSwitch => "lut-switch",
        ProgramCacheRole::LutBranch => "lut-branch",
    };
    let mut gates = Vec::new();
    let mut luts = Vec::new();
    let mut families = Vec::new();
    for declaration in program_cache_declarations(&program.accumulation) {
        match declaration.location {
            ProgramCacheLocation::Gate { step } => {
                let entry =
                    gates.iter_mut().find(|entry: &&mut ProgramCacheGateNames| entry.step == step);
                if let Some(entry) = entry {
                    match declaration.role {
                        ProgramCacheRole::K_rhs => {
                            entry.rhs =
                                format!("{prefix}-gate-{step}-{}", role_name(declaration.role))
                        }
                        ProgramCacheRole::LutSwitch | ProgramCacheRole::LutBranch => {
                            return Err(RefreshSetupError::InvalidManifest)
                        }
                    }
                } else {
                    gates.push(ProgramCacheGateNames {
                        step,
                        rhs: format!("{prefix}-gate-{step}-k-rhs"),
                    });
                }
            }
            ProgramCacheLocation::Family { step, body_gate } => {
                let block = program.accumulation.steps()[..step]
                    .iter()
                    .filter(|step| matches!(step, ProgramStep::Family { .. }))
                    .count();
                let count =
                    active.bucket_active_width(block).ok_or(RefreshSetupError::InvalidManifest)?;
                let entry = families.iter_mut().find(|entry: &&mut ProgramCacheFamilyNames| {
                    entry.step == step && entry.body_gate == body_gate
                });
                if entry.is_none() {
                    let output_indexed = matches!(
                        program.accumulation.steps().get(step),
                        Some(ProgramStep::Family { body, .. })
                            if matches!(body.gates().get(body_gate), Some(Gate::MulPublic { .. }))
                    );
                    families.push(ProgramCacheFamilyNames {
                        step,
                        body_gate,
                        count,
                        output_indexed,
                        rhs: format!("{prefix}-family-{step}-body-{body_gate}-k-rhs"),
                    });
                }
            }
            ProgramCacheLocation::Lut { step, branch } => {
                let entry = luts.iter_mut().find(|entry: &&mut ProgramCacheLutNames| {
                    entry.step == step && entry.body_gate.is_none()
                });
                let entry = if let Some(entry) = entry {
                    entry
                } else {
                    let width = match program.accumulation.steps().get(step) {
                        Some(ProgramStep::Gate(Gate::Lut { lut, .. })) => program
                            .accumulation
                            .lut(*lut)
                            .map(|table| table.values().len())
                            .ok_or(RefreshSetupError::InvalidManifest)?,
                        _ => return Err(RefreshSetupError::InvalidManifest),
                    };
                    luts.push(ProgramCacheLutNames {
                        step,
                        body_gate: None,
                        count: 1,
                        width,
                        switch: format!("{prefix}-lut-{step}-lut-switch-family"),
                        branch: format!("{prefix}-lut-{step}-lut-branch-family"),
                    });
                    luts.last_mut().expect("inserted LUT cache names")
                };
                match declaration.role {
                    ProgramCacheRole::LutSwitch | ProgramCacheRole::LutBranch => {
                        if branch >= entry.width {
                            return Err(RefreshSetupError::InvalidManifest);
                        }
                    }
                    _ => return Err(RefreshSetupError::InvalidManifest),
                }
            }
            ProgramCacheLocation::FamilyLut { step, body_gate, branch } => {
                let entry = luts.iter_mut().find(|entry: &&mut ProgramCacheLutNames| {
                    entry.step == step && entry.body_gate == Some(body_gate)
                });
                let entry = if let Some(entry) = entry {
                    entry
                } else {
                    let (body, width) = match program.accumulation.steps().get(step) {
                        Some(ProgramStep::Family { body, .. }) => {
                            let width = match body.gates().get(body_gate) {
                                Some(Gate::Lut { lut, .. }) => program
                                    .accumulation
                                    .lut(*lut)
                                    .map(|table| table.values().len())
                                    .ok_or(RefreshSetupError::InvalidManifest)?,
                                _ => return Err(RefreshSetupError::InvalidManifest),
                            };
                            (body, width)
                        }
                        _ => return Err(RefreshSetupError::InvalidManifest),
                    };
                    let _ = body;
                    let block = program.accumulation.steps()[..step]
                        .iter()
                        .filter(|step| matches!(step, ProgramStep::Family { .. }))
                        .count();
                    let count = active
                        .bucket_active_width(block)
                        .ok_or(RefreshSetupError::InvalidManifest)?;
                    luts.push(ProgramCacheLutNames {
                        step,
                        body_gate: Some(body_gate),
                        count,
                        width,
                        switch: format!(
                            "{prefix}-family-{step}-body-{body_gate}-lut-switch-family"
                        ),
                        branch: format!(
                            "{prefix}-family-{step}-body-{body_gate}-lut-branch-family"
                        ),
                    });
                    luts.last_mut().expect("inserted family LUT cache names")
                };
                if branch >= entry.width ||
                    !matches!(
                        declaration.role,
                        ProgramCacheRole::LutSwitch | ProgramCacheRole::LutBranch
                    )
                {
                    return Err(RefreshSetupError::InvalidManifest);
                }
            }
        }
    }
    Ok(ProgramCacheArtifactNames { gates, luts, families })
}

fn names_without_program_cache(
    names: &RefreshPreprocessingArtifactNames,
) -> RefreshPreprocessingArtifactNames {
    let mut names = names.clone();
    names.program_cache = ProgramCacheArtifactNames::default();
    names
}

fn validate_program_cache_names(
    names: &ProgramCacheArtifactNames,
    setup_identity: [u8; 32],
) -> Result<(), RefreshSetupError> {
    let prefix = format!("mxx-refresh-{}-program-cache", hex(&setup_identity));
    let mut gate_steps = BTreeMap::new();
    for entry in &names.gates {
        if gate_steps.insert(entry.step, ()).is_some() ||
            entry.rhs != format!("{prefix}-gate-{}-k-rhs", entry.step)
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
    }
    let mut family_keys = BTreeMap::new();
    let mut lut_steps = BTreeMap::new();
    for entry in &names.luts {
        if entry.width == 0 ||
            lut_steps.insert(entry.step, ()).is_some() ||
            entry.switch != format!("{prefix}-lut-{}-lut-switch-family", entry.step) ||
            entry.branch != format!("{prefix}-lut-{}-lut-branch-family", entry.step)
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
    }
    for entry in &names.families {
        if entry.count == 0 ||
            family_keys.insert((entry.step, entry.body_gate), ()).is_some() ||
            entry.rhs != format!("{prefix}-family-{}-body-{}-k-rhs", entry.step, entry.body_gate)
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
    }
    Ok(())
}
/// Constructs the declaration committed by a preprocessing producer.
fn make_declaration(
    p: &RefreshSetupParameters,
    contract: &RefreshPrfContract,
    n: RefreshPreprocessingArtifactNames,
    pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    decoder_preimage_bound: &mxx_ir_core::IntExpr,
    readout_identity: crate::noise_refresh::readout::ValueReadoutIdentity,
    readout_helper_sigmas: Vec<usize>,
) -> RefreshPreprocessingDeclaration {
    RefreshPreprocessingDeclaration {
        identity: setup_identity_digest(
            p,
            contract,
            pbc_layout_id,
            decoder_preimage_bound,
            readout_identity,
        ),
        pbc_layout_id,
        refresh_id: p.refresh_id,
        program_id: contract.program_id(),
        prf_q: contract.q(),
        prf_p: contract.p(),
        prf_native_width: contract.native_width(),
        prf_ring_dimension: contract.ring_dimension(),
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
        readout_identity,
        readout_helper_sigmas,
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
        declaration.prf_native_width,
        declaration.prf_q,
        declaration.prf_p,
        declaration.prf_ring_dimension,
    )
}

/// Exports producer wires under their canonical public/private artifact names.
fn add_setup_outputs(
    mut context: DslContext,
    wires: &RefreshPreprocessingWires,
    names: &RefreshPreprocessingArtifactNames,
) -> Result<DslContext, RefreshSetupError> {
    // The sampled K_t witnesses and setup secret t remain private artifacts.
    // The state vector, online secret s, decoder anchor b, and PRF mask/fresh
    // vectors are deliberately absent from this preprocessing graph.
    context = context
        .private_output(names.setup_secret_t.clone(), wires.setup_secret_t.clone())?
        .public_output(names.selector_gsw.clone(), wires.selector_gsw.clone())?
        .public_output(names.setup_public_key_rhs.clone(), wires.setup_public_key_rhs.clone())?
        .public_output(
            names.mask_route_rhs_decompositions.clone(),
            wires.mask_route_rhs_decompositions.clone(),
        )?
        .public_output(
            names.fresh_route_rhs_decompositions.clone(),
            wires.fresh_route_rhs_decompositions.clone(),
        )?
        .public_output(names.state_public_matrix.clone(), wires.state.matrix.clone())?
        .public_output(names.a_prime.clone(), wires.a_prime.clone())?
        .public_output(names.public_matrix_b.clone(), wires.public_b.clone())?;
    for (slot, target) in wires.refresh_targets.iter().enumerate() {
        context = context.public_output(names.refresh_targets[slot].clone(), target.clone())?;
    }
    for (slot, decomposition) in wires.state_scaling.iter().enumerate() {
        context =
            context.public_output(names.state_scaling[slot].clone(), decomposition.clone())?;
    }
    for slot in 0..wires.masks.len() {
        context =
            context.private_output(names.preimages[slot].clone(), wires.preimages[slot].clone())?;
    }
    let readout = &names.readout;
    context = context
        .public_output(
            readout.target_decomposition.clone(),
            wires.readout_artifacts.target_decompositions().clone(),
        )?
        .public_output(
            readout.normalization_decomposition.clone(),
            wires.readout_artifacts.normalization_decomposition().clone(),
        )?
        .public_output(
            readout.helper_switch_decompositions.clone(),
            wires.readout_artifacts.helper_switch_decompositions().clone(),
        )?
        .public_output(
            readout.helper_public_decompositions.clone(),
            wires.readout_artifacts.helper_public_decompositions().clone(),
        )?;
    if !names.program_cache.gates.is_empty() || !names.program_cache.luts.is_empty() {
        return Err(RefreshSetupError::InvalidManifest);
    }
    for entry in &names.program_cache.families {
        let (per_output_count, output_indexed, rhs) = wires
            .program_cache
            .family_entry(entry.step, entry.body_gate)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        if per_output_count != entry.count || output_indexed != entry.output_indexed {
            return Err(RefreshSetupError::InvalidManifest);
        }
        context = context.public_output(entry.rhs.clone(), rhs.clone())?;
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
    pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
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
        prf_q: usize,
        prf_p: usize,
        prf_native_width: usize,
        prf_ring_dimension: usize,
        prf_output_wire: crate::program::WireId,
        pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
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
        prf_q: contract.q(),
        prf_p: contract.p(),
        prf_native_width: contract.native_width(),
        prf_ring_dimension: contract.ring_dimension(),
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

/// Extends the refresh identity with the exact value-form readout.  The
/// legacy geometry identity remains available to noise planning, while the
/// artifact declaration must also bind the readout table/trace semantics.
fn setup_identity_digest(
    p: &RefreshSetupParameters,
    contract: &RefreshPrfContract,
    pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    decoder_preimage_bound: &mxx_ir_core::IntExpr,
    readout_identity: crate::noise_refresh::readout::ValueReadoutIdentity,
) -> [u8; 32] {
    #[derive(Serialize)]
    struct Payload {
        base: [u8; 32],
        readout_identity: crate::noise_refresh::readout::ValueReadoutIdentity,
    }
    hash_canonical(&Payload {
        base: identity_digest(p, contract, pbc_layout_id, decoder_preimage_bound),
        readout_identity,
    })
    .expect("canonical refresh setup/readout identity")
}
fn hex(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn expected_setup_artifact_count(
    slots: usize,
    _helper_count: usize,
    program_cache_artifact_count: usize,
) -> Option<usize> {
    10usize
        .checked_add(slots.checked_mul(3)?)
        .and_then(|count| count.checked_add(2))
        .and_then(|count| count.checked_add(program_cache_artifact_count))
}

fn exact_family_count(actual: Option<usize>, expected_count: usize) -> bool {
    actual == Some(expected_count)
}

/// Validates one setup helper's ordinary vector/public-matrix artifact pair.
///
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
            artifact.family_count.is_some() ||
            artifact.artifact_type != artifact_type ||
            (confidentiality == ArtifactConfidentiality::Private &&
                artifact.content_hash.is_some()) ||
            (confidentiality == ArtifactConfidentiality::Public &&
                artifact.content_hash.is_none())
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        Ok(())
    };
    let slots = p.refresh.crt_plaintext_moduli.len();
    let helper_count = p.lut_width.saturating_sub(1);
    let output_count = refresh_prf_output_count(p)?;
    let mask_route_count = refresh_route_family_count(p, p.mask_base_p_digit_count)?;
    let fresh_route_count = refresh_route_family_count(p, p.fresh_error_base_p_digit_count)?;
    let cols = p.layout.public_key_columns();
    let b_columns = p
        .component_count
        .checked_mul(p.layout.digit_count + 2)
        .ok_or(RefreshSetupError::InvalidManifest)?;
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
    // In addition to refresh/readout material, the producer exports the
    // setup-only `t` and the selector GSW family. Both are protocol setup
    // artifacts (the latter public); neither is a decomposition cache.
    let program_cache_artifact_count = n
        .program_cache
        .gates
        .len()
        .checked_add(n.program_cache.luts.iter().map(|_| 2).sum::<usize>())
        .and_then(|count| count.checked_add(n.program_cache.families.len()))
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let expected_artifact_count =
        expected_setup_artifact_count(slots, helper_count, program_cache_artifact_count)
            .ok_or(RefreshSetupError::InvalidManifest)?;
    if m.artifacts.len() != expected_artifact_count {
        return Err(RefreshSetupError::InvalidManifest);
    }
    if n.preimages.len() != slots {
        return Err(RefreshSetupError::InvalidManifest);
    }
    check(&n.setup_secret_t, ArtifactConfidentiality::Private, 1, p.layout.secret_dimension)?;
    let selector = m.artifacts.get(&n.selector_gsw).ok_or(RefreshSetupError::InvalidManifest)?;
    let active_count = match selector.family_count {
        Some(count) if count != 0 => count,
        _ => return Err(RefreshSetupError::InvalidManifest),
    };
    if selector.confidentiality != ArtifactConfidentiality::Public ||
        !matches!(selector.artifact_type, ArtifactType::Matrix(_)) ||
        selector.content_hash.is_none()
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let setup_public_key_count =
        output_count.checked_mul(active_count).ok_or(RefreshSetupError::InvalidManifest)?;
    check(&n.state_public_matrix, ArtifactConfidentiality::Public, p.component_count, cols)?;
    check(&n.a_prime, ArtifactConfidentiality::Public, p.component_count, cols)?;
    check(&n.public_matrix_b, ArtifactConfidentiality::Public, p.component_count, b_columns)?;
    if n.refresh_targets.len() != slots {
        return Err(RefreshSetupError::InvalidManifest);
    }
    for name in &n.refresh_targets {
        check(name, ArtifactConfidentiality::Public, p.component_count, cols)?;
    }
    for preimage in &n.preimages {
        check_preimage(preimage, b_columns, cols)?;
    }
    let gadget_base = p
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
    let readout_bound = gadget_base.clone() / BigInt::from(2u8);
    let check_readout_preimage = |name: &str| {
        check_type(
            name,
            ArtifactConfidentiality::Public,
            ArtifactType::Preimage {
                matrix: ConcreteMatrixType {
                    modulus: modulus.clone(),
                    ring_dimension,
                    rows: p
                        .component_count
                        .checked_mul(p.layout.digit_count)
                        .ok_or(RefreshSetupError::InvalidManifest)?,
                    columns: cols,
                },
                max_coefficient_bound: readout_bound.clone(),
            },
        )
    };
    for (name, expected_count) in [
        (&n.readout.target_decomposition, output_count),
        (&n.readout.helper_switch_decompositions, helper_count),
        (
            &n.readout.helper_public_decompositions,
            output_count.checked_mul(helper_count).ok_or(RefreshSetupError::InvalidManifest)?,
        ),
    ] {
        let artifact = m.artifacts.get(name).ok_or(RefreshSetupError::InvalidManifest)?;
        if artifact.confidentiality != ArtifactConfidentiality::Public ||
            artifact.family_count != Some(expected_count) ||
            artifact.artifact_type !=
                (ArtifactType::Preimage {
                    matrix: ConcreteMatrixType {
                        modulus: modulus.clone(),
                        ring_dimension,
                        rows: p
                            .component_count
                            .checked_mul(p.layout.digit_count)
                            .ok_or(RefreshSetupError::InvalidManifest)?,
                        columns: cols,
                    },
                    max_coefficient_bound: readout_bound.clone(),
                }) ||
            artifact.content_hash.is_none()
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
    }
    check_readout_preimage(&n.readout.normalization_decomposition)?;
    let cache_bound = gadget_base / BigInt::from(2u8);
    let cache_matrix = ConcreteMatrixType {
        modulus: modulus.clone(),
        ring_dimension,
        rows: p
            .component_count
            .checked_mul(p.layout.digit_count)
            .ok_or(RefreshSetupError::InvalidManifest)?,
        columns: cols,
    };
    let setup_public_key_rhs =
        m.artifacts.get(&n.setup_public_key_rhs).ok_or(RefreshSetupError::InvalidManifest)?;
    if setup_public_key_rhs.confidentiality != ArtifactConfidentiality::Public ||
        !exact_family_count(setup_public_key_rhs.family_count, setup_public_key_count) ||
        setup_public_key_rhs.artifact_type !=
            (ArtifactType::Preimage {
                matrix: cache_matrix.clone(),
                max_coefficient_bound: cache_bound.clone(),
            }) ||
        setup_public_key_rhs.content_hash.is_none()
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let check_cache = |name: &str, family_count: Option<usize>| {
        let artifact = m.artifacts.get(name).ok_or(RefreshSetupError::InvalidManifest)?;
        if artifact.confidentiality != ArtifactConfidentiality::Public ||
            artifact.artifact_type !=
                (ArtifactType::Preimage {
                    matrix: cache_matrix.clone(),
                    max_coefficient_bound: cache_bound.clone(),
                }) ||
            artifact.family_count != family_count ||
            artifact.content_hash.is_none()
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        Ok::<(), RefreshSetupError>(())
    };
    if !n.program_cache.gates.is_empty() || !n.program_cache.luts.is_empty() {
        return Err(RefreshSetupError::InvalidManifest);
    }
    for entry in &n.program_cache.families {
        if entry.count == 0 {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let count = if entry.output_indexed {
            output_count.checked_mul(entry.count).ok_or(RefreshSetupError::InvalidManifest)?
        } else {
            entry.count
        };
        check_cache(&entry.rhs, Some(count))?;
    }
    for name in &n.state_scaling {
        check_cache(name, None)?;
    }
    for (name, expected_count) in [
        (&n.mask_route_rhs_decompositions, mask_route_count),
        (&n.fresh_route_rhs_decompositions, fresh_route_count),
    ] {
        let artifact = m.artifacts.get(name.as_str()).ok_or(RefreshSetupError::InvalidManifest)?;
        if artifact.confidentiality != ArtifactConfidentiality::Public ||
            !exact_family_count(artifact.family_count, expected_count) ||
            artifact.artifact_type !=
                (ArtifactType::Preimage {
                    matrix: cache_matrix.clone(),
                    max_coefficient_bound: cache_bound.clone(),
                }) ||
            artifact.content_hash.is_none()
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
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
    public_key: &Mat,
    mask_secret: &Mat,
    payload_secret: &Mat,
    expected_plaintext: &Mat,
    gadget_base: impl Into<mxx_ir_core::IntExpr>,
    gadget_digit_count: usize,
    plaintext_modulus: impl Into<mxx_ir_core::IntExpr>,
    decode_length: usize,
) -> Result<RefreshVerification, RefreshSetupError> {
    let scalar = expected_plaintext.matrix_type();
    if scalar.rows != 1.into() || scalar.columns != 1.into() {
        return Err(RefreshSetupError::InvalidParameters("expected plaintext must be 1x1"));
    }
    let gadget = public_key.matrix_type();
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
    let public_key_columns = public_key
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
        public_key.matrix_type().modulus.clone(),
        public_key.matrix_type().ring_dimension.clone(),
    );
    let g = ring.gadget(mask_secret.matrix_type().columns.clone(), gadget_base, gadget_digit_count);
    let residual = encoding.vector.clone() - mask_secret.clone() * public_key.clone() +
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
/// BuiltGraph and declaration validation occurs while the
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
    /// Frozen residual wire selected from the validated graph.
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
    generated_layout: crate::noise_refresh::pbc::PbcGeneratedKeyLayout,
    key_instance_id: [u8; 32],
    expected_plaintext: Mat,
    decode_length: usize,
}

/// Samples the setup public-key family at hash indices `1..=active_count` and
/// derives its deterministic gadget decompositions without projecting family
/// members into Rust. Index zero remains reserved exactly as in
/// [`mxx_bgg::BggPublicKeySampler::sample`].
fn sample_setup_public_key_families(
    sampler: &mxx_bgg::BggPublicKeySampler,
    hash_key: Bytes,
    active_count: usize,
) -> Result<(Family<Mat>, Family<Preimage>), RefreshSetupError> {
    let sampled = sampler.sample_family(
        hash_key,
        b"refresh-parameter-search-setup-keys".as_slice(),
        active_count.checked_add(1).ok_or(RefreshSetupError::InvalidManifest)?,
        sampler.layout.public_key_columns(),
    )?;
    let indices =
        Parallel::range(active_count).map_values(|index| index.as_int().add(Int::constant(1)))?;
    let public_keys = sampled.matrices.parallel_gather(indices)?;
    let decompositions = public_keys.clone().parallel_map_values(|_, matrix| {
        matrix.decompose(
            sampler.layout.gadget_base.clone(),
            IntExpr::constant(sampler.layout.digit_count),
        )
    })?;
    Ok((public_keys, decompositions))
}

/// Samples an online public-exponent input directly from the cached RHS
/// decomposition.  Since `G*K_rhs = A`, the expression is
/// `(sG)*K_rhs - x*(sG) + e`; no public matrix is reconstructed online.
/// The label's public monomial encoding uses the same online secret `s` for
/// both mask and payload. The private accumulator separately uses setup `t`.
fn sample_online_label_vector_from_preimage(
    sampler: &ExponentLutEncodingSampler,
    online_secret_s: &Mat,
    decomposition: mxx_dsl::Preimage,
    plaintext: Mat,
) -> Result<Mat, RefreshSetupError> {
    let ring = sampler.layout.ring();
    let gadget = ring.gadget(
        sampler.layout.secret_dimension,
        sampler.layout.gadget_base.clone(),
        sampler.layout.digit_count,
    );
    let mask_secret_gadget = online_secret_s.clone() * gadget.clone();
    let payload_secret_gadget = online_secret_s.clone() * gadget;
    let public_term = decomposition.mul_small_rhs(mask_secret_gadget);
    let error = match (&sampler.gaussian_sigma, &sampler.gaussian_max_coefficient_bound) {
        (Some(sigma), Some(bound)) => {
            ring.gaussian((1, sampler.layout.public_key_columns()), sigma.clone(), bound.clone())
        }
        (None, None) => ring.zero((1, sampler.layout.public_key_columns())),
        _ => return Err(RefreshSetupError::InvalidManifest),
    };
    Ok(public_term - plaintext * payload_secret_gadget + error)
}

impl RefreshParameterSimulationRequest {
    /// Creates a request after checking all cross-layer dimensions and the
    /// selected decoder cutoff against the official policy. The sparse support and one-hot
    /// schedule remain private inside the generated PBC layout.
    pub fn new(
        setup: RefreshSetupParameters,
        profile: SparseLwrPrfProfile,
        generated_layout: crate::noise_refresh::pbc::PbcGeneratedKeyLayout,
        key_instance_id: [u8; 32],
        decoder_sigma: mxx_ir_core::RealExpr,
        expected_plaintext: Mat,
        decode_length: usize,
    ) -> Result<Self, RefreshSetupError> {
        setup.validate()?;
        if setup.decoder_sigma != decoder_sigma || decode_length == 0 {
            return Err(RefreshSetupError::InvalidParameters(
                "decoder sigma or decode length does not match setup",
            ));
        }
        generated_layout
            .public_layout
            .validate()
            .map_err(|_| RefreshSetupError::InvalidParameters("generated PBC layout is invalid"))?;
        if profile.ring_dimension()? !=
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

    /// Builds and validates the three graph stages and their artifact boundary.
    /// The resulting Exponent-LUT noise snapshot is the
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

    /// Builds only the setup-owned application-specific noise snapshot.
    ///
    /// Parameter search uses this path before a candidate is selected.  It
    /// derives the same compact decomposition plans and refresh descriptors
    /// as [`RefreshPreprocessingProducer::noise_snapshot_plans`], but does not
    /// lower the label-major PRF family or construct executable graph stages.
    /// The full graph remains available through [`Self::build`] and
    /// [`Self::build_benchmark`] for the selected candidate and benchmark.
    pub fn build_noise_snapshot_only(self) -> Result<ExponentLutNoiseSnapshot, RefreshSetupError> {
        let setup = self.setup;
        setup.validate()?;
        let pbc_layout = self.generated_layout.public_layout;
        pbc_layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let program = SparseLwrPrfProgram::new(
            self.profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )?;
        let active_cells = PbcActiveCellIndex::build(&pbc_layout)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid PBC active-cell index"))?;
        let readout_schema = compact_readout_plan_schema(&setup, &program)?;
        let (gain_plan, route_decompositions) = compact_noise_snapshot_plans(
            &setup,
            &program,
            &pbc_layout,
            &active_cells,
            &readout_schema,
        )?;
        setup.build_noise_snapshot(program, &pbc_layout, gain_plan, route_decompositions)
    }

    /// Builds the compact application-specific noise snapshot from a validated
    /// public PBC layout without constructing the private schedule or any
    /// executable graph.  Parameter search uses this entry point after it has
    /// cached the candidate's immutable layout context.
    pub fn build_noise_snapshot_from_public_layout(
        setup: RefreshSetupParameters,
        profile: SparseLwrPrfProfile,
        pbc_layout: &PbcPublicLayout,
        active_cells: &PbcActiveCellIndex,
    ) -> Result<ExponentLutNoiseSnapshot, RefreshSetupError> {
        setup.validate()?;
        pbc_layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let program = SparseLwrPrfProgram::new(
            profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )?;
        let readout_schema = compact_readout_plan_schema(&setup, &program)?;
        let (gain_plan, route_decompositions) = compact_noise_snapshot_plans(
            &setup,
            &program,
            pbc_layout,
            &active_cells,
            &readout_schema,
        )?;
        setup.build_noise_snapshot(program, pbc_layout, gain_plan, route_decompositions)
    }

    /// Returns the invariant gate/readout plan for a validated public layout.
    /// This is used by search to avoid regenerating decomposition plans while
    /// it scans mask digit counts.
    pub fn noise_gain_plan_from_public_layout(
        setup: &RefreshSetupParameters,
        profile: SparseLwrPrfProfile,
        pbc_layout: &PbcPublicLayout,
        active_cells: &PbcActiveCellIndex,
    ) -> Result<crate::noise_refresh::simulation::NoiseGainPlan, RefreshSetupError> {
        setup.validate()?;
        pbc_layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let program = SparseLwrPrfProgram::new(
            profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )?;
        let readout_schema = compact_readout_plan_schema(setup, &program)?;
        compact_noise_snapshot_plans(setup, &program, pbc_layout, active_cells, &readout_schema)
            .map(|(gain_plan, _)| gain_plan)
    }

    /// Returns the invariant decomposition routes together with their gain
    /// plan.  The caller may reuse these routes while scanning `d_m`; only the
    /// P-adic mask route changes with that parameter.
    pub fn noise_gain_and_route_invariants_from_public_layout(
        setup: &RefreshSetupParameters,
        profile: SparseLwrPrfProfile,
        pbc_layout: &PbcPublicLayout,
        active_cells: &PbcActiveCellIndex,
    ) -> Result<
        (crate::noise_refresh::simulation::NoiseGainPlan, RefreshNoiseGainDecompositions),
        RefreshSetupError,
    > {
        setup.validate()?;
        pbc_layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let program = SparseLwrPrfProgram::new(
            profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )?;
        let readout_schema = compact_readout_plan_schema(setup, &program)?;
        compact_noise_snapshot_plans(setup, &program, pbc_layout, active_cells, &readout_schema)
    }

    /// Derives the exact aggregate mask route gain for one `d_m` without
    /// constructing a program, snapshot, or executable graph.  This is the
    /// only route component that changes during a mask-digit scan.
    pub fn mask_route_gain_for_digit_count(
        setup: &RefreshSetupParameters,
        mask_digit_count: usize,
    ) -> Result<crate::noise::RightActionNoiseGain, RefreshSetupError> {
        let mut candidate = setup.clone();
        candidate.mask_base_p_digit_count = mask_digit_count;
        let (plans, multiplicity) = compact_mask_routes(&candidate)?;
        let gains =
            crate::noise_refresh::simulation::NoiseGainPlanBuilder::derive_route_gains(&plans)
                .map_err(|_| RefreshSetupError::InvalidParameters("invalid mask route plan"))?;
        let mut aggregate =
            crate::noise::RightActionNoiseGain { l1: 0u8.into(), l2_squared: 0u8.into() };
        for gain in gains {
            aggregate.l1 += gain.l1 * multiplicity;
            aggregate.l2_squared += gain.l2_squared * multiplicity;
        }
        Ok(aggregate)
    }

    /// Builds a compact snapshot with a previously derived invariant plan.
    /// Only the `d_m`-dependent route decompositions are rebuilt.
    pub fn build_noise_snapshot_from_public_layout_with_gain_plan(
        setup: RefreshSetupParameters,
        profile: SparseLwrPrfProfile,
        pbc_layout: &PbcPublicLayout,
        active_cells: &PbcActiveCellIndex,
        gain_plan: &crate::noise_refresh::simulation::NoiseGainPlan,
    ) -> Result<ExponentLutNoiseSnapshot, RefreshSetupError> {
        setup.validate()?;
        pbc_layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let program = SparseLwrPrfProgram::new(
            profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )?;
        let readout_schema = compact_readout_plan_schema(&setup, &program)?;
        let routes = compact_noise_snapshot_routes(&setup, &readout_schema)?;
        if active_cells.bucket_active_widths().count() != pbc_layout.cells.len() {
            return Err(RefreshSetupError::InvalidParameters(
                "PBC active-cell index shape mismatch",
            ));
        }
        setup.build_noise_snapshot(program, pbc_layout, gain_plan.clone(), routes)
    }

    /// Builds a compact snapshot using cached state/fresh/decoder routes.  A
    /// new mask route is constructed for the setup's current `d_m`, preserving
    /// the exact route schema while avoiding repeated invariant decomposition.
    pub fn build_noise_snapshot_from_public_layout_with_cached_routes(
        setup: RefreshSetupParameters,
        profile: SparseLwrPrfProfile,
        pbc_layout: &PbcPublicLayout,
        active_cells: &PbcActiveCellIndex,
        gain_plan: &crate::noise_refresh::simulation::NoiseGainPlan,
        invariant_routes: &RefreshNoiseGainDecompositions,
    ) -> Result<ExponentLutNoiseSnapshot, RefreshSetupError> {
        setup.validate()?;
        pbc_layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let program = SparseLwrPrfProgram::new(
            profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )?;
        if active_cells.bucket_active_widths().count() != pbc_layout.cells.len() {
            return Err(RefreshSetupError::InvalidParameters(
                "PBC active-cell index shape mismatch",
            ));
        }
        if invariant_routes.state_scaling.len() != setup.refresh.crt_plaintext_moduli.len() ||
            invariant_routes.fresh.len() != setup.refresh.crt_plaintext_moduli.len() ||
            invariant_routes.decoder.len() != setup.refresh.crt_plaintext_moduli.len()
        {
            return Err(RefreshSetupError::InvalidParameters(
                "cached refresh route slot count mismatch",
            ));
        }
        let (mask, mask_multiplicity) = compact_mask_routes(&setup)?;
        let routes = RefreshNoiseGainDecompositions {
            state_scaling: invariant_routes.state_scaling.clone(),
            mask,
            mask_multiplicity,
            fresh: invariant_routes.fresh.clone(),
            fresh_multiplicity: invariant_routes.fresh_multiplicity.clone(),
            decoder: invariant_routes.decoder.clone(),
        };
        let family_active_counts = active_cells.bucket_active_widths().collect::<Vec<_>>();
        setup.build_noise_snapshot_with_family_active_counts(
            program,
            pbc_layout,
            gain_plan.clone(),
            routes,
            family_active_counts,
        )
    }

    fn build_with_mode(
        self,
        benchmark_only: bool,
    ) -> Result<RefreshParameterSimulationBundle, RefreshSetupError> {
        let profile = self.profile;
        let setup = self.setup;
        setup.validate()?;
        let layout = self.generated_layout.public_layout.clone();
        layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let ring = setup.layout.ring();
        let hash_key = ring.bytes_input("refresh-parameter-search-hash-key", 32);
        let sampler = ExponentLutEncodingSampler {
            layout: setup.layout.clone(),
            exponent_layout: profile.layout,
            gaussian_sigma: Some(setup.encoding_error_sigma.clone()),
            gaussian_max_coefficient_bound: Some(setup.encoding_error_bound.clone()),
        };
        // Setup selector/GSW generation uses `t`, retained as a private setup
        // artifact. Online BGG encodings and the decoder anchor use one fresh
        // independent `s`, which is never exported.
        let setup_secret = sampler.sample_secret()?;
        let online_secret = sampler.sample_secret()?;
        let selector_bits = PbcTrustedSelectorBits::from_schedule(
            &self.generated_layout,
            &ring,
            self.key_instance_id,
        )
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let selector_runtime_binding = selector_bits.runtime_binding(&ring)?;
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
            setup_secret.clone(),
            setup_secret.clone(),
            hash_key.clone(),
            &layout,
            self.key_instance_id,
        )
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        // Selector GSW material is retained by the preprocessing producer.
        // Freezing it in a separate graph would freeze another copy of the
        // captured `t` sampler and cause setup to be sampled twice.
        let structural_gsw = structural.gsw;

        let active_count = crate::noise_refresh::pbc::PbcActiveCellIndex::build(&layout)
            .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?
            .len();
        let initial_attribute = profile.layout.mono(&ring, &BigInt::zero()).map_err(|_| {
            RefreshSetupError::InvalidParameters("invalid native initial attribute")
        })?;
        let online_label_count = refresh_prf_output_count(&setup)?;
        let total_labels = online_label_count
            .checked_mul(active_count)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let (setup_public_keys, setup_public_key_rhs) = sample_setup_public_key_families(
            &mxx_bgg::BggPublicKeySampler { layout: sampler.layout.clone() },
            hash_key.clone(),
            total_labels,
        )?;
        // The public-key program uses the structural sampled family directly.
        // Its matching K_rhs family is exported independently for online input
        // construction and is never expanded back into G*K here.
        let compiler = ExponentLutEncodingCompiler::with_layout(
            BggPublicKeyCompiler {
                ring: ring.clone(),
                base: setup.layout.gadget_base.clone(),
                digit_count: setup.layout.digit_count.into(),
            },
            profile.layout,
        );
        let refresh_compiler_encoding = ExponentLutEncodingCompiler::with_layout(
            BggPublicKeyCompiler {
                ring: ring.clone(),
                base: setup.layout.gadget_base.clone(),
                digit_count: setup.layout.digit_count.into(),
            },
            profile.layout,
        );
        let program =
            SparseLwrPrfProgram::new(profile, layout.bucket_width, layout.parameters.bucket_count)
                .map_err(|_| RefreshSetupError::InvalidParameters("invalid native PRF program"))?;
        let public_compiler = ExponentLutPublicKeyCompiler::with_layout(
            BggPublicKeyCompiler {
                ring: ring.clone(),
                base: setup.layout.gadget_base.clone(),
                digit_count: setup.layout.digit_count.into(),
            },
            profile.layout,
        );
        let public_selectors =
            PublicRhsFamily::new_for_layout(structural_gsw.clone(), layout.layout_id)?;
        let public_inputs = NativePrfPublicInputs::new_for_layout(
            setup_public_keys.clone(),
            online_label_count,
            layout.layout_id,
        )?;
        let readout_values = program
            .readout
            .canonical_coefficients()
            .ok_or(RefreshSetupError::InvalidParameters("native readout is not canonical"))?
            .iter()
            .map(|value| value.to_usize().ok_or(RefreshSetupError::InvalidManifest))
            .collect::<Result<Vec<_>, _>>()?;
        let readout_lut =
            LutTable::unary(profile.native_width(), profile.native_width(), readout_values)
                .map_err(|_| RefreshSetupError::InvalidParameters("invalid native readout LUT"))?;
        let readout_helpers = sampler.sample_native_readout_helper_families(
            setup_secret.clone(),
            online_secret.clone(),
            hash_key.clone(),
            &readout_lut,
            b"refresh-parameter-search-readout-mask-bank".as_slice(),
            b"refresh-parameter-search-readout-helpers".as_slice(),
        )?;
        let online_masks = readout_helpers.online_mask_vectors.clone();
        let initial_public_key = mxx_bgg::BggPublicKeySampler { layout: sampler.layout.clone() }
            .sample(hash_key.clone(), b"refresh-parameter-search-initial".as_slice(), &[false])
            .into_iter()
            .nth(1)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let public_prf = RefreshPublicPrfSetup {
            program: program.clone(),
            compiler: public_compiler,
            initial: initial_public_key.clone(),
            selectors: public_selectors,
            inputs: public_inputs,
            layout: layout.clone(),
            helpers: readout_helpers,
            normalization_gadget: ring.gadget(
                setup.component_count,
                setup.layout.gadget_base.clone(),
                setup.layout.digit_count,
            ),
        };
        let producer = RefreshPreprocessingProducer::build(RefreshPreprocessingRequest {
            parameters: setup.clone(),
            public_prf,
            compiler,
            state: initial_public_key,
            setup_secret_t: setup_secret.clone(),
            selector_gsw: structural_gsw,
            selector_gsw_name: selector_family_artifact_name(&selector_artifacts, "gsw"),
            setup_public_key_rhs,
            hash_key,
        })?;
        let producer_validated =
            producer.built.validate(&ParamEnv::default()).map_err(|error| {
                RefreshSetupError::SetupStage {
                    stage: "validate preprocessing graph",
                    detail: error.to_string(),
                }
            })?;
        let producer_spec_hash =
            spec_hash(&producer_validated.source, &producer_validated.bindings).map_err(
                |error| RefreshSetupError::SetupStage {
                    stage: "hash preprocessing graph",
                    detail: error.to_string(),
                },
            )?;
        let producer_production =
            ProductionId { spec_hash: producer_spec_hash, execution_nonce: [0x53; 32] };
        let mut producer_manifest = mxx_ir_core::artifact::export_validated_manifest(
            producer_production.clone(),
            &producer_validated,
        )
        .map_err(|error| RefreshSetupError::SetupStage {
            stage: "export preprocessing manifest",
            detail: error.to_string(),
        })?;
        for artifact in producer_manifest.artifacts.values_mut() {
            if artifact.confidentiality == ArtifactConfidentiality::Public {
                artifact.content_hash = Some([0xab; 32]);
            }
        }
        producer.finalize_export_manifest(&mut producer_manifest).map_err(|error| {
            RefreshSetupError::SetupStage {
                stage: "finalize preprocessing manifest",
                detail: error.to_string(),
            }
        })?;
        selector_artifacts.finalize_export_manifest(&mut producer_manifest).map_err(|error| {
            RefreshSetupError::SetupStage {
                stage: "finalize selector manifest",
                detail: error.to_string(),
            }
        })?;
        let imported = ImportedRefreshSetup::import(
            setup.clone(),
            producer.declaration().clone(),
            &producer_manifest,
        )
        .map_err(|error| RefreshSetupError::SetupStage {
            stage: "import preprocessing manifest",
            detail: error.to_string(),
        })?;
        let mut verification_manifests = BTreeMap::new();
        verification_manifests.insert(producer_production.clone(), producer_manifest.clone());
        let imported_selector_artifacts = PbcSelectorArtifacts::import(
            producer_production.clone(),
            &layout,
            self.key_instance_id,
            &producer_manifest,
            selector_artifacts.names().clone(),
            &sampler.layout,
            self.key_instance_id,
            &sampler.layout,
            self.key_instance_id,
        )
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let selector_inputs = PbcSelectorFamilyInputs::from_artifacts(
            &ring,
            &layout,
            &imported_selector_artifacts,
            &sampler.layout,
            self.key_instance_id,
            &sampler.layout,
            self.key_instance_id,
        )
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let selectors =
            selector_inputs.encoding_family().map_err(|_| RefreshSetupError::InvalidManifest)?;
        let initial_vector = sampler
            .sample_encodings_for_public_matrices(
                online_secret.clone(),
                Some(imported.setup_secret_t().clone()),
                &[imported.state_public_key().clone()],
                std::slice::from_ref(&initial_attribute),
            )?
            .into_iter()
            .next()
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let initial = BggEncodingWire {
            vector: initial_vector.vector,
            plaintext: Some(initial_attribute.clone()),
        };
        let setup_public_key_rhs = imported.setup_public_key_rhs.clone();
        // `a_j[i]` is an online-evaluation input, created only after the
        // preprocessing graph has been frozen and its artifacts imported.
        // Keeping this family symbolic avoids host expansion while ensuring
        // neither the exponents nor their monomials enter setup/cache state.
        let label_plaintext_family =
            ring.input_family("refresh-parameter-search-native-monomials", total_labels, (1, 1));
        let online_secret_for_labels = online_secret.clone();
        let label_plaintext_family_for_sampling = label_plaintext_family.clone();
        let label_vectors = (setup_public_key_rhs, label_plaintext_family_for_sampling)
            .parallel_zip_tuple_result(move |_, (rhs, plaintext)| {
                sample_online_label_vector_from_preimage(
                    &sampler,
                    &online_secret_for_labels,
                    rhs,
                    plaintext,
                )
                .map_err(|_| mxx_dsl::DslError::Schema)
            })
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let labels = NativePrfOnlineBatchInputs::new_for_layout(
            label_vectors,
            label_plaintext_family,
            online_label_count,
            layout.layout_id,
        )?;
        let refresh_compiler = setup.refresh.clone();
        // Online evaluation receives the actual state encoding and the
        // label-major `[label][active-cell]` vectors plus sampled public
        // monomials.  The imported preprocessing bundle supplies only public
        // readout keys/decompositions and decoder preimages.
        let online = RefreshOnlineEvaluationRequest {
            refresh: &refresh_compiler,
            compiler: &refresh_compiler_encoding,
            setup: &imported,
            program: &program,
            online_secret_s: online_secret.clone(),
            state: initial.clone(),
            selectors: &selectors,
            inputs: &labels,
            online_masks,
            layout: &layout,
        }
        .evaluate()?;
        let refreshed = online.refreshed;
        let benchmark_online_graph = DslContext::new("refresh-parameter-search-online-eval")
            .private_output("refreshed-encoding-vector", refreshed.encoding().vector.clone())?
            .build()?;
        validate_cache_required_online_graph(&benchmark_online_graph)?;
        let preprocessing_graph = BuiltGraph {
            graph: producer.built().graph.clone(),
            anchors: producer.built().anchors.clone(),
            derivation_attachments: producer.built().derivation_attachments.clone(),
        };
        let benchmark_preprocessing_graph = BuiltGraph {
            graph: preprocessing_graph.graph.clone(),
            anchors: preprocessing_graph.anchors.clone(),
            derivation_attachments: preprocessing_graph.derivation_attachments.clone(),
        };
        let public_identity = producer.declaration().identity;
        let (gain_plan, route_decompositions) = producer.noise_snapshot_plans()?;
        let noise_snapshot = setup.build_noise_snapshot(
            program.clone(),
            &layout,
            gain_plan,
            route_decompositions,
        )?;
        if noise_snapshot.setup_identity() != &public_identity {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        if benchmark_only {
            return Ok(RefreshParameterSimulationBundle {
                selector_graph: BuiltGraph {
                    graph: producer.built().graph.clone(),
                    anchors: producer.built().anchors.clone(),
                    derivation_attachments: producer.built().derivation_attachments.clone(),
                },
                preprocessing_graph: None,
                verification_graph: None,
                benchmark_preprocessing_graph,
                benchmark_online_graph,
                preprocessing_manifests: verification_manifests.clone(),
                verification_manifests,
                public_identity,
                selector_runtime_binding,
                metadata: BTreeMap::new(),
                targets: Vec::new(),
                noise_snapshot,
            });
        }
        let verification = build_refresh_verification(
            refreshed.encoding(),
            &imported.a_prime,
            &online_secret,
            imported.setup_secret_t(),
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
        Ok(RefreshParameterSimulationBundle {
            selector_graph: BuiltGraph {
                graph: producer.built().graph.clone(),
                anchors: producer.built().anchors.clone(),
                derivation_attachments: producer.built().derivation_attachments.clone(),
            },
            preprocessing_graph: Some(preprocessing_graph),
            verification_graph: Some(verification_graph),
            benchmark_preprocessing_graph,
            benchmark_online_graph,
            preprocessing_manifests: verification_manifests.clone(),
            verification_manifests,
            public_identity,
            selector_runtime_binding,
            metadata: BTreeMap::new(),
            targets,
            noise_snapshot,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise_refresh::pbc::{PbcParameters, PbcRootSeed, generate_key_layout};
    use mxx_ir_core::node::NodeKind;
    use num_bigint::BigInt;

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
    fn tiny_native_reference_oracle_covers_every_z_mod_q() {
        let profile =
            SparseLwrPrfProfile::new(crate::layout::ExpLayout { stride_log2: 1, width_log2: 4 }, 2)
                .expect("n=32, Q=32, p'=2 profile");
        assert_eq!(profile.ring_dimension().unwrap(), 32);
        assert_eq!(profile.q().unwrap(), 32);
        assert_eq!(profile.p_prime, 2);
        assert_eq!(profile.layout.stride_log2, 1);
        let coefficients = profile
            .canonical_readout()
            .expect("canonical readout")
            .canonical_coefficients()
            .expect("constant readout table")
            .to_vec();
        assert_eq!(coefficients.len(), 16);

        // Readout represents f(z)-1/2 in doubled centered coordinates.
        // Native negacyclic wrap sends z+n' to the additive inverse, so both
        // halves of Z_Q are covered without an out-of-range table lookup.
        let doubled = (0..32)
            .map(|z| {
                let k = z % 16;
                let value = &coefficients[k] * BigInt::from(2_u8) - BigInt::from(1_u8);
                if z < 16 { value } else { -value }
            })
            .collect::<Vec<_>>();
        for z in 0..16 {
            let expected = BigInt::from(2 * (2 * z / 16 + 1) - 1);
            assert_eq!(doubled[z], expected);
            assert_eq!(doubled[z + 16], -expected);
        }
    }

    #[test]
    fn setup_public_key_sampling_does_not_unroll_active_members() {
        let ring = mxx_dsl::Ring::new(257, 8);
        let layout = BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 8.into(),
            secret_dimension: 2,
            digit_count: 4,
            gadget_base: 4.into(),
        };
        let hash_key = ring.bytes_input("setup-family-hash-key", 32);
        let (public_keys, decompositions) = sample_setup_public_key_families(
            &mxx_bgg::BggPublicKeySampler { layout },
            hash_key,
            49_172,
        )
        .expect("structural setup public-key families");
        let graph = DslContext::new("structural-setup-public-key-families")
            .public_output("public-keys", public_keys)
            .expect("public-key family output")
            .public_output("decompositions", decompositions)
            .expect("decomposition family output")
            .build()
            .expect("setup family graph");
        graph.validate(&ParamEnv::default()).expect("valid structural setup family graph");

        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert!(
            nodes.len() < 32,
            "structural setup family graph must not grow with active_count: {} nodes",
            nodes.len()
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::FamilyGetStatic { .. }))
                .count(),
            0,
            "setup public keys and decompositions must not be host-unrolled"
        );
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
    fn setup_manifest_count_matches_exported_artifact_schema() {
        // Fixed outputs include protocol data and the two fixed readout
        // preimages. Each slot emits one scale K, one public target, and one
        // decoder preimage; all helpers share two family artifact names. The
        // online decoder anchor is intentionally not an artifact.
        assert_eq!(expected_setup_artifact_count(2, 3, 10), Some(28));
    }

    #[test]
    fn setup_family_count_contract_accepts_the_exact_output_and_route_shapes() {
        let setup = parameters(4, 2);
        let output_count = refresh_prf_output_count(&setup).expect("output count");
        assert_eq!(output_count, 48);
        assert_eq!(
            refresh_route_family_count(&setup, setup.mask_base_p_digit_count)
                .expect("mask route count"),
            32
        );
        assert_eq!(
            refresh_route_family_count(&setup, setup.fresh_error_base_p_digit_count)
                .expect("fresh route count"),
            32
        );
        let active_count = 5usize;
        let setup_public_key_count = output_count.checked_mul(active_count).unwrap();
        assert!(exact_family_count(Some(setup_public_key_count), setup_public_key_count));
    }

    #[test]
    fn setup_family_count_contract_rejects_nonexact_and_overflowing_shapes() {
        assert!(!exact_family_count(None, 7));
        assert!(!exact_family_count(Some(0), 7));
        assert!(!exact_family_count(Some(6), 7));
        let mut overflowing = parameters(4, 2);
        overflowing.coefficient_count = usize::MAX;
        assert!(
            refresh_route_family_count(&overflowing, overflowing.mask_base_p_digit_count).is_err()
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
            profile,
            generated.clone(),
            [0x78; 32],
            1.into(),
            expected_plaintext,
            1,
        )
        .expect("explicit cutoff above official is accepted");
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
            profile,
            generated.clone(),
            [0x7a; 32],
            1.into(),
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
        let (gain_plan, invariant_routes) =
            RefreshParameterSimulationRequest::noise_gain_and_route_invariants_from_public_layout(
                &setup,
                profile,
                &pbc_layout,
                &active_cells,
            )
            .expect("invariant compact plans");
        let cached = RefreshParameterSimulationRequest::build_noise_snapshot_from_public_layout_with_cached_routes(
            setup,
            profile,
            &pbc_layout,
            &active_cells,
            &gain_plan,
            &invariant_routes,
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
            setup_for_digits.mask_base_p_digit_count = mask_digits;
            let expected_plaintext =
                setup_for_digits.layout.ring().polynomial([BigInt::from(0_u8).into()]);
            let full_request = RefreshParameterSimulationRequest::new(
                setup_for_digits.clone(),
                profile,
                generated.clone(),
                [0x7a; 32],
                1.into(),
                expected_plaintext,
                1,
            )
            .expect("per-digit-count simulation request");
            let full = full_request.build_noise_snapshot_only().expect("ordinary compact snapshot");
            let cached =
                RefreshParameterSimulationRequest::build_noise_snapshot_from_public_layout_with_cached_routes(
                    setup_for_digits,
                    profile,
                    &pbc_layout,
                    &active_cells,
                    &gain_plan,
                    &invariant_routes,
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
            AverageVariance::new(4u8.into(), 1u8.into()).expect("positive variance")
        );
    }

    #[test]
    fn refresh_noise_parameters_bind_full_q_topology_and_digit_variances() {
        let setup = parameters(4, 2);
        let refresh = setup.refresh_noise_parameters().expect("complete refresh noise schema");
        assert_eq!(refresh.global_modulus, Some(BigUint::from(6u8)));
        assert_eq!(refresh.plaintext_moduli, vec![BigUint::from(2u8), BigUint::from(3u8)]);
        assert_eq!(
            refresh.transcript_topology,
            Some(RefreshTranscriptTopology {
                crt_slot_count: 2,
                component_columns: 4,
                coefficient_count: 4,
                mask_digit_count: 1,
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
        assert!(refresh.mask_digit_variances.is_empty());
        assert!(refresh.fresh_digit_variances.is_empty());
        assert!(refresh.route_gains.is_none());
    }

    #[test]
    fn refresh_route_gains_require_actual_per_slot_decompositions() {
        let plan =
            NoiseDecompositionPlan::new(BigUint::from(2u8), 1, vec![BigInt::from(2u8)], 1, 1, 1)
                .expect("concrete balanced decomposition");
        let decompositions = RefreshNoiseGainDecompositions {
            state_scaling: vec![plan.clone(), plan.clone()],
            mask: vec![plan.clone()],
            mask_multiplicity: 1,
            fresh: vec![vec![plan.clone()], vec![plan.clone()]],
            fresh_multiplicity: vec![1, 1],
            decoder: vec![plan.clone(), plan.clone()],
        };
        let gains = decompositions.derive_route_gains(2).expect("per-slot route gains");
        assert_eq!(gains.state_scaling[0].l1, BigUint::from(2u8));
        assert_eq!(gains.state_scaling[0].l2_squared, BigUint::from(4u8));
        assert_eq!(gains.fresh.len(), 2);
        assert!(
            RefreshNoiseGainDecompositions { fresh: vec![], ..decompositions }
                .derive_route_gains(2)
                .is_err()
        );
    }

    #[test]
    fn refresh_route_gains_sum_each_p_digit_and_ring_coordinate() {
        // P=2 is intentionally distinct from gadget beta=4.  Each of three
        // P-adic digits is emitted for each of three ring coordinates, while
        // state scaling remains one exact action per unequal CRT slot.
        let crt = vec![BigUint::from(101u16), BigUint::from(103u16), BigUint::from(107u16)];
        let route = |value: BigInt, ring_coefficient: usize| {
            NoiseDecompositionPlan::sparse_monomial(
                BigUint::from(4u8),
                12,
                2,
                4,
                8,
                1,
                0,
                ring_coefficient,
                value,
            )
            .unwrap()
            .bind_crt_layout(crt.clone(), 4)
            .unwrap()
        };
        let mask: Vec<_> = (0u32..3)
            .flat_map(|digit| (0..3).map(move |j| route(BigUint::from(2u8).pow(digit).into(), j)))
            .collect();
        let fresh: Vec<Vec<_>> = [BigUint::from(7u8), BigUint::from(3u8)]
            .into_iter()
            .map(|kappa| {
                (0u32..3)
                    .flat_map(|digit| {
                        let value = &kappa * BigUint::from(2u8).pow(digit);
                        (0..3).map(move |j| route(value.clone().into(), j))
                    })
                    .collect()
            })
            .collect();
        let state_scaling = [BigUint::from(7u8), BigUint::from(3u8)]
            .into_iter()
            .map(|kappa| {
                NoiseDecompositionPlan::gadget_scalar(
                    BigUint::from(4u8),
                    12,
                    2,
                    24,
                    8,
                    kappa.into(),
                )
                .unwrap()
                .bind_crt_layout(crt.clone(), 4)
                .unwrap()
            })
            .collect();
        let decoder = vec![route(1.into(), 0), route(1.into(), 0)];
        let decompositions = RefreshNoiseGainDecompositions {
            state_scaling,
            mask,
            mask_multiplicity: 1,
            fresh_multiplicity: vec![1, 1],
            fresh,
            decoder,
        };
        let gains = decompositions.derive_route_gains(2).unwrap();
        let sum = |plans: &[NoiseDecompositionPlan]| {
            plans.iter().fold(
                RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::zero() },
                |mut total, plan| {
                    let gain = plan.action_gain();
                    total.l1 += gain.l1;
                    total.l2_squared += gain.l2_squared;
                    total
                },
            )
        };
        assert_eq!(gains.mask, sum(&decompositions.mask));
        assert_eq!(gains.fresh[0], sum(&decompositions.fresh[0]));
        assert_eq!(gains.fresh[1], sum(&decompositions.fresh[1]));
        assert_ne!(gains.fresh[0], gains.fresh[1]);
        assert_ne!(gains.state_scaling[0], gains.state_scaling[1]);

        // Compact route plans represent all ring-coordinate routes with a
        // multiplicity; they must produce exactly the same gain as the
        // explicit per-coordinate plans.  The PRF component axis is already
        // the target-column axis of `sparse_monomial` and is therefore not a
        // second multiplicity factor.
        // A single action per P-adic digit is the compact representation used
        // by the graph-free path.  Repeat it for each ring coordinate and
        // compare against the corresponding explicit route sum.
        let compact = RefreshNoiseGainDecompositions {
            state_scaling: decompositions.state_scaling.clone(),
            mask: vec![route(1.into(), 0), route(2.into(), 0), route(4.into(), 0)],
            mask_multiplicity: 3,
            fresh: vec![
                vec![route(7.into(), 0), route(14.into(), 0), route(28.into(), 0)],
                vec![route(3.into(), 0), route(6.into(), 0), route(12.into(), 0)],
            ],
            fresh_multiplicity: vec![3, 3],
            decoder: decompositions.decoder.clone(),
        };
        let explicit = decompositions.derive_route_gains(2).unwrap();
        let compact_gains = compact.derive_route_gains(2).unwrap();
        assert_eq!(explicit.mask, compact_gains.mask);
        assert_eq!(explicit.fresh, compact_gains.fresh);
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
    fn decoder_preimage_plans_preserve_slot_shape_and_bound() {
        let ring = parameters(4, 2).layout.ring();
        let preimages = vec![
            ring.preimage_input("decoder-slot-0", (8, 4), 2),
            ring.preimage_input("decoder-slot-1", (7, 5), 3),
        ];
        let bounds = [BigUint::from(2u8), BigUint::from(3u8)];
        let plans = preimages
            .iter()
            .enumerate()
            .map(|(slot, preimage)| {
                direct_bounded_preimage_plan(preimage, &BigUint::from(4u8), bounds[slot].clone())
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("each decoder slot uses its own preimage");

        assert_eq!((plans[0].target_rows, plans[0].target_columns), (8, 4));
        assert_eq!((plans[1].target_rows, plans[1].target_columns), (7, 5));
        assert_eq!(plans[0].action_gain().l1, BigUint::from(64u8));
        assert_eq!(plans[1].action_gain().l1, BigUint::from(84u8));
        assert!(
            direct_bounded_preimage_plan(&preimages[1], &BigUint::from(4u8), bounds[0].clone(),)
                .is_err()
        );
    }

    #[test]
    fn setup_decomposition_changes_route_gain_and_digit_variance() {
        use crate::{
            noise::{
                GateNoiseDecomposition, GateNoiseGains, GateNoiseOperation, RightActionNoiseGain,
            },
            noise_refresh::simulation::{TraceNoiseGains, ValueReadoutNoiseGains},
            program::{Gate, ProgramStep},
        };

        let setup = parameters(4, 2);
        let pbc_parameters = PbcParameters::custom(4, 1, 2, 2, 1, None);
        let pbc_layout = crate::noise_refresh::pbc::PbcPublicLayout::build(
            &pbc_parameters,
            crate::noise_refresh::pbc::derive_attempt_seed(PbcRootSeed([0x19; 32]), 0),
            0,
        )
        .expect("PBC layout");
        let profile =
            SparseLwrPrfProfile::new(crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 }, 1)
                .expect("native profile");
        let program = SparseLwrPrfProgram::new(
            profile,
            pbc_layout.bucket_width,
            pbc_layout.parameters.bucket_count,
        )
        .expect("native program");
        let unit_plan =
            NoiseDecompositionPlan::new(BigUint::from(2u8), 1, vec![BigInt::from(1u8)], 1, 1, 1)
                .expect("unit decomposition");
        let changed_plan =
            NoiseDecompositionPlan::new(BigUint::from(2u8), 1, vec![BigInt::from(2u8)], 1, 1, 1)
                .expect("changed decomposition");
        let make_gain_plan = |plan: &NoiseDecompositionPlan| {
            let mut decompositions = Vec::new();
            let mut gates = Vec::new();
            for (step_index, step) in program.accumulation.steps().iter().enumerate() {
                let count = match step {
                    ProgramStep::Family { body, .. } => {
                        assert!(matches!(
                            body.gates(),
                            [Gate::MulPrivate { .. }, Gate::MulPublic { .. }]
                        ));
                        // The compact noise plan keeps one action per
                        // structural family-body operation; active members
                        // are applied by `simulate_program`.
                        body.gates().len() + 1
                    }
                    ProgramStep::Gate(Gate::MulPrivate { .. }) |
                    ProgramStep::Gate(Gate::MulPublic { .. }) |
                    ProgramStep::Gate(Gate::Add { .. }) |
                    ProgramStep::Gate(Gate::Lut { .. }) => 1,
                };
                for branch_index in 0..count {
                    let operation = match step {
                        ProgramStep::Family { body, .. } => {
                            body.gates().get(branch_index).map_or(GateNoiseOperation::Add, |gate| {
                                match gate {
                                    Gate::MulPrivate { .. } => GateNoiseOperation::MulPrivate,
                                    Gate::MulPublic { .. } => GateNoiseOperation::MulPublic,
                                    Gate::Add { .. } => GateNoiseOperation::Add,
                                    Gate::Lut { .. } => GateNoiseOperation::Lut,
                                }
                            })
                        }
                        ProgramStep::Gate(Gate::MulPrivate { .. }) => {
                            GateNoiseOperation::MulPrivate
                        }
                        ProgramStep::Gate(Gate::MulPublic { .. }) => GateNoiseOperation::MulPublic,
                        ProgramStep::Gate(Gate::Add { .. }) => GateNoiseOperation::Add,
                        ProgramStep::Gate(Gate::Lut { .. }) => GateNoiseOperation::Lut,
                    };
                    decompositions
                        .push(GateNoiseDecomposition { action: plan.clone(), helper_action: None });
                    gates.push(GateNoiseGains {
                        step_index,
                        operation_index: gates.len(),
                        branch_index,
                        operation,
                        gate: "setup-test".to_owned(),
                        action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
                        helper_action: None,
                        inherited_bound: BigUint::zero(),
                        fresh_bound: BigUint::zero(),
                        output_bound: BigUint::zero(),
                        inherited_variance: AverageVariance::zero(),
                        fresh_variance: AverageVariance::zero(),
                        output_variance: AverageVariance::zero(),
                    });
                }
            }
            let trace_count = program
                .readout
                .trace_plan(profile.ring_dimension().unwrap())
                .unwrap()
                .helper_count();
            let trace = (0..trace_count)
                .map(|_| TraceNoiseGains {
                    action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
                    helper_action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
                })
                .collect();
            let readout_gains = ValueReadoutNoiseGains {
                target: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
                identity_trace: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
                nonidentity_trace: trace,
                normalization: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
                inherited_bound: BigUint::zero(),
                output_bound: BigUint::zero(),
                output_variance: AverageVariance::zero(),
            };
            NoiseGainPlanBuilder::new(
                decompositions,
                plan.clone(),
                (0..trace_count).map(|_| (plan.clone(), plan.clone())).collect(),
                plan.clone(),
            )
            .unwrap()
            .build(gates, readout_gains)
            .unwrap()
        };
        let make_routes = |plan: &NoiseDecompositionPlan| RefreshNoiseGainDecompositions {
            state_scaling: vec![plan.clone(), plan.clone()],
            mask: vec![plan.clone()],
            mask_multiplicity: setup.coefficient_count,
            fresh: vec![vec![plan.clone()], vec![plan.clone()]],
            fresh_multiplicity: vec![setup.coefficient_count; 2],
            decoder: vec![plan.clone(), plan.clone()],
        };
        let first = setup
            .build_noise_snapshot(
                program.clone(),
                &pbc_layout,
                make_gain_plan(&unit_plan),
                make_routes(&unit_plan),
            )
            .expect("first snapshot");
        let second = setup
            .build_noise_snapshot(
                program.clone(),
                &pbc_layout,
                make_gain_plan(&changed_plan),
                make_routes(&changed_plan),
            )
            .expect("second snapshot");
        let first_routes = first.refresh.route_gains.as_ref().unwrap();
        let second_routes = second.refresh.route_gains.as_ref().unwrap();
        assert_ne!(first_routes.mask.l2_squared, second_routes.mask.l2_squared);
        assert_ne!(first.refresh.mask_digit_variances[0], second.refresh.mask_digit_variances[0]);
        assert_ne!(first.refresh.fresh_digit_variances[0], second.refresh.fresh_digit_variances[0]);
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
        let decoder_bound = 1.into();
        let identity = |setup: &RefreshSetupParameters| {
            identity_digest(
                setup,
                &contract,
                crate::noise_refresh::pbc::PbcLayoutId([9; 32]),
                &decoder_bound,
            )
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
    fn online_decoder_base_samples_one_error_without_exporting_the_anchor() {
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

        let consumer = ring.input("decoder-anchor-consumer", (8, 1));
        let context = DslContext::new("online-decoder-base-test")
            .private_output("decoder-result", base.vector * consumer)
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

    #[test]
    fn online_label_encodings_use_s_for_both_mask_and_payload_when_s_differs_from_t()
    -> Result<(), RefreshSetupError> {
        let setup = parameters(4, 2);
        let sampler = ExponentLutEncodingSampler {
            layout: setup.layout.clone(),
            exponent_layout: crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 },
            gaussian_sigma: Some(setup.encoding_error_sigma.clone()),
            gaussian_max_coefficient_bound: Some(setup.encoding_error_bound.clone()),
        };
        let ring = setup.layout.ring();
        let s = sampler.sample_secret().expect("online secret s");
        let t = sampler.sample_secret().expect("setup secret t");
        let key = mxx_bgg::BggPublicKeySampler { layout: sampler.layout.clone() }
            .sample(
                ring.bytes_input("online-s-test-key", 32),
                b"online-s-test".as_slice(),
                &[false],
            )
            .into_iter()
            .nth(1)
            .expect("public key");
        let plaintext = ring.identity(1);
        let decomposition = key
            .matrix
            .clone()
            .decompose(setup.layout.gadget_base.clone(), setup.layout.digit_count);
        let initial = sampler
            .sample_encodings_for_public_matrices(
                s.clone(),
                Some(t.clone()),
                std::slice::from_ref(&key),
                std::slice::from_ref(&plaintext),
            )?
            .into_iter()
            .next()
            .ok_or(RefreshSetupError::InvalidManifest)?
            .vector;
        let public_a =
            sample_online_label_vector_from_preimage(&sampler, &s, decomposition, plaintext)?;
        let context = DslContext::new("online-s-shared-test")
            .private_output("initial", initial)
            .expect("initial output")
            .private_output("public-a", public_a)
            .expect("public-a output")
            .build()?;
        let uniform_nodes = context
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::UniformIntervalSample { .. }))
            .count();
        assert_eq!(
            uniform_nodes, 2,
            "online labels must reuse s for both mask and payload; setup t is distinct and unused"
        );
        assert_ne!(s.value_handle(), t.value_handle());
        Ok(())
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
/// BuiltGraph and declaration validation occurs during bundle
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
    selector_runtime_binding: PbcSelectorRuntimeBinding,
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
    /// Materializes the private zero/one selector-family input for one
    /// preprocessing execution. The symbolic graph itself never contains one
    /// matrix node per selector bit.
    pub fn selector_runtime_inputs<B: mxx_runtime::Backend>(
        &self,
        backend: &mut B,
    ) -> Result<BTreeMap<String, mxx_runtime::RuntimeValue<B>>, RefreshSetupError> {
        self.selector_runtime_binding.materialize(backend).map_err(|error| {
            RefreshSetupError::SetupStage {
                stage: "materialize selector runtime input",
                detail: error.to_string(),
            }
        })
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
    ) -> Result<ExponentLutNoiseReport, crate::noise_refresh::simulation::NoiseSimulationError>
    {
        if self.noise_snapshot.setup_identity() != &self.public_identity {
            return Err(crate::noise_refresh::simulation::NoiseSimulationError::InvalidRefresh(
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
    ) -> Result<ExponentLutAverageNoiseReport, crate::noise_refresh::simulation::NoiseSimulationError>
    {
        if self.noise_snapshot.setup_identity() != &self.public_identity {
            return Err(crate::noise_refresh::simulation::NoiseSimulationError::InvalidRefresh(
                "average simulation snapshot identity mismatch",
            ));
        }
        self.noise_snapshot.simulate_average(config)
    }
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
