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
//! The decoder base `b_t` yields `d_t=b_t K_t`; imported declarations retain
//! these exact graph relations while keeping secrets and plaintext out of the
//! public identity.

use crate::{
    PowerLutEncodingCompiler, PowerLutError,
    encoding::{EncodingSelectorFamily, FlatLutHelperSet, PowerLutEncodingSampler},
    pbc::{
        PbcSelectorArtifactNames, PbcSelectorArtifacts, PbcTrustedSelectorBits,
        build_structural_selector_families,
    },
    prf::{
        RefreshPrfBatchInputs, SparseLwrPrfProfile, SparseLwrPrfProgram, SparseLwrPrfTerminalForm,
    },
    program::PowerLutProgramId,
    refresh::{
        RefreshCompiler, RefreshError, RefreshFreshErrorMaterial, RefreshMaskMaterial,
        RefreshPrfContract, RefreshPrfCoverage, RefreshSetupManifest,
        aggregate_refresh_fresh_error, aggregate_refresh_mask,
    },
};
use bigdecimal::BigDecimal;
use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout};
use mxx_dsl::{
    Bool, BuiltGraph, Bytes, DerivationAttachmentValue, DslContext, Family, HashTag, Mat,
    SemanticAnchor,
};
use mxx_ir_core::{
    ParamEnv, ScopedWireRef,
    artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId},
    encoding::{IR_VERSION, hash_canonical, spec_hash},
    graph::{FrozenGraphScopeId, Graph},
    types::{ConcreteMatrixType, NodeId, WireType},
};
use num_bigint::ToBigInt;
use num_traits::{FromPrimitive, ToPrimitive};
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
    /// A Power-LUT boundary rejected the setup inputs.
    Power(#[from] PowerLutError),
    #[error(transparent)]
    /// Sparse-LWR encoding setup sampling failed.
    Sampling(#[from] crate::encoding::PowerLutSamplingError),
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
/// `lut_width` is the Power-LUT coefficient-sieve width; it is independent of
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
    /// Number of gadget digits in each sampled BGG object.
    pub digit_count: usize,
    /// Explicit Power-LUT sieve width `W`; this is not the PBC bucket width.
    pub lut_width: usize,
    /// BGG sampler dimensions shared by setup artifacts.
    pub layout: BggSamplerLayout,
    /// CRT modulus and reconstruction data for the refresh equations.
    pub refresh: RefreshCompiler,
    /// Gaussian width used when sampling the decoder trapdoor.
    pub decoder_sigma: mxx_ir_core::RealExpr,
    /// Policy for the decoder preimage rejection cutoff.  The policy is
    /// resolved to one concrete integer before the producer graph is built.
    pub decoder_preimage_bound: mxx_bgg::PreimageCoefficientBound,
    /// Human-readable setup name used by callers when selecting a production.
    pub name: String,
}
impl std::fmt::Debug for RefreshSetupParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshSetupParameters")
            .field("refresh_id", &self.refresh_id)
            .field("base_p", &self.base_p)
            .field("component_count", &self.component_count)
            .field("coefficient_count", &self.coefficient_count)
            .field("digit_count", &self.digit_count)
            .field("lut_width", &self.lut_width)
            .field("layout", &self.layout)
            .field("decoder_sigma", &self.decoder_sigma)
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
        digit_count: usize,
        lut_width: usize,
        layout: BggSamplerLayout,
        refresh: RefreshCompiler,
        decoder_sigma: mxx_ir_core::RealExpr,
        name: impl Into<String>,
    ) -> Self {
        Self {
            refresh_id,
            base_p,
            component_count,
            coefficient_count,
            digit_count,
            lut_width,
            layout,
            refresh,
            decoder_sigma,
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
    /// accidentally reused as the Power-LUT domain width.
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
            self.coefficient_count > n ||
            self.lut_width == 0 ||
            !self.lut_width.is_power_of_two() ||
            self.lut_width > n ||
            n % self.lut_width != 0 ||
            self.digit_count != self.layout.digit_count ||
            self.digit_count < 2 ||
            self.refresh.full_modulus.canonicalize() != self.layout.modulus.canonicalize()
        {
            return Err(RefreshSetupError::InvalidParameters("inconsistent dimensions"));
        }
        self.refresh.validate_layout()?;
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
}

impl RefreshPrfInputs {
    /// Creates the refresh PRF aggregate from outputs of the real PBC
    /// lowering.  Slot order and complete component/coefficient/digit
    /// coverage are checked before any material is retained.
    pub fn from_pbc_outputs(
        parameters: &RefreshSetupParameters,
        expected_program: &SparseLwrPrfProgram,
        masks_by_slot: Vec<Vec<crate::refresh::RefreshMaskPrfOutput>>,
        fresh_error: Vec<crate::refresh::RefreshFreshErrorPrfOutput>,
    ) -> Result<Self, RefreshSetupError> {
        parameters.validate()?;
        let contract = RefreshPrfContract::from_program(expected_program);
        contract.validate_for(parameters)?;
        let coverage = RefreshPrfCoverage::new(
            parameters.refresh_id,
            parameters.component_count,
            parameters.coefficient_count,
            parameters.digit_count,
        )?;
        let slot_count = parameters.refresh.crt_plaintext_moduli.len();
        if masks_by_slot.len() != slot_count {
            return Err(RefreshSetupError::Refresh(RefreshError::SlotOrderMismatch));
        }
        let mut layout_id = None;
        let mut masks = Vec::with_capacity(slot_count);
        for (slot, outputs) in masks_by_slot.into_iter().enumerate() {
            let material = RefreshMaskMaterial::new(coverage.clone(), slot, contract, outputs)?;
            if let Some(expected) = layout_id {
                if material.layout_id() != expected {
                    return Err(RefreshSetupError::Refresh(RefreshError::PrfOutputMismatch));
                }
            } else {
                layout_id = Some(material.layout_id());
            }
            masks.push(material);
        }
        let layout_id = layout_id.ok_or(RefreshSetupError::Refresh(RefreshError::InvalidLayout))?;
        let fresh_error = RefreshFreshErrorMaterial::new(coverage, contract, fresh_error)?;
        if fresh_error.layout_id() != layout_id {
            return Err(RefreshSetupError::Refresh(RefreshError::PrfOutputMismatch));
        }
        let inputs = Self { masks, fresh_error, contract, layout_id };
        inputs.validate_for(parameters)?;
        Ok(inputs)
    }

    /// Re-runs all producer-side identity and coverage checks at consumption.
    pub(crate) fn validate_for(
        &self,
        parameters: &RefreshSetupParameters,
    ) -> Result<(), RefreshSetupError> {
        parameters.validate()?;
        self.contract.validate_for(parameters)?;
        let coverage = RefreshPrfCoverage::new(
            parameters.refresh_id,
            parameters.component_count,
            parameters.coefficient_count,
            parameters.digit_count,
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
    /// Private Power-LUT compiler used to aggregate PRF digits.
    pub compiler: PowerLutEncodingCompiler,
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
    /// Private fresh-error vector artifact.
    pub fresh_vector: String,
    /// Public matrix paired with [`Self::fresh_vector`].
    pub fresh_public_matrix: String,
    /// Per-CRT-slot private mask vectors and public matrices.
    pub masks: Vec<RefreshArtifactPairNames>,
    /// Per-slot decoder base vectors paired with `B`.
    pub decoder_base_vectors: Vec<String>,
    /// Per-slot sampled preimage matrices.
    pub preimages: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct RefreshPreprocessingWires {
    pub(crate) state: BggEncodingWire,
    pub(crate) secret: Mat,
    pub(crate) a_prime: Mat,
    pub(crate) public_b: Mat,
    pub(crate) fresh: BggEncodingWire,
    pub(crate) masks: Vec<BggEncodingWire>,
    pub(crate) decoder_bases: Vec<BggEncodingWire>,
    pub(crate) preimages: Vec<Mat>,
    pub(crate) targets: Vec<Mat>,
    pub(crate) trapdoor: mxx_ir_core::ValueHandle,
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
    /// Sparse-LWR/Power-LUT program identity.
    pub program_id: PowerLutProgramId,
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
    /// Number of coefficients per slot.
    pub coefficient_count: usize,
    /// Number of gadget digits per coefficient.
    pub digit_count: usize,
    /// Decoder trapdoor Gaussian width.
    pub decoder_sigma: mxx_ir_core::RealExpr,
    /// Concrete maximum coefficient accepted by preimage sampling.
    pub decoder_preimage_bound: mxx_ir_core::IntExpr,
    /// Number of digits in the decoder trapdoor.
    pub trapdoor_digit_count: usize,
    /// Per-slot CRT scaling polynomials.
    pub slot_scales: Vec<mxx_ir_core::IntExpr>,
    /// Full modulus used by the BGG layout.
    pub layout_modulus: mxx_ir_core::IntExpr,
    /// Ring dimension used by the BGG layout.
    pub layout_ring_dimension: mxx_ir_core::IntExpr,
    /// Gadget base used by the BGG layout.
    pub layout_gadget_base: mxx_ir_core::IntExpr,
}

/// Private relation record tying one K-as-mat wrapper to its exact preimage
/// sample and decoder-base vector in the frozen producer graph.
#[derive(Clone)]
struct RefreshPreimageRelationAttestation {
    slot: usize,
    target: ScopedWireRef,
    preimage: ScopedWireRef,
    k_as_mat: ScopedWireRef,
    decoder_base_vector: ScopedWireRef,
}

/// Frozen setup graph attestation.  The relation records and attachment are
/// private so callers cannot substitute a same-shaped decoder or preimage.
#[derive(Clone)]
/// Frozen producer graph plus private relation attestations.
///
/// This is the attested setup boundary: the graph hash and attachment identify
/// the producer, while private relation records tie each sampled preimage to
/// its exact target and decoder-base output.
pub struct RefreshProducerAttestation {
    graph: Graph,
    producer_spec_hash: mxx_ir_core::artifact::SpecHash,
    attachment: mxx_dsl::FrozenDerivationAttachment,
    relations: Vec<RefreshPreimageRelationAttestation>,
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
    pub(crate) fresh: BggEncodingWire,
    pub(crate) masks: Vec<BggEncodingWire>,
    pub(crate) decoder_bases: Vec<BggEncodingWire>,
    pub(crate) preimages: Vec<Mat>,
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
        let fresh =
            aggregate_refresh_fresh_error(&request.compiler, p.base_p, &request.prf.fresh_error)?;
        let masks = request
            .prf
            .masks
            .iter()
            .map(|m| aggregate_refresh_mask(&request.compiler, p.base_p, m))
            .collect::<Result<Vec<_>, _>>()?;
        // `A'` is the public random matrix used in every slot target
        // `T_t = A_{sum,t} - mu_t A'`; its domain tag binds it to this refresh.
        let a_prime = ring.hash_matrix(
            request.hash_key.clone(),
            HashTag::from(
                format!("mxx-power-lut/refresh/a-prime/v1/{}", hex(&p.refresh_id)).into_bytes(),
            ),
            (p.layout.secret_dimension, p.layout.public_key_columns()),
        );
        let trapdoor_digits = p.digit_count;
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
            .checked_mul(p.digit_count + 2)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let trapdoor_handle = trapdoor.value_handle().clone();
        let mut bases = Vec::new();
        let mut scales = Vec::new();
        for slot in 0..masks.len() {
            // `scale = mu_t = q/q_t` is the scalar represented by the
            // setup-fixed RNS scaling LUT for CRT slot t.
            let scale = ring.polynomial([p.refresh.scale_expression(slot)?]);
            let base = BggEncodingWire {
                vector: request.secret.clone() * public_b.clone(),
                pubkey: BggPublicKeyWire { matrix: public_b.clone(), reveal_plaintext: false },
                plaintext: None,
            };
            bases.push(base);
            scales.push(scale);
        }
        let mask_public_family =
            Family::pack(masks.iter().map(|mask| mask.pubkey.matrix.clone()).collect())?;
        let scale_family = Family::pack(scales)?;
        // Build every slot target in one structural loop, then sample all
        // preimages in a second structural loop. Each family element computes
        // `T_t = A_t + A_{m,t} + A_{e,t} - mu_t A'`; the slot index is structural,
        // never plaintext. The target family is Zip and the trapdoor is
        // captured once as a Broadcast input.
        let target_family = Family::<Mat>::parallel_zip_many_values(
            vec![mask_public_family, scale_family],
            |_, mut items| {
                let mask_public = items.remove(0);
                let scale = items.remove(0);
                let scaled_state = request.compiler.bgg.large_scalar_mul(&request.state, &scale);
                let scaled_fresh = request.compiler.bgg.large_scalar_mul(&fresh, &scale);
                scaled_state.pubkey.matrix + mask_public + scaled_fresh.pubkey.matrix -
                    scale * a_prime.clone()
            },
        )?;
        let targets =
            (0..masks.len()).map(|slot| target_family.get_static(slot)).collect::<Vec<_>>();
        // Sample one preimage K_t per target; the trapdoor is captured once
        // and broadcast, so the resulting relation is `B K_t = T_t`.
        let preimage_family = target_family.parallel_map(|_, target| {
            trapdoor.sample_preimage(target, (b_columns, p.layout.public_key_columns())).as_mat()
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
            secret: request.secret.clone(),
            a_prime,
            public_b,
            fresh,
            masks,
            decoder_bases: bases,
            preimages: ks,
            targets,
            trapdoor: trapdoor_handle,
            names,
            declaration,
        };
        let names = wires.names.clone();
        let roles = setup_roles(&wires)?;
        let attached_b = wires.public_b.clone().derivation_attachment(
            "mxx-power-lut",
            "refresh-preprocessing",
            roles,
        )?;
        let mut context = DslContext::new("mxx-power-lut-refresh-setup");
        context = add_setup_outputs(context, &wires, &names, attached_b)?;
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
        let attachment = built
            .derivation_attachments
            .iter()
            .find(|a| a.namespace == "mxx-power-lut" && a.rule == "refresh-preprocessing")
            .cloned()
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let relations = relation_attestations(&built.graph, &attachment, masks_len(&wires))?;
        let attestation = RefreshProducerAttestation {
            graph: built.graph.clone(),
            producer_spec_hash,
            attachment,
            relations,
        };
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
            w.decoder_bases.len() != w.masks.len() ||
            w.preimages.len() != w.masks.len()
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        crate::ensure_ciphertext_only(&w.state)?;
        crate::ensure_ciphertext_only(&w.fresh)?;
        let ring = p.layout.ring();
        let cols = p.layout.public_key_columns();
        let b_columns = p.component_count * (p.digit_count + 2);
        if !same_matrix_type(w.state.vector.matrix_type(), &ring.matrix_type((1, cols))) ||
            !same_matrix_type(
                w.state.pubkey.matrix.matrix_type(),
                &ring.matrix_type((p.component_count, cols)),
            ) ||
            !same_matrix_type(w.fresh.vector.matrix_type(), &ring.matrix_type((1, cols))) ||
            !same_matrix_type(
                w.fresh.pubkey.matrix.matrix_type(),
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
        for (slot, ((mask, base), k)) in
            w.masks.iter().zip(w.decoder_bases.iter()).zip(w.preimages.iter()).enumerate()
        {
            crate::ensure_ciphertext_only(mask)?;
            crate::ensure_ciphertext_only(base)?;
            if base.pubkey.matrix.value_handle() != &b_handle ||
                base.pubkey.reveal_plaintext ||
                !same_matrix_type(base.vector.matrix_type(), &ring.matrix_type((1, b_columns))) ||
                !same_matrix_type(
                    base.pubkey.matrix.matrix_type(),
                    &ring.matrix_type((p.component_count, b_columns)),
                ) ||
                !same_matrix_type(mask.vector.matrix_type(), &ring.matrix_type((1, cols))) ||
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
                &self.request.compiler.bgg.large_scalar_mul(&w.fresh, &scale),
            )?;
            let target = combined.pubkey.matrix - scale * w.a_prime.clone();
            if !same_matrix_type(
                (w.public_b.clone() * k.clone()).matrix_type(),
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
            self.program_id == PowerLutProgramId::from_digest([0; 32])
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
            self.coefficient_count == 0 ||
            self.digit_count == 0
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

    /// Validates the frozen producer graph and its private relation records.
    pub fn validate_graph(
        &self,
        attestation: &RefreshProducerAttestation,
    ) -> Result<(), RefreshSetupError> {
        self.validate_slot_count()?;
        if self.component_count == 0 ||
            self.coefficient_count == 0 ||
            self.digit_count == 0 ||
            self.slot_scales.len() != self.slot_count ||
            self.names.masks.len() != self.slot_count ||
            self.names.decoder_base_vectors.len() != self.slot_count ||
            self.names.preimages.len() != self.slot_count
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let actual = spec_hash(&attestation.graph, &ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidManifest)?;
        if actual != self.producer_spec_hash ||
            attestation.producer_spec_hash != self.producer_spec_hash ||
            attestation.attachment.namespace != "mxx-power-lut" ||
            attestation.attachment.rule != "refresh-preprocessing"
        {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let graph = &attestation.graph;
        let mut expected_outputs = std::collections::BTreeMap::new();
        expected_outputs.insert(self.names.state_vector.clone(), ArtifactConfidentiality::Private);
        expected_outputs
            .insert(self.names.state_public_matrix.clone(), ArtifactConfidentiality::Public);
        expected_outputs.insert(self.names.a_prime.clone(), ArtifactConfidentiality::Public);
        expected_outputs
            .insert(self.names.public_matrix_b.clone(), ArtifactConfidentiality::Public);
        expected_outputs.insert(self.names.fresh_vector.clone(), ArtifactConfidentiality::Private);
        expected_outputs
            .insert(self.names.fresh_public_matrix.clone(), ArtifactConfidentiality::Public);
        for slot in 0..self.slot_count {
            expected_outputs
                .insert(self.names.masks[slot].vector.clone(), ArtifactConfidentiality::Private);
            expected_outputs.insert(
                self.names.masks[slot].public_matrix.clone(),
                ArtifactConfidentiality::Public,
            );
            expected_outputs.insert(
                self.names.decoder_base_vectors[slot].clone(),
                ArtifactConfidentiality::Private,
            );
            expected_outputs
                .insert(self.names.preimages[slot].clone(), ArtifactConfidentiality::Private);
        }
        let artifact_output_count = graph
            .outputs()
            .iter()
            .filter(|(name, output)| {
                expected_outputs.contains_key(*name) && output.confidentiality.is_some()
            })
            .count();
        if artifact_output_count != expected_outputs.len() ||
            graph.outputs().iter().any(|(name, _)| !expected_outputs.contains_key(name)) ||
            expected_outputs.iter().any(|(name, confidentiality)| {
                graph.outputs().get(name).and_then(|output| output.confidentiality) !=
                    Some(*confidentiality)
            })
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let root = graph.root_scope();
        let role = |name: &str| -> Result<ScopedWireRef, RefreshSetupError> {
            let mut found =
                attestation.attachment.roles.iter().filter(|(candidate, _)| candidate == name);
            let wire = found
                .next()
                .map(|(_, wire)| wire.clone())
                .ok_or(RefreshSetupError::InvalidManifest)?;
            if found.next().is_some() || wire.scope != FrozenGraphScopeId::Root {
                return Err(RefreshSetupError::InvalidManifest);
            }
            Ok(wire)
        };
        let output_role = |role_name: &str, output_name: &str| -> Result<(), RefreshSetupError> {
            if role(role_name)?.wire !=
                graph.outputs().get(output_name).ok_or(RefreshSetupError::InvalidManifest)?.value
            {
                return Err(RefreshSetupError::InvalidManifest);
            }
            Ok(())
        };
        let expected_role_count = 8usize
            .checked_add(self.slot_count.checked_mul(6).ok_or(RefreshSetupError::InvalidManifest)?)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let mut role_names = std::collections::BTreeSet::new();
        for (name, _) in &attestation.attachment.roles {
            if !role_names.insert(name) {
                return Err(RefreshSetupError::InvalidManifest);
            }
        }
        if role_names.len() != expected_role_count {
            return Err(RefreshSetupError::InvalidManifest);
        }
        for name in [
            "state-vector",
            "state-public",
            "a-prime",
            "public-b",
            "trapdoor",
            "fresh-vector",
            "fresh-public",
            "secret",
        ] {
            role(name)?;
        }
        output_role("state-vector", &self.names.state_vector)?;
        output_role("state-public", &self.names.state_public_matrix)?;
        output_role("a-prime", &self.names.a_prime)?;
        output_role("public-b", &self.names.public_matrix_b)?;
        output_role("fresh-vector", &self.names.fresh_vector)?;
        output_role("fresh-public", &self.names.fresh_public_matrix)?;
        for slot in 0..self.slot_count {
            output_role(&format!("slot-{slot}-mask-vector"), &self.names.masks[slot].vector)?;
            output_role(
                &format!("slot-{slot}-mask-public"),
                &self.names.masks[slot].public_matrix,
            )?;
            output_role(&format!("slot-{slot}-k-as-mat"), &self.names.preimages[slot])?;
            output_role(
                &format!("slot-{slot}-decoder-base-vector"),
                &self.names.decoder_base_vectors[slot],
            )?;
        }
        let root_node = |wire: &ScopedWireRef| root.node(wire.wire.node);
        let a_prime = role("a-prime")?;
        let a_node = root_node(&a_prime).ok_or(RefreshSetupError::InvalidManifest)?;
        let expected_tag =
            format!("mxx-power-lut/refresh/a-prime/v1/{}", hex(&self.refresh_id)).into_bytes();
        match a_node.kind() {
            mxx_ir_core::node::NodeKind::HashSample {
                matrix_type,
                variant,
                tag_prefix,
                tag_expressions,
                tag_decimal_expressions,
                tag_u64_le_expressions,
                base,
                digit_count,
            } if a_prime.wire.port.0 == 0 &&
                a_node.arguments().len() == 1 &&
                *matrix_type ==
                    mxx_dsl::Ring::new(
                        self.layout_modulus.clone(),
                        self.layout_ring_dimension.clone(),
                    )
                    .matrix_type((
                        self.component_count,
                        self.component_count * self.digit_count,
                    )) &&
                *variant == mxx_ir_core::node::HashVariant::Plain &&
                *tag_prefix == expected_tag &&
                tag_expressions.is_empty() &&
                tag_decimal_expressions.is_empty() &&
                tag_u64_le_expressions.is_empty() &&
                base.is_none() &&
                digit_count.is_none() => {}
            _ => return Err(RefreshSetupError::InvalidManifest),
        }
        let public_b = role("public-b")?;
        let trapdoor = role("trapdoor")?;
        let b_node = root_node(&public_b).ok_or(RefreshSetupError::InvalidManifest)?;
        let t_node = root_node(&trapdoor).ok_or(RefreshSetupError::InvalidManifest)?;
        if public_b.wire.port.0 != 0 ||
            trapdoor.wire.port.0 != 1 ||
            public_b.wire.node != trapdoor.wire.node ||
            !matches!(b_node.kind(), mxx_ir_core::node::NodeKind::TrapdoorSample { .. }) ||
            !matches!(t_node.kind(), mxx_ir_core::node::NodeKind::TrapdoorSample { matrix_type, sigma, gadget_base, digit_count, preimage_max_coefficient_bound, .. }
                if matrix_type == &mxx_dsl::Ring::new(
                    self.layout_modulus.clone(), self.layout_ring_dimension.clone(),
                ).matrix_type((
                    self.component_count,
                    self.component_count * (self.digit_count + 2),
                )) &&
                    sigma == &self.decoder_sigma && gadget_base == &self.layout_gadget_base &&
                    digit_count == &mxx_ir_core::IntExpr::constant(self.trapdoor_digit_count) &&
                    preimage_max_coefficient_bound == &self.decoder_preimage_bound)
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let mut relation_slots = std::collections::BTreeSet::new();
        for relation in &attestation.relations {
            if relation.slot >= self.slot_count ||
                !relation_slots.insert(relation.slot) ||
                relation.target != role(&format!("slot-{}-target", relation.slot))? ||
                relation.preimage != role(&format!("slot-{}-preimage", relation.slot))? ||
                relation.k_as_mat != role(&format!("slot-{}-k-as-mat", relation.slot))? ||
                relation.decoder_base_vector !=
                    role(&format!("slot-{}-decoder-base-vector", relation.slot))?
            {
                return Err(RefreshSetupError::InvalidManifest);
            }
            validate_relation_graph(graph, &role, relation, &public_b, &trapdoor, self)?;
        }
        if attestation.relations.len() != self.slot_count {
            return Err(RefreshSetupError::InvalidManifest);
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
            declaration.coefficient_count != parameters.coefficient_count ||
            declaration.digit_count != parameters.digit_count ||
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
        let b_columns = parameters.component_count * (parameters.digit_count + 2);
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
        let fresh_vector = imported(
            &n.fresh_vector,
            ArtifactConfidentiality::Private,
            false,
            &parameters.layout.ring().matrix_type((1, cols)),
        )?;
        let fresh_public_matrix = imported(
            &n.fresh_public_matrix,
            ArtifactConfidentiality::Public,
            false,
            &parameters.layout.ring().matrix_type((parameters.component_count, cols)),
        )?;
        let state = BggEncodingWire {
            vector: state_vector,
            pubkey: BggPublicKeyWire { matrix: state_public_matrix, reveal_plaintext: false },
            plaintext: None,
        };
        let fresh = BggEncodingWire {
            vector: fresh_vector,
            pubkey: BggPublicKeyWire { matrix: fresh_public_matrix, reveal_plaintext: false },
            plaintext: None,
        };
        let mut masks = Vec::new();
        let mut bases = Vec::new();
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
        for x in &n.decoder_base_vectors {
            let vector = imported(
                x,
                ArtifactConfidentiality::Private,
                false,
                &parameters.layout.ring().matrix_type((1, b_columns)),
            )?;
            bases.push(BggEncodingWire {
                vector,
                pubkey: BggPublicKeyWire {
                    // The base was sampled under the exact public B handle.
                    // Keep that handle when importing; reconstructing a
                    // second artifact input would lose the anchor identity.
                    matrix: public_b.clone(),
                    reveal_plaintext: false,
                },
                plaintext: None,
            });
        }
        for x in &n.preimages {
            // The graph attestation already proved the producer's
            // PreimageSample/K-as-mat relation. Imported execution consumes
            // only the ordinary private matrix artifact.
            ks.push(imported(
                x,
                ArtifactConfidentiality::Private,
                false,
                &parameters.layout.ring().matrix_type((b_columns, cols)),
            )?);
        }
        Ok(Self {
            production_id,
            parameters,
            state,
            a_prime,
            public_b,
            fresh,
            masks,
            decoder_bases: bases,
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
        compiler: &PowerLutEncodingCompiler,
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
            setup.fresh.clone(),
            setup.masks.clone(),
            setup.decoder_bases.clone(),
            setup.preimages.clone(),
        )?)
    }
}

/// Derives all preprocessing artifact names from the public setup identity.
///
/// Slot-indexed names are emitted in CRT order, so an importer cannot silently
/// exchange two mask, decoder-base, or preimage artifacts.
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
        fresh_vector: n("fresh-vector", 0),
        fresh_public_matrix: n("fresh-public", 0),
        masks: (0..s)
            .map(|i| RefreshArtifactPairNames {
                vector: n("mask-vector", i),
                public_matrix: n("mask-public", i),
            })
            .collect(),
        decoder_base_vectors: (0..s).map(|i| n("decoder-base-vector", i)).collect(),
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
        coefficient_count: p.coefficient_count,
        digit_count: p.digit_count,
        decoder_sigma: p.decoder_sigma.clone(),
        decoder_preimage_bound: decoder_preimage_bound.clone(),
        trapdoor_digit_count: p.digit_count,
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

fn masks_len(wires: &RefreshPreprocessingWires) -> usize {
    wires.masks.len()
}

/// Collects the private role map used by the frozen producer attestation.
///
/// Role handles point at graph values, including the exact preimage sample and
/// its target; they are not reconstructed from artifact names during import.
fn setup_roles(
    wires: &RefreshPreprocessingWires,
) -> Result<Vec<(String, mxx_ir_core::ValueHandle)>, RefreshSetupError> {
    let mut roles = vec![
        ("state-vector".to_owned(), wires.state.vector.value_handle().clone()),
        ("state-public".to_owned(), wires.state.pubkey.matrix.value_handle().clone()),
        ("secret".to_owned(), wires.secret.value_handle().clone()),
        ("a-prime".to_owned(), wires.a_prime.value_handle().clone()),
        ("public-b".to_owned(), wires.public_b.value_handle().clone()),
        ("trapdoor".to_owned(), wires.trapdoor.clone()),
        ("fresh-vector".to_owned(), wires.fresh.vector.value_handle().clone()),
        ("fresh-public".to_owned(), wires.fresh.pubkey.matrix.value_handle().clone()),
    ];
    for slot in 0..wires.masks.len() {
        roles.extend([
            (format!("slot-{slot}-mask-vector"), wires.masks[slot].vector.value_handle().clone()),
            (
                format!("slot-{slot}-mask-public"),
                wires.masks[slot].pubkey.matrix.value_handle().clone(),
            ),
            (format!("slot-{slot}-target"), wires.targets[slot].value_handle().clone()),
            (format!("slot-{slot}-k-as-mat"), wires.preimages[slot].value_handle().clone()),
            (
                format!("slot-{slot}-preimage"),
                wires.preimages[slot]
                    .value_handle()
                    .node()
                    .arguments()
                    .first()
                    .cloned()
                    .ok_or(RefreshSetupError::InvalidManifest)?,
            ),
            (
                format!("slot-{slot}-decoder-base-vector"),
                wires.decoder_bases[slot].vector.value_handle().clone(),
            ),
        ]);
    }
    Ok(roles)
}

/// Exports producer wires under their canonical public/private artifact names.
fn add_setup_outputs(
    mut context: DslContext,
    wires: &RefreshPreprocessingWires,
    names: &RefreshPreprocessingArtifactNames,
    attached_b: Mat,
) -> Result<DslContext, RefreshSetupError> {
    // Private vectors and K_t remain private artifacts; A', B, and the public
    // projections are exported with the declaration's canonical roles.
    context = context
        .private_output(names.state_vector.clone(), wires.state.vector.clone())?
        .public_output(names.state_public_matrix.clone(), wires.state.pubkey.matrix.clone())?
        .public_output(names.a_prime.clone(), wires.a_prime.clone())?
        .public_output(names.public_matrix_b.clone(), attached_b)?
        .private_output(names.fresh_vector.clone(), wires.fresh.vector.clone())?
        .public_output(names.fresh_public_matrix.clone(), wires.fresh.pubkey.matrix.clone())?;
    for slot in 0..wires.masks.len() {
        context = context
            .private_output(names.masks[slot].vector.clone(), wires.masks[slot].vector.clone())?
            .public_output(
                names.masks[slot].public_matrix.clone(),
                wires.masks[slot].pubkey.matrix.clone(),
            )
            .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?
            .private_output(names.preimages[slot].clone(), wires.preimages[slot].clone())?
            .private_output(
                names.decoder_base_vectors[slot].clone(),
                wires.decoder_bases[slot].vector.clone(),
            )?;
    }
    Ok(context)
}

/// Extracts one exact graph relation record for each CRT slot.
fn relation_attestations(
    _graph: &Graph,
    attachment: &mxx_dsl::FrozenDerivationAttachment,
    slots: usize,
) -> Result<Vec<RefreshPreimageRelationAttestation>, RefreshSetupError> {
    let role = |name: String| {
        attachment
            .roles
            .iter()
            .find(|(candidate, _)| candidate == &name)
            .map(|(_, wire)| wire.clone())
    };
    (0..slots)
        .map(|slot| {
            Ok(RefreshPreimageRelationAttestation {
                slot,
                target: role(format!("slot-{slot}-target"))
                    .ok_or(RefreshSetupError::InvalidManifest)?,
                preimage: role(format!("slot-{slot}-preimage"))
                    .ok_or(RefreshSetupError::InvalidManifest)?,
                k_as_mat: role(format!("slot-{slot}-k-as-mat"))
                    .ok_or(RefreshSetupError::InvalidManifest)?,
                decoder_base_vector: role(format!("slot-{slot}-decoder-base-vector"))
                    .ok_or(RefreshSetupError::InvalidManifest)?,
            })
        })
        .collect()
}

/// Validates the graph shape and operand identities for one slot relation.
///
/// This check distinguishes the exact target/preimage/decoder chain from a
/// merely same-shaped replacement wire.
fn validate_relation_graph(
    graph: &Graph,
    role: &impl Fn(&str) -> Result<ScopedWireRef, RefreshSetupError>,
    relation: &RefreshPreimageRelationAttestation,
    public_b: &ScopedWireRef,
    trapdoor: &ScopedWireRef,
    declaration: &RefreshPreprocessingDeclaration,
) -> Result<(), RefreshSetupError> {
    let packed = declaration
        .component_count
        .checked_mul(declaration.digit_count)
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let b_columns = declaration
        .component_count
        .checked_mul(declaration.digit_count + 2)
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let root = graph.root_scope();
    let node = |wire: &ScopedWireRef| {
        (wire.scope == FrozenGraphScopeId::Root).then(|| root.node(wire.wire.node)).flatten()
    };
    let state_public = role("state-public")?;
    let mask_public = role(&format!("slot-{}-mask-public", relation.slot))?;
    let fresh_public = role("fresh-public")?;
    let target_selector = node(&relation.target).ok_or(RefreshSetupError::InvalidManifest)?;
    let target_selector_args =
        root.arguments(target_selector).ok_or(RefreshSetupError::InvalidManifest)?;
    let target_family = *target_selector_args.first().ok_or(RefreshSetupError::InvalidManifest)?;
    let target_loop = root.node(target_family.node).ok_or(RefreshSetupError::InvalidManifest)?;
    let mxx_ir_core::node::NodeKind::ParallelLoop(target_loop_spec) = target_loop.kind() else {
        return Err(RefreshSetupError::InvalidManifest);
    };
    let target_loop_args = root.arguments(target_loop).ok_or(RefreshSetupError::InvalidManifest)?;
    if relation.target.wire.port.0 != 0 ||
        target_family.port.0 != 0 ||
        !matches!(
            target_selector.kind(),
            mxx_ir_core::node::NodeKind::FamilyGetStatic { index }
                if index == &mxx_ir_core::IntExpr::constant(relation.slot)
        ) ||
        target_loop_spec.count != mxx_ir_core::IntExpr::constant(declaration.slot_count) ||
        target_loop_spec.input_modes !=
            vec![
                mxx_ir_core::node::LoopInputMode::Zip,
                mxx_ir_core::node::LoopInputMode::Zip,
                mxx_ir_core::node::LoopInputMode::Broadcast,
                mxx_ir_core::node::LoopInputMode::Broadcast,
                mxx_ir_core::node::LoopInputMode::Broadcast,
            ] ||
        target_loop_args.len() != 5 ||
        target_loop_args[2] != state_public.wire ||
        target_loop_args[3] != fresh_public.wire ||
        target_loop_args[4] != role("a-prime")?.wire
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let mask_family_node =
        root.node(target_loop_args[0].node).ok_or(RefreshSetupError::InvalidManifest)?;
    let scale_family_node =
        root.node(target_loop_args[1].node).ok_or(RefreshSetupError::InvalidManifest)?;
    let mask_family_args =
        root.arguments(mask_family_node).ok_or(RefreshSetupError::InvalidManifest)?;
    let scale_family_args =
        root.arguments(scale_family_node).ok_or(RefreshSetupError::InvalidManifest)?;
    if !matches!(
        mask_family_node.kind(),
        mxx_ir_core::node::NodeKind::FamilyPack { count }
            if count == &mxx_ir_core::IntExpr::constant(declaration.slot_count)
    ) || !matches!(
        scale_family_node.kind(),
        mxx_ir_core::node::NodeKind::FamilyPack { count }
            if count == &mxx_ir_core::IntExpr::constant(declaration.slot_count)
    ) || mask_family_args.len() != declaration.slot_count ||
        scale_family_args.len() != declaration.slot_count ||
        mask_family_args[relation.slot] != mask_public.wire ||
        !matches!(
            root.node(scale_family_args[relation.slot].node)
                .ok_or(RefreshSetupError::InvalidManifest)?
                .kind(),
            mxx_ir_core::node::NodeKind::ConstantMatrix {
                value: mxx_ir_core::node::ConstantMatrix::Polynomial { coefficients },
                ..
            } if coefficients.len() == 1 &&
                coefficients[0] == declaration.slot_scales[relation.slot]
        )
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let target_body_id = graph
        .child_scope_id(&FrozenGraphScopeId::Root, target_family.node)
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let target_body = graph.scope(&target_body_id).ok_or(RefreshSetupError::InvalidManifest)?;
    if target_body.inputs().len() != 5 || target_body.outputs().len() != 1 {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let target_node = target_body
        .node(target_body.outputs()[0].node)
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let target_args =
        target_body.arguments(target_node).ok_or(RefreshSetupError::InvalidManifest)?;
    let target_scale = target_args.get(1).and_then(|wire| target_body.node(wire.node));
    let target_sum = target_args.first().and_then(|wire| target_body.node(wire.node));
    let Some(target_scale) = target_scale else { return Err(RefreshSetupError::InvalidManifest) };
    let Some(target_sum) = target_sum else { return Err(RefreshSetupError::InvalidManifest) };
    let sum_args = target_body.arguments(target_sum).ok_or(RefreshSetupError::InvalidManifest)?;
    let first_sum = sum_args.first().and_then(|wire| target_body.node(wire.node));
    let Some(first_sum) = first_sum else { return Err(RefreshSetupError::InvalidManifest) };
    let first_sum_args =
        target_body.arguments(first_sum).ok_or(RefreshSetupError::InvalidManifest)?;
    let scale_args =
        target_body.arguments(target_scale).ok_or(RefreshSetupError::InvalidManifest)?;
    if !matches!(
        target_node.kind(),
        mxx_ir_core::node::NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Subtract)
    ) || target_args.len() != 2 ||
        !is_large_scale_node(target_body, sum_args[1], target_body.inputs()[3]) ||
        !matches!(
            target_sum.kind(),
            mxx_ir_core::node::NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Add)
        ) ||
        sum_args.len() != 2 ||
        !matches!(
            first_sum.kind(),
            mxx_ir_core::node::NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Add)
        ) ||
        first_sum_args.len() != 2 ||
        first_sum_args[1] != target_body.inputs()[0] ||
        !is_large_scale_node(target_body, first_sum_args[0], target_body.inputs()[2]) ||
        !matches!(
            target_scale.kind(),
            mxx_ir_core::node::NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply)
        ) ||
        scale_args.len() != 2 ||
        scale_args[0] != target_body.inputs()[1] ||
        scale_args[1] != target_body.inputs()[4]
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let preimage_loop = node(&relation.preimage).ok_or(RefreshSetupError::InvalidManifest)?;
    let mxx_ir_core::node::NodeKind::ParallelLoop(preimage_loop_spec) = preimage_loop.kind() else {
        return Err(RefreshSetupError::InvalidManifest);
    };
    let preimage_loop_args =
        root.arguments(preimage_loop).ok_or(RefreshSetupError::InvalidManifest)?;
    if relation.preimage.wire.port.0 != 0 ||
        preimage_loop_spec.count != mxx_ir_core::IntExpr::constant(declaration.slot_count) ||
        preimage_loop_spec.input_modes !=
            [
                mxx_ir_core::node::LoopInputMode::Zip,
                mxx_ir_core::node::LoopInputMode::Broadcast,
                mxx_ir_core::node::LoopInputMode::Broadcast,
            ] ||
        preimage_loop_args.len() != 3 ||
        preimage_loop_args[0] != target_family ||
        preimage_loop_args[1] != public_b.wire ||
        preimage_loop_args[2] != trapdoor.wire
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let preimage_body_id = graph
        .child_scope_id(
            &FrozenGraphScopeId::Root,
            root.node_id(preimage_loop).ok_or(RefreshSetupError::InvalidManifest)?,
        )
        .ok_or(RefreshSetupError::InvalidManifest)?;
    let body = graph.scope(&preimage_body_id).ok_or(RefreshSetupError::InvalidManifest)?;
    if body.inputs().len() != 3 {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let preimage_nodes = body
        .nodes()
        .iter()
        .filter(|candidate| {
            matches!(candidate.kind(), mxx_ir_core::node::NodeKind::PreimageSample { .. })
        })
        .collect::<Vec<_>>();
    if preimage_nodes.len() != 1 {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let preimage_node = preimage_nodes[0];
    let preimage_args = body.arguments(preimage_node).ok_or(RefreshSetupError::InvalidManifest)?;
    let preimage_wire = mxx_ir_core::WireRef {
        node: body.node_id(preimage_node).ok_or(RefreshSetupError::InvalidManifest)?,
        port: mxx_ir_core::types::Port(0),
    };
    if !matches!(
        preimage_node.kind(),
        mxx_ir_core::node::NodeKind::PreimageSample { matrix_type, max_coefficient_bound }
            if max_coefficient_bound == &declaration.decoder_preimage_bound &&
                matrix_type.rows == b_columns.into() && matrix_type.columns == packed.into()
    ) || preimage_args.len() != 3 ||
        preimage_args[0] != body.inputs()[1] ||
        preimage_args[1] != body.inputs()[2] ||
        preimage_args[2] != body.inputs()[0] ||
        body.outputs().len() != 1 ||
        {
            match body.outputs().first().and_then(|output| body.node(output.node)) {
                Some(output_node) => {
                    let output_args = body.arguments(output_node);
                    !matches!(
                        (output_node.kind(), output_args.as_deref()),
                        (
                            mxx_ir_core::node::NodeKind::MatrixScale { scalar },
                            Some([argument])
                        ) if scalar == &mxx_ir_core::IntExpr::constant(1) &&
                            *argument == preimage_wire &&
                            matches!(
                                output_node.output_types().first(),
                                Some(WireType::Matrix(matrix_type))
                                    if matrix_type.rows == b_columns.into() &&
                                        matrix_type.columns == packed.into()
                            )
                    )
                }
                None => true,
            }
        }
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let k_node = node(&relation.k_as_mat).ok_or(RefreshSetupError::InvalidManifest)?;
    let k_args = root.arguments(k_node).ok_or(RefreshSetupError::InvalidManifest)?;
    if relation.k_as_mat.wire.port.0 != 0 ||
        !matches!(
            k_node.kind(),
            mxx_ir_core::node::NodeKind::FamilyGetStatic { index }
                if index == &mxx_ir_core::IntExpr::constant(relation.slot)
        ) ||
        k_args.len() != 1 ||
        k_args[0] != relation.preimage.wire
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let base = role(&format!("slot-{}-decoder-base-vector", relation.slot))?;
    let base_node = node(&base).ok_or(RefreshSetupError::InvalidManifest)?;
    let secret = role("secret")?;
    let base_args = root.arguments(base_node).ok_or(RefreshSetupError::InvalidManifest)?;
    if base.wire.port.0 != 0 ||
        !matches!(
            base_node.kind(),
            mxx_ir_core::node::NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply)
        ) ||
        base_args.len() != 2 ||
        base_args[0] != secret.wire ||
        base_args[1] != public_b.wire
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    Ok(())
}

fn is_large_scale_node(
    root: &mxx_ir_core::graph::GraphScope,
    wire: mxx_ir_core::WireRef,
    input: mxx_ir_core::WireRef,
) -> bool {
    let Some(node) = root.node(wire.node) else { return false };
    if !matches!(
        node.kind(),
        mxx_ir_core::node::NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply)
    ) {
        return false;
    }
    let Some(arguments) = root.arguments(node) else { return false };
    arguments.len() == 2 && arguments[0] == input
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
        digit_count: usize,
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
        program_id: PowerLutProgramId,
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
        coefficient_count: usize,
        digit_count: usize,
        lut_width: usize,
        layout: Layout,
        refresh: Refresh,
        decoder_sigma: &'a mxx_ir_core::RealExpr,
        decoder_preimage_bound: &'a mxx_ir_core::IntExpr,
    }
    let payload = Payload {
        schema: "mxx-power-lut/refresh-setup/v4",
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
        coefficient_count: p.coefficient_count,
        digit_count: p.digit_count,
        lut_width: p.lut_width,
        layout: Layout {
            modulus: p.layout.modulus.clone().canonicalize(),
            ring_dimension: p.layout.ring_dimension.clone().canonicalize(),
            secret_dimension: p.layout.secret_dimension,
            digit_count: p.layout.digit_count,
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
    let check = |name: &str,
                 confidentiality: ArtifactConfidentiality,
                 rows: usize,
                 columns: usize|
     -> Result<(), RefreshSetupError> {
        let artifact = m.artifacts.get(name).ok_or(RefreshSetupError::InvalidManifest)?;
        if artifact.confidentiality != confidentiality ||
            artifact.family_count.is_some() ||
            artifact.artifact_type != matrix_type(rows, columns) ||
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
    let slots = p.refresh.crt_plaintext_moduli.len();
    let expected_artifact_count = 6usize
        .checked_add(slots.checked_mul(4).ok_or(RefreshSetupError::InvalidManifest)?)
        .ok_or(RefreshSetupError::InvalidManifest)?;
    if m.artifacts.len() != expected_artifact_count {
        return Err(RefreshSetupError::InvalidManifest);
    }
    if n.masks.len() != slots || n.decoder_base_vectors.len() != slots || n.preimages.len() != slots
    {
        return Err(RefreshSetupError::InvalidManifest);
    }
    let cols = p.layout.public_key_columns();
    let b_columns = p
        .component_count
        .checked_mul(p.digit_count + 2)
        .ok_or(RefreshSetupError::InvalidManifest)?;
    check(&n.state_vector, ArtifactConfidentiality::Private, 1, cols)?;
    check(&n.state_public_matrix, ArtifactConfidentiality::Public, p.component_count, cols)?;
    check(&n.a_prime, ArtifactConfidentiality::Public, p.component_count, cols)?;
    check(&n.public_matrix_b, ArtifactConfidentiality::Public, p.component_count, b_columns)?;
    check(&n.fresh_vector, ArtifactConfidentiality::Private, 1, cols)?;
    check(&n.fresh_public_matrix, ArtifactConfidentiality::Public, p.component_count, cols)?;
    for mask in &n.masks {
        check(&mask.vector, ArtifactConfidentiality::Private, 1, cols)?;
        check(&mask.public_matrix, ArtifactConfidentiality::Public, p.component_count, cols)?;
    }
    for base in &n.decoder_base_vectors {
        check(base, ArtifactConfidentiality::Private, 1, b_columns)?;
    }
    for preimage in &n.preimages {
        check(preimage, ArtifactConfidentiality::Private, b_columns, cols)?;
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
            context = context.private_output(
                format!("refresh-decoder-residual-{column}"),
                residual.clone().semantic_anchor(format!("refresh.decoder.residual.{column}"))?,
            )?;
        }
        let decoded_prefix = decoded_prefix.into();
        for (index, value) in self.decoded.iter().enumerate() {
            context = context.bool_output(
                format!("{decoded_prefix}_{index}"),
                value.clone().semantic_anchor(format!("refresh.decoder.output.{index}"))?,
            )?;
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
    digit_count: usize,
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
    let g = ring.gadget(mask_secret.matrix_type().columns.clone(), gadget_base, digit_count);
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

/// Exact matrix facts exported by the refresh simulation adapter for the
/// protocol-agnostic operational checker.  This type deliberately mirrors
/// only structural matrix facts; it has no dependency on `mxx-correctness`.
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
    encoding_error_sigma: mxx_ir_core::RealExpr,
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
        encoding_error_sigma: mxx_ir_core::RealExpr,
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
        let encoding_sigma = encoding_error_sigma
            .evaluate_f64(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters("encoding sigma must be concrete"))?;
        if !encoding_sigma.is_finite() || encoding_sigma <= 0.0 {
            return Err(RefreshSetupError::InvalidParameters(
                "encoding sigma must be positive and finite",
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
            encoding_error_sigma,
            expected_plaintext,
            decode_length,
        })
    }

    /// Builds the three real graph stages used by the generic checker.
    pub fn build(self) -> Result<RefreshParameterSimulationBundle, RefreshSetupError> {
        let setup = self.setup;
        let layout = self.generated_layout.public_layout.clone();
        let ring = setup.layout.ring();
        let hash_key = ring.bytes_input("refresh-parameter-search-hash-key", 32);
        let encoding_sigma = self
            .encoding_error_sigma
            .evaluate_f64(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters("encoding sigma must be concrete"))?;
        let sigma_bound = BigDecimal::from_f64(encoding_sigma)
            .ok_or(RefreshSetupError::InvalidParameters("encoding sigma is not finite"))?;
        let encoding_bound =
            mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma_bound)
                .to_bigint()
                .ok_or(RefreshSetupError::InvalidParameters("encoding cutoff overflow"))?;
        let sampler = PowerLutEncodingSampler {
            layout: setup.layout.clone(),
            gaussian_sigma: Some(self.encoding_error_sigma.clone()),
            gaussian_max_coefficient_bound: Some(encoding_bound.into()),
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
        let selectors = EncodingSelectorFamily::new(gsw)?;
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
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
        let reduction_table =
            (0..setup.lut_width).map(|value| value % self.profile.q_l()).collect::<Vec<_>>();
        let reduction_lut = crate::program::LutTable::unary(
            setup.lut_width,
            setup.lut_width,
            reduction_table.clone(),
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid reduction LUT"))?;
        let program = SparseLwrPrfProgram::new(self.profile.clone(), layout.bucket_width)?;
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
            setup.lut_width.max(rounding_lut.values().len()),
            b"refresh-parameter-search-mask-bank".as_slice(),
        )?;
        let helpers = BTreeMap::from([(
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
        let rounding_helpers = sampler.sample_flat_helpers_for_lut(
            secret.clone(),
            None,
            hash_key.clone(),
            rounding_lut,
            mask_bank.as_ref(),
            b"refresh-parameter-search-rounding".as_slice(),
        )?;
        let rounding_helpers = FlatLutHelperSet::new(rounding_lut, rounding_helpers)?;
        let labels = crate::refresh::RefreshPrfLabelIndex::new(
            setup.refresh_id,
            setup.refresh.crt_plaintext_moduli.len(),
            setup.component_count,
            setup.coefficient_count,
            setup.digit_count,
        )?;
        let batch = RefreshPrfBatchInputs::new(&layout, self.profile, &labels)?;
        let mask_count = setup.refresh.crt_plaintext_moduli.len() *
            setup.component_count *
            setup.coefficient_count *
            setup.digit_count;
        let fresh_count = setup.component_count * setup.coefficient_count * setup.digit_count;
        let total = mask_count + fresh_count;
        let outputs = program.compile_pbc_encoding_family_typed_with_batch_and_rounding_helpers(
            &compiler,
            Family::pack(
                (0..total)
                    .map(|index| {
                        if index < mask_count { state.vector.clone() } else { fresh.vector.clone() }
                    })
                    .collect(),
            )?,
            Family::pack(
                (0..total)
                    .map(|index| {
                        if index < mask_count {
                            state.pubkey.matrix.clone()
                        } else {
                            fresh.pubkey.matrix.clone()
                        }
                    })
                    .collect(),
            )?,
            &batch,
            selectors,
            &helpers,
            &rounding_helpers,
        )?;
        let mut masks =
            (0..setup.refresh.crt_plaintext_moduli.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for index in 0..mask_count {
            let label = labels.label(index).ok_or(RefreshSetupError::InvalidManifest)?;
            let crate::refresh::RefreshPrfLabel::Mask {
                slot, component, coefficient, digit, ..
            } = label
            else {
                return Err(RefreshSetupError::InvalidManifest);
            };
            masks[slot].push(crate::refresh::RefreshMaskPrfOutput::from_pbc_evaluation(
                outputs.project(index).map_err(|_| RefreshSetupError::InvalidManifest)?,
                setup.refresh_id,
                slot,
                component,
                coefficient,
                digit,
            )?);
        }
        let mut fresh_error = Vec::with_capacity(fresh_count);
        for offset in 0..fresh_count {
            let index = mask_count + offset;
            let label = labels.label(index).ok_or(RefreshSetupError::InvalidManifest)?;
            let crate::refresh::RefreshPrfLabel::FreshError {
                component, coefficient, digit, ..
            } = label
            else {
                return Err(RefreshSetupError::InvalidManifest);
            };
            fresh_error.push(crate::refresh::RefreshFreshErrorPrfOutput::from_pbc_evaluation(
                outputs.project(index).map_err(|_| RefreshSetupError::InvalidManifest)?,
                setup.refresh_id,
                component,
                coefficient,
                digit,
            )?);
        }
        let prf = RefreshPrfInputs::from_pbc_outputs(&setup, &program, masks, fresh_error)?;
        let producer = RefreshPreprocessingProducer::build(RefreshPreprocessingRequest {
            parameters: setup.clone(),
            prf,
            compiler: PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
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
        let producer_validated = producer
            .built
            .validate_with_manifests(
                &ParamEnv::default(),
                &std::collections::BTreeMap::from([(selector_production, selector_manifest)]),
            )
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
            producer_production,
            setup.clone(),
            producer.declaration().clone(),
            producer.attestation(),
            &producer_manifest,
        )?;
        let manifest = setup.refresh.bind_imported_setup(&compiler, &imported)?;
        let refreshed = setup.refresh.refresh(&compiler, &manifest)?;
        let verification = build_refresh_verification(
            refreshed.encoding(),
            &secret_input,
            &secret_input,
            &self.expected_plaintext,
            setup.layout.gadget_base.clone(),
            setup.digit_count,
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
        let preprocessing_graph = producer.built;
        Ok(RefreshParameterSimulationBundle {
            selector_graph,
            preprocessing_graph,
            verification_graph,
            public_identity,
            metadata,
            targets,
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
        WireType::IndexedFamily { element, count } if expected_family => {
            let WireType::Matrix(matrix) = element.as_ref() else {
                return Err(RefreshSetupError::InvalidManifest);
            };
            (matrix, Some(count))
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
    match (count, expected_count) {
        (None, None) => {}
        (Some(actual), Some(expected))
            if actual.evaluate(&ParamEnv::default()).ok().and_then(|v| v.to_usize()) ==
                Some(expected) => {}
        _ => return Err(RefreshSetupError::InvalidManifest),
    }
    Ok((matrix.clone(), count.cloned()))
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

/// The three immutable graph stages plus the exact metadata needed by the
/// generic operational checker.  No schedules, secrets, or raw constructors
/// are exposed by this type.
pub struct RefreshParameterSimulationBundle {
    selector_graph: BuiltGraph,
    preprocessing_graph: BuiltGraph,
    verification_graph: BuiltGraph,
    public_identity: [u8; 32],
    metadata: std::collections::BTreeMap<String, RefreshSimulationMatrixInputMetadata>,
    targets: Vec<RefreshSimulationDecoderTarget>,
}

impl RefreshParameterSimulationBundle {
    /// Returns the graph entrypoint expected by the simulation adapter.
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
        &self.preprocessing_graph
    }
    /// Returns the verification graph consumed by the simulation adapter.
    pub fn verification_graph(&self) -> &BuiltGraph {
        &self.verification_graph
    }
    /// Returns exact matrix facts collected from all three graph stages.
    pub fn matrix_input_metadata(
        &self,
    ) -> &std::collections::BTreeMap<String, RefreshSimulationMatrixInputMetadata> {
        &self.metadata
    }
    /// Returns the frozen residual-to-decoder links for every target column.
    pub fn decoder_targets(&self) -> &[RefreshSimulationDecoderTarget] {
        &self.targets
    }
}

/// Collects constant-coefficient bounds for matrix inputs in all graph stages.
///
/// The simulation adapter consumes these facts as metadata; it does not infer
/// them from a runtime ciphertext or from a private setup vector.
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
                wire_type: WireType::IndexedFamily { element, .. },
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

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_bgg::BggEncodingCompiler;
    use mxx_ir_core::artifact::SpecHash;

    struct RelationFixture {
        graph: Graph,
        roles: BTreeMap<String, ScopedWireRef>,
        relation: RefreshPreimageRelationAttestation,
        wrong_target: ScopedWireRef,
        wrong_preimage: ScopedWireRef,
        public_b: ScopedWireRef,
        trapdoor: ScopedWireRef,
        declaration: RefreshPreprocessingDeclaration,
    }

    fn relation_fixture() -> RelationFixture {
        let ring = mxx_dsl::Ring::new(97, 4);
        let bgg = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 2.into(),
                digit_count: 2.into(),
            },
        };
        let state = BggEncodingWire {
            vector: ring.input("state-vector", (1, 4)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("state-public", (2, 4)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let fresh = BggEncodingWire {
            vector: ring.input("fresh-vector", (1, 4)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("fresh-public", (2, 4)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let a_prime = ring.input("a-prime", (2, 4));
        let mask_public = ring.input("mask-public", (2, 4));
        let wrong_mask_public = ring.input("wrong-mask-public", (2, 4));
        let trapdoor = ring.sample_trapdoor(2, 1, 2, 2, 100);
        let public_b = trapdoor.public_matrix();
        let masks = Family::pack(vec![mask_public.clone()]).expect("mask family");
        let scales = Family::pack(vec![ring.polynomial([1.into()])]).expect("scale family");
        let target_family =
            Family::<Mat>::parallel_zip_many_values(vec![masks, scales.clone()], |_, mut items| {
                let mask = items.remove(0);
                let scale = items.remove(0);
                bgg.large_scalar_mul(&state, &scale).pubkey.matrix +
                    mask +
                    bgg.large_scalar_mul(&fresh, &scale).pubkey.matrix -
                    scale * a_prime.clone()
            })
            .expect("target family");
        let preimage_family = target_family
            .clone()
            .parallel_map(|_, target| trapdoor.sample_preimage(target, (8, 4)).as_mat())
            .expect("preimage family");
        let wrong_masks = Family::pack(vec![wrong_mask_public.clone()]).expect("wrong mask family");
        let wrong_target_family = Family::<Mat>::parallel_zip_many_values(
            vec![wrong_masks, scales.clone()],
            |_, mut items| {
                let mask = items.remove(0);
                let scale = items.remove(0);
                bgg.large_scalar_mul(&state, &scale).pubkey.matrix +
                    mask +
                    bgg.large_scalar_mul(&fresh, &scale).pubkey.matrix -
                    scale * a_prime.clone()
            },
        )
        .expect("wrong target family");
        let wrong_preimage_family = wrong_target_family
            .parallel_map(|_, target| trapdoor.sample_preimage(target, (8, 4)).as_mat())
            .expect("wrong preimage family");
        let target = target_family.get_static(0);
        let wrong_target = target_family.get_static(1);
        let k_as_mat = preimage_family.get_static(0);
        let wrong_k_as_mat = wrong_preimage_family.get_static(0);
        let decoder_base = state.vector.clone() * public_b.clone();
        let mut context = DslContext::new("refresh-setup-attestation-test");
        context = context
            .output("target", target.clone())
            .expect("target output")
            .output("wrong-target", wrong_target.clone())
            .expect("wrong target output")
            .output("preimage", k_as_mat.clone())
            .expect("preimage output")
            .output("wrong-preimage", wrong_k_as_mat)
            .expect("wrong preimage output")
            .output("public-b", public_b.clone())
            .expect("public B output")
            .output("state-public", state.pubkey.matrix.clone())
            .expect("state public output")
            .output("fresh-public", fresh.pubkey.matrix.clone())
            .expect("fresh public output")
            .output("a-prime", a_prime.clone())
            .expect("a-prime output")
            .output("mask-public", mask_public.clone())
            .expect("mask public output")
            .output("state-vector", state.vector.clone())
            .expect("state vector output")
            .output("decoder-base-vector", decoder_base)
            .expect("decoder base output")
            .private_trapdoor_output("trapdoor", trapdoor.clone())
            .expect("trapdoor output");
        let graph = context.build().expect("attestation graph").graph;
        let root = graph.root_scope();
        let output = |name: &str| ScopedWireRef {
            scope: FrozenGraphScopeId::Root,
            wire: graph.outputs().get(name).expect("test output").value,
        };
        let target = output("target");
        let wrong_target = output("wrong-target");
        let k_as_mat = output("preimage");
        let wrong_k_as_mat = output("wrong-preimage");
        let public_b = output("public-b");
        let trapdoor = output("trapdoor");
        let preimage_node = root.node(k_as_mat.wire.node).expect("preimage output node");
        let preimage = ScopedWireRef {
            scope: FrozenGraphScopeId::Root,
            wire: root.arguments(preimage_node).expect("preimage output arguments")[0],
        };
        let wrong_preimage_node =
            root.node(wrong_k_as_mat.wire.node).expect("wrong preimage output node");
        let wrong_preimage = ScopedWireRef {
            scope: FrozenGraphScopeId::Root,
            wire: root.arguments(wrong_preimage_node).expect("wrong preimage output arguments")[0],
        };
        let mut roles = BTreeMap::from([
            ("state-public".to_owned(), output("state-public")),
            ("fresh-public".to_owned(), output("fresh-public")),
            ("a-prime".to_owned(), output("a-prime")),
            ("public-b".to_owned(), public_b.clone()),
            ("trapdoor".to_owned(), trapdoor.clone()),
            ("secret".to_owned(), output("state-vector")),
        ]);
        roles.insert("slot-0-mask-public".to_owned(), output("mask-public"));
        roles.insert("slot-0-decoder-base-vector".to_owned(), output("decoder-base-vector"));
        let declaration = RefreshPreprocessingDeclaration {
            identity: [0; 32],
            producer_spec_hash: SpecHash([0; 32]),
            pbc_layout_id: crate::pbc::PbcLayoutId([0; 32]),
            refresh_id: [0; 32],
            program_id: PowerLutProgramId::from_digest([1; 32]),
            prf_q_l: 2,
            prf_p: 2,
            prf_lut_width: 4,
            prf_ring_dimension: 4,
            prf_terminal_form: SparseLwrPrfTerminalForm::RawScalar,
            prf_output_wire: crate::program::ProgramWireId::from_index(0),
            names: RefreshPreprocessingArtifactNames {
                state_vector: String::new(),
                state_public_matrix: String::new(),
                a_prime: String::new(),
                public_matrix_b: String::new(),
                fresh_vector: String::new(),
                fresh_public_matrix: String::new(),
                masks: Vec::new(),
                decoder_base_vectors: Vec::new(),
                preimages: Vec::new(),
            },
            slot_count: 1,
            component_count: 2,
            coefficient_count: 1,
            digit_count: 2,
            decoder_sigma: 1.into(),
            decoder_preimage_bound: 100.into(),
            trapdoor_digit_count: 2,
            slot_scales: vec![1.into()],
            layout_modulus: 97.into(),
            layout_ring_dimension: 4.into(),
            layout_gadget_base: 2.into(),
        };
        let relation = RefreshPreimageRelationAttestation {
            slot: 0,
            target,
            preimage,
            k_as_mat,
            decoder_base_vector: public_b.clone(),
        };
        RelationFixture {
            graph,
            roles,
            relation,
            wrong_target,
            wrong_preimage,
            public_b,
            trapdoor,
            declaration,
        }
    }

    fn validate_fixture(
        fixture: &RelationFixture,
        relation: &RefreshPreimageRelationAttestation,
    ) -> Result<(), RefreshSetupError> {
        let role =
            |name: &str| fixture.roles.get(name).cloned().ok_or(RefreshSetupError::InvalidManifest);
        validate_relation_graph(
            &fixture.graph,
            &role,
            relation,
            &fixture.public_b,
            &fixture.trapdoor,
            &fixture.declaration,
        )
    }

    #[test]
    fn relation_attestation_accepts_two_separate_structural_loops() {
        let fixture = relation_fixture();
        validate_fixture(&fixture, &fixture.relation).expect("valid two-loop attestation");
    }

    #[test]
    fn relation_attestation_rejects_wrong_target_family() {
        let fixture = relation_fixture();
        let mut relation = fixture.relation.clone();
        relation.target = relation.k_as_mat.clone();
        assert!(validate_fixture(&fixture, &relation).is_err());
    }

    #[test]
    fn relation_attestation_rejects_wrong_target_index() {
        let fixture = relation_fixture();
        let mut relation = fixture.relation.clone();
        relation.target = fixture.wrong_target.clone();
        assert!(validate_fixture(&fixture, &relation).is_err());
    }

    #[test]
    fn relation_attestation_rejects_wrong_preimage_loop_output() {
        let fixture = relation_fixture();
        let mut relation = fixture.relation.clone();
        relation.preimage = relation.k_as_mat.clone();
        assert!(validate_fixture(&fixture, &relation).is_err());
    }

    #[test]
    fn relation_attestation_rejects_wrong_preimage_target() {
        let fixture = relation_fixture();
        let mut relation = fixture.relation.clone();
        relation.preimage = fixture.wrong_preimage.clone();
        assert!(validate_fixture(&fixture, &relation).is_err());
    }

    #[test]
    fn relation_attestation_rejects_wrong_preimage_loop_input_mode() {
        let fixture = relation_fixture();
        let mut relation = fixture.relation.clone();
        relation.preimage = relation.target.clone();
        assert!(validate_fixture(&fixture, &relation).is_err());
    }

    #[test]
    fn preprocessing_declaration_rejects_single_crt_slot() {
        let fixture = relation_fixture();
        assert!(matches!(
            fixture.declaration.validate_slot_count(),
            Err(RefreshSetupError::InvalidManifest)
        ));
    }
}
