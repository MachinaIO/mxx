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
//! Setup material is organized per CRT slot `t`. The producer switches the
//! complete `A_{sum,t} = A_t + A_{m,t} + A_{e,t}` to the selected ring `R_P`,
//! then targets `T_t = MS(A_{sum,t}) - (P_t/q_t) reduce(A')` with `B_P K_t=T_t`.
//! Each distinct `P` has one preprocessing-fixed public matrix and trapdoor.
//! Online evaluation forms one fresh `b_P=reduce(s) B_P+e_P` per group and
//! uses `d_t=b_P K_t` for that group's slots. Neither the
//! online secret `s` nor `b` crosses the artifact boundary.

use crate::{
    bgg::{
        BggPreparedReadout, BggRefreshParameters, BggScheme,
        encoding::{ExponentLutEncodingCompiler, ExponentLutEncodingSampler},
        noise::{
            BggNoiseSnapshot, NoiseGainPlan, ProgramNoiseReport, RefreshNoiseGainDecompositions,
            compact_noise_snapshot_plans, compact_readout_plan_schema,
            producer_readout_plan_schema,
        },
        pbc_selector_metadata, setup_identity_digest,
    },
    noise_refresh::{
        pbc::{
            PbcActiveCellIndex, PbcPublicLayout, PbcSelectorArtifactNames, PbcSelectorArtifacts,
            PbcSelectorRuntimeBinding, PbcTrustedSelectorBits, selector_family_artifact_name,
        },
        prf::{
            PreparedSparseLwrPrf, PreparedSparseLwrPrfDescriptor, PrfPublicSetup,
            SparseLwrPrfProgram,
        },
        refresh::{RefreshError, RefreshPrfContract},
        simulation::{AverageCaseConfig, ExponentLutAverageNoiseReport, ExponentLutNoiseReport},
    },
    program::ProgramValue,
    scheme::ExponentLUTFixedRhsScheme,
};
use mxx_bgg::{BggEncodingWire, BggPublicKeyWire};
use mxx_dsl::{
    Bool, BuiltGraph, Bytes, DslContext, Family, GraphValue, GraphValueSchema, HashTag, Mat,
    Preimage,
};
use mxx_ir_core::{
    BenchmarkRole, IntExpr, ParamEnv, ScopedWireRef,
    artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId},
    encoding::spec_hash,
    graph::FrozenGraphScopeId,
    types::{NodeId, WireType},
    with_benchmark_role,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
/// Errors raised while building or importing trusted refresh setup.
pub enum RefreshSetupError {
    #[error(transparent)]
    Prf(#[from] crate::noise_refresh::prf::PrfError),
    #[error(transparent)]
    /// The refresh declaration or its CRT equations are invalid.
    Refresh(#[from] RefreshError),
    /// Backend errors retain their original type and source chain. Conversion
    /// belongs to each adapter, so this application error has no concrete
    /// encoding, sampler, or trapdoor dependency.
    #[error("{0}")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),
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

/// Inputs for constructing the trusted refresh preprocessing graph.
///
/// The public PRF initial key is also the state encoding's public key.
/// The state vector and online secret never enter this request. The resulting
/// graph exports named artifacts for typed import at the execution boundary.
pub struct RefreshPreprocessingRequest {
    /// Frozen PRF schedule shared by preparation, execution and noise tracing.
    pub program: SparseLwrPrfProgram,
    /// Public structural PBC layout used to resolve active selector ranges.
    pub layout: PbcPublicLayout,
    /// Validated dimensions and CRT refresh declaration.
    pub parameters: BggRefreshParameters,
    /// Setup-fixed public PRF and readout inputs. This value contains no
    /// label-dependent vectors, monomials, or online refresh outputs.
    pub public_prf: PrfPublicSetup<BggScheme>,
    /// Setup secret `t` used by the preprocessing equations. This is not the
    /// online evaluator secret `s`; it crosses the trusted local boundary only
    /// as a typed private setup artifact and is never transmitted.
    pub setup_secret_t: Mat,
    /// The setup-generated selector GSW family. It is emitted by this same
    /// producer graph so the selector and decoder setup share one `t` sample.
    pub selector_gsw: Family<Mat>,
    /// Canonical PBC artifact name for the selector family.
    pub selector_gsw_name: String,
    /// Public native label monomials, used only to compose preprocessing-fixed RHSs.
    pub label_plaintexts: Family<Mat>,
    /// Hash key used for deterministic public matrix derivation.
    pub hash_key: Bytes,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Canonical artifact names emitted by refresh preprocessing.
pub struct RefreshPreprocessingArtifactNames {
    /// Private setup secret `t`, retained locally for online/verification use.
    pub setup_secret_t: String,
    /// Setup-fixed selector GSW family used by online accumulation.
    pub selector_gsw: String,
    /// Public typed decompositions of each CRT `kappa_t` state scale.
    pub state_scaling: Vec<String>,
    /// Public typed route decompositions for mask and fresh label families.
    pub mask_route_rhs_decompositions: String,
    pub fresh_route_rhs_decompositions: String,
    /// Public matrix carried by the online state encoding.
    pub state_public_matrix: String,
    /// Public random matrix used in the target equation.
    pub a_prime: String,
    /// One public trapdoor matrix `B_P` per distinct switching modulus,
    /// in ascending modulus order.
    pub public_matrix_b: Vec<String>,
    /// Exact public refresh target matrix for each CRT slot.
    pub refresh_targets: Vec<String>,
    /// Per-slot sampled preimage matrices.
    pub preimages: Vec<String>,
    /// Public value-readout and typed decomposition-cache artifacts.
    pub readout: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Public declaration of a preprocessing producer and its artifact schema.
///
/// The identity commits to all dimensions, setup identities, decoder policy,
/// scales, and canonical artifact names. Parameters remain authoritative in
/// the producer/import owner: this declaration binds their digest instead of
/// serializing a second, independently mutable projection of their fields.
/// The PRF binding and prepared-value schema are supplied by the encoding
/// capability; this declaration does not prescribe a cache representation.
pub struct RefreshPreprocessingDeclaration<B, D> {
    /// Digest of the resolved setup parameters, PRF contract and readout identity.
    pub identity: [u8; 32],
    /// PBC layout identity used by the PRF outputs.
    pub pbc_layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    /// Canonical program and terminal contract, serialized without reconstructing it.
    pub(crate) prf_contract: RefreshPrfContract,
    /// Canonical names and roles of all produced artifacts.
    pub names: RefreshPreprocessingArtifactNames,
    pub prepared_prf: PreparedSparseLwrPrfDescriptor<B, D>,
    /// Stable identity of the value-form readout used by this setup.
    pub readout_identity: crate::readout::ValueReadoutIdentity,
}

#[derive(Clone)]
/// Runtime setup material imported from a validated producer declaration.
///
/// Private setup values and preimages remain crate-visible only. Consumers use
/// them through the online construction stage after all manifest,
/// identity, and artifact-schema checks succeed.
pub struct ImportedRefreshSetup {
    parameters: BggRefreshParameters,
    pub(crate) state: BggPublicKeyWire,
    pub(crate) setup_secret_t: Mat,
    pub(crate) a_prime: Mat,
    pub(crate) public_b: Vec<Mat>,
    pub(crate) refresh_targets: Vec<Mat>,
    pub(crate) readout_artifacts: BggPreparedReadout,
    pub(crate) program_cache: PreparedSparseLwrPrf<BggScheme>,
    pub(crate) state_scaling: Vec<mxx_dsl::Preimage>,
    pub(crate) mask_route_rhs_decompositions: Family<mxx_dsl::Preimage>,
    pub(crate) fresh_route_rhs_decompositions: Family<mxx_dsl::Preimage>,
    pub(crate) preimages: Vec<mxx_dsl::Preimage>,
    declaration: RefreshPreprocessingDeclaration<
        <BggScheme as ExponentLUTFixedRhsScheme>::FixedProductBinding,
        <<BggScheme as ExponentLUTFixedRhsScheme>::PreparedFixedProduct as GraphValue>::Schema,
    >,
}
impl std::fmt::Debug for ImportedRefreshSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedRefreshSetup")
            .field("identity", &self.declaration.identity)
            .field("slot_count", &self.state_scaling.len())
            .finish()
    }
}

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

/// Immutable producer for the trusted refresh preprocessing graph.
///
/// Building this value samples setup artifacts, freezes the graph, and
/// validates the declaration and output schema. Live construction wires and
/// the request are released after freezing; callers export the declaration and use
/// [`ImportedRefreshSetup::import`] at an execution boundary.
pub struct RefreshPreprocessingProducer {
    parameters: BggRefreshParameters,
    noise_plans: (NoiseGainPlan, RefreshNoiseGainDecompositions),
    built: BuiltGraph,
    declaration: RefreshPreprocessingDeclaration<
        <BggScheme as ExponentLUTFixedRhsScheme>::FixedProductBinding,
        <<BggScheme as ExponentLUTFixedRhsScheme>::PreparedFixedProduct as GraphValue>::Schema,
    >,
}

impl RefreshPreprocessingProducer {
    /// Builds and validates the immutable preprocessing graph.
    pub fn build(request: RefreshPreprocessingRequest) -> Result<Self, RefreshSetupError> {
        request.public_prf.validate(&request.parameters)?;
        let decoder_preimage_bound = request.parameters.resolve_decoder_preimage_bound()?;
        let p = &request.parameters;
        let ring = p.layout.ring();
        p.validate_prf(&RefreshPrfContract::from_program(&request.program))?;
        request.layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let scheme = BggScheme::new(p.public_key_compiler(), request.program.profile().layout);
        let selectors = crate::bgg::encoding::SetupRhsFamily::new_for_layout(
            request.selector_gsw.clone(),
            request.layout.layout_id,
        )?;
        let blocks = request.program.selection_inputs(
            &request.layout,
            &selectors,
            &request.label_plaintexts,
            request.layout.layout_id.0,
        )?;
        let (public_outputs, prepared_prf) =
            with_benchmark_role(BenchmarkRole::PublicPrfAccumulation, || {
                PreparedSparseLwrPrf::new(
                    &scheme,
                    request.program.clone(),
                    ProgramValue::Scalar(request.public_prf.initial.clone()),
                    &blocks,
                )
            })
            .map_err(|error| RefreshSetupError::SetupStage {
                stage: "prepare fixed-RHS PRF",
                detail: error.to_string(),
            })?;
        // The readout cache schema is the deterministic `Mat::decompose`
        // bound, not the Gaussian helper cutoff `B_chi`.
        let (readout_artifacts, public_readout_output) =
            with_benchmark_role(BenchmarkRole::PublicReadout, || {
                let readout_artifacts = request
                    .program
                    .prepare_readout(
                        &scheme,
                        &public_outputs,
                        &request.public_prf.readout_preparation,
                    )
                    .map_err(|error| RefreshSetupError::SetupStage {
                        stage: "prepare public readout",
                        detail: error.to_string(),
                    })?;
                let public_readout_output = request
                    .program
                    .project_readout(
                        &scheme,
                        &public_outputs,
                        &readout_artifacts,
                        &request.public_prf.readout_public_masks,
                    )
                    .map_err(|error| RefreshSetupError::SetupStage {
                        stage: "prepare public readout",
                        detail: error.to_string(),
                    })?;
                Ok::<_, RefreshSetupError>((readout_artifacts, public_readout_output))
            })?;
        // `A'` is the public random matrix used in every slot target
        // `T_t = MS(A_{sum,t}) - Delta_t reduce(A')`; its tag binds this refresh.
        let a_prime = ring.hash_matrix(
            request.hash_key.clone(),
            HashTag::from(
                format!("mxx-exponent-lut/refresh/a-prime/v1/{}", hex(&p.refresh_id)).into_bytes(),
            ),
            (p.layout.secret_dimension, p.layout.public_key_columns()),
        );
        let (state_scaling, mask_route_rhs_decompositions, fresh_route_rhs_decompositions) =
            p.prepare_refresh_route_witnesses(public_readout_output.element_type())?;
        // Public routing and online routing consume the same preprocessed
        // decompositions, including digit weights and route corrections.
        let label_index =
            super::refresh::RefreshPrfLabelIndex::new(p.refresh_id, &p.transcript_topology())?;
        let (public_masks, public_fresh) = super::refresh::prepare_refresh_public_routes(
            &scheme,
            &label_index,
            public_readout_output,
            mask_route_rhs_decompositions.clone(),
            fresh_route_rhs_decompositions.clone(),
        )?;
        let mask_public_family = Family::pack(public_masks)?;
        let scaled_fresh_public_family = Family::pack(public_fresh)?;
        let sum_family = super::refresh::prepare_refresh_public_sums(
            &scheme,
            request.public_prf.initial.matrix.clone(),
            Family::pack(state_scaling.clone())?,
            mask_public_family,
            scaled_fresh_public_family,
        )?;
        let (public_b, refresh_targets, ks) =
            p.sample_refresh_decoders(&scheme, sum_family, &a_prime, &decoder_preimage_bound)?;
        let expected_readout_schema = p
            .refresh_readout_schema(refresh_prf_output_count(p)?)
            .map_err(|error| RefreshSetupError::SetupStage {
                stage: "prepare public readout",
                detail: error.to_string(),
            })?;
        if readout_artifacts.schema() != expected_readout_schema {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let prf_contract = RefreshPrfContract::from_program(&request.program);
        let readout_identity = request.program.readout.identity();
        let identity = setup_identity_digest(
            p,
            &prf_contract,
            request.layout.layout_id,
            &decoder_preimage_bound,
            readout_identity,
        );
        let names = RefreshPreprocessingArtifactNames::new(
            &identity,
            p.refresh.crt_plaintext_moduli.len(),
            public_b.len(),
            request.selector_gsw_name,
        );
        let (context, prepared_descriptor) = prepared_prf.export(
            &scheme,
            DslContext::new("mxx-exponent-lut-refresh-setup"),
            &format!("{}-fixed-prf", names.readout),
        )?;
        let declaration = RefreshPreprocessingDeclaration {
            identity,
            pbc_layout_id: request.layout.layout_id,
            prf_contract,
            names,
            prepared_prf: prepared_descriptor,
            readout_identity,
        };
        let names = &declaration.names;
        // Export the construction-owned values directly. Setup secret t and
        // sampled decoder witnesses stay private; online values are absent.
        let mut context = context
            .private_output(names.setup_secret_t.clone(), request.setup_secret_t)?
            .public_output(names.selector_gsw.clone(), request.selector_gsw)?
            .public_output(
                names.mask_route_rhs_decompositions.clone(),
                mask_route_rhs_decompositions,
            )?
            .public_output(
                names.fresh_route_rhs_decompositions.clone(),
                fresh_route_rhs_decompositions,
            )?
            .public_output(
                names.state_public_matrix.clone(),
                request.public_prf.initial.matrix.clone(),
            )?
            .public_output(names.a_prime.clone(), a_prime)?;
        for (name, matrix) in names.public_matrix_b.iter().zip(public_b) {
            context = context.public_output(name.clone(), matrix)?;
        }
        for (name, target) in names.refresh_targets.iter().zip(refresh_targets) {
            context = context.public_output(name.clone(), target)?;
        }
        for (name, decomposition) in names.state_scaling.iter().zip(state_scaling) {
            context = context.public_output(name.clone(), decomposition)?;
        }
        for (name, preimage) in names.preimages.iter().zip(&ks) {
            context = context.private_output(name.clone(), preimage.clone())?;
        }
        let context = context.public_output(names.readout.clone(), readout_artifacts.clone())?;
        let built = context.build()?;
        // Read the actual artifact schemas once, while construction handles
        // are available. Retain only compact plans after the graph is frozen.
        let expected_schema = compact_readout_plan_schema(p, &request.program)?;
        let actual_schema = producer_readout_plan_schema(p, &readout_artifacts, &ks)?;
        if expected_schema != actual_schema {
            return Err(RefreshSetupError::InvalidManifest);
        }
        let noise_plans =
            compact_noise_snapshot_plans(p, prepared_prf.program(), &[], &actual_schema)?;
        Ok(Self { parameters: request.parameters, noise_plans, built, declaration })
    }
    /// Returns the frozen preprocessing graph and its artifact outputs.
    pub fn built(&self) -> &BuiltGraph {
        &self.built
    }

    /// Returns the public declaration describing this producer's artifacts.
    pub fn declaration(
        &self,
    ) -> &RefreshPreprocessingDeclaration<
        <BggScheme as ExponentLUTFixedRhsScheme>::FixedProductBinding,
        <<BggScheme as ExponentLUTFixedRhsScheme>::PreparedFixedProduct as GraphValue>::Schema,
    > {
        &self.declaration
    }

    /// Applies the declaration's role/type checks to an exported manifest.
    pub fn finalize_export_manifest(
        &self,
        manifest: &mut Manifest,
    ) -> Result<(), RefreshSetupError> {
        self.parameters.validate_refresh_manifest(&self.declaration, manifest)
    }
}

fn refresh_prf_output_count(p: &BggRefreshParameters) -> Result<usize, RefreshSetupError> {
    p.transcript_topology().joint_event_count().map_err(|_| RefreshSetupError::InvalidManifest)
}

impl ImportedRefreshSetup {
    /// Returns the validated setup parameters used by this import.
    pub fn parameters(&self) -> &BggRefreshParameters {
        &self.parameters
    }

    /// Returns the declaration that was checked during import.
    pub fn declaration(
        &self,
    ) -> &RefreshPreprocessingDeclaration<
        <BggScheme as ExponentLUTFixedRhsScheme>::FixedProductBinding,
        <<BggScheme as ExponentLUTFixedRhsScheme>::PreparedFixedProduct as GraphValue>::Schema,
    > {
        &self.declaration
    }

    /// Returns the setup-fixed public state key paired with the online state
    /// vector supplied to the online construction stage.
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
        parameters: BggRefreshParameters,
        program: SparseLwrPrfProgram,
        declaration: RefreshPreprocessingDeclaration<
            <BggScheme as ExponentLUTFixedRhsScheme>::FixedProductBinding,
            <<BggScheme as ExponentLUTFixedRhsScheme>::PreparedFixedProduct as GraphValue>::Schema,
        >,
        manifest: &Manifest,
    ) -> Result<Self, RefreshSetupError> {
        parameters.validate_refresh_manifest(&declaration, manifest)?;
        if program.id() != declaration.prf_contract.program_id() {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let production_id = manifest.production_id.clone();
        let n = &declaration.names;
        // The adapter already validated every concrete field, bound and role.
        // Import that exact contract instead of reconstructing gadget dimensions.
        let matrix_type =
            |matrix: &mxx_ir_core::types::ConcreteMatrixType| mxx_ir_core::types::MatrixType {
                modulus: matrix.modulus.clone().into(),
                ring_dimension: matrix.ring_dimension.into(),
                rows: matrix.rows.into(),
                columns: matrix.columns.into(),
            };
        let imported = |name: &str| -> Result<Mat, RefreshSetupError> {
            let field = &manifest.artifacts[name];
            let ArtifactType::Matrix(matrix) = &field.artifact_type else {
                return Err(RefreshSetupError::InvalidManifest);
            };
            Ok(mxx_dsl::MatType(matrix_type(matrix)).artifact_input(
                production_id.clone(),
                name,
                field.confidentiality,
            )?)
        };
        let preimage_schema = |name: &str| -> Result<mxx_dsl::PreimageType, RefreshSetupError> {
            let ArtifactType::Preimage { matrix, max_coefficient_bound } =
                &manifest.artifacts[name].artifact_type
            else {
                return Err(RefreshSetupError::InvalidManifest);
            };
            Ok(mxx_dsl::PreimageType {
                matrix: matrix_type(matrix),
                max_coefficient_bound: max_coefficient_bound.clone().into(),
            })
        };
        let state_public_matrix = imported(&n.state_public_matrix)?;
        let setup_secret_t = imported(&n.setup_secret_t)?;
        let a_prime = imported(&n.a_prime)?;
        let public_b =
            n.public_matrix_b.iter().map(|name| imported(name)).collect::<Result<Vec<_>, _>>()?;
        let refresh_targets =
            n.refresh_targets.iter().map(|name| imported(name)).collect::<Result<Vec<_>, _>>()?;
        let state = BggPublicKeyWire { matrix: state_public_matrix, reveal_plaintext: false };
        // Preserve the bounded preimage type, so evaluation keeps mul_small_rhs.
        let ks = n
            .preimages
            .iter()
            .map(|name| {
                Ok(preimage_schema(name)?.artifact_input(
                    production_id.clone(),
                    name,
                    manifest.artifacts[name].confidentiality,
                )?)
            })
            .collect::<Result<Vec<_>, RefreshSetupError>>()?;
        let readout_artifacts = parameters
            .refresh_readout_schema(refresh_prf_output_count(&parameters)?)
            .map_err(|_| RefreshSetupError::InvalidManifest)?
            .artifact_input(
                production_id.clone(),
                n.readout.clone(),
                ArtifactConfidentiality::Public,
            )?;
        let import_cache_family = |name: &str| -> Result<Family<Preimage>, RefreshSetupError> {
            // validate_manifest established the exact positive count and
            // schema before any consumer handle is constructed.
            let count = manifest.artifacts[name].family_count.expect("validated family count");
            Ok(mxx_dsl::FamilyType {
                element: preimage_schema(name)?,
                count: count.into(),
                shared: vec![false],
            }
            .artifact_input(
                production_id.clone(),
                name,
                ArtifactConfidentiality::Public,
            )?)
        };
        let state_scaling = n
            .state_scaling
            .iter()
            .map(|name| {
                Ok(preimage_schema(name)?.artifact_input(
                    production_id.clone(),
                    name.clone(),
                    manifest.artifacts[name].confidentiality,
                )?)
            })
            .collect::<Result<Vec<_>, RefreshSetupError>>()?;
        let mask_route_rhs_decompositions = import_cache_family(&n.mask_route_rhs_decompositions)?;
        let fresh_route_rhs_decompositions =
            import_cache_family(&n.fresh_route_rhs_decompositions)?;
        let scheme = BggScheme::new(parameters.public_key_compiler(), program.profile().layout);
        let program_cache = PreparedSparseLwrPrf::import(
            &scheme,
            program,
            &declaration.prepared_prf,
            production_id,
            declaration.prepared_prf.identity()?,
        )?;
        Ok(Self {
            parameters,
            state,
            setup_secret_t,
            a_prime,
            public_b,
            refresh_targets,
            readout_artifacts,
            program_cache,
            state_scaling,
            mask_route_rhs_decompositions,
            fresh_route_rhs_decompositions,
            preimages: ks,
            declaration,
        })
    }
}

/// Derives all preprocessing artifact names from the public setup identity.
///
/// Slot-indexed names are emitted in CRT order, so an importer cannot silently
/// exchange two mask or preimage artifacts.
impl RefreshPreprocessingArtifactNames {
    /// Naming depends only on the committed setup identity and shared channel
    /// schedule. Cryptographic parameter binding belongs to the adapter.
    pub(crate) fn new(
        identity: &[u8; 32],
        slot_count: usize,
        group_count: usize,
        selector_gsw_name: String,
    ) -> Self {
        let prefix = hex(identity);
        let n = |r: &str, i: usize| format!("mxx-refresh-{prefix}-{r}-{i}");
        RefreshPreprocessingArtifactNames {
            setup_secret_t: n("setup-secret-t", 0),
            selector_gsw: selector_gsw_name,
            state_scaling: (0..slot_count).map(|i| n("state-scaling-k-rhs", i)).collect(),
            mask_route_rhs_decompositions: n("mask-route-k-rhs", 0),
            fresh_route_rhs_decompositions: n("fresh-route-k-rhs", 0),
            state_public_matrix: n("state-public", 0),
            a_prime: n("a-prime", 0),
            public_matrix_b: (0..group_count).map(|group| n("public-b", group)).collect(),
            refresh_targets: (0..slot_count).map(|i| n("refresh-target", i)).collect(),
            preimages: (0..slot_count).map(|i| n("preimage", i)).collect(),
            readout: n("readout", 0),
        }
    }
}

fn hex(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Checks concrete backend fields and typed PRF/readout ports using the same
/// transport rules. Backend parameters and dimension formulas stay outside.
pub(crate) fn validate_manifest<B, D: GraphValueSchema>(
    m: &Manifest,
    readout_name: &str,
    prepared_prf: &PreparedSparseLwrPrfDescriptor<B, D>,
    mut fields: Vec<(String, ArtifactConfidentiality, ArtifactType, Option<usize>)>,
    readout_schema: impl GraphValueSchema,
) -> Result<(), RefreshSetupError> {
    let artifact_ports = |prefix: String, ports: Vec<WireType>| {
        let arity = ports.len();
        ports.into_iter().enumerate().map(move |(port, wire_type)| {
            let name = if arity == 1 { prefix.clone() } else { format!("{prefix}.{port}") };
            (name, wire_type)
        })
    };
    let public_ports = artifact_ports(readout_name.to_owned(), readout_schema.wire_types()).chain(
        prepared_prf
            .artifacts()
            .flat_map(|(name, schema)| artifact_ports(name, schema.wire_types())),
    );
    for (name, wire_type) in public_ports {
        let concrete = mxx_ir_core::validate::concretize_wire_type(
            &wire_type,
            &ParamEnv::default(),
            &FrozenGraphScopeId::Root,
            NodeId(0),
        )
        .map_err(|_| RefreshSetupError::InvalidManifest)?;
        let (element, count) = match &concrete {
            mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, count } => {
                (element.as_ref(), Some(*count))
            }
            scalar => (scalar, None),
        };
        let expected =
            ArtifactType::from_wire_type(element).ok_or(RefreshSetupError::InvalidManifest)?;
        fields.push((name, ArtifactConfidentiality::Public, expected, count));
    }
    if m.artifacts.len() != fields.len() {
        return Err(RefreshSetupError::InvalidManifest);
    }
    for (name, confidentiality, expected, count) in fields {
        let artifact = m.artifacts.get(&name).ok_or(RefreshSetupError::InvalidManifest)?;
        if artifact.artifact_type != expected ||
            artifact.family_count != count ||
            artifact.confidentiality != confidentiality ||
            artifact.content_hash.is_some() !=
                (confidentiality == ArtifactConfidentiality::Public)
        {
            return Err(RefreshSetupError::InvalidManifest);
        }
    }
    mxx_ir_core::artifact::validate_manifest(m).map_err(|_| RefreshSetupError::InvalidManifest)?;
    Ok(())
}

/// Verification graph outputs and their exact threshold-decoder operands.
/// The backend supplies its encoding relation and scalar residual components;
/// this boundary connects them to decoders without changing the row carrier.
pub struct RefreshVerification<V: mxx_dsl::GraphValue> {
    encoding_vector: V,
    anchor: V,
    residual: V,
    /// The exact scalar matrix passed as the input to each threshold decoder,
    /// in public-key-column order.  Keeping these wires prevents consumers
    /// from reconstructing an equivalent-looking slice and accidentally
    /// binding the correctness checker to a different graph operand.
    decoder_residuals: Vec<Mat>,
    decoded: Vec<Bool>,
}

impl<V: mxx_dsl::GraphValue> RefreshVerification<V> {
    /// The backend constructs the relation residual and its scalar components
    /// together. The same component handles are retained and passed to the
    /// threshold decoders, so output linkage never reconstructs those operands.
    /// `decode_length` is positive under the setup construction contract.
    pub fn new(
        encoding_vector: V,
        anchor: V,
        residual: V,
        decoder_residuals: Vec<Mat>,
        plaintext_modulus: impl Into<IntExpr>,
        decode_length: usize,
    ) -> Result<Self, RefreshSetupError> {
        let decoded_count = decoder_residuals
            .len()
            .checked_mul(decode_length)
            .ok_or(RefreshSetupError::InvalidParameters("decoded output count overflow"))?;
        let plaintext_modulus = plaintext_modulus.into();
        let mut decoded = Vec::with_capacity(decoded_count);
        for scalar_residual in &decoder_residuals {
            decoded.extend(
                scalar_residual
                    .clone()
                    .threshold_decode_bools(plaintext_modulus.clone(), decode_length),
            );
        }
        Ok(Self { encoding_vector, anchor, residual, decoder_residuals, decoded })
    }

    /// The relation residual in the backend's original graph-value carrier.
    pub fn residual(&self) -> &V {
        &self.residual
    }

    /// Returns the exact threshold-decoder operands, one 1x1 matrix per
    /// residual component, in canonical order.
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
        let mut context = context
            .private_output("refresh-verification-vector", self.encoding_vector.clone())?
            .private_output("refresh-verification-anchor", self.anchor.clone())?
            .private_output(residual_name, self.residual.clone())?;
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
            context = context.output(format!("{decoded_prefix}_{index}"), value.clone())?;
        }
        Ok(context)
    }
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
/// The trusted key provider supplies the validated setup, sparse-LWR Program,
/// generated PBC layout, and key identity.  Sampling, selector construction,
/// preprocessing, refresh, and verification are intentionally kept inside the
/// consuming [`RefreshParameterSimulationBundle`] builder.
pub struct RefreshParameterSimulationRequest {
    setup: BggRefreshParameters,
    program: SparseLwrPrfProgram,
    generated_layout: crate::noise_refresh::pbc::PbcGeneratedKeyLayout,
    key_instance_id: [u8; 32],
    expected_plaintext: Mat,
    decode_length: usize,
}

impl RefreshParameterSimulationRequest {
    /// Creates a request after checking all cross-layer dimensions and the
    /// selected decoder cutoff against the official policy. Decoder sampling
    /// parameters come exclusively from `setup`; the request keeps that
    /// validated value private until graph construction consumes it.
    /// The sparse support and one-hot schedule remain private inside the generated PBC layout.
    /// Verification builds encode `expected_plaintext` as the initial state
    /// and independently use it in the output relation. Benchmark-only builds
    /// retain the fixed identity attribute and do not construct this oracle.
    pub fn new(
        setup: BggRefreshParameters,
        program: SparseLwrPrfProgram,
        generated_layout: crate::noise_refresh::pbc::PbcGeneratedKeyLayout,
        key_instance_id: [u8; 32],
        expected_plaintext: Mat,
        decode_length: usize,
    ) -> Result<Self, RefreshSetupError> {
        let profile = program.profile();
        setup.validate()?;
        if decode_length == 0 {
            return Err(RefreshSetupError::InvalidParameters("decode length must be positive"));
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
            program,
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
    /// as [`RefreshPreprocessingProducer`], but does not
    /// lower the label-major PRF family or construct executable graph stages.
    /// The full graph remains available through [`Self::build`] and
    /// [`Self::build_benchmark`] for the selected candidate and benchmark.
    pub fn build_noise_snapshot_only(self) -> Result<BggNoiseSnapshot, RefreshSetupError> {
        let layout = &self.generated_layout.public_layout;
        let active_cells = PbcActiveCellIndex::build(layout)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid PBC active-cell index"))?;
        self.setup.build_noise_snapshot_from_public_layout(
            &self.program,
            layout,
            &active_cells,
            None,
        )
    }

    fn build_with_mode(
        self,
        benchmark_only: bool,
    ) -> Result<RefreshParameterSimulationBundle, RefreshSetupError> {
        let program = self.program;
        let profile = program.profile();
        let setup = self.setup;
        // The request owns parameters and layout validated by its constructor.
        let layout = self.generated_layout.public_layout.clone();
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
        let selector_runtime_binding = selector_bits
            .runtime_binding(&ring)
            .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let selector_names =
            PbcSelectorArtifactNames::canonicalize_schema(&layout, self.key_instance_id)
                .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let selector_artifacts = pbc_selector_metadata(
            &layout,
            self.key_instance_id,
            &sampler.layout,
            self.key_instance_id,
            &sampler.layout,
            self.key_instance_id,
        )
        .and_then(|metadata| {
            PbcSelectorArtifacts::from_structural(
                &layout,
                self.key_instance_id,
                selector_names,
                metadata,
            )
        })
        .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let structural_gsw = sampler
            .sample_pbc_selectors(
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

        let active_count = crate::noise_refresh::pbc::PbcActiveCellIndex::build(&layout)
            .map_err(|error| RefreshSetupError::Pbc(error.to_string()))?
            .len();
        let initial_attribute = if benchmark_only {
            profile.layout.mono(&ring, &BigInt::zero()).map_err(|_| {
                RefreshSetupError::InvalidParameters("invalid native initial attribute")
            })?
        } else {
            self.expected_plaintext.clone()
        };
        let online_label_count = refresh_prf_output_count(&setup)?;
        let total_labels = online_label_count
            .checked_mul(active_count)
            .ok_or(RefreshSetupError::InvalidManifest)?;
        let public_prf = PrfPublicSetup::<BggScheme>::sample(
            &setup,
            &sampler,
            profile.native_width(),
            setup_secret.clone(),
            hash_key.clone(),
        )?;
        let compiler =
            ExponentLutEncodingCompiler::with_layout(setup.public_key_compiler(), profile.layout);
        let public_readout_masks = public_prf.readout_public_masks.clone();
        let producer = RefreshPreprocessingProducer::build(RefreshPreprocessingRequest {
            program: program.clone(),
            layout: layout.clone(),
            parameters: setup.clone(),
            public_prf,
            setup_secret_t: setup_secret.clone(),
            selector_gsw: structural_gsw,
            selector_gsw_name: selector_family_artifact_name(&selector_artifacts, "gsw"),
            label_plaintexts: ring.input_family(
                "refresh-parameter-search-native-monomials",
                total_labels,
                (1, 1),
            ),
            hash_key,
        })
        .map_err(|error| RefreshSetupError::SetupStage {
            stage: "construct preprocessing",
            detail: error.to_string(),
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
        selector_artifacts.finalize_export_manifest(&mut producer_manifest, "gsw").map_err(
            |error| RefreshSetupError::SetupStage {
                stage: "finalize selector manifest",
                detail: error.to_string(),
            },
        )?;
        let imported = ImportedRefreshSetup::import(
            setup.clone(),
            program.clone(),
            producer.declaration().clone(),
            &producer_manifest,
        )
        .map_err(|error| RefreshSetupError::SetupStage {
            stage: "import preprocessing manifest",
            detail: error.to_string(),
        })?;
        let mut verification_manifests = BTreeMap::new();
        verification_manifests.insert(producer_production.clone(), producer_manifest.clone());
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
        let refresh_compiler = setup.refresh.clone();
        let online_masks =
            sampler.sample_native_readout_masks(online_secret.clone(), &public_readout_masks)?;
        let scheme = BggScheme::new(setup.public_key_compiler(), compiler.exponent_layout);
        let accumulation = with_benchmark_role(BenchmarkRole::EncodingPrfAccumulation, || {
            imported.program_cache.evaluate(&scheme, ProgramValue::Scalar(initial.clone()))
        })
        .map_err(|error| RefreshSetupError::SetupStage {
            stage: "evaluate online fixed-RHS PRF",
            detail: error.to_string(),
        })?;
        let readout = program
            .evaluate_readout(&scheme, &accumulation, &imported.readout_artifacts, &online_masks)
            .map_err(|error| RefreshSetupError::SetupStage {
                stage: "evaluate online PRF readout",
                detail: error.to_string(),
            })?;
        let manifest = refresh_compiler
            .bind_online_outputs(
                &compiler,
                &imported,
                online_secret.clone(),
                initial,
                &program,
                &readout,
            )
            .map_err(|error| RefreshSetupError::SetupStage {
                stage: "bind online refresh inputs",
                detail: error.to_string(),
            })?;
        let refreshed = refresh_compiler.refresh(&compiler, &manifest).map_err(|error| {
            RefreshSetupError::SetupStage {
                stage: "apply online refresh",
                detail: error.to_string(),
            }
        })?;
        let benchmark_online_graph = DslContext::new("refresh-parameter-search-online-eval")
            .private_output("refreshed-encoding-vector", refreshed.encoding().vector.clone())?
            .build()?;
        validate_cache_required_online_graph(&benchmark_online_graph).map_err(|error| {
            RefreshSetupError::SetupStage {
                stage: "check online preparation boundary",
                detail: error.to_string(),
            }
        })?;
        let preprocessing_graph = BuiltGraph { graph: producer.built().graph.clone() };
        let benchmark_preprocessing_graph = BuiltGraph { graph: preprocessing_graph.graph.clone() };
        let public_identity = producer.declaration().identity;
        let (gain_plan, route_decompositions) = producer.noise_plans.clone();
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
                selector_graph: BuiltGraph { graph: producer.built().graph.clone() },
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
        let verification = crate::bgg::build_refresh_verification(
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
            selector_graph: BuiltGraph { graph: producer.built().graph.clone() },
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

/// The three immutable graph stages plus request-level external-input facts.
/// BuiltGraph and declaration validation occurs during bundle
/// construction. No schedules, secrets, or raw constructors are exposed, and
/// the embedded Exponent-LUT snapshot remains the operational-noise authority.
pub struct RefreshParameterSimulationBundle {
    selector_graph: BuiltGraph,
    pub(crate) preprocessing_graph: Option<BuiltGraph>,
    pub(crate) verification_graph: Option<BuiltGraph>,
    benchmark_preprocessing_graph: BuiltGraph,
    benchmark_online_graph: BuiltGraph,
    preprocessing_manifests: std::collections::BTreeMap<ProductionId, Manifest>,
    verification_manifests: std::collections::BTreeMap<ProductionId, Manifest>,
    public_identity: [u8; 32],
    selector_runtime_binding: PbcSelectorRuntimeBinding,
    metadata: std::collections::BTreeMap<String, RefreshSimulationMatrixInputMetadata>,
    targets: Vec<RefreshSimulationDecoderTarget>,
    pub(crate) noise_snapshot: BggNoiseSnapshot,
}

impl RefreshParameterSimulationBundle {
    /// Returns the frozen noise inputs used by this bundle's simulator.
    pub fn noise_snapshot(&self) -> &BggNoiseSnapshot {
        &self.noise_snapshot
    }

    /// Registers the source ring and every distinct selected decoder ring.
    /// Selection preserves the concrete CRT basis and, on GPUs, execution owner.
    pub fn ring_parameters<P: mxx_primitives::poly::PolyParams>(
        &self,
        source: P,
    ) -> Result<Vec<P>, RefreshSetupError> {
        let source_modulus: std::sync::Arc<BigUint> = source.modulus().into();
        if self.noise_snapshot.refresh.global_modulus.as_ref() != Some(source_modulus.as_ref()) {
            return Err(RefreshSetupError::IdentityMismatch);
        }
        let mut rings = vec![source.clone()];
        let selected = self
            .noise_snapshot
            .refresh
            .switch_moduli
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        for modulus in selected {
            if modulus != source_modulus.as_ref() {
                rings.push(source.select_modulus(modulus).ok_or(
                    RefreshSetupError::InvalidParameters(
                        "selected ring is not a source CRT subset",
                    ),
                )?);
            }
        }
        Ok(rings)
    }

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
    ) -> Result<
        ExponentLutNoiseReport<ProgramNoiseReport>,
        crate::noise_refresh::simulation::NoiseSimulationError,
    > {
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
    ) -> Result<
        ExponentLutAverageNoiseReport<ProgramNoiseReport>,
        crate::noise_refresh::simulation::NoiseSimulationError,
    > {
        if self.noise_snapshot.setup_identity() != &self.public_identity {
            return Err(crate::noise_refresh::simulation::NoiseSimulationError::InvalidRefresh(
                "average simulation snapshot identity mismatch",
            ));
        }
        self.noise_snapshot.simulate_average(config)
    }
}

/// Resolves and validates each frozen residual-to-decoder output link.
fn decoder_targets<V: mxx_dsl::GraphValue>(
    graph: &BuiltGraph,
    verification: &RefreshVerification<V>,
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
