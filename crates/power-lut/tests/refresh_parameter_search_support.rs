//! Small, deterministic helpers for the ignored Section-7 parameter search.
//!
//! The search is intentionally kept in the integration-test tree.  It is a
//! measurement harness, not a new Power-LUT parameter abstraction: production
//! code continues to use the ordinary `RefreshSetupParameters`, PBC, PRF
//! program, and correctness checker APIs.

use std::{
    collections::BTreeSet, fs, io::ErrorKind, ops::RangeInclusive, path::Path, process::Command,
};

#[path = "sparse_lwr_parameter_support.rs"]
mod sparse_lwr_parameter_support;

#[allow(unused_imports)]
pub use self::sparse_lwr_parameter_support::run_estimator;
use self::sparse_lwr_parameter_support::{
    REVIEWED_ESTIMATOR_COMMIT, SparseLwrParameterTuple, SparseLwrSecurityTier,
    reviewed_phase1_tuple_grid,
};
use mxx_bgg::{BggSamplerLayout, PreimageCoefficientBound};
use mxx_dsl::BuiltGraph;
use mxx_ir_core::{IntExpr, RealExpr, artifact::ProductionId, encoding::spec_hash};
use mxx_power_lut::{
    AverageCaseConfig,
    pbc::{PbcParameters, PbcRootSeed, clear_pbc_inner_product, generate_key_layout},
    prf::{SparseLwrPrfProfile, SparseLwrPrfProgram},
    refresh::RefreshCompiler,
    refresh_setup::{
        RefreshParameterSimulationBundle, RefreshParameterSimulationRequest, RefreshSetupParameters,
    },
};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use rand::{SeedableRng, rngs::StdRng, seq::index::sample};
use serde::Serialize;
use tracing::info;

const REVIEWED_ESTIMATOR_COST_MODEL: &str = "MATZOV";
const REVIEWED_ESTIMATOR_SHAPE_MODEL: &str = "gsa";
const PHASE1_PRIMARY_SECURITY_BITS: u64 = 100;
pub const NO_CANDIDATE_ERROR: &str =
    "no candidate in the declared finite grid passed security and correctness";

/// Preparation can reject one candidate for exact arithmetic infeasibility,
/// while malformed setup/configuration remains a fatal search error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CandidatePreparationError {
    Infeasible(String),
    Fatal(String),
}

impl From<String> for CandidatePreparationError {
    fn from(error: String) -> Self {
        Self::Fatal(error)
    }
}

impl std::fmt::Display for CandidatePreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infeasible(error) | Self::Fatal(error) => formatter.write_str(error),
        }
    }
}

/// Validates the frozen refresh stages and then runs the application-specific
/// Power-LUT noise simulator bound to the constructed setup.
///
/// Parameter-search candidates are intentionally not runtime executions.
/// `RefreshParameterSimulationRequest::build` validates the BuiltGraph,
/// declaration, and attestation boundaries. The mxx-noise-simulator request
/// only validates request-level stage/root/external facts; the embedded
/// Power-LUT snapshot is the operational-noise authority.
pub fn check_refresh_bundle(
    config: &SearchConfig,
    _candidate: Candidate,
    prepared: &PreparedCandidate,
) -> Result<bool, String> {
    let bundle = prepared.bundle.as_ref().ok_or_else(|| {
        "the refresh graph validator requires a built simulation bundle".to_owned()
    })?;
    let stage = |id: &str,
                 graph: &mxx_ir_core::Graph|
     -> Result<mxx_noise_simulator::SimulationStage, String> {
        let hash = spec_hash(graph, &mxx_ir_core::ParamEnv::default())
            .map_err(|error| error.to_string())?;
        Ok(mxx_noise_simulator::SimulationStage {
            id: mxx_noise_simulator::StageId(id.to_owned()),
            production_id: ProductionId { spec_hash: hash, execution_nonce: [0; 32] },
            graph: graph.clone(),
        })
    };
    let stage_graphs = [
        ("selector", bundle.selector_graph()),
        ("preprocessing", bundle.preprocessing_graph()),
        ("verification", bundle.verification_graph()),
    ];
    let roots = bundle
        .decoder_targets()
        .iter()
        .map(|target| mxx_noise_simulator::SimulationRoot {
            stage: mxx_noise_simulator::StageId("verification".to_owned()),
            output: target.residual_output_name.clone(),
        })
        .collect();
    let request = mxx_noise_simulator::SimulationRequest {
        program: mxx_noise_simulator::SimulationProgram {
            stages: stage_graphs
                .iter()
                .map(|(id, graph)| stage(id, &graph.graph))
                .collect::<Result<Vec<_>, _>>()?,
        },
        environment: mxx_ir_core::ParamEnv::default(),
        roots,
        external_inputs: simulation_external_inputs(&stage_graphs, bundle.matrix_input_metadata())?,
        limits: mxx_noise_simulator::SimulationLimits::default(),
    };
    request.validate().map_err(|error| format!("generic request validation: {error}"))?;
    if config.noise_model == NoiseSearchModel::AverageCase {
        let average_config = average_case_config(config)?;
        let average = bundle
            .simulate_average_noise(&average_config)
            .map_err(|error| format!("AverageCase simulation: {error}"))?;
        info!(
            mask_domain_accepted = average.refresh.domain_accepted,
            mask_smudging_accepted = average.refresh.mask_smudging_accepted,
            fresh_error_accepted = average.refresh.fresh_error_accepted,
            mask_bound_bits = average.refresh.mask_bound.bits(),
            average_rounding_accepted = average.correctness_accepted,
            average_smudging_accepted = average.refresh.mask_smudging_accepted,
            average_domain_accepted = average.refresh.domain_accepted,
            average_favg_bits = average.refresh.mask_smudging_max_favg.bits(),
            average_joint_event_count = ?average.refresh.joint_event_count,
            average_epsilon_numerator_bits = average.refresh.epsilon_joint.numerator.bits(),
            average_epsilon_denominator_bits = average.refresh.epsilon_joint.denominator.bits(),
            average_masking_distance_numerator_bits = average.refresh.masking_distance_bound.numerator.bits(),
            average_masking_distance_denominator_bits = average.refresh.masking_distance_bound.denominator.bits(),
            average_mask_smudging_margin_sign = signed_margin_sign(&average.refresh.mask_smudging_margin),
            average_mask_smudging_margin_bits = average.refresh.mask_smudging_margin.magnitude().bits(),
            average_domain_margin_sign = signed_margin_sign(&average.refresh.domain_margin),
            average_domain_margin_bits = average.refresh.domain_margin.magnitude().bits(),
            average_event_log2 = ?average.refresh.event_budget.log2_events(),
            average_security_authority = ?average.security_authority,
            average_correctness_authority = ?average.correctness_authority,
            average_hard_authority_accepted = average.hard_authority_accepted,
            average_correctness_accepted = average.correctness_accepted,
            average_accepted = average.accepted,
            "AverageCase noise result paired with WorstCase authority"
        );
        return Ok(average.hard_authority_accepted &&
            average.refresh.domain_accepted &&
            average.refresh.fresh_error_accepted &&
            average.refresh.mask_smudging_accepted &&
            average.refresh.rounding_accepted &&
            average.accepted);
    }
    let report = bundle.simulate_noise().map_err(|error| error.to_string())?;
    info!(
        q = report.prf.q_l,
        p = report.prf.p,
        bucket_count = report.prf.bucket_count,
        k = report.prf.k,
        lut_width = report.prf.lut_width,
        intermediate_groups = report.prf.intermediate_groups,
        terminal_start = report.prf.terminal_start_bucket,
        terminal_len = report.prf.terminal_bucket_len,
        "sparse-PRF grouped plan"
    );
    for bucket in &report.prf.bucket_stages {
        info!(
            bucket = bucket.bucket,
            active_count = bucket.active_count,
            input_bits = bucket.input_bound.bits(),
            gamma_selector_bits = bucket.gamma_selector.bits(),
            one_hot_output_bits = bucket.one_hot_output_bound.bits(),
            one_hot_additive_bits = bucket.one_hot_additive_bound.bits(),
            one_hot_bit_growth = bucket.one_hot_bit_growth,
            gamma_c_bits = bucket.gamma_c.bits(),
            gamma_a_bits = bucket.gamma_a.bits(),
            selection_inherited_bits = bucket.selection_inherited_bits,
            selection_additive_bits = bucket.selection_additive_bits,
            lut_output_bits = bucket.lut_output_bound.bits(),
            lut_additive_bits = bucket.lut_additive_bound.bits(),
            lut_bit_growth = bucket.lut_bit_growth,
            "sparse-PRF bucket noise stages"
        );
    }
    for group in &report.prf.group_stages {
        info!(
            group = group.group,
            start_bucket = group.start_bucket,
            bucket_len = group.bucket_len,
            lut_width = group.lut_width,
            input_bits = group.input_bound.bits(),
            unreduced_bits = group.unreduced_bound.bits(),
            inherited_bits = group.inherited_bound.bits(),
            base_helper_additive_bits = group.base_helper_additive.bits(),
            gamma_a_additive_bits = group.gamma_a_additive.bits(),
            additive_bits = group.additive_bound.bits(),
            output_bits = group.output_bits,
            bit_growth = group.bit_growth,
            gamma_c_bits = group.gamma_c.bits(),
            gamma_a_bits = group.gamma_a.bits(),
            "sparse-PRF grouped reduction noise stage"
        );
    }
    info!(
        terminal_lut_width = report.prf.terminal_lut_width,
        terminal_input_bits = report.prf.terminal_input_bound.bits(),
        terminal_output_bits = report.prf.terminal_output_bound.bits(),
        terminal_additive_bits = report.prf.terminal_additive_bound.bits(),
        terminal_bit_growth = report.prf.terminal_bit_growth,
        terminal_range_start = report.prf.terminal_start_bucket,
        terminal_range_len = report.prf.terminal_bucket_len,
        terminal_W = report.prf.terminal_lut_width,
        terminal_gamma_c_bits = report.prf.terminal_gamma_c.bits(),
        terminal_gamma_a_bits = report.prf.terminal_gamma_a.bits(),
        terminal_inherited_bits = report.prf.terminal_inherited_bits,
        terminal_base_helper_additive_bits = report.prf.terminal_base_helper_additive_bits,
        terminal_gamma_a_additive_bits = report.prf.terminal_gamma_a_additive_bits,
        "sparse-PRF terminal LUT noise stage"
    );
    Ok(report.refresh.accepted)
}

fn simulation_external_inputs(
    stage_graphs: &[(&str, &BuiltGraph)],
    metadata: &std::collections::BTreeMap<
        String,
        mxx_power_lut::refresh_setup::RefreshSimulationMatrixInputMetadata,
    >,
) -> Result<Vec<mxx_noise_simulator::ExternalInputFact>, String> {
    let mut facts = Vec::new();
    for (stage_name, built) in stage_graphs {
        let stage = mxx_noise_simulator::StageId((*stage_name).to_owned());
        for node in built.graph.root_scope().nodes() {
            let mxx_ir_core::node::NodeKind::Input { name, wire_type, artifact: None } =
                node.kind()
            else {
                continue;
            };
            let value = match wire_type {
                mxx_ir_core::types::WireType::Bytes { .. } => {
                    mxx_noise_simulator::ExternalInputValue::Bytes
                }
                mxx_ir_core::types::WireType::Matrix(matrix) => {
                    matrix_fact(name, matrix, metadata)?
                }
                mxx_ir_core::types::WireType::Family { element, shape } => {
                    let shape = shape
                        .iter()
                        .map(|entry| {
                            entry
                                .evaluate(&mxx_ir_core::ParamEnv::default())
                                .map_err(|error| error.to_string())?
                                .to_usize()
                                .ok_or_else(|| "family shape is not usize".to_owned())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let mxx_ir_core::types::WireType::Matrix(matrix) = element.as_ref() else {
                        return Err(format!("unsupported external family input {name}"));
                    };
                    mxx_noise_simulator::ExternalInputValue::Family {
                        shape,
                        element: Box::new(matrix_fact(name, matrix, metadata)?),
                    }
                }
                other => return Err(format!("unsupported external input {name}: {other:?}")),
            };
            facts.push(mxx_noise_simulator::ExternalInputFact {
                stage: stage.clone(),
                input: name.clone(),
                value,
            });
        }
    }
    let mut unique = BTreeSet::new();
    if facts.iter().any(|fact| !unique.insert((fact.stage.clone(), fact.input.clone()))) {
        return Err("duplicate external input fact".to_owned());
    }
    Ok(facts)
}

fn matrix_fact(
    name: &str,
    _matrix: &mxx_ir_core::types::MatrixType,
    metadata: &std::collections::BTreeMap<
        String,
        mxx_power_lut::refresh_setup::RefreshSimulationMatrixInputMetadata,
    >,
) -> Result<mxx_noise_simulator::ExternalInputValue, String> {
    let bound = metadata
        .get(name)
        .and_then(|facts| facts.canonical_coefficient_exclusive_upper_bound.as_ref())
        .map(|bound| bound.evaluate(&mxx_ir_core::ParamEnv::default()))
        .transpose()
        .map_err(|error| error.to_string())?
        .and_then(|bound| bound.to_biguint())
        .unwrap_or_else(|| BigUint::one());
    let maximum = if bound.is_zero() { BigUint::zero() } else { &bound - BigUint::one() };
    let is_constant = metadata.get(name).is_some_and(|facts| facts.is_constant_polynomial);
    Ok(mxx_noise_simulator::ExternalInputValue::Matrix {
        maximum_absolute_coefficient_error: BigUint::zero(),
        maximum_absolute_coefficient_value: Some(maximum),
        is_constant_polynomial: is_constant,
    })
}

/// The fixed profile requested for the first Power-LUT refresh search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Test-harness noise authority; production defaults to WorstCase.
    pub noise_model: NoiseSearchModel,
    /// Inclusive CRT-depth range. The outer search order is ascending depth.
    pub crt_depths: RangeInclusive<usize>,
    /// Inclusive log2(ring-dimension) range. The inner order is ascending.
    pub log_ring_dimensions: RangeInclusive<usize>,
    pub security_bits: u64,
    /// Statistical security target for the joint mask transcript.
    pub mask_statistical_security_bits: usize,
    pub crt_bits: usize,
    pub base_bits: u32,
    /// Ordered Phase-2 gadget parameter grid.  For each CRT-prime width the
    /// base widths are listed from largest to smallest, as required by the
    /// ell-column minimization policy.
    pub crt_base_bits_grid: Vec<(usize, u32)>,
    pub secret_dimension: usize,
    pub error_sigma: f64,
    pub decoder_sigma: f64,
    pub sparse_lwr_universe: usize,
    pub sparse_lwr_weight: usize,
    /// Explicit ascending universe-size grid used by Phase 1.  The selected
    /// value is frozen before any DCRT candidate is prepared.
    pub sparse_lwr_universe_grid: Vec<usize>,
    pub sparse_lwr_modulus: usize,
    pub sparse_lwr_output_modulus: usize,
    /// Explicit ordered Phase-1 tuple grid.  This is the source of truth for
    /// sparse-LWR selection; the scalar fields above are the selected tuple
    /// projected into Phase 2.
    pub sparse_lwr_phase1_grid: Vec<SparseLwrParameterTuple>,
    pub phase1_fallback_security_bits: u64,
    pub phase1_estimator_commit: String,
    pub phase1_estimator_cost_model: String,
    pub phase1_estimator_shape_model: String,
    pub lut_width: usize,
    pub pbc_max_attempts: u32,
    pub one_nontrivial_refresh_round: bool,
    pub plaintext_modulus: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum NoiseSearchModel {
    WorstCase,
    AverageCase,
}

impl SearchConfig {
    /// Returns the reviewed finite grid. Phase 1 uses the three ordered
    /// p=2 tuple points selected for the integration search. Phase 2 is
    /// searched through the largest valid depth for the widest practical CRT
    /// primes (`62 * 32 < 2000`) and ring dimensions `2^16..=2^17`. Gadget
    /// bases are considered globally in descending order, with CRT width as a
    /// descending tie-break.
    pub fn reviewed() -> Self {
        Self {
            noise_model: NoiseSearchModel::WorstCase,
            crt_depths: 30..=62,
            log_ring_dimensions: 16..=17,
            security_bits: 100,
            mask_statistical_security_bits: 100,
            crt_bits: 32,
            base_bits: 16,
            crt_base_bits_grid: descending_crt_base_bits_grid(32),
            secret_dimension: 2,
            error_sigma: 4.0,
            decoder_sigma: 4.578,
            sparse_lwr_universe: 451,
            sparse_lwr_weight: 31,
            sparse_lwr_universe_grid: vec![451],
            sparse_lwr_modulus: 16,
            sparse_lwr_output_modulus: 2,
            sparse_lwr_phase1_grid: reviewed_phase1_tuple_grid(),
            phase1_fallback_security_bits: 100,
            phase1_estimator_commit: REVIEWED_ESTIMATOR_COMMIT.to_owned(),
            phase1_estimator_cost_model: REVIEWED_ESTIMATOR_COST_MODEL.to_owned(),
            phase1_estimator_shape_model: REVIEWED_ESTIMATOR_SHAPE_MODEL.to_owned(),
            lut_width: 512,
            pbc_max_attempts: 128,
            one_nontrivial_refresh_round: true,
            plaintext_modulus: 2,
        }
    }

    /// Reads the optional test-only parameter overlay from the process
    /// environment.  The canonical reviewed profile remains the fallback for
    /// every unset variable; production code does not consume this harness
    /// configuration.
    pub fn from_env() -> Result<Self, String> {
        Self::from_lookup(|name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{name} must contain valid UTF-8"))
            }
        })
    }

    /// Applies an environment-like lookup without reading process-global
    /// state.  Keeping the lookup injectable makes parser tests deterministic
    /// and race-free while preserving the exact environment schema used by
    /// the ignored simulation: security, sparse-LWR grid/weight/moduli, LUT
    /// width, CRT/base bits, and MIN/MAX CRT-depth and ring-dimension bounds.
    /// Every schema name uses the `MXX_POWER_LUT_REFRESH_` prefix.
    pub fn from_lookup<L>(lookup: L) -> Result<Self, String>
    where
        L: Fn(&str) -> Result<Option<String>, String>,
    {
        let mut config = Self::reviewed();
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_SECURITY_BITS")? {
            config.security_bits = parse_lookup_u64("MXX_POWER_LUT_REFRESH_SECURITY_BITS", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_NOISE_MODEL")? {
            config.noise_model = match value.as_str() {
                "worst-case" => NoiseSearchModel::WorstCase,
                "average-case" => NoiseSearchModel::AverageCase,
                _ => {
                    return Err(
                        "MXX_POWER_LUT_REFRESH_NOISE_MODEL must be worst-case or average-case"
                            .to_owned(),
                    )
                }
            };
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS")? {
            config.mask_statistical_security_bits =
                parse_lookup_usize("MXX_POWER_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_SPARSE_LWR_UNIVERSE_GRID")? {
            config.sparse_lwr_universe_grid =
                parse_lookup_grid("MXX_POWER_LUT_REFRESH_SPARSE_LWR_UNIVERSE_GRID", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_SPARSE_LWR_WEIGHT")? {
            config.sparse_lwr_weight =
                parse_lookup_usize("MXX_POWER_LUT_REFRESH_SPARSE_LWR_WEIGHT", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_SPARSE_LWR_MODULUS")? {
            config.sparse_lwr_modulus =
                parse_lookup_usize("MXX_POWER_LUT_REFRESH_SPARSE_LWR_MODULUS", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS")? {
            config.sparse_lwr_output_modulus =
                parse_lookup_usize("MXX_POWER_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_LUT_WIDTH")? {
            config.lut_width = parse_lookup_usize("MXX_POWER_LUT_REFRESH_LUT_WIDTH", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_CRT_BITS")? {
            config.crt_bits = parse_lookup_usize("MXX_POWER_LUT_REFRESH_CRT_BITS", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_BASE_BITS")? {
            config.base_bits = parse_lookup_u32("MXX_POWER_LUT_REFRESH_BASE_BITS", &value)?;
        }
        config.crt_base_bits_grid = descending_crt_base_bits_grid(config.crt_bits);
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_MIN_CRT_DEPTH")? {
            config.crt_depths =
                parse_lookup_usize("MXX_POWER_LUT_REFRESH_MIN_CRT_DEPTH", &value)?..=
                    *config.crt_depths.end();
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_MAX_CRT_DEPTH")? {
            config.crt_depths = *config.crt_depths.start()..=
                parse_lookup_usize("MXX_POWER_LUT_REFRESH_MAX_CRT_DEPTH", &value)?;
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_MIN_LOG_RING_DIMENSION")? {
            config.log_ring_dimensions =
                parse_lookup_usize("MXX_POWER_LUT_REFRESH_MIN_LOG_RING_DIMENSION", &value)?..=
                    *config.log_ring_dimensions.end();
        }
        if let Some(value) = lookup("MXX_POWER_LUT_REFRESH_MAX_LOG_RING_DIMENSION")? {
            config.log_ring_dimensions = *config.log_ring_dimensions.start()..=
                parse_lookup_usize("MXX_POWER_LUT_REFRESH_MAX_LOG_RING_DIMENSION", &value)?;
        }
        config.validate()
    }

    fn validate(&self) -> Result<Self, String> {
        let crt_start = *self.crt_depths.start();
        let crt_end = *self.crt_depths.end();
        if crt_start == 0 || crt_start > crt_end {
            return Err("CRT depth range must be positive and ascending".to_owned());
        }
        let log_ring_start = *self.log_ring_dimensions.start();
        let log_ring_end = *self.log_ring_dimensions.end();
        if log_ring_start > log_ring_end ||
            log_ring_end >= usize::BITS as usize ||
            log_ring_end > 17
        {
            return Err(
                "log ring-dimension range must be ascending, shiftable, and bounded by N <= 2^17"
                    .to_owned(),
            );
        }
        if self.security_bits == 0 || self.crt_bits == 0 || self.base_bits == 0 {
            return Err("security, CRT, and base bit widths must be positive".to_owned());
        }
        if self.crt_bits > 32 || self.base_bits > (self.crt_bits / 2) as u32 {
            return Err("CRT/base bit widths must satisfy base_bits <= floor(crt_bits/2) and crt_bits <= 32".to_owned());
        }
        if self.crt_base_bits_grid.is_empty() ||
            self.crt_base_bits_grid.iter().any(|(crt_bits, base_bits)| {
                *crt_bits == 0 ||
                    *crt_bits > 32 ||
                    *base_bits == 0 ||
                    *base_bits > (*crt_bits / 2) as u32
            }) ||
            self.crt_base_bits_grid.windows(2).any(|window| {
                window[0].1 < window[1].1 ||
                    (window[0].1 == window[1].1 && window[0].0 <= window[1].0)
            })
        {
            return Err("CRT/base bit grid must be ordered by CRT width and descending base width"
                .to_owned());
        }
        if self.mask_statistical_security_bits == 0 {
            return Err("mask statistical security must be positive".to_owned());
        }
        if self.sparse_lwr_universe_grid.is_empty() ||
            self.sparse_lwr_universe_grid.iter().any(|&universe| universe == 0) ||
            self.sparse_lwr_universe_grid.windows(2).any(|window| window[0] >= window[1])
        {
            return Err(
                "SPARSE_LWR_UNIVERSE_GRID must be positive and strictly ascending".to_owned()
            );
        }
        if self.sparse_lwr_weight == 0 ||
            self.sparse_lwr_universe_grid
                .iter()
                .any(|&universe| self.sparse_lwr_weight > universe)
        {
            return Err("SPARSE_LWR_WEIGHT must be positive and fit every grid universe".to_owned());
        }
        if self.sparse_lwr_phase1_grid.is_empty() {
            return Err("Phase-1 sparse-LWR tuple grid must not be empty".to_owned());
        }
        for tuple in &self.sparse_lwr_phase1_grid {
            if tuple.p != 2 || tuple.q_l == 0 || tuple.q_l % 2 != 0 {
                return Err("Phase-1 tuple grid requires p == 2 and even Q_L".to_owned());
            }
            if !tuple.estimator_minimum_classical_bits.is_finite() ||
                tuple.estimator_minimum_classical_bits < 0.0 ||
                tuple.estimator_minimum_classical_bits.floor() as u64 !=
                    tuple.estimator_security_bits
            {
                return Err(
                    "Phase-1 tuple estimator minimum must be finite and match its conservative floor"
                        .to_owned(),
                );
            }
            tuple.candidate().map_err(|error| format!("invalid Phase-1 tuple: {error}"))?;
        }
        if self.sparse_lwr_phase1_grid.windows(2).any(|window| {
            window[0].q_l > window[1].q_l ||
                (window[0].q_l == window[1].q_l && window[0].nu >= window[1].nu)
        }) {
            return Err("Phase-1 tuple grid must be in reviewed ascending order".to_owned());
        }
        if self.phase1_fallback_security_bits != 100 {
            return Err("Phase-1 fallback tier must be exactly 100 bits".to_owned());
        }
        if self.phase1_estimator_commit.len() != 40 ||
            !self.phase1_estimator_commit.bytes().all(|byte| byte.is_ascii_hexdigit()) ||
            self.phase1_estimator_cost_model.is_empty() ||
            self.phase1_estimator_shape_model.is_empty()
        {
            return Err("Phase-1 estimator identity is incomplete".to_owned());
        }
        sparse_lwr_error_bounds(self.sparse_lwr_modulus, self.sparse_lwr_output_modulus)?;
        if self.sparse_lwr_output_modulus != 2 || self.sparse_lwr_modulus % 2 != 0 {
            return Err("selected sparse-LWR profile requires p == 2 and even Q_L".to_owned());
        }
        let minimum_ring_dimension = 1usize
            .checked_shl(log_ring_start as u32)
            .ok_or_else(|| "minimum log ring dimension shift overflows usize".to_owned())?;
        if self.lut_width == 0 ||
            !self.lut_width.is_power_of_two() ||
            self.lut_width > minimum_ring_dimension ||
            minimum_ring_dimension % self.lut_width != 0 ||
            self.sparse_lwr_modulus > self.lut_width
        {
            return Err("LUT_WIDTH must be a power of two compatible with the minimum ring and Q_L"
                .to_owned());
        }
        Ok(self.clone())
    }
}

fn parse_lookup_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|error| format!("{name} must be an unsigned integer: {error}"))
}

fn parse_lookup_u32(name: &str, value: &str) -> Result<u32, String> {
    value
        .trim()
        .parse::<u32>()
        .map_err(|error| format!("{name} must be an unsigned 32-bit integer: {error}"))
}

fn parse_lookup_u64(name: &str, value: &str) -> Result<u64, String> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("{name} must be an unsigned 64-bit integer: {error}"))
}

fn parse_lookup_grid(name: &str, value: &str) -> Result<Vec<usize>, String> {
    value.split(',').map(|entry| parse_lookup_usize(name, entry)).collect()
}

/// Returns the reviewed CRT/base search order: base widths descend globally,
/// with CRT widths descending as the tie-break. The reviewed practical widths
/// are 32, 30, and 28; a smaller configured maximum is represented directly
/// for tiny parser/test profiles.
pub fn descending_crt_base_bits_grid(max_crt_bits: usize) -> Vec<(usize, u32)> {
    let mut widths = [32usize, 30, 28]
        .into_iter()
        .filter(|&crt_bits| crt_bits <= max_crt_bits)
        .collect::<Vec<_>>();
    if widths.is_empty() && max_crt_bits != 0 {
        widths.push(max_crt_bits);
    }
    let max_base_bits = widths.iter().map(|&crt_bits| crt_bits / 2).max().unwrap_or(0);
    (1..=max_base_bits as u32)
        .rev()
        .flat_map(|base_bits| {
            widths
                .iter()
                .copied()
                .filter(move |&crt_bits| base_bits <= (crt_bits / 2) as u32)
                .map(move |crt_bits| (crt_bits, base_bits))
        })
        .collect()
}

/// One point in the finite search grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
pub struct Candidate {
    pub crt_depth: usize,
    pub log_ring_dimension: usize,
    pub crt_bits: usize,
    pub base_bits: u32,
}

/// The public, non-secret facts retained for one prepared candidate.
pub struct PreparedCandidate {
    pub candidate: Candidate,
    pub ring_dimension: usize,
    pub bucket_width: usize,
    pub official_preimage_bound: BigInt,
    pub layout_id: [u8; 32],
    pub program_id: [u8; 32],
    /// One-based accepted PBC layout retry count (public diagnostics only).
    pub pbc_attempts_used: u32,
    /// Present for the production search; omitted by mocked ordering tests.
    pub bundle: Option<RefreshParameterSimulationBundle>,
}

/// Security values and application-specific noise result for the selected point.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SearchResult {
    pub candidate: Candidate,
    pub achieved_security_bits: u64,
    pub bgg_rlwe_security_bits: u64,
    pub sparse_lwr_security_bits: u64,
    pub raw_key_entropy_bits: f64,
    pub sparse_lwr_universe: usize,
    pub sparse_lwr_weight: usize,
    pub official_preimage_bound: String,
    pub ring_dimension: usize,
    pub bucket_width: usize,
    pub pbc_attempts_used: u32,
    pub layout_id: String,
    pub program_id: String,
    pub checker_accepted: bool,
    pub sparse_lwr_q_l: usize,
    pub sparse_lwr_p: usize,
    pub sparse_lwr_tier: SparseLwrSecurityTier,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
}

/// One aggregate, non-secret row persisted for a Phase-1 universe that was
/// actually evaluated.  It records enough evidence to audit the first
/// qualifying grid point without exposing the sparse support or schedule.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct SparseLwrEvaluation {
    pub q_l: usize,
    pub p: usize,
    pub universe: usize,
    pub weight: usize,
    pub tier: SparseLwrSecurityTier,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
    pub error_lower: i64,
    pub error_upper: i64,
    pub sparse_lwr_security_bits: u64,
    pub estimator_security_bits: u64,
    pub raw_key_entropy_bits: f64,
    pub minimum_security_bits: u64,
    pub qualified: bool,
}

/// The Phase-1 result that is reused unchanged by every DCRT candidate.
///
/// Keeping the estimator outputs with the selected universe size makes the
/// minimality claim auditable: Phase 2 cannot accidentally rerun the sparse
/// estimator with a candidate-dependent value.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct SelectedSparseLwrProfile {
    pub tuple: SparseLwrParameterTuple,
    pub parameter_grid: Vec<SparseLwrParameterTuple>,
    pub q_l: usize,
    pub p: usize,
    pub universe: usize,
    pub weight: usize,
    pub tier: SparseLwrSecurityTier,
    pub error_lower: i64,
    pub error_upper: i64,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
    pub sparse_lwr_security_bits: u64,
    pub raw_key_entropy_bits: f64,
    pub evaluations: Vec<SparseLwrEvaluation>,
}

/// Environment variable naming an explicit Phase-1 checkpoint file.
///
/// A checkpoint is never discovered implicitly.  When this variable is
/// absent, the test performs a fresh Phase-1 search and keeps the result only
/// in memory.  When it names a missing file, the fresh result is written to
/// that exact path before Phase 2 starts.  An existing file is reused only
/// after its declaration has an exact match with the current search model.
pub const PHASE1_CHECKPOINT_ENV: &str = "MXX_POWER_LUT_REFRESH_PHASE1_CHECKPOINT";

const PHASE1_CHECKPOINT_VERSION: u32 = 1;

/// The public declaration against which a persisted Phase-1 result is checked.
///
/// These fields describe both the input grid and the security model used to
/// score each row.  They intentionally contain no sparse support, schedule,
/// key material, or protocol/IR value.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct Phase1Declaration {
    pub parameter_grid: Vec<SparseLwrParameterTuple>,
    pub selected: SparseLwrParameterTuple,
    pub raw_key_entropy_bits: Vec<f64>,
    pub security_target_bits: u64,
    pub sparse_secret_model: String,
    pub sparse_error_model: String,
    pub error_intervals: Vec<(usize, usize, i64, i64)>,
    pub fallback_security_target_bits: u64,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
    pub exact_estimator: bool,
    pub crt_base_bits_grid: Vec<(usize, u32)>,
    pub phase2_search_order: String,
}

impl Phase1Declaration {
    /// Derive the complete public declaration from the current test config.
    pub fn from_config(
        config: &SearchConfig,
        selected: &SelectedSparseLwrProfile,
    ) -> Result<Self, String> {
        Ok(Self {
            parameter_grid: config.sparse_lwr_phase1_grid.clone(),
            selected: selected.tuple.clone(),
            raw_key_entropy_bits: config
                .sparse_lwr_phase1_grid
                .iter()
                .map(|tuple| raw_key_entropy_bits(tuple.nu, tuple.h))
                .collect(),
            security_target_bits: PHASE1_PRIMARY_SECURITY_BITS,
            sparse_secret_model: "SparseBinary".to_owned(),
            sparse_error_model: "Uniform".to_owned(),
            error_intervals: config
                .sparse_lwr_phase1_grid
                .iter()
                .map(|tuple| {
                    let (lower, upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p)?;
                    Ok((tuple.q_l, tuple.p, lower, upper))
                })
                .collect::<Result<Vec<_>, String>>()?,
            fallback_security_target_bits: config.phase1_fallback_security_bits,
            estimator_commit: config.phase1_estimator_commit.clone(),
            estimator_cost_model: config.phase1_estimator_cost_model.clone(),
            estimator_shape_model: config.phase1_estimator_shape_model.clone(),
            exact_estimator: true,
            crt_base_bits_grid: config.crt_base_bits_grid.clone(),
            phase2_search_order: "sparse_profile_then_security_correctness_qbits_n_then_largest_base_bits; base_bits_descending_then_crt_bits_descending_then_crt_depth_then_log_ring_dimension".to_owned(),
        })
    }
}

/// Public data written immediately after Phase 1 and before any DCRT work.
///
/// The evaluated rows are duplicated in `selected.evaluations` because the
/// selected profile is also the value consumed by Phase 2.  Keeping both
/// named fields makes the checkpoint self-describing and lets validation
/// reject a partially or manually edited checkpoint without reconstructing a
/// security estimate.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct Phase1Checkpoint {
    pub version: u32,
    pub declaration: Phase1Declaration,
    pub evaluated: Vec<SparseLwrEvaluation>,
    pub selected: SelectedSparseLwrProfile,
    pub accepted_phase2: Option<AcceptedPhase2Profile>,
}

#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct AcceptedPhase2Profile {
    pub tuple: SparseLwrParameterTuple,
    pub candidate: Candidate,
}

impl Phase1Checkpoint {
    pub fn from_selection(
        config: &SearchConfig,
        selected: &SelectedSparseLwrProfile,
    ) -> Result<Self, String> {
        let declaration = Phase1Declaration::from_config(config, selected)?;
        let checkpoint = Self {
            version: PHASE1_CHECKPOINT_VERSION,
            declaration,
            evaluated: selected.evaluations.clone(),
            selected: selected.clone(),
            accepted_phase2: None,
        };
        checkpoint.validate(config)?;
        Ok(checkpoint)
    }

    /// Validate an on-disk checkpoint before allowing it to skip Phase 1.
    pub fn validate(&self, config: &SearchConfig) -> Result<(), String> {
        if self.version != PHASE1_CHECKPOINT_VERSION {
            return Err(format!("unsupported Phase-1 checkpoint version {}", self.version));
        }
        let expected = Phase1Declaration::from_config(config, &self.selected)?;
        if self.declaration != expected {
            return Err(
                "Phase-1 checkpoint declaration does not match the current search model".to_owned()
            );
        }
        if self.evaluated != self.selected.evaluations {
            return Err(
                "Phase-1 checkpoint evaluated rows do not match the selected profile".to_owned()
            );
        }
        if let Some(accepted) = &self.accepted_phase2 {
            let row = self
                .evaluated
                .iter()
                .find(|row| {
                    row.q_l == accepted.tuple.q_l &&
                        row.p == accepted.tuple.p &&
                        row.universe == accepted.tuple.nu &&
                        row.weight == accepted.tuple.h
                })
                .ok_or_else(|| "accepted Phase-2 profile is not an evaluated tuple".to_owned())?;
            if accepted.tuple !=
                self.selected
                    .parameter_grid
                    .iter()
                    .find(|tuple| **tuple == accepted.tuple)
                    .cloned()
                    .ok_or_else(|| "accepted Phase-2 tuple is not declared".to_owned())? ||
                !row.qualified ||
                row.tier != self.selected.tier
            {
                return Err(
                    "accepted Phase-2 profile is not a qualified selected-tier row".to_owned()
                );
            }
            if !candidates(config).any(|candidate| candidate == accepted.candidate) {
                return Err(
                    "accepted Phase-2 candidate is outside the declared search grid".to_owned()
                );
            }
        }
        if self.evaluated.len() != config.sparse_lwr_phase1_grid.len() {
            return Err("Phase-1 checkpoint must exhaust the declared tuple grid".to_owned());
        }
        for (index, row) in self.evaluated.iter().enumerate() {
            let tuple = &config.sparse_lwr_phase1_grid[index];
            let (error_lower, error_upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p)?;
            let expected_tier = if row.minimum_security_bits >= PHASE1_PRIMARY_SECURITY_BITS {
                SparseLwrSecurityTier::Primary100
            } else {
                SparseLwrSecurityTier::Fallback100
            };
            if row.q_l != tuple.q_l ||
                row.p != tuple.p ||
                row.universe != tuple.nu ||
                row.weight != tuple.h ||
                row.tier != expected_tier ||
                row.estimator_commit != config.phase1_estimator_commit ||
                row.estimator_cost_model != config.phase1_estimator_cost_model ||
                row.estimator_shape_model != config.phase1_estimator_shape_model ||
                (row.error_lower, row.error_upper) != (error_lower, error_upper) ||
                row.estimator_security_bits != tuple.estimator_security_bits ||
                row.sparse_lwr_security_bits != row.estimator_security_bits ||
                (row.raw_key_entropy_bits - raw_key_entropy_bits(row.universe, row.weight)).abs() >
                    1e-9 ||
                row.qualified !=
                    (row.minimum_security_bits >=
                        if row.tier == SparseLwrSecurityTier::Primary100 {
                            PHASE1_PRIMARY_SECURITY_BITS
                        } else {
                            config.phase1_fallback_security_bits
                        }) ||
                row.minimum_security_bits != row.sparse_lwr_security_bits
            {
                return Err("Phase-1 checkpoint contains an inconsistent evaluated row".to_owned());
            }
        }
        let selected_index = self
            .evaluated
            .iter()
            .position(|row| {
                row.q_l == self.selected.q_l &&
                    row.p == self.selected.p &&
                    row.universe == self.selected.universe &&
                    row.weight == self.selected.weight
            })
            .ok_or_else(|| "Phase-1 checkpoint selected tuple is not evaluated".to_owned())?;
        let selected_row = &self.evaluated[selected_index];
        let earlier_primary = self.evaluated[..selected_index]
            .iter()
            .any(|row| row.qualified && row.tier == SparseLwrSecurityTier::Primary100);
        if !selected_row.qualified ||
            earlier_primary ||
            self.selected.tuple != config.sparse_lwr_phase1_grid[selected_index] ||
            self.selected.tier != selected_row.tier ||
            self.selected.q_l != selected_row.q_l ||
            self.selected.p != selected_row.p ||
            self.selected.universe != selected_row.universe ||
            self.selected.weight != selected_row.weight ||
            (self.selected.error_lower, self.selected.error_upper) !=
                (selected_row.error_lower, selected_row.error_upper) ||
            self.selected.estimator_commit != selected_row.estimator_commit ||
            self.selected.estimator_cost_model != selected_row.estimator_cost_model ||
            self.selected.estimator_shape_model != selected_row.estimator_shape_model ||
            self.selected.sparse_lwr_security_bits != selected_row.sparse_lwr_security_bits ||
            self.selected.tuple.estimator_security_bits != selected_row.estimator_security_bits ||
            (self.selected.raw_key_entropy_bits - selected_row.raw_key_entropy_bits).abs() > 1e-9
        {
            return Err(
                "Phase-1 checkpoint selected profile does not match its final row".to_owned()
            );
        }
        Ok(())
    }
}

/// Persist the accepted Phase-2 profile while retaining and revalidating all
/// Phase-1 evidence. The replacement is written beside the checkpoint and
/// renamed into place only after complete validation.
pub fn persist_accepted_phase2_profile(
    path: &Path,
    config: &SearchConfig,
    selected: &SelectedSparseLwrProfile,
    candidate: Candidate,
) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|error| format!("read Phase-1 checkpoint: {error}"))?;
    let mut checkpoint: Phase1Checkpoint = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse Phase-1 checkpoint: {error}"))?;
    checkpoint.validate(config)?;
    let row = selected
        .evaluations
        .iter()
        .find(|row| {
            row.q_l == selected.tuple.q_l &&
                row.p == selected.tuple.p &&
                row.universe == selected.tuple.nu &&
                row.weight == selected.tuple.h
        })
        .ok_or_else(|| "selected Phase-1 profile has no evaluated row".to_owned())?;
    if !row.qualified || row.tier != selected.tier {
        return Err("selected Phase-1 profile is not qualified for acceptance".to_owned());
    }
    checkpoint.accepted_phase2 =
        Some(AcceptedPhase2Profile { tuple: selected.tuple.clone(), candidate });
    checkpoint.validate(config)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&checkpoint).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write Phase-1 checkpoint: {error}"))?;
    fs::rename(&temporary, path).map_err(|error| format!("replace Phase-1 checkpoint: {error}"))
}

/// Read an explicitly named checkpoint, or perform and persist a fresh
/// Phase-1 search when that path does not exist.  A missing path is the only
/// condition that starts a fresh search when a checkpoint path was supplied;
/// malformed files and declaration mismatches fail closed.
pub fn load_or_search_phase1<Security>(
    config: &SearchConfig,
    checkpoint_path: Option<&Path>,
    sparse_security: Security,
) -> Result<SelectedSparseLwrProfile, String>
where
    Security: FnMut(usize, usize, usize, usize) -> Result<u64, String>,
{
    if let Some(path) = checkpoint_path {
        match fs::read(path) {
            Ok(bytes) => {
                let checkpoint: Phase1Checkpoint = serde_json::from_slice(&bytes)
                    .map_err(|error| format!("read Phase-1 checkpoint: {error}"))?;
                checkpoint.validate(config)?;
                info!(path = %path.display(), "reused Phase-1 checkpoint");
                return Ok(checkpoint.selected);
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!("read Phase-1 checkpoint {}: {error}", path.display()));
            }
        }
    }

    let selected = select_sparse_lwr_profile(config, sparse_security)?;
    if let Some(path) = checkpoint_path {
        let checkpoint = Phase1Checkpoint::from_selection(config, &selected)?;
        let bytes = serde_json::to_vec_pretty(&checkpoint)
            .map_err(|error| format!("encode Phase-1 checkpoint: {error}"))?;
        fs::write(path, bytes)
            .map_err(|error| format!("write Phase-1 checkpoint {}: {error}", path.display()))?;
        info!(path = %path.display(), "persisted Phase-1 checkpoint before Phase 2");
    }
    Ok(selected)
}

/// The complete public declaration of the Phase-1 search.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Phase1SearchReport {
    /// Complete public input and security-model declaration for these rows.
    pub declaration: Phase1Declaration,
    pub parameter_grid: Vec<SparseLwrParameterTuple>,
    pub universe_grid: Vec<usize>,
    pub support_weight: usize,
    pub q_l: usize,
    pub output_modulus: usize,
    pub lut_width: usize,
    pub security_target_bits: u64,
    pub evaluated: Vec<SparseLwrEvaluation>,
    pub selected_universe: usize,
    pub selected_q_l: usize,
    pub selected_p: usize,
    pub selected_weight: usize,
    pub selected_tier: SparseLwrSecurityTier,
    pub selected_raw_key_entropy_bits: f64,
    pub selected_error_lower: i64,
    pub selected_error_upper: i64,
    pub estimator_commit: String,
    pub estimator_cost_model: String,
    pub estimator_shape_model: String,
}

/// The complete public declaration of the Phase-2 search order and bounds.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Phase2SearchReport {
    pub crt_depth_min: usize,
    pub crt_depth_max: usize,
    pub log_ring_dimension_min: usize,
    pub log_ring_dimension_max: usize,
    pub order: &'static str,
    pub security_target_bits: u64,
    pub overall_target_policy: String,
    pub crt_base_bits_grid: Vec<(usize, u32)>,
    pub noise_model: NoiseSearchModel,
}

/// Persisted report wrapper for the parameter-search test.
///
/// `result` is deliberately accompanied by the declared search domains and
/// all evaluated Phase-1 rows.  This prevents a reader from mistaking the
/// selected point for a global minimum when it is only minimal in the
/// explicitly declared finite grids.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RefreshParameterSearchReport {
    pub phase1: Phase1SearchReport,
    pub phase2: Phase2SearchReport,
    pub result: SearchResult,
}

/// Builds the public report metadata without retaining any secret-dependent
/// layout, support, selector, or schedule information.
pub fn search_report(
    config: &SearchConfig,
    sparse_profile: &SelectedSparseLwrProfile,
    result: SearchResult,
) -> RefreshParameterSearchReport {
    RefreshParameterSearchReport {
        phase1: Phase1SearchReport {
            declaration: Phase1Declaration::from_config(config, sparse_profile)
                .expect("reviewed Phase-1 config must have a valid error model"),
            parameter_grid: config.sparse_lwr_phase1_grid.clone(),
            universe_grid: vec![sparse_profile.universe],
            support_weight: sparse_profile.weight,
            q_l: sparse_profile.q_l,
            output_modulus: sparse_profile.p,
            lut_width: sparse_profile.tuple.lut_width,
            security_target_bits: PHASE1_PRIMARY_SECURITY_BITS,
            evaluated: sparse_profile.evaluations.clone(),
            selected_universe: sparse_profile.universe,
            selected_q_l: sparse_profile.q_l,
            selected_p: sparse_profile.p,
            selected_weight: sparse_profile.weight,
            selected_tier: sparse_profile.tier,
            selected_raw_key_entropy_bits: sparse_profile.raw_key_entropy_bits,
            selected_error_lower: sparse_profile.error_lower,
            selected_error_upper: sparse_profile.error_upper,
            estimator_commit: sparse_profile.estimator_commit.clone(),
            estimator_cost_model: sparse_profile.estimator_cost_model.clone(),
            estimator_shape_model: sparse_profile.estimator_shape_model.clone(),
        },
        phase2: Phase2SearchReport {
            crt_depth_min: *config.crt_depths.start(),
            crt_depth_max: *config.crt_depths.end(),
            log_ring_dimension_min: *config.log_ring_dimensions.start(),
            log_ring_dimension_max: *config.log_ring_dimensions.end(),
            order: "base_bits_descending_then_crt_bits_descending_then_crt_depth_then_log_ring_dimension",
            security_target_bits: if sparse_profile.tier == SparseLwrSecurityTier::Fallback100 {
                config.phase1_fallback_security_bits
            } else {
                config.security_bits
            },
            overall_target_policy: if sparse_profile.tier == SparseLwrSecurityTier::Fallback100 {
                "fallback100"
            } else {
                "primary100"
            }
            .to_owned(),
            crt_base_bits_grid: config.crt_base_bits_grid.clone(),
            noise_model: config.noise_model,
        },
        result,
    }
}

/// Selects a qualifying sparse-LWR tuple from the explicit ordered Phase-1
/// grid. The callback is invoked exactly once per row and the whole grid is
/// exhausted before applying the primary-100-over-fallback-100 policy.
pub fn select_sparse_lwr_profile<Security>(
    config: &SearchConfig,
    mut sparse_security: Security,
) -> Result<SelectedSparseLwrProfile, String>
where
    Security: FnMut(usize, usize, usize, usize) -> Result<u64, String>,
{
    if config.sparse_lwr_phase1_grid.is_empty() {
        return Err("sparse-LWR Phase-1 tuple grid must not be empty".to_owned());
    }
    for tuple in &config.sparse_lwr_phase1_grid {
        if tuple.p != 2 || tuple.q_l == 0 || tuple.q_l % 2 != 0 {
            return Err("sparse-LWR Phase-1 tuple grid requires p == 2 and even Q_L".to_owned());
        }
        tuple.candidate().map_err(|error| format!("invalid Phase-1 tuple: {error}"))?;
    }
    if config.sparse_lwr_phase1_grid.windows(2).any(|window| {
        window[0].q_l > window[1].q_l ||
            (window[0].q_l == window[1].q_l && window[0].nu >= window[1].nu)
    }) {
        return Err("sparse-LWR Phase-1 tuple grid must be in reviewed ascending order".to_owned());
    }
    let mut evaluations = Vec::new();
    let mut primary_selection = None;
    let mut fallback_selection = None;
    for tuple in &config.sparse_lwr_phase1_grid {
        let universe = tuple.nu;
        let weight = tuple.h;
        let q_l = tuple.q_l;
        let p = tuple.p;
        let sparse_bits = sparse_security(universe, weight, q_l, p)?;
        let exact_minimum = tuple.estimator_minimum_classical_bits;
        if !exact_minimum.is_finite() || exact_minimum < 0.0 {
            return Err(format!(
                "estimator declaration for (Q_L={}, p={}, nu={}, h={}) is not a finite nonnegative minimum",
                tuple.q_l, tuple.p, tuple.nu, tuple.h
            ));
        }
        let expected_floor = exact_minimum.floor() as u64;
        if sparse_bits != expected_floor || sparse_bits != tuple.estimator_security_bits {
            return Err(format!(
                "estimator result for (Q_L={}, p={}, nu={}, h={}) was {} bits, expected declared floor {}",
                tuple.q_l, tuple.p, tuple.nu, tuple.h, sparse_bits, expected_floor
            ));
        }
        let tier = if exact_minimum >= PHASE1_PRIMARY_SECURITY_BITS as f64 {
            SparseLwrSecurityTier::Primary100
        } else {
            SparseLwrSecurityTier::Fallback100
        };
        let qualified = exact_minimum >= tier.target_bits() as f64;
        let (error_lower, error_upper) = sparse_lwr_error_bounds(q_l, p)?;
        evaluations.push(SparseLwrEvaluation {
            q_l,
            p,
            universe,
            weight,
            tier,
            estimator_commit: config.phase1_estimator_commit.clone(),
            estimator_cost_model: config.phase1_estimator_cost_model.clone(),
            estimator_shape_model: config.phase1_estimator_shape_model.clone(),
            error_lower,
            error_upper,
            sparse_lwr_security_bits: sparse_bits,
            estimator_security_bits: tuple.estimator_security_bits,
            raw_key_entropy_bits: raw_key_entropy_bits(universe, weight),
            minimum_security_bits: sparse_bits,
            qualified,
        });
        info!(
            sparse_lwr_q_l = q_l,
            sparse_lwr_p = p,
            sparse_lwr_universe = universe,
            sparse_lwr_weight = weight,
            sparse_lwr_security_bits = sparse_bits,
            tier = ?tier,
            "evaluated sparse-LWR Phase-1 profile"
        );
        if qualified && tier == SparseLwrSecurityTier::Primary100 && primary_selection.is_none() {
            primary_selection = Some((tuple.clone(), sparse_bits));
        } else if qualified &&
            tier == SparseLwrSecurityTier::Fallback100 &&
            fallback_selection.is_none()
        {
            fallback_selection = Some((tuple.clone(), sparse_bits));
        }
    }
    let (tuple, sparse_bits) = primary_selection.or(fallback_selection).ok_or_else(|| {
        format!(
            "no sparse-LWR profile in the declared tuple grid passed the {}-bit primary or {}-bit fallback floor",
            PHASE1_PRIMARY_SECURITY_BITS, config.phase1_fallback_security_bits
        )
    })?;
    let (error_lower, error_upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p)?;
    Ok(SelectedSparseLwrProfile {
        tuple: tuple.clone(),
        parameter_grid: config.sparse_lwr_phase1_grid.clone(),
        q_l: tuple.q_l,
        p: tuple.p,
        universe: tuple.nu,
        weight: tuple.h,
        tier: if tuple.estimator_minimum_classical_bits >= PHASE1_PRIMARY_SECURITY_BITS as f64 {
            SparseLwrSecurityTier::Primary100
        } else {
            SparseLwrSecurityTier::Fallback100
        },
        error_lower,
        error_upper,
        estimator_commit: config.phase1_estimator_commit.clone(),
        estimator_cost_model: config.phase1_estimator_cost_model.clone(),
        estimator_shape_model: config.phase1_estimator_shape_model.clone(),
        sparse_lwr_security_bits: sparse_bits,
        raw_key_entropy_bits: raw_key_entropy_bits(tuple.nu, tuple.h),
        evaluations,
    })
}

/// Returns every qualifying Phase-1 profile in deterministic tuple order.
/// The estimator evidence has already been collected by
/// `select_sparse_lwr_profile`; this projection never invokes it again.
pub fn qualified_sparse_lwr_profiles(
    selected: &SelectedSparseLwrProfile,
) -> Vec<SelectedSparseLwrProfile> {
    selected
        .parameter_grid
        .iter()
        .filter_map(|tuple| {
            let row = selected.evaluations.iter().find(|row| {
                row.q_l == tuple.q_l && row.universe == tuple.nu && row.weight == tuple.h
            })?;
            if !row.qualified || row.tier != selected.tier {
                return None;
            }
            Some(SelectedSparseLwrProfile {
                tuple: tuple.clone(),
                parameter_grid: selected.parameter_grid.clone(),
                q_l: tuple.q_l,
                p: tuple.p,
                universe: tuple.nu,
                weight: tuple.h,
                tier: row.tier,
                error_lower: row.error_lower,
                error_upper: row.error_upper,
                estimator_commit: row.estimator_commit.clone(),
                estimator_cost_model: row.estimator_cost_model.clone(),
                estimator_shape_model: row.estimator_shape_model.clone(),
                sparse_lwr_security_bits: row.sparse_lwr_security_bits,
                raw_key_entropy_bits: row.raw_key_entropy_bits,
                evaluations: selected.evaluations.clone(),
            })
        })
        .collect()
}

/// Advance through qualified Phase-1 profiles until one completes Phase 2.
/// Only the canonical finite-grid exhaustion result advances to the next
/// profile; setup, estimator, and checker failures remain fatal.
pub fn search_qualified_profiles<Run>(
    selected: &SelectedSparseLwrProfile,
    mut run: Run,
) -> Result<(SelectedSparseLwrProfile, SearchResult), String>
where
    Run: FnMut(&SelectedSparseLwrProfile) -> Result<SearchResult, String>,
{
    for profile in qualified_sparse_lwr_profiles(selected) {
        match run(&profile) {
            Ok(result) => return Ok((profile, result)),
            Err(error) if error == NO_CANDIDATE_ERROR => {}
            Err(error) => return Err(error),
        }
    }
    Err("no qualified sparse-LWR profile produced a valid Phase-2 candidate".to_owned())
}

fn exact_reconstruction_coefficients(dcrt: &DCRTPolyParams) -> Vec<IntExpr> {
    dcrt.reconst_coeffs()
        .into_iter()
        .map(|value| BigInt::from_biguint(num_bigint::Sign::Plus, value).into())
        .collect()
}

/// Enumerates the grid in the exact order used for minimality claims: globally
/// descending base bits (then CRT width), followed by depth and ring
/// dimension. Entries whose concrete `qbits` would reach 2000 are skipped for
/// that pair only.
pub fn candidates(config: &SearchConfig) -> impl Iterator<Item = Candidate> + '_ {
    config.crt_base_bits_grid.iter().copied().flat_map(move |(crt_bits, base_bits)| {
        config.crt_depths.clone().flat_map(move |crt_depth| {
            config.log_ring_dimensions.clone().filter_map(move |log_ring_dimension| {
                (crt_depth.checked_mul(crt_bits).map_or(false, |q_bits| q_bits < 2000))
                    .then_some(Candidate { crt_depth, log_ring_dimension, crt_bits, base_bits })
            })
        })
    })
}

/// Runs a finite search with injectable security and checker functions.
///
/// The hooks make the ordering and minimality tests independent of Sage and
/// the full operational-noise graph.  The production ignored test supplies
/// the lattice-estimator wrapper and a checker hook after building the actual
/// PBC/PRF graph metadata for the candidate.
pub fn search_with_hooks<Prepare, Security, Checker>(
    config: &SearchConfig,
    sparse_profile: &SelectedSparseLwrProfile,
    mut prepare: Prepare,
    mut security: Security,
    mut checker: Checker,
) -> Result<SearchResult, String>
where
    Prepare: FnMut(Candidate) -> Result<PreparedCandidate, CandidatePreparationError>,
    Security: FnMut(Candidate) -> Result<u64, String>,
    Checker: FnMut(&PreparedCandidate) -> Result<bool, String>,
{
    if sparse_profile.q_l != config.sparse_lwr_modulus ||
        sparse_profile.p != config.sparse_lwr_output_modulus ||
        sparse_profile.universe != config.sparse_lwr_universe ||
        sparse_profile.weight != config.sparse_lwr_weight ||
        sparse_profile.tuple.lut_width != config.lut_width
    {
        return Err(
            "selected sparse-LWR profile does not match the frozen search configuration".to_owned()
        );
    }
    let overall_security_target = if sparse_profile.tier == SparseLwrSecurityTier::Fallback100 {
        config.phase1_fallback_security_bits
    } else {
        config.security_bits
    };
    for candidate in candidates(config) {
        info!(
            crt_depth = candidate.crt_depth,
            log_ring_dimension = candidate.log_ring_dimension,
            crt_bits = candidate.crt_bits,
            base_bits = candidate.base_bits,
            "evaluating Power-LUT refresh search candidate"
        );
        let bgg = security(candidate)?;
        let achieved = bgg.min(sparse_profile.sparse_lwr_security_bits);
        if achieved < overall_security_target {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                crt_bits = candidate.crt_bits,
                base_bits = candidate.base_bits,
                achieved_security_bits = achieved,
                "candidate rejected by security floor"
            );
            continue;
        }
        let prepared = match prepare(candidate) {
            Ok(prepared) => prepared,
            Err(CandidatePreparationError::Infeasible(error)) => {
                info!(
                    crt_depth = candidate.crt_depth,
                    log_ring_dimension = candidate.log_ring_dimension,
                    crt_bits = candidate.crt_bits,
                    base_bits = candidate.base_bits,
                    reason = %error,
                    "candidate rejected by exact preparation bounds"
                );
                continue;
            }
            Err(CandidatePreparationError::Fatal(error)) => return Err(error),
        };
        if !checker(&prepared)? {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                crt_bits = candidate.crt_bits,
                base_bits = candidate.base_bits,
                "candidate rejected by Power-LUT noise threshold"
            );
            continue;
        }
        return Ok(SearchResult {
            candidate,
            achieved_security_bits: achieved,
            bgg_rlwe_security_bits: bgg,
            sparse_lwr_security_bits: sparse_profile.sparse_lwr_security_bits,
            raw_key_entropy_bits: sparse_profile.raw_key_entropy_bits,
            sparse_lwr_universe: sparse_profile.universe,
            sparse_lwr_weight: sparse_profile.weight,
            official_preimage_bound: prepared.official_preimage_bound.to_string(),
            ring_dimension: prepared.ring_dimension,
            bucket_width: prepared.bucket_width,
            pbc_attempts_used: prepared.pbc_attempts_used,
            layout_id: hex(prepared.layout_id),
            program_id: hex(prepared.program_id),
            checker_accepted: true,
            sparse_lwr_q_l: sparse_profile.q_l,
            sparse_lwr_p: sparse_profile.p,
            sparse_lwr_tier: sparse_profile.tier,
            estimator_commit: sparse_profile.estimator_commit.clone(),
            estimator_cost_model: sparse_profile.estimator_cost_model.clone(),
            estimator_shape_model: sparse_profile.estimator_shape_model.clone(),
        });
    }
    Err(NO_CANDIDATE_ERROR.to_owned())
}

/// Builds the concrete public setup and ordinary sparse-LWR program for one
/// candidate.  It does not execute a backend, sample artifacts, or perform a
/// round trip; those are intentionally outside this symbolic search.
pub fn prepare_candidate(
    config: &SearchConfig,
    candidate: Candidate,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    let ring_dimension = 1usize
        .checked_shl(candidate.log_ring_dimension as u32)
        .ok_or_else(|| "ring dimension shift overflow".to_owned())?;
    let dcrt = DCRTPolyParams::new(
        ring_dimension as u32,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
    );
    let (crt_moduli, _, _) = dcrt.to_crt();
    let modulus = BigInt::from_biguint(num_bigint::Sign::Plus, dcrt.modulus().as_ref().clone());
    let layout = BggSamplerLayout {
        modulus: modulus.clone().into(),
        ring_dimension: ring_dimension.into(),
        secret_dimension: config.secret_dimension,
        digit_count: dcrt.modulus_digits(),
        gadget_base: (BigInt::from(1_u8) << candidate.base_bits).into(),
    };
    let refresh = RefreshCompiler {
        full_modulus: modulus.clone().into(),
        crt_plaintext_moduli: crt_moduli.iter().map(|value| BigInt::from(*value).into()).collect(),
        reconstruction_coefficients: exact_reconstruction_coefficients(&dcrt),
    };
    let pbc_parameters =
        PbcParameters::paper_evaluation(config.sparse_lwr_universe, config.sparse_lwr_weight);
    if pbc_parameters.max_seed_attempts != config.pbc_max_attempts {
        return Err("paper-evaluation PBC profile does not use the reviewed retry count"
            .to_owned()
            .into());
    }
    let mut support_rng = StdRng::from_seed([0x6a; 32]);
    let support = sample(&mut support_rng, config.sparse_lwr_universe, config.sparse_lwr_weight)
        .into_iter()
        .collect::<Vec<_>>();
    let generated = generate_key_layout(&pbc_parameters, PbcRootSeed([0x19; 32]), &support)
        .map_err(|error| format!("PBC layout generation: {error}"))?;
    generated
        .private_schedule()
        .validate(&generated.public_layout)
        .map_err(|error| format!("PBC private schedule validation: {error}"))?;
    if generated.public_layout.parameters.hash_count != 3 ||
        generated.public_layout.parameters.bucket_count != config.sparse_lwr_weight + 3
    {
        return Err("paper-evaluation PBC must provide exactly three dummy-selected buckets"
            .to_owned()
            .into());
    }
    // `validate` proves every selected bucket is either one distinct real
    // support coordinate or a dummy cell.  Since it also proves the exact
    // support-assignment count, the three remaining buckets are dummy
    // selections without exposing the private schedule fields here.
    let dummy_buckets = generated.public_layout.parameters.bucket_count - support.len();
    if dummy_buckets != 3 {
        return Err(
            format!("expected exactly 3 dummy-selected buckets, found {dummy_buckets}").into()
        );
    }
    for seed in [0_u64, 1, 2, 3] {
        let public = (0..config.sparse_lwr_universe)
            .map(|index| {
                ((index as u64).wrapping_mul(seed + 3) + 7) % config.sparse_lwr_modulus as u64
            })
            .collect::<Vec<_>>();
        let actual = clear_pbc_inner_product(
            &generated.public_layout,
            generated.private_schedule(),
            &public,
            config.sparse_lwr_modulus as u64,
        )
        .map_err(|error| format!("PBC clear inner product: {error}"))?;
        let expected = support
            .iter()
            .fold(0_u64, |sum, &index| (sum + public[index]) % config.sparse_lwr_modulus as u64);
        if actual != expected {
            return Err(format!("PBC clear inner product mismatch: {actual} != {expected}").into());
        }
    }
    let profile = SparseLwrPrfProfile::new(
        config.sparse_lwr_modulus,
        config.sparse_lwr_output_modulus,
        config.lut_width,
        ring_dimension,
    )
    .map_err(|error| format!("sparse-LWR profile: {error}"))?;
    let program = SparseLwrPrfProgram::new(
        profile.clone(),
        generated.public_layout.bucket_width,
        generated.public_layout.parameters.bucket_count,
    )
    .map_err(|error| format!("sparse-LWR program: {error}"))?;
    let base_p = BigUint::from(config.sparse_lwr_output_modulus);
    let q_l = BigUint::from(config.sparse_lwr_modulus);
    let fresh_error_digits = fresh_error_base_p_digit_count(&base_p, &crt_moduli)?;
    let provisional_mask_digits =
        minimum_mask_base_p_digit_count(config.sparse_lwr_modulus, base_p.to_usize().unwrap())?;
    // Constructing this provisional value is cheap and gives the selector the
    // authoritative sigma-derived B_chi. No graph or encoded family is built
    // until the exact digit search has selected d_m and d_e.
    let sigma_decoder = RealExpr::from_f64_exact(config.decoder_sigma)
        .map_err(|error| format!("decoder sigma: {error}"))?;
    let sigma_encoding = RealExpr::from_f64_exact(config.error_sigma)
        .map_err(|error| format!("encoding sigma: {error}"))?;
    let provisional_setup = RefreshSetupParameters::new(
        [0x52; 32],
        base_p.to_usize().ok_or_else(|| "base-p modulus does not fit usize".to_owned())?,
        config.secret_dimension,
        ring_dimension,
        provisional_mask_digits,
        fresh_error_digits,
        config.mask_statistical_security_bits,
        config.lut_width,
        layout.clone(),
        refresh.clone(),
        sigma_decoder.clone(),
        sigma_encoding.clone(),
        "refresh-parameter-search-provisional",
    );
    let helper_error_bound = provisional_setup
        .encoding_error_bound
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|error| format!("encoding cutoff evaluation: {error}"))?
        .to_biguint()
        .ok_or_else(|| "encoding cutoff must be non-negative".to_owned())?;
    let decoder_preimage_bound = provisional_setup
        .resolve_decoder_preimage_bound()
        .map_err(|error| format!("official decoder bound: {error}"))?
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|error| format!("official decoder bound evaluation: {error}"))?
        .to_biguint()
        .ok_or_else(|| "decoder bound must be non-negative".to_owned())?;
    let noise_model = mxx_power_lut::PowerLutNoiseParameters::dense(
        ring_dimension,
        layout
            .gadget_base
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|error| format!("gadget base evaluation: {error}"))?
            .to_biguint()
            .ok_or_else(|| "gadget base must be positive".to_owned())?,
        layout.digit_count,
        helper_error_bound.clone(),
    )
    .map_err(|error| format!("exact noise model: {error}"))?;
    let prf_report = mxx_power_lut::noise::simulate_sparse_prf(
        &program,
        &noise_model,
        &generated.public_layout,
        helper_error_bound.clone(),
    )
    .map_err(|error| format!("exact sparse-PRF bound: {error}"))?;
    info!(
        q = prf_report.q_l,
        p = prf_report.p,
        bucket_count = prf_report.bucket_count,
        k = prf_report.k,
        lut_width = prf_report.lut_width,
        intermediate_groups = prf_report.intermediate_groups,
        terminal_start = prf_report.terminal_start_bucket,
        terminal_len = prf_report.terminal_bucket_len,
        "sparse-PRF grouped plan during candidate preparation"
    );
    for bucket in &prf_report.bucket_stages {
        info!(
            bucket = bucket.bucket,
            active_count = bucket.active_count,
            input_bits = bucket.input_bound.bits(),
            gamma_selector_bits = bucket.gamma_selector.bits(),
            one_hot_output_bits = bucket.one_hot_output_bound.bits(),
            one_hot_additive_bits = bucket.one_hot_additive_bound.bits(),
            one_hot_bit_growth = bucket.one_hot_bit_growth,
            gamma_c_bits = bucket.gamma_c.bits(),
            gamma_a_bits = bucket.gamma_a.bits(),
            selection_inherited_bits = bucket.selection_inherited_bits,
            selection_additive_bits = bucket.selection_additive_bits,
            lut_output_bits = bucket.lut_output_bound.bits(),
            lut_additive_bits = bucket.lut_additive_bound.bits(),
            lut_bit_growth = bucket.lut_bit_growth,
            "sparse-PRF bucket noise stages during candidate preparation"
        );
    }
    for group in &prf_report.group_stages {
        info!(
            group = group.group,
            start_bucket = group.start_bucket,
            bucket_len = group.bucket_len,
            lut_width = group.lut_width,
            input_bits = group.input_bound.bits(),
            unreduced_bits = group.unreduced_bound.bits(),
            inherited_bits = group.inherited_bound.bits(),
            base_helper_additive_bits = group.base_helper_additive.bits(),
            gamma_a_additive_bits = group.gamma_a_additive.bits(),
            additive_bits = group.additive_bound.bits(),
            output_bits = group.output_bits,
            bit_growth = group.bit_growth,
            gamma_c_bits = group.gamma_c.bits(),
            gamma_a_bits = group.gamma_a.bits(),
            "sparse-PRF grouped reduction noise stage during candidate preparation"
        );
    }
    info!(
        terminal_lut_width = prf_report.terminal_lut_width,
        terminal_input_bits = prf_report.terminal_input_bound.bits(),
        terminal_output_bits = prf_report.terminal_output_bound.bits(),
        terminal_additive_bits = prf_report.terminal_additive_bound.bits(),
        terminal_bit_growth = prf_report.terminal_bit_growth,
        terminal_range_start = prf_report.terminal_start_bucket,
        terminal_range_len = prf_report.terminal_bucket_len,
        terminal_W = prf_report.terminal_lut_width,
        terminal_gamma_c_bits = prf_report.terminal_gamma_c.bits(),
        terminal_gamma_a_bits = prf_report.terminal_gamma_a.bits(),
        terminal_inherited_bits = prf_report.terminal_inherited_bits,
        terminal_base_helper_additive_bits = prf_report.terminal_base_helper_additive_bits,
        terminal_gamma_a_additive_bits = prf_report.terminal_gamma_a_additive_bits,
        "sparse-PRF terminal LUT noise stage during candidate preparation"
    );
    let prf_output_bound = prf_report.output_bound;
    let mask_digits = if config.noise_model == NoiseSearchModel::AverageCase {
        let average_config = average_case_config(config)?;
        select_average_mask_base_p_digit_count(
            &provisional_setup,
            program.clone(),
            generated.public_layout.clone(),
            &base_p,
            &q_l,
            &crt_moduli.iter().map(|value| BigUint::from(*value)).collect::<Vec<_>>(),
            &average_config,
        )?
    } else {
        select_mask_base_p_digit_count(
            &base_p,
            &q_l,
            &modulus.to_biguint().ok_or_else(|| "full modulus must be positive".to_owned())?,
            &crt_moduli.iter().map(|value| BigUint::from(*value)).collect::<Vec<_>>(),
            ring_dimension,
            layout.digit_count,
            &layout
                .gadget_base
                .evaluate(&mxx_ir_core::ParamEnv::default())
                .map_err(|error| format!("gadget base evaluation: {error}"))?
                .to_biguint()
                .ok_or_else(|| "gadget base must be positive".to_owned())?,
            &helper_error_bound,
            &decoder_preimage_bound,
            &helper_error_bound,
            &prf_output_bound,
            config.mask_statistical_security_bits,
            ring_dimension,
            fresh_error_digits,
        )?
    };
    let setup = RefreshSetupParameters::new(
        [0x52; 32],
        base_p.to_usize().ok_or_else(|| "base-p modulus does not fit usize".to_owned())?,
        config.secret_dimension,
        ring_dimension,
        mask_digits,
        fresh_error_digits,
        config.mask_statistical_security_bits,
        config.lut_width,
        layout,
        refresh,
        sigma_decoder,
        sigma_encoding,
        "refresh-parameter-search",
    );
    if setup.decoder_preimage_bound != PreimageCoefficientBound::Official {
        return Err("refresh setup did not retain the official preimage bound policy"
            .to_owned()
            .into());
    }
    let official_preimage_bound = setup
        .resolve_decoder_preimage_bound()
        .map_err(|error| format!("official decoder bound: {error}"))?
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|error| format!("official decoder bound evaluation: {error}"))?;
    if !config.one_nontrivial_refresh_round || config.plaintext_modulus != 2 {
        return Err("the reviewed refresh profile was changed".to_owned().into());
    }
    let bucket_width = generated.public_layout.bucket_width;
    let layout_id = generated.public_layout.layout_id.0;
    let pbc_attempts_used = generated.public_layout.accepted_attempt + 1;
    let program_id = *program.id().as_bytes();
    let ring = setup.layout.ring();
    let expected_plaintext = ring.polynomial([BigInt::from(0_u8).into()]);
    let bundle = RefreshParameterSimulationRequest::new(
        setup,
        profile,
        generated,
        [0x4b; 32],
        RealExpr::from_f64_exact(config.decoder_sigma)
            .map_err(|error| format!("decoder sigma: {error}"))?,
        expected_plaintext,
        1,
    )
    .map_err(|error| format!("refresh simulation request: {error}"))?
    .build()
    .map_err(|error| format!("refresh simulation graph: {error}"))?;
    Ok(PreparedCandidate {
        candidate,
        ring_dimension,
        bucket_width,
        pbc_attempts_used,
        official_preimage_bound,
        layout_id,
        program_id,
        bundle: Some(bundle),
    })
}

fn minimum_mask_base_p_digit_count(q_l: usize, base_p: usize) -> Result<usize, String> {
    if q_l == 0 || base_p < 2 {
        return Err("invalid base-p digit-count inputs".to_owned());
    }
    let mut covered = 1usize;
    let mut digits = 0usize;
    while covered < q_l {
        covered =
            covered.checked_mul(base_p).ok_or_else(|| "base-p digit-count overflow".to_owned())?;
        digits = digits.checked_add(1).ok_or_else(|| "base-p digit-count overflow".to_owned())?;
    }
    Ok(digits)
}

/// Chooses the smallest fresh-error digit count for which B_e=p^d_e-1 is
/// strictly below every authoritative CRT tower modulus.
fn fresh_error_base_p_digit_count(
    base_p: &BigUint,
    crt_moduli: &[u64],
) -> Result<usize, CandidatePreparationError> {
    if base_p < &BigUint::from(2u8) ||
        crt_moduli.is_empty() ||
        crt_moduli.iter().any(|modulus| *modulus <= 1)
    {
        return Err(CandidatePreparationError::Fatal(
            "invalid fresh-error digit-count inputs".to_owned(),
        ));
    }
    let min_modulus = crt_moduli.iter().copied().min().unwrap();
    let fresh_bound = base_p - BigUint::one();
    if fresh_bound < BigUint::from(min_modulus) {
        Ok(1)
    } else {
        Err(CandidatePreparationError::Infeasible(
            "no fresh-error digit count satisfies B_e < every CRT modulus".to_owned(),
        ))
    }
}

/// Selects the first AverageCase mask digit count using the complete
/// setup-owned snapshot for every candidate.  The deterministic doubled mask
/// bound is the finite stopping condition; correctness and both hard gates
/// come only from the production report.
fn select_average_mask_base_p_digit_count(
    setup: &RefreshSetupParameters,
    program: SparseLwrPrfProgram,
    layout: mxx_power_lut::pbc::PbcPublicLayout,
    base_p: &BigUint,
    q_l: &BigUint,
    crt_moduli: &[BigUint],
    average_config: &AverageCaseConfig,
) -> Result<usize, CandidatePreparationError> {
    let mut digits = minimum_mask_base_p_digit_count(
        q_l.to_usize()
            .ok_or_else(|| CandidatePreparationError::Fatal("Q_L does not fit usize".to_owned()))?,
        base_p.to_usize().ok_or_else(|| {
            CandidatePreparationError::Fatal("base-p modulus does not fit usize".to_owned())
        })?,
    )?;
    let min_spacing = minimum_crt_spacing(crt_moduli)?;
    loop {
        let d2 = base_p.pow(u32::try_from(digits).map_err(|_| {
            CandidatePreparationError::Fatal("mask digit count overflow".to_owned())
        })?) - BigUint::one();
        // AverageCase uses the strict doubled-coordinate deterministic
        // precondition.  Do not evaluate the boundary itself: every later
        // digit has a still larger deterministic mask displacement and is
        // therefore infeasible.  No monotonicity assumption is made about
        // hiding, F_avg, or rounding before this exact boundary.
        if d2 * 2u8 >= min_spacing.clone() * 2u8 {
            return Err(CandidatePreparationError::Infeasible(format!(
                "AverageCase deterministic mask boundary reached at d_m={digits}"
            )));
        }
        let report = setup
            .evaluate_average_candidate(program.clone(), layout.clone(), digits, average_config)
            .map_err(|error| {
                CandidatePreparationError::Fatal(format!("AverageCase candidate: {error}"))
            })?;
        info!(
            mask_digits = digits,
            mask_domain_accepted = report.refresh.domain_accepted,
            mask_smudging_accepted = report.refresh.mask_smudging_accepted,
            fresh_error_accepted = report.refresh.fresh_error_accepted,
            mask_bound_bits = report.refresh.mask_bound.bits(),
            average_favg_bits = report.refresh.mask_smudging_max_favg.bits(),
            average_joint_event_count = ?report.refresh.joint_event_count,
            average_epsilon_numerator_bits = report.refresh.epsilon_joint.numerator.bits(),
            average_epsilon_denominator_bits = report.refresh.epsilon_joint.denominator.bits(),
            average_masking_distance_numerator_bits = report.refresh.masking_distance_bound.numerator.bits(),
            average_masking_distance_denominator_bits = report.refresh.masking_distance_bound.denominator.bits(),
            average_mask_smudging_margin_sign = signed_margin_sign(&report.refresh.mask_smudging_margin),
            average_mask_smudging_margin_bits = report.refresh.mask_smudging_margin.magnitude().bits(),
            average_domain_margin_sign = signed_margin_sign(&report.refresh.domain_margin),
            average_domain_margin_bits = report.refresh.domain_margin.magnitude().bits(),
            average_event_log2 = ?report.refresh.event_budget.log2_events(),
            mandatory_hard_gates_accepted = report.hard_authority_accepted,
            average_smudging_accepted = report.refresh.mask_smudging_accepted,
            average_rounding_accepted = report.refresh.rounding_accepted,
            accepted = report.accepted,
            "evaluated AverageCase mask digit candidate"
        );
        if let Some(slot) = report.refresh.slots.iter().max_by_key(|slot| &slot.favg) {
            info!(
                mask_digits = digits,
                slot = slot.slot,
                spacing_bits = slot.spacing.bits(),
                favg_bits = slot.favg.bits(),
                stochastic_variance_numerator_bits = slot.stochastic_variance.numerator.bits(),
                stochastic_variance_denominator_bits = slot.stochastic_variance.denominator.bits(),
                squared_margin_sign = signed_margin_sign(&slot.squared_margin),
                squared_margin_bits = slot.squared_margin.magnitude().bits(),
                squared_deficit_sign = signed_margin_sign(&slot.squared_deficit),
                squared_deficit_bits = slot.squared_deficit.magnitude().bits(),
                fresh_error_accepted = slot.fresh_error_below_plaintext_modulus,
                average_rounding_accepted = slot.rounding_accepted,
                slot_count = report.refresh.slots.len(),
                "AverageCase worst per-slot mask candidate diagnostics"
            );
        }
        if report.hard_authority_accepted &&
            report.refresh.domain_accepted &&
            report.refresh.fresh_error_accepted &&
            report.refresh.mask_smudging_accepted &&
            report.refresh.rounding_accepted &&
            report.accepted
        {
            return Ok(digits);
        }
        digits = digits.checked_add(1).ok_or_else(|| {
            CandidatePreparationError::Fatal("mask digit count overflow".to_owned())
        })?;
    }
}

/// Builds the single AverageCase policy used by both d_m selection and the
/// final bundle check.  Keeping this projection centralized prevents a
/// selector from silently evaluating a different event/failure budget than
/// the accepted graph.
fn average_case_config(config: &SearchConfig) -> Result<AverageCaseConfig, String> {
    Ok(AverageCaseConfig {
        failure_exponent: u32::try_from(config.security_bits)
            .map_err(|_| "AverageCase security target does not fit u32".to_owned())?,
        allow_average_acceptance: true,
        ..Default::default()
    })
}

fn signed_margin_sign(value: &BigInt) -> &'static str {
    if value.sign() == num_bigint::Sign::Minus { "negative" } else { "nonnegative" }
}

fn minimum_crt_spacing(crt_moduli: &[BigUint]) -> Result<BigUint, CandidatePreparationError> {
    let full_modulus = crt_moduli.iter().fold(BigUint::one(), |product, modulus| product * modulus);
    crt_moduli
        .iter()
        .map(|modulus| {
            if (&full_modulus % modulus) != BigUint::zero() {
                return Err(CandidatePreparationError::Fatal(
                    "CRT modulus does not divide full modulus for AverageCase selector".to_owned(),
                ));
            }
            Ok(&full_modulus / modulus)
        })
        .collect::<Result<Vec<_>, CandidatePreparationError>>()?
        .into_iter()
        .min()
        .ok_or_else(|| {
            CandidatePreparationError::Fatal(
                "missing CRT spacing for AverageCase selector".to_owned(),
            )
        })
}

/// Selects d_m by exact integer checks before any expensive refresh graph is
/// built. The upper bound is derived from the strict smallest CRT spacing:
/// once 2*(p^d-1) reaches that spacing, no later d can satisfy correctness.
fn select_mask_base_p_digit_count(
    base_p: &BigUint,
    q_l: &BigUint,
    full_modulus: &BigUint,
    crt_moduli: &[BigUint],
    ring_dimension: usize,
    gadget_digit_count: usize,
    gadget_base: &BigUint,
    helper_error_bound: &BigUint,
    decoder_preimage_bound: &BigUint,
    state_bound: &BigUint,
    prf_output_bound: &BigUint,
    security_bits: usize,
    coefficient_count: usize,
    fresh_error_digits: usize,
) -> Result<usize, CandidatePreparationError> {
    if base_p < &BigUint::from(2u8) ||
        q_l.is_zero() ||
        full_modulus.is_zero() ||
        crt_moduli.is_empty() ||
        ring_dimension == 0 ||
        gadget_digit_count == 0 ||
        gadget_base <= &BigUint::one() ||
        coefficient_count == 0 ||
        fresh_error_digits == 0
    {
        return Err(CandidatePreparationError::Fatal("invalid mask digit-count inputs".to_owned()));
    }
    validate_authoritative_crt_moduli(full_modulus, crt_moduli)?;
    let min_spacing = crt_moduli
        .iter()
        .map(|modulus| full_modulus / modulus)
        .min()
        .ok_or_else(|| "missing CRT spacing".to_owned())?;
    let component_columns = gadget_digit_count
        .checked_mul(2)
        .ok_or_else(|| "component-column count overflow".to_owned())?;
    let transcript_coordinates = BigUint::from(crt_moduli.len()) *
        BigUint::from(component_columns) *
        BigUint::from(coefficient_count);
    let security_shift =
        security_bits.checked_add(1).ok_or_else(|| "security shift overflow".to_owned())?;
    let security_factor = BigUint::one() << security_shift;
    let decoder_action_gain = BigUint::from(2u8) *
        BigUint::from(
            gadget_digit_count
                .checked_add(2)
                .ok_or_else(|| "decoder columns overflow".to_owned())?,
        ) *
        BigUint::from(ring_dimension) *
        decoder_preimage_bound;
    let decoder_term = helper_error_bound * decoder_action_gain;
    let mut digits = minimum_mask_base_p_digit_count(
        q_l.to_usize().ok_or_else(|| "Q_L does not fit usize".to_owned())?,
        base_p.to_usize().ok_or_else(|| "base-p modulus does not fit usize".to_owned())?,
    )?;
    loop {
        let exponent = u32::try_from(digits)
            .map_err(|_| CandidatePreparationError::Fatal("mask digit count overflow".into()))?;
        let mask_modulus = base_p.pow(exponent);
        let mask_bound = &mask_modulus - BigUint::one();
        let gains = mxx_power_lut::noise::refresh_action_gains(
            full_modulus,
            crt_moduli,
            base_p,
            digits,
            fresh_error_digits,
            gadget_base,
            gadget_digit_count,
            coefficient_count,
        )
        .map_err(|error| format!("exact refresh action gains: {error}"))?;
        let mut max_noise = BigUint::zero();
        let mut rounding_ok = true;
        for (slot, modulus) in crt_moduli.iter().enumerate() {
            let state_term = &gains.gamma_kappa[slot] * state_bound;
            let mask_term = &gains.mask_route_gain * prf_output_bound;
            let fresh_term = &gains.fresh_error_route_gains[slot] * prf_output_bound;
            let operation: BigUint = state_term + mask_term + fresh_term + &decoder_term;
            max_noise = max_noise.max(operation.clone());
            if BigUint::from(2u8) * (&mask_bound + operation) >= full_modulus / modulus {
                rounding_ok = false;
            }
        }
        let required_hiding = &security_factor * &transcript_coordinates * &max_noise;
        if mask_modulus >= *q_l && mask_modulus >= required_hiding && rounding_ok {
            return Ok(digits);
        }
        let slot_diagnostics = crt_moduli
            .iter()
            .enumerate()
            .map(|(slot, modulus)| {
                let state = &gains.gamma_kappa[slot] * state_bound;
                let mask = &gains.mask_route_gain * prf_output_bound;
                let fresh = &gains.fresh_error_route_gains[slot] * prf_output_bound;
                let operation = &state + &mask + &fresh + &decoder_term;
                let spacing = full_modulus / modulus;
                format!(
                    "slot={slot}: gamma_bits={} state_bits={} Rm_bits={} mask_bits={} Re_bits={} fresh_bits={} decoder_bits={} F_bits={} spacing_bits={} twice_pre_bits={}",
                    gains.gamma_kappa[slot].bits(),
                    state.bits(),
                    gains.mask_route_gain.bits(),
                    mask.bits(),
                    gains.fresh_error_route_gains[slot].bits(),
                    fresh.bits(),
                    decoder_term.bits(),
                    operation.bits(),
                    spacing.bits(),
                    (BigUint::from(2u8) * (&mask_bound + &operation)).bits(),
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        // Correctness gives a finite upper bound. Increasing d only raises
        // both B_m and the operation noise, so crossing it is terminal.
        if BigUint::from(2u8) * mask_bound >= min_spacing {
            return Err(CandidatePreparationError::Infeasible(format!(
                "no mask digit count satisfies strict refresh rounding: d_m={digits}, M_m_bits={}, max_F_bits={}, min_spacing_bits={}, required_hiding_bits={}, security_ok={}, domain_ok={}, diagnostics=[{slot_diagnostics}]",
                mask_modulus.bits(),
                max_noise.bits(),
                min_spacing.bits(),
                required_hiding.bits(),
                mask_modulus >= *q_l,
                mask_modulus >= required_hiding,
            )));
        }
        digits = digits.checked_add(1).ok_or_else(|| "mask digit count overflow".to_owned())?;
    }
}

/// Validates the exact CRT tower supplied by the authoritative DCRT setup.
/// The search consumes these moduli directly and never factors the full q.
fn validate_authoritative_crt_moduli(
    full_modulus: &BigUint,
    crt_moduli: &[BigUint],
) -> Result<(), String> {
    let mut product = BigUint::one();
    for (index, modulus) in crt_moduli.iter().enumerate() {
        if modulus <= &BigUint::one() {
            return Err("CRT tower modulus must be greater than one".to_owned());
        }
        if full_modulus % modulus != BigUint::zero() {
            return Err("CRT tower modulus must divide the full modulus".to_owned());
        }
        if crt_moduli[..index]
            .iter()
            .any(|previous| gcd_biguint(previous, modulus) != BigUint::one())
        {
            return Err("CRT tower moduli must be pairwise coprime".to_owned());
        }
        product *= modulus;
    }
    if product != *full_modulus {
        return Err("CRT tower moduli must have exact product q".to_owned());
    }
    Ok(())
}

fn gcd_biguint(left: &BigUint, right: &BigUint) -> BigUint {
    let mut left = left.clone();
    let mut right = right.clone();
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    left
}

/// Builds the arguments for the BGG ternary/Gaussian estimator invocation.
fn bgg_estimator_args(ring_dimension: usize, modulus: &BigUint, sigma: f64) -> Vec<String> {
    vec![
        ring_dimension.to_string(),
        modulus.to_string(),
        "--s-dist".to_owned(),
        r#"{"name":"Ternary"}"#.to_owned(),
        "--e-dist".to_owned(),
        format!(r#"{{"name":"DiscreteGaussian","stddev":{sigma}}}"#),
    ]
}

/// Derives the centered rounding-error interval for a lifted LWR sample.
///
/// The sparse-LWR estimator models the marginal error after lifting the
/// output modulus.  With `Delta = Q_L / p`, write the transformed sample as
/// `x = Delta * floor(x / Delta) + r`, where `0 <= r < Delta`; after the
/// known centered shift `floor(Delta / 2)`, the estimator error is
/// `e = r - floor(Delta / 2)`, giving the returned interval `[a, b]`.  This
/// is only a marginal surrogate: it does not model correlations between the
/// public coefficient and its rounding error.  The separate sparse-support
/// MITM floor remains part of the final minimum and is not folded into this
/// estimate.
pub fn sparse_lwr_error_bounds(q_l: usize, p: usize) -> Result<(i64, i64), String> {
    if p == 0 || q_l == 0 || q_l % p != 0 {
        return Err("sparse-LWR output modulus must divide a positive Q_L".to_owned());
    }
    let delta = q_l / p;
    let shift = delta / 2;
    let delta = i64::try_from(delta).map_err(|_| "sparse-LWR Delta overflows i64".to_owned())?;
    let shift = i64::try_from(shift).map_err(|_| "sparse-LWR shift overflows i64".to_owned())?;
    let upper = delta
        .checked_sub(1)
        .and_then(|value| value.checked_sub(shift))
        .ok_or_else(|| "sparse-LWR error interval overflows i64".to_owned())?;
    Ok((-shift, upper))
}

/// Builds the exact sparse-LWR estimator invocation for the reviewed model.
/// `--m` is intentionally omitted, so the CLI uses its default infinite
/// lifted dimension.  This does not ask the CLI to derive a finite sample
/// count or replace the transformed-error model documented above.
pub fn sparse_lwr_estimator_args(
    universe: usize,
    weight: usize,
    q_l: usize,
    p: usize,
) -> Result<Vec<String>, String> {
    let (a, b) = sparse_lwr_error_bounds(q_l, p)?;
    Ok(vec![
        universe.to_string(),
        q_l.to_string(),
        "--s-dist".to_owned(),
        format!(r#"{{"name":"SparseBinary","hw":{weight},"n":{universe}}}"#),
        "--e-dist".to_owned(),
        format!(r#"{{"name":"Uniform","a":{a},"b":{b}}}"#),
        "--exact".to_owned(),
    ])
}

/// Parses the estimator output while keeping the security result unambiguous.
///
/// The bundled CLI prints human-readable algorithm diagnostics followed by
/// one unsigned-decimal security estimate.  Diagnostics are allowed, but the
/// parser rejects output containing zero or multiple decimal result lines so
/// that a changed CLI format cannot silently select an arbitrary estimate.
pub fn parse_security_bits(stdout: &[u8]) -> Result<u64, String> {
    let text = std::str::from_utf8(stdout)
        .map_err(|error| format!("lattice-estimator-cli output is not UTF-8: {error}"))?;
    let result_lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| line.bytes().all(|byte| byte.is_ascii_digit()))
        .collect::<Vec<_>>();
    let line = match result_lines.as_slice() {
        [line] => line,
        [] => {
            return Err(
                "lattice-estimator-cli output contains no unsigned-decimal security estimate"
                    .to_owned(),
            );
        }
        _ => {
            return Err(
                "lattice-estimator-cli output contains multiple security estimates".to_owned()
            );
        }
    };
    line.parse::<u64>().map_err(|error| format!("invalid security estimate: {error}"))
}

fn run_lattice_estimator(args: Vec<String>) -> Result<u64, String> {
    let output = Command::new("lattice-estimator-cli")
        .args(args)
        .output()
        .map_err(|error| format!("failed to start lattice-estimator-cli: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "lattice-estimator-cli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_security_bits(&output.stdout)
}

/// Calls the existing BGG estimator wrapper used by the parameter-search
/// tests.  The result is parsed as one unambiguous security-bits line.
pub fn lattice_security_bits(
    ring_dimension: usize,
    modulus: &BigUint,
    sigma: f64,
) -> Result<u64, String> {
    run_lattice_estimator(bgg_estimator_args(ring_dimension, modulus, sigma))
}

/// Raw support-key entropy, retained as a diagnostic rather than a separate
/// hard security floor.
fn raw_key_entropy_bits(universe: usize, weight: usize) -> f64 {
    (0..weight).map(|index| ((universe - index) as f64).log2() - ((index + 1) as f64).log2()).sum()
}

pub fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reviewed_estimator_bits(universe: usize, q_l: usize) -> u64 {
        match (universe, q_l) {
            (450, 16) => 99,
            (451, 16) => 100,
            other => panic!("unexpected reviewed tuple {other:?}"),
        }
    }

    fn mock_prepared(candidate: Candidate) -> PreparedCandidate {
        PreparedCandidate {
            candidate,
            ring_dimension: 1 << candidate.log_ring_dimension,
            bucket_width: 3,
            official_preimage_bound: 10.into(),
            layout_id: [candidate.crt_depth as u8; 32],
            program_id: [candidate.log_ring_dimension as u8; 32],
            pbc_attempts_used: 1,
            bundle: None,
        }
    }

    fn mock_sparse_profile(config: &SearchConfig) -> SelectedSparseLwrProfile {
        let tuple = config.sparse_lwr_phase1_grid[0].clone();
        let (error_lower, error_upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p).unwrap();
        SelectedSparseLwrProfile {
            tuple: tuple.clone(),
            parameter_grid: config.sparse_lwr_phase1_grid.clone(),
            q_l: config.sparse_lwr_modulus,
            p: config.sparse_lwr_output_modulus,
            universe: config.sparse_lwr_universe,
            weight: config.sparse_lwr_weight,
            tier: SparseLwrSecurityTier::Primary100,
            error_lower,
            error_upper,
            estimator_commit: config.phase1_estimator_commit.clone(),
            estimator_cost_model: config.phase1_estimator_cost_model.clone(),
            estimator_shape_model: config.phase1_estimator_shape_model.clone(),
            sparse_lwr_security_bits: 200,
            raw_key_entropy_bits: raw_key_entropy_bits(
                config.sparse_lwr_universe,
                config.sparse_lwr_weight,
            ),
            evaluations: Vec::new(),
        }
    }

    #[test]
    fn grid_is_lexicographic_depth_then_ring_dimension_when_base_is_fixed() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 2..=3;
        config.log_ring_dimensions = 5..=6;
        config.crt_base_bits_grid = vec![(32, 16)];
        assert_eq!(
            candidates(&config).collect::<Vec<_>>(),
            vec![
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 2, log_ring_dimension: 6, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 3, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 3, log_ring_dimension: 6, crt_bits: 32, base_bits: 16 },
            ]
        );
    }

    #[test]
    fn reviewed_grid_declares_phase_two_minimality_scope() {
        let config = SearchConfig::reviewed();
        assert_eq!(config.crt_depths, 30..=62);
        assert_eq!(config.log_ring_dimensions, 16..=17);
        assert_eq!(config.crt_bits, 32);
        assert_eq!(config.base_bits, 16);
        assert_eq!(config.crt_base_bits_grid, descending_crt_base_bits_grid(32));
        assert_eq!(config.crt_base_bits_grid[0], (32, 16));
    }

    #[test]
    fn crt_base_grid_descends_base_width_for_each_crt_width() {
        let grid = descending_crt_base_bits_grid(6);
        assert_eq!(grid, vec![(6, 3), (6, 2), (6, 1)]);
    }

    #[test]
    fn phase_two_candidates_keep_depth_ring_order_and_descend_base_bits() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=1;
        config.log_ring_dimensions = 5..=5;
        config.crt_base_bits_grid = vec![(5, 2), (4, 2), (5, 1), (4, 1)];
        assert_eq!(
            candidates(&config).collect::<Vec<_>>(),
            vec![
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 5, base_bits: 2 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 4, base_bits: 2 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 5, base_bits: 1 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 4, base_bits: 1 },
            ]
        );
    }

    #[test]
    fn phase_two_candidates_exhaust_largest_base_before_smaller_base() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=2;
        config.log_ring_dimensions = 5..=5;
        config.crt_base_bits_grid = vec![(32, 16), (30, 15), (32, 15)];
        let candidates = candidates(&config).collect::<Vec<_>>();
        assert_eq!(
            candidates,
            vec![
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 30, base_bits: 15 },
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 30, base_bits: 15 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 15 },
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 15 },
            ]
        );
    }

    #[test]
    fn average_selector_uses_minimum_crt_spacing_not_smallest_tower() {
        let spacing =
            minimum_crt_spacing(&[BigUint::from(2u8), BigUint::from(3u8), BigUint::from(5u8)])
                .unwrap();
        assert_eq!(spacing, BigUint::from(6u8));
        assert!(spacing > BigUint::from(2u8));
    }

    #[test]
    fn average_policy_is_derived_once_from_search_config() {
        let mut config = SearchConfig::reviewed();
        config.security_bits = 37;
        let policy = average_case_config(&config).unwrap();
        assert_eq!(policy.failure_exponent, 37);
        assert!(policy.allow_average_acceptance);
        assert_eq!(policy, average_case_config(&config).unwrap());
    }

    #[test]
    fn average_boundary_enumerates_only_strictly_valid_doubled_masks() {
        let base = BigUint::from(2u8);
        let spacing = BigUint::from(6u8);
        let valid = (1..=4)
            .map(|digits| (digits, base.pow(digits) - BigUint::one()))
            .take_while(|(_, d2)| d2 * 2u8 < spacing.clone() * 2u8)
            .map(|(digits, _)| digits)
            .collect::<Vec<_>>();
        assert_eq!(valid, vec![1, 2]);
        assert!(base.pow(3) - BigUint::one() >= spacing);
    }

    #[test]
    fn phase_two_skips_qbit_overflow_per_crt_base_pair() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 62..=62;
        config.log_ring_dimensions = 16..=16;
        config.crt_base_bits_grid = vec![(32, 16), (30, 15), (28, 14)];
        assert_eq!(
            candidates(&config).collect::<Vec<_>>(),
            vec![
                Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 30, base_bits: 15 },
                Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 28, base_bits: 14 },
            ]
        );
        config.crt_depths = 63..=63;
        config.crt_base_bits_grid = vec![(32, 16)];
        assert!(candidates(&config).next().is_none());
    }

    #[test]
    fn lookup_overlay_applies_the_configured_tiny_profile() {
        let values = std::collections::BTreeMap::from([
            ("MXX_POWER_LUT_REFRESH_SECURITY_BITS", "1"),
            ("MXX_POWER_LUT_REFRESH_SPARSE_LWR_UNIVERSE_GRID", "4"),
            ("MXX_POWER_LUT_REFRESH_SPARSE_LWR_WEIGHT", "1"),
            ("MXX_POWER_LUT_REFRESH_SPARSE_LWR_MODULUS", "4"),
            ("MXX_POWER_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", "2"),
            ("MXX_POWER_LUT_REFRESH_LUT_WIDTH", "8"),
            ("MXX_POWER_LUT_REFRESH_CRT_BITS", "4"),
            ("MXX_POWER_LUT_REFRESH_BASE_BITS", "2"),
            ("MXX_POWER_LUT_REFRESH_MIN_CRT_DEPTH", "1"),
            ("MXX_POWER_LUT_REFRESH_MAX_CRT_DEPTH", "1"),
            ("MXX_POWER_LUT_REFRESH_MIN_LOG_RING_DIMENSION", "5"),
            ("MXX_POWER_LUT_REFRESH_MAX_LOG_RING_DIMENSION", "5"),
        ]);
        let config =
            SearchConfig::from_lookup(|name| Ok(values.get(name).map(|value| (*value).to_owned())))
                .unwrap();
        assert_eq!(config.security_bits, 1);
        assert_eq!(config.sparse_lwr_universe_grid, vec![4]);
        assert_eq!(config.sparse_lwr_weight, 1);
        assert_eq!(config.sparse_lwr_modulus, 4);
        assert_eq!(config.sparse_lwr_output_modulus, 2);
        assert_eq!(config.lut_width, 8);
        assert_eq!(config.crt_bits, 4);
        assert_eq!(config.base_bits, 2);
        assert_eq!(config.crt_depths, 1..=1);
        assert_eq!(config.log_ring_dimensions, 5..=5);
    }

    #[test]
    fn lookup_overlay_keeps_unset_values_from_reviewed_profile() {
        let config = SearchConfig::from_lookup(|name| {
            Ok((name == "MXX_POWER_LUT_REFRESH_SECURITY_BITS").then(|| "1".to_owned()))
        })
        .unwrap();
        assert_eq!(config.security_bits, 1);
        assert_eq!(config.crt_bits, SearchConfig::reviewed().crt_bits);
        assert_eq!(config.base_bits, SearchConfig::reviewed().base_bits);
        assert_eq!(
            config.sparse_lwr_universe_grid,
            SearchConfig::reviewed().sparse_lwr_universe_grid
        );
    }

    #[test]
    fn lookup_overlay_rejects_invalid_ranges() {
        let invalid_range = std::collections::BTreeMap::from([
            ("MXX_POWER_LUT_REFRESH_MIN_CRT_DEPTH", "2"),
            ("MXX_POWER_LUT_REFRESH_MAX_CRT_DEPTH", "1"),
        ]);
        let error = SearchConfig::from_lookup(|name| {
            Ok(invalid_range.get(name).map(|value| (*value).to_owned()))
        })
        .unwrap_err();
        assert!(error.contains("CRT depth range"));
    }

    #[test]
    fn lookup_overlay_propagates_non_unicode_lookup_failure() {
        let error = SearchConfig::from_lookup(|name| {
            if name == "MXX_POWER_LUT_REFRESH_SECURITY_BITS" {
                Err("MXX_POWER_LUT_REFRESH_SECURITY_BITS must contain valid UTF-8".to_owned())
            } else {
                Ok(None)
            }
        })
        .unwrap_err();
        assert!(error.contains("valid UTF-8"));
    }

    #[test]
    fn mocked_security_and_checker_select_first_minimal_candidate() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=3;
        config.log_ring_dimensions = 5..=5;
        config.crt_base_bits_grid = vec![(32, 16)];
        let sparse_profile = mock_sparse_profile(&config);
        let result = search_with_hooks(
            &config,
            &sparse_profile,
            |candidate| Ok(mock_prepared(candidate)),
            |candidate| Ok(candidate.crt_depth as u64 * 100),
            |prepared| Ok(prepared.candidate.crt_depth >= 2),
        )
        .unwrap();
        assert_eq!(
            result.candidate,
            Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 }
        );
    }

    #[test]
    fn phase_one_exhausts_ordered_grid_before_selecting_primary_profile() {
        let mut config = SearchConfig::reviewed();
        config.security_bits = 100;
        let mut calls = Vec::new();
        let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
            calls.push(universe);
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        assert_eq!(calls, vec![450, 451]);
        assert_eq!(selected.universe, 451);
        assert_eq!(selected.q_l, 16);
        assert_eq!(selected.tier, SparseLwrSecurityTier::Primary100);
        assert_eq!(selected.sparse_lwr_security_bits, 100);
    }

    #[test]
    fn phase_one_rejects_an_estimator_result_that_does_not_match_declared_evidence() {
        let config = SearchConfig::reviewed();
        let error = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(128)).unwrap_err();
        assert!(error.contains("expected declared"));
    }

    #[test]
    fn qualified_profiles_preserve_all_primary_rows_for_phase_two_advancement() {
        let config = SearchConfig::reviewed();
        let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let profiles = qualified_sparse_lwr_profiles(&selected);
        assert_eq!(profiles.iter().map(|profile| profile.q_l).collect::<Vec<_>>(), [16]);
        assert!(profiles.iter().all(|profile| profile.tier == SparseLwrSecurityTier::Primary100));
    }

    #[test]
    fn phase_two_caches_sparse_security_and_defers_prepare_until_bgg_floor() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=1;
        config.log_ring_dimensions = 5..=6;
        config.crt_base_bits_grid = vec![(32, 16)];
        let sparse_profile = mock_sparse_profile(&config);
        let mut security_calls = Vec::new();
        let mut prepare_calls = Vec::new();
        let mut checker_calls = Vec::new();
        let result = search_with_hooks(
            &config,
            &sparse_profile,
            |candidate| {
                prepare_calls.push(candidate);
                Ok(mock_prepared(candidate))
            },
            |candidate| {
                security_calls.push(candidate);
                Ok(if candidate.log_ring_dimension == 5 { 99 } else { 100 })
            },
            |prepared| {
                checker_calls.push(prepared.candidate);
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(security_calls.len(), 2);
        assert_eq!(
            prepare_calls,
            vec![Candidate { crt_depth: 1, log_ring_dimension: 6, crt_bits: 32, base_bits: 16 }]
        );
        assert_eq!(checker_calls, prepare_calls);
        assert_eq!(result.sparse_lwr_universe, config.sparse_lwr_universe);
        assert_eq!(result.sparse_lwr_weight, config.sparse_lwr_weight);
    }

    #[test]
    fn phase_two_skips_exactly_infeasible_preparation_but_propagates_fatal_errors() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=2;
        config.log_ring_dimensions = 5..=5;
        config.crt_base_bits_grid = vec![(32, 16)];
        let sparse_profile = mock_sparse_profile(&config);
        let mut prepared_candidates = Vec::new();
        let result = search_with_hooks(
            &config,
            &sparse_profile,
            |candidate| {
                prepared_candidates.push(candidate);
                if candidate.crt_depth == 1 {
                    Err(CandidatePreparationError::Infeasible(
                        "strict rounding is infeasible".to_owned(),
                    ))
                } else {
                    Ok(mock_prepared(candidate))
                }
            },
            |_| Ok(128),
            |_| Ok(true),
        )
        .unwrap();
        assert_eq!(
            prepared_candidates,
            vec![
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
            ]
        );
        assert_eq!(
            result.candidate,
            Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 }
        );

        let fatal = search_with_hooks(
            &config,
            &sparse_profile,
            |_| Err(CandidatePreparationError::Fatal("invalid setup identity".to_owned())),
            |_| Ok(128),
            |_| Ok(true),
        )
        .unwrap_err();
        assert_eq!(fatal, "invalid setup identity");
    }

    #[test]
    fn phase_one_rejects_a_non_ascending_grid() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_phase1_grid[1] = config.sparse_lwr_phase1_grid[0].clone();
        let error = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(200)).unwrap_err();
        assert!(error.contains("tuple grid") || error.contains("p == 2"));
    }

    fn checkpoint_test_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mxx-power-lut-{name}-{}.json", std::process::id()))
    }

    #[test]
    fn phase_one_checkpoint_reuses_an_exact_declaration() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_phase1_grid = config.sparse_lwr_phase1_grid[..2].to_vec();
        let path = checkpoint_test_path("phase1-reuse");
        let _ = std::fs::remove_file(&path);

        let mut first_calls = 0;
        let first = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            first_calls += 1;
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();

        let mut second_calls = 0;
        let second = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
            second_calls += 1;
            Err("an exact checkpoint must not rerun Phase 1".to_owned())
        })
        .unwrap();
        assert_eq!(first_calls, 2);
        assert_eq!(second_calls, 0);
        assert_eq!(first.universe, second.universe);
        assert_eq!(second.universe, 451);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn phase_one_checkpoint_rejects_mismatch_and_missing_path_starts_fresh_search() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_phase1_grid = config.sparse_lwr_phase1_grid[..2].to_vec();
        let path = checkpoint_test_path("phase1-mismatch");
        let _ = std::fs::remove_file(&path);

        let mut calls = 0;
        let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            calls += 1;
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(selected.universe, 451);

        let mut changed = config.clone();
        changed.phase1_estimator_cost_model = "ALT".to_owned();
        let error = load_or_search_phase1(&changed, Some(&path), |_, _, _, _| {
            Err("a declaration mismatch must not fall back to a fresh search".to_owned())
        })
        .unwrap_err();
        assert!(error.contains("does not match"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn phase_one_checkpoint_binds_tuple_tier_error_and_estimator_identity() {
        let config = SearchConfig::reviewed();
        let path = checkpoint_test_path("phase1-identity");
        let _ = std::fs::remove_file(&path);
        load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let mut changed = config.clone();
        changed.sparse_lwr_phase1_grid[0].q_l = 10;
        assert!(
            load_or_search_phase1(&changed, Some(&path), |universe, _, q_l, _| Ok(
                reviewed_estimator_bits(universe, q_l)
            ))
            .unwrap_err()
            .contains("does not match")
        );
        let mut checkpoint: Phase1Checkpoint =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        checkpoint.selected.tier = SparseLwrSecurityTier::Fallback100;
        std::fs::write(&path, serde_json::to_vec_pretty(&checkpoint).unwrap()).unwrap();
        assert!(
            load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| Ok(
                reviewed_estimator_bits(universe, q_l)
            ))
            .unwrap_err()
            .contains("final row")
        );
        std::fs::remove_file(&path).unwrap();
        load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let mut changed = config.clone();
        changed.phase1_estimator_shape_model = "primal".to_owned();
        assert!(
            load_or_search_phase1(&changed, Some(&path), |universe, _, q_l, _| Ok(
                reviewed_estimator_bits(universe, q_l)
            ))
            .unwrap_err()
            .contains("does not match")
        );
        let mut changed = config.clone();
        changed.phase1_fallback_security_bits = 101;
        assert!(
            load_or_search_phase1(&changed, Some(&path), |_, _, _, _| Ok(128))
                .unwrap_err()
                .contains("does not match")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn phase_one_checkpoint_round_trips_the_boundary_tuple() {
        for selected_universe in [451] {
            let mut config = SearchConfig::reviewed();
            config.sparse_lwr_phase1_grid = config
                .sparse_lwr_phase1_grid
                .into_iter()
                .filter(|tuple| tuple.nu == selected_universe)
                .collect();
            let path = checkpoint_test_path(&format!("phase1-selected-{selected_universe}"));
            let _ = std::fs::remove_file(&path);
            let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            assert_eq!(selected.universe, selected_universe);
            let reused = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
                Err("an exact checkpoint must not rerun Phase 1".to_owned())
            })
            .unwrap();
            assert_eq!(reused.tuple, selected.tuple);
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn accepted_phase_two_profile_persists_q16_and_rejects_out_of_grid_tampering() {
        let config = SearchConfig::reviewed();
        let path = checkpoint_test_path("phase2-accepted");
        let _ = std::fs::remove_file(&path);
        let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let mut order = Vec::new();
        let (q16, result) = search_qualified_profiles(&selected, |profile| {
            order.push(profile.q_l);
            Ok(SearchResult {
                candidate: Candidate {
                    crt_depth: 30,
                    log_ring_dimension: 16,
                    crt_bits: 32,
                    base_bits: 16,
                },
                achieved_security_bits: 128,
                bgg_rlwe_security_bits: 128,
                sparse_lwr_security_bits: profile.sparse_lwr_security_bits,
                raw_key_entropy_bits: profile.raw_key_entropy_bits,
                sparse_lwr_universe: profile.universe,
                sparse_lwr_weight: profile.weight,
                official_preimage_bound: "1".to_owned(),
                ring_dimension: 1 << 16,
                bucket_width: 1,
                pbc_attempts_used: 1,
                layout_id: "aa".repeat(32),
                program_id: "bb".repeat(32),
                checker_accepted: true,
                sparse_lwr_q_l: profile.q_l,
                sparse_lwr_p: profile.p,
                sparse_lwr_tier: profile.tier,
                estimator_commit: profile.estimator_commit.clone(),
                estimator_cost_model: profile.estimator_cost_model.clone(),
                estimator_shape_model: profile.estimator_shape_model.clone(),
            })
        })
        .unwrap();
        assert_eq!(order, vec![16]);
        assert_eq!(q16.q_l, 16);
        let candidate = result.candidate;
        persist_accepted_phase2_profile(&path, &config, &q16, candidate).unwrap();
        let checkpoint: Phase1Checkpoint =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        checkpoint.validate(&config).unwrap();
        assert_eq!(checkpoint.accepted_phase2.as_ref().unwrap().tuple.q_l, 16);
        assert_eq!(checkpoint.accepted_phase2.as_ref().unwrap().candidate, candidate);

        let mut tampered = checkpoint;
        tampered.accepted_phase2.as_mut().unwrap().candidate.crt_bits = 31;
        std::fs::write(&path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(
            Phase1Checkpoint::validate(&tampered, &config)
                .unwrap_err()
                .contains("outside the declared search grid")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn profile_advancement_propagates_noncanonical_errors() {
        let config = SearchConfig::reviewed();
        let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let error = search_qualified_profiles(&selected, |_| Err("fatal setup error".to_owned()))
            .unwrap_err();
        assert_eq!(error, "fatal setup error");
        let error = search_qualified_profiles(&selected, |_| {
            Err(format!("{NO_CANDIDATE_ERROR}: extra diagnostic"))
        })
        .unwrap_err();
        assert_eq!(error, format!("{NO_CANDIDATE_ERROR}: extra diagnostic"));
    }

    #[test]
    fn malformed_phase_one_checkpoint_fails_before_security_callback() {
        let config = SearchConfig::reviewed();
        let path = checkpoint_test_path("phase1-malformed");
        std::fs::write(&path, b"{not valid json").unwrap();

        let mut calls = 0;
        let error = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
            calls += 1;
            Ok(129)
        })
        .unwrap_err();
        assert!(error.contains("read Phase-1 checkpoint"));
        assert_eq!(calls, 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn result_serialization_is_redacted() {
        let result = SearchResult {
            candidate: Candidate {
                crt_depth: 2,
                log_ring_dimension: 5,
                crt_bits: 32,
                base_bits: 16,
            },
            achieved_security_bits: 128,
            bgg_rlwe_security_bits: 130,
            sparse_lwr_security_bits: 140,
            raw_key_entropy_bits: raw_key_entropy_bits(512, 64),
            sparse_lwr_universe: 512,
            sparse_lwr_weight: 64,
            official_preimage_bound: "123".to_owned(),
            ring_dimension: 32,
            bucket_width: 4,
            pbc_attempts_used: 1,
            layout_id: "aa".repeat(32),
            program_id: "bb".repeat(32),
            checker_accepted: true,
            sparse_lwr_q_l: 8,
            sparse_lwr_p: 2,
            sparse_lwr_tier: SparseLwrSecurityTier::Primary100,
            estimator_commit: REVIEWED_ESTIMATOR_COMMIT.to_owned(),
            estimator_cost_model: REVIEWED_ESTIMATOR_COST_MODEL.to_owned(),
            estimator_shape_model: REVIEWED_ESTIMATOR_SHAPE_MODEL.to_owned(),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("support"));
        assert!(!json.contains("selected"));
        assert!(json.contains("official_preimage_bound"));
    }

    #[test]
    fn persisted_report_declares_minimality_scope_and_evaluated_phase_one_rows() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 2..=3;
        config.log_ring_dimensions = 5..=6;
        let sparse_profile = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let result = SearchResult {
            candidate: Candidate {
                crt_depth: 2,
                log_ring_dimension: 5,
                crt_bits: 32,
                base_bits: 16,
            },
            achieved_security_bits: 128,
            bgg_rlwe_security_bits: 128,
            sparse_lwr_security_bits: sparse_profile.sparse_lwr_security_bits,
            raw_key_entropy_bits: sparse_profile.raw_key_entropy_bits,
            sparse_lwr_universe: sparse_profile.universe,
            sparse_lwr_weight: sparse_profile.weight,
            official_preimage_bound: "123".to_owned(),
            ring_dimension: 32,
            bucket_width: 4,
            pbc_attempts_used: 1,
            layout_id: "aa".repeat(32),
            program_id: "bb".repeat(32),
            checker_accepted: true,
            sparse_lwr_q_l: sparse_profile.q_l,
            sparse_lwr_p: sparse_profile.p,
            sparse_lwr_tier: sparse_profile.tier,
            estimator_commit: sparse_profile.estimator_commit.clone(),
            estimator_cost_model: sparse_profile.estimator_cost_model.clone(),
            estimator_shape_model: sparse_profile.estimator_shape_model.clone(),
        };
        let json = serde_json::to_string(&search_report(&config, &sparse_profile, result)).unwrap();

        assert!(json.contains("\"parameter_grid\":["));
        assert!(json.contains("\"universe_grid\":[451]"));
        assert!(json.contains(&format!("\"support_weight\":{}", sparse_profile.weight)));
        assert!(json.contains("\"q_l\":16"));
        assert!(json.contains("\"output_modulus\":2"));
        assert!(json.contains("\"lut_width\":512"));
        assert!(json.contains("\"security_target_bits\":100"));
        assert!(json.contains("\"sparse_secret_model\":\"SparseBinary\""));
        assert!(json.contains("\"sparse_error_model\":\"Uniform\""));
        assert!(json.contains("\"selected_error_lower\":-4"));
        assert!(json.contains("\"selected_error_upper\":3"));
        assert!(json.contains("\"exact_estimator\":true"));
        assert!(json.contains("\"evaluated\":["));
        assert!(json.contains("\"universe\":451"));
        assert!(json.contains("\"minimum_security_bits\""));
        assert!(json.contains("\"qualified\":true"));
        assert!(json.contains("\"selected_universe\":451"));
        assert!(json.contains("\"crt_depth_min\":2"));
        assert!(json.contains("\"crt_depth_max\":3"));
        assert!(json.contains("\"log_ring_dimension_min\":5"));
        assert!(json.contains("\"log_ring_dimension_max\":6"));
        assert!(
            json.contains("\"order\":\"base_bits_descending_then_crt_bits_descending_then_crt_depth_then_log_ring_dimension\"")
        );
        assert!(json.contains("\"crt_base_bits_grid\":[[32,16]"));
        assert!(!json.contains("selected_slots"));
        assert!(!json.contains("support_coordinates"));
    }

    #[test]
    fn sparse_lwr_estimator_arguments_use_exact_reviewed_model() {
        assert_eq!(sparse_lwr_error_bounds(16, 2).unwrap(), (-4, 3));
        assert_eq!(
            sparse_lwr_estimator_args(512, 64, 16, 2).unwrap(),
            vec![
                "512",
                "16",
                "--s-dist",
                r#"{"name":"SparseBinary","hw":64,"n":512}"#,
                "--e-dist",
                r#"{"name":"Uniform","a":-4,"b":3}"#,
                "--exact",
            ]
        );
        assert!(sparse_lwr_error_bounds(15, 2).is_err());
        assert!(sparse_lwr_error_bounds(16, 0).is_err());
    }

    #[test]
    fn reconstruction_coefficients_come_from_the_dcrt_parameters() {
        let dcrt = DCRTPolyParams::new(32, 2, 28, 27);
        let coefficients = exact_reconstruction_coefficients(&dcrt);
        assert_eq!(coefficients.len(), 2);
        assert!(dcrt.reconst_coeffs().iter().any(|value| value != &BigUint::from(1_u8)));
    }

    #[test]
    fn security_output_parser_is_strict_and_fail_closed() {
        assert_eq!(parse_security_bits(b"128\n"), Ok(128));
        assert!(parse_security_bits(b"\n128\n\n").is_ok());
        assert_eq!(parse_security_bits(b"Algorithm ... failed\nalgorithm-result\n128\n"), Ok(128));
        assert!(parse_security_bits(b"128\n129\n").is_err());
        assert!(parse_security_bits(b"security: 128\n").is_err());
        assert!(parse_security_bits(b"+128\n").is_err());
        assert!(parse_security_bits(b"18446744073709551616\n").is_err());
    }

    #[test]
    fn fresh_digit_selection_is_strict_and_finite() {
        assert_eq!(fresh_error_base_p_digit_count(&BigUint::from(32u8), &[32, 37]), Ok(1));
        assert!(fresh_error_base_p_digit_count(&BigUint::from(32u8), &[31, 32]).is_err());
    }

    #[test]
    fn mask_digit_selection_checks_domain_hiding_and_rounding_together() {
        // This small profile has an exact finite answer. The first digit
        // covers Q_L but cannot meet the joint hiding requirement; the second
        // digit is rejected by strict rounding, so the selector fails closed.
        let result = select_mask_base_p_digit_count(
            &BigUint::from(2u8),
            &BigUint::from(2u8),
            &BigUint::from(15u8),
            &[BigUint::from(3u8), BigUint::from(5u8)],
            1,
            1,
            &BigUint::from(2u8),
            &BigUint::one(),
            &BigUint::one(),
            &BigUint::zero(),
            &BigUint::zero(),
            0,
            1,
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn mask_digit_selection_returns_the_first_digit_after_hiding_threshold() {
        // The two CRT towers require ell_beta=14. d_m=1 gives M_m=64 and
        // covers Q_L=2, but the exact joint hiding requirement is 3584.
        // d_m=2 gives M_m=4096 and satisfies the strict rounding bound (the
        // smallest spacing is 9001).
        let q = BigUint::from(9_001u16) * BigUint::from(9_007u16);
        let result = select_mask_base_p_digit_count(
            &BigUint::from(64u8),
            &BigUint::from(2u8),
            &q,
            &[BigUint::from(9_001u16), BigUint::from(9_007u16)],
            1,
            14,
            &BigUint::from(4u8),
            &BigUint::one(),
            &BigUint::one(),
            &BigUint::zero(),
            &BigUint::zero(),
            0,
            1,
            1,
        )
        .unwrap();
        assert_eq!(result, 2);
    }

    #[test]
    fn mask_digit_selection_rejects_incomplete_or_non_coprime_crt() {
        let args = |full_modulus: BigUint, crt_moduli: Vec<BigUint>| {
            select_mask_base_p_digit_count(
                &BigUint::from(8u8),
                &BigUint::from(2u8),
                &full_modulus,
                &crt_moduli,
                1,
                1,
                &BigUint::from(2u8),
                &BigUint::one(),
                &BigUint::zero(),
                &BigUint::zero(),
                &BigUint::one(),
                0,
                1,
                1,
            )
        };
        assert!(args(BigUint::from(12u8), vec![BigUint::from(2u8), BigUint::from(3u8)]).is_err());
        assert!(
            args(
                BigUint::from(12u8),
                vec![BigUint::from(2u8), BigUint::from(2u8), BigUint::from(3u8)]
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "diagnostic-only large exact-noise profile"]
    fn emits_large_profile_sparse_prf_stage_diagnostics() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 62..=62;
        config.log_ring_dimensions = 16..=16;
        config.crt_bits = 32;
        config.base_bits = 1;
        let ring_dimension = 1usize << 16;
        let dcrt = DCRTPolyParams::new(ring_dimension as u32, 62, 32, 1);
        let q_bits = dcrt.modulus().bits();
        assert!(q_bits < 2_000, "diagnostic profile q bits = {q_bits}");
        info!(q_bits, ring_dimension, "large exact-noise diagnostic profile");
        let result = prepare_candidate(
            &config,
            Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 32, base_bits: 1 },
        );
        info!(accepted = result.is_ok(), "large exact-noise diagnostic completed");
    }
}
