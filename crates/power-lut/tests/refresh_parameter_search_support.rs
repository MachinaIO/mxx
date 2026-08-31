//! Small, deterministic helpers for the ignored Section-7 parameter search.
//!
//! The search is intentionally kept in the integration-test tree.  It is a
//! measurement harness, not a new Power-LUT parameter abstraction: production
//! code continues to use the ordinary `RefreshSetupParameters`, PBC, PRF
//! program, and correctness checker APIs.

use std::{fs, io::ErrorKind, ops::RangeInclusive, path::Path, process::Command};

use mxx_bgg::{BggSamplerLayout, PreimageCoefficientBound};
use mxx_ir_core::{IntExpr, RealExpr};
use mxx_power_lut::{
    pbc::{PbcParameters, PbcRootSeed, generate_key_layout},
    prf::{SparseLwrPrfProfile, SparseLwrPrfProgram},
    refresh::RefreshCompiler,
    refresh_setup::{
        RefreshParameterSimulationBundle, RefreshParameterSimulationRequest, RefreshSetupParameters,
    },
};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use serde::Serialize;
use tracing::info;

/// Runs the actual protocol-agnostic operational checker for every decoder
/// target exported by the refresh adapter.  The adapter owns the graph
/// linkage; this test-only bridge only converts its structural metadata into
/// the correctness crate's request types.
pub fn check_refresh_bundle(
    config: &SearchConfig,
    candidate: Candidate,
    prepared: &PreparedCandidate,
) -> Result<bool, String> {
    use mxx_correctness::{
        ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
        EndpointSemanticBinding, EndpointSpecId, OperationalDecoderKind, OperationalDecoderTarget,
        OutputRef, StageId,
        operational_noise::{
            OperationalCheckRequest, OperationalGadgetLayout,
            check_operational_noise_candidate_with_progress,
        },
        operational_protocol_from_graphs,
    };
    use mxx_dsl::{Bool, DslContext, IdealSpec};
    use std::collections::BTreeMap;

    let bundle = prepared.bundle.as_ref().ok_or_else(|| {
        "the production refresh checker requires a built simulation bundle".to_owned()
    })?;
    let dcrt = DCRTPolyParams::new(
        prepared.ring_dimension as u32,
        candidate.crt_depth,
        config.crt_bits,
        config.base_bits,
    );
    let (crt_moduli, crt_bits, _) = dcrt.to_crt();
    let mut exact = BTreeMap::new();
    for (name, metadata) in bundle.matrix_input_metadata() {
        exact.insert(
            name.clone(),
            mxx_correctness::ExactMatrixInputMetadata {
                canonical_coefficient_exclusive_upper_bound: metadata
                    .canonical_coefficient_exclusive_upper_bound
                    .clone(),
                is_constant_polynomial: metadata.is_constant_polynomial,
            },
        );
    }
    let decoder_stage = StageId(bundle.entrypoint().to_owned());
    let layout = OperationalGadgetLayout {
        params_id: format!("refresh-parameter-search-{candidate:?}"),
        ring_dimension: prepared.ring_dimension,
        smallest_crt_modulus: *crt_moduli
            .iter()
            .min()
            .ok_or_else(|| "DCRT layout has no CRT moduli".to_owned())?,
        crt_moduli,
        crt_bits,
        base_bits: config.base_bits as usize,
        base: BigInt::from(1_u8) << config.base_bits,
        regular_digit_count: dcrt.modulus_digits(),
        small_digit_count: crt_bits.div_ceil(config.base_bits as usize),
    };
    for target in bundle.decoder_targets() {
        // The correctness registry accepts one ThresholdDecode endpoint spec
        // per protocol declaration.  Build a declaration for this exact
        // target so its residual output, decoder node, decoded output, and
        // semantic anchor cannot be confused with another column.
        let target_decoder = OperationalDecoderTarget {
            target_id: target.target_id.clone(),
            residual_stage: decoder_stage.clone(),
            residual_output: target.residual_output_name.clone(),
            decoder_stage: decoder_stage.clone(),
            decoder_node: target.decoder_node,
            kind: OperationalDecoderKind::ThresholdDecode {
                plaintext_modulus: target.plaintext_modulus.clone(),
            },
        };
        let decoded_output_name = target.decoded_output_name.clone();
        let decoder_anchor = target.decoder_anchor.clone();
        let ideal = IdealSpec::new(
            DslContext::new("refresh-parameter-search-ideal")
                .bool_output(&decoded_output_name, Bool::constant(false))
                .map_err(|error| error.to_string())?
                .build()
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let protocol = operational_protocol_from_graphs(
            vec![
                ("selector".to_owned(), bundle.selector_graph()),
                ("preprocessing".to_owned(), bundle.preprocessing_graph()),
                ("verification".to_owned(), bundle.verification_graph()),
            ],
            bundle.entrypoint(),
            &exact,
            &BTreeMap::new(),
            |closed| {
                closed.ideal = ideal;
                closed.comparator = ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint: EndpointSpecId::ThresholdDecode,
                        actual_input: decoded_output_name.clone(),
                        ideal_input: decoded_output_name.clone(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                };
                closed.endpoints = EndpointAnchors {
                    entries: vec![EndpointAnchor {
                        spec: EndpointSpecId::ThresholdDecode,
                        stage: decoder_stage.clone(),
                        semantic_anchor: decoder_anchor,
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: decoder_stage.clone(),
                            output: decoded_output_name.clone(),
                        },
                        ideal_output: decoded_output_name.clone(),
                    }],
                };
                closed.operational_decoder_targets = vec![target_decoder];
                closed.endpoint_specs = vec![EndpointSpecId::ThresholdDecode];
            },
        )
        .map_err(|error| error.to_string())?;
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: vec![layout.clone()],
            target_id: target.target_id.clone(),
        };
        let report =
            check_operational_noise_candidate_with_progress(&protocol, &request, |event| {
                info!(
                    target = %target.target_id,
                    phase = ?event.phase,
                    event = ?event.event,
                    processed = event.processed,
                    total_or_discovered = ?event.total_or_discovered,
                    "refresh parameter search checker progress"
                );
            })
            .map_err(|error| error.to_string())?;
        if !report.accepted {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The fixed profile requested for the first Power-LUT refresh search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Inclusive CRT-depth range. The outer search order is ascending depth.
    pub crt_depths: RangeInclusive<usize>,
    /// Inclusive log2(ring-dimension) range. The inner order is ascending.
    pub log_ring_dimensions: RangeInclusive<usize>,
    pub security_bits: u64,
    pub crt_bits: usize,
    pub base_bits: u32,
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
    pub lut_width: usize,
    pub pbc_max_attempts: u32,
    pub one_nontrivial_refresh_round: bool,
    pub plaintext_modulus: usize,
}

impl SearchConfig {
    /// Returns the reviewed finite grid. Phase 2 is searched in ascending CRT
    /// depth 30..=40 and log2(ring dimension) 15..=16. `base_bits = 27` gives
    /// exactly two base digits for the 28-bit CRT primes.
    pub fn reviewed() -> Self {
        Self {
            crt_depths: 30..=40,
            log_ring_dimensions: 15..=16,
            security_bits: 128,
            crt_bits: 28,
            base_bits: 27,
            secret_dimension: 2,
            error_sigma: 4.0,
            decoder_sigma: 4.578,
            sparse_lwr_universe: 512,
            sparse_lwr_weight: 64,
            sparse_lwr_universe_grid: vec![
                512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992,
                1024,
            ],
            sparse_lwr_modulus: 16,
            sparse_lwr_output_modulus: 2,
            lut_width: 32,
            pbc_max_attempts: 16,
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
        if log_ring_start > log_ring_end || log_ring_end >= usize::BITS as usize {
            return Err("log ring-dimension range must be ascending and shiftable".to_owned());
        }
        if self.security_bits == 0 || self.crt_bits == 0 || self.base_bits == 0 {
            return Err("security, CRT, and base bit widths must be positive".to_owned());
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
        sparse_lwr_error_bounds(self.sparse_lwr_modulus, self.sparse_lwr_output_modulus)?;
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

/// One point in the finite search grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Candidate {
    pub crt_depth: usize,
    pub log_ring_dimension: usize,
}

/// The public, non-secret facts retained for one prepared candidate.
pub struct PreparedCandidate {
    pub candidate: Candidate,
    pub ring_dimension: usize,
    pub bucket_width: usize,
    pub official_preimage_bound: BigInt,
    pub layout_id: [u8; 32],
    pub program_id: [u8; 32],
    /// Present for the production search; omitted by mocked ordering tests.
    pub bundle: Option<RefreshParameterSimulationBundle>,
}

/// Security values and generic-checker result for the selected point.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SearchResult {
    pub candidate: Candidate,
    pub achieved_security_bits: u64,
    pub bgg_rlwe_security_bits: u64,
    pub sparse_lwr_security_bits: u64,
    pub mitm_security_bits: u64,
    pub sparse_lwr_universe: usize,
    pub sparse_lwr_weight: usize,
    pub official_preimage_bound: String,
    pub ring_dimension: usize,
    pub bucket_width: usize,
    pub layout_id: String,
    pub program_id: String,
    pub checker_accepted: bool,
}

/// One aggregate, non-secret row persisted for a Phase-1 universe that was
/// actually evaluated.  It records enough evidence to audit the first
/// qualifying grid point without exposing the sparse support or schedule.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
pub struct SparseLwrEvaluation {
    pub universe: usize,
    pub weight: usize,
    pub sparse_lwr_security_bits: u64,
    pub mitm_security_bits: u64,
    pub minimum_security_bits: u64,
    pub qualified: bool,
}

/// The Phase-1 result that is reused unchanged by every DCRT candidate.
///
/// Keeping the estimator outputs with the selected universe size makes the
/// minimality claim auditable: Phase 2 cannot accidentally rerun the sparse
/// estimator with a candidate-dependent value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
pub struct SelectedSparseLwrProfile {
    pub universe: usize,
    pub weight: usize,
    pub sparse_lwr_security_bits: u64,
    pub mitm_security_bits: u64,
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
pub struct Phase1Declaration {
    pub universe_grid: Vec<usize>,
    pub support_weight: usize,
    pub q_l: usize,
    pub output_modulus: usize,
    pub lut_width: usize,
    pub security_target_bits: u64,
    pub sparse_secret_model: String,
    pub sparse_error_model: String,
    pub sparse_error_lower: i64,
    pub sparse_error_upper: i64,
    pub exact_estimator: bool,
    pub support_recovery_model: String,
}

impl Phase1Declaration {
    /// Derive the complete public declaration from the current test config.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        let (sparse_error_lower, sparse_error_upper) =
            sparse_lwr_error_bounds(config.sparse_lwr_modulus, config.sparse_lwr_output_modulus)?;
        Ok(Self {
            universe_grid: config.sparse_lwr_universe_grid.clone(),
            support_weight: config.sparse_lwr_weight,
            q_l: config.sparse_lwr_modulus,
            output_modulus: config.sparse_lwr_output_modulus,
            lut_width: config.lut_width,
            security_target_bits: config.security_bits,
            sparse_secret_model: "SparseBinary".to_owned(),
            sparse_error_model: "Uniform".to_owned(),
            sparse_error_lower,
            sparse_error_upper,
            exact_estimator: true,
            support_recovery_model: "binomial_mitm_floor".to_owned(),
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
pub struct Phase1Checkpoint {
    pub version: u32,
    pub declaration: Phase1Declaration,
    pub evaluated: Vec<SparseLwrEvaluation>,
    pub selected: SelectedSparseLwrProfile,
}

impl Phase1Checkpoint {
    pub fn from_selection(
        config: &SearchConfig,
        selected: &SelectedSparseLwrProfile,
    ) -> Result<Self, String> {
        let declaration = Phase1Declaration::from_config(config)?;
        let checkpoint = Self {
            version: PHASE1_CHECKPOINT_VERSION,
            declaration,
            evaluated: selected.evaluations.clone(),
            selected: selected.clone(),
        };
        checkpoint.validate(config)?;
        Ok(checkpoint)
    }

    /// Validate an on-disk checkpoint before allowing it to skip Phase 1.
    pub fn validate(&self, config: &SearchConfig) -> Result<(), String> {
        if self.version != PHASE1_CHECKPOINT_VERSION {
            return Err(format!("unsupported Phase-1 checkpoint version {}", self.version));
        }
        let expected = Phase1Declaration::from_config(config)?;
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
        if self.evaluated.is_empty() {
            return Err("Phase-1 checkpoint has no evaluated rows".to_owned());
        }

        let selected_index = self
            .evaluated
            .iter()
            .position(|row| row.universe == self.selected.universe)
            .ok_or_else(|| "Phase-1 checkpoint selected universe is not evaluated".to_owned())?;
        if selected_index + 1 != self.evaluated.len() {
            return Err("Phase-1 checkpoint contains rows after the selected universe".to_owned());
        }
        for (index, row) in self.evaluated.iter().enumerate() {
            let expected_universe = *config
                .sparse_lwr_universe_grid
                .get(index)
                .ok_or_else(|| "Phase-1 checkpoint has too many evaluated rows".to_owned())?;
            if row.universe != expected_universe ||
                row.weight != config.sparse_lwr_weight ||
                row.qualified != (row.minimum_security_bits >= config.security_bits) ||
                row.minimum_security_bits !=
                    row.sparse_lwr_security_bits.min(row.mitm_security_bits)
            {
                return Err("Phase-1 checkpoint contains an inconsistent evaluated row".to_owned());
            }
            if index < selected_index && row.qualified {
                return Err("Phase-1 checkpoint skips an earlier qualified universe".to_owned());
            }
        }
        let selected_row = &self.evaluated[selected_index];
        if !selected_row.qualified ||
            self.selected.weight != selected_row.weight ||
            self.selected.sparse_lwr_security_bits != selected_row.sparse_lwr_security_bits ||
            self.selected.mitm_security_bits != selected_row.mitm_security_bits
        {
            return Err(
                "Phase-1 checkpoint selected profile does not match its final row".to_owned()
            );
        }
        Ok(())
    }
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Phase1SearchReport {
    /// Complete public input and security-model declaration for these rows.
    pub declaration: Phase1Declaration,
    pub universe_grid: Vec<usize>,
    pub support_weight: usize,
    pub q_l: usize,
    pub output_modulus: usize,
    pub lut_width: usize,
    pub security_target_bits: u64,
    pub evaluated: Vec<SparseLwrEvaluation>,
    pub selected_universe: usize,
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
}

/// Persisted report wrapper for the parameter-search test.
///
/// `result` is deliberately accompanied by the declared search domains and
/// all evaluated Phase-1 rows.  This prevents a reader from mistaking the
/// selected point for a global minimum when it is only minimal in the
/// explicitly declared finite grids.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
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
            declaration: Phase1Declaration::from_config(config)
                .expect("reviewed Phase-1 config must have a valid error model"),
            universe_grid: config.sparse_lwr_universe_grid.clone(),
            support_weight: config.sparse_lwr_weight,
            q_l: config.sparse_lwr_modulus,
            output_modulus: config.sparse_lwr_output_modulus,
            lut_width: config.lut_width,
            security_target_bits: config.security_bits,
            evaluated: sparse_profile.evaluations.clone(),
            selected_universe: sparse_profile.universe,
        },
        phase2: Phase2SearchReport {
            crt_depth_min: *config.crt_depths.start(),
            crt_depth_max: *config.crt_depths.end(),
            log_ring_dimension_min: *config.log_ring_dimensions.start(),
            log_ring_dimension_max: *config.log_ring_dimensions.end(),
            order: "crt_depth_then_log_ring_dimension",
            security_target_bits: config.security_bits,
        },
        result,
    }
}

/// Selects the first qualifying sparse-LWR universe size from the explicit
/// ascending Phase-1 grid.  The callback is invoked exactly once per grid
/// value; the support-recovery MITM floor is computed once beside it.
pub fn select_sparse_lwr_profile<Security>(
    config: &SearchConfig,
    mut sparse_security: Security,
) -> Result<SelectedSparseLwrProfile, String>
where
    Security: FnMut(usize, usize, usize, usize) -> Result<u64, String>,
{
    if config.sparse_lwr_universe_grid.windows(2).any(|window| window[0] >= window[1]) {
        return Err("sparse-LWR Phase-1 universe grid must be strictly ascending".to_owned());
    }
    let mut evaluations = Vec::new();
    for &universe in &config.sparse_lwr_universe_grid {
        let sparse_bits = sparse_security(
            universe,
            config.sparse_lwr_weight,
            config.sparse_lwr_modulus,
            config.sparse_lwr_output_modulus,
        )?;
        let mitm_bits = sparse_support_mitm_bits(universe, config.sparse_lwr_weight);
        let achieved = sparse_bits.min(mitm_bits);
        let qualified = achieved >= config.security_bits;
        evaluations.push(SparseLwrEvaluation {
            universe,
            weight: config.sparse_lwr_weight,
            sparse_lwr_security_bits: sparse_bits,
            mitm_security_bits: mitm_bits,
            minimum_security_bits: achieved,
            qualified,
        });
        info!(
            sparse_lwr_universe = universe,
            sparse_lwr_weight = config.sparse_lwr_weight,
            sparse_lwr_security_bits = sparse_bits,
            mitm_security_bits = mitm_bits,
            achieved_security_bits = achieved,
            "evaluated sparse-LWR Phase-1 profile"
        );
        if qualified {
            return Ok(SelectedSparseLwrProfile {
                universe,
                weight: config.sparse_lwr_weight,
                sparse_lwr_security_bits: sparse_bits,
                mitm_security_bits: mitm_bits,
                evaluations,
            });
        }
    }
    Err(format!(
        "no sparse-LWR profile in the declared nu grid passed the {}-bit security floor",
        config.security_bits
    ))
}

fn exact_reconstruction_coefficients(dcrt: &DCRTPolyParams) -> Vec<IntExpr> {
    dcrt.reconst_coeffs()
        .into_iter()
        .map(|value| BigInt::from_biguint(num_bigint::Sign::Plus, value).into())
        .collect()
}

/// Enumerates the grid in the exact order used for minimality claims.
pub fn candidates(config: &SearchConfig) -> impl Iterator<Item = Candidate> + '_ {
    config.crt_depths.clone().flat_map(|crt_depth| {
        config
            .log_ring_dimensions
            .clone()
            .map(move |log_ring_dimension| Candidate { crt_depth, log_ring_dimension })
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
    Prepare: FnMut(Candidate) -> Result<PreparedCandidate, String>,
    Security: FnMut(Candidate) -> Result<u64, String>,
    Checker: FnMut(&PreparedCandidate) -> Result<bool, String>,
{
    if sparse_profile.universe != config.sparse_lwr_universe ||
        sparse_profile.weight != config.sparse_lwr_weight
    {
        return Err(
            "selected sparse-LWR profile does not match the frozen search configuration".to_owned()
        );
    }
    for candidate in candidates(config) {
        info!(
            crt_depth = candidate.crt_depth,
            log_ring_dimension = candidate.log_ring_dimension,
            "evaluating Power-LUT refresh search candidate"
        );
        let bgg = security(candidate)?;
        let achieved =
            bgg.min(sparse_profile.sparse_lwr_security_bits).min(sparse_profile.mitm_security_bits);
        if achieved < config.security_bits {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                achieved_security_bits = achieved,
                "candidate rejected by security floor"
            );
            continue;
        }
        let prepared = prepare(candidate)?;
        if !checker(&prepared)? {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                "candidate rejected by operational checker"
            );
            continue;
        }
        return Ok(SearchResult {
            candidate,
            achieved_security_bits: achieved,
            bgg_rlwe_security_bits: bgg,
            sparse_lwr_security_bits: sparse_profile.sparse_lwr_security_bits,
            mitm_security_bits: sparse_profile.mitm_security_bits,
            sparse_lwr_universe: sparse_profile.universe,
            sparse_lwr_weight: sparse_profile.weight,
            official_preimage_bound: prepared.official_preimage_bound.to_string(),
            ring_dimension: prepared.ring_dimension,
            bucket_width: prepared.bucket_width,
            layout_id: hex(prepared.layout_id),
            program_id: hex(prepared.program_id),
            checker_accepted: true,
        });
    }
    Err("no candidate in the declared finite grid passed security and correctness".to_owned())
}

/// Builds the concrete public setup and ordinary sparse-LWR program for one
/// candidate.  It does not execute a backend, sample artifacts, or perform a
/// round trip; those are intentionally outside this symbolic search.
pub fn prepare_candidate(
    config: &SearchConfig,
    candidate: Candidate,
) -> Result<PreparedCandidate, String> {
    let ring_dimension = 1usize
        .checked_shl(candidate.log_ring_dimension as u32)
        .ok_or_else(|| "ring dimension shift overflow".to_owned())?;
    let dcrt = DCRTPolyParams::new(
        ring_dimension as u32,
        candidate.crt_depth,
        config.crt_bits,
        config.base_bits,
    );
    let (crt_moduli, _, _) = dcrt.to_crt();
    let modulus = BigInt::from_biguint(num_bigint::Sign::Plus, dcrt.modulus().as_ref().clone());
    let layout = BggSamplerLayout {
        modulus: modulus.clone().into(),
        ring_dimension: ring_dimension.into(),
        secret_dimension: config.secret_dimension,
        digit_count: dcrt.modulus_digits(),
        gadget_base: (BigInt::from(1_u8) << config.base_bits).into(),
    };
    let refresh = RefreshCompiler {
        full_modulus: modulus.clone().into(),
        crt_plaintext_moduli: crt_moduli.iter().map(|value| BigInt::from(*value).into()).collect(),
        reconstruction_coefficients: exact_reconstruction_coefficients(&dcrt),
    };
    let setup = RefreshSetupParameters::new(
        [0x52; 32],
        2,
        config.secret_dimension,
        1,
        layout.digit_count,
        config.lut_width,
        layout,
        refresh,
        RealExpr::from_f64_exact(config.decoder_sigma)
            .map_err(|error| format!("decoder sigma: {error}"))?,
        "refresh-parameter-search",
    );
    if setup.decoder_preimage_bound != PreimageCoefficientBound::Official {
        return Err("refresh setup did not retain the official bound policy".to_owned());
    }
    let official_preimage_bound = setup
        .resolve_decoder_preimage_bound()
        .map_err(|error| format!("official decoder bound: {error}"))?
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|error| format!("official decoder bound evaluation: {error}"))?;

    let pbc_parameters =
        PbcParameters::conservative(config.sparse_lwr_universe, config.sparse_lwr_weight);
    if pbc_parameters.max_seed_attempts != config.pbc_max_attempts {
        return Err("conservative PBC profile does not use the reviewed retry count".to_owned());
    }
    let support = (0..config.sparse_lwr_weight)
        .map(|index| index.checked_mul(7).ok_or_else(|| "support overflow".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    let generated = generate_key_layout(&pbc_parameters, PbcRootSeed([0x19; 32]), &support)
        .map_err(|error| format!("PBC layout generation: {error}"))?;
    let profile = SparseLwrPrfProfile::new(
        config.sparse_lwr_modulus,
        config.sparse_lwr_output_modulus,
        config.lut_width,
        ring_dimension,
    )
    .map_err(|error| format!("sparse-LWR profile: {error}"))?;
    let program = SparseLwrPrfProgram::new(profile.clone(), generated.public_layout.bucket_width)
        .map_err(|error| format!("sparse-LWR program: {error}"))?;
    if !config.one_nontrivial_refresh_round || config.plaintext_modulus != 2 {
        return Err("the reviewed refresh profile was changed".to_owned());
    }
    let bucket_width = generated.public_layout.bucket_width;
    let layout_id = generated.public_layout.layout_id.0;
    let program_id = *program.id().as_bytes();
    let ring = setup.layout.ring();
    let expected_plaintext = ring.polynomial([BigInt::from(0_u8).into()]);
    let bundle = RefreshParameterSimulationRequest::new(
        setup,
        profile,
        generated,
        [0x4b; 32],
        RealExpr::from_f64_exact(config.error_sigma)
            .map_err(|error| format!("encoding sigma: {error}"))?,
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
        official_preimage_bound,
        layout_id,
        program_id,
        bundle: Some(bundle),
    })
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

/// Estimates the lifted sparse-LWR marginal security for the reviewed
/// sparse-binary secret model.  Support recovery is deliberately estimated
/// independently by [`sparse_support_mitm_bits`] and combined by `min`.
pub fn sparse_lwr_security_bits(
    universe: usize,
    weight: usize,
    q_l: usize,
    p: usize,
) -> Result<u64, String> {
    run_lattice_estimator(sparse_lwr_estimator_args(universe, weight, q_l, p)?)
}

/// Conservative meet-in-the-middle floor for sparse binary support recovery.
pub fn sparse_support_mitm_bits(universe: usize, weight: usize) -> u64 {
    let numerator = (0..weight)
        .fold(BigUint::from(1_u8), |value, index| value * BigUint::from(universe - index));
    let denominator =
        (1..=weight).fold(BigUint::from(1_u8), |value, index| value * BigUint::from(index));
    let combinations = numerator / denominator;
    combinations.bits().saturating_sub(1) / 2
}

pub fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_prepared(candidate: Candidate) -> PreparedCandidate {
        PreparedCandidate {
            candidate,
            ring_dimension: 1 << candidate.log_ring_dimension,
            bucket_width: 3,
            official_preimage_bound: 10.into(),
            layout_id: [candidate.crt_depth as u8; 32],
            program_id: [candidate.log_ring_dimension as u8; 32],
            bundle: None,
        }
    }

    fn mock_sparse_profile(config: &SearchConfig) -> SelectedSparseLwrProfile {
        SelectedSparseLwrProfile {
            universe: config.sparse_lwr_universe,
            weight: config.sparse_lwr_weight,
            sparse_lwr_security_bits: 200,
            mitm_security_bits: 200,
            evaluations: Vec::new(),
        }
    }

    #[test]
    fn grid_is_lexicographic_depth_then_ring_dimension() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 2..=3;
        config.log_ring_dimensions = 5..=6;
        assert_eq!(
            candidates(&config).collect::<Vec<_>>(),
            vec![
                Candidate { crt_depth: 2, log_ring_dimension: 5 },
                Candidate { crt_depth: 2, log_ring_dimension: 6 },
                Candidate { crt_depth: 3, log_ring_dimension: 5 },
                Candidate { crt_depth: 3, log_ring_dimension: 6 },
            ]
        );
    }

    #[test]
    fn reviewed_grid_declares_phase_two_minimality_scope() {
        let config = SearchConfig::reviewed();
        assert_eq!(config.crt_depths, 30..=40);
        assert_eq!(config.log_ring_dimensions, 15..=16);
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
        let sparse_profile = mock_sparse_profile(&config);
        let result = search_with_hooks(
            &config,
            &sparse_profile,
            |candidate| Ok(mock_prepared(candidate)),
            |candidate| Ok(candidate.crt_depth as u64 * 100),
            |prepared| Ok(prepared.candidate.crt_depth >= 2),
        )
        .unwrap();
        assert_eq!(result.candidate, Candidate { crt_depth: 2, log_ring_dimension: 5 });
    }

    #[test]
    fn phase_one_uses_ascending_grid_and_stops_at_first_qualified_profile() {
        let mut config = SearchConfig::reviewed();
        config.security_bits = 128;
        config.sparse_lwr_universe_grid = vec![512, 544, 576, 608];
        let mut calls = Vec::new();
        let selected = select_sparse_lwr_profile(&config, |universe, _, _, _| {
            calls.push(universe);
            Ok(if universe >= 576 { 200 } else { 100 })
        })
        .unwrap();
        assert_eq!(calls, vec![512, 544, 576]);
        assert_eq!(selected.universe, 576);
        assert_eq!(selected.sparse_lwr_security_bits, 200);
        assert_eq!(selected.mitm_security_bits, sparse_support_mitm_bits(576, 64));
    }

    #[test]
    fn phase_one_reports_no_qualified_profile_after_exhausting_grid() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_universe_grid = vec![512, 544, 576];
        let mut calls = 0;
        let error = select_sparse_lwr_profile(&config, |_, _, _, _| {
            calls += 1;
            Ok(0)
        })
        .unwrap_err();
        assert_eq!(calls, 3);
        assert!(error.contains("no sparse-LWR profile"));
        assert!(error.contains("nu grid"));
    }

    #[test]
    fn phase_two_caches_sparse_security_and_defers_prepare_until_bgg_floor() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=1;
        config.log_ring_dimensions = 5..=6;
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
                Ok(if candidate.log_ring_dimension == 5 { 127 } else { 128 })
            },
            |prepared| {
                checker_calls.push(prepared.candidate);
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(security_calls.len(), 2);
        assert_eq!(prepare_calls, vec![Candidate { crt_depth: 1, log_ring_dimension: 6 }]);
        assert_eq!(checker_calls, prepare_calls);
        assert_eq!(result.sparse_lwr_universe, config.sparse_lwr_universe);
        assert_eq!(result.sparse_lwr_weight, config.sparse_lwr_weight);
    }

    #[test]
    fn phase_one_rejects_a_non_ascending_grid() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_universe_grid = vec![512, 512, 544];
        let error = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(200)).unwrap_err();
        assert!(error.contains("strictly ascending"));
    }

    fn checkpoint_test_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mxx-power-lut-{name}-{}.json", std::process::id()))
    }

    #[test]
    fn phase_one_checkpoint_reuses_an_exact_declaration() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_universe_grid = vec![512, 544];
        let path = checkpoint_test_path("phase1-reuse");
        let _ = std::fs::remove_file(&path);

        let mut first_calls = 0;
        let first = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
            first_calls += 1;
            Ok(200)
        })
        .unwrap();

        let mut second_calls = 0;
        let second = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
            second_calls += 1;
            Err("an exact checkpoint must not rerun Phase 1".to_owned())
        })
        .unwrap();
        assert_eq!(first_calls, 1);
        assert_eq!(second_calls, 0);
        assert_eq!(first.universe, second.universe);
        assert_eq!(second.universe, 512);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn phase_one_checkpoint_rejects_mismatch_and_missing_path_starts_fresh_search() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_universe_grid = vec![512, 544];
        let path = checkpoint_test_path("phase1-mismatch");
        let _ = std::fs::remove_file(&path);

        let mut calls = 0;
        let selected = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
            calls += 1;
            Ok(200)
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(selected.universe, 512);

        let mut changed = config.clone();
        changed.security_bits += 1;
        let error = load_or_search_phase1(&changed, Some(&path), |_, _, _, _| {
            Err("a declaration mismatch must not fall back to a fresh search".to_owned())
        })
        .unwrap_err();
        assert!(error.contains("does not match"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn malformed_phase_one_checkpoint_fails_before_security_callback() {
        let config = SearchConfig::reviewed();
        let path = checkpoint_test_path("phase1-malformed");
        std::fs::write(&path, b"{not valid json").unwrap();

        let mut calls = 0;
        let error = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
            calls += 1;
            Ok(200)
        })
        .unwrap_err();
        assert!(error.contains("read Phase-1 checkpoint"));
        assert_eq!(calls, 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn result_serialization_is_redacted() {
        let result = SearchResult {
            candidate: Candidate { crt_depth: 2, log_ring_dimension: 5 },
            achieved_security_bits: 128,
            bgg_rlwe_security_bits: 130,
            sparse_lwr_security_bits: 140,
            mitm_security_bits: 129,
            sparse_lwr_universe: 512,
            sparse_lwr_weight: 64,
            official_preimage_bound: "123".to_owned(),
            ring_dimension: 32,
            bucket_width: 4,
            layout_id: "aa".repeat(32),
            program_id: "bb".repeat(32),
            checker_accepted: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("support"));
        assert!(!json.contains("selected"));
        assert!(json.contains("official_preimage_bound"));
    }

    #[test]
    fn persisted_report_declares_minimality_scope_and_evaluated_phase_one_rows() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_universe_grid = vec![512, 544, 576];
        config.crt_depths = 2..=3;
        config.log_ring_dimensions = 5..=6;
        let sparse_profile = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(200)).unwrap();
        let result = SearchResult {
            candidate: Candidate { crt_depth: 2, log_ring_dimension: 5 },
            achieved_security_bits: 128,
            bgg_rlwe_security_bits: 128,
            sparse_lwr_security_bits: sparse_profile.sparse_lwr_security_bits,
            mitm_security_bits: sparse_profile.mitm_security_bits,
            sparse_lwr_universe: sparse_profile.universe,
            sparse_lwr_weight: sparse_profile.weight,
            official_preimage_bound: "123".to_owned(),
            ring_dimension: 32,
            bucket_width: 4,
            layout_id: "aa".repeat(32),
            program_id: "bb".repeat(32),
            checker_accepted: true,
        };
        let json = serde_json::to_string(&search_report(&config, &sparse_profile, result)).unwrap();

        assert!(json.contains("\"universe_grid\":[512,544,576]"));
        assert!(json.contains("\"support_weight\":64"));
        assert!(json.contains("\"q_l\":16"));
        assert!(json.contains("\"output_modulus\":2"));
        assert!(json.contains("\"lut_width\":32"));
        assert!(json.contains("\"security_target_bits\":128"));
        assert!(json.contains("\"sparse_secret_model\":\"SparseBinary\""));
        assert!(json.contains("\"sparse_error_model\":\"Uniform\""));
        assert!(json.contains("\"sparse_error_lower\":-4"));
        assert!(json.contains("\"sparse_error_upper\":3"));
        assert!(json.contains("\"exact_estimator\":true"));
        assert!(json.contains("\"support_recovery_model\":\"binomial_mitm_floor\""));
        assert!(json.contains("\"evaluated\":["));
        assert!(json.contains("\"universe\":512"));
        assert!(json.contains("\"minimum_security_bits\""));
        assert!(json.contains("\"qualified\":true"));
        assert!(json.contains("\"selected_universe\":512"));
        assert!(json.contains("\"crt_depth_min\":2"));
        assert!(json.contains("\"crt_depth_max\":3"));
        assert!(json.contains("\"log_ring_dimension_min\":5"));
        assert!(json.contains("\"log_ring_dimension_max\":6"));
        assert!(json.contains("\"order\":\"crt_depth_then_log_ring_dimension\""));
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
}
