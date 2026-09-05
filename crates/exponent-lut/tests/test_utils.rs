// Small, deterministic helpers for the ignored Section-7 parameter search.
//
// The search is intentionally kept as test-support source. It is a measurement
// harness, not a new Exponent-LUT parameter abstraction: production code continues
// to use the ordinary `RefreshSetupParameters`, PBC, PRF program, and checker APIs.

use std::{fs, io::ErrorKind, ops::RangeInclusive, path::Path, process::Command};

pub(crate) mod sparse_lwr_parameter_support {
    // Read-only parameter and estimator support for the native mod-Q sparse-LWR PRF.
    //
    // This file is deliberately independent of the refresh search harness. It
    // contains the paper-anchored LWR candidate and glue for the pinned estimator.

    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::{
        collections::{BTreeMap, BTreeSet},
        path::Path,
        process::Command,
    };

    pub const LATTICE_ESTIMATOR_REPOSITORY: &str = "https://github.com/malb/lattice-estimator.git";
    pub const REVIEWED_ESTIMATOR_COMMIT: &str = "53da5982597709ba0fdf94ea37a84d822310fd84";
    pub const PBC_REPLICATION_FACTOR: usize = 3;
    pub const PBC_BUCKET_OFFSET: usize = 3;
    pub const EXPECTED_ATTACKS: &[&str] =
        &["arora-gb", "bkw", "usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid", "dual", "dual_hybrid"];

    /// Derives the only exponent layout compatible with native modulus `Q` and a
    /// concrete encoding ring dimension `N`. `Q` is not an independent LUT width:
    /// `Q = 2N / 2^m`, while the operation width is `log2(Q/2)`.
    pub fn native_exp_layout(
        q: usize,
        ring_dimension: usize,
    ) -> Result<mxx_exponent_lut::ExpLayout, String> {
        if q < 2 || !q.is_power_of_two() || ring_dimension == 0 || !ring_dimension.is_power_of_two()
        {
            return Err(
                "native Q and encoding ring dimension must be positive powers of two".to_owned()
            );
        }
        let twice_ring = ring_dimension
            .checked_mul(2)
            .ok_or_else(|| "encoding ring dimension overflows 2N".to_owned())?;
        if twice_ring < q || twice_ring % q != 0 {
            return Err("native Q must divide 2N and satisfy Q <= 2N".to_owned());
        }
        let stride = twice_ring / q;
        if !stride.is_power_of_two() {
            return Err("native exponent stride must be a power of two".to_owned());
        }
        let stride_log2 = stride.trailing_zeros();
        let width_log2 = (q / 2).trailing_zeros();
        let layout = mxx_exponent_lut::ExpLayout { stride_log2, width_log2 };
        layout.validate(ring_dimension).map_err(|error| error.to_string())?;
        if layout.effective_modulus(ring_dimension) != q {
            return Err("derived exponent layout does not have native modulus Q".to_owned());
        }
        Ok(layout)
    }

    /// Whether a tuple meets the explicitly requested computational target.
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub enum SparseLwrSecurityTier {
        Requested,
        Rejected,
    }

    /// One reviewed Phase-1 point.  The tuple is ordered as `(Q, P, nu, h)`;
    /// The estimator result is part of the declaration so a checkpoint cannot
    /// silently substitute a different model or security policy. Tier is assigned
    /// from the reviewed primary/fallback thresholds after evaluating this row.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct SparseLwrParameterTuple {
        pub q_l: usize,
        pub p: usize,
        pub nu: usize,
        pub h: usize,
        /// Exact finite minimum over the requested attack set, when supplied by
        /// an external reviewed estimator run or a validated checkpoint. Search
        /// configuration never fabricates this evidence.
        pub estimator_minimum_classical_bits: Option<f64>,
        /// Conservative floor corresponding to the external exact minimum.
        pub estimator_security_bits: Option<u64>,
    }

    /// Returns the security floor carried by a reviewed tuple, if the tuple has
    /// complete pinned estimator evidence.  A tuple with no evidence is an
    /// explicit custom point and returns `None`; a partially populated or
    /// inconsistent declaration is rejected before any estimator process is
    /// started.
    pub fn pinned_estimator_security_bits(
        tuple: &SparseLwrParameterTuple,
    ) -> Result<Option<u64>, String> {
        match (tuple.estimator_minimum_classical_bits, tuple.estimator_security_bits) {
            (None, None) => Ok(None),
            (Some(minimum), Some(floor))
                if minimum.is_finite() && minimum >= 0.0 && minimum.floor() as u64 == floor =>
            {
                Ok(Some(floor))
            }
            (Some(_), Some(_)) => Err("Phase-1 tuple estimator evidence is invalid".to_owned()),
            _ => Err("Phase-1 tuple estimator evidence is incomplete".to_owned()),
        }
    }

    impl PartialEq for SparseLwrParameterTuple {
        fn eq(&self, other: &Self) -> bool {
            self.q_l == other.q_l &&
                self.p == other.p &&
                self.nu == other.nu &&
                self.h == other.h &&
                self.estimator_minimum_classical_bits == other.estimator_minimum_classical_bits &&
                self.estimator_security_bits == other.estimator_security_bits
        }
    }

    impl SparseLwrParameterTuple {
        pub fn candidate(&self) -> Result<SparseLwrCandidate, String> {
            let minimum_ring = self
                .q_l
                .checked_div(2)
                .filter(|minimum| *minimum > 0)
                .ok_or_else(|| "native Q must be an even positive modulus".to_owned())?;
            let mut n_enc_candidates = Vec::new();
            for exponent in (0..=16).map(|index| minimum_ring.checked_shl(index as u32)) {
                let Some(n) = exponent else { continue };
                if n <= (1usize << 16) {
                    n_enc_candidates.push(n);
                }
            }
            SparseLwrCandidate::new(
                format!("q{}_p{}_nu{}_h{}", self.q_l, self.p, self.nu, self.h),
                self.nu,
                self.h,
                self.q_l,
                self.p,
                n_enc_candidates,
            )
        }
    }

    fn reviewed_tuple(
        q_l: usize,
        p: usize,
        nu: usize,
        h: usize,
        estimator_minimum_classical_bits: f64,
        estimator_security_bits: u64,
    ) -> SparseLwrParameterTuple {
        SparseLwrParameterTuple {
            q_l,
            p,
            nu,
            h,
            estimator_minimum_classical_bits: Some(estimator_minimum_classical_bits),
            estimator_security_bits: Some(estimator_security_bits),
        }
    }

    /// The finite, ordered primary Phase-1 grid reviewed for integration.  It is
    /// intentionally a tuple list, not a generated Cartesian product.
    pub fn reviewed_phase1_tuple_grid() -> Vec<SparseLwrParameterTuple> {
        vec![
            // The adjacent lower-ν point is retained as explicit minimality
            // evidence and must reject the 100-bit floor.
            reviewed_tuple(2048, 16, 16384, 17, 100.5636531151896, 100),
            reviewed_tuple(4096, 32, 16384, 17, 100.56464330713312, 100),
            reviewed_tuple(4096, 32, 8192, 18, 103.18615790798486, 103),
        ]
    }

    /// A concrete native sparse-LWR candidate. `q` is the native exponent modulus
    /// `Q`, and `p` is the complete balanced digit base `P=2p'`.
    /// `n_enc_candidates` belongs to the encoding layer; the estimator dimension
    /// is always `nu` and is independent of the encoding ring dimension.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct SparseLwrCandidate {
        pub candidate_id: String,
        pub nu: usize,
        pub h: usize,
        pub q: usize,
        pub p: usize,
        pub n_enc_candidates: Vec<usize>,
    }

    impl SparseLwrCandidate {
        pub fn new(
            candidate_id: impl Into<String>,
            nu: usize,
            h: usize,
            q: usize,
            p: usize,
            n_enc_candidates: Vec<usize>,
        ) -> Result<Self, String> {
            let candidate =
                Self { candidate_id: candidate_id.into(), nu, h, q, p, n_enc_candidates };
            candidate.derived()?;
            Ok(candidate)
        }

        pub fn derived(&self) -> Result<SparseLwrDerived, String> {
            if self.nu == 0 || self.h == 0 || self.h > self.nu {
                return Err("native sparse-LWR candidate requires 0 < h <= nu".to_owned());
            }
            if self.q == 0 || !self.q.is_power_of_two() {
                return Err("native Q must be a positive power of two".to_owned());
            }
            if self.p < 2 || !self.p.is_power_of_two() || self.p > self.q || self.q % self.p != 0 {
                return Err(
                    "native P must be a power of two, satisfy 2 <= P <= Q, and divide Q".to_owned()
                );
            }
            if self.n_enc_candidates.is_empty() {
                return Err("at least one encoding ring dimension is required".to_owned());
            }

            // P=2p' is the complete balanced digit base.  Requiring P to be an
            // even power of two makes p'=P/2 integral and keeps the native
            // exponent/error interval aligned with the native power-of-two modulus.
            let delta = self.q / self.p;
            let shift =
                i64::try_from(delta / 2).map_err(|_| "LWR error shift overflows i64".to_owned())?;
            let error_lower = -shift;
            let error_upper = i64::try_from(delta - 1)
                .map_err(|_| "LWR error upper bound overflows i64".to_owned())?
                .checked_sub(shift)
                .ok_or_else(|| "LWR error upper bound overflows i64".to_owned())?;
            let maximum_ring_dimension = 1usize << 16;
            if self
                .n_enc_candidates
                .iter()
                .any(|&n| native_exp_layout(self.q, n).is_err() || n > maximum_ring_dimension)
            {
                return Err(
                "every encoding ring dimension must derive a valid native layout with Q <= 2^16"
                    .to_owned(),
            );
            }

            let delta_squared = delta
                .checked_mul(delta)
                .ok_or_else(|| "native Q/P error variance overflows usize".to_owned())?;
            Ok(SparseLwrDerived {
                p_prime: self.p / 2,
                delta,
                error_lower,
                error_upper,
                error_stddev: ((delta_squared.saturating_sub(1) as f64) / 12.0).sqrt(),
                h_prime: self
                    .h
                    .checked_add(PBC_BUCKET_OFFSET)
                    .ok_or_else(|| "h' overflows usize".to_owned())?,
                total_slots: PBC_REPLICATION_FACTOR
                    .checked_mul(self.nu)
                    .ok_or_else(|| "PBC total slot count overflows usize".to_owned())?,
                average_bucket_size: PBC_REPLICATION_FACTOR as f64 * self.nu as f64 /
                    (self.h + PBC_BUCKET_OFFSET) as f64,
                raw_key_entropy_bits: log2_binomial(self.nu, self.h),
                gcd_condition: true,
            })
        }
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct SparseLwrDerived {
        /// Positive half-width of the balanced native output (`P = 2p'`).
        pub p_prime: usize,
        pub delta: usize,
        pub error_lower: i64,
        pub error_upper: i64,
        pub error_stddev: f64,
        pub h_prime: usize,
        pub total_slots: usize,
        pub average_bucket_size: f64,
        pub raw_key_entropy_bits: f64,
        pub gcd_condition: bool,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct EstimatorReport {
        pub schema_version: u32,
        pub repository_url: String,
        /// Commit of the imported lattice-estimator source checkout (not its wrapper).
        pub git_commit: String,
        pub sage_version: String,
        pub python_version: String,
        pub cost_model: String,
        pub shape_model: String,
        pub quantum: bool,
        pub sample_count: String,
        pub attacks: BTreeMap<String, Value>,
        pub failures: Vec<String>,
        pub infinite_attacks: Vec<String>,
        pub minimum_classical_bits: f64,
    }

    impl EstimatorReport {
        pub(crate) fn validate(&self) -> Result<(), String> {
            if self.schema_version != 1 {
                return Err("unsupported estimator report schema".to_owned());
            }
            if self.repository_url != LATTICE_ESTIMATOR_REPOSITORY {
                return Err("estimator repository is not the official malb repository".to_owned());
            }
            if self.git_commit.len() != 40 ||
                !self.git_commit.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return Err("estimator report does not contain a pinned Git commit".to_owned());
            }
            if self.sage_version.is_empty() || self.python_version.is_empty() {
                return Err("estimator report is missing runtime versions".to_owned());
            }
            if self.cost_model != "MATZOV" || self.shape_model != "gsa" {
                return Err("estimator report does not use RC.MATZOV with GSA".to_owned());
            }
            if self.quantum || self.sample_count != "infinity" {
                return Err(
                    "estimator report is not the requested classical m=infinity run".to_owned()
                );
            }
            if !self.failures.is_empty() {
                return Err("estimator failures are fail-closed".to_owned());
            }
            let infinite = self.infinite_attacks.iter().cloned().collect::<BTreeSet<_>>();
            if infinite.len() != self.infinite_attacks.len() ||
                infinite
                    .iter()
                    .any(|name| !EXPECTED_ATTACKS.iter().any(|attack| *attack == name.as_str()))
            {
                return Err("estimator infinite attack list is unknown or duplicated".to_owned());
            }
            for attack in EXPECTED_ATTACKS {
                let fields = self
                    .attacks
                    .get(*attack)
                    .ok_or_else(|| format!("full estimator result is missing attack {attack}"))?;
                let listed_infinite = infinite.contains(*attack);
                if listed_infinite && fields.get("rop_log2") != Some(&Value::Null) {
                    return Err(format!("infinite attack {attack} must have null rop_log2"));
                }
                let Some(bits) = fields.get("rop_log2").and_then(Value::as_f64) else {
                    if listed_infinite {
                        continue;
                    }
                    return Err(format!("attack {attack} has no finite rop_log2"));
                };
                if !bits.is_finite() {
                    return Err(format!("attack {attack} has non-finite rop_log2"));
                }
            }
            if !self.minimum_classical_bits.is_finite() {
                return Err("minimum classical security is non-finite".to_owned());
            }
            let minimum = EXPECTED_ATTACKS
                .iter()
                .filter_map(|attack| self.attacks[*attack].get("rop_log2").and_then(Value::as_f64))
                .fold(f64::INFINITY, f64::min);
            if !minimum.is_finite() {
                return Err("estimator has no finite attack cost".to_owned());
            }
            if (minimum - self.minimum_classical_bits).abs() > 1e-6 {
                return Err("minimum classical security does not match attack results".to_owned());
            }
            Ok(())
        }
    }

    pub fn parse_estimator_report(bytes: &[u8]) -> Result<EstimatorReport, String> {
        let report: EstimatorReport = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid sparse-LWR estimator JSON: {error}"))?;
        report.validate()?;
        Ok(report)
    }

    /// Invoke the Sage script and parse its machine-readable full estimate.
    pub fn run_estimator(
        candidate: &SparseLwrCandidate,
        script: &Path,
        estimator_root: &Path,
        sage_binary: &Path,
        expected_commit: Option<&str>,
    ) -> Result<EstimatorReport, String> {
        let output = Command::new(sage_binary)
            .args(["-python"])
            .arg(script)
            .args([
                "--nu",
                &candidate.nu.to_string(),
                "--h",
                &candidate.h.to_string(),
                "--q",
                &candidate.q.to_string(),
                "--p",
                &candidate.p.to_string(),
            ])
            .env("MXX_LATTICE_ESTIMATOR_ROOT", estimator_root)
            .env("DOT_SAGE", "/tmp/mxx-sage-cache")
            .output()
            .map_err(|error| format!("failed to start Sage estimator: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "Sage estimator failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let report = parse_estimator_report(&output.stdout)?;
        if let Some(expected) = expected_commit {
            if report.git_commit != expected {
                return Err(format!(
                    "estimator source commit mismatch: expected {expected}, got {}",
                    report.git_commit
                ));
            }
        }
        Ok(report)
    }

    fn log2_binomial(n: usize, k: usize) -> f64 {
        (0..k).map(|index| ((n - index) as f64).ln() - ((index + 1) as f64).ln()).sum::<f64>() /
            2.0_f64.ln()
    }
}

use self::sparse_lwr_parameter_support::{
    REVIEWED_ESTIMATOR_COMMIT, SparseLwrParameterTuple, SparseLwrSecurityTier, native_exp_layout,
    reviewed_phase1_tuple_grid,
};
#[allow(unused_imports)]
pub use self::sparse_lwr_parameter_support::{pinned_estimator_security_bits, run_estimator};
use mxx_bgg::BggSamplerLayout;
use mxx_exponent_lut::{
    AverageCaseConfig, AverageVariance, ExponentLutAverageNoiseReport, ExponentLutNoiseReport,
    ExponentLutNoiseSnapshot,
    noise::RightActionNoiseGain,
    noise_refresh::{
        pbc::{
            PbcActiveCellIndex, PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed,
            generate_key_layout,
        },
        prf::{SparseLwrPrfProfile, SparseLwrPrfProgram},
        refresh::RefreshCompiler,
        setup::{
            RefreshNoiseGainDecompositions, RefreshParameterSimulationBundle,
            RefreshParameterSimulationRequest, RefreshSetupParameters,
        },
        simulation::{
            PreRefreshNoiseEvaluation, centered_base_p_digit_variance,
            fresh_error_below_every_prime_factor,
        },
    },
};
use mxx_ir_core::{IntExpr, RealExpr};
use mxx_primitives::{
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::bounds::TrapdoorPreimageCutoffPolicy,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use rand::{SeedableRng, rngs::StdRng, seq::index::sample};
use serde::Serialize;
use tracing::info;

const REVIEWED_ESTIMATOR_COST_MODEL: &str = "MATZOV";
const REVIEWED_ESTIMATOR_SHAPE_MODEL: &str = "gsa";
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

/// Validates the frozen refresh bundle and runs the application-specific
/// native Exponent-LUT noise authority.  Generic graph-noise conversion is
/// intentionally not part of the search boundary.
pub fn check_refresh_bundle(
    config: &SearchConfig,
    _candidate: Candidate,
    prepared: &mut PreparedCandidate,
) -> Result<bool, String> {
    if config.noise_model == NoiseSearchModel::AverageCase {
        let average = if let Some(bundle) = prepared.bundle.as_ref() {
            bundle.simulate_average_noise(&average_case_config(config)?)
        } else if let Some(snapshot) = prepared.noise_snapshot.as_ref() {
            if let Some(pre_refresh) = prepared.pre_refresh.as_ref() {
                snapshot.simulate_average_with_pre_refresh(
                    &average_case_config(config)?,
                    pre_refresh.clone(),
                )
            } else {
                snapshot.simulate_average(&average_case_config(config)?)
            }
        } else {
            return Err("candidate has neither a simulation bundle nor a noise snapshot".to_owned());
        }
        .map_err(|error| format!("AverageCase simulation: {error}"))?;
        info!(
            steps = average.program.steps.len(),
            accepted = average.accepted,
            hard_authority_accepted = average.hard_authority_accepted,
            correctness_accepted = average.correctness_accepted,
            "native AverageCase noise report"
        );
        prepared.production_report = Some(ProductionNoiseReport {
            worst_case: None,
            average_case: Some(average.clone()),
            fresh_error_margins: fresh_error_margins(config, _candidate, prepared)?,
        });
        log_average_noise_report(_candidate, &average);
        prepared.average_report = Some(average.clone());
        return Ok(average.accepted);
    }
    let report = if let Some(bundle) = prepared.bundle.as_ref() {
        bundle.simulate_noise()
    } else if let Some(snapshot) = prepared.noise_snapshot.as_ref() {
        if let Some(pre_refresh) = prepared.pre_refresh.as_ref() {
            snapshot.simulate_worst_from_pre_refresh(pre_refresh.clone())
        } else {
            snapshot.simulate()
        }
    } else {
        return Err("candidate has neither a simulation bundle nor a noise snapshot".to_owned());
    }
    .map_err(|error| error.to_string())?;
    info!(
        steps = report.program.steps.len(),
        readout_bits = report.readout_bound.bits(),
        "native Exponent-LUT noise report"
    );
    let accepted = report.refresh.accepted;
    log_worst_case_noise_report(_candidate, &report);
    prepared.production_report = Some(ProductionNoiseReport {
        worst_case: Some(report),
        average_case: None,
        fresh_error_margins: fresh_error_margins(config, _candidate, prepared)?,
    });
    Ok(accepted)
}

fn log_worst_case_noise_report(candidate: Candidate, report: &ExponentLutNoiseReport) {
    for step in &report.program.steps {
        info!(
            crt_depth = candidate.crt_depth,
            step = step.step_index,
            inherited_bits = step.inherited_bits,
            fresh_bits = step.fresh_bits,
            output_bits = step.output_bits,
            bit_growth = step.output_bits.saturating_sub(step.inherited_bits),
            family_member_count = step.family_member_count,
            family_representative_rows = step.family_body.len(),
            one_hot = !step.family_body.is_empty(),
            "production worst-case gate noise growth"
        );
        for body_gate in &step.family_body {
            info!(
                step = step.step_index,
                body_gate = body_gate.body_gate_index,
                operation = ?body_gate.operation,
                inherited_bits = body_gate.inherited_bits,
                fresh_bits = body_gate.fresh_bits,
                output_bits = body_gate.output_bits,
                bit_growth = body_gate.bit_growth,
                "production worst-case family body noise growth"
            );
        }
        if step.family_member_count > 0 {
            info!(
                step = step.step_index,
                family_member_count = step.family_member_count,
                constituent_count = step.constituent_bounds.len(),
                output_bits = step.output_bits,
                "production worst-case family aggregate noise growth"
            );
        }
    }
    info!(readout_bits = report.readout_bound.bits(), "production worst-case readout noise bound");
    for slot in &report.refresh.slots {
        info!(
            slot = slot.slot,
            rounding_lhs_bits = slot.rounding_lhs.bits(),
            rounding_margin = %slot.rounding_margin,
            fresh_error_accepted = slot.fresh_error_accepted,
            "production worst-case refresh slot margins"
        );
    }
}

fn log_average_noise_report(candidate: Candidate, report: &ExponentLutAverageNoiseReport) {
    for step in &report.program.steps {
        info!(
            crt_depth = candidate.crt_depth,
            step = step.step_index,
            inherited_bits = step.inherited_bits,
            fresh_bits = step.fresh_bits,
            output_bits = step.output_bits,
            bit_growth = step.output_bits.saturating_sub(step.inherited_bits),
            family_member_count = step.family_member_count,
            family_representative_rows = step.family_body.len(),
            one_hot = !step.family_body.is_empty(),
            "production average-case gate noise growth"
        );
        for body_gate in &step.family_body {
            info!(
                step = step.step_index,
                body_gate = body_gate.body_gate_index,
                operation = ?body_gate.operation,
                inherited_bits = body_gate.inherited_bits,
                fresh_bits = body_gate.fresh_bits,
                output_bits = body_gate.output_bits,
                bit_growth = body_gate.bit_growth,
                "production average-case family body noise growth"
            );
        }
    }
    info!(
        readout_bits = report.program.output_bound.bits(),
        accepted = report.accepted,
        "production average-case readout noise bound"
    );
    for slot in &report.refresh.slots {
        info!(
            slot = slot.slot,
            rounding_lhs_bits = slot.rounding_lhs.bits(),
            rounding_margin = %slot.rounding_margin,
            rounding_deficit_bits = slot.rounding_deficit.bits(),
            fresh_error_accepted = slot.fresh_error_accepted,
            "production average-case refresh slot margins"
        );
    }
}

fn fresh_error_margins(
    config: &SearchConfig,
    candidate: Candidate,
    prepared: &PreparedCandidate,
) -> Result<Vec<String>, String> {
    let parameters = DCRTPolyParams::new(
        prepared.ring_dimension as u32,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
    );
    let bound = BigUint::from(config.sparse_lwr_output_modulus)
        .pow(prepared.fresh_error_base_p_digit_count as u32) -
        BigUint::one();
    parameters
        .to_crt()
        .0
        .into_iter()
        .map(|modulus| {
            let factors =
                mxx_exponent_lut::noise_refresh::simulation::prime_factors_of_crt_modulus(
                    &BigUint::from(modulus),
                )
                .map_err(|error| error.to_string())?;
            let smallest_factor = factors
                .into_iter()
                .next()
                .ok_or_else(|| "CRT modulus has no prime factors".to_owned())?;
            Ok::<String, String>(
                (BigInt::from(smallest_factor) - BigInt::from(bound.clone())).to_string(),
            )
        })
        .collect()
}

/// The reviewed finite profile grid used by the Exponent-LUT refresh search.
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
    /// Selected decoder cutoff policy. It is resolved and compared against
    /// the default trapdoor-sampler cutoff at concrete candidate preparation time.
    pub decoder_preimage_bound: TrapdoorPreimageCutoffPolicy,
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
    pub phase1_estimator_commit: String,
    pub phase1_estimator_cost_model: String,
    pub phase1_estimator_shape_model: String,
    pub pbc_max_attempts: u32,
    pub one_nontrivial_refresh_round: bool,
    pub plaintext_modulus: usize,
    /// Optional reviewed BGG/RLWE monotone security envelope.  When absent,
    /// Phase 2 uses the external lattice estimator as before.
    pub bgg_security_envelope: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum NoiseSearchModel {
    WorstCase,
    AverageCase,
}

impl SearchConfig {
    /// Returns the reviewed finite grid. Phase 1 uses the ordered native
    /// `(Q, P, nu, h)` tuple points selected for the integration search. Phase 2 is
    /// searched through the reviewed depth and ring-dimension ranges. Gadget
    /// bases are considered globally in descending order, with CRT width as a
    /// descending tie-break.
    pub fn reviewed() -> Self {
        Self {
            noise_model: NoiseSearchModel::WorstCase,
            crt_depths: 1..=62,
            log_ring_dimensions: 16..=16,
            security_bits: 100,
            mask_statistical_security_bits: 100,
            crt_bits: 50,
            base_bits: 25,
            crt_base_bits_grid: vec![(50, 25)],
            secret_dimension: 2,
            error_sigma: 4.0,
            decoder_sigma: 4.578,
            decoder_preimage_bound: TrapdoorPreimageCutoffPolicy::Default,
            sparse_lwr_universe: 16384,
            sparse_lwr_weight: 17,
            sparse_lwr_universe_grid: vec![16384],
            sparse_lwr_modulus: 2048,
            sparse_lwr_output_modulus: 16,
            sparse_lwr_phase1_grid: reviewed_phase1_tuple_grid(),
            phase1_estimator_commit: REVIEWED_ESTIMATOR_COMMIT.to_owned(),
            phase1_estimator_cost_model: REVIEWED_ESTIMATOR_COST_MODEL.to_owned(),
            phase1_estimator_shape_model: REVIEWED_ESTIMATOR_SHAPE_MODEL.to_owned(),
            pbc_max_attempts: 128,
            one_nontrivial_refresh_round: true,
            plaintext_modulus: 2,
            bgg_security_envelope: None,
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
    /// state. Keeping the lookup injectable makes parser tests deterministic
    /// and race-free while preserving the exact environment schema used by
    /// the ignored simulation: security, native sparse-LWR tuple, CRT/base
    /// bits, and MIN/MAX CRT-depth and ring-dimension bounds.
    /// Every schema name uses the `MXX_EXPONENT_LUT_REFRESH_` prefix.
    pub fn from_lookup<L>(lookup: L) -> Result<Self, String>
    where
        L: Fn(&str) -> Result<Option<String>, String>,
    {
        let mut config = Self::reviewed();
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS")? {
            config.security_bits =
                parse_lookup_u64("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", &value)?;
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL")? {
            config.noise_model =
                match value.as_str() {
                    "worst-case" => NoiseSearchModel::WorstCase,
                    "average-case" => NoiseSearchModel::AverageCase,
                    _ => return Err(
                        "MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL must be worst-case or average-case"
                            .to_owned(),
                    ),
                };
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_BGG_SECURITY_ENVELOPE")? {
            if value != "reviewed_monotone_envelope" {
                return Err(
                    "MXX_EXPONENT_LUT_REFRESH_BGG_SECURITY_ENVELOPE must be reviewed_monotone_envelope"
                        .to_owned(),
                );
            }
            config.bgg_security_envelope = Some(value);
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS")? {
            config.mask_statistical_security_bits = parse_lookup_usize(
                "MXX_EXPONENT_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS",
                &value,
            )?;
        }
        let explicit_nu = lookup("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE")?;
        let explicit_h = lookup("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT")?;
        let explicit_q = lookup("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS")?;
        let explicit_p = lookup("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS")?;
        match (explicit_nu, explicit_h, explicit_q, explicit_p) {
            (Some(nu), Some(h), Some(q), Some(p)) => {
                let nu = parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE", &nu)?;
                let h = parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT", &h)?;
                let q = parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS", &q)?;
                let p =
                    parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", &p)?;
                config.sparse_lwr_universe = nu;
                config.sparse_lwr_weight = h;
                config.sparse_lwr_universe_grid = vec![nu];
                config.sparse_lwr_modulus = q;
                config.sparse_lwr_output_modulus = p;
                // The native tuple is the sole Phase-1 source of truth for an
                // explicit profile. Reuse reviewed evidence when this exact
                // tuple is selected; an unknown tuple remains evidence-free
                // and therefore requires an external estimator/checkpoint.
                config.sparse_lwr_phase1_grid = vec![
                    reviewed_phase1_tuple_grid()
                        .into_iter()
                        .find(|tuple| {
                            tuple.q_l == q && tuple.p == p && tuple.nu == nu && tuple.h == h
                        })
                        .unwrap_or(SparseLwrParameterTuple {
                            q_l: q,
                            p,
                            nu,
                            h,
                            estimator_minimum_classical_bits: None,
                            estimator_security_bits: None,
                        }),
                ];
            }
            (None, None, None, None) => {}
            _ => {
                return Err(
                    "MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE, _WEIGHT, _MODULUS, and _OUTPUT_MODULUS must be supplied together"
                        .to_owned(),
                );
            }
        }
        let explicit_crt_bits = lookup("MXX_EXPONENT_LUT_REFRESH_CRT_BITS")?;
        let explicit_base_bits = lookup("MXX_EXPONENT_LUT_REFRESH_BASE_BITS")?;
        match (explicit_crt_bits, explicit_base_bits) {
            (Some(crt_bits), Some(base_bits)) => {
                config.crt_bits =
                    parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", &crt_bits)?;
                config.base_bits =
                    parse_lookup_u32("MXX_EXPONENT_LUT_REFRESH_BASE_BITS", &base_bits)?;
                // An explicit pair is an explicit singleton, not a request to
                // regenerate the historical 32/30/28 search grid.
                config.crt_base_bits_grid = vec![(config.crt_bits, config.base_bits)];
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                return Err(
                    "MXX_EXPONENT_LUT_REFRESH_CRT_BITS and MXX_EXPONENT_LUT_REFRESH_BASE_BITS must be supplied together"
                        .to_owned(),
                );
            }
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MIN_CRT_DEPTH")? {
            config.crt_depths =
                parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_MIN_CRT_DEPTH", &value)?..=
                    *config.crt_depths.end();
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH")? {
            config.crt_depths = *config.crt_depths.start()..=
                parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH", &value)?;
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MIN_LOG_RING_DIMENSION")? {
            config.log_ring_dimensions =
                parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_MIN_LOG_RING_DIMENSION", &value)?..=
                    *config.log_ring_dimensions.end();
        }
        if let Some(value) = lookup("MXX_EXPONENT_LUT_REFRESH_MAX_LOG_RING_DIMENSION")? {
            config.log_ring_dimensions = *config.log_ring_dimensions.start()..=
                parse_lookup_usize("MXX_EXPONENT_LUT_REFRESH_MAX_LOG_RING_DIMENSION", &value)?;
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
        let ring_dimensions = [log_ring_start, log_ring_end]
            .into_iter()
            .map(|log_ring| {
                1usize
                    .checked_shl(log_ring as u32)
                    .ok_or_else(|| "log ring-dimension range is not representable".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if self.security_bits == 0 {
            return Err("security bits must be positive".to_owned());
        }
        if self.crt_base_bits_grid.is_empty() ||
            self.crt_base_bits_grid.iter().any(|(crt_bits, base_bits)| {
                ring_dimensions.iter().any(|&ring_dimension| {
                    u32::try_from(ring_dimension)
                        .ok()
                        .and_then(|ring_dimension| {
                            DCRTPolyParams::validate_capability(
                                ring_dimension,
                                crt_start,
                                *crt_bits,
                                *base_bits,
                            )
                            .ok()
                        })
                        .is_none()
                })
            }) ||
            self.crt_base_bits_grid.windows(2).any(|window| {
                window[0].1 < window[1].1 ||
                    (window[0].1 == window[1].1 && window[0].0 <= window[1].0)
            })
        {
            return Err(
                "CRT/base bit grid must be ordered and supported by the DCRT backend".to_owned()
            );
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
            tuple.candidate().map_err(|error| format!("invalid Phase-1 tuple: {error}"))?;
            pinned_estimator_security_bits(tuple)?;
        }
        if self.sparse_lwr_phase1_grid.windows(2).any(|window| {
            window[0].q_l > window[1].q_l ||
                (window[0].q_l == window[1].q_l && window[0].p > window[1].p) ||
                window[0] == window[1]
        }) {
            return Err("Phase-1 tuple grid must be in reviewed ascending order".to_owned());
        }
        if self.security_bits == 0 {
            return Err("security target must be positive".to_owned());
        }
        if self.phase1_estimator_commit.len() != 40 ||
            !self.phase1_estimator_commit.bytes().all(|byte| byte.is_ascii_hexdigit()) ||
            self.phase1_estimator_cost_model.is_empty() ||
            self.phase1_estimator_shape_model.is_empty()
        {
            return Err("Phase-1 estimator identity is incomplete".to_owned());
        }
        sparse_lwr_error_bounds(self.sparse_lwr_modulus, self.sparse_lwr_output_modulus)?;
        let minimum_ring_dimension = 1usize
            .checked_shl(log_ring_start as u32)
            .ok_or_else(|| "minimum log ring dimension shift overflows usize".to_owned())?;
        native_exp_layout(self.sparse_lwr_modulus, minimum_ring_dimension)
            .map_err(|error| format!("native exponent layout: {error}"))?;
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

/// Returns the reviewed CRT/base search order: base widths descend globally,
/// with CRT widths descending as the tie-break. The backend validates each
/// generated pair for the concrete candidate; this helper does not impose a
/// base-width heuristic such as `base_bits <= crt_bits / 2`.
pub fn descending_crt_base_bits_grid(max_crt_bits: usize) -> Vec<(usize, u32)> {
    let mut widths = [max_crt_bits, 32, 30, 28]
        .into_iter()
        .filter(|&crt_bits| crt_bits != 0 && crt_bits <= max_crt_bits)
        .collect::<Vec<_>>();
    widths.sort_unstable_by(|left, right| right.cmp(left));
    widths.dedup();
    let max_base_bits = widths.iter().copied().max().unwrap_or(0).saturating_sub(1);
    (1..=max_base_bits as u32)
        .rev()
        .flat_map(|base_bits| widths.iter().copied().map(move |crt_bits| (crt_bits, base_bits)))
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
    /// The concrete cutoff resolved from a final setup clone carrying the
    /// Default policy. This is the value reported as the official bound.
    pub official_preimage_bound: BigInt,
    /// The concrete cutoff selected by SearchConfig. It may exceed the
    /// official value after the explicit >= comparison.
    pub selected_preimage_bound: BigInt,
    pub layout_id: [u8; 32],
    pub program_id: [u8; 32],
    /// One-based accepted PBC layout retry count (public diagnostics only).
    pub pbc_attempts_used: u32,
    pub mask_base_p_digit_count: usize,
    pub fresh_error_base_p_digit_count: usize,
    /// Exact structural factors of the deferred refresh-PRF batch, retained for
    /// benchmark diagnostics without expanding the family on the host.
    pub prf_component_count: usize,
    pub prf_coefficient_count: usize,
    pub prf_active_count: usize,
    pub prf_label_count: usize,
    pub prf_value_count: usize,
    /// Present for the production search; omitted by mocked ordering tests.
    pub bundle: Option<RefreshParameterSimulationBundle>,
    /// Compact production-equivalent noise authority used during ordinary
    /// search preparation. It contains no executable graph or label family.
    pub noise_snapshot: Option<ExponentLutNoiseSnapshot>,
    /// Cached program/readout evaluation shared by all `d_m` candidates.
    pub pre_refresh: Option<PreRefreshNoiseEvaluation>,
    /// The complete production report used to accept this candidate.  It is
    /// populated by `check_refresh_bundle`, never by a shape-only estimate.
    pub production_report: Option<ProductionNoiseReport>,
    pub average_report: Option<ExponentLutAverageNoiseReport>,
}

/// Complete setup-owned evidence for one candidate.  The nested program
/// report retains every gate/family member bit transition; the refresh report
/// retains the exact per-slot rounding and fresh-error margins.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionNoiseReport {
    pub worst_case: Option<ExponentLutNoiseReport>,
    pub average_case: Option<ExponentLutAverageNoiseReport>,
    /// Exact smallest-prime-factor minus `(P^d_e-1)` margin for every CRT slot.
    pub fresh_error_margins: Vec<String>,
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
    pub average_evidence: Option<AverageAcceptanceEvidence>,
    pub production_report: Option<ProductionNoiseReport>,
    pub bgg_security_authority: String,
    pub bgg_security_evidence_source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AverageAcceptanceEvidence {
    pub setup_identity: String,
    pub layout_id: String,
    pub program_id: String,
    pub mask_digit_count: usize,
    pub fresh_error_digit_count: usize,
    pub event_budget_log2: u32,
    pub tail_correction_bits: u32,
    pub joint_event_count: String,
    pub z_squared: String,
    pub mask_smudging_max_favg: String,
    pub hard_authority_accepted: bool,
    pub correctness_accepted: bool,
    pub accepted: bool,
    pub security_authority: String,
    pub correctness_authority: String,
    pub heuristics: Vec<String>,
    pub slots: Vec<AverageSlotEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AverageSlotEvidence {
    pub slot: usize,
    pub favg: String,
    pub kappa: String,
    pub spacing: String,
    pub deterministic_mask_bound: String,
    pub rounding_lhs: String,
    pub rounding_margin: String,
    pub rounding_deficit: String,
    pub rounding_accepted: bool,
    pub fresh_error_below_prime_factors: bool,
    pub fresh_error_margin: String,
    pub variance: String,
    pub readout_mask_variance: String,
    pub readout_fresh_variance: String,
    pub decoder_variance: String,
    pub state_variance: String,
    pub mask_digit_variance: String,
    pub fresh_digit_variance: String,
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
pub const PHASE1_CHECKPOINT_ENV: &str = "MXX_EXPONENT_LUT_REFRESH_PHASE1_CHECKPOINT";

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
            security_target_bits: config.security_bits,
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
            estimator_commit: config.phase1_estimator_commit.clone(),
            estimator_cost_model: config.phase1_estimator_cost_model.clone(),
            estimator_shape_model: config.phase1_estimator_shape_model.clone(),
            exact_estimator: true,
            crt_base_bits_grid: config.crt_base_bits_grid.clone(),
            phase2_search_order: "all_qualified_sparse_profiles; crt_depth_ascending_then_log_ring_dimension_ascending_then_base_bits_descending_then_crt_bits_descending; profile_tuple_q_l_p_nu_h_ascending".to_owned(),
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
            let expected_tier = if row.minimum_security_bits >= config.security_bits {
                SparseLwrSecurityTier::Requested
            } else {
                SparseLwrSecurityTier::Rejected
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
                row.sparse_lwr_security_bits != row.estimator_security_bits ||
                (row.raw_key_entropy_bits - raw_key_entropy_bits(row.universe, row.weight)).abs() >
                    1e-9 ||
                row.qualified != (row.minimum_security_bits >= config.security_bits) ||
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
        let earlier_primary = self.evaluated[..selected_index].iter().any(|row| row.qualified);
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
            self.selected
                .tuple
                .estimator_security_bits
                .is_some_and(|bits| bits != selected_row.estimator_security_bits) ||
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
    /// Independent Phase-2 outcome for every qualified Phase-1 profile.  A
    /// missing result records the canonical finite-grid exhaustion for that
    /// profile; fatal construction errors abort the search instead.
    pub profile_results: Vec<ProfileSearchOutcome>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProfileSearchOutcome {
    pub profile: SelectedSparseLwrProfile,
    pub result: Option<SearchResult>,
    pub error: Option<String>,
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
            universe_grid: config
                .sparse_lwr_phase1_grid
                .iter()
                .map(|tuple| tuple.nu)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            support_weight: sparse_profile.weight,
            q_l: sparse_profile.q_l,
            output_modulus: sparse_profile.p,
            security_target_bits: config.security_bits,
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
            order: "crt_depth_ascending_then_log_ring_dimension_ascending_then_base_bits_descending_then_crt_bits_descending; profile_tuple_q_l_p_nu_h_ascending",
            security_target_bits: config.security_bits,
            overall_target_policy: "requested_target".to_owned(),
            crt_base_bits_grid: config.crt_base_bits_grid.clone(),
            noise_model: config.noise_model,
        },
        result,
        profile_results: Vec::new(),
    }
}

/// Builds a report retaining the independently evaluated result for every
/// qualifying Phase-1 profile.  The selected result remains duplicated in
/// `result` for stable consumers, while `profile_results` carries the global
/// minimality evidence.
pub fn search_report_with_profiles(
    config: &SearchConfig,
    selected: &SelectedSparseLwrProfile,
    result: SearchResult,
    profile_results: Vec<ProfileSearchOutcome>,
) -> RefreshParameterSearchReport {
    let mut report = search_report(config, selected, result);
    report.profile_results = profile_results;
    report
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
        tuple.candidate().map_err(|error| format!("invalid Phase-1 tuple: {error}"))?;
    }
    if config.sparse_lwr_phase1_grid.windows(2).any(|window| {
        window[0].q_l > window[1].q_l ||
            (window[0].q_l == window[1].q_l && window[0].p > window[1].p) ||
            window[0] == window[1]
    }) {
        return Err("sparse-LWR Phase-1 tuple grid must be in reviewed ascending order".to_owned());
    }
    let mut evaluations = Vec::new();
    let mut primary_selection = None;
    for tuple in &config.sparse_lwr_phase1_grid {
        let universe = tuple.nu;
        let weight = tuple.h;
        let q_l = tuple.q_l;
        let p = tuple.p;
        let sparse_bits = sparse_security(universe, weight, q_l, p)?;
        if let Some(expected_floor) = pinned_estimator_security_bits(tuple)? {
            if sparse_bits != expected_floor {
                return Err(format!(
                    "estimator result for (Q_L={}, p={}, nu={}, h={}) was {} bits, expected declared floor {}",
                    tuple.q_l, tuple.p, tuple.nu, tuple.h, sparse_bits, expected_floor
                ));
            }
        }
        if sparse_bits == 0 {
            return Err(format!(
                "estimator result for (Q_L={}, p={}, nu={}, h={}) was zero bits",
                tuple.q_l, tuple.p, tuple.nu, tuple.h
            ));
        }
        let exact_minimum = tuple.estimator_minimum_classical_bits.unwrap_or(sparse_bits as f64);
        let tier = if exact_minimum >= config.security_bits as f64 {
            SparseLwrSecurityTier::Requested
        } else {
            SparseLwrSecurityTier::Rejected
        };
        let qualified = exact_minimum >= config.security_bits as f64;
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
            estimator_security_bits: sparse_bits,
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
        if qualified && primary_selection.is_none() {
            primary_selection = Some((tuple.clone(), sparse_bits));
        }
    }
    let (tuple, sparse_bits) = primary_selection.ok_or_else(|| {
        format!(
            "no sparse-LWR profile in the declared tuple grid passed the requested {}-bit floor",
            config.security_bits
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
        tier: SparseLwrSecurityTier::Requested,
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
                row.q_l == tuple.q_l &&
                    row.p == tuple.p &&
                    row.universe == tuple.nu &&
                    row.weight == tuple.h
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

/// Evaluate every qualified Phase-1 profile. Only the canonical finite-grid
/// exhaustion result advances to the next profile; setup, estimator, and
/// checker failures remain fatal. Keeping all outcomes is required for a
/// global CRT-depth minimum rather than first-success selection.
pub fn search_all_qualified_profiles<Run>(
    selected: &SelectedSparseLwrProfile,
    mut run: Run,
) -> Result<Vec<ProfileSearchOutcome>, String>
where
    Run: FnMut(&SelectedSparseLwrProfile) -> Result<SearchResult, String>,
{
    let mut outcomes = Vec::new();
    for profile in qualified_sparse_lwr_profiles(selected) {
        match run(&profile) {
            Ok(result) => {
                outcomes.push(ProfileSearchOutcome { profile, result: Some(result), error: None })
            }
            Err(error) if error == NO_CANDIDATE_ERROR => {
                outcomes.push(ProfileSearchOutcome { profile, result: None, error: Some(error) });
            }
            Err(error) => return Err(error),
        }
    }
    if outcomes.iter().all(|outcome| outcome.result.is_none()) {
        return Err("no qualified sparse-LWR profile produced a valid Phase-2 candidate".to_owned());
    }
    Ok(outcomes)
}

/// Advance through all qualified Phase-1 profiles and select the global
/// minimum. The documented tie-break is depth, ring dimension, descending
/// base width, descending CRT width, then the reviewed tuple order.
pub fn search_qualified_profiles<Run>(
    selected: &SelectedSparseLwrProfile,
    run: Run,
) -> Result<(SelectedSparseLwrProfile, SearchResult), String>
where
    Run: FnMut(&SelectedSparseLwrProfile) -> Result<SearchResult, String>,
{
    let outcomes = search_all_qualified_profiles(selected, run)?;
    select_best_profile_result(outcomes)
}

/// Selects the best successful outcome without rerunning any production
/// report. This is kept separate so callers can persist all profile outcomes
/// and the selected result from one evaluation pass.
pub fn select_best_profile_result(
    outcomes: Vec<ProfileSearchOutcome>,
) -> Result<(SelectedSparseLwrProfile, SearchResult), String> {
    outcomes
        .into_iter()
        .filter_map(|outcome| outcome.result.map(|result| (outcome.profile, result)))
        .min_by(|(left_profile, left), (right_profile, right)| {
            let left_key = (
                left.candidate.crt_depth,
                left.candidate.log_ring_dimension,
                std::cmp::Reverse(left.candidate.base_bits),
                std::cmp::Reverse(left.candidate.crt_bits),
                left_profile.q_l,
                left_profile.p,
                left_profile.universe,
                left_profile.weight,
            );
            let right_key = (
                right.candidate.crt_depth,
                right.candidate.log_ring_dimension,
                std::cmp::Reverse(right.candidate.base_bits),
                std::cmp::Reverse(right.candidate.crt_bits),
                right_profile.q_l,
                right_profile.p,
                right_profile.universe,
                right_profile.weight,
            );
            left_key.cmp(&right_key)
        })
        .ok_or_else(|| {
            "no qualified sparse-LWR profile produced a valid Phase-2 candidate".to_owned()
        })
}

fn exact_reconstruction_coefficients(dcrt: &DCRTPolyParams) -> Vec<IntExpr> {
    dcrt.reconst_coeffs()
        .into_iter()
        .map(|value| BigInt::from_biguint(num_bigint::Sign::Plus, value).into())
        .collect()
}

/// Enumerates the grid in the exact order used for minimality claims: CRT
/// depth first, then ring dimension, then the reviewed descending base/CRT
/// tie-break. There is no hidden total-modulus-bit cap here: backend
/// feasibility is established when the concrete candidate is prepared.
pub fn candidates(config: &SearchConfig) -> impl Iterator<Item = Candidate> + '_ {
    config.crt_depths.clone().flat_map(move |crt_depth| {
        config.log_ring_dimensions.clone().flat_map(move |log_ring_dimension| {
            config.crt_base_bits_grid.iter().copied().map(move |(crt_bits, base_bits)| Candidate {
                crt_depth,
                log_ring_dimension,
                crt_bits,
                base_bits,
            })
        })
    })
}

fn final_average_evidence(
    config: &SearchConfig,
    prepared: &PreparedCandidate,
    report: &ExponentLutAverageNoiseReport,
) -> Result<Option<AverageAcceptanceEvidence>, String> {
    if config.noise_model != NoiseSearchModel::AverageCase {
        return Ok(None);
    }
    let policy = average_case_config(config)?;
    let refresh = &report.refresh;
    let fresh_margins = fresh_error_margins(config, prepared.candidate, prepared)?;
    let topology = refresh.joint_event_count.clone();
    let event_budget_log2 = {
        let bits = topology.bits();
        if topology == (BigUint::one() << (bits - 1)) {
            u32::try_from(bits - 1)
                .map_err(|_| "AverageCase event budget is too large".to_owned())?
        } else {
            u32::try_from(bits).map_err(|_| "AverageCase event budget is too large".to_owned())?
        }
    };
    let z_squared = &refresh.z_squared;
    Ok(Some(AverageAcceptanceEvidence {
        setup_identity: hex(report.snapshot_identity),
        layout_id: hex(prepared.layout_id),
        program_id: hex(prepared.program_id),
        mask_digit_count: prepared.mask_base_p_digit_count,
        fresh_error_digit_count: prepared.fresh_error_base_p_digit_count,
        event_budget_log2,
        tail_correction_bits: policy.tail_correction_bits,
        joint_event_count: topology.to_string(),
        z_squared: format!("{}/{}", z_squared.numerator, z_squared.denominator),
        mask_smudging_max_favg: refresh.max_f_avg.to_string(),
        hard_authority_accepted: report.hard_authority_accepted,
        correctness_accepted: report.correctness_accepted,
        accepted: report.accepted,
        security_authority: format!("{:?}", report.security_authority),
        correctness_authority: format!("{:?}", report.correctness_authority),
        heuristics: Vec::new(),
        slots: refresh
            .slots
            .iter()
            .map(|slot| {
                Ok::<AverageSlotEvidence, String>(AverageSlotEvidence {
                    slot: slot.slot,
                    favg: slot.f_avg.to_string(),
                    kappa: slot.kappa.to_string(),
                    spacing: slot.spacing.to_string(),
                    deterministic_mask_bound: slot.deterministic_mask_bound.to_string(),
                    rounding_lhs: slot.rounding_lhs.to_string(),
                    rounding_margin: slot.rounding_margin.to_string(),
                    rounding_deficit: slot.rounding_deficit.to_string(),
                    rounding_accepted: slot.rounding_margin.sign() == num_bigint::Sign::Plus,
                    fresh_error_below_prime_factors: slot.fresh_error_accepted,
                    fresh_error_margin: fresh_margins
                        .get(slot.slot)
                        .cloned()
                        .ok_or_else(|| "average report slot lacks fresh-error margin".to_owned())?,
                    variance: format!("{}/{}", slot.variance.numerator, slot.variance.denominator),
                    readout_mask_variance: rational_string(&slot.variance_breakdown.readout_mask),
                    readout_fresh_variance: rational_string(&slot.variance_breakdown.readout_fresh),
                    decoder_variance: rational_string(&slot.variance_breakdown.decoder),
                    state_variance: rational_string(&slot.variance_breakdown.state),
                    mask_digit_variance: rational_string(&slot.variance_breakdown.mask_digit),
                    fresh_digit_variance: rational_string(&slot.variance_breakdown.fresh_digit),
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    }))
}

fn rational_string(value: &AverageVariance) -> String {
    format!("{}/{}", value.numerator, value.denominator)
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
    Checker: FnMut(&mut PreparedCandidate) -> Result<bool, String>,
{
    if sparse_profile.q_l != config.sparse_lwr_modulus ||
        sparse_profile.p != config.sparse_lwr_output_modulus ||
        sparse_profile.universe != config.sparse_lwr_universe ||
        sparse_profile.weight != config.sparse_lwr_weight
    {
        return Err(
            "selected sparse-LWR profile does not match the frozen search configuration".to_owned()
        );
    }
    let overall_security_target = config.security_bits;
    for candidate in candidates(config) {
        info!(
            crt_depth = candidate.crt_depth,
            log_ring_dimension = candidate.log_ring_dimension,
            crt_bits = candidate.crt_bits,
            base_bits = candidate.base_bits,
            "evaluating Exponent-LUT refresh search candidate"
        );
        if let Err(error) = validate_backend_candidate(candidate) {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                crt_bits = candidate.crt_bits,
                base_bits = candidate.base_bits,
                reason = %error,
                "candidate rejected by backend CRT/base capability"
            );
            continue;
        }
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
        let mut prepared = prepared;
        if !checker(&mut prepared)? {
            info!(
                crt_depth = candidate.crt_depth,
                log_ring_dimension = candidate.log_ring_dimension,
                crt_bits = candidate.crt_bits,
                base_bits = candidate.base_bits,
                "candidate rejected by Exponent-LUT noise threshold"
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
            average_evidence: prepared
                .average_report
                .as_ref()
                .map(|report| final_average_evidence(config, &prepared, report))
                .transpose()?
                .flatten(),
            production_report: prepared.production_report.clone(),
            bgg_security_authority: bgg_security_authority(config).to_owned(),
            bgg_security_evidence_source: bgg_security_evidence_source(config).to_owned(),
        });
    }
    Err(NO_CANDIDATE_ERROR.to_owned())
}

const REVIEWED_BGG_ENVELOPE: &str = "reviewed_monotone_envelope";
const REVIEWED_BGG_ENVELOPE_SOURCE: &str = "reviewed_monotone_envelope:q27:136";

fn bgg_security_authority(config: &SearchConfig) -> &'static str {
    if config.bgg_security_envelope.as_deref() == Some(REVIEWED_BGG_ENVELOPE) {
        REVIEWED_BGG_ENVELOPE
    } else {
        "external_lattice_estimator"
    }
}

fn bgg_security_evidence_source(config: &SearchConfig) -> &'static str {
    if config.bgg_security_envelope.as_deref() == Some(REVIEWED_BGG_ENVELOPE) {
        REVIEWED_BGG_ENVELOPE_SOURCE
    } else {
        "external_lattice_estimator"
    }
}

/// Returns the reviewed monotone BGG/RLWE envelope floor.  This is deliberately
/// narrower than the external estimator path: the envelope is valid only for
/// the exact reviewed N/CRT/base/sigma/distribution domain and up to q_27.
// This helper is imported by the parameter-search integration tests; Cargo also
// compiles `test_utils.rs` as its own test target, where it has no caller.
#[allow(dead_code)]
pub(crate) fn reviewed_bgg_security_bits(
    config: &SearchConfig,
    candidate: Candidate,
) -> Result<u64, String> {
    if config.bgg_security_envelope.as_deref() != Some(REVIEWED_BGG_ENVELOPE) {
        return Err("reviewed BGG envelope is not selected".to_owned());
    }
    if candidate.log_ring_dimension != 16 ||
        candidate.crt_bits != 50 ||
        candidate.base_bits != 25 ||
        config.error_sigma.to_bits() != 4.0_f64.to_bits()
    {
        return Err(format!("{REVIEWED_BGG_ENVELOPE_SOURCE} does not cover candidate domain"));
    }
    if candidate.crt_depth == 0 || candidate.crt_depth > 27 {
        return Err(format!("{REVIEWED_BGG_ENVELOPE_SOURCE} covers CRT depth 1..=27 only"));
    }
    let q27 = DCRTPolyParams::new(1 << 16, 27, 50, 25).modulus();
    let candidate_q = DCRTPolyParams::new(
        1 << candidate.log_ring_dimension,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
    )
    .modulus();
    if candidate_q.as_ref() > q27.as_ref() {
        return Err(format!("{REVIEWED_BGG_ENVELOPE_SOURCE} rejects modulus above q_27"));
    }
    Ok(136)
}

/// Applies the backend-owned capability check before any security callback or
/// OpenFHE constructor can observe a candidate. The search deliberately keeps
/// no duplicate CRT/base limits of its own.
fn validate_backend_candidate(candidate: Candidate) -> Result<(), String> {
    let ring_dimension = 1usize
        .checked_shl(candidate.log_ring_dimension as u32)
        .ok_or_else(|| "ring dimension shift overflow".to_owned())?;
    let ring_dimension = u32::try_from(ring_dimension)
        .map_err(|_| "ring dimension does not fit the DCRT backend API".to_owned())?;
    DCRTPolyParams::validate_capability(
        ring_dimension,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
    )
    .map_err(|error| error.to_string())
}

/// Builds the concrete public setup and ordinary sparse-LWR program for one
/// candidate.  It does not execute a backend, sample artifacts, or perform a
/// round trip; those are intentionally outside this symbolic search.
pub fn prepare_candidate(
    config: &SearchConfig,
    candidate: Candidate,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    prepare_candidate_with_digits_inner(config, candidate, None, false)
}

pub fn prepare_candidate_with_fixed_digits(
    config: &SearchConfig,
    candidate: Candidate,
    mask_digits: usize,
    fresh_error_digits: usize,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    if mask_digits == 0 || fresh_error_digits == 0 {
        return Err("fixed d_m and d_e must be positive".to_owned().into());
    }
    prepare_candidate_with_digits_inner(
        config,
        candidate,
        Some((mask_digits, fresh_error_digits)),
        true,
    )
}

struct CandidatePreparationContext {
    candidate: Candidate,
    ring_dimension: usize,
    layout: mxx_exponent_lut::ExpLayout,
    profile: SparseLwrPrfProfile,
    crt_moduli: Vec<u64>,
    sampler_layout: BggSamplerLayout,
    refresh: RefreshCompiler,
    generated_layout: PbcGeneratedKeyLayout,
    active_cells: PbcActiveCellIndex,
    pbc_active_count: usize,
    base_p: usize,
    q_l: BigUint,
    derived_fresh_error_digits: usize,
}

fn build_candidate_context(
    config: &SearchConfig,
    candidate: Candidate,
) -> Result<CandidatePreparationContext, CandidatePreparationError> {
    let ring_dimension =
        1usize.checked_shl(candidate.log_ring_dimension as u32).ok_or_else(|| {
            CandidatePreparationError::Fatal("ring dimension shift overflow".to_owned())
        })?;
    validate_backend_candidate(candidate).map_err(CandidatePreparationError::Infeasible)?;
    let layout = native_exp_layout(config.sparse_lwr_modulus, ring_dimension)
        .map_err(CandidatePreparationError::Infeasible)?;
    let profile = SparseLwrPrfProfile::new(layout, config.sparse_lwr_output_modulus / 2)
        .map_err(|error| CandidatePreparationError::Infeasible(error.to_string()))?;
    let dcrt = DCRTPolyParams::new(
        ring_dimension as u32,
        candidate.crt_depth,
        candidate.crt_bits,
        candidate.base_bits,
    );
    let (crt_moduli, _, _) = dcrt.to_crt();
    let modulus = BigInt::from_biguint(num_bigint::Sign::Plus, dcrt.modulus().as_ref().clone());
    let sampler_layout = BggSamplerLayout {
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
        return Err(CandidatePreparationError::Fatal(
            "paper-evaluation PBC profile does not use the reviewed retry count".to_owned(),
        ));
    }
    let mut support_rng = StdRng::from_seed([0x6a; 32]);
    let support = sample(&mut support_rng, config.sparse_lwr_universe, config.sparse_lwr_weight)
        .into_iter()
        .collect::<Vec<_>>();
    let generated = generate_key_layout(&pbc_parameters, PbcRootSeed([0x19; 32]), &support)
        .map_err(|error| {
            CandidatePreparationError::Fatal(format!("PBC layout generation: {error}"))
        })?;
    generated.private_schedule().validate(&generated.public_layout).map_err(|error| {
        CandidatePreparationError::Fatal(format!("PBC schedule validation: {error}"))
    })?;
    let active_cells = PbcActiveCellIndex::build(&generated.public_layout).map_err(|error| {
        CandidatePreparationError::Fatal(format!("PBC active-cell index: {error}"))
    })?;
    let pbc_active_count = active_cells.len();
    let base_p = config.sparse_lwr_output_modulus;
    let q_l = BigUint::from(config.sparse_lwr_modulus);
    let derived_fresh_error_digits = fresh_error_base_p_digit_count(&base_p.into(), &crt_moduli)
        .map_err(|error| CandidatePreparationError::Infeasible(error.to_string()))?;
    Ok(CandidatePreparationContext {
        candidate,
        ring_dimension,
        layout,
        profile,
        crt_moduli,
        sampler_layout,
        refresh,
        generated_layout: generated,
        active_cells,
        pbc_active_count,
        base_p,
        q_l,
        derived_fresh_error_digits,
    })
}

fn setup_for_candidate_context(
    config: &SearchConfig,
    context: &CandidatePreparationContext,
    mask_digits: usize,
    fresh_error_digits: usize,
) -> Result<RefreshSetupParameters, CandidatePreparationError> {
    let sigma_decoder = RealExpr::from_f64_exact(config.decoder_sigma)
        .map_err(|error| CandidatePreparationError::Fatal(format!("decoder sigma: {error}")))?;
    let sigma_encoding = RealExpr::from_f64_exact(config.error_sigma)
        .map_err(|error| CandidatePreparationError::Fatal(format!("encoding sigma: {error}")))?;
    Ok(RefreshSetupParameters::new(
        [0x52; 32],
        context.base_p,
        config.secret_dimension,
        context.ring_dimension,
        mask_digits,
        fresh_error_digits,
        config.mask_statistical_security_bits,
        context.layout.width(),
        context.sampler_layout.clone(),
        context.refresh.clone(),
        sigma_decoder,
        sigma_encoding,
        "refresh-parameter-search",
    )
    .with_decoder_preimage_bound(config.decoder_preimage_bound.clone()))
}

/// Builds the actual verification program with fresh test-owned PBC randomness.
/// Search fixtures remain deterministic; runtime correctness never reuses their seeds.
pub fn round_trip_bundle(
    config: &SearchConfig,
    candidate: Candidate,
    mask_digits: usize,
    fresh_digits: usize,
    plaintext_exponent: usize,
) -> Result<(RefreshParameterSimulationBundle, PbcGeneratedKeyLayout, SparseLwrPrfProfile), String>
{
    let mut context = build_candidate_context(config, candidate).map_err(|e| e.to_string())?;
    let mut rng = rand::rng();
    let support = sample(&mut rng, config.sparse_lwr_universe, config.sparse_lwr_weight).into_vec();
    let root_seed = rand::Rng::random(&mut rng);
    context.generated_layout = generate_key_layout(
        &PbcParameters::paper_evaluation(config.sparse_lwr_universe, config.sparse_lwr_weight),
        PbcRootSeed(root_seed),
        &support,
    )
    .map_err(|e| e.to_string())?;
    let setup = setup_for_candidate_context(config, &context, mask_digits, fresh_digits)
        .map_err(|e| e.to_string())?;
    let plaintext = setup.layout.ring().constant(
        (1, 1),
        mxx_ir_core::node::ConstantMatrix::Rotation { exponent: plaintext_exponent.into() },
    );
    let bundle = RefreshParameterSimulationRequest::new(
        setup,
        context.profile,
        context.generated_layout.clone(),
        rand::Rng::random(&mut rng),
        RealExpr::from_f64_exact(config.decoder_sigma).map_err(|e| e.to_string())?,
        plaintext,
        1,
    )
    .map_err(|e| e.to_string())?
    .build()
    .map_err(|e| e.to_string())?;
    Ok((bundle, context.generated_layout, context.profile))
}

fn prepare_candidate_with_digits_inner(
    config: &SearchConfig,
    candidate: Candidate,
    fixed_digits: Option<(usize, usize)>,
    build_full_bundle: bool,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    let context = build_candidate_context(config, candidate)?;
    match fixed_digits {
        Some((mask_digits, fresh_error_digits)) => prepare_candidate_from_context(
            config,
            &context,
            mask_digits,
            fresh_error_digits,
            build_full_bundle,
            None,
            None,
            None,
        ),
        None => {
            let base_p_big = BigUint::from(context.base_p);
            let (first_digits, last_digits) =
                mask_digit_search_bounds(&base_p_big, &context.q_l, &context.crt_moduli)?;
            let invariant_setup = setup_for_candidate_context(
                config,
                &context,
                first_digits,
                context.derived_fresh_error_digits,
            )?;
            let (gain_plan, invariant_routes) =
            RefreshParameterSimulationRequest::noise_gain_and_route_invariants_from_public_layout(
                &invariant_setup,
                context.profile,
                &context.generated_layout.public_layout,
                &context.active_cells,
            )
            .map_err(|error| CandidatePreparationError::Infeasible(error.to_string()))?;
            let mut first_prepared = prepare_candidate_from_context(
                config,
                &context,
                first_digits,
                context.derived_fresh_error_digits,
                false,
                Some(&gain_plan),
                Some(&invariant_routes),
                None,
            )?;
            let pre_refresh = first_prepared.pre_refresh.clone();
            let scan_base = first_prepared
                .noise_snapshot
                .as_ref()
                .ok_or_else(|| {
                    CandidatePreparationError::Fatal("compact scan snapshot missing".to_owned())
                })?
                .clone();
            let mask_gains = (first_digits..=last_digits)
                .map(|mask_digits| {
                    RefreshParameterSimulationRequest::mask_route_gain_for_digit_count(
                        &invariant_setup,
                        mask_digits,
                    )
                    .map_err(|error| CandidatePreparationError::Infeasible(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (offset, mask_digits) in (first_digits..=last_digits).enumerate() {
                let accepted = if offset == 0 {
                    check_refresh_bundle(config, candidate, &mut first_prepared)
                        .map_err(CandidatePreparationError::Fatal)?
                } else {
                    let scan_snapshot =
                        snapshot_with_mask_gain(&scan_base, mask_digits, &mask_gains[offset])
                            .map_err(CandidatePreparationError::Infeasible)?;
                    if config.noise_model == NoiseSearchModel::AverageCase {
                        scan_snapshot
                            .average_refresh_accepted_from_pre_refresh(
                                &average_case_config(config)
                                    .map_err(CandidatePreparationError::Fatal)?,
                                pre_refresh.as_ref().ok_or_else(|| {
                                    CandidatePreparationError::Fatal(
                                        "compact scan pre-refresh missing".to_owned(),
                                    )
                                })?,
                            )
                            .map_err(|error| {
                                CandidatePreparationError::Infeasible(error.to_string())
                            })?
                    } else {
                        scan_snapshot
                            .worst_refresh_accepted_from_pre_refresh(
                                pre_refresh.as_ref().ok_or_else(|| {
                                    CandidatePreparationError::Fatal(
                                        "compact scan pre-refresh missing".to_owned(),
                                    )
                                })?,
                            )
                            .map_err(|error| {
                                CandidatePreparationError::Infeasible(error.to_string())
                            })?
                    }
                };
                if accepted {
                    if offset == 0 {
                        return Ok(first_prepared);
                    }
                    let mut prepared = prepare_candidate_from_context(
                        config,
                        &context,
                        mask_digits,
                        context.derived_fresh_error_digits,
                        false,
                        Some(&gain_plan),
                        Some(&invariant_routes),
                        pre_refresh.as_ref(),
                    )?;
                    let checked = check_refresh_bundle(config, candidate, &mut prepared)
                        .map_err(CandidatePreparationError::Fatal)?;
                    if checked {
                        return Ok(prepared);
                    }
                    return Err(CandidatePreparationError::Fatal(
                        "cached scan and materialized snapshot disagree".to_owned(),
                    ));
                }
            }
            Err(CandidatePreparationError::Infeasible(format!(
                "no mask digit count passed the complete production report in d_m={first_digits}..={last_digits}"
            )))
        }
    }
}

pub(crate) fn snapshot_with_mask_gain(
    base: &ExponentLutNoiseSnapshot,
    mask_digits: usize,
    mask_gain: &RightActionNoiseGain,
) -> Result<ExponentLutNoiseSnapshot, String> {
    let mut snapshot = base.clone();
    let mut refresh = snapshot.refresh.clone();
    refresh.mask_digit_count = mask_digits;
    let topology = refresh
        .transcript_topology
        .as_mut()
        .ok_or_else(|| "compact snapshot transcript topology is missing".to_owned())?;
    topology.mask_digit_count = mask_digits;
    let mask_events = topology
        .crt_slot_count
        .checked_mul(topology.component_columns)
        .and_then(|value| value.checked_mul(topology.coefficient_count))
        .and_then(|value| value.checked_mul(mask_digits))
        .ok_or_else(|| "compact scan mask event count overflows".to_owned())?;
    let fresh_events = topology
        .component_columns
        .checked_mul(topology.coefficient_count)
        .and_then(|value| value.checked_mul(topology.fresh_digit_count))
        .ok_or_else(|| "compact scan fresh event count overflows".to_owned())?;
    let transcript_coordinates = mask_events
        .checked_add(fresh_events)
        .ok_or_else(|| "compact scan event count overflows".to_owned())?;
    refresh
        .smudging
        .as_mut()
        .ok_or_else(|| "compact snapshot smudging parameters are missing".to_owned())?
        .transcript_coordinates = transcript_coordinates;
    let mut route_gains = refresh
        .route_gains
        .clone()
        .ok_or_else(|| "compact snapshot route gains are missing".to_owned())?;
    route_gains.mask = mask_gain.clone();
    refresh.route_gains = Some(route_gains);
    let digit_variance = centered_base_p_digit_variance(&refresh.base_p.clone().into())
        .map_err(|error| error.to_string())?
        .scaled(&mask_gain.l2_squared);
    refresh.mask_digit_variances = vec![digit_variance; refresh.slot_count];
    snapshot.refresh = refresh;
    Ok(snapshot)
}

fn prepare_candidate_from_context(
    config: &SearchConfig,
    context: &CandidatePreparationContext,
    mask_digits: usize,
    fresh_error_digits: usize,
    build_full_bundle: bool,
    cached_gain_plan: Option<&mxx_exponent_lut::noise_refresh::simulation::NoiseGainPlan>,
    cached_invariant_routes: Option<&RefreshNoiseGainDecompositions>,
    cached_pre_refresh: Option<&PreRefreshNoiseEvaluation>,
) -> Result<PreparedCandidate, CandidatePreparationError> {
    let candidate = context.candidate;
    let ring_dimension = context.ring_dimension;
    let profile = context.profile;
    let pbc_active_count = context.pbc_active_count;
    let generated_layout = &context.generated_layout;
    if fresh_error_digits != context.derived_fresh_error_digits {
        return Err(CandidatePreparationError::Infeasible(
            "fixed fresh-error digit count is not the setup-derived injective count".to_owned(),
        ));
    }
    if mask_digits == 0 || fresh_error_digits == 0 {
        return Err(CandidatePreparationError::Fatal(
            "native digit counts must be positive".to_owned(),
        ));
    }
    let setup = setup_for_candidate_context(config, context, mask_digits, fresh_error_digits)?;
    let selected_preimage_bound_expr = setup.resolve_decoder_preimage_bound().map_err(|error| {
        CandidatePreparationError::Infeasible(format!("selected decoder bound: {error}"))
    })?;
    let official_preimage_bound_expr = setup
        .clone()
        .with_decoder_preimage_bound(TrapdoorPreimageCutoffPolicy::Default)
        .resolve_decoder_preimage_bound()
        .map_err(|error| {
            CandidatePreparationError::Infeasible(format!("official decoder bound: {error}"))
        })?;
    let selected_value =
        selected_preimage_bound_expr.evaluate(&mxx_ir_core::ParamEnv::default()).map_err(
            |error| CandidatePreparationError::Fatal(format!("selected bound evaluation: {error}")),
        )?;
    let official_value =
        official_preimage_bound_expr.evaluate(&mxx_ir_core::ParamEnv::default()).map_err(
            |error| CandidatePreparationError::Fatal(format!("official bound evaluation: {error}")),
        )?;
    validate_decoder_preimage_cutoff(
        &selected_value.to_biguint().ok_or_else(|| {
            CandidatePreparationError::Fatal(
                "selected decoder bound must be non-negative".to_owned(),
            )
        })?,
        &official_value.to_biguint().ok_or_else(|| {
            CandidatePreparationError::Fatal(
                "official decoder bound must be non-negative".to_owned(),
            )
        })?,
    )
    .map_err(CandidatePreparationError::Fatal)?;
    let prf_component_count = setup.prf_component_count();
    let prf_label_count = setup
        .refresh
        .crt_plaintext_moduli
        .len()
        .checked_mul(prf_component_count)
        .and_then(|value| value.checked_mul(setup.coefficient_count))
        .and_then(|value| value.checked_mul(mask_digits))
        .and_then(|mask| {
            prf_component_count
                .checked_mul(setup.coefficient_count)
                .and_then(|value| value.checked_mul(fresh_error_digits))
                .and_then(|fresh| mask.checked_add(fresh))
        })
        .ok_or_else(|| {
            CandidatePreparationError::Fatal("refresh PRF label count overflows usize".to_owned())
        })?;
    let prf_value_count = prf_label_count.checked_mul(pbc_active_count).ok_or_else(|| {
        CandidatePreparationError::Fatal("refresh PRF value count overflows usize".to_owned())
    })?;
    let pbc_layout = &generated_layout.public_layout;
    let layout_id = pbc_layout.layout_id.0;
    let pbc_attempts_used = pbc_layout.accepted_attempt + 1;
    let bucket_width = pbc_layout.bucket_width;
    let prf_coefficient_count = setup.coefficient_count;
    let program = SparseLwrPrfProgram::new(
        profile,
        pbc_layout.bucket_width,
        pbc_layout.parameters.bucket_count,
    )
    .map_err(|error| CandidatePreparationError::Fatal(format!("native PRF program: {error}")))?;
    let program_id = *program.id().as_bytes();
    let (bundle, noise_snapshot) = if build_full_bundle {
        // Fixed-digit callers are selected candidates used by benchmark
        // estimation; retain the complete production benchmark graph there.
        let expected_plaintext = setup.layout.ring().polynomial([BigInt::zero().into()]);
        let request = RefreshParameterSimulationRequest::new(
            setup,
            profile,
            generated_layout.clone(),
            [0x4b; 32],
            RealExpr::from_f64_exact(config.decoder_sigma).map_err(|error| {
                CandidatePreparationError::Fatal(format!("decoder sigma: {error}"))
            })?,
            expected_plaintext,
            1,
        )
        .map_err(|error| {
            CandidatePreparationError::Fatal(format!("refresh simulation request: {error}"))
        })?;
        let bundle = request.build_benchmark().map_err(|error| {
            CandidatePreparationError::Infeasible(format!("refresh benchmark graph: {error}"))
        })?;
        (Some(bundle), None)
    } else {
        // Ordinary search uses the same setup-owned snapshot construction but
        // does not expand the label-major executable graph for every mask
        // digit candidate.
        let snapshot = if let (Some(gain_plan), Some(invariant_routes)) =
            (cached_gain_plan, cached_invariant_routes)
        {
            RefreshParameterSimulationRequest::build_noise_snapshot_from_public_layout_with_cached_routes(
                setup,
                profile,
                pbc_layout,
                &context.active_cells,
                gain_plan,
                invariant_routes,
            )
        } else if let Some(gain_plan) = cached_gain_plan {
            RefreshParameterSimulationRequest::build_noise_snapshot_from_public_layout_with_gain_plan(
                    setup,
                    profile,
                    pbc_layout,
                    &context.active_cells,
                    gain_plan,
                )
        } else {
            RefreshParameterSimulationRequest::build_noise_snapshot_from_public_layout(
                setup,
                profile,
                pbc_layout,
                &context.active_cells,
            )
        }
            .map_err(|error| CandidatePreparationError::Infeasible(format!("compact noise snapshot: {error}")))?;
        (None, Some(snapshot))
    };
    let pre_refresh = noise_snapshot.as_ref().map(|snapshot| {
        cached_pre_refresh.cloned().unwrap_or_else(|| {
            snapshot
                .evaluate_pre_refresh()
                .expect("validated compact snapshot pre-refresh evaluation")
        })
    });
    Ok(PreparedCandidate {
        candidate,
        ring_dimension,
        bucket_width,
        official_preimage_bound: official_value,
        selected_preimage_bound: selected_value,
        layout_id,
        program_id,
        pbc_attempts_used,
        mask_base_p_digit_count: mask_digits,
        fresh_error_base_p_digit_count: fresh_error_digits,
        prf_component_count,
        prf_coefficient_count,
        prf_active_count: pbc_active_count,
        prf_label_count,
        prf_value_count,
        bundle,
        noise_snapshot,
        pre_refresh,
        production_report: None,
        average_report: None,
    })
}

/// Accepts a selected decoder cutoff only when it is at least the resolved
/// official cutoff. This keeps a larger explicitly reviewed cutoff possible
/// while rejecting the unsafe below-official boundary.
pub fn validate_decoder_preimage_cutoff(
    selected: &BigUint,
    official: &BigUint,
) -> Result<(), String> {
    if selected < official {
        return Err("selected decoder preimage cutoff is below the official bound".to_owned());
    }
    Ok(())
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

/// Returns the finite `d_m` interval that may satisfy the strict deterministic
/// CRT precondition.  The lower endpoint is exactly `ceil_log_P(Q_L)`; the
/// upper endpoint is the last digit before `B_m >= min kappa`.  The exact
/// report checks `B_m + 2F < kappa`; this weaker monotone stopping condition
/// cannot skip a valid candidate and leaves the final decision to production
/// report evaluation.
fn mask_digit_search_bounds(
    base_p: &BigUint,
    q_l: &BigUint,
    crt_moduli: &[u64],
) -> Result<(usize, usize), CandidatePreparationError> {
    if base_p < &BigUint::from(2u8) || q_l.is_zero() || crt_moduli.is_empty() {
        return Err(CandidatePreparationError::Fatal(
            "invalid mask digit-search boundary inputs".to_owned(),
        ));
    }
    let first = minimum_mask_base_p_digit_count(
        q_l.to_usize()
            .ok_or_else(|| CandidatePreparationError::Fatal("Q_L does not fit usize".to_owned()))?,
        base_p.to_usize().ok_or_else(|| {
            CandidatePreparationError::Fatal("base-p modulus does not fit usize".to_owned())
        })?,
    )?;
    let moduli = crt_moduli.iter().copied().map(BigUint::from).collect::<Vec<_>>();
    let full_modulus = moduli.iter().fold(BigUint::one(), |product, modulus| product * modulus);
    let min_spacing = moduli
        .iter()
        .map(|modulus| &full_modulus / modulus)
        .min()
        .ok_or_else(|| CandidatePreparationError::Fatal("missing CRT spacing".to_owned()))?;
    let mut last = first;
    loop {
        let exponent = u32::try_from(last).map_err(|_| {
            CandidatePreparationError::Fatal("mask digit count overflow".to_owned())
        })?;
        let bound = base_p.pow(exponent) - BigUint::one();
        if bound >= min_spacing {
            if last == first {
                return Err(CandidatePreparationError::Infeasible(
                    "strict deterministic mask boundary is below the domain minimum".to_owned(),
                ));
            }
            return Ok((first, last - 1));
        }
        last = last.checked_add(1).ok_or_else(|| {
            CandidatePreparationError::Fatal("mask digit count overflow".to_owned())
        })?;
    }
}

/// Chooses the smallest fresh-error digit count for which B_e=p^d_e-1 is
/// strictly below every prime factor of every authoritative CRT tower.
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
    let fresh_bound = base_p - BigUint::one();
    for modulus in crt_moduli {
        let accepted = fresh_error_below_every_prime_factor(&fresh_bound, &BigUint::from(*modulus))
            .map_err(|error| CandidatePreparationError::Fatal(error.to_string()))?;
        if !accepted {
            return Err(CandidatePreparationError::Infeasible(
                "no fresh-error digit count satisfies B_e < every prime factor of every CRT modulus"
                    .to_owned(),
            ));
        }
    }
    Ok(1)
}

/// Builds the single AverageCase policy used by both d_m selection and the
/// final bundle check.  Keeping this projection centralized prevents a
/// selector from silently evaluating a different event/failure budget than
/// the accepted graph.
pub fn average_case_config(config: &SearchConfig) -> Result<AverageCaseConfig, String> {
    Ok(AverageCaseConfig {
        failure_exponent: u32::try_from(config.security_bits)
            .map_err(|_| "AverageCase security target does not fit u32".to_owned())?,
        allow_average_acceptance: true,
        ..Default::default()
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
    if q_l == 0 || !q_l.is_power_of_two() {
        return Err("native Q must be a positive power of two".to_owned());
    }
    if p < 2 || !p.is_power_of_two() || p > q_l || q_l % p != 0 {
        return Err("native P must be a power of two, satisfy 2 <= P <= Q, and divide Q".to_owned());
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

/// Small real-graph fixture shared by runtime smoke tests.
pub fn tiny_native_refresh_bundle() -> (DCRTPolyParams, RefreshParameterSimulationBundle, usize) {
    let poly_parameters = DCRTPolyParams::new(32, 2, 10, 10);
    let modulus = poly_parameters.modulus().as_ref().clone();
    let (crt_moduli, _, _) = poly_parameters.to_crt();
    let ring = mxx_dsl::Ring::new(
        BigInt::from_biguint(num_bigint::Sign::Plus, modulus.clone()),
        poly_parameters.ring_dimension() as usize,
    );
    let setup = RefreshSetupParameters::new(
        [0x91; 32],
        4,
        2,
        32,
        3,
        1,
        1,
        16,
        BggSamplerLayout {
            modulus: BigInt::from_biguint(num_bigint::Sign::Plus, modulus.clone()).into(),
            ring_dimension: 32.into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: (1usize << 10).into(),
        },
        RefreshCompiler {
            full_modulus: BigInt::from_biguint(num_bigint::Sign::Plus, modulus).into(),
            crt_plaintext_moduli: crt_moduli
                .iter()
                .copied()
                .map(|value| IntExpr::constant(BigInt::from(value)))
                .collect(),
            reconstruction_coefficients: poly_parameters
                .reconst_coeffs()
                .into_iter()
                .map(|value| IntExpr::constant(BigInt::from_biguint(num_bigint::Sign::Plus, value)))
                .collect(),
        },
        1.into(),
        7.into(),
        "tiny-native-refresh",
    );
    let pbc = PbcParameters::custom(1, 1, 2, 2, 1, None);
    let generated =
        generate_key_layout(&pbc, PbcRootSeed([0x92; 32]), &[0]).expect("tiny native PBC layout");
    let active_count = PbcActiveCellIndex::build(&generated.public_layout)
        .expect("tiny native active cells")
        .len();
    let profile =
        SparseLwrPrfProfile::new(mxx_exponent_lut::ExpLayout { stride_log2: 1, width_log2: 4 }, 2)
            .expect("tiny native profile");
    let online_label_count = 3 * setup.prf_component_count() * setup.coefficient_count;
    let total_labels = online_label_count * active_count;
    let request = RefreshParameterSimulationRequest::new(
        setup,
        profile,
        generated,
        [0x93; 32],
        1.into(),
        ring.identity(1),
        1,
    )
    .expect("tiny native simulation request");
    (poly_parameters, request.build().expect("tiny native refresh bundle"), total_labels)
}
