// Read-only parameter and estimator support for the sparse-binary LWR PRF.
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

/// Security tier used by the reviewed Phase-1 tuple grid.  A fallback tier is
/// deliberately distinct from the fallback tier, while both are evaluated
/// against the reviewed 100-bit target.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SparseLwrSecurityTier {
    Primary100,
    Fallback100,
}

impl SparseLwrSecurityTier {
    pub const fn target_bits(self) -> u64 {
        match self {
            Self::Primary100 => 100,
            Self::Fallback100 => 100,
        }
    }
}

/// One reviewed Phase-1 point.  The tuple is ordered as `(Q_L, p, nu, h)`;
/// The estimator result is part of the declaration so a checkpoint cannot
/// silently substitute a different model or security policy. Tier is assigned
/// from the reviewed primary/fallback thresholds after evaluating this row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseLwrParameterTuple {
    pub q_l: usize,
    pub p: usize,
    pub nu: usize,
    pub h: usize,
    pub lut_width: usize,
    /// Exact finite minimum over the requested attack set.  This is the
    /// security value used for the threshold comparison; the integer field
    /// below is only its conservative persisted floor.
    pub estimator_minimum_classical_bits: f64,
    pub estimator_security_bits: u64,
}

impl PartialEq for SparseLwrParameterTuple {
    fn eq(&self, other: &Self) -> bool {
        self.q_l == other.q_l &&
            self.p == other.p &&
            self.nu == other.nu &&
            self.h == other.h &&
            self.lut_width == other.lut_width &&
            (self.estimator_minimum_classical_bits - other.estimator_minimum_classical_bits)
                .abs() <=
                1e-12 &&
            self.estimator_security_bits == other.estimator_security_bits
    }
}

impl SparseLwrParameterTuple {
    pub fn candidate(&self) -> Result<SparseLwrCandidate, String> {
        let derived_w = self
            .q_l
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| "Q_L is too large for W_mod derivation".to_owned())?
            .checked_next_power_of_two()
            .ok_or_else(|| "W_mod overflows usize".to_owned())?;
        let mut n_enc_candidates = Vec::new();
        for exponent in (0..=16).map(|index| derived_w.checked_shl(index as u32)) {
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
    lut_width: usize,
    estimator_minimum_classical_bits: f64,
    estimator_security_bits: u64,
) -> SparseLwrParameterTuple {
    SparseLwrParameterTuple {
        q_l,
        p,
        nu,
        h,
        lut_width,
        estimator_minimum_classical_bits,
        estimator_security_bits,
    }
}

/// The finite, ordered primary Phase-1 grid reviewed for integration.  It is
/// intentionally a tuple list, not a generated Cartesian product.
pub fn reviewed_phase1_tuple_grid() -> Vec<SparseLwrParameterTuple> {
    vec![
        // The adjacent lower-ν point is retained as explicit minimality
        // evidence and must reject the 100-bit floor.
        reviewed_tuple(16, 2, 450, 31, 512, 99.95743319144975, 99),
        reviewed_tuple(16, 2, 451, 31, 512, 100.00493881140125, 100),
    ]
}

/// A concrete sparse-binary LWR candidate.  `n_enc_candidates` belongs to the
/// encoding layer; the estimator dimension is always `nu`.
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
        let candidate = Self { candidate_id: candidate_id.into(), nu, h, q, p, n_enc_candidates };
        candidate.derived()?;
        Ok(candidate)
    }

    pub fn derived(&self) -> Result<SparseLwrDerived, String> {
        if self.nu == 0 || self.h == 0 || self.h > self.nu {
            return Err("sparse-binary candidate requires 0 < h <= nu".to_owned());
        }
        if self.p != 2 || self.q == 0 || self.q % 2 != 0 {
            return Err("Phase-1 sparse-LWR candidates require p == 2 and even Q_L".to_owned());
        }
        if self.n_enc_candidates.is_empty() {
            return Err("at least one encoding ring dimension is required".to_owned());
        }

        let delta = self.q / self.p;
        let shift =
            i64::try_from(delta / 2).map_err(|_| "LWR error shift overflows i64".to_owned())?;
        let error_lower = -shift;
        let error_upper = i64::try_from(delta - 1)
            .map_err(|_| "LWR error upper bound overflows i64".to_owned())?
            .checked_sub(shift)
            .ok_or_else(|| "LWR error upper bound overflows i64".to_owned())?;
        let doubled_q_minus_one = self
            .q
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| "Q is too large for W_mod derivation".to_owned())?;
        let w_mod = doubled_q_minus_one
            .checked_next_power_of_two()
            .ok_or_else(|| "W_mod overflows usize".to_owned())?;
        let maximum_ring_dimension = 1usize << 16;
        if self
            .n_enc_candidates
            .iter()
            .any(|&n| n == 0 || n > maximum_ring_dimension || n % w_mod != 0)
        {
            return Err("every N_enc must satisfy W_mod | N_enc <= 2^16".to_owned());
        }

        Ok(SparseLwrDerived {
            delta,
            error_lower,
            error_upper,
            error_stddev: (((delta * delta - 1) as f64) / 12.0).sqrt(),
            h_prime: self
                .h
                .checked_add(PBC_BUCKET_OFFSET)
                .ok_or_else(|| "h' overflows usize".to_owned())?,
            w_mod,
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
    pub delta: usize,
    pub error_lower: i64,
    pub error_upper: i64,
    pub error_stddev: f64,
    pub h_prime: usize,
    pub w_mod: usize,
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
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err("unsupported estimator report schema".to_owned());
        }
        if self.repository_url != LATTICE_ESTIMATOR_REPOSITORY {
            return Err("estimator repository is not the official malb repository".to_owned());
        }
        if self.git_commit.len() != 40 || !self.git_commit.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("estimator report does not contain a pinned Git commit".to_owned());
        }
        if self.sage_version.is_empty() || self.python_version.is_empty() {
            return Err("estimator report is missing runtime versions".to_owned());
        }
        if self.cost_model != "MATZOV" || self.shape_model != "gsa" {
            return Err("estimator report does not use RC.MATZOV with GSA".to_owned());
        }
        if self.quantum || self.sample_count != "infinity" {
            return Err("estimator report is not the requested classical m=infinity run".to_owned());
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
        return Err(format!("Sage estimator failed: {}", String::from_utf8_lossy(&output.stderr)));
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
