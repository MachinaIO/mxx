//! Read-only parameter and estimator support for the sparse-binary LWR PRF.
//!
//! This file is deliberately independent of the refresh search harness.  It
//! contains the paper-anchored LWR candidate and the small amount of glue
//! needed to invoke the pinned Sage estimator.  It does not build a refresh
//! graph or run an integration test.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path, process::Command};

pub const LATTICE_ESTIMATOR_REPOSITORY: &str = "https://github.com/malb/lattice-estimator.git";
pub const TARGET_SECURITY_BITS: f64 = 128.0;
pub const PBC_REPLICATION_FACTOR: usize = 3;
pub const PBC_BUCKET_OFFSET: usize = 3;
pub const EXPECTED_ATTACKS: &[&str] =
    &["arora-gb", "bkw", "usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid", "dual", "dual_hybrid"];

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

    pub fn candidate_a() -> Result<Self, String> {
        Self::new("A", 1450, 29, 512, 32, vec![4096, 8192, 16384, 32768, 65536])
    }

    pub fn derived(&self) -> Result<SparseLwrDerived, String> {
        if self.nu == 0 || self.h == 0 || self.h > self.nu {
            return Err("sparse-binary candidate requires 0 < h <= nu".to_owned());
        }
        if self.q == 0 || self.p == 0 || self.p > self.q || self.q % self.p != 0 {
            return Err("baseline LWR candidate requires 0 < p <= q and p | q".to_owned());
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

/// The direct-LWR algorithm is intentionally a separate coverage item.  A
/// generic LWE estimate must never silently upgrade this field.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DirectLwrStatus {
    NotEvaluated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttackCoverage {
    pub direct_lwr: DirectLwrStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AssessmentStatus {
    Below128Lwe,
    IncompleteAttackCoverage,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseLwrAssessment {
    pub candidate: SparseLwrCandidate,
    pub derived: SparseLwrDerived,
    pub estimator: EstimatorReport,
    pub attack_coverage: AttackCoverage,
    pub status: AssessmentStatus,
}

pub type Assessment = SparseLwrAssessment;

impl SparseLwrAssessment {
    pub fn from_estimator(
        candidate: SparseLwrCandidate,
        estimator: EstimatorReport,
    ) -> Result<Self, String> {
        let derived = candidate.derived()?;
        let status = if estimator.minimum_classical_bits < TARGET_SECURITY_BITS ||
            derived.raw_key_entropy_bits < TARGET_SECURITY_BITS
        {
            AssessmentStatus::Below128Lwe
        } else {
            AssessmentStatus::IncompleteAttackCoverage
        };
        Ok(Self {
            candidate,
            derived,
            estimator,
            attack_coverage: AttackCoverage { direct_lwr: DirectLwrStatus::NotEvaluated },
            status,
        })
    }

    /// The generic LWE surrogate passes the requested 128-bit floor.  The
    /// overall assessment remains incomplete until direct-LWR coverage exists.
    pub fn passes_lwe_floor(&self) -> bool {
        self.estimator.minimum_classical_bits >= TARGET_SECURITY_BITS &&
            self.derived.raw_key_entropy_bits >= TARGET_SECURITY_BITS
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimatorReport {
    pub schema_version: u32,
    pub repository_url: String,
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
        if !self.failures.is_empty() || !self.infinite_attacks.is_empty() {
            return Err("estimator failures and infinite attack costs are fail-closed".to_owned());
        }
        for attack in EXPECTED_ATTACKS {
            let fields = self
                .attacks
                .get(*attack)
                .ok_or_else(|| format!("full estimator result is missing attack {attack}"))?;
            let bits = fields
                .get("rop_log2")
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("attack {attack} has no finite rop_log2"))?;
            if !bits.is_finite() {
                return Err(format!("attack {attack} has non-finite rop_log2"));
            }
        }
        if !self.minimum_classical_bits.is_finite() {
            return Err("minimum classical security is non-finite".to_owned());
        }
        let minimum = EXPECTED_ATTACKS
            .iter()
            .map(|attack| self.attacks[*attack].get("rop_log2").unwrap().as_f64().unwrap())
            .fold(f64::INFINITY, f64::min);
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
                "estimator commit mismatch: expected {expected}, got {}",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_a_derives_reviewed_values() {
        let candidate = SparseLwrCandidate::candidate_a().unwrap();
        let derived = candidate.derived().unwrap();
        assert_eq!(derived.h_prime, 32);
        assert_eq!(derived.delta, 16);
        assert_eq!((derived.error_lower, derived.error_upper), (-8, 7));
        assert_eq!(derived.w_mod, 1024);
        assert_eq!(derived.total_slots, 4350);
        assert!((derived.raw_key_entropy_bits - 201.3444740929121).abs() < 1e-10);
    }

    #[test]
    fn non_dividing_output_modulus_is_rejected() {
        assert!(SparseLwrCandidate::new("bad", 1450, 29, 512, 31, vec![4096]).is_err());
    }

    #[test]
    fn incompatible_encoding_ring_is_rejected() {
        assert!(SparseLwrCandidate::new("bad", 1450, 29, 512, 32, vec![2048]).is_err());
    }

    #[test]
    fn infinity_is_not_treated_as_security() {
        let mut attacks = BTreeMap::new();
        for attack in EXPECTED_ATTACKS {
            attacks.insert((*attack).to_owned(), serde_json::json!({"rop_log2": 200.0}));
        }
        attacks.get_mut("bkw").unwrap()["rop_log2"] = Value::Null;
        let report = EstimatorReport {
            schema_version: 1,
            repository_url: LATTICE_ESTIMATOR_REPOSITORY.to_owned(),
            git_commit: "e35f45b7976a90a79c3c6625a45bbc344c1abc67".to_owned(),
            sage_version: "SageMath version 10.7".to_owned(),
            python_version: "3.12.12".to_owned(),
            cost_model: "MATZOV".to_owned(),
            shape_model: "gsa".to_owned(),
            quantum: false,
            sample_count: "infinity".to_owned(),
            attacks,
            failures: Vec::new(),
            infinite_attacks: vec!["bkw".to_owned()],
            minimum_classical_bits: 200.0,
        };
        assert!(report.validate().is_err());
    }
}
