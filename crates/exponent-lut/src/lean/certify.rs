//! Final acceptance gate for an actual refresh implementation and backend.
use std::{collections::BTreeMap, fs, io::Write, path::Path, time::Duration};

use mxx_ir_core::ParamEnv;
use mxx_runtime::lean::LeanBackendArtifact;

use super::check::{CheckedCandidate, check_candidate};
use crate::noise_refresh::setup::RefreshParameterSimulationBundle;

pub const PROOF_MODULE: &str = "RefreshCorrectness";
pub const PROOF_THEOREM: &str = "ExponentGeneratedRefreshCorrectness.correctness";
/// Default shared wall-clock budget for the entire proof closure (six hours).
/// Individual generated modules can take more than thirteen minutes.
pub const CHECK_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);

/// `MXX_EXPONENT_LUT_LEAN_TIMEOUT_SECONDS`: positive whole seconds for the
/// entire checker invocation, including dependency builds and the final audit.
/// Unset selects six hours; malformed, zero and overflowing values are fatal.
/// Explicit `certify` timeout arguments remain authoritative.
pub fn check_timeout() -> Result<Duration, Box<dyn std::error::Error>> {
    parse_check_timeout(std::env::var("MXX_EXPONENT_LUT_LEAN_TIMEOUT_SECONDS"))
}

fn parse_check_timeout(value: Result<String, std::env::VarError>) -> Result<Duration, Error> {
    let value = match value {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(CHECK_TIMEOUT),
        Err(error) => return Err(error.into()),
    };
    let seconds = value.parse::<u64>().map_err(
        |_| "MXX_EXPONENT_LUT_LEAN_TIMEOUT_SECONDS must be a positive integer in seconds",
    )?;
    if seconds == 0 || std::time::Instant::now().checked_add(Duration::from_secs(seconds)).is_none()
    {
        return Err("MXX_EXPONENT_LUT_LEAN_TIMEOUT_SECONDS is zero or too large".into());
    }
    Ok(Duration::from_secs(seconds))
}

type Error = Box<dyn std::error::Error>;

fn complete_sources(
    directory: &Path,
    mut sources: BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, Error> {
    if !sources.contains_key(PROOF_MODULE) {
        return Err(format!(
            "missing {PROOF_MODULE}.lean in {}; final application proof is required for acceptance",
            directory.display()
        )
        .into());
    }
    if !sources.contains_key("Claim") {
        return Err("missing IR-generated Claim.lean".into());
    }
    for (name, source) in &sources {
        if fs::read(directory.join(format!("{name}.lean")))? != source.as_bytes() {
            return Err(format!("exported sources differ: {name}").into());
        }
    }
    let certificate = mxx_ir_core::lean::claim::assemble_certificate(PROOF_MODULE, PROOF_THEOREM)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("Certificate.lean"))?;
    file.write_all(certificate.as_bytes())?;
    sources.insert("Certificate".into(), certificate);
    Ok(sources)
}

/// Export into a uniquely reserved directory, then check every generated source.
/// Missing proofs, tools, rejected bounds, timeouts and checker errors are fatal.
/// The durable record contains source/dependency hashes and per-module log paths.
pub fn certify(
    bundle: &RefreshParameterSimulationBundle,
    bindings: &ParamEnv,
    backend: &LeanBackendArtifact,
    artifact_parent: &Path,
    timeout: Duration,
) -> Result<CheckedCandidate, Error> {
    if !bundle.simulate_noise()?.refresh.accepted {
        return Err("worst-case refresh bounds reject this certificate candidate".into());
    }
    fs::create_dir_all(artifact_parent)?;
    let identity =
        bundle.public_identity().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    let directory = (0u64..)
        .find_map(|nonce| {
            let path = artifact_parent.join(format!("{identity}-{nonce}"));
            match fs::create_dir(&path) {
                Ok(()) => Some(Ok(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .ok_or("candidate directory identifiers exhausted")??;
    let exported = super::export_refresh_graphs(bundle, bindings, backend, &directory)?;
    let sources = complete_sources(&directory, exported.sources)?;
    for name in sources.keys() {
        let path = directory.join(format!("{name}.lean"));
        let mut permissions = fs::metadata(&path)?.permissions();
        permissions.set_readonly(true);
        fs::set_permissions(path, permissions)?;
    }
    let checked = check_candidate(&directory, &sources, PROOF_MODULE, PROOF_THEOREM, timeout)?;
    let record = serde_json::json!({
        "sources": checked.sources,
        "dependency_sha256": checked.dependency_sha256,
        "workspace_sources": checked.workspace_sources,
        "check_directory": checked.check_directory,
        "logs": checked.logs,
        "elapsed_seconds": checked.elapsed.as_secs_f64(),
    });
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("checked-record.json"))?;
    file.write_all(&serde_json::to_vec_pretty(&record)?)?;
    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the final RefreshCorrectness exporter and hours of Lean checking"]
    fn test_actual_full_certificate() {
        let (bundle, backend) = super::super::tests::accepted_worst_case_fixture();
        assert!(bundle.simulate_noise().unwrap().refresh.accepted);
        let parent = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/exponent_lut_actual_full_certificate");
        let checked =
            certify(&bundle, &ParamEnv::default(), &backend, &parent, check_timeout().unwrap())
                .expect("actual accepted candidate must pass the full certificate checker");
        assert!(checked.check_directory.join("Certificate.olean").is_file());
        let record_path = checked.check_directory.parent().unwrap().join("checked-record.json");
        let record: serde_json::Value =
            serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
        assert_eq!(record["sources"], serde_json::to_value(&checked.sources).unwrap());
        assert!(checked.logs.contains_key("Certificate"));
        eprintln!("actual full certificate checked; record: {}", record_path.display());
    }

    #[test]
    #[ignore = "requires final RefreshCorrectness and the actual full Lean proof closure"]
    fn test_actual_certificate_rejects_wrong_theorem_type() {
        let (bundle, backend) = super::super::tests::accepted_worst_case_fixture();
        assert!(bundle.simulate_noise().unwrap().refresh.accepted);
        let parent = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/exponent_lut_actual_wrong_theorem");
        fs::create_dir_all(&parent).unwrap();
        let directory =
            tempfile::Builder::new().prefix("candidate-").tempdir_in(&parent).unwrap().keep();
        let exported = super::super::export_refresh_graphs(
            &bundle,
            &ParamEnv::default(),
            &backend,
            &directory,
        )
        .unwrap();
        assert!(
            exported.sources.contains_key(PROOF_MODULE),
            "final application proof must be hooked first"
        );
        let mut sources = exported.sources;
        let actual_claim = sources["Claim"].clone();
        // Preserve the actual IR claim and every other generated source. A proof
        // of True cannot inhabit this execution-dependent correctness proposition.
        // This tests certificate type binding, not a zero-residual counterexample.
        let wrong = "import Claim\nnamespace ExponentGeneratedRefreshCorrectness\ntheorem correctness : True := True.intro\nend ExponentGeneratedRefreshCorrectness\n";
        sources.insert(PROOF_MODULE.into(), wrong.into());
        fs::write(directory.join(format!("{PROOF_MODULE}.lean")), wrong).unwrap();
        let sources = complete_sources(&directory, sources).unwrap();
        assert_eq!(sources["Claim"], actual_claim);
        assert_eq!(fs::read(directory.join("Claim.lean")).unwrap(), actual_claim.as_bytes());
        // Disk and expected bytes agree, so source-integrity rejection is not
        // success. Only Lean's type error in the core Certificate is expected.
        let error = check_candidate(
            &directory,
            &sources,
            PROOF_MODULE,
            PROOF_THEOREM,
            check_timeout().unwrap(),
        )
        .unwrap_err();
        match error {
            super::super::check::CheckError::Process { module, status, log } => {
                assert_eq!(module, "Certificate");
                let diagnostics = fs::read_to_string(&log).unwrap();
                assert!(diagnostics.contains("type mismatch"), "{diagnostics}");
                assert!(diagnostics.contains("GeneratedClaim.CorrectnessClaim"), "{diagnostics}");
                use sha2::{Digest, Sha256};
                let source_hashes: BTreeMap<_, _> = sources
                    .iter()
                    .map(|(name, source)| {
                        (name, format!("{:x}", Sha256::digest(source.as_bytes())))
                    })
                    .collect();
                let record = serde_json::json!({
                    "error_kind": "Process",
                    "module": module,
                    "exit_code": status.code(),
                    "log": log,
                    "sources_sha256": source_hashes,
                    "actual_claim_preserved": true,
                    "expected_failure": "Certificate theorem type mismatch",
                });
                fs::write(
                    directory.join("rejection-record.json"),
                    serde_json::to_vec_pretty(&record).unwrap(),
                )
                .unwrap();
                eprintln!("actual certificate rejected wrong theorem type; log: {}", log.display());
            }
            other => panic!("expected Lean certificate type rejection, got {other}"),
        }
    }

    #[test]
    fn test_check_timeout_configuration() {
        assert_eq!(
            parse_check_timeout(Err(std::env::VarError::NotPresent)).unwrap(),
            CHECK_TIMEOUT
        );
        assert_eq!(parse_check_timeout(Ok("28800".into())).unwrap(), Duration::from_secs(28800));
        for value in ["0", "-1", "abc", "1.5", "18446744073709551616"] {
            assert!(parse_check_timeout(Ok(value.into())).is_err(), "{value}");
        }
        assert!(
            parse_check_timeout(Err(std::env::VarError::NotUnicode("invalid".into()))).is_err()
        );
    }

    #[test]
    fn test_missing_final_proof_is_blocking() {
        let directory = tempfile::tempdir().unwrap();
        let sources = BTreeMap::from([("Claim".into(), "-- claim".into())]);
        fs::write(directory.path().join("Claim.lean"), "-- claim").unwrap();
        assert!(
            complete_sources(directory.path(), sources.clone())
                .unwrap_err()
                .to_string()
                .contains("missing RefreshCorrectness")
        );
        assert!(!directory.path().join("Certificate.lean").exists());
    }

    #[test]
    fn test_missing_ir_claim_is_blocking() {
        let directory = tempfile::tempdir().unwrap();
        let mut sources = BTreeMap::new();
        sources.insert("RefreshCorrectness".into(), "-- proof".into());
        fs::write(directory.path().join("RefreshCorrectness.lean"), "-- proof").unwrap();
        assert!(
            complete_sources(directory.path(), sources.clone())
                .unwrap_err()
                .to_string()
                .contains("missing IR-generated Claim.lean")
        );
        assert!(!directory.path().join("Certificate.lean").exists());
    }

    #[test]
    fn test_exact_certificate_and_tamper_rejection() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("Claim.lean"), "-- claim").unwrap();
        let mut sources = BTreeMap::new();
        sources.insert("RefreshCorrectness".into(), "-- proof".into());
        fs::write(directory.path().join("RefreshCorrectness.lean"), "-- proof").unwrap();
        sources.insert("Claim".into(), "-- claim".into());
        let sources = complete_sources(directory.path(), sources.clone()).unwrap();
        assert_eq!(
            sources["Certificate"],
            mxx_ir_core::lean::claim::assemble_certificate(PROOF_MODULE, PROOF_THEOREM).unwrap()
        );
        assert!(complete_sources(directory.path(), sources.clone()).is_err());
        fs::write(directory.path().join("Claim.lean"), "-- altered bound").unwrap();
        let error = check_candidate(
            directory.path(),
            &sources,
            PROOF_MODULE,
            PROOF_THEOREM,
            Duration::ZERO,
        )
        .unwrap_err();
        assert!(error.to_string().contains("sources differ"));
    }
    #[test]
    fn test_pre_snapshot_source_replacement_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let sources = BTreeMap::from([
            ("Claim".into(), "-- original IR claim".into()),
            (PROOF_MODULE.into(), "-- original application proof".into()),
            ("Backend".into(), "-- original backend".into()),
        ]);
        for target in sources.keys() {
            for (name, source) in &sources {
                fs::write(directory.path().join(format!("{name}.lean")), source).unwrap();
            }
            fs::write(directory.path().join(format!("{target}.lean")), "-- replacement").unwrap();
            assert!(
                complete_sources(directory.path(), sources.clone())
                    .unwrap_err()
                    .to_string()
                    .contains("sources differ")
            );
            assert!(!directory.path().join("Certificate.lean").exists());
        }
    }
}
