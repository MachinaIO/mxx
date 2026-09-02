//! Candidate-level Diamond correctness checking.

use super::{
    bounds::{DiamondBoundParameters, derive_output_noise_bound},
    cache::{
        CacheError, DiamondCacheKey, DiamondCacheKeyInput, LEAN_SOURCE_SCHEMA_VERSION,
        canonical_cache_key, claim_instance_sha256, materialize_cache_package,
        read_cache_record_checked, sha256_bytes, source_manifest_sha256_for_package,
    },
    emit::{
        DiamondCandidateSemanticRefs, DiamondLeanClaimRequest, EmitMode,
        emit_diamond_lean_correctness,
    },
    runner::{DIAMOND_PROOF_THEOREM, LeanRunRequest, LeanRunner},
};
use crate::diamond::parameter_search::DiamondSelectedParameters;
use mxx_ir_core::{ValidatedLinkedProgram, encoding::IR_VERSION, render_lean_program};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use std::{
    fs,
    path::{Path, PathBuf},
};

const CHECK_FILE: &str = "MxxGenerated/DiamondCandidate.lean";
const CLAIM_TEMPLATE_SOURCE: &[u8] = include_bytes!("emit.rs");
const DIAMOND_PROOF_SOURCES: &[&[u8]] = &[
    include_bytes!("../../../lean/MxxWe/DiamondWE/Parameters.lean"),
    include_bytes!("../../../lean/MxxWe/DiamondWE/Model.lean"),
    include_bytes!("../../../lean/MxxWe/DiamondWE/Operational.lean"),
    include_bytes!("../../../lean/MxxWe/DiamondWE/Exact.lean"),
    include_bytes!("../../../lean/MxxWe/DiamondWE/Noise.lean"),
    include_bytes!("../../../lean/MxxWe/DiamondWE/Decoder.lean"),
    include_bytes!("../../../lean/MxxWe/DiamondWE/BggFuse.lean"),
    include_bytes!("../../../lean/MxxWe/DiamondWE/Correctness.lean"),
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanSemanticIdentity {
    pub ir_version: u32,
    pub linked_program_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiamondCorrectnessVerdict {
    LeanVerified {
        semantic_identity: LeanSemanticIdentity,
        claim_instance_sha256: [u8; 32],
        theorem: String,
        artifact_directory: PathBuf,
    },
    Rejected {
        bound: BigUint,
        decoder_threshold: BigUint,
    },
    InfrastructureError {
        error: String,
    },
}

#[derive(Clone, Debug)]
pub struct DiamondCorrectnessChecker {
    pub runner: LeanRunner,
    pub cache_target_directory: PathBuf,
    pub theorem: String,
}

impl DiamondCorrectnessChecker {
    pub fn new(cache_target_directory: impl Into<PathBuf>) -> Self {
        Self {
            runner: LeanRunner::default(),
            cache_target_directory: cache_target_directory.into(),
            theorem: DIAMOND_PROOF_THEOREM.to_owned(),
        }
    }

    /// Derive the parameter-pinned bound, apply the decoder prefilter, and verify the generated
    /// candidate with the production Lean runner.
    pub fn check_candidate(
        &self,
        program: &ValidatedLinkedProgram,
        parameters: &DiamondSelectedParameters,
        refs: DiamondCandidateSemanticRefs<'_>,
    ) -> DiamondCorrectnessVerdict {
        if !self.runner.is_production() {
            return infrastructure("production Lean runner cannot be replaced for verification");
        }
        if self.theorem != DIAMOND_PROOF_THEOREM {
            return infrastructure("checker theorem is not the fixed Diamond proof theorem");
        }
        let rendered = match render_lean_program(program, "MxxGenerated.DiamondProgram") {
            Ok(rendered) => rendered,
            Err(error) => return infrastructure(error.to_string()),
        };
        let threshold = decoder_threshold(&parameters.modulus);
        let bound = match derive_output_noise_bound(program, parameters) {
            Ok(bound) => bound,
            Err(error) => return infrastructure(error.to_string()),
        };
        if !decoder_accepts(&parameters.modulus, &bound.value) {
            return DiamondCorrectnessVerdict::Rejected {
                bound: bound.value,
                decoder_threshold: threshold,
            };
        }
        let request = DiamondLeanClaimRequest {
            linked: program,
            program: &rendered,
            parameters,
            bound: &bound,
            refs,
        };
        let artifact = match emit_diamond_lean_correctness(&request, EmitMode::Check) {
            Ok(artifact) => artifact,
            Err(error) => return infrastructure(error.to_string()),
        };
        let claim_source = match artifact.claim_source() {
            Some(source) => source,
            None => return infrastructure("generated Diamond claim source is missing"),
        };
        if artifact.claim_instance_sha256 != sha256_bytes(claim_source) {
            return infrastructure("generated Diamond claim digest is inconsistent");
        }
        let parameters_source = &[];
        let claim_instance_sha256 = claim_instance_sha256(parameters_source, claim_source);
        let lean_package = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lean");
        let check_file = Path::new(CHECK_FILE);
        let source_manifest_sha256 = match source_manifest_sha256_for_package(
            &lean_package,
            &artifact.manifest,
            check_file,
            claim_source,
        ) {
            Ok(digest) => digest,
            Err(error) => return infrastructure(error.to_string()),
        };
        let bound_parameters = match DiamondBoundParameters::from_selected(parameters) {
            Ok(parameters) => parameters,
            Err(error) => return infrastructure(error.to_string()),
        };
        let bound_expression = match serde_json::to_vec(&bound.expression) {
            Ok(bytes) => bytes,
            Err(error) => return infrastructure(error.to_string()),
        };
        let lean_toolchain = match read_trimmed(&lean_package.join("lean-toolchain")) {
            Ok(toolchain) => toolchain,
            Err(error) => return infrastructure(error),
        };
        let primitives_manifest =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../primitives/lean/lake-manifest.json");
        let mathlib_revision = match lake_manifest_git_revision(&primitives_manifest, "mathlib") {
            Ok(revision) => revision,
            Err(error) => return infrastructure(error),
        };
        let key = match canonical_cache_key(DiamondCacheKeyInput {
            ir_version: IR_VERSION,
            lean_source_schema_version: LEAN_SOURCE_SCHEMA_VERSION,
            linked_program_sha256: rendered.linked_program_sha256,
            claim_template_sha256: sha256_bytes(CLAIM_TEMPLATE_SOURCE),
            bound_expression_sha256: sha256_bytes(&bound_expression),
            claim_instance_sha256,
            proof_source_sha256: DIAMOND_PROOF_SOURCES
                .iter()
                .map(|source| sha256_bytes(source))
                .collect(),
            source_manifest_sha256,
            lean_toolchain,
            mathlib_revision,
            parameters: bound_parameters,
            bound,
            theorem: self.theorem.clone(),
            check_file: check_file.to_path_buf(),
            check_file_sha256: sha256_bytes(claim_source),
        }) {
            Ok(key) => key,
            Err(error) => return infrastructure(error.to_string()),
        };
        match read_cache_record_checked(
            &self.cache_target_directory,
            &key,
            parameters_source,
            claim_source,
        ) {
            Ok(Some(_)) => {}
            Ok(None) => {
                match materialize_cache_package(
                    &self.cache_target_directory,
                    &key,
                    &lean_package,
                    &artifact.manifest,
                    &self.theorem,
                    check_file,
                    claim_source,
                ) {
                    Ok(_) | Err(CacheError::AlreadyPublished) => {}
                    Err(error) => return infrastructure(error.to_string()),
                }
            }
            Err(error) => return infrastructure(error.to_string()),
        }
        self.verify_cached_artifact(&key, parameters_source, claim_source)
    }

    /// Re-check a previously materialized cache package and return a verified verdict only after
    /// Lake/Lean succeeds.  The regenerated parameter and claim bytes must match the cache key;
    /// callers cannot turn a stale or missing package into a mathematical rejection.
    pub fn verify_cached_artifact(
        &self,
        key: &DiamondCacheKey,
        parameters_source: &[u8],
        claim_source: &[u8],
    ) -> DiamondCorrectnessVerdict {
        if !self.runner.is_production() {
            return infrastructure("production Lean runner cannot be replaced for verification");
        }
        if key.input.theorem != DIAMOND_PROOF_THEOREM {
            return infrastructure("cache theorem is not the fixed Diamond proof theorem");
        }
        let record = match read_cache_record_checked(
            &self.cache_target_directory,
            key,
            parameters_source,
            claim_source,
        ) {
            Ok(Some(record)) => record,
            Ok(None) => return infrastructure("Lean correctness cache entry is missing"),
            Err(error) => return infrastructure(error.to_string()),
        };
        let request = LeanRunRequest {
            package_directory: record.artifact_directory.clone(),
            check_file: record.check_file.clone(),
            theorem: record.theorem.clone(),
            cache_key: key.sha256,
            check_file_sha256: record.check_file_sha256,
        };
        if let Err(error) = self.runner.run(&request) {
            return infrastructure(error.to_string());
        }
        DiamondCorrectnessVerdict::LeanVerified {
            semantic_identity: LeanSemanticIdentity {
                ir_version: key.input.ir_version,
                linked_program_sha256: key.input.linked_program_sha256,
            },
            claim_instance_sha256: key.input.claim_instance_sha256,
            theorem: record.theorem,
            artifact_directory: record.artifact_directory,
        }
    }
}

fn read_trimmed(path: &Path) -> Result<String, String> {
    let value = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let value = value.trim();
    if value.is_empty() {
        return Err(format!("{} is empty", path.display()));
    }
    Ok(value.to_owned())
}

fn lake_manifest_git_revision(manifest: &Path, dependency: &str) -> Result<String, String> {
    let source = fs::read(manifest)
        .map_err(|error| format!("could not read {}: {error}", manifest.display()))?;
    let document: serde_json::Value = serde_json::from_slice(&source)
        .map_err(|error| format!("could not parse {}: {error}", manifest.display()))?;
    let packages = document
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} has no package list", manifest.display()))?;
    let matches = packages
        .iter()
        .filter(|package| {
            package.get("name").and_then(serde_json::Value::as_str) == Some(dependency)
        })
        .collect::<Vec<_>>();
    let [package] = matches.as_slice() else {
        return Err(format!(
            "{} must contain exactly one {dependency} package entry",
            manifest.display()
        ));
    };
    if package.get("type").and_then(serde_json::Value::as_str) != Some("git") {
        return Err(format!("Lake dependency {dependency} is not a Git package"));
    }
    let revision = package
        .get("rev")
        .and_then(serde_json::Value::as_str)
        .filter(|revision| {
            revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        .ok_or_else(|| format!("Lake dependency {dependency} has no exact 40-hex revision"))?;
    Ok(revision.to_owned())
}

pub fn check_diamond_candidate(
    cache_target_directory: &Path,
    program: &ValidatedLinkedProgram,
    parameters: &DiamondSelectedParameters,
    refs: DiamondCandidateSemanticRefs<'_>,
) -> DiamondCorrectnessVerdict {
    DiamondCorrectnessChecker::new(cache_target_directory)
        .check_candidate(program, parameters, refs)
}

fn infrastructure(error: impl Into<String>) -> DiamondCorrectnessVerdict {
    DiamondCorrectnessVerdict::InfrastructureError { error: error.into() }
}

pub(crate) fn decoder_threshold(modulus: &BigUint) -> BigUint {
    if modulus < &BigUint::from(2u8) {
        return BigUint::ZERO;
    }
    let numerator = BigInt::from(modulus.clone() - BigUint::from(2u8));
    let rounded = (numerator * BigInt::from(2u8) + BigInt::from(4u8)).div_floor(&BigInt::from(8u8));
    rounded.to_biguint().unwrap_or_default()
}

pub(crate) fn decoder_accepts(modulus: &BigUint, noise: &BigUint) -> bool {
    if modulus < &BigUint::from(4u8) {
        return false;
    }
    let quarter = decoder_threshold(modulus);
    let half = modulus / BigUint::from(2u8);
    let noise = noise.clone();
    quarter > noise &&
        modulus > &(BigUint::from(3u8) * &quarter + &noise) &&
        half >= quarter.clone() + &noise &&
        BigUint::from(3u8) * quarter >= half + noise
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            bounds::DiamondBoundParameters,
            cache::{
                DiamondCacheKeyInput, LEAN_SOURCE_SCHEMA_VERSION, claim_instance_sha256,
                materialize_cache_package,
            },
        },
        *,
    };
    use mxx_ir_core::encoding::IR_VERSION;
    use num_bigint::BigUint;
    use std::{fs, path::Path};

    #[test]
    fn insufficient_noise_margin_is_mathematical_rejection() {
        assert!(!decoder_accepts(&BigUint::from(4u8), &BigUint::from(1u8)));
        assert_eq!(decoder_threshold(&BigUint::from(257u16)), BigUint::from(64u8));
    }

    #[test]
    fn threshold_is_zero_for_tiny_moduli() {
        assert_eq!(decoder_threshold(&BigUint::from(1u8)), BigUint::ZERO);
    }

    #[test]
    fn production_lean_package_has_cache_identity_inputs() {
        let lean_package = Path::new(env!("CARGO_MANIFEST_DIR")).join("lean");
        assert!(!read_trimmed(&lean_package.join("lean-toolchain")).unwrap().is_empty());
        let primitives_manifest =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../primitives/lean/lake-manifest.json");
        assert_eq!(
            lake_manifest_git_revision(&primitives_manifest, "mathlib").unwrap(),
            "8f9d9cff6bd728b17a24e163c9402775d9e6a365"
        );
        let manifest = super::super::emit::LeanFileManifest::new([]).unwrap();
        source_manifest_sha256_for_package(
            &lean_package,
            &manifest,
            Path::new("Check.lean"),
            b"import MxxWe\n",
        )
        .unwrap();
    }

    #[test]
    fn lake_manifest_revision_is_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = directory.path().join("lake-manifest.json");
        let revision = "8f9d9cff6bd728b17a24e163c9402775d9e6a365";

        fs::write(
            &manifest,
            format!(r#"{{"packages":[{{"name":"mathlib","type":"git","rev":"{revision}"}}]}}"#,),
        )
        .unwrap();
        assert_eq!(lake_manifest_git_revision(&manifest, "mathlib").unwrap(), revision);

        fs::write(&manifest, r#"{"packages":[{"name":"mathlib","type":"git","rev":"v4.28.0"}]}"#)
            .unwrap();
        assert!(lake_manifest_git_revision(&manifest, "mathlib").is_err());

        fs::write(
            &manifest,
            format!(r#"{{"packages":[{{"name":"other","type":"git","rev":"{revision}"}}]}}"#,),
        )
        .unwrap();
        assert!(lake_manifest_git_revision(&manifest, "mathlib").is_err());

        fs::write(
            &manifest,
            format!(
                r#"{{"packages":[{{"name":"mathlib","type":"git","rev":"{revision}"}},{{"name":"mathlib","type":"git","rev":"{revision}"}}]}}"#,
            ),
        )
        .unwrap();
        assert!(lake_manifest_git_revision(&manifest, "mathlib").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn cached_artifact_is_verified_only_after_runner_success() {
        use std::os::unix::fs::PermissionsExt;

        let target = tempfile::tempdir().unwrap();
        let lean_package = tempfile::tempdir().unwrap();
        fs::write(lean_package.path().join("lakefile.toml"), "name = \"MxxWe\"\n").unwrap();
        let launcher = target.path().join("lake");
        fs::write(&launcher, b"#!/bin/sh\nshift 2\nexec lean \"$@\"\n").unwrap();
        fs::set_permissions(&launcher, fs::Permissions::from_mode(0o755)).unwrap();

        let parameters_source = b"parameters";
        let claim_source = b"claim";
        let parameters = DiamondBoundParameters {
            modulus: BigUint::from(97u8),
            ring_dimension: 8,
            state_rows: 2,
            state_columns: 2,
            gadget_columns: 1,
            error_coefficient_bound: 1u8.into(),
            preimage_coefficient_bound: 1u8.into(),
            gadget_decomposition_bound: 1u8.into(),
            input_steps: 1,
            circuit_layers: 1,
        };
        let bound =
            super::super::bounds::derive_output_noise_bound_from_parameters(&parameters).unwrap();
        let check_file = Path::new("Check.lean");
        let check_contents = b"theorem T : True := by trivial\n";
        let files = super::super::emit::LeanFileManifest::new([]).unwrap();
        let source_manifest_sha256 = super::super::cache::source_manifest_sha256_for_package(
            lean_package.path(),
            &files,
            check_file,
            check_contents,
        )
        .unwrap();
        let key = super::super::cache::canonical_cache_key(DiamondCacheKeyInput {
            ir_version: IR_VERSION,
            lean_source_schema_version: LEAN_SOURCE_SCHEMA_VERSION,
            linked_program_sha256: [7; 32],
            claim_template_sha256: [8; 32],
            bound_expression_sha256: [9; 32],
            claim_instance_sha256: claim_instance_sha256(parameters_source, claim_source),
            proof_source_sha256: vec![[10; 32]],
            source_manifest_sha256,
            lean_toolchain: "test".to_owned(),
            mathlib_revision: "test".to_owned(),
            parameters,
            bound,
            theorem: "T".to_owned(),
            check_file: check_file.to_path_buf(),
            check_file_sha256: super::super::cache::sha256_bytes(check_contents),
        })
        .unwrap();
        materialize_cache_package(
            target.path(),
            &key,
            lean_package.path(),
            &files,
            "T",
            check_file,
            check_contents,
        )
        .unwrap();
        let checker = DiamondCorrectnessChecker {
            runner: LeanRunner::for_test(launcher, std::time::Duration::from_secs(10)),
            cache_target_directory: target.path().to_path_buf(),
            theorem: "T".to_owned(),
        };
        let verdict = checker.verify_cached_artifact(&key, parameters_source, claim_source);
        assert!(matches!(verdict, DiamondCorrectnessVerdict::InfrastructureError { .. }));
    }
}
