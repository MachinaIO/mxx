//! Diamond-owned correctness generation and checking infrastructure.

mod bounds;
mod cache;
mod checker;
mod emit;
mod runner;

pub use bounds::{
    BoundData, BoundEnvironment, BoundError, BoundEvalError, BoundExpr, BoundParameter,
    DIAMOND_BOUND_SCHEMA_VERSION, DiamondBoundParameters, derive_output_noise_bound,
    derive_output_noise_bound_with_parameters,
};
pub use cache::{
    CacheError, DIAMOND_DEPLOYMENT_SCHEMA_VERSION, DiamondCacheKey, DiamondCacheKeyInput,
    DiamondCacheRecord, DiamondCandidateArtifactIdentity, DiamondDeploymentIdentity,
    DiamondDeploymentIdentityInput, DiamondDeploymentSecurityIdentity, LEAN_SOURCE_SCHEMA_VERSION,
    SourceManifest, SourceManifestEntry, cache_directory, canonical_cache_key,
    claim_instance_sha256, materialize_cache_package, read_cache_record_checked, sha256_bytes,
    source_manifest_sha256_for_files, source_manifest_sha256_for_package,
    validate_source_manifest_for_package, write_cache_record,
};
pub use checker::{
    DiamondCorrectnessChecker, DiamondCorrectnessVerdict, LeanSemanticIdentity,
    check_diamond_candidate,
};
pub use emit::{
    DiamondCandidateSemanticRefs, DiamondLeanArtifact, DiamondLeanClaimRequest, DiamondLeanError,
    EmitMode, GeneratedLeanFile, LeanFileManifest, ManifestError, emit_diamond_lean_correctness,
    validate_relative_path,
};
pub use runner::{
    LeanRunOutput, LeanRunRequest, LeanRunner, LeanRunnerError, write_claim_manifest,
    write_claim_manifest_for_check,
};
