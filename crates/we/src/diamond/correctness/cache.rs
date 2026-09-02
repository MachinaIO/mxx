//! Content-addressed cache identities for generated Diamond proofs.

use super::{
    bounds::{
        BoundData, DIAMOND_BOUND_SCHEMA_VERSION, DiamondBoundParameters,
        derive_output_noise_bound_from_parameters,
    },
    emit::LeanFileManifest,
};
use crate::diamond::{DcrtRuntimeRepresentation, DiamondSelectedParameters};
use mxx_ir_core::encoding::IR_VERSION;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use thiserror::Error;

pub const LEAN_SOURCE_SCHEMA_VERSION: u32 = 1;
/// Schema for the deployment identity that binds a semantic proof artifact to a runtime layout.
pub const DIAMOND_DEPLOYMENT_SCHEMA_VERSION: u32 = 1;
pub(crate) const CACHE_MANIFEST_FILE: &str = "manifest.json";
static MATERIALIZE_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiamondCacheKeyInput {
    pub ir_version: u32,
    pub lean_source_schema_version: u32,
    pub linked_program_sha256: [u8; 32],
    pub claim_template_sha256: [u8; 32],
    pub bound_expression_sha256: [u8; 32],
    pub claim_instance_sha256: [u8; 32],
    pub proof_source_sha256: Vec<[u8; 32]>,
    /// Digest of the ordered path/bytes manifest used by the executable Lake package.
    /// This is computed by [`source_manifest_sha256_for_package`], never trusted as a
    /// substitute for the manifest revalidation performed before Lean runs.
    pub source_manifest_sha256: [u8; 32],
    pub lean_toolchain: String,
    pub mathlib_revision: String,
    pub parameters: DiamondBoundParameters,
    pub bound: BoundData,
    pub theorem: String,
    pub check_file: PathBuf,
    pub check_file_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiamondCacheKey {
    pub input: DiamondCacheKeyInput,
    pub sha256: [u8; 32],
}

impl DiamondCacheKey {
    pub fn new(mut input: DiamondCacheKeyInput) -> Result<Self, CacheError> {
        input.proof_source_sha256.sort();
        if input.ir_version != IR_VERSION {
            return Err(CacheError::Key(format!(
                "unsupported IR version {}, expected {IR_VERSION}",
                input.ir_version
            )));
        }
        if input.lean_source_schema_version != LEAN_SOURCE_SCHEMA_VERSION {
            return Err(CacheError::Key(format!(
                "unsupported Lean source schema version {}, expected {LEAN_SOURCE_SCHEMA_VERSION}",
                input.lean_source_schema_version
            )));
        }
        if input.bound.schema_version != DIAMOND_BOUND_SCHEMA_VERSION {
            return Err(CacheError::Key("unsupported bound schema version".to_owned()));
        }
        if input.theorem.is_empty() {
            return Err(CacheError::Key("theorem name must not be empty".to_owned()));
        }
        let evaluated = input.bound.expression.evaluate(&input.bound.environment)?;
        if evaluated != input.bound.value {
            return Err(CacheError::Key(
                "bound value does not match its expression and environment".to_owned(),
            ));
        }
        if input.bound.environment.modulus != input.parameters.modulus ||
            input.bound.environment.ring_dimension != input.parameters.ring_dimension.into() ||
            input.bound.environment.state_rows != input.parameters.state_rows.into() ||
            input.bound.environment.state_columns != input.parameters.state_columns.into() ||
            input.bound.environment.gadget_columns != input.parameters.gadget_columns.into() ||
            input.bound.environment.error_coefficient_bound !=
                input.parameters.error_coefficient_bound ||
            input.bound.environment.preimage_coefficient_bound !=
                input.parameters.preimage_coefficient_bound ||
            input.bound.environment.gadget_decomposition_bound !=
                input.parameters.gadget_decomposition_bound ||
            input.bound.environment.input_steps != input.parameters.input_steps.into() ||
            input.bound.environment.circuit_layers != input.parameters.circuit_layers.into()
        {
            return Err(CacheError::Key(
                "bound environment does not match candidate parameters".to_owned(),
            ));
        }
        let canonical =
            derive_output_noise_bound_from_parameters(&input.parameters).map_err(|error| {
                CacheError::Key(format!("invalid canonical Diamond bound: {error}"))
            })?;
        if input.bound != canonical {
            return Err(CacheError::Key(
                "bound does not equal the canonical Diamond recurrence".to_owned(),
            ));
        }
        super::emit::validate_relative_path(&input.check_file)
            .map_err(|_| CacheError::Key("check file path is invalid".to_owned()))?;
        let bytes = serde_json::to_vec(&input).map_err(CacheError::Json)?;
        let sha256 = Sha256::digest(bytes).into();
        Ok(Self { input, sha256 })
    }

    pub fn hex(&self) -> String {
        hex_bytes(&self.sha256)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CacheError> {
        serde_json::to_vec(&self.input).map_err(CacheError::Json)
    }

    fn recompute_digest(&self) -> Result<[u8; 32], CacheError> {
        Ok(Sha256::digest(self.canonical_bytes()?).into())
    }

    /// Reject a copied or stale semantic key before it is committed to a deployment identity.
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.recompute_digest()? != self.sha256 {
            return Err(CacheError::InvalidRecord);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiamondDeploymentSecurityIdentity {
    /// Security estimate recorded by parameter search for this deployed candidate.
    achieved_security_bits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiamondDeploymentIdentityInput {
    /// The Lean semantic proof/cache key remains a separate identity component.
    semantic_proof_cache_key: DiamondCacheKey,
    runtime_representation: DcrtRuntimeRepresentation,
    schema_version: u32,
    security: DiamondDeploymentSecurityIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiamondDeploymentIdentity {
    input: DiamondDeploymentIdentityInput,
    sha256: [u8; 32],
}

impl DiamondDeploymentIdentity {
    pub fn from_selected(
        semantic_proof_cache_key: DiamondCacheKey,
        selected: &DiamondSelectedParameters,
    ) -> Result<Self, CacheError> {
        semantic_proof_cache_key.validate()?;
        let runtime_representation = selected
            .runtime_representation()
            .map_err(|error| CacheError::Deployment(error.to_string()))?;
        let input = DiamondDeploymentIdentityInput {
            semantic_proof_cache_key,
            runtime_representation,
            schema_version: DIAMOND_DEPLOYMENT_SCHEMA_VERSION,
            security: DiamondDeploymentSecurityIdentity {
                achieved_security_bits: selected.achieved_security_bits,
            },
        };
        let sha256 = Sha256::digest(serde_json::to_vec(&input)?).into();
        Ok(Self { input, sha256 })
    }

    pub fn runtime_representation(&self) -> &DcrtRuntimeRepresentation {
        &self.input.runtime_representation
    }

    pub fn semantic_proof_cache_key(&self) -> &DiamondCacheKey {
        &self.input.semantic_proof_cache_key
    }

    pub fn achieved_security_bits(&self) -> u64 {
        self.input.security.achieved_security_bits
    }

    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub fn hex(&self) -> String {
        hex_bytes(&self.sha256)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CacheError> {
        serde_json::to_vec(&self.input).map_err(CacheError::Json)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        let representation_valid = self.input.runtime_representation.validate_canonical().is_ok();
        if self.input.schema_version != DIAMOND_DEPLOYMENT_SCHEMA_VERSION ||
            self.input.semantic_proof_cache_key.validate().is_err() ||
            !representation_valid ||
            Sha256::digest(self.canonical_bytes()?) != self.sha256.into()
        {
            return Err(CacheError::InvalidRecord);
        }
        Ok(())
    }
}

/// Compatibility spelling for consumers that call the deployment record a candidate artifact.
pub type DiamondCandidateArtifactIdentity = DiamondDeploymentIdentity;

/// Computes a key in one deterministic operation.  Execution nonces are not
/// fields of this type and therefore cannot accidentally enter the key.
pub fn canonical_cache_key(input: DiamondCacheKeyInput) -> Result<DiamondCacheKey, CacheError> {
    DiamondCacheKey::new(input)
}

/// Hash the canonical generated parameter and claim files with explicit
/// lengths, avoiding concatenation ambiguity.
pub fn claim_instance_sha256(parameters_source: &[u8], claim_source: &[u8]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(16 + parameters_source.len() + claim_source.len());
    bytes.extend_from_slice(&(parameters_source.len() as u64).to_le_bytes());
    bytes.extend_from_slice(parameters_source);
    bytes.extend_from_slice(&(claim_source.len() as u64).to_le_bytes());
    bytes.extend_from_slice(claim_source);
    Sha256::digest(bytes).into()
}

pub fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Compute the source-manifest digest before materializing a cache package.
///
/// The input is exactly the generated file set and check file that
/// [`materialize_cache_package`] will publish.  The digest includes the generated local
/// lakefile, copied toolchain file, every local path dependency reachable from the source
/// package's lakefiles, and every generated Lean/TOML/toolchain file.  Absolute paths are never
/// hashed; only stable manifest labels and file bytes are committed.
pub fn source_manifest_sha256_for_package(
    lean_package: &Path,
    files: &LeanFileManifest,
    check_file: &Path,
    check_contents: &[u8],
) -> Result<[u8; 32], CacheError> {
    super::emit::validate_relative_path(check_file).map_err(|_| CacheError::InvalidRecord)?;
    let lean_package = canonical_source_root(lean_package)?;
    if !lean_package.join("lakefile.toml").is_file() {
        return Err(CacheError::InvalidRecord);
    }
    let roots = discover_dependency_roots(std::slice::from_ref(&lean_package))?;
    let seed_index =
        roots.iter().position(|root| root == &lean_package).ok_or(CacheError::InvalidRecord)?;
    let package_lakefile = format!(
        "name = \"mxx-diamond-candidate\"\ndefaultTargets = [\"MxxGenerated\"]\n\n[[require]]\nname = \"MxxWe\"\npath = \"{}\"\n\n[[lean_lib]]\nname = \"MxxGenerated\"\n",
        format!("deps/{seed_index}")
    );
    let mut virtual_files = BTreeMap::<PathBuf, Vec<u8>>::new();
    for (path, contents) in files.files() {
        if include_source_file(path) {
            virtual_files.insert(PathBuf::from("package").join(path), contents.to_vec());
        }
    }
    virtual_files.insert(PathBuf::from("package/lakefile.toml"), package_lakefile.into_bytes());
    if let Some(toolchain) = find_ancestor_file(&lean_package, "lean-toolchain")? {
        virtual_files.insert(PathBuf::from("package/lean-toolchain"), fs::read(toolchain)?);
    }
    let check_label = PathBuf::from("package").join(check_file);
    virtual_files.insert(check_label, check_contents.to_vec());
    let roots = discover_dependency_roots(std::slice::from_ref(&lean_package))?;
    for (index, root) in roots.iter().enumerate() {
        let namespace = format!("imported/{index}");
        let mut actual = BTreeMap::new();
        // Every discovered root is copied below `deps/{index}` during materialization,
        // including the seed package.  Its Lake metadata is therefore part of the bound
        // dependency bytes; only the generated cache package root has its top manifest excluded.
        collect_source_files(root, root, &namespace, &mut actual)?;
        for (label, path) in actual {
            let contents = if path.file_name().is_some_and(|name| name == "lakefile.toml") {
                rewrite_lakefile(&fs::read_to_string(path)?, &root, index, &roots)?
            } else {
                fs::read(path)?
            };
            virtual_files.insert(label, contents);
        }
    }
    source_manifest_digest_bytes(&virtual_files)
}

/// Compute the manifest digest for the low-level generated/check-only cache record API.
pub fn source_manifest_sha256_for_files(
    files: &LeanFileManifest,
    check_file: &Path,
    check_contents: &[u8],
) -> Result<[u8; 32], CacheError> {
    super::emit::validate_relative_path(check_file).map_err(|_| CacheError::InvalidRecord)?;
    let mut virtual_files = BTreeMap::<PathBuf, Vec<u8>>::new();
    for (path, contents) in files.files() {
        if include_source_file(path) {
            virtual_files.insert(PathBuf::from("package").join(path), contents.to_vec());
        }
    }
    virtual_files.insert(PathBuf::from("package").join(check_file), check_contents.to_vec());
    source_manifest_digest_bytes(&virtual_files)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiamondCacheRecord {
    pub cache_key: DiamondCacheKey,
    pub theorem: String,
    pub artifact_directory: PathBuf,
    pub check_file: PathBuf,
    pub check_file_sha256: [u8; 32],
    pub source_manifest: SourceManifest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SourceManifest {
    pub roots: Vec<PathBuf>,
    pub entries: Vec<SourceManifestEntry>,
    pub sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SourceManifestEntry {
    pub label: PathBuf,
    pub path: PathBuf,
    pub sha256: [u8; 32],
}

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("cache key is invalid: {0}")]
    Key(String),
    #[error("cache JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("cache bound expression could not be evaluated: {0}")]
    Bound(#[from] super::bounds::BoundEvalError),
    #[error("cache I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("cache record is corrupt or does not match the requested key")]
    InvalidRecord,
    #[error("cache package has already been published for this key")]
    AlreadyPublished,
    #[error("deployment identity is invalid: {0}")]
    Deployment(String),
}

/// Cache entries are always below Cargo's target directory.
pub fn cache_directory(cargo_target_dir: &Path, key: &DiamondCacheKey) -> PathBuf {
    cargo_target_dir.join("mxx-lean").join("diamond").join(key.hex())
}

pub fn write_cache_record(
    cargo_target_dir: &Path,
    key: &DiamondCacheKey,
    theorem: &str,
    check_file: &Path,
    check_contents: &[u8],
) -> Result<PathBuf, CacheError> {
    validate_key(key)?;
    super::emit::validate_relative_path(check_file).map_err(|_| CacheError::InvalidRecord)?;
    if key.input.check_file != check_file ||
        key.input.check_file_sha256 != sha256_bytes(check_contents)
    {
        return Err(CacheError::InvalidRecord);
    }
    if key.input.theorem != theorem {
        return Err(CacheError::InvalidRecord);
    }
    let directory = cache_directory(cargo_target_dir, key);
    if directory.exists() {
        return Err(CacheError::AlreadyPublished);
    }
    if let Some(parent) = directory.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir_all(&directory)?;
    let check_path = directory.join(check_file);
    if let Some(parent) = check_path.parent() {
        fs::create_dir_all(parent)?;
    }
    atomic_write(&check_path, check_contents)?;
    let source_manifest = source_manifest(&directory, &[])?;
    if source_manifest.sha256 != key.input.source_manifest_sha256 {
        let _ = fs::remove_dir_all(&directory);
        return Err(CacheError::InvalidRecord);
    }
    write_cache_record_in_directory(
        &directory,
        key,
        theorem,
        check_file,
        check_contents,
        source_manifest,
    )
}

fn write_cache_record_in_directory(
    directory: &Path,
    key: &DiamondCacheKey,
    theorem: &str,
    check_file: &Path,
    check_contents: &[u8],
    source_manifest: SourceManifest,
) -> Result<PathBuf, CacheError> {
    let check_path = directory.join(check_file);
    if let Some(parent) = check_path.parent() {
        fs::create_dir_all(parent)?;
    }
    atomic_write(&check_path, check_contents)?;
    let record = DiamondCacheRecord {
        cache_key: key.clone(),
        theorem: theorem.to_owned(),
        artifact_directory: directory.to_path_buf(),
        check_file: check_file.to_path_buf(),
        check_file_sha256: sha256_bytes(check_contents),
        source_manifest,
    };
    let path = directory.join(CACHE_MANIFEST_FILE);
    atomic_write(&path, &serde_json::to_vec_pretty(&record)?)?;
    Ok(path)
}

/// Materialize generated Lean files as a cache-local Lake package.
///
/// The package is addressed solely by the complete cache key.  The source package path is used
/// only in the local `lakefile.toml`; it is never part of the key or generated theorem identity.
pub fn materialize_cache_package(
    cargo_target_dir: &Path,
    key: &DiamondCacheKey,
    lean_package: &Path,
    files: &LeanFileManifest,
    theorem: &str,
    check_file: &Path,
    check_contents: &[u8],
) -> Result<PathBuf, CacheError> {
    validate_key(key)?;
    if key.input.theorem != theorem || key.input.check_file != check_file {
        return Err(CacheError::InvalidRecord);
    }
    super::emit::validate_relative_path(check_file).map_err(|_| CacheError::InvalidRecord)?;
    let lean_package = canonical_source_root(lean_package)?;
    if !lean_package.join("lakefile.toml").is_file() {
        return Err(CacheError::Key(format!(
            "Lean package is missing lakefile.toml: {}",
            lean_package.display()
        )));
    }
    let expected_source_manifest =
        source_manifest_sha256_for_package(&lean_package, files, check_file, check_contents)?;
    if expected_source_manifest != key.input.source_manifest_sha256 {
        return Err(CacheError::InvalidRecord);
    }
    let directory = cache_directory(cargo_target_dir, key);
    if directory.exists() {
        return Err(CacheError::AlreadyPublished);
    }
    let parent = directory.parent().ok_or(CacheError::InvalidRecord)?;
    fs::create_dir_all(parent)?;
    let nonce = MATERIALIZE_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{}.{}.{}.tmp", key.hex(), std::process::id(), nonce));
    fs::create_dir(&temporary)?;
    let result = (|| {
        files.write(&temporary).map_err(|error| match error {
            super::emit::ManifestError::Io(error) => CacheError::Io(error),
            other => CacheError::Key(other.to_string()),
        })?;
        let roots = discover_dependency_roots(std::slice::from_ref(&lean_package))?;
        let seed_index =
            roots.iter().position(|root| root == &lean_package).ok_or(CacheError::InvalidRecord)?;
        let lakefile = format!(
            "name = \"mxx-diamond-candidate\"\ndefaultTargets = [\"MxxGenerated\"]\n\n[[require]]\nname = \"MxxWe\"\npath = \"{}\"\n\n[[lean_lib]]\nname = \"MxxGenerated\"\n",
            format!("deps/{seed_index}")
        );
        atomic_write(&temporary.join("lakefile.toml"), lakefile.as_bytes())?;
        for (index, root) in roots.iter().enumerate() {
            let destination = temporary.join("deps").join(index.to_string());
            copy_source_tree(root, &destination)?;
            let lakefile = root.join("lakefile.toml");
            atomic_write(
                &destination.join("lakefile.toml"),
                &rewrite_lakefile(&fs::read_to_string(lakefile)?, root, index, &roots)?,
            )?;
        }
        if let Some(toolchain) = find_ancestor_file(&lean_package, "lean-toolchain")? {
            atomic_write(&temporary.join("lean-toolchain"), &fs::read(toolchain)?)?;
        }
        // The check source is part of the executable package manifest.  Write it before hashing
        // so the recorded digest is identical to the pre-materialization digest.
        let check_path = temporary.join(check_file);
        if let Some(parent) = check_path.parent() {
            fs::create_dir_all(parent)?;
        }
        atomic_write(&check_path, check_contents)?;
        let source_manifest =
            source_manifest(&temporary, &[temporary.join(format!("deps/{seed_index}"))])?;
        if source_manifest.sha256 != key.input.source_manifest_sha256 {
            return Err(CacheError::Key(format!(
                "source manifest changed during materialization: expected {}, actual {}",
                hex_bytes(&key.input.source_manifest_sha256),
                hex_bytes(&source_manifest.sha256)
            )));
        }
        write_cache_record_in_directory(
            &temporary,
            key,
            theorem,
            check_file,
            check_contents,
            source_manifest,
        )?;
        super::runner::write_claim_manifest_for_check(
            &temporary,
            theorem,
            &key.sha256,
            check_file,
            check_contents,
        )
        .map_err(|error| CacheError::Key(error.to_string()))?;
        files.check(&temporary).map_err(|error| CacheError::Key(error.to_string()))?;
        if !temporary.join(CACHE_MANIFEST_FILE).is_file() ||
            !temporary.join("mxx-diamond-claim.json").is_file()
        {
            return Err(CacheError::InvalidRecord);
        }
        // Hashes were computed over the temporary bytes, but the published record must point at
        // the final immutable location.  Relocate only path metadata; never re-read or re-hash
        // through the future path before publication.
        let manifest_path = temporary.join(CACHE_MANIFEST_FILE);
        let mut record: DiamondCacheRecord = serde_json::from_slice(&fs::read(&manifest_path)?)
            .map_err(|_| CacheError::InvalidRecord)?;
        record.artifact_directory = directory.to_path_buf();
        record.source_manifest.roots = record
            .source_manifest
            .roots
            .into_iter()
            .map(|path| relocate_path(&path, &temporary, &directory))
            .collect();
        for entry in &mut record.source_manifest.entries {
            entry.path = relocate_path(&entry.path, &temporary, &directory);
        }
        atomic_write(&manifest_path, &serde_json::to_vec_pretty(&record)?)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temporary, &directory) {
        let _ = fs::remove_dir_all(&temporary);
        return Err(if directory.exists() || error.kind() == io::ErrorKind::AlreadyExists {
            CacheError::AlreadyPublished
        } else {
            CacheError::Io(error)
        });
    }
    Ok(directory)
}

fn relocate_path(path: &Path, from: &Path, to: &Path) -> PathBuf {
    path.strip_prefix(from).map_or_else(|_| path.to_path_buf(), |relative| to.join(relative))
}

fn rewrite_lakefile(
    source: &str,
    root: &Path,
    root_index: usize,
    roots: &[PathBuf],
) -> Result<Vec<u8>, CacheError> {
    let mut output = String::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(value) =
            trimmed.strip_prefix("path").and_then(|v| v.trim_start().strip_prefix('='))
        {
            let value = value.trim();
            if let Some(path) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
                let dependency = root.join(path).canonicalize()?;
                let index = roots
                    .iter()
                    .position(|candidate| candidate == &dependency)
                    .ok_or(CacheError::InvalidRecord)?;
                let indent = &line[..line.len() - line.trim_start().len()];
                output.push_str(&format!("{indent}path = \"../{index}\"\n"));
                continue;
            }
        }
        output.push_str(line);
        output.push('\n');
    }
    if root_index == 0 { Ok(output.into_bytes()) } else { Ok(output.into_bytes()) }
}

fn copy_source_tree(source: &Path, destination: &Path) -> Result<(), CacheError> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let ty = entry.file_type()?;
        if ty.is_symlink() {
            return Err(CacheError::InvalidRecord);
        }
        if path.file_name().is_some_and(|name| name == "target") {
            continue;
        }
        let relative = path.strip_prefix(source).map_err(|_| CacheError::InvalidRecord)?;
        let target = destination.join(relative);
        if ty.is_dir() {
            // `.lake` is generated resolver/build state.  The tracked lakefile and any tracked
            // root `lake-manifest.json` already pin source dependencies; copying `.lake` would
            // import machine-local package caches and their ordinary symlinks.
            if path.file_name().is_some_and(|name| name == ".lake") {
                continue;
            }
            copy_source_tree(&path, &target)?;
        } else if ty.is_file() {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

fn find_ancestor_file(start: &Path, file_name: &str) -> Result<Option<PathBuf>, CacheError> {
    let mut directory = start;
    loop {
        let candidate = directory.join(file_name);
        if let Ok(metadata) = fs::symlink_metadata(&candidate) {
            if metadata.file_type().is_symlink() {
                return Err(CacheError::InvalidRecord);
            }
            if metadata.is_file() {
                return Ok(Some(candidate));
            }
        }
        directory = match directory.parent() {
            Some(parent) => parent,
            None => return Ok(None),
        };
    }
}

fn atomic_write(path: &Path, contents: &[u8]) -> io::Result<()> {
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    fs::write(&temporary, contents)?;
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

fn source_manifest(
    package_directory: &Path,
    imported_roots: &[PathBuf],
) -> Result<SourceManifest, CacheError> {
    let package_directory = canonical_source_root(package_directory)?;
    let seeds = if imported_roots.is_empty() && package_directory.join("lakefile.toml").is_file() {
        vec![package_directory.clone()]
    } else {
        imported_roots.to_vec()
    };
    let imported = discover_import_roots(&package_directory, &seeds)?;
    let mut roots = vec![package_directory.clone()];
    roots.extend(imported.iter().cloned());
    let mut files = BTreeMap::<PathBuf, PathBuf>::new();
    collect_source_files(&package_directory, &package_directory, "package", &mut files)?;
    for (index, root) in imported.iter().enumerate() {
        collect_source_files(root, root, &format!("imported/{index}"), &mut files)?;
    }
    let entries = files
        .iter()
        .map(|(label, path)| {
            Ok(SourceManifestEntry {
                label: label.clone(),
                path: path.clone(),
                sha256: sha256_bytes(&fs::read(path)?),
            })
        })
        .collect::<Result<Vec<_>, CacheError>>()?;
    let sha256 = source_manifest_digest(&files)?;
    Ok(SourceManifest { roots, entries, sha256 })
}

/// Resolve every local Lake path dependency reachable from the supplied package roots.  Lake
/// package names are not sufficient for a source manifest: a changed local dependency must be
/// visible even when its package name and revision are unchanged.
fn discover_import_roots(
    package_directory: &Path,
    seeds: &[PathBuf],
) -> Result<Vec<PathBuf>, CacheError> {
    let package_directory = canonical_source_root(package_directory)?;
    let mut roots = discover_dependency_roots(seeds)?;
    roots.retain(|root| root != &package_directory);
    Ok(roots)
}

fn canonical_source_root(path: &Path) -> Result<PathBuf, CacheError> {
    let metadata = fs::symlink_metadata(path).map_err(CacheError::Io)?;
    if metadata.file_type().is_symlink() {
        return Err(CacheError::InvalidRecord);
    }
    Ok(path.canonicalize()?)
}

fn discover_dependency_roots(seeds: &[PathBuf]) -> Result<Vec<PathBuf>, CacheError> {
    let mut pending = seeds.iter().map(PathBuf::as_path).map(Path::to_path_buf).collect::<Vec<_>>();
    let mut discovered = BTreeMap::<PathBuf, ()>::new();
    while let Some(root) = pending.pop() {
        let root = canonical_source_root(&root)?;
        if discovered.contains_key(&root) {
            continue;
        }
        if !root.join("lakefile.toml").is_file() {
            return Err(CacheError::Key(format!(
                "Lake path dependency is missing lakefile.toml: {}",
                root.display()
            )));
        }
        discovered.insert(root.clone(), ());
        for dependency in lake_path_dependencies(&root)? {
            pending.push(dependency);
        }
    }
    Ok(discovered.into_keys().collect())
}

fn lake_path_dependencies(package_directory: &Path) -> Result<Vec<PathBuf>, CacheError> {
    let source = fs::read_to_string(package_directory.join("lakefile.toml"))?;
    let mut dependencies = Vec::new();
    let mut expect_path = false;
    for line in source.lines() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line == "[[require]]" {
            expect_path = true;
            continue;
        }
        if line.starts_with("[[") {
            expect_path = false;
        }
        if expect_path &&
            line.strip_prefix("path").is_some_and(|value| value.trim_start().starts_with('='))
        {
            let value = line.split_once('=').map(|(_, value)| value.trim()).unwrap_or_default();
            let value = value
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .ok_or_else(|| {
                    CacheError::Key(format!("invalid Lake path in {}", package_directory.display()))
                })?;
            dependencies.push(package_directory.join(value));
            expect_path = false;
        }
    }
    Ok(dependencies)
}

fn collect_source_files(
    root: &Path,
    directory: &Path,
    namespace: &str,
    files: &mut BTreeMap<PathBuf, PathBuf>,
) -> Result<(), CacheError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(CacheError::InvalidRecord);
        }
        if file_type.is_file() &&
            matches!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("olean") | Some("ilean")
            )
        {
            return Err(CacheError::InvalidRecord);
        }
        if file_type.is_dir() {
            if path
                .file_name()
                .is_some_and(|name| name == "target" || name == "deps" || name == ".lake")
            {
                continue;
            }
            collect_source_files(root, &path, namespace, files)?;
        } else if file_type.is_file() && include_source_file(&path) {
            let relative = path.strip_prefix(root).map_err(|_| CacheError::InvalidRecord)?;
            files.insert(PathBuf::from(namespace).join(relative), path);
        }
    }
    Ok(())
}

fn include_source_file(path: &Path) -> bool {
    if path.file_name().is_some_and(|name| {
        name == CACHE_MANIFEST_FILE ||
            name == "mxx-diamond-claim.json" ||
            name == "MxxDiamondCheck.lean"
    }) {
        return false;
    }
    matches!(path.extension().and_then(|extension| extension.to_str()), Some("lean") | Some("toml")) ||
        path.file_name().is_some_and(|name| name == "lake-manifest.json") ||
        path.file_name().is_some_and(|name| name == "lean-toolchain")
}

fn source_manifest_digest(files: &BTreeMap<PathBuf, PathBuf>) -> Result<[u8; 32], CacheError> {
    let mut contents = BTreeMap::new();
    for (label, path) in files {
        contents.insert(label.clone(), fs::read(path)?);
    }
    source_manifest_digest_bytes(&contents)
}

fn source_manifest_digest_bytes(
    files: &BTreeMap<PathBuf, Vec<u8>>,
) -> Result<[u8; 32], CacheError> {
    let mut bytes = Vec::new();
    for (label, contents) in files {
        let label = label.to_string_lossy();
        bytes.extend_from_slice(&(label.len() as u64).to_le_bytes());
        bytes.extend_from_slice(label.as_bytes());
        bytes.extend_from_slice(&(contents.len() as u64).to_le_bytes());
        bytes.extend_from_slice(contents);
    }
    Ok(Sha256::digest(bytes).into())
}

impl SourceManifest {
    pub fn validate(&self) -> Result<(), CacheError> {
        let package_root = self.roots.first().ok_or(CacheError::InvalidRecord)?;
        if package_root.join("lakefile.toml").is_file() {
            let discovered = discover_import_roots(package_root, &[package_root.clone()])?;
            if discovered != self.roots[1..].to_vec() {
                return Err(CacheError::InvalidRecord);
            }
        } else if self.roots.len() != 1 {
            return Err(CacheError::InvalidRecord);
        }
        let mut expected = BTreeMap::new();
        for entry in &self.entries {
            if expected.insert(entry.label.clone(), (entry.path.clone(), entry.sha256)).is_some() {
                return Err(CacheError::InvalidRecord);
            }
            let contents = fs::read(&entry.path).map_err(|_| CacheError::InvalidRecord)?;
            if sha256_bytes(&contents) != entry.sha256 {
                return Err(CacheError::InvalidRecord);
            }
        }
        let mut actual = BTreeMap::<PathBuf, PathBuf>::new();
        for (index, root) in self.roots.iter().enumerate() {
            let root = canonical_source_root(root).map_err(|_| CacheError::InvalidRecord)?;
            let namespace =
                if index == 0 { "package".to_owned() } else { format!("imported/{}", index - 1) };
            collect_source_files(&root, &root, &namespace, &mut actual)
                .map_err(|_| CacheError::InvalidRecord)?;
        }
        if actual.len() != expected.len() ||
            actual.iter().any(|(label, path)| {
                expected.get(label).is_none_or(|(expected_path, _)| expected_path != path)
            })
        {
            return Err(CacheError::InvalidRecord);
        }
        let digest = source_manifest_digest(&actual).map_err(|_| CacheError::InvalidRecord)?;
        if digest != self.sha256 {
            return Err(CacheError::InvalidRecord);
        }
        Ok(())
    }
}

/// A malformed or mismatched entry is rejected, never treated as proof.
fn read_cache_record(
    cargo_target_dir: &Path,
    expected: &DiamondCacheKey,
) -> Result<Option<DiamondCacheRecord>, CacheError> {
    validate_key(expected)?;
    let path = cache_directory(cargo_target_dir, expected).join(CACHE_MANIFEST_FILE);
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let record: DiamondCacheRecord =
        serde_json::from_slice(&bytes).map_err(|_| CacheError::InvalidRecord)?;
    let check_path = directory_for_record(cargo_target_dir, expected, &record)?;
    let check_bytes = fs::read(check_path).map_err(|_| CacheError::InvalidRecord)?;
    if record.cache_key != *expected ||
        record.theorem != expected.input.theorem ||
        record.check_file != expected.input.check_file ||
        record.check_file_sha256 != expected.input.check_file_sha256 ||
        record.cache_key.recompute_digest()? != expected.sha256 ||
        record.check_file_sha256 != sha256_bytes(&check_bytes) ||
        record.source_manifest.sha256 != expected.input.source_manifest_sha256 ||
        record.source_manifest.validate().is_err()
    {
        return Err(CacheError::InvalidRecord);
    }
    Ok(Some(record))
}

fn validate_key(key: &DiamondCacheKey) -> Result<(), CacheError> {
    let canonical = DiamondCacheKey::new(key.input.clone())?;
    if canonical != *key {
        return Err(CacheError::InvalidRecord);
    }
    Ok(())
}

/// A cache hit must also re-hash the freshly regenerated Parameters.lean and
/// Claim.lean bytes.  A matching semantic hash alone is insufficient.
pub fn read_cache_record_checked(
    cargo_target_dir: &Path,
    expected: &DiamondCacheKey,
    parameters_source: &[u8],
    claim_source: &[u8],
) -> Result<Option<DiamondCacheRecord>, CacheError> {
    let record = read_cache_record(cargo_target_dir, expected)?;
    if let Some(record) = record {
        if claim_instance_sha256(parameters_source, claim_source) !=
            expected.input.claim_instance_sha256
        {
            return Err(CacheError::InvalidRecord);
        }
        Ok(Some(record))
    } else {
        Ok(None)
    }
}

/// Revalidate all executable package and imported proof inputs immediately before execution.
pub fn validate_source_manifest_for_package(package_directory: &Path) -> Result<(), CacheError> {
    let path = package_directory.join(CACHE_MANIFEST_FILE);
    let bytes = fs::read(path).map_err(|_| CacheError::InvalidRecord)?;
    let record: DiamondCacheRecord =
        serde_json::from_slice(&bytes).map_err(|_| CacheError::InvalidRecord)?;
    record.source_manifest.validate()
}

fn directory_for_record(
    cargo_target_dir: &Path,
    expected: &DiamondCacheKey,
    record: &DiamondCacheRecord,
) -> Result<PathBuf, CacheError> {
    let directory = cache_directory(cargo_target_dir, expected);
    if record.artifact_directory != directory {
        return Err(CacheError::InvalidRecord);
    }
    super::emit::validate_relative_path(&record.check_file)
        .map_err(|_| CacheError::InvalidRecord)?;
    Ok(directory.join(&record.check_file))
}

pub(crate) fn hex_bytes(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diamond::{
        DiamondWeCompiler, DiamondWeConfig, default_preimage_max_coefficient_bound,
    };
    use mxx_gadgets::circuit::BooleanCircuitShape;
    use mxx_ir_core::RealExpr;
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
    use num_bigint::BigUint;

    fn selected(crt_depth: usize, achieved_security_bits: u64) -> DiamondSelectedParameters {
        let parameters = DCRTPolyParams::new(8, crt_depth, 20, 4);
        let modulus = parameters.modulus();
        let trapdoor_sigma = RealExpr::from_f64_exact(4.578).unwrap();
        let gadget_base = 16.into();
        let preimage_max_coefficient_bound = default_preimage_max_coefficient_bound(
            &trapdoor_sigma,
            parameters.ring_dimension() as usize,
            parameters.modulus_digits(),
            &gadget_base,
        )
        .unwrap();
        let compiler = DiamondWeCompiler::new(
            DiamondWeConfig {
                modulus: modulus.as_ref().clone().into(),
                ring_dimension: parameters.ring_dimension() as usize,
                input_count: 1,
                digit_base: 2,
                batch_bits: 1,
                gadget_base,
                digit_count: parameters.modulus_digits(),
                trapdoor_sigma,
                error_sigma: RealExpr::from_integer(0),
                error_max_coefficient_bound: 0.into(),
                preimage_max_coefficient_bound,
                bgg_tag: b"deployment-identity-test".to_vec(),
            },
            BooleanCircuitShape {
                instance_width: 1,
                witness_width: 1,
                depth: 1,
                max_layer_width: 2,
            },
        )
        .unwrap();
        DiamondSelectedParameters {
            parameters,
            compiler,
            crt_depth,
            log_ring_dimension: 3,
            ring_dimension: 8,
            modulus: modulus.as_ref().clone(),
            modulus_bits: modulus.bits() as usize,
            achieved_security_bits,
            noise_bound: 1u8.into(),
        }
    }

    fn input(nonce_ignored: u8) -> DiamondCacheKeyInput {
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
        let bound = derive_output_noise_bound_from_parameters(&parameters).unwrap();
        DiamondCacheKeyInput {
            ir_version: IR_VERSION,
            lean_source_schema_version: LEAN_SOURCE_SCHEMA_VERSION,
            linked_program_sha256: [nonce_ignored; 32],
            claim_template_sha256: [2; 32],
            bound_expression_sha256: [3; 32],
            claim_instance_sha256: [4; 32],
            proof_source_sha256: vec![[5; 32]],
            source_manifest_sha256: source_manifest_sha256_for_files(
                &LeanFileManifest::new([]).unwrap(),
                Path::new("Check.lean"),
                b"check",
            )
            .unwrap(),
            lean_toolchain: "leanprover/lean4:v4.19.0".to_owned(),
            mathlib_revision: "mathlib-rev".to_owned(),
            parameters,
            bound,
            theorem: "Mxx.We.Correctness.claim".to_owned(),
            check_file: PathBuf::from("Check.lean"),
            check_file_sha256: sha256_bytes(b"check"),
        }
    }

    #[test]
    fn key_is_deterministic() {
        assert_eq!(canonical_cache_key(input(0)).unwrap(), canonical_cache_key(input(0)).unwrap());
    }

    #[test]
    fn semantic_identity_is_cache_visible() {
        assert_ne!(
            canonical_cache_key(input(0)).unwrap().sha256,
            canonical_cache_key(input(1)).unwrap().sha256
        );
    }

    #[test]
    fn deployment_identity_binds_runtime_representation_and_security() {
        let key = canonical_cache_key(input(0)).unwrap();
        let identity =
            DiamondDeploymentIdentity::from_selected(key.clone(), &selected(2, 128)).unwrap();
        assert!(identity.validate().is_ok());
        let changed_identity =
            DiamondDeploymentIdentity::from_selected(key.clone(), &selected(1, 128)).unwrap();
        assert_ne!(identity.sha256(), changed_identity.sha256());
        assert_ne!(
            identity.sha256(),
            DiamondDeploymentIdentity::from_selected(key, &selected(2, 129)).unwrap().sha256()
        );
    }

    #[test]
    fn stale_semantic_cache_key_is_rejected() {
        let mut key = canonical_cache_key(input(0)).unwrap();
        key.sha256[0] ^= 1;
        assert!(matches!(
            DiamondDeploymentIdentity::from_selected(key, &selected(1, 1)),
            Err(CacheError::InvalidRecord)
        ));
    }

    #[test]
    fn inconsistent_bound_data_is_rejected() {
        let mut candidate = input(0);
        candidate.bound.value = 4u8.into();
        assert!(matches!(canonical_cache_key(candidate), Err(CacheError::Key(message)) if
            message.contains("bound value")));
    }

    #[test]
    fn cache_package_contains_lakefile_and_manifest() {
        let target = tempfile::tempdir().unwrap();
        let lean_source = tempfile::tempdir().unwrap();
        fs::write(lean_source.path().join("lakefile.toml"), "name = \"MxxWe\"\n").unwrap();
        fs::write(lean_source.path().join("lean-toolchain"), "leanprover/lean4:v4.19.0\n").unwrap();
        fs::write(lean_source.path().join("MxxWe.lean"), "def imported := True\n").unwrap();
        let files = LeanFileManifest::new([super::super::emit::GeneratedLeanFile {
            path: PathBuf::from("MxxGenerated/Program.lean"),
            contents: b"def generated := True\n".to_vec(),
        }])
        .unwrap();
        let mut key_input = input(0);
        key_input.check_file = PathBuf::from("MxxGenerated/Program.lean");
        key_input.check_file_sha256 = sha256_bytes(b"def generated := True\n");
        key_input.claim_instance_sha256 = claim_instance_sha256(b"parameters", b"claim");
        key_input.source_manifest_sha256 = source_manifest_sha256_for_package(
            lean_source.path(),
            &files,
            &key_input.check_file,
            b"def generated := True\n",
        )
        .unwrap();
        let key = canonical_cache_key(key_input).unwrap();
        let directory = materialize_cache_package(
            target.path(),
            &key,
            lean_source.path(),
            &files,
            &key.input.theorem,
            Path::new("MxxGenerated/Program.lean"),
            b"def generated := True\n",
        )
        .unwrap();
        assert!(directory.join("lakefile.toml").is_file());
        assert!(directory.join(CACHE_MANIFEST_FILE).is_file());
        assert!(directory.join("MxxGenerated/Program.lean").is_file());
        let published = read_cache_record(target.path(), &key).unwrap().unwrap();
        assert_eq!(published.artifact_directory, directory);
        assert!(published.source_manifest.validate().is_ok());
        assert!(
            read_cache_record_checked(target.path(), &key, b"parameters", b"claim")
                .unwrap()
                .is_some()
        );
        fs::create_dir_all(lean_source.path().join(".lake/build")).unwrap();
        fs::write(lean_source.path().join(".lake/build/stale.olean"), b"stale").unwrap();
        assert!(read_cache_record(target.path(), &key).unwrap().is_some());
        let lakefile = fs::read_to_string(directory.join("lakefile.toml")).unwrap();
        assert!(lakefile.contains("name = \"MxxWe\""));
        assert_eq!(
            fs::read_to_string(directory.join("lean-toolchain")).unwrap(),
            "leanprover/lean4:v4.19.0\n"
        );
        assert!(matches!(
            materialize_cache_package(
                target.path(),
                &key,
                lean_source.path(),
                &files,
                &key.input.theorem,
                Path::new("MxxGenerated/Program.lean"),
                b"def generated := True\n",
            ),
            Err(CacheError::AlreadyPublished)
        ));
        fs::write(lean_source.path().join("MxxWe.lean"), "def imported := False\n").unwrap();
        assert!(read_cache_record(target.path(), &key).unwrap().is_some());
        fs::write(lean_source.path().join("MxxWe.lean"), "def imported := True\n").unwrap();
        fs::write(directory.join("deps/0/MxxWe.lean"), "def imported := False\n").unwrap();
        assert!(matches!(read_cache_record(target.path(), &key), Err(CacheError::InvalidRecord)));
        fs::write(directory.join("deps/0/MxxWe.lean"), "def imported := True\n").unwrap();
        fs::write(directory.join("lakefile.toml"), "mutated = true\n").unwrap();
        assert!(matches!(read_cache_record(target.path(), &key), Err(CacheError::InvalidRecord)));
        fs::write(directory.join("lakefile.toml"), lakefile).unwrap();
        fs::write(directory.join("lean-toolchain"), "leanprover/lean4:v4.19.1\n").unwrap();
        assert!(matches!(read_cache_record(target.path(), &key), Err(CacheError::InvalidRecord)));
    }

    #[test]
    fn concurrent_materialization_publishes_exactly_once() {
        use std::{
            sync::{Arc, Barrier},
            thread,
        };

        let target = tempfile::tempdir().unwrap();
        let lean_source = tempfile::tempdir().unwrap();
        fs::write(lean_source.path().join("lakefile.toml"), "name = \"MxxWe\"\n").unwrap();
        fs::write(lean_source.path().join("MxxWe.lean"), "def imported := True\n").unwrap();
        let files = LeanFileManifest::new([]).unwrap();
        let check_file = Path::new("Check.lean");
        let check_contents = b"check";
        let parameters_source = b"parameters";
        let claim_source = b"claim";
        let mut key_input = input(0);
        key_input.claim_instance_sha256 = claim_instance_sha256(parameters_source, claim_source);
        key_input.source_manifest_sha256 = source_manifest_sha256_for_package(
            lean_source.path(),
            &files,
            check_file,
            check_contents,
        )
        .unwrap();
        let key = canonical_cache_key(key_input).unwrap();
        let worker_count = 8;
        let barrier = Arc::new(Barrier::new(worker_count));
        let results = thread::scope(|scope| {
            let handles = (0..worker_count)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    let target = target.path();
                    let lean_source = lean_source.path();
                    let key = &key;
                    let files = &files;
                    scope.spawn(move || {
                        barrier.wait();
                        materialize_cache_package(
                            target,
                            key,
                            lean_source,
                            files,
                            &key.input.theorem,
                            check_file,
                            check_contents,
                        )
                    })
                })
                .collect::<Vec<_>>();
            handles.into_iter().map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
        });

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(CacheError::AlreadyPublished)))
                .count(),
            worker_count - 1
        );
        assert!(
            read_cache_record_checked(target.path(), &key, parameters_source, claim_source)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn cache_record_mismatch_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let key = canonical_cache_key(input(0)).unwrap();
        write_cache_record(
            directory.path(),
            &key,
            &key.input.theorem,
            Path::new("Check.lean"),
            b"check",
        )
        .unwrap();
        assert!(read_cache_record(directory.path(), &key).unwrap().is_some());

        let path = cache_directory(directory.path(), &key).join(CACHE_MANIFEST_FILE);
        let mut json: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        json["theorem"] = serde_json::Value::String("other".to_owned());
        fs::write(path, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(matches!(
            read_cache_record(directory.path(), &key),
            Err(CacheError::InvalidRecord)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn source_manifest_rejects_symlinked_inputs_and_roots() {
        use std::os::unix::fs::symlink;
        let package = tempfile::tempdir().unwrap();
        fs::write(package.path().join("lakefile.toml"), "name = \"MxxWe\"\n").unwrap();
        fs::write(package.path().join("MxxWe.lean"), "def imported := True\n").unwrap();
        symlink(package.path().join("MxxWe.lean"), package.path().join("Alias.lean")).unwrap();
        symlink(package.path(), package.path().join("AliasDir")).unwrap();
        let files = LeanFileManifest::new([]).unwrap();
        assert!(matches!(
            source_manifest_sha256_for_package(
                package.path(),
                &files,
                Path::new("Check.lean"),
                b"check"
            ),
            Err(CacheError::InvalidRecord)
        ));

        let clean = tempfile::tempdir().unwrap();
        fs::write(clean.path().join("lakefile.toml"), "name = \"MxxWe\"\n").unwrap();
        let alias = clean.path().join("alias");
        symlink(package.path(), &alias).unwrap();
        assert!(matches!(
            source_manifest_sha256_for_package(&alias, &files, Path::new("Check.lean"), b"check"),
            Err(CacheError::InvalidRecord)
        ));
    }

    #[test]
    fn source_manifest_labels_multiple_import_roots_without_collision() {
        let package = tempfile::tempdir().unwrap();
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        for root in [&first, &second] {
            fs::write(root.path().join("lakefile.toml"), "name = \"Dep\"\n").unwrap();
            fs::write(root.path().join("Foo.lean"), root.path().to_string_lossy().as_bytes())
                .unwrap();
        }
        fs::write(
            package.path().join("lakefile.toml"),
            format!(
                "name = \"MxxWe\"\n[[require]]\nname = \"A\"\npath = \"{}\"\n[[require]]\nname = \"B\"\npath = \"{}\"\n",
                first.path().display(),
                second.path().display()
            ),
        )
        .unwrap();
        fs::write(package.path().join("Foo.lean"), b"package").unwrap();
        let manifest = source_manifest(package.path(), &[]).unwrap();
        let labels = manifest.entries.iter().map(|entry| entry.label.clone()).collect::<Vec<_>>();
        assert!(labels.contains(&PathBuf::from("imported/0/Foo.lean")), "{labels:?}");
        assert!(labels.contains(&PathBuf::from("imported/1/Foo.lean")), "{labels:?}");
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn real_we_source_package_materializes_without_generated_lake_state() {
        let target = tempfile::tempdir().unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("lean");
        let files = LeanFileManifest::new([]).unwrap();
        let check_file = Path::new("Check.lean");
        let check = b"import MxxWe\n";
        let mut key_input = input(0);
        key_input.check_file = check_file.to_path_buf();
        key_input.check_file_sha256 = sha256_bytes(check);
        key_input.source_manifest_sha256 =
            source_manifest_sha256_for_package(&source, &files, check_file, check).unwrap();
        let key = canonical_cache_key(key_input).unwrap();
        let directory = materialize_cache_package(
            target.path(),
            &key,
            &source,
            &files,
            &key.input.theorem,
            check_file,
            check,
        )
        .unwrap();
        assert!(directory.join("Check.lean").is_file());
        for dependency in fs::read_dir(directory.join("deps")).unwrap() {
            assert!(!dependency.unwrap().path().join(".lake").exists());
        }
    }

    #[test]
    fn materialize_production_run_ignores_lake_state_and_revalidates_sources() {
        let target = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let dep = tempfile::tempdir().unwrap();
        fs::write(
            dep.path().join("lakefile.toml"),
            "name = \"Dep\"\n[[lean_lib]]\nname = \"Dep\"\n",
        )
        .unwrap();
        fs::create_dir_all(dep.path().join(".lake")).unwrap();
        fs::write(dep.path().join(".lake/lake-manifest.json"), b"{\"name\":\"Dep\"}\n").unwrap();
        fs::write(dep.path().join("Dep.lean"), "def dep := True\n").unwrap();
        fs::write(source.path().join("lakefile.toml"), format!("name = \"MxxWe\"\n[[require]]\nname = \"Dep\"\npath = \"{}\"\n[[lean_lib]]\nname = \"MxxWe\"\n", dep.path().display())).unwrap();
        fs::create_dir_all(source.path().join(".lake")).unwrap();
        fs::write(source.path().join(".lake/lake-manifest.json"), b"{\"name\":\"MxxWe\"}\n")
            .unwrap();
        fs::write(source.path().join("MxxWe.lean"), "def imported := True\n").unwrap();
        let files = LeanFileManifest::new([super::super::emit::GeneratedLeanFile {
            path: PathBuf::from("MxxGenerated.lean"),
            contents: b"def generated := True\n".to_vec(),
        }])
        .unwrap();
        let check = b"namespace Mxx.We.DiamondWE\ndef CorrectnessClaim (_ : Nat) : Prop := True\nend Mxx.We.DiamondWE\nnamespace Mxx.We.Golden.DiamondWE\ndef candidate : Nat := 0\ntheorem correct : Mxx.We.DiamondWE.CorrectnessClaim candidate := True.intro\nend Mxx.We.Golden.DiamondWE\n";
        let mut key_input = input(0);
        key_input.theorem = super::super::runner::DIAMOND_PROOF_THEOREM.to_owned();
        key_input.check_file = PathBuf::from("Check.lean");
        key_input.check_file_sha256 = sha256_bytes(check);
        key_input.claim_instance_sha256 = claim_instance_sha256(b"parameters", b"claim");
        key_input.source_manifest_sha256 =
            source_manifest_sha256_for_package(source.path(), &files, &key_input.check_file, check)
                .unwrap();
        let key = canonical_cache_key(key_input).unwrap();
        let directory = materialize_cache_package(
            target.path(),
            &key,
            source.path(),
            &files,
            &key.input.theorem,
            &key.input.check_file,
            check,
        )
        .unwrap();
        let request = super::super::runner::LeanRunRequest {
            package_directory: directory.clone(),
            check_file: key.input.check_file.clone(),
            theorem: key.input.theorem.clone(),
            cache_key: key.sha256,
            check_file_sha256: key.input.check_file_sha256,
        };
        super::super::runner::LeanRunner::default().run(&request).unwrap();
        assert!(
            read_cache_record_checked(target.path(), &key, b"parameters", b"claim")
                .unwrap()
                .is_some()
        );
        assert!(
            read_cache_record_checked(target.path(), &key, b"parameters", b"claim")
                .unwrap()
                .is_some()
        );
        let copied_dependencies = fs::read_dir(directory.join("deps"))
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        assert_eq!(copied_dependencies.len(), 2);
        for dependency in &copied_dependencies {
            assert!(!dependency.join(".lake").exists());
            fs::create_dir(dependency.join(".lake")).unwrap();
            fs::write(dependency.join(".lake/lake-manifest.json"), b"generated\n").unwrap();
        }
        assert!(
            read_cache_record_checked(target.path(), &key, b"parameters", b"claim")
                .unwrap()
                .is_some()
        );
        let dependency_source = copied_dependencies
            .iter()
            .map(|dependency| dependency.join("Dep.lean"))
            .find(|path| path.is_file())
            .unwrap();
        fs::write(dependency_source, b"mutated\n").unwrap();
        assert!(matches!(
            read_cache_record_checked(target.path(), &key, b"parameters", b"claim"),
            Err(CacheError::InvalidRecord)
        ));
    }
}
