//! Build-integrity metadata for generated Lean protocol modules.
//!
//! These hashes prevent stale generated files from being paired with a newer
//! workflow or verifier. They are not premises of the correctness theorem.

use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;

pub const GENERATOR_VERSION: &str = "mxx-correctness-emitter-v8";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FreshnessMetadata {
    pub generator_version: String,
    pub protocol_source_paths: Vec<String>,
    pub protocol_source_hash: String,
    pub workflow_hash: String,
    pub toolkit_hash: String,
}

#[derive(Debug, Error)]
pub enum FreshnessError {
    #[error("failed to read correctness toolkit source: {0}")]
    Io(#[from] std::io::Error),
    #[error("correctness toolkit path is outside the workspace: {0}")]
    OutsideWorkspace(PathBuf),
    #[error("generated correctness module is stale: {field}")]
    Mismatch { field: &'static str },
}

/// Hashes every generic Lean source below `lean/Mxx` in canonical path order.
pub fn toolkit_hash(workspace_root: &Path) -> Result<String, FreshnessError> {
    let toolkit_root = workspace_root.join("lean/Mxx");
    let mut sources = Vec::new();
    collect_lean_sources(&toolkit_root, &mut sources)?;
    sources.sort();

    let mut digest = Sha256::new();
    for source in sources {
        let relative = source
            .strip_prefix(workspace_root)
            .map_err(|_| FreshnessError::OutsideWorkspace(source.clone()))?;
        let relative = relative.to_string_lossy();
        let bytes = fs::read(&source)?;
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Hashes the complete owner-declared protocol source set in canonical workspace-relative order.
///
/// Directory entries are traversed recursively. Both each relative path and its bytes contribute
/// to the digest, so adding, removing, renaming, or changing a source invalidates generated Lean.
pub fn protocol_source_hash(
    workspace_root: &Path,
    relative_paths: &[&str],
) -> Result<String, FreshnessError> {
    let mut sources = Vec::new();
    for relative in relative_paths {
        let path = workspace_root.join(relative);
        if path.is_dir() {
            collect_files(&path, &mut sources)?;
        } else {
            sources.push(path);
        }
    }
    sources.sort();
    sources.dedup();
    hash_sources(workspace_root, sources)
}

pub fn verify_freshness(
    generated: &FreshnessMetadata,
    expected: &FreshnessMetadata,
) -> Result<(), FreshnessError> {
    if generated.generator_version != expected.generator_version {
        return Err(FreshnessError::Mismatch { field: "generatorVersion" });
    }
    if generated.protocol_source_paths != expected.protocol_source_paths {
        return Err(FreshnessError::Mismatch { field: "protocolSourcePaths" });
    }
    if generated.protocol_source_hash != expected.protocol_source_hash {
        return Err(FreshnessError::Mismatch { field: "protocolSourceHash" });
    }
    if generated.workflow_hash != expected.workflow_hash {
        return Err(FreshnessError::Mismatch { field: "workflowHash" });
    }
    if generated.toolkit_hash != expected.toolkit_hash {
        return Err(FreshnessError::Mismatch { field: "toolkitHash" });
    }
    Ok(())
}

fn collect_lean_sources(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_lean_sources(&path, output)?;
        } else if path.extension().is_some_and(|extension| extension == "lean") {
            output.push(path);
        }
    }
    Ok(())
}

fn collect_files(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, output)?;
        } else if path.is_file() {
            output.push(path);
        }
    }
    Ok(())
}

fn hash_sources(workspace_root: &Path, sources: Vec<PathBuf>) -> Result<String, FreshnessError> {
    let mut digest = Sha256::new();
    for source in sources {
        let relative = source
            .strip_prefix(workspace_root)
            .map_err(|_| FreshnessError::OutsideWorkspace(source.clone()))?;
        let relative = relative.to_string_lossy();
        let bytes = fs::read(&source)?;
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatch_names_the_stale_component() {
        let generated = FreshnessMetadata {
            generator_version: GENERATOR_VERSION.to_owned(),
            protocol_source_paths: vec!["crates/example/src".to_owned()],
            protocol_source_hash: "source".to_owned(),
            workflow_hash: "old".to_owned(),
            toolkit_hash: "toolkit".to_owned(),
        };
        let expected = FreshnessMetadata { workflow_hash: "new".to_owned(), ..generated.clone() };
        assert!(matches!(
            verify_freshness(&generated, &expected),
            Err(FreshnessError::Mismatch { field: "workflowHash" })
        ));
    }
}
