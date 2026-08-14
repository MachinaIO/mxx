use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub const GENERATOR_VERSION: &str = "mxx-correctness-emitter-v11";

pub struct GeneratedFreshness {
    pub workflow_hash: String,
    pub derivation_hash: String,
}

pub fn emit_rerun_paths(workspace: &Path, source_paths: &[&str], owner_lean: &Path) {
    for source in source_paths {
        println!("cargo:rerun-if-changed={}", workspace.join(source).display());
    }
    println!("cargo:rerun-if-changed={}", workspace.join("lean/Mxx").display());
    println!("cargo:rerun-if-changed={}", owner_lean.display());
}

#[allow(dead_code)]
pub fn verify_generated_freshness(
    workspace: &Path,
    generated: &Path,
    lean_name: &str,
    source_paths: &[&str],
) -> Result<GeneratedFreshness, String> {
    verify_generated_freshness_with_protocol_source_hash(
        workspace,
        generated,
        lean_name,
        source_paths,
        true,
    )
}

pub fn verify_generated_freshness_without_protocol_source_hash(
    workspace: &Path,
    generated: &Path,
    lean_name: &str,
    source_paths: &[&str],
) -> Result<GeneratedFreshness, String> {
    verify_generated_freshness_with_protocol_source_hash(
        workspace,
        generated,
        lean_name,
        source_paths,
        false,
    )
}

fn verify_generated_freshness_with_protocol_source_hash(
    workspace: &Path,
    generated: &Path,
    lean_name: &str,
    source_paths: &[&str],
    verify_protocol_source_hash: bool,
) -> Result<GeneratedFreshness, String> {
    let generated_source = fs::read_to_string(generated)
        .map_err(|error| format!("failed to read {}: {error}", generated.display()))?;
    let mut canonical_paths = source_paths.to_vec();
    canonical_paths.sort_unstable();
    canonical_paths.dedup();
    let expected_paths =
        canonical_paths.iter().map(|path| format!("\"{path}\"")).collect::<Vec<_>>().join(", ");
    require_definition(
        &generated_source,
        &format!("{lean_name}_generatorVersion"),
        &format!("String := \"{GENERATOR_VERSION}\""),
    )?;
    require_definition(
        &generated_source,
        &format!("{lean_name}_protocolSourcePaths"),
        &format!("List String := [{expected_paths}]"),
    )?;
    if verify_protocol_source_hash {
        require_definition(
            &generated_source,
            &format!("{lean_name}_protocolSourceHash"),
            &format!("String := \"{}\"", hash_protocol_sources(workspace, &canonical_paths)?),
        )?;
    }
    require_definition(
        &generated_source,
        &format!("{lean_name}_toolkitHash"),
        &format!("String := \"{}\"", hash_toolkit(workspace)?),
    )?;
    Ok(GeneratedFreshness {
        workflow_hash: string_definition(&generated_source, &format!("{lean_name}_workflowHash"))?,
        derivation_hash: string_definition(
            &generated_source,
            &format!("{lean_name}_derivationHash"),
        )?,
    })
}

pub fn lake_build(lean_root: &Path, targets: &[&str], description: &str) -> Result<(), String> {
    let status = Command::new("lake")
        .arg("build")
        .args(targets)
        .current_dir(lean_root)
        .status()
        .map_err(|error| format!("failed to start {description}: {error}"))?;
    status.success().then_some(()).ok_or_else(|| format!("{description} failed"))
}

#[allow(dead_code)]
pub fn verify_theorem_axioms(
    lean_root: &Path,
    out_dir: &Path,
    proof_module: &str,
    theorem_name: &str,
    allow_native_decide: bool,
) -> Result<String, String> {
    let probe = out_dir.join("CorrectnessAxiomProbe.lean");
    fs::write(
        &probe,
        format!("import {proof_module}\n#check {theorem_name}\n#print axioms {theorem_name}\n"),
    )
    .map_err(|error| format!("failed to write {}: {error}", probe.display()))?;
    let output = Command::new("lake")
        .args(["env", "lean"])
        .arg(&probe)
        .current_dir(lean_root)
        .output()
        .map_err(|error| format!("failed to start theorem verification: {error}"))?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        return Err(format!("theorem verification failed:\n{combined}"));
    }
    reject_forbidden_axioms(&combined, allow_native_decide)?;
    Ok(combined)
}

pub fn verify_no_proof_holes(workspace: &Path, source_directories: &[&Path]) -> Result<(), String> {
    for directory in source_directories {
        verify_no_proof_holes_in(directory, workspace)?;
    }
    Ok(())
}

fn require_definition(source: &str, name: &str, value: &str) -> Result<(), String> {
    let expected = format!("def {name} : {value}");
    source
        .contains(&expected)
        .then_some(())
        .ok_or_else(|| format!("generated correctness module is stale: expected `{expected}`"))
}

fn string_definition(source: &str, name: &str) -> Result<String, String> {
    let prefix = format!("def {name} : String := \"");
    let remainder = source
        .split_once(&prefix)
        .map(|(_, remainder)| remainder)
        .ok_or_else(|| format!("generated correctness module is missing `{name}`"))?;
    remainder
        .split_once('"')
        .map(|(value, _)| value.to_owned())
        .ok_or_else(|| format!("generated correctness module has malformed `{name}`"))
}

fn hash_protocol_sources(workspace: &Path, source_paths: &[&str]) -> Result<String, String> {
    let mut sources = Vec::new();
    for relative in source_paths {
        let path = workspace.join(relative);
        if path.is_dir() {
            collect_files(&path, None, &mut sources)?;
        } else if path.is_file() {
            sources.push(path);
        } else {
            return Err(format!("protocol source does not exist: {}", path.display()));
        }
    }
    hash_sources(workspace, sources)
}

fn verify_no_proof_holes_in(directory: &Path, workspace: &Path) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
    {
        let path = entry
            .map_err(|error| format!("failed to read {} entry: {error}", directory.display()))?
            .path();
        if path.is_dir() {
            verify_no_proof_holes_in(&path, workspace)?;
        } else if path.extension().is_some_and(|extension| extension == "lean") {
            let source = fs::read_to_string(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            let relative = path.strip_prefix(workspace).unwrap_or(&path);
            let mut comment_depth = 0usize;
            for (line_index, line) in source.lines().enumerate() {
                let code = lean_code_without_comments(line, &mut comment_depth);
                if code
                    .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .any(|token| token == "axiom")
                {
                    return Err(format!(
                        "Lean axiom declaration is forbidden at {}:{}",
                        relative.display(),
                        line_index + 1
                    ));
                }
                for forbidden in ["sorry", "admit"] {
                    if code
                        .split(|character: char| {
                            !character.is_ascii_alphanumeric() && character != '_'
                        })
                        .any(|token| token == forbidden)
                    {
                        return Err(format!(
                            "Lean proof hole `{forbidden}` is forbidden at {}:{}",
                            relative.display(),
                            line_index + 1
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn lean_code_without_comments(line: &str, comment_depth: &mut usize) -> String {
    let bytes = line.as_bytes();
    let mut output = String::new();
    let mut index = 0;
    while index < bytes.len() {
        if *comment_depth == 0 &&
            index + 1 < bytes.len() &&
            bytes[index] == b'-' &&
            bytes[index + 1] == b'-'
        {
            break;
        }
        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'-' {
            *comment_depth += 1;
            index += 2;
        } else if *comment_depth > 0 &&
            index + 1 < bytes.len() &&
            bytes[index] == b'-' &&
            bytes[index + 1] == b'/'
        {
            *comment_depth -= 1;
            index += 2;
        } else {
            if *comment_depth == 0 {
                output.push(bytes[index] as char);
            }
            index += 1;
        }
    }
    output
}

fn hash_toolkit(workspace: &Path) -> Result<String, String> {
    let mut sources = Vec::new();
    collect_files(&workspace.join("lean/Mxx"), Some("lean"), &mut sources)?;
    hash_sources(workspace, sources)
}

fn collect_files(
    directory: &Path,
    extension: Option<&str>,
    output: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
    {
        let path = entry
            .map_err(|error| format!("failed to read {} entry: {error}", directory.display()))?
            .path();
        if path.is_dir() {
            collect_files(&path, extension, output)?;
        } else if path.is_file() &&
            extension
                .is_none_or(|expected| path.extension().is_some_and(|actual| actual == expected))
        {
            output.push(path);
        }
    }
    Ok(())
}

fn hash_sources(workspace: &Path, mut sources: Vec<PathBuf>) -> Result<String, String> {
    sources.sort();
    sources.dedup();
    let mut digest = Sha256::new();
    for source in sources {
        let relative = source
            .strip_prefix(workspace)
            .map_err(|_| format!("source is outside workspace: {}", source.display()))?;
        let relative = relative.to_string_lossy();
        let bytes = fs::read(&source)
            .map_err(|error| format!("failed to read {}: {error}", source.display()))?;
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[allow(dead_code)]
fn reject_forbidden_axioms(output: &str, allow_native_decide: bool) -> Result<(), String> {
    for axiom in theorem_axioms(output)? {
        if !["propext", "Classical.choice", "Quot.sound"].contains(&axiom.as_str()) &&
            !(allow_native_decide && axiom.contains(".native_decide."))
        {
            return Err(format!("the correctness theorem depends on forbidden axiom `{axiom}`"));
        }
    }
    Ok(())
}

pub fn theorem_axioms(output: &str) -> Result<Vec<String>, String> {
    if output.contains("does not depend on any axioms") {
        return Ok(Vec::new());
    }
    let marker = "depends on axioms:";
    let after = output
        .split_once(marker)
        .map(|(_, after)| after)
        .ok_or_else(|| format!("Lean did not print an axiom dependency list:\n{output}"))?;
    let axioms = after
        .split_once('[')
        .and_then(|(_, after)| after.split_once(']'))
        .map(|(axioms, _)| axioms)
        .ok_or_else(|| format!("malformed axiom dependency list:\n{output}"))?;
    Ok(axioms
        .split(',')
        .map(str::trim)
        .filter(|axiom| !axiom.is_empty())
        .map(str::to_owned)
        .collect())
}
