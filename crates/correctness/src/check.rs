use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use thiserror::Error;

use crate::FreshnessMetadata;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TheoremReport {
    pub axioms: Vec<String>,
    pub freshness: FreshnessMetadata,
    pub native_decide_uses: Vec<NativeDecideUse>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeDecideUse {
    pub source_path: String,
    pub line: usize,
    pub declaration: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeDecideAllowance<'a> {
    pub source_path: &'a str,
    pub declaration: &'a str,
}

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("verification I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("Lean verification failed: {0}")]
    Lean(String),
    #[error("the theorem depends on a forbidden axiom: {0}")]
    ForbiddenAxiom(String),
    #[error("Lean source contains a forbidden axiom declaration: {0}")]
    ForbiddenDeclaration(PathBuf),
    #[error("Lean source contains the proof hole `{term}` at {path}:{line}")]
    ForbiddenProofHole { path: PathBuf, line: usize, term: &'static str },
    #[error("Lean source contains native_decide outside the checker-evaluation allowlist: {0:?}")]
    ForbiddenNativeDecide(NativeDecideUse),
}

pub fn verify_theorem_at(
    protocol: &str,
    freshness: &FreshnessMetadata,
    proof_module: &str,
    theorem_name: &str,
    module_root: &str,
    source_directories: &[&str],
    native_decide_allowlist: &[NativeDecideAllowance<'_>],
) -> Result<TheoremReport, VerifyError> {
    let root = workspace_root();
    let lean_name = lean_identifier(protocol);
    for directory in source_directories {
        reject_unreviewed_constructs(&root.join(directory))?;
    }
    let scratch = ScratchDir::new("verify")?;
    let probe = scratch.path.join("Probe.lean");
    fs::write(
        &probe,
        verification_probe(proof_module, theorem_name, module_root, &lean_name, freshness),
    )?;
    let output = Command::new("lake")
        .args(["env", "lean"])
        .arg(&probe)
        .current_dir(root.join("lean"))
        .output()?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        return Err(VerifyError::Lean(combined));
    }
    let axioms = parse_axioms(&combined)?;
    let native_decide_uses =
        source_directories.iter().try_fold(Vec::new(), |mut uses, directory| {
            collect_native_decide_uses(&root.join(directory), &root, &mut uses)?;
            Ok::<_, std::io::Error>(uses)
        })?;
    for usage in &native_decide_uses {
        if !native_decide_allowlist.iter().any(|allowed| {
            usage.source_path == allowed.source_path &&
                usage.declaration.as_deref() == Some(allowed.declaration)
        }) {
            return Err(VerifyError::ForbiddenNativeDecide(usage.clone()));
        }
    }
    Ok(TheoremReport { axioms, freshness: freshness.clone(), native_decide_uses })
}

fn verification_probe(
    proof_module: &str,
    theorem_name: &str,
    module_root: &str,
    lean_name: &str,
    freshness: &FreshnessMetadata,
) -> String {
    let source_paths = freshness
        .protocol_source_paths
        .iter()
        .map(|path| format!("\"{}\"", path.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "import {proof_module}\nopen {module_root}.Generated.{lean_name}\nexample : {lean_name}_generatorVersion = \"{}\" := rfl\nexample : {lean_name}_protocolSourcePaths = [{source_paths}] := rfl\nexample : {lean_name}_protocolSourceHash = \"{}\" := rfl\nexample : {lean_name}_workflowHash = \"{}\" := rfl\nexample : {lean_name}_toolkitHash = \"{}\" := rfl\n#check {theorem_name}\n#print axioms {theorem_name}\n",
        freshness.generator_version,
        freshness.protocol_source_hash,
        freshness.workflow_hash,
        freshness.toolkit_hash,
    )
}

fn parse_axioms(output: &str) -> Result<Vec<String>, VerifyError> {
    if output.contains("does not depend on any axioms") {
        return Ok(Vec::new());
    }
    let marker = "depends on axioms:";
    let Some(start) = output.find(marker) else {
        return Err(VerifyError::Lean(format!(
            "Lean did not print an axiom dependency list:\n{output}"
        )));
    };
    let after = &output[start + marker.len()..];
    let open = after
        .find('[')
        .ok_or_else(|| VerifyError::Lean(format!("malformed axiom output:\n{output}")))?;
    let close = after[open + 1..]
        .find(']')
        .map(|offset| open + 1 + offset)
        .ok_or_else(|| VerifyError::Lean(format!("malformed axiom output:\n{output}")))?;
    let allowed = ["propext", "Classical.choice", "Quot.sound"];
    let mut axioms = Vec::new();
    for axiom in after[open + 1..close].split(',').map(str::trim).filter(|value| !value.is_empty())
    {
        if !allowed.contains(&axiom) {
            return Err(VerifyError::ForbiddenAxiom(axiom.to_owned()));
        }
        axioms.push(axiom.to_owned());
    }
    Ok(axioms)
}

fn reject_unreviewed_constructs(directory: &Path) -> Result<(), VerifyError> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            reject_unreviewed_constructs(&path)?;
        } else if path.extension().is_some_and(|extension| extension == "lean") {
            let source = fs::read_to_string(&path)?;
            let mut comment_depth = 0usize;
            for (line_index, line) in source.lines().enumerate() {
                let code = lean_code_without_comments(line, &mut comment_depth);
                let tokens = code
                    .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .collect::<Vec<_>>();
                if tokens.contains(&"axiom") {
                    return Err(VerifyError::ForbiddenDeclaration(path));
                }
                for term in ["sorry", "admit"] {
                    if tokens.contains(&term) {
                        return Err(VerifyError::ForbiddenProofHole {
                            path,
                            line: line_index + 1,
                            term,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

fn collect_native_decide_uses(
    directory: &Path,
    workspace_root: &Path,
    output: &mut Vec<NativeDecideUse>,
) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_native_decide_uses(&path, workspace_root, output)?;
        } else if path.extension().is_some_and(|extension| extension == "lean") {
            let source = fs::read_to_string(&path)?;
            let source_path =
                path.strip_prefix(workspace_root).unwrap_or(&path).to_string_lossy().into_owned();
            let mut comment_depth = 0usize;
            let mut declaration = None;
            for (line_index, line) in source.lines().enumerate() {
                let code = lean_code_without_comments(line, &mut comment_depth);
                if let Some(name) = declaration_name(&code) {
                    declaration = Some(name);
                }
                if code
                    .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .any(|token| token == "native_decide")
                {
                    output.push(NativeDecideUse {
                        source_path: source_path.clone(),
                        line: line_index + 1,
                        declaration: declaration.clone(),
                    });
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

fn declaration_name(code: &str) -> Option<String> {
    let trimmed = code.trim_start();
    let trimmed = ["private ", "protected ", "noncomputable "]
        .iter()
        .find_map(|prefix| trimmed.strip_prefix(prefix))
        .unwrap_or(trimmed);
    let rest =
        ["theorem ", "lemma ", "def "].iter().find_map(|prefix| trimmed.strip_prefix(prefix))?;
    let name = rest
        .split(|character: char| character.is_whitespace() || matches!(character, ':' | '(' | '{'))
        .next()?;
    (!name.is_empty()).then(|| name.to_owned())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("correctness crate is inside workspace/crates")
        .to_path_buf()
}

struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    fn new(purpose: &str) -> Result<Self, std::io::Error> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time is after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir()
            .join(format!("mxx-correctness-{purpose}-{}-{nonce}", std::process::id()));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn lean_identifier(value: &str) -> String {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars.next().unwrap().to_ascii_uppercase().to_string() + chars.as_str()
        })
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axiom_parser_accepts_only_the_core_allowlist() {
        let output = "'proof' depends on axioms: [propext, Classical.choice, Quot.sound]";
        assert_eq!(
            parse_axioms(output).unwrap(),
            vec!["propext", "Classical.choice", "Quot.sound"]
        );
        assert!(matches!(
            parse_axioms("'proof' depends on axioms: [sorryAx]"),
            Err(VerifyError::ForbiddenAxiom(axiom)) if axiom == "sorryAx"
        ));
        assert_eq!(
            parse_axioms("'proof' does not depend on any axioms").unwrap(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn probe_checks_the_lean_owned_theorem_without_reconstructing_its_type() {
        let freshness = FreshnessMetadata {
            generator_version: "generator".to_owned(),
            protocol_source_paths: vec!["crates/example/src".to_owned()],
            protocol_source_hash: "source".to_owned(),
            workflow_hash: "workflow".to_owned(),
            toolkit_hash: "toolkit".to_owned(),
        };
        let probe = verification_probe(
            "MxxWe.Proofs.DiamondWe",
            "MxxWe.Proofs.DiamondWe.correct",
            "MxxWe",
            "DiamondWeFamily",
            &freshness,
        );
        assert!(probe.contains("#check MxxWe.Proofs.DiamondWe.correct"));
        assert!(probe.contains("#print axioms MxxWe.Proofs.DiamondWe.correct"));
        assert!(!probe.contains("CorrectStatement"));
        assert!(!probe.contains("Checker :="));
    }

    #[test]
    fn native_decide_scanner_ignores_comments_and_records_the_declaration() {
        let mut depth = 0;
        assert_eq!(lean_code_without_comments("/-- native_decide -/", &mut depth), "");
        assert_eq!(depth, 0);
        let code = lean_code_without_comments(
            "theorem checker_accepts : true := by native_decide -- native_decide",
            &mut depth,
        );
        assert_eq!(declaration_name(&code).as_deref(), Some("checker_accepts"));
        assert_eq!(
            code.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                .filter(|token| *token == "native_decide")
                .count(),
            1
        );
        assert_eq!(
            declaration_name("private theorem closedCheckerFact : True := by trivial").as_deref(),
            Some("closedCheckerFact")
        );
    }

    #[test]
    fn source_gate_ignores_comments_and_rejects_proof_holes() {
        let scratch = ScratchDir::new("source-gate").unwrap();
        let source = scratch.path.join("Proof.lean");
        fs::write(&source, "/-- sorry axiom -/\ntheorem complete : True := by trivial\n").unwrap();
        reject_unreviewed_constructs(&scratch.path).unwrap();

        fs::write(&source, "private theorem incomplete : True := by sorry\n").unwrap();
        assert!(matches!(
            reject_unreviewed_constructs(&scratch.path),
            Err(VerifyError::ForbiddenProofHole { term: "sorry", .. })
        ));
    }
}
