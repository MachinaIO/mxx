use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TheoremReport {
    pub axioms: Vec<String>,
    pub protocol_hash: String,
    pub uses_native_decide: bool,
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
}

pub fn verify_theorem_at(
    protocol: &str,
    protocol_hash: &str,
    proof_module: &str,
    module_root: &str,
    source_directories: &[&str],
) -> Result<TheoremReport, VerifyError> {
    let root = workspace_root();
    let lean_name = lean_identifier(protocol);
    let lower = lower_identifier(protocol);
    for directory in source_directories {
        reject_axiom_declarations(&root.join(directory))?;
    }
    let scratch = ScratchDir::new("verify")?;
    let probe = scratch.path.join("Probe.lean");
    fs::write(
        &probe,
        format!(
            "import {proof_module}\nopen {module_root}.Generated.{lean_name}\nexample : {lean_name}CorrectStatement {lower}Checker := {lower}_correct\nexample : {lean_name}_protocolHash = \"{protocol_hash}\" := rfl\n#print axioms {lower}_correct\n"
        ),
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
    Ok(TheoremReport {
        axioms,
        protocol_hash: protocol_hash.to_owned(),
        uses_native_decide: source_directories.iter().try_fold(false, |found, directory| {
            if found {
                Ok(true)
            } else {
                source_tree_contains(&root.join(directory), "native_decide")
            }
        })?,
    })
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

fn reject_axiom_declarations(directory: &Path) -> Result<(), VerifyError> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            reject_axiom_declarations(&path)?;
        } else if path.extension().is_some_and(|extension| extension == "lean") &&
            fs::read_to_string(&path)?
                .lines()
                .any(|line| line.trim_start().starts_with("axiom "))
        {
            return Err(VerifyError::ForbiddenDeclaration(path));
        }
    }
    Ok(())
}

fn source_tree_contains(directory: &Path, needle: &str) -> Result<bool, std::io::Error> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            if source_tree_contains(&path, needle)? {
                return Ok(true);
            }
        } else if path.extension().is_some_and(|extension| extension == "lean") &&
            fs::read_to_string(path)?.contains(needle)
        {
            return Ok(true);
        }
    }
    Ok(false)
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

fn lower_identifier(value: &str) -> String {
    let value = lean_identifier(value);
    let mut chars = value.chars();
    chars.next().unwrap_or('p').to_ascii_lowercase().to_string() + chars.as_str()
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
}
