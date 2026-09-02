//! Small fail-closed Lake/Lean subprocess runner.

use super::{
    cache::{hex_bytes, sha256_bytes, validate_source_manifest_for_package},
    emit::validate_relative_path,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};
use thiserror::Error;

const MANIFEST_FILE: &str = "mxx-diamond-claim.json";
const WRAPPER_FILE: &str = "MxxDiamondCheck.lean";
pub(crate) const DIAMOND_PROOF_THEOREM: &str = "Mxx.We.Golden.DiamondWE.correct";
const DIAMOND_PROOF_TYPE: &str =
    "Mxx.We.DiamondWE.CorrectnessClaim Mxx.We.Golden.DiamondWE.candidate";
const ALLOWED_LEAN_AXIOMS: &[&str] = &["propext", "Classical.choice", "Quot.sound"];

#[derive(Clone, Debug)]
pub struct LeanRunner {
    lake_executable: PathBuf,
    timeout: Duration,
    production: bool,
}

impl Default for LeanRunner {
    fn default() -> Self {
        Self {
            lake_executable: PathBuf::from("lake"),
            timeout: Duration::from_secs(120),
            production: true,
        }
    }
}

impl LeanRunner {
    #[cfg(test)]
    pub(crate) fn for_test(lake_executable: PathBuf, timeout: Duration) -> Self {
        Self { lake_executable, timeout, production: false }
    }

    pub(crate) fn is_production(&self) -> bool {
        self.production
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanRunRequest {
    pub package_directory: PathBuf,
    pub check_file: PathBuf,
    pub theorem: String,
    pub cache_key: [u8; 32],
    pub check_file_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanRunOutput {
    pub status: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub elapsed: Duration,
}

#[derive(Debug, Error)]
pub enum LeanRunnerError {
    #[error("Lean claim manifest is missing from {0}")]
    MissingManifest(PathBuf),
    #[error("Lean claim manifest is invalid")]
    InvalidManifest,
    #[error("Lean theorem name or cache key does not match its manifest")]
    ManifestMismatch,
    #[error("Lean check file path is invalid")]
    InvalidCheckPath,
    #[error("could not start Lake/Lean: {0}")]
    ToolNotFound(#[source] io::Error),
    #[error("Lake/Lean timed out after {0:?}")]
    Timeout(Duration),
    #[error("Lake/Lean exited unsuccessfully ({status}): {stderr}")]
    Failed { status: i32, stderr: String },
    #[error("Lean theorem axiom audit found disallowed axioms: {0}")]
    DisallowedAxioms(String),
    #[error("Lean theorem axiom audit did not produce a recognized report for {0}")]
    InvalidAxiomReport(String),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct ClaimManifest {
    theorem: String,
    cache_key: String,
    check_file: PathBuf,
    check_file_sha256: [u8; 32],
}

impl LeanRunner {
    pub fn run(&self, request: &LeanRunRequest) -> Result<LeanRunOutput, LeanRunnerError> {
        relative_check_file(&request.check_file)?;
        validate_theorem_name(&request.theorem)?;
        if !request.package_directory.is_dir() {
            return Err(LeanRunnerError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                "Lean package directory does not exist",
            )));
        }
        if request.package_directory.join("manifest.json").is_file() {
            validate_source_manifest_for_package(&request.package_directory)
                .map_err(|_| LeanRunnerError::ManifestMismatch)?;
        }
        let manifest_path = request.package_directory.join(MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                LeanRunnerError::MissingManifest(manifest_path.clone())
            } else {
                LeanRunnerError::Io(error)
            }
        })?;
        let manifest: ClaimManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|_| LeanRunnerError::InvalidManifest)?;
        if manifest.theorem != request.theorem ||
            manifest.cache_key != hex_bytes(&request.cache_key) ||
            manifest.check_file != request.check_file
        {
            return Err(LeanRunnerError::ManifestMismatch);
        }
        let check_bytes = fs::read(request.package_directory.join(&request.check_file))
            .map_err(LeanRunnerError::Io)?;
        if sha256_bytes(&check_bytes) != manifest.check_file_sha256 ||
            manifest.check_file_sha256 != request.check_file_sha256
        {
            return Err(LeanRunnerError::ManifestMismatch);
        }

        let candidate_source =
            std::str::from_utf8(&check_bytes).map_err(|_| LeanRunnerError::InvalidManifest)?;
        let isolated = if self.production {
            let temp = tempfile::tempdir()?;
            copy_directory(&request.package_directory, temp.path())?;
            Some(temp)
        } else {
            None
        };
        let execution_directory =
            isolated.as_ref().map_or(request.package_directory.as_path(), tempfile::TempDir::path);
        let theorem_check = if request.theorem == DIAMOND_PROOF_THEOREM {
            format!("#check ({} : {})", request.theorem, DIAMOND_PROOF_TYPE)
        } else {
            // Generic runner tests and lower-level callers may check another declaration, but
            // the production Diamond verdict rejects such a request in the checker below.
            format!("#check {}", request.theorem)
        };
        let audit_delimiter = format!("MXX_AXIOM_AUDIT_{:032x}", rand::random::<u128>());
        let wrapper = format!(
            "{}\n{theorem_check}\n#eval IO.println \"{audit_delimiter}\"\n#print axioms {theorem}\ndef mxxDiamondExpectedCacheKey : String := \"{key}\"\nexample : mxxDiamondExpectedCacheKey = \"{key}\" := rfl\n",
            candidate_source,
            theorem_check = theorem_check,
            theorem = request.theorem,
            audit_delimiter = audit_delimiter,
            key = hex_bytes(&request.cache_key),
        );
        let wrapper_path = execution_directory.join(WRAPPER_FILE);
        fs::write(&wrapper_path, wrapper)?;

        // Build the cache-local package and copied path dependencies from source before checking
        // the wrapper.  This prevents Lake from consulting a live workspace `.lake` tree.
        if self.production {
            let mut build = Command::new(&self.lake_executable);
            build
                .current_dir(execution_directory)
                .args(["--no-cache", "--rehash", "build", "MxxGenerated"])
                .env_clear()
                .env("PATH", std::env::var_os("PATH").unwrap_or_default());
            let build = run_bounded_command(build, self.timeout)?;
            if build.status != 0 {
                return Err(LeanRunnerError::Failed {
                    status: build.status,
                    stderr: format!(
                        "{}{}",
                        String::from_utf8_lossy(&build.stderr),
                        String::from_utf8_lossy(&build.stdout)
                    ),
                });
            }
        }

        let mut command = Command::new(&self.lake_executable);
        command
            .current_dir(execution_directory)
            .arg("env")
            .arg("lean")
            .arg(WRAPPER_FILE)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = run_bounded_command(command, self.timeout)?;
        if output.status != 0 {
            return Err(LeanRunnerError::Failed {
                status: output.status,
                stderr: format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stderr),
                    String::from_utf8_lossy(&output.stdout)
                ),
            });
        }
        audit_axiom_report(&output.stdout, &request.theorem, &audit_delimiter)?;
        Ok(output)
    }
}

/// Validate the machine-readable report emitted by Lean's `#print axioms` command.
///
/// The report is intentionally parsed only after the theorem has typechecked. Missing or
/// malformed output is an infrastructure failure: accepting a successful `#check` without a
/// corresponding audit would make the soundness gate fail open. The three foundational Lean
/// axioms are the only axioms permitted by the repository proof policy.
fn audit_axiom_report(
    stdout: &[u8],
    theorem: &str,
    delimiter: &str,
) -> Result<(), LeanRunnerError> {
    let output = String::from_utf8_lossy(stdout);
    let mut offset = 0;
    let mut audit_start = None;
    for line in output.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        if content == delimiter {
            audit_start = Some(offset + line.len());
        }
        offset += line.len();
    }
    let Some(audit_start) = audit_start else {
        return Err(LeanRunnerError::InvalidAxiomReport(theorem.to_owned()));
    };
    let audit_output = &output[audit_start..];
    let Some(report) = audit_output.strip_suffix('\n') else {
        return Err(LeanRunnerError::InvalidAxiomReport(theorem.to_owned()));
    };
    if report.is_empty() || report.contains('\n') {
        return Err(LeanRunnerError::InvalidAxiomReport(theorem.to_owned()));
    }
    let prefix = format!("'{theorem}' ");
    if !report.starts_with(&prefix) {
        return Err(LeanRunnerError::InvalidAxiomReport(theorem.to_owned()));
    }
    if report == format!("'{theorem}' does not depend on any axioms") {
        return Ok(());
    }
    let Some(axioms) = report
        .strip_prefix(&format!("'{theorem}' depends on axioms: ["))
        .and_then(|value| value.strip_suffix(']'))
    else {
        return Err(LeanRunnerError::InvalidAxiomReport(theorem.to_owned()));
    };
    let mut disallowed = Vec::new();
    for axiom in axioms.split(',').map(str::trim) {
        if axiom.is_empty() || !ALLOWED_LEAN_AXIOMS.contains(&axiom) {
            disallowed.push(axiom.to_owned());
        }
    }
    if disallowed.is_empty() {
        Ok(())
    } else {
        Err(LeanRunnerError::DisallowedAxioms(disallowed.join(", ")))
    }
}

fn relative_check_file(path: &Path) -> Result<PathBuf, LeanRunnerError> {
    validate_relative_path(path).map_err(|_| LeanRunnerError::InvalidCheckPath)?;
    Ok(path.to_path_buf())
}

fn copy_directory(source: &Path, destination: &Path) -> Result<(), LeanRunnerError> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let target = destination.join(entry.file_name());
        let ty = entry.file_type()?;
        if ty.is_symlink() {
            return Err(LeanRunnerError::InvalidManifest);
        } else if ty.is_dir() {
            copy_directory(&path, &target)?;
        } else if ty.is_file() {
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

fn read_pipe<R: Read>(mut reader: R) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn run_bounded_command(
    mut command: Command,
    timeout: Duration,
) -> Result<LeanRunOutput, LeanRunnerError> {
    let started = Instant::now();
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        command.pre_exec(|| {
            if libc::setpgid(0, 0) == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
        });
    }
    let mut child = command.spawn().map_err(LeanRunnerError::ToolNotFound)?;
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let stdout_thread = thread::spawn(move || read_pipe(stdout));
    let stderr_thread = thread::spawn(move || read_pipe(stderr));
    let status = match wait_with_timeout(&mut child, timeout) {
        Ok(status) => status,
        Err(error) => {
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            return Err(error);
        }
    };
    let stdout =
        stdout_thread.join().unwrap_or_else(|_| Err(io::Error::other("stdout reader panicked")))?;
    let stderr =
        stderr_thread.join().unwrap_or_else(|_| Err(io::Error::other("stderr reader panicked")))?;
    Ok(LeanRunOutput { status, stdout, stderr, elapsed: started.elapsed() })
}

fn wait_with_timeout(child: &mut Child, timeout: Duration) -> Result<i32, LeanRunnerError> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status.code().unwrap_or(-1));
        }
        if started.elapsed() >= timeout {
            #[cfg(unix)]
            unsafe {
                // The child is the process-group leader. Kill the complete
                // Lake/Lean tree, including descendants holding our pipes.
                let _ = libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(LeanRunnerError::Timeout(timeout));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

pub fn write_claim_manifest(
    package_directory: &Path,
    theorem: &str,
    cache_key: &[u8; 32],
) -> Result<(), LeanRunnerError> {
    let contents = Vec::new();
    write_claim_manifest_for_check(
        package_directory,
        theorem,
        cache_key,
        Path::new("Check.lean"),
        &contents,
    )
}

pub fn write_claim_manifest_for_check(
    package_directory: &Path,
    theorem: &str,
    cache_key: &[u8; 32],
    check_file: &Path,
    check_contents: &[u8],
) -> Result<(), LeanRunnerError> {
    validate_relative_path(check_file).map_err(|_| LeanRunnerError::InvalidCheckPath)?;
    validate_theorem_name(theorem)?;
    fs::create_dir_all(package_directory)?;
    let check_path = package_directory.join(check_file);
    if let Some(parent) = check_path.parent() {
        fs::create_dir_all(parent)?;
    }
    atomic_write(&check_path, check_contents)?;
    let manifest = ClaimManifest {
        theorem: theorem.to_owned(),
        cache_key: hex_bytes(cache_key),
        check_file: check_file.to_path_buf(),
        check_file_sha256: sha256_bytes(check_contents),
    };
    let encoded = serde_json::to_vec(&manifest).map_err(|_| LeanRunnerError::InvalidManifest)?;
    atomic_write(&package_directory.join(MANIFEST_FILE), &encoded)?;
    Ok(())
}

fn validate_theorem_name(theorem: &str) -> Result<(), LeanRunnerError> {
    if theorem.is_empty() ||
        theorem.split('.').any(|part| {
            part.is_empty() ||
                !part.chars().enumerate().all(|(index, character)| {
                    character == '_' ||
                        character.is_ascii_alphanumeric() &&
                            (index > 0 || character.is_ascii_alphabetic() || character == '_')
                })
        })
    {
        return Err(LeanRunnerError::InvalidManifest);
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_tool_is_infrastructure_error() {
        let directory = tempfile::tempdir().unwrap();
        let contents = b"theorem T : True := by trivial\n".to_vec();
        write_claim_manifest_for_check(
            directory.path(),
            "T",
            &[1; 32],
            Path::new("Check.lean"),
            &contents,
        )
        .unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("Check.lean"),
            theorem: "T".to_owned(),
            cache_key: [1; 32],
            check_file_sha256: sha256_bytes(&contents),
        };
        let runner = LeanRunner::for_test(
            PathBuf::from("/definitely/missing/lake"),
            Duration::from_millis(10),
        );
        assert!(matches!(runner.run(&request), Err(LeanRunnerError::ToolNotFound(_))));
    }

    #[test]
    fn actual_lean_checks_the_expected_theorem() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let lean_launcher = directory.path().join("lake");
        fs::write(&lean_launcher, b"#!/bin/sh\nshift 2\nexec lean \"$@\"\n").unwrap();
        fs::set_permissions(&lean_launcher, fs::Permissions::from_mode(0o755)).unwrap();
        let key = [3; 32];
        let contents = b"theorem T : True := by trivial\n".to_vec();
        write_claim_manifest_for_check(
            directory.path(),
            "T",
            &key,
            Path::new("Check.lean"),
            &contents,
        )
        .unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("Check.lean"),
            theorem: "T".to_owned(),
            cache_key: key,
            check_file_sha256: sha256_bytes(&contents),
        };
        let output = LeanRunner::for_test(lean_launcher, Duration::from_secs(10))
            .run(&request)
            .unwrap_or_else(|error| panic!("actual Lean runner failed: {error:?}"));
        assert_eq!(output.status, 0);
    }

    #[test]
    fn theorem_using_allowed_foundational_axiom_is_accepted() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let lean_launcher = directory.path().join("lake");
        fs::write(&lean_launcher, b"#!/bin/sh\nshift 2\nexec lean \"$@\"\n").unwrap();
        fs::set_permissions(&lean_launcher, fs::Permissions::from_mode(0o755)).unwrap();
        let key = [7; 32];
        let contents = b"theorem T : True :=\n  Classical.choice (show Nonempty True from Nonempty.intro True.intro)\n";
        write_claim_manifest_for_check(
            directory.path(),
            "T",
            &key,
            Path::new("Check.lean"),
            contents,
        )
        .unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("Check.lean"),
            theorem: "T".to_owned(),
            cache_key: key,
            check_file_sha256: sha256_bytes(contents),
        };
        let output = LeanRunner::for_test(lean_launcher, Duration::from_secs(10))
            .run(&request)
            .unwrap_or_else(|error| panic!("allowed foundational axiom was rejected: {error:?}"));
        assert_eq!(output.status, 0);
    }

    #[test]
    fn theorem_using_custom_axiom_is_rejected_by_axiom_audit() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let lean_launcher = directory.path().join("lake");
        fs::write(&lean_launcher, b"#!/bin/sh\nshift 2\nexec lean \"$@\"\n").unwrap();
        fs::set_permissions(&lean_launcher, fs::Permissions::from_mode(0o755)).unwrap();
        let key = [5; 32];
        let contents = b"#eval IO.println \"'T' does not depend on any axioms\"\naxiom projectAxiom : False\ntheorem T : True := by exact projectAxiom.elim\n";
        write_claim_manifest_for_check(
            directory.path(),
            "T",
            &key,
            Path::new("Check.lean"),
            contents,
        )
        .unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("Check.lean"),
            theorem: "T".to_owned(),
            cache_key: key,
            check_file_sha256: sha256_bytes(contents),
        };
        let error =
            LeanRunner::for_test(lean_launcher, Duration::from_secs(10)).run(&request).unwrap_err();
        assert!(
            matches!(error, LeanRunnerError::DisallowedAxioms(ref names) if names == "projectAxiom")
        );
    }

    #[test]
    fn theorem_using_sorry_is_rejected_by_axiom_audit() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let lean_launcher = directory.path().join("lake");
        fs::write(&lean_launcher, b"#!/bin/sh\nshift 2\nexec lean \"$@\"\n").unwrap();
        fs::set_permissions(&lean_launcher, fs::Permissions::from_mode(0o755)).unwrap();
        let key = [6; 32];
        let contents = b"theorem T : True := by sorry\n";
        write_claim_manifest_for_check(
            directory.path(),
            "T",
            &key,
            Path::new("Check.lean"),
            contents,
        )
        .unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("Check.lean"),
            theorem: "T".to_owned(),
            cache_key: key,
            check_file_sha256: sha256_bytes(contents),
        };
        let error =
            LeanRunner::for_test(lean_launcher, Duration::from_secs(10)).run(&request).unwrap_err();
        assert!(
            matches!(error, LeanRunnerError::DisallowedAxioms(ref names) if names == "sorryAx")
        );
    }

    #[test]
    fn axiom_audit_rejects_duplicate_matching_reports_after_delimiter() {
        let output = b"'T' does not depend on any axioms\nMXX_AXIOM_AUDIT_nonce\n'T' does not depend on any axioms\n'T' does not depend on any axioms\n";
        assert!(matches!(
            audit_axiom_report(output, "T", "MXX_AXIOM_AUDIT_nonce"),
            Err(LeanRunnerError::InvalidAxiomReport(theorem)) if theorem == "T"
        ));
    }

    #[test]
    fn axiom_audit_rejects_data_before_or_after_report() {
        for output in [
            b"MXX_AXIOM_AUDIT_nonce\nleading\n'T' does not depend on any axioms\n".as_slice(),
            b"MXX_AXIOM_AUDIT_nonce\n'T' does not depend on any axioms\ntrailing\n".as_slice(),
            b"MXX_AXIOM_AUDIT_nonce\n\n'T' does not depend on any axioms\n".as_slice(),
            b"MXX_AXIOM_AUDIT_nonce\n'T' does not depend on any axioms\n\n".as_slice(),
        ] {
            assert!(matches!(
                audit_axiom_report(output, "T", "MXX_AXIOM_AUDIT_nonce"),
                Err(LeanRunnerError::InvalidAxiomReport(theorem)) if theorem == "T"
            ));
        }
    }

    #[test]
    fn axiom_audit_uses_the_last_delimiter_and_rejects_ordered_duplicate_output() {
        let output = b"MXX_AXIOM_AUDIT_nonce\n'T' does not depend on any axioms\nMXX_AXIOM_AUDIT_nonce\n'T' does not depend on any axioms\nextra\n";
        assert!(matches!(
            audit_axiom_report(output, "T", "MXX_AXIOM_AUDIT_nonce"),
            Err(LeanRunnerError::InvalidAxiomReport(theorem)) if theorem == "T"
        ));
    }

    #[test]
    fn production_runner_rebuilds_clean_two_dependency_package() {
        let directory = tempfile::tempdir().unwrap();
        for (name, module) in [("dep_a", "DepA"), ("dep_b", "DepB")] {
            let dep = directory.path().join(name);
            fs::create_dir_all(&dep).unwrap();
            fs::write(
                dep.join("lakefile.toml"),
                format!("name = \"{module}\"\n[[lean_lib]]\nname = \"{module}\"\n"),
            )
            .unwrap();
            fs::write(dep.join(format!("{module}.lean")), "def value := True\n").unwrap();
        }
        fs::write(directory.path().join("lakefile.toml"), "name = \"candidate\"\n[[require]]\nname = \"DepA\"\npath = \"dep_a\"\n[[require]]\nname = \"DepB\"\npath = \"dep_b\"\n[[lean_lib]]\nname = \"MxxGenerated\"\n").unwrap();
        fs::write(directory.path().join("MxxGenerated.lean"), "def generated := True\n").unwrap();
        let contents = b"namespace Mxx.We.DiamondWE\ndef CorrectnessClaim (_ : Nat) : Prop := True\nend Mxx.We.DiamondWE\nnamespace Mxx.We.Golden.DiamondWE\ndef candidate : Nat := 0\ntheorem correct : Mxx.We.DiamondWE.CorrectnessClaim candidate := True.intro\nend Mxx.We.Golden.DiamondWE\n";
        write_claim_manifest_for_check(
            directory.path(),
            DIAMOND_PROOF_THEOREM,
            &[9; 32],
            Path::new("Check.lean"),
            contents,
        )
        .unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("Check.lean"),
            theorem: DIAMOND_PROOF_THEOREM.to_owned(),
            cache_key: [9; 32],
            check_file_sha256: sha256_bytes(contents),
        };
        let output = LeanRunner::default().run(&request).unwrap();
        assert_eq!(output.status, 0);
    }

    #[test]
    fn metadata_without_theorem_check_is_rejected() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let lean_launcher = directory.path().join("lake");
        fs::write(&lean_launcher, b"#!/bin/sh\nshift 2\nexec lean \"$@\"\n").unwrap();
        fs::set_permissions(&lean_launcher, fs::Permissions::from_mode(0o755)).unwrap();
        let key = [4; 32];
        let valid = b"theorem Other : True := by trivial\n".to_vec();
        write_claim_manifest_for_check(
            directory.path(),
            "T",
            &key,
            Path::new("Check.lean"),
            &valid,
        )
        .unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("Check.lean"),
            theorem: "T".to_owned(),
            cache_key: key,
            check_file_sha256: sha256_bytes(&valid),
        };
        let error = LeanRunner::for_test(lean_launcher, Duration::from_secs(120))
            .run(&request)
            .unwrap_err();
        assert!(matches!(&error, LeanRunnerError::Failed { .. }));
        assert!(
            error.to_string().contains("unknown identifier") ||
                error.to_string().contains("unknown constant") ||
                error.to_string().contains("Unknown identifier"),
            "unexpected Lean diagnostic: {error}"
        );
    }

    #[test]
    fn traversal_check_file_is_rejected_before_spawn() {
        let directory = tempfile::tempdir().unwrap();
        write_claim_manifest(directory.path(), "T", &[1; 32]).unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("../Check.lean"),
            theorem: "T".to_owned(),
            cache_key: [1; 32],
            check_file_sha256: [0; 32],
        };
        assert!(matches!(
            LeanRunner::default().run(&request),
            Err(LeanRunnerError::InvalidCheckPath)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn timeout_kills_a_child_spawning_process_group() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("fake-lake");
        let contents = b"theorem Other : True := by trivial\n".to_vec();
        let child_pid_file = directory.path().join("child.pid");
        fs::write(
            &script,
            format!("#!/bin/sh\nsleep 30 & echo $! > '{}'; wait\n", child_pid_file.display()),
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        write_claim_manifest_for_check(
            directory.path(),
            "T",
            &[2; 32],
            Path::new("Check.lean"),
            &contents,
        )
        .unwrap();
        let request = LeanRunRequest {
            package_directory: directory.path().to_path_buf(),
            check_file: PathBuf::from("Check.lean"),
            theorem: "T".to_owned(),
            cache_key: [2; 32],
            check_file_sha256: sha256_bytes(&contents),
        };
        let runner = LeanRunner::for_test(script, Duration::from_millis(30));
        let started = Instant::now();
        assert!(matches!(runner.run(&request), Err(LeanRunnerError::Timeout(_))));
        assert!(started.elapsed() < Duration::from_secs(2));
        let child_pid: i32 = fs::read_to_string(child_pid_file).unwrap().trim().parse().unwrap();
        let mut gone = false;
        for _ in 0..200 {
            // The process group kill must include the spawned `sleep`.
            if unsafe { libc::kill(child_pid, 0) } != 0 {
                gone = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(gone, "descendant process {child_pid} survived timeout");
    }
}
