//! Uncached, dependency-ordered checking of locally generated Lean modules.
//!
//! The final certificate is fixed by the IR linker and checked with a closed axiom policy.

use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CheckError {
    #[error("{operation}: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("could not start {program}: {source}")]
    Spawn {
        program: String,
        #[source]
        source: io::Error,
    },
    #[error("module {module} imports missing dependency {dependency}")]
    MissingDependency { module: String, dependency: String },
    #[error("local Lean import cycle: {0:?}")]
    Cycle(Vec<String>),
    #[error("no Lean source modules in {0}")]
    Empty(PathBuf),
    #[error("invalid Lean environment: {0}")]
    Environment(String),
    #[error("Lean process {module} failed ({status}); log: {log}")]
    Process { module: String, status: ExitStatus, log: PathBuf },
    #[error("Lean checking timed out during {module}; log: {log}")]
    Timeout { module: String, log: PathBuf },
}

fn io_error(operation: &'static str) -> impl FnOnce(io::Error) -> CheckError {
    move |source| CheckError::Io { operation, source }
}

/// Generated files use plain, single-line imports. This is not a general Lean parser.
fn local_imports(directory: &Path) -> Result<BTreeMap<String, BTreeSet<String>>, CheckError> {
    let mut modules = BTreeMap::new();
    for entry in fs::read_dir(directory).map_err(io_error("read generated directory"))? {
        let path = entry.map_err(io_error("read generated entry"))?.path();
        if path.extension().is_none_or(|extension| extension != "lean") {
            continue;
        }
        let module = path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or_else(|| CheckError::Environment("generated module name is not UTF-8".into()))?;
        let source = fs::read_to_string(&path).map_err(io_error("read generated source"))?;
        let imports = source
            .lines()
            .filter_map(|line| line.trim().strip_prefix("import "))
            .flat_map(|line| line.split("--").next().unwrap().split_whitespace())
            .map(|name| name.replace(['«', '»'], ""))
            .collect();
        modules.insert(module.to_owned(), imports);
    }
    if modules.is_empty() {
        return Err(CheckError::Empty(directory.to_owned()));
    }
    Ok(modules)
}

fn validate_dependencies(
    modules: &BTreeMap<String, BTreeSet<String>>,
    library_paths: &[PathBuf],
) -> Result<(), CheckError> {
    for (module, imports) in modules {
        for dependency in imports {
            if !modules.contains_key(dependency) {
                let relative = PathBuf::from(dependency.replace('.', "/")).with_extension("olean");
                if !library_paths.iter().any(|path| path.join(&relative).is_file()) {
                    return Err(CheckError::MissingDependency {
                        module: module.clone(),
                        dependency: dependency.clone(),
                    });
                }
            }
        }
    }
    let mut remaining: BTreeSet<_> = modules.keys().cloned().collect();
    while !remaining.is_empty() {
        let ready: Vec<_> = remaining
            .iter()
            .filter(|name| modules[*name].iter().all(|dependency| !remaining.contains(dependency)))
            .cloned()
            .collect();
        if ready.is_empty() {
            return Err(CheckError::Cycle(remaining.into_iter().collect()));
        }
        for name in ready {
            remaining.remove(&name);
        }
    }
    Ok(())
}

struct Running {
    module: String,
    child: Child,
    log: PathBuf,
}

impl Drop for Running {
    fn drop(&mut self) {
        // Every early return, including a sibling failure, terminates and reaps our children.
        #[cfg(unix)]
        {
            // Lake may own compiler children; terminate the whole isolated group.
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{}", self.child.id())])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(command: &mut Command, module: String, log: PathBuf) -> Result<Running, CheckError> {
    let output = File::create(&log).map_err(io_error("create Lean log"))?;
    let stderr = output.try_clone().map_err(io_error("clone Lean log handle"))?;
    let program = command.get_program().to_string_lossy().into_owned();
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(output))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|source| CheckError::Spawn { program, source })?;
    Ok(Running { module, child, log })
}

fn poll(running: &mut Running, deadline: Instant) -> Result<bool, CheckError> {
    if let Some(status) = running.child.try_wait().map_err(io_error("poll Lean process"))? {
        if !status.success() {
            return Err(CheckError::Process {
                module: running.module.clone(),
                status,
                log: running.log.clone(),
            });
        }
        return Ok(true);
    }
    if Instant::now() >= deadline {
        return Err(CheckError::Timeout {
            module: running.module.clone(),
            log: running.log.clone(),
        });
    }
    Ok(false)
}

#[derive(Debug)]
pub struct CheckedCandidate {
    pub sources: BTreeMap<String, String>,
    pub dependency_sha256: String,
    pub workspace_sources: BTreeMap<PathBuf, String>,
    pub check_directory: PathBuf,
    pub logs: BTreeMap<String, PathBuf>,
    pub elapsed: Duration,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn source_identity(directory: &Path) -> Result<BTreeMap<String, String>, CheckError> {
    local_imports(directory)?
        .keys()
        .map(|name| {
            let bytes = fs::read(directory.join(format!("{name}.lean")))
                .map_err(io_error("hash candidate source"))?;
            Ok((name.clone(), digest(&bytes)))
        })
        .collect()
}

fn workspace_source_identity(
    source_roots: &[PathBuf],
) -> Result<BTreeMap<PathBuf, String>, CheckError> {
    fn visit(path: &Path, sources: &mut BTreeMap<PathBuf, String>) -> Result<(), CheckError> {
        for entry in fs::read_dir(path).map_err(io_error("read workspace Lean sources"))? {
            let path = entry.map_err(io_error("read workspace Lean source entry"))?.path();
            if path.file_name().is_some_and(|name| name == ".lake") {
                continue;
            }
            if path.is_dir() {
                visit(&path, sources)?;
            } else {
                sources.insert(
                    path.clone(),
                    digest(&fs::read(path).map_err(io_error("hash workspace Lean source"))?),
                );
            }
        }
        Ok(())
    }
    let mut sources = BTreeMap::new();
    for path in source_roots {
        visit(path, &mut sources)?;
    }
    Ok(sources)
}

fn dependency_identity(
    paths: &[PathBuf],
    lean: &Path,
    deadline: Instant,
) -> Result<String, CheckError> {
    fn visit(path: &Path, files: &mut BTreeSet<PathBuf>) -> Result<(), CheckError> {
        for entry in fs::read_dir(path).map_err(io_error("read dependency directory"))? {
            let path = entry.map_err(io_error("read dependency entry"))?.path();
            if path.is_dir() {
                visit(&path, files)?;
            } else {
                files.insert(path);
            }
        }
        Ok(())
    }
    let mut files = BTreeSet::from([lean.to_owned()]);
    let toolchain_library = lean
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| CheckError::Environment("invalid pinned Lean executable path".into()))?
        .join("lib/lean");
    visit(&toolchain_library, &mut files)?;
    for path in paths {
        visit(path, &mut files)?;
    }
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 65536];
    for path in files {
        hash.update(path.to_string_lossy().as_bytes());
        hash.update([0]);
        let mut file = File::open(path).map_err(io_error("open dependency"))?;
        loop {
            if Instant::now() >= deadline {
                return Err(CheckError::Environment("timeout while hashing dependencies".into()));
            }
            let count = file.read(&mut buffer).map_err(io_error("hash dependency"))?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_axioms(log: &str) -> Result<(), CheckError> {
    let prefix = "'GeneratedCertificate.correctness' depends on axioms:";
    let text = log.trim();
    if text == "'GeneratedCertificate.correctness' does not depend on any axioms" {
        return Ok(());
    }
    let list = text
        .strip_prefix(prefix)
        .map(str::trim)
        .and_then(|text| text.strip_prefix('['))
        .and_then(|text| text.strip_suffix(']'))
        .ok_or_else(|| {
            CheckError::Environment("missing or unexpected correctness axiom output".into())
        })?;
    for axiom in list.split(',').map(str::trim).filter(|name| !name.is_empty()) {
        if !matches!(axiom, "propext" | "Classical.choice" | "Quot.sound") {
            return Err(CheckError::Environment(format!("untrusted correctness axiom: {axiom}")));
        }
    }
    Ok(())
}

fn lean_environment(
    package: &Path,
    extra_paths: &[PathBuf],
    directory: &Path,
    deadline: Instant,
) -> Result<(Vec<PathBuf>, PathBuf), CheckError> {
    let mut command = Command::new("lake");
    command.current_dir(&package).env_remove("LEAN_PATH").args([
        "env",
        "printenv",
        "LEAN_PATH",
        "PATH",
    ]);
    if !extra_paths.is_empty() {
        command.env(
            "LEAN_PATH",
            std::env::join_paths(extra_paths)
                .map_err(|error| CheckError::Environment(error.to_string()))?,
        );
    }
    let mut environment =
        start(&mut command, "environment".into(), directory.join("environment.log"))?;
    while !poll(&mut environment, deadline)? {
        thread::sleep(Duration::from_millis(10));
    }
    let text = fs::read_to_string(&environment.log).map_err(io_error("read environment"))?;
    let lines: Vec<_> = text.lines().collect();
    if lines.len() != 2 {
        return Err(CheckError::Environment("unexpected Lake environment output".into()));
    }
    let libraries: Vec<_> = std::env::split_paths(lines[0])
        .map(|path| if path.is_absolute() { path } else { package.join(path) })
        .filter(|path| path.is_dir())
        .map(|path| fs::canonicalize(path).map_err(io_error("resolve dependency path")))
        .collect::<Result<_, _>>()?;
    let lean = std::env::split_paths(lines[1])
        .map(|path| path.join("lean"))
        .find(|path| path.is_file())
        .ok_or_else(|| CheckError::Environment("missing pinned Lean".into()))?;
    Ok((libraries, lean))
}

/// Compile a generated DAG against existing package objects. The caller owns
/// the final theorem policy; certificate checking below additionally binds the
/// core claim and fingerprints source/dependency snapshots.
pub fn check_generated_modules(
    package: &Path,
    extra_paths: &[PathBuf],
    directory: &Path,
    timeout: Duration,
) -> Result<(), CheckError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| CheckError::Environment("timeout is too large".into()))?;
    let directory = fs::canonicalize(directory).map_err(io_error("resolve generated directory"))?;
    let (libraries, lean) = lean_environment(package, extra_paths, &directory, deadline)?;
    check_modules(package, &directory, &libraries, &lean, deadline)?;
    Ok(())
}

fn check_modules(
    package: &Path,
    directory: &Path,
    libraries: &[PathBuf],
    lean: &Path,
    deadline: Instant,
) -> Result<BTreeMap<String, PathBuf>, CheckError> {
    let modules = local_imports(directory)?;
    validate_dependencies(&modules, libraries)?;
    if modules
        .iter()
        .any(|(name, imports)| name != "Certificate" && imports.contains("Certificate"))
    {
        return Err(CheckError::Environment("certificate must be the final module".into()));
    }
    let lean_path: OsString = std::env::join_paths(
        std::iter::once(directory).chain(libraries.iter().map(PathBuf::as_path)),
    )
    .map_err(|error| CheckError::Environment(error.to_string()))?;
    // Preserve WE's bounded parallelism without expanding the dependency DAG.
    let concurrency = thread::available_parallelism().map_or(1, |count| count.get().min(3));
    let mut pending: BTreeSet<_> = modules.keys().cloned().collect();
    let mut completed = BTreeSet::new();
    let mut running = Vec::<Running>::new();
    let mut logs = BTreeMap::new();
    while !pending.is_empty() || !running.is_empty() {
        if Instant::now() >= deadline {
            return Err(CheckError::Timeout {
                module: "generated DAG".into(),
                log: directory.join("environment.log"),
            });
        }
        let ready: Vec<_> = pending
            .iter()
            .filter(|name| {
                (name.as_str() != "Certificate" || (pending.len() == 1 && running.is_empty())) &&
                    modules[*name]
                        .iter()
                        .all(|dep| !modules.contains_key(dep) || completed.contains(dep))
            })
            .take(concurrency - running.len())
            .cloned()
            .collect();
        for name in ready {
            let mut command = Command::new(lean);
            command
                .current_dir(package)
                .env("LEAN_PATH", &lean_path)
                .args(["-s131072", "-DmaxHeartbeats=2000000", "-DmaxRecDepth=16384", "-R"])
                .arg(directory)
                .arg("-o")
                .arg(directory.join(format!("{name}.olean")))
                .arg(directory.join(format!("{name}.lean")));
            running.push(start(&mut command, name.clone(), directory.join(format!("{name}.log")))?);
            pending.remove(&name);
        }
        let mut index = 0;
        while index < running.len() {
            if !poll(&mut running[index], deadline)? {
                index += 1;
                continue;
            }
            let process = running.swap_remove(index);
            let log = fs::read_to_string(&process.log).map_err(io_error("read checker log"))?;
            if log.contains("declaration uses 'sorry'") || log.contains("error:") {
                return Err(CheckError::Environment(format!(
                    "untrusted diagnostics in {}",
                    process.module
                )));
            }
            if !directory.join(format!("{}.olean", process.module)).is_file() {
                return Err(CheckError::Environment(format!(
                    "missing checked object for {}",
                    process.module
                )));
            }
            logs.insert(process.module.clone(), process.log.clone());
            completed.insert(process.module.clone());
        }
        if !running.is_empty() {
            thread::sleep(Duration::from_millis(10));
        }
    }
    Ok(logs)
}

/// Check exactly the supplied source snapshot, with the IR-generated theorem
/// statement and a fresh local object directory. No candidate object is reused.
/// The returned record identifies the checked bytes, dependencies, and durable logs.
/// `package` selects the Lake default targets; `source_roots` lists the complete
/// local Lean dependency sources whose identity must remain fixed during checking.
pub fn check_candidate(
    package: &Path,
    source_roots: &[PathBuf],
    directory: &Path,
    expected_sources: &BTreeMap<String, String>,
    proof_module: &str,
    proof_theorem: &str,
    timeout: Duration,
) -> Result<CheckedCandidate, CheckError> {
    let started = Instant::now();
    let deadline = started
        .checked_add(timeout)
        .ok_or_else(|| CheckError::Environment("timeout is too large".into()))?;
    let directory = fs::canonicalize(directory).map_err(io_error("resolve candidate"))?;
    let certificate = mxx_ir_core::lean::claim::assemble_certificate(proof_module, proof_theorem)
        .map_err(CheckError::Environment)?;
    if expected_sources.get("Certificate") != Some(&certificate) ||
        !expected_sources.contains_key("Claim") ||
        !expected_sources.contains_key(proof_module)
    {
        return Err(CheckError::Environment(
            "candidate must contain the IR-generated certificate, claim, and proof".into(),
        ));
    }
    let expected: BTreeMap<_, _> = expected_sources
        .iter()
        .map(|(name, source)| (name.clone(), digest(source.as_bytes())))
        .collect();
    if source_identity(&directory)? != expected {
        return Err(CheckError::Environment(
            "candidate sources differ from expected snapshot".into(),
        ));
    }
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let work = loop {
        let nonce = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = directory.join(format!("checked-{}-{nonce}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => break path,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_error("create fresh checker directory")(error)),
        }
    };
    for (name, source) in expected_sources {
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(CheckError::Environment(
                "candidate requires simple local module names".into(),
            ));
        }
        fs::write(work.join(format!("{name}.lean")), source)
            .map_err(io_error("write frozen source"))?;
    }
    let source_dependencies = workspace_source_identity(source_roots)?;
    // Lake validates/rebuilds the static closure from current sources before any
    // object identity is captured. Candidate modules never enter this package.
    let mut build = Command::new("lake");
    build.current_dir(&package).env_remove("LEAN_PATH").args(["build"]);
    let mut build = start(&mut build, "static dependencies".into(), work.join("dependencies.log"))?;
    while !poll(&mut build, deadline)? {
        thread::sleep(Duration::from_millis(10));
    }
    if workspace_source_identity(source_roots)? != source_dependencies {
        return Err(CheckError::Environment(
            "workspace Lean sources changed during dependency build".into(),
        ));
    }
    let (libraries, lean) = lean_environment(package, &[], &work, deadline)?;
    let dependencies = dependency_identity(&libraries, &lean, deadline)?;
    let logs = check_modules(package, &work, &libraries, &lean, deadline)?;
    validate_axioms(
        &fs::read_to_string(&logs["Certificate"]).map_err(io_error("read certificate log"))?,
    )?;
    if source_identity(&directory)? != expected ||
        source_identity(&work)? != expected ||
        dependency_identity(&libraries, &lean, deadline)? != dependencies ||
        workspace_source_identity(source_roots)? != source_dependencies
    {
        return Err(CheckError::Environment("source or dependency changed while checking".into()));
    }
    if Instant::now() >= deadline {
        return Err(CheckError::Timeout {
            module: "identity verification".into(),
            log: work.join("environment.log"),
        });
    }
    Ok(CheckedCandidate {
        sources: expected,
        dependency_sha256: dependencies,
        workspace_sources: source_dependencies,
        check_directory: work,
        logs,
        elapsed: started.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the pinned Lean toolchain and static package build"]
    fn checks_sound_core_certificate_with_real_package_environment() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/runtime_checker_positive");
        fs::create_dir_all(&directory).unwrap();
        let sources = BTreeMap::from([
            ("Claim".into(), "namespace GeneratedClaim\ndef CorrectnessClaim : Prop := True\nend GeneratedClaim\n".into()),
            ("Proof".into(), "import Claim\nnamespace Proof\ntheorem correctness : GeneratedClaim.CorrectnessClaim := True.intro\nend Proof\n".into()),
            ("Certificate".into(), mxx_ir_core::lean::claim::assemble_certificate("Proof", "Proof.correctness").unwrap()),
        ]);
        for (name, source) in &sources {
            fs::write(directory.join(format!("{name}.lean")), source).unwrap();
        }
        let checked = check_candidate(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("lean"),
            &[
                Path::new(env!("CARGO_MANIFEST_DIR")).join("lean"),
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../primitives/lean"),
            ],
            &directory,
            &sources,
            "Proof",
            "Proof.correctness",
            Duration::from_secs(600),
        )
        .unwrap();
        assert_eq!(checked.sources.len(), 3);
        assert!(checked.check_directory.join("Certificate.olean").is_file());
        validate_axioms(&fs::read_to_string(&checked.logs["Certificate"]).unwrap()).unwrap();
        assert!(!checked.workspace_sources.is_empty());
    }

    #[test]
    fn certificate_axioms_are_closed_and_theorem_specific() {
        validate_axioms("'GeneratedCertificate.correctness' depends on axioms: [propext, Classical.choice, Quot.sound]\n").unwrap();
        validate_axioms("'GeneratedCertificate.correctness' does not depend on any axioms\n")
            .unwrap();
        for log in [
            "'GeneratedCertificate.correctness' depends on axioms: [sorryAx]",
            "'GeneratedCertificate.correctness' depends on axioms: [My.assumption]",
            "'Other.correctness' depends on axioms: [propext]",
            "error: failure\n'GeneratedCertificate.correctness' depends on axioms: [propext]",
            "'GeneratedCertificate.correctness' depends on axioms: [propext]\nextra",
        ] {
            assert!(validate_axioms(log).is_err(), "{log}");
        }
    }

    #[test]
    fn candidate_requires_mechanical_certificate_before_spawning() {
        let directory = tempfile::tempdir().unwrap();
        let sources =
            BTreeMap::from([("Certificate".into(), "theorem fake : True := trivial".into())]);
        assert!(matches!(
            check_candidate(
                directory.path(),
                &[],
                directory.path(),
                &sources,
                "Proof",
                "Proof.correctness",
                Duration::from_secs(1)
            ),
            Err(CheckError::Environment(_))
        ));
    }

    #[test]
    fn source_identity_detects_addition_and_mutation() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("A.lean"), "theorem a : True := trivial").unwrap();
        let first = source_identity(directory.path()).unwrap();
        fs::write(directory.path().join("A.lean"), "theorem a : False := by sorry").unwrap();
        assert_ne!(first, source_identity(directory.path()).unwrap());
        fs::write(directory.path().join("B.lean"), "import A").unwrap();
        assert_eq!(source_identity(directory.path()).unwrap().len(), 2);
    }

    #[test]
    fn local_dag_orders_shared_dependencies_and_detects_cycles() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("A.lean"), "theorem a : True := True.intro\n").unwrap();
        fs::write(directory.path().join("B.lean"), "import A -- actual dependency\n").unwrap();
        fs::write(directory.path().join("C.lean"), "import A\nimport B\n").unwrap();
        let modules = local_imports(directory.path()).unwrap();
        validate_dependencies(&modules, &[]).unwrap();
        fs::write(directory.path().join("A.lean"), "import C\n").unwrap();
        assert!(matches!(
            validate_dependencies(&local_imports(directory.path()).unwrap(), &[]),
            Err(CheckError::Cycle(_))
        ));
    }

    #[test]
    fn stale_local_olean_does_not_supply_a_missing_source_dependency() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("A.lean"), "import Missing\n").unwrap();
        fs::write(directory.path().join("Missing.olean"), "stale").unwrap();
        assert!(matches!(
            validate_dependencies(&local_imports(directory.path()).unwrap(), &[]),
            Err(CheckError::MissingDependency { .. })
        ));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn exited_leader_does_not_leave_its_descendants_running() {
        let directory = tempfile::tempdir().unwrap();
        let pid_file = directory.path().join("child.pid");
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 30 & echo $! > \"$1\"; exit 1", "sh"]).arg(&pid_file);
        let mut process =
            start(&mut command, "exited leader".into(), directory.path().join("log")).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match poll(&mut process, deadline) {
                Err(CheckError::Process { .. }) => break,
                Ok(false) => thread::sleep(Duration::from_millis(10)),
                result => panic!("unexpected result: {result:?}"),
            }
        }
        let pid = fs::read_to_string(pid_file).unwrap();
        let status_path = PathBuf::from(format!("/proc/{}/status", pid.trim()));
        drop(process);
        loop {
            match fs::read_to_string(&status_path) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => break,
                Ok(status)
                    if status
                        .lines()
                        .any(|line| line.starts_with("State:") && line.contains('Z')) =>
                {
                    break
                }
                _ if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                result => panic!("descendant still running: {result:?}"),
            }
        }
    }

    #[test]
    fn timeout_reaps_child_and_nonzero_is_distinct() {
        let directory = tempfile::tempdir().unwrap();
        let mut command = Command::new("sleep");
        command.arg("10");
        let mut child =
            start(&mut command, "sleep".into(), directory.path().join("timeout.log")).unwrap();
        assert!(matches!(poll(&mut child, Instant::now()), Err(CheckError::Timeout { .. })));
        let pid = child.child.id();
        drop(child);
        #[cfg(target_os = "linux")]
        assert!(!Path::new(&format!("/proc/{pid}")).exists());
        let mut child =
            start(&mut Command::new("false"), "false".into(), directory.path().join("failure.log"))
                .unwrap();
        loop {
            match poll(&mut child, Instant::now() + Duration::from_secs(10)) {
                Err(CheckError::Process { .. }) => break,
                Ok(false) => thread::sleep(Duration::from_millis(10)),
                other => panic!("unexpected result: {other:?}"),
            }
        }
    }

    #[test]
    fn output_is_logged_without_a_pipe_and_spawn_failure_is_distinct() {
        let directory = tempfile::tempdir().unwrap();
        let log = directory.path().join("large.log");
        let mut command = Command::new("seq");
        command.args(["1", "100000"]);
        let mut child = start(&mut command, "seq".into(), log.clone()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !poll(&mut child, deadline).unwrap() {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(fs::metadata(log).unwrap().len() > 65536);
        assert!(matches!(
            start(
                &mut Command::new(directory.path().join("no-program")),
                "missing".into(),
                directory.path().join("missing.log")
            ),
            Err(CheckError::Spawn { .. })
        ));
    }
    #[test]
    #[ignore = "requires the pinned Lean/Lake installation and built crate dependencies"]
    fn checks_fresh_dag_and_rejects_changed_source() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("A.lean"), "theorem a : True := True.intro\n").unwrap();
        fs::write(directory.path().join("B.lean"), "import A\nexample : True := a\n").unwrap();
        fs::write(directory.path().join("C.lean"), "theorem c : True := True.intro\n").unwrap();
        fs::write(directory.path().join("D.lean"), "import B\nimport C\nexample : True := c\n")
            .unwrap();
        if let Err(error) = check_generated_modules(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("lean"),
            &[],
            directory.path(),
            Duration::from_secs(30),
        ) {
            if let CheckError::Process { log, .. } = &error {
                panic!("{error}: {}", fs::read_to_string(log).unwrap());
            }
            panic!("{error}");
        }
        assert!(directory.path().join("A.olean").is_file());
        assert!(directory.path().join("B.log").is_file());
        fs::write(directory.path().join("A.lean"), "theorem a : False := by exact True.intro\n")
            .unwrap();
        assert!(matches!(
            check_generated_modules(
                &Path::new(env!("CARGO_MANIFEST_DIR")).join("lean"),
                &[],
                directory.path(),
                Duration::from_secs(30)
            ),
            Err(CheckError::Process { .. })
        ));
    }

    #[test]
    #[ignore = "requires the pinned Lean/Lake installation"]
    fn generated_module_timeout_is_an_infrastructure_error() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("Slow.lean"), "#eval IO.sleep 10000\n").unwrap();
        let started = Instant::now();
        assert!(matches!(
            check_generated_modules(
                &Path::new(env!("CARGO_MANIFEST_DIR")).join("lean"),
                &[],
                directory.path(),
                Duration::from_secs(1)
            ),
            Err(CheckError::Timeout { .. })
        ));
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
