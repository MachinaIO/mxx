//! Uncached, dependency-ordered checking of locally generated Lean modules.
//!
//! This runner checks compilation only. Its caller owns theorem identity and axiom policy.

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::{self, File},
    io,
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
            .map(str::to_owned)
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
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(command: &mut Command, module: String, log: PathBuf) -> Result<Running, CheckError> {
    let output = File::create(&log).map_err(io_error("create Lean log"))?;
    let stderr = output.try_clone().map_err(io_error("clone Lean log handle"))?;
    let program = command.get_program().to_string_lossy().into_owned();
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

/// Recompile every local module. `timeout` is a deadline for the entire invocation.
/// Prebuilt crate/mathlib dependencies are read-only; no `lake build` or artifact cache is used.
pub fn check_generated_modules(directory: &Path, timeout: Duration) -> Result<(), CheckError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| CheckError::Environment("timeout is too large".into()))?;
    let directory = fs::canonicalize(directory).map_err(io_error("resolve generated directory"))?;
    let modules = local_imports(&directory)?;
    let crate_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let runtime = crate_path.join("../runtime/lean");
    let extra_paths = [
        crate_path.join("../ir-core/lean/.lake/build/lib/lean"),
        crate_path.join("lean/.lake/build/lib/lean"),
        crate_path.join("../bgg/lean/.lake/build/lib/lean"),
    ];
    let extra = std::env::join_paths(&extra_paths)
        .map_err(|error| CheckError::Environment(error.to_string()))?;
    // Ask Lake for the pinned runtime package environment, without inheriting candidate paths.
    // Invoke the resolved Lean executable directly later, so timeout kills Lean, not a wrapper.
    let mut environment = Command::new("lake");
    environment.current_dir(&runtime).env("LEAN_PATH", extra).args([
        "env",
        "printenv",
        "LEAN_PATH",
        "PATH",
    ]);
    let mut setup =
        start(&mut environment, "environment".into(), directory.join("lean-environment.log"))?;
    while !poll(&mut setup, deadline)? {
        thread::sleep(Duration::from_millis(10));
    }
    let text = fs::read_to_string(&setup.log).map_err(io_error("read Lean environment"))?;
    let mut lines = text.lines();
    let library_line =
        lines.next().ok_or_else(|| CheckError::Environment("missing LEAN_PATH".into()))?;
    let executable_line =
        lines.next().ok_or_else(|| CheckError::Environment("missing PATH".into()))?;
    if lines.next().is_some() {
        return Err(CheckError::Environment(format!(
            "unexpected Lake output; see {}",
            setup.log.display()
        )));
    }
    let libraries: Vec<_> = std::env::split_paths(library_line).collect();
    validate_dependencies(&modules, &libraries)?;
    let lean = std::env::split_paths(executable_line)
        .map(|path| path.join("lean"))
        .find(|path| path.is_file())
        .ok_or_else(|| CheckError::Environment("Lean executable missing from Lake PATH".into()))?;
    let mut paths = vec![directory.clone()];
    paths.extend(libraries);
    let lean_path: OsString =
        std::env::join_paths(paths).map_err(|error| CheckError::Environment(error.to_string()))?;
    // Three parallel checkers match the completed generated-DAG validation on this host.
    // Each writes only its own module output/log; shared dependencies are never rebuilt.
    let concurrency = thread::available_parallelism().map_or(1, |count| count.get().min(3));
    let mut pending: BTreeSet<_> = modules.keys().cloned().collect();
    let mut completed = BTreeSet::new();
    let mut running = Vec::<Running>::new();
    while !pending.is_empty() || !running.is_empty() {
        if Instant::now() >= deadline {
            return Err(CheckError::Timeout {
                module: "generated DAG".into(),
                log: directory.join("lean-environment.log"),
            });
        }
        let ready: Vec<_> = pending
            .iter()
            .filter(|name| {
                modules[*name].iter().all(|dependency| {
                    !modules.contains_key(dependency) || completed.contains(dependency)
                })
            })
            .take(concurrency - running.len())
            .cloned()
            .collect();
        for name in ready {
            let mut command = Command::new(&lean);
            command
                .current_dir(&runtime)
                .env("LEAN_PATH", &lean_path)
                .arg("-R")
                .arg(&directory)
                .arg("-o")
                .arg(directory.join(format!("{name}.olean")))
                .arg(directory.join(format!("{name}.lean")));
            running.push(start(&mut command, name.clone(), directory.join(format!("{name}.log")))?);
            pending.remove(&name);
        }
        let mut index = 0;
        while index < running.len() {
            if poll(&mut running[index], deadline)? {
                let finished = running.swap_remove(index);
                completed.insert(finished.module.clone());
            } else {
                index += 1;
            }
        }
        if !running.is_empty() {
            thread::sleep(Duration::from_millis(10));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        if let Err(error) = check_generated_modules(directory.path(), Duration::from_secs(30)) {
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
            check_generated_modules(directory.path(), Duration::from_secs(30)),
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
            check_generated_modules(directory.path(), Duration::from_secs(1)),
            Err(CheckError::Timeout { .. })
        ));
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
