from __future__ import annotations

import argparse
import json
import os
import shlex
import tomllib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence, TextIO

DEFAULT_GPU_REPEAT_COUNT = 300
EDITED_DIFF_FILTER = "ACDMR"


@dataclass(frozen=True)
class RepeatSummary:
    iterations: int
    failed_iterations: list[int]

    @property
    def failure_count(self) -> int:
        return len(self.failed_iterations)


def edited_paths_from_git(repo_root: Path, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> list[str]:
    runner = runner or subprocess.run
    commands = (
        ("git", "diff", "--name-only", f"--diff-filter={EDITED_DIFF_FILTER}"),
        ("git", "diff", "--cached", "--name-only", f"--diff-filter={EDITED_DIFF_FILTER}"),
        ("git", "ls-files", "--others", "--exclude-standard"),
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for command in commands:
        completed = runner(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Failed to inspect edited files with: {' '.join(command)}")
        for line in completed.stdout.splitlines():
            path = line.strip()
            if path and path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def is_gpu_rust_path(path: str) -> bool:
    normalized = PurePosixPath(path)
    return normalized.suffix == ".rs" and "gpu" in normalized.name.lower()


def is_gpu_repeat_validation_trigger(path: str) -> bool:
    normalized = PurePosixPath(path)
    if normalized.parts[:3] in {("crates", "backends", "gpu"), ("crates", "backends", "cuda"), ("crates", "fhe", "cuda")}:
        return True
    if path in {"crates/backends/build.rs", "crates/fhe/build.rs", "crates/backends/Cargo.toml", "crates/fhe/Cargo.toml"}:
        return True
    return is_gpu_rust_path(path) and normalized.parts[:3] == ("crates", "backends", "src")


def warning_free_rustflags(environment: dict[str, str], repo_root: Path | None = None) -> str:
    """Keep Cargo's effective flags, including the repository OpenFHE rpath."""
    if "CARGO_ENCODED_RUSTFLAGS" in environment:
        flags = environment["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") if environment["CARGO_ENCODED_RUSTFLAGS"] else []
    elif "RUSTFLAGS" in environment:
        flags = shlex.split(environment["RUSTFLAGS"])
    else:
        repo_root = repo_root or Path(__file__).resolve().parents[2]
        with (repo_root / ".cargo/config.toml").open("rb") as config_file:
            config = tomllib.load(config_file)
        # This is the repository's configured host target rule, not a generic
        # reimplementation of Cargo's target/cfg resolution.
        flags = config.get("target", {}).get('cfg(target_os = "linux")', {}).get("rustflags", []) if sys.platform.startswith("linux") else []
    return "\x1f".join([*flags, "-D", "warnings"])


def gpu_validation_environment(environment: dict[str, str]) -> dict[str, str]:
    """Select one native backend without requiring an SDK for CPU validation."""
    env = environment.copy()
    backend = env.get("MXX_GPU_BACKEND", "cuda")
    if backend not in {"cuda", "hip"}:
        raise ValueError(f"Unsupported MXX_GPU_BACKEND: {backend!r}; expected cuda or hip")
    env["MXX_GPU_BACKEND"] = backend
    env.setdefault("CARGO_TARGET_DIR", f"target/gpu-{backend}")
    env.setdefault("RUST_LOG", "debug")
    env["CARGO_ENCODED_RUSTFLAGS"] = warning_free_rustflags(env)
    return env


def is_gpu_single_run_validation_trigger(path: str) -> bool:
    normalized = PurePosixPath(path)
    return (
        is_gpu_rust_path(path)
        and not is_gpu_repeat_validation_trigger(path)
        and len(normalized.parts) >= 4
        and normalized.parts[0] == "crates"
        and normalized.parts[2] == "src"
    )


def gpu_repeat_validation_trigger_paths(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if is_gpu_repeat_validation_trigger(path)]


def gpu_single_run_validation_trigger_paths(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if is_gpu_single_run_validation_trigger(path)]


def parse_cargo_test_executables(stdout_text: str) -> list[Path]:
    executables: list[Path] = []
    seen: set[Path] = set()
    for line in stdout_text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("reason") != "compiler-artifact":
            continue
        executable = payload.get("executable")
        target = payload.get("target")
        if not isinstance(executable, str) or not executable:
            continue
        if not isinstance(target, dict) or target.get("test") is not True:
            continue
        path = Path(executable)
        if path not in seen:
            seen.add(path)
            executables.append(path)
    return executables


def compile_gpu_test_binaries(
    repo_root: Path,
    env: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> list[Path]:
    runner = runner or subprocess.run
    completed = runner(
        ("cargo", "test", "gpu", "-r", "--workspace", "--lib", "--features", "gpu", "--no-run", "--message-format=json"),
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError("Failed to compile GPU test binaries.")
    executables = parse_cargo_test_executables(completed.stdout)
    if not executables:
        raise RuntimeError("Cargo did not report any GPU test executables.")
    selected: list[Path] = []
    test_filter = env.get("GPU_TEST_FILTER", "gpu")
    for executable in executables:
        listed = runner(
            (str(executable), test_filter, "--ignored", "--list"),
            cwd=repo_root, env=env, check=False, capture_output=True, text=True,
        )
        if listed.returncode != 0:
            raise RuntimeError(f"Failed to list GPU unit tests in {executable}")
        if any(line.endswith(": test") for line in listed.stdout.splitlines()):
            selected.append(executable)
    if not selected:
        raise RuntimeError(f"No ignored GPU unit tests match {test_filter!r}; refusing an empty device gate")
    return selected


def run_gpu_repeat_suite(
    binaries: Sequence[Path],
    repeat_count: int,
    executor: Callable[[Path], int],
    log: TextIO,
) -> RepeatSummary:
    failed_iterations: list[int] = []
    for iteration in range(1, repeat_count + 1):
        iteration_failed = False
        for binary in binaries:
            returncode = executor(binary)
            if returncode != 0:
                iteration_failed = True
                log.write(
                    f"[gpu-repeat] iteration {iteration}/{repeat_count}: {binary.name} failed with exit code {returncode}\n"
                )
        if iteration_failed:
            failed_iterations.append(iteration)
            log.write(f"[gpu-repeat] iteration {iteration}/{repeat_count}: FAIL\n")
        else:
            log.write(f"[gpu-repeat] iteration {iteration}/{repeat_count}: PASS\n")
    return RepeatSummary(iterations=repeat_count, failed_iterations=failed_iterations)


def run_gpu_binary(binary: Path, repo_root: Path, env: dict[str, str]) -> int:
    completed = subprocess.run(
        (str(binary), env.get("GPU_TEST_FILTER", "gpu"), "--ignored"),
        cwd=repo_root,
        env=env,
        check=False,
    )
    return completed.returncode


def maybe_run_gpu_repeat_validation(repo_root: Path, repeat_count: int, log: TextIO, force: bool = False) -> int:
    edited_paths = edited_paths_from_git(repo_root)
    repeat_trigger_paths = gpu_repeat_validation_trigger_paths(edited_paths)
    single_run_trigger_paths = gpu_single_run_validation_trigger_paths(edited_paths)
    if force:
        repeat_trigger_paths.append("explicit full GPU unit validation")
    if not repeat_trigger_paths and not single_run_trigger_paths:
        log.write(
            "[gpu-repeat] skipped: no edited files under native GPU, GPU build, or matching *gpu*.rs in configured crate source paths\n"
        )
        return 0

    if repeat_count < 1:
        raise ValueError("GPU repeat count must be positive")
    env = gpu_validation_environment(dict(os.environ))
    log.write(f"[gpu-repeat] backend={env['MXX_GPU_BACKEND']} filter={env.get('GPU_TEST_FILTER', 'gpu')} (ignored device unit tests)\n")
    binaries = compile_gpu_test_binaries(repo_root, env)
    if repeat_trigger_paths:
        log.write("[gpu-repeat] repeat mode triggered by edited files:\n")
        for path in repeat_trigger_paths:
            log.write(f"[gpu-repeat]   {path}\n")
        log.write(
            f"[gpu-repeat] compiled {len(binaries)} test binaries once; running {repeat_count} sequential iterations\n"
        )
        summary = run_gpu_repeat_suite(
            binaries=binaries,
            repeat_count=repeat_count,
            executor=lambda binary: run_gpu_binary(binary, repo_root, env),
            log=log,
        )
        if summary.failure_count:
            failed = ", ".join(str(iteration) for iteration in summary.failed_iterations)
            log.write(
                f"[gpu-repeat] FAIL: {summary.failure_count}/{summary.iterations} iterations failed. Failed iterations: {failed}\n"
            )
            return 1

        log.write(f"[gpu-repeat] PASS: all {summary.iterations} iterations succeeded\n")
        return 0

    log.write("[gpu-repeat] single-run mode triggered by edited files:\n")
    for path in single_run_trigger_paths:
        log.write(f"[gpu-repeat]   {path}\n")
    log.write(f"[gpu-repeat] compiled {len(binaries)} test binaries once; running 1 sequential iteration\n")
    summary = run_gpu_repeat_suite(
        binaries=binaries,
        repeat_count=1,
        executor=lambda binary: run_gpu_binary(binary, repo_root, env),
        log=log,
    )
    if summary.failure_count:
        log.write(
            "[gpu-repeat] FAIL: single-run GPU validation failed\n"
        )
        return 1

    log.write("[gpu-repeat] PASS: single-run GPU validation succeeded\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository validation helpers")
    parser.add_argument(
        "command",
        choices=("maybe-run-gpu-repeat", "warning-free-rustflags"),
        help="Run conditional repository validation routines.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=int(os.environ.get("GPU_REPEAT_COUNT", DEFAULT_GPU_REPEAT_COUNT)),
        help="How many sequential GPU iterations to run when GPU-triggering files were edited.",
    )
    parser.add_argument("--force", action="store_true", help="Run the GPU unit suite even with a clean checkout.")
    args = parser.parse_args(argv)
    repo_root = Path.cwd()
    if args.command == "warning-free-rustflags":
        sys.stdout.write(warning_free_rustflags(dict(os.environ), repo_root))
        return 0
    if args.command == "maybe-run-gpu-repeat":
        return maybe_run_gpu_repeat_validation(repo_root, args.repeat_count, sys.stdout, force=args.force)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
