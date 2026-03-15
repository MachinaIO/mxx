from __future__ import annotations

import argparse
import json
import os
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


def is_gpu_validation_trigger(path: str) -> bool:
    normalized = PurePosixPath(path)
    if normalized.parts and normalized.parts[0] == "cuda":
        return True
    return normalized.suffix == ".rs" and "gpu" in normalized.name.lower()


def gpu_validation_trigger_paths(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if is_gpu_validation_trigger(path)]


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
        ("cargo", "test", "gpu", "-r", "--lib", "--features", "gpu", "--no-run", "--message-format=json"),
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
    return executables


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
        (str(binary), "gpu"),
        cwd=repo_root,
        env=env,
        check=False,
    )
    return completed.returncode


def maybe_run_gpu_repeat_validation(repo_root: Path, repeat_count: int, log: TextIO) -> int:
    edited_paths = edited_paths_from_git(repo_root)
    trigger_paths = gpu_validation_trigger_paths(edited_paths)
    if not trigger_paths:
        log.write("[gpu-repeat] skipped: no edited files under cuda/ or matching *gpu*.rs\n")
        return 0

    log.write("[gpu-repeat] triggered by edited files:\n")
    for path in trigger_paths:
        log.write(f"[gpu-repeat]   {path}\n")

    env = os.environ.copy()
    env.setdefault("RUST_LOG", "debug")
    binaries = compile_gpu_test_binaries(repo_root, env)
    log.write(f"[gpu-repeat] compiled {len(binaries)} test binaries once; running {repeat_count} sequential iterations\n")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository validation helpers")
    parser.add_argument(
        "command",
        choices=("maybe-run-gpu-repeat",),
        help="Run conditional repository validation routines.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=int(os.environ.get("GPU_REPEAT_COUNT", DEFAULT_GPU_REPEAT_COUNT)),
        help="How many sequential GPU iterations to run when GPU-triggering files were edited.",
    )
    args = parser.parse_args(argv)
    repo_root = Path.cwd()
    if args.command == "maybe-run-gpu-repeat":
        return maybe_run_gpu_repeat_validation(repo_root, args.repeat_count, sys.stdout)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
