from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .paths import RepoPaths


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label.lower())
    return cleaned.strip("-") or "run"


def _summarize_text(text: str, max_lines: int = 6) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-max_lines:])


def _has_failure_signal(text: str) -> bool:
    return any(
        needle in text
        for needle in (
            "test result: FAILED",
            "error: test failed",
            "[gpu-repeat] FAIL",
            "FAILED.",
            "panicked at ",
        )
    )


def summarize_logs(stdout_path: Path | None, stderr_path: Path | None) -> str:
    stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path and stderr_path.exists() else ""
    stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path and stdout_path.exists() else ""
    if _has_failure_signal(stdout_text) and not _has_failure_signal(stderr_text):
        summary = _summarize_text(stdout_text)
        if summary:
            return summary
    summary = _summarize_text(stderr_text)
    if summary:
        return summary
    summary = _summarize_text(stdout_text)
    if summary:
        return summary
    return "No diagnostic output was captured."


@dataclass(frozen=True)
class StructuredExecResult:
    ok: bool
    payload: dict[str, Any] | None = None
    result: str | None = None
    msg: str | None = None
    error: str | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    output_path: Path | None = None


@dataclass(frozen=True)
class FinalTestResult:
    ok: bool
    summary: str
    returncode: int
    stdout_path: Path | None = None
    stderr_path: Path | None = None


class StructuredExecRunner(Protocol):
    def run(self, prompt: str, schema_path: Path, label: str) -> StructuredExecResult:
        ...


class FinalTestRunner(Protocol):
    def run(self, label: str, *, run_python: bool = True, run_rust: bool = True) -> FinalTestResult:
        ...


class CodexExecRunner:
    def __init__(self, paths: RepoPaths, session_id: str) -> None:
        self.paths = paths
        self.session_id = session_id

    def run(self, prompt: str, schema_path: Path, label: str) -> StructuredExecResult:
        run_id = uuid.uuid4().hex
        safe_label = _sanitize_label(label)
        stdout_path = self.paths.active_revision_logs_dir / f"{self.session_id}-{safe_label}-{run_id}.stdout.log"
        stderr_path = self.paths.active_revision_logs_dir / f"{self.session_id}-{safe_label}-{run_id}.stderr.log"
        output_path = self.paths.active_revision_logs_dir / f"{self.session_id}-{safe_label}-{run_id}.json"
        command = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--disable",
            "codex_hooks",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            "--cd",
            str(self.paths.repo_root),
            prompt,
        ]
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=self.paths.repo_root,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            return StructuredExecResult(
                ok=False,
                error=summarize_logs(stdout_path, stderr_path),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_path=output_path,
            )
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            return StructuredExecResult(
                ok=False,
                error=f"Failed to parse structured Codex output: {exc}",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_path=output_path,
            )
        if not isinstance(payload, dict):
            return StructuredExecResult(
                ok=False,
                error="Structured Codex output was not a JSON object.",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_path=output_path,
            )
        result = payload.get("result")
        msg = payload.get("msg")
        if result is not None and not isinstance(result, str):
            result = None
        if msg is not None and not isinstance(msg, str):
            msg = None
        return StructuredExecResult(
            ok=True,
            payload=payload,
            result=result,
            msg=msg,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            output_path=output_path,
        )


class ShellFinalTestRunner:
    def __init__(self, paths: RepoPaths, session_id: str) -> None:
        self.paths = paths
        self.session_id = session_id

    def run(self, label: str, *, run_python: bool = True, run_rust: bool = True) -> FinalTestResult:
        run_id = uuid.uuid4().hex
        safe_label = _sanitize_label(label)
        stdout_path = self.paths.active_revision_logs_dir / f"{self.session_id}-{safe_label}-{run_id}.stdout.log"
        stderr_path = self.paths.active_revision_logs_dir / f"{self.session_id}-{safe_label}-{run_id}.stderr.log"
        command = [str(self.paths.scripts_dir / "run_tests.sh")]
        if run_python and not run_rust:
            command.append("--python")
        elif run_rust and not run_python:
            command.append("--rust")
        elif not run_python and not run_rust:
            raise ValueError("ShellFinalTestRunner.run requires at least one test suite.")
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=self.paths.repo_root,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        summary = summarize_logs(stdout_path, stderr_path)
        return FinalTestResult(
            ok=completed.returncode == 0,
            summary=summary,
            returncode=completed.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
