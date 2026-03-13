from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO

from .paths import RepoPaths


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label.lower())
    return cleaned.strip("-") or "run"


def _summarize_text(text: str, max_lines: int = 6) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-max_lines:])


def summarize_logs(stdout_path: Path | None, stderr_path: Path | None) -> str:
    stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path and stderr_path.exists() else ""
    stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path and stdout_path.exists() else ""
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


@dataclass(frozen=True)
class BuilderExecResult:
    ok: bool
    summary: str
    returncode: int
    thread_id: str | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None


class StructuredExecRunner(Protocol):
    def run(self, prompt: str, schema_path: Path, label: str) -> StructuredExecResult:
        ...


class FinalTestRunner(Protocol):
    def run(self, label: str) -> FinalTestResult:
        ...


class BuilderExecRunner(Protocol):
    def run(self, prompt: str, label: str) -> BuilderExecResult:
        ...


def _pump_stream(
    pipe: Any,
    log_handle: TextIO,
    mirror_stream: TextIO | None,
    line_handler: Callable[[str], None] | None = None,
) -> None:
    try:
        for chunk in iter(pipe.readline, ""):
            log_handle.write(chunk)
            log_handle.flush()
            if line_handler is not None:
                line_handler(chunk)
            if mirror_stream is not None:
                mirror_stream.write(chunk)
                mirror_stream.flush()
    finally:
        pipe.close()


def _run_streaming_subprocess(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    mirror_stream: TextIO | None = None,
    stdout_line_handler: Callable[[str], None] | None = None,
    stderr_line_handler: Callable[[str], None] | None = None,
) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=_pump_stream,
            args=(process.stdout, stdout_handle, mirror_stream, stdout_line_handler),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_pump_stream,
            args=(process.stderr, stderr_handle, mirror_stream, stderr_line_handler),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        return returncode


def _extract_thread_id_from_event_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "thread.started":
        return None
    thread_id = payload.get("thread_id")
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id
    return None


class CodexExecRunner:
    def __init__(self, paths: RepoPaths, session_id: str) -> None:
        self.paths = paths
        self.session_id = session_id

    def run(self, prompt: str, schema_path: Path, label: str) -> StructuredExecResult:
        run_id = uuid.uuid4().hex
        safe_label = _sanitize_label(label)
        stdout_path = self.paths.tmp_dir / f"{self.session_id}-{safe_label}-{run_id}.stdout.log"
        stderr_path = self.paths.tmp_dir / f"{self.session_id}-{safe_label}-{run_id}.stderr.log"
        output_path = self.paths.tmp_dir / f"{self.session_id}-{safe_label}-{run_id}.json"
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


class CodexBuilderRunner:
    def __init__(self, paths: RepoPaths, session_id: str, progress_stream: TextIO | None = None) -> None:
        self.paths = paths
        self.session_id = session_id
        self.progress_stream = progress_stream if progress_stream is not None else sys.stderr

    def run(self, prompt: str, label: str) -> BuilderExecResult:
        run_id = uuid.uuid4().hex
        safe_label = _sanitize_label(label)
        stdout_path = self.paths.tmp_dir / f"{self.session_id}-{safe_label}-{run_id}.stdout.log"
        stderr_path = self.paths.tmp_dir / f"{self.session_id}-{safe_label}-{run_id}.stderr.log"
        observed_thread_id: dict[str, str | None] = {"value": None}

        def capture_stdout(line: str) -> None:
            if observed_thread_id["value"] is not None:
                return
            observed_thread_id["value"] = _extract_thread_id_from_event_line(line)

        command = [
            "codex",
            "exec",
            "resume",
            "--skip-git-repo-check",
            "--disable",
            "codex_hooks",
            "--json",
            self.session_id,
            prompt,
        ]
        returncode = _run_streaming_subprocess(
            command=command,
            cwd=self.paths.repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            mirror_stream=self.progress_stream,
            stdout_line_handler=capture_stdout,
        )
        summary = summarize_logs(stdout_path, stderr_path)
        thread_id = observed_thread_id["value"]
        ok = returncode == 0
        if ok and thread_id is None:
            ok = False
            summary = (
                "Nested builder did not emit a `thread.started` event, so the resumed session id could not be verified."
            )
        elif ok and thread_id != self.session_id:
            ok = False
            summary = (
                f"Nested builder resumed unexpected session id `{thread_id}`; expected `{self.session_id}`."
            )
        return BuilderExecResult(
            ok=ok,
            summary=summary,
            returncode=returncode,
            thread_id=thread_id,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


class ShellFinalTestRunner:
    def __init__(self, paths: RepoPaths, session_id: str) -> None:
        self.paths = paths
        self.session_id = session_id

    def run(self, label: str) -> FinalTestResult:
        run_id = uuid.uuid4().hex
        safe_label = _sanitize_label(label)
        stdout_path = self.paths.tmp_dir / f"{self.session_id}-{safe_label}-{run_id}.stdout.log"
        stderr_path = self.paths.tmp_dir / f"{self.session_id}-{safe_label}-{run_id}.stderr.log"
        command = [str(self.paths.scripts_dir / "run_tests.sh")]
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
