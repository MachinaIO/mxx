from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .hooks import HookOutcome, handle_session_start, handle_stop


def emit_outcome(outcome: HookOutcome, stdout_stream: Any, stderr_stream: Any) -> int:
    if outcome.stdout_payload is not None:
        stdout_stream.write(json.dumps(outcome.stdout_payload, separators=(",", ":")))
    if outcome.stderr_message:
        stderr_stream.write(outcome.stderr_message)
        if not outcome.stderr_message.endswith("\n"):
            stderr_stream.write("\n")
    return outcome.exit_code


def run_command(command: str, payload: dict[str, Any], repo_root: Path) -> HookOutcome:
    if command == "session-start":
        return handle_session_start(payload, repo_root)
    if command == "stop":
        return handle_stop(payload, repo_root)
    raise ValueError(f"Unsupported workflow command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository-local Codex workflow harness")
    parser.add_argument("command", choices=("session-start", "stop"))
    args = parser.parse_args(argv)
    payload = json.load(sys.stdin)
    repo_root = Path.cwd()
    outcome = run_command(args.command, payload, repo_root)
    return emit_outcome(outcome, sys.stdout, sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
