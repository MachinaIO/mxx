from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            os.chmod(path, mode)
        except PermissionError:
            pass
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_write_json(path: Path, payload: Any, mode: int = 0o644) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False) + "\n", mode=mode)
