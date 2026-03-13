from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path

    @property
    def agents_dir(self) -> Path:
        return self.repo_root / ".agents"

    @property
    def plans_dir(self) -> Path:
        return self.repo_root / "plans"

    @property
    def tmp_dir(self) -> Path:
        return self.agents_dir / "tmp"

    @property
    def scripts_dir(self) -> Path:
        return self.repo_root / "scripts"

    @property
    def schemas_dir(self) -> Path:
        return self.repo_root / "schemas"

    def default_plan_path(self, session_id: str) -> Path:
        return self.plans_dir / f"session-{session_id}.md"

    def ensure_directories(self) -> None:
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
