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
    def current_session_id_path(self) -> Path:
        return self.agents_dir / "current-session-id"

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

    def session_state_path(self, session_id: str) -> Path:
        return self.agents_dir / f"session-{session_id}.json"

    def default_plan_path(self, session_id: str) -> Path:
        return self.plans_dir / f"session-{session_id}.md"

    def resolve_plan_path(self, plan_doc: str) -> Path:
        plan_path = Path(plan_doc)
        if plan_path.is_absolute():
            return plan_path
        return self.repo_root / plan_path

    def ensure_directories(self) -> None:
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
