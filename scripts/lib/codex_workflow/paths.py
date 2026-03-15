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
    def active_plans_dir(self) -> Path:
        return self.plans_dir / "active"

    @property
    def completed_plans_dir(self) -> Path:
        return self.plans_dir / "completed"

    @property
    def revision_logs_dir(self) -> Path:
        return self.repo_root / "revision_logs"

    @property
    def active_revision_logs_dir(self) -> Path:
        return self.revision_logs_dir / "active"

    @property
    def completed_revision_logs_dir(self) -> Path:
        return self.revision_logs_dir / "completed"

    @property
    def scripts_dir(self) -> Path:
        return self.repo_root / "scripts"

    @property
    def schemas_dir(self) -> Path:
        return self.repo_root / "schemas"

    def active_plan_path(self, session_id: str) -> Path:
        return self.active_plans_dir / f"session-{session_id}.md"

    def completed_plan_path(self, session_id: str) -> Path:
        return self.completed_plans_dir / f"session-{session_id}.md"

    def default_plan_path(self, session_id: str) -> Path:
        return self.active_plan_path(session_id)

    def active_revision_log_paths(self, session_id: str) -> list[Path]:
        pattern = f"{session_id}-*"
        return sorted(self.active_revision_logs_dir.glob(pattern))

    def move_session_revision_logs_to_completed(self, session_id: str) -> list[Path]:
        moved_paths: list[Path] = []
        for source in self.active_revision_log_paths(session_id):
            destination = self.completed_revision_logs_dir / source.name
            source.replace(destination)
            moved_paths.append(destination)
        return moved_paths

    def move_completed_plan_to_active(self, session_id: str) -> Path | None:
        active_path = self.active_plan_path(session_id)
        if active_path.exists():
            return active_path
        completed_path = self.completed_plan_path(session_id)
        if not completed_path.exists():
            return None
        completed_path.replace(active_path)
        return active_path

    def move_active_plan_to_completed(self, session_id: str) -> Path | None:
        active_path = self.active_plan_path(session_id)
        if not active_path.exists():
            return None
        destination = self.completed_plans_dir / f"session-{session_id}.md"
        active_path.replace(destination)
        return destination

    def ensure_directories(self) -> None:
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.active_plans_dir.mkdir(parents=True, exist_ok=True)
        self.completed_plans_dir.mkdir(parents=True, exist_ok=True)
        self.revision_logs_dir.mkdir(parents=True, exist_ok=True)
        self.active_revision_logs_dir.mkdir(parents=True, exist_ok=True)
        self.completed_revision_logs_dir.mkdir(parents=True, exist_ok=True)
