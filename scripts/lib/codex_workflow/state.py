from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .atomic import atomic_write_json, atomic_write_text
from .paths import RepoPaths

PHASES = {"planning", "implementation"}
CHECK_RESULTS = {None, "accept", "revision"}
FINAL_STATUSES = {None, "approved"}


def utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _recovery_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass
class ReviewSnapshot:
    result: str | None
    msg: str | None
    at: str | None

    @classmethod
    def empty(cls) -> "ReviewSnapshot":
        return cls(result=None, msg=None, at=None)

    @classmethod
    def from_dict(cls, field_name: str, payload: object) -> "ReviewSnapshot":
        if not isinstance(payload, dict):
            raise ValueError(f"{field_name} must be an object")
        result = payload.get("result")
        msg = payload.get("msg")
        at = payload.get("at")
        if result not in CHECK_RESULTS:
            raise ValueError(f"{field_name}.result must be one of {sorted(x for x in CHECK_RESULTS if x)} or null")
        if msg is not None and not isinstance(msg, str):
            raise ValueError(f"{field_name}.msg must be a string or null")
        if at is not None and not isinstance(at, str):
            raise ValueError(f"{field_name}.at must be a string or null")
        return cls(result=result, msg=msg, at=at)

    def to_dict(self) -> dict[str, str | None]:
        return {"result": self.result, "msg": self.msg, "at": self.at}


@dataclass
class SessionState:
    version: int
    session_id: str
    phase: str
    awaiting_plan_reply: bool
    completed: bool
    final_status: str | None
    plan_doc: str
    last_plan_check: ReviewSnapshot
    last_review: ReviewSnapshot
    created_at: str
    updated_at: str

    @classmethod
    def build_initial(cls, session_id: str, created_at: str | None = None) -> "SessionState":
        now = created_at or utc_now_rfc3339()
        return cls(
            version=1,
            session_id=session_id,
            phase="planning",
            awaiting_plan_reply=False,
            completed=False,
            final_status=None,
            plan_doc=f"./plans/session-{session_id}.md",
            last_plan_check=ReviewSnapshot.empty(),
            last_review=ReviewSnapshot.empty(),
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def from_dict(cls, payload: object, expected_session_id: str | None = None) -> "SessionState":
        if not isinstance(payload, dict):
            raise ValueError("session state must be an object")
        version = payload.get("version")
        if version != 1:
            raise ValueError("session state version must be 1")
        session_id = _require_string("session_id", payload.get("session_id"))
        if expected_session_id is not None and session_id != expected_session_id:
            raise ValueError("session_id does not match the requested session")
        phase = payload.get("phase")
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {sorted(PHASES)}")
        awaiting_plan_reply = payload.get("awaiting_plan_reply", False)
        if not isinstance(awaiting_plan_reply, bool):
            raise ValueError("awaiting_plan_reply must be a bool")
        completed = _require_bool("completed", payload.get("completed"))
        final_status = payload.get("final_status")
        if final_status not in FINAL_STATUSES:
            raise ValueError("final_status must be null or approved")
        plan_doc = _require_string("plan_doc", payload.get("plan_doc"))
        created_at = _require_string("created_at", payload.get("created_at"))
        updated_at = _require_string("updated_at", payload.get("updated_at"))
        return cls(
            version=version,
            session_id=session_id,
            phase=phase,
            awaiting_plan_reply=awaiting_plan_reply,
            completed=completed,
            final_status=final_status,
            plan_doc=plan_doc,
            last_plan_check=ReviewSnapshot.from_dict("last_plan_check", payload.get("last_plan_check")),
            last_review=ReviewSnapshot.from_dict("last_review", payload.get("last_review")),
            created_at=created_at,
            updated_at=updated_at,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "phase": self.phase,
            "awaiting_plan_reply": self.awaiting_plan_reply,
            "completed": self.completed,
            "final_status": self.final_status,
            "plan_doc": self.plan_doc,
            "last_plan_check": self.last_plan_check.to_dict(),
            "last_review": self.last_review.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class StateLoadResult:
    state: SessionState
    existed: bool
    recovered: bool
    recovery_path: Path | None = None


def write_current_session_id(paths: RepoPaths, session_id: str) -> None:
    atomic_write_text(paths.current_session_id_path, f"{session_id}\n")


def read_current_session_id(paths: RepoPaths) -> str | None:
    try:
        value = paths.current_session_id_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return value or None


def _backup_corrupt_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.corrupt-{_recovery_suffix()}")
    try:
        os.replace(path, backup)
    except OSError:
        return None
    return backup


def load_state(paths: RepoPaths, session_id: str) -> StateLoadResult:
    state_path = paths.session_state_path(session_id)
    if not state_path.exists():
        return StateLoadResult(
            state=SessionState.build_initial(session_id),
            existed=False,
            recovered=False,
            recovery_path=None,
        )
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        state = SessionState.from_dict(raw, expected_session_id=session_id)
        return StateLoadResult(state=state, existed=True, recovered=False, recovery_path=None)
    except (json.JSONDecodeError, OSError, ValueError):
        backup_path = _backup_corrupt_file(state_path)
        return StateLoadResult(
            state=SessionState.build_initial(session_id),
            existed=False,
            recovered=True,
            recovery_path=backup_path,
        )


def save_state(paths: RepoPaths, state: SessionState) -> None:
    paths.ensure_directories()
    state.updated_at = utc_now_rfc3339()
    atomic_write_json(paths.session_state_path(state.session_id), state.to_dict())
