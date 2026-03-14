from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .atomic import atomic_write_text

PLAN_APPROVAL_HEADING = "## Plan approval"
PLAN_APPROVAL_VALUES = {"approved", "unapproved"}
ORDERED_SUBTASKS_HEADING = "## Ordered subtasks"
FOLLOW_UP_SUBTASKS_HEADING = "## Follow-up subtasks (append-only)"
CHECKBOX_RE = re.compile(r"^\s*-\s\[(?P<mark>[ xX])\]\s+(?P<text>.+?)\s*$")


@dataclass(frozen=True)
class CheckboxItem:
    section: str
    text: str
    checked: bool
    line_no: int


@dataclass(frozen=True)
class PlanAnalysis:
    approval_status: str | None
    ordered_items: list[CheckboxItem]
    follow_up_items: list[CheckboxItem]
    missing_sections: list[str]
    empty_sections: list[str]
    invalid_sections: list[str]

    @property
    def tracked_items(self) -> list[CheckboxItem]:
        return [*self.ordered_items, *self.follow_up_items]

    @property
    def is_approved(self) -> bool:
        return self.approval_status == "approved"

    @property
    def phase(self) -> str:
        return "implementation" if self.is_approved else "planning"

    @property
    def all_checked(self) -> bool:
        return not self.missing_sections and not self.empty_sections and not self.invalid_sections and all(
            item.checked for item in self.tracked_items
        )

    @property
    def first_unchecked(self) -> CheckboxItem | None:
        for item in self.tracked_items:
            if not item.checked:
                return item
        return None


def utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def render_session_plan(session_id: str, created_at: str | None = None, approval_status: str = "unapproved") -> str:
    if approval_status not in PLAN_APPROVAL_VALUES:
        raise ValueError(f"approval_status must be one of {sorted(PLAN_APPROVAL_VALUES)}")
    stamp = created_at or utc_now_rfc3339()
    return f"""# Session Plan: {session_id}

{PLAN_APPROVAL_HEADING}
{approval_status}

## Goal
Describe the concrete user-visible outcome for this session.

## Constraints
- Preserve completed subtasks as historical record.
- Keep each implementation subtask small enough to complete, debug, and validate within one context window.
- Run the most relevant unit tests immediately after each subtask before checking it off.

## Repo facts / assumptions
- Each workflow hook invocation must provide `session_id` in its JSON payload.
- The stop hook derives planning vs implementation from this plan's approval flag and mechanically checks markdown checkboxes in the required subtask sections.
- The active plan path for this session is `plans/active/session-{session_id}.md`; accepted sessions are archived under `plans/completed/`.

## Acceptance criteria
- The workflow harness uses repository-local Codex hooks.
- Planning transitions to implementation only after plan approval.
- Final completion requires the stop hook to block with actionable resume messages until all tracked checkboxes are checked, final tests pass, and the reviewer approves.

{ORDERED_SUBTASKS_HEADING}
- [ ] Replace this placeholder with the first approved implementation subtask.

{FOLLOW_UP_SUBTASKS_HEADING}
- [x] No follow-up subtasks have been added yet.

## Per-subtask validation
- Record the test command and result immediately after each completed subtask.

## Final validation
- Stop hook blocked with actionable resume messages until all tracked plan checkboxes were checked
- `scripts/run_tests.sh`
- Hooks-disabled read-only reviewer via `codex exec` until acceptance

## Decision log
- {stamp}: Session plan initialized.

## Progress log
- {stamp}: Session plan initialized.
"""


def _find_section_bounds(lines: list[str], heading: str) -> tuple[int, int, int] | None:
    heading_idx = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            heading_idx = index
            break
    if heading_idx is None:
        return None
    start = heading_idx + 1
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return heading_idx, start, end


def _parse_checkboxes(lines: list[str], start: int, end: int, section: str) -> list[CheckboxItem]:
    items: list[CheckboxItem] = []
    for line_no in range(start, end):
        match = CHECKBOX_RE.match(lines[line_no])
        if not match:
            continue
        items.append(
            CheckboxItem(
                section=section,
                text=match.group("text").strip(),
                checked=match.group("mark").lower() == "x",
                line_no=line_no + 1,
            )
        )
    return items


def _parse_plan_approval(lines: list[str], heading: str) -> tuple[str | None, list[str], list[str]]:
    bounds = _find_section_bounds(lines, heading)
    missing_sections: list[str] = []
    invalid_sections: list[str] = []
    if bounds is None:
        missing_sections.append(heading)
        return None, missing_sections, invalid_sections
    _, start, end = bounds
    value: str | None = None
    for line_no in range(start, end):
        candidate = lines[line_no].strip()
        if not candidate:
            continue
        value = candidate
        break
    if value not in PLAN_APPROVAL_VALUES:
        invalid_sections.append(heading)
        return None, missing_sections, invalid_sections
    return value, missing_sections, invalid_sections


def analyze_plan(plan_text: str) -> PlanAnalysis:
    lines = plan_text.splitlines()
    approval_status, missing_sections, invalid_sections = _parse_plan_approval(lines, PLAN_APPROVAL_HEADING)
    ordered_bounds = _find_section_bounds(lines, ORDERED_SUBTASKS_HEADING)
    follow_up_bounds = _find_section_bounds(lines, FOLLOW_UP_SUBTASKS_HEADING)
    empty_sections: list[str] = []
    ordered_items: list[CheckboxItem] = []
    follow_up_items: list[CheckboxItem] = []

    if ordered_bounds is None:
        missing_sections.append(ORDERED_SUBTASKS_HEADING)
    else:
        ordered_items = _parse_checkboxes(lines, ordered_bounds[1], ordered_bounds[2], ORDERED_SUBTASKS_HEADING)
        if not ordered_items:
            empty_sections.append(ORDERED_SUBTASKS_HEADING)

    if follow_up_bounds is None:
        missing_sections.append(FOLLOW_UP_SUBTASKS_HEADING)
    else:
        follow_up_items = _parse_checkboxes(
            lines, follow_up_bounds[1], follow_up_bounds[2], FOLLOW_UP_SUBTASKS_HEADING
        )
        if not follow_up_items:
            empty_sections.append(FOLLOW_UP_SUBTASKS_HEADING)

    return PlanAnalysis(
        approval_status=approval_status,
        ordered_items=ordered_items,
        follow_up_items=follow_up_items,
        missing_sections=missing_sections,
        empty_sections=empty_sections,
        invalid_sections=invalid_sections,
    )


def _normalize_item_text(text: str) -> str:
    cleaned = re.sub(r"^\s*-\s+\[[ xX]\]\s*", "", text.strip())
    cleaned = re.sub(r"^\s*-\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _dedupe_key(text: str) -> str:
    return _normalize_item_text(text).rstrip(".").lower()


def append_follow_up_subtasks(plan_text: str, raw_items: list[str]) -> str:
    lines = plan_text.splitlines()
    bounds = _find_section_bounds(lines, FOLLOW_UP_SUBTASKS_HEADING)
    analysis = analyze_plan(plan_text)
    existing_open = {_dedupe_key(item.text) for item in analysis.follow_up_items if not item.checked}
    new_items: list[str] = []
    seen_new: set[str] = set()
    for raw in raw_items:
        cleaned = _normalize_item_text(raw)
        if not cleaned:
            continue
        key = _dedupe_key(cleaned)
        if key in existing_open or key in seen_new:
            continue
        seen_new.add(key)
        new_items.append(cleaned)

    if not new_items:
        return plan_text

    if bounds is None:
        insert_at = len(lines)
        for index, line in enumerate(lines):
            if line.strip() == "## Per-subtask validation":
                insert_at = index
                break
        section_lines = [FOLLOW_UP_SUBTASKS_HEADING, *[f"- [ ] {item}" for item in new_items], ""]
        if insert_at < len(lines) and insert_at > 0 and lines[insert_at - 1].strip():
            section_lines.insert(0, "")
        lines[insert_at:insert_at] = section_lines
    else:
        _, start, end = bounds
        insert_at = end
        while insert_at > start and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines[insert_at:insert_at] = [f"- [ ] {item}" for item in new_items]

    return "\n".join(lines) + "\n"


def append_follow_up_subtasks_file(plan_path: Path, raw_items: list[str]) -> str:
    original = plan_path.read_text(encoding="utf-8")
    updated = append_follow_up_subtasks(original, raw_items)
    if updated != original:
        atomic_write_text(plan_path, updated)
    return updated
