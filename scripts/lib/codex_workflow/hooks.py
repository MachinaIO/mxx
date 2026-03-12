from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text
from .paths import RepoPaths
from .plan import (
    FOLLOW_UP_SUBTASKS_HEADING,
    ORDERED_SUBTASKS_HEADING,
    analyze_plan,
    append_follow_up_subtasks_file,
    render_session_plan,
)
from .runners import CodexExecRunner, FinalTestRunner, ShellFinalTestRunner, StructuredExecRunner
from .state import ReviewSnapshot, SessionState, load_state, save_state, write_current_session_id
from .transcript import latest_assistant_message_from_transcript, latest_user_message_from_transcript


@dataclass(frozen=True)
class HookOutcome:
    exit_code: int
    stdout_payload: dict[str, object] | None = None
    stderr_message: str | None = None


def stop_outcome(reason: str) -> HookOutcome:
    return HookOutcome(exit_code=0, stdout_payload={"continue": False, "stopReason": reason})


def block_outcome(message: str) -> HookOutcome:
    return HookOutcome(exit_code=2, stderr_message=message)


def _find_string_field(payload: object, candidate_keys: tuple[str, ...]) -> str | None:
    if isinstance(payload, dict):
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in payload.values():
            found = _find_string_field(value, candidate_keys)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_string_field(item, candidate_keys)
            if found:
                return found
    return None


def extract_session_id(payload: dict[str, Any], paths: RepoPaths) -> str | None:
    session_id = _find_string_field(payload, ("session_id", "sessionId"))
    if session_id:
        return session_id
    try:
        pointer = paths.current_session_id_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return pointer or None


def extract_latest_user_message(payload: dict[str, Any]) -> str | None:
    direct_message = _find_string_field(
        payload,
        ("latest_user_message", "latestUserMessage", "last_user_message", "lastUserMessage"),
    )
    if direct_message:
        return direct_message
    transcript_path = _find_string_field(payload, ("transcript_path", "transcriptPath"))
    if not transcript_path:
        return None
    return latest_user_message_from_transcript(Path(transcript_path))


def extract_latest_assistant_message(payload: dict[str, Any]) -> str | None:
    direct_message = _find_string_field(
        payload,
        ("latest_assistant_message", "latestAssistantMessage", "last_assistant_message", "lastAssistantMessage"),
    )
    if direct_message:
        return direct_message
    transcript_path = _find_string_field(payload, ("transcript_path", "transcriptPath"))
    if not transcript_path:
        return None
    return latest_assistant_message_from_transcript(Path(transcript_path))


def _ensure_plan_exists(paths: RepoPaths, session_id: str, state: SessionState) -> Path:
    plan_path = paths.resolve_plan_path(state.plan_doc)
    if not plan_path.exists():
        atomic_write_text(plan_path, render_session_plan(session_id))
    return plan_path


def _compact_message(text: str, max_len: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _record_review(result: str, msg: str | None) -> ReviewSnapshot:
    from .state import utc_now_rfc3339

    return ReviewSnapshot(result=result, msg=msg, at=utc_now_rfc3339())


def _make_test_follow_ups(summary: str) -> list[str]:
    detail = _compact_message(summary)
    return [f"Fix the failing final validation from `scripts/run_tests.sh`: {detail}"]


def _make_review_follow_ups(summary: str) -> list[str]:
    detail = _compact_message(summary)
    return [f"Address reviewer feedback from the final read-only Codex review: {detail}"]


def _planning_accept_block_message() -> str:
    return (
        "Plan approval recorded. Switch to implementation immediately, read BUILDER.md, "
        "read the session plan, and start from the first unchecked subtask now."
    )


def _planning_revision_block_message(message: str | None) -> str:
    if message:
        return (
            "Plan revisions are still required. Update the session plan before implementing any code. "
            f"Requested revisions: {message}"
        )
    return (
        "Plan revisions are still required. Update the session plan before implementing any code."
    )


def _recovered_state_block_message() -> str:
    return (
        "Session workflow state was missing or malformed and has been recovered to planning mode. "
        "Read BUILDER.md, review the session plan, and request an explicit ACCEPT or concrete plan revisions."
    )


def _missing_plan_structure_message() -> str:
    return (
        "The session plan is missing required checkbox sections. Restore "
        f"`{ORDERED_SUBTASKS_HEADING}` and `{FOLLOW_UP_SUBTASKS_HEADING}` before trying to finish."
    )


def handle_session_start(payload: dict[str, Any], repo_root: Path) -> HookOutcome:
    paths = RepoPaths(repo_root)
    paths.ensure_directories()
    session_id = extract_session_id(payload, paths)
    if not session_id:
        return HookOutcome(exit_code=1, stderr_message="SessionStart hook could not determine the current session id.")

    write_current_session_id(paths, session_id)
    state_result = load_state(paths, session_id)
    state = state_result.state
    if not state_result.existed or state_result.recovered:
        save_state(paths, state)
    _ensure_plan_exists(paths, session_id, state)
    return HookOutcome(exit_code=0)


def _handle_planning_stop(
    payload: dict[str, Any],
    paths: RepoPaths,
    session_id: str,
    state: SessionState,
    exec_runner: StructuredExecRunner,
) -> HookOutcome:
    if not state.awaiting_plan_reply:
        state.awaiting_plan_reply = True
        save_state(paths, state)
        return HookOutcome(exit_code=0)

    state.awaiting_plan_reply = False
    latest_user_message = extract_latest_user_message(payload)

    schema_path = paths.schemas_dir / "plan-approval.schema.json"
    prompt = (
        "You are classifying a user's response to a proposed session plan.\n"
        "The user was asked whether they approve the plan or want revisions.\n\n"
        f"User response: {latest_user_message or '(no message provided)'}\n\n"
        "If the user approves or accepts the plan, output is_approve=true.\n"
        "If the user requests changes or provides feedback, "
        "output is_approve=false and msg=<concise summary of requested revisions>.\n"
        "If there is no clear user message, output is_approve=false "
        "and msg='No user response was provided.'\n"
    )
    classify_result = exec_runner.run(prompt=prompt, schema_path=schema_path, label="plan-classify")

    if not classify_result.ok or classify_result.payload is None:
        state.last_plan_check = _record_review("revision", None)
        save_state(paths, state)
        return block_outcome(_planning_revision_block_message(
            "Plan approval classification could not complete. "
            "Treat user feedback as revision request."
        ))

    is_approve = classify_result.payload.get("is_approve")
    msg = classify_result.payload.get("msg")
    if not isinstance(msg, str):
        msg = latest_user_message

    if is_approve is True:
        state.last_plan_check = _record_review("accept", latest_user_message)
        state.phase = "implementation"
        save_state(paths, state)
        return block_outcome(_planning_accept_block_message())

    state.last_plan_check = _record_review("revision", msg)
    save_state(paths, state)
    return block_outcome(_planning_revision_block_message(msg))


def _handle_implementation_stop(
    paths: RepoPaths,
    session_id: str,
    state: SessionState,
    exec_runner: StructuredExecRunner,
    test_runner: FinalTestRunner,
) -> HookOutcome:
    plan_path = _ensure_plan_exists(paths, session_id, state)
    analysis = analyze_plan(plan_path.read_text(encoding="utf-8"))
    if analysis.missing_sections or analysis.empty_sections:
        return block_outcome(_missing_plan_structure_message())

    if not analysis.all_checked:
        next_item = analysis.first_unchecked
        detail = f" Next unchecked subtask: {next_item.text}" if next_item else ""
        return block_outcome(
            "Implementation is not complete. Continue with the next unchecked subtask and keep the plan document updated."
            + detail
        )

    test_result = test_runner.run(label="final-tests")
    if not test_result.ok:
        follow_up_items = _make_test_follow_ups(test_result.summary)
        append_follow_up_subtasks_file(plan_path, follow_up_items)
        return block_outcome(
            "Final tests failed. New follow-up subtasks were appended to the session plan. "
            f"Address them first. Failure summary: {test_result.summary}"
        )

    schema_path = paths.schemas_dir / "review-decision.schema.json"
    prompt = (
        "You are a read-only reviewer.\n"
        f"The builder session id is {session_id}.\n"
        f"Read the corresponding session plan at {state.plan_doc}.\n"
        "Verify whether the implementation satisfies the goal, constraints, and acceptance criteria in that plan.\n"
        "If acceptable, output result=accept.\n"
        "Otherwise output result=revision and msg=<concrete problems and required fixes>.\n"
    )
    review_result = exec_runner.run(prompt=prompt, schema_path=schema_path, label="final-review")
    if not review_result.ok or review_result.result not in {"accept", "revision"}:
        return block_outcome(
            "The final read-only reviewer could not complete. Stay in implementation mode, perform self-review, "
            "and continue fixing issues before trying again."
        )

    state.last_review = _record_review(review_result.result, review_result.msg)
    if review_result.result == "revision":
        append_follow_up_subtasks_file(plan_path, _make_review_follow_ups(review_result.msg or "Reviewer requested changes."))
        save_state(paths, state)
        return block_outcome(
            "Reviewer requested changes. New follow-up subtasks were appended to the session plan. "
            f"Address them first. Reviewer feedback: {review_result.msg or 'No review message was provided.'}"
        )

    state.completed = True
    state.final_status = "approved"
    save_state(paths, state)
    return stop_outcome("All subtasks are complete, final tests passed, and reviewer approved.")


def handle_stop(
    payload: dict[str, Any],
    repo_root: Path,
    exec_runner: StructuredExecRunner | None = None,
    test_runner: FinalTestRunner | None = None,
) -> HookOutcome:
    paths = RepoPaths(repo_root)
    paths.ensure_directories()
    session_id = extract_session_id(payload, paths)
    if not session_id:
        return block_outcome(
            "Session workflow state is unavailable because the current session id could not be determined."
        )

    write_current_session_id(paths, session_id)
    state_result = load_state(paths, session_id)
    state = state_result.state
    _ensure_plan_exists(paths, session_id, state)

    if not state_result.existed or state_result.recovered:
        save_state(paths, state)
        return block_outcome(_recovered_state_block_message())

    exec_runner = exec_runner or CodexExecRunner(paths, session_id)
    test_runner = test_runner or ShellFinalTestRunner(paths, session_id)

    if state.phase == "planning":
        return _handle_planning_stop(payload, paths, session_id, state, exec_runner)
    return _handle_implementation_stop(paths, session_id, state, exec_runner, test_runner)
