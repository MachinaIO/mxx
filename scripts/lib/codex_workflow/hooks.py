from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Callable

from .atomic import atomic_write_text
from .paths import RepoPaths
from .plan import (
    FOLLOW_UP_SUBTASKS_HEADING,
    ORDERED_SUBTASKS_HEADING,
    PLAN_APPROVAL_HEADING,
    analyze_plan,
    append_follow_up_subtasks_file,
    render_session_plan,
)
from .runners import (
    BuilderExecRunner,
    BuilderExecResult,
    CodexBuilderRunner,
    CodexExecRunner,
    FinalTestRunner,
    ShellFinalTestRunner,
    StructuredExecResult,
    StructuredExecRunner,
)
from .transcript import initial_user_message_from_transcript

RETRY_DELAY_SECONDS = 1.0


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


def extract_session_id(payload: dict[str, Any]) -> str | None:
    return _find_string_field(payload, ("session_id", "sessionId"))


def extract_initial_user_message(payload: dict[str, Any]) -> str | None:
    direct_message = _find_string_field(
        payload,
        ("initial_user_message", "initialUserMessage", "first_user_message", "firstUserMessage", "user_message", "userMessage"),
    )
    if direct_message:
        return direct_message
    transcript_path = _find_string_field(payload, ("transcript_path", "transcriptPath"))
    if not transcript_path:
        return None
    return initial_user_message_from_transcript(Path(transcript_path))


def _ensure_plan_exists(paths: RepoPaths, session_id: str) -> Path:
    plan_path = paths.default_plan_path(session_id)
    if not plan_path.exists():
        atomic_write_text(plan_path, render_session_plan(session_id))
    return plan_path


def _compact_message(text: str, max_len: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _make_test_follow_ups(summary: str) -> list[str]:
    detail = _compact_message(summary)
    return [f"Fix the failing final validation from `scripts/run_tests.sh`: {detail}"]


def _make_review_follow_ups(summary: str) -> list[str]:
    detail = _compact_message(summary)
    return [f"Address reviewer feedback from the final read-only Codex review: {detail}"]


def _missing_plan_structure_message(
    missing_sections: list[str],
    invalid_sections: list[str],
    empty_sections: list[str] | None = None,
) -> str:
    missing_text = ", ".join(f"`{section}`" for section in missing_sections)
    invalid_text = ", ".join(f"`{section}`" for section in invalid_sections)
    empty_text = ", ".join(f"`{section}`" for section in (empty_sections or []))
    details: list[str] = []
    if missing_text:
        details.append(f"missing: {missing_text}")
    if invalid_text:
        details.append(f"invalid: {invalid_text}")
    if empty_text:
        details.append(f"empty: {empty_text}")
    detail_text = "; ".join(details)
    return (
        "The session plan is missing required machine-checkable structure. Restore "
        f"`{PLAN_APPROVAL_HEADING}`, `{ORDERED_SUBTASKS_HEADING}`, and `{FOLLOW_UP_SUBTASKS_HEADING}`. "
        f"Current issues: {detail_text}"
    )


def _log_progress(message: str) -> None:
    sys.stderr.write(message)
    if not message.endswith("\n"):
        sys.stderr.write("\n")
    sys.stderr.flush()


@dataclass(frozen=True)
class ImplementationCheckResult:
    status: str
    message: str


def _build_session_start_prompt(initial_user_message: str) -> str:
    return (
        "You are classifying the user's initial prompt for a Codex workflow session.\n"
        "Return `result=review` only when the user explicitly asks for a review, code review, audit, or inspection of "
        "existing work.\n"
        "Return `result=build` for everything else, including implementation requests, bug fixes, explanations, "
        "brainstorming, or ambiguous prompts.\n"
        "Set `msg` to a short reason.\n"
        "\n"
        "User prompt:\n"
        f"{initial_user_message}\n"
    )


def _build_builder_prompt(session_id: str, plan_display_path: str, reason: str | None) -> str:
    prompt = (
        "You are the builder for the current Codex workflow session.\n"
        f"Session id: {session_id}.\n"
        f"Read `BUILDER.md` and `{plan_display_path}` before acting.\n"
        "Use the explicit session id in this prompt; do not infer it from repository-local pointer files.\n"
        "Treat the session plan as the only source of task scope and completion criteria.\n"
        "Work through unchecked subtasks in order.\n"
        "After each completed subtask, run the most relevant tests immediately and only then check it off.\n"
        "Update the session plan's per-subtask validation, decision log, and progress log as you work.\n"
        "If final tests or reviewer feedback create new obligations, append NEW unchecked items under "
        f"`{FOLLOW_UP_SUBTASKS_HEADING}` instead of rewriting completed history.\n"
        "Stop only when every tracked checkbox in the required subtask sections is checked.\n"
    )
    if reason:
        prompt += f"\nImmediate priority from the outer stop hook: {reason}\n"
    return prompt


def _build_review_prompt(session_id: str, plan_display_path: str) -> str:
    return (
        "You are a read-only reviewer.\n"
        f"The builder session id is {session_id}.\n"
        f"Read the corresponding session plan at {plan_display_path}.\n"
        "Verify whether the implementation satisfies the goal, constraints, and acceptance criteria in that plan.\n"
        "If acceptable, output result=accept.\n"
        "Otherwise output result=revision and msg=<concrete problems and required fixes>.\n"
    )


def _run_builder_with_retry(
    builder_runner: BuilderExecRunner,
    prompt: str,
    label: str,
    sleep_fn: Callable[[float], None],
) -> BuilderExecResult:
    while True:
        result = builder_runner.run(prompt=prompt, label=label)
        if result.ok:
            return result
        _log_progress(
            "[stop hook] Nested builder failed. "
            f"Retrying in {RETRY_DELAY_SECONDS:.1f}s. Summary: {result.summary}"
        )
        sleep_fn(RETRY_DELAY_SECONDS)


def _run_review_with_retry(
    paths: RepoPaths,
    exec_runner: StructuredExecRunner,
    prompt: str,
    sleep_fn: Callable[[float], None],
) -> StructuredExecResult:
    schema_path = paths.schemas_dir / "review-decision.schema.json"
    while True:
        review_result = exec_runner.run(prompt=prompt, schema_path=schema_path, label="final-review")
        if review_result.ok and review_result.result in {"accept", "revision"}:
            return review_result
        detail = review_result.error or "Reviewer did not return a valid structured decision."
        _log_progress(
            "[stop hook] Nested reviewer failed. "
            f"Retrying in {RETRY_DELAY_SECONDS:.1f}s. Summary: {detail}"
        )
        sleep_fn(RETRY_DELAY_SECONDS)


def _classify_session_start_intent(
    paths: RepoPaths,
    session_id: str,
    initial_user_message: str,
    exec_runner: StructuredExecRunner,
) -> StructuredExecResult:
    schema_path = paths.schemas_dir / "session-start-intent.schema.json"
    return exec_runner.run(
        prompt=_build_session_start_prompt(initial_user_message),
        schema_path=schema_path,
        label="session-start-intent",
    )


def _evaluate_implementation_once(
    paths: RepoPaths,
    session_id: str,
    plan_path: Path,
    plan_display_path: str,
    exec_runner: StructuredExecRunner,
    test_runner: FinalTestRunner,
    sleep_fn: Callable[[float], None],
) -> ImplementationCheckResult:
    analysis = analyze_plan(plan_path.read_text(encoding="utf-8"))
    if analysis.missing_sections or analysis.invalid_sections or analysis.empty_sections:
        return ImplementationCheckResult(
            status="builder",
            message=_missing_plan_structure_message(
                analysis.missing_sections,
                analysis.invalid_sections,
                analysis.empty_sections,
            ),
        )

    if not analysis.is_approved:
        return ImplementationCheckResult(
            status="builder",
            message=(
                "The session plan is still unapproved. Continue the planning interview and do not start implementation "
                "until the plan approval flag is `approved`."
            ),
        )

    if not analysis.all_checked:
        next_item = analysis.first_unchecked
        detail = f" Next unchecked subtask: {next_item.text}" if next_item else ""
        return ImplementationCheckResult(
            status="builder",
            message=(
                "Implementation is not complete. Continue with the next unchecked subtask and keep the plan document updated."
                + detail
            ),
        )

    test_result = test_runner.run(label="final-tests")
    if not test_result.ok:
        follow_up_items = _make_test_follow_ups(test_result.summary)
        append_follow_up_subtasks_file(plan_path, follow_up_items)
        return ImplementationCheckResult(
            status="builder",
            message=(
                "Final tests failed. New follow-up subtasks were appended to the session plan. "
                f"Address them first. Failure summary: {test_result.summary}"
            ),
        )

    review_result = _run_review_with_retry(
        paths=paths,
        exec_runner=exec_runner,
        prompt=_build_review_prompt(session_id, plan_display_path),
        sleep_fn=sleep_fn,
    )
    if review_result.result == "revision":
        append_follow_up_subtasks_file(
            plan_path,
            _make_review_follow_ups(review_result.msg or "Reviewer requested changes."),
        )
        return ImplementationCheckResult(
            status="builder",
            message=(
                "Reviewer requested changes. New follow-up subtasks were appended to the session plan. "
                f"Address them first. Reviewer feedback: {review_result.msg or 'No review message was provided.'}"
            ),
        )

    return ImplementationCheckResult(
        status="accepted",
        message="All subtasks are complete, final tests passed, and reviewer approved.",
    )


def _run_implementation_loop(
    paths: RepoPaths,
    session_id: str,
    plan_path: Path,
    plan_display_path: str,
    exec_runner: StructuredExecRunner,
    builder_runner: BuilderExecRunner,
    test_runner: FinalTestRunner,
    sleep_fn: Callable[[float], None],
) -> HookOutcome:
    builder_reason = "Complete the remaining unchecked work in the session plan."
    attempt = 0
    while True:
        check_result = _evaluate_implementation_once(
            paths=paths,
            session_id=session_id,
            plan_path=plan_path,
            plan_display_path=plan_display_path,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=sleep_fn,
        )
        if check_result.status == "accepted":
            return stop_outcome(check_result.message)
        builder_reason = check_result.message
        _log_progress(f"[stop hook] Additional implementation required. {builder_reason}")

        attempt += 1
        _log_progress(f"[stop hook] Launching nested builder attempt {attempt} for session {session_id}.")
        _run_builder_with_retry(
            builder_runner=builder_runner,
            prompt=_build_builder_prompt(session_id, plan_display_path, builder_reason),
            label=f"builder-attempt-{attempt}",
            sleep_fn=sleep_fn,
        )
        _log_progress(f"[stop hook] Nested builder attempt {attempt} finished. Reevaluating completion gate.")


def handle_session_start(
    payload: dict[str, Any],
    repo_root: Path,
    exec_runner: StructuredExecRunner | None = None,
) -> HookOutcome:
    paths = RepoPaths(repo_root)
    paths.ensure_directories()
    session_id = extract_session_id(payload)
    if not session_id:
        return HookOutcome(exit_code=1, stderr_message="SessionStart hook requires `session_id` in the hook payload.")
    initial_user_message = extract_initial_user_message(payload)
    if not initial_user_message:
        return HookOutcome(
            exit_code=1,
            stderr_message="SessionStart hook requires the initial user prompt in the hook payload or transcript.",
        )

    exec_runner = exec_runner or CodexExecRunner(paths, session_id)
    intent_result = _classify_session_start_intent(paths, session_id, initial_user_message, exec_runner)
    if not intent_result.ok:
        detail = intent_result.error or "Intent classifier did not return a valid structured decision."
        return HookOutcome(exit_code=1, stderr_message=f"SessionStart hook could not classify the initial prompt. {detail}")
    if intent_result.result == "review":
        return HookOutcome(exit_code=0)
    if intent_result.result != "build":
        return HookOutcome(
            exit_code=1,
            stderr_message="SessionStart hook classifier returned an unexpected intent result.",
        )

    _ensure_plan_exists(paths, session_id)
    return HookOutcome(exit_code=0)


def handle_stop(
    payload: dict[str, Any],
    repo_root: Path,
    exec_runner: StructuredExecRunner | None = None,
    builder_runner: BuilderExecRunner | None = None,
    test_runner: FinalTestRunner | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> HookOutcome:
    paths = RepoPaths(repo_root)
    paths.ensure_directories()
    session_id = extract_session_id(payload)
    if not session_id:
        return block_outcome("Stop hook requires `session_id` in the hook payload.")

    plan_path = paths.default_plan_path(session_id)
    if not plan_path.exists():
        return HookOutcome(exit_code=0)
    plan_display_path = str(plan_path.relative_to(paths.repo_root))
    analysis = analyze_plan(plan_path.read_text(encoding="utf-8"))
    if analysis.approval_status == "unapproved":
        return HookOutcome(exit_code=0)
    if analysis.missing_sections or analysis.invalid_sections or analysis.empty_sections:
        return block_outcome(
            _missing_plan_structure_message(
                analysis.missing_sections,
                analysis.invalid_sections,
                analysis.empty_sections,
            )
        )

    exec_runner = exec_runner or CodexExecRunner(paths, session_id)
    builder_runner = builder_runner or CodexBuilderRunner(paths, session_id)
    test_runner = test_runner or ShellFinalTestRunner(paths, session_id)
    sleep_fn = sleep_fn or time.sleep
    return _run_implementation_loop(
        paths=paths,
        session_id=session_id,
        plan_path=plan_path,
        plan_display_path=plan_display_path,
        exec_runner=exec_runner,
        builder_runner=builder_runner,
        test_runner=test_runner,
        sleep_fn=sleep_fn,
    )
