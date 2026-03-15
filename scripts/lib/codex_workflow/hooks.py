from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import sys
import time
from typing import Any, Callable

from repo_validation import edited_paths_from_git

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
    CodexExecRunner,
    FinalTestRunner,
    ShellFinalTestRunner,
    StructuredExecResult,
    StructuredExecRunner,
)
from .transcript import initial_user_message_from_transcript

RETRY_DELAY_SECONDS = 1.0
MAX_INFRA_RETRY_ATTEMPTS = 100


@dataclass(frozen=True)
class HookOutcome:
    exit_code: int
    stdout_payload: dict[str, object] | None = None
    stderr_message: str | None = None


class RetryLimitExceeded(RuntimeError):
    """Raised when repeated nested Codex infra failures should stop the hook."""


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
    plan_path = paths.active_plan_path(session_id)
    if plan_path.exists():
        return plan_path
    reactivated_path = paths.move_completed_plan_to_active(session_id)
    if reactivated_path is not None:
        return reactivated_path
    atomic_write_text(plan_path, render_session_plan(session_id))
    return plan_path


def _find_stop_plan(paths: RepoPaths, session_id: str) -> Path | None:
    active_path = paths.active_plan_path(session_id)
    if active_path.exists():
        return active_path
    return paths.move_completed_plan_to_active(session_id)


def _finalize_completed_session(paths: RepoPaths, session_id: str) -> None:
    paths.move_session_revision_logs_to_completed(session_id)
    paths.move_active_plan_to_completed(session_id)


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


def _raise_retry_limit_exceeded(kind: str, detail: str) -> None:
    raise RetryLimitExceeded(
        "Stop hook aborted after "
        f"{MAX_INFRA_RETRY_ATTEMPTS} failed nested {kind} attempts. Last summary: {detail}"
    )


@dataclass(frozen=True)
class ImplementationCheckResult:
    status: str
    message: str


@dataclass(frozen=True)
class FinalTestSelection:
    run_python: bool
    run_rust: bool

    @property
    def requires_any(self) -> bool:
        return self.run_python or self.run_rust


def _default_edited_paths_provider(repo_root: Path) -> list[str] | None:
    try:
        return edited_paths_from_git(repo_root)
    except RuntimeError:
        return None


def _select_final_tests(edited_paths: list[str]) -> FinalTestSelection:
    run_python = False
    run_rust = False
    for path in edited_paths:
        normalized = PurePosixPath(path)
        if normalized.suffix == ".py":
            run_python = True
        if normalized.name == "Cargo.toml":
            run_rust = True
        if normalized.suffix == ".rs":
            run_rust = True
        if normalized.parts and normalized.parts[0] == "cuda":
            run_rust = True
    return FinalTestSelection(run_python=run_python, run_rust=run_rust)


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


def _build_review_prompt(session_id: str, plan_display_path: str) -> str:
    return (
        "You are a read-only reviewer.\n"
        f"The builder session id is {session_id}.\n"
        f"Read the corresponding session plan at {plan_display_path}.\n"
        "Verify whether the implementation satisfies the goal, constraints, and acceptance criteria in that plan.\n"
        "If acceptable, output result=accept.\n"
        "Otherwise output result=revision and msg=<concrete problems and required fixes>.\n"
    )


def _run_review_with_retry(
    paths: RepoPaths,
    exec_runner: StructuredExecRunner,
    prompt: str,
    sleep_fn: Callable[[float], None],
) -> StructuredExecResult:
    schema_path = paths.schemas_dir / "review-decision.schema.json"
    for attempt in range(1, MAX_INFRA_RETRY_ATTEMPTS + 1):
        review_result = exec_runner.run(prompt=prompt, schema_path=schema_path, label="final-review")
        if review_result.ok and review_result.result in {"accept", "revision"}:
            return review_result
        detail = review_result.error or "Reviewer did not return a valid structured decision."
        if attempt == MAX_INFRA_RETRY_ATTEMPTS:
            _raise_retry_limit_exceeded("reviewer", detail)
        _log_progress(
            "[stop hook] Nested reviewer failed. "
            f"Attempt {attempt}/{MAX_INFRA_RETRY_ATTEMPTS}. "
            f"Retrying in {RETRY_DELAY_SECONDS:.1f}s. Summary: {detail}"
        )
        sleep_fn(RETRY_DELAY_SECONDS)
    _raise_retry_limit_exceeded("reviewer", "Retry loop exited unexpectedly.")


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
    edited_paths_provider: Callable[[Path], list[str] | None],
) -> ImplementationCheckResult:
    analysis = analyze_plan(plan_path.read_text(encoding="utf-8"))
    if analysis.missing_sections or analysis.invalid_sections or analysis.empty_sections:
        return ImplementationCheckResult(
            status="blocked",
            message=_missing_plan_structure_message(
                analysis.missing_sections,
                analysis.invalid_sections,
                analysis.empty_sections,
            ),
        )

    if not analysis.is_approved:
        return ImplementationCheckResult(
            status="blocked",
            message=(
                "The session plan is still unapproved. Continue the planning interview and do not start implementation "
                "until the plan approval flag is `approved`."
            ),
        )

    if not analysis.all_checked:
        next_item = analysis.first_unchecked
        detail = f" Next unchecked subtask: {next_item.text}" if next_item else ""
        return ImplementationCheckResult(
            status="blocked",
            message=(
                "Implementation is not complete. Continue with the next unchecked subtask and keep the plan document updated."
                + detail
            ),
        )

    edited_paths = edited_paths_provider(paths.repo_root)
    if edited_paths == []:
        return ImplementationCheckResult(
            status="accepted",
            message="All subtasks are complete and no files changed, so final tests and reviewer checks were skipped.",
        )

    tests_were_run = False
    test_selection = (
        FinalTestSelection(run_python=True, run_rust=True)
        if edited_paths is None
        else _select_final_tests(edited_paths)
    )
    if test_selection.requires_any:
        tests_were_run = True
        test_result = test_runner.run(
            label="final-tests",
            run_python=test_selection.run_python,
            run_rust=test_selection.run_rust,
        )
        if not test_result.ok:
            follow_up_items = _make_test_follow_ups(test_result.summary)
            append_follow_up_subtasks_file(plan_path, follow_up_items)
            return ImplementationCheckResult(
                status="blocked",
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
            status="blocked",
            message=(
                "Reviewer requested changes. New follow-up subtasks were appended to the session plan. "
                f"Address them first. Reviewer feedback: {review_result.msg or 'No review message was provided.'}"
            ),
        )

    return ImplementationCheckResult(
        status="accepted",
        message=(
            "All subtasks are complete, final tests passed, and reviewer approved."
            if tests_were_run
            else "All subtasks are complete, final tests were skipped because no Python, Rust, Cargo.toml, or cuda/ files changed, and reviewer approved."
        ),
    )


def _run_implementation_loop(
    paths: RepoPaths,
    session_id: str,
    plan_path: Path,
    plan_display_path: str,
    exec_runner: StructuredExecRunner,
    test_runner: FinalTestRunner,
    sleep_fn: Callable[[float], None],
    edited_paths_provider: Callable[[Path], list[str] | None],
) -> HookOutcome:
    try:
        check_result = _evaluate_implementation_once(
            paths=paths,
            session_id=session_id,
            plan_path=plan_path,
            plan_display_path=plan_display_path,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=sleep_fn,
            edited_paths_provider=edited_paths_provider,
        )
    except RetryLimitExceeded as exc:
        return block_outcome(str(exc))
    if check_result.status == "accepted":
        try:
            _finalize_completed_session(paths, session_id)
        except OSError as exc:
            return block_outcome(f"Failed to archive completed workflow artifacts: {exc}")
        return stop_outcome(check_result.message)
    _log_progress(f"[stop hook] Additional implementation required. {check_result.message}")
    return block_outcome(check_result.message)


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
    test_runner: FinalTestRunner | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    edited_paths_provider: Callable[[Path], list[str] | None] | None = None,
) -> HookOutcome:
    paths = RepoPaths(repo_root)
    paths.ensure_directories()
    session_id = extract_session_id(payload)
    if not session_id:
        return block_outcome("Stop hook requires `session_id` in the hook payload.")

    plan_path = _find_stop_plan(paths, session_id)
    if plan_path is None:
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
    test_runner = test_runner or ShellFinalTestRunner(paths, session_id)
    sleep_fn = sleep_fn or time.sleep
    edited_paths_provider = edited_paths_provider or _default_edited_paths_provider
    return _run_implementation_loop(
        paths=paths,
        session_id=session_id,
        plan_path=plan_path,
        plan_display_path=plan_display_path,
        exec_runner=exec_runner,
        test_runner=test_runner,
        sleep_fn=sleep_fn,
        edited_paths_provider=edited_paths_provider,
    )
