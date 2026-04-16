from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codex_workflow.atomic import atomic_write_text
from codex_workflow.cli import emit_outcome
from codex_workflow.hooks import MAX_INFRA_RETRY_ATTEMPTS, handle_session_start, handle_stop
from codex_workflow.paths import RepoPaths
from codex_workflow.plan import (
    FOLLOW_UP_SUBTASKS_HEADING,
    ORDERED_SUBTASKS_HEADING,
    PLAN_APPROVAL_HEADING,
    PLAN_PHASE_HEADING,
    analyze_plan,
    append_follow_up_subtasks,
    render_session_plan,
)
from codex_workflow.runners import (
    FinalTestResult,
    ShellFinalTestRunner,
    StructuredExecResult,
    summarize_logs,
)
from codex_workflow.transcript import (
    initial_user_message_from_transcript,
    latest_assistant_message_from_transcript,
    latest_user_message_from_transcript,
)


def build_plan_text(
    session_id: str,
    ordered_items: list[tuple[bool, str]],
    follow_up_items: list[tuple[bool, str]] | None = None,
    approval_status: str = "approved",
    phase: str | None = None,
) -> str:
    if follow_up_items is None:
        follow_up_items = [(True, "No follow-up subtasks have been added yet.")]
    if phase is None:
        phase = "planning" if approval_status == "unapproved" else "implementation"
    ordered_block = "\n".join(f"- [{'x' if checked else ' '}] {text}" for checked, text in ordered_items)
    follow_up_block = "\n".join(f"- [{'x' if checked else ' '}] {text}" for checked, text in follow_up_items)
    return f"""# Session Plan: {session_id}

{PLAN_APPROVAL_HEADING}
{approval_status}

{PLAN_PHASE_HEADING}
{phase}

## Goal
Goal

## Constraints
- Constraint

## Repo facts / assumptions
- Fact

## Acceptance criteria
- Criterion

{ORDERED_SUBTASKS_HEADING}
{ordered_block}

{FOLLOW_UP_SUBTASKS_HEADING}
{follow_up_block}

## Per-subtask validation
- Validation

## Final validation
- Final validation

## Decision log
- Initialized

## Progress log
- Initialized
"""


class FakeExecRunner:
    def __init__(self, *results: StructuredExecResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, Path, str]] = []

    def run(self, prompt: str, schema_path: Path, label: str) -> StructuredExecResult:
        self.calls.append((prompt, schema_path, label))
        return self.results.pop(0)


class FakeTestRunner:
    def __init__(self, *results: FinalTestResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, bool, bool]] = []

    def run(self, label: str, *, run_python: bool = True, run_rust: bool = True) -> FinalTestResult:
        self.calls.append((label, run_python, run_rust))
        return self.results.pop(0)


class WorkflowHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tmpdir.name)
        self.paths = RepoPaths(self.repo_root)
        self.paths.ensure_directories()
        (self.repo_root / "schemas").mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.paths.schemas_dir / "review-decision.schema.json", "{}\n")
        atomic_write_text(self.paths.schemas_dir / "session-start-intent.schema.json", "{}\n")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def write_plan(
        self,
        session_id: str = "session-1",
        ordered_items: list[tuple[bool, str]] | None = None,
        follow_up_items: list[tuple[bool, str]] | None = None,
        approval_status: str = "approved",
        phase: str | None = None,
    ) -> Path:
        if ordered_items is None:
            ordered_items = [(False, "Replace this placeholder with a real subtask.")]
        plan_path = self.paths.default_plan_path(session_id)
        atomic_write_text(
            plan_path,
            build_plan_text(
                session_id,
                ordered_items=ordered_items,
                follow_up_items=follow_up_items,
                approval_status=approval_status,
                phase=phase,
            ),
        )
        return plan_path

    def write_completed_plan(
        self,
        session_id: str = "session-1",
        ordered_items: list[tuple[bool, str]] | None = None,
        follow_up_items: list[tuple[bool, str]] | None = None,
        approval_status: str = "approved",
        phase: str | None = None,
    ) -> Path:
        if ordered_items is None:
            ordered_items = [(False, "Replace this placeholder with a real subtask.")]
        plan_path = self.paths.completed_plan_path(session_id)
        atomic_write_text(
            plan_path,
            build_plan_text(
                session_id,
                ordered_items=ordered_items,
                follow_up_items=follow_up_items,
                approval_status=approval_status,
                phase=phase,
            ),
        )
        return plan_path

    def test_session_start_initializes_plan_for_non_review_requests(self) -> None:
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="build", msg="implementation request"))
        outcome = handle_session_start(
            {"session_id": "abc123", "initial_user_message": "Implement the stop hook changes."},
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 0)
        plan_path = self.paths.default_plan_path("abc123")
        self.assertTrue(plan_path.exists())
        analysis = analyze_plan(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(analysis.approval_status, "unapproved")
        self.assertEqual(analysis.phase, "planning")
        self.assertFalse((self.paths.agents_dir / "current-session-id").exists())
        self.assertEqual(list(self.paths.agents_dir.glob("session-abc123.json")), [])
        self.assertEqual(len(exec_runner.calls), 1)
        prompt, schema_path, label = exec_runner.calls[0]
        self.assertIn("Implement the stop hook changes.", prompt)
        self.assertTrue(str(schema_path).endswith("session-start-intent.schema.json"))
        self.assertEqual(label, "session-start-intent")

    def test_repo_paths_creates_revision_logs_directory(self) -> None:
        self.assertTrue(self.paths.revision_logs_dir.exists())
        self.assertEqual(self.paths.revision_logs_dir, self.repo_root / "revision_logs")
        self.assertTrue(self.paths.active_revision_logs_dir.exists())
        self.assertTrue(self.paths.completed_revision_logs_dir.exists())
        self.assertEqual(self.paths.active_revision_logs_dir, self.repo_root / "revision_logs" / "active")
        self.assertEqual(self.paths.completed_revision_logs_dir, self.repo_root / "revision_logs" / "completed")
        self.assertTrue(self.paths.active_plans_dir.exists())
        self.assertTrue(self.paths.completed_plans_dir.exists())
        self.assertEqual(self.paths.active_plans_dir, self.repo_root / "plans" / "active")
        self.assertEqual(self.paths.completed_plans_dir, self.repo_root / "plans" / "completed")

    def test_session_start_skips_plan_creation_for_explicit_review_requests(self) -> None:
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="review", msg="explicit review request"))
        outcome = handle_session_start(
            {"session_id": "review-123", "initial_user_message": "Please review this patch."},
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertFalse(self.paths.default_plan_path("review-123").exists())
        self.assertEqual(len(exec_runner.calls), 1)

    def test_session_start_is_idempotent_for_existing_plan(self) -> None:
        plan_path = self.write_plan(
            session_id="abc123",
            ordered_items=[(True, "Done")],
            approval_status="approved",
        )
        original = plan_path.read_text(encoding="utf-8")
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="build", msg="implementation request"))

        outcome = handle_session_start(
            {"session_id": "abc123", "initial_user_message": "Continue implementing the approved plan."},
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(plan_path.read_text(encoding="utf-8"), original)

    def test_session_start_requires_session_id_in_payload(self) -> None:
        outcome = handle_session_start({}, self.repo_root)

        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("requires `session_id`", outcome.stderr_message)

    def test_session_start_requires_initial_user_prompt(self) -> None:
        outcome = handle_session_start({"session_id": "abc123"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("initial user prompt", outcome.stderr_message)

    def test_session_start_rejects_latest_user_message_as_initial_prompt(self) -> None:
        outcome = handle_session_start({"session_id": "abc123", "latest_user_message": "Please review this patch."}, self.repo_root)

        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("initial user prompt", outcome.stderr_message)

    def test_session_start_can_extract_initial_prompt_from_transcript(self) -> None:
        transcript_path = self.repo_root / "session-start-transcript.jsonl"
        atomic_write_text(
            transcript_path,
            "\n".join(
                [
                    "{\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"Implement this workflow change.\"}}",
                    "{\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"Please review the implementation.\"}}",
                ]
            )
            + "\n",
        )
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="build", msg="initial prompt is a build request"))

        outcome = handle_session_start(
            {"session_id": "review-transcript", "transcript_path": str(transcript_path)},
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertTrue(self.paths.default_plan_path("review-transcript").exists())
        self.assertEqual(len(exec_runner.calls), 1)
        prompt, _, _ = exec_runner.calls[0]
        self.assertIn("Implement this workflow change.", prompt)
        self.assertNotIn("Please review the implementation.", prompt)

    def test_session_start_blocks_when_intent_classifier_fails(self) -> None:
        exec_runner = FakeExecRunner(StructuredExecResult(ok=False, error="classifier unavailable"))

        outcome = handle_session_start(
            {"session_id": "abc123", "initial_user_message": "Implement the change."},
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("could not classify", outcome.stderr_message)

    def test_review_decision_schema_requires_all_defined_properties(self) -> None:
        schema_path = Path(__file__).resolve().parents[4] / "schemas" / "review-decision.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(set(schema["properties"]), {"result", "msg"})
        self.assertEqual(set(schema["required"]), {"result", "msg"})

    def test_session_start_intent_schema_requires_all_defined_properties(self) -> None:
        schema_path = Path(__file__).resolve().parents[4] / "schemas" / "session-start-intent.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(set(schema["properties"]), {"result", "msg"})
        self.assertEqual(set(schema["required"]), {"result", "msg"})

    def test_planning_stop_allows_stop_without_running_nested_work(self) -> None:
        self.write_plan(session_id="plan-first", approval_status="unapproved")
        exec_runner = FakeExecRunner()
        test_runner = FakeTestRunner()

        outcome = handle_stop(
            {"session_id": "plan-first"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIsNone(outcome.stdout_payload)
        self.assertEqual(exec_runner.calls, [])
        self.assertEqual(test_runner.calls, [])

    def test_planning_stop_allows_stop_even_when_required_checkbox_section_is_empty(self) -> None:
        plan_path = self.write_plan(session_id="planning-empty", approval_status="unapproved")
        broken = plan_path.read_text(encoding="utf-8").replace("- [ ] Replace this placeholder with a real subtask.\n", "", 1)
        atomic_write_text(plan_path, broken)
        exec_runner = FakeExecRunner()
        test_runner = FakeTestRunner()

        outcome = handle_stop(
            {"session_id": "planning-empty"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIsNone(outcome.stdout_payload)
        self.assertEqual(exec_runner.calls, [])
        self.assertEqual(test_runner.calls, [])

    def test_planning_stop_blocks_when_phase_section_is_missing(self) -> None:
        plan_path = self.write_plan(session_id="planning-missing-phase", approval_status="unapproved")
        original = plan_path.read_text(encoding="utf-8")
        broken = original.replace(f"{PLAN_PHASE_HEADING}\nplanning\n\n", "", 1)
        atomic_write_text(plan_path, broken)

        outcome = handle_stop({"session_id": "planning-missing-phase"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn(PLAN_PHASE_HEADING, outcome.stderr_message)

    def test_stop_requires_session_id_in_payload(self) -> None:
        self.write_plan(
            session_id="pointer-session",
            ordered_items=[(True, "Approved work already completed.")],
            approval_status="approved",
        )
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        outcome = handle_stop(
            {},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("requires `session_id`", outcome.stderr_message)
        self.assertEqual(test_runner.calls, [])

    def test_stop_allows_explicit_review_sessions_without_plan_file(self) -> None:
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        outcome = handle_stop(
            {"session_id": "review-only"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIsNone(outcome.stdout_payload)
        self.assertEqual(exec_runner.calls, [])
        self.assertEqual(test_runner.calls, [])

    def test_stop_reactivates_completed_plan_before_evaluating(self) -> None:
        self.write_completed_plan(
            session_id="reactivate-plan",
            ordered_items=[(False, "Continue the interrupted task.")],
            approval_status="approved",
        )

        outcome = handle_stop({"session_id": "reactivate-plan"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertTrue(self.paths.active_plan_path("reactivate-plan").exists())
        self.assertFalse(self.paths.completed_plan_path("reactivate-plan").exists())
        self.assertIn("Continue the interrupted task.", outcome.stderr_message)

    def test_stop_blocks_when_plan_approval_section_is_missing(self) -> None:
        plan_path = self.write_plan(session_id="missing-approval", approval_status="approved")
        original = plan_path.read_text(encoding="utf-8")
        broken = original.replace(f"{PLAN_APPROVAL_HEADING}\napproved\n\n", "", 1)
        atomic_write_text(plan_path, broken)

        outcome = handle_stop({"session_id": "missing-approval"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn(PLAN_APPROVAL_HEADING, outcome.stderr_message)

    def test_stop_blocks_when_phase_section_is_missing(self) -> None:
        plan_path = self.write_plan(session_id="missing-phase", approval_status="approved")
        original = plan_path.read_text(encoding="utf-8")
        broken = original.replace(f"{PLAN_PHASE_HEADING}\nimplementation\n\n", "", 1)
        atomic_write_text(plan_path, broken)

        outcome = handle_stop({"session_id": "missing-phase"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn(PLAN_PHASE_HEADING, outcome.stderr_message)

    def test_stop_blocks_when_required_checkbox_section_is_empty(self) -> None:
        plan_path = self.write_plan(session_id="empty-ordered", approval_status="approved")
        original = plan_path.read_text(encoding="utf-8")
        broken = original.replace("- [ ] Replace this placeholder with a real subtask.\n", "", 1)
        atomic_write_text(plan_path, broken)

        outcome = handle_stop({"session_id": "empty-ordered"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("empty", outcome.stderr_message)
        self.assertIn(ORDERED_SUBTASKS_HEADING, outcome.stderr_message)

    def test_stop_blocks_when_phase_is_incompatible_with_approval(self) -> None:
        self.write_plan(
            session_id="invalid-phase",
            ordered_items=[(True, "Approved work already completed.")],
            approval_status="approved",
            phase="planning",
        )

        outcome = handle_stop({"session_id": "invalid-phase"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn(PLAN_PHASE_HEADING, outcome.stderr_message)

    def test_implementation_phase_runs_selected_tests_without_reviewer_or_archiving(self) -> None:
        self.write_plan(
            session_id="approved-loop",
            ordered_items=[(True, "Approved work already completed.")],
            approval_status="approved",
        )
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        outcome = handle_stop(
            {"session_id": "approved-loop"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(
            outcome.stdout_payload,
            {
                "continue": False,
                "stopReason": "All subtasks are complete and the selected final tests passed. Reviewer execution was skipped because the session phase is `implementation`.",
            },
        )
        self.assertEqual(test_runner.calls, [("final-tests", True, True)])
        self.assertEqual(exec_runner.calls, [])
        self.assertTrue(self.paths.active_plan_path("approved-loop").exists())
        self.assertFalse(self.paths.completed_plan_path("approved-loop").exists())

    def test_stop_blocks_when_unchecked_ordered_subtask_remains(self) -> None:
        self.write_plan(
            session_id="impl-unchecked",
            ordered_items=[(False, "Implement the session start hook.")],
            approval_status="approved",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "impl-unchecked"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Implementation is not complete.", outcome.stderr_message)
        self.assertIn("Implement the session start hook.", outcome.stderr_message)
        self.assertEqual(test_runner.calls, [])
        self.assertEqual(exec_runner.calls, [])

    def test_stop_blocks_when_unchecked_follow_up_subtask_remains(self) -> None:
        self.write_plan(
            session_id="impl-follow-up-open",
            ordered_items=[(True, "Done ordered work.")],
            follow_up_items=[(False, "Fix the reviewer-reported edge case.")],
            approval_status="approved",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "impl-follow-up-open"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Implementation is not complete.", outcome.stderr_message)
        self.assertIn("Fix the reviewer-reported edge case.", outcome.stderr_message)
        self.assertEqual(test_runner.calls, [])
        self.assertEqual(exec_runner.calls, [])

    def test_test_failure_appends_follow_up_and_blocks(self) -> None:
        plan_path = self.write_plan(
            session_id="tests-fail",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        test_runner = FakeTestRunner(
            FinalTestResult(ok=False, summary="workflow unit tests failed", returncode=1)
        )
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "tests-fail"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Final tests failed.", outcome.stderr_message)
        self.assertEqual(exec_runner.calls, [])
        plan_text = plan_path.read_text(encoding="utf-8")
        self.assertIn("Fix the failing final validation from `scripts/run_tests.sh`", plan_text)
        self.assertIn("- [x] Implement everything.", plan_text)
        self.assertEqual(test_runner.calls, [("final-tests", True, True)])

    def test_review_phase_reviewer_revision_appends_follow_up_and_blocks(self) -> None:
        plan_path = self.write_plan(
            session_id="review-revision",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
            phase="review",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=True, result="revision", msg="Handle the malformed state path."),
        )

        outcome = handle_stop(
            {"session_id": "review-revision"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Reviewer requested changes.", outcome.stderr_message)
        plan_text = plan_path.read_text(encoding="utf-8")
        self.assertIn("Address reviewer feedback from the final read-only Codex review", plan_text)

    def test_review_phase_runs_tests_and_reviewer_before_archiving(self) -> None:
        self.write_plan(
            session_id="precheck-complete",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
            phase="review",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "precheck-complete"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(test_runner.calls, [("final-tests", True, True)])
        self.assertEqual(len(exec_runner.calls), 1)
        self.assertFalse(self.paths.active_plan_path("precheck-complete").exists())
        self.assertTrue(self.paths.completed_plan_path("precheck-complete").exists())

    def test_accepted_stop_moves_active_revision_logs_to_completed(self) -> None:
        self.write_plan(
            session_id="archive-logs",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
            phase="review",
        )
        active_stdout = self.paths.active_revision_logs_dir / "archive-logs-final-review-1.stdout.log"
        active_stderr = self.paths.active_revision_logs_dir / "archive-logs-final-review-1.stderr.log"
        atomic_write_text(active_stdout, "stdout\n")
        atomic_write_text(active_stderr, "stderr\n")
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "archive-logs"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertFalse(active_stdout.exists())
        self.assertFalse(active_stderr.exists())
        self.assertTrue((self.paths.completed_revision_logs_dir / active_stdout.name).exists())
        self.assertTrue((self.paths.completed_revision_logs_dir / active_stderr.name).exists())
        self.assertFalse(self.paths.active_plan_path("archive-logs").exists())
        self.assertTrue(self.paths.completed_plan_path("archive-logs").exists())

    def test_blocked_stop_keeps_plan_and_revision_logs_active(self) -> None:
        plan_path = self.write_plan(
            session_id="keep-active",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        active_stdout = self.paths.active_revision_logs_dir / "keep-active-final-tests-1.stdout.log"
        atomic_write_text(active_stdout, "stdout\n")
        test_runner = FakeTestRunner(FinalTestResult(ok=False, summary="workflow unit tests failed", returncode=1))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "keep-active"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertTrue(plan_path.exists())
        self.assertFalse(self.paths.completed_plan_path("keep-active").exists())
        self.assertTrue(active_stdout.exists())
        self.assertFalse((self.paths.completed_revision_logs_dir / active_stdout.name).exists())

    def test_implementation_phase_skips_tests_and_reviewer_when_no_files_changed(self) -> None:
        self.write_plan(
            session_id="no-file-changes",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "no-file-changes"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
            edited_paths_provider=lambda _: [],
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(
            outcome.stdout_payload,
            {
                "continue": False,
                "stopReason": "All subtasks are complete and no files changed, so selected final tests were skipped.",
            },
        )
        self.assertEqual(test_runner.calls, [])
        self.assertEqual(exec_runner.calls, [])

    def test_review_phase_skips_tests_but_still_runs_reviewer_for_non_selected_changes(self) -> None:
        self.write_plan(
            session_id="docs-only-changes",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
            phase="review",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "docs-only-changes"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
            edited_paths_provider=lambda _: ["README.md", "docs/notes.txt"],
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(
            outcome.stdout_payload,
            {
                "continue": False,
                "stopReason": "All subtasks are complete, selected final tests were skipped because no Python, Rust, Cargo.toml, or cuda/ files changed, and reviewer approved.",
            },
        )
        self.assertEqual(test_runner.calls, [])
        self.assertEqual(len(exec_runner.calls), 1)

    def test_complete_plan_runs_final_tests_when_cargo_toml_changed(self) -> None:
        self.write_plan(
            session_id="cargo-toml-change",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "cargo-toml-change"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
            edited_paths_provider=lambda _: ["Cargo.toml", "README.md"],
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(test_runner.calls, [("final-tests", False, True)])
        self.assertEqual(exec_runner.calls, [])

    def test_complete_plan_runs_final_tests_when_rust_file_changed(self) -> None:
        self.write_plan(
            session_id="rust-change",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "rust-change"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
            edited_paths_provider=lambda _: ["src/lib.rs"],
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(test_runner.calls, [("final-tests", False, True)])
        self.assertEqual(exec_runner.calls, [])

    def test_complete_plan_runs_only_python_tests_when_python_file_changed(self) -> None:
        self.write_plan(
            session_id="python-change",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "python-change"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
            edited_paths_provider=lambda _: ["scripts/lib/codex_workflow/hooks.py"],
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(test_runner.calls, [("final-tests", True, False)])
        self.assertEqual(exec_runner.calls, [])

    def test_complete_plan_runs_python_and_rust_tests_for_mixed_changes(self) -> None:
        self.write_plan(
            session_id="mixed-change",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "mixed-change"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
            edited_paths_provider=lambda _: ["scripts/lib/codex_workflow/hooks.py", "src/lib.rs"],
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(test_runner.calls, [("final-tests", True, True)])
        self.assertEqual(exec_runner.calls, [])

    def test_shell_final_test_runner_uses_python_only_flag(self) -> None:
        runner = ShellFinalTestRunner(self.paths, "python-only")
        with mock.patch("codex_workflow.runners.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0

            runner.run(label="final-tests", run_python=True, run_rust=False)

        command = run_mock.call_args.args[0]
        self.assertEqual(command, [str(self.paths.scripts_dir / "run_tests.sh"), "--python"])

    def test_shell_final_test_runner_uses_rust_only_flag(self) -> None:
        runner = ShellFinalTestRunner(self.paths, "rust-only")
        with mock.patch("codex_workflow.runners.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0

            runner.run(label="final-tests", run_python=False, run_rust=True)

        command = run_mock.call_args.args[0]
        self.assertEqual(command, [str(self.paths.scripts_dir / "run_tests.sh"), "--rust"])

    def test_shell_final_test_runner_uses_default_command_for_mixed_selection(self) -> None:
        runner = ShellFinalTestRunner(self.paths, "mixed")
        with mock.patch("codex_workflow.runners.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0

            runner.run(label="final-tests", run_python=True, run_rust=True)

        command = run_mock.call_args.args[0]
        self.assertEqual(command, [str(self.paths.scripts_dir / "run_tests.sh")])

    def test_summarize_logs_prefers_stdout_when_it_contains_test_failure_and_stderr_only_has_compile_noise(
        self,
    ) -> None:
        stdout_path = self.repo_root / "stdout.log"
        stderr_path = self.repo_root / "stderr.log"
        stdout_path.write_text(
            "\n".join(
                [
                    "test something ... ok",
                    "failures:",
                    "---- bench_estimator::tests::test_estimate_gate_bench_panics_on_sub_circuit_output_placeholder stdout ----",
                    "note: panic did not contain expected string",
                    'expected substring: "unexpected SubCircuitOutput"',
                    "test result: FAILED. 284 passed; 1 failed; 4 ignored; 0 measured; 0 filtered out; finished in 430.53s",
                    "error: test failed, to rerun pass `--lib`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text(
            "\n".join(
                [
                    "warning: `mxx` (lib) generated 3 warnings",
                    "warning: `mxx` (lib test) generated 3 warnings (3 duplicates)",
                    "Finished `release` profile [optimized] target(s) in 1m 05s",
                    "Executable tests/test_gpu_ggh15_negacyclic_conv_mul.rs (target/release/deps/test_gpu_ggh15_negacyclic_conv_mul-b997dc1c6882de89)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = summarize_logs(stdout_path, stderr_path)

        self.assertIn("test result: FAILED", summary)
        self.assertIn("error: test failed", summary)
        self.assertNotIn("Executable tests/test_gpu_ggh15_negacyclic_conv_mul.rs", summary)

    def test_reviewer_infra_failure_retries_with_backoff(self) -> None:
        self.write_plan(
            session_id="review-retry",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
            phase="review",
        )
        sleep_calls: list[float] = []
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=False, error="review infra failure"),
            StructuredExecResult(ok=True, result="accept", msg=None),
        )

        outcome = handle_stop(
            {"session_id": "review-retry"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=sleep_calls.append,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(sleep_calls, [1.0])
        self.assertEqual(len(exec_runner.calls), 2)

    def test_reviewer_infra_failure_stops_after_retry_limit(self) -> None:
        self.write_plan(
            session_id="review-retry-limit",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
            phase="review",
        )
        sleep_calls: list[float] = []
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(
            *[
                StructuredExecResult(ok=False, error="permanent review failure")
                for _ in range(MAX_INFRA_RETRY_ATTEMPTS)
            ]
        )

        outcome = handle_stop(
            {"session_id": "review-retry-limit"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
            sleep_fn=sleep_calls.append,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("100 failed nested reviewer attempts", outcome.stderr_message)
        self.assertIn("permanent review failure", outcome.stderr_message)
        self.assertEqual(len(exec_runner.calls), MAX_INFRA_RETRY_ATTEMPTS)
        self.assertEqual(len(sleep_calls), MAX_INFRA_RETRY_ATTEMPTS - 1)

    def test_transcript_parser_returns_latest_user_message(self) -> None:
        transcript_path = self.repo_root / "transcript.jsonl"
        atomic_write_text(
            transcript_path,
            "\n".join(
                [
                    "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"First\"}]}}",
                    "not-json",
                    "{\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"Final accept\"}}",
                ]
            )
            + "\n",
        )

        self.assertEqual(latest_user_message_from_transcript(transcript_path), "Final accept")

    def test_transcript_parser_returns_initial_user_message(self) -> None:
        transcript_path = self.repo_root / "initial-transcript.jsonl"
        atomic_write_text(
            transcript_path,
            "\n".join(
                [
                    "{\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"First prompt\"}}",
                    "{\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"Later follow-up\"}}",
                ]
            )
            + "\n",
        )

        self.assertEqual(initial_user_message_from_transcript(transcript_path), "First prompt")

    def test_transcript_parser_returns_latest_assistant_message(self) -> None:
        transcript_path = self.repo_root / "assistant-transcript.jsonl"
        atomic_write_text(
            transcript_path,
            "\n".join(
                [
                    "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"First draft\"}]}}",
                    "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Final plan prompt\"}]}}",
                ]
            )
            + "\n",
        )

        self.assertEqual(latest_assistant_message_from_transcript(transcript_path), "Final plan prompt")

    def test_atomic_write_text_overwrites_without_leaving_temp_files(self) -> None:
        target = self.repo_root / "sample.txt"
        atomic_write_text(target, "one\n")
        atomic_write_text(target, "two\n")

        self.assertEqual(target.read_text(encoding="utf-8"), "two\n")
        leftovers = list(target.parent.glob(".sample.txt.tmp-*"))
        self.assertEqual(leftovers, [])

    def test_emit_outcome_outputs_only_strict_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = emit_outcome(
            outcome=type(
                "Outcome",
                (),
                {"exit_code": 0, "stdout_payload": {"continue": False, "stopReason": "ok"}, "stderr_message": None},
            )(),
            stdout_stream=stdout,
            stderr_stream=stderr,
        )

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "{\"continue\":false,\"stopReason\":\"ok\"}")
        self.assertEqual(stderr.getvalue(), "")

    def test_plan_helper_preserves_completed_history_and_appends_new_follow_up_tasks(self) -> None:
        original = build_plan_text(
            "plan-helper",
            ordered_items=[(True, "Completed historical task.")],
            follow_up_items=[
                (True, "Already fixed earlier."),
                (False, "Existing open obligation."),
            ],
        )

        updated = append_follow_up_subtasks(
            original,
            [
                "Existing open obligation.",
                "Investigate the new regression from reviewer feedback.",
            ],
        )

        self.assertIn("- [x] Completed historical task.", updated)
        self.assertEqual(updated.count("Existing open obligation."), 1)
        self.assertIn("- [ ] Investigate the new regression from reviewer feedback.", updated)

    def test_plan_analysis_requires_required_sections_and_checkboxes(self) -> None:
        analysis = analyze_plan(render_session_plan("analysis-test"))

        self.assertFalse(analysis.all_checked)
        self.assertEqual(analysis.approval_status, "unapproved")
        self.assertEqual(analysis.phase_status, "planning")
        self.assertFalse(analysis.is_approved)
        self.assertEqual(analysis.phase, "planning")
        self.assertEqual(analysis.missing_sections, [])
        self.assertEqual(analysis.empty_sections, [])
        self.assertEqual(analysis.invalid_sections, [])
        self.assertEqual(len(analysis.tracked_items), 2)

    def test_plan_analysis_detects_approved_flag(self) -> None:
        analysis = analyze_plan(
            build_plan_text(
                "approved-analysis",
                ordered_items=[(True, "Done ordered work.")],
                approval_status="approved",
            )
        )

        self.assertEqual(analysis.approval_status, "approved")
        self.assertEqual(analysis.phase_status, "implementation")
        self.assertTrue(analysis.is_approved)
        self.assertEqual(analysis.phase, "implementation")

    def test_plan_analysis_preserves_explicit_review_phase(self) -> None:
        analysis = analyze_plan(
            build_plan_text(
                "review-analysis",
                ordered_items=[(True, "Done ordered work.")],
                approval_status="approved",
                phase="review",
            )
        )

        self.assertEqual(analysis.approval_status, "approved")
        self.assertEqual(analysis.phase_status, "review")
        self.assertEqual(analysis.phase, "review")

    def test_plan_analysis_requires_phase_section(self) -> None:
        missing_phase_plan = build_plan_text(
            "missing-phase-analysis",
            ordered_items=[(True, "Done ordered work.")],
            approval_status="approved",
        ).replace(f"\n{PLAN_PHASE_HEADING}\nimplementation\n", "", 1)

        analysis = analyze_plan(missing_phase_plan)

        self.assertEqual(analysis.approval_status, "approved")
        self.assertIsNone(analysis.phase_status)
        self.assertIn(PLAN_PHASE_HEADING, analysis.missing_sections)


if __name__ == "__main__":
    unittest.main()
