from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest import mock

from codex_workflow.atomic import atomic_write_text
from codex_workflow.cli import emit_outcome
from codex_workflow.hooks import MAX_INFRA_RETRY_ATTEMPTS, handle_session_start, handle_stop
from codex_workflow.paths import RepoPaths
from codex_workflow.plan import (
    FOLLOW_UP_SUBTASKS_HEADING,
    ORDERED_SUBTASKS_HEADING,
    PLAN_APPROVAL_HEADING,
    analyze_plan,
    append_follow_up_subtasks,
    render_session_plan,
)
from codex_workflow.runners import BuilderExecResult, CodexBuilderRunner, FinalTestResult, StructuredExecResult
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
) -> str:
    if follow_up_items is None:
        follow_up_items = [(True, "No follow-up subtasks have been added yet.")]
    ordered_block = "\n".join(f"- [{'x' if checked else ' '}] {text}" for checked, text in ordered_items)
    follow_up_block = "\n".join(f"- [{'x' if checked else ' '}] {text}" for checked, text in follow_up_items)
    return f"""# Session Plan: {session_id}

{PLAN_APPROVAL_HEADING}
{approval_status}

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
        self.calls: list[str] = []

    def run(self, label: str) -> FinalTestResult:
        self.calls.append(label)
        return self.results.pop(0)


class FakeBuilderRunner:
    def __init__(
        self,
        *results: BuilderExecResult,
        progress_messages: list[str] | None = None,
        on_calls: list[Callable[[], None] | None] | None = None,
    ) -> None:
        self.results = list(results)
        self.progress_messages = list(progress_messages or [])
        self.on_calls = list(on_calls or [])
        self.calls: list[tuple[str, str]] = []

    def run(self, prompt: str, label: str) -> BuilderExecResult:
        self.calls.append((prompt, label))
        call_index = len(self.calls) - 1
        if call_index < len(self.progress_messages):
            sys.stderr.write(self.progress_messages[call_index])
            sys.stderr.flush()
        if call_index < len(self.on_calls):
            callback = self.on_calls[call_index]
            if callback is not None:
                callback()
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
        self.assertFalse((self.paths.agents_dir / "current-session-id").exists())
        self.assertEqual(list(self.paths.agents_dir.glob("session-abc123.json")), [])
        self.assertEqual(len(exec_runner.calls), 1)
        prompt, schema_path, label = exec_runner.calls[0]
        self.assertIn("Implement the stop hook changes.", prompt)
        self.assertTrue(str(schema_path).endswith("session-start-intent.schema.json"))
        self.assertEqual(label, "session-start-intent")

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
        builder_runner = FakeBuilderRunner()
        test_runner = FakeTestRunner()

        outcome = handle_stop(
            {"session_id": "plan-first"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIsNone(outcome.stdout_payload)
        self.assertEqual(exec_runner.calls, [])
        self.assertEqual(builder_runner.calls, [])
        self.assertEqual(test_runner.calls, [])

    def test_planning_stop_allows_stop_even_when_required_checkbox_section_is_empty(self) -> None:
        plan_path = self.write_plan(session_id="planning-empty", approval_status="unapproved")
        broken = plan_path.read_text(encoding="utf-8").replace("- [ ] Replace this placeholder with a real subtask.\n", "", 1)
        atomic_write_text(plan_path, broken)
        exec_runner = FakeExecRunner()
        builder_runner = FakeBuilderRunner()
        test_runner = FakeTestRunner()

        outcome = handle_stop(
            {"session_id": "planning-empty"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIsNone(outcome.stdout_payload)
        self.assertEqual(exec_runner.calls, [])
        self.assertEqual(builder_runner.calls, [])
        self.assertEqual(test_runner.calls, [])

    def test_stop_requires_session_id_in_payload(self) -> None:
        self.write_plan(
            session_id="pointer-session",
            ordered_items=[(True, "Approved work already completed.")],
            approval_status="approved",
        )
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="builder ok", returncode=0))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        outcome = handle_stop(
            {},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("requires `session_id`", outcome.stderr_message)
        self.assertEqual(len(builder_runner.calls), 0)
        self.assertEqual(test_runner.calls, [])

    def test_stop_allows_explicit_review_sessions_without_plan_file(self) -> None:
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="builder ok", returncode=0))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        outcome = handle_stop(
            {"session_id": "review-only"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIsNone(outcome.stdout_payload)
        self.assertEqual(exec_runner.calls, [])
        self.assertEqual(builder_runner.calls, [])
        self.assertEqual(test_runner.calls, [])

    def test_stop_blocks_when_plan_approval_section_is_missing(self) -> None:
        plan_path = self.write_plan(session_id="missing-approval", approval_status="approved")
        original = plan_path.read_text(encoding="utf-8")
        broken = original.replace(f"{PLAN_APPROVAL_HEADING}\napproved\n\n", "", 1)
        atomic_write_text(plan_path, broken)

        outcome = handle_stop({"session_id": "missing-approval"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn(PLAN_APPROVAL_HEADING, outcome.stderr_message)

    def test_stop_blocks_when_required_checkbox_section_is_empty(self) -> None:
        plan_path = self.write_plan(session_id="empty-ordered", approval_status="approved")
        original = plan_path.read_text(encoding="utf-8")
        broken = original.replace("- [ ] Replace this placeholder with a real subtask.\n", "", 1)
        atomic_write_text(plan_path, broken)

        outcome = handle_stop({"session_id": "empty-ordered"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("empty", outcome.stderr_message)
        self.assertIn(ORDERED_SUBTASKS_HEADING, outcome.stderr_message)

    def test_approved_plan_enters_implementation_loop(self) -> None:
        self.write_plan(
            session_id="approved-loop",
            ordered_items=[(True, "Approved work already completed.")],
            approval_status="approved",
        )
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="builder ok", returncode=0))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        outcome = handle_stop(
            {"session_id": "approved-loop"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(
            outcome.stdout_payload,
            {
                "continue": False,
                "stopReason": "All subtasks are complete, final tests passed, and reviewer approved.",
            },
        )
        self.assertEqual(len(builder_runner.calls), 0)
        self.assertEqual(test_runner.calls, ["final-tests"])
        self.assertEqual(len(exec_runner.calls), 1)

    def test_stop_relaunches_builder_until_unchecked_ordered_subtask_is_completed(self) -> None:
        self.write_plan(
            session_id="impl-unchecked",
            ordered_items=[(False, "Implement the session start hook.")],
            approval_status="approved",
        )
        plan_path = self.paths.default_plan_path("impl-unchecked")
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder pass 1", returncode=0),
            BuilderExecResult(ok=True, summary="builder pass 2", returncode=0),
            on_calls=[
                None,
                lambda: atomic_write_text(
                    plan_path,
                    build_plan_text(
                        "impl-unchecked",
                        ordered_items=[(True, "Implement the session start hook.")],
                        approval_status="approved",
                    ),
                ),
            ],
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "impl-unchecked"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(len(builder_runner.calls), 2)
        self.assertEqual(test_runner.calls, ["final-tests"])
        self.assertEqual(len(exec_runner.calls), 1)

    def test_stop_relaunches_builder_until_follow_up_checkbox_is_completed(self) -> None:
        self.write_plan(
            session_id="impl-follow-up-open",
            ordered_items=[(True, "Done ordered work.")],
            follow_up_items=[(False, "Fix the reviewer-reported edge case.")],
            approval_status="approved",
        )
        plan_path = self.paths.default_plan_path("impl-follow-up-open")
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder pass 1", returncode=0),
            BuilderExecResult(ok=True, summary="builder pass 2", returncode=0),
            on_calls=[
                None,
                lambda: atomic_write_text(
                    plan_path,
                    build_plan_text(
                        "impl-follow-up-open",
                        ordered_items=[(True, "Done ordered work.")],
                        follow_up_items=[(True, "Fix the reviewer-reported edge case.")],
                        approval_status="approved",
                    ),
                ),
            ],
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "impl-follow-up-open"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(len(builder_runner.calls), 2)
        self.assertEqual(test_runner.calls, ["final-tests"])

    def test_test_failure_appends_follow_up_and_relaunches_builder(self) -> None:
        plan_path = self.write_plan(
            session_id="tests-fail",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        expected_follow_up = "Fix the failing final validation from `scripts/run_tests.sh`: workflow unit tests failed"
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder pass 1", returncode=0),
            BuilderExecResult(ok=True, summary="builder pass 2", returncode=0),
            on_calls=[
                None,
                lambda: atomic_write_text(
                    plan_path,
                    plan_path.read_text(encoding="utf-8").replace(
                        f"- [ ] {expected_follow_up}",
                        f"- [x] {expected_follow_up}",
                    ),
                ),
            ],
        )
        test_runner = FakeTestRunner(
            FinalTestResult(ok=False, summary="workflow unit tests failed", returncode=1),
            FinalTestResult(ok=True, summary="ok", returncode=0),
        )
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "tests-fail"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(len(builder_runner.calls), 2)
        self.assertEqual(exec_runner.calls[0][2], "final-review")
        plan_text = plan_path.read_text(encoding="utf-8")
        self.assertIn("Fix the failing final validation from `scripts/run_tests.sh`", plan_text)
        self.assertIn("- [x] Implement everything.", plan_text)

    def test_reviewer_revision_appends_follow_up_and_relaunches_builder(self) -> None:
        plan_path = self.write_plan(
            session_id="review-revision",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        expected_follow_up = "Address reviewer feedback from the final read-only Codex review: Handle the malformed state path."
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder pass 1", returncode=0),
            BuilderExecResult(ok=True, summary="builder pass 2", returncode=0),
            on_calls=[
                None,
                lambda: atomic_write_text(
                    plan_path,
                    plan_path.read_text(encoding="utf-8").replace(
                        f"- [ ] {expected_follow_up}",
                        f"- [x] {expected_follow_up}",
                    ),
                ),
            ],
        )
        test_runner = FakeTestRunner(
            FinalTestResult(ok=True, summary="ok", returncode=0),
            FinalTestResult(ok=True, summary="ok", returncode=0),
        )
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=True, result="revision", msg="Handle the malformed state path."),
            StructuredExecResult(ok=True, result="accept", msg=None),
        )

        outcome = handle_stop(
            {"session_id": "review-revision"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(len(builder_runner.calls), 2)
        plan_text = plan_path.read_text(encoding="utf-8")
        self.assertIn("Address reviewer feedback from the final read-only Codex review", plan_text)

    def test_builder_progress_is_streamed_to_stderr_only(self) -> None:
        plan_path = self.write_plan(
            session_id="stderr-progress",
            ordered_items=[(False, "Implement everything.")],
            approval_status="approved",
        )
        stderr_buffer = io.StringIO()
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder ok", returncode=0),
            progress_messages=["builder progress line\n"],
            on_calls=[
                lambda: atomic_write_text(
                    plan_path,
                    build_plan_text(
                        "stderr-progress",
                        ordered_items=[(True, "Implement everything.")],
                        approval_status="approved",
                    ),
                )
            ],
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        with contextlib.redirect_stderr(stderr_buffer):
            outcome = handle_stop(
                {"session_id": "stderr-progress"},
                self.repo_root,
                exec_runner=exec_runner,
                builder_runner=builder_runner,
                test_runner=test_runner,
                sleep_fn=lambda _: None,
            )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIn("builder progress line", stderr_buffer.getvalue())
        self.assertEqual(
            outcome.stdout_payload,
            {
                "continue": False,
                "stopReason": "All subtasks are complete, final tests passed, and reviewer approved.",
            },
        )

    def test_builder_infra_failure_retries_with_backoff(self) -> None:
        plan_path = self.write_plan(
            session_id="builder-retry",
            ordered_items=[(False, "Implement everything.")],
            approval_status="approved",
        )
        sleep_calls: list[float] = []
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=False, summary="temporary builder failure", returncode=1),
            BuilderExecResult(ok=True, summary="builder ok", returncode=0),
            on_calls=[
                None,
                lambda: atomic_write_text(
                    plan_path,
                    build_plan_text(
                        "builder-retry",
                        ordered_items=[(True, "Implement everything.")],
                        approval_status="approved",
                    ),
                ),
            ],
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "builder-retry"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=sleep_calls.append,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(sleep_calls, [1.0])
        self.assertEqual(len(builder_runner.calls), 2)

    def test_builder_infra_failure_stops_after_retry_limit(self) -> None:
        self.write_plan(
            session_id="builder-retry-limit",
            ordered_items=[(False, "Implement everything.")],
            approval_status="approved",
        )
        sleep_calls: list[float] = []
        builder_runner = FakeBuilderRunner(
            *[
                BuilderExecResult(ok=False, summary="permanent builder failure", returncode=1)
                for _ in range(MAX_INFRA_RETRY_ATTEMPTS)
            ]
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "builder-retry-limit"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=sleep_calls.append,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("100 failed nested builder attempts", outcome.stderr_message)
        self.assertIn("permanent builder failure", outcome.stderr_message)
        self.assertEqual(len(builder_runner.calls), MAX_INFRA_RETRY_ATTEMPTS)
        self.assertEqual(len(sleep_calls), MAX_INFRA_RETRY_ATTEMPTS - 1)

    def test_complete_plan_is_evaluated_before_builder_launch(self) -> None:
        self.write_plan(
            session_id="precheck-complete",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="unused", returncode=0))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "precheck-complete"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(builder_runner.calls, [])
        self.assertEqual(test_runner.calls, ["final-tests"])
        self.assertEqual(len(exec_runner.calls), 1)

    def test_reviewer_infra_failure_retries_with_backoff(self) -> None:
        self.write_plan(
            session_id="review-retry",
            ordered_items=[(True, "Implement everything.")],
            approval_status="approved",
        )
        sleep_calls: list[float] = []
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="builder ok", returncode=0))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=False, error="review infra failure"),
            StructuredExecResult(ok=True, result="accept", msg=None),
        )

        outcome = handle_stop(
            {"session_id": "review-retry"},
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
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
        )
        sleep_calls: list[float] = []
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="builder ok", returncode=0))
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
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=sleep_calls.append,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("100 failed nested reviewer attempts", outcome.stderr_message)
        self.assertIn("permanent review failure", outcome.stderr_message)
        self.assertEqual(len(exec_runner.calls), MAX_INFRA_RETRY_ATTEMPTS)
        self.assertEqual(len(sleep_calls), MAX_INFRA_RETRY_ATTEMPTS - 1)

    def test_codex_builder_runner_resumes_same_session_id_on_first_and_later_runs(self) -> None:
        runner = CodexBuilderRunner(self.paths, "builder-runner", progress_stream=io.StringIO())
        commands: list[list[str]] = []

        def fake_run_streaming_subprocess(
            command: list[str],
            cwd: Path,
            stdout_path: Path,
            stderr_path: Path,
            mirror_stream: io.StringIO | None = None,
            stdout_line_handler: Callable[[str], None] | None = None,
            stderr_line_handler: Callable[[str], None] | None = None,
        ) -> int:
            commands.append(command)
            event = '{"type":"thread.started","thread_id":"builder-runner"}\n'
            stdout_path.write_text(event, encoding="utf-8")
            stderr_path.write_text("builder stderr\n", encoding="utf-8")
            if stdout_line_handler is not None:
                stdout_line_handler(event)
            return 0

        with mock.patch("codex_workflow.runners._run_streaming_subprocess", side_effect=fake_run_streaming_subprocess):
            first = runner.run(prompt="first prompt", label="first")
            second = runner.run(prompt="second prompt", label="second")

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(first.thread_id, "builder-runner")
        self.assertEqual(second.thread_id, "builder-runner")
        self.assertEqual(commands[0][:3], ["codex", "exec", "resume"])
        self.assertNotIn("--sandbox", commands[0])
        self.assertNotIn("workspace-write", commands[0])
        self.assertIn("--disable", commands[0])
        self.assertIn("codex_hooks", commands[0])
        self.assertIn("builder-runner", commands[0])
        self.assertEqual(commands[1][:3], ["codex", "exec", "resume"])
        self.assertNotIn("--sandbox", commands[1])
        self.assertNotIn("workspace-write", commands[1])
        self.assertIn("--disable", commands[1])
        self.assertIn("codex_hooks", commands[1])
        self.assertIn("builder-runner", commands[1])

    def test_codex_builder_runner_still_resumes_outer_session_after_failed_attempt(self) -> None:
        runner = CodexBuilderRunner(self.paths, "builder-runner", progress_stream=io.StringIO())
        commands: list[list[str]] = []
        returncodes = iter([1, 0])

        def fake_run_streaming_subprocess(
            command: list[str],
            cwd: Path,
            stdout_path: Path,
            stderr_path: Path,
            mirror_stream: io.StringIO | None = None,
            stdout_line_handler: Callable[[str], None] | None = None,
            stderr_line_handler: Callable[[str], None] | None = None,
        ) -> int:
            commands.append(command)
            if len(commands) == 1:
                stdout_path.write_text("", encoding="utf-8")
            else:
                event = '{"type":"thread.started","thread_id":"builder-runner"}\n'
                stdout_path.write_text(event, encoding="utf-8")
                if stdout_line_handler is not None:
                    stdout_line_handler(event)
            stderr_path.write_text("builder stderr\n", encoding="utf-8")
            return next(returncodes)

        with mock.patch("codex_workflow.runners._run_streaming_subprocess", side_effect=fake_run_streaming_subprocess):
            first = runner.run(prompt="first prompt", label="first")
            second = runner.run(prompt="second prompt", label="second")

        self.assertFalse(first.ok)
        self.assertTrue(second.ok)
        self.assertIsNone(first.thread_id)
        self.assertEqual(second.thread_id, "builder-runner")
        self.assertEqual(commands[0][:3], ["codex", "exec", "resume"])
        self.assertEqual(commands[1][:3], ["codex", "exec", "resume"])
        self.assertNotIn("--sandbox", commands[0])
        self.assertNotIn("--sandbox", commands[1])
        self.assertIn("builder-runner", commands[0])
        self.assertIn("builder-runner", commands[1])

    def test_codex_builder_runner_fails_when_first_resume_emits_different_session_id(self) -> None:
        runner = CodexBuilderRunner(self.paths, "builder-runner", progress_stream=io.StringIO())

        def fake_run_streaming_subprocess(
            command: list[str],
            cwd: Path,
            stdout_path: Path,
            stderr_path: Path,
            mirror_stream: io.StringIO | None = None,
            stdout_line_handler: Callable[[str], None] | None = None,
            stderr_line_handler: Callable[[str], None] | None = None,
        ) -> int:
            event = '{"type":"thread.started","thread_id":"wrong-thread"}\n'
            stdout_path.write_text(event, encoding="utf-8")
            stderr_path.write_text("builder stderr\n", encoding="utf-8")
            if stdout_line_handler is not None:
                stdout_line_handler(event)
            return 0

        with mock.patch("codex_workflow.runners._run_streaming_subprocess", side_effect=fake_run_streaming_subprocess):
            result = runner.run(prompt="first prompt", label="first")

        self.assertFalse(result.ok)
        self.assertEqual(result.thread_id, "wrong-thread")
        self.assertIn("wrong-thread", result.summary)
        self.assertIn("builder-runner", result.summary)

    def test_codex_builder_runner_fails_when_later_resume_emits_different_session_id(self) -> None:
        runner = CodexBuilderRunner(self.paths, "builder-runner", progress_stream=io.StringIO())
        observed_ids = iter(["builder-runner", "wrong-thread"])

        def fake_run_streaming_subprocess(
            command: list[str],
            cwd: Path,
            stdout_path: Path,
            stderr_path: Path,
            mirror_stream: io.StringIO | None = None,
            stdout_line_handler: Callable[[str], None] | None = None,
            stderr_line_handler: Callable[[str], None] | None = None,
        ) -> int:
            event = f'{{"type":"thread.started","thread_id":"{next(observed_ids)}"}}\n'
            stdout_path.write_text(event, encoding="utf-8")
            stderr_path.write_text("builder stderr\n", encoding="utf-8")
            if stdout_line_handler is not None:
                stdout_line_handler(event)
            return 0

        with mock.patch("codex_workflow.runners._run_streaming_subprocess", side_effect=fake_run_streaming_subprocess):
            first = runner.run(prompt="first prompt", label="first")
            second = runner.run(prompt="second prompt", label="second")

        self.assertTrue(first.ok)
        self.assertEqual(first.thread_id, "builder-runner")
        self.assertFalse(second.ok)
        self.assertEqual(second.thread_id, "wrong-thread")
        self.assertIn("wrong-thread", second.summary)
        self.assertIn("builder-runner", second.summary)

    def test_codex_builder_runner_fails_when_resume_does_not_emit_thread_started(self) -> None:
        runner = CodexBuilderRunner(self.paths, "builder-runner", progress_stream=io.StringIO())

        def fake_run_streaming_subprocess(
            command: list[str],
            cwd: Path,
            stdout_path: Path,
            stderr_path: Path,
            mirror_stream: io.StringIO | None = None,
            stdout_line_handler: Callable[[str], None] | None = None,
            stderr_line_handler: Callable[[str], None] | None = None,
        ) -> int:
            stdout_path.write_text('{"type":"response.output_text.delta","delta":"partial"}\n', encoding="utf-8")
            stderr_path.write_text("builder stderr\n", encoding="utf-8")
            if stdout_line_handler is not None:
                stdout_line_handler('{"type":"response.output_text.delta","delta":"partial"}\n')
            return 0

        with mock.patch("codex_workflow.runners._run_streaming_subprocess", side_effect=fake_run_streaming_subprocess):
            result = runner.run(prompt="first prompt", label="first")

        self.assertFalse(result.ok)
        self.assertIsNone(result.thread_id)
        self.assertIn("did not emit a `thread.started` event", result.summary)

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
        self.assertTrue(analysis.is_approved)
        self.assertEqual(analysis.phase, "implementation")


if __name__ == "__main__":
    unittest.main()
