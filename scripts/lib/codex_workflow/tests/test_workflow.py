from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Callable

from codex_workflow.atomic import atomic_write_text
from codex_workflow.cli import emit_outcome
from codex_workflow.hooks import handle_session_start, handle_stop
from codex_workflow.paths import RepoPaths
from codex_workflow.plan import (
    FOLLOW_UP_SUBTASKS_HEADING,
    ORDERED_SUBTASKS_HEADING,
    analyze_plan,
    append_follow_up_subtasks,
    render_session_plan,
)
from codex_workflow.runners import BuilderExecResult, CodexBuilderRunner, FinalTestResult, StructuredExecResult
from codex_workflow.state import load_state, save_state, write_current_session_id
from codex_workflow.transcript import latest_assistant_message_from_transcript, latest_user_message_from_transcript


def build_plan_text(
    session_id: str,
    ordered_items: list[tuple[bool, str]],
    follow_up_items: list[tuple[bool, str]] | None = None,
) -> str:
    if follow_up_items is None:
        follow_up_items = [(True, "No follow-up subtasks have been added yet.")]
    ordered_block = "\n".join(f"- [{'x' if checked else ' '}] {text}" for checked, text in ordered_items)
    follow_up_block = "\n".join(f"- [{'x' if checked else ' '}] {text}" for checked, text in follow_up_items)
    return f"""# Session Plan: {session_id}

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
        atomic_write_text(self.paths.schemas_dir / "plan-approval.schema.json", "{}\n")
        atomic_write_text(self.paths.schemas_dir / "review-decision.schema.json", "{}\n")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def session_state(self, session_id: str = "session-1", phase: str = "planning") -> Path:
        state = load_state(self.paths, session_id).state
        state.phase = phase
        save_state(self.paths, state)
        write_current_session_id(self.paths, session_id)
        return self.paths.default_plan_path(session_id)

    def write_plan(
        self,
        session_id: str = "session-1",
        phase: str = "planning",
        ordered_items: list[tuple[bool, str]] | None = None,
        follow_up_items: list[tuple[bool, str]] | None = None,
    ) -> Path:
        if ordered_items is None:
            ordered_items = [(False, "Replace this placeholder with a real subtask.")]
        plan_path = self.session_state(session_id=session_id, phase=phase)
        atomic_write_text(plan_path, build_plan_text(session_id, ordered_items, follow_up_items))
        return plan_path

    def test_session_start_initializes_state_plan_and_pointer(self) -> None:
        outcome = handle_session_start({"session_id": "abc123"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(self.paths.current_session_id_path.read_text(encoding="utf-8").strip(), "abc123")
        state = load_state(self.paths, "abc123").state
        self.assertEqual(state.phase, "planning")
        self.assertFalse(state.awaiting_plan_reply)
        self.assertEqual(state.plan_doc, "./plans/session-abc123.md")
        self.assertTrue(self.paths.default_plan_path("abc123").exists())

    def test_plan_approval_schema_requires_all_defined_properties(self) -> None:
        schema_path = Path(__file__).resolve().parents[4] / "schemas" / "plan-approval.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(set(schema["properties"]), {"is_approve", "msg"})
        self.assertEqual(set(schema["required"]), {"is_approve", "msg"})

    def test_review_decision_schema_requires_all_defined_properties(self) -> None:
        schema_path = Path(__file__).resolve().parents[4] / "schemas" / "review-decision.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(set(schema["properties"]), {"result", "msg"})
        self.assertEqual(set(schema["required"]), {"result", "msg"})

    def test_session_start_is_idempotent_for_existing_state(self) -> None:
        plan_path = self.write_plan(session_id="abc123", phase="implementation", ordered_items=[(True, "Done")])
        original = load_state(self.paths, "abc123").state
        original.updated_at = "2026-01-01T00:00:00Z"
        save_state(self.paths, original)

        outcome = handle_session_start({"session_id": "abc123"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 0)
        state = load_state(self.paths, "abc123").state
        self.assertEqual(state.phase, "implementation")
        self.assertEqual(state.plan_doc, "./plans/session-abc123.md")
        self.assertTrue(plan_path.exists())

    def test_session_start_recovers_malformed_state(self) -> None:
        state_path = self.paths.session_state_path("broken")
        atomic_write_text(state_path, "{not-json\n")

        outcome = handle_session_start({"session_id": "broken"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 0)
        state = load_state(self.paths, "broken").state
        self.assertEqual(state.phase, "planning")
        backups = list(self.paths.agents_dir.glob("session-broken.json.corrupt-*"))
        self.assertTrue(backups)

    def test_planning_stop_first_turn_sets_awaiting_flag_and_ends_turn(self) -> None:
        self.write_plan(session_id="plan-first", phase="planning")

        outcome = handle_stop(
            {"session_id": "plan-first"},
            self.repo_root,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIsNone(outcome.stdout_payload)
        state = load_state(self.paths, "plan-first").state
        self.assertEqual(state.phase, "planning")
        self.assertTrue(state.awaiting_plan_reply)

    def test_planning_stop_accept_reply_transitions_to_implementation(self) -> None:
        self.write_plan(
            session_id="accept-transition",
            phase="planning",
            ordered_items=[(True, "Approved work already completed.")],
        )
        state = load_state(self.paths, "accept-transition").state
        state.awaiting_plan_reply = True
        save_state(self.paths, state)
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=True, payload={"is_approve": True}),
            StructuredExecResult(ok=True, result="accept", msg=None),
        )
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="builder ok", returncode=0))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        outcome = handle_stop(
            {
                "session_id": "accept-transition",
                "latest_user_message": "Looks good, go ahead.",
            },
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(outcome.exit_code, 0)
        state = load_state(self.paths, "accept-transition").state
        self.assertEqual(state.phase, "implementation")
        self.assertFalse(state.awaiting_plan_reply)
        self.assertEqual(state.last_plan_check.result, "accept")
        self.assertIsNotNone(state.last_plan_check.at)
        self.assertTrue(state.completed)
        self.assertEqual(len(builder_runner.calls), 1)
        self.assertEqual(test_runner.calls, ["final-tests"])

    def test_planning_stop_revision_reply_stays_in_planning(self) -> None:
        self.write_plan(session_id="revision-reply", phase="planning")
        state = load_state(self.paths, "revision-reply").state
        state.awaiting_plan_reply = True
        save_state(self.paths, state)
        exec_runner = FakeExecRunner(
            StructuredExecResult(
                ok=True,
                payload={"is_approve": False, "msg": "Add error handling for the edge case."},
            )
        )

        outcome = handle_stop(
            {
                "session_id": "revision-reply",
                "latest_user_message": "Please add error handling for the edge case.",
            },
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Plan revisions are still required", outcome.stderr_message)
        self.assertIn("error handling", outcome.stderr_message)
        state = load_state(self.paths, "revision-reply").state
        self.assertEqual(state.phase, "planning")
        self.assertFalse(state.awaiting_plan_reply)
        self.assertEqual(state.last_plan_check.result, "revision")
        self.assertEqual(state.last_plan_check.msg, "Add error handling for the edge case.")

    def test_planning_stop_exec_failure_falls_back_to_revision(self) -> None:
        self.write_plan(session_id="exec-fail", phase="planning")
        state = load_state(self.paths, "exec-fail").state
        state.awaiting_plan_reply = True
        save_state(self.paths, state)
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=False, error="codex exec crashed")
        )

        outcome = handle_stop(
            {
                "session_id": "exec-fail",
                "latest_user_message": "ACCEPT",
            },
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Plan revisions are still required", outcome.stderr_message)
        state = load_state(self.paths, "exec-fail").state
        self.assertEqual(state.phase, "planning")
        self.assertFalse(state.awaiting_plan_reply)
        self.assertEqual(state.last_plan_check.result, "revision")

    def test_planning_stop_passes_user_message_in_exec_prompt(self) -> None:
        self.write_plan(
            session_id="prompt-check",
            phase="planning",
            ordered_items=[(True, "Approved work already completed.")],
        )
        state = load_state(self.paths, "prompt-check").state
        state.awaiting_plan_reply = True
        save_state(self.paths, state)
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=True, payload={"is_approve": True}),
            StructuredExecResult(ok=True, result="accept", msg=None),
        )
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="builder ok", returncode=0))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        handle_stop(
            {
                "session_id": "prompt-check",
                "latest_user_message": "Ship it!",
            },
            self.repo_root,
            exec_runner=exec_runner,
            builder_runner=builder_runner,
            test_runner=test_runner,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(len(exec_runner.calls), 2)
        prompt, schema_path, label = exec_runner.calls[0]
        self.assertIn("Ship it!", prompt)
        self.assertTrue(str(schema_path).endswith("plan-approval.schema.json"))
        self.assertEqual(label, "plan-classify")

    def test_planning_stop_revision_without_msg_uses_user_message(self) -> None:
        self.write_plan(session_id="no-msg-field", phase="planning")
        state = load_state(self.paths, "no-msg-field").state
        state.awaiting_plan_reply = True
        save_state(self.paths, state)
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=True, payload={"is_approve": False})
        )

        outcome = handle_stop(
            {
                "session_id": "no-msg-field",
                "latest_user_message": "Add more tests please.",
            },
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Add more tests please.", outcome.stderr_message)
        state = load_state(self.paths, "no-msg-field").state
        self.assertEqual(state.last_plan_check.msg, "Add more tests please.")

    def test_stop_relaunches_builder_until_unchecked_ordered_subtask_is_completed(self) -> None:
        self.write_plan(
            session_id="impl-unchecked",
            phase="implementation",
            ordered_items=[(False, "Implement the session start hook.")],
        )
        plan_path = self.paths.default_plan_path("impl-unchecked")
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder pass 1", returncode=0),
            BuilderExecResult(ok=True, summary="builder pass 2", returncode=0),
            on_calls=[
                None,
                lambda: atomic_write_text(
                    plan_path,
                    build_plan_text("impl-unchecked", ordered_items=[(True, "Implement the session start hook.")]),
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
            phase="implementation",
            ordered_items=[(True, "Done ordered work.")],
            follow_up_items=[(False, "Fix the reviewer-reported edge case.")],
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
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
        )
        expected_follow_up = "Fix the failing final validation from `scripts/run_tests.sh`: workflow unit tests failed"
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder pass 1", returncode=0),
            BuilderExecResult(ok=True, summary="builder pass 2", returncode=0),
            on_calls=[
                None,
                lambda: atomic_write_text(
                    plan_path,
                    plan_path.read_text(encoding="utf-8").replace(f"- [ ] {expected_follow_up}", f"- [x] {expected_follow_up}"),
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
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
        )
        expected_follow_up = "Address reviewer feedback from the final read-only Codex review: Handle the malformed state path."
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder pass 1", returncode=0),
            BuilderExecResult(ok=True, summary="builder pass 2", returncode=0),
            on_calls=[
                None,
                lambda: atomic_write_text(
                    plan_path,
                    plan_path.read_text(encoding="utf-8").replace(f"- [ ] {expected_follow_up}", f"- [x] {expected_follow_up}"),
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
        state = load_state(self.paths, "review-revision").state
        self.assertEqual(state.last_review.result, "accept")

    def test_reviewer_accept_completes_the_session(self) -> None:
        self.write_plan(
            session_id="review-accept",
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
        )
        builder_runner = FakeBuilderRunner(BuilderExecResult(ok=True, summary="builder ok", returncode=0))
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "review-accept"},
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
        state = load_state(self.paths, "review-accept").state
        self.assertTrue(state.completed)
        self.assertEqual(state.final_status, "approved")
        self.assertEqual(state.last_review.result, "accept")

    def test_builder_progress_is_streamed_to_stderr_only(self) -> None:
        self.write_plan(
            session_id="stderr-progress",
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
        )
        stderr_buffer = io.StringIO()
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=True, summary="builder ok", returncode=0),
            progress_messages=["builder progress line\n"],
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
        self.write_plan(
            session_id="builder-retry",
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
        )
        sleep_calls: list[float] = []
        builder_runner = FakeBuilderRunner(
            BuilderExecResult(ok=False, summary="temporary builder failure", returncode=1),
            BuilderExecResult(ok=True, summary="builder ok", returncode=0),
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

    def test_reviewer_infra_failure_retries_with_backoff(self) -> None:
        self.write_plan(
            session_id="review-retry",
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
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

    def test_codex_builder_runner_resumes_same_thread_after_first_run(self) -> None:
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
            event = '{"type":"thread.started","thread_id":"thread-123"}\n'
            stdout_path.write_text(event, encoding="utf-8")
            stderr_path.write_text("builder stderr\n", encoding="utf-8")
            if stdout_line_handler is not None:
                stdout_line_handler(event)
            return 0

        with mock.patch("codex_workflow.runners._run_streaming_subprocess", side_effect=fake_run_streaming_subprocess):
            first = runner.run(prompt="first prompt", label="first")
            second = runner.run(prompt="second prompt", label="second")

        self.assertEqual(first.thread_id, "thread-123")
        self.assertEqual(second.thread_id, "thread-123")
        self.assertEqual(
            commands[0][:8],
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "--disable", "codex_hooks", "--json"],
        )
        self.assertNotIn("resume", commands[0])
        self.assertEqual(commands[1][:3], ["codex", "exec", "resume"])
        self.assertIn("--sandbox", commands[1])
        self.assertIn("workspace-write", commands[1])
        self.assertIn("thread-123", commands[1])

    def test_stop_recovers_missing_state_file_defensively(self) -> None:
        write_current_session_id(self.paths, "missing-state")

        outcome = handle_stop({"session_id": "missing-state"}, self.repo_root)

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("recovered to planning mode", outcome.stderr_message)
        state = load_state(self.paths, "missing-state").state
        self.assertEqual(state.phase, "planning")

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
            outcome=type("Outcome", (), {"exit_code": 0, "stdout_payload": {"continue": False, "stopReason": "ok"}, "stderr_message": None})(),
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
        self.assertEqual(analysis.missing_sections, [])
        self.assertEqual(analysis.empty_sections, [])
        self.assertEqual(len(analysis.tracked_items), 2)

    def test_legacy_state_without_awaiting_plan_reply_loads_as_false(self) -> None:
        state_path = self.paths.session_state_path("legacy")
        atomic_write_text(
            state_path,
            (
                "{\n"
                '  "version": 1,\n'
                '  "session_id": "legacy",\n'
                '  "phase": "planning",\n'
                '  "completed": false,\n'
                '  "final_status": null,\n'
                '  "plan_doc": "./plans/session-legacy.md",\n'
                '  "last_plan_check": {"result": null, "msg": null, "at": null},\n'
                '  "last_review": {"result": null, "msg": null, "at": null},\n'
                '  "created_at": "2026-01-01T00:00:00Z",\n'
                '  "updated_at": "2026-01-01T00:00:00Z"\n'
                "}\n"
            ),
        )

        state = load_state(self.paths, "legacy").state

        self.assertFalse(state.awaiting_plan_reply)


if __name__ == "__main__":
    unittest.main()
