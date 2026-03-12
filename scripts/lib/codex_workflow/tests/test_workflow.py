from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

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
from codex_workflow.runners import FinalTestResult, StructuredExecResult
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

    def test_planning_stop_allows_plan_proposal_turn_and_sets_waiting_flag(self) -> None:
        self.write_plan(session_id="plan-accept", phase="planning")

        outcome = handle_stop(
            {
                "session_id": "plan-accept",
                "latest_assistant_message": "Here is the revised plan.\n\nReply with ACCEPT to approve this plan. If you want changes, describe the revisions concretely.",
            },
            self.repo_root,
        )

        self.assertEqual(outcome.exit_code, 0)
        self.assertIsNone(outcome.stdout_payload)
        state = load_state(self.paths, "plan-accept").state
        self.assertEqual(state.phase, "planning")
        self.assertTrue(state.awaiting_plan_reply)

    def test_planning_stop_blocks_when_plan_turn_does_not_end_with_approval_prompt(self) -> None:
        self.write_plan(session_id="nested-accept", phase="planning")
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "nested-accept", "latest_assistant_message": "Here is a draft plan, but I forgot the required ending."},
            self.repo_root,
            exec_runner=exec_runner,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Planning turns must end", outcome.stderr_message)
        self.assertEqual(len(exec_runner.calls), 0)
        state = load_state(self.paths, "nested-accept").state
        self.assertEqual(state.phase, "planning")
        self.assertFalse(state.awaiting_plan_reply)

    def test_planning_stop_blocks_if_reply_was_pending_but_builder_did_not_process_it(self) -> None:
        self.write_plan(session_id="qualified-accept", phase="planning")
        state = load_state(self.paths, "qualified-accept").state
        state.awaiting_plan_reply = True
        save_state(self.paths, state)

        outcome = handle_stop(
            {
                "session_id": "qualified-accept",
                "latest_assistant_message": "I need more time to think about that.",
            },
            self.repo_root,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("A user reply to the current session plan is still pending.", outcome.stderr_message)
        state = load_state(self.paths, "qualified-accept").state
        self.assertEqual(state.phase, "planning")
        self.assertTrue(state.awaiting_plan_reply)

    def test_stop_blocks_implementation_when_ordered_subtask_is_unchecked(self) -> None:
        self.write_plan(
            session_id="impl-unchecked",
            phase="implementation",
            ordered_items=[(False, "Implement the session start hook.")],
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "impl-unchecked"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Implementation is not complete", outcome.stderr_message)
        self.assertEqual(test_runner.calls, [])
        self.assertEqual(exec_runner.calls, [])

    def test_stop_requires_all_follow_up_checkboxes_to_be_checked_before_final_tests(self) -> None:
        self.write_plan(
            session_id="impl-follow-up-open",
            phase="implementation",
            ordered_items=[(True, "Done ordered work.")],
            follow_up_items=[(False, "Fix the reviewer-reported edge case.")],
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))

        outcome = handle_stop(
            {"session_id": "impl-follow-up-open"},
            self.repo_root,
            exec_runner=FakeExecRunner(),
            test_runner=test_runner,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertEqual(test_runner.calls, [])

    def test_test_failure_appends_new_follow_up_subtask(self) -> None:
        plan_path = self.write_plan(
            session_id="tests-fail",
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
        )
        test_runner = FakeTestRunner(
            FinalTestResult(ok=False, summary="workflow unit tests failed", returncode=1)
        )
        exec_runner = FakeExecRunner()

        outcome = handle_stop(
            {"session_id": "tests-fail"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Final tests failed", outcome.stderr_message)
        self.assertEqual(exec_runner.calls, [])
        plan_text = plan_path.read_text(encoding="utf-8")
        self.assertIn("Fix the failing final validation from `scripts/run_tests.sh`", plan_text)
        self.assertIn("- [x] Implement everything.", plan_text)

    def test_reviewer_revision_appends_new_follow_up_subtask(self) -> None:
        plan_path = self.write_plan(
            session_id="review-revision",
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(
            StructuredExecResult(ok=True, result="revision", msg="Handle the malformed state path.")
        )

        outcome = handle_stop(
            {"session_id": "review-revision"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
        )

        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Reviewer requested changes", outcome.stderr_message)
        plan_text = plan_path.read_text(encoding="utf-8")
        self.assertIn("Address reviewer feedback from the final read-only Codex review", plan_text)
        state = load_state(self.paths, "review-revision").state
        self.assertEqual(state.last_review.result, "revision")

    def test_reviewer_accept_completes_the_session(self) -> None:
        self.write_plan(
            session_id="review-accept",
            phase="implementation",
            ordered_items=[(True, "Implement everything.")],
        )
        test_runner = FakeTestRunner(FinalTestResult(ok=True, summary="ok", returncode=0))
        exec_runner = FakeExecRunner(StructuredExecResult(ok=True, result="accept", msg=None))

        outcome = handle_stop(
            {"session_id": "review-accept"},
            self.repo_root,
            exec_runner=exec_runner,
            test_runner=test_runner,
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
