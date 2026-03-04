# Reverse Resume/Task Prompt Order in Fixed Loop

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `docs/design/index.md`, `docs/design/pr_autoloop_builder_reviewer_contract.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

ExecPlan start context:
- Branch at start: `feat/pr-autoloop-skill`
- Commit at start: `507a3d9d0ff84220948b032adb1255048c15e2f9`
- Target PR: `https://github.com/MachinaIO/mxx/pull/63`

## Purpose / Big Picture

After this change, interactive loop startup asks whether to resume an existing active PR before asking for task text when both `--pr-url` and `--task/--task-file` are omitted. This makes resume target selection the first decision in startup flow and keeps existing non-interactive behavior unchanged.

## Progress

- [x] (2026-03-04 02:26Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_reverse_resume_task_prompt_order.md,docs/prs/active/pr_feat_pr-autoloop-skill.md; verify_events=none; worker_type=default; initialize this plan and capture pre-creation linkage/ledger evidence.
- [x] (2026-03-04 02:26Z) action_id=a1; mode=serial; depends_on=a0; file_locks=scripts/run_builder_reviewer_loop.sh; verify_events=action.pr_autoloop,action.tooling; worker_type=default; reorder interactive startup flow so active-PR resume selection runs before task prompt.
- [x] (2026-03-04 02:26Z) action_id=a2; mode=serial; depends_on=a1; file_locks=docs/design/pr_autoloop_builder_reviewer_contract.md,docs/architecture/scope/automation_orchestration.md; verify_events=action.tooling; worker_type=default; align long-lived design/architecture docs with the updated interactive prompt order.
- [x] (2026-03-04 02:27Z) action_id=a3; mode=serial; depends_on=a2; file_locks=docs/plans/active/plan_reverse_resume_task_prompt_order.md; verify_events=action.tooling,action.pr_autoloop; worker_type=default; run required action verification gates and record evidence.
- [x] (2026-03-04 02:28Z) action_id=a4; mode=serial; depends_on=a3; file_locks=docs/plans/active/plan_reverse_resume_task_prompt_order.md,docs/plans/completed/plan_reverse_resume_task_prompt_order.md; verify_events=action.tooling; worker_type=default; finalize plan document and move it to completed.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 02:26Z; finished_at=2026-03-04 02:26Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_pr-autoloop-skill.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 02:27Z; finished_at=2026-03-04 02:27Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=1; status=pass; started_at=2026-03-04 02:27Z; finished_at=2026-03-04 02:27Z; commands=bash -n scripts/run_builder_reviewer_doctor.sh bash -n scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer scripts/run_builder_reviewer_loop.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE scripts/run_builder_reviewer_loop.sh rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|auto_stage_commit_and_push scripts/run_builder_reviewer_loop.sh rg -n gh\ api\ graphql scripts/run_builder_reviewer_loop.sh rg -F -n comments(first:100 scripts/run_builder_reviewer_loop.sh rg -F -n reviews(first:100 scripts/run_builder_reviewer_loop.sh rg -n mergedAt|state|OPEN|headRefName scripts/run_builder_reviewer_loop.sh scripts/run_builder_reviewer_doctor.sh rg -n gh\ auth\ status|codex\ login\ status scripts/run_builder_reviewer_doctor.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 02:27Z; finished_at=2026-03-04 02:27Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=2; status=pass; started_at=2026-03-04 02:28Z; finished_at=2026-03-04 02:28Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=3; status=pass; started_at=2026-03-04 02:28Z; finished_at=2026-03-04 02:28Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- None yet.

## Decision Log

- Decision: Treat interactive prompt order as part of the fixed-loop contract and update both script behavior and contract docs together.
  Rationale: prompt ordering determines operator startup flow, so contract docs should match runtime behavior.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed:

- `scripts/run_builder_reviewer_loop.sh` now asks interactive active-PR resume/new selection before fallback task prompt.
- `docs/design/pr_autoloop_builder_reviewer_contract.md` and `docs/architecture/scope/automation_orchestration.md` now document the same prompt order.
- Required action gates passed: `action.tooling`, `action.pr_autoloop`.
- Lifecycle completion gate passed: `execplan.post_completion`.

Remaining:

- None.

## Design/Architecture/Verification Impact Note

- Modified design document: `docs/design/pr_autoloop_builder_reviewer_contract.md` to make the startup prompt order explicit (resume selection before fallback task prompt).
- Modified architecture document: `docs/architecture/scope/automation_orchestration.md` to mirror the same interface-order contract.
- Verification skills/scripts referenced and unchanged: `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `scripts/execplan_gate.sh`, `scripts/execplan_notify.sh`; existing event mappings and procedures already covered this change class without requiring policy updates.

## Context and Orientation

The fixed loop script currently prompts for task text before optional interactive active-PR resume selection. The requested behavior reverses this order so resume/new selection is asked first when both prompts are needed.

## Plan of Work

Move the interactive task prompt execution later so it happens after `prompt_for_resume_target_if_needed`, then update the long-lived contract docs to explicitly describe the new sequence. Validate with `action.tooling` and `action.pr_autoloop` gate events.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --plan docs/plans/active/plan_reverse_resume_task_prompt_order.md --event execplan.pre_creation
    # edit scripts/run_builder_reviewer_loop.sh
    # edit docs/design/pr_autoloop_builder_reviewer_contract.md
    # edit docs/architecture/scope/automation_orchestration.md
    scripts/execplan_gate.sh --plan docs/plans/active/plan_reverse_resume_task_prompt_order.md --event action.tooling
    scripts/execplan_gate.sh --plan docs/plans/active/plan_reverse_resume_task_prompt_order.md --event action.pr_autoloop
    mv docs/plans/active/plan_reverse_resume_task_prompt_order.md docs/plans/completed/plan_reverse_resume_task_prompt_order.md
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_reverse_resume_task_prompt_order.md --event execplan.post_completion

## Validation and Acceptance

Acceptance requires all of the following:

1. In interactive mode with omitted `--pr-url` and with active PR tracking docs, resume/new selection prompt appears before any task-text prompt.
2. Existing non-interactive behavior still requires `--task` or `--task-file`.
3. Design and architecture docs describing the fixed-loop contract reflect the new startup order.
4. `action.tooling` and `action.pr_autoloop` gate events pass and are recorded.

## Idempotence and Recovery

The change is a startup-flow reorder only. If verification fails, revert only the affected startup-order edits and rerun the same gate events.

## Plan Revision Notes

- 2026-03-04 02:26Z: Initial plan drafted for reversing interactive resume/task prompt order.
- 2026-03-04 02:26Z: Implemented prompt-order change in loop script and aligned design/architecture docs.
- 2026-03-04 02:27Z: `action.tooling` and `action.pr_autoloop` gates passed.
- 2026-03-04 02:28Z: Moved plan to `docs/plans/completed/` and prepared post-completion validation.
- 2026-03-04 02:28Z: `execplan.post_completion` gate passed.
- 2026-03-04 02:28Z: Re-ran `execplan.post_completion` after final plan-note updates; attempt 2 passed.
- execplan_start_branch: feat/pr-autoloop-skill
- execplan_start_commit: 507a3d9d0ff84220948b032adb1255048c15e2f9

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: ec1033985d295b075ee7a80efc5cad7cd62b7275	docs/prs/active/pr_feat_pr-autoloop-skill.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 3a9ebe9bbdc14f97f67e2706a96bffc92e38bff8	docs/plans/active/plan_reverse_resume_task_prompt_order.md
<!-- execplan-start-untracked:end -->
