# Shift Commit/Push Responsibility to Loop Script and Add Interactive Task/PR Resume Inputs

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `REVIEW.md`, `docs/design/index.md`, `docs/design/pr_autoloop_builder_reviewer_contract.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`, `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `.agents/skills/execplan-event-post-completion/scripts/run_event.sh`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

ExecPlan start context:
- Branch at start: `feat/pr-autoloop-skill`
- Commit at start: `2b21a700ccbb38a8019b7db7ffd586d7544d6c5e`
- Target PR: `https://github.com/MachinaIO/mxx/pull/63`

## Purpose / Big Picture

After this change, `execplan.post_completion` verification becomes pure validation and no longer stages/commits/pushes anything. The fixed loop script `scripts/run_builder_reviewer_loop.sh` takes ownership of mechanical git finalization (add non-baseline untracked files, commit, push), supports interactive task input when no task argument is provided, and offers an interactive resume/new selection when `--pr-url` is omitted but active PR tracking documents exist.

## Progress

- [x] (2026-03-04 00:56Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md,docs/prs/active/pr_feat_pr-autoloop-skill.md; verify_events=execplan.pre_creation; worker_type=default; initialize this plan and capture pre-creation linkage/ledger evidence.
- [x] (2026-03-04 01:10Z) action_id=a1; mode=serial; depends_on=a0; file_locks=scripts/run_builder_reviewer_loop.sh; verify_events=action.pr_autoloop,action.tooling; worker_type=default; add loop-owned auto stage/commit/push behavior plus interactive task input and interactive active-PR resume selection.
- [x] (2026-03-04 01:12Z) action_id=a2; mode=serial; depends_on=a1; file_locks=.agents/skills/execplan-event-post-completion/SKILL.md,.agents/skills/execplan-event-post-completion/scripts/run_event.sh; verify_events=action.tooling; worker_type=default; remove git add/commit/push behavior from post-completion verification script and skill description.
- [x] (2026-03-04 01:13Z) action_id=a3; mode=serial; depends_on=a2; file_locks=PLANS.md,docs/design/pr_autoloop_builder_reviewer_contract.md,docs/architecture/scope/automation_orchestration.md; verify_events=action.tooling; worker_type=default; align lifecycle/design/architecture docs with the new ownership boundary and interactive loop behavior.
- [x] (2026-03-04 01:13Z) action_id=a4; mode=serial; depends_on=a3; file_locks=docs/plans/active/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md; verify_events=action.tooling,action.pr_autoloop; worker_type=default; run required action verification gates and record evidence.
- [x] (2026-03-04 01:16Z) action_id=a5; mode=serial; depends_on=a4; file_locks=docs/plans/active/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md,docs/plans/completed/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md; verify_events=action.tooling; worker_type=default; finalize this plan and move it to completed.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 00:55Z; finished_at=2026-03-04 00:55Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_pr-autoloop-skill.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 00:59Z; finished_at=2026-03-04 00:59Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=1; status=pass; started_at=2026-03-04 00:59Z; finished_at=2026-03-04 00:59Z; commands=bash -n scripts/run_builder_reviewer_doctor.sh bash -n scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer scripts/run_builder_reviewer_loop.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE scripts/run_builder_reviewer_loop.sh rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|auto_stage_commit_and_push scripts/run_builder_reviewer_loop.sh rg -n gh\ api\ graphql scripts/run_builder_reviewer_loop.sh rg -F -n comments(first:100 scripts/run_builder_reviewer_loop.sh rg -F -n reviews(first:100 scripts/run_builder_reviewer_loop.sh rg -n mergedAt|state|OPEN|headRefName scripts/run_builder_reviewer_loop.sh scripts/run_builder_reviewer_doctor.sh rg -n gh\ auth\ status|codex\ login\ status scripts/run_builder_reviewer_doctor.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 00:59Z; finished_at=2026-03-04 00:59Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=2; status=pass; started_at=2026-03-04 01:00Z; finished_at=2026-03-04 01:00Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=3; status=pass; started_at=2026-03-04 01:00Z; finished_at=2026-03-04 01:00Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=2; status=pass; started_at=2026-03-04 01:00Z; finished_at=2026-03-04 01:00Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=2; status=pass; started_at=2026-03-04 01:00Z; finished_at=2026-03-04 01:00Z; commands=bash -n scripts/run_builder_reviewer_doctor.sh bash -n scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer scripts/run_builder_reviewer_loop.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE scripts/run_builder_reviewer_loop.sh rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|auto_stage_commit_and_push scripts/run_builder_reviewer_loop.sh rg -n gh\ api\ graphql scripts/run_builder_reviewer_loop.sh rg -F -n comments(first:100 scripts/run_builder_reviewer_loop.sh rg -F -n reviews(first:100 scripts/run_builder_reviewer_loop.sh rg -n mergedAt|state|OPEN|headRefName scripts/run_builder_reviewer_loop.sh scripts/run_builder_reviewer_doctor.sh rg -n gh\ auth\ status|codex\ login\ status scripts/run_builder_reviewer_doctor.sh; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: Once `execplan.post_completion` becomes validation-only, no workflow-owned script remains to persist final staged/untracked outputs unless the loop script handles git finalization directly.
  Evidence: previous post-completion script contained all remaining `git add/commit/push` execution paths.

## Decision Log

- Decision: Move mechanical git finalization into `run_builder_reviewer_loop.sh` cleanup loop (`auto_stage_commit_and_push`) and remove all git mutation from post-completion verification.
  Rationale: satisfies the requested ownership split and keeps lifecycle verification side-effect free.
  Date/Author: 2026-03-04 / Codex
- Decision: Keep `--task` and `--task-file` arguments supported but add interactive stdin fallback only when both are omitted in a TTY session.
  Rationale: preserves existing non-interactive automation while enabling manual CLI startup.
  Date/Author: 2026-03-04 / Codex
- Decision: Add interactive resume-vs-new selection only when `--pr-url` is omitted and active PR tracking docs exist.
  Rationale: this adds explicit operator control without changing non-interactive behavior.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed:

- `scripts/run_builder_reviewer_loop.sh` now:
  - prompts for task text interactively when no `--task/--task-file` is provided,
  - offers interactive resume/new selection when active PR docs exist and `--pr-url` is omitted,
  - stages tracked changes plus non-baseline untracked files, commits when needed, and pushes during cleanup loops.
- `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` is now validation-only (no git add/commit/push).
- `.agents/skills/execplan-event-post-completion/SKILL.md`, `PLANS.md`, `docs/design/pr_autoloop_builder_reviewer_contract.md`, and `docs/architecture/scope/automation_orchestration.md` were aligned with the new ownership boundary.
- Required action gates passed: `action.tooling`, `action.pr_autoloop`.

Remaining:

- None.

## Context and Orientation

The latest loop script already owns approval detection and PR tracking doc completion transition, but post-completion still performs git finalization. This request requires moving all remaining automatic git finalization to the loop script and adding deterministic interactive CLI fallbacks for task input and PR selection when arguments are missing.

## Plan of Work

Implement loop-side mechanical git finalization and interactive entry points, then simplify post-completion verification to read-only lifecycle validation semantics. Update lifecycle/design/architecture documents so long-lived contracts match implementation, run action verification gates, and close lifecycle verification.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --plan docs/plans/active/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md --event execplan.pre_creation
    # edit scripts/run_builder_reviewer_loop.sh
    # edit .agents/skills/execplan-event-post-completion/scripts/run_event.sh
    # edit .agents/skills/execplan-event-post-completion/SKILL.md
    # edit PLANS.md
    # edit docs/design/pr_autoloop_builder_reviewer_contract.md
    # edit docs/architecture/scope/automation_orchestration.md
    scripts/execplan_gate.sh --plan docs/plans/active/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md --event action.tooling
    scripts/execplan_gate.sh --plan docs/plans/active/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md --event action.pr_autoloop
    mv docs/plans/active/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md docs/plans/completed/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md --event execplan.post_completion

## Validation and Acceptance

Acceptance requires all of the following:

1. `execplan.post_completion` script and skill no longer run git add/commit/push.
2. `run_builder_reviewer_loop.sh` performs mechanical git finalization by staging non-baseline untracked files, committing as needed, and pushing.
3. If neither `--task` nor `--task-file` is provided and stdin is interactive, loop script prompts the user for task text interactively.
4. If `--pr-url` is omitted and `docs/prs/active/*.md` exists in an interactive terminal, loop script offers resume-vs-new selection and uses the selected target.
5. Action verification gates (`action.tooling`, `action.pr_autoloop`) pass and are recorded.

## Idempotence and Recovery

The loop script logic will remain bounded by existing retry limits and will avoid touching baseline untracked files recorded at start. If post-completion validation fails, plan rollback behavior remains lifecycle-managed and retries proceed within policy bounds.

## Plan Revision Notes

- 2026-03-04 10:08Z: Initial plan drafted for moving post-completion git finalization into loop script and adding interactive CLI fallbacks.
- 2026-03-04 01:10Z: Implemented loop-side automatic stage/commit/push plus interactive task/resume prompts.
- 2026-03-04 01:12Z: Converted post-completion skill/script to validation-only behavior.
- 2026-03-04 01:13Z: Updated lifecycle/design/architecture docs and passed action-level verification gates.
- 2026-03-04 01:16Z: Moved this plan to completed and `execplan.post_completion` passed on attempt 1.
- 2026-03-04 01:17Z: Re-ran `execplan.post_completion` after final plan-note edits; attempt 2 passed.
- execplan_start_branch: feat/pr-autoloop-skill
- execplan_start_commit: 2b21a700ccbb38a8019b7db7ffd586d7544d6c5e

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 409e761ca78c5bb65f0abbbdb0c62300171e35ad	docs/prs/active/pr_feat_pr-autoloop-skill.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: b645035d149c5732b5d99955e6c54c0cce7e93c4	docs/plans/active/plan_shift_loop_commit_push_and_add_interactive_resume_inputs.md
<!-- execplan-start-untracked:end -->
