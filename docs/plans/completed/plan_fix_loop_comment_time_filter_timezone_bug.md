# Fix Loop Reviewer Comment Time Filter to Avoid Timezone Comparison Misses

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `docs/design/index.md`, `docs/design/pr_autoloop_builder_reviewer_contract.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`, `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `.agents/skills/execplan-event-post-completion/scripts/run_event.sh`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

ExecPlan start context:
- Branch at start: `feat/pr-autoloop-skill`
- Commit at start: `082c64da0c227c2d7f059086f3b5a491c0c2270f`
- Target PR: `https://github.com/MachinaIO/mxx/pull/63`

## Purpose / Big Picture

After this fix, `run_builder_reviewer_loop.sh` will no longer miss freshly posted reviewer comments due to timezone-offset string comparison. The script will compare timestamps as epoch seconds, so comments for the current target commit are detected reliably and APPROVE termination can occur immediately.

## Progress

- [x] (2026-03-04 02:40Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_fix_loop_comment_time_filter_timezone_bug.md,docs/prs/active/pr_feat_pr-autoloop-skill.md; verify_events=execplan.pre_creation; worker_type=default; initialize this plan and capture pre-creation linkage/ledger evidence.
- [x] (2026-03-04 02:43Z) action_id=a1; mode=serial; depends_on=a0; file_locks=scripts/run_builder_reviewer_loop.sh; verify_events=action.pr_autoloop,action.tooling; worker_type=default; replace ISO-string time filtering with epoch-second filtering for issue/review comment collection.
- [x] (2026-03-04 02:44Z) action_id=a2; mode=serial; depends_on=a1; file_locks=docs/design/pr_autoloop_builder_reviewer_contract.md; verify_events=action.tooling; worker_type=default; document normalized-time comparison behavior in the design contract.
- [x] (2026-03-04 02:44Z) action_id=a3; mode=serial; depends_on=a2; file_locks=docs/plans/active/plan_fix_loop_comment_time_filter_timezone_bug.md; verify_events=action.tooling,action.pr_autoloop; worker_type=default; run action-level verification and record evidence.
- [x] (2026-03-04 02:49Z) action_id=a4; mode=serial; depends_on=a3; file_locks=docs/plans/active/plan_fix_loop_comment_time_filter_timezone_bug.md,docs/plans/completed/plan_fix_loop_comment_time_filter_timezone_bug.md; verify_events=action.tooling; worker_type=default; finalize this plan and move it to completed.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 02:47Z; finished_at=2026-03-04 02:47Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_pr-autoloop-skill.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 02:48Z; finished_at=2026-03-04 02:48Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=1; status=pass; started_at=2026-03-04 02:48Z; finished_at=2026-03-04 02:48Z; commands=bash -n scripts/run_builder_reviewer_doctor.sh bash -n scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer scripts/run_builder_reviewer_loop.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE scripts/run_builder_reviewer_loop.sh rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|auto_stage_commit_and_push scripts/run_builder_reviewer_loop.sh rg -n fromdateiso8601|%ct scripts/run_builder_reviewer_loop.sh rg -n gh\ api\ graphql scripts/run_builder_reviewer_loop.sh rg -F -n comments(first:100 scripts/run_builder_reviewer_loop.sh rg -F -n reviews(first:100 scripts/run_builder_reviewer_loop.sh rg -n mergedAt|state|OPEN|headRefName scripts/run_builder_reviewer_loop.sh scripts/run_builder_reviewer_doctor.sh rg -n gh\ auth\ status|codex\ login\ status scripts/run_builder_reviewer_doctor.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 02:48Z; finished_at=2026-03-04 02:48Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=2; status=pass; started_at=2026-03-04 02:48Z; finished_at=2026-03-04 02:48Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: The failure was not an API delay but a timestamp representation mismatch (`+09:00` vs `Z`) combined with lexical comparison.
  Evidence: reviewer comments existed on PR #63 with matching `AUTO_TARGET_COMMIT`, yet they were skipped by `createdAt > $since` string filters.

## Decision Log

- Decision: Use commit epoch seconds (`git show --format=%ct`) and jq numeric timestamp comparison (`fromdateiso8601`) for both issue comments and review bodies.
  Rationale: epoch comparison is timezone-agnostic and deterministic for ordering checks.
  Date/Author: 2026-03-04 / Codex
- Decision: Extend `action.pr_autoloop` verification to require presence of epoch-based timestamp filtering markers.
  Rationale: prevents regression to fragile lexical timestamp filtering.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed:

- Updated reviewer comment filtering in `scripts/run_builder_reviewer_loop.sh` to use epoch-based comparison:
  - `LATEST_COMMIT` timestamp source changed to `%ct`,
  - GraphQL comment/review timestamps converted via `fromdateiso8601` before numeric comparison.
- Updated `action.pr_autoloop` verifier to enforce the new timestamp-filter implementation markers.
- Updated design contract documentation with the timezone-safe timestamp rule.
- Passed `action.tooling` and `action.pr_autoloop` verification gates.

Remaining:

- None.

## Context and Orientation

The loop currently uses `git show --format=%cI` and compares that string directly against GraphQL `createdAt/submittedAt` strings inside jq. Because one timestamp may include a timezone offset (`+09:00`) while GraphQL returns `Z`, lexicographic comparison can drop valid newer comments.

## Plan of Work

Change reviewer comment collection to numeric epoch comparison, update the design contract for this invariant, validate with `action.tooling` and `action.pr_autoloop`, then close lifecycle verification.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --plan docs/plans/active/plan_fix_loop_comment_time_filter_timezone_bug.md --event execplan.pre_creation
    # edit scripts/run_builder_reviewer_loop.sh
    # edit docs/design/pr_autoloop_builder_reviewer_contract.md
    scripts/execplan_gate.sh --plan docs/plans/active/plan_fix_loop_comment_time_filter_timezone_bug.md --event action.tooling
    scripts/execplan_gate.sh --plan docs/plans/active/plan_fix_loop_comment_time_filter_timezone_bug.md --event action.pr_autoloop
    mv docs/plans/active/plan_fix_loop_comment_time_filter_timezone_bug.md docs/plans/completed/plan_fix_loop_comment_time_filter_timezone_bug.md
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_fix_loop_comment_time_filter_timezone_bug.md --event execplan.post_completion

## Validation and Acceptance

Acceptance requires all of the following:

1. Comment/review fetch filtering compares numeric timestamps and is not dependent on timezone-string lexical ordering.
2. Reviewer comments for the target commit (with matching `AUTO_TARGET_COMMIT`) are discoverable immediately after posting.
3. `action.tooling` and `action.pr_autoloop` gates pass.

## Idempotence and Recovery

The change is additive and safe: only the time-filter predicate is replaced. Existing retry/iteration bounds remain unchanged.

## Plan Revision Notes

- 2026-03-04 11:39Z: Initial bug-fix plan created for timezone-safe reviewer comment filtering.
- 2026-03-04 02:44Z: Implemented epoch-based timestamp filtering and updated verifier/design docs.
- 2026-03-04 02:49Z: Finalized plan and prepared completed-plan transition.
- 2026-03-04 02:50Z: `execplan.post_completion` passed on first attempt.
- execplan_start_branch: feat/pr-autoloop-skill
- execplan_start_commit: 082c64da0c227c2d7f059086f3b5a491c0c2270f

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 81481b2aa771e52ef5a34b2d7a588f73e6964f46	docs/prs/active/pr_feat_pr-autoloop-skill.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: d6f7daaf793ff86f64d79be0b262904132e68c9e	docs/plans/active/plan_fix_loop_comment_time_filter_timezone_bug.md
<!-- execplan-start-untracked:end -->
