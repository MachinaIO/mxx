# Move PR Completion Transition From Post-Completion Event to Builder/Reviewer Loop Approval

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `REVIEW.md`, `docs/design/index.md`, `docs/design/pr_autoloop_builder_reviewer_contract.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`, `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `.agents/skills/execplan-event-post-completion/scripts/run_event.sh`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

ExecPlan start context:
- Branch at start: `feat/pr-autoloop-skill`
- Commit at start: `0d81c656b9dd059c187d2a2231a84e0b8aaebbd0`
- Target PR: `https://github.com/MachinaIO/mxx/pull/63`

## Purpose / Big Picture

After this change, `execplan.post_completion` verification will stop handling PR ready/completed transitions. Instead, `scripts/run_builder_reviewer_loop.sh` will own the mechanical PR-document completion transition at the moment reviewer approval is accepted, updating the PR tracking doc to completed state (`review OPEN`) and moving it from `docs/prs/active/` to `docs/prs/completed/`.

## Progress

- [x] (2026-03-04 00:42Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_move_pr_completion_to_loop_approval.md,docs/prs/active/pr_feat_pr-autoloop-skill.md; verify_events=execplan.pre_creation; worker_type=default; initialize this plan and capture pre-creation linkage/ledger evidence.
- [x] (2026-03-04 00:44Z) action_id=a1; mode=serial; depends_on=a0; file_locks=scripts/run_builder_reviewer_loop.sh; verify_events=action.pr_autoloop,action.tooling; worker_type=default; add approval-time PR tracking completion transition in the loop script.
- [x] (2026-03-04 00:51Z) action_id=a2; mode=serial; depends_on=a1; file_locks=.agents/skills/execplan-event-post-completion/SKILL.md,.agents/skills/execplan-event-post-completion/scripts/run_event.sh; verify_events=action.tooling; worker_type=default; remove PR-ready/completed transition behavior from post-completion event implementation and skill text.
- [x] (2026-03-04 00:53Z) action_id=a3; mode=serial; depends_on=a2; file_locks=docs/design/pr_autoloop_builder_reviewer_contract.md,docs/architecture/scope/automation_orchestration.md; verify_events=action.tooling; worker_type=default; align long-lived design/architecture docs with new ownership boundary.
- [x] (2026-03-04 00:54Z) action_id=a4; mode=serial; depends_on=a3; file_locks=docs/plans/active/plan_move_pr_completion_to_loop_approval.md; verify_events=action.tooling,action.pr_autoloop; worker_type=default; execute required action verification events and record attempts.
- [x] (2026-03-04 00:58Z) action_id=a5; mode=serial; depends_on=a4; file_locks=docs/plans/active/plan_move_pr_completion_to_loop_approval.md,docs/plans/completed/plan_move_pr_completion_to_loop_approval.md; verify_events=action.tooling; worker_type=default; finalize plan sections and move this plan to completed.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 00:41Z; finished_at=2026-03-04 00:41Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_pr-autoloop-skill.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 00:45Z; finished_at=2026-03-04 00:45Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=1; status=pass; started_at=2026-03-04 00:45Z; finished_at=2026-03-04 00:45Z; commands=bash -n scripts/run_builder_reviewer_doctor.sh bash -n scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer scripts/run_builder_reviewer_loop.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE scripts/run_builder_reviewer_loop.sh rg -n gh\ api\ graphql scripts/run_builder_reviewer_loop.sh rg -F -n comments(first:100 scripts/run_builder_reviewer_loop.sh rg -F -n reviews(first:100 scripts/run_builder_reviewer_loop.sh rg -n mergedAt|state|OPEN|headRefName scripts/run_builder_reviewer_loop.sh scripts/run_builder_reviewer_doctor.sh rg -n gh\ auth\ status|codex\ login\ status scripts/run_builder_reviewer_doctor.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=fail; started_at=2026-03-04 00:46Z; finished_at=2026-03-04 00:46Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md rollback plan docs/plans/completed/plan_move_pr_completion_to_loop_approval.md -> docs/plans/active/plan_move_pr_completion_to_loop_approval.md; failure_summary=plan still contains incomplete Progress actions; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=2; status=pass; started_at=2026-03-04 00:46Z; finished_at=2026-03-04 00:46Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short git add changed plan files skip unrelated untracked files git commit -m <finalize-message> git push origin feat/pr-autoloop-skill; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `execplan.post_completion` previously bundled three independent responsibilities (lifecycle validation, PR tracking transition, and `gh pr ready`), which made ownership drift easy.
  Evidence: prior script sections managed `EXECPLAN_PR_READY`, moved `docs/prs/active/*` to `docs/prs/completed/*`, and invoked `gh pr ready`.

## Decision Log

- Decision: Move PR tracking completion transition into `scripts/run_builder_reviewer_loop.sh` approval path and execute it mechanically before success exit.
  Rationale: reviewer approval is the only deterministic completion moment in this flow; lifecycle verification should not mutate review readiness state.
  Date/Author: 2026-03-04 / Codex
- Decision: Keep `execplan.post_completion` focused on lifecycle validation plus final persistence (stage/commit/push changed lifecycle files), with explicit failure when plan actions or latest non-lifecycle events remain unresolved.
  Rationale: aligns lifecycle gates with enforcement-only policy and avoids hidden state transitions that can bypass loop ownership.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed:

- Added approval-time PR tracking completion logic to `scripts/run_builder_reviewer_loop.sh`:
  - resolves PR tracking doc by PR URL/branch,
  - writes completion metadata including `review state: OPEN`,
  - moves tracking doc to `docs/prs/completed/`,
  - commits/pushes that document transition when needed.
- Removed PR-ready/completed transition behavior from `execplan.post_completion` script and skill documentation.
- Updated long-lived design/architecture documents so ownership is explicit: loop owns PR tracking completion; lifecycle post-completion does not.
- Passed required action gates: `action.tooling`, `action.pr_autoloop`.

Remaining:

- None.

## Context and Orientation

Current behavior still leaves PR completion responsibilities inside `execplan.post_completion` verification script, while loop approval only stops execution. This creates the exact ownership mismatch requested for removal.

## Plan of Work

Shift completion-transition ownership into loop approval handling, then simplify post-completion event to lifecycle-only completion checks/commit persistence. Update design and architecture documents to keep long-lived contracts aligned, run action verification events, and complete lifecycle closure.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --plan docs/plans/active/plan_move_pr_completion_to_loop_approval.md --event execplan.pre_creation
    # edit scripts/run_builder_reviewer_loop.sh
    # edit .agents/skills/execplan-event-post-completion/scripts/run_event.sh
    # edit .agents/skills/execplan-event-post-completion/SKILL.md
    # edit docs/design/pr_autoloop_builder_reviewer_contract.md
    # edit docs/architecture/scope/automation_orchestration.md
    scripts/execplan_gate.sh --plan docs/plans/active/plan_move_pr_completion_to_loop_approval.md --event action.tooling
    scripts/execplan_gate.sh --plan docs/plans/active/plan_move_pr_completion_to_loop_approval.md --event action.pr_autoloop
    mv docs/plans/active/plan_move_pr_completion_to_loop_approval.md docs/plans/completed/plan_move_pr_completion_to_loop_approval.md
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_move_pr_completion_to_loop_approval.md --event execplan.post_completion

## Validation and Acceptance

Acceptance requires all of the following:

1. `execplan.post_completion` no longer decides PR readiness, edits blocker sections for readiness, moves PR docs to completed, or runs `gh pr ready`.
2. `run_builder_reviewer_loop.sh` performs PR tracking document completion transition when approval is accepted.
3. The loop transition writes completed status including `review OPEN` in PR tracking content and moves the file to `docs/prs/completed/`.
4. Action verification gates `action.tooling` and `action.pr_autoloop` pass with evidence recorded in this plan.
5. Design and architecture documents explicitly describe new responsibility ownership.

## Idempotence and Recovery

The loop transition will be implemented idempotently: if completed PR doc already exists and active doc does not, it should avoid destructive overwrite. If post-completion gate fails, follow skill-defined remediation and rerun until bounded policy resolution.

## Plan Revision Notes

- 2026-03-04 09:42Z: Initial plan drafted for moving PR completion transition from post-completion verification to loop approval.
- 2026-03-04 00:44Z: Implemented approval-path PR tracking completion in `scripts/run_builder_reviewer_loop.sh`.
- 2026-03-04 00:51Z: Removed post-completion ready/completed transitions from skill/script implementation.
- 2026-03-04 00:53Z: Updated design and architecture docs to reflect new ownership boundary.
- 2026-03-04 00:54Z: Action verification gates (`action.tooling`, `action.pr_autoloop`) passed.
- 2026-03-04 00:58Z: First `execplan.post_completion` attempt failed because `a5` was still unchecked; marked `a5` complete before retry.
- 2026-03-04 01:00Z: `execplan.post_completion` attempt 2 passed after re-moving the plan to `docs/plans/completed/`.
- execplan_start_branch: feat/pr-autoloop-skill
- execplan_start_commit: 0d81c656b9dd059c187d2a2231a84e0b8aaebbd0

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: (none)	(none)
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 49487f558f803b9c5574d612a2d05a1893f34e71	docs/plans/active/plan_move_pr_completion_to_loop_approval.md
- start_untracked_file: bb311536e9d4f464c58359837c4ffb8a897b323c	docs/prs/active/pr_feat_pr-autoloop-skill.md
<!-- execplan-start-untracked:end -->
