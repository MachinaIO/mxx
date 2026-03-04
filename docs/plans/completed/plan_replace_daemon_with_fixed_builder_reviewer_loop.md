# Replace Daemon-Coupled PR Automation With Fixed Builder/Reviewer Loop Scripts

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `REVIEW.md`, `docs/design/index.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`, `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-pre-creation/scripts/run_event.sh`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `.agents/skills/execplan-event-post-completion/scripts/run_event.sh`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

ExecPlan start context:
- Branch at start: `feat/pr-autoloop-skill`
- Commit at start: `a8e6cf5bd48cc2a40ccd907c912b1554f1c2b23e`
- Target PR: `https://github.com/MachinaIO/mxx/pull/63`

## Purpose / Big Picture

After this change, lifecycle verification events no longer start or wait for reviewer automation, and PR builder/reviewer iteration is run only through fixed scripts in `scripts/`. The script accepts a task and optional PR URL, enforces git-cleanup/push loops, creates or reuses PRs deterministically, launches a reviewer pass through Codex, and repeats until approval criteria or configured stop bounds are reached. Legacy daemon-centered skill files are removed.

## Progress

- [x] (2026-03-04 00:06Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_replace_daemon_with_fixed_builder_reviewer_loop.md,docs/prs/active/pr_feat_pr-autoloop-skill.md; verify_events=execplan.pre_creation; worker_type=default; initialized this plan and attached pre-creation ledger/linkage metadata.
- [x] (2026-03-04 00:07Z) action_id=a1; mode=serial; depends_on=a0; file_locks=.agents/skills/execplan-event-pre-creation/SKILL.md,.agents/skills/execplan-event-pre-creation/scripts/run_event.sh; verify_events=action.tooling; worker_type=default; removed reviewer startup and PR auto-creation from pre-creation event behavior and documentation.
- [x] (2026-03-04 00:07Z) action_id=a2; mode=serial; depends_on=a1; file_locks=.agents/skills/execplan-event-post-completion/SKILL.md,.agents/skills/execplan-event-post-completion/scripts/run_event.sh; verify_events=action.tooling; worker_type=default; removed reviewer request/wait and approval parsing from post-completion event behavior and documentation.
- [x] (2026-03-04 00:32Z) action_id=a3; mode=serial; depends_on=a2; file_locks=scripts/run_builder_reviewer_doctor.sh,scripts/run_builder_reviewer_loop.sh; verify_events=action.tooling,action.pr_autoloop; worker_type=default; added fixed-script doctor and builder/reviewer loop with PR URL validation, deterministic commit/push cleanup, GraphQL comment/review collection, and bounded retries.
- [x] (2026-03-04 00:32Z) action_id=a4; mode=serial; depends_on=a3; file_locks=.agents/skills/execplan-event-action-pr-autoloop/SKILL.md,.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh,.agents/skills/pr-autoloop/; verify_events=action.pr_autoloop,action.tooling; worker_type=default; removed obsolete daemon-era `.agents/skills/pr-autoloop/` tree and replaced `action.pr_autoloop` verification to fixed-script checks plus daemon-path rejection.
- [x] (2026-03-04 00:32Z) action_id=a5; mode=serial; depends_on=a4; file_locks=REVIEW.md,docs/design/pr_autoloop_builder_reviewer_contract.md,docs/architecture/scope/automation_orchestration.md,docs/architecture/dependencies/native_and_toolchain.md,.agents/skills/execplan-sandbox-escalation/SKILL.md,.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md; verify_events=action.tooling; worker_type=default; updated policy/design/architecture/escalation docs to fixed-script model and added reviewer requirement for latest failed-by-3 plan handling.
- [x] (2026-03-04 00:32Z) action_id=a6; mode=serial; depends_on=a5; file_locks=docs/plans/active/plan_replace_daemon_with_fixed_builder_reviewer_loop.md; verify_events=action.tooling,action.pr_autoloop; worker_type=default; ran action gates (`action.tooling` pass, `action.pr_autoloop` fail then pass after regex-literal fix) and recorded all attempts in ledger.
- [x] (2026-03-04 00:33Z) action_id=a7; mode=serial; depends_on=a6; file_locks=docs/plans/active/plan_replace_daemon_with_fixed_builder_reviewer_loop.md,docs/plans/completed/plan_replace_daemon_with_fixed_builder_reviewer_loop.md; verify_events=action.tooling; worker_type=default; finalized plan sections and moved this plan to completed.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 00:05Z; finished_at=2026-03-04 00:06Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --start --runtime-dir .agents/skills/pr-autoloop/runtime --head-branch feat/pr-autoloop-skill --pr-url https://github.com/MachinaIO/mxx/pull/63 capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_pr-autoloop-skill.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 00:32Z; finished_at=2026-03-04 00:32Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=1; status=fail; started_at=2026-03-04 00:32Z; finished_at=2026-03-04 00:32Z; commands=bash -n scripts/run_builder_reviewer_doctor.sh bash -n scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer scripts/run_builder_reviewer_loop.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE scripts/run_builder_reviewer_loop.sh rg -n gh\ api\ graphql|comments\(first:100|reviews\(first:100 scripts/run_builder_reviewer_loop.sh rg -n mergedAt|state|OPEN|headRefName scripts/run_builder_reviewer_loop.sh scripts/run_builder_reviewer_doctor.sh rg -n gh\ auth\ status|codex\ login\ status scripts/run_builder_reviewer_doctor.sh; failure_summary=loop script missing issue comment collection query; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=2; status=pass; started_at=2026-03-04 00:32Z; finished_at=2026-03-04 00:32Z; commands=bash -n scripts/run_builder_reviewer_doctor.sh bash -n scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer scripts/run_builder_reviewer_loop.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE scripts/run_builder_reviewer_loop.sh rg -n gh\ api\ graphql scripts/run_builder_reviewer_loop.sh rg -F -n comments(first:100 scripts/run_builder_reviewer_loop.sh rg -F -n reviews(first:100 scripts/run_builder_reviewer_loop.sh rg -n mergedAt|state|OPEN|headRefName scripts/run_builder_reviewer_loop.sh scripts/run_builder_reviewer_doctor.sh rg -n gh\ auth\ status|codex\ login\ status scripts/run_builder_reviewer_doctor.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=2; status=pass; started_at=2026-03-04 00:33Z; finished_at=2026-03-04 00:33Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=3; status=pass; started_at=2026-03-04 00:33Z; finished_at=2026-03-04 00:33Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 00:34Z; finished_at=2026-03-04 00:34Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md clear stale blockers sections mv docs/prs/active/pr_feat_pr-autoloop-skill.md docs/prs/completed/pr_feat_pr-autoloop-skill.md git status --short git add changed plan files git commit -m <finalize-message> git push origin feat/pr-autoloop-skill gh pr view https://github.com/MachinaIO/mxx/pull/63 --json isDraft pr already ready; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `execplan.pre_creation` currently records reviewer-daemon startup command usage directly in verification ledger output.
  Evidence: pre-creation ledger `commands=` field includes `.agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --start ...`.
- Observation: `action.pr_autoloop` attempt 1 failed due unescaped literal `(` in `rg` pattern (`comments(first:100`).
  Evidence: gate stderr showed `regex parse error: unclosed group`; attempt 2 passed after switching to `rg -F`.

## Decision Log

- Decision: lifecycle event scripts no longer run reviewer orchestration or PR creation logic; those responsibilities move to fixed scripts under `scripts/`.
  Rationale: enforce deterministic operator-invoked control and eliminate hidden side effects in lifecycle gates.
  Date/Author: 2026-03-04 / Codex
- Decision: Approval acceptance in loop script requires both `APPROVE` token presence and `AUTO_TARGET_COMMIT` exact equality with the current loop target commit.
  Rationale: prevent stale approval comments from incorrectly terminating the loop.
  Date/Author: 2026-03-04 / Codex
- Decision: Remove `.agents/skills/pr-autoloop/` entirely and treat fixed scripts under `scripts/` as the only runtime implementation.
  Rationale: user explicitly requested no agent-owned daemon startup path and no backward compatibility requirement.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed:

- Removed reviewer startup and PR auto-creation from `execplan.pre_creation`.
- Removed reviewer wait/approval gating from `execplan.post_completion`.
- Added fixed orchestration scripts:
  - `scripts/run_builder_reviewer_doctor.sh`
  - `scripts/run_builder_reviewer_loop.sh`
- Replaced `action.pr_autoloop` verification with fixed-script checks and daemon-era path rejection.
- Deleted daemon-era `.agents/skills/pr-autoloop/` implementation.
- Updated long-lived design/architecture/review/escalation documents to fixed-script model.
- Verified required action events and recorded one remediated `action.pr_autoloop` failure.

Remaining:

- None.

## Context and Orientation

The repository currently couples lifecycle gates (`execplan.pre_creation` and `execplan.post_completion`) to daemon-driven reviewer behavior under `.agents/skills/pr-autoloop/`. The requested model removes that coupling and treats review looping as an explicit operator-invoked script workflow under `scripts/`, with event gates reduced to lifecycle validation and state transitions only.

## Plan of Work

First, simplify lifecycle event scripts so they no longer orchestrate reviewer automation or PR creation. Next, add fixed scripts for doctor checks and loop execution with strict git and PR state handling. Then remove daemon-era skill files and update verification scripts and long-lived docs to the new architecture. Finally run action gates, close the plan, and run post-completion lifecycle verification.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --plan docs/plans/active/plan_replace_daemon_with_fixed_builder_reviewer_loop.md --event execplan.pre_creation
    # edit lifecycle event skills/scripts
    # add scripts/run_builder_reviewer_doctor.sh and scripts/run_builder_reviewer_loop.sh
    # remove .agents/skills/pr-autoloop/
    # update action.pr_autoloop verification and policy/design/architecture docs
    scripts/execplan_gate.sh --plan docs/plans/active/plan_replace_daemon_with_fixed_builder_reviewer_loop.md --event action.tooling
    scripts/execplan_gate.sh --plan docs/plans/active/plan_replace_daemon_with_fixed_builder_reviewer_loop.md --event action.pr_autoloop
    mv docs/plans/active/plan_replace_daemon_with_fixed_builder_reviewer_loop.md docs/plans/completed/plan_replace_daemon_with_fixed_builder_reviewer_loop.md
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_replace_daemon_with_fixed_builder_reviewer_loop.md --event execplan.post_completion

## Validation and Acceptance

Acceptance requires all of the following:

1. `execplan.pre_creation` and `execplan.post_completion` scripts no longer start/wait reviewer automation or auto-create PRs.
2. New fixed loop script supports task input, optional `--pr-url`, PR-open validation, deterministic commit/push cleanup, and bounded iteration retries.
3. `--pr-url` merged/closed inputs fail with explicit input error; open PR input binds execution to the PR head branch.
4. `action.pr_autoloop` verification validates the new scripts and fails if daemon-era files still exist.
5. `.agents/skills/pr-autoloop/` is removed and docs are aligned with the fixed-script model.
6. `REVIEW.md` requires reviewer to flag latest plan in PR when it shows 3-attempt unresolved failure state.
7. Required action verification gates pass and evidence is captured in this plan ledger.

## Idempotence and Recovery

Script and doc updates are idempotent. If lifecycle post-completion fails, follow skill-defined rollback behavior and resume from unfinished actions in this same plan unless escalation policy requires force-closing and creating a new plan.

## Plan Revision Notes

- 2026-03-04 00:00Z: Initial plan created for replacing daemon-based flow with fixed loop scripts and PR URL validation behavior.
- 2026-03-04 00:06Z: Recorded pre-creation pass evidence and marked action `a0` complete.
- 2026-03-04 00:07Z: Completed `a1` and `a2` by decoupling lifecycle event scripts from reviewer daemon and PR auto-creation behavior.
- 2026-03-04 00:32Z: Implemented `a3`/`a4`/`a5`: added fixed scripts, removed daemon-era skill tree, updated verifier logic, and aligned review/design/architecture/escalation docs.
- 2026-03-04 00:32Z: Completed `a6` by running action-level gates and remediating one `action.pr_autoloop` regex failure in the verification script.
- 2026-03-04 00:33Z: Completed `a7` by finalizing this plan and moving it to `docs/plans/completed/`.
- 2026-03-04 00:34Z: `execplan.post_completion` passed on first attempt.
- execplan_start_branch: feat/pr-autoloop-skill
- execplan_start_commit: a8e6cf5bd48cc2a40ccd907c912b1554f1c2b23e

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: a214faa37e82d894c7f61ec372a0b4c95ff0977c	docs/prs/active/pr_feat_pr-autoloop-skill.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 08f0d6a550ccde80e63581a405be01c59adcc018	docs/plans/active/plan_replace_daemon_with_fixed_builder_reviewer_loop.md
<!-- execplan-start-untracked:end -->
