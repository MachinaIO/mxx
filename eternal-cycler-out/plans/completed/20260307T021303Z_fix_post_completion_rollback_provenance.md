# Fix post-completion rollback provenance validation

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated as work proceeds.

This document follows `.agents/skills/eternal-cycler/PLANS.md` from the repository root.

## Purpose / Big Picture

This change closes the remaining reviewer-reported bypass in `execplan.post_completion`. After the change, an active-path plan will only be accepted as a retry when the previous failed post-completion run actually rolled that same plan back out of `eternal-cycler-out/plans/completed/`; simply copying a completed plan into `eternal-cycler-out/plans/active/` and appending a fake failed ledger line must no longer pass. A reviewer should be able to reproduce the previous copy-and-edit bypass, observe that it now fails, then observe that a real rollback retry still passes and gets promoted back into `eternal-cycler-out/plans/completed/`.

## Progress

- [x] (2026-03-07 02:20Z) action_id=a1; mode=serial; depends_on=none; file_locks=.agents/skills/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/execplan-event-post-completion/SKILL.md,.agents/skills/eternal-cycler/PLANS.md; verify_events=action.tooling; worker_type=default; added rollback provenance enforcement to the live and default `execplan.post_completion` runners, documented the provenance-backed retry contract, and verified that a copied active-path fake retry now fails while a real rollback retry still passes through the gate and returns to `completed/`.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-07 02:13Z; finished_at=2026-03-07 02:13Z; commands=git branch --show-current git status --short gh pr status mkdir -p eternal-cycler-out/prs/active gh pr list --state open --head feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020 --json url,title,state,headRefName,baseRefName,updatedAt --limit 20 write eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md append PR Tracking Linkage to plan capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-07 02:21Z; finished_at=2026-03-07 02:21Z; commands=bash -n .agents/skills/eternal-cycler/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-07 02:22Z; finished_at=2026-03-07 02:22Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md gh pr view https://github.com/MachinaIO/mxx/pull/69 --json url,state git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: The current retry admission check only consults the latest `execplan.post_completion` ledger status, so any copied plan can impersonate a rollback retry by editing text inside the plan itself.
  Evidence: The reviewer reproduced the bypass by copying `eternal-cycler-out/plans/completed/20260307T001329Z_fix_reviewer_feedback_for_execplan_loop_and_gates.md` into `eternal-cycler-out/plans/active/`, appending a failed `execplan.post_completion` ledger entry, and rerunning the event script successfully.
- Observation: A real rollback retry must be exercised through `.agents/skills/eternal-cycler/scripts/execplan_gate.sh`, not by calling the event runner alone, because the gate is what appends the failed `execplan.post_completion` ledger entry after the event rolls the plan back to `active/`.
  Evidence: Directly rerunning `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` on a freshly rolled-back active plan still reports "requires a completed plan path" until the gate has appended the failed attempt record. Running the gate on the same temp plan produced `STATUS=fail` on the rollback attempt and `STATUS=pass` on the next retry.

## Decision Log

- Decision: Start a new ExecPlan on the existing review branch instead of reopening the just-completed 2026-03-07 tooling plan.
  Rationale: The reviewer feedback is a new blocked objective with its own implementation and verification evidence, so it needs a separate lifecycle and ledger.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Require both an in-plan rollback provenance block and a sidecar receipt under `eternal-cycler-out/plans/active/.post-completion-rollbacks/` before admitting an active-path retry.
  Rationale: The plan file itself is editable text, so the retry check needs provenance that is minted by the rollback operation and stored outside the plan as well. Matching the block and receipt prevents the old "append a fake fail ledger line" bypass while preserving the documented rollback-and-retry lifecycle.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Keep the rollback provenance block in the plan after a successful retry but delete the sidecar receipt.
  Rationale: Keeping the block preserves a human-readable audit trail in the plan, while deleting the receipt ensures a copied completed plan cannot be reused as a valid active retry without a fresh rollback.
  Date/Author: 2026-03-07 / BUILDER agent.

## Outcomes & Retrospective

Completed the reviewer-requested provenance repair on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`. The live and default `execplan.post_completion` runners now mint rollback provenance when they move a failed completed plan back into `eternal-cycler-out/plans/active/`, and active-path retries are admitted only when that provenance still matches the current retry file. `.agents/skills/execplan-event-post-completion/SKILL.md` and `.agents/skills/eternal-cycler/PLANS.md` now document that provenance-backed retry requirement.

Verification status: `action.tooling` passed on attempt 1 for this ExecPlan. The copied-plan bypass now fails with `FAILURE_SUMMARY=missing rollback provenance for active post_completion retry; rerun from the completed plan path to create a real rollback`. A real rollback smoke test on a temporary completed-plan copy failed through the gate with `PLAN_PATH=eternal-cycler-out/plans/active/20260307T021303Z_real_post_completion_retry_smoke.md`, created `eternal-cycler-out/plans/active/.post-completion-rollbacks/20260307T021303Z_real_post_completion_retry_smoke.md.receipt`, and then passed on the next gate retry after the injected incomplete checkbox was fixed.

Verification scripts modified:

- `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` to write and validate rollback provenance and to delete the sidecar receipt after a successful retry.
- `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh` to keep future `setup.sh` installs aligned with the live provenance fix.

Verification scripts referenced and left unchanged:

- `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` already handled attempt tracking and plan-path updates correctly; it only needed to consume the event runner's `PLAN_PATH=` output.
- `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh` already provided the appropriate shell-syntax validation path for this tooling-only change and required no modification.

## Context and Orientation

The bug lives in `.agents/skills/execplan-event-post-completion/scripts/run_event.sh`, which is the repository-local verification script dispatched by `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` for `execplan.post_completion`. The same script is also copied from `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh` when the default verification assets are reinstalled, so both copies must stay aligned.

`execplan.post_completion` is the last lifecycle gate described in `.agents/skills/eternal-cycler/PLANS.md`. On a failed validation attempt against a completed plan, the script is supposed to move the plan back into `eternal-cycler-out/plans/active/` so the builder can fix the issue and retry. The current implementation checks only whether the active-path plan's latest `execplan.post_completion` ledger entry says `fail` or `escalated`, which is insufficient because the ledger is plain text inside the plan file and can be synthesized by copying a completed plan.

The fix therefore needs an explicit rollback provenance artifact that is written only when the event script performs an actual rollback and is checked before an active-path retry is promoted back into `completed/`. The lifecycle docs should explain that requirement so future changes preserve the same contract.

## Plan of Work

First, patch the live `execplan.post_completion` runner so a failing completed-plan validation writes an explicit rollback provenance record alongside the active-path retry, and so a retry is admitted only when that provenance record exists and matches the plan being retried. Then mirror the same logic into the default-verification asset copy. After that, update `.agents/skills/execplan-event-post-completion/SKILL.md` and `.agents/skills/eternal-cycler/PLANS.md` so the documented rollback flow matches the new provenance requirement. Finally, run the tooling gate and targeted shell smoke tests that prove the copied-plan bypass now fails while a real rollback retry still succeeds and gets promoted back into `completed/`.

## Concrete Steps

From the repository root:

1. Update `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` to write and validate rollback provenance for active-path retries.
2. Mirror the same logic into `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh`.
3. Update `.agents/skills/execplan-event-post-completion/SKILL.md` and `.agents/skills/eternal-cycler/PLANS.md` so the lifecycle documentation explains the provenance-backed rollback retry requirement.
4. Run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260307T021303Z_fix_post_completion_rollback_provenance.md --event action.tooling`.
5. Run targeted shell smoke tests that show an arbitrary copied active-path plan fails and a real rollback retry succeeds and is promoted back into `eternal-cycler-out/plans/completed/`.

## Validation and Acceptance

Acceptance requires three observable results. First, rerunning `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` on an arbitrary active-path copy with a synthetic failed ledger entry must now return `STATUS=fail`. Second, forcing a real post-completion validation failure on a completed plan must move that plan into `eternal-cycler-out/plans/active/`, emit rollback provenance, and allow a later retry on that same rolled-back file to return `STATUS=pass` and move the plan back into `eternal-cycler-out/plans/completed/`. Third, `action.tooling` must pass for the touched shell scripts and docs.

## Idempotence and Recovery

The shell and documentation edits are idempotent because rerunning them should converge on the same rollback-provenance checks and lifecycle wording. The smoke tests will operate on temporary copies under `eternal-cycler-out/plans/active/` and `eternal-cycler-out/plans/completed/` so they can be cleaned up safely. If a validation attempt fails, inspect the event output, adjust the provenance handling, and rerun the failed gate or smoke test until the three-attempt bound is reached.

## Artifacts and Notes

All lifecycle gate attempts will be recorded in this plan's `Verification Ledger`. The key smoke-test evidence from this cycle is:

  fake copied retry: `STATUS=fail` with `FAILURE_SUMMARY=missing rollback provenance for active post_completion retry; rerun from the completed plan path to create a real rollback`
  real rollback setup through the gate: `PLAN_PATH=eternal-cycler-out/plans/active/20260307T021303Z_real_post_completion_retry_smoke.md` plus a matching receipt under `.post-completion-rollbacks/`
  real rollback retry through the gate: `STATUS=pass` and the temporary plan returned to `eternal-cycler-out/plans/completed/`

## Interfaces and Dependencies

This task changes repository-local shell tooling and lifecycle policy text only. The relevant interfaces are the `execplan.post_completion` event runner output fields (`COMMANDS=`, `FAILURE_SUMMARY=`, `PLAN_PATH=`, `STATUS=`) and the lifecycle contract in `.agents/skills/eternal-cycler/PLANS.md`. No Rust or CUDA code paths are involved.

Revision note (2026-03-07, BUILDER): Created this ExecPlan after `execplan.pre_creation` passed on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`.
Revision note (2026-03-07, BUILDER): Implemented rollback provenance via a plan block plus sidecar receipt, updated the lifecycle docs, and captured targeted smoke-test evidence for fake and real retry paths.

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md

- execplan_start_branch: feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020

- execplan_start_commit: db6c46bb2d0158305a8ccebdac27db2af1d138bc

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: ebffc5b6fdb1ce15ce0a081eb05ec22ddae94733	eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: ed5ace9a5aafa5776ec894f45e7fd94463392f6a	eternal-cycler-out/plans/active/20260307T021303Z_fix_post_completion_rollback_provenance.md
<!-- execplan-start-untracked:end -->
