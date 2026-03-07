# Fix reviewer feedback for ExecPlan loop and gates

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated as work proceeds.

This document follows `.agents/skills/eternal-cycler/PLANS.md` from the repository root.

## Purpose / Big Picture

This change repairs four tooling regressions that currently block the autonomous builder/reviewer loop. After the change, the loop will stop creating a new unreviewed commit after reviewer approval, `execplan.post_completion` will only pass when lifecycle finalization is genuinely in the completed-plan state (or on an explicit rollback retry), CPU and GPU verification events will be mapped again, and `execplan.resume` will refresh the exact PR tracking document recorded in the plan instead of silently rewriting a branch-default path. The branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020` must also have an open replacement PR again so the next review cycle has a live target.

## Progress

- [x] (2026-03-07 00:25Z) action_id=a1; mode=serial; depends_on=none; file_locks=.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh,.agents/skills/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/execplan-event-resume/scripts/run_event.sh,.agents/skills/execplan-event-index/references/event_skill_map.tsv,.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/eternal-cycler/assets/default-verification/execplan-event-resume/scripts/run_event.sh,.agents/skills/eternal-cycler/assets/default-verification/execplan-event-index/references/event_skill_map.tsv,eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md; verify_events=action.tooling; worker_type=default; repaired the reviewer-blocked loop and verification tooling, created replacement PR `#69`, refreshed the tracked PR metadata, and validated the reported regressions.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-07 00:15Z; finished_at=2026-03-07 00:16Z; commands=git branch --show-current git status --short gh pr status mkdir -p eternal-cycler-out/prs/active gh pr list --state open --head feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020 --json url,title,state,headRefName,baseRefName,updatedAt --limit 20 write eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md append PR Tracking Linkage to plan capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-07 00:22Z; finished_at=2026-03-07 00:22Z; commands=bash -n .agents/skills/eternal-cycler/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-07 00:26Z; finished_at=2026-03-07 00:26Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md gh pr view https://github.com/MachinaIO/mxx/pull/69 --json url,state git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `execplan.post_completion` plan relocation helpers intentionally normalize active/completed paths through the repository root, so retry-promotion smoke tests must use repo-local temp plan paths instead of arbitrary absolute paths outside the repository.
  Evidence: An early retry smoke test on an off-repo absolute copy promoted the file into `eternal-cycler-out/plans/completed/` under the repository root because `to_plan_style_path()` rewrites plan-style paths relative to `REPO_ROOT`.
- Observation: Replacing the closed branch PR was simpler than reopening it because GitHub accepted a new PR on the same branch immediately.
  Evidence: `gh pr create --base main --head feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020 ...` returned `https://github.com/MachinaIO/mxx/pull/69` while `gh pr view https://github.com/MachinaIO/mxx/pull/68 --json state` continued to report `CLOSED`.

## Decision Log

- Decision: Start a new ExecPlan on the existing review branch instead of editing the previously completed 2026-03-06 plan in place.
  Rationale: The reviewer feedback is a new blocked objective with additional implementation work, validation, and PR-state repair, so it needs its own lifecycle and ledger.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Keep PR-tracking finalization local to the worktree after reviewer approval instead of creating another commit.
  Rationale: The reviewer-approved head must remain the final merge candidate. Recording the completed tracking state locally avoids introducing a new unreviewed commit after `approve_merge=true`.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Allow `execplan.post_completion` to accept an active-path plan only when it is a retry after a failed post-completion attempt, and promote that retry back into `eternal-cycler-out/plans/completed/` before a pass.
  Rationale: This blocks arbitrary active-plan bypasses while preserving the rollback-and-retry lifecycle described in `.agents/skills/eternal-cycler/PLANS.md`.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Patch both the live verification scripts under `.agents/skills/` and the default copies under `.agents/skills/eternal-cycler/assets/default-verification/`.
  Rationale: Future `setup.sh` runs copy from the default-verification assets, so leaving the source copies stale would reintroduce the same regressions.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Create replacement PR `#69` instead of attempting to reuse closed PR `#68`.
  Rationale: The closed PR was no longer a valid review target, and a fresh open PR on the same branch immediately restored the autonomous loop’s review surface without waiting for local commits to be pushed.
  Date/Author: 2026-03-07 / BUILDER agent.

## Outcomes & Retrospective

Completed the reviewer-requested tooling repair on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`. `run_builder_reviewer_loop.sh` now finalizes PR-tracking metadata locally without creating a post-approval commit, `execplan.post_completion` only accepts completed plans or explicit rollback retries and promotes a passing retry back into `eternal-cycler-out/plans/completed/`, `execplan.resume` now writes the plan’s recorded `pr_tracking_doc:` path, and both live/default event maps register `action.cpu_behavior` plus `action.gpu_behavior` again. The branch now has replacement PR `https://github.com/MachinaIO/mxx/pull/69`, and `eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md` records that live metadata.

Verification status: `action.tooling` passed on attempt 1 for this ExecPlan. Additional smoke tests confirmed that an arbitrary active-path post-completion copy now fails, a rollback retry copy passes and is promoted back into `eternal-cycler-out/plans/completed/`, the resume runner updates a custom plan-recorded tracking file without touching the branch-default tracking document, and `execplan_gate.sh` no longer rejects `action.cpu_behavior` or `action.gpu_behavior` as unmapped.

Verification scripts modified:

- `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` to stop creating a post-approval tracking commit and to record `review state: APPROVED`.
- `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` to enforce completed-plan state, support rollback retries safely, and report promoted retry paths back to the gate.
- `.agents/skills/execplan-event-resume/scripts/run_event.sh` to honor the plan-recorded `pr_tracking_doc:` path.
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` to restore `action.cpu_behavior` and `action.gpu_behavior`.
- `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh`, `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-resume/scripts/run_event.sh`, and `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-index/references/event_skill_map.tsv` to keep future `setup.sh` copies aligned with the live fixes.

Verification scripts referenced and left unchanged:

- `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` already enforced lifecycle retry bounds and required-pass checks correctly, so only the event-local logic and event map needed changes.
- `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh` already provided the appropriate shell-syntax verification path for this tooling-only change and required no modification.

## Context and Orientation

The reviewer reported four concrete blockers on the branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`. First, `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` currently moves the PR tracking document to `eternal-cycler-out/prs/completed/` and then commits and pushes that change only after the reviewer has already returned `approve_merge=true`. That produces a new branch head after approval and invalidates the review. Second, `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` currently passes even when given an arbitrary plan copy under `eternal-cycler-out/plans/active/`, so step 8 of the lifecycle can be bypassed without a legitimate completed-plan state. Third, `.agents/skills/execplan-event-index/references/event_skill_map.tsv` and the copied default map under `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-index/references/event_skill_map.tsv` no longer register `action.cpu_behavior` or `action.gpu_behavior`, which breaks `execplan_gate.sh` for existing plans that still use those action IDs. Fourth, `.agents/skills/execplan-event-resume/scripts/run_event.sh` refreshes `eternal-cycler-out/prs/active/pr_<branch>.md` instead of honoring the plan’s recorded `pr_tracking_doc:` field, so resumed plans can keep pointing at stale metadata.

At the start of this work, the active PR tracking file for this branch was `eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md`. It pointed at GitHub PR `https://github.com/MachinaIO/mxx/pull/68`, but `gh pr view` reported that PR as `CLOSED`, and there was no open PR for the branch. This plan therefore needed both code fixes and a live PR replacement so the next reviewer cycle would have a target.

The affected tooling is entirely shell-script based. The core loop logic lives in `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh`. Lifecycle validation is mediated by `.agents/skills/eternal-cycler/scripts/execplan_gate.sh`, which dispatches through `.agents/skills/execplan-event-index/references/event_skill_map.tsv` into repository-local event runners under `.agents/skills/execplan-event-*/scripts/run_event.sh`. The asset copies under `.agents/skills/eternal-cycler/assets/default-verification/` must stay aligned with the live copies so future `setup.sh` runs do not reintroduce the same regression.

## Plan of Work

First, update the builder/reviewer loop so reviewer approval no longer causes a new committed branch head after the final review decision. Then harden the lifecycle event runners: `execplan.post_completion` must distinguish a legitimate completed-plan validation from an arbitrary active-plan copy while still supporting the documented rollback-and-retry path, and `execplan.resume` must resolve its tracking file from the plan’s recorded `pr_tracking_doc:` when present. After that, restore CPU and GPU action-event mappings in both the live event map and the default asset map so old and new ExecPlans can verify again. Once the scripts are patched, refresh the branch PR tracking document to the new live PR link by reopening or recreating the GitHub PR for this branch, then run the tooling gate plus targeted smoke tests that reproduce the reviewer’s failure cases and show they are fixed.

## Concrete Steps

From the repository root:

1. Update `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` so PR tracking finalization no longer creates and pushes an extra post-approval commit.
2. Update `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` and `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh` so active-path post-completion validation only succeeds on a rollback retry and restores the plan to `eternal-cycler-out/plans/completed/` before a passing retry.
3. Update `.agents/skills/execplan-event-resume/scripts/run_event.sh` and `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-resume/scripts/run_event.sh` so resume uses the plan’s recorded `pr_tracking_doc:` path when available.
4. Restore `action.cpu_behavior` and `action.gpu_behavior` in `.agents/skills/execplan-event-index/references/event_skill_map.tsv` and `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-index/references/event_skill_map.tsv`.
5. Reopen or create a replacement PR for branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`, then refresh `eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md` to point at the live PR metadata.
6. Run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260307T001329Z_fix_reviewer_feedback_for_execplan_loop_and_gates.md --event action.tooling`, then run targeted shell smoke tests for the post-completion lifecycle, the restored CPU/GPU event mapping, and the resume tracking-path behavior.

## Validation and Acceptance

Acceptance requires five observable results. The loop must no longer create a new committed branch head after reviewer approval. The post-completion event runner must fail when given an arbitrary active-plan copy, but it must still support rollback retries by moving a failed active retry back to `eternal-cycler-out/plans/completed/` on a later pass. `execplan_gate.sh --event action.cpu_behavior` and `--event action.gpu_behavior` must resolve again when pointed at a plan that lists those events. Resume validation must update the plan’s recorded PR tracking document instead of a branch-default path. Finally, this branch must have an open PR again, and the active PR tracking file must reference that live PR URL.

## Idempotence and Recovery

The shell-script edits are idempotent because repeated runs should keep producing the same path-handling and event-resolution behavior. Reopening or recreating the PR is also safe because the validation step only requires that one open PR exists for the branch and that the tracking document records its current URL and metadata. If any validation step fails, inspect the exact script or tracking file named in the failure output, tighten the implementation, and rerun the failed gate attempt until the three-attempt bound is reached.

## Artifacts and Notes

All verification evidence for lifecycle events will be recorded directly in this plan’s `Verification Ledger`. Additional smoke-test observations that are not gate events will be summarized in `Surprises & Discoveries` and `Outcomes & Retrospective`.

## Interfaces and Dependencies

This task changes repository-local shell tooling and plan/PR-tracking metadata only. It does not change Rust or CUDA implementation behavior, but it does restore the CPU and GPU verification-event IDs that existing plans depend on. GitHub CLI is required to inspect and recreate the branch PR and to refresh the tracked PR metadata with the live PR URL.

Revision note (2026-03-07, BUILDER): Created this ExecPlan after `execplan.pre_creation` passed on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`.
Revision note (2026-03-07, BUILDER): Completed the loop/gate fixes, restored CPU/GPU event mappings, created replacement PR `#69`, and validated the reviewer-reported regressions before finalization.

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md

- execplan_start_branch: feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020

- execplan_start_commit: 52bd2d1d46818fbf4e68779a505e72bc0f897b4e

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 9f9c905e918c401a2ea6173d6094260c580326e5	eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: acc6f630696b7b18e73518c1294b813d3d66dc34	eternal-cycler-out/plans/active/20260307T001329Z_fix_reviewer_feedback_for_execplan_loop_and_gates.md
<!-- execplan-start-untracked:end -->
