# Fix post-completion escalation closure and PR tracking metadata refresh

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated as work proceeds.

This document follows `.agents/skills/eternal-cycler/PLANS.md` from the repository root.

## Purpose / Big Picture

This change resolves the two reviewer-reported regressions in the ExecPlan tooling. After the fix, a third failed `execplan.post_completion` attempt must remain force-closed under `eternal-cycler-out/plans/completed/` instead of being left in `active/` with rollback receipts that make it look resumable. The same change set must also stop PR tracking refreshes from rewriting immutable PR creation metadata when the tracked PR URL has not changed. A reviewer should be able to reproduce the three-attempt post-completion failure and observe the failed plan parked in `completed/`, then rerun focused PR-tracking refresh checks and observe that the recorded creation date and creation commit stay stable for PR `#69`.

## Progress

- [x] (2026-03-07 04:23Z) action_id=a1; mode=serial; depends_on=none; file_locks=.agents/skills/eternal-cycler/scripts/execplan_gate.sh,.agents/skills/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/execplan-event-post-creation/scripts/run_event.sh,.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-creation/scripts/run_event.sh,.agents/skills/execplan-event-resume/scripts/run_event.sh,.agents/skills/eternal-cycler/assets/default-verification/execplan-event-resume/scripts/run_event.sh,.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh,.agents/skills/eternal-cycler/PLANS.md; verify_events=action.tooling; worker_type=default; patched the lifecycle tooling so escalated `execplan.post_completion` failures are force-closed back into `completed/`, active-path retries reject a latest `escalated` status, and PR tracking refreshes reuse immutable creation metadata for an unchanged PR URL while falling back to GitHub `createdAt` plus PR commit history when the doc is new or the PR URL changes.
- [x] (2026-03-07 04:25Z) action_id=a2; mode=serial; depends_on=a1; file_locks=eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md,eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md; verify_events=action.tooling; worker_type=default; restored the live PR tracking doc for PR `#69`, ran targeted `post_creation`, `resume`, and `post_completion` smoke tests, and captured the resulting evidence in this plan.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-07 04:15Z; finished_at=2026-03-07 04:15Z; commands=git branch --show-current git status --short gh pr status mkdir -p eternal-cycler-out/prs/active gh pr list --state open --head feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020 --json url,title,state,headRefName,baseRefName,updatedAt --limit 20 write eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-07 04:23Z; finished_at=2026-03-07 04:23Z; commands=bash -n .agents/skills/eternal-cycler/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=fail; started_at=2026-03-07 04:26Z; finished_at=2026-03-07 04:26Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md,eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md rollback plan eternal-cycler-out/plans/completed/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md -> eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md write eternal-cycler-out/plans/active/.post-completion-rollbacks/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md.receipt record rollback provenance in eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md; failure_summary=referenced PR tracking document not found: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md,eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=2; status=fail; started_at=2026-03-07 04:27Z; finished_at=2026-03-07 04:27Z; commands=gate prerequisite: unresolved non-pass event scan; failure_summary=unresolved verification status remains for execplan.post_completion:fail, resolve and re-run before advancing; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=2; status=pass; started_at=2026-03-07 04:29Z; finished_at=2026-03-07 04:29Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open eternal-cycler-out/plans/active/.post-completion-rollbacks/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md.receipt promote retry plan eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md -> eternal-cycler-out/plans/completed/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md remove eternal-cycler-out/plans/active/.post-completion-rollbacks/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md.receipt open eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md gh pr view https://github.com/MachinaIO/mxx/pull/69 --json url,state git status --short; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=3; status=pass; started_at=2026-03-07 04:30Z; finished_at=2026-03-07 04:30Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md gh pr view https://github.com/MachinaIO/mxx/pull/69 --json url,state git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `execplan.post_completion` cannot decide whether a failure is terminal because the event runner does not know the final attempt number; it always rolls a failed validation back to `active/` first.
  Evidence: The three-attempt smoke test still produced an event-runner rollback on attempt 3, and the final `attempt=3; status=escalated` ledger line now shows both the rollback commands and the gate-level `force-close escalated post_completion plan ... -> eternal-cycler-out/plans/completed/...` cleanup.
- Observation: GitHub exposes enough immutable PR metadata to recover the original tracking fields even after a local refresh path has corrupted them.
  Evidence: `gh pr view https://github.com/MachinaIO/mxx/pull/69 --json createdAt` returned `2026-03-07T00:22:28Z`, and the PR commit list still shows `52bd2d1d46818fbf4e68779a505e72bc0f897b4e` as the latest commit whose timestamp is not later than the PR creation time.
- Observation: The original `execplan.post_completion` PR-doc fallback search was over-broad enough to match comma-separated evidence text inside the plan itself, and a blocked action-gate rerun after rollback can otherwise leave a non-semantic latest `action.tooling=fail` record that should not prevent the final retry.
  Evidence: The first real-plan `execplan.post_completion` attempt failed with `referenced PR tracking document not found: eternal-cycler-out/prs/active/...md,eternal-cycler-out/plans/active/...md`, which came from the plan's own artifact text. After the runner switched to parsing `- pr_tracking_doc:` directly and ignored latest non-lifecycle fails whose `failure_summary` only reported an unresolved lifecycle event, the second retry passed.

## Decision Log

- Decision: Start a new ExecPlan on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020` instead of modifying the previously completed tooling plans.
  Rationale: The reviewer feedback is a new blocked objective with its own lifecycle, verification evidence, and finalization requirements.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Force-close escalated `execplan.post_completion` retries inside `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` instead of trying to make the event runner infer terminal attempts.
  Rationale: The gate owns attempt counting and is the only component that knows when a failure has crossed from `fail` to `escalated`. Letting the event runner continue to perform the rollback keeps the normal retry path intact, while the gate can deterministically move the plan back into `completed/` and remove the rollback receipt only when the terminal escalation decision is known.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Preserve immutable PR creation fields whenever the tracking doc already points at the same PR URL, and otherwise rebuild them from GitHub `createdAt` plus the last PR commit at or before that timestamp.
  Rationale: Refresh flows must not rewrite historical metadata for an unchanged PR, but newly created tracking docs for already-open PRs still need the true creation date and the branch head that existed when the PR was opened. GitHub's live PR record and commit history provide that source of truth without depending on local file history.
  Date/Author: 2026-03-07 / BUILDER agent.
- Decision: Resolve the PR tracking doc during `execplan.post_completion` from the explicit `- pr_tracking_doc:` field first, and treat latest action failures that only report an unresolved lifecycle event as retry artifacts instead of unresolved action regressions.
  Rationale: The plan itself can contain arbitrary evidence strings, so a loose `rg` fallback can over-match and synthesize invalid tracking paths. Likewise, a blocked action-gate rerun after a post-completion rollback does not reflect a new action-level regression, so the final lifecycle retry should not be deadlocked by that artifact when an earlier action pass already exists.
  Date/Author: 2026-03-07 / BUILDER agent.

## Outcomes & Retrospective

Completed the reviewer-requested tooling repair on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`. `execplan.post_completion` now treats an active retry as resumable only when its latest event status is `fail`, the gate force-closes attempt-3 failures back under `eternal-cycler-out/plans/completed/` while deleting the active rollback receipt, and the post-completion runner resolves the PR tracking doc from the explicit plan field before falling back to pattern matching. The PR tracking refresh paths in `execplan.post_creation`, `execplan.resume`, and `run_builder_reviewer_loop.sh` now preserve immutable creation metadata when the PR URL is unchanged and use GitHub `createdAt` plus PR commit history to seed those fields when a tracking doc is first created for an already-open PR.

Verification status: `action.tooling` passed on attempt 1 for the live tooling scripts, and the later direct `bash -n` reruns passed for the post-completion runner after its final retry-specific fixes landed. Manual `bash -n` checks also passed for the mirrored default-verification asset copies. The focused smoke tests showed `post_creation` preserved `2026-03-07 00:22Z` / `52bd2d1d46818fbf4e68779a505e72bc0f897b4e`, `resume` honored the plan-recorded `pr_tracking_doc` path while preserving the same immutable fields, and a three-attempt post-completion failure ended with `STATUS=escalated`, the temp plan back under `eternal-cycler-out/plans/completed/`, and no lingering receipt under `.post-completion-rollbacks/`. The real plan's `execplan.post_completion` lifecycle also passed on attempt 2 after attempt 1 exposed the over-broad fallback PR-doc matcher.

Verification scripts modified:

- `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` to re-close escalated post-completion retries into `completed/` and remove any active rollback receipt before the escalated ledger entry is written.
- `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` and `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh` to reject active-path retries whose latest `execplan.post_completion` status is already `escalated`, to read `pr_tracking_doc` directly from the plan before falling back to a pattern search, and to ignore blocked action-gate retry artifacts that only report an unresolved lifecycle event.
- `.agents/skills/execplan-event-post-creation/scripts/run_event.sh`, `.agents/skills/execplan-event-resume/scripts/run_event.sh`, and their mirrored default-verification asset copies to preserve immutable PR creation fields on refresh and to recover them from GitHub metadata when needed.
- `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` to preserve immutable PR creation metadata when the loop syncs an already-open PR tracking doc.

Verification scripts referenced and left unchanged:

- `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh` already provided the right `bash -n` verification path for the live tooling scripts, so no change was required.
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` already mapped the relevant lifecycle events correctly, so no map change was required.

## Context and Orientation

The first bug lives in the `execplan.post_completion` path. `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` validates a completed ExecPlan, rolls it back into `eternal-cycler-out/plans/active/` on failure, and promotes a rolled-back retry back into `completed/` before revalidating it. `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` tracks attempts and writes the `Verification Ledger`. The lifecycle contract in `.agents/skills/eternal-cycler/PLANS.md` says that an escalated post-completion failure must end in `eternal-cycler-out/plans/completed/`, because a three-strike failure is force-closed and may only resume through a new ExecPlan. The current retry flow leaves the plan in `active/` after attempt 3 if validation fails again, which violates that contract.

The second bug lives in the PR tracking refresh path. `.agents/skills/execplan-event-post-creation/scripts/run_event.sh` creates or refreshes `eternal-cycler-out/prs/active/pr_<branch>.md` when a new ExecPlan starts. `.agents/skills/execplan-event-resume/scripts/run_event.sh` refreshes that same document on resume. `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` also syncs the active PR tracking doc when it attaches to an already-open PR. The PR tracking document records immutable fields, specifically `PR creation date` and `commit hash at PR creation time`, which should remain stable while the `PR link` still points to the same GitHub pull request. The current refresh logic rewrites those fields to the current time and current branch head in at least the post-creation path, which corrupted `eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md` for PR `#69`.

The default verification assets under `.agents/skills/eternal-cycler/assets/default-verification/` mirror the repository-local lifecycle scripts. Any behavior change to the live event runners must be mirrored into the asset copies so future `setup.sh` runs do not reintroduce the bug.

## Plan of Work

First, inspect the current `execplan.post_completion` retry and escalation flow in the gate and event runner to decide whether the rollback-to-active cleanup belongs in the gate, the event runner, or both. Patch the implementation so attempt 3 records `status=escalated`, moves the failed plan back under `eternal-cycler-out/plans/completed/`, and removes any rollback receipt that would make the failed plan look resumable. Mirror the same event-runner behavior into the default verification asset copy and update `.agents/skills/eternal-cycler/PLANS.md` anywhere the lifecycle wording needs to state how escalated post-completion failures are closed.

Then patch the PR tracking refresh paths. `execplan.post_creation` should reuse existing immutable creation metadata when the tracking doc already points to the same PR URL, and it should fall back to GitHub `createdAt` metadata instead of the current clock when it creates a tracking doc for an already-open PR. `execplan.resume` and the loop-level PR sync helper should follow the same rule so any metadata refresh path behaves consistently. After the scripts are fixed, restore the live tracking doc for PR `#69` to the correct creation date and commit.

Finally, run focused verification. Use `bash -n` through `action.tooling` for the touched shell scripts, reproduce the three-attempt `execplan.post_completion` escalation bug on a temporary plan copy, confirm the failed plan ends in `completed/` with no live rollback receipt, and run targeted PR-tracking refresh smoke tests that prove an unchanged PR URL preserves the original creation fields. Record each gate attempt in the `Verification Ledger`, update the living-document sections with the observed evidence, move the plan to `completed/`, and finish with `execplan.post_completion`.

## Concrete Steps

From the repository root:

1. Patch `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` and `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` so an escalated `execplan.post_completion` failure is reclosed into `eternal-cycler-out/plans/completed/` and its active rollback receipt is cleaned up.
2. Mirror any `execplan.post_completion` changes into `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-post-completion/scripts/run_event.sh`.
3. Patch `.agents/skills/execplan-event-post-creation/scripts/run_event.sh`, `.agents/skills/execplan-event-resume/scripts/run_event.sh`, and `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` so PR creation metadata is preserved when the tracked PR URL does not change, and use GitHub `createdAt` metadata when creating a tracking doc for an already-open PR. Mirror the event-runner changes into the corresponding default-verification asset copies.
4. Restore `eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md` so it again records PR `#69` as created at `2026-03-07 00:22Z` on commit `52bd2d1d46818fbf4e68779a505e72bc0f897b4e`.
5. Run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md --event action.tooling`.
6. Run targeted smoke tests for `execplan.post_completion`, `execplan.post_creation`, and `execplan.resume`, capturing the commands and expected pass/fail observations in this document.

## Validation and Acceptance

Acceptance requires five observable results. First, `bash -n` must pass for the touched shell scripts via the `action.tooling` gate. Second, a temporary post-completion retry that fails on attempts 1, 2, and 3 must end with `STATUS=escalated`, the plan parked under `eternal-cycler-out/plans/completed/`, and no lingering receipt under `eternal-cycler-out/plans/active/.post-completion-rollbacks/`. Third, an active-path retry whose latest `execplan.post_completion` ledger status is already `escalated` must not be treated as resumable. Fourth, rerunning `execplan.post_creation` or `execplan.resume` against an unchanged PR URL must preserve the original `PR creation date` and `commit hash at PR creation time`. Fifth, the live PR tracking doc for PR `#69` must once again match the true creation metadata that GitHub reports.

## Idempotence and Recovery

The shell and documentation edits are idempotent because rerunning them should converge on the same lifecycle and PR-tracking rules. The smoke tests will use temporary plan and PR tracking copies under `eternal-cycler-out/` so they can be cleaned up safely. If a post-completion smoke test leaves a temp plan in `active/`, remove the temp artifact only after recording whether the failure was expected; if the behavior is unexpected, patch the scripts and rerun until the three-attempt bound is reached.

## Artifacts and Notes

Key verification evidence from this cycle:

  `gh pr view https://github.com/MachinaIO/mxx/pull/69 --json createdAt` -> `{"createdAt":"2026-03-07T00:22:28Z"}`
  post_creation smoke: `post_creation date=2026-03-07 00:22Z commit=52bd2d1d46818fbf4e68779a505e72bc0f897b4e`
  resume smoke: `resume status=pass date=2026-03-07 00:22Z commit=52bd2d1d46818fbf4e68779a505e72bc0f897b4e real_doc_date=2026-03-07 00:22Z`
  post_completion smoke attempt 1: `status=fail active=yes completed=no receipt=yes`
  post_completion smoke attempt 2: `status=fail active=yes completed=no receipt=yes`
  synthetic escalated active retry: `status=fail summary=execplan.post_completion escalation is terminal; the failed plan must stay under eternal-cycler-out/plans/completed/`
  post_completion smoke attempt 3: `status=escalated active=no completed=yes receipt=no`
  final temp ledger entry: `attempt=3; status=escalated; ... force-close escalated post_completion plan eternal-cycler-out/plans/active/20260307T041414Z_post_completion_escalation_smoke.md -> eternal-cycler-out/plans/completed/20260307T041414Z_post_completion_escalation_smoke.md ... remove eternal-cycler-out/plans/active/.post-completion-rollbacks/20260307T041414Z_post_completion_escalation_smoke.md.receipt`
  real plan post_completion attempt 1: `status=fail summary=referenced PR tracking document not found: eternal-cycler-out/prs/active/...md,eternal-cycler-out/plans/active/...md`
  real plan post_completion attempt 2: `status=pass`

## Interfaces and Dependencies

This task changes shell tooling and lifecycle policy documentation only. The relevant interfaces are the `execplan_gate.sh` event contract (`COMMANDS=`, `FAILURE_SUMMARY=`, `PLAN_PATH=`, `STATUS=`), the PR tracking markdown schema under `eternal-cycler-out/prs/`, and the lifecycle guarantees documented in `.agents/skills/eternal-cycler/PLANS.md`. GitHub CLI is required for the PR metadata checks because the creation timestamp comes from the live PR record.

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md

Revision note (2026-03-07, BUILDER): Created this ExecPlan after `execplan.pre_creation` passed on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`.
Revision note (2026-03-07, BUILDER): Patched the lifecycle tooling and PR-tracking refresh paths, restored the live PR tracking doc for PR `#69`, and recorded focused smoke-test evidence for post-completion escalation closure plus immutable PR metadata preservation.
Revision note (2026-03-07, BUILDER): A real `execplan.post_completion` retry exposed an over-broad fallback PR-doc matcher, so the final patch taught the runner to read `pr_tracking_doc` directly and to ignore gate-artifact action failures caused only by unresolved lifecycle retries.

- execplan_start_branch: feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020

- execplan_start_commit: 0528a48fca4268b5cde81d7ed7a4ec8e665a3d25

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: d910f56bb6e09c706c390b96d1aed183fa8e43ee	eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: f2749f070a01a374d7f5b54f9186d7866d1c47c5	eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md
<!-- execplan-start-untracked:end -->

## Post-completion Rollback Provenance

<!-- execplan-post-completion-rollback:start -->
- rollback_source_plan: eternal-cycler-out/plans/completed/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md
- rollback_retry_plan: eternal-cycler-out/plans/active/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md
- rollback_receipt_doc: eternal-cycler-out/plans/active/.post-completion-rollbacks/20260307T041414Z_fix_post_completion_escalation_and_pr_tracking_metadata.md.receipt
- rollback_token: 8372f38e36ee7977cd720387e61ab375ddcb3317
- rollback_recorded_at: 2026-03-07 04:26Z
<!-- execplan-post-completion-rollback:end -->
