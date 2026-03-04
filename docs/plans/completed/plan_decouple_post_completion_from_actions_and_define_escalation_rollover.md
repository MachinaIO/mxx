# Decouple Post-Completion Verification From Action Metadata and Define Escalation Rollover

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-action-docs-only/SKILL.md`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, and `scripts/execplan_gate.sh`.

## Purpose / Big Picture

After this change, ExecPlan action metadata will no longer allow `execplan.post_completion` as an action-level verification event, eliminating the lifecycle contradiction where plan completion depended on an event embedded inside action execution metadata. This change also updates escalation policy so a plan that fails the same event three consecutive times is explicitly closed as failed, then retried through a newly created ExecPlan driven by human operator feedback.

## Progress

- [x] (2026-03-03 22:50Z) ran lifecycle pre-creation gate (`execplan.pre_creation`) and recorded linkage metadata for this plan.
- [x] (2026-03-03 22:51Z) action_id=a1; mode=serial; depends_on=none; file_locks=PLANS.md; verify_events=action.docs_only; worker_type=default; updated lifecycle rules to prohibit lifecycle events in action `verify_events` and kept post-completion as a lifecycle-only step.
- [x] (2026-03-03 22:51Z) action_id=a2; mode=serial; depends_on=a1; file_locks=PLANS.md; verify_events=action.docs_only; worker_type=default; revised retry/escalation rules so three consecutive failures force-close the current plan as failed and require a new active ExecPlan for retries.
- [x] (2026-03-03 22:51Z) action_id=a3; mode=serial; depends_on=a2; file_locks=docs/plans/active/plan_decouple_post_completion_from_actions_and_define_escalation_rollover.md; verify_events=action.docs_only; worker_type=default; ran docs-only verification and recorded attempts in `Verification Ledger`.
- [x] (2026-03-03 22:52Z) action_id=a4; mode=serial; depends_on=a3; file_locks=docs/plans/active/plan_decouple_post_completion_from_actions_and_define_escalation_rollover.md,docs/plans/completed/plan_decouple_post_completion_from_actions_and_define_escalation_rollover.md; verify_events=action.docs_only; worker_type=default; finalized plan state and moved this document to `docs/plans/completed/`.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-03 22:50Z; finished_at=2026-03-03 22:50Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --start --runtime-dir .agents/skills/pr-autoloop/runtime --head-branch feat/pr-autoloop-skill --pr-url https://github.com/MachinaIO/mxx/pull/63 capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_pr-autoloop-skill.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.docs_only; attempt=1; status=pass; started_at=2026-03-03 22:51Z; finished_at=2026-03-03 22:51Z; commands=git diff --name-only --relative HEAD -- git ls-files --others --exclude-standard rg -n <placeholder-pattern> <changed-doc-targets>; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.docs_only; attempt=2; status=pass; started_at=2026-03-03 22:53Z; finished_at=2026-03-03 22:53Z; commands=git diff --name-only --relative HEAD -- git ls-files --others --exclude-standard rg -n <placeholder-pattern> <changed-doc-targets>; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `execplan.pre_creation` rewrote the PR tracking metadata file as part of its required setup flow.
  Evidence: `docs/prs/active/pr_feat_pr-autoloop-skill.md` changed immediately after running the gate.

## Decision Log

- Decision: Keep `execplan.pre_creation` and `execplan.post_completion` as lifecycle-only events and explicitly forbid them in action `verify_events`.
  Rationale: This removes circular completion logic and makes action verification boundaries deterministic.
  Date/Author: 2026-03-04 / Codex
- Decision: On three consecutive failures, force-close the current plan as failed and continue only in a newly created plan after human feedback.
  Rationale: This provides explicit closure for failed attempts and avoids indefinite retries in a single stale plan document.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed:

- `PLANS.md` now enforces that action `verify_events` can contain only `action.*` events.
- `execplan.post_completion` is explicitly lifecycle-only and cannot be embedded inside `Progress` action metadata.
- Three consecutive failures now require force-closing the current plan as failed, then retrying through a new operator-seeded plan.
- Required verification evidence for this policy update was captured (`execplan.pre_creation`, `action.docs_only`).

Remaining:

- None.

## Context and Orientation

`PLANS.md` currently treats lifecycle events and action events as one flat verification space in action metadata, while the lifecycle section separately defines `execplan.post_completion` as a terminal lifecycle gate. This plan resolves the ambiguity by defining an explicit boundary: action `verify_events` can contain only `action.*` events; lifecycle events remain outside action metadata and are triggered only by lifecycle steps.

## Plan of Work

First, run pre-creation for this plan to capture branch/PR linkage. Next, edit `PLANS.md` in the lifecycle and escalation sections. Then run docs-only verification, update plan records, and move the plan to completed.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --plan docs/plans/active/plan_decouple_post_completion_from_actions_and_define_escalation_rollover.md --event execplan.pre_creation
    # edit PLANS.md
    scripts/execplan_gate.sh --plan docs/plans/active/plan_decouple_post_completion_from_actions_and_define_escalation_rollover.md --event action.docs_only
    mv docs/plans/active/plan_decouple_post_completion_from_actions_and_define_escalation_rollover.md docs/plans/completed/plan_decouple_post_completion_from_actions_and_define_escalation_rollover.md

## Validation and Acceptance

Acceptance requires all of the following:

1. `PLANS.md` explicitly disallows `execplan.pre_creation` and `execplan.post_completion` in action `verify_events`.
2. `PLANS.md` keeps `execplan.post_completion` as a lifecycle-only step after actions are complete.
3. `PLANS.md` defines that three consecutive failures on the same event force-close the current plan as failed with explicit documentation in that plan.
4. `PLANS.md` requires retry work to continue in a newly created active ExecPlan that references operator feedback and the failed plan.
5. Verification evidence is recorded in this plan `Verification Ledger`.

## Idempotence and Recovery

This change is documentation-only. Reapplying edits is safe as long as the final lifecycle text remains internally consistent.

## Plan Revision Notes

- 2026-03-04 00:00Z: Plan created to update lifecycle policy around post-completion/event escalation semantics.
- 2026-03-03 22:50Z: Recorded pre-creation gate evidence and PR tracking linkage metadata.
- 2026-03-03 22:51Z: Updated `PLANS.md` lifecycle and escalation policy text per requested behavior.
- 2026-03-03 22:51Z: Recorded `action.docs_only` pass evidence for this policy-only change.
- 2026-03-03 22:52Z: Finalized plan and moved it to `docs/plans/completed/`.
- 2026-03-03 22:53Z: Re-ran `action.docs_only` after final PR-tracking doc touch; second attempt passed.
- execplan_start_branch: feat/pr-autoloop-skill
- execplan_start_commit: 9b80754db9e326d373505f4c898c0f0f3469c960

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: c814619469805c08618c7cc4324359809593ca7c	docs/prs/active/pr_feat_pr-autoloop-skill.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 41b00b7be60c004ce7296efd0f873c20bc30c03c	docs/plans/active/plan_decouple_post_completion_from_actions_and_define_escalation_rollover.md
<!-- execplan-start-untracked:end -->
