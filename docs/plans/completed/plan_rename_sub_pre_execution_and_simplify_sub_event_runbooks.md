# Rename Sub Pre Event File and Simplify Sub Event Runbooks

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/sub_execplan_pre_execution.md`, `docs/verification/sub_execplan_post_completion.md`, `docs/verification/docs_only_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

PR tracking document: `docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle_followup_sub_events.md`.

ExecPlan start context:

- Branch: `docs/sub-execplan-main-sub-lifecycle`
- Start commit: `23eba39907c278aead99c7c93ebabeffe616dbba`

## Purpose / Big Picture

After this change, sub-ExecPlan verification naming and semantics will be aligned with the requested lifecycle meaning. The pre-sub event document will represent execution timing (not creation timing), and both sub pre/post event runbooks will define no required actions beyond following `PLANS.md`.

## Progress

- [x] (2026-03-02 15:04Z) Ran pre-creation verification checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed branch/PR alignment with PR #58.
- [x] (2026-03-02 15:05Z) Created/updated an active PR tracking document for this follow-up scope at `docs/prs/active/pr_docs_sub_execplan_main_sub_lifecycle_followup_sub_events.md`.
- [x] (2026-03-02 15:06Z) Renamed `docs/verification/sub_execplan_pre_creation.md` to `docs/verification/sub_execplan_pre_execution.md`.
- [x] (2026-03-02 15:07Z) Rewrote `sub_execplan_pre_execution.md` as a pre-execution event (not pre-creation) with no required actions.
- [x] (2026-03-02 15:08Z) Rewrote `docs/verification/sub_execplan_post_completion.md` to no required actions and explicit deferral to sub ExecPlan lifecycle behavior in `PLANS.md`.
- [x] (2026-03-02 15:09Z) Updated references in `PLANS.md`, `docs/verification/index.md`, and `docs/verification/main_execplan_pre_creation.md` to use `sub_execplan_pre_execution.md`.
- [x] (2026-03-02 15:10Z) Ran docs-only verification checks and stale-reference checks (`git diff --name-only --`, `rg -n "sub_execplan_pre_creation|sub_execplan_pre_execution|sub_execplan_post_completion" PLANS.md docs/verification -S`, `rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md`).
- [x] (2026-03-02 15:12Z) Finalized this plan, moved it to `docs/plans/completed/`, ran post-main-ExecPlan verification checks (`rg -n "docs/prs/active/|docs/prs/completed/" ...`, PR tracking review, `gh pr view 58 --json ...`, `gh pr ready 58`), and recorded readiness decision `ready`.
- [x] (2026-03-02 15:13Z) Transitioned follow-up PR tracking doc from active to completed (`mv docs/prs/active/pr_docs_sub_execplan_main_sub_lifecycle_followup_sub_events.md docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle_followup_sub_events.md`) and completed final commit/push for this plan state.

## Surprises & Discoveries

- Observation: `docs/verification/index.md` already had an in-progress local edit at plan start, including a partial mention of `sub_execplan_pre_execution.md` with an old link target.
  Evidence: `git status --short` showed `M docs/verification/index.md` before this plan; current text mixes new title with old path.

## Decision Log

- Decision: Keep and build on the existing unstaged `docs/verification/index.md` change rather than reverting it.
  Rationale: User explicitly requested preserving that ongoing change while applying this update.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Core requested edits are complete: the pre-sub file is now `sub_execplan_pre_execution.md`, and both sub runbooks now define no required actions with simplified lifecycle guidance.

Reference alignment is complete in `PLANS.md`, `docs/verification/index.md`, and `docs/verification/main_execplan_pre_creation.md`.

Post-main lifecycle closure is complete for this follow-up plan: PR #58 readiness was revalidated, and the follow-up PR tracking document was moved from `docs/prs/active/` to `docs/prs/completed/`.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none.
- Created/modified: none.

Architecture documents:

- Referenced: none.
- Created/modified: none.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/sub_execplan_pre_creation.md`, `docs/verification/sub_execplan_post_completion.md`, `docs/verification/docs_only_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Created/modified: `docs/verification/sub_execplan_pre_execution.md`, `docs/verification/sub_execplan_post_completion.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`.

## Context and Orientation

Current sub-event verification runbooks still include implementation-heavy required actions. The request is to simplify these runbooks: pre-sub should be a pre-execution event with no required actions, and post-sub should also have no required actions other than following `PLANS.md` lifecycle guidance. The pre-sub file also needs a filename change from `pre_creation` to `pre_execution`.

## Plan of Work

Rename the pre-sub verification file, rewrite both sub runbooks to minimal required-actions semantics, and align all references in index/lifecycle cross-links. Then run docs-only checks, close this plan, execute post-main verification bookkeeping, and persist with commit/push.

## Concrete Steps

Run from `.`:

    mv docs/verification/sub_execplan_pre_creation.md docs/verification/sub_execplan_pre_execution.md
    apply_patch << 'PATCH'
    ... update sub_execplan_pre_execution.md ...
    ... update sub_execplan_post_completion.md ...
    ... update docs/verification/index.md ...
    ... update PLANS.md ...
    ... update docs/verification/main_execplan_pre_creation.md ...
    PATCH
    git diff --name-only --
    rg -n "sub_execplan_pre_creation|sub_execplan_pre_execution|sub_execplan_post_completion" PLANS.md docs/verification -S
    rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md

## Validation and Acceptance

Acceptance criteria:

1. `sub_execplan_pre_creation.md` is renamed to `sub_execplan_pre_execution.md`.
2. `sub_execplan_pre_execution.md` states this is an event before executing a sub ExecPlan and defines no required actions.
3. `sub_execplan_post_completion.md` defines no required actions and explicitly defers to sub ExecPlan lifecycle behavior in `PLANS.md`.
4. Updated references in `PLANS.md` and verification index/docs point to `sub_execplan_pre_execution.md`.
5. Docs-only verification checks are recorded.

## Idempotence and Recovery

This is docs-only work. If references are broken, rerun `rg` checks and patch remaining stale paths.

## Artifacts and Notes

Expected touched files:

    PLANS.md
    docs/verification/index.md
    docs/verification/main_execplan_pre_creation.md
    docs/verification/sub_execplan_pre_execution.md
    docs/verification/sub_execplan_post_completion.md
    docs/plans/completed/plan_rename_sub_pre_execution_and_simplify_sub_event_runbooks.md
    docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle_followup_sub_events.md

## Interfaces and Dependencies

No runtime interfaces or dependencies change.

Revision note (2026-03-02, Codex): Initial plan created for sub pre-event rename and sub runbook simplification.
Revision note (2026-03-02, Codex): Updated progress after renaming/simplifying sub runbooks and aligning cross-references.
Revision note (2026-03-02, Codex): Added post-main verification readiness evidence and completion-state file transitions.
