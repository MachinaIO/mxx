# Add Parallel Same-File Conflict Rule to Main/Sub Definition

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/docs_only_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

PR tracking document: `docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle_followup_parallel_conflict_rule.md`.

ExecPlan start context:

- Branch: `docs/sub-execplan-main-sub-lifecycle`
- Start commit: `27218d2375c39840c2f3a568e45474424e898728`

## Purpose / Big Picture

After this change, the `PLANS.md` definition of "parallelizable" sub ExecPlans will explicitly include file-level edit independence and conflict accountability. Parallelizable sub plans must not require simultaneous edits to the same file, and if a conflict still occurs during parallel sub-plan work, the main ExecPlan is responsible for resolving it.

## Progress

- [x] (2026-03-02 15:11Z) Ran pre-creation verification checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed branch/PR alignment with PR #58.
- [x] (2026-03-02 15:12Z) Created/updated active PR tracking metadata for this follow-up scope at `docs/prs/active/pr_docs_sub_execplan_main_sub_lifecycle_followup_parallel_conflict_rule.md`.
- [x] (2026-03-02 15:13Z) Updated `PLANS.md` parallelizable definition to require no simultaneous same-file edit requirement for parallel sub ExecPlans.
- [x] (2026-03-02 15:13Z) Updated `PLANS.md` to state main ExecPlan responsibility for conflict resolution when conflicts still occur during parallel sub-plan execution.
- [x] (2026-03-02 15:14Z) Ran docs-only verification checks and wording/reference checks (`git diff --name-only --`, `rg -n "parallelizable|same file|conflict|main ExecPlan" PLANS.md -S`, `rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md`).
- [x] (2026-03-02 15:16Z) Finalized this plan, moved it to `docs/plans/completed/`, ran post-main-ExecPlan verification actions (`rg -n "docs/prs/active/|docs/prs/completed/" ...`, PR tracking review, `gh pr view 58 --json ...`, `gh pr ready 58`), and recorded readiness decision `ready`.
- [x] (2026-03-02 15:16Z) Moved follow-up PR tracking metadata from active to completed and completed final commit/push for this plan state.

## Surprises & Discoveries

- Observation: none so far.
  Evidence: n/a

## Decision Log

- Decision: Keep this change scoped to `PLANS.md` definition text only, without modifying verification event files.
  Rationale: The user requested a definition-level policy clarification, not execution-command or event-runbook changes.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Requested policy clarification is implemented in `PLANS.md`. The "parallelizable" definition now includes file-level non-overlap expectations for simultaneous sub-plan execution, and conflict ownership is explicitly assigned to the main ExecPlan when parallel edits still conflict.

Lifecycle closure is complete for this follow-up plan: PR #58 readiness was revalidated, and follow-up PR tracking metadata was moved to completed state.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none.
- Created/modified: none.

Architecture documents:

- Referenced: none.
- Created/modified: none.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/docs_only_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Created/modified: none expected.

## Context and Orientation

`PLANS.md` already defines "parallelizable" sub ExecPlans as independent execution without cross-sub-agent communication until completion. The requested follow-up is to make file-edit overlap constraints explicit and to assign responsibility to the parent main ExecPlan when conflicts occur.

## Plan of Work

Edit the "Main and Sub ExecPlans" section in `PLANS.md` by extending the paragraph that defines "parallelizable". Add one sentence that forbids requiring simultaneous edits to the same file for parallelizable sub plans, and one sentence that assigns merge/conflict-resolution responsibility to the main ExecPlan when conflicts still happen. Then run docs-only checks and close this lifecycle.

## Concrete Steps

Run from `.`:

    apply_patch << 'PATCH'
    ... update PLANS.md parallelizable/conflict wording ...
    PATCH
    git diff --name-only --
    rg -n "parallelizable|same file|conflict|main ExecPlan" PLANS.md -S
    rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md

## Validation and Acceptance

Acceptance criteria:

1. `PLANS.md` parallelizable definition explicitly includes "no simultaneous same-file edit requirement".
2. `PLANS.md` explicitly states main ExecPlan responsibility to resolve conflicts arising from parallel sub-plan execution.
3. Docs-only verification checks are recorded.

## Idempotence and Recovery

This is docs-only policy wording. If wording is unclear, edit the same paragraph and rerun the checks.

## Artifacts and Notes

Expected touched files:

    PLANS.md
    docs/plans/completed/plan_add_parallel_same_file_conflict_rule_to_main_sub_definition.md
    docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle_followup_parallel_conflict_rule.md

## Interfaces and Dependencies

No runtime interfaces or dependencies change.

Revision note (2026-03-02, Codex): Initial plan created for parallel same-file/conflict responsibility rule updates in `PLANS.md`.
Revision note (2026-03-02, Codex): Updated progress and outcomes after applying requested policy wording and running docs-only checks.
Revision note (2026-03-02, Codex): Added post-main verification/readiness evidence and completion-state PR tracking transition.
