# Align Sub Pre-Execution Scope with PLANS Step 1/Step 2

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/sub_execplan_pre_execution.md`, `docs/verification/docs_only_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

PR tracking document: `docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle_followup_step1_step2_alignment.md`.

ExecPlan start context:

- Branch: `docs/sub-execplan-main-sub-lifecycle`
- Start commit: `19870df254d140a8d5a19007877ddc0bd187969c`

## Purpose / Big Picture

After this change, `PLANS.md` lifecycle step 1 and step 2 will no longer conflict with `docs/verification/sub_execplan_pre_execution.md`. The policy will clearly separate main-plan pre-creation behavior from sub-plan pre-execution behavior so a sub agent can follow one unambiguous path.

## Progress

- [x] (2026-03-02 15:21Z) Ran pre-creation verification checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed branch/PR alignment with PR #58.
- [x] (2026-03-02 15:22Z) Added follow-up PR tracking metadata at `docs/prs/active/pr_docs_sub_execplan_main_sub_lifecycle_followup_step1_step2_alignment.md`.
- [x] (2026-03-02 15:23Z) Updated `PLANS.md` step 1 wording so sub ExecPlan flow explicitly applies before execution (not before plan creation/update).
- [x] (2026-03-02 15:23Z) Updated `PLANS.md` step 2 wording so main/sub plan-document handling is unambiguous (`main` creates; `sub` selects existing parent-defined sub plan).
- [x] (2026-03-02 15:24Z) Ran docs-only verification checks and lifecycle wording consistency checks (`git diff --name-only --`, `rg -n "step 1|step 2|sub ExecPlan|pre-execution|pre-creation" PLANS.md docs/verification -S`, `rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md`).
- [x] (2026-03-02 15:25Z) Finalized this plan, moved it to `docs/plans/completed/`, and ran post-main-ExecPlan verification actions (`rg -n "docs/prs/active/|docs/prs/completed/" ...`, PR tracking review, `gh pr view 58 --json ...`, `gh pr ready 58`).
- [x] (2026-03-02 15:25Z) Recorded readiness decision `ready`, moved follow-up PR tracking file to completed state, and completed final commit/push for this lifecycle.

## Surprises & Discoveries

- Observation: The contradiction is localized to `PLANS.md` step 1/step 2 wording; verification docs already state the intended sub pre-execution scope.
  Evidence: `docs/verification/sub_execplan_pre_execution.md` line 5 says the event is not for sub-plan creation.

## Decision Log

- Decision: Resolve inconsistency by updating `PLANS.md` lifecycle wording (not by broadening `sub_execplan_pre_execution.md` scope).
  Rationale: The user requested aligning PLANS with existing verification docs.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

The lifecycle inconsistency is resolved in `PLANS.md` without changing verification-document scope. Step 1 now distinguishes main pre-creation timing from sub pre-execution timing, and step 2 now explicitly prevents sub agents from being instructed to create a new sub plan after pre-execution verification.

`PLANS.md` now matches `docs/verification/sub_execplan_pre_execution.md` ("not for sub-plan creation"), giving sub agents one unambiguous flow.

Post-main lifecycle closure is complete: PR #58 readiness was revalidated and the follow-up PR tracking file for this scope was moved from `docs/prs/active/` to `docs/prs/completed/`.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none.
- Created/modified: none.

Architecture documents:

- Referenced: none.
- Created/modified: none.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/sub_execplan_pre_execution.md`, `docs/verification/docs_only_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Created/modified: none expected.

## Context and Orientation

`PLANS.md` currently says step 1 runs pre-verification before creating/updating the target plan file, and step 2 then says to add a new plan document for the target. For sub ExecPlans this conflicts with `sub_execplan_pre_execution.md`, which is explicitly pre-execution and not for sub-plan creation.

## Plan of Work

Adjust lifecycle prose in `PLANS.md` step 1 and step 2 to define separate timing semantics: main flow remains pre-creation, while sub flow is pre-execution and depends on an already-created sub-plan document linked by a parent main plan. Preserve the existing intent of main-plan decomposition and sub-plan reporting behavior.

## Concrete Steps

Run from `.`:

    apply_patch << 'PATCH'
    ... update PLANS.md step 1/step 2 wording ...
    PATCH
    git diff --name-only --
    rg -n "step 1|step 2|sub ExecPlan|pre-execution|pre-creation" PLANS.md docs/verification -S
    rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md

## Validation and Acceptance

Acceptance criteria:

1. `PLANS.md` step 1 clearly distinguishes main pre-creation timing and sub pre-execution timing.
2. `PLANS.md` step 2 no longer instructs a sub agent to create a sub plan after pre-execution verification.
3. Wording is consistent with `docs/verification/sub_execplan_pre_execution.md`.
4. Docs-only verification checks are recorded.

## Idempotence and Recovery

This is docs-only wording alignment. If wording remains ambiguous, patch the same lines and rerun the checks.

## Artifacts and Notes

Expected touched files:

    PLANS.md
    docs/plans/completed/plan_align_sub_pre_execution_scope_with_plans_step1_step2.md
    docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle_followup_step1_step2_alignment.md

## Interfaces and Dependencies

No runtime interfaces or dependencies change.

Revision note (2026-03-02, Codex): Initial plan created for step 1/step 2 wording alignment with sub pre-execution scope.
Revision note (2026-03-02, Codex): Updated progress/outcomes after applying step 1/step 2 wording fixes and docs-only verification checks.
Revision note (2026-03-02, Codex): Added post-main verification/readiness evidence and completion-state PR tracking transition.
