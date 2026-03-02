# Integrate Main/Sub ExecPlan Policy into Plan and Verification Docs

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/docs_only_changes.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/main_execplan_post_completion.md`, `docs/verification/sub_execplan_pre_creation.md`, and `docs/verification/sub_execplan_post_completion.md`.

PR tracking document: `docs/prs/active/pr_docs_sub_execplan_main_sub_lifecycle.md`.

ExecPlan start context:

- Branch: `docs/sub-execplan-main-sub-lifecycle`
- Start commit: `c8bfa4543157b66fbf0cc0126c57e1cb6f640c66`

## Purpose / Big Picture

After this change, repository policy will explicitly support hierarchical planning: a large `main ExecPlan` can define multiple `sub ExecPlans`, each with its own plan file and lifecycle. The lifecycle and verification docs will also distinguish which pre/post event documents apply to `main` versus `sub` plans so contributors can execute policy consistently.

## Progress

- [x] (2026-03-02 14:03Z) Ran pre-ExecPlan checks: `git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, and `gh pr view --json number,title,body,state,headRefName,baseRefName,url` (expected failure because `main` has no PR).
- [x] (2026-03-02 14:03Z) Switched from `main` to `docs/sub-execplan-main-sub-lifecycle` per pre-creation policy.
- [x] (2026-03-02 14:04Z) Created PR tracking doc `docs/prs/active/pr_docs_sub_execplan_main_sub_lifecycle.md` and recorded draft-PR creation blocker (`gh pr create --draft --fill` requires pushing branch first).
- [x] (2026-03-02 14:09Z) Updated `PLANS.md` to define `main ExecPlan` and `sub ExecPlan`, including sub-plan placement/linkage, decomposition guidance, autonomy rule, and required parallel/serial scheduling semantics in `Progress`.
- [x] (2026-03-02 14:09Z) Updated `PLANS.md` lifecycle steps 1-7 for main/sub split: distinct pre/post verification docs, optional sub-plan creation in step 2, main-only validation mapping in step 3, step-5 action updates that may create new sub plans, and step-7 sub-to-main reporting requirements.
- [x] (2026-03-02 14:10Z) Renamed verification pre/post files to main-only names and rewrote their wording for explicit main-plan scope.
- [x] (2026-03-02 14:11Z) Added `docs/verification/sub_execplan_pre_creation.md` and `docs/verification/sub_execplan_post_completion.md` to make step-1/step-7 sub-plan verification targets explicit.
- [x] (2026-03-02 14:11Z) Updated `docs/verification/index.md` event map and cross-references to the new main/sub verification filenames.
- [x] (2026-03-02 14:12Z) Ran docs-only validation checks: `git diff --name-only --` (docs-only paths), `rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md` (no stale placeholders in touched files), and targeted filename/lifecycle `rg` checks.
- [x] (2026-03-02 14:16Z) Finalized this plan, moved it to `docs/plans/completed/`, and executed post-main-ExecPlan verification (`rg -n "docs/prs/active/|docs/prs/completed/" ...`, PR tracking doc review, `gh pr status`). Readiness decision: `not ready` because no PR exists yet for this branch.

## Surprises & Discoveries

- Observation: The current verification filenames (`execplan_pre_creation.md` and `execplan_post_completion.md`) are semantically generic, so introducing `sub ExecPlan` lifecycle requires explicit naming to avoid ambiguity.
  Evidence: `docs/verification/index.md` currently maps "Before creating a new ExecPlan" and "After completing an ExecPlan" without main/sub distinction.
- Observation: Post-completion PR readiness could not advance to ready-for-review because this branch still has no PR object on GitHub.
  Evidence: `gh pr status` reported "There is no pull request associated with [docs/sub-execplan-main-sub-lifecycle]".

## Decision Log

- Decision: Treat this change as documentation policy integration only, without creating a new design artifact under `docs/design/`.
  Rationale: The requested updates define process policy within existing meta-rule documents (`PLANS.md` and `docs/verification/*`), not a reusable product/system behavior contract.
  Date/Author: 2026-03-02 / Codex
- Decision: Add explicit sub-plan pre/post verification runbooks (`sub_execplan_pre_creation.md` and `sub_execplan_post_completion.md`) instead of describing sub-plan lifecycle implicitly.
  Rationale: The user requested clear distinction between main/sub lifecycle verification documents in steps 1 and 7; dedicated sub runbooks make that distinction executable.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

This change introduced explicit hierarchical planning policy in `PLANS.md`: `main ExecPlan` and `sub ExecPlan` concepts, mandatory sub-plan linkage/mapping rules, main-plan execution-topology requirements, and autonomous sub-to-main completion handoff.

Verification policy now has explicit main/sub event routing. Legacy pre/post runbooks were renamed to `main_execplan_pre_creation.md` and `main_execplan_post_completion.md`, and new sub runbooks were added for sub-plan lifecycle entry/exit.

No runtime code behavior changed; this is documentation-policy integration only.

Post-main-ExecPlan verification was executed and recorded. PR readiness remained `not ready` because branch push and draft PR creation are still pending.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: `DESIGN.md`, `docs/design/index.md`.
- Created/modified: none expected.

Architecture documents:

- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md` (impact check only).
- Created/modified: none expected.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/docs_only_changes.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/main_execplan_post_completion.md`, `docs/verification/sub_execplan_pre_creation.md`, `docs/verification/sub_execplan_post_completion.md`.
- Created/modified: `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/main_execplan_post_completion.md`, `docs/verification/sub_execplan_pre_creation.md`, `docs/verification/sub_execplan_post_completion.md`.

## Context and Orientation

`PLANS.md` currently defines a single-plan lifecycle and does not explain how to split a large plan into smaller executable units. The verification runbooks under `docs/verification/` currently use filenames that imply universal applicability, but the requested policy requires these pre/post runbooks to be explicitly `main ExecPlan` runbooks. This plan updates lifecycle semantics and filename conventions so sub-plan execution can remain autonomous while feeding progress back to the parent plan.

## Plan of Work

Edit `PLANS.md` to add a dedicated section that defines `main ExecPlan` and `sub ExecPlan`, mandates one sub-plan document per sub-plan under `docs/plans/active`, requires sub plans to link back to their main plan path, and requires explicit mapping from sub plans to main-plan scope. In the same file, update lifecycle steps 1–7 to separate main/sub verification documents and clarify where event mapping and progress handoff occur. Rename verification pre/post files to main-only names, add sub-plan pre/post verification runbooks, and update `docs/verification/index.md` references accordingly.

## Concrete Steps

Run from `.`:

    apply_patch << 'PATCH'
    ... update PLANS.md ...
    PATCH

    mv docs/verification/execplan_pre_creation.md docs/verification/main_execplan_pre_creation.md
    mv docs/verification/execplan_post_completion.md docs/verification/main_execplan_post_completion.md

    apply_patch << 'PATCH'
    ... update renamed main files, add sub files, and update docs/verification/index.md ...
    PATCH

    rg -n "execplan_pre_creation|execplan_post_completion|main ExecPlan|sub ExecPlan|ExecPlan Lifecycle" PLANS.md docs/verification -S
    git diff --name-only --
    rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md

## Validation and Acceptance

Acceptance criteria:

1. `PLANS.md` defines `main ExecPlan` and `sub ExecPlan` with decomposition, linking, scheduling, and autonomy/reporting rules.
2. Lifecycle steps 1 and 7 explicitly distinguish verification documents for main vs sub plans.
3. Lifecycle step updates requested by the user (steps 2, 3, 5, and 7 behavior) are explicitly reflected in `PLANS.md`.
4. Verification pre/post filenames are renamed to main-only names and all touched references are updated.
5. Docs-only validation commands run successfully and no stale placeholders remain in touched policy docs.
6. Main/sub verification runbook routing is executable from `docs/verification/index.md` without ambiguity.

## Idempotence and Recovery

This work is docs-only and can be retried safely. If a rename introduces broken references, rerun `rg` over `docs/verification` and `PLANS.md`, then patch the remaining stale filenames.

## Artifacts and Notes

Expected touched files:

    PLANS.md
    docs/verification/index.md
    docs/verification/main_execplan_pre_creation.md
    docs/verification/main_execplan_post_completion.md
    docs/verification/sub_execplan_pre_creation.md
    docs/verification/sub_execplan_post_completion.md
    docs/plans/completed/plan_main_sub_execplan_policy_integration.md
    docs/prs/active/pr_docs_sub_execplan_main_sub_lifecycle.md

## Interfaces and Dependencies

No runtime code interfaces change. This plan updates documentation contracts that govern planning and verification workflows for contributors and agents.

Revision note (2026-03-02, Codex): Initial plan created for main/sub ExecPlan policy integration, verification filename renaming, and lifecycle alignment.
Revision note (2026-03-02, Codex): Updated plan after implementing main/sub lifecycle policy, adding sub verification runbooks, and running docs-only validation checks.
Revision note (2026-03-02, Codex): Added post-main-ExecPlan verification result and readiness decision (`not ready`), then marked lifecycle completion for this plan state.
