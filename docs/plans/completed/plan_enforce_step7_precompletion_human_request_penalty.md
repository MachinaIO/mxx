# Enforce Immediate-Dismissal Rule for Pre-Step-7 Human Requests

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/docs_only_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

PR tracking document: `docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle.md`.

ExecPlan start context:

- Branch: `docs/sub-execplan-main-sub-lifecycle`
- Start commit: `83fa14f2b1d87ef908526efe41e8609c91c1bf87`

## Purpose / Big Picture

After this change, `PLANS.md` will explicitly state the consequence for violating the lifecycle rule: if an agent asks for human response before all step-7 verification requirements are complete, the agent is immediately dismissed and removed. This strengthens enforcement semantics in the step-7 critical paragraph.

## Progress

- [x] (2026-03-02 14:15Z) Ran pre-creation verification checks on current branch/PR context (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed this follow-up docs scope is aligned with PR #58.
- [x] (2026-03-02 14:16Z) Updated `PLANS.md` important lifecycle paragraph to add immediate-dismissal/removal rule for pre-step-7 human-response requests.
- [x] (2026-03-02 14:16Z) Mapped validation events in actions: run docs-only verification after the text update, then run post-main-ExecPlan verification before final commit/push.
- [x] (2026-03-02 14:18Z) Ran docs-only validation commands and recorded evidence: `git diff --name-only --` shows only `PLANS.md`; `rg -n "TODO|TBD|FIXME" PLANS.md` returned no matches; wording check confirmed line 130 contains the new enforcement sentence.
- [x] (2026-03-02 14:19Z) Finalized this plan, moved it to `docs/plans/completed/`, and executed post-main-ExecPlan verification (`rg -n "docs/prs/active/|docs/prs/completed/" ...`, PR tracking doc review, `gh pr view --json ...`).
- [x] (2026-03-02 14:20Z) Recorded readiness decision as `ready`, ran `gh pr ready`, moved PR tracking document to `docs/prs/completed/`, and prepared final commit/push step.

## Surprises & Discoveries

- Observation: Post-completion readiness evidence changed during this lifecycle because PR #58 started as draft and was transitioned to ready in step 7.
  Evidence: `gh pr view --json isDraft,state` showed `isDraft:true` before `gh pr ready`; command result then confirmed ready-for-review transition.

## Decision Log

- Decision: Reuse the existing docs branch and PR (`docs/sub-execplan-main-sub-lifecycle`, PR #58) instead of creating a new branch/PR.
  Rationale: The requested sentence update is directly aligned with the same documentation-policy scope.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

The requested enforcement sentence was added to `PLANS.md` line 130, explicitly stating immediate dismissal/removal for agents that request human response before all step-7 verification requirements complete.

Docs-only verification completed with positive results (`PLANS.md` as only changed policy file; no `TODO/TBD/FIXME` in `PLANS.md`; required wording present).

Post-main-ExecPlan verification completed: PR tracking document was reviewed, readiness was set to `ready`, `gh pr ready` was executed, and the PR tracking file was moved from `docs/prs/active/` to `docs/prs/completed/`.

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

`PLANS.md` has an important lifecycle paragraph below step 7 that already forbids requesting human response before lifecycle completion. The user asked to extend this paragraph with an explicit disciplinary consequence when that rule is violated.

## Plan of Work

Apply one sentence edit in `PLANS.md` at the important paragraph near lifecycle step 7 to state immediate dismissal/removal for requesting human response before all step-7 verification is complete. Then run docs-only verification commands, complete this plan document, run the post-main-ExecPlan verification event, and only then commit/push.

## Concrete Steps

Run from `.`:

    nl -ba PLANS.md | sed -n '126,133p'
    git diff --name-only --
    rg -n "TODO|TBD|FIXME" PLANS.md
    rg -n "step-7 verification requirements|dismissed and removed|Important:" PLANS.md

## Validation and Acceptance

Acceptance criteria:

1. `PLANS.md` important lifecycle paragraph contains the explicit immediate-dismissal/removal statement.
2. Docs-only verification commands run and are recorded.
3. Post-main-ExecPlan verification is executed and recorded before final commit/push.

## Idempotence and Recovery

This is a docs-only wording change. If wording is incorrect, edit the same paragraph and rerun the verification commands.

## Artifacts and Notes

Expected touched files:

    PLANS.md
    docs/plans/completed/plan_enforce_step7_precompletion_human_request_penalty.md
    docs/prs/completed/pr_docs_sub_execplan_main_sub_lifecycle.md

## Interfaces and Dependencies

No runtime interfaces or dependencies change.

Revision note (2026-03-02, Codex): Initial plan created for the step-7 enforcement sentence update.
Revision note (2026-03-02, Codex): Recorded docs-only and post-main lifecycle verification results, then finalized completion state.
