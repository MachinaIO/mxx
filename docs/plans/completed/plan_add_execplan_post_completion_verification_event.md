# Add Verification Event for After ExecPlan Completion

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

ExecPlan start context:
- Branch at start: `feat/harness_enginnering`
- Commit at start: `2c202a2`
- PR tracking document: `docs/prs/active/pr_feat_harness_enginnering.md`

Repository-document context used for this plan: `PLANS.md`, `VERIFICATION.md`, `docs/verification/index.md`, and `docs/verification/execplan_pre_creation.md`.

## Purpose / Big Picture

After this change, the verification framework will include a dedicated event for the stage after an ExecPlan is completed. Agents will have concrete instructions to evaluate whether the linked PR scope is actually ready for review and, if ready, transition both the GitHub PR state and PR tracking document state.

## Progress

- [x] (2026-03-02 02:39Z) Completed pre-ExecPlan checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, and `gh pr status` fallback due to missing CLI).
- [x] (2026-03-02 02:39Z) Added/updated PR tracking context document at `docs/prs/active/pr_feat_harness_enginnering.md` for current branch reuse.
- [x] (2026-03-02 02:38Z) Created `docs/verification/execplan_post_completion.md` with required post-completion PR readiness and state-transition actions.
- [x] (2026-03-02 02:38Z) Updated `docs/verification/index.md` to map the new event.
- [x] (2026-03-02 02:39Z) Ran docs-only verification checks and recorded outcomes (`git diff --name-only --`, `rg -n "TODO|TBD|FIXME" ...`, and verification document content checks).
- [x] (2026-03-02 02:39Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: GitHub CLI is unavailable locally.
  Evidence: `gh pr status` returned `/bin/bash: gh: command not found`.

## Decision Log

- Decision: Include conditional GitHub CLI commands with explicit manual web-UI fallback for PR ready-for-review transition.
  Rationale: Event requirements include GitHub PR state transitions, but local CLI tooling may be unavailable.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

A new post-ExecPlan completion verification event is now available. It explicitly requires: (1) reading the PR document linked from the completed ExecPlan, (2) deciding ready-for-review status based on achieved scope, and (3) if ready, transitioning PR state to ready-for-review and moving the corresponding PR tracking file from `docs/prs/active/` to `docs/prs/completed/`.

The verification index now includes this event, making it discoverable from the required entrypoint.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none.
- Created/modified: none.

Architecture documents:

- Referenced: none.
- Created/modified: none.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`.
- Modified:
  - `docs/verification/index.md`
- Created:
  - `docs/verification/execplan_post_completion.md`

## Context and Orientation

Current verification event documents cover pre-ExecPlan, docs-only, CPU behavior, and GPU/CUDA behavior. There is no event document for the post-ExecPlan completion gate that validates PR readiness and transitions PR/doc state from active to completed.

## Plan of Work

Create a new verification event document for the post-ExecPlan completion stage. It will require reading the PR tracking document linked by the plan, deciding whether PR scope is fully achieved and ready for review, and when ready, marking the PR as ready for review and moving the PR tracking file from `docs/prs/active/` to `docs/prs/completed/`. Then update the verification index to include this event.

## Concrete Steps

Run from repository root (`.`):

    cat > docs/verification/execplan_post_completion.md << '...'
    apply_patch << 'PATCH'
    ...
    PATCH
    sed -n '1,320p' docs/verification/execplan_post_completion.md
    sed -n '1,260p' docs/verification/index.md
    git diff --name-only --

## Validation and Acceptance

Acceptance criteria:

1. `docs/verification/execplan_post_completion.md` exists and defines executable post-ExecPlan completion checks.
2. The document requires reading the PR document linked by the plan and deciding ready-for-review status.
3. If ready, the document requires setting GitHub PR to ready for review and moving the PR doc from `docs/prs/active/` to `docs/prs/completed/`.
4. `docs/verification/index.md` includes mapping for this new event.
5. Docs-only checks are run and recorded in this plan.

## Idempotence and Recovery

This is a documentation-only update. Re-running edits is safe. If necessary, revert or move PR docs back to active state when transitions were premature.

## Artifacts and Notes

Expected modified files:

    docs/verification/execplan_post_completion.md
    docs/verification/index.md

## Interfaces and Dependencies

No code interfaces or runtime behavior changes.

Revision note (2026-03-02, Codex): Initial plan created for adding post-ExecPlan completion verification event.
Revision note (2026-03-02, Codex): Updated progress/outcomes after creating post-completion event doc and wiring it into verification index.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.
