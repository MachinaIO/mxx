# Add Verification Event Doc for Pre-ExecPlan Creation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `VERIFICATION.md`, and `docs/verification/index.md`.

## Purpose / Big Picture

After this change, the repository will include an explicit verification event document for the stage before creating an ExecPlan. Agents will have concrete, executable steps to decide branch/PR alignment, enforce branch switching rules, and avoid starting new planning work on the wrong branch.

## Progress

- [x] (2026-03-02 02:03Z) Reviewed `PLANS.md` lifecycle and current `docs/verification` structure.
- [x] (2026-03-02 02:10Z) Created `docs/verification/execplan_pre_creation.md` with actionable branch/PR alignment validation steps.
- [x] (2026-03-02 02:11Z) Updated `docs/verification/index.md` event map to include this new pre-ExecPlan event.
- [x] (2026-03-02 02:12Z) Validation: ran docs-only verification checks and recorded outcomes (`git diff --name-only --`, `rg -n \"TODO|TBD|FIXME\" ...`).
- [x] (2026-03-02 02:12Z) Validation: re-checked index/file consistency after edits (`find docs/verification ...`, `rg -n \"execplan_pre_creation\" ...`).
- [x] (2026-03-02 02:13Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: No existing verification document currently covers the pre-ExecPlan branch/PR-alignment event.
  Evidence: `docs/verification/index.md` currently maps only docs-only, CPU behavior, and GPU/CUDA events.
- Observation: Workspace already contained unrelated non-doc changes, so raw docs-only check output included files outside this task scope.
  Evidence: `git diff --name-only --` included `.gitignore` in addition to verification-doc changes.

## Decision Log

- Decision: Add a dedicated document named `execplan_pre_creation.md` and map it explicitly in the verification index.
  Rationale: This event is distinct from docs-only/CPU/GPU execution checks and must be discoverable before plan creation.
  Date/Author: 2026-03-02 / Codex
- Decision: Include GitHub CLI (`gh`) commands as conditional checks with a local-evidence fallback.
  Rationale: PR-content alignment is required by the new event, but environments may not always have `gh` installed.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

The verification framework now includes a dedicated pre-ExecPlan event document. Agents can validate branch and PR alignment before creating plan files, enforce mandatory branch switching from `main`, and confirm commit state before switching when scope is not aligned with current work.

`docs/verification/index.md` now explicitly routes the pre-ExecPlan case to this new document, so the event is discoverable from the required entrypoint.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none.
- Created/modified: none.

Architecture documents:

- Referenced: none.
- Created/modified: none.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, and existing event documents.
- Modified:
  - `docs/verification/index.md`
- Created:
  - `docs/verification/execplan_pre_creation.md`

## Context and Orientation

`PLANS.md` lifecycle now requires a pre-ExecPlan validation step, but the current verification directory has no dedicated event document for branch/PR alignment before plan creation. This change fills that gap with a concrete runbook and adds it to the index.

## Plan of Work

Create a new verification event document that instructs agents to assess branch/PR alignment before creating an ExecPlan, enforce commit checks before branch switching, and always branch off when currently on `main`. Then update the verification index to point to this event and explain when to use it.

## Concrete Steps

Run from `.`:

    cat > docs/verification/execplan_pre_creation.md << '...'
    apply_patch << 'PATCH'
    ...
    PATCH
    git diff --name-only --
    rg -n "ExecPlan|pre-ExecPlan|execplan_pre_creation" docs/verification -S

## Validation and Acceptance

Acceptance criteria:

1. New file `docs/verification/execplan_pre_creation.md` exists with executable actions.
2. Document covers branch/PR consistency check, commit confirmation before switching, and mandatory branch switch from `main`.
3. `docs/verification/index.md` maps the pre-ExecPlan event to the new document.
4. Docs-only checks are run and recorded in this plan.

## Idempotence and Recovery

This is documentation-only; re-running edits is safe. Rollback is straightforward by reverting changed markdown files.

## Artifacts and Notes

Expected modified files:

    docs/verification/execplan_pre_creation.md
    docs/verification/index.md

## Interfaces and Dependencies

No code interface or runtime behavior changes.

Revision note (2026-03-02, Codex): Initial plan created to add pre-ExecPlan verification event documentation.
Revision note (2026-03-02, Codex): Updated progress and outcomes after creating the new event doc, updating the index, and running documentation validation checks.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.
