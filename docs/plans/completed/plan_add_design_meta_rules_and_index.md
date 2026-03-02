# Add Design Documentation Meta-Rules and Index

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are updated as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/harness_enginnering`
- Commit at start: `651c47c`
- PR tracking document: `docs/prs/active/pr_feat_harness_enginnering.md`

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `ARCHITECTURE.md`, `docs/verification/index.md`, `docs/verification/docs_only_changes.md`, and `docs/verification/execplan_post_completion.md`.

## Purpose / Big Picture

After this change, the repository will include explicit design-documentation governance through a new `DESIGN.md` meta-rules file and a mandatory design index at `docs/design/index.md`. This gives agents and new contributors a clear design-first reference that is separate from architecture implementation detail.

## Progress

- [x] (2026-03-02 03:53Z) Recorded start context (branch, commit, PR tracking path) and checked repository meta-doc tone references.
- [x] (2026-03-02 03:54Z) Added `DESIGN.md` with design-document roles, precedence, longevity, and location/index rules.
- [x] (2026-03-02 03:54Z) Created `docs/design/index.md` as the mandatory design-document index entry point.
- [x] (2026-03-02 03:54Z) Ran docs-only verification checks and recorded results (`git status --short`, `rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md DESIGN.md`).
- [ ] Move this plan to `docs/plans/completed/`, run post-ExecPlan validation event, and persist final state via commit/push.

## Surprises & Discoveries

- None so far.

## Decision Log

- Decision: Mirror section style and policy language from `ARCHITECTURE.md` for consistency across meta-rule documents.
  Rationale: The request requires the same tone and similar operational detail.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Completed:
- Added root-level design governance rules in `DESIGN.md` with the requested three roles and explicit design-first precedence.
- Added `docs/design/index.md` as the required design documentation entry point and index.

Pending:
- Final post-ExecPlan validation decision recording and lifecycle-persistence commit/push.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md` requirements from `AGENTS.md`.
- Created/modified:
  - `DESIGN.md` (new)
  - `docs/design/index.md` (new)

Architecture documents:
- Referenced: `ARCHITECTURE.md`.
- Created/modified: none.

Verification documents:
- Referenced: `docs/verification/docs_only_changes.md`, `docs/verification/execplan_post_completion.md`.
- Created/modified: none.

## Context and Orientation

`AGENTS.md` already requires reading `DESIGN.md` for long-lived non-obvious decisions, but the repository does not yet contain a design meta-rules document or a design index tree. This task establishes both in a long-lived format consistent with architecture and verification policy documents.

## Plan of Work

Create `DESIGN.md` as a root-level policy document defining the role, scope, and maintenance rules for design artifacts. Then create `docs/design/index.md` as the mandatory entry point for concrete design docs. Finally, run docs-only checks, finalize this plan, and move it to completed.

## Concrete Steps

Run from repository root (`.`):

    apply_patch ... (create DESIGN.md)
    mkdir -p docs/design
    apply_patch ... (create docs/design/index.md)
    git diff --name-only --
    rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md DESIGN.md

## Validation and Acceptance

Acceptance criteria:

1. `DESIGN.md` exists and uses architecture-style policy tone/detail.
2. `DESIGN.md` defines the three required design-document roles.
3. `DESIGN.md` explicitly states design/spec precedence over implementation and requires consistency with implementation over time.
4. `DESIGN.md` states that concrete design docs live under `docs/design/`.
5. `DESIGN.md` states `docs/design/index.md` is the design-doc index.
6. `docs/design/index.md` exists and functions as an entry-point index.

## Idempotence and Recovery

This is documentation-only work. Reapplying file creation and text updates is safe.

## Artifacts and Notes

Expected touched files:

    DESIGN.md
    docs/design/index.md
    docs/plans/active/plan_add_design_meta_rules_and_index.md

## Interfaces and Dependencies

No code interfaces or runtime behavior change.

Revision note (2026-03-02, Codex): Initial plan created for adding `DESIGN.md` and `docs/design/index.md`.
Revision note (2026-03-02, Codex): Updated progress and outcomes after creating `DESIGN.md` and `docs/design/index.md` and running docs-only checks.
