# Align Verification Filenames and Replace Determinism Section with Sandbox Approval Rule

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `VERIFICATION.md`, and `docs/verification/index.md`. `DESIGN.md` does not exist in the current tree, so no design-policy file was referenced.

## Purpose / Big Picture

After this change, verification event documents will use clear filenames without the `event_` prefix (for example `docs_only_changes.md`), and `VERIFICATION.md` will replace the existing command determinism section with explicit sandbox-approval operation guidance that minimizes repeated human approvals.

## Progress

- [x] (2026-03-02 01:27Z) Reviewed `PLANS.md` and current verification docs.
- [x] (2026-03-02 01:28Z) Renamed verification event files to remove the `event_` prefix and updated links in `docs/verification/index.md`.
- [x] (2026-03-02 01:28Z) Updated `VERIFICATION.md` filename examples and removed `Command source and determinism rule`.
- [x] (2026-03-02 01:28Z) Added sandbox-approval command-reuse rule in `VERIFICATION.md` with the efficiency exception and minimal-approval principle.
- [x] (2026-03-02 01:28Z) Validated file/link/reference consistency for verification docs.
- [x] (2026-03-02 01:28Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: `docs/verification/index.md` directly links current `event_*.md` names, so filename policy changes require synchronized link updates.
  Evidence: the current index links all three `event_*.md` files.

## Decision Log

- Decision: Apply filename policy change to both policy text and actual file names in one change.
  Rationale: Keeping old filenames while changing policy would create immediate policy drift.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

The verification filename policy now matches the requested style: event documents no longer use the `event_` prefix, and index links point to the new names. The old `Command source and determinism rule` section in `VERIFICATION.md` was removed and replaced with a sandbox-approval-focused rule that prioritizes command reuse for approval minimization while allowing new approvals for major efficiency gains.

This change is documentation-only and introduces no runtime behavior changes.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none (no `DESIGN.md` exists).
- Created/modified: none.
- Why unchanged: this task is verification-policy editing.

Architecture documents:

- Referenced: none.
- Created/modified: none.
- Why unchanged: no architecture-scope change is involved.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, and event documents under `docs/verification/`.
- Modified:
  - `VERIFICATION.md`
  - `docs/verification/index.md`
- Renamed:
  - `docs/verification/event_docs_only_changes.md` -> `docs/verification/docs_only_changes.md`
  - `docs/verification/event_cpu_behavior_changes.md` -> `docs/verification/cpu_behavior_changes.md`
  - `docs/verification/event_gpu_cuda_changes.md` -> `docs/verification/gpu_cuda_changes.md`

## Context and Orientation

Current verification policy examples and current files use an `event_` prefix that is now explicitly disallowed by the user request. Current `VERIFICATION.md` also contains a `Command source and determinism rule` section that must be removed and replaced with a new rule about sandbox approval minimization and command reuse, with an efficiency exception.

## Plan of Work

Rename the three existing verification event files to non-prefixed names and update `docs/verification/index.md` links. Then edit `VERIFICATION.md` so filename examples use non-prefixed naming, remove the entire `Command source and determinism rule` section, and insert a replacement section that captures the requested sandbox-approval behavior and minimization principle. Finally run consistency checks and complete the plan.

## Concrete Steps

Run from `.`:

    mv docs/verification/event_docs_only_changes.md docs/verification/docs_only_changes.md
    mv docs/verification/event_cpu_behavior_changes.md docs/verification/cpu_behavior_changes.md
    mv docs/verification/event_gpu_cuda_changes.md docs/verification/gpu_cuda_changes.md
    rg -n "event_.*\.md|Command source and determinism rule" VERIFICATION.md docs/verification/*.md
    find docs/verification -maxdepth 2 -type f | sort

## Validation and Acceptance

Acceptance criteria:

1. Verification event filenames no longer use the `event_` prefix.
2. `VERIFICATION.md` filename examples use non-prefixed examples.
3. `VERIFICATION.md` no longer contains `Command source and determinism rule`.
4. `VERIFICATION.md` contains the requested sandbox-approval command-reuse rule and efficiency exception.
5. `docs/verification/index.md` links resolve to renamed files.

## Idempotence and Recovery

This is a documentation-only rename/edit change. If needed, filenames can be reverted and links restored. Re-running text checks is safe.

## Artifacts and Notes

Expected changed files:

    VERIFICATION.md
    docs/verification/index.md
    docs/verification/docs_only_changes.md
    docs/verification/cpu_behavior_changes.md
    docs/verification/gpu_cuda_changes.md

## Interfaces and Dependencies

No production code interfaces are changed. This task changes documentation naming and verification policy language only.

Revision note (2026-03-02, Codex): Initial plan created for verification filename policy alignment and sandbox-approval rule replacement.
Revision note (2026-03-02, Codex): Updated progress and outcomes after completing filename renames, section replacement, and link validation.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.
